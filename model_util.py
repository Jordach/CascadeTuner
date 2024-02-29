# Single file import version of Stable Cascade modules

import torchvision
from torch import nn
import torch
import numpy as np
import math
import kornia
import cv2
from core_util import load_or_fail
import warnings

from xformers_util import FlashAttention2D

# Common
class Linear(torch.nn.Linear):
    def reset_parameters(self):
        return None

class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        return None
        
class Attention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        x = self.attn(x, kv, kv, need_weights=False)[0]
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class GlobalResponseNorm(nn.Module):
    "from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):  # , num_heads=4, expansion=2):
        super().__init__()
        self.depthwise = Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        #         self.depthwise = SAMBlock(c, num_heads, expansion)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            Linear(c + c_skip, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            Linear(c * 4, c)
        )

    def forward(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res

class AttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0, flash_attention=False):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        if flash_attention:
            self.attention = FlashAttention2D(c, nhead, dropout) 
        else:
            self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            Linear(c_cond, c)
        )

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            Linear(c, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            Linear(c * 4, c)
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=['sca']):
        super().__init__()
        self.mapper = Linear(c_timestep, c * 2)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", Linear(c_timestep, c * 2))

    def forward(self, x, t):
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b

# EfficientNet
class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT').features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),  # then normalize them to have mean 0 and std 1
        )

    def forward(self, x):
        return self.mapper(self.backbone(x))

# ControlNet
from insightface.app.common import Face
from cnet_modules.pidinet import PidiNetDetector
from cnet_modules.inpainting.saliency_model import MicroResNet
from cnet_modules.face_id.arcface import FaceDetector, ArcFaceRecognizer

class CNetResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blocks = nn.Sequential(
            LayerNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            LayerNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.blocks(x)

class ControlNet(nn.Module):
    def __init__(self, c_in=3, c_proj=2048, proj_blocks=None, bottleneck_mode=None):
        super().__init__()
        if bottleneck_mode is None:
            bottleneck_mode = 'effnet'
        self.proj_blocks = proj_blocks
        if bottleneck_mode == 'effnet':
            embd_channels = 1280
            self.backbone = torchvision.models.efficientnet_v2_s(weights='DEFAULT').features.eval()
            if c_in != 3:
                in_weights = self.backbone[0][0].weight.data
                self.backbone[0][0] = nn.Conv2d(c_in, 24, kernel_size=3, stride=2, bias=False)
                if c_in > 3:
                    nn.init.constant_(self.backbone[0][0].weight, 0)
                    self.backbone[0][0].weight.data[:, :3] = in_weights[:, :3].clone()
                else:
                    self.backbone[0][0].weight.data = in_weights[:, :c_in].clone()
        elif bottleneck_mode == 'simple':
            embd_channels = c_in
            self.backbone = nn.Sequential(
                nn.Conv2d(embd_channels, embd_channels * 4, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embd_channels * 4, embd_channels, kernel_size=3, padding=1),
            )
        elif bottleneck_mode == 'large':
            self.backbone = nn.Sequential(
                nn.Conv2d(c_in, 4096 * 4, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(4096 * 4, 1024, kernel_size=1),
                *[CNetResBlock(1024) for _ in range(8)],
                nn.Conv2d(1024, 1280, kernel_size=1),
            )
            embd_channels = 1280
        else:
            raise ValueError(f'Unknown bottleneck mode: {bottleneck_mode}')
        self.projections = nn.ModuleList()
        for _ in range(len(proj_blocks)):
            self.projections.append(nn.Sequential(
                nn.Conv2d(embd_channels, embd_channels, kernel_size=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embd_channels, c_proj, kernel_size=1, bias=False),
            ))
            nn.init.constant_(self.projections[-1][-1].weight, 0)  # zero output projection

    def forward(self, x):
        x = self.backbone(x)
        proj_outputs = [None for _ in range(max(self.proj_blocks) + 1)]
        for i, idx in enumerate(self.proj_blocks):
            proj_outputs[idx] = self.projections[i](x)
        return proj_outputs


class ControlNetDeliverer():
    def __init__(self, controlnet_projections):
        self.controlnet_projections = controlnet_projections
        self.restart()

    def restart(self):
        self.idx = 0
        return self

    def __call__(self):
        if self.idx < len(self.controlnet_projections):
            output = self.controlnet_projections[self.idx]
        else:
            output = None
        self.idx += 1
        return output


# CONTROLNET FILTERS ----------------------------------------------------

class BaseFilter():
    def __init__(self, device):
        self.device = device

    def num_channels(self):
        return 3

    def __call__(self, x):
        return x


class CannyFilter(BaseFilter):
    def __init__(self, device, resize=224):
        super().__init__(device)
        self.resize = resize

    def num_channels(self):
        return 1

    def __call__(self, x):
        orig_size = x.shape[-2:]
        if self.resize is not None:
            x = nn.functional.interpolate(x, size=(self.resize, self.resize), mode='bilinear')
        edges = [cv2.Canny(x[i].mul(255).permute(1, 2, 0).cpu().numpy().astype(np.uint8), 100, 200) for i in range(len(x))]
        edges = torch.stack([torch.tensor(e).div(255).unsqueeze(0) for e in edges], dim=0)
        if self.resize is not None:
            edges = nn.functional.interpolate(edges, size=orig_size, mode='bilinear')
        return edges


class QRFilter(BaseFilter):
    def __init__(self, device, resize=224, blobify=True, dilation_kernels=[3, 5, 7], blur_kernels=[15]):
        super().__init__(device)
        self.resize = resize
        self.blobify = blobify
        self.dilation_kernels = dilation_kernels
        self.blur_kernels = blur_kernels

    def num_channels(self):
        return 1

    def __call__(self, x):
        x = x.to(self.device)
        orig_size = x.shape[-2:]
        if self.resize is not None:
            x = nn.functional.interpolate(x, size=(self.resize, self.resize), mode='bilinear')

        x = kornia.color.rgb_to_hsv(x)[:, -1:]
        # blobify
        if self.blobify:
            d_kernel = np.random.choice(self.dilation_kernels)
            d_blur = np.random.choice(self.blur_kernels)
            if d_blur > 0:
                x = torchvision.transforms.GaussianBlur(d_blur)(x)
            if d_kernel > 0:
                blob_mask = ((torch.linspace(-0.5, 0.5, d_kernel).pow(2)[None] + torch.linspace(-0.5, 0.5,
                                                                                                d_kernel).pow(2)[:,
                                                                                 None]) < 0.3).float().to(self.device)
                x = kornia.morphology.dilation(x, blob_mask)
                x = kornia.morphology.erosion(x, blob_mask)
        # mask
        vmax, vmin = x.amax(dim=[2, 3], keepdim=True)[0], x.amin(dim=[2, 3], keepdim=True)[0]
        th = (vmax - vmin) * 0.33
        high_brightness, low_brightness = (x > (vmax - th)).float(), (x < (vmin + th)).float()
        mask = (torch.ones_like(x) - low_brightness + high_brightness) * 0.5

        if self.resize is not None:
            mask = nn.functional.interpolate(mask, size=orig_size, mode='bilinear')
        return mask.cpu()


class PidiFilter(BaseFilter):
    def __init__(self, device, resize=224, dilation_kernels=[0, 3, 5, 7, 9], binarize=True):
        super().__init__(device)
        self.resize = resize
        self.model = PidiNetDetector(device)
        self.dilation_kernels = dilation_kernels
        self.binarize = binarize

    def num_channels(self):
        return 1

    def __call__(self, x):
        x = x.to(self.device)
        orig_size = x.shape[-2:]
        if self.resize is not None:
            x = nn.functional.interpolate(x, size=(self.resize, self.resize), mode='bilinear')

        x = self.model(x)
        d_kernel = np.random.choice(self.dilation_kernels)
        if d_kernel > 0:
            blob_mask = ((torch.linspace(-0.5, 0.5, d_kernel).pow(2)[None] + torch.linspace(-0.5, 0.5, d_kernel).pow(2)[
                                                                             :, None]) < 0.3).float().to(self.device)
            x = kornia.morphology.dilation(x, blob_mask)
        if self.binarize:
            th = np.random.uniform(0.05, 0.7)
            x = (x > th).float()

        if self.resize is not None:
            x = nn.functional.interpolate(x, size=orig_size, mode='bilinear')
        return x.cpu()


class SRFilter(BaseFilter):
    def __init__(self, device, scale_factor=1 / 4):
        super().__init__(device)
        self.scale_factor = scale_factor

    def num_channels(self):
        return 3

    def __call__(self, x):
        x = torch.nn.functional.interpolate(x.clone(), scale_factor=self.scale_factor, mode="nearest")
        return torch.nn.functional.interpolate(x, scale_factor=1 / self.scale_factor, mode="nearest")


class SREffnetFilter(BaseFilter):
    def __init__(self, device, scale_factor=1/2):
        super().__init__(device)
        self.scale_factor = scale_factor

        self.effnet_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

        self.effnet = EfficientNetEncoder().to(self.device)
        effnet_checkpoint = load_or_fail("models/effnet_encoder.safetensors")
        self.effnet.load_state_dict(effnet_checkpoint)
        self.effnet.eval().requires_grad_(False)

    def num_channels(self):
        return 16

    def __call__(self, x):
        x = torch.nn.functional.interpolate(x.clone(), scale_factor=self.scale_factor, mode="nearest")
        with torch.no_grad():
            effnet_embedding = self.effnet(self.effnet_preprocess(x.to(self.device))).cpu()
        effnet_embedding = torch.nn.functional.interpolate(effnet_embedding, scale_factor=1/self.scale_factor, mode="nearest")
        upscaled_image = torch.nn.functional.interpolate(x, scale_factor=1/self.scale_factor, mode="nearest")
        return effnet_embedding, upscaled_image


class InpaintFilter(BaseFilter):
    def __init__(self, device, thresold=[0.04, 0.4], p_outpaint=0.4):
        super().__init__(device)
        self.saliency_model = MicroResNet().eval().requires_grad_(False).to(device)
        self.saliency_model.load_state_dict(load_or_fail("modules/cnet_modules/inpainting/saliency_model.pt"))
        self.thresold = thresold
        self.p_outpaint = p_outpaint

    def num_channels(self):
        return 4

    def __call__(self, x, mask=None, threshold=None, outpaint=None):
        x = x.to(self.device)
        resized_x = torchvision.transforms.functional.resize(x, 240, antialias=True)
        if threshold is None:
            threshold = np.random.uniform(self.thresold[0], self.thresold[1])
        if mask is None:
            saliency_map = self.saliency_model(resized_x) > threshold
            if outpaint is None:
                if np.random.rand() < self.p_outpaint:
                    saliency_map = ~saliency_map
            else:
                if outpaint:
                    saliency_map = ~saliency_map
            interpolated_saliency_map = torch.nn.functional.interpolate(saliency_map.float(), size=x.shape[2:], mode="nearest")
            saliency_map = torchvision.transforms.functional.gaussian_blur(interpolated_saliency_map, 141) > 0.5
            inpainted_images = torch.where(saliency_map, torch.ones_like(x), x)
            mask = torch.nn.functional.interpolate(saliency_map.float(), size=inpainted_images.shape[2:], mode="nearest")
        else:
            mask = mask.to(self.device)
            inpainted_images = torch.where(mask, torch.ones_like(x), x)
        c_inpaint = torch.cat([inpainted_images, mask], dim=1)
        return c_inpaint.cpu()


# IDENTITY
class IdentityFilter(BaseFilter):
    def __init__(self, device, max_faces=4, p_drop=0.05, p_full=0.3):
        detector_path = 'modules/cnet_modules/face_id/models/buffalo_l/det_10g.onnx'
        recognizer_path = 'modules/cnet_modules/face_id/models/buffalo_l/w600k_r50.onnx'

        super().__init__(device)
        self.max_faces = max_faces
        self.p_drop = p_drop
        self.p_full = p_full

        self.detector = FaceDetector(detector_path, device=device)
        self.recognizer = ArcFaceRecognizer(recognizer_path, device=device)

        self.id_colors = torch.tensor([
            [1.0, 0.0, 0.0],  # RED
            [0.0, 1.0, 0.0],  # GREEN
            [0.0, 0.0, 1.0],  # BLUE
            [1.0, 0.0, 1.0],  # PURPLE
            [0.0, 1.0, 1.0],  # CYAN
            [1.0, 1.0, 0.0],  # YELLOW
            [0.5, 0.0, 0.0],  # DARK RED
            [0.0, 0.5, 0.0],  # DARK GREEN
            [0.0, 0.0, 0.5],  # DARK BLUE
            [0.5, 0.0, 0.5],  # DARK PURPLE
            [0.0, 0.5, 0.5],  # DARK CYAN
            [0.5, 0.5, 0.0],  # DARK YELLOW
        ])

    def num_channels(self):
        return 512

    def get_faces(self, image):
        npimg = image.permute(1, 2, 0).mul(255).to(device="cpu", dtype=torch.uint8).cpu().numpy()
        bgr = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
        bboxes, kpss = self.detector.detect(bgr, max_num=self.max_faces)
        N = len(bboxes)
        ids = torch.zeros((N, 512), dtype=torch.float32)
        for i in range(N):
            face = Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
            ids[i, :] = self.recognizer.get(bgr, face)
        tbboxes = torch.tensor(bboxes[:, :4], dtype=torch.int)

        ids = ids / torch.linalg.norm(ids, dim=1, keepdim=True)
        return tbboxes, ids  # returns bounding boxes (N x 4) and ID vectors (N x 512)

    def __call__(self, x):
        visual_aid = x.clone().cpu()
        face_mtx = torch.zeros(x.size(0), 512, x.size(-2) // 32, x.size(-1) // 32)

        for i in range(x.size(0)):
            bounding_boxes, ids = self.get_faces(x[i])
            for j in range(bounding_boxes.size(0)):
                if np.random.rand() > self.p_drop:
                    sx, sy, ex, ey = (bounding_boxes[j] / 32).clamp(min=0).round().int().tolist()
                    ex, ey = max(ex, sx + 1), max(ey, sy + 1)
                    if bounding_boxes.size(0) == 1 and np.random.rand() < self.p_full:
                        sx, sy, ex, ey = 0, 0, x.size(-1) // 32, x.size(-2) // 32
                    face_mtx[i, :, sy:ey, sx:ex] = ids[j:j + 1, :, None, None]
                    visual_aid[i, :, int(sy * 32):int(ey * 32), int(sx * 32):int(ex * 32)] += self.id_colors[j % 13, :,
                                                                                              None, None]
                    visual_aid[i, :, int(sy * 32):int(ey * 32), int(sx * 32):int(ex * 32)] *= 0.5

        return face_mtx.to(x.device), visual_aid.to(x.device)

# Stage C Related
class UpDownBlock2d(nn.Module):
    def __init__(self, c_in, c_out, mode, enabled=True):
        super().__init__()
        assert mode in ['up', 'down']
        interpolation = nn.Upsample(scale_factor=2 if mode == 'up' else 0.5, mode='bilinear',
                                    align_corners=True) if enabled else nn.Identity()
        mapping = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == 'up' else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x.float())
        return x

class StageC(nn.Module):
    def __init__(self, c_in=16, c_out=16, c_r=64, patch_size=1, c_cond=2048, c_hidden=[2048, 2048], nhead=[32, 32],
                 blocks=[[8, 24], [24, 8]], block_repeat=[[1, 1], [1, 1]], level_config=['CTA', 'CTA'],
                 c_clip_text=1280, c_clip_text_pooled=1280, c_clip_img=768, c_clip_seq=4, kernel_size=3,
                 dropout=[0.1, 0.1], self_attn=True, t_conds=['sca', 'crp'], switch_level=[False], settings=None, flash_attention=False):
        super().__init__()
        flash_attention = flash_attention
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)

        # CONDITIONING
        self.clip_txt_mapper = nn.Linear(c_clip_text, c_cond)
        self.clip_txt_pooled_mapper = nn.Linear(c_clip_text_pooled, c_cond * c_clip_seq)
        self.clip_img_mapper = nn.Linear(c_clip_img, c_cond * c_clip_seq)
        self.clip_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(c_in * (patch_size ** 2), c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6)
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout, flash_attention=flash_attention)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r, conds=t_conds)
            else:
                raise Exception(f'Block type {block_type} not supported')

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                    UpDownBlock2d(c_hidden[i - 1], c_hidden[i], mode='down', enabled=switch_level[i - 1])
                ))
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                    UpDownBlock2d(c_hidden[i], c_hidden[i - 1], mode='up', enabled=switch_level[i - 1])
                ))
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i],
                                      self_attn=self_attn[i])
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], c_out * (patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        nn.init.normal_(self.clip_txt_mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.clip_txt_pooled_mapper.weight, std=0.02)  # conditionings
        nn.init.normal_(self.clip_img_mapper.weight, std=0.02)  # conditionings
        torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
        nn.init.constant_(self.clf[1].weight, 0)  # outputs

        # blocks
        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
                elif isinstance(block, TimestepBlock):
                    for layer in block.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.constant_(layer.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def gen_c_embeddings(self, clip_txt, clip_txt_pooled, clip_img):
        clip_txt = self.clip_txt_mapper(clip_txt)
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pool = clip_txt_pooled.unsqueeze(1)
        if len(clip_img.shape) == 2:
            clip_img = clip_img.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.c_clip_seq, -1)
        clip_img = self.clip_img_mapper(clip_img).view(clip_img.size(0), clip_img.size(1) * self.c_clip_seq, -1)
        clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip, cnet=None):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  ResBlock)):
                        if cnet is not None:
                            next_cnet = cnet()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode='bilinear',
                                                                  align_corners=True)
                        x = block(x)
                    elif isinstance(block, AttnBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  AttnBlock)):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  TimestepBlock)):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip, cnet=None):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  ResBlock)):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x.float(), skip.shape[-2:], mode='bilinear',
                                                                align_corners=True)
                        if cnet is not None:
                            next_cnet = cnet()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode='bilinear',
                                                                  align_corners=True)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  AttnBlock)):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  TimestepBlock)):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(self, x, r, clip_text, clip_text_pooled, clip_img, cnet=None, **kwargs):
        # Process the conditioning embeddings

        # One of these gradient checkpoints returns a None gradient; but other wise function fine
        # but it's being silenced here by force due to checkpointing working, but all tensors are
        # not using a grad - this certainly can't cause any problems later on, right?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_embed = torch.utils.checkpoint.checkpoint(self.gen_r_embedding, r, use_reentrant=False)
            for c in self.t_conds:
                t_cond = kwargs.get(c, torch.zeros_like(r))
                r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)
            
            clip = torch.utils.checkpoint.checkpoint(self.gen_c_embeddings, clip_text, clip_text_pooled, clip_img, use_reentrant=True)

            # Model Blocks
            x = torch.utils.checkpoint.checkpoint(self.embedding, x, use_reentrant=False)
            if cnet is not None:
                cnet = torch.utils.checkpoint.checkpoint(ControlNetDeliverer, cnet, use_reentrant=True)

            level_outputs = torch.utils.checkpoint.checkpoint(self._down_encode, x, r_embed, clip, cnet, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._up_decode, level_outputs, r_embed, clip, cnet, use_reentrant=True)
            return self.clf(x)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone().to(self_params.device) * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(self_buffers.device) * (1 - beta)