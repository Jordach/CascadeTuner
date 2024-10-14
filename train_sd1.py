import argparse
import os
import yaml
import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torchvision
import transformers
import itertools
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from bucketeer import StrictBucketeer
from core_util import minSNR_weighting_loss
from dataset_util import BucketWalker
from tokeniser_util import get_text_embeds, tokenize_respecting_boundaries
from optim_util import get_optimizer, step_adafactor
from sd1_util import SD1CachedLatents, vae_encode, vae_preprocess, save_sd1_pipeline
from zstd_util import save_torch_zstd
from contextlib import nullcontext
from tag_weighting_util import load_from_json_storage, save_to_json_storage, add_tags_from_batch, get_loss_multiplier_for_batch

logger = get_logger(__name__)

def process_latent_caches(settings, accelerator, original_latent_caches, latent_cache, print_info=False):
    if settings["create_latent_cache"] or settings["use_latent_cache"]:
        # Delete all old entries
        latent_cache.reset_batch_list()

        for batch in original_latent_caches:
            latent_cache.add_latent_batch(batch, False)

        # Handle duplicates for unconditional training
        if settings["dropout"] > 0:
            if len(original_latent_caches) > 100:
                total_batches = int((len(original_latent_caches)-1) * settings["dropout"])

                dropouts = random.sample(original_latent_caches, total_batches)
                for batch in dropouts:
                    latent_cache.add_latent_batch(batch, True)

                if print_info:
                    accelerator.print(f"Original Cached Step Count: {len(original_latent_caches)}")
                    accelerator.print(f"Duplicated {len(dropouts)} caches for caption dropout.")
                    accelerator.print(f"Total Cached Step Count: {len(latent_cache)}")
        
        dataloader = torch.utils.data.DataLoader(
            latent_cache, batch_size=1, collate_fn=lambda x: x, shuffle=False, 
        )
        return dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Custom training script for SD1.")
    parser.add_argument("--yaml", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--cache_only", default=False, action="store_true", help="Whether to quit after latent caching.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    settings = {}

    settings["dataset_cache"] = "__no_path__"
    settings["tag_shuffling"] = False
    settings["unet_optim"] = "_____no_path.pt"
    settings["text_enc_optim"] = "_____no_path.pt"
    settings["tag_weighting_path"] = "__no_path__" # Requires a valid path to function
    settings["tag_weighting_used"] = False
    settings["tag_weighting_count_low"] = 10 # Anything under this count will be treated as multi_max
    settings["tag_weighting_count_high"] = 4000 # Anything over this count will be treated as multi_min
    settings["tag_weighting_multi_min"] = 1
    settings["tag_weighting_multi_max"] = 4
    settings["tag_weighting_exponent"] = 7
    settings["tag_dropout_total_min"] = 25
    settings["tag_dropout_percentage"] = 0.3

    # Load settings from YAML config
    with open(args.yaml, "r") as f:
        config = yaml.safe_load(f)
        settings = settings | config

    main_dtype = getattr(torch, settings["dtype"]) if "dtype" in settings else torch.float32
    if settings["dtype"] == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(
       gradient_accumulation_steps=settings["grad_accum_steps"],
       log_with="tensorboard",
       project_dir=f"{settings['checkpoint_path']}"
    )

    tag_weighting_dict = {}
    if settings["tag_weighting_path"] != "__no_path__":
        settings["tag_weighting_used"] = True
        # Only load it if we need to
        if os.path.exists(settings["tag_weighting_path"]) and settings["use_latent_cache"]:
            load_from_json_storage(settings["tag_weighting_path"], tag_weighting_dict)
            accelerator.print("Will load tag counts from cache for loss weighting.")
        else:
            accelerator.print("Will collect tag counts for loss weighting.")

    # Ensure output directory exists:
    if accelerator.is_main_process:
        if not os.path.exists(settings["checkpoint_path"]):
            os.makedirs(settings["checkpoint_path"])
    
    if settings["use_pytorch_cross_attention"]:
        accelerator.print("Activating efficient cross attentions.")
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Set up seed
    set_seed(settings["seed"])

    # Load Tokeniser
    tokenizer = CLIPTokenizer.from_pretrained(settings["model_name"], subfolder="tokenizer")

    pre_dataset = []

    # Setup Dataloader Phase:
    if not settings["use_latent_cache"]:
        accelerator.print("Loading Dataset[s].")

        did_load_from_cache = False
        if settings["dataset_cache"] != "__no_path__" and os.path.exists(settings["dataset_cache"]):
            accelerator.print("Loading dataset cache from disk.")
            pre_dataset = torch.load(settings["dataset_cache"])
            did_load_from_cache = True
        else:
            if settings["dataset_cache"] != "__no_path__":
                accelerator.print("Will create a dataset cache after searching folders.")
            pre_dataset = BucketWalker(
               reject_aspects=settings["reject_aspects"],
               tokenizer=tokenizer
            )

        if "local_dataset_path" in settings and not did_load_from_cache:
            if type(settings["local_dataset_path"]) is list:
                for dir in settings["local_dataset_path"]:
                    pre_dataset.scan_folder(dir)
            elif type(settings["local_dataset_path"]) is str:
                pre_dataset.scan_folder(settings["local_dataset_path"])
            else:
                raise ValueError("'local_dataset_path' must either be a string, or list of strings containing paths.")

        # Save state of the dataset class if it wasn't instanced from 
        if settings["dataset_cache"] != "__no_path__" and not did_load_from_cache:
            torch.save(pre_dataset, f"{settings['dataset_cache']}")
            accelerator.print("Saved dataset cache.")

        pre_dataset.bucketize(settings["batch_size"])
        accelerator.print(f"Total Invalid Files:  {pre_dataset.get_rejects()}")
        if "multi_aspect_ratio" in settings:
            one_over_ratios = []
            accelerator.print("Notice: Using custom supplied bucketing ratios")
            for bucket in settings["multi_aspect_ratio"]:
                if bucket > 1:
                    one_over_ratios.append(1/bucket)
            settings["multi_aspect_ratio"].extend(one_over_ratios)
            
            custom_ratios = [f"{r:.2f}" for r in settings["multi_aspect_ratio"]]
            accelerator.print(f"Custom Ratios:")
            accelerator.print(f"{', '.join(r.strip() for r in custom_ratios)}")
        else:
            accelerator.print("Notice: Using automated bucketing ratios")
            settings["multi_aspect_ratio"] = pre_dataset.get_buckets()

        # accelerator.print("Bucketing Info:")

    def pre_collate(batch):
        # Do NOT load images - save that for the second dataloader pass
        images = [data["images"] for data in batch]
        caption = [data["caption"] for data in batch]

        if settings["tag_weighting_used"]:
            add_tags_from_batch(tag_weighting_dict, caption)
        
        aspect = batch[0]["aspects"]
        return {"images": images, "caption": caption, "dropout": False, "aspect": aspect}

    pre_dataloader = DataLoader(
        pre_dataset, batch_size=settings["batch_size"], shuffle=False, collate_fn=pre_collate, pin_memory=False,
    )

    # Create second dataset so all images are batched if we're either caching latents or loading direct from disk
    dataset = []

    # Skip initial dataloading pass if we're using a latent cache
    if not settings["use_latent_cache"]:
        for batch in tqdm(pre_dataloader, desc="Dataloader Warmup"):
            dataset.append(batch)

    if settings["tag_weighting_used"] and settings["tag_weighting_path"] != "__no_path__":
        save_to_json_storage(settings["tag_weighting_path"], tag_weighting_dict)

    auto_bucketer = StrictBucketeer(
        density=settings["image_size"]**2,
        factor=8,
        aspect_ratios=settings["multi_aspect_ratio"],
        transforms=vae_preprocess,
    )

    def collate(batch):
        images = []
        # The reason for not unrolling the images in the prior dataloader was so we can load them only when training,
        # rather than storing all transformed images in memory!
        img = batch[0]["images"]
        ratio = 1
        aspect = batch[0]["aspect"]
        for i in range(0, len(batch[0]["images"])):
            _img, _ratio = auto_bucketer(img[i], ratio=aspect)
            ratio = _ratio
            images.append(_img)
        images = torch.stack(images)
        images = images.to(memory_format=torch.contiguous_format)
        images = images.to(accelerator.device)
        captions = batch[0]["caption"]
        # Tokenisation should be done during latent caching/runtime VAE encoding process to avoid storing lots of tokens in system memory.
        tokens, att_mask = tokenize_respecting_boundaries(tokenizer, captions)
        dropout = batch[0]["dropout"]
        return {"images": images, "tokens": tokens, "att_mask": att_mask, "captions": captions, "dropout": dropout, "aspect": aspect, "bucket": ratio}

    # Shuffle the dataset and initialise the dataloader if we're not latent caching
    set_seed(settings["seed"])
    if not settings["create_latent_cache"]:
        random.shuffle(dataset)

    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collate, shuffle=False, pin_memory=False
    )

    # Uncomment this to figure out what's wrong with the dataloader:
    # for batch in tqdm(dataloader):
    # 	pass
    # return
    
    # Load the VAE
    vae = AutoencoderKL.from_pretrained(settings["model_name"], subfolder="vae")
    vae.to(accelerator.device, dtype=torch.float16)
    vae.requires_grad_(False)
    vae.enable_slicing()
    if is_xformers_available():
        try:
            vae.enable_xformers_memory_efficient_attention()
        except Exception as e:
            raise Exception("Could not enable memory efficient attention. Make sure xformers is installed")

    latent_cache = SD1CachedLatents(accelerator=accelerator, settings=settings, tokenizer=tokenizer, tag_shuffle=settings["tag_shuffling"])
    # Create a latent cache if we're not going to load an existing one:
    original_latent_caches = []
    if settings["create_latent_cache"] and not settings["use_latent_cache"]:
        # Ensure cache output directory exists:
        if accelerator.is_main_process:
            os.makedirs(settings["latent_cache_location"], exist_ok=True)

        step = 0
        latent_caching_bar = tqdm(dataloader, desc="Latent Caching")
        for batch in latent_caching_bar:
            latent_caching_bar.set_postfix({
                "aspect": batch["aspect"],
                "bucket": batch["bucket"]
            })

            with torch.no_grad():
                batch["vae_encoded"] = vae_encode(batch["images"], vae)
            del batch["images"]

            file_name = f"latent_cache_{settings['experiment_id']}_{step}.zpt"
            cache_path = os.path.join(settings["latent_cache_location"], file_name)
            save_torch_zstd(batch, cache_path)
            original_latent_caches.append(cache_path)
            step += 1
        if args.cache_only:
            return 0
        # Better method to handle latent caching
        dataloader = process_latent_caches(settings, accelerator, original_latent_caches, latent_cache, print_info=True)
    elif settings["use_latent_cache"]:
        # Load all latent caches from disk. Note that batch size is ignored here and can theoretically be mixed.
        if not os.path.exists(settings["latent_cache_location"]):
            raise Exception("Latent Cache folder does not exist. Please run latent caching first.")

        if len(os.listdir(settings["latent_cache_location"])) == 0:
            raise Exception("No latent caches to load. Please run latent caching first.")

        accelerator.print("Loading media from the Latent Cache.")
        for cache in os.listdir(settings["latent_cache_location"]):
            latent_path = os.path.join(settings["latent_cache_location"], cache)
            original_latent_caches.append(latent_path)

        # Better method to handle latent caching
        dataloader = process_latent_caches(settings, accelerator, original_latent_caches, latent_cache, print_info=True)

        # We don't need the VAE anymore - so just delete it from memory
        del vae
        vae = None

    accelerator.print("Now loading core model files.")
    noise_scheduler = DDPMScheduler.from_pretrained(settings["model_name"], subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(settings["model_name"], subfolder="unet", main_dtype=main_dtype)
    text_model = CLIPTextModel.from_pretrained(settings["model_name"], subfolder="text_encoder")
    text_model = text_model.to(dtype=main_dtype)
    if settings["enable_gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
        if settings["train_text_encoder"]:
            text_model.gradient_checkpointing_enable()

    # Apply xformers to unet:
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            raise Exception("Could not enable memory efficient attention. Make sure xformers is installed")

    accelerator.print("Loading optimizer[s].")
    # Unet optimizer and LR scheduler:
    unet_optimizer, unet_optimizer_kwargs = get_optimizer(settings["optimizer_type"], settings)
    unet_optimizer = unet_optimizer((unet.parameters()), lr=settings["lr"], **unet_optimizer_kwargs)
    if os.path.exists(settings["unet_optim"]):
        unet_optimizer.load_state_dict(torch.load(settings["unet_optim"]))
    elif settings["unet_optim"] == "_____no_path.pt":
        pass
    else:
        raise ValueError("Cannot load Unet optimizer from disk, does it exist?")

    if settings["optimizer_type"].lower() == "adafactorstoch":
        unet_optimizer.step = step_adafactor.__get__(unet_optimizer, transformers.optimization.Adafactor)
    
    unet_scheduler = get_scheduler(
        settings["lr_scheduler"],
        optimizer=unet_optimizer,
        num_warmup_steps=settings["warmup_updates"],
        num_training_steps=len(dataloader) * settings["num_epochs"]
    )

    # Text Encoder optimizer and LR scheduler:
    text_optimizer = ""
    text_scheduler = ""

    if settings["train_text_encoder"]:
        text_optimizer, text_optimizer_kwargs = get_optimizer(settings["text_optimizer_type"], settings)

        text_optimizer = text_optimizer((text_model.parameters()), lr=settings["text_lr"], **text_optimizer_kwargs)
        if os.path.exists(settings["text_enc_optim"]) and settings["text_enc_optim"] != "_____no_path.pt":
            text_optimizer.load_state_dict(torch.load(settings["text_enc_optim"]))
        elif settings["text_enc_optim"] == "_____no_path.pt":
            pass
        else:
            raise ValueError("Cannot load Text Encoder optimizer state from disk, does it exist?")
    
        text_scheduler = get_scheduler(
            settings["text_lr_scheduler"],
            optimizer=text_optimizer,
            num_warmup_steps=settings["warmup_updates"],
            num_training_steps=len(dataloader) * settings["num_epochs"]
        )

    noise_scheduler = accelerator.prepare(noise_scheduler)
    unet, text_model = accelerator.prepare(unet, text_model)
    unet_optimizer, unet_scheduler = accelerator.prepare(unet_optimizer, unet_scheduler)
    if settings["train_text_encoder"]:
        text_optimizer, text_scheduler = accelerator.prepare(text_optimizer, text_scheduler)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("training")
    
    accelerator.wait_for_everyone()
    dataloader = accelerator.prepare(dataloader)
    
    # Training Loop:
    steps_bar = tqdm(range(len(dataloader)), desc="Steps to Epoch", disable=not accelerator.is_local_main_process)
    epoch_bar = tqdm(range(settings["num_epochs"]), desc="Epochs", disable=not accelerator.is_local_main_process)
    total_steps = 0
    total_epochs = 0

    torch.cuda.empty_cache()

    # Handle text encoder context
    text_encoder_context = nullcontext() if settings["train_text_encoder"] else torch.no_grad()
    last_grad_norm = 0
    # Set whether models are in training mode
    unet.train()
    if settings["train_text_encoder"]:
        text_model.train()
    else:
        text_model.eval()

    if "min_snr_gamma" in settings:
        accelerator.print("Using minSNR timestep weighting.")
    
    for e in epoch_bar:
        current_step = 0
        # Reset the unconditional batches here per epoch after the first one;
        # since it's already had caches added - preserving the pRNG is important here
        if total_epochs > 0:
            accelerator.wait_for_everyone()
            dataloader = process_latent_caches(settings, accelerator, original_latent_caches, latent_cache, print_info=False)
            dataloader = accelerator.prepare(dataloader)
        steps_bar.reset(total=len(dataloader))

        for step, batch in enumerate(dataloader):
            loss_weighting_mult = 1
            with accelerator.accumulate(unet, text_model) if settings["train_text_encoder"] else accelerator.accumulate(unet):
                with accelerator.autocast():
                    captions = batch[0]["tokens"]
                    attn_mask = batch[0]["att_mask"]
                    latents = batch[0]["vae_encoded"]
                    dropout = batch[0]["dropout"]
                    batch_size = len(batch[0]["captions"])

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    with text_encoder_context:
                        text_embeds = None
                        text_pool = None
                        text_embeds, text_pool = get_text_embeds(dropout, text_model, accelerator, captions, attn_mask, tokenizer, settings, batch_size)
                    
                    model_pred = unet(noisy_latents, timesteps, text_embeds).sample
                    target = noise

                    if "min_snr_gamma" in settings:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean([1, 2, 3])
                        loss = minSNR_weighting_loss(loss, timesteps, noise_scheduler, settings["min_snr_gamma"])
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Handle loss weighting for tags
                    if settings["tag_weighting_used"]:
                        loss_weighting_mult = get_loss_multiplier_for_batch(tag_weighting_dict, settings, batch[0]["captions"]) 
                        loss = loss * loss_weighting_mult

                    accelerator.backward(loss)
                    del timesteps, noise, latents, noisy_latents, text_embeds, target

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(itertools.chain(unet.parameters(), text_model.parameters()) if settings["train_text_encoder"] else unet.parameters(), 1.0)
                        last_grad_norm = grad_norm.mean().item()
                    
                    unet_optimizer.step()
                    unet_scheduler.step()
                    unet_optimizer.zero_grad()

                    if settings["train_text_encoder"]:
                        text_optimizer.step()
                        text_scheduler.step()
                        text_optimizer.zero_grad()
                    
                steps_bar.update(1)
                current_step += 1
                total_steps += 1

            if accelerator.is_main_process:
                logs = {
                    "loss": loss.item(),
                    "grad_norm": last_grad_norm,
                    "lr": unet_scheduler.get_last_lr()[0]
                }

                if settings["train_text_encoder"]:
                    logs["te_lr"] = text_scheduler.get_last_lr()[0]

                if settings["tag_weighting_used"]:
                    logs["loss_w"] = loss_weighting_mult
                
                epoch_bar.set_postfix(logs)
                accelerator.log(logs, step=total_steps)

            if current_step % settings["save_every"] == 0:
                step_path = os.path.join(settings["checkpoint_path"], f"{settings['experiment_id']}_e{e}_s{current_step}")
                save_sd1_pipeline(step_path, settings, accelerator, unet, text_model)
                accelerator.print(f"Model saved to: {step_path}")
                accelerator.wait_for_everyone()
        if (e+1) % settings["save_every_n_epoch"] == 0 or settings["save_every_n_epoch"] == 1:
            epoch_path = os.path.join(settings["checkpoint_path"], f"{settings['experiment_id']}_e{e+1}")
            save_sd1_pipeline(epoch_path, settings, accelerator, unet, text_model)
            accelerator.print(f"Model saved to: {epoch_path}")
            accelerator.wait_for_everyone()
        
        total_epochs += 1
        settings["seed"] += 1
        set_seed(settings["seed"])

if __name__ == "__main__":
    main()