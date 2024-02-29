import torch
from torch import nn
from torch.nn import Linear
from einops import rearrange

"""
Copyright [2022] [kohya-ss]

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

def convert_state_dict_mha_to_normal_attn(state_dict):
    # convert nn.MultiheadAttention to to_q/k/v and out_proj
    for key in list(state_dict.keys()):
        if "attention.attn." in key:
            if "in_proj_bias" in key:
                value = state_dict.pop(key)
                qkv = torch.chunk(value, 3, dim=0)
                state_dict[key.replace("in_proj_bias", "to_q.bias")] = qkv[0]
                state_dict[key.replace("in_proj_bias", "to_k.bias")] = qkv[1]
                state_dict[key.replace("in_proj_bias", "to_v.bias")] = qkv[2]
            elif "in_proj_weight" in key:
                value = state_dict.pop(key)
                qkv = torch.chunk(value, 3, dim=0)
                state_dict[key.replace("in_proj_weight", "to_q.weight")] = qkv[0]
                state_dict[key.replace("in_proj_weight", "to_k.weight")] = qkv[1]
                state_dict[key.replace("in_proj_weight", "to_v.weight")] = qkv[2]
            elif "out_proj.bias" in key:
                value = state_dict.pop(key)
                state_dict[key.replace("out_proj.bias", "out_proj.bias")] = value
            elif "out_proj.weight" in key:
                value = state_dict.pop(key)
                state_dict[key.replace("out_proj.weight", "out_proj.weight")] = value
    return state_dict


def convert_state_dict_normal_attn_to_mha(state_dict):
    # convert to_q/k/v and out_proj to nn.MultiheadAttention
    for key in list(state_dict.keys()):
        if "attention.attn." in key:
            if "to_q.bias" in key:
                q = state_dict.pop(key)
                k = state_dict.pop(key.replace("to_q.bias", "to_k.bias"))
                v = state_dict.pop(key.replace("to_q.bias", "to_v.bias"))
                state_dict[key.replace("to_q.bias", "in_proj_bias")] = torch.cat([q, k, v])
            elif "to_q.weight" in key:
                q = state_dict.pop(key)
                k = state_dict.pop(key.replace("to_q.weight", "to_k.weight"))
                v = state_dict.pop(key.replace("to_q.weight", "to_v.weight"))
                state_dict[key.replace("to_q.weight", "in_proj_weight")] = torch.cat([q, k, v])
            elif "out_proj.bias" in key:
                v = state_dict.pop(key)
                state_dict[key.replace("out_proj.bias", "out_proj.bias")] = v
            elif "out_proj.weight" in key:
                v = state_dict.pop(key)
                state_dict[key.replace("out_proj.weight", "out_proj.weight")] = v
    return state_dict

class Attention(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()

        self.to_q = Linear(c, c, bias=True)
        self.to_k = Linear(c, c, bias=True)
        self.to_v = Linear(c, c, bias=True)
        self.out_proj = Linear(c, c, bias=True)
        self.nhead = nhead
        self.dropout = dropout
        self.scale = (c // nhead) ** -0.5

    def forward(self, q_in, k_in, v_in):
        q_in = self.to_q(q_in)
        k_in = self.to_k(k_in)
        v_in = self.to_v(v_in)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.nhead), (q_in, k_in, v_in))
        del q_in, k_in, v_in
        out = self.forward_memory_efficient_xformers(q, k, v)
        del q, k, v
        out = rearrange(out, "b n h d -> b n (h d)", h=self.nhead)

        return self.out_proj(out)

    def _attention(self, query, key, value):
        # if self.upcast_attention:
        #     query = query.float()
        #     key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        return hidden_states

    def forward_memory_efficient_xformers(self, q, k, v):
        import xformers.ops

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる
        del q, k, v

        return out

class FlashAttention2D(nn.Module):

    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        # self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)
        self.attn = Attention(c, nhead, dropout=dropout)  # , bias=True, batch_first=True)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        # x = self.attn(x, kv, kv, need_weights=False)[0]
        x = self.attn(x, kv, kv)
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x