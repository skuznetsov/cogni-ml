"""Text-only final full-attention Qwen3.5 block, initialized from local GGUF."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TailConfig:
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    eps: float = 1e-6
    rope_base: float = 10000000.0


def rms(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


class LastBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = nn.Parameter(torch.ones(cfg.hidden))
        self.post_attention_norm = nn.Parameter(torch.ones(cfg.hidden))
        self.attn_q_norm = nn.Parameter(torch.ones(cfg.head_dim))
        self.attn_k_norm = nn.Parameter(torch.ones(cfg.head_dim))
        for name, out_dim, in_dim in [
            ("attn_q", 2 * cfg.heads * cfg.head_dim, cfg.hidden),
            ("attn_k", cfg.kv_heads * cfg.head_dim, cfg.hidden),
            ("attn_v", cfg.kv_heads * cfg.head_dim, cfg.hidden),
            ("attn_output", cfg.hidden, cfg.heads * cfg.head_dim),
            ("ffn_gate", cfg.intermediate, cfg.hidden),
            ("ffn_up", cfg.intermediate, cfg.hidden),
            ("ffn_down", cfg.hidden, cfg.intermediate),
        ]:
            setattr(self, name, nn.Linear(in_dim, out_dim, bias=False))

    def rope(self, x):
        c = self.cfg
        half = c.rotary_dim // 2
        freq = c.rope_base ** (-torch.arange(half, device=x.device).float() * 2 / c.rotary_dim)
        theta = torch.arange(x.shape[0], device=x.device).float()[:, None, None] * freq
        a, b = x[..., :half], x[..., half:c.rotary_dim]
        return torch.cat([a * theta.cos() - b * theta.sin(),
                          b * theta.cos() + a * theta.sin(), x[..., c.rotary_dim:]], -1)

    def forward(self, x):
        c, n = self.cfg, x.shape[0]
        cur = rms(x, self.attn_norm, c.eps)
        q_gate = self.attn_q(cur).reshape(n, c.heads, 2 * c.head_dim)
        q, gate = q_gate.split(c.head_dim, dim=-1)
        k = self.attn_k(cur).reshape(n, c.kv_heads, c.head_dim)
        v = self.attn_v(cur).reshape(n, c.kv_heads, c.head_dim)
        q = self.rope(rms(q, self.attn_q_norm, c.eps))
        k = self.rope(rms(k, self.attn_k_norm, c.eps))
        k = k.repeat_interleave(c.heads // c.kv_heads, dim=1)
        v = v.repeat_interleave(c.heads // c.kv_heads, dim=1)
        a = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                                           v.transpose(0, 1), is_causal=True)
        a = a.transpose(0, 1) * gate.sigmoid()
        x = x + self.attn_output(a.reshape(n, -1))
        cur = rms(x, self.post_attention_norm, c.eps)
        return x + self.ffn_down(F.silu(self.ffn_gate(cur)) * self.ffn_up(cur))


class QwenTail(nn.Module):
    def __init__(self, cfg, labels):
        super().__init__()
        self.cfg = cfg
        self.block = LastBlock(cfg)
        self.register_buffer("output_norm", torch.ones(cfg.hidden))
        self.head = nn.Linear(cfg.hidden, labels, bias=False)

    def forward(self, frozen_prefix):
        hidden = self.block(frozen_prefix.detach())[-1]
        return self.head(rms(hidden, self.output_norm, self.cfg.eps))


def configure_mode(model, mode):
    if mode not in {"baseline", "head", "last-block"}:
        raise ValueError(f"unknown mode {mode}")
    for name, param in model.named_parameters():
        param.requires_grad_(name == "head.weight" if mode == "head"
                             else name.startswith("block.") if mode == "last-block" else False)


def from_gguf(path, label_ids):
    # Existing gguf package; no HF model loading, network or package installation.
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize
    reader = GGUFReader(str(path))
    fields = reader.fields
    def value(name, default=None):
        f = fields.get("qwen35." + name)
        return f.contents() if f is not None else default
    layers = int(value("block_count")) - int(value("nextn_predict_layers", 0))
    if layers % int(value("full_attention_interval")):
        raise ValueError("last model block is recurrent, not supported by this experiment")
    cfg = TailConfig(int(value("embedding_length")), int(value("feed_forward_length")),
                     int(value("attention.head_count")), int(value("attention.head_count_kv")),
                     int(value("attention.key_length")), int(value("rope.dimension_count")),
                     float(value("attention.layer_norm_rms_epsilon", 1e-6)),
                     float(value("rope.freq_base", 10000000.0)))
    tensors = {x.name: x for x in reader.tensors}
    def array(name):
        t = tensors[name]
        a = dequantize(t.data, t.tensor_type).reshape(tuple(t.shape[::-1]))
        return torch.from_numpy(np.array(a, dtype=np.float32, copy=True))
    model = QwenTail(cfg, len(label_ids))
    with torch.no_grad():
        for name, param in model.block.named_parameters():
            key = name if name.endswith(".weight") else name + ".weight"
            param.copy_(array(f"blk.{layers - 1}.{key}"))
        model.output_norm.copy_(array("output_norm.weight"))
        head = tensors.get("output.weight", tensors["token_embd.weight"])
        rows = [dequantize(head.data[int(i):int(i) + 1], head.tensor_type).reshape(cfg.hidden)
                for i in label_ids]
        model.head.weight.copy_(torch.from_numpy(np.stack(rows).astype(np.float32)))
    return model
