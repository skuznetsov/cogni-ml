#!/usr/bin/env python3
"""Export a tiny CPU-only TRELLIS.2 shared-modulated cross-block oracle.

The equations and parameter layout are transcribed from the pinned upstream
sources listed in PROVENANCE. This script intentionally uses no pretrained
weights and never selects an accelerator device.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
PROVENANCE = {
    "repository": "microsoft/TRELLIS.2",
    "commit": PIN,
    "model": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/models/sparse_structure_flow.py",
    "block": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/modules/transformer/modulated.py",
    "attention": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/modules/attention/modules.py",
    "rope": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/modules/attention/rope.py",
    "config": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/configs/gen/ss_flow_img_dit_1_3B_64_bf16.json",
    "oracle_kind": "pinned-source-transcribed PyTorch formula oracle",
}


def values(count: int, start: float, step: float) -> torch.Tensor:
    return torch.tensor(
        [start + step * index for index in range(count)],
        dtype=torch.float32,
        device="cpu",
    )


def wave_values(count: int, amplitude: float, phase: float) -> torch.Tensor:
    return torch.tensor(
        [
            amplitude * math.sin(index * 0.37 + phase)
            + amplitude * 0.31 * math.cos(index * 0.11 - phase)
            for index in range(count)
        ],
        dtype=torch.float32,
        device="cpu",
    )


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight, bias)


def multi_head_rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    scale = math.sqrt(x.shape[-1])
    return F.normalize(x.float(), dim=-1) * gamma * scale


def rotary(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    real = pairs[..., 0]
    imag = pairs[..., 1]
    cos = phases[..., 0].unsqueeze(-2)
    sin = phases[..., 1].unsqueeze(-2)
    rotated = torch.stack(
        (real * cos - imag * sin, real * sin + imag * cos),
        dim=-1,
    )
    return rotated.reshape_as(x)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return (weights @ v).permute(0, 2, 1, 3)


def flat(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1)]


def tensor_payload(tensor: torch.Tensor) -> dict[str, object]:
    return {"shape": list(tensor.shape), "values": flat(tensor)}


def build_fixture() -> dict[str, object]:
    torch.set_num_threads(1)

    batch = 2
    length = 3
    context_length = 2
    channels = 16
    context_channels = 5
    heads = 2
    head_dim = channels // heads
    hidden_channels = 20
    eps = 1e-6

    x = values(batch * length * channels, -0.55, 0.021).reshape(
        batch, length, channels
    )
    mod = values(batch * 6 * channels, 0.17, -0.0023).reshape(
        batch, 6 * channels
    )
    context = values(
        batch * context_length * context_channels, -0.31, 0.037
    ).reshape(batch, context_length, context_channels)

    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 1.0, 4.0]],
        dtype=torch.float32,
        device="cpu",
    )
    frequency_dimension = head_dim // 2 // 3
    frequencies = torch.arange(
        frequency_dimension, dtype=torch.float32, device="cpu"
    ) / frequency_dimension
    frequencies = 1.0 / (10000.0**frequencies)
    angles = torch.outer(coordinates.reshape(-1), frequencies).reshape(
        length, -1
    )
    phase_pairs = head_dim // 2
    if angles.shape[-1] < phase_pairs:
        angles = torch.cat(
            (
                angles,
                torch.zeros(
                    length,
                    phase_pairs - angles.shape[-1],
                    dtype=torch.float32,
                    device="cpu",
                ),
            ),
            dim=-1,
        )
    phases = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    parameters = {
        "modulation": values(6 * channels, -0.09, 0.0027),
        "norm2.weight": values(channels, 0.83, 0.019),
        "norm2.bias": values(channels, -0.06, 0.009),
        "self_attn.to_qkv.weight": values(3 * channels * channels, -0.12, 0.00031).reshape(
            3 * channels, channels
        ),
        "self_attn.to_qkv.bias": values(3 * channels, 0.031, -0.0013),
        "self_attn.q_rms_norm.gamma": values(channels, 0.91, 0.013).reshape(
            heads, head_dim
        ),
        "self_attn.k_rms_norm.gamma": values(channels, 1.07, -0.009).reshape(
            heads, head_dim
        ),
        "self_attn.to_out.weight": values(channels * channels, 0.08, -0.00057).reshape(
            channels, channels
        ),
        "self_attn.to_out.bias": values(channels, -0.025, 0.004),
        "cross_attn.to_q.weight": wave_values(channels * channels, 0.071, 0.23).reshape(
            channels, channels
        ),
        "cross_attn.to_q.bias": wave_values(channels, 0.029, -0.41),
        "cross_attn.to_kv.weight": values(
            2 * channels * context_channels, 0.095, -0.00081
        ).reshape(2 * channels, context_channels),
        "cross_attn.to_kv.bias": values(2 * channels, -0.027, 0.0017),
        "cross_attn.q_rms_norm.gamma": values(channels, 1.02, -0.006).reshape(
            heads, head_dim
        ),
        "cross_attn.k_rms_norm.gamma": values(channels, 0.88, 0.008).reshape(
            heads, head_dim
        ),
        "cross_attn.to_out.weight": values(
            channels * channels, -0.065, 0.00049
        ).reshape(channels, channels),
        "cross_attn.to_out.bias": values(channels, 0.014, -0.0021),
        "mlp.mlp.0.weight": values(
            hidden_channels * channels, 0.057, -0.00043
        ).reshape(hidden_channels, channels),
        "mlp.mlp.0.bias": values(hidden_channels, -0.032, 0.003),
        "mlp.mlp.2.weight": values(
            channels * hidden_channels, -0.044, 0.00037
        ).reshape(channels, hidden_channels),
        "mlp.mlp.2.bias": values(channels, 0.021, -0.0025),
    }

    combined_mod = mod + parameters["modulation"]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        combined_mod.chunk(6, dim=1)
    )

    norm1 = F.layer_norm(x.float(), (channels,), eps=eps)
    modulated_self = norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

    self_qkv = linear(
        modulated_self,
        parameters["self_attn.to_qkv.weight"],
        parameters["self_attn.to_qkv.bias"],
    ).reshape(batch, length, 3, heads, head_dim)
    self_q, self_k, self_v = self_qkv.unbind(dim=2)
    self_q = multi_head_rms_norm(
        self_q, parameters["self_attn.q_rms_norm.gamma"]
    )
    self_k = multi_head_rms_norm(
        self_k, parameters["self_attn.k_rms_norm.gamma"]
    )
    self_q = rotary(self_q, phases)
    self_k = rotary(self_k, phases)
    self_attention = attention(self_q, self_k, self_v).reshape(
        batch, length, channels
    )
    self_attention = linear(
        self_attention,
        parameters["self_attn.to_out.weight"],
        parameters["self_attn.to_out.bias"],
    )
    after_self = x + self_attention * gate_msa.unsqueeze(1)

    norm2 = F.layer_norm(
        after_self.float(),
        (channels,),
        parameters["norm2.weight"],
        parameters["norm2.bias"],
        eps,
    )
    cross_q = linear(
        norm2,
        parameters["cross_attn.to_q.weight"],
        parameters["cross_attn.to_q.bias"],
    ).reshape(batch, length, heads, head_dim)
    cross_kv = linear(
        context,
        parameters["cross_attn.to_kv.weight"],
        parameters["cross_attn.to_kv.bias"],
    ).reshape(batch, context_length, 2, heads, head_dim)
    cross_k, cross_v = cross_kv.unbind(dim=2)
    cross_q = multi_head_rms_norm(
        cross_q, parameters["cross_attn.q_rms_norm.gamma"]
    )
    cross_k = multi_head_rms_norm(
        cross_k, parameters["cross_attn.k_rms_norm.gamma"]
    )
    cross_attention = attention(cross_q, cross_k, cross_v).reshape(
        batch, length, channels
    )
    cross_attention = linear(
        cross_attention,
        parameters["cross_attn.to_out.weight"],
        parameters["cross_attn.to_out.bias"],
    )
    after_cross = after_self + cross_attention

    norm3 = F.layer_norm(after_cross.float(), (channels,), eps=eps)
    modulated_mlp = norm3 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    mlp_hidden = linear(
        modulated_mlp,
        parameters["mlp.mlp.0.weight"],
        parameters["mlp.mlp.0.bias"],
    )
    mlp_hidden = F.gelu(mlp_hidden, approximate="tanh")
    mlp_output = linear(
        mlp_hidden,
        parameters["mlp.mlp.2.weight"],
        parameters["mlp.mlp.2.bias"],
    )
    output = after_cross + mlp_output * gate_mlp.unsqueeze(1)

    named = {
        "combined_mod": combined_mod,
        "norm1": norm1,
        "modulated_self": modulated_self,
        "self_q": self_q,
        "self_k": self_k,
        "self_attention": self_attention,
        "after_self": after_self,
        "norm2": norm2,
        "cross_q": cross_q,
        "cross_k": cross_k,
        "cross_attention": cross_attention,
        "after_cross": after_cross,
        "norm3": norm3,
        "modulated_mlp": modulated_mlp,
        "mlp_output": mlp_output,
        "output": output,
    }

    return {
        "schema": "cogni-ml/trellis2/shared-modulated-cross-block-oracle/v1",
        "provenance": {
            **PROVENANCE,
            "generator": "tools/trellis2_oracle/export_shared_modulated_cross_block.py",
            "torch_version": torch.__version__,
        },
        "tolerance": {
            "absolute": 2e-5,
            "relative": 2e-5,
        },
        "config": {
            "batch": batch,
            "length": length,
            "context_length": context_length,
            "channels": channels,
            "context_channels": context_channels,
            "heads": heads,
            "mlp_ratio": 1.25,
            "eps": eps,
            "share_mod": True,
            "use_rope": True,
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
            "device": "cpu",
            "dtype": "float32",
        },
        "inputs": {
            "x": tensor_payload(x),
            "mod": tensor_payload(mod),
            "context": tensor_payload(context),
            "phases": tensor_payload(phases),
        },
        "parameters": {
            name: tensor_payload(value) for name, value in parameters.items()
        },
        "expected": {name: tensor_payload(value) for name, value in named.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    fixture = build_fixture()
    tensor_path = args.manifest.with_suffix(".safetensors")
    tensors = {}
    for section, prefix in (
        ("inputs", "input"),
        ("parameters", "parameter"),
        ("expected", "expected"),
    ):
        compact = {}
        for name, payload in fixture[section].items():
            tensor_name = f"{prefix}.{name}"
            tensors[tensor_name] = torch.tensor(
                payload["values"], dtype=torch.float32, device="cpu"
            ).reshape(payload["shape"])
            compact[name] = {
                "tensor": tensor_name,
                "shape": payload["shape"],
                "dtype": "F32",
            }
        fixture[section] = compact

    fixture["tensor_file"] = tensor_path.name
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        tensor_path,
        metadata={
            "oracle": json.dumps(
                {
                    "device": "cpu",
                    "schema": fixture["schema"],
                    "upstream_commit": PIN,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    )
    args.manifest.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.manifest} and {tensor_path} "
        f"(torch={torch.__version__}, device=cpu)"
    )


if __name__ == "__main__":
    main()
