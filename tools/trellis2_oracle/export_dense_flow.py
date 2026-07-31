#!/usr/bin/env python3
"""Export one unified tiny CPU-only TRELLIS.2 dense-flow forward oracle.

The formulas are transcribed from the pinned upstream sources. The fixture uses
deterministic analytic F32 parameters, no pretrained weights, and no accelerator.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SCHEMA = "cogni-ml/trellis2/dense-flow-oracle/v1"
PROVENANCE = {
    "repository": "microsoft/TRELLIS.2",
    "commit": PIN,
    "model": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/models/sparse_structure_flow.py",
    "block": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/modules/transformer/modulated.py",
    "attention": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/modules/attention/modules.py",
    "rope": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/modules/attention/rope.py",
    "config": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/configs/gen/ss_flow_img_dit_1_3B_64_bf16.json",
    "oracle_kind": "pinned-source-transcribed unified PyTorch formula oracle",
}


def ramp(count: int, start: float, step: float) -> torch.Tensor:
    return torch.tensor(
        [start + step * index for index in range(count)],
        dtype=torch.float32,
        device="cpu",
    )


def wave(count: int, amplitude: float, phase: float) -> torch.Tensor:
    return torch.tensor(
        [
            amplitude * math.sin(index * 0.37 + phase)
            + amplitude * 0.29 * math.cos(index * 0.13 - phase)
            for index in range(count)
        ],
        dtype=torch.float32,
        device="cpu",
    )


def timestep_embedding(
    timesteps: torch.Tensor, dimension: int, max_period: float
) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device="cpu")
        / half
    )
    arguments = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if dimension % 2:
        embedding = torch.cat(
            (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
        )
    return embedding


def multi_head_rms_norm(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1) * gamma * math.sqrt(x.shape[-1])


def rotary(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    real = pairs[..., 0]
    imaginary = pairs[..., 1]
    cosine = phases[..., 0].unsqueeze(-2)
    sine = phases[..., 1].unsqueeze(-2)
    return torch.stack(
        (
            real * cosine - imaginary * sine,
            real * sine + imaginary * cosine,
        ),
        dim=-1,
    ).reshape_as(x)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return (weights @ v).permute(0, 2, 1, 3)


def build_fixture() -> dict[str, object]:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    batch = 2
    resolution = 2
    voxel_count = resolution**3
    in_channels = 3
    model_channels = 16
    context_channels = 5
    context_length = 3
    out_channels = 4
    num_heads = 2
    head_dim = model_channels // num_heads
    mlp_ratio = 1.25
    hidden_channels = int(model_channels * mlp_ratio)
    frequency_dim = 256
    max_period = 10000.0
    block_eps = 1e-6
    final_eps = 1e-5

    voxels = wave(
        batch * in_channels * voxel_count, 0.73, -0.19
    ).reshape(batch, in_channels, resolution, resolution, resolution)
    # Break any accidental channel/axis symmetry with an independent ramp.
    voxels = voxels + ramp(voxels.numel(), -0.17, 0.0061).reshape_as(voxels)
    timesteps = torch.tensor([0.1875, 1.375], dtype=torch.float32, device="cpu")
    context = wave(
        batch * context_length * context_channels, 0.41, 0.71
    ).reshape(batch, context_length, context_channels)
    context = context + ramp(context.numel(), 0.09, -0.0047).reshape_as(context)

    parameters = {
        "input_layer.weight": wave(model_channels * in_channels, 0.081, 0.11).reshape(
            model_channels, in_channels
        ),
        "input_layer.bias": wave(model_channels, 0.027, -0.43),
        "timestep.first_linear.weight": wave(
            model_channels * frequency_dim, 0.039, 0.23
        ).reshape(model_channels, frequency_dim),
        "timestep.first_linear.bias": wave(model_channels, 0.021, -0.37),
        "timestep.second_linear.weight": wave(
            model_channels * model_channels, 0.057, 0.59
        ).reshape(model_channels, model_channels),
        "timestep.second_linear.bias": wave(model_channels, 0.018, 0.91),
        "top_modulation.linear.weight": wave(
            6 * model_channels * model_channels, 0.046, -0.49
        ).reshape(6 * model_channels, model_channels),
        "top_modulation.linear.bias": wave(6 * model_channels, 0.016, 1.03),
        "block.modulation": ramp(6 * model_channels, -0.087, 0.00261),
        "block.norm2.weight": ramp(model_channels, 0.84, 0.017),
        "block.norm2.bias": ramp(model_channels, -0.057, 0.0083),
        "block.self_attn.to_qkv.weight": ramp(
            3 * model_channels * model_channels, -0.113, 0.00029
        ).reshape(3 * model_channels, model_channels),
        "block.self_attn.to_qkv.bias": ramp(3 * model_channels, 0.029, -0.00117),
        "block.self_attn.q_rms_norm.gamma": ramp(
            model_channels, 0.92, 0.011
        ).reshape(num_heads, head_dim),
        "block.self_attn.k_rms_norm.gamma": ramp(
            model_channels, 1.06, -0.008
        ).reshape(num_heads, head_dim),
        "block.self_attn.to_out.weight": ramp(
            model_channels * model_channels, 0.073, -0.00051
        ).reshape(model_channels, model_channels),
        "block.self_attn.to_out.bias": ramp(model_channels, -0.022, 0.0037),
        "block.cross_attn.to_q.weight": wave(
            model_channels * model_channels, 0.068, 0.31
        ).reshape(model_channels, model_channels),
        "block.cross_attn.to_q.bias": wave(model_channels, 0.026, -0.39),
        "block.cross_attn.to_kv.weight": ramp(
            2 * model_channels * context_channels, 0.089, -0.00073
        ).reshape(2 * model_channels, context_channels),
        "block.cross_attn.to_kv.bias": ramp(2 * model_channels, -0.024, 0.00153),
        "block.cross_attn.q_rms_norm.gamma": ramp(
            model_channels, 1.01, -0.0053
        ).reshape(num_heads, head_dim),
        "block.cross_attn.k_rms_norm.gamma": ramp(
            model_channels, 0.89, 0.0071
        ).reshape(num_heads, head_dim),
        "block.cross_attn.to_out.weight": ramp(
            model_channels * model_channels, -0.061, 0.00043
        ).reshape(model_channels, model_channels),
        "block.cross_attn.to_out.bias": ramp(model_channels, 0.013, -0.0019),
        "block.mlp.mlp.0.weight": ramp(
            hidden_channels * model_channels, 0.052, -0.00039
        ).reshape(hidden_channels, model_channels),
        "block.mlp.mlp.0.bias": ramp(hidden_channels, -0.029, 0.0027),
        "block.mlp.mlp.2.weight": ramp(
            model_channels * hidden_channels, -0.041, 0.00033
        ).reshape(model_channels, hidden_channels),
        "block.mlp.mlp.2.bias": ramp(model_channels, 0.019, -0.0022),
        "out_layer.weight": wave(
            out_channels * model_channels, 0.063, -0.77
        ).reshape(out_channels, model_channels),
        "out_layer.bias": wave(out_channels, 0.024, 0.47),
    }

    expected: dict[str, torch.Tensor] = {}
    flattened = voxels.view(batch, in_channels, -1).permute(0, 2, 1).contiguous()
    expected["flattened"] = flattened
    h = F.linear(
        flattened, parameters["input_layer.weight"], parameters["input_layer.bias"]
    )
    expected["input_projected"] = h

    t_freq = timestep_embedding(timesteps, frequency_dim, max_period)
    expected["conditioning.t_freq"] = t_freq
    first_linear = F.linear(
        t_freq,
        parameters["timestep.first_linear.weight"],
        parameters["timestep.first_linear.bias"],
    )
    expected["conditioning.first_linear"] = first_linear
    first_silu = F.silu(first_linear)
    expected["conditioning.first_silu"] = first_silu
    t_emb = F.linear(
        first_silu,
        parameters["timestep.second_linear.weight"],
        parameters["timestep.second_linear.bias"],
    )
    expected["conditioning.t_emb"] = t_emb
    top_silu = F.silu(t_emb)
    expected["conditioning.top_silu"] = top_silu
    top_mod = F.linear(
        top_silu,
        parameters["top_modulation.linear.weight"],
        parameters["top_modulation.linear.bias"],
    )
    expected["conditioning.mod"] = top_mod

    axis = torch.arange(resolution, dtype=torch.float32, device="cpu")
    coordinates = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    expected["coordinates"] = coordinates
    rope_frequency_dim = head_dim // 2 // 3
    frequencies = torch.arange(
        rope_frequency_dim, dtype=torch.float32, device="cpu"
    ) / rope_frequency_dim
    frequencies = 1.0 / (10000.0**frequencies)
    expected["rope.frequencies"] = frequencies
    angles = torch.outer(coordinates.reshape(-1), frequencies).reshape(
        voxel_count, -1
    )
    expected["rope.angles"] = angles
    phase_angles = angles
    if phase_angles.shape[-1] < head_dim // 2:
        phase_angles = torch.cat(
            (
                phase_angles,
                torch.zeros(
                    voxel_count,
                    head_dim // 2 - phase_angles.shape[-1],
                    dtype=torch.float32,
                    device="cpu",
                ),
            ),
            dim=-1,
        )
    phases = torch.stack((torch.cos(phase_angles), torch.sin(phase_angles)), dim=-1)
    expected["rope.phases"] = phases

    combined_mod = top_mod + parameters["block.modulation"]
    expected["block.combined_mod"] = combined_mod
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        combined_mod.chunk(6, dim=1)
    )
    norm1 = F.layer_norm(h.float(), (model_channels,), eps=block_eps)
    expected["block.norm1"] = norm1
    modulated_self = norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    expected["block.modulated_self"] = modulated_self

    self_qkv = F.linear(
        modulated_self,
        parameters["block.self_attn.to_qkv.weight"],
        parameters["block.self_attn.to_qkv.bias"],
    ).reshape(batch, voxel_count, 3, num_heads, head_dim)
    self_q, self_k, self_v = self_qkv.unbind(dim=2)
    self_q = multi_head_rms_norm(
        self_q, parameters["block.self_attn.q_rms_norm.gamma"]
    )
    self_k = multi_head_rms_norm(
        self_k, parameters["block.self_attn.k_rms_norm.gamma"]
    )
    self_q = rotary(self_q, phases)
    self_k = rotary(self_k, phases)
    expected["block.self_q"] = self_q
    expected["block.self_k"] = self_k
    self_attention = attention(self_q, self_k, self_v).reshape(
        batch, voxel_count, model_channels
    )
    self_attention = F.linear(
        self_attention,
        parameters["block.self_attn.to_out.weight"],
        parameters["block.self_attn.to_out.bias"],
    )
    expected["block.self_attention"] = self_attention
    after_self = h + self_attention * gate_msa.unsqueeze(1)
    expected["block.after_self"] = after_self

    norm2 = F.layer_norm(
        after_self.float(),
        (model_channels,),
        parameters["block.norm2.weight"],
        parameters["block.norm2.bias"],
        block_eps,
    )
    expected["block.norm2"] = norm2
    cross_q = F.linear(
        norm2,
        parameters["block.cross_attn.to_q.weight"],
        parameters["block.cross_attn.to_q.bias"],
    ).reshape(batch, voxel_count, num_heads, head_dim)
    cross_kv = F.linear(
        context,
        parameters["block.cross_attn.to_kv.weight"],
        parameters["block.cross_attn.to_kv.bias"],
    ).reshape(batch, context_length, 2, num_heads, head_dim)
    cross_k, cross_v = cross_kv.unbind(dim=2)
    cross_q = multi_head_rms_norm(
        cross_q, parameters["block.cross_attn.q_rms_norm.gamma"]
    )
    cross_k = multi_head_rms_norm(
        cross_k, parameters["block.cross_attn.k_rms_norm.gamma"]
    )
    expected["block.cross_q"] = cross_q
    expected["block.cross_k"] = cross_k
    cross_attention = attention(cross_q, cross_k, cross_v).reshape(
        batch, voxel_count, model_channels
    )
    cross_attention = F.linear(
        cross_attention,
        parameters["block.cross_attn.to_out.weight"],
        parameters["block.cross_attn.to_out.bias"],
    )
    expected["block.cross_attention"] = cross_attention
    after_cross = after_self + cross_attention
    expected["block.after_cross"] = after_cross

    norm3 = F.layer_norm(after_cross.float(), (model_channels,), eps=block_eps)
    expected["block.norm3"] = norm3
    modulated_mlp = norm3 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    expected["block.modulated_mlp"] = modulated_mlp
    mlp_hidden = F.linear(
        modulated_mlp,
        parameters["block.mlp.mlp.0.weight"],
        parameters["block.mlp.mlp.0.bias"],
    )
    mlp_hidden = F.gelu(mlp_hidden, approximate="tanh")
    mlp_output = F.linear(
        mlp_hidden,
        parameters["block.mlp.mlp.2.weight"],
        parameters["block.mlp.mlp.2.bias"],
    )
    expected["block.mlp_output"] = mlp_output
    block_output = after_cross + mlp_output * gate_mlp.unsqueeze(1)
    expected["block.output"] = block_output

    final_norm = F.layer_norm(block_output.float(), (model_channels,), eps=final_eps)
    expected["final_norm"] = final_norm
    output_tokens = F.linear(
        final_norm, parameters["out_layer.weight"], parameters["out_layer.bias"]
    )
    expected["output_tokens"] = output_tokens
    output = output_tokens.permute(0, 2, 1).reshape(
        batch, out_channels, resolution, resolution, resolution
    ).contiguous()
    expected["output"] = output

    return {
        "schema": SCHEMA,
        "provenance": {
            **PROVENANCE,
            "generator": "tools/trellis2_oracle/export_dense_flow.py",
            "torch_version": torch.__version__,
        },
        "tolerance": {"absolute": 2e-5, "relative": 2e-5},
        "config": {
            "batch": batch,
            "resolution": resolution,
            "voxel_count": voxel_count,
            "in_channels": in_channels,
            "model_channels": model_channels,
            "context_channels": context_channels,
            "context_length": context_length,
            "out_channels": out_channels,
            "num_blocks": 1,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "mlp_ratio": mlp_ratio,
            "hidden_channels": hidden_channels,
            "frequency_dim": frequency_dim,
            "max_period": max_period,
            "block_eps": block_eps,
            "final_eps": final_eps,
            "pe_mode": "rope",
            "share_mod": True,
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
            "device": "cpu",
            "dtype": "float32",
        },
        "inputs": {
            "voxels": voxels,
            "timesteps": timesteps,
            "context": context,
        },
        "parameters": parameters,
        "expected": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    fixture = build_fixture()
    tensor_path = args.manifest.with_suffix(".safetensors")
    tensors: dict[str, torch.Tensor] = {}
    for section, prefix in (
        ("inputs", "input"),
        ("parameters", "parameter"),
        ("expected", "expected"),
    ):
        compact = {}
        for name, tensor in fixture[section].items():
            tensor_name = f"{prefix}.{name}"
            tensor = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
            tensors[tensor_name] = tensor
            compact[name] = {
                "tensor": tensor_name,
                "shape": list(tensor.shape),
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
                {"schema": SCHEMA, "upstream_commit": PIN},
                separators=(",", ":"),
                sort_keys=True,
            )
        },
    )
    args.manifest.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.manifest} and {tensor_path} "
        f"(torch={torch.__version__}, device=cpu)"
    )


if __name__ == "__main__":
    main()
