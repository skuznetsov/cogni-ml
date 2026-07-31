#!/usr/bin/env python3
"""Export a tiny CPU-only TRELLIS.2 timestep and RoPE oracle.

The formulas are transcribed from microsoft/TRELLIS.2 at the pinned commit
below.  This script uses deterministic analytic parameters, no pretrained
weights, and never selects an accelerator device.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SCHEMA = "cogni-ml/trellis2/conditioning-oracle/v1"
PROVENANCE = {
    "repository": "microsoft/TRELLIS.2",
    "commit": PIN,
    "model": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/models/sparse_structure_flow.py",
    "rope": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/modules/attention/rope.py",
    "config": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/configs/gen/ss_flow_img_dit_1_3B_64_bf16.json",
    "oracle_kind": "pinned-source-transcribed PyTorch formula oracle",
}


def wave_values(count: int, amplitude: float, phase: float) -> torch.Tensor:
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


def build_fixture() -> dict[str, object]:
    torch.set_num_threads(1)

    batch = 3
    channels = 9
    frequency_dim = 256
    max_period = 10000.0
    coordinate_count = 5
    spatial_dim = 3
    head_dim = 128
    rope_low = 1.0
    rope_high = 10000.0

    timesteps = torch.tensor(
        [0.0, 0.3125, 1.75], dtype=torch.float32, device="cpu"
    )
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 1.0, 7.0],
            [15.0, 8.0, 2.0],
            [3.5, 6.25, 9.75],
        ],
        dtype=torch.float32,
        device="cpu",
    )

    parameters = {
        "timestep.first_linear.weight": wave_values(
            channels * frequency_dim, 0.041, 0.17
        ).reshape(channels, frequency_dim),
        "timestep.first_linear.bias": wave_values(channels, 0.023, -0.31),
        "timestep.second_linear.weight": wave_values(
            channels * channels, 0.067, 0.53
        ).reshape(channels, channels),
        "timestep.second_linear.bias": wave_values(channels, 0.019, 0.83),
        "modulation.linear.weight": wave_values(
            6 * channels * channels, 0.052, -0.47
        ).reshape(6 * channels, channels),
        "modulation.linear.bias": wave_values(6 * channels, 0.017, 1.07),
    }

    t_freq = timestep_embedding(timesteps, frequency_dim, max_period)
    first_linear = F.linear(
        t_freq,
        parameters["timestep.first_linear.weight"],
        parameters["timestep.first_linear.bias"],
    )
    first_silu = F.silu(first_linear)
    t_emb = F.linear(
        first_silu,
        parameters["timestep.second_linear.weight"],
        parameters["timestep.second_linear.bias"],
    )
    top_silu = F.silu(t_emb)
    modulation = F.linear(
        top_silu,
        parameters["modulation.linear.weight"],
        parameters["modulation.linear.bias"],
    )

    rope_frequency_dim = head_dim // 2 // spatial_dim
    frequencies = (
        torch.arange(rope_frequency_dim, dtype=torch.float32, device="cpu")
        / rope_frequency_dim
    )
    frequencies = rope_low / (rope_high**frequencies)
    angles = torch.outer(coordinates.reshape(-1), frequencies).reshape(
        coordinate_count, -1
    )
    phase_pairs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    pad_pairs = head_dim // 2 - angles.shape[-1]
    if pad_pairs:
        phase_pairs = torch.cat(
            (
                phase_pairs,
                torch.tensor(
                    [1.0, 0.0], dtype=torch.float32, device="cpu"
                ).reshape(1, 1, 2).expand(coordinate_count, pad_pairs, 2),
            ),
            dim=1,
        )

    return {
        "schema": SCHEMA,
        "provenance": {
            **PROVENANCE,
            "generator": "tools/trellis2_oracle/export_conditioning.py",
            "torch_version": torch.__version__,
        },
        "tolerance": {"absolute": 2e-5, "relative": 2e-5},
        "config": {
            "batch": batch,
            "channels": channels,
            "frequency_dim": frequency_dim,
            "max_period": max_period,
            "coordinate_count": coordinate_count,
            "spatial_dim": spatial_dim,
            "head_dim": head_dim,
            "rope_low": rope_low,
            "rope_high": rope_high,
            "device": "cpu",
            "dtype": "float32",
        },
        "inputs": {"timesteps": timesteps, "coordinates": coordinates},
        "parameters": parameters,
        "expected": {
            "t_freq": t_freq,
            "first_linear": first_linear,
            "first_silu": first_silu,
            "t_emb": t_emb,
            "top_silu": top_silu,
            "mod": modulation,
            "frequencies": frequencies,
            "angles": angles,
            "phases": phase_pairs,
        },
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
