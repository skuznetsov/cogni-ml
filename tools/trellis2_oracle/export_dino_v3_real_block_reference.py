#!/usr/bin/env python3
"""Export an independent real-weight DINOv3 layer-0 CPU reference.

The local Transformers installation does not expose the pinned DINOv3 runtime,
so this oracle uses the pinned source formulas directly with torch and reads
only layer-0 F32 tensors from the user-provided safetensors checkpoint. It
uses eight tokens (CLS, four registers, three patches) to make the real
1024/4096-width block measurable without opening the full encoder corridor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
TRANSFORMERS_MODELING_SHA256 = (
    "6073b7665eea50fb2260d86984011e6af8ed68cbd37ae57a6095e5e90e0eea34"
)
TRANSFORMERS_CONFIG_SHA256 = (
    "9a13d3c9ea8020aaed7057db28a7ffe0ffd9bb094a6178261bcc45ed04e9bbfc"
)
MODEL_REFERENCE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MODEL_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
SCHEMA = "cogni-ml/trellis2/dino-v3-real-block-reference/v1"

TOKEN_COUNT = 8
PATCH_COUNT = 3
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
HEADS = 16
HEAD_DIM = 64
REGISTERS = 4
EPS = 1.0e-5

PARAMETER_KEYS = (
    "layer.0.norm1.weight",
    "layer.0.norm1.bias",
    "layer.0.attention.q_proj.weight",
    "layer.0.attention.q_proj.bias",
    "layer.0.attention.k_proj.weight",
    "layer.0.attention.v_proj.weight",
    "layer.0.attention.v_proj.bias",
    "layer.0.attention.o_proj.weight",
    "layer.0.attention.o_proj.bias",
    "layer.0.layer_scale1.lambda1",
    "layer.0.norm2.weight",
    "layer.0.norm2.bias",
    "layer.0.mlp.up_proj.weight",
    "layer.0.mlp.up_proj.bias",
    "layer.0.mlp.down_proj.weight",
    "layer.0.mlp.down_proj.bias",
    "layer.0.layer_scale2.lambda1",
)


def f32le_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()


def f32le_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(f32le_bytes(tensor)).hexdigest()


def load_parameters(checkpoint: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        parameters = {key: handle.get_tensor(key).contiguous() for key in PARAMETER_KEYS}
    for key, tensor in parameters.items():
        if tensor.dtype != torch.float32:
            raise AssertionError(f"{key} is {tensor.dtype}, expected float32")
    return parameters


def build_input() -> torch.Tensor:
    values = [(((index * 17 + 5) % 43) - 21) / 19.0 for index in range(TOKEN_COUNT * HIDDEN_SIZE)]
    return torch.tensor(values, dtype=torch.float32).reshape(1, TOKEN_COUNT, HIDDEN_SIZE)


def build_rope() -> tuple[torch.Tensor, torch.Tensor]:
    # This is the pinned DINOv3ViTRopePositionEmbedding formula for a 1x3
    # patch grid. Prefix tokens are intentionally excluded from the arrays.
    inv_freq = 1.0 / (100.0 ** torch.arange(0, 1, 4.0 / HEAD_DIM, dtype=torch.float32))
    coords_h = torch.arange(0.5, 1.0, dtype=torch.float32) / 1.0
    coords_w = torch.arange(0.5, 3.0, dtype=torch.float32) / 3.0
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    angles = 2.0 * torch.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)
    return torch.cos(angles).contiguous(), torch.sin(angles).contiguous()


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def source_block(parameters: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    hidden = build_input()
    rope_cos, rope_sin = build_rope()
    norm1 = F.layer_norm(
        hidden,
        (HIDDEN_SIZE,),
        parameters["layer.0.norm1.weight"],
        parameters["layer.0.norm1.bias"],
        EPS,
    )
    query = F.linear(norm1, parameters["layer.0.attention.q_proj.weight"], parameters["layer.0.attention.q_proj.bias"])
    key = F.linear(norm1, parameters["layer.0.attention.k_proj.weight"], None)
    value = F.linear(norm1, parameters["layer.0.attention.v_proj.weight"], parameters["layer.0.attention.v_proj.bias"])
    q_heads = query.view(1, TOKEN_COUNT, HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    k_heads = key.view(1, TOKEN_COUNT, HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    v_heads = value.view(1, TOKEN_COUNT, HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    q_prefix, q_patches = q_heads.split((1 + REGISTERS, PATCH_COUNT), dim=-2)
    k_prefix, k_patches = k_heads.split((1 + REGISTERS, PATCH_COUNT), dim=-2)
    q_rope = torch.cat((q_prefix, q_patches * rope_cos + rotate_half(q_patches) * rope_sin), dim=-2).contiguous()
    k_rope = torch.cat((k_prefix, k_patches * rope_cos + rotate_half(k_patches) * rope_sin), dim=-2).contiguous()
    scores = torch.matmul(q_rope, k_rope.transpose(2, 3)) * (HEAD_DIM**-0.5)
    probabilities = torch.softmax(scores, dim=-1)
    attention_context = torch.matmul(probabilities, v_heads).transpose(1, 2).contiguous()
    context_flat = attention_context.reshape(1, TOKEN_COUNT, HIDDEN_SIZE).contiguous()
    output_projection = F.linear(
        context_flat,
        parameters["layer.0.attention.o_proj.weight"],
        parameters["layer.0.attention.o_proj.bias"],
    )
    layer_scale1 = output_projection * parameters["layer.0.layer_scale1.lambda1"]
    first_residual = layer_scale1 + hidden
    norm2 = F.layer_norm(
        first_residual,
        (HIDDEN_SIZE,),
        parameters["layer.0.norm2.weight"],
        parameters["layer.0.norm2.bias"],
        EPS,
    )
    mlp_up = F.linear(
        norm2,
        parameters["layer.0.mlp.up_proj.weight"],
        parameters["layer.0.mlp.up_proj.bias"],
    )
    exact_gelu = F.gelu(mlp_up, approximate="none")
    mlp_down = F.linear(
        exact_gelu,
        parameters["layer.0.mlp.down_proj.weight"],
        parameters["layer.0.mlp.down_proj.bias"],
    )
    layer_scale2 = mlp_down * parameters["layer.0.layer_scale2.lambda1"]
    block_output = layer_scale2 + first_residual
    extractor_final = F.layer_norm(block_output.float(), (HIDDEN_SIZE,), eps=EPS)
    return {
        "input": hidden,
        "norm1": norm1,
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "q_rope": q_rope,
        "k_rope": k_rope,
        "scores": scores,
        "probabilities": probabilities,
        "attention_context": attention_context,
        "output_projection": output_projection,
        "layer_scale1": layer_scale1,
        "first_residual": first_residual,
        "norm2": norm2,
        "mlp_up": mlp_up,
        "exact_gelu": exact_gelu,
        "mlp_down": mlp_down,
        "layer_scale2": layer_scale2,
        "block_output": block_output,
        "extractor_final": extractor_final,
    }


def write_reference(checkpoint: Path, output: Path) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parameters = load_parameters(checkpoint)
    expected = source_block(parameters)
    output.mkdir(parents=True, exist_ok=True)

    parameter_bytes = b"".join(f32le_bytes(parameters[key]) for key in PARAMETER_KEYS)
    outputs: dict[str, dict[str, object]] = {}
    for name, tensor in expected.items():
        path = output / f"{name}.f32"
        path.write_bytes(f32le_bytes(tensor))
        outputs[name] = {
            "file": path.name,
            "shape": list(tensor.shape),
            "byte_length": tensor.numel() * 4,
            "f32le_sha256": f32le_sha256(tensor),
        }
    rope_cos, rope_sin = build_rope()
    multiply_adds = (
        4 * TOKEN_COUNT * HIDDEN_SIZE * HIDDEN_SIZE
        + 2 * TOKEN_COUNT * HIDDEN_SIZE * INTERMEDIATE_SIZE
        + 2 * TOKEN_COUNT * TOKEN_COUNT * HIDDEN_SIZE
    )
    metadata = {
        "schema": SCHEMA,
        "reference": {
            "kind": "independent source-formula torch CPU reference",
            "trellis_revision": TRELLIS_PIN,
            "transformers_modeling_sha256": TRANSFORMERS_MODELING_SHA256,
            "transformers_config_sha256": TRANSFORMERS_CONFIG_SHA256,
            "transformers_dinov3_runtime_available": False,
            "transformers_version": None,
            "device": "cpu",
            "network": "none",
        },
        "checkpoint": {
            "path_name": checkpoint.name,
            "byte_length": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "model": {
            "reference": MODEL_REFERENCE,
            "revision": MODEL_REVISION,
            "layer_index": 0,
            "token_count": TOKEN_COUNT,
            "patch_count": PATCH_COUNT,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_attention_heads": HEADS,
            "head_dim": HEAD_DIM,
            "num_register_tokens": REGISTERS,
            "parameter_bytes": len(parameter_bytes),
        },
        "parameter_order": list(PARAMETER_KEYS),
        "parameter_f32le_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        "inputs": {
            "input": {
                "shape": list(expected["input"].shape),
                "f32le_sha256": f32le_sha256(expected["input"]),
            },
            "rope_cos": {
                "shape": list(rope_cos.shape),
                "values": [float(value) for value in rope_cos.reshape(-1)],
                "f32le_sha256": f32le_sha256(rope_cos),
            },
            "rope_sin": {
                "shape": list(rope_sin.shape),
                "values": [float(value) for value in rope_sin.reshape(-1)],
                "f32le_sha256": f32le_sha256(rope_sin),
            },
        },
        "budgets": {
            "multiply_adds": multiply_adds,
            "max_score_elements": HEADS * TOKEN_COUNT * TOKEN_COUNT,
            "trace_bytes": sum(tensor.numel() for tensor in expected.values()) * 4,
        },
        "tolerance": {
            "absolute": 5.0e-4,
            "relative": 5.0e-4,
            "comparison": "bounded absolute/relative tolerance; not byte identity",
        },
        "outputs": outputs,
    }
    (output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_reference(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
