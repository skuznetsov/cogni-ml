#!/usr/bin/env python3
"""Export a bounded, weight-free CPU oracle for DINOv3 embeddings and RoPE.

The oracle instantiates only tiny synthetic DINOv3 embedding parameters. It
does not access the network, load a checkpoint, execute transformer blocks, or
select an accelerator.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import DINOv3ViTConfig
from transformers import DINOv3ViTModel
from transformers.models.dinov3_vit import modeling_dinov3_vit as dino


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SCHEMA = "cogni-ml/trellis2/dino-v3-embeddings-oracle/v1"
MODEL_REFERENCE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TRANSFORMERS_PIN = "5.8.1"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
TRANSFORMERS_MODELING_SHA256 = "6073b7665eea50fb2260d86984011e6af8ed68cbd37ae57a6095e5e90e0eea34"
TRANSFORMERS_CONFIG_SHA256 = "9a13d3c9ea8020aaed7057db28a7ffe0ffd9bb094a6178261bcc45ed04e9bbfc"
PATCH_SIZE = 16
CHANNELS = 3
HIDDEN_SIZE = 16
HEADS = 2
REGISTER_TOKENS = 4
ROPE_THETA = 100.0


def f32le_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )


def f32le_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(f32le_bytes(tensor)).hexdigest()


def truncated_1e4le_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    quantized = np.trunc(values * 10_000.0).astype("<i4")
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe_input(edge: int) -> torch.Tensor:
    channels = torch.arange(CHANNELS, dtype=torch.int64)[None, :, None, None]
    rows = torch.arange(edge, dtype=torch.int64)[None, None, :, None]
    columns = torch.arange(edge, dtype=torch.int64)[None, None, None, :]
    values = ((channels * 5 + rows * 3 + columns * 7) % 33) - 16
    return values.to(torch.float32) / 16.0


def recipe_parameters() -> dict[str, torch.Tensor]:
    outputs = torch.arange(HIDDEN_SIZE, dtype=torch.int64)[:, None, None, None]
    channels = torch.arange(CHANNELS, dtype=torch.int64)[None, :, None, None]
    rows = torch.arange(PATCH_SIZE, dtype=torch.int64)[None, None, :, None]
    columns = torch.arange(PATCH_SIZE, dtype=torch.int64)[None, None, None, :]
    weight = ((outputs * 11 + channels * 7 + rows * 5 + columns * 3) % 17) - 8
    bias = torch.arange(HIDDEN_SIZE, dtype=torch.float32) - 8
    cls = torch.arange(HIDDEN_SIZE, dtype=torch.float32) - 8
    register = torch.arange(
        REGISTER_TOKENS * HIDDEN_SIZE, dtype=torch.float32
    ).reshape(REGISTER_TOKENS, HIDDEN_SIZE)
    return {
        "patch_weight": weight.to(torch.float32) / 64.0,
        "patch_bias": bias / 32.0,
        "cls_token": cls / 16.0,
        "register_tokens": (register - 32.0) / 32.0,
    }


def lcg_f32(count: int, seed: int) -> torch.Tensor:
    state = seed
    values = np.empty(count, dtype=np.float32)
    for index in range(count):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        signed = ((state >> 8) % 2001) - 1000
        values[index] = np.float32(signed) / np.float32(997.0)
    return torch.from_numpy(values)


def probe(tensor: torch.Tensor, indices: tuple[tuple[int, ...], ...]) -> list[dict[str, object]]:
    return [
        {"index": list(index), "value": float(tensor[index].item())}
        for index in indices
    ]


def export_case(edge: int, parameters: dict[str, torch.Tensor]) -> dict[str, object]:
    config = DINOv3ViTConfig(
        image_size=edge,
        patch_size=PATCH_SIZE,
        num_channels=CHANNELS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=HEADS,
        num_register_tokens=REGISTER_TOKENS,
        rope_theta=ROPE_THETA,
        pos_embed_shift=None,
        pos_embed_jitter=None,
        pos_embed_rescale=2.0,
    )
    image = recipe_input(edge)
    source_before = f32le_sha256(image)

    embeddings_module = dino.DINOv3ViTEmbeddings(config).eval()
    rope_module = dino.DINOv3ViTRopePositionEmbedding(config).eval()
    with torch.no_grad():
        embeddings_module.patch_embeddings.weight.copy_(parameters["patch_weight"])
        embeddings_module.patch_embeddings.bias.copy_(parameters["patch_bias"])
        embeddings_module.cls_token.copy_(parameters["cls_token"].reshape(1, 1, -1))
        embeddings_module.register_tokens.copy_(
            parameters["register_tokens"].reshape(1, REGISTER_TOKENS, -1)
        )
        patches = (
            embeddings_module.patch_embeddings(image)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )
        embeddings = embeddings_module(image).contiguous()
        patch_edge = edge // PATCH_SIZE
        coordinates = dino.get_patches_center_coordinates(
            patch_edge, patch_edge, dtype=torch.float32, device=image.device
        ).contiguous()
        rope_cos, rope_sin = rope_module(image)
        rope_cos = rope_cos.contiguous()
        rope_sin = rope_sin.contiguous()

    assert source_before == f32le_sha256(image)
    patch_count = patch_edge * patch_edge
    middle_patch = patch_count // 2
    return {
        "name": f"synthetic_{edge}",
        "input": {
            "shape": list(image.shape),
            "recipe": "x[b,c,y,x]=(((5*c+3*y+7*x)%33)-16)/16",
            "f32le_sha256": source_before,
        },
        "expected": {
            "patches": {
                "shape": list(patches.shape),
                "f32le_sha256": f32le_sha256(patches),
                "probes": probe(
                    patches,
                    ((0, 0, 0), (0, 1, 3), (0, middle_patch, 7), (0, patch_count - 1, 15)),
                ),
            },
            "embeddings": {
                "shape": list(embeddings.shape),
                "f32le_sha256": f32le_sha256(embeddings),
                "probes": probe(
                    embeddings,
                    ((0, 0, 0), (0, 1, 0), (0, REGISTER_TOKENS, 15), (0, REGISTER_TOKENS + 1, 0)),
                ),
            },
            "coordinates": {
                "shape": list(coordinates.shape),
                "f32le_sha256": f32le_sha256(coordinates),
                "probes": probe(
                    coordinates,
                    ((0, 0), (0, 1), (middle_patch, 0), (patch_count - 1, 1)),
                ),
            },
            "rope_cos": {
                "shape": list(rope_cos.shape),
                "truncated_1e4le_sha256": truncated_1e4le_sha256(rope_cos),
                "probes": probe(
                    rope_cos,
                    ((0, 0), (0, 1), (middle_patch, 4), (patch_count - 1, 7)),
                ),
            },
            "rope_sin": {
                "shape": list(rope_sin.shape),
                "truncated_1e4le_sha256": truncated_1e4le_sha256(rope_sin),
                "probes": probe(
                    rope_sin,
                    ((0, 0), (0, 1), (middle_patch, 4), (patch_count - 1, 7)),
                ),
            },
        },
    }


def export_float_stress_case() -> dict[str, object]:
    edge = 512
    hidden = 4
    heads = 1
    registers = 2
    config = DINOv3ViTConfig(
        image_size=edge,
        patch_size=PATCH_SIZE,
        num_channels=CHANNELS,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_register_tokens=registers,
        rope_theta=ROPE_THETA,
        pos_embed_shift=None,
        pos_embed_jitter=None,
        pos_embed_rescale=2.0,
    )
    image = lcg_f32(CHANNELS * edge * edge, 0x13579BDF).reshape(
        1, CHANNELS, edge, edge
    )
    patch_weight = lcg_f32(
        hidden * CHANNELS * PATCH_SIZE * PATCH_SIZE, 0x2468ACE0
    ).reshape(hidden, CHANNELS, PATCH_SIZE, PATCH_SIZE)
    patch_bias = lcg_f32(hidden, 0x10203040)
    cls_token = lcg_f32(hidden, 0x50607080).reshape(1, 1, hidden)
    register_tokens = lcg_f32(registers * hidden, 0x90ABCDEF).reshape(
        1, registers, hidden
    )
    module = dino.DINOv3ViTEmbeddings(config).eval()
    with torch.no_grad():
        module.patch_embeddings.weight.copy_(patch_weight)
        module.patch_embeddings.bias.copy_(patch_bias)
        module.cls_token.copy_(cls_token)
        module.register_tokens.copy_(register_tokens)
        patches = module.patch_embeddings(image).flatten(2).transpose(1, 2).contiguous()
        embeddings = module(image).contiguous()

    parameter_bytes = b"".join(
        f32le_bytes(value)
        for value in (patch_weight, patch_bias, cls_token, register_tokens)
    )
    return {
        "name": "non_binary_conv2d_512",
        "config": {
            "patch_size": PATCH_SIZE,
            "num_channels": CHANNELS,
            "hidden_size": hidden,
            "num_attention_heads": heads,
            "num_register_tokens": registers,
            "rope_theta": ROPE_THETA,
        },
        "input": {
            "shape": list(image.shape),
            "seed": "0x13579bdf",
            "f32le_sha256": f32le_sha256(image),
        },
        "parameters": {
            "seeds": ["0x2468ace0", "0x10203040", "0x50607080", "0x90abcdef"],
            "f32le_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        },
        "expected": {
            "absolute_tolerance": 5.0e-5,
            "patches": {
                "shape": list(patches.shape),
                "probes": probe(
                    patches,
                    (
                        (0, 0, 0),
                        (0, 0, 3),
                        (0, 1, 1),
                        (0, 31, 2),
                        (0, 32, 3),
                        (0, 255, 0),
                        (0, 511, 1),
                        (0, 1023, 3),
                    ),
                ),
            },
            "embeddings": {
                "shape": list(embeddings.shape),
                "probes": probe(
                    embeddings,
                    ((0, 0, 0), (0, 1, 0), (0, 2, 3), (0, 3, 0), (0, 4, 1), (0, 1026, 3)),
                ),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)

    modeling_path = Path(dino.__file__).resolve()
    config_module = __import__(DINOv3ViTConfig.__module__, fromlist=["unused"])
    resolved_config_path = Path(config_module.__file__).resolve()
    modeling_sha256 = source_sha256(modeling_path)
    config_sha256 = source_sha256(resolved_config_path)
    if transformers.__version__ != TRANSFORMERS_PIN:
        raise RuntimeError(
            f"oracle requires Transformers {TRANSFORMERS_PIN}, "
            f"found {transformers.__version__}"
        )
    if torch.__version__ != TORCH_PIN:
        raise RuntimeError(f"oracle requires PyTorch {TORCH_PIN}, found {torch.__version__}")
    if np.__version__ != NUMPY_PIN:
        raise RuntimeError(f"oracle requires NumPy {NUMPY_PIN}, found {np.__version__}")
    if modeling_sha256 != TRANSFORMERS_MODELING_SHA256:
        raise RuntimeError(
            "DINOv3 modeling source digest drift: "
            f"expected {TRANSFORMERS_MODELING_SHA256}, found {modeling_sha256}"
        )
    if config_sha256 != TRANSFORMERS_CONFIG_SHA256:
        raise RuntimeError(
            "DINOv3 config source digest drift: "
            f"expected {TRANSFORMERS_CONFIG_SHA256}, found {config_sha256}"
        )
    parameters = recipe_parameters()
    parameter_bytes = b"".join(
        f32le_bytes(parameters[name])
        for name in ("patch_weight", "patch_bias", "cls_token", "register_tokens")
    )
    compatibility_model = DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=32,
            patch_size=16,
            num_channels=3,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_register_tokens=1,
        )
    )
    upstream_layer_path_available = hasattr(compatibility_model, "layer")
    local_layer_path_available = hasattr(compatibility_model.model, "layer")

    fixture = {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "extractor": f"https://github.com/microsoft/TRELLIS.2/blob/{TRELLIS_PIN}/trellis2/modules/image_feature_extractor.py",
            "model_reference": MODEL_REFERENCE,
            "transformers_version": TRANSFORMERS_PIN,
            "transformers_modeling_sha256": TRANSFORMERS_MODELING_SHA256,
            "transformers_config_sha256": TRANSFORMERS_CONFIG_SHA256,
            "transformers_license": "Apache-2.0",
            "transformers_license_url": "https://github.com/huggingface/transformers/blob/v5.8.1/LICENSE",
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "generator": "tools/trellis2_oracle/export_dino_v3_embeddings.py",
            "oracle_kind": "synthetic Transformers DINOv3 embedding and RoPE CPU execution",
            "weights": "synthetic only; no checkpoint",
            "device": "cpu",
            "network": "none",
            "real_model_config": "unadmitted; gated checkpoint config was not loaded",
        },
        "compatibility": {
            "pinned_upstream_layer_path": "model.layer",
            "transformers_5_8_1_layer_path": "model.model.layer",
            "pinned_upstream_path_available_on_instance": upstream_layer_path_available,
            "transformers_5_8_1_path_available_on_instance": local_layer_path_available,
            "status": "blocked for full encoder; embeddings and RoPE paths remain valid",
        },
        "synthetic_config": {
            "patch_size": PATCH_SIZE,
            "num_channels": CHANNELS,
            "hidden_size": HIDDEN_SIZE,
            "num_attention_heads": HEADS,
            "head_dim": HIDDEN_SIZE // HEADS,
            "num_register_tokens": REGISTER_TOKENS,
            "rope_theta": ROPE_THETA,
            "training": False,
        },
        "parameters": {
            "recipe": "power-of-two rational dense patch kernel plus explicit prefix tokens",
            "order": ["patch_weight", "patch_bias", "cls_token", "register_tokens"],
            "f32le_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        },
        "scope": {
            "includes": [
                "CPU Conv2d patch projection",
                "row-major patch flattening",
                "CLS and register token concatenation",
                "eval-mode dynamic patch-center coordinates",
                "DINOv3 RoPE cos and sin",
            ],
            "excludes": [
                "mask token replacement",
                "attention RoPE application",
                "transformer blocks",
                "real model configuration or weights",
                "BF16/F16",
                "Metal",
            ],
        },
        "cases": [export_case(512, parameters), export_case(1024, parameters)],
        "float_stress_case": export_float_stress_case(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output} (transformers={transformers.__version__}, "
        "weights=synthetic-only, device=cpu, network=none)"
    )


if __name__ == "__main__":
    main()
