#!/usr/bin/env python3
"""Export a tiny, weight-free DINOv3 block CPU oracle.

This oracle is intentionally limited to one synthetic DINOv3 transformer block.
It executes the pinned Transformers implementation on CPU with hand-authored
float32 parameters and records every boundary needed by the Crystal port.  It
never downloads a checkpoint, reads a real model configuration, or selects an
accelerator device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from transformers.models.dinov3_vit import modeling_dinov3_vit as dino
from transformers.models.dinov3_vit import configuration_dinov3_vit as dino_config


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
MODEL_REFERENCE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MODEL_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
TRANSFORMERS_PIN = "5.8.1"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
TRANSFORMERS_MODELING_SHA256 = (
    "6073b7665eea50fb2260d86984011e6af8ed68cbd37ae57a6095e5e90e0eea34"
)
TRANSFORMERS_CONFIG_SHA256 = (
    "9a13d3c9ea8020aaed7057db28a7ffe0ffd9bb094a6178261bcc45ed04e9bbfc"
)

SCHEMA = "cogni-ml/trellis2/dino-v3-block-oracle/v1"
ABSOLUTE_TOLERANCE = 5.0e-5
RELATIVE_TOLERANCE = 5.0e-5
EXTRACTOR_EPS = 1.0e-5

CONFIG = {
    "image_size": 32,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 8,
    "intermediate_size": 12,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_register_tokens": 2,
    "rope_theta": 100.0,
    "layer_norm_eps": 1.0e-5,
    "hidden_act": "gelu",
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "layerscale_value": 1.0,
    "attention_dropout": 0.0,
    "drop_path_rate": 0.0,
    "use_gated_mlp": False,
    "attn_implementation": "eager",
    "training": False,
    "dtype": "float32",
    "device": "cpu",
}

PARAMETER_ORDER = (
    "norm1_weight",
    "norm1_bias",
    "q_weight",
    "q_bias",
    "k_weight",
    "v_weight",
    "v_bias",
    "o_weight",
    "o_bias",
    "layer_scale1",
    "norm2_weight",
    "norm2_bias",
    "up_weight",
    "up_bias",
    "down_weight",
    "down_bias",
    "layer_scale2",
)

EXPECTED_ORDER = (
    "input",
    "norm1",
    "q_heads",
    "k_heads",
    "v_heads",
    "q_rope",
    "k_rope",
    "scores",
    "probabilities",
    "attention_context",
    "output_projection",
    "layer_scale1",
    "first_residual",
    "norm2",
    "mlp_up",
    "exact_gelu",
    "mlp_down",
    "layer_scale2",
    "block_output",
    "extractor_final",
)


def f32le_bytes(tensor: torch.Tensor) -> bytes:
    values = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return values.tobytes(order="C")


def f32le_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(f32le_bytes(tensor)).hexdigest()


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_pins() -> dict[str, str]:
    """Reject an oracle run if its numerical/source inputs drift."""

    versions = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    expected = {
        "transformers": TRANSFORMERS_PIN,
        "torch": TORCH_PIN,
        "numpy": NUMPY_PIN,
    }
    if versions != expected:
        raise RuntimeError(f"pinned package drift: expected {expected}, got {versions}")

    source_hashes = {
        "transformers_modeling_sha256": source_sha256(Path(dino.__file__)),
        "transformers_config_sha256": source_sha256(Path(dino_config.__file__)),
    }
    expected_hashes = {
        "transformers_modeling_sha256": TRANSFORMERS_MODELING_SHA256,
        "transformers_config_sha256": TRANSFORMERS_CONFIG_SHA256,
    }
    if source_hashes != expected_hashes:
        raise RuntimeError(
            f"pinned Transformers source drift: expected {expected_hashes}, got {source_hashes}"
        )
    return source_hashes


def lattice(count: int, multiplier: int, offset: int, modulus: int, divisor: float) -> torch.Tensor:
    indices = np.arange(count, dtype=np.int64)
    values = ((indices * multiplier + offset) % modulus).astype(np.float32)
    values = (values - np.float32(modulus // 2)) / np.float32(divisor)
    return torch.from_numpy(values.copy())


def build_input() -> torch.Tensor:
    values = lattice(7 * 8, 17, 5, 43, 19.0)
    return values.reshape(1, 7, 8)


def build_parameters() -> dict[str, torch.Tensor]:
    # Explicit affine vectors make both LayerNorms and both layer scales
    # observably non-uniform rather than relying on random initialization.
    norm1_weight = torch.tensor(
        [0.73, 1.11, 0.89, 1.27, 0.97, 0.81, 1.19, 0.67], dtype=torch.float32
    )
    norm1_bias = torch.tensor(
        [-0.23, 0.17, 0.05, -0.31, 0.29, -0.07, 0.13, -0.19], dtype=torch.float32
    )
    norm2_weight = torch.tensor(
        [1.21, 0.79, 1.07, 0.93, 1.31, 0.69, 1.17, 0.87], dtype=torch.float32
    )
    norm2_bias = torch.tensor(
        [0.11, -0.21, 0.09, 0.27, -0.15, 0.03, -0.29, 0.19], dtype=torch.float32
    )
    layer_scale1 = torch.tensor(
        [0.37, 0.43, 0.59, 0.71, 0.83, 0.47, 0.65, 0.91], dtype=torch.float32
    )
    layer_scale2 = torch.tensor(
        [0.41, 0.53, 0.67, 0.79, 0.61, 0.89, 0.73, 0.97], dtype=torch.float32
    )
    return {
        "norm1_weight": norm1_weight,
        "norm1_bias": norm1_bias,
        "q_weight": lattice(8 * 8, 19, 2, 47, 113.0).reshape(8, 8),
        "q_bias": lattice(8, 11, 7, 31, 37.0),
        "k_weight": lattice(8 * 8, 23, 3, 53, 127.0).reshape(8, 8),
        "v_weight": lattice(8 * 8, 29, 5, 59, 109.0).reshape(8, 8),
        "v_bias": lattice(8, 13, 1, 29, 41.0),
        "o_weight": lattice(8 * 8, 31, 9, 61, 131.0).reshape(8, 8),
        "o_bias": lattice(8, 17, 4, 37, 43.0),
        "layer_scale1": layer_scale1,
        "norm2_weight": norm2_weight,
        "norm2_bias": norm2_bias,
        "up_weight": lattice(12 * 8, 37, 4, 67, 151.0).reshape(12, 8),
        "up_bias": lattice(12, 19, 3, 41, 47.0),
        "down_weight": lattice(8 * 12, 41, 8, 71, 139.0).reshape(8, 12),
        "down_bias": lattice(8, 23, 6, 43, 53.0),
        "layer_scale2": layer_scale2,
    }


def tensor_payload(tensor: torch.Tensor, probes: list[tuple[int, ...]] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "shape": list(tensor.shape),
        "values": [float(value) for value in tensor.detach().cpu().reshape(-1)],
        "f32le_sha256": f32le_sha256(tensor),
    }
    if probes is not None:
        payload["probes"] = [
            {"index": list(index), "value": float(tensor[index].item())}
            for index in probes
        ]
    return payload


def probe_indices(shape: Iterable[int]) -> list[tuple[int, ...]]:
    dims = tuple(int(value) for value in shape)
    if dims == (1, 7, 8):
        return [(0, 0, 0), (0, 2, 5), (0, 5, 1), (0, 6, 7)]
    if dims == (1, 2, 7, 4):
        return [(0, 0, 0, 0), (0, 1, 2, 3), (0, 0, 5, 1), (0, 1, 6, 2)]
    if dims == (1, 2, 7, 7):
        return [(0, 0, 0, 0), (0, 1, 2, 5), (0, 0, 5, 1), (0, 1, 6, 3)]
    if dims == (1, 7, 2, 4):
        return [(0, 0, 0, 0), (0, 2, 1, 3), (0, 5, 0, 1), (0, 6, 1, 2)]
    if dims == (1, 7, 12):
        return [(0, 0, 0), (0, 2, 7), (0, 5, 3), (0, 6, 11)]
    raise AssertionError(f"no probe recipe for shape {dims}")


def install_parameters(layer: torch.nn.Module, parameters: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        layer.norm1.weight.copy_(parameters["norm1_weight"])
        layer.norm1.bias.copy_(parameters["norm1_bias"])
        layer.attention.q_proj.weight.copy_(parameters["q_weight"])
        layer.attention.q_proj.bias.copy_(parameters["q_bias"])
        if layer.attention.k_proj.bias is not None:
            raise AssertionError("DINOv3 key projection unexpectedly has a bias")
        layer.attention.k_proj.weight.copy_(parameters["k_weight"])
        layer.attention.v_proj.weight.copy_(parameters["v_weight"])
        layer.attention.v_proj.bias.copy_(parameters["v_bias"])
        layer.attention.o_proj.weight.copy_(parameters["o_weight"])
        layer.attention.o_proj.bias.copy_(parameters["o_bias"])
        layer.layer_scale1.lambda1.copy_(parameters["layer_scale1"])
        layer.norm2.weight.copy_(parameters["norm2_weight"])
        layer.norm2.bias.copy_(parameters["norm2_bias"])
        layer.mlp.up_proj.weight.copy_(parameters["up_weight"])
        layer.mlp.up_proj.bias.copy_(parameters["up_bias"])
        layer.mlp.down_proj.weight.copy_(parameters["down_weight"])
        layer.mlp.down_proj.bias.copy_(parameters["down_bias"])
        layer.layer_scale2.lambda1.copy_(parameters["layer_scale2"])


def build_fixture() -> dict[str, object]:
    torch.set_num_threads(1)
    source_hashes = require_pins()

    config = DINOv3ViTConfig(**CONFIG)
    default_config_kwargs = {
        name: value
        for name, value in CONFIG.items()
        if name not in {"attn_implementation", "dtype", "device"}
    }
    default_config = DINOv3ViTConfig(**default_config_kwargs)
    default_model = DINOv3ViTModel(default_config).eval()
    if default_model.config._attn_implementation != "sdpa":
        raise AssertionError(
            "Transformers default attention backend drifted; expected sdpa contrast"
        )
    if config._attn_implementation != "eager":
        raise AssertionError("the oracle must use eager attention for named scores")
    if config.hidden_act != "gelu" or config.use_gated_mlp:
        raise AssertionError("the oracle requires the exact non-gated GELU MLP")
    model = DINOv3ViTModel(config).eval()

    # The object tree and serialized state-dict paths are intentionally checked
    # independently: this catches the common model.model.layer vs model.layer
    # context-bridge mistake before any numerical data is emitted.
    if hasattr(model, "layer"):
        raise AssertionError("DINOv3ViTModel unexpectedly exposes root .layer")
    if not hasattr(model, "model") or not hasattr(model.model, "layer"):
        raise AssertionError("DINOv3ViTModel.model.layer is missing")
    layer = model.model.layer[0]
    state_keys = tuple(model.state_dict().keys())
    layer_keys = tuple(key for key in state_keys if key.startswith("model.layer.0."))
    expected_layer_keys = tuple(
        f"model.layer.0.{suffix}" for suffix in layer.state_dict().keys()
    )
    if not layer_keys or set(layer_keys) != set(expected_layer_keys):
        raise AssertionError("state_dict layer keys do not start with model.layer.0")
    if any(key.startswith("layer.0.") for key in state_keys):
        raise AssertionError("state_dict contains an invalid root layer.0 path")

    parameters = build_parameters()
    install_parameters(layer, parameters)
    layer.eval()

    input_tensor = build_input()
    # Pixel values are shape-only here; their contents never enter the block.
    dummy_image = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        rope_cos, rope_sin = model.rope_embeddings(dummy_image)
        rope_cos = rope_cos.contiguous()
        rope_sin = rope_sin.contiguous()

        norm1 = layer.norm1(input_tensor)
        q = layer.attention.q_proj(norm1)
        k = layer.attention.k_proj(norm1)
        v = layer.attention.v_proj(norm1)
        batch, length, _ = input_tensor.shape
        q_heads = q.view(batch, length, config.num_attention_heads, -1).transpose(1, 2).contiguous()
        k_heads = k.view(batch, length, config.num_attention_heads, -1).transpose(1, 2).contiguous()
        v_heads = v.view(batch, length, config.num_attention_heads, -1).transpose(1, 2).contiguous()
        q_rope, k_rope = dino.apply_rotary_pos_emb(q_heads, k_heads, rope_cos, rope_sin)
        q_rope = q_rope.contiguous()
        k_rope = k_rope.contiguous()
        scores = torch.matmul(q_rope, k_rope.transpose(2, 3)) * layer.attention.scaling
        probabilities = torch.softmax(scores, dim=-1)
        attention_context = torch.matmul(probabilities, v_heads).transpose(1, 2).contiguous()
        output_projection = layer.attention.o_proj(
            attention_context.reshape(batch, length, -1).contiguous()
        )
        layer_scale1 = layer.layer_scale1(output_projection)
        first_residual = layer_scale1 + input_tensor
        norm2 = layer.norm2(first_residual)
        mlp_up = layer.mlp.up_proj(norm2)
        exact_gelu = F.gelu(mlp_up, approximate="none")
        mlp_down = layer.mlp.down_proj(exact_gelu)
        layer_scale2 = layer.layer_scale2(mlp_down)
        block_output = layer_scale2 + first_residual
        traced_output = layer(
            input_tensor,
            position_embeddings=(rope_cos, rope_sin),
        )
        if not torch.equal(block_output, traced_output):
            raise AssertionError("manual intermediates do not reproduce DINOv3ViTLayer")
        extractor_final = F.layer_norm(
            block_output.float(), (config.hidden_size,), eps=EXTRACTOR_EPS
        )

    expected = {
        "input": input_tensor,
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
    if tuple(expected) != EXPECTED_ORDER:
        raise AssertionError("expected boundary order drift")

    parameter_bytes = b"".join(f32le_bytes(parameters[name]) for name in PARAMETER_ORDER)
    parameter_sha = hashlib.sha256(parameter_bytes).hexdigest()
    input_sha = f32le_sha256(input_tensor)
    fixture: dict[str, object] = {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "extractor": f"https://github.com/microsoft/TRELLIS.2/blob/{TRELLIS_PIN}/trellis2/modules/image_feature_extractor.py",
            "generator": "tools/trellis2_oracle/export_dino_v3_block.py",
            "model_reference": MODEL_REFERENCE,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            **source_hashes,
            "transformers_license": "Apache-2.0",
            "transformers_license_url": "https://github.com/huggingface/transformers/blob/v5.8.1/LICENSE",
            "oracle_kind": "synthetic DINOv3 one-block CPU execution",
            "weights": "synthetic only; no checkpoint",
            "network": "none",
            "device": "cpu",
            "real_model_config_status": "unavailable/gated/not_used",
            "real_model_config": "gated model config was not loaded or used",
        },
        "object_path": {
            "root_layer_attribute_present": False,
            "root_model_layer_attribute_present": True,
            "state_dict_layer_prefix": "model.layer.0",
            "state_dict_layer_key_count": len(layer_keys),
        },
        "attention_backend": {
            "transformers_default": default_model.config._attn_implementation,
            "oracle": config._attn_implementation,
            "reason": "eager is required to capture scores and probabilities; SDPA is not byte-compared",
        },
        "config": CONFIG,
        "tolerance": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": RELATIVE_TOLERANCE,
            "extractor_final_layer_norm_eps": EXTRACTOR_EPS,
        },
        "inputs": {
            "input": {
                **tensor_payload(input_tensor, probe_indices(input_tensor.shape)),
                "recipe": "x[i]=(((17*i+5)%43)-21)/19 for row-major i over [1,7,8]",
            },
            "rope_cos": tensor_payload(rope_cos, [(0, 0), (1, 1), (2, 2), (3, 3)]),
            "rope_sin": tensor_payload(rope_sin, [(0, 0), (1, 1), (2, 2), (3, 3)]),
            "dummy_image": {
                "shape": list(dummy_image.shape),
                "f32le_sha256": f32le_sha256(dummy_image),
                "recipe": "all-zero shape-only [1,3,32,32]; pixels are not consumed by the block",
            },
        },
        "parameters": {
            "order": list(PARAMETER_ORDER),
            "f32le_sha256": parameter_sha,
            "tensors": {
                name: tensor_payload(parameters[name]) for name in PARAMETER_ORDER
            },
        },
        "expected_order": list(EXPECTED_ORDER),
        "expected": {
            name: tensor_payload(value, probe_indices(value.shape))
            for name, value in expected.items()
        },
        "context_bridge": {
            "root_path": ".layer absent",
            "encoder_path": ".model.layer present",
            "state_dict_prefix": "model.layer.0",
        },
    }
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    fixture = build_fixture()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
