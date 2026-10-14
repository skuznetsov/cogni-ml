#!/usr/bin/env python3
"""Check one actual quantized token embedding against the native inspection row.

Only the selected packed-weight, scale, and bias rows are read from the model
shard. The lookup uses MLX's upstream ``mx.dequantize`` operation, which is the
operation called by ``mlx.nn.QuantizedEmbedding.__call__``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import pathlib
import platform
import sys
from typing import Any

import numpy as np


MLX_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
TRACE_FORMAT = "drama3-reference-traces-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def run_probe(
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
    inspect_path: pathlib.Path,
    token_id: int,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    from safetensors import __version__ as safetensors_version
    from safetensors import safe_open

    actual_revision = __import__("subprocess").check_output(
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != MLX_REVISION:
        raise RuntimeError(f"MLX-audio source is {actual_revision}; expected {MLX_REVISION}")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    global_quantization = config["quantization"]
    embedding_quantization = global_quantization.get("backbone.embed_tokens", global_quantization)
    group_size = int(embedding_quantization["group_size"])
    bits = int(embedding_quantization["bits"])
    mode = str(global_quantization.get("mode", "affine"))
    if group_size <= 0 or bits <= 0 or 32 % bits:
        raise ValueError(f"unsupported embedding quantization: group_size={group_size}, bits={bits}")
    weight_map = index["weight_map"]
    keys = {
        "weight": "backbone.embed_tokens.weight",
        "scales": "backbone.embed_tokens.scales",
        "biases": "backbone.embed_tokens.biases",
    }
    if any(key not in weight_map for key in keys.values()):
        raise KeyError("checkpoint index lacks packed backbone.embed_tokens weights/scales/biases")
    shards = {weight_map[key] for key in keys.values()}
    if len(shards) != 1:
        raise ValueError(f"embedding quantization tensors span different shards: {sorted(shards)}")
    shard = next(iter(shards))
    vocab_size = int(config["text_config"]["vocab_size"])
    hidden_size = int(config["text_config"]["hidden_size"])
    if not 0 <= token_id < vocab_size:
        raise ValueError(f"token id {token_id} is outside vocabulary size {vocab_size}")

    slices: dict[str, np.ndarray] = {}
    slice_hashes: dict[str, str] = {}
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as tensors:
        for kind, key in keys.items():
            row = tensors.get_slice(key)[token_id]
            if kind == "weight":
                if row.dtype != torch.uint32:
                    raise TypeError(f"{key} row has dtype {row.dtype}, expected uint32")
                value = row.numpy()
                expected_shape = (hidden_size // (32 // bits),)
            else:
                if row.dtype != torch.bfloat16:
                    raise TypeError(f"{key} row has dtype {row.dtype}, expected bfloat16")
                value = row.float().numpy()
                expected_shape = (hidden_size // group_size,)
            if value.shape != expected_shape:
                raise ValueError(f"{key} row shape {value.shape}; expected {expected_shape}")
            slices[kind] = np.ascontiguousarray(value)
            slice_hashes[key] = sha256_array(value)

    sys.path.insert(0, str(mlx_root))
    from mlx_audio.lm.models.qwen3_5 import Qwen3_5TextModel  # noqa: F401

    # Exercise the upstream quantized embedding call itself using a one-row
    # view of the vocabulary. The pinned backbone replay promotes checkpoint
    # BF16 scales/biases to F32 on load, preserving their exact values.
    embedding = nn.QuantizedEmbedding(1, hidden_size, group_size, bits, mode)
    embedding.weight = mx.array(slices["weight"][None, :], dtype=mx.uint32)
    embedding.scales = mx.array(slices["scales"][None, :], dtype=mx.float32)
    embedding.biases = mx.array(slices["biases"][None, :], dtype=mx.float32)
    decoded = embedding(mx.array([0], dtype=mx.int32))
    mx.eval(decoded)
    reference_values = np.asarray(decoded, dtype=np.float32).reshape(-1)
    if reference_values.shape != (hidden_size,) or not np.isfinite(reference_values).all():
        raise FloatingPointError("MLX embedding row has invalid shape or non-finite values")

    inspected = json.loads(inspect_path.read_text(encoding="utf-8"))
    ids = inspected.get("ids")
    expected = inspected.get("embedding")
    if not isinstance(ids, list) or not ids or int(ids[0]) != token_id:
        raise ValueError(f"inspection input does not begin with token id {token_id}")
    expected_values = np.asarray(expected, dtype=np.float32)
    if expected_values.shape != (hidden_size,) or not np.isfinite(expected_values).all():
        raise ValueError("inspection embedding has invalid shape or non-finite values")

    name = f"backbone.embed_tokens.token{token_id}"
    reference_bundle = {
        "format": TRACE_FORMAT,
        "traces": [{
            "name": name,
            "shape": [hidden_size],
            "dtype": "float32",
            "values": reference_values.tolist(),
        }],
    }
    expected_path = output_path.with_suffix(output_path.suffix + ".expected.json")
    expected_bundle = {
        "format": TRACE_FORMAT,
        "traces": [{
            "name": name,
            "shape": [hidden_size],
            "dtype": "float32",
            "values": expected_values.tolist(),
        }],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reference_bundle, separators=(",", ":")) + "\n", encoding="utf-8")
    expected_path.write_text(json.dumps(expected_bundle, separators=(",", ":")) + "\n", encoding="utf-8")

    mlx_package_file = pathlib.Path(mx.__file__).resolve()
    embedding_source = pathlib.Path(
        importlib.import_module(nn.QuantizedEmbedding.__module__).__file__
    ).resolve()
    qwen_source = mlx_root / "mlx_audio/lm/models/qwen3_5.py"
    manifest = {
        "kind": "actual_pinned_q4_embedding_row_probe",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "upstream": {
            "mlx-audio": {"repository": "https://github.com/cguldogan/mlx-audio", "revision": MLX_REVISION}
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "safetensors": safetensors_version,
        },
        "operation": "mlx.nn.QuantizedEmbedding.__call__ on a one-row view of the selected vocabulary entry",
        "token_id": token_id,
        "selected_row": 0,
        "quantization": {
            "mode": mode,
            "bits": bits,
            "group_size": group_size,
            "scale_bias_input_dtype": "float32",
            "output_dtype": "float32",
        },
        "inputs": {"path": inspect_path.name, "sha256": sha256_file(inspect_path)},
        "model_files": {
            "config.json": sha256_file(config_path),
            "model.safetensors.index.json": sha256_file(index_path),
        },
        "shard": shard,
        "shard_sha256_recorded": False,
        "selected_tensor_row_sha256": slice_hashes,
        "reference_files": {
            "qwen3_5.py": {"path": qwen_source.relative_to(mlx_root).as_posix(), "sha256": sha256_file(qwen_source)},
            "mlx_quantized_embedding.py": {"path": str(embedding_source), "sha256": sha256_file(embedding_source)},
            "mlx_core_module": {"path": str(mlx_package_file), "sha256": sha256_file(mlx_package_file)},
        },
        "trace": {"path": output_path.name, "sha256": sha256_file(output_path)},
        "expected": {"path": expected_path.name, "sha256": sha256_file(expected_path)},
        "trace_name": name,
        "hidden_size": hidden_size,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--inspect", type=pathlib.Path, required=True, help="native_inspect.json for the selected token")
    parser.add_argument("--token-id", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    for field in ("mlx_root", "model_dir", "inspect", "output"):
        setattr(args, field, getattr(args, field).resolve())
    manifest = run_probe(args.mlx_root, args.model_dir, args.inspect, args.token_id, args.output)
    print(json.dumps({"trace": str(args.output), "expected": str(args.output) + ".expected.json", "manifest": str(args.output) + ".manifest.json", "slice_keys": len(manifest["selected_tensor_row_sha256"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
