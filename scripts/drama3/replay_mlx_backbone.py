#!/usr/bin/env python3
"""Replay selected Q4 backbone tensors through the pinned upstream MLX model.

The checkpoint is sharded with codec and tower tensors interleaved. This
runner reads only indexed ``backbone.*`` tensors through safetensors' PyTorch
reader into MLX arrays; it does not call the upstream all-shard loader or materialize a dense
copy of the checkpoint. The source checkout and checkpoint revision are
explicitly pinned below and recorded with SHA-256 hashes in a sidecar manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch


MLX_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
TRACE_FORMAT = "drama3-reference-traces-v1"
DENSE_INPUT_FORMAT = "drama3-dense-projection-inputs-v1"
DENSE_REFERENCE_FORMAT = "drama3-dense-projection-reference-v1"
DENSE_PROJECTION = "backbone.layers.0.linear_attn"
DENSE_PROJECTION_NAMES = ("qkv", "z", "b", "a")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_backbone_map(weight_map: dict[str, str]) -> dict[str, str]:
    selected = {name: shard for name, shard in weight_map.items() if name.startswith("backbone.")}
    if not selected:
        raise ValueError("checkpoint index contains no backbone.* tensors")
    return selected


def record(traces: list[dict[str, Any]], name: str, value: Any, width: int) -> None:
    data = float32_values(value, name, width)
    traces.append(
        {"name": name, "shape": [width], "dtype": "float32", "values": data.tolist()}
    )


def float32_values(value: Any, name: str, width: int) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    if data.size != width:
        raise ValueError(f"{name}: expected {width} values, found {data.size}")
    if not np.isfinite(data).all():
        raise FloatingPointError(f"{name}: non-finite reference activation")
    return data


def load_selected_weights(
    model: Any,
    model_dir: pathlib.Path,
    selected: dict[str, str],
    mx: Any,
    hash_prefix: str | None = None,
) -> dict[str, Any] | None:
    """Load only backbone entries, one shard at a time, into the quantized model.

    The installed safetensors MLX reader rejects this checkpoint's BF16 dtype.
    Read through the official PyTorch safetensors reader, promote floating
    values to float32, and preserve packed integer quantized tensors verbatim.
    """
    from safetensors import safe_open

    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)

    loaded = 0
    tensor_digest = hashlib.sha256() if hash_prefix is not None else None
    tensor_names: list[str] = []
    tensor_dtypes: dict[str, str] = {}
    tensor_shapes: dict[str, list[int]] = {}
    for shard in sorted(by_shard):
        shard_path = model_dir / shard
        names = sorted(by_shard[shard])
        with safe_open(str(shard_path), framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            missing = [name for name in names if name not in available]
            if missing:
                raise ValueError(f"{shard}: indexed backbone tensors missing: {missing[:5]}")
            tensors_for_model = []
            for name in names:
                tensor = tensors.get_tensor(name)
                if hash_prefix is not None and name.startswith(hash_prefix):
                    raw = tensor.contiguous().view(torch.uint8).numpy()
                    tensor_digest.update(name.encode("utf-8") + b"\0")
                    tensor_digest.update(str(tensor.dtype).encode("ascii") + b"\0")
                    tensor_digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
                    tensor_digest.update(raw)
                    tensor_names.append(name)
                    tensor_dtypes[name] = str(tensor.dtype).removeprefix("torch.")
                    tensor_shapes[name] = [int(dimension) for dimension in tensor.shape]
                values = (
                    tensor.to(dtype=torch.float32).numpy()
                    if tensor.is_floating_point()
                    else tensor.numpy()
                )
                tensors_for_model.append((name, mx.array(values)))
                del tensor, values
        model.load_weights(tensors_for_model, strict=False)
        loaded += len(tensors_for_model)
        print(f"loaded {loaded}/{len(selected)} backbone tensors from {shard}", file=sys.stderr)
        del tensors_for_model
        mx.clear_cache()

    if loaded != len(selected):
        raise AssertionError(f"loaded {loaded} tensors, expected {len(selected)}")
    if hash_prefix is None:
        return None
    expected_names = {
        f"{DENSE_PROJECTION}.in_proj_{name}.weight"
        for name in DENSE_PROJECTION_NAMES
    }
    if set(tensor_names) != expected_names:
        raise ValueError(
            "layer-0 dense projection tensor set differs from the expected four weights: "
            f"{sorted(tensor_names)}"
        )
    return {
        "sha256": tensor_digest.hexdigest(),
        "tensor_names": sorted(tensor_names),
        "tensor_dtypes": tensor_dtypes,
        "tensor_shapes": tensor_shapes,
    }


def run_replay(
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
    inputs_path: pathlib.Path,
    output_path: pathlib.Path,
    hash_shards: bool = False,
    dense_fixture_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    from safetensors import __version__ as safetensors_version

    sys.path.insert(0, str(mlx_root))
    from mlx_audio.lm.models import qwen3_5
    from mlx_audio.lm.models.qwen3_5 import Qwen3_5TextModel, TextModelArgs
    from mlx_audio.utils import apply_quantization

    actual_revision = subprocess.check_output(
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != MLX_REVISION:
        raise RuntimeError(f"MLX-audio source is {actual_revision}; expected {MLX_REVISION}")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    selected = selected_backbone_map(index["weight_map"])
    text_config = config["text_config"]
    args = TextModelArgs.from_dict(text_config)
    layer_types = text_config.get("layer_types")
    if layer_types is not None:
        expected_types = [
            "full_attention" if (i + 1) % args.full_attention_interval == 0 else "linear_attention"
            for i in range(args.num_hidden_layers)
        ]
        if list(layer_types) != expected_types:
            raise ValueError("checkpoint layer_types do not match pinned Qwen3.5 interval")

    class BackboneOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Qwen3_5TextModel(args)

    model = BackboneOnly()
    # The official MLX loader uses this same quantizer before loading tensors.
    # The wrapper retains the checkpoint's `backbone.*` parameter prefix while
    # avoiding allocation of the unrelated tower and codec.
    apply_quantization(model, config, selected)
    dense_tensor_hash = load_selected_weights(
        model,
        model_dir,
        selected,
        mx,
        hash_prefix=f"{DENSE_PROJECTION}.in_proj_" if dense_fixture_path else None,
    )
    model.eval()

    input_bundle = json.loads(inputs_path.read_text(encoding="utf-8"))
    embeddings = np.asarray(input_bundle["input_embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != args.hidden_size:
        raise ValueError(
            f"input_embeddings must have shape [rows, {args.hidden_size}], got {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("input_embeddings contain non-finite values")

    original_layer_call = qwen3_5.DecoderLayer.__call__
    layer_ids = {id(layer): i for i, layer in enumerate(model.backbone.layers)}
    row_index = [0]
    traces: list[dict[str, Any]] = []
    dense_inputs: list[dict[str, Any]] = []
    dense_reference_rows: list[dict[str, Any]] = []

    def traced_layer(self: Any, x: Any, mask: Any = None, cache: Any = None) -> Any:
        layer = layer_ids[id(self)]
        normalized = self.input_layernorm(x)
        if dense_fixture_path is not None and layer == 0:
            if not self.is_linear:
                raise ValueError("layer 0 is not a linear-attention layer")
            row = row_index[0]
            dense_inputs.append(
                {
                    "row": row,
                    "values": float32_values(
                        normalized, f"dense row {row} input", args.hidden_size
                    ).tolist(),
                }
            )
            projections = {
                "qkv": self.linear_attn.in_proj_qkv(normalized),
                "z": self.linear_attn.in_proj_z(normalized),
                "b": self.linear_attn.in_proj_b(normalized),
                "a": self.linear_attn.in_proj_a(normalized),
            }
            mx.eval(*projections.values())
            dense_reference_rows.append(
                {
                    "row": row,
                    "projections": {
                        name: {
                            "shape": [int(value.size)],
                            "dtype": "float32",
                            "values": float32_values(
                                value,
                                f"dense row {row} {name} reference",
                                int(value.size),
                            ).tolist(),
                        }
                        for name, value in projections.items()
                    },
                }
            )
        if self.is_linear:
            residual = self.linear_attn(normalized, mask, cache)
        else:
            residual = self.self_attn(normalized, mask, cache)
        attention_residual = x + residual
        row = row_index[0]
        record(
            traces,
            f"backbone.row{row}.layers.{layer}.attention_residual",
            attention_residual,
            args.hidden_size,
        )
        hidden = attention_residual + self.mlp(
            self.post_attention_layernorm(attention_residual)
        )
        record(
            traces,
            f"backbone.row{row}.layers.{layer}.hidden",
            hidden,
            args.hidden_size,
        )
        return hidden

    qwen3_5.DecoderLayer.__call__ = traced_layer
    cache = model.backbone.make_cache()
    try:
        for row, embedding in enumerate(embeddings):
            row_index[0] = row
            output = model.backbone(
                inputs=mx.zeros((1, 1), dtype=mx.int32),
                cache=cache,
                input_embeddings=mx.array(embedding[None, None, :]),
            )
            mx.eval(output)
            record(traces, f"backbone.row{row}.hidden", output, args.hidden_size)
            print(f"traced row {row + 1}/{len(embeddings)}", file=sys.stderr)
    finally:
        qwen3_5.DecoderLayer.__call__ = original_layer_call

    traces.sort(key=lambda item: item["name"])
    expected_trace_count = len(embeddings) * (2 * args.num_hidden_layers + 1)
    if len(traces) != expected_trace_count:
        raise AssertionError(f"emitted {len(traces)} traces, expected {expected_trace_count}")
    if dense_fixture_path is not None and (
        len(dense_inputs) != len(embeddings)
        or len(dense_reference_rows) != len(embeddings)
    ):
        raise AssertionError(
            "dense projection capture did not produce one input/reference record per row"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"format": TRACE_FORMAT, "traces": traces}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    dense_paths: dict[str, pathlib.Path] = {}
    if dense_fixture_path is not None:
        dense_fixture_path.parent.mkdir(parents=True, exist_ok=True)
        input_provenance = {
            key: input_bundle[key]
            for key in ("source_prompt", "prompt_token_ids", "provenance")
            if key in input_bundle
        }
        dense_input_bundle = {
            "format": DENSE_INPUT_FORMAT,
            "projection": DENSE_PROJECTION,
            "input_dtype": "float32",
            "input_shape": [args.hidden_size],
            "rows": dense_inputs,
            "source_model": MODEL_REPOSITORY,
            "source_revision": MODEL_REVISION,
            "source_hashes": {
                "config.json": sha256_file(config_path),
                "model.safetensors.index.json": sha256_file(index_path),
                "mlx_audio/lm/models/qwen3_5.py": sha256_file(
                    mlx_root / "mlx_audio/lm/models/qwen3_5.py"
                ),
                "mlx_audio/utils.py": sha256_file(mlx_root / "mlx_audio/utils.py"),
                "input_bundle": sha256_file(inputs_path),
                "layer0_projection_tensors": dense_tensor_hash["sha256"],
            },
            "source_tensor_names": dense_tensor_hash["tensor_names"],
            "source_tensor_dtypes": dense_tensor_hash["tensor_dtypes"],
            "source_tensor_shapes": dense_tensor_hash["tensor_shapes"],
            "input_provenance": input_provenance,
        }
        reference_path = dense_fixture_path.with_name(
            dense_fixture_path.stem + ".reference.json"
        )
        dense_input_path = dense_fixture_path
        dense_input_path.write_text(
            json.dumps(dense_input_bundle, separators=(",", ":"), allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        dense_reference_bundle = {
            "format": DENSE_REFERENCE_FORMAT,
            "projection": DENSE_PROJECTION,
            "source_model": MODEL_REPOSITORY,
            "source_revision": MODEL_REVISION,
            "upstream_revision": MLX_REVISION,
            "input_fixture": {
                "path": dense_input_path.name,
                "sha256": sha256_file(dense_input_path),
            },
            "source_hashes": dense_input_bundle["source_hashes"],
            "rows": dense_reference_rows,
        }
        reference_path.write_text(
            json.dumps(dense_reference_bundle, separators=(",", ":"), allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        dense_paths = {"input": dense_input_path, "reference": reference_path}

    shards = sorted(set(selected.values()))
    source_files = {
        "qwen3_5": mlx_root / "mlx_audio/lm/models/qwen3_5.py",
        "fish_omni_loader": mlx_root / "mlx_audio/tts/models/fish_omni/fish_omni.py",
        "utils": mlx_root / "mlx_audio/utils.py",
    }
    model_files = {
        "config.json": sha256_file(config_path),
        "model.safetensors.index.json": sha256_file(index_path),
    }
    if hash_shards:
        model_files.update({shard: sha256_file(model_dir / shard) for shard in shards})
    manifest = {
        "kind": "actual_pinned_q4_backbone_replay",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "upstream": {
            "mlx-audio": {"repository": "https://github.com/cguldogan/mlx-audio", "revision": MLX_REVISION}
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
            "numpy": np.__version__,
            "safetensors": safetensors_version,
        },
        "inputs": {"path": inputs_path.name, "sha256": sha256_file(inputs_path)},
        "model_files": model_files,
        "shard_sha256_recorded": hash_shards,
        "reference_files": {
            name: {"path": path.relative_to(mlx_root).as_posix(), "sha256": sha256_file(path)}
            for name, path in source_files.items()
        },
        "backbone_tensor_count": len(selected),
        "backbone_shards": shards,
        "row_count": len(embeddings),
        "trace_count": len(traces),
        "trace_sha256": sha256_file(output_path),
        "trace_names": [trace["name"] for trace in traces],
    }
    if dense_paths:
        manifest["dense_projection_fixture"] = {
            "input_path": dense_paths["input"].name,
            "input_sha256": sha256_file(dense_paths["input"]),
            "reference_path": dense_paths["reference"].name,
            "reference_sha256": sha256_file(dense_paths["reference"]),
            "source_tensor_sha256": dense_tensor_hash["sha256"],
        }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--inputs", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="list selected backbone shards without loading weights")
    parser.add_argument("--hash-shards", action="store_true", help="hash selected multi-gigabyte weight shards after replay")
    parser.add_argument(
        "--dense-fixture",
        type=pathlib.Path,
        help=(
            "also write layer-0 linear-attention inputs plus a sibling .reference.json "
            "with upstream qkv/z/b/a outputs"
        ),
    )
    args = parser.parse_args()
    args.mlx_root = args.mlx_root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.inputs = args.inputs.resolve()
    args.output = args.output.resolve()
    if args.dense_fixture is not None:
        args.dense_fixture = args.dense_fixture.resolve()

    if args.plan_only:
        index = json.loads((args.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        selected = selected_backbone_map(index["weight_map"])
        shards = sorted(set(selected.values()))
        print(json.dumps({"backbone_tensor_count": len(selected), "backbone_shards": shards}))
        return 0

    manifest = run_replay(
        args.mlx_root,
        args.model_dir,
        args.inputs,
        args.output,
        args.hash_shards,
        args.dense_fixture,
    )
    result = {
        "trace_count": manifest["trace_count"],
        "manifest": str(args.output) + ".manifest.json",
    }
    if args.dense_fixture is not None:
        result["dense_fixture"] = str(args.dense_fixture)
        result["dense_reference"] = str(
            args.dense_fixture.with_name(args.dense_fixture.stem + ".reference.json")
        )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
