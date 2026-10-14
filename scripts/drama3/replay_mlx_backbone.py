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
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    if data.size != width:
        raise ValueError(f"{name}: expected {width} values, found {data.size}")
    if not np.isfinite(data).all():
        raise FloatingPointError(f"{name}: non-finite reference activation")
    traces.append(
        {"name": name, "shape": [width], "dtype": "float32", "values": data.tolist()}
    )


def load_selected_weights(
    model: Any,
    model_dir: pathlib.Path,
    selected: dict[str, str],
    mx: Any,
) -> None:
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


def run_replay(
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
    inputs_path: pathlib.Path,
    output_path: pathlib.Path,
    hash_shards: bool = False,
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
    load_selected_weights(model, model_dir, selected, mx)
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

    def traced_layer(self: Any, x: Any, mask: Any = None, cache: Any = None) -> Any:
        if self.is_linear:
            residual = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            residual = self.self_attn(self.input_layernorm(x), mask, cache)
        attention_residual = x + residual
        layer = layer_ids[id(self)]
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"format": TRACE_FORMAT, "traces": traces}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

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
    args = parser.parse_args()
    args.mlx_root = args.mlx_root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.inputs = args.inputs.resolve()
    args.output = args.output.resolve()

    if args.plan_only:
        index = json.loads((args.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        selected = selected_backbone_map(index["weight_map"])
        shards = sorted(set(selected.values()))
        print(json.dumps({"backbone_tensor_count": len(selected), "backbone_shards": shards}))
        return 0

    manifest = run_replay(args.mlx_root, args.model_dir, args.inputs, args.output, args.hash_shards)
    print(json.dumps({"trace_count": manifest["trace_count"], "manifest": str(args.output) + ".manifest.json"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
