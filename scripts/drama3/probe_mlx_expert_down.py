#!/usr/bin/env python3
"""Compare pinned MLX Q4 expert-down paths on an upstream-captured activation.

The full eight-expert gather_qmm arm reproduces the upstream M=1 switch path
and is the required bitwise control. Single-expert gather_qmm, direct
quantized_matmul, and dequantize-plus-F32 matmul are diagnostic comparisons;
they are not treated as equivalent dispatches or as an FP32 oracle. All MLX
work is retained until one explicit terminal mx.eval, then read back.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_FORMAT = "drama3-moe-layer-diagnostic-v1"
INPUT_FORMAT = "drama3-moe-input-v1"
ABLATION_FORMAT = "drama3-moe-ablation-v1"
PROBE_FORMAT = "drama3-expert-down-probe-v1"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
UPSTREAM_REPOSITORY = "https://github.com/cguldogan/mlx-audio"
UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
COORDINATE = {"row": 1, "layer": 39}
EXPERT_COUNT = 256
TOP_K = 8
INPUT_WIDTH = 2048
INTERMEDIATE_WIDTH = 512
TARGET_EXPERT = 108
GROUP_SIZE = 64
BITS = 4
DEFAULT_ATOL = 6e-5
DEFAULT_RTOL = 3e-4
REQUIRED_SOURCE_HASHES = {
    "config.json",
    "model.safetensors.index.json",
    "mlx_audio/lm/models/qwen3_5.py",
    "mlx_audio/lm/models/switch_layers.py",
    "mlx_audio/utils.py",
    "scripts/drama3/replay_mlx_backbone.py",
    "scripts/drama3/replay_mlx_moe.py",
    "input_bundle",
    "target_moe_weights",
}
EXPECTED_CAPTURE_KINDS = {
    "expert_activation_outputs": "upstream_swiglu_return",
    "expert_down_outputs": "upstream_switch_mlp_return",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, where: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{where}: expected lowercase SHA-256 hex")
    return value


def _f32(value: Any, where: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)),
             f"{where}: expected numeric float32 value")
    number = float(value)
    _require(math.isfinite(number), f"{where}: expected finite float32 value")
    try:
        result = np.asarray(number, dtype="<f4").item()
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{where}: value outside float32 range") from exc
    _require(math.isfinite(result), f"{where}: value outside finite float32 range")
    return float(result)


def _float_values(record: Any, shape: list[int], where: str,
                  capture_kind: str) -> list[float]:
    _require(isinstance(record, Mapping), f"{where}: expected tensor record")
    _require(record.get("shape") == shape, f"{where}: expected shape {shape}")
    _require(record.get("dtype") == "float32", f"{where}: expected float32")
    _require(record.get("capture_kind") == capture_kind,
             f"{where}: expected capture_kind {capture_kind!r}")
    values = record.get("values")
    count = math.prod(shape)
    _require(isinstance(values, list) and len(values) == count,
             f"{where}: expected {count} values")
    return [_f32(value, f"{where}.values[{i}]") for i, value in enumerate(values)]


def _route_ids(record: Any, where: str) -> list[int]:
    _require(isinstance(record, Mapping), f"{where}: expected ID tensor record")
    _require(record.get("shape") == [TOP_K] and record.get("dtype") == "int32",
             f"{where}: expected int32 shape [{TOP_K}]")
    _require(record.get("capture_kind") == "upstream_local",
             f"{where}: unexpected route ID capture kind")
    values = record.get("values")
    _require(isinstance(values, list) and len(values) == TOP_K,
             f"{where}: expected {TOP_K} route IDs")
    ids: list[int] = []
    for i, value in enumerate(values):
        _require(type(value) is int and 0 <= value < EXPERT_COUNT,
                 f"{where}.values[{i}]: expert ID outside range")
        ids.append(value)
    _require(len(set(ids)) == TOP_K, f"{where}: duplicate route IDs")
    return ids


def _source_hashes(value: Any, where: str) -> dict[str, str]:
    _require(isinstance(value, Mapping), f"{where}: expected source_hashes object")
    missing = REQUIRED_SOURCE_HASHES - set(value)
    _require(not missing, f"{where}: missing source hashes {sorted(missing)}")
    return {str(name): _sha256(digest, f"{where}.{name}")
            for name, digest in value.items()}


def _pinned_identity(value: Mapping[str, Any], where: str) -> None:
    _require(value.get("coordinate") == COORDINATE,
             f"{where}: expected row 1 layer 39")
    model = value.get("model")
    upstream = value.get("upstream")
    _require(isinstance(model, Mapping) and
             model.get("repository") == MODEL_REPOSITORY and
             model.get("revision") == MODEL_REVISION,
             f"{where}: wrong pinned model")
    _require(isinstance(upstream, Mapping) and
             upstream.get("repository") == UPSTREAM_REPOSITORY and
             upstream.get("revision") == UPSTREAM_REVISION,
             f"{where}: wrong pinned upstream")


def validate_probe_inputs(upstream: Mapping[str, Any],
                          ablation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate pinned upstream and native-ablation records without MLX/GPU."""
    _require(isinstance(upstream, Mapping), "upstream: expected object")
    _require(isinstance(ablation, Mapping), "ablation: expected object")
    _require(upstream.get("format") == UPSTREAM_FORMAT and
             upstream.get("mode") == "full_run",
             "upstream: expected full-run pinned MoE diagnostic")
    _pinned_identity(upstream, "upstream")
    config = upstream.get("model_config")
    expected_config = {
        "hidden_size": INPUT_WIDTH,
        "moe_intermediate_size": INTERMEDIATE_WIDTH,
        "num_experts": EXPERT_COUNT,
        "num_experts_per_tok": TOP_K,
    }
    _require(isinstance(config, Mapping), "upstream.model_config: missing")
    for name, expected in expected_config.items():
        _require(type(config.get(name)) is int and config[name] == expected,
                 f"upstream.model_config.{name}: expected {expected}")
    runtime = upstream.get("runtime")
    _require(isinstance(runtime, Mapping) and runtime.get("mlx") == "0.31.1",
             "upstream.runtime.mlx: expected pinned MLX 0.31.1")
    _require(isinstance(runtime.get("device"), str) and
             "gpu" in runtime["device"].lower(),
             "upstream.runtime.device: expected GPU execution")
    input_record = upstream.get("input")
    _require(isinstance(input_record, Mapping), "upstream.input: missing")
    _require(input_record.get("format") == INPUT_FORMAT,
             "upstream.input: wrong fixture format")
    path = input_record.get("path")
    _require(isinstance(path, str) and path not in ("", ".", "..") and
             Path(path).name == path and "/" not in path and "\\" not in path,
             "upstream.input.path: expected sibling filename")
    input_hash = _sha256(input_record.get("sha256"), "upstream.input.sha256")
    activation_hash = _sha256(input_record.get("activation_sha256"),
                              "upstream.input.activation_sha256")
    _require(input_record.get("shape") == [INPUT_WIDTH] and
             input_record.get("dtype") == "float32" and
             input_record.get("activation_name") == "post_attention_layernorm",
             "upstream.input: wrong normalized activation metadata")
    source_hashes = _source_hashes(upstream.get("source_hashes"),
                                   "upstream.source_hashes")
    _require(source_hashes["target_moe_weights"] ==
             _sha256(source_hashes["target_moe_weights"],
                     "upstream.source_hashes.target_moe_weights"),
             "upstream target weight digest malformed")
    control = upstream.get("capture_control")
    _require(isinstance(control, Mapping) and control.get("passed") is True,
             "upstream.capture_control: must pass")

    route_ids = _route_ids(upstream.get("stages", {}).get(
        "expert_ids_argpartition_order") if isinstance(upstream.get("stages"), Mapping) else None,
        "upstream.stages.expert_ids_argpartition_order")
    rank_map = upstream.get("expert_rank_to_id")
    _require(isinstance(rank_map, list) and len(rank_map) == TOP_K,
             "upstream.expert_rank_to_id: expected eight explicit rows")
    pairs: dict[int, int] = {}
    for i, item in enumerate(rank_map):
        _require(isinstance(item, Mapping) and type(item.get("rank")) is int and
                 type(item.get("expert_id")) is int,
                 f"upstream.expert_rank_to_id[{i}]: malformed mapping")
        rank, expert_id = item["rank"], item["expert_id"]
        _require(rank not in pairs and 0 <= expert_id < EXPERT_COUNT,
                 f"upstream.expert_rank_to_id[{i}]: duplicate or invalid mapping")
        pairs[rank] = expert_id
    _require(set(pairs) == set(range(TOP_K)),
             "upstream.expert_rank_to_id: ranks must be 0..7")
    mapped = [pairs[i] for i in range(TOP_K)]
    _require(mapped == route_ids,
             "upstream.expert_rank_to_id: order must match route IDs")
    _require(TARGET_EXPERT in route_ids,
             f"expert {TARGET_EXPERT} is absent from the route")
    target_rank = route_ids.index(TARGET_EXPERT)

    stages = upstream.get("stages")
    _require(isinstance(stages, Mapping), "upstream.stages: missing")
    _require(stages.get("expert_ids_argpartition_order", {}).get("values") == route_ids,
             "upstream route IDs are inconsistent")
    activation_values = _float_values(
        stages.get("expert_activation_outputs"), [TOP_K, INTERMEDIATE_WIDTH],
        "upstream.stages.expert_activation_outputs",
        EXPECTED_CAPTURE_KINDS["expert_activation_outputs"],
    )
    activation_payload = np.asarray(activation_values, dtype="<f4").tobytes(order="C")
    target_activation = activation_values[
        target_rank * INTERMEDIATE_WIDTH:(target_rank + 1) * INTERMEDIATE_WIDTH
    ]
    target_activation_payload = np.asarray(target_activation, dtype="<f4").tobytes(order="C")
    down_values = _float_values(
        stages.get("expert_down_outputs"), [TOP_K, INPUT_WIDTH],
        "upstream.stages.expert_down_outputs",
        EXPECTED_CAPTURE_KINDS["expert_down_outputs"],
    )
    for stage_name in ("expert_activation_outputs", "expert_down_outputs"):
        record = stages[stage_name]
        _require(record.get("expert_ids") == route_ids,
                 f"upstream.stages.{stage_name}: explicit IDs/order must match route")

    _require(ablation.get("format") == ABLATION_FORMAT,
             "ablation: wrong format")
    _pinned_identity(ablation, "ablation")
    provenance = ablation.get("model_weight_provenance")
    required_provenance = {
        "target_layer", "target_moe_weights_sha256", "tensor_count", "model_path",
        "config_sha256", "index_sha256", "verified_before_gpu",
    }
    _require(isinstance(provenance, Mapping) and
             set(provenance) == required_provenance,
             "ablation.model_weight_provenance: incomplete provenance")
    _require(provenance.get("target_layer") == COORDINATE["layer"] and
             type(provenance.get("tensor_count")) is int and
             provenance.get("tensor_count") == 14 and
             provenance.get("verified_before_gpu") is True and
             isinstance(provenance.get("model_path"), str) and
             Path(provenance["model_path"]).is_absolute(),
             "ablation.model_weight_provenance: unexpected target or verification state")
    _require(_sha256(provenance.get("target_moe_weights_sha256"),
                     "ablation.target_moe_weights_sha256") ==
             source_hashes["target_moe_weights"],
             "ablation target MoE digest differs from upstream source hash")
    _require(_sha256(provenance.get("config_sha256"), "ablation.config_sha256") ==
             source_hashes["config.json"] and
             _sha256(provenance.get("index_sha256"), "ablation.index_sha256") ==
             source_hashes["model.safetensors.index.json"],
             "ablation config/index hashes differ from upstream source hashes")
    input_hashes = ablation.get("input_hashes")
    _require(isinstance(input_hashes, Mapping), "ablation.input_hashes: missing")
    for name in ("upstream_trace_sha256", "native_trace_sha256"):
        _sha256(input_hashes.get(name), f"ablation.input_hashes.{name}")
    _require(input_hashes.get("upstream_input_bundle_sha256") == input_hash and
             input_hashes.get("native_fixture_sha256") == input_hash and
             input_hashes.get("upstream_activation_sha256") == activation_hash and
             input_hashes.get("input_fixture_bytes_equal") is True and
             input_hashes.get("normalized_input_bitwise_equal") is True,
             "ablation.input_hashes: fixture/input binding mismatch")
    ablation_control = ablation.get("capture_control")
    down_control = ablation.get("native_down_replay_control")
    _require(isinstance(ablation_control, Mapping) and
             ablation_control.get("passed") is True,
             "ablation.capture_control: must pass")
    _require(isinstance(down_control, Mapping) and down_control.get("passed") is True,
             "ablation.native_down_replay_control: must pass")
    return {
        "route_ids": route_ids,
        "expert108_rank": target_rank,
        "activation_values": activation_values,
        "expert_activation_sha256": hashlib.sha256(activation_payload).hexdigest(),
        "target_expert_activation_sha256": hashlib.sha256(target_activation_payload).hexdigest(),
        "upstream_down_values": down_values,
        "input_path": path,
        "input_sha256": input_hash,
        "activation_sha256": activation_hash,
        "target_moe_weights_sha256": source_hashes["target_moe_weights"],
        "source_hashes": source_hashes,
        "model_path": provenance["model_path"],
        "ablation_upstream_trace_sha256": input_hashes["upstream_trace_sha256"],
        "ablation_native_trace_sha256": input_hashes["native_trace_sha256"],
    }


def _canonical_f32(values: Sequence[Any], where: str) -> tuple[list[float], bytes]:
    _require(isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
             f"{where}: expected numeric sequence")
    parsed = [_f32(item, f"{where}[{i}]") for i, item in enumerate(values)]
    array = np.asarray(parsed, dtype="<f4")
    return parsed, array.tobytes(order="C")


def validate_fixture_binding(upstream: Mapping[str, Any],
                             fixture: Mapping[str, Any],
                             raw_bytes: bytes) -> dict[str, Any]:
    """Validate the sibling activation fixture and return its CPU F32 values."""
    _require(isinstance(upstream, Mapping), "upstream: expected object")
    _require(isinstance(fixture, Mapping), "fixture: expected object")
    _require(isinstance(raw_bytes, bytes), "fixture bytes: expected bytes")
    input_record = upstream.get("input")
    _require(isinstance(input_record, Mapping), "upstream.input: missing")
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    _require(raw_hash == input_record.get("sha256"),
             "fixture bytes: SHA-256 differs from upstream input record")
    _require(fixture.get("format") == INPUT_FORMAT,
             "fixture: wrong drama3-moe-input-v1 format")
    _require(fixture.get("coordinate") == COORDINATE and
             fixture.get("activation_name") == "post_attention_layernorm" and
             fixture.get("activation_shape") == [INPUT_WIDTH] and
             fixture.get("activation_dtype") == "float32",
             "fixture: wrong normalized activation identity/shape/dtype")
    _pinned_identity(fixture, "fixture")
    expected_sources = _source_hashes(upstream.get("source_hashes"),
                                      "upstream.source_hashes")
    fixture_sources = _source_hashes(fixture.get("source_hashes"),
                                     "fixture.source_hashes")
    _require(fixture_sources == expected_sources,
             "fixture source hashes differ from upstream trace")
    values, payload = _canonical_f32(fixture.get("activation"), "fixture.activation")
    _require(len(values) == INPUT_WIDTH,
             f"fixture.activation: expected {INPUT_WIDTH} values")
    activation_hash = hashlib.sha256(payload).hexdigest()
    _require(activation_hash == fixture.get("activation_sha256") ==
             input_record.get("activation_sha256"),
             "fixture activation hash differs from upstream trace")
    _require(input_record.get("path") and Path(input_record["path"]).name == input_record["path"],
             "upstream.input.path: expected sibling filename")
    stages = upstream.get("stages")
    _require(isinstance(stages, Mapping), "upstream.stages: missing")
    normalized = stages.get("normalized_input")
    normalized_values = _float_values(
        normalized, [INPUT_WIDTH], "upstream.stages.normalized_input",
        "upstream_sparse_argument",
    )
    _require(np.asarray(normalized_values, dtype="<f4").tobytes() == payload,
             "fixture activation differs from captured normalized_input stage")
    return {
        "activation": values,
        "activation_sha256": activation_hash,
        "fixture_sha256": raw_hash,
        "model": dict(fixture["model"]),
        "upstream": dict(fixture["upstream"]),
    }


def compare_float32(actual: Sequence[Any], expected: Sequence[Any],
                    atol: float = DEFAULT_ATOL,
                    rtol: float = DEFAULT_RTOL) -> dict[str, Any]:
    """Compare finite F32 values with exact-bit and non-widenable tolerances."""
    _require(not isinstance(atol, bool) and isinstance(atol, (int, float)) and
             math.isfinite(float(atol)) and 0 <= atol <= DEFAULT_ATOL,
             f"atol must be finite, nonnegative, and no greater than {DEFAULT_ATOL}")
    _require(not isinstance(rtol, bool) and isinstance(rtol, (int, float)) and
             math.isfinite(float(rtol)) and 0 <= rtol <= DEFAULT_RTOL,
             f"rtol must be finite, nonnegative, and no greater than {DEFAULT_RTOL}")
    left, left_bytes = _canonical_f32(actual, "actual")
    right, right_bytes = _canonical_f32(expected, "expected")
    _require(len(left) == len(right), "actual and expected value counts differ")
    errors = [abs(a - b) for a, b in zip(left, right)]
    rel_errors = [error / max(abs(reference), 1.0e-30)
                  for error, reference in zip(errors, right)]
    failures = [i for i, (a, b, error) in enumerate(zip(left, right, errors))
                if error > float(atol) + float(rtol) * abs(b)]
    worst = max(range(len(errors)), key=errors.__getitem__) if errors else None
    return {
        "count": len(left),
        "bitwise_equal": left_bytes == right_bytes,
        "max_abs_error": max(errors, default=0.0),
        "max_rel_error": max(rel_errors, default=0.0),
        "atol": float(atol),
        "rtol": float(rtol),
        "tolerance_passed": not failures,
        "failure_count": len(failures),
        "failure_indices": failures[:16],
        "worst_index": worst,
        "actual_at_worst": left[worst] if worst is not None else None,
        "expected_at_worst": right[worst] if worst is not None else None,
    }


class ExecutionGuard:
    """Enforce one explicit terminal eval and no reads through this guard before it."""

    def __init__(self) -> None:
        self.terminal_eval_calls = 0
        self.readback_count = 0
        self.evaluated = False
        self.readback_names: list[str] = []

    def terminal_eval(self, mx: Any, *arrays: Any) -> None:
        if self.terminal_eval_calls != 0:
            raise RuntimeError("terminal mx.eval may be called only once")
        if not arrays:
            raise RuntimeError("terminal mx.eval requires retained result arrays")
        self.terminal_eval_calls += 1
        mx.eval(*arrays)
        self.evaluated = True

    def readback(self, name: str, array: Any) -> Any:
        if not self.evaluated:
            raise RuntimeError(f"readback {name!r} attempted before terminal mx.eval")
        self.readback_count += 1
        self.readback_names.append(name)
        return array.tolist()


def _read_json(path: Path, where: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        result = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{where}: cannot read JSON: {exc}") from exc
    _require(isinstance(result, dict), f"{where}: expected JSON object")
    return raw, result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_sources(mlx_root: Path, model_dir: Path,
                            source_hashes: Mapping[str, str]) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config_sha = _sha256_file(config_path)
    index_sha = _sha256_file(index_path)
    _require(config_sha == source_hashes["config.json"],
             "model config hash differs from pinned fixture")
    _require(index_sha == source_hashes["model.safetensors.index.json"],
             "model index hash differs from pinned fixture")
    try:
        revision = subprocess.run(
            ["git", "-C", str(mlx_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"mlx-root: cannot verify pinned Git revision: {exc}") from exc
    _require(revision == UPSTREAM_REVISION,
             f"mlx-root: expected {UPSTREAM_REVISION}, found {revision}")
    checked_sources = {}
    for relative in (
        "mlx_audio/lm/models/qwen3_5.py",
        "mlx_audio/lm/models/switch_layers.py",
        "mlx_audio/utils.py",
    ):
        actual = _sha256_file(mlx_root / relative)
        _require(actual == source_hashes[relative],
                 f"pinned source hash mismatch: {relative}")
        checked_sources[relative] = actual
    checked_replay_sources = {}
    script_dir = Path(__file__).resolve().parent
    for name in ("replay_mlx_backbone.py", "replay_mlx_moe.py"):
        relative = f"scripts/drama3/{name}"
        actual = _sha256_file(script_dir / name)
        _require(actual == source_hashes[relative],
                 f"local replay source hash mismatch: {relative}")
        checked_replay_sources[relative] = actual
    return {
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "upstream_git_revision": revision,
        "upstream_source_sha256": checked_sources,
        "local_replay_source_sha256": checked_replay_sources,
    }


def _load_down_weights(model_dir: Path, index: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from safetensors import safe_open

    prefix = "backbone.layers.39.mlp.switch_mlp.down_proj."
    names = [prefix + name for name in ("weight", "scales", "biases")]
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, Mapping), "model index: missing weight_map")
    missing = [name for name in names if name not in weight_map]
    _require(not missing, f"model index missing down-projection tensors: {missing}")
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(str(weight_map[name]), []).append(name)
    digest = hashlib.sha256()
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"tensor_names": sorted(names), "tensor_dtypes": {}, "tensor_shapes": {}}
    for shard in sorted(by_shard):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            _require(set(by_shard[shard]) <= available,
                     f"{shard}: missing indexed down-projection tensor")
            for name in sorted(by_shard[shard]):
                tensor = tensors.get_tensor(name).contiguous()
                dtype = str(tensor.dtype).removeprefix("torch.")
                shape = [int(dim) for dim in tensor.shape]
                raw = tensor.view(torch.uint8).numpy()
                digest.update(name.encode("utf-8") + b"\0")
                digest.update(dtype.encode("ascii") + b"\0")
                digest.update(np.asarray(shape, dtype="<i8").tobytes())
                digest.update(raw)
                metadata["tensor_dtypes"][name] = dtype
                metadata["tensor_shapes"][name] = shape
                if name.endswith(".weight"):
                    value = tensor.view(torch.int32).numpy().view(np.uint32).copy()
                else:
                    value = tensor.float().numpy().astype(np.float32, copy=True)
                arrays[name] = value
                del raw, tensor
    expected_shapes = {
        names[0]: [EXPERT_COUNT, INPUT_WIDTH, 64],
        names[1]: [EXPERT_COUNT, INPUT_WIDTH, 8],
        names[2]: [EXPERT_COUNT, INPUT_WIDTH, 8],
    }
    for name, shape in expected_shapes.items():
        _require(metadata["tensor_shapes"][name] == shape,
                 f"{name}: expected shape {shape}")
    _require(metadata["tensor_dtypes"][names[0]] == "uint32" and
             metadata["tensor_dtypes"][names[1]] == "bfloat16" and
             metadata["tensor_dtypes"][names[2]] == "bfloat16",
             "down-projection packed/scales/biases dtypes differ from pinned Q4 layout")
    metadata["sha256"] = digest.hexdigest()
    return arrays, metadata


def _array_metadata(name: str, array: Any, expected_shape: Sequence[int],
                    expected_dtype: Any) -> dict[str, Any]:
    shape = [int(dim) for dim in array.shape]
    dtype = str(array.dtype)
    _require(shape == list(expected_shape),
             f"{name}: expected shape {list(expected_shape)}, found {shape}")
    _require(array.dtype == expected_dtype,
             f"{name}: expected dtype {expected_dtype}, found {dtype}")
    return {"shape": shape, "dtype": dtype}


def _run_gpu_probe(probe: Mapping[str, Any], fixture_info: Mapping[str, Any],
                   model_dir: Path, model_hashes: Mapping[str, Any],
                   raw_target: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                   down_meta: Mapping[str, Any], output_path: Path,
                   tf32_env_before: str | None) -> int:
    # Import MLX only after the caller records the precision-related environment.
    import mlx.core as mx

    mlx_version = importlib.metadata.version("mlx")
    _require(mlx_version == "0.31.1", f"runtime MLX expected 0.31.1, found {mlx_version}")
    mx.set_default_device(mx.gpu)
    device = str(mx.default_device())
    _require("gpu" in device.lower(), f"expected GPU default device, found {device}")

    script_dir = Path(__file__).resolve().parent
    prefix = "backbone.layers.39.mlp.switch_mlp.down_proj."
    w_cpu = arrays[prefix + "weight"]
    s_cpu = arrays[prefix + "scales"]
    b_cpu = arrays[prefix + "biases"]
    # Device arrays are lazily retained. The all-expert tensors are needed for
    # both the official-shape gather and its single-expert gather control.
    w_all = mx.array(w_cpu)
    s_all = mx.array(s_cpu)
    b_all = mx.array(b_cpu)
    rank = int(probe["expert108_rank"])
    expert_ids = np.asarray(probe["route_ids"], dtype=np.int32)
    activation = np.asarray(probe["activation_values"], dtype=np.float32).reshape(TOP_K, INTERMEDIATE_WIDTH)
    expected = np.asarray(probe["upstream_down_values"], dtype=np.float32).reshape(TOP_K, INPUT_WIDTH)
    x_full = mx.array(activation.reshape(1, 1, TOP_K, 1, INTERMEDIATE_WIDTH))
    ids_full = mx.array(expert_ids.reshape(1, 1, TOP_K))
    full_raw = mx.gather_qmm(
        x_full, w_all, s_all, b_all, rhs_indices=ids_full,
        transpose=True, group_size=GROUP_SIZE, bits=BITS, mode="affine",
        sorted_indices=False, stream=mx.gpu,
    )
    full8 = mx.reshape(full_raw, (TOP_K, INPUT_WIDTH))

    x_single = mx.array(activation[rank].reshape(1, 1, 1, 1, INTERMEDIATE_WIDTH))
    id_single = mx.array(np.asarray([TARGET_EXPERT], dtype=np.int32).reshape(1, 1, 1))
    single_raw = mx.gather_qmm(
        x_single, w_all, s_all, b_all, rhs_indices=id_single,
        transpose=True, group_size=GROUP_SIZE, bits=BITS, mode="affine",
        sorted_indices=False, stream=mx.gpu,
    )
    single_gather = mx.reshape(single_raw, (INPUT_WIDTH,))

    w108, s108, b108 = w_all[TARGET_EXPERT], s_all[TARGET_EXPERT], b_all[TARGET_EXPERT]
    x108 = mx.array(activation[rank][None, :])
    direct_raw = mx.quantized_matmul(
        x108, w108, s108, b108, transpose=True, group_size=GROUP_SIZE, bits=BITS,
        mode="affine", stream=mx.gpu,
    )
    direct_qmm = mx.reshape(direct_raw, (INPUT_WIDTH,))

    dequantized = mx.dequantize(
        w108, s108, b108, group_size=GROUP_SIZE, bits=BITS, mode="affine",
        dtype=mx.float32, stream=mx.gpu,
    )
    f32_raw = mx.matmul(x108, dequantized.T, stream=mx.gpu)
    f32_matmul = mx.reshape(f32_raw, (INPUT_WIDTH,))

    # Decode the packed low-to-high uint4 fields in an independent GPU graph;
    # compare the resulting F32 weights with mx.dequantize without reading back
    # the million-element matrices.
    shifts = mx.array((np.arange(8, dtype=np.uint32) * np.uint32(4)).reshape(1, 1, 8))
    nibble_words = mx.reshape(mx.expand_dims(w108, -1), (INPUT_WIDTH, 64, 1))
    q = mx.bitwise_and(mx.right_shift(nibble_words, shifts), 15)
    q = mx.reshape(q, (INPUT_WIDTH, INTERMEDIATE_WIDTH)).astype(mx.float32)
    scale_expanded = mx.repeat(s108.astype(mx.float32), GROUP_SIZE, axis=-1)
    bias_expanded = mx.repeat(b108.astype(mx.float32), GROUP_SIZE, axis=-1)
    explicit_decode = q * scale_expanded + bias_expanded
    explicit_bits = mx.view(explicit_decode, mx.uint32, stream=mx.gpu)
    dequantized_bits = mx.view(dequantized, mx.uint32, stream=mx.gpu)
    decoded_equal_count = mx.sum(mx.equal(explicit_bits, dequantized_bits).astype(mx.int32), stream=mx.gpu)
    decoded_max_abs = mx.max(mx.abs(explicit_decode - dequantized), stream=mx.gpu)
    decoded_weight_count = INPUT_WIDTH * INTERMEDIATE_WIDTH

    # Read shape/dtype metadata from lazy array headers only. These assertions
    # pin M=1 and the source Q4 representation before the single terminal eval.
    actual_tensor_metadata = {
        "x_full": _array_metadata("x_full", x_full, [1, 1, TOP_K, 1, INTERMEDIATE_WIDTH], mx.float32),
        "x_single": _array_metadata("x_single", x_single, [1, 1, 1, 1, INTERMEDIATE_WIDTH], mx.float32),
        "x108": _array_metadata("x108", x108, [1, INTERMEDIATE_WIDTH], mx.float32),
        "w_all": _array_metadata("w_all", w_all, [EXPERT_COUNT, INPUT_WIDTH, 64], mx.uint32),
        "s_all": _array_metadata("s_all", s_all, [EXPERT_COUNT, INPUT_WIDTH, 8], mx.float32),
        "b_all": _array_metadata("b_all", b_all, [EXPERT_COUNT, INPUT_WIDTH, 8], mx.float32),
        "w108": _array_metadata("w108", w108, [INPUT_WIDTH, 64], mx.uint32),
        "s108": _array_metadata("s108", s108, [INPUT_WIDTH, 8], mx.float32),
        "b108": _array_metadata("b108", b108, [INPUT_WIDTH, 8], mx.float32),
        "full_raw": _array_metadata("full_raw", full_raw, [1, 1, TOP_K, 1, INPUT_WIDTH], mx.float32),
        "single_raw": _array_metadata("single_raw", single_raw, [1, 1, 1, 1, INPUT_WIDTH], mx.float32),
        "direct_raw": _array_metadata("direct_raw", direct_raw, [1, INPUT_WIDTH], mx.float32),
        "dequantized": _array_metadata("dequantized", dequantized, [INPUT_WIDTH, INTERMEDIATE_WIDTH], mx.float32),
        "f32_raw": _array_metadata("f32_raw", f32_raw, [1, INPUT_WIDTH], mx.float32),
    }

    guard = ExecutionGuard()
    guard.terminal_eval(
        mx, full8, single_gather, direct_qmm, f32_matmul,
        decoded_equal_count, decoded_max_abs,
    )

    # Every GPU->host conversion follows the single explicit terminal eval.
    full_values = np.asarray(guard.readback("full8_gather_qmm", full8), dtype=np.float32).reshape(-1).tolist()
    single_values = np.asarray(guard.readback("single_expert_gather_qmm", single_gather), dtype=np.float32).reshape(-1).tolist()
    direct_values = np.asarray(guard.readback("direct_quantized_matmul", direct_qmm), dtype=np.float32).reshape(-1).tolist()
    f32_values = np.asarray(guard.readback("dequantized_float32_matmul", f32_matmul), dtype=np.float32).reshape(-1).tolist()
    equal_count_value = int(np.asarray(guard.readback("decoded_weight_equal_count", decoded_equal_count)).item())
    decode_max_value = float(np.asarray(guard.readback("decoded_weight_max_abs", decoded_max_abs)).item())

    full_array = np.asarray(full_values, dtype=np.float32).reshape(TOP_K, INPUT_WIDTH)
    full_target = full_array[rank].tolist()
    expected_target = expected[rank].tolist()
    comparison = {
        "full8_gather_qmm_vs_upstream_all": compare_float32(full_values, expected.reshape(-1).tolist()),
        "full8_gather_qmm_expert108_vs_upstream": compare_float32(full_target, expected_target),
        "single_expert_gather_qmm_vs_upstream": compare_float32(single_values, expected_target),
        "direct_quantized_matmul_vs_upstream": compare_float32(direct_values, expected_target),
        "dequantized_float32_matmul_vs_upstream": compare_float32(f32_values, expected_target),
        "single_gather_vs_full8_expert108": compare_float32(single_values, full_target),
        "direct_qmm_vs_single_gather": compare_float32(direct_values, single_values),
        "f32_matmul_vs_single_gather": compare_float32(f32_values, single_values),
    }
    official_control_passed = comparison["full8_gather_qmm_vs_upstream_all"]["bitwise_equal"]
    script_hashes = {
        "probe_mlx_expert_down.py": _sha256_file(Path(__file__).resolve()),
        "replay_mlx_moe.py": _sha256_file(script_dir / "replay_mlx_moe.py"),
        "replay_mlx_backbone.py": _sha256_file(script_dir / "replay_mlx_backbone.py"),
    }
    result = {
        "format": PROBE_FORMAT,
        "coordinate": dict(COORDINATE),
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION},
        "runtime": {"mlx": mlx_version, "device": device},
        "actual_tensor_metadata": actual_tensor_metadata,
        "precision_environment": {
            "MLX_ENABLE_TF32_before_import": tf32_env_before,
            "MLX_ENABLE_TF32_effective_environment": os.environ.get("MLX_ENABLE_TF32"),
            "note": "Environment recorded; pinned M=1 gather-qmv arms use the low-M path. Environment does not prove a compile-time MLX TF32 setting.",
        },
        "input": {
            "format": INPUT_FORMAT,
            "path": probe["input_path"],
            "sha256": probe["input_sha256"],
            "activation_sha256": fixture_info["activation_sha256"],
            "activation_name": "post_attention_layernorm",
            "shape": [INPUT_WIDTH],
            "dtype": "float32",
        },
        "expert_activation": {
            "source_stage": "expert_activation_outputs",
            "capture_kind": EXPECTED_CAPTURE_KINDS["expert_activation_outputs"],
            "shape": [TOP_K, INTERMEDIATE_WIDTH],
            "dtype": "float32",
            "sha256": probe["expert_activation_sha256"],
            "expert_ids": list(probe["route_ids"]),
            "expert_rank_to_id": [
                {"rank": i, "expert_id": expert_id}
                for i, expert_id in enumerate(probe["route_ids"])
            ],
            "target_expert_id": TARGET_EXPERT,
            "target_expert_rank": rank,
            "target_expert_activation_sha256": probe["target_expert_activation_sha256"],
        },
        "provenance": {
            "model_path": str(model_dir.resolve()),
            "target_moe_weights_sha256_actual": raw_target["sha256"],
            "target_moe_tensor_count": len(raw_target["tensor_names"]),
            "down_tensor_sha256_actual": down_meta["sha256"],
            "down_tensor_count": len(down_meta["tensor_names"]),
            "down_tensor_names": down_meta["tensor_names"],
            "down_tensor_dtypes": down_meta["tensor_dtypes"],
            "down_tensor_shapes": down_meta["tensor_shapes"],
            "verified_before_gpu": True,
            "model_source_checks": dict(model_hashes),
            "script_sha256": script_hashes,
            "upstream_source_hashes": dict(probe["source_hashes"]),
            "upstream_trace_sha256": probe["ablation_upstream_trace_sha256"],
            "native_trace_sha256": probe["ablation_native_trace_sha256"],
        },
        "arms": {
            "full8_gather_qmm": {
                "capture_kind": "pinned_upstream_switch_path_shape_control",
                "shape": [TOP_K, INPUT_WIDTH],
                "values": full_values,
            },
            "single_expert_gather_qmm": {
                "capture_kind": "diagnostic_single_rhs_expert_gather",
                "shape": [INPUT_WIDTH],
                "expert_id": TARGET_EXPERT,
                "values": single_values,
            },
            "direct_quantized_matmul": {
                "capture_kind": "diagnostic_direct_quantized_matmul",
                "shape": [INPUT_WIDTH],
                "expert_id": TARGET_EXPERT,
                "values": direct_values,
            },
            "dequantized_float32_matmul": {
                "capture_kind": "diagnostic_mx_dequantize_then_mx_matmul",
                "shape": [INPUT_WIDTH],
                "expert_id": TARGET_EXPERT,
                "values": f32_values,
            },
        },
        "decoded_weight_control": {
            "capture_kind": "diagnostic_gpu_uint4_decode_vs_mx_dequantize",
            "weight_count": decoded_weight_count,
            "bitwise_differing_count": decoded_weight_count - equal_count_value,
            "max_abs_error": decode_max_value,
            "values_read_back": 0,
        },
        "comparisons": comparison,
        "official_capture_control": {
            "passed": official_control_passed,
            "criterion": "full8_gather_qmm output is bitwise equal to all eight upstream expert_down_outputs vectors",
            "value_count": TOP_K * INPUT_WIDTH,
            "mismatch_count": comparison["full8_gather_qmm_vs_upstream_all"]["count"] - sum(
                1 for a, b in zip(full_values, expected.reshape(-1).tolist())
                if np.asarray(a, dtype="<f4").tobytes() == np.asarray(b, dtype="<f4").tobytes()
            ),
            "comparison": comparison["full8_gather_qmm_vs_upstream_all"],
        },
        "execution_audit": {
            "explicit_terminal_mx_eval_calls": guard.terminal_eval_calls,
            "guarded_host_readback_calls": guard.readback_count,
            "host_readback_names": guard.readback_names,
            "host_reads_before_terminal_eval": False,
            "explicit_synchronize_api_calls_in_probe_source": "none",
            "backend_internal_synchronization": "not_instrumented",
            "all_gpu_metric_nodes_built_before_terminal_eval": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return 0 if official_control_passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True,
                        help="checkout of pinned mlx-audio revision")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream-trace", type=Path, required=True)
    parser.add_argument("--native-ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.mlx_root = args.mlx_root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.upstream_trace = args.upstream_trace.resolve()
    args.native_ablation = args.native_ablation.resolve()
    args.output = args.output.resolve()

    tf32_env_before = os.environ.get("MLX_ENABLE_TF32")
    # Keep the environment explicit for comparability before importing MLX. The
    # output records this variable but does not treat it as proof of a build flag.
    if tf32_env_before is None:
        os.environ["MLX_ENABLE_TF32"] = "0"
    raw_upstream_trace, upstream = _read_json(args.upstream_trace, "upstream trace")
    _, ablation = _read_json(args.native_ablation, "native ablation")
    probe = validate_probe_inputs(upstream, ablation)
    fixture_path = args.upstream_trace.parent / probe["input_path"]
    _require(fixture_path.resolve().parent == args.upstream_trace.parent.resolve(),
             "upstream input fixture must be a sibling file")
    fixture_raw = fixture_path.read_bytes()
    try:
        fixture = json.loads(fixture_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"upstream input fixture: invalid JSON: {exc}") from exc
    fixture_info = validate_fixture_binding(upstream, fixture, fixture_raw)
    _require(hashlib.sha256(raw_upstream_trace).hexdigest() ==
             probe["ablation_upstream_trace_sha256"],
             "native ablation binds a different upstream trace")
    model_hashes = _validate_model_sources(args.mlx_root, args.model_dir,
                                           probe["source_hashes"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from replay_mlx_moe import target_moe_weight_hash
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    actual_target = target_moe_weight_hash(args.model_dir, index["weight_map"])
    _require(actual_target["sha256"] == probe["target_moe_weights_sha256"],
             "loaded checkpoint target-MoE digest differs from pinned fixture")
    arrays, down_meta = _load_down_weights(args.model_dir, index)
    # Verify checkpoint bytes before importing MLX or creating any GPU arrays.
    return _run_gpu_probe(
        probe, fixture_info, args.model_dir, model_hashes, actual_target,
        arrays, down_meta, args.output, tf32_env_before,
    )


if __name__ == "__main__":
    raise SystemExit(main())
