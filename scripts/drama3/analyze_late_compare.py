#!/usr/bin/env python3
"""Validate the pinned DRAMA-3 late-layer MLX fixture and its route identities.

This is an offline artifact checker. It reads fixture bytes only; it does not
load MLX, execute model code, or materialize activation tables in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Iterable

import numpy as np


FORMAT = "drama3-late-matched-intermediates-v1"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
UPSTREAM_REPOSITORY = "https://github.com/cguldogan/mlx-audio"
UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
ROWS = 158
WIDTH = 2048
LAYERS = tuple(range(40))
SELECTED_LAYERS = tuple(range(34, 40))
TOP_K = 8
EXPERTS = 256
EXPERT_WIDTH = 512
INTERMEDIATE_ATOL = 6e-5
RTOL = 3e-4
ENDPOINT_ATOL = 3e-5
NATIVE_FORMAT = "drama3-late-native-comparison-v1"
NATIVE_KINDS = (
    "post_attention_rms_replay",
    "expert_down_native",
    "expert_down_mlx_q4",
    "shared_projection",
    "shared_activation_replay",
    "shared_down_replay",
    "shared_weighted_replay",
    "moe_sum_add_replay",
    "moe_weighted_add_replay",
    "whole_moe_replay",
    "residual_upstream_add_control",
    "residual_native_moe_replay",
)

MOE_STAGES = (
    "normalized_input",
    "router_logits",
    "router_probabilities",
    "expert_ids",
    "switch_expert_ids",
    "switch_inverse_order",
    "selected_probabilities",
    "selected_weights",
    "expert_gate_outputs",
    "expert_up_outputs",
    "expert_activation_outputs",
    "expert_down_outputs",
    "expert_weighted_outputs",
    "routed_sum",
    "per_rank_prefix_sums",
    "shared_gate_output",
    "shared_up_output",
    "shared_activation_output",
    "shared_expert_output",
    "shared_gate_logits",
    "shared_weighted_output",
    "moe_output",
    "canonical_rank_order",
    "canonical_expert_ids",
    "canonical_route_ids_from_switch",
    "canonical_selected_probabilities",
    "canonical_selected_weights",
    "canonical_expert_gate_outputs",
    "canonical_expert_up_outputs",
    "canonical_expert_activation_outputs",
    "canonical_expert_down_outputs",
    "canonical_expert_weighted_outputs",
)


class ValidationError(ValueError):
    """The evidence artifact does not satisfy its declared capture contract."""


def _fail(message: str) -> None:
    raise ValidationError(message)


def _validated_readback_ledger(readbacks: Any, case_count: int) -> dict[str, int]:
    expected = {"diagnostic": 0, "control": 4, "terminal": case_count * 52 * 4}
    if not isinstance(readbacks, dict):
        _fail("readbacks byte ledger is missing")
    if set(readbacks) - set(expected) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in readbacks.values()
    ):
        _fail("readbacks byte ledger has unknown keys or invalid byte counts")
    normalized = {key: readbacks.get(key, 0) for key in expected}
    if normalized != expected:
        _fail(f"readback byte counts differ from one compact metrics receipt per case: {readbacks}")
    return normalized


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: pathlib.Path, relative: Any, label: str) -> pathlib.Path:
    if not isinstance(relative, str) or not relative:
        _fail(f"{label} path must be a non-empty relative string")
    candidate = pathlib.Path(relative)
    if candidate.is_absolute():
        _fail(f"{label} path must be relative: {relative}")
    joined = root / candidate
    resolved = joined.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        _fail(f"{label} path escapes fixture directory: {relative}")
    # A symlink in the fixture can change the bytes after validation. Reject it
    # even when its current target remains beneath the fixture root.
    cursor = root
    for component in candidate.parts:
        if component in ("", ".", ".."):
            _fail(f"{label} path contains a non-canonical component: {relative}")
        cursor = cursor / component
        if cursor.is_symlink():
            _fail(f"{label} path traverses a symlink: {relative}")
    return resolved


def _shape(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        _fail(f"{label} shape must be a non-empty integer list")
    dims: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            _fail(f"{label} has invalid dimension {dim!r}")
        dims.append(dim)
    return tuple(dims)


def _product(shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _validate_finite_f32(path: pathlib.Path) -> None:
    chunk_bytes = 4 * 262_144
    with path.open("rb") as stream:
        while True:
            payload = stream.read(chunk_bytes)
            if not payload:
                return
            if len(payload) % 4:
                _fail(f"float32 fixture has a partial element: {path}")
            values = np.frombuffer(payload, dtype="<f4")
            if not np.isfinite(values).all():
                _fail(f"float32 fixture contains NaN or infinity: {path}")


def _verify_source_hashes(manifest: dict[str, Any]) -> int:
    """Verify the manifest's declared source files against their pinned bytes."""
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        _fail("manifest source_hashes are missing")
    upstream = manifest.get("upstream", {})
    checkpoint = manifest.get("checkpoint", {})
    source_root = pathlib.Path(str(upstream.get("source_root", ""))).resolve()
    model_dir = pathlib.Path(str(checkpoint.get("model_dir", ""))).resolve()
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    for relative, expected_digest in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_digest, str) or len(expected_digest) != 64:
            _fail("source_hashes contains a malformed entry")
        if relative in ("config.json", "model.safetensors.index.json"):
            path = model_dir / relative
        elif relative.startswith("mlx_audio/"):
            path = source_root / relative
        elif relative.startswith("scripts/"):
            path = repo_root / relative
        else:
            _fail(f"unrecognized pinned source path {relative!r}")
        if not path.is_file() or sha256_file(path) != expected_digest:
            _fail(f"pinned source hash mismatch or missing source: {relative}")
    if checkpoint.get("config_sha256") != source_hashes.get("config.json"):
        _fail("checkpoint config hash differs from source_hashes")
    if checkpoint.get("index_sha256") != source_hashes.get("model.safetensors.index.json"):
        _fail("checkpoint index hash differs from source_hashes")
    return len(source_hashes)


def _validate_tensor_record(
    root: pathlib.Path,
    record: Any,
    *,
    expected_name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        _fail(f"{expected_name}: tensor record must be an object")
    if record.get("name") != expected_name:
        _fail(f"expected tensor {expected_name}, got {record.get('name')!r}")
    shape = _shape(record.get("shape"), expected_name)
    if shape != expected_shape:
        _fail(f"{expected_name}: shape {shape} != expected {expected_shape}")
    dtype = record.get("dtype")
    if dtype != expected_dtype:
        _fail(f"{expected_name}: dtype {dtype!r} != expected {expected_dtype!r}")
    expected_bytes = _product(shape) * 4
    if record.get("bytes") != expected_bytes:
        _fail(f"{expected_name}: byte count does not match shape and dtype")
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        _fail(f"{expected_name}: malformed SHA-256")
    path = _safe_child(root, record.get("file"), expected_name)
    if not path.is_file():
        _fail(f"{expected_name}: missing tensor file {path}")
    if path.stat().st_size != expected_bytes:
        _fail(f"{expected_name}: file size does not match declared shape")
    actual_digest = sha256_file(path)
    if actual_digest != digest:
        _fail(f"{expected_name}: SHA-256 mismatch")
    if dtype == "float32":
        _validate_finite_f32(path)
    return {"name": expected_name, "shape": list(shape), "dtype": dtype, "sha256": digest}


def expected_tensor_shapes(arm: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """Pinned exact names/shapes for the 40-layer, 158-row capture."""
    if arm not in ("scalar", "sequence"):
        _fail(f"unknown arm {arm!r}")
    expected: dict[str, tuple[tuple[int, ...], str]] = {}

    def add(name: str, shape: tuple[int, ...], dtype: str = "float32") -> None:
        if name in expected:
            _fail(f"internal duplicate expected tensor {name}")
        expected[name] = (shape, dtype)

    row_hidden = (ROWS, WIDTH)
    for layer in LAYERS:
        prefix = f"backbone.layers.{layer}."
        for stage in ("attention_output", "attention_residual", "hidden"):
            add(prefix + stage, row_hidden)
    for layer in SELECTED_LAYERS:
        add(f"backbone.layers.{layer}.decoder_input", row_hidden)
    add("backbone.hidden", row_hidden)

    for layer in SELECTED_LAYERS:
        prefix = f"backbone.layers.{layer}.mlp."
        for stage in MOE_STAGES:
            if stage == "switch_inverse_order" and arm == "scalar":
                continue
            if stage in ("expert_ids", "canonical_rank_order", "canonical_expert_ids", "canonical_route_ids_from_switch"):
                shape, dtype = (ROWS, TOP_K), "int32"
            elif stage == "switch_expert_ids":
                shape, dtype = ((ROWS, TOP_K) if arm == "scalar" else (ROWS * TOP_K,)), "int32"
            elif stage == "switch_inverse_order":
                shape, dtype = (ROWS * TOP_K,), "int32"
            elif stage in ("router_logits", "router_probabilities"):
                shape, dtype = (ROWS, EXPERTS), "float32"
            elif stage in ("selected_probabilities", "selected_weights", "canonical_selected_probabilities", "canonical_selected_weights"):
                shape, dtype = (ROWS, TOP_K), "float32"
            elif stage in ("expert_gate_outputs", "expert_up_outputs", "expert_activation_outputs"):
                shape = (ROWS, TOP_K, 1, EXPERT_WIDTH) if arm == "scalar" else (ROWS * TOP_K, 1, EXPERT_WIDTH)
                dtype = "float32"
            elif stage in ("canonical_expert_gate_outputs", "canonical_expert_up_outputs", "canonical_expert_activation_outputs"):
                shape, dtype = (ROWS, TOP_K, EXPERT_WIDTH), "float32"
            elif stage in ("expert_down_outputs", "expert_weighted_outputs", "canonical_expert_down_outputs", "canonical_expert_weighted_outputs", "per_rank_prefix_sums"):
                shape, dtype = (ROWS, TOP_K, WIDTH), "float32"
            elif stage == "shared_gate_logits":
                shape, dtype = (ROWS, 1), "float32"
            elif stage in ("shared_gate_output", "shared_up_output", "shared_activation_output"):
                shape, dtype = (ROWS, EXPERT_WIDTH), "float32"
            elif stage in ("normalized_input", "expert_down_outputs", "routed_sum", "shared_expert_output", "shared_weighted_output", "moe_output"):
                shape, dtype = row_hidden, "float32"
            else:
                _fail(f"unhandled expected MoE stage: {stage}")
            add(prefix + stage, shape, dtype)
    return expected


def _read_int_tensor(root: pathlib.Path, record: dict[str, Any]) -> np.ndarray:
    path = _safe_child(root, record["file"], record["name"])
    shape = tuple(record["shape"])
    return np.fromfile(path, dtype="<i4").reshape(shape)


def validate_canonical_route_values(
    *,
    raw_values: np.ndarray,
    canonical_values: np.ndarray,
    rank_order: np.ndarray,
    inverse_order: np.ndarray | None,
    rows: int,
    top_k: int,
) -> None:
    """Require canonical per-token routes to be an exact view of official values.

    Sequence switch projections/activation are captured in expert-sorted flat
    order and restored with the upstream inverse permutation. Down and weighted
    outputs are already returned in token/original-rank order. Both paths then
    gather ranks by the independently validated per-token expert sort.
    """
    raw = np.asarray(raw_values)
    canonical = np.asarray(canonical_values)
    rank = np.asarray(rank_order)
    if raw.dtype.kind != "f" or raw.dtype.itemsize != 4 or canonical.dtype != raw.dtype:
        _fail("raw and canonical route values must both be float32")
    if rank.shape != (rows, top_k) or rank.dtype.kind not in "iu":
        _fail("canonical rank order has an incompatible shape or dtype")
    if raw.ndim == 4 and raw.shape[:3] == (rows, top_k, 1):
        original = raw[:, :, 0, :]
    elif raw.ndim == 3 and raw.shape[:2] == (rows, top_k):
        original = raw
    elif raw.ndim == 3 and raw.shape[:2] == (rows * top_k, 1):
        if inverse_order is None:
            _fail("flattened sequence route values require switch_inverse_order")
        inverse = np.asarray(inverse_order)
        if inverse.shape != (rows * top_k,) or inverse.dtype.kind not in "iu":
            _fail("route-value inverse order has an incompatible shape or dtype")
        flat = raw[:, 0, :]
        original = flat[inverse.astype(np.intp, copy=False)].reshape(rows, top_k, -1)
    elif raw.ndim == 2 and raw.shape[0] == rows * top_k:
        if inverse_order is None:
            _fail("flattened sequence route values require switch_inverse_order")
        inverse = np.asarray(inverse_order)
        if inverse.shape != (rows * top_k,) or inverse.dtype.kind not in "iu":
            _fail("route-value inverse order has an incompatible shape or dtype")
        original = raw[inverse.astype(np.intp, copy=False)].reshape(rows, top_k, -1)
    else:
        _fail(f"unsupported official route-value shape {raw.shape}")
    if canonical.shape != original.shape:
        _fail(f"canonical route-value shape {canonical.shape} != {original.shape}")
    gather = np.broadcast_to(rank.astype(np.intp, copy=False)[..., None], original.shape)
    expected = np.take_along_axis(original, gather, axis=1)
    # Compare raw F32 payloads, not numeric equality, so signed zero and any
    # future payload-level discrepancy cannot slip through as a valid reorder.
    if not np.array_equal(expected.view(np.uint32), canonical.view(np.uint32)):
        _fail("canonical route values are not the exact official-value permutation")


def validate_route_arrays(
    *,
    expert_ids: np.ndarray,
    switch_ids: np.ndarray,
    canonical_rank_order: np.ndarray,
    canonical_expert_ids: np.ndarray,
    canonical_route_ids_from_switch: np.ndarray,
    switch_inverse_order: np.ndarray | None,
    number_of_experts: int,
) -> dict[str, int]:
    """Check original/sorted route bijection and canonical rank ordering."""
    ids = np.asarray(expert_ids)
    if ids.ndim != 2 or ids.dtype.kind not in "iu":
        _fail("expert_ids must be a two-dimensional integer table")
    rows, top_k = ids.shape
    if rows <= 0 or top_k <= 0 or number_of_experts <= 0:
        _fail("route dimensions and expert count must be positive")
    if np.any(ids < 0) or np.any(ids >= number_of_experts):
        _fail("expert_ids contains an out-of-range expert")
    if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        _fail("expert_ids contains duplicate experts in a token route")

    expected_rank = np.argsort(ids, axis=1, kind="stable")
    if not np.array_equal(canonical_rank_order, expected_rank):
        _fail("canonical_rank_order is not the per-token stable expert sort")
    expected_canonical = np.take_along_axis(ids, expected_rank, axis=1)
    if not np.array_equal(canonical_expert_ids, expected_canonical):
        _fail("canonical_expert_ids does not match sorted expert_ids")

    switch = np.asarray(switch_ids)
    if switch.ndim == 2:
        if switch.shape != ids.shape:
            _fail("unflattened switch_expert_ids shape differs from expert_ids")
        if switch_inverse_order is not None:
            _fail("unflattened switch_expert_ids must not carry inverse order")
        restored = switch
    elif switch.ndim == 1:
        if switch.size != rows * top_k:
            _fail("flattened switch_expert_ids has the wrong length")
        if switch_inverse_order is None:
            _fail("flattened switch_expert_ids requires switch_inverse_order")
        inverse = np.asarray(switch_inverse_order)
        if inverse.shape != (rows * top_k,) or inverse.dtype.kind not in "iu":
            _fail("switch_inverse_order must be a flattened integer permutation")
        if not np.array_equal(np.sort(inverse), np.arange(rows * top_k, dtype=inverse.dtype)):
            _fail("switch_inverse_order is not a bijection")
        stable_order = np.argsort(ids.reshape(-1), kind="stable")
        expected_inverse = np.argsort(stable_order, kind="stable").astype(inverse.dtype, copy=False)
        if not np.array_equal(inverse, expected_inverse):
            _fail("switch_inverse_order differs from the upstream stable expert sort")
        if not np.array_equal(switch, ids.reshape(-1)[stable_order]):
            _fail("switch_expert_ids differs from the upstream stable expert sort")
        restored = switch[inverse.astype(np.intp, copy=False)].reshape(rows, top_k)
        if np.any(np.diff(switch) < 0):
            _fail("flattened switch_expert_ids is not sorted by expert")
    else:
        _fail("switch_expert_ids must be a 2D original-order or 1D sorted table")
    if not np.array_equal(restored, ids):
        _fail("switch IDs and inverse transform do not restore expert_ids")
    expected_route = np.take_along_axis(restored, expected_rank, axis=1)
    if not np.array_equal(canonical_route_ids_from_switch, expected_route):
        _fail("canonical_route_ids_from_switch differs from reconstructed sorted IDs")
    return {"rows": rows, "top_k": top_k, "experts": number_of_experts}


def _route_check(root: pathlib.Path, records: dict[str, dict[str, Any]], arm: str, layer: int) -> None:
    prefix = f"backbone.layers.{layer}.mlp."
    inverse_record = records.get(prefix + "switch_inverse_order")
    expert_ids = _read_int_tensor(root, records[prefix + "expert_ids"])
    switch_ids = _read_int_tensor(root, records[prefix + "switch_expert_ids"])
    rank_order = _read_int_tensor(root, records[prefix + "canonical_rank_order"])
    inverse = (
        _read_int_tensor(root, inverse_record) if inverse_record is not None else None
    )
    validate_route_arrays(
        expert_ids=expert_ids,
        switch_ids=switch_ids,
        canonical_rank_order=rank_order,
        canonical_expert_ids=_read_int_tensor(root, records[prefix + "canonical_expert_ids"]),
        canonical_route_ids_from_switch=_read_int_tensor(root, records[prefix + "canonical_route_ids_from_switch"]),
        switch_inverse_order=inverse,
        number_of_experts=EXPERTS,
    )
    switch_shape = tuple(records[prefix + "switch_expert_ids"]["shape"])
    expected = (ROWS, TOP_K) if arm == "scalar" else (ROWS * TOP_K,)
    if switch_shape != expected:
        _fail(f"{arm} layer {layer}: switch route representation {switch_shape} != {expected}")
    for stage in (
        "expert_gate_outputs",
        "expert_up_outputs",
        "expert_activation_outputs",
        "expert_down_outputs",
        "expert_weighted_outputs",
    ):
        validate_canonical_route_values(
            raw_values=_read_float_tensor(root, records[prefix + stage]),
            canonical_values=_read_float_tensor(root, records[prefix + "canonical_" + stage]),
            rank_order=rank_order,
            inverse_order=inverse,
            rows=ROWS,
            top_k=TOP_K,
        )


def _read_float_tensor(root: pathlib.Path, record: dict[str, Any]) -> np.ndarray:
    path = _safe_child(root, record["file"], record["name"])
    shape = tuple(record["shape"])
    return np.fromfile(path, dtype="<f4").reshape(shape)


def validate_manifest(manifest_path: pathlib.Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    root = manifest_path.parent
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        _fail(f"manifest format must be {FORMAT}")
    model = manifest.get("model", {})
    upstream = manifest.get("upstream", {})
    if (model.get("repository"), model.get("revision")) != (MODEL_REPOSITORY, MODEL_REVISION):
        _fail("manifest model repository/revision differs from the pinned checkpoint")
    if (upstream.get("repository"), upstream.get("revision")) != (UPSTREAM_REPOSITORY, UPSTREAM_REVISION):
        _fail("manifest upstream repository/revision differs from the pinned source")
    source_file_count = _verify_source_hashes(manifest)
    capture = manifest.get("capture_contract")
    if not isinstance(capture, dict):
        _fail("manifest capture_contract is missing")
    if capture.get("selected_moe_layers") != list(SELECTED_LAYERS):
        _fail("manifest selected_moe_layers differs from pinned layers 34..39")
    if capture.get("decoder_layer_count") != 40:
        _fail("manifest decoder_layer_count must be 40")
    if capture.get("decoder_boundaries") != ["attention_residual", "hidden"]:
        _fail("manifest decoder boundary contract changed")

    input_meta = manifest.get("input")
    if not isinstance(input_meta, dict):
        _fail("manifest input metadata is missing")
    if (input_meta.get("used_rows"), input_meta.get("width")) != (ROWS, WIDTH):
        _fail("fixture input dimensions must be 158x2048")
    input_shape_bytes = ROWS * WIDTH * 4
    if input_meta.get("used_f32_bytes") != input_shape_bytes:
        _fail("used input byte count differs from 158x2048 F32")
    digest = input_meta.get("used_f32_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or input_meta.get("sha256") != digest:
        _fail("input SHA-256 fields are malformed or inconsistent")
    input_file = _safe_child(root, input_meta.get("used_f32_file"), "used embeddings")
    if not input_file.is_file() or input_file.stat().st_size != input_shape_bytes:
        _fail("used embedding bytes are missing or have the wrong length")
    if sha256_file(input_file) != digest:
        _fail("used embedding SHA-256 mismatch")
    _validate_finite_f32(input_file)

    tolerances = manifest.get("tolerances", {})
    expected_tolerances = {
        "intermediate": (INTERMEDIATE_ATOL, RTOL),
        "endpoint": (ENDPOINT_ATOL, RTOL),
    }
    for name, (atol, rtol) in expected_tolerances.items():
        declared = tolerances.get(name, {})
        if (declared.get("atol"), declared.get("rtol")) != (atol, rtol):
            _fail(f"{name} tolerance differs from the immutable contract")

    arms = manifest.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"scalar", "sequence"}:
        _fail("manifest must contain exactly scalar and sequence arms")
    summary: dict[str, Any] = {
        "format": "drama3-late-fixture-validation-v1",
        "manifest_sha256": sha256_file(manifest_path),
        "source_file_count": source_file_count,
        "input_sha256": digest,
        "rows": ROWS,
        "width": WIDTH,
        "tolerances": {key: {"atol": value[0], "rtol": value[1]} for key, value in expected_tolerances.items()},
        "arms": {},
    }
    for arm in ("scalar", "sequence"):
        arm_obj = arms[arm]
        tensor_list = arm_obj.get("tensors") if isinstance(arm_obj, dict) else None
        if not isinstance(tensor_list, list):
            _fail(f"{arm} arm tensors must be a list")
        expected = expected_tensor_shapes(arm)
        records: dict[str, dict[str, Any]] = {}
        for record in tensor_list:
            if not isinstance(record, dict) or not isinstance(record.get("name"), str):
                _fail(f"{arm} arm contains a malformed tensor record")
            name = record["name"]
            if name in records:
                _fail(f"{arm} arm repeats tensor name {name}")
            if name not in expected:
                _fail(f"{arm} arm has unexpected tensor {name}")
            shape, dtype = expected[name]
            _validate_tensor_record(
                root,
                record,
                expected_name=name,
                expected_shape=shape,
                expected_dtype=dtype,
            )
            records[name] = record
        missing = sorted(set(expected) - set(records))
        if missing:
            _fail(f"{arm} arm missing required tensors: {missing[:8]}")
        for layer in SELECTED_LAYERS:
            _route_check(root, records, arm, layer)
        summary["arms"][arm] = {
            "tensor_count": len(records),
            "boundary_count": 81,
            "selected_moe_layers": list(SELECTED_LAYERS),
            "route_bijections": 6,
        }
    return summary


def _boundary_stages() -> list[str]:
    stages: list[str] = []
    for layer in LAYERS:
        prefix = f"backbone.layers.{layer}."
        stages.extend((prefix + "attention_residual", prefix + "hidden"))
    stages.append("backbone.hidden")
    return stages


def _replay_specs() -> dict[str, dict[str, Any]]:
    """Expected replay matrix and direct upstream operands, independent of JSON order."""
    result: dict[str, dict[str, Any]] = {}

    def add(arm: str, layer: int, kind: str, stage: str, operands: list[tuple[str, str]],
            model_tensors: list[str] | None = None, route_ids: str | None = None) -> None:
        case_id = f"{kind}|{arm}|{layer}|{stage}"
        result[case_id] = {
            "kind": kind,
            "arm": arm,
            "layer": layer,
            "stage": stage,
            "operands": operands,
            "model_tensors": model_tensors or [],
            "route_ids": route_ids,
        }

    for arm in ("scalar", "sequence"):
        for layer in SELECTED_LAYERS:
            prefix = f"backbone.layers.{layer}"
            mlp = prefix + ".mlp."
            norm_weight = prefix + ".post_attention_layernorm.weight"
            down_weight = mlp + "switch_mlp.down_proj.weight"
            down_stem = down_weight[:-7]
            shared_gate_weight = mlp + "shared_expert.gate_proj.weight"
            shared_up_weight = mlp + "shared_expert.up_proj.weight"
            shared_down_weight = mlp + "shared_expert.down_proj.weight"
            add(arm, layer, "post_attention_rms_replay", mlp + "normalized_input",
                [(prefix + ".attention_residual", "upstream_fixture")], [norm_weight])
            add(arm, layer, "expert_down_native", mlp + "canonical_expert_down_outputs",
                [(mlp + "canonical_expert_activation_outputs", "upstream_fixture"),
                 (mlp + "canonical_expert_ids", "upstream_fixture")],
                [down_weight, down_stem + ".scales", down_stem + ".biases"],
                route_ids=mlp + "canonical_expert_ids")
            add(arm, layer, "expert_down_mlx_q4", mlp + "canonical_expert_down_outputs",
                [(mlp + "canonical_expert_activation_outputs", "upstream_fixture"),
                 (mlp + "canonical_expert_ids", "upstream_fixture")],
                [down_weight, down_stem + ".scales", down_stem + ".biases"],
                route_ids=mlp + "canonical_expert_ids")
            for stage, weight in (("shared_gate_output", shared_gate_weight),
                                  ("shared_up_output", shared_up_weight)):
                add(arm, layer, "shared_projection", mlp + stage,
                    [(mlp + "normalized_input", "upstream_fixture")], [weight])
            add(arm, layer, "shared_activation_replay", mlp + "shared_activation_output",
                [(mlp + "shared_gate_output", "upstream_fixture"),
                 (mlp + "shared_up_output", "upstream_fixture")])
            add(arm, layer, "shared_down_replay", mlp + "shared_expert_output",
                [(mlp + "shared_activation_output", "upstream_fixture")], [shared_down_weight])
            add(arm, layer, "shared_weighted_replay", mlp + "shared_weighted_output",
                [(mlp + "shared_expert_output", "upstream_fixture"),
                 (mlp + "shared_gate_logits", "upstream_fixture")])
            add(arm, layer, "moe_sum_add_replay", mlp + "moe_output",
                [(mlp + "routed_sum", "upstream_fixture"),
                 (mlp + "shared_weighted_output", "upstream_fixture")])
            add(arm, layer, "moe_weighted_add_replay", mlp + "moe_output",
                [(mlp + "routed_sum", "upstream_fixture"),
                 (mlp + "shared_expert_output", "upstream_fixture"),
                 (mlp + "shared_gate_logits", "upstream_fixture")])
            whole_weights = [
                mlp + "gate.weight",
                mlp + "switch_mlp.gate_proj.weight",
                mlp + "switch_mlp.up_proj.weight",
                down_weight,
                shared_gate_weight,
                shared_up_weight,
                shared_down_weight,
                mlp + "shared_expert_gate.weight",
            ]
            add(arm, layer, "whole_moe_replay", mlp + "moe_output",
                [(mlp + "normalized_input", "upstream_fixture")], whole_weights)
            add(arm, layer, "residual_upstream_add_control", prefix + ".hidden",
                [(prefix + ".attention_residual", "upstream_fixture"),
                 (mlp + "moe_output", "upstream_fixture")])
            producer = f"whole_moe_replay|{arm}|{layer}|{mlp}moe_output"
            add(arm, layer, "residual_native_moe_replay", prefix + ".hidden",
                [(prefix + ".attention_residual", "upstream_fixture"),
                 (mlp + "native_moe_output", "native_device_result")])
            result[f"residual_native_moe_replay|{arm}|{layer}|{prefix}.hidden"]["producer"] = producer
    return result


def _float32(value: Any, label: str) -> np.float32:
    if not isinstance(value, str):
        _fail(f"{label} must be serialized as a float string")
    if value == "NaN":
        return np.float32(np.nan)
    if value == "+Inf":
        return np.float32(np.inf)
    if value == "-Inf":
        return np.float32(-np.inf)
    try:
        return np.float32(float(value))
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} is not an F32 value: {value!r}") from exc


def _hex_u32(value: Any, label: str) -> int:
    if not isinstance(value, str) or len(value) != 10 or not value.startswith("0x"):
        _fail(f"{label} must be an eight-digit 0x F32 bit pattern")
    try:
        bits = int(value[2:], 16)
    except ValueError as exc:
        raise ValidationError(f"{label} has malformed hex bits") from exc
    if bits < 0 or bits > 0xFFFFFFFF:
        _fail(f"{label} is outside uint32")
    return bits


def _as_f32(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def _same_f32_text_bits(text: Any, bits: int, label: str) -> np.float32:
    value = _float32(text, label)
    decoded = _as_f32(bits)
    if np.isnan(decoded):
        if not np.isnan(value):
            _fail(f"{label} does not agree with its NaN bit pattern")
    elif np.asarray([value], dtype=np.float32).view(np.uint32)[0] != bits:
        _fail(f"{label} does not round-trip to its stated F32 bits")
    return decoded


def _f32_equal(left: np.float32, right: np.float32) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    return int(np.asarray([left], dtype=np.float32).view(np.uint32)[0]) == int(
        np.asarray([right], dtype=np.float32).view(np.uint32)[0]
    )


def _record_value_at(root: pathlib.Path, record: dict[str, Any], index: int, dtype: str) -> Any:
    path = _safe_child(root, record["file"], record["name"])
    offset = index * 4
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(4)
    if len(raw) != 4:
        _fail(f"witness index is outside reference bytes for {record['name']}")
    return np.frombuffer(raw, dtype="<f4" if dtype == "float32" else "<i4")[0]


def _validate_witness(
    witness: Any,
    *,
    label: str,
    case: dict[str, Any],
    reference_record: dict[str, Any],
    fixture_root: pathlib.Path,
    tolerances: tuple[float, float],
    witness_kind: str,
    route_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(witness, dict):
        _fail(f"{label} witness must be an object")
    shape = tuple(case["shape"])
    total = _product(shape)
    index = witness.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < total:
        _fail(f"{label} witness index is outside its tensor")
    actual_bits = _hex_u32(witness.get("actual_bits"), label + ".actual_bits")
    reference_bits = _hex_u32(witness.get("reference_bits"), label + ".reference_bits")
    actual = _same_f32_text_bits(witness.get("actual_f32"), actual_bits, label + ".actual_f32")
    reference = _same_f32_text_bits(witness.get("reference_f32"), reference_bits, label + ".reference_f32")
    raw_reference = _record_value_at(fixture_root, reference_record, index, "float32")
    raw_reference_bits = int(np.asarray([raw_reference], dtype=np.float32).view(np.uint32)[0])
    if reference_bits != raw_reference_bits:
        _fail(f"{label} reference bits do not match the pinned fixture at index {index}")

    coordinates = []
    remainder = index
    for dimension in reversed(shape):
        coordinates.append(remainder % dimension)
        remainder //= dimension
    coordinates.reverse()
    coordinates_obj = witness.get("coordinates")
    expected_coordinates: dict[str, Any] = {"indices": coordinates}
    if coordinates:
        expected_coordinates["row"] = coordinates[0]
    if len(coordinates) >= 3:
        expected_coordinates["rank"] = coordinates[1]
    if len(coordinates) >= 2:
        expected_coordinates["channel"] = coordinates[-1]
    if coordinates_obj != expected_coordinates:
        _fail(f"{label} coordinates do not decode the flat index")

    error = _float32(witness.get("error_f32"), label + ".error_f32")
    allowance = _float32(witness.get("allowance_f32"), label + ".allowance_f32")
    expected_error = np.float32(np.abs(np.float32(actual - reference)))
    expected_allowance = np.float32(
        np.float32(tolerances[0]) + np.float32(tolerances[1]) * np.abs(reference)
    )
    if not _f32_equal(error, expected_error):
        _fail(f"{label} error does not match the witness values")
    if not _f32_equal(allowance, expected_allowance):
        _fail(f"{label} allowance differs from the immutable tolerance")
    if witness_kind == "first_bit" and actual_bits == reference_bits:
        _fail(f"{label} does not show a bit mismatch")
    if witness_kind == "first_tolerance":
        if not np.isfinite(actual) or not np.isfinite(reference) or not error > allowance:
            _fail(f"{label} does not show a strict finite tolerance violation")
    if witness_kind == "first_nonfinite" and np.isfinite(actual) and np.isfinite(reference):
        _fail(f"{label} does not contain a nonfinite value")
    if route_record is not None:
        route_index = index // shape[-1]
        expected_expert = int(_record_value_at(fixture_root, route_record, route_index, "int32"))
        if witness.get("expert_id") != expected_expert:
            _fail(f"{label} selected expert ID differs from canonical route fixture")
    elif "expert_id" in witness:
        _fail(f"{label} has an unexpected expert ID")
    return {"index": index, "coordinates": coordinates_obj, "actual": witness["actual_f32"],
            "reference": witness["reference_f32"], "error": witness["error_f32"],
            "allowance": witness["allowance_f32"],
            **({"expert_id": witness["expert_id"]} if route_record is not None else {})}


def _expected_axes(shape: tuple[int, ...]) -> list[str]:
    if len(shape) == 2:
        return ["row", "channel"]
    if len(shape) == 3:
        return ["row", "rank", "channel"]
    return [f"axis_{index}" for index in range(len(shape))]


def _validate_native_case(
    case: Any,
    *,
    expected: dict[str, Any],
    tensors: dict[str, dict[str, Any]],
    arm: str,
    fixture_root: pathlib.Path,
    fixture_manifest_sha: str,
    input_sha: str,
    model_config_sha: str,
    model_index_sha: str,
    native_identity: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(case, dict):
        _fail("native case must be an object")
    case_id = case.get("id")
    kind, ref_arm, stage = expected["kind"], expected["arm"], expected["stage"]
    if (case_id, case.get("kind"), case.get("ref_arm"), case.get("stage")) != (
        expected["id"], kind, ref_arm, stage
    ):
        _fail(f"case identity mismatch for expected {expected['id']}")
    reference_record = tensors.get(stage)
    if reference_record is None:
        _fail(f"expected reference tensor is absent from fixture: {stage}")
    shape = tuple(reference_record["shape"])
    if case.get("shape") != list(shape) or case.get("axes") != _expected_axes(shape):
        _fail(f"{case_id}: comparison shape/axes differ from fixture")
    endpoint = kind == "full_boundary" and stage == "backbone.hidden"
    expected_atol = np.float32(ENDPOINT_ATOL if endpoint else INTERMEDIATE_ATOL)
    if not _f32_equal(np.float32(case.get("atol")), expected_atol) or not _f32_equal(
        np.float32(case.get("rtol")), np.float32(RTOL)
    ):
        _fail(f"{case_id}: comparison tolerance differs from the immutable contract")
    atol, rtol = float(expected_atol), float(np.float32(RTOL))

    reference = case.get("reference")
    if not isinstance(reference, dict) or (
        reference.get("name"), reference.get("sha256"), reference.get("shape"), reference.get("dtype")
    ) != (stage, reference_record["sha256"], list(shape), "float32"):
        _fail(f"{case_id}: reference record is not the exact upstream fixture tensor")

    operands = case.get("operands")
    if not isinstance(operands, list) or len(operands) != len(expected["operands"]):
        _fail(f"{case_id}: operand list has the wrong length")
    expected_source_uses: dict[tuple[str, str, tuple[int, ...], str], set[str]] = {}
    direct_operand_sources: list[dict[str, Any]] = []

    def add_expected_source(name: str, digest: str, tensor_shape: list[int], source_arm: str, role: str) -> None:
        key = (name, digest, tuple(tensor_shape), source_arm)
        expected_source_uses.setdefault(key, set()).add(role)

    add_expected_source(stage, reference_record["sha256"], list(shape), ref_arm, "reference")
    for index, (operand, (expected_name, expected_origin)) in enumerate(zip(operands, expected["operands"])):
        if not isinstance(operand, dict) or operand.get("name") != expected_name or operand.get("origin") != expected_origin:
            _fail(f"{case_id}: operand {index} does not match expected lineage")
        if expected_origin == "upstream_fixture":
            record = tensors.get(expected_name)
            if record is None:
                _fail(f"{case_id}: upstream operand is absent from fixture: {expected_name}")
            if (operand.get("sha256"), operand.get("shape"), operand.get("ref_arm")) != (
                record["sha256"], record["shape"], ref_arm
            ):
                _fail(f"{case_id}: operand {index} is not bound to its exact fixture record")
            add_expected_source(expected_name, record["sha256"], record["shape"], ref_arm, f"operand:{index}")
            direct_operand_sources.append({"kind": "upstream", "record": record, "role": f"operand:{index}"})
        elif expected_origin == "input_fixture":
            if operand.get("shape") != [ROWS, WIDTH] or operand.get("sha256") != input_sha:
                _fail(f"{case_id}: input fixture operand is not bound to the exact input bytes")
            direct_operand_sources.append({"kind": "input", "index": index, "name": expected_name})
        elif expected_origin == "native_device_result":
            producer = expected["producer"]
            expected_native_shape = list(reference_record["shape"])
            if operand.get("shape") != expected_native_shape or operand.get("produced_by") != producer:
                _fail(f"{case_id}: native operand does not name its exact producer and shape")
            direct_operand_sources.append({"kind": "native", "index": index, "name": expected_name,
                                           "producer": producer, "shape": expected_native_shape})
        else:
            _fail(f"{case_id}: unsupported operand origin {expected_origin}")

    route_record = None
    if expected["route_ids"] is None:
        if case.get("route_ids") is not None:
            _fail(f"{case_id}: unexpected route-ID record")
    else:
        route_name = expected["route_ids"]
        route_record = tensors.get(route_name)
        route_obj = case.get("route_ids")
        if route_record is None or not isinstance(route_obj, dict) or (
            route_obj.get("name"), route_obj.get("sha256"), route_obj.get("shape"), route_obj.get("dtype")
        ) != (route_name, route_record["sha256"], route_record["shape"], "int32"):
            _fail(f"{case_id}: route-ID record is not the exact canonical fixture tensor")
        add_expected_source(route_name, route_record["sha256"], route_record["shape"], ref_arm, "route_ids")

    case_identity = case.get("identity")
    if not isinstance(case_identity, dict):
        _fail(f"{case_id}: case identity is missing")
    for name, value in (
        ("manifest_sha256", fixture_manifest_sha), ("input_sha256", input_sha),
        ("model_config_sha256", model_config_sha), ("model_index_sha256", model_index_sha),
        ("native_chunk", native_identity.get("native_chunk")),
        ("native_graph_depth", native_identity.get("native_graph_depth")),
    ):
        if case_identity.get(name) != value:
            _fail(f"{case_id}: identity field {name} differs from the bound run")

    sources = case_identity.get("source_tensors")
    if not isinstance(sources, list):
        _fail(f"{case_id}: source_tensors identity list is missing")
    actual_upstream: dict[tuple[str, str, tuple[int, ...], str], set[str]] = {}
    special_sources: list[dict[str, Any]] = []
    seen_source_keys: set[tuple[Any, ...]] = set()
    for source in sources:
        if not isinstance(source, dict):
            _fail(f"{case_id}: malformed source_tensors entry")
        origin = source.get("origin")
        if origin == "upstream_fixture":
            source_shape = source.get("shape")
            if not isinstance(source_shape, list):
                _fail(f"{case_id}: upstream source shape must be an array")
            _shape(source_shape, f"{case_id} upstream source")
            digest = source.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                _fail(f"{case_id}: upstream source hash is malformed")
            key = (source.get("name"), digest, tuple(source_shape), source.get("ref_arm"))
            if key in seen_source_keys:
                _fail(f"{case_id}: duplicate coalesced upstream source identity")
            seen_source_keys.add(key)
            if not isinstance(source.get("uses"), list) or any(not isinstance(role, str) for role in source["uses"]):
                _fail(f"{case_id}: malformed upstream source uses")
            actual_upstream[key] = set(source["uses"])
        elif origin in ("input_fixture", "native_device_result"):
            special_sources.append(source)
        else:
            _fail(f"{case_id}: unknown source origin {origin!r}")
    if actual_upstream != expected_source_uses:
        _fail(f"{case_id}: source_tensors does not bind exactly its reference/operands/routes")
    if len(special_sources) != sum(item["kind"] != "upstream" for item in direct_operand_sources):
        _fail(f"{case_id}: special source identity count differs from direct operands")
    for source_info in direct_operand_sources:
        if source_info["kind"] == "upstream":
            continue
        matches = [source for source in special_sources
                   if source.get("name") == source_info["name"]
                   and source.get("operand_index") == source_info["index"]]
        if len(matches) != 1:
            _fail(f"{case_id}: missing or duplicate special source identity")
        source = matches[0]
        common = {
            "manifest_sha256": fixture_manifest_sha,
        }
        if any(source.get(key) != value for key, value in common.items()):
            _fail(f"{case_id}: special source is not bound to this fixture/checkpoint")
        if source.get("shape") != source_info.get("shape", [ROWS, WIDTH]):
            _fail(f"{case_id}: special source shape differs from its direct operand")
        if source_info["kind"] == "native" and (
            source.get("produced_by") != source_info["producer"]
            or source.get("origin") != "native_device_result"
            or source.get("input_sha256") != input_sha
            or source.get("model_config_sha256") != model_config_sha
            or source.get("model_index_sha256") != model_index_sha
        ):
            _fail(f"{case_id}: generated native operand has incorrect producer identity")
        if source_info["kind"] == "input" and (
            source.get("origin") != "input_fixture"
            or source.get("sha256") != input_sha
        ):
            _fail(f"{case_id}: input source is not bound to the exact input fixture")

    model_tensors = case_identity.get("model_tensors")
    expected_weights = expected["model_tensors"]
    if not isinstance(model_tensors, list):
        _fail(f"{case_id}: model_tensors list is missing")
    names = []
    for tensor in model_tensors:
        if not isinstance(tensor, dict):
            _fail(f"{case_id}: malformed model tensor identity")
        name, digest = tensor.get("name"), tensor.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            _fail(f"{case_id}: model tensor name/hash is malformed")
        _shape(tensor.get("shape"), f"{case_id} model tensor {name}")
        if not isinstance(tensor.get("dtype"), str) or not tensor["dtype"]:
            _fail(f"{case_id}: model tensor dtype is missing")
        names.append(name)
    layer = expected.get("layer")
    prefixed_weights = [
        name if not layer or name.startswith("backbone.layers.") else f"backbone.layers.{layer}.{name}"
        for name in expected_weights
    ]
    if names != prefixed_weights:
        _fail(f"{case_id}: model tensor set/order differs from the operation")

    metrics = case.get("metrics")
    if not isinstance(metrics, dict):
        _fail(f"{case_id}: metrics object is missing")
    total = _product(shape)
    metric_names = ("bit_mismatch_count", "tolerance_failure_count", "nonfinite_count")
    counts: dict[str, int] = {}
    for name in metric_names:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= total:
            _fail(f"{case_id}: {name} is outside [0, tensor elements]")
        counts[name] = value
    for name in ("max_abs", "max_ratio"):
        _float32(metrics.get(name), f"{case_id}.{name}")
    witnesses: dict[str, Any] = {}
    for name, count_name in (("first_bit", "bit_mismatch_count"),
                             ("first_tolerance", "tolerance_failure_count"),
                             ("first_nonfinite", "nonfinite_count")):
        witness = metrics.get(name)
        if (witness is None) != (counts[count_name] == 0):
            _fail(f"{case_id}: {name} presence does not match {count_name}")
        if witness is not None:
            witnesses[name] = _validate_witness(
                witness, label=f"{case_id}.{name}", case=case, reference_record=reference_record,
                fixture_root=fixture_root, tolerances=(atol, rtol), witness_kind=name,
                route_record=route_record,
            )
    return {"id": case_id, "kind": kind, "arm": ref_arm, "stage": stage,
            "counts": counts, "witnesses": witnesses}


def _summarize_native_cases(validated: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select earliest cases separately for bit, finite-tolerance, and nonfinite predicates."""
    boundary_order = {stage: index for index, stage in enumerate(_boundary_stages())}
    arm_summary: dict[str, Any] = {}
    for arm in ("scalar", "sequence"):
        ordered = sorted(
            (case for case in validated if case["kind"] == "full_boundary" and case["arm"] == arm),
            key=lambda case: boundary_order[case["stage"]],
        )
        arm_summary[arm] = {
            "first_boundary_bit": next((case for case in ordered if case["counts"]["bit_mismatch_count"]), None),
            "first_boundary_finite_tolerance": next((case for case in ordered if case["counts"]["tolerance_failure_count"]), None),
            "first_boundary_nonfinite": next((case for case in ordered if case["counts"]["nonfinite_count"]), None),
            "boundary_cases_with_bits": sum(case["counts"]["bit_mismatch_count"] > 0 for case in ordered),
            "boundary_cases_with_tolerance_failures": sum(case["counts"]["tolerance_failure_count"] > 0 for case in ordered),
            "boundary_nonfinite_count": sum(case["counts"]["nonfinite_count"] for case in ordered),
        }
    replay_summary: dict[str, Any] = {}
    for arm in ("scalar", "sequence"):
        for kind in NATIVE_KINDS:
            group = sorted(
                (case for case in validated if case["kind"] == kind and case["arm"] == arm),
                key=lambda case: (int(case["stage"].split(".")[2]), case["stage"]),
            )
            replay_summary[f"{arm}:{kind}"] = {
                "first_bit": next((case for case in group if case["counts"]["bit_mismatch_count"]), None),
                "first_finite_tolerance": next((case for case in group if case["counts"]["tolerance_failure_count"]), None),
                "first_nonfinite": next((case for case in group if case["counts"]["nonfinite_count"]), None),
                "failing_layers_by_tolerance": [
                    int(case["stage"].split(".")[2]) for case in group
                    if case["counts"]["tolerance_failure_count"]
                ],
            }
    return arm_summary, replay_summary


def analyze_native_report(native_path: pathlib.Path, manifest_path: pathlib.Path) -> dict[str, Any]:
    """Validate the complete terminal comparator report and summarize first witnesses.

    Boundary results are cumulative full-model observations. Matched replay
    failures are operation-local only for the exact upstream operands named in
    each case; no one replay alone proves a causal history for a boundary drift.
    """
    manifest_path = manifest_path.resolve(strict=True)
    fixture_root = manifest_path.parent
    validate_manifest(manifest_path)
    fixture_manifest = load_json(manifest_path)
    report = load_json(native_path.resolve(strict=True))
    if not isinstance(report, dict) or report.get("format") != NATIVE_FORMAT:
        _fail(f"native report format must be {NATIVE_FORMAT}")
    manifest_binding = report.get("manifest")
    if not isinstance(manifest_binding, dict):
        _fail("native report manifest binding is missing")
    fixture_manifest_sha = sha256_file(manifest_path)
    input_sha = fixture_manifest["input"]["used_f32_sha256"]
    if (manifest_binding.get("sha256"), manifest_binding.get("input_sha256")) != (fixture_manifest_sha, input_sha):
        _fail("native report is bound to a different fixture or input")
    report_fixture_path = manifest_binding.get("path")
    if not isinstance(report_fixture_path, str) or pathlib.Path(report_fixture_path).resolve() != manifest_path:
        _fail("native report manifest path differs from the analyzed fixture")

    identity = report.get("identity")
    if not isinstance(identity, dict):
        _fail("native run identity is missing")
    checkpoint = fixture_manifest["checkpoint"]
    model_config_sha, model_index_sha = checkpoint["config_sha256"], checkpoint["index_sha256"]
    if (identity.get("model_config_sha256"), identity.get("model_index_sha256")) != (
        model_config_sha, model_index_sha
    ):
        _fail("native report model config/index differ from the pinned fixture")
    for name in ("native_chunk", "native_graph_depth"):
        value = identity.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"native run identity {name} is invalid")
    if identity["native_chunk"] <= 0:
        _fail("native chunk must be positive")
    if identity["native_graph_depth"] not in (0, 2):
        _fail("native graph depth is outside the tested contract")
    if report.get("readback_policy") != "single terminal compact metrics read; activations remain device resident":
        _fail("native report readback policy declaration changed")

    arm_tensors = {
        arm: {record["name"]: record for record in fixture_manifest["arms"][arm]["tensors"]}
        for arm in ("scalar", "sequence")
    }
    expected: dict[str, dict[str, Any]] = {}
    for arm in ("scalar", "sequence"):
        for stage in _boundary_stages():
            layer_part = "final" if stage == "backbone.hidden" else stage.split(".")[2]
            case_id = f"full_boundary|{arm}|{layer_part}|{stage}"
            expected[case_id] = {
                "id": case_id, "kind": "full_boundary", "arm": arm, "stage": stage,
                "operands": [("input.used_f32_file", "input_fixture")],
                "model_tensors": [], "route_ids": None,
            }
    for case_id, spec in _replay_specs().items():
        spec = dict(spec)
        spec["id"] = case_id
        expected[case_id] = spec
    cases = report.get("cases")
    if not isinstance(cases, list):
        _fail("native report cases must be a list")
    if len(cases) != 318:
        _fail(f"native report must contain exactly 318 cases, found {len(cases)}")
    actual_ids = [case.get("id") if isinstance(case, dict) else None for case in cases]
    if any(not isinstance(case_id, str) for case_id in actual_ids):
        _fail("every native case must have a string ID")
    if len(set(actual_ids)) != len(actual_ids):
        _fail("native report repeats a case ID")
    if set(actual_ids) != set(expected):
        missing = sorted(set(expected) - set(actual_ids))[:5]
        unexpected = sorted(set(actual_ids) - set(expected), key=str)[:5]
        _fail(f"native case matrix differs; missing={missing}, unexpected={unexpected}")
    if report.get("boundary_count") != 162 or report.get("replay_count") != 156:
        _fail("native report boundary/replay counts are inconsistent with the required matrix")

    validated = []
    for case in cases:
        case_id = case["id"]
        validated.append(_validate_native_case(
            case, expected=expected[case_id], tensors=arm_tensors[expected[case_id]["arm"]],
            arm=expected[case_id]["arm"], fixture_root=fixture_root,
            fixture_manifest_sha=fixture_manifest_sha, input_sha=input_sha,
            model_config_sha=model_config_sha, model_index_sha=model_index_sha,
            native_identity=identity,
        ))

    arm_summary, replay_summary = _summarize_native_cases(validated)
    strict_pass = all(
        case["counts"]["tolerance_failure_count"] == 0 and case["counts"]["nonfinite_count"] == 0
        for case in validated if case["kind"] == "full_boundary"
    )
    if report.get("strict_full_native_pass") is not strict_pass:
        _fail("strict_full_native_pass does not agree with full-boundary tolerance/nonfinite counts")
    readbacks = report.get("readbacks")
    normalized_readbacks = _validated_readback_ledger(readbacks, len(cases))
    for field, key in (("terminal_readback_bytes", "terminal"),
                       ("control_readback_bytes", "control"),
                       ("diagnostic_readback_bytes", "diagnostic")):
        if report.get(field) != normalized_readbacks[key]:
            _fail(f"{field} disagrees with the native readback byte ledger")
    return {
        "format": "drama3-late-native-analysis-v1",
        "manifest_sha256": fixture_manifest_sha,
        "input_sha256": input_sha,
        "native_identity": identity,
        "full_boundary_strict_tolerance_pass": strict_pass,
        "readbacks": normalized_readbacks,
        "arms": arm_summary,
        "replays": replay_summary,
        "interpretation": (
            "Full-boundary first failures are cumulative; replay failures are local to the named exact operands. "
            "Bit mismatch and strict finite-tolerance failure are separate predicates."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path, help="late matched-intermediates manifest.json")
    parser.add_argument("--native", type=pathlib.Path, help="optional native terminal comparison JSON")
    args = parser.parse_args(argv)
    try:
        summary = validate_manifest(args.manifest)
        if args.native is not None:
            summary = {"fixture": summary, "native": analyze_native_report(args.native, args.manifest)}
    except ValidationError as exc:
        print(f"late fixture validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
