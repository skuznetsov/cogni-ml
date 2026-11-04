#!/usr/bin/env python3
"""Compare observed two-block refiner tensors on MPS without a model forward.

Raw comparisons retain the existing tolerance. Rounded views are diagnostic;
they never qualify a connected boundary or fix propagated block errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

try:
    from . import hy_nomas_reference_refiner_compare as prelude
except ImportError:
    import hy_nomas_reference_refiner_compare as prelude

shared = prelude.shared
Error = prelude.Error
SCHEMA = "cogni_ml.hy_nomas_reference_refiner_blocks_compare/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_reference_refiner_blocks_trace/v1"
SOURCE_SCHEMA = "cogni_ml.hy_nomas_reference_refiner_blocks_source_trace/v1"
STAGES = ("condition_silu", "adaLN", "gate_msa", "gate_mlp", "norm1", "qkv",
          "q", "k", "v", "attention", "proj", "residual1", "norm2", "mlp_fc1",
          "mlp_silu", "mlp_fc2", "output")
TENSOR_ORDER = ("block0.input", "c", *(f"block0.{stage}" for stage in STAGES),
                "block1.input", *(f"block1.{stage}" for stage in STAGES))
NATIVE_SOURCE_FILES = (
    *prelude.NATIVE_SOURCE_FILES[:3],
    "examples/support/hy_nomas_reference_refiner_blocks_trace.cr",
    *prelude.NATIVE_SOURCE_FILES[3:11],
    "src/ml/video/hy_nomas_attention_mps.cr",
    *prelude.NATIVE_SOURCE_FILES[11:],
)
SOURCE_ATTENTION_SOURCE_FILES = (
    "src/ml/video/hy_nomas_vector_attention.cr",
    "src/ml/video/kernels/hy_nomas_vector_attention.metal",
)
ATOL, RTOL = prelude.ATOL, prelude.RTOL
BLOCK_CONTROL_FIELD = "block_boundaries_bf16_control"
QKV_CONTROL_FIELD = "qkv_bf16_control"
SOURCE_ATTENTION_CONTROL_FIELD = "source_attention_control"
SOURCE_PROJECTION_RESIDUAL_CONTROL_FIELD = "source_projection_residual_control"
BLOCK_CONTROL_TENSORS = tuple(
    f"block{block}.{stage}" for block in range(2)
    for stage in ("condition_silu", "adaLN", "gate_msa", "gate_mlp", "norm1")
)
QKV_CONTROL_TENSORS = tuple(
    f"block{block}.{stage}" for block in range(2)
    for stage in ("qkv", "q", "k", "v")
)
SOURCE_ATTENTION_TENSORS = ("block0.attention", "block1.attention")
SOURCE_ATTENTION_VALID_COUNT = 263 * 2048
SOURCE_ATTENTION_PADDING_COUNT = (1280 - 263) * 2048
SOURCE_PROJECTION_RESIDUAL_TENSORS = (
    "block0.proj", "block0.residual1", "block1.proj", "block1.residual1")
SOURCE_PROJECTION_RESIDUAL_COUNT = 1280 * 2048


def _controlled_boundaries(native: Mapping[str, Any]) -> tuple[str, ...]:
    if type(native.get(BLOCK_CONTROL_FIELD)) is not bool:
        raise Error("native block-boundary BF16 control must be an explicit bool")
    return BLOCK_CONTROL_TENSORS if native[BLOCK_CONTROL_FIELD] else ()


def _controlled_outputs(native: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    boundaries = _controlled_boundaries(native)
    if type(native.get(QKV_CONTROL_FIELD)) is not bool:
        raise Error("native QKV BF16 control must be an explicit bool")
    if native[QKV_CONTROL_FIELD] and not boundaries:
        raise Error("native QKV BF16 control requires block-boundary BF16 control")
    qkv = QKV_CONTROL_TENSORS if native[QKV_CONTROL_FIELD] else ()
    return boundaries, qkv


def _source_attention_control(native: Mapping[str, Any]) -> bool:
    if SOURCE_ATTENTION_CONTROL_FIELD not in native:
        enabled = False
    else:
        enabled = native[SOURCE_ATTENTION_CONTROL_FIELD]
        if type(enabled) is not bool:
            raise Error("native source-attention control must be a boolean when present")
    if enabled:
        _controlled_outputs(native)
        _require_fields(native, {BLOCK_CONTROL_FIELD: True, QKV_CONTROL_FIELD: True,
                                 "valid_text_tokens": 263, "processed_tokens": 1280},
                        "native source-attention control")
    return enabled


def _source_projection_residual_control(native: Mapping[str, Any],
                                        source_attention_control: bool) -> bool:
    if type(source_attention_control) is not bool:
        raise Error("native source-attention control must be a validated boolean")
    if SOURCE_PROJECTION_RESIDUAL_CONTROL_FIELD not in native:
        enabled = False
    else:
        enabled = native[SOURCE_PROJECTION_RESIDUAL_CONTROL_FIELD]
        if type(enabled) is not bool:
            raise Error("native source projection/residual control must be a boolean when present")
    if enabled and not source_attention_control:
        raise Error("native source projection/residual control requires source-attention control")
    return enabled


def _validate_control_prerequisites(native: Mapping[str, Any],
                                    linked_prelude: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    prelude_controls = prelude._validate_prelude_controls(linked_prelude)
    for name, enabled in prelude_controls.items():
        if not enabled:
            raise Error(f"native block prelude control must also be enabled in the linked trace: {name}")
        _require_fields(native, {name: True}, "native block prelude control")
    return _controlled_outputs(native)


def _controlled_checks_passed(rows: list[dict[str, Any]], selected: tuple[str, ...],
                              kind: str) -> bool | None:
    checked = [row for row in rows if row["kind"] == kind]
    if tuple(row["tensor"] for row in checked) != selected:
        raise Error(f"controlled BF16 coverage differs from the admitted mode: {kind}")
    return all(row["passed"] for row in checked) if selected else None


def _source_attention_checks_passed(rows: list[dict[str, Any]], enabled: bool, *,
                                    kind: str = "source_attention_valid_exact",
                                    region: str = "valid", expected_count: int = SOURCE_ATTENTION_VALID_COUNT
                                    ) -> tuple[bool | None, bool | None]:
    checked = [row for row in rows if row["kind"] == kind]
    expected = tuple((name, region) for name in SOURCE_ATTENTION_TENSORS) if enabled else ()
    actual = tuple((row.get("tensor"), row.get("region")) for row in checked)
    if actual != expected:
        raise Error("source-attention exact coverage differs from the admitted mode")
    if not enabled:
        return None, None

    def zero_metric(value: Any) -> bool:
        return (type(value) in (int, float) and type(value) is not bool
                and math.isfinite(value) and value == 0.0)

    def exact_passed(row: Mapping[str, Any]) -> bool:
        count = row.get("count")
        return (type(row.get("passed")) is bool and row["passed"] is True
                and type(row.get("atol")) is float and row["atol"] == 0.0
                and type(row.get("rtol")) is float and row["rtol"] == 0.0
                and type(count) is int and count == expected_count
                and type(row.get("finite_count")) is int and row["finite_count"] == count
                and type(row.get("mismatch_count")) is int and row["mismatch_count"] == 0
                and zero_metric(row.get("max_abs")) and zero_metric(row.get("mae"))
                and zero_metric(row.get("rms")))

    return exact_passed(checked[0]), exact_passed(checked[1])


def _source_projection_residual_checks_passed(rows: list[dict[str, Any]], enabled: bool
                                              ) -> dict[str, bool] | None:
    kind = "source_projection_residual_full_exact"
    checked = [row for row in rows if row.get("kind") == kind]
    expected = tuple((name, "full") for name in SOURCE_PROJECTION_RESIDUAL_TENSORS) if enabled else ()
    actual = tuple((row.get("tensor"), row.get("region")) for row in checked)
    if actual != expected:
        raise Error("source projection/residual exact coverage differs from the admitted mode")
    if not enabled:
        return None

    def zero_metric(value: Any) -> bool:
        return (type(value) in (int, float) and type(value) is not bool
                and math.isfinite(value) and value == 0.0)

    def exact_passed(row: Mapping[str, Any]) -> bool:
        count = row.get("count")
        return (type(row.get("passed")) is bool and row["passed"] is True
                and type(row.get("atol")) is float and row["atol"] == 0.0
                and type(row.get("rtol")) is float and row["rtol"] == 0.0
                and type(count) is int and count == SOURCE_PROJECTION_RESIDUAL_COUNT
                and type(row.get("finite_count")) is int and row["finite_count"] == count
                and type(row.get("mismatch_count")) is int and row["mismatch_count"] == 0
                and zero_metric(row.get("max_abs")) and zero_metric(row.get("mae"))
                and zero_metric(row.get("rms")))

    return {row["tensor"]: exact_passed(row) for row in checked}


def shapes() -> dict[str, tuple[int, ...]]:
    result = {name: (1, 1280, 2048) for name in TENSOR_ORDER}
    result["c"] = (1, 2048)
    for block in range(2):
        for stage in ("condition_silu", "gate_msa", "gate_mlp"):
            result[f"block{block}.{stage}"] = (1, 2048)
        result[f"block{block}.adaLN"] = (1, 4096)
        result[f"block{block}.qkv"] = (1, 1280, 3, 2048)
        for stage in ("mlp_fc1", "mlp_silu"):
            result[f"block{block}.{stage}"] = (1, 1280, 8192)
    return result


def _require_fields(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        if type(actual.get(key)) is not type(value) or actual[key] != value:
            raise Error(f"{label} contract differs: {key}")


def _validate_layout(doc: Mapping[str, Any]) -> int:
    if doc.get("tensor_order") != list(TENSOR_ORDER) or tuple(doc.get("tensors", {})) != TENSOR_ORDER:
        raise Error("refiner block tensor inventory/order mismatch")
    offset = 0
    for name in TENSOR_ORDER:
        record = doc["tensors"][name]
        expected_shape = list(shapes()[name])
        size = 4 * math.prod(expected_shape)
        if (not isinstance(record, dict) or record.get("dtype") != "F32"
                or record.get("shape") != expected_shape
                or any(type(v) is not int for v in record["shape"])
                or type(record.get("offset")) is not int
                or type(record.get("nbytes")) is not int or record["nbytes"] != size
                or not shared._is_sha256(record.get("sha256"))):
            raise Error(f"invalid refiner block tensor record: {name}")
        if name == "block1.input":
            origin = doc["tensors"]["block0.output"]
            if (record.get("alias_of") != "block0.output"
                    or any(record[key] != origin[key] for key in ("offset", "nbytes", "sha256", "shape", "dtype"))):
                raise Error("refiner inter-block alias differs from block0.output")
        else:
            if record["offset"] != offset or record.get("alias_of") is not None:
                raise Error(f"refiner block tensor range/alias differs: {name}")
            offset += size
    return offset


def _validate_payload(manifest: Path, doc: Mapping[str, Any], *, native: bool) -> bytes:
    extent = _validate_layout(doc)
    artifact = doc.get("tensor_artifact", {}) if native else dict(
        basename=doc.get("payload"), bytes=doc.get("payload_bytes"), sha256=doc.get("payload_sha256"))
    if (artifact.get("basename") != f"{manifest.stem}.tensors.bin"
            or type(artifact.get("bytes")) is not int or artifact["bytes"] != extent
            or not shared._is_sha256(artifact.get("sha256"))):
        raise Error("refiner block payload name/extent/identity mismatch")
    path = shared._canonical_file(manifest.parent / artifact["basename"], "refiner block payload")
    if path.stat().st_size != extent:
        raise Error("refiner block payload extent mismatch")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
        raise Error("refiner block payload SHA-256 mismatch")
    for name in TENSOR_ORDER:
        record = doc["tensors"][name]
        view = memoryview(payload)[record["offset"]:record["offset"] + record["nbytes"]]
        if hashlib.sha256(view).hexdigest() != record["sha256"]:
            raise Error(f"refiner block tensor SHA-256 mismatch: {name}")
    return payload


def _validate_exact_inputs(native: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    for name in ("block0.input", "c"):
        if native["tensors"][name]["sha256"] != source["tensors"][name]["sha256"]:
            raise Error(f"refiner block operands are not byte-identical: {name}")


def _validate_pins(pins: Any, *, source_attention_control: bool = False) -> None:
    expected_files = (NATIVE_SOURCE_FILES + SOURCE_ATTENTION_SOURCE_FILES
                      if source_attention_control else NATIVE_SOURCE_FILES)
    if not isinstance(pins, dict) or tuple(pins) != expected_files:
        raise Error("refiner block compiled source pin inventory mismatch")
    root = Path(__file__).resolve().parents[2]
    for name, digest in pins.items():
        if not shared._is_sha256(digest) or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise Error(f"refiner block compiled source differs from reviewed working tree: {name}")


def _validate_source_contract(source: Mapping[str, Any], linked: Mapping[str, Any], valid: int) -> None:
    _require_fields(source, {"device": "mps", "producer_dtype": "BF16",
                            "full_text_refiner_encode_executed": True,
                            "full_upstream_t0_preprocessing_parity": False,
                            "capture_boundary": "after_block1_output", "blocks_executed": 2},
                    "source block capture")
    _require_fields(source.get("execution", {}), {
        "mode": "source-bf16", "device": "mps", "valid_text_tokens": valid,
        "processed_text_rows": 1280, "padded_text_rows": 1280-valid,
        "refiner_block_forward_calls": 2, "refiner_attention_calls": 2,
        "observed_attention_calls": 2, "observed_gate_calls": 4,
        "dit_block_forward_calls": 0, "loaded_dit_blocks": 0,
        "cache_kv_written": False, "terminal_cpu_exports": 1,
        "unique_payload_tensor_count": 36,
        "source_dtype_by_tensor": dict.fromkeys(TENSOR_ORDER, "BF16"),
        "export_dtype_by_tensor": dict.fromkeys(TENSOR_ORDER, "F32"),
    }, "source block execution")
    source_identity = source.get("source", {})
    linked_identity = linked.get("source", {})
    for key in ("upstream_revision", "upstream_source_sha256", "conditioning_manifest_sha256"):
        if source_identity.get(key) != linked_identity.get(key) or source_identity.get(key) is None:
            raise Error(f"source block identity differs from linked prelude: {key}")
    if source.get("aliases") != {"block1.input": "block0.output"} or source.get("unique_payload_tensor_order") != [
            name for name in TENSOR_ORDER if name != "block1.input"]:
        raise Error("source block unique payload/alias inventory mismatch")
    for name in TENSOR_ORDER:
        checkpoint = {"block0.input": "input_embed", "block1.input": "block0.output"}.get(name, name)
        shape = [1, 1280, 16, 128] if name.endswith((".q", ".k", ".v")) else list(shapes()[name])
        _require_fields(source["tensors"][name], {"source_dtype": "BF16", "source_shape": shape,
                                                  "source_checkpoint": checkpoint}, f"source checkpoint {name}")


def _validate_prefix_links(doc: Mapping[str, Any], payload: bytes, linked: Mapping[str, Any], valid: int,
                           links: tuple[tuple[str, str], ...]) -> None:
    # Byte hashing is metadata authentication, not CPU tensor arithmetic.
    for name, linked_name in links:
        if linked_name not in linked["tensors"]:
            raise Error(f"linked refiner checkpoint is missing: {linked_name}")
        record = doc["tensors"][name]
        prefix = memoryview(payload)[record["offset"]:record["offset"]+4*valid*2048]
        if hashlib.sha256(prefix).hexdigest() != linked["tensors"][linked_name]["sha256"]:
            raise Error(f"refiner block valid prefix is detached from linked capture: {name}")


def validate_inputs(args: Any) -> dict[str, Any]:
    # Reuse the stricter existing input, source revision, writer-call, mask and
    # prelude-control admission instead of trusting newly supplied labels.
    prelude_args = SimpleNamespace(**{**vars(args), "native": args.native_refiner,
                                     "native_sha256": args.native_refiner_sha256,
                                     "source": args.source_refiner,
                                     "source_sha256": args.source_refiner_sha256})
    admitted = prelude.validate_inputs(prelude_args)
    if admitted["source_mode"] != "source-bf16":
        raise Error("refiner block source must be the authenticated BF16 control")
    npth, native = prelude._load_document(args.native, args.native_sha256, NATIVE_SCHEMA)
    spth, source = prelude._load_document(args.source, args.source_sha256, SOURCE_SCHEMA)
    valid = admitted["native_document"]["valid_text_tokens"]
    common = dict(processed_tokens=1280, valid_text_tokens=valid, padding_tokens=1280-valid)
    _require_fields(native, {**common, "device": "mps", "producer_dtype": "F32",
                            "full_text_refiner_encode_executed": True, "blocks_executed": 2,
                            "full_upstream_t0_preprocessing_parity": False,
                            "capture_boundary": "after_block1_output",
                            "encode_allocations": 0, "encode_readbacks": 0,
                            "preterminal_waits": 0, "terminal_waits": 1,
                            "prelude_trace_manifest_sha256": args.native_refiner_sha256,
                            "cache_fixture_manifest_sha256": args.native_preblock_sha256,
                            "timeline_manifest_sha256": args.timeline_sha256,
                            "reference_latent_artifact_sha256": args.reference_latent_sha256}, "native refiner block")
    _validate_control_prerequisites(native, admitted["native_document"])
    source_attention_control = _source_attention_control(native)
    _source_projection_residual_control(native, source_attention_control)
    _validate_pins(native.get("compiled_source_sha256"),
                   source_attention_control=source_attention_control)
    _require_fields(source, {**common, "source_refiner_manifest_sha256": args.source_refiner_sha256,
                            "source_refiner_payload_sha256": admitted["source_document"]["payload_sha256"]},
                    "source refiner block")
    native_payload = _validate_payload(npth, native, native=True)
    source_payload = _validate_payload(spth, source, native=False)
    _validate_source_contract(source, admitted["source_document"], valid)
    _validate_exact_inputs(native, source)
    # c is a full checkpoint in both manifests and must be the same consumed
    # buffer bytes, not an independently generated agreeing value.
    for kind, blocks, linked in (("native", native, admitted["native_document"]),
                                 ("source", source, admitted["source_document"])):
        if blocks["tensors"]["c"]["sha256"] != linked["tensors"]["c"]["sha256"]:
            raise Error(f"{kind} refiner block conditioning is detached from linked prelude")
    _validate_prefix_links(native, native_payload, admitted["native_document"], valid,
                           (("block0.input", "input_embed"),))
    native_preblock, _ = shared._read_unique_json(Path(args.native_preblock), "native preblock manifest")
    _validate_prefix_links(native, native_payload, native_preblock, valid,
                           (("block1.output", "block.txt_valid_f32"),))
    _validate_prefix_links(source, source_payload, admitted["source_document"], valid,
                           (("block0.input", "input_embed"), ("block1.output", "block.txt_valid_f32")))
    return dict(native_document=native, source_document=source, native_payload=native_payload,
                source_payload=source_payload, valid_text_tokens=valid,
                prelude_admission=admitted)


def first_divergence(rows: list[dict[str, Any]], region: str) -> str | None:
    by_name = {row["tensor"]: row for row in rows if row["kind"] == "raw" and row["region"] == region}
    return next((name for name in TENSOR_ORDER if name in by_name and not by_name[name]["passed"]), None)


def _run_mps(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    if not torch.backends.mps.is_available():
        raise Error("refiner block comparison requires MPS")
    device = torch.device("mps")
    queued = []
    controlled_boundaries, controlled_qkv = _controlled_outputs(inputs["native_document"])
    source_attention_control = _source_attention_control(inputs["native_document"])
    source_projection_residual_control = _source_projection_residual_control(
        inputs["native_document"], source_attention_control)

    def compare(kind: str, name: str, region: str, actual: Any, expected: Any,
                *, atol: float = ATOL, rtol: float = RTOL) -> None:
        queued.append((kind, name, region, shared._metric_stat_tensors(
            torch, actual, expected, atol=atol, rtol=rtol), atol, rtol))

    for name in TENSOR_ORDER:
        actual, expected = [shared._load_tensor(torch, np, inputs[f"{kind}_payload"],
                           inputs[f"{kind}_document"]["tensors"][name], device)
                            for kind in ("native", "source")]
        regions = {"full": (actual, expected)}
        if actual.ndim >= 3:
            valid = inputs["valid_text_tokens"]
            regions["valid"] = (actual[:, :valid], expected[:, :valid])
            if valid < 1280:
                regions["padding"] = (actual[:, valid:], expected[:, valid:])
        for region, (left, right) in regions.items():
            compare("raw", name, region, left, right)
            compare("native_bf16_rounding_view", name, region, left.bfloat16().float(), right)
        compare("source_bf16_representation", name, "full", expected,
                expected.bfloat16().float(), atol=0.0, rtol=0.0)
        if name in controlled_boundaries:
            # Check the captured connected output itself. The diagnostic
            # rounded-vs-source view above cannot establish this property.
            compare("native_controlled_bf16_representation", name, "full", actual,
                    actual.bfloat16().float(), atol=0.0, rtol=0.0)
            compare("controlled_boundary_source_exact", name, "full", actual,
                    expected, atol=0.0, rtol=0.0)
        if name in controlled_qkv:
            # These are the actual captured fused and split Q/K/V outputs;
            # retain evidence independently from the older boundary controls.
            compare("native_controlled_qkv_bf16_representation", name, "full", actual,
                    actual.bfloat16().float(), atol=0.0, rtol=0.0)
            compare("controlled_qkv_source_exact", name, "full", actual,
                    expected, atol=0.0, rtol=0.0)
        if source_attention_control and name in SOURCE_ATTENTION_TENSORS:
            actual_valid, expected_valid = regions["valid"]
            compare("source_attention_valid_exact", name, "valid", actual_valid,
                    expected_valid, atol=0.0, rtol=0.0)
            actual_padding, _expected_padding = regions["padding"]
            compare("source_attention_padding_zero_exact", name, "padding", actual_padding,
                    torch.zeros_like(actual_padding), atol=0.0, rtol=0.0)
        if source_projection_residual_control and name in SOURCE_PROJECTION_RESIDUAL_TENSORS:
            compare("source_projection_residual_full_exact", name, "full", actual,
                    expected, atol=0.0, rtol=0.0)
    control = torch.arange(16, device=device, dtype=torch.float32)
    compare("sensitivity_control", "a_vs_a", "full", control, control)
    compare("sensitivity_control", "deliberate_perturbation", "full", control+0.01, control)
    non_bf16 = control + 0.001
    compare("sensitivity_control", "non_bf16_representation", "full", non_bf16,
            non_bf16.bfloat16().float(), atol=0.0, rtol=0.0)
    flat = [stats[key] for _kind, _name, _region, stats, _atol, _rtol in queued for key in shared.METRIC_KEYS]
    # All neural comparisons and reductions precede this one terminal export.
    values = torch.stack(flat).detach().cpu().tolist()
    stride = len(shared.METRIC_KEYS)
    rows = [dict(kind=kind, tensor=name, region=region, **shared._stats_record(
        values[i*stride:(i+1)*stride], atol=atol, rtol=rtol))
        for i, (kind, name, region, _stats, atol, rtol) in enumerate(queued)]
    raw = [row for row in rows if row["kind"] == "raw" and row["region"] == "full"]
    representation = [row for row in rows if row["kind"] == "source_bf16_representation"]
    controls = {row["tensor"]: row for row in rows if row["kind"] == "sensitivity_control"}
    source_attention_block0_passed, source_attention_block1_passed = \
        _source_attention_checks_passed(rows, source_attention_control)
    source_attention_block0_padding_passed, source_attention_block1_padding_passed = \
        _source_attention_checks_passed(rows, source_attention_control,
            kind="source_attention_padding_zero_exact", region="padding",
            expected_count=SOURCE_ATTENTION_PADDING_COUNT)
    source_projection_residual_stage_passed = _source_projection_residual_checks_passed(
        rows, source_projection_residual_control)
    return dict(numeric_gate_passed=all(row["passed"] for row in raw),
                source_bf16_representation_passed=all(row["passed"] for row in representation),
                block_boundaries_bf16_control=inputs["native_document"][BLOCK_CONTROL_FIELD],
                controlled_boundary_tensors=list(controlled_boundaries),
                controlled_boundary_count=len(controlled_boundaries),
                native_controlled_bf16_representation_passed=_controlled_checks_passed(
                    rows, controlled_boundaries, "native_controlled_bf16_representation"),
                controlled_boundary_source_exact_passed=_controlled_checks_passed(
                    rows, controlled_boundaries, "controlled_boundary_source_exact"),
                qkv_bf16_control=inputs["native_document"][QKV_CONTROL_FIELD],
                controlled_qkv_tensors=list(controlled_qkv),
                controlled_qkv_count=len(controlled_qkv),
                native_controlled_qkv_bf16_representation_passed=_controlled_checks_passed(
                    rows, controlled_qkv, "native_controlled_qkv_bf16_representation"),
                controlled_qkv_source_exact_passed=_controlled_checks_passed(
                    rows, controlled_qkv, "controlled_qkv_source_exact"),
                source_attention_control=source_attention_control,
                source_attention_block0_exact_passed=source_attention_block0_passed,
                source_attention_block1_exact_passed=source_attention_block1_passed,
                source_attention_block0_padding_zero_passed=source_attention_block0_padding_passed,
                source_attention_block1_padding_zero_passed=source_attention_block1_padding_passed,
                source_attention_exact_passed=(None if not source_attention_control else
                    source_attention_block0_passed is True and source_attention_block1_passed is True
                    and source_attention_block0_padding_passed is True
                    and source_attention_block1_padding_passed is True),
                source_projection_residual_control=source_projection_residual_control,
                source_projection_residual_stage_exact_passed=source_projection_residual_stage_passed,
                source_projection_residual_exact_passed=(None if source_projection_residual_stage_passed is None
                    else all(source_projection_residual_stage_passed.values())),
                first_divergence={region: first_divergence(rows, region) for region in ("full", "valid", "padding")},
                first_divergence_order="canonical dependency order; gates follow their adaLN producer, not hook observation time",
                same_input_bytes=True, terminal_cpu_exports=1,
                sensitivity_controls_passed=(controls["a_vs_a"]["passed"]
                    and not controls["deliberate_perturbation"]["passed"]
                    and not controls["non_bf16_representation"]["passed"]),
                rows=rows,
                interpretation="One authenticated finite T0 V263 input only. Source uses pinned PyTorch MPS attention adaptation. Full/valid/padding raw rows are separate; post-rounded views are diagnostic and never qualify connected arithmetic. When enabled, source-attention control adds exact valid-output comparisons for block 0 and block 1 independently and exact zero-padding checks for both native attention outputs; it makes no attention-internal or tail-probability claim. When source-projection/residual control is enabled, exact full-tensor comparisons cover the outputs after block 0 projection, block 0 first residual, block 1 projection, and block 1 first residual independently. Native controlled representation rows check the actual captured outputs, not source parity. Block1 errors may propagate from block0; this capture does not isolate block1. No logits/softmax internals, DiT K/V, motion, performance, or full generation qualification follows.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "native-sha256", "source", "source-sha256", "native-refiner",
                 "native-refiner-sha256", "source-refiner", "source-refiner-sha256",
                 "native-preblock", "native-preblock-sha256", "reference-latent",
                 "reference-latent-sha256", "timeline-sha256", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        output = shared.output_path(args.output)
        inputs = validate_inputs(args)
        result = _run_mps(inputs)
        document = dict(schema=SCHEMA, device="mps", atol=ATOL, rtol=RTOL,
                        native_manifest_sha256=args.native_sha256, source_manifest_sha256=args.source_sha256,
                        native_refiner_manifest_sha256=args.native_refiner_sha256,
                        source_refiner_manifest_sha256=args.source_refiner_sha256,
                        timeline_manifest_sha256=args.timeline_sha256, **result)
        shared._publish_fresh(output, document)
    except (Error, shared.native_cache.ReferenceCacheError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps(dict(schema=SCHEMA, status="rejected", reason=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(dict(schema=SCHEMA, output=str(output), numeric_gate_passed=result["numeric_gate_passed"],
                          first_divergence=result["first_divergence"],
                          sensitivity_controls_passed=result["sensitivity_controls_passed"],
                          block_boundaries_bf16_control=result["block_boundaries_bf16_control"],
                          controlled_boundary_source_exact_passed=result[
                              "controlled_boundary_source_exact_passed"],
                          qkv_bf16_control=result["qkv_bf16_control"],
                          controlled_qkv_source_exact_passed=result["controlled_qkv_source_exact_passed"],
                          source_attention_control=result["source_attention_control"],
                          source_attention_block0_exact_passed=result["source_attention_block0_exact_passed"],
                          source_attention_block1_exact_passed=result["source_attention_block1_exact_passed"],
                          source_attention_block0_padding_zero_passed=result[
                              "source_attention_block0_padding_zero_passed"],
                          source_attention_block1_padding_zero_passed=result[
                              "source_attention_block1_padding_zero_passed"],
                          source_attention_exact_passed=result["source_attention_exact_passed"],
                          source_projection_residual_control=result[
                              "source_projection_residual_control"],
                          source_projection_residual_stage_exact_passed=result[
                              "source_projection_residual_stage_exact_passed"],
                          source_projection_residual_exact_passed=result[
                              "source_projection_residual_exact_passed"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
