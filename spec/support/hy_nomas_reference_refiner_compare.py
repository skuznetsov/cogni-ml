#!/usr/bin/env python3
"""Compare captured native/source refiner boundaries without a model forward.

Raw comparisons qualify only the captured boundary. A BF16-rounded view of a
native F32 result is diagnostic, since preceding native operands may differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from . import hy_nomas_reference_preblock_compare as shared
    from . import hy_nomas_reference_refiner_trace as source_profile
except ImportError:
    import hy_nomas_reference_preblock_compare as shared
    import hy_nomas_reference_refiner_trace as source_profile

SCHEMA = "cogni_ml.hy_nomas_reference_refiner_compare/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_reference_refiner_trace/v1"
SOURCE_SCHEMA = "cogni_ml.hy_nomas_reference_refiner_source_trace/v1"
COMMON_ORDER = (
    "input_compact_valid", "masked_mean", "timestep",
    "refiner_timestep_sinusoidal", "refiner_timestep_mlp1",
    "refiner_timestep_silu", "refiner_timestep", "context_linear1",
    "context_silu", "context_embed", "c", "input_embed",
    "time_in.sinusoidal", "time_in.mlp1", "time_in.silu", "vec",
)
RAW_SINUSOIDS = ("refiner_timestep_sinusoidal_raw", "time_in.sinusoidal_raw")
SOURCE_ONLY = (*RAW_SINUSOIDS, "refiner_mask")
SOURCE_ORDER = (*shared.TENSOR_ORDER, *COMMON_ORDER, *SOURCE_ONLY)
NATIVE_SOURCE_FILES = (
    "examples/hy_nomas_reference_cache_probe.cr",
    "examples/support/hy_nomas_reference_cache_probe.cr",
    "examples/support/hy_nomas_reference_refiner_trace.cr",
    "src/ml/video/hy_nomas_text_refiner.cr",
    "src/ml/metal/kernels/hy_nomas/hy_nomas_text_refiner.metal",
    "src/ml/video/metal_ops.cr", "src/ml/video/metal_linear_mps.cr",
    "src/ml/metal/kernels/shared/linear_mps.metal",
    "src/ml/video/hy_nomas_mps_activation.cr",
    "src/ml/video/hy_nomas_mps_reductions_bridge.mm",
    "src/ml/metal/kernels/hy_nomas/common.metal",
    "src/ml/video/wan_t5.cr",
    "src/ml/metal/kernels/shared/wan_t5.metal",
    "src/ml/video/wan_t5_mps_bridge.mm",
)
ATOL = shared.ATOL
RTOL = shared.RTOL
Error = shared.PreblockCompareError


def shapes(valid: int) -> dict[str, tuple[int, ...]]:
    if type(valid) is not int or not 1 <= valid <= 1280:
        raise Error("valid text prefix must be in [1,1280]")
    result = {name: (1, 2048) for name in COMMON_ORDER}
    result.update({
        "input_compact_valid": (1, valid, 3584),
        "masked_mean": (1, 3584), "timestep": (1,),
        "refiner_mask": (1, 1280),
        "input_embed": (1, valid, 2048),
        **{name: (1, 256) for name in (
            "refiner_timestep_sinusoidal", "time_in.sinusoidal", *RAW_SINUSOIDS
        )},
    })
    return result


def _load_document(path: str | Path, expected_sha: str, schema: str) -> tuple[Path, dict[str, Any]]:
    canonical = shared._canonical_file(path, "refiner trace manifest")
    document, raw = shared._read_unique_json(canonical, "refiner trace manifest")
    if not shared._is_sha256(expected_sha) or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise Error("refiner trace manifest SHA-256 mismatch")
    if document.get("schema") != schema or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise Error("refiner trace schema/version mismatch")
    return canonical, document


def _validate_payload(
    manifest: Path, doc: Mapping[str, Any], *, native: bool,
) -> bytes:
    order = COMMON_ORDER if native else SOURCE_ORDER
    if doc.get("tensor_order") != list(order) or tuple(doc.get("tensors", {})) != order:
        raise Error("refiner trace tensor inventory/order mismatch")
    valid = doc.get("valid_text_tokens")
    expected_shapes = shapes(valid)
    if not native:
        expected_shapes.update(shared._expected_source_shapes(valid))
    offset = 0
    for name in order:
        record = doc["tensors"][name]
        dtype = "BF16" if name == "reference.latent_bf16" else "F32"
        nbytes = math.prod(expected_shapes[name]) * (2 if dtype == "BF16" else 4)
        if (not isinstance(record, dict) or record.get("dtype") != dtype
                or record.get("shape") != list(expected_shapes[name])
                or any(type(v) is not int for v in record["shape"])
                or type(record.get("offset")) is not int or record["offset"] != offset
                or type(record.get("nbytes")) is not int or record["nbytes"] != nbytes
                or not shared._is_sha256(record.get("sha256"))):
            raise Error(f"invalid refiner tensor record: {name}")
        offset += nbytes
    artifact = doc.get("tensor_artifact", {}) if native else {
        "basename": doc.get("payload"), "bytes": doc.get("payload_bytes"),
        "sha256": doc.get("payload_sha256"),
    }
    if (artifact.get("basename") != f"{manifest.stem}.tensors.bin"
            or type(artifact.get("bytes")) is not int or artifact["bytes"] != offset
            or not shared._is_sha256(artifact.get("sha256"))):
        raise Error("refiner tensor artifact extent/name/identity mismatch")
    payload_path = shared._canonical_file(manifest.parent / artifact["basename"], "refiner payload")
    if payload_path.stat().st_size != offset:
        raise Error("refiner payload byte extent mismatch")
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
        raise Error("refiner payload SHA-256 mismatch")
    for name in order:
        record = doc["tensors"][name]
        view = memoryview(payload)[record["offset"]:record["offset"] + record["nbytes"]]
        if hashlib.sha256(view).hexdigest() != record["sha256"]:
            raise Error(f"refiner tensor SHA-256 mismatch: {name}")
    if native:
        first, second = (doc["tensors"][name] for name in (
            "refiner_timestep_sinusoidal", "time_in.sinusoidal"
        ))
        if second.get("alias_of") != "refiner_timestep_sinusoidal" or first["sha256"] != second["sha256"]:
            raise Error("native shared sinusoidal alias differs")
    return payload


def _validate_exact_mean_operands(native: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    """Bind replay to actual exported operands, independently of the tolerance gate."""
    if native["tensors"]["input_compact_valid"]["sha256"] != source["tensors"]["input_compact_valid"]["sha256"]:
        raise Error("mean replay requires byte-identical consumed Qwen inputs")
    valid = native["valid_text_tokens"]
    prefix = b"\x00\x00\x80\x3f" * valid + b"\x00\x00\x00\x00" * (1280 - valid)
    if source["tensors"]["refiner_mask"]["sha256"] != hashlib.sha256(prefix).hexdigest():
        raise Error("mean replay requires the actual source refiner mask to be the authenticated valid prefix")


def _validate_compiled_pins(pins: Any) -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in NATIVE_SOURCE_FILES}
    if pins != expected:
        raise Error("native refiner compiled source pins differ from the reviewed working tree")


def _validate_prelude_controls(document: Mapping[str, Any]) -> dict[str, bool]:
    controls = {name: document.get(name) for name in (
        "prelude_input_bf16_control", "prelude_linear_bf16_control",
        "prelude_silu_bf16_control", "prelude_second_linear_bf16_control",
        "prelude_add_bf16_control")}
    if any(type(value) is not bool for value in controls.values()):
        raise Error("all five native refiner precision controls must be present and boolean")
    if document.get("full_bf16_prelude_parity") is not False:
        raise Error("native full BF16 prelude parity must remain explicitly false")
    if controls["prelude_linear_bf16_control"] and not controls["prelude_input_bf16_control"]:
        raise Error("linear BF16 control requires input BF16 control")
    if controls["prelude_silu_bf16_control"] and not controls["prelude_linear_bf16_control"]:
        raise Error("SiLU BF16 control requires linear BF16 control")
    if controls["prelude_second_linear_bf16_control"] and not controls["prelude_silu_bf16_control"]:
        raise Error("second linear BF16 control requires SiLU BF16 control")
    if controls["prelude_add_bf16_control"] and not controls["prelude_second_linear_bf16_control"]:
        raise Error("addition BF16 control requires second linear BF16 control")
    return controls


def _controlled_boundaries(controls: Mapping[str, bool]) -> tuple[str, ...]:
    """List only consumed boundaries selected by validated cumulative controls."""
    names = []
    if controls["prelude_input_bf16_control"]:
        names.extend(("masked_mean", "refiner_timestep_sinusoidal", "time_in.sinusoidal"))
    if controls["prelude_linear_bf16_control"]:
        names.extend(("refiner_timestep_mlp1", "context_linear1", "input_embed", "time_in.mlp1"))
    if controls["prelude_silu_bf16_control"]:
        names.extend(("refiner_timestep_silu", "context_silu", "time_in.silu"))
    if controls["prelude_second_linear_bf16_control"]:
        names.extend(("refiner_timestep", "context_embed", "vec"))
    if controls["prelude_add_bf16_control"]:
        names.append("c")
    return tuple(names)


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    native_path, native = _load_document(args.native, args.native_sha256, NATIVE_SCHEMA)
    source_path, source = _load_document(args.source, args.source_sha256, SOURCE_SCHEMA)
    native_payload = _validate_payload(native_path, native, native=True)
    source_payload = _validate_payload(source_path, source, native=False)
    validation = shared.native_cache.validate_reference_cache_manifest(
        args.native_preblock, reference_latent_path=args.reference_latent,
        expected_reference_latent_artifact_sha256=args.reference_latent_sha256,
        expected_timeline_manifest_sha256=args.timeline_sha256,
        expected_manifest_sha256=args.native_preblock_sha256,
    )
    if (native.get("cache_fixture_manifest_sha256") != validation["manifest_sha256"]
            or native.get("timeline_manifest_sha256") != args.timeline_sha256
            or native.get("reference_latent_artifact_sha256") != args.reference_latent_sha256
            or native.get("valid_text_tokens") != validation["valid_text_tokens"]
            or source.get("valid_text_tokens") != native["valid_text_tokens"]):
        raise Error("native/source refiner input identities differ")
    for key, expected in {
        "device": "mps", "producer_dtype": "F32",
        "full_upstream_t0_preprocessing_parity": False,
        "encode_allocations": 0, "encode_readbacks": 0,
        "preterminal_waits": 0, "terminal_waits": 1,
    }.items():
        if type(native.get(key)) is not type(expected) or native[key] != expected:
            raise Error(f"native refiner execution contract differs: {key}")
    _validate_prelude_controls(native)
    _validate_compiled_pins(native.get("compiled_source_sha256"))
    identity = source.get("source", {})
    if (identity.get("conditioning_manifest_sha256") != args.timeline_sha256
            or source["tensors"]["reference.latent_bf16"]["sha256"] != validation["reference_latent_tensor_sha256"]):
        raise Error("source refiner timeline/clean latent differs from native")
    if set(identity) != shared._SOURCE_FIELDS or not all(
        shared._is_sha256(value) for key, value in identity.items()
        if key not in ("structured_control_schema", "upstream_revision", "upstream_source_sha256")
    ):
        raise Error("source refiner authenticated identities are incomplete")
    native_document, _ = shared._read_unique_json(Path(args.native_preblock), "native preblock manifest")
    if (identity.get("structured_control_schema") != "cogni_ml.hy_nomas_structured_reference_control/v1"
            or identity.get("upstream_revision") != native_document["preprocessing"]["timeline_upstream_revision"]
            or not isinstance(identity.get("upstream_source_sha256"), dict)
            or not identity["upstream_source_sha256"]
            or not all(shared._is_sha256(digest) for digest in identity["upstream_source_sha256"].values())):
        raise Error("source refiner upstream pins differ from the authenticated native timeline")
    execution = source.get("execution", {})
    mode = execution.get("mode")
    if mode not in shared.SOURCE_SCOPES or execution.get("device") != "mps":
        raise Error("source refiner precision/device differs")
    for key, expected in {
        "loaded_dit_blocks": 0, "block_weight_tensor_loads": 0,
        "block_forward_calls": 0, "first_block_boundary_calls": 1,
        "terminal_cpu_exports": 1, "cache_kv_written": False,
        "observed_stage_order": ["time_in", "txt_in", "img_in", "block0"],
        "block_loader_calls": 1,
        "t0_valid_text_tokens": native["valid_text_tokens"],
    }.items():
        if type(execution.get(key)) is not type(expected) or execution[key] != expected:
            raise Error(f"source refiner execution contract differs: {key}")
    writer = execution.get("writer_call", {})
    expected_writer = dict(kind="writer", segment=0, step=None, condition="t0_caption",
                           valid_text_tokens=native["valid_text_tokens"], cache_valid_len=0,
                           update_cache=True, cache_txt=True, rope_bank=0, timestep=206.897)
    if writer != expected_writer:
        raise Error("source refiner writer differs from the authenticated T0 forward")
    expected_types = source_profile._all_source_types(mode == "source-f32-precision-control")
    if (execution.get("source_dtype_by_tensor") != expected_types
            or execution.get("export_dtype_by_tensor") != {
                name: ("BF16" if name == "reference.latent_bf16" else "F32") for name in SOURCE_ORDER
            }):
        raise Error("source refiner consumed/exported dtypes differ from its precision profile")
    trace = execution.get("refiner_trace", {})
    if (trace.get("event_order") != list(source_profile.TRACE_EVENT_ORDER)
            or trace.get("common_tensor_order") != list(COMMON_ORDER)
            or trace.get("source_only_tensor_order") != list(SOURCE_ONLY)
            or type(trace.get("valid_prefix_tokens")) is not int
            or trace["valid_prefix_tokens"] != native["valid_text_tokens"]
            or trace.get("full_text_refiner_encode_executed") is not True
            or trace.get("text_refiner_entry_calls") != 1
            or trace.get("source_forward_computations_replayed") is not False):
        raise Error("source refiner observational trace contract differs")
    if mode == "source-f32-precision-control" and execution.get("prelude_precision_control") != shared._PRELUDE_F32_CONTROL:
        raise Error("source refiner F32 control does not preserve the selected precision contract")
    _validate_exact_mean_operands(native, source)
    return dict(native_document=native, source_document=source,
                native_payload=native_payload, source_payload=source_payload,
                native_validation=validation, source_mode=mode)


def _mean_replay(torch: Any, compact: Any, valid: int) -> dict[str, Any]:
    """Replay the source's padded BF16 expression without host tensor reads."""
    full = torch.cat((compact.to(torch.bfloat16), torch.zeros(
        (1, 1280 - valid, compact.shape[-1]), device=compact.device, dtype=torch.bfloat16)), dim=1)
    mask = torch.cat((torch.ones((1, valid, 1), device=compact.device, dtype=torch.bfloat16),
                      torch.zeros((1, 1280-valid, 1), device=compact.device, dtype=torch.bfloat16)), dim=1)
    numerator = (full * mask).sum(dim=1)
    denominator = mask.sum(dim=1)
    return dict(source_bf16_expression=numerator / denominator,
                native_f32_expression=compact.sum(dim=1) / valid,
                round_sum_divide_f32_count=(numerator.float() / valid).to(torch.bfloat16).float(),
                denominator=denominator)


def _run_mps(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    if not torch.backends.mps.is_available():
        raise Error("refiner comparison requires MPS")
    device = torch.device("mps")
    ndoc, sdoc = inputs["native_document"], inputs["source_document"]
    tensors: dict[str, dict[str, Any]] = {}
    for kind, doc, payload in (("native", ndoc, inputs["native_payload"]),
                               ("source", sdoc, inputs["source_payload"])):
        tensors[kind] = {name: shared._load_tensor(torch, np, payload, doc["tensors"][name], device)
                         for name in COMMON_ORDER}
    queued: list[tuple[str, str, dict[str, Any], float, float]] = []

    def compare(kind: str, name: str, actual: Any, expected: Any,
                *, atol: float = ATOL, rtol: float = RTOL) -> None:
        queued.append((kind, name, shared._metric_stat_tensors(
            torch, actual, expected, atol=atol, rtol=rtol), atol, rtol))

    for name in COMMON_ORDER:
        actual, expected = tensors["native"][name], tensors["source"][name]
        compare("raw", name, actual, expected)
        if name not in ("input_compact_valid", "timestep"):
            compare("native_bf16_rounding_view", name,
                    actual.to(torch.bfloat16).to(torch.float32), expected)
    # A flag must describe the captured connected values, not just intent or
    # a post-rounded diagnostic view. Use zero tolerance for this contract.
    controls_enabled = _validate_prelude_controls(ndoc)
    for name in _controlled_boundaries(controls_enabled):
        actual = tensors["native"][name]
        compare("connected_bf16_boundary", name, actual,
                actual.to(torch.bfloat16).float(), atol=0.0, rtol=0.0)
    for name, raw_name in zip(("refiner_timestep_sinusoidal", "time_in.sinusoidal"), RAW_SINUSOIDS):
        raw = shared._load_tensor(torch, np, inputs["source_payload"], sdoc["tensors"][raw_name], device)
        compare("source_raw_sinusoid", name, tensors["native"][name], raw)

    # Isolated replay of the pinned BF16 mean expression, with the same valid
    # Qwen prefix and the full 1280-row binary mask. No linear/refiner is run.
    valid = ndoc["valid_text_tokens"]
    compact = tensors["native"]["input_compact_valid"]
    replay = _mean_replay(torch, compact, valid)
    if inputs["source_mode"] == "source-bf16":
        compare("same_input_mean_replay", "source_bf16_expression", replay["source_bf16_expression"], tensors["source"]["masked_mean"])
        compare("same_input_mean_replay", "round_sum_divide_f32_count",
                replay["round_sum_divide_f32_count"], tensors["source"]["masked_mean"])
    compare("reduction_order_diagnostic", "native_f32_expression", replay["native_f32_expression"],
            tensors["native"]["masked_mean"])

    control = torch.arange(16, device=device, dtype=torch.float32)
    compare("sensitivity_control", "a_vs_a", control, control)
    compare("sensitivity_control", "deliberate_perturbation", control + 0.01, control)
    non_bf16 = control + 0.001
    compare("sensitivity_control", "non_bf16_boundary", non_bf16,
            non_bf16.to(torch.bfloat16).float(), atol=0.0, rtol=0.0)
    flat = [stats[key] for _kind, _name, stats, _atol, _rtol in queued for key in shared.METRIC_KEYS]
    # One terminal export for all comparison/replay metrics and the mask count.
    values = torch.stack([*flat, replay["denominator"].reshape(()).float()]).detach().cpu().tolist()
    rows = []
    stride = len(shared.METRIC_KEYS)
    for index, (kind, name, _stats, atol, rtol) in enumerate(queued):
        rows.append(dict(kind=kind, tensor=name, **shared._stats_record(
            values[index*stride:(index+1)*stride], atol=atol, rtol=rtol)))
    raw_rows = [row for row in rows if row["kind"] == "raw"]
    boundary_rows = [row for row in rows if row["kind"] == "connected_bf16_boundary"]
    controls = {row["tensor"]: row for row in rows if row["kind"] == "sensitivity_control"}
    return dict(
        numeric_gate_passed=all(row["passed"] for row in [*raw_rows, *boundary_rows]), rows=rows,
        connected_bf16_boundary_count=len(boundary_rows),
        connected_bf16_boundaries_passed=(all(row["passed"] for row in boundary_rows)
                                          if boundary_rows else None),
        native_prelude_input_bf16_control=ndoc.get("prelude_input_bf16_control", False),
        native_prelude_linear_bf16_control=ndoc.get("prelude_linear_bf16_control", False),
        native_prelude_silu_bf16_control=ndoc.get("prelude_silu_bf16_control", False),
        native_prelude_second_linear_bf16_control=ndoc.get("prelude_second_linear_bf16_control", False),
        native_prelude_add_bf16_control=ndoc.get("prelude_add_bf16_control", False),
        actual_source_bf16_mask_sum_replay=values[-1], true_valid_text_tokens=valid,
        replay_qwen_byte_identical=True,
        replay_mask_equivalent_after_source_cast=True,
        source_refiner_mask_exact_prefix=True,
        sensitivity_controls_passed=(controls["a_vs_a"]["passed"]
                                     and not controls["deliberate_perturbation"]["passed"]
                                     and not controls["non_bf16_boundary"]["passed"]),
        terminal_cpu_exports=1,
        interpretation="Post-rounded views are diagnostic only. BF16 source-expression replay uses byte-identical consumed Qwen and an exact authenticated mask; it is compared only in source-bf16 mode. The F32 expression uses a different reduction order from native and is diagnostic. This comparator runs no refiner forward; the native capture identifies enabled connected input, first-linear, first-SiLU, second-linear and conditioning-add controls. The numeric gate covers only the 16 captured prelude rows on this input, plus selected BF16 representation checks; source byte identity requires separate tensor hashes. No complete refiner, K/V or motion qualification follows.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "native-sha256", "source", "source-sha256", "native-preblock",
                 "native-preblock-sha256", "reference-latent", "reference-latent-sha256", "timeline-sha256", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        output = shared.output_path(args.output)
        inputs = validate_inputs(args)
        result = _run_mps(inputs)
        document = dict(schema=SCHEMA, source_mode=inputs["source_mode"], device="mps",
                        native_manifest_sha256=args.native_sha256,
                        source_manifest_sha256=args.source_sha256,
                        timeline_manifest_sha256=args.timeline_sha256,
                        atol=ATOL, rtol=RTOL, **result)
        shared._publish_fresh(output, document)
    except (Error, shared.native_cache.ReferenceCacheError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps(dict(schema=SCHEMA, status="rejected", reason=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(dict(schema=SCHEMA, output=str(output), numeric_gate_passed=result["numeric_gate_passed"],
                          sensitivity_controls_passed=result["sensitivity_controls_passed"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
