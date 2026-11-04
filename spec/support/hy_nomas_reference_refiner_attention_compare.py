#!/usr/bin/env python3
"""Compare a native connected QK/softmax/PV capture with a pinned MPS replay.

All arithmetic/metrics run on MPS, followed by one terminal metric export.
The general-263 probability witness is source-exact for the first 256 queries;
its last seven rows are explicitly a counterfactual to the source vector path.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

try:
    from . import hy_nomas_reference_refiner_attention_replay as replay
except ImportError:
    import hy_nomas_reference_refiner_attention_replay as replay

shared, Error = replay.shared, replay.Error
SCHEMA = "cogni_ml.hy_nomas_refiner_attention_compare/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_refiner_attention_probe_capture/v1"
ATOL, RTOL = replay.ATOL, replay.RTOL
CAP = 512*1024*1024
ATTENTION_SHAPE = (1, 263, 16, 128)
SCORE_SHAPE = (1, 263, 16, 263)
Q_BYTES = math.prod(ATTENTION_SHAPE)*4
SCORE_BYTES = math.prod(SCORE_SHAPE)*4
# The native probe's base graph owns input, output and packed-plan buffers.
BASE_NAMED_BYTES = 18*Q_BYTES + 10*SCORE_BYTES
CONNECTED_NAMED_BYTES = 10*Q_BYTES + 2*SCORE_BYTES
QUALIFICATION = (
    "source_chunked_vs_observed", "source_chunked_a_vs_a",
    "source_main_general_vs_chunked_main", "source_full_vs_main_probabilities",
    "f32_general_vs_observed_native", "scaled_only_p_vs_general_p",
    "scaled_only_pv_vs_general", "scaled_only_pv_vs_observed_main",
    "common_pv_f32_bf16_round_vs_source_general", "sensitivity_a_vs_a",
)
NEGATIVE_QUALIFICATION = ("sensitivity_perturbation", "raw_round_only_p_vs_general_p_main")
NATIVE_ORDER = (
    "input.q", "input.k", "input.v", "input.source_probabilities",
    "qk_raw", "qk_scaled", "softmax", "pv", "canonical_attention", "pv_source_probabilities",
)
CONNECTED_NATIVE_ORDER = NATIVE_ORDER + (
    "bf16_boundary_attention", "bf16_boundary_attention_with_probabilities",
    "bf16_boundary_probabilities",
)
CONNECTED_ROWS = (
    "connected_main256_gate", "connected_general_control",
    "connected_instrumented_context", "connected_probability_main256_gate",
    "connected_probability_general_control", "source_tail_actual",
)
COMPILED_SOURCE_FILES = (
    "examples/hy_nomas_refiner_attention_probe.cr",
    "examples/support/hy_nomas_refiner_attention_probe.cr",
    "spec/support/hy_nomas_mps_reductions.cr",
    "src/ml/video/hy_nomas_mps_reductions_bridge.mm",
    "src/ml/video/hy_nomas_attention_mps.cr",
    "src/ml/video/metal_mps_command_chain.cr",
    "src/ml/video/metal_ops.cr", "src/ml/core/buffer.cr",
    "src/ml/metal/device.cr", "src/ml/video/hy_nomas_block.cr",
)
# These embedded fingerprints are selected sources, not full binary or
# toolchain provenance. The guarded build retains the wider source snapshot.
LOCALIZATION_POSITIVE = (
    "scaled_logits_bf16_round_vs_qualified_candidate_main",
    "common_probabilities_pv_bf16_round_vs_observed_main",
)
LOCALIZATION_NEGATIVE = (
    "scaled_logits_vs_qualified_bf16_candidate_main",
    "probabilities_vs_source_main", "pv_vs_source_observed_main",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Error(message)


def _validate_layout(document: Mapping[str, Any], payload: bytes) -> None:
    order, records = document.get("tensor_order"), document.get("tensors")
    _require(type(order) is list and order and all(type(n) is str for n in order)
             and len(order) == len(set(order)) and type(records) is dict
             and set(order) == set(records), "packet tensor inventory differs")
    cursor = 0
    for name in order:
        record = records[name]
        _require(type(record) is dict, f"invalid tensor record: {name}")
        shape, offset, count = record.get("shape"), record.get("offset"), record.get("nbytes")
        _require(type(shape) is list and 1 <= len(shape) <= 4 and
                 all(type(v) is int and 0 < v <= 1280*2048 for v in shape),
                 f"invalid tensor shape: {name}")
        _require(record.get("dtype") == "F32" and type(offset) is int and type(count) is int
                 and offset == cursor and count == math.prod(shape)*4 and 0 < count <= CAP,
                 f"invalid tensor extent/dtype: {name}")
        cursor += count
        _require(cursor <= len(payload) <= CAP and
                 hashlib.sha256(payload[offset:cursor]).hexdigest() == record.get("sha256"),
                 f"tensor payload digest differs: {name}")
    _require(cursor == len(payload), "packet has trailing payload bytes")


def _load_packet(path_arg: str, digest: str, schema: str) -> tuple[dict[str, Any], bytes]:
    _require(isinstance(digest, str) and replay.admission.HEX_SHA256.fullmatch(digest) is not None,
             "invalid packet manifest digest")
    path = replay.admission._canonical_absolute_file(path_arg, "packet manifest")
    _require(replay.admission._sha256_file(path) == digest, "packet manifest digest differs")
    document, _ = shared._read_unique_json(path, "attention packet")
    _require(document.get("schema") == schema and type(document.get("schema_version")) is int
             and document["schema_version"] == 1, "packet schema differs")
    _require(document.get("source_exact_qkv") is True, "packet lacks exact QKV admission")
    geometry = document.get("geometry")
    _require(type(geometry) is dict and set(geometry) == set(replay.GEOMETRY) and
             all(type(geometry[k]) is int and geometry[k] == v for k, v in replay.GEOMETRY.items()),
             "packet geometry differs")
    artifact = document.get("tensor_artifact")
    _require(type(artifact) is dict, "packet lacks tensor artifact")
    name, count = artifact.get("basename"), artifact.get("bytes")
    _require(type(name) is str and name not in ("", ".", "..") and Path(name).name == name
             and "/" not in name and "\\" not in name and type(count) is int and 0 < count <= CAP,
             "invalid packet payload path/size")
    payload_path = replay.admission._canonical_absolute_file(path.parent/name, "packet payload")
    _require(payload_path.parent == path.parent and payload_path.stat().st_size == count,
             "packet payload location/size differs")
    payload = payload_path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == artifact.get("sha256"), "packet payload digest differs")
    _validate_layout(document, payload)
    return document, payload


def _qualify_replay(document: Mapping[str, Any]) -> None:
    rows = document.get("rows")
    _require(type(rows) is list and all(type(row) is dict for row in rows), "replay lacks metric rows")
    names = [row.get("kind") for row in rows]
    _require(all(type(name) is str for name in names) and len(names) == len(set(names))
             and set(QUALIFICATION + NEGATIVE_QUALIFICATION) <= set(names),
             "replay qualification coverage differs")
    by_name = {row["kind"]: row for row in rows}
    _require(all(by_name[name].get("passed") is True for name in QUALIFICATION)
             and all(by_name[name].get("passed") is False for name in NEGATIVE_QUALIFICATION),
             "replay faithfulness or sensitivity gate failed")
    for name in ("scaled_only_p_vs_general_p", "scaled_only_pv_vs_general",
                 "scaled_only_pv_vs_observed_main", "common_pv_f32_bf16_round_vs_source_general"):
        _require(by_name[name].get("atol") == 0.0 and by_name[name].get("rtol") == 0.0,
                 "source-exact explanation requires exact row tolerance")
    negative = by_name["raw_round_only_p_vs_general_p_main"]
    _require(negative.get("atol") == ATOL and negative.get("rtol") == RTOL,
             "raw-round alternative requires the unchanged mixed tolerance")
    _require(document.get("numeric_tolerance") == dict(atol=ATOL, rtol=RTOL)
             and type(document.get("terminal_cpu_exports")) is int
             and document["terminal_cpu_exports"] == 1
             and document.get("source_query_chunks") == [256, 7], "replay execution scope differs")
    _require(document.get("device") == "mps"
             and type(document.get("torch_version")) is str
             and document["torch_version"].split("+", 1)[0].startswith("2.8.")
             and all(type(document.get(key)) is str and document[key]
                     for key in ("macos_version", "os_build"))
             and document.get("source_mask") == replay.SOURCE_MASK
             and document.get("source_probabilities_scope") == replay.PROBABILITIES_SCOPE
             and document.get("explicit_stages_scope") == replay.EXPLICIT_SCOPE
             and document.get("full_generation_executed") is False
             and document.get("attention_production_changed") is False,
             "replay runtime or diagnostic scope differs")
    pins = document.get("source_upstream_source_sha256")
    _require(document.get("upstream_revision") == replay.admission.PINNED_REVISION
             and type(pins) is dict and set(pins) == set(replay.admission.REQUIRED_UPSTREAM_FILES)
             and all(type(digest) is str and replay.admission.HEX_SHA256.fullmatch(digest)
                     for digest in pins.values()), "replay upstream identity differs")


def _validate_compiled_pins(pins: Any) -> None:
    _require(type(pins) is dict and set(pins) == set(COMPILED_SOURCE_FILES),
             "native compiled source pin inventory differs")
    root = Path(__file__).resolve().parents[2]
    for relative, digest in pins.items():
        _require(replay.admission._sha256_file(root/relative) == digest,
                 f"compiled source pin differs: {relative}")


def _validate_native_capture_contract(document: Mapping[str, Any]) -> bool:
    control = document.get("bf16_boundaries_control", False)
    _require(type(control) is bool, "native BF16 boundaries control must be an exact bool")
    order = document.get("tensor_order")
    tensors = document.get("tensors")
    expected_order = CONNECTED_NATIVE_ORDER if control else NATIVE_ORDER
    _require(type(order) is list and order == list(expected_order),
             "native capture tensor order differs from BF16 boundaries control")
    _require(type(tensors) is dict and set(tensors) == set(expected_order),
             "native capture tensor inventory differs from BF16 boundaries control")
    if control:
        for name, shape in (
            ("bf16_boundary_attention", ATTENTION_SHAPE),
            ("bf16_boundary_attention_with_probabilities", ATTENTION_SHAPE),
            ("bf16_boundary_probabilities", SCORE_SHAPE),
        ):
            record = tensors.get(name)
            _require(type(record) is dict and record.get("dtype") == "F32"
                     and record.get("shape") == list(shape),
                     f"invalid connected BF16-boundary tensor: {name}")

    expected_named_bytes = BASE_NAMED_BYTES + (CONNECTED_NAMED_BYTES if control else 0)
    setup_bytes, max_bytes = document.get("setup_named_bytes"), document.get("max_named_bytes")
    _require(type(setup_bytes) is int and type(max_bytes) is int
             and setup_bytes == expected_named_bytes and 0 < setup_bytes <= max_bytes <= CAP,
             "native attention buffer budget differs")
    for key, expected in dict(encode_allocations=0, encode_readbacks=0,
                              preterminal_waits=0, terminal_waits=1).items():
        _require(type(document.get(key)) is int and document[key] == expected,
                 f"native counter differs: {key}")
    return control


def _qualify_connected_rows(rows: list[dict[str, Any]], enabled: bool) -> dict[str, Any]:
    result = dict(
        bf16_boundaries_control=enabled,
        production_attention_changed=enabled,
        connected_main256_gate_passed=None,
        connected_general_control_passed=None,
        connected_instrumented_context_passed=None,
        connected_probability_main256_gate_passed=None,
        connected_probability_general_control_passed=None,
        source_tail_actual_passed=None,
        connected_probability_scope=(
            "instrumented graph only; production context-only graph does not expose internal probabilities"
            if enabled else None
        ),
    )
    by_name = {row.get("kind"): row for row in rows}
    present = set(by_name) & set(CONNECTED_ROWS)
    if not enabled:
        _require(not present, "disabled BF16 boundaries control emitted connected comparison rows")
        return result

    _require(present == set(CONNECTED_ROWS), "connected BF16 comparison coverage differs")
    exact_rows = CONNECTED_ROWS[:-1]
    for name in exact_rows:
        row = by_name[name]
        _require(type(row.get("passed")) is bool and row.get("atol") == 0.0 and row.get("rtol") == 0.0,
                 f"connected comparison must use exact tolerance: {name}")
    tail = by_name["source_tail_actual"]
    _require(type(tail.get("passed")) is bool and tail.get("atol") == ATOL and tail.get("rtol") == RTOL,
             "source tail actual comparison must use the unchanged mixed tolerance")

    _require(by_name["connected_main256_gate"]["passed"],
             "connected candidate failed the actual-source main256 gate")
    _require(by_name["connected_general_control"]["passed"],
             "connected candidate differs from the general-263 context control")
    _require(by_name["connected_instrumented_context"]["passed"],
             "instrumented connected context differs from production context")
    _require(by_name["connected_probability_main256_gate"]["passed"],
             "instrumented connected probabilities failed the actual-source main256 gate")
    _require(by_name["connected_probability_general_control"]["passed"],
             "instrumented connected probabilities differ from the general-263 control")
    result.update(
        connected_main256_gate_passed=by_name["connected_main256_gate"]["passed"],
        connected_general_control_passed=by_name["connected_general_control"]["passed"],
        connected_instrumented_context_passed=by_name["connected_instrumented_context"]["passed"],
        connected_probability_main256_gate_passed=by_name["connected_probability_main256_gate"]["passed"],
        connected_probability_general_control_passed=by_name["connected_probability_general_control"]["passed"],
        source_tail_actual_passed=tail["passed"],
    )
    return result


def _qualify_localization(rows: list[dict[str, Any]]) -> None:
    by_name = {row["kind"]: row for row in rows}
    _require(len(by_name) == len(rows)
             and all(by_name.get(name, {}).get("passed") is True for name in LOCALIZATION_POSITIVE)
             and all(by_name.get(name, {}).get("passed") is False for name in LOCALIZATION_NEGATIVE),
             "native localization expectations failed; re-localize this input")
    for name in LOCALIZATION_POSITIVE:
        _require(by_name[name].get("atol") == 0.0 and by_name[name].get("rtol") == 0.0,
                 "native boundary localization requires exact cast/PV checks")
    for name in LOCALIZATION_NEGATIVE:
        _require(by_name[name].get("atol") == ATOL and by_name[name].get("rtol") == RTOL,
                 "native divergence requires the unchanged mixed tolerance")


def validate_inputs(args: Any) -> dict[str, Any]:
    source, source_bytes = _load_packet(args.replay, args.replay_sha256, replay.SCHEMA)
    native, native_bytes = _load_packet(args.native, args.native_sha256, NATIVE_SCHEMA)
    _qualify_replay(source)
    _require(native.get("input_manifest_sha256") == args.replay_sha256
             and native.get("input_payload_sha256") == source["tensor_artifact"]["sha256"],
             "native capture is detached from the supplied replay")
    bf16_boundaries_control = _validate_native_capture_contract(native)
    _validate_compiled_pins(native.get("compiled_source_sha256"))
    for name in ("q", "k", "v", "source_probabilities"):
        sr, nr = source["tensors"][name], native["tensors"]["input."+name]
        _require(nr["shape"] == sr["shape"] and nr["sha256"] == sr["sha256"]
                 and replay._record_bytes(native_bytes, nr) == replay._record_bytes(source_bytes, sr),
                 f"native replay input is not exact: {name}")
    return dict(source_document=source, source_payload=source_bytes,
                native_document=native, native_payload=native_bytes,
                bf16_boundaries_control=bf16_boundaries_control)


def _run_mps(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    _require(torch.backends.mps.is_available(), "attention comparison requires MPS")
    _require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") in ("", "0"), "MPS fallback must be disabled")
    source_document = inputs["source_document"]
    _require(str(torch.__version__) == source_document["torch_version"]
             and platform.mac_ver()[0] == source_document["macos_version"]
             and platform.version() == source_document["os_build"],
             "comparison runtime differs from the qualified replay")
    device, queued = torch.device("mps"), []

    def load(kind: str, name: str) -> Any:
        return shared._load_tensor(torch, np, inputs[f"{kind}_payload"],
                                   inputs[f"{kind}_document"]["tensors"][name], device)

    def compare(name: str, actual: Any, expected: Any, *, exact: bool = False) -> None:
        atol, rtol = (0.0, 0.0) if exact else (ATOL, RTOL)
        queued.append((name, shared._metric_stat_tensors(torch, actual, expected, atol=atol, rtol=rtol), atol, rtol))

    with torch.inference_mode():
        n = {name: load("native", name) for name in NATIVE_ORDER[4:]}
        connected = bool(inputs["bf16_boundaries_control"])
        for name, source_name in (("qk_raw", "f32_qk_raw"), ("qk_scaled", "f32_qk_scaled"),
                                  ("softmax", "f32_softmax"), ("pv", "f32_pv"),
                                  ("canonical_attention", "observed_native"),
                                  ("pv_source_probabilities", "f32_pv_source_probabilities")):
            compare("native_f32_faithfulness_"+name, n[name], load("source", source_name))
        compare("native_explicit_vs_canonical", n["pv"], n["canonical_attention"])
        scaled_candidate = load("source", "scaled_only_bf16")
        compare("scaled_logits_vs_qualified_bf16_candidate_main", n["qk_scaled"][:, :256], scaled_candidate[:, :256])
        compare("scaled_logits_bf16_round_vs_qualified_candidate_main", n["qk_scaled"][:, :256].to(torch.bfloat16).float(), scaled_candidate[:, :256], exact=True)
        compare("scaled_logits_vs_general_bf16_candidate_tail", n["qk_scaled"][:, 256:], scaled_candidate[:, 256:])
        compare("probabilities_vs_source_main", n["softmax"][:, :256], load("source", "source_main_probabilities"))
        compare("probabilities_vs_source_general_all", n["softmax"], load("source", "source_probabilities"))
        compare("pv_vs_source_observed_main", n["pv"][:, :256], load("source", "observed_source")[:, :256])
        compare("pv_vs_source_observed_tail", n["pv"][:, 256:], load("source", "observed_source")[:, 256:])
        compare("common_probabilities_pv_vs_source_general", n["pv_source_probabilities"], load("source", "source_sdpa_general"))
        compare("common_probabilities_pv_bf16_round_vs_source_general", n["pv_source_probabilities"].to(torch.bfloat16).float(), load("source", "source_sdpa_general"), exact=True)
        # Same-source P,V isolates PV from inherited QK/softmax drift.
        compare("common_probabilities_pv_bf16_round_vs_observed_main", n["pv_source_probabilities"][:, :256].to(torch.bfloat16).float(), load("source", "observed_source")[:, :256], exact=True)
        control = n["qk_scaled"][0, :1, :1, :16]
        compare("sensitivity_a_vs_a", control, control, exact=True)
        compare("sensitivity_perturbation", control+0.125, control, exact=True)
        if connected:
            production_context = load("native", "bf16_boundary_attention")
            instrumented_context = load("native", "bf16_boundary_attention_with_probabilities")
            instrumented_probabilities = load("native", "bf16_boundary_probabilities")
            observed_source = load("source", "observed_source")
            compare("connected_main256_gate", production_context[:, :256], observed_source[:, :256], exact=True)
            compare("connected_general_control", production_context, load("source", "source_sdpa_general"), exact=True)
            compare("connected_instrumented_context", instrumented_context, production_context, exact=True)
            compare("connected_probability_main256_gate", instrumented_probabilities[:, :256],
                    load("source", "source_main_probabilities")[:, :256], exact=True)
            compare("connected_probability_general_control", instrumented_probabilities,
                    load("source", "source_probabilities"), exact=True)
            # The last seven rows use the source vector path. This is reported
            # at the unchanged tolerance, without making it a qualification gate.
            compare("source_tail_actual", production_context[:, 256:], observed_source[:, 256:])
        values = torch.stack([stats[key] for _, stats, _, _ in queued for key in shared.METRIC_KEYS]).detach().cpu().tolist()
    stride = len(shared.METRIC_KEYS)
    rows = [dict(kind=name, **shared._stats_record(values[i*stride:(i+1)*stride], atol=atol, rtol=rtol))
            for i, (name, _, atol, rtol) in enumerate(queued)]
    by_name = {row["kind"]: row for row in rows}
    faithful = all(row["passed"] for row in rows if row["kind"].startswith("native_f32_faithfulness_"))
    _require(faithful and by_name["native_explicit_vs_canonical"]["passed"], "native F32 replay faithfulness failed")
    _require(by_name["sensitivity_a_vs_a"]["passed"] and not by_name["sensitivity_perturbation"]["passed"], "metric sensitivity failed")
    _qualify_localization(rows)
    connected_result = _qualify_connected_rows(rows, connected)
    return dict(device="mps", torch_version=str(torch.__version__), numeric_tolerance=dict(atol=ATOL, rtol=RTOL),
                terminal_cpu_exports=1, native_f32_faithfulness_passed=faithful, rows=rows,
                localization_gate_passed=True,
                first_qualified_main256_difference="scaled_logits_bf16_boundary",
                source_tail_scope="7 queries use vector attention; no source tail probability claim",
                stage_scope="QK is a matched F32 reduction control; scaled-logit candidate is qualified by exact actual-source main256 P and PV",
                connected_candidate_scope=("opt-in BF16-boundary block-0 attention; no full refiner wiring"
                                           if connected else None),
                full_generation_executed=False, **connected_result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "native-sha256", "replay", "replay-sha256", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        output = shared.output_path(args.output)
        inputs = validate_inputs(args)
        result = _run_mps(inputs)
        document = dict(schema=SCHEMA, schema_version=1, native_manifest_sha256=args.native_sha256,
                        replay_manifest_sha256=args.replay_sha256, **result)
        shared._publish_fresh(output, document)
    except (Error, OSError, ValueError, RuntimeError, ImportError) as error:
        import json
        print(json.dumps(dict(schema=SCHEMA, status="rejected", reason=str(error))), file=sys.stderr)
        return 2
    import json
    print(json.dumps(dict(schema=SCHEMA, output=str(output),
                         first_qualified_main256_difference=result["first_qualified_main256_difference"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
