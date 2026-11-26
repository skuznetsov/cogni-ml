#!/usr/bin/env python3
"""Authenticate the bounded three-arm render and independently check its tensors.

This post-drain CPU checker establishes trajectory parity and saved-zero replay.
It does not establish actor motion, visual quality, or resource-guard compliance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import hy_nomas_soft_route_qualify as frozen


CASES = ("zero", "A1_high", "A2_high")
LATENT_SHAPE = (1, 32, 30, 9, 16)
TENSOR_NAMES = tuple(f"{case}.step{step}.latent" for case in CASES for step in range(1, 5)) + tuple(
    f"{case}.step1.velocity" for case in CASES
) + ("zero.step1.patch", "zero.step1.text_valid", "zero.step1.vector")
SHAPES = {name: LATENT_SHAPE for name in TENSOR_NAMES}
SHAPES.update({"zero.step1.patch": (1, 4320, 2048),
               "zero.step1.text_valid": (1, 224, 2048), "zero.step1.vector": (1, 2048)})
BINDINGS = {
    "parent_sha256": frozen.PARENT_SHA256,
    "bias_sha256": frozen.BIAS_SHA256,
    "roi_source_sha256": "ee7a830eb287cffd5d4ee2297d1ce5bd99e7f26a3dcc36442358c12f6ba72dd8",
    "noise_selection_sha256": "8abfe96dc1a2b8ee3cd47f5bfac606c5f25630d95c637b535ec7811b122107ef",
    "initial_latent_sha256": "503615dc8920afe80d441fa025380c4b764b1e81f07d3a44525127010ddb2e5a",
    "t1_conditioning_sha256": "4a7e2cc71586b0380735f6301c382efcae7ba7fc6ed3eaa0c43b37eb4b57c3c5",
    "t1_qwen_sha256": "673d17f11549d89bf5bb4010b22cd9e3e4923ed3071b55a0c7e7348c4e50faf9",
}
FROZEN_SOURCE_SHA = "e7e0eefc258cc6b50d8df4e352b5106c7a022cee4c41a2a196249db96f637b31"
REPLAY_NAMES = {f"zero.step{step}.latent": f"t1.step{step}.after_euler" for step in range(1, 5)}
REPLAY_NAMES.update({f"zero.step1.{kind}": f"t1.step1.{kind}"
                     for kind in ("velocity", "patch", "text_valid", "vector")})


def load_render(path: Path, digest: str, kind: str):
    canonical, identity, doc = frozen._read_pinned_json(path, digest, f"{kind} render")
    expected_schema = "hy-nomas-soft-route-render-v1" if kind == "source" else "hy-nomas-soft-route-render-native-v1"
    if doc.get("schema") != expected_schema or doc.get("bindings") != BINDINGS:
        raise frozen.QualificationError(f"{kind} render schema/consumed-input binding mismatch")
    if doc.get("case_order") != list(CASES):
        raise frozen.QualificationError(f"{kind} render case order mismatch")
    validate_execution(doc, kind)
    payload = doc.get("payload")
    if type(payload) is not dict:
        raise frozen.QualificationError(f"{kind} render payload missing")
    basename = payload.get("basename")
    if type(basename) is not str or basename in ("", ".", "..") or Path(basename).name != basename:
        raise frozen.QualificationError(f"{kind} render payload must be a sibling basename")
    raw = frozen._canonical_regular(canonical.parent / basename, f"{kind} render payload")
    if (raw.parent != canonical.parent or type(payload.get("bytes")) is not int
            or raw.stat().st_size != payload["bytes"]
            or frozen.sha256_file(raw) != frozen._digest(payload.get("sha256"), "payload SHA-256")):
        raise frozen.QualificationError(f"{kind} render payload size/digest mismatch")
    records = doc.get("tensors")
    if type(records) is not dict or set(records) != set(TENSOR_NAMES):
        raise frozen.QualificationError(f"{kind} render tensor inventory mismatch")
    fields = ("shape", "dtype", "source_dtype", "offset_bytes", "nbytes", "sha256")
    if "checkpoint_boundary" in records[TENSOR_NAMES[0]]:
        fields += ("checkpoint_boundary",)
        if any(row.get("checkpoint_boundary") != name.split(".", 1)[1] for name, row in records.items()):
            raise frozen.QualificationError(f"{kind} render checkpoint boundary mismatch")
    arrays = frozen.validate_tensor_payload(raw, records, TENSOR_NAMES, SHAPES,
                                            record_fields=fields, label=kind)
    return {"path": str(canonical), "sha256": identity, "document": doc,
            "payload_path": str(raw), "arrays": arrays}


def validate_execution(doc: Mapping[str, Any], kind: str) -> None:
    """Check instrumented receipt bounds; the external memory guard is separate."""
    execution = doc.get("execution")
    if type(execution) is not dict or execution.get("case_order") != list(CASES):
        raise frozen.QualificationError(f"{kind} render execution receipt missing")
    if kind == "native":
        counts = execution
        expected = {"steps": 4, "full_block_calls": 864, "t0_block_calls": 216,
                    "t1_block_calls": 648, "refiner_calls": 4, "patch_calls": 12,
                    "final_head_calls": 12, "euler_updates": 12, "command_chains": 1,
                    "terminal_waits": 1, "named_allocations_during_encode": 0,
                    "activation_readbacks_before_terminal_wait": 0,
                    "postwait_tensor_readbacks": 18, "decoder_calls": 0}
        observed = counts.get("observed_attention_calls")
        if (type(observed) is not dict or set(observed) != {"T0", *CASES}
                or any(type(value) is not int or value != 216 for value in observed.values())):
            raise frozen.QualificationError("native render observed attention coverage mismatch")
        resources = doc.get("resource_accounting", {})
        budget = doc.get("max_named_bytes")
        actual = resources.get("actual_setup_named_bytes")
        peak = resources.get("peak_named_bytes")
        if (any(type(value) is not int for value in (budget, actual, peak))
                or not 0 < actual <= peak <= budget <= 21 * 1024**3):
            raise frozen.QualificationError("native render named-buffer budget mismatch")
    else:
        counts = execution.get("runtime_receipt")
        expected = {"steps": 4, "block_calls": 864, "t0_writer_block_calls": 216,
                    "t1_arm_block_calls": 648, "refiner_calls": 4, "final_head_calls": 12,
                    "euler_updates": 12, "attention_calls": 864, "t0_attention_calls": 216,
                    "t1_attention_calls": 648, "attention_query_chunks": 15552,
                    "explicit_terminal_mps_synchronize_count": 1, "terminal_export_count": 1,
                    "intermediate_readbacks": 0, "captures": 18}
    if type(counts) is not dict or any(type(counts.get(key)) is not int or counts[key] != value
                                       for key, value in expected.items()):
        raise frozen.QualificationError(f"{kind} render execution counter mismatch")


def load_roi(path: Path) -> dict[str, np.memmap]:
    canonical, _, doc = frozen._read_pinned_json(path, BINDINGS["roi_source_sha256"], "accepted ROI")
    info = doc["capture_payload"]
    raw = frozen._canonical_regular(canonical.parent / info["path"], "accepted ROI payload")
    if raw.stat().st_size != info["bytes"] or frozen.sha256_file(raw) != info["sha256"]:
        raise frozen.QualificationError("accepted ROI payload digest mismatch")
    records = {r["name"]: r for r in info["records"]}
    result = {}
    for output, name in REPLAY_NAMES.items():
        row = records[name]
        if (row["shape"] != list(SHAPES[output]) or row["source_dtype"] != "BF16"
                or frozen._sha256_range(raw, row["offset"], row["nbytes"]) != row["sha256"]):
            raise frozen.QualificationError(f"accepted ROI record mismatch: {name}")
        array = np.memmap(raw, dtype="<f4", mode="r", offset=row["offset"], shape=tuple(row["shape"]))
        frozen.validate_bf16_expansion(array, name)
        result[output] = array
    return result


def load_frozen_velocities(path: Path) -> dict[str, np.memmap]:
    canonical, _, doc = frozen._read_pinned_json(path, FROZEN_SOURCE_SHA, "qualified frozen source")
    payload = doc["payload"]
    raw = frozen._canonical_regular(canonical.parent / payload["basename"], "frozen source payload")
    if raw.stat().st_size != payload["bytes"] or frozen.sha256_file(raw) != payload["sha256"]:
        raise frozen.QualificationError("qualified frozen source payload digest mismatch")
    result = {}
    for case in CASES:
        row = doc["tensors"][f"{case}.velocity"]
        if row["shape"] != list(LATENT_SHAPE) or frozen._sha256_range(raw, row["offset_bytes"], row["nbytes"]) != row["sha256"]:
            raise frozen.QualificationError(f"qualified frozen velocity mismatch: {case}")
        result[f"{case}.step1.velocity"] = np.memmap(raw, dtype="<f4", mode="r",
                                                    offset=row["offset_bytes"], shape=LATENT_SHAPE)
    return result


def evaluate_arrays(source: Mapping[str, Any], native: Mapping[str, Any],
                    accepted: Mapping[str, Any], frozen_velocities: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute gates from bytes; producer-reported passed fields are not evidence."""
    if set(source) != set(TENSOR_NAMES) or set(native) != set(TENSOR_NAMES):
        raise frozen.QualificationError("render comparison tensor inventory mismatch")
    if set(accepted) != set(REPLAY_NAMES) or set(frozen_velocities) != {f"{c}.step1.velocity" for c in CASES}:
        raise frozen.QualificationError("render replay inventory mismatch")
    parity = {name: frozen.compare_arrays(source[name], native[name]) for name in TENSOR_NAMES}
    replays = {}
    for kind, arrays in (("source", source), ("native", native)):
        for name, value in accepted.items():
            replays[f"{kind}.accepted.{name}"] = frozen.compare_arrays(value, arrays[name])
        for name, value in frozen_velocities.items():
            replays[f"{kind}.frozen.{name}"] = frozen.compare_arrays(value, arrays[name])
    failed = [f"parity:{name}" for name, row in parity.items() if row["outside_atol_rtol"]]
    failed.extend(f"exact-replay:{name}" for name, row in replays.items() if row["exact_bit_differences"])
    effects = {f"{kind}.{case}": frozen.compare_arrays(arrays["zero.step4.latent"], arrays[f"{case}.step4.latent"])
               for kind, arrays in (("source", source), ("native", native)) for case in CASES[1:]}
    return {"passed": not failed, "failed_gates": failed, "parity": parity,
            "exact_replays": replays, "final_latent_effects": effects,
            "atol": frozen.ATOL, "rtol": frozen.RTOL,
            "scope": "trajectory parity and saved-zero replay only; effects do not prove visible acting"}


def qualify(source_path: Path, source_sha: str, native_path: Path, native_sha: str,
            roi_path: Path, frozen_source_path: Path) -> dict[str, Any]:
    source = load_render(source_path, source_sha, "source")
    native = load_render(native_path, native_sha, "native")
    result = evaluate_arrays(source["arrays"], native["arrays"], load_roi(roi_path), load_frozen_velocities(frozen_source_path))
    result.update({"schema": "cogni_ml.hy_nomas_soft_route_render_qualification/v1",
                   "bindings": BINDINGS, "artifacts": {kind: {k: v for k, v in item.items() if k not in ("arrays", "document")}
                                                      for kind, item in (("source", source), ("native", native))}})
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "native", "roi", "frozen-source", "output"):
        p.add_argument(f"--{key}", required=True, type=Path)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--native-sha", required=True)
    args = p.parse_args()
    result = qualify(args.source, args.source_sha, args.native, args.native_sha, args.roi, args.frozen_source)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "failed_gates": result["failed_gates"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
