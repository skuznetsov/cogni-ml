#!/usr/bin/env python3
"""Authenticate and compare the bounded Hy-Nomas segment-edit resume matrix.

This CPU-only comparator validates launcher receipts, exact checkpoint file
digests and F32 captures. It does not execute the model, parse checkpoint
semantics, or establish upstream/full-pixel quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import hy_nomas_compact_resume_reference as compact
import hy_nomas_sequence_resume_reference as resume


INDEX_SCHEMA = "cogni_ml.hy_nomas_segment_edit_resume_matrix/v1"
BASE_RECEIPT_SCHEMA = "cogni_ml.hy_nomas_compact_sequence_probe/v1"
EDIT_RECEIPT_SCHEMA = "cogni_ml.hy_nomas_segment_edit_probe/v1"
REPORT_SCHEMA = "cogni_ml.hy_nomas_segment_edit_resume_reference/v1"
INPUTS_SCHEMA = "cogni_ml.hy_nomas_continuation_inputs/v1"
SEGMENTS = 3
RESIDENT_SEGMENTS = 6
TARGET_SEGMENT = 1
SOURCE_SEGMENT = 5
CHANNELS = 32
SEGMENT_LATENTS = 30
HEIGHT = 9
WIDTH = 16
STEPS_PER_SEGMENT = 4
F32_BYTES = 4
TIMELINE_SHAPE = [1, CHANNELS, RESIDENT_SEGMENTS * SEGMENT_LATENTS, HEIGHT, WIDTH]
STEP_SHAPE = [1, CHANNELS, SEGMENT_LATENTS, HEIGHT, WIDTH]
TIMELINE_BYTES = math.prod(TIMELINE_SHAPE) * F32_BYTES
STEP_BYTES = math.prod(STEP_SHAPE) * F32_BYTES
ATOL = RTOL = 2e-5
NOISE_EDIT = {
    "schema": "cogni_ml.hy_nomas_segment_noise_edit/v1",
    "run_segments": SEGMENTS,
    "target_segment": TARGET_SEGMENT,
    "source_segment": SOURCE_SEGMENT,
    "resident_segments": RESIDENT_SEGMENTS,
}
ARM_INTERVALS = {
    "parent_a": [(0, 1)],
    "original_control": [(1, 3)],
    "edited_control": [(1, 3)],
    "edited_restart_chain": [(1, 2), (2, 3)],
}
IDENTITY_FIELDS = (
    "executable_sha256",
    "inputs",
    "inputs_sha256",
    "payload_sha256",
    "conditioning_sha256",
    "model_index_sha256",
    "upstream_revision",
    "segments",
    "steps_per_segment",
    "valid_text_tokens",
    "timeline_shape",
    "step_shape",
)
METRIC_FIELDS = {
    "encode_allocations",
    "encode_readbacks",
    "preterminal_waits",
    "command_segments",
    "terminal_waits",
}
EDIT_INVALIDATION = {
    "latent_frames": [TARGET_SEGMENT * SEGMENT_LATENTS, SEGMENTS * SEGMENT_LATENTS],
    "decoder_pixel_frames": [TARGET_SEGMENT * 120 - 20, SEGMENTS * 120],
    "prior_segment_overlap_pixel_frames": [TARGET_SEGMENT * 120 - 20, TARGET_SEGMENT * 120],
    "units": "half-open global frame ranges; regenerate every descendant",
}
EDIT_NOISE_SCOPE = (
    "one initial-noise gather from an unused donor in the authenticated state; "
    "target RoPE/cache epsilon and shared conditioning unchanged"
)


class SegmentEditResumeReferenceError(compact.CompactResumeReferenceError):
    """A matrix, receipt, checkpoint, or capture violates the edit contract."""


def _fail(message: str) -> None:
    raise SegmentEditResumeReferenceError(message)


def _canonical_edit_contract_sha(base_contract_sha256: str) -> str:
    # Match Crystal's insertion-ordered JSON for HyNomasSegmentNoiseEdit#bind_contract.
    contract = {
        "schema": "cogni_ml.hy_nomas_edited_sequence_boundary/v1",
        "base_contract_sha256": base_contract_sha256,
        "noise_edit": NOISE_EDIT,
    }
    raw = json.dumps(contract, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _edited_boundary_contracts(base_contracts: dict[str, str]) -> dict[str, str]:
    return {
        "1": base_contracts["1"],
        "2": _canonical_edit_contract_sha(base_contracts["2"]),
    }


def _validate_receipt(
    receipt: dict[str, Any],
    arm: str,
    interval: tuple[int, int],
    source: dict[str, Any],
) -> None:
    label = f"{arm} receipt"
    start, stop = interval
    edited = arm in ("edited_control", "edited_restart_chain")
    expected_schema = EDIT_RECEIPT_SCHEMA if edited else BASE_RECEIPT_SCHEMA
    if receipt.get("schema") != expected_schema:
        _fail(f"{label} schema does not match its edit treatment")
    expected_mode = "save" if start == 0 else "resume"
    if receipt.get("mode") != expected_mode:
        _fail(f"{label} mode does not match its interval")

    for key, expected in (
        ("segments", SEGMENTS),
        ("start_segment", start),
        ("stop_after_segments", stop),
        ("resume_next_segment", start),
        ("steps_per_segment", STEPS_PER_SEGMENT),
    ):
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            _fail(f"{label} {key} does not match its admitted interval")
    if type(receipt.get("pid")) is not int or receipt["pid"] <= 0:
        _fail(f"{label} pid must be a positive integer")
    for key in (
        "executable_sha256", "inputs_sha256", "payload_sha256",
        "conditioning_sha256", "model_index_sha256",
    ):
        compact._sha(receipt.get(key), f"{label} {key}")
    if type(receipt.get("inputs")) is not str or not receipt["inputs"]:
        _fail(f"{label} inputs must name its authenticated manifest")
    if type(receipt.get("upstream_revision")) is not str or not receipt["upstream_revision"]:
        _fail(f"{label} upstream_revision must be non-empty")
    if type(receipt.get("valid_text_tokens")) is not int or receipt["valid_text_tokens"] <= 0:
        _fail(f"{label} valid_text_tokens must be positive")
    for key, expected in (("timeline_shape", TIMELINE_SHAPE), ("step_shape", STEP_SHAPE)):
        value = receipt.get(key)
        if type(value) is not list or any(type(dim) is not int for dim in value) or value != expected:
            _fail(f"{label} {key} does not match the six-bank DiT geometry")

    if receipt.get("state_format") != "exact" or receipt.get("cache_roundtrip") != "exact":
        _fail(f"{label} must use exact checkpoint and cache-roundtrip treatment")
    if receipt.get("qbit_roundtrip_boundaries") != []:
        _fail(f"{label} cannot report QBit cache roundtrips")
    if type(receipt.get("compact_pack_boundary")) is not int or receipt["compact_pack_boundary"] != 0:
        _fail(f"{label} cannot report compact checkpoint packing")
    if type(receipt.get("codec")) is not dict or receipt["codec"]:
        _fail(f"{label} cannot report codec operations")

    boundary_map = receipt.get("boundary_contract_sha256")
    expected_boundaries = {"1", "2"}
    if type(boundary_map) is not dict or set(boundary_map) != expected_boundaries:
        _fail(f"{label} boundary contract map must contain boundaries 1 and 2")
    for boundary, digest in boundary_map.items():
        compact._sha(digest, f"{label} boundary contract {boundary}")
    contract = receipt.get("contract_sha256")
    if start == 0:
        if contract != "":
            _fail(f"{label} start-zero contract_sha256 must be empty")
    elif compact._sha(contract, f"{label} contract_sha256") != boundary_map[str(start)]:
        _fail(f"{label} contract SHA does not match its start boundary")
    state_input_sha = receipt.get("state_input_sha256")
    if start == 0:
        if state_input_sha != "":
            _fail(f"{label} start-zero interval cannot consume a state")
    else:
        compact._sha(state_input_sha, f"{label} state_input_sha256")

    metrics = receipt.get("metrics")
    if type(metrics) is not dict or set(metrics) != METRIC_FIELDS:
        _fail(f"{label} metrics must contain exactly the five instrumented counters")
    for key, value in metrics.items():
        compact._int(value, f"{label} metric {key}")
    for key in ("encode_allocations", "encode_readbacks", "preterminal_waits"):
        if metrics[key] != 0:
            _fail(f"{label} metric {key} must be zero")
    if metrics["terminal_waits"] != 1:
        _fail(f"{label} must report exactly one terminal drain")

    checkpoint_file_bytes = compact._int(
        receipt.get("checkpoint_file_bytes"), f"{label} checkpoint_file_bytes"
    )
    logical_packed = compact._int(
        receipt.get("logical_packed_kv_bytes"), f"{label} logical_packed_kv_bytes"
    )
    compact._int(receipt.get("resident_f32_kv_bytes"), f"{label} resident_f32_kv_bytes", minimum=1)
    checkpoint_payload_bytes = compact._int(
        receipt.get("checkpoint_payload_bytes"), f"{label} checkpoint_payload_bytes"
    )
    if stop < SEGMENTS:
        if checkpoint_file_bytes <= 0 or checkpoint_payload_bytes <= TIMELINE_BYTES:
            _fail(f"{label} checkpoint metadata omits its saved boundary")
    elif checkpoint_file_bytes != 0 or checkpoint_payload_bytes != 0:
        _fail(f"{label} terminal interval cannot claim an output checkpoint")
    if logical_packed != 0:
        _fail(f"{label} exact treatment cannot report packed K/V bytes")

    expected_edit = NOISE_EDIT if edited else None
    if receipt.get("noise_edit") != expected_edit:
        _fail(f"{label} noise_edit identity does not match its treatment")
    if edited:
        if receipt.get("invalidation") != EDIT_INVALIDATION:
            _fail(f"{label} invalidation ranges are incomplete or incorrect")
        if receipt.get("noise_scope") != EDIT_NOISE_SCOPE:
            _fail(f"{label} noise scope does not describe the admitted unused-donor edit")
    else:
        if receipt.get("invalidation") is not None:
            _fail(f"{label} unedited control cannot claim edit invalidation")
        if receipt.get("noise_scope") != "unchanged authenticated latent/epsilon banks; no reseeding":
            _fail(f"{label} must report unchanged latent/epsilon banks")

    if receipt.get("decoder") is not False or receipt.get("full_generation") is not False:
        _fail(f"{label} must remain a DiT-only interval probe")
    if receipt.get("full_upstream_video_parity") is not False:
        _fail(f"{label} cannot claim full upstream video parity")

    receipt_manifest = compact._absolute_file(receipt["inputs"], f"{label} input manifest")
    if receipt_manifest != Path(source["manifest_path"]):
        _fail(f"{label} input manifest does not match the authenticated source bank")
    if receipt["inputs_sha256"] != source["manifest_sha256"]:
        _fail(f"{label} inputs_sha256 does not match the source manifest")
    if receipt["payload_sha256"] != source["payload_sha256"]:
        _fail(f"{label} payload_sha256 does not match the source payload")


def _load_run(
    arm: str,
    entry: dict[str, Any],
    expected_interval: tuple[int, int],
    source: dict[str, Any],
) -> dict[str, Any]:
    if type(entry) is not dict or set(entry) != {"receipt_path", "interval"}:
        _fail(f"{arm} run entry must contain only receipt_path and interval")
    receipt_path = compact._absolute_file(entry["receipt_path"], f"{arm} receipt_path")
    if receipt_path.name != "receipt.json":
        _fail(f"{arm} receipt_path must name receipt.json")
    directory = resume._directory(receipt_path.parent, f"{arm} output directory")
    if receipt_path.parent != directory:
        _fail(f"{arm} receipt_path must use its canonical output directory")
    raw_interval = entry["interval"]
    if type(raw_interval) is not list or len(raw_interval) != 2:
        _fail(f"{arm} interval must be a two-integer [start, stop] list")
    interval = (
        compact._int(raw_interval[0], f"{arm} interval start"),
        compact._int(raw_interval[1], f"{arm} interval stop", minimum=1),
    )
    if interval != expected_interval:
        _fail(f"{arm} interval must be {list(expected_interval)}")
    receipt = resume._read_json(receipt_path, label=f"{arm} receipt")
    _validate_receipt(receipt, arm, interval, source)

    start, stop = interval
    diagnostics = receipt.get("diagnostics")
    expected_artifacts = compact._artifact_names(start, stop, SEGMENTS, "exact")
    if type(diagnostics) is not dict or set(diagnostics) != expected_artifacts:
        _fail(f"{arm} diagnostics do not name exactly the interval artifacts")
    timeline, timeline_meta = resume._read_artifact(
        directory,
        diagnostics,
        "timeline",
        "timeline.f32le",
        TIMELINE_BYTES,
        arm,
        retain_payload=True,
    )
    resume._finite(timeline, f"{arm} timeline")

    steps: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for segment in range(start, stop):
        for step in range(1, STEPS_PER_SEGMENT + 1):
            key = f"segment{segment}-step{step}"
            payload, metadata = resume._read_artifact(
                directory,
                diagnostics,
                key,
                f"{key}.f32le",
                STEP_BYTES,
                arm,
                retain_payload=True,
            )
            resume._finite(payload, f"{arm} {key}")
            steps[key] = (payload, metadata)
        final_step = steps[f"segment{segment}-step{STEPS_PER_SEGMENT}"][0]
        timeline_segment = compact._time_slice(
            timeline, TIMELINE_SHAPE, segment * SEGMENT_LATENTS, (segment + 1) * SEGMENT_LATENTS
        )
        if timeline_segment != final_step:
            _fail(f"{arm} timeline does not contain its final segment {segment} step capture")

    state = None
    if stop < SEGMENTS:
        state_filename = "boundary.hnrs"
        try:
            _, state_meta = resume._read_artifact(
                directory,
                diagnostics,
                "state",
                state_filename,
                receipt["checkpoint_file_bytes"],
                arm,
                retain_payload=False,
            )
        except resume.ResumeReferenceError as error:
            raise SegmentEditResumeReferenceError(
                f"{arm} checkpoint file byte extent or digest mismatch: {error}"
            ) from error
        state = {
            **state_meta,
            "sha256": diagnostics["state_sha256"],
            "file_bytes": receipt["checkpoint_file_bytes"],
            "checkpoint_payload_bytes": receipt["checkpoint_payload_bytes"],
        }
    return {
        "arm": arm,
        "interval": interval,
        "directory": directory,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "timeline": timeline,
        "timeline_meta": timeline_meta,
        "steps": steps,
        "state": state,
    }


def _load_arms(index: dict[str, Any], source: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw_arms = index.get("arms")
    if type(raw_arms) is not dict or set(raw_arms) != set(ARM_INTERVALS):
        _fail("index arms must contain exactly parent_a, original_control, edited_control, and edited_restart_chain")
    arms: dict[str, list[dict[str, Any]]] = {}
    directories: set[Path] = set()
    pids: set[int] = set()
    for arm, intervals in ARM_INTERVALS.items():
        document = raw_arms[arm]
        if type(document) is not dict or set(document) != {"runs"} or type(document["runs"]) is not list:
            _fail(f"{arm} must contain a runs list")
        if len(document["runs"]) != len(intervals):
            _fail(f"{arm} run count does not match its required intervals")
        runs = []
        for entry, interval in zip(document["runs"], intervals):
            run = _load_run(arm, entry, interval, source)
            if run["directory"] in directories:
                _fail("every matrix interval must use a distinct output directory")
            directories.add(run["directory"])
            pid = run["receipt"]["pid"]
            if pid in pids:
                _fail("every matrix interval must use a distinct process ID")
            pids.add(pid)
            runs.append(run)
        arms[arm] = runs
    return arms


def _validate_identities(
    arms: dict[str, list[dict[str, Any]]], source: dict[str, Any]
) -> dict[str, Any]:
    reference = arms["parent_a"][0]["receipt"]
    for arm_runs in arms.values():
        for run in arm_runs:
            receipt = run["receipt"]
            for key in IDENTITY_FIELDS:
                if receipt.get(key) != reference.get(key):
                    _fail(f"all matrix runs must share identity field {key}")
            if compact._absolute_file(receipt["inputs"], "receipt input manifest") != Path(source["manifest_path"]):
                _fail("all matrix runs must name the authenticated source manifest")
            if receipt["inputs_sha256"] != source["manifest_sha256"]:
                _fail("all matrix runs must share the authenticated input manifest digest")
            if receipt["payload_sha256"] != source["payload_sha256"]:
                _fail("all matrix runs must share the authenticated source payload digest")
    return {
        "executable_sha256": reference["executable_sha256"],
        "inputs": reference["inputs"],
        "inputs_sha256": reference["inputs_sha256"],
        "payload_sha256": reference["payload_sha256"],
        "conditioning_sha256": reference["conditioning_sha256"],
        "model_index_sha256": reference["model_index_sha256"],
        "upstream_revision": reference["upstream_revision"],
        "distinct_processes": True,
        "pids": {
            arm: [run["receipt"]["pid"] for run in runs]
            for arm, runs in arms.items()
        },
    }


def _validate_contracts(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base_contracts = arms["original_control"][0]["receipt"]["boundary_contract_sha256"]
    if arms["parent_a"][0]["receipt"]["boundary_contract_sha256"] != base_contracts:
        _fail("parent A and original control do not share the base boundary contracts")
    edited_contracts = _edited_boundary_contracts(base_contracts)
    edited_runs = arms["edited_control"] + arms["edited_restart_chain"]
    for run in edited_runs:
        if run["receipt"]["boundary_contract_sha256"] != edited_contracts:
            _fail("edited boundary contract does not preserve after-A and bind the fork after-B")
    return {
        "base": dict(base_contracts),
        "edited": edited_contracts,
        "after_a_preserved": True,
        "after_b_binds_edit_identity": True,
    }


def _validate_state_chains(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    parent_a = arms["parent_a"][0]
    original = arms["original_control"][0]
    edited_control = arms["edited_control"][0]
    edited_chain = arms["edited_restart_chain"]
    parent_state = parent_a["state"]
    if parent_state is None:
        _fail("parent A must save the exact after-A checkpoint")
    for run in (original, edited_control, edited_chain[0]):
        if run["receipt"]["state_input_sha256"] != parent_state["sha256"]:
            _fail("start-after-A run parent state does not match the authenticated parent A checkpoint")
    edited_b_state = edited_chain[0]["state"]
    if edited_b_state is None:
        _fail("edited B checkpoint interval must save its after-B state")
    if edited_chain[1]["receipt"]["state_input_sha256"] != edited_b_state["sha256"]:
        _fail("edited C restart parent state does not match the edited B checkpoint")
    return {
        "parent_a": {
            "interval": [0, 1],
            "state_input_sha256": parent_a["receipt"]["state_input_sha256"],
            "state": parent_state,
        },
        "original_control": {
            "interval": [1, 3],
            "state_input_sha256": original["receipt"]["state_input_sha256"],
        },
        "edited_control": {
            "interval": [1, 3],
            "state_input_sha256": edited_control["receipt"]["state_input_sha256"],
        },
        "edited_restart_chain": [
            {
                "interval": list(run["interval"]),
                "state_input_sha256": run["receipt"]["state_input_sha256"],
                "state": run["state"],
            }
            for run in edited_chain
        ],
        "parent_a_linked_to_controls": True,
        "edited_b_linked_to_c_restart": True,
    }


def _validate_timeline_prefix_and_suffix(
    arms: dict[str, list[dict[str, Any]]], source_bank: bytes
) -> dict[str, Any]:
    parent_a = arms["parent_a"][0]
    baseline_a = compact._time_slice(parent_a["timeline"], TIMELINE_SHAPE, 0, SEGMENT_LATENTS)
    for arm_runs in arms.values():
        for run in arm_runs:
            actual_a = compact._time_slice(run["timeline"], TIMELINE_SHAPE, 0, SEGMENT_LATENTS)
            if actual_a != baseline_a:
                _fail(f"{run['arm']} timeline changed the already-generated A prefix")
            stop = run["interval"][1]
            suffix_start = stop * SEGMENT_LATENTS
            actual_suffix = compact._time_slice(
                run["timeline"], TIMELINE_SHAPE, suffix_start, TIMELINE_SHAPE[2]
            )
            source_suffix = compact._time_slice(
                source_bank, TIMELINE_SHAPE, suffix_start, TIMELINE_SHAPE[2]
            )
            if actual_suffix != source_suffix:
                _fail(f"{run['arm']} mutated the future/source-bank suffix including unused donor F")
    return {
        "a_prefix_bitwise_unchanged": True,
        "a_prefix_bytes": len(baseline_a),
        "future_suffix_matches_source_bank": True,
        "donor_segment_5_in_unexecuted_suffix": True,
    }


def _segment_timeline(run: dict[str, Any], segment: int) -> bytes:
    start = segment * SEGMENT_LATENTS
    return compact._time_slice(run["timeline"], TIMELINE_SHAPE, start, start + SEGMENT_LATENTS)


def _compare_segment(
    left: dict[str, Any], right: dict[str, Any], segment: int, label: str
) -> tuple[dict[str, Any], bool]:
    steps: dict[str, Any] = {}
    changed = False
    for step in range(1, STEPS_PER_SEGMENT + 1):
        key = f"segment{segment}-step{step}"
        result = resume._compare_f32(left["steps"][key][0], right["steps"][key][0], f"{label} {key}")
        steps[str(step)] = result
        changed = changed or not result["bitwise_equal"]
    timeline = resume._compare_f32(
        _segment_timeline(left, segment), _segment_timeline(right, segment),
        f"{label} timeline segment {segment}",
    )
    changed = changed or not timeline["bitwise_equal"]
    return {"steps": steps, "timeline": timeline}, changed


def _compare_edit_effect(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    original = arms["original_control"][0]
    edited = arms["edited_control"][0]
    changed_by_segment: dict[str, bool] = {}
    comparisons: dict[str, Any] = {}
    for segment in (TARGET_SEGMENT, TARGET_SEGMENT + 1):
        report, changed = _compare_segment(
            original, edited, segment, "edited control vs original control"
        )
        comparisons[str(segment)] = report
        changed_by_segment[str(segment)] = changed
    if not changed_by_segment[str(TARGET_SEGMENT)]:
        _fail("segment edit produced no nonvacuous B output change")
    if not changed_by_segment[str(TARGET_SEGMENT + 1)]:
        _fail("segment edit produced no nonvacuous descendant C output change")
    return {
        "b_changed_from_original_control": changed_by_segment[str(TARGET_SEGMENT)],
        "c_changed_from_original_control": changed_by_segment[str(TARGET_SEGMENT + 1)],
        "segments": comparisons,
        "scope": "observed bitwise B/C output differences only; no quality claim",
    }


def _compare_edited_restart(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    control = arms["edited_control"][0]
    restart_by_segment = {
        segment: run
        for run in arms["edited_restart_chain"]
        for segment in range(*run["interval"])
    }
    per_segment: dict[str, Any] = {}
    all_steps_bitwise = True
    all_steps_within = True
    all_timeline_bitwise = True
    all_timeline_within = True
    for segment in (TARGET_SEGMENT, TARGET_SEGMENT + 1):
        comparison, _ = _compare_segment(
            control, restart_by_segment[segment], segment,
            "edited uninterrupted vs split restart",
        )
        step_results = list(comparison["steps"].values())
        all_steps_bitwise = all_steps_bitwise and all(result["bitwise_equal"] for result in step_results)
        all_steps_within = all_steps_within and all(result["within_tolerance"] for result in step_results)
        timeline_result = comparison["timeline"]
        all_timeline_bitwise = all_timeline_bitwise and timeline_result["bitwise_equal"]
        all_timeline_within = all_timeline_within and timeline_result["within_tolerance"]
        per_segment[str(segment)] = comparison
    full_timeline = resume._compare_f32(
        control["timeline"], arms["edited_restart_chain"][-1]["timeline"],
        "edited uninterrupted vs split restart full timeline",
    )
    if not all_steps_within or not all_timeline_within or not full_timeline["within_tolerance"]:
        _fail("edited uninterrupted vs split restart outputs exceed the existing 2e-5 tolerance")
    return {
        "within_tolerance": True,
        "all_steps_within_tolerance": all_steps_within,
        "all_steps_bitwise_equal": all_steps_bitwise,
        "all_timeline_segments_within_tolerance": all_timeline_within,
        "all_timeline_segments_bitwise_equal": all_timeline_bitwise,
        "full_timeline": full_timeline,
        "segments": per_segment,
    }


def _artifact_report(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        arm: [
            {
                "interval": list(run["interval"]),
                "receipt_path": str(run["receipt_path"]),
                "timeline": run["timeline_meta"],
                "steps": {
                    key: metadata
                    for key, (_, metadata) in sorted(run["steps"].items())
                },
                "state": run["state"],
            }
            for run in runs
        ]
        for arm, runs in arms.items()
    }


def compare_index(index_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Authenticate the five-run exact/edit matrix and compare its captures."""
    path = compact._absolute_file(str(Path(index_path).absolute()), "matrix index")
    index = resume._read_json(path, label="matrix index")
    expected_keys = {"schema", "segments", "steps_per_segment", "noise_edit", "arms", "source_bank"}
    if set(index) != expected_keys or index.get("schema") != INDEX_SCHEMA:
        _fail("matrix index schema or key inventory is unsupported")
    if type(index.get("segments")) is not int or index["segments"] != SEGMENTS:
        _fail("matrix must contain exactly the three active A/B/C segments")
    if type(index.get("steps_per_segment")) is not int or index["steps_per_segment"] != STEPS_PER_SEGMENT:
        _fail("matrix must include all four Euler steps per segment")
    if index.get("noise_edit") != NOISE_EDIT:
        _fail("matrix noise_edit must target B from unused resident segment F")

    source_bank, source_metadata = compact._load_source_bank(index)
    if source_metadata["shape"] != TIMELINE_SHAPE:
        _fail("source bank must have six resident segments of the admitted DiT geometry")
    if source_metadata["manifest_path"] != str(Path(index["source_bank"]["manifest_path"]).resolve(strict=True)):
        _fail("source bank manifest path is not canonical")

    arms = _load_arms(index, source_metadata)
    identity = _validate_identities(arms, source_metadata)
    contracts = _validate_contracts(arms)
    state_chains = _validate_state_chains(arms)
    prefix_and_future = _validate_timeline_prefix_and_suffix(arms, source_bank)
    edit_effect = _compare_edit_effect(arms)
    edited_restart = _compare_edited_restart(arms)

    return {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "scope": (
            "three-segment native DiT segment-edit continuation; compares self-consistent launcher "
            "receipts and F32 captures; does not establish upstream/full-pixel quality"
        ),
        "checkpoint_validation": (
            "Checkpoint validation is limited to artifact SHA-256 and receipt parent-state SHA-256 "
            "linkage; checkpoint payload semantics are not parsed."
        ),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "identity": identity,
        "source_bank": source_metadata,
        "noise_edit": dict(NOISE_EDIT),
        "intervals": {
            arm: [list(run["interval"]) for run in runs]
            for arm, runs in arms.items()
        },
        "metrics": {
            arm: [run["receipt"]["metrics"] for run in runs]
            for arm, runs in arms.items()
        },
        "contracts": contracts,
        "state_chains": state_chains,
        "a_prefix_and_future": prefix_and_future,
        "edit_effect": edit_effect,
        "edited_restart": edited_restart,
        "artifacts": _artifact_report(arms),
        "finite": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Five-run segment-edit matrix JSON")
    parser.add_argument("--output", required=True, help="Fresh JSON report path")
    args = parser.parse_args(argv)
    try:
        report = compare_index(args.index)
        output = Path(args.output).absolute()
        resume._write_fresh_json(output, report)
    except (OSError, ValueError) as error:
        print(f"segment-edit resume reference rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": report["passed"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
