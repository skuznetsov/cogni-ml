#!/usr/bin/env python3
"""Frozen-head, same-pass affinity screen; no model or GPU execution here.

Input preparation is BF16 byte selection/permutation only. The existing native
and MPS CLIs perform all QK/metric math on GPU; evaluation follows their drain.
This does not test independent scenes, semantic correspondence or transport.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve()
REFERENCE_PATH = HERE.with_name("hy_nomas_query_correspondence_reference.py")
CAPTURE_PATH = HERE.with_name("hy_nomas_query_capture.py")
ref = _module("query_head_reference", REFERENCE_PATH)
capture = _module("query_head_capture", CAPTURE_PATH)
DISCOVERY_SHA256 = "02024f72af2b09de6994637fde4799d5320a04f7cabc769710c192a5d01ed2a8"
PLAN_SCHEMA = "hy-nomas-query-head-robustness-plan-v1"
FRAME_BYTES = 144 * 16 * 128 * 2
SOURCE_SHAPE = (1, 4320, 16, 128)
BOUNDARIES = {"pre": ("img.q_qknorm", "img.k_qknorm"),
              "post": ("img.q_rope", "img.k_rope")}


def frozen_policy():
    return {"heads": [12, 15], "discovery_frame": 0,
            "held_out_frames": [1, 14, 29], "strict_diagonal_minimum": 108,
            "positions_per_frame": 144, "key_roll": 17, "roll_frame": 14,
            "independent_states": 1, "independent_scenes_tested": False,
            "screen_is_transport_gate": False, "atol": 2e-5, "rtol": 2e-5}


def validate_policy(policy):
    if policy != frozen_policy():
        raise ValueError("frozen policy differs from the declared screen")


def case_specs():
    cases = [{"name": "f00-pre", "frame": 0, "boundary": "pre", "key_roll": 0}]
    for frame in (1, 14, 29):
        for boundary in ("pre", "post"):
            cases.append({"name": f"f{frame:02}-{boundary}", "frame": frame,
                          "boundary": boundary, "key_roll": 0})
    for boundary in ("pre", "post"):
        cases.append({"name": f"f14-{boundary}-keyroll17", "frame": 14,
                      "boundary": boundary, "key_roll": 17})
    return cases


def validate_case_specs(plan):
    validate_policy(plan.get("policy"))
    if plan.get("cases") != case_specs():
        raise ValueError("case matrix differs from the frozen frame/boundary/roll selection")


def _frame(raw, frame):
    if type(frame) is not int or not 0 <= frame < 30 or len(raw) != FRAME_BYTES * 30:
        raise ValueError("source frame or full BF16 tensor extent is invalid")
    return raw[frame * FRAME_BYTES:(frame + 1) * FRAME_BYTES]


def select_case(raw_tensors, spec):
    if spec not in case_specs():
        raise ValueError("case differs from the frozen frame/boundary/roll selection")
    names = BOUNDARIES[spec["boundary"]]
    query, key = [capture._expand_bf16(_frame(raw_tensors[name], spec["frame"]), ref.SHAPE)
                  for name in names]
    if spec["key_roll"]:
        key = np.roll(key, spec["key_roll"], axis=0)
    return np.ascontiguousarray(query), np.ascontiguousarray(key)


def score_summary(metrics):
    metrics = ref.checked_array(metrics, ref.METRIC_SHAPE)
    result = {}
    for head in frozen_policy()["heads"]:
        count = int(np.count_nonzero(metrics[head, :, 2] > 0))
        result[str(head)] = {"strict_diagonal_count": count, "fraction": count / 144,
                             "screen_passed": count >= 108,
                             "self_mass_mean": float(np.mean(metrics[head, :, 5], dtype=np.float64)),
                             "margin_mean": float(np.mean(metrics[head, :, 2], dtype=np.float64))}
    return result


def check_roll(base_scores, rolled_scores, rolled_metrics):
    base_scores = ref.checked_array(base_scores, ref.SCORE_SHAPE)
    rolled_scores = ref.checked_array(rolled_scores, ref.SCORE_SHAPE)
    rolled_metrics = ref.checked_array(rolled_metrics, ref.METRIC_SHAPE)
    expected = np.roll(base_scores, 17, axis=-1)
    scores_exact = np.array_equal(rolled_scores, expected)
    # Ties crossing wrap can change which maximum has the lowest index.
    argmax_exact = np.array_equal(rolled_metrics[..., 4], np.argmax(expected, axis=-1))
    return {"passed": bool(scores_exact and argmax_exact),
            "scores_exact": bool(scores_exact), "stable_argmax_exact": bool(argmax_exact)}


def verify_case_binding(admitted, spec, expected_query, expected_key, expected_provenance):
    if spec not in case_specs() or admitted["document"].get("provenance") != expected_provenance:
        raise ValueError("case provenance differs from authenticated source selection")
    if (admitted["query"].tobytes() != expected_query.tobytes()
            or admitted["key"].tobytes() != expected_key.tobytes()):
        raise ValueError("submitted Q/K bytes differ from authenticated source frame")


def _implementations():
    return {str(path): ref.file_sha256(path) for path in (HERE, REFERENCE_PATH, CAPTURE_PATH)}


def _anchor(path, digest):
    if digest != DISCOVERY_SHA256:
        raise ValueError("anchor differs from the frozen discovery input")
    validation = capture.verify_published_input(path, digest)
    admitted = ref.load_input(path, digest)
    source = admitted["document"]["provenance"]["source_oracle"]
    receipt_path = Path(source["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    artifact_path = Path(source["artifact_path"])
    records = receipt["tensor_artifact"]["tensors"]
    raw = {}
    for name in ("img.q_qknorm", "img.k_qknorm", "img.q_rope", "img.k_rope"):
        raw[name], _ = capture._read_safetensors_range(
            artifact_path, name, expected_shape=SOURCE_SHAPE,
            expected_dtype="BF16", expected_sha256=records[name]["sha256"])
    binding = {"receipt_path": str(receipt_path), "receipt_sha256": ref.file_sha256(receipt_path),
               "artifact_path": str(artifact_path), "artifact_sha256": source["artifact_sha256"],
               "tensor_records": {name: records[name] for name in raw},
               "source_revision": source["upstream_revision"],
               "weights_sha256": source["weights_sha256"],
               "state": "one saved T0 forward, sigma 0.206897, timestep 206.897"}
    return {"validation": validation, "source": binding, "raw": raw}


def _provenance(context, spec):
    names = BOUNDARIES[spec["boundary"]]
    return {"kind": "authenticated_same_pass_query_head_screen", "selection": spec,
            "source_receipt_sha256": context["source"]["receipt_sha256"],
            "source_artifact_sha256": context["source"]["artifact_sha256"],
            "source_tensor_names": list(names),
            "source_rows": [spec["frame"] * 144, (spec["frame"] + 1) * 144],
            "source_bf16_range_sha256": {
                name: ref.sha256(_frame(context["raw"][name], spec["frame"])) for name in names},
            "anchor_sha256": DISCOVERY_SHA256,
            "permutation_scope": "key content and phases move together; not a position-preserving null",
            "scope": "same-pass frame affinity; no independent-state, transport or acting claim"}


def prepare(anchor, digest, directory):
    context = _anchor(anchor, digest)
    directory = directory.absolute()
    if directory.exists() or directory.is_symlink():
        raise ValueError("case directory must be fresh")
    directory.mkdir()
    directory = directory.resolve()
    cases = []
    for spec in case_specs():
        query, key = select_case(context["raw"], spec)
        path, sha = ref.publish_input(directory / f"input-{spec['name']}.json", query, key,
                                      _provenance(context, spec))
        cases.append({"name": spec["name"], "spec": spec,
                      "input": {"path": str(path), "sha256": sha}})
    plan = {"schema": PLAN_SCHEMA, "policy": frozen_policy(),
            "anchor": {"path": str(Path(anchor).resolve()), "sha256": digest},
            "anchor_validation": context["validation"], "source": context["source"],
            "implementations": _implementations(), "cases": cases}
    path = directory / "plan.json"
    with path.open("x") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")
    return {"plan": str(path), "sha256": ref.file_sha256(path), "cases": len(cases)}


def qualify(plan_path, digest, report):
    if report.exists() or report.is_symlink():
        raise ValueError("report must be fresh")
    if plan_path.is_symlink() or ref.file_sha256(plan_path) != digest:
        raise ValueError("plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != PLAN_SCHEMA or plan.get("implementations") != _implementations():
        raise ValueError("plan schema or implementation bytes changed")
    validate_policy(plan.get("policy"))
    context = _anchor(Path(plan["anchor"]["path"]), plan["anchor"]["sha256"])
    if (plan.get("anchor_validation") != context["validation"]
            or plan.get("source") != context["source"]):
        raise ValueError("plan source lineage changed")
    specs = case_specs()
    validate_case_specs({"policy": plan["policy"],
                         "cases": [case.get("spec") for case in plan.get("cases", [])]})
    if (type(plan.get("cases")) is not list or len(plan["cases"]) != len(specs)
            or any(case.get("name") != spec["name"] or case.get("spec") != spec
                   for case, spec in zip(plan["cases"], specs))):
        raise ValueError("plan cases differ from the frozen selection")
    cases, outputs = {}, {}
    for case, spec in zip(plan["cases"], specs):
        name, binding = spec["name"], case["input"]
        path = Path(binding["path"])
        if path != plan_path.parent / f"input-{name}.json":
            raise ValueError("case input path differs from its frozen sibling path")
        admitted = ref.load_input(path, binding["sha256"])
        query, key = select_case(context["raw"], spec)
        verify_case_binding(admitted, spec, query, key, _provenance(context, spec))
        source_path, native_path = [plan_path.parent / f"{label}-{name}.json"
                                    for label in ("source", "native")]
        numerical = ref.qualify(path, binding["sha256"], source_path, native_path)
        outputs[name] = {}
        for label, result_path in (("source", source_path), ("native", native_path)):
            document = json.loads(result_path.read_text())
            outputs[name][label] = ref.load_payload(result_path, document,
                                                   {"scores": ref.SCORE_SHAPE, "metrics": ref.METRIC_SHAPE})
        cases[name] = {"selection": spec, "lineage_passed": True, "numerical": numerical,
                       "frozen_heads": score_summary(outputs[name]["native"]["metrics"])}
    rolls = {}
    for boundary in ("pre", "post"):
        base = f"f14-{boundary}"
        rolls[boundary] = {}
        for label in ("source", "native"):
            rolled = outputs[base + "-keyroll17"][label]
            rolls[boundary][label] = check_roll(outputs[base][label]["scores"],
                                                 rolled["scores"], rolled["metrics"])
    screen = {str(head): all(cases[f"f{frame:02}-post"]["frozen_heads"][str(head)]["screen_passed"]
                             for frame in (1, 14, 29)) for head in (12, 15)}
    passed = (all(case["numerical"]["passed"] for case in cases.values())
              and all(item["passed"] for boundary in rolls.values() for item in boundary.values()))
    result = {"schema": "hy-nomas-query-head-robustness-report-v1", "passed": passed,
              "plan": {"path": str(plan_path), "sha256": digest}, "policy": frozen_policy(),
              "source": context["source"], "cases": cases, "roll_equivariance": rolls,
              "high_diagonal_affinity_screen": screen,
              "scope": "Nine authenticated same-pass array cases; no independent scenes, cross-frame matching, phase-preserving null, transport, causal rollout or acting claim"}
    with report.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--anchor", type=Path, required=True)
    prep.add_argument("--anchor-sha256", required=True)
    prep.add_argument("--directory", type=Path, required=True)
    check = commands.add_parser("qualify")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--plan-sha256", required=True)
    check.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.anchor, args.anchor_sha256, args.directory)))
        return 0
    result = qualify(args.plan, args.plan_sha256, args.report)
    print(json.dumps({"report": str(args.report), "passed": result["passed"],
                      "screen": result["high_diagonal_affinity_screen"],
                      "counts": {name: value["frozen_heads"] for name, value in result["cases"].items()}}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
