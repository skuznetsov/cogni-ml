#!/usr/bin/env python3
"""Fixed-destination-phase K-content null on one authenticated saved T0 state.

CPU preparation only selects/expands stored bytes. Native Metal and the pinned
upstream RoPE equation perform gathers, RoPE, BF16 rounding and QK on GPU.
CPU scoring audits run after the terminal drain. This is not actor transport.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("fixed_phase_heads", HERE.with_name("hy_nomas_query_head_robustness.py"))
heads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heads)
ref, capture = heads.ref, heads.capture
INPUT_SCHEMA = "hy-nomas-query-fixed-phase-input-v1"
PLAN_SCHEMA = "hy-nomas-query-fixed-phase-plan-v1"
PHASE_SHAPE = (144, 128)
OUTPUT_SHAPES = {"query": ref.SHAPE, "key": ref.SHAPE,
                 "scores": ref.SCORE_SHAPE, "metrics": ref.METRIC_SHAPE}
PREBLOCK_SHA = "0fd4898e12699ba058a8ad5047b55eb9b4a2898c8cb9a9f795235732a321488a"
PREBLOCK_PAYLOAD_SHA = "44b527535f2e3835b7168055b49d387987a37bcd43ddea97879cacf9ee03a257"
POSEMB_SHA = "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa"


def policy():
    return {"heads": [12, 15], "frames": [1, 14, 29], "key_roll": 17,
            "positions": 144, "atol": ref.ATOL, "rtol": ref.RTOL,
            "independent_saved_states": 1, "transport_gate": False}


def case_specs():
    return [{"frame": frame, "arm": arm, "key_roll": 0 if arm == "identity" else 17}
            for frame in (1, 14, 29) for arm in ("identity", "moving", "fixed")]


def validate_case(value):
    if (type(value) is not dict or set(value) != {"frame", "arm", "key_roll"}
            or type(value.get("frame")) is not int or type(value.get("key_roll")) is not int
            or value not in case_specs()):
        raise ValueError("case differs from frozen frame/arm/roll selection")


def case_name(value):
    validate_case(value)
    return f"f{value['frame']:02}-{value['arm']}"


def content_targets(arm):
    if arm not in ("identity", "moving", "fixed"):
        raise ValueError("unknown arm")
    return (np.arange(144) + (0 if arm == "identity" else 17)) % 144


def _canonical(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("input must be a canonical regular file without symlink components")
    return path


def _json(path, digest):
    path = _canonical(path)
    if path.stat().st_size > 65536 or ref.file_sha256(path) != digest:
        raise ValueError("JSON extent or trusted digest mismatch")
    return json.loads(path.read_text())


def publish_case(path, qk, cosine, sine, case, provenance):
    validate_case(case)
    path, payload = ref.fresh_paths(path)
    arrays = {"cosine": ref.checked_array(cosine, PHASE_SHAPE),
              "sine": ref.checked_array(sine, PHASE_SHAPE)}
    doc = {"schema": INPUT_SCHEMA, "schema_version": 1, "case": case,
           "qk": qk, "provenance": provenance, "tensors": {}}
    raw = bytearray()
    for name, array in arrays.items():
        data = array.tobytes()
        doc["tensors"][name] = {"dtype": "F32", "shape": list(PHASE_SHAPE),
                                "offset_bytes": len(raw), "nbytes": len(data), "sha256": ref.sha256(data)}
        raw.extend(data)
    doc["payload"] = {"path": payload.name, "bytes": len(raw), "sha256": ref.sha256(raw)}
    with payload.open("xb") as stream:
        stream.write(raw)
    with path.open("x") as stream:
        stream.write(json.dumps(doc, indent=2) + "\n")
    return {"path": str(path.resolve()), "sha256": ref.file_sha256(path)}


def load_input(path, digest):
    path = _canonical(path)
    doc = _json(path, digest)
    if (set(doc) != {"schema", "schema_version", "case", "qk", "payload", "tensors", "provenance"}
            or doc["schema"] != INPUT_SCHEMA or doc["schema_version"] != 1
            or type(doc["provenance"]) is not dict or set(doc["qk"]) != {"path", "sha256"}):
        raise ValueError("input schema or fields mismatch")
    validate_case(doc["case"])
    qk_path = _canonical(doc["qk"]["path"])
    qk = ref.load_input(qk_path, doc["qk"]["sha256"])
    if set(doc["payload"]) != {"path", "bytes", "sha256"}:
        raise ValueError("phase payload fields mismatch")
    for record in doc["tensors"].values():
        if set(record) != {"dtype", "shape", "offset_bytes", "nbytes", "sha256"}:
            raise ValueError("phase tensor fields mismatch")
    phases = ref.load_payload(path, doc, {"cosine": PHASE_SHAPE, "sine": PHASE_SHAPE})
    return {"path": str(path), "sha256": digest, "document": doc, "qk": qk, **phases}


def _phase_context(anchor_context, manifest):
    manifest = _canonical(manifest)
    doc = _json(manifest, PREBLOCK_SHA)
    payload = _canonical(manifest.parent / doc["payload"])
    if doc["payload_sha256"] != PREBLOCK_PAYLOAD_SHA or ref.file_sha256(payload) != PREBLOCK_PAYLOAD_SHA:
        raise ValueError("authenticated preblock payload mismatch")
    # The receipt must bind the same payload that supplied the source Q/K.
    receipt = json.loads(Path(anchor_context["source"]["receipt_path"]).read_text())
    source_input = receipt["input_bundle"]
    if (source_input["manifest_sha256"] != PREBLOCK_SHA
            or source_input["payload_sha256"] != PREBLOCK_PAYLOAD_SHA):
        raise ValueError("phase bundle differs from source Q/K capture input")
    arrays = {}
    raw_payload = payload.read_bytes()
    for short, name in (("cosine", "block.rope_cos_f32"), ("sine", "block.rope_sin_f32")):
        record = doc["tensors"][name]
        if record["dtype"] != "F32" or record["shape"] != [4320, 128] or record["nbytes"] != 4320 * 128 * 4:
            raise ValueError("full phase tensor geometry mismatch")
        raw = raw_payload[record["offset"]:record["offset"] + record["nbytes"]]
        if ref.sha256(raw) != record["sha256"]:
            raise ValueError("full phase tensor digest mismatch")
        if source_input["tensor_records"][name] != {k: record[k] for k in ("dtype", "shape", "sha256")}:
            raise ValueError("phase record differs from captured input")
        arrays[short] = ref.checked_array(np.frombuffer(raw, dtype="<f4").reshape(4320, 128), (4320, 128))
    return {"manifest": {"path": str(manifest), "sha256": PREBLOCK_SHA},
            "payload_sha256": PREBLOCK_PAYLOAD_SHA, "arrays": arrays}


def _implementations():
    paths = (HERE, heads.HERE, heads.REFERENCE_PATH, heads.CAPTURE_PATH,
             REPO / "examples/hy_nomas_query_fixed_phase_probe.cr",
             REPO / "examples/support/hy_nomas_query_fixed_phase_probe.cr",
             REPO / "examples/support/hy_nomas_query_correspondence_probe.cr",
             REPO / "src/ml/video/hy_nomas_mps_pointwise.cr",
             REPO / "src/ml/video/hy_nomas_block.cr",
             REPO / "src/ml/metal/kernels/hy_nomas/hy_nomas_mps_pointwise.metal",
             REPO / "src/ml/metal/kernels/hy_nomas/hy_nomas_block.metal")
    return {str(path): ref.file_sha256(path) for path in paths}


def _provenance(context, phases, case):
    frame = case["frame"]
    return {"kind": "same_pass_fixed_phase_content_null", "selection": case,
            "source_receipt_sha256": context["source"]["receipt_sha256"],
            "source_artifact_sha256": context["source"]["artifact_sha256"],
            "source_rows": [frame * 144, (frame + 1) * 144],
            "phase_manifest": phases["manifest"], "phase_payload_sha256": phases["payload_sha256"],
            "phase_ranges_sha256": {name: ref.sha256(value[frame * 144:(frame + 1) * 144].tobytes())
                                     for name, value in phases["arrays"].items()},
            "scope": "Restricted same-pass affinity; no semantic ground truth, transport or acting"}


def validate_binding(admitted, context, phases, case):
    validate_case(case)
    frame = case["frame"]
    if (admitted["document"]["case"] != case
            or admitted["document"]["provenance"] != _provenance(context, phases, case)):
        raise ValueError("case or provenance differs from authenticated selection")
    for short, source_name in zip(("query", "key"), heads.BOUNDARIES["pre"]):
        expected = capture._expand_bf16(heads._frame(context["raw"][source_name], frame), ref.SHAPE)
        if admitted["qk"][short].tobytes() != expected.tobytes():
            raise ValueError("Q/K bytes differ from authenticated pre-RoPE frame")
    for short in ("cosine", "sine"):
        if admitted[short].tobytes() != phases["arrays"][short][frame * 144:(frame + 1) * 144].tobytes():
            raise ValueError("destination phase bytes differ from authenticated frame")


def prepare(anchor, digest, phases_manifest, directory):
    context = heads._anchor(anchor, digest)
    phases = _phase_context(context, phases_manifest)
    if directory.exists() or directory.is_symlink():
        raise ValueError("case directory must be fresh")
    directory.mkdir()
    directory = directory.resolve()
    cases = []
    for frame in policy()["frames"]:
        query, key = [capture._expand_bf16(heads._frame(context["raw"][name], frame), ref.SHAPE)
                      for name in heads.BOUNDARIES["pre"]]
        qk_path, qk_sha = ref.publish_input(directory / f"qk-f{frame:02}.json", query, key,
                                          {"source": context["source"], "frame": frame, "boundary": "pre"})
        qk = {"path": str(qk_path.resolve()), "sha256": qk_sha}
        for case in [item for item in case_specs() if item["frame"] == frame]:
            name = case_name(case)
            binding = publish_case(directory / f"input-{name}.json", qk,
                                   phases["arrays"]["cosine"][frame * 144:(frame + 1) * 144],
                                   phases["arrays"]["sine"][frame * 144:(frame + 1) * 144],
                                   case, _provenance(context, phases, case))
            cases.append({"name": name, "spec": case, "input": binding})
    plan = {"schema": PLAN_SCHEMA, "policy": policy(), "anchor": {"path": str(anchor.resolve()), "sha256": digest},
            "phase_manifest": phases["manifest"], "source": context["source"],
            "implementations": _implementations(), "cases": cases}
    path = directory / "plan.json"
    with path.open("x") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")
    return {"plan": str(path), "sha256": ref.file_sha256(path), "cases": len(cases)}


def _write_output(report, doc, arrays):
    report, payload = ref.fresh_paths(report)
    raw = bytearray()
    doc["tensors"] = {}
    for name, shape in OUTPUT_SHAPES.items():
        data = ref.checked_array(arrays[name], shape, bf16=name in ("query", "key")).tobytes()
        doc["tensors"][name] = {"dtype": "F32", "shape": list(shape), "offset_bytes": len(raw),
                                "nbytes": len(data), "sha256": ref.sha256(data)}
        raw.extend(data)
    doc["payload"] = {"path": payload.name, "bytes": len(raw), "sha256": ref.sha256(raw), "byte_order": "little"}
    with payload.open("xb") as stream:
        stream.write(raw)
    with report.open("x") as stream:
        stream.write(json.dumps(doc, indent=2) + "\n")
    return report


def run_source(admitted, report, upstream):
    # Verify all pinned dependencies before loading the actual upstream function.
    report, _ = ref.fresh_paths(report)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ValueError("CPU fallback is forbidden")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    cache = capture._load_module(capture.SOURCE_CACHE_PATH, "fixed_phase_cache")
    helper = cache._load_support_oracle()
    upstream, source_hashes, revision = cache._verify_upstream_root(upstream.resolve(), helper)
    posemb_path = upstream / "hymm/models/dit/modules/posemb_layers.py"
    if ref.file_sha256(posemb_path) != POSEMB_SHA:
        raise ValueError("pinned upstream RoPE source changed")
    posemb = capture._load_module(posemb_path, "fixed_phase_posemb")
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("upstream equation requires MPS")
    doc, case = admitted["document"], admitted["document"]["case"]
    with torch.inference_mode():
        query = torch.from_numpy(admitted["qk"]["query"].copy()).to("mps", dtype=torch.bfloat16).unsqueeze(0)
        key = torch.from_numpy(admitted["qk"]["key"].copy()).to("mps", dtype=torch.bfloat16).unsqueeze(0)
        cosine, sine = [torch.from_numpy(admitted[name].copy()).to("mps") for name in ("cosine", "sine")]
        if case["arm"] == "fixed":
            key = torch.roll(key, 17, dims=1)
        query, key = posemb.apply_rotary_emb(query, key, (cosine, sine), head_first=False)
        if case["arm"] == "moving":
            key = torch.roll(key, 17, dims=1)
        query, key = query[0].float(), key[0].float()
        scores = torch.matmul(query.permute(1, 0, 2), key.permute(1, 2, 0)) * (128 ** -0.5)
        diagonal = scores.diagonal(dim1=-2, dim2=-1)
        maximum, argmax = scores.max(dim=-1)
        other = scores.masked_fill(torch.eye(144, dtype=torch.bool, device="mps")[None], -torch.inf).max(dim=-1).values
        greater = (scores > diagonal.unsqueeze(-1)).sum(dim=-1).float()
        self_mass = torch.softmax(scores, dim=-1).diagonal(dim1=-2, dim2=-1)
        metrics = torch.stack((diagonal, maximum, diagonal - other, greater, argmax.float(), self_mass), dim=-1)
        packed = torch.cat([value.reshape(-1) for value in (query, key, scores, metrics)])
        torch.mps.synchronize()
        exported = packed.cpu().numpy()
    arrays, offset = {}, 0
    for name, shape in OUTPUT_SHAPES.items():
        count = int(np.prod(shape))
        arrays[name] = exported[offset:offset + count].reshape(shape)
        offset += count
    return _write_output(report, {
        "schema": "hy-nomas-query-fixed-phase-source-v1", "schema_version": 1,
        "input_manifest_path": admitted["path"], "input_manifest_sha256": admitted["sha256"],
        "input_qk_manifest_sha256": doc["qk"]["sha256"],
        "input_phase_payload_sha256": doc["payload"]["sha256"], "case": case,
        "metrics_order": ref.METRICS,
        "oracle": {"device": "mps", "cpu_fallback": False, "script_sha256": ref.file_sha256(HERE),
                   "torch_version": torch.__version__, "upstream_root": str(upstream), "revision": revision,
                   "source_sha256": source_hashes, "posemb_sha256": ref.file_sha256(posemb_path),
                   "boundary": "BF16 Q/K -> pinned F32 apply_rotary_emb -> BF16 -> F32 QK"},
        "execution": {"explicit_terminal_waits": 1, "joint_exports": 1,
                      "intermediate_readbacks": 0, "model_launched": False}}, arrays)


def summary(scores, metrics, arm):
    scores = ref.checked_array(scores, ref.SCORE_SHAPE)
    metrics = ref.checked_array(metrics, ref.METRIC_SHAPE)
    targets = content_targets(arm)
    selected = scores[:, np.arange(144), targets]
    other = scores.copy()
    other[:, np.arange(144), targets] = -np.inf
    margin = selected - other.max(axis=-1)
    return {str(head): {"strict_diagonal_count": int(np.count_nonzero(metrics[head, :, 2] > 0)),
                        "strict_relocated_content_count": int(np.count_nonzero(margin[head] > 0)),
                        "content_argmax_count": int(np.count_nonzero(metrics[head, :, 4] == targets)),
                        "diagonal_margin_mean": float(np.mean(metrics[head, :, 2], dtype=np.float64)),
                        "content_margin_mean": float(np.mean(margin[head], dtype=np.float64))}
            for head in policy()["heads"]}


def qualify(plan_path, digest, report):
    if report.exists() or report.is_symlink():
        raise ValueError("report must be fresh")
    plan = _json(plan_path, digest)
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("policy") != policy()
            or plan.get("implementations") != _implementations()
            or [item.get("spec") for item in plan.get("cases", [])] != case_specs()):
        raise ValueError("plan policy, implementation or case matrix mismatch")
    context = heads._anchor(Path(plan["anchor"]["path"]), plan["anchor"]["sha256"])
    phases = _phase_context(context, Path(plan["phase_manifest"]["path"]))
    receipt = json.loads(Path(context["source"]["receipt_path"]).read_text())
    source_hashes = receipt["upstream_dependency_sha256"] | receipt["inference_dtype_source_sha256"]
    upstream_root = Path(receipt["upstream_source_root"]).resolve()
    if plan["source"] != context["source"] or plan["phase_manifest"] != phases["manifest"]:
        raise ValueError("plan lineage mismatch")
    results, outputs = {}, {}
    for item, case in zip(plan["cases"], case_specs()):
        name, frame = case_name(case), case["frame"]
        path = Path(item["input"]["path"])
        if item["name"] != name or path != plan_path.parent / f"input-{name}.json":
            raise ValueError("case input path or label mismatch")
        admitted = load_input(path, item["input"]["sha256"])
        validate_binding(admitted, context, phases, case)
        outputs[name] = {}
        reports = {}
        for label in ("source", "native"):
            output_path = plan_path.parent / f"{label}-{name}.json"
            doc = json.loads(_canonical(output_path).read_text())
            if (doc.get("schema") != f"hy-nomas-query-fixed-phase-{label}-v1"
                    or doc.get("schema_version") != 1 or doc.get("case") != case
                    or doc.get("input_manifest_sha256") != admitted["sha256"]
                    or doc.get("input_qk_manifest_sha256") != admitted["document"]["qk"]["sha256"]
                    or doc.get("input_phase_payload_sha256") != admitted["document"]["payload"]["sha256"]
                    or doc.get("metrics_order") != ref.METRICS):
                raise ValueError("output identity or schema mismatch")
            arrays = ref.load_payload(output_path, doc, OUTPUT_SHAPES)
            for short in ("query", "key"):
                ref.checked_array(arrays[short], ref.SHAPE, bf16=True)
            if label == "source":
                oracle = doc["oracle"]
                if (oracle.get("script_sha256") != ref.file_sha256(HERE) or oracle.get("device") != "mps"
                        or oracle.get("cpu_fallback") is not False
                        or oracle.get("revision") != context["source"]["source_revision"]
                        or oracle.get("posemb_sha256") != POSEMB_SHA
                        or oracle.get("source_sha256") != source_hashes
                        or Path(oracle["upstream_root"]) != upstream_root
                        or ref.file_sha256(Path(oracle["upstream_root"]) / "hymm/models/dit/modules/posemb_layers.py") != oracle["posemb_sha256"]
                        or doc.get("execution") != {"explicit_terminal_waits": 1, "joint_exports": 1,
                                                    "intermediate_readbacks": 0, "model_launched": False}):
                    raise ValueError("upstream equation or execution contract mismatch")
            else:
                required = {"device": "metal", "named_encode_allocations": 0, "encode_readbacks": 0,
                            "encode_waits": 0, "terminal_waits": 1, "terminal_readbacks": 4,
                            "command_chains": 1, "named_buffer_count": 10,
                            "named_bytes_after_setup": 8607744, "dispatches": 8,
                            "pre_key_roll": 17 if case["arm"] == "fixed" else 0,
                            "post_key_roll": 17 if case["arm"] == "moving" else 0}
                if any(doc.get("execution", {}).get(k) != v for k, v in required.items()):
                    raise ValueError("native encode/drain execution contract mismatch")
                sources = doc.get("native_sources", {})
                expected_sources = {
                    "production_uncontracted_rope_sha256": ref.file_sha256(REPO / "src/ml/metal/kernels/hy_nomas/hy_nomas_mps_pointwise.metal"),
                    "production_bf16_roundtrip_helper_sha256": ref.file_sha256(REPO / "src/ml/metal/kernels/hy_nomas/hy_nomas_block.metal"),
                    "rope_kernel": "hy_nomas_rope_uncontracted_f32",
                    "bf16_roundtrip_helper": "hy_nomas_bf16_roundtrip_f32"}
                if any(sources.get(k) != v for k, v in expected_sources.items()):
                    raise ValueError("native production RoPE or BF16 helper changed")
            outputs[name][label] = arrays
            reports[label] = {"path": str(output_path), "sha256": ref.file_sha256(output_path)}
        source, native = outputs[name]["source"], outputs[name]["native"]
        checks = {short: ref.compare(native[short], source[short], discrete_columns=(3, 4) if short == "metrics" else ())
                  for short in OUTPUT_SHAPES}
        for label, arrays in outputs[name].items():
            checks[f"{label}_metric_audit"] = ref.audit_metrics(arrays["scores"], arrays["metrics"])
            if case["arm"] == "identity":
                for short, source_name in zip(("query", "key"), heads.BOUNDARIES["post"]):
                    expected = capture._expand_bf16(heads._frame(context["raw"][source_name], frame), ref.SHAPE)
                    checks[f"{label}_{short}_saved_identity"] = {"passed": arrays[short].tobytes() == expected.tobytes()}
        results[name] = {"case": case, "reports": reports, "checks": checks,
                         "summary": summary(native["scores"], native["metrics"], case["arm"])}
    rolls, contrasts = {}, {}
    for frame in policy()["frames"]:
        for label in ("source", "native"):
            base, moving, fixed = [outputs[f"f{frame:02}-{arm}"][label] for arm in ("identity", "moving", "fixed")]
            checks = heads.check_roll(base["scores"], moving["scores"], moving["metrics"])
            checks["query_same_all_arms"] = np.array_equal(base["query"], moving["query"]) and np.array_equal(base["query"], fixed["query"])
            checks["moving_key_exact_roll"] = np.array_equal(moving["key"], np.roll(base["key"], 17, axis=0))
            checks["passed"] = all(checks.values())
            rolls[f"f{frame:02}-{label}"] = checks
            contrasts[f"f{frame:02}-{label}"] = {
                "fixed_vs_moving_key_values_changed": int(np.count_nonzero(fixed["key"] != moving["key"])),
                "fixed_vs_moving_key_max_abs": float(np.max(np.abs(fixed["key"] - moving["key"]))),
                "fixed_vs_moving_score_max_abs": float(np.max(np.abs(fixed["scores"] - moving["scores"]))),
                "interpretation": "Magnitude of the phase contrast; zero is a null effect, not phase sensitivity"}
    passed = (all(check["passed"] for result in results.values() for check in result["checks"].values())
              and all(check["passed"] for check in rolls.values()))
    result = {"schema": "hy-nomas-query-fixed-phase-report-v1", "passed": bool(passed),
              "plan": {"path": str(plan_path), "sha256": digest}, "policy": policy(),
              "cases": results, "moving_arm_equivariance": rolls, "phase_contrasts": contrasts,
              "scope": "One saved T0 state; fixed-phase content null, no semantic labels, cross-state transport, causal rollout or acting claim"}
    with report.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--anchor", type=Path, required=True)
    prep.add_argument("--anchor-sha256", required=True)
    prep.add_argument("--phases", type=Path, required=True)
    prep.add_argument("--directory", type=Path, required=True)
    source = commands.add_parser("source")
    source.add_argument("--manifest", type=Path, required=True)
    source.add_argument("--manifest-sha256", required=True)
    source.add_argument("--upstream", type=Path, required=True)
    source.add_argument("--report", type=Path, required=True)
    source.add_argument("--execute", action="store_true")
    check = commands.add_parser("qualify")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--plan-sha256", required=True)
    check.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.anchor, args.anchor_sha256, args.phases, args.directory)))
    elif args.command == "source":
        if not args.execute:
            parser.error("source GPU execution requires --execute")
        print(json.dumps({"report": str(run_source(load_input(args.manifest, args.manifest_sha256), args.report, args.upstream))}))
    else:
        result = qualify(args.plan, args.plan_sha256, args.report)
        print(json.dumps({"passed": result["passed"], "report": str(args.report),
                          "summaries": {name: item["summary"] for name, item in result["cases"].items()}}))
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
