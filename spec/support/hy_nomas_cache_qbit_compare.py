#!/usr/bin/env python3
"""Post-drain diagnostics for an exact/QBit continuation pair.

This authenticates terminal artifacts and reports lossy error. It deliberately
does not promote a local error threshold to a visual-quality or upstream gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

import hy_nomas_sequence_resume_reference as exact


CODEBOOK_SHA256 = "028a7941b371e7b03331acfc686965b6087f40001165ec27d608f22c72d99ed1"
POLICY = {"name": "hy-nomas-cache-qbit-row-error-v1", "row_values": 128,
          "header_bytes_per_row": 16, "p4_max_error_over_sigma": 0.50,
          "p4_rms_error_over_sigma": 0.12, "p5_max_error_over_sigma": 0.25,
          "p5_rms_error_over_sigma": 0.06, "p8_max_error_over_sigma": 0.10,
          "p8_rms_error_over_sigma": 0.02, "escape": "exact-f32"}


def _temporal(payload: bytes, start: int, stop: int) -> bytes:
    stride = 180 * 9 * 16 * 4
    frame = 9 * 16 * 4
    return b"".join(payload[c * stride + start * frame:c * stride + stop * frame]
                    for c in range(32))


def _error(reference: bytes, actual: bytes) -> dict:
    report = exact._compare_f32(reference, actual, "lossy diagnostic")
    squared_error = squared_reference = absolute_error = 0.0
    for (a,), (b,) in zip(struct.iter_unpack("<f", reference), struct.iter_unpack("<f", actual)):
        squared_error += (b - a) ** 2
        squared_reference += a * a
        absolute_error += abs(b - a)
    count = report["elements"]
    report.update(rmse=math.sqrt(squared_error / count),
                  mean_absolute_error=absolute_error / count,
                  relative_l2=math.sqrt(squared_error / max(squared_reference, 1e-300)))
    return report


def compare_runs(exact_dir: str, qbit_dir: str) -> dict:
    directories = {"exact": exact._directory(exact_dir, "exact"),
                   "qbit": exact._directory(qbit_dir, "qbit")}
    receipts = {name: exact._read_json(path / "receipt.json", label=name)
                for name, path in directories.items()}
    control, quantized = receipts["exact"], receipts["qbit"]
    for name, receipt in receipts.items():
        codec_bytes = receipt.get("codec_bytes")
        if type(codec_bytes) is not int or codec_bytes < 0:
            raise exact.ResumeReferenceError("invalid codec budget")
        normalized = dict(receipt, peak_named_bytes=receipt["peak_named_bytes"] - codec_bytes)
        exact._validate_receipt(normalized, "resume")
        if receipt.get("cache_roundtrip") != name:
            raise exact.ResumeReferenceError("wrong cache treatment")
    if control["codec_bytes"] != 0 or control.get("codec") is not None:
        raise exact.ResumeReferenceError("control must retain exact F32")
    if control["pid"] == quantized["pid"]:
        raise exact.ResumeReferenceError("the two arms require distinct fresh processes")
    for key in ("inputs_sha256", "contract_sha256", "state_input_sha256", "upstream_revision",
                "timeline_shape", "step_shape", "segments", "steps_per_segment",
                "checkpoint_payload_bytes", "dit_peak_named_bytes", "capture_bytes", "valid_text_tokens"):
        if control[key] != quantized[key]:
            raise exact.ResumeReferenceError(f"pair identity mismatch: {key}")
    exact._required_sha(control, "state_input_sha256", "exact")
    codec = quantized.get("codec")
    if not isinstance(codec, dict) or codec.get("invalid_rows") != 0:
        raise exact.ResumeReferenceError("QBit stats missing or contain nonfinite rows")
    if codec.get("schema") != "hy-nomas-cache-qbit-report-v1":
        raise exact.ResumeReferenceError("QBit codec schema mismatch")
    policy = codec.get("policy")
    if not isinstance(policy, dict) or any(policy.get(k) != v for k, v in POLICY.items()):
        raise exact.ResumeReferenceError("QBit selector policy mismatch")
    codebook = codec.get("codebook")
    if not isinstance(codebook, dict) or codebook.get("sha256") != CODEBOOK_SHA256:
        raise exact.ResumeReferenceError("QBit Gaussian codebook mismatch")
    if codec.get("resident_bytes") != quantized["codec_bytes"]:
        raise exact.ResumeReferenceError("codec resident bytes do not match admission")
    records = codec.get("records")
    names = {f"history.layer{i}.{kind}" for i in range(54) for kind in ("key", "value")}
    if not isinstance(records, list) or len(records) != 108 or {r.get("name") for r in records} != names:
        raise exact.ResumeReferenceError("QBit cache inventory mismatch")
    logical = original = 0
    totals = dict.fromkeys(("p4", "p5", "p8", "f32"), 0)
    valid_text = control["valid_text_tokens"]
    if type(valid_text) is not int or not 0 < valid_text <= 1280:
        raise exact.ResumeReferenceError("invalid authenticated text extent")
    expected_count = (4320 + valid_text) * 16 * 128
    for record in records:
        rows = record.get("rows")
        counts = record.get("tier_counts", {})
        if record.get("count") != expected_count or rows != expected_count // 128:
            raise exact.ResumeReferenceError("QBit row geometry mismatch")
        if set(counts) != set(totals) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise exact.ResumeReferenceError("invalid QBit tier counts")
        packed = rows * 16 + sum(counts[t] * n for t, n in (("p4", 64), ("p5", 80), ("p8", 128), ("f32", 512)))
        if sum(counts.values()) != rows or record.get("logical_packed_bytes") != packed:
            raise exact.ResumeReferenceError("QBit compact byte accounting mismatch")
        if record.get("exact_bytes") != expected_count * 4 or record.get("invalid_rows") != 0:
            raise exact.ResumeReferenceError("QBit exact extent or finite gate mismatch")
        logical += packed
        original += expected_count * 4
        for tier in totals:
            totals[tier] += counts[tier]
    if (codec.get("logical_packed_bytes") != logical or codec.get("exact_bytes") != original
            or codec.get("tier_counts") != totals):
        raise exact.ResumeReferenceError("QBit aggregate accounting mismatch")
    artifacts = {}
    for name, receipt in receipts.items():
        artifacts[name] = {}
        for artifact, size in (("timeline", exact.TIMELINE_BYTES),
                               *((s, exact.STEP_BYTES) for s in exact.STEP_NAMES)):
            artifacts[name][artifact] = exact._read_artifact(
                directories[name], receipt["diagnostics"], artifact, f"{artifact}.f32le", size, name)[0]
    a, b = artifacts["exact"]["timeline"], artifacts["qbit"]["timeline"]
    unchanged_a = _temporal(a, 0, 30) == _temporal(b, 0, 30)
    unchanged_future = _temporal(a, 60, 180) == _temporal(b, 60, 180)
    if not unchanged_a or not unchanged_future:
        raise exact.ResumeReferenceError("QBit arm modified A or untouched future noise")
    timeline = _error(a, b)
    steps = {s: _error(artifacts["exact"][s], artifacts["qbit"][s]) for s in exact.STEP_NAMES}
    if timeline["bitwise_equal"] or all(s["bitwise_equal"] for s in steps.values()):
        raise exact.ResumeReferenceError("lossy treatment is inactive; no visual contrast to test")
    return {"schema": "cogni_ml.hy_nomas_cache_qbit_comparison/v1", "pair_integrity_passed": True,
            "identity": {key: control[key] for key in ("inputs_sha256", "contract_sha256", "state_input_sha256")},
            "a_latent_bitwise_equal": unchanged_a, "future_noise_bitwise_equal": unchanged_future,
            "timeline": timeline, "b_latent": _error(_temporal(a, 30, 60), _temporal(b, 30, 60)),
            "b_steps": steps, "codec": codec,
            "quality_verdict": "pending visual review; error statistics are diagnostic, not a quality gate",
            "storage_scope": "logical compact K/V representation; F32 attention cache remains resident; exact restart bundle unchanged"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--qbit-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_runs(args.exact_dir, args.qbit_dir)
    exact._write_fresh_json(Path(args.output).absolute(), result)
    print(json.dumps({"pair_integrity_passed": True, "output": args.output}))


if __name__ == "__main__":
    main()
