#!/usr/bin/env python3
"""Offline comparator for the bounded native Conv3D reduction diagnostic.

CPU arithmetic here runs only on post-drain exported evidence. Gate status is
explicit: controls and a required candidate are distinct from observed errors.
"""
from __future__ import annotations

import argparse
from array import array
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import struct
import sys

ATOL = RTOL = 2e-5
VECTOR_K, VECTOR_COUNT = 32, 256


def raw(path: Path, code: str, count: int) -> array:
    values = array(code)
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != count:
        raise ValueError(f"{path}: expected {count} elements, got {len(values)}")
    return values


def write_raw(path: Path, code: str, values) -> None:
    if path.exists():
        raise FileExistsError(path)
    a = array(code, values)
    if sys.byteorder != "little":
        a.byteswap()
    path.write_bytes(a.tobytes())


def bf16(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def round_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7fffffff > 0x7f800000:
        return bits >> 16 | 0x40
    return ((bits + 0x7fff + (bits >> 16 & 1)) >> 16) & 0xffff


def fraction(bits: int) -> Fraction:
    exponent, mantissa = bits >> 7 & 255, bits & 127
    if exponent == 255:
        raise ValueError("Exact-vector source must be finite")
    if exponent:
        mantissa |= 128
    power = max(exponent, 1) - 134
    value = Fraction(mantissa << max(power, 0), 1 << max(-power, 0))
    return -value if bits & 0x8000 else value


def exact_bits(value: Fraction, precision: int) -> int:
    """Direct RN-even to BF16 (p8) or IEEE F32 (p24), with no double round."""
    if precision not in (8, 24):
        raise ValueError("Only BF16/F32 validation formats are admitted")
    sign = int(value < 0) << (precision + 7)
    value = abs(value)
    if value == 0:
        return 0
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    below_power = n < d << exponent if exponent >= 0 else n << -exponent < d
    if below_power:
        exponent -= 1
    low = max(exponent, -126) - (precision - 1)
    numerator, denominator = (n, d << low) if low >= 0 else (n << -low, d)
    significand, remainder = divmod(numerator, denominator)
    if remainder * 2 > denominator or remainder * 2 == denominator and significand & 1:
        significand += 1
    if exponent < -126:
        return sign | significand  # includes carry into the minimum normal
    if significand == 1 << precision:
        significand >>= 1
        exponent += 1
    if exponent > 127:
        return sign | 255 << (precision - 1)
    return sign | (exponent + 127) << (precision - 1) | (significand & ((1 << (precision - 1)) - 1))


def make_vectors(root: Path) -> None:
    root.mkdir()
    rng = random.Random(0xdcae2026)
    rows = [
        [(0x3f80, 0x3f80), (0x3b80, 0x3f80)],  # BF16 midpoint, even down
        [(0x3f81, 0x3f80), (0x3b80, 0x3f80)],  # midpoint, even up
        [(0x3f80, 0x3f80), (0x3b80, 0x3f80), (1, 1)],  # just above midpoint
        [(0x3f80, 0x3f80), (0x3b80, 0x3f80), (0x8001, 1)],
        [(1, 0x3f00)], [(3, 0x3f00)], [(0x7f, 0x3f80), (1, 0x3f00)],
        [(0x7f7f, 0x7f7f), (0xff7f, 0x7f7f), (1, 1)],
        [(0x7f7f, 0x4000)], [(0xff7f, 0x4000)],
        [(0x3f80, 0x3f80), (0xbf80, 0x3f80)],
    ]
    exponents = [0, 1, 10, 55, 100, 120, 126, 127, 128, 150, 200, 254]
    while len(rows) < VECTOR_COUNT:
        row = []
        for _ in range(VECTOR_K):
            pair = tuple(rng.randrange(2) << 15 | rng.choice(exponents) << 7 | rng.randrange(128) for _ in range(2))
            row.append(pair)
        rows.append(row)
    expected = {"bf16": [], "f32": []}
    x, w = [], []
    for row in rows:
        row += [(0, 0)] * (VECTOR_K - len(row))
        x.extend(a for a, _ in row)
        w.extend(b for _, b in row)
        total = sum((fraction(a) * fraction(b) for a, b in row), Fraction())
        expected["bf16"].append(exact_bits(total, 8))
        expected["f32"].append(exact_bits(total, 24))
    write_raw(root / "x.bf16", "H", x)
    write_raw(root / "w.bf16", "H", w)
    (root / "expected.json").write_text(json.dumps(expected, sort_keys=True) + "\n")


def metrics(actual, reference, indices, lo=0, hi=None) -> dict:
    hi = len(actual) if hi is None else hi
    if not 0 <= lo < hi <= len(actual) or len(actual) != len(reference) or len(actual) != len(indices):
        raise ValueError("Invalid or empty comparison range")
    bit, fails, nonfinite = 0, 0, 0
    max_abs, squared = 0.0, 0.0
    first = []
    for i in range(lo, hi):
        a, b = bf16(actual[i]), bf16(reference[i])
        bit += actual[i] != reference[i]
        finite = math.isfinite(a) and math.isfinite(b)
        nonfinite += not finite
        delta = abs(a - b) if finite else math.inf
        max_abs = max(max_abs, delta)
        squared += delta * delta
        fail = not finite or delta > ATOL + RTOL * abs(b)
        fails += fail
        if fail and len(first) < 8:
            first.append({"sample": i, "coordinate": indices[i], "actual_bits": actual[i],
                          "reference_bits": reference[i], "actual": a if math.isfinite(a) else None,
                          "reference": b if math.isfinite(b) else None})
    return {"count": hi - lo, "bit_mismatches": bit, "tolerance_failures": fails,
            "nonfinite": nonfinite, "max_abs": None if nonfinite else max_abs,
            "rms": None if nonfinite else math.sqrt(squared / (hi - lo)),
            "first_failures": first}


def compare(root: Path, require_candidate: str | None, require_upstream: bool, positive_control: bool) -> tuple[dict, bool]:
    m = json.loads((root / "manifest.json").read_text())
    if m["schema"] != "dcae_conv3d_reduction_probe/v1" or m["samples"] != 8192 or m["uniform_samples"] != 4096:
        raise ValueError("Unsupported probe schema/sampling geometry")
    n = m["samples"]
    total = math.prod(m["shape"])
    expected = json.loads((Path(m["vectors_directory"]) / "expected.json").read_text())
    vector_pre = raw(root / "vectors.bf16", "H", VECTOR_COUNT)
    vector_raw = raw(root / "vectors.f32", "I", VECTOR_COUNT)
    vector_failures = {"bf16": sum(a != b for a, b in zip(vector_pre, expected["bf16"], strict=True)),
                       "f32": sum(a != b for a, b in zip(vector_raw, expected["f32"], strict=True))}
    controls_ok = not any(vector_failures.values()) and m["terminal_waits"] == 1 and all(v == 0 for v in m["hot_metrics"].values())
    result = {"schema": "dcae_conv3d_reduction_comparison/v1", "atol": ATOL, "rtol": RTOL,
              "shape": m["shape"], "sampling": m["sampling"], "exact_vector_failures": vector_failures, "cases": {}}
    required_ok = True
    positive_failures = 0
    for case in ("invalid", "valid"):
        indices = raw(root / f"{case}-indices.u32", "I", n)
        if any(i >= total for i in indices):
            raise ValueError("Sample index outside full operator geometry")
        stats = raw(root / f"{case}-full-stats.u32", "I", 6)
        controls_ok &= all(stats[i] == 0 for i in (0, 1, 2, 5))
        reference = raw(root / f"{case}-mps-pre.bf16", "H", n)
        post = raw(root / f"{case}-mps-post.bf16", "H", n)
        original = raw(root / f"{case}-original.bf16", "H", n)
        controls_ok &= post == original
        f32 = raw(root / f"{case}-mps-f32.f32", "f", n)
        sampled_round = metrics([round_bf16(v) for v in f32], reference, indices)
        controls_ok &= sampled_round["nonfinite"] == 0
        controls_ok &= not (stats[3] == 0 and sampled_round["bit_mismatches"] != 0)
        controls_ok &= not (stats[4] == 0 and sampled_round["tolerance_failures"] != 0)
        report = {"full_output_control": dict(zip(m["full_stats"], stats, strict=True)),
                  # Independent post-drain fallback for the GPU full-output
                  # round/compare counter; it is still the separate F32 lane.
                  "sampled_mps_f32_round_vs_bf16_pre": sampled_round,
                  "candidates": {}}
        for name in m["candidates"]:
            candidate = raw(root / f"{case}-{name}.pre.bf16", "H", n)
            raw_f32 = raw(root / f"{case}-{name}.f32", "f", n)
            f32_nonfinite = sum(not math.isfinite(a) or not math.isfinite(b)
                                for a, b in zip(raw_f32, f32, strict=True))
            detail = {"pre": metrics(candidate, reference, indices),
                      "boundary": metrics(candidate, reference, indices, 0, 512),
                      "stratified": metrics(candidate, reference, indices, 512, 4096),
                      "near_midpoint": metrics(candidate, reference, indices, 4096, n),
                      "f32_nonfinite": f32_nonfinite,
                      "max_f32_delta_to_separate_mps_f32": None if f32_nonfinite else max(abs(a - b) for a, b in zip(raw_f32, f32, strict=True))}
            candidate_post = raw(root / f"{case}-{name}.post.bf16", "H", n)
            detail["post"] = metrics(candidate_post, post, indices)
            report["candidates"][name] = detail
            if name == require_candidate:
                required_ok &= detail["pre"]["tolerance_failures"] == 0 and detail["post"]["tolerance_failures"] == 0 and f32_nonfinite == 0
            if name == "order0-lanes1-tree0":
                positive_failures += detail["pre"]["tolerance_failures"]
        oracle = root / f"torch-{case}-pre.bf16"
        if oracle.exists():
            oracle_indices = raw(root / f"torch-{case}-indices.u32", "I", n)
            if oracle_indices != indices:
                raise ValueError("Upstream sample coordinate lineage differs")
            report["upstream_pre"] = metrics(reference, raw(oracle, "H", n), indices)
            report["upstream_post"] = metrics(post, raw(root / f"torch-{case}-post.bf16", "H", n), indices)
            report["upstream_biased_call_vs_separate"] = metrics(
                raw(root / f"torch-{case}-biased.bf16", "H", n),
                raw(root / f"torch-{case}-post.bf16", "H", n), indices)
            controls_ok &= report["upstream_pre"]["bit_mismatches"] == 0 and report["upstream_post"]["bit_mismatches"] == 0
            controls_ok &= report["upstream_biased_call_vs_separate"]["bit_mismatches"] == 0
        elif require_upstream:
            controls_ok = False
        result["cases"][case] = report
    if require_candidate and require_candidate not in m["candidates"]:
        raise ValueError("Required candidate absent")
    controls_ok &= not positive_control or positive_failures > 0
    result.update(controls_passed=bool(controls_ok), required_candidate=require_candidate,
                  required_candidate_passed=bool(required_ok) if require_candidate else None,
                  ordered_fma_positive_control_failures=positive_failures,
                  gate_passed=bool(controls_ok and require_candidate is not None and required_ok),
                  scope="Synthetic real-geometry operator; sampled native evidence only. No hidden MPS schedule, real-activation, full-model or speedup claim.")
    return result, result["gate_passed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--make-vectors", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--require-candidate")
    mode.add_argument("--observe-only", action="store_true", help="Exit status checks controls; never a candidate gate pass")
    parser.add_argument("--require-upstream", action="store_true")
    parser.add_argument("--positive-control", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.make_vectors:
        make_vectors(args.root)
        return 0
    if not args.require_candidate and not args.observe_only:
        parser.error("Choose --require-candidate NAME or --observe-only explicitly")
    result, passed = compare(args.root, args.require_candidate, args.require_upstream, args.positive_control)
    result["mode"] = "observation-only" if args.observe_only else "required-candidate"
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.write_text(encoded)
    summary = {k: v for k, v in result.items() if k != "cases"}
    summary["pre_failures"] = {c: {n: d["pre"]["tolerance_failures"] for n, d in v["candidates"].items()} for c, v in result["cases"].items()}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if (result["controls_passed"] if args.observe_only else passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
