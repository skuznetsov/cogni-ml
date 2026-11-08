#!/usr/bin/env python3
"""Offline comparator for the bounded native Conv3D reduction diagnostic.

CPU arithmetic here runs only on post-drain exported evidence. Gate status is
explicit: controls and a required candidate are distinct from observed errors.
"""
from __future__ import annotations

import argparse
from array import array
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

ATOL = RTOL = 2e-5
VECTOR_K, VECTOR_COUNT = 32, 256
PREFIX_TARGET = "decoder.stages.2.op_list.1.main.conv1.conv"
PREFIX_UPSTREAM_SHA = "38163ebead9a6bb643d27843022c1757c45c6b8638c334a476aee5b373edf8c1"
PREFIX_UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
PREFIX_CHECKPOINT_SHA = "ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"
PREFIX_LINEAGE_SHA = "7961bd5f4ed32793e83ddd5eb5e982c49a3958d647f5718046ce3f867eaa256c"
PREFIX_SOURCE_REFERENCE_SHA = "0e873cba80446ef26920469ef3c1700f6e1686d021fb35b973872de4e383c2c7"
PREFIX_INPUT_SHA = "43081cf72cc0feda5edff7531f449209a542a2fb8d5ba8ae15d3067a06462a3c"


def prefix_report_consistency(reference: dict, expected: dict) -> dict:
    """Check pinned report claims; source/launch receipts establish execution separately."""
    checkpoint = reference.get("checkpoint", {})
    cache = reference.get("native_weight_cache", {})
    source = reference.get("upstream", {})
    fixture = reference.get("reference", {})
    loaded = reference.get("prefix_weight_load", {})
    return {
        "schema": reference.get("schema") == "cogni_ml.dcae_v_activation_prefix_upstream/v1",
        "flat_boundary": all(reference.get(k) == v for k, v in expected.items()),
        "original_checkpoint": checkpoint.get("sha256") == PREFIX_CHECKPOINT_SHA,
        "original_upstream": (source.get("revision") == PREFIX_UPSTREAM_REVISION and
                              source.get("module_sha256") == PREFIX_UPSTREAM_SHA),
        "native_cache_join": (cache.get("sha256") == expected["weights_sha256"] and
                              cache.get("selected_prefix_tensor_count") == 90 and
                              cache.get("selected_prefix_bytes_bitwise_joined_to_original_checkpoint") == 2096936960),
        "reference_lineage": (fixture.get("sha256") == expected["reference_sha256"] and
                              fixture.get("tensor") == "input.latent" and
                              fixture.get("tensor_dtype") == "BF16" and
                              fixture.get("tensor_shape") == [1, 32, 20, 9, 16] and
                              fixture.get("tensor_sha256") == PREFIX_INPUT_SHA and
                              fixture.get("lineage_audit_sha256") == PREFIX_LINEAGE_SHA and
                              fixture.get("full_source_reference_sha256") == PREFIX_SOURCE_REFERENCE_SHA),
        "bounded_weight_load": (loaded.get("materialized_tensor_count") == 90 and
                                loaded.get("materialized_bf16_bytes") == 2096936960 and
                                loaded.get("encoder_weights_materialized") is False and
                                loaded.get("later_decoder_weights_materialized") is False and
                                loaded.get("target_conv_weights_materialized") is False),
    }


def file_digest(path: Path, expected_bytes: int) -> str:
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Wrong full prefix tensor extent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_cache_invariant(root: Path, shape: list[int]) -> dict:
    """Audit old-cache transport after drain, independent of either prefix."""
    batch, channels, frames, height, width = shape
    if batch != 1 or frames < 2:
        raise ValueError("Unsupported cache audit geometry")
    frame_bytes = height * width * 2
    with (root / "invalid-cache.bf16").open("rb") as stream:
        cold_zero = all(not any(chunk) for chunk in iter(lambda: stream.read(1024 * 1024), b""))
    tail = hashlib.sha256()
    with (root / "invalid-input.bf16").open("rb") as stream:
        for channel in range(channels):
            stream.seek((channel * frames + frames - 1) * frame_bytes)
            raw_frame = stream.read(frame_bytes)
            if len(raw_frame) != frame_bytes:
                raise ValueError("Truncated first prefix activation")
            tail.update(raw_frame)
    cached_sha = file_digest(root / "valid-cache.bf16", channels * frame_bytes)
    return {"cold_cache_all_zero": cold_zero, "previous_activation_tail_sha256": tail.hexdigest(),
            "old_cache_sha256": cached_sha, "cached_input_is_previous_activation_tail": tail.hexdigest() == cached_sha}


def authenticate_prefix(root: Path, manifest: dict, oracle: Path | None) -> tuple[dict | None, bool]:
    """Post-drain, full-byte prefix check, separate from sampled operator checks."""
    prefix = manifest.get("activation_prefix")
    if not prefix:
        if oracle is not None:
            raise ValueError("Prefix oracle supplied for a synthetic fixture")
        return None, True
    expected = {"reference_path": prefix["reference_path"], "reference_sha256": prefix["reference_sha256"],
                "target_conv": PREFIX_TARGET, "tile_count": 2,
                "activation_shape": [1, 512, 40, 72, 128], "cache_shape": [1, 512, 1, 72, 128],
                "weights_sha256": manifest["weights_sha256"], "upstream_sha256": PREFIX_UPSTREAM_SHA}
    if (manifest["shape"] != expected["activation_shape"] or
            any(prefix.get(k) != expected[k] for k in ("target_conv", "tile_count", "activation_shape", "cache_shape"))):
        raise ValueError("Unadmitted native prefix boundary")
    result = {"scope": "Full BF16 bytes for each real prefix activation and the OLD target cache, independent of Conv3D output samples",
              "required": True, "provenance_matches": False, "files": {}}
    if oracle is None or not (oracle / "upstream-prefix.json").is_file():
        result["failure"] = "Separate upstream prefix evidence is required"
        return result, False
    reference = json.loads((oracle / "upstream-prefix.json").read_text())
    result["reported_lineage_checks"] = prefix_report_consistency(reference, expected)
    result["provenance_matches"] = all(result["reported_lineage_checks"].values())
    result["execution_evidence"] = "Report consistency is not proof of execution; use the separate pinned source/launch receipts"
    ok = result["provenance_matches"]
    for case in ("invalid", "valid"):
        for suffix, shape in (("input", expected["activation_shape"]), ("cache", expected["cache_shape"])):
            name = f"{case}-{suffix}.bf16"
            size = math.prod(shape) * 2
            actual_sha, reference_sha = file_digest(root / name, size), file_digest(oracle / name, size)
            equal = actual_sha == reference_sha
            artifact = reference.get("artifacts", {}).get(name, {})
            metadata_matches = (artifact.get("bytes") == size and artifact.get("dtype") == "BF16" and
                                artifact.get("shape") == shape and artifact.get("sha256") == reference_sha)
            result["files"][name] = {"bytes": size, "native_sha256": actual_sha,
                                     "upstream_sha256": reference_sha, "bitwise_equal": equal,
                                     "upstream_artifact_record_matches": metadata_matches}
            ok &= equal and metadata_matches
    transport = causal_cache_invariant(root, expected["activation_shape"])
    result["causal_cache_transport"] = transport
    ok &= transport["cold_cache_all_zero"] and transport["cached_input_is_previous_activation_tail"]
    result["passed"] = bool(ok)
    return result, bool(ok)


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


def compare(root: Path, require_candidate: str | None, require_upstream: bool, positive_control: bool,
            prefix_oracle: Path | None = None) -> tuple[dict, bool]:
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
    prefix_report, prefix_ok = authenticate_prefix(root, m, prefix_oracle)
    controls_ok &= prefix_ok
    result = {"schema": "dcae_conv3d_reduction_comparison/v1", "atol": ATOL, "rtol": RTOL,
              "shape": m["shape"], "sampling": m["sampling"], "exact_vector_failures": vector_failures,
              "activation_prefix": prefix_report, "cases": {}}
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
                  scope=("Real decoder-prefix operator inputs, independently authenticated in full; sampled native reductions only. No hidden MPS schedule, full-decoder or speedup claim."
                         if prefix_report and prefix_ok else "Synthetic or unauthenticated real-geometry operator; sampled native evidence only. No hidden MPS schedule, real-activation parity, full-model or speedup claim."))
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
    parser.add_argument("--prefix-oracle", type=Path, help="Separate upstream full-prefix activation/cache exports; required for real inputs")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.make_vectors:
        make_vectors(args.root)
        return 0
    if not args.require_candidate and not args.observe_only:
        parser.error("Choose --require-candidate NAME or --observe-only explicitly")
    result, passed = compare(args.root, args.require_candidate, args.require_upstream, args.positive_control, args.prefix_oracle)
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
