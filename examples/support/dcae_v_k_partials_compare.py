#!/usr/bin/env python3
"""Validate and summarize the drained DCAE Conv3D reduction observations."""

from __future__ import annotations

import argparse
from array import array
from ctypes import c_float
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

from dcae_v_k_partials import (
    CASES,
    SAMPLES,
    SHADER,
    SHAPE,
    SINGLETON_CHANNEL_STRIDE,
    SPARSE_TILES,
    TILE_COUNT,
    TILE_WIDTH,
    UPSTREAM_SHA,
    fixture_pins,
)
from dcae_v_reduction_compare import ATOL, RTOL, bf16, round_bf16

POLICIES = ("ff", "fr", "rf", "rr")
ARMS = ("full", *(f"local.{tile}" for tile in range(len(SPARSE_TILES))), "union")
SCOPES = {
    "boundary": range(512),
    "stratified": range(512, 4096),
    "midpoint_enriched": range(4096, SAMPLES),
}
PAIRS = (("ff", "rf"), ("fr", "rr"), ("ff", "fr"), ("rf", "rr"))


def tree_names():
    for left, right in combinations(range(4), 2):
        third, last = (i for i in range(4) if i not in (left, right))
        yield f"chain-{left}{right}{third}{last}"
        yield f"chain-{left}{right}{last}{third}"
    for right in (1, 2, 3):
        third, last = (i for i in (1, 2, 3) if i != right)
        yield f"pair-0{right}{third}{last}"


TREES = tuple(tree_names())


def replay(tree, parts):
    mode, order = tree.split("-")
    a, b, c, d = (parts[int(i)] for i in order)
    first = c_float(a + b).value
    if mode == "pair":
        return c_float(first + c_float(c + d).value).value
    return c_float(c_float(first + c).value + d).value


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_column(path, code, digest, count):
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, f"Digest mismatch: {path}")
    values = array(code, payload)
    require(len(values) == count, f"Unexpected extent: {path}")
    return values


def failures(candidate, reference):
    return {
        i
        for i, (a, b) in enumerate(zip(candidate, reference, strict=True))
        if abs(a - b) > ATOL + RTOL * abs(b)
    }


def changed(left, right, positions=range(SAMPLES)):
    return sum(left[i] != right[i] for i in positions)


def measure(candidate, reference):
    return {
        "numeric_changed": changed(candidate, reference),
        "gate_failures": len(failures(candidate, reference)),
        "max_abs": max(abs(a - b) for a, b in zip(candidate, reference, strict=True)),
    }


def compare_case(fixture, pins, case, rows):
    def retained(suffix, code):
        return read_column(fixture / suffix, code, pins[suffix], SAMPLES)

    indices = retained(f"{case}-indices.u32", "I")
    bf16_control = array("f", map(bf16, retained(f"torch-{case}-pre.bf16", "H")))
    f32_control = retained(f"{case}-mps-f32.f32", "f")
    previous = list(map(bf16, retained(f"{case}-order0-lanes1-tree0.pre.bf16", "H")))
    require(
        len(set(indices)) == SAMPLES and max(indices) < math.prod(SHAPE),
        f"{case}: invalid sampled coordinates",
    )
    require(
        rows["full.upstream.bf16"].tobytes() == bf16_control.tobytes()
        and rows["full.upstream.f32"].tobytes() == f32_control.tobytes(),
        f"{case}: fresh upstream baseline differs from the retained control",
    )
    old_failed = failures(previous, bf16_control)
    require(
        old_failed and all(map(math.isfinite, previous)),
        f"{case}: no valid prior failure set",
    )

    for name, values in rows.items():
        if name.endswith(".f32"):
            rounded = rows[f"{name.removesuffix('.f32')}.rounded"]
            require(
                all(
                    bf16(round_bf16(v)) == r
                    for v, r in zip(values, rounded, strict=True)
                ),
                f"{case}: invalid BF16 rounding for {name}",
            )
    for arm in ARMS:
        require(
            rows[f"{arm}.upstream.bf16"] == rows[f"{arm}.upstream.rounded"],
            f"{case}: upstream BF16 differs from rounded F32 for {arm}",
        )

    variants = {}
    for policy in POLICIES:
        prefix = f"full.native.{policy}"
        failed = failures(rows[f"{prefix}.rounded"], bf16_control)
        variants[policy] = {
            "bf16_failures": len(failed),
            "bf16_failures_on_prior_points": len(failed & old_failed),
            "bf16_failures_by_scope": {
                name: sum(i in failed for i in positions)
                for name, positions in SCOPES.items()
            },
            "f32_vs_upstream": measure(rows[f"{prefix}.f32"], f32_control),
        }

    paired = {}
    for left, right in PAIRS:
        effects = {
            fmt: changed(
                rows[f"full.native.{left}.{fmt}"],
                rows[f"full.native.{right}.{fmt}"],
                old_failed,
            )
            for fmt in ("f32", "rounded")
        }
        require(
            effects["f32"], f"{case}: {left}/{right} did not exercise the F32 factor"
        )
        paired[f"{left}/{right}"] = {
            "f32_changed": effects["f32"],
            "bf16_changed": effects["rounded"],
        }

    frame_volume = math.prod(SHAPE[2:])
    singleton = [
        p
        for p, index in enumerate(indices)
        if (index // frame_volume) % SINGLETON_CHANNEL_STRIDE == 0
    ]
    parts = [rows[f"local.{tile}.upstream.f32"] for tile in range(len(SPARSE_TILES))]
    local = {}
    for tile, part in enumerate(parts):
        local[str(SPARSE_TILES[tile])] = {}
        for order in ("f", "r"):
            candidate = rows[f"local.{tile}.native.{order}.f32"]
            require(
                not changed(candidate, part, singleton),
                f"{case}: singleton local control failed",
            )
            local[str(SPARSE_TILES[tile])][order] = measure(candidate, part)

    active = sum(all(part[p] != 0 for part in parts) for p in singleton)
    require(active, f"{case}: no singleton controls with four nonzero products")
    union = rows["union.upstream.f32"]
    union_bf16 = rows["union.upstream.bf16"]
    merged = [rows[f"union.merge.{tree}.f32"] for tree in TREES]
    merged_bf16 = [rows[f"union.merge.{tree}.rounded"] for tree in TREES]
    trees = {}
    for tree, candidate, rounded in zip(TREES, merged, merged_bf16, strict=True):
        require(
            all(
                value == replay(tree, [part[p] for part in parts])
                for p, value in enumerate(candidate)
            ),
            f"{case}: GPU output differs from F32 replay of {tree}",
        )
        require(
            not changed(candidate, union, singleton),
            f"{case}: singleton merge control failed",
        )
        trees[tree] = {
            "f32_vs_upstream": measure(candidate, union),
            "bf16_vs_upstream": measure(rounded, union_bf16),
        }

    no_exact = {
        p for p, value in enumerate(union) if all(row[p] != value for row in merged)
    }
    return {
        "prior_failure_output_indices": [indices[p] for p in sorted(old_failed)],
        "full_native": variants,
        "paired_changes_on_prior_points": paired,
        "singleton_controls": {
            "points": len(singleton),
            "all_four_products_nonzero": active,
        },
        "local_f32_vs_upstream": local,
        "union": {
            "trees": trees,
            "no_tree_exact_f32_points": len(no_exact),
            "no_tree_exact_f32_on_prior_points": len(no_exact & old_failed),
            "max_best_tree_f32_error": max(
                min(abs(row[p] - value) for row in merged)
                for p, value in enumerate(union)
            ),
        },
    }


def compare(fixture, observed):
    manifest_pin, pins = fixture_pins(fixture)
    metadata = json.loads((observed / "k-partials.json").read_text())
    expected = {
        "schema": "dcae_k_partials_raw/v1",
        "cases": list(CASES),
        "samples": SAMPLES,
        "shape": list(SHAPE),
        "tile_width": TILE_WIDTH,
        "tile_count": TILE_COUNT,
        "sparse_tiles": list(SPARSE_TILES),
        "singleton_channel_stride": SINGLETON_CHANNEL_STRIDE,
        "prior_manifest_sha256": manifest_pin,
        "fixture_sha256": pins,
        "shader_sha256": hashlib.sha256(SHADER.read_bytes()).hexdigest(),
        "upstream_sha256": UPSTREAM_SHA,
        "upstream_functions": ["_conv3d_split_channel", "_custom_conv3d"],
        "torch_version": "2.8.0",
        "cpu_fallback": False,
        "terminal_tensor_transfers": 1,
        "explicit_mps_synchronizations": 0,
        "arithmetic_witness": [0.0, 2.0**-24, 0.0, 2.0**-24],
    }
    for key, value in expected.items():
        require(metadata.get(key) == value, f"Unexpected K-partials metadata: {key}")
    prefixes = [
        *(f"{arm}.upstream" for arm in ARMS),
        *(f"full.native.{policy}" for policy in POLICIES),
        *(
            f"local.{t}.native.{order}"
            for t in range(len(SPARSE_TILES))
            for order in ("f", "r")
        ),
        *(f"union.merge.{tree}" for tree in TREES),
    ]
    columns = metadata["columns"]
    names = {f"{prefix}.{fmt}" for prefix in prefixes for fmt in ("f32", "rounded")}
    names.update(f"{arm}.upstream.bf16" for arm in ARMS)
    require(
        len(columns) == len(names) and set(columns) == names,
        "Unexpected observation columns",
    )
    digest = metadata["observations_sha256"]
    stride = len(columns) * SAMPLES
    data = read_column(observed / "k-partials.f32", "f", digest, len(CASES) * stride)
    require(all(map(math.isfinite, data)), "Non-finite GPU observations")
    results = {}
    for case_i, case in enumerate(CASES):
        base = case_i * stride
        rows = {
            name: data[base + j * SAMPLES : base + (j + 1) * SAMPLES]
            for j, name in enumerate(columns)
        }
        results[case] = compare_case(fixture, pins, case, rows)
    passing = [
        tree
        for tree in TREES
        if all(
            r["union"]["trees"][tree]["bf16_vs_upstream"]["gate_failures"] == 0
            for r in results.values()
        )
    ]
    return {
        "schema": "dcae_k_partials_comparison/v1",
        "observation_sha256": digest,
        "atol": ATOL,
        "rtol": RTOL,
        "evidence_admission_passed": True,
        "sparse_union_bf16_passing_both_cases": passing,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sys.byteorder != "little":
        parser.error("Expected a new output path on a little-endian host")
    result = compare(args.fixture, args.observed)
    with args.output.open("x") as stream:
        stream.write(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
    print(f"Evidence admitted: {args.output}")


if __name__ == "__main__":
    main()
