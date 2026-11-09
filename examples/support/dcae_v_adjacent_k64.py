#!/usr/bin/env python3
"""Compare adjacent K64 Conv3D tiles with carry or one F32 addition."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
from ctypes import c_float, string_at
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from dcae_v_k_partials import (
    CASES,
    SAMPLES,
    SHADER,
    SHAPE,
    SINGLETON_CHANNEL_STRIDE,
    TILE_WIDTH,
    UPSTREAM_SHA,
    fixture_pins,
    host_fixture,
    pinned_conv3d,
    upstream_sample,
)
from dcae_v_k_partials_compare import changed, failures, measure, read_column, require
from dcae_v_reduction_compare import ATOL, RTOL, bf16, round_bf16

PAIR_STARTS = (106, 181)
POLICIES = ("merge", "carry_ab", "carry_ba")
WITNESS = 2.0**-24
WITNESS_ROWS = [
    1.0,
    WITNESS - 1.0,
    WITNESS - 1.0,
    1.0,
    WITNESS,
    WITNESS,
    0.0,
    WITNESS,
    WITNESS,
    0.0,
]


def stage_weights(torch, mask, weight):
    arms = {"full.upstream": (weight, weight.float())}
    for first in PAIR_STARTS:
        for name, offset, count in (("a", 0, 1), ("b", 1, 1), ("union", 0, 2)):
            masked = torch.empty_like(weight)
            mask(
                weight,
                masked,
                first + offset,
                count,
                threads=weight.numel(),
                arg_casts={2: "int32", 3: "int32"},
            )
            arms[f"{first}.upstream.{name}"] = (masked, masked.float())
    return arms


def observe_case(torch, conv, kernel, device, arms, case):
    cache = device[f"{case}-cache.bf16"]
    joined = torch.cat(
        (cache, device[f"{case}-input.bf16"], torch.zeros_like(cache)), dim=2
    )
    joined_f32 = joined.float()
    indices = device[f"{case}-indices.u32"]
    weight = device["weight.bf16"]
    require(
        all(x.is_contiguous() for x in (joined, weight, indices)),
        f"{case}: expected contiguous Metal inputs",
    )
    rows = {}
    for first in PAIR_STARTS:
        out = torch.empty((5, SAMPLES), dtype=torch.float32, device="mps")
        kernel(
            joined, weight, indices, out, first, threads=SAMPLES, arg_casts={4: "int32"}
        )
        for i, name in enumerate(("a", "b", *POLICIES)):
            rows[f"{first}.native.{name}.f32"] = out[i]
            if name in POLICIES:
                rows[f"{first}.native.{name}.rounded"] = (
                    out[i].to(torch.bfloat16).float()
                )
    for name, (weight_bf16, weight_f32) in arms.items():
        raw = upstream_sample(torch, conv, joined_f32, weight_f32, indices)
        direct = upstream_sample(torch, conv, joined, weight_bf16, indices)
        rows[f"{name}.f32"] = raw
        rows[f"{name}.rounded"] = raw.to(torch.bfloat16).float()
        rows[f"{name}.bf16"] = direct.float()
    require(
        all(
            x.dtype == torch.float32
            and x.device.type == "mps"
            and x.shape == (SAMPLES,)
            for x in rows.values()
        ),
        f"{case}: invalid observation column",
    )
    return rows


def summarize_pair(rows, first, singleton, prior_failed):
    def get(side, name, dtype):
        return rows[f"{first}.{side}.{name}.{dtype}"]

    a, b = (get("native", tile, "f32") for tile in ("a", "b"))
    for tile, local in (("a", a), ("b", b)):
        require(
            local == get("upstream", tile, "f32"),
            f"{first}: isolated tile {tile} differs",
        )
    ref_f32 = get("upstream", "union", "f32")
    ref_bf16 = get("upstream", "union", "bf16")
    candidates = {name: get("native", name, "f32") for name in POLICIES}
    require(
        all(
            merged == c_float(x + y).value
            for merged, x, y in zip(candidates["merge"], a, b, strict=True)
        ),
        f"{first}: merge differs from one F32 addition",
    )
    require(
        all(not changed(values, ref_f32, singleton) for values in candidates.values()),
        f"{first}: singleton control differs",
    )
    sensitivity = {
        f"{left}/{right}": changed(candidates[left], candidates[right])
        for left, right in (
            ("merge", "carry_ab"),
            ("merge", "carry_ba"),
            ("carry_ab", "carry_ba"),
        )
    }
    require(
        all(sensitivity.values()),
        f"{first}: carry and merge controls are not discriminating",
    )
    active = [
        i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != 0.0 and y != 0.0
    ]
    singleton_active = sum(a[i] != 0.0 and b[i] != 0.0 for i in singleton)
    require(
        singleton_active, f"{first}: no singleton control with two nonzero products"
    )
    patterns = [
        "+".join(name for name, values in candidates.items() if values[i] == ref_f32[i])
        or "none"
        for i in range(SAMPLES)
    ]
    return {
        "f32_vs_upstream": {
            name: measure(values, ref_f32) for name, values in candidates.items()
        },
        "bf16_vs_upstream": {
            name: measure(get("native", name, "rounded"), ref_bf16) for name in POLICIES
        },
        "upstream_rounded_f32_vs_actual_bf16": {
            name: measure(
                get("upstream", name, "rounded"), get("upstream", name, "bf16")
            )
            for name in ("a", "b", "union")
        },
        "f32_changes_between_policies": sensitivity,
        "f32_exact_match_patterns": dict(Counter(patterns)),
        "f32_exact_match_patterns_prior_failed": dict(
            Counter(patterns[i] for i in prior_failed)
        ),
        "f32_exact_match_patterns_both_nonzero": dict(
            Counter(patterns[i] for i in active)
        ),
        "singleton_controls": {
            "points": len(singleton),
            "both_nonzero": singleton_active,
        },
    }


def summarize_case(fixture, pins, case, rows):
    def retained(name, code):
        return read_column(fixture / name, code, pins[name], SAMPLES)

    f32_control = retained(f"{case}-mps-f32.f32", "f")
    bf16_control = array("f", map(bf16, retained(f"torch-{case}-pre.bf16", "H")))
    previous = list(map(bf16, retained(f"{case}-order0-lanes1-tree0.pre.bf16", "H")))
    require(
        rows["full.upstream.f32"].tobytes() == f32_control.tobytes()
        and rows["full.upstream.bf16"].tobytes() == bf16_control.tobytes(),
        f"{case}: fresh unmasked outputs differ from retained controls",
    )
    for name, values in rows.items():
        if name.endswith(".rounded"):
            raw = rows[f"{name.removesuffix('.rounded')}.f32"]
            require(
                all(bf16(round_bf16(x)) == y for x, y in zip(raw, values, strict=True)),
                f"{case}: unexpected BF16 rounding for {name}",
            )
    prior_failed = failures(previous, bf16_control)
    require(prior_failed, f"{case}: missing prior failing coordinates")
    indices = retained(f"{case}-indices.u32", "I")
    plane = math.prod(SHAPE[2:])
    singleton = [
        i
        for i, value in enumerate(indices)
        if (value // plane) % SINGLETON_CHANNEL_STRIDE == 0
    ]
    return {
        "prior_failed_points": len(prior_failed),
        "pairs": {
            f"{first}-{first + 1}": summarize_pair(rows, first, singleton, prior_failed)
            for first in PAIR_STARTS
        },
    }


def capture(fixture, output):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    require(
        torch.backends.mps.is_available() and torch.__version__ == "2.8.0",
        "Requires pinned Torch 2.8.0 MPS",
    )
    torch.set_grad_enabled(False)
    manifest_pin, pins = fixture_pins(fixture)
    host = host_fixture(torch, fixture, pins)
    values = [0.0] * (2 * TILE_WIDTH)
    values[0], values[TILE_WIDTH], values[-1] = 1.0, WITNESS, -1.0
    host["witness_values"] = torch.tensor(values, dtype=torch.bfloat16)
    host["witness_ones"] = torch.ones(TILE_WIDTH, dtype=torch.bfloat16)
    # Keep nonblocking copy sources alive through the sole terminal readout.
    device = {
        name: tensor.to("mps", non_blocking=True) for name, tensor in host.items()
    }

    conv, names = pinned_conv3d(torch)
    shader = (
        SHADER.read_text() + "\n" + Path(__file__).with_suffix(".metal").read_text()
    )
    library = torch.mps.compile_shader(shader)
    witness = torch.empty((5, 2), dtype=torch.float32, device="mps")
    library.adjacent_witness(
        device["witness_values"], device["witness_ones"], witness, threads=2
    )
    arms = stage_weights(torch, library.mask_adjacent_weights, device["weight.bf16"])
    cases = []
    for case in CASES:
        cases.append(
            observe_case(torch, conv, library.adjacent_samples, device, arms, case)
        )
        print(
            f"Enqueued {case}: two K64 pairs and the unmasked upstream controls",
            flush=True,
        )
    columns = tuple(cases[0])
    require(
        all(tuple(rows) == columns for rows in cases), "Observation columns drifted"
    )
    flat = torch.stack([rows[name] for rows in cases for name in columns]).flatten()
    drained = torch.cat((flat, witness.flatten())).to("cpu")
    host.clear()
    measured = drained[-witness.numel() :].tolist()
    payload = string_at(drained.data_ptr(), flat.numel() * 4)
    report = {
        "schema": "dcae_v_adjacent_k64/v1",
        "observations_sha256": hashlib.sha256(payload).hexdigest(),
        "shape": SHAPE,
        "samples": SAMPLES,
        "columns": columns,
        "case_order": CASES,
        "prior_manifest_sha256": manifest_pin,
        "fixture_sha256": pins,
        "upstream_sha256": UPSTREAM_SHA,
        "upstream_functions": names,
        "shader_sha256": hashlib.sha256(shader.encode()).hexdigest(),
        "torch_version": torch.__version__,
        "atol": ATOL,
        "rtol": RTOL,
        "cancellation_witness": measured,
        "terminal_tensor_transfers": 1,
        "explicit_mps_synchronizations": 0,
        "cpu_fallback": False,
    }
    with (output / "observations.f32").open("xb") as stream:
        stream.write(payload)
    with (output / "observations.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    require(
        measured == WITNESS_ROWS,
        f"Adjacent carry cancellation witness failed: {measured}",
    )
    observations = array("f", payload)
    require(all(map(math.isfinite, observations)), "Non-finite GPU observations")
    reports = {}
    for i, case in enumerate(CASES):
        start = i * len(columns) * SAMPLES
        rows = {
            name: observations[start + j * SAMPLES : start + (j + 1) * SAMPLES]
            for j, name in enumerate(columns)
        }
        reports[case] = summarize_case(fixture, pins, case, rows)
    report["cases"] = reports
    with (output / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        f"Wrote {len(payload)} bytes and comparison after the sole terminal readout",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--admitted", action="store_true")
    args = parser.parse_args()
    if not args.admitted:
        parser.error("Launch through the cache-aware memory guard")
    if (
        sys.byteorder != "little"
        or not args.output.is_dir()
        or any(
            (args.output / name).exists()
            for name in ("observations.f32", "observations.json", "comparison.json")
        )
    ):
        parser.error("Expected a fresh output directory on a little-endian host")
    tiles_per_tap = SHAPE[1] // TILE_WIDTH
    require(
        all(
            first // tiles_per_tap == (first + 1) // tiles_per_tap
            for first in PAIR_STARTS
        ),
        "Each pair must be adjacent within a single tap",
    )
    capture(args.fixture, args.output)


if __name__ == "__main__":
    main()
