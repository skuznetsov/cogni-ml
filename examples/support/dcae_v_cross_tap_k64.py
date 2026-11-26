#!/usr/bin/env python3
"""Test cross-tap K64 Conv3D accumulation and replay its sampled MPS output."""

from __future__ import annotations

import argparse
from array import array
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
    TILE_COUNT,
    TILE_WIDTH,
    UPSTREAM_SHA,
    fixture_pins,
    host_fixture,
    pinned_conv3d,
    upstream_sample,
)
from dcae_v_k_partials_compare import changed, failures, measure, read_column, require
from dcae_v_reduction_compare import ATOL, RTOL, bf16, round_bf16

PAIRS = ((34, 106), (106, 178))
GROUPS = (1, 2, 4, 8, 16, 32, 64)
DIRECTIONS = ("ab", "ba")
POLICIES = ("merge", *(f"g{g}_{d}" for g in GROUPS for d in DIRECTIONS))
NATIVE = ("a", "b", *POLICIES)
WITNESS_TINY = 2.0**-24


def pair_key(pair):
    return f"{pair[0]}-{pair[1]}"


ARMS = {
    "full": None,
    **{str(tile): (tile, tile) for tile in sorted({t for p in PAIRS for t in p})},
    **{pair_key(pair): pair for pair in PAIRS},
}
COLUMNS = (
    *(
        f"native.{pair_key(pair)}.{name}.{dtype}"
        for pair in PAIRS
        for name in NATIVE
        for dtype in (("f32", "rounded") if name in POLICIES else ("f32",))
    ),
    *(
        f"upstream.{name}.{dtype}"
        for name in ARMS
        for dtype in ("f32", "rounded", "bf16")
    ),
)


def witness_inputs_and_rows():
    inputs = []
    rows = {name: [] for name in NATIVE}
    # Lane-zero witnesses identify the group width; lane one splits the two
    # one-channel schedules, whose lane-zero signatures are both all zero.
    for pivot, tiny_lane in (*((g, 0) for g in GROUPS[:-1]), (1, 1)):
        for cancel in DIRECTIONS:
            large = [0.0] * TILE_WIDTH
            large[0], large[pivot] = 1.0, -1.0
            tiny = [0.0] * TILE_WIDTH
            tiny[tiny_lane] = WITNESS_TINY
            inputs.extend(large + tiny if cancel == "ab" else tiny + large)
            rows["a"].append(0.0 if cancel == "ab" else WITNESS_TINY)
            rows["b"].append(WITNESS_TINY if cancel == "ab" else 0.0)
            rows["merge"].append(WITNESS_TINY)
            for g in GROUPS:
                for d in DIRECTIONS:
                    rows[f"g{g}_{d}"].append(
                        WITNESS_TINY
                        if d == cancel and (tiny_lane or g > pivot)
                        else 0.0
                    )
    return inputs, rows


def combined_shader():
    return "\n".join(
        path.read_text()
        for path in (
            SHADER,
            SHADER.with_name("dcae_v_adjacent_k64.metal"),
            Path(__file__).with_suffix(".metal"),
        )
    )


def metadata(manifest_pin, pins):
    return {
        "schema": "dcae_cross_tap_k64/v1",
        "cases": list(CASES),
        "columns": list(COLUMNS),
        "native_witness_rows": list(NATIVE),
        "shape": list(SHAPE),
        "samples": SAMPLES,
        "pairs": [list(pair) for pair in PAIRS],
        "groups": list(GROUPS),
        "tile_width": TILE_WIDTH,
        "fixture_manifest_sha256": manifest_pin,
        "fixture_sha256": pins,
        "shader_sha256": hashlib.sha256(combined_shader().encode()).hexdigest(),
        "upstream_sha256": UPSTREAM_SHA,
        "torch_version": "2.8.0",
        "atol": ATOL,
        "rtol": RTOL,
        "terminal_tensor_transfers": 1,
        "explicit_mps_synchronizations": 0,
        "cpu_fallback": False,
    }


def capture(fixture, output):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    require(
        torch.backends.mps.is_available() and torch.__version__ == "2.8.0",
        "Requires the pinned Torch 2.8.0 MPS runtime",
    )
    torch.set_grad_enabled(False)
    manifest_pin, pins = fixture_pins(fixture)
    host = host_fixture(torch, fixture, pins)
    values, expected = witness_inputs_and_rows()
    count = len(expected["a"])
    host["witness"] = torch.tensor(values, dtype=torch.bfloat16)
    host["ones"] = torch.ones(TILE_WIDTH, dtype=torch.bfloat16)
    # Retain the nonblocking upload sources through the sole terminal transfer.
    device = {
        name: tensor.to("mps", non_blocking=True) for name, tensor in host.items()
    }
    conv, functions = pinned_conv3d(torch)
    shader = combined_shader()
    library = torch.mps.compile_shader(shader)
    witness = torch.empty((len(NATIVE), count), dtype=torch.float32, device="mps")
    library.cross_tap_witness(
        device["witness"],
        device["ones"],
        witness,
        count,
        threads=count,
        arg_casts={3: "int32"},
    )
    weight = device["weight.bf16"]
    arms = {}
    for name, pair in ARMS.items():
        masked = weight if pair is None else torch.empty_like(weight)
        if pair is not None:
            library.mask_aligned_taps(
                weight,
                masked,
                *pair,
                threads=weight.numel(),
                arg_casts={2: "int32", 3: "int32"},
            )
        arms[name] = (masked, masked.float())

    stacked = []
    for case in CASES:
        cache = device[f"{case}-cache.bf16"]
        joined = torch.cat(
            (cache, device[f"{case}-input.bf16"], torch.zeros_like(cache)), dim=2
        )
        joined_f32 = joined.float()
        indices = device[f"{case}-indices.u32"]
        require(
            all(x.is_contiguous() for x in (joined, weight, indices)),
            "Metal input layout changed",
        )
        rows = {}
        for pair in PAIRS:
            out = torch.empty((len(NATIVE), SAMPLES), dtype=torch.float32, device="mps")
            library.cross_tap_samples(
                joined,
                weight,
                indices,
                out,
                *pair,
                threads=SAMPLES,
                arg_casts={4: "int32", 5: "int32"},
            )
            for row, name in zip(out.unbind(0), NATIVE, strict=True):
                key = f"native.{pair_key(pair)}.{name}"
                rows[f"{key}.f32"] = row
                if name in POLICIES:
                    rows[f"{key}.rounded"] = row.to(torch.bfloat16).float()
        for name, (w_bf16, w_f32) in arms.items():
            raw = upstream_sample(torch, conv, joined_f32, w_f32, indices)
            direct = upstream_sample(torch, conv, joined, w_bf16, indices)
            rows[f"upstream.{name}.f32"] = raw
            rows[f"upstream.{name}.rounded"] = raw.to(torch.bfloat16).float()
            rows[f"upstream.{name}.bf16"] = direct.float()
        require(
            set(rows) == set(COLUMNS)
            and all(
                x.dtype == torch.float32
                and x.device.type == "mps"
                and x.shape == (SAMPLES,)
                for x in rows.values()
            ),
            f"{case}: observation schema mismatch",
        )
        stacked.append(torch.stack([rows[name] for name in COLUMNS]).flatten())
        print(f"Enqueued {case}: two tap pairs and full upstream controls", flush=True)

    host_readout = torch.cat((*stacked, witness.flatten())).to("cpu")
    payload = string_at(
        host_readout.data_ptr(), host_readout.numel() * host_readout.element_size()
    )
    header = metadata(manifest_pin, pins)
    require(
        header["shader_sha256"] == hashlib.sha256(shader.encode()).hexdigest(),
        "Shader source changed during capture",
    )
    header.update(
        payload_sha256=hashlib.sha256(payload).hexdigest(), upstream_functions=functions
    )
    with (output / "observations.f32").open("xb") as stream:
        stream.write(payload)
    with (output / "observations.json").open("x") as stream:
        json.dump(header, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {len(payload)} bytes after the sole terminal readout", flush=True)


def bit_view(values):
    return memoryview(values).cast("B").cast("I")


def summarize_pair(rows, case, pair, singleton, first, last, prior):
    key = pair_key(pair)
    a, b = (rows[f"native.{key}.{tile}.f32"] for tile in ("a", "b"))
    for tile, native in zip(pair, (a, b), strict=True):
        require(
            native.tobytes() == rows[f"upstream.{tile}.f32"].tobytes(),
            f"{case}: isolated tile {tile} differs",
        )
    ref_f32 = rows[f"upstream.{key}.f32"]
    ref_bf16 = rows[f"upstream.{key}.bf16"]
    candidates = {name: rows[f"native.{key}.{name}.f32"] for name in POLICIES}
    require(
        all(
            v == c_float(x + y).value
            for v, x, y in zip(candidates["merge"], a, b, strict=True)
        ),
        f"{case}/{key}: merge is not one F32 addition",
    )
    require(
        len({v.tobytes() for v in candidates.values()}) == len(candidates),
        f"{case}/{key}: aliased policies",
    )
    active = {
        i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != 0.0 and y != 0.0
    }
    require(
        active.intersection(singleton)
        and all(not changed(v, ref_f32, singleton) for v in candidates.values()),
        f"{case}/{key}: singleton control failed",
    )

    def check_zero(zero, retained, positions):
        require(
            positions
            and any(retained[i] != 0.0 for i in positions)
            and all(zero[i] == 0.0 for i in positions)
            and all(
                not changed(v, retained, positions)
                for v in (ref_f32, *candidates.values())
            ),
            f"{case}/{key}: cache/padding zero control failed",
        )

    cached_active = (
        active.intersection(first) if pair == PAIRS[0] and case == "valid" else set()
    )
    cached_changes = changed(candidates["g1_ab"], candidates["g64_ab"], cached_active)
    if pair == PAIRS[0]:
        if case == "invalid":
            check_zero(a, b, first)
        else:
            require(
                cached_changes and cached_active.intersection(singleton),
                f"{case}/{key}: cache witness does not distinguish tap orders",
            )
    else:
        check_zero(b, a, last)

    scopes = {
        "first_frame": first,
        "last_frame": last,
        "both_nonzero": active,
        "active_cache": cached_active,
        "prior_failed": prior,
    }
    reference_bits = bit_view(ref_f32)
    exact = {
        name: {
            i
            for i, (x, y) in enumerate(zip(bit_view(v), reference_bits, strict=True))
            if x == y
        }
        for name, v in candidates.items()
    }
    return {
        "f32_vs_upstream": {
            name: {
                **measure(v, ref_f32),
                "bitwise_exact": len(exact[name]),
                "bitwise_exact_by_scope": {
                    s: len(exact[name].intersection(points))
                    for s, points in scopes.items()
                },
            }
            for name, v in candidates.items()
        },
        "bf16_vs_upstream": {
            name: measure(rows[f"native.{key}.{name}.rounded"], ref_bf16)
            for name in POLICIES
        },
        "scope_points": {s: len(points) for s, points in scopes.items()},
        "singleton_points": len(singleton),
        "singleton_both_nonzero": len(active.intersection(singleton)),
        "cached_active_singleton_points": len(cached_active.intersection(singleton)),
        "g1_vs_g64_ab_f32_changes": changed(candidates["g1_ab"], candidates["g64_ab"]),
        "g1_vs_g64_ab_f32_changes_active_cache": cached_changes,
        "no_policy_exact_f32_points": SAMPLES - len(set.union(*exact.values())),
    }


def summarize_case(fixture, pins, case, rows):
    def retained(name, code):
        return read_column(fixture / name, code, pins[name], SAMPLES)

    f32 = retained(f"{case}-mps-f32.f32", "f")
    direct = array("f", map(bf16, retained(f"torch-{case}-pre.bf16", "H")))
    prior_values = list(
        map(bf16, retained(f"{case}-order0-lanes1-tree0.pre.bf16", "H"))
    )
    prior = failures(prior_values, direct)
    require(
        prior and all(map(math.isfinite, prior_values)),
        f"{case}: missing valid prior failure set",
    )
    require(
        rows["upstream.full.f32"].tobytes() == f32.tobytes()
        and rows["upstream.full.bf16"].tobytes() == direct.tobytes(),
        f"{case}: fresh full upstream outputs differ from pinned baseline",
    )
    for name, values in rows.items():
        if name.endswith(".rounded"):
            raw = rows[f"{name.removesuffix('.rounded')}.f32"]
            require(
                all(
                    round_bf16(x) << 16 == bits
                    for x, bits in zip(raw, bit_view(values), strict=True)
                ),
                f"{case}: unexpected BF16 rounding for {name}",
            )
    indices = retained(f"{case}-indices.u32", "I")
    require(
        len(set(indices)) == SAMPLES and max(indices) < math.prod(SHAPE),
        f"{case}: invalid indices",
    )
    plane = math.prod(SHAPE[3:])
    frames = [value // plane % SHAPE[2] for value in indices]
    first = {i for i, t in enumerate(frames) if t == 0}
    last = {i for i, t in enumerate(frames) if t == SHAPE[2] - 1}
    singleton = {
        i
        for i, value in enumerate(indices)
        if value // (plane * SHAPE[2]) % SINGLETON_CHANNEL_STRIDE == 0
    }
    return {
        "upstream_rounded_f32_vs_actual_bf16": {
            name: measure(
                rows[f"upstream.{name}.rounded"], rows[f"upstream.{name}.bf16"]
            )
            for name in ARMS
        },
        "pairs": {
            pair_key(p): summarize_pair(rows, case, p, singleton, first, last, prior)
            for p in PAIRS
        },
    }


def replay(fixture, output):
    manifest_pin, pins = fixture_pins(fixture)
    header = json.loads((output / "observations.json").read_text())
    require(
        all(header.get(k) == v for k, v in metadata(manifest_pin, pins).items()),
        "Unexpected observation metadata",
    )
    _, expected = witness_inputs_and_rows()
    count = len(expected["a"])
    offset = len(CASES) * len(COLUMNS) * SAMPLES
    values = read_column(
        output / "observations.f32",
        "f",
        header["payload_sha256"],
        offset + len(NATIVE) * count,
    )
    require(all(map(math.isfinite, values)), "Non-finite GPU observations")
    witness = {
        name: list(values[offset + i * count : offset + (i + 1) * count])
        for i, name in enumerate(NATIVE)
    }
    require(
        witness == expected
        and len({tuple(witness[p]) for p in POLICIES}) == len(POLICIES),
        "Cancellation witness does not identify each policy",
    )
    cases = {}
    for i, case in enumerate(CASES):
        base = i * len(COLUMNS)
        rows = {
            name: values[(base + j) * SAMPLES : (base + j + 1) * SAMPLES]
            for j, name in enumerate(COLUMNS)
        }
        cases[case] = summarize_case(fixture, pins, case, rows)
    pairs = [pair for case in cases.values() for pair in case["pairs"].values()]
    return {
        "observations": header,
        "witness": witness,
        "cases": cases,
        "fixed_exact_f32_policies": [
            name
            for name in POLICIES
            if all(
                pair["f32_vs_upstream"][name]["bitwise_exact"] == SAMPLES
                for pair in pairs
            )
        ],
        "fixed_bf16_gate_policies": [
            name
            for name in POLICIES
            if all(
                not pair["bf16_vs_upstream"][name]["gate_failures"] for pair in pairs
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--admitted",
        action="store_true",
        help="Launch through the cache-aware memory guard",
    )
    mode.add_argument(
        "--replay",
        action="store_true",
        help="Validate and compare saved observations without Metal",
    )
    args = parser.parse_args()
    if sys.byteorder != "little" or not args.output.is_dir():
        parser.error("Expected a little-endian host and an existing scratch directory")
    tiles_per_tap = SHAPE[1] // TILE_WIDTH
    for a, b in PAIRS:
        require(
            0 <= a < b < TILE_COUNT
            and a % tiles_per_tap == b % tiles_per_tap
            and a // tiles_per_tap % 9 == b // tiles_per_tap % 9 == 4,
            "Expected two center-spatial taps with aligned input-channel ranges",
        )
    if args.replay:
        print(
            json.dumps(
                replay(args.fixture, args.output),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if any(
        (args.output / name).exists()
        for name in ("observations.f32", "observations.json")
    ):
        parser.error("Capture needs a fresh output directory")
    capture(args.fixture, args.output)


if __name__ == "__main__":
    main()
