#!/usr/bin/env python3
"""Test a fixed Conv3D G8 order on aligned and crossed stage-2 masks."""

from __future__ import annotations

import argparse
from array import array
from ctypes import c_float, string_at
import hashlib
from itertools import accumulate, combinations
import json
import math
import os
from pathlib import Path
import sys

from dcae_v_cross_tap_k64 import bit_view
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
from dcae_v_k_partials_compare import failures, measure, read_column, require
from dcae_v_reduction_compare import ATOL, RTOL, bf16, round_bf16

MASKS = {"aligned": (34, 106, 178), "crossed": (99, 106, 107)}
POLICIES = ("g4", "g8", "g16", "g8_reverse", "g64", "tap_first")
LOCALS = ("local0", "local1", "local2")
ROWS = (*LOCALS, *POLICIES)
GROUPS = {"g4": 4, "g8": 8, "g16": 16, "g8_reverse": 8, "g64": 64}
COLUMNS = (
    *(
        f"native.{mask}.{name}.{dtype}"
        for mask in MASKS
        for name in ROWS
        for dtype in (("f32", "rounded") if name in POLICIES else ("f32",))
    ),
    "upstream.99.f32",
    "upstream.full.f32",
    "upstream.full.bf16",
    *(
        f"upstream.{mask}.{dtype}"
        for mask in MASKS
        for dtype in ("f32", "rounded", "bf16")
    ),
)
BASE = Path("/opt/mldata_ssd/cogni-ml")
# Header pins bind the columns, fixture, backend and payload digests.
PRIOR = {
    "pair": (
        BASE / "dcae-cross-tap-k64-20261108.8x8migwz",
        "1b357eb7dbf1eb1e1aa8d59d0afb6f57de9916bab52123d85dd116a83768c921",
        "cases",
        "fixture_manifest_sha256",
        "payload_sha256",
        17 * 14,
    ),
    "adjacent": (
        BASE / "dcae-adjacent-k64-20261108._hlrsdg4",
        "67d7b8812c9a84f6b5d49c94b9fca54a4deb3c210b20d06440f0186210b963ca",
        "case_order",
        "prior_manifest_sha256",
        "observations_sha256",
        0,
    ),
}
TINY = 2.0**-24
WITNESSES = {
    "aligned": (
        ((0, 1.0), (4, -1.0), (64, TINY)),
        ((0, 1.0), (8, -1.0), (64, TINY)),
        ((0, 1.0), (16, -1.0), (64, TINY)),
        ((0, 1.0), (128, -1.0), (64, TINY)),
    ),
    "crossed": (
        ((64, 1.0), (0, -1.0), (65, TINY)),
        ((0, 1.0), (4, -1.0), (128, TINY)),
        ((0, 1.0), (8, -1.0), (128, TINY)),
        ((0, 1.0), (16, -1.0), (128, TINY)),
    ),
}


def schedules(tiles):
    terms = tuple(range(len(tiles) * TILE_WIDTH))
    per_tap = SHAPE[1] // TILE_WIDTH

    def position(term, policy):
        slot, lane = divmod(term, TILE_WIDTH)
        tap, block = divmod(tiles[slot], per_tap)
        channel = block * TILE_WIDTH + lane
        return (
            channel // GROUPS[policy],
            (-tap if policy == "g8_reverse" else tap),
            channel,
        )

    return {
        **{
            name: terms[slot * TILE_WIDTH : (slot + 1) * TILE_WIDTH]
            for slot, name in enumerate(LOCALS)
        },
        **{
            policy: tuple(sorted(terms, key=lambda term: position(term, policy)))
            for policy in GROUPS
        },
        "tap_first": terms,
    }


def witness_inputs(mask, schedule):
    width = len(MASKS[mask]) * TILE_WIDTH
    inputs = []
    expected = {name: [] for name in ROWS}
    for terms in WITNESSES[mask]:
        values = [0.0] * width
        for term, value in terms:
            values[term] = value
        inputs.extend(values)
        # All witness products are exact with a BF16 weight of one.
        for name in ROWS:
            acc = 0.0
            for term in schedule[name]:
                acc = c_float(acc + values[term]).value
            expected[name].append(acc)
    return inputs, expected


def combined_shader():
    return SHADER.read_text() + "\n" + Path(__file__).with_suffix(".metal").read_text()


def metadata(manifest_pin, pins):
    return {
        "schema": "dcae_crossed_g8/v1",
        "cases": list(CASES),
        "columns": list(COLUMNS),
        "shape": list(SHAPE),
        "samples": SAMPLES,
        "masks": {name: list(tiles) for name, tiles in MASKS.items()},
        "rows": list(ROWS),
        "orders": {
            mask: {name: list(terms) for name, terms in schedules(tiles).items()}
            for mask, tiles in MASKS.items()
        },
        "fixture_manifest_sha256": manifest_pin,
        "fixture_sha256": pins,
        "prior_headers_sha256": {name: pin[1] for name, pin in PRIOR.items()},
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
    for mask, tiles in MASKS.items():
        schedule = schedules(tiles)
        # mask_weights expects four slots. Duplicating the last is idempotent.
        host[f"{mask}.tiles"] = torch.tensor((*tiles, tiles[-1]), dtype=torch.int32)
        host[f"{mask}.terms"] = torch.tensor(
            [t for name in ROWS for t in schedule[name]], dtype=torch.int32
        )
        host[f"{mask}.offsets"] = torch.tensor(
            [0, *accumulate(len(schedule[name]) for name in ROWS)], dtype=torch.int32
        )
        values, _ = witness_inputs(mask, schedule)
        host[f"{mask}.witness"] = torch.tensor(values, dtype=torch.bfloat16)
    host["ones"] = torch.ones(TILE_WIDTH, dtype=torch.bfloat16)
    # Nonblocking uploads require their host owners until the terminal readout.
    device = {name: t.to("mps", non_blocking=True) for name, t in host.items()}
    conv, functions = pinned_conv3d(torch)
    shader = combined_shader()
    library = torch.mps.compile_shader(shader)
    weight = device["weight.bf16"]
    arms = {"full": (weight, weight.float())}
    witnesses = []
    for mask in MASKS:
        count = len(WITNESSES[mask])
        witness = torch.empty((len(ROWS), count), dtype=torch.float32, device="mps")
        library.scheduled_witness(
            device[f"{mask}.witness"],
            device["ones"],
            device[f"{mask}.terms"],
            device[f"{mask}.offsets"],
            witness,
            count,
            threads=witness.numel(),
            arg_casts={5: "int32"},
        )
        witnesses.append(witness.flatten())
        masked = torch.empty_like(weight)
        library.mask_weights(
            weight,
            device[f"{mask}.tiles"],
            masked,
            4,
            threads=weight.numel(),
            arg_casts={3: "int32"},
        )
        arms[mask] = (masked, masked.float())

    isolated = torch.empty_like(weight)
    library.mask_weights(
        weight,
        device["crossed.tiles"],
        isolated,
        0,
        threads=weight.numel(),
        arg_casts={3: "int32"},
    )
    isolated_f32 = isolated.float()
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
        for mask in MASKS:
            out = torch.empty((len(ROWS), SAMPLES), dtype=torch.float32, device="mps")
            library.scheduled_samples(
                joined,
                weight,
                indices,
                device[f"{mask}.tiles"],
                device[f"{mask}.terms"],
                device[f"{mask}.offsets"],
                out,
                threads=out.numel(),
            )
            for name, row in zip(ROWS, out.unbind(0), strict=True):
                rows[f"native.{mask}.{name}.f32"] = row
                if name in POLICIES:
                    rows[f"native.{mask}.{name}.rounded"] = row.to(
                        torch.bfloat16
                    ).float()
        rows["upstream.99.f32"] = upstream_sample(
            torch, conv, joined_f32, isolated_f32, indices
        )
        for name, (w_bf16, w_f32) in arms.items():
            raw = upstream_sample(torch, conv, joined_f32, w_f32, indices)
            rows[f"upstream.{name}.f32"] = raw
            rows[f"upstream.{name}.bf16"] = upstream_sample(
                torch, conv, joined, w_bf16, indices
            ).float()
            if name != "full":
                rows[f"upstream.{name}.rounded"] = raw.to(torch.bfloat16).float()
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
        print(
            f"Enqueued {case}: two three-tile masks and upstream controls", flush=True
        )

    host_readout = torch.cat((*stacked, *witnesses)).to("cpu")
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


def read_prior(manifest_pin, pins):
    sources = {}
    for name, (
        root,
        digest,
        case_key,
        manifest_key,
        payload_key,
        tail,
    ) in PRIOR.items():
        raw = (root / "observations.json").read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == digest, f"{name}: prior header changed"
        )
        header = json.loads(raw)
        require(
            header[case_key] == list(CASES)
            and header[manifest_key] == manifest_pin
            and header["fixture_sha256"] == pins
            and header["samples"] == SAMPLES
            and header["shape"] == list(SHAPE)
            and header["upstream_sha256"] == UPSTREAM_SHA
            and header["torch_version"] == "2.8.0",
            f"{name}: prior capture used a different fixture or backend",
        )
        columns = header["columns"]
        values = read_column(
            root / "observations.f32",
            "f",
            header[payload_key],
            len(CASES) * len(columns) * SAMPLES + tail,
        )
        sources[name] = (columns, values)

    def prior_column(source, case, column):
        columns, values = sources[source]
        start = (CASES.index(case) * len(columns) + columns.index(column)) * SAMPLES
        return values[start : start + SAMPLES]

    return prior_column


def check_boundaries(case, rows, locals_by_mask, prior_column, first, last, left):
    edges = [
        ("aligned", "last_frame", last, 2, "pair", "upstream.34-106"),
        ("crossed", "left_pad", left, 0, "adjacent", "106.upstream.union"),
    ]
    if case == "invalid":
        edges.append(
            ("aligned", "cold_first_frame", first, 0, "pair", "upstream.106-178")
        )
    results = {}
    for mask, name, points, slot, source, column in edges:
        local = locals_by_mask[mask]
        require(
            points
            and all(local[slot][i] == 0.0 for i in points)
            and any(
                local[other][i] != 0.0
                for other in range(len(LOCALS))
                if other != slot
                for i in points
            ),
            f"{case}/{name}: zero-tile control is inactive or invalid",
        )
        for dtype in ("f32", "bf16"):
            got = bit_view(rows[f"upstream.{mask}.{dtype}"])
            want = bit_view(prior_column(source, case, f"{column}.{dtype}"))
            require(
                all(got[i] == want[i] for i in points),
                f"{case}/{name}: projected upstream {dtype} differs from retained pair",
            )
        results[name] = len(points)
    return results


def summarize_case(fixture, pins, case, rows, prior_column):
    def retained(name, code):
        return read_column(fixture / name, code, pins[name], SAMPLES)

    indices = retained(f"{case}-indices.u32", "I")
    full = retained(f"{case}-mps-f32.f32", "f")
    full_bf16 = array("f", map(bf16, retained(f"torch-{case}-pre.bf16", "H")))
    previous = list(map(bf16, retained(f"{case}-order0-lanes1-tree0.pre.bf16", "H")))
    prior_failed = failures(previous, full_bf16)
    require(
        prior_failed
        and all(map(math.isfinite, previous))
        and len(set(indices)) == SAMPLES
        and max(indices) < math.prod(SHAPE),
        f"{case}: invalid sampled coordinates or missing prior failures",
    )
    require(
        rows["upstream.full.f32"].tobytes() == full.tobytes()
        and rows["upstream.full.bf16"].tobytes() == full_bf16.tobytes(),
        f"{case}: fresh unmasked upstream differs from pinned baseline",
    )
    for name, rounded in rows.items():
        if name.endswith(".rounded"):
            raw = rows[f"{name.removesuffix('.rounded')}.f32"]
            require(
                all(
                    round_bf16(x) << 16 == bits
                    for x, bits in zip(raw, bit_view(rounded), strict=True)
                ),
                f"{case}: unexpected BF16 rounding for {name}",
            )
    plane = math.prod(SHAPE[3:])
    frames = [idx // plane % SHAPE[2] for idx in indices]
    first = {i for i, t in enumerate(frames) if t == 0}
    last = {i for i, t in enumerate(frames) if t == SHAPE[2] - 1}
    left = {i for i, idx in enumerate(indices) if idx % SHAPE[-1] == 0}
    singleton = {
        i
        for i, idx in enumerate(indices)
        if idx // (plane * SHAPE[2]) % SINGLETON_CHANNEL_STRIDE == 0
    }

    locals_by_mask = {}
    masks = {}
    for mask, tiles in MASKS.items():
        local = [rows[f"native.{mask}.{name}.f32"] for name in LOCALS]
        locals_by_mask[mask] = local
        isolated = {
            "99": rows["upstream.99.f32"],
            "107": prior_column("adjacent", case, "106.upstream.b.f32"),
        }
        for tile, value in zip(tiles, local, strict=True):
            key = str(tile)
            expected = isolated.get(key)
            if expected is None:
                expected = prior_column("pair", case, f"upstream.{key}.f32")
            require(
                value.tobytes() == expected.tobytes(),
                f"{case}/{mask}: isolated tile {tile} differs",
            )
        active = {i for i, terms in enumerate(zip(*local, strict=True)) if all(terms)}
        candidate = {
            name: bit_view(rows[f"native.{mask}.{name}.f32"]) for name in POLICIES
        }
        schedule = schedules(tiles)
        for a, b in combinations(POLICIES, 2):
            if schedule[a] == schedule[b]:
                require(
                    candidate[a] == candidate[b],
                    f"{case}/{mask}: same schedule differs",
                )
            else:
                require(candidate[a] != candidate[b], f"{case}/{mask}: aliased {a}/{b}")
        rival = "g8_reverse" if mask == "aligned" else "tap_first"
        differs = {i for i in active if candidate["g8"][i] != candidate[rival][i]}
        require(differs, f"{case}/{mask}: no active three-tile discriminator")
        active_cache = (
            active & first if mask == "aligned" and case == "valid" else set()
        )
        if mask == "aligned" and case == "valid":
            require(
                active_cache & singleton and active_cache & differs,
                "Aligned mask did not exercise three contributions on the cached frame",
            )

        scopes = {
            "all_three_nonzero": active,
            "first_frame": first,
            "last_frame": last,
            "left_pad": left,
            "active_cache": active_cache,
            "prior_failed": prior_failed,
        }
        ref_f32 = bit_view(rows[f"upstream.{mask}.f32"])
        ref_bf16 = rows[f"upstream.{mask}.bf16"]
        compared = {}
        for name, bits in candidate.items():
            bad = {
                i for i, (a, b) in enumerate(zip(bits, ref_f32, strict=True)) if a != b
            }
            compared[name] = {
                "f32_bit_mismatches": len(bad),
                "f32_bit_mismatches_by_scope": {
                    s: len(bad & p) for s, p in scopes.items()
                },
                "bf16_vs_upstream": measure(
                    rows[f"native.{mask}.{name}.rounded"], ref_bf16
                ),
            }
        masks[mask] = {
            "policies": compared,
            "scope_points": {s: len(p) for s, p in scopes.items()},
            "active_cache_singletons": len(active_cache & singleton),
            "rival": rival,
            "rival_differs_all_three_nonzero": len(differs),
            "rival_differs_active_cache": len(differs & active_cache),
            "upstream_rounded_f32_vs_actual_bf16": measure(
                rows[f"upstream.{mask}.rounded"], ref_bf16
            ),
        }
    return {
        "masks": masks,
        "projected_pair_controls": check_boundaries(
            case, rows, locals_by_mask, prior_column, first, last, left
        ),
    }


def replay(fixture, output):
    manifest_pin, pins = fixture_pins(fixture)
    header = json.loads((output / "observations.json").read_text())
    require(
        all(header.get(k) == v for k, v in metadata(manifest_pin, pins).items())
        and header.get("upstream_functions")
        == ["_conv3d_split_channel", "_custom_conv3d"],
        "Unexpected observation metadata",
    )
    prior_column = read_prior(manifest_pin, pins)
    offset = len(CASES) * len(COLUMNS) * SAMPLES
    values = read_column(
        output / "observations.f32",
        "f",
        header["payload_sha256"],
        offset + len(ROWS) * sum(map(len, WITNESSES.values())),
    )
    require(all(map(math.isfinite, values)), "Non-finite GPU observations")
    witness = {}
    for mask, tiles in MASKS.items():
        _, expected = witness_inputs(mask, schedules(tiles))
        count = len(WITNESSES[mask])
        witness[mask] = {}
        for name in ROWS:
            row = values[offset : offset + count]
            require(
                row.tobytes() == array("f", expected[name]).tobytes(),
                f"{mask}/{name}: cancellation witness failed",
            )
            witness[mask][name] = list(row)
            offset += count
    require(
        len({tuple(v for rows in witness.values() for v in rows[p]) for p in POLICIES})
        == len(POLICIES),
        "Cancellation witnesses do not distinguish the policies across masks",
    )

    cases = {}
    for i, case in enumerate(CASES):
        base = i * len(COLUMNS)
        rows = {
            name: values[(base + j) * SAMPLES : (base + j + 1) * SAMPLES]
            for j, name in enumerate(COLUMNS)
        }
        cases[case] = summarize_case(fixture, pins, case, rows, prior_column)
    masks = [m for case in cases.values() for m in case["masks"].values()]
    return {
        "observations": header,
        "witness": witness,
        "cases": cases,
        "fixed_exact_f32_policies": [
            p
            for p in POLICIES
            if all(m["policies"][p]["f32_bit_mismatches"] == 0 for m in masks)
        ],
        "fixed_bf16_gate_policies": [
            p
            for p in POLICIES
            if all(
                m["policies"][p]["bf16_vs_upstream"]["gate_failures"] == 0
                for m in masks
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--admitted", action="store_true", help="Launch through the memory guard"
    )
    mode.add_argument(
        "--replay", action="store_true", help="Compare drained observations"
    )
    args = parser.parse_args()
    if sys.byteorder != "little" or not args.output.is_dir():
        parser.error("Expected a little-endian host and an existing scratch directory")
    if args.replay:
        print(
            json.dumps(
                replay(args.fixture, args.output),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    elif any((args.output / f"observations.{ext}").exists() for ext in ("json", "f32")):
        parser.error("Expected a scratch directory without previous observations")
    else:
        capture(args.fixture, args.output)


if __name__ == "__main__":
    main()
