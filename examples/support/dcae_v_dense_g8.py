#!/usr/bin/env python3
"""Compare sampled dense G8 Conv3D against the pinned F32 and BF16 operator."""

from __future__ import annotations

import argparse
from array import array
from ctypes import string_at
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from dcae_v_crossed_g8 import MASKS, bit_view
from dcae_v_k_partials import (
    CASES,
    SAMPLES,
    SHADER,
    SHAPE,
    UPSTREAM_SHA,
    WEIGHT,
    fixture_pins,
    host_fixture,
    pinned_conv3d,
    upstream_sample,
)
from dcae_v_k_partials_compare import SCOPES, failures, measure, read_column, require
from dcae_v_reduction_compare import ATOL, RTOL, bf16, round_bf16

MASK_REFERENCE = Path("/opt/mldata_ssd/cogni-ml/dcae-crossed-g8-20261108.Cz0kzvYV")
MASK_HEADER_SHA = "fcc3f54ccd42938018fae72b92730f519bac44b0629e9417060dec732a3d3012"
COLUMNS = (
    "native.f32",
    "native.rounded",
    "upstream.f32",
    "upstream.rounded",
    "upstream.bf16",
    *(f"masked.{mask}.f32" for mask in MASKS),
)


def shader_source():
    return SHADER.read_text() + "\n" + Path(__file__).with_suffix(".metal").read_text()


def metadata(manifest, pins, shader):
    return {
        "schema": "dcae_dense_g8/v1",
        "cases": list(CASES),
        "columns": list(COLUMNS),
        "shape": list(SHAPE),
        "weight_shape": list(WEIGHT),
        "samples": SAMPLES,
        "k": math.prod(WEIGHT[1:]),
        "order": "channel-group-8/tap-ascending/lane-ascending",
        "fixture_manifest_sha256": manifest,
        "fixture_sha256": pins,
        "mask_reference_header_sha256": MASK_HEADER_SHA,
        "shader_sha256": hashlib.sha256(shader.encode()).hexdigest(),
        "upstream_sha256": UPSTREAM_SHA,
        "upstream_functions": ["_conv3d_split_channel", "_custom_conv3d"],
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
    manifest, pins = fixture_pins(fixture)
    host = host_fixture(torch, fixture, pins)
    for mask, tiles in MASKS.items():
        # The common mask kernel has four tile slots. Union is idempotent.
        host[mask] = torch.tensor((*tiles, tiles[-1]), dtype=torch.int32)
    # Keep host owners alive until all nonblocking uploads have drained.
    gpu = {name: value.to("mps", non_blocking=True) for name, value in host.items()}
    weight = gpu["weight.bf16"]
    weight_f32 = weight.float()
    shader = shader_source()
    library = torch.mps.compile_shader(shader)
    conv, functions = pinned_conv3d(torch)
    header = metadata(manifest, pins, shader)
    require(
        functions == header["upstream_functions"], "Pinned upstream functions changed"
    )
    masked = {}
    for mask in MASKS:
        masked[mask] = torch.empty_like(weight)
        library.mask_weights(
            weight,
            gpu[mask],
            masked[mask],
            4,
            threads=weight.numel(),
            arg_casts={3: "int32"},
        )

    stacked = []
    for case in CASES:
        cache = gpu[f"{case}-cache.bf16"]
        joined = torch.cat(
            (cache, gpu[f"{case}-input.bf16"], torch.zeros_like(cache)), dim=2
        )
        indices = gpu[f"{case}-indices.u32"]
        require(
            all(v.is_contiguous() for v in (joined, weight, indices)),
            "Metal input layout changed",
        )

        def sampled(weights):
            values = torch.empty(SAMPLES, dtype=torch.float32, device="mps")
            library.dense_g8_samples(joined, weights, indices, values, threads=SAMPLES)
            return values

        native = sampled(weight)
        native_masked = [sampled(weights) for weights in masked.values()]
        upstream = upstream_sample(torch, conv, joined.float(), weight_f32, indices)
        upstream_bf16 = upstream_sample(torch, conv, joined, weight, indices)
        rows = (
            native,
            native.to(torch.bfloat16).float(),
            upstream,
            upstream.to(torch.bfloat16).float(),
            upstream_bf16.float(),
            *native_masked,
        )
        require(
            len(rows) == len(COLUMNS)
            and all(r.device.type == "mps" and r.shape == (SAMPLES,) for r in rows),
            f"{case}: observation layout changed",
        )
        stacked.append(torch.stack(rows).flatten())
        print(f"Enqueued {case}: dense G8, masked controls and upstream", flush=True)

    readout = torch.cat(stacked).to("cpu")
    payload = string_at(readout.data_ptr(), readout.numel() * readout.element_size())
    header["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    with (output / "observations.f32").open("xb") as stream:
        stream.write(payload)
    with (output / "observations.json").open("x") as stream:
        json.dump(header, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {len(payload)} bytes after the sole terminal readout", flush=True)


def masked_reference(manifest, pins):
    raw = (MASK_REFERENCE / "observations.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MASK_HEADER_SHA, "Masked header changed")
    header = json.loads(raw)
    require(
        header["fixture_manifest_sha256"] == manifest
        and header["fixture_sha256"] == pins
        and header["cases"] == list(CASES)
        and header["masks"] == {mask: list(tiles) for mask, tiles in MASKS.items()}
        and header["samples"] == SAMPLES
        and header["upstream_sha256"] == UPSTREAM_SHA
        and header["torch_version"] == "2.8.0",
        "Masked control fixture or backend changed",
    )
    columns = header["columns"]
    values = read_column(
        MASK_REFERENCE / "observations.f32",
        "f",
        header["payload_sha256"],
        len(CASES) * len(columns) * SAMPLES + 2 * 9 * 4,
    )
    controls = {}
    for i, case in enumerate(CASES):
        controls[case] = {}
        for mask in MASKS:
            start = (i * len(columns) + columns.index(f"upstream.{mask}.f32")) * SAMPLES
            controls[case][mask] = values[start : start + SAMPLES]
    return controls


def summarize_case(fixture, pins, case, rows, controls):
    def retained(name, code):
        return read_column(fixture / name, code, pins[name], SAMPLES)

    indices = retained(f"{case}-indices.u32", "I")
    require(
        len(set(indices)) == SAMPLES and max(indices) < math.prod(SHAPE),
        f"{case}: invalid sampled coordinates",
    )
    dense_f32 = retained(f"{case}-mps-f32.f32", "f")
    actual_bf16 = array("f", map(bf16, retained(f"torch-{case}-pre.bf16", "H")))
    require(
        rows["upstream.f32"].tobytes() == dense_f32.tobytes()
        and rows["upstream.bf16"].tobytes() == actual_bf16.tobytes(),
        f"{case}: fresh dense upstream differs from pinned baseline",
    )
    for mask in MASKS:
        require(
            rows[f"masked.{mask}.f32"].tobytes() == controls[mask].tobytes(),
            f"{case}: dense kernel failed the {mask} masked control",
        )
    for name in ("native", "upstream"):
        require(
            all(
                round_bf16(x) << 16 == rounded
                for x, rounded in zip(
                    rows[f"{name}.f32"], bit_view(rows[f"{name}.rounded"]), strict=True
                )
            ),
            f"{case}: incorrect BF16 rounding in {name}",
        )

    previous = array(
        "f", map(bf16, retained(f"{case}-order0-lanes1-tree0.pre.bf16", "H"))
    )
    prior = failures(previous, actual_bf16)
    require(
        prior and all(map(math.isfinite, previous)), f"{case}: invalid old baseline"
    )
    native_f32 = rows["native.f32"]
    native_bf16 = rows["native.rounded"]
    native_failed = failures(native_bf16, actual_bf16)
    f32_mismatches = {
        i
        for i, (a, b) in enumerate(
            zip(bit_view(native_f32), bit_view(dense_f32), strict=True)
        )
        if a != b
    }
    scopes = {name: set(points) for name, points in SCOPES.items()} | {
        "prior_failed": prior
    }

    first = {}
    for name, points in (("f32", f32_mismatches), ("bf16", native_failed)):
        if points:
            i = min(points)
            c, rest = divmod(indices[i], math.prod(SHAPE[2:]))
            t, rest = divmod(rest, math.prod(SHAPE[3:]))
            y, x = divmod(rest, SHAPE[-1])
            first[name] = {
                "sample": i,
                "ctyx": [c, t, y, x],
                "native_f32": native_f32[i],
                "upstream_f32": dense_f32[i],
                "native_bf16": native_bf16[i],
                "upstream_bf16": actual_bf16[i],
            }
    return {
        "masked_f32_bit_matches": {mask: SAMPLES for mask in MASKS},
        "dense_f32": {
            **measure(native_f32, dense_f32),
            "bit_mismatches": len(f32_mismatches),
        },
        "native_rounded_vs_actual_bf16": measure(native_bf16, actual_bf16),
        "upstream_rounded_vs_actual_bf16": measure(
            rows["upstream.rounded"], actual_bf16
        ),
        "scopes": {
            name: {
                "samples": len(points),
                "f32_bit_mismatches": len(f32_mismatches & points),
                "old_bf16_failures": len(prior & points),
                "native_bf16_failures": len(native_failed & points),
            }
            for name, points in scopes.items()
        },
        "bf16_failure_churn": {
            "old_total": len(prior),
            "resolved": len(prior - native_failed),
            "still_failing": len(prior & native_failed),
            "new": len(native_failed - prior),
        },
        "first_mismatch": first,
    }


def replay(fixture, output):
    manifest, pins = fixture_pins(fixture)
    header = json.loads((output / "observations.json").read_text())
    require(
        all(
            header.get(k) == v
            for k, v in metadata(manifest, pins, shader_source()).items()
        ),
        "Unexpected observation metadata",
    )
    controls = masked_reference(manifest, pins)
    values = read_column(
        output / "observations.f32",
        "f",
        header["payload_sha256"],
        len(CASES) * len(COLUMNS) * SAMPLES,
    )
    require(all(map(math.isfinite, values)), "Non-finite GPU observations")
    cases = {}
    for i, case in enumerate(CASES):
        rows = {}
        for j, name in enumerate(COLUMNS):
            start = (i * len(COLUMNS) + j) * SAMPLES
            rows[name] = values[start : start + SAMPLES]
        cases[case] = summarize_case(fixture, pins, case, rows, controls[case])
    return {"observations": header, "cases": cases}


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
