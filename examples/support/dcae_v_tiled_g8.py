#!/usr/bin/env python3
"""Full stage-2 pre-bias Conv3D parity against the pinned MPS operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

from dcae_v_k_partials import (
    CASES,
    SAMPLES,
    SHAPE,
    UPSTREAM_SHA,
    WEIGHT,
    fixture_pins,
    host_fixture,
    pinned_conv3d,
    read_tensor,
)
from dcae_v_k_partials_compare import read_column, require
from dcae_v_reduction_compare import ATOL, RTOL


SHADER = Path(__file__).with_suffix(".metal")
ELEMENTS = math.prod(SHAPE)
TILE_CHANNELS = 4
CHANNELS_PER_DISPATCH = 32
THREADGROUP = 128
COMPARE_THREADS = 65536
STATS = (
    "covered",
    "bit_mismatches",
    "outside_tolerance",
    "nonfinite",
    "first_bit_mismatch",
    "max_abs",
)


def check_tile_partition():
    channels, frames, height, width = SHAPE[1:]
    xs = [col // 32 * 64 + col % 32 for col in range(width // 2)]
    require(sorted(xs + [x + 32 for x in xs]) == list(range(width)), "X coverage")
    tiles = [
        first + tile * TILE_CHANNELS + lane
        for first in range(0, channels, CHANNELS_PER_DISPATCH)
        for tile in range(CHANNELS_PER_DISPATCH // TILE_CHANNELS)
        for lane in range(TILE_CHANNELS)
    ]
    require(tiles == list(range(channels)), "Channel coverage")
    threads = CHANNELS_PER_DISPATCH // TILE_CHANNELS * frames * height * (width // 2)
    require(threads % THREADGROUP == 0 and width == 128, "Tiled launch geometry")
    outputs_per_thread = TILE_CHANNELS * 2
    dispatches = channels // CHANNELS_PER_DISPATCH
    require(threads * dispatches * outputs_per_thread == ELEMENTS, "Extent")
    return threads


def check_tensor(value, shape, dtype):
    require(
        value.shape == shape
        and value.dtype == dtype
        and value.device.type == "mps"
        and value.is_contiguous()
        and value.storage_offset() == 0,
        "Expected an unoffset contiguous GPU tensor with the admitted shape and dtype",
    )


def prepare(torch, fixture, *, sample_controls):
    manifest, pins = fixture_pins(fixture)
    host = host_fixture(torch, fixture, pins)
    if sample_controls:
        for case in CASES:
            name = f"{case}-mps-f32.f32"
            values = read_column(fixture / name, "f", pins[name], SAMPLES)
            host[name] = torch.frombuffer(values, dtype=torch.float32)
            name = f"torch-{case}-pre.bf16"
            host[name] = read_tensor(
                torch, fixture / name, torch.uint16, (SAMPLES,), sha256=pins[name]
            )
    # Retain host owners through the terminal drain of these asynchronous uploads.
    gpu = {name: value.to("mps", non_blocking=True) for name, value in host.items()}
    weight = gpu["weight.bf16"]
    check_tensor(weight, WEIGHT, torch.bfloat16)
    for case in CASES:
        cache = gpu[f"{case}-cache.bf16"]
        joined = torch.cat(
            (cache, gpu[f"{case}-input.bf16"], torch.zeros_like(cache)), dim=2
        )
        check_tensor(joined, (*SHAPE[:2], SHAPE[2] + 2, *SHAPE[3:]), weight.dtype)
        gpu[f"{case}-joined"] = joined

    shader = SHADER.read_text()
    library = torch.mps.compile_shader(shader)
    conv, functions = pinned_conv3d(torch)
    header = {
        "fixture_manifest_sha256": manifest,
        "fixture_sha256": pins,
        "shader_sha256": hashlib.sha256(shader.encode()).hexdigest(),
        "upstream_sha256": UPSTREAM_SHA,
        "upstream_functions": functions,
        "torch_version": torch.__version__,
        "shape": list(SHAPE),
        "weight_shape": list(WEIGHT),
        "k": math.prod(WEIGHT[1:]),
        "fma_order": "channel-group-8/tap-ascending/lane-ascending",
        "tile": [TILE_CHANNELS, 2],
        "channels_per_dispatch": CHANNELS_PER_DISPATCH,
        "terminal_tensor_transfers": 1,
        "explicit_mps_synchronizations": 0,
        "cpu_fallback": False,
        "atol": ATOL,
        "rtol": RTOL,
    }
    packed = torch.empty_like(weight)
    library.pack_g8_weights(weight, packed, threads=weight.numel() // TILE_CHANNELS)
    gpu["packed-weight"] = packed
    return host, gpu, library, conv, header


def launch_tiled(kernel, joined, packed, output, threads):
    for first in range(0, SHAPE[1], CHANNELS_PER_DISPATCH):
        kernel(
            joined,
            packed,
            output,
            first // TILE_CHANNELS,
            threads=threads,
            group_size=THREADGROUP,
            arg_casts={3: "int32"},
        )


def upstream(conv, joined, weight):
    out = conv(joined, weight, None, (1, 1, 1), (0, 1, 1), (1, 1, 1), 1)
    check_tensor(out, SHAPE, weight.dtype)
    return out


def compare(torch, kernel, actual, expected):
    check_tensor(actual, expected.shape, expected.dtype)
    check_tensor(expected, actual.shape, actual.dtype)
    require(actual.data_ptr() != expected.data_ptr(), "Aliased comparison operands")
    require(kernel.thread_execution_width == 32, "Unexpected comparison SIMD width")
    stats = torch.zeros(len(STATS), dtype=torch.int32, device="mps")
    stats[4].fill_(-1)
    kernel(
        actual,
        expected,
        stats,
        actual.numel(),
        ATOL,
        RTOL,
        threads=COMPARE_THREADS,
        group_size=THREADGROUP,
        arg_casts={3: "int32"},
    )
    return stats


def decode_stats(raw):
    coverage, unequal, outside, nonfinite, first, max_abs = raw
    return dict(
        zip(
            STATS,
            (
                coverage,
                unequal,
                outside,
                nonfinite,
                None if first == -1 else first,
                struct.unpack("<f", struct.pack("<I", max_abs))[0],
            ),
            strict=True,
        )
    )


def comparison_witness(torch, kernel, dtype):
    count = COMPARE_THREADS + 5
    actual = torch.zeros(count, dtype=dtype, device="mps")
    expected = torch.zeros_like(actual)
    bits_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    actual.view(bits_dtype)[0].fill_(-(1 << (actual.element_size() * 8 - 1)))
    actual[31].fill_(2.0**-17)
    actual[32].fill_(float("nan"))
    expected[32].fill_(float("nan"))
    actual[-1].fill_(2.0)
    expected[-1].fill_(1.0)
    return compare(torch, kernel, actual, expected)


def write_result(output, name, result):
    with (output / name).open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def parity(torch, fixture, output):
    threads = check_tile_partition()
    host, gpu, library, conv, header = prepare(torch, fixture, sample_controls=True)
    stats = {}
    for precision, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        kernel = getattr(library, f"tiled_g8_{precision}")
        comparator = getattr(library, f"compare_{precision}")
        stats[f"witness/{precision}"] = comparison_witness(torch, comparator, dtype)
        native = torch.empty(SHAPE, dtype=dtype, device="mps")
        check_tensor(native, SHAPE, dtype)
        weight = gpu["weight.bf16"].to(dtype)
        for case in CASES:
            joined = gpu[f"{case}-joined"]
            native.fill_(float("nan"))
            launch_tiled(kernel, joined, gpu["packed-weight"], native, threads)
            reference = upstream(conv, joined.to(dtype), weight)
            stats[f"{case}/{precision}/full"] = compare(
                torch, comparator, native, reference
            )
            sample = torch.index_select(
                reference.reshape(-1), 0, gpu[f"{case}-indices.u32"]
            )
            name = (
                f"{case}-mps-f32.f32"
                if precision == "f32"
                else f"torch-{case}-pre.bf16"
            )
            stats[f"{case}/{precision}/sample_pin"] = compare(
                torch, comparator, sample, gpu[name]
            )
            del reference
            print(
                f"Enqueued {case} {precision} full output and retained control",
                flush=True,
            )

    readout = torch.stack(tuple(stats.values())).to("cpu").tolist()
    host.clear()
    measured = {key: decode_stats(row) for key, row in zip(stats, readout, strict=True)}
    failed = {}
    for key, row in measured.items():
        if key.startswith("witness/"):
            expected = (COMPARE_THREADS + 5, 3, 1, 1, 0, 1.0)
        else:
            count = ELEMENTS if key.endswith("/full") else SAMPLES
            expected = (count, 0, 0, 0, None, 0.0)
        if tuple(row.values()) != expected:
            failed[key] = row
    result = {
        "schema": "dcae_tiled_g8_parity/v1",
        "header": header,
        "stats": measured,
        "passed": not failed,
    }
    write_result(output, "parity.json", result)
    require(not failed, f"Full-output or control comparison failed: {failed}")
    print(
        f"Matched all {ELEMENTS:,} outputs in F32 and BF16 on both caches", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--admitted", action="store_true")
    args = parser.parse_args()
    if (
        not args.admitted
        or sys.byteorder != "little"
        or not args.output.is_dir()
        or (args.output / "parity.json").exists()
    ):
        parser.error("Use the cache-aware guard and a fresh scratch directory")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    require(
        torch.__version__ == "2.8.0" and torch.backends.mps.is_available(),
        "Requires the pinned Torch 2.8.0 MPS runtime",
    )
    torch.set_grad_enabled(False)
    parity(torch, args.fixture, args.output)


if __name__ == "__main__":
    main()
