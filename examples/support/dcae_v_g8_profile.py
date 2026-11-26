#!/usr/bin/env python3
"""Counterbalanced load and dependency probes for the exact tiled G8 Conv3D."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

from dcae_v_tiled_g8 import (
    CASES,
    CHANNELS_PER_DISPATCH,
    COMPARE_THREADS,
    SAMPLES,
    SHADER,
    SHAPE,
    THREADGROUP,
    TILE_CHANNELS,
    check_tensor,
    check_tile_partition,
    compare,
    comparison_witness,
    decode_stats,
    launch_tiled,
    prepare,
    require,
    upstream,
    write_result,
)
from dcae_v_tiled_g8_bench import EXACT, output_tail


PROFILE_SHADER = Path(__file__).with_suffix(".metal")
FIRST_TILES = (0, 32, 64, 96)
SLICE_CHANNELS = len(FIRST_TILES) * CHANNELS_PER_DISPATCH
SLICE_VALUES = SLICE_CHANNELS * math.prod(SHAPE[2:])
PROFILE_PAIRS = (
    ("strided", "strided_again"),
    ("strided", "packed"),
    ("strided", "strided_split"),
    ("packed", "packed_split"),
    ("strided_split", "packed_split"),
)
CANDIDATES = ("packed",)


def setup(torch, fixture, *, sample_controls):
    host, gpu, _, conv, header = prepare(
        torch, fixture, sample_controls=sample_controls
    )
    source = PROFILE_SHADER.read_text()
    library = torch.mps.compile_shader(SHADER.read_text() + "\n" + source)
    header["profile_shader_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return host, gpu, library, conv, header


def pack(library, joined, packed):
    library.g8_pack_input(joined, packed, threads=joined.numel() // 8, group_size=128)


def time_pairs(case, arms, tails, pairs, read_tail, calls):
    records = []
    summary = {}
    for left, right in pairs:
        samples = {left: [], right: []}
        ratios = []
        for order, sequence in enumerate(
            ((left, right, right, left), (right, left, left, right))
        ):
            durations = []
            for position, arm in enumerate(sequence):
                run = arms[arm]
                started = perf_counter_ns()
                for _ in range(calls):
                    output = run()
                encoded = perf_counter_ns()
                tail = read_tail(output).to("cpu").item()
                finished = perf_counter_ns()
                require(tail == tails[arm], f"{case}/{arm}: terminal output differs")
                elapsed = (finished - started) / 1e6
                encode_return = (encoded - started) / 1e6
                require(0 < encode_return <= elapsed, f"{case}/{arm}: invalid interval")
                durations.append(elapsed)
                samples[arm].append(elapsed / calls)
                records.append(
                    dict(
                        case=case,
                        pair=[left, right],
                        order=order,
                        position=position,
                        arm=arm,
                        calls=calls,
                        elapsed_ms=elapsed,
                        encode_return_ms=encode_return,
                        tail_bits_i16=tail,
                    )
                )
            for offset in (0, 2):
                pair = dict(
                    zip(
                        sequence[offset : offset + 2],
                        durations[offset : offset + 2],
                        strict=True,
                    )
                )
                ratios.append(pair[right] / pair[left])
        summary[f"{right}/{left}"] = {
            "ms_per_call": {arm: median(times) for arm, times in samples.items()},
            "paired_right_over_left": ratios,
            "median_paired_right_over_left": median(ratios),
        }
        print(
            f"Measured {case} {right}/{left}: {summary[f'{right}/{left}']}", flush=True
        )
    return records, summary


def profile(torch, fixture, output):
    host, gpu, library, _, header = setup(torch, fixture, sample_controls=False)
    threads = check_tile_partition()
    native = torch.empty(SHAPE, dtype=torch.bfloat16, device="mps")
    packed = torch.empty_like(gpu[f"{CASES[0]}-joined"])
    check_tensor(packed, gpu[f"{CASES[0]}-joined"].shape, torch.bfloat16)
    weight = gpu["packed-weight"]
    tail_index = (FIRST_TILES[-1] * TILE_CHANNELS + CHANNELS_PER_DISPATCH) * math.prod(
        SHAPE[2:]
    ) - 1

    def read_tail(result):
        return result.view(torch.int16).reshape(-1)[tail_index : tail_index + 1]

    report = dict(
        schema="dcae_g8_profile/v1",
        header=header,
        first_output_tiles=list(FIRST_TILES),
        slice_values=SLICE_VALUES,
        measurement="Completed BF16 slices, C8 setup outside measured batches",
        cases={},
    )
    for case in CASES:
        joined = gpu[f"{case}-joined"]
        pack(library, joined, packed)

        def sliced(kernel, source):
            def run():
                for first in FIRST_TILES:
                    kernel(
                        source,
                        weight,
                        native,
                        first,
                        threads=threads,
                        group_size=THREADGROUP,
                        arg_casts={3: "int32"},
                    )
                return native

            return run

        arms = {
            "strided": sliced(library.g8_profile_strided_bf16, joined),
            "strided_split": sliced(library.g8_profile_strided_split_bf16, joined),
            "packed": sliced(library.g8_profile_packed_bf16, packed),
            "packed_split": sliced(library.g8_profile_packed_split_bf16, packed),
        }

        def selected(result):
            return torch.cat(
                tuple(
                    result[
                        :,
                        first * TILE_CHANNELS : first * TILE_CHANNELS
                        + CHANNELS_PER_DISPATCH,
                    ]
                    for first in FIRST_TILES
                ),
                dim=1,
            )

        native.fill_(float("nan"))
        reference = selected(sliced(library.tiled_g8_bf16, joined)())
        checks = []
        for run in arms.values():
            native.fill_(float("nan"))
            value = run()
            stats = compare(torch, library.compare_bf16, selected(value), reference)
            checks.append(torch.cat((stats, read_tail(value).to(torch.int32))))
        # The warmup, selected-output checks and signatures use one terminal drain.
        raw = torch.stack(checks).to("cpu").tolist()
        measured = {
            arm: decode_stats(row[:-1]) for arm, row in zip(arms, raw, strict=True)
        }
        require(
            all(
                row["covered"] == SLICE_VALUES and row["nonfinite"] == 0
                for row in measured.values()
            )
            and measured["strided"]["bit_mismatches"] == 0
            and measured["packed"]["bit_mismatches"] == 0
            and measured["strided_split"] == measured["packed_split"]
            and measured["strided_split"]["bit_mismatches"] > 0,
            f"{case}: ablation output check failed: {measured}",
        )
        del reference
        tails = {arm: row[-1] for arm, row in zip(arms, raw, strict=True)}
        tails["strided_again"] = tails["strided"]
        arms["strided_again"] = arms["strided"]
        batches, summary = time_pairs(case, arms, tails, PROFILE_PAIRS, read_tail, 1)
        report["cases"][case] = dict(
            output_stats=measured, batches=batches, summary=summary
        )
    host.clear()
    write_result(output, "profile.json", report)


def candidate_launch(
    library, candidate, precision, joined, scratch, weight, out, threads
):
    if candidate == "packed":
        pack(library, joined, scratch)
        joined = scratch
    launch_tiled(
        getattr(library, f"g8_profile_{candidate}_{precision}"),
        joined,
        weight,
        out,
        threads,
    )
    return out


def parity(torch, fixture, output, candidate):
    host, gpu, library, conv, header = setup(torch, fixture, sample_controls=True)
    threads = check_tile_partition()
    scratch = (
        torch.empty_like(gpu[f"{CASES[0]}-joined"]) if candidate == "packed" else None
    )
    stats = {}
    for precision, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        comparator = getattr(library, f"compare_{precision}")
        stats[f"witness/{precision}"] = comparison_witness(torch, comparator, dtype)
        native = torch.empty(SHAPE, dtype=dtype, device="mps")
        weight = gpu["weight.bf16"].to(dtype)
        for case in CASES:
            joined = gpu[f"{case}-joined"]
            native.fill_(float("nan"))
            candidate_launch(
                library,
                candidate,
                precision,
                joined,
                scratch,
                gpu["packed-weight"],
                native,
                threads,
            )
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
            print(f"Enqueued {case} {precision} {candidate} parity", flush=True)

    readout = torch.stack(tuple(stats.values())).to("cpu").tolist()
    host.clear()
    checks = {key: decode_stats(row) for key, row in zip(stats, readout, strict=True)}
    failed = {}
    for key, row in checks.items():
        expected = EXACT.values()
        if key.startswith("witness/"):
            expected = (COMPARE_THREADS + 5, 3, 1, 1, 0, 1.0)
        elif key.endswith("/sample_pin"):
            expected = (SAMPLES, 0, 0, 0, None, 0.0)
        if tuple(row.values()) != tuple(expected):
            failed[key] = row
    write_result(
        output,
        f"{candidate}.parity.json",
        dict(
            schema="dcae_g8_profile_parity/v1",
            header=header,
            candidate=candidate,
            stats=checks,
            passed=not failed,
        ),
    )
    require(
        not failed, f"{candidate}: full-output or retained control failed: {failed}"
    )


def timing(torch, fixture, output, candidate):
    raw_parity = (output / f"{candidate}.parity.json").read_bytes()
    control = json.loads(raw_parity)
    host, gpu, library, conv, header = setup(torch, fixture, sample_controls=False)
    require(
        control["schema"] == "dcae_g8_profile_parity/v1"
        and control["candidate"] == candidate
        and control["header"] == header
        and control["passed"] is True
        and all(
            control["stats"][f"{case}/{precision}/full"] == EXACT
            for case in CASES
            for precision in ("f32", "bf16")
        ),
        "Full candidate parity on identical inputs must pass before timing",
    )
    threads = check_tile_partition()
    native = torch.empty(SHAPE, dtype=torch.bfloat16, device="mps")
    scratch = (
        torch.empty_like(gpu[f"{CASES[0]}-joined"]) if candidate == "packed" else None
    )
    report = dict(
        schema="dcae_g8_profile_timing/v1",
        header=header,
        candidate=candidate,
        parity_sha256=hashlib.sha256(raw_parity).hexdigest(),
        measurement="Completed BF16 calls including any per-call input pack",
        cases={},
    )
    for case in CASES:
        joined = gpu[f"{case}-joined"]

        def baseline():
            launch_tiled(
                library.tiled_g8_bf16, joined, gpu["packed-weight"], native, threads
            )
            return native

        def exact_candidate():
            return candidate_launch(
                library,
                candidate,
                "bf16",
                joined,
                scratch,
                gpu["packed-weight"],
                native,
                threads,
            )

        def mps():
            return upstream(conv, joined, gpu["weight.bf16"])

        arms = dict(strided=baseline, candidate=exact_candidate, mps=mps)
        reference = mps()
        warm = []
        for arm in ("strided", "candidate", "candidate", "strided"):
            value = arms[arm]()
            warm.append(compare(torch, library.compare_bf16, value, reference))

        def read_tail(result):
            return output_tail(result, torch)

        tail = read_tail(reference).to(torch.int32)
        rows = torch.cat((*warm, tail)).to("cpu").tolist()
        del reference
        warm_checks = [
            decode_stats(rows[i : i + len(EXACT)])
            for i in range(0, len(rows) - 1, len(EXACT))
        ]
        require(
            all(row == EXACT for row in warm_checks), f"{case}: warm BF16 parity failed"
        )
        tails = dict.fromkeys(arms, rows[-1])
        batches, summary = time_pairs(
            case,
            arms,
            tails,
            (("strided", "candidate"), ("mps", "candidate")),
            read_tail,
            2,
        )
        report["cases"][case] = dict(
            warm_output_stats=warm_checks, batches=batches, summary=summary
        )
    host.clear()
    write_result(output, f"{candidate}.timing.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("profile", "parity", "timing"))
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--admitted", action="store_true")
    args = parser.parse_args()
    if not args.admitted or sys.byteorder != "little" or not args.output.is_dir():
        parser.error("Use the cache-aware guard and an existing scratch directory")
    if (args.mode == "profile") == (args.candidate is not None):
        parser.error("Parity and timing require an exact candidate")
    result = (
        "profile.json"
        if args.mode == "profile"
        else f"{args.candidate}.{args.mode}.json"
    )
    if (args.output / result).exists():
        parser.error("Use a fresh report path")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    require(
        torch.__version__ == "2.8.0" and torch.backends.mps.is_available(),
        "Requires the pinned Torch 2.8.0 MPS runtime",
    )
    torch.set_grad_enabled(False)
    if args.mode == "profile":
        profile(torch, args.fixture, args.output)
    elif args.mode == "parity":
        parity(torch, args.fixture, args.output, args.candidate)
    else:
        timing(torch, args.fixture, args.output, args.candidate)


if __name__ == "__main__":
    main()
