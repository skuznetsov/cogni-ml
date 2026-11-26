#!/usr/bin/env python3
"""Compare warm BF16 Conv3D batches with one terminal readout per batch."""

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
    ELEMENTS,
    SHAPE,
    STATS,
    check_tile_partition,
    compare,
    decode_stats,
    launch_tiled,
    prepare,
    require,
    upstream,
    write_result,
)


WARMUP = ("native", "mps", "mps", "native")
ORDERS = (WARMUP, ("mps", "native", "native", "mps"))
REPEATS = 2
EXACT = dict(zip(STATS, (ELEMENTS, 0, 0, 0, None, 0.0), strict=True))


def output_tail(out, torch):
    return out.view(torch.int16).reshape(-1)[-1:]


def warm(torch, arms_by_case, comparator):
    rows = []
    for arms in arms_by_case.values():
        outputs = {arm: arms[arm]() for arm in WARMUP}
        stats = compare(torch, comparator, outputs["native"], outputs["mps"])
        tail = output_tail(outputs["mps"], torch).to(torch.int32)
        rows.append(torch.cat((stats, tail)))

    # One terminal drain for the shared setup, warmups and full-output check.
    readout = torch.stack(rows).to("cpu").tolist()
    checks = {
        case: decode_stats(row[:-1]) for case, row in zip(CASES, readout, strict=True)
    }
    require(
        all(row == EXACT for row in checks.values()),
        f"Warm BF16 parity failed: {checks}",
    )
    tails = {case: row[-1] for case, row in zip(CASES, readout, strict=True)}
    return checks, tails


def time_batches(torch, arms_by_case, tails):
    records = []
    for case, arms in arms_by_case.items():
        for order, sequence in enumerate(ORDERS):
            for position, arm in enumerate(sequence):
                run = arms[arm]
                started = perf_counter_ns()
                for _ in range(REPEATS):
                    out = run()
                encoded = perf_counter_ns()
                tail = output_tail(out, torch).to("cpu").item()
                finished = perf_counter_ns()
                require(tail == tails[case], f"{case}/{arm}: terminal output differs")
                records.append(
                    {
                        "case": case,
                        "order": order,
                        "position": position,
                        "arm": arm,
                        "calls": REPEATS,
                        "elapsed_ms": (finished - started) / 1e6,
                        "encode_return_ms": (encoded - started) / 1e6,
                        "tail_bits_i16": tail,
                    }
                )
            print(f"Measured {case} order {' '.join(sequence)}", flush=True)
    return records


def summarize(records):
    require(len(records) == len(CASES) * sum(map(len, ORDERS)), "Wrong batch count")
    summary = {}
    for case in CASES:
        rows = [row for row in records if row["case"] == case]
        samples = {"native": [], "mps": []}
        ratios = []
        for order, sequence in enumerate(ORDERS):
            block = [row for row in rows if row["order"] == order]
            require(
                tuple(row["arm"] for row in block) == sequence, "Wrong paired order"
            )
            for position, row in enumerate(block):
                require(
                    row["position"] == position
                    and row["calls"] == REPEATS
                    and math.isfinite(row["elapsed_ms"])
                    and 0 < row["encode_return_ms"] <= row["elapsed_ms"],
                    f"{case}: invalid batch interval",
                )
                samples[row["arm"]].append(row["elapsed_ms"] / REPEATS)
            for a, b in zip(block[::2], block[1::2], strict=True):
                pair = {row["arm"]: row["elapsed_ms"] for row in (a, b)}
                ratios.append(pair["native"] / pair["mps"])
        summary[case] = {
            "native_ms_per_call": median(samples["native"]),
            "mps_ms_per_call": median(samples["mps"]),
            "paired_native_over_mps": ratios,
            "median_paired_native_over_mps": median(ratios),
        }
    return summary


def make_arms(kernel, conv, joined, weight, packed, native, threads):
    def tiled():
        launch_tiled(kernel, joined, packed, native, threads)
        return native

    def mps():
        return upstream(conv, joined, weight)

    return {"native": tiled, "mps": mps}


def benchmark(torch, fixture, output):
    raw_parity = (output / "parity.json").read_bytes()
    parity = json.loads(raw_parity)
    require(
        parity["schema"] == "dcae_tiled_g8_parity/v1"
        and parity["passed"] is True
        and all(
            parity["stats"].get(f"{case}/{dtype}/full") == EXACT
            for case in CASES
            for dtype in ("f32", "bf16")
        ),
        "Full-output parity must pass before timing",
    )
    host, gpu, library, conv, header = prepare(torch, fixture, sample_controls=False)
    require(header == parity["header"], "Parity and timing inputs differ")
    threads = check_tile_partition()
    arms = {
        case: make_arms(
            library.tiled_g8_bf16,
            conv,
            gpu[f"{case}-joined"],
            gpu["weight.bf16"],
            gpu["packed-weight"],
            torch.empty(SHAPE, dtype=torch.bfloat16, device="mps"),
            threads,
        )
        for case in CASES
    }
    warm_checks, tails = warm(torch, arms, library.compare_bf16)
    host.clear()
    print("Warmups and full BF16 checks complete", flush=True)
    records = time_batches(torch, arms, tails)
    report = {
        "schema": "dcae_tiled_g8_timing/v1",
        "header": header,
        "parity_sha256": hashlib.sha256(raw_parity).hexdigest(),
        "measurement": "Warm batch wall time with one terminal scalar readout per batch",
        "mps_baseline": "Pinned upstream Conv3D on Torch MPS, eager output",
        "native": "Metal G8 tiled kernel, prepared packed weights and reusable output",
        "warmup_output_stats": warm_checks,
        "warmup": list(WARMUP),
        "batches": records,
        "summary": summarize(records),
    }
    write_result(output, "timing.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)


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
        or (args.output / "timing.json").exists()
    ):
        parser.error("Use the cache-aware guard and a scratch directory with parity")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    require(
        torch.__version__ == "2.8.0" and torch.backends.mps.is_available(),
        "Requires the pinned Torch 2.8.0 MPS runtime",
    )
    torch.set_grad_enabled(False)
    benchmark(torch, args.fixture, args.output)


if __name__ == "__main__":
    main()
