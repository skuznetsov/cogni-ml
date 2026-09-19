#!/usr/bin/env python3
"""Check bounded no-GPU residency samples, not full-model safety or performance."""
import json
import math
import re
import sys
from pathlib import Path

SIZE = 64 << 20
SIGNAL = 48 << 20
TOLERANCE = 8 << 20


def read_run(path, mode):
    plain_mode = mode.removeprefix("file-")
    if plain_mode not in ("control", "request", "hold"):
        raise ValueError("unknown mode")
    text = Path(path).read_text()
    exits = re.findall(r"\[EXIT: ([^\]]+)\]", text)
    if exits != ["0"] or any(x in text for x in ("[KILL]", "[ABORT]", "FAIL:")):
        raise ValueError(f"{mode}: missing clean runner exit")
    rows = [json.loads(x) for x in text.splitlines() if x.startswith('{')]
    cycles = 1 if plain_mode == "hold" else 3
    expected_config = dict(event="config", mode=mode, bytes=SIZE, cycles=cycles,
                           device="Apple M2 Max", gpu_commands=0)
    if not rows or rows[0] != expected_config or rows[-1] != dict(event="complete", mode=mode, gpu_commands=0):
        raise ValueError(f"{mode}: bad config/completion")
    phases = ["baseline", "touched", "prepared"]
    phases += (["retained", "after_250ms", "after_1000ms", "after_5000ms", "released"]
               if plain_mode == "hold" else ["released", "after_250ms", "after_1000ms", "after_5000ms"])
    samples = rows[1:-1]
    if len(samples) != cycles * len(phases):
        raise ValueError(f"{mode}: wrong sample count")
    result = []
    prior_time = -1
    for cycle in range(cycles):
        batch = samples[cycle * len(phases):(cycle + 1) * len(phases)]
        for row, phase in zip(batch, phases):
            if any(row.get(k) != v for k, v in dict(event="sample", mode=mode, cycle=cycle, phase=phase, bytes=SIZE).items()):
                raise ValueError(f"{mode}: sample identity/order")
            for field in ("footprint", "resident", "metal_allocated", "elapsed_ms"):
                value = row.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{mode}: invalid {field}")
            if row["elapsed_ms"] < prior_time:
                raise ValueError(f"{mode}: time regression")
            prior_time = row["elapsed_ms"]
        by_phase = {x["phase"]: x for x in batch}
        start = by_phase["retained" if plain_mode == "hold" else "released"]["elapsed_ms"]
        for delay in (250, 1000, 5000):
            if by_phase[f"after_{delay}ms"]["elapsed_ms"] - start < delay - 1:
                raise ValueError(f"{mode}: observation interval too short")
        result.append(by_phase)
    return result


def evaluate(control, request, hold, *, file_backed=False):
    # Clean file pages may not contribute to phys_footprint. A resident-size
    # positive control qualifies task accounting only, never global unpinning.
    metric = "resident" if file_backed else "footprint"
    runs = {"control": control, "request": request, "hold": hold}
    for mode, batches in runs.items():
        for batch in batches:
            if batch["touched"][metric] - batch["baseline"][metric] < SIGNAL:
                raise ValueError(f"{mode}: insufficient touched-page signal")
    h = hold[0]
    if h["after_5000ms"][metric] - h["baseline"][metric] < SIGNAL:
        raise ValueError("held positive control was not detected")
    if abs(h["released"][metric] - h["baseline"][metric]) > TOLERANCE:
        raise ValueError("held control did not reclaim within tolerance")
    deltas = {}
    for mode in ("control", "request"):
        batches = runs[mode]
        base = batches[0]["baseline"][metric]
        # Initial baseline, not per-cycle baseline: cumulative retention stays visible.
        deltas[mode] = [b["after_5000ms"][metric] - base for b in batches]
    if any(abs(x) > TOLERANCE for x in deltas["control"]):
        raise ValueError(f"control baseline drift: {deltas['control']}")
    bounded_pass = all(abs(x) <= TOLERANCE for x in deltas["request"]) and all(
        abs(r - c) <= TOLERANCE for c, r in zip(deltas["control"], deltas["request"]))
    result = dict(scope=("file_readonly_private" if file_backed else "anonymous") + "_nocopy_64MiB_three_cycles_5s_no_gpu",
                **{("bounded_task_accounting_recovery" if file_backed else "bounded_reclamation"): bounded_pass,
                   f"{metric}_deltas_bytes": deltas}, tolerance_bytes=TOLERANCE,
                model_safety_proven=False, system_reclamation_proven=False)
    if file_backed:
        # A no-residency hold does not qualify the detector after a residency
        # request. Record whether even a known-live requested mapping is seen.
        live = [b["prepared"]["resident"] - b["baseline"]["resident"] for b in request]
        result.update(requested_live_resident_deltas_bytes=live,
                      requested_live_resident_signal_detected=all(x >= SIGNAL for x in live),
                      residency_retention_verdict="unqualified")
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    file_backed = bool(args and args[0] == "--file")
    if file_backed:
        args = args[1:]
    if len(args) != 3:
        sys.exit("usage: check_metal_residency_footprint.py [--file] CONTROL_LOG REQUEST_LOG HOLD_LOG")
    try:
        modes = [("file-" if file_backed else "") + m for m in ("control", "request", "hold")]
        result = evaluate(*(read_run(p, m) for p, m in zip(args, modes)), file_backed=file_backed)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["bounded_task_accounting_recovery" if file_backed else "bounded_reclamation"] else 1)
    except (ValueError, KeyError, TypeError) as error:
        sys.exit(f"INCONCLUSIVE: {error}")
