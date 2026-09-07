#!/usr/bin/env python3
"""Summarize trusted headless CogniQwen logs, not model quality or GPU time."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PHASES = ("render_ms", "load_ms", "tokenize_ms", "cache_lookup_ms",
          "prefill_top1_ms", "decode_body_ms")


def summarize(text: str, wall_ms: float) -> dict:
    if not math.isfinite(wall_ms) or wall_ms <= 0:
        raise ValueError("session wall_ms must be finite and positive")
    calls = []
    waiting = False
    finished = False
    for raw in text.splitlines():
        line = ANSI.sub("", raw).strip()
        if line.startswith("💭 Calling LLM ("):
            if finished:
                raise ValueError("multiple sessions in one log")
            if waiting:
                raise ValueError("missing per-call performance record")
            waiting = True
        elif line.startswith("💭 CogniQwen perf: "):
            if not waiting:
                raise ValueError("unpaired per-call performance record")
            pairs = [part.split("=", 1) for part in line.split(": ", 1)[1].split(",")]
            fields = dict(pairs)
            if len(fields) != len(pairs):
                raise ValueError("duplicate performance field")
            missing = {"route", "total_ms", *PHASES} - fields.keys()
            if missing:
                raise ValueError("missing performance fields: " + ", ".join(sorted(missing)))
            call = {"route": fields["route"]}
            for key in ("total_ms", *PHASES):
                value = float(fields[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {key}")
                call[key] = value
            # greedy_ms is a parent of prefill/decode: never add it again.
            remainder = call["total_ms"] - sum(call[key] for key in PHASES)
            if remainder < -1.0:  # independently rounded 0.1 ms fields
                raise ValueError("phase times exceed provider wall; check nesting")
            call["provider_unattributed_ms"] = max(0.0, remainder)
            calls.append(call)
            waiting = False
        elif line == "✅ Agent finished":
            if finished or waiting or not calls:
                raise ValueError("invalid session termination")
            finished = True
        # The non-thought final summary repeats the last call; ignore it.
    if waiting or not calls or not finished:
        raise ValueError("incomplete or absent per-call performance records")
    totals = {key: sum(call[key] for call in calls)
              for key in ("total_ms", *PHASES, "provider_unattributed_ms")}
    remainder = wall_ms - totals["total_ms"]
    if remainder < -1.0:
        raise ValueError("provider totals exceed session wall")
    return {
        "schema": "qwen-coding-budget-v1", "call_count": len(calls),
        "calls": calls, "provider_totals_ms": totals, "session_wall_ms": wall_ms,
        "outside_provider_ms": max(0.0, remainder),
        "outside_provider_scope": "startup, tools, orchestration and uninstrumented work; not tool time alone",
        "quality": "not_inferred; requires external task tests",
        "clock": "host wall; not GPU utilization or kernel attribution",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wall-ms", type=float, required=True,
                        help="Externally measured session wall (declare any runner overhead), excluding later hidden tests")
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.log.read_text(), args.wall_ms), indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"budget unavailable: {exc}\n")
