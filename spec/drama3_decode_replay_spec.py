#!/usr/bin/env python3
"""Contract tests for the offline dense decode replay analyzer."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/drama3/analyze_decode_replay.py"
SPEC = importlib.util.spec_from_file_location("analyze_decode_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


FRAMES = [0, 29, 59]
FAMILIES = ["backbone.experts.down", "main.head"]
CELLS = [
    (0, "backbone.experts.down"),
    (29, "backbone.experts.down"),
    (29, "main.head"),
    (59, "backbone.experts.down"),
    (59, "main.head"),
]
VALUES = {
    CELLS[0]: 5,
    CELLS[1]: 7,
    CELLS[2]: 2,
    CELLS[3]: 11,
    CELLS[4]: 2,
}
BASE = 120
WARMUPS = 1
PAIRS = 3
ROUNDS = WARMUPS + PAIRS
PAIR_COUNT = len(CELLS) * ROUNDS
REPLAY_END = BASE + PAIR_COUNT * 3


def graph(submitted: int) -> dict:
    return {
        "depth": 2,
        "submitted": submitted,
        "completed": submitted,
        "completion_waits": submitted,
        "pending": 0,
        "retained_bytes": 0,
        "max_pending": 2,
        "max_retained_bytes": 8192,
    }


def readbacks(terminal: int) -> dict:
    return {"control": 496, "terminal": terminal, "diagnostic": 0}


def gpu_command(index: int, seconds: float = 0.001) -> dict:
    return {
        "event": "drama3_gpu_command",
        "index": index,
        "completion_status": 0,
        "gpu_seconds": seconds,
        "gpu_timing_valid": True,
    }


def fixture_objects() -> tuple[list[dict], list[dict]]:
    initial_terminal = 494_160
    event_rows = [
        {
            "event": "capture",
            "manifest": "/private/path/capture.json",
            "generation": {"frame_count": 60},
            "graph": graph(BASE),
            "readbacks": readbacks(initial_terminal),
            "wav": {
                "sha256": "a" * 64,
                "expected_sha256": "a" * 64,
                "sample_rate": 44_100,
                "samples": 100,
            },
            "capture_summary": {
                "scope": "dense_expert_projections_only",
                "records": len(CELLS),
                "frames": FRAMES,
                "families": FAMILIES,
                "activation_readbacks": 0,
                "full_decode_coverage_claimed": False,
            },
            "capture_records": len(CELLS),
        },
        {
            "event": "decode_replay_invocation",
            "format": 1,
            "scope": "dense_expert_projections_only",
            "pairs": PAIRS,
            "warmups": WARMUPS,
            "repeats": 1,
            "frames": FRAMES,
            "families": FAMILIES,
            "omitted_cells": [[0, "main.head"]],
            "records": len(CELLS),
            "covered_values": sum(VALUES.values()),
        },
    ]
    timeline_rows = [gpu_command(BASE - 1, 0.02)]
    terminal = initial_terminal
    ordinal = 0
    value_total = 0
    for pair_index in range(ROUNDS):
        schedule = CELLS if pair_index % 2 == 0 else list(reversed(CELLS))
        for frame, family in schedule:
            cell = (frame, family)
            negative = pair_index == 0 and cell == CELLS[0]
            receipt = 16 if negative else 8
            before_index = BASE + ordinal * 3
            order = ["a", "b"] if pair_index % 2 == 0 else ["b", "a"]
            indices = {order[0]: before_index, order[1]: before_index + 1}
            terminal_after = terminal + receipt
            event_rows.append({
                "event": "decode_replay_pair",
                "frame": frame,
                "family": family,
                "pair": pair_index,
                "warmup": pair_index < WARMUPS,
                "repeats": 1,
                "order": order,
                "command_indices": indices,
                "host_encode_seconds": {"a": 0.0004, "b": 0.0005},
                "records": 1,
                "values_per_arm": VALUES[cell],
                "bit_mismatches": 0,
                "nonfinite_pairs": 0,
                "negative_control": [2, 1] if negative else None,
                "terminal_receipt_bytes": receipt,
                "graph_before": graph(before_index),
                "graph_after": graph(before_index + 3),
                "readbacks_before": readbacks(terminal),
                "readbacks_after": readbacks(terminal_after),
            })
            for command_offset in range(3):
                timeline_rows.append(gpu_command(before_index + command_offset,
                                                 0.001 + CELLS.index(cell) * 0.001 + command_offset * 0.0001))
            terminal = terminal_after
            value_total += VALUES[cell]
            ordinal += 1
    event_rows.append({
        "event": "decode_replay_summary",
        "scope": "dense_expert_projections_only",
        "pairs_completed": PAIR_COUNT,
        "verified_values_per_arm": value_total,
        "graph": graph(REPLAY_END),
        "readbacks": readbacks(terminal),
        "activation_readbacks": 0,
        "full_decode_coverage_claimed": False,
    })
    timeline_rows.append(gpu_command(REPLAY_END, 0.03))
    return event_rows, timeline_rows


def as_jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(row, allow_nan=True) for row in rows) + "\n"


def fixture() -> tuple[list[dict], list[dict]]:
    events, timeline = fixture_objects()
    return copy.deepcopy(events), copy.deepcopy(timeline)


def run(events: list[dict], timeline: list[dict]) -> dict:
    return analyzer.analyze(as_jsonl(events), as_jsonl(timeline))


class DecodeReplayAnalyzerSpec(unittest.TestCase):
    def test_happy_fixture_reports_local_paired_timings_and_uncosted_setup(self):
        events, timeline = fixture()
        report = run(events, timeline)
        self.assertEqual(report["pair_count"], PAIR_COUNT)
        self.assertEqual(report["scope"], "dense_expert_projections_only")
        self.assertFalse(report["claim_boundary"]["request_cost_additivity_assumed"])
        self.assertFalse(report["claim_boundary"]["legacy_profile_subtraction_performed"])
        self.assertEqual([row["index"] for row in report["uncosted_setup_commands"]], [BASE - 1, REPLAY_END])
        self.assertEqual(len(report["cells"]), len(CELLS))
        self.assertTrue(all(row["local_rank_within_frame"] >= 1 for row in report["cells"]))
        first = report["cells"][0]
        self.assertEqual(first["measured_pairs"], PAIRS)
        self.assertEqual(first["gpu_seconds_per_repeat"]["arm_a"]["count"], PAIRS)
        self.assertIn("relative_fraction", first["identical_arm_noise"])

    def test_local_rank_one_is_highest_gpu_cost(self):
        events, timeline = fixture()
        report = run(events, timeline)
        frame_29 = {row["family"]: row for row in report["cells"] if row["frame"] == 29}
        self.assertGreater(frame_29["main.head"]["paired_arm_mean_median_seconds"],
                           frame_29["backbone.experts.down"]["paired_arm_mean_median_seconds"])
        self.assertEqual(frame_29["main.head"]["local_rank_within_frame"], 1)
        self.assertEqual(frame_29["backbone.experts.down"]["local_rank_within_frame"], 2)
        self.assertIn("rank 1 is highest paired median GPU cost",
                      report["claim_boundary"]["ranking"])

    def test_missing_replay_command_fails_closed(self):
        events, timeline = fixture()
        timeline = [row for row in timeline if row.get("index") != BASE]
        with self.assertRaisesRegex(analyzer.ReplayError, "missing GPU command"):
            run(events, timeline)

    def test_duplicate_command_index_fails_closed(self):
        events, timeline = fixture()
        timeline.append(copy.deepcopy(timeline[1]))
        with self.assertRaisesRegex(analyzer.ReplayError, "duplicate GPU command index"):
            run(events, timeline)

    def test_wrong_frame_breaks_rectangular_schedule(self):
        events, timeline = fixture()
        pair = next(row for row in events if row.get("event") == "decode_replay_pair")
        pair["frame"] = 29
        with self.assertRaisesRegex(analyzer.ReplayError, "frame/family does not match"):
            run(events, timeline)

    def test_nonfinite_gpu_duration_is_rejected(self):
        events, timeline = fixture()
        replay = next(row for row in timeline if row.get("index") == BASE)
        replay["gpu_seconds"] = math.nan
        with self.assertRaisesRegex(analyzer.ReplayError, "finite and positive"):
            run(events, timeline)

    def test_invalid_gpu_timing_flag_is_rejected(self):
        events, timeline = fixture()
        replay = next(row for row in timeline if row.get("index") == BASE)
        replay["gpu_timing_valid"] = False
        with self.assertRaisesRegex(analyzer.ReplayError, "GPU timing is not valid"):
            run(events, timeline)

    def test_malformed_seeded_negative_control_is_rejected(self):
        events, timeline = fixture()
        first = next(row for row in events if row.get("event") == "decode_replay_pair")
        first["negative_control"] = [2, 0]
        with self.assertRaisesRegex(analyzer.ReplayError, r"exactly \[2,1\]"):
            run(events, timeline)

    def test_graph_counter_regression_is_rejected(self):
        events, timeline = fixture()
        first = next(row for row in events if row.get("event") == "decode_replay_pair")
        first["graph_after"]["submitted"] -= 1
        with self.assertRaisesRegex(analyzer.ReplayError, "submitted did not advance by three"):
            run(events, timeline)

    def test_readback_regression_is_rejected(self):
        events, timeline = fixture()
        first = next(row for row in events if row.get("event") == "decode_replay_pair")
        first["readbacks_after"]["control"] += 4
        with self.assertRaisesRegex(analyzer.ReplayError, "unexpected control readback delta"):
            run(events, timeline)

    def test_readback_snapshots_must_continue_across_pairs(self):
        events, timeline = fixture()
        pair_events = [row for row in events if row.get("event") == "decode_replay_pair"]
        pair_events[1]["readbacks_before"]["control"] += 4
        pair_events[1]["readbacks_after"]["control"] += 4
        with self.assertRaisesRegex(analyzer.ReplayError, "readback snapshots are discontinuous"):
            run(events, timeline)

    def test_extra_invocation_is_rejected(self):
        events, timeline = fixture()
        invocation = next(row for row in events if row.get("event") == "decode_replay_invocation")
        events.append(copy.deepcopy(invocation))
        with self.assertRaisesRegex(analyzer.ReplayError, "exactly one invocation"):
            run(events, timeline)

    def test_missing_pair_event_is_rejected(self):
        events, timeline = fixture()
        index = next(i for i, row in enumerate(events) if row.get("event") == "decode_replay_pair")
        del events[index]
        with self.assertRaisesRegex(analyzer.ReplayError, "expected .* pair events"):
            run(events, timeline)


if __name__ == "__main__":
    unittest.main()
