"""Offline validation tests for the DRAMA-3 GDN scan analyzer."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "analyze_gdn_scan.py"
SPEC = importlib.util.spec_from_file_location("drama3_analyze_gdn_scan", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _metric(bit: float = 0.0) -> list[float]:
    return [bit, 0.0, 0.0, 0.0, 0.0]


def _fixture(depth: int = 0) -> tuple[dict, bytes]:
    source_hashes = {path: "2" * 64 for path in analysis.REQUIRED_SOURCE_KEYS}
    source_hashes["examples/support/drama3_gdn_scan.metal"] = "4" * 64
    controls = dict(analysis.EXPECTED_CONTROLS)
    controls["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = str(depth)
    report = {
        "schema": "drama3-gdn-scan-probe-v1",
        "source_head": "a" * 40,
        "executable_sha256": "1" * 64,
        "compiled_source_sha256": source_hashes,
        "source_tree_checked_before_and_after": True,
        "embedded_shader_sha256": source_hashes["examples/support/drama3_gdn_scan.metal"],
        "model_files_sha256": {"config.json": "5" * 64},
        "text": "test prompt",
        "prompt_ids": list(range(29)),
        "rows": 29,
        "layers": [0, 38],
        "geometry": [32, 128, 16, 128],
        "controls": controls,
        "graph_depth": depth,
        "comparator_qualification": {
            "clean": _metric(),
            "seeded": [1.0, 1.0, 0.001, 2.0, 1.0],
        },
        "capture_counts": {"0": 29, "38": 29},
        "capture_seconds": 1.0,
        "capture_encoding_added_reads_submits_waits": False,
        "groups": [],
        "graph": {
            "depth": depth,
            "submitted": 320,
            "completed": 320,
            "pending": 0,
        },
        "readbacks": {"diagnostic": 0, "control": 4},
        "activation_readbacks": 0,
        "production_routing_changed": False,
        "scope": "actual-input recurrence only; no full-prefill/WAV claim",
        "native_gate": analysis.NATIVE_GATE_TEXT,
        "tolerance_metric_only": analysis.TOLERANCE_GATE_TEXT,
    }
    timeline: list[dict] = []
    next_index = 0
    for layer in (0, 38):
        for base, rows in ((0, 29), (7, 19)):
            for block_size in dict.fromkeys((8, 16, rows)):
                blocks = (rows + block_size - 1) // block_size
                checks = {
                    "replay_vs_capture.output": _metric(),
                    "replay_vs_capture.state": _metric(),
                    "q_varies": _metric(1.0),
                    "initial_nonzero": _metric(1.0 if base else 0.0),
                }
                for mode in ("serial_tile", "affine_scan"):
                    checks.update({f"{mode}.output.{row}": _metric() for row in range(rows)})
                    checks[f"{mode}.state"] = _metric()
                checks.update({f"affine_scan.prefix.{block}": _metric() for block in range(blocks)})
                modes = {"affine_scan"} | ({"serial_tile"} if block_size == 8 else set())
                validation = {
                    "checks": checks,
                    "comparison_columns": list(analysis.METRIC_COLUMNS),
                    "serial_tile_native_gate_pass": True,
                    "affine_scan_native_gate_pass": True,
                    "affine_scan_tolerance_gate_pass": True,
                }
                measurements = {}
                for mode in sorted(modes):
                    pairs = []
                    for pair_index in range(10):
                        order = ["sequential", mode] if pair_index % 4 in (0, 3) else [mode, "sequential"]
                        command_indices = {}
                        for arm in order:
                            command_indices[arm] = next_index
                            duration = 2.0 if arm == "sequential" else 1.0
                            event = {"event": "drama3_gpu_command", "index": next_index, "gpu_seconds": duration}
                            if depth == 2:
                                event.update({"completion_status": 0, "gpu_timing_valid": True})
                            timeline.append(event)
                            next_index += 1
                        pairs.append(
                            {
                                "pair": pair_index,
                                "warmup": pair_index < 2,
                                "mode": mode,
                                "order": order,
                                "command_indices": command_indices,
                                "host_encode_seconds": {"sequential": 0.2, mode: 0.1},
                                "checks": {"output": _metric(), "state": _metric()},
                                "native_gate_pass": True,
                                "graph": {"depth": depth},
                                "readbacks": {"diagnostic": 0},
                                "timing_source": "drama3_gpu_command events",
                            }
                        )
                    measurements[mode] = pairs
                report["groups"].append(
                    {
                        "layer": layer,
                        "base": base,
                        "rows": rows,
                        "block_size": block_size,
                        "blocks": blocks,
                        "ragged_tail": rows % block_size,
                        "scratch_bytes": (2 * blocks * 32 * 128 * 256 + blocks * 32 * 128 * 128) * 4,
                        "validation": validation,
                        "measurements": measurements,
                    }
                )
    report["graph"]["submitted"] = next_index
    report["graph"]["completed"] = next_index
    log_lines = ["[PREFLIGHT] example preserved stderr noise"]
    log_lines.extend(json.dumps(event, separators=(",", ":")) for event in timeline)
    log_lines.append(json.dumps({"event": "gdn_scan_complete", "groups": 12}))
    return report, ("\n".join(log_lines) + "\n").encode()


class GdnScanAnalysisSpec(unittest.TestCase):
    def test_inconsistent_finite_receipts_cannot_pass_the_bit_gate(self):
        for impossible in ([0.0, 0.0, 0.001, 0.1, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]):
            with self.subTest(receipt=impossible):
                report, log = _fixture()
                pair = report["groups"][0]["measurements"]["serial_tile"][0]
                pair["checks"]["output"] = impossible
                with self.assertRaisesRegex(analysis.ValidationError, "contradict the bit mismatch count"):
                    analysis.analyze(json.dumps(report).encode(), log)

    def test_complete_depth0_capture_summarizes_only_measured_pairs(self):
        report, log = _fixture()
        result = analysis.analyze(json.dumps(report).encode(), log)
        self.assertEqual(len(result["groups"]), 12)
        cohorts = [
            (group, mode, summary)
            for group in result["groups"]
            for mode, summary in group["modes"].items()
        ]
        self.assertEqual(len(cohorts), 16)
        for _, _, summary in cohorts:
            self.assertEqual(summary["measured_pairs"], 8)
            self.assertEqual(summary["warmups_excluded"], 2)
            self.assertEqual(
                summary["paired_gpu_speedup_ratio_sequential_over_candidate"],
                {"median": 2.0, "min": 2.0, "max": 2.0},
            )
            self.assertEqual(summary["validation_drift"]["final_state"]["bit_mismatches"], 0)
        self.assertIn("no full-prefill or WAV", result["interpretation"]["claim_scope"])
        self.assertEqual(result["digests"]["report_sha256"], analysis._sha256_bytes(json.dumps(report).encode()))
        self.assertEqual(result["digests"]["stderr_log_sha256"], analysis._sha256_bytes(log))

    def test_depth2_requires_valid_timeline_and_marks_intervals_nonadditive(self):
        report, log = _fixture(depth=2)
        result = analysis.analyze(json.dumps(report).encode(), log)
        self.assertIn("non-additive", result["interpretation"]["gpu_intervals"])
        bad = log.replace(b'"gpu_timing_valid":true', b'"gpu_timing_valid":false', 1)
        with self.assertRaisesRegex(analysis.ValidationError, "invalid GPU timing"):
            analysis.analyze(json.dumps(report).encode(), bad)

    def test_missing_timing_duplicate_timing_and_reused_index_are_rejected(self):
        report, log = _fixture()
        events = [line for line in log.splitlines() if b'"event":"drama3_gpu_command"' in line]
        with self.assertRaisesRegex(analysis.ValidationError, "missing timing"):
            analysis.analyze(json.dumps(report).encode(), b"\n".join(events[1:]))
        with self.assertRaisesRegex(analysis.ValidationError, "duplicate drama3_gpu_command index"):
            analysis.analyze(json.dumps(report).encode(), log + events[0] + b"\n")
        reused = copy.deepcopy(report)
        first = reused["groups"][0]["measurements"]["affine_scan"][0]["command_indices"]
        second = reused["groups"][0]["measurements"]["affine_scan"][1]["command_indices"]
        second["sequential"] = first["sequential"]
        with self.assertRaisesRegex(analysis.ValidationError, "reused"):
            analysis.analyze(json.dumps(reused).encode(), log)

    def test_partial_group_and_output_receipts_are_rejected(self):
        report, log = _fixture()
        partial_grid = copy.deepcopy(report)
        partial_grid["groups"].pop()
        with self.assertRaisesRegex(analysis.ValidationError, "complete pinned 12-group grid"):
            analysis.analyze(json.dumps(partial_grid).encode(), log)
        partial_output = copy.deepcopy(report)
        del partial_output["groups"][0]["validation"]["checks"]["affine_scan.output.0"]
        with self.assertRaisesRegex(analysis.ValidationError, "missing/extra coverage"):
            analysis.analyze(json.dumps(partial_output).encode(), log)
        partial_pair = copy.deepcopy(report)
        del partial_pair["groups"][0]["measurements"]["affine_scan"][0]["checks"]["output"]
        with self.assertRaisesRegex(analysis.ValidationError, "full output/state receipts"):
            analysis.analyze(json.dumps(partial_pair).encode(), log)

    def test_forged_gate_nan_receipt_and_bad_abba_order_are_rejected(self):
        report, log = _fixture()
        forged = copy.deepcopy(report)
        forged["groups"][0]["validation"]["affine_scan_native_gate_pass"] = False
        with self.assertRaisesRegex(analysis.ValidationError, "forged validation gate"):
            analysis.analyze(json.dumps(forged).encode(), log)
        nonfinite = copy.deepcopy(report)
        nonfinite["groups"][0]["validation"]["checks"]["affine_scan.output.0"][0] = float("nan")
        with self.assertRaisesRegex(analysis.ValidationError, "non-finite JSON constant"):
            analysis.analyze(json.dumps(nonfinite).encode(), log)
        reordered = copy.deepcopy(report)
        pair = reordered["groups"][0]["measurements"]["affine_scan"][2]
        pair["order"] = list(reversed(pair["order"]))
        with self.assertRaisesRegex(analysis.ValidationError, "ABBA command order"):
            analysis.analyze(json.dumps(reordered).encode(), log)
        misindexed = copy.deepcopy(report)
        pair = misindexed["groups"][0]["measurements"]["affine_scan"][2]
        pair["command_indices"]["sequential"], pair["command_indices"]["affine_scan"] = (
            pair["command_indices"]["affine_scan"], pair["command_indices"]["sequential"]
        )
        with self.assertRaisesRegex(analysis.ValidationError, "indices contradict"):
            analysis.analyze(json.dumps(misindexed).encode(), log)

    def test_missing_composed_prefix_coverage_is_rejected(self):
        report, log = _fixture()
        del report["groups"][0]["validation"]["checks"]["affine_scan.prefix.0"]
        with self.assertRaisesRegex(analysis.ValidationError, "missing/extra coverage"):
            analysis.analyze(json.dumps(report).encode(), log)

    def test_native_rejection_remains_a_measurement_separate_from_tolerance(self):
        report, log = _fixture()
        group = report["groups"][0]
        group["validation"]["checks"]["affine_scan.output.0"] = [1.0, 0.0, 0.000001, 0.1, 0.0]
        group["validation"]["affine_scan_native_gate_pass"] = False
        self.assertTrue(group["validation"]["affine_scan_tolerance_gate_pass"])
        summary = analysis.analyze(json.dumps(report).encode(), log)["groups"][0]["modes"]["affine_scan"]
        self.assertFalse(summary["native_bit_gate_pass"])
        self.assertTrue(summary["tolerance_gate_pass"])

    def test_all_pair_receipts_including_warmups_contribute_to_final_gates(self):
        report, log = _fixture()
        pair = report["groups"][0]["measurements"]["affine_scan"][0]
        pair["checks"]["output"] = [1.0, 0.0, 0.000001, 0.1, 0.0]
        pair["native_gate_pass"] = False
        summary = analysis.analyze(json.dumps(report).encode(), log)["groups"][0]["modes"]["affine_scan"]
        self.assertTrue(summary["validation_native_bit_gate_pass"])
        self.assertFalse(summary["timed_pair_receipts_native_gate_pass"])
        self.assertFalse(summary["native_bit_gate_pass"])
        self.assertTrue(summary["tolerance_gate_pass"])
        self.assertEqual(summary["all_pair_output_drift_including_warmups"]["bit_mismatches"], 1)

        report, log = _fixture()
        pair = report["groups"][0]["measurements"]["affine_scan"][1]
        pair["checks"]["state"] = [1.0, 1.0, 0.0001, 1.1, 0.0]
        pair["native_gate_pass"] = False
        summary = analysis.analyze(json.dumps(report).encode(), log)["groups"][0]["modes"]["affine_scan"]
        self.assertTrue(summary["validation_tolerance_gate_pass"])
        self.assertFalse(summary["timed_pair_receipts_tolerance_gate_pass"])
        self.assertFalse(summary["native_bit_gate_pass"])
        self.assertFalse(summary["tolerance_gate_pass"])
        self.assertEqual(summary["all_pair_final_state_drift_including_warmups"]["tolerance_failures"], 1)

    def test_probe_controls_gate_wording_and_shader_binding_are_pinned(self):
        cases = (
            ({"controls": {"DRAMA3_BACKEND": "metal"}}, "controls differ"),
            ({"native_gate": "different words"}, "native gate wording"),
            ({"tolerance_metric_only": "different words"}, "tolerance gate wording"),
            ({"embedded_shader_sha256": "9" * 64}, "embedded_shader_sha256 differs"),
        )
        for change, error in cases:
            with self.subTest(error=error):
                report, log = _fixture()
                report.update(change)
                with self.assertRaisesRegex(analysis.ValidationError, error):
                    analysis.analyze(json.dumps(report).encode(), log)
        report, log = _fixture()
        del report["compiled_source_sha256"]["examples/support/drama3_gdn_scan.metal"]
        with self.assertRaisesRegex(analysis.ValidationError, "lacks required keys"):
            analysis.analyze(json.dumps(report).encode(), log)


if __name__ == "__main__":
    unittest.main()
