"""CPU-only fail-closed tests for the text-compaction report analyzer."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "support" / "hy_nomas_text_compaction_analysis.py"
SPEC = importlib.util.spec_from_file_location("hy_nomas_text_compaction_analysis", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _telemetry(total: float = 5.0) -> dict:
    wait = 0.25
    return {
        "encode_seconds": total - wait,
        "terminal_wait_seconds": wait,
        "total_seconds": total,
        "gpu_elapsed_seconds": total - 0.1,
        "command_segments": 1244,
        "named_allocations_during_encode": 0,
        "activation_readbacks_before_terminal_wait": 0,
        "terminal_wait_count": 1,
        "diagnostic_readbacks": 0,
    }


def _guard_events() -> list[dict]:
    physical = 64 * 1024**3
    footprint_cap = 24 * 1024**3
    return [
        {
            "event": "limits",
            "monotonic": 1.0,
            "runtime_floor_pct": 10,
            "runtime_floor_pct_exact": "10",
            "max_footprint_bytes": footprint_cap,
            "max_swap_growth_bytes": 64 * 1024**2,
            "timeout_sec": 900,
            "memory_accounting": "file-cache",
        },
        {
            "event": "sample",
            "monotonic": 2.0,
            "pressure_free_pct": 55,
            "physical_bytes": physical,
            "headroom_bytes": physical // 2,
            "group_footprint_bytes": 0,
            "group_live_processes": 0,
            "swap_used_bytes": 4 * 1024**3,
            "file_backed_bytes": 20 * 1024**3,
            "free_spec_bytes": 14 * 1024**3,
            "speculative_bytes": 100 * 1024**2,
        },
        {"event": "started", "monotonic": 2.5, "pid": 1234, "pgid": 1234},
        {
            "event": "sample",
            "monotonic": 3.0,
            "pressure_free_pct": 48,
            "physical_bytes": physical,
            "headroom_bytes": physical // 3,
            "group_footprint_bytes": 18 * 1024**3,
            "group_live_processes": 1,
            "swap_used_bytes": 4 * 1024**3 + 4 * 1024**2,
            "file_backed_bytes": 18 * 1024**3,
            "free_spec_bytes": 4 * 1024**3,
            "speculative_bytes": 100 * 1024**2,
        },
        {"event": "finished", "monotonic": 4.0, "returncode": 0},
    ]


def _report() -> dict:
    validations = []
    for mode in analysis.MODES:
        hashes: dict[str, str] = {}
        for request in (0, 1):
            comparisons = []
            for name in analysis.CAPTURE_ELEMENTS:
                value = _digest(f"{mode}:{name}")
                hashes[name] = value
                comparisons.append({
                    "name": name,
                    "elements": analysis._expected_elements(
                        mode, name, analysis.GEOMETRY["valid_text_tokens"]
                    ),
                    "failures": 0,
                    "raw_bit_differences": 0,
                    "max_abs": 0.0,
                    "finite": True,
                    "native_sha256": value,
                    "first_request_sha256_match": None if request == 0 else True,
                })
            validations.append({
                "mode": mode,
                "request": request,
                "passed": True,
                "scratch_poisoned": request == 1,
                "poison_buffer_count": 214 if request == 1 else 0,
                "named_allocations_during_encode": 0,
                "activation_readbacks_before_terminal_wait": 0,
                "terminal_wait_count": 1,
                "excluded_suffix_poisoned": mode == "compact" and request == 1,
                "comparisons": comparisons,
            })

    warmup_orders = {0: ["padded", "compact"], 1: ["compact", "padded"]}
    epochs = [
        {"epoch": epoch, "warmup_order": warmup_orders[epoch], "pairs": 3}
        for epoch in (0, 1)
    ]
    warmups = []
    for epoch in (0, 1):
        for position, mode in enumerate(warmup_orders[epoch]):
            warmups.append({
                "epoch": epoch,
                "position": position,
                "mode": mode,
                **_telemetry(5.0 if mode == "padded" else 4.8),
            })

    samples = []
    for epoch in (0, 1):
        for pair, pair_order in enumerate(analysis.SAMPLE_ORDERS[epoch]):
            for position, mode in enumerate(pair_order):
                base = 5.0 if mode == "padded" else 4.8
                samples.append({
                    "epoch": epoch,
                    "pair": pair,
                    "position": position,
                    "mode": mode,
                    **_telemetry(base + pair * 0.01),
                })

    return {
        "schema": analysis.SCHEMA,
        "schema_version": analysis.SCHEMA_VERSION,
        "reference_sha256": analysis.REFERENCE_SHA256,
        "cache_index_sha256": analysis.CACHE_INDEX_SHA256,
        "precision_mode": "q4",
        "geometry": {**copy.deepcopy(analysis.GEOMETRY), "compact_text_tokens": 64},
        "tolerance": {"atol": 2e-5, "rtol": 2e-5},
        "diffusion_generation": False,
        "modes": {
            "padded": {"named_bytes": 10_895_796_804, "dit_text_tokens": 1280},
            "compact": {"named_bytes": 10_500_000_000, "dit_text_tokens": 64},
        },
        "max_combined_named_bytes": 24 * 1024**3,
        "passed": True,
        "validations": validations,
        "epochs": epochs,
        "warmups": warmups,
        "samples": samples,
    }


class TextCompactionAnalysisSpec(unittest.TestCase):
    def test_summarizes_fixed_input_compaction_and_balanced_pairs(self):
        result = analysis.analyze(_report(), _guard_events())
        self.assertEqual(result["checks"]["validations"], 4)
        self.assertEqual(result["checks"]["measured_samples"], 12)
        self.assertGreater(result["representations"]["named_bytes_saved"], 0)
        self.assertEqual(result["timing"]["measured_pairs"], 6)
        self.assertEqual(result["guard"]["runtime_floor_pct"], 10)
        self.assertEqual(result["geometry"]["valid_text_tokens"], 23)
        self.assertEqual(result["geometry"]["compact_text_tokens"], 64)
        self.assertEqual(result["representations"]["compact"]["dit_text_tokens"], 64)
        self.assertEqual(result["representations"]["compact"]["valid_text_tokens"], 23)
        self.assertEqual(result["oracle_gate"],
                         "producer-declared per-element comparisons passed; this CPU analyzer validates receipts, not tensor bytes")
        self.assertTrue(result["passed"])

    def test_rejects_stale_geometry(self):
        report = _report()
        report["geometry"]["video_tokens"] -= 1
        with self.assertRaisesRegex(analysis.ValidationError, "geometry"):
            analysis.analyze(report, _guard_events())

        report = _report()
        report["schema_version"] = True
        with self.assertRaisesRegex(analysis.ValidationError, "schema/version"):
            analysis.analyze(report, _guard_events())

    def test_rejects_mismatched_fixture_cache_or_representation(self):
        mutations = (
            ("reference_sha256", "0" * 64, "upstream fixture"),
            ("cache_index_sha256", "1" * 64, "cache index"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                report = _report()
                report[field] = value
                with self.assertRaisesRegex(analysis.ValidationError, message):
                    analysis.analyze(report, _guard_events())

        report = _report()
        report["modes"]["compact"]["dit_text_tokens"] = 32
        with self.assertRaisesRegex(analysis.ValidationError, "representation mismatch"):
            analysis.analyze(report, _guard_events())

        report = _report()
        report["modes"]["padded"]["named_bytes"] = report["max_combined_named_bytes"]
        with self.assertRaisesRegex(analysis.ValidationError, "combined named residency"):
            analysis.analyze(report, _guard_events())

    def test_accepts_bound_aligned_compact_storage_but_keeps_23_row_oracle(self):
        report = _report()
        report["geometry"]["compact_text_tokens"] = 96
        report["modes"]["compact"]["dit_text_tokens"] = 96
        result = analysis.analyze(report, _guard_events())
        self.assertEqual(result["geometry"]["compact_text_tokens"], 96)
        self.assertEqual(result["representations"]["compact"]["valid_text_tokens"], 23)
        compact_text_comparison = next(
            row for validation in report["validations"]
            if validation["mode"] == "compact" and validation["request"] == 0
            for row in validation["comparisons"]
            if row["name"] == "block0.txt.output"
        )
        self.assertEqual(compact_text_comparison["elements"], 23 * 2048)

        mismatched = _report()
        mismatched["geometry"]["compact_text_tokens"] = 96
        with self.assertRaisesRegex(analysis.ValidationError, "representation mismatch"):
            analysis.analyze(mismatched, _guard_events())

    def test_rejects_out_of_bounds_or_unaligned_compact_storage(self):
        for compact_tokens, message in (
            (22, "geometry.compact_text_tokens"),
            (65, "32-token aligned"),
            (1280, "below padded geometry"),
        ):
            with self.subTest(compact_tokens=compact_tokens):
                report = _report()
                report["geometry"]["compact_text_tokens"] = compact_tokens
                report["modes"]["compact"]["dit_text_tokens"] = compact_tokens
                with self.assertRaisesRegex(analysis.ValidationError, message):
                    analysis.analyze(report, _guard_events())

    def test_rejects_nonfinite_or_failed_comparison(self):
        for mutation in (
            lambda row: row.update(finite=False, max_abs=None),
            lambda row: row.update(failures=1),
            lambda row: row.update(max_abs=float("nan")),
        ):
            report = _report()
            mutation(report["validations"][0]["comparisons"][0])
            with self.subTest(mutation=mutation), self.assertRaises(analysis.ValidationError):
                analysis.analyze(report, _guard_events())

    def test_rejects_unbalanced_or_noninterleaved_pair_order(self):
        report = _report()
        report["samples"][2]["mode"] = "padded"
        with self.assertRaisesRegex(analysis.ValidationError, "interleaved AB/BA"):
            analysis.analyze(report, _guard_events())

        report = _report()
        report["epochs"][1]["warmup_order"] = ["padded", "compact"]
        with self.assertRaisesRegex(analysis.ValidationError, "counterbalance"):
            analysis.analyze(report, _guard_events())

    def test_total_wall_time_allows_bookkeeping_gap_but_not_undercount(self):
        report = _report()
        report["samples"][0]["total_seconds"] += 0.05
        self.assertTrue(analysis.analyze(report, _guard_events())["passed"])

        report = _report()
        report["samples"][0]["total_seconds"] -= 0.01
        with self.assertRaisesRegex(analysis.ValidationError, "underreports"):
            analysis.analyze(report, _guard_events())

    def test_rejects_output_drift_after_poisoned_repeat(self):
        report = _report()
        repeat = report["validations"][1]["comparisons"][0]
        repeat["native_sha256"] = _digest("different result")
        with self.assertRaisesRegex(analysis.ValidationError, "drifted"):
            analysis.analyze(report, _guard_events())

    def test_rejects_added_sync_or_allocation(self):
        report = _report()
        report["samples"][0]["terminal_wait_count"] = 2
        with self.assertRaisesRegex(analysis.ValidationError, "exactly one terminal wait"):
            analysis.analyze(report, _guard_events())

        report = _report()
        report["samples"][0]["named_allocations_during_encode"] = 1
        with self.assertRaisesRegex(analysis.ValidationError, "allocation/readback boundary"):
            analysis.analyze(report, _guard_events())

    def test_rejects_missing_or_failed_memory_guard_evidence(self):
        with self.assertRaisesRegex(analysis.ValidationError, "guard log is missing"):
            analysis.analyze(_report(), [])

        events = _guard_events()
        del events[1]["headroom_bytes"]
        with self.assertRaisesRegex(analysis.ValidationError, "headroom_bytes"):
            analysis.analyze(_report(), events)

        events = _guard_events()
        events[3]["pressure_free_pct"] = 9
        with self.assertRaisesRegex(analysis.ValidationError, "10 percent memory floor"):
            analysis.analyze(_report(), events)

        events = _guard_events()
        events[-1]["returncode"] = 1
        with self.assertRaisesRegex(analysis.ValidationError, "did not exit successfully"):
            analysis.analyze(_report(), events)

        for field, value, message in (
            ("max_footprint_bytes", 24 * 1024**3 + 1, "24 GiB paired-run limit"),
            ("timeout_sec", 0, "timeout_sec"),
            ("timeout_sec", 901, "900-second run limit"),
        ):
            with self.subTest(field=field, value=value):
                events = _guard_events()
                events[0][field] = value
                with self.assertRaisesRegex(analysis.ValidationError, message):
                    analysis.analyze(_report(), events)

    def test_cli_parser_rejects_nonfinite_json_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            guard_path = Path(directory) / "guard.jsonl"
            report = _report()
            report["samples"][0]["encode_seconds"] = float("nan")
            report_path.write_text(json.dumps(report, allow_nan=True), encoding="utf-8")
            guard_path.write_text("\n".join(json.dumps(row) for row in _guard_events()), encoding="utf-8")
            with self.assertRaisesRegex(analysis.ValidationError, "nonfinite JSON number NaN"):
                analysis.analyze_files(report_path, guard_path)


if __name__ == "__main__":
    unittest.main()
