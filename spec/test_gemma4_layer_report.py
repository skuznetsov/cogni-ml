"""Tests for the standalone Gemma 4 decoder-depth report renderer."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


REPORT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gemma4_layer_report.py"
REPORT_SPEC = importlib.util.spec_from_file_location("gemma4_layer_report", REPORT_SCRIPT)
assert REPORT_SPEC is not None and REPORT_SPEC.loader is not None
layer_report = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(layer_report)

SCHEMA = "cogni.gemma4.layer_depth.v1"
PROMPT_HASH = "a" * 64
MODEL_HASH = "b" * 64


def power_measurement(
    kind: str = "active_power",
    unit: str = "W",
    *,
    mean: float = 104.0,
    energy: float = 68.0,
) -> dict:
    current = kind == "current_ac"
    return {
        "status": "available",
        "kind": kind,
        "power_unit": unit,
        "duration_seconds": 1.0,
        "sample_count": 3,
        "mean_active_power_watts": None if current else mean,
        "active_energy_joules": None if current else energy,
        "mean_apparent_power_va": mean if current else None,
        "apparent_energy_va_seconds": energy if current else None,
        "estimated_active_energy_joules": None,
        "assumed_voltage_volts": None,
        "assumed_power_factor": None,
        "curve": [
            {"elapsed_ms": 0, "value": mean - 4},
            {"elapsed_ms": 500, "value": mean},
            {"elapsed_ms": 1000, "value": mean + 4},
        ],
    }


def utilization_measurement(mean_percent: float = 72.5) -> dict:
    return {
        "status": "available",
        "mean_percent": mean_percent,
        "duration_seconds": 1.0,
        "sample_count": 3,
        "curve": [
            {"elapsed_ms": 0, "value": mean_percent - 2},
            {"elapsed_ms": 500, "value": mean_percent},
            {"elapsed_ms": 1000, "value": mean_percent + 2},
        ],
    }


def run_record(
    depth: int,
    label: str | None = None,
    *,
    trial: int = 1,
    status: str = "completed",
    actual_tokens: int | None = 256,
    requested_tokens: int | None = 256,
    elapsed_ms: int | None = 1000,
    gpu_measurement: dict | None = None,
    circuit_measurement: dict | None = None,
    gpu_utilization: dict | None = None,
    answer_preview: str | None = "A short synthetic answer.",
    error: str | None = None,
) -> dict:
    return {
        "label": label or f"depth{depth}-trial{trial}",
        "decoder_layers": depth,
        "trial": trial,
        "status": status,
        "requested_tokens": requested_tokens,
        "started_at_epoch_ms": 1000,
        "ended_at_epoch_ms": 1000 + (elapsed_ms or 0),
        "elapsed_ms": elapsed_ms,
        "actual_tokens": actual_tokens,
        "model_identity": "gemma-4-test",
        "model_sha256": MODEL_HASH,
        "parameter_count": 25_000_000_000,
        "weight_bytes": 14_000_000_000,
        "answer_preview": answer_preview,
        "error": error,
        "gpu_measurement": gpu_measurement,
        "circuit_measurement": circuit_measurement,
        "gpu_utilization_measurement": gpu_utilization,
    }


def report_with(*runs: dict, requested_tokens: int = 256) -> dict:
    return {
        "schema": SCHEMA,
        "campaign_id": "gemma4-depth-fixed-256",
        "prompt_sha256": PROMPT_HASH,
        "requested_tokens": requested_tokens,
        "notes": "Fixed generation with ignore_eos.",
        "runs": list(runs),
    }


class LayerReportTests(unittest.TestCase):
    def test_fixed_count_gate_and_report_caveats_are_explicit(self) -> None:
        report = report_with(
            run_record(48, gpu_measurement=power_measurement(), circuit_measurement=power_measurement("current_ac", "VA")),
            run_record(32, gpu_measurement=power_measurement(mean=90, energy=59), circuit_measurement=power_measurement("current_ac", "VA", mean=158, energy=101)),
            run_record(16, gpu_measurement=power_measurement(mean=78, energy=50), circuit_measurement=power_measurement("current_ac", "VA", mean=145, energy=92)),
        )
        rendered = layer_report.render_html(report)

        self.assertIn("Fixed-token comparison gate: eligible", rendered)
        self.assertIn("expected 16/32/48 decoder-depth set is represented", rendered)
        self.assertIn("Actual tokens: 256", rendered)
        self.assertIn("Observed throughput: 256 tokens/s", rendered)
        self.assertIn("GPU board energy: 68 J", rendered)
        self.assertIn("GPU energy per actual token: 0.2656 J/token", rendered)
        self.assertIn("Fluke apparent energy: 68 VA·s", rendered)
        self.assertIn("Answer quality is unvalidated", rendered)
        self.assertIn("nonsense accuracy is accepted", rendered)
        self.assertIn("does not establish full-depth generation", rendered)
        self.assertIn("shared host", rendered)
        self.assertIn("sample window: 1 s", rendered)
        self.assertIn("weight size is not a power measurement", rendered)

    def test_mismatch_failure_and_unknown_count_block_fixed_token_comparison(self) -> None:
        report = report_with(
            run_record(48, actual_tokens=256),
            run_record(32, actual_tokens=255, label="depth32-short"),
            run_record(16, status="failed", actual_tokens=None, label="depth16-failed", error="backend stopped"),
        )
        rendered = layer_report.render_html(report)

        self.assertIn("Fixed-token comparison gate: blocked", rendered)
        self.assertIn("depth32-short: observed 255 of 256 requested", rendered)
        self.assertIn("depth16-failed: status is failed", rendered)
        self.assertIn("depth16-failed: actual token count unavailable", rendered)

    def test_per_run_requested_tokens_must_match_report_level_request(self) -> None:
        report = report_with(
            run_record(48, requested_tokens=512),
            run_record(32, requested_tokens=256),
            requested_tokens=256,
        )
        rendered = layer_report.render_html(report)

        self.assertIn("Fixed-token comparison gate: blocked", rendered)
        self.assertIn("depth48-trial1: run requested 512, report requests 256", rendered)
        self.assertIn("512 run-requested", rendered)

    def test_missing_per_run_request_blocks_eligibility(self) -> None:
        report = report_with(
            run_record(48, requested_tokens=None),
            run_record(32),
        )
        rendered = layer_report.render_html(report)

        self.assertIn("Fixed-token comparison gate: blocked", rendered)
        self.assertIn("depth48-trial1: run requested token count unavailable", rendered)

    def test_two_depths_are_labeled_partial_campaign(self) -> None:
        rendered = layer_report.render_html(report_with(run_record(48), run_record(32)))

        self.assertIn("Fixed-token comparison gate: eligible", rendered)
        self.assertIn(
            "Eligible for a partial depth comparison; expected campaign depths are 16/32/48, with 16 not represented.",
            rendered,
        )

    def test_label_summary_averages_exact_trials_and_keeps_before_after_separate(self) -> None:
        report = report_with(
            run_record(
                48,
                "depth48-before",
                trial=1,
                elapsed_ms=1000,
                gpu_measurement=power_measurement(mean=100, energy=50),
            ),
            run_record(
                48,
                "depth48-before",
                trial=2,
                elapsed_ms=3000,
                gpu_measurement=power_measurement(mean=200, energy=70),
            ),
            run_record(
                48,
                "depth48-after",
                status="failed",
                actual_tokens=None,
                error="backend stopped",
            ),
            run_record(32, "depth32-before"),
        )
        rendered = layer_report.render_html(report)

        before_match = re.search(r"<tr><td>depth48-before</td>.*?</tr>", rendered, re.DOTALL)
        after_match = re.search(r"<tr><td>depth48-after</td>.*?</tr>", rendered, re.DOTALL)
        self.assertIsNotNone(before_match)
        self.assertIsNotNone(after_match)
        before_row = before_match.group(0)
        after_row = after_match.group(0)
        self.assertIn("2 / 2", before_row)
        self.assertIn("25,000,000,000", before_row)
        self.assertIn("<td>2</td>", before_row)  # mean elapsed seconds
        self.assertIn("<td>150</td>", before_row)  # mean-of-trials GPU watts
        self.assertIn("<td>60</td>", before_row)  # mean GPU joules per request
        self.assertIn("<td>0.2344</td>", before_row)  # mean GPU joules per actual token
        self.assertIn("0 / 1", after_row)
        self.assertGreater(rendered.index("Summary by run label"), rendered.index("Fixed-token comparison gate"))
        self.assertLess(rendered.index("Summary by run label"), rendered.index("Sampled curves"))
        self.assertLess(rendered.index("Sampled curves"), rendered.index("Run comparison (details)"))
        self.assertIn("Unavailable", after_row)

    def test_label_summary_watts_require_available_active_power_units(self) -> None:
        report = report_with(
            run_record(
                48,
                "depth48-mixed-unit",
                trial=1,
                gpu_measurement=power_measurement("active_power", "W"),
            ),
            run_record(
                48,
                "depth48-mixed-unit",
                trial=2,
                gpu_measurement=power_measurement("current_ac", "VA"),
            ),
            run_record(32, "depth32-before"),
        )
        rendered = layer_report.render_html(report)
        row_match = re.search(r"<tr><td>depth48-mixed-unit</td>.*?</tr>", rendered, re.DOTALL)

        self.assertIsNotNone(row_match)
        self.assertIn("2 / 2", row_match.group(0))
        self.assertEqual(row_match.group(0).count("Unavailable"), 3)

    def test_missing_units_do_not_turn_energy_into_watts_or_joules(self) -> None:
        measurement = power_measurement()
        measurement.pop("kind")
        measurement.pop("power_unit")
        report = report_with(
            run_record(
                48,
                gpu_measurement=measurement,
                circuit_measurement={"status": "unavailable", "reason": "Fluke <offline>"},
            ),
            run_record(32, gpu_measurement=None, circuit_measurement=None),
        )
        rendered = layer_report.render_html(report)

        self.assertIn("GPU board power: Unavailable", rendered)
        self.assertIn("power unit unavailable or inconsistent", rendered)
        self.assertNotIn("GPU board energy: 68 J", rendered)
        self.assertNotIn("GPU board power: 104 W", rendered)
        self.assertIn("Fluke unavailable: Fluke &lt;offline&gt;", rendered)

    def test_gpu_and_circuit_plots_use_separate_axes_and_skip_mixed_gpu_units(self) -> None:
        report = report_with(
            run_record(
                48,
                gpu_measurement=power_measurement("active_power", "W"),
                circuit_measurement=power_measurement("current_ac", "VA", mean=170, energy=110),
            ),
            run_record(
                32,
                gpu_measurement=power_measurement("current_ac", "VA", mean=96, energy=62),
                circuit_measurement=power_measurement("current_ac", "VA", mean=155, energy=100),
            ),
        )
        rendered = layer_report.render_html(report)

        self.assertIn("GPU chart omitted because available curves mix W and VA", rendered)
        self.assertIn('aria-label="Circuit apparent power (VA) over elapsed time (ms)"', rendered)
        self.assertIn("decoder depth 48", rendered)
        self.assertIn("decoder depth 32", rendered)

    def test_gpu_utilization_has_its_own_percent_plot(self) -> None:
        report = report_with(
            run_record(48, gpu_utilization=utilization_measurement(76.5)),
            run_record(32, gpu_utilization=utilization_measurement(64.0)),
        )
        rendered = layer_report.render_html(report)

        self.assertIn("Mean GPU utilization: 76.5%", rendered)
        self.assertIn('aria-label="GPU utilization (%) over elapsed time (ms)"', rendered)

    def test_escapes_untrusted_fields_and_marks_failed_measurements_unavailable(self) -> None:
        run = run_record(
            48,
            label='<img src=x onerror="alert(1)">',
            status="failed",
            actual_tokens=None,
            error="backend <crashed>",
            answer_preview='<script>alert("x")</script>',
            gpu_measurement={"status": "unavailable", "reason": "device <busy>"},
            circuit_measurement={"status": "unavailable", "reason": "meter unavailable"},
        )
        report = report_with(run)
        report["campaign_id"] = '<b>campaign</b>'
        report["notes"] = "<svg onload=alert(1)>"
        rendered = layer_report.render_html(report)

        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", rendered)
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", rendered)
        self.assertIn("backend &lt;crashed&gt;", rendered)
        self.assertIn("&lt;svg onload=alert(1)&gt;", rendered)
        self.assertNotIn('<script>alert("x")</script>', rendered)
        self.assertIn("GPU unavailable: device &lt;busy&gt;", rendered)

    def test_rejects_invalid_depth_counts_and_non_finite_samples(self) -> None:
        with self.assertRaises(layer_report.ReportError):
            layer_report.render_html(report_with(run_record(0)))
        with self.assertRaises(layer_report.ReportError):
            layer_report.render_html(report_with(run_record(48, actual_tokens=1.5)))
        bad_measurement = power_measurement()
        bad_measurement["curve"][1]["value"] = float("nan")
        with self.assertRaises(layer_report.ReportError):
            layer_report.render_html(report_with(run_record(48, gpu_measurement=bad_measurement)))

    def test_merge_cli_combines_per_depth_batches(self) -> None:
        left = report_with(run_record(48, label="depth48-before"))
        right = report_with(run_record(16, label="depth16-before"))
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "depth48.json"
            second = Path(directory) / "depth16.json"
            output = Path(directory) / "layer-report.html"
            first.write_text(json.dumps(left), encoding="utf-8")
            second.write_text(json.dumps(right), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPORT_SCRIPT),
                    "merge",
                    "--input",
                    str(first),
                    str(second),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("Fixed-token comparison gate: eligible", rendered)
            self.assertIn("depth48-before", rendered)
            self.assertIn("depth16-before", rendered)
            self.assertIn("depth48.json", rendered)
            self.assertIn("depth16.json", rendered)

    def test_merge_recomputes_full_depth_and_drops_mixed_model_file_size(self) -> None:
        before = report_with(run_record(48, label="depth48-before"))
        before.update(
            {
                "full_depth": True,
                "model_file_bytes": 1000,
                "ignore_eos": True,
                "cache_prompt": False,
                "temperature": 0,
                "seed": 1,
                "warmup_tokens": 8,
            }
        )
        after = report_with(run_record(16, label="depth16-after"))
        after["runs"][0]["model_sha256"] = "c" * 64
        after["runs"][0]["model_identity"] = "another-gemma-4-test"
        after.update(
            {
                "full_depth": False,
                "model_file_bytes": 900,
                "ignore_eos": True,
                "cache_prompt": False,
                "temperature": 0,
                "seed": 1,
                "warmup_tokens": 8,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "before.json"
            second = Path(directory) / "after.json"
            first.write_text(json.dumps(before), encoding="utf-8")
            second.write_text(json.dumps(after), encoding="utf-8")

            merged = layer_report._merge_reports([first, second])

        self.assertFalse(merged["full_depth"])
        self.assertNotIn("model_file_bytes", merged)
        self.assertEqual(merged["ignore_eos"], True)
        self.assertEqual(merged["warmup_tokens"], 8)

    def test_merge_rejects_different_or_partially_missing_generation_settings(self) -> None:
        cases = (
            ({"temperature": 0}, {"temperature": 0.5}),
            ({"ignore_eos": True}, {"ignore_eos": False}),
            ({"cache_prompt": False}, {"cache_prompt": True}),
            ({"seed": 1}, {"seed": 2}),
            ({"warmup_tokens": 8}, {}),
        )
        for left_settings, right_settings in cases:
            with self.subTest(setting=(left_settings, right_settings)):
                left = report_with(run_record(48, label="depth48-before"))
                left.update(left_settings)
                right = report_with(run_record(16, label="depth16-after"))
                right.update(right_settings)
                with tempfile.TemporaryDirectory() as directory:
                    first = Path(directory) / "before.json"
                    second = Path(directory) / "after.json"
                    first.write_text(json.dumps(left), encoding="utf-8")
                    second.write_text(json.dumps(right), encoding="utf-8")

                    with self.assertRaises(layer_report.ReportError):
                        layer_report._merge_reports([first, second])

    def test_single_report_cli_writes_standalone_html(self) -> None:
        report = report_with(run_record(48))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            output = Path(directory) / "report.html"
            source.write_text(json.dumps(report), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(REPORT_SCRIPT), str(source), "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("Gemma 4 depth report", rendered)
            self.assertIn("Fixed-token comparison gate: blocked", rendered)


if __name__ == "__main__":
    unittest.main()
