"""Tests for the standalone Gemma 4 energy report renderer."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPORT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gemma4_energy_report.py"
REPORT_SPEC = importlib.util.spec_from_file_location("gemma4_energy_report", REPORT_SCRIPT)
assert REPORT_SPEC is not None and REPORT_SPEC.loader is not None
gemma4_energy_report = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(gemma4_energy_report)


SCHEMA = "cogni.gemma4.energy.v1"


def report_with(*runs: dict) -> dict:
    return {
        "schema": SCHEMA,
        "model_identity": "gemma-4-test",
        "model_sha256": "model-hash",
        "prompt_sha256": "prompt-hash",
        "full_depth": True,
        "notes": "Whole circuit request window.",
        "runs": list(runs),
    }


def run_record(
    name: str,
    *,
    measurement: dict | None,
    answer: str = "The answer is 391.",
    reasoning_budget: int = 0,
    reasoning_actual: int = 0,
    answer_budget: int = 256,
    answer_actual: int = 12,
    reasoning_outcome: str = "skipped_zero_budget",
    answer_outcome: str = "turn_end",
    reasoning_forced_close: bool = False,
    reasoning_truncated: bool = False,
    answer_truncated: bool = False,
) -> dict:
    return {
        "profile": {
            "name": name,
            "reasoning_tokens": reasoning_budget,
            "answer_tokens": answer_budget,
        },
        "round": 0,
        "answer": answer,
        "phases": {
            "reasoning": {
                "predicted_tokens": reasoning_actual,
                "outcome": reasoning_outcome,
                "forced_close": reasoning_forced_close,
                "truncated": reasoning_truncated,
            },
            "answer": {
                "predicted_tokens": answer_actual,
                "outcome": answer_outcome,
                "truncated": answer_truncated,
            },
        },
        "started_at_epoch_ms": 1_000,
        "ended_at_epoch_ms": 2_000,
        "elapsed_ms": 987,
        "measurement": measurement,
    }


def available_measurement(kind: str, unit: str, energy_field: str, energy: float) -> dict:
    return {
        "status": "available",
        "kind": kind,
        "power_unit": unit,
        "curve": [
            {"elapsed_ms": 0, "value": 140.0},
            {"elapsed_ms": 500, "value": 150.0},
            {"elapsed_ms": 1_000, "value": 160.0},
        ],
        "duration_seconds": 1.0,
        "mean_apparent_power_va": 150.0 if unit == "VA" else None,
        "apparent_energy_va_seconds": energy if energy_field == "apparent_energy_va_seconds" else None,
        "mean_active_power_watts": 120.0 if unit == "W" else None,
        "active_energy_joules": energy if energy_field == "active_energy_joules" else None,
        "estimated_active_energy_joules": None,
        "assumed_voltage_volts": None,
        "assumed_power_factor": None,
        "sample_count": 3,
    }


class EnergyReportTests(unittest.TestCase):
    def test_escapes_answer_and_metadata_in_html(self) -> None:
        run = run_record(
            "Eco",
            measurement=available_measurement("current_ac", "VA", "apparent_energy_va_seconds", 125.0),
            answer='<script>alert("x")</script> & <b>private</b>',
        )
        source = report_with(run)
        source["model_identity"] = '<img src=x onerror="alert(1)">'
        rendered = gemma4_energy_report.render_html(source)

        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &lt;b&gt;private&lt;/b&gt;", rendered)
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", rendered)
        self.assertNotIn('<script>alert("x")</script>', rendered)
        self.assertNotIn('<img src=x onerror="alert(1)">', rendered)
        self.assertIn("987 ms", rendered)
        self.assertIn("0 / 0", rendered)
        self.assertIn("256 / 12", rendered)
        self.assertIn("skipped_zero_budget", rendered)

    def test_unavailable_and_null_values_are_labelled_unavailable(self) -> None:
        run = run_record(
            "Balanced",
            measurement={"status": "unavailable", "reason": "collector <not ready>"},
        )
        run["elapsed_ms"] = None
        run["phases"]["answer"]["predicted_tokens"] = None
        run["answer"] = None
        rendered = gemma4_energy_report.render_html(report_with(run))

        self.assertIn("Unavailable", rendered)
        self.assertIn("collector &lt;not ready&gt;", rendered)
        self.assertIn("Answer unavailable", rendered)
        self.assertNotIn("0 J", rendered)
        self.assertNotIn("0 VA", rendered)

    def test_current_measurement_keeps_va_seconds_and_estimated_joules_distinct(self) -> None:
        measurement = available_measurement("current_ac", "VA", "apparent_energy_va_seconds", 180.0)
        measurement["estimated_active_energy_joules"] = 172.8
        measurement["assumed_voltage_volts"] = 120.0
        measurement["assumed_power_factor"] = 0.8
        rendered = gemma4_energy_report.render_html(report_with(run_record("Eco", measurement=measurement)))

        self.assertIn("180 VA·s", rendered)
        self.assertIn("Estimated active energy", rendered)
        self.assertIn("172.8 J", rendered)
        self.assertIn("assumed", rendered.lower())
        self.assertNotIn("Active energy: 180 J", rendered)
        self.assertNotIn("180 W", rendered)

    def test_active_power_reports_watts_and_joules(self) -> None:
        measurement = available_measurement("active_power", "W", "active_energy_joules", 120.0)
        rendered = gemma4_energy_report.render_html(report_with(run_record("Quality", measurement=measurement)))

        self.assertIn("120 J", rendered)
        self.assertIn("120 W", rendered)
        self.assertNotIn("VA·s", rendered)
        self.assertIn('aria-label="Active power (W) over elapsed time (ms)"', rendered)

    def test_reasoning_forced_close_and_answer_truncation_are_visible(self) -> None:
        run = run_record(
            "Balanced",
            measurement=None,
            reasoning_budget=384,
            reasoning_actual=384,
            answer_actual=256,
            reasoning_outcome="forced_close",
            answer_outcome="answer_limit",
            reasoning_forced_close=True,
            answer_truncated=True,
        )
        rendered = gemma4_energy_report.render_html(report_with(run))

        self.assertIn("384 / 384", rendered)
        self.assertIn("Forced close: forced close", rendered)
        self.assertIn("Outcome: answer_limit", rendered)
        self.assertIn("Truncated: truncated", rendered)

    def test_failed_run_keeps_known_budgets_and_shows_missing_phase_data(self) -> None:
        run = {
            "status": "failed",
            "error": "server <closed>",
            "profile": {"name": "Quality", "reasoning_tokens": 1024, "answer_tokens": 512},
            "round": 2,
            "started_at_epoch_ms": 1000,
            "ended_at_epoch_ms": 1100,
            "measurement": {"status": "unavailable", "reason": "no window"},
        }
        rendered = gemma4_energy_report.render_html(report_with(run))

        self.assertIn("Failed: server &lt;closed&gt;", rendered)
        self.assertIn("1024 / Unavailable", rendered)
        self.assertIn("512 / Unavailable", rendered)
        self.assertIn("Answer unavailable", rendered)

    def test_mixed_units_do_not_share_a_chart_axis(self) -> None:
        report = report_with(
            run_record("Eco", measurement=available_measurement("current_ac", "VA", "apparent_energy_va_seconds", 125.0)),
            run_record("Quality", measurement=available_measurement("active_power", "W", "active_energy_joules", 125.0)),
        )
        rendered = gemma4_energy_report.render_html(report)

        self.assertIn("Chart omitted because available curves mix VA and W", rendered)
        self.assertNotIn("<svg", rendered)

    def test_consistent_chart_uses_common_elapsed_axis_and_profile_legend(self) -> None:
        report = report_with(
            run_record("Eco", measurement=available_measurement("current_ac", "VA", "apparent_energy_va_seconds", 125.0)),
            run_record("Balanced", measurement=available_measurement("current_ac", "VA", "apparent_energy_va_seconds", 145.0)),
        )
        rendered = gemma4_energy_report.render_html(report)

        self.assertIn('<svg', rendered)
        self.assertIn('aria-label="Apparent power (VA) over elapsed time (ms)"', rendered)
        self.assertIn("Eco", rendered)
        self.assertIn("Balanced", rendered)
        self.assertIn("Elapsed time (ms)", rendered)
        self.assertIn("polyline", rendered)

    def test_rejects_wrong_schema_and_non_object_runs(self) -> None:
        with self.assertRaises(gemma4_energy_report.ReportError):
            gemma4_energy_report.render_html({"schema": "other", "runs": []})
        with self.assertRaises(gemma4_energy_report.ReportError):
            gemma4_energy_report.render_html(report_with("not a run"))

    def test_cli_writes_standalone_html(self) -> None:
        report = report_with(run_record("Eco", measurement=None))
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "report.json"
            output_path = Path(directory) / "report.html"
            input_path.write_text(json.dumps(report), encoding="utf-8")
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(
                [sys.executable, "scripts/gemma4_energy_report.py", str(input_path), "--output", str(output_path)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = output_path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", output.lower())
            self.assertIn("Energy comparison", output)
            self.assertIn("The answer is 391.", output)


if __name__ == "__main__":
    unittest.main()
