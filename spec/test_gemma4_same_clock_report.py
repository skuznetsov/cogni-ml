"""Falsifiers for the same-clock Gemma 4 campaign postprocessor."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gemma4_same_clock_report.py"
SPEC = importlib.util.spec_from_file_location("gemma4_same_clock_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
same_clock = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = same_clock
SPEC.loader.exec_module(same_clock)

rw = same_clock.remote_write


def series(labels: dict[str, str], points: list[tuple[int, float]]) -> rw.TimeSeries:
    return rw.TimeSeries(
        tuple(sorted(labels.items())),
        tuple(rw.Sample(timestamp, value) for timestamp, value in points),
    )


def ut_row(
    timestamp: int,
    *,
    valid: bool = True,
    mode: str = "AC",
    current: float | None = 1.0,
    voltage: float = 116.0,
) -> dict:
    row = {
        "schema_version": 1,
        "source": "ut61b",
        "circuit": "gputer",
        "receipt_epoch_ms": timestamp,
        "valid": valid,
        "mode": mode,
        "voltage_assumption_v": voltage,
        "voltage_provenance": "assumed_default",
    }
    if valid and current is not None:
        row["current_amperes"] = current
    else:
        row["reason"] = "synthetic rejected frame"
    return row


def token_run(*, actual_tokens: int = 256, final_tokens: int = 256) -> dict:
    labels = {
        "campaign": "c1",
        "host": "gputer",
        "job": "cogni-gemma4",
        "label": "depth48-before",
        "layers": "48",
        "phase": "answer",
        "trial": "1",
    }
    return {
        "label": "depth48-before",
        "decoder_layers": 48,
        "trial": 1,
        "status": "completed",
        "requested_tokens": 256,
        "actual_tokens": actual_tokens,
        "started_at_epoch_ms": 1000,
        "ended_at_epoch_ms": 2000,
        "token_progress": {
            "status": "available",
            "basis": "received_token_ids",
            "requested_tokens": 256,
            "final_tokens": final_tokens,
            "observed_tokens": final_tokens,
            "remote_write": {
                "status": "available",
                "metric": "cogni_llm_generated_tokens_total",
                "selector": "cogni_llm_generated_tokens_total{" + ",".join(
                    f'{name}={json.dumps(value)}' for name, value in sorted(labels.items())
                ) + "}",
                "sample_count": 3,
            },
            "points": [
                {"elapsed_ms": 0, "generated_tokens": 0},
                {"elapsed_ms": 500, "generated_tokens": 120},
                {"elapsed_ms": 1000, "generated_tokens": final_tokens},
            ],
        },
    }


class _MockResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status

    def read(self, _limit=-1):
        return self.body


class _MockVM:
    """A tiny VM stand-in that round-trips the exact series sent by the writer."""

    def __init__(self):
        self.series = ()
        self.export_selectors = []

    def open(self, request, timeout=None):
        self.assert_timeout(timeout)
        query = parse_qs(urlsplit(request.full_url).query)
        selector = query["match[]"][0]
        self.export_selectors.append(selector)
        metric = selector.split("{", 1)[0]
        wanted = dict(
            (key, value)
            for key, value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', selector)
        )
        start_ms = int(float(query["start"][0]) * 1000)
        end_ms = int(float(query["end"][0]) * 1000)
        rows = []
        for item in self.series:
            labels = dict(item.labels)
            if labels.get("__name__") != metric or any(labels.get(key) != value for key, value in wanted.items()):
                continue
            points = [sample for sample in item.samples if start_ms <= sample.timestamp_ms <= end_ms]
            if points:
                rows.append({
                    "metric": labels,
                    "values": [sample.value for sample in points],
                    "timestamps": [sample.timestamp_ms for sample in points],
                })
        return _MockResponse(("\n".join(json.dumps(row) for row in rows) + "\n").encode())

    @staticmethod
    def assert_timeout(timeout):
        if timeout is None or timeout <= 0:
            raise AssertionError("export request must use a bounded timeout")


def _batch_report(label: str, depth: int, campaign: str, runs: list[dict]) -> dict:
    return {
        "schema": "cogni.gemma4.layer_depth.v1",
        "campaign_id": campaign,
        "prompt_sha256": "a" * 64,
        "requested_tokens": 256,
        "circuit_meter": "fluke",
        "measurement_clock": "gputer",
        "notes": f"Legacy {label} report measured by Fluke.",
        "ignore_eos": True,
        "cache_prompt": True,
        "temperature": 0.0,
        "seed": 17,
        "warmup_tokens": 8,
        "runs": runs,
    }


def _campaign_run(label: str, depth: int, trial: int, start: int, campaign: str) -> dict:
    token_labels = {
        "campaign": campaign,
        "host": "gputer",
        "job": "cogni-gemma4",
        "label": label,
        "layers": str(depth),
        "phase": "answer",
        "trial": str(trial),
    }
    selector = "cogni_llm_generated_tokens_total{" + ",".join(
        f'{key}={json.dumps(value)}' for key, value in sorted(token_labels.items())
    ) + "}"
    return {
        "label": label,
        "decoder_layers": depth,
        "trial": trial,
        "status": "completed",
        "requested_tokens": 256,
        "actual_tokens": 256,
        "started_at_epoch_ms": start,
        "ended_at_epoch_ms": start + 1000,
        "elapsed_ms": 1000,
        "gpu_measurement": None,
        "circuit_measurement": {"status": "available", "kind": "current_ac", "power_unit": "VA", "marker": "stale Fluke source"},
        "token_progress": {
            "status": "available",
            "basis": "received_token_ids",
            "requested_tokens": 256,
            "final_tokens": 256,
            "observed_tokens": 256,
            "remote_write": {
                "status": "available",
                "metric": "cogni_llm_generated_tokens_total",
                "selector": selector,
                "sample_count": 3,
            },
            "points": [
                {"elapsed_ms": 0, "generated_tokens": 0},
                {"elapsed_ms": 500, "generated_tokens": 128},
                {"elapsed_ms": 1000, "generated_tokens": 256},
            ],
        },
    }


class SameClockEvidenceTests(unittest.TestCase):
    def test_piecewise_integration_requires_both_request_boundaries(self) -> None:
        with self.assertRaisesRegex(same_clock.ReportError, "coverage"):
            same_clock.integrate_window([(101, 1.0), (500, 2.0)], 100, 900)

    def test_piecewise_integration_rejects_gaps_over_two_seconds(self) -> None:
        with self.assertRaisesRegex(same_clock.ReportError, "gap"):
            same_clock.integrate_window([(0, 1.0), (2001, 2.0), (3000, 2.0)], 500, 2500)

    def test_global_held_validity_includes_unknown_mode_but_allows_prior_rejection(self) -> None:
        rows = [
            ut_row(0, valid=False, mode="unknown", current=None),
            ut_row(150, valid=True, mode="AC", current=0.9),
            ut_row(450, valid=True, mode="AC", current=1.1),
            ut_row(1050, valid=True, mode="AC", current=1.0),
        ]
        expected, _ = same_clock.parse_ut_capture("\n".join(json.dumps(row) for row in rows))
        events = same_clock.validity_events(expected)
        same_clock.require_valid_held(events, 200, 900, 2000)

        held_invalid = rows + [ut_row(600, valid=False, mode="unknown", current=None)]
        expected_invalid, _ = same_clock.parse_ut_capture(
            "\n".join(json.dumps(row) for row in held_invalid)
        )
        with self.assertRaisesRegex(same_clock.ReportError, "invalid"):
            same_clock.require_valid_held(
                same_clock.validity_events(expected_invalid), 200, 900, 2000
            )

    def test_vm_export_rejects_conflicting_duplicate_samples(self) -> None:
        labels = {"__name__": "cogni_gpu_power_watts", "host": "gputer", "gpu": "0", "run": "c1"}
        raw = "\n".join(
            json.dumps({"metric": labels, "values": [value], "timestamps": [1000]})
            for value in (42.0, 43.0)
        )
        with self.assertRaisesRegex(same_clock.ReportError, "conflicting duplicate"):
            same_clock.parse_vm_export(raw)

    def test_vm_visibility_retries_missing_tail_then_saves_exact_raw_body(self) -> None:
        expected = [series({"__name__": "m", "source": "synthetic"}, [(1000, 1.0), (2000, 2.0)])]
        partial = json.dumps({
            "metric": dict(expected[0].labels),
            "values": [1.0],
            "timestamps": [1000],
        }).encode()
        complete = json.dumps({
            "metric": dict(expected[0].labels),
            "values": [1.0, 2.0],
            "timestamps": [1000, 2000],
        }).encode()
        calls = []

        def opener(_request, timeout=None):
            self.assertIsNotNone(timeout)
            calls.append(timeout)
            return _MockResponse(partial if len(calls) < 3 else complete)

        sleeps = []
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "export.raw.jsonl"
            with mock.patch.object(same_clock.time, "sleep", side_effect=sleeps.append):
                actual, retry_count = same_clock._fetch_export(
                    opener=opener,
                    endpoint="http://127.0.0.1:8428/api/v1/export",
                    selector='m{source="synthetic"}',
                    capture_range=same_clock.CaptureRange(1000, 2000),
                    expected_series=expected,
                    destination=destination,
                )
            self.assertEqual(len(calls), 3)
            self.assertEqual(retry_count, 2)
            self.assertEqual(sleeps, [0.5, 1.0])
            self.assertEqual(destination.read_bytes(), complete)
            self.assertEqual(actual[expected[0].labels], {1000: 1.0, 2000: 2.0})

    def test_roundtrip_rejects_missing_extra_and_mismatched_samples(self) -> None:
        expected = [series({"__name__": "m", "run": "c1"}, [(100, 1.0), (200, 2.0)])]
        missing = {expected[0].labels: {100: 1.0}}
        with self.assertRaisesRegex(same_clock.ReportError, "missing"):
            same_clock.compare_samples(expected, missing, 100, 200)

        extra = {expected[0].labels: {100: 1.0, 200: 2.0, 201: 3.0}}
        with self.assertRaisesRegex(same_clock.ReportError, "extra"):
            same_clock.compare_samples(expected, extra, 100, 201)

        wrong = {expected[0].labels: {100: 1.0, 200: 2.0001}}
        with self.assertRaisesRegex(same_clock.ReportError, "value mismatch"):
            same_clock.compare_samples(expected, wrong, 100, 200)

    def test_token_progress_requires_exact_received_id_work_and_sample_count(self) -> None:
        report = {
            "schema": "cogni.gemma4.layer_depth.v1",
            "campaign_id": "c1",
            "requested_tokens": 256,
            "runs": [token_run()],
        }
        token_series = same_clock.token_series_from_reports([report])
        self.assertEqual(len(token_series), 1)
        self.assertEqual(
            [(sample.timestamp_ms, sample.value) for sample in token_series[0].samples],
            [(1000, 0.0), (1500, 120.0), (2000, 256.0)],
        )

        bad = token_run(actual_tokens=255, final_tokens=256)
        with self.assertRaisesRegex(same_clock.ReportError, "actual_tokens"):
            same_clock.token_series_from_reports(
                [{"schema": "cogni.gemma4.layer_depth.v1", "campaign_id": "c1", "requested_tokens": 256, "runs": [bad]}]
            )

        bad_count = token_run()
        bad_count["token_progress"]["remote_write"]["sample_count"] = 2
        with self.assertRaisesRegex(same_clock.ReportError, "sample_count"):
            same_clock.token_series_from_reports(
                [{"schema": "cogni.gemma4.layer_depth.v1", "campaign_id": "c1", "requested_tokens": 256, "runs": [bad_count]}]
            )

    def test_ut_capture_requires_the_declared_116_volt_assumption(self) -> None:
        good = [ut_row(100, voltage=116.0)]
        same_clock.parse_ut_capture(json.dumps(good[0]))
        bad = [ut_row(100, voltage=120.0)]
        with self.assertRaisesRegex(same_clock.ReportError, "116 V"):
            same_clock.parse_ut_capture(json.dumps(bad[0]))

    def test_combined_report_replaces_fluke_and_sets_meter_and_clock(self) -> None:
        report = {
            "schema": "cogni.gemma4.layer_depth.v1",
            "campaign_id": "c1",
            "requested_tokens": 256,
            "runs": [token_run()],
        }
        stale_fluke = {"status": "available", "kind": "current_ac", "power_unit": "VA", "marker": "Fluke"}
        report["runs"][0]["circuit_measurement"] = stale_fluke
        output = same_clock.install_campaign_metadata(
            report,
            source_reports=["batch.json"],
            evidence_exports={"ut61b": "campaign.ut61b.raw.jsonl"},
        )
        self.assertEqual(output["circuit_meter"], "ut61b")
        self.assertEqual(output["measurement_clock"], "gputer")
        self.assertEqual(output["source_reports"], ["batch.json"])
        self.assertNotIn("marker", output["runs"][0]["circuit_measurement"])
        self.assertIn("116 V", output["notes"])
        self.assertNotIn("Fluke", output["notes"])

    def test_campaign_matrix_requires_all_named_batches_and_three_trials(self) -> None:
        batches = [("depth48-before", 48), ("depth32", 32), ("depth16", 16), ("depth48-after", 48)]
        report = {
            "requested_tokens": 256,
            "runs": [
                {"label": label, "decoder_layers": depth, "trial": trial}
                for label, depth in batches
                for trial in (1, 2, 3)
            ],
        }
        same_clock.validate_campaign_matrix(report)
        with self.assertRaisesRegex(same_clock.ReportError, "exactly 3 trials"):
            same_clock.validate_campaign_matrix({**report, "runs": report["runs"][:-1]})

    def test_ut_same_timestamp_conflict_is_not_last_write_wins(self) -> None:
        rows = [ut_row(100, current=0.9), ut_row(100, current=1.0)]
        with self.assertRaisesRegex(same_clock.ReportError, "conflicting duplicate"):
            same_clock.parse_ut_capture("\n".join(json.dumps(row) for row in rows))

    def test_verify_existing_roundtrip_uses_real_batch_identity_without_writing(self) -> None:
        campaign = "depth-ut61b-synthetic"
        batches = [
            ("depth48-before", 48),
            ("depth32", 32),
            ("depth16", 16),
            ("depth48-after", 48),
        ]
        report_values = []
        run_number = 0
        for label, depth in batches:
            runs = []
            for trial in (1, 2, 3):
                start = 10_000 + run_number * 2_000
                runs.append(_campaign_run(label, depth, trial, start, campaign))
                run_number += 1
            report_values.append(_batch_report(label, depth, campaign, runs))

        gpu_rows = []
        for timestamp in range(9_000, 34_001, 1_000):
            gpu_rows.append({
                "schema": "cogni.gemma4.gpu-telemetry.v1",
                "record_type": "sample",
                "run": campaign,
                "host": "gputer",
                "gpu": "0",
                "gpu_index": 0,
                "sample_status": "ok",
                "timestamp_unix_ms": timestamp,
                "power_watts": 250.0,
                "utilization_percent": 50.0,
                "memory_used_bytes": 8_000_000_000,
            })
        ut_rows = [ut_row(9_000, valid=False, mode="unknown", current=None)]
        ut_rows.extend(
            ut_row(timestamp, current=1.0)
            for timestamp in range(10_000, 34_001, 1_000)
        )

        vm = _MockVM()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_paths = []
            for index, report in enumerate(report_values):
                path = root / f"batch-{index}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                report_paths.append(path)
            gpu_path = root / "gpu.jsonl"
            ut_path = root / "ut61b.jsonl"
            gpu_path.write_text("\n".join(json.dumps(row) for row in gpu_rows) + "\n", encoding="utf-8")
            ut_path.write_text("\n".join(json.dumps(row) for row in ut_rows) + "\n", encoding="utf-8")

            gpu_series, _ = same_clock.parse_gpu_capture(gpu_path.read_text(), campaign_id=campaign)
            ut_series, _ = same_clock.parse_ut_capture(ut_path.read_text())
            token_series = same_clock.token_series_from_reports(report_values)
            vm.series = (*gpu_series, *ut_series, *token_series)

            output = root / "combined.json"
            evidence_prefix = root / "campaign-vm"
            with (
                mock.patch.object(same_clock.urllib.request, "build_opener", return_value=vm),
                mock.patch.object(
                    same_clock.remote_write,
                    "RemoteWriteClient",
                    side_effect=AssertionError("verification-only mode must not write"),
                ),
            ):
                report, status = same_clock.build_report(
                    inputs=report_paths,
                    gpu_jsonl=gpu_path,
                    ut61b_jsonl=ut_path,
                    vm_url="http://127.0.0.1:8428",
                    output=output,
                    evidence_prefix=evidence_prefix,
                    verify_existing=True,
                )

            self.assertEqual(status, 0)
            self.assertEqual(len(report["runs"]), 12)
            self.assertEqual(report["measurement_coverage"]["status"], "available")
            self.assertEqual(report["measurement_clock"], "gputer")
            self.assertEqual(report["circuit_meter"], "ut61b")
            self.assertEqual(report["remote_write"]["status"], "verified_existing")
            self.assertFalse(report["remote_write"]["write_attempted"])
            self.assertEqual(report["remote_write"]["expected_sample_count"], 12 * 3 + len(gpu_series[0].samples) * 3 + sum(len(item.samples) for item in ut_series))
            self.assertEqual(len(vm.export_selectors), 7)
            self.assertNotIn("Fluke", report["notes"])
            for run in report["runs"]:
                self.assertEqual(run["circuit_measurement"]["meter"], "ut61b")
                self.assertNotIn("marker", run["circuit_measurement"])
                self.assertEqual(run["circuit_measurement"]["status"], "available")
                self.assertEqual(run["gpu_measurement"]["status"], "available")
            self.assertEqual(set(report["evidence_exports"]), {
                "gpu_power", "gpu_utilization", "gpu_memory", "ut61b_valid",
                "ut61b_current", "ut61b_apparent", "tokens",
            })
            self.assertTrue(output.is_file())
            self.assertTrue(all((root / name).is_file() for name in report["evidence_exports"].values()))


if __name__ == "__main__":
    unittest.main()
