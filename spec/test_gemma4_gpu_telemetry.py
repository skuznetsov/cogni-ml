"""Tests for bounded Gemma 4 GPU sampling and local metric import."""

from __future__ import annotations

from contextlib import redirect_stderr
from datetime import datetime, timezone
import importlib.util
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gemma4_gpu_telemetry.py"
SPEC = importlib.util.spec_from_file_location("gemma4_gpu_telemetry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gemma4_gpu_telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gemma4_gpu_telemetry)


def sample_line(
    power: str = "32.26",
    utilization: str = "4",
    memory_used: str = "10400",
    memory_total: str = "16311",
) -> str:
    return (
        "2026/10/17 19:11:50.468, NVIDIA GeForce RTX 5060 Ti, "
        "GPU-426eaf30-0b2f-f612-17a7-2aa3cc3497fb, 595.104.02, "
        f"{power}, {utilization}, {memory_used}, {memory_total}"
    )


class TelemetryTests(unittest.TestCase):
    def parse(self, line: str) -> dict:
        return gemma4_gpu_telemetry.parse_sample_line(
            line,
            run_label="depth-eco",
            host="gputer",
            gpu_index=0,
            wall_time=datetime(2026, 10, 17, 19, 11, 51, 123000, tzinfo=timezone.utc),
            monotonic_ns=123456789,
        )

    def test_parses_gpu_identity_units_and_both_clock_domains(self) -> None:
        record = self.parse(sample_line())

        self.assertEqual(record["sample_status"], "ok")
        self.assertEqual(record["gpu_name"], "NVIDIA GeForce RTX 5060 Ti")
        self.assertEqual(record["gpu_uuid"], "GPU-426eaf30-0b2f-f612-17a7-2aa3cc3497fb")
        self.assertEqual(record["driver_version"], "595.104.02")
        self.assertEqual(record["power_source_field"], "power.draw.instant")
        self.assertEqual(record["power_watts"], 32.26)
        self.assertEqual(record["utilization_percent"], 4)
        self.assertEqual(record["memory_used_bytes"], 10400 * 1024 * 1024)
        self.assertEqual(record["memory_total_bytes"], 16311 * 1024 * 1024)
        self.assertEqual(record["timestamp_utc"], "2026-10-17T19:11:51.123Z")
        self.assertEqual(record["monotonic_ns"], 123456789)

    def test_missing_or_nonfinite_fields_are_errors_not_zero(self) -> None:
        record = self.parse(sample_line(power="N/A", utilization="nan", memory_used="inf"))

        self.assertEqual(record["sample_status"], "partial")
        self.assertIsNone(record["power_watts"])
        self.assertIsNone(record["utilization_percent"])
        self.assertIsNone(record["memory_used_bytes"])
        self.assertEqual(record["memory_total_bytes"], 16311 * 1024 * 1024)
        self.assertTrue(any("power.draw.instant unavailable" in item for item in record["sample_errors"]))
        self.assertTrue(any("non-finite" in item for item in record["sample_errors"]))
        self.assertNotIn(0, (record["power_watts"], record["utilization_percent"], record["memory_used_bytes"]))

    def test_real_zero_is_preserved_and_out_of_range_utilization_is_rejected(self) -> None:
        zero = self.parse(sample_line(power="0", utilization="0", memory_used="0"))
        self.assertEqual(zero["power_watts"], 0)
        self.assertEqual(zero["utilization_percent"], 0)
        self.assertEqual(zero["memory_used_bytes"], 0)

        invalid = self.parse(sample_line(utilization="101"))
        self.assertEqual(invalid["sample_status"], "partial")
        self.assertIsNone(invalid["utilization_percent"])
        self.assertIn("utilization.gpu is above 100 percent", invalid["sample_errors"])

    def test_malformed_csv_row_is_a_timestamped_error_record(self) -> None:
        record = self.parse("NVIDIA-SMI failed to query power")

        self.assertEqual(record["sample_status"], "error")
        self.assertEqual(record["timestamp_unix_ms"], 1792264311123)
        self.assertIn("expected 8 CSV columns", record["sample_errors"][0])

    def test_import_payload_uses_campaign_labels_and_only_observed_values(self) -> None:
        complete = self.parse(sample_line())
        partial = self.parse(sample_line(power="N/A", utilization="5"))
        lines = gemma4_gpu_telemetry.vm_import_lines(
            [complete, partial], run_label="depth-eco", host="gputer", gpu_index=0
        )
        payloads = [json.loads(line) for line in lines]
        by_name = {item["metric"]["__name__"]: item for item in payloads}

        self.assertEqual(set(by_name), {name for name, _ in gemma4_gpu_telemetry.METRICS})
        for item in payloads:
            self.assertEqual(item["metric"]["host"], "gputer")
            self.assertEqual(item["metric"]["gpu"], "0")
            self.assertEqual(item["metric"]["run"], "depth-eco")
            self.assertEqual(len(item["values"]), len(item["timestamps"]))
        self.assertEqual(by_name["cogni_gpu_power_watts"]["values"], [32.26])
        self.assertEqual(by_name["cogni_gpu_utilization_percent"]["values"], [4, 5])
        self.assertEqual(
            by_name["cogni_gpu_memory_used_bytes"]["values"],
            [10400 * 1024 * 1024, 10400 * 1024 * 1024],
        )

    def test_loopback_url_guard_rejects_remote_or_modified_urls(self) -> None:
        self.assertEqual(
            gemma4_gpu_telemetry._loopback_import_url("http://127.0.0.1:8428"),
            "http://127.0.0.1:8428/api/v1/import",
        )
        self.assertEqual(
            gemma4_gpu_telemetry._loopback_import_url("http://[::1]:8428/"),
            "http://[::1]:8428/api/v1/import",
        )
        for url in (
            "https://127.0.0.1:8428",
            "http://example.com:8428",
            "http://127.0.0.1:8428/api/v1/import",
            "http://user@127.0.0.1:8428",
        ):
            with self.subTest(url=url), self.assertRaises(gemma4_gpu_telemetry.TelemetryError):
                gemma4_gpu_telemetry._loopback_import_url(url)

    def test_optional_import_posts_only_to_a_local_endpoint(self) -> None:
        received: list[tuple[str, bytes]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - standard library handler API
                received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            record = self.parse(sample_line())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "samples.jsonl"
                path.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")
                count = gemma4_gpu_telemetry.import_jsonl_to_vm(
                    path,
                    vm_url=f"http://127.0.0.1:{server.server_port}",
                    run_label="depth-eco",
                    host="gputer",
                    gpu_index=0,
                )
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

        self.assertEqual(count, 3)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], "/api/v1/import")
        imported = [json.loads(line) for line in received[0][1].decode().splitlines()]
        self.assertEqual(len(imported), 3)
        self.assertIn("cogni_gpu_power_watts", {item["metric"]["__name__"] for item in imported})

    def test_live_import_posts_during_capture_and_flushes_final_batch(self) -> None:
        received: list[tuple[str, bytes]] = []
        first_import = threading.Event()
        received_lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - standard library handler API
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                with received_lock:
                    received.append((self.path, payload))
                    first_import.set()
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        importer = gemma4_gpu_telemetry.LiveVMImporter(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/api/v1/import",
            run_label="depth-eco",
            host="gputer",
            gpu_index=0,
            flush_interval_seconds=0.15,
            timeout_seconds=1.0,
        )
        try:
            importer.sample_sink(self.parse(sample_line()))
            self.assertTrue(first_import.wait(2), "live batch was not imported before close")
            with received_lock:
                self.assertEqual(len(received), 1)

            importer.sample_sink(self.parse(sample_line(power="33.5")))
            errors, imported_series = importer.close()
        finally:
            importer.close()
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

        self.assertEqual(errors, [])
        self.assertEqual(imported_series, 6)
        self.assertEqual(len(received), 2)
        self.assertTrue(all(path == "/api/v1/import" for path, _ in received))
        first = [json.loads(line) for line in received[0][1].decode().splitlines()]
        final = [json.loads(line) for line in received[1][1].decode().splitlines()]
        self.assertEqual({item["metric"]["__name__"] for item in first}, {name for name, _ in gemma4_gpu_telemetry.METRICS})
        self.assertEqual([item["values"] for item in first if item["metric"]["__name__"] == "cogni_gpu_power_watts"], [[32.26]])
        self.assertEqual([item["values"] for item in final if item["metric"]["__name__"] == "cogni_gpu_power_watts"], [[33.5]])

    def test_live_import_failure_keeps_raw_samples_and_is_reported(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - standard library handler API
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(503)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                fake_smi = Path(directory) / "fake-smi"
                fake_smi.write_text(
                    "#!/usr/bin/env python3\n"
                    "import time\n"
                    "print('2026/10/17 19:11:50.468, Fake GPU, GPU-test, 1.0, 30, 10, 1024, 8192', flush=True)\n"
                    "time.sleep(10)\n",
                    encoding="utf-8",
                )
                fake_smi.chmod(0o755)
                output = Path(directory) / "capture.jsonl"
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    result = gemma4_gpu_telemetry.main(
                        [
                            "--run-label", "failure-test",
                            "--duration-seconds", "0.5",
                            "--interval-ms", "100",
                            "--nvidia-smi", str(fake_smi),
                            "--output", str(output),
                            "--vm-url", f"http://127.0.0.1:{server.server_port}",
                        ]
                    )
                records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

        self.assertEqual(result, 1)
        self.assertIn("captured", stderr.getvalue())
        self.assertTrue(any(record.get("record_type") == "sample" for record in records))
        self.assertTrue(
            any(
                record.get("component") == "victoria_metrics_import"
                and "HTTP 503" in record.get("error", "")
                for record in records
            )
        )

    def test_output_creation_never_truncates_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.jsonl"
            output.write_text("previous evidence\n", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr), mock.patch.object(
                gemma4_gpu_telemetry, "capture", side_effect=AssertionError("capture must not start")
            ) as capture:
                result = gemma4_gpu_telemetry.main(
                    ["--run-label", "overwrite-test", "--output", str(output)]
                )

            self.assertEqual(result, 1)
            capture.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "previous evidence\n")
            self.assertIn("File exists", stderr.getvalue())

    def test_invalid_vm_url_is_rejected_before_capture_or_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.jsonl"
            stderr = io.StringIO()
            with redirect_stderr(stderr), mock.patch.object(
                gemma4_gpu_telemetry, "capture", side_effect=AssertionError("capture must not start")
            ) as capture:
                result = gemma4_gpu_telemetry.main(
                    [
                        "--run-label", "invalid-url",
                        "--output", str(output),
                        "--vm-url", "http://example.com:8428",
                    ]
                )

        self.assertEqual(result, 1)
        capture.assert_not_called()
        self.assertFalse(output.exists())
        self.assertIn("loopback", stderr.getvalue())

    def test_capture_stops_the_sampler_at_the_requested_duration(self) -> None:
        class FakeStdout:
            def fileno(self) -> int:
                return 17

            def close(self) -> None:
                self.closed = True

            closed = False

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = FakeStdout()
                self.returncode = None
                self.terminated = False
                self.wait_timeout = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout: float) -> int:
                self.wait_timeout = timeout
                self.returncode = -15
                return self.returncode

        class FakeClock:
            # The fake clock passes the capture deadline between rows delivered
            # in one read, so only the first row may be recorded.
            values = iter((10.0, 10.1, 10.2, 10.25, 10.31, 10.32))

            @classmethod
            def monotonic(cls) -> float:
                return next(cls.values)

            @staticmethod
            def monotonic_ns() -> int:
                return 123456789

        process = FakeProcess()
        output = io.StringIO()
        chunk = (
            b"2026/10/17 19:11:50.468, Fake GPU, GPU-test, 1.0, 30, 10, 1024, 8192\n"
            b"2026/10/17 19:11:50.718, Fake GPU, GPU-test, 1.0, 31, 11, 1025, 8192\n"
        )
        with (
            mock.patch.object(gemma4_gpu_telemetry, "time", FakeClock),
            mock.patch.object(gemma4_gpu_telemetry.subprocess, "Popen", return_value=process) as popen,
            mock.patch.object(
                gemma4_gpu_telemetry.select,
                "select",
                return_value=([process.stdout], [], []),
            ) as select_call,
            mock.patch.object(gemma4_gpu_telemetry.os, "read", return_value=chunk) as read_call,
        ):
            count, errors = gemma4_gpu_telemetry.capture(
                run_label="bounded-test",
                host="gputer",
                gpu_index=0,
                duration_seconds=0.3,
                interval_ms=250,
                output=output,
                nvidia_smi="fake-nvidia-smi",
            )

        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(count, 1)
        self.assertEqual(errors, 0)
        popen.assert_called_once()
        select_call.assert_called_once_with([process.stdout], [], [], mock.ANY)
        self.assertAlmostEqual(select_call.call_args.args[3], 0.1)
        read_call.assert_called_once_with(17, 65536)
        self.assertTrue(process.terminated)
        self.assertEqual(process.wait_timeout, 1)
        self.assertTrue(process.stdout.closed)
        self.assertEqual(records[0]["event"], "start")
        self.assertEqual(records[-1]["event"], "end")
        self.assertEqual(records[-1]["sample_count"], count)
        self.assertTrue(all(record["sample_status"] == "ok" for record in records[1:-1]))


if __name__ == "__main__":
    unittest.main()
