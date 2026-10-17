#!/usr/bin/env python3

import http.server
import contextlib
import io
import json
import math
import struct
import sys
import threading
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prometheus_remote_write import (  # noqa: E402
    RemoteWriteClient,
    RemoteWriteError,
    Sample,
    TimeSeries,
    encode_write_request,
    main,
    parse_input_jsonl,
    snappy_compress_raw,
)


def read_varint(data, offset=0):
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, offset
        shift += 7


def decode_literal_snappy(data):
    expected, offset = read_varint(data)
    output = bytearray()
    while offset < len(data):
        tag = data[offset]
        offset += 1
        kind = tag & 0x03
        if kind != 0:
            raise AssertionError("fixture decoder only accepts literal Snappy tags")
        encoded_length = tag >> 2
        if encoded_length < 60:
            length = encoded_length + 1
        else:
            width = encoded_length - 59
            length = int.from_bytes(data[offset : offset + width], "little") + 1
            offset += width
        output.extend(data[offset : offset + length])
        offset += length
    if len(output) != expected:
        raise AssertionError(f"bad Snappy length: expected {expected}, decoded {len(output)}")
    return bytes(output)


def decode_write_request(data):
    """Independent minimal protobuf decoder for WriteRequest/TimeSeries/Label/Sample."""
    result = []
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        field, wire = key >> 3, key & 7
        if field != 1 or wire != 2:
            raise AssertionError("unexpected WriteRequest field")
        size, offset = read_varint(data, offset)
        body = data[offset : offset + size]
        offset += size
        result.append(decode_timeseries(body))
    return result


def decode_timeseries(data):
    labels, samples = [], []
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        field, wire = key >> 3, key & 7
        if wire != 2 or field not in (1, 2):
            raise AssertionError("unexpected TimeSeries field")
        size, offset = read_varint(data, offset)
        body = data[offset : offset + size]
        offset += size
        if field == 1:
            labels.append(decode_label(body))
        else:
            samples.append(decode_sample(body))
    return {"labels": labels, "samples": samples}


def decode_label(data):
    fields = {}
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        field, wire = key >> 3, key & 7
        if wire != 2 or field not in (1, 2):
            raise AssertionError("unexpected Label field")
        size, offset = read_varint(data, offset)
        fields[field] = data[offset : offset + size].decode("utf-8")
        offset += size
    return fields[1], fields[2]


def decode_sample(data):
    value = timestamp = None
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        field, wire = key >> 3, key & 7
        if field == 1 and wire == 1:
            value = struct.unpack("<d", data[offset : offset + 8])[0]
            offset += 8
        elif field == 2 and wire == 0:
            timestamp, offset = read_varint(data, offset)
        else:
            raise AssertionError("unexpected Sample field")
    return timestamp, value


def encode_string_field(field, value):
    raw = value.encode("utf-8")
    return bytes((field << 3 | 2,)) + bytes((len(raw),)) + raw


def encode_varint(value):
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


class RemoteWriteTests(unittest.TestCase):
    def test_snappy_literal_block_matches_independent_known_fixture(self):
        self.assertEqual(snappy_compress_raw(b""), b"\x00")
        self.assertEqual(snappy_compress_raw(b"abc"), b"\x03\x08abc")
        self.assertEqual(decode_literal_snappy(snappy_compress_raw(b"x" * 100)), b"x" * 100)

    def test_protobuf_wire_matches_independent_fixture_and_round_trips(self):
        series = TimeSeries(
            labels=(("job", "sensor"), ("__name__", "test_metric")),
            samples=(Sample(1500, 1.5),),
        )
        raw = encode_write_request((series,))

        name = encode_string_field(1, "__name__") + encode_string_field(2, "test_metric")
        job = encode_string_field(1, "job") + encode_string_field(2, "sensor")
        sample = b"\x09" + struct.pack("<d", 1.5) + b"\x10" + encode_varint(1500)
        time_series = b"\x0a" + encode_varint(len(name)) + name
        time_series += b"\x0a" + encode_varint(len(job)) + job
        time_series += b"\x12" + encode_varint(len(sample)) + sample
        expected = b"\x0a" + encode_varint(len(time_series)) + time_series

        self.assertEqual(raw, expected)
        self.assertEqual(
            decode_write_request(raw),
            [
                {
                    "labels": [("__name__", "test_metric"), ("job", "sensor")],
                    "samples": [(1500, 1.5)],
                }
            ],
        )

    def test_rejects_duplicate_or_invalid_labels_and_unordered_values(self):
        invalid = (
            TimeSeries((("__name__", "m"), ("job", "a"), ("job", "b")), (Sample(1, 1),)),
            TimeSeries((("__name__", "m"), ("bad-name", "x")), (Sample(1, 1),)),
            TimeSeries((("job", "a"),), (Sample(1, 1),)),
            TimeSeries((("__name__", "m"),), (Sample(2, 1), Sample(1, 2))),
            TimeSeries((("__name__", "m"),), (Sample(2, math.nan),)),
            TimeSeries((("__name__", "m"),), (Sample(-1, 1),)),
        )
        for series in invalid:
            with self.subTest(series=series), self.assertRaises(RemoteWriteError):
                encode_write_request((series,))

    def test_collector_jsonl_maps_ut61b_metrics_without_fluke_labels(self):
        rows = [
            {
                "receipt_epoch_ms": 1000,
                "receipt_monotonic_ns": 10,
                "valid": True,
                "mode": "AC",
                "circuit": "gputer",
                "current_amperes": 1.25,
                "voltage_assumption_v": 116.0,
                "voltage_provenance": "assumed_default",
            },
            {
                "receipt_epoch_ms": 1001,
                "receipt_monotonic_ns": 20,
                "valid": False,
                "mode": "AC",
                "reason": "hold_mode",
                "circuit": "gputer",
                "voltage_assumption_v": 116.0,
                "voltage_provenance": "assumed_default",
            },
            {
                "receipt_epoch_ms": 1002,
                "receipt_monotonic_ns": 30,
                "valid": True,
                "mode": "DC",
                "circuit": "gputer",
                "current_amperes": -0.25,
                "voltage_assumption_v": 116.0,
                "voltage_provenance": "assumed_default",
            },
        ]
        series, coalesced = parse_input_jsonl("\n".join(json.dumps(row) for row in rows))
        by_metric_and_mode = {
            (dict(item.labels)["__name__"], dict(item.labels)["mode"]): item for item in series
        }
        self.assertEqual(coalesced, 0)
        self.assertEqual(
            set(name for name, _mode in by_metric_and_mode),
            {"ut61b_measurement_valid", "ut61b_current_amperes", "ut61b_apparent_power_va"},
        )
        self.assertEqual(len(by_metric_and_mode[("ut61b_measurement_valid", "ac")].samples), 2)
        self.assertEqual(len(by_metric_and_mode[("ut61b_measurement_valid", "dc")].samples), 1)
        self.assertEqual(len(by_metric_and_mode[("ut61b_current_amperes", "ac")].samples), 1)
        self.assertEqual(len(by_metric_and_mode[("ut61b_current_amperes", "dc")].samples), 1)
        self.assertEqual(by_metric_and_mode[("ut61b_apparent_power_va", "ac")].samples[0].value, 145.0)
        self.assertNotIn("fluke_current_amperes", {dict(item.labels)["__name__"] for item in series})
        apparent_labels = dict(by_metric_and_mode[("ut61b_apparent_power_va", "ac")].labels)
        self.assertEqual(apparent_labels["voltage_assumption_v"], "116")
        self.assertEqual(apparent_labels["voltage_source"], "assumed")

    def test_invalid_rows_never_emit_current_and_same_millisecond_is_coalesced(self):
        rows = [
            {"receipt_epoch_ms": 1000, "valid": False, "mode": "AC", "reason": "hold_mode", "circuit": "gputer"},
            {"receipt_epoch_ms": 1000, "valid": True, "mode": "AC", "current_amperes": 0.5, "circuit": "gputer", "voltage_assumption_v": 116, "voltage_provenance": "assumed_default"},
        ]
        series, coalesced = parse_input_jsonl("\n".join(json.dumps(row) for row in rows))
        names = {dict(item.labels)["__name__"] for item in series}
        self.assertEqual(coalesced, 1)
        self.assertEqual(names, {"ut61b_measurement_valid", "ut61b_current_amperes", "ut61b_apparent_power_va"})
        validity = next(item for item in series if dict(item.labels)["__name__"] == "ut61b_measurement_valid")
        self.assertEqual([sample.value for sample in validity.samples], [1])
        invalid_only, _ = parse_input_jsonl(
            json.dumps({"receipt_epoch_ms": 999, "valid": False, "mode": "unknown", "reason": "junk", "circuit": "gputer"})
        )
        self.assertEqual([dict(item.labels)["__name__"] for item in invalid_only], ["ut61b_measurement_valid"])

    def test_rejects_timestamp_regression_and_current_on_invalid_row(self):
        regressed = [
            {"receipt_epoch_ms": 1001, "valid": True, "mode": "AC", "current_amperes": 1, "circuit": "gputer", "voltage_assumption_v": 116, "voltage_provenance": "assumed_default"},
            {"receipt_epoch_ms": 1000, "valid": True, "mode": "AC", "current_amperes": 2, "circuit": "gputer", "voltage_assumption_v": 116, "voltage_provenance": "assumed_default"},
        ]
        with self.assertRaises(RemoteWriteError):
            parse_input_jsonl("\n".join(json.dumps(row) for row in regressed))
        invalid_with_current = {"receipt_epoch_ms": 1000, "valid": False, "mode": "AC", "current_amperes": 1, "circuit": "gputer"}
        with self.assertRaises(RemoteWriteError):
            parse_input_jsonl(json.dumps(invalid_with_current))


class RemoteWriteHTTPTests(unittest.TestCase):
    def make_server(self, statuses):
        class Handler(http.server.BaseHTTPRequestHandler):
            requests = []
            replies = list(statuses)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                self.__class__.requests.append((dict(self.headers), body))
                status = self.__class__.replies.pop(0) if self.__class__.replies else 204
                self.send_response(status)
                self.end_headers()

            def log_message(self, *_args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, Handler

    def test_http_headers_payload_and_retry_after_5xx(self):
        server, thread, handler = self.make_server([503, 204])
        try:
            client = RemoteWriteClient(f"http://127.0.0.1:{server.server_port}", sleeper=lambda _delay: None)
            series = TimeSeries((("__name__", "m"),), (Sample(123, 2.0),))
            stats = client.write((series,))
            self.assertEqual(stats.attempts, 2)
            self.assertEqual(stats.http_retryable, 1)
            self.assertEqual(stats.http_success, 1)
            self.assertEqual(len(handler.requests), 2)
            headers, body = handler.requests[-1]
            self.assertEqual(headers["Content-Encoding"], "snappy")
            self.assertEqual(headers["Content-Type"], "application/x-protobuf")
            self.assertEqual(headers["X-Prometheus-Remote-Write-Version"], "0.1.0")
            self.assertTrue(headers["User-Agent"].startswith("cogni-ml/"))
            self.assertEqual(decode_write_request(decode_literal_snappy(body))[0]["samples"], [(123, 2.0)])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_4xx_fails_without_retry_and_cli_reports_compact_nonzero_failure(self):
        server, thread, handler = self.make_server([400])
        try:
            row = {
                "receipt_epoch_ms": 1000,
                "valid": True,
                "mode": "AC",
                "current_amperes": 0.5,
                "circuit": "gputer",
                "voltage_assumption_v": 116,
                "voltage_provenance": "assumed_default",
            }
            with tempfile.TemporaryDirectory() as directory:
                input_path = Path(directory) / "capture.jsonl"
                input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    result = main(["--input", str(input_path), "--vm-url", f"http://127.0.0.1:{server.server_port}"])
                self.assertEqual(result, 2)
                self.assertTrue(input_path.exists())
                self.assertIn("attempts=1", stderr.getvalue())
                self.assertIn("http_failure=1", stderr.getvalue())
                self.assertIn("failures=1", stderr.getvalue())
            self.assertEqual(len(handler.requests), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_network_failures_retry_only_to_the_configured_bound(self):
        class Offline:
            calls = 0

            def __call__(self, _request, timeout):
                self.calls += 1
                raise OSError("offline")

        offline = Offline()
        client = RemoteWriteClient("http://localhost:8428", max_attempts=3, sleeper=lambda _delay: None)
        client.opener = offline
        series = TimeSeries((("__name__", "m"),), (Sample(123, 2.0),))
        with self.assertRaisesRegex(RemoteWriteError, "network failure after 3 attempts"):
            client.write((series,))
        self.assertEqual(offline.calls, 3)
        self.assertEqual(client.stats.attempts, 3)
        self.assertEqual(client.stats.network_errors, 3)
        self.assertEqual(client.stats.failures, 1)

    def test_url_must_be_loopback_plain_http_without_credentials(self):
        for url in (
            "https://127.0.0.1:8428",
            "http://user:secret@127.0.0.1:8428",
            "http://example.com:8428",
        ):
            with self.subTest(url=url), self.assertRaises(RemoteWriteError):
                RemoteWriteClient(url)

    def test_cli_success_reports_counts_and_refuses_missing_data(self):
        server, thread, _handler = self.make_server([204])
        row = {
            "receipt_epoch_ms": 1000,
            "valid": True,
            "mode": "AC",
            "current_amperes": 0.5,
            "circuit": "gputer",
            "voltage_assumption_v": 116,
            "voltage_provenance": "assumed_default",
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                input_path = Path(directory) / "capture.jsonl"
                input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    result = main(["--input", str(input_path), "--vm-url", f"http://127.0.0.1:{server.server_port}"])
                self.assertEqual(result, 0)
                self.assertIn("samples=3", stdout.getvalue())
                self.assertIn("series=3", stdout.getvalue())
                self.assertIn("http_success=1", stdout.getvalue())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
