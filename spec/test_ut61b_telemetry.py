#!/usr/bin/env python3

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ut61b_telemetry import (  # noqa: E402
    FrameCandidate,
    FrameFramer,
    _capture,
    decode_frame,
    measurement_record,
    validate_capture_args,
)


def make_frame(
    *,
    sign=b"+",
    digits=b"1234",
    decimal=b"1",
    flags=0x08,
    status=0,
    prefix=0,
    unit=0x40,
    bargraph=0,
):
    return sign + digits + b" " + decimal + bytes((flags, status, prefix, unit, bargraph)) + b"\r\n"


class UT61BTelemetryTests(unittest.TestCase):
    def test_captured_bargraph_frame_decodes_across_split_crlf(self):
        # Captured report row 1; mode-byte bit 0 signals the active bargraph.
        raw = bytes.fromhex("2b30383634203129000040080d0a")
        framer = FrameFramer()

        self.assertEqual(framer.feed(raw[:-1]), [])
        candidates = framer.feed(raw[-1:])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].raw, raw)
        self.assertIsNone(candidates[0].error)

        measurement = decode_frame(candidates[0].raw)
        self.assertTrue(measurement.valid, measurement.reason)
        self.assertEqual(measurement.mode, "AC")
        self.assertAlmostEqual(measurement.current_amperes, 0.864)

    def test_current_units_and_decimal_exponents_are_scaled_to_amperes(self):
        amps = decode_frame(make_frame())
        milliamps = decode_frame(make_frame(prefix=0x40))
        microamps = decode_frame(make_frame(prefix=0x80))

        self.assertTrue(amps.valid)
        self.assertEqual(amps.mode, "AC")
        self.assertAlmostEqual(amps.current_amperes, 1.234)
        self.assertAlmostEqual(milliamps.current_amperes, 0.001234)
        self.assertAlmostEqual(microamps.current_amperes, 0.000001234)
        self.assertEqual((amps.unit_prefix, milliamps.unit_prefix, microamps.unit_prefix), ("A", "mA", "uA"))

    def test_dc_and_zero_are_valid_measurements(self):
        result = decode_frame(make_frame(flags=0x10, digits=b"0000"))
        self.assertTrue(result.valid)
        self.assertEqual(result.mode, "DC")
        self.assertEqual(result.current_amperes, 0.0)

    def test_rejects_hold_relative_minmax_overload_and_negative_ac(self):
        invalid_frames = (
            (make_frame(flags=0x0A), "hold_mode"),
            (make_frame(flags=0x0C), "relative_mode"),
            (make_frame(status=0x10), "min_mode"),
            (make_frame(status=0x20), "max_mode"),
            (make_frame(digits=b"?0:?"), "overload"),
            (make_frame(sign=b"-"), "negative_ac"),
        )
        for raw, expected_reason in invalid_frames:
            with self.subTest(expected_reason=expected_reason):
                result = decode_frame(raw)
                self.assertFalse(result.valid)
                self.assertEqual(result.reason, expected_reason)
                self.assertIsNone(result.current_amperes)

    def test_auto_poweroff_status_is_metadata_and_does_not_change_current(self):
        raw = make_frame(digits=b"0864", status=0x08)
        result = decode_frame(raw)
        record = measurement_record(
            FrameCandidate(raw, None),
            circuit="gputer",
            voltage_v=116.0,
            voltage_provenance="assumed_default",
            receipt_epoch_ms=1_800_000_000_125,
            receipt_monotonic_ns=987656321,
        )

        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.current_amperes, 0.864)
        self.assertTrue(record["auto_poweroff_active"])
        self.assertTrue(record["valid"])
        self.assertAlmostEqual(record["current_amperes"], 0.864)

    def test_nano_status_is_rejected_and_unknown_status_symbols_stay_rejected(self):
        nano = decode_frame(make_frame(status=0x02))
        unknown_symbol = decode_frame(make_frame(status=0x01))

        self.assertFalse(nano.valid)
        self.assertEqual(nano.reason, "unsupported_prefix_or_function")
        self.assertIsNone(nano.current_amperes)
        self.assertFalse(unknown_symbol.valid)
        self.assertEqual(unknown_symbol.reason, "reserved_status_flags")

    def test_rejects_ambiguous_modes_prefixes_and_non_current_ranges(self):
        invalid_frames = (
            make_frame(flags=0x18),
            make_frame(flags=0),
            make_frame(prefix=0xC0),
            make_frame(prefix=0x20),
            make_frame(unit=0),
        )
        for raw in invalid_frames:
            with self.subTest(raw=raw):
                result = decode_frame(raw)
                self.assertFalse(result.valid)
                self.assertIsNone(result.current_amperes)

    def test_rejects_malformed_frames_and_preserves_mode_when_known(self):
        malformed = decode_frame(make_frame(digits=b"12x4"))
        self.assertFalse(malformed.valid)
        self.assertEqual(malformed.mode, "AC")
        self.assertEqual(malformed.reason, "digits")
        self.assertFalse(decode_frame(make_frame()[:-2]).valid)

    def test_framer_recovers_valid_frames_after_junk_and_partial_chunks(self):
        raw = make_frame()
        framer = FrameFramer()
        self.assertEqual(framer.feed(b"junk"), [])
        first = framer.feed(raw[:6])
        self.assertEqual(first, [])
        second = framer.feed(raw[6:])

        self.assertEqual(len(second), 2)
        self.assertEqual(second[0].error, "resync_junk")
        self.assertEqual(second[0].raw, b"junk")
        self.assertIsNone(second[1].error)
        self.assertEqual(second[1].raw, raw)
        self.assertEqual(framer.finish(), [])

    def test_framer_reports_trailing_partial_input_without_unbounded_buffer(self):
        framer = FrameFramer(max_junk_bytes=16)
        events = framer.feed(b"x" * 100 + b"+12")
        tail = framer.finish()
        self.assertLessEqual(framer.buffered_bytes, 16 + 13)
        self.assertTrue(events)
        self.assertTrue(all(event.error == "resync_junk" for event in events))
        self.assertTrue(tail[-1].raw.endswith(b"+12"))
        self.assertLessEqual(len(tail[-1].raw), 13)
        self.assertEqual(tail[-1].error, "partial_frame_at_end")

    def test_json_record_keeps_receipt_clocks_and_omits_current_on_rejection(self):
        valid = measurement_record(
            FrameCandidate(make_frame(), None),
            circuit="gputer",
            voltage_v=116.0,
            voltage_provenance="assumed_default",
            receipt_epoch_ms=1_800_000_000_123,
            receipt_monotonic_ns=987654321,
        )
        rejected = measurement_record(
            FrameCandidate(make_frame(flags=0x0A), None),
            circuit="gputer",
            voltage_v=116.0,
            voltage_provenance="assumed_default",
            receipt_epoch_ms=1_800_000_000_124,
            receipt_monotonic_ns=987655321,
        )
        self.assertEqual(valid["receipt_epoch_ms"], 1_800_000_000_123)
        self.assertEqual(valid["receipt_monotonic_ns"], 987654321)
        self.assertEqual(valid["mode"], "AC")
        self.assertAlmostEqual(valid["current_amperes"], 1.234)
        self.assertEqual(valid["voltage_provenance"], "assumed_default")
        self.assertEqual(valid["raw_frame_hex"], make_frame().hex())
        self.assertFalse(rejected["valid"])
        self.assertEqual(rejected["reason"], "hold_mode")
        self.assertNotIn("current_amperes", rejected)

    def test_capture_duration_and_voltage_are_bounded(self):
        validate_capture_args(duration_s=1.0, voltage_v=116.0)
        for duration in (0, -1, 300.001):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                validate_capture_args(duration_s=duration, voltage_v=116.0)
        with self.assertRaises(ValueError):
            validate_capture_args(duration_s=1, voltage_v=0)

    def test_bounded_capture_writes_valid_jsonl_and_refuses_overwrite(self):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, make_frame())
        os.close(write_fd)
        try:
            with tempfile.TemporaryDirectory() as directory:
                output_path = Path(directory) / "capture.jsonl"
                with patch("ut61b_telemetry.open_serial_port", return_value=read_fd):
                    result = _capture("synthetic", 0.01, output_path, "gputer", 116.0)
                rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(result, 0)
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]["valid"])
                self.assertAlmostEqual(rows[0]["current_amperes"], 1.234)
                with self.assertRaises(FileExistsError):
                    _capture("synthetic", 0.01, output_path, "gputer", 116.0)
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    def test_bounded_capture_with_no_valid_frames_retains_rejection_and_returns_nonzero(self):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"unrecognized bytes")
        os.close(write_fd)
        try:
            with tempfile.TemporaryDirectory() as directory:
                output_path = Path(directory) / "capture.jsonl"
                stderr = io.StringIO()
                with patch("ut61b_telemetry.open_serial_port", return_value=read_fd), contextlib.redirect_stderr(stderr):
                    result = _capture("synthetic", 0.01, output_path, "gputer", 116.0)
                rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(result, 2)
                self.assertTrue(rows)
                self.assertTrue(all(not row["valid"] for row in rows))
                self.assertTrue(all(row["reason"] for row in rows))
                self.assertTrue(all("current_amperes" not in row for row in rows))
                self.assertIn("no valid UT61B current frames", stderr.getvalue())
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
