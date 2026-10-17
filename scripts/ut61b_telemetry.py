#!/usr/bin/env python3
"""Bounded, dependency-free telemetry capture for a UNI-T UT61B current range."""

from __future__ import annotations

import argparse
import array
import fcntl
import json
import math
import os
import select
import sys
import termios
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BAUD = 2400
FRAME_SIZE = 14
DECIMAL_EXPONENTS = {
    ord("0"): 0,
    ord("1"): -3,
    ord("2"): -2,
    ord("4"): -1,
}
PREFIX_BITS = {0: ("A", 1.0), 0x40: ("mA", 1e-3), 0x80: ("uA", 1e-6)}
MAX_CAPTURE_SECONDS = 300.0


class TelemetryError(ValueError):
    """A capture or frame-contract error that should be shown without a traceback."""


@dataclass(frozen=True)
class Measurement:
    valid: bool
    mode: str
    current_amperes: Optional[float] = None
    unit_prefix: Optional[str] = None
    low_battery: Optional[bool] = None
    auto_poweroff_active: Optional[bool] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class FrameCandidate:
    raw: bytes
    error: Optional[str] = None


class FrameFramer:
    """Frame CRLF-delimited packets while retaining at most 13 bytes plus bounded junk."""

    def __init__(self, max_junk_bytes: int = 256):
        if max_junk_bytes < 1:
            raise ValueError("max_junk_bytes must be positive")
        self._buffer = bytearray()
        self._junk = bytearray()
        self._max_junk_bytes = max_junk_bytes

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer) + len(self._junk)

    def _flush_junk(self, output: list[FrameCandidate], *, all_bytes: bool = False) -> None:
        while len(self._junk) >= self._max_junk_bytes or (all_bytes and self._junk):
            take = len(self._junk) if all_bytes else self._max_junk_bytes
            raw = bytes(self._junk[:take])
            del self._junk[:take]
            output.append(FrameCandidate(raw, "resync_junk"))

    def feed(self, data: bytes) -> list[FrameCandidate]:
        output: list[FrameCandidate] = []
        for byte in data:
            self._buffer.append(byte)
            if len(self._buffer) >= 2 and self._buffer[-2:] == b"\r\n":
                self._flush_junk(output, all_bytes=True)
                raw = bytes(self._buffer)
                self._buffer.clear()
                output.append(FrameCandidate(raw, None if len(raw) == FRAME_SIZE else "frame_length"))
                continue

            if len(self._buffer) > FRAME_SIZE - 1:
                excess = len(self._buffer) - (FRAME_SIZE - 1)
                self._junk.extend(self._buffer[:excess])
                del self._buffer[:excess]
                self._flush_junk(output)
        return output

    def finish(self) -> list[FrameCandidate]:
        output: list[FrameCandidate] = []
        self._flush_junk(output, all_bytes=True)
        if self._buffer:
            output.append(FrameCandidate(bytes(self._buffer), "partial_frame_at_end"))
            self._buffer.clear()
        return output


def _mode_from_flags(flags: int) -> str:
    ac = bool(flags & 0x08)
    dc = bool(flags & 0x10)
    if ac == dc:
        return "unknown"
    return "AC" if ac else "DC"


def _invalid(
    reason: str,
    mode: str,
    low_battery: Optional[bool],
    auto_poweroff_active: Optional[bool],
) -> Measurement:
    return Measurement(
        False,
        mode,
        low_battery=low_battery,
        auto_poweroff_active=auto_poweroff_active,
        reason=reason,
    )


def decode_frame(raw: bytes) -> Measurement:
    """Decode an FS9922 UT61B current frame into amperes, failing closed on ambiguity."""
    mode = _mode_from_flags(raw[7]) if len(raw) > 7 else "unknown"
    low_battery = bool(raw[8] & 0x04) if len(raw) > 8 else None
    auto_poweroff_active = bool(raw[8] & 0x08) if len(raw) > 8 else None

    if len(raw) != FRAME_SIZE:
        return _invalid("frame_length", mode, low_battery, auto_poweroff_active)
    if raw[12:14] != b"\r\n":
        return _invalid("frame_terminator", mode, low_battery, auto_poweroff_active)
    if raw[0] not in (ord("+"), ord("-")):
        return _invalid("sign", mode, low_battery, auto_poweroff_active)
    if raw[1:5] == b"?0:?":
        return _invalid("overload", mode, low_battery, auto_poweroff_active)
    if raw[5] != 0x20:
        return _invalid("separator", mode, low_battery, auto_poweroff_active)
    if not all(ord("0") <= digit <= ord("9") for digit in raw[1:5]):
        return _invalid("digits", mode, low_battery, auto_poweroff_active)

    exponent = DECIMAL_EXPONENTS.get(raw[6])
    if exponent is None:
        return _invalid("decimal_exponent", mode, low_battery, auto_poweroff_active)

    flags = raw[7]
    if flags & 0x02:
        return _invalid("hold_mode", mode, low_battery, auto_poweroff_active)
    if flags & 0x04:
        return _invalid("relative_mode", mode, low_battery, auto_poweroff_active)
    # Bit 0 reports whether the display bargraph is active; it does not
    # change the numeric reading. Bits 1..5 are HOLD, REL, AC, DC, and auto.
    if flags & ~0x3F:
        return _invalid("reserved_mode_flags", mode, low_battery, auto_poweroff_active)

    status = raw[8]
    if status & 0x02:
        return _invalid("unsupported_prefix_or_function", mode, low_battery, auto_poweroff_active)
    if status & 0x10:
        return _invalid("min_mode", mode, low_battery, auto_poweroff_active)
    if status & 0x20:
        return _invalid("max_mode", mode, low_battery, auto_poweroff_active)
    # Bit 1 is the nano multiplier (rejected above); bit 3 is APO metadata.
    if status & ~0x3E:
        return _invalid("reserved_status_flags", mode, low_battery, auto_poweroff_active)

    prefix_flags = raw[9]
    if prefix_flags & 0x3F:
        return _invalid("unsupported_prefix_or_function", mode, low_battery, auto_poweroff_active)
    scale_flags = prefix_flags & 0xC0
    if scale_flags == 0xC0:
        return _invalid("ambiguous_prefix", mode, low_battery, auto_poweroff_active)
    prefix = PREFIX_BITS.get(scale_flags)
    if prefix is None:
        return _invalid("unsupported_prefix_or_function", mode, low_battery, auto_poweroff_active)
    if raw[10] != 0x40:
        return _invalid("non_current_mode", mode, low_battery, auto_poweroff_active)
    if mode == "unknown":
        return _invalid("ambiguous_mode", mode, low_battery, auto_poweroff_active)

    value = int(raw[1:5]) * (10.0**exponent) * prefix[1]
    if raw[0] == ord("-"):
        value = -value
    if mode == "AC" and value < 0:
        return _invalid("negative_ac", mode, low_battery, auto_poweroff_active)
    if not math.isfinite(value):
        return _invalid("non_finite_current", mode, low_battery, auto_poweroff_active)

    return Measurement(
        True,
        mode,
        current_amperes=value,
        unit_prefix=prefix[0],
        low_battery=low_battery,
        auto_poweroff_active=auto_poweroff_active,
    )


def measurement_record(
    candidate: FrameCandidate,
    *,
    circuit: str,
    voltage_v: float,
    voltage_provenance: str,
    receipt_epoch_ms: int,
    receipt_monotonic_ns: int,
) -> dict:
    measurement = decode_frame(candidate.raw)
    reason = candidate.error or measurement.reason
    valid = candidate.error is None and measurement.valid
    record = {
        "schema_version": 1,
        "source": "ut61b",
        "circuit": circuit,
        "raw_frame_hex": candidate.raw.hex(),
        "receipt_epoch_ms": int(receipt_epoch_ms),
        "receipt_monotonic_ns": int(receipt_monotonic_ns),
        "valid": valid,
        "mode": measurement.mode,
        "reason": None if valid else (reason or "invalid_frame"),
        "voltage_assumption_v": voltage_v,
        "voltage_provenance": voltage_provenance,
        "low_battery": measurement.low_battery,
        "auto_poweroff_active": measurement.auto_poweroff_active,
    }
    if valid:
        record["current_amperes"] = measurement.current_amperes
        record["unit_prefix"] = measurement.unit_prefix
    return record


def validate_capture_args(duration_s: float, voltage_v: float) -> None:
    if not math.isfinite(duration_s) or duration_s <= 0 or duration_s > MAX_CAPTURE_SECONDS:
        raise ValueError(f"duration must be greater than 0 and at most {MAX_CAPTURE_SECONDS:g} seconds")
    if not math.isfinite(voltage_v) or voltage_v <= 0:
        raise ValueError("voltage must be a finite positive assumption in volts")


def set_modem_lines(fd: int) -> None:
    """Set DTR high and RTS low, preserving any other modem-control lines."""
    required = ("TIOCMGET", "TIOCMSET", "TIOCM_DTR", "TIOCM_RTS")
    if not all(hasattr(termios, name) for name in required):
        raise TelemetryError("this POSIX serial driver cannot expose DTR/RTS controls")
    state = array.array("i", [0])
    try:
        fcntl.ioctl(fd, termios.TIOCMGET, state, True)
        state[0] = (state[0] & ~(termios.TIOCM_DTR | termios.TIOCM_RTS)) | termios.TIOCM_DTR
        fcntl.ioctl(fd, termios.TIOCMSET, state)
    except OSError as exc:
        raise TelemetryError(f"could not set serial DTR=1 RTS=0: {exc.strerror or exc}") from exc


def open_serial_port(port: str) -> int:
    flags = os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK
    fd = os.open(port, flags)
    try:
        attributes = termios.tcgetattr(fd)
        cflag = attributes[2]
        clear_mask = termios.CSIZE | termios.PARENB | termios.CSTOPB | getattr(termios, "CRTSCTS", 0)
        attributes[0] = 0  # No software flow control or input translations.
        attributes[1] = 0
        attributes[2] = (cflag & ~clear_mask) | termios.CS8 | termios.CREAD | termios.CLOCAL
        attributes[3] = 0
        attributes[4] = termios.B2400
        attributes[5] = termios.B2400
        attributes[6][termios.VMIN] = 0
        attributes[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attributes)
        set_modem_lines(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        return fd
    except Exception:
        os.close(fd)
        raise


def _capture(port: str, duration_s: float, output_path: Path, circuit: str, voltage_v: float) -> int:
    validate_capture_args(duration_s, voltage_v)
    if not circuit.strip():
        raise TelemetryError("circuit label must not be empty")
    provenance = "assumed_default" if voltage_v == 116.0 else "assumed_cli"
    valid_count = 0
    rejected_count = 0

    # Exclusive creation prevents silently appending to or replacing prior evidence.
    with output_path.open("x", encoding="utf-8", newline="\n") as report:
        fd = open_serial_port(port)
        framer = FrameFramer()
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + int(duration_s * 1_000_000_000)

        def record(candidate: FrameCandidate) -> None:
            nonlocal valid_count, rejected_count
            row = measurement_record(
                candidate,
                circuit=circuit,
                voltage_v=voltage_v,
                voltage_provenance=provenance,
                receipt_epoch_ms=time.time_ns() // 1_000_000,
                receipt_monotonic_ns=time.monotonic_ns(),
            )
            report.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            report.flush()
            if row["valid"]:
                valid_count += 1
            else:
                rejected_count += 1

        try:
            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                readable, _, _ = select.select([fd], [], [], remaining_ns / 1_000_000_000)
                if not readable:
                    break
                try:
                    data = os.read(fd, 256)
                except BlockingIOError:
                    continue
                if data:
                    for candidate in framer.feed(data):
                        record(candidate)
            for candidate in framer.finish():
                record(candidate)
        finally:
            os.close(fd)

    print(f"valid_frames={valid_count} rejected_frames={rejected_count} output={output_path}")
    if valid_count == 0:
        print("capture failed: no valid UT61B current frames were received", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="capture UT61B current frames for a bounded duration")
    capture.add_argument("--port", required=True, help="POSIX serial device path")
    capture.add_argument("--duration", type=float, required=True, help="capture duration in seconds (maximum 300)")
    capture.add_argument("--output", type=Path, required=True, help="new JSONL report path; existing files are rejected")
    capture.add_argument("--circuit", default="gputer", help="stable circuit label (default: gputer)")
    capture.add_argument("--voltage", type=float, default=116.0, help="assumed AC voltage in volts (default: 116)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _capture(args.port, args.duration, args.output, args.circuit, args.voltage)
    except (TelemetryError, OSError, termios.error, ValueError) as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
