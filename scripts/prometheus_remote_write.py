#!/usr/bin/env python3
"""Dependency-free Prometheus remote-write v1 sender for UT61B JSONL reports."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, TextIO


LABEL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
METRIC_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
DEFAULT_BATCH_SAMPLES = 500
MAX_BATCH_SAMPLES = 10_000
MAX_ATTEMPTS = 6
MAX_INPUT_ROWS = 200_000
MAX_INPUT_LINE_BYTES = 65_536


class RemoteWriteError(ValueError):
    """Invalid remote-write input or an exhausted/terminal HTTP write failure."""


@dataclass(frozen=True)
class Sample:
    timestamp_ms: int
    value: float


@dataclass(frozen=True)
class TimeSeries:
    labels: Sequence[tuple[str, str]]
    samples: Sequence[Sample]


@dataclass
class WriteStats:
    sample_count: int = 0
    series_count: int = 0
    batches: int = 0
    attempts: int = 0
    http_success: int = 0
    http_retryable: int = 0
    http_failure: int = 0
    network_errors: int = 0
    failures: int = 0

    @property
    def retry_count(self) -> int:
        return self.http_retryable + self.network_errors


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise RemoteWriteError("negative protobuf varint is not supported")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _length_delimited(field_number: int, payload: bytes) -> bytes:
    return _encode_varint((field_number << 3) | 2) + _encode_varint(len(payload)) + payload


def _normalize_labels(labels: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    try:
        items = list(labels.items()) if hasattr(labels, "items") else list(labels)
    except (TypeError, ValueError) as exc:
        raise RemoteWriteError("labels must be name/value pairs") from exc
    seen: set[str] = set()
    normalized: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise RemoteWriteError("labels must be name/value pairs")
        name, value = item
        if not isinstance(name, str) or not LABEL_NAME_RE.fullmatch(name):
            raise RemoteWriteError(f"invalid label name: {name!r}")
        if name in seen:
            raise RemoteWriteError(f"duplicate label name: {name}")
        if not isinstance(value, str):
            raise RemoteWriteError(f"label {name} value must be a string")
        if "\x00" in value:
            raise RemoteWriteError(f"label {name} contains a NUL byte")
        seen.add(name)
        normalized.append((name, value))
    normalized.sort(key=lambda pair: pair[0])
    metric = dict(normalized).get("__name__")
    if metric is None or not METRIC_NAME_RE.fullmatch(metric):
        raise RemoteWriteError("exactly one valid __name__ metric label is required")
    return tuple(normalized)


def _normalize_samples(samples: Sequence[Sample]) -> tuple[Sample, ...]:
    normalized: list[Sample] = []
    previous: Optional[int] = None
    if not samples:
        raise RemoteWriteError("a time series must contain at least one sample")
    for sample in samples:
        if not isinstance(sample, Sample):
            raise RemoteWriteError("samples must be Sample(timestamp_ms, value) values")
        timestamp = sample.timestamp_ms
        value = sample.value
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0 or timestamp > (2**63 - 1):
            raise RemoteWriteError("sample timestamp must be a nonnegative int64 epoch-millisecond value")
        if previous is not None and timestamp <= previous:
            raise RemoteWriteError("sample timestamps must be strictly increasing within a series")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RemoteWriteError("sample value must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise RemoteWriteError("sample value must be finite")
        normalized.append(Sample(timestamp, numeric_value))
        previous = timestamp
    return tuple(normalized)


def _normalize_series(series: Iterable[TimeSeries]) -> tuple[TimeSeries, ...]:
    result: list[TimeSeries] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for item in series:
        if not isinstance(item, TimeSeries):
            raise RemoteWriteError("series must contain TimeSeries values")
        labels = _normalize_labels(item.labels)
        if labels in seen:
            raise RemoteWriteError("duplicate label set in one WriteRequest")
        seen.add(labels)
        result.append(TimeSeries(labels, _normalize_samples(item.samples)))
    if not result:
        raise RemoteWriteError("no time series to write")
    return tuple(result)


def encode_write_request(series: Iterable[TimeSeries]) -> bytes:
    """Encode a Prometheus WriteRequest protobuf with canonical label ordering."""
    normalized = _normalize_series(series)
    request = bytearray()
    for item in normalized:
        encoded_series = bytearray()
        for name, value in item.labels:
            label = _length_delimited(1, name.encode("utf-8")) + _length_delimited(2, value.encode("utf-8"))
            encoded_series.extend(_length_delimited(1, label))
        for sample in item.samples:
            # prometheus.Sample.value is fixed64 field 1; timestamp is int64 field 2.
            encoded_sample = b"\x09" + struct.pack("<d", sample.value)
            encoded_sample += _encode_varint(0x10) + _encode_varint(sample.timestamp_ms)
            encoded_series.extend(_length_delimited(2, encoded_sample))
        request.extend(_length_delimited(1, bytes(encoded_series)))
    return bytes(request)


def snappy_compress_raw(data: bytes) -> bytes:
    """Emit a valid raw Snappy block using literal chunks only (no Snappy dependency)."""
    output = bytearray(_encode_varint(len(data)))
    max_literal = 65_536
    for start in range(0, len(data), max_literal):
        literal = data[start : start + max_literal]
        length_minus_one = len(literal) - 1
        if length_minus_one < 60:
            output.append(length_minus_one << 2)
        else:
            width = max(1, (length_minus_one.bit_length() + 7) // 8)
            if width > 4:
                raise RemoteWriteError("raw Snappy literal exceeds the supported length")
            output.append((59 + width) << 2)
            output.extend(length_minus_one.to_bytes(width, "little"))
        output.extend(literal)
    return bytes(output)


def _input_lines(source: str | TextIO | Iterable[str]) -> Iterable[str]:
    if isinstance(source, str):
        return source.splitlines()
    return source


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemoteWriteError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RemoteWriteError(f"{field_name} must be finite")
    return result


def _timestamp(row: dict, line_number: int) -> int:
    value = row.get("receipt_epoch_ms")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > (2**63 - 1):
        raise RemoteWriteError(f"line {line_number}: receipt_epoch_ms must be a nonnegative int64")
    return value


def parse_input_jsonl(source: str | TextIO | Iterable[str]) -> tuple[tuple[TimeSeries, ...], int]:
    """Map collector rows to distinct UT61B metric series; coalesce same-ms points."""
    values_by_series: dict[tuple[tuple[str, str], ...], list[Sample]] = {}
    coalesced = 0
    row_count = 0

    def add(labels: tuple[tuple[str, str], ...], timestamp: int, value: float, line_number: int) -> None:
        nonlocal coalesced
        samples = values_by_series.setdefault(labels, [])
        if samples and timestamp < samples[-1].timestamp_ms:
            raise RemoteWriteError(f"line {line_number}: timestamp regressed within a metric series")
        sample = Sample(timestamp, value)
        if samples and timestamp == samples[-1].timestamp_ms:
            samples[-1] = sample
            coalesced += 1
        else:
            samples.append(sample)

    for line_number, line in enumerate(_input_lines(source), start=1):
        if len(line.encode("utf-8")) > MAX_INPUT_LINE_BYTES:
            raise RemoteWriteError(f"line {line_number}: JSONL row exceeds {MAX_INPUT_LINE_BYTES} bytes")
        if not line.strip():
            continue
        row_count += 1
        if row_count > MAX_INPUT_ROWS:
            raise RemoteWriteError(f"input exceeds the {MAX_INPUT_ROWS} row safety bound")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RemoteWriteError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise RemoteWriteError(f"line {line_number}: each JSONL row must be an object")
        valid = row.get("valid")
        if not isinstance(valid, bool):
            raise RemoteWriteError(f"line {line_number}: valid must be a boolean")
        mode = row.get("mode")
        if mode not in ("AC", "DC", "unknown"):
            raise RemoteWriteError(f"line {line_number}: mode must be AC, DC, or unknown")
        circuit = row.get("circuit")
        if not isinstance(circuit, str) or not circuit:
            raise RemoteWriteError(f"line {line_number}: circuit must be a nonempty string")
        timestamp = _timestamp(row, line_number)
        common = {
            "circuit": circuit,
            "meter": "ut61b",
            "mode": mode.lower(),
        }
        validity_labels = tuple(sorted({"__name__": "ut61b_measurement_valid", **common}.items()))
        add(validity_labels, timestamp, 1.0 if valid else 0.0, line_number)

        if not valid:
            if "current_amperes" in row:
                raise RemoteWriteError(f"line {line_number}: invalid rows must not contain current_amperes")
            continue
        if mode not in ("AC", "DC"):
            raise RemoteWriteError(f"line {line_number}: a valid current row must have AC or DC mode")
        current = _number(row.get("current_amperes"), f"line {line_number} current_amperes")
        if mode == "AC" and current < 0:
            raise RemoteWriteError(f"line {line_number}: negative AC current is invalid")
        current_labels = tuple(sorted({"__name__": "ut61b_current_amperes", **common}.items()))
        add(current_labels, timestamp, current, line_number)

        if mode == "AC":
            voltage = _number(row.get("voltage_assumption_v"), f"line {line_number} voltage_assumption_v")
            provenance = row.get("voltage_provenance")
            if voltage <= 0:
                raise RemoteWriteError(f"line {line_number}: voltage assumption must be positive")
            if not isinstance(provenance, str) or not provenance:
                raise RemoteWriteError(f"line {line_number}: voltage_provenance must be a nonempty string")
            apparent_labels = tuple(
                sorted(
                    {
                        "__name__": "ut61b_apparent_power_va",
                        **common,
                        "voltage_assumption_v": format(voltage, ".17g"),
                        "voltage_source": "assumed",
                        "voltage_provenance": provenance,
                    }.items()
                )
            )
            add(apparent_labels, timestamp, voltage * current, line_number)

    if not values_by_series:
        raise RemoteWriteError("input contains no measurement rows")
    series = tuple(TimeSeries(labels, tuple(samples)) for labels, samples in sorted(values_by_series.items()))
    return series, coalesced


def _remote_write_endpoint(base_url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(base_url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise RemoteWriteError("--vm-url must be a valid loopback HTTP URL") from exc
    if parsed.scheme != "http" or not host or parsed.username is not None or parsed.password is not None:
        raise RemoteWriteError("--vm-url must be plain HTTP on loopback and contain no credentials")
    if parsed.query or parsed.fragment:
        raise RemoteWriteError("--vm-url must not contain a query or fragment")
    try:
        loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise RemoteWriteError("--vm-url host must be localhost or a loopback IP address")
    if parsed.path not in ("", "/", "/api/v1/write"):
        raise RemoteWriteError("--vm-url path must be empty, /, or /api/v1/write")
    netloc = parsed.netloc
    path = parsed.path.rstrip("/") or "/api/v1/write"
    return urllib.parse.urlunsplit(("http", netloc, path, "", ""))


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RemoteWriteClient:
    """Bounded retry sender for Prometheus remote-write v1 to loopback HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        max_attempts: int = 4,
        timeout_s: float = 10.0,
        backoff_base_s: float = 0.2,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= MAX_ATTEMPTS:
            raise RemoteWriteError(f"max_attempts must be between 1 and {MAX_ATTEMPTS}")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise RemoteWriteError("timeout_s must be finite and positive")
        if not math.isfinite(backoff_base_s) or backoff_base_s < 0:
            raise RemoteWriteError("backoff_base_s must be finite and nonnegative")
        self.endpoint = _remote_write_endpoint(base_url)
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self.backoff_base_s = backoff_base_s
        self.sleeper = sleeper
        self.opener = urllib.request.build_opener(_NoRedirectHandler()).open
        self.stats = WriteStats()

    def _post(self, body: bytes) -> None:
        headers = {
            "Content-Encoding": "snappy",
            "Content-Type": "application/x-protobuf",
            "User-Agent": "cogni-ml/1.0",
            "X-Prometheus-Remote-Write-Version": "0.1.0",
        }
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        for attempt in range(1, self.max_attempts + 1):
            self.stats.attempts += 1
            try:
                response = self.opener(request, timeout=self.timeout_s)
                with response:
                    status = response.getcode()
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status in (429,) or 500 <= status <= 599:
                    self.stats.http_retryable += 1
                    if attempt < self.max_attempts:
                        self.sleeper(min(self.backoff_base_s * (2 ** (attempt - 1)), 2.0))
                        continue
                self.stats.http_failure += 1
                raise RemoteWriteError(f"remote write failed with HTTP {status}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.stats.network_errors += 1
                if attempt < self.max_attempts:
                    self.sleeper(min(self.backoff_base_s * (2 ** (attempt - 1)), 2.0))
                    continue
                raise RemoteWriteError(f"remote write network failure after {attempt} attempts") from None

            if 200 <= status < 300:
                self.stats.http_success += 1
                return
            if status == 429 or 500 <= status <= 599:
                self.stats.http_retryable += 1
                if attempt < self.max_attempts:
                    self.sleeper(min(self.backoff_base_s * (2 ** (attempt - 1)), 2.0))
                    continue
            self.stats.http_failure += 1
            raise RemoteWriteError(f"remote write failed with HTTP {status}")

    def write(self, series: Iterable[TimeSeries], *, max_samples_per_batch: int = DEFAULT_BATCH_SAMPLES) -> WriteStats:
        if isinstance(max_samples_per_batch, bool) or not isinstance(max_samples_per_batch, int):
            raise RemoteWriteError("max_samples_per_batch must be an integer")
        if not 1 <= max_samples_per_batch <= MAX_BATCH_SAMPLES:
            raise RemoteWriteError(f"max_samples_per_batch must be between 1 and {MAX_BATCH_SAMPLES}")
        normalized = _normalize_series(series)
        self.stats = WriteStats(
            sample_count=sum(len(item.samples) for item in normalized),
            series_count=len(normalized),
        )

        batch: list[TimeSeries] = []
        batch_size = 0

        def flush() -> None:
            nonlocal batch, batch_size
            if not batch:
                return
            payload = snappy_compress_raw(encode_write_request(batch))
            self.stats.batches += 1
            try:
                self._post(payload)
            except RemoteWriteError:
                self.stats.failures += 1
                raise
            batch = []
            batch_size = 0

        for item in normalized:
            offset = 0
            samples = item.samples
            while offset < len(samples):
                if batch_size == max_samples_per_batch:
                    flush()
                available = max_samples_per_batch - batch_size
                part = tuple(samples[offset : offset + available])
                batch.append(TimeSeries(item.labels, part))
                batch_size += len(part)
                offset += len(part)
        flush()
        return self.stats


def _summary(stats: WriteStats, coalesced: int) -> str:
    return (
        f"samples={stats.sample_count} series={stats.series_count} batches={stats.batches} "
        f"attempts={stats.attempts} http_success={stats.http_success} "
        f"http_failure={stats.http_failure} retryable_http={stats.http_retryable} "
        f"network_errors={stats.network_errors} "
        f"coalesced_same_ms={coalesced} failures={stats.failures}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UT61B collector JSONL report")
    parser.add_argument("--vm-url", required=True, help="loopback VictoriaMetrics base URL, such as http://127.0.0.1:8428")
    parser.add_argument("--batch-samples", type=int, default=DEFAULT_BATCH_SAMPLES, help="maximum samples per HTTP request")
    parser.add_argument("--max-attempts", type=int, default=4, help=f"bounded attempts per batch (maximum {MAX_ATTEMPTS})")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    coalesced = 0
    stats = WriteStats()
    try:
        with args.input.open("r", encoding="utf-8") as source:
            series, coalesced = parse_input_jsonl(source)
        client = RemoteWriteClient(args.vm_url, max_attempts=args.max_attempts)
        stats = client.write(series, max_samples_per_batch=args.batch_samples)
        print(f"remote_write {_summary(stats, coalesced)}")
        return 0
    except (OSError, RemoteWriteError) as exc:
        if "client" in locals():
            stats = client.stats
        stats.failures = max(stats.failures, 1)
        print(f"remote_write failed: {exc}; {_summary(stats, coalesced)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
