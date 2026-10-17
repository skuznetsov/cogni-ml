#!/usr/bin/env python3
"""Capture bounded NVIDIA GPU telemetry as JSONL, with optional local VM import.

The power source is NVIDIA's ``power.draw.instant`` query field. The selected
field and the raw nvidia-smi timestamp are recorded in the session and each
sample. Utilization is the driver's GPU utilization percentage. There is no
silent substitution if a driver does not support that field.

The collector uses nvidia-smi's own 250 ms sampling loop and timestamps each
received row with both UTC wall time and a monotonic clock. Missing, malformed,
or non-finite readings are represented as null with a field-specific error;
they are never converted into zero. Optional VictoriaMetrics import sends
only finite observed values to a loopback ``/api/v1/import`` endpoint.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import select
import subprocess
import sys
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable, Iterable, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SCHEMA = "cogni.gemma4.gpu-telemetry.v1"
DEFAULT_INTERVAL_MS = 250
MAX_DURATION_SECONDS = 300
POWER_FIELD = "power.draw.instant"
QUERY_PREFIX_FIELDS = ("timestamp", "name", "uuid", "driver_version")
QUERY_SUFFIX_FIELDS = ("utilization.gpu", "memory.used", "memory.total")
METRICS = (
    ("cogni_gpu_power_watts", "power_watts"),
    ("cogni_gpu_utilization_percent", "utilization_percent"),
    ("cogni_gpu_memory_used_bytes", "memory_used_bytes"),
)
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class TelemetryError(ValueError):
    """Raised for invalid collector options or unusable telemetry data."""


def _stamp(
    wall_time: datetime | None = None,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    wall_time = wall_time or datetime.now(timezone.utc)
    if wall_time.tzinfo is None:
        wall_time = wall_time.replace(tzinfo=timezone.utc)
    wall_time = wall_time.astimezone(timezone.utc)
    return {
        "timestamp_utc": wall_time.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "timestamp_unix_ms": int(wall_time.timestamp() * 1000),
        "monotonic_ns": time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
    }


def _numeric(raw: str, field: str) -> tuple[float | None, str | None]:
    value = raw.strip()
    if not value or value.casefold() in {"n/a", "na", "[not supported]", "not supported"}:
        return None, f"{field} unavailable ({value or 'empty'})"
    try:
        number = float(value)
    except ValueError:
        return None, f"{field} is not numeric ({value!r})"
    if not math.isfinite(number):
        return None, f"{field} is non-finite ({value!r})"
    if number < 0:
        return None, f"{field} is negative ({value!r})"
    return number, None


def parse_sample_line(
    line: str,
    *,
    run_label: str,
    host: str,
    gpu_index: int,
    wall_time: datetime | None = None,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Parse one CSV row from the fixed nvidia-smi query into a JSON record."""
    values = next(csv.reader([line], skipinitialspace=True), [])
    expected = len(QUERY_PREFIX_FIELDS) + 1 + len(QUERY_SUFFIX_FIELDS)
    common = {
        "schema": SCHEMA,
        "record_type": "sample",
        "run": run_label,
        "host": host,
        "gpu": str(gpu_index),
        "gpu_index": gpu_index,
        "power_source_field": POWER_FIELD,
        "power_source_semantics": "instantaneous board power reported by nvidia-smi",
        **_stamp(wall_time, monotonic_ns),
    }
    if len(values) != expected:
        common.update(
            {
                "sample_status": "error",
                "sample_errors": [f"expected {expected} CSV columns; received {len(values)}"],
                "nvidia_smi_line": line.rstrip("\r\n"),
            }
        )
        return common

    row = dict(zip((*QUERY_PREFIX_FIELDS, POWER_FIELD, *QUERY_SUFFIX_FIELDS), (item.strip() for item in values)))
    power, power_error = _numeric(row[POWER_FIELD], POWER_FIELD)
    utilization, utilization_error = _numeric(row["utilization.gpu"], "utilization.gpu")
    used_mib, used_error = _numeric(row["memory.used"], "memory.used")
    total_mib, total_error = _numeric(row["memory.total"], "memory.total")

    errors = [error for error in (power_error, utilization_error, used_error, total_error) if error]
    if utilization is not None and utilization > 100:
        errors.append("utilization.gpu is above 100 percent")
        utilization = None

    common.update(
        {
            "sample_status": "ok" if not errors else ("partial" if any(value is not None for value in (power, utilization, used_mib, total_mib)) else "error"),
            "sample_errors": errors,
            "nvidia_timestamp": row["timestamp"],
            "gpu_name": row["name"],
            "gpu_uuid": row["uuid"],
            "driver_version": row["driver_version"],
            "power_watts": power,
            "utilization_percent": utilization,
            "memory_used_mib": used_mib,
            "memory_total_mib": total_mib,
            "memory_used_bytes": round(used_mib * 1024 * 1024) if used_mib is not None else None,
            "memory_total_bytes": round(total_mib * 1024 * 1024) if total_mib is not None else None,
        }
    )
    return common


def _event(event: str, *, run_label: str, host: str, gpu_index: int, **fields: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "record_type": "session",
        "event": event,
        "run": run_label,
        "host": host,
        "gpu": str(gpu_index),
        "gpu_index": gpu_index,
        **_stamp(),
        **fields,
    }


def _write_jsonl(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def _nvidia_smi_command(binary: str, gpu_index: int, interval_ms: int) -> list[str]:
    fields = (*QUERY_PREFIX_FIELDS, POWER_FIELD, *QUERY_SUFFIX_FIELDS)
    return [
        binary,
        "--id",
        str(gpu_index),
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
        "--loop-ms",
        str(interval_ms),
    ]


def capture(
    *,
    run_label: str,
    host: str,
    gpu_index: int,
    duration_seconds: float,
    interval_ms: int,
    output: TextIO,
    nvidia_smi: str = "nvidia-smi",
    sample_sink: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, int]:
    """Write a session and bounded sample stream. Returns (sample, error) counts."""
    _write_jsonl(
        output,
        _event(
            "start",
            run_label=run_label,
            host=host,
            gpu_index=gpu_index,
            duration_seconds=duration_seconds,
            sampling_interval_ms=interval_ms,
            query_fields=[*QUERY_PREFIX_FIELDS, POWER_FIELD, *QUERY_SUFFIX_FIELDS],
            power_source_field=POWER_FIELD,
            power_source_semantics="instantaneous board power reported by nvidia-smi",
            sampler="nvidia-smi --loop-ms",
            vm_import_mode="live_async_batches" if sample_sink is not None else "disabled",
        ),
    )
    command = _nvidia_smi_command(nvidia_smi, gpu_index, interval_ms)
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    except OSError as exc:
        _write_jsonl(
            output,
            _event("error", run_label=run_label, host=host, gpu_index=gpu_index, error=f"could not start nvidia-smi: {exc}"),
        )
        _write_jsonl(output, _event("end", run_label=run_label, host=host, gpu_index=gpu_index, sample_count=0, error_count=1))
        return 0, 1

    samples = 0
    errors = 0
    deadline = time.monotonic() + duration_seconds
    pending = bytearray()
    try:
        assert process.stdout is not None
        descriptor = process.stdout.fileno()
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([process.stdout], [], [], min(remaining, 0.5))
            if not readable:
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            pending.extend(chunk)
            while b"\n" in pending:
                raw, _, tail = pending.partition(b"\n")
                pending = bytearray(tail)
                if time.monotonic() >= deadline:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r")
                record = parse_sample_line(
                    line,
                    run_label=run_label,
                    host=host,
                    gpu_index=gpu_index,
                )
                _write_jsonl(output, record)
                samples += 1
                if record["sample_status"] != "ok":
                    errors += 1
                if sample_sink is not None:
                    try:
                        sample_sink(record)
                    except Exception as exc:
                        errors += 1
                        _write_jsonl(
                            output,
                            _event(
                                "error",
                                run_label=run_label,
                                host=host,
                                gpu_index=gpu_index,
                                component="sample_sink",
                                error=f"sample sink failed: {exc}",
                            ),
                        )
        if pending and time.monotonic() < deadline:
            line = pending.decode("utf-8", errors="replace").rstrip("\r")
            if line:
                record = parse_sample_line(line, run_label=run_label, host=host, gpu_index=gpu_index)
                _write_jsonl(output, record)
                samples += 1
                if record["sample_status"] != "ok":
                    errors += 1
                if sample_sink is not None:
                    try:
                        sample_sink(record)
                    except Exception as exc:
                        errors += 1
                        _write_jsonl(
                            output,
                            _event(
                                "error",
                                run_label=run_label,
                                host=host,
                                gpu_index=gpu_index,
                                component="sample_sink",
                                error=f"sample sink failed: {exc}",
                            ),
                        )
        return_code = process.poll()
        if return_code is not None and return_code != 0:
            errors += 1
            _write_jsonl(
                output,
                _event(
                    "error",
                    run_label=run_label,
                    host=host,
                    gpu_index=gpu_index,
                    error=f"nvidia-smi exited with status {return_code}",
                ),
            )
        _write_jsonl(
            output,
            _event(
                "end",
                run_label=run_label,
                host=host,
                gpu_index=gpu_index,
                sample_count=samples,
                error_count=errors,
                process_exit_status=return_code,
            ),
        )
        return samples, errors
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        if process.stdout is not None:
            process.stdout.close()


def vm_import_lines(
    samples: Iterable[dict[str, Any]],
    *,
    run_label: str,
    host: str,
    gpu_index: int,
) -> list[str]:
    """Build VictoriaMetrics JSON-line import objects from finite samples only."""
    series: dict[str, tuple[str, list[Any], list[int]]] = {
        metric: (field, [], []) for metric, field in METRICS
    }
    for sample in samples:
        if sample.get("record_type") != "sample":
            continue
        timestamp = sample.get("timestamp_unix_ms")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            continue
        for metric, (field, values, timestamps) in series.items():
            value = sample.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            values.append(value)
            timestamps.append(timestamp)

    lines: list[str] = []
    for metric, (_, values, timestamps) in series.items():
        if not values:
            continue
        payload = {
            "metric": {
                "__name__": metric,
                "host": host,
                "gpu": str(gpu_index),
                "run": run_label,
            },
            "values": values,
            "timestamps": timestamps,
        }
        lines.append(json.dumps(payload, allow_nan=False, separators=(",", ":")))
    return lines


def _post_vm_samples(
    endpoint: str,
    samples: Iterable[dict[str, Any]],
    *,
    run_label: str,
    host: str,
    gpu_index: int,
    timeout_seconds: float,
) -> int:
    lines = vm_import_lines(samples, run_label=run_label, host=host, gpu_index=gpu_index)
    if not lines:
        return 0
    body = ("\n".join(lines) + "\n").encode("utf-8")
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            if not 200 <= response.status < 300:
                raise TelemetryError(f"VictoriaMetrics import returned HTTP {response.status}")
    except HTTPError as exc:
        raise TelemetryError(f"VictoriaMetrics import returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise TelemetryError(f"VictoriaMetrics import failed: {exc.reason}") from exc
    return len(lines)


class LiveVMImporter:
    """Post sample batches on a worker thread so HTTP latency cannot pause capture."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        run_label: str,
        host: str,
        gpu_index: int,
        flush_interval_seconds: float = 1.0,
        timeout_seconds: float = 5.0,
        queue_capacity: int = 256,
    ) -> None:
        if not math.isfinite(flush_interval_seconds) or flush_interval_seconds <= 0:
            raise TelemetryError("live VM flush interval must be finite and positive")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise TelemetryError("VM import timeout must be finite and positive")
        if queue_capacity < 1:
            raise TelemetryError("live VM queue capacity must be positive")
        self.endpoint_url = endpoint_url
        self.run_label = run_label
        self.host = host
        self.gpu_index = gpu_index
        self.flush_interval_seconds = flush_interval_seconds
        self.timeout_seconds = timeout_seconds
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=queue_capacity)
        self._closing = threading.Event()
        self._disabled = threading.Event()
        self._lock = threading.Lock()
        self._errors: list[str] = []
        self._imported_series = 0
        self._worker = threading.Thread(target=self._run, name="gemma4-vm-import", daemon=True)
        self._worker.start()

    def sample_sink(self, sample: dict[str, Any]) -> None:
        """Queue a sample without waiting for VM I/O; JSONL remains authoritative."""
        if self._disabled.is_set() or self._closing.is_set():
            return
        try:
            self._queue.put_nowait(sample)
        except Full:
            self._mark_error("live VM import queue filled; samples remain in JSONL only")
            self._disabled.set()

    def _mark_error(self, message: str) -> None:
        with self._lock:
            if message not in self._errors:
                self._errors.append(message)

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        if not batch or self._disabled.is_set():
            return
        try:
            imported = _post_vm_samples(
                self.endpoint_url,
                batch,
                run_label=self.run_label,
                host=self.host,
                gpu_index=self.gpu_index,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            self._mark_error(f"live VM import failed; samples remain in JSONL only: {exc}")
            self._disabled.set()
            return
        with self._lock:
            self._imported_series += imported

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()
        while True:
            timeout = max(0.0, self.flush_interval_seconds - (time.monotonic() - last_flush))
            try:
                batch.append(self._queue.get(timeout=timeout))
            except Empty:
                pass
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except Empty:
                    break

            closing_and_drained = self._closing.is_set() and self._queue.empty()
            interval_elapsed = time.monotonic() - last_flush >= self.flush_interval_seconds
            if batch and (closing_and_drained or interval_elapsed):
                self._flush(batch)
                batch.clear()
                last_flush = time.monotonic()
            if closing_and_drained and not batch:
                return
            if self._disabled.is_set():
                batch.clear()

    def close(self) -> tuple[list[str], int]:
        """Flush the final batch and stop waiting at the bounded shutdown deadline."""
        self._closing.set()
        self._worker.join(timeout=self.timeout_seconds + 2.0)
        if self._worker.is_alive():
            self._disabled.set()
            self._mark_error("live VM importer did not drain before its bounded shutdown deadline")
        with self._lock:
            return list(self._errors), self._imported_series


def _read_sample_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TelemetryError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if isinstance(record, dict):
                yield record


def _loopback_import_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise TelemetryError("VM import URL must be an unauthenticated loopback http URL")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise TelemetryError("VM import URL must be a base URL without path, query, or fragment")
    hostname = parsed.hostname.casefold()
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = hostname == "localhost"
    if not is_loopback:
        raise TelemetryError("VM import is restricted to localhost/loopback")
    try:
        port = parsed.port
    except ValueError as exc:
        raise TelemetryError("VM import URL has an invalid port") from exc
    authority = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        authority += f":{port}"
    return f"http://{authority}/api/v1/import"


def import_jsonl_to_vm(
    path: Path,
    *,
    vm_url: str,
    run_label: str,
    host: str,
    gpu_index: int,
    timeout_seconds: float = 5.0,
) -> int:
    """Import sampled metrics to a local VictoriaMetrics instance; return series count."""
    target = _loopback_import_url(vm_url)
    return _post_vm_samples(
        target,
        _read_sample_records(path),
        run_label=run_label,
        host=host,
        gpu_index=gpu_index,
        timeout_seconds=timeout_seconds,
    )


def _label(value: str, name: str) -> str:
    if not LABEL_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{name} must start with a letter or digit and contain only letters, digits, '.', '_' or '-' (max 64 chars)"
        )
    return value


def _duration(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("duration must be numeric") from exc
    if not math.isfinite(number) or number <= 0 or number > MAX_DURATION_SECONDS:
        raise argparse.ArgumentTypeError(f"duration must be > 0 and <= {MAX_DURATION_SECONDS} seconds")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return number


def _interval(value: str) -> int:
    number = _positive_int(value)
    if not 100 <= number <= 5000:
        raise argparse.ArgumentTypeError("sampling interval must be between 100 and 5000 ms")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True, type=lambda value: _label(value, "run label"))
    parser.add_argument("--host", default="gputer", type=lambda value: _label(value, "host"))
    parser.add_argument("--gpu-index", default=0, type=_positive_int)
    parser.add_argument("--duration-seconds", default=60.0, type=_duration)
    parser.add_argument("--interval-ms", default=DEFAULT_INTERVAL_MS, type=_interval)
    parser.add_argument("--output", type=Path, required=True, help="JSONL destination (required to preserve raw samples)")
    parser.add_argument("--nvidia-smi", default="nvidia-smi", help="nvidia-smi executable")
    parser.add_argument(
        "--vm-url",
        help="optional loopback VictoriaMetrics base URL; imports one-second batches plus a final flush",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    live_importer: LiveVMImporter | None = None
    imported_series = 0
    live_import_errors: list[str] = []
    try:
        endpoint_url = _loopback_import_url(args.vm_url) if args.vm_url else None
        with args.output.open("x", encoding="utf-8") as output:
            if endpoint_url is not None:
                live_importer = LiveVMImporter(
                    endpoint_url=endpoint_url,
                    run_label=args.run_label,
                    host=args.host,
                    gpu_index=args.gpu_index,
                )
            sample_count = 0
            error_count = 0
            try:
                sample_count, error_count = capture(
                    run_label=args.run_label,
                    host=args.host,
                    gpu_index=args.gpu_index,
                    duration_seconds=args.duration_seconds,
                    interval_ms=args.interval_ms,
                    output=output,
                    nvidia_smi=args.nvidia_smi,
                    sample_sink=live_importer.sample_sink if live_importer is not None else None,
                )
            finally:
                if live_importer is not None:
                    live_import_errors, imported_series = live_importer.close()
                    for error in live_import_errors:
                        _write_jsonl(
                            output,
                            _event(
                                "error",
                                run_label=args.run_label,
                                host=args.host,
                                gpu_index=args.gpu_index,
                                component="victoria_metrics_import",
                                error=error,
                            ),
                        )
    except (OSError, TelemetryError) as exc:
        print(f"gemma4_gpu_telemetry: {exc}", file=sys.stderr)
        return 1
    print(
        f"captured {sample_count} samples with {error_count} sample errors; "
        f"imported {imported_series} metric series",
        file=sys.stderr,
    )
    return 0 if sample_count > 0 and error_count == 0 and not live_import_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
