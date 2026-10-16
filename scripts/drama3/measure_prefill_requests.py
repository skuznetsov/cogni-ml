#!/usr/bin/env python3
"""Compare scalar and chunked DRAMA-3 requests in one warm native worker.

The first scalar/chunked pair is an unscored prime. The remaining schedule is
scalar/chunked/chunked/scalar (ABBA). This records fixed-request wall times;
it does not claim a general generation speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time
from typing import Any
import wave


PRIME_COUNT = 2
FIXED_SAMPLING = {"temperature": 0.8, "top_k": 30, "top_p": 0.9}
EVENT_LIMIT = 1_048_576
TIMELINE_EVENTS = {"prefill_request_begin", "prefill_request_end", "drama3_gpu_command"}


class ProbeError(RuntimeError):
    """A fail-closed input, protocol, or parity error."""


def schedule(chunk: int) -> list[dict[str, Any]]:
    if not 2 <= chunk <= 32:
        raise ProbeError("chunk must be 2..32")
    chunks = (1, chunk, 1, chunk, chunk, 1)
    return [
        {"arm": arm, "chunk": value, "prime": arm < PRIME_COUNT,
         "position": arm if arm >= PRIME_COUNT else None}
        for arm, value in enumerate(chunks)
    ]


def fixture_request(fixture: Any, fixture_path: Path, max_frames: int, seed: int) -> dict[str, Any]:
    if not isinstance(fixture, dict):
        raise ProbeError("fixture must be a JSON object")
    source = fixture.get("input", fixture)
    if not isinstance(source, dict):
        raise ProbeError("fixture input must be a JSON object")
    text = source.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ProbeError("fixture must contain non-empty text")
    if not 1 <= max_frames <= 4096:
        raise ProbeError("max-frames must be 1..4096")
    if not 0 <= seed <= (1 << 63) - 1:
        raise ProbeError("seed must fit a non-negative signed 64-bit integer")

    request: dict[str, Any] = {
        "text": text,
        "seed": seed,
        "max_frames": max_frames,
        **FIXED_SAMPLING,
    }
    for key in ("reference_audio", "reference_text"):
        value = source.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise ProbeError(f"fixture {key} must be a string")
            if key == "reference_audio":
                ref = Path(value).expanduser()
                if not ref.is_absolute():
                    ref = fixture_path.parent / ref
                value = str(ref.resolve())
            request[key] = value
    return request


def create_output_dir(path: Path) -> Path:
    if path.exists():
        raise ProbeError(f"output directory already exists: {path}")
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ProbeError(f"output directory already exists: {path}") from error
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_wav_file(path: Path, samples: int) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            shape = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes())
    except (wave.Error, EOFError) as error:
        raise ProbeError(f"cannot parse generated PCM16 WAV: {error}") from error
    if shape != (1, 2, 44100, samples):
        raise ProbeError("WAV format or sample count disagrees with the report")


def _counter(graph: Any, key: str, label: str) -> int:
    if not isinstance(graph, dict):
        raise ProbeError(f"{label} resident_graph must be an object")
    value = graph.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProbeError(f"{label} resident_graph.{key} must be a non-negative integer")
    return value


def validate_record(record: dict[str, Any], label: str,
                    expected_input: dict[str, Any] | None = None) -> None:
    report = record.get("report")
    if not isinstance(report, dict):
        raise ProbeError(f"{label} report must be an object")
    frames = report.get("frames")
    if (not isinstance(frames, list) or not frames or
            any(not isinstance(frame, list) or len(frame) != 11 or
                any(isinstance(code, bool) or not isinstance(code, int) for code in frame)
                for frame in frames)):
        raise ProbeError(f"{label} report must contain complete integer code frames")
    samples = report.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ProbeError(f"{label} report.samples must be positive")
    if report.get("sample_rate") != 44100:
        raise ProbeError(f"{label} sample rate changed")
    if expected_input is not None:
        for key in ("text", "seed", "max_frames", "temperature", "top_k", "top_p"):
            if report.get(key) != expected_input[key]:
                raise ProbeError(f"{label} report does not match fixed request: {key}")
    tape = report.get("sampler_inputs")
    tape_hash = tape.get("sha256") if isinstance(tape, dict) else None
    if (not isinstance(tape, dict) or tape.get("dtype") != "float32-le" or
            isinstance(tape.get("uniform_count"), bool) or
            not isinstance(tape.get("uniform_count"), int) or tape["uniform_count"] <= 0 or
            not isinstance(tape_hash, str) or len(tape_hash) != 64 or
            any(character not in "0123456789abcdef" for character in tape_hash) or
            tape.get("provenance") != "host input tape uploaded once"):
        raise ProbeError(f"{label} sampler input provenance is missing or malformed")
    readbacks = report.get("resident_readbacks")
    if not isinstance(readbacks, dict) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in readbacks.values()):
        raise ProbeError(f"{label} resident_readbacks must contain non-negative integer counts")
    if set(readbacks) != {"control", "terminal"} or any(value <= 0 for value in readbacks.values()):
        raise ProbeError(f"{label} resident_readbacks must be the existing control/terminal boundaries")
    wav_hash = record.get("wav_sha256")
    if (not isinstance(wav_hash, str) or len(wav_hash) != 64 or
            any(character not in "0123456789abcdef" for character in wav_hash)):
        raise ProbeError(f"{label} WAV digest is missing or malformed")

    graph = report.get("resident_graph")
    submitted = _counter(graph, "submitted", label)
    completed = _counter(graph, "completed", label)
    _counter(graph, "completion_waits", label)
    pending = _counter(graph, "pending", label)
    if pending != 0 or completed != submitted:
        raise ProbeError(f"{label} GPU graph is not fully drained")


def validate_worker_lifecycle(report: dict[str, Any], arm: int, label: str) -> None:
    """Require one persistent worker across the ordered request schedule."""
    job = report.get("worker_job")
    if isinstance(job, bool) or not isinstance(job, int) or job != arm + 1:
        raise ProbeError(f"{label} worker job order does not match the request schedule")
    expected_warm = arm > 0
    if report.get("warm_worker") is not expected_warm:
        raise ProbeError(f"{label} persistent worker state does not match the request schedule")


def _prefill_submit_elision(prompt_rows: Any, chunk: Any, label: str) -> int:
    """Count the one legal empty final-row submit skipped by batched prefill."""
    if isinstance(prompt_rows, bool) or not isinstance(prompt_rows, int) or prompt_rows < 1:
        raise ProbeError(f"{label} prompt_rows must be a positive integer")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or not 1 <= chunk <= 32:
        raise ProbeError(f"{label} chunk must be an integer in 1..32")
    # Scalar prefill always encodes the final row. Batched prefill encodes all
    # rows in each chunk's first submission; a multirow final chunk's last
    # row has no teacher-force dispatch, so Context#submit legitimately skips it.
    return int(chunk > 1 and prompt_rows % chunk != 1)


def validate_against_scalar_prime(record: dict[str, Any], scalar_prime: dict[str, Any], label: str) -> None:
    validate_record(record, label)
    validate_record(scalar_prime, "scalar prime")
    report = record["report"]
    baseline = scalar_prime["report"]
    for key in ("frames", "samples", "sample_rate", "reference_sha256", "prompt_rows",
                "sampler_inputs", "resident_readbacks"):
        if report[key] != baseline[key]:
            raise ProbeError(f"{label} differs from scalar prime: {key}")
    if record["wav_sha256"] != scalar_prime["wav_sha256"]:
        raise ProbeError(f"{label} differs from scalar prime: WAV bytes")
    graph = report["resident_graph"]
    base_graph = baseline["resident_graph"]
    chunk = record.get("chunk", report.get("chunk"))
    prompt_rows = report.get("prompt_rows")
    elision = _prefill_submit_elision(prompt_rows, chunk, label)
    for key in ("submitted", "completion_waits"):
        expected = base_graph[key] - elision
        if expected < 0 or graph[key] != expected:
            raise ProbeError(
                f"{label} resident_graph.{key} does not match the expected final-row elision"
            )


def _timeline_events(log_text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    events = []
    for number, line in enumerate(log_text.splitlines(), 1):
        start = line.find("{")
        if start < 0:
            if any(name in line for name in TIMELINE_EVENTS):
                raise ProbeError(f"invalid JSON timeline event at stderr line {number}")
            continue
        try:
            value, _ = decoder.raw_decode(line[start:])
        except json.JSONDecodeError as error:
            if any(name in line for name in TIMELINE_EVENTS):
                raise ProbeError(f"invalid JSON timeline event at stderr line {number}") from error
            continue
        if isinstance(value, dict) and value.get("event") in TIMELINE_EVENTS:
            events.append(value)
    return events


def _timeline_integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProbeError(f"{label} must be an integer >= {minimum}")
    return value


def _timeline_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ProbeError(f"{label} must be finite")
    return number


def _command_timing(event: dict[str, Any], *, rich_required: bool) -> dict[str, Any]:
    seconds = _timeline_number(event.get("gpu_seconds"), "GPU duration")
    if seconds <= 0:
        raise ProbeError("GPU duration must be positive")

    rich_fields = {"gpu_timing_valid", "gpu_start_s", "gpu_end_s"}
    has_rich_timing = any(field in event for field in rich_fields)
    if rich_required:
        if not has_rich_timing:
            raise ProbeError("absolute GPU timing missing for graph queue")
        status = _timeline_integer(event.get("completion_status"), "completion status")
        if status != 0:
            raise ProbeError("GPU command completion status is not zero")
        if event.get("gpu_timing_valid") is not True:
            raise ProbeError("GPU command interval is not valid")
        start = _timeline_number(event.get("gpu_start_s"), "GPU interval start")
        end = _timeline_number(event.get("gpu_end_s"), "GPU interval end")
        if end <= start:
            raise ProbeError("GPU interval must have positive duration")
        if not math.isclose(end - start, seconds, rel_tol=1e-9, abs_tol=1e-12):
            raise ProbeError("GPU duration disagrees with completed interval")
        return {"gpu_seconds": seconds, "gpu_start_s": start, "gpu_end_s": end}

    if has_rich_timing or "completion_status" in event:
        raise ProbeError("depth-zero timing must use checked duration-only completion")
    return {"gpu_seconds": seconds, "gpu_start_s": None, "gpu_end_s": None}


def qualify_request_timeline(log_text: str, records: list[dict[str, Any]],
                             graph_depth: str) -> dict[str, Any]:
    """Bind checked command timings to complete worker request scopes."""
    if graph_depth not in {"0", "1", "2"}:
        raise ProbeError("invalid graph depth for GPU timeline")
    rich_required = graph_depth != "0"
    expected = {record["id"]: record for record in records}
    if len(expected) != len(records):
        raise ProbeError("duplicate request identities in measured schedule")
    ordered_ids = [record["id"] for record in records]

    current: dict[str, Any] | None = None
    seen: set[str] = set()
    command_indices: set[int] = set()
    last_command_index: int | None = None
    scope_position = 0
    total_commands = 0
    unscoped_commands = 0
    for event in _timeline_events(log_text):
        kind = event["event"]
        if kind == "prefill_request_begin":
            if current is not None:
                raise ProbeError("nested worker request timeline scope")
            request_id = event.get("id")
            if not isinstance(request_id, str) or request_id not in expected:
                raise ProbeError("worker timeline began an unknown request")
            if scope_position >= len(ordered_ids) or request_id != ordered_ids[scope_position]:
                raise ProbeError("worker request timeline order does not match the schedule")
            if request_id in seen:
                raise ProbeError("duplicate worker request timeline scope")
            record = expected[request_id]
            chunk = _timeline_integer(event.get("chunk"), "request begin chunk", 1)
            if chunk != record["chunk"]:
                raise ProbeError("worker request begin chunk mismatch")
            current = {"id": request_id, "chunk": chunk, "record": record, "commands": []}
        elif kind == "drama3_gpu_command":
            if current is None:
                unscoped_commands += 1
                continue
            index = _timeline_integer(event.get("index"), "GPU command index")
            if index in command_indices:
                raise ProbeError("duplicate GPU command timeline index")
            if last_command_index is not None and index <= last_command_index:
                raise ProbeError("GPU command timeline indices are out of order")
            command_indices.add(index)
            last_command_index = index
            current["commands"].append({
                "index": index,
                **_command_timing(event, rich_required=rich_required),
            })
            total_commands += 1
        elif kind == "prefill_request_end":
            if current is None:
                raise ProbeError("worker request timeline end without begin")
            request_id = event.get("id")
            chunk = _timeline_integer(event.get("chunk"), "request end chunk", 1)
            if request_id != current["id"] or chunk != current["chunk"]:
                raise ProbeError("worker request timeline end identity mismatch")
            commands = current["commands"]
            graph = current["record"]["report"]["resident_graph"]
            if len(commands) != graph["submitted"]:
                raise ProbeError("request GPU command count does not match submissions")
            if not commands:
                raise ProbeError("request timeline has no completed GPU commands")
            try:
                gpu_sum = math.fsum(command["gpu_seconds"] for command in commands)
            except OverflowError as error:
                raise ProbeError("invalid completed GPU duration sum") from error
            if not math.isfinite(gpu_sum) or gpu_sum <= 0:
                raise ProbeError("invalid completed GPU duration sum")
            starts = [command["gpu_start_s"] for command in commands]
            ends = [command["gpu_end_s"] for command in commands]
            if rich_required:
                span = max(ends) - min(starts)
                if not math.isfinite(span) or span <= 0:
                    raise ProbeError("invalid completed GPU span")
            else:
                span = None
            record = current["record"]
            record["gpu_command_count"] = len(commands)
            record["completed_gpu_interval_sum_seconds"] = gpu_sum
            record["gpu_span_seconds"] = span
            record["gpu_timing_evidence"] = (
                "completed GPU intervals with status 0" if rich_required else
                "duration from existing checked synchronous completion; no absolute span"
            )
            seen.add(request_id)
            scope_position += 1
            current = None

    if current is not None:
        raise ProbeError("unterminated worker request timeline scope")
    if scope_position != len(ordered_ids) or seen != set(expected):
        raise ProbeError("missing worker request timeline scope")
    if unscoped_commands:
        raise ProbeError("GPU commands appeared outside worker request scopes")
    return {
        "graph_depth": int(graph_depth),
        "requests": len(seen),
        "completed_commands": total_commands,
        "unscoped_commands": unscoped_commands,
        "timing_mode": "absolute_valid_intervals" if rich_required else "checked_duration_only",
    }


def _timing_distribution(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ProbeError("cannot summarize invalid timing samples")
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "range": [min(values), max(values)]}


def summarize_request_timings(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [record for record in records if not record["prime"]]
    arms = {
        "scalar": [record for record in measured if record["chunk"] == 1],
        "batched": [record for record in measured if record["chunk"] > 1],
    }
    if any(len(rows) != 2 for rows in arms.values()):
        raise ProbeError("request timing schedule must contain one measured ABBA block")

    summary: dict[str, Any] = {"arms": {}}
    for name, rows in arms.items():
        spans = [row["gpu_span_seconds"] for row in rows]
        if not (all(value is None for value in spans) or all(value is not None for value in spans)):
            raise ProbeError("inconsistent request GPU span availability")
        summary["arms"][name] = {
            "samples": len(rows),
            "request_wall_seconds": _timing_distribution([row["request_wall_seconds"] for row in rows]),
            "completed_gpu_interval_sum_seconds": _timing_distribution(
                [row["completed_gpu_interval_sum_seconds"] for row in rows]),
            "gpu_span_seconds": _timing_distribution(spans) if spans[0] is not None else None,
        }
    scalar = summary["arms"]["scalar"]
    batched = summary["arms"]["batched"]
    summary["descriptive_scalar_over_batched"] = {
        "request_wall": (scalar["request_wall_seconds"]["median"] /
                         batched["request_wall_seconds"]["median"]),
        "completed_gpu_interval_sum": (
            scalar["completed_gpu_interval_sum_seconds"]["median"] /
            batched["completed_gpu_interval_sum_seconds"]["median"]),
    }
    return summary


class JsonLineWorker:
    def __init__(self, process: subprocess.Popen[bytes], timeout_seconds: float):
        self.process = process
        self.timeout_seconds = timeout_seconds
        self.buffer = bytearray()
        self.lines: list[bytes] = []

    def read_event(self, deadline: float) -> dict[str, Any]:
        while not self.lines:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProbeError("worker response timed out")
            if self.process.poll() is not None:
                raise ProbeError(f"worker exited before response (status {self.process.returncode})")
            stdout = self.process.stdout
            if stdout is None:
                raise ProbeError("worker stdout is unavailable")
            readable, _, _ = select.select([stdout], [], [], remaining)
            if not readable:
                raise ProbeError("worker response timed out")
            chunk = os.read(stdout.fileno(), 64 * 1024)
            if not chunk:
                raise ProbeError("worker closed stdout before response")
            self.buffer.extend(chunk)
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                if not line:
                    raise ProbeError("worker emitted an empty JSON line")
                if len(line) > EVENT_LIMIT:
                    raise ProbeError("worker JSON line exceeded 1 MiB")
                self.lines.append(line)
            if len(self.buffer) > EVENT_LIMIT:
                raise ProbeError("worker JSON line exceeded 1 MiB")

        line = self.lines.pop(0)
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProbeError(f"worker emitted malformed JSON: {error}") from error
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise ProbeError("worker event must be a JSON object with an event name")
        return event

    def send(self, request: dict[str, Any]) -> None:
        stdin = self.process.stdin
        if stdin is None:
            raise ProbeError("worker stdin is unavailable")
        stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"))
        stdin.flush()


def _write_event(handle, arm: int | None, started: float, event: dict[str, Any]) -> None:
    handle.write(json.dumps({"arm": arm, "after_seconds": time.monotonic() - started,
                             "worker_event": event}, sort_keys=True) + "\n")
    handle.flush()


def run(args: argparse.Namespace) -> dict[str, Any]:
    binary = args.binary.expanduser().resolve()
    model = args.model.expanduser().resolve()
    fixture_path = args.fixture.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ProbeError(f"binary must be an executable file: {binary}")
    if not model.is_dir():
        raise ProbeError(f"model directory does not exist: {model}")
    if not fixture_path.is_file():
        raise ProbeError(f"fixture file does not exist: {fixture_path}")
    if not 1 <= args.max_frames <= 4096:
        raise ProbeError("max-frames must be 1..4096")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ProbeError("timeout-seconds must be positive")
    try:
        fixture_bytes = fixture_path.read_bytes()
        fixture = json.loads(fixture_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"cannot read JSON fixture: {error}") from error
    request_input = fixture_request(fixture, fixture_path, args.max_frames, args.seed)
    arms = schedule(args.chunk)
    output_dir = create_output_dir(args.output_dir.expanduser().resolve())
    fixture_sha = hashlib.sha256(fixture_bytes).hexdigest()
    events_path = output_dir / "worker-events.jsonl"
    arms_path = output_dir / "arms.jsonl"
    stderr_path = output_dir / "worker.stderr.timeline.log"
    environment = os.environ.copy()
    # Timeline timestamps are passive completion observations and this setting
    # remains fixed for every arm. Other runtime controls are inherited intact.
    environment["DRAMA3_METAL_TIMELINE"] = "1"
    tracked_controls = (
        "DRAMA3_BACKEND", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_METAL_TIMELINE",
        "DRAMA3_GENERATION_SCRATCH", "DRAMA3_MOE_BATCH", "DRAMA3_DELTA_TRANSPOSE",
        "DRAMA3_BACKBONE_DENSE_REDUCTION", "DRAMA3_EXPERT_DOWN_REDUCTION",
        "DRAMA3_TOWER_EMBEDDING_FUSION",
    )
    effective_controls = {key: environment.get(key) for key in tracked_controls}

    records: list[dict[str, Any]] = []
    scalar_prime: dict[str, Any] | None = None
    with events_path.open("x", encoding="utf-8") as event_log, \
            arms_path.open("x", encoding="utf-8") as arm_log, \
            stderr_path.open("xb") as stderr_log:
        process = subprocess.Popen([str(binary), str(model)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=stderr_log, env=environment)
        worker = JsonLineWorker(process, args.timeout_seconds)
        started = time.monotonic()
        try:
            ready = worker.read_event(time.monotonic() + args.timeout_seconds)
            _write_event(event_log, None, started, ready)
            if ready.get("event") != "ready":
                raise ProbeError(f"worker did not become ready: {ready}")
            ready_seconds = time.monotonic() - started

            for arm in arms:
                index = arm["arm"]
                wav_path = output_dir / f"arm-{index:02d}-chunk-{arm['chunk']}.wav"
                sidecar = Path(str(wav_path) + ".json")
                if wav_path.exists() or sidecar.exists():
                    raise ProbeError(f"refusing pre-existing arm output: {wav_path}")
                request_id = f"prefill-arm-{index:02d}"
                worker_request = {"id": request_id, "chunk": arm["chunk"],
                                  "input": request_input, "output": str(wav_path)}
                request_started = time.monotonic()
                worker.send(worker_request)
                progress_count = 0
                while True:
                    event = worker.read_event(request_started + args.timeout_seconds)
                    _write_event(event_log, index, request_started, event)
                    if event["event"] == "error":
                        raise ProbeError(f"worker error in arm {index}: {event.get('message', event)}")
                    if event["event"] == "progress":
                        progress_count += 1
                        continue
                    if event["event"] != "result":
                        raise ProbeError(f"unexpected worker event in arm {index}: {event}")
                    if event.get("id") != request_id or event.get("chunk") != arm["chunk"]:
                        raise ProbeError(f"worker result identity mismatch in arm {index}")
                    break

                if not wav_path.is_file() or not sidecar.is_file():
                    raise ProbeError(f"worker omitted WAV or JSON report in arm {index}")
                report = event.get("report")
                try:
                    sidecar_report = json.loads(sidecar.read_bytes())
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ProbeError(f"cannot read report sidecar for arm {index}: {error}") from error
                if sidecar_report != report:
                    raise ProbeError(f"worker event/report sidecar mismatch in arm {index}")
                record = {
                    **arm,
                    "id": request_id,
                    "request_wall_seconds": time.monotonic() - request_started,
                    "progress_events": progress_count,
                    "wav": wav_path.name,
                    "wav_sha256": sha256_file(wav_path),
                    "report_sidecar_sha256": sha256_file(sidecar),
                    "report": report,
                }
                validate_record(record, f"arm {index}", request_input)
                validate_worker_lifecycle(report, index, f"arm {index}")
                record["expected_final_row_submit_elision"] = _prefill_submit_elision(
                    report.get("prompt_rows"), arm["chunk"], f"arm {index}"
                )
                validate_wav_file(wav_path, report["samples"])
                if scalar_prime is None:
                    if arm["chunk"] != 1 or not arm["prime"]:
                        raise ProbeError("schedule did not begin with a scalar prime")
                    scalar_prime = record
                else:
                    validate_against_scalar_prime(record, scalar_prime, f"arm {index}")
                records.append(record)

            process.stdin.close()
            process.stdin = None
            try:
                status = process.wait(timeout=30)
            except subprocess.TimeoutExpired as error:
                raise ProbeError("worker did not exit after stdin closed") from error
            if status != 0:
                raise ProbeError(f"worker exited with status {status}")
            stderr_log.flush()
            timeline_summary = qualify_request_timeline(
                stderr_path.read_text(encoding="utf-8"), records,
                effective_controls["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] or "0",
            )
            timing_summary = summarize_request_timings(records)
            for record in records:
                arm_log.write(json.dumps(record, sort_keys=True) + "\n")
            arm_log.flush()
        except BaseException:
            if process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait()
            raise

    if len(records) != len(arms):
        raise ProbeError(f"incomplete schedule: got {len(records)} of {len(arms)} arms")
    run_report = {
        "format": "drama3-prefill-request-measurement-v1",
        "status": "PARITY_PASS",
        "model": str(model),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "fixture": str(fixture_path),
        "fixture_sha256": fixture_sha,
        "fixed_request": request_input,
        "worker_ready_seconds": ready_seconds,
        "effective_controls": effective_controls,
        "schedule": records,
        "prime_arms": PRIME_COUNT,
        "measured_arms": [record["arm"] for record in records if not record["prime"]],
        "timeline_stderr": stderr_path.name,
        "timeline_stderr_sha256": sha256_file(stderr_path),
        "timeline_qualification": timeline_summary,
        "timing_summary": timing_summary,
        "measured_arms_use_persistent_worker": all(
            record["report"]["warm_worker"]
            for record in records if not record["prime"]
        ),
        "timing_scope": (
            "send-to-result request wall includes generation, codec, WAV/report work, worker own-job GC, "
            "per-command completion-timeline serialization, and driver-side progress/result pipe handling "
            "and event logging; descriptive for this fixed request and environment"
        ),
        "observer_overhead_limitation": (
            "timeline serialization is inside request wall and scales with command count, so it can bias the "
            "scalar-versus-batched wall contrast; completed GPU durations are reported separately"
        ),
        "parity_requirements": {
            "exact_frames_wav_sha256_sample_count_sampler_inputs_readbacks": True,
            "submission_and_completion_wait_counts_match_derived_elision": True,
            "completed_equals_submitted_and_pending_zero": True,
            "scoped_timeline_commands_match_submissions": True,
            "measured_arms_use_persistent_worker": True,
        },
    }
    report_path = output_dir / "run.json"
    with report_path.open("x", encoding="utf-8") as target:
        json.dump(run_report, target, indent=2, sort_keys=True)
        target.write("\n")
    return run_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path,
                        help="JSON object with a fixed text field and optional reference audio")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="must be a new directory; all WAVs and reports are unique")
    parser.add_argument("--chunk", type=int, default=16, help="prefill chunk, 2..32")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (ProbeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "arms": len(report["schedule"]),
                      "measured_arms": report["measured_arms"],
                      "exact_parity": True, "timing_scope": report["timing_scope"],
                      "timing_summary": report["timing_summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
