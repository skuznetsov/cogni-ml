#!/usr/bin/env python3
"""Measure overlap-safe Shader Timeline coverage for one pinned DRAMA-3 decode.

This analyzer consumes only the capture manifest, native stderr timeline, and
the three selected xctrace XML exports. It never opens a raw .trace bundle.
Shader rows are reported as sampled interval coverage; they are not GPU-runtime
measurements and are never added into a total or ranked as kernel cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import capture_decode_profile as capture  # noqa: E402


FORMAT = "drama3-shader-sampled-coverage-v1"
EVENT = "drama3_gpu_command"
CODEC_LABEL_PREFIX = "dr3c_"
REPRESENTATIVE_CONFLICT_MIN_OVERLAP_NS = 10_000
EXPECTED_INPUT = {
    "text": "Hello, this is a voice test.",
    "voice_id": None,
    "seed": 1,
    "max_frames": 64,
    "temperature": 0.8,
    "top_k": 30,
    "top_p": 0.9,
}
EXPECTED_PHASES = (
    ("prefill", 0, 28),
    ("selector", 29, 29),
    ("decode", 30, 89),
    ("dac", 90, 121),
)
TABLES = {
    "shader": "metal-shader-profiler-intervals",
    "gpu": "metal-gpu-intervals",
    "submissions": "metal-application-command-buffer-submissions",
}
INTEGER_TAGS = {
    "start-time", "duration", "pid", "uint32", "uint64",
    "metal-command-buffer-id", "metal-nesting-level",
}
SENTINEL_COMMAND_BUFFER = (1 << 64) - 1
TIME_UNIT_NS = {
    "ns": Decimal(1),
    "us": Decimal(1_000),
    "µs": Decimal(1_000),
    "μs": Decimal(1_000),
    "ms": Decimal(1_000_000),
    "s": Decimal(1_000_000_000),
}


class CoverageError(ValueError):
    """The selected trace cannot be joined or structurally validated."""


class TimeUnitEvidence:
    """Check integer xctrace time cells against their human-readable formats."""

    _DURATION = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(ns|us|µs|μs|ms|s)\s*$")
    _START = re.compile(r"^(\d+):(\d{2})\.(\d{3})\.(\d{3})$")

    def __init__(self) -> None:
        self.checked = Counter()
        self.max_error_ns = {"start-time": Decimal(0), "duration": Decimal(0)}

    def check(self, element: ET.Element) -> None:
        tag = element.tag
        if tag not in ("start-time", "duration"):
            return
        raw = (element.text or "").strip()
        fmt = element.get("fmt")
        if not raw or not raw.isdecimal() or fmt is None:
            raise CoverageError(f"xctrace {tag} cell lacks integer text or a formatted unit label")
        value = Decimal(raw)
        if tag == "start-time":
            match = self._START.fullmatch(fmt)
            if not match:
                raise CoverageError(f"xctrace start-time format is not mm:ss.mmm.uuu: {fmt!r}")
            minutes, seconds, milliseconds, microseconds = map(int, match.groups())
            displayed_ns = Decimal(
                ((minutes * 60 + seconds) * 1_000_000 + milliseconds * 1_000 + microseconds)
                * 1_000
            )
            # xctrace's elapsed-time label is precise to a microsecond.
            error = abs(value - displayed_ns)
            if error >= 1_000:
                raise CoverageError("xctrace start-time integer is inconsistent with its microsecond label")
        else:
            match = self._DURATION.fullmatch(fmt)
            if not match:
                raise CoverageError(f"xctrace duration format has no recognized time unit: {fmt!r}")
            number_text, unit = match.groups()
            try:
                displayed_ns = Decimal(number_text) * TIME_UNIT_NS[unit]
            except InvalidOperation as exc:
                raise CoverageError(f"xctrace duration format is not numeric: {fmt!r}") from exc
            places = len(number_text.partition(".")[2])
            resolution_ns = TIME_UNIT_NS[unit] / (Decimal(10) ** places)
            error = abs(value - displayed_ns)
            if error > resolution_ns / 2 + 1:
                raise CoverageError("xctrace duration integer is inconsistent with its formatted time unit")
        self.checked[tag] += 1
        self.max_error_ns[tag] = max(self.max_error_ns[tag], error)

    def as_json(self) -> dict:
        if not self.checked["start-time"] or not self.checked["duration"]:
            raise CoverageError("xctrace time-unit validation did not cover start and duration cells")
        return {
            "integer_unit": "nanoseconds",
            "basis": "integer start-time/duration values agree with xctrace fmt labels",
            "formatted_start_time_cells_checked": self.checked["start-time"],
            "formatted_duration_cells_checked": self.checked["duration"],
            "maximum_display_rounding_error_ns": {
                key: int(value) for key, value in self.max_error_ns.items()
            },
        }


def _integer(value: object, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CoverageError(f"{where}: expected integer >= {minimum}")
    return value


def _positive_count(value: object, where: str) -> int:
    return _integer(value, where, minimum=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_phase_ranges(text: str) -> tuple[tuple[str, int, int], ...]:
    phases = []
    seen = set()
    for raw in text.split(","):
        if raw.count("=") != 1 or raw.count(":") != 1:
            raise CoverageError("phase ranges must use name=start:end comma-separated syntax")
        name, bounds = raw.split("=", 1)
        start_text, end_text = bounds.split(":", 1)
        name = name.strip()
        if not name or name in seen:
            raise CoverageError("phase names must be nonempty and unique")
        if not start_text.isdecimal() or not end_text.isdecimal():
            raise CoverageError(f"phase {name!r}: indexes must be nonnegative integers")
        start, end = int(start_text), int(end_text)
        if end < start:
            raise CoverageError(f"phase {name!r}: end index precedes start index")
        seen.add(name)
        phases.append((name, start, end))
    if tuple(phases) != EXPECTED_PHASES:
        raise CoverageError(
            "phase ranges do not match the pinned 29-row prompt / 60-frame / "
            "122-command trajectory"
        )
    return tuple(phases)


def _request_records(manifest: dict) -> tuple[dict, list[dict], list[dict]]:
    if manifest.get("format") != "drama3-complete-decode-capture-v1":
        raise CoverageError("capture manifest format mismatch")
    if manifest.get("input") != EXPECTED_INPUT:
        raise CoverageError("capture input does not match the pinned request")
    if manifest.get("raw_trace_removed") is not True:
        raise CoverageError("raw trace was not confirmed removed")
    for key in ("binary_sha256", "template_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise CoverageError(f"capture manifest {key} is invalid")

    records = manifest.get("requests")
    if not isinstance(records, list):
        raise CoverageError("capture requests must be a list")
    expected_keys = {
        ("off-before", 0), ("off-before", 1),
        ("timeline", 0), ("timeline", 1), ("timeline", 2), ("timeline", 3),
        ("off-after", 0), ("off-after", 1),
    }
    by_key = {}
    for record in records:
        if not isinstance(record, dict):
            raise CoverageError("capture request record is not an object")
        arm, repeat = record.get("arm"), record.get("repeat")
        if not isinstance(arm, str) or isinstance(repeat, bool) or not isinstance(repeat, int):
            raise CoverageError("capture request arm/repeat is invalid")
        key = (arm, repeat)
        if key in by_key:
            raise CoverageError(f"duplicate capture request {key}")
        by_key[key] = record
    if set(by_key) != expected_keys:
        raise CoverageError("capture is missing or has extra OFF/timeline requests")

    timeline = [by_key[("timeline", repeat)] for repeat in range(4)]
    off = [by_key[(arm, repeat)] for arm, repeat in sorted(expected_keys)
           if arm in ("off-before", "off-after")]
    pids = set()
    for repeat, record in enumerate(timeline):
        if record.get("timeline") is not True:
            raise CoverageError(f"timeline repeat {repeat}: timeline flag is false")
        if record.get("instruments") is not (repeat == 2):
            raise CoverageError(f"timeline repeat {repeat}: capture marker mismatch")
        pid = _positive_count(record.get("pid"), f"timeline repeat {repeat} PID")
        pids.add(pid)
        report = record.get("report")
        if not isinstance(report, dict):
            raise CoverageError(f"timeline repeat {repeat}: report missing")
        if report.get("worker_job") != repeat + 1 or report.get("warm_worker") is not (repeat > 0):
            raise CoverageError(f"timeline repeat {repeat}: worker job order mismatch")
        _validate_request_report(report, f"timeline repeat {repeat}")
    if len(pids) != 1:
        raise CoverageError("timeline worker PID changed across the captured requests")

    for record in off:
        if record.get("timeline") is not False or record.get("instruments") is not False:
            raise CoverageError("profiler-OFF request has timeline/capture enabled")
        if not isinstance(record.get("report"), dict):
            raise CoverageError("profiler-OFF request report missing")
        _validate_request_report(record["report"], "profiler-OFF request")

    selected = timeline[2]
    if selected.get("warm") is not True:
        raise CoverageError("captured request is not a warm request")
    warm = [record for record in records if record.get("warm") is True]
    # Each arm's first request is the cold worker request. The two OFF arms
    # contribute one warm request each and the four-request timeline arm
    # contributes three, for five warm requests total.
    if len(warm) != 5:
        raise CoverageError("expected five warm requests across OFF/timeline arms")
    selected_signature = capture.trajectory(selected["report"])
    selected_wav = selected.get("wav_sha256")
    if not isinstance(selected_wav, str) or not re.fullmatch(r"[0-9a-f]{64}", selected_wav):
        raise CoverageError("captured request WAV digest is invalid")
    for record in warm:
        if record.get("wav_sha256") != selected_wav:
            raise CoverageError("warm request WAV digests differ across OFF/timeline arms")
        if capture.trajectory(record["report"]) != selected_signature:
            raise CoverageError("warm request trajectory differs across OFF/timeline arms")

    # The known request produces 29 prompt rows and EOS after 60 frames. These
    # facts bind the explicit phase map below; max_frames=64 alone is not enough.
    report = selected["report"]
    frames = report.get("frames")
    if report.get("prompt_rows") != 29 or not isinstance(frames, list) or len(frames) != 60:
        raise CoverageError("captured report is not the pinned 29-row / 60-frame trajectory")
    if report.get("stopped_on_eos") is not True or report.get("truncated") is not False:
        raise CoverageError("captured report did not complete the pinned EOS trajectory")
    graph = report.get("resident_graph")
    if not isinstance(graph, dict):
        raise CoverageError("captured resident graph report missing")
    if graph.get("submitted") != 122 or graph.get("completed") != 122 or graph.get("pending") != 0:
        raise CoverageError("captured report is not a completed 122-command request")

    return selected, timeline, off


def _validate_request_report(report: dict, where: str) -> None:
    expected = {
        "text": EXPECTED_INPUT["text"], "seed": EXPECTED_INPUT["seed"],
        "max_frames": EXPECTED_INPUT["max_frames"],
        "temperature": EXPECTED_INPUT["temperature"],
        "top_k": EXPECTED_INPUT["top_k"], "top_p": EXPECTED_INPUT["top_p"],
        "reference_sha256": None,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise CoverageError(f"{where}: report {key} does not match pinned input")


def _timeline_events(path: Path, timeline: list[dict], selected: dict) -> tuple[dict[int, dict], tuple[int, int]]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if EVENT in line:
                raise CoverageError(f"stderr line {line_number}: malformed command event JSON") from exc
            continue
        if isinstance(value, dict) and value.get("event") == EVENT:
            events.append(value)

    expected_total = 0
    ranges = []
    for repeat, record in enumerate(timeline):
        report = record["report"]
        count = _positive_count(report.get("resident_graph", {}).get("submitted"),
                                f"timeline repeat {repeat} submitted")
        ranges.append((expected_total, expected_total + count - 1))
        expected_total += count
    if len(events) != expected_total:
        raise CoverageError(f"timeline stderr has {len(events)} command events; reports require {expected_total}")

    by_index = {}
    observed_order = []
    for event in events:
        index = event.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise CoverageError("timeline event index must be a nonnegative integer")
        if index in by_index:
            raise CoverageError(f"duplicate timeline command index {index}")
        if event.get("completion_status") != 0:
            raise CoverageError(f"timeline command {index} did not complete successfully")
        by_index[index] = event
        observed_order.append(index)
    expected_indexes = list(range(expected_total))
    if observed_order != expected_indexes:
        raise CoverageError("timeline command events are dropped, reordered, or outside the reports")

    selected_repeat = next(i for i, item in enumerate(timeline) if item is selected)
    start, end = ranges[selected_repeat]
    selected_indexes = list(range(start, end + 1))
    if len(selected_indexes) != 122:
        raise CoverageError("captured request event range is not exactly 122 commands")
    return {index: by_index[index] for index in selected_indexes}, (start, end)


def _parse_number(tag: str, element: ET.Element) -> object:
    if tag == "sentinel":
        return None
    raw = (element.text or "").strip()
    if tag in INTEGER_TAGS:
        if not raw:
            raise CoverageError(f"xctrace {tag} cell has no integer text")
        try:
            return int(raw)
        except ValueError as exc:
            raise CoverageError(f"xctrace {tag} cell is not an integer") from exc
    if tag == "percent":
        if not raw:
            return None
        try:
            number = float(raw)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return raw or element.get("fmt") or None


def _resolve_cell(
    element: ET.Element,
    intern: dict[tuple[str, str], object],
    time_units: TimeUnitEvidence | None = None,
) -> object:
    tag = element.tag
    ref = element.get("ref")
    if ref is not None:
        try:
            return intern[(tag, ref)]
        except KeyError as exc:
            raise CoverageError(f"xctrace unresolved typed-cell reference {tag}#{ref}") from exc
    children = {child.tag: _resolve_cell(child, intern, time_units) for child in element}
    if time_units is not None:
        time_units.check(element)
    if tag == "process":
        pid = children.get("pid")
        value = (element.get("fmt") or "", pid)
    elif tag == "sentinel":
        value = None
    else:
        value = _parse_number(tag, element)
    ident = element.get("id")
    # xctrace exports typed-cell references for both columns we consume and
    # unrelated columns in the same selected table (for example `thread`,
    # `formatted-label` and `connection-uuid64`). Preserve every typed cell by
    # tag+ID so unrelated columns do not make an otherwise valid row unreadable.
    if ident is not None:
        key = (tag, ident)
        if key in intern and intern[key] != value:
            raise CoverageError(f"xctrace duplicate typed-cell ID {tag}#{ident}")
        intern[key] = value
    return value


def _process_pid(value: object) -> tuple[str, int] | None:
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    name, raw_pid = value
    pid = raw_pid if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) else None
    match = re.search(r"\((\d+)\)$", str(name))
    if match:
        suffix_pid = int(match.group(1))
        if pid is not None and pid != suffix_pid:
            raise CoverageError("xctrace process PID cell conflicts with process label")
        pid = suffix_pid
    if pid is None:
        return None
    return str(name), pid


def _read_target_table(
    path: Path,
    expected_schema: str,
    target_pid: int,
    time_units: TimeUnitEvidence,
) -> tuple[list[dict], str, int]:
    columns: list[str] | None = None
    schema_count = 0
    rows: list[dict] = []
    process_names = set()
    unattributed_gpu_vertex_rows = 0
    intern: dict[tuple[str, str], object] = {}
    stack: list[ET.Element] = []
    try:
        # Owning the file handle makes iterator cleanup deterministic on Python
        # versions where ElementTree's iterparse iterator has no close method.
        with path.open("rb") as source:
            for event, element in ET.iterparse(source, events=("start", "end")):
                if event == "start":
                    stack.append(element)
                    continue
                if element.tag == "schema":
                    if element.get("name") == expected_schema:
                        schema_count += 1
                        if schema_count != 1:
                            raise CoverageError(f"{path}: duplicate schema {expected_schema}")
                        columns = [col.findtext("mnemonic") for col in element.findall("col")]
                        if not columns or any(not isinstance(col, str) for col in columns):
                            raise CoverageError(f"{path}: malformed schema columns")
                        if len(set(columns)) != len(columns):
                            raise CoverageError(f"{path}: duplicate schema column names")
                    if len(stack) > 1:
                        stack[-2].remove(element)
                elif element.tag == "row" and columns is not None:
                    cells = [_resolve_cell(cell, intern, time_units) for cell in element]
                    if len(cells) != len(columns):
                        raise CoverageError(f"{path}: row width differs from schema")
                    record = dict(zip(columns, cells))
                    identity = _process_pid(record.get("process"))
                    if identity is None:
                        # The selected GPU table includes global Active/Vertex
                        # rows with no process and cmdbuffer-id 0. They cannot
                        # be joined to this Compute-only request; admit only
                        # that fully classified, explicitly unkeyed class.
                        if (expected_schema == TABLES["gpu"]
                                and record.get("state") == "Active"
                                and record.get("channel-name") == "Vertex"
                                and record.get("cmdbuffer-id") == 0):
                            unattributed_gpu_vertex_rows += 1
                            if len(stack) > 1:
                                stack[-2].remove(element)
                            stack.pop()
                            continue
                        raise CoverageError(f"{path}: row has no valid process PID identity")
                    if identity[1] == target_pid:
                        process_names.add(identity[0])
                        rows.append(record)
                    if len(stack) > 1:
                        stack[-2].remove(element)
                stack.pop()
    except ET.ParseError as exc:
        raise CoverageError(f"{path}: malformed xctrace XML") from exc
    if schema_count != 1 or columns is None:
        raise CoverageError(f"{path}: missing schema {expected_schema}")
    if not rows:
        raise CoverageError(f"{path}: no rows for target PID {target_pid}")
    if len(process_names) != 1:
        raise CoverageError(f"{path}: target process identity is missing or inconsistent")
    return rows, next(iter(process_names)), unattributed_gpu_vertex_rows


def _verify_export(manifest: dict, schema: str, path: Path) -> None:
    exports = manifest.get("exports")
    if not isinstance(exports, dict) or schema not in exports:
        raise CoverageError(f"capture manifest has no export record for {schema}")
    record = exports[schema]
    if not isinstance(record, dict):
        raise CoverageError(f"capture export record for {schema} is invalid")
    expected_path = record.get("path")
    if not isinstance(expected_path, str) or Path(expected_path).resolve() != path.resolve():
        raise CoverageError(f"{schema}: selected export path does not match capture manifest")
    expected_bytes = _positive_count(record.get("bytes"), f"{schema} export bytes")
    if path.stat().st_size != expected_bytes:
        raise CoverageError(f"{schema}: selected export byte count changed")
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
        raise CoverageError(f"{schema}: selected export digest mismatch")


def _merge(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _clip(intervals: Iterable[tuple[int, int]], windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    source = _merge(intervals)
    clipped = []
    left_index = 0
    for start, end in source:
        while left_index < len(windows) and windows[left_index][1] <= start:
            left_index += 1
        index = left_index
        while index < len(windows) and windows[index][0] < end:
            lo = max(start, windows[index][0])
            hi = min(end, windows[index][1])
            if hi > lo:
                clipped.append((lo, hi))
            index += 1
    return _merge(clipped)


def _length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge(intervals))


def _coverage(groups: dict[str, list[tuple[int, int]]]) -> dict[str, dict[str, int]]:
    """Return per-group inclusive, exclusive and overlap coverage by sweep line."""
    merged = {name: _merge(intervals) for name, intervals in groups.items()}
    events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for name, intervals in merged.items():
        for start, end in intervals:
            events[start].append((name, 1))
            events[end].append((name, -1))
    result = {name: {"inclusive_units": 0, "exclusive_units": 0, "overlap_units": 0}
              for name in merged}
    active: set[str] = set()
    previous: int | None = None
    for point in sorted(events):
        if previous is not None and point > previous and active:
            span = point - previous
            if len(active) == 1:
                only = next(iter(active))
                result[only]["inclusive_units"] += span
                result[only]["exclusive_units"] += span
            else:
                for name in active:
                    result[name]["inclusive_units"] += span
                    result[name]["overlap_units"] += span
        for name, delta in events[point]:
            if delta < 0:
                if name not in active:
                    raise CoverageError("interval sweep encountered an unmatched end")
                active.remove(name)
            else:
                if name in active:
                    raise CoverageError("interval sweep encountered duplicate active interval")
                active.add(name)
        previous = point
    if active:
        raise CoverageError("interval sweep ended with active coverage")
    return result


def _family(shader_label: str) -> str:
    # xctrace may append an instance ordinal. Group those instances by their
    # function label; preserve the exact label separately in per-shader output.
    return re.sub(r"\s+\(\d+\)$", "", shader_label)


def _coverage_rows(coverage: dict[str, dict[str, int]]) -> list[dict]:
    rows = []
    for name in sorted(coverage):
        item = coverage[name]
        if item["inclusive_units"] != item["exclusive_units"] + item["overlap_units"]:
            raise CoverageError(f"{name}: coverage partition invariant failed")
        rows.append({
            "name": name,
            "inclusive_sample_interval_units": item["inclusive_units"],
            "exclusive_sample_interval_units": item["exclusive_units"],
            "overlap_or_ambiguous_sample_interval_units": item["overlap_units"],
        })
    return rows


def analyze(
    capture_path: Path,
    stderr_path: Path,
    shader_path: Path,
    gpu_path: Path,
    submissions_path: Path,
    phase_ranges_text: str,
) -> dict:
    manifest = json.loads(capture_path.read_text())
    selected, timeline, _off = _request_records(manifest)
    phases = _parse_phase_ranges(phase_ranges_text)
    selected_events, event_range = _timeline_events(stderr_path, timeline, selected)
    pid = selected["pid"]
    if len(selected_events) != 122:
        raise CoverageError("selected native command event count differs from the request report")

    for schema, path in (
        (TABLES["shader"], shader_path),
        (TABLES["gpu"], gpu_path),
        (TABLES["submissions"], submissions_path),
    ):
        _verify_export(manifest, schema, path)

    time_units = TimeUnitEvidence()
    submission_rows, process_name, _ignored_submission_rows = _read_target_table(
        submissions_path, TABLES["submissions"], pid, time_units
    )
    if len(submission_rows) != 122:
        raise CoverageError(f"target PID has {len(submission_rows)} submission rows, expected 122")
    for row in submission_rows:
        if row.get("event-type") != "CommandBufferSubmission":
            raise CoverageError("target submission table contains a non-submission event")
    submission_rows.sort(key=lambda row: _integer(row.get("start"), "submission start"))
    starts = [_integer(row.get("start"), "submission start") for row in submission_rows]
    if any(right <= left for left, right in zip(starts, starts[1:])):
        raise CoverageError("target submission start times are duplicated or not strictly ordered")
    command_buffer_ids = []
    for row in submission_rows:
        command_buffer = _integer(row.get("cmdbuffer-id"), "submission command-buffer ID", minimum=1)
        if command_buffer == SENTINEL_COMMAND_BUFFER:
            raise CoverageError("submission command-buffer ID is the xctrace sentinel")
        command_buffer_ids.append(command_buffer)
    if len(set(command_buffer_ids)) != 122:
        raise CoverageError("duplicate target command-buffer IDs")

    # Bind timeline indices to xctrace command buffers only after exact count,
    # process identity, uniqueness and monotone submission creation are proven.
    absolute_to_cb = dict(zip(range(event_range[0], event_range[1] + 1), command_buffer_ids))
    relative_to_absolute = {
        relative: event_range[0] + relative for relative in range(122)
    }
    phase_for_relative = {}
    for name, first, last in phases:
        for relative in range(first, last + 1):
            if relative in phase_for_relative:
                raise CoverageError("phase command indexes overlap")
            phase_for_relative[relative] = name
    if set(phase_for_relative) != set(range(122)):
        raise CoverageError("phase ranges do not cover every command exactly once")
    decode_ids = {
        absolute_to_cb[relative_to_absolute[index]]
        for index, phase in phase_for_relative.items() if phase == "decode"
    }
    if len(decode_ids) != 60:
        raise CoverageError("decode phase does not map to exactly 60 command buffers")

    gpu_rows, gpu_process, unattributed_gpu_vertex_rows = _read_target_table(
        gpu_path, TABLES["gpu"], pid, time_units
    )
    if gpu_process != process_name:
        raise CoverageError("GPU and submission table process identities disagree")
    by_command: dict[int, list[tuple[int, int]]] = defaultdict(list)
    seen_gpu_rows = set()
    command_buffer_id_set = set(command_buffer_ids)
    for row in gpu_rows:
        state, channel = row.get("state"), row.get("channel-name")
        if state == "Active":
            if channel != "Compute":
                raise CoverageError("target process has active non-Compute GPU intervals")
            command_buffer = row.get("cmdbuffer-id")
            if isinstance(command_buffer, bool) or not isinstance(command_buffer, int):
                raise CoverageError("active Compute interval has no command-buffer ID")
            if command_buffer not in command_buffer_id_set:
                raise CoverageError("active Compute interval has no matching submission row")
            start = _integer(row.get("start"), "GPU interval start")
            duration = _positive_count(row.get("duration"), "GPU interval duration")
            key = (command_buffer, start, duration, state, channel)
            if key in seen_gpu_rows:
                raise CoverageError("duplicate active Compute GPU interval row")
            seen_gpu_rows.add(key)
            by_command[command_buffer].append((start, start + duration))
        elif state is None or channel is None:
            raise CoverageError("target GPU interval lacks state/channel classification")
    missing_gpu = set(command_buffer_ids) - set(by_command)
    if missing_gpu:
        raise CoverageError(f"{len(missing_gpu)} submitted command buffers lack Active/Compute intervals")

    decode_windows = _merge(interval for command_buffer in decode_ids
                            for interval in by_command[command_buffer])
    if not decode_windows:
        raise CoverageError("decode phase has no nonempty Active/Compute GPU windows")
    decode_gpu_units = _length(decode_windows)

    shader_rows, shader_process, _ignored_shader_rows = _read_target_table(
        shader_path, TABLES["shader"], pid, time_units
    )
    if shader_process != process_name:
        raise CoverageError("Shader Timeline and submission process identities disagree")
    all_sample_intervals: list[tuple[int, int]] = []
    codec_conflicts: list[dict] = []
    shader_sample_rows_overlapping_decode = 0
    for row in shader_rows:
        if row.get("event-name") != "Shader Timeline" or row.get("channel-name") != "Compute":
            raise CoverageError("target shader row is not a Compute Shader Timeline sample")
        if row.get("shader-type") != "Compute":
            raise CoverageError("target shader row has an unknown/non-Compute shader type")
        label = row.get("label") or row.get("name")
        if not isinstance(label, str) or not label.strip():
            raise CoverageError("target shader row has no stable shader label")
        start = _integer(row.get("start"), "shader sample start")
        duration = _positive_count(row.get("duration"), "shader sample duration")
        clipped = _clip(((start, start + duration),), decode_windows)
        if not clipped:
            continue
        shader_sample_rows_overlapping_decode += 1
        all_sample_intervals.extend(clipped)
        if label.startswith(CODEC_LABEL_PREFIX):
            codec_conflicts.append({
                "label": label,
                "start_ns": start,
                "duration_ns": duration,
                "decode_window_overlap_ns": _length(clipped),
            })

    if not all_sample_intervals:
        raise CoverageError("no target Shader Timeline intervals overlap decode GPU windows")
    all_sample_intervals = _merge(all_sample_intervals)
    observed_shader_union = _length(all_sample_intervals)
    if observed_shader_union > decode_gpu_units:
        raise CoverageError("clipped Shader Timeline union exceeds decode GPU window union")

    # In this pinned no-reference request, dr3c_* labels are emitted by the
    # resident codec path, which starts after generator.generate returns.
    # Their overlap with earlier mapped decode windows falsifies the shortcut
    # "Shader Timeline clock overlap implies phase ownership". Since the
    # selected Shader Timeline schema has no command-buffer identity, do not
    # publish per-family or per-shader phase tables, even if this particular
    # conflict is absent: their ownership is not directly joined to a command.
    conflict_labels = sorted({item["label"] for item in codec_conflicts})
    conflict_examples = []
    for expected_label in (
        "dr3c_conv1d_coalesced (1)",
        "dr3c_layer_norm (1)",
        "dr3c_embedding (0)",
    ):
        candidates = [
            item for item in codec_conflicts
            if item["label"] == expected_label
            and item["decode_window_overlap_ns"] >= REPRESENTATIVE_CONFLICT_MIN_OVERLAP_NS
        ]
        if candidates:
            conflict_examples.append(min(candidates, key=lambda item: item["start_ns"]))

    return {
        "format": FORMAT,
        "status": "phase_attribution_rejected",
        "structural_status": "structurally_complete_sample_coverage",
        "capture": {
            "target_pid": pid,
            "target_process": process_name,
            "binary_sha256": manifest["binary_sha256"],
            "template_sha256": manifest["template_sha256"],
            "captured_worker_job": selected["report"]["worker_job"],
            "captured_repeat": selected["repeat"],
            "native_command_indices_inclusive": [event_range[0], event_range[1]],
            "native_command_count": len(selected_events),
            "submission_count": len(submission_rows),
            "unique_command_buffer_count": len(set(command_buffer_ids)),
        },
        "trajectory": {
            "prompt_rows": selected["report"]["prompt_rows"],
            "generated_frames": len(selected["report"]["frames"]),
            "submitted_commands": selected["report"]["resident_graph"]["submitted"],
            "matched_profiler_off_wav_sha256": selected["wav_sha256"],
            "warm_profiler_off_and_timeline_trajectory_match": True,
        },
        "phase_ranges_relative_inclusive": [
            {"name": name, "start_index": start, "end_index": end}
            for name, start, end in phases
        ],
        "join": {
            "basis": "order-based association: cumulative native command indexes mapped to target-PID submission rows sorted by strictly increasing xctrace start time",
            "native_event_to_submission_identity_key": False,
            "decode_command_buffer_count": len(decode_ids),
            "all_submitted_command_buffers_have_active_compute_intervals": True,
            "active_compute_intervals_without_submission_join": 0,
            "unattributed_gpu_vertex_rows_excluded": unattributed_gpu_vertex_rows,
        },
        "coverage": {
            "interpretation": "formal union of sampled Shader Timeline intervals clipped to the order-mapped decode Active/Compute GPU windows; this is not phase ownership, true GPU runtime, or an additive family measure",
            "interval_unit_validation": time_units.as_json(),
            "interval_unit": "nanoseconds, cross-checked against xctrace fmt labels; coverage remains sampled interval coverage",
            "decode_gpu_active_compute_union_units": decode_gpu_units,
            "shader_sample_interval_union_units": observed_shader_union,
            "shader_sample_union_fraction_of_decode_gpu_windows": observed_shader_union / decode_gpu_units,
            "decode_gpu_window_without_shader_sample_interval_units": decode_gpu_units - observed_shader_union,
            "shader_sample_rows_overlapping_decode_windows": shader_sample_rows_overlapping_decode,
            "family_coverage": None,
            "individual_shader_coverage": None,
            "phase_specific_tables_emitted": False,
            "durations_or_percentages_summed_across_rows": False,
        },
        "phase_attribution": {
            "status": "rejected",
            "reason": "Shader Timeline rows have no command-buffer key, and codec-labeled dr3c_* samples overlap GPU windows mapped to generator decode even though the pinned no-reference source path calls CodecResident.decode only after generator.generate returns.",
            "codec_namespace_rows_overlapping_decode_windows": len(codec_conflicts),
            "codec_namespace_labels_overlapping_decode_windows": conflict_labels,
            "representative_example_minimum_overlap_ns": REPRESENTATIVE_CONFLICT_MIN_OVERLAP_NS,
            "representative_conflicts": conflict_examples,
            "conflict_interpretation": "These are contradictory interval-label/time-overlap observations, not proof that DAC kernels executed during generation. The export documentation only identifies the rows as Shader Timeline intervals; row ownership semantics and the order-only native-to-submission association remain unresolved.",
        },
        "limitations": [
            "Native timeline events and xctrace submissions expose no shared command ID; exact counts, target PID and monotone ordering support only an order-based association, not independent per-command identity proof.",
            "Shader Timeline rows have no command-buffer/submission key. Codec-labeled dr3c_* rows overlap the mapped decode windows, so clock clipping cannot establish phase ownership; the cause could be Shader Timeline interval semantics or the order-only association, and this export does not distinguish them.",
            "The aggregate interval union describes only observed sampled-interval envelopes, not lower/upper bounds on actual shader runtime. Per-family and per-shader phase tables are suppressed.",
            "The selected exports expose no independent dropped-sample counter, so command/GPU join completeness is structural and cannot prove every Shader Timeline sample was retained.",
            "Processless Active/Vertex rows with cmdbuffer-id 0 are excluded as unjoinable GPU work; only process-attributed Active/Compute windows enter decode clipping.",
            "This output does not attribute or rank kernel runtime by phase or establish a dominant GPU family.",
        ],
        "next_discriminator": "Replay captured decode operands directly with a diagnostic-only homogeneous workload and batched drains; do not add intermediate synchronization to the full request. Preserve the profiler-OFF full-WAV comparison as the throughput check.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path, help="capture.json from capture_decode_profile.py")
    parser.add_argument("--stderr", required=True, type=Path, help="timeline.stderr from the same capture")
    parser.add_argument("--shader", required=True, type=Path, help="selected Shader Timeline XML export")
    parser.add_argument("--gpu", required=True, type=Path, help="selected GPU intervals XML export")
    parser.add_argument("--submissions", required=True, type=Path, help="selected command-buffer submissions XML export")
    parser.add_argument("--phase-ranges", required=True,
                        help="pinned inclusive ranges: prefill=0:28,selector=29:29,decode=30:89,dac=90:121")
    args = parser.parse_args()
    try:
        report = analyze(args.capture, args.stderr, args.shader, args.gpu,
                         args.submissions, args.phase_ranges)
    except (CoverageError, OSError, json.JSONDecodeError, KeyError, TypeError,
            AttributeError, OverflowError) as exc:
        parser.exit(2, f"analyze_shader_coverage: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
