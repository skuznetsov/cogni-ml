#!/usr/bin/env python3
"""Compile a small Hy-Nomas director project into explicit offline inputs.

This tool validates and quantizes a prompt timeline. It does not execute
Hy-Nomas, load model weights, or claim that a compiled project is generatable.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PINNED_FPS = 24
PIXEL_FRAMES_PER_LATENT = 4
PROMPT_SEPARATOR = "|||"
PROJECT_SCHEMA = "hy_nomas_director_project_v1"
COMPILED_SCHEMA = "hy_nomas_timeline_compile_v1"


class TimelineError(ValueError):
    """A director project cannot be mapped to the pinned timeline contract."""


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TimelineError(f"{field} must be a finite JSON number")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise TimelineError(f"{field} must be a finite JSON number") from error
    if not number.is_finite():
        raise TimelineError(f"{field} must be finite")
    return number


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _frame_floor(seconds: Decimal) -> int:
    # The pinned upstream parser uses int(seconds * fps), which truncates
    # toward zero. Input seconds are nonnegative, so this is floor.
    numerator, denominator = seconds.as_integer_ratio()
    return numerator * PINNED_FPS // denominator


def _align_time_up(seconds: Decimal) -> int:
    numerator, denominator = seconds.as_integer_ratio()
    scaled_denominator = denominator * PIXEL_FRAMES_PER_LATENT
    latent_boundary = (
        numerator * PINNED_FPS + scaled_denominator - 1
    ) // scaled_denominator
    return latent_boundary * PIXEL_FRAMES_PER_LATENT


def _effective_seconds(frame: int) -> str:
    return _decimal_text(Decimal(frame) / Decimal(PINNED_FPS))


def _validate_project(
    project: Any,
) -> tuple[str, int, int, Decimal, list[dict[str, Any]]]:
    if not isinstance(project, dict):
        raise TimelineError("project must be a JSON object")
    expected = {
        "schema", "project_id", "fps", "width", "height", "duration_seconds",
        "prompt_changes",
    }
    if set(project) != expected:
        missing = sorted(expected - set(project))
        extra = sorted(set(project) - expected)
        raise TimelineError(f"project keys mismatch; missing={missing}, extra={extra}")
    if project["schema"] != PROJECT_SCHEMA:
        raise TimelineError(f"schema must be {PROJECT_SCHEMA!r}")
    project_id = project["project_id"]
    if not isinstance(project_id, str) or not project_id.strip():
        raise TimelineError("project_id must be a nonempty string")
    if type(project["fps"]) is not int or project["fps"] != PINNED_FPS:
        raise TimelineError(f"fps must be the pinned value {PINNED_FPS}")
    for name in ("width", "height"):
        value = project[name]
        if type(value) is not int or value <= 0 or value % 16:
            raise TimelineError(f"{name} must be a positive multiple of 16")

    duration = _decimal(project["duration_seconds"], "duration_seconds")
    if duration <= 0:
        raise TimelineError("duration_seconds must be positive")
    requested_duration_frame = _frame_floor(duration)
    if requested_duration_frame <= 0:
        raise TimelineError("duration must include at least one upstream pixel frame")
    effective_duration_frame = _align_time_up(duration)

    raw_changes = project["prompt_changes"]
    if not isinstance(raw_changes, list) or not raw_changes:
        raise TimelineError("prompt_changes must be a nonempty ordered array")
    changes: list[dict[str, Any]] = []
    previous_requested: Decimal | None = None
    previous_effective = -1
    for index, raw_change in enumerate(raw_changes):
        if (not isinstance(raw_change, dict)
                or set(raw_change) != {"at_seconds", "prompt"}):
            raise TimelineError(
                f"prompt_changes[{index}] must contain at_seconds and prompt")
        requested_time = _decimal(
            raw_change["at_seconds"], f"prompt_changes[{index}].at_seconds")
        if requested_time < 0 or requested_time >= duration:
            raise TimelineError(f"prompt_changes[{index}] must be within [0, duration)")
        if previous_requested is not None and requested_time <= previous_requested:
            raise TimelineError("prompt_changes must be strictly ordered by requested time")
        prompt = raw_change["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise TimelineError(f"prompt_changes[{index}].prompt must be nonempty text")
        if PROMPT_SEPARATOR in prompt:
            raise TimelineError(
                f"prompt_changes[{index}].prompt contains reserved separator {PROMPT_SEPARATOR!r}")
        requested_frame = _frame_floor(requested_time)
        effective_frame = _align_time_up(requested_time)
        if index == 0 and requested_time != 0:
            raise TimelineError("the first prompt change must start at 0 seconds")
        if effective_frame <= previous_effective:
            raise TimelineError("prompt changes collapse to the same effective frame boundary")
        if effective_frame >= effective_duration_frame:
            raise TimelineError("prompt change reaches or overshoots the effective duration")
        changes.append({
            "requested_time": requested_time,
            "requested_frame": requested_frame,
            "effective_frame": effective_frame,
            "prompt": prompt.strip(),
        })
        previous_requested = requested_time
        previous_effective = effective_frame

    if changes[0]["effective_frame"] != 0:
        raise TimelineError("the first effective prompt boundary must be frame 0")
    if effective_duration_frame <= changes[-1]["effective_frame"]:
        raise TimelineError("final prompt interval has zero duration")
    return (
        project_id.strip(), project["width"], project["height"], duration, changes)


def compile_project(project: Any) -> dict[str, Any]:
    """Compile a validated project to explicit pinned-upstream inputs."""
    project_id, width, height, requested_duration, changes = _validate_project(project)
    requested_end_frame = _frame_floor(requested_duration)
    frame_count = _align_time_up(requested_duration)
    latent_frames = (frame_count - 1) // PIXEL_FRAMES_PER_LATENT + 1
    boundaries = [change["effective_frame"] for change in changes] + [frame_count]
    latent_boundaries = [boundary // PIXEL_FRAMES_PER_LATENT for boundary in boundaries]
    if latent_boundaries[-1] != latent_frames:
        # This should be impossible because the compiler aligns the total end
        # boundary to the upstream temporal ratio. Keep this guard executable.
        raise TimelineError("compiled end boundary disagrees with upstream latent length")
    if any(left >= right for left, right in zip(latent_boundaries, latent_boundaries[1:])):
        raise TimelineError("a prompt interval collapses to zero latent duration")

    segments: list[dict[str, Any]] = []
    for index, change in enumerate(changes):
        requested_end = (changes[index + 1]["requested_time"]
                         if index + 1 < len(changes) else requested_duration)
        effective_start = change["effective_frame"]
        effective_end = boundaries[index + 1]
        segments.append({
            "prompt": change["prompt"],
            "requested_start_seconds": _decimal_text(change["requested_time"]),
            "requested_end_seconds": _decimal_text(requested_end),
            "requested_start_frame": change["requested_frame"],
            "requested_end_frame": _frame_floor(requested_end),
            "effective_start_frame": effective_start,
            "effective_end_frame": effective_end,
            "effective_start_seconds": _effective_seconds(effective_start),
            "effective_end_seconds": _effective_seconds(effective_end),
            "effective_start_latent_frame": latent_boundaries[index],
            "effective_end_latent_frame": latent_boundaries[index + 1],
        })

    prompt_count = len(changes)
    rest_field = "prompt" if prompt_count > 1 else ""
    return {
        "schema": COMPILED_SCHEMA,
        "status": "compiled_only",
        "project_id": project_id,
        "fps": PINNED_FPS,
        "width": width,
        "height": height,
        "requested_duration_seconds": _decimal_text(requested_duration),
        "requested_frame_count": requested_end_frame,
        "effective_duration_seconds": _effective_seconds(frame_count),
        "frame_count": frame_count,
        "latent_frames": latent_frames,
        "requested_frame_boundaries": (
            [change["requested_frame"] for change in changes] + [requested_end_frame]),
        "segment_frame_boundaries": boundaries,
        "segment_latent_boundaries": latent_boundaries,
        "prompts": [change["prompt"] for change in changes],
        "segments": segments,
        "upstream_input": [{
            "test_id": project_id,
            "test_type": "hy_nomas_director_timeline",
            "segments": [{"prompt": change["prompt"]} for change in changes],
        }],
        "upstream_overrides": {
            "inference.rollout_type": "seg_ar",
            "inference.video_length": "event",
            "inference.video_fps": PINNED_FPS,
            "inference.video_size": [height, width],
            "inference.num_frames": frame_count,
            "inference.segment_frame_boundaries": boundaries,
            "inference.max_segments": prompt_count,
            "inference.desc_field_first": "prompt",
            "inference.desc_field_rest": rest_field,
        },
        "warnings": [
            "compile-only: this file does not execute or qualify Hy-Nomas generation",
            "prompt changes are effective only at aligned segment boundaries;"
            " upstream seg_ar does not provide arbitrary subframe control",
            "frame boundaries are rounded up to a multiple of four pixel frames;"
            " inspect requested/effective deltas before generation",
        ],
    }


def load_project(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        try:
            project = json.load(stream, parse_float=Decimal)
        except (json.JSONDecodeError, InvalidOperation) as error:
            raise TimelineError(f"invalid project JSON: {error}") from error
    _validate_project(project)
    return project


def write_compiled(project_path: Path, output_path: Path) -> int:
    if output_path.exists():
        raise FileExistsError(output_path)
    if output_path.suffix.lower() != ".json" or not output_path.parent.is_dir():
        raise ValueError("output must be a .json file in an existing directory")
    compiled = compile_project(load_project(project_path))
    payload = (json.dumps(compiled, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=".json", dir=output_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking is atomic and fails if another writer creates output first.
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True,
                        help="director project JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="new compiled JSON path; existing files are never replaced")
    args = parser.parse_args()
    try:
        write_compiled(args.project, args.output)
    except (OSError, TimelineError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "compiled_only", "output": str(args.output.resolve())},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
