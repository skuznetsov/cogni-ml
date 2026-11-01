#!/usr/bin/env python3
"""Validate and render an offline gallery for completed Hy-Nomas drafts.

The ordinary gallery refresh is incremental and cheap. Pass ``--validate-final``
to stream-hash every completed raw pixel tensor and run ffprobe on each MP4.
This file never runs a model or an encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


SUITE_SCHEMA = "cogni_ml.hy_nomas_draft_suite/v1"
RESULTS_SCHEMA = "cogni_ml.hy_nomas_draft_results/v1"
RECEIPT_SCHEMA = "cogni_ml.hy_nomas_native_continuation/v1"
RAW_PIXELS_BYTES = 424_673_280
FRAME_COUNT = 240
FRAME_RATE = 24
VIDEO_SECONDS = 10.0
WIDTH = 512
HEIGHT = 288
LATENT_FRAMES = 60
STEPS_PER_SEGMENT = 4
DECODER_WINDOWS = 11
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class DraftResultsError(ValueError):
    """A suite, result, receipt, or media artifact violates the contract."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise DraftResultsError(message)


def _sha256_string(value: Any, where: str) -> str:
    _need(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
          f"{where} must be a lowercase SHA-256")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _need(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _integer(value: Any, expected: int, where: str) -> None:
    _need(type(value) is int and value == expected,
          f"{where} must be {expected}")


def _finite_nonnegative(value: Any, where: str) -> float:
    _need(type(value) in (int, float) and math.isfinite(float(value))
          and float(value) >= 0.0, f"{where} must be finite and nonnegative")
    return float(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rawpixel_size(actual_bytes: int, expected_bytes: int = RAW_PIXELS_BYTES) -> None:
    _integer(actual_bytes, expected_bytes, "raw pixel byte count")


def validate_rawpixels(path: Path, expected_sha256: str,
                       expected_bytes: int = RAW_PIXELS_BYTES) -> dict[str, Any]:
    """Check the exact byte count and streaming SHA-256 of a raw pixel tensor.

    ``expected_bytes`` is injectable so the unit tests can use tiny fixtures;
    the production/final CLI always uses ``RAW_PIXELS_BYTES``.
    """
    expected_sha256 = _sha256_string(expected_sha256, "raw pixel SHA-256")
    _need(path.is_file(), f"raw pixel file is missing: {path}")
    actual_bytes = path.stat().st_size
    validate_rawpixel_size(actual_bytes, expected_bytes)
    actual_sha256 = sha256_file(path)
    _need(actual_sha256 == expected_sha256, "raw pixel SHA-256 mismatch")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}


def validate_suite(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    _need(isinstance(suite, Mapping) and suite.get("schema") == SUITE_SCHEMA,
          "unsupported draft suite schema")
    settings = suite.get("settings")
    _need(isinstance(settings, Mapping), "suite settings must be an object")
    expected_settings = {
        "segments": 2,
        "seconds": 10,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FRAME_RATE,
        "steps_per_segment": STEPS_PER_SEGMENT,
        "weights": "original BF16",
        "history": "F32 once-per-scene text cache",
        "noise": "The same authenticated seed-42 noise banks for every scene; prompt-only variation.",
        "selection": "One first take per scene, including visible failures.",
    }
    for key, expected in expected_settings.items():
        _need(settings.get(key) == expected,
              f"suite setting {key!r} must be {expected!r}")

    scenes = suite.get("scenes")
    _need(isinstance(scenes, list) and scenes, "suite scenes must be a nonempty list")
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        where = f"suite.scenes[{index}]"
        _need(isinstance(scene, dict), f"{where} must be an object")
        scene_id = scene.get("id")
        _need(isinstance(scene_id, str) and SCENE_ID_RE.fullmatch(scene_id) is not None,
              f"{where}.id is invalid")
        _need(scene_id not in seen, f"duplicate scene id: {scene_id}")
        seen.add(scene_id)
        _need(isinstance(scene.get("title"), str) and scene["title"].strip(),
              f"{where}.title is required")
        _need(isinstance(scene.get("prompt"), str) and scene["prompt"].strip(),
              f"{where}.prompt is required")
        checks = scene.get("checks", [])
        _need(isinstance(checks, list) and all(isinstance(item, str) for item in checks),
              f"{where}.checks must be a list of strings")
        checked.append(scene)
    return checked


def validate_receipt_metadata(receipt: Mapping[str, Any], prompt: str,
                              inputs_sha256: str) -> dict[str, Any]:
    """Validate the native two-segment sequence receipt without reading media."""
    _need(isinstance(receipt, Mapping) and receipt.get("schema") == RECEIPT_SCHEMA,
          "native sequence receipt schema mismatch")
    _need(receipt.get("prompt") == prompt, "native receipt prompt differs from suite prompt")
    inputs_sha256 = _sha256_string(inputs_sha256, "expected inputs SHA-256")
    receipt_inputs_sha = _sha256_string(receipt.get("inputs_sha256"),
                                        "native receipt inputs_sha256")
    _need(receipt_inputs_sha == inputs_sha256,
          "native receipt inputs_sha256 differs from result provenance")
    _need(receipt.get("full_upstream_video_parity") is False,
          "receipt must explicitly disclaim full upstream video parity")

    for field, expected in (("segments", 2), ("frames", FRAME_COUNT),
                            ("clean_latent_frames", LATENT_FRAMES),
                            ("width", WIDTH), ("height", HEIGHT),
                            ("fps", FRAME_RATE),
                            ("steps_per_segment", STEPS_PER_SEGMENT)):
        _integer(receipt.get(field), expected, f"receipt.{field}")
    _need(receipt.get("latent_bank_shape") == [1, 32, 180, 9, 16],
          "receipt latent bank shape mismatch")
    _need(receipt.get("cache_precision") == "F32 exact native K/V boundary; upstream cache is BF16",
          "receipt cache precision mismatch")
    _need(receipt.get("noise_scope") ==
          "authenticated MPS-generated fixed banks; stock upstream CPU RNG value parity is false",
          "receipt noise scope does not match the fixed authenticated-bank contract")

    pixelumm = receipt.get("pixelumm")
    _need(isinstance(pixelumm, Mapping), "receipt.pixelumm must be an object")
    for field, expected in (("steps", 1), ("window_latents", 10),
                            ("overlap_latents", 5), ("global_windows", DECODER_WINDOWS)):
        _integer(pixelumm.get(field), expected, f"receipt.pixelumm.{field}")

    metrics = receipt.get("metrics")
    _need(isinstance(metrics, Mapping), "receipt.metrics must be an object")
    for phase in ("dit", "decoder"):
        for metric in ("encode_allocations", "encode_readbacks", "preterminal_waits"):
            _integer(metrics.get(f"{phase}_{metric}"), 0,
                     f"receipt.metrics.{phase}_{metric}")
        _integer(metrics.get(f"{phase}_waits"), 1,
                 f"receipt.metrics.{phase}_waits")
        segments = metrics.get(f"{phase}_segments")
        _need(type(segments) is int and segments > 0,
              f"receipt.metrics.{phase}_segments must be positive")

    diagnostics = receipt.get("diagnostics")
    _need(isinstance(diagnostics, Mapping), "receipt.diagnostics must be an object")
    pixel_path = diagnostics.get("pixels")
    _need(isinstance(pixel_path, str) and pixel_path.strip(),
          "receipt.diagnostics.pixels is required")
    pixel_sha = _sha256_string(diagnostics.get("pixels_sha256"),
                               "receipt.diagnostics.pixels_sha256")
    return {"inputs_sha256": receipt_inputs_sha,
            "pixels": pixel_path,
            "pixels_sha256": pixel_sha}


def validate_native_prompt(native_prompt: str, suite_prompt: str) -> None:
    """Bind the exact native JSON prompt to one suite scene description."""
    _need(isinstance(native_prompt, str) and native_prompt,
          "native_prompt must be the exact nonempty text passed to inference")
    try:
        document = json.loads(native_prompt, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise DraftResultsError("native_prompt is not valid JSON") from exc
    _need(isinstance(document, dict), "native_prompt must contain a JSON object")
    _need(document.get("Style") == "Photorealistic real-world footage",
          "native_prompt style mismatch")
    _need(document.get("perspective_view") == "third-person observer",
          "native_prompt perspective mismatch")
    _need(isinstance(document.get("subject_list"), list),
          "native_prompt subject_list must be a list")
    segments = document.get("segments")
    _need(isinstance(segments, list) and len(segments) == 1 and isinstance(segments[0], dict),
          "native_prompt must contain exactly one scene segment")
    segment = segments[0]
    _need(segment.get("start_time") == "00:00.000" and
          segment.get("end_time") == "00:10.000",
          "native_prompt segment must span exactly ten seconds")
    _need(segment.get("visual_description_long") == suite_prompt,
          "native_prompt visual description differs from suite prompt")
    _need(segment.get("description_change") is None,
          "native_prompt description_change must be null")
    _need(segment.get("contained_actions") == [],
          "native_prompt must not add contained actions")


def validate_metadata(receipt: Mapping[str, Any], native_prompt: str,
                      inputs_sha256: str) -> dict[str, Any]:
    """Short API used by the runner for cheap per-scene receipt checks."""
    return validate_receipt_metadata(receipt, native_prompt, inputs_sha256)


def _path_under(root: Path, relative: Any, where: str) -> Path:
    _need(isinstance(relative, str) and relative.strip(), f"{where} is required")
    pure = PurePosixPath(relative)
    _need(not pure.is_absolute() and not any(part in ("", ".", "..") for part in pure.parts)
          and "\\" not in relative and "%" not in relative,
          f"{where} must be a safe relative local path")
    _need(all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in pure.parts),
          f"{where} contains an unsafe path component")
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*pure.parts)).resolve()
    _need(candidate.is_relative_to(resolved_root), f"{where} escapes the artifact root")
    return candidate


def _receipt_file(path: Path) -> Mapping[str, Any]:
    _need(path.is_file(), f"receipt file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=_unique_json_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise DraftResultsError(f"cannot read receipt JSON: {path}") from exc
    _need(isinstance(data, dict), "receipt JSON must contain an object")
    return data


def ffprobe_mp4(path: Path, ffprobe: str = "ffprobe") -> Mapping[str, Any]:
    executable = shutil.which(ffprobe) if Path(ffprobe).name == ffprobe else ffprobe
    _need(executable is not None and Path(executable).is_file(),
          f"ffprobe executable not found: {ffprobe}")
    command = [str(executable), "-v", "error", "-count_frames", "-select_streams", "v:0",
               "-show_entries",
               "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames:format=duration",
               "-of", "json", str(path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True,
                                   text=True, timeout=120)
        data = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise DraftResultsError(f"ffprobe failed for {path}") from exc
    _need(isinstance(data, dict), "ffprobe output must be an object")
    return data


def _frame_rate(value: Any, where: str) -> None:
    _need(isinstance(value, str) and "/" in value,
          f"ffprobe {where} must be a rational frame rate")
    numerator, denominator = value.split("/", 1)
    try:
        rate = int(numerator) / int(denominator)
    except (ValueError, ZeroDivisionError) as exc:
        raise DraftResultsError(f"ffprobe {where} is malformed") from exc
    _need(math.isclose(rate, FRAME_RATE, rel_tol=0.0, abs_tol=1e-9),
          f"ffprobe {where} must equal 24 fps")


def validate_ffprobe(probe: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(probe, Mapping), "ffprobe result must be an object")
    streams = probe.get("streams")
    _need(isinstance(streams, list) and len(streams) == 1 and isinstance(streams[0], Mapping),
          "ffprobe must report exactly one selected video stream")
    stream = streams[0]
    _need(stream.get("codec_type") == "video", "ffprobe selected stream is not video")
    _integer(stream.get("width"), WIDTH, "ffprobe width")
    _integer(stream.get("height"), HEIGHT, "ffprobe height")
    for field in ("r_frame_rate", "avg_frame_rate"):
        _frame_rate(stream.get(field), field)
    try:
        decoded_frames = int(stream.get("nb_read_frames"))
    except (TypeError, ValueError) as exc:
        raise DraftResultsError("ffprobe did not count decoded frames") from exc
    _integer(decoded_frames, FRAME_COUNT, "ffprobe decoded frame count")
    if stream.get("nb_frames") not in (None, "N/A"):
        try:
            declared_frames = int(stream["nb_frames"])
        except (TypeError, ValueError) as exc:
            raise DraftResultsError("ffprobe declared frame count is malformed") from exc
        _integer(declared_frames, FRAME_COUNT, "ffprobe declared frame count")
    format_info = probe.get("format")
    _need(isinstance(format_info, Mapping), "ffprobe format information is required")
    raw_duration = format_info.get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise DraftResultsError("ffprobe duration must be numeric") from exc
    _need(math.isfinite(duration) and duration >= 0.0,
          "ffprobe duration must be finite and nonnegative")
    _need(math.isclose(duration, VIDEO_SECONDS, rel_tol=0.0, abs_tol=1e-3),
          "ffprobe duration must be 10 seconds")
    return {"frames": decoded_frames, "fps": FRAME_RATE, "seconds": duration,
            "width": WIDTH, "height": HEIGHT}


def _scene_inputs_sha(result: Mapping[str, Any]) -> str:
    # Native receipts use ``inputs_sha256``. ``input_sha256`` is accepted for
    # early result writers that used the singular spelling; either is bound to
    # the receipt before acceptance.
    plural = result.get("inputs_sha256")
    singular = result.get("input_sha256")
    _need(not (plural is not None and singular is not None and plural != singular),
          "result input SHA aliases disagree")
    return _sha256_string(plural if plural is not None else singular,
                          "result inputs_sha256")


def _validate_timings_and_memory(result: Mapping[str, Any]) -> None:
    timings = result.get("timings")
    _need(isinstance(timings, Mapping) and timings,
          "completed result timings must be a nonempty object")
    for key, value in timings.items():
        _finite_nonnegative(value, f"result.timings.{key}")
    _finite_nonnegative(result.get("memory_peak_gib"), "result.memory_peak_gib")


def validate_completed_result(scene: Mapping[str, Any], result: Mapping[str, Any],
                              artifact_root: Path, *, full_media_validation: bool = False,
                              ffprobe: str = "ffprobe",
                              probe_function: Callable[[Path], Mapping[str, Any]] | None = None
                              ) -> dict[str, Any]:
    """Validate one completed run. Full validation hashes pixels and probes MP4."""
    scene_id = scene["id"]
    _need(result.get("id") == scene_id, "result scene id does not match suite")
    _need(result.get("status") in ("complete", "completed"),
          f"scene {scene_id} is not complete")
    inputs_sha = _scene_inputs_sha(result)
    native_prompt = result.get("native_prompt")
    validate_native_prompt(native_prompt, scene["prompt"])
    expected_prompt_sha = hashlib.sha256(native_prompt.encode("utf-8")).hexdigest()
    prompt_sha = _sha256_string(result.get("prompt_sha256"), "result.prompt_sha256")
    _need(prompt_sha == expected_prompt_sha, "result prompt_sha256 differs from native_prompt")
    _validate_timings_and_memory(result)

    receipt_path = _path_under(artifact_root, result.get("receipt"), "result.receipt")
    receipt = _receipt_file(receipt_path)
    metadata = validate_receipt_metadata(receipt, native_prompt, inputs_sha)
    pixels_sha = _sha256_string(result.get("pixels_sha256"), "result.pixels_sha256")
    _need(pixels_sha == metadata["pixels_sha256"],
          "result pixels_sha256 differs from receipt diagnostics")
    video_sha = _sha256_string(result.get("video_sha256"), "result.video_sha256")

    video_path = _path_under(artifact_root, result.get("video"), "result.video")
    _need(video_path.is_file(), f"completed MP4 is missing: {video_path}")
    video_href = PurePosixPath(result["video"]).as_posix()
    media_probe = None
    if full_media_validation:
        raw_value = Path(metadata["pixels"])
        raw_path = raw_value if raw_value.is_absolute() else receipt_path.parent / raw_value
        raw_path = raw_path.resolve()
        _need(raw_path.is_relative_to(artifact_root.resolve()),
              "receipt raw pixels path escapes the artifact root")
        raw_report = validate_rawpixels(raw_path, metadata["pixels_sha256"])
        _need(isinstance(raw_report, Mapping) and raw_report.get("sha256") == pixels_sha,
              "raw pixel bytes differ from result and receipt digest")
        actual_video_sha = sha256_file(video_path)
        _need(actual_video_sha == video_sha,
              "current MP4 SHA-256 differs from result video_sha256")
        probe_data = (probe_function(video_path) if probe_function
                      else ffprobe_mp4(video_path, ffprobe))
        media_probe = validate_ffprobe(probe_data)
    else:
        raw_report = {"bytes_expected": RAW_PIXELS_BYTES,
                      "sha256": metadata["pixels_sha256"], "validated": False}
    return {"id": scene_id, "video_path": video_path, "video_href": video_href,
            "receipt_path": receipt_path, "prompt_sha256": prompt_sha,
            "inputs_sha256": inputs_sha, "pixels_sha256": pixels_sha,
            "video_sha256": video_sha, "raw_pixels": raw_report,
            "probe": media_probe}


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source, object_pairs_hook=_unique_json_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise DraftResultsError(f"cannot read JSON: {path}") from exc


def _results_list(document: Any, expected_suite_sha256: str | None = None) -> list[dict[str, Any]]:
    _need(isinstance(document, dict) and document.get("schema") == RESULTS_SCHEMA,
          "unsupported draft results schema")
    bound_sha = _sha256_string(document.get("suite_sha256"), "results.suite_sha256")
    if expected_suite_sha256 is not None:
        _need(bound_sha == expected_suite_sha256,
              "results suite_sha256 differs from the loaded suite manifest")
    rows = document.get("scenes")
    _need(isinstance(rows, list), "results must be a list")
    _need(all(isinstance(row, dict) for row in rows), "each result must be an object")
    return rows


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def render_gallery(suite: Mapping[str, Any], results_document: Any,
                   gallery_path: Path, artifact_root: Path,
                   *, full_media_validation: bool = False,
                   suite_sha256: str | None = None,
                   ffprobe: str = "ffprobe",
                   probe_function: Callable[[Path], Mapping[str, Any]] | None = None
                   ) -> dict[str, Any]:
    scenes = validate_suite(suite)
    result_rows = _results_list(results_document, suite_sha256)
    by_id: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        scene_id = row.get("id")
        _need(isinstance(scene_id, str) and scene_id not in by_id,
              "results contain an invalid or duplicate scene id")
        by_id[scene_id] = row
    scene_ids = {scene["id"] for scene in scenes}
    _need(set(by_id).issubset(scene_ids), "results contain a scene absent from the suite")

    cards: list[str] = []
    summary: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = scene["id"]
        result = by_id.get(scene_id)
        title = html.escape(scene["title"])
        prompt = html.escape(scene["prompt"])
        checks = scene.get("checks", [])
        checks_html = "".join(f"<li>{html.escape(item)}</li>" for item in checks)
        if result is None:
            cards.append(f'<article class="card pending"><h2>{title}</h2>'
                         f'<p class="state">Pending</p><p>{prompt}</p>'
                         f'<ul>{checks_html}</ul></article>')
            summary.append({"id": scene_id, "status": "pending"})
            continue

        status = result.get("status")
        if status in ("complete", "completed"):
            checked = validate_completed_result(
                scene, result, artifact_root,
                full_media_validation=full_media_validation,
                ffprobe=ffprobe, probe_function=probe_function)
            timings = result["timings"]
            time_items = "".join(
                f"<li>{html.escape(str(key))}: {html.escape(_display_value(value))} s</li>"
                for key, value in timings.items())
            memory = html.escape(_display_value(result["memory_peak_gib"]))
            href = html.escape(checked["video_href"], quote=True)
            provenance = result.get("provenance", {})
            provenance_text = html.escape(_display_value(provenance)) if provenance else ""
            probe_line = ""
            if checked["probe"]:
                probe = checked["probe"]
                probe_line = (f'<p class="facts">{probe["frames"]} decoded frames · '
                              f'{probe["fps"]} fps · {probe["seconds"]:.3f} s · '
                              f'{probe["width"]}×{probe["height"]}</p>')
            provenance_html = (f'<details><summary>Provenance</summary><pre>{provenance_text}</pre></details>'
                               if provenance_text else "")
            cards.append(f'<article class="card complete"><h2>{title}</h2>'
                         f'<p class="state">Completed first take</p>'
                         f'<video controls preload="metadata" src="{href}"></video>'
                         f'{probe_line}<p>{prompt}</p><ul>{checks_html}</ul>'
                         f'<p class="facts">Sampled process peak: {memory} GiB</p>'
                         f'<ul class="timings">{time_items}</ul>{provenance_html}</article>')
            # Keep the public CLI report JSON-native. The validator uses Path
            # instances internally for containment checks and file I/O, but
            # those are not JSON values and should not leak into this report.
            summary.append({
                "id": scene_id,
                "status": "completed",
                "video_path": str(checked["video_path"]),
                "video_href": checked["video_href"],
                "receipt_path": str(checked["receipt_path"]),
                "prompt_sha256": checked["prompt_sha256"],
                "inputs_sha256": checked["inputs_sha256"],
                "pixels_sha256": checked["pixels_sha256"],
                "video_sha256": checked["video_sha256"],
                "raw_pixels": checked["raw_pixels"],
                "probe": checked["probe"],
            })
        elif status in ("pending", "running", "failed"):
            detail = result.get("detail", result.get("error", ""))
            detail_html = f'<p class="detail">{html.escape(str(detail))}</p>' if detail else ""
            cards.append(f'<article class="card {status}"><h2>{title}</h2>'
                         f'<p class="state">{html.escape(status.title())}</p>'
                         f'{detail_html}<p>{prompt}</p><ul>{checks_html}</ul></article>')
            summary.append({"id": scene_id, "status": status})
        else:
            raise DraftResultsError(f"scene {scene_id} has unsupported status {status!r}")

    suite_title = html.escape(str(suite.get("title", "Hy-Nomas draft gallery")))
    settings = suite["settings"]
    settings_html = " · ".join(
        f"{html.escape(str(label))}: {html.escape(_display_value(settings[key]))}"
        for key, label in (("weights", "Weights"), ("history", "History"),
                           ("noise", "Noise"), ("selection", "Selection")))
    notice = ("BF16 weights · fixed same-noise banks · perceptual review only. "
              "This gallery does not claim upstream end-to-end video parity.")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{suite_title}</title>
<style>
:root{{color-scheme:dark;--bg:#10141b;--panel:#1a222d;--text:#edf1f7;--muted:#aeb8c6;--accent:#8ec5ff;--ok:#8ad5a0;--warn:#f4c56b;--bad:#ff9b9b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1440px;margin:auto;padding:28px}}h1{{margin:0 0 8px;font-size:2rem}}h2{{margin:0 0 6px;font-size:1.25rem}}
.notice{{margin:20px 0;padding:14px 18px;border:1px solid #ad8645;border-radius:10px;background:#302718;color:#ffe0a2;font-weight:650}}
.settings{{color:var(--muted);font-size:.9rem;margin-bottom:22px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:18px}}
.card{{background:var(--panel);border:1px solid #303b49;border-radius:14px;padding:16px;min-width:0}}
.card.complete{{border-color:#477d58}}.card.failed{{border-color:#9c5353}}.card.pending,.card.running{{border-color:#806d40}}
.state{{font-weight:700;color:var(--ok);margin:2px 0 12px}}.pending .state,.running .state{{color:var(--warn)}}.failed .state{{color:var(--bad)}}
video{{display:block;width:100%;aspect-ratio:16/9;background:#080a0d;border-radius:8px;margin:12px 0}}
p{{overflow-wrap:anywhere}}ul{{padding-left:22px;color:#d9e0e9}}.facts{{color:var(--muted);font-size:.9rem}}.detail{{color:var(--bad)}}
details{{margin-top:10px;color:var(--muted)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}footer{{margin:26px 0;color:var(--muted);font-size:.85rem}}
</style></head><body><main>
<h1>{suite_title}</h1><div class="notice">{html.escape(notice)}</div>
<p class="settings">{settings_html}</p><section class="grid">{''.join(cards)}</section>
<footer>First takes are shown without curation, including failures. Video links are local files; this page makes no network requests.</footer>
</main></body></html>
"""
    gallery_path = gallery_path.resolve()
    _need(gallery_path.parent.resolve() == artifact_root.resolve(),
          "gallery must be written at the artifact root for local relative video links")
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    gallery_path.write_text(document, encoding="utf-8")
    complete_count = sum(row["status"] == "completed" for row in summary)
    final_passed = (complete_count == len(scenes)) if full_media_validation else None
    return {"schema": "cogni_ml.hy_nomas_draft_gallery_report/v1",
            "gallery": str(gallery_path),
            "full_media_validation": full_media_validation,
            "passed": final_passed,
            "scenes": summary}


def write_gallery(suite: Mapping[str, Any], results_document: Any,
                  gallery_path: Path, artifact_root: Path,
                  **kwargs: Any) -> dict[str, Any]:
    """Compatibility name for callers that incrementally refresh the page."""
    return render_gallery(suite, results_document, gallery_path, artifact_root, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True,
                        help="photorealistic draft suite manifest")
    parser.add_argument("--results", type=Path, required=True,
                        help="incremental result index JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="write index.html at the artifact root")
    parser.add_argument("--validate-final", action="store_true",
                        help="stream-hash all completed raw pixels and run ffprobe")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    try:
        suite = _load_json(args.suite)
        results = _load_json(args.results)
        suite_sha = sha256_file(args.suite)
        artifact_root = args.results.resolve().parent
        report = render_gallery(suite, results, args.output, artifact_root,
                                full_media_validation=args.validate_final,
                                suite_sha256=suite_sha,
                                ffprobe=args.ffprobe)
    except DraftResultsError as exc:
        print(f"draft gallery validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if args.validate_final and report["passed"] is not True:
        print("final validation incomplete: every suite scene needs a completed result",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
