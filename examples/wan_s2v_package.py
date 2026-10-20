#!/usr/bin/env python3
"""Package a native Wan S2V F32 output and its original audio as an MP4.

This is final-output I/O only. It does not import or run a model. The command
accepts only the experimental five-second runner output, verifies its receipt
and input manifest, streams one BCTHW frame at a time to ffmpeg, and publishes
the encoded MP4 without replacing an existing path.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import BinaryIO, Sequence

import numpy as np


@dataclass(frozen=True)
class VideoSpec:
    """Geometry expected at the packaging boundary."""

    frames: int
    height: int
    width: int
    fps: int
    audio_sample_rate: int
    audio_samples: int
    canvas_height: int
    canvas_width: int

    def __post_init__(self) -> None:
        if min(self.frames, self.height, self.width, self.fps,
               self.audio_sample_rate, self.audio_samples,
               self.canvas_height, self.canvas_width) <= 0:
            raise ValueError("Video and audio dimensions must be positive")

    @property
    def tensor_shape(self) -> tuple[int, int, int, int, int]:
        return (1, 3, self.frames, self.height, self.width)

    @property
    def raw_bytes(self) -> int:
        return int(np.prod(self.tensor_shape, dtype=np.int64)) * 4


PRODUCTION_SPEC = VideoSpec(
    frames=80,
    height=480,
    width=832,
    fps=16,
    audio_sample_rate=16_000,
    audio_samples=80_000,
    canvas_height=512,
    canvas_width=832,
)
RUN_RECEIPT_SCHEMA = "wan_s2v_run_receipt_v1"
INPUT_MANIFEST_SCHEMA = "wan_s2v_native_inputs_v1"
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _sha256_stream(file: BinaryIO) -> str:
    digest = hashlib.sha256()
    file.seek(0)
    while chunk := file.read(HASH_CHUNK_BYTES):
        digest.update(chunk)
    file.seek(0)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as file:
        return _sha256_stream(file)


def _read_json(path: Path, description: str) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value, payload


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _safe_basename(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{field} must be a basename")
    return value


def _resolve_executable(value: str | None, name: str) -> str:
    candidate = value or name
    found = shutil.which(candidate)
    if found is None:
        raise ValueError(f"Cannot find executable {candidate!r} on PATH")
    return found


def _validate_input_manifest(inputs_dir: Path, spec: VideoSpec) -> tuple[Path, str, str]:
    manifest_path = inputs_dir / "manifest.json"
    manifest, manifest_bytes = _read_json(manifest_path, "input manifest")
    if manifest.get("schema") != INPUT_MANIFEST_SCHEMA:
        raise ValueError("Unsupported input manifest schema")

    geometry = _mapping(manifest.get("geometry"), "input manifest geometry")
    expected_geometry = {
        "canvas_height": spec.canvas_height,
        "canvas_width": spec.canvas_width,
        "deliver_frames": spec.frames,
        "fps": spec.fps,
    }
    for field, expected in expected_geometry.items():
        if _integer(geometry.get(field), f"input manifest geometry.{field}") != expected:
            raise ValueError(f"Input manifest geometry.{field} does not match packaging geometry")

    audio = _mapping(manifest.get("audio"), "input manifest audio")
    if (_integer(audio.get("sample_rate"), "input manifest audio.sample_rate") != spec.audio_sample_rate
            or _integer(audio.get("sample_count"), "input manifest audio.sample_count") != spec.audio_samples):
        raise ValueError("Input manifest audio metadata does not match packaging geometry")

    tensors = _mapping(manifest.get("tensors"), "input manifest tensors")
    audio_tensor = _mapping(tensors.get("audio.f32"), "input manifest audio.f32 tensor")
    if audio_tensor.get("dtype") != "F32_LE":
        raise ValueError("Input audio tensor dtype must be F32_LE")
    shape = audio_tensor.get("shape")
    if not isinstance(shape, list) or shape != [spec.audio_samples]:
        raise ValueError("Input audio tensor shape does not match the mono sample count")
    expected_audio_bytes = spec.audio_samples * 4
    if _integer(audio_tensor.get("bytes"), "input manifest audio.f32 bytes") != expected_audio_bytes:
        raise ValueError("Input audio tensor byte count does not match its shape")
    expected_audio_hash = audio_tensor.get("sha256")
    if not isinstance(expected_audio_hash, str) or not SHA256_RE.fullmatch(expected_audio_hash):
        raise ValueError("Input audio tensor SHA-256 is malformed")

    audio_name = _safe_basename(audio_tensor.get("file"), "input audio tensor file")
    if audio_name != "audio.f32":
        raise ValueError("Input manifest audio tensor must name audio.f32")
    audio_path = inputs_dir / audio_name
    try:
        with audio_path.open("rb") as audio_file:
            if os.fstat(audio_file.fileno()).st_size != expected_audio_bytes:
                raise ValueError("Input audio tensor size does not match its manifest")
            if _sha256_stream(audio_file) != expected_audio_hash:
                raise ValueError("Input audio tensor SHA-256 does not match its manifest")
            samples = np.fromfile(audio_file, dtype=np.dtype("<f4"), count=spec.audio_samples)
    except OSError as exc:
        raise ValueError(f"Cannot read input audio tensor {audio_path}: {exc}") from exc
    if samples.size != spec.audio_samples or not np.isfinite(samples).all():
        raise ValueError("Input audio tensor is truncated or contains non-finite values")
    return audio_path, hashlib.sha256(manifest_bytes).hexdigest(), expected_audio_hash


def _validate_receipt(raw_path: Path, receipt_path: Path, manifest_sha256: str,
                      spec: VideoSpec) -> tuple[dict[str, object], str]:
    receipt, _ = _read_json(receipt_path, "run receipt")
    if receipt.get("schema") != RUN_RECEIPT_SCHEMA:
        raise ValueError("Unsupported Wan S2V run receipt schema")
    if receipt.get("status") != "experimental_run_admitted":
        raise ValueError("Run receipt status must be experimental_run_admitted")

    recorded_output = receipt.get("output")
    if not isinstance(recorded_output, str) or not recorded_output:
        raise ValueError("Run receipt output identity is missing")
    recorded_path = Path(recorded_output).expanduser()
    if not recorded_path.is_absolute():
        recorded_path = Path.cwd() / recorded_path
    try:
        if recorded_path.resolve(strict=True) != raw_path.resolve(strict=True):
            raise ValueError("Run receipt output path does not identify --raw")
    except OSError as exc:
        raise ValueError(f"Cannot resolve run receipt output identity: {exc}") from exc

    try:
        raw_file = raw_path.open("rb")
    except OSError as exc:
        raise ValueError(f"Cannot read raw tensor {raw_path}: {exc}") from exc
    with raw_file:
        initial_stat = os.fstat(raw_file.fileno())
        if not os.path.isfile(raw_path) or initial_stat.st_size != spec.raw_bytes:
            raise ValueError(
                f"Raw tensor must be F32_LE BCTHW {spec.tensor_shape} ({spec.raw_bytes} bytes)"
            )
        receipt_bytes = _integer(receipt.get("output_bytes"), "run receipt output_bytes")
        if receipt_bytes != initial_stat.st_size:
            raise ValueError("Run receipt output byte count does not match --raw")
        expected_raw_hash = receipt.get("output_sha256")
        if not isinstance(expected_raw_hash, str) or not SHA256_RE.fullmatch(expected_raw_hash):
            raise ValueError("Run receipt output SHA-256 is malformed")
        raw_hash = _sha256_stream(raw_file)
        if raw_hash != expected_raw_hash:
            raise ValueError("Raw tensor SHA-256 does not match the run receipt")
        raw_stat = os.fstat(raw_file.fileno())
        if (raw_stat.st_size, raw_stat.st_mtime_ns) != (initial_stat.st_size, initial_stat.st_mtime_ns):
            raise ValueError("Raw tensor changed while its identity was being checked")

    expected_receipt_geometry = {
        "fps": spec.fps,
        "frames": spec.frames,
        "height": spec.height,
        "width": spec.width,
    }
    for field, expected in expected_receipt_geometry.items():
        if _integer(receipt.get(field), f"run receipt {field}") != expected:
            raise ValueError(f"Run receipt {field} does not match packaging geometry")
    if receipt.get("input_manifest_sha256") != manifest_sha256:
        raise ValueError("Run receipt input manifest hash does not match --inputs/manifest.json")

    source_revision = receipt.get("source_revision")
    if not isinstance(source_revision, str) or not REVISION_RE.fullmatch(source_revision):
        raise ValueError("Run receipt source_revision must be a 40-character Git SHA")
    return receipt, raw_hash


def _write_frames(process: subprocess.Popen[bytes], raw_path: Path, spec: VideoSpec) -> None:
    if process.stdin is None:
        raise RuntimeError("ffmpeg did not provide a video input pipe")
    try:
        tensor = np.memmap(raw_path, dtype=np.dtype("<f4"), mode="r", shape=spec.tensor_shape)
        frame = np.empty((3, spec.height, spec.width), dtype=np.float32)
        for frame_index in range(spec.frames):
            np.copyto(frame, tensor[0, :, frame_index, :, :], casting="no")
            if not np.isfinite(frame).all():
                raise ValueError(f"Raw tensor frame {frame_index} contains non-finite values")
            minimum = float(frame.min())
            maximum = float(frame.max())
            if minimum < -1.0 or maximum > 1.0:
                raise ValueError(
                    f"Raw tensor frame {frame_index} is outside [-1, 1]: [{minimum}, {maximum}]"
                )
            # The native output contract is [-1, 1]. Convert each frame to
            # RGB24 in place after validating the range; no implicit clipping.
            frame += np.float32(1.0)
            frame *= np.float32(127.5)
            np.rint(frame, out=frame)
            rgb = np.ascontiguousarray(np.moveaxis(frame.astype(np.uint8), 0, -1))
            process.stdin.write(memoryview(rgb))
            del rgb
    finally:
        if "tensor" in locals():
            del tensor


def _ffprobe(staged_path: Path, ffprobe_path: str, spec: VideoSpec) -> None:
    command = [
        ffprobe_path,
        "-v", "error",
        "-count_frames",
        "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,duration,sample_rate,channels:format=duration",
        "-of", "json",
        str(staged_path),
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        report = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "stderr", None)
        if detail:
            detail = detail.strip()[-2000:]
        raise RuntimeError(f"ffprobe could not validate the staged MP4{': ' + detail if detail else ''}") from exc

    streams = report.get("streams")
    if not isinstance(streams, list):
        raise RuntimeError("ffprobe returned no stream list for the staged MP4")
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise RuntimeError("Staged MP4 must contain exactly one video and one audio stream")
    video_stream = video[0]
    audio_stream = audio[0]
    format_info = report.get("format")
    if not isinstance(format_info, dict):
        raise RuntimeError("ffprobe returned no format information for the staged MP4")

    expected_video_duration = Decimal(spec.frames) / Decimal(spec.fps)
    expected_audio_duration = Decimal(spec.audio_samples) / Decimal(spec.audio_sample_rate)
    if abs(expected_video_duration - expected_audio_duration) > Decimal(1) / Decimal(spec.audio_sample_rate):
        raise RuntimeError("Packaging video and audio durations differ by more than one audio sample")

    def validate_duration(value: object, field: str, expected: Decimal) -> None:
        if not isinstance(value, str):
            raise RuntimeError(f"Staged MP4 {field} duration is missing or malformed")
        try:
            actual = Decimal(value)
        except InvalidOperation as exc:
            raise RuntimeError(f"Staged MP4 {field} duration is malformed: {value!r}") from exc
        if not actual.is_finite() or actual < 0:
            raise RuntimeError(f"Staged MP4 {field} duration is malformed: {value!r}")
        # ffprobe's decimal duration is rounded at the least significant
        # reported place. Allow half of that unit plus one audio sample.
        reported_unit = Decimal(1).scaleb(actual.as_tuple().exponent)
        tolerance = (Decimal(1) / Decimal(spec.audio_sample_rate)) + abs(reported_unit) / 2
        if abs(actual - expected) > tolerance:
            raise RuntimeError(
                f"Staged MP4 {field} duration is {value!r}s; expected "
                f"{float(expected):.9f}s within one audio sample plus ffprobe rounding"
            )

    validate_duration(video_stream.get("duration"), "video", expected_video_duration)
    validate_duration(audio_stream.get("duration"), "audio", expected_audio_duration)
    # Production packaging is fixed at 80 frames / 16 fps, i.e. five seconds.
    # Use the spec-derived value to keep the CPU-only geometry fixture useful.
    validate_duration(format_info.get("duration"), "container", expected_video_duration)

    expected_video = {
        "codec_name": "h264",
        "width": spec.width,
        "height": spec.height,
        "r_frame_rate": f"{spec.fps}/1",
        "nb_read_frames": str(spec.frames),
    }
    for field, expected in expected_video.items():
        if video_stream.get(field) != expected:
            raise RuntimeError(f"Staged MP4 video {field} is {video_stream.get(field)!r}; expected {expected!r}")
    expected_audio = {
        "codec_name": "aac",
        "sample_rate": str(spec.audio_sample_rate),
        "channels": 1,
    }
    for field, expected in expected_audio.items():
        if audio_stream.get(field) != expected:
            raise RuntimeError(f"Staged MP4 audio {field} is {audio_stream.get(field)!r}; expected {expected!r}")


def package_video(raw_path: Path, receipt_path: Path, inputs_dir: Path, output_path: Path, *,
                  spec: VideoSpec = PRODUCTION_SPEC, ffmpeg: str | None = None,
                  ffprobe: str | None = None) -> dict[str, object]:
    """Validate and package a run; ``spec`` exists for small CPU-only fixtures."""
    raw_path = raw_path.expanduser()
    receipt_path = receipt_path.expanduser()
    inputs_dir = inputs_dir.expanduser()
    output_path = output_path.expanduser()
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")
    if not output_path.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output_path.parent}")
    if os.path.lexists(output_path):
        raise FileExistsError(f"Output path already exists: {output_path}")
    if not raw_path.is_file():
        raise ValueError(f"Raw tensor does not exist or is not a regular file: {raw_path}")
    if not inputs_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {inputs_dir}")

    ffmpeg_path = _resolve_executable(ffmpeg, "ffmpeg")
    ffprobe_path = _resolve_executable(ffprobe, "ffprobe")
    audio_path, manifest_sha256, audio_sha256 = _validate_input_manifest(inputs_dir, spec)
    receipt, raw_sha256 = _validate_receipt(raw_path, receipt_path, manifest_sha256, spec)
    raw_before = raw_path.stat()
    audio_before = audio_path.stat()

    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.package-", dir=output_path.parent) as workdir:
        staged_path = Path(workdir) / output_path.name
        command = [
            ffmpeg_path,
            "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{spec.width}x{spec.height}",
            "-framerate", str(spec.fps),
            "-i", "pipe:0",
            "-f", "f32le", "-ar", str(spec.audio_sample_rate), "-ac", "1",
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-frames:v", str(spec.frames),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-f", "mp4", str(staged_path),
        ]
        with tempfile.TemporaryFile(mode="w+b") as ffmpeg_stderr:
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=ffmpeg_stderr)
            except OSError as exc:
                raise RuntimeError(f"Cannot start ffmpeg: {exc}") from exc
            try:
                _write_frames(process, raw_path, spec)
                if process.stdin is not None:
                    process.stdin.close()
                return_code = process.wait()
            except BaseException:
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                raise
            ffmpeg_stderr.seek(0)
            error_text = ffmpeg_stderr.read().decode("utf-8", errors="replace").strip()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg exited with status {return_code}: {error_text[-2000:]}")

        if not staged_path.is_file() or staged_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not produce a nonempty MP4")
        _ffprobe(staged_path, ffprobe_path, spec)

        raw_after = raw_path.stat()
        audio_after = audio_path.stat()
        if (raw_after.st_size, raw_after.st_mtime_ns) != (raw_before.st_size, raw_before.st_mtime_ns):
            raise ValueError("Raw tensor changed while packaging")
        if (audio_after.st_size, audio_after.st_mtime_ns) != (audio_before.st_size, audio_before.st_mtime_ns):
            raise ValueError("Input audio tensor changed while packaging")
        if _sha256_path(audio_path) != audio_sha256:
            raise ValueError("Input audio tensor SHA-256 changed while packaging")
        if _sha256_path(raw_path) != raw_sha256:
            raise ValueError("Raw tensor SHA-256 changed while packaging")

        # A same-filesystem hard link publishes atomically and fails if another
        # process created the destination after the initial existence check.
        try:
            os.link(staged_path, output_path)
        except FileExistsError:
            raise FileExistsError(f"Output path appeared during packaging: {output_path}")

    return {
        "schema": "wan_s2v_mp4_package_v1",
        "packaging_status": "completed",
        "run_status": receipt["status"],
        "source_revision": receipt["source_revision"],
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256_path(output_path),
        "fps": spec.fps,
        "frames": spec.frames,
        "height": spec.height,
        "width": spec.width,
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_sample_rate": spec.audio_sample_rate,
        "audio_samples": spec.audio_samples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="native F32_LE BCTHW output from wan_s2v_pipeline.cr")
    parser.add_argument("--receipt", type=Path,
                        help="run receipt JSON (default: append .json to --raw)")
    parser.add_argument("--inputs", type=Path, required=True,
                        help="input directory produced by wan_s2v_inputs.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="new MP4 output path; existing paths are rejected")
    parser.add_argument("--ffmpeg", help="ffmpeg executable (default: PATH lookup)")
    parser.add_argument("--ffprobe", help="ffprobe executable (default: PATH lookup)")
    args = parser.parse_args(argv)

    raw_path = args.raw.expanduser()
    receipt_path = args.receipt.expanduser() if args.receipt else Path(f"{raw_path}.json")
    try:
        result = package_video(raw_path, receipt_path, args.inputs, args.output,
                               ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
