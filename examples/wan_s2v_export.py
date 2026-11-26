#!/usr/bin/env python3
"""Export a completed native BCTHW F32 video to MP4, optionally adding audio.

This terminal file operation performs no neural inference. The native runner
must finish its GPU chain before invoking it. Frames are mapped and converted
one at a time rather than copying the complete decoded video into host RAM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np


def frame_to_rgb24(frame: np.ndarray, pixel_range: str = "minus_one_to_one") -> np.ndarray:
    """Convert one CHW F32 frame to RGB24 using an explicit source range."""
    if pixel_range not in ("minus_one_to_one", "zero_to_one"):
        raise ValueError("pixel_range must be minus_one_to_one or zero_to_one")
    values = np.array(frame, dtype=np.float32, copy=True)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("frame must have CHW shape with exactly three channels")
    if not np.isfinite(values).all():
        raise ValueError("non-finite decoded frame")
    if pixel_range == "minus_one_to_one":
        # Keep the pre-existing Wan save_video operation order unchanged.
        np.clip(values, -1, 1, out=values)
        values += np.float32(1)
        values /= np.float32(2)
        values *= np.float32(255)
    else:
        # Hy-Nomas video_to_uint8_numpy computes (frame * 255).clamp(0, 255).
        values *= np.float32(255)
        np.clip(values, 0, 255, out=values)
    return np.ascontiguousarray(values.transpose(1, 2, 0), dtype=np.uint8)


def export_video(raw: Path, output: Path, frames: int, height: int, width: int,
                 fps: int, audio: Path | None = None,
                 pixel_range: str = "minus_one_to_one") -> dict:
    if min(frames, height, width, fps) <= 0 or height % 2 or width % 2:
        raise ValueError("Positive dimensions and even H/W are required for YUV420 MP4")
    if pixel_range not in ("minus_one_to_one", "zero_to_one"):
        raise ValueError("pixel_range must be minus_one_to_one or zero_to_one")
    if output.exists():
        raise FileExistsError(output)
    if output.suffix.lower() != ".mp4" or not output.parent.is_dir():
        raise ValueError("Output must be an MP4 in an existing directory")
    expected_bytes = 3 * frames * height * width * 4
    if not raw.is_file() or raw.stat().st_size != expected_bytes:
        raise ValueError(f"Expected exactly {expected_bytes} BCTHW F32 bytes")
    if audio is not None and not audio.is_file():
        raise FileNotFoundError(audio)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for terminal MP4 encoding")

    video = np.memmap(raw, mode="r", dtype="<f4", shape=(3, frames, height, width))
    # Reject non-finite output before starting the encoder. This only reads the
    # completed terminal file; it adds no GPU synchronization or activation I/O.
    minimum = float("inf")
    maximum = float("-inf")
    for index in range(frames):
        frame = video[:, index]
        if not np.isfinite(frame).all():
            raise ValueError(f"Non-finite decoded video at frame {index}")
        minimum = min(minimum, float(frame.min()))
        maximum = max(maximum, float(frame.max()))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "pipe:0"]
    if audio is not None:
        command += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", "192k"]
    command += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-t", str(frames / fps), "-movflags", "+faststart", str(temporary)]
    started = time.monotonic()
    try:
        with tempfile.TemporaryFile(dir=output.parent) as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                assert process.stdin is not None
                for index in range(frames):
                    rgb = frame_to_rgb24(video[:, index], pixel_range)
                    process.stdin.write(rgb.tobytes())
                process.stdin.close()
                return_code = process.wait()
            except BaseException:
                try:
                    if process.stdin:
                        process.stdin.close()
                except BrokenPipeError:
                    pass
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait()
                raise
            if return_code:
                errors.seek(0)
                raise RuntimeError(f"ffmpeg failed ({return_code}): "
                                   + errors.read().decode("utf-8", errors="replace"))
        # No-clobber publication even if another writer creates output meanwhile.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"schema": "wan_s2v_terminal_export_v1", "status": "complete",
            "input": str(raw.resolve()), "input_layout": "B1_C3_T_H_W_F32_LE",
            "pixel_range": pixel_range,
            "input_bytes": expected_bytes, "decoded_minimum": minimum,
            "decoded_maximum": maximum, "frames": frames, "height": height,
            "width": width, "fps": fps, "duration_seconds": frames / fps,
            "output": str(output.resolve()), "output_bytes": output.stat().st_size,
            "output_sha256": digest.hexdigest(), "audio": str(audio.resolve()) if audio else None,
            "terminal_encode_seconds": time.monotonic() - started,
            "neural_inference": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--pixel-range", choices=("minus_one_to_one", "zero_to_one"),
                        default="minus_one_to_one")
    args = parser.parse_args()
    print(json.dumps(export_video(args.input, args.output, args.frames, args.height,
                                  args.width, args.fps, args.audio, args.pixel_range),
                     sort_keys=True))


if __name__ == "__main__":
    main()
