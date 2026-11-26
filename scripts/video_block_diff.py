#!/usr/bin/env python3
"""Offline local-change diagnostic: one sequential RGB decode, bounded frame storage.

Compare area-averaged pixels at fixed positions, then average absolute RGB
differences within spatial blocks. Values are in [0, 1]. This detects image
change, not correct acting: flicker/camera motion count, and pooling can hide
fine motion. Use scale=1 as a sensitivity control. Nothing enters inference.
Requires NumPy, ffmpeg and ffprobe. Times assume constant frame rate.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from fractions import Fraction

import numpy as np


def read_rgb_frames(stream, width: int, height: int):
    size = width * height * 3
    while True:
        chunks = bytearray()
        while len(chunks) < size:
            part = stream.read(size - len(chunks))
            if not part:
                break
            chunks.extend(part)
        if not chunks:
            return
        if len(chunks) != size:
            raise ValueError("truncated RGB frame")
        yield np.frombuffer(chunks, dtype=np.uint8).reshape(height, width, 3)


def analyze_frames(frames, *, fps: float, scale: int = 4, block: int = 16,
                   lags=(1, 6), rois=None):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if scale < 1 or block < 1 or block % scale:
        raise ValueError("block must be a positive multiple of scale")
    lags = sorted(set(lags))
    if not lags or any(not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("lags must be positive frame counts")
    history = deque(maxlen=max(lags) + 1)
    reports = {str(lag): {"pairs": 0, "series": []} for lag in lags}
    sums, regions, shape = {}, {}, None
    count = peak = 0
    for index, frame in enumerate(frames):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
            raise ValueError("frames must be nonempty HxWx3 uint8 arrays")
        if shape is None:
            shape = frame.shape
            height, width, _ = shape
            ys, xs = np.arange(0, height, scale), np.arange(0, width, scale)
            areas = np.minimum(scale, height - ys)[:, None] * np.minimum(scale, width - xs)
            by = np.arange(0, len(ys), block // scale)
            bx = np.arange(0, len(xs), block // scale)

            def block_sum(values):
                return np.add.reduceat(np.add.reduceat(values, by, axis=0), bx, axis=1)

            block_areas = block_sum(areas)
            rois = dict(rois) if rois else {"full": (0, 0, width, height)}
            for name, (x, y, w, h) in rois.items():
                if (not name or min(x, y) < 0 or min(w, h) <= 0
                        or x + w > width or y + h > height
                        or x % block or y % block
                        or ((x + w) % block and x + w != width)
                        or ((y + h) % block and y + h != height)):
                    raise ValueError("ROIs must be nonempty, in bounds and aligned to blocks")
                regions[name] = (slice(y // block, (y + h + block - 1) // block),
                                 slice(x // block, (x + w + block - 1) // block))
            sums = {str(lag): np.zeros(block_areas.shape, dtype=np.float64) for lag in lags}
        elif frame.shape != shape:
            raise ValueError("frame geometry changed")
        pooled_sum = np.add.reduceat(np.add.reduceat(frame, ys, axis=0, dtype=np.uint32),
                                     xs, axis=1, dtype=np.uint32)
        pooled = pooled_sum.astype(np.float32) / areas[:, :, None] / 255.0
        history.append(pooled)
        peak = max(peak, len(history))
        for lag in lags:
            if index < lag:
                continue
            delta = np.abs(pooled - history[-lag - 1]).mean(axis=2)
            block_delta = block_sum(delta * areas) / block_areas
            report = reports[str(lag)]
            report["pairs"] += 1
            sums[str(lag)] += block_delta
            region_delta = {
                name: float(np.sum(block_delta[region] * block_areas[region])
                            / np.sum(block_areas[region]))
                for name, region in regions.items()
            }
            report["series"].append({
                "frame": index, "time_s": index / fps, "since_s": (index - lag) / fps,
                "mean_delta": float(np.sum(block_delta * block_areas) / (height * width)),
                "max_block_delta": float(block_delta.max()), "roi_mean_delta": region_delta,
            })
        count += 1
    if count <= max(lags):
        raise ValueError("video must contain more frames than every requested lag")
    for lag, report in reports.items():
        report["mean_block_delta"] = (sums[lag] / report["pairs"]).tolist()
    return {"frames": count, "width": width, "height": height, "fps": fps,
            "scale": scale, "block": block, "rois": rois, "lags": reports,
            "retained_coarse_frames_peak": peak}


def analyze_video(path: Path, **options):
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate", "-of", "json", str(path),
    ]))["streams"][0]
    fps = float(Fraction(metadata["avg_frame_rate"]))
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-i", str(path),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-pix_fmt", "rgb24",
            "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1",
        ], stdout=subprocess.PIPE, stderr=errors)
        try:
            result = analyze_frames(read_rgb_frames(process.stdout, metadata["width"], metadata["height"]),
                                    fps=fps, **options)
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError("ffmpeg failed: " + errors.read(4096).decode(errors="replace"))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--roi", action="append", default=[], metavar="NAME:X,Y,W,H")
    args = parser.parse_args()
    try:
        rois = {}
        for value in args.roi:
            name, coordinates = value.split(":", 1)
            rectangle = tuple(int(part) for part in coordinates.split(","))
            if len(rectangle) != 4 or not name or name in rois:
                raise ValueError("ROI names must be unique; rectangles need four coordinates")
            rois[name] = rectangle
        if args.output.resolve() in [path.resolve() for path in args.videos]:
            raise ValueError("output must not overwrite an input video")
        report = {
            "schema": "cogni_ml.video_block_diff/v1",
            "metric": "area-weighted mean absolute RGB difference after spatial area pooling; [0,1]",
            "scope": "Offline image-change diagnostic; no semantic motion, visual quality or tensor-parity verdict",
            "time_basis": "frame index / average frame rate; constant-frame-rate assumption",
            "videos": [analyze_video(path, scale=args.scale, block=args.block, lags=args.lags, rois=rois)
                       for path in args.videos],
        }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError, IndexError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
