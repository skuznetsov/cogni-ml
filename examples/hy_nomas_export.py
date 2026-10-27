#!/usr/bin/env python3
"""Terminal MP4 export for a completed Hy-Nomas display-range F32_LE file.

This encodes an existing file only. It does not run Hy-Nomas or establish that
the native model produced the supplied B1C3THW tensor. Pixels must already have
undergone the upstream postprocessing to the zero-to-one display range.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wan_s2v_export import export_video as export_completed_video


FPS = 24
TEMPORAL_RATIO = 4


def expected_latent_frames(frames: int) -> int:
    if type(frames) is not int or frames <= 0:
        raise ValueError("frames must be a positive integer")
    # Pinned Hy-Nomas causal_ar inference formula.
    return (frames - 1) // TEMPORAL_RATIO + 1


def export_video(raw: Path, output: Path, frames: int, height: int, width: int,
                 fps: int, latent_frames: int, audio: Path | None = None) -> dict:
    if fps != FPS:
        raise ValueError(f"fps must match the pinned Hy-Nomas preset ({FPS})")
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("height and width must be positive integers")
    if height % 16 or width % 16:
        raise ValueError("height and width must already be multiples of 16")
    expected_latents = expected_latent_frames(frames)
    if type(latent_frames) is not int or latent_frames != expected_latents:
        raise ValueError(
            f"latent_frames mismatch: pinned formula for {frames} pixel frames "
            f"requires {expected_latents}, got {latent_frames}"
        )
    manifest = export_completed_video(
        raw, output, frames, height, width, fps, audio,
        pixel_range="zero_to_one",
    )
    manifest["schema"] = "hy_nomas_terminal_export_v1"
    manifest["latent_frames"] = latent_frames
    manifest["latent_frames_formula"] = "(pixel_frames - 1) // 4 + 1"
    manifest["pixel_range"] = "zero_to_one"
    manifest["neural_inference"] = False
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="completed display-range B1C3THW F32_LE terminal file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True,
                        help="pixel frame count from the compiled project")
    parser.add_argument("--latent-frames", type=int, required=True,
                        help="must match the pinned causal_ar formula")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    result = export_video(args.input, args.output, args.frames, args.height,
                          args.width, args.fps, args.latent_frames, args.audio)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
