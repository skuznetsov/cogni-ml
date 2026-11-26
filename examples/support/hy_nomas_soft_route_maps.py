"""Author and freeze small video-query logit priors; no model/GPU dependency.

These static priors are hand-authored from a reference image. They are not
segmentation, optical flow, pose tracks, or output-dependent feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

CASES = ("zero", "A1_low", "A1_high", "A2_low", "A2_high",
         "A1_sham", "A2_sham", "zero_repeat")
VIDEO, QUERIES, HEIGHT, WIDTH = 4320, 4544, 9, 16


def bf16(value: float) -> float:
    """F32 operation boundary followed by BF16 round-to-nearest-even."""
    bits, = struct.unpack("<I", struct.pack("<f", value))
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def prior(action: int, sham: bool = False) -> list[float]:
    if action not in (1, 2):
        raise ValueError("only the two authored actions are admitted")
    center_h, center_w = ((1.5, 5.5) if action == 1 else (4.5, 11.5))
    start, stop = ((1, 10) if action == 1 else (12, 28))
    radius_h = 2.0 if action == 1 else 1.5
    normalizer = 0.5625 if action == 1 else 0.5
    shift = (3 if action == 1 else -2) if sham else 0
    result = [0.0] * QUERIES
    for t in range(start, stop):
        for h in range(HEIGHT):
            for w in range(WIDTH):
                value = (max(0.0, 1 - abs(h - shift - center_h) / radius_h)
                         * max(0.0, 1 - abs(w - center_w) / 2) / normalizer)
                result[t * HEIGHT * WIDTH + h * WIDTH + w] = value
    return result


def biases() -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for case in CASES:
        if case in ("zero", "zero_repeat"):
            result[case] = [0.0] * QUERIES
            continue
        action = int(case[1])
        dose = math.log(1.5 if case.endswith("low") else 2.0)
        # Cast the product to F32, then BF16 once, as the source mask does.
        result[case] = [bf16(dose * v) for v in prior(action, case.endswith("sham"))]
    return result


def build_document(image: Path) -> dict:
    image = image.resolve(strict=True)
    values = biases()
    return {
        "schema": "hy-nomas-soft-route-bias-v1",
        "geometry": {"video_rows": VIDEO, "query_rows": QUERIES},
        "case_order": list(CASES),
        "reference_image": {"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()},
        "maps": {
            "kind": "authored static separable tent, not measured tracks",
            "grid": [30, HEIGHT, WIDTH],
            "cell_centers_xy": "(32*(w+0.5),32*(h+0.5)) in pinned 512x288 image",
            "A1": {"center_hw": [1.5, 5.5], "frames": [1, 10], "sham_shift_hw": [3, 0],
                   "radius_cells": [2, 2], "peak_normalizer": 0.5625},
            "A2": {"center_hw": [4.5, 11.5], "frames": [12, 28], "sham_shift_hw": [-2, 0],
                   "radius_cells": [1.5, 2], "peak_normalizer": 0.5},
            "doses": {"low": math.log(1.5), "high": math.log(2)},
            "rounding": "F32 product then BF16 nearest-even, lossless F32 expansion",
            "support": "accepted M11; no edges added; video queries only",
        },
        "biases": values,
        "bias_f32le_sha256": {
            case: hashlib.sha256(struct.pack(f"<{QUERIES}f", *v)).hexdigest()
            for case, v in values.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = build_document(args.reference_image)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(args.output.resolve()), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "case_order": list(CASES), "metal_resources_created": False}))


if __name__ == "__main__":
    main()
