#!/usr/bin/env python3
"""Export tiny, weight-free TRELLIS.2 image-preprocess oracle cases.

This invokes NumPy and Pillow with the exact operation order used by the
pinned TRELLIS.2 pipeline. It does not import transformers, load a model, use
an accelerator, or access the network.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SCHEMA = "cogni-ml/trellis2/dino-v3-source-preprocess-oracle/v1"


def preprocess_rgba(image: Image.Image) -> tuple[Image.Image, tuple[float, ...]]:
    if image.mode != "RGBA":
        raise ValueError("oracle cases must provide explicit RGBA")
    alpha = np.array(image)[:, :, 3]
    if np.all(alpha == 255):
        raise ValueError("opaque inputs require the separate rembg boundary")
    if max(image.size) > 1024:
        raise ValueError("oversize LANCZOS downsampling is a later oracle slice")

    output_np = np.array(image)
    alpha = output_np[:, :, 3]
    foreground = np.argwhere(alpha > 0.8 * 255)
    if foreground.size == 0:
        raise ValueError("empty foreground")
    bounds = (
        np.min(foreground[:, 1]),
        np.min(foreground[:, 0]),
        np.max(foreground[:, 1]),
        np.max(foreground[:, 0]),
    )
    center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    size = int(max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 1)
    crop = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    output = image.crop(crop)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output, crop


def set_pixel(array: np.ndarray, x: int, y: int, rgba: tuple[int, ...]) -> None:
    array[y, x, :] = np.asarray(rgba, dtype=np.uint8)


def boundary_case() -> np.ndarray:
    array = np.zeros((6, 7, 4), dtype=np.uint8)
    array[:, :, :3] = np.asarray((200, 100, 50), dtype=np.uint8)
    for pixel in (
        (1, 2, (10, 20, 30, 255)),
        (5, 2, (1, 2, 3, 255)),
        (2, 3, (200, 100, 50, 128)),
        (3, 3, (90, 180, 45, 204)),
        (4, 3, (100, 50, 200, 205)),
        (1, 4, (255, 128, 64, 255)),
        (5, 4, (7, 8, 9, 255)),
    ):
        set_pixel(array, *pixel)
    return array


def padded_negative_case() -> np.ndarray:
    array = np.zeros((5, 4, 4), dtype=np.uint8)
    array[:, :, :3] = np.asarray((11, 22, 33), dtype=np.uint8)
    for y in range(5):
        for x in range(2):
            set_pixel(array, x, y, (20 + x, 40 + y, 60 + x + y, 255))
    return array


def export_case(name: str, rgba: np.ndarray) -> dict[str, object]:
    output, requested_crop = preprocess_rgba(Image.fromarray(rgba, mode="RGBA"))
    rounded_crop = tuple(int(round(value)) for value in requested_crop)
    output_array = np.asarray(output, dtype=np.uint8)
    return {
        "name": name,
        "input": {
            "width": int(rgba.shape[1]),
            "height": int(rgba.shape[0]),
            "rgba": rgba.reshape(-1).tolist(),
        },
        "expected": {
            "requested_crop": list(requested_crop),
            "rounded_crop": list(rounded_crop),
            "width": int(output_array.shape[1]),
            "height": int(output_array.shape[0]),
            "rgb": output_array.reshape(-1).tolist(),
        },
    }


def export_premultiply_sweep() -> dict[str, object]:
    size = 256
    x = np.arange(size, dtype=np.uint16)[None, :]
    y = np.arange(size, dtype=np.uint16)[:, None]
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, 0] = x
    rgba[:, :, 1] = 255 - x
    rgba[:, :, 2] = (x + y) % 256
    rgba[:, :, 3] = y
    output, requested_crop = preprocess_rgba(Image.fromarray(rgba, mode="RGBA"))
    output_array = np.asarray(output, dtype=np.uint8)
    return {
        "recipe": "rgba(x,y) = [x, 255-x, (x+y)%256, y] on 256x256",
        "requested_crop": list(requested_crop),
        "rounded_crop": [int(round(value)) for value in requested_crop],
        "width": int(output_array.shape[1]),
        "height": int(output_array.shape[0]),
        "rgb_sha256": hashlib.sha256(output_array.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    fixture = {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": PIN,
            "pipeline": f"https://github.com/microsoft/TRELLIS.2/blob/{PIN}/trellis2/pipelines/trellis2_image_to_3d.py",
            "pillow_crop": f"https://github.com/python-pillow/Pillow/blob/{Image.__version__}/src/PIL/Image.py",
            "generator": "tools/trellis2_oracle/export_dino_v3_preprocess.py",
            "oracle_kind": "pinned-source NumPy/Pillow execution",
            "numpy_version": np.__version__,
            "pillow_version": Image.__version__,
            "weights": "none",
            "device": "cpu",
        },
        "scope": {
            "max_source_edge": 1024,
            "alpha_foreground_rule": "alpha > 204",
            "includes": ["alpha bbox", "Pillow crop", "black premultiply"],
            "excludes": ["rembg", "LANCZOS resize", "ImageNet normalize", "DINOv3 encoder"],
        },
        "cases": [
            export_case("threshold_and_exclusive_bounds", boundary_case()),
            export_case("negative_padded_crop", padded_negative_case()),
        ],
        "premultiply_sweep": export_premultiply_sweep(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output} (numpy={np.__version__}, "
        f"pillow={Image.__version__}, weights=none, device=cpu)"
    )


if __name__ == "__main__":
    main()
