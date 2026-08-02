#!/usr/bin/env python3
"""Export weight-free CPU oracles for the TRELLIS.2 DINOv3 image tensor.

The script reproduces the pinned TRELLIS.2 feature-extractor preprocessing only:
Pillow LANCZOS resize, RGB uint8 to float32 `/255`, CHW/batch layout, and
ImageNet normalization. It never loads DINOv3 configuration or weights, never
selects an accelerator, and never accesses the network.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SCHEMA = "cogni-ml/trellis2/dino-v3-image-conditioner-oracle/v1"
PILLOW_PIN = "12.2.0"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def recipe_rgb(width: int, height: int) -> np.ndarray:
    x = np.arange(width, dtype=np.uint64)[None, :]
    y = np.arange(height, dtype=np.uint64)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (17 * x + 13 * y + 3) % 256
    image[:, :, 1] = (5 * x + 29 * y + 7) % 256
    image[:, :, 2] = (x * x + 3 * y + 11) % 256
    return image


def f32le_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def export_case(name: str, width: int, height: int, target: int) -> dict[str, object]:
    source = recipe_rgb(width, height)
    resized_image = Image.fromarray(source, mode="RGB").resize(
        (target, target), Image.Resampling.LANCZOS
    )
    resized = np.asarray(resized_image, dtype=np.uint8)
    tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255
    mean = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(STD, dtype=torch.float32)[:, None, None]
    normalized = ((tensor - mean) / std).unsqueeze(0).contiguous()
    probe_coordinates = (
        (0, 0, 0, 0),
        (0, 1, target // 3, target // 2),
        (0, 2, target - 1, target - 1),
        (0, 0, target // 2, target // 3),
    )
    probes = [
        {
            "index": list(index),
            "value": float(normalized[index].item()),
        }
        for index in probe_coordinates
    ]
    return {
        "name": name,
        "input": {
            "width": width,
            "height": height,
            "recipe": "rgb(x,y)=[(17*x+13*y+3)%256,(5*x+29*y+7)%256,(x*x+3*y+11)%256]",
            "rgb_sha256": hashlib.sha256(source.tobytes(order="C")).hexdigest(),
        },
        "target": target,
        "expected": {
            "resized_rgb_sha256": hashlib.sha256(
                resized.tobytes(order="C")
            ).hexdigest(),
            "normalized_shape": list(normalized.shape),
            "normalized_f32le_sha256": f32le_sha256(normalized),
            "probes": probes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if Image.__version__ != PILLOW_PIN:
        raise RuntimeError(
            f"oracle requires Pillow {PILLOW_PIN}, found {Image.__version__}"
        )
    torch.set_num_threads(1)

    fixture = {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "extractor": f"https://github.com/microsoft/TRELLIS.2/blob/{TRELLIS_PIN}/trellis2/modules/image_feature_extractor.py",
            "pillow_resample": f"https://github.com/python-pillow/Pillow/blob/{PILLOW_PIN}/src/libImaging/Resample.c",
            "pillow_license": "MIT-CMU",
            "pillow_license_url": f"https://github.com/python-pillow/Pillow/blob/{PILLOW_PIN}/LICENSE",
            "generator": "tools/trellis2_oracle/export_dino_v3.py",
            "oracle_kind": "pinned-source Pillow and PyTorch CPU execution",
            "numpy_version": np.__version__,
            "pillow_version": PILLOW_PIN,
            "torch_version": torch.__version__,
            "weights": "none",
            "device": "cpu",
        },
        "normalization": {"mean": list(MEAN), "std": list(STD)},
        "scope": {
            "targets": [512, 1024],
            "includes": ["RGB LANCZOS resize", "float32 /255", "CHW", "batch", "ImageNet normalize"],
            "excludes": ["alpha/rembg crop", "DINOv3 embeddings", "DINOv3 blocks", "weights", "Metal"],
        },
        "cases": [
            export_case("single_pixel_to_512", 1, 1, 512),
            export_case("upscale_7_to_1024", 7, 7, 1024),
            export_case("downscale_769_to_512", 769, 769, 512),
            export_case("max_downscale_1024_to_512", 1024, 1024, 512),
            export_case("identity_512", 512, 512, 512),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output} (torch={torch.__version__}, "
        f"pillow={Image.__version__}, weights=none, device=cpu)"
    )


if __name__ == "__main__":
    main()
