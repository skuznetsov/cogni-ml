"""Prepare authenticated RGB8 input; normalization and DCAE encoding stay on GPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import warnings

from PIL import Image, ImageOps


SCHEMA = "cogni_ml.hy_nomas_reference_image/v1"
UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
CHECKPOINT_SHA256 = "ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"
WIDTH, HEIGHT, FRAMES = 512, 288, 120
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 32 * 1024 * 1024
FIT_POLICY = "exif-rgb-lanczos-pad-black-center"


def prepare_reference_image(image_path: Path, output_dir: Path,
                            encoder_safetensors_sha256: str) -> dict:
    """Decode/fit once on ingestion; emit RGB8, never CPU-produced latents."""
    if not re.fullmatch(r"[0-9a-f]{64}", encoder_safetensors_sha256):
        raise ValueError("a trusted encoder safetensors SHA-256 is required")
    image_path = image_path.resolve(strict=True)
    if not image_path.is_file() or not 0 < image_path.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError("reference image must be a regular file of at most 64 MiB")
    # A single bounded byte snapshot binds decoding and the source digest even
    # when the original path changes during preparation.
    with image_path.open("rb") as source:
        raw = source.read(MAX_IMAGE_BYTES + 1)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("reference image exceeds the byte bound")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("reference image exceeds the decoded pixel bound")
            # Match the immutable upstream entry point exactly. RGB conversion
            # precedes padding; transparent pixels are not alpha-composited.
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            fitted = ImageOps.pad(rgb, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS,
                                  color=(0, 0, 0), centering=(0.5, 0.5))
            pixels = fitted.tobytes()
    if len(pixels) != WIDTH * HEIGHT * 3:
        raise ValueError("prepared RGB8 extent is invalid")
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise ValueError("reference preparation requires a fresh output directory")
    output_dir.mkdir(parents=True)
    output_dir = output_dir.resolve(strict=True)
    payload_path = output_dir / "pixels.rgb8"
    with payload_path.open("xb") as target:
        target.write(pixels)
    manifest = {
        "schema": SCHEMA, "schema_version": 1,
        "upstream_revision": UPSTREAM_REVISION,
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "width": WIDTH, "height": HEIGHT, "frames": FRAMES,
        "fit_policy": FIT_POLICY, "payload": payload_path.name,
        "payload_bytes": len(pixels),
        "payload_sha256": hashlib.sha256(pixels).hexdigest(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "encoder_safetensors_sha256": encoder_safetensors_sha256,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("xb") as target:
        target.write(manifest_bytes)
    return {"manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "payload_bytes": len(pixels), "image_sha256": manifest["image_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_reference_image(args.image, args.output_dir, args.encoder_sha),
                     sort_keys=True))


if __name__ == "__main__":
    main()
