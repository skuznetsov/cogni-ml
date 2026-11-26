#!/usr/bin/env python3
"""Decode final Qwen-Image 2.1 DiT latents with the local Diffusers VAE only.

The Crystal producer writes a JSON manifest and a raw little-endian float32
payload. Values are token-major HWC: one contiguous 64-channel vector per
spatial latent position. This script does not load or run a text encoder or
transformer, and it never downloads model files.

Example:
    python3 scripts/qwen_image21_vae_decode.py \
        --manifest /tmp/qwen_image21_latents.json \
        --model-dir /models/Qwen-Image-2.1 \
        --output /tmp/image.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAT = "qwen-image21-latents-v1"
MODEL_ID = "Qwen/Qwen-Image-2.1"
CHANNELS = 64
SPATIAL_SCALE = 16
MAX_LATENT_SIDE = 256  # Bounds the bridge to at most 4096 pixels per output side.


@dataclass(frozen=True)
class LatentBundle:
    manifest_path: Path
    payload_path: Path
    height: int
    width: int
    payload: bytes


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_latent_bundle(manifest_path: str | Path) -> LatentBundle:
    """Read and validate the versioned Crystal-to-VAE exchange bundle."""
    manifest_path = Path(manifest_path)
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read latent manifest: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("latent manifest must be a JSON object")
    expected = {
        "format": FORMAT,
        "model_id": MODEL_ID,
        "layout": "tokens_hwc",
        "channels": CHANNELS,
        "dtype": "float32-le",
        "scaling": "diffusers_normalized",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key!r} must equal {value!r}")

    height = manifest.get("latent_height")
    width = manifest.get("latent_width")
    if not _is_int(height) or not _is_int(width) or height <= 0 or width <= 0:
        raise ValueError("manifest latent_height and latent_width must be positive integers")
    if height > MAX_LATENT_SIDE or width > MAX_LATENT_SIDE:
        raise ValueError(
            f"latent side exceeds {MAX_LATENT_SIDE}; refusing an unexpectedly large VAE decode"
        )
    # The official pipeline rounds image dimensions to multiples of 32. Its
    # 16x spatial VAE therefore produces even latent dimensions.
    if height % 2 or width % 2:
        raise ValueError("latent height and width must both be even for the Qwen-Image 2.1 pipeline")

    expected_image_height = height * SPATIAL_SCALE
    expected_image_width = width * SPATIAL_SCALE
    image_height = manifest.get("image_height")
    image_width = manifest.get("image_width")
    if not _is_int(image_height) or image_height != expected_image_height:
        raise ValueError(f"manifest image_height must equal {expected_image_height}")
    if not _is_int(image_width) or image_width != expected_image_width:
        raise ValueError(f"manifest image_width must equal {expected_image_width}")

    payload_name = manifest.get("payload")
    if not isinstance(payload_name, str) or not payload_name:
        raise ValueError("manifest payload must name a raw latent file")
    payload_component = Path(payload_name)
    if payload_component.name != payload_name or payload_name in {".", ".."}:
        raise ValueError("manifest payload must be a filename in the manifest directory")

    payload_path = manifest_path.parent / payload_component
    expected_size = height * width * CHANNELS * 4
    payload_bytes = manifest.get("payload_bytes")
    if not _is_int(payload_bytes) or payload_bytes != expected_size:
        raise ValueError(f"manifest payload_bytes must equal {expected_size}")
    try:
        with payload_path.open("rb") as stream:
            payload = stream.read(expected_size + 1)
    except OSError as exc:
        raise ValueError(f"cannot read latent payload: {exc}") from exc
    if len(payload) != expected_size:
        raise ValueError(
            f"latent payload has {len(payload)} bytes; expected exactly {expected_size} "
            f"for HWC shape ({height}, {width}, {CHANNELS})"
        )

    # Decode float32 values using explicit little-endian semantics before
    # handing them to PyTorch; malformed NaN/Inf values must never reach VAE.
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to decode the latent bundle") from exc
    values = np.frombuffer(payload, dtype="<f4")
    if not bool(np.isfinite(values).all()):
        raise ValueError("latent payload contains NaN or infinity")

    return LatentBundle(manifest_path, payload_path, height, width, payload)


def tokens_hwc_to_vae_input(payload: bytes, height: int, width: int):
    """Convert token-major HWC float32 bytes to contiguous [1,64,1,H,W]."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to decode the latent bundle") from exc
    expected_values = height * width * CHANNELS
    if len(payload) != expected_values * 4:
        raise ValueError("latent payload size does not match its HWC dimensions")
    values = np.frombuffer(payload, dtype="<f4")
    if not bool(np.isfinite(values).all()):
        raise ValueError("latent payload contains NaN or infinity")
    # Qwen-Image-2.1 uses plain spatial flattening: [B,C,H,W] <-> [B,H*W,C].
    # Add its single frame dimension for AutoencoderKLQwenImage21.decode.
    return np.ascontiguousarray(
        values.reshape(height, width, CHANNELS)
        .transpose(2, 0, 1)[None, :, None, :, :]
    )


def _load_local_vae(model_dir: Path, device: str, dtype_name: str):
    if not model_dir.is_dir():
        raise ValueError(f"local model directory does not exist: {model_dir}")
    if not (model_dir / "vae" / "config.json").is_file():
        raise ValueError(f"expected local Diffusers VAE config at {model_dir / 'vae' / 'config.json'}")

    try:
        import torch
        from diffusers import AutoencoderKLQwenImage21
    except ImportError as exc:
        raise RuntimeError(
            "VAE decoding requires PyTorch and a Diffusers release exposing "
            "AutoencoderKLQwenImage21"
        ) from exc

    requested_device = device
    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested_device = "mps"
        elif torch.cuda.is_available():
            requested_device = "cuda"
        else:
            requested_device = "cpu"
    if requested_device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    try:
        vae = AutoencoderKLQwenImage21.from_pretrained(
            str(model_dir),
            subfolder="vae",
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(f"could not load the local Qwen-Image 2.1 VAE: {exc}") from exc

    config = vae.config
    if getattr(config, "z_dim", None) != CHANNELS:
        raise ValueError(f"VAE z_dim must be {CHANNELS}")
    if getattr(config, "out_channels", None) != 4:
        raise ValueError("Qwen-Image 2.1 VAE must have four RGBA output channels")
    if getattr(config, "scale_factor_spatial", None) != SPATIAL_SCALE:
        raise ValueError(f"VAE spatial scale must be {SPATIAL_SCALE}")
    means = getattr(config, "latents_mean", None)
    stds = getattr(config, "latents_std", None)
    if not isinstance(means, (list, tuple)) or len(means) != CHANNELS:
        raise ValueError("VAE config must provide 64 latent means")
    if not isinstance(stds, (list, tuple)) or len(stds) != CHANNELS:
        raise ValueError("VAE config must provide 64 latent standard deviations")
    try:
        means = [float(value) for value in means]
        stds = [float(value) for value in stds]
    except (TypeError, ValueError) as exc:
        raise ValueError("VAE latent means and standard deviations must be numeric") from exc
    if not all(math.isfinite(value) for value in means):
        raise ValueError("VAE latent means must all be finite")
    if not all(math.isfinite(value) and value > 0 for value in stds):
        raise ValueError("VAE latent standard deviations must be finite and positive")

    vae = vae.to(device=requested_device, dtype=dtype)
    vae.eval()
    return torch, vae, requested_device, dtype


def decode_bundle(
    manifest_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    device: str = "auto",
    dtype_name: str = "float32",
) -> tuple[int, int]:
    """Decode validated DiT latents through the local VAE and write RGBA PNG."""
    bundle = load_latent_bundle(manifest_path)
    vae_root = Path(model_dir)
    torch, vae, device_name, dtype = _load_local_vae(vae_root, device, dtype_name)
    latent_array = tokens_hwc_to_vae_input(bundle.payload, bundle.height, bundle.width)
    latents = torch.from_numpy(latent_array).to(device=device_name, dtype=dtype)

    means = torch.tensor([float(value) for value in vae.config.latents_mean], device=device_name, dtype=dtype).view(
        1, CHANNELS, 1, 1, 1
    )
    stds = torch.tensor([float(value) for value in vae.config.latents_std], device=device_name, dtype=dtype).view(
        1, CHANNELS, 1, 1, 1
    )
    # This exactly reverses the normalization used by QwenImage21Pipeline.
    vae_latents = latents * stds + means

    with torch.inference_mode():
        decoded = vae.decode(vae_latents, return_dict=False)[0]
    expected_shape = (1, 4, 1, bundle.height * SPATIAL_SCALE, bundle.width * SPATIAL_SCALE)
    if tuple(decoded.shape) != expected_shape:
        raise RuntimeError(f"VAE returned shape {tuple(decoded.shape)}, expected {expected_shape}")
    decoded_frame = decoded[0, :, 0].float()
    if not bool(torch.isfinite(decoded_frame).all().item()):
        raise RuntimeError("VAE returned NaN or infinity")
    # Diffusers VaeImageProcessor postprocess convention: model range [-1,1]
    # to rounded uint8 pixels [0,255]. Keep the fourth alpha channel intact.
    pixels = (
        decoded_frame.permute(1, 2, 0)
        .div(2.0)
        .add(0.5)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    if pixels.shape[-1] != 4:
        raise RuntimeError("Qwen-Image 2.1 VAE output did not contain RGBA pixels")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save the decoded PNG") from exc

    output = Path(output_path)
    Image.fromarray(pixels).save(output, format="PNG")
    return pixels.shape[1], pixels.shape[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Crystal latent JSON sidecar")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="local Hugging Face Diffusers model directory containing vae/config.json",
    )
    parser.add_argument("--output", required=True, help="destination PNG path")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        width, height = decode_bundle(
            args.manifest, args.model_dir, args.output, args.device, args.dtype
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"qwen_image21_vae_decode: error: {exc}", file=sys.stderr)
        return 2
    print(f"saved {width}x{height} RGBA image to {args.output}")
    print("decoder: local Qwen-Image 2.1 VAE only; Crystal supplied the DiT latents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
