#!/usr/bin/env python3
"""Prepare Qwen-Image 2.1 text conditioning and seeded initial noise for Crystal.

This uses the official Diffusers QwenImage21Pipeline prompt encoder and latent
preparation helpers only. It does not call the pipeline, transformer, scheduler,
or VAE, so all denoising remains in the native Crystal/Metal engine.

The output directory contains `qwen_image21_conditioning.json` and
`qwen_image21_conditioning.bin`. Tensor offsets and dtypes are described by the
versioned JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "qwen-image21-conditioning"
SCHEMA_VERSION = 1
MODEL_REPO = "Qwen/Qwen-Image-2.1"
PIPELINE_CLASS = "QwenImage21Pipeline"
TEXT_ENCODER_CLASS = "Qwen3VLForConditionalGeneration"
CONTEXT_DIM = 4096
LATENT_CHANNELS = 64
VAE_SCALE_FACTOR = 16
MAX_IMAGE_SIDE = 4096
MANIFEST_NAME = "qwen_image21_conditioning.json"
PAYLOAD_NAME = "qwen_image21_conditioning.bin"
REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_image_dimensions(width: int, height: int) -> tuple[int, int]:
    if not _is_positive_int(width) or not _is_positive_int(height):
        raise ValueError("image width and height must be positive integers")
    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
        raise ValueError(f"image sides must not exceed {MAX_IMAGE_SIDE} pixels")
    if width % (VAE_SCALE_FACTOR * 2) or height % (VAE_SCALE_FACTOR * 2):
        raise ValueError("image width and height must be multiples of 32")
    return height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR


def _as_numpy(value: Any):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to write the conditioning bundle") from exc

    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu")
        if hasattr(value, "float"):
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _float32_bytes(name: str, value: Any, expected_shape: tuple[int, ...]) -> bytes:
    import numpy as np

    array = _as_numpy(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain numeric values")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype="<f4").tobytes(order="C")


def _mask_bytes(name: str, value: Any, expected_shape: tuple[int, ...]) -> bytes:
    import numpy as np

    array = _as_numpy(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {array.shape}")
    if not bool(np.logical_or(array == 0, array == 1).all()):
        raise ValueError(f"{name} values must be 0 or 1")
    return np.ascontiguousarray(array, dtype=np.uint8).tobytes(order="C")


def _check_revision(revision: str) -> str:
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise ValueError("model revision must be a full 40-character Hugging Face commit SHA")
    return revision.lower()


def _tensor_descriptor(dtype: str, shape: list[int], offset: int, nbytes: int) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "offset_bytes": offset, "nbytes": nbytes}


def write_conditioning_bundle(
    *,
    output_dir: str | Path,
    prompt: str,
    revision: str,
    width: int,
    height: int,
    seed: int,
    encoder_hidden_states: Any,
    encoder_hidden_states_mask: Any,
    encoder_img_mask: Any,
    initial_target_latents: Any,
    source_dtype: str,
    device: str,
    torch_version: str,
    diffusers_version: str,
    diffusers_commit: str | None = None,
) -> Path:
    """Validate and serialize the exact batch-one conditioning exchange."""
    import numpy as np

    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not prompt:
        raise ValueError("prompt must not be empty")
    revision = _check_revision(revision)
    latent_height, latent_width = _validate_image_dimensions(width, height)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1):
        raise ValueError("seed must be an integer in [0, 2^63-1]")
    if source_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("source_dtype must be bfloat16, float16, or float32")

    hidden = _as_numpy(encoder_hidden_states)
    if hidden.ndim != 2 or hidden.shape[0] <= 0 or hidden.shape[1] != CONTEXT_DIM:
        raise ValueError(f"encoder_hidden_states must have shape [seq_len, {CONTEXT_DIM}]")
    seq_len = int(hidden.shape[0])
    target_tokens = latent_height * latent_width
    hidden_bytes = _float32_bytes(
        "encoder_hidden_states", hidden, (seq_len, CONTEXT_DIM)
    )
    attention_bytes = _mask_bytes(
        "encoder_hidden_states_mask", encoder_hidden_states_mask, (seq_len,)
    )
    image_mask_bytes = _mask_bytes("encoder_img_mask", encoder_img_mask, (seq_len,))
    latent_bytes = _float32_bytes(
        "initial_target_latents", initial_target_latents, (target_tokens, LATENT_CHANNELS)
    )

    tensor_bytes = [
        ("encoder_hidden_states", "float32-le", [seq_len, CONTEXT_DIM], hidden_bytes),
        ("encoder_hidden_states_mask", "uint8", [seq_len], attention_bytes),
        ("encoder_img_mask", "uint8", [seq_len], image_mask_bytes),
        ("initial_target_latents", "float32-le", [target_tokens, LATENT_CHANNELS], latent_bytes),
    ]
    payload = bytearray()
    tensors: dict[str, dict[str, Any]] = {}
    for name, dtype, shape, data in tensor_bytes:
        tensors[name] = _tensor_descriptor(dtype, shape, len(payload), len(data))
        payload.extend(data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    payload_path = output_dir / PAYLOAD_NAME
    if manifest_path.exists() or payload_path.exists():
        raise FileExistsError(
            f"conditioning bundle already exists in {output_dir}; choose an empty directory"
        )

    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "model": {
            "repo": MODEL_REPO,
            "revision": revision,
            "pipeline_class": PIPELINE_CLASS,
            "text_encoder_class": TEXT_ENCODER_CLASS,
        },
        "prompt": prompt,
        "image": {
            "width": width,
            "height": height,
            "vae_scale_factor": VAE_SCALE_FACTOR,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "img_shapes": [[1, latent_height, latent_width]],
        },
        "noise": {
            "seed": seed,
            "generator_device": "cpu",
            "source_dtype": source_dtype,
            "layout": "tokens_hwc",
        },
        "runtime": {
            "device": device,
            "torch_version": torch_version,
            "diffusers_version": diffusers_version,
            "diffusers_commit": diffusers_commit,
        },
        "payload_file": PAYLOAD_NAME,
        "payload_nbytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "tensors": tensors,
    }

    # Write data first and the manifest last. A manifest therefore never points
    # to a payload that this invocation has not completely written.
    payload_path.write_bytes(payload)
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception:
        payload_path.unlink(missing_ok=True)
        raise
    return manifest_path


def _read_cached_revision(model_dir: Path) -> str | None:
    """Read the source commit recorded by Hugging Face's local snapshot cache."""
    rel_paths = (
        "model_index.json",
        "processor/tokenizer.json",
        "text_encoder/config.json",
    )
    revisions: set[str] = set()
    for rel_path in rel_paths:
        metadata = model_dir / ".cache" / "huggingface" / "download" / f"{rel_path}.metadata"
        if not metadata.is_file():
            continue
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if lines and REVISION_RE.fullmatch(lines[0]):
            revisions.add(lines[0].lower())
    if len(revisions) > 1:
        raise ValueError(f"local model files come from different revisions: {sorted(revisions)}")
    return next(iter(revisions), None)


def _validate_model_directory(model_dir: Path) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise ValueError(f"local model directory does not exist: {model_dir}")
    index_path = model_dir / "model_index.json"
    config_path = model_dir / "text_encoder" / "config.json"
    processor_path = model_dir / "processor"
    if not index_path.is_file() or not config_path.is_file() or not processor_path.is_dir():
        raise ValueError("expected local model_index.json, text_encoder/config.json, and processor/")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read local Qwen-Image 2.1 config: {exc}") from exc

    if index.get("_class_name") != PIPELINE_CLASS:
        raise ValueError(f"model_index.json must select {PIPELINE_CLASS}")
    if index.get("text_encoder") != ["transformers", TEXT_ENCODER_CLASS]:
        raise ValueError(f"model_index.json must select {TEXT_ENCODER_CLASS}")
    text_config = config.get("text_config", {})
    if config.get("model_type") != "qwen3_vl" or config.get("architectures") != [TEXT_ENCODER_CLASS]:
        raise ValueError("text_encoder/config.json must describe Qwen3-VL conditional generation")
    if text_config.get("hidden_size") != CONTEXT_DIM:
        raise ValueError(f"Qwen3-VL text hidden_size must be {CONTEXT_DIM}")
    return index


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _installed_diffusers_commit() -> str | None:
    try:
        direct_url = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
        if not direct_url:
            return None
        vcs_info = json.loads(direct_url).get("vcs_info", {})
        commit = vcs_info.get("commit_id")
        if isinstance(commit, str) and REVISION_RE.fullmatch(commit):
            return commit.lower()
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        return None
    return None


def _load_local_pipeline(model_dir: Path, device: str, dtype_name: str):
    try:
        import torch
        import diffusers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "conditioning requires PyTorch, Transformers with Qwen3-VL, and Diffusers "
            "with QwenImage21Pipeline"
        ) from exc

    requested_device = _resolve_device(torch, device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_dir / "processor"), local_files_only=True
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir / "text_encoder"),
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not load local Qwen3-VL processor/text encoder (no network fallback): {exc}"
        ) from exc

    text_encoder = text_encoder.to(device=requested_device)
    text_encoder.eval()
    # None components are deliberate: this helper only invokes prompt encoding
    # and prepare_latents, never the Diffusers denoising pipeline.
    pipeline = QwenImage21Pipeline(
        scheduler=None,
        vae=None,
        text_encoder=text_encoder,
        processor=processor,
        transformer=None,
    )
    return torch, diffusers, pipeline, requested_device


def prepare_conditioning(
    *,
    model_dir: str | Path,
    output_dir: str | Path,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    revision: str | None = None,
    device: str = "auto",
    dtype_name: str = "bfloat16",
) -> Path:
    """Run official prompt/noise preparation locally and write a bundle."""
    model_root = Path(model_dir)
    _validate_model_directory(model_root)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    cached_revision = _read_cached_revision(model_root)
    if revision is None:
        if cached_revision is None:
            raise ValueError(
                "could not determine local model revision; pass the full --revision commit SHA"
            )
        revision = cached_revision
    revision = _check_revision(revision)
    if cached_revision is not None and cached_revision != revision:
        raise ValueError(
            f"requested revision {revision} differs from local model cache revision {cached_revision}"
        )

    latent_height, latent_width = _validate_image_dimensions(width, height)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1):
        raise ValueError("seed must be an integer in [0, 2^63-1]")
    if dtype_name not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")

    torch, diffusers, pipeline, device_name = _load_local_pipeline(
        model_root, device, dtype_name
    )
    try:
        with torch.inference_mode():
            prompt_embeds, attention_mask, image_mask = pipeline._get_qwen_prompt_embeds(
                prompt, image=None, device=torch.device(device_name)
            )
            if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
                raise ValueError("official prompt encoder must return one batch of token embeddings")
            # Preserve the pipeline's CPU-generator semantics so the initial
            # noise is reproducible independently of the encoder device.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            target_latents, condition_latents = pipeline.prepare_latents(
                None,
                1,
                LATENT_CHANNELS,
                height,
                width,
                prompt_embeds.dtype,
                torch.device(device_name),
                generator,
                latents=None,
            )
    except Exception as exc:
        raise RuntimeError(f"official QwenImage21 conditioning preparation failed: {exc}") from exc

    if condition_latents is not None:
        raise RuntimeError("text-to-image preparation unexpectedly produced condition-image latents")
    if tuple(target_latents.shape) != (1, latent_height * latent_width, LATENT_CHANNELS):
        raise RuntimeError(
            "official QwenImage21Pipeline.prepare_latents returned an unexpected latent shape "
            f"{tuple(target_latents.shape)}"
        )
    if tuple(prompt_embeds.shape[1:]) != (attention_mask.shape[1], CONTEXT_DIM):
        raise RuntimeError("official Qwen3-VL prompt embeddings/mask dimensions do not match the bridge")
    if tuple(image_mask.shape) != tuple(attention_mask.shape):
        raise RuntimeError("official Qwen3-VL image-pad mask dimensions do not match attention mask")

    return write_conditioning_bundle(
        output_dir=output_dir,
        prompt=prompt,
        revision=revision,
        width=width,
        height=height,
        seed=seed,
        encoder_hidden_states=prompt_embeds[0],
        encoder_hidden_states_mask=attention_mask[0],
        encoder_img_mask=image_mask[0],
        initial_target_latents=target_latents[0],
        source_dtype=dtype_name,
        device=device_name,
        torch_version=torch.__version__,
        diffusers_version=diffusers.__version__,
        diffusers_commit=_installed_diffusers_commit(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="local Qwen-Image-2.1 snapshot directory")
    parser.add_argument("--output-dir", required=True, help="new or empty conditioning output directory")
    parser.add_argument("--prompt", required=True, help="text prompt to encode")
    parser.add_argument("--width", type=int, default=1024, help="output image width (multiple of 32)")
    parser.add_argument("--height", type=int, default=1024, help="output image height (multiple of 32)")
    parser.add_argument("--seed", type=int, default=0, help="CPU-generator seed for initial noise")
    parser.add_argument("--revision", help="full Hugging Face source commit SHA; inferred from cache if available")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_conditioning(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            seed=args.seed,
            revision=args.revision,
            device=args.device,
            dtype_name=args.dtype,
        )
    except Exception as exc:
        print(f"qwen_image21_prepare_conditioning: error: {exc}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
