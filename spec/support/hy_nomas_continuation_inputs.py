#!/usr/bin/env python3
"""Prepare one authenticated six-segment Hy-Nomas continuation input bundle.

The producer builds one fixed-duration initial-noise bank on MPS, exports its
first 180 latents, creates six globally offset DiT RoPE tables, and creates one
global PixelUMM decoder-noise bank. It does not load checkpoints or run models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

SCHEMA = "cogni_ml.hy_nomas_continuation_inputs/v1"
SCHEMA_VERSION = 1
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
DEFAULT_QWEN_INPUTS = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-quality-20261030/inputs-detailed.json"
)
OUTPUT_ROOT = Path("/opt/mldata_ssd")

SEGMENTS = 6
LATENTS_PER_SEGMENT = 30
PIXELS_PER_LATENT = 4
PIXELS_PER_SEGMENT = LATENTS_PER_SEGMENT * PIXELS_PER_LATENT
LATENT_FRAMES = SEGMENTS * LATENTS_PER_SEGMENT
PIXEL_FRAMES = LATENT_FRAMES * PIXELS_PER_LATENT
LATENT_SEED = 42
CACHE_EPS_SEED = 42
CACHE_AT_STEP = -1
CACHE_NOISE_SIGMA = 0.206897
CACHE_TIMESTEP = 206.897
INFER_STEPS_VIDEO = 4
SOURCE_CONFIG_INFER_STEPS_VIDEO = 8
FIXED_NOISE_MAX_SECONDS = 120
VIDEO_FPS = 24
FIXED_NOISE_LATENT_FRAMES = ((FIXED_NOISE_MAX_SECONDS * VIDEO_FPS - 1) // 4) + 1

LATENT_BANK_SHAPE = (1, 32, LATENT_FRAMES, 9, 16)
FULL_LATENT_DRAW_SHAPE = (1, 32, FIXED_NOISE_LATENT_FRAMES, 9, 16)
QWEN_SHAPE = (1, 1280, 3584)
DIT_ROPE_GRID = (LATENTS_PER_SEGMENT, 9, 16)
DIT_ROPE_SHAPE = (math.prod(DIT_ROPE_GRID) * 1, 128)
DIT_ROPE_BANK_SHAPE = (SEGMENTS, *DIT_ROPE_SHAPE)
CACHE_EPS_DRAW_SHAPE = (1, 32, LATENTS_PER_SEGMENT, 9, 16)
CACHE_EPS_BANK_SHAPE = (SEGMENTS, *CACHE_EPS_DRAW_SHAPE)
PIXEL_NOISE_SHAPE = (1, 3, PIXEL_FRAMES, 288, 512)
PIXEL_ROPE_GRID = (10, 18, 32)
PIXEL_ROPE_SHAPE = (1, math.prod(PIXEL_ROPE_GRID), 1, 128)
PIXEL_NOISE_SEED_FORMULA = "seed * 1000003 + global_latent_frame_index"
ROPE_OFFSETS = tuple(index * LATENTS_PER_SEGMENT for index in range(SEGMENTS))

UPSTREAM_SOURCE_PATHS = (
    "hymm/models/dit/modules/posemb_layers.py",
    "hymm/models/vae/pixelumm_decoder/model.py",
    "hymm/models/vae/pixelumm_decoder/autoencoder.py",
    "hymm/pipeline/causal_ar.py",
    "configs/experiment/inference.yaml",
)
PINNED_UPSTREAM_SOURCE_SHA256 = {
    "hymm/models/dit/modules/posemb_layers.py": "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa",
    "hymm/models/vae/pixelumm_decoder/model.py": "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec",
    "hymm/models/vae/pixelumm_decoder/autoencoder.py": "65e0d77b9f2e9a3ca457d6a51bfdb239f7b9dc3fc8a0ad014025c36d6be9f0cf",
    "hymm/pipeline/causal_ar.py": "f9aae49b14becb87d958665f763e510c28eeaaad9292f00ac511d1146202949f",
    "configs/experiment/inference.yaml": "d4db57b63597f98d4bddddd98355198e4e8c25da366d2e27bc7e74a79d26ec67",
}

TENSOR_SPECS = (
    ("input.latent_bank", "F32", LATENT_BANK_SHAPE),
    ("input.qwen_bf16", "BF16", QWEN_SHAPE),
    ("input.rope_cos", "F32", DIT_ROPE_BANK_SHAPE),
    ("input.rope_sin", "F32", DIT_ROPE_BANK_SHAPE),
    ("cache.eps_bf16", "BF16", CACHE_EPS_BANK_SHAPE),
    ("decoder.noise", "F32", PIXEL_NOISE_SHAPE),
    ("decoder.rope_cos", "F32", PIXEL_ROPE_SHAPE),
    ("decoder.rope_sin", "F32", PIXEL_ROPE_SHAPE),
)
_WIDTH = {"F32": 4, "BF16": 2}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_SHA256 = "98542072506741dc256d1f0d69c023efbc45f0b2fa057dc617c2ea1159b35435"
_DETAILED_PAYLOAD_SHA256 = "078fc68e0e398c080fa4b70c4bbf641659178f13e6da09f7de551ac9a38ec1a1"
ROPE_ATOL = 2e-5
ROPE_RTOL = 2e-5


class InputsError(RuntimeError):
    """Continuation input provenance, geometry, or MPS preparation failure."""


def tensor_nbytes(dtype: str, shape: tuple[int, ...] | list[int]) -> int:
    if dtype not in _WIDTH:
        raise InputsError(f"unsupported tensor dtype: {dtype}")
    if (not isinstance(shape, (tuple, list)) or not shape
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise InputsError("tensor shape must contain positive integers")
    return math.prod(shape) * _WIDTH[dtype]


def expected_payload_bytes() -> int:
    return sum(tensor_nbytes(dtype, shape) for _, dtype, shape in TENSOR_SPECS)


def _same_json_types(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return (type(actual) is dict and set(actual) == set(expected)
                and all(_same_json_types(value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (type(actual) is list and len(actual) == len(expected)
                and all(_same_json_types(left, right) for left, right in zip(expected, actual)))
    return type(actual) is type(expected) and actual == expected


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_range(path: Path, offset: int, size: int) -> str:
    if offset < 0 or size <= 0 or offset + size > path.stat().st_size:
        raise InputsError("source tensor range is outside its payload")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            block = stream.read(min(remaining, 8 * 1024 * 1024))
            if not block:
                raise InputsError("source tensor payload was truncated while hashing")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _regular_canonical_file(path_arg: Path | str, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise InputsError(f"{label} must be an existing absolute file") from None
    output_alias = path == OUTPUT_ROOT or OUTPUT_ROOT in path.parents
    if (path.is_symlink() or not resolved.is_file()
            or (path != resolved and not output_alias)):
        raise InputsError(f"{label} must be a canonical regular file")
    return resolved


def validate_output_paths(output_arg: Path | str) -> tuple[Path, Path]:
    output = Path(output_arg).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.suffix.lower() != ".json":
        raise InputsError("--output must have a .json suffix")
    try:
        root = OUTPUT_ROOT.resolve(strict=True)
        parent = output.parent.resolve(strict=True)
    except OSError:
        raise InputsError("output root and parent must already exist") from None
    if parent != root and root not in parent.parents:
        raise InputsError("output parent must be canonical and beneath /opt/mldata_ssd")
    manifest = parent / output.name
    payload = parent / f"{output.stem}.tensors.bin"
    if os.path.lexists(manifest) or os.path.lexists(payload):
        raise InputsError("output manifest or sibling payload already exists")
    return manifest, payload


def _admit_qwen_inputs(path_arg: Path | str, expected_manifest_sha256: str) -> dict[str, Any]:
    if not isinstance(expected_manifest_sha256, str) or not _DIGEST.fullmatch(expected_manifest_sha256):
        raise InputsError("--qwen-inputs-sha256 must be 64 lowercase hex characters")
    path = _regular_canonical_file(path_arg, "Qwen input manifest")
    before = path.stat()
    if before.st_size > 1024 * 1024:
        raise InputsError("Qwen input manifest exceeds its size limit")
    raw = path.read_bytes()
    after = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise InputsError("Qwen input manifest changed while being read")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise InputsError("Qwen input manifest digest differs from the admitted digest")
    if path == DEFAULT_QWEN_INPUTS.resolve(strict=True) and manifest_sha256 != _MANIFEST_SHA256:
        raise InputsError("detailed v2 source manifest is not the pinned artifact")
    try:
        document = json.loads(raw.decode("utf-8"))
        import hy_nomas_generation_inputs as generation_inputs
        generation_inputs.validate_document(document)
    except Exception as error:
        raise InputsError(f"Qwen generation-input manifest rejected: {error}") from error
    if document.get("schema") != "cogni_ml.hy_nomas_generation_inputs/v2":
        raise InputsError("continuation inputs require a detailed v2 Qwen bundle")
    if document.get("upstream_revision") != PINNED_REVISION:
        raise InputsError("Qwen input upstream revision mismatch")
    qwen_record = document["tensors"]["input.qwen_bf16"]
    if (qwen_record.get("dtype") != "BF16" or qwen_record.get("shape") != list(QWEN_SHAPE)
            or qwen_record.get("nbytes") != tensor_nbytes("BF16", QWEN_SHAPE)):
        raise InputsError("Qwen source tensor geometry mismatch")
    payload_path = _regular_canonical_file(path.parent / document["payload"], "Qwen input payload")
    if payload_path.parent != path.parent:
        raise InputsError("Qwen input payload must be a sibling of its manifest")
    payload_before = payload_path.stat()
    if payload_before.st_size != document.get("payload_bytes"):
        raise InputsError("Qwen input payload extent mismatch")
    payload_sha256 = _hash_file(payload_path)
    payload_after = payload_path.stat()
    if ((payload_before.st_dev, payload_before.st_ino, payload_before.st_size, payload_before.st_mtime_ns)
            != (payload_after.st_dev, payload_after.st_ino, payload_after.st_size, payload_after.st_mtime_ns)):
        raise InputsError("Qwen input payload changed while being authenticated")
    if payload_sha256 != document.get("payload_sha256"):
        raise InputsError("Qwen input aggregate payload digest mismatch")
    if path == DEFAULT_QWEN_INPUTS.resolve(strict=True) and payload_sha256 != _DETAILED_PAYLOAD_SHA256:
        raise InputsError("detailed v2 source payload is not the pinned artifact")
    if _hash_range(payload_path, qwen_record["offset"], qwen_record["nbytes"]) != qwen_record["sha256"]:
        raise InputsError("Qwen tensor record digest differs from its input payload")

    source_cos = document["tensors"]["input.rope_cos"]
    source_sin = document["tensors"]["input.rope_sin"]
    for name, record in (("input.rope_cos", source_cos), ("input.rope_sin", source_sin)):
        if _hash_range(payload_path, record["offset"], record["nbytes"]) != record["sha256"]:
            raise InputsError(f"source first-segment RoPE digest mismatch: {name}")

    qwen_source = document["qwen_source"]
    reference_path = _regular_canonical_file(qwen_source["reference_path"], "Qwen reference")
    sidecar_path = _regular_canonical_file(qwen_source["sidecar_path"], "Qwen hidden-state sidecar")
    if _hash_file(reference_path) != qwen_source["reference_sha256"]:
        raise InputsError("Qwen reference manifest digest mismatch")
    if (sidecar_path.stat().st_size != qwen_source["sidecar_bytes"]
            or _hash_file(sidecar_path) != qwen_source["sidecar_sha256"]):
        raise InputsError("Qwen hidden-state sidecar identity mismatch")
    try:
        reference_document = json.loads(reference_path.read_text(encoding="utf-8"))
        descriptor = reference_document["sidecars"]["bf16"]
        described_sidecar = (reference_path.parent / descriptor["path"]).resolve(strict=True)
    except Exception as error:
        raise InputsError(f"Qwen reference descriptor is malformed: {error}") from error
    if (reference_document.get("schema") != "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference"
            or reference_document.get("schema_version") != 1
            or reference_document.get("hy_nomas_prompt") != qwen_source["prompt_contract"]
            or described_sidecar != sidecar_path
            or descriptor.get("sha256") != qwen_source["sidecar_sha256"]
            or descriptor.get("nbytes") != qwen_source["sidecar_bytes"]
            or descriptor.get("encoding") != "bfloat16-le"):
        raise InputsError("Qwen reference does not bind the admitted source sidecar and prompt")
    crop_start, crop_stop = qwen_source["crop"]
    source_qwen_offset = (
        (qwen_source["state_index"] * qwen_source["raw_sequence_length"] + crop_start)
        * QWEN_SHAPE[2] * 2
    )
    source_qwen_sha256 = _hash_range(sidecar_path, source_qwen_offset, qwen_record["nbytes"])
    if source_qwen_sha256 != qwen_record["sha256"]:
        raise InputsError("Qwen tensor does not match the authenticated state26 crop")

    return {
        "path": str(path),
        "manifest_sha256": manifest_sha256,
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha256,
        "document": document,
        "qwen_record": qwen_record,
        "qwen_source": qwen_source,
        "qwen_sidecar_sha256": source_qwen_sha256,
        "rope_records": {"cos": source_cos, "sin": source_sin},
    }


def _verify_upstream(upstream_arg: Path | str) -> tuple[Path, dict[str, str]]:
    upstream = Path(upstream_arg).expanduser().resolve(strict=True)
    if not upstream.is_dir():
        raise InputsError("--upstream must be a directory")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        raise InputsError("cannot verify pinned upstream revision") from None
    if revision != PINNED_REVISION:
        raise InputsError("upstream revision mismatch")
    observed: dict[str, str] = {}
    for relative in UPSTREAM_SOURCE_PATHS:
        try:
            changed = subprocess.check_output(
                ["git", "-C", str(upstream), "status", "--porcelain", "--", relative],
                text=True, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            raise InputsError(f"cannot inspect pinned upstream source: {relative}") from None
        if changed.strip():
            raise InputsError(f"pinned upstream source has local changes: {relative}")
        source = upstream / relative
        if not source.is_file() or source.is_symlink():
            raise InputsError(f"pinned upstream source is not a regular file: {relative}")
        actual = _hash_file(source)
        expected = PINNED_UPSTREAM_SOURCE_SHA256[relative]
        if actual != expected:
            raise InputsError(f"pinned upstream source digest mismatch: {relative}")
        observed[relative] = actual
    return upstream, observed


def _geometry() -> dict[str, Any]:
    return {
        "total_latents": LATENT_FRAMES,
        "segments": SEGMENTS,
        "segment_latents": LATENTS_PER_SEGMENT,
        "total_pixel_frames": PIXEL_FRAMES,
        "segment_pixel_frames": PIXELS_PER_SEGMENT,
        "latent_bank_shape": list(LATENT_BANK_SHAPE),
        "full_latent_draw_shape": list(FULL_LATENT_DRAW_SHAPE),
        "dit_rope_offsets": list(ROPE_OFFSETS),
        "dit_rope_grid": list(DIT_ROPE_GRID),
        "dit_rope_axis_dims": [16, 56, 56],
        "dit_rope_theta": 256.0,
        "cache_eps_shape": list(CACHE_EPS_BANK_SHAPE),
        "qwen_shape": list(QWEN_SHAPE),
        "decoder_noise_shape": list(PIXEL_NOISE_SHAPE),
        "decoder_rope_grid": list(PIXEL_ROPE_GRID),
        "decoder_rope_shape": list(PIXEL_ROPE_SHAPE),
        "decoder_rope_sections": [64, 32, 32],
        "decoder_rope_theta": [1_000_000.0, 10_000.0, 10_000.0],
        "decoder_local_attention_tile_tokens": 8,
    }


def _noise_seeds() -> dict[str, Any]:
    return {
        "latent_seed": LATENT_SEED,
        "latent_generator_device": "mps",
        "latent_draw_shape": list(FULL_LATENT_DRAW_SHAPE),
        "latent_export_slice": [0, LATENT_FRAMES],
        "upstream_interactive_bank": {
            "seconds": FIXED_NOISE_MAX_SECONDS,
            "video_fps": VIDEO_FPS,
            "latent_frames": FIXED_NOISE_LATENT_FRAMES,
            "seed": LATENT_SEED,
            "generator_device": "cpu",
            "note": "MPS values are not claimed identical to upstream CPU torch.randn values",
        },
        "pixel_noise_seed": LATENT_SEED,
        "pixel_noise_seed_formula": PIXEL_NOISE_SEED_FORMULA,
        "pixel_noise_latent_indices": list(range(LATENT_FRAMES)),
        "cache_eps_seed": CACHE_EPS_SEED,
        "cache_eps_generator_device": "mps",
        "cache_eps_sample_dtype": "F32",
        "cache_eps_output_dtype": "BF16",
        "cache_eps_draw_shape": list(CACHE_EPS_DRAW_SHAPE),
        "cache_eps_draw_count": SEGMENTS,
        "cache_eps_draw_order": "six sequential segment-shaped draws from a separate MPS generator",
        "cache_eps_stock_upstream_cpu_rng_parity": False,
    }


def _cache_writer_contract() -> dict[str, Any]:
    """Pin the demo cache-write path while keeping MPS RNG claims explicit."""
    return {
        "config_path": "configs/experiment/inference.yaml",
        "config_sha256": PINNED_UPSTREAM_SOURCE_SHA256["configs/experiment/inference.yaml"],
        "source_config_default_infer_steps_video": SOURCE_CONFIG_INFER_STEPS_VIDEO,
        "resolved_override": {"inference.infer_steps_video": INFER_STEPS_VIDEO},
        "preserve_kv_cache": True,
        "fixed_initial_noise_timeline": {
            "seconds": FIXED_NOISE_MAX_SECONDS,
            "video_fps": VIDEO_FPS,
            "latent_frames": FIXED_NOISE_LATENT_FRAMES,
            "seed": LATENT_SEED,
            "generator_device": "mps",
            "sample_dtype": "F32",
            "stock_upstream_generator_device": "cpu",
            "stock_upstream_sample_dtype": "BF16",
            "stock_upstream_rng_values_identical": False,
        },
        "cache_at_step": CACHE_AT_STEP,
        "cache_noise_sigma": CACHE_NOISE_SIGMA,
        "cache_timestep": CACHE_TIMESTEP,
        "cache_latent_source": "final denoised segment latent selected by cache_at_step=-1",
        "cache_write_formula": "((1-sigma)*cache_latents.float()+sigma*cache_eps.float()).to(cache_latents.dtype)",
        "cache_eps_tensor": "cache.eps_bf16",
        "cache_eps_seed": CACHE_EPS_SEED,
        "cache_eps_generator_device": "mps",
        "cache_eps_sample_dtype": "F32 then cast to BF16",
        "cache_eps_dtype": "BF16",
        "cache_eps_upstream_cpu_rng_parity": False,
        "cache_eps_draws": [
            {
                "segment_index": index,
                "global_latent_start": index * LATENTS_PER_SEGMENT,
                "global_latent_stop": (index + 1) * LATENTS_PER_SEGMENT,
                "shape": list(CACHE_EPS_DRAW_SHAPE),
                "dtype": "BF16",
                "generator_device": "mps",
                "seed": CACHE_EPS_SEED,
                "draw_order": index,
            }
            for index in range(SEGMENTS)
        ],
    }


def tensor_specs() -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    return TENSOR_SPECS


def _load_pinned_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise InputsError(f"cannot load pinned upstream module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise InputsError(f"imported a different module than the pinned source: {path}")
    return module


def _load_pixelumm_modules(upstream: Path) -> tuple[Any, Any]:
    upstream_text = str(upstream)
    if upstream_text not in sys.path:
        sys.path.insert(0, upstream_text)
    try:
        pixel_model = importlib.import_module("hymm.models.vae.pixelumm_decoder.model")
        autoencoder = importlib.import_module("hymm.models.vae.pixelumm_decoder.autoencoder")
    except Exception as error:
        raise InputsError(f"cannot import pinned PixelUMM input helpers: {error}") from error
    expected = {
        pixel_model: upstream / "hymm/models/vae/pixelumm_decoder/model.py",
        autoencoder: upstream / "hymm/models/vae/pixelumm_decoder/autoencoder.py",
    }
    for module, source_path in expected.items():
        if Path(module.__file__).resolve() != source_path.resolve():
            raise InputsError(f"Python imported a different PixelUMM module: {source_path}")
    return pixel_model, autoencoder


def _draw_latent_bank(torch: Any, device: Any) -> Any:
    """Draw one 120-second MPS bank, returning exactly its first 180 latents."""
    generator = torch.Generator(device=device)
    generator.manual_seed(LATENT_SEED)
    full_bank = torch.randn(
        FULL_LATENT_DRAW_SHAPE,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    if full_bank.device.type != "mps" or tuple(full_bank.shape) != FULL_LATENT_DRAW_SHAPE:
        raise InputsError("fixed-duration latent draw did not stay on MPS with pinned geometry")
    return full_bank[:, :, :LATENT_FRAMES, :, :].contiguous()


def _draw_dit_rope_bank(torch: Any, posemb_layers: Any, device: Any) -> tuple[Any, Any]:
    cos_tables = []
    sin_tables = []
    with torch.inference_mode():
        for offset in ROPE_OFFSETS:
            cos, sin = posemb_layers.get_nd_rotary_pos_embed(
                [16, 56, 56],
                (offset, 0, 0),
                (offset + LATENTS_PER_SEGMENT, 9, 16),
                theta=256.0,
                use_real=True,
                theta_rescale_factor=1,
                device=device,
            )
            cos_tables.append(cos)
            sin_tables.append(sin)
        return torch.stack(cos_tables, dim=0), torch.stack(sin_tables, dim=0)


def _read_qwen_on_mps(torch: Any, np: Any, source: dict[str, Any], device: Any) -> Any:
    record = source["qwen_record"]
    payload_path = Path(source["payload_path"])
    raw = np.memmap(payload_path, mode="r", dtype=np.uint8)
    start, stop = record["offset"], record["offset"] + record["nbytes"]
    qwen_bytes = np.array(raw[start:stop], copy=True)
    del raw
    if hashlib.sha256(qwen_bytes).hexdigest() != record["sha256"]:
        raise InputsError("Qwen tensor bytes changed after source admission")
    qwen_cpu = torch.from_numpy(qwen_bytes).view(torch.bfloat16).reshape(QWEN_SHAPE)
    del qwen_bytes
    qwen = qwen_cpu.to(device=device, dtype=torch.bfloat16)
    del qwen_cpu
    if qwen.device.type != "mps" or qwen.dtype is not torch.bfloat16 or tuple(qwen.shape) != QWEN_SHAPE:
        raise InputsError("Qwen input transport changed device, dtype, or geometry")
    return qwen


def _draw_decoder_noise(torch: Any, autoencoder: Any, latent_bank: Any) -> Any:
    noise = autoencoder.PixelUMMDCAELatentDecoder._make_noise(
        SimpleNamespace(seed=LATENT_SEED), latent_bank, 0, LATENT_FRAMES
    )
    if noise.device.type != "mps" or tuple(noise.shape) != PIXEL_NOISE_SHAPE:
        raise InputsError("pinned PixelUMM noise method returned unexpected device or geometry")
    return noise


def _draw_cache_eps_bank(torch: Any, device: Any) -> Any:
    """Generate six external MPS cache-epsilon inputs in segment draw order.

    The pinned web path draws BF16 epsilons from its CPU generator after the
    fixed CPU timeline bank. This harness intentionally makes independent MPS
    values for both implementations to share; it does not claim CPU RNG parity.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(CACHE_EPS_SEED)
    draws = []
    for _ in range(SEGMENTS):
        sample = torch.randn(
            CACHE_EPS_DRAW_SHAPE,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        draws.append(sample.to(dtype=torch.bfloat16))
    bank = torch.stack(draws, dim=0)
    if (bank.device.type != "mps" or bank.dtype is not torch.bfloat16
            or tuple(bank.shape) != CACHE_EPS_BANK_SHAPE):
        raise InputsError("cache epsilon bank has unexpected MPS device, BF16 dtype, or geometry")
    return bank


def _draw_decoder_rope(torch: Any, pixel_model: Any, device: Any) -> tuple[Any, Any]:
    with torch.inference_mode():
        cos, sin = pixel_model.mrope_tables(pixel_model.QWEN3_8B_GEN, PIXEL_ROPE_GRID, device)
        layout = pixel_model.local_attention_layout(PIXEL_ROPE_GRID, 8, device)
        if layout is None:
            raise InputsError("pinned PixelUMM local layout returned no tile permutation")
        permutation, inverse, block_mask = layout
        cos = cos[:, permutation]
        sin = sin[:, permutation]
        del inverse, block_mask, layout, permutation
    if (cos.device.type != "mps" or sin.device.type != "mps"
            or tuple(cos.shape) != PIXEL_ROPE_SHAPE or tuple(sin.shape) != PIXEL_ROPE_SHAPE):
        raise InputsError("pinned PixelUMM RoPE returned unexpected device or geometry")
    return cos, sin


def _tensor_byte_views(torch: Any, tensors: Mapping[str, Any]) -> list[Any]:
    expected = {name: (dtype, shape) for name, dtype, shape in TENSOR_SPECS}
    if set(tensors) != set(expected):
        raise InputsError("continuation tensor inventory is incomplete")
    byte_views = []
    for name, tensor in tensors.items():
        dtype, shape = expected[name]
        wanted_dtype = torch.float32 if dtype == "F32" else torch.bfloat16
        if (tensor.device.type != "mps" or tensor.dtype is not wanted_dtype
                or tuple(tensor.shape) != shape):
            raise InputsError(f"continuation tensor has unexpected device/dtype/shape: {name}")
        byte_views.append(tensor.contiguous().view(torch.uint8).reshape(-1))
    return byte_views


def _terminal_export(torch: Any, tensors: Mapping[str, Any]) -> memoryview:
    packed_mps = torch.cat(_tensor_byte_views(torch, tensors), dim=0)
    if packed_mps.device.type != "mps" or packed_mps.dtype is not torch.uint8:
        raise InputsError("packed continuation payload left MPS before terminal export")
    # Exactly one MPS-to-host transfer and synchronization for produced tensors.
    packed_cpu = packed_mps.cpu()
    # Keep a zero-copy host view alive through validation and file publication.
    return memoryview(packed_cpu.numpy())


def _records_from_payload(payload: bytes | bytearray | memoryview) -> dict[str, dict[str, Any]]:
    if len(payload) != expected_payload_bytes():
        raise InputsError("packed payload extent differs from the fixed continuation inventory")
    view = memoryview(payload)
    records = {}
    offset = 0
    for name, dtype, shape in TENSOR_SPECS:
        nbytes = tensor_nbytes(dtype, shape)
        records[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "offset": offset,
            "nbytes": nbytes,
            "sha256": hashlib.sha256(view[offset:offset + nbytes]).hexdigest(),
        }
        offset += nbytes
    return records


def validate_document(document: Mapping[str, Any], *, payload_bytes: int | None = None) -> None:
    required = {
        "schema", "schema_version", "payload", "payload_bytes", "payload_sha256",
        "tensors", "geometry", "prompt", "qwen_source", "qwen_input_bundle",
        "noise_seeds", "cache_writer", "upstream_revision", "upstream_sources", "producer",
    }
    if type(document) is not dict or set(document) != required:
        raise InputsError("continuation input manifest keys are malformed")
    if document.get("schema") != SCHEMA or type(document.get("schema_version")) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise InputsError("continuation input schema mismatch")
    if (not isinstance(document.get("payload"), str) or Path(document["payload"]).name != document["payload"]
            or not document["payload"].endswith(".tensors.bin")):
        raise InputsError("continuation payload must be a sibling .tensors.bin basename")
    if not isinstance(document.get("payload_sha256"), str) or not _DIGEST.fullmatch(document["payload_sha256"]):
        raise InputsError("continuation aggregate payload digest is malformed")
    if not _same_json_types(_geometry(), document.get("geometry")):
        raise InputsError("continuation geometry mismatch")
    if not _same_json_types(_noise_seeds(), document.get("noise_seeds")):
        raise InputsError("continuation noise seed inventory mismatch")
    if not _same_json_types(_cache_writer_contract(), document.get("cache_writer")):
        raise InputsError("cache-writer configuration or MPS epsilon contract mismatch")
    if document.get("upstream_revision") != PINNED_REVISION:
        raise InputsError("pinned upstream revision mismatch")
    if not _same_json_types(PINNED_UPSTREAM_SOURCE_SHA256, document.get("upstream_sources")):
        raise InputsError("pinned upstream source digest inventory mismatch")
    if not isinstance(document.get("prompt"), str) or not document["prompt"]:
        raise InputsError("continuation prompt must be nonempty text")
    if type(document.get("qwen_source")) is not dict:
        raise InputsError("continuation Qwen source record is malformed")
    qwen_source = document["qwen_source"]
    prompt_contract = qwen_source.get("prompt_contract")
    if (type(prompt_contract) is not dict or prompt_contract.get("prompt") != document["prompt"]
            or qwen_source.get("prompt_sha256") != hashlib.sha256(document["prompt"].encode()).hexdigest()
            or prompt_contract.get("prompt_sha256") != qwen_source.get("prompt_sha256")
            or qwen_source.get("state_index") != 26 or qwen_source.get("crop_tokens") != 1280
            or qwen_source.get("hidden_size") != 3584 or qwen_source.get("upstream_revision") != PINNED_REVISION):
        raise InputsError("continuation Qwen record does not bind its prompt and source geometry")
    bundle = document.get("qwen_input_bundle")
    if (type(bundle) is not dict or set(bundle) != {
            "manifest_path", "manifest_sha256", "payload_path", "payload_sha256",
            "qwen_tensor_sha256", "first_rope_cos_sha256", "first_rope_sin_sha256",
            "first_rope_atol", "first_rope_rtol"}):
        raise InputsError("Qwen source bundle identity is malformed")
    for key in ("manifest_sha256", "payload_sha256", "qwen_tensor_sha256", "first_rope_cos_sha256", "first_rope_sin_sha256"):
        if not isinstance(bundle.get(key), str) or not _DIGEST.fullmatch(bundle[key]):
            raise InputsError(f"Qwen source bundle digest is malformed: {key}")
    for key in ("manifest_path", "payload_path"):
        if not isinstance(bundle.get(key), str) or not Path(bundle[key]).is_absolute():
            raise InputsError(f"Qwen source bundle {key} must be absolute")
    if bundle.get("first_rope_atol") != ROPE_ATOL or bundle.get("first_rope_rtol") != ROPE_RTOL:
        raise InputsError("first-segment RoPE comparison tolerance mismatch")

    records = document.get("tensors")
    if type(records) is not dict or set(records) != {name for name, _, _ in TENSOR_SPECS}:
        raise InputsError("continuation tensor inventory is incomplete")
    offset = 0
    for name, dtype, shape in TENSOR_SPECS:
        record = records[name]
        nbytes = tensor_nbytes(dtype, shape)
        if (type(record) is not dict or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}
                or record.get("dtype") != dtype or not _same_json_types(list(shape), record.get("shape"))
                or type(record.get("offset")) is not int or record["offset"] != offset
                or type(record.get("nbytes")) is not int or record["nbytes"] != nbytes
                or not isinstance(record.get("sha256"), str) or not _DIGEST.fullmatch(record["sha256"])):
            raise InputsError(f"continuation tensor record mismatch: {name}")
        offset += nbytes
    if bundle["qwen_tensor_sha256"] != records["input.qwen_bf16"]["sha256"]:
        raise InputsError("Qwen source bundle digest differs from copied tensor record")
    if type(document.get("payload_bytes")) is not int or document["payload_bytes"] != offset:
        raise InputsError("continuation payload byte count mismatch")
    if payload_bytes is not None and payload_bytes != offset:
        raise InputsError("continuation payload length differs from its inventory")
    producer = document.get("producer")
    if (type(producer) is not dict or producer.get("device") != "mps"
            or producer.get("mps_fallback") != "disabled"
            or producer.get("model_inference") is not False
            or producer.get("tensor_generation_device") != "mps"
            or producer.get("host_export") != "one packed terminal MPS-to-host readback"
            or type(producer.get("torch_version")) is not str or not producer["torch_version"]
            or type(producer.get("python_version")) is not str or not producer["python_version"]):
        raise InputsError("continuation producer scope is not admitted")


def validate_payload(document: Mapping[str, Any], payload: bytes | bytearray | memoryview) -> None:
    validate_document(document, payload_bytes=len(payload))
    view = memoryview(payload)
    if hashlib.sha256(view).hexdigest() != document["payload_sha256"]:
        raise InputsError("continuation aggregate payload digest mismatch")
    for name, _, _ in TENSOR_SPECS:
        record = document["tensors"][name]
        start = record["offset"]
        stop = start + record["nbytes"]
        if hashlib.sha256(view[start:stop]).hexdigest() != record["sha256"]:
            raise InputsError(f"continuation tensor payload digest mismatch: {name}")


def _build_document(
    payload: bytes | bytearray | memoryview,
    payload_name: str,
    source_hashes: Mapping[str, str],
    qwen_source: Mapping[str, Any],
    qwen_bundle: Mapping[str, Any],
    torch: Any,
    *,
    rope_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    records = _records_from_payload(payload)
    bundle = dict(qwen_bundle)
    bundle["first_rope_cos_sha256"] = rope_comparison["cos_source_sha256"]
    bundle["first_rope_sin_sha256"] = rope_comparison["sin_source_sha256"]
    bundle["first_rope_atol"] = ROPE_ATOL
    bundle["first_rope_rtol"] = ROPE_RTOL
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "payload": payload_name,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "tensors": records,
        "geometry": _geometry(),
        "prompt": qwen_source["prompt_contract"]["prompt"],
        "qwen_source": dict(qwen_source),
        "qwen_input_bundle": bundle,
        "noise_seeds": _noise_seeds(),
        "cache_writer": _cache_writer_contract(),
        "upstream_revision": PINNED_REVISION,
        "upstream_sources": dict(source_hashes),
        "producer": {
            "device": "mps",
            "mps_fallback": "disabled",
            "model_inference": False,
            "tensor_generation_device": "mps",
            "host_export": "one packed terminal MPS-to-host readback",
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "rope_first_segment_check": dict(rope_comparison),
        },
    }
    validate_document(document, payload_bytes=len(payload))
    validate_payload(document, payload)
    return document


def _compare_first_rope(payload: bytes | bytearray | memoryview, source: dict[str, Any], np: Any) -> dict[str, Any]:
    records = _records_from_payload(payload)
    result = {}
    for tensor_name, source_name, key in (
        ("input.rope_cos", "cos", "cos"),
        ("input.rope_sin", "sin", "sin"),
    ):
        output_record = records[tensor_name]
        source_record = source["rope_records"][source_name]
        output_bytes = payload[output_record["offset"]:output_record["offset"] + output_record["nbytes"] // SEGMENTS]
        source_bytes = _read_range_bytes(Path(source["payload_path"]), source_record["offset"], source_record["nbytes"])
        output_array = np.frombuffer(output_bytes, dtype="<f4").reshape(DIT_ROPE_SHAPE)
        source_array = np.frombuffer(source_bytes, dtype="<f4").reshape(DIT_ROPE_SHAPE)
        if not np.allclose(output_array, source_array, atol=ROPE_ATOL, rtol=ROPE_RTOL):
            max_abs = float(np.max(np.abs(output_array - source_array)))
            raise InputsError(f"first-segment {key} RoPE differs from detailed v2 input (max_abs={max_abs})")
        result[f"{key}_source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        result[f"{key}_output_sha256"] = hashlib.sha256(output_bytes).hexdigest()
        result[f"{key}_max_abs_error"] = float(np.max(np.abs(output_array - source_array)))
    result["atol"] = ROPE_ATOL
    result["rtol"] = ROPE_RTOL
    result["source_manifest_sha256"] = source["manifest_sha256"]
    return result


def _read_range_bytes(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(offset)
        value = stream.read(size)
    if len(value) != size:
        raise InputsError("source tensor payload was truncated while reading")
    return value


def _publish_noreplace(path: Path, data: bytes | bytearray | memoryview) -> None:
    if os.path.lexists(path):
        raise InputsError(f"output already exists: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_bundle(
    output: Path,
    payload_path: Path,
    document: Mapping[str, Any],
    payload: bytes | bytearray | memoryview,
) -> None:
    if os.path.lexists(output) or os.path.lexists(payload_path):
        raise InputsError("output manifest or sibling payload already exists")
    published_identity = None
    try:
        _publish_noreplace(payload_path, payload)
        stat = payload_path.stat()
        published_identity = (stat.st_dev, stat.st_ino)
        encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        _publish_noreplace(output, encoded)
    except Exception:
        if published_identity is not None and os.path.lexists(payload_path):
            stat = payload_path.stat()
            if (stat.st_dev, stat.st_ino) == published_identity:
                payload_path.unlink(missing_ok=True)
        raise


def _plan(
    output_arg: Path | str,
    qwen_inputs_arg: Path | str,
    qwen_inputs_sha256: str,
    upstream_arg: Path | str,
) -> dict[str, Any]:
    output, payload = validate_output_paths(output_arg)
    source = _admit_qwen_inputs(qwen_inputs_arg, qwen_inputs_sha256)
    upstream, hashes = _verify_upstream(upstream_arg)
    return {
        "mode": "plan-only",
        "schema": SCHEMA,
        "output": str(output),
        "payload": str(payload),
        "payload_bytes": expected_payload_bytes(),
        "tensor_inventory": [
            {"name": name, "dtype": dtype, "shape": list(shape), "nbytes": tensor_nbytes(dtype, shape)}
            for name, dtype, shape in TENSOR_SPECS
        ],
        "qwen_input_manifest_sha256": source["manifest_sha256"],
        "qwen_input_payload_sha256": source["payload_sha256"],
        "qwen_tensor_sha256": source["qwen_record"]["sha256"],
        "prompt_sha256": source["document"]["qwen_source"]["prompt_sha256"],
        "valid_text_tokens": source["document"]["qwen_source"]["valid_tokens"],
        "upstream_revision": PINNED_REVISION,
        "upstream_sources": hashes,
        "latent_bank_draw_shape": list(FULL_LATENT_DRAW_SHAPE),
        "latent_bank_export_shape": list(LATENT_BANK_SHAPE),
        "cache_writer": _cache_writer_contract(),
        "rope_offsets": list(ROPE_OFFSETS),
    }


def _load_qwen_for_output(torch: Any, np: Any, source: dict[str, Any], device: Any) -> Any:
    return _read_qwen_on_mps(torch, np, source, device)


def produce(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise InputsError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    output, payload_path = validate_output_paths(args.output)
    source = _admit_qwen_inputs(args.qwen_inputs, args.qwen_inputs_sha256)
    upstream, source_hashes = _verify_upstream(args.upstream)

    import numpy as np
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise InputsError("MPS is required for continuation input generation")
    device = torch.device("mps")
    posemb = _load_pinned_module(
        "hy_nomas_continuation_pinned_posemb_layers",
        upstream / "hymm/models/dit/modules/posemb_layers.py",
    )
    pixel_model, autoencoder = _load_pixelumm_modules(upstream)
    with torch.inference_mode():
        latent_bank = _draw_latent_bank(torch, device)
        qwen = _load_qwen_for_output(torch, np, source, device)
        rope_cos, rope_sin = _draw_dit_rope_bank(torch, posemb, device)
        decoder_noise = _draw_decoder_noise(torch, autoencoder, latent_bank)
        cache_eps = _draw_cache_eps_bank(torch, device)
        decoder_rope_cos, decoder_rope_sin = _draw_decoder_rope(torch, pixel_model, device)

    tensors = {
        "input.latent_bank": latent_bank,
        "input.qwen_bf16": qwen,
        "input.rope_cos": rope_cos,
        "input.rope_sin": rope_sin,
        "cache.eps_bf16": cache_eps,
        "decoder.noise": decoder_noise,
        "decoder.rope_cos": decoder_rope_cos,
        "decoder.rope_sin": decoder_rope_sin,
    }
    payload = _terminal_export(torch, tensors)
    qwen_offset = _records_from_payload(payload)["input.qwen_bf16"]["offset"]
    qwen_nbytes = tensor_nbytes("BF16", QWEN_SHAPE)
    output_qwen_sha = hashlib.sha256(payload[qwen_offset:qwen_offset + qwen_nbytes]).hexdigest()
    if output_qwen_sha != source["qwen_record"]["sha256"]:
        raise InputsError("Qwen BF16 tensor changed while copying from the authenticated v2 bundle")
    rope_comparison = _compare_first_rope(payload, source, np)
    qwen_bundle = {
        "manifest_path": source["path"],
        "manifest_sha256": source["manifest_sha256"],
        "payload_path": source["payload_path"],
        "payload_sha256": source["payload_sha256"],
        "qwen_tensor_sha256": source["qwen_record"]["sha256"],
        "first_rope_cos_sha256": rope_comparison["cos_source_sha256"],
        "first_rope_sin_sha256": rope_comparison["sin_source_sha256"],
        "first_rope_atol": ROPE_ATOL,
        "first_rope_rtol": ROPE_RTOL,
    }
    document = _build_document(
        payload,
        payload_path.name,
        source_hashes,
        source["document"]["qwen_source"],
        qwen_bundle,
        torch,
        rope_comparison=rope_comparison,
    )
    _publish_bundle(output, payload_path, document, payload)
    return {
        "mode": "produced",
        "manifest": str(output),
        "payload": str(payload_path),
        "payload_bytes": len(payload),
        "payload_sha256": document["payload_sha256"],
        "qwen_sha256": output_qwen_sha,
        "first_rope": rope_comparison,
        "latent_draw_shape": list(FULL_LATENT_DRAW_SHAPE),
        "latent_export_shape": list(LATENT_BANK_SHAPE),
        "cache_eps_shape": list(CACHE_EPS_BANK_SHAPE),
        "cache_writer": _cache_writer_contract(),
        "upstream_revision": PINNED_REVISION,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qwen-inputs", type=Path, default=DEFAULT_QWEN_INPUTS)
    parser.add_argument("--qwen-inputs-sha256", required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (_plan(args.output, args.qwen_inputs, args.qwen_inputs_sha256, args.upstream)
                  if args.plan_only else produce(args))
    except InputsError as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
