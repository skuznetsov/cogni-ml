#!/usr/bin/env python3
"""Produce bounded real-pose PixelUMM stem inputs on MPS.

The producer runs only the pinned DCAE encoder and PixelUMM input stem. It
never constructs or runs decoder layers. All tensor inference and noise
generation run on MPS; unsupported MPS operations fail the run without a CPU
fallback. The terminal MPS synchronization occurs before payload readback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

import hy_nomas_pixelumm_checkpoint as checkpoint_inventory


UPSTREAM_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_MODEL_RELATIVE = "hymm/models/vae/pixelumm_decoder/model.py"
DCAE_ENCODER_RELATIVE = "hymm/models/vae/autoencoder_dcae_v.py"
PIXELUMM_NOISE_RELATIVE = "hymm/models/vae/pixelumm_decoder/autoencoder.py"
UPSTREAM_MODEL_SHA256 = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
DCAE_ENCODER_SHA256 = "38163ebead9a6bb643d27843022c1757c45c6b8638c334a476aee5b373edf8c1"
PIXELUMM_NOISE_SHA256 = "65e0d77b9f2e9a3ca457d6a51bfdb239f7b9dc3fc8a0ad014025c36d6be9f0cf"
SOURCE_IMAGE = Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/inputs/pose.png")
SOURCE_IMAGE_SHA256 = "53a5d9b435adaf15dd8ffcab1a833b61da4a63079200fb9cec33127ee10f733b"
SIXTEEN_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261027-sixteen")
SIXTEEN_SOURCE_DIR = (SIXTEEN_ROOT / "source").resolve()
SUBSET_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/source")
HEADER_PATH = SUBSET_ROOT / "pixelumm-header.json"
ENCODER_SUBSET_PATH = SUBSET_ROOT / "encoder-f32.safetensors"
ENCODER_RECEIPT_PATH = SUBSET_ROOT / "encoder-f32.receipt.json"
STEM_SUBSET_PATH = SUBSET_ROOT / "stem-f32.safetensors"
STEM_RECEIPT_PATH = SUBSET_ROOT / "stem-f32.receipt.json"
REAL_INPUT_SCHEMA = "cogni_ml.pixelumm_real_input/v1"
DEFAULT_MANIFEST_NAME = "pixelumm-real-input.json"
DEFAULT_PAYLOAD_NAME = "pixelumm-real-input.bin"
GIB = 1024**3
ENCODER_F32_BYTES = 3_666_620_032
STEM_F32_BYTES = 52_510_720
STEM_RAW_KEYS = (
    "decoder.cond_proj.bias", "decoder.cond_proj.weight",
    "decoder.embed.bias", "decoder.embed.weight",
)


class RealInputError(ValueError):
    """The pinned real-input producer precondition or operation failed."""


class Profile:
    def __init__(
        self, name: str, grid: tuple[int, int, int], frames: int,
        crop: tuple[int, int, int, int], height: int, width: int,
        latent_shape: tuple[int, int, int, int, int], default_mps_bytes: int,
    ):
        self.name = name
        self.grid = grid
        self.frames = frames
        self.crop = crop
        self.height = height
        self.width = width
        self.latent_shape = latent_shape
        self.default_mps_bytes = default_mps_bytes
        self.xt_shape = (1, 3, frames, height, width)
        self.x_shape = (1, math.prod(grid), 4096)


PROFILES = {
    "full": Profile(
        "full", (6, 44, 80), 24, (0, 8, 1280, 712), 704, 1280,
        (1, 32, 6, 22, 40), 24 * GIB,
    ),
    "small": Profile(
        "small", (2, 32, 32), 8, (384, 104, 896, 616), 512, 512,
        (1, 32, 2, 16, 16), 8 * GIB,
    ),
}


def manifest_basename(profile: str, seed: int) -> str:
    """Return one of the consumer's fixed, basename-only artifact names."""
    _profile(profile)
    if type(seed) is not int or seed not in (42, 43):
        raise RealInputError("noise seed must be 42 or 43")
    if profile == "full" and seed == 42:
        return DEFAULT_MANIFEST_NAME
    return f"pixelumm-real-input-{profile}-seed{seed}.json"


def payload_basename(manifest_name: str) -> str:
    allowed = {
        DEFAULT_MANIFEST_NAME,
        *(f"pixelumm-real-input-{profile}-seed{seed}.json"
          for profile in PROFILES for seed in (42, 43)),
    }
    if manifest_name not in allowed or Path(manifest_name).name != manifest_name:
        raise RealInputError("manifest basename is outside the pinned real-input set")
    return Path(manifest_name).with_suffix(".bin").name


def _profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise RealInputError("profile must be full or small") from error


def _manifest_profile_metadata(profile: Profile, seed: int) -> dict[str, object]:
    if type(seed) is not int or seed not in (42, 43):
        raise RealInputError("noise seed must be 42 or 43")
    return {
        "schema": REAL_INPUT_SCHEMA,
        "profile": profile.name,
        "noise_seed": seed,
        "grid": list(profile.grid),
        "latent_shape": list(profile.latent_shape),
        "xt_shape": list(profile.xt_shape),
        "x_shape": list(profile.x_shape),
        "token_order": "tile_major",
    }


def expected_tensor_bytes(dtype: str, shape: Sequence[int]) -> int:
    width = {"BF16": 2, "F32": 4}.get(dtype)
    if width is None or not shape or any(type(value) is not int or value <= 0 for value in shape):
        raise RealInputError("tensor dtype or shape is unsupported")
    return math.prod(shape) * width


def estimate_mps_peak_bytes(profile: Profile) -> int:
    """Conservative static preflight for encoder, stem, and required tensors.

    Six full project-in activation buffers cover the wide first DCAE stage and
    convolution temporaries. The estimate also counts BF16 weights, the large
    condition projection temporary, all retained output tensors, and a 1 GiB
    allocator/kernel margin. MPS's per-process cap is enforced separately.
    """
    encoder_bf16 = ENCODER_F32_BYTES // 2
    stem_bf16 = STEM_F32_BYTES // 2
    input_video = expected_tensor_bytes("BF16", (1, 3, profile.frames, profile.height, profile.width))
    latent = expected_tensor_bytes("BF16", profile.latent_shape)
    xt = expected_tensor_bytes("F32", profile.xt_shape)
    x = expected_tensor_bytes("BF16", profile.x_shape)
    max_encoder_activation = (
        1 * 256 * profile.frames * (profile.height // 2) * (profile.width // 2) * 2
    )
    latent_t, latent_h, latent_w = profile.latent_shape[2:]
    condition_projection = latent_t * latent_h * latent_w * (4 * 4096) * 2
    return (
        encoder_bf16 + stem_bf16 + input_video + latent + xt + x
        + 6 * max_encoder_activation + condition_projection + GIB
    )


def memory_preflight(
    *, physical_bytes: int, driver_allocated_bytes: int,
    recommended_max_bytes: int, estimated_peak_bytes: int, max_mps_bytes: int,
) -> dict[str, int | float]:
    """Apply the 10% physical-memory floor and an explicit MPS allocator cap."""
    values = (physical_bytes, recommended_max_bytes, estimated_peak_bytes, max_mps_bytes)
    if any(type(value) is not int or value <= 0 for value in values):
        raise RealInputError("memory preflight inputs must be positive byte counts")
    if (type(driver_allocated_bytes) is not int or driver_allocated_bytes < 0
            or driver_allocated_bytes > physical_bytes):
        raise RealInputError("driver allocated bytes must be between zero and physical memory")
    floor = physical_bytes // 10
    available_for_run = physical_bytes - driver_allocated_bytes - floor
    if estimated_peak_bytes > max_mps_bytes:
        raise RealInputError("estimated MPS peak exceeds --max-mps-bytes")
    if max_mps_bytes > available_for_run:
        raise RealInputError("MPS cap would cross the required 10% physical-memory floor")
    if max_mps_bytes > recommended_max_bytes:
        raise RealInputError("MPS cap exceeds the device recommended memory limit")
    fraction = max_mps_bytes / recommended_max_bytes
    if not 0.0 < fraction <= 1.0:
        raise RealInputError("MPS memory cap cannot be represented as a safe allocator fraction")
    return {
        "physical_bytes": physical_bytes,
        "driver_allocated_bytes": driver_allocated_bytes,
        "recommended_max_bytes": recommended_max_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "max_mps_bytes": max_mps_bytes,
        "required_floor_bytes": floor,
        "headroom_after_cap_bytes": available_for_run - max_mps_bytes,
        "mps_memory_fraction": fraction,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, description: str, *, max_bytes: int | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RealInputError(f"{description} must be a regular non-symlink file")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise RealInputError(f"{description} exceeds its bounded size")
    return path


def _verified_upstream_sources(root: Path) -> dict[str, str]:
    root = root.expanduser().resolve()
    if root != UPSTREAM_ROOT.resolve():
        raise RealInputError("upstream root must be the pinned local checkout")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RealInputError("cannot read pinned upstream revision") from error
    if revision != UPSTREAM_REVISION:
        raise RealInputError("upstream checkout revision mismatch")
    expected = {
        "pixelumm_model_source_sha256": (UPSTREAM_MODEL_RELATIVE, UPSTREAM_MODEL_SHA256),
        "dcae_encoder_source_sha256": (DCAE_ENCODER_RELATIVE, DCAE_ENCODER_SHA256),
        "pixelumm_noise_source_sha256": (PIXELUMM_NOISE_RELATIVE, PIXELUMM_NOISE_SHA256),
    }
    observed = {}
    for key, (relative, pinned_digest) in expected.items():
        source = _regular_file(root / relative, f"upstream source {relative}")
        try:
            changed = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--", relative],
                text=True, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RealInputError(f"cannot verify upstream source status: {relative}") from error
        digest = _sha256_file(source)
        if changed.strip() or digest != pinned_digest:
            raise RealInputError(f"pinned upstream source changed: {relative}")
        observed[key] = digest
    return observed


def _raw_safetensors_payload_sha256(path: Path, expected_bytes: int) -> str:
    """Hash the contiguous raw F32 payload, excluding the safetensors header."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RealInputError("safetensors subset header is truncated")
        header_bytes = int.from_bytes(prefix, "little")
        if not 0 < header_bytes <= 2 * 1024 * 1024:
            raise RealInputError("safetensors subset header size is invalid")
        if len(stream.read(header_bytes)) != header_bytes:
            raise RealInputError("safetensors subset header is truncated")
        digest = hashlib.sha256()
        remaining = expected_bytes
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RealInputError("safetensors subset payload is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise RealInputError("safetensors subset has trailing data")
    return digest.hexdigest()


def _named_subset_file_bytes(checked: Mapping[str, object], kind: str,
                             expected_payload_bytes: int) -> int:
    """Validate F32 data extent while retaining the complete subset-file size."""
    if (checked.get("selected_subset") != kind
            or checked.get("source_f32_bytes") != expected_payload_bytes):
        raise RealInputError(f"pinned {kind} subset F32 payload extent mismatch")
    file_bytes = checked.get("subset_bytes")
    if type(file_bytes) is not int or file_bytes <= 0:
        raise RealInputError(f"pinned {kind} subset file byte count is malformed")
    return file_bytes


def _validate_subset_inputs() -> dict[str, object]:
    header_path = _regular_file(HEADER_PATH, "pinned checkpoint header", max_bytes=100_000)
    raw_header = header_path.read_bytes()
    try:
        checkpoint_inventory.parse_pinned_header(raw_header)
    except checkpoint_inventory.InventoryError as error:
        raise RealInputError(f"pinned checkpoint header rejected: {error}") from error
    result: dict[str, object] = {"header_sha256": hashlib.sha256(raw_header).hexdigest()}
    for kind, subset_path, receipt_path, expected_bytes in (
        ("encoder", ENCODER_SUBSET_PATH, ENCODER_RECEIPT_PATH, ENCODER_F32_BYTES),
        ("stem", STEM_SUBSET_PATH, STEM_RECEIPT_PATH, STEM_F32_BYTES),
    ):
        _regular_file(subset_path, f"pinned {kind} subset")
        receipt_path = _regular_file(receipt_path, f"pinned {kind} receipt", max_bytes=2 * 1024 * 1024)
        receipt_raw = receipt_path.read_bytes()
        try:
            receipt = checkpoint_inventory.parse_header(receipt_raw)
            checked = checkpoint_inventory.validate_named_subset_receipt(
                raw_header, receipt, subset_path, kind
            )
        except checkpoint_inventory.InventoryError as error:
            raise RealInputError(f"pinned {kind} subset receipt rejected: {error}") from error
        file_bytes = _named_subset_file_bytes(checked, kind, expected_bytes)
        result[f"{kind}_subset_sha256"] = checked["subset_sha256"]
        result[f"{kind}_receipt_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
        result[f"{kind}_subset_bytes"] = file_bytes
    result["stem_weights_sha256"] = _raw_safetensors_payload_sha256(
        STEM_SUBSET_PATH, STEM_F32_BYTES
    )
    return result


def _verify_image(profile: Profile) -> tuple[object, str]:
    image_path = _regular_file(SOURCE_IMAGE, "pinned source image", max_bytes=10 * 1024 * 1024)
    digest = _sha256_file(image_path)
    if digest != SOURCE_IMAGE_SHA256:
        raise RealInputError("official pose image digest mismatch")
    try:
        from PIL import Image
        with Image.open(image_path) as opened:
            if opened.format != "PNG" or opened.size != (1280, 720):
                raise RealInputError("official pose image must be the pinned 1280x720 PNG")
            image = opened.convert("RGB")
    except RealInputError:
        raise
    except Exception as error:
        raise RealInputError(f"cannot decode pinned source image: {error}") from error
    image = image.crop(profile.crop)
    if image.size != (profile.width, profile.height):
        raise RealInputError("profile crop dimensions do not match the admitted geometry")
    return image, digest


def _verify_mps(profile: Profile, requested_cap: int | None) -> dict[str, int | float]:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RealInputError("PyTorch MPS is unavailable; the producer has no CPU fallback")
    if not hasattr(torch.mps, "recommended_max_memory") or not hasattr(torch.mps, "driver_allocated_memory"):
        raise RealInputError("PyTorch MPS does not expose the required memory preflight APIs")
    try:
        physical_bytes = _physical_memory_bytes()
        driver_bytes = int(torch.mps.driver_allocated_memory())
        recommended_bytes = int(torch.mps.recommended_max_memory())
    except Exception as error:
        raise RealInputError(f"cannot query MPS memory state: {error}") from error
    cap = profile.default_mps_bytes if requested_cap is None else requested_cap
    return memory_preflight(
        physical_bytes=physical_bytes,
        driver_allocated_bytes=driver_bytes,
        recommended_max_bytes=recommended_bytes,
        estimated_peak_bytes=estimate_mps_peak_bytes(profile),
        max_mps_bytes=cap,
    )


def _physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise RealInputError("cannot determine physical memory for the 10% safety floor") from error
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as error:
        raise RealInputError("cannot determine physical memory for the 10% safety floor") from error


def _set_mps_memory_cap(memory: Mapping[str, int | float]) -> None:
    setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
    if setter is None:
        raise RealInputError("PyTorch MPS lacks a per-process allocator cap; refusing inference")
    try:
        setter(float(memory["mps_memory_fraction"]))
    except Exception as error:
        raise RealInputError(f"cannot install the MPS allocator cap: {error}") from error


def _mps_capability_probe() -> dict[str, object]:
    """Exercise the first required DCAE primitive on MPS before loading weights."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RealInputError("PYTORCH_ENABLE_MPS_FALLBACK is enabled; refusing a CPU fallback")
    operation = "torch.nn.functional.conv3d(dtype=torch.bfloat16, device=mps)"
    try:
        sample = torch.ones((1, 1, 3, 4, 4), dtype=torch.bfloat16, device="mps")
        weight = torch.ones((1, 1, 3, 3, 3), dtype=torch.bfloat16, device="mps")
        output = torch.nn.functional.conv3d(sample, weight, padding=1)
        if output.device.type != "mps" or output.dtype != torch.bfloat16 or tuple(output.shape) != (1, 1, 3, 4, 4):
            raise RealInputError(f"{operation} returned an unexpected result")
        # Synchronize this bounded capability check before allocating model weights.
        torch.mps.synchronize()
    except RealInputError:
        raise
    except Exception as error:
        raise RealInputError(f"MPS capability probe failed at {operation}: {error}") from error
    return {"operation": operation, "device": "mps", "dtype": "BF16", "passed": True}


def _load_modules(upstream: Path):
    upstream = upstream.expanduser().resolve()
    root_text = str(upstream)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_specs = (
        ("hymm.models.vae.autoencoder_dcae_v", DCAE_ENCODER_RELATIVE),
        ("hymm.models.vae.pixelumm_decoder.model", UPSTREAM_MODEL_RELATIVE),
        ("hymm.models.vae.pixelumm_decoder.autoencoder", PIXELUMM_NOISE_RELATIVE),
    )
    modules = {}
    for module_name, relative in module_specs:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise RealInputError(f"cannot import pinned upstream module {module_name}: {error}") from error
        if Path(module.__file__).resolve() != upstream / relative:
            raise RealInputError(f"Python imported a different module for {relative}")
        modules[module_name] = module
    return (
        modules["hymm.models.vae.autoencoder_dcae_v"],
        modules["hymm.models.vae.pixelumm_decoder.model"],
        modules["hymm.models.vae.pixelumm_decoder.autoencoder"],
    )


def _load_encoder(dcae_source, device: torch.device) -> nn.Module:
    from safetensors import safe_open

    config = dcae_source.dc_ae_v_f32t4c32_config()
    with torch.device("meta"):
        encoder = dcae_source.DCAEVEncoder(config.encoder)
    expected_state = encoder.state_dict()
    expected_shapes = {name: tuple(value.shape) for name, value in expected_state.items()}
    expected = set(expected_shapes)
    state: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(ENCODER_SUBSET_PATH), framework="pt", device="cpu") as subset:
            original_keys = set(subset.keys())
            expected_keys = {f"encoder.{name}" for name in expected}
            if original_keys != expected_keys:
                raise RealInputError("encoder subset key ownership differs from pinned DCAE encoder")
            for original in sorted(original_keys):
                value = subset.get_tensor(original)
                target_name = original.removeprefix("encoder.")
                if value.dtype != torch.float32 or tuple(value.shape) != expected_shapes[target_name]:
                    raise RealInputError(f"encoder subset tensor shape or dtype mismatch: {original}")
                state[target_name] = value
        incompatible = encoder.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RealInputError("strict DCAE encoder state assignment failed")
    except RealInputError:
        raise
    except Exception as error:
        raise RealInputError(f"cannot load authenticated DCAE encoder subset: {error}") from error
    del state
    encoder.eval().requires_grad_(False)
    try:
        encoder.to(device=device, dtype=torch.bfloat16)
    except Exception as error:
        raise RealInputError(f"cannot move DCAE encoder to MPS BF16: {error}") from error
    return encoder


class _PixelUMMInputStem(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.embed = nn.Linear(source.TUBE_DIM, source.QWEN3_8B_GEN.hidden_size)
        self.cond_proj = nn.Linear(
            source.LATENT_CHANNELS,
            source.LATENT_TOKEN_GRID**2 * source.QWEN3_8B_GEN.hidden_size,
        )


def _load_stem(pixel_source, device: torch.device) -> nn.Module:
    from safetensors import safe_open

    with torch.device("meta"):
        stem = _PixelUMMInputStem(pixel_source)
    expected_state = stem.state_dict()
    expected_shapes = {name: tuple(value.shape) for name, value in expected_state.items()}
    expected = set(expected_shapes)
    state: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(STEM_SUBSET_PATH), framework="pt", device="cpu") as subset:
            original_keys = set(subset.keys())
            expected_keys = {"decoder." + name for name in expected}
            if original_keys != expected_keys or tuple(sorted(original_keys)) != tuple(sorted(STEM_RAW_KEYS)):
                raise RealInputError("stem subset key ownership differs from pinned PixelUMM stem")
            for original in sorted(original_keys):
                value = subset.get_tensor(original)
                target_name = original.removeprefix("decoder.")
                if value.dtype != torch.float32 or tuple(value.shape) != expected_shapes[target_name]:
                    raise RealInputError(f"stem subset tensor shape or dtype mismatch: {original}")
                state[target_name] = value
        incompatible = stem.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RealInputError("strict PixelUMM input-stem state assignment failed")
    except RealInputError:
        raise
    except Exception as error:
        raise RealInputError(f"cannot load authenticated PixelUMM stem subset: {error}") from error
    del state
    stem.eval().requires_grad_(False)
    try:
        stem.to(device=device, dtype=torch.bfloat16)
    except Exception as error:
        raise RealInputError(f"cannot move PixelUMM input stem to MPS BF16: {error}") from error
    return stem


def _image_to_mps_video(image, profile: Profile, device: torch.device) -> torch.Tensor:
    raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    raw = raw.reshape(profile.height, profile.width, 3).permute(2, 0, 1).unsqueeze(0)
    frame = raw.to(device=device, dtype=torch.float32)
    frame = frame.mul(2.0 / 255.0).sub(1.0).to(dtype=torch.bfloat16)
    return frame.unsqueeze(2).expand(1, 3, profile.frames, profile.height, profile.width).contiguous()


def _tensor_raw_chunks(tensor: torch.Tensor, name: str) -> Iterable[memoryview]:
    """Read back, validate, and yield one already-synchronized tensor's bytes."""
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    # Callers perform one terminal MPS synchronization before serialization.
    # This finite check runs only on the CPU copy used for the payload.
    if not torch.isfinite(cpu_tensor).all().item():
        raise RealInputError(f"{name} tensor contains non-finite values")
    raw = cpu_tensor.view(torch.uint8)
    yield memoryview(raw.numpy()).cast("B")


def _atomic_write_artifact(
    manifest_path: Path, payload_path: Path, *,
    tensor_records: Sequence[tuple[str, str, Sequence[int], Iterable[memoryview]]],
    manifest_factory: Callable[[dict, str, int], Mapping],
) -> dict:
    """Write a complete payload and manifest without replacing existing files."""
    for path in (manifest_path, payload_path):
        if path.exists() or path.is_symlink():
            raise RealInputError(f"refusing to replace existing artifact: {path.name}")
    source_dir = manifest_path.parent
    if source_dir.resolve() != SIXTEEN_SOURCE_DIR or payload_path.parent.resolve() != SIXTEEN_SOURCE_DIR:
        raise RealInputError("real-input artifacts must be direct children of the pinned source directory")
    payload_tmp = None
    manifest_tmp = None
    published_payload = False
    payload_identity = None
    succeeded = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{payload_path.name}.", suffix=".partial",
            dir=source_dir, delete=False,
        ) as stream:
            payload_tmp = Path(stream.name)
            aggregate = hashlib.sha256()
            records: dict[str, dict] = {}
            offset = 0
            for name, dtype, shape, chunks in tensor_records:
                if name in records:
                    raise RealInputError("duplicate tensor name in real-input payload")
                record_hash = hashlib.sha256()
                written = 0
                for chunk in chunks:
                    view = memoryview(chunk).cast("B")
                    stream.write(view)
                    record_hash.update(view)
                    aggregate.update(view)
                    written += view.nbytes
                expected = expected_tensor_bytes(dtype, shape)
                if written != expected:
                    raise RealInputError(f"{name} payload bytes differ from its declared shape")
                records[name] = {
                    "dtype": dtype, "shape": list(shape), "offset": offset,
                    "nbytes": written, "sha256": record_hash.hexdigest(),
                }
                offset += written
            stream.flush()
            os.fsync(stream.fileno())
        payload_sha = aggregate.hexdigest()
        manifest = dict(manifest_factory(records, payload_sha, offset))
        manifest.update({
            "payload": payload_path.name,
            "payload_bytes": offset,
            "payload_sha256": payload_sha,
            "inputs": records,
        })
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            prefix=f".{manifest_path.name}.", suffix=".partial",
            dir=source_dir, delete=False,
        ) as stream:
            manifest_tmp = Path(stream.name)
            json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        payload_identity = (payload_tmp.stat().st_dev, payload_tmp.stat().st_ino)
        os.link(payload_tmp, payload_path)
        published_payload = True
        payload_tmp.unlink()
        payload_tmp = None
        os.link(manifest_tmp, manifest_path)
        manifest_tmp.unlink()
        manifest_tmp = None
        succeeded = True
        return manifest
    except FileExistsError as error:
        raise RealInputError("an artifact appeared during publication; no existing file was replaced") from error
    finally:
        for path in (payload_tmp, manifest_tmp):
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        if published_payload and payload_identity is not None and not succeeded:
            try:
                actual = payload_path.stat()
                if (actual.st_dev, actual.st_ino) == payload_identity:
                    payload_path.unlink()
            except FileNotFoundError:
                pass


def _lineage(
    profile: Profile, seed: int, sources: Mapping[str, str], subsets: Mapping[str, object],
    image_sha256: str, permutation_sha256: str,
) -> dict:
    return {
        "source_image_sha256": image_sha256,
        "source_image_width": 1280,
        "source_image_height": 720,
        "source_image_color_order": "RGB",
        "source_image_crop_xyxy": list(profile.crop),
        "source_image_resize": False,
        "clip_frames": profile.frames,
        "clip_construction": "repeat_static_image",
        "vae_input_range": [-1.0, 1.0],
        "upstream_revision": UPSTREAM_REVISION,
        "dcae_encoder_source_sha256": sources["dcae_encoder_source_sha256"],
        "pixelumm_model_source_sha256": sources["pixelumm_model_source_sha256"],
        "pixelumm_noise_source_sha256": sources["pixelumm_noise_source_sha256"],
        "upstream_model_sha256": sources["pixelumm_model_source_sha256"],
        "num_pad_frames": 0,
        "encode_temporal_tile_size": 40,
        "encode_temporal_tile_latent_size": 10,
        "encode_temporal_tile_overlap_factor": 0.0,
        "latent_shape": list(profile.latent_shape),
        "latent_dtype": "BF16",
        "latent_device": "mps",
        "xt_shape": list(profile.xt_shape),
        "xt_dtype": "F32",
        "xt_device": "mps",
        "xt_seed": seed,
        "xt_frame_seed_formula": "seed*1000003+frame_index",
        "xt_noise_source": (
            "hymm.models.vae.pixelumm_decoder.autoencoder."
            "PixelUMMDCAELatentDecoder._make_noise"
        ),
        "encoder_subset_sha256": subsets["encoder_subset_sha256"],
        "encoder_receipt_sha256": subsets["encoder_receipt_sha256"],
        "encoder_subset_bytes": subsets["encoder_subset_bytes"],
        "stem_subset_sha256": subsets["stem_subset_sha256"],
        "stem_receipt_sha256": subsets["stem_receipt_sha256"],
        "stem_subset_bytes": subsets["stem_subset_bytes"],
        "stem_weights_sha256": subsets["stem_weights_sha256"],
        "stem_weights_hash_definition": (
            "SHA256 of the contiguous raw F32 safetensors payload for the four stem tensors "
            "in original checkpoint data-offset order; the safetensors header is excluded"
        ),
        "x_shape": list(profile.x_shape),
        "x_dtype": "BF16",
        "x_token_order": "tile_major",
        "x_permutation_sha256": permutation_sha256,
        "x_stem_expression": "embed(patchify(xt.to(dtype))) + _condition_tokens(latent.to(dtype))",
    }


def _output_paths(profile: Profile, seed: int) -> tuple[Path, Path]:
    name = manifest_basename(profile.name, seed)
    payload_name = payload_basename(name)
    if SIXTEEN_SOURCE_DIR.is_symlink() or not SIXTEEN_SOURCE_DIR.is_dir():
        raise RealInputError("pinned sixteen-block source directory is missing or is a symlink")
    manifest_path = SIXTEEN_SOURCE_DIR / name
    payload_path = SIXTEEN_SOURCE_DIR / payload_name
    if manifest_path.parent.resolve() != SIXTEEN_SOURCE_DIR or payload_path.parent.resolve() != SIXTEEN_SOURCE_DIR:
        raise RealInputError("real-input output paths escape the pinned source directory")
    return manifest_path, payload_path


def produce(*, profile_name: str = "full", seed: int = 42,
            max_mps_bytes: int | None = None, preflight_only: bool = False,
            capability_probe_only: bool = False) -> dict:
    profile = _profile(profile_name)
    if type(seed) is not int or seed not in (42, 43):
        raise RealInputError("noise seed must be 42 or 43")
    memory = _verify_mps(profile, max_mps_bytes)
    if capability_probe_only:
        _set_mps_memory_cap(memory)
        return {
            "profile": profile.name,
            "grid": list(profile.grid),
            "memory_preflight": memory,
            "capability_probe": _mps_capability_probe(),
            "model_allocated": False,
            "inference_run": False,
        }
    manifest_path, payload_path = _output_paths(profile, seed)
    if manifest_path.exists() or manifest_path.is_symlink() or payload_path.exists() or payload_path.is_symlink():
        raise RealInputError("refusing to replace an existing real-input artifact")

    image, image_sha = _verify_image(profile)
    sources = _verified_upstream_sources(UPSTREAM_ROOT)
    subsets = _validate_subset_inputs()
    summary = {
        "schema": REAL_INPUT_SCHEMA,
        "profile": profile.name,
        "grid": list(profile.grid),
        "latent_shape": list(profile.latent_shape),
        "xt_shape": list(profile.xt_shape),
        "x_shape": list(profile.x_shape),
        "noise_seed": seed,
        "manifest_path": str(manifest_path),
        "payload_path": str(payload_path),
        "payload_bytes": sum((
            expected_tensor_bytes("BF16", profile.latent_shape),
            expected_tensor_bytes("F32", profile.xt_shape),
            expected_tensor_bytes("BF16", profile.x_shape),
        )),
        "source_image_sha256": image_sha,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_sources": sources,
        "subset_provenance": subsets,
        "memory_preflight": memory,
        "inference_device": "mps",
        "cpu_fallback": False,
    }
    if preflight_only:
        summary["preflight_only"] = True
        return summary

    _set_mps_memory_cap(memory)
    capability_probe = _mps_capability_probe()
    device = torch.device("mps")
    dcae_source, pixel_source, noise_source = _load_modules(UPSTREAM_ROOT)
    try:
        from hy_nomas_pixelumm_reference import make_attention_plan
        plan, _, _ = make_attention_plan(pixel_source, profile.grid)
    except Exception as error:
        raise RealInputError(f"cannot derive pinned tile-major token order: {error}") from error

    encoder = _load_encoder(dcae_source, device)
    stem = _load_stem(pixel_source, device)
    video = _image_to_mps_video(image, profile, device)
    try:
        with torch.inference_mode():
            latent = encoder(video)
    except Exception as error:
        raise RealInputError(f"DCAE encoder failed on MPS; CPU fallback is disabled: {error}") from error
    if tuple(latent.shape) != profile.latent_shape or latent.dtype != torch.bfloat16 or latent.device.type != "mps":
        raise RealInputError("DCAE encoder output shape, dtype, or device mismatch")

    try:
        noise_owner = SimpleNamespace(seed=seed)
        xt = noise_source.PixelUMMDCAELatentDecoder._make_noise(
            noise_owner, latent, 0, profile.latent_shape[2]
        )
        if tuple(xt.shape) != profile.xt_shape or xt.dtype != torch.float32 or xt.device.type != "mps":
            raise RealInputError("PixelUMM noise shape, dtype, or device mismatch")
        with torch.inference_mode(), torch.autocast(device_type="mps", dtype=torch.bfloat16):
            patch_tokens = stem.embed(pixel_source.patchify(xt.to(dtype=stem.embed.weight.dtype)))
            context = SimpleNamespace(cond_proj=stem.cond_proj, arch=pixel_source.QWEN3_8B_GEN)
            conditioned = pixel_source.DCAEVPixelUMMDecoder._condition_tokens(
                context, latent.to(dtype=stem.embed.weight.dtype)
            )
            thw_tokens = patch_tokens + conditioned
            permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
            x = thw_tokens.index_select(1, permutation)
    except Exception as error:
        raise RealInputError(f"PixelUMM stem failed on MPS; CPU fallback is disabled: {error}") from error
    if tuple(x.shape) != profile.x_shape or x.dtype != torch.bfloat16 or x.device.type != "mps":
        raise RealInputError("PixelUMM stem output shape, dtype, or device mismatch")

    # The sole execution/readback boundary for all neural work in this producer.
    try:
        torch.mps.synchronize()
    except Exception as error:
        raise RealInputError(f"terminal MPS synchronization failed: {error}") from error

    lineage = _lineage(profile, seed, sources, subsets, image_sha, plan.permutation_sha256)
    provenance_base = {
        "kind": "real_latent_xt_pixelumm_stem",
        "stem_weights_sha256": subsets["stem_weights_sha256"],
    }
    records_to_write = (
        ("latent", "BF16", profile.latent_shape, _tensor_raw_chunks(latent, "latent")),
        ("xt", "F32", profile.xt_shape, _tensor_raw_chunks(xt, "xt")),
        ("x", "BF16", profile.x_shape, _tensor_raw_chunks(x, "x")),
    )

    def make_manifest(records: dict, aggregate_sha: str, payload_bytes: int) -> Mapping:
        provenance = {
            **provenance_base,
            "latent_sha256": records["latent"]["sha256"],
            "xt_sha256": records["xt"]["sha256"],
            "derived_x_sha256": records["x"]["sha256"],
        }
        return {
            **_manifest_profile_metadata(profile, seed),
            "input_provenance": provenance,
            "input_lineage": lineage,
            "producer": {
                "tool": "hy_nomas_pixelumm_real_input.py",
                "source_sha256": _sha256_file(Path(__file__).resolve()),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "inference_device": "mps",
                "terminal_mps_synchronize_before_readback": True,
                "cpu_fallback": False,
                "mps_capability_probe": capability_probe,
            },
        }

    result = _atomic_write_artifact(
        manifest_path, payload_path, tensor_records=records_to_write,
        manifest_factory=make_manifest,
    )
    summary["preflight_only"] = False
    summary["capability_probe"] = capability_probe
    summary["payload_sha256"] = result["payload_sha256"]
    summary["input_provenance"] = result["input_provenance"]
    del video, encoder, stem, latent, xt, x, patch_tokens, conditioned, thw_tokens
    return summary


def _parse_bytes(raw: str) -> int:
    try:
        return int(raw, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max MPS bytes must be a positive integer") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--noise-seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--max-mps-bytes", type=_parse_bytes, default=None,
                        help="hard per-process MPS allocator cap (defaults to 24 GiB full, 8 GiB small)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate all pinned sources/receipts and report budgets without inference")
    parser.add_argument("--capability-probe-only", action="store_true",
                        help="cap MPS and test BF16 Conv3d without loading any model weights")
    args = parser.parse_args(argv)
    try:
        result = produce(
            profile_name=args.profile, seed=args.noise_seed,
            max_mps_bytes=args.max_mps_bytes, preflight_only=args.preflight_only,
            capability_probe_only=args.capability_probe_only,
        )
    except Exception as error:
        print(f"real-input producer failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
