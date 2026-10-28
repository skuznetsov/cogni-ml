#!/usr/bin/env python3
"""Bounded pretrained PixelUMM decoder references with local attention.

The helper imports pinned upstream GenLayer/RoPE/layout code, loads only the
authenticated subset for the requested decoder block or consecutive prefix,
and replaces FlexAttention with per-spatial-tile gathered QK/softmax/PV. Legacy
fixtures use controlled synthetic BF16 hidden states. The sixteen-block
reference requires a provenance-pinned real pose image, VAE latent,
scheduler-noise xt, and upstream PixelUMM stem result. The full target exports
all per-block attention boundaries and only the final two output boundaries.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from contextlib import nullcontext
import hashlib
import importlib
import json
import math
import mmap
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable, Mapping

import torch

import hy_nomas_pixelumm_checkpoint as checkpoint_inventory

UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_MODEL_PATH = "hymm/models/vae/pixelumm_decoder/model.py"
CHECKPOINT_SUBSET_SHA256 = "ae82bb41c121a955493f5b1502dc512ac8aaba05a6cec67e98edeaedfe720485"
CHECKPOINT_SUBSET_BYTES = 771_787_128
CHECKPOINT_PREFIX = "decoder.layers.0."
MPS_ORACLE_TORCH_VERSION = "2.6.0"
SCHEMA = "cogni_ml.pixelumm_block_reference/v1"
TWO_BLOCK_SCHEMA = "cogni_ml.pixelumm_two_block_reference/v1"
FOUR_BLOCK_SCHEMA = "cogni_ml.pixelumm_four_block_reference/v1"
EIGHT_BLOCK_SCHEMA = "cogni_ml.pixelumm_eight_block_reference/v1"
SIXTEEN_BLOCK_SCHEMA = "cogni_ml.pixelumm_sixteen_block_reference/v1"
THIRTY_SIX_BLOCK_SCHEMA = "cogni_ml.pixelumm_thirty_six_block_reference/v1"
THIRTY_SIX_CAPTURE_POLICY = "two_bank_survivors"
SIXTEEN_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261027-sixteen")
SIXTEEN_SOURCE_DIR = (SIXTEEN_ROOT / "source").resolve()
SIXTEEN_RESULTS_DIR = (SIXTEEN_ROOT / "results").resolve()
THIRTY_SIX_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-36-20261027")
THIRTY_SIX_SOURCE_DIR = (THIRTY_SIX_ROOT / "source").resolve()
THIRTY_SIX_RESULTS_DIR = (THIRTY_SIX_ROOT / "results").resolve()
REAL_INPUT_SCHEMA = "cogni_ml.pixelumm_real_input/v1"
REAL_INPUT_MANIFEST_NAME = "pixelumm-real-input.json"
REAL_INPUT_SOURCE_IMAGE = Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/inputs/pose.png")
REAL_INPUT_SOURCE_IMAGE_SHA256 = "53a5d9b435adaf15dd8ffcab1a833b61da4a63079200fb9cec33127ee10f733b"
REAL_INPUT_DCAE_SOURCE_SHA256 = "38163ebead9a6bb643d27843022c1757c45c6b8638c334a476aee5b373edf8c1"
REAL_INPUT_PIXELUMM_SOURCE_SHA256 = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
REAL_INPUT_NOISE_SOURCE_SHA256 = "65e0d77b9f2e9a3ca457d6a51bfdb239f7b9dc3fc8a0ad014025c36d6be9f0cf"
REAL_INPUT_NOISE_OWNER = (
    "hymm.models.vae.pixelumm_decoder.autoencoder.PixelUMMDCAELatentDecoder._make_noise"
)
SIXTEEN_GRID = (6, 44, 80)
SIXTEEN_TOKENS = math.prod(SIXTEEN_GRID)
SMALL_REAL_INPUT_GRID = (2, 32, 32)
REAL_INPUT_ARTIFACTS = {
    "pixelumm-real-input.json": ("full", 42, "pixelumm-real-input.bin"),
    **{
        f"pixelumm-real-input-{profile}-seed{seed}.json": (
            profile, seed, f"pixelumm-real-input-{profile}-seed{seed}.bin",
        )
        for profile in ("small", "full") for seed in (42, 43)
    },
}
SIXTEEN_FIXTURE_PAYLOAD_CAP = 16 * 1024 * 1024 * 1024
THIRTY_SIX_FIXTURE_PAYLOAD_CAP = 32 * 1024 * 1024 * 1024
TILE_TOKENS = 8
ATOL = RTOL = 2e-5

ALL_STAGES = (
    "input_norm_statistics", "input_norm_pre_gain", "input_norm",
    "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "q_rope", "k_rope",
    "q_norm_statistics", "k_norm_statistics", "q_norm_inverse", "k_norm_inverse",
    "q_ready", "k_ready",
    "attention", "o_proj", "attn_residual",
    "post_attn_norm", "gate_proj", "up_proj", "silu_gate", "product",
    "down_proj", "output",
)
FULL_TARGET_STAGES = ("attention", "output")


class ReferenceError(ValueError):
    """The requested source, checkpoint, geometry, or fixture is unsupported."""


@dataclass(frozen=True)
class TileRegion:
    tile_h: int
    tile_w: int
    query_start: int
    query_stop: int
    key_positions: tuple[int, ...]


@dataclass(frozen=True)
class AttentionPlan:
    grid: tuple[int, int, int]
    tile: int
    permutation: tuple[int, ...]
    regions: tuple[TileRegion, ...]
    permutation_sha256: str


def _check_grid(grid: Iterable[int], tile: int = TILE_TOKENS) -> tuple[int, int, int]:
    values = tuple(grid)
    if len(values) != 3 or any(type(value) is not int or value <= 0 for value in values):
        raise ReferenceError("grid must contain three positive integers [T,H,W]")
    if type(tile) is not int or tile <= 0:
        raise ReferenceError("tile size must be a positive integer")
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _admitted_real_input_manifest_path(manifest_path: Path | str) -> Path:
    """Require an admitted manifest's canonical path under the pinned source dir."""
    original = Path(manifest_path).expanduser()
    if (not original.is_absolute() or original.parent != SIXTEEN_SOURCE_DIR
            or original.name not in REAL_INPUT_ARTIFACTS or original.is_symlink()):
        raise ReferenceError("real-input manifest must be a regular non-symlink file")
    try:
        canonical = original.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReferenceError("real-input manifest must be a regular non-symlink file") from error
    if (canonical != original or canonical.parent != SIXTEEN_SOURCE_DIR
            or not canonical.is_file()):
        raise ReferenceError("real-input manifest must use its canonical non-symlink path")
    return canonical


def _bf16_bytes_are_finite(raw: bytes) -> bool:
    """Check one little-endian BF16 payload chunk on CPU without device transfer."""
    if len(raw) % 2:
        return False
    if not raw:
        return True
    values = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
    return bool(torch.isfinite(values).all())


def _validate_real_input_subset_sizes(lineage: Mapping, checked_subsets: Mapping) -> None:
    """Bind producer byte counts to authenticated full-file and tensor-payload extents."""
    for kind in ("encoder", "stem"):
        checked = checked_subsets.get(kind)
        if not isinstance(checked, Mapping):
            raise ReferenceError(f"authenticated {kind} subset receipt is missing")
        subset_bytes = checked.get("subset_bytes")
        source_f32_bytes = checked.get("source_f32_bytes")
        lineage_bytes = lineage.get(f"{kind}_subset_bytes")
        # `subset_bytes` includes the safetensors header; `source_f32_bytes`
        # counts only the selected raw tensor payload authenticated by the receipt.
        if (type(subset_bytes) is not int or type(source_f32_bytes) is not int
                or source_f32_bytes <= 0 or subset_bytes <= source_f32_bytes
                or type(lineage_bytes) is not int or lineage_bytes != subset_bytes):
            raise ReferenceError(f"real-input {kind} complete subset byte count mismatch")


def _real_input_source_subsets(lineage: Mapping) -> dict[str, object]:
    """Revalidate the fixed encoder/stem subset receipts behind a real input."""
    source_dir = (checkpoint_inventory.ROOT / "source").resolve()
    header_path = source_dir / "pixelumm-header.json"
    if header_path.is_symlink() or not header_path.is_file():
        raise ReferenceError("pinned PixelUMM header is missing or is a symlink")
    raw_header = header_path.read_bytes()
    checkpoint_inventory.parse_pinned_header(raw_header)
    checked_subsets = {}
    for kind in ("encoder", "stem"):
        subset_path = source_dir / f"{kind}-f32.safetensors"
        receipt_path = source_dir / f"{kind}-f32.receipt.json"
        if (subset_path.is_symlink() or receipt_path.is_symlink()
                or not subset_path.is_file() or not receipt_path.is_file()):
            raise ReferenceError(f"pinned {kind} subset or receipt is missing")
        receipt_raw = receipt_path.read_bytes()
        if len(receipt_raw) > 2 * 1024 * 1024:
            raise ReferenceError(f"pinned {kind} receipt exceeds metadata limit")
        if hashlib.sha256(receipt_raw).hexdigest() != lineage[f"{kind}_receipt_sha256"]:
            raise ReferenceError(f"pinned {kind} receipt digest changed")
        try:
            receipt = checkpoint_inventory.parse_header(receipt_raw)
            checked = checkpoint_inventory.validate_named_subset_receipt(
                raw_header, receipt, subset_path, kind
            )
        except checkpoint_inventory.InventoryError as error:
            raise ReferenceError(f"pinned {kind} subset receipt rejected: {error}") from error
        if checked["subset_sha256"] != lineage[f"{kind}_subset_sha256"]:
            raise ReferenceError(f"pinned {kind} subset digest changed")
        if checked["subset_bytes"] != lineage[f"{kind}_subset_bytes"]:
            raise ReferenceError(f"pinned {kind} subset byte count changed")
        checked_subsets[kind] = checked

    _validate_real_input_subset_sizes(lineage, checked_subsets)

    # The stem provenance digest is the raw F32 tensor payload, excluding its
    # safetensors header. The receipt separately authenticates the full file.
    stem_path = source_dir / "stem-f32.safetensors"
    with stem_path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ReferenceError("stem subset header is truncated")
        header_size = int.from_bytes(prefix, "little")
        if header_size <= 0 or header_size > 2 * 1024 * 1024:
            raise ReferenceError("stem subset header size is invalid")
        if len(stream.read(header_size)) != header_size:
            raise ReferenceError("stem subset header is truncated")
        digest = hashlib.sha256()
        remaining = 52_510_720
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ReferenceError("stem subset payload is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ReferenceError("stem subset has trailing data")
    if digest.hexdigest() != lineage["stem_weights_sha256"]:
        raise ReferenceError("stem source-weight digest changed")
    checked_subsets["stem_weights_sha256"] = digest.hexdigest()
    return checked_subsets


def validate_real_input_manifest(manifest_path: Path | str) -> dict:
    """Validate one admitted real-pose latent/xt/x artifact without loading a model."""
    path = _admitted_real_input_manifest_path(manifest_path)
    profile, expected_seed, expected_payload_name = REAL_INPUT_ARTIFACTS[path.name]
    raw_manifest = path.read_bytes()
    if len(raw_manifest) > 2 * 1024 * 1024:
        raise ReferenceError("real-input manifest exceeds the metadata limit")
    try:
        manifest = checkpoint_inventory.parse_header(raw_manifest)
    except checkpoint_inventory.InventoryError as error:
        raise ReferenceError(f"cannot parse real-input manifest: {error}") from error
    if manifest.get("schema") != REAL_INPUT_SCHEMA:
        raise ReferenceError("unsupported real-input schema")
    profile_shapes = {
        "full": {
            "grid": [6, 44, 80], "frames": 24,
            "latent_shape": [1, 32, 6, 22, 40],
            "xt_shape": [1, 3, 24, 704, 1280],
        },
        "small": {
            "grid": [2, 32, 32], "frames": 8,
            "latent_shape": [1, 32, 2, 16, 16],
            "xt_shape": [1, 3, 8, 512, 512],
        },
    }
    profile_spec = profile_shapes[profile]
    expected_x_shape = [1, math.prod(profile_spec["grid"]), 4096]
    profile_arrays = {
        "grid": profile_spec["grid"],
        "latent_shape": profile_spec["latent_shape"],
        "xt_shape": profile_spec["xt_shape"],
        "x_shape": expected_x_shape,
    }
    if (manifest.get("profile") != profile or type(manifest.get("noise_seed")) is not int
            or manifest.get("noise_seed") != expected_seed
            or any(not isinstance(manifest.get(key), list)
                   or any(type(value) is not int for value in manifest[key])
                   or manifest[key] != expected
                   for key, expected in profile_arrays.items())
            or manifest.get("token_order") != "tile_major"):
        raise ReferenceError("real-input profile geometry or token ordering mismatch")
    payload_name = manifest.get("payload")
    if payload_name != expected_payload_name or Path(payload_name).name != payload_name:
        raise ReferenceError("real-input payload must use its pinned basename")
    payload_path = path.parent / payload_name
    if payload_path.is_symlink() or not payload_path.is_file():
        raise ReferenceError("real-input payload is missing or is a symlink")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"latent", "xt", "x"}:
        raise ReferenceError("real-input manifest must contain exactly latent, xt, and x")
    expected = {
        "latent": ("BF16", profile_spec["latent_shape"]),
        "xt": ("F32", profile_spec["xt_shape"]),
        "x": ("BF16", expected_x_shape),
    }
    cursor = 0
    records = []
    for name in ("latent", "xt", "x"):
        record = inputs[name]
        if not isinstance(record, dict) or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}:
            raise ReferenceError(f"real-input {name} record is malformed")
        dtype_name, shape = expected[name]
        width = 2 if dtype_name == "BF16" else 4
        nbytes = math.prod(shape) * width
        actual_shape = record.get("shape")
        if (record.get("dtype") != dtype_name or not isinstance(actual_shape, list)
                or any(type(value) is not int for value in actual_shape)
                or actual_shape != shape
                or type(record.get("offset")) is not int or record["offset"] != cursor
                or type(record.get("nbytes")) is not int or record["nbytes"] != nbytes
                or not _digest(record.get("sha256"))):
            raise ReferenceError(f"real-input {name} shape, dtype, extent, or digest mismatch")
        records.append((name, cursor, nbytes, record["sha256"]))
        cursor += nbytes
    if (type(manifest.get("payload_bytes")) is not int
            or manifest["payload_bytes"] != cursor
            or payload_path.stat().st_size != cursor
            or not _digest(manifest.get("payload_sha256"))):
        raise ReferenceError("real-input payload extent or aggregate digest is malformed")

    provenance = manifest.get("input_provenance")
    provenance_fields = {
        "kind", "latent_sha256", "xt_sha256", "stem_weights_sha256", "derived_x_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != provenance_fields:
        raise ReferenceError("real-input provenance fields are malformed")
    if provenance.get("kind") != "real_latent_xt_pixelumm_stem":
        raise ReferenceError("real-input provenance kind mismatch")
    for key in provenance_fields - {"kind"}:
        if not _digest(provenance.get(key)):
            raise ReferenceError(f"real-input provenance digest is malformed: {key}")
    if (provenance["latent_sha256"] != inputs["latent"]["sha256"]
            or provenance["xt_sha256"] != inputs["xt"]["sha256"]
            or provenance["derived_x_sha256"] != inputs["x"]["sha256"]):
        raise ReferenceError("real-input provenance digests disagree with tensor records")

    lineage = manifest.get("input_lineage")
    if not isinstance(lineage, dict):
        raise ReferenceError("real-input lineage is missing")
    expected_lineage = {
        "source_image_sha256": REAL_INPUT_SOURCE_IMAGE_SHA256,
        "source_image_width": 1280,
        "source_image_height": 720,
        "source_image_color_order": "RGB",
        "source_image_crop_xyxy": ([0, 8, 1280, 712] if profile == "full"
                                    else [384, 104, 896, 616]),
        "source_image_resize": False,
        "clip_frames": profile_spec["frames"],
        "clip_construction": "repeat_static_image",
        "vae_input_range": [-1.0, 1.0],
        "upstream_revision": UPSTREAM_REVISION,
        "dcae_encoder_source_sha256": REAL_INPUT_DCAE_SOURCE_SHA256,
        "pixelumm_model_source_sha256": REAL_INPUT_PIXELUMM_SOURCE_SHA256,
        "pixelumm_noise_source_sha256": REAL_INPUT_NOISE_SOURCE_SHA256,
        "upstream_model_sha256": REAL_INPUT_PIXELUMM_SOURCE_SHA256,
        "num_pad_frames": 0,
        "encode_temporal_tile_size": 40,
        "encode_temporal_tile_latent_size": 10,
        "encode_temporal_tile_overlap_factor": 0.0,
        "latent_shape": profile_spec["latent_shape"],
        "latent_dtype": "BF16",
        "latent_device": "mps",
        "xt_shape": profile_spec["xt_shape"],
        "xt_dtype": "F32",
        "xt_device": "mps",
        "xt_seed": expected_seed,
        "xt_frame_seed_formula": "seed*1000003+frame_index",
        "xt_noise_source": REAL_INPUT_NOISE_OWNER,
        "stem_weights_sha256": provenance["stem_weights_sha256"],
        "stem_weights_hash_definition": (
            "SHA256 of the contiguous raw F32 safetensors payload for the four stem tensors "
            "in original checkpoint data-offset order; the safetensors header is excluded"
        ),
        "x_shape": expected_x_shape,
        "x_dtype": "BF16",
        "x_token_order": "tile_major",
        "x_stem_expression": "embed(patchify(xt.to(dtype))) + _condition_tokens(latent.to(dtype))",
    }
    lineage_fields = set(expected_lineage) | {
        "encoder_subset_sha256", "encoder_receipt_sha256", "encoder_subset_bytes",
        "stem_subset_sha256", "stem_receipt_sha256", "stem_subset_bytes",
        "x_permutation_sha256",
    }
    if set(lineage) != lineage_fields:
        raise ReferenceError("real-input lineage must contain only the pinned provenance fields")
    for key, value in expected_lineage.items():
        if lineage.get(key) != value:
            raise ReferenceError(f"real-input lineage mismatch: {key}")
    int_fields = (
        "clip_frames", "num_pad_frames", "encode_temporal_tile_size",
        "encode_temporal_tile_latent_size", "xt_seed",
    )
    shape_fields = ("latent_shape", "xt_shape", "x_shape")
    if (any(type(lineage.get(key)) is not int for key in int_fields)
            or any(not isinstance(lineage.get(key), list)
                   or any(type(value) is not int for value in lineage[key])
                   for key in shape_fields)
            or type(lineage.get("source_image_resize")) is not bool
            or type(lineage.get("encode_temporal_tile_overlap_factor")) is not float
            or not isinstance(lineage.get("vae_input_range"), list)
            or any(type(value) is not float for value in lineage["vae_input_range"])
            or not isinstance(lineage.get("source_image_crop_xyxy"), list)
            or any(type(value) is not int for value in lineage["source_image_crop_xyxy"])):
        raise ReferenceError("real-input lineage contains malformed profile scalar types")
    for key in ("encoder_subset_sha256", "encoder_receipt_sha256",
                "stem_subset_sha256", "stem_receipt_sha256"):
        if not _digest(lineage.get(key)):
            raise ReferenceError(f"real-input lineage digest is malformed: {key}")
    if not _digest(lineage.get("x_permutation_sha256")):
        raise ReferenceError("real-input tile permutation digest is malformed")
    if (type(lineage.get("encoder_subset_bytes")) is not int
            or type(lineage.get("stem_subset_bytes")) is not int):
        raise ReferenceError("real-input encoder/stem subset byte count is malformed")
    for key in ("source_image_width", "source_image_height"):
        if type(lineage.get(key)) is not int:
            raise ReferenceError(f"real-input lineage scalar type mismatch: {key}")

    producer = manifest.get("producer")
    producer_script = Path(__file__).with_name("hy_nomas_pixelumm_real_input.py")
    if (producer_script.is_symlink() or not producer_script.is_file()
            or not isinstance(producer, dict)
            or producer.get("tool") != "hy_nomas_pixelumm_real_input.py"
            or producer.get("source_sha256") != _sha256_file(producer_script)
            or not isinstance(producer.get("python_version"), str)
            or not producer["python_version"]
            or not isinstance(producer.get("torch_version"), str)
            or not producer["torch_version"]
            or producer.get("inference_device") != "mps"
            or producer.get("terminal_mps_synchronize_before_readback") is not True
            or producer.get("cpu_fallback") is not False):
        raise ReferenceError("real-input producer receipt does not bind the pinned MPS helper")
    if (REAL_INPUT_SOURCE_IMAGE.is_symlink() or not REAL_INPUT_SOURCE_IMAGE.is_file()
            or _sha256_file(REAL_INPUT_SOURCE_IMAGE) != REAL_INPUT_SOURCE_IMAGE_SHA256):
        raise ReferenceError("official real-input source image is missing or changed")

    before = payload_path.stat()
    aggregate = hashlib.sha256()
    with payload_path.open("rb") as stream:
        for name, offset, nbytes, expected_sha in records:
            digest = hashlib.sha256()
            stream.seek(offset)
            remaining = nbytes
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ReferenceError(f"real-input {name} payload is truncated")
                if name == "x" and not _bf16_bytes_are_finite(chunk):
                    raise ReferenceError("real-input x contains a non-finite BF16 value")
                digest.update(chunk)
                aggregate.update(chunk)
                remaining -= len(chunk)
            actual = digest.hexdigest()
            if actual != expected_sha:
                raise ReferenceError(f"real-input {name} payload digest mismatch")
    after = payload_path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ReferenceError("real-input payload changed during validation")
    aggregate_sha = aggregate.hexdigest()
    if aggregate_sha != manifest["payload_sha256"]:
        raise ReferenceError("real-input aggregate payload digest mismatch")
    _real_input_source_subsets(lineage)

    return {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "payload_path": str(payload_path),
        "payload_bytes": cursor,
        "payload_sha256": aggregate_sha,
        "grid": profile_spec["grid"],
        "profile": profile,
        "seed": expected_seed,
        "x_permutation_sha256": lineage.get("x_permutation_sha256"),
        "inputs": inputs,
        "input_provenance": provenance,
        "input_lineage": lineage,
    }


def _load_real_input_x(validated: Mapping, device: torch.device) -> torch.Tensor:
    """Map the validated pair payload and copy only its x tensor to the device."""
    payload_path = Path(validated["payload_path"])
    record = validated["inputs"]["x"]
    with payload_path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    try:
        byte_tensor = torch.frombuffer(mapped, dtype=torch.uint8, count=validated["payload_bytes"])
        raw_x = byte_tensor.narrow(0, record["offset"], record["nbytes"])
        x_cpu = raw_x.view(torch.bfloat16).view(record["shape"])
        x = x_cpu.to(device=device)
        del x_cpu, raw_x, byte_tensor
        return x
    finally:
        mapped.close()


def _upstream_permutation_formula(
    grid: tuple[int, int, int], tile: int,
) -> torch.Tensor:
    """The pinned helper's sorting formula, without constructing its Flex mask.

    At the 21120-token target, constructing a diagnostic Flex BlockMask may
    evaluate a quadratic mask domain. The formula below is copied from the
    pinned source's ``local_attention_layout`` permutation setup; small grids
    are checked directly against that helper in ``make_attention_plan``.
    """
    frames, height, width = grid
    tiles_w = (width + tile - 1) // tile
    t, h, w = torch.meshgrid(
        *(torch.arange(n, device="cpu") for n in grid), indexing="ij"
    )
    tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
    order = (((tile_h * tiles_w + tile_w) * frames + t.flatten()) * tile + (h % tile).flatten()) * tile
    return torch.argsort(order + (w % tile).flatten())


def load_upstream(upstream: Path | str, *, verify_revision: bool = True):
    """Import the pinned source module without constructing its full 36-layer model."""
    root = Path(upstream).expanduser().resolve()
    source_file = root / UPSTREAM_MODEL_PATH
    if not source_file.is_file():
        raise ReferenceError(f"missing upstream source file: {source_file}")
    if verify_revision:
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            changed = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--", UPSTREAM_MODEL_PATH],
                text=True, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ReferenceError("cannot verify pinned upstream git source") from error
        if revision != UPSTREAM_REVISION or changed.strip():
            raise ReferenceError("upstream revision or model.py worktree identity mismatch")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        module = importlib.import_module("hymm.models.vae.pixelumm_decoder.model")
    except Exception as error:  # preserve a compact CLI error at the outer boundary
        raise ReferenceError(f"cannot import pinned PixelUMM model module: {error}") from error
    imported_path = Path(module.__file__).resolve()
    if imported_path != source_file.resolve():
        raise ReferenceError("Python imported a different PixelUMM model.py")
    return module, _sha256_file(source_file)


def make_attention_plan(source, grid: Iterable[int], tile: int = TILE_TOKENS) -> tuple[AttentionPlan, torch.Tensor, object]:
    """Use upstream local_attention_layout for permutation and mask construction.

    Returns (plan, inverse, source_block_mask). Grids whose source semantics are
    global attention are rejected so this adapter can never accidentally create
    a dense target score matrix.
    """
    grid = _check_grid(grid, tile)
    _, height, width = grid
    tiles_h = (height + tile - 1) // tile
    tiles_w = (width + tile - 1) // tile
    if tiles_h <= 3 and tiles_w <= 3:
        raise ReferenceError(
            "local adapter requires more than 3 tiles on at least one spatial axis"
        )
    count = math.prod(grid)
    if count <= 2048:
        layout = source.local_attention_layout(grid, tile, "cpu")
        if layout is None:
            raise ReferenceError("upstream helper reported global attention for a local grid")
        perm, inverse, block_mask = layout
        formula_perm = _upstream_permutation_formula(grid, tile)
        if not torch.equal(perm, formula_perm):
            raise ReferenceError("local permutation formula disagrees with pinned source helper")
    else:
        # Never ask eager Flex to build a full-target BlockMask. The native and
        # reference paths consume only these static bounded tile neighborhoods.
        perm = _upstream_permutation_formula(grid, tile)
        inverse = torch.argsort(perm)
        block_mask = None
    if perm.device.type != "cpu" or tuple(perm.shape) != (count,):
        raise ReferenceError("upstream layout returned an unexpected permutation")
    permutation = tuple(int(value) for value in perm.tolist())
    if sorted(permutation) != list(range(count)):
        raise ReferenceError("upstream token permutation is not a bijection")

    tile_coordinates: list[tuple[int, int]] = []
    for original in permutation:
        h = (original // width) % height
        w = original % width
        tile_coordinates.append((h // tile, w // tile))

    regions: list[TileRegion] = []
    cursor = 0
    for tile_h in range(tiles_h):
        for tile_w in range(tiles_w):
            q_positions = [
                position for position, coordinate in enumerate(tile_coordinates)
                if coordinate == (tile_h, tile_w)
            ]
            if not q_positions:
                raise ReferenceError("upstream permutation omitted a spatial tile")
            if q_positions != list(range(q_positions[0], q_positions[-1] + 1)):
                raise ReferenceError("upstream tile-major query positions are not contiguous")
            center_h = min(max(tile_h, 1), max(tiles_h - 2, 1))
            center_w = min(max(tile_w, 1), max(tiles_w - 2, 1))
            key_positions = tuple(
                position for position, (key_h, key_w) in enumerate(tile_coordinates)
                if abs(center_h - key_h) <= 1 and abs(center_w - key_w) <= 1
            )
            regions.append(TileRegion(
                tile_h=tile_h, tile_w=tile_w,
                query_start=q_positions[0], query_stop=q_positions[-1] + 1,
                key_positions=key_positions,
            ))
            cursor += len(q_positions)
    if cursor != count:
        raise ReferenceError("spatial tile plan does not cover every token")
    perm_bytes = torch.tensor(permutation, dtype=torch.int32).numpy().tobytes()
    plan = AttentionPlan(
        grid=grid, tile=tile, permutation=permutation,
        regions=tuple(regions), permutation_sha256=hashlib.sha256(perm_bytes).hexdigest(),
    )
    return plan, inverse, block_mask


def gathered_local_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: AttentionPlan,
) -> torch.Tensor:
    """Per-spatial-tile QK/softmax/PV with physically gathered, ordered KV rows.

    q/k/v use [B,J,heads,D], all in the layer's projected dtype. Keys for each
    tile are selected in the upstream permutation order, including every time
    position; only spatial tile distance is constrained.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ReferenceError("q, k, and v must have shape [B,J,heads,D]")
    batch, count, q_heads, dim = q.shape
    if count != math.prod(plan.grid) or k.shape[:2] != (batch, count) or v.shape != k.shape:
        raise ReferenceError("q/k/v shapes do not match the attention plan")
    if k.shape[-1] != dim or q_heads % k.shape[2] or q.device != k.device or k.device != v.device:
        raise ReferenceError("incompatible attention heads, devices, or dimensions")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ReferenceError("q/k/v projected dtypes must match")
    group = q_heads // k.shape[2]
    output = torch.empty_like(q)
    scale = dim ** -0.5
    device = q.device
    f32_context = (
        torch.autocast(device_type=device.type, enabled=False)
        if device.type in {"cpu", "mps"} else nullcontext()
    )
    device_regions = [
        (
            region,
            torch.arange(region.query_start, region.query_stop, dtype=torch.long, device=device),
            torch.tensor(region.key_positions, dtype=torch.long, device=device),
        )
        for region in plan.regions
    ]
    with f32_context:
        for region, query_idx, key_idx in device_regions:
            # Indices were statically built and copied before the first tile's
            # math; no nonzero/item-based discovery or host readback occurs in
            # the MPS attention loop.
            q_tile = q.index_select(1, query_idx).transpose(1, 2).float()
            k_tile = k.index_select(1, key_idx).repeat_interleave(group, dim=2).transpose(1, 2).float()
            v_tile = v.index_select(1, key_idx).repeat_interleave(group, dim=2).transpose(1, 2).float()
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1))
            scores.mul_(scale)
            probabilities = torch.softmax(scores, dim=-1)
            attended = torch.matmul(probabilities, v_tile).to(dtype=q.dtype)
            output[:, region.query_start:region.query_stop] = attended.transpose(1, 2)
    return output


def _prepare_qk_with_stages(attention, raw, cos, sin, norm_t, norm_hw, dtype):
    normalized = attention._split_norm(raw.float(), norm_t, norm_hw)
    rotated = attention_source_apply_mrope(attention, normalized, cos, sin)
    return rotated.to(dtype), normalized, rotated


def _qk_norm_diagnostics(raw: torch.Tensor, norm_t, norm_hw):
    """Report source RMSNorm statistics and inverse roots without feeding forward.

    The upstream QK path independently applies RMSNorm to the first and second
    half of each head. Keep this diagnostic as separate eager ops so it cannot
    change the original ``_split_norm``/RoPE path being compared.
    """
    temporal, spatial = raw.float().chunk(2, dim=-1)
    statistics = torch.cat((
        temporal.pow(2).mean(-1, keepdim=True) + norm_t.eps,
        spatial.pow(2).mean(-1, keepdim=True) + norm_hw.eps,
    ), dim=-1)
    inverse = torch.rsqrt(statistics)
    return statistics, inverse


def attention_source_apply_mrope(attention, normalized, cos, sin):
    # Call the exact source function rather than maintaining a second RoPE impl.
    source = sys.modules.get("hymm.models.vae.pixelumm_decoder.model")
    if source is None:
        raise ReferenceError("pinned upstream module is not loaded")
    return source.apply_mrope(normalized, cos, sin, attention.arch.rope_sections)


def run_layer_reference(
    layer,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: AttentionPlan,
    *,
    capture_stages: Iterable[str] = (),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run upstream GenLayer equations, substituting only bounded local attention."""
    requested = set(capture_stages)
    unknown = requested - set(ALL_STAGES)
    if unknown:
        raise ReferenceError(f"unknown capture stage(s): {', '.join(sorted(unknown))}")
    if x.ndim != 3 or x.shape[1] != math.prod(plan.grid):
        raise ReferenceError("x must have shape [B,J,hidden] for the attention plan")
    captures: dict[str, torch.Tensor] = {}
    attention = layer.attn
    batch, count, _ = x.shape

    if requested & {"input_norm_statistics", "input_norm_pre_gain"}:
        input_f32 = x.float()
        input_norm_statistics = (
            input_f32.pow(2).mean(-1, keepdim=True) + layer.input_norm.eps
        )
        if "input_norm_statistics" in requested:
            captures["input_norm_statistics"] = input_norm_statistics
        if "input_norm_pre_gain" in requested:
            captures["input_norm_pre_gain"] = (
                input_f32 * torch.rsqrt(input_norm_statistics)
            ).to(dtype=x.dtype)
        del input_f32, input_norm_statistics
    normalized_input = layer.input_norm(x)
    if "input_norm" in requested:
        captures["input_norm"] = normalized_input
    q_raw = attention.q_proj(normalized_input).view(
        batch, count, attention.arch.num_heads, attention.arch.head_dim
    )
    k_raw = attention.k_proj(normalized_input).view(
        batch, count, attention.arch.num_kv_heads, attention.arch.head_dim
    )
    v = attention.v_proj(normalized_input).view(
        batch, count, attention.arch.num_kv_heads, attention.arch.head_dim
    )
    if "q_proj" in requested:
        captures["q_proj"] = q_raw
    if "k_proj" in requested:
        captures["k_proj"] = k_raw
    if "v_proj" in requested:
        captures["v_proj"] = v

    # These captures inspect the exact two 64-wide RMSNorm domains per Q/K
    # head. They are computed independently and never replace the source
    # _prepare_qk/_split_norm results used below.
    if requested & {"q_norm_statistics", "q_norm_inverse"}:
        q_statistics, q_inverse = _qk_norm_diagnostics(
            q_raw, attention.q_norm_t, attention.q_norm_hw
        )
        if "q_norm_statistics" in requested:
            captures["q_norm_statistics"] = q_statistics
        if "q_norm_inverse" in requested:
            captures["q_norm_inverse"] = q_inverse
    if requested & {"k_norm_statistics", "k_norm_inverse"}:
        k_statistics, k_inverse = _qk_norm_diagnostics(
            k_raw, attention.k_norm_t, attention.k_norm_hw
        )
        if "k_norm_statistics" in requested:
            captures["k_norm_statistics"] = k_statistics
        if "k_norm_inverse" in requested:
            captures["k_norm_inverse"] = k_inverse

    capture_qk_steps = bool(requested & {
        "q_norm", "k_norm", "q_rope", "k_rope", "q_ready", "k_ready",
    })
    if capture_qk_steps:
        q, q_norm, q_rope = _prepare_qk_with_stages(
            attention, q_raw, cos, sin, attention.q_norm_t, attention.q_norm_hw, v.dtype
        )
        k, k_norm, k_rope = _prepare_qk_with_stages(
            attention, k_raw, cos, sin, attention.k_norm_t, attention.k_norm_hw, v.dtype
        )
        if "q_norm" in requested:
            captures["q_norm"] = q_norm
        if "k_norm" in requested:
            captures["k_norm"] = k_norm
        if "q_rope" in requested:
            captures["q_rope"] = q_rope
        if "k_rope" in requested:
            captures["k_rope"] = k_rope
        if "q_ready" in requested:
            captures["q_ready"] = q
        if "k_ready" in requested:
            captures["k_ready"] = k
    else:
        q = attention._prepare_qk(
            q_raw, cos, sin, attention.q_norm_t, attention.q_norm_hw, v.dtype
        )
        k = attention._prepare_qk(
            k_raw, cos, sin, attention.k_norm_t, attention.k_norm_hw, v.dtype
        )

    attended = gathered_local_attention(q, k, v, plan)
    if "attention" in requested:
        captures["attention"] = attended
    projected_attention = attention.o_proj(
        attended.reshape(batch, count, attention.arch.num_heads * attention.arch.head_dim)
    )
    if "o_proj" in requested:
        captures["o_proj"] = projected_attention
    residual = x + projected_attention
    if "attn_residual" in requested:
        captures["attn_residual"] = residual
    normalized_mlp = layer.post_attn_norm(residual)
    if "post_attn_norm" in requested:
        captures["post_attn_norm"] = normalized_mlp
    gate = layer.mlp.gate_proj(normalized_mlp)
    up = layer.mlp.up_proj(normalized_mlp)
    if "gate_proj" in requested:
        captures["gate_proj"] = gate
    if "up_proj" in requested:
        captures["up_proj"] = up
    silu_gate = torch.nn.functional.silu(gate)
    product = silu_gate * up
    if "silu_gate" in requested:
        captures["silu_gate"] = silu_gate
    if "product" in requested:
        captures["product"] = product
    down = layer.mlp.down_proj(product)
    if "down_proj" in requested:
        captures["down_proj"] = down
    output = residual + down
    if "output" in requested:
        captures["output"] = output
    return output, captures


def _load_layer(source, checkpoint: Path, device: torch.device):
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required to load the bounded block subset") from error
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_SUBSET_BYTES:
        raise ReferenceError("layer-zero subset file is missing or has an unexpected size")
    actual_sha = _sha256_file(checkpoint)
    if actual_sha != CHECKPOINT_SUBSET_SHA256:
        raise ReferenceError("layer-zero subset SHA-256 does not match the pinned inventory")

    arch = source.QWEN3_8B_GEN
    with torch.device("meta"):
        layer = source.GenLayer(arch)
    expected = set(layer.state_dict())
    state: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
            original_keys = set(file.keys())
            expected_keys = {CHECKPOINT_PREFIX + key for key in expected}
            if original_keys != expected_keys:
                raise ReferenceError("checkpoint subset does not contain exactly GenLayer 0")
            for original in sorted(original_keys):
                name = original.removeprefix(CHECKPOINT_PREFIX)
                tensor = file.get_tensor(original)
                if tensor.dtype != torch.float32 or tuple(tensor.shape) != tuple(layer.state_dict()[name].shape):
                    raise ReferenceError(f"checkpoint tensor shape/dtype mismatch: {original}")
                state[name] = tensor
        layer.load_state_dict(state, strict=True, assign=True)
    except ReferenceError:
        raise
    except Exception as error:
        raise ReferenceError(f"cannot load checkpoint layer zero: {error}") from error
    del state
    layer.eval()
    layer.requires_grad_(False)
    # Source preset stores F32 checkpoint weights but runs the decoder in BF16.
    # Transfer the checkpoint subset first, then cast it on the selected device.
    layer.to(device=device)
    layer.to(dtype=torch.bfloat16)
    return layer, actual_sha


def _physical_memory_bytes() -> int:
    try:
        value = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(value)
    except (OSError, ValueError, subprocess.CalledProcessError):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            raise ReferenceError("cannot determine physical memory for the MPS safety floor")


def _capture_bytes_per_token(stage: str, arch) -> int:
    """Storage bytes for one token in a captured intermediate stage."""
    hidden = arch.hidden_size
    q_width = arch.num_heads * arch.head_dim
    kv_width = arch.num_kv_heads * arch.head_dim
    ffn = arch.intermediate_size
    widths = {
        "input_norm_statistics": 4,
        "input_norm_pre_gain": hidden * 2,
        "input_norm": hidden * 2,
        "q_proj": q_width * 2,
        "k_proj": kv_width * 2,
        "v_proj": kv_width * 2,
        "q_norm": q_width * 4,
        "k_norm": kv_width * 4,
        "q_rope": q_width * 4,
        "k_rope": kv_width * 4,
        "q_norm_statistics": arch.num_heads * 2 * 4,
        "k_norm_statistics": arch.num_kv_heads * 2 * 4,
        "q_norm_inverse": arch.num_heads * 2 * 4,
        "k_norm_inverse": arch.num_kv_heads * 2 * 4,
        "q_ready": q_width * 2,
        "k_ready": kv_width * 2,
        "attention": q_width * 2,
        "o_proj": hidden * 2,
        "attn_residual": hidden * 2,
        "post_attn_norm": hidden * 2,
        "gate_proj": ffn * 2,
        "up_proj": ffn * 2,
        "silu_gate": ffn * 2,
        "product": ffn * 2,
        "down_proj": hidden * 2,
        "output": hidden * 2,
    }
    try:
        return widths[stage]
    except KeyError as error:
        raise ReferenceError(f"unknown PixelUMM capture stage: {stage}") from error


def _estimated_peak_bytes(
    grid: tuple[int, int, int], arch, *, layers: int = 1,
    resident_layers: int | None = None,
    capture_stages: Iterable[str] | None = None,
    full36_capture_policy: str | None = None,
) -> int:
    if type(layers) is not int or layers not in {1, 2, 4, 8, 16, 36}:
        raise ReferenceError("reference memory estimate supports one, two, four, eight, sixteen, or thirty-six blocks")
    if (layers == 36) != (full36_capture_policy is not None):
        raise ReferenceError("thirty-six-block memory estimate requires its explicit capture policy")
    if layers == 36 and full36_capture_policy != THIRTY_SIX_CAPTURE_POLICY:
        raise ReferenceError("unsupported thirty-six-block memory capture policy")
    if resident_layers is None:
        resident_layers = layers
    if (type(resident_layers) is not int or not 1 <= resident_layers <= layers
            or (layers == 36 and resident_layers != 1)):
        raise ReferenceError("thirty-six-block reference must materialize one resident layer at a time")
    tokens = math.prod(grid)
    h = arch.hidden_size
    f = arch.intermediate_size
    block_weights_f32 = 771_785_728
    block_weights_bf16 = 385_892_864
    hidden = tokens * h * 2
    ffn = tokens * f * 2
    _, height, width = grid
    tiles_h = (height + TILE_TOKENS - 1) // TILE_TOKENS
    tiles_w = (width + TILE_TOKENS - 1) // TILE_TOKENS
    max_q = min(TILE_TOKENS, height) * min(TILE_TOKENS, width) * grid[0]
    max_k = min(3, tiles_h) * min(3, tiles_w) * min(TILE_TOKENS, height) * min(TILE_TOKENS, width) * grid[0]
    score = arch.num_heads * max_q * max_k * 4
    # Loading a later block briefly retains the previous resident BF16 layer,
    # the new block in F32, and its BF16 copy. The full36 oracle streams one
    # layer at a time while retaining only two-bank survivor captures, so
    # weight and transient hidden slots scale with `resident_layers`.
    weight_bytes = block_weights_f32 + resident_layers * block_weights_bf16
    hidden_slots = 7 + 4 * (resident_layers - 1)
    captured_bytes = 0
    if layers == 36 and capture_stages is None:
        raise ReferenceError("thirty-six-block memory estimate requires its capture inventory")
    if capture_stages is not None:
        stages = tuple(capture_stages)
        if not set(stages) <= set(ALL_STAGES):
            raise ReferenceError("memory estimate received an unsupported capture stage")
        if layers == 36:
            if stages != FULL_TARGET_STAGES:
                raise ReferenceError("thirty-six-block memory estimate requires attention/output boundaries")
            captured_bytes = tokens * (
                layers * _capture_bytes_per_token("attention", arch)
                + 2 * _capture_bytes_per_token("output", arch)
            )
        else:
            captured_bytes = layers * tokens * sum(
                _capture_bytes_per_token(stage, arch) for stage in stages
            )
    return (weight_bytes + hidden_slots * hidden + 4 * ffn + 2 * score
            + 512 * 1024 * 1024 + captured_bytes)


def _estimated_fixture_payload_bytes(
    grid: tuple[int, int, int], arch, *, layers: int,
    capture_stages: Iterable[str],
    full36_capture_policy: str | None = None,
) -> int:
    """Exact serialized extent for the pinned BF16 inputs, weights and captures."""
    if type(layers) is not int or layers not in {1, 2, 4, 8, 16, 36}:
        raise ReferenceError("fixture payload estimate supports one, two, four, eight, sixteen, or thirty-six blocks")
    checked_grid = _check_grid(grid)
    tokens = math.prod(checked_grid)
    stages = tuple(capture_stages)
    if not set(stages) <= set(ALL_STAGES):
        raise ReferenceError("fixture payload estimate received an unsupported capture stage")
    if (layers == 36) != (full36_capture_policy is not None):
        raise ReferenceError("thirty-six-block fixture estimate requires its explicit capture policy")
    if layers == 36 and full36_capture_policy != THIRTY_SIX_CAPTURE_POLICY:
        raise ReferenceError("unsupported thirty-six-block fixture capture policy")
    if layers == 36 and (checked_grid != SIXTEEN_GRID or stages != FULL_TARGET_STAGES):
        raise ReferenceError("thirty-six-block fixture admits full-grid two-bank survivor captures only")
    input_bytes = tokens * (arch.hidden_size * 2 + 2 * arch.head_dim * 4)
    weights = layers * 385_892_864
    if layers == 36:
        captures = tokens * (
            layers * _capture_bytes_per_token("attention", arch)
            + 2 * _capture_bytes_per_token("output", arch)
        )
    else:
        captures = layers * tokens * sum(
            _capture_bytes_per_token(stage, arch) for stage in stages
        )
    result = input_bytes + weights + captures
    if layers == 16 and result > SIXTEEN_FIXTURE_PAYLOAD_CAP:
        raise ReferenceError("sixteen-block fixture exceeds its 16 GiB payload cap")
    if layers == 36 and result > THIRTY_SIX_FIXTURE_PAYLOAD_CAP:
        raise ReferenceError("thirty-six-block fixture exceeds its 32 GiB payload cap")
    return result


def mps_memory_floor(
    device: torch.device, grid: tuple[int, int, int], arch, *, layers: int = 1,
    resident_layers: int | None = None,
    capture_stages: Iterable[str] | None = None,
    full36_capture_policy: str | None = None,
) -> dict[str, int | bool]:
    if device.type != "mps":
        return {"applies": False}
    total = _physical_memory_bytes()
    allocated = int(torch.mps.driver_allocated_memory())
    estimated = _estimated_peak_bytes(
        grid, arch, layers=layers, resident_layers=resident_layers,
        capture_stages=capture_stages,
        full36_capture_policy=full36_capture_policy,
    )
    floor = total // 10
    if total - allocated - estimated < floor:
        raise ReferenceError(
            "MPS reference preflight would leave less than the required 10% physical-memory floor"
        )
    return {
        "applies": True, "physical_bytes": total, "driver_allocated_bytes": allocated,
        "estimated_peak_bytes": estimated, "required_floor_bytes": floor,
        "resident_blocks": layers if resident_layers is None else resident_layers,
        "capture_blocks": layers,
    }


def _tensor_dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.bfloat16:
        return "BF16"
    if tensor.dtype == torch.float32:
        return "F32"
    if tensor.dtype == torch.int32:
        return "I32"
    raise ReferenceError(f"fixture does not support tensor dtype {tensor.dtype}")


def _write_cache_weight_records(
    target, *, cache_manifest: Mapping, cache_payload_path: Path,
    records: dict, payload_digest, start_offset: int,
) -> int:
    """Copy validated full36 BF16 weight records with bounded host memory."""
    block_count = len(cache_manifest.get("blocks", {}))
    if (block_count != 36 or not isinstance(cache_manifest.get("weights"), dict)
            or not isinstance(cache_manifest.get("blocks"), dict)):
        raise ReferenceError("streamed fixture weights require a validated 36-block cache")
    expected_keys = {
        f"block{index}.{name}"
        for index in range(36) for name in checkpoint_inventory.block_schema(index)
    }
    cache_weights = cache_manifest["weights"]
    if set(cache_weights) != expected_keys:
        raise ReferenceError("full36 cache weight inventory differs from the pinned decoder")

    cache_payload_original = Path(cache_payload_path).expanduser()
    if cache_payload_original.is_symlink() or not cache_payload_original.is_file():
        raise ReferenceError("streamed BF16 cache payload must be a regular file")
    cache_payload_path = cache_payload_original.resolve(strict=True)
    before = cache_payload_path.stat()
    if (type(cache_manifest.get("payload_bytes")) is not int
            or cache_manifest["payload_bytes"] != before.st_size):
        raise ReferenceError("streamed BF16 cache payload extent differs from its manifest")
    cache_aggregate = hashlib.sha256()
    cursor = start_offset
    cache_cursor = 0
    with cache_payload_path.open("rb") as source_stream:
        for index in range(36):
            block = cache_manifest["blocks"].get(str(index))
            block_keys = sorted(key for key in expected_keys if key.startswith(f"block{index}."))
            if (not isinstance(block, dict) or block.get("selected_block") != index
                    or block.get("weight_keys") != block_keys):
                raise ReferenceError(f"full36 cache block {index} ownership is malformed")
            for key in block_keys:
                source_record = cache_weights[key]
                shape = list(checkpoint_inventory.block_schema(index)[key.removeprefix(f"block{index}.")])
                nbytes = math.prod(shape) * 2
                if (not isinstance(source_record, dict)
                        or source_record.get("dtype") != "BF16"
                        or source_record.get("shape") != shape
                        or source_record.get("offset") != cache_cursor
                        or source_record.get("nbytes") != nbytes
                        or not _digest(source_record.get("sha256"))):
                    raise ReferenceError(f"full36 cache tensor record is malformed: {key}")
                source_stream.seek(cache_cursor)
                record_digest = hashlib.sha256()
                remaining = nbytes
                while remaining:
                    chunk = source_stream.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ReferenceError(f"full36 cache tensor is truncated: {key}")
                    target.write(chunk)
                    payload_digest.update(chunk)
                    cache_aggregate.update(chunk)
                    record_digest.update(chunk)
                    remaining -= len(chunk)
                actual_sha = record_digest.hexdigest()
                if actual_sha != source_record["sha256"]:
                    raise ReferenceError(f"full36 cache tensor digest changed: {key}")
                records[key] = {
                    "dtype": "BF16", "shape": shape, "offset": cursor,
                    "nbytes": nbytes, "sha256": actual_sha,
                }
                cursor += nbytes
                cache_cursor += nbytes
    after = cache_payload_path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ReferenceError("full36 cache payload changed while copying fixture weights")
    if (cache_cursor != before.st_size
            or cache_aggregate.hexdigest() != cache_manifest.get("payload_sha256")):
        raise ReferenceError("full36 cache payload digest or coverage mismatch")
    return cursor


def _queue_cache_weight_word_checks(
    layer, *, cache_manifest: Mapping, cache_words: torch.Tensor,
    block_index: int, device: torch.device,
) -> dict[str, torch.Tensor]:
    """Queue exact device-side BF16-word checks against the independent cache.

    Cache words move to the selected device in bounded chunks. Each returned
    scalar stays on device until the sequence's single terminal synchronization.
    """
    block = cache_manifest["blocks"].get(str(block_index))
    if not isinstance(block, dict):
        raise ReferenceError(f"BF16 cache has no block {block_index} ownership record")
    parameters = dict(layer.named_parameters())
    expected_names = sorted(checkpoint_inventory.block_schema(block_index))
    expected_keys = [f"block{block_index}.{name}" for name in expected_names]
    if block.get("weight_keys") != expected_keys or set(parameters) != set(expected_names):
        raise ReferenceError(f"BF16 cache block {block_index} parameter ownership mismatch")
    cache_records = cache_manifest["weights"]
    words_per_chunk = (8 * 1024 * 1024) // 2
    checks: dict[str, torch.Tensor] = {}
    for name in expected_names:
        key = f"block{block_index}.{name}"
        parameter = parameters[name].detach()
        record = cache_records[key]
        if (parameter.dtype != torch.bfloat16 or not parameter.is_contiguous()
                or list(parameter.shape) != record.get("shape")
                or record.get("dtype") != "BF16"
                or type(record.get("offset")) is not int
                or type(record.get("nbytes")) is not int
                or record["nbytes"] != parameter.numel() * 2
                or record["offset"] % 2):
            raise ReferenceError(f"BF16 cache record does not match loaded parameter: {key}")
        source_words = parameter.view(torch.int16).reshape(-1)
        cache_start = record["offset"] // 2
        cache_count = record["nbytes"] // 2
        if cache_start + cache_count > cache_words.numel():
            raise ReferenceError(f"BF16 cache record exceeds its payload: {key}")
        combined = None
        for first in range(0, cache_count, words_per_chunk):
            count = min(words_per_chunk, cache_count - first)
            source_chunk = source_words.narrow(0, first, count)
            cache_chunk = cache_words.narrow(0, cache_start + first, count).to(device=device)
            equal = source_chunk.eq(cache_chunk).all()
            combined = equal if combined is None else combined.logical_and(equal)
            del source_chunk, cache_chunk, equal
        if combined is None:
            raise ReferenceError(f"BF16 cache record is empty: {key}")
        checks[key] = combined
    del parameters
    return checks


def _write_fixture(path: Path, *, metadata: Mapping, inputs: Mapping[str, torch.Tensor],
                   weights: Mapping[str, torch.Tensor], captures: Mapping[str, torch.Tensor],
                   schema: str = SCHEMA,
                   weight_cache: tuple[Mapping, Path] | None = None) -> dict:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = path.with_suffix(".bin")
    if path.exists() or path.is_symlink() or payload_path.exists() or payload_path.is_symlink():
        raise ReferenceError("refusing to replace an existing reference JSON or payload")
    payload_digest = hashlib.sha256()
    payload_bytes = 0
    manifest = {
        **metadata,
        "schema": schema,
        "payload": payload_path.name,
        "inputs": {}, "weights": {}, "captures": {},
    }
    with payload_path.open("xb") as stream:
        for group_name, tensors in (("inputs", inputs), ("weights", weights), ("captures", captures)):
            if group_name == "weights" and weight_cache is not None:
                if tensors:
                    raise ReferenceError("fixture weights cannot mix tensors and streamed cache records")
                cache_manifest, cache_payload_path = weight_cache
                payload_bytes = _write_cache_weight_records(
                    stream, cache_manifest=cache_manifest,
                    cache_payload_path=cache_payload_path,
                    records=manifest["weights"], payload_digest=payload_digest,
                    start_offset=payload_bytes,
                )
                continue
            for name, tensor in tensors.items():
                cpu = tensor.detach().contiguous().cpu()
                dtype_name = _tensor_dtype_name(cpu)
                array = cpu.view(torch.uint8).reshape(-1).numpy()
                byte_view = memoryview(array).cast("B")
                nbytes = len(byte_view)
                expected_bytes = cpu.numel() * cpu.element_size()
                if nbytes != expected_bytes:
                    raise ReferenceError(f"unexpected serialized tensor extent for {name}")
                record_digest = hashlib.sha256()
                chunk_size = 8 * 1024 * 1024
                for first in range(0, nbytes, chunk_size):
                    chunk = byte_view[first:first + chunk_size]
                    stream.write(chunk)
                    payload_digest.update(chunk)
                    record_digest.update(chunk)
                manifest[group_name][name] = {
                    "dtype": dtype_name,
                    "shape": list(cpu.shape),
                    "offset": payload_bytes,
                    "nbytes": nbytes,
                    "sha256": record_digest.hexdigest(),
                }
                payload_bytes += nbytes
        stream.flush()
        os.fsync(stream.fileno())
    manifest["payload_bytes"] = payload_bytes
    manifest["payload_sha256"] = payload_digest.hexdigest()
    manifest["capture_order"] = list(captures)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def _read_receipt(path: Path) -> tuple[dict, str]:
    path = path.expanduser().resolve()
    try:
        raw = path.read_bytes()
        value = checkpoint_inventory.parse_header(raw)
    except (OSError, checkpoint_inventory.InventoryError) as error:
        raise ReferenceError(f"cannot read block subset receipt: {path.name}") from error
    return value, hashlib.sha256(raw).hexdigest()


def _load_layer_for_block(
    source, checkpoint: Path, receipt_path: Path, header_path: Path,
    device: torch.device, block_index: int,
):
    """Load one individually receipted source block into the pinned GenLayer."""
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required to load bounded block subsets") from error
    checkpoint_original = Path(checkpoint).expanduser()
    receipt_original = Path(receipt_path).expanduser()
    header_original = Path(header_path).expanduser()
    if any(path.is_symlink() for path in (
        checkpoint_original, receipt_original, header_original,
    )):
        raise ReferenceError("block reference source files must not be symlinks")
    checkpoint = checkpoint_original.resolve()
    receipt_path = receipt_original.resolve()
    header_path = header_original.resolve()
    base_source_dir = (checkpoint_inventory.ROOT / "source").resolve()
    expected_source_dir = (
        THIRTY_SIX_SOURCE_DIR if block_index >= 16 else base_source_dir
    )
    if (checkpoint.parent != expected_source_dir
            or receipt_path.parent != expected_source_dir
            or header_path.parent != base_source_dir):
        raise ReferenceError("block reference source files must be in their pinned task source directory")
    try:
        raw_header = header_path.read_bytes()
        receipt, receipt_sha = _read_receipt(receipt_path)
        checked = checkpoint_inventory.validate_block_receipt(
            raw_header, receipt, checkpoint, block_index
        )
    except (OSError, checkpoint_inventory.InventoryError) as error:
        raise ReferenceError(f"block-{block_index} source receipt rejected: {error}") from error

    arch = source.QWEN3_8B_GEN
    with torch.device("meta"):
        layer = source.GenLayer(arch)
    expected = set(layer.state_dict())
    prefix = f"decoder.layers.{block_index}."
    state: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
            original_keys = set(file.keys())
            expected_keys = {prefix + key for key in expected}
            if original_keys != expected_keys:
                raise ReferenceError(f"block-{block_index} file does not contain exactly 13 expected tensors")
            for original in sorted(original_keys):
                name = original.removeprefix(prefix)
                tensor = file.get_tensor(original)
                if (tensor.dtype != torch.float32
                        or tuple(tensor.shape) != tuple(layer.state_dict()[name].shape)):
                    raise ReferenceError(f"block-{block_index} checkpoint tensor shape/dtype mismatch: {original}")
                state[name] = tensor
        layer.load_state_dict(state, strict=True, assign=True)
    except ReferenceError:
        raise
    except Exception as error:
        raise ReferenceError(f"cannot load checkpoint block {block_index}: {error}") from error
    del state
    layer.eval()
    layer.requires_grad_(False)
    # Match the pinned source runtime order: transfer F32, then cast on device.
    layer.to(device=device)
    layer.to(dtype=torch.bfloat16)
    return layer, checked, receipt_sha


def single_block_capture_stages(grid: Iterable[int], capture_mode: str) -> tuple[str, ...]:
    grid = _check_grid(grid)
    if capture_mode not in {"auto", "all", "boundary"}:
        raise ReferenceError("capture mode must be auto, all, or boundary")
    count = math.prod(grid)
    chosen = (ALL_STAGES if count <= 2048 else FULL_TARGET_STAGES)
    if capture_mode == "all":
        chosen = ALL_STAGES
    elif capture_mode == "boundary":
        chosen = FULL_TARGET_STAGES
    if count > 2048 and grid != (10, 18, 32) and chosen == ALL_STAGES:
        raise ReferenceError("all-stage captures require at most 2048 tokens or the J5760 preview window")
    return chosen


def validate_preview_oracle_runtime(grid: Iterable[int], device_name: str) -> None:
    # MPS inverse-root lowering changes between Torch 2.6 and 2.8. A new
    # preview fixture must use the same runtime as the existing decoder/head
    # controls; otherwise a backend change looks like a native regression.
    if (tuple(grid) == (10, 18, 32) and device_name == "mps"
            and torch.__version__ != MPS_ORACLE_TORCH_VERSION):
        raise ReferenceError(
            f"preview MPS oracle requires torch {MPS_ORACLE_TORCH_VERSION}; "
            f"got {torch.__version__} from {sys.executable}"
        )


def make_fixture(
    *, upstream: Path | str, checkpoint: Path | str, output: Path | str,
    grid: Iterable[int], device_name: str = "mps", capture_mode: str = "auto",
    seed: int = 20261026,
) -> dict:
    output_path = Path(output).expanduser().absolute()
    payload_path = output_path.with_suffix(".bin")
    if (output_path.exists() or output_path.is_symlink()
            or payload_path.exists() or payload_path.is_symlink()):
        raise ReferenceError("refusing to replace an existing reference JSON or payload")
    grid = _check_grid(grid)
    if device_name not in {"cpu", "mps"}:
        raise ReferenceError("device must be cpu or mps")
    validate_preview_oracle_runtime(grid, device_name)
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is not available")
    count = math.prod(grid)
    chosen = single_block_capture_stages(grid, capture_mode)

    source, source_sha = load_upstream(upstream)
    plan, _, _ = make_attention_plan(source, grid)
    arch = source.QWEN3_8B_GEN
    memory = mps_memory_floor(
        device, grid, arch, capture_stages=chosen
    )
    layer, checkpoint_sha = _load_layer(source, Path(checkpoint), device)

    # Generate a reproducible synthetic hidden state on CPU before any GPU math.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_original = torch.randn((1, count, arch.hidden_size), generator=generator, dtype=torch.float32)
    x_original = x_original.to(device=device).to(dtype=torch.bfloat16)
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    x = x_original.index_select(1, permutation)
    del x_original
    cos, sin = source.mrope_tables(arch, grid, device)
    cos = cos.index_select(1, permutation)
    sin = sin.index_select(1, permutation)
    with torch.inference_mode():
        # The upstream decoder runs this layer after moving the complete model
        # to BF16. Explicit autocast mirrors its BF16 inference context.
        context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        with context:
            result, captures = run_layer_reference(
                layer, x, cos, sin, plan, capture_stages=chosen
            )
    if device.type == "mps":
        # One terminal synchronization before any artifact readback.
        torch.mps.synchronize()

    state = {name: parameter.detach() for name, parameter in layer.named_parameters()}
    if len(state) != 13:
        raise ReferenceError("loaded layer parameter inventory changed")
    inputs = {"x": x, "cos": cos, "sin": sin}
    meta = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_path": UPSTREAM_MODEL_PATH,
        "upstream_model_sha256": source_sha,
        "checkpoint_subset_sha256": checkpoint_sha,
        "grid": list(grid),
        "tile_tokens": TILE_TOKENS,
        "token_order": "tile_major",
        "permutation_sha256": plan.permutation_sha256,
        "runtime_dtype": "BF16",
        "input_generator": {"distribution": "standard_normal", "seed": seed, "cast": "BF16"},
        "source_config": asdict(arch),
        "attention_adapter": "gathered_spatial_tiles_qk_softmax_pv_fp32_accum",
        "gqa_group_size": arch.num_heads // arch.num_kv_heads,
        "comparison": {"atol": ATOL, "rtol": RTOL},
        "memory_preflight": memory,
        "captures_are_tile_major": True,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_module": str(Path(torch.__file__).resolve()),
    }
    fixture = _write_fixture(
        Path(output), metadata=meta, inputs=inputs, weights=state, captures=captures
    )
    return {
        "fixture": str(Path(output).expanduser().resolve()),
        "payload": str(Path(output).expanduser().resolve().with_suffix(".bin")),
        "grid": list(grid), "tokens": count, "capture_order": fixture["capture_order"],
        "payload_bytes": fixture["payload_bytes"],
        "estimated_payload_bytes": _estimated_fixture_payload_bytes(
            grid, arch, layers=1, capture_stages=chosen
        ),
        "payload_sha256": fixture["payload_sha256"],
    }


def _block_io_edge(index: int) -> dict[str, str]:
    """Describe the predecessor edge for one of the admitted block slots."""
    if type(index) is not int or not 0 <= index < 36:
        raise ReferenceError("block I/O edges are defined for block indices zero through thirty-five")
    return {
        "input_source": "inputs.x" if index == 0 else f"captures.block{index - 1}.output",
        "output_key": f"captures.block{index}.output",
    }


def _full36_capture_stages(index: int) -> tuple[str, ...]:
    """Return the survivor stages retained for one full-depth block."""
    if type(index) is not int or not 0 <= index < 36:
        raise ReferenceError("full36 capture slots are defined for blocks zero through thirty-five")
    return ("attention",) if index < 34 else FULL_TARGET_STAGES


def _full36_capture_order() -> tuple[str, ...]:
    return tuple(
        f"block{index}.{stage}"
        for index in range(36)
        for stage in _full36_capture_stages(index)
    )


def _validate_full36_capture_inventory(
    *, policy: object, capture_order: Iterable[str], blocks: Iterable[Mapping],
) -> None:
    """Fail closed unless the fixture contains the exact two-bank survivors."""
    if policy != THIRTY_SIX_CAPTURE_POLICY:
        raise ReferenceError("full36 fixture requires the two-bank survivor capture policy")
    if (not isinstance(capture_order, (list, tuple))
            or tuple(capture_order) != _full36_capture_order()):
        raise ReferenceError("full36 fixture capture inventory is not the exact 38 survivor keys")
    if not isinstance(blocks, (list, tuple)):
        raise ReferenceError("full36 fixture block capture inventories are malformed")
    blocks = tuple(blocks)
    if len(blocks) != 36:
        raise ReferenceError("full36 fixture must describe exactly 36 block capture inventories")
    for index, block in enumerate(blocks):
        expected_keys = [
            f"block{index}.{stage}" for stage in _full36_capture_stages(index)
        ]
        if (not isinstance(block, Mapping)
                or type(block.get("index")) is not int
                or block.get("index") != index
                or block.get("capture_keys") != expected_keys):
            raise ReferenceError(f"full36 block {index} capture inventory is malformed")


def _make_block_sequence_fixture(
    *, upstream: Path | str,
    block_sources: Iterable[tuple[Path | str, Path | str]],
    header: Path | str, output: Path | str, grid: Iterable[int],
    schema: str, device_name: str = "mps", capture_mode: str = "auto",
    seed: int = 20261026, real_input_manifest: Path | str | None = None,
    weight_cache_manifest: Path | str | None = None,
) -> dict:
    """Execute an admitted block sequence on one device and retain its edges."""
    block_sources = tuple(block_sources)
    if len(block_sources) not in {2, 4, 8, 16, 36}:
        raise ReferenceError("reference fixture requires exactly 2, 4, 8, 16, or 36 consecutive blocks")
    expected_schema = {
        2: TWO_BLOCK_SCHEMA,
        4: FOUR_BLOCK_SCHEMA,
        8: EIGHT_BLOCK_SCHEMA,
        16: SIXTEEN_BLOCK_SCHEMA,
        36: THIRTY_SIX_BLOCK_SCHEMA,
    }[len(block_sources)]
    if schema != expected_schema:
        raise ReferenceError("block count and reference schema do not match")
    output_path = Path(output).expanduser().absolute()
    payload_path = output_path.with_suffix(".bin")
    block_count = len(block_sources)
    expected_results_dir = THIRTY_SIX_RESULTS_DIR if block_count == 36 else (
        SIXTEEN_RESULTS_DIR if block_count == 16 else (
            checkpoint_inventory.ROOT / "results"
        ).resolve()
    )
    if output_path.parent.resolve() != expected_results_dir:
        raise ReferenceError("multi-block reference output must be in its pinned task results directory")
    if (output_path.exists() or output_path.is_symlink()
            or payload_path.exists() or payload_path.is_symlink()):
        raise ReferenceError("refusing to replace an existing multi-block reference JSON or payload")
    grid = _check_grid(grid)
    if device_name not in {"cpu", "mps"}:
        raise ReferenceError("device must be cpu or mps")
    if block_count in {16, 36} and device_name != "mps":
        raise ReferenceError("sixteen- and thirty-six-block neural references require MPS execution")
    if (block_count in {16, 36}) != (real_input_manifest is not None):
        raise ReferenceError("sixteen- and thirty-six-block references require exactly one real-input manifest")
    if (block_count == 36) != (weight_cache_manifest is not None):
        raise ReferenceError("thirty-six-block reference requires exactly one BF16 weight-cache manifest")
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is not available")
    if capture_mode not in {"auto", "all", "boundary"}:
        raise ReferenceError("capture mode must be auto, all, or boundary")
    count = math.prod(grid)
    real_input = None
    if block_count in {16, 36}:
        if tuple(grid) not in {SIXTEEN_GRID, SMALL_REAL_INPUT_GRID}:
            raise ReferenceError(
                "sixteen- and thirty-six-block references admit only grids [6, 44, 80] and [2, 32, 32]"
            )
        if block_count == 36 and tuple(grid) != SIXTEEN_GRID:
            raise ReferenceError("thirty-six-block reference requires the full [6, 44, 80] grid")
        manifest_path = Path(real_input_manifest).expanduser()
        if manifest_path.name not in REAL_INPUT_ARTIFACTS:
            raise ReferenceError("multi-block reference requires a pinned real-input manifest")
        profile, _, _ = REAL_INPUT_ARTIFACTS[manifest_path.name]
        if ((profile == "full") != (tuple(grid) == SIXTEEN_GRID)):
            raise ReferenceError("real-input profile does not match the requested token grid")
        if block_count == 36 and profile != "full":
            raise ReferenceError("thirty-six-block reference requires a full real-input manifest")
        if (profile == "small" and capture_mode == "boundary"):
            raise ReferenceError("small real-input profile requires all-stage captures")
        if (profile == "full" and capture_mode == "all"):
            raise ReferenceError("full real-input profile permits attention/output captures only")
    if count <= 2048:
        chosen = ALL_STAGES if capture_mode in {"auto", "all"} else FULL_TARGET_STAGES
    else:
        chosen = FULL_TARGET_STAGES if capture_mode in {"auto", "boundary"} else ALL_STAGES
    if count > 2048 and chosen == ALL_STAGES:
        raise ReferenceError("all-stage captures are restricted to at most 2048 tokens")
    if block_count == 16 and len(block_sources) * len(chosen) != (
        416 if tuple(grid) == SMALL_REAL_INPUT_GRID else 32
    ):
        raise ReferenceError("sixteen-block profile has an unsupported capture inventory")
    if block_count == 36 and chosen != FULL_TARGET_STAGES:
        raise ReferenceError("thirty-six-block reference captures attention and output boundaries only")
    full36_capture_policy = (
        THIRTY_SIX_CAPTURE_POLICY if block_count == 36 else None
    )

    source, source_sha = load_upstream(upstream)
    plan, _, _ = make_attention_plan(source, grid)
    arch = source.QWEN3_8B_GEN
    estimated_payload_bytes = _estimated_fixture_payload_bytes(
        grid, arch, layers=block_count, capture_stages=chosen,
        full36_capture_policy=full36_capture_policy,
    )
    memory = mps_memory_floor(
        device, grid, arch, layers=len(block_sources),
        resident_layers=1 if block_count == 36 else None,
        capture_stages=chosen,
        full36_capture_policy=full36_capture_policy,
    )
    cache_manifest = None
    cache_manifest_sha = None
    cache_payload_path = None
    if block_count == 36:
        cache_input = Path(weight_cache_manifest).expanduser()
        if (cache_input.is_symlink() or not cache_input.is_absolute()
                or cache_input.parent.resolve() != THIRTY_SIX_SOURCE_DIR
                or cache_input.name != "pixelumm-thirty-six-bf16-cache.json"):
            raise ReferenceError("thirty-six-block reference requires the pinned full-depth BF16 cache")
        cache_path = cache_input.resolve(strict=True)
        try:
            import hy_nomas_pixelumm_bf16_cache as cache_module

            cache_module.validate_cache_file(cache_path)
            raw_cache_manifest = cache_path.read_bytes()
            cache_manifest = checkpoint_inventory.parse_header(raw_cache_manifest)
        except (OSError, checkpoint_inventory.InventoryError, ValueError) as error:
            raise ReferenceError(f"thirty-six-block BF16 cache was rejected: {error}") from error
        cache_manifest_sha = hashlib.sha256(raw_cache_manifest).hexdigest()
        cache_payload_path = cache_path.parent / cache_manifest["payload"]
        if cache_payload_path.is_symlink() or not cache_payload_path.is_file():
            raise ReferenceError("thirty-six-block BF16 cache payload is missing or a symlink")
    if block_count in {16, 36}:
        real_input = validate_real_input_manifest(real_input_manifest)
        if tuple(real_input["grid"]) != tuple(grid):
            raise ReferenceError("real-input manifest grid differs from the requested grid")
        if real_input["x_permutation_sha256"] != plan.permutation_sha256:
            raise ReferenceError(
                "real-input x tile-major permutation differs from the pinned upstream plan"
            )
        dcae_source = Path(upstream).expanduser().resolve() / "hymm/models/vae/autoencoder_dcae_v.py"
        if (dcae_source.is_symlink() or not dcae_source.is_file()
                or _sha256_file(dcae_source) != REAL_INPUT_DCAE_SOURCE_SHA256
                or source_sha != REAL_INPUT_PIXELUMM_SOURCE_SHA256):
            raise ReferenceError("real-input upstream source hashes changed")
    header_path = Path(header)
    layers = []
    subset_provenance = []
    if block_count != 36:
        for index, (checkpoint_path, receipt_path) in enumerate(block_sources):
            layer, checked, receipt_sha = _load_layer_for_block(
                source, Path(checkpoint_path), Path(receipt_path), header_path, device, index
            )
            layers.append(layer)
            subset_provenance.append({
                "index": index,
                "checkpoint_subset_sha256": checked["subset_sha256"],
                "checkpoint_subset_bytes": checked["subset_bytes"],
                "checkpoint_subset_file": checked["subset_file"],
                "source_receipt_sha256": receipt_sha,
                "source_extent": checked["block_data_offsets"],
                "legacy_extent_derived_from_pinned_header": checked[
                    "legacy_extent_derived_from_pinned_header"
                ],
            })

    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    if real_input is None:
        # Preserve the established synthetic path unchanged for 2/4/8 blocks.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        x_original = torch.randn((1, count, arch.hidden_size), generator=generator, dtype=torch.float32)
        x_original = x_original.to(device=device).to(dtype=torch.bfloat16)
        x = x_original.index_select(1, permutation)
        del x_original
    else:
        # The producer already applies upstream THW -> tile-major ordering.
        x = _load_real_input_x(real_input, device)
    cos, sin = source.mrope_tables(arch, grid, device)
    cos = cos.index_select(1, permutation)
    sin = sin.index_select(1, permutation)

    block_captures: dict[str, torch.Tensor] = {}
    current = x
    weight_word_checks: dict[str, torch.Tensor] = {}
    weight_word_check_count = 0
    cache_words = None
    if block_count == 36:
        cache_payload_bytes = cache_manifest["payload_bytes"]
        if cache_payload_bytes % 2 or cache_payload_bytes > 16 * 1024**3:
            raise ReferenceError("full36 BF16 cache payload extent is unsupported")
        # A read-only file mapping avoids materializing another 13 GiB CPU copy.
        cache_bytes = torch.from_file(
            str(cache_payload_path), shared=False,
            size=cache_payload_bytes, dtype=torch.uint8,
        )
        cache_words = cache_bytes.view(torch.int16)
        del cache_bytes
    with torch.inference_mode():
        context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        with context:
            for index in range(block_count):
                if block_count == 36:
                    checkpoint_path, receipt_path = block_sources[index]
                    layer, checked, receipt_sha = _load_layer_for_block(
                        source, Path(checkpoint_path), Path(receipt_path),
                        header_path, device, index,
                    )
                    cache_block = cache_manifest["blocks"][str(index)]
                    if (cache_block.get("subset_sha256") != checked["subset_sha256"]
                            or cache_block.get("subset_bytes") != checked["subset_bytes"]
                            or cache_block.get("source_extent") != checked["block_data_offsets"]
                            or cache_block.get("source_receipt_sha256") != receipt_sha):
                        raise ReferenceError(
                            f"full36 cache source provenance differs for block {index}"
                        )
                    subset_provenance.append({
                        "index": index,
                        "checkpoint_subset_sha256": checked["subset_sha256"],
                        "checkpoint_subset_bytes": checked["subset_bytes"],
                        "checkpoint_subset_file": checked["subset_file"],
                        "source_receipt_sha256": receipt_sha,
                        "source_extent": checked["block_data_offsets"],
                        "legacy_extent_derived_from_pinned_header": checked[
                            "legacy_extent_derived_from_pinned_header"
                        ],
                    })
                    weight_word_checks.update(_queue_cache_weight_word_checks(
                        layer, cache_manifest=cache_manifest,
                        cache_words=cache_words, block_index=index, device=device,
                    ))
                else:
                    layer = layers[index]
                block_stages = (
                    _full36_capture_stages(index) if block_count == 36 else chosen
                )
                result, captures = run_layer_reference(
                    layer, current, cos, sin, plan, capture_stages=block_stages
                )
                for stage in block_stages:
                    block_captures[f"block{index}.{stage}"] = captures[stage]
                # No inter-block device-to-host handoff occurs: this tensor
                # feeds the next layer directly on the selected device.
                current = result
                if block_count == 36:
                    del layer, captures, result
    word_check_vector = None
    if block_count == 36:
        expected_check_keys = {
            f"block{index}.{name}"
            for index in range(36)
            for name in checkpoint_inventory.block_schema(index)
        }
        if set(weight_word_checks) != expected_check_keys or len(weight_word_checks) != 468:
            raise ReferenceError("full36 MPS weight-word checks did not cover all 468 parameters")
        weight_word_check_count = len(weight_word_checks)
        word_check_vector = torch.stack([
            weight_word_checks[key] for key in sorted(weight_word_checks)
        ])
    if device.type == "mps":
        # Single terminal synchronization after the entire decoder prefix.
        torch.mps.synchronize()

    if block_count == 36:
        if not all(word_check_vector.cpu().tolist()):
            raise ReferenceError("loaded MPS BF16 words differ from authenticated cache records")
        del word_check_vector, weight_word_checks, cache_words

    weights: dict[str, torch.Tensor] = {}
    blocks_metadata = []
    for index in range(block_count):
        prefix = f"block{index}."
        if block_count == 36:
            weight_keys = list(cache_manifest["blocks"][str(index)]["weight_keys"])
        else:
            layer = layers[index]
            state = {name: parameter.detach() for name, parameter in layer.named_parameters()}
            if len(state) != 13:
                raise ReferenceError(f"loaded block-{index} parameter inventory changed")
            for name in sorted(state):
                weights[prefix + name] = state[name]
            weight_keys = [prefix + name for name in sorted(state)]
        block_stages = (
            _full36_capture_stages(index) if block_count == 36 else chosen
        )
        capture_keys = [f"block{index}.{stage}" for stage in block_stages]
        block_metadata = {
            "index": index,
            "checkpoint_subset_sha256": subset_provenance[index]["checkpoint_subset_sha256"],
            "checkpoint_subset_bytes": subset_provenance[index]["checkpoint_subset_bytes"],
            "source_receipt_sha256": subset_provenance[index]["source_receipt_sha256"],
            "source_extent": subset_provenance[index]["source_extent"],
            "weight_prefix": prefix,
            "weight_keys": weight_keys,
            "capture_keys": capture_keys,
            **_block_io_edge(index),
        }
        if block_count == 36:
            block_metadata["weight_word_check_count"] = 13
        blocks_metadata.append(block_metadata)
    metadata = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_path": UPSTREAM_MODEL_PATH,
        "upstream_model_sha256": source_sha,
        "hf_revision": checkpoint_inventory.REVISION,
        "hf_source_path": checkpoint_inventory.SOURCE_PATH,
        "checkpoint_header_sha256": checkpoint_inventory.HEADER_SHA256,
        "blocks": blocks_metadata,
        "grid": list(grid),
        "tile_tokens": TILE_TOKENS,
        "token_order": "tile_major",
        "permutation_sha256": plan.permutation_sha256,
        "runtime_dtype": "BF16",
        "source_config": asdict(arch),
        "attention_adapter": "gathered_spatial_tiles_qk_softmax_pv_fp32_accum",
        "gqa_group_size": arch.num_heads // arch.num_kv_heads,
        "comparison": {"atol": ATOL, "rtol": RTOL},
        "memory_preflight": memory,
        "captures_are_tile_major": True,
        "block_execution": "same_device_output_to_next_input",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    if real_input is None:
        metadata["input_generator"] = {
            "distribution": "standard_normal", "seed": seed, "cast": "BF16",
        }
    else:
        metadata["input_provenance"] = real_input["input_provenance"]
        metadata["input_lineage"] = real_input["input_lineage"]
        metadata["real_input_manifest_sha256"] = real_input["manifest_sha256"]
        metadata["real_input_payload_sha256"] = real_input["payload_sha256"]
        metadata["real_input_payload_bytes"] = real_input["payload_bytes"]
    if block_count == 36:
        metadata.update({
            "full36_capture_policy": full36_capture_policy,
            "weight_cache_manifest_sha256": cache_manifest_sha,
            "weight_cache_payload_sha256": cache_manifest["payload_sha256"],
            "weight_cache_payload_bytes": cache_manifest["payload_bytes"],
            "weight_word_verification": {
                "method": "loaded_mps_bf16_int16_word_equality",
                "tensor_count": 468,
                "all_equal": True,
            },
        })
        _validate_full36_capture_inventory(
            policy=metadata.get("full36_capture_policy"),
            capture_order=tuple(block_captures),
            blocks=blocks_metadata,
        )
    fixture = _write_fixture(
        output_path, metadata=metadata, inputs={"x": x, "cos": cos, "sin": sin},
        weights=weights, captures=block_captures, schema=schema,
        weight_cache=(cache_manifest, cache_payload_path) if block_count == 36 else None,
    )
    if block_count in {16, 36} and fixture["payload_bytes"] != estimated_payload_bytes:
        raise ReferenceError("multi-block fixture payload differs from its exact estimate")
    return {
        "fixture": str(output_path.resolve()),
        "payload": str(payload_path.resolve()),
        "grid": list(grid), "tokens": count, "capture_order": fixture["capture_order"],
        "payload_bytes": fixture["payload_bytes"],
        "estimated_payload_bytes": estimated_payload_bytes,
        "payload_sha256": fixture["payload_sha256"],
        "blocks": subset_provenance,
        "estimated_peak_bytes": memory.get("estimated_peak_bytes"),
        **({
            "weight_cache_manifest_sha256": cache_manifest_sha,
            "weight_cache_payload_sha256": cache_manifest["payload_sha256"],
            "weight_cache_payload_bytes": cache_manifest["payload_bytes"],
            "weight_word_check_count": weight_word_check_count,
        } if block_count == 36 else {}),
    }


def make_two_block_fixture(
    *, upstream: Path | str, block0: Path | str, block1: Path | str,
    receipt0: Path | str, receipt1: Path | str, header: Path | str,
    output: Path | str, grid: Iterable[int], device_name: str = "mps",
    capture_mode: str = "auto", seed: int = 20261026,
) -> dict:
    """Backward-compatible pair entry point and schema."""
    return _make_block_sequence_fixture(
        upstream=upstream,
        block_sources=((block0, receipt0), (block1, receipt1)),
        header=header, output=output, grid=grid, schema=TWO_BLOCK_SCHEMA,
        device_name=device_name, capture_mode=capture_mode, seed=seed,
    )


def make_four_block_fixture(
    *, upstream: Path | str,
    block0: Path | str, block1: Path | str, block2: Path | str, block3: Path | str,
    receipt0: Path | str, receipt1: Path | str, receipt2: Path | str, receipt3: Path | str,
    header: Path | str, output: Path | str, grid: Iterable[int],
    device_name: str = "mps", capture_mode: str = "auto", seed: int = 20261026,
) -> dict:
    """Run blocks zero through three sequentially on one device."""
    return _make_block_sequence_fixture(
        upstream=upstream,
        block_sources=(
            (block0, receipt0), (block1, receipt1),
            (block2, receipt2), (block3, receipt3),
        ),
        header=header, output=output, grid=grid, schema=FOUR_BLOCK_SCHEMA,
        device_name=device_name, capture_mode=capture_mode, seed=seed,
    )


def make_eight_block_fixture(
    *, upstream: Path | str,
    block0: Path | str, block1: Path | str, block2: Path | str, block3: Path | str,
    block4: Path | str, block5: Path | str, block6: Path | str, block7: Path | str,
    receipt0: Path | str, receipt1: Path | str, receipt2: Path | str, receipt3: Path | str,
    receipt4: Path | str, receipt5: Path | str, receipt6: Path | str, receipt7: Path | str,
    header: Path | str, output: Path | str, grid: Iterable[int],
    device_name: str = "mps", capture_mode: str = "auto", seed: int = 20261026,
) -> dict:
    """Run blocks zero through seven sequentially on one device."""
    return _make_block_sequence_fixture(
        upstream=upstream,
        block_sources=(
            (block0, receipt0), (block1, receipt1),
            (block2, receipt2), (block3, receipt3),
            (block4, receipt4), (block5, receipt5),
            (block6, receipt6), (block7, receipt7),
        ),
        header=header, output=output, grid=grid, schema=EIGHT_BLOCK_SCHEMA,
        device_name=device_name, capture_mode=capture_mode, seed=seed,
    )


def make_sixteen_block_fixture(
    *, upstream: Path | str, block_sources: Iterable[tuple[Path | str, Path | str]],
    header: Path | str, output: Path | str, grid: Iterable[int],
    real_input_manifest: Path | str, device_name: str = "mps",
    capture_mode: str = "auto",
) -> dict:
    """Run the bounded real-input reference through decoder blocks 0–15."""
    return _make_block_sequence_fixture(
        upstream=upstream, block_sources=block_sources,
        header=header, output=output, grid=grid,
        schema=SIXTEEN_BLOCK_SCHEMA, device_name=device_name,
        capture_mode=capture_mode, real_input_manifest=real_input_manifest,
    )


def make_thirty_six_block_fixture(
    *, upstream: Path | str, block_sources: Iterable[tuple[Path | str, Path | str]],
    header: Path | str, output: Path | str, grid: Iterable[int],
    real_input_manifest: Path | str, weight_cache_manifest: Path | str,
    device_name: str = "mps", capture_mode: str = "auto",
) -> dict:
    """Run a fresh, streamed real-input reference through blocks 0–35."""
    return _make_block_sequence_fixture(
        upstream=upstream, block_sources=block_sources,
        header=header, output=output, grid=grid,
        schema=THIRTY_SIX_BLOCK_SCHEMA, device_name=device_name,
        capture_mode=capture_mode, real_input_manifest=real_input_manifest,
        weight_cache_manifest=weight_cache_manifest,
    )


def geometry_report(upstream: Path | str, grid: Iterable[int]) -> dict:
    source, source_sha = load_upstream(upstream)
    plan, _, _ = make_attention_plan(source, grid)
    max_q = max(region.query_stop - region.query_start for region in plan.regions)
    max_k = max(len(region.key_positions) for region in plan.regions)
    return {
        "schema": "cogni_ml.pixelumm_block_geometry/v1",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_sha256": source_sha,
        "grid": list(plan.grid), "tile_tokens": plan.tile,
        "tokens": math.prod(plan.grid), "tiles": len(plan.regions),
        "max_query_tokens_per_tile": max_q, "max_key_tokens_per_tile": max_k,
        "permutation_sha256": plan.permutation_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--block0", type=Path)
    parser.add_argument("--block1", type=Path)
    parser.add_argument("--block2", type=Path)
    parser.add_argument("--block3", type=Path)
    parser.add_argument("--block4", type=Path)
    parser.add_argument("--block5", type=Path)
    parser.add_argument("--block6", type=Path)
    parser.add_argument("--block7", type=Path)
    for index in range(8, 16):
        parser.add_argument(f"--block{index}", type=Path)
    for index in range(16, 36):
        parser.add_argument(f"--block{index}", type=Path)
    parser.add_argument("--receipt0", type=Path)
    parser.add_argument("--receipt1", type=Path)
    parser.add_argument("--receipt2", type=Path)
    parser.add_argument("--receipt3", type=Path)
    parser.add_argument("--receipt4", type=Path)
    parser.add_argument("--receipt5", type=Path)
    parser.add_argument("--receipt6", type=Path)
    parser.add_argument("--receipt7", type=Path)
    for index in range(8, 16):
        parser.add_argument(f"--receipt{index}", type=Path)
    for index in range(16, 36):
        parser.add_argument(f"--receipt{index}", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid", required=True, type=int, nargs=3, metavar=("T", "H", "W"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--capture", choices=("auto", "all", "boundary"), default="auto")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--real-input-manifest", type=Path)
    parser.add_argument("--weight-cache-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.geometry_only:
            print(json.dumps(geometry_report(args.upstream, args.grid), sort_keys=True))
            return 0
        block_paths = tuple(getattr(args, f"block{index}") for index in range(36))
        receipt_paths = tuple(getattr(args, f"receipt{index}") for index in range(36))
        if any(path is not None for path in (*block_paths, *receipt_paths)):
            if args.checkpoint is not None:
                parser.error("--checkpoint cannot be combined with block inputs")
            if args.header is None or args.output is None:
                parser.error("block reference requires --header and --output")
            present = []
            for index, (subset, receipt) in enumerate(zip(block_paths, receipt_paths)):
                if (subset is None) != (receipt is None):
                    parser.error(f"block {index} requires both --block{index} and --receipt{index}")
                if subset is not None:
                    present.append(index)
            count = len(present)
            if count not in {2, 4, 8, 16, 36} or present != list(range(count)):
                parser.error("block inputs must be consecutive indices 0-1, 0-3, 0-7, 0-15, or 0-35")
            if count in {16, 36} and args.real_input_manifest is None:
                parser.error("sixteen- and thirty-six-block references require --real-input-manifest")
            if count not in {16, 36} and args.real_input_manifest is not None:
                parser.error("--real-input-manifest is reserved for the sixteen- and thirty-six-block references")
            if count == 36 and args.weight_cache_manifest is None:
                parser.error("thirty-six-block reference requires --weight-cache-manifest")
            if count != 36 and args.weight_cache_manifest is not None:
                parser.error("--weight-cache-manifest is reserved for the thirty-six-block reference")
            common = {
                "upstream": args.upstream, "header": args.header,
                "output": args.output, "grid": args.grid,
                "device_name": args.device, "capture_mode": args.capture,
                "seed": args.seed,
            }
            if count == 2:
                report = make_two_block_fixture(
                    **common,
                    block0=block_paths[0], block1=block_paths[1],
                    receipt0=receipt_paths[0], receipt1=receipt_paths[1],
                )
            elif count == 4:
                report = make_four_block_fixture(
                    **common,
                    **{f"block{i}": block_paths[i] for i in range(4)},
                    **{f"receipt{i}": receipt_paths[i] for i in range(4)},
                )
            else:
                if count == 8:
                    report = make_eight_block_fixture(
                        **common,
                        **{f"block{i}": block_paths[i] for i in range(8)},
                        **{f"receipt{i}": receipt_paths[i] for i in range(8)},
                    )
                else:
                    factory = make_sixteen_block_fixture
                    extra = {}
                    if count == 36:
                        factory = make_thirty_six_block_fixture
                        extra["weight_cache_manifest"] = args.weight_cache_manifest
                    report = factory(
                        upstream=args.upstream,
                        block_sources=tuple(zip(block_paths[:count], receipt_paths[:count])),
                        header=args.header, output=args.output, grid=args.grid,
                        real_input_manifest=args.real_input_manifest,
                        device_name=args.device, capture_mode=args.capture,
                        **extra,
                    )
        else:
            if args.real_input_manifest is not None:
                parser.error("--real-input-manifest requires blocks 0–15 or 0–35")
            if args.weight_cache_manifest is not None:
                parser.error("--weight-cache-manifest requires blocks 0–35")
            if args.checkpoint is None or args.output is None:
                parser.error("--checkpoint and --output are required unless --geometry-only or pair inputs are used")
            report = make_fixture(
                upstream=args.upstream, checkpoint=args.checkpoint, output=args.output,
                grid=args.grid, device_name=args.device, capture_mode=args.capture, seed=args.seed,
            )
        print(json.dumps(report, sort_keys=True))
        return 0
    except ReferenceError as error:
        print(f"PixelUMM reference rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
