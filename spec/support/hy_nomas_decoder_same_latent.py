#!/usr/bin/env python3
"""Compare native and pinned-upstream PixelUMM on one captured latent/noise pair.

The neural route is MPS-only.  It imports the pinned upstream model equations,
loads authenticated stem/head parameters and streams one of the 36 authenticated
BF16 decoder blocks at a time.  Local FlexAttention is represented by the
existing bounded gathered spatial-tile adapter; this is not an unmodified eager
FlexAttention comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import torch

SUPPORT_DIR = Path(__file__).resolve().parent
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

import hy_nomas_generation_inputs as generation_inputs
import hy_nomas_pixelumm_bf16_cache as cache_inventory
import hy_nomas_pixelumm_checkpoint as checkpoint_inventory
import hy_nomas_pixelumm_head_checkpoint as head_checkpoint
import hy_nomas_pixelumm_head_reference as head_reference
import hy_nomas_pixelumm_real_input as real_input
import hy_nomas_pixelumm_reference as block_reference
import pixelumm_generation_reference as pixelumm_reference


SCHEMA = "cogni_ml.hy_nomas_decoder_same_latent/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_native_preview/v1"
NATIVE_PRECISION = "BF16 weights and boundary rounding, F32 native arithmetic/MPS accumulation"
UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
SCRATCH_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-quality-20261030")
DEFAULT_RESOLUTION = "512x288"
HIGH_RESOLUTION = "768x416"


@dataclass(frozen=True)
class DecoderGeometry:
    resolution: str
    width: int
    height: int
    latent_height: int
    latent_width: int

    @property
    def token_grid(self) -> tuple[int, int, int]:
        return (10, 2 * self.latent_height, 2 * self.latent_width)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return (1, 32, 30, self.latent_height, self.latent_width)

    @property
    def noise_shape(self) -> tuple[int, ...]:
        return (1, 3, 120, self.height, self.width)

    @property
    def raw_shape(self) -> tuple[int, ...]:
        return self.noise_shape

    @property
    def first_window_shape(self) -> tuple[int, ...]:
        return (1, 3, 40, self.height, self.width)

    @property
    def first_prefix_elements(self) -> int:
        return 20 * 3 * self.height * self.width

    @property
    def latent_bytes(self) -> int:
        return math.prod(self.latent_shape) * 4

    @property
    def noise_bytes(self) -> int:
        return math.prod(self.noise_shape) * 4

    @property
    def raw_bytes(self) -> int:
        return math.prod(self.raw_shape) * 4


DEFAULT_GEOMETRY = DecoderGeometry(DEFAULT_RESOLUTION, 512, 288, 9, 16)
HIGH_RESOLUTION_GEOMETRY = DecoderGeometry(HIGH_RESOLUTION, 768, 416, 13, 24)
# Keep the original module constants as the v1/default compatibility surface.
GRID = DEFAULT_GEOMETRY.token_grid
LATENT_SHAPE = DEFAULT_GEOMETRY.latent_shape
NOISE_SHAPE = DEFAULT_GEOMETRY.noise_shape
RAW_SHAPE = DEFAULT_GEOMETRY.raw_shape
FIRST_WINDOW_SHAPE = DEFAULT_GEOMETRY.first_window_shape
FIRST_PREFIX_ELEMENTS = DEFAULT_GEOMETRY.first_prefix_elements
LATENT_BYTES = DEFAULT_GEOMETRY.latent_bytes
NOISE_BYTES = DEFAULT_GEOMETRY.noise_bytes
RAW_BYTES = DEFAULT_GEOMETRY.raw_bytes
ATOL = RTOL = 2e-5
GIB = 1024**3
EXTRA_GPU_RESERVE_BYTES = 1536 * 1024**2
METRIC_CHUNK_ELEMENTS = 1 << 20
ATTENTION_SCOPE = "gathered_spatial_tiles_qk_softmax_pv_fp32_accum"
STEM_SUBSET_SHA256 = "7eb73ebf42f6681e21870dc2d2d4b592846ba15ed5bf5f9a6c7a83d39c2d8d04"
STEM_RECEIPT_SHA256 = "88d8dbda211d5dbe0c817d1036069e1efd57b6567f0eb01b9ab5b691c3409014"
STEM_WEIGHTS_SHA256 = "7257a5d07644a9fddd38918a127eec94abaeb954b0527f2c33777a61c0f0ec1b"


class SameLatentError(ValueError):
    """The supplied capture, pinned inputs, or MPS run failed admission."""


def _geometry_for_resolution(resolution: str) -> DecoderGeometry:
    if resolution == DEFAULT_RESOLUTION:
        return DEFAULT_GEOMETRY
    if resolution == HIGH_RESOLUTION:
        return HIGH_RESOLUTION_GEOMETRY
    raise SameLatentError("only 512x288 default and authenticated 768x416 inputs are admitted")


def _digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SameLatentError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path, what: str, *, max_bytes: int = 2 * 1024 * 1024) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise SameLatentError(f"{what} must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise SameLatentError(f"{what} exceeds the metadata size limit")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SameLatentError(f"cannot parse {what}") from error
    if type(document) is not dict:
        raise SameLatentError(f"{what} must be a JSON object")
    return document, raw


def _canonical_file(path_value: str | Path, what: str) -> Path:
    """Resolve an admitted input boundary and reject symlink files."""
    path = Path(path_value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SameLatentError(f"{what} must be an absolute regular non-symlink file")
    try:
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SameLatentError(f"cannot resolve {what}") from error
    return canonical


def validate_native_receipt_document(
    document: Mapping, *, expected_inputs_sha256: str,
) -> dict[str, object]:
    """Check the narrow native-preview contract before opening tensor files."""
    if type(document) is not dict or document.get("schema") != NATIVE_SCHEMA:
        raise SameLatentError("unsupported native preview receipt schema")
    if document.get("upstream_revision") != block_reference.UPSTREAM_REVISION:
        raise SameLatentError("native receipt upstream revision mismatch")
    if document.get("precision") != NATIVE_PRECISION:
        raise SameLatentError("native receipt precision is unknown")
    if document.get("dit_weights") != "bf16" or document.get("dit_quantization_experimental") is not False:
        raise SameLatentError("native DiT precision contract is unsupported")
    for field, expected in (("seed", 42), ("frames", 120),
                            ("fps", 24), ("steps", 4),
                            ("shift", 5), ("cfg", 1)):
        if type(document.get(field)) is not int or document[field] != expected:
            raise SameLatentError(f"native receipt {field} does not match the admitted preview")
    dimensions = (document.get("width"), document.get("height"))
    if any(type(value) is not int for value in dimensions):
        raise SameLatentError("native receipt display dimensions are malformed")
    resolution_by_dimensions = {(512, 288): DEFAULT_RESOLUTION,
                                (768, 416): HIGH_RESOLUTION}
    if dimensions not in resolution_by_dimensions:
        raise SameLatentError("native receipt resolution is not admitted")
    geometry = _geometry_for_resolution(resolution_by_dimensions[dimensions])
    if document.get("camera_conditioning") is not False or document.get("action_conditioning") is not False:
        raise SameLatentError("native capture has unsupported camera/action conditioning")
    if type(document.get("prompt")) is not str or not document["prompt"]:
        raise SameLatentError("native receipt prompt is missing")
    if (not _digest(expected_inputs_sha256)
            or document.get("inputs_sha256") != expected_inputs_sha256):
        raise SameLatentError("native receipt does not authenticate the supplied input digest")
    if not isinstance(document.get("inputs"), str) or not Path(document["inputs"]).is_absolute():
        raise SameLatentError("native receipt input manifest path is malformed")
    expected_pixelumm = {
        "steps": 1, "window_latents": 10, "overlap_latents": 5,
        "starts": [0, 5, 10, 15, 20],
    }
    if document.get("pixelumm") != expected_pixelumm:
        raise SameLatentError("native receipt PixelUMM window contract mismatch")
    diagnostics = document.get("diagnostics")
    if type(diagnostics) is not dict:
        raise SameLatentError("native receipt diagnostics are missing")
    for path_key, hash_key in (("latent", "latent_sha256"),
                               ("raw_pixels", "raw_pixels_sha256")):
        path_value = diagnostics.get(path_key)
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise SameLatentError(f"native receipt diagnostics.{path_key} path is malformed")
        if not _digest(diagnostics.get(hash_key)):
            raise SameLatentError(f"native receipt diagnostics.{hash_key} is malformed")
    display_path = document.get("output")
    if not isinstance(display_path, str) or not Path(display_path).is_absolute():
        raise SameLatentError("native receipt top-level display output path is malformed")
    if (type(document.get("output_bytes")) is not int
            or document["output_bytes"] != geometry.raw_bytes):
        raise SameLatentError("native receipt top-level display output byte extent is invalid")
    if not _digest(document.get("output_sha256")):
        raise SameLatentError("native receipt top-level display output SHA-256 is malformed")
    result = dict(diagnostics)
    result.update({
        "width": geometry.width,
        "height": geometry.height,
        "resolution": geometry.resolution,
        "display_pixels": display_path,
        "display_pixels_bytes": geometry.raw_bytes,
        "display_pixels_sha256": document["output_sha256"],
    })
    return result


def _regular_artifact(path_value: str, what: str, expected_sha256: str,
                      expected_bytes: int) -> Path:
    path = _canonical_file(path_value, what)
    if path.stat().st_size != expected_bytes:
        raise SameLatentError(f"{what} byte extent does not match its admitted shape")
    if not _digest(expected_sha256) or _sha256_file(path) != expected_sha256:
        raise SameLatentError(f"{what} SHA-256 mismatch")
    return path


def _validate_payload_file(document: Mapping, payload_path: Path) -> None:
    """Hash the packed payload and each contiguous record in one bounded pass."""
    generation_inputs.validate_document(document, payload_bytes=payload_path.stat().st_size)
    if payload_path.is_symlink() or not payload_path.is_file():
        raise SameLatentError("generation-input payload must be a regular non-symlink file")
    aggregate = hashlib.sha256()
    with payload_path.open("rb") as stream:
        for name, _, _ in generation_inputs.TENSOR_SPECS:
            record = document["tensors"][name]
            if stream.tell() != record["offset"]:
                raise SameLatentError("generation-input payload has a gap or overlap")
            record_hash = hashlib.sha256()
            remaining = record["nbytes"]
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise SameLatentError(f"generation-input tensor is truncated: {name}")
                record_hash.update(chunk)
                aggregate.update(chunk)
                remaining -= len(chunk)
            if record_hash.hexdigest() != record["sha256"]:
                raise SameLatentError(f"generation-input tensor digest mismatch: {name}")
        if stream.read(1):
            raise SameLatentError("generation-input payload has trailing bytes")
    if aggregate.hexdigest() != document.get("payload_sha256"):
        raise SameLatentError("generation-input aggregate payload digest mismatch")


def _validate_inputs(path: Path, expected_sha256: str, receipt: Mapping) -> tuple[dict, Path, dict]:
    path = _canonical_file(path, "generation-input manifest")
    manifest, raw = _read_json(path, "generation-input manifest")
    actual_manifest_sha = hashlib.sha256(raw).hexdigest()
    if actual_manifest_sha != expected_sha256 or actual_manifest_sha != receipt["inputs_sha256"]:
        raise SameLatentError("generation-input manifest SHA-256 differs from both authorizations")
    try:
        receipt_input_path = _canonical_file(receipt["inputs"], "native receipt input manifest")
        if receipt_input_path != path:
            raise SameLatentError("generation-input path differs from the native receipt")
    except OSError as error:
        raise SameLatentError("generation-input manifest path cannot be resolved") from error
    payload_name = manifest.get("payload", "")
    payload_path = path.parent / payload_name
    if not isinstance(payload_name, str) or payload_path.name != payload_name:
        raise SameLatentError("generation-input payload must be a sibling basename")
    payload_path = _canonical_file(payload_path, "generation-input payload")
    try:
        _validate_payload_file(manifest, payload_path)
    except generation_inputs.InputsError as error:
        raise SameLatentError(f"generation-input manifest rejected: {error}") from error
    if manifest.get("prompt") != receipt.get("prompt"):
        raise SameLatentError("native receipt prompt differs from the authenticated input bundle")
    if manifest.get("upstream_revision") != block_reference.UPSTREAM_REVISION:
        raise SameLatentError("generation-input upstream revision mismatch")
    if manifest.get("noise_seeds", {}).get("latent_seed") != 42:
        raise SameLatentError("generation-input latent seed mismatch")
    if manifest.get("noise_seeds", {}).get("pixel_noise_seed_formula") != "seed * 1000003 + latent_frame_index":
        raise SameLatentError("generation-input decoder noise seed formula mismatch")
    if manifest.get("noise_seeds", {}).get("pixel_noise_seeds") != [42 * 1_000_003 + i for i in range(30)]:
        raise SameLatentError("generation-input decoder noise seed inventory mismatch")
    noise_record = _validate_decoder_noise_geometry(manifest)
    return manifest, payload_path, noise_record


def _geometry_for_input_manifest(manifest: Mapping) -> DecoderGeometry:
    """Resolve only the legacy/default bundles or the explicit high-resolution v3 bundle."""
    schema = manifest.get("schema")
    if not isinstance(schema, str):
        raise SameLatentError("generation-input schema is malformed")
    if schema in (generation_inputs.SCHEMA, generation_inputs.DYNAMIC_SCHEMA):
        geometry = DEFAULT_GEOMETRY
    elif schema == generation_inputs.HIGH_RESOLUTION_SCHEMA:
        geometry = HIGH_RESOLUTION_GEOMETRY
    else:
        raise SameLatentError("generation-input schema does not select an admitted resolution")

    specs = {
        name: (dtype, list(shape))
        for name, dtype, shape in generation_inputs.tensor_specs_for_resolution(
            geometry.resolution
        )
    }
    tensors = manifest.get("tensors")
    if type(tensors) is not dict:
        raise SameLatentError("generation-input tensor inventory is missing")
    for name in ("input.latent", "decoder.noise", "decoder.rope_cos", "decoder.rope_sin"):
        record = tensors.get(name)
        if (type(record) is not dict
                or (record.get("dtype"), record.get("shape")) != specs[name]
                or record.get("nbytes") != generation_inputs.tensor_nbytes(*specs[name])):
            raise SameLatentError(f"generation-input {name} shape/dtype differs from its resolution")

    return geometry


def _validate_decoder_noise_geometry(
    manifest: Mapping, geometry: DecoderGeometry | None = None,
) -> dict:
    """Admit the exact decoder noise shape, RoPE extent and partial-tile grid."""
    geometry = geometry or _geometry_for_input_manifest(manifest)
    token_grid = geometry.token_grid
    decoder_token_count = math.prod(token_grid)
    manifest_geometry = manifest.get("geometry", {})
    required_geometry = {
        "dit_latent_shape": list(geometry.latent_shape),
        "decoder_latent_grid": [30, geometry.latent_height, geometry.latent_width],
        "decoder_noise_shape": list(geometry.noise_shape),
        "decoder_rope_grid": list(token_grid),
        "decoder_rope_shape": [1, decoder_token_count, 1, 128],
        "decoder_local_attention_tile_pixels": 128,
        "decoder_local_attention_tile_tokens": 8,
        "decoder_noise_piece_frames": 4,
        "decoder_noise_piece_latents": 1,
    }
    if (type(manifest_geometry) is not dict
            or any(manifest_geometry.get(key) != value for key, value in required_geometry.items())):
        raise SameLatentError("generation-input decoder geometry differs from the admitted resolution")
    tensors = manifest.get("tensors")
    if type(tensors) is not dict:
        raise SameLatentError("generation-input tensor inventory is missing")
    noise_record = tensors.get("decoder.noise")
    if (type(noise_record) is not dict or noise_record.get("dtype") != "F32"
            or noise_record.get("shape") != list(geometry.noise_shape)
            or noise_record.get("nbytes") != geometry.noise_bytes):
        raise SameLatentError("generation-input decoder noise tensor shape/dtype mismatch")
    return noise_record


def _resolve_geometry(receipt_contract: Mapping, input_manifest: Mapping) -> DecoderGeometry:
    geometry = _geometry_for_input_manifest(input_manifest)
    if (receipt_contract.get("resolution") != geometry.resolution
            or receipt_contract.get("width") != geometry.width
            or receipt_contract.get("height") != geometry.height):
        raise SameLatentError("native receipt resolution differs from authenticated input geometry")
    _validate_decoder_noise_geometry(input_manifest, geometry)
    schema = input_manifest.get("schema")
    if schema in {generation_inputs.SCHEMA, generation_inputs.HIGH_RESOLUTION_SCHEMA}:
        try:
            pixelumm_reference.resolve_geometry(schema, geometry.resolution)
        except pixelumm_reference.ReferenceError as error:
            raise SameLatentError(f"PixelUMM reference geometry rejected: {error}") from error
    return geometry


def _output_paths(path_arg: Path) -> tuple[Path, Path, Path, Path]:
    path = path_arg.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.suffix.lower() != ".json":
        raise SameLatentError("--output must have a .json suffix")
    try:
        root = SCRATCH_ROOT.resolve(strict=True)
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise SameLatentError("scratch root and output parent must already exist") from error
    if root != parent and root not in parent.parents:
        raise SameLatentError("oracle outputs must remain beneath the authorized scratch root")
    manifest = parent / path.name
    raw = manifest.with_suffix(".f32le")
    first_window_raw = manifest.with_name(manifest.stem + "-first-window.f32le")
    display = manifest.with_name(manifest.stem + "-display.f32le")
    if any(os.path.lexists(candidate) for candidate in (manifest, raw, first_window_raw, display)):
        raise SameLatentError("refusing to replace an existing oracle result")
    return manifest, raw, first_window_raw, display


def _mapped_f32(path: Path, *, offset: int, nbytes: int,
                shape: tuple[int, ...]) -> torch.Tensor:
    if offset < 0 or offset % 4 or nbytes != math.prod(shape) * 4:
        raise SameLatentError("F32 artifact offset or geometry is invalid")
    total_bytes = path.stat().st_size
    if offset + nbytes > total_bytes or total_bytes % 4:
        raise SameLatentError("F32 artifact extent exceeds its file")
    storage = torch.from_file(
        str(path), shared=False, size=total_bytes // 4, dtype=torch.float32
    )
    return storage.narrow(0, offset // 4, nbytes // 4).view(shape)


def _memory_budget(
    *, physical_bytes: int, driver_allocated_bytes: int,
    reference_peak_bytes: int, cache_resident_allowance_bytes: int,
    comparison_buffer_bytes: int = 0,
) -> dict[str, int]:
    values = (physical_bytes, driver_allocated_bytes, reference_peak_bytes,
              cache_resident_allowance_bytes, comparison_buffer_bytes)
    if any(type(value) is not int or value < 0 for value in values) or physical_bytes == 0:
        raise SameLatentError("MPS memory preflight received invalid byte counts")
    estimated = (reference_peak_bytes + comparison_buffer_bytes
                 + EXTRA_GPU_RESERVE_BYTES + cache_resident_allowance_bytes)
    reserve_floor = physical_bytes // 10
    if physical_bytes - driver_allocated_bytes - estimated < reserve_floor:
        raise SameLatentError(
            "same-latent MPS preflight would leave less than the required 10% physical-memory reserve"
        )
    return {
        "physical_bytes": physical_bytes,
        "driver_allocated_bytes": driver_allocated_bytes,
        "reference_peak_estimate_bytes": reference_peak_bytes,
        "raw_display_comparison_buffer_bound_bytes": comparison_buffer_bytes,
        "capture_io_and_allocator_margin_bytes": EXTRA_GPU_RESERVE_BYTES,
        "mapped_full36_cache_resident_allowance_bytes": cache_resident_allowance_bytes,
        "estimated_peak_bytes": estimated,
        "required_reserve_bytes": reserve_floor,
    }


def _validate_execution_runtime(geometry: DecoderGeometry) -> None:
    """Reject an incompatible MPS runtime before hashing decoder/checkpoint assets."""
    try:
        block_reference.validate_preview_oracle_runtime(geometry.token_grid, "mps")
    except block_reference.ReferenceError as error:
        raise SameLatentError(str(error)) from error
    if torch.__version__ != block_reference.MPS_ORACLE_TORCH_VERSION:
        raise SameLatentError(
            f"same-latent MPS oracle requires torch {block_reference.MPS_ORACLE_TORCH_VERSION}; "
            f"got {torch.__version__} from {sys.executable}"
        )
    if not torch.backends.mps.is_available():
        raise SameLatentError("PyTorch MPS is unavailable; CPU model fallback is prohibited")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        raise SameLatentError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")


def _comparison_buffer_bound_bytes(geometry: DecoderGeometry) -> int:
    # Native raw/display are two full F32 buffers. Six full-buffer units also
    # cover the upstream BF16 raw output, display intermediates/output, and
    # bounded reduction scratch without relying on the neural estimate's base.
    return 6 * geometry.raw_bytes


def _mps_preflight(
    arch, *, cache_resident_allowance_bytes: int,
    geometry: DecoderGeometry = DEFAULT_GEOMETRY,
) -> dict[str, int | str]:
    if not torch.backends.mps.is_available():
        raise SameLatentError("PyTorch MPS is unavailable; CPU model fallback is prohibited")
    if torch.__version__ != block_reference.MPS_ORACLE_TORCH_VERSION:
        raise SameLatentError(
            f"MPS PixelUMM oracle requires torch {block_reference.MPS_ORACLE_TORCH_VERSION}; "
            f"got {torch.__version__}"
        )
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise SameLatentError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    device = torch.device("mps")
    layers = 36
    base = block_reference._estimated_peak_bytes(
        geometry.token_grid, arch, layers=layers, resident_layers=1,
        capture_stages=block_reference.FULL_TARGET_STAGES,
        full36_capture_policy=block_reference.THIRTY_SIX_CAPTURE_POLICY,
    )
    physical = block_reference._physical_memory_bytes()
    allocated = int(torch.mps.driver_allocated_memory())
    budget = _memory_budget(
        physical_bytes=physical,
        driver_allocated_bytes=allocated,
        reference_peak_bytes=base,
        cache_resident_allowance_bytes=cache_resident_allowance_bytes,
        comparison_buffer_bytes=_comparison_buffer_bound_bytes(geometry),
    )
    return {
        "device": str(device), "torch_version": torch.__version__,
        **budget,
        "resident_decoder_blocks": 1,
    }


def _authenticate_stem() -> tuple[bytes, Path, Path]:
    source_dir = (checkpoint_inventory.ROOT / "source").resolve()
    header_path = source_dir / "pixelumm-header.json"
    stem_path = source_dir / "stem-f32.safetensors"
    receipt_path = source_dir / "stem-f32.receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (header_path, stem_path, receipt_path)):
        raise SameLatentError("pinned PixelUMM header/stem subset/receipt is missing or a symlink")
    raw_header = header_path.read_bytes()
    try:
        checkpoint_inventory.parse_pinned_header(raw_header)
        receipt_raw = receipt_path.read_bytes()
        if hashlib.sha256(receipt_raw).hexdigest() != STEM_RECEIPT_SHA256:
            raise SameLatentError("pinned PixelUMM stem receipt digest changed")
        stem_receipt = checkpoint_inventory.parse_header(receipt_raw)
        checked = checkpoint_inventory.validate_named_subset_receipt(
            raw_header, stem_receipt, stem_path, "stem"
        )
    except checkpoint_inventory.InventoryError as error:
        raise SameLatentError(f"authenticated PixelUMM stem subset rejected: {error}") from error
    if checked.get("subset_sha256") != STEM_SUBSET_SHA256:
        raise SameLatentError("pinned PixelUMM stem subset digest changed")
    if real_input._raw_safetensors_payload_sha256(stem_path, real_input.STEM_F32_BYTES) != STEM_WEIGHTS_SHA256:
        raise SameLatentError("pinned PixelUMM stem F32 weight digest changed")
    return raw_header, stem_path, receipt_path


def _authenticate_head(raw_header: bytes) -> dict[str, object]:
    """Validate the pinned head subset and receipt without constructing MPS modules."""
    source_dir = head_reference.SOURCE_DIR
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise SameLatentError("pinned PixelUMM head source directory is missing or a symlink")
    try:
        canonical_dir = source_dir.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SameLatentError("cannot resolve pinned PixelUMM head source directory") from error
    subset_path = _canonical_file(
        canonical_dir / "head-f32.safetensors", "pinned PixelUMM head subset"
    )
    receipt_path = _canonical_file(
        canonical_dir / "head-f32.receipt.json", "pinned PixelUMM head receipt"
    )
    try:
        receipt_raw = receipt_path.read_bytes()
        if len(receipt_raw) > 2 * 1024 * 1024:
            raise SameLatentError("pinned PixelUMM head receipt exceeds metadata limit")
        receipt = checkpoint_inventory.parse_header(receipt_raw)
        checked = head_checkpoint.validate_head_subset_receipt(
            raw_header, receipt, subset_path
        )
        expected_fields = {
            "hf_revision": checkpoint_inventory.REVISION,
            "source_commit": checkpoint_inventory.SOURCE_COMMIT,
            "source_path": checkpoint_inventory.SOURCE_PATH,
            "header_sha256": checkpoint_inventory.HEADER_SHA256,
            "source_data_offsets": [head_checkpoint.HEAD_DATA_START,
                                    head_checkpoint.HEAD_DATA_END],
            "source_f32_bytes": head_checkpoint.HEAD_F32_BYTES,
        }
        if any(receipt.get(key) != value for key, value in expected_fields.items()):
            raise SameLatentError("head receipt does not bind the pinned checkpoint source")
    except (checkpoint_inventory.InventoryError, head_checkpoint.HeadCheckpointError) as error:
        raise SameLatentError(f"authenticated PixelUMM head subset rejected: {error}") from error
    return {
        "subset_path": subset_path,
        "receipt_path": receipt_path,
        "subset_sha256": checked["subset_sha256"],
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
    }


def _validate_upstream_source_metadata() -> dict[str, str]:
    """Authenticate the pinned source tree and decoder sources without importing it."""
    if UPSTREAM.is_symlink() or not UPSTREAM.is_dir():
        raise SameLatentError("pinned upstream source directory is missing or a symlink")
    try:
        root = UPSTREAM.resolve(strict=True)
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--",
             block_reference.UPSTREAM_MODEL_PATH,
             real_input.DCAE_ENCODER_RELATIVE,
             real_input.PIXELUMM_NOISE_RELATIVE],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SameLatentError("cannot verify pinned upstream source metadata") from error
    if revision != block_reference.UPSTREAM_REVISION or changed.strip():
        raise SameLatentError("upstream revision or decoder source worktree identity mismatch")
    expected_files = (
        (block_reference.UPSTREAM_MODEL_PATH, real_input.UPSTREAM_MODEL_SHA256),
        (real_input.DCAE_ENCODER_RELATIVE, real_input.DCAE_ENCODER_SHA256),
        (real_input.PIXELUMM_NOISE_RELATIVE, real_input.PIXELUMM_NOISE_SHA256),
    )
    result: dict[str, str] = {"revision": revision}
    for relative, expected_sha in expected_files:
        path = _canonical_file(root / relative, f"pinned upstream source {relative}")
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise SameLatentError(f"pinned upstream source digest changed: {relative}")
        result[relative] = actual_sha
    return result


def _authenticate_full36_cache() -> tuple[dict, Path, Path, str, str]:
    """Validate all 36 cached block records and source receipts without MPS allocation."""
    cache_path = _canonical_file(
        cache_inventory.THIRTY_SIX_CACHE_DIR / "pixelumm-thirty-six-bf16-cache.json",
        "pinned full36 BF16 cache manifest",
    )
    try:
        cache_inventory.validate_cache_file(cache_path)
        manifest, raw = _read_json(cache_path, "full36 BF16 cache manifest")
        cache_payload = _canonical_file(
            cache_path.parent / manifest["payload"], "pinned full36 BF16 cache payload"
        )
    except (ValueError, OSError, KeyError) as error:
        raise SameLatentError(f"pinned full36 BF16 cache rejected: {error}") from error
    payload_bytes = manifest.get("payload_bytes")
    if type(payload_bytes) is not int or payload_bytes != cache_payload.stat().st_size:
        raise SameLatentError("pinned full36 BF16 cache payload extent is invalid")
    return (
        manifest, cache_path, cache_payload,
        hashlib.sha256(raw).hexdigest(), manifest["payload_sha256"],
    )


def _preflight_artifacts(
    *, receipt: Mapping, diagnostics: Mapping, input_document: Mapping,
    input_payload_path: Path, geometry: DecoderGeometry = DEFAULT_GEOMETRY,
) -> dict[str, object]:
    """Finish CPU-only source and input admission before importing/loading models."""
    sources = _validate_upstream_source_metadata()
    latent_path = _regular_artifact(
        diagnostics["latent"], "native latent", diagnostics["latent_sha256"], geometry.latent_bytes
    )
    native_raw_path = _regular_artifact(
        diagnostics["raw_pixels"], "native raw pixels", diagnostics["raw_pixels_sha256"], geometry.raw_bytes
    )
    native_display_path = _regular_artifact(
        diagnostics["display_pixels"], "native display pixels",
        diagnostics["display_pixels_sha256"], geometry.raw_bytes,
    )
    raw_header, stem_path, stem_receipt_path = _authenticate_stem()
    head = _authenticate_head(raw_header)
    cache_manifest, cache_path, cache_payload_path, cache_manifest_sha, cache_payload_sha = (
        _authenticate_full36_cache()
    )
    return {
        "source_metadata": sources,
        "latent_path": latent_path,
        "native_raw_path": native_raw_path,
        "native_display_path": native_display_path,
        "geometry": geometry,
        "raw_header": raw_header,
        "stem_path": stem_path,
        "stem_receipt_path": stem_receipt_path,
        "head": head,
        "cache_manifest": cache_manifest,
        "cache_path": cache_path,
        "cache_payload_path": cache_payload_path,
        "cache_manifest_sha256": cache_manifest_sha,
        "cache_payload_sha256": cache_payload_sha,
        "input_payload_path": _canonical_file(input_payload_path, "generation-input payload"),
        "input_payload_sha256": input_document["payload_sha256"],
        "input_manifest_sha256": receipt["inputs_sha256"],
    }


def _admission_summary(admission: Mapping) -> dict[str, object]:
    head = admission["head"]
    return {
        "schema": SCHEMA,
        "admission": "passed",
        "upstream_revision": admission["source_metadata"]["revision"],
        "native_input_manifest_sha256": admission["input_manifest_sha256"],
        "generation_input_payload_sha256": admission["input_payload_sha256"],
        "stem_subset": str(admission["stem_path"]),
        "head_subset": str(head["subset_path"]),
        "head_subset_sha256": head["subset_sha256"],
        "head_receipt_sha256": head["receipt_sha256"],
        "decoder_cache_manifest_sha256": admission["cache_manifest_sha256"],
        "decoder_cache_payload_sha256": admission["cache_payload_sha256"],
        "latent": str(admission["latent_path"]),
        "native_raw_pixels": str(admission["native_raw_path"]),
        "native_display_pixels": str(admission["native_display_path"]),
        "resolution": admission["geometry"].resolution,
    }


def _load_full36_cache(
    admission: Mapping[str, object],
) -> tuple[dict, Path, torch.Tensor, str, str]:
    manifest = admission["cache_manifest"]
    cache_payload = admission["cache_payload_path"]
    cache_manifest_sha = admission["cache_manifest_sha256"]
    cache_payload_sha = admission["cache_payload_sha256"]
    payload_bytes = manifest.get("payload_bytes")
    if (type(payload_bytes) is not int or payload_bytes <= 0 or payload_bytes % 2
            or payload_bytes != cache_payload.stat().st_size):
        raise SameLatentError("pinned full36 BF16 cache payload extent is invalid")
    cache_bytes = torch.from_file(
        str(cache_payload), shared=False, size=payload_bytes, dtype=torch.uint8
    )
    return (
        manifest, cache_payload, cache_bytes.view(torch.int16),
        cache_manifest_sha, cache_payload_sha,
    )


def _comparison_element_count(
    actual_shape: tuple[int, ...], expected_shape: tuple[int, ...],
    element_count: int | None = None,
    *, geometry: DecoderGeometry = DEFAULT_GEOMETRY,
) -> int:
    """Validate either the whole raw clip or first-window/first-prefix comparison."""
    if expected_shape != geometry.raw_shape:
        raise SameLatentError("native raw-pixel comparison geometry is not the captured clip")
    if element_count is None:
        if actual_shape != geometry.raw_shape:
            raise SameLatentError("full raw-pixel comparison requires the complete clip")
        return geometry.raw_bytes // 4
    if (actual_shape != geometry.first_window_shape
            or element_count != geometry.first_prefix_elements):
        raise SameLatentError("first-window comparison geometry does not match its 20-frame prefix")
    return element_count


def _first_prefix_tensors(
    actual: torch.Tensor, expected: torch.Tensor, *, frames: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the same leading temporal frames across every NCTHW channel."""
    if (actual.ndim != 5 or expected.ndim != 5
            or actual.shape[0] != expected.shape[0]
            or actual.shape[1] != expected.shape[1]
            or actual.shape[3:] != expected.shape[3:]
            or type(frames) is not int or not 0 < frames <= actual.shape[2]
            or frames > expected.shape[2]):
        raise SameLatentError("first-window temporal prefix cannot be selected from these tensors")
    return actual[:, :, :frames], expected[:, :, :frames]


def _queue_pixel_comparison(
    actual: torch.Tensor, expected_native: torch.Tensor, *, element_count: int | None = None,
    geometry: DecoderGeometry = DEFAULT_GEOMETRY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Queue bounded GPU reductions; callers perform the single terminal wait."""
    element_count = _comparison_element_count(
        tuple(actual.shape), tuple(expected_native.shape), element_count,
        geometry=geometry,
    )
    if element_count == geometry.first_prefix_elements:
        actual, expected_native = _first_prefix_tensors(
            actual, expected_native, frames=20
        )
    actual_flat = actual.reshape(-1)
    expected_flat = expected_native.reshape(-1)
    actual_flat = actual_flat[:element_count]
    expected_flat = expected_flat[:element_count]
    maxima: list[torch.Tensor] = []
    sums: list[torch.Tensor] = []
    mismatches: list[torch.Tensor] = []
    finite: list[torch.Tensor] = []
    for start in range(0, actual_flat.numel(), METRIC_CHUNK_ELEMENTS):
        stop = min(start + METRIC_CHUNK_ELEMENTS, actual_flat.numel())
        left = actual_flat[start:stop]
        right = expected_flat[start:stop]
        delta = (left - right).abs()
        close = delta <= (ATOL + RTOL * right.abs())
        maxima.append(delta.max())
        sums.append(delta.sum())
        mismatches.append((~close).any())
        finite.append(torch.isfinite(left).all().logical_and(torch.isfinite(right).all()))
    max_abs = torch.stack(maxima).max()
    mean_abs = torch.stack(sums).sum() / actual_flat.numel()
    any_mismatch = torch.stack(mismatches).any()
    all_finite = torch.stack(finite).all()
    return max_abs, mean_abs, any_mismatch, all_finite


def run_oracle(
    *, receipt: Mapping, native_receipt_sha256: str, diagnostics: Mapping,
    input_document: Mapping, input_payload_path: Path, noise_record: Mapping,
    artifact_admission: Mapping[str, object],
    output_json: Path, output_raw: Path, output_first_window: Path,
    output_display: Path, geometry: DecoderGeometry = DEFAULT_GEOMETRY,
) -> dict:
    """Run the pinned windows on MPS; compare raw and display output to the capture."""
    device = torch.device("mps")
    source, model_sha = block_reference.load_upstream(UPSTREAM)
    if model_sha != real_input.UPSTREAM_MODEL_SHA256:
        raise SameLatentError("pinned upstream model.py source digest changed")
    dcae_source, source_again, autoencoder_source = real_input._load_modules(UPSTREAM)
    if source_again is not source and Path(source_again.__file__).resolve() != Path(source.__file__).resolve():
        raise SameLatentError("upstream module import identity changed")
    if _sha256_file(Path(dcae_source.__file__)) != real_input.DCAE_ENCODER_SHA256:
        raise SameLatentError("pinned upstream DCAE source digest changed")
    if _sha256_file(Path(autoencoder_source.__file__)) != real_input.PIXELUMM_NOISE_SHA256:
        raise SameLatentError("pinned upstream PixelUMM decoder/blender source digest changed")
    block_reference.validate_preview_oracle_runtime(geometry.token_grid, "mps")
    if torch.__version__ != block_reference.MPS_ORACLE_TORCH_VERSION:
        raise SameLatentError(
            f"same-latent MPS oracle requires torch {block_reference.MPS_ORACLE_TORCH_VERSION}; "
            f"got {torch.__version__}"
        )
    arch = source.QWEN3_8B_GEN

    # Main calls _preflight_artifacts before this source import. Reuse those
    # canonical paths and CPU-validated receipts before constructing MPS weights.
    latent_path = artifact_admission["latent_path"]
    native_raw_path = artifact_admission["native_raw_path"]
    native_display_path = artifact_admission["native_display_path"]
    raw_header = artifact_admission["raw_header"]
    stem_path = artifact_admission["stem_path"]
    stem_receipt_path = artifact_admission["stem_receipt_path"]
    head_artifacts = artifact_admission["head"]
    cache_manifest, cache_payload_path, cache_words, cache_manifest_sha, cache_payload_sha = (
        _load_full36_cache(artifact_admission)
    )
    memory = _mps_preflight(
        arch, cache_resident_allowance_bytes=cache_payload_path.stat().st_size,
        geometry=geometry,
    )
    header_path = _canonical_file(
        checkpoint_inventory.ROOT / "source" / "pixelumm-header.json",
        "pinned PixelUMM checkpoint header",
    )
    plan, inverse_cpu, _ = block_reference.make_attention_plan(
        source, geometry.token_grid, tile=8
    )
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    inverse = inverse_cpu.to(device=device)
    cos, sin = source.mrope_tables(arch, geometry.token_grid, device)
    cos = cos.index_select(1, permutation)
    sin = sin.index_select(1, permutation)

    # This is the pinned upstream stem/head path, authenticated before use.
    stem = real_input._load_stem(source, device)
    final_norm, head, head_runtime_weights, head_source = head_reference._load_head_state(
        source=source,
        subset_path=head_artifacts["subset_path"],
        raw_header=raw_header,
        receipt_path=head_artifacts["receipt_path"],
    )
    del head_runtime_weights

    latent_cpu = _mapped_f32(
        latent_path, offset=0, nbytes=geometry.latent_bytes, shape=geometry.latent_shape
    )
    latent = latent_cpu.to(device=device)
    del latent_cpu
    native_cpu = _mapped_f32(
        native_raw_path, offset=0, nbytes=geometry.raw_bytes, shape=geometry.raw_shape
    )
    native_raw = native_cpu.to(device=device)
    del native_cpu
    native_display_cpu = _mapped_f32(
        native_display_path, offset=0, nbytes=geometry.raw_bytes, shape=geometry.raw_shape
    )
    native_display = native_display_cpu.to(device=device)
    del native_display_cpu
    payload_cpu = torch.from_file(
        str(input_payload_path), shared=False,
        size=input_payload_path.stat().st_size // 4, dtype=torch.float32,
    )
    noise = payload_cpu.narrow(
        0, noise_record["offset"] // 4, noise_record["nbytes"] // 4
    ).view(geometry.noise_shape).to(device=device)
    del payload_cpu

    blender = autoencoder_source._TemporalBlender(total_frames=geometry.noise_shape[2])
    condition = type("_ConditionProjection", (), {})()
    condition.cond_proj = stem.cond_proj
    condition.arch = arch
    layer_checks: list[torch.Tensor] = []
    latent_starts = (0, 5, 10, 15, 20)
    current_windows: list[torch.Tensor] = []
    with torch.inference_mode():
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            for latent_start in latent_starts:
                latent_window = latent[:, :, latent_start:latent_start + 10]
                noise_window = noise[:, :, latent_start * 4:(latent_start + 10) * 4]
                xt_tokens = stem.embed(source.patchify(noise_window.to(stem.embed.weight.dtype)))
                condition_tokens = source.DCAEVPixelUMMDecoder._condition_tokens(
                    condition, latent_window.to(stem.cond_proj.weight.dtype)
                )
                current_windows.append(
                    (xt_tokens + condition_tokens).index_select(1, permutation)
                )
                del latent_window, noise_window, xt_tokens, condition_tokens

            # Stream each authenticated block once, advancing all five
            # independent windows before releasing that block's weights.
            # This retains only five BF16 hidden states plus one decoder block.
            for block_index in range(36):
                block_record = cache_manifest["blocks"][str(block_index)]
                source_dir = cache_inventory._source_dir_for_block(36, block_index)
                layer, checked, receipt_sha = block_reference._load_layer_for_block(
                    source,
                    source_dir / block_record["subset_file"],
                    source_dir / block_record["source_receipt_file"],
                    header_path,
                    device,
                    block_index,
                )
                if (block_record.get("subset_sha256") != checked["subset_sha256"]
                        or block_record.get("subset_bytes") != checked["subset_bytes"]
                        or block_record.get("source_extent") != checked["block_data_offsets"]
                        or block_record.get("source_receipt_sha256") != receipt_sha):
                    raise SameLatentError(
                        f"BF16 cache provenance differs from authenticated source for block {block_index}"
                    )
                # The cache file is validated once; source/cache BF16 words are
                # compared on-device on the first load of every block.
                layer_checks.extend(block_reference._queue_cache_weight_word_checks(
                    layer,
                    cache_manifest=cache_manifest,
                    cache_words=cache_words,
                    block_index=block_index,
                    device=device,
                ).values())
                for window_index, current in enumerate(current_windows):
                    current_windows[window_index], captures = block_reference.run_layer_reference(
                        layer, current, cos, sin, plan
                    )
                    if captures:
                        raise SameLatentError("unexpected decoder intermediate capture was retained")
                del layer

        first_window_raw: torch.Tensor | None = None
        for window_index, latent_start in enumerate(latent_starts):
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                decoder_tokens = final_norm(current_windows[window_index])
                decoder_tokens = decoder_tokens.index_select(1, inverse)
            # Pinned DCAEVPixelUMMDecoder.forward exits autocast before PixelHead.
            prediction_tokens = head(decoder_tokens)
            prediction = source.unpatchify(prediction_tokens, geometry.token_grid)
            blender.add(prediction, latent_start * 4)
            if window_index == 0:
                first_window_raw = prediction
            del decoder_tokens, prediction_tokens, prediction

    upstream_raw = blender.result()
    raw_max, raw_mean, raw_mismatch, raw_finite = _queue_pixel_comparison(
        upstream_raw, native_raw, geometry=geometry
    )
    if first_window_raw is None:
        raise SameLatentError("upstream decoder did not retain its first window prediction")
    first_max, first_mean, first_mismatch, first_finite = _queue_pixel_comparison(
        first_window_raw, native_raw,
        element_count=geometry.first_prefix_elements, geometry=geometry,
    )
    upstream_display = pixelumm_reference.postprocess_bf16(torch, upstream_raw)
    display_max, display_mean, display_mismatch, display_finite = _queue_pixel_comparison(
        upstream_display, native_display, geometry=geometry
    )
    checks_pass = torch.stack(layer_checks).all()
    metric_vector = torch.stack((
        raw_max, raw_mean, raw_mismatch.to(torch.float32),
        raw_finite.logical_not().to(torch.float32),
        display_max, display_mean, display_mismatch.to(torch.float32),
        display_finite.logical_not().to(torch.float32),
        checks_pass.logical_not().to(torch.float32),
        first_max, first_mean, first_mismatch.to(torch.float32),
        first_finite.logical_not().to(torch.float32),
    ))
    # One terminal MPS boundary; all neural outputs/checks remain on GPU until here.
    torch.mps.synchronize()
    (raw_max_value, raw_mean_value, raw_mismatch_bit, raw_nonfinite_bit,
     display_max_value, display_mean_value, display_mismatch_bit, display_nonfinite_bit,
     bad_cache_bit, first_max_value, first_mean_value, first_mismatch_bit,
     first_nonfinite_bit) = metric_vector.cpu().tolist()
    upstream_cpu = upstream_raw.to(dtype=torch.float32).contiguous().cpu()
    upstream_display_cpu = upstream_display.contiguous().cpu()
    first_window_cpu = None
    if raw_mismatch_bit or display_mismatch_bit or first_mismatch_bit:
        first_window_cpu = first_window_raw.to(dtype=torch.float32).contiguous().cpu()
    del metric_vector, cache_words, latent, noise, native_raw, native_display

    if raw_nonfinite_bit or display_nonfinite_bit or first_nonfinite_bit:
        raise SameLatentError("native/upstream raw or display pixels contain non-finite values")
    if bad_cache_bit:
        raise SameLatentError("one or more upstream source block weights differ from the BF16 cache")
    try:
        with output_raw.open("xb") as stream:
            upstream_cpu.numpy().astype("<f4", copy=False).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise SameLatentError(f"cannot write upstream raw-pixel boundary artifact: {error}") from error
    output_raw_sha = _sha256_file(output_raw)
    try:
        with output_display.open("xb") as stream:
            upstream_display_cpu.numpy().astype("<f4", copy=False).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise SameLatentError(f"cannot write upstream display-pixel boundary artifact: {error}") from error
    output_display_sha = _sha256_file(output_display)
    first_window_sha = None
    if first_window_cpu is not None:
        with output_first_window.open("xb") as stream:
            first_window_cpu.numpy().astype("<f4", copy=False).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        first_window_sha = _sha256_file(output_first_window)
    result = {
        "schema": SCHEMA,
        "upstream_revision": block_reference.UPSTREAM_REVISION,
        "upstream_model_sha256": model_sha,
        "native_receipt_sha256": native_receipt_sha256,
        "native_raw_pixels_sha256": diagnostics["raw_pixels_sha256"],
        "native_display_pixels_sha256": diagnostics["display_pixels_sha256"],
        "native_precision": receipt["precision"],
        "input_manifest_sha256": receipt["inputs_sha256"],
        "input_payload_sha256": input_document["payload_sha256"],
        "prompt": receipt["prompt"],
        "seed": receipt["seed"],
        "dit_steps": receipt["steps"],
        "decoder_semantics": {
            "prediction": "one direct x0 prediction per window; upstream decoder has no timestep argument",
            "window_latent_starts": [0, 5, 10, 15, 20],
            "window_latents": 10,
            "overlap_latents": 5,
            "pixel_frame_starts": [0, 20, 40, 60, 80],
            "noise_source": "authenticated generation-input decoder.noise F32 tensor",
            "merge": "pinned upstream _TemporalBlender cosine overlap with BF16 values",
        },
        "comparison_scope": {
            "raw_pixels_before_display_normalization": True,
            "display_pixels_after_pinned_bf16_postprocess": True,
            "attention_adapter": ATTENTION_SCOPE,
            "unmodified_eager_flex_attention_parity": False,
            "runtime_dtype": "BF16",
            "atol": ATOL,
            "rtol": RTOL,
        },
        "geometry": {
            "resolution": geometry.resolution,
            "latent": list(geometry.latent_shape),
            "raw_pixels": list(geometry.raw_shape),
            "display_pixels": list(geometry.raw_shape),
            "first_window_raw_pixels": list(geometry.first_window_shape),
            "token_grid": list(geometry.token_grid),
        },
        "memory_preflight": memory,
        "decoder_cache_manifest_sha256": cache_manifest_sha,
        "decoder_cache_payload_sha256": cache_payload_sha,
        "stem_subset_sha256": checkpoint_inventory.validate_named_subset_receipt(
            raw_header,
            checkpoint_inventory.parse_header(stem_receipt_path.read_bytes()),
            stem_path,
            "stem",
        )["subset_sha256"],
        "head_subset_sha256": head_source["subset_sha256"],
        "head_receipt_sha256": head_source["receipt_sha256"],
        "weight_word_checks": len(layer_checks),
        "metrics": {
            "max_abs_error": raw_max_value,
            "mean_abs_error": raw_mean_value,
            "any_value_outside_tolerance": bool(raw_mismatch_bit or display_mismatch_bit),
            "all_values_finite": not bool(raw_nonfinite_bit or display_nonfinite_bit),
            "raw_pixels": {
                "max_abs_error": raw_max_value,
                "mean_abs_error": raw_mean_value,
                "any_value_outside_tolerance": bool(raw_mismatch_bit),
                "all_values_finite": not bool(raw_nonfinite_bit),
            },
            "display_pixels": {
                "max_abs_error": display_max_value,
                "mean_abs_error": display_mean_value,
                "any_value_outside_tolerance": bool(display_mismatch_bit),
                "all_values_finite": not bool(display_nonfinite_bit),
            },
            "all_source_weights_match_bf16_cache": not bool(bad_cache_bit),
            "first_unblended_20_frames": {
                "max_abs_error": first_max_value,
                "mean_abs_error": first_mean_value,
                "any_value_outside_tolerance": bool(first_mismatch_bit),
                "all_values_finite": not bool(first_nonfinite_bit),
            },
            "result": "match" if not (raw_mismatch_bit or display_mismatch_bit) else "mismatch",
        },
        "upstream_raw_pixels": str(output_raw),
        "upstream_raw_pixels_bytes": geometry.raw_bytes,
        "upstream_raw_pixels_sha256": output_raw_sha,
        "upstream_display_pixels": str(output_display),
        "upstream_display_pixels_bytes": geometry.raw_bytes,
        "upstream_display_pixels_sha256": output_display_sha,
        "first_window_raw_pixels": str(output_first_window) if first_window_sha else None,
        "first_window_raw_pixels_shape": list(geometry.first_window_shape) if first_window_sha else None,
        "first_window_raw_pixels_sha256": first_window_sha,
    }
    with output_json.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt-sha256", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--admission-only", action="store_true",
        help="authenticate all source and input artifacts on CPU, without loading models or using MPS",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt_path = _canonical_file(args.native_receipt, "native receipt")
        receipt, raw_receipt = _read_json(receipt_path, "native receipt")
        actual_receipt_sha = hashlib.sha256(raw_receipt).hexdigest()
        if not _digest(args.native_receipt_sha256) or actual_receipt_sha != args.native_receipt_sha256:
            raise SameLatentError("native receipt SHA-256 does not match caller authorization")
        diagnostics = validate_native_receipt_document(
            receipt, expected_inputs_sha256=args.inputs_sha256
        )
        input_manifest, input_payload_path, noise_record = _validate_inputs(
            args.inputs.expanduser(), args.inputs_sha256, receipt
        )
        geometry = _resolve_geometry(diagnostics, input_manifest)
        output_paths = None
        if not args.admission_only:
            if args.output is None:
                raise SameLatentError("--output is required unless --admission-only is selected")
            output_paths = _output_paths(args.output)
            _validate_execution_runtime(geometry)
        artifact_admission = _preflight_artifacts(
            receipt=receipt,
            diagnostics=diagnostics,
            input_document=input_manifest,
            input_payload_path=input_payload_path,
            geometry=geometry,
        )
        if args.admission_only:
            print(json.dumps(_admission_summary(artifact_admission), indent=2))
            return 0
        if output_paths is None:
            raise SameLatentError("non-admission execution did not validate its output paths")
        output_json, output_raw, output_first_window, output_display = output_paths
        result = run_oracle(
            receipt=receipt,
            native_receipt_sha256=actual_receipt_sha,
            diagnostics=diagnostics,
            input_document=input_manifest,
            input_payload_path=input_payload_path,
            noise_record=noise_record,
            artifact_admission=artifact_admission,
            output_json=output_json,
            output_raw=output_raw,
            output_first_window=output_first_window,
            output_display=output_display,
            geometry=geometry,
        )
    except (SameLatentError, block_reference.ReferenceError,
            head_reference.HeadReferenceError, real_input.RealInputError,
            cache_inventory.CacheError, checkpoint_inventory.InventoryError,
            generation_inputs.InputsError, head_checkpoint.HeadCheckpointError,
            RuntimeError, OSError) as error:
        print(f"same-latent decoder oracle failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output_json), "result": result["metrics"]["result"],
                      "memory_preflight": result["memory_preflight"]}, indent=2))
    return 0 if result["metrics"]["result"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
