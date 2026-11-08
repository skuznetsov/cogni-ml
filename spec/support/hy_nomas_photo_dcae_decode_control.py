#!/usr/bin/env python3
"""Decode the original photo's authenticated T0 latents with upstream DCAE-V.

Default/``--plan-only`` mode authenticates the original photo, prepared RGB8,
native T0 lineage, compact encoder latent, pinned upstream source and original
BF16 checkpoint using CPU metadata only. ``--execute`` materializes only the
upstream decoder on MPS, decodes 30 latents as three cached ten-latent tiles,
and transfers one packed terminal buffer containing clamped RGB8 frames and
comparison metrics. Run execute under an external 18 GiB cap with a 10%
physical-memory reserve; private MPS workspace is opaque to the estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


SUPPORT = Path(__file__).resolve().parent
REPO_ROOT = SUPPORT.parents[1]
EXAMPLES_SUPPORT = REPO_ROOT / "examples" / "support"
for entry in (str(EXAMPLES_SUPPORT), str(SUPPORT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import dcae_v_checkpoint as checkpoint  # noqa: E402
import dcae_v_reference as reference  # noqa: E402


DEFAULT_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue")
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8")
DEFAULT_PHOTO = Path("/Users/sergey/Documents/sergey_facebook_avatar.jpg")
DEFAULT_WEIGHTS = Path(
    "/opt/mldata_ssd/models/hy-nomas/dcae-source/VAE/DCAE-V/"
    "dc-ae-v-1.0-f32t4c32-bf16.pt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "photo-original-dcae-reconstruction-v1"

UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DCAE_MODULE_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
DCAE_MODULE_SHA256 = "38163ebead9a6bb643d27843022c1757c45c6b8638c334a476aee5b373edf8c1"
DCAE_CHECKPOINT_SHA256 = "ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"
DCAE_CHECKPOINT_BYTES = 4_277_389_995
PHOTO_SHA256 = "d38704c64cffcb83e1c43218964cff8824d63a3a3bca9e77dcc88369a0756ed9"
PHOTO_BYTES = 14_748
REFERENCE_MANIFEST_SHA256 = "04dd1a12f22d41336499b1d3d9d83d2bc92313900cd6116d3565392da04e6628"
PREPARED_RGB8_SHA256 = "c5d6ea9b0ade1f91946844f22f36b764231bccb37098f5cb61ba543fe0e884b2"
REFERENCE_RGB8_BYTES = 442_368
NATIVE_RECEIPT_SHA256 = "ca8be135ef659497594dec9647c72240f911be8cfd6be989f0740fbf76cd66f5"
NATIVE_BANK_SHA256 = "1717edc3d6cf33a3a374852a3dc58cd9d82d5f74965fab6dd0239bbe93da943b"
ENCODER_SHA256 = "d0bbacbebe8960b147388d0bf61ec8bbc0eb8d575574d6ec5c5abe4878c8d6c1"
T0_LATENT_SHA256 = "6730c2618b6534768d626bbbd70119a46e2808bbcce7a7c20e6dd87e8e22e7ed"
T0_RECEIPT_SHA256 = "ed5f4f1f46ce23c8d698b5f1ee36f4934586fcc3006389829a7d0bb8efe8dbf9"

REFERENCE_SHAPE = (1, 3, 120, 288, 512)
LATENT_SHAPE = (1, 32, 30, 9, 16)
NATIVE_BANK_SHAPE = (1, 32, 180, 9, 16)
T0_LATENT_BYTES = math.prod(LATENT_SHAPE) * 4
DECODER_TENSOR_COUNT = 171
DECODER_BF16_BYTES = 2_443_960_856
ENCODER_TENSOR_COUNT = 169
ATOL = RTOL = 2e-5
TILE_LATENTS = 10
TILE_FRAMES = 40
GIB = 1024**3
EXTERNAL_CAP_BYTES = 18 * GIB
RESERVE_FRACTION = 0.10


class ControlError(ValueError):
    """A photo, lineage, source, checkpoint, runtime, or output was rejected."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str, *, max_bytes: int = 8 * 1024 * 1024) -> tuple[dict, bytes]:
    raw = _read_regular(path, label, max_bytes=max_bytes)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"invalid {label}: {exc}") from None
    if type(value) is not dict:
        raise ControlError(f"{label} must be a JSON object")
    return value, raw


def _read_regular(path: Path, label: str, *, max_bytes: int | None = None) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ControlError(f"{label} must be a regular non-symlink file: {path}")
    try:
        with path.open("rb") as stream:
            raw = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
    except OSError as exc:
        raise ControlError(f"cannot read {label}: {path}") from exc
    if max_bytes is not None and len(raw) > max_bytes:
        raise ControlError(f"{label} exceeds its size limit")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ControlError(f"{label} SHA-256 mismatch")


def _positive_shape(shape: object, label: str) -> list[int]:
    if (type(shape) not in (list, tuple) or not shape
            or any(type(dim) is not int or dim <= 0 for dim in shape)):
        raise ControlError(f"{label} shape is malformed")
    return list(shape)


def validate_photo_manifest(
    manifest: Mapping[str, Any], manifest_sha256: str, photo_sha256: str,
    payload_bytes: int, payload_sha256: str,
) -> None:
    """Require the exact original-photo and cached-RGB8 preparation contract."""
    if manifest_sha256 != REFERENCE_MANIFEST_SHA256:
        raise ControlError("reference manifest identity mismatch")
    expected = {
        "schema": "cogni_ml.hy_nomas_reference_image/v1",
        "schema_version": 1,
        "upstream_revision": DCAE_MODULE_REVISION,
        "checkpoint_sha256": DCAE_CHECKPOINT_SHA256,
        "encoder_safetensors_sha256": ENCODER_SHA256,
        "image_path": str(DEFAULT_PHOTO),
        "image_sha256": PHOTO_SHA256,
        "payload": "pixels.rgb8",
        "payload_bytes": REFERENCE_RGB8_BYTES,
        "payload_sha256": PREPARED_RGB8_SHA256,
        "width": 512,
        "height": 288,
        "frames": 120,
        "fit_policy": "exif-rgb-lanczos-pad-black-center",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ControlError("reference image or prepared RGB8 manifest differs from the admitted photo")
    if photo_sha256 != PHOTO_SHA256 or payload_sha256 != PREPARED_RGB8_SHA256:
        raise ControlError("original photo or prepared RGB8 payload digest mismatch")
    if payload_bytes != REFERENCE_RGB8_BYTES:
        raise ControlError("prepared RGB8 payload extent mismatch")


def validate_t0_lineage(
    native: Mapping[str, Any], dcae: Mapping[str, Any], *, latent_bytes: int,
    latent_sha256: str, native_receipt_sha256: str,
) -> None:
    """Join clean native reference T0, upstream encode receipt, and latent bytes."""
    if native_receipt_sha256 != NATIVE_RECEIPT_SHA256:
        raise ControlError("native receipt identity mismatch")
    if native.get("schema") != "cogni_ml.hy_nomas_native_continuation/v1":
        raise ControlError("native receipt schema mismatch")
    if (native.get("reference_initialization") is not True
            or native.get("technical_reference_segment_index") != 0
            or native.get("technical_reference_latent_frames") != 30
            or native.get("generated_segment_start_index") != 1):
        raise ControlError("native receipt does not identify a clean technical T0 reference")
    diagnostics = native.get("diagnostics")
    if type(diagnostics) is not dict:
        raise ControlError("native latent-bank diagnostics are malformed")
    if (native.get("reference_manifest_sha256") != REFERENCE_MANIFEST_SHA256
            or native.get("reference_image_sha256") != PHOTO_SHA256
            or native.get("reference_payload_sha256") != PREPARED_RGB8_SHA256
            or native.get("reference_encoder_sha256") != ENCODER_SHA256
            or diagnostics.get("latent_bank_sha256") != NATIVE_BANK_SHA256):
        raise ControlError("native reference photo or latent-bank lineage mismatch")

    if dcae.get("schema") != "cogni_ml.hy_nomas_portrait_dcae_control/v1":
        raise ControlError("upstream DCAE T0 receipt schema mismatch")
    if (dcae.get("dcae_module_history_revision") != DCAE_MODULE_REVISION
            or dcae.get("dcae_module_sha256") != DCAE_MODULE_SHA256
            or dcae.get("recovered_encoder_sha256") != ENCODER_SHA256):
        raise ControlError("upstream DCAE encoder source lineage mismatch")
    if (dcae.get("reference_manifest_sha256") != REFERENCE_MANIFEST_SHA256
            or dcae.get("reference_payload_sha256") != PREPARED_RGB8_SHA256
            or dcae.get("native_receipt_sha256") != NATIVE_RECEIPT_SHA256
            or dcae.get("native_latent_bank_sha256") != NATIVE_BANK_SHA256):
        raise ControlError("upstream DCAE T0 receipt photo/native lineage mismatch")
    runtime, input_info, latent = dcae.get("runtime"), dcae.get("input"), dcae.get("latent")
    if any(type(value) is not dict for value in (runtime, input_info, latent)):
        raise ControlError("upstream DCAE T0 runtime/input/latent records are malformed")
    if (runtime != {"torch": "2.8.0", "device": "mps", "dtype": "BF16",
                    "mps_fallback": False, "autocast": False}
            or input_info.get("shape") != list(REFERENCE_SHAPE)
            or input_info.get("dtype") != "BF16"
            or input_info.get("temporal_feature_cache") is not True
            or input_info.get("tile_frames") != 40):
        raise ControlError("upstream DCAE T0 encode runtime/input contract mismatch")
    if (latent.get("shape") != list(LATENT_SHAPE) or latent.get("dtype") != "BF16"
            or latent.get("export_dtype") != "F32"
            or latent.get("sha256") != T0_LATENT_SHA256
            or latent_sha256 != T0_LATENT_SHA256):
        raise ControlError("upstream DCAE T0 latent identity mismatch")
    if latent_bytes != T0_LATENT_BYTES:
        raise ControlError("upstream DCAE T0 latent extent mismatch")
    comparison = dcae.get("comparison")
    if type(comparison) is not dict:
        raise ControlError("upstream DCAE T0 comparison record is malformed")
    if (comparison.get("atol") != ATOL or comparison.get("rtol") != RTOL
            or comparison.get("values") != math.prod(LATENT_SHAPE)
            or comparison.get("bitwise_equal") is not True
            or comparison.get("max_abs") != 0.0
            or comparison.get("unequal_f32_values") != 0):
        raise ControlError("upstream encoder receipt does not prove exact native T0 parity")
    if (dcae.get("intermediate_readbacks") != 0
            or dcae.get("explicit_mps_synchronize_calls") != 0
            or dcae.get("terminal_transfer_count") != 1):
        raise ControlError("upstream T0 encoder receipt violates the terminal-transfer boundary")


def validate_decoder_metadata(
    actual: Mapping[str, Mapping[str, Any]], expected_shapes: Mapping[str, list[int] | tuple[int, ...]],
) -> dict[str, int]:
    """Validate a CPU-mmap checkpoint's decoder keys without copying tensor data."""
    if set(actual) != set(expected_shapes):
        raise ControlError("decoder checkpoint tensor inventory mismatch")
    total_bytes = 0
    for name, expected in expected_shapes.items():
        record = actual[name]
        shape = _positive_shape(record.get("shape"), f"decoder tensor {name}")
        if shape != _positive_shape(expected, f"upstream tensor {name}"):
            raise ControlError(f"decoder checkpoint shape mismatch: {name}")
        if record.get("dtype") != "BF16":
            raise ControlError(f"decoder checkpoint tensor is not BF16: {name}")
        if record.get("device") != "cpu" or record.get("contiguous") is not True:
            raise ControlError(f"decoder checkpoint tensor is not contiguous CPU storage: {name}")
        total_bytes += math.prod(shape) * 2
    return {"tensor_count": len(actual), "bf16_bytes": total_bytes}


def _regular_path(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ControlError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ControlError("cannot identify the pinned upstream checkout") from None


def _read_checkpoint_metadata(weights_path: Path, module_path: Path, torch: Any) -> tuple[dict, dict[str, Any]]:
    """Load only CPU-mmap metadata and derive exact decoder keys from meta tensors."""
    upstream = reference._load_module(module_path)
    config = upstream.dc_ae_v_f32t4c32_config()
    if (config.latent_channels != 32 or config.decode_temporal_tile_size != TILE_FRAMES
            or config.decode_temporal_tile_latent_size != TILE_LATENTS
            or config.decode_temporal_tile_overlap_factor != 0
            or config.use_feature_cache is not True or config.num_pad_frames != 0):
        raise ControlError("upstream DCAE decode tile/cache config differs from the admitted geometry")
    with torch.device("meta"):
        decoder = upstream.DCAEVDecoder(config.decoder)
    expected_shapes = {
        f"decoder.{name}": [int(dim) for dim in tensor.shape]
        for name, tensor in decoder.state_dict().items()
    }
    if len(expected_shapes) != DECODER_TENSOR_COUNT:
        raise ControlError("upstream DCAE decoder metadata tensor count mismatch")
    if sum(math.prod(shape) * 2 for shape in expected_shapes.values()) != DECODER_BF16_BYTES:
        raise ControlError("upstream DCAE decoder metadata byte count mismatch")

    try:
        checkpoint_object = torch.load(
            str(weights_path), map_location="cpu", weights_only=True, mmap=True,
        )
        state, _selected = checkpoint._select_state_dict(checkpoint_object)
    except Exception as exc:
        raise ControlError(f"cannot safely mmap original DCAE checkpoint metadata: {exc}") from None
    decoder_state: dict[str, Any] = {}
    encoder_count = 0
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ControlError("original DCAE checkpoint contains a non-tensor state entry")
        if name.startswith("decoder."):
            decoder_state[name] = tensor
        elif name.startswith("encoder."):
            encoder_count += 1
        else:
            raise ControlError(f"original DCAE checkpoint has unknown state ownership: {name}")
    if encoder_count != ENCODER_TENSOR_COUNT:
        raise ControlError("original DCAE checkpoint encoder inventory differs from pinned source")
    metadata = {
        name: {"shape": list(tensor.shape), "dtype": "BF16" if tensor.dtype is torch.bfloat16 else str(tensor.dtype),
               "device": tensor.device.type, "contiguous": tensor.is_contiguous()}
        for name, tensor in decoder_state.items()
    }
    decoder_report = validate_decoder_metadata(metadata, expected_shapes)
    if decoder_report["bf16_bytes"] != DECODER_BF16_BYTES:
        raise ControlError("original decoder BF16 payload byte count mismatch")
    return decoder_state, {
        "config": config,
        "expected_shapes": expected_shapes,
        "decoder_report": decoder_report,
        "upstream": upstream,
    }


def _validate_native_prefix(bank_path: Path, latent_path: Path) -> None:
    """Compare contiguous channel prefixes without materializing an array."""
    spatial_frame_bytes = math.prod(NATIVE_BANK_SHAPE[3:]) * 4
    bank_channel_bytes = NATIVE_BANK_SHAPE[2] * spatial_frame_bytes
    latent_channel_bytes = LATENT_SHAPE[2] * spatial_frame_bytes
    with bank_path.open("rb") as bank, latent_path.open("rb") as latent:
        for channel in range(NATIVE_BANK_SHAPE[1]):
            bank.seek(channel * bank_channel_bytes)
            native_prefix = bank.read(latent_channel_bytes)
            upstream_prefix = latent.read(latent_channel_bytes)
            if len(native_prefix) != latent_channel_bytes or native_prefix != upstream_prefix:
                raise ControlError(f"upstream T0 latent differs from native clean T0 channel {channel}")
        if latent.read(1):
            raise ControlError("upstream T0 latent has trailing bytes")


def admit(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    root = Path(args.input_root).expanduser().resolve(strict=True)
    photo_path = _regular_path(args.photo, "original photo")
    manifest_path = _regular_path(root / "photo-reference/manifest.json", "photo manifest")
    manifest_raw = _read_regular(manifest_path, "photo manifest", max_bytes=128 * 1024)
    manifest = json.loads(manifest_raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if type(manifest) is not dict:
        raise ControlError("photo manifest root must be an object")
    image_sha256 = _sha256_file(photo_path)
    payload_path = _regular_path(root / "photo-reference/pixels.rgb8", "prepared RGB8 payload")
    payload_size = payload_path.stat().st_size
    payload_sha256 = _sha256_file(payload_path)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    validate_photo_manifest(manifest, manifest_sha256, image_sha256, payload_size, payload_sha256)
    if photo_path.stat().st_size != PHOTO_BYTES:
        raise ControlError("original photo byte extent mismatch")

    native_path = _regular_path(root / "photo-native-v2/receipt.json", "native receipt")
    native_raw = _read_regular(native_path, "native receipt", max_bytes=1 * 1024 * 1024)
    _require_digest(hashlib.sha256(native_raw).hexdigest(), NATIVE_RECEIPT_SHA256, "native receipt")
    native = json.loads(native_raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if type(native) is not dict:
        raise ControlError("native receipt root must be an object")
    dcae_path = _regular_path(root / "photo-upstream-dcae-t0-v1/receipt.json", "upstream DCAE T0 receipt")
    dcae, dcae_raw = _read_json(dcae_path, "upstream DCAE T0 receipt", max_bytes=128 * 1024)
    dcae_sha256 = hashlib.sha256(dcae_raw).hexdigest()
    _require_digest(dcae_sha256, T0_RECEIPT_SHA256, "upstream DCAE T0 receipt")
    latent_path = _regular_path(root / "photo-upstream-dcae-t0-v1/latent.f32le", "upstream DCAE T0 latent")
    latent_sha256 = _sha256_file(latent_path)
    validate_t0_lineage(
        native, dcae, latent_bytes=latent_path.stat().st_size,
        latent_sha256=latent_sha256, native_receipt_sha256=NATIVE_RECEIPT_SHA256,
    )
    bank_path = _regular_path(root / "photo-native-v2/latent-bank.f32le", "native latent bank")
    bank_size = math.prod(NATIVE_BANK_SHAPE) * 4
    if bank_path.stat().st_size != bank_size:
        raise ControlError("native latent bank extent mismatch")
    bank_sha256 = _sha256_file(bank_path)
    _require_digest(bank_sha256, NATIVE_BANK_SHA256, "native latent bank")
    _validate_native_prefix(bank_path, latent_path)

    upstream_root = Path(args.upstream).expanduser().resolve(strict=True)
    if _git_head(upstream_root) != UPSTREAM_REVISION:
        raise ControlError("upstream checkout revision mismatch")
    module_path = _regular_path(upstream_root / "hymm/models/vae/autoencoder_dcae_v.py", "upstream DCAE module")
    module_sha256 = _sha256_file(module_path)
    _require_digest(module_sha256, DCAE_MODULE_SHA256, "upstream DCAE module")
    weights_path = _regular_path(args.weights, "original BF16 DCAE checkpoint")
    if weights_path.stat().st_size != DCAE_CHECKPOINT_BYTES:
        raise ControlError("original BF16 DCAE checkpoint extent mismatch")
    weights_sha256 = _sha256_file(weights_path)
    _require_digest(weights_sha256, DCAE_CHECKPOINT_SHA256, "original BF16 DCAE checkpoint")

    decoder_state, schema = _read_checkpoint_metadata(weights_path, module_path, torch)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise ControlError("output-dir must be a fresh absolute path")
    parent = output_dir.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ControlError("output-dir parent must be a directory")
    output_dir = parent / output_dir.name

    # This is a named-byte inventory, not a peak guarantee: MPSGraph/workspace
    # allocations and source module temporaries are opaque and excluded.
    named_bytes = (
        DECODER_BF16_BYTES
        + math.prod(LATENT_SHAPE) * 2
        + 2 * math.prod(REFERENCE_SHAPE) * 2  # full BF16 tiles and joined BF16 output
        + math.prod(REFERENCE_SHAPE) * 4       # F32 compare/render staging
        + math.prod(REFERENCE_SHAPE[2:]) * 3   # terminal RGB8 buffer
    )
    return {
        "root": root, "photo_path": photo_path, "photo_sha256": image_sha256,
        "manifest_path": manifest_path, "manifest_sha256": manifest_sha256,
        "payload_path": payload_path, "payload_sha256": payload_sha256,
        "native_receipt_path": native_path, "native_receipt_sha256": NATIVE_RECEIPT_SHA256,
        "native_bank_path": bank_path, "native_bank_sha256": bank_sha256,
        "dcae_receipt_path": dcae_path, "dcae_receipt_sha256": dcae_sha256,
        "latent_path": latent_path, "latent_sha256": latent_sha256,
        "upstream_root": upstream_root, "module_path": module_path,
        "module_sha256": module_sha256, "weights_path": weights_path,
        "weights_sha256": weights_sha256, "decoder_state": decoder_state,
        "schema": schema, "output_dir": output_dir,
        "schema_inspection_torch_version": torch.__version__,
        "memory_estimate": {
            "named_bytes": named_bytes, "named_gib": named_bytes / GIB,
            "decoder_bf16_weight_bytes": DECODER_BF16_BYTES,
            "external_cap_bytes": EXTERNAL_CAP_BYTES,
            "physical_reserve_fraction": RESERVE_FRACTION,
            "external_guard_required": True,
            "opaque_mps_workspace_included": False,
        },
    }


def validate_decode_geometry(shape: object, render_start_frame: object) -> tuple[int, ...]:
    """Admit clean T0 or context-preserving T0+T1, rendering exactly T1."""
    latent_shape = _positive_shape(shape, "decode latent")
    if type(render_start_frame) is not int:
        raise ControlError("decode render start must be an integer")
    expected_start = {
        LATENT_SHAPE: 0,
        (1, 32, 60, 9, 16): 120,
    }.get(tuple(latent_shape))
    if expected_start is None or render_start_frame != expected_start:
        raise ControlError("decode must preserve the admitted T0/T1 context and render range")
    return (1, 3, latent_shape[2] * 4, 288, 512)


def execute(admitted: dict[str, Any], torch: Any) -> dict[str, Any]:
    import numpy as np

    if torch.__version__ != "2.8.0" or not torch.backends.mps.is_available():
        raise ControlError("execution requires the pinned Torch 2.8.0 MPS runtime")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ControlError("PYTORCH_ENABLE_MPS_FALLBACK must be exactly '0'")
    if not torch.backends.mps.is_built():
        raise ControlError("this Torch build has no MPS backend")
    if admitted["weights_path"].stat().st_size != DCAE_CHECKPOINT_BYTES:
        raise ControlError("original checkpoint changed after admission")

    decode_input = admitted.get("decode_input", {
        "path": admitted["latent_path"], "sha256": admitted["latent_sha256"],
        "shape": list(LATENT_SHAPE), "render_start_frame": 0,
        "scope": "authenticated clean original-photo T0; no DiT",
    })
    latent_shape = tuple(decode_input["shape"])
    render_start = decode_input["render_start_frame"]
    full_output_shape = validate_decode_geometry(latent_shape, render_start)
    pixel_reference = admitted.get("pixelumm_reference")
    prefix_control = admitted.get("prefix_control")
    if latent_shape[2] == 60:
        if (not isinstance(pixel_reference, dict) or not isinstance(prefix_control, dict)
                or pixel_reference.get("shape") != [1, 3, 240, 288, 512]
                or pixel_reference.get("crop") != [120, 240]):
            raise ControlError("paired decode requires authenticated PixelUMM and causal-prefix controls")
    elif pixel_reference is not None or prefix_control is not None:
        raise ControlError("clean T0 decode cannot use final-endpoint comparison controls")
    latent_np = np.fromfile(decode_input["path"], dtype="<f4")
    if latent_np.size != math.prod(latent_shape):
        raise ControlError("decode latent extent changed before execution")
    _require_digest(hashlib.sha256(latent_np.tobytes()).hexdigest(),
                    decode_input["sha256"], "decode latent before execution")
    if pixel_reference is not None:
        pixel_path = _regular_path(pixel_reference["path"], "PixelUMM reference pixels")
        if pixel_path.stat().st_size != math.prod(pixel_reference["shape"]) * 4:
            raise ControlError("PixelUMM reference pixel extent changed before execution")
        # Pin the bytes used for upload before allocating the GPU model. Hashing
        # a path and reopening it later would permit a different reference crop.
        pixel_bytes = _read_regular(pixel_path, "PixelUMM reference pixels",
                                    max_bytes=math.prod(pixel_reference["shape"]) * 4)
        _require_digest(hashlib.sha256(pixel_bytes).hexdigest(), pixel_reference["sha256"],
                        "PixelUMM reference pixels before execution")
        pixels = np.frombuffer(pixel_bytes, dtype="<f4").reshape(pixel_reference["shape"])
        selected_pixels = pixels[:, :, 120:240].copy()
        del pixels, pixel_bytes

    upstream = admitted["schema"]["upstream"]
    config = admitted["schema"]["config"]
    with torch.device("meta"):
        model = upstream.DCAEV(config)
    model.decoder.to(dtype=torch.bfloat16)
    model.decoder.to_empty(device="mps")
    local_decoder = {
        name.removeprefix("decoder."): tensor
        for name, tensor in admitted["decoder_state"].items()
    }
    model.decoder.load_state_dict(local_decoder, strict=True)
    del local_decoder
    model.eval()

    latent = torch.from_numpy(latent_np.reshape(latent_shape).copy())
    latent = latent.to(device="mps", dtype=torch.bfloat16).contiguous()
    if tuple(latent.shape) != latent_shape:
        raise ControlError("decode latent tensor shape changed during load")

    if pixel_reference is not None:
        source = torch.from_numpy(selected_pixels).to(device="mps", dtype=torch.float32)
    else:
        rgb8 = np.fromfile(admitted["payload_path"], dtype=np.uint8)
        if rgb8.size != REFERENCE_RGB8_BYTES:
            raise ControlError("prepared RGB8 extent changed before execution")
        source = torch.from_numpy(rgb8.reshape(288, 512, 3).copy()).permute(2, 0, 1)
        source = source.to(device="mps", dtype=torch.bfloat16).div_(127.5).sub_(1)
        source = source[None, :, None].contiguous()

    started = time.monotonic()
    with torch.no_grad():
        decoded = model.decode(latent)
    decode_enqueue_seconds = time.monotonic() - started
    if tuple(decoded.shape) != full_output_shape or decoded.dtype is not torch.bfloat16:
        raise ControlError(f"DCAE decoder output violates geometry/dtype: {tuple(decoded.shape)}, {decoded.dtype}")
    prefix_display = None
    if prefix_control is not None:
        # Reproduce the earlier clean-T0 render in this same terminal pack.
        # It is a causal-prefix regression check, not another model request.
        prefix_display = decoded[:, :, :120].float().clamp(-1.0, 1.0).add_(1.0).mul_(127.5)
        prefix_display = prefix_display.round_().clamp_(0, 255)
        prefix_display = prefix_display[0].permute(1, 2, 3, 0).contiguous().to(torch.uint8).reshape(-1)
    # Keep the source feature cache across every T0/T1 tile. Select the display
    # interval only after the complete upstream decode, without a readback.
    decoded = decoded[:, :, render_start:render_start + REFERENCE_SHAPE[2]]

    # Comparison stays in F32 on MPS. One HWC RGB8 render and these scalars are
    # packed as bytes for the sole MPS->CPU transfer; no .item() or sync occurs.
    # The paired control uses exactly the production BF16 display boundary.
    # Preserve the existing normalized [-1,1] T0-fidelity path otherwise.
    decoded_f32 = (decoded.mul(0.5).add(0.5).clamp(0, 1).float()
                   if pixel_reference is not None else decoded.float())
    source_f32 = source.float()
    delta = (decoded_f32 - source_f32).abs()
    if pixel_reference is not None:
        third_metric = delta.mean()
    else:
        limit = ATOL + RTOL * source_f32.abs()
        third_metric = (delta > limit).sum().to(torch.float32)
    metrics = torch.stack((
        delta.max(), delta.square().mean().sqrt(),
        third_metric,
        torch.isfinite(decoded_f32).all().to(torch.float32),
    )).contiguous()
    if pixel_reference is not None:
        display = decoded_f32.mul(255).round_().clamp_(0, 255)
    else:
        display = decoded_f32.clamp(-1.0, 1.0).add_(1.0).mul_(127.5).round_().clamp_(0, 255)
    display = display[0].permute(1, 2, 3, 0).contiguous().to(torch.uint8).reshape(-1)
    parts = [display]
    if prefix_display is not None:
        parts.append(prefix_display)
    parts.append(metrics.view(torch.uint8))
    packed = torch.cat(parts).contiguous().cpu().numpy()
    through_terminal_seconds = time.monotonic() - started
    display_bytes = math.prod(REFERENCE_SHAPE[2:]) * 3
    rendered = packed[:display_bytes]
    metrics_start = display_bytes
    prefix_result = None
    if prefix_display is not None:
        prefix_payload = packed[display_bytes:2 * display_bytes].tobytes()
        prefix_sha = hashlib.sha256(prefix_payload).hexdigest()
        _require_digest(prefix_sha, prefix_control["expected_rgb8_sha256"], "decoded causal T0 prefix")
        prefix_result = {"frames": [0, 120], "rgb8_sha256": prefix_sha,
                         "bitwise_equal_to_prior_clean_dcae": True,
                         "scope": "legacy F32-normalized render, same six-tile decoder request"}
        metrics_start += display_bytes
    metric_values = packed[metrics_start:].view("<f4")
    if rendered.size != display_bytes or metric_values.size != 4:
        raise ControlError("terminal packed result has an invalid extent")
    if not np.isfinite(metric_values).all() or metric_values[3] != 1.0:
        raise ControlError("original DCAE reconstruction or metrics are non-finite")

    output_dir = admitted["output_dir"]
    if output_dir.exists() or output_dir.is_symlink():
        raise ControlError("output-dir ceased to be fresh during execution")
    output_dir.mkdir()
    image_path = output_dir / "reconstruction.rgb8"
    payload = rendered.tobytes()
    with image_path.open("xb") as stream:
        stream.write(payload)
    comparison = {
        "reference": "prepared original photo RGB8 normalized to BF16 [-1,1], repeated across 120 frames",
        "atol": ATOL, "rtol": RTOL,
        "max_abs": float(metric_values[0]), "rmse": float(metric_values[1]),
        "outside_tolerance": int(metric_values[2]), "values": math.prod(REFERENCE_SHAPE),
    }
    if pixel_reference is not None:
        comparison = {
            "reference": "authenticated native PixelUMM T1 display frames [120,240)",
            "reference_pixels_sha256": pixel_reference["sha256"],
            "representation": "F32 planar RGB [0,1] after BF16 multiply/add/clamp",
            "scope": "descriptive cross-decoder difference; no equality, likeness or quality gate",
            "max_abs": float(metric_values[0]), "rmse": float(metric_values[1]),
            "mean_abs": float(metric_values[2]), "values": math.prod(REFERENCE_SHAPE),
            "all_values_finite": True,
        }
    result = {
        "schema": admitted.get("output_schema", "cogni_ml.hy_nomas_photo_dcae_decode/v1"),
        "upstream_revision": UPSTREAM_REVISION,
        "dcae_module_history_revision": DCAE_MODULE_REVISION,
        "dcae_module_sha256": admitted["module_sha256"],
        "checkpoint_sha256": admitted["weights_sha256"],
        "photo_sha256": admitted["photo_sha256"],
        "reference_manifest_sha256": admitted["manifest_sha256"],
        "prepared_rgb8_sha256": admitted["payload_sha256"],
        "native_receipt_sha256": admitted["native_receipt_sha256"],
        "native_latent_bank_sha256": admitted["native_bank_sha256"],
        "upstream_t0_receipt_sha256": admitted["dcae_receipt_sha256"],
        "t0_latent_sha256": admitted["latent_sha256"],
        "runtime": {"torch": torch.__version__, "device": "mps", "dtype": "BF16",
                    "mps_fallback": False, "autocast": False},
        "decode": {"input_shape": list(latent_shape), "input_sha256": decode_input["sha256"],
                   "full_output_shape": list(full_output_shape),
                   "output_shape": list(REFERENCE_SHAPE), "render_start_frame": render_start,
                   "scope": decode_input["scope"],
                   "tile_latents": TILE_LATENTS, "tile_frames": TILE_FRAMES,
                   "overlap_latents": 0, "temporal_feature_cache": True,
                   "tiles": latent_shape[2] // TILE_LATENTS, "decoder_weights_only": True},
        "comparison": comparison,
        "render": {"path": str(image_path), "shape": [120, 288, 512, 3],
                   "dtype": "RGB8", "conversion": (
                       "BF16 x*0.5+0.5, clamp [0,1], F32 multiply by255, round"
                       if pixel_reference is not None else "clamp [-1,1], map to [0,255], round"),
                   "payload_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest()},
        "terminal_transfer_count": 1,
        "intermediate_readbacks": 0,
        "explicit_mps_synchronize_calls": 0,
        "seconds_decode_enqueue": decode_enqueue_seconds,
        "seconds_through_terminal": through_terminal_seconds,
        "memory_estimate": admitted["memory_estimate"],
    }
    if "counterfactual" in admitted:
        result["counterfactual"] = admitted["counterfactual"]
    if prefix_result is not None:
        result["causal_prefix_control"] = prefix_result
    (output_dir / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--photo", type=Path, default=DEFAULT_PHOTO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="authenticate CPU metadata; do not run MPS")
    mode.add_argument("--execute", action="store_true", help="run only the original DCAE decoder on MPS")
    args = parser.parse_args()
    try:
        import torch

        admitted = admit(args, torch)
        if args.execute:
            result = execute(admitted, torch)
        else:
            result = {
                "validated": True, "model_launched": False,
                "upstream_revision": UPSTREAM_REVISION,
                "dcae_module_sha256": admitted["module_sha256"],
                "checkpoint_sha256": admitted["weights_sha256"],
                "decoder_schema": admitted["schema"]["decoder_report"],
                "schema_inspection_torch_version": admitted["schema_inspection_torch_version"],
                "photo_sha256": admitted["photo_sha256"],
                "prepared_rgb8_sha256": admitted["payload_sha256"],
                "native_receipt_sha256": admitted["native_receipt_sha256"],
                "t0_latent_sha256": admitted["latent_sha256"],
                "memory_estimate": admitted["memory_estimate"],
                "output_dir": str(admitted["output_dir"]),
            }
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
