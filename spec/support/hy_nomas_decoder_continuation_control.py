#!/usr/bin/env python3
"""Compare the saved 60-latent native decode against pinned upstream PixelUMM.

This diagnostic consumes an authenticated native terminal latent and the first
240 frames of its authenticated 720-frame continuation noise record. It runs
all eleven overlapping windows through the pinned 36-block decoder and head,
then compares the complete display-normalized F32 pixel clip. Run ``--execute``
beneath ``examples/support/wan_memory_guard.py``; default mode performs CPU admission
and prints the bounded plan without constructing model modules or using MPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

SUPPORT_DIR = Path(__file__).resolve().parent
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

import hy_nomas_continuation_inputs as continuation_inputs
import hy_nomas_decoder_boundary as decoder_boundary
import hy_nomas_decoder_same_latent as same_latent

torch = same_latent.torch
block_reference = same_latent.block_reference
cache_inventory = same_latent.cache_inventory
checkpoint_inventory = same_latent.checkpoint_inventory
head_reference = same_latent.head_reference
head_checkpoint = same_latent.head_checkpoint
real_input = same_latent.real_input

SCHEMA = "cogni_ml.hy_nomas_decoder_continuation_control/v2"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_decode_latent/v1"
UPSTREAM_REVISION = block_reference.UPSTREAM_REVISION
WIDTH = 512
HEIGHT = 288
FPS = 24
LATENT_SHAPE = (1, 32, 60, 9, 16)
LATENT_BANK_SHAPE = (1, 32, 180, 9, 16)
NOISE_SOURCE_SHAPE = (1, 3, 720, HEIGHT, WIDTH)
NOISE_PREFIX_SHAPE = (1, 3, 240, HEIGHT, WIDTH)
RAW_SHAPE = NOISE_PREFIX_SHAPE
WINDOW_LATENTS = 10
OVERLAP_LATENTS = 5
WINDOW_STARTS = tuple(range(0, LATENT_SHAPE[2] - WINDOW_LATENTS + 1, OVERLAP_LATENTS))
LATENT_BYTES = math.prod(LATENT_SHAPE) * 4
LATENT_BANK_BYTES = math.prod(LATENT_BANK_SHAPE) * 4
NOISE_SOURCE_BYTES = math.prod(NOISE_SOURCE_SHAPE) * 4
RAW_BYTES = math.prod(RAW_SHAPE) * 4
GIB = 1024**3
MAX_PROCESS_ESTIMATE_BYTES = 25 * GIB + GIB // 2
MULTI_WINDOW_RESERVE_BYTES = GIB
# Anchored to the prior same-latent one-window full36 estimate. Execute mode
# replaces this planning value with the live upstream geometry estimate.
PLAN_BASE_PEAK_BYTES = 5_327_865_344
PLAN_CACHE_BYTES = 13_892_143_104
ATOL = RTOL = 2e-5
METRIC_CHUNK_ELEMENTS = 1 << 20
EXPECTED_LATENT_BOUNDARY = (
    "caller-supplied terminal diagnostic latent bank; any shorter decode prefix "
    "is gathered on GPU; not produced by this request's DiT or solver"
)
STANDALONE_LATENT_SELECTION = "authenticated latent bank global frames [0, 60) consumed directly"
PREFIX_LATENT_SELECTION = (
    "global latent frames [0, 60) from each NCTHW channel using "
    "HyNomasTimelineOps::PrefixPlan"
)
EXPECTED_NOISE_SELECTION = (
    "global pixel-noise frames [0, 240) from each NCTHW channel using "
    "HyNomasTimelineOps::PrefixPlan"
)
EXPECTED_PRECISION = (
    "BF16 decoder block/head weights; F32 stem weights and native activation/MPS "
    "arithmetic; original fixed noise and PixelUMM path"
)
EXPECTED_NATIVE_CACHE_SHA256 = "74ee5e84f2b63af0189ffd7be0e4c758b8dfa9ff925e3c9cb389d207ce546dba"
EXPECTED_NATIVE_STEM_SHA256 = "7eb73ebf42f6681e21870dc2d2d4b592846ba15ed5bf5f9a6c7a83d39c2d8d04"
EXPECTED_NATIVE_HEAD_SHA256 = "d0e985e45b6b5fcebde381dd4b61f3b77394bd288323cd8afe2a7604584e6e9f"
EXPECTED_NATIVE_EXECUTABLE_SHA256 = "99385cc63bc536c02553ea5fe32ffff8d8aeff95407f89e12540376e62282f80"
NATIVE_DISPLAY_REPRESENTATION = (
    "F32 planar RGB display pixels in [0,1] after native BF16-rounded multiply, add, and clamp"
)


class DecoderControlError(ValueError):
    """Receipt, continuation bundle, decoder source, or MPS admission failed."""


class ControlGeometry:
    """The fixed upstream window and whole-clip shapes for this control."""

    resolution = "512x288"
    width = WIDTH
    height = HEIGHT
    latent_height = 9
    latent_width = 16

    @property
    def token_grid(self) -> tuple[int, int, int]:
        return (10, 2 * self.latent_height, 2 * self.latent_width)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return LATENT_SHAPE

    @property
    def raw_shape(self) -> tuple[int, ...]:
        return RAW_SHAPE

    @property
    def first_window_shape(self) -> tuple[int, ...]:
        return (1, 3, WINDOW_LATENTS * 4, HEIGHT, WIDTH)

    @property
    def first_prefix_elements(self) -> int:
        return 20 * 3 * HEIGHT * WIDTH

    @property
    def latent_bytes(self) -> int:
        return LATENT_BYTES

    @property
    def raw_bytes(self) -> int:
        return RAW_BYTES


GEOMETRY = ControlGeometry()


def _digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DecoderControlError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticate_native_executable(path_value: Path, expected_sha256: str) -> Path:
    """Bind native output semantics to the exact executable used by its receipt."""
    if expected_sha256 != EXPECTED_NATIVE_EXECUTABLE_SHA256:
        raise DecoderControlError("native executable digest is not the pinned decoder")
    path = same_latent._canonical_file(path_value, "native decode executable")
    if _sha256_file(path) != EXPECTED_NATIVE_EXECUTABLE_SHA256:
        raise DecoderControlError("native decode executable bytes differ from the pinned decoder")
    return path


def _validate_display_postprocess_source() -> dict[str, str]:
    """Verify the upstream source that defines the BF16 display conversion."""
    reference = same_latent.pixelumm_reference
    try:
        reference._verify_upstream_files(same_latent.UPSTREAM)
    except reference.ReferenceError as error:
        raise DecoderControlError(f"pinned display postprocess source rejected: {error}") from error
    relative = reference.POSTPROCESS_PATH
    return {"path": relative, "sha256": reference.SOURCE_SHA256[relative]}


def _display_pixels_from_bf16(raw):
    """Apply the pinned BF16 multiply/add/clamp conversion to a raw source tensor."""
    if raw.dtype is not torch.bfloat16:
        raise DecoderControlError("upstream raw decoder result must be an actual BF16 tensor")
    return same_latent.pixelumm_reference.postprocess_bf16(torch, raw)


def _queue_display_comparison(upstream_raw_bf16, native_display_f32, *, geometry=GEOMETRY):
    """Queue pinned display conversion and chunked metrics without host readback."""
    if tuple(upstream_raw_bf16.shape) != geometry.raw_shape:
        raise DecoderControlError("upstream raw decoder result geometry differs from the display clip")
    if native_display_f32.dtype is not torch.float32:
        raise DecoderControlError("native display comparison reference must be F32")
    if tuple(native_display_f32.shape) != geometry.raw_shape:
        raise DecoderControlError("native display comparison reference geometry mismatch")
    if native_display_f32.device != upstream_raw_bf16.device:
        raise DecoderControlError("native display and upstream raw tensors must share a device")
    upstream_display = _display_pixels_from_bf16(upstream_raw_bf16)
    display_metrics = same_latent._queue_pixel_comparison(
        upstream_display, native_display_f32, geometry=geometry,
    )
    raw_finite = torch.isfinite(upstream_raw_bf16).all()
    return upstream_display, (*display_metrics, raw_finite)


def _read_json(path: Path, what: str, *, max_bytes: int = 2 * 1024 * 1024) -> tuple[dict, bytes]:
    try:
        path = same_latent._canonical_file(path, what)
        raw = path.read_bytes()
    except (OSError, same_latent.SameLatentError) as error:
        raise DecoderControlError(str(error)) from error
    if len(raw) > max_bytes:
        raise DecoderControlError(f"{what} exceeds the metadata size limit")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecoderControlError(f"cannot parse {what}") from error
    if type(document) is not dict:
        raise DecoderControlError(f"{what} must be a JSON object")
    return document, raw


def _require_int(document: Mapping, field: str, expected: int) -> None:
    if type(document.get(field)) is not int or document[field] != expected:
        raise DecoderControlError(f"native decode receipt {field} does not match admitted geometry")


def _require_shape(document: Mapping, field: str, expected: tuple[int, ...]) -> tuple[int, ...]:
    value = document.get(field)
    if (type(value) is not list or len(value) != len(expected)
            or any(type(dimension) is not int for dimension in value)
            or tuple(value) != expected):
        raise DecoderControlError(f"native decode receipt {field} mismatch")
    return tuple(value)


def validate_native_decode_receipt_document(document: Mapping) -> dict[str, object]:
    """Admit the fixed decode contract and derive output semantics from its pinned binary."""
    if type(document) is not dict or document.get("schema") != NATIVE_SCHEMA:
        raise DecoderControlError("unsupported native decode receipt schema")
    if document.get("upstream_revision") != UPSTREAM_REVISION:
        raise DecoderControlError("native decode receipt upstream revision mismatch")
    if document.get("full_upstream_video_parity") is not False:
        raise DecoderControlError("native receipt must declare full upstream video parity=false")
    if document.get("precision") != EXPECTED_PRECISION:
        raise DecoderControlError("native decode receipt precision contract is unsupported")
    executable = document.get("executable")
    if not isinstance(executable, str) or not Path(executable).is_absolute():
        raise DecoderControlError("native decode receipt executable path must be absolute")
    if document.get("executable_sha256") != EXPECTED_NATIVE_EXECUTABLE_SHA256:
        raise DecoderControlError("native decode receipt executable SHA-256 is not the pinned decoder")
    if "output_range" in document or "output_representation" in document:
        raise DecoderControlError(
            "native output representation is derived from the pinned native executable, not a receipt claim"
        )
    for field, expected in (
        ("latent_frames", 60), ("latent_selected_frames", 60),
        ("latent_prefix_start_frame", 0),
        ("frames", 240), ("width", WIDTH), ("height", HEIGHT), ("fps", FPS),
        ("pixel_noise_frames", 240), ("pixel_noise_prefix_start_frame", 0),
        ("output_bytes", RAW_BYTES),
    ):
        _require_int(document, field, expected)
    _require_shape(document, "latent_shape", LATENT_SHAPE)
    source_bank_value = document.get("latent_bank_shape")
    if type(source_bank_value) is list and all(type(dimension) is int for dimension in source_bank_value) \
            and tuple(source_bank_value) == LATENT_SHAPE:
        source_bank_shape_tuple = LATENT_SHAPE
        source_bank_bytes = LATENT_BYTES
        _require_int(document, "latent_bytes", LATENT_BYTES)
        _require_int(document, "latent_bank_bytes", LATENT_BYTES)
        _require_int(document, "latent_bank_frames", 60)
        if document.get("latent_selection") != STANDALONE_LATENT_SELECTION:
            raise DecoderControlError(
                "native latent selection is not the authenticated 60-frame terminal bank"
            )
        gpu_gather = False
    elif type(source_bank_value) is list and all(type(dimension) is int for dimension in source_bank_value) \
            and tuple(source_bank_value) == LATENT_BANK_SHAPE:
        source_bank_shape_tuple = LATENT_BANK_SHAPE
        source_bank_bytes = LATENT_BANK_BYTES
        _require_int(document, "latent_bytes", LATENT_BANK_BYTES)
        _require_int(document, "latent_bank_bytes", LATENT_BANK_BYTES)
        _require_int(document, "latent_bank_frames", 180)
        if document.get("latent_selection") != PREFIX_LATENT_SELECTION:
            raise DecoderControlError(
                "native latent selection is not the admitted global 60-frame prefix"
            )
        metrics = document.get("metrics")
        if type(metrics) is not dict or type(metrics.get("latent_prefix_gpu_gathered")) is not int \
                or metrics["latent_prefix_gpu_gathered"] != 1:
            raise DecoderControlError(
                "native decode receipt must attest latent_prefix_gpu_gathered=1 for a bank prefix"
            )
        gpu_gather = True
    else:
        raise DecoderControlError("native decode receipt latent_bank_shape mismatch")
    if document.get("latent_dtype") != "F32 little-endian":
        raise DecoderControlError("native latent dtype is not F32 little-endian")
    if document.get("latent_boundary") != EXPECTED_LATENT_BOUNDARY:
        raise DecoderControlError("native latent boundary is not the admitted caller-supplied terminal bank")
    for field in ("latent_sha256", "latent_bank_sha256", "decoder_weight_cache_sha256",
                  "stem_weights_sha256", "head_weights_sha256", "continuation_manifest_sha256",
                  "pixel_noise_manifest_sha256", "pixel_noise_record_sha256", "output_sha256"):
        if not _digest(document.get(field)):
            raise DecoderControlError(f"native decode receipt {field} is malformed")
    if document["latent_sha256"] != document["latent_bank_sha256"]:
        raise DecoderControlError("native latent and latent-bank digests differ")
    for field, expected in (
        ("decoder_weight_cache_sha256", EXPECTED_NATIVE_CACHE_SHA256),
        ("stem_weights_sha256", EXPECTED_NATIVE_STEM_SHA256),
        ("head_weights_sha256", EXPECTED_NATIVE_HEAD_SHA256),
    ):
        if document[field] != expected:
            raise DecoderControlError(f"native decode receipt {field} does not match admitted runtime inputs")
    if not isinstance(document.get("latent"), str):
        raise DecoderControlError("native latent path is malformed")
    latent_path = Path(document["latent"])
    if not latent_path.is_absolute():
        raise DecoderControlError("native latent path must be absolute")
    continuation_source = document.get("continuation_source")
    noise_source = document.get("pixel_noise_source")
    if (not isinstance(continuation_source, str) or not Path(continuation_source).is_absolute()
            or noise_source != continuation_source):
        raise DecoderControlError("continuation manifest and global pixel-noise source paths differ")
    if document["continuation_manifest_sha256"] != document["pixel_noise_manifest_sha256"]:
        raise DecoderControlError("continuation and pixel-noise manifest digests differ")
    if document.get("pixel_noise_shape") != list(NOISE_PREFIX_SHAPE):
        raise DecoderControlError("native pixel-noise prefix geometry mismatch")
    if document.get("pixel_noise_selection") != EXPECTED_NOISE_SELECTION:
        raise DecoderControlError("native pixel-noise prefix selection is not the global axis-two prefix")
    if document.get("pixelumm") != {
        "steps": 1,
        "window_latents": WINDOW_LATENTS,
        "overlap_latents": OVERLAP_LATENTS,
        "starts": list(WINDOW_STARTS),
    }:
        raise DecoderControlError("native PixelUMM window schedule does not match eleven global windows")
    if document.get("output_dtype") != "F32 little-endian planar RGB":
        raise DecoderControlError("native display pixel output dtype is unsupported")
    if document.get("output_shape") != list(RAW_SHAPE):
        raise DecoderControlError("native display pixel output geometry mismatch")
    output_path = document.get("output")
    if not isinstance(output_path, str) or not Path(output_path).is_absolute():
        raise DecoderControlError("native display pixel output path must be absolute")
    return {
        "latent_path": latent_path,
        "latent_sha256": document["latent_sha256"],
        "continuation_path": Path(continuation_source),
        "continuation_sha256": document["continuation_manifest_sha256"],
        "noise_record_sha256": document["pixel_noise_record_sha256"],
        "native_executable_path": Path(executable),
        "native_executable_sha256": document["executable_sha256"],
        "native_output_representation": NATIVE_DISPLAY_REPRESENTATION,
        "native_display_path": Path(output_path),
        "native_display_sha256": document["output_sha256"],
        "window_starts": WINDOW_STARTS,
        "latent_shape": LATENT_SHAPE,
        "source_bank_shape": source_bank_shape_tuple,
        "source_bank_bytes": source_bank_bytes,
        "latent_prefix_gpu_gathered": gpu_gather,
        "noise_prefix_shape": NOISE_PREFIX_SHAPE,
        "full_upstream_video_parity": False,
    }


def authenticate_receipt(path: Path, expected_sha256: str) -> tuple[dict, str, dict[str, object]]:
    """Bind receipt bytes to caller authorization before following its paths."""
    if not _digest(expected_sha256):
        raise DecoderControlError("--native-decode-receipt-sha256 must be 64 lowercase hex characters")
    document, raw = _read_json(path, "native decode receipt")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise DecoderControlError("native decode receipt SHA-256 differs from caller authorization")
    admitted = validate_native_decode_receipt_document(document)
    return document, actual_sha256, admitted


def validate_noise_record(record: Mapping, *, expected_sha256: str) -> dict[str, object]:
    """Require the exact full 720-frame F32 source record and receipt digest."""
    if type(record) is not dict or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}:
        raise DecoderControlError("continuation decoder.noise record is malformed")
    if (record.get("dtype") != "F32" or record.get("shape") != list(NOISE_SOURCE_SHAPE)
            or type(record.get("offset")) is not int or record["offset"] < 0
            or record["offset"] % 4 or type(record.get("nbytes")) is not int
            or record["nbytes"] != NOISE_SOURCE_BYTES):
        raise DecoderControlError("continuation decoder.noise record geometry mismatch")
    if not _digest(expected_sha256) or record.get("sha256") != expected_sha256:
        raise DecoderControlError("continuation noise record digest differs from native receipt")
    return dict(record)


def validate_weight_identities(
    receipt: Mapping, *, cache_manifest_sha256: str,
    stem_file_sha256: str, head_file_sha256: str,
) -> None:
    """Bind native receipt provenance to the artifacts admitted for this run."""
    for field, actual in (
        ("decoder_weight_cache_sha256", cache_manifest_sha256),
        ("stem_weights_sha256", stem_file_sha256),
        ("head_weights_sha256", head_file_sha256),
    ):
        if not _digest(actual) or receipt.get(field) != actual:
            raise DecoderControlError(
                f"native receipt {field} differs from the authenticated local artifact"
            )


def output_paths(prefix: Path | str) -> tuple[Path, Path]:
    """Resolve a fresh absolute ``<prefix>.json`` and ``<prefix>.pixels.f32le`` pair."""
    prefix = Path(prefix).expanduser()
    if not prefix.is_absolute():
        prefix = Path.cwd() / prefix
    if not prefix.name or prefix.name in {".", ".."} or prefix.is_symlink():
        raise DecoderControlError("output prefix must be a named, non-symlink path")
    try:
        parent = prefix.parent.resolve(strict=True)
    except OSError as error:
        raise DecoderControlError("output prefix parent must already exist") from error
    if not parent.is_dir():
        raise DecoderControlError("output prefix parent is not a directory")
    prefix = parent / prefix.name
    outputs = (Path(f"{prefix}.json"), Path(f"{prefix}.pixels.f32le"))
    if prefix.exists() or any(os.path.lexists(path) for path in outputs):
        raise DecoderControlError("refusing to replace an existing output prefix or control result")
    return outputs


def estimate_plan_memory(*, cache_bytes: int = PLAN_CACHE_BYTES) -> dict[str, int | float | str]:
    """Print a conservative pre-run bound; live MPS headroom remains unmeasured."""
    if type(cache_bytes) is not int or cache_bytes <= 0:
        raise DecoderControlError("full36 cache allowance must be a positive byte count")
    comparison_bytes = 6 * RAW_BYTES
    total = (PLAN_BASE_PEAK_BYTES + cache_bytes + 1536 * 1024**2
             + comparison_bytes + MULTI_WINDOW_RESERVE_BYTES)
    if total > MAX_PROCESS_ESTIMATE_BYTES:
        raise DecoderControlError("planned process estimate exceeds the 25.5 GiB guard cap")
    return {
        "estimated_process_bytes": total,
        "estimated_process_gib": total / GIB,
        "process_cap_bytes": MAX_PROCESS_ESTIMATE_BYTES,
        "process_cap_mib": 26_112,
        "one_window_full36_base_estimate_bytes": PLAN_BASE_PEAK_BYTES,
        "full36_cache_resident_allowance_bytes": cache_bytes,
        "capture_io_allocator_margin_bytes": 1536 * 1024**2,
        "six_raw_clip_comparison_units_bytes": comparison_bytes,
        "eleven_window_reserve_bytes": MULTI_WINDOW_RESERVE_BYTES,
        "external_guard": "wan_memory_guard.py --memory-accounting file-cache --runtime-floor-pct 10 --max-footprint-mib 26112 --max-swap-growth-mib 64",
        "limitations": (
            "planning estimate anchored to prior same-latent one-window full36 estimate; "
            "adds six raw-clip buffers and a 1 GiB multi-window reserve; not an observed peak; "
            "the external guard measures this run and enforces the 10% physical-memory floor"
        ),
    }


def _authenticate_input_bundle(
    receipt: Mapping, admitted: Mapping[str, object],
) -> tuple[dict, Path, dict[str, object], str]:
    continuation_path = same_latent._canonical_file(
        admitted["continuation_path"], "continuation input manifest"
    )
    receipt_path = same_latent._canonical_file(
        receipt["continuation_source"], "receipt continuation input manifest"
    )
    if continuation_path != receipt_path:
        raise DecoderControlError("continuation manifest path differs from authenticated native receipt")
    document, raw = same_latent._read_json(continuation_path, "continuation input manifest")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != admitted["continuation_sha256"]:
        raise DecoderControlError("continuation manifest digest differs from native decode receipt")
    try:
        continuation_inputs.validate_document(document)
    except continuation_inputs.InputsError as error:
        raise DecoderControlError(f"continuation input manifest rejected: {error}") from error
    if document.get("upstream_revision") != UPSTREAM_REVISION:
        raise DecoderControlError("continuation input upstream revision mismatch")
    payload_path = same_latent._canonical_file(
        continuation_path.parent / document["payload"], "continuation input payload"
    )
    try:
        decoder_boundary._verify_input_payload(document, payload_path)
    except decoder_boundary.DecoderBoundaryError as error:
        raise DecoderControlError(str(error)) from error
    noise = validate_noise_record(
        document["tensors"]["decoder.noise"],
        expected_sha256=admitted["noise_record_sha256"],
    )
    return document, payload_path, noise, actual_sha


def preflight(*, receipt_path: Path, receipt_sha256: str, output_prefix: Path) -> dict[str, object]:
    """Authenticate every CPU input and the fresh output boundary before MPS."""
    receipt, actual_receipt_sha, admitted = authenticate_receipt(receipt_path, receipt_sha256)
    outputs = output_paths(output_prefix)
    manifest, payload_path, noise_record, manifest_sha = _authenticate_input_bundle(receipt, admitted)
    executable_path = _authenticate_native_executable(
        admitted["native_executable_path"], admitted["native_executable_sha256"]
    )
    latent_path = same_latent._regular_artifact(
        str(admitted["latent_path"]), "native terminal latent",
        admitted["latent_sha256"], admitted["source_bank_bytes"],
    )
    display_path = same_latent._regular_artifact(
        str(admitted["native_display_path"]), "native display pixels",
        admitted["native_display_sha256"], RAW_BYTES,
    )
    sources = same_latent._validate_upstream_source_metadata()
    display_postprocess_source = _validate_display_postprocess_source()
    raw_header, stem_path, stem_receipt_path = same_latent._authenticate_stem()
    head = same_latent._authenticate_head(raw_header)
    cache = same_latent._authenticate_full36_cache()
    cache_manifest, cache_path, cache_payload_path, cache_manifest_sha, cache_payload_sha = cache
    # _authenticate_stem fully checks the file's safetensors header and payload
    # against this admitted complete-subset digest; head admission returns its
    # equivalent complete-file digest after hashing the selected subset.
    stem_sha = same_latent.STEM_SUBSET_SHA256
    head_sha = head["subset_sha256"]
    validate_weight_identities(
        receipt,
        cache_manifest_sha256=cache_manifest_sha,
        stem_file_sha256=stem_sha,
        head_file_sha256=head_sha,
    )
    memory_plan = estimate_plan_memory(cache_bytes=cache_payload_path.stat().st_size)
    return {
        "receipt": receipt,
        "receipt_sha256": actual_receipt_sha,
        "admitted": admitted,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "payload_path": payload_path,
        "payload_sha256": manifest["payload_sha256"],
        "noise_record": noise_record,
        "latent_path": latent_path,
        "native_executable_path": executable_path,
        "native_executable_sha256": admitted["native_executable_sha256"],
        "native_output_representation": admitted["native_output_representation"],
        "native_display_path": display_path,
        "native_display_sha256": admitted["native_display_sha256"],
        "display_postprocess_source": display_postprocess_source,
        "outputs": outputs,
        "source_metadata": sources,
        "raw_header": raw_header,
        "stem_path": stem_path,
        "stem_receipt_path": stem_receipt_path,
        "head": head,
        "cache_manifest": cache_manifest,
        "cache_path": cache_path,
        "cache_payload_path": cache_payload_path,
        "cache_manifest_sha256": cache_manifest_sha,
        "cache_payload_sha256": cache_payload_sha,
        "memory_plan": memory_plan,
    }


def _mps_memory_preflight(arch, cache_bytes: int) -> dict[str, int | float | str]:
    if not torch.backends.mps.is_available():
        raise DecoderControlError("PyTorch MPS is unavailable; CPU model fallback is prohibited")
    if torch.__version__ != block_reference.MPS_ORACLE_TORCH_VERSION:
        raise DecoderControlError(
            f"MPS PixelUMM control requires torch {block_reference.MPS_ORACLE_TORCH_VERSION}; "
            f"got {torch.__version__}"
        )
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        raise DecoderControlError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    try:
        base = block_reference._estimated_peak_bytes(
            GEOMETRY.token_grid, arch, layers=36, resident_layers=1,
            capture_stages=block_reference.FULL_TARGET_STAGES,
            full36_capture_policy=block_reference.THIRTY_SIX_CAPTURE_POLICY,
        )
        physical = block_reference._physical_memory_bytes()
        allocated = int(torch.mps.driver_allocated_memory())
    except (block_reference.ReferenceError, RuntimeError, OSError) as error:
        raise DecoderControlError(f"cannot establish live MPS memory estimate: {error}") from error
    compare_bytes = 6 * RAW_BYTES
    total = (base + cache_bytes + same_latent.EXTRA_GPU_RESERVE_BYTES
             + compare_bytes + MULTI_WINDOW_RESERVE_BYTES)
    reserve = physical // 10
    if total > MAX_PROCESS_ESTIMATE_BYTES:
        raise DecoderControlError("live process estimate exceeds the 25.5 GiB external guard cap")
    if physical - allocated - total < reserve:
        raise DecoderControlError("MPS preflight would leave less than 10% physical-memory reserve")
    return {
        "physical_bytes": physical,
        "driver_allocated_bytes": allocated,
        "one_window_full36_reference_peak_bytes": base,
        "eleven_window_reserve_bytes": MULTI_WINDOW_RESERVE_BYTES,
        "comparison_buffer_bound_bytes": compare_bytes,
        "capture_io_allocator_margin_bytes": same_latent.EXTRA_GPU_RESERVE_BYTES,
        "full36_cache_resident_allowance_bytes": cache_bytes,
        "estimated_process_bytes": total,
        "estimated_process_gib": total / GIB,
        "required_physical_memory_floor_bytes": reserve,
        "process_cap_bytes": MAX_PROCESS_ESTIMATE_BYTES,
        "external_guard": "file-cache accounting; 25.5 GiB cap; 10% runtime floor",
        "limitations": "estimate plus external runtime measurement; not an observed peak until guarded execution completes",
    }


def _mapped_f32(path: Path, *, shape: tuple[int, ...]) -> torch.Tensor:
    nbytes = math.prod(shape) * 4
    return same_latent._mapped_f32(path, offset=0, nbytes=nbytes, shape=shape)


def _write_f32_fresh(path: Path, tensor: torch.Tensor) -> str:
    try:
        with path.open("xb") as stream:
            tensor.numpy().astype("<f4", copy=False).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise DecoderControlError(f"cannot write upstream raw-pixel artifact: {error}") from error
    if path.stat().st_size != RAW_BYTES:
        raise DecoderControlError("upstream raw-pixel artifact has an unexpected byte extent")
    return _sha256_file(path)


def run_control(admission: Mapping[str, object]) -> dict[str, object]:
    """Run eleven windows block-major; keep all activations on MPS until comparison."""
    device = torch.device("mps")
    source, model_sha = block_reference.load_upstream(same_latent.UPSTREAM)
    if model_sha != real_input.UPSTREAM_MODEL_SHA256:
        raise DecoderControlError("pinned upstream PixelUMM model source digest changed")
    dcae_source, source_again, autoencoder_source = real_input._load_modules(same_latent.UPSTREAM)
    if (source_again is not source
            and Path(source_again.__file__).resolve() != Path(source.__file__).resolve()):
        raise DecoderControlError("upstream module import identity changed")
    if same_latent._sha256_file(Path(dcae_source.__file__)) != real_input.DCAE_ENCODER_SHA256:
        raise DecoderControlError("pinned upstream DCAE source digest changed")
    if same_latent._sha256_file(Path(autoencoder_source.__file__)) != real_input.PIXELUMM_NOISE_SHA256:
        raise DecoderControlError("pinned upstream PixelUMM blender source digest changed")
    try:
        block_reference.validate_preview_oracle_runtime(GEOMETRY.token_grid, "mps")
    except block_reference.ReferenceError as error:
        raise DecoderControlError(str(error)) from error
    arch = source.QWEN3_8B_GEN
    cache_bytes = admission["cache_payload_path"].stat().st_size
    memory = _mps_memory_preflight(arch, cache_bytes)

    cache_manifest, cache_payload_path, cache_words, cache_manifest_sha, cache_payload_sha = (
        same_latent._load_full36_cache(admission)
    )
    header_path = same_latent._canonical_file(
        checkpoint_inventory.ROOT / "source" / "pixelumm-header.json",
        "pinned PixelUMM checkpoint header",
    )
    plan, inverse_cpu, _ = block_reference.make_attention_plan(source, GEOMETRY.token_grid, tile=8)
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    inverse = inverse_cpu.to(device=device)
    cos, sin = source.mrope_tables(arch, GEOMETRY.token_grid, device)
    cos = cos.index_select(1, permutation)
    sin = sin.index_select(1, permutation)

    stem = real_input._load_stem(source, device)
    final_norm, head, head_runtime_weights, head_source = head_reference._load_head_state(
        source=source,
        subset_path=admission["head"]["subset_path"],
        raw_header=admission["raw_header"],
        receipt_path=admission["head"]["receipt_path"],
    )
    del head_runtime_weights

    if admission["admitted"]["latent_prefix_gpu_gathered"]:
        latent_bank_cpu = _mapped_f32(
            admission["latent_path"], shape=admission["admitted"]["source_bank_shape"]
        )
        latent_bank = latent_bank_cpu.to(device=device)
        del latent_bank_cpu
        # The native receipt attests this exact axis-two NCTHW GPU prefix. Keep
        # the upload at full-bank extent and make the selected view contiguous.
        latent = latent_bank[:, :, :LATENT_SHAPE[2]].contiguous()
        del latent_bank
    else:
        latent_cpu = _mapped_f32(admission["latent_path"], shape=LATENT_SHAPE)
        latent = latent_cpu.to(device=device)
        del latent_cpu
    native_display_cpu = _mapped_f32(admission["native_display_path"], shape=RAW_SHAPE)
    native_display = native_display_cpu.to(device=device)
    del native_display_cpu
    payload_path = admission["payload_path"]
    payload_cpu = torch.from_file(
        str(payload_path), shared=False,
        size=payload_path.stat().st_size // 4, dtype=torch.float32,
    )
    noise_record = admission["noise_record"]
    noise_source_cpu = payload_cpu.narrow(
        0, noise_record["offset"] // 4, noise_record["nbytes"] // 4
    ).view(NOISE_SOURCE_SHAPE)
    noise = noise_source_cpu.narrow(2, 0, NOISE_PREFIX_SHAPE[2]).to(device=device)
    del payload_cpu, noise_source_cpu

    blender = autoencoder_source._TemporalBlender(total_frames=RAW_SHAPE[2])
    condition = type("_ConditionProjection", (), {})()
    condition.cond_proj = stem.cond_proj
    condition.arch = arch
    layer_checks: list[torch.Tensor] = []
    current_windows: list[torch.Tensor] = []
    with torch.inference_mode():
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            for latent_start in WINDOW_STARTS:
                latent_window = latent[:, :, latent_start:latent_start + WINDOW_LATENTS]
                noise_window = noise[:, :, latent_start * 4:(latent_start + WINDOW_LATENTS) * 4]
                xt_tokens = stem.embed(source.patchify(noise_window.to(stem.embed.weight.dtype)))
                condition_tokens = source.DCAEVPixelUMMDecoder._condition_tokens(
                    condition, latent_window.to(stem.cond_proj.weight.dtype)
                )
                current_windows.append(
                    (xt_tokens + condition_tokens).index_select(1, permutation)
                )
                del latent_window, noise_window, xt_tokens, condition_tokens

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
                    raise DecoderControlError(
                        f"BF16 cache provenance differs from source for block {block_index}"
                    )
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
                        raise DecoderControlError("unexpected decoder intermediate capture was retained")
                del layer

        for window_index, latent_start in enumerate(WINDOW_STARTS):
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                decoder_tokens = final_norm(current_windows[window_index])
                decoder_tokens = decoder_tokens.index_select(1, inverse)
            prediction_tokens = head(decoder_tokens)
            prediction = source.unpatchify(prediction_tokens, GEOMETRY.token_grid)
            blender.add(prediction, latent_start * 4)
            del decoder_tokens, prediction_tokens, prediction

    upstream_raw = blender.result()
    if upstream_raw.device.type != "mps" or tuple(upstream_raw.shape) != RAW_SHAPE:
        raise DecoderControlError("upstream raw decoder result has unexpected device or geometry")
    upstream_display, display_metrics = _queue_display_comparison(upstream_raw, native_display)
    (display_max_abs, display_mean_abs, display_mismatch,
     display_finite, raw_finite) = display_metrics
    checks_pass = torch.stack(layer_checks).all()
    metrics = torch.stack((
        display_max_abs, display_mean_abs, display_mismatch.to(torch.float32),
        display_finite.logical_not().to(torch.float32), raw_finite.logical_not().to(torch.float32),
        checks_pass.logical_not().to(torch.float32),
    ))
    # One terminal boundary: no activation or comparison readback precedes it.
    torch.mps.synchronize()
    (display_max_abs_value, display_mean_abs_value, mismatch_bit,
     display_nonfinite_bit, raw_nonfinite_bit, bad_cache_bit) = metrics.cpu().tolist()
    upstream_cpu = upstream_raw.to(dtype=torch.float32).contiguous().cpu()
    del metrics, cache_words, latent, noise, native_display, upstream_display

    if display_nonfinite_bit or raw_nonfinite_bit:
        raise DecoderControlError("native/upstream display or raw pixels contain non-finite values")
    if bad_cache_bit:
        raise DecoderControlError("one or more source decoder weights differ from the authenticated BF16 cache")
    output_json, output_raw = admission["outputs"]
    output_raw_sha = _write_f32_fresh(output_raw, upstream_cpu)
    result = {
        "schema": SCHEMA,
        "upstream_revision": block_reference.UPSTREAM_REVISION,
        "upstream_model_sha256": model_sha,
        "native_decode_receipt_sha256": admission["receipt_sha256"],
        "native_receipt_claimed_full_upstream_video_parity": False,
        "native_receipt_is_runtime_source_proof": False,
        "native_terminal_latent_sha256": admission["admitted"]["latent_sha256"],
        "native_latent_bank_shape": list(admission["admitted"]["source_bank_shape"]),
        "native_selected_latent_shape": list(admission["admitted"]["latent_shape"]),
        "native_latent_prefix_gpu_gathered": admission["admitted"]["latent_prefix_gpu_gathered"],
        "continuation_manifest_sha256": admission["manifest_sha256"],
        "continuation_payload_sha256": admission["payload_sha256"],
        "pixel_noise_record_sha256": noise_record["sha256"],
        "native_executable_sha256": admission["native_executable_sha256"],
        "native_output_representation": admission["native_output_representation"],
        "native_display_pixels_sha256": admission["native_display_sha256"],
        "display_postprocess_source": admission["display_postprocess_source"],
        "decoder_semantics": {
            "native_latent_source": (
                "authenticated caller-supplied terminal latent bank; decoder consumes "
                "global frames [0, 60) per NCTHW channel; no DiT rerun"
            ),
            "source_bank_shape": list(admission["admitted"]["source_bank_shape"]),
            "selected_latent_shape": list(admission["admitted"]["latent_shape"]),
            "latent_prefix_gpu_gathered": admission["admitted"]["latent_prefix_gpu_gathered"],
            "noise_source": "authenticated global decoder.noise NCTHW F32 record sliced on axis 2",
            "noise_prefix_frames": [0, 240],
            "window_latent_starts": list(WINDOW_STARTS),
            "window_latents": WINDOW_LATENTS,
            "overlap_latents": OVERLAP_LATENTS,
            "pixel_frame_starts": [start * 4 for start in WINDOW_STARTS],
            "merge": "pinned upstream _TemporalBlender cosine overlap with BF16 values",
            "blocks": 36,
            "head": "pinned upstream PixelHead",
        },
        "comparison": {
            "representation": "display F32 planar RGB [0,1] after pinned BF16 multiply/add/clamp",
            "shape": list(RAW_SHAPE),
            "atol": ATOL,
            "rtol": RTOL,
            "any_value_outside_tolerance": bool(mismatch_bit),
            "all_values_finite": not bool(display_nonfinite_bit or raw_nonfinite_bit),
            "all_source_weights_match_bf16_cache": not bool(bad_cache_bit),
            "max_abs_error": display_max_abs_value,
            "mean_abs_error": display_mean_abs_value,
            "result": "match" if not mismatch_bit else "mismatch",
        },
        "attention_adapter": same_latent.ATTENTION_SCOPE,
        "unmodified_eager_flex_attention_parity": False,
        "memory_preflight": memory,
        "decoder_cache_manifest_sha256": cache_manifest_sha,
        "decoder_cache_payload_sha256": cache_payload_sha,
        "stem_subset_sha256": checkpoint_inventory.validate_named_subset_receipt(
            admission["raw_header"],
            checkpoint_inventory.parse_header(admission["stem_receipt_path"].read_bytes()),
            admission["stem_path"],
            "stem",
        )["subset_sha256"],
        "head_subset_sha256": head_source["subset_sha256"],
        "head_receipt_sha256": head_source["receipt_sha256"],
        "weight_word_checks": len(layer_checks),
        "upstream_raw_pixels": str(output_raw),
        "upstream_raw_pixels_bytes": RAW_BYTES,
        "upstream_raw_pixels_sha256": output_raw_sha,
    }
    try:
        with output_json.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise DecoderControlError(f"cannot write control result manifest: {error}") from error
    result["output_json"] = str(output_json)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-decode-receipt", type=Path, required=True)
    parser.add_argument("--native-decode-receipt-sha256", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="run the full 36-block MPS control")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        admitted = preflight(
            receipt_path=args.native_decode_receipt,
            receipt_sha256=args.native_decode_receipt_sha256,
            output_prefix=args.output_prefix,
        )
        if not args.execute:
            print(json.dumps({
                "schema": SCHEMA,
                "mode": "plan-only",
                "admission": "passed",
                "native_decode_receipt_sha256": admitted["receipt_sha256"],
                "native_executable_sha256": admitted["native_executable_sha256"],
                "native_output_representation": admitted["native_output_representation"],
                "native_terminal_latent_sha256": admitted["admitted"]["latent_sha256"],
                "native_latent_bank_shape": list(admitted["admitted"]["source_bank_shape"]),
                "native_selected_latent_shape": list(admitted["admitted"]["latent_shape"]),
                "native_latent_prefix_gpu_gathered": admitted["admitted"]["latent_prefix_gpu_gathered"],
                "native_display_pixels_sha256": admitted["admitted"]["native_display_sha256"],
                "display_postprocess_source": admitted["display_postprocess_source"],
                "continuation_manifest_sha256": admitted["manifest_sha256"],
                "continuation_payload_sha256": admitted["payload_sha256"],
                "pixel_noise_record_sha256": admitted["noise_record"]["sha256"],
                "latent_shape": list(LATENT_SHAPE),
                "pixel_shape": list(RAW_SHAPE),
                "window_starts": list(WINDOW_STARTS),
                "outputs": [str(path) for path in admitted["outputs"]],
                "memory_plan": admitted["memory_plan"],
                "source_metadata": admitted["source_metadata"],
                "decoder_cache_manifest_sha256": admitted["cache_manifest_sha256"],
                "decoder_cache_payload_sha256": admitted["cache_payload_sha256"],
            }, indent=2))
            return 0
        try:
            block_reference.validate_preview_oracle_runtime(GEOMETRY.token_grid, "mps")
        except block_reference.ReferenceError as error:
            raise DecoderControlError(str(error)) from error
        result = run_control(admitted)
    except (DecoderControlError, same_latent.SameLatentError,
            block_reference.ReferenceError, head_reference.HeadReferenceError,
            real_input.RealInputError, cache_inventory.CacheError,
            checkpoint_inventory.InventoryError,
            continuation_inputs.InputsError, decoder_boundary.DecoderBoundaryError,
            RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"upstream PixelUMM continuation control failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": result["output_json"],
        "result": result["comparison"]["result"],
        "max_abs_error": result["comparison"]["max_abs_error"],
        "memory_preflight": result["memory_preflight"],
    }, indent=2))
    return 0 if result["comparison"]["result"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
