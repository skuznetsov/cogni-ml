#!/usr/bin/env python3
"""Authenticate matched decoder outputs for the bounded QBit visual review.

This gate verifies that two authenticated DiT timelines feed the same decoder
and noise/settings, then authenticates the produced pixels. It does not assess
visual quality or claim full upstream video parity. In decoder receipts,
legacy `latent_sha256` and `latent_bytes` identify the full source bank;
`latent_shape` describes the gathered decoder prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any

import hy_nomas_cache_qbit_compare as dit_comparison
import hy_nomas_compact_resume_reference as compact_comparison
import hy_nomas_sequence_resume_reference as exact


DECODER_RECEIPT_SCHEMA = "cogni_ml.hy_nomas_decode_latent/v1"
REPORT_SCHEMA = "cogni_ml.hy_nomas_cache_qbit_video_comparison/v1"
LATENT_BANK_SHAPE = (1, 32, 180, 9, 16)
DEFAULT_LATENT_FRAMES = 60
COMPACT_LATENT_FRAMES = 90
PIXEL_FRAMES_PER_LATENT = 4
LATENT_SHAPE = (1, 32, DEFAULT_LATENT_FRAMES, 9, 16)
PIXEL_SHAPE = (1, 3, DEFAULT_LATENT_FRAMES * PIXEL_FRAMES_PER_LATENT, 288, 512)
PIXEL_BYTES = math.prod(PIXEL_SHAPE) * 4
PIXEL_NOISE_SHAPE = PIXEL_SHAPE
PIXEL_CHUNK_BYTES = 1024 * 1024
BASE_INPUTS_SHA256 = "f5001fcbc786401488de1eb652061899ae4fdda88febb67ebd8bd6a22e569bcb"
CONTINUATION_INPUTS_SHA256 = "2807e11814f07975c807f367b0a4f15e3c6bdde1eb62ce9bda851814d09a6e47"
PIXELUMM_WINDOW_LATENTS = 10
PIXELUMM_OVERLAP_LATENTS = 5
PIXELUMM_SETTINGS = {
    "steps": 1,
    "window_latents": PIXELUMM_WINDOW_LATENTS,
    "overlap_latents": PIXELUMM_OVERLAP_LATENTS,
    "starts": list(range(0, DEFAULT_LATENT_FRAMES - PIXELUMM_WINDOW_LATENTS + 1,
                          PIXELUMM_OVERLAP_LATENTS)),
}
LATENT_SELECTION = (
    "first 60 global frames from each NCTHW channel using "
    "HyNomasTimelineOps::PrefixPlan"
)
NOISE_SELECTION = (
    "first 240 global frames from each NCTHW channel using "
    "HyNomasTimelineOps::PrefixPlan"
)
PAIR_IDENTITY_SHA_FIELDS = (
    "executable_sha256",
    "inputs_sha256",
    "decoder_weight_cache_sha256",
    "stem_weights_sha256",
    "head_weights_sha256",
    "pixel_noise_manifest_sha256",
    "pixel_noise_record_sha256",
)
PAIR_IDENTITY_FIELDS = (
    "upstream_revision",
    "pixel_noise_selection",
    "precision",
)


def _expect_int(receipt: dict[str, Any], key: str, expected: int, label: str) -> None:
    if type(receipt.get(key)) is not int or receipt[key] != expected:
        raise exact.ResumeReferenceError(f"{label} {key} must be {expected}")


def _expect_shape(
    receipt: dict[str, Any], key: str, expected: tuple[int, ...], label: str
) -> tuple[int, ...]:
    value = receipt.get(key)
    if type(value) is not list or any(type(dimension) is not int for dimension in value):
        raise exact.ResumeReferenceError(f"{label} {key} must be an integer shape")
    if value != list(expected):
        raise exact.ResumeReferenceError(f"{label} {key} does not match the admitted geometry")
    return tuple(value)


def _f32_extent(shape: tuple[int, ...]) -> int:
    return math.prod(shape) * 4


def _expect_string(receipt: dict[str, Any], key: str, expected: str, label: str) -> None:
    if type(receipt.get(key)) is not str or receipt[key] != expected:
        raise exact.ResumeReferenceError(f"{label} {key} must be {expected!r}")


def _admitted_geometry(latent_frames: int) -> dict[str, Any]:
    if type(latent_frames) is not int or latent_frames not in (60, 90):
        raise exact.ResumeReferenceError("latent_frames must be one of the admitted 60 or 90 frames")
    pixel_frames = latent_frames * PIXEL_FRAMES_PER_LATENT
    latent_shape = (1, 32, latent_frames, 9, 16)
    pixel_shape = (1, 3, pixel_frames, 288, 512)
    settings = {
        "steps": 1,
        "window_latents": PIXELUMM_WINDOW_LATENTS,
        "overlap_latents": PIXELUMM_OVERLAP_LATENTS,
        "starts": list(range(
            0,
            latent_frames - PIXELUMM_WINDOW_LATENTS + 1,
            PIXELUMM_OVERLAP_LATENTS,
        )),
    }
    return {
        "latent_frames": latent_frames,
        "pixel_frames": pixel_frames,
        "latent_shape": latent_shape,
        "selected_prefix_bytes": _f32_extent(latent_shape),
        "pixel_shape": pixel_shape,
        "pixel_bytes": _f32_extent(pixel_shape),
        "settings": settings,
        "latent_selection": (
            f"first {latent_frames} global frames from each NCTHW channel using "
            "HyNomasTimelineOps::PrefixPlan"
        ),
        "noise_selection": (
            f"first {pixel_frames} global frames from each NCTHW channel using "
            "HyNomasTimelineOps::PrefixPlan"
        ),
    }


def _validate_decoder_receipt(
    receipt: dict[str, Any], label: str, timeline_sha256: str, *, latent_frames: int = DEFAULT_LATENT_FRAMES
) -> dict[str, Any]:
    geometry = _admitted_geometry(latent_frames)
    if receipt.get("schema") != DECODER_RECEIPT_SCHEMA:
        raise exact.ResumeReferenceError(f"{label} decoder receipt schema is unsupported")
    if receipt.get("full_upstream_video_parity") is not False:
        raise exact.ResumeReferenceError(f"{label} must not claim full upstream video parity")
    exact._required_string(receipt, "pixel_noise_source", f"{label} decoder receipt")
    exact._required_string(receipt, "executable", f"{label} decoder receipt")

    for key in ("executable_sha256", "inputs_sha256", "decoder_weight_cache_sha256",
                "stem_weights_sha256", "head_weights_sha256", "pixel_noise_manifest_sha256",
                "pixel_noise_record_sha256", "output_sha256"):
        exact._required_sha(receipt, key, f"{label} decoder receipt")
    if receipt["inputs_sha256"] != BASE_INPUTS_SHA256:
        raise exact.ResumeReferenceError(f"{label} decoder base input manifest differs from the pinned bundle")
    bank_sha256 = exact._required_sha(receipt, "latent_bank_sha256", f"{label} decoder receipt")
    latent_sha256 = exact._required_sha(receipt, "latent_sha256", f"{label} decoder receipt")
    if bank_sha256 != timeline_sha256 or latent_sha256 != timeline_sha256:
        raise exact.ResumeReferenceError(
            f"{label} decoder latent bank SHA-256 does not match the authenticated DiT timeline"
        )

    source_bank_shape = _expect_shape(receipt, "latent_bank_shape", LATENT_BANK_SHAPE, label)
    selected_prefix_shape = _expect_shape(receipt, "latent_shape", geometry["latent_shape"], label)
    source_bank_bytes = _f32_extent(source_bank_shape)
    selected_prefix_bytes = _f32_extent(selected_prefix_shape)
    _expect_shape(receipt, "pixel_noise_shape", geometry["pixel_shape"], label)
    _expect_shape(receipt, "output_shape", geometry["pixel_shape"], label)
    for key, expected in (
        ("latent_bank_frames", 180),
        ("latent_frames", geometry["latent_frames"]),
        ("latent_selected_frames", geometry["latent_frames"]),
        ("pixel_noise_frames", geometry["pixel_frames"]),
        ("frames", geometry["pixel_frames"]),
        ("width", 512),
        ("height", 288),
        ("fps", 24),
        ("output_bytes", geometry["pixel_bytes"]),
    ):
        _expect_int(receipt, key, expected, label)
    # The older generic fields describe the uploaded source bank, not the
    # shorter `latent_shape` prefix gathered for this decoder request.
    if type(receipt.get("latent_bytes")) is not int or receipt["latent_bytes"] != source_bank_bytes:
        raise exact.ResumeReferenceError(
            f"{label} legacy latent_bytes must equal source-bank extent from latent_bank_shape"
        )
    if type(receipt.get("latent_bank_bytes")) is not int or receipt["latent_bank_bytes"] != source_bank_bytes:
        raise exact.ResumeReferenceError(
            f"{label} latent_bank_bytes must equal source-bank extent from latent_bank_shape"
        )
    _expect_string(receipt, "latent_dtype", "F32 little-endian", label)
    _expect_string(receipt, "latent_selection", geometry["latent_selection"], label)
    _expect_string(receipt, "pixel_noise_selection", geometry["noise_selection"], label)
    _expect_string(receipt, "output_dtype", "F32 little-endian planar RGB", label)

    settings = receipt.get("pixelumm")
    expected_settings = geometry["settings"]
    if type(settings) is not dict or set(settings) != set(expected_settings):
        raise exact.ResumeReferenceError(f"{label} PixelUMM settings differ from the admitted decode")
    for key in ("steps", "window_latents", "overlap_latents"):
        if type(settings.get(key)) is not int or settings[key] != expected_settings[key]:
            raise exact.ResumeReferenceError(f"{label} PixelUMM setting {key} is invalid")
    starts = settings.get("starts")
    if (type(starts) is not list or any(type(start) is not int for start in starts)
            or starts != expected_settings["starts"]):
        raise exact.ResumeReferenceError(f"{label} PixelUMM window starts are invalid")

    metrics = receipt.get("metrics")
    expected_metrics = {
        "decoder_encode_allocations": 0,
        "decoder_encode_readbacks": 0,
        "decoder_preterminal_waits": 0,
        "decoder_waits": 1,
        "latent_prefix_gpu_gathered": 1,
    }
    if type(metrics) is not dict:
        raise exact.ResumeReferenceError(f"{label} decoder metrics must be an object")
    for key, expected in expected_metrics.items():
        if type(metrics.get(key)) is not int or metrics[key] != expected:
            raise exact.ResumeReferenceError(f"{label} decoder metric {key} must be {expected}")

    return {
        "output_sha256": receipt["output_sha256"],
        "latent_sha256": latent_sha256,
        "source_bank_shape": source_bank_shape,
        "source_bank_bytes": source_bank_bytes,
        "selected_prefix_shape": selected_prefix_shape,
        "selected_prefix_bytes": selected_prefix_bytes,
    }


def validate_decoder_receipts(
    exact_receipt: dict[str, Any],
    qbit_receipt: dict[str, Any],
    exact_timeline_sha256: str,
    qbit_timeline_sha256: str,
    exact_dit_inputs_sha256: str,
    qbit_dit_inputs_sha256: str,
    *,
    latent_frames: int = DEFAULT_LATENT_FRAMES,
) -> dict[str, Any]:
    """Validate the per-arm DIT→decoder links and matched decoder identities."""
    geometry = _admitted_geometry(latent_frames)
    if (type(exact_dit_inputs_sha256) is not str
            or exact.SHA256_RE.fullmatch(exact_dit_inputs_sha256) is None
            or type(qbit_dit_inputs_sha256) is not str
            or exact.SHA256_RE.fullmatch(qbit_dit_inputs_sha256) is None):
        raise exact.ResumeReferenceError("DiT continuation manifest identities must be lowercase SHA-256")
    if exact_dit_inputs_sha256 != qbit_dit_inputs_sha256:
        raise exact.ResumeReferenceError("DiT continuation manifest pair identity mismatch")
    if exact_dit_inputs_sha256 != CONTINUATION_INPUTS_SHA256:
        raise exact.ResumeReferenceError("DiT continuation manifest differs from the pinned bundle")
    if exact_timeline_sha256 == qbit_timeline_sha256:
        raise exact.ResumeReferenceError("QBit DiT timeline treatment is inactive")

    exact_artifacts = _validate_decoder_receipt(
        exact_receipt, "exact", exact_timeline_sha256, latent_frames=latent_frames
    )
    qbit_artifacts = _validate_decoder_receipt(
        qbit_receipt, "qbit", qbit_timeline_sha256, latent_frames=latent_frames
    )
    exact_output_sha = exact_artifacts["output_sha256"]
    qbit_output_sha = qbit_artifacts["output_sha256"]

    for name, receipt, dit_inputs_sha256 in (
        ("exact", exact_receipt, exact_dit_inputs_sha256),
        ("qbit", qbit_receipt, qbit_dit_inputs_sha256),
    ):
        exact._required_sha(receipt, "inputs_sha256", f"{name} decoder receipt")
        if exact._required_sha(receipt, "pixel_noise_manifest_sha256", f"{name} decoder receipt") != dit_inputs_sha256:
            raise exact.ResumeReferenceError(
                f"{name} decoder noise manifest does not match its DiT continuation manifest"
            )

    for key in PAIR_IDENTITY_SHA_FIELDS:
        left = exact._required_sha(exact_receipt, key, "exact decoder receipt")
        right = exact._required_sha(qbit_receipt, key, "qbit decoder receipt")
        if left != right:
            raise exact.ResumeReferenceError(f"decoder pair identity mismatch: {key}")
    for key in PAIR_IDENTITY_FIELDS:
        left = exact._required_string(exact_receipt, key, "exact decoder receipt")
        right = exact._required_string(qbit_receipt, key, "qbit decoder receipt")
        if left != right:
            raise exact.ResumeReferenceError(f"decoder pair identity mismatch: {key}")

    if exact_receipt["latent_bank_sha256"] == qbit_receipt["latent_bank_sha256"]:
        raise exact.ResumeReferenceError("decoder latent treatment is inactive")
    if exact_output_sha == qbit_output_sha:
        raise exact.ResumeReferenceError("pixel output treatment is inactive")

    return {
        "decoder_pair_integrity_passed": True,
        "decoder_sha256": exact_receipt["executable_sha256"],
        "upstream_revision": exact_receipt["upstream_revision"],
        "precision": exact_receipt["precision"],
        "pixel_noise_selection": exact_receipt["pixel_noise_selection"],
        "inputs_sha256": exact_receipt["inputs_sha256"],
        "dit_continuation_manifest_sha256": exact_dit_inputs_sha256,
        "weights": {
            "decoder_cache_sha256": exact_receipt["decoder_weight_cache_sha256"],
            "stem_sha256": exact_receipt["stem_weights_sha256"],
            "head_sha256": exact_receipt["head_weights_sha256"],
        },
        "pixel_noise": {
            "manifest_sha256": exact_receipt["pixel_noise_manifest_sha256"],
            "record_sha256": exact_receipt["pixel_noise_record_sha256"],
            "shape": list(geometry["pixel_shape"]),
        },
        "settings": exact_receipt["pixelumm"],
        "timeline_sha256": {"exact": exact_timeline_sha256, "qbit": qbit_timeline_sha256},
        "latent_bank_sha256": {
            "exact": exact_receipt["latent_bank_sha256"],
            "qbit": qbit_receipt["latent_bank_sha256"],
        },
        "latent_extents": {
            "source_bank": {
                "sha256": {"exact": exact_artifacts["latent_sha256"],
                           "qbit": qbit_artifacts["latent_sha256"]},
                "shape": list(LATENT_BANK_SHAPE),
                "f32_bytes": {
                    "exact": exact_artifacts["source_bank_bytes"],
                    "qbit": qbit_artifacts["source_bank_bytes"],
                },
            },
            "selected_decoder_prefix": {
                "shape": list(geometry["latent_shape"]),
                "frames": geometry["latent_frames"],
                "f32_bytes": {
                    "exact": exact_artifacts["selected_prefix_bytes"],
                    "qbit": qbit_artifacts["selected_prefix_bytes"],
                },
                "selection": geometry["latent_selection"],
            },
        },
        "output_sha256": {"exact": exact_output_sha, "qbit": qbit_output_sha},
        "active_latent_treatment": True,
        "active_pixel_output": True,
    }


def _validate_sibling_artifact_path(
    directory: Path, receipt: dict[str, Any], key: str, filename: str, label: str
) -> Path:
    path_value = receipt.get(key)
    if type(path_value) is not str or not path_value:
        raise exact.ResumeReferenceError(f"{label} receipt missing {key} path")
    raw_path = Path(path_value)
    if not raw_path.is_absolute() or raw_path.name != filename:
        raise exact.ResumeReferenceError(f"{label} {key} path must be the absolute {filename} sibling")
    if raw_path.is_symlink():
        raise exact.ResumeReferenceError(f"{label} {key} must not be a symlink")
    try:
        parent = raw_path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise exact.ResumeReferenceError(f"cannot resolve {label} {key} parent") from error
    if parent != directory:
        raise exact.ResumeReferenceError(f"{label} {key} must be beside receipt.json")
    return raw_path


def _validate_latent_path(
    decoder_receipt: dict[str, Any], dit_receipt: dict[str, Any], dit_directory: Path, label: str
) -> None:
    source = _validate_sibling_artifact_path(
        dit_directory, decoder_receipt, "latent", "timeline.f32le", f"{label} decoder latent"
    )
    diagnostics = dit_receipt.get("diagnostics")
    if type(diagnostics) is not dict:
        raise exact.ResumeReferenceError(f"{label} DiT receipt diagnostics must be an object")
    timeline = _validate_sibling_artifact_path(
        dit_directory, diagnostics, "timeline", "timeline.f32le", f"{label} DiT timeline"
    )
    try:
        if source.resolve(strict=True) != timeline.resolve(strict=True):
            raise exact.ResumeReferenceError(f"{label} decoder did not consume its authenticated DiT timeline path")
    except (OSError, RuntimeError) as error:
        raise exact.ResumeReferenceError(f"cannot resolve {label} decoder latent path") from error


def authenticate_pixels(
    directory: Path, receipt: dict[str, Any], label: str, *, expected_bytes: int = PIXEL_BYTES
) -> dict[str, Any]:
    """Stream-hash and finite-check an F32 artifact without retaining its payload."""
    if type(expected_bytes) is not int or expected_bytes <= 0 or expected_bytes % 4:
        raise exact.ResumeReferenceError("pixel artifact extent must be a positive F32 byte count")
    output_bytes = receipt.get("output_bytes")
    if type(output_bytes) is not int or output_bytes != expected_bytes:
        raise exact.ResumeReferenceError(f"{label} pixel output byte extent does not match the admitted geometry")
    expected_sha256 = exact._required_sha(receipt, "output_sha256", f"{label} decoder receipt")
    path = _validate_sibling_artifact_path(directory, receipt, "output", "pixels.f32le", f"{label} pixels")

    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise exact.ResumeReferenceError(f"{label} pixels must be a regular non-symlink file")
        if before.st_size != expected_bytes:
            raise exact.ResumeReferenceError(
                f"{label} pixels have {before.st_size} bytes; expected {expected_bytes}"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or exact._stat_identity(opened) != exact._stat_identity(before):
                raise exact.ResumeReferenceError(f"{label} pixels changed before being opened")
            digest = hashlib.sha256()
            carry = b""
            bytes_read = 0
            finite_values = 0
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while bytes_read <= expected_bytes:
                    read_size = min(PIXEL_CHUNK_BYTES, expected_bytes + 1 - bytes_read)
                    chunk = stream.read(read_size)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    digest.update(chunk)
                    aligned = carry + chunk
                    aligned_size = len(aligned) - (len(aligned) % 4)
                    values = aligned[:aligned_size]
                    for (value,) in struct.iter_unpack("<f", values):
                        if not math.isfinite(value):
                            raise exact.ResumeReferenceError(f"{label} pixels contain a non-finite F32 value")
                    finite_values += aligned_size // 4
                    carry = aligned[aligned_size:]
                    if bytes_read > expected_bytes:
                        break
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.stat(path, follow_symlinks=False)
    except exact.ResumeReferenceError:
        raise
    except OSError as error:
        raise exact.ResumeReferenceError(f"cannot read {label} pixels") from error

    if (
        exact._stat_identity(before) != exact._stat_identity(after_open)
        or exact._stat_identity(before) != exact._stat_identity(after_path)
        or bytes_read != expected_bytes
        or carry
        or finite_values != expected_bytes // 4
    ):
        raise exact.ResumeReferenceError(f"{label} pixels changed while being read")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise exact.ResumeReferenceError(f"{label} pixel SHA-256 does not match its receipt")
    return {
        "path": str(path),
        "bytes": expected_bytes,
        "sha256": actual_sha256,
        "finite_f32": True,
        "finite_elements": finite_values,
    }


_COMPACT_MATRIX_INTERVALS = {
    "exact_control": [[0, 3]],
    "exact_restart_chain": [[0, 1], [1, 2], [2, 3]],
    "qbit_control": [[0, 3]],
    "compact_restart_chain": [[0, 1], [1, 2], [2, 3]],
}


def _compact_matrix_timeline(matrix_report: dict[str, Any], arm: str) -> dict[str, Any]:
    """Re-check the selected terminal timeline link from an authenticated matrix report."""
    if type(matrix_report) is not dict or matrix_report.get("passed") is not True:
        raise exact.ResumeReferenceError("compact DiT matrix did not pass its integrity comparator")

    source_bank = matrix_report.get("source_bank")
    if type(source_bank) is not dict:
        raise exact.ResumeReferenceError("compact DiT matrix source bank metadata is missing")
    _expect_shape({"shape": source_bank.get("shape")}, "shape", LATENT_BANK_SHAPE, "compact DiT matrix")
    source_bank_bytes = _f32_extent(LATENT_BANK_SHAPE)
    if type(source_bank.get("record_bytes")) is not int or source_bank["record_bytes"] != source_bank_bytes:
        raise exact.ResumeReferenceError("compact DiT matrix source bank extent is invalid")

    intervals = matrix_report.get("intervals")
    if type(intervals) is not dict or any(
        intervals.get(name) != expected for name, expected in _COMPACT_MATRIX_INTERVALS.items()
    ):
        raise exact.ResumeReferenceError("compact DiT matrix must be the admitted three-segment interval set")
    identity = matrix_report.get("identity")
    if type(identity) is not dict:
        raise exact.ResumeReferenceError("compact DiT matrix identity metadata is missing")
    dit_inputs_sha256 = exact._required_sha(identity, "inputs_sha256", "compact DiT matrix")

    artifacts = matrix_report.get("artifacts")
    arm_runs = artifacts.get(arm) if type(artifacts) is dict else None
    if type(arm_runs) is not list or len(arm_runs) != 3:
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} artifact chain is incomplete")
    selected = arm_runs[-1]
    if type(selected) is not dict or selected.get("interval") != [2, 3]:
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} must select terminal interval [2, 3]")

    receipt_value = selected.get("receipt_path")
    if type(receipt_value) is not str or not receipt_value:
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} terminal receipt path is missing")
    receipt_path = Path(receipt_value)
    if not receipt_path.is_absolute() or receipt_path.name != "receipt.json" or receipt_path.is_symlink():
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} terminal receipt path is invalid")
    directory = exact._directory(receipt_path.parent, f"compact DiT matrix {arm} terminal run")
    receipt = exact._read_json(receipt_path, label=f"compact DiT matrix {arm} terminal receipt")
    _expect_shape(receipt, "timeline_shape", LATENT_BANK_SHAPE, f"compact DiT matrix {arm}")

    timeline_meta = selected.get("timeline")
    if type(timeline_meta) is not dict:
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} timeline metadata is missing")
    timeline_path_value = timeline_meta.get("path")
    if type(timeline_path_value) is not str or not timeline_path_value:
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} timeline path is missing")
    timeline_path = Path(timeline_path_value)
    if not timeline_path.is_absolute() or timeline_path.name != "timeline.f32le" or timeline_path.is_symlink():
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} timeline path is invalid")
    expected_timeline_sha256 = exact._required_sha(
        timeline_meta, "sha256", f"compact DiT matrix {arm} timeline metadata"
    )
    if type(timeline_meta.get("bytes")) is not int or timeline_meta["bytes"] != source_bank_bytes:
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} timeline byte extent is invalid")

    diagnostics = receipt.get("diagnostics")
    if type(diagnostics) is not dict:
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} terminal diagnostics are missing")
    if (diagnostics.get("timeline") != timeline_path_value
            or diagnostics.get("timeline_sha256") != expected_timeline_sha256):
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} terminal timeline differs from its report")
    _, authenticated_timeline = exact._read_artifact(
        directory,
        diagnostics,
        "timeline",
        "timeline.f32le",
        source_bank_bytes,
        f"compact DiT matrix {arm} terminal",
        retain_payload=False,
    )
    if authenticated_timeline != timeline_meta:
        raise exact.ResumeReferenceError(f"compact DiT matrix {arm} terminal timeline changed after authentication")
    return {
        "receipt_path": str(receipt_path),
        "timeline_path": str(timeline_path),
        "timeline_sha256": expected_timeline_sha256,
        "timeline_bytes": source_bank_bytes,
        "dit_inputs_sha256": dit_inputs_sha256,
    }


def _validate_compact_latent_path(
    decoder_receipt: dict[str, Any], matrix_timeline: dict[str, Any], label: str
) -> None:
    latent_value = decoder_receipt.get("latent")
    if type(latent_value) is not str or not latent_value:
        raise exact.ResumeReferenceError(f"{label} decoder latent path is missing")
    latent_path = Path(latent_value)
    matrix_path = Path(matrix_timeline["timeline_path"])
    if (not latent_path.is_absolute() or latent_path.name != "timeline.f32le"
            or latent_path.is_symlink() or matrix_path.is_symlink()):
        raise exact.ResumeReferenceError(f"{label} decoder latent path must name its authenticated matrix timeline")
    try:
        if latent_path.resolve(strict=True) != matrix_path.resolve(strict=True):
            raise exact.ResumeReferenceError(
                f"{label} decoder did not consume its selected compact-matrix timeline"
            )
    except (OSError, RuntimeError) as error:
        raise exact.ResumeReferenceError(f"cannot resolve {label} compact decoder latent path") from error


def compare_compact_runs(
    index_path: str,
    exact_decoder_dir: str,
    qbit_decoder_dir: str,
) -> dict[str, Any]:
    """Authenticate the three-segment compact restart timelines and 90-frame decodes."""
    matrix_report = compact_comparison.compare_index(index_path)
    matrix_timelines = {
        "exact": _compact_matrix_timeline(matrix_report, "exact_restart_chain"),
        "qbit": _compact_matrix_timeline(matrix_report, "compact_restart_chain"),
    }
    decoder_directories = {
        "exact": exact._directory(exact_decoder_dir, "exact decoder"),
        "qbit": exact._directory(qbit_decoder_dir, "qbit decoder"),
    }
    decoder_receipts = {
        name: exact._read_json(directory / "receipt.json", label=f"{name} decoder")
        for name, directory in decoder_directories.items()
    }
    for name in ("exact", "qbit"):
        _validate_compact_latent_path(decoder_receipts[name], matrix_timelines[name], name)

    decoder_report = validate_decoder_receipts(
        decoder_receipts["exact"],
        decoder_receipts["qbit"],
        matrix_timelines["exact"]["timeline_sha256"],
        matrix_timelines["qbit"]["timeline_sha256"],
        matrix_timelines["exact"]["dit_inputs_sha256"],
        matrix_timelines["qbit"]["dit_inputs_sha256"],
        latent_frames=COMPACT_LATENT_FRAMES,
    )
    geometry = _admitted_geometry(COMPACT_LATENT_FRAMES)
    pixel_artifacts = {
        name: authenticate_pixels(
            decoder_directories[name], decoder_receipts[name], name,
            expected_bytes=geometry["pixel_bytes"],
        )
        for name in ("exact", "qbit")
    }
    return {
        "schema": REPORT_SCHEMA,
        "pair_integrity_passed": True,
        "dit_pair_integrity_passed": matrix_report["passed"],
        "decoder": decoder_report,
        "dit": matrix_report,
        "pixels": pixel_artifacts,
        "pixel_error": None,
        "quality_verdict": "not assessed; visual review is pending",
        "full_upstream_video_parity": False,
    }


def compare_runs(
    exact_dit_dir: str,
    qbit_dit_dir: str,
    exact_decoder_dir: str,
    qbit_decoder_dir: str,
) -> dict[str, Any]:
    dit_directories = {
        "exact": exact._directory(exact_dit_dir, "exact DiT"),
        "qbit": exact._directory(qbit_dit_dir, "qbit DiT"),
    }
    decoder_directories = {
        "exact": exact._directory(exact_decoder_dir, "exact decoder"),
        "qbit": exact._directory(qbit_decoder_dir, "qbit decoder"),
    }

    # This existing comparator authenticates each full 180-frame artifact and
    # enforces the continuation pair's A-prefix/future-noise controls.
    dit_report = dit_comparison.compare_runs(
        str(dit_directories["exact"]), str(dit_directories["qbit"])
    )
    if dit_report.get("pair_integrity_passed") is not True:
        raise exact.ResumeReferenceError("DiT continuation pair did not pass its integrity comparator")
    dit_receipts = {
        name: exact._read_json(directory / "receipt.json", label=f"{name} DiT")
        for name, directory in dit_directories.items()
    }
    decoder_receipts = {
        name: exact._read_json(directory / "receipt.json", label=f"{name} decoder")
        for name, directory in decoder_directories.items()
    }

    timeline_sha256 = {}
    for name, receipt in dit_receipts.items():
        diagnostics = receipt.get("diagnostics")
        if type(diagnostics) is not dict:
            raise exact.ResumeReferenceError(f"{name} DiT receipt diagnostics must be an object")
        declared_timeline_sha = exact._required_sha(
            diagnostics, "timeline_sha256", f"{name} DiT receipt"
        )
        _, timeline_metadata = exact._read_artifact(
            dit_directories[name], diagnostics, "timeline", "timeline.f32le",
            exact.TIMELINE_BYTES, name, retain_payload=False,
        )
        timeline_sha256[name] = timeline_metadata["sha256"]
        if timeline_sha256[name] != declared_timeline_sha:
            raise exact.ResumeReferenceError(f"{name} DiT timeline SHA changed after pair authentication")
        _validate_latent_path(decoder_receipts[name], receipt, dit_directories[name], name)

    decoder_report = validate_decoder_receipts(
        decoder_receipts["exact"], decoder_receipts["qbit"],
        timeline_sha256["exact"], timeline_sha256["qbit"],
        exact._required_sha(dit_receipts["exact"], "inputs_sha256", "exact DiT receipt"),
        exact._required_sha(dit_receipts["qbit"], "inputs_sha256", "qbit DiT receipt"),
    )
    pixel_artifacts = {
        name: authenticate_pixels(decoder_directories[name], decoder_receipts[name], name)
        for name in ("exact", "qbit")
    }

    return {
        "schema": REPORT_SCHEMA,
        "pair_integrity_passed": True,
        "dit_pair_integrity_passed": dit_report["pair_integrity_passed"],
        "decoder": decoder_report,
        "dit": dit_report,
        "pixels": pixel_artifacts,
        "pixel_error": None,
        "quality_verdict": "not assessed; visual review is pending",
        "full_upstream_video_parity": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-dit-dir")
    parser.add_argument("--qbit-dit-dir")
    parser.add_argument("--compact-index", help="Authenticated three-segment compact resume matrix JSON")
    parser.add_argument("--exact-decoder-dir", required=True)
    parser.add_argument("--qbit-decoder-dir", required=True)
    parser.add_argument("--output", required=True, help="Fresh JSON receipt path")
    args = parser.parse_args(argv)
    if args.compact_index is not None:
        if args.exact_dit_dir is not None or args.qbit_dit_dir is not None:
            parser.error("--compact-index cannot be combined with --exact-dit-dir or --qbit-dit-dir")
    elif args.exact_dit_dir is None or args.qbit_dit_dir is None:
        parser.error("provide --compact-index or both --exact-dit-dir and --qbit-dit-dir")
    try:
        if args.compact_index is not None:
            report = compare_compact_runs(
                args.compact_index, args.exact_decoder_dir, args.qbit_decoder_dir
            )
        else:
            report = compare_runs(
                args.exact_dit_dir, args.qbit_dit_dir,
                args.exact_decoder_dir, args.qbit_decoder_dir,
            )
        output = Path(args.output).absolute()
        exact._write_fresh_json(output, report)
    except (OSError, exact.ResumeReferenceError) as error:
        print(f"QBit video comparison rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"pair_integrity_passed": True, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
