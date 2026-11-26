#!/usr/bin/env python3
"""Authenticate a matched native video decode pair for the Hy-Nomas segment edit.

The comparator freshly authenticates the DiT matrix, binds each decoder to its
selected full latent bank and shared decoder-noise record, then stream-checks
the raw F32 pixels. It checks the decoder overlap prefix and reports regional
pixel errors as diagnostics; it does not assess visual quality or establish
full upstream video parity.
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

try:  # Optional acceleration for the large native pixel banks.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on minimal Python installs.
    _np = None

import hy_nomas_cache_qbit_video_compare as video
import hy_nomas_compact_resume_reference as compact
import hy_nomas_segment_edit_resume_reference as segment_edit
import hy_nomas_sequence_resume_reference as exact


DECODER_RECEIPT_SCHEMA = "cogni_ml.hy_nomas_decode_latent/v1"
REPORT_SCHEMA = "cogni_ml.hy_nomas_segment_edit_video_comparison/v1"
LATENT_FRAMES = 90
PIXEL_FRAMES = 360
PIXEL_HEIGHT = 288
PIXEL_WIDTH = 512
PIXEL_CHANNELS = 3
PREFIX_PIXEL_FRAMES = 100
PIXEL_BYTES = PIXEL_CHANNELS * PIXEL_FRAMES * PIXEL_HEIGHT * PIXEL_WIDTH * 4
PIXEL_FRAME_BYTES = PIXEL_HEIGHT * PIXEL_WIDTH * 4
PIXEL_CHUNK_FRAMES = 8
EXPECTED_PRECISION = (
    "BF16 decoder block/head weights; F32 stem weights and native activation/MPS arithmetic; "
    "original fixed noise and PixelUMM path"
)
PIXEL_REGIONS = (
    ("untouched_prefix", 0, 100),
    ("a_overlap_tail", 100, 120),
    ("b", 120, 240),
    ("c", 240, 360),
)
SEGMENT_INTERVALS = {
    arm: [list(interval) for interval in intervals]
    for arm, intervals in segment_edit.ARM_INTERVALS.items()
}


class SegmentEditVideoComparisonError(exact.ResumeReferenceError):
    """The authenticated matrix or decoded pixel pair violates its contract."""


def _fail(message: str) -> None:
    raise SegmentEditVideoComparisonError(message)


def _read_authenticated_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw, _ = compact._read_hashed_file(
        path,
        expected_sha256,
        "authenticated decoder-noise manifest",
        retain_payload=True,
    )
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=exact._unique_object,
            parse_constant=exact._reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SegmentEditVideoComparisonError("cannot parse authenticated decoder-noise manifest") from error
    if type(document) is not dict:
        _fail("authenticated decoder-noise manifest must be a JSON object")
    return document


def _matrix_identity(matrix_report: dict[str, Any]) -> dict[str, Any]:
    if type(matrix_report) is not dict or matrix_report.get("passed") is not True:
        _fail("segment-edit DiT matrix did not pass its fresh integrity comparator")
    if matrix_report.get("schema") != segment_edit.REPORT_SCHEMA:
        _fail("segment-edit DiT matrix schema is unsupported")
    if matrix_report.get("noise_edit") != segment_edit.NOISE_EDIT:
        _fail("segment-edit DiT matrix does not bind the admitted B-from-F noise edit")

    intervals = matrix_report.get("intervals")
    if type(intervals) is not dict or set(intervals) != set(SEGMENT_INTERVALS):
        _fail("segment-edit DiT matrix interval inventory is invalid")
    for arm, expected in SEGMENT_INTERVALS.items():
        value = intervals.get(arm)
        if (type(value) is not list or len(value) != len(expected)
                or any(type(pair) is not list or len(pair) != 2
                       or any(type(item) is not int for item in pair)
                       for pair in value)
                or value != expected):
            _fail(f"segment-edit DiT matrix {arm} intervals are invalid")

    source = matrix_report.get("source_bank")
    if type(source) is not dict:
        _fail("segment-edit DiT matrix source-bank metadata is missing")
    video._expect_shape(source, "shape", video.LATENT_BANK_SHAPE, "segment-edit DiT matrix")
    bank_bytes = video._f32_extent(video.LATENT_BANK_SHAPE)
    if type(source.get("record_bytes")) is not int or source["record_bytes"] != bank_bytes:
        _fail("segment-edit DiT matrix source-bank extent is invalid")
    latent_record_sha = exact._required_sha(source, "record_sha256", "segment-edit DiT matrix source bank")
    manifest_path = compact._absolute_file(source.get("manifest_path"), "segment-edit continuation manifest")
    manifest_sha = exact._required_sha(source, "manifest_sha256", "segment-edit DiT matrix source bank")

    identity = matrix_report.get("identity")
    if type(identity) is not dict:
        _fail("segment-edit DiT matrix identity metadata is missing")
    identity_manifest = compact._absolute_file(identity.get("inputs"), "segment-edit DiT matrix inputs")
    identity_manifest_sha = exact._required_sha(identity, "inputs_sha256", "segment-edit DiT matrix")
    if identity_manifest != manifest_path or identity_manifest_sha != manifest_sha:
        _fail("segment-edit DiT inputs do not match its authenticated source-bank manifest")

    manifest = _read_authenticated_manifest(manifest_path, manifest_sha)
    qwen_bundle = manifest.get("qwen_input_bundle")
    if type(qwen_bundle) is not dict:
        _fail("authenticated continuation manifest lacks its original Qwen input bundle")
    base_manifest_path = compact._absolute_file(
        qwen_bundle.get("manifest_path"), "authenticated original decoder input manifest"
    )
    base_manifest_sha = exact._required_sha(
        qwen_bundle, "manifest_sha256", "authenticated original decoder input bundle"
    )
    conditioning_sha = exact._required_sha(identity, "conditioning_sha256", "segment-edit DiT matrix")
    if base_manifest_sha != conditioning_sha:
        _fail("segment-edit conditioning SHA does not match its authenticated original input bundle")
    compact._read_hashed_file(
        base_manifest_path,
        base_manifest_sha,
        "authenticated original decoder input manifest",
        retain_payload=False,
    )
    upstream_revision = exact._required_string(identity, "upstream_revision", "segment-edit DiT matrix")

    tensors = manifest.get("tensors")
    noise_record = tensors.get("decoder.noise") if type(tensors) is dict else None
    if type(noise_record) is not dict or noise_record.get("dtype") != "F32":
        _fail("authenticated continuation manifest lacks its F32 decoder.noise record")
    noise_shape = noise_record.get("shape")
    if (type(noise_shape) is not list or any(type(dim) is not int for dim in noise_shape)
            or noise_shape != [1, 3, 720, 288, 512]):
        _fail("authenticated decoder.noise record geometry is invalid")
    noise_record_sha = exact._required_sha(
        noise_record, "sha256", "authenticated continuation decoder.noise record"
    )

    return {
        "inputs_path": str(identity_manifest),
        "inputs_sha256": identity_manifest_sha,
        "conditioning_path": str(base_manifest_path),
        "conditioning_sha256": conditioning_sha,
        "upstream_revision": upstream_revision,
        "latent_record_sha256": latent_record_sha,
        "latent_record_bytes": bank_bytes,
        "latent_shape": list(video.LATENT_BANK_SHAPE),
        "noise_record_sha256": noise_record_sha,
        "noise_manifest_path": str(manifest_path),
        "noise_manifest_sha256": manifest_sha,
        "noise_shape": noise_shape,
    }


def _selected_matrix_timeline(
    matrix_report: dict[str, Any],
    arm: str,
    expected_interval: list[int],
    expected_run_count: int,
) -> dict[str, Any]:
    artifacts = matrix_report.get("artifacts")
    runs = artifacts.get(arm) if type(artifacts) is dict else None
    if type(runs) is not list or len(runs) != expected_run_count:
        _fail(f"segment-edit DiT matrix {arm} artifact chain is incomplete")
    selected = runs[-1]
    if type(selected) is not dict or selected.get("interval") != expected_interval:
        _fail(f"segment-edit DiT matrix {arm} selected interval must be {expected_interval}")

    receipt_value = selected.get("receipt_path")
    if (type(receipt_value) is not str or not receipt_value
            or not Path(receipt_value).is_absolute() or Path(receipt_value).name != "receipt.json"):
        _fail(f"segment-edit DiT matrix {arm} terminal receipt path is invalid")
    receipt_path = Path(receipt_value)
    if receipt_path.is_symlink():
        _fail(f"segment-edit DiT matrix {arm} terminal receipt cannot be a symlink")
    directory = exact._directory(receipt_path.parent, f"segment-edit DiT matrix {arm} terminal run")
    receipt = exact._read_json(receipt_path, label=f"segment-edit DiT matrix {arm} terminal receipt")
    video._expect_shape(receipt, "timeline_shape", video.LATENT_BANK_SHAPE, f"segment-edit DiT matrix {arm}")
    start, stop = expected_interval
    for key, expected in (("start_segment", start), ("stop_after_segments", stop)):
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            _fail(f"segment-edit DiT matrix {arm} terminal receipt {key} is invalid")
    if receipt.get("noise_edit") != (segment_edit.NOISE_EDIT if arm == "edited_restart_chain" else None):
        _fail(f"segment-edit DiT matrix {arm} terminal receipt has the wrong edit treatment")
    expected_noise_scope = (
        segment_edit.EDIT_NOISE_SCOPE if arm == "edited_restart_chain"
        else "unchanged authenticated latent/epsilon banks; no reseeding"
    )
    if receipt.get("noise_scope") != expected_noise_scope:
        _fail(f"segment-edit DiT matrix {arm} terminal receipt has the wrong noise scope")

    state_chains = matrix_report.get("state_chains")
    if arm == "original_control":
        state_link = state_chains.get("original_control") if type(state_chains) is dict else None
    else:
        edited_chain = state_chains.get("edited_restart_chain") if type(state_chains) is dict else None
        state_link = edited_chain[-1] if type(edited_chain) is list and edited_chain else None
    if type(state_link) is not dict or receipt.get("state_input_sha256") != state_link.get("state_input_sha256"):
        _fail(f"segment-edit DiT matrix {arm} terminal receipt is not linked to its authenticated state chain")

    timeline_meta = selected.get("timeline")
    if type(timeline_meta) is not dict:
        _fail(f"segment-edit DiT matrix {arm} terminal timeline metadata is missing")
    timeline_path_value = timeline_meta.get("path")
    if (type(timeline_path_value) is not str or not timeline_path_value
            or not Path(timeline_path_value).is_absolute()
            or Path(timeline_path_value).name != "timeline.f32le"):
        _fail(f"segment-edit DiT matrix {arm} terminal timeline path is invalid")
    timeline_path = Path(timeline_path_value)
    if timeline_path.is_symlink():
        _fail(f"segment-edit DiT matrix {arm} terminal timeline cannot be a symlink")
    try:
        if timeline_path.parent.resolve(strict=True) != directory:
            _fail(f"segment-edit DiT matrix {arm} timeline must be beside its receipt")
    except (OSError, RuntimeError) as error:
        raise SegmentEditVideoComparisonError(
            f"cannot resolve segment-edit DiT matrix {arm} terminal timeline"
        ) from error
    timeline_sha = exact._required_sha(timeline_meta, "sha256", f"segment-edit DiT matrix {arm} timeline")
    if type(timeline_meta.get("bytes")) is not int or timeline_meta["bytes"] != video._f32_extent(video.LATENT_BANK_SHAPE):
        _fail(f"segment-edit DiT matrix {arm} timeline extent is invalid")

    diagnostics = receipt.get("diagnostics")
    if type(diagnostics) is not dict:
        _fail(f"segment-edit DiT matrix {arm} terminal diagnostics are missing")
    if diagnostics.get("timeline") != timeline_path_value or diagnostics.get("timeline_sha256") != timeline_sha:
        _fail(f"segment-edit DiT matrix {arm} terminal timeline differs from its receipt")
    _, authenticated = exact._read_artifact(
        directory,
        diagnostics,
        "timeline",
        "timeline.f32le",
        video._f32_extent(video.LATENT_BANK_SHAPE),
        f"segment-edit DiT matrix {arm} terminal",
        retain_payload=False,
    )
    if authenticated != timeline_meta:
        _fail(f"segment-edit DiT matrix {arm} terminal timeline changed after authentication")
    return {
        "interval": expected_interval,
        "receipt_path": str(receipt_path),
        "timeline_path": str(timeline_path.resolve(strict=True)),
        "timeline_sha256": timeline_sha,
        "timeline_bytes": timeline_meta["bytes"],
    }


def _required_absolute_path(document: dict[str, Any], key: str, label: str) -> Path:
    value = exact._required_string(document, key, label)
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{label} {key} must be an absolute path")
    return compact._absolute_file(value, f"{label} {key}")


def _validate_decoder_receipt(
    receipt: dict[str, Any],
    label: str,
    timeline: dict[str, Any],
    matrix_identity: dict[str, Any],
    decoder_input_identity: dict[str, Any] | None,
    decoder_directory: Path,
) -> dict[str, Any]:
    geometry = video._admitted_geometry(LATENT_FRAMES)
    if receipt.get("schema") != DECODER_RECEIPT_SCHEMA:
        _fail(f"{label} decoder receipt schema is unsupported")
    if receipt.get("full_upstream_video_parity") is not False:
        _fail(f"{label} decoder receipt must not claim full upstream video parity")

    executable = exact._required_string(receipt, "executable", f"{label} decoder receipt")
    if not Path(executable).is_absolute():
        _fail(f"{label} decoder executable path must be absolute")
    executable_sha = exact._required_sha(receipt, "executable_sha256", f"{label} decoder receipt")
    executable_path = compact._absolute_file(executable, f"{label} decoder executable")
    compact._read_hashed_file(
        executable_path, executable_sha, f"{label} decoder executable", retain_payload=False
    )
    upstream_revision = exact._required_string(receipt, "upstream_revision", f"{label} decoder receipt")
    if upstream_revision != matrix_identity["upstream_revision"]:
        _fail(f"{label} decoder upstream revision does not match the authenticated DiT matrix")

    for key in (
        "latent_sha256", "latent_bank_sha256", "decoder_weight_cache_sha256",
        "stem_weights_sha256", "head_weights_sha256", "inputs_sha256",
        "pixel_noise_manifest_sha256", "pixel_noise_record_sha256", "output_sha256",
    ):
        exact._required_sha(receipt, key, f"{label} decoder receipt")
    if (receipt["latent_sha256"] != timeline["timeline_sha256"]
            or receipt["latent_bank_sha256"] != timeline["timeline_sha256"]):
        _fail(f"{label} decoder latent bank SHA-256 does not match its authenticated timeline")

    latent_path = _required_absolute_path(receipt, "latent", f"{label} decoder receipt")
    if latent_path.name != "timeline.f32le" or latent_path != Path(timeline["timeline_path"]):
        _fail(f"{label} decoder did not consume its selected {timeline['arm']} timeline")

    bank_bytes = video._f32_extent(video.LATENT_BANK_SHAPE)
    selected_bytes = video._f32_extent(geometry["latent_shape"])
    video._expect_shape(receipt, "latent_bank_shape", video.LATENT_BANK_SHAPE, label)
    video._expect_shape(receipt, "latent_shape", geometry["latent_shape"], label)
    video._expect_shape(receipt, "pixel_noise_shape", geometry["pixel_shape"], label)
    video._expect_shape(receipt, "output_shape", geometry["pixel_shape"], label)
    for key, expected in (
        ("latent_bank_frames", 180),
        ("latent_frames", LATENT_FRAMES),
        ("latent_selected_frames", LATENT_FRAMES),
        ("pixel_noise_frames", PIXEL_FRAMES),
        ("frames", PIXEL_FRAMES),
        ("width", PIXEL_WIDTH),
        ("height", PIXEL_HEIGHT),
        ("fps", 24),
        ("output_bytes", PIXEL_BYTES),
    ):
        video._expect_int(receipt, key, expected, label)
    for key in ("latent_bytes", "latent_bank_bytes"):
        if type(receipt.get(key)) is not int or receipt[key] != bank_bytes:
            _fail(f"{label} {key} must equal the full source-bank byte extent")
    if exact._required_string(receipt, "latent_dtype", f"{label} decoder receipt") != "F32 little-endian":
        _fail(f"{label} latent dtype must be F32 little-endian")
    if receipt.get("latent_selection") != geometry["latent_selection"]:
        _fail(f"{label} decoder latent prefix selection is invalid")
    if receipt.get("pixel_noise_selection") != geometry["noise_selection"]:
        _fail(f"{label} decoder noise prefix selection is invalid")
    if receipt.get("output_dtype") != "F32 little-endian planar RGB":
        _fail(f"{label} decoder pixel output dtype is invalid")
    if receipt.get("precision") != EXPECTED_PRECISION:
        _fail(f"{label} decoder precision does not match the admitted BF16/F32 path")

    settings = receipt.get("pixelumm")
    expected_settings = geometry["settings"]
    if type(settings) is not dict or set(settings) != set(expected_settings):
        _fail(f"{label} PixelUMM settings differ from the admitted decode")
    for key in ("steps", "window_latents", "overlap_latents"):
        if type(settings.get(key)) is not int or settings[key] != expected_settings[key]:
            _fail(f"{label} PixelUMM setting {key} is invalid")
    starts = settings.get("starts")
    if (type(starts) is not list or any(type(start) is not int for start in starts)
            or starts != expected_settings["starts"]):
        _fail(f"{label} PixelUMM window starts are invalid")

    metrics = receipt.get("metrics")
    if type(metrics) is not dict:
        _fail(f"{label} decoder metrics must be an object")
    for key, expected in (
        ("decoder_encode_allocations", 0),
        ("decoder_encode_readbacks", 0),
        ("decoder_preterminal_waits", 0),
        ("decoder_waits", 1),
    ):
        if type(metrics.get(key)) is not int or metrics[key] != expected:
            _fail(f"{label} decoder metric {key} must be {expected}")

    input_path = _required_absolute_path(receipt, "inputs", f"{label} decoder receipt")
    input_sha = receipt["inputs_sha256"]
    if (input_path != Path(matrix_identity["conditioning_path"])
            or input_sha != matrix_identity["conditioning_sha256"]):
        _fail(f"{label} decoder inputs do not match the authenticated original conditioning bundle")
    if decoder_input_identity is not None:
        if input_path != decoder_input_identity["path"] or input_sha != decoder_input_identity["sha256"]:
            _fail("decoder pair identity mismatch: inputs path or SHA-256")
    noise_source = _required_absolute_path(receipt, "pixel_noise_source", f"{label} decoder receipt")
    if noise_source != Path(matrix_identity["noise_manifest_path"]):
        _fail(f"{label} decoder noise source path does not match the authenticated DiT manifest")
    if receipt["pixel_noise_manifest_sha256"] != matrix_identity["noise_manifest_sha256"]:
        _fail(f"{label} decoder noise manifest does not match the authenticated DiT inputs")
    if receipt["pixel_noise_record_sha256"] != matrix_identity["noise_record_sha256"]:
        _fail(f"{label} decoder.noise record does not match the authenticated continuation record")

    output_path = video._validate_sibling_artifact_path(
        decoder_directory, receipt, "output", "pixels.f32le", f"{label} pixels"
    )
    return {
        "executable": executable,
        "executable_sha256": executable_sha,
        "upstream_revision": upstream_revision,
        "inputs_path": str(input_path),
        "inputs_sha256": input_sha,
        "decoder_weight_cache_sha256": receipt["decoder_weight_cache_sha256"],
        "stem_weights_sha256": receipt["stem_weights_sha256"],
        "head_weights_sha256": receipt["head_weights_sha256"],
        "pixel_noise_source": str(noise_source),
        "pixel_noise_manifest_sha256": receipt["pixel_noise_manifest_sha256"],
        "pixel_noise_record_sha256": receipt["pixel_noise_record_sha256"],
        "precision": receipt["precision"],
        "pixelumm": settings,
        "latent_path": str(latent_path),
        "latent_sha256": receipt["latent_sha256"],
        "latent_bank_shape": list(video.LATENT_BANK_SHAPE),
        "latent_bank_bytes": bank_bytes,
        "latent_shape": list(geometry["latent_shape"]),
        "latent_selected_bytes": selected_bytes,
        "output_path": str(output_path),
        "output_sha256": receipt["output_sha256"],
        "output_bytes": PIXEL_BYTES,
    }


def _validate_decoder_pair(
    original_receipt: dict[str, Any],
    edited_receipt: dict[str, Any],
    timelines: dict[str, dict[str, Any]],
    matrix_identity: dict[str, Any],
    decoder_directories: dict[str, Path],
) -> dict[str, Any]:
    original = _validate_decoder_receipt(
        original_receipt, "original_control", timelines["original_control"], matrix_identity, None,
        decoder_directories["original_control"],
    )
    edited = _validate_decoder_receipt(
        edited_receipt, "edited_restart_chain", timelines["edited_restart_chain"],
        matrix_identity, {"path": Path(original["inputs_path"]), "sha256": original["inputs_sha256"]},
        decoder_directories["edited_restart_chain"],
    )

    for key in (
        "executable", "executable_sha256", "upstream_revision", "inputs_path", "inputs_sha256",
        "decoder_weight_cache_sha256", "stem_weights_sha256", "head_weights_sha256",
        "pixel_noise_source", "pixel_noise_manifest_sha256", "pixel_noise_record_sha256", "precision",
    ):
        if original[key] != edited[key]:
            _fail(f"decoder pair identity mismatch: {key}")
    if original["pixelumm"] != edited["pixelumm"]:
        _fail("decoder pair identity mismatch: PixelUMM settings")

    return {
        "decoder_executable": original["executable"],
        "decoder_sha256": original["executable_sha256"],
        "upstream_revision": original["upstream_revision"],
        "inputs": {"path": original["inputs_path"], "sha256": original["inputs_sha256"]},
        "conditioning_inputs": {
            "path": original["inputs_path"],
            "sha256": original["inputs_sha256"],
        },
        "weights": {
            "decoder_cache_sha256": original["decoder_weight_cache_sha256"],
            "stem_sha256": original["stem_weights_sha256"],
            "head_sha256": original["head_weights_sha256"],
        },
        "precision": original["precision"],
        "pixel_noise": {
            "source": original["pixel_noise_source"],
            "manifest_sha256": original["pixel_noise_manifest_sha256"],
            "record_sha256": original["pixel_noise_record_sha256"],
            "shape": [1, PIXEL_CHANNELS, PIXEL_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH],
        },
        "latent_bank_sha256": {
            "original": original["latent_sha256"],
            "edited": edited["latent_sha256"],
        },
        "output_sha256": {
            "original": original["output_sha256"],
            "edited": edited["output_sha256"],
        },
        "pixelumm": original["pixelumm"],
        "latent_banks": {
            "original_control": {
                "path": original["latent_path"],
                "sha256": original["latent_sha256"],
                "shape": original["latent_bank_shape"],
                "bytes": original["latent_bank_bytes"],
                "selected_prefix_shape": original["latent_shape"],
                "selected_prefix_bytes": original["latent_selected_bytes"],
            },
            "edited_restart_chain": {
                "path": edited["latent_path"],
                "sha256": edited["latent_sha256"],
                "shape": edited["latent_bank_shape"],
                "bytes": edited["latent_bank_bytes"],
                "selected_prefix_shape": edited["latent_shape"],
                "selected_prefix_bytes": edited["latent_selected_bytes"],
            },
        },
        "pixel_outputs": {
            "original_control": {
                "path": original["output_path"],
                "sha256": original["output_sha256"],
                "bytes": original["output_bytes"],
            },
            "edited_restart_chain": {
                "path": edited["output_path"],
                "sha256": edited["output_sha256"],
                "bytes": edited["output_bytes"],
            },
        },
    }


def _open_pixel_file(path: Path, label: str, expected_bytes: int) -> tuple[int, os.stat_result]:
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} pixels must be a regular non-symlink file")
        if before.st_size != expected_bytes:
            _fail(f"{label} pixels have {before.st_size} bytes; expected {expected_bytes}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or exact._stat_identity(opened) != exact._stat_identity(before):
            os.close(descriptor)
            _fail(f"{label} pixels changed before being opened")
        return descriptor, before
    except SegmentEditVideoComparisonError:
        raise
    except OSError as error:
        raise SegmentEditVideoComparisonError(f"cannot open {label} pixels") from error


def _empty_region(start: int, stop: int) -> dict[str, Any]:
    return {
        "frames": [start, stop],
        "elements": 0,
        "sum_squared_error": 0.0,
        "sum_absolute_error": 0.0,
        "max_absolute_error": 0.0,
        "bitwise_changed_elements": 0,
    }


def _accumulate_region(
    metrics: dict[str, Any], original_bytes: bytes, edited_bytes: bytes, label: str
) -> None:
    if _np is not None:
        original = _np.frombuffer(original_bytes, dtype="<f4")
        edited = _np.frombuffer(edited_bytes, dtype="<f4")
        if not bool(_np.isfinite(original).all()) or not bool(_np.isfinite(edited).all()):
            _fail(f"{label} pixels contain a non-finite F32 value")
        original64 = original.astype(_np.float64)
        edited64 = edited.astype(_np.float64)
        delta = original64 - edited64
        absolute = _np.abs(delta)
        metrics["sum_squared_error"] += float(_np.sum(delta * delta, dtype=_np.float64))
        metrics["sum_absolute_error"] += float(_np.sum(absolute, dtype=_np.float64))
        if absolute.size:
            metrics["max_absolute_error"] = max(
                metrics["max_absolute_error"], float(_np.max(absolute))
            )
        original_words = _np.frombuffer(original_bytes, dtype="<u4")
        edited_words = _np.frombuffer(edited_bytes, dtype="<u4")
        metrics["bitwise_changed_elements"] += int(_np.count_nonzero(original_words != edited_words))
        metrics["elements"] += int(original.size)
        return

    for (left,), (right,), (left_word,), (right_word,) in zip(
        struct.iter_unpack("<f", original_bytes),
        struct.iter_unpack("<f", edited_bytes),
        struct.iter_unpack("<I", original_bytes),
        struct.iter_unpack("<I", edited_bytes),
    ):
        if not math.isfinite(left) or not math.isfinite(right):
            _fail(f"{label} pixels contain a non-finite F32 value")
        difference = float(left) - float(right)
        absolute = abs(difference)
        metrics["sum_squared_error"] += difference * difference
        metrics["sum_absolute_error"] += absolute
        metrics["max_absolute_error"] = max(metrics["max_absolute_error"], absolute)
        metrics["bitwise_changed_elements"] += int(left_word != right_word)
        metrics["elements"] += 1


def _finish_region(metrics: dict[str, Any]) -> dict[str, Any]:
    count = metrics["elements"]
    if count <= 0:
        _fail("pixel diagnostic region is empty")
    return {
        "frames": metrics["frames"],
        "elements": count,
        "bitwise_equal": metrics["bitwise_changed_elements"] == 0,
        "bitwise_changed_elements": metrics["bitwise_changed_elements"],
        "rmse": math.sqrt(metrics["sum_squared_error"] / count),
        "mean_absolute_error": metrics["sum_absolute_error"] / count,
        "max_absolute_error": metrics["max_absolute_error"],
    }


def _compare_pixel_files(
    original_path: Path,
    edited_path: Path,
    *,
    original_sha256: str,
    edited_sha256: str,
    expected_bytes: int,
    frames: int = PIXEL_FRAMES,
    height: int = PIXEL_HEIGHT,
    width: int = PIXEL_WIDTH,
    channels: int = PIXEL_CHANNELS,
    prefix_frames: int = PREFIX_PIXEL_FRAMES,
    regions: tuple[tuple[str, int, int], ...] = PIXEL_REGIONS,
) -> dict[str, Any]:
    """Hash, finite-check, and compare planar NCTHW F32 pixel files in bounded chunks."""
    for label, digest in (("original", original_sha256), ("edited", edited_sha256)):
        if type(digest) is not str or exact.SHA256_RE.fullmatch(digest) is None:
            _fail(f"{label} pixel SHA-256 must be lowercase SHA-256")
    if (type(expected_bytes) is not int or expected_bytes <= 0 or expected_bytes % 4
            or type(frames) is not int or frames <= 0
            or type(height) is not int or height <= 0
            or type(width) is not int or width <= 0
            or type(channels) is not int or channels <= 0
            or type(prefix_frames) is not int or not 0 < prefix_frames <= frames):
        _fail("pixel comparison geometry is invalid")
    required_bytes = channels * frames * height * width * 4
    if expected_bytes != required_bytes:
        _fail("pixel artifact extent differs from its planar F32 geometry")
    if (type(regions) is not tuple or not regions or regions[0] != ("untouched_prefix", 0, prefix_frames)):
        _fail("pixel diagnostic regions must begin with the exact unchanged prefix")
    cursor = 0
    seen: set[str] = set()
    for region in regions:
        if (type(region) is not tuple or len(region) != 3
                or type(region[0]) is not str or not region[0] or region[0] in seen
                or type(region[1]) is not int or type(region[2]) is not int
                or region[1] != cursor or region[2] <= region[1] or region[2] > frames):
            _fail("pixel diagnostic regions must be a contiguous frame partition")
        cursor = region[2]
        seen.add(region[0])
    if cursor != frames:
        _fail("pixel diagnostic regions must cover every output frame")

    descriptors: list[int] = []
    streams: list[Any] = []
    before_stats: list[os.stat_result] = []
    digests = [hashlib.sha256(), hashlib.sha256()]
    region_metrics = {name: _empty_region(start, stop) for name, start, stop in regions}
    total_bytes = 0
    frame_values = height * width
    frame_bytes = frame_values * 4
    plane_bytes = frames * frame_bytes
    try:
        for path, label in ((original_path, "original"), (edited_path, "edited")):
            descriptor, before = _open_pixel_file(path, label, expected_bytes)
            descriptors.append(descriptor)
            before_stats.append(before)
        streams = [os.fdopen(descriptor, "rb", closefd=False) for descriptor in descriptors]

        for channel in range(channels):
            plane_offset = channel * plane_bytes
            for region_name, start, stop in regions:
                for region_offset in range(start, stop, PIXEL_CHUNK_FRAMES):
                    block_frames = min(PIXEL_CHUNK_FRAMES, stop - region_offset)
                    byte_count = block_frames * frame_bytes
                    absolute_offset = plane_offset + region_offset * frame_bytes
                    for stream in streams:
                        stream.seek(absolute_offset)
                    original_chunk = streams[0].read(byte_count)
                    edited_chunk = streams[1].read(byte_count)
                    if len(original_chunk) != byte_count or len(edited_chunk) != byte_count:
                        _fail("pixel output changed or ended before its admitted extent")
                    digests[0].update(original_chunk)
                    digests[1].update(edited_chunk)
                    total_bytes += byte_count
                    _accumulate_region(region_metrics[region_name], original_chunk, edited_chunk, region_name)

        if total_bytes != expected_bytes:
            _fail("pixel stream did not cover the complete output extent")
        for index, descriptor in enumerate(descriptors):
            if exact._stat_identity(os.fstat(descriptor)) != exact._stat_identity(before_stats[index]):
                _fail("pixel output changed while being read")
    except SegmentEditVideoComparisonError:
        raise
    except OSError as error:
        raise SegmentEditVideoComparisonError("cannot read pixel outputs") from error
    finally:
        for stream in streams:
            try:
                stream.close()
            except OSError:
                pass
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass

    for path, label, before in zip((original_path, edited_path), ("original", "edited"), before_stats):
        try:
            after_path = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise SegmentEditVideoComparisonError(f"cannot restat {label} pixels") from error
        if exact._stat_identity(before) != exact._stat_identity(after_path):
            _fail(f"{label} pixels changed while being read")
    for label, actual, expected in zip(("original", "edited"), (d.hexdigest() for d in digests),
                                       (original_sha256, edited_sha256)):
        if actual != expected:
            _fail(f"{label} pixel SHA-256 does not match its receipt")

    finished_regions = {name: _finish_region(metrics) for name, metrics in region_metrics.items()}
    prefix = finished_regions["untouched_prefix"]
    if not prefix["bitwise_equal"]:
        _fail(f"prefix frames [0, {prefix_frames}) are not bitwise equal")
    return {
        "prefix_integrity_passed": True,
        "untouched_prefix": prefix,
        "a_overlap_tail": finished_regions.get("a_overlap_tail"),
        "regions": finished_regions,
        "diagnostic_only": True,
    }


def compare_runs(edit_index: str, original_decoder_dir: str, edited_decoder_dir: str) -> dict[str, Any]:
    """Freshly authenticate and compare the segment-edit native decode pair."""
    index_path = compact._absolute_file(str(Path(edit_index).absolute()), "segment-edit matrix index")
    index_sha_before = _stable_file_sha256(index_path, "segment-edit matrix index")
    index_identity_before = exact._stat_identity(os.stat(index_path, follow_symlinks=False))
    matrix_report = segment_edit.compare_index(str(index_path))
    index_sha_after = _stable_file_sha256(index_path, "segment-edit matrix index")
    index_identity_after = exact._stat_identity(os.stat(index_path, follow_symlinks=False))
    if index_sha_before != index_sha_after or index_identity_before != index_identity_after:
        _fail("segment-edit matrix index changed during fresh authentication")
    matrix_identity = _matrix_identity(matrix_report)
    timelines = {
        "original_control": _selected_matrix_timeline(
            matrix_report, "original_control", [1, 3], 1
        ),
        "edited_restart_chain": _selected_matrix_timeline(
            matrix_report, "edited_restart_chain", [2, 3], 2
        ),
    }
    timelines["original_control"]["arm"] = "original_control"
    timelines["edited_restart_chain"]["arm"] = "edited_restart_chain"
    decoder_directories = {
        "original_control": exact._directory(original_decoder_dir, "original decoder"),
        "edited_restart_chain": exact._directory(edited_decoder_dir, "edited decoder"),
    }
    if decoder_directories["original_control"] == decoder_directories["edited_restart_chain"]:
        _fail("original and edited decoders must use distinct output directories")
    receipts = {
        arm: exact._read_json(directory / "receipt.json", label=f"{arm} decoder receipt")
        for arm, directory in decoder_directories.items()
    }
    decoder_report = _validate_decoder_pair(
        receipts["original_control"],
        receipts["edited_restart_chain"],
        timelines,
        matrix_identity,
        decoder_directories,
    )
    pixel_comparison = _compare_pixel_files(
        Path(decoder_report["pixel_outputs"]["original_control"]["path"]),
        Path(decoder_report["pixel_outputs"]["edited_restart_chain"]["path"]),
        original_sha256=decoder_report["pixel_outputs"]["original_control"]["sha256"],
        edited_sha256=decoder_report["pixel_outputs"]["edited_restart_chain"]["sha256"],
        expected_bytes=PIXEL_BYTES,
    )
    return {
        "schema": REPORT_SCHEMA,
        "pair_integrity_passed": True,
        "dit": {
            "schema": matrix_report["schema"],
            "passed": True,
            "index": {"path": str(index_path), "sha256": index_sha_before},
            "inputs": {
                "path": matrix_identity["inputs_path"],
                "sha256": matrix_identity["inputs_sha256"],
            },
            "source_bank": {
                "shape": matrix_identity["latent_shape"],
                "bytes": matrix_identity["latent_record_bytes"],
                "record_sha256": matrix_identity["latent_record_sha256"],
            },
            "noise_edit": dict(matrix_report["noise_edit"]),
            "selected_timelines": timelines,
        },
        "decoder": decoder_report,
        "pixels": {
            "shape": [1, PIXEL_CHANNELS, PIXEL_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH],
            "dtype": "F32 little-endian planar RGB",
            "frames": PIXEL_FRAMES,
            "fps": 24,
            "original_control": decoder_report["pixel_outputs"]["original_control"],
            "edited_restart_chain": decoder_report["pixel_outputs"]["edited_restart_chain"],
            "original": {
                **decoder_report["pixel_outputs"]["original_control"],
                "finite_f32": True,
            },
            "edited": {
                **decoder_report["pixel_outputs"]["edited_restart_chain"],
                "finite_f32": True,
            },
            "comparison": pixel_comparison,
            "losses_by_segment": {
                "A": _combine_regions(
                    pixel_comparison["regions"]["untouched_prefix"],
                    pixel_comparison["regions"]["a_overlap_tail"],
                    [0, 120],
                ),
                "B": pixel_comparison["regions"]["b"],
                "C": pixel_comparison["regions"]["c"],
                "A_overlap_tail": pixel_comparison["regions"]["a_overlap_tail"],
            },
            "losses_scope": "raw finite F32 pair diagnostics only; no quality threshold or perceptual verdict",
        },
        "quality_verdict": "not assessed",
        "full_upstream_video_parity": False,
        "scope": (
            "Matched native decoder pair from the authenticated segment-edit latent banks; "
            "checks the first 100 decoded pixel frames bitwise and reports raw F32 losses only."
        ),
    }


def _stable_file_sha256(path: Path, label: str) -> str:
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} must be a regular non-symlink file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or exact._stat_identity(opened) != exact._stat_identity(before):
                _fail(f"{label} changed before it was opened")
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.stat(path, follow_symlinks=False)
    except SegmentEditVideoComparisonError:
        raise
    except OSError as error:
        raise SegmentEditVideoComparisonError(f"cannot hash {label}") from error
    if (exact._stat_identity(before) != exact._stat_identity(after_open)
            or exact._stat_identity(before) != exact._stat_identity(after_path)):
        _fail(f"{label} changed while being hashed")
    return digest.hexdigest()


def _combine_regions(left: dict[str, Any], right: dict[str, Any], frames: list[int]) -> dict[str, Any]:
    count = left["elements"] + right["elements"]
    sum_squared = (left["rmse"] ** 2 * left["elements"]
                   + right["rmse"] ** 2 * right["elements"])
    sum_absolute = (left["mean_absolute_error"] * left["elements"]
                    + right["mean_absolute_error"] * right["elements"])
    max_absolute = max(left["max_absolute_error"], right["max_absolute_error"])
    changed = left["bitwise_changed_elements"] + right["bitwise_changed_elements"]
    return {
        "frames": frames,
        "elements": count,
        "bitwise_equal": changed == 0,
        "bitwise_changed_elements": changed,
        "rmse": math.sqrt(sum_squared / count),
        "mean_absolute_error": sum_absolute / count,
        "max_absolute_error": max_absolute,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-index", required=True, help="Freshly authenticated segment-edit matrix index")
    parser.add_argument("--original-decoder-dir", required=True)
    parser.add_argument("--edited-decoder-dir", required=True)
    parser.add_argument("--output", required=True, help="Fresh JSON comparison receipt path")
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.edit_index, args.original_decoder_dir, args.edited_decoder_dir)
        output = Path(args.output).absolute()
        exact._write_fresh_json(output, report)
    except (OSError, exact.ResumeReferenceError) as error:
        print(f"Hy-Nomas segment-edit video comparison rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"pair_integrity_passed": True, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
