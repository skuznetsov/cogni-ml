#!/usr/bin/env python3
"""Decode one authenticated Hy-Nomas recorder final latent with pinned PixelUMM.

This is an upstream RGB decode diagnostic, not a native comparison.  It reads
the saved recorder trace and same-lineage decoder noise, streams one BF16
PixelUMM block at a time, and substitutes the existing gathered spatial-tile
attention adapter for upstream FlexAttention.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import torch

SUPPORT = Path(__file__).resolve().parent
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_activation_trace as activation_trace  # noqa: E402
import hy_nomas_generation_inputs as generation_inputs  # noqa: E402
import hy_nomas_pixelumm_checkpoint as checkpoint  # noqa: E402
import hy_nomas_pixelumm_head_checkpoint as head_checkpoint  # noqa: E402
import hy_nomas_pixelumm_reference as block_reference  # noqa: E402
import hy_nomas_pixelumm_real_input as real_input  # noqa: E402
import hy_nomas_soft_route_render_reference as render_reference  # noqa: E402
import hy_nomas_upstream_activation_recording as recorder  # noqa: E402
import pixelumm_generation_reference as pixelumm_reference  # noqa: E402


SCHEMA = "cogni_ml.hy_nomas_upstream_trace_decode/v1"
RUN_ROOT = Path("/opt/mldata_ssd/cogni-ml")
WEIGHTS_ROOT = Path("/opt/mldata_ssd/models/hy-nomas/pixelumm-ft-bac719a8/weights").resolve()
RECOVERY_ROOT = WEIGHTS_ROOT.parent / "receipts"
CACHE_MANIFEST = WEIGHTS_ROOT / "pixelumm-thirty-six-bf16-cache.json"
CACHE_PAYLOAD = WEIGHTS_ROOT / "pixelumm-thirty-six-bf16-cache.bin"
HEADER_PATH = WEIGHTS_ROOT / "pixelumm-header.json"
STEM_PATH = WEIGHTS_ROOT / "stem-f32.safetensors"
HEAD_PATH = WEIGHTS_ROOT / "head-f32.safetensors"
RECOVERY_BUNDLE = RECOVERY_ROOT / "recovery-bundle.json"
SOURCE_DOWNLOAD = RECOVERY_ROOT / "source-download.json"
UPSTREAM_ROOT = real_input.UPSTREAM_ROOT.resolve()

RECORDER_SCHEMA = recorder.SCHEMA
TRACE_SCHEMA = activation_trace.SCHEMA
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_MODEL_SHA256 = real_input.UPSTREAM_MODEL_SHA256
DCAE_SOURCE_SHA256 = real_input.DCAE_ENCODER_SHA256
AUTOENCODER_SOURCE_SHA256 = real_input.PIXELUMM_NOISE_SHA256
HEADER_SHA256 = checkpoint.HEADER_SHA256
SOURCE_CHECKPOINT_SHA256 = "0bbfe8fd44bd2ebf1059fe65079caba50fd6f52ea8e1b5b35393e3050b9b8378"
RECOVERY_BUNDLE_SHA256 = "674b7724dbe34961d0876a7f4f787482dabffd7db0d2e928f1c912510941e698"
SOURCE_DOWNLOAD_SHA256 = "968ad3227bcc5a52d5989cd69f55530d256d465cb3a539548ca564d5be876e83"
CACHE_MANIFEST_SHA256 = "7cbf6d8a3587f0cd23e07120a9ad1a222b4ffadfa4c1c840609c425d16e3476d"
CACHE_PAYLOAD_SHA256 = "152ba021b52d9546e4c8821b17ab11f67f46486d521b500de3766f14dd2217ca"
STEM_SHA256 = "7eb73ebf42f6681e21870dc2d2d4b592846ba15ed5bf5f9a6c7a83d39c2d8d04"
HEAD_SHA256 = "d0e985e45b6b5fcebde381dd4b61f3b77394bd288323cd8afe2a7604584e6e9f"
RECOVERY_CACHE_BYTES = 13_892_143_104
FULL_CHECKPOINT_BYTES = 31_553_868_864
CACHE_BLOCK_BYTES = 385_892_864
GIB = 1024**3
HASH_CHUNK_BYTES = 8 * 1024 * 1024
MAX_ESTIMATED_PROCESS_BYTES = 10 * GIB
MPS_TORCH_VERSION = "2.8.0"
ATTENTION_SCOPE = "gathered_spatial_tiles_qk_softmax_pv_fp32_accum"
LATENT_SHAPE = (1, 32, 30, 9, 16)
NOISE_SHAPE = (1, 3, 120, 288, 512)
RAW_SHAPE = NOISE_SHAPE
LATENT_BYTES = math.prod(LATENT_SHAPE) * 4
NOISE_BYTES = math.prod(NOISE_SHAPE) * 4
RAW_BYTES = math.prod(RAW_SHAPE) * 4
DISPLAY_RGB_BYTES = 120 * 288 * 512 * 3
WINDOW_STARTS = (0, 5, 10, 15, 20)
WINDOW_LATENTS = 10
WINDOW_PIXELS = 40
ARMS = ("zero", "A1_high", "A2_high")


class DecodeError(ValueError):
    """An input, source, cache, runtime, or output failed closed admission."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DecodeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str, *, limit: int = 4 * 1024 * 1024) -> tuple[dict, bytes]:
    raw = _read_regular_bytes(path, label, max_bytes=limit)
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError, DecodeError) as error:
        raise DecodeError(f"{label} is invalid JSON: {error}") from error
    if type(value) is not dict:
        raise DecodeError(f"{label} must be a JSON object")
    return value, raw


def _path_matches_fd(path: Path, descriptor: int, label: str) -> None:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise DecodeError(f"cannot revalidate {label} identity") from error
    if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
        raise DecodeError(f"{label} changed while it was being authenticated")


def _identity_from_stat(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_regular_snapshot(
    path: Path, label: str, *, max_bytes: int | None = None,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read and bind exact consumed bytes through one no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DecodeError(f"cannot open {label} without following links") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (max_bytes is not None and info.st_size > max_bytes):
            raise DecodeError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise DecodeError(f"{label} changed or was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise DecodeError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        if (after.st_size != info.st_size or after.st_mtime_ns != info.st_mtime_ns
                or after.st_ctime_ns != info.st_ctime_ns):
            raise DecodeError(f"{label} changed while reading")
        _path_matches_fd(path, descriptor, label)
        return b"".join(chunks), _identity_from_stat(after)
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, label: str, *, max_bytes: int | None = None) -> bytes:
    return _read_regular_snapshot(path, label, max_bytes=max_bytes)[0]


def _read_range_and_hash_file(
    path: Path, label: str, *, offset: int, nbytes: int,
    expected_file_sha256: str, expected_file_bytes: int,
) -> tuple[bytearray, tuple[int, int, int, int, int]]:
    """Hash a full immutable payload and return its authenticated range bytes."""
    if (type(offset) is not int or offset < 0 or type(nbytes) is not int or nbytes < 0
            or type(expected_file_bytes) is not int or expected_file_bytes < offset + nbytes):
        raise DecodeError(f"{label} range is invalid")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DecodeError(f"cannot open {label} without following links") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_file_bytes:
            raise DecodeError(f"{label} extent changed")
        digest = hashlib.sha256()
        selected = bytearray()
        position = 0
        while position < before.st_size:
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, before.st_size - position))
            if not chunk:
                raise DecodeError(f"{label} was truncated while reading")
            digest.update(chunk)
            lo, hi = max(offset, position), min(offset + nbytes, position + len(chunk))
            if hi > lo:
                selected.extend(chunk[lo - position:hi - position])
            position += len(chunk)
        after = os.fstat(descriptor)
        if (after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns):
            raise DecodeError(f"{label} changed while reading")
        _path_matches_fd(path, descriptor, label)
        _check_digest(digest.hexdigest(), expected_file_sha256, label)
        if len(selected) != nbytes:
            raise DecodeError(f"{label} selected range is truncated")
        return selected, _identity_from_stat(after)
    finally:
        os.close(descriptor)


def _canonical_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise DecodeError(f"{label} must be a regular non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DecodeError(f"cannot resolve {label}") from error
    if path.is_absolute() and path != resolved:
        raise DecodeError(f"{label} must use its canonical absolute path")
    return resolved


def _check_digest(actual: str, expected: str, label: str) -> None:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or actual != expected):
        raise DecodeError(f"{label} SHA-256 mismatch")


def _expected_replay_gates(arm: str) -> list[str]:
    return [f"{arm}.step{step}.latent" for step in range(1, 5)]


def validate_replay_document(
    replay: Mapping[str, Any], *, arm: str, final_latent_path: Path,
    expected_latent_sha256: str,
) -> None:
    """Admit only the completed four-boundary exact replay that wrote the latent."""
    if arm not in ARMS:
        raise DecodeError("recorded arm is outside the accepted render cases")
    if replay.get("schema") != RECORDER_SCHEMA or replay.get("pilot") is not False:
        raise DecodeError("recorder replay is not a completed non-pilot trace")
    if replay.get("all_euler_boundaries_exact") is not True:
        raise DecodeError("recorder did not establish exact Euler boundary replay")
    if replay.get("trace_hashes_validated") is not True:
        raise DecodeError("recorder did not validate the completed trace hashes")
    gates = replay.get("gates")
    if not isinstance(gates, list) or len(gates) != 4:
        raise DecodeError("recorder replay must contain exactly four Euler gates")
    for gate, expected_name in zip(gates, _expected_replay_gates(arm)):
        if (type(gate) is not dict or gate.get("name") != expected_name
                or gate.get("bitwise_equal") is not True
                or type(gate.get("max_abs_error")) is not float
                or gate.get("max_abs_error") != 0.0):
            raise DecodeError(f"Euler replay gate failed or changed order: {expected_name}")
    runtime = replay.get("runtime")
    diagnostic_runtime = runtime.get("diagnostic_activation_exports") if type(runtime) is dict else None
    if (type(runtime) is not dict or runtime.get("device") != "mps"
            or runtime.get("total_block_invocations") != 432
            or type(diagnostic_runtime) is not dict
            or diagnostic_runtime.get("block_calls") != 432
            or diagnostic_runtime.get("source_math_device") != "mps"
            or diagnostic_runtime.get("exports_add_diagnostic_waits") is not True):
        raise DecodeError("recorder runtime receipt does not describe the complete four-step MPS pass")
    latent = replay.get("final_latent")
    if (type(latent) is not dict
            or set(latent) != {"path", "sha256", "shape", "dtype"}
            or latent.get("shape") != list(LATENT_SHAPE)
            or latent.get("dtype") != "float32"):
        raise DecodeError("replay final-latent binding is malformed")
    if latent.get("path") != str(final_latent_path):
        raise DecodeError("replay final-latent path is not the canonical recording sibling")
    if latent.get("sha256") != expected_latent_sha256:
        raise DecodeError("replay final-latent digest differs from caller authorization")


def validate_recording_metadata(
    index: Mapping[str, Any], *, arm: str, expected_inputs: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate recorder identity and the source/input lineage without touching models."""
    if arm not in ARMS:
        raise DecodeError("recorded arm is outside the accepted render cases")
    if (index.get("schema") != TRACE_SCHEMA or index.get("version") != 1
            or index.get("complete") is not True):
        raise DecodeError("activation index must be a completed supported trace")
    metadata = index.get("metadata")
    if type(metadata) is not dict or metadata.get("schema") != RECORDER_SCHEMA:
        raise DecodeError("activation index does not belong to the upstream recorder")
    if metadata.get("arm") != arm or metadata.get("diagnostic_only") is not True:
        raise DecodeError("activation index arm/diagnostic scope mismatch")
    if metadata.get("upstream_revision") != UPSTREAM_REVISION:
        raise DecodeError("activation trace upstream revision mismatch")
    plan = metadata.get("plan")
    if (type(plan) is not dict or plan.get("steps") != 4 or plan.get("blocks") != 54
            or plan.get("block_calls") != 432 or plan.get("recording_exports_are_diagnostic") is not True):
        raise DecodeError("activation trace does not bind the accepted four-step recorder plan")
    source = metadata.get("accepted_source")
    if (type(source) is not dict or source.get("path") != str(recorder.BASELINE.resolve())
            or source.get("sha256") != recorder.BASELINE_SHA256):
        raise DecodeError("activation trace accepted-source binding mismatch")
    model = metadata.get("model")
    if (type(model) is not dict or model.get("upstream_revision") != UPSTREAM_REVISION
            or not _is_pinned_upstream_path(model.get("upstream_path"))):
        raise DecodeError("activation trace source model binding mismatch")
    source_hashes = model.get("source_hashes")
    if type(source_hashes) is not dict:
        raise DecodeError("activation trace source-code hash inventory is missing")
    for relative, expected in (
        (real_input.UPSTREAM_MODEL_RELATIVE, UPSTREAM_MODEL_SHA256),
        (real_input.PIXELUMM_NOISE_RELATIVE, AUTOENCODER_SOURCE_SHA256),
    ):
        if source_hashes.get(relative) != expected:
            raise DecodeError(f"activation trace source-code digest mismatch: {relative}")
    if expected_inputs is not None and metadata.get("inputs") != expected_inputs:
        raise DecodeError("activation trace inputs differ from current authenticated render ancestry")
    diagnostic_sources = metadata.get("diagnostic_sources")
    try:
        current_diagnostic_sources = recorder.diagnostic_source_bindings()
    except (OSError, ValueError) as error:
        raise DecodeError("cannot revalidate recorder diagnostic source inventory") from error
    if diagnostic_sources != current_diagnostic_sources:
        raise DecodeError("activation trace diagnostic source inventory changed")
    for field, source_path in (
        ("render_helper_sha256", Path(render_reference.__file__)),
        ("recorder_sha256", Path(recorder.__file__)),
    ):
        digest = metadata.get(field)
        if not isinstance(digest, str) or _sha256_file(source_path) != digest:
            raise DecodeError(f"activation trace {field} no longer matches its producer")
    return metadata


def _is_pinned_upstream_path(value: Any) -> bool:
    """Accept path aliases only when they resolve to the pinned source checkout."""
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        return False
    try:
        return path.resolve(strict=True) == UPSTREAM_ROOT.resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def _validate_replay_trace_counts(replay: Mapping[str, Any], index: Mapping[str, Any]) -> None:
    records = index.get("records")
    total_bytes = index.get("total_bytes")
    if (type(records) is not list or type(total_bytes) is not int
            or type(replay.get("trace_records")) is not int
            or replay.get("trace_records") != len(records)
            or type(replay.get("trace_payload_bytes")) is not int
            or replay.get("trace_payload_bytes") != total_bytes):
        raise DecodeError("replay trace record/byte counts differ from the authenticated index")


def expand_bf16_le_to_f32_bytes(payload: bytes) -> bytes:
    """Expand little-endian BF16 words losslessly to little-endian F32 bytes."""
    if len(payload) % 2:
        raise DecodeError("BF16 payload byte count must be even")
    if sys.byteorder != "little":
        raise DecodeError("this recorder format requires a little-endian host")
    import numpy as np

    words = np.frombuffer(payload, dtype="<u2")
    expanded = (words.astype(np.uint32) << np.uint32(16)).astype("<u4", copy=False)
    return expanded.tobytes(order="C")


def _validate_final_latent(
    run_dir: Path, index: Mapping[str, Any], replay: Mapping[str, Any],
    *, arm: str, expected_latent_sha256: str,
) -> tuple[Path, dict, bytes, tuple[int, int, int, int, int]]:
    latent_path = _canonical_file(run_dir / "final-latent.f32le", "final latent")
    latent_bytes, latent_identity = _read_regular_snapshot(
        latent_path, "final latent", max_bytes=LATENT_BYTES,
    )
    if len(latent_bytes) != LATENT_BYTES:
        raise DecodeError("final latent byte extent differs from pinned 30x512x288 geometry")
    _check_digest(hashlib.sha256(latent_bytes).hexdigest(), expected_latent_sha256, "final latent")
    validate_replay_document(
        replay, arm=arm, final_latent_path=latent_path,
        expected_latent_sha256=expected_latent_sha256,
    )
    records = {record.get("name"): record for record in index.get("records", [])
               if type(record) is dict}
    name = f"step4.{arm}.head.after_euler"
    record = records.get(name)
    if (type(record) is not dict or record.get("dtype") != "bfloat16"
            or record.get("shape") != list(LATENT_SHAPE)
            or record.get("nbytes") != LATENT_BYTES // 2):
        raise DecodeError("trace has no exact BF16 step-4 post-Euler latent record")
    shard_name = record.get("shard")
    if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
        raise DecodeError("final latent trace record has an unsafe shard reference")
    shard_path = _canonical_file(run_dir / "activations" / shard_name, "final latent shard")
    words = _read_file_extent(shard_path, "final latent shard record",
                              offset=record["offset"], nbytes=record["nbytes"])
    if len(words) != record["nbytes"]:
        raise DecodeError("final latent trace record is truncated")
    _check_digest(hashlib.sha256(words).hexdigest(), record.get("sha256"),
                  "final latent BF16 trace record")
    expanded = expand_bf16_le_to_f32_bytes(words)
    if expanded != latent_bytes:
        raise DecodeError("F32 final latent is not the exact BF16 trace expansion")
    import numpy as np

    if not np.isfinite(np.frombuffer(latent_bytes, dtype="<f4")).all():
        raise DecodeError("final latent contains non-finite values")
    return latent_path, record, latent_bytes, latent_identity


def _read_file_extent(path: Path, label: str, *, offset: int, nbytes: int) -> bytearray:
    if type(offset) is not int or offset < 0 or type(nbytes) is not int or nbytes < 0:
        raise DecodeError(f"{label} range is malformed")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DecodeError(f"cannot open {label} without following links") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or offset + nbytes > before.st_size:
            raise DecodeError(f"{label} extent exceeds its file")
        os.lseek(descriptor, offset, os.SEEK_SET)
        payload = bytearray()
        remaining = nbytes
        while remaining:
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (len(payload) != nbytes or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns):
            raise DecodeError(f"{label} changed or was truncated while reading")
        _path_matches_fd(path, descriptor, label)
        return payload
    finally:
        os.close(descriptor)


def _authenticate_trace(
    run_dir: Path, *, index_sha256: str, replay_sha256: str,
    latent_sha256: str, authenticate_render_ancestry: bool = True,
) -> dict[str, Any]:
    """Bind caller-authorized replay/index/latent and validate all trace shards."""
    if not run_dir.is_absolute() or run_dir.is_symlink() or not run_dir.is_dir():
        raise DecodeError("recording directory must be an existing absolute non-symlink directory")
    resolved_run = run_dir.resolve(strict=True)
    if resolved_run != run_dir or not resolved_run.is_relative_to(RUN_ROOT.resolve(strict=True)):
        raise DecodeError("recording directory must be canonical beneath /opt/mldata_ssd/cogni-ml")
    index_path = _canonical_file(run_dir / "activations" / "index.json", "activation index")
    replay_path = _canonical_file(run_dir / "replay.json", "replay receipt")
    index, index_bytes = _read_json(index_path, "activation index", limit=64 * 1024 * 1024)
    replay, replay_bytes = _read_json(replay_path, "replay receipt")
    actual_index_sha = hashlib.sha256(index_bytes).hexdigest()
    actual_replay_sha = hashlib.sha256(replay_bytes).hexdigest()
    _check_digest(actual_index_sha, index_sha256, "activation index")
    _check_digest(actual_replay_sha, replay_sha256, "replay receipt")
    try:
        validated_index = activation_trace.validate_trace(run_dir / "activations", require_complete=True)
    except (OSError, ValueError) as error:
        raise DecodeError(f"activation trace failed structural/hash validation: {error}") from error
    if validated_index != index:
        raise DecodeError("activation trace parser views differ for the saved index")
    metadata = index.get("metadata")
    arm = metadata.get("arm") if type(metadata) is dict else None
    validate_recording_metadata(index, arm=arm)
    _validate_replay_trace_counts(replay, index)
    latent_path, latent_record, latent_bytes, latent_identity = _validate_final_latent(
        run_dir, index, replay, arm=arm, expected_latent_sha256=latent_sha256,
    )
    authenticated_inputs = None
    noise = None
    if authenticate_render_ancestry:
        checked = render_reference.authenticate_inputs()
        authenticated_inputs = render_reference._input_document(checked)
        validate_recording_metadata(index, arm=arm, expected_inputs=authenticated_inputs)
        noise = _validate_noise_anchor(authenticated_inputs)
    return {
        "run_dir": run_dir,
        "index_path": index_path,
        "index_sha256": actual_index_sha,
        "replay_path": replay_path,
        "replay_sha256": actual_replay_sha,
        "latent_path": latent_path,
        "latent_sha256": latent_sha256,
        "latent_bytes": latent_bytes,
        "latent_file_identity": latent_identity,
        "latent_record": latent_record,
        "index": index,
        "replay": replay,
        "arm": arm,
        "authenticated_inputs": authenticated_inputs,
        "noise": noise,
        "render_checked": checked if authenticate_render_ancestry else None,
    }


def _validate_noise_anchor(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact authenticated anchor payload and its decoder.noise record."""
    anchor_binding = inputs.get("anchor")
    if type(anchor_binding) is not dict:
        raise DecodeError("recorder input ancestry is missing the conditioning anchor")
    anchor_path = _canonical_file(Path(anchor_binding.get("path", "")), "anchor manifest")
    anchor, anchor_bytes = _read_json(anchor_path, "anchor manifest")
    anchor_sha = hashlib.sha256(anchor_bytes).hexdigest()
    _check_digest(anchor_sha, render_reference.ANCHOR_SHA256, "anchor manifest")
    if anchor_binding.get("sha256") != anchor_sha:
        raise DecodeError("recorder anchor manifest digest differs from authenticated render")
    try:
        generation_inputs.validate_document(anchor, payload_bytes=anchor.get("payload_bytes"))
    except generation_inputs.InputsError as error:
        raise DecodeError(f"anchor manifest rejected: {error}") from error
    payload_path = _canonical_file(Path(anchor_binding.get("payload_path", "")), "anchor payload")
    if payload_path != anchor_path.parent / anchor.get("payload"):
        raise DecodeError("anchor payload path does not match the manifest basename")
    payload_size = anchor.get("payload_bytes")
    if type(payload_size) is not int or payload_size < 0:
        raise DecodeError("anchor manifest payload byte count is malformed")
    if (anchor_binding.get("payload_sha256") != render_reference.ANCHOR_PAYLOAD_SHA256
            or anchor_binding.get("payload_bytes") != anchor.get("payload_bytes")):
        raise DecodeError("recorder anchor payload binding differs from the pinned source")
    unused = anchor_binding.get("unused_latent")
    if type(unused) is not dict or unused.get("used") is not False:
        raise DecodeError("anchor latent must remain explicitly unused by the DiT replay")
    noise = anchor.get("tensors", {}).get("decoder.noise")
    if (type(noise) is not dict or noise.get("dtype") != "F32"
            or noise.get("shape") != list(NOISE_SHAPE)
            or noise.get("nbytes") != NOISE_BYTES):
        raise DecodeError("same-lineage decoder.noise record has wrong geometry")
    noise_bytes, payload_identity = _read_range_and_hash_file(
        payload_path, "anchor payload", offset=noise["offset"], nbytes=noise["nbytes"],
        expected_file_sha256=render_reference.ANCHOR_PAYLOAD_SHA256,
        expected_file_bytes=payload_size,
    )
    digest_sha = hashlib.sha256(noise_bytes).hexdigest()
    _check_digest(digest_sha, noise["sha256"], "decoder.noise tensor")
    return {"manifest_path": str(anchor_path), "manifest_sha256": anchor_sha,
            "payload_path": str(payload_path), "payload_sha256": render_reference.ANCHOR_PAYLOAD_SHA256,
            "payload_bytes": payload_size,
            "noise_record": dict(noise), "noise_record_sha256": digest_sha,
            "noise_bytes": noise_bytes, "payload_file_identity": payload_identity}


def _authenticate_weight_bundle() -> dict[str, Any]:
    """Authenticate recovered header/stem/head and all cache metadata without a whole-cache map."""
    recovery, recovery_bytes = _read_json(RECOVERY_BUNDLE, "recovery bundle")
    download, download_bytes = _read_json(SOURCE_DOWNLOAD, "source-download receipt")
    _check_digest(hashlib.sha256(recovery_bytes).hexdigest(), RECOVERY_BUNDLE_SHA256,
                  "recovery bundle")
    _check_digest(hashlib.sha256(download_bytes).hexdigest(), SOURCE_DOWNLOAD_SHA256,
                  "source-download receipt")
    source_record = recovery.get("source_checkpoint")
    if (type(source_record) is not dict or source_record.get("bytes") != FULL_CHECKPOINT_BYTES
            or source_record.get("sha256") != SOURCE_CHECKPOINT_SHA256
            or source_record.get("full_file_hash_verified") is not True
            or download.get("sha256") != SOURCE_CHECKPOINT_SHA256
            or download.get("bytes") != FULL_CHECKPOINT_BYTES
            or download.get("full_file_hash_verified") is not True):
        raise DecodeError("recovered source checkpoint lineage is incomplete")
    raw_header_path = _canonical_file(HEADER_PATH, "PixelUMM checkpoint header")
    raw_header = _read_regular_bytes(raw_header_path, "PixelUMM checkpoint header", max_bytes=16 * 1024 * 1024)
    _check_digest(hashlib.sha256(raw_header).hexdigest(), HEADER_SHA256,
                  "PixelUMM checkpoint header")
    try:
        header = checkpoint.parse_pinned_header(raw_header)
    except checkpoint.InventoryError as error:
        raise DecodeError(f"PixelUMM checkpoint header rejected: {error}") from error
    stem_path = _canonical_file(STEM_PATH, "PixelUMM stem weights")
    stem_bytes = _read_regular_bytes(stem_path, "PixelUMM stem weights", max_bytes=STEM_PATH.stat().st_size)
    head_path = _canonical_file(HEAD_PATH, "PixelUMM head weights")
    head_bytes = _read_regular_bytes(head_path, "PixelUMM head weights", max_bytes=HEAD_PATH.stat().st_size)
    _check_digest(hashlib.sha256(stem_bytes).hexdigest(), STEM_SHA256, "PixelUMM stem weights")
    _check_digest(hashlib.sha256(head_bytes).hexdigest(), HEAD_SHA256, "PixelUMM head weights")
    if (recovery.get("header", {}).get("sha256") != HEADER_SHA256
            or recovery.get("stem", {}).get("sha256") != STEM_SHA256
            or recovery.get("head", {}).get("sha256") != HEAD_SHA256):
        raise DecodeError("recovered stem/head/header hashes do not match the authenticated bundle")
    manifest_path = _canonical_file(CACHE_MANIFEST, "36-block BF16 cache manifest")
    raw_manifest = _read_regular_bytes(manifest_path, "36-block BF16 cache manifest",
                                       max_bytes=64 * 1024 * 1024)
    _check_digest(hashlib.sha256(raw_manifest).hexdigest(), CACHE_MANIFEST_SHA256,
                  "36-block BF16 cache manifest")
    try:
        manifest = json.loads(raw_manifest, object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError, DecodeError) as error:
        raise DecodeError(f"36-block BF16 cache manifest is invalid: {error}") from error
    payload_path = _canonical_file(CACHE_PAYLOAD, "36-block BF16 cache payload")
    if (payload_path.stat().st_size != RECOVERY_CACHE_BYTES
            or manifest.get("schema") != "cogni_ml.pixelumm_bf16_weight_cache/v1"
            or manifest.get("source_commit") != checkpoint.SOURCE_COMMIT
            or manifest.get("upstream_revision") != UPSTREAM_REVISION
            or manifest.get("upstream_model_sha256") != UPSTREAM_MODEL_SHA256
            or manifest.get("checkpoint_header_sha256") != HEADER_SHA256
            or manifest.get("payload") != payload_path.name
            or manifest.get("payload_bytes") != RECOVERY_CACHE_BYTES
            or manifest.get("payload_sha256") != CACHE_PAYLOAD_SHA256):
        raise DecodeError("recovered full36 cache identity/extent differs from pinned metadata")
    bundle_cache = recovery.get("cache_payload")
    if (type(bundle_cache) is not dict or bundle_cache.get("bytes") != RECOVERY_CACHE_BYTES
            or bundle_cache.get("sha256") != CACHE_PAYLOAD_SHA256
            or bundle_cache.get("matches_original_payload_sha256") is not True):
        raise DecodeError("recovery bundle does not bind the full36 BF16 payload")
    conversion = manifest.get("conversion")
    if (type(conversion) is not dict or conversion.get("device") != "cpu"
            or conversion.get("rounding") != "round_to_nearest_even"
            or conversion.get("mps_conversion_claimed") is not False):
        raise DecodeError("cache precision provenance is not the recovered CPU RNE conversion")
    _validate_cache_manifest_structure(manifest, header, payload_path.stat().st_size)
    return {
        "header": header,
        "header_path": raw_header_path,
        "stem_path": stem_path,
        "stem_bytes": stem_bytes,
        "head_path": head_path,
        "head_bytes": head_bytes,
        "cache_manifest": manifest,
        "cache_manifest_path": manifest_path,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
        "cache_payload_path": payload_path,
        "cache_payload_sha256": CACHE_PAYLOAD_SHA256,
        "recovery_bundle_sha256": RECOVERY_BUNDLE_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
    }


def _validate_cache_manifest_structure(
    manifest: Mapping[str, Any], header: Mapping[str, Any], payload_bytes: int,
    *, expected_blocks: int = 36,
) -> None:
    """Check every cache key, extent and source block range without reading the payload."""
    if expected_blocks != 36:
        raise DecodeError("only the authenticated complete 36-block cache is admitted")
    blocks = manifest.get("blocks")
    records = manifest.get("weights")
    if (type(blocks) is not dict or set(blocks) != {str(i) for i in range(36)}
            or type(records) is not dict):
        raise DecodeError("full36 BF16 cache block/weight inventories are malformed")
    cursor = 0
    expected_names: set[str] = set()
    for block_index in range(36):
        schema = checkpoint.block_schema(block_index)
        block = blocks[str(block_index)]
        if (type(block) is not dict or block.get("selected_block") != block_index
                or block.get("source_file_sha256") != SOURCE_CHECKPOINT_SHA256
                or block.get("conversion_device") != "cpu"
                or block.get("bf16_bytes") != CACHE_BLOCK_BYTES
                or block.get("source_f32_bytes") != CACHE_BLOCK_BYTES * 2):
            raise DecodeError(f"cache provenance differs for PixelUMM block {block_index}")
        source_extents = []
        # The authenticated BF16 cache is serialized in lexical tensor-name
        # order, matching the block loader; module state-dict insertion order
        # is different for this upstream architecture.
        for name, shape in sorted(schema.items()):
            full_name = f"block{block_index}.{name}"
            expected_names.add(full_name)
            source_key = f"decoder.layers.{block_index}.{name}"
            source = header.get(source_key)
            if type(source) is not dict or source.get("data_offsets") is None:
                raise DecodeError(f"pinned header is missing {source_key}")
            source_extents.append(source["data_offsets"])
            record = records.get(full_name)
            nbytes = math.prod(shape) * 2
            if (type(record) is not dict or record.get("dtype") != "BF16"
                    or record.get("shape") != list(shape) or record.get("nbytes") != nbytes
                    or record.get("offset") != cursor
                    or not _is_digest(record.get("sha256"))):
                raise DecodeError(f"cache BF16 tensor metadata mismatch: {full_name}")
            cursor += nbytes
        expected_source_extent = [min(item[0] for item in source_extents),
                                  max(item[1] for item in source_extents)]
        if (block.get("source_data_offsets") != expected_source_extent
                or expected_source_extent[1] - expected_source_extent[0] != CACHE_BLOCK_BYTES * 2):
            raise DecodeError(f"cached source range differs from pinned header for block {block_index}")
    if set(records) != expected_names or cursor != payload_bytes:
        raise DecodeError("BF16 cache records do not cover the exact full36 payload")


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _verify_upstream_sources(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise DecodeError("pinned upstream source directory must be canonical and non-symlink")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--",
             real_input.UPSTREAM_MODEL_RELATIVE, real_input.DCAE_ENCODER_RELATIVE,
             real_input.PIXELUMM_NOISE_RELATIVE], text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DecodeError("cannot verify pinned upstream source tree") from error
    if revision != UPSTREAM_REVISION or changed.strip():
        raise DecodeError("upstream source tree revision/working files changed")
    hashes = {"revision": revision}
    for relative, expected in (
        (real_input.UPSTREAM_MODEL_RELATIVE, UPSTREAM_MODEL_SHA256),
        (real_input.DCAE_ENCODER_RELATIVE, DCAE_SOURCE_SHA256),
        (real_input.PIXELUMM_NOISE_RELATIVE, AUTOENCODER_SOURCE_SHA256),
    ):
        path = _canonical_file(root / relative, f"pinned upstream source {relative}")
        _check_digest(_sha256_file(path), expected, f"pinned upstream source {relative}")
        hashes[relative] = expected
    return hashes


def _load_authenticated_noise(noise_bytes: bytearray) -> torch.Tensor:
    if len(noise_bytes) != NOISE_BYTES or len(noise_bytes) % 4:
        raise DecodeError("authenticated decoder.noise bytes have wrong extent")
    return torch.frombuffer(noise_bytes, dtype=torch.float32).view(NOISE_SHAPE)


def _load_authenticated_latent(latent_bytes: bytes) -> torch.Tensor:
    if len(latent_bytes) != LATENT_BYTES or len(latent_bytes) % 4:
        raise DecodeError("authenticated final-latent bytes have wrong extent")
    return torch.frombuffer(bytearray(latent_bytes), dtype=torch.float32).view(LATENT_SHAPE)


def _assert_file_identity(path: Path, expected: tuple[int, int, int, int, int], label: str) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise DecodeError(f"{label} disappeared after admission") from error
    if not stat.S_ISREG(current.st_mode) or _identity_from_stat(current) != expected:
        raise DecodeError(f"{label} changed after admission; refusing model execution")


def _revalidate_admission_sources(admission: Mapping[str, Any]) -> None:
    """Reject stale source paths before MPS while consuming only admitted snapshots."""
    _assert_file_identity(admission["latent_path"], admission["latent_file_identity"],
                          "final latent")
    noise = admission.get("noise")
    if type(noise) is not dict:
        raise DecodeError("authenticated decoder.noise snapshot is missing")
    _assert_file_identity(Path(noise["payload_path"]), noise["payload_file_identity"],
                          "anchor payload")


def _load_stem(source: Any, stem_data: bytes, device: torch.device) -> torch.nn.Module:
    from safetensors.torch import load as load_safetensors

    with torch.device("meta"):
        stem = real_input._PixelUMMInputStem(source)
    shapes = {name: tuple(value.shape) for name, value in stem.state_dict().items()}
    expected = {"decoder." + name for name in shapes}
    try:
        source_state = load_safetensors(stem_data)
    except Exception as error:
        raise DecodeError("authenticated stem safetensors bytes could not be parsed") from error
    if set(source_state) != expected:
        raise DecodeError("recovered stem safetensors keys differ from the pinned stem inventory")
    state = {}
    for name in sorted(expected):
        value = source_state[name]
        local = name.removeprefix("decoder.")
        if value.dtype != torch.float32 or tuple(value.shape) != shapes[local]:
            raise DecodeError(f"recovered stem tensor shape/dtype mismatch: {name}")
        state[local] = value
    stem.load_state_dict(state, strict=True, assign=True)
    del source_state, state
    stem.eval().requires_grad_(False)
    stem.to(device=device, dtype=torch.bfloat16)
    return stem


def _load_head(source: Any, head_data: bytes, device: torch.device):
    from safetensors.torch import load as load_safetensors

    arch = source.QWEN3_8B_GEN
    with torch.device("meta"):
        final_norm = source.RMSNorm(arch.hidden_size, arch.rms_norm_eps)
        head = source.PixelHead(arch.hidden_size)
    destinations = {
        "decoder.final_norm.weight": (final_norm, "weight"),
        "decoder.head.norm.weight": (head.norm, "weight"),
        "decoder.head.linear.weight": (head.linear, "weight"),
        "decoder.head.linear.bias": (head.linear, "bias"),
    }
    expected = {name: tuple(getattr(module, local).shape)
                for name, (module, local) in destinations.items()}
    try:
        state = load_safetensors(head_data)
    except Exception as error:
        raise DecodeError("authenticated head safetensors bytes could not be parsed") from error
    if set(state) != set(head_checkpoint.HEAD_TENSORS):
        raise DecodeError("recovered head safetensors keys differ from the pinned head inventory")
    for name in sorted(expected):
        value = state[name]
        if value.dtype != torch.float32 or tuple(value.shape) != expected[name]:
            raise DecodeError(f"recovered head tensor shape/dtype mismatch: {name}")
    with torch.no_grad():
        for name, (module, local) in destinations.items():
            parameter = torch.nn.Parameter(state[name], requires_grad=False)
            setattr(module, local, parameter)
    del state
    final_norm.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    final_norm.to(device=device, dtype=torch.bfloat16)
    head.to(device=device, dtype=torch.bfloat16)
    return final_norm, head


def _load_cached_layer_for_run(
    source: Any, manifest: Mapping[str, Any], payload_path: Path,
    block_index: int, device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, str]]]:
    """Hash and consume one authenticated BF16 block without opening the full cache."""
    schema = checkpoint.block_schema(block_index)
    records = manifest["weights"]
    block_records = [(name, records[f"block{block_index}.{name}"]) for name in sorted(schema)]
    start = min(record["offset"] for _, record in block_records)
    end = max(record["offset"] + record["nbytes"] for _, record in block_records)
    if end - start != CACHE_BLOCK_BYTES:
        raise DecodeError(f"BF16 block {block_index} does not form one bounded extent")
    verification: list[dict[str, str]] = []
    block_bytes = _read_file_extent(payload_path, f"cached PixelUMM block {block_index}",
                                    offset=start, nbytes=end - start)
    with torch.device("meta"):
        layer = source.GenLayer(source.QWEN3_8B_GEN)
    expected = layer.state_dict()
    state = {}
    for name, record in block_records:
        if name not in expected or tuple(expected[name].shape) != tuple(record["shape"]):
            raise DecodeError(f"cached block {block_index} parameter mismatch: {name}")
        relative = record["offset"] - start
        view = memoryview(block_bytes)[relative:relative + record["nbytes"]]
        actual_sha = hashlib.sha256(view).hexdigest()
        _check_digest(actual_sha, record["sha256"], f"block {block_index} tensor {name}")
        value = torch.frombuffer(view, dtype=torch.bfloat16).view(record["shape"])
        if not value.is_contiguous():
            raise DecodeError(f"cached BF16 tensor is not contiguous: {name}")
        state[name] = value
        verification.append({"name": f"block{block_index}.{name}", "sha256": actual_sha})
    layer.load_state_dict(state, strict=True, assign=True)
    del state, expected, value, view
    layer.eval().requires_grad_(False)
    layer.to(device=device)
    del block_bytes
    gc.collect()
    return layer, verification


def _memory_components(arch: Any) -> dict[str, int]:
    peak = block_reference._estimated_peak_bytes(
        (10, 18, 32), arch, layers=36, resident_layers=1,
        capture_stages=block_reference.FULL_TARGET_STAGES,
        full36_capture_policy=block_reference.THIRTY_SIX_CAPTURE_POLICY,
    )
    weight_stream = CACHE_BLOCK_BYTES * 2
    output_scratch = 3 * RAW_BYTES
    input_and_stem_head = 2 * NOISE_BYTES + 2 * (52_510_720 + head_checkpoint.HEAD_F32_BYTES)
    margin = 1536 * 1024**2
    estimate = peak + weight_stream + output_scratch + input_and_stem_head + margin
    no_capture_reference = block_reference._estimated_peak_bytes(
        (10, 18, 32), arch, layers=1, resident_layers=1,
    )
    return {
        "diagnostic_capture_policy_overbound_bytes": peak,
        "single_block_no_capture_reference_bytes": no_capture_reference,
        "one_block_cpu_staging_and_device_allowance_bytes": weight_stream,
        "noise_and_stem_head_cpu_device_allowance_bytes": input_and_stem_head,
        "raw_display_blender_export_scratch_bytes": output_scratch,
        "allocator_and_capture_margin_bytes": margin,
        "estimated_total_bytes": estimate,
    }


def _memory_plan(arch: Any, *, physical_bytes: int, allocated_bytes: int = 0) -> dict[str, int]:
    if (type(physical_bytes) is not int or physical_bytes <= 0
            or type(allocated_bytes) is not int or allocated_bytes < 0):
        raise DecodeError("memory preflight received invalid physical/allocated bytes")
    components = _memory_components(arch)
    reserve = physical_bytes // 10
    estimate = components["estimated_total_bytes"]
    if estimate > MAX_ESTIMATED_PROCESS_BYTES:
        raise DecodeError("conservative upstream decoder estimate exceeds the 10 GiB admission cap")
    if physical_bytes - allocated_bytes - estimate < reserve:
        raise DecodeError("upstream decode memory preflight would cross the 10% host reserve")
    return {
        "physical_bytes": physical_bytes,
        "already_allocated_bytes": allocated_bytes,
        **components,
        "admission_memory_cap_bytes": MAX_ESTIMATED_PROCESS_BYTES,
        "required_reserve_bytes": reserve,
        "resident_decoder_blocks": 1,
        "full_bf16_cache_mapped": False,
    }


def _physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True,
                                               stderr=subprocess.DEVNULL).strip())
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise DecodeError("cannot determine physical memory for the reserve check") from error
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as error:
        raise DecodeError("cannot determine physical memory for the reserve check") from error


def _validate_runtime() -> dict[str, Any]:
    if torch.__version__ != MPS_TORCH_VERSION:
        raise DecodeError(f"execution requires torch {MPS_TORCH_VERSION}; got {torch.__version__}")
    if not torch.backends.mps.is_available():
        raise DecodeError("PyTorch MPS is unavailable; CPU model fallback is prohibited")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise DecodeError("PYTORCH_ENABLE_MPS_FALLBACK must be exactly 0")
    return {"device": "mps", "torch_version": torch.__version__,
            "pytorch_enable_mps_fallback": "0", "python": sys.version.split()[0]}


def _fresh_output(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or os.path.lexists(path):
        raise DecodeError("execution output must be a fresh absolute directory")
    try:
        parent = path.parent.resolve(strict=True)
        root = Path("/opt/mldata_ssd").resolve(strict=True)
    except OSError as error:
        raise DecodeError("output parent and /opt/mldata_ssd must already exist") from error
    if parent != path.parent or not parent.is_dir() or not parent.is_relative_to(root):
        raise DecodeError("output parent must be canonical and beneath /opt/mldata_ssd")
    required = RAW_BYTES + DISPLAY_RGB_BYTES + 2 * GIB
    if shutil.disk_usage(parent).free < required:
        raise DecodeError("external SSD lacks room for raw/display outputs and reserve")
    return path


def _save_raw(path: Path, tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(dtype=torch.float32).contiguous().cpu()
    import numpy as np

    values = cpu.numpy()
    if not np.isfinite(values).all():
        raise DecodeError("upstream raw output contains non-finite pixels")
    with path.open("xb") as stream:
        values.astype("<f4", copy=False).tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.stat().st_size != RAW_BYTES:
        raise DecodeError("raw output byte extent mismatch")
    return _sha256_file(path)


def _save_display_rgb(path: Path, tensor: torch.Tensor) -> str:
    import numpy as np

    values = tensor.detach().contiguous().cpu().numpy()
    if values.shape != RAW_SHAPE or not np.isfinite(values).all():
        raise DecodeError("postprocessed display output shape/nonfinite check failed")
    rgb = np.rint(np.clip(values[0].transpose(1, 2, 3, 0), 0.0, 1.0) * 255.0)
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    if rgb.shape != (120, 288, 512, 3):
        raise DecodeError("display RGB export geometry mismatch")
    with path.open("xb") as stream:
        rgb.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.stat().st_size != DISPLAY_RGB_BYTES:
        raise DecodeError("display RGB output byte extent mismatch")
    return _sha256_file(path)


def _encode_mp4(rgb_path: Path, mp4_path: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"encoded": False, "reason": "ffmpeg unavailable"}
    command = [ffmpeg, "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s:v", "512x288", "-r", "24", "-i", str(rgb_path), "-frames:v", "120",
               "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4_path)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        try:
            mp4_path.unlink()
        except FileNotFoundError:
            pass
        return {"encoded": False, "reason": completed.stderr[-2000:] or "ffmpeg failed"}
    if not mp4_path.is_file() or mp4_path.stat().st_size <= 0:
        return {"encoded": False, "reason": "ffmpeg returned success without an MP4"}
    return {"encoded": True, "path": str(mp4_path), "bytes": mp4_path.stat().st_size,
            "sha256": _sha256_file(mp4_path), "ffmpeg": Path(ffmpeg).name,
            "fps": 24, "frames": 120, "codec": "libx264", "pixel_format": "yuv420p"}


def run_decode(
    *, admission: Mapping[str, Any], weights: Mapping[str, Any],
    source_hashes: Mapping[str, str], output: Path,
) -> dict[str, Any]:
    """Run the saved latent through pinned upstream PixelUMM and export RGB."""
    _revalidate_admission_sources(admission)
    runtime = _validate_runtime()
    device = torch.device("mps")
    source, model_sha = block_reference.load_upstream(UPSTREAM_ROOT)
    if model_sha != UPSTREAM_MODEL_SHA256:
        raise DecodeError("upstream PixelUMM model source hash changed after preflight")
    dcae_source, source_again, autoencoder_source = real_input._load_modules(UPSTREAM_ROOT)
    if source_again is not source and Path(source_again.__file__).resolve() != Path(source.__file__).resolve():
        raise DecodeError("upstream source import identity differs")
    if (_sha256_file(Path(dcae_source.__file__)) != DCAE_SOURCE_SHA256
            or _sha256_file(Path(autoencoder_source.__file__)) != AUTOENCODER_SOURCE_SHA256):
        raise DecodeError("upstream decoder/blender source hash changed after preflight")
    arch = source.QWEN3_8B_GEN
    physical = _physical_memory_bytes()
    allocated = int(torch.mps.driver_allocated_memory())
    memory = _memory_plan(arch, physical_bytes=physical, allocated_bytes=allocated)
    latent_cpu = _load_authenticated_latent(admission["latent_bytes"])
    latent = latent_cpu.to(device=device)
    del latent_cpu
    noise_cpu = _load_authenticated_noise(admission["noise"]["noise_bytes"])
    noise = noise_cpu.to(device=device)
    del noise_cpu
    admission["noise"].pop("noise_bytes")
    plan, inverse_cpu, _ = block_reference.make_attention_plan(source, (10, 18, 32), tile=8)
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    inverse = inverse_cpu.to(device=device)
    cos, sin = source.mrope_tables(arch, (10, 18, 32), device)
    cos = cos.index_select(1, permutation)
    sin = sin.index_select(1, permutation)
    stem = _load_stem(source, weights["stem_bytes"], device)
    weights.pop("stem_bytes")
    final_norm, head = _load_head(source, weights["head_bytes"], device)
    weights.pop("head_bytes")
    blender = autoencoder_source._TemporalBlender(total_frames=120)
    condition = type("_ConditionProjection", (), {})()
    condition.cond_proj = stem.cond_proj
    condition.arch = arch
    current_windows = []
    checks: list[dict[str, str]] = []
    with torch.inference_mode():
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            for latent_start in WINDOW_STARTS:
                latent_window = latent[:, :, latent_start:latent_start + WINDOW_LATENTS]
                noise_window = noise[:, :, latent_start * 4:(latent_start + WINDOW_LATENTS) * 4]
                xt_tokens = stem.embed(source.patchify(noise_window.to(stem.embed.weight.dtype)))
                condition_tokens = source.DCAEVPixelUMMDecoder._condition_tokens(
                    condition, latent_window.to(stem.cond_proj.weight.dtype)
                )
                current_windows.append((xt_tokens + condition_tokens).index_select(1, permutation))
                del latent_window, noise_window, xt_tokens, condition_tokens
            for block_index in range(36):
                layer, block_checks = _load_cached_layer_for_run(
                    source, weights["cache_manifest"], weights["cache_payload_path"],
                    block_index, device,
                )
                checks.extend(block_checks)
                for window_index, current in enumerate(current_windows):
                    current_windows[window_index], captures = block_reference.run_layer_reference(
                        layer, current, cos, sin, plan,
                    )
                    if captures:
                        raise DecodeError("unexpected decoder capture retained during pure decode")
                del layer
                gc.collect()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        for window_index, latent_start in enumerate(WINDOW_STARTS):
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                tokens = final_norm(current_windows[window_index]).index_select(1, inverse)
            prediction = head(tokens)
            pixels = source.unpatchify(prediction, (10, 18, 32))
            blender.add(pixels, latent_start * 4)
            del tokens, prediction, pixels
    raw = blender.result()
    display = pixelumm_reference.postprocess_bf16(torch, raw)
    # This is the only explicit device completion boundary. Exports and ffmpeg
    # start only after all upstream decoder work has become terminal.
    torch.mps.synchronize()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    marker = output / ".incomplete"
    marker.write_text("upstream trace decode not finalized\n", encoding="utf-8")
    raw_path = output / "raw.f32le"
    display_path = output / "display.rgb24"
    mp4_path = output / "display.mp4"
    raw_sha = _save_raw(raw_path, raw)
    display_sha = _save_display_rgb(display_path, display)
    del raw, display, latent, noise, current_windows, cos, sin, permutation, inverse
    gc.collect()
    mp4 = _encode_mp4(display_path, mp4_path)
    result = {
        "schema": SCHEMA,
        "recording": {
            "run_dir": str(admission["run_dir"]),
            "index_sha256": admission["index_sha256"],
            "replay_sha256": admission["replay_sha256"],
            "final_latent_path": str(admission["latent_path"]),
            "final_latent_sha256": admission["latent_sha256"],
            "final_trace_record": admission["latent_record"]["name"],
            "arm": admission["arm"],
        },
        "inputs": {"anchor_manifest_sha256": admission["noise"]["manifest_sha256"],
                   "anchor_payload_sha256": admission["noise"]["payload_sha256"],
                   "decoder_noise_sha256": admission["noise"]["noise_record_sha256"],
                   "anchor_latent_used": False},
        "upstream": {"revision": UPSTREAM_REVISION, "model_sha256": model_sha,
                     "dcae_source_sha256": source_hashes[real_input.DCAE_ENCODER_RELATIVE],
                     "pixelumm_source_sha256": source_hashes[real_input.PIXELUMM_NOISE_RELATIVE]},
        "weights": {
            "checkpoint_header_sha256": HEADER_SHA256,
            "full_source_checkpoint_sha256": weights["source_checkpoint_sha256"],
            "stem_safetensors_sha256": STEM_SHA256,
            "head_safetensors_sha256": HEAD_SHA256,
            "cache_manifest_sha256": weights["cache_manifest_sha256"],
            "cache_payload_sha256_declared": weights["cache_payload_sha256"],
            "cache_payload_verification": "all 468 BF16 tensor ranges hashed individually as streamed",
            "verified_tensor_count": len(checks),
            "conversion": "CPU IEEE F32 to BF16 round-to-nearest-ties-to-even; MPS conversion not claimed",
            "full_bf16_cache_mapped": False,
            "resident_blocks": 1,
        },
        "decoder": {
            "resolution": "512x288",
            "latent_shape": list(LATENT_SHAPE),
            "noise_shape": list(NOISE_SHAPE),
            "window_latent_starts": list(WINDOW_STARTS),
            "window_latents": WINDOW_LATENTS,
            "overlap_latents": 5,
            "pixel_frame_starts": [start * 4 for start in WINDOW_STARTS],
            "merge": "pinned upstream _TemporalBlender with BF16 window predictions",
            "runtime_dtype": "BF16",
            "attention_adapter": ATTENTION_SCOPE,
            "unmodified_eager_flex_attention_parity": False,
        },
        "runtime": runtime,
        "memory_preflight": memory,
        "outputs": {
            "raw": {"path": str(raw_path), "shape": list(RAW_SHAPE), "dtype": "F32LE",
                    "bytes": raw_path.stat().st_size, "sha256": raw_sha},
            "display_rgb": {"path": str(display_path), "shape": [120, 288, 512, 3],
                            "dtype": "RGB24", "bytes": display_path.stat().st_size,
                            "sha256": display_sha,
                            "conversion": "pinned BF16 postprocess then round-to-nearest-even uint8"},
            "mp4": mp4,
        },
        "claim_scope": "upstream RGB decode of one saved DiT final latent; no native decoder comparison",
    }
    receipt_path = output / "receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    marker.unlink()
    descriptor = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return result


def plan_document() -> dict[str, Any]:
    """Return a CPU-only execution plan; this function does not inspect MPS."""
    arch = SimpleNamespace(hidden_size=4096, intermediate_size=12288, num_heads=32,
                           num_kv_heads=8, head_dim=128)
    memory_components = _memory_components(arch)
    return {
        "schema": SCHEMA,
        "model_launched": False,
        "device": "MPS execution only",
        "required_torch_version": MPS_TORCH_VERSION,
        "required_mps_fallback": "0",
        "geometry": {"latent": list(LATENT_SHAPE), "noise": list(NOISE_SHAPE),
                     "raw_bytes": RAW_BYTES, "display_rgb_bytes": DISPLAY_RGB_BYTES},
        "windows": {"starts": list(WINDOW_STARTS), "latents_each": WINDOW_LATENTS,
                    "overlap_latents": 5, "decoder_blocks": 36},
        "weights": {"cache_manifest": str(CACHE_MANIFEST),
                    "cache_payload_sha256": CACHE_PAYLOAD_SHA256,
                    "full_cache_mapped": False, "resident_decoder_blocks": 1,
                    "per_tensor_hashes_required": 468,
                    "conversion_provenance": "CPU round-to-nearest-even"},
        "attention_adapter": ATTENTION_SCOPE,
        "unmodified_eager_flex_attention_parity": False,
        "memory_plan": {
            **memory_components,
            "admission_memory_cap_bytes": MAX_ESTIMATED_PROCESS_BYTES,
            "minimum_host_reserve_fraction": 0.10,
            "execution_check": "physical memory minus current MPS allocation and estimate must retain 10% host reserve",
            "full_bf16_cache_mapped": False,
        },
        "outputs": ["raw.f32le", "display.rgb24", "display.mp4 if ffmpeg is available", "receipt.json"],
        "terminal_mps_synchronization_before_cpu_export_and_ffmpeg": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--recording-dir", type=Path)
    parser.add_argument("--index-sha256")
    parser.add_argument("--replay-sha256")
    parser.add_argument("--latent-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-live-render-ancestry", action="store_true",
                        help="use only for CPU fixture tests; execution does not admit this option")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.plan_only:
            print(json.dumps(plan_document(), indent=2, sort_keys=True))
            return 0
        if (args.recording_dir is None or args.index_sha256 is None
                or args.replay_sha256 is None or args.latent_sha256 is None
                or args.output_dir is None):
            raise DecodeError("--execute requires recording-dir, index/replay/latent SHA-256, and output-dir")
        if args.skip_live_render_ancestry:
            raise DecodeError("live render ancestry cannot be skipped for execution")
        admission = _authenticate_trace(
            args.recording_dir.expanduser(), index_sha256=args.index_sha256,
            replay_sha256=args.replay_sha256, latent_sha256=args.latent_sha256,
        )
        weights = _authenticate_weight_bundle()
        source_hashes = _verify_upstream_sources(UPSTREAM_ROOT)
        output = _fresh_output(args.output_dir.expanduser())
        result = run_decode(admission=admission, weights=weights,
                            source_hashes=source_hashes, output=output)
    except (DecodeError, block_reference.ReferenceError,
            generation_inputs.InputsError, checkpoint.InventoryError,
            RuntimeError, OSError, ValueError) as error:
        print(f"upstream trace decode failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "raw_sha256": result["outputs"]["raw"]["sha256"],
                      "display_rgb_sha256": result["outputs"]["display_rgb"]["sha256"],
                      "mp4_encoded": result["outputs"]["mp4"]["encoded"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
