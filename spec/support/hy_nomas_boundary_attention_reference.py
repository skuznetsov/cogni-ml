#!/usr/bin/env python3
"""Authenticated CPU/MPS oracle for one real Hy-Nomas segment boundary.

The oracle consumes a native capture of block 0's segment-0 cache writer and
segment-1 first read.  It authenticates continuation inputs, the saved clean
latent bank and receipt, and the pinned BF16 checkpoint files before running
the unchanged pinned ``MMDoubleStreamBlock.forward_with_cache`` in F32 on MPS.
Its attention adapter is bounded PyTorch MPS SDPA, not Magi/CUDA parity.
``--validate-only`` is CPU-only and never imports torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_continuation_inputs as continuation_inputs  # noqa: E402
import hy_nomas_real_attention_reference as block_reference  # noqa: E402
import hy_nomas_reference as pinned_reference  # noqa: E402


SCHEMA = "cogni_ml.hy_nomas_boundary_attention_reference/v1"
CAPTURE_SCHEMA = "cogni_ml.hy_nomas_boundary_attention_capture/v1"
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
CONDITIONING_SHA256 = "2807e11814f07975c807f367b0a4f15e3c6bdde1eb62ce9bda851814d09a6e47"
RECEIPT_SHA256 = "16cd52cf89e4a370f0b0112e14e8eefba08db76fcb2da6789e98f888ab6f24a4"
LATENT_BANK_SHA256 = "003b116c48e55809c092ce67f64242f80e6f24a26c89bc2d2fe38352e4da365b"
PREVIEW_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-cache-fix-20261030/preview-10s/native"
)
PREVIEW_RECEIPT = PREVIEW_ROOT / "receipt.json"
PREVIEW_LATENT_BANK = PREVIEW_ROOT / "latent-bank.f32le"
ATOL = 2e-5
RTOL = 2e-5
HASH_CHUNK = 8 * 1024 * 1024
QUERY_CHUNK = 64
GEOMETRY = {
    "batch": 1,
    "segments": 2,
    "video_tokens": 4320,
    "text_storage_tokens": 576,
    "valid_text_tokens": 557,
    "joint_storage_rows": 4896,
    "active_rows": 4877,
    "heads": 16,
    "head_dim": 128,
    "dim": 2048,
    "history_rows": 4877,
    "current_rows": 4320,
    "query_rows": 4877,
    "key_rows": 9197,
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CAPTURE_TOP_KEYS = {
    "schema", "upstream_revision", "conditioning_manifest", "conditioning_sha256",
    "preview_receipt", "preview_receipt_sha256", "preview_latent_bank",
    "preview_latent_bank_sha256", "weight_index", "weight_index_sha256",
    "block_weights", "block_weights_sha256", "global_weights", "global_weights_sha256",
    "geometry", "timesteps", "cache_noise", "history", "aliases", "precision", "scope",
    "named_bytes", "encode_allocations", "encode_readbacks", "preterminal_waits",
    "terminal_waits", "payload", "payload_bytes", "payload_sha256", "tensors",
}
_SHAPES: dict[str, tuple[int, ...]] = {
    "input.latent.writer_clean": (1, 32, 30, 9, 16),
    "input.latent.writer_noisy": (1, 32, 30, 9, 16),
    "input.latent.reader_initial": (1, 32, 30, 9, 16),
    "input.latent.reader_model": (1, 32, 30, 9, 16),
    "input.rope_cos.writer": (4320, 128),
    "input.rope_sin.writer": (4320, 128),
    "input.rope_cos.reader": (4320, 128),
    "input.rope_sin.reader": (4320, 128),
    "input.txt.writer": (1, 576, 2048),
    "input.txt.reader": (1, 576, 2048),
    "input.vec.writer": (1, 2048),
    "input.vec.reader": (1, 2048),
    "input.vec_txt.writer": (1, 2048),
    "input.vec_txt.reader": (1, 2048),
    "input.img.writer": (1, 4320, 2048),
    "input.img.reader": (1, 4320, 2048),
    **{
        f"{phase}.{axis}": (1, 4896, 16, 128)
        for phase in ("writer", "reader") for axis in ("query", "key", "value")
    },
    "history.key": (1, 4877, 16, 128),
    "history.value": (1, 4877, 16, 128),
    "reader.history_attention": (1, 4877, 16, 128),
}


class BoundaryReferenceError(RuntimeError):
    """Fail-closed source, capture, provenance, or execution error."""


def _json_no_duplicates(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BoundaryReferenceError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except BoundaryReferenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise BoundaryReferenceError(f"invalid {label} JSON") from None
    if type(value) is not dict:
        raise BoundaryReferenceError(f"{label} root must be a JSON object")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise BoundaryReferenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _same_json_types(expected: Any, actual: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""
    if isinstance(expected, dict):
        return (type(actual) is dict and set(actual) == set(expected)
                and all(_same_json_types(value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (type(actual) is list and len(actual) == len(expected)
                and all(_same_json_types(left, right) for left, right in zip(expected, actual)))
    return type(actual) is type(expected) and actual == expected


def _regular_file(path_arg: str | Path, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise BoundaryReferenceError(f"{label} does not exist: {path}") from None
    if path.is_symlink() or not resolved.is_file():
        raise BoundaryReferenceError(f"{label} must be a canonical regular file: {path}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_range(path: Path, offset: int, nbytes: int) -> str:
    if type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes <= 0:
        raise BoundaryReferenceError("invalid tensor byte range")
    if offset + nbytes > path.stat().st_size:
        raise BoundaryReferenceError("tensor range is outside its payload")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = stream.read(min(remaining, HASH_CHUNK))
            if not chunk:
                raise BoundaryReferenceError("tensor payload was truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _ranges_equal(path: Path, left: int, right: int, nbytes: int) -> bool:
    if min(left, right) < 0 or nbytes <= 0 or max(left, right) + nbytes > path.stat().st_size:
        raise BoundaryReferenceError("alias byte range is outside its payload")
    with path.open("rb") as stream:
        remaining = nbytes
        while remaining:
            count = min(remaining, HASH_CHUNK)
            stream.seek(left + (nbytes - remaining))
            a = stream.read(count)
            stream.seek(right + (nbytes - remaining))
            b = stream.read(count)
            if len(a) != count or len(b) != count:
                raise BoundaryReferenceError("alias payload was truncated")
            if a != b:
                return False
            remaining -= count
    return True


def _bf16_expand_words(words: Any) -> Any:
    """Expand BF16 bit patterns in uint16 storage to little-endian F32."""
    import numpy as np

    array = np.asarray(words, dtype="<u2")
    bits = array.astype("<u4") << np.uint32(16)
    if np.any((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)):
        raise BoundaryReferenceError("BF16 source contains nonfinite values")
    return bits.view("<f4")


def _bf16_rne_roundtrip_f32(values: Any) -> Any:
    """Apply exact finite F32 -> BF16 RNE -> expanded F32 using integer bits."""
    import numpy as np

    array = np.asarray(values, dtype="<f4")
    bits = array.view("<u4")
    if np.any((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)):
        raise BoundaryReferenceError("F32 source contains nonfinite values")
    wide = bits.astype("<u8")
    retained_lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (wide + np.uint64(0x7FFF) + retained_lsb.astype("<u8")) >> np.uint64(16)
    if np.any((rounded & np.uint64(0x7F80)) == np.uint64(0x7F80)):
        raise BoundaryReferenceError("F32 source overflows to nonfinite BF16")
    return (rounded.astype("<u4") << np.uint32(16)).view("<f4")


def _validate_capture(capture: dict[str, Any], capture_path: Path,
                      supplied_sha256: str) -> tuple[Path, dict[str, Any]]:
    if _sha256_file(capture_path) != _digest(supplied_sha256, "capture SHA-256"):
        raise BoundaryReferenceError("capture manifest digest differs from supplied digest")
    if set(capture) != _CAPTURE_TOP_KEYS or capture.get("schema") != CAPTURE_SCHEMA:
        raise BoundaryReferenceError("boundary capture schema or top-level key inventory mismatch")
    if capture.get("upstream_revision") != PINNED_REVISION:
        raise BoundaryReferenceError("boundary capture upstream revision mismatch")
    if not _same_json_types(GEOMETRY, capture.get("geometry")):
        raise BoundaryReferenceError("boundary capture geometry mismatch")
    if (capture.get("conditioning_sha256") != CONDITIONING_SHA256
            or capture.get("preview_receipt_sha256") != RECEIPT_SHA256
            or capture.get("preview_latent_bank_sha256") != LATENT_BANK_SHA256):
        raise BoundaryReferenceError("boundary capture is not bound to admitted real inputs")
    if type(capture.get("tensors")) is not dict or set(capture["tensors"]) != set(_SHAPES):
        raise BoundaryReferenceError("boundary capture tensor inventory mismatch")
    payload_name = capture.get("payload")
    if (not isinstance(payload_name, str) or Path(payload_name).name != payload_name
            or not payload_name.endswith(".tensors.bin")):
        raise BoundaryReferenceError("capture payload must be a sibling .tensors.bin basename")
    payload = _regular_file(capture_path.parent / payload_name, "boundary capture payload")
    if type(capture.get("payload_bytes")) is not int or capture["payload_bytes"] <= 0:
        raise BoundaryReferenceError("capture payload byte extent is malformed")
    payload_sha = _digest(capture.get("payload_sha256"), "capture payload SHA-256")
    records = capture["tensors"]
    # Validate record objects before sorting so malformed JSON remains a
    # controlled admission failure rather than an incidental attribute error.
    for name, shape in _SHAPES.items():
        record = records[name]
        expected_nbytes = math.prod(shape) * 4
        if (type(record) is not dict or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}
                or record.get("dtype") != "F32" or not _same_json_types(list(shape), record.get("shape"))
                or type(record.get("offset")) is not int or type(record.get("nbytes")) is not int
                or record["nbytes"] != expected_nbytes):
            raise BoundaryReferenceError(f"capture tensor dtype/shape/extent mismatch: {name}")
        _digest(record.get("sha256"), f"capture tensor {name} SHA-256")
    ordered = sorted(records.items(), key=lambda pair: pair[1]["offset"])
    offset = 0
    for name, record in ordered:
        if record["offset"] != offset:
            raise BoundaryReferenceError(f"capture tensor ranges contain a gap or overlap before {name}")
        stop = offset + record["nbytes"]
        if _hash_range(payload, offset, record["nbytes"]) != record["sha256"]:
            raise BoundaryReferenceError(f"capture tensor bytes differ from record digest: {name}")
        offset = stop
    if offset != capture["payload_bytes"] or payload.stat().st_size != offset:
        raise BoundaryReferenceError("capture tensor records do not cover payload extent")
    if _sha256_file(payload) != payload_sha:
        raise BoundaryReferenceError("capture payload digest mismatch")
    return payload, records


def _authenticate_sources(capture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    """Authenticate exact continuation/Qwen/cache sources and the preview receipt."""
    if _regular_file(capture["preview_receipt"], "preview receipt") != _regular_file(
            PREVIEW_RECEIPT, "pinned preview receipt"):
        raise BoundaryReferenceError("preview receipt path is not the admitted saved run")
    if _regular_file(capture["preview_latent_bank"], "preview latent bank") != _regular_file(
            PREVIEW_LATENT_BANK, "pinned preview latent bank"):
        raise BoundaryReferenceError("preview latent-bank path is not the admitted saved run")
    receipt_path = _regular_file(capture["preview_receipt"], "preview receipt")
    receipt_sha = _digest(capture["preview_receipt_sha256"], "preview receipt SHA-256")
    if receipt_sha != RECEIPT_SHA256 or _sha256_file(receipt_path) != receipt_sha:
        raise BoundaryReferenceError("preview receipt does not match the admitted digest")
    receipt = _json_no_duplicates(receipt_path.read_bytes(), "preview receipt")
    if (receipt.get("schema") != "cogni_ml.hy_nomas_native_continuation/v1"
            or receipt.get("upstream_revision") != PINNED_REVISION
            or receipt.get("inputs_sha256") != CONDITIONING_SHA256
            or _regular_file(receipt.get("inputs", ""), "receipt continuation input")
            != _regular_file(capture["conditioning_manifest"], "continuation manifest")
            or _regular_file(receipt.get("diagnostics", {}).get("latent_bank", ""), "receipt latent bank")
            != _regular_file(capture["preview_latent_bank"], "preview latent bank")
            or receipt.get("diagnostics", {}).get("latent_bank_sha256") != LATENT_BANK_SHA256
            or not _same_json_types([1, 32, 180, 9, 16], receipt.get("latent_bank_shape"))
            or receipt.get("clean_latent_frames") != 60
            or receipt.get("segments") != 2 or receipt.get("frames") != 240
            or receipt.get("fps") != 24 or receipt.get("width") != 512
            or receipt.get("height") != 288 or receipt.get("steps_per_segment") != 4
            or receipt.get("shift") != 5 or receipt.get("cfg") != 1
            or receipt.get("text_cache_policy") != "once_per_scene"
            or receipt.get("qwen_valid_tokens") != 557
            or receipt.get("continuation_experimental") is not True
            or receipt.get("full_upstream_video_parity") is not False
            or receipt.get("history") != "all prior chunk video rows plus the first chunk's scene text, layer-specific, no truncation"
            or not _same_json_types({
                "sigma": 0.206897, "timestep": 206.897, "separate_noisy_forward": True,
            }, receipt.get("cache_writer"))):
        raise BoundaryReferenceError("preview receipt does not bind the expected cache/history run")
    latent_path = _regular_file(capture["preview_latent_bank"], "preview latent bank")
    if (capture.get("preview_latent_bank_sha256") != LATENT_BANK_SHA256
            or latent_path.stat().st_size != math.prod((1, 32, 180, 9, 16)) * 4
            or _sha256_file(latent_path) != LATENT_BANK_SHA256):
        raise BoundaryReferenceError("preview latent bank digest or geometry mismatch")
    if (not _same_json_types({"writer_cache": 206.897, "reader_initial": 1000}, capture.get("timesteps"))
            or not _same_json_types({
                "sigma": 0.206897,
                "timestep": 206.897,
                "writer_latent_output": "BF16-RNE expanded to F32 after separate F32 mul/add",
                "eps_source": "conditioning manifest cache.eps_bf16 segment0",
            }, capture.get("cache_noise"))
            or not _same_json_types({
                "writer_rows": 4877,
                "reader_history_rows": 4877,
                "reader_current_rows": 4320,
                "reader_key_rows": 9197,
                "writer_kv_source": "block0 QKV first 4877 rows",
                "append": "HyNomasContinuationHistoryKernels::AppendPlan",
                "format": "F32 BQHD row-major",
                "reader_attention": "direct HyNomasHistoryAttentionMPSPlan output; separate from block0 compact current-only attention",
            }, capture.get("history"))
            or not _same_json_types({
                "native_vec_txt_alias_of_vec": True,
                "capture_vec_txt_is_owned_copy": True,
            }, capture.get("aliases"))
            or capture.get("precision") != (
                "BF16 checkpoint weights; F32 native activation/MPS arithmetic; cache-noise BF16 RNE; "
                "imported initial reader latent Euler BF16 RNE; F32 persistent history K/V"
            )
            or capture.get("scope") != (
                "authentic writer0 and reader1 patch/refiner/block0 inputs/QKV; direct history attention only; "
                "block0 default attention remains native current-only compact attention and is unused; no attached "
                "history provider, history-aware block output, layers>0, full DiT, or media generation"
            )
            or not _same_json_types(
                [0, 0, 0, 1],
                [capture.get("encode_allocations"), capture.get("encode_readbacks"),
                 capture.get("preterminal_waits"), capture.get("terminal_waits")],
            )
            or type(capture.get("named_bytes")) is not int or capture["named_bytes"] < 0):
        raise BoundaryReferenceError("boundary capture execution contract or diagnostics mismatch")

    source_capture = {
        **capture,
        "cache_index": capture["weight_index"],
        "cache_index_sha256": capture["weight_index_sha256"],
    }
    source_info, _prelude = block_reference._authenticate_sources(source_capture)
    if capture["conditioning_sha256"] != CONDITIONING_SHA256:
        raise BoundaryReferenceError("continuation manifest digest is not the admitted digest")
    conditioning_path = _regular_file(capture["conditioning_manifest"], "continuation manifest")
    condition = block_reference._json_no_duplicates(conditioning_path.read_bytes(), "continuation manifest")
    condition_payload = _regular_file(
        conditioning_path.parent / condition["payload"], "continuation input payload"
    )
    if condition["upstream_revision"] != PINNED_REVISION:
        raise BoundaryReferenceError("continuation manifest is not pinned to the expected source")
    return source_info, condition, condition_payload, {"receipt": receipt, "preview_latent": latent_path}


def _validate_aliases(capture: dict[str, Any], payload: Path, condition: dict[str, Any],
                      condition_payload: Path, preview_latent_path: Path) -> dict[str, str]:
    """Bind capture latent/RoPE/cache aliases to authenticated saved inputs."""
    import numpy as np

    tensors = capture["tensors"]
    preview = np.memmap(preview_latent_path, mode="r", dtype="<f4", shape=(1, 32, 180, 9, 16))
    clean_prefix = np.ascontiguousarray(preview[:, :, :30, :, :], dtype="<f4")
    clean_hash = hashlib.sha256(clean_prefix.tobytes()).hexdigest()
    if clean_hash != tensors["input.latent.writer_clean"]["sha256"]:
        raise BoundaryReferenceError("writer clean latent is not the saved preview latent-bank prefix")
    eps_record = condition["tensors"]["cache.eps_bf16"]
    eps = np.memmap(
        condition_payload, mode="r", dtype="<u2", offset=eps_record["offset"],
        shape=tuple(eps_record["shape"]),
    )
    latent_bf16 = _bf16_rne_roundtrip_f32(clean_prefix)
    epsilon_f32 = _bf16_expand_words(np.asarray(eps[0], dtype="<u2"))
    sigma = np.float32(0.206897)
    clean_term = np.multiply(np.float32(1.0) - sigma, latent_bf16, dtype=np.float32)
    noise_term = np.multiply(sigma, epsilon_f32, dtype=np.float32)
    noisy = _bf16_rne_roundtrip_f32(np.add(clean_term, noise_term, dtype=np.float32))
    noisy_hash = hashlib.sha256(np.ascontiguousarray(noisy, dtype="<f4").tobytes()).hexdigest()
    if noisy_hash != tensors["input.latent.writer_noisy"]["sha256"]:
        raise BoundaryReferenceError("writer noisy latent does not match authenticated BF16 cache-noise input")

    bank_record = condition["tensors"]["input.latent_bank"]
    latent_bank = np.memmap(
        condition_payload, mode="r", dtype="<f4", offset=bank_record["offset"],
        shape=tuple(bank_record["shape"]),
    )
    reader_latent = np.ascontiguousarray(latent_bank[:, :, 30:60, :, :], dtype="<f4")
    reader_hash = hashlib.sha256(reader_latent.tobytes()).hexdigest()
    if reader_hash != tensors["input.latent.reader_initial"]["sha256"]:
        raise BoundaryReferenceError("reader initial latent is not continuation input frames 30:60")
    reader_model = _bf16_rne_roundtrip_f32(reader_latent)
    reader_model_hash = hashlib.sha256(
        np.ascontiguousarray(reader_model, dtype="<f4").tobytes()
    ).hexdigest()
    if reader_model_hash != tensors["input.latent.reader_model"]["sha256"]:
        raise BoundaryReferenceError("reader model latent is not the initial latent's BF16 RNE roundtrip")

    rope_record_cos = condition["tensors"]["input.rope_cos"]
    rope_record_sin = condition["tensors"]["input.rope_sin"]
    for phase, segment in (("writer", 0), ("reader", 1)):
        for axis, record in (("cos", rope_record_cos), ("sin", rope_record_sin)):
            rope = np.memmap(
                condition_payload, mode="r", dtype="<f4", offset=record["offset"],
                shape=tuple(record["shape"]),
            )
            expected = np.ascontiguousarray(rope[segment], dtype="<f4").tobytes()
            if hashlib.sha256(expected).hexdigest() != tensors[f"input.rope_{axis}.{phase}"]["sha256"]:
                raise BoundaryReferenceError(f"{phase} RoPE {axis} differs from authenticated global-offset bank")

    # The text projection is phase-specific: the refiner timestep differs
    # between writer0 (206.897) and reader1 (1000). Keep these inputs distinct.
    for phase in ("writer", "reader"):
        for left, right in ((f"input.vec.{phase}", f"input.vec_txt.{phase}"),):
            rec_a, rec_b = tensors[left], tensors[right]
            if not _ranges_equal(payload, rec_a["offset"], rec_b["offset"], rec_a["nbytes"]):
                raise BoundaryReferenceError(f"authenticated scene conditioning alias differs: {left}/{right}")
    for axis in ("key", "value"):
        record = tensors[f"writer.{axis}"]
        history = tensors[f"history.{axis}"]
        prefix_bytes = 4877 * 16 * 128 * 4
        if not _ranges_equal(payload, record["offset"], history["offset"], prefix_bytes):
            raise BoundaryReferenceError(f"history.{axis} is not writer [V0,T0-valid557] prefix")
    return {
        "writer_clean_latent_sha256": clean_hash,
        "writer_noisy_latent_sha256": noisy_hash,
        "reader_initial_latent_sha256": reader_hash,
        "reader_model_latent_sha256": reader_model_hash,
    }


class _Metric:
    """Bounded MPS-to-CPU comparison summary for one captured tensor."""

    def __init__(self, torch: Any, device: Any, name: str, expected_count: int):
        self.torch, self.device, self.name, self.expected_count = torch, device, name, expected_count
        self.seen = 0
        self.max_abs = torch.zeros((), dtype=torch.float32, device=device)
        self.sum_abs = torch.zeros((), dtype=torch.float32, device=device)
        self.sum_sq = torch.zeros((), dtype=torch.float32, device=device)
        self.bad_count = torch.zeros((), dtype=torch.int64, device=device)
        self.first_bad = torch.full((), -1, dtype=torch.int64, device=device)
        self.all_finite = torch.ones((), dtype=torch.bool, device=device)

    def add(self, actual: Any, flat_offset: int, expected_cpu: Any) -> None:
        torch = self.torch
        actual = actual.detach().reshape(-1).to(dtype=torch.float32)
        expected = torch.from_numpy(expected_cpu).to(device=self.device, dtype=torch.float32)
        if actual.device.type != "mps" or actual.numel() != expected.numel():
            raise BoundaryReferenceError(f"{self.name} comparison range mismatch")
        diff = (actual - expected).abs()
        finite = torch.isfinite(actual) & torch.isfinite(expected)
        bad = (~finite) | (diff > ATOL + RTOL * expected.abs())
        self.max_abs = torch.maximum(self.max_abs, torch.nan_to_num(diff, nan=0.0, posinf=float("inf")).max())
        self.sum_abs = self.sum_abs + torch.nan_to_num(diff, nan=0.0, posinf=float("inf")).sum()
        self.sum_sq = self.sum_sq + torch.nan_to_num(diff.square(), nan=0.0, posinf=float("inf")).sum()
        self.bad_count = self.bad_count + bad.sum(dtype=torch.int64)
        candidate = torch.argmax(bad.to(torch.int64)) + flat_offset
        has_bad = bad.any()
        self.first_bad = torch.where(
            (self.first_bad < 0) & has_bad, candidate.to(torch.int64), self.first_bad,
        )
        self.all_finite = self.all_finite & finite.all()
        self.seen += actual.numel()

    def finish(self) -> dict[str, Any]:
        if self.seen != self.expected_count:
            raise BoundaryReferenceError(f"{self.name} compared {self.seen}/{self.expected_count} elements")
        # Keep exact integer summaries on their native dtype until scalar transfer.
        max_abs = float(self.max_abs.detach().cpu().item())
        mean_abs = float((self.sum_abs / max(1, self.seen)).detach().cpu().item())
        rms_abs = float((self.sum_sq / max(1, self.seen)).sqrt().detach().cpu().item())
        bad_count = int(self.bad_count.detach().cpu().item())
        first_bad = int(self.first_bad.detach().cpu().item())
        finite = bool(self.all_finite.detach().cpu().item())
        return {
            "checkpoint": self.name,
            "elements": self.seen,
            "max_abs": max_abs if math.isfinite(max_abs) else None,
            "mean_abs": mean_abs if math.isfinite(mean_abs) else None,
            "rms_abs": rms_abs if math.isfinite(rms_abs) else None,
            "outside_atol_rtol": bad_count,
            "first_failure_flat_index": first_bad if first_bad >= 0 else None,
            "all_finite": finite,
            "allclose_atol_2e-5_rtol_2e-5": bad_count == 0 and finite,
        }


def _load_tensor(torch: Any, np: Any, payload_map: Any, record: dict[str, Any],
                 shape: tuple[int, ...], device: Any, dtype: Any = None) -> Any:
    count = math.prod(shape)
    start = record["offset"] // 4
    cpu = torch.from_numpy(payload_map[start:start + count].copy()).reshape(shape)
    return cpu.to(device=device, dtype=dtype or torch.float32)


def _compare_tensor(torch: Any, np: Any, payload_map: Any, payload: Path,
                    record: dict[str, Any], name: str, actual: Any) -> dict[str, Any]:
    metric = _Metric(torch, actual.device, name, record["nbytes"] // 4)
    actual = actual.detach().reshape(-1)
    offset = 0
    while offset < actual.numel():
        stop = min(offset + 1_048_576, actual.numel())
        record_start = record["offset"] // 4 + offset
        expected = payload_map[record_start:record_start + stop - offset].copy()
        metric.add(actual[offset:stop], offset, expected)
        offset = stop
    return metric.finish()


def _run_boundary_control(torch: Any, np: Any, capture: dict[str, Any], records: dict[str, Any],
                          payload: Path, payload_map: Any, upstream: Path) -> dict[str, Any]:
    device = torch.device("mps")
    dtype = torch.float32
    heads, head_dim, dim = 16, 128, 2048
    helpers = pinned_reference._install_upstream_modules(upstream)
    posemb = helpers[-1]
    phase_state: dict[str, Any] = {"phase": None, "writer_key": None, "writer_value": None}
    replay_metrics: dict[str, list[dict[str, Any]]] = {"native_qkv": []}

    def _attend(query: Any, key: Any, value: Any, *, capture_record: dict[str, Any] | None,
                name: str) -> Any:
        qh = query.permute(0, 2, 1, 3).contiguous()
        kh = key.permute(0, 2, 1, 3).contiguous()
        vh = value.permute(0, 2, 1, 3).contiguous()
        chunks = []
        for start in range(0, qh.shape[2], QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, qh.shape[2])
            chunks.append(torch.nn.functional.scaled_dot_product_attention(
                qh[:, :, start:stop], kh, vh, attn_mask=None, dropout_p=0.0,
                is_causal=False,
            ))
        output = torch.cat(chunks, dim=2).permute(0, 2, 1, 3).contiguous()
        if capture_record is not None:
            summary = _compare_tensor(torch, np, payload_map, payload, capture_record, name, output)
            replay_metrics.setdefault(name, []).append(summary)
        return output

    def attention_adapter(q: tuple[Any, Any], k: tuple[Any, Any], v: tuple[Any, Any],
                          kv_cache: Any, cache_valid_len: int = 0,
                          attn_mode: str | None = None, text_mask: Any = None,
                          update_cache: bool = False, cache_txt: bool = True,
                          cache_mask: Any = None, intra_seg_txt_offsets: Any = None,
                          intra_seg_vid_windows: Any = None, tokens_per_frame: int = 0,
                          seg_ar_attention_backend: str = "magi", sparse_context: Any = None) -> tuple[Any, Any]:
        phase = phase_state["phase"]
        if phase not in ("writer", "reader"):
            raise BoundaryReferenceError("pinned block called attention outside writer/read boundary")
        if (attn_mode != "magi_mp_causal" or seg_ar_attention_backend != "magi"
                or text_mask is not None or cache_mask is not None or sparse_context is not None):
            raise BoundaryReferenceError("pinned block changed attention mode or boundary mask")
        query, key, value = (torch.cat(parts, dim=1) for parts in (q, k, v))
        expected_shape = (1, 4896, heads, head_dim)
        if any(tuple(tensor.shape) != expected_shape or tensor.dtype != dtype
               or tensor.device.type != "mps" for tensor in (query, key, value)):
            raise BoundaryReferenceError("upstream Q/K/V does not match real boundary capture geometry")
        valid = 4320 + 557
        for axis, actual in (("query", query), ("key", key), ("value", value)):
            rec = records[f"{phase}.{axis}"]
            replay_metrics.setdefault(f"{phase}.{axis}", []).append(_compare_tensor(
                torch, np, payload_map, payload, rec, f"upstream.{phase}.{axis}", actual,
            ))

        if phase == "writer":
            if not update_cache or not cache_txt or cache_valid_len != 0:
                raise BoundaryReferenceError("writer call did not request empty-cache [V0,T0] retention")
            phase_state["writer_key"] = key[:, :valid].detach()
            phase_state["writer_value"] = value[:, :valid].detach()
            for axis, value_up in (("key", phase_state["writer_key"]),
                                   ("value", phase_state["writer_value"])):
                rec = records[f"history.{axis}"]
                replay_metrics.setdefault(f"history.{axis}", []).append(_compare_tensor(
                    torch, np, payload_map, payload, rec,
                    f"upstream.history.{axis}", value_up,
                ))
            _attend(query[:, :valid], key[:, :valid], value[:, :valid],
                    capture_record=None, name="upstream.writer.attention")
        else:
            if update_cache or cache_txt or cache_valid_len != valid:
                raise BoundaryReferenceError("reader call does not match once-per-scene history contract")
            hist_k, hist_v = phase_state["writer_key"], phase_state["writer_value"]
            if hist_k is None or hist_v is None:
                raise BoundaryReferenceError("reader ran before authenticated upstream writer outputs")
            current_k, current_v = key[:, :4320], value[:, :4320]
            _attend(
                query[:, :valid], torch.cat((hist_k, current_k), dim=1),
                torch.cat((hist_v, current_v), dim=1),
                capture_record=records["reader.history_attention"],
                name="upstream.reader.history_attention",
            )
        raise _BoundaryAttentionCaptured()

    block_type = pinned_reference._extract_pinned_block(
        upstream, torch, attention_adapter, posemb.apply_rotary_emb, helpers,
    )
    # forward_with_cache is source-authentic; only its attention dispatch is
    # replaced by the bounded adapter above.
    block_type.forward_with_cache.__globals__["causal_attention_with_cache_mp"] = attention_adapter
    block = block_type(
        hidden_size=dim, heads_num=heads, mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh", attn_mode="magi_mp_causal", qk_norm=True,
        qk_norm_type="rms", qkv_bias=True, dtype=dtype, device=torch.device("meta"),
    ).eval()
    state = block_reference._load_block_weights(torch, Path(capture["block_weights"]), dtype)
    block.load_state_dict(state, strict=True, assign=True)
    block.to(device=device, dtype=dtype).eval()
    del state
    kv_cache = {"k": torch.empty((1, 0, heads, head_dim), dtype=dtype, device=device),
                "v": torch.empty((1, 0, heads, head_dim), dtype=dtype, device=device)}
    results = []
    with torch.inference_mode():
        for phase in ("writer", "reader"):
            phase_state["phase"] = phase
            img = _load_tensor(torch, np, payload_map, records[f"input.img.{phase}"], (1, 4320, dim), device)
            txt = _load_tensor(torch, np, payload_map, records[f"input.txt.{phase}"], (1, 576, dim), device)
            vec = _load_tensor(torch, np, payload_map, records[f"input.vec.{phase}"], (1, dim), device)
            vec_txt = _load_tensor(torch, np, payload_map, records[f"input.vec_txt.{phase}"], (1, dim), device)
            cos = _load_tensor(torch, np, payload_map, records[f"input.rope_cos.{phase}"], (4320, 128), device)
            sin = _load_tensor(torch, np, payload_map, records[f"input.rope_sin.{phase}"], (4320, 128), device)
            try:
                block.forward_with_cache(
                    img=img, txt=txt, vec=vec, vec_txt=vec_txt,
                    freqs_cis=(cos, sin), text_mask=None, kv_cache=kv_cache,
                    cache_valid_len=0 if phase == "writer" else 4877,
                    update_cache=phase == "writer", attn_mode="magi_mp_causal",
                    cache_txt=phase == "writer",
                )
            except _BoundaryAttentionCaptured:
                pass
            else:
                raise BoundaryReferenceError("pinned forward did not reach bounded attention stop")
            results.append({"phase": phase, "attention_calls": 1})

    # The common-native-QKV replay is the adapter discriminator: it preserves
    # captured Q/K/V and retained history and changes only SDPA execution.
    q = _load_tensor(torch, np, payload_map, records["reader.query"], (1, 4896, heads, head_dim), device)
    hist_k = _load_tensor(torch, np, payload_map, records["history.key"], (1, 4877, heads, head_dim), device)
    hist_v = _load_tensor(torch, np, payload_map, records["history.value"], (1, 4877, heads, head_dim), device)
    current_k = _load_tensor(torch, np, payload_map, records["reader.key"], (1, 4896, heads, head_dim), device)[:, :4320]
    current_v = _load_tensor(torch, np, payload_map, records["reader.value"], (1, 4896, heads, head_dim), device)[:, :4320]
    _attend(
        q[:, :4877], torch.cat((hist_k, current_k), dim=1),
        torch.cat((hist_v, current_v), dim=1),
        capture_record=records["reader.history_attention"],
        name="common_native_qkv.history_attention",
    )

    def first_failure(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        return next((item for item in items if not item["allclose_atol_2e-5_rtol_2e-5"]), None)

    outputs = {}
    for name, items in replay_metrics.items():
        if items:
            outputs[name] = items[-1]
    ordered_keys = (
        "writer.query", "writer.key", "writer.value", "history.key", "history.value",
        "reader.query", "reader.key", "reader.value", "upstream.reader.history_attention",
    )
    ordered = [outputs[key] for key in ordered_keys if key in outputs]
    return {
        "status": "completed",
        "device": "mps",
        "compute_and_parameter_dtype": "F32 with exact pinned BF16 checkpoint values expanded to F32",
        "attention": "chunked PyTorch MPS SDPA diagnostic adapter; not Magi/CUDA parity",
        "forward_calls": results,
        "history_contract": {
            "writer_tokens": 4877,
            "reader_query_tokens": 4877,
            "reader_current_video_key_tokens": 4320,
            "reader_total_key_tokens": 9197,
        },
        "first_upstream_checkpoint_outside_tolerance": first_failure(ordered),
        "upstream_checkpoints": ordered,
        "common_native_qkv_replay": replay_metrics["common_native_qkv.history_attention"][-1],
        "interpretation_limit": (
            "A mismatch is a first-boundary diagnostic, not proof of perceived-quality cause. "
            "The common-QKV replay isolates this MPS SDPA adapter from block pre-attention QKV; "
            "neither result qualifies Magi/CUDA kernels or full generated video."
        ),
    }


class _BoundaryAttentionCaptured(Exception):
    pass


def _write_json_fresh(path_arg: str | Path, document: dict[str, Any]) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    parent = path.parent.resolve(strict=True)
    path = parent / path.name
    if path.exists() or path.is_symlink():
        raise BoundaryReferenceError(f"refusing to overwrite oracle report: {path}")
    encoded = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _verify_upstream_sources(upstream_arg: str | Path) -> tuple[Path, dict[str, str]]:
    upstream_path = Path(upstream_arg).expanduser()
    if not upstream_path.is_absolute():
        upstream_path = Path.cwd() / upstream_path
    upstream = upstream_path.resolve(strict=True)
    if upstream_path.is_symlink() or not upstream.is_dir():
        raise BoundaryReferenceError("upstream source root must be a canonical directory")
    return upstream, pinned_reference._verify_upstream(upstream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="authenticated boundary capture JSON")
    parser.add_argument("--capture-sha256", required=True, help="expected capture manifest SHA-256")
    parser.add_argument("--output", required=True, help="fresh report JSON path")
    parser.add_argument("--upstream", required=True, help="pinned upstream checkout root")
    parser.add_argument("--validate-only", action="store_true", help="CPU-only source/capture validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "pinned_upstream_revision": PINNED_REVISION,
        "comparison_tolerance": {"atol": ATOL, "rtol": RTOL},
        "capture_manifest": str(Path(args.capture).expanduser().resolve()),
        "capture_manifest_sha256": _digest(args.capture_sha256, "capture SHA-256"),
        "status": "rejected",
        "diagnostic_status": "not_run",
        "scope": "writer segment 0 and reader segment 1, block 0, first boundary only",
        "controls": None,
    }
    exit_code = 1
    try:
        capture_path = _regular_file(args.capture, "boundary capture manifest")
        capture = _json_no_duplicates(capture_path.read_bytes(), "boundary capture manifest")
        payload, records = _validate_capture(capture, capture_path, report["capture_manifest_sha256"])
        source_info, condition, condition_payload, source_paths = _authenticate_sources(capture)
        aliases = _validate_aliases(
            capture, payload, condition, condition_payload, source_paths["preview_latent"],
        )
        upstream, upstream_hashes = _verify_upstream_sources(args.upstream)
        report.update({
            "capture_payload_bytes": capture["payload_bytes"],
            "capture_payload_sha256": capture["payload_sha256"],
            "tensor_record_count": len(records),
            "authenticated_source_info": source_info,
            "authenticated_aliases": aliases,
            "upstream_source_sha256": upstream_hashes,
        })
        if args.validate_only:
            report["status"] = "validated_only"
            report["diagnostic_status"] = "not_run"
            exit_code = 0
        else:
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
                raise BoundaryReferenceError("MPS fallback must be disabled")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            try:
                import numpy as np
                import torch
            except ImportError as error:
                raise BoundaryReferenceError(f"MPS reference dependency unavailable: {error}") from error
            if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
                raise BoundaryReferenceError("boundary reference requires an available PyTorch MPS device")
            payload_map = np.memmap(
                payload, mode="r", dtype="<f4", shape=(capture["payload_bytes"] // 4,),
            )
            control = _run_boundary_control(
                torch, np, capture, records, payload, payload_map, upstream,
            )
            report["controls"] = control
            report["status"] = "completed"
            report["diagnostic_status"] = (
                "outside_tolerance"
                if control["first_upstream_checkpoint_outside_tolerance"] is not None
                or not control["common_native_qkv_replay"]["allclose_atol_2e-5_rtol_2e-5"]
                else "within_tolerance"
            )
            exit_code = 0
    except Exception as error:
        report["status"] = "rejected"
        report["error"] = str(error)
        exit_code = 1
    try:
        report_path = _write_json_fresh(args.output, report)
    except Exception as error:
        print(f"report publication failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"report": str(report_path), "status": report["status"],
                      "diagnostic_status": report["diagnostic_status"],
                      "error": report.get("error")}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
