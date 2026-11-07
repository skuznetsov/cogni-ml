#!/usr/bin/env python3
"""Run an authenticated T0-reference to T1-action upstream continuation control.

The control consumes a supplied clean T0 latent, writes its noised cache once
with T0 caption conditioning, then denoises the original T1 noise with a fresh
T1 action Qwen query while reading the T0 text/image history. It runs four
512x288 latent steps and exports one terminal two-segment F32 latent bank.
Use --plan-only for CPU-only provenance and schedule admission; --execute is
an explicit MPS execution request and expects an external memory guard.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import typing
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_bf16_dit_quality_control as single  # noqa: E402
import hy_nomas_continuation_inputs as continuation_inputs  # noqa: E402
import hy_nomas_generation_inputs as generation_inputs  # noqa: E402
import hy_nomas_q4_dit_reference as cache_reference  # noqa: E402
import hy_nomas_real_attention_reference as attention_reference  # noqa: E402
import hy_nomas_reference as block_reference  # noqa: E402
import hy_nomas_two_segment_quality_control as two_segment  # noqa: E402

SCHEMA = "cogni_ml.hy_nomas_reference_continuation_control/v1"
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
ACTION_VALID_TOKENS = 369
LATENTS_PER_SEGMENT = 30
SEGMENT_INDEX_T0 = 0
SEGMENT_INDEX_T1 = 1
STEPS = 4
FLOW_SHIFT = 5.0
WRITER_SIGMA = 0.206897
WRITER_TIMESTEP = 206.897
LATENT_BANK_SHAPE = (1, 32, 180, 9, 16)
LATENT_SEGMENT_SHAPE = (1, 32, 30, 9, 16)
QWEN_SHAPE = (1, 1280, 3584)
QWEN_ROWS = 1280
DIT_TEXT_ROWS = 576
VIDEO_TOKENS = 30 * 9 * 16
HEADS = 16
HEAD_DIM = 128
DIM = 2048
CACHE_CAPACITY = VIDEO_TOKENS + DIT_TEXT_ROWS
QUERY_CHUNK = attention_reference.QUERY_CHUNK
HASH_CHUNK = 8 * 1024 * 1024
MAX_PLANNED_CAP_GIB = 25.5
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def compact_t0_history_visibility_layout(keep_latents: int, t0_valid_tokens: int) -> dict[str, int]:
    """Describe a same-shape T1 mask over compact T0-video/text/T1-video keys."""
    if keep_latents not in (10, 30):
        raise ControlError("T0 history diagnostic keeps either 10 or all 30 latent frames")
    if not 0 < t0_valid_tokens <= DIT_TEXT_ROWS:
        raise ControlError("T0 history diagnostic requires admitted valid caption rows")
    tokens_per_latent = VIDEO_TOKENS // LATENTS_PER_SEGMENT
    hidden_video_rows = (LATENTS_PER_SEGMENT - keep_latents) * tokens_per_latent
    key_rows = 2 * VIDEO_TOKENS + t0_valid_tokens
    return {
        "key_rows": key_rows,
        "t0_video_rows": VIDEO_TOKENS,
        "hidden_t0_video_prefix_rows": hidden_video_rows,
        "visible_t0_video_rows": VIDEO_TOKENS - hidden_video_rows,
        "t0_text_start": VIDEO_TOKENS,
        "t0_text_end": VIDEO_TOKENS + t0_valid_tokens,
        "t1_video_start": VIDEO_TOKENS + t0_valid_tokens,
        "t1_video_end": key_rows,
    }

DEFAULT_T0_LATENT_BANK = Path(
    "/opt/mldata_ssd/cogni-ml/rolewright-reference-video-20261102.ux_lr9kv/"
    "results/latent-bank.f32le"
)
T0_LATENT_BANK_SHA256 = (
    "69c29e798062bb93518122279ceb38c5324f3662af56f882fb307f19db88697c"
)
DEFAULT_T1_QWEN_INPUTS = Path(
    "/opt/mldata_ssd/cogni-ml/rolewright-reference-video-20261102.ux_lr9kv/"
    "conditioning/inputs.json"
)
T1_QWEN_INPUTS_SHA256 = (
    "9d4b93fd6c1f99301266a46096377e1d3cc794d0834c6d9815721a7cf13dac39"
)


class ControlError(RuntimeError):
    """Fail-closed artifact, source, schedule, or MPS execution error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, size: int) -> str:
    if offset < 0 or size <= 0 or offset + size > path.stat().st_size:
        raise ControlError("authenticated tensor range is outside its payload extent")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            block = stream.read(min(remaining, HASH_CHUNK))
            if not block:
                raise ControlError("authenticated tensor range was truncated while hashing")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _regular_file(path_arg: Path | str, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ControlError(f"{label} must be an existing absolute file") from None
    # /opt/mldata_ssd is an installed alias on this host; authenticate the
    # canonical target while still rejecting a symlink at the final file name.
    approved_alias = Path("/opt/mldata_ssd")
    alias_path = path == approved_alias or approved_alias in path.parents
    if path.is_symlink() or not resolved.is_file() or (path != resolved and not alias_path):
        raise ControlError(f"{label} must be a canonical regular file")
    return resolved


def _regular_dir(path_arg: Path | str, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ControlError(f"{label} must be an existing absolute directory") from None
    approved_alias = Path("/opt/mldata_ssd")
    alias_path = path == approved_alias or approved_alias in path.parents
    if path.is_symlink() or not resolved.is_dir() or (path != resolved and not alias_path):
        raise ControlError(f"{label} must be a canonical directory")
    return resolved


def _authenticate_file(
    path_arg: Path | str,
    expected_sha256: str,
    label: str,
    *,
    expected_bytes: int | None = None,
) -> str:
    if not isinstance(expected_sha256, str) or _HEX.fullmatch(expected_sha256) is None:
        raise ControlError(f"{label} expected digest must be lowercase SHA-256")
    path = _regular_file(path_arg, label)
    before = _identity(path)
    if expected_bytes is not None and before[2] != expected_bytes:
        raise ControlError(f"{label} byte extent mismatch")
    actual = _sha256(path)
    if _identity(path) != before:
        raise ControlError(f"{label} changed while being authenticated")
    if actual != expected_sha256:
        raise ControlError(f"{label} digest mismatch")
    return actual


def _verify_source_files(root: Path, expected_sources: dict[str, str]) -> dict[str, str]:
    """Hash pinned primary sources and reject path or byte drift."""
    actual: dict[str, str] = {}
    for relative, expected in sorted(expected_sources.items()):
        if not isinstance(expected, str) or _HEX.fullmatch(expected) is None:
            raise ControlError(f"pinned source digest is malformed: {relative}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ControlError(f"pinned source is missing or non-canonical: {relative}")
        digest = _sha256(path)
        if digest != expected:
            raise ControlError(f"pinned source digest mismatch: {relative}")
        actual[relative] = digest
    return actual


def _validate_mask(mask: Any, label: str, expected_valid_tokens: int | None = None) -> int:
    if (type(mask) is not list or len(mask) != QWEN_ROWS
            or any(type(value) is not int or value not in (0, 1) for value in mask)):
        raise ControlError(f"{label} mask must contain 1280 integer values")
    first_padding = next((index for index, value in enumerate(mask) if value == 0), len(mask))
    valid = sum(mask)
    if mask[0] != 1 or any(mask[first_padding:]):
        raise ControlError(f"{label} mask must be one contiguous valid prefix")
    if expected_valid_tokens is not None and valid != expected_valid_tokens:
        raise ControlError(f"{label} valid-token count differs from its admitted identity")
    return valid


def _validate_tensor_record(record: Any, dtype: str, shape: list[int]) -> int:
    if (type(record) is not dict or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}
            or record.get("dtype") != dtype or record.get("shape") != shape):
        raise ControlError("authenticated tensor dtype or geometry mismatch")
    width = {"F32": 4, "BF16": 2}.get(dtype)
    if width is None:
        raise ControlError("authenticated tensor dtype is unsupported")
    expected = math.prod(shape) * width
    if type(record.get("nbytes")) is not int or record["nbytes"] != expected:
        raise ControlError("authenticated tensor byte extent mismatch")
    if type(record.get("offset")) is not int or record["offset"] < 0:
        raise ControlError("authenticated tensor offset is malformed")
    if not isinstance(record.get("sha256"), str) or _HEX.fullmatch(record["sha256"]) is None:
        raise ControlError("authenticated tensor digest is malformed")
    return expected


def _validate_t0_selection(
    bank_shape: list[int], frame_start: int, frame_count: int
) -> list[int]:
    if bank_shape != list(LATENT_BANK_SHAPE):
        raise ControlError("clean T0 latent bank geometry differs from the admitted source")
    if (type(frame_start) is not int or type(frame_count) is not int
            or frame_start != 0 or frame_count != LATENTS_PER_SEGMENT):
        raise ControlError("clean T0 frame selection must be exactly [0, 30)")
    return [frame_start, frame_start + frame_count]


def _forward_plan(caption_valid_tokens: int, action_valid_tokens: int) -> list[dict[str, Any]]:
    plan = [{
        "kind": "writer",
        "segment": 0,
        "step": None,
        "condition": "t0_caption",
        "valid_text_tokens": caption_valid_tokens,
        "cache_valid_len": 0,
        "update_cache": True,
        "cache_txt": True,
        "rope_bank": 0,
        "timestep": WRITER_TIMESTEP,
    }]
    plan.extend({
        "kind": "denoise",
        "segment": 1,
        "step": step,
        "condition": "t1_action",
        "valid_text_tokens": action_valid_tokens,
        "cache_valid_len": CACHE_CAPACITY,
        "update_cache": False,
        "cache_txt": True,
        "rope_bank": 1,
        "timestep": "euler",
    } for step in range(STEPS))
    return plan


def _validate_forward_plan(
    plan: list[dict[str, Any]], caption_valid_tokens: int, action_valid_tokens: int
) -> None:
    expected = _forward_plan(caption_valid_tokens, action_valid_tokens)
    if plan != expected or len(plan) != STEPS + 1:
        raise ControlError("reference-continuation schedule differs from the admitted cache contract")


def _block_cache_contract(call: dict[str, Any]) -> dict[str, Any]:
    """Return the exact cache flags consumed by the pinned block wrapper."""
    return {
        "cache_valid_len": call["cache_valid_len"],
        "update_cache": call["update_cache"],
        "cache_txt": call["cache_txt"],
    }


def _manifest_document(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    before = _identity(path)
    if before[2] > 2 * 1024 * 1024:
        raise ControlError(f"{label} exceeds the admitted size bound")
    raw = path.read_bytes()
    if _identity(path) != before:
        raise ControlError(f"{label} changed while being read")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ControlError(f"{label} digest mismatch")
    try:
        document = attention_reference._json_no_duplicates(raw, label)
    except Exception as error:
        raise ControlError(f"{label} rejected: {error}") from None
    return document


def _authenticate_conditioning(args: argparse.Namespace) -> dict[str, Any]:
    condition_path = _regular_file(args.conditioning_manifest, "target continuation manifest")
    condition = _manifest_document(
        condition_path, args.conditioning_manifest_sha256, "target continuation manifest"
    )
    try:
        continuation_inputs.validate_document(condition)
    except Exception as error:
        raise ControlError(f"target continuation manifest rejected: {error}") from None
    if condition.get("upstream_revision") != PINNED_REVISION:
        raise ControlError("target continuation upstream revision mismatch")

    payload_path = _regular_file(
        condition_path.parent / condition["payload"], "target continuation payload"
    )
    if payload_path.parent != condition_path.parent:
        raise ControlError("target continuation payload must be a sibling of its manifest")
    payload_record = condition.get("payload_sha256")
    if payload_record != args.conditioning_payload_sha256:
        raise ControlError("target continuation payload SHA differs from the explicit digest")
    payload_identity = _identity(payload_path)
    if payload_identity[2] != condition["payload_bytes"]:
        raise ControlError("target continuation payload byte extent mismatch")
    if _sha256(payload_path) != args.conditioning_payload_sha256:
        raise ControlError("target continuation payload digest mismatch")
    if _identity(payload_path) != payload_identity:
        raise ControlError("target continuation payload changed while being authenticated")

    records = condition["tensors"]
    expected_records = {
        "input.latent_bank": ("F32", list(LATENT_BANK_SHAPE)),
        "input.qwen_bf16": ("BF16", list(QWEN_SHAPE)),
        "input.rope_cos": ("F32", [6, VIDEO_TOKENS, HEAD_DIM]),
        "input.rope_sin": ("F32", [6, VIDEO_TOKENS, HEAD_DIM]),
        "cache.eps_bf16": ("BF16", [6, 1, 32, LATENTS_PER_SEGMENT, 9, 16]),
    }
    for name, (dtype, shape) in expected_records.items():
        record = records.get(name)
        _validate_tensor_record(record, dtype, shape)
        if _sha256_range(payload_path, record["offset"], record["nbytes"]) != record["sha256"]:
            raise ControlError(f"target continuation tensor digest mismatch: {name}")

    bundle = condition["qwen_input_bundle"]
    t0_manifest = _regular_file(bundle["manifest_path"], "T0 caption Qwen input manifest")
    try:
        t0_source = continuation_inputs._admit_qwen_inputs(
            t0_manifest, bundle["manifest_sha256"]
        )
    except Exception as error:
        raise ControlError(f"T0 caption Qwen inputs rejected: {error}") from None
    t0_document = t0_source["document"]
    t0_mask = t0_document["qwen_source"]["cropped_attention_mask"]
    t0_valid = _validate_mask(t0_mask, "T0 caption")
    condition_qwen = condition["qwen_source"]
    if (condition_qwen.get("cropped_attention_mask") != t0_mask
            or condition_qwen.get("valid_tokens") != t0_valid):
        raise ControlError("T0 caption mask differs between continuation and Qwen source")
    if (t0_document["tensors"]["input.qwen_bf16"]["sha256"]
            != records["input.qwen_bf16"]["sha256"]
            or t0_source["payload_sha256"] != bundle["payload_sha256"]):
        raise ControlError("T0 caption Qwen tensor/payload differs from target continuation binding")
    if t0_document["tensors"]["input.rope_cos"]["sha256"] != bundle["first_rope_cos_sha256"]:
        raise ControlError("T0 source first RoPE cosine differs from continuation binding")
    if t0_document["tensors"]["input.rope_sin"]["sha256"] != bundle["first_rope_sin_sha256"]:
        raise ControlError("T0 source first RoPE sine differs from continuation binding")
    rope_segment_bytes = VIDEO_TOKENS * HEAD_DIM * 4
    for key in ("cos", "sin"):
        record = records[f"input.rope_{key}"]
        first_digest = _sha256_range(payload_path, record["offset"], rope_segment_bytes)
        if first_digest != bundle[f"first_rope_{key}_sha256"]:
            raise ControlError(f"target continuation RoPE bank 0 differs from T0 Qwen binding: {key}")

    try:
        t1_source = continuation_inputs._admit_qwen_inputs(
            args.t1_qwen_inputs, args.t1_qwen_inputs_sha256
        )
    except Exception as error:
        raise ControlError(f"T1 action Qwen inputs rejected: {error}") from None
    t1_document = t1_source["document"]
    if generation_inputs._resolution_for_schema(t1_document) != generation_inputs.DEFAULT_RESOLUTION:
        raise ControlError("T1 action Qwen inputs must use 512x288 geometry")
    t1_mask = t1_document["qwen_source"]["cropped_attention_mask"]
    t1_valid = _validate_mask(t1_mask, "T1 action", ACTION_VALID_TOKENS)

    t0_bundle_path = Path(bundle["manifest_path"]).resolve(strict=True)
    if t0_bundle_path != Path(t0_source["path"]).resolve(strict=True):
        raise ControlError("T0 Qwen manifest path differs from its authenticated bundle")
    return {
        "conditioning_path": condition_path,
        "conditioning": condition,
        "payload_path": payload_path,
        "latent_record": records["input.latent_bank"],
        "qwen_record": records["input.qwen_bf16"],
        "rope_cos_record": records["input.rope_cos"],
        "rope_sin_record": records["input.rope_sin"],
        "cache_epsilon_record": records["cache.eps_bf16"],
        "t0_qwen": t0_source,
        "t1_qwen": t1_source,
        "t0_valid_tokens": t0_valid,
        "t1_valid_tokens": t1_valid,
        "t0_mask": list(t0_mask),
        "t1_mask": list(t1_mask),
        "payload_sha256": args.conditioning_payload_sha256,
    }


def _authenticate_t0_latent(args: argparse.Namespace) -> dict[str, Any]:
    if args.t0_latent_bank_sha256 != T0_LATENT_BANK_SHA256:
        raise ControlError("clean T0 latent SHA differs from the explicitly pinned reference")
    selection = _validate_t0_selection(
        list(LATENT_BANK_SHAPE), args.t0_frame_start, args.t0_frame_count
    )
    expected_bytes = math.prod(LATENT_BANK_SHAPE) * 4
    digest = _authenticate_file(
        args.t0_latent_bank,
        args.t0_latent_bank_sha256,
        "clean T0 latent bank",
        expected_bytes=expected_bytes,
    )
    return {
        "path": _regular_file(args.t0_latent_bank, "clean T0 latent bank"),
        "sha256": digest,
        "shape": list(LATENT_BANK_SHAPE),
        "dtype": "F32 little-endian",
        "frame_selection": selection,
    }


def _authenticate_model_sources(args: argparse.Namespace) -> dict[str, Any]:
    upstream = _regular_dir(args.upstream, "pinned upstream checkout")
    cache_index_path = _regular_file(args.cache_index, "BF16 cache index")
    if _sha256(cache_index_path) != args.cache_index_sha256:
        raise ControlError("BF16 cache index digest mismatch")
    try:
        index = attention_reference._json_no_duplicates(
            cache_index_path.read_bytes(), "BF16 cache index"
        )
        cache_reference.validate_cache_index_document(index)
    except Exception as error:
        raise ControlError(f"BF16 cache index rejected: {error}") from None

    try:
        upstream_revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if upstream_revision != PINNED_REVISION:
            raise ControlError("pinned upstream revision mismatch")
        block_sources = block_reference._verify_upstream(upstream)
        prelude_sources = attention_reference._verify_prelude_sources(upstream)
        _, continuation_sources = continuation_inputs._verify_upstream(upstream)
        cache_storage = _verify_source_files(
            upstream,
            {"hymm/models/dit/modules/kv_cache_storage.py": two_segment.PINNED_CACHE_STORAGE_SHA256},
        )
        euler_path = upstream / single.EULER_RELATIVE
        euler_sha = _sha256(euler_path)
        if euler_sha != single.EULER_SHA256:
            raise ControlError("pinned Euler source digest mismatch")
    except ControlError:
        raise
    except Exception as error:
        raise ControlError(f"pinned upstream source admission failed: {error}") from None

    identities: dict[str, tuple[int, int, int, int, int]] = {}
    block_records = []
    total_weight_bytes = 0
    cache_dir = cache_index_path.parent
    for block_index, entry in enumerate(index["blocks"]):
        path = single._canonical_file(cache_dir / entry["bf16"], f"block {block_index} BF16 weights")
        expected = {
            name: shape
            for name, shape in cache_reference.expected_block_shapes(block_index).items()
        }
        record = single._validate_cache_shard(path, entry["bf16_sha256"], expected)
        if record["tensor_count"] != len(expected):
            raise ControlError(f"block {block_index} has an unexpected tensor count")
        identities[str(path)] = record["identity"]
        block_records.append(record)
        total_weight_bytes += record["bytes"]
    globals_path = single._canonical_file(
        cache_dir / index["globals"]["bf16"], "BF16 global weights"
    )
    globals_record = single._validate_cache_shard(
        globals_path, index["globals"]["sha256"], None, global_file=True
    )
    identities[str(globals_path)] = globals_record["identity"]
    total_weight_bytes += globals_record["bytes"]

    return {
        "upstream_path": upstream,
        "upstream_revision": upstream_revision,
        "source_hashes": {
            **block_sources,
            **prelude_sources,
            **continuation_sources,
            **cache_storage,
            single.EULER_RELATIVE: euler_sha,
        },
        "cache_index_path": cache_index_path,
        "cache_index_sha256": args.cache_index_sha256,
        "cache_index": index,
        "cache_identities": identities,
        "cache_shards": block_records,
        "global_shard": globals_record,
        "total_weight_bytes": total_weight_bytes,
    }


def _authenticate(args: argparse.Namespace) -> dict[str, Any]:
    conditioning = _authenticate_conditioning(args)
    t0_latent = _authenticate_t0_latent(args)
    model = _authenticate_model_sources(args)
    plan = _forward_plan(conditioning["t0_valid_tokens"], conditioning["t1_valid_tokens"])
    _validate_forward_plan(plan, conditioning["t0_valid_tokens"], conditioning["t1_valid_tokens"])
    return {**conditioning, **t0_latent, **model, "forward_plan": plan}


def _load_qwen(torch: Any, np: Any, source: dict[str, Any], device: Any) -> Any:
    record = source["qwen_record"]
    raw = np.memmap(
        Path(source["payload_path"]), dtype="<u2", mode="r",
        offset=record["offset"], shape=tuple(record["shape"]),
    )
    cpu = np.array(raw, copy=True)
    del raw
    qwen = torch.from_numpy(cpu).view(torch.bfloat16).to(device=device)
    if tuple(qwen.shape) != QWEN_SHAPE:
        raise ControlError("authenticated Qwen tensor geometry changed during load")
    return qwen


def _load_t0_clean(torch: Any, np: Any, admitted: dict[str, Any], device: Any) -> Any:
    raw = np.memmap(
        admitted["path"], dtype="<f4", mode="r", shape=LATENT_BANK_SHAPE
    )
    value = np.array(raw[:, :, 0:LATENTS_PER_SEGMENT, :, :], copy=True)
    del raw
    latent = torch.from_numpy(value).to(device=device, dtype=torch.bfloat16)
    if tuple(latent.shape) != LATENT_SEGMENT_SHAPE:
        raise ControlError("selected clean T0 segment has an unexpected extent")
    return latent


def _load_t1_noise(torch: Any, np: Any, admitted: dict[str, Any], device: Any) -> Any:
    raw = np.memmap(
        admitted["payload_path"], dtype="<f4", mode="r",
        offset=admitted["latent_record"]["offset"],
        shape=tuple(admitted["latent_record"]["shape"]),
    )
    value = np.array(raw[:, :, LATENTS_PER_SEGMENT:2 * LATENTS_PER_SEGMENT, :, :], copy=True)
    del raw
    latent = torch.from_numpy(value).to(device=device, dtype=torch.bfloat16)
    if tuple(latent.shape) != LATENT_SEGMENT_SHAPE:
        raise ControlError("original T1 noise segment has an unexpected extent")
    return latent


def _load_segment_rope(
    torch: Any, np: Any, admitted: dict[str, Any], name: str, segment: int, device: Any
) -> Any:
    record = admitted[f"rope_{name}_record"]
    raw = np.memmap(
        admitted["payload_path"], dtype="<f4", mode="r",
        offset=record["offset"], shape=tuple(record["shape"]),
    )
    value = np.array(raw[segment], copy=True)
    del raw
    rope = torch.from_numpy(value).to(device=device, dtype=torch.float32)
    if tuple(rope.shape) != (VIDEO_TOKENS, HEAD_DIM):
        raise ControlError("authenticated global RoPE segment geometry changed")
    return rope


def _load_cache_epsilon(torch: Any, np: Any, admitted: dict[str, Any], device: Any) -> Any:
    record = admitted["cache_epsilon_record"]
    raw = np.memmap(
        admitted["payload_path"], dtype="<u2", mode="r",
        offset=record["offset"], shape=tuple(record["shape"]),
    )
    value = np.array(raw[SEGMENT_INDEX_T0], copy=True)
    del raw
    epsilon = torch.from_numpy(value).view(torch.bfloat16).to(device=device)
    if tuple(epsilon.shape) != LATENT_SEGMENT_SHAPE:
        raise ControlError("authenticated T0 cache epsilon segment geometry changed")
    return epsilon


def _cache_attention_dispatch_factory(torch: Any, state: dict[str, Any]) -> Any:
    functional = torch.nn.functional

    def dispatch(
        q_full: Any,
        full_k: Any,
        full_v: Any,
        q_mask: Any,
        kv_mask: Any,
        cache_valid_len: int,
        update_cache: bool,
        cache_txt: bool,
    ) -> Any:
        call = state.get("call")
        if call is None or (
            cache_valid_len != call["cache_valid_len"]
            or update_cache != call["update_cache"]
            or cache_txt != call["cache_txt"]
        ):
            raise ControlError("pinned cache body escaped the admitted forward schedule")
        expected_k = cache_valid_len + VIDEO_TOKENS + (DIT_TEXT_ROWS if cache_txt else 0)
        if (
            tuple(q_full.shape) != (1, VIDEO_TOKENS + DIT_TEXT_ROWS, HEADS, HEAD_DIM)
            or tuple(full_k.shape) != (1, expected_k, HEADS, HEAD_DIM)
            or tuple(full_v.shape) != tuple(full_k.shape)
            or tuple(q_mask.shape) != (1, VIDEO_TOKENS + DIT_TEXT_ROWS)
            or tuple(kv_mask.shape) != (1, expected_k)
            or q_mask.device != q_full.device or kv_mask.device != q_full.device
            or q_mask.dtype != torch.bool or kv_mask.dtype != torch.bool
        ):
            raise ControlError("pinned cache Q/K/V tensors escaped admitted 512x288 geometry")
        expected_dtype = state["dtype"]
        if any(
            value.device.type != "mps" or value.dtype != expected_dtype
            for value in (q_full, full_k, full_v)
        ):
            raise ControlError("cached Q/K/V escaped the admitted MPS BF16 path")

        current_valid = VIDEO_TOKENS + call["valid_text_tokens"]
        q = q_full[:, :current_valid].permute(0, 2, 1, 3).contiguous()
        attention_k, attention_v = full_k, full_v
        if state["attention_layout"] == "compact":
            if cache_valid_len == 0:
                key_parts_k = [full_k[:, :VIDEO_TOKENS]]
                key_parts_v = [full_v[:, :VIDEO_TOKENS]]
                if cache_txt:
                    key_parts_k.append(
                        full_k[:, VIDEO_TOKENS:VIDEO_TOKENS + call["valid_text_tokens"]]
                    )
                    key_parts_v.append(
                        full_v[:, VIDEO_TOKENS:VIDEO_TOKENS + call["valid_text_tokens"]]
                    )
            elif cache_valid_len == CACHE_CAPACITY:
                valid_history_rows = CACHE_CAPACITY - (
                    DIT_TEXT_ROWS - state["t0_valid_tokens"]
                )
                key_parts_k = [full_k[:, :valid_history_rows]]
                key_parts_v = [full_v[:, :valid_history_rows]]
                current_image_start = CACHE_CAPACITY
                key_parts_k.append(
                    full_k[:, current_image_start:current_image_start + VIDEO_TOKENS]
                )
                key_parts_v.append(
                    full_v[:, current_image_start:current_image_start + VIDEO_TOKENS]
                )
                if cache_txt:
                    current_text_start = current_image_start + VIDEO_TOKENS
                    key_parts_k.append(
                        full_k[:, current_text_start:current_text_start + call["valid_text_tokens"]]
                    )
                    key_parts_v.append(
                        full_v[:, current_text_start:current_text_start + call["valid_text_tokens"]]
                    )
            else:
                raise ControlError("compact attention used an unadmitted history length")
            attention_k = key_parts_k[0] if len(key_parts_k) == 1 else torch.cat(key_parts_k, dim=1)
            attention_v = key_parts_v[0] if len(key_parts_v) == 1 else torch.cat(key_parts_v, dim=1)
            expected_compact = (
                VIDEO_TOKENS
                + (call["valid_text_tokens"] if cache_txt else 0)
                if cache_valid_len == 0
                else VIDEO_TOKENS
                + state["t0_valid_tokens"]
                + (VIDEO_TOKENS if cache_valid_len > 0 else 0)
                + (call["valid_text_tokens"] if cache_txt else 0)
            )
            if tuple(attention_k.shape) != (1, expected_compact, HEADS, HEAD_DIM):
                raise ControlError("compacted cache rows differ from the authenticated T0/T1 masks")
            attention_mask = None
            keep_t0_latents = state.get("t0_history_keep_latents")
            if keep_t0_latents is not None and cache_valid_len == CACHE_CAPACITY:
                if cache_txt:
                    raise ControlError("T0 history visibility diagnostic requires the admitted T1 cache read")
                mask = state.get("t0_history_attention_mask")
                if (
                    mask is None
                    or tuple(mask.shape) != (1, expected_compact)
                    or mask.device != q_full.device
                    or mask.dtype != torch.bool
                    or expected_compact != 2 * VIDEO_TOKENS + state["t0_valid_tokens"]
                ):
                    raise ControlError("T0 history visibility mask escaped compact T1 key geometry")
                attention_mask = mask
                state["receipt"]["t0_history_visibility_mask_calls"] = (
                    state["receipt"].get("t0_history_visibility_mask_calls", 0) + 1
                )
        elif state["attention_layout"] == "padded":
            attention_mask = kv_mask[:, None, None, :]
        else:
            raise ControlError("cache attention layout is not admitted")

        keys = attention_k.permute(0, 2, 1, 3).contiguous()
        values = attention_v.permute(0, 2, 1, 3).contiguous()
        result = torch.zeros(
            (1, VIDEO_TOKENS + DIT_TEXT_ROWS, HEADS, HEAD_DIM),
            dtype=expected_dtype, device=q_full.device,
        )
        chunks = 0
        for start in range(0, current_valid, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, current_valid)
            attended = functional.scaled_dot_product_attention(
                q[:, :, start:stop], keys, values,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
            result[:, start:stop].copy_(attended.permute(0, 2, 1, 3))
            chunks += 1
        receipt = state["receipt"]
        receipt["attention_calls"] += 1
        receipt["attention_query_chunks"] += chunks
        row_key = f"{expected_k}->{int(attention_k.shape[1])}"
        inventory = receipt.setdefault("attention_row_inventory", {})
        inventory[row_key] = inventory.get(row_key, 0) + 1
        return result

    return dispatch


def _build_pinned_cache_function(torch: Any, upstream: Path, state: dict[str, Any]) -> Any:
    """Compile only the pinned upstream cache body with this run's dynamic-mask SDPA."""
    source_path = upstream / "hymm/models/dit/nomas_cpt.py"
    source_sha = block_reference.PINNED_SOURCE_SHA256[source_path.relative_to(upstream).as_posix()]
    if _sha256(source_path) != source_sha:
        raise ControlError("pinned cache attention source digest mismatch")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    target = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)
         and node.name == "causal_attention_with_cache_mp"),
        None,
    )
    if target is None:
        raise ControlError("pinned upstream cache attention function is missing")
    transformer = two_segment._CacheAttentionRewrite()
    target = ast.fix_missing_locations(transformer.visit(target))
    if transformer.replacements != 1:
        raise ControlError("pinned cache body did not expose exactly one torch branch")

    helper_path = upstream / "hymm/models/dit/modules/kv_cache_storage.py"
    package = sys.modules.get("hymm.models.dit.modules")
    package_path = getattr(package, "__path__", None)
    if not package_path or Path(package_path[0]).resolve() != helper_path.parent.resolve():
        raise ControlError("pinned KV-cache package context differs from admitted source")
    if _sha256(helper_path) != two_segment.PINNED_CACHE_STORAGE_SHA256:
        raise ControlError("pinned KV-cache storage helper digest mismatch")
    loaded = sys.modules.get("hymm.models.dit.modules.kv_cache_storage")
    loaded_path = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded is not None and (loaded_path is None or Path(loaded_path).resolve() != helper_path.resolve()):
        raise ControlError("a different KV-cache storage helper is already imported")
    namespace = {
        "__name__": "hymm.models.dit.nomas_cpt",
        "__package__": "hymm.models.dit",
        "torch": torch,
        "F": torch.nn.functional,
        "Dict": typing.Dict,
        "List": typing.List,
        "Optional": typing.Optional,
        "get_sequence_parallel_state": lambda: False,
        "nccl_info": SimpleNamespace(sp_group=None, rank_within_spgroup=0),
        "ARCHIVED_HISTORY_SAMPLINGS": (),
        "HISTORY_SAMPLINGS": (),
        "_HAS_AR_VIS_PROFILER": False,
        "_ar_profile_fa_func": None,
        "cache_attention_dispatch": _cache_attention_dispatch_factory(torch, state),
    }
    module = ast.Module(body=[target], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["causal_attention_with_cache_mp"]


def _new_caches(torch: Any, device: Any) -> tuple[list[dict[str, Any]], list[Any]]:
    caches: list[dict[str, Any]] = []
    masks: list[Any] = []
    for _ in range(54):
        caches.append({
            "k": torch.zeros((1, CACHE_CAPACITY, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device),
            "v": torch.zeros((1, CACHE_CAPACITY, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device),
        })
        masks.append(torch.zeros((1, CACHE_CAPACITY), dtype=torch.bool, device=device))
    return caches, masks


def _run_control(
    admitted: dict[str, Any], args: argparse.Namespace, torch: Any,
    scheduler: Any, diffusers_version: str,
) -> tuple[dict[str, Any], bytes]:
    import numpy as np

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ControlError("PyTorch MPS is required; CPU fallback is not permitted")
    device = torch.device("mps")
    dtype = torch.bfloat16
    state: dict[str, Any] = {
        "dtype": dtype,
        "device_type": "mps",
        "call": None,
        "attention_layout": args.attention_layout,
        "t0_valid_tokens": admitted["t0_valid_tokens"],
        "receipt": {"attention_calls": 0, "attention_query_chunks": 0},
    }
    upstream = Path(admitted["upstream_path"])
    helpers = block_reference._install_upstream_modules(upstream)
    attention_state = {
        "mode": "bf16",
        "compare": single._NoopCompare(),
        "receipt": {"attention_calls": 0, "attention_query_chunks": 0},
    }
    regular_attention = attention_reference._build_attention_adapter(torch, attention_state)
    refiner_state: dict[str, Any] = {
        "mode": "bf16",
        "valid_text_tokens": admitted["t0_valid_tokens"],
        "attention_block": 0,
        "attention_query_chunks": 0,
        "compare": single._NoopCompare(),
    }
    _, refiner_attention = attention_reference._install_refiner_attention(torch, refiner_state)
    single._bind_refiner_attention_adapter(upstream, refiner_attention, refiner_state)
    model_globals = attention_reference._load_prelude_weights(
        torch, Path(admitted["global_shard"]["path"]), upstream, helpers, device, dtype
    )
    single._bind_refiner_attention_adapter(
        upstream, refiner_attention, refiner_state, model_globals["txt_in"]
    )
    blocks = single._load_all_blocks(torch, admitted, helpers, regular_attention, device)
    cache_function = _build_pinned_cache_function(torch, upstream, state)
    two_segment._bind_runtime_cache_functions(blocks, None, cache_function)
    caches, cache_masks = _new_caches(torch, device)

    t0_qwen = _load_qwen(torch, np, admitted["t0_qwen"], device)
    t0_full_mask = torch.tensor([admitted["t0_mask"]], dtype=torch.bool, device=device)
    t0_text_mask = t0_full_mask[:, :DIT_TEXT_ROWS]
    clean_t0 = _load_t0_clean(torch, np, admitted, device)
    t0_rope_cos = _load_segment_rope(torch, np, admitted, "cos", SEGMENT_INDEX_T0, device)
    t0_rope_sin = _load_segment_rope(torch, np, admitted, "sin", SEGMENT_INDEX_T0, device)
    epsilon = _load_cache_epsilon(torch, np, admitted, device)

    clean_part = clean_t0.float() * (1.0 - WRITER_SIGMA)
    noise_part = epsilon.float() * WRITER_SIGMA
    writer_latent = (clean_part + noise_part).to(dtype)
    writer_call = admitted["forward_plan"][0]
    state["call"] = writer_call
    writer_timestep = torch.full((1,), WRITER_TIMESTEP, dtype=torch.float32, device=device)
    writer_vector = model_globals["time_in"](writer_timestep)
    refiner_state["valid_text_tokens"] = admitted["t0_valid_tokens"]
    refiner_state["attention_block"] = 0
    refiner_state["attention_query_chunks"] = 0
    writer_text_full = model_globals["txt_in"](t0_qwen, writer_timestep, t0_full_mask)
    if refiner_state["attention_block"] != 2 or tuple(writer_text_full.shape) != (1, QWEN_ROWS, DIM):
        raise ControlError("pinned T0 caption refiner did not complete both attention blocks")
    writer_refiner_chunks = refiner_state["attention_query_chunks"]
    if writer_refiner_chunks != 2 * math.ceil(admitted["t0_valid_tokens"] / QUERY_CHUNK):
        raise ControlError("T0 caption refiner query chunks differ from its valid mask")
    refiner_chunks_per_forward = [writer_refiner_chunks]
    writer_text = writer_text_full[:, :DIT_TEXT_ROWS].contiguous()
    writer_image = model_globals["img_in"](writer_latent)
    if tuple(writer_image.shape) != (1, VIDEO_TOKENS, DIM):
        raise ControlError("pinned T0 writer patch embedding returned unexpected geometry")
    for block_index, block in enumerate(blocks):
        writer_image, writer_text = block(
            img=writer_image, txt=writer_text, vec=writer_vector, vec_txt=writer_vector,
            freqs_cis=(t0_rope_cos, t0_rope_sin), text_mask=t0_text_mask,
            attn_param=None, is_flash=False, txt_tokens_per_block=DIT_TEXT_ROWS,
            prompt_video_boundaries=[0, VIDEO_TOKENS], kv_cache=caches[block_index],
            **_block_cache_contract(writer_call),
            cache_attn_mode="torch", cache_mask=cache_masks[block_index],
        )
    writer_prediction = model_globals["final_layer"](writer_image, writer_vector)
    if tuple(writer_prediction.shape) != (1, VIDEO_TOKENS, 32):
        raise ControlError("pinned T0 cache writer final head returned unexpected geometry")
    del writer_prediction, writer_image, writer_text, writer_text_full, writer_vector
    del clean_part, noise_part, writer_latent, epsilon, t0_qwen, t0_full_mask, t0_text_mask
    del t0_rope_cos, t0_rope_sin

    t1_qwen = _load_qwen(torch, np, admitted["t1_qwen"], device)
    t1_full_mask = torch.tensor([admitted["t1_mask"]], dtype=torch.bool, device=device)
    t1_text_mask = t1_full_mask[:, :DIT_TEXT_ROWS]
    latent = _load_t1_noise(torch, np, admitted, device)
    rope_cos = _load_segment_rope(torch, np, admitted, "cos", SEGMENT_INDEX_T1, device)
    rope_sin = _load_segment_rope(torch, np, admitted, "sin", SEGMENT_INDEX_T1, device)
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    dts = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).to(device=device, dtype=torch.float32)
    for step in range(STEPS):
        call = admitted["forward_plan"][step + 1]
        state["call"] = call
        vector = model_globals["time_in"](timesteps[step].reshape(1))
        refiner_state["valid_text_tokens"] = admitted["t1_valid_tokens"]
        refiner_state["attention_block"] = 0
        refiner_state["attention_query_chunks"] = 0
        text_full = model_globals["txt_in"](t1_qwen, timesteps[step].reshape(1), t1_full_mask)
        if refiner_state["attention_block"] != 2 or tuple(text_full.shape) != (1, QWEN_ROWS, DIM):
            raise ControlError("pinned T1 action refiner did not complete both attention blocks")
        action_refiner_chunks = refiner_state["attention_query_chunks"]
        if action_refiner_chunks != 2 * math.ceil(admitted["t1_valid_tokens"] / QUERY_CHUNK):
            raise ControlError("T1 action refiner query chunks differ from its valid mask")
        refiner_chunks_per_forward.append(action_refiner_chunks)
        text = text_full[:, :DIT_TEXT_ROWS].contiguous()
        image = model_globals["img_in"](latent)
        if tuple(image.shape) != (1, VIDEO_TOKENS, DIM):
            raise ControlError("pinned T1 patch embedding returned unexpected geometry")
        for block_index, block in enumerate(blocks):
            image, text = block(
                img=image, txt=text, vec=vector, vec_txt=vector,
                freqs_cis=(rope_cos, rope_sin), text_mask=t1_text_mask,
                attn_param=None, is_flash=False, txt_tokens_per_block=DIT_TEXT_ROWS,
                prompt_video_boundaries=[0, VIDEO_TOKENS], kv_cache=caches[block_index],
                **_block_cache_contract(call),
                cache_attn_mode="torch", cache_mask=cache_masks[block_index],
            )
        prediction_tokens = model_globals["final_layer"](image, vector)
        if tuple(prediction_tokens.shape) != (1, VIDEO_TOKENS, 32):
            raise ControlError("pinned T1 final head returned unexpected geometry")
        prediction = prediction_tokens.reshape(1, LATENTS_PER_SEGMENT, 9, 16, 32)
        prediction = prediction.permute(0, 4, 1, 2, 3).contiguous()
        latent = (latent.float() + prediction.float() * dts[step]).to(dtype)
        del image, text, text_full, vector, prediction_tokens, prediction

    state["call"] = None
    expected_calls = (STEPS + 1) * 54
    if state["receipt"]["attention_calls"] != expected_calls:
        raise ControlError("cached attention call count differs from five-forward schedule")
    writer_sdpa_rows = (
        VIDEO_TOKENS + admitted["t0_valid_tokens"]
        if args.attention_layout == "compact" else CACHE_CAPACITY
    )
    reader_sdpa_rows = (
        VIDEO_TOKENS + admitted["t0_valid_tokens"]
        + VIDEO_TOKENS + admitted["t1_valid_tokens"]
        if args.attention_layout == "compact" else CACHE_CAPACITY + VIDEO_TOKENS
        + DIT_TEXT_ROWS
    )
    writer_rows = f"{CACHE_CAPACITY}->{writer_sdpa_rows}"
    reader_physical_rows = CACHE_CAPACITY + VIDEO_TOKENS + DIT_TEXT_ROWS
    reader_rows = f"{reader_physical_rows}->{reader_sdpa_rows}"
    expected_inventory = {writer_rows: 54, reader_rows: STEPS * 54}
    if state["receipt"].get("attention_row_inventory") != expected_inventory:
        raise ControlError("attention key-row inventory differs from writer/reader schedule")
    expected_refiner_chunks = [
        2 * math.ceil(admitted["t0_valid_tokens"] / QUERY_CHUNK)
    ] + [2 * math.ceil(admitted["t1_valid_tokens"] / QUERY_CHUNK)] * STEPS
    if refiner_chunks_per_forward != expected_refiner_chunks:
        raise ControlError("T1 refiner query chunk inventory differs from the fresh action mask")

    terminal = (
        torch.cat((clean_t0, latent), dim=2)
        .detach().float().contiguous().reshape(-1).cpu().numpy()
    )
    if terminal.size != 1 * 32 * 60 * 9 * 16 or not np.isfinite(terminal).all():
        raise ControlError("terminal two-segment latent has invalid extent or values")
    payload = np.asarray(terminal, dtype="<f4").tobytes(order="C")
    summary = {
        "device": "mps",
        "mode": "source-bf16",
        "parameter_activation_dtype": "BF16",
        "attention": "pinned upstream cache body with source key mask and chunked MPS SDPA",
        "model_forwards": STEPS + 1,
        "block_invocations": (STEPS + 1) * 54,
        "t0_denoise_forwards": 0,
        "t0_cache_writer_forwards": 1,
        "t1_denoise_forwards": STEPS,
        "cache_writer_final_head_executed": True,
        "cache_text_source": "T0 caption Qwen K/V and text history",
        "t1_current_text_source": "fresh T1 action Qwen query on every denoise step",
        "cached_attention_calls": int(state["receipt"]["attention_calls"]),
        "cached_attention_query_chunks": int(state["receipt"]["attention_query_chunks"]),
        "refiner_query_chunks_per_forward": refiner_chunks_per_forward,
        "attention_layout": args.attention_layout,
        "attention_row_inventory": [
            {"physical_key_rows": int(key.split("->")[0]),
             "sdpa_key_rows": int(key.split("->")[1]), "calls": count}
            for key, count in sorted(state["receipt"]["attention_row_inventory"].items())
        ],
        "t0_valid_text_tokens": admitted["t0_valid_tokens"],
        "t1_valid_text_tokens": admitted["t1_valid_tokens"],
        "cache_rows_written_per_block": CACHE_CAPACITY,
        "cache_valid_rows_written_per_block": VIDEO_TOKENS + admitted["t0_valid_tokens"],
        "cache_text_write_count": 1,
        "cache_writer_timestep": WRITER_TIMESTEP,
        "cache_writer_sigma": WRITER_SIGMA,
        "rope_bank_sequence": [0, 1, 1, 1, 1],
        "activation_readbacks_before_terminal_export": 0,
        "terminal_export_count": 1,
        "terminal_values": int(terminal.size),
        "terminal_finite": True,
        "cache_storage_dtype": "BF16",
        "persistent_cache_storage_bytes": 2 * 54 * CACHE_CAPACITY * HEADS * HEAD_DIM * 2
        + 54 * CACHE_CAPACITY,
        "diffusers_version": diffusers_version,
        "schedule": single._schedule_record(scheduler),
    }
    return summary, payload


def _manifest(
    admitted: dict[str, Any], args: argparse.Namespace, summary: dict[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    payload_sha = hashlib.sha256(payload).hexdigest()
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "scope": "512x288 two-segment reference continuation: supplied clean T0 plus four T1 action denoise steps",
        "result": {
            "payload": "pending.latent.f32le",
            "payload_bytes": len(payload),
            "payload_sha256": payload_sha,
            "dtype": "F32",
            "shape": [1, 32, 60, 9, 16],
            "t0_values": "supplied clean T0 source values cast to BF16 and expanded to F32",
            "t1_values": "four-step upstream BF16 denoised action continuation expanded to F32",
        },
        "source": {
            "conditioning_manifest": str(admitted["conditioning_path"]),
            "conditioning_manifest_sha256": args.conditioning_manifest_sha256,
            "conditioning_payload_sha256": admitted["payload_sha256"],
            "t0_caption_qwen_manifest_sha256": admitted["t0_qwen"]["manifest_sha256"],
            "t0_caption_qwen_payload_sha256": admitted["t0_qwen"]["payload_sha256"],
            "t0_caption_qwen_tensor_sha256": admitted["t0_qwen"]["qwen_record"]["sha256"],
            "t1_action_qwen_manifest_sha256": admitted["t1_qwen"]["manifest_sha256"],
            "t1_action_qwen_payload_sha256": admitted["t1_qwen"]["payload_sha256"],
            "t1_action_qwen_tensor_sha256": admitted["t1_qwen"]["qwen_record"]["sha256"],
            "clean_t0_latent_bank": str(admitted["path"]),
            "clean_t0_latent_bank_sha256": admitted["sha256"],
            "clean_t0_frame_selection": admitted["frame_selection"],
            "cache_index": str(admitted["cache_index_path"]),
            "cache_index_sha256": admitted["cache_index_sha256"],
            "global_weights": {
                key: value for key, value in admitted["global_shard"].items()
                if key != "identity"
            },
            "block_shards": [
                {key: value for key, value in record.items() if key != "identity"}
                for record in admitted["cache_shards"]
            ],
            "upstream_revision": admitted["upstream_revision"],
            "upstream_source_sha256": admitted["source_hashes"],
            "total_resident_checkpoint_bytes": admitted["total_weight_bytes"],
        },
        "configuration": {
            "mode": "source-bf16",
            "segment_latents": LATENTS_PER_SEGMENT,
            "steps": STEPS,
            "shift": FLOW_SHIFT,
            "video_tokens": VIDEO_TOKENS,
            "qwen_rows": QWEN_ROWS,
            "dit_text_rows": DIT_TEXT_ROWS,
            "t0_caption_valid_prefix_rows": admitted["t0_valid_tokens"],
            "t1_action_valid_prefix_rows": admitted["t1_valid_tokens"],
            "cache_writer_sigma": WRITER_SIGMA,
            "cache_writer_timestep": WRITER_TIMESTEP,
            "planned_process_cap_gib": args.planned_cap_gib,
            "planned_process_cap_os_enforced_by_script": False,
            "attention_adapter": {
                "implementation": "PyTorch MPS scaled_dot_product_attention",
                "query_chunk": QUERY_CHUNK,
                "magi_cuda_parity": False,
            },
        },
        "execution": summary,
        "limitations": [
            "T0 denoising is skipped; the cache writer consumes the supplied authenticated clean latent.",
            "Chunked MPS SDPA replaces the pinned Magi dispatch; CUDA/Magi parity is not claimed.",
            "PixelUMM decoding, full media generation, and native-runtime equivalence are outside this control.",
        ],
    }


def _validate_output_target(path_arg: Path | str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute() or path.suffix.lower() != ".json" or not path.parent.is_dir():
        raise ControlError("--output must be a new absolute .json path in an existing directory")
    payload = path.with_name(f"{path.stem}.latent.f32le")
    if path.is_symlink() or payload.is_symlink() or os.path.lexists(path) or os.path.lexists(payload):
        raise ControlError("output manifest and latent payload must both be new non-symlink files")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioning-manifest", type=Path, required=True)
    parser.add_argument("--conditioning-manifest-sha256", required=True)
    parser.add_argument("--conditioning-payload-sha256", required=True)
    parser.add_argument("--t1-qwen-inputs", type=Path, default=DEFAULT_T1_QWEN_INPUTS)
    parser.add_argument("--t1-qwen-inputs-sha256", default=T1_QWEN_INPUTS_SHA256)
    parser.add_argument("--t0-latent-bank", type=Path, default=DEFAULT_T0_LATENT_BANK)
    parser.add_argument("--t0-latent-bank-sha256", default=T0_LATENT_BANK_SHA256)
    parser.add_argument("--t0-frame-start", type=int, default=0)
    parser.add_argument("--t0-frame-count", type=int, default=LATENTS_PER_SEGMENT)
    parser.add_argument("--cache-index", type=Path, default=single.DEFAULT_CACHE_INDEX)
    parser.add_argument("--cache-index-sha256", default=single.CACHE_INDEX_SHA256)
    parser.add_argument("--upstream", type=Path, default=single.DEFAULT_UPSTREAM)
    parser.add_argument("--attention-layout", choices=("padded", "compact"), default="padded")
    parser.add_argument("--planned-cap-gib", type=float, default=25.5)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    for value, label in (
        (args.conditioning_manifest_sha256, "--conditioning-manifest-sha256"),
        (args.conditioning_payload_sha256, "--conditioning-payload-sha256"),
        (args.t1_qwen_inputs_sha256, "--t1-qwen-inputs-sha256"),
        (args.t0_latent_bank_sha256, "--t0-latent-bank-sha256"),
        (args.cache_index_sha256, "--cache-index-sha256"),
    ):
        if not isinstance(value, str) or _HEX.fullmatch(value) is None:
            parser.error(f"{label} must be a lowercase SHA-256 digest")
    if args.t1_qwen_inputs_sha256 != T1_QWEN_INPUTS_SHA256:
        parser.error("T1 action Qwen manifest SHA differs from the pinned source input")
    if args.t0_latent_bank_sha256 != T0_LATENT_BANK_SHA256:
        parser.error("clean T0 latent SHA differs from the pinned reference")
    if args.cache_index_sha256 != single.CACHE_INDEX_SHA256:
        parser.error("cache-index SHA differs from the pinned BF16 weights index")
    if not math.isfinite(args.planned_cap_gib) or not 0 < args.planned_cap_gib <= MAX_PLANNED_CAP_GIB:
        parser.error(f"--planned-cap-gib must be in (0, {MAX_PLANNED_CAP_GIB}] GiB")
    if args.execute and args.output is None:
        parser.error("--output is required with --execute")
    if args.plan_only and args.output is not None:
        parser.error("--output is valid only with --execute")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        admitted = _authenticate(args)
        if args.plan_only:
            print(json.dumps({
                "schema": SCHEMA,
                "status": "validated_plan_only",
                "model_launched": False,
                "mps_device_used": False,
                "conditioning_manifest_sha256": args.conditioning_manifest_sha256,
                "conditioning_payload_sha256": admitted["payload_sha256"],
                "t0_caption_valid_tokens": admitted["t0_valid_tokens"],
                "t1_action_valid_tokens": admitted["t1_valid_tokens"],
                "t0_latent_bank_sha256": admitted["sha256"],
                "t0_frame_selection": admitted["frame_selection"],
                "upstream_revision": admitted["upstream_revision"],
                "primary_source_sha256": admitted["source_hashes"],
                "forward_plan": admitted["forward_plan"],
                "geometry": {
                    "resolution": "512x288",
                    "segments": 2,
                    "latent_frames_per_segment": LATENTS_PER_SEGMENT,
                    "dit_video_tokens": VIDEO_TOKENS,
                    "dit_text_rows": DIT_TEXT_ROWS,
                    "qwen_rows": QWEN_ROWS,
                    "cache_capacity_rows": CACHE_CAPACITY,
                },
                "planned_cap_gib_is_metadata_only": True,
            }, sort_keys=True))
            return 0

        args.output = _validate_output_target(args.output)
        single._configure_mps_runtime()
        torch, scheduler, diffusers_version = single._prepare_cpu_scheduler(
            Path(admitted["upstream_path"]), "source-bf16"
        )
        with torch.inference_mode():
            summary, payload = _run_control(admitted, args, torch, scheduler, diffusers_version)
        document = _manifest(admitted, args, summary, payload)
        payload_path, manifest_sha = single._publish(args.output, document, payload)
        print(json.dumps({
            "status": "completed",
            "manifest": str(args.output),
            "payload": str(payload_path),
            "manifest_sha256": manifest_sha,
            "payload_sha256": document["result"]["payload_sha256"],
            "model_forwards": summary["model_forwards"],
            "block_invocations": summary["block_invocations"],
            "terminal_export_count": summary["terminal_export_count"],
        }, sort_keys=True))
        return 0
    except (ControlError, single.ControlError, OSError, ValueError, RuntimeError) as error:
        print(f"reference-continuation control rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
