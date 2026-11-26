#!/usr/bin/env python3
"""Run a bounded source-pinned two-segment Hy-Nomas boundary control.

This extends the authenticated one-segment arithmetic control with exactly
four denoise forwards for segment 0, one noisy segment-0 cache writer, and
four denoise forwards for segment 1. The pinned upstream cache function is
executed unchanged except for replacing its Flash/Magi dispatch with chunked
MPS SDPA using the function's own computed key mask. Same-scene text is cached
once: segment 0 uses its current text K/V and writer; segment 1 keeps current
text Q while attending the cached segment-0 text K/V.

No GPU/model execution occurs with --validate-only. Full execution requires
an external memory-guarded launcher. The planned cap is metadata only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import typing
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_bf16_dit_quality_control as single  # noqa: E402
import hy_nomas_real_attention_reference as attention_reference  # noqa: E402
import hy_nomas_reference as block_reference  # noqa: E402

SCHEMA = "cogni_ml.hy_nomas_two_segment_quality_control/v1"
BASE_CONTROL_SHA256 = "e711f50dcde9e037b04334938eb9d0a1ad899381f2dd74069bc88d858379356d"
SEGMENT_COUNT = 2
SEGMENT_LATENTS = 30
STEPS = 4
FLOW_SHIFT = 5.0
WRITER_SIGMA = 0.206897
WRITER_TIMESTEP = 206.897
VIDEO_TOKENS = 30 * 9 * 16
TEXT_TOKENS = 576
VALID_TEXT_TOKENS = 557
TEXT_RMS_TOKENS = 1280
DIM = 2048
HEADS = 16
HEAD_DIM = 128
CACHE_ROWS_PER_SOURCE_CHUNK = VIDEO_TOKENS + TEXT_TOKENS
CACHE_VALID_ROWS_PER_SOURCE_CHUNK = VIDEO_TOKENS + VALID_TEXT_TOKENS
# Only segment 0 is written. Segment 1 reads this history and appends current
# image K/V for attention without writing a second cache segment.
CACHE_CAPACITY = CACHE_ROWS_PER_SOURCE_CHUNK
MAX_ATTENTION_KEY_ROWS = CACHE_CAPACITY + VIDEO_TOKENS
QUERY_CHUNK = attention_reference.QUERY_CHUNK
CPU_CACHE_SMOKE_GEOMETRY = {
    "video_tokens": 2,
    "text_tokens": 2,
    "valid_text_tokens": 1,
    "heads": 2,
    "head_dim": 4,
    "query_chunk": 2,
    "cache_capacity": 4,
}
MAX_PLANNED_CAP_GIB = 26.0
HASH_CHUNK = 8 * 1024 * 1024
PINNED_CACHE_STORAGE_SHA256 = (
    "53fdb67aaa967b9195f17c1d829feb12cd1e17dfa21a4e5bc0a8492790f8f0c3"
)


class ControlError(RuntimeError):
    """Fail-closed source, artifact, schedule, or execution error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, HASH_CHUNK))
            if not chunk:
                raise ControlError("authenticated tensor slice is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _forward_plan() -> list[dict[str, Any]]:
    """The only admitted schedule: 4 denoise, writer0, 4 denoise."""
    plan: list[dict[str, Any]] = []
    for step in range(STEPS):
        plan.append(
            {
                "kind": "denoise",
                "segment": 0,
                "step": step,
                "cache_valid_len": 0,
                "update_cache": False,
                "cache_txt": True,
                "rope_bank": 0,
                "timestep": "euler",
            }
        )
    plan.append(
        {
            "kind": "writer",
            "segment": 0,
            "step": None,
            "cache_valid_len": 0,
            "update_cache": True,
            "cache_txt": True,
            "rope_bank": 0,
            "timestep": WRITER_TIMESTEP,
        }
    )
    for step in range(STEPS):
        plan.append(
            {
                "kind": "denoise",
                "segment": 1,
                "step": step,
                "cache_valid_len": CACHE_ROWS_PER_SOURCE_CHUNK,
                "update_cache": False,
                "cache_txt": False,
                "rope_bank": 1,
                "timestep": "euler",
            }
        )
    return plan


def _validate_forward_plan(plan: list[dict[str, Any]]) -> None:
    expected = _forward_plan()
    if plan != expected or len(plan) != 2 * STEPS + 1:
        raise ControlError(
            "two-segment forward/cache-text schedule differs from admission"
        )
    if sum(call["update_cache"] for call in plan) != 1:
        raise ControlError("the two-segment control must contain one cache writer")


def _expected_attention_key_rows(cache_valid_len: int, cache_txt: bool) -> int:
    if type(cache_valid_len) is not int or type(cache_txt) is not bool:
        raise ControlError("attention key extent requires admitted cache state types")
    if cache_valid_len not in (0, CACHE_ROWS_PER_SOURCE_CHUNK):
        raise ControlError("attention key extent uses an unadmitted history length")
    return cache_valid_len + VIDEO_TOKENS + (TEXT_TOKENS if cache_txt else 0)


def _cache_storage_inventory() -> dict[str, int]:
    blocks = 54
    mask_bytes = blocks * CACHE_CAPACITY
    bf16_kv_bytes = 2 * blocks * CACHE_CAPACITY * HEADS * HEAD_DIM * 2
    f32_kv_bytes = 2 * blocks * CACHE_CAPACITY * HEADS * HEAD_DIM * 4
    return {
        "block_count": blocks,
        "capacity_rows_per_block": CACHE_CAPACITY,
        "mask_bytes": mask_bytes,
        "bf16_kv_bytes": bf16_kv_bytes,
        "bf16_total_bytes": bf16_kv_bytes + mask_bytes,
        "f32_kv_bytes": f32_kv_bytes,
        "f32_total_bytes": f32_kv_bytes + mask_bytes,
    }


def _round_cache_writer_input(torch: Any, noisy_f32: Any, output_dtype: Any) -> Any:
    """Apply native/source BF16 cache-writer storage rounding after F32 math."""
    rounded_bf16 = noisy_f32.to(torch.bfloat16)
    if output_dtype == torch.bfloat16:
        return rounded_bf16
    if output_dtype == torch.float32:
        return rounded_bf16.to(torch.float32)
    raise ControlError("cache-writer output dtype is not an admitted mode")


def _geometry() -> dict[str, Any]:
    return {
        "segment_count": SEGMENT_COUNT,
        "latent_frames_per_segment": SEGMENT_LATENTS,
        "terminal_latent_shape": [1, 32, SEGMENT_COUNT * SEGMENT_LATENTS, 9, 16],
        "video_tokens": VIDEO_TOKENS,
        "qwen_storage_rows": TEXT_RMS_TOKENS,
        "dit_text_storage_rows": TEXT_TOKENS,
        "valid_text_rows": VALID_TEXT_TOKENS,
        "upstream_cache_writer_rows": CACHE_ROWS_PER_SOURCE_CHUNK,
        "upstream_cache_writer_valid_rows": CACHE_VALID_ROWS_PER_SOURCE_CHUNK,
        "upstream_cache_capacity_rows": CACHE_CAPACITY,
        "maximum_attention_key_rows": MAX_ATTENTION_KEY_ROWS,
        "persistent_cache_storage_bytes": _cache_storage_inventory(),
        "native_compact_history_rows_after_writer": CACHE_VALID_ROWS_PER_SOURCE_CHUNK,
        "source_padded_cache_rows_are_masked": True,
        "max_query_rows": VIDEO_TOKENS + TEXT_TOKENS,
        "valid_query_rows": VIDEO_TOKENS + VALID_TEXT_TOKENS,
        "attention_query_chunk": QUERY_CHUNK,
    }


def _validate_writer_contract(writer: Any) -> dict[str, Any]:
    if type(writer) is not dict:
        raise ControlError("authenticated cache-writer contract is missing")
    expected = {
        "cache_at_step": -1,
        "cache_noise_sigma": WRITER_SIGMA,
        "cache_timestep": WRITER_TIMESTEP,
        "cache_latent_source": "final denoised segment latent selected by cache_at_step=-1",
        "cache_write_formula": "((1-sigma)*cache_latents.float()+sigma*cache_eps.float()).to(cache_latents.dtype)",
        "cache_eps_tensor": "cache.eps_bf16",
        "cache_eps_dtype": "BF16",
        "cache_eps_seed": 42,
        "cache_eps_generator_device": "mps",
        "cache_eps_sample_dtype": "F32 then cast to BF16",
        "cache_eps_upstream_cpu_rng_parity": False,
    }
    if any(writer.get(name) != value for name, value in expected.items()):
        raise ControlError(
            "authenticated cache-writer metadata differs from saved Echo contract"
        )
    draws = writer.get("cache_eps_draws")
    if type(draws) is not list or len(draws) != 6:
        raise ControlError(
            "authenticated cache-writer epsilon draw inventory is malformed"
        )
    for index, draw in enumerate(draws):
        if (
            type(draw) is not dict
            or draw.get("segment_index") != index
            or draw.get("draw_order") != index
            or draw.get("seed") != 42
            or draw.get("generator_device") != "mps"
            or draw.get("global_latent_start") != index * SEGMENT_LATENTS
            or draw.get("global_latent_stop") != (index + 1) * SEGMENT_LATENTS
            or draw.get("shape") != [1, 32, SEGMENT_LATENTS, 9, 16]
            or draw.get("dtype") != "BF16"
        ):
            raise ControlError(
                "authenticated cache-writer epsilon draw order/geometry changed"
            )
    return writer


def _validate_text_mask(text_mask: Any) -> None:
    if (
        type(text_mask) is not list
        or len(text_mask) != TEXT_RMS_TOKENS
        or sum(bool(value) for value in text_mask) != VALID_TEXT_TOKENS
        or text_mask[:VALID_TEXT_TOKENS] != [True] * VALID_TEXT_TOKENS
        or any(text_mask[VALID_TEXT_TOKENS:])
    ):
        raise ControlError(
            "authenticated text mask is not the admitted valid-prefix layout"
        )


def _validate_conditioning_records(admitted: dict[str, Any]) -> dict[str, Any]:
    condition = admitted["conditioning"]
    records = condition["tensors"]
    latent = records["input.latent_bank"]
    cos = records["input.rope_cos"]
    sin = records["input.rope_sin"]
    eps = records["cache.eps_bf16"]
    if latent.get("dtype") != "F32" or latent.get("shape") != [1, 32, 180, 9, 16]:
        raise ControlError(
            "authenticated Echo latent bank is not six 30-frame F32 segments"
        )
    if (
        cos.get("dtype") != "F32"
        or sin.get("dtype") != "F32"
        or cos.get("shape") != [6, VIDEO_TOKENS, HEAD_DIM]
        or sin.get("shape") != [6, VIDEO_TOKENS, HEAD_DIM]
    ):
        raise ControlError("authenticated global RoPE banks have unexpected geometry")
    if eps.get("dtype") != "BF16" or eps.get("shape") != [6, 1, 32, 30, 9, 16]:
        raise ControlError("authenticated cache epsilon bank has unexpected geometry")
    _validate_text_mask(admitted["base_qwen"]["valid_text_mask"])

    payload = Path(admitted["conditioning_payload"])
    rope_bank_bytes = VIDEO_TOKENS * HEAD_DIM * 4
    eps_segment_bytes = 1 * 32 * SEGMENT_LATENTS * 9 * 16 * 2

    def slice_hash(record: dict[str, Any], index: int, stride: int) -> str:
        return _sha256_range(payload, record["offset"] + index * stride, stride)

    def latent_segment_hash(index: int) -> str:
        # The latent bank is NCHTW, so a temporal segment is a discontiguous
        # run per channel. Hash the canonical contiguous segment serialization.
        digest = hashlib.sha256()
        bytes_per_channel = SEGMENT_LATENTS * 9 * 16 * 4
        for channel in range(32):
            offset = latent["offset"] + (
                (channel * 180 + index * SEGMENT_LATENTS) * 9 * 16 * 4
            )
            with payload.open("rb") as stream:
                stream.seek(offset)
                remaining = bytes_per_channel
                while remaining:
                    chunk = stream.read(min(remaining, HASH_CHUNK))
                    if not chunk:
                        raise ControlError("authenticated latent segment is truncated")
                    digest.update(chunk)
                    remaining -= len(chunk)
        return digest.hexdigest()

    return {
        "latent_segment_sha256": [latent_segment_hash(i) for i in range(SEGMENT_COUNT)],
        "rope_cos_bank_sha256": [
            slice_hash(cos, i, rope_bank_bytes) for i in range(SEGMENT_COUNT)
        ],
        "rope_sin_bank_sha256": [
            slice_hash(sin, i, rope_bank_bytes) for i in range(SEGMENT_COUNT)
        ],
        "cache_epsilon_segment_sha256": [
            slice_hash(eps, i, eps_segment_bytes) for i in range(SEGMENT_COUNT)
        ],
        "latent_record_sha256": latent["sha256"],
        "rope_cos_record_sha256": cos["sha256"],
        "rope_sin_record_sha256": sin["sha256"],
        "cache_epsilon_record_sha256": eps["sha256"],
        "writer_contract": _validate_writer_contract(condition.get("cache_writer")),
    }


def _authenticate(args: argparse.Namespace) -> dict[str, Any]:
    base_sha = _sha256(Path(single.__file__))
    if base_sha != BASE_CONTROL_SHA256:
        raise ControlError("frozen one-segment control source changed")
    admitted = single._authenticate(args)
    admitted["two_segment_inputs"] = _validate_conditioning_records(admitted)
    return admitted


def _load_latent_segment(
    torch: Any, np: Any, admitted: dict[str, Any], segment: int, dtype: Any, device: Any
) -> Any:
    record = admitted["latent_record"]
    raw = np.memmap(
        Path(admitted["conditioning_payload"]),
        dtype="<f4",
        mode="r",
        offset=record["offset"],
        shape=tuple(record["shape"]),
    )
    value = np.array(
        raw[:, :, segment * SEGMENT_LATENTS : (segment + 1) * SEGMENT_LATENTS],
        copy=True,
    )
    del raw
    result = torch.from_numpy(value).to(device=device, dtype=dtype)
    if tuple(result.shape) != (1, 32, SEGMENT_LATENTS, 9, 16):
        raise ControlError(f"Echo latent segment {segment} has an unexpected extent")
    return result


def _load_rope_bank(
    torch: Any, np: Any, admitted: dict[str, Any], key: str, segment: int, device: Any
) -> Any:
    record = admitted[f"rope_{key}_record"]
    raw = np.memmap(
        Path(admitted["conditioning_payload"]),
        dtype="<f4",
        mode="r",
        offset=record["offset"],
        shape=tuple(record["shape"]),
    )
    value = np.array(raw[segment], copy=True)
    del raw
    tensor = torch.from_numpy(value).to(device=device, dtype=torch.float32)
    if tuple(tensor.shape) != (VIDEO_TOKENS, HEAD_DIM):
        raise ControlError(f"global RoPE bank {segment}/{key} has wrong shape")
    return tensor


def _load_cache_epsilon(
    torch: Any, np: Any, admitted: dict[str, Any], segment: int, device: Any
) -> Any:
    record = admitted["conditioning"]["tensors"]["cache.eps_bf16"]
    raw = np.memmap(
        Path(admitted["conditioning_payload"]),
        dtype="<u2",
        mode="r",
        offset=record["offset"],
        shape=tuple(record["shape"]),
    )
    value = np.array(raw[segment], copy=True)
    del raw
    tensor = torch.from_numpy(value).view(torch.bfloat16).to(device=device)
    if tuple(tensor.shape) != (1, 32, SEGMENT_LATENTS, 9, 16):
        raise ControlError("saved cache epsilon has an unexpected segment shape")
    return tensor


class _CacheAttentionRewrite(ast.NodeTransformer):
    """Replace only pinned cache function's torch SDPA branch with mask-aware MPS SDPA."""

    def __init__(self) -> None:
        self.replacements = 0

    @staticmethod
    def _is_torch_branch(test: ast.expr) -> bool:
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "attn_mode"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "torch"
        )

    def visit_If(self, node: ast.If) -> Any:
        node = self.generic_visit(node)
        if self._is_torch_branch(node.test):
            self.replacements += 1
            call = ast.Call(
                func=ast.Name(id="cache_attention_dispatch", ctx=ast.Load()),
                args=[
                    ast.Name(id=name, ctx=ast.Load())
                    for name in (
                        "q_full",
                        "full_k",
                        "full_v",
                        "q_mask",
                        "kv_mask",
                        "cache_valid_len",
                        "update_cache",
                        "cache_txt",
                    )
                ],
                keywords=[],
            )
            node.body = [
                ast.Assign(
                    targets=[ast.Name(id="hidden", ctx=ast.Store())],
                    value=call,
                )
            ]
        return node


def _build_pinned_cache_function(
    torch: Any, upstream: Path, state: dict[str, Any]
) -> Any:
    source_path = upstream / "hymm/models/dit/nomas_cpt.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "causal_attention_with_cache_mp"
        ),
        None,
    )
    if target is None:
        raise ControlError("pinned cache attention function is missing")
    target = ast.fix_missing_locations(target)
    transformer = _CacheAttentionRewrite()
    target = transformer.visit(target)
    target = ast.fix_missing_locations(target)
    if transformer.replacements != 1:
        raise ControlError("expected exactly one pinned torch cache-attention branch")

    helper_path = upstream / "hymm/models/dit/modules/kv_cache_storage.py"
    package = sys.modules.get("hymm.models.dit.modules")
    package_path = getattr(package, "__path__", None)
    if (
        not package_path
        or Path(package_path[0]).resolve() != helper_path.parent.resolve()
    ):
        raise ControlError(
            "pinned KV-cache package context is not installed at the admitted source"
        )
    if not helper_path.is_file() or _sha256(helper_path) != PINNED_CACHE_STORAGE_SHA256:
        raise ControlError("pinned KV-cache storage helper digest mismatch")
    loaded_helper = sys.modules.get("hymm.models.dit.modules.kv_cache_storage")
    if loaded_helper is not None:
        loaded_path = getattr(loaded_helper, "__file__", None)
        if loaded_path is None or Path(loaded_path).resolve() != helper_path.resolve():
            raise ControlError("a different KV-cache storage module is already loaded")
    namespace = {
        "__name__": "hymm.models.dit.nomas_cpt",
        "__package__": "hymm.models.dit",
        "torch": torch,
        "F": torch.nn.functional,
        "Dict": typing.Dict,
        "List": typing.List,
        "Optional": typing.Optional,
        "get_sequence_parallel_state": lambda: False,
        # The pinned function reads this attribute before checking the SP
        # branch; false SP state means the value is never otherwise consumed.
        "nccl_info": SimpleNamespace(sp_group=None, rank_within_spgroup=0),
        "ARCHIVED_HISTORY_SAMPLINGS": (),
        "HISTORY_SAMPLINGS": (),
        "_HAS_AR_VIS_PROFILER": False,
        "_ar_profile_fa_func": None,
        "cache_attention_dispatch": _cache_attention_dispatch_factory(torch, state),
    }
    module = ast.Module(body=[target], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace
    )
    return namespace["causal_attention_with_cache_mp"]


def _compact_attention_keys(
    torch: Any,
    full_k: Any,
    full_v: Any,
    cache_valid_len: int,
    cache_txt: bool,
    geometry: dict[str, int],
) -> tuple[Any, Any]:
    """Drop only source-masked text padding while preserving source row order."""
    video_tokens = geometry["video_tokens"]
    text_tokens = geometry["text_tokens"]
    valid_text_tokens = geometry["valid_text_tokens"]
    cache_capacity = geometry["cache_capacity"]
    if cache_valid_len == 0:
        parts_k = [full_k[:, :video_tokens]]
        parts_v = [full_v[:, :video_tokens]]
        if cache_txt:
            parts_k.append(full_k[:, video_tokens : video_tokens + valid_text_tokens])
            parts_v.append(full_v[:, video_tokens : video_tokens + valid_text_tokens])
    elif cache_valid_len == cache_capacity:
        # A history chunk is image rows followed by padded text rows. Its valid
        # text prefix is authenticated from the saved Qwen mask. The current
        # segment contributes image rows and, only for cache_txt=True, text K/V.
        valid_history_rows = cache_capacity - (text_tokens - valid_text_tokens)
        parts_k = [full_k[:, :valid_history_rows]]
        parts_v = [full_v[:, :valid_history_rows]]
        parts_k.append(full_k[:, cache_valid_len : cache_valid_len + video_tokens])
        parts_v.append(full_v[:, cache_valid_len : cache_valid_len + video_tokens])
        if cache_txt:
            text_start = cache_valid_len + video_tokens
            parts_k.append(full_k[:, text_start : text_start + valid_text_tokens])
            parts_v.append(full_v[:, text_start : text_start + valid_text_tokens])
    else:
        raise ControlError("compact attention used an unadmitted history length")
    if len(parts_k) == 1:
        return parts_k[0], parts_v[0]
    return torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)


def _cache_attention_dispatch_factory(torch: Any, state: dict[str, Any]) -> Any:
    functional = torch.nn.functional
    smoke_geometry = state.get("cpu_smoke_geometry")
    if smoke_geometry is None:
        geometry = {
            "video_tokens": VIDEO_TOKENS,
            "text_tokens": TEXT_TOKENS,
            "valid_text_tokens": VALID_TEXT_TOKENS,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "query_chunk": QUERY_CHUNK,
            "cache_capacity": CACHE_CAPACITY,
        }
    else:
        geometry = smoke_geometry

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
        context = state.get("call")
        if context is None:
            raise ControlError("cache attention ran outside an admitted forward")
        valid_len = context["cache_valid_len"]
        if (
            cache_valid_len != valid_len
            or update_cache != context["update_cache"]
            or cache_txt != context["cache_txt"]
        ):
            raise ControlError(
                "pinned cache function call differs from the admitted schedule"
            )
        video_tokens = geometry["video_tokens"]
        text_tokens = geometry["text_tokens"]
        valid_text_tokens = geometry["valid_text_tokens"]
        heads = geometry["heads"]
        head_dim = geometry["head_dim"]
        attention_layout = state.get("attention_layout", "padded")
        if attention_layout not in ("padded", "compact"):
            raise ControlError("attention layout is not admitted")
        if smoke_geometry is None:
            expected_k = _expected_attention_key_rows(valid_len, cache_txt)
        else:
            if valid_len not in (0, geometry["cache_capacity"]):
                raise ControlError("CPU cache smoke used an unadmitted history length")
            expected_k = valid_len + video_tokens + (text_tokens if cache_txt else 0)
        if (
            tuple(q_full.shape) != (1, video_tokens + text_tokens, heads, head_dim)
            or tuple(full_k.shape) != (1, expected_k, heads, head_dim)
            or tuple(full_v.shape) != tuple(full_k.shape)
            or tuple(q_mask.shape) != (1, video_tokens + text_tokens)
            or tuple(kv_mask.shape) != (1, expected_k)
            or q_mask.device != q_full.device
            or kv_mask.device != q_full.device
            or q_mask.dtype != torch.bool
            or kv_mask.dtype != torch.bool
        ):
            raise ControlError(
                "pinned cache attention tensors differ from two-segment geometry"
            )
        expected_dtype = state["dtype"]
        if any(
            value.device.type != state["device_type"] or value.dtype != expected_dtype
            for value in (q_full, full_k, full_v)
        ):
            raise ControlError("cached Q/K/V escaped admitted device or precision")
        valid_queries = video_tokens + valid_text_tokens
        query_heads = q_full[:, :valid_queries].permute(0, 2, 1, 3).contiguous()
        if attention_layout == "compact":
            attention_k, attention_v = _compact_attention_keys(
                torch, full_k, full_v, valid_len, cache_txt, geometry
            )
            expected_sdpa_k = expected_k - (text_tokens - valid_text_tokens) * (
                int(valid_len > 0) + int(cache_txt)
            )
            if tuple(attention_k.shape) != (
                1,
                expected_sdpa_k,
                heads,
                head_dim,
            ) or tuple(attention_v.shape) != tuple(attention_k.shape):
                raise ControlError(
                    "compact attention key extent differs from the authenticated text prefix"
                )
            allowed = None
        else:
            attention_k, attention_v = full_k, full_v
            allowed = kv_mask[:, None, None, :]
        key_heads = attention_k.permute(0, 2, 1, 3).contiguous()
        value_heads = attention_v.permute(0, 2, 1, 3).contiguous()
        result = torch.zeros(
            (1, video_tokens + text_tokens, heads, head_dim),
            dtype=expected_dtype,
            device=q_full.device,
        )
        query_chunks = 0
        query_chunk = geometry["query_chunk"]
        for start in range(0, valid_queries, query_chunk):
            stop = min(start + query_chunk, valid_queries)
            attended = functional.scaled_dot_product_attention(
                query_heads[:, :, start:stop],
                key_heads,
                value_heads,
                attn_mask=allowed,
                dropout_p=0.0,
                is_causal=False,
            )
            result[:, start:stop].copy_(attended.permute(0, 2, 1, 3))
            query_chunks += 1
        receipt = state["receipt"]
        receipt["attention_calls"] += 1
        receipt["attention_query_chunks"] += query_chunks
        physical_key_rows = int(full_k.shape[1])
        sdpa_key_rows = int(attention_k.shape[1])
        row_key = f"{physical_key_rows}->{sdpa_key_rows}"
        inventory = receipt.setdefault("attention_row_inventory", {})
        inventory[row_key] = inventory.get(row_key, 0) + 1
        if smoke_geometry is not None:
            receipt.setdefault("cpu_smoke_calls", []).append(
                {
                    "attention_layout": attention_layout,
                    "cache_valid_len": valid_len,
                    "update_cache": update_cache,
                    "cache_txt": cache_txt,
                    "query_rows": int(q_full.shape[1]),
                    "valid_query_rows": valid_queries,
                    # Keep the original physical-row field for older padded
                    # receipts; record rows actually presented to SDPA too.
                    "key_rows": physical_key_rows,
                    "sdpa_key_rows": sdpa_key_rows,
                    "kv_mask": [bool(value) for value in kv_mask[0].tolist()],
                }
            )
        return result

    return dispatch


def _bind_cache_function(blocks: list[Any], cache_function: Any) -> None:
    if not blocks:
        raise ControlError("pinned cache function cannot bind without loaded blocks")
    namespaces = [block.forward_with_cache.__globals__ for block in blocks]
    if any(namespace is not namespaces[0] for namespace in namespaces):
        raise ControlError(
            "loaded source blocks do not share the pinned class namespace"
        )
    namespaces[0]["causal_attention_with_cache_mp"] = cache_function
    if any(
        namespace.get("causal_attention_with_cache_mp") is not cache_function
        for namespace in namespaces
    ):
        raise ControlError(
            "pinned block forward is not bound to cache function adapter"
        )


def _bind_runtime_cache_functions(
    blocks: list[Any], f32_block: Any | None, cache_function: Any
) -> None:
    """Bind independently compiled source/F32 class namespaces to the same adapter."""
    _bind_cache_function(blocks, cache_function)
    if f32_block is not None:
        # _new_reusable_f32_block extracts the pinned class into a fresh module
        # namespace. Binding only the BF16-loaded classes leaves this F32 class's
        # unchanged forward_with_cache body resolving the missing upstream global.
        _bind_cache_function([f32_block], cache_function)


def _execute_pinned_cache_cpu_smoke(
    torch: Any,
    block_type: Any,
    cache_function: Any,
    state: dict[str, Any],
    upstream: Path,
) -> dict[str, Any]:
    """Execute the pinned body through its no-cache, writer, and reader branches."""
    smoke = CPU_CACHE_SMOKE_GEOMETRY
    block = block_type(
        hidden_size=smoke["heads"] * smoke["head_dim"],
        heads_num=smoke["heads"],
        mlp_width_ratio=2.0,
        mlp_act_type="gelu_tanh",
        attn_mode="magi_mp_causal",
        qk_norm=True,
        qk_norm_type="rms",
        qkv_bias=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).eval()
    _bind_cache_function([block], cache_function)
    if (
        block.forward_with_cache.__globals__.get("causal_attention_with_cache_mp")
        is not cache_function
    ):
        raise ControlError("CPU smoke block did not bind the compiled cache function")

    hidden = smoke["heads"] * smoke["head_dim"]
    img = (
        torch.arange(
            1, 1 + smoke["video_tokens"] * hidden, dtype=torch.float32
        ).reshape(1, smoke["video_tokens"], hidden)
        / 16
    )
    txt = (
        torch.arange(1, 1 + smoke["text_tokens"] * hidden, dtype=torch.float32).reshape(
            1, smoke["text_tokens"], hidden
        )
        / 11
    )
    vec = torch.linspace(-0.25, 0.25, hidden, dtype=torch.float32).reshape(1, hidden)
    text_mask = torch.tensor([[True, False]], dtype=torch.bool)
    cache = {
        "k": torch.zeros(
            (1, smoke["cache_capacity"], smoke["heads"], smoke["head_dim"]),
            dtype=torch.float32,
        ),
        "v": torch.zeros(
            (1, smoke["cache_capacity"], smoke["heads"], smoke["head_dim"]),
            dtype=torch.float32,
        ),
    }
    cache_mask = torch.zeros((1, smoke["cache_capacity"]), dtype=torch.bool)
    calls = (
        {"cache_valid_len": 0, "update_cache": False, "cache_txt": True},
        {"cache_valid_len": 0, "update_cache": True, "cache_txt": True},
        {
            "cache_valid_len": smoke["cache_capacity"],
            "update_cache": False,
            "cache_txt": False,
        },
    )
    with torch.inference_mode():
        for index, call in enumerate(calls):
            state["call"] = call
            img_out, txt_out = block.forward_with_cache(
                img,
                txt,
                vec,
                text_mask=text_mask,
                kv_cache=cache,
                cache_valid_len=call["cache_valid_len"],
                update_cache=call["update_cache"],
                attn_mode="torch",
                cache_txt=call["cache_txt"],
                cache_mask=cache_mask,
            )
            if tuple(img_out.shape) != tuple(img.shape) or tuple(
                txt_out.shape
            ) != tuple(txt.shape):
                raise ControlError(
                    "pinned cache CPU smoke returned wrong block output shapes"
                )
            if not torch.isfinite(img_out).all() or not torch.isfinite(txt_out).all():
                raise ControlError("pinned cache CPU smoke returned non-finite output")
            if index == 0 and (
                cache_mask.any()
                or torch.count_nonzero(cache["k"]).item() != 0
                or torch.count_nonzero(cache["v"]).item() != 0
            ):
                raise ControlError(
                    "pinned no-cache branch unexpectedly wrote KV history"
                )

    helper_path = upstream / "hymm/models/dit/modules/kv_cache_storage.py"
    loaded_helper = sys.modules.get("hymm.models.dit.modules.kv_cache_storage")
    loaded_path = getattr(loaded_helper, "__file__", None)
    if loaded_path is None or Path(loaded_path).resolve() != helper_path.resolve():
        raise ControlError(
            "pinned relative KV-cache helper import resolved incorrectly"
        )
    if not callable(getattr(loaded_helper, "cache_format", None)) or not callable(
        getattr(loaded_helper, "cache_geometry", None)
    ):
        raise ControlError("pinned KV-cache helper exports are missing")
    receipt = state["receipt"]
    expected_calls = [
        {
            "attention_layout": state.get("attention_layout", "padded"),
            "cache_valid_len": 0,
            "update_cache": False,
            "cache_txt": True,
            "query_rows": 4,
            "valid_query_rows": 3,
            "key_rows": 4,
            "sdpa_key_rows": 3 if state.get("attention_layout") == "compact" else 4,
            "kv_mask": [True, True, True, False],
        },
        {
            "attention_layout": state.get("attention_layout", "padded"),
            "cache_valid_len": 0,
            "update_cache": True,
            "cache_txt": True,
            "query_rows": 4,
            "valid_query_rows": 3,
            "key_rows": 4,
            "sdpa_key_rows": 3 if state.get("attention_layout") == "compact" else 4,
            "kv_mask": [True, True, True, False],
        },
        {
            "attention_layout": state.get("attention_layout", "padded"),
            "cache_valid_len": 4,
            "update_cache": False,
            "cache_txt": False,
            "query_rows": 4,
            "valid_query_rows": 3,
            "key_rows": 6,
            "sdpa_key_rows": 5 if state.get("attention_layout") == "compact" else 6,
            "kv_mask": [True, True, True, False, True, True],
        },
    ]
    if receipt.get("cpu_smoke_calls") != expected_calls:
        raise ControlError(
            "pinned cache CPU smoke did not traverse expected cache branches"
        )
    if not torch.equal(cache_mask, torch.tensor([[True, True, True, False]])):
        raise ControlError("pinned cache writer failed to persist the source text mask")
    return {
        "executed": True,
        "device": "cpu",
        "block_forward_bound_to_cache_function": True,
        "cache_function_package": cache_function.__globals__.get("__package__"),
        "relative_helper": str(helper_path),
        "relative_helper_sha256": PINNED_CACHE_STORAGE_SHA256,
        "branches": ["no_cache", "writer", "reader"],
        "attention_calls": receipt["attention_calls"],
        "attention_query_chunks": receipt["attention_query_chunks"],
        "observed_key_rows": [call["key_rows"] for call in expected_calls],
        "observed_sdpa_key_rows": [call["sdpa_key_rows"] for call in expected_calls],
        "attention_layout": state.get("attention_layout", "padded"),
        "attention_row_inventory": [
            {
                "physical_key_rows": int(key.split("->")[0]),
                "sdpa_key_rows": int(key.split("->")[1]),
                "calls": count,
            }
            for key, count in sorted(receipt.get("attention_row_inventory", {}).items())
        ],
        "observed_kv_masks": [call["kv_mask"] for call in expected_calls],
    }


def _validate_pinned_cache_contracts(
    torch: Any, upstream: Path, mode: str, attention_layout: str = "padded"
) -> dict[str, Any]:
    """Preflight pinned block/cache signatures and execute tiny CPU cache paths."""
    if attention_layout not in ("padded", "compact"):
        raise ControlError("attention layout is not admitted")
    helpers = block_reference._install_upstream_modules(upstream)
    state = {
        "dtype": torch.float32,
        "device_type": "cpu",
        "cpu_smoke_geometry": CPU_CACHE_SMOKE_GEOMETRY,
        "attention_layout": attention_layout,
        "receipt": {
            "attention_calls": 0,
            "attention_query_chunks": 0,
            "cpu_smoke_calls": [],
            "attention_row_inventory": {},
        },
    }
    attention_adapter = attention_reference._build_attention_adapter(
        torch,
        {
            "mode": "f32" if mode == "native-f32" else "bf16",
            "compare": single._NoopCompare(),
            "receipt": {"attention_calls": 0, "attention_query_chunks": 0},
        },
    )
    block_type = block_reference._extract_pinned_block(
        upstream,
        torch,
        attention_adapter,
        helpers[-1].apply_rotary_emb,
        helpers,
    )
    block = block_type(
        hidden_size=DIM,
        heads_num=HEADS,
        mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh",
        attn_mode="magi_mp_causal",
        qk_norm=True,
        qk_norm_type="rms",
        qkv_bias=True,
        dtype=single._mode_dtype(torch, mode),
        device=torch.device("meta"),
    )
    # The native-f32 control creates this reusable class through a second
    # _extract_pinned_block call, so its globals are a distinct namespace.
    f32_block_type = block_reference._extract_pinned_block(
        upstream,
        torch,
        attention_adapter,
        helpers[-1].apply_rotary_emb,
        helpers,
    )
    f32_block = f32_block_type(
        hidden_size=DIM,
        heads_num=HEADS,
        mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh",
        attn_mode="magi_mp_causal",
        qk_norm=True,
        qk_norm_type="rms",
        qkv_bias=True,
        dtype=torch.float32,
        device=torch.device("meta"),
    )
    block_kwargs = {
        "img": None,
        "txt": None,
        "vec": None,
        "vec_txt": None,
        "freqs_cis": None,
        "text_mask": None,
        "kv_cache": {"k": None, "v": None},
        "cache_valid_len": 0,
        "update_cache": False,
        "cache_attn_mode": "torch",
        "cache_txt": True,
        "cache_mask": None,
    }
    try:
        __import__("inspect").signature(block.forward).bind(**block_kwargs)
        __import__("inspect").signature(block.forward_with_cache).bind(
            img=None,
            txt=None,
            vec=None,
            vec_txt=None,
            freqs_cis=None,
            text_mask=None,
            kv_cache={},
            cache_valid_len=0,
            update_cache=False,
            attn_mode="torch",
            cache_txt=True,
            cache_mask=None,
        )
        function = _build_pinned_cache_function(torch, upstream, state)
        __import__("inspect").signature(function).bind(
            (None, None),
            (None, None),
            (None, None),
            kv_cache={},
            cache_valid_len=0,
            attn_mode="torch",
            update_cache=False,
            cache_txt=True,
            cache_mask=None,
            seg_ar_attention_backend="magi",
        )
        _bind_runtime_cache_functions([block], f32_block, function)
        if (
            block.forward_with_cache.__globals__.get("causal_attention_with_cache_mp")
            is not function
            or f32_block.forward_with_cache.__globals__.get(
                "causal_attention_with_cache_mp"
            )
            is not function
        ):
            raise ControlError(
                "source and reusable F32 block namespaces must share the cache adapter"
            )
        smoke = _execute_pinned_cache_cpu_smoke(
            torch, block_type, function, state, upstream
        )
        if function.__globals__.get("__package__") != "hymm.models.dit":
            raise ControlError(
                "compiled pinned cache function has incorrect package context"
            )
        return smoke
    except TypeError as error:
        raise ControlError(
            f"pinned two-segment call contract rejected: {error}"
        ) from None
    except ControlError:
        raise
    except Exception as error:
        raise ControlError(
            f"pinned cache call-contract preflight failed: {type(error).__name__}: {error}"
        ) from None


def _new_caches(
    torch: Any, dtype: Any, device: Any
) -> tuple[list[dict[str, Any]], list[Any]]:
    caches = []
    masks = []
    for _ in range(54):
        caches.append(
            {
                "k": torch.zeros(
                    (1, CACHE_CAPACITY, HEADS, HEAD_DIM), dtype=dtype, device=device
                ),
                "v": torch.zeros(
                    (1, CACHE_CAPACITY, HEADS, HEAD_DIM), dtype=dtype, device=device
                ),
            }
        )
        masks.append(torch.zeros((1, CACHE_CAPACITY), dtype=torch.bool, device=device))
    return caches, masks


def _run_control_impl(
    admitted: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
    scheduler: Any,
    diffusers_version: str,
) -> tuple[dict[str, Any], bytes]:
    import numpy as np

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ControlError("a PyTorch MPS device is required")
    device = torch.device("mps")
    dtype = single._mode_dtype(torch, args.mode)
    native_f32 = args.mode == "native-f32"
    adapter_mode = "f32" if native_f32 else "bf16"
    state: dict[str, Any] = {
        "dtype": dtype,
        "device_type": "mps",
        "call": None,
        "attention_layout": args.attention_layout,
        "receipt": {"attention_calls": 0, "attention_query_chunks": 0},
    }
    helpers = block_reference._install_upstream_modules(Path(admitted["upstream_path"]))
    regular_attention = attention_reference._build_attention_adapter(
        torch,
        {
            "mode": adapter_mode,
            "compare": single._NoopCompare(),
            "receipt": {"attention_calls": 0, "attention_query_chunks": 0},
        },
    )
    refiner_state: dict[str, Any] = {
        "mode": adapter_mode,
        "valid_text_tokens": VALID_TEXT_TOKENS,
        "attention_block": 0,
        "attention_query_chunks": 0,
        "compare": single._NoopCompare(),
    }
    _, refiner_attention = attention_reference._install_refiner_attention(
        torch, refiner_state
    )
    upstream = Path(admitted["upstream_path"])
    single._bind_refiner_attention_adapter(upstream, refiner_attention, refiner_state)
    model_globals = attention_reference._load_prelude_weights(
        torch,
        Path(admitted["global_shard"]["path"]),
        upstream,
        helpers,
        device,
        dtype,
    )
    single._bind_refiner_attention_adapter(
        upstream, refiner_attention, refiner_state, model_globals["txt_in"]
    )
    blocks = single._load_all_blocks(
        torch, admitted, helpers, regular_attention, device
    )
    cache_function = _build_pinned_cache_function(torch, upstream, state)
    f32_block = f32_copy_plans = f32_storage_pointers = None
    if native_f32:
        f32_block = single._new_reusable_f32_block(
            torch,
            upstream,
            helpers,
            regular_attention,
            device,
        )
        f32_copy_plans, f32_storage_pointers = single._prepare_f32_block_copy_plan(
            torch,
            blocks,
            f32_block,
        )
    _bind_runtime_cache_functions(blocks, f32_block, cache_function)
    caches, cache_masks = _new_caches(torch, dtype, device)

    qwen = single._load_qwen(torch, np, admitted, device)
    if native_f32:
        qwen = qwen.to(torch.float32)
    full_mask = torch.tensor(
        [admitted["base_qwen"]["valid_text_mask"]], dtype=torch.bool, device=device
    )
    text_mask = full_mask[:, :TEXT_TOKENS]
    if tuple(full_mask.shape) != (1, TEXT_RMS_TOKENS) or tuple(text_mask.shape) != (
        1,
        TEXT_TOKENS,
    ):
        raise ControlError("full Qwen mask or retained text query extent changed")

    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    dts = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).to(
        device=device, dtype=torch.float32
    )
    plan = _forward_plan()
    _validate_forward_plan(plan)
    terminal_segments: list[Any] = []
    for segment in range(SEGMENT_COUNT):
        latent = _load_latent_segment(torch, np, admitted, segment, dtype, device)
        if native_f32:
            latent = latent.to(torch.bfloat16).to(torch.float32)
        rope_cos = _load_rope_bank(torch, np, admitted, "cos", segment, device)
        rope_sin = _load_rope_bank(torch, np, admitted, "sin", segment, device)
        denoise_calls = [
            call
            for call in plan
            if call["kind"] == "denoise" and call["segment"] == segment
        ]
        for call in denoise_calls:
            step = call["step"]
            assert isinstance(step, int)
            state["call"] = call
            vector = model_globals["time_in"](timesteps[step].reshape(1))
            refiner_state["attention_block"] = 0
            text_full = model_globals["txt_in"](
                qwen, timesteps[step].reshape(1), full_mask
            )
            if refiner_state["attention_block"] != 2 or tuple(text_full.shape) != (
                1,
                TEXT_RMS_TOKENS,
                DIM,
            ):
                raise ControlError(
                    "pinned full-row refiner did not complete both attention blocks"
                )
            text = text_full[:, :TEXT_TOKENS].contiguous()
            image = model_globals["img_in"](latent)
            if tuple(image.shape) != (1, VIDEO_TOKENS, DIM):
                raise ControlError(
                    "pinned patch embedding returned unexpected geometry"
                )
            for block_index, source_block in enumerate(blocks):
                if native_f32:
                    assert (
                        f32_block is not None
                        and f32_copy_plans is not None
                        and f32_storage_pointers is not None
                    )
                    single._copy_source_block_into_f32(
                        torch,
                        f32_copy_plans[block_index],
                        f32_storage_pointers,
                        block_index,
                    )
                    block = f32_block
                else:
                    block = source_block
                image, text = block(
                    img=image,
                    txt=text,
                    vec=vector,
                    vec_txt=vector,
                    freqs_cis=(rope_cos, rope_sin),
                    text_mask=text_mask,
                    attn_param=None,
                    is_flash=False,
                    txt_tokens_per_block=TEXT_TOKENS,
                    prompt_video_boundaries=[0, VIDEO_TOKENS],
                    kv_cache=caches[block_index],
                    cache_valid_len=call["cache_valid_len"],
                    update_cache=False,
                    cache_attn_mode="torch",
                    cache_txt=call["cache_txt"],
                    cache_mask=cache_masks[block_index],
                )
            prediction_tokens = model_globals["final_layer"](image, vector)
            if tuple(prediction_tokens.shape) != (1, VIDEO_TOKENS, 32):
                raise ControlError("pinned final head returned unexpected geometry")
            prediction = prediction_tokens.reshape(1, SEGMENT_LATENTS, 9, 16, 32)
            prediction = prediction.permute(0, 4, 1, 2, 3).contiguous()
            if native_f32:
                velocity = prediction.to(torch.bfloat16).to(torch.float32)
                delta = velocity * dts[step]
                latent = (latent + delta).to(torch.bfloat16).to(torch.float32)
                del velocity, delta
            else:
                latent = (latent.float() + prediction.float() * dts[step]).to(dtype)
            del image, text, text_full, vector, prediction_tokens, prediction

        if segment == 0:
            clean = latent
            epsilon = _load_cache_epsilon(torch, np, admitted, 0, device)
            clean_part = clean.float() * (1.0 - WRITER_SIGMA)
            noise_part = epsilon.float() * WRITER_SIGMA
            noisy_f32 = clean_part + noise_part
            noisy = _round_cache_writer_input(torch, noisy_f32, dtype)
            writer = next(call for call in plan if call["kind"] == "writer")
            state["call"] = writer
            writer_timestep = torch.full(
                (1,), WRITER_TIMESTEP, dtype=torch.float32, device=device
            )
            vector = model_globals["time_in"](writer_timestep)
            refiner_state["attention_block"] = 0
            text_full = model_globals["txt_in"](qwen, writer_timestep, full_mask)
            if refiner_state["attention_block"] != 2 or tuple(text_full.shape) != (
                1,
                TEXT_RMS_TOKENS,
                DIM,
            ):
                raise ControlError("pinned noisy-cache writer refiner did not complete")
            text = text_full[:, :TEXT_TOKENS].contiguous()
            image = model_globals["img_in"](noisy)
            for block_index, source_block in enumerate(blocks):
                if native_f32:
                    assert (
                        f32_block is not None
                        and f32_copy_plans is not None
                        and f32_storage_pointers is not None
                    )
                    single._copy_source_block_into_f32(
                        torch,
                        f32_copy_plans[block_index],
                        f32_storage_pointers,
                        block_index,
                    )
                    block = f32_block
                else:
                    block = source_block
                image, text = block(
                    img=image,
                    txt=text,
                    vec=vector,
                    vec_txt=vector,
                    freqs_cis=(rope_cos, rope_sin),
                    text_mask=text_mask,
                    attn_param=None,
                    is_flash=False,
                    txt_tokens_per_block=TEXT_TOKENS,
                    prompt_video_boundaries=[0, VIDEO_TOKENS],
                    kv_cache=caches[block_index],
                    cache_valid_len=0,
                    update_cache=True,
                    cache_attn_mode="torch",
                    cache_txt=True,
                    cache_mask=cache_masks[block_index],
                )
            # The upstream cache-writing segment still traverses the pinned
            # final head; its prediction is discarded because this forward
            # populates history rather than advancing a denoise state.
            writer_prediction = model_globals["final_layer"](image, vector)
            if tuple(writer_prediction.shape) != (1, VIDEO_TOKENS, 32):
                raise ControlError(
                    "pinned cache-writer final head returned wrong geometry"
                )
            del writer_prediction, clean_part, noise_part, noisy, epsilon
            del image, text, text_full, vector
            terminal_segments.append(latent.detach().clone())
        else:
            terminal_segments.append(latent.detach().clone())
        del latent, rope_cos, rope_sin

    state["call"] = None
    expected_calls = (2 * STEPS + 1) * 54
    calls = int(state["receipt"]["attention_calls"])
    chunks_per_call = math.ceil((VIDEO_TOKENS + VALID_TEXT_TOKENS) / QUERY_CHUNK)
    chunks = int(state["receipt"]["attention_query_chunks"])
    if calls != expected_calls or chunks != expected_calls * chunks_per_call:
        raise ControlError(
            "cached-attention forward/chunk count differs from 2-segment schedule"
        )
    if refiner_state["attention_query_chunks"] != (2 * STEPS + 1) * 2 * math.ceil(
        VALID_TEXT_TOKENS / QUERY_CHUNK
    ):
        raise ControlError("text refiner query chunk inventory differs from schedule")
    if args.attention_layout == "compact":
        writer_sdpa_rows = CACHE_VALID_ROWS_PER_SOURCE_CHUNK
        reader_sdpa_rows = CACHE_VALID_ROWS_PER_SOURCE_CHUNK + VIDEO_TOKENS
    else:
        writer_sdpa_rows = CACHE_ROWS_PER_SOURCE_CHUNK
        reader_sdpa_rows = CACHE_ROWS_PER_SOURCE_CHUNK + VIDEO_TOKENS
    expected_row_inventory = {
        f"{CACHE_ROWS_PER_SOURCE_CHUNK}->{writer_sdpa_rows}": 5 * 54,
        f"{CACHE_ROWS_PER_SOURCE_CHUNK + VIDEO_TOKENS}->{reader_sdpa_rows}": 4 * 54,
    }
    if state["receipt"].get("attention_row_inventory") != expected_row_inventory:
        raise ControlError(
            "physical/SDPA attention row inventory differs from the two-segment schedule"
        )

    terminal = (
        torch.cat(terminal_segments, dim=2)
        .detach()
        .float()
        .contiguous()
        .reshape(-1)
        .cpu()
        .numpy()
    )
    if terminal.size != 1 * 32 * 60 * 9 * 16 or not np.isfinite(terminal).all():
        raise ControlError("terminal two-segment latent has invalid extent or values")
    payload = np.asarray(terminal, dtype="<f4").tobytes(order="C")
    summary = {
        "device": "mps",
        "mode": args.mode,
        "parameter_activation_dtype": "F32" if native_f32 else "BF16",
        "attention": (
            "pinned upstream cache body; source-masked padded keys passed to chunked MPS SDPA"
            if args.attention_layout == "padded"
            else "pinned upstream cache body; false text-padding rows compacted before unmasked chunked MPS SDPA"
        ),
        "model_forwards": len(plan),
        "block_invocations": len(plan) * 54,
        "cache_writer_final_head_executed": True,
        "cached_attention_calls": calls,
        "cached_attention_query_chunks": chunks,
        "attention_layout": args.attention_layout,
        "attention_row_inventory": [
            {
                "physical_key_rows": int(key.split("->")[0]),
                "sdpa_key_rows": int(key.split("->")[1]),
                "calls": count,
            }
            for key, count in sorted(
                state["receipt"].get("attention_row_inventory", {}).items()
            )
        ],
        "refiner_forwards": len(plan),
        "refiner_attention_calls": len(plan) * 2,
        "activation_readbacks_before_terminal_export": 0,
        "terminal_export_count": 1,
        "terminal_values": int(terminal.size),
        "terminal_finite": True,
        "cache_storage_dtype": "F32" if native_f32 else "BF16",
        "persistent_cache_storage_bytes": _cache_storage_inventory()[
            "f32_total_bytes" if native_f32 else "bf16_total_bytes"
        ],
        "cache_rows_written_per_block": CACHE_ROWS_PER_SOURCE_CHUNK,
        "cache_valid_rows_written_per_block": CACHE_VALID_ROWS_PER_SOURCE_CHUNK,
        "cache_valid_len_for_segment1": CACHE_ROWS_PER_SOURCE_CHUNK,
        "cache_text_write_count": 1,
        "cache_writer_timestep": WRITER_TIMESTEP,
        "cache_writer_sigma": WRITER_SIGMA,
        "rope_bank_sequence": [0, 0, 1],
        "diffusers_version": diffusers_version,
        "schedule": single._schedule_record(scheduler),
    }
    return summary, payload


def _run_control(
    admitted: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
    scheduler: Any,
    diffusers_version: str,
) -> tuple[dict[str, Any], bytes]:
    with torch.inference_mode():
        return _run_control_impl(admitted, args, torch, scheduler, diffusers_version)


def _manifest(
    admitted: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    native_f32 = args.mode == "native-f32"
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "scope": "two Echo segments; four denoise steps each; one segment-0 cache writer; 54 blocks",
        "result": {
            "payload": "pending",
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "dtype": "F32",
            "shape": [1, 32, 60, 9, 16],
            "value_semantics": "final states after per-step BF16 Euler storage, expanded to F32",
        },
        "source": {
            "conditioning_manifest": admitted["conditioning_path"],
            "conditioning_manifest_sha256": single.CONDITIONING_SHA256,
            "conditioning_payload_sha256": admitted["conditioning"]["payload_sha256"],
            "base_qwen_manifest_sha256": admitted["source_info"][
                "base_qwen_manifest_sha256"
            ],
            "base_qwen_payload_sha256": admitted["source_info"][
                "base_qwen_payload_sha256"
            ],
            "qwen_tensor_sha256": admitted["base_qwen"]["qwen_tensor_sha256"],
            "cache_index": admitted["cache_index_path"],
            "cache_index_sha256": single.CACHE_INDEX_SHA256,
            "global_weights": admitted["global_shard"],
            "block_shards": [
                {k: v for k, v in row.items() if k != "identity"}
                for row in admitted["cache_shards"]
            ],
            "upstream_revision": admitted["upstream_revision"],
            "upstream_source_sha256": admitted["upstream_source_sha256"],
            "prelude_source_sha256": admitted["prelude_source_sha256"],
            "continuation_upstream_source_sha256": admitted[
                "continuation_upstream_source_sha256"
            ],
            "euler_source_sha256": admitted["euler_source_sha256"],
            "two_segment_tensor_identities": admitted["two_segment_inputs"],
            "two_segment_script_sha256": _sha256(Path(__file__)),
            "base_one_segment_script_sha256": BASE_CONTROL_SHA256,
        },
        "configuration": {
            "mode": args.mode,
            "attention_layout": args.attention_layout,
            "geometry": _geometry(),
            "forward_schedule": _forward_plan(),
            "cache_text_policy": "OncePerScene; current text queries remain active; segment-0 text K/V cached once",
            "cache_function": "pinned causal_attention_with_cache_mp AST; only attn_mode=torch SDPA dispatch replaced",
            "cache_attention_mask": (
                "pinned source-built kv_mask passed into chunked SDPA; physical padded rows remain stored and masked"
                if args.attention_layout == "padded"
                else "pinned source-built kv_mask is retained and shape-checked; authenticated false text-padding rows are sliced from K/V before unmasked SDPA"
            ),
            "cache_writer_formula": "F32 multiply/add; round noised latent to BF16, then retain BF16 in source-bf16 or expand to F32 in native-f32",
            "cache_writer_timestep": WRITER_TIMESTEP,
            "cache_writer_sigma": WRITER_SIGMA,
            "rope_bank_sequence": [0, 0, 1],
            "steps": STEPS,
            "shift": FLOW_SHIFT,
            "precision_boundaries": {
                "latent_input": "raw F32 latent rounded to BF16 and expanded in native-f32; BF16 in source-bf16",
                "patch_refiner_blocks_final": "pinned upstream bodies; BF16 checkpoint weights; activation dtype follows selected mode",
                "history_kv": "source cache dtype follows selected mode; mask and row order are source cache function outputs",
                "euler": "native-f32 rounds velocity and each updated latent to BF16 with separate F32 multiply/add; source-bf16 follows pinned F32 expression then BF16 state boundary",
            },
            "attention_adapter": {
                "implementation": "PyTorch MPS SDPA",
                "query_chunk": QUERY_CHUNK,
                "layout": args.attention_layout,
                "valid_text_prefix": VALID_TEXT_TOKENS,
                "text_padding_rows_removed_per_text_bank": (
                    TEXT_TOKENS - VALID_TEXT_TOKENS
                    if args.attention_layout == "compact"
                    else 0
                ),
                "magi_cuda_parity": False,
            },
            "planned_process_cap_gib": args.planned_cap_gib,
            "planned_process_cap_os_enforced_by_script": False,
        },
        "execution": summary,
        "limitations": [
            "The source cache function's pinned composition, cache writes, row ordering, and masks are retained; only its torch attention dispatch is replaced with chunked MPS SDPA.",
            "This MPS attention-layout control does not establish Magi/CUDA parity or native compact-history Metal parity.",
            "No Magi/CUDA parity, Qwen inference, complete video, or PixelUMM decode is included.",
            "This is an arithmetic/trajectory control and does not establish that the eye or motion artifact is fixed or caused by a segment boundary.",
        ]
        + (
            [
                "native-f32 uses BF16 checkpoint weights with F32 activations and native BF16 Euler rounding; it is not a stock-F32 claim."
            ]
            if native_f32
            else [
                "source-bf16 covers only two 5-second segments, not full video generation."
            ]
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("source-bf16", "native-f32"), default="source-bf16"
    )
    parser.add_argument(
        "--attention-layout", choices=("padded", "compact"), default="padded"
    )
    parser.add_argument(
        "--conditioning-manifest", type=Path, default=single.DEFAULT_CONDITIONING
    )
    parser.add_argument("--cache-index", type=Path, default=single.DEFAULT_CACHE_INDEX)
    parser.add_argument("--upstream", type=Path, default=single.DEFAULT_UPSTREAM)
    parser.add_argument("--planned-cap-gib", type=float, default=26.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.planned_cap_gib)
        or not 0 < args.planned_cap_gib <= MAX_PLANNED_CAP_GIB
    ):
        parser.error(f"--planned-cap-gib must be in (0, {MAX_PLANNED_CAP_GIB}] GiB")
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only is selected")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if not args.validate_only:
            args.output = single._validate_output_target(args.output)
            single._configure_mps_runtime()
        admitted = _authenticate(args)
        torch, scheduler, diffusers_version = single._prepare_cpu_scheduler(
            Path(admitted["upstream_path"]),
            args.mode,
        )
        cache_preflight = _validate_pinned_cache_contracts(
            torch,
            Path(admitted["upstream_path"]),
            args.mode,
            args.attention_layout,
        )
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "status": "validated_only",
                        "mode": args.mode,
                        "attention_layout": args.attention_layout,
                        "torch_version": str(torch.__version__),
                        "diffusers_version": diffusers_version,
                        "scheduler_device": "cpu",
                        "mps_device_used": False,
                        "model_launched": False,
                        "schedule": single._schedule_record(scheduler),
                        "geometry": _geometry(),
                        "forward_schedule": _forward_plan(),
                        "source_contracts": "pinned block/cache/refiner signatures, cache no-cache/writer/reader CPU smoke, and one AST dispatch replacement admitted",
                        "cache_body_cpu_smoke": cache_preflight,
                    },
                    sort_keys=True,
                )
            )
            return 0
        summary, payload = _run_control(
            admitted, args, torch, scheduler, diffusers_version
        )
        document = _manifest(admitted, args, summary, payload)
        payload_path, manifest_sha = single._publish(args.output, document, payload)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "manifest": str(args.output),
                    "payload": str(payload_path),
                    "manifest_sha256": manifest_sha,
                    "payload_sha256": document["result"]["payload_sha256"],
                    "mode": args.mode,
                    "model_forwards": summary["model_forwards"],
                    "block_invocations": summary["block_invocations"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (ControlError, single.ControlError) as error:
        print(f"control error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
