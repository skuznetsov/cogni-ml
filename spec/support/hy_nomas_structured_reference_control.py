#!/usr/bin/env python3
"""Run a pinned T0-cache to structured T1-action Hy-Nomas control.

The existing continuation control owns artifact admission and the 54-block,
four-step BF16 execution. This wrapper supplies an authenticated T1 action
profile and replaces only its cache-attention dispatch: T0 remains unmasked,
while T1 uses the exact pinned upstream intra-segment range builder with an
explicit chunked PyTorch MPS SDPA adapter.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import typing
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
EXAMPLE_SUPPORT = ROOT / "examples" / "support"
for _path in (SUPPORT, EXAMPLE_SUPPORT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hy_nomas_action_conditioning as action_conditioning  # noqa: E402
import hy_nomas_reference_continuation_control as base  # noqa: E402

SCHEMA = "cogni_ml.hy_nomas_structured_reference_control/v1"
PROFILE_SCHEMA = "cogni_ml.hy_nomas_action_conditioning/v1"
PROFILE_TEMPLATE_SHA256 = "3e7b69bdf2a057bf2d6a8322dfb8ad619e686e4e8240b65f29e94492bee0c4a6"
TEXT_ROW_ALIGNMENT = 64
ACTION_LOCAL_START = 0.5
ACTION_LOCAL_END = 4.5
ACTION_VIDEO_WINDOWS = [[0, 3, -1], [3, 27, 0], [27, 30, -1]]
ACTION_DESCRIPTION_TERMS = ("head", "turn")
_SOURCE_RANGE_FUNCTION = "_build_intra_seg_cache_ranges"
_SOURCE_RANGE_RELATIVE = "hymm/models/dit/nomas_cpt.py"


class StructuredControlError(RuntimeError):
    """Fail-closed profile, mask, or MPS execution rejection."""


def _round_dit_text_rows(caption_valid_tokens: int, action_valid_tokens: int) -> int:
    """Round the largest admitted text prefix to the source's 64-row block."""
    for value, label in ((caption_valid_tokens, "caption"), (action_valid_tokens, "action")):
        if type(value) is not int or not 0 < value <= base.QWEN_ROWS:
            raise StructuredControlError(f"{label} valid-token count is outside Qwen geometry")
    rows = ((max(caption_valid_tokens, action_valid_tokens) + TEXT_ROW_ALIGNMENT - 1)
            // TEXT_ROW_ALIGNMENT) * TEXT_ROW_ALIGNMENT
    if rows > base.QWEN_ROWS:
        raise StructuredControlError("rounded DIT text rows exceed the authenticated Qwen extent")
    return rows


def _validate_t0_subject1_alex_binding(prompt: str) -> None:
    """Require the caption-only T0 prompt to define the native Subject1 roster."""
    try:
        document = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        raise StructuredControlError("T0 caption prompt is not structured JSON") from None
    if type(document) is not dict:
        raise StructuredControlError("T0 caption prompt is not structured JSON")
    segments = document.get("segments")
    if type(segments) is not list or len(segments) != 1 or type(segments[0]) is not dict:
        raise StructuredControlError("T0 caption prompt must contain exactly one scene segment")
    actions = segments[0].get("contained_actions")
    if actions is not None and actions != []:
        raise StructuredControlError("T0 caption prompt cannot contain an action tail")

    subjects = document.get("subject_list")
    if type(subjects) is not list or len(subjects) != 1 or type(subjects[0]) is not dict:
        raise StructuredControlError("T0 caption prompt must define exactly one Subject1 roster entry")
    subject = subjects[0]
    if (
        subject.get("subject_id") != "<|Subject1|>"
        or subject.get("display_name") != "Alex"
        or not isinstance(subject.get("subject_attribute"), str)
        or not subject["subject_attribute"].strip()
    ):
        raise StructuredControlError(
            "T0 caption prompt must map the single Subject1 roster entry to Alex with an attribute"
        )


def _profile_sha256(path: Path, expected_sha256: str) -> dict[str, Any]:
    profile = base._manifest_document(path, expected_sha256, "T1 action profile")
    if profile.get("schema") != PROFILE_SCHEMA:
        raise StructuredControlError("T1 action profile schema mismatch")
    return profile


def _validate_profile_bindings(
    profile: dict[str, Any],
    profile_sha256: str,
    t1_qwen: dict[str, Any],
    t1_manifest_sha256: str,
    upstream_root: Path,
) -> dict[str, Any]:
    """Bind the authenticated profile to the exact T1 prompt/token artifact."""
    qwen = t1_qwen["qwen_source"]
    document = t1_qwen["document"]
    prompt_contract = qwen.get("prompt_contract")
    expected_prompt = prompt_contract.get("prompt") if type(prompt_contract) is dict else None
    expected_payload_sha256 = t1_qwen["payload_sha256"]
    if (
        profile.get("conditioning_manifest_sha256") != t1_manifest_sha256
        or profile.get("conditioning_payload_sha256") != expected_payload_sha256
        or profile.get("prompt") != expected_prompt
        or not isinstance(expected_prompt, str)
        or profile.get("prompt_sha256") != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
        or qwen.get("prompt_sha256") != profile.get("prompt_sha256")
        or qwen.get("valid_tokens") != profile.get("valid_text_tokens")
        or document.get("qwen_source") != qwen
    ):
        raise StructuredControlError("T1 action profile does not bind the supplied Qwen prompt bytes")

    try:
        prompt_action = action_conditioning.validate_t1_prompt(expected_prompt)
    except (action_conditioning.ActionConditioningError, UnicodeEncodeError) as error:
        raise StructuredControlError(f"bound T1 prompt does not contain the admitted action: {error}") from None

    expected_source_hashes = action_conditioning.UPSTREAM_SOURCE_SHA256
    source = profile.get("source")
    if (
        type(source) is not dict
        or source.get("upstream_revision") != action_conditioning.UPSTREAM_REVISION
        or source.get("source_sha256") != expected_source_hashes
        or source.get("time_origin") != "segment_local_seconds"
        or source.get("resolve_subject_ids") is not False
        or source.get("action_wrapper_representation") != "literal_text_markup_not_token_ids"
    ):
        raise StructuredControlError("T1 action profile source lineage differs from the pinned exporter")
    for relative, expected_sha in expected_source_hashes.items():
        source_path = upstream_root / relative
        if source_path.is_symlink() or not source_path.is_file() or base._sha256(source_path) != expected_sha:
            raise StructuredControlError(f"profile shaping/tokenizer source differs from pinned bytes: {relative}")

    geometry = profile.get("geometry")
    if geometry != {
        "run_segments": 2,
        "segment_index": 1,
        "segment_start_seconds": 5.0,
        "segment_end_seconds": 10.0,
        "latent_frames": base.LATENTS_PER_SEGMENT,
        "tokens_per_frame": 144,
        "video_tokens": base.VIDEO_TOKENS,
        "text_length": base.QWEN_ROWS,
    }:
        raise StructuredControlError("T1 action profile segment geometry is not the admitted 512x288 layout")

    tokens = profile.get("token_contract")
    if (
        type(tokens) is not dict
        or tokens.get("template_name") != action_conditioning.TEMPLATE_NAME
        or tokens.get("template_sha256") != PROFILE_TEMPLATE_SHA256
        or tokens.get("text_length") != base.QWEN_ROWS
        or tokens.get("valid_text_tokens") != profile.get("valid_text_tokens")
        or tokens.get("prompt_sha256") != profile.get("prompt_sha256")
    ):
        raise StructuredControlError("T1 action profile token contract is incomplete or inconsistent")

    valid = profile.get("valid_text_tokens")
    offsets = profile.get("text_offsets")
    source_offsets = offsets.get("source_offsets_raw") if type(offsets) is dict else None
    effective_offsets = offsets.get("effective_offsets") if type(offsets) is dict else None
    if (
        type(valid) is not int or not 0 < valid <= action_conditioning.MAX_VALID_TEXT_TOKENS
        or type(offsets) is not dict
        or offsets.get("head_end") != offsets.get("action_start")
        or type(offsets.get("head_end")) is not int
        or not 0 < offsets["head_end"] < valid
        or offsets.get("action_end") != valid
        or offsets.get("valid_text_tokens") != valid
        or type(source_offsets) is not list or len(source_offsets) != 2
        or any(type(value) is not int or not 0 <= value <= base.QWEN_ROWS for value in source_offsets)
        or effective_offsets != [offsets.get("head_end"), valid]
        or effective_offsets != [min(value, valid) for value in source_offsets]
        or offsets.get("offset_method_endpoint_note") != (
            "BasePipeline returns the raw attention-mask sum; pinned nomas_cpt clamps it "
            "with min(offset, valid current text query tokens), yielding the authenticated cropped-mask count"
        )
    ):
        raise StructuredControlError("T1 action text offsets do not define one visible head/action split")

    action = profile.get("action")
    description = action.get("action_description") if type(action) is dict else None
    if (
        type(action) is not dict
        or action.get("action_id") != prompt_action["action_id"]
        or action.get("subject_id") != prompt_action["subject_id"]
        or action.get("time_origin") != prompt_action["time_origin"]
        or action.get("source_segment_start_seconds") != prompt_action["source_segment_start_seconds"]
        or action.get("source_segment_end_seconds") != prompt_action["source_segment_end_seconds"]
        or action.get("start_time_local_seconds") != prompt_action["start_time_local_seconds"]
        or action.get("end_time_local_seconds") != prompt_action["end_time_local_seconds"]
        or action.get("start_time_local_seconds") != ACTION_LOCAL_START
        or action.get("end_time_local_seconds") != ACTION_LOCAL_END
        or action.get("latent_start") != 3
        or action.get("latent_end") != 27
        or description != prompt_action["action_description"]
        or not isinstance(description, str)
        or "<|Subject1|>" not in description
        or any(term not in description.lower() for term in ACTION_DESCRIPTION_TERMS)
    ):
        raise StructuredControlError("T1 action profile metadata differs from the bound prompt action")
    windows = profile.get("video_windows")
    if (
        profile.get("intra_segment_active") is not True
        or windows != ACTION_VIDEO_WINDOWS
        or profile.get("labels") != {
            "global": 0,
            "action": 1,
            "padding": -1,
            "video_gap_window_index": -1,
            "query_scope": "global_head_and_matching_action_window",
            "key_scope": "global_history_or_matching_action",
        }
    ):
        raise StructuredControlError("T1 action profile windows or structured label contract changed")

    # The profile exporter runs on a31; these shaping/tokenizer sources are
    # byte-identical in the continuation control's pinned 911 checkout.
    return {
        "profile_sha256": profile_sha256,
        "manifest_sha256": t1_manifest_sha256,
        "payload_sha256": expected_payload_sha256,
        "prompt_sha256": profile["prompt_sha256"],
        "valid_text_tokens": valid,
        "head_end": offsets["head_end"],
        "action_end": valid,
        "action_id": 1,
        "tokens_per_frame": geometry["tokens_per_frame"],
        "video_windows": [list(window) for window in windows],
        "source_revision": source["upstream_revision"],
        "source_sha256": dict(source["source_sha256"]),
        "source_range_offsets": [offsets["head_end"], valid],
    }


def _build_pinned_range_builder(upstream: Path, torch: Any) -> tuple[Any, str, int, int]:
    source_path = upstream / _SOURCE_RANGE_RELATIVE
    expected_sha = base.block_reference.PINNED_SOURCE_SHA256[_SOURCE_RANGE_RELATIVE]
    actual_sha = base._sha256(source_path)
    if actual_sha != expected_sha:
        raise StructuredControlError("pinned intra-segment range-builder source digest mismatch")
    try:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise StructuredControlError(f"pinned range-builder source cannot be parsed: {error}") from None
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)
         and node.name == _SOURCE_RANGE_FUNCTION),
        None,
    )
    if function is None or function.decorator_list:
        raise StructuredControlError("pinned source range builder is missing or decorated")
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "torch": torch,
        "List": typing.List,
        "Optional": typing.Optional,
        "_intra_seg_range_cache": {},
    }
    try:
        exec(compile(isolated, str(source_path), "exec"), namespace)
    except Exception as error:
        raise StructuredControlError(f"could not isolate pinned range builder: {error}") from None
    return namespace[_SOURCE_RANGE_FUNCTION], actual_sha, function.lineno, function.end_lineno


def _source_mask_chunks_cpu(
    torch: Any,
    range_builder: Any,
    *,
    history_rows: int,
    image_rows: int,
    valid_text_rows: int,
    offsets: list[int],
    video_windows: list[list[int]],
    query_chunk: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Apply upstream ranges to compact Q/K coordinates and split for SDPA."""
    if min(history_rows, image_rows, valid_text_rows) < 1 or query_chunk < 1:
        raise StructuredControlError("source range mask geometry must be positive")
    query_rows = image_rows + valid_text_rows
    key_rows = history_rows + query_rows
    ranges_q, ranges_k, type_map = range_builder(
        hvis=history_rows,
        limg=image_rows,
        valid=valid_text_rows,
        offs=list(offsets),
        windows=[list(window) for window in video_windows],
        device=torch.device("cpu"),
    )
    if (
        ranges_q.device.type != "cpu" or ranges_k.device.type != "cpu"
        or ranges_q.dtype != torch.int32 or ranges_k.dtype != torch.int32
        or ranges_q.ndim != 2 or ranges_q.shape[1] != 2
        or ranges_k.shape != ranges_q.shape
        or type_map.dtype != torch.int32 or type_map.numel() != ranges_q.shape[0]
    ):
        raise StructuredControlError("pinned source range builder returned unexpected range tensors")
    q_values = ranges_q.tolist()
    k_values = ranges_k.tolist()
    chunks: list[Any] = []
    for start in range(0, query_rows, query_chunk):
        stop = min(start + query_chunk, query_rows)
        mask = torch.zeros((stop - start, key_rows), dtype=torch.bool, device="cpu")
        for (query_start, query_stop), (key_start, key_stop) in zip(q_values, k_values):
            if not (0 <= query_start <= query_stop <= query_rows
                    and 0 <= key_start <= key_stop <= key_rows):
                raise StructuredControlError("pinned source range escaped compact Q/K extent")
            overlap_start = max(start, query_start)
            overlap_stop = min(stop, query_stop)
            if overlap_stop > overlap_start and key_stop > key_start:
                mask[overlap_start - start:overlap_stop - start, key_start:key_stop] = True
        if not bool(mask.any(dim=1).all()):
            raise StructuredControlError("pinned source ranges leave an active query without visible keys")
        chunks.append(mask)
    return chunks, {
        "query_rows": query_rows,
        "key_rows": key_rows,
        "query_chunk": query_chunk,
        "range_count": len(q_values),
        "q_ranges": q_values,
        "k_ranges": k_values,
        "attn_type_map": type_map.tolist(),
    }


def _action_cache_attention_dispatch_factory(
    torch: Any, state: dict[str, Any], profile: dict[str, Any], upstream: Path,
    *, legacy_factory: Any,
) -> Any:
    """Keep the base T0 path; route only T1 reads through source action ranges."""
    legacy_dispatch = legacy_factory(torch, state)
    range_builder, source_sha, source_line_start, source_line_end = _build_pinned_range_builder(
        upstream, torch
    )
    windows = [
        [int(start) * profile["tokens_per_frame"],
         int(stop) * profile["tokens_per_frame"], int(action_index)]
        for start, stop, action_index in profile["video_windows"]
    ]
    mask_chunks_cpu, range_record = _source_mask_chunks_cpu(
        torch,
        range_builder,
        history_rows=base.VIDEO_TOKENS + state["t0_valid_tokens"],
        image_rows=base.VIDEO_TOKENS,
        valid_text_rows=profile["valid_text_tokens"],
        offsets=profile["source_range_offsets"],
        video_windows=windows,
        query_chunk=base.QUERY_CHUNK,
    )
    receipt = state["receipt"]
    receipt["source_action_range_builder_invocations"] = 1
    receipt["source_action_range_count"] = range_record["range_count"]
    receipt["source_action_range_builder_lines"] = [source_line_start, source_line_end]
    receipt["source_action_range_builder_sha256"] = source_sha
    receipt["t0_unmasked_cache_attention_calls"] = 0
    receipt["t1_action_masked_cache_attention_calls"] = 0
    device_masks: list[Any] | None = None

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
        nonlocal device_masks
        call = state.get("call")
        if call is None:
            raise StructuredControlError("cache attention ran outside the admitted forward schedule")
        if call["condition"] == "t0_caption":
            receipt["t0_unmasked_cache_attention_calls"] += 1
            return legacy_dispatch(
                q_full, full_k, full_v, q_mask, kv_mask,
                cache_valid_len, update_cache, cache_txt,
            )
        if call["condition"] != "t1_action":
            raise StructuredControlError("cache attention has an unadmitted text condition")
        if (
            state.get("attention_layout") != "compact"
            or cache_valid_len != call["cache_valid_len"]
            or update_cache is not False
            or cache_txt is not True
            or cache_valid_len != base.CACHE_CAPACITY
        ):
            raise StructuredControlError("T1 action attention escaped the no-write continuation schedule")

        expected_k = cache_valid_len + base.VIDEO_TOKENS + base.DIT_TEXT_ROWS
        expected_q_shape = (1, base.VIDEO_TOKENS + base.DIT_TEXT_ROWS,
                            base.HEADS, base.HEAD_DIM)
        if (
            tuple(q_full.shape) != expected_q_shape
            or tuple(full_k.shape) != (1, expected_k, base.HEADS, base.HEAD_DIM)
            or tuple(full_v.shape) != tuple(full_k.shape)
            or tuple(q_mask.shape) != (1, expected_q_shape[1])
            or tuple(kv_mask.shape) != (1, expected_k)
            or q_mask.device != q_full.device or kv_mask.device != q_full.device
            or q_mask.dtype != torch.bool or kv_mask.dtype != torch.bool
            or q_full.device.type != "mps"
            or any(value.device != q_full.device or value.dtype != state["dtype"]
                   for value in (full_k, full_v, q_full))
        ):
            raise StructuredControlError("T1 action Q/K/V tensors escaped admitted MPS BF16 geometry")

        valid_text = profile["valid_text_tokens"]
        if call["valid_text_tokens"] != valid_text:
            raise StructuredControlError("T1 action profile token count differs from the active forward")
        active_queries = base.VIDEO_TOKENS + valid_text
        q = q_full[:, :active_queries].permute(0, 2, 1, 3).contiguous()

        visible_history_rows = base.CACHE_CAPACITY - (
            base.DIT_TEXT_ROWS - state["t0_valid_tokens"]
        )
        key_parts_k = [full_k[:, :visible_history_rows]]
        key_parts_v = [full_v[:, :visible_history_rows]]
        image_start = base.CACHE_CAPACITY
        key_parts_k.append(full_k[:, image_start:image_start + base.VIDEO_TOKENS])
        key_parts_v.append(full_v[:, image_start:image_start + base.VIDEO_TOKENS])
        text_start = image_start + base.VIDEO_TOKENS
        key_parts_k.append(full_k[:, text_start:text_start + valid_text])
        key_parts_v.append(full_v[:, text_start:text_start + valid_text])
        compact_k = torch.cat(key_parts_k, dim=1)
        compact_v = torch.cat(key_parts_v, dim=1)
        expected_compact_rows = visible_history_rows + base.VIDEO_TOKENS + valid_text
        if tuple(compact_k.shape) != (1, expected_compact_rows, base.HEADS, base.HEAD_DIM):
            raise StructuredControlError("T1 action compact K/V rows differ from the authenticated history")
        if device_masks is None:
            device_masks = [mask.to(device=q_full.device) for mask in mask_chunks_cpu]

        keys = compact_k.permute(0, 2, 1, 3).contiguous()
        values = compact_v.permute(0, 2, 1, 3).contiguous()
        result = torch.zeros(
            expected_q_shape, dtype=state["dtype"], device=q_full.device,
        )
        for chunk_index, start in enumerate(range(0, active_queries, base.QUERY_CHUNK)):
            stop = min(start + base.QUERY_CHUNK, active_queries)
            if chunk_index >= len(device_masks) or device_masks[chunk_index].shape[0] != stop - start:
                raise StructuredControlError("T1 action SDPA chunk inventory differs from source ranges")
            attended = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, start:stop],
                keys,
                values,
                attn_mask=device_masks[chunk_index][None, None, :, :],
                dropout_p=0.0,
                is_causal=False,
            )
            result[:, start:stop].copy_(attended.permute(0, 2, 1, 3))

        receipt["attention_calls"] += 1
        receipt["attention_query_chunks"] += len(device_masks)
        row_key = f"{expected_k}->{expected_compact_rows}"
        inventory = receipt.setdefault("attention_row_inventory", {})
        inventory[row_key] = inventory.get(row_key, 0) + 1
        receipt["t1_action_masked_cache_attention_calls"] += 1
        return result

    return dispatch


@contextmanager
def _bind_dynamic_base_contract(action_valid_tokens: int, dit_text_rows: int) -> Iterator[None]:
    saved = {
        "ACTION_VALID_TOKENS": base.ACTION_VALID_TOKENS,
        "DIT_TEXT_ROWS": base.DIT_TEXT_ROWS,
        "CACHE_CAPACITY": base.CACHE_CAPACITY,
    }
    try:
        base.ACTION_VALID_TOKENS = action_valid_tokens
        base.DIT_TEXT_ROWS = dit_text_rows
        base.CACHE_CAPACITY = base.VIDEO_TOKENS + dit_text_rows
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioning-manifest", type=Path, required=True)
    parser.add_argument("--conditioning-manifest-sha256", required=True)
    parser.add_argument("--conditioning-payload-sha256", required=True)
    parser.add_argument("--t1-qwen-inputs", type=Path, required=True)
    parser.add_argument("--t1-qwen-inputs-sha256", required=True)
    parser.add_argument("--action-profile", type=Path, required=True)
    parser.add_argument("--action-profile-sha256", required=True)
    parser.add_argument("--t0-latent-bank", type=Path, default=base.DEFAULT_T0_LATENT_BANK)
    parser.add_argument("--t0-latent-bank-sha256", default=base.T0_LATENT_BANK_SHA256)
    parser.add_argument("--t0-frame-start", type=int, default=0)
    parser.add_argument("--t0-frame-count", type=int, default=base.LATENTS_PER_SEGMENT)
    parser.add_argument("--cache-index", type=Path, default=base.single.DEFAULT_CACHE_INDEX)
    parser.add_argument("--cache-index-sha256", default=base.single.CACHE_INDEX_SHA256)
    parser.add_argument("--upstream", type=Path, default=base.single.DEFAULT_UPSTREAM)
    parser.add_argument("--planned-cap-gib", type=float, default=base.MAX_PLANNED_CAP_GIB)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    for value, label in (
        (args.conditioning_manifest_sha256, "--conditioning-manifest-sha256"),
        (args.conditioning_payload_sha256, "--conditioning-payload-sha256"),
        (args.t1_qwen_inputs_sha256, "--t1-qwen-inputs-sha256"),
        (args.action_profile_sha256, "--action-profile-sha256"),
        (args.t0_latent_bank_sha256, "--t0-latent-bank-sha256"),
        (args.cache_index_sha256, "--cache-index-sha256"),
    ):
        if not isinstance(value, str) or base._HEX.fullmatch(value) is None:
            parser.error(f"{label} must be a lowercase SHA-256 digest")
    if args.t0_latent_bank_sha256 != base.T0_LATENT_BANK_SHA256:
        parser.error("clean T0 latent SHA differs from the explicitly pinned reference")
    if args.cache_index_sha256 != base.single.CACHE_INDEX_SHA256:
        parser.error("cache-index SHA differs from the pinned BF16 weights index")
    if not math.isfinite(args.planned_cap_gib) or not 0 < args.planned_cap_gib <= base.MAX_PLANNED_CAP_GIB:
        parser.error(f"--planned-cap-gib must be in (0, {base.MAX_PLANNED_CAP_GIB}] GiB")
    if args.execute and args.output is None:
        parser.error("--output is required with --execute")
    if args.plan_only and args.output is not None:
        parser.error("--output is valid only with --execute")
    return args


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        conditioning_manifest=args.conditioning_manifest,
        conditioning_manifest_sha256=args.conditioning_manifest_sha256,
        conditioning_payload_sha256=args.conditioning_payload_sha256,
        t1_qwen_inputs=args.t1_qwen_inputs,
        t1_qwen_inputs_sha256=args.t1_qwen_inputs_sha256,
        t0_latent_bank=args.t0_latent_bank,
        t0_latent_bank_sha256=args.t0_latent_bank_sha256,
        t0_frame_start=args.t0_frame_start,
        t0_frame_count=args.t0_frame_count,
        cache_index=args.cache_index,
        cache_index_sha256=args.cache_index_sha256,
        upstream=args.upstream,
        attention_layout="compact",
        planned_cap_gib=args.planned_cap_gib,
        plan_only=args.plan_only,
        execute=args.execute,
        output=args.output,
    )


def _authenticate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], int]:
    profile = _profile_sha256(args.action_profile, args.action_profile_sha256)
    action_valid_tokens = profile.get("valid_text_tokens")
    if type(action_valid_tokens) is not int or not 0 < action_valid_tokens <= action_conditioning.MAX_VALID_TEXT_TOKENS:
        raise StructuredControlError("authenticated profile has invalid T1 action token count")
    base_args = _base_args(args)
    # The upstream continuation admission authenticates the new T1 Qwen
    # manifest/payload and all T0 inputs. Its action count is bound to profile.
    with _bind_dynamic_base_contract(action_valid_tokens, base.DIT_TEXT_ROWS):
        conditioning = base._authenticate_conditioning(base_args)
    action = _validate_profile_bindings(
        profile,
        args.action_profile_sha256,
        conditioning["t1_qwen"],
        args.t1_qwen_inputs_sha256,
        base._regular_dir(args.upstream, "pinned upstream checkout"),
    )
    _validate_t0_subject1_alex_binding(conditioning["conditioning"].get("prompt"))
    dit_text_rows = _round_dit_text_rows(
        conditioning["t0_valid_tokens"], action["valid_text_tokens"]
    )
    with _bind_dynamic_base_contract(action_valid_tokens, dit_text_rows):
        t0_latent = base._authenticate_t0_latent(base_args)
        model = base._authenticate_model_sources(base_args)
        plan = base._forward_plan(conditioning["t0_valid_tokens"], action_valid_tokens)
        base._validate_forward_plan(plan, conditioning["t0_valid_tokens"], action_valid_tokens)
        admitted = {**conditioning, **t0_latent, **model, "forward_plan": plan}
    admitted["action_profile"] = action
    admitted["action_profile_path"] = base._regular_file(args.action_profile, "T1 action profile")
    admitted["dit_text_rows"] = dit_text_rows
    return admitted, base_args, action_valid_tokens


def _augment_manifest(
    document: dict[str, Any], admitted: dict[str, Any], args: argparse.Namespace,
) -> dict[str, Any]:
    document["schema"] = SCHEMA
    document["scope"] = (
        "512x288 T0 caption cache writer plus four source-BF16 T1 Alex head-turn steps "
        "with upstream intra-segment action ranges"
    )
    document["source"]["t1_action_profile"] = str(admitted["action_profile_path"])
    document["source"]["t1_action_profile_sha256"] = admitted["action_profile"]["profile_sha256"]
    document["source"]["t1_action_profile_source_revision"] = admitted["action_profile"]["source_revision"]
    document["source"]["t1_action_profile_source_sha256"] = admitted["action_profile"]["source_sha256"]
    document["configuration"]["dit_text_rows"] = admitted["dit_text_rows"]
    document["configuration"]["attention_adapter"]["action_mask"] = (
        "pinned _build_intra_seg_cache_ranges union applied as compact boolean Q/K mask"
    )
    document["configuration"]["attention_adapter"]["historical_action_tails"] = False
    document["execution"]["action_masking"] = {
        "t0_caption_cache_writer": "unmasked global attention",
        "t1_action_denoise": "upstream source ranges, history globally visible, one action window, global gaps",
        "profile_valid_text_tokens": admitted["action_profile"]["valid_text_tokens"],
        "head_end": admitted["action_profile"]["head_end"],
        "action_id": admitted["action_profile"]["action_id"],
        "video_windows_latent": admitted["action_profile"]["video_windows"],
        "video_windows_tokens": [
            [int(start) * admitted["action_profile"]["tokens_per_frame"],
             int(stop) * admitted["action_profile"]["tokens_per_frame"], int(action_id)]
            for start, stop, action_id in admitted["action_profile"]["video_windows"]
        ],
        "source_range_builder_invocations": document["execution"].get(
            "source_action_range_builder_invocations", 0
        ),
        "source_range_masked_attention_calls": document["execution"].get(
            "t1_action_masked_cache_attention_calls", 0
        ),
        "unmasked_t0_attention_calls": document["execution"].get(
            "t0_unmasked_cache_attention_calls", 0
        ),
        "source_range_builder_sha256": document["execution"].get(
            "source_action_range_builder_sha256"
        ),
        "source_range_builder_lines": document["execution"].get(
            "source_action_range_builder_lines"
        ),
    }
    document["limitations"].append(
        "The source action ranges use an explicit MPS SDPA adapter; Magi/Flash parity is not claimed."
    )
    return document


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        admitted, base_args, action_valid_tokens = _authenticate(args)
        if args.plan_only:
            print(json.dumps({
                "schema": SCHEMA,
                "status": "validated_plan_only",
                "model_launched": False,
                "mps_device_used": False,
                "conditioning_manifest_sha256": args.conditioning_manifest_sha256,
                "conditioning_payload_sha256": admitted["payload_sha256"],
                "t1_qwen_manifest_sha256": admitted["t1_qwen"]["manifest_sha256"],
                "t1_qwen_payload_sha256": admitted["t1_qwen"]["payload_sha256"],
                "action_profile_sha256": admitted["action_profile"]["profile_sha256"],
                "action_profile_source_revision": admitted["action_profile"]["source_revision"],
                "t0_caption_valid_tokens": admitted["t0_valid_tokens"],
                "t1_action_valid_tokens": action_valid_tokens,
                "dit_text_rows": admitted["dit_text_rows"],
                "t0_latent_bank_sha256": admitted["sha256"],
                "t0_frame_selection": admitted["frame_selection"],
                "upstream_revision": admitted["upstream_revision"],
                "primary_source_sha256": admitted["source_hashes"],
                "forward_plan": admitted["forward_plan"],
                "video_windows_latent": admitted["action_profile"]["video_windows"],
                "planned_cap_gib_is_metadata_only": True,
            }, sort_keys=True))
            return 0

        base_args.output = base._validate_output_target(args.output)
        base.single._configure_mps_runtime()
        torch, scheduler, diffusers_version = base.single._prepare_cpu_scheduler(
            Path(admitted["upstream_path"]), "source-bf16"
        )
        original_factory = base._cache_attention_dispatch_factory
        action_receipt: dict[str, Any] = {}

        def make_dispatch(runtime_torch: Any, state: dict[str, Any]) -> Any:
            dispatch = _action_cache_attention_dispatch_factory(
                runtime_torch, state, admitted["action_profile"], Path(admitted["upstream_path"]),
                legacy_factory=original_factory,
            )
            action_receipt["receipt"] = state["receipt"]
            return dispatch

        try:
            base._cache_attention_dispatch_factory = make_dispatch
            with _bind_dynamic_base_contract(action_valid_tokens, admitted["dit_text_rows"]):
                with torch.inference_mode():
                    summary, payload = base._run_control(
                        admitted, base_args, torch, scheduler, diffusers_version
                    )
        finally:
            base._cache_attention_dispatch_factory = original_factory
        receipt = action_receipt.get("receipt")
        if type(receipt) is not dict:
            raise StructuredControlError("pinned cache function did not create the action receipt")
        for name in (
            "source_action_range_builder_invocations",
            "source_action_range_count",
            "source_action_range_builder_lines",
            "source_action_range_builder_sha256",
            "t0_unmasked_cache_attention_calls",
            "t1_action_masked_cache_attention_calls",
        ):
            if name not in receipt:
                raise StructuredControlError(f"cache action receipt omits {name}")
            summary[name] = receipt[name]
        expected_action_receipt = {
            "source_action_range_builder_invocations": 1,
            "t0_unmasked_cache_attention_calls": 54,
            "t1_action_masked_cache_attention_calls": 4 * 54,
        }
        if any(receipt.get(name) != expected for name, expected in expected_action_receipt.items()):
            raise StructuredControlError("cache action receipt differs from the 54-block five-forward schedule")
        if type(receipt.get("source_action_range_count")) is not int or receipt["source_action_range_count"] < 1:
            raise StructuredControlError("pinned source range builder returned no action ranges")

        with _bind_dynamic_base_contract(action_valid_tokens, admitted["dit_text_rows"]):
            document = _augment_manifest(
                base._manifest(admitted, base_args, summary, payload), admitted, args
            )
        payload_path, manifest_sha = base.single._publish(base_args.output, document, payload)
        print(json.dumps({
            "status": "completed",
            "manifest": str(base_args.output),
            "payload": str(payload_path),
            "manifest_sha256": manifest_sha,
            "payload_sha256": document["result"]["payload_sha256"],
            "model_forwards": summary["model_forwards"],
            "block_invocations": summary["block_invocations"],
            "terminal_export_count": summary["terminal_export_count"],
            "t1_action_masked_cache_attention_calls": summary["t1_action_masked_cache_attention_calls"],
        }, sort_keys=True))
        return 0
    except (
        StructuredControlError,
        base.ControlError,
        base.single.ControlError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"structured reference-continuation control rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
