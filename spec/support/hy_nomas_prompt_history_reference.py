#!/usr/bin/env python3
"""Compare variable-prompt native history against pinned upstream attention.

The native fixture is a synthetic F32 Q/K/V bank. This runs the pinned
``causal_attention_with_cache_mp`` AST on CPU, with no model weights or device
selection, then checks valid query rows and the final cache at 2e-5 atol/rtol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from hy_nomas_continuation_reference import ReferenceError, load_upstream_attention


NATIVE_REPORT_SCHEMA = "cogni_ml.hy_nomas_prompt_history_probe"
REFERENCE_SCHEMA = "cogni_ml.hy_nomas_prompt_history_reference"
SEGMENT_TEXT_TOKENS = (1, 2, 1)
SEGMENT_CACHE_TEXT = (True, True, True)
SEGMENT_COUNT = 3
LAYERS = 54
VIDEO_TOKENS = 2
MAX_TEXT_TOKENS = 2
QUERY_ROWS = VIDEO_TOKENS + MAX_TEXT_TOKENS
STORAGE_ROWS = QUERY_ROWS
HEADS = 2
HEAD_DIM = 2
FORWARD_COUNT = 14
MAX_CACHE_ROWS = sum(VIDEO_TOKENS + n for n in SEGMENT_TEXT_TOKENS[:-1])
ATOL = RTOL = 2e-5
ARTIFACT_SHAPES = {
    "query": (FORWARD_COUNT, LAYERS, STORAGE_ROWS, HEADS, HEAD_DIM),
    "key": (FORWARD_COUNT, LAYERS, STORAGE_ROWS, HEADS, HEAD_DIM),
    "value": (FORWARD_COUNT, LAYERS, STORAGE_ROWS, HEADS, HEAD_DIM),
    "output": (FORWARD_COUNT, LAYERS, QUERY_ROWS, HEADS, HEAD_DIM),
    "cache_key": (LAYERS, MAX_CACHE_ROWS, HEADS, HEAD_DIM),
    "cache_value": (LAYERS, MAX_CACHE_ROWS, HEADS, HEAD_DIM),
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_report(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ReferenceError("native report must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ReferenceError("native report exceeds the metadata limit")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceError("cannot parse native report JSON") from error
    if type(document) is not dict or document.get("schema") != NATIVE_REPORT_SCHEMA:
        raise ReferenceError("unsupported native prompt-history report")
    if document.get("schema_version") != 1 or document.get("passed") is not True:
        raise ReferenceError("native prompt-history report did not pass its probe")
    expected_fields = {
        "segments": SEGMENT_COUNT,
        "layers": LAYERS,
        "denoise_forwards_per_segment": 4,
        "writer_forwards": 2,
        "forward_count": FORWARD_COUNT,
        "video_tokens": VIDEO_TOKENS,
        "valid_text_tokens": MAX_TEXT_TOKENS,
        "storage_text_tokens": MAX_TEXT_TOKENS,
        "active_query_rows": QUERY_ROWS,
        "storage_rows": STORAGE_ROWS,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "final_cache_rows": MAX_CACHE_ROWS,
    }
    for field, expected in expected_fields.items():
        if type(document.get(field)) is not int or document[field] != expected:
            raise ReferenceError(f"native report {field} differs from the admitted geometry")
    if document.get("segment_text_tokens") != list(SEGMENT_TEXT_TOKENS):
        raise ReferenceError("native per-segment text token counts differ")
    if document.get("segment_cache_text") != list(SEGMENT_CACHE_TEXT):
        raise ReferenceError("native per-segment cache flags differ")
    if document.get("encode_buffer_allocations") != 0 or document.get("encode_readbacks") != 0:
        raise ReferenceError("native encode performed a buffer allocation or readback")
    if document.get("preterminal_waits") != 0 or document.get("terminal_wait_count") != 1:
        raise ReferenceError("native probe did not use exactly one terminal wait")
    if document.get("postterminal_readbacks") != 7:
        raise ReferenceError("native diagnostic readback count differs")

    artifacts = document.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != set(ARTIFACT_SHAPES):
        raise ReferenceError("native artifact inventory is incomplete")
    parent = path.parent.resolve(strict=True)
    resolved = {}
    for name, shape in ARTIFACT_SHAPES.items():
        record = artifacts[name]
        if type(record) is not dict or set(record) != {"path", "sha256", "shape"}:
            raise ReferenceError(f"{name} artifact metadata is malformed")
        if (type(record.get("path")) is not str or
                type(record.get("sha256")) is not str or
                type(record.get("shape")) is not list or
                any(type(dimension) is not int for dimension in record["shape"]) or
                record["shape"] != list(shape)):
            raise ReferenceError(f"{name} artifact shape differs from the admitted geometry")
        relative = Path(record["path"])
        if relative.is_absolute() or relative.name != record["path"]:
            raise ReferenceError(f"{name} artifact path must be a sibling basename")
        artifact = parent / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise ReferenceError(f"{name} artifact must be a regular non-symlink file")
        artifact = artifact.resolve(strict=True)
        if artifact.parent != parent:
            raise ReferenceError(f"{name} artifact escapes the report directory")
        payload = artifact.read_bytes()
        if len(payload) != math.prod(shape) * 4 or _sha256(payload) != record["sha256"]:
            raise ReferenceError(f"{name} artifact extent or digest mismatch")
        resolved[name] = torch.frombuffer(bytearray(payload), dtype=torch.float32).view(shape)
    if sys.byteorder != "little":
        raise ReferenceError("native little-endian F32 artifacts require a little-endian host")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in resolved.values()):
        raise ReferenceError("native artifacts contain non-finite values")
    return {"document": document, "tensors": resolved}


def _forward_index(segment: int, denoise: int) -> int:
    return segment * 5 + denoise


def run_reference(tensors: dict, attention) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run four denoise calls and an optional writer per prompt, per layer."""
    device = torch.device("cpu")
    query, key, value = (tensors[name].to(device=device, dtype=torch.float32)
                         for name in ("query", "key", "value"))
    outputs = torch.zeros_like(tensors["output"])
    cache_key = torch.zeros((LAYERS, MAX_CACHE_ROWS, HEADS, HEAD_DIM), dtype=torch.float32)
    cache_value = torch.zeros_like(cache_key)
    torch.set_num_threads(1)

    with torch.no_grad():
        for layer in range(LAYERS):
            cache = {
                "k": torch.zeros((1, MAX_CACHE_ROWS, HEADS, HEAD_DIM), dtype=torch.float32),
                "v": torch.zeros((1, MAX_CACHE_ROWS, HEADS, HEAD_DIM), dtype=torch.float32),
            }
            valid_len = 0
            for segment, text_tokens in enumerate(SEGMENT_TEXT_TOKENS):
                active_rows = VIDEO_TOKENS + text_tokens
                text_mask = torch.ones((1, text_tokens), dtype=torch.bool)
                for denoise in range(4):
                    forward = _forward_index(segment, denoise)
                    q = query[forward, layer, :active_rows].unsqueeze(0)
                    k = key[forward, layer, :active_rows].unsqueeze(0)
                    v = value[forward, layer, :active_rows].unsqueeze(0)
                    img_out, txt_out = attention(
                        (q[:, :VIDEO_TOKENS], q[:, VIDEO_TOKENS:]),
                        (k[:, :VIDEO_TOKENS], k[:, VIDEO_TOKENS:]),
                        (v[:, :VIDEO_TOKENS], v[:, VIDEO_TOKENS:]),
                        kv_cache=cache,
                        cache_valid_len=valid_len,
                        attn_mode="torch",
                        text_mask=text_mask,
                        update_cache=False,
                        cache_txt=True,
                        seg_ar_attention_backend="magi",
                    )
                    outputs[forward, layer, :active_rows] = torch.cat(
                        (img_out.reshape(1, VIDEO_TOKENS, HEADS, HEAD_DIM),
                         txt_out.reshape(1, text_tokens, HEADS, HEAD_DIM)), dim=1
                    )[0]

                if segment < SEGMENT_COUNT - 1:
                    forward = _forward_index(segment, 4)
                    q = query[forward, layer, :active_rows].unsqueeze(0)
                    k = key[forward, layer, :active_rows].unsqueeze(0)
                    v = value[forward, layer, :active_rows].unsqueeze(0)
                    img_out, txt_out = attention(
                        (q[:, :VIDEO_TOKENS], q[:, VIDEO_TOKENS:]),
                        (k[:, :VIDEO_TOKENS], k[:, VIDEO_TOKENS:]),
                        (v[:, :VIDEO_TOKENS], v[:, VIDEO_TOKENS:]),
                        kv_cache=cache,
                        cache_valid_len=valid_len,
                        attn_mode="torch",
                        text_mask=text_mask,
                        update_cache=True,
                        cache_txt=True,
                        seg_ar_attention_backend="magi",
                    )
                    outputs[forward, layer, :active_rows] = torch.cat(
                        (img_out.reshape(1, VIDEO_TOKENS, HEADS, HEAD_DIM),
                         txt_out.reshape(1, text_tokens, HEADS, HEAD_DIM)), dim=1
                    )[0]
                    valid_len += active_rows

            if valid_len != MAX_CACHE_ROWS:
                raise ReferenceError("pinned attention produced an unexpected cache prefix")
            cache_key[layer] = cache["k"][0, :valid_len]
            cache_value[layer] = cache["v"][0, :valid_len]
    return outputs, cache_key, cache_value


def _compare(reference: torch.Tensor, native: torch.Tensor, name: str) -> dict[str, object]:
    reference = reference.detach().to(device="cpu", dtype=torch.float32).contiguous()
    native = native.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if reference.shape != native.shape:
        raise ReferenceError(f"{name} reference/native shapes differ")
    if not bool(torch.isfinite(reference).all().item()) or not bool(torch.isfinite(native).all().item()):
        raise ReferenceError(f"{name} comparison includes non-finite values")
    delta = (reference - native).abs()
    allowed = ATOL + RTOL * reference.abs()
    failures = int((delta > allowed).sum().item())
    denominator = reference.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "name": name,
        "elements": reference.numel(),
        "failures": failures,
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "max_rel": float((delta / denominator).max().item()) if delta.numel() else 0.0,
        "atol": ATOL,
        "rtol": RTOL,
    }


def compare_report(report_path: Path) -> dict[str, object]:
    loaded = _read_report(report_path)
    tensors = loaded["tensors"]
    attention, upstream = load_upstream_attention()
    expected_output, expected_key, expected_value = run_reference(tensors, attention)
    native_output = tensors["output"]
    # Compare only the prompt's actual query rows. Padded query rows do not
    # exist upstream and cannot affect any other query's attention result.
    output_slices = []
    for segment, text_tokens in enumerate(SEGMENT_TEXT_TOKENS):
        rows = VIDEO_TOKENS + text_tokens
        forwards = range(_forward_index(segment, 0), _forward_index(segment, 4) + (segment < SEGMENT_COUNT - 1))
        for forward in forwards:
            output_slices.append((expected_output[forward, :, :rows], native_output[forward, :, :rows]))
    output_reference = torch.cat([pair[0].reshape(-1) for pair in output_slices])
    output_native = torch.cat([pair[1].reshape(-1) for pair in output_slices])
    comparisons = [
        _compare(output_reference, output_native, "active_query_outputs"),
        _compare(expected_key, tensors["cache_key"], "final_cache_key"),
        _compare(expected_value, tensors["cache_value"], "final_cache_value"),
    ]
    return {
        "schema": REFERENCE_SCHEMA,
        "schema_version": 1,
        "scope": "CPU execution of pinned upstream cache attention over the native synthetic Q/K/V bank",
        "device": "cpu",
        "upstream": upstream,
        "native_report_sha256": _sha256(report_path.read_bytes()),
        "reference_script_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "text_tokens_per_segment": list(SEGMENT_TEXT_TOKENS),
        "cache_text_per_segment": list(SEGMENT_CACHE_TEXT),
        "comparisons": comparisons,
        "passed": all(entry["failures"] == 0 for entry in comparisons),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-report", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare_report(args.native_report)
    except ReferenceError as error:
        raise SystemExit(f"reference rejected: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
