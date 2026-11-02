#!/usr/bin/env python3
"""Create a CPU F32 action-mask fixture from the pinned upstream range builder.

The fixture is synthetic and spans one 256-row query tile plus an active/padded tail. It extracts only
``_build_intra_seg_cache_ranges`` from the pinned source; it does not import the
model package, load weights, or select a GPU device. The masked result uses the
exact ranges returned by that function. A second result models its unmasked
legacy branch over the same compact visible rows.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F


UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
OLDER_COMPATIBLE_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_RELATIVE_SOURCE = "hymm/models/dit/nomas_cpt.py"
UPSTREAM_SOURCE_SHA256 = "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee"

# Synthetic geometry includes a 256-row tile, one active tail row, and two padded rows.
HISTORY_ROWS = 3
IMAGE_ROWS = 250
VALID_TEXT_ROWS = 7
PADDED_TEXT_ROWS = 2
QUERY_ROWS = IMAGE_ROWS + VALID_TEXT_ROWS + PADDED_TEXT_ROWS
CURRENT_ROWS = QUERY_ROWS
KEY_ROWS = HISTORY_ROWS + CURRENT_ROWS
HEADS = 2
HEAD_DIM = 4
OFFSETS = (2, 4, 7)
VIDEO_WINDOWS = ((0, 3, 0), (3, 4, -1), (4, 6, 1), (6, IMAGE_ROWS, -1))
ATOL = 2.0e-5


class ReferenceError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_range_builder(source: bytes) -> tuple[Callable[..., Any], int, int]:
    """Compile only the pinned upstream builder, with no package imports."""
    try:
        parsed = ast.parse(source.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ReferenceError(f"upstream source cannot be parsed: {error}") from None
    function = next(
        (
            node
            for node in parsed.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_build_intra_seg_cache_ranges"
        ),
        None,
    )
    if not isinstance(function, ast.FunctionDef):
        raise ReferenceError("pinned upstream range builder is absent or asynchronous")
    if function.decorator_list:
        raise ReferenceError("upstream range builder unexpectedly has decorators")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "torch": torch,
        "List": list,
        "Optional": __import__("typing").Optional,
        "Dict": dict,
        "Tuple": tuple,
        "_intra_seg_range_cache": {},
    }
    try:
        exec(compile(module, UPSTREAM_RELATIVE_SOURCE, "exec"), namespace)
    except Exception as error:
        raise ReferenceError(f"could not compile the isolated range builder: {error}") from None
    return namespace["_build_intra_seg_cache_ranges"], function.lineno, function.end_lineno


def validate_upstream_source(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ReferenceError("upstream source must be a regular non-symlink file")
    source = path.read_bytes()
    actual = sha256(source)
    if actual != UPSTREAM_SOURCE_SHA256:
        raise ReferenceError(
            "upstream nomas_cpt.py digest mismatch: "
            f"expected {UPSTREAM_SOURCE_SHA256}, got {actual}"
        )
    return source


def build_range_mask(
    q_ranges: torch.Tensor, k_ranges: torch.Tensor, query_rows: int, key_rows: int
) -> torch.Tensor:
    """Expand the upstream range rectangles as an exact bool Q-by-K mask."""
    if q_ranges.dtype != torch.int32 or k_ranges.dtype != torch.int32:
        raise ReferenceError("upstream range tensors must be int32")
    if q_ranges.ndim != 2 or q_ranges.shape[1] != 2 or q_ranges.shape != k_ranges.shape:
        raise ReferenceError("upstream q/k range arrays must be paired Nx2 tensors")
    mask = torch.zeros((query_rows, key_rows), dtype=torch.bool, device="cpu")
    for (q_start, q_end), (k_start, k_end) in zip(q_ranges.tolist(), k_ranges.tolist()):
        if not (0 <= q_start <= q_end <= query_rows and 0 <= k_start <= k_end <= key_rows):
            raise ReferenceError("upstream range exceeds the synthetic dense layout")
        mask[q_start:q_end, k_start:k_end] = True
    return mask


def build_label_mask(query_labels: list[int], key_labels: list[int]) -> torch.Tensor:
    """Native contract: global keys plus keys matching an active action query."""
    if any(label < -1 for label in query_labels + key_labels):
        raise ReferenceError("labels are restricted to -1, 0, or positive action IDs")
    q = torch.tensor(query_labels, dtype=torch.int32, device="cpu")
    k = torch.tensor(key_labels, dtype=torch.int32, device="cpu")
    active = q[:, None] >= 0
    allowed = (k[None, :] == 0) | ((q[:, None] > 0) & (k[None, :] == q[:, None]))
    return active & allowed


def attention_f32(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if query.device.type != "cpu" or key.device.type != "cpu" or value.device.type != "cpu":
        raise ReferenceError("attention oracle accepts CPU tensors only")
    if any(tensor.dtype != torch.float32 for tensor in (query, key, value)):
        raise ReferenceError("attention oracle accepts F32 tensors only")
    if mask.shape != (query.shape[1], key.shape[1]) or mask.dtype != torch.bool:
        raise ReferenceError("attention mask geometry or dtype mismatch")
    q_bhsd = query.transpose(1, 2).contiguous()
    k_bhsd = key.transpose(1, 2).contiguous()
    v_bhsd = value.transpose(1, 2).contiguous()
    active_rows = mask.any(dim=-1)
    output = F.scaled_dot_product_attention(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        attn_mask=mask[None, None, :, :],
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2).contiguous()
    # Upstream initializes hidden to zero and scatters only valid query rows.
    output = torch.where(active_rows[None, :, None, None], output, torch.zeros_like(output))
    return output.contiguous()


def deterministic_tensor(rows: int, salt: int) -> torch.Tensor:
    count = rows * HEADS * HEAD_DIM
    indices = torch.arange(count, dtype=torch.int64, device="cpu")
    numerator = torch.remainder(indices * 37 + salt * 997, 4093) - 2046
    return (numerator.to(torch.float32) / 4096.0).reshape(1, rows, HEADS, HEAD_DIM).contiguous()


def source_masks(
    source: bytes,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    list[int],
    list[int],
    torch.Tensor,
]:
    builder, line_start, line_end = extract_range_builder(source)
    q_ranges, k_ranges, type_map = builder(
        hvis=HISTORY_ROWS,
        limg=IMAGE_ROWS,
        valid=VALID_TEXT_ROWS,
        offs=list(OFFSETS),
        windows=[list(window) for window in VIDEO_WINDOWS],
        device=torch.device("cpu"),
    )
    masked = build_range_mask(q_ranges, k_ranges, QUERY_ROWS, KEY_ROWS)

    query_labels = [0] * QUERY_ROWS
    for start, end, action_index in VIDEO_WINDOWS:
        action_label = 0 if action_index < 0 else action_index + 1
        query_labels[start:end] = [action_label] * (end - start)
    query_labels[IMAGE_ROWS : IMAGE_ROWS + OFFSETS[0]] = [0] * OFFSETS[0]
    for action_index, (start, end) in enumerate(zip(OFFSETS[:-1], OFFSETS[1:])):
        query_labels[IMAGE_ROWS + start : IMAGE_ROWS + end] = [action_index + 1] * (end - start)
    query_labels[IMAGE_ROWS + VALID_TEXT_ROWS :] = [-1] * PADDED_TEXT_ROWS

    # History, all current video, and scene/head text are global. Action text
    # receives its positive ID. The two padded current-text keys are hidden.
    key_labels = [0] * (HISTORY_ROWS + IMAGE_ROWS + OFFSETS[0])
    for action_index, (start, end) in enumerate(zip(OFFSETS[:-1], OFFSETS[1:])):
        key_labels.extend([action_index + 1] * (end - start))
    key_labels.extend([-1] * PADDED_TEXT_ROWS)
    if len(query_labels) != QUERY_ROWS or len(key_labels) != KEY_ROWS:
        raise ReferenceError("label construction did not match the pinned geometry")
    label_mask = build_label_mask(query_labels, key_labels)
    if not torch.equal(masked, label_mask):
        mismatch = torch.nonzero(masked ^ label_mask, as_tuple=False)[0].tolist()
        raise ReferenceError(f"upstream ranges and native labels disagree at {mismatch}")

    # Upstream's offsets=None path supplies one range across compact visible Q/K.
    # The trailing text padding is omitted from both compact sequences and is
    # restored as zero output rows, matching _intra_seg_cache_forward.
    legacy = torch.zeros((QUERY_ROWS, KEY_ROWS), dtype=torch.bool, device="cpu")
    visible_queries = IMAGE_ROWS + VALID_TEXT_ROWS
    visible_keys = HISTORY_ROWS + IMAGE_ROWS + VALID_TEXT_ROWS
    legacy[:visible_queries, :visible_keys] = True
    if type_map.dtype != torch.int32 or type_map.numel() != q_ranges.shape[0]:
        raise ReferenceError("source range builder returned an unexpected attention-type map")
    return q_ranges, k_ranges, type_map, masked, line_start, line_end, query_labels, key_labels, legacy


@dataclass(frozen=True)
class FixtureData:
    query: torch.Tensor
    current_key: torch.Tensor
    current_value: torch.Tensor
    history_key: torch.Tensor
    history_value: torch.Tensor
    expected_masked: torch.Tensor
    expected_legacy: torch.Tensor
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    type_map: torch.Tensor
    query_labels: list[int]
    key_labels: list[int]
    legacy_mask: torch.Tensor
    masked_mask: torch.Tensor
    builder_line_start: int
    builder_line_end: int


def make_fixture(source: bytes) -> FixtureData:
    (
        q_ranges,
        k_ranges,
        type_map,
        masked_mask,
        line_start,
        line_end,
        query_labels,
        key_labels,
        legacy_mask,
    ) = source_masks(source)
    query = deterministic_tensor(QUERY_ROWS, 11)
    current_key = deterministic_tensor(CURRENT_ROWS, 23)
    current_value = deterministic_tensor(CURRENT_ROWS, 41)
    history_key = deterministic_tensor(HISTORY_ROWS, 59)
    history_value = deterministic_tensor(HISTORY_ROWS, 73)

    # High contrast sentinels make action cross-talk and padded-key leakage fail
    # by a wide margin while all legal queries retain ordinary global values.
    current_value[0, IMAGE_ROWS + OFFSETS[0] : IMAGE_ROWS + OFFSETS[1], :, :] = torch.tensor(
        [1536.0, -1024.0, 512.0, -256.0], dtype=torch.float32
    )
    current_value[0, IMAGE_ROWS + OFFSETS[1] : IMAGE_ROWS + OFFSETS[2], :, :] = torch.tensor(
        [-1536.0, 1024.0, -512.0, 256.0], dtype=torch.float32
    )
    current_value[0, IMAGE_ROWS + VALID_TEXT_ROWS :, :, :] = torch.tensor(
        [8192.0, -8192.0, 4096.0, -4096.0], dtype=torch.float32
    )

    keys = torch.cat((history_key, current_key), dim=1)
    values = torch.cat((history_value, current_value), dim=1)
    expected_masked = attention_f32(query, keys, values, masked_mask)
    expected_legacy = attention_f32(query, keys, values, legacy_mask)
    return FixtureData(
        query=query,
        current_key=current_key,
        current_value=current_value,
        history_key=history_key,
        history_value=history_value,
        expected_masked=expected_masked,
        expected_legacy=expected_legacy,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        type_map=type_map,
        query_labels=query_labels,
        key_labels=key_labels,
        legacy_mask=legacy_mask,
        masked_mask=masked_mask,
        builder_line_start=line_start,
        builder_line_end=line_end,
    )


def write_tensor(directory: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise ReferenceError(f"{name} must be a CPU F32 tensor")
    contiguous = tensor.detach().contiguous()
    data = contiguous.numpy().astype("<f4", copy=False).tobytes(order="C")
    filename = f"{name}.f32le"
    (directory / filename).write_bytes(data)
    return {
        "file": filename,
        "shape": list(contiguous.shape),
        "dtype": "F32",
        "byte_order": "little",
        "layout": "contiguous BTHD",
        "nbytes": len(data),
        "sha256": sha256(data),
    }


def write_fixture(output_dir: Path, upstream_source_path: Path) -> tuple[Path, str]:
    source = validate_upstream_source(upstream_source_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise ReferenceError(f"refusing to replace existing fixture directory: {output_dir}")
    data = make_fixture(source)
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=False)
    tensors = {
        "query": write_tensor(output_dir, "query", data.query),
        "current_key": write_tensor(output_dir, "current_key", data.current_key),
        "current_value": write_tensor(output_dir, "current_value", data.current_value),
        "history_key": write_tensor(output_dir, "history_key", data.history_key),
        "history_value": write_tensor(output_dir, "history_value", data.history_value),
        "expected_masked": write_tensor(output_dir, "expected_masked", data.expected_masked),
        "expected_legacy": write_tensor(output_dir, "expected_legacy", data.expected_legacy),
    }
    manifest = {
        "schema": "cogni_ml.hy_nomas_action_attention_reference",
        "schema_version": 1,
        "upstream": {
            "repository": "hy-nomas-reference-handoff-20261101/upstream-a31",
            "revision": UPSTREAM_REVISION,
            "byte_identical_older_revision": OLDER_COMPATIBLE_REVISION,
            "source_file": UPSTREAM_RELATIVE_SOURCE,
            "source_file_sha256": sha256(source),
            "range_builder": "_build_intra_seg_cache_ranges",
            "range_builder_lines": [data.builder_line_start, data.builder_line_end],
            "masked_route": "_intra_seg_cache_forward offsets branch; source ranges expanded to bool Q-by-K mask",
            "legacy_route": "_intra_seg_cache_forward offsets=None branch; one full range over compact visible Q/K",
        },
        "oracle": {
            "script_sha256": sha256(Path(__file__).resolve().read_bytes()),
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "torch.float32",
            "equation": "F.scaled_dot_product_attention(Q^T,K^T,V^T, dropout_p=0.0, is_causal=False), then transpose output to BTHD",
            "mask": "source builder range union; visible history/image/head global, action text has positive action id, padded text hidden; padded Q outputs zero",
            "label_rule": "active query sees global key label 0 and positive keys matching its own positive action label; label -1 query outputs zero and label -1 key is hidden",
            "atol": ATOL,
        },
        "geometry": {
            "batch": 1,
            "history_rows": HISTORY_ROWS,
            "image_rows": IMAGE_ROWS,
            "valid_text_rows": VALID_TEXT_ROWS,
            "padded_text_rows": PADDED_TEXT_ROWS,
            "query_rows": QUERY_ROWS,
            "current_rows": CURRENT_ROWS,
            "key_rows": KEY_ROWS,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "layout": "BTHD; history prefix followed by current image and text rows",
        },
        "mask": {
            "text_offsets": list(OFFSETS),
            "video_windows": [list(window) for window in VIDEO_WINDOWS],
            "q_ranges": data.q_ranges.tolist(),
            "k_ranges": data.k_ranges.tolist(),
            "attn_type_map": data.type_map.tolist(),
            "query_labels": data.query_labels,
            "key_labels": data.key_labels,
            "masked_rows": data.masked_mask.tolist(),
            "legacy_rows": data.legacy_mask.tolist(),
        },
        "tensors": tensors,
        "cases": [
            {"name": "source_range_masked", "expected": tensors["expected_masked"]},
            {"name": "source_unmasked_legacy", "expected": tensors["expected_legacy"]},
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    return manifest_path, sha256(manifest_bytes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--upstream-source", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest_path, manifest_sha = write_fixture(args.output_dir, args.upstream_source)
    except ReferenceError as error:
        raise SystemExit(str(error)) from None
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={manifest_sha}")
    print("oracle_device=cpu model_weights=none source_range_builder=isolated_AST")
    return 0


if __name__ == "__main__":
    sys.exit(main())
