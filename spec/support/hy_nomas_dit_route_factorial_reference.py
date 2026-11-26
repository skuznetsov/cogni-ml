#!/usr/bin/env python3
"""CPU admission and pinned-source oracle for the direct DiT route factorial.

The experiment starts from authenticated saved T0/T1 states. It replays the
pinned BF16 MPS DiT blocks, changing only video-query to matching action-text
edges after the accepted ROI mask. No encoder, refiner, scheduler update,
decoder, or full generation is part of this oracle.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

SCHEMA = "hy-nomas-dit-route-factorial-reference-v1"
SCHEMA_VERSION = 1
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
ROI_MANIFEST_SHA256 = "ee7a830eb287cffd5d4ee2297d1ce5bd99e7f26a3dcc36442358c12f6ba72dd8"
ROI_PAYLOAD_SHA256 = "f59abfea419c9f2fe5bddc6c8595665ce55a96038f431f0273f88475a9f6388c"
ROI_CAPTURE_SCRIPT_SHA256 = "1b7f1a09a1d71954cf93f1439e1b77fbc2aa1fa6f3c4a33be4174da76c7129da"
SOURCE_RANGE_BUILDER_SHA256 = "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee"
CONTINUATION_MANIFEST_SHA256 = "f23608e9b93bb1a2da0b90e9d48f9edf1506fb65bd2b2ae783d9c1440a1c8cec"
CONTINUATION_PAYLOAD_SHA256 = "b1e10540ccf3dbf58c62511f37d4f00e32b0dec4b60348ef0dff85fd584be8f9"
FROZEN_INPUTS_SHA256 = "71abfe81a63cc3042063444d76a494e960e704386408a81a45a16f280398c3cf"
ROPE_COS_SHA256 = "7ddb17895b9bcd14ed840a8b80eb399c0470ccf6933b0777903cda1536d0afb8"
ROPE_SIN_SHA256 = "272ac7251cb39de8a442c7221792e4bef8440c140cff639496a2b82f90c50306"
ACCEPTED_COMPARISON_SHA256 = "2bea5367da4dea43783243b6aefa0daa06e29eadf7879160e88eba23cbc51bd2"

DEFAULT_FROZEN_INPUTS = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-direct-routes-20261105.klzuhg0r/frozen-inputs.json")
DEFAULT_ROI_MANIFEST = Path("/opt/mldata_ssd/cogni-ml/rolewright-sequential-actions-20261105.w7m2k9qa/roi/source/source-v2-roi.json")
DEFAULT_CONTINUATION_MANIFEST = Path("/opt/mldata_ssd/cogni-ml/rolewright-t1-acting.ENJss8/dual/continuation.json")
DEFAULT_CACHE_INDEX = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/weights/q4-cache-v2/index.json")
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024")
SOURCE_REPLAY_SUPPORT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-final-euler-20261104.c7zj7lnM")
ROI_SOURCE_SCRIPT = Path("/opt/mldata_ssd/cogni-ml/rolewright-sequential-actions-20261105.w7m2k9qa/roi/source/source_v2_roi.py")
CACHE_INDEX_SHA256 = "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0"

VIDEO_ROWS = 4320
T0_VALID_TEXT_ROWS = 263
T1_VALID_TEXT_ROWS = 224
TEXT_STORAGE_ROWS = 320
PHYSICAL_CACHE_ROWS = 4640
HISTORY_ROWS = VIDEO_ROWS + T0_VALID_TEXT_ROWS
QUERY_ROWS = VIDEO_ROWS + T1_VALID_TEXT_ROWS
KEY_ROWS = HISTORY_ROWS + QUERY_ROWS
HIDDEN = 2048
HEADS = 16
HEAD_DIM = 128
BLOCKS = 54
CASE_ORDER = ("M11", "M00", "M10", "M01", "M11_repeat")
CHECKPOINTS = ("velocity", "block0.image", "block0.text_valid", "block53.image", "block53.text_valid")
ATOL = 2e-5
RTOL = 2e-5
WINDOWS = ((0, 144, -1), (144, 1440, 0), (1440, 1728, -1),
           (1728, 4032, 1), (4032, 4320, -1))
ROIS = {"action1": {"h": [0, 9], "w": [1, 8]},
        "action2": {"h": [2, 8], "w": [10, 15]}}
FACTORS = {
    "M11": {"action1_video_to_text": True, "action2_video_to_text": True},
    "M00": {"action1_video_to_text": False, "action2_video_to_text": False},
    "M10": {"action1_video_to_text": True, "action2_video_to_text": False},
    "M01": {"action1_video_to_text": False, "action2_video_to_text": True},
    "M11_repeat": {"action1_video_to_text": True, "action2_video_to_text": True},
}
INPUT_SHAPES = {
    "t0.patch": [1, VIDEO_ROWS, HIDDEN],
    "t0.text_valid": [1, T0_VALID_TEXT_ROWS, HIDDEN],
    "t0.vector": [1, HIDDEN],
    "t1.step1.patch": [1, VIDEO_ROWS, HIDDEN],
    "t1.step1.text_valid": [1, T1_VALID_TEXT_ROWS, HIDDEN],
    "t1.step1.vector": [1, HIDDEN],
    "t1.initial_latent": [1, 32, 30, 9, 16],
    "t1.step1.velocity": [1, 32, 30, 9, 16],
}
INPUT_SOURCE_RANGES = {
    "t0.patch": (3268608, "bbcc0e5fdfb37c3099bcef32bde4b599910e6f84262818939c3a834a7b09ba1f"),
    "t0.text_valid": (1114112, "c51b363ce28586b2589eb801790f0d73a73289f808813723f0dadf01f946d79f"),
    "t0.vector": (1105920, "e9f579a4363097651df785dda4837e468949319afab4ad47763920b90f217697"),
    "t1.step1.patch": (153686016, "d6925473556e9fb99adc1cc9ec1d4160b89eafb7da025073cd0c519f8d73d4bc"),
    "t1.step1.text_valid": (151851008, "082473f87ae24b6104afa8a8a69e4a9e7ba57eef46437aa61dd3df7a732dc089"),
    "t1.step1.vector": (151842816, "68c388122813f799812431a52d7af5f5b863eda2a923ef0365fe079f6971081d"),
    "t1.initial_latent": (151289856, "503615dc8920afe80d441fa025380c4b764b1e81f07d3a44525127010ddb2e5a"),
    "t1.step1.velocity": (263524352, "bce56894f9a6dacd12e78e7171c7770d98206aa3f8a2dff3c5bdd30a110d2222"),
}
OUTPUT_SHAPES = {
    "velocity": [1, 32, 30, 9, 16],
    "block0.image": [1, VIDEO_ROWS, HIDDEN],
    "block0.text_valid": [1, T1_VALID_TEXT_ROWS, HIDDEN],
    "block53.image": [1, VIDEO_ROWS, HIDDEN],
    "block53.text_valid": [1, T1_VALID_TEXT_ROWS, HIDDEN],
}
PARENT_CHECKPOINT_NAMES = {
    "velocity": "t1.step1.velocity",
    "block0.image": "t1.step1.block0.image",
    "block0.text_valid": "t1.step1.block0.text",
    "block53.image": "t1.step1.block53.image",
    "block53.text_valid": "t1.step1.block53.text",
}
MASK_REMOVALS = {"action1": 47_061, "action2": 44_160}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_HASH_CHUNK = 8 * 1024 * 1024


class ReferenceError(RuntimeError):
    """Fail-closed admission, mask, source, or comparator error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ReferenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ReferenceError(f"{label} does not exist: {path}") from None
    if path.is_symlink() or not resolved.is_file():
        raise ReferenceError(f"{label} must be a non-symlink regular file: {path}")
    return resolved


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != _digest(expected_sha256, f"{label} SHA-256"):
        raise ReferenceError(f"{label} digest mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    except ReferenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError(f"invalid {label} JSON") from None
    if type(value) is not dict:
        raise ReferenceError(f"{label} root must be an object")
    return value


def _range_sha256(path: Path, offset: int, nbytes: int) -> str:
    if type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes < 0:
        raise ReferenceError("tensor range bounds are malformed")
    if offset + nbytes > path.stat().st_size:
        raise ReferenceError(f"tensor range escapes payload: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = stream.read(min(remaining, _HASH_CHUNK))
            if not chunk:
                raise ReferenceError(f"truncated tensor range: {path.name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _require_file_digest(value: str | Path, expected: str, label: str) -> Path:
    path = _canonical_file(value, label)
    if sha256_file(path) != _digest(expected, f"{label} SHA-256"):
        raise ReferenceError(f"{label} digest mismatch")
    return path


def query_key_labels() -> tuple[list[int], list[int]]:
    """Return the active compact labels; physical padded rows are excluded."""
    q = [-1] * VIDEO_ROWS
    for start, stop, label in WINDOWS:
        if label >= 0:
            q[start:stop] = [label + 1] * (stop - start)
    q.extend([0] * 49 + [1] * 83 + [2] * 92)
    # History and all current-image keys retain global label zero. The action
    # labels exist only on the corresponding current text-key spans.
    k = [0] * (HISTORY_ROWS + VIDEO_ROWS)
    k.extend([0] * 49 + [1] * 83 + [2] * 92)
    if len(q) != QUERY_ROWS or len(k) != KEY_ROWS:
        raise ReferenceError("compact query/key label geometry drift")
    return q, k


def _roi_contains(q_index: int, action_index: int) -> bool:
    frame, spatial = divmod(q_index, 144)
    h, w = divmod(spatial, 16)
    if action_index == 0:
        return 1 <= frame < 10 and 0 <= h < 9 and 1 <= w < 8
    if action_index == 1:
        return 12 <= frame < 28 and 2 <= h < 8 and 10 <= w < 15
    raise ReferenceError("unknown action index")


def factorial_mask(case: str) -> bytes:
    """Build one compact row-major u8 mask under the frozen route contract."""
    if case not in FACTORS:
        raise ReferenceError(f"unadmitted route case: {case}")
    q_labels, k_labels = query_key_labels()
    # Four row templates cover all label combinations. Copying these rows into
    # the bytearray avoids Python-level iteration over the 41-million-cell grid.
    row_templates = {
        label: bytes(1 if (key_label == 0 or (label >= 0 and key_label == label)) else 0
                     for key_label in k_labels)
        for label in (-1, 0, 1, 2)
    }
    matrix = bytearray(QUERY_ROWS * KEY_ROWS)
    action_text = ((49, 132), (132, 224))
    for qi, qlabel in enumerate(q_labels):
        row_start = qi * KEY_ROWS
        row = row_templates[qlabel]
        matrix[row_start:row_start + KEY_ROWS] = row
        if qi >= VIDEO_ROWS or qlabel not in (1, 2):
            continue
        action_index = qlabel - 1
        span_start, span_stop = action_text[action_index]
        factor = FACTORS[case][f"action{action_index + 1}_video_to_text"]
        if not factor or not _roi_contains(qi, action_index):
            start = row_start + HISTORY_ROWS + VIDEO_ROWS + span_start
            stop = row_start + HISTORY_ROWS + VIDEO_ROWS + span_stop
            matrix[start:stop] = bytes(span_stop - span_start)
    return bytes(matrix)


def mask_proof() -> dict[str, Any]:
    q, k = query_key_labels()
    cases: dict[str, Any] = {}
    expected_removed = {
        case: {
            "action1": 0 if FACTORS[case]["action1_video_to_text"] else MASK_REMOVALS["action1"],
            "action2": 0 if FACTORS[case]["action2_video_to_text"] else MASK_REMOVALS["action2"],
        }
        for case in CASE_ORDER
    }
    for case in CASE_ORDER:
        matrix = factorial_mask(case)
        removed = expected_removed[case]
        cases[case] = {
            "matrix_sha256": hashlib.sha256(matrix).hexdigest(),
            "allowed_edges": matrix.count(1),
            "removed_edges_from_M11": removed,
            "route_bits": FACTORS[case],
        }
    return {
        "query_labels": q,
        "key_labels": k,
        "shape": [QUERY_ROWS, KEY_ROWS],
        "encoding": "row-major uint8; 0=masked, 1=allowed",
        "label_rule": "global plus same-action labels; interventions clear only matching action-text edges from positive video queries",
        "roi": ROIS,
        "source_range_builder_sha256": SOURCE_RANGE_BUILDER_SHA256,
        "cases": cases,
    }


def _record_for_roi(roi: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    records = roi.get("capture_payload", {}).get("records")
    if type(records) is not list:
        raise ReferenceError("ROI capture record inventory missing")
    matches = [item for item in records if type(item) is dict and item.get("name") == name]
    if len(matches) != 1:
        raise ReferenceError(f"ROI source record missing or duplicated: {name}")
    return matches[0]


def authenticate_inputs(
    frozen_inputs_path: str | Path = DEFAULT_FROZEN_INPUTS,
    roi_manifest_path: str | Path = DEFAULT_ROI_MANIFEST,
    continuation_manifest_path: str | Path = DEFAULT_CONTINUATION_MANIFEST,
) -> dict[str, Any]:
    """Authenticate frozen inputs, accepted ROI ranges, continuation payload, and RoPE."""
    frozen_path = _require_file_digest(frozen_inputs_path, FROZEN_INPUTS_SHA256, "frozen input manifest")
    roi_path = _require_file_digest(roi_manifest_path, ROI_MANIFEST_SHA256, "ROI source manifest")
    continuation_path = _require_file_digest(
        continuation_manifest_path, CONTINUATION_MANIFEST_SHA256, "continuation manifest"
    )
    frozen = _read_json(frozen_path, "frozen input manifest")
    roi = _read_json(roi_path, "ROI source manifest")
    continuation = _read_json(continuation_path, "continuation manifest")
    if (frozen.get("schema") != "cogni_ml.direct_route_frozen_input_binding/v1"
            or frozen.get("parent_source_sha256") != ROI_MANIFEST_SHA256
            or frozen.get("accepted_parent_comparison_sha256") != ACCEPTED_COMPARISON_SHA256
            or frozen.get("roi_spec_sha256") != "ab1721c868ea3d178057d453a8ef7cdc5104195ec6acd9c54ddaff148f5d0618"):
        raise ReferenceError("frozen input parent binding mismatch")
    if (roi.get("schema") != "cogni_ml.hy_nomas_structured_reference_control/v2"
            or roi.get("source", {}).get("upstream_revision") != PINNED_REVISION
            or roi.get("capture_payload", {}).get("sha256") != ROI_PAYLOAD_SHA256
            or roi.get("capture_payload", {}).get("capture_script_sha256") != ROI_CAPTURE_SCRIPT_SHA256
            or roi.get("query_roi_control", {}).get("roi_spec_sha256") != frozen.get("roi_spec_sha256")):
        raise ReferenceError("accepted ROI source binding mismatch")
    if (continuation.get("schema") != "cogni_ml.hy_nomas_continuation_inputs/v1"
            or continuation.get("upstream_revision") != PINNED_REVISION
            or continuation.get("payload_sha256") != CONTINUATION_PAYLOAD_SHA256):
        raise ReferenceError("continuation source binding mismatch")
    roi_payload = _require_file_digest(
        roi_path.parent / roi["capture_payload"]["path"], ROI_PAYLOAD_SHA256, "ROI source payload"
    )
    continuation_payload = _require_file_digest(
        continuation_path.parent / continuation["payload"], CONTINUATION_PAYLOAD_SHA256,
        "continuation payload",
    )
    frozen_records = frozen.get("inputs")
    names = {item.get("name") for item in frozen_records if type(item) is dict} if type(frozen_records) is list else set()
    if names != set(INPUT_SHAPES):
        raise ReferenceError("frozen input tensor inventory mismatch")
    admitted_records: dict[str, Any] = {}
    for item in frozen_records:
        name = item["name"]
        if item.get("matches_accepted_source_record") is not True:
            raise ReferenceError(f"frozen input lacks accepted-source match: {name}")
        source_record = _record_for_roi(roi, name)
        expected_bytes = math.prod(INPUT_SHAPES[name]) * 4
        if (item.get("source_range_offset") != source_record.get("offset")
                or item.get("source_range_bytes") != source_record.get("nbytes")
                or item.get("sha256") != source_record.get("sha256")
                or item.get("source_range_bytes") != expected_bytes):
            raise ReferenceError(f"frozen input/source tensor binding mismatch: {name}")
        native_path = _canonical_file(item.get("native_file", ""), f"frozen input {name}")
        if native_path.stat().st_size != item.get("bytes") or sha256_file(native_path) != item.get("sha256"):
            raise ReferenceError(f"frozen input file digest mismatch: {name}")
        if _range_sha256(roi_payload, source_record["offset"], source_record["nbytes"]) != source_record["sha256"]:
            raise ReferenceError(f"ROI source tensor range digest mismatch: {name}")
        admitted_records[name] = {
            "path": str(native_path), "bytes": item["bytes"], "sha256": item["sha256"],
            "shape": INPUT_SHAPES[name], "source_range_offset": source_record["offset"],
            "source_range_bytes": source_record["nbytes"],
        }
    parent_checkpoints: dict[str, Any] = {}
    for checkpoint, source_name in PARENT_CHECKPOINT_NAMES.items():
        record = _record_for_roi(roi, source_name)
        expected_shape = OUTPUT_SHAPES[checkpoint]
        expected_bytes = math.prod(expected_shape) * 4
        if (record.get("shape") != expected_shape or record.get("nbytes") != expected_bytes
                or record.get("storage_dtype") != "F32" or record.get("source_dtype") != "BF16"
                or type(record.get("offset")) is not int or record["offset"] < 0):
            raise ReferenceError(f"accepted parent checkpoint record mismatch: {checkpoint}")
        _digest(record.get("sha256"), f"accepted parent checkpoint {checkpoint} SHA-256")
        if _range_sha256(roi_payload, record["offset"], record["nbytes"]) != record["sha256"]:
            raise ReferenceError(f"accepted parent checkpoint range digest mismatch: {checkpoint}")
        parent_checkpoints[checkpoint] = record
    tensors = continuation.get("tensors")
    rope_records = {}
    for name, expected_sha, offset in (
        ("input.rope_cos", ROPE_COS_SHA256, 12_492_800),
        ("input.rope_sin", ROPE_SIN_SHA256, 25_763_840),
    ):
        record = tensors.get(name) if type(tensors) is dict else None
        nbytes = 13_271_040
        if (type(record) is not dict or record.get("dtype") != "F32"
                or record.get("shape") != [6, VIDEO_ROWS, HEAD_DIM]
                or record.get("offset") != offset or record.get("nbytes") != nbytes
                or record.get("sha256") != expected_sha):
            raise ReferenceError(f"continuation {name} record mismatch")
        if _range_sha256(continuation_payload, offset, nbytes) != expected_sha:
            raise ReferenceError(f"continuation {name} payload range mismatch")
        bank_bytes = VIDEO_ROWS * HEAD_DIM * 4
        banks = {}
        for bank_name, bank_index in (("T0", 0), ("T1", 1)):
            bank_offset = offset + bank_index * bank_bytes
            banks[bank_name] = {
                "selected_bank": bank_index,
                "selected_range_offset": bank_offset,
                "selected_range_bytes": bank_bytes,
                "selected_range_sha256": _range_sha256(continuation_payload, bank_offset, bank_bytes),
            }
        rope_records[name] = {
            "record": record, "banks": banks,
        }
    return {
        "frozen_inputs_path": str(frozen_path), "frozen_inputs_sha256": FROZEN_INPUTS_SHA256,
        "roi_manifest_path": str(roi_path), "roi_manifest_sha256": ROI_MANIFEST_SHA256,
        "roi_payload_path": str(roi_payload), "roi_payload_sha256": ROI_PAYLOAD_SHA256,
        "continuation_manifest_path": str(continuation_path),
        "continuation_manifest_sha256": CONTINUATION_MANIFEST_SHA256,
        "continuation_payload_path": str(continuation_payload),
        "continuation_payload_sha256": CONTINUATION_PAYLOAD_SHA256,
        "accepted_parent_comparison_sha256": ACCEPTED_COMPARISON_SHA256,
        "inputs": admitted_records, "rope": rope_records,
        "parent_checkpoints": parent_checkpoints,
        "roi_document": roi, "continuation_document": continuation,
    }


def _load_roi_source_module() -> Any:
    for path in (SOURCE_REPLAY_SUPPORT, ROI_SOURCE_SCRIPT.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if sha256_file(_canonical_file(ROI_SOURCE_SCRIPT, "ROI source adapter")) != ROI_CAPTURE_SCRIPT_SHA256:
        raise ReferenceError("ROI source adapter digest mismatch")
    spec = importlib.util.spec_from_file_location("hy_nomas_route_factorial_roi_source", ROI_SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ReferenceError("could not load accepted ROI source adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ReferenceError(f"accepted ROI source adapter import failed: {type(error).__name__}: {error}") from None
    return module


def build_pinned_source_masks(torch: Any, upstream: Path) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Build source ROI mask once and derive the five permitted edge removals."""
    roi_source = _load_roi_source_module()
    range_builder, source_sha, source_start, source_end = roi_source._build_pinned_range_builder(upstream, torch)
    if source_sha != SOURCE_RANGE_BUILDER_SHA256:
        raise ReferenceError("pinned source range builder hash mismatch")
    q_labels, _ = query_key_labels()
    windows = [list(window) for window in WINDOWS]
    source_chunks, range_record = roi_source._source_mask_chunks_cpu(
        torch, range_builder, history_rows=HISTORY_ROWS, image_rows=VIDEO_ROWS,
        valid_text_rows=T1_VALID_TEXT_ROWS, offsets=[49, 132, 224],
        video_windows=windows, query_chunk=256,
    )
    if (range_record.get("query_rows") != QUERY_ROWS or range_record.get("key_rows") != KEY_ROWS
            or range_record.get("query_chunk") != 256):
        raise ReferenceError("pinned source range-builder returned unexpected mask geometry")
    proof = mask_proof()
    result: dict[str, list[Any]] = {}
    source_rows: list[bytes] = []
    for chunk in source_chunks:
        if chunk.device.type != "cpu" or tuple(chunk.shape)[1] != KEY_ROWS:
            raise ReferenceError("pinned source mask chunk is not compact CPU geometry")
        source_rows.append(chunk.to(dtype=torch.uint8).numpy().tobytes(order="C"))
    source_base = b"".join(source_rows)
    if hashlib.sha256(source_base).hexdigest() != proof["cases"]["M11"]["matrix_sha256"]:
        raise ReferenceError("pinned accepted ROI M11 mask differs from independent label-mask proof")

    for case in CASE_ORDER:
        chunks = [chunk.clone() for chunk in source_chunks]
        if case not in ("M11", "M11_repeat"):
            for chunk_index, chunk in enumerate(chunks):
                q_start = chunk_index * 256
                q_stop = min(q_start + int(chunk.shape[0]), QUERY_ROWS)
                for action_index, (span_start, span_stop) in enumerate(((49, 132), (132, 224))):
                    factor = FACTORS[case][f"action{action_index + 1}_video_to_text"]
                    if factor:
                        continue
                    key_start = HISTORY_ROWS + VIDEO_ROWS + span_start
                    key_stop = HISTORY_ROWS + VIDEO_ROWS + span_stop
                    for query in range(q_start, min(q_stop, VIDEO_ROWS)):
                        if q_labels[query] == action_index + 1:
                            chunk[query - q_start, key_start:key_stop] = False
        digest = hashlib.sha256()
        expected = factorial_mask(case)
        cursor = 0
        for chunk in chunks:
            raw = chunk.to(dtype=torch.uint8).numpy().tobytes(order="C")
            digest.update(raw)
            if expected[cursor:cursor + len(raw)] != raw:
                raise ReferenceError(f"source mask differs from factorial contract: {case}")
            cursor += len(raw)
        if cursor != len(expected) or digest.hexdigest() != proof["cases"][case]["matrix_sha256"]:
            raise ReferenceError(f"source route mask digest mismatch: {case}")
        result[case] = chunks
    return result, {
        "source_range_builder_sha256": source_sha,
        "source_range_builder_lines": [source_start, source_end],
        "source_range_count": range_record["range_count"],
        "query_chunk": 256,
        "chunks_per_case": len(source_chunks),
        "query_rows": QUERY_ROWS,
        "key_rows": KEY_ROWS,
    }


def compare_checkpoint(label: str, expected: Any, actual: Any) -> dict[str, Any]:
    """Compare arrays using the declared elementwise fixed tolerance."""
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for checkpoint comparison: {error}") from error
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ReferenceError(f"checkpoint shape/nonfinite mismatch: {label}")
    lhs, rhs = expected.astype(np.float64), actual.astype(np.float64)
    delta = np.abs(lhs - rhs)
    outside = int(np.count_nonzero(delta > ATOL + RTOL * np.abs(lhs)))
    return {
        "checkpoint": label, "elements": int(delta.size),
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "outside_atol_rtol": outside, "all_finite": True, "passed": outside == 0,
        "atol": ATOL, "rtol": RTOL,
    }


def _validate_stats(value: Any, label: str, checkpoint: str) -> None:
    elements = math.prod(OUTPUT_SHAPES[checkpoint])
    if (type(value) is not dict or value.get("checkpoint") != label
            or type(value.get("elements")) is not int or value.get("elements") != elements
            or type(value.get("outside_atol_rtol")) is not int or value["outside_atol_rtol"] != 0
            or value.get("all_finite") is not True or value.get("passed") is not True
            or value.get("atol") != ATOL or value.get("rtol") != RTOL
            or any(type(value.get(key)) not in (int, float) or not math.isfinite(float(value[key])) or value[key] < 0
                   for key in ("max_abs", "mean_abs", "rms_abs"))):
        raise ReferenceError(f"comparison gate failed: {label}")


PRECISION_CONTRACT = {
    "resident_weights": "BF16",
    "input_storage": "F32 lossless promotion of BF16 saved states",
    "activation_dtype": "BF16",
    "attention_adapter": "pinned Hy-Nomas source cache body with compact boolean MPS SDPA chunks of 256; BF16 Q/K/V",
    "cache_dtype": "BF16",
    "all_54_block_shards_authenticated": True,
}

EXECUTION_FIXED = {
    "case_order": list(CASE_ORDER),
    "scope": "saved-state direct DiT only; no encoder/refiner/Euler/DCAE",
    "schedule": "layer-outer T0 writer then five independent T1 arms per layer; one resident BF16 block and one T0 K/V cache",
    "full_blocks_per_arm": BLOCKS,
    "t0_writer_block_calls": BLOCKS,
    "t1_arm_block_calls": BLOCKS * 5,
    "source_range_builder_invocations": 1,
    "source_mask_cases": len(CASE_ORDER),
    "terminal_export_count": 1,
    "explicit_terminal_mps_synchronize_count": 1,
    "block0_text_equal_all_cases": True,
}

RUNTIME_RECEIPT_FIXED = {
    "attention_calls": 324,
    "attention_query_chunks": 5832,
    "t0_attention_calls": 54,
    "t1_attention_calls": 270,
    "explicit_terminal_synchronizations": 1,
    "intermediate_readbacks": 0,
    "scheduler_calls": 0,
    "decoder_calls": 0,
    "refiner_calls": 0,
    "weights_staged": 54,
    "arms_per_layer": 6,
    "one_layer_cache": True,
    "autocast": "mps BF16 full block and final head",
}


def _absolute_path(value: Any, label: str) -> None:
    if type(value) is not str or not Path(value).is_absolute():
        raise ReferenceError(f"{label} must be an absolute path")


def _validate_source(source: Any) -> None:
    keys = {"upstream_path", "upstream_revision", "source_hashes", "cache_index_path",
            "cache_index_sha256", "block_shards", "global_shard", "total_weight_bytes"}
    if type(source) is not dict or set(source) != keys:
        raise ReferenceError("pinned source keys mismatch")
    _absolute_path(source.get("upstream_path"), "upstream path")
    _absolute_path(source.get("cache_index_path"), "cache index path")
    if (source.get("upstream_revision") != PINNED_REVISION
            or source.get("cache_index_sha256") != CACHE_INDEX_SHA256):
        raise ReferenceError("pinned source revision/cache-index mismatch")
    hashes = source.get("source_hashes")
    if (type(hashes) is not dict or not hashes
            or any(type(name) is not str or not name or Path(name).is_absolute() for name in hashes)
            or any(_HEX.fullmatch(value) is None if type(value) is str else True for value in hashes.values())):
        raise ReferenceError("pinned source file hash inventory malformed")
    shards = source.get("block_shards")
    shard_keys = {"block", "path", "bytes", "sha256", "tensor_count"}
    if type(shards) is not list or len(shards) != BLOCKS:
        raise ReferenceError("pinned source block-shard inventory mismatch")
    total = 0
    for index, shard in enumerate(shards):
        if (type(shard) is not dict or set(shard) != shard_keys or shard.get("block") != index
                or type(shard.get("bytes")) is not int or shard["bytes"] <= 0
                or type(shard.get("tensor_count")) is not int or shard["tensor_count"] != 32):
            raise ReferenceError(f"pinned source block-shard record malformed: {index}")
        _absolute_path(shard.get("path"), f"block {index} shard path")
        _digest(shard.get("sha256"), f"block {index} shard SHA-256")
        total += shard["bytes"]
    global_shard = source.get("global_shard")
    global_keys = {"path", "bytes", "sha256", "tensor_count"}
    if (type(global_shard) is not dict or set(global_shard) != global_keys
            or type(global_shard.get("bytes")) is not int or global_shard["bytes"] <= 0
            or type(global_shard.get("tensor_count")) is not int or global_shard["tensor_count"] <= 0):
        raise ReferenceError("pinned source global-shard record malformed")
    _absolute_path(global_shard.get("path"), "global shard path")
    _digest(global_shard.get("sha256"), "global shard SHA-256")
    total += global_shard["bytes"]
    if type(source.get("total_weight_bytes")) is not int or source["total_weight_bytes"] != total:
        raise ReferenceError("pinned source total weight extent mismatch")


def _validate_source_inputs(source_inputs: Any) -> None:
    keys = {"frozen_inputs", "roi", "continuation", "accepted_parent_comparison_sha256", "tensors"}
    if type(source_inputs) is not dict or set(source_inputs) != keys:
        raise ReferenceError("source input keys mismatch")
    frozen = source_inputs["frozen_inputs"]
    if type(frozen) is not dict or set(frozen) != {"path", "sha256"}:
        raise ReferenceError("frozen input binding malformed")
    _absolute_path(frozen.get("path"), "frozen input manifest path")
    if frozen.get("sha256") != FROZEN_INPUTS_SHA256:
        raise ReferenceError("frozen input manifest binding mismatch")
    roi = source_inputs["roi"]
    if type(roi) is not dict or set(roi) != {
        "manifest_path", "manifest_sha256", "payload_path", "payload_sha256", "capture_script_sha256"
    }:
        raise ReferenceError("accepted ROI input binding malformed")
    for key in ("manifest_path", "payload_path"):
        _absolute_path(roi.get(key), f"ROI {key}")
    for key, expected in (("manifest_sha256", ROI_MANIFEST_SHA256),
                          ("payload_sha256", ROI_PAYLOAD_SHA256),
                          ("capture_script_sha256", ROI_CAPTURE_SCRIPT_SHA256)):
        if roi.get(key) != expected:
            raise ReferenceError(f"accepted ROI {key} mismatch")
    continuation = source_inputs["continuation"]
    if type(continuation) is not dict or set(continuation) != {
        "manifest_path", "manifest_sha256", "payload_path", "payload_sha256", "rope_cos", "rope_sin"
    }:
        raise ReferenceError("continuation input binding malformed")
    for key in ("manifest_path", "payload_path"):
        _absolute_path(continuation.get(key), f"continuation {key}")
    if (continuation.get("manifest_sha256") != CONTINUATION_MANIFEST_SHA256
            or continuation.get("payload_sha256") != CONTINUATION_PAYLOAD_SHA256):
        raise ReferenceError("continuation manifest/payload binding mismatch")
    for name, expected_sha, expected_offset in (
        ("rope_cos", ROPE_COS_SHA256, 12_492_800),
        ("rope_sin", ROPE_SIN_SHA256, 25_763_840),
    ):
        record = continuation.get(name)
        if type(record) is not dict or set(record) != {"record", "banks"}:
            raise ReferenceError(f"{name} record/bank binding malformed")
        tensor = record["record"]
        if (type(tensor) is not dict or set(tensor) != {"dtype", "shape", "offset", "nbytes", "sha256"}
                or tensor.get("dtype") != "F32" or tensor.get("shape") != [6, VIDEO_ROWS, HEAD_DIM]
                or tensor.get("offset") != expected_offset or tensor.get("nbytes") != 13_271_040
                or tensor.get("sha256") != expected_sha):
            raise ReferenceError(f"{name} source record mismatch")
        banks = record["banks"]
        if type(banks) is not dict or set(banks) != {"T0", "T1"}:
            raise ReferenceError(f"{name} bank inventory mismatch")
        bank_bytes = VIDEO_ROWS * HEAD_DIM * 4
        for bank_name, bank_index in (("T0", 0), ("T1", 1)):
            bank = banks[bank_name]
            if (type(bank) is not dict
                    or set(bank) != {"selected_bank", "selected_range_offset", "selected_range_bytes", "selected_range_sha256"}
                    or bank.get("selected_bank") != bank_index
                    or bank.get("selected_range_offset") != expected_offset + bank_index * bank_bytes
                    or bank.get("selected_range_bytes") != bank_bytes):
                raise ReferenceError(f"{name} {bank_name} bank range mismatch")
            _digest(bank.get("selected_range_sha256"), f"{name} {bank_name} SHA-256")
    if source_inputs.get("accepted_parent_comparison_sha256") != ACCEPTED_COMPARISON_SHA256:
        raise ReferenceError("accepted parent comparison binding mismatch")
    tensors = source_inputs.get("tensors")
    if type(tensors) is not dict or set(tensors) != set(INPUT_SHAPES):
        raise ReferenceError("source input tensor inventory mismatch")
    for name, shape in INPUT_SHAPES.items():
        record = tensors[name]
        expected_offset, expected_sha = INPUT_SOURCE_RANGES[name]
        nbytes = math.prod(shape) * 4
        if (type(record) is not dict or set(record) != {
            "path", "bytes", "sha256", "dtype", "shape", "source_range_offset", "source_range_bytes"
        } or record.get("dtype") != "F32" or record.get("shape") != shape
                or record.get("bytes") != nbytes or record.get("source_range_bytes") != nbytes
                or record.get("source_range_offset") != expected_offset or record.get("sha256") != expected_sha):
            raise ReferenceError(f"source input tensor provenance mismatch: {name}")
        _absolute_path(record.get("path"), f"source input {name}")
        _digest(record.get("sha256"), f"source input {name} SHA-256")


def validate_reference_document(document: Mapping[str, Any]) -> None:
    """Validate exact source result schema and all declared pass gates."""
    required = {"schema", "schema_version", "upstream_revision", "source", "source_inputs",
                "geometry", "case_factors", "mask_proof", "precision", "execution", "payload", "tensors"}
    if type(document) is not dict or set(document) != required:
        raise ReferenceError("route factorial reference keys mismatch")
    if document.get("schema") != SCHEMA or document.get("schema_version") != SCHEMA_VERSION:
        raise ReferenceError("route factorial schema/version mismatch")
    if document.get("upstream_revision") != PINNED_REVISION or document.get("case_factors") != FACTORS:
        raise ReferenceError("route factorial source revision or factors mismatch")
    _validate_source(document.get("source"))
    _validate_source_inputs(document.get("source_inputs"))
    geometry = {
        "video_rows": VIDEO_ROWS, "t0_valid_text_rows": T0_VALID_TEXT_ROWS,
        "t1_valid_text_rows": T1_VALID_TEXT_ROWS, "text_storage_rows": TEXT_STORAGE_ROWS,
        "physical_cache_rows": PHYSICAL_CACHE_ROWS, "history_rows": HISTORY_ROWS,
        "query_rows": QUERY_ROWS, "key_rows": KEY_ROWS, "hidden": HIDDEN,
        "heads": HEADS, "head_dim": HEAD_DIM, "blocks": BLOCKS, "attention_layout": "compact",
    }
    if document.get("geometry") != geometry:
        raise ReferenceError("route factorial geometry mismatch")
    proof = document.get("mask_proof")
    expected_proof = mask_proof()
    for key in ("shape", "encoding", "query_labels", "key_labels", "label_rule",
                "roi", "source_range_builder_sha256", "cases"):
        if type(proof) is not dict or proof.get(key) != expected_proof[key]:
            raise ReferenceError(f"route factorial mask proof mismatch: {key}")
    if set(proof) != set(expected_proof):
        raise ReferenceError("route factorial mask proof keyset mismatch")
    if document.get("precision") != PRECISION_CONTRACT:
        raise ReferenceError("route factorial precision contract mismatch")
    execution = document.get("execution")
    expected_execution_keys = set(EXECUTION_FIXED) | {"parent_replay", "source_aa"}
    if (type(execution) is not dict or set(execution) != expected_execution_keys
            or any(execution.get(key) != value for key, value in EXECUTION_FIXED.items())):
        raise ReferenceError("route factorial case order mismatch")
    replay = execution.get("parent_replay")
    if type(replay) is not dict or set(replay) != {"M11"} or set(replay["M11"]) != set(CHECKPOINTS):
        raise ReferenceError("parent M11 replay checkpoint inventory mismatch")
    for checkpoint in CHECKPOINTS:
        _validate_stats(replay["M11"][checkpoint], checkpoint, checkpoint)
    aa = execution.get("source_aa")
    if type(aa) is not dict or set(aa) != set(CHECKPOINTS):
        raise ReferenceError("M11 A/A checkpoint inventory mismatch")
    for checkpoint in CHECKPOINTS:
        _validate_stats(aa[checkpoint], checkpoint, checkpoint)
    payload = document.get("payload")
    if (type(payload) is not dict or set(payload) != {"basename", "bytes", "sha256", "encoding"}
            or type(payload.get("basename")) is not str or Path(payload["basename"]).name != payload["basename"]
            or type(payload.get("bytes")) is not int or payload["bytes"] <= 0
            or payload.get("encoding") != "f32le_bf16_expansion"):
        raise ReferenceError("route factorial payload descriptor mismatch")
    _digest(payload.get("sha256"), "route factorial payload SHA-256")
    names = {f"{case}.{checkpoint}" for case in CASE_ORDER for checkpoint in CHECKPOINTS}
    tensors = document.get("tensors")
    if type(tensors) is not dict or set(tensors) != names:
        raise ReferenceError("route factorial tensor inventory mismatch")
    cursor = 0
    for case in CASE_ORDER:
        for checkpoint in CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            record = tensors[name]
            nbytes = math.prod(OUTPUT_SHAPES[checkpoint]) * 4
            if (type(record) is not dict
                    or set(record) != {"dtype", "source_dtype", "shape", "offset_bytes", "nbytes", "sha256", "checkpoint_boundary"}
                    or record.get("dtype") != "F32" or record.get("source_dtype") != "BF16"
                    or record.get("shape") != OUTPUT_SHAPES[checkpoint]
                    or type(record.get("offset_bytes")) is not int or record["offset_bytes"] != cursor
                    or type(record.get("nbytes")) is not int or record["nbytes"] != nbytes
                    or record.get("checkpoint_boundary") != checkpoint):
                raise ReferenceError(f"route factorial output record mismatch: {name}")
            _digest(record.get("sha256"), f"{name} SHA-256")
            cursor += nbytes
    if payload["bytes"] != cursor:
        raise ReferenceError("route factorial payload size mismatch")


def authenticate_model_sources(cache_index_path: str | Path, upstream_path: str | Path) -> dict[str, Any]:
    """Authenticate pinned upstream files and all 54 BF16 cache shards on CPU."""
    cache_index = _require_file_digest(cache_index_path, CACHE_INDEX_SHA256, "BF16 cache index")
    upstream = Path(upstream_path).expanduser()
    try:
        resolved_upstream = upstream.resolve(strict=True)
    except OSError:
        raise ReferenceError(f"pinned upstream directory does not exist: {upstream}") from None
    if upstream.is_symlink() or not resolved_upstream.is_dir():
        raise ReferenceError("pinned upstream must be a non-symlink directory")
    if str(SOURCE_REPLAY_SUPPORT) not in sys.path:
        sys.path.insert(0, str(SOURCE_REPLAY_SUPPORT))
    try:
        source_helper = importlib.import_module("replay_source_autocast_full_base")
        model = source_helper._authenticate_model_sources(SimpleNamespace(
            upstream=resolved_upstream,
            cache_index=cache_index,
            cache_index_sha256=CACHE_INDEX_SHA256,
        ))
    except Exception as error:
        raise ReferenceError(
            f"pinned model-source authentication failed: {type(error).__name__}: {error}"
        ) from None
    if model.get("upstream_revision") != PINNED_REVISION:
        raise ReferenceError("pinned model source revision mismatch")
    return model


def source_document(model: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the upstream source-authentication receipt to the locked schema."""
    try:
        shards = [
            {
                "block": index,
                "path": str(record["path"]),
                "bytes": int(record["bytes"]),
                "sha256": record["sha256"],
                "tensor_count": int(record["tensor_count"]),
            }
            for index, record in enumerate(model["cache_shards"])
        ]
        global_record = model["global_shard"]
        document = {
            "upstream_path": str(model["upstream_path"]),
            "upstream_revision": model["upstream_revision"],
            "source_hashes": dict(model["source_hashes"]),
            "cache_index_path": str(model["cache_index_path"]),
            "cache_index_sha256": CACHE_INDEX_SHA256,
            "block_shards": shards,
            "global_shard": {
                "path": str(global_record["path"]),
                "bytes": int(global_record["bytes"]),
                "sha256": global_record["sha256"],
                "tensor_count": int(global_record["tensor_count"]),
            },
            "total_weight_bytes": int(model["total_weight_bytes"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ReferenceError(f"model-source authentication receipt is malformed: {error}") from None
    _validate_source(document)
    return document


def source_inputs_document(admitted: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the saved-state, accepted ROI and continuation inputs in the output."""
    inputs = admitted["inputs"]
    return {
        "frozen_inputs": {
            "path": admitted["frozen_inputs_path"],
            "sha256": admitted["frozen_inputs_sha256"],
        },
        "roi": {
            "manifest_path": admitted["roi_manifest_path"],
            "manifest_sha256": admitted["roi_manifest_sha256"],
            "payload_path": admitted["roi_payload_path"],
            "payload_sha256": admitted["roi_payload_sha256"],
            "capture_script_sha256": ROI_CAPTURE_SCRIPT_SHA256,
        },
        "continuation": {
            "manifest_path": admitted["continuation_manifest_path"],
            "manifest_sha256": admitted["continuation_manifest_sha256"],
            "payload_path": admitted["continuation_payload_path"],
            "payload_sha256": admitted["continuation_payload_sha256"],
            "rope_cos": admitted["rope"]["input.rope_cos"],
            "rope_sin": admitted["rope"]["input.rope_sin"],
        },
        "accepted_parent_comparison_sha256": admitted["accepted_parent_comparison_sha256"],
        "tensors": {
            name: {**record, "dtype": "F32"}
            for name, record in inputs.items()
        },
    }


def validate_runtime_receipt(receipt: Any) -> dict[str, Any]:
    if type(receipt) is not dict or receipt != RUNTIME_RECEIPT_FIXED:
        raise ReferenceError("pinned source runtime receipt differs from the admitted one-layer schedule")
    return dict(receipt)


def _geometry_document() -> dict[str, Any]:
    return {
        "video_rows": VIDEO_ROWS, "t0_valid_text_rows": T0_VALID_TEXT_ROWS,
        "t1_valid_text_rows": T1_VALID_TEXT_ROWS, "text_storage_rows": TEXT_STORAGE_ROWS,
        "physical_cache_rows": PHYSICAL_CACHE_ROWS, "history_rows": HISTORY_ROWS,
        "query_rows": QUERY_ROWS, "key_rows": KEY_ROWS, "hidden": HIDDEN,
        "heads": HEADS, "head_dim": HEAD_DIM, "blocks": BLOCKS,
        "attention_layout": "compact",
    }


def validate_capture_map(captures: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact 25 finite F32 checkpoints returned by the source runtime."""
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for source captures: {error}") from error
    expected_names = {f"{case}.{checkpoint}" for case in CASE_ORDER for checkpoint in CHECKPOINTS}
    if not isinstance(captures, Mapping) or set(captures) != expected_names:
        raise ReferenceError("source runtime checkpoint inventory mismatch")
    result: dict[str, Any] = {}
    for case in CASE_ORDER:
        for checkpoint in CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            value = np.asarray(captures[name])
            if (list(value.shape) != OUTPUT_SHAPES[checkpoint]
                    or value.dtype != np.dtype("float32") or not np.isfinite(value).all()):
                raise ReferenceError(f"source runtime checkpoint shape/dtype/nonfinite mismatch: {name}")
            result[name] = value
    first_text = result["M11.block0.text_valid"]
    for case in CASE_ORDER[1:]:
        if not np.array_equal(first_text, result[f"{case}.block0.text_valid"]):
            raise ReferenceError("block0 text_valid changed across route cases")
    return result


def build_comparison_gates(admitted: Mapping[str, Any], captures: Mapping[str, Any]) -> dict[str, Any]:
    """Compare M11 with accepted ROI checkpoints and the M11 repeat A/A arm."""
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for source comparisons: {error}") from error
    source_payload = Path(admitted["roi_payload_path"])
    parent_records = admitted["parent_checkpoints"]
    parent_gates: dict[str, Any] = {}
    aa_gates: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        record = parent_records[checkpoint]
        parent = np.memmap(
            source_payload, dtype="<f4", mode="r", offset=record["offset"],
            shape=tuple(OUTPUT_SHAPES[checkpoint]),
        )
        parent_gates[checkpoint] = compare_checkpoint(
            checkpoint, parent, captures[f"M11.{checkpoint}"]
        )
        del parent
        aa_gates[checkpoint] = compare_checkpoint(
            checkpoint, captures[f"M11.{checkpoint}"], captures[f"M11_repeat.{checkpoint}"]
        )
    if any(not stats["passed"] for stats in parent_gates.values()):
        failed = [name for name, stats in parent_gates.items() if not stats["passed"]]
        raise ReferenceError(f"accepted M11 parent replay failed: {','.join(failed)}")
    if any(not stats["passed"] for stats in aa_gates.values()):
        failed = [name for name, stats in aa_gates.items() if not stats["passed"]]
        raise ReferenceError(f"source M11 A/A repeat failed: {','.join(failed)}")
    return {"parent_replay": {"M11": parent_gates}, "source_aa": aa_gates}


def _render_document(
    *, source: Mapping[str, Any], source_inputs: Mapping[str, Any], gates: Mapping[str, Any],
    tensor_records: Mapping[str, Any], payload_basename: str, payload_bytes: int,
    payload_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "upstream_revision": PINNED_REVISION,
        "source": dict(source),
        "source_inputs": dict(source_inputs),
        "geometry": _geometry_document(),
        "case_factors": FACTORS,
        "mask_proof": mask_proof(),
        "precision": PRECISION_CONTRACT,
        "execution": {
            **EXECUTION_FIXED,
            "parent_replay": gates["parent_replay"],
            "source_aa": gates["source_aa"],
        },
        "payload": {
            "basename": payload_basename,
            "bytes": payload_bytes,
            "sha256": payload_sha256,
            "encoding": "f32le_bf16_expansion",
        },
        "tensors": dict(tensor_records),
    }


def _new_output_path(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    parent = expanded.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ReferenceError(f"{label} parent is not a directory")
    final = parent / expanded.name
    if final.is_symlink() or os.path.lexists(final):
        raise ReferenceError(f"{label} must be a new file: {final}")
    return final


def publish_reference(
    output_manifest: str | Path,
    *, source: Mapping[str, Any], admitted: Mapping[str, Any], captures: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Write payload and manifest with no-replace publication after all gates pass."""
    import numpy as np

    captures = validate_capture_map(captures)
    gates = build_comparison_gates(admitted, captures)
    manifest_path = _new_output_path(Path(output_manifest), "output manifest")
    payload_path = _new_output_path(
        manifest_path.with_name(f"{manifest_path.stem}.tensors.f32le"), "output tensor payload"
    )
    payload_temp: Path | None = None
    manifest_temp: Path | None = None
    published_payload = False
    try:
        digest = hashlib.sha256()
        records: dict[str, Any] = {}
        cursor = 0
        fd, temp_name = tempfile.mkstemp(prefix=f".{payload_path.name}.", suffix=".tmp", dir=payload_path.parent)
        payload_temp = Path(temp_name)
        with os.fdopen(fd, "wb") as stream:
            for case in CASE_ORDER:
                for checkpoint in CHECKPOINTS:
                    name = f"{case}.{checkpoint}"
                    value = np.asarray(captures[name]).astype("<f4", copy=False)
                    raw = value.tobytes(order="C")
                    if len(raw) != math.prod(OUTPUT_SHAPES[checkpoint]) * 4:
                        raise ReferenceError(f"source runtime checkpoint byte count mismatch: {name}")
                    tensor_sha = hashlib.sha256(raw).hexdigest()
                    records[name] = {
                        "dtype": "F32", "source_dtype": "BF16",
                        "shape": OUTPUT_SHAPES[checkpoint], "offset_bytes": cursor,
                        "nbytes": len(raw), "sha256": tensor_sha,
                        "checkpoint_boundary": checkpoint,
                    }
                    stream.write(raw)
                    digest.update(raw)
                    cursor += len(raw)
                    del raw
            stream.flush()
            os.fsync(stream.fileno())
        document = _render_document(
            source=source,
            source_inputs=source_inputs_document(admitted),
            gates=gates,
            tensor_records=records,
            payload_basename=payload_path.name,
            payload_bytes=cursor,
            payload_sha256=digest.hexdigest(),
        )
        validate_reference_document(document)
        fd, temp_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent)
        manifest_temp = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(payload_temp, payload_path)
        published_payload = True
        os.unlink(payload_temp)
        payload_temp = None
        os.link(manifest_temp, manifest_path)
        os.unlink(manifest_temp)
        manifest_temp = None
        return manifest_path, payload_path, document
    except Exception:
        if published_payload:
            try:
                os.unlink(payload_path)
            except OSError:
                pass
        raise
    finally:
        for temp in (payload_temp, manifest_temp):
            if temp is not None:
                try:
                    os.unlink(temp)
                except OSError:
                    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-inputs", type=Path, default=DEFAULT_FROZEN_INPUTS)
    parser.add_argument("--roi-manifest", type=Path, default=DEFAULT_ROI_MANIFEST)
    parser.add_argument("--continuation-manifest", type=Path, default=DEFAULT_CONTINUATION_MANIFEST)
    parser.add_argument("--cache-index", type=Path, default=DEFAULT_CACHE_INDEX)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--admit-only", action="store_true", help="authenticate source inputs and masks without importing torch")
    parser.add_argument("--validate-source-mask", action="store_true",
                        help="validate accepted ROI plus factorial mask cases on CPU only")
    return parser


def _run_source_reference(args: argparse.Namespace, admitted: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    if args.output_manifest is None:
        raise ReferenceError("--output-manifest is required for source execution")
    manifest_path = _new_output_path(args.output_manifest, "output manifest")
    payload_path = _new_output_path(
        manifest_path.with_name(f"{manifest_path.stem}.tensors.f32le"), "output tensor payload"
    )
    del payload_path
    try:
        import torch
    except ImportError as error:
        raise ReferenceError(f"PyTorch is required for pinned source execution: {error}") from error
    if not torch.backends.mps.is_available():
        raise ReferenceError("pinned source reference requires an available MPS device")
    masks, mask_receipt = build_pinned_source_masks(torch, args.upstream)
    if (mask_receipt.get("chunks_per_case") != math.ceil(QUERY_ROWS / 256)
            or mask_receipt.get("query_rows") != QUERY_ROWS
            or mask_receipt.get("key_rows") != KEY_ROWS
            or set(masks) != set(CASE_ORDER)):
        raise ReferenceError("pinned source mask admission receipt mismatch")
    model = authenticate_model_sources(args.cache_index, args.upstream)
    source = source_document(model)
    runtime_inputs = dict(admitted)
    runtime_inputs["model"] = model
    try:
        runtime = importlib.import_module("hy_nomas_dit_route_factorial_runtime")
        result = runtime.run(runtime_inputs, args.cache_index, args.upstream, masks)
    except Exception as error:
        raise ReferenceError(f"pinned source runtime failed: {type(error).__name__}: {error}") from None
    if type(result) is not tuple or len(result) != 2:
        raise ReferenceError("pinned source runtime must return captures and a receipt")
    captures, runtime_receipt = result
    runtime_receipt = validate_runtime_receipt(runtime_receipt)
    manifest_path, final_payload_path, document = publish_reference(
        manifest_path, source=source, admitted=admitted, captures=captures,
    )
    return manifest_path, final_payload_path, document, runtime_receipt


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        admitted = authenticate_inputs(args.frozen_inputs, args.roi_manifest, args.continuation_manifest)
        proof = mask_proof()
        if args.admit_only:
            if args.validate_source_mask or args.output_manifest:
                raise ReferenceError("--admit-only cannot be combined with another execution option")
            print(json.dumps({
                "schema": SCHEMA, "status": "admitted_cpu_only", "model_launched": False,
                "mps_device_used": False, "upstream_revision": PINNED_REVISION,
                "geometry": {"history_rows": HISTORY_ROWS, "query_rows": QUERY_ROWS,
                             "key_rows": KEY_ROWS, "valid_text_rows": T1_VALID_TEXT_ROWS},
                "authenticated_inputs": sorted(admitted["inputs"]),
                "mask_matrix_sha256": {case: proof["cases"][case]["matrix_sha256"] for case in CASE_ORDER},
            }, sort_keys=True))
            return 0
        if args.validate_source_mask:
            if args.output_manifest:
                raise ReferenceError("--validate-source-mask cannot be combined with --output-manifest")
            try:
                import torch
            except ImportError as error:
                raise ReferenceError(f"PyTorch is required for CPU source-mask validation: {error}") from error
            masks, receipt = build_pinned_source_masks(torch, args.upstream)
            if any(chunk.device.type != "cpu" for chunks in masks.values() for chunk in chunks):
                raise ReferenceError("source-mask validation escaped CPU")
            proof = mask_proof()
            print(json.dumps({
                "schema": SCHEMA, "status": "source_masks_validated_cpu_only",
                "model_launched": False, "mps_device_used": False,
                "upstream_revision": PINNED_REVISION, "receipt": receipt,
                "mask_matrix_sha256": {case: proof["cases"][case]["matrix_sha256"] for case in CASE_ORDER},
            }, sort_keys=True))
            return 0
        manifest_path, payload_path, document, runtime_receipt = _run_source_reference(args, admitted)
        print(json.dumps({
            "schema": SCHEMA,
            "status": "source_reference_completed",
            "model_launched": True,
            "mps_device_used": True,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "payload": str(payload_path),
            "payload_sha256": document["payload"]["sha256"],
            "checkpoint_count": len(document["tensors"]),
            "parent_replay_checkpoints": len(document["execution"]["parent_replay"]["M11"]),
            "source_aa_checkpoints": len(document["execution"]["source_aa"]),
            "runtime_receipt": runtime_receipt,
        }, sort_keys=True))
        return 0
    except ReferenceError as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as error:
        print(json.dumps({
            "schema": SCHEMA, "status": "rejected",
            "error": f"{type(error).__name__}: {error}",
        }, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
