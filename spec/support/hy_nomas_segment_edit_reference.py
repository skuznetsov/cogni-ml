"""CPU-only verifier for the reduced GPU segment-noise fork probe.

The expected bytes encode the deterministic 6-resident/3-active A/B/C fixture,
with B regenerated from F at the after-A boundary. This checks gather/scatter,
prefix equivalence, continuation into C, and the reported encode boundary; it
does not qualify neural inference or checkpoint serialization.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any


RECEIPT_SCHEMA = "cogni_ml.hy_nomas_segment_edit_timeline_probe/v1"
REPORT_SCHEMA = "cogni_ml.hy_nomas_segment_edit_timeline_reference/v1"
BASE_CONTRACT = "cogni_ml.hy_nomas_segment_edit_timeline_probe/base-after-a/v1"
UPSTREAM = {
    "revision": "911cea6458a89c2663513ff7099561ec190dd491",
    "source_sha256": {
        "hymm/pipeline/causal_ar.py": "f9aae49b14becb87d958665f763e510c28eeaaad9292f00ac511d1146202949f",
        "hymm/models/dit/nomas_cpt.py": "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee",
    },
    "contract_scope": (
        "channel-first noise slicing and global RoPE offset source anchors; "
        "no DiT parity claim"
    ),
}
RUN_SEGMENTS = 3
RESIDENT_SEGMENTS = 6
TARGET_SEGMENT = 1
SOURCE_SEGMENT = 5
CHANNELS = 2
TOTAL_FRAMES = 18
SEGMENT_FRAMES = 3
HEIGHT = 1
WIDTH = 2
ROPE_HEAD_DIM = 2
SPATIAL = HEIGHT * WIDTH
SEGMENT_ELEMENTS = CHANNELS * SEGMENT_FRAMES * SPATIAL
TIMELINE_ELEMENTS = CHANNELS * TOTAL_FRAMES * SPATIAL
ROPE_ELEMENTS_PER_SEGMENT = SEGMENT_FRAMES * SPATIAL * ROPE_HEAD_DIM
F32 = struct.Struct("<f")
F32_BYTES = F32.size
ARTIFACTS = {
    "after_a_timeline": ("after-a-timeline.f32le", TIMELINE_ELEMENTS,
                         [1, CHANNELS, TOTAL_FRAMES, HEIGHT, WIDTH]),
    "timeline": ("timeline.f32le", TIMELINE_ELEMENTS, [1, CHANNELS, TOTAL_FRAMES, HEIGHT, WIDTH]),
    "b_noise": ("b-noise.f32le", SEGMENT_ELEMENTS, [1, CHANNELS, SEGMENT_FRAMES, HEIGHT, WIDTH]),
    "b_cosine": ("b-cosine.f32le", ROPE_ELEMENTS_PER_SEGMENT,
                      [SEGMENT_FRAMES * HEIGHT * WIDTH, ROPE_HEAD_DIM]),
    "b_sine": ("b-sine.f32le", ROPE_ELEMENTS_PER_SEGMENT,
                    [SEGMENT_FRAMES * HEIGHT * WIDTH, ROPE_HEAD_DIM]),
    "c_noise": ("c-noise.f32le", SEGMENT_ELEMENTS, [1, CHANNELS, SEGMENT_FRAMES, HEIGHT, WIDTH]),
    "c_cosine": ("c-cosine.f32le", ROPE_ELEMENTS_PER_SEGMENT,
                      [SEGMENT_FRAMES * HEIGHT * WIDTH, ROPE_HEAD_DIM]),
    "c_sine": ("c-sine.f32le", ROPE_ELEMENTS_PER_SEGMENT,
                    [SEGMENT_FRAMES * HEIGHT * WIDTH, ROPE_HEAD_DIM]),
}
NOISE_EDIT = {
    "schema": "cogni_ml.hy_nomas_segment_noise_edit/v1",
    "run_segments": RUN_SEGMENTS,
    "target_segment": TARGET_SEGMENT,
    "source_segment": SOURCE_SEGMENT,
    "resident_segments": RESIDENT_SEGMENTS,
}
CHECKS = {
    "clean_a_retained": True,
    "target_b_noise_from_f": True,
    "target_b_scatter_only_to_b": True,
    "target_b_rope_uses_b": True,
    "c_noise_uses_c": True,
    "c_rope_uses_c": True,
    "future_suffix_untouched": True,
    "after_a_snapshot_preserved": True,
}
METRICS = {
    "encode_allocations": 0,
    "encode_readbacks": 0,
    "preterminal_waits": 0,
    "terminal_waits": 1,
    "postterminal_readbacks": 8,
}


class SegmentEditReferenceError(ValueError):
    """The receipt or an artifact violates the admitted reduced-probe contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SegmentEditReferenceError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SegmentEditReferenceError("receipt must be a regular non-symlink file")
    before = path.stat(follow_symlinks=False)
    if before.st_size > 1024 * 1024:
        raise SegmentEditReferenceError("receipt exceeds the 1 MiB limit")
    raw = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if _identity(before) != _identity(after) or len(raw) != after.st_size:
        raise SegmentEditReferenceError("receipt changed while being read")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SegmentEditReferenceError("receipt is not valid UTF-8 JSON") from error
    if type(document) is not dict:
        raise SegmentEditReferenceError("receipt must be a JSON object")
    return document


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _read_artifact(root: Path, receipt: dict[str, Any], name: str) -> bytes:
    filename, elements, _shape = ARTIFACTS[name]
    records = receipt.get("artifacts")
    if type(records) is not dict or set(records) != set(ARTIFACTS):
        raise SegmentEditReferenceError("artifacts must name exactly the eight admitted outputs")
    record = records.get(name)
    if type(record) is not dict or set(record) != {"path", "sha256", "bytes", "shape"}:
        raise SegmentEditReferenceError(f"{name} artifact record has the wrong fields")
    if record.get("path") != filename:
        raise SegmentEditReferenceError(f"{name} artifact must be the sibling {filename}")
    expected_bytes = elements * F32_BYTES
    if type(record.get("bytes")) is not int or record["bytes"] != expected_bytes:
        raise SegmentEditReferenceError(f"{name} artifact byte extent is wrong")
    if record.get("shape") != ARTIFACTS[name][2]:
        raise SegmentEditReferenceError(f"{name} artifact shape is wrong")
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise SegmentEditReferenceError(f"{name} artifact must be a regular non-symlink file")
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise SegmentEditReferenceError(f"{name} artifact has the wrong type or byte extent")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _identity(before) != _identity(opened):
                raise SegmentEditReferenceError(f"{name} artifact changed before it was opened")
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while chunk := stream.read(64 * 1024):
                    chunks.append(chunk)
                    digest.update(chunk)
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.stat(follow_symlinks=False)
    except OSError as error:
        raise SegmentEditReferenceError(f"cannot read {name} artifact") from error
    payload = b"".join(chunks)
    if (_identity(before) != _identity(after_open) or _identity(before) != _identity(after_path)
            or len(payload) != expected_bytes):
        raise SegmentEditReferenceError(f"{name} artifact changed while being read")
    if type(record.get("sha256")) is not str or record["sha256"] != digest.hexdigest():
        raise SegmentEditReferenceError(f"{name} artifact SHA-256 does not match its receipt")
    return payload


def _f32_values(payload: bytes, label: str) -> list[float]:
    if len(payload) % F32_BYTES:
        raise SegmentEditReferenceError(f"{label} is not a whole F32 array")
    values = [value[0] for value in struct.iter_unpack("<f", payload)]
    if any(not math.isfinite(value) for value in values):
        raise SegmentEditReferenceError(f"{label} contains a non-finite value")
    return values


def _pack(values: list[float]) -> bytes:
    return b"".join(F32.pack(value) for value in values)


def _initial_timeline() -> bytes:
    return _pack([(index + 1) / 16.0 for index in range(TIMELINE_ELEMENTS)])


def _gather_segment(payload: bytes, segment_index: int) -> bytes:
    values = _f32_values(payload, "fixture timeline")
    channel_stride = TOTAL_FRAMES * SPATIAL
    segment_stride = SEGMENT_FRAMES * SPATIAL
    gathered = []
    for channel in range(CHANNELS):
        start = channel * channel_stride + segment_index * segment_stride
        gathered.extend(values[start:start + segment_stride])
    return _pack(gathered)


def _expected_after_a(initial: bytes) -> bytes:
    values = _f32_values(initial, "initial timeline")
    segment_stride = SEGMENT_FRAMES * SPATIAL
    for channel in range(CHANNELS):
        for local in range(segment_stride):
            segment_element = channel * segment_stride + local
            index = channel * TOTAL_FRAMES * SPATIAL + local
            values[index] = 0.5 + segment_element * 0.125
    return _pack(values)


def _expected_after_b(after_a: bytes) -> bytes:
    values = _f32_values(after_a, "after-A timeline")
    segment_stride = SEGMENT_FRAMES * SPATIAL
    for channel in range(CHANNELS):
        for local in range(segment_stride):
            segment_element = channel * segment_stride + local
            index = channel * TOTAL_FRAMES * SPATIAL + TARGET_SEGMENT * segment_stride + local
            values[index] = 4.0 + segment_element * 0.0625
    return _pack(values)


def _rope_bank(kind: str) -> bytes:
    if kind == "cosine":
        values = [(index + 1) / 16.0 for index in range(
            RESIDENT_SEGMENTS * ROPE_ELEMENTS_PER_SEGMENT)]
    else:
        values = [(index + 65) / 32.0 for index in range(
            RESIDENT_SEGMENTS * ROPE_ELEMENTS_PER_SEGMENT)]
    return _pack(values)


def _rope_slice(bank: bytes, segment_index: int) -> bytes:
    start = segment_index * ROPE_ELEMENTS_PER_SEGMENT * F32_BYTES
    stop = start + ROPE_ELEMENTS_PER_SEGMENT * F32_BYTES
    return bank[start:stop]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise SegmentEditReferenceError("unsupported receipt schema")
    if receipt.get("upstream") != UPSTREAM:
        raise SegmentEditReferenceError("upstream source pins do not match the local source inventory")
    if receipt.get("scope") != (
        "synthetic GPU after-A boundary, B-from-F gather/scatter, and C prepare; "
        "no neural model or decoder"
    ):
        raise SegmentEditReferenceError("receipt scope is not the admitted synthetic probe")
    expected_scalars = {
        "resident_segments": RESIDENT_SEGMENTS,
        "run_segments": RUN_SEGMENTS,
        "checkpoint_boundary_next_segment": TARGET_SEGMENT,
        "target_segment": TARGET_SEGMENT,
        "source_segment": SOURCE_SEGMENT,
    }
    for key, expected in expected_scalars.items():
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            raise SegmentEditReferenceError(f"receipt {key} is outside the admitted fork geometry")
    if receipt.get("noise_edit") != NOISE_EDIT:
        raise SegmentEditReferenceError("noise edit identity does not match target B from donor F")
    if receipt.get("checks") != CHECKS:
        raise SegmentEditReferenceError("receipt is missing a required fork-boundary check")
    if receipt.get("metrics") != METRICS:
        raise SegmentEditReferenceError("encode/readback/wait metrics violate the probe contract")
    if receipt.get("passed") is not True:
        raise SegmentEditReferenceError("native probe did not report passed=true")

    identity = json.dumps(NOISE_EDIT, separators=(",", ":"), ensure_ascii=False).encode()
    if receipt.get("noise_edit_identity_sha256") != _sha(identity):
        raise SegmentEditReferenceError("noise edit identity SHA-256 is invalid")
    if receipt.get("boundary_contract_sha256") != _sha(BASE_CONTRACT.encode()):
        raise SegmentEditReferenceError("boundary contract must remain the exact parent identity")
    fork_contract = {
        "schema": "cogni_ml.hy_nomas_edited_sequence_boundary/v1",
        "base_contract_sha256": _sha(BASE_CONTRACT.encode()),
        "noise_edit": NOISE_EDIT,
    }
    expected_fork_sha = _sha(json.dumps(
        fork_contract, separators=(",", ":"), ensure_ascii=False
    ).encode())
    if receipt.get("fork_contract_sha256") != expected_fork_sha:
        raise SegmentEditReferenceError("post-B boundary contract is missing fork lineage")


def validate_probe_directory(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Authenticate output and verify the exact B-from-F fork and C continuation."""
    root_arg = Path(directory)
    if root_arg.is_symlink() or not root_arg.is_dir():
        raise SegmentEditReferenceError("probe directory must be a regular non-symlink directory")
    root = root_arg.resolve(strict=True)
    receipt = _read_json(root / "receipt.json")
    _validate_receipt(receipt)
    artifacts = {
        name: _read_artifact(root, receipt, name)
        for name in ARTIFACTS
    }

    initial = _initial_timeline()
    expected_after_a = _expected_after_a(initial)
    if artifacts["after_a_timeline"] != expected_after_a:
        raise SegmentEditReferenceError("after-A snapshot must preserve clean A and initial B-F")
    expected_timeline = _expected_after_b(expected_after_a)
    if artifacts["timeline"] != expected_timeline:
        raise SegmentEditReferenceError("final timeline must scatter regenerated B only")
    expected_noise = _gather_segment(initial, SOURCE_SEGMENT)
    if artifacts["b_noise"] != expected_noise:
        raise SegmentEditReferenceError("B noise does not equal the untouched F donor slice")
    cosine_bank = _rope_bank("cosine")
    sine_bank = _rope_bank("sine")
    expected_b_cosine = _rope_slice(cosine_bank, TARGET_SEGMENT)
    expected_b_sine = _rope_slice(sine_bank, TARGET_SEGMENT)
    if artifacts["b_cosine"] != expected_b_cosine or artifacts["b_sine"] != expected_b_sine:
        raise SegmentEditReferenceError("B must retain its own global RoPE while using F's noise")
    if expected_b_cosine == _rope_slice(cosine_bank, SOURCE_SEGMENT) or \
            expected_b_sine == _rope_slice(sine_bank, SOURCE_SEGMENT):
        raise SegmentEditReferenceError("fixture failed to distinguish target and donor RoPE")
    expected_c_noise = _gather_segment(initial, 2)
    if artifacts["c_noise"] != expected_c_noise:
        raise SegmentEditReferenceError("C must gather its own untouched initial noise")
    expected_c_cosine = _rope_slice(cosine_bank, 2)
    expected_c_sine = _rope_slice(sine_bank, 2)
    if artifacts["c_cosine"] != expected_c_cosine or artifacts["c_sine"] != expected_c_sine:
        raise SegmentEditReferenceError("C prepare must retain C's own global RoPE")
    if expected_c_noise == expected_noise:
        raise SegmentEditReferenceError("fixture must distinguish C noise from F donor noise")

    return {
        "schema": REPORT_SCHEMA,
        "segments": {"target": TARGET_SEGMENT, "noise_source": SOURCE_SEGMENT,
                     "active_run": RUN_SEGMENTS, "resident": RESIDENT_SEGMENTS},
        "checks": dict(CHECKS),
        "metrics": dict(METRICS),
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in artifacts.items()
        },
        "passed": True,
    }
