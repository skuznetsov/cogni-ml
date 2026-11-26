#!/usr/bin/env python3
"""Stage two bounded PixelUMM windows through two adjacent BF16 decoder blocks.

Inputs are the saved preview's initial latent/noise passed through the pinned
upstream PixelUMM stem. This does not run DiT, DCAE, or full generation. All
stem and decoder layer math runs on MPS with the authenticated cached BF16
weights. Four layer calls capture six FFN boundaries each.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

SCHEMA = "cogni_ml.pixelumm_ffn_windows_reference/v1"
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
MODEL_SHA256 = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
TORCH_VERSION = "2.6.0"
GRID = (10, 18, 32)
TOKEN_COUNT = math.prod(GRID)
TILE_TOKENS = 8
COMPARISON = {"atol": 2e-5, "rtol": 2e-5}
CAPTURE_ORDER = (
    "post_attn_norm", "gate_proj", "up_proj", "product", "down_proj", "output",
)
WINDOWS = {
    "w0": {"latent_frame_start": 0, "noise_frame_start": 0},
    "w1": {"latent_frame_start": 5, "noise_frame_start": 20},
}
LATENT_FRAME_COUNT = 10
NOISE_FRAME_COUNT = 40
WINDOW_OFFSETS = (
    "latent starts 0/5; noise starts 0/20; each source window independently "
    "follows upstream streaming crop semantics"
)
HIDDEN_SIZE = 4096
FFN_SIZE = 12288
BF16 = "BF16"
F32 = "F32"
GIB = 1024**3
MPS_CAP_BYTES = 12 * GIB
INPUT_MANIFEST = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-first-video-20261028/inputs.json"
)
INPUT_MANIFEST_SHA256 = "5b162b8bf4b28c6cfacd8a3d8c14a7758b699034b2a46e84c224522eb0a75a7b"
INPUT_PAYLOAD_SHA256 = "627cd65f67ed7721c9b0292430bb7ac220a09ea1e5c89a369018dfbbcd920d42"
WEIGHT_CACHE_MANIFEST = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-36-20261027/source/"
    "pixelumm-thirty-six-bf16-cache.json"
)
WEIGHT_CACHE_MANIFEST_SHA256 = "74ee5e84f2b63af0189ffd7be0e4c758b8dfa9ff925e3c9cb389d207ce546dba"
WEIGHT_CACHE_PAYLOAD_SHA256 = "152ba021b52d9546e4c8821b17ab11f67f46486d521b500de3766f14dd2217ca"
WEIGHT_CACHE_PAYLOAD_BYTES = 13_892_143_104
UPSTREAM_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
STEM_SUBSET_PATH = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/source/stem-f32.safetensors"
)
STEM_SUBSET_SHA256 = "7eb73ebf42f6681e21870dc2d2d4b592846ba15ed5bf5f9a6c7a83d39c2d8d04"
STEM_SUBSET_BYTES = 52_511_088
STEM_RECEIPT_PATH = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/source/stem-f32.receipt.json"
)
STEM_RECEIPT_SHA256 = "88d8dbda211d5dbe0c817d1036069e1efd57b6567f0eb01b9ab5b691c3409014"
STEM_WEIGHTS_SHA256 = "7257a5d07644a9fddd38918a127eec94abaeb954b0527f2c33777a61c0f0ec1b"
STEM_F32_BYTES = 52_510_720
LATENT_SOURCE = {
    "dtype": F32, "shape": [1, 32, 30, 9, 16],
    "offset": 0, "nbytes": 552_960,
    "sha256": "61a5ed5ec852b8abf9c3d799df4fca420eae5e320adb7596d2da0a4c1670cf33",
}
NOISE_SOURCE = {
    "dtype": F32, "shape": [1, 3, 120, 288, 512],
    "offset": 14_151_680, "nbytes": 212_336_640,
    "sha256": "064167f7d75c79a72f2896c61c988cebd6f71f7586f1f2177276b7ada69009b3",
}
COS_SOURCE = {
    "dtype": F32, "shape": [1, 5760, 1, 128],
    "offset": 226_488_320, "nbytes": 2_949_120,
    "sha256": "c704e1397520e7a8536994143e56672bea6cab31170b0ffe7a364cb8abb4ac0e",
}
SIN_SOURCE = {
    "dtype": F32, "shape": [1, 5760, 1, 128],
    "offset": 229_437_440, "nbytes": 2_949_120,
    "sha256": "0f3d33d7ce2581e918bc2d91cc73639d9bdf42eb2dfd28a20785b6face97c639",
}
INPUT_PAYLOAD_BYTES = 232_386_560
INPUT_SOURCE_RECORDS = {
    "latent": LATENT_SOURCE,
    "noise": NOISE_SOURCE,
    "cos": COS_SOURCE,
    "sin": SIN_SOURCE,
}
BLOCK_SOURCE_EXPECTED = {
    "0": {
        "selected_block": 0,
        "subset_file": "block0-f32.safetensors",
        "subset_bytes": 771_787_128,
        "subset_sha256": "ae82bb41c121a955493f5b1502dc512ac8aaba05a6cec67e98edeaedfe720485",
        "source_receipt_file": "block0-f32.receipt.json",
        "source_receipt_sha256": "72a55054289f432f5be145f61b09eae1560dcb084c5a4bdd5a9f7df76631902d",
        "source_extent": [102_887_424, 874_673_152],
        "source_f32_bytes": 771_785_728,
        "bf16_bytes": 385_892_864,
    },
    "1": {
        "selected_block": 1,
        "subset_file": "block1-f32.safetensors",
        "subset_bytes": 771_787_128,
        "subset_sha256": "d33d7cfec3d582bdca30029cdab23bba8fec353ed8937fccf8e66648af080361",
        "source_receipt_file": "block1-f32.receipt.json",
        "source_receipt_sha256": "11379cf75c6ac2b76ffe0adcc7132009512b1c628713d8a5e0793de9cd02ce0e",
        "source_extent": [874_673_152, 1_646_458_880],
        "source_f32_bytes": 771_785_728,
        "bf16_bytes": 385_892_864,
    },
}
_BLOCK_WEIGHT_NAMES = (
    "attn.k_norm_hw.weight", "attn.k_norm_t.weight", "attn.k_proj.weight",
    "attn.o_proj.weight", "attn.q_norm_hw.weight", "attn.q_norm_t.weight",
    "attn.q_proj.weight", "attn.v_proj.weight", "input_norm.weight",
    "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "post_attn_norm.weight",
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STAGE_SHAPES = {
    "post_attn_norm": (BF16, (1, TOKEN_COUNT, HIDDEN_SIZE)),
    "gate_proj": (BF16, (1, TOKEN_COUNT, FFN_SIZE)),
    "up_proj": (BF16, (1, TOKEN_COUNT, FFN_SIZE)),
    "product": (BF16, (1, TOKEN_COUNT, FFN_SIZE)),
    "down_proj": (BF16, (1, TOKEN_COUNT, HIDDEN_SIZE)),
    "output": (BF16, (1, TOKEN_COUNT, HIDDEN_SIZE)),
}
_INPUT_SHAPES = {
    "x": (BF16, (1, TOKEN_COUNT, HIDDEN_SIZE)),
    "cos": (F32, (1, TOKEN_COUNT, 1, 128)),
    "sin": (F32, (1, TOKEN_COUNT, 1, 128)),
}
_MANIFEST_KEYS = {
    "schema", "upstream_revision", "upstream_model_sha256", "model_sha256",
    "torch_version", "grid", "comparison", "token_order", "tile_tokens",
    "runtime_dtype", "capture_order", "block_execution", "input_source",
    "input_provenance", "weight_cache", "windows", "payload", "payload_bytes",
    "payload_sha256", "producer",
}


class WindowsReferenceError(ValueError):
    """A pinned source, bounded fixture, or MPS precondition was rejected."""


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _regular_file(path: Path, description: str, *, max_bytes: int | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise WindowsReferenceError(f"{description} must be a regular non-symlink file")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise WindowsReferenceError(f"{description} exceeds its bounded size")
    return path.resolve()


def _tensor_nbytes(dtype: str, shape: Iterable[int]) -> int:
    width = {BF16: 2, F32: 4}.get(dtype)
    dimensions = tuple(shape)
    if width is None or not dimensions or any(type(value) is not int or value <= 0 for value in dimensions):
        raise WindowsReferenceError("unsupported tensor dtype or shape")
    return math.prod(dimensions) * width


def _validate_record(record: Any, dtype: str, shape: tuple[int, ...], offset: int) -> int:
    nbytes = _tensor_nbytes(dtype, shape)
    if (type(record) is not dict
            or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}
            or record.get("dtype") != dtype or record.get("shape") != list(shape)
            or type(record.get("offset")) is not int or record.get("offset") != offset
            or type(record.get("nbytes")) is not int or record.get("nbytes") != nbytes
            or not _valid_digest(record.get("sha256"))):
        raise WindowsReferenceError("tensor record does not match the fixed fixture inventory")
    return offset + nbytes


def _validate_block_cache_record(record: Any, block_name: str) -> None:
    expected = BLOCK_SOURCE_EXPECTED[block_name]
    if type(record) is not dict:
        raise WindowsReferenceError(f"block {block_name} cache receipt is malformed")
    for key, value in expected.items():
        if record.get(key) != value:
            raise WindowsReferenceError(f"block {block_name} cache source identity mismatch: {key}")
    keys = [f"block{block_name}.{name}" for name in _BLOCK_WEIGHT_NAMES]
    if record.get("weight_keys") != keys:
        raise WindowsReferenceError(f"block {block_name} cache weight ownership mismatch")
    chunks = record.get("source_range_chunks")
    if (type(chunks) is not list or len(chunks) != 12
            or any(type(chunk) is not dict
                   or set(chunk) != {"start", "end_inclusive", "sha256"}
                   or type(chunk.get("start")) is not int
                   or type(chunk.get("end_inclusive")) is not int
                   or not _valid_digest(chunk.get("sha256")) for chunk in chunks)):
        raise WindowsReferenceError(f"block {block_name} source chunk receipts are malformed")


def validate_document(document: Mapping[str, Any], *, payload_bytes: int | None = None) -> None:
    """Validate strict experiment provenance and complete packed tensor inventory."""
    if type(document) is not dict or set(document) != _MANIFEST_KEYS:
        raise WindowsReferenceError("FFN windows manifest keys are malformed")
    if (document.get("schema") != SCHEMA
            or document.get("upstream_revision") != UPSTREAM_REVISION
            or document.get("upstream_model_sha256") != MODEL_SHA256
            or document.get("model_sha256") != MODEL_SHA256
            or document.get("torch_version") != TORCH_VERSION
            or document.get("grid") != list(GRID)
            or document.get("comparison") != COMPARISON
            or document.get("token_order") != "tile_major"
            or document.get("tile_tokens") != TILE_TOKENS
            or document.get("runtime_dtype") != BF16
            or document.get("capture_order") != list(CAPTURE_ORDER)):
        raise WindowsReferenceError("source, runtime, shape, tolerance, or capture order mismatch")
    if document.get("block_execution") != {
        "block0_input": "inputs.x",
        "block1_input": "captures.block0.output",
        "edge": "same_window_output_to_next_input",
    }:
        raise WindowsReferenceError("adjacent-block input lineage mismatch")
    if not isinstance(document.get("payload"), str) or Path(document["payload"]).name != document["payload"] or not document["payload"].endswith(".bin"):
        raise WindowsReferenceError("payload must be a sibling .bin basename")
    if not _valid_digest(document.get("payload_sha256")):
        raise WindowsReferenceError("payload SHA-256 is malformed")

    input_source = document.get("input_source")
    if (type(input_source) is not dict
            or input_source.get("manifest_path") != str(INPUT_MANIFEST)
            or input_source.get("manifest_sha256") != INPUT_MANIFEST_SHA256
            or input_source.get("payload_path") != str(INPUT_MANIFEST.parent / "inputs.tensors.bin")
            or input_source.get("payload_bytes") != INPUT_PAYLOAD_BYTES
            or input_source.get("payload_sha256") != INPUT_PAYLOAD_SHA256
            or input_source.get("tensors") != INPUT_SOURCE_RECORDS):
        raise WindowsReferenceError("saved preview input source binding mismatch")

    provenance = document.get("input_provenance")
    stem = provenance.get("stem") if isinstance(provenance, dict) else None
    if (type(provenance) is not dict
            or provenance.get("kind") != "saved_initial_noise_conditioning_upstream_stem"
            or provenance.get("model_inference") is not False
            or provenance.get("interpretation") != (
                "saved initial latent and decoder noise passed through the pinned "
                "upstream stem; not a DCAE-encoded pose or generated DiT latent"
            )
            or provenance.get("x_expression") != (
                "embed(patchify(decoder.noise.to(dtype))) + "
                "_condition_tokens(input.latent.to(dtype))"
            )
            or provenance.get("window_offsets") != WINDOW_OFFSETS
            or type(stem) is not dict
            or stem.get("subset_path") != str(STEM_SUBSET_PATH)
            or stem.get("subset_sha256") != STEM_SUBSET_SHA256
            or stem.get("subset_bytes") != STEM_SUBSET_BYTES
            or stem.get("receipt_path") != str(STEM_RECEIPT_PATH)
            or stem.get("receipt_sha256") != STEM_RECEIPT_SHA256
            or stem.get("weights_sha256") != STEM_WEIGHTS_SHA256):
        raise WindowsReferenceError("input provenance or stem receipt mismatch")

    weight_cache = document.get("weight_cache")
    if (type(weight_cache) is not dict
            or weight_cache.get("manifest_path") != str(WEIGHT_CACHE_MANIFEST)
            or weight_cache.get("manifest_sha256") != WEIGHT_CACHE_MANIFEST_SHA256
            or weight_cache.get("payload_path") != str(WEIGHT_CACHE_MANIFEST.parent / "pixelumm-thirty-six-bf16-cache.bin")
            or weight_cache.get("payload_bytes") != WEIGHT_CACHE_PAYLOAD_BYTES
            or weight_cache.get("payload_sha256") != WEIGHT_CACHE_PAYLOAD_SHA256
            or weight_cache.get("schema") != "cogni_ml.pixelumm_bf16_weight_cache/v1"
            or weight_cache.get("upstream_revision") != UPSTREAM_REVISION
            or weight_cache.get("upstream_model_sha256") != MODEL_SHA256):
        raise WindowsReferenceError("BF16 weight-cache source binding mismatch")
    cache_blocks = weight_cache.get("blocks")
    if type(cache_blocks) is not dict or set(cache_blocks) != {"0", "1"}:
        raise WindowsReferenceError("weight cache must bind the two selected block receipts")
    for index in ("0", "1"):
        _validate_block_cache_record(cache_blocks[index], index)

    windows = document.get("windows")
    if type(windows) is not dict or set(windows) != set(WINDOWS):
        raise WindowsReferenceError("fixture must contain exactly windows w0 and w1")
    cursor = 0
    input_x_hashes = []
    for name in ("w0", "w1"):
        window = windows[name]
        if type(window) is not dict or set(window) != {"source_slices", "inputs", "captures"}:
            raise WindowsReferenceError(f"{name} window keys are malformed")
        slices = window["source_slices"]
        expected_offsets = WINDOWS[name]
        if (type(slices) is not dict
                or set(slices) != {
                    "latent_frame_start", "latent_frame_count", "noise_frame_start",
                    "noise_frame_count", "latent_slice_sha256", "noise_slice_sha256",
                }
                or slices.get("latent_frame_start") != expected_offsets["latent_frame_start"]
                or slices.get("latent_frame_count") != LATENT_FRAME_COUNT
                or slices.get("noise_frame_start") != expected_offsets["noise_frame_start"]
                or slices.get("noise_frame_count") != NOISE_FRAME_COUNT
                or not _valid_digest(slices.get("latent_slice_sha256"))
                or not _valid_digest(slices.get("noise_slice_sha256"))):
            raise WindowsReferenceError(f"{name} source slice lineage mismatch")
        inputs, captures = window["inputs"], window["captures"]
        if type(inputs) is not dict or set(inputs) != set(_INPUT_SHAPES):
            raise WindowsReferenceError(f"{name} inputs are incomplete")
        if type(captures) is not dict or set(captures) != {"block0", "block1"}:
            raise WindowsReferenceError(f"{name} block captures are incomplete")
        for input_name in ("x", "cos", "sin"):
            dtype, shape = _INPUT_SHAPES[input_name]
            cursor = _validate_record(inputs[input_name], dtype, shape, cursor)
        input_x_hashes.append(inputs["x"]["sha256"])
        for block_name in ("block0", "block1"):
            block_capture = captures[block_name]
            if type(block_capture) is not dict or set(block_capture) != set(CAPTURE_ORDER):
                raise WindowsReferenceError(f"{name} {block_name} capture order is incomplete")
            for stage in CAPTURE_ORDER:
                dtype, shape = _STAGE_SHAPES[stage]
                cursor = _validate_record(block_capture[stage], dtype, shape, cursor)
    if (windows["w0"]["source_slices"]["latent_slice_sha256"]
            == windows["w1"]["source_slices"]["latent_slice_sha256"]
            or windows["w0"]["source_slices"]["noise_slice_sha256"]
            == windows["w1"]["source_slices"]["noise_slice_sha256"]
            or input_x_hashes[0] == input_x_hashes[1]):
        raise WindowsReferenceError("the two source windows and stem inputs must differ")

    producer = document.get("producer")
    memory = producer.get("memory_preflight") if isinstance(producer, dict) else None
    if (type(producer) is not dict
            or producer.get("device") != "mps"
            or producer.get("mps_fallback") is not False
            or producer.get("full_generation") is not False
            or producer.get("bounded_layer_calls") != 4
            or producer.get("captures_per_call") != len(CAPTURE_ORDER)
            or producer.get("explicit_mps_synchronizations") != 1
            or producer.get("host_export") != "packed tensor records after one terminal MPS synchronization"
            or type(producer.get("python_version")) is not str
            or type(memory) is not dict):
        raise WindowsReferenceError("producer execution boundary is malformed")
    required_memory = {
        "physical_bytes", "driver_allocated_bytes", "recommended_max_bytes",
        "estimated_peak_bytes", "max_mps_bytes", "required_floor_bytes",
        "headroom_after_cap_bytes", "mps_memory_fraction",
    }
    if set(memory) != required_memory:
        raise WindowsReferenceError("memory preflight record is malformed")
    byte_fields = required_memory - {"mps_memory_fraction"}
    if any(type(memory.get(key)) is not int or memory[key] < 0 for key in byte_fields):
        raise WindowsReferenceError("memory preflight byte values are malformed")
    physical = memory["physical_bytes"]
    allocated = memory["driver_allocated_bytes"]
    recommended = memory["recommended_max_bytes"]
    estimated = memory["estimated_peak_bytes"]
    cap = memory["max_mps_bytes"]
    floor = memory["required_floor_bytes"]
    if (physical <= 0 or allocated > physical or recommended <= 0
            or estimated <= 0 or cap != MPS_CAP_BYTES or estimated > cap
            or floor != physical // 10 or cap > recommended
            or allocated + cap > physical - floor
            or memory["headroom_after_cap_bytes"] != physical - allocated - floor - cap
            or not isinstance(memory["mps_memory_fraction"], (int, float))
            or abs(memory["mps_memory_fraction"] - cap / recommended) > 1e-12):
        raise WindowsReferenceError("memory cap or required 10% floor does not pass")
    expected_payload = cursor
    if type(document.get("payload_bytes")) is not int or document["payload_bytes"] != expected_payload:
        raise WindowsReferenceError("payload byte count differs from the fixed capture inventory")
    if payload_bytes is not None and (type(payload_bytes) is not int or payload_bytes != expected_payload):
        raise WindowsReferenceError("packed payload byte count mismatch")


def _load_pinned_inputs():
    import hy_nomas_generation_inputs as generation_inputs

    manifest_path = _regular_file(INPUT_MANIFEST, "preview input manifest", max_bytes=1024 * 1024)
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_MANIFEST_SHA256:
        raise WindowsReferenceError("saved preview manifest digest mismatch")
    document = json.loads(raw)
    try:
        generation_inputs.validate_document(document)
    except Exception as error:
        raise WindowsReferenceError(f"saved preview input manifest rejected: {error}") from error
    payload_path = _regular_file(manifest_path.parent / document["payload"], "preview input payload")
    if (payload_path.stat().st_size != INPUT_PAYLOAD_BYTES
            or document["payload_sha256"] != INPUT_PAYLOAD_SHA256
            or _sha256_file(payload_path) != INPUT_PAYLOAD_SHA256):
        raise WindowsReferenceError("saved preview payload identity mismatch")
    return document, manifest_path, payload_path, _file_identity(payload_path)


def _load_pinned_cache():
    import hy_nomas_pixelumm_bf16_cache as cache

    path = _regular_file(WEIGHT_CACHE_MANIFEST, "BF16 cache manifest", max_bytes=2 * 1024 * 1024)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WEIGHT_CACHE_MANIFEST_SHA256:
        raise WindowsReferenceError("BF16 cache manifest digest mismatch")
    manifest = json.loads(raw)
    payload_path = _regular_file(path.parent / manifest.get("payload", ""), "BF16 cache payload")
    try:
        cache.validate_cache_file(path)
    except Exception as error:
        raise WindowsReferenceError(f"authenticated BF16 cache rejected: {error}") from error
    if (manifest.get("schema") != "cogni_ml.pixelumm_bf16_weight_cache/v1"
            or manifest.get("upstream_revision") != UPSTREAM_REVISION
            or manifest.get("upstream_model_sha256") != MODEL_SHA256
            or manifest.get("stored_dtype") != "F32"
            or manifest.get("runtime_dtype") != BF16
            or manifest.get("payload_bytes") != WEIGHT_CACHE_PAYLOAD_BYTES
            or manifest.get("payload_sha256") != WEIGHT_CACHE_PAYLOAD_SHA256
            or payload_path.name != "pixelumm-thirty-six-bf16-cache.bin"
            or set(manifest.get("blocks", {})) != {str(index) for index in range(36)}):
        raise WindowsReferenceError("full-depth cache identity or representation mismatch")
    for index in ("0", "1"):
        _validate_block_cache_record(manifest["blocks"][index], index)
    return manifest, path, payload_path, _file_identity(payload_path)


def _validate_stem_source():
    import hy_nomas_pixelumm_checkpoint as checkpoint
    import hy_nomas_pixelumm_real_input as real_input

    subset_path = _regular_file(STEM_SUBSET_PATH, "PixelUMM stem F32 subset")
    receipt_path = _regular_file(STEM_RECEIPT_PATH, "PixelUMM stem receipt", max_bytes=1024 * 1024)
    header_path = _regular_file(real_input.HEADER_PATH, "PixelUMM checkpoint header", max_bytes=8 * 1024 * 1024)
    receipt_raw = receipt_path.read_bytes()
    if hashlib.sha256(receipt_raw).hexdigest() != STEM_RECEIPT_SHA256:
        raise WindowsReferenceError("PixelUMM stem receipt digest mismatch")
    try:
        checked = checkpoint.validate_named_subset_receipt(
            header_path.read_bytes(), checkpoint.parse_header(receipt_raw), subset_path, "stem"
        )
    except Exception as error:
        raise WindowsReferenceError(f"PixelUMM stem source receipt rejected: {error}") from error
    if (checked.get("subset_sha256") != STEM_SUBSET_SHA256
            or checked.get("subset_bytes") != STEM_SUBSET_BYTES
            or real_input._raw_safetensors_payload_sha256(subset_path, STEM_F32_BYTES) != STEM_WEIGHTS_SHA256):
        raise WindowsReferenceError("PixelUMM stem F32 weights changed")
    return {
        "subset_path": str(STEM_SUBSET_PATH),
        "subset_sha256": checked["subset_sha256"],
        "subset_bytes": checked["subset_bytes"],
        "receipt_path": str(STEM_RECEIPT_PATH),
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "weights_sha256": STEM_WEIGHTS_SHA256,
    }


def _read_slice(payload_path: Path, record: Mapping[str, Any], *, channel_count: int,
                source_frames: int, frame_start: int, frame_count: int,
                height: int, width: int) -> tuple[bytearray, str]:
    """Read a temporal slice from channel-major F32 bytes without tensor math."""
    floats_per_frame = height * width
    frame_bytes = floats_per_frame * 4
    data = bytearray(channel_count * frame_count * frame_bytes)
    cursor = 0
    with payload_path.open("rb") as stream:
        for channel in range(channel_count):
            offset = (
                record["offset"]
                + channel * source_frames * frame_bytes
                + frame_start * frame_bytes
            )
            nbytes = frame_count * frame_bytes
            stream.seek(offset)
            chunk = stream.read(nbytes)
            if len(chunk) != nbytes:
                raise WindowsReferenceError("saved input temporal slice is truncated")
            data[cursor:cursor + nbytes] = chunk
            cursor += nbytes
    digest = hashlib.sha256(data).hexdigest()
    return data, digest


def _read_rope(payload_path: Path, record: Mapping[str, Any], torch: Any, device: Any):
    with payload_path.open("rb") as stream:
        stream.seek(record["offset"])
        raw = bytearray(stream.read(record["nbytes"]))
    if len(raw) != record["nbytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise WindowsReferenceError("saved preview RoPE tensor digest mismatch")
    return torch.frombuffer(raw, dtype=torch.float32).reshape(record["shape"]).to(device)


def _make_window_inputs(*, torch: Any, source: Any, stem: Any, plan: Any,
                        input_document: Mapping[str, Any], input_payload: Path,
                        name: str, cos: Any, sin: Any):
    from types import SimpleNamespace

    start = WINDOWS[name]
    latent_record = input_document["tensors"]["input.latent"]
    noise_record = input_document["tensors"]["decoder.noise"]
    latent_raw, latent_sha = _read_slice(
        input_payload, latent_record, channel_count=32, source_frames=30,
        frame_start=start["latent_frame_start"], frame_count=LATENT_FRAME_COUNT,
        height=9, width=16,
    )
    noise_raw, noise_sha = _read_slice(
        input_payload, noise_record, channel_count=3, source_frames=120,
        frame_start=start["noise_frame_start"], frame_count=NOISE_FRAME_COUNT,
        height=288, width=512,
    )
    if len(latent_raw) != 1 * 32 * LATENT_FRAME_COUNT * 9 * 16 * 4:
        raise WindowsReferenceError("latent slice byte count mismatch")
    if len(noise_raw) != 1 * 3 * NOISE_FRAME_COUNT * 288 * 512 * 4:
        raise WindowsReferenceError("noise slice byte count mismatch")
    device = torch.device("mps")
    latent = torch.frombuffer(latent_raw, dtype=torch.float32).reshape(
        1, 32, LATENT_FRAME_COUNT, 9, 16
    ).to(device)
    noise = torch.frombuffer(noise_raw, dtype=torch.float32).reshape(
        1, 3, NOISE_FRAME_COUNT, 288, 512
    ).to(device)
    del latent_raw, noise_raw
    with torch.inference_mode(), torch.autocast(device_type="mps", dtype=torch.bfloat16):
        patch_tokens = stem.embed(source.patchify(noise.to(dtype=stem.embed.weight.dtype)))
        context = SimpleNamespace(cond_proj=stem.cond_proj, arch=source.QWEN3_8B_GEN)
        conditioned = source.DCAEVPixelUMMDecoder._condition_tokens(
            context, latent.to(dtype=stem.embed.weight.dtype)
        )
        thw_tokens = patch_tokens + conditioned
        permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
        x = thw_tokens.index_select(1, permutation)
    del latent, noise, patch_tokens, conditioned, thw_tokens, permutation
    if (x.device.type != "mps" or x.dtype is not torch.bfloat16
            or tuple(x.shape) != (1, TOKEN_COUNT, HIDDEN_SIZE)):
        raise WindowsReferenceError(f"{name} upstream stem output shape/device/dtype mismatch")
    return {
        "source_slices": {
            "latent_frame_start": start["latent_frame_start"],
            "latent_frame_count": LATENT_FRAME_COUNT,
            "noise_frame_start": start["noise_frame_start"],
            "noise_frame_count": NOISE_FRAME_COUNT,
            "latent_slice_sha256": latent_sha,
            "noise_slice_sha256": noise_sha,
        },
        "inputs": {"x": x, "cos": cos, "sin": sin},
        "captures": {"block0": {}, "block1": {}},
    }


def _load_cached_block(source: Any, cache_manifest: Mapping[str, Any],
                       cache_payload_path: Path, index: int, torch: Any):
    prefix = f"block{index}."
    with torch.device("meta"):
        layer = source.GenLayer(source.QWEN3_8B_GEN)
    expected = set(layer.state_dict())
    expected_keys = {prefix + name for name in expected}
    block_keys = set(cache_manifest["blocks"][str(index)]["weight_keys"])
    if expected_keys != block_keys or len(expected) != 13:
        raise WindowsReferenceError(f"cached block {index} weight ownership mismatch")
    state = {}
    with cache_payload_path.open("rb") as stream:
        for name in sorted(expected):
            record = cache_manifest["weights"][prefix + name]
            stream.seek(record["offset"])
            raw = bytearray(stream.read(record["nbytes"]))
            if (len(raw) != record["nbytes"]
                    or hashlib.sha256(raw).hexdigest() != record["sha256"]):
                raise WindowsReferenceError(f"cached block {index} tensor digest mismatch: {name}")
            state[name] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(
                record["shape"]
            ).to("mps")
    try:
        loaded = layer.load_state_dict(state, strict=True, assign=True)
    except Exception as error:
        raise WindowsReferenceError(f"cached block {index} cannot be assigned to GenLayer: {error}") from error
    if loaded.missing_keys or loaded.unexpected_keys:
        raise WindowsReferenceError(f"cached block {index} strict assignment failed")
    layer.eval().requires_grad_(False)
    if any(value.device.type != "mps" or value.dtype is not torch.bfloat16 for value in layer.parameters()):
        raise WindowsReferenceError(f"cached block {index} is not resident as BF16 on MPS")
    return layer


def _memory_preflight(torch: Any, *, max_mps_bytes: int) -> dict[str, int | float]:
    import hy_nomas_pixelumm_reference as reference

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise WindowsReferenceError("PyTorch MPS is required; CPU fallback is forbidden")
    if not all(hasattr(torch.mps, name) for name in (
        "driver_allocated_memory", "recommended_max_memory",
        "set_per_process_memory_fraction", "synchronize",
    )):
        raise WindowsReferenceError("PyTorch MPS lacks required memory and synchronization APIs")
    arch = type("PinnedArch", (), {
        "hidden_size": HIDDEN_SIZE, "intermediate_size": FFN_SIZE,
        "num_heads": 32, "num_kv_heads": 8, "head_dim": 128,
    })()
    estimate = reference._estimated_peak_bytes(
        GRID, arch, layers=4, resident_layers=1, capture_stages=CAPTURE_ORDER,
    )
    # Include a live source noise slice, stem weights/workspace, and allocator
    # margin on top of the reference helper's one-layer/six-capture estimate.
    stem_and_input = (
        STEM_F32_BYTES // 2
        + 3 * NOISE_FRAME_COUNT * 288 * 512 * 4
        + 4 * TOKEN_COUNT * HIDDEN_SIZE * 2
        + 256 * 1024**2
    )
    estimated = estimate + stem_and_input
    try:
        physical = _physical_memory_bytes()
        allocated = int(torch.mps.driver_allocated_memory())
        recommended = int(torch.mps.recommended_max_memory())
    except Exception as error:
        raise WindowsReferenceError(f"cannot query MPS memory preflight: {error}") from error
    floor = physical // 10
    if (type(max_mps_bytes) is not int or max_mps_bytes <= 0
            or estimated > max_mps_bytes
            or max_mps_bytes > recommended
            or allocated + max_mps_bytes > physical - floor):
        raise WindowsReferenceError("MPS cap or estimated peak violates the 10% physical-memory floor")
    fraction = max_mps_bytes / recommended
    try:
        torch.mps.set_per_process_memory_fraction(fraction)
    except Exception as error:
        raise WindowsReferenceError(f"cannot install the MPS process cap: {error}") from error
    return {
        "physical_bytes": physical,
        "driver_allocated_bytes": allocated,
        "recommended_max_bytes": recommended,
        "estimated_peak_bytes": estimated,
        "max_mps_bytes": max_mps_bytes,
        "required_floor_bytes": floor,
        "headroom_after_cap_bytes": physical - allocated - floor - max_mps_bytes,
        "mps_memory_fraction": fraction,
    }


def _physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            ).strip())
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise WindowsReferenceError("cannot determine physical memory floor") from error
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as error:
        raise WindowsReferenceError("cannot determine physical memory floor") from error


def _tensor_record(tensor: Any, *, torch: Any, output_stream: Any,
                   aggregate: Any, offset: int) -> tuple[dict[str, Any], int]:
    if tensor.device.type != "mps" or not tensor.is_contiguous():
        raise WindowsReferenceError("only contiguous MPS captures may be exported")
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        raise WindowsReferenceError("unsupported captured tensor dtype")
    cpu = tensor.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise WindowsReferenceError("captured tensor contains nonfinite values")
    raw = memoryview(cpu.view(torch.uint8).numpy()).cast("B")
    item_hash = hashlib.sha256()
    cursor = 0
    while cursor < raw.nbytes:
        chunk = raw[cursor:cursor + 8 * 1024 * 1024]
        output_stream.write(chunk)
        aggregate.update(chunk)
        item_hash.update(chunk)
        cursor += chunk.nbytes
    dtype = BF16 if tensor.dtype is torch.bfloat16 else F32
    shape = list(tensor.shape)
    record = {
        "dtype": dtype, "shape": shape, "offset": offset,
        "nbytes": raw.nbytes, "sha256": item_hash.hexdigest(),
    }
    del raw, cpu
    return record, offset + record["nbytes"]


def _place_record(windows: dict[str, Any], key: str, record: dict[str, Any]) -> None:
    parts = key.split(".")
    if len(parts) == 4 and parts[0] == "windows" and parts[1] in windows and parts[2] == "inputs":
        windows[parts[1]]["inputs"][parts[3]] = record
        return
    if len(parts) == 5 and parts[0] == "windows" and parts[1] in windows and parts[2] == "captures":
        windows[parts[1]]["captures"][parts[3]][parts[4]] = record
        return
    raise WindowsReferenceError("internal packed tensor key is malformed")


def _publish_artifact(output: Path, payload_tensors: list[tuple[str, Any]],
                      document_base: dict[str, Any], *, torch: Any) -> dict[str, Any]:
    payload_path = output.with_suffix(".bin")
    if output.exists() or output.is_symlink() or payload_path.exists() or payload_path.is_symlink():
        raise WindowsReferenceError("refusing to replace an existing fixture artifact")
    if (output.parent.resolve() != Path(
        "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-ffn-staging-20261028"
    ).resolve() or output.suffix != ".json"):
        raise WindowsReferenceError("output must be a fresh JSON artifact in the pinned staging directory")
    temp_payload = None
    temp_manifest = None
    published_payload = False
    payload_identity = None
    success = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{payload_path.name}.", suffix=".partial",
            dir=output.parent, delete=False,
        ) as stream:
            temp_payload = Path(stream.name)
            aggregate = hashlib.sha256()
            records: dict[str, dict[str, Any]] = {}
            offset = 0
            for key, tensor in payload_tensors:
                record, offset = _tensor_record(
                    tensor, torch=torch, output_stream=stream,
                    aggregate=aggregate, offset=offset,
                )
                records[key] = record
            stream.flush()
            os.fsync(stream.fileno())
        payload_sha = aggregate.hexdigest()
        document = dict(document_base)
        document["payload"] = payload_path.name
        document["payload_bytes"] = offset
        document["payload_sha256"] = payload_sha
        for key, record in records.items():
            _place_record(document["windows"], key, record)
        validate_document(document, payload_bytes=offset)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            prefix=f".{output.name}.", suffix=".partial",
            dir=output.parent, delete=False,
        ) as stream:
            temp_manifest = Path(stream.name)
            json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        payload_identity = (temp_payload.stat().st_dev, temp_payload.stat().st_ino)
        os.link(temp_payload, payload_path)
        published_payload = True
        temp_payload.unlink()
        temp_payload = None
        os.link(temp_manifest, output)
        temp_manifest.unlink()
        temp_manifest = None
        success = True
        return document
    except BaseException:
        raise
    finally:
        for path in (temp_payload, temp_manifest):
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        if published_payload and payload_identity is not None and not success:
            try:
                actual = payload_path.stat()
                if (actual.st_dev, actual.st_ino) == payload_identity:
                    payload_path.unlink()
            except FileNotFoundError:
                pass


def produce(*, output: Path, max_mps_bytes: int = MPS_CAP_BYTES) -> dict[str, Any]:
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in (None, "", "0"):
        raise WindowsReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    if str(torch.__version__) != TORCH_VERSION:
        raise WindowsReferenceError(f"expected PyTorch {TORCH_VERSION}, got {torch.__version__}")

    input_document, input_manifest_path, input_payload_path, input_identity = _load_pinned_inputs()
    cache_manifest, cache_manifest_path, cache_payload_path, cache_identity = _load_pinned_cache()
    stem_source = _validate_stem_source()
    memory = _memory_preflight(torch, max_mps_bytes=max_mps_bytes)

    import hy_nomas_pixelumm_reference as reference
    source, model_sha = reference.load_upstream(UPSTREAM_ROOT)
    if model_sha != MODEL_SHA256:
        raise WindowsReferenceError("pinned upstream PixelUMM model digest mismatch")
    plan, _, _ = reference.make_attention_plan(source, GRID)
    device = torch.device("mps")
    stem = None
    payload_windows: dict[str, dict[str, Any]] = {}
    saved_cos = _read_rope(
        input_payload_path, input_document["tensors"]["decoder.rope_cos"], torch, device
    )
    saved_sin = _read_rope(
        input_payload_path, input_document["tensors"]["decoder.rope_sin"], torch, device
    )
    try:
        import hy_nomas_pixelumm_real_input as real_input
        stem = real_input._load_stem(source, device)
        for name in ("w0", "w1"):
            payload_windows[name] = _make_window_inputs(
                torch=torch, source=source, stem=stem, plan=plan,
                input_document=input_document, input_payload=input_payload_path,
                name=name, cos=saved_cos, sin=saved_sin,
            )
        stem = None

        with torch.inference_mode():
            autocast = torch.autocast(device_type="mps", dtype=torch.bfloat16)
            with autocast:
                for index in (0, 1):
                    layer = _load_cached_block(
                        source, cache_manifest, cache_payload_path, index, torch
                    )
                    for window_name in ("w0", "w1"):
                        window = payload_windows[window_name]
                        layer_input = (
                            window["inputs"]["x"] if index == 0
                            else window["captures"]["block0"]["output"]
                        )
                        result, captures = reference.run_layer_reference(
                            layer, layer_input, window["inputs"]["cos"],
                            window["inputs"]["sin"], plan,
                            capture_stages=CAPTURE_ORDER,
                        )
                        if (set(captures) != set(CAPTURE_ORDER)
                                or captures["output"] is not result):
                            raise WindowsReferenceError("upstream layer capture inventory or output edge mismatch")
                        if (result.device.type != "mps" or result.dtype is not torch.bfloat16
                                or tuple(result.shape) != (1, TOKEN_COUNT, HIDDEN_SIZE)):
                            raise WindowsReferenceError(f"block {index} output shape/device/dtype mismatch")
                        window["captures"][f"block{index}"] = captures
                    del layer
        # This is the sole explicit synchronization for all neural work. The
        # serializer then performs the bounded terminal tensor readbacks.
        torch.mps.synchronize()
    except WindowsReferenceError:
        raise
    except Exception as error:
        raise WindowsReferenceError(
            f"bounded MPS stem/reference execution failed without CPU fallback: {error}"
        ) from error
    finally:
        stem = None

    input_source = {
        "manifest_path": str(INPUT_MANIFEST),
        "manifest_sha256": INPUT_MANIFEST_SHA256,
        "payload_path": str(INPUT_MANIFEST.parent / "inputs.tensors.bin"),
        "payload_bytes": INPUT_PAYLOAD_BYTES,
        "payload_sha256": INPUT_PAYLOAD_SHA256,
        "tensors": INPUT_SOURCE_RECORDS,
    }
    input_provenance = {
        "kind": "saved_initial_noise_conditioning_upstream_stem",
        "model_inference": False,
        "interpretation": (
            "saved initial latent and decoder noise passed through the pinned "
            "upstream stem; not a DCAE-encoded pose or generated DiT latent"
        ),
        "x_expression": (
            "embed(patchify(decoder.noise.to(dtype))) + "
            "_condition_tokens(input.latent.to(dtype))"
        ),
        "window_offsets": WINDOW_OFFSETS,
        "stem": stem_source,
    }
    weight_cache = {
        "manifest_path": str(WEIGHT_CACHE_MANIFEST),
        "manifest_sha256": WEIGHT_CACHE_MANIFEST_SHA256,
        "payload_path": str(
            WEIGHT_CACHE_MANIFEST.parent / "pixelumm-thirty-six-bf16-cache.bin"
        ),
        "payload_bytes": WEIGHT_CACHE_PAYLOAD_BYTES,
        "payload_sha256": WEIGHT_CACHE_PAYLOAD_SHA256,
        "schema": cache_manifest["schema"],
        "upstream_revision": cache_manifest["upstream_revision"],
        "upstream_model_sha256": cache_manifest["upstream_model_sha256"],
        "blocks": {index: cache_manifest["blocks"][index] for index in ("0", "1")},
    }
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_sha256": MODEL_SHA256,
        "model_sha256": model_sha,
        "torch_version": str(torch.__version__),
        "grid": list(GRID),
        "comparison": dict(COMPARISON),
        "token_order": "tile_major",
        "tile_tokens": TILE_TOKENS,
        "runtime_dtype": BF16,
        "capture_order": list(CAPTURE_ORDER),
        "block_execution": {
            "block0_input": "inputs.x",
            "block1_input": "captures.block0.output",
            "edge": "same_window_output_to_next_input",
        },
        "input_source": input_source,
        "input_provenance": input_provenance,
        "weight_cache": weight_cache,
        "windows": payload_windows,
        "producer": {
            "device": "mps",
            "mps_fallback": False,
            "full_generation": False,
            "bounded_layer_calls": 4,
            "captures_per_call": len(CAPTURE_ORDER),
            "explicit_mps_synchronizations": 1,
            "host_export": "packed tensor records after one terminal MPS synchronization",
            "python_version": platform.python_version(),
            "memory_preflight": memory,
        },
    }
    ordered_tensors = []
    for window_name in ("w0", "w1"):
        window = payload_windows[window_name]
        for input_name in ("x", "cos", "sin"):
            ordered_tensors.append((
                f"windows.{window_name}.inputs.{input_name}",
                window["inputs"][input_name],
            ))
        for block_name in ("block0", "block1"):
            for stage in CAPTURE_ORDER:
                ordered_tensors.append((
                    f"windows.{window_name}.captures.{block_name}.{stage}",
                    window["captures"][block_name][stage],
                ))
    if (_file_identity(input_payload_path) != input_identity
            or _file_identity(cache_payload_path) != cache_identity):
        raise WindowsReferenceError("a pinned source payload changed during MPS execution")
    if hashlib.sha256(input_manifest_path.read_bytes()).hexdigest() != INPUT_MANIFEST_SHA256:
        raise WindowsReferenceError("saved preview manifest changed during MPS execution")
    if hashlib.sha256(cache_manifest_path.read_bytes()).hexdigest() != WEIGHT_CACHE_MANIFEST_SHA256:
        raise WindowsReferenceError("BF16 cache manifest changed during MPS execution")
    return _publish_artifact(output.absolute(), ordered_tensors, base, torch=torch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh JSON path in the pinned external staging directory")
    parser.add_argument("--max-mps-bytes", type=int, default=MPS_CAP_BYTES)
    args = parser.parse_args(argv)
    try:
        document = produce(output=args.output, max_mps_bytes=args.max_mps_bytes)
    except (WindowsReferenceError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FFN windows reference rejected: {error}\n")
    manifest_path = args.output.absolute()
    print(json.dumps({
        "schema": document["schema"],
        "output": str(manifest_path),
        "payload": str(manifest_path.with_suffix(".bin")),
        "payload_bytes": document["payload_bytes"],
        "payload_sha256": document["payload_sha256"],
        "manifest_sha256": _sha256_file(manifest_path),
        "estimated_peak_bytes": document["producer"]["memory_preflight"]["estimated_peak_bytes"],
        "max_mps_bytes": document["producer"]["memory_preflight"]["max_mps_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
