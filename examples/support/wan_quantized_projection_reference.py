#!/usr/bin/env python3
"""Prepare a bounded Wan FFN packed-weight projection fixture.

This script reads only blocks.0.ffn.0.{weight,bias} from the pinned BF16
checkpoint shards, quantizes one matrix on CPU with the canonical ggml and
repository QBit producers, then performs every dequantized-weight conversion
and F.linear reference projection on MPS. It writes a new immutable fixture
directory and never loads or invokes a Wan model.

The four source rows are the captured upstream ``expected.ffn_mod`` values
from the seeded Wan block-0 fixture. ``--rows`` controls periodic input-row
tiling on MPS; it does not turn the seeded fixture into a real-video capture.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F


WAN_REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
WAN_CHECKPOINT_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
SOURCE_ROWS = 4
INPUT_DIM = 5120
OUTPUT_DIM = 13824
MAX_ROWS = 4480
BLOCK_SIZE = 256
Q4_K_BLOCK_BYTES = 144
QBIT_P4_BLOCK_BYTES = 8 + 4 * 32
QBIT_P5_BLOCK_BYTES = 8 + 5 * 32

DEFAULT_CHECKPOINT_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/checkpoints/wan"
)
DEFAULT_CAPTURE_MANIFEST = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-official-block0/wan-official.json"
)
DEFAULT_UPSTREAM_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit"
)
DEFAULT_CODEC_SOURCE = Path(__file__).resolve().parents[2] / "src/ml/gguf/qwen_qbit_gaussian_codec.cr"
DEFAULT_GGML_ROOT = Path("/opt/mldata_ssd/SrcArchives/AI/llama.cpp")
# The canonical reference quantizer is exported by base itself. Loading cpu
# would unnecessarily depend on a build-time rpath after moving the checkout.
DEFAULT_GGML_LIBRARY = DEFAULT_GGML_ROOT / "cmake-build-debug/bin/libggml-base.dylib"

CAPTURE_MANIFEST_SHA256 = "fb7943d637bc8a6853707cd2ed284595e6744c4368083c36993fd1dd26d78da9"
CHECKPOINT_INDEX_SHA256 = "b5ea0e7789d82c00bd25cfbeb940a1675e048986b1527198251da74d700508d2"
CHECKPOINT_EXTRACTION_SHA256 = "c2238de5e5200c2514230f60d52eb96faa4a517a99ac3d5505fe4617210d2f0c"
SOURCE_WEIGHT_SHA256 = "cd1037a3eb9c98cf5baed570e6825e7bb2463165563707cf7f1c3b367afca557"
SOURCE_BIAS_SHA256 = "6fd35eb26bebf22c9cdffcb9049fa350f3cb83732e529034074cb4605c2a0ff3"
QBIT_CODEC_SHA256 = "6397a42e05a44bf6c1a5b588aa02e197de617252ecb6bab0044f8146efd7eab5"
GGML_LIBRARY_SHA256 = "f3905d482cfb1f51b930b4023462c6495397a0f5a2bdfc2bbf7ca4bd88f6f75b"
GGML_QUANTS_SOURCE_SHA256 = "07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d"
GGML_COMMON_HEADER_SHA256 = "af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0"
GGML_REVISION = "7e4c0a96880dae4fc4268ad441f8a6446bd5460a"

UPSTREAM_SOURCE_SHA256 = {
    "wan_modules_model.py": "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432",
    "wan_modules_s2v_model_s2v.py": "91fb14df44edab2d01b5634420b463c5f1563aecdf6dac905da6b3acac3145cc",
    "wan_modules_s2v_audio_utils.py": "76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd",
    "wan_modules_s2v_s2v_utils.py": "e45ae6c274b17e4fdddcb0439b598897a37679b8934ebd71f87255f92d5f0b2f",
    "wan_modules_s2v_auxi_blocks.py": "6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6",
    "diffusers-normalization-v0.31.0.py": "7f2d2086f669e19bc008d71743f3963017274fbc6ad86d0ab9bf18db211e7c5a",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{description} SHA-256 mismatch: expected {expected}, found {actual} ({path})"
        )
    return actual


def require_mps(tensor: torch.Tensor, label: str) -> torch.Tensor:
    if tensor.device.type != "mps":
        raise RuntimeError(f"{label} must be on MPS, found {tensor.device}")
    return tensor


def raw_array_file(
    path: Path, array: np.ndarray, dtype_name: str | None = None
) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    with path.open("wb") as stream:
        contiguous.tofile(stream)
    return file_record(path, dtype_name or contiguous.dtype.str, list(contiguous.shape))


def file_record(path: Path, dtype: str, shape: list[int]) -> dict[str, Any]:
    return {
        "path": path.name,
        "dtype": dtype,
        "shape": shape,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tensor_to_bf16_raw(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu":
        raise RuntimeError(f"Expected CPU BF16 tensor for {path.name}")
    raw_u16 = tensor.contiguous().view(torch.uint16).numpy()
    little_endian = np.asarray(raw_u16, dtype="<u2")
    return raw_array_file(path, little_endian, "BF16")


def load_captured_input(manifest_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    require_sha256(manifest_path, CAPTURE_MANIFEST_SHA256, "captured Wan manifest")
    manifest = json.loads(manifest_path.read_text())
    upstream = manifest.get("upstream", {})
    config = manifest.get("config", {})
    if upstream.get("revision") != WAN_REVISION:
        raise RuntimeError(f"Unexpected Wan source revision: {upstream.get('revision')}")
    if upstream.get("seed") != 7031:
        raise RuntimeError(f"Unexpected captured input seed: {upstream.get('seed')}")
    if config.get("dim") != INPUT_DIM or config.get("ffn_dim") != OUTPUT_DIM:
        raise RuntimeError(f"Unexpected captured Wan geometry: {config}")
    captured = manifest.get("expected", {}).get("ffn_mod")
    if not isinstance(captured, list) or len(captured) != SOURCE_ROWS * INPUT_DIM:
        raise RuntimeError("expected.ffn_mod is not the pinned four-row projection input")
    source = np.asarray(captured, dtype="<f4").reshape(SOURCE_ROWS, INPUT_DIM)
    if not np.isfinite(source).all():
        raise RuntimeError("Captured projection input contains non-finite values")
    provenance = {
        "kind": "upstream_expected_ffn_mod_seeded_capture",
        "manifest_path": str(manifest_path),
        "manifest_sha256": CAPTURE_MANIFEST_SHA256,
        "upstream_revision": upstream["revision"],
        "capture_seed": upstream["seed"],
        "capture_fixture_mode": upstream.get("fixture_mode"),
        "capture_hook": "expected.ffn_mod (input to upstream block.ffn module)",
        "checkpoint_repo": upstream.get("checkpoint_source", {}).get("repo"),
        "checkpoint_revision": upstream.get("checkpoint_source", {}).get("revision"),
        "checkpoint_sidecar_sha256": manifest.get("weights_file", {}).get("sha256"),
        "real_video_derived": False,
    }
    return source, provenance


def verify_upstream_sources(upstream_root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, expected in UPSTREAM_SOURCE_SHA256.items():
        actual[name] = require_sha256(upstream_root / name, expected, f"pinned Wan source {name}")
    return actual


def checkpoint_tensor_source(checkpoint_root: Path) -> tuple[Path, dict[str, Any]]:
    index_path = checkpoint_root / "diffusion_pytorch_model.safetensors.index.json"
    extraction_path = checkpoint_root / "block0-official-extraction.json"
    require_sha256(index_path, CHECKPOINT_INDEX_SHA256, "Wan checkpoint index")
    require_sha256(extraction_path, CHECKPOINT_EXTRACTION_SHA256, "Wan extraction manifest")
    index = json.loads(index_path.read_text())
    weight_name = "blocks.0.ffn.0.weight"
    bias_name = "blocks.0.ffn.0.bias"
    weight_shard = index.get("weight_map", {}).get(weight_name)
    bias_shard = index.get("weight_map", {}).get(bias_name)
    if not weight_shard or weight_shard != bias_shard:
        raise RuntimeError("Wan FFN projection weight and bias must share one indexed shard")
    shard_path = checkpoint_root / weight_shard
    extraction = json.loads(extraction_path.read_text())
    records = {record["name"]: record for record in extraction.get("tensors", [])}
    if records.get(weight_name, {}).get("sha256") != SOURCE_WEIGHT_SHA256:
        raise RuntimeError("Pinned extraction manifest has an unexpected FFN weight hash")
    if records.get(bias_name, {}).get("sha256") != SOURCE_BIAS_SHA256:
        raise RuntimeError("Pinned extraction manifest has an unexpected FFN bias hash")
    metadata = {
        "index_path": str(index_path),
        "index_sha256": CHECKPOINT_INDEX_SHA256,
        "extraction_manifest_path": str(extraction_path),
        "extraction_manifest_sha256": CHECKPOINT_EXTRACTION_SHA256,
        "repo": "Wan-AI/Wan2.2-S2V-14B",
        "revision": WAN_CHECKPOINT_REVISION,
        "weight_tensor": weight_name,
        "bias_tensor": bias_name,
        "shard_path": str(shard_path),
        "shard_bytes": shard_path.stat().st_size,
        "expected_weight_raw_sha256": SOURCE_WEIGHT_SHA256,
        "expected_bias_raw_sha256": SOURCE_BIAS_SHA256,
        "dtype": "BF16",
    }
    return shard_path, metadata


def load_source_tensors(shard_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    with safe_open(str(shard_path), framework="pt", device="cpu") as source:
        weight_slice = source.get_slice("blocks.0.ffn.0.weight")
        bias_slice = source.get_slice("blocks.0.ffn.0.bias")
        if weight_slice.get_shape() != [OUTPUT_DIM, INPUT_DIM]:
            raise RuntimeError(f"Unexpected weight shape: {weight_slice.get_shape()}")
        if bias_slice.get_shape() != [OUTPUT_DIM]:
            raise RuntimeError(f"Unexpected bias shape: {bias_slice.get_shape()}")
        if weight_slice.get_dtype() != "BF16" or bias_slice.get_dtype() != "BF16":
            raise RuntimeError("The pinned FFN weight and bias must both be BF16")
        weight = source.get_tensor("blocks.0.ffn.0.weight").contiguous()
        bias = source.get_tensor("blocks.0.ffn.0.bias").contiguous()
    return weight, bias


def write_ggml_q4_k(
    weight: torch.Tensor,
    destination: Path,
    library_path: Path,
    row_chunk: int,
) -> dict[str, Any]:
    require_sha256(library_path, GGML_LIBRARY_SHA256, "pinned ggml reference-quantizer library")
    library = ctypes.CDLL(str(library_path))
    try:
        quantize_row = library.quantize_row_q4_K_ref
    except AttributeError as error:
        raise RuntimeError("Pinned ggml library lacks quantize_row_q4_K_ref") from error
    quantize_row.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    quantize_row.restype = None
    blocks_per_row = INPUT_DIM // BLOCK_SIZE
    row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES
    with destination.open("wb") as stream:
        for first in range(0, OUTPUT_DIM, row_chunk):
            last = min(OUTPUT_DIM, first + row_chunk)
            rows = weight[first:last].to(dtype=torch.float32).contiguous().numpy()
            packed = np.empty((last - first, row_bytes), dtype=np.uint8)
            for row_index, source_row in enumerate(rows):
                quantize_row(
                    source_row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    packed[row_index].ctypes.data_as(ctypes.c_void_p),
                    INPUT_DIM,
                )
            packed.tofile(stream)
    return {
        "producer": "ggml quantize_row_q4_K_ref",
        "library_path": str(library_path),
        "library_sha256": GGML_LIBRARY_SHA256,
        "source_revision": GGML_REVISION,
        "quantizer_source_sha256": GGML_QUANTS_SOURCE_SHA256,
        "common_header_sha256": GGML_COMMON_HEADER_SHA256,
        "block_size": BLOCK_SIZE,
        "block_bytes": Q4_K_BLOCK_BYTES,
        "values_per_row": INPUT_DIM,
    }


def parse_qbit_source(codec_path: Path) -> tuple[np.ndarray, dict[int, np.ndarray], str]:
    digest = require_sha256(codec_path, QBIT_CODEC_SHA256, "repository QBit codec source")
    source = codec_path.read_text()

    levels_match = re.search(
        r"private POSITIVE_LEVELS\s*=\s*\[(.*?)\]\s+of Float32", source, re.S
    )
    if not levels_match:
        raise RuntimeError("Could not find repository POSITIVE_LEVELS table")
    level_tokens = re.findall(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)_f32",
        levels_match.group(1),
    )
    levels = np.asarray([np.float32(float(token)) for token in level_tokens], dtype=np.float32)
    if levels.shape != (128,):
        raise RuntimeError(f"Expected 128 repository LloydMax levels, found {levels.size}")

    centroid_bits: dict[int, np.ndarray] = {}
    for precision in (4, 5):
        name = f"P{precision}_POSITIVE_PREFIX_CENTROID_BITS"
        match = re.search(rf"private {name}\s*=\s*\[(.*?)\]\s+of UInt32", source, re.S)
        if not match:
            raise RuntimeError(f"Could not find repository {name}")
        tokens = re.findall(r"0x([0-9a-fA-F]+)_u32", match.group(1))
        bits = np.asarray([int(token, 16) for token in tokens], dtype="<u4")
        expected_count = 1 << (precision - 1)
        if bits.shape != (expected_count,):
            raise RuntimeError(f"Expected {expected_count} {name} values, found {bits.size}")
        centroid_bits[precision] = bits
    return levels, centroid_bits, digest


def qbit_lut(precision: int, centroid_bits: np.ndarray) -> np.ndarray:
    suffix = 8 - precision
    prefix_width = 1 << suffix
    values = centroid_bits.view("<f4")
    lut = np.zeros(256, dtype="<f4")
    for prefix, value in enumerate(values):
        first = prefix * prefix_width
        end = first + prefix_width
        positive_codes = np.arange(first, end, dtype=np.uint16)
        lut[positive_codes] = value
        lut[255 - positive_codes] = -value
    return lut


def exact_block_moments(block: np.ndarray) -> tuple[np.float32, np.float32]:
    total = 0.0
    for value in block:
        total += float(value)
    mean64 = total / BLOCK_SIZE
    squared = 0.0
    for value in block:
        delta = float(value) - mean64
        squared += delta * delta
    sigma64 = math.sqrt(squared / BLOCK_SIZE)
    return np.float32(mean64), np.float32(sigma64)


def encode_qbit_chunk(
    values: np.ndarray,
    precision: int,
    boundaries: np.ndarray,
) -> bytes:
    if values.size % BLOCK_SIZE:
        raise RuntimeError("QBit encoding chunk must end at a 256-value block boundary")
    values = np.asarray(values, dtype=np.float32).reshape(-1, BLOCK_SIZE)
    stride = 8 + precision * 32
    payload = np.empty((values.shape[0], stride), dtype=np.uint8)
    positions = np.arange(BLOCK_SIZE, dtype=np.uint16)
    bit_index = 255 - (positions ^ 7)
    byte_order = (bit_index // 8).astype(np.int64)
    bit_mask = (1 << (bit_index % 8)).astype(np.uint8)
    for block_index, block in enumerate(values):
        mean, sigma = exact_block_moments(block)
        payload[block_index, :8] = np.frombuffer(
            np.asarray([mean, sigma], dtype="<f4").tobytes(), dtype=np.uint8
        )
        if sigma == np.float32(0.0):
            normalized = np.zeros(BLOCK_SIZE, dtype=np.float32)
        else:
            normalized = np.asarray((block - mean) / sigma, dtype=np.float32)
        raw = np.searchsorted(boundaries, np.abs(normalized), side="left").astype(np.uint8)
        negative = np.signbit(normalized)
        raw[negative] = (255 - raw[negative]).astype(np.uint8)
        for plane in range(precision):
            plane_bits = ((raw >> (7 - plane)) & 1).astype(np.uint8)
            plane_bytes = np.packbits(plane_bits, bitorder="little")[::-1]
            payload[
                block_index,
                8 + plane * 32 : 8 + (plane + 1) * 32,
            ] = plane_bytes
    return payload.tobytes()


def write_qbit_payloads(
    weight: torch.Tensor,
    output_dir: Path,
    levels: np.ndarray,
    centroid_bits: dict[int, np.ndarray],
    row_chunk: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    boundaries = np.asarray(
        (levels[:-1] + levels[1:]) * np.float32(0.5), dtype=np.float32
    )
    block_count = OUTPUT_DIM * INPUT_DIM // BLOCK_SIZE
    if OUTPUT_DIM * INPUT_DIM % BLOCK_SIZE:
        raise RuntimeError("Wan projection weight must have complete QBit blocks")
    lut_records: dict[int, dict[str, Any]] = {}
    luts: dict[int, np.ndarray] = {}
    for precision in (4, 5):
        lut = qbit_lut(precision, centroid_bits[precision])
        luts[precision] = lut
        lut_records[precision] = raw_array_file(
            output_dir / f"lut_p{precision}.f32", lut, "F32"
        )

    payload_records: dict[int, dict[str, Any]] = {}
    row_block_count = INPUT_DIM // BLOCK_SIZE
    for precision in (4, 5):
        path = output_dir / f"qbit_p{precision}.bin"
        with path.open("wb") as stream:
            for first in range(0, OUTPUT_DIM, row_chunk):
                last = min(OUTPUT_DIM, first + row_chunk)
                chunk = weight[first:last].to(dtype=torch.float32).contiguous().numpy()
                payload = encode_qbit_chunk(chunk.reshape(-1), precision, boundaries)
                stream.write(payload)
        expected_size = block_count * (8 + precision * 32)
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Unexpected QBit p{precision} size: {path.stat().st_size}")
        payload_records[precision] = file_record(
            path,
            "QBitBlockBytes",
            [OUTPUT_DIM, row_block_count, 8 + precision * 32],
        )
    return payload_records, lut_records


def read_q4_integer_fields(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, Q4_K_BLOCK_BYTES)
    block_count = blocks.shape[0]
    half_bits = np.ascontiguousarray(blocks[:, :4]).view("<u2").reshape(block_count, 2)
    packed_scales = blocks[:, 4 : 4 + 12]
    scales = np.empty((block_count, 8), dtype=np.uint8)
    mins = np.empty((block_count, 8), dtype=np.uint8)
    for index in range(8):
        if index < 4:
            scales[:, index] = packed_scales[:, index] & 0x3F
            mins[:, index] = packed_scales[:, index + 4] & 0x3F
        else:
            scales[:, index] = (packed_scales[:, index + 4] & 0x0F) | (
                (packed_scales[:, index - 4] >> 6) << 4
            )
            mins[:, index] = (packed_scales[:, index + 4] >> 4) | (
                (packed_scales[:, index] >> 6) << 4
            )
    packed_quants = blocks[:, 16:].reshape(block_count, 4, 32)
    codes = np.concatenate(
        (packed_quants & 0x0F, packed_quants >> 4), axis=2
    ).reshape(block_count, BLOCK_SIZE)
    return half_bits, scales, mins, codes


def decode_q4_k_mps(payload: bytes, row_count: int) -> torch.Tensor:
    half_bits, scales, mins, codes = read_q4_integer_fields(payload)
    block_count = row_count * (INPUT_DIM // BLOCK_SIZE)
    if half_bits.shape[0] != block_count:
        raise RuntimeError("Q4_K payload length does not match requested rows")
    device = torch.device("mps")
    half_words = torch.from_numpy(half_bits.astype("<u2", copy=False).view("<i2").copy())
    half_mps = require_mps(half_words.to(device=device).view(torch.float16), "Q4_K half metadata")
    half_values = half_mps.to(dtype=torch.float32).reshape(block_count, 2)
    d = half_values[:, 0]
    dmin = half_values[:, 1]
    scale_mps = require_mps(torch.from_numpy(scales.copy()).to(device=device).float(), "Q4_K scales")
    min_mps = require_mps(torch.from_numpy(mins.copy()).to(device=device).float(), "Q4_K mins")
    code_mps = require_mps(torch.from_numpy(codes.copy()).to(device=device).float(), "Q4_K codes")
    q = code_mps.reshape(block_count, 8, 32)
    dscale = (d[:, None] * scale_mps)
    dminimum = (dmin[:, None] * min_mps)
    decoded = dscale[:, :, None] * q - dminimum[:, :, None]
    return require_mps(decoded.reshape(row_count, INPUT_DIM), "Q4_K dequantized weight")


def read_qbit_integer_fields(
    payload: bytes,
    precision: int,
) -> tuple[np.ndarray, np.ndarray]:
    stride = 8 + precision * 32
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, stride)
    moments = np.ascontiguousarray(blocks[:, :8]).view("<f4").reshape(-1, 2).copy()
    raw_codes = np.zeros((blocks.shape[0], BLOCK_SIZE), dtype=np.uint8)
    for plane in range(precision):
        plane_data = blocks[:, 8 + plane * 32 : 8 + (plane + 1) * 32]
        natural_bits = np.unpackbits(plane_data[:, ::-1], axis=1, bitorder="little")
        raw_codes |= natural_bits.astype(np.uint8) << (7 - plane)
    return moments, raw_codes


def decode_qbit_mps(payload: bytes, precision: int, lut: np.ndarray, row_count: int) -> torch.Tensor:
    moments, raw_codes = read_qbit_integer_fields(payload, precision)
    expected_blocks = row_count * (INPUT_DIM // BLOCK_SIZE)
    if moments.shape[0] != expected_blocks:
        raise RuntimeError(f"QBit p{precision} payload length does not match requested rows")
    device = torch.device("mps")
    moments_mps = require_mps(torch.from_numpy(moments.copy()).to(device=device), "QBit moments")
    lut_mps = require_mps(torch.from_numpy(lut.copy()).to(device=device), "QBit LUT")
    codes_mps = require_mps(
        torch.from_numpy(raw_codes.astype(np.int64)).to(device=device), "QBit raw codes"
    )
    reconstruction = torch.index_select(lut_mps, 0, codes_mps.reshape(-1))
    reconstruction = reconstruction.reshape(expected_blocks, BLOCK_SIZE)
    decoded = moments_mps[:, 0:1] + moments_mps[:, 1:2] * reconstruction
    return require_mps(decoded.reshape(row_count, INPUT_DIM), f"QBit p{precision} dequantized weight")


def expand_input_on_mps(source: np.ndarray, rows: int) -> torch.Tensor:
    source_mps = require_mps(
        torch.from_numpy(np.ascontiguousarray(source)).to(device="mps", dtype=torch.float32),
        "captured projection input",
    )
    row_indices = torch.arange(rows, dtype=torch.int64, device="mps") % SOURCE_ROWS
    return require_mps(torch.index_select(source_mps, 0, row_indices), "expanded projection input")


def projection_reference(
    name: str,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    packed_path: Path | None,
    output_dir: Path,
    rows: int,
    input_mps: torch.Tensor,
    lut: np.ndarray | None,
    weight_row_chunk: int,
) -> dict[str, Any]:
    decoded_path = output_dir / f"decoded_{name}.f32"
    projection_path = output_dir / f"projection_{name}.f32"
    expected_shape = [rows, OUTPUT_DIM]
    projection = np.memmap(
        projection_path, dtype="<f4", mode="w+", shape=(rows, OUTPUT_DIM)
    )
    if name == "bf16":
        decoded_stream: BinaryIO | None = decoded_path.open("wb")
    else:
        if packed_path is None:
            raise RuntimeError(f"Missing packed payload for {name}")
        packed_stream = packed_path.open("rb")
        decoded_stream = decoded_path.open("wb")
    bias_mps = require_mps(source_bias.to(device="mps", dtype=torch.float32), "projection bias")
    decoded_weight_mps = require_mps(
        torch.empty((OUTPUT_DIM, INPUT_DIM), dtype=torch.float32, device="mps"),
        f"{name} full decoded weight matrix",
    )
    try:
        for first in range(0, OUTPUT_DIM, weight_row_chunk):
            last = min(OUTPUT_DIM, first + weight_row_chunk)
            count = last - first
            if name == "bf16":
                weight_mps = require_mps(
                    source_weight[first:last].to(device="mps", dtype=torch.float32),
                    "BF16 decoded weight",
                )
            elif name == "q4_k":
                row_bytes = (INPUT_DIM // BLOCK_SIZE) * Q4_K_BLOCK_BYTES
                packed_stream.seek(first * row_bytes)
                packed = packed_stream.read(count * row_bytes)
                if len(packed) != count * row_bytes:
                    raise RuntimeError("Short read from Q4_K payload")
                weight_mps = decode_q4_k_mps(packed, count)
            elif name in ("qbit_p4", "qbit_p5"):
                precision = 4 if name.endswith("p4") else 5
                stride = 8 + precision * 32
                row_bytes = (INPUT_DIM // BLOCK_SIZE) * stride
                packed_stream.seek(first * row_bytes)
                packed = packed_stream.read(count * row_bytes)
                if len(packed) != count * row_bytes:
                    raise RuntimeError(f"Short read from QBit p{precision} payload")
                if lut is None:
                    raise RuntimeError(f"Missing p{precision} reconstruction LUT")
                weight_mps = decode_qbit_mps(packed, precision, lut, count)
            else:
                raise RuntimeError(f"Unknown reference variant: {name}")

            weight_cpu = require_mps(weight_mps, f"{name} weight chunk").to("cpu").contiguous()
            weight_cpu.numpy().astype("<f4", copy=False).tofile(decoded_stream)
            decoded_weight_mps[first:last] = weight_mps
        result = F.linear(input_mps, decoded_weight_mps, bias_mps)
        require_mps(result, f"{name} F.linear result")
        if result.dtype != torch.float32 or tuple(result.shape) != tuple(expected_shape):
            raise RuntimeError(f"Unexpected {name} projection result: {result.dtype} {result.shape}")
        projection[:, :] = result.to("cpu").contiguous().numpy()
        projection.flush()
    finally:
        if decoded_stream is not None:
            decoded_stream.close()
        if name != "bf16":
            packed_stream.close()
        del decoded_weight_mps
        del projection
    return {
        "packed_weight_file": "weight.bf16" if name == "bf16" else packed_path.name,
        "decoded_weight_file": decoded_path.name,
        "expected_projection_file": projection_path.name,
        "bias_file": "bias.f32",
        "decoded_weight": file_record(decoded_path, "F32", [OUTPUT_DIM, INPUT_DIM]),
        "expected_projection": file_record(projection_path, "F32", expected_shape),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_fixture(args: argparse.Namespace) -> Path:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must remain disabled")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for all dequantization and projection references")
    if args.rows not in (SOURCE_ROWS, MAX_ROWS):
        raise ValueError(f"--rows must be one of {SOURCE_ROWS} or {MAX_ROWS}")
    if args.row_chunk < 1 or args.weight_row_chunk < 1:
        raise ValueError("chunk sizes must be positive")
    output_path = args.output.expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(f"Fixture output already exists; immutable output is never overwritten: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent)
    )
    try:
        input_source, input_provenance = load_captured_input(args.capture_manifest)
        upstream_hashes = verify_upstream_sources(args.upstream_root)
        shard_path, checkpoint_provenance = checkpoint_tensor_source(args.checkpoint_root)
        source_weight, source_bias = load_source_tensors(shard_path)

        weight_raw_record = tensor_to_bf16_raw(stage_path / "weight.bf16", source_weight)
        bias_raw_record = tensor_to_bf16_raw(stage_path / "bias.bf16", source_bias)
        if weight_raw_record["sha256"] != SOURCE_WEIGHT_SHA256:
            raise RuntimeError("Checkpoint FFN weight bytes do not match the pinned extraction")
        if bias_raw_record["sha256"] != SOURCE_BIAS_SHA256:
            raise RuntimeError("Checkpoint FFN bias bytes do not match the pinned extraction")
        bias_f32 = source_bias.to(dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False)
        bias_f32_record = raw_array_file(stage_path / "bias.f32", bias_f32, "F32")
        input_record = raw_array_file(stage_path / "input.f32", input_source, "F32")

        q4_path = stage_path / "q4_k.bin"
        print("Preparing canonical ggml Q4_K payload...", flush=True)
        q4_provenance = write_ggml_q4_k(
            source_weight, q4_path, args.ggml_library, args.row_chunk
        )
        q4_record = file_record(
            q4_path,
            "Q4_K",
            [OUTPUT_DIM, INPUT_DIM // BLOCK_SIZE, Q4_K_BLOCK_BYTES],
        )

        print("Preparing repository QBit p4/p5 payloads and LUTs...", flush=True)
        levels, centroid_bits, codec_hash = parse_qbit_source(args.codec_source)
        qbit_records, lut_records = write_qbit_payloads(
            source_weight, stage_path, levels, centroid_bits, args.row_chunk
        )

        input_mps = expand_input_on_mps(input_source, args.rows)
        variants: dict[str, dict[str, Any]] = {}
        for name, payload_path, lut in (
            ("bf16", None, None),
            ("q4_k", q4_path, None),
            ("qbit_p4", stage_path / "qbit_p4.bin", np.fromfile(stage_path / "lut_p4.f32", dtype="<f4")),
            ("qbit_p5", stage_path / "qbit_p5.bin", np.fromfile(stage_path / "lut_p5.f32", dtype="<f4")),
        ):
            print(f"Computing MPS decoded-weight and F.linear reference: {name}...", flush=True)
            record = projection_reference(
                name,
                source_weight,
                source_bias,
                payload_path,
                stage_path,
                args.rows,
                input_mps,
                lut,
                args.weight_row_chunk,
            )
            record["packed_weight"] = (
                weight_raw_record
                if name == "bf16"
                else q4_record
                if name == "q4_k"
                else qbit_records[4]
                if name == "qbit_p4"
                else qbit_records[5]
            )
            record["lut_file"] = None if name not in ("qbit_p4", "qbit_p5") else f"lut_p{int(name[-1])}.f32"
            record["lut"] = None if name not in ("qbit_p4", "qbit_p5") else lut_records[int(name[-1])]
            record["bias"] = bias_f32_record
            record["decoder"] = {
                "device": "MPS",
                "fallback_disabled": True,
                "operation_order": (
                    "source BF16 bits -> F32 on MPS"
                    if name == "bf16"
                    else "F16(d,dmin)->F32 on MPS; dscale=d*scale; dminimum=dmin*min; value=dscale*code-dminimum"
                    if name == "q4_k"
                    else "raw code from transposed MSB planes; value=mean+sigma*LUT[raw_code]"
                ),
            }
            variants[name] = record

        tiled_rows = args.rows
        manifest = {
            "schema": "wan_quantized_projection_reference_v1",
            "created_by": Path(__file__).name,
            "geometry": {
                "source_rows": SOURCE_ROWS,
                "rows": tiled_rows,
                "expected_projection_rows": tiled_rows,
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "input_layout": "row_major",
                "input_expansion": {
                    "operation": "periodic row tile",
                    "source_file": "input.f32",
                    "source_rows": SOURCE_ROWS,
                    "expanded_rows": tiled_rows,
                    "repetitions": tiled_rows // SOURCE_ROWS,
                    "partial_repetition_rows": tiled_rows % SOURCE_ROWS,
                    "row_mapping": "expanded_input[i] = input[i % source_rows]",
                    "performed_on": "MPS",
                },
            },
            "input": input_record,
            "input_provenance": input_provenance,
            "checkpoint_provenance": checkpoint_provenance,
            "source_hashes": {
                "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
                "checkpoint_index_sha256": CHECKPOINT_INDEX_SHA256,
                "checkpoint_extraction_manifest_sha256": CHECKPOINT_EXTRACTION_SHA256,
                "checkpoint_weight_tensor_sha256": SOURCE_WEIGHT_SHA256,
                "checkpoint_bias_tensor_sha256": SOURCE_BIAS_SHA256,
                "upstream_revision": WAN_REVISION,
                "upstream_sources": upstream_hashes,
                "ggml_library_sha256": GGML_LIBRARY_SHA256,
                "ggml_quantizer_source_sha256": GGML_QUANTS_SOURCE_SHA256,
                "ggml_common_header_sha256": GGML_COMMON_HEADER_SHA256,
                "qbit_codec_source_sha256": codec_hash,
                "script_sha256": sha256_file(Path(__file__).resolve()),
            },
            "source_tensor": {
                "weight_name": "blocks.0.ffn.0.weight",
                "bias_name": "blocks.0.ffn.0.bias",
                "weight_shape": [OUTPUT_DIM, INPUT_DIM],
                "bias_shape": [OUTPUT_DIM],
                "dtype": "BF16",
                "weight_file": weight_raw_record,
                "bias_bf16_file": bias_raw_record,
                "bias_f32_file": bias_f32_record,
            },
            "producers": {
                "q4_k": q4_provenance,
                "qbit": {
                    "producer": "src/ml/gguf/qwen_qbit_gaussian_codec.cr exact tables and layout",
                    "offline_encoding_device": "CPU",
                    "block_size": BLOCK_SIZE,
                    "mean_sigma": "sequential Float64 accumulation, then Float32 mean and sqrt(variance)",
                    "normalization": "Float32 (value-mean)/sigma",
                    "quantizer": "nearest reflection-symmetric 256-level LloydMax using Float32 midpoint binary search",
                    "sign_mapping": "negative raw = 0xff - positive raw",
                    "plane_order": "most-significant prefix planes p4 or p5",
                    "plane_layout": "repository transposed bit order: bit_index=255-(within^7), byte=bit_index//8, mask=1<<(bit_index%8)",
                    "prefix_lut": "exact UInt32 centroid bit patterns parsed from pinned Crystal source",
                    "payload_files": {"p4": qbit_records[4], "p5": qbit_records[5]},
                    "lut_files": {"p4": lut_records[4], "p5": lut_records[5]},
                },
            },
            "variants": variants,
            "comparison": {
                "dtype": "F32",
                "atol": 2e-5,
                "rtol": 2e-5,
                "predicate": "finite(actual) && abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)",
                "scope": "native kernel parity against the same-format MPS-decoded F.linear reference",
                "lossy_format_vs_bf16": "separate quality comparison; not covered by this parity tolerance",
            },
            "oracle": {
                "device": "MPS",
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "python_version": sys.version,
                "fallback_disabled": True,
                "projection": "torch.nn.functional.linear(input, decoded_weight, bias)",
                "output_rows": tiled_rows,
                "projection_call_shape": [tiled_rows, OUTPUT_DIM],
                "weight_decode_row_chunk": args.weight_row_chunk,
            },
            "fixture_policy": {
                "immutable": True,
                "output_directory_must_not_exist": True,
                "one_projection_matrix_only": True,
                "full_wan_model_loaded": False,
                "input_is_real_video_derived": False,
                "source_rows_are_periodically_tiled": True,
            },
        }
        write_manifest(stage_path / "manifest.json", manifest)
        if os.path.lexists(output_path):
            raise FileExistsError(f"Fixture output appeared while preparing: {output_path}")
        os.replace(stage_path, output_path)
        return output_path
    except BaseException:
        if stage_path.exists():
            shutil.rmtree(stage_path)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new immutable fixture directory")
    parser.add_argument(
        "--rows",
        type=int,
        choices=(SOURCE_ROWS, MAX_ROWS),
        default=MAX_ROWS,
        help=f"MPS projection rows ({SOURCE_ROWS} smoke rows or {MAX_ROWS} benchmark rows)",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--capture-manifest", type=Path, default=DEFAULT_CAPTURE_MANIFEST)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--codec-source", type=Path, default=DEFAULT_CODEC_SOURCE)
    parser.add_argument("--ggml-library", type=Path, default=DEFAULT_GGML_LIBRARY)
    parser.add_argument("--row-chunk", type=int, default=16, help="offline CPU quantization rows per chunk")
    parser.add_argument(
        "--weight-row-chunk",
        type=int,
        default=16,
        help="MPS decoded-weight rows staged per chunk before one full-geometry F.linear",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    result = build_fixture(args)
    print(result / "manifest.json")


if __name__ == "__main__":
    main()
