#!/usr/bin/env python3
"""Produce a two-block, actual-weight Hy-Nomas MPS reference.

This producer starts from the authenticated saved text-conditioning fixture;
it does not rerun Qwen or the text refiner. It runs the pinned upstream
``MMDoubleStreamBlock`` source for actual checkpoint blocks 0 and 1 in order,
passing block 0's MPS outputs directly to block 1. Parameters hold F32 values
rounded through BF16, while their storage and all computation remain F32. The
MPS SDPA adapter is a bounded surrogate for Magi/Flash attention, and the
saved video input is a seeded hidden state rather than a full video-latent
conditioning path.
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
from typing import Any

import hy_nomas_reference as block_reference
import hy_nomas_text_conditioning_reference as conditioning_reference


PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
DEFAULT_CONDITIONING_REFERENCE = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-text-conditioning-20261025/fixtures/text-conditioning-v2.json"
)
DEFAULT_BLOCK0_WEIGHTS = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024/weights/distilled-block0-f32.safetensors"
)
DEFAULT_BLOCK0_MANIFEST = Path(str(DEFAULT_BLOCK0_WEIGHTS) + ".manifest.json")
DEFAULT_BLOCK1_WEIGHTS = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-pretrained-chain-20261025/weights/distilled-block1-f32.safetensors"
)
DEFAULT_BLOCK1_MANIFEST = Path(str(DEFAULT_BLOCK1_WEIGHTS) + ".manifest.json")

CONDITIONING_REFERENCE_SHA256 = "167dd498c75efa396ca1f8ae33b68fa34368d2e4c96feb5d4de2d89c2666992e"
CONDITIONING_SIDECAR_SHA256 = "1668ef1699da4d66cc8d88005fb7c3e3a288feba53999d3019a81af5dfb7b849"
CONDITIONING_SIDECAR_BYTES = 15_605_764

BLOCK_PINS: dict[str, dict[str, Any]] = {
    "block0": {
        "block_index": 0,
        "file_bytes": 604_231_456,
        "file_sha256": "6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1",
        "manifest_sha256": "c7afec8da5ac22a37d8e38b42c9f5ecdabab7929b5c40da9260e7c30c9c73016",
        "source_payload_sha256": "4711e53c3b9ab29abe552f25b95d3cea6f9d75df74e098d7a6acc880befae28b",
        "source_header_sha256": None,
        "range_start_inclusive": 968_270_320,
        "range_end_inclusive": 1_572_497_903,
    },
    "block1": {
        "block_index": 1,
        "file_bytes": 604_231_488,
        "file_sha256": "026568ecff75c5c4a4d771d313469a9245eb115c34f9117af7ac21290369651c",
        "manifest_sha256": "2903edf38d98e4006414ab58d39bfd36ced879fe8d41d218430bd8336bbd3d22",
        "source_payload_sha256": "d23df0c9abdcd65c4a858d6f1df7b6f07741e8c5253512c32756aeb047fd2e94",
        "source_header_sha256": "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200",
        "range_start_inclusive": 1_572_497_904,
        "range_end_inclusive": 2_176_725_487,
    },
}

SOURCE_REPOSITORY = "tencent/Hy-Nomas"
SOURCE_CHECKPOINT_REVISION = "bac719a80c8d02f91c485fca804bb8221e149b70"
SOURCE_CHECKPOINT_PATH = "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
SOURCE_CHECKPOINT_LFS_SHA256 = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"
SOURCE_CHECKPOINT_BYTES = 34_213_991_280

SCHEMA = "hy-nomas-pretrained-serial-reference-v1"
SIDE_CAR_LIMIT = 64 * 1024 * 1024
ATOL = 2e-5
RTOL = 2e-5
GEOMETRY = {
    "batches": 1,
    "text_tokens": 23,
    "dim": 2048,
    "heads": 16,
    "head_dim": 128,
    "ffn_dim": 8192,
    "video_tokens": 6,
    "segments": 1,
    "chunk_size_tokens": 0,
    "text_seg_pos_emb_num": 0,
}
PRECISION = {
    "source_weight_dtype": "F32",
    "parameter_values": "BF16-rounded values",
    "parameter_storage": "F32",
    "compute": "F32",
    "attention": "PyTorch MPS SDPA F32",
    "weight_rounding": "source F32 values converted F32->BF16->F32 on MPS",
    "native_bf16_parameter_storage": False,
    "native_bf16_compute": False,
    "cuda_bf16_parity": False,
}
CONDITIONING_SOURCE = {
    "reference_sha256": CONDITIONING_REFERENCE_SHA256,
    "tensor_file_name": "text-conditioning-v2.tensors.f32le",
    "tensor_sha256": CONDITIONING_SIDECAR_SHA256,
    "tensor_nbytes": CONDITIONING_SIDECAR_BYTES,
    "source_schema": "hy-nomas-text-conditioning-reference-v1",
    "upstream_revision": PINNED_REVISION,
}
_CONDITIONING_SHAPES = {
    "input.video": [1, 6, 2048],
    "input.rope_cos": [6, 128],
    "input.rope_sin": [6, 128],
    "refiner.projected_text": [1, 23, 2048],
    "refiner.vec": [1, 2048],
}

_BLOCK0_NAMES = (
    "joint.attention",
    "joint.key",
    "joint.query",
    "joint.value",
    "img.after_attention",
    "img.attn.k",
    "img.attn.k_norm",
    "img.attn.k_rope",
    "img.attn.modulated",
    "img.attn.norm",
    "img.attn.projected",
    "img.attn.q",
    "img.attn.q_norm",
    "img.attn.q_rope",
    "img.attn.raw",
    "img.attn.v",
    "img.ff.activated",
    "img.ff.hidden",
    "img.ff.modulated",
    "img.ff.norm",
    "img.ff.projected",
    "img.modulation",
    "img.output",
    "txt.after_attention",
    "txt.attn.k",
    "txt.attn.k_norm",
    "txt.attn.modulated",
    "txt.attn.norm",
    "txt.attn.projected",
    "txt.attn.q",
    "txt.attn.q_norm",
    "txt.attn.raw",
    "txt.attn.v",
    "txt.ff.activated",
    "txt.ff.hidden",
    "txt.ff.modulated",
    "txt.ff.norm",
    "txt.ff.projected",
    "txt.modulation",
    "txt.output",
)
BLOCK_CAPTURE_NAMES = _BLOCK0_NAMES

_CAPTURE_SHAPES: dict[str, list[int]] = {
    "joint.attention": [1, 29, 2048],
    "joint.key": [1, 29, 16, 128],
    "joint.query": [1, 29, 16, 128],
    "joint.value": [1, 29, 16, 128],
    "img.after_attention": [1, 6, 2048],
    "img.attn.k": [1, 6, 16, 128],
    "img.attn.k_norm": [1, 6, 16, 128],
    "img.attn.k_rope": [1, 6, 16, 128],
    "img.attn.modulated": [1, 6, 2048],
    "img.attn.norm": [1, 6, 2048],
    "img.attn.projected": [1, 6, 2048],
    "img.attn.q": [1, 6, 16, 128],
    "img.attn.q_norm": [1, 6, 16, 128],
    "img.attn.q_rope": [1, 6, 16, 128],
    "img.attn.raw": [1, 6, 2048],
    "img.attn.v": [1, 6, 16, 128],
    "img.ff.activated": [1, 6, 8192],
    "img.ff.hidden": [1, 6, 8192],
    "img.ff.modulated": [1, 6, 2048],
    "img.ff.norm": [1, 6, 2048],
    "img.ff.projected": [1, 6, 2048],
    "img.modulation": [1, 12288],
    "img.output": [1, 6, 2048],
    "txt.after_attention": [1, 23, 2048],
    "txt.attn.k": [1, 23, 16, 128],
    "txt.attn.k_norm": [1, 23, 16, 128],
    "txt.attn.modulated": [1, 23, 2048],
    "txt.attn.norm": [1, 23, 2048],
    "txt.attn.projected": [1, 23, 2048],
    "txt.attn.q": [1, 23, 16, 128],
    "txt.attn.q_norm": [1, 23, 16, 128],
    "txt.attn.raw": [1, 23, 2048],
    "txt.attn.v": [1, 23, 16, 128],
    "txt.ff.activated": [1, 23, 8192],
    "txt.ff.hidden": [1, 23, 8192],
    "txt.ff.modulated": [1, 23, 2048],
    "txt.ff.norm": [1, 23, 2048],
    "txt.ff.projected": [1, 23, 2048],
    "txt.modulation": [1, 12288],
    "txt.output": [1, 23, 2048],
}

EXPECTED_TENSOR_SHAPES: dict[str, list[int]] = {
    "input.img": [1, 6, 2048],
    "input.txt": [1, 23, 2048],
    "input.vec": [1, 2048],
    "input.vec_txt": [1, 2048],
    "input.rope_cos": [6, 128],
    "input.rope_sin": [6, 128],
}
for _block_name in ("block0", "block1"):
    for _capture_name, _capture_shape in _CAPTURE_SHAPES.items():
        EXPECTED_TENSOR_SHAPES[f"{_block_name}.{_capture_name}"] = _capture_shape


class ReferenceError(RuntimeError):
    """A fail-closed input, provenance, execution, or fixture error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_arg(value: str, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReferenceError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _element_count(shape: list[int] | tuple[int, ...]) -> int:
    if (not isinstance(shape, (list, tuple)) or not shape
            or len(shape) > 4 or any(type(value) is not int or value <= 0 for value in shape)):
        raise ReferenceError("invalid tensor shape")
    return math.prod(shape)


def validate_tensor_layout(tensors: dict[str, Any], sidecar_bytes: int) -> None:
    if type(sidecar_bytes) is not int or not 0 < sidecar_bytes <= SIDE_CAR_LIMIT:
        raise ReferenceError("tensor sidecar size outside 64 MiB limit")
    if not isinstance(tensors, dict) or set(tensors) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("tensor inventory is incomplete or unexpected")
    cursor = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        metadata = tensors.get(name)
        expected_shape = EXPECTED_TENSOR_SHAPES[name]
        if not isinstance(metadata, dict) or metadata.get("dtype") != "F32":
            raise ReferenceError(f"invalid F32 tensor metadata: {name}")
        shape = metadata.get("shape")
        if shape != expected_shape:
            raise ReferenceError(f"tensor shape mismatch for {name}")
        size = _element_count(expected_shape) * 4
        if (type(metadata.get("offset_bytes")) is not int
                or metadata["offset_bytes"] != cursor
                or type(metadata.get("nbytes")) is not int
                or metadata["nbytes"] != size):
            raise ReferenceError(f"noncontiguous or malformed tensor range/offset: {name}")
        if size > sidecar_bytes - cursor:
            raise ReferenceError(f"tensor range exceeds sidecar: {name}")
        cursor += size
    if cursor != sidecar_bytes:
        raise ReferenceError("tensor sidecar has trailing or missing bytes")


def validate_rotary_capture(capture: dict[str, Any], *, calls: int,
                            block_index: int) -> None:
    """Require one source rotary invocation and its two named checkpoints."""
    if type(calls) is not int or calls != 1:
        raise ReferenceError(f"block {block_index} rotary capture must run exactly once")
    required = {"img.attn.q_rope", "img.attn.k_rope"}
    if not isinstance(capture, dict) or not required.issubset(capture):
        raise ReferenceError(f"block {block_index} rotary checkpoints are missing")


def _valid_basename(value: Any, suffix: str) -> bool:
    if not isinstance(value, str) or not value or Path(value).name != value:
        return False
    return value.endswith(suffix) and value not in (".", "..")


def _validate_block_document(name: str, value: Any) -> None:
    pin = BLOCK_PINS[name]
    index = pin["block_index"]
    if not isinstance(value, dict):
        raise ReferenceError(f"{name} provenance is missing")
    expected = {
        "block_index": index,
        "runtime_layer_name": f"double_block_{index + 1}",
        "file_bytes": pin["file_bytes"],
        "file_sha256": pin["file_sha256"],
        "manifest_sha256": pin["manifest_sha256"],
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_CHECKPOINT_REVISION,
        "source_git_commit": PINNED_REVISION,
        "checkpoint_path": SOURCE_CHECKPOINT_PATH,
        "checkpoint_lfs_sha256": SOURCE_CHECKPOINT_LFS_SHA256,
        "source_payload_sha256": pin["source_payload_sha256"],
        "source_header_sha256": pin["source_header_sha256"],
        "range_start_inclusive": pin["range_start_inclusive"],
        "range_end_inclusive": pin["range_end_inclusive"],
        "tensor_prefix": f"double_blocks.{index}.",
        "tensor_count": 32,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ReferenceError(f"{name} pinned block {key} mismatch")
    if (not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or not value["path"].endswith(".safetensors")):
        raise ReferenceError(f"{name} checkpoint path is invalid")
    if (not isinstance(value.get("manifest_path"), str)
            or not Path(value["manifest_path"]).is_absolute()
            or not value["manifest_path"].endswith(".safetensors.manifest.json")):
        raise ReferenceError(f"{name} manifest path is invalid")


def validate_reference_document(document: dict[str, Any]) -> None:
    """Validate the authority envelope without importing torch."""
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1):
        raise ReferenceError("missing or invalid pretrained serial reference schema")
    if document.get("upstream_revision") != PINNED_REVISION:
        raise ReferenceError("upstream revision mismatch")
    if document.get("geometry") != GEOMETRY:
        raise ReferenceError("two-block geometry mismatch")
    if document.get("precision") != PRECISION:
        raise ReferenceError("unsupported precision policy")
    tolerance = document.get("tolerance")
    if (not isinstance(tolerance, dict) or type(tolerance.get("atol")) not in (int, float)
            or type(tolerance.get("rtol")) not in (int, float)
            or float(tolerance["atol"]) != ATOL or float(tolerance["rtol"]) != RTOL):
        raise ReferenceError("numerical tolerance mismatch")
    producer = document.get("producer")
    if (not isinstance(producer, dict) or producer.get("device") != "mps"
            or producer.get("activation_readback") != "one packed final F32 tensor sidecar readback"
            or producer.get("attention_adapter_calls") != 2):
        raise ReferenceError("producer execution receipt mismatch")

    source = document.get("source")
    if not isinstance(source, dict):
        raise ReferenceError("source provenance is missing")
    conditioning = source.get("conditioning")
    if conditioning != CONDITIONING_SOURCE:
        raise ReferenceError("conditioning source pin mismatch")
    blocks = source.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != {"block0", "block1"}:
        raise ReferenceError("block provenance inventory mismatch")
    _validate_block_document("block0", blocks["block0"])
    _validate_block_document("block1", blocks["block1"])
    if (blocks["block0"]["file_sha256"] == blocks["block1"]["file_sha256"]
            or blocks["block0"]["manifest_sha256"] == blocks["block1"]["manifest_sha256"]
            or blocks["block0"]["source_payload_sha256"] == blocks["block1"]["source_payload_sha256"]):
        raise ReferenceError("block1 must be distinct from block0; identical block substitution rejected")

    tensor_file = document.get("tensor_file")
    if (not isinstance(tensor_file, dict)
            or not _valid_basename(tensor_file.get("name"), ".tensors.f32le")
            or tensor_file.get("encoding") != "f32-le"
            or type(tensor_file.get("nbytes")) is not int
            or not isinstance(tensor_file.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", tensor_file["sha256"]) is None):
        raise ReferenceError("tensor sidecar metadata is invalid")
    tensors = document.get("tensors")
    validate_tensor_layout(tensors, tensor_file["nbytes"])


def _load_json_file(path: Path, *, label: str, max_bytes: int) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size <= 0 or path.stat().st_size > max_bytes):
        raise ReferenceError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError(f"{label} is not valid UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise ReferenceError(f"{label} root must be an object")
    return value


def _validate_conditioning(reference_path: Path, supplied_sha256: str) -> tuple[dict[str, Any], Path, str]:
    expected_sha = _digest_arg(supplied_sha256, "conditioning reference digest")
    if expected_sha != CONDITIONING_REFERENCE_SHA256:
        raise ReferenceError("conditioning reference digest is not the pinned saved fixture")
    if reference_path.is_symlink():
        raise ReferenceError("conditioning reference symlinks are not admitted")
    resolved = reference_path.resolve(strict=True)
    if (not resolved.is_file() or resolved.is_symlink()
            or resolved.stat().st_size > 8 * 1024 * 1024):
        raise ReferenceError("conditioning reference must be a bounded regular JSON file")
    actual_sha = sha256_file(resolved)
    if actual_sha != expected_sha:
        raise ReferenceError("conditioning reference SHA-256 mismatch")
    document = _load_json_file(resolved, label="conditioning reference", max_bytes=8 * 1024 * 1024)
    try:
        conditioning_reference.validate_reference_document(document)
    except (conditioning_reference.ReferenceError, KeyError, TypeError) as exc:
        raise ReferenceError(f"conditioning reference validation failed: {exc}") from None
    sidecar_meta = document.get("tensor_file")
    if not isinstance(sidecar_meta, dict) or sidecar_meta != {
        "name": CONDITIONING_SOURCE["tensor_file_name"],
        "nbytes": CONDITIONING_SIDECAR_BYTES,
        "sha256": CONDITIONING_SIDECAR_SHA256,
    }:
        raise ReferenceError("conditioning tensor sidecar pin mismatch")
    sidecar_name = sidecar_meta["name"]
    if not _valid_basename(sidecar_name, ".tensors.f32le"):
        raise ReferenceError("conditioning tensor sidecar must be a sibling basename")
    sidecar_path = resolved.parent / sidecar_name
    if (not sidecar_path.is_file() or sidecar_path.is_symlink()
            or sidecar_path.stat().st_size != CONDITIONING_SIDECAR_BYTES
            or sha256_file(sidecar_path) != CONDITIONING_SIDECAR_SHA256):
        raise ReferenceError("conditioning tensor sidecar bytes do not match the saved fixture")
    try:
        conditioning_reference.validate_tensor_layout(
            document.get("tensors"), sidecar_path.stat().st_size
        )
    except (conditioning_reference.ReferenceError, KeyError, TypeError) as exc:
        raise ReferenceError(f"conditioning tensor layout validation failed: {exc}") from None
    return document, sidecar_path, actual_sha


def _block_expected_keys(torch: Any) -> dict[str, list[int]]:
    return block_reference._expected_block_keys(2048, 8192, 16)


def _expected_source_names(index: int, expected: dict[str, list[int]]) -> list[str]:
    return sorted(f"double_blocks.{index}.{name}" for name in expected)


def _validate_block_manifest(
    manifest: dict[str, Any],
    *,
    name: str,
    path: Path,
    file_sha256: str,
) -> None:
    pin = BLOCK_PINS[name]
    index = pin["block_index"]
    output = manifest.get("output")
    source = manifest.get("source")
    subset = manifest.get("subset")
    if (manifest.get("format") != "hy-nomas-block-subset-provenance-v1"
            or not isinstance(output, dict) or not isinstance(source, dict)
            or not isinstance(subset, dict)):
        raise ReferenceError(f"{name} manifest format/provenance mismatch")
    if (output.get("filename") != path.name
            or output.get("sha256") != file_sha256
            or output.get("size_bytes") != pin["file_bytes"]):
        raise ReferenceError(f"{name} manifest output pin mismatch")
    source_values = {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_CHECKPOINT_REVISION,
        "checkpoint_path": SOURCE_CHECKPOINT_PATH,
        "checkpoint_size_bytes": SOURCE_CHECKPOINT_BYTES,
        "checkpoint_lfs_sha256": SOURCE_CHECKPOINT_LFS_SHA256,
        "source_git_commit": PINNED_REVISION,
        "range_start_inclusive": pin["range_start_inclusive"],
        "range_end_inclusive": pin["range_end_inclusive"],
        "payload_sha256_observed": pin["source_payload_sha256"],
    }
    for key, expected in source_values.items():
        if source.get(key) != expected:
            raise ReferenceError(f"{name} manifest source {key} mismatch")
    if source.get("header_sha256_observed") != pin["source_header_sha256"]:
        raise ReferenceError(f"{name} manifest source header digest mismatch")
    if (subset.get("tensor_prefix") != f"double_blocks.{index}."
            or subset.get("tensor_count") != 32
            or subset.get("dtype") != "F32"
            or subset.get("stored_bytes_preserved") is not True
            or subset.get("conversion_performed") is not False
            or subset.get("payload_bytes") != 604_227_584
            or subset.get("tensor_names") != _expected_source_names(index, block_reference._expected_block_keys(2048, 8192, 16))):
        raise ReferenceError(f"{name} manifest tensor inventory mismatch")
    if "block_index" in subset and subset.get("block_index") != index:
        raise ReferenceError(f"{name} manifest block index mismatch")
    if index == 1 and subset.get("block_index") != 1:
        raise ReferenceError("block1 manifest must explicitly authenticate index 1")


def _trusted_weight_pair(
    *,
    name: str,
    weights: Path,
    weights_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[Path, str, Path, dict[str, Any]]:
    pin = BLOCK_PINS[name]
    expected_weight_sha = _digest_arg(weights_sha256, f"{name} weights digest")
    expected_manifest_sha = _digest_arg(manifest_sha256, f"{name} manifest digest")
    if expected_weight_sha != pin["file_sha256"] or expected_manifest_sha != pin["manifest_sha256"]:
        raise ReferenceError(f"{name} file/manifest pins are not the admitted source subset")
    if weights.is_symlink() or manifest_path.is_symlink():
        raise ReferenceError(f"{name} checkpoint and manifest symlinks are not admitted")
    weight_path = weights.resolve(strict=True)
    manifest_resolved = manifest_path.resolve(strict=True)
    for label, path in (("weights", weight_path), ("manifest", manifest_resolved)):
        if not path.is_file() or path.is_symlink():
            raise ReferenceError(f"{name} {label} must be a regular file")
    if weight_path.suffix != ".safetensors" or weight_path.stat().st_size != pin["file_bytes"]:
        raise ReferenceError(f"{name} safetensors size/type mismatch")
    if sha256_file(weight_path) != expected_weight_sha:
        raise ReferenceError(f"{name} safetensors SHA-256 mismatch")
    if manifest_resolved.stat().st_size <= 0 or manifest_resolved.stat().st_size > 2 * 1024 * 1024:
        raise ReferenceError(f"{name} manifest exceeds its size bound")
    if sha256_file(manifest_resolved) != expected_manifest_sha:
        raise ReferenceError(f"{name} manifest SHA-256 mismatch")
    manifest = _load_json_file(manifest_resolved, label=f"{name} manifest", max_bytes=2 * 1024 * 1024)
    _validate_block_manifest(manifest, name=name, path=weight_path, file_sha256=expected_weight_sha)
    return weight_path, expected_weight_sha, manifest_resolved, manifest


def _load_block_state(torch: Any, path: Path, index: int) -> dict[str, Any]:
    """Load exactly one finite F32 block state from the authenticated file."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ReferenceError("actual block weights require safetensors") from exc
    expected = _block_expected_keys(torch)
    prefix = f"double_blocks.{index}."
    expected_source = {f"{prefix}{key}": (key, shape) for key, shape in expected.items()}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        actual_keys = set(source.keys())
        expected_keys = set(expected_source)
        if actual_keys != expected_keys:
            raise ReferenceError(
                f"block {index} checkpoint inventory mismatch: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        state: dict[str, Any] = {}
        for source_name, (relative_name, expected_shape) in expected_source.items():
            value = source.get_tensor(source_name)
            if value.dtype != torch.float32 or list(value.shape) != expected_shape:
                raise ReferenceError(f"block {index} tensor dtype/shape mismatch: {source_name}")
            if not bool(torch.isfinite(value).all().item()):
                raise ReferenceError(f"block {index} contains non-finite source weights: {source_name}")
            state[relative_name] = value
    if len(state) != 32:
        raise ReferenceError(f"block {index} did not load exactly 32 weight tensors")
    return state


def _load_conditioning_inputs(torch: Any, np: Any, sidecar: Path,
                              document: dict[str, Any], device: Any) -> dict[str, Any]:
    tensor_map = np.memmap(sidecar, dtype="<f4", mode="r")
    result: dict[str, Any] = {}
    aliases = {
        "img": "input.video",
        "txt": "refiner.projected_text",
        "vec": "refiner.vec",
        "rope_cos": "input.rope_cos",
        "rope_sin": "input.rope_sin",
    }
    for target, source_name in aliases.items():
        metadata = document["tensors"].get(source_name)
        if not isinstance(metadata, dict) or metadata.get("dtype") != "F32":
            raise ReferenceError(f"saved conditioning tensor is missing or not F32: {source_name}")
        shape = metadata.get("shape")
        if shape != _CONDITIONING_SHAPES.get(source_name):
            raise ReferenceError(f"saved conditioning tensor shape changed: {source_name}")
        offset = metadata["offset_bytes"] // 4
        count = metadata["nbytes"] // 4
        values = np.array(tensor_map[offset:offset + count], dtype=np.float32, copy=True)
        if values.size != math.prod(shape) or not bool(np.isfinite(values).all()):
            raise ReferenceError(f"saved conditioning tensor extent/nonfinite check failed: {source_name}")
        cpu_tensor = torch.from_numpy(values.reshape(shape))
        value = cpu_tensor.to(device=device, dtype=torch.float32)
        if value.device.type != "mps" or value.dtype != torch.float32:
            raise ReferenceError(f"conditioning tensor failed MPS F32 transfer: {source_name}")
        result[target] = value
    result["vec_txt"] = result["vec"]
    if (tuple(result["img"].shape) != (1, 6, 2048)
            or tuple(result["txt"].shape) != (1, 23, 2048)
            or tuple(result["vec"].shape) != (1, 2048)
            or tuple(result["rope_cos"].shape) != (6, 128)
            or tuple(result["rope_sin"].shape) != (6, 128)):
        raise ReferenceError("saved conditioning inputs do not match admitted block geometry")
    return result


def _run_one_block(
    torch: Any,
    block_type: Any,
    state: dict[str, Any],
    index: int,
    device: Any,
    capture_state: dict[str, Any],
    receipt: dict[str, Any],
    inputs: dict[str, Any],
    current: dict[str, Any],
) -> tuple[dict[str, Any], tuple[Any, Any], Any]:
    capture_state["current"] = current
    capture_state["attention_replay"] = {}

    # The helper compiles the exact pinned class AST. The only substituted
    # operations are rotary capture and the MPS-only attention adapter.
    block = block_type(
        hidden_size=2048,
        heads_num=16,
        mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh",
        attn_mode="magi_mp_causal",
        qk_norm=True,
        qk_norm_type="rms",
        qkv_bias=True,
        dtype=torch.float32,
        device=torch.device("meta"),
    )
    block.load_state_dict(state, strict=True, assign=True)
    block.to(device=device, dtype=torch.float32).eval()
    del state
    block_reference._round_parameters_bf16_in_place(torch, block, device)
    hooks = block_reference._capture_hooks(block, capture_state, heads=16, head_dim=128)

    attn_param = {
        "num_prompts": 1,
        "txt_tokens_per_block": 23,
        "prompt_video_boundaries_tokens": [0, 6],
        "prompt_video_boundaries": [0, 6],
        "chunk_size_tokens": 0,
        "seg_ar_attention_backend": "magi",
    }
    before_calls = receipt["calls"]
    try:
        with torch.inference_mode():
            img_output, txt_output = block(
                img=inputs["img"],
                txt=inputs["txt"],
                vec=inputs["vec"],
                vec_txt=inputs["vec_txt"],
                freqs_cis=(inputs["rope_cos"], inputs["rope_sin"]),
                text_mask=None,
                attn_param=attn_param,
                is_flash=False,
                txt_tokens_per_block=23,
                prompt_video_boundaries=[0, 6],
            )
    finally:
        for hook in hooks:
            hook.remove()
    if receipt["calls"] != before_calls + 1:
        raise ReferenceError(f"block {index} did not make exactly one MPS attention call")
    for label, output in (("img", img_output), ("txt", txt_output)):
        if output.device.type != "mps" or output.dtype != torch.float32:
            raise ReferenceError(f"block {index} {label} output left MPS F32")
    current["img.output"] = img_output.detach().clone()
    current["txt.output"] = txt_output.detach().clone()
    if set(current) != set(BLOCK_CAPTURE_NAMES):
        raise ReferenceError(
            f"block {index} checkpoint inventory mismatch: "
            f"missing={sorted(set(BLOCK_CAPTURE_NAMES) - set(current))}, "
            f"extra={sorted(set(current) - set(BLOCK_CAPTURE_NAMES))}"
        )
    for name, value in current.items():
        if value.device.type != "mps" or value.dtype != torch.float32:
            raise ReferenceError(f"block {index} checkpoint is not MPS F32: {name}")
        if list(value.shape) != _CAPTURE_SHAPES[name]:
            raise ReferenceError(f"block {index} checkpoint shape mismatch: {name}")
    return current, (img_output, txt_output), block


def _serialize_tensors(torch: Any, records: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    if set(records) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("terminal tensor inventory differs from the admitted 86 names")
    flat: list[Any] = []
    finite: list[Any] = []
    metadata: dict[str, Any] = {}
    offset = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        value = records[name]
        if value.device.type != "mps" or value.dtype != torch.float32:
            raise ReferenceError(f"terminal checkpoint is not MPS F32: {name}")
        if list(value.shape) != EXPECTED_TENSOR_SHAPES[name]:
            raise ReferenceError(f"terminal checkpoint shape changed: {name}")
        contiguous = value.detach().contiguous()
        size = int(contiguous.numel()) * 4
        metadata[name] = {
            "dtype": "F32",
            "shape": list(contiguous.shape),
            "offset_bytes": offset,
            "nbytes": size,
        }
        offset += size
        flat.append(contiguous.reshape(-1))
        finite.append(torch.isfinite(contiguous).all().to(dtype=torch.float32).reshape(1))
    if not flat or offset > SIDE_CAR_LIMIT:
        raise ReferenceError("terminal sidecar exceeds the 64 MiB bound")
    # This is the sole activation/checkpoint transfer to CPU. Finite flags are
    # appended to the same packed transfer and excluded from the published data.
    packed = torch.cat(flat + finite).to(device="cpu", dtype=torch.float32).contiguous()
    array = packed.numpy()
    value_count = offset // 4
    if not bool((array[value_count:] == 1.0).all()):
        raise ReferenceError("non-finite value in the one packed final MPS readback")
    raw = array[:value_count].astype("<f4", copy=False).tobytes()
    if len(raw) != offset:
        raise ReferenceError("terminal F32 sidecar extent mismatch")
    validate_tensor_layout(metadata, len(raw))
    return raw, metadata


def _publish_noreplace(path: Path, data: bytes) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {path.parent}")
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fixture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new reference JSON path in an existing scratch directory")
    parser.add_argument("--conditioning-reference", type=Path,
                        default=DEFAULT_CONDITIONING_REFERENCE)
    parser.add_argument("--conditioning-reference-sha256", required=True)
    parser.add_argument("--block0-weights", type=Path, default=DEFAULT_BLOCK0_WEIGHTS)
    parser.add_argument("--block0-weights-sha256", required=True)
    parser.add_argument("--block0-manifest", type=Path, default=DEFAULT_BLOCK0_MANIFEST)
    parser.add_argument("--block0-manifest-sha256", required=True)
    parser.add_argument("--block1-weights", type=Path, default=DEFAULT_BLOCK1_WEIGHTS)
    parser.add_argument("--block1-weights-sha256", required=True)
    parser.add_argument("--block1-manifest", type=Path, default=DEFAULT_BLOCK1_MANIFEST)
    parser.add_argument("--block1-manifest-sha256", required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    return parser


def produce_fixture(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    output = args.output.absolute()
    if (output.suffix.lower() != ".json" or os.path.lexists(output)
            or not output.parent.is_dir()):
        raise ReferenceError("--output must be a new .json file in an existing directory")
    tensor_name = f"{output.stem}.tensors.f32le"
    tensor_path = output.parent / tensor_name
    if os.path.lexists(tensor_path):
        raise ReferenceError(f"tensor sidecar already exists: {tensor_path}")

    conditioning_doc, conditioning_sidecar, conditioning_sha = _validate_conditioning(
        args.conditioning_reference, args.conditioning_reference_sha256
    )
    weights: dict[str, tuple[Path, str, Path, dict[str, Any]]] = {}
    for name, weight_arg, weight_sha_arg, manifest_arg, manifest_sha_arg in (
        ("block0", args.block0_weights, args.block0_weights_sha256,
         args.block0_manifest, args.block0_manifest_sha256),
        ("block1", args.block1_weights, args.block1_weights_sha256,
         args.block1_manifest, args.block1_manifest_sha256),
    ):
        weights[name] = _trusted_weight_pair(
            name=name,
            weights=weight_arg,
            weights_sha256=weight_sha_arg,
            manifest_path=manifest_arg,
            manifest_sha256=manifest_sha_arg,
        )
    if weights["block0"][1] == weights["block1"][1]:
        raise ReferenceError("block1 cannot substitute block0 weights")

    upstream = args.upstream.resolve(strict=True)
    try:
        source_hashes = block_reference._verify_upstream(upstream)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReferenceError(f"pinned upstream source validation failed: {exc}") from None

    import numpy as np
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
    device = torch.device("mps")

    # Only authenticated, already-saved conditioning tensors are copied to MPS.
    conditioning_inputs = _load_conditioning_inputs(
        torch, np, conditioning_sidecar, conditioning_doc, device
    )
    original_inputs = {
        "input.img": conditioning_inputs["img"],
        "input.txt": conditioning_inputs["txt"],
        "input.vec": conditioning_inputs["vec"],
        "input.vec_txt": conditioning_inputs["vec_txt"],
        "input.rope_cos": conditioning_inputs["rope_cos"],
        "input.rope_sin": conditioning_inputs["rope_sin"],
    }

    try:
        helpers = block_reference._install_upstream_modules(upstream)
        posemb_layers = helpers[-1]
        capture_state: dict[str, Any] = {"current": {}, "attention_replay": {}}
        receipt: dict[str, Any] = {"calls": 0, "attention_precision": "f32"}
        adapter = block_reference._build_sdpa_adapter(
            torch, capture_state, receipt, "f32"
        )
        def make_rotary_capture(target: dict[str, Any], rotary_receipt: dict[str, int]):
            def apply_rotary_capture(q: Any, k: Any, freqs_cis: Any,
                                     head_first: bool = False) -> tuple[Any, Any]:
                q_rot, k_rot = posemb_layers.apply_rotary_emb(
                    q, k, freqs_cis, head_first=head_first
                )
                target["img.attn.q_rope"] = q_rot.detach().clone()
                target["img.attn.k_rope"] = k_rot.detach().clone()
                rotary_receipt["calls"] += 1
                return q_rot, k_rot
            return apply_rotary_capture

        block_outputs: dict[str, tuple[Any, Any]] = {}
        block_traces: dict[str, dict[str, Any]] = {}
        block_models: list[Any] = []
        # Block instances are built from the unchanged pinned class AST. Each
        # gets an adapter class whose rotary capture writes only to its trace.
        for index, name in ((0, "block0"), (1, "block1")):
            target: dict[str, Any] = {}
            rotary_receipt = {"calls": 0}
            block_type = block_reference._extract_pinned_block(
                upstream, torch, adapter,
                make_rotary_capture(target, rotary_receipt), helpers
            )
            state = _load_block_state(torch, weights[name][0], index)
            if index == 0:
                block_inputs = dict(conditioning_inputs)
            else:
                previous_img, previous_txt = block_outputs["block0"]
                block_inputs = dict(conditioning_inputs)
                block_inputs["img"] = previous_img
                block_inputs["txt"] = previous_txt
                if block_inputs["img"] is not previous_img or block_inputs["txt"] is not previous_txt:
                    raise ReferenceError("block0 MPS outputs were not handed directly to block1")
            trace, outputs, block = _run_one_block(
                torch,
                block_type,
                state,
                index,
                device,
                capture_state,
                receipt,
                block_inputs,
                target,
            )
            # The capture dictionary also contains the other 38 checkpoints;
            # invocation count proves the source-compiled rotary adapter ran.
            validate_rotary_capture(
                trace, calls=rotary_receipt["calls"], block_index=index
            )
            block_traces[name] = trace
            block_outputs[name] = outputs
            block_models.append(block)
        if receipt["calls"] != 2:
            raise ReferenceError("two-block run must call the MPS attention adapter exactly twice")
    except ReferenceError:
        raise
    except Exception as exc:
        raise ReferenceError(f"two-block MPS oracle execution failed: {type(exc).__name__}: {exc}") from None

    records: dict[str, Any] = dict(original_inputs)
    for block_name, trace in block_traces.items():
        for name, value in trace.items():
            records[f"{block_name}.{name}"] = value
    raw_tensor_bytes, tensor_metadata = _serialize_tensors(torch, records)
    tensor_sha = hashlib.sha256(raw_tensor_bytes).hexdigest()
    # Keep both distinct parameter owners alive through the terminal packed
    # readback; no allocator flush or explicit synchronization separates them.

    block_source_metadata: dict[str, Any] = {}
    for name, (path, file_sha, manifest_path, manifest) in weights.items():
        source = manifest["source"]
        index = BLOCK_PINS[name]["block_index"]
        block_source_metadata[name] = {
            "block_index": index,
            "runtime_layer_name": f"double_block_{index + 1}",
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "file_sha256": file_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": BLOCK_PINS[name]["manifest_sha256"],
            "source_repository": source["repository"],
            "source_revision": source["revision"],
            "source_git_commit": source["source_git_commit"],
            "checkpoint_path": source["checkpoint_path"],
            "checkpoint_lfs_sha256": source["checkpoint_lfs_sha256"],
            "source_payload_sha256": source["payload_sha256_observed"],
            "source_header_sha256": source.get("header_sha256_observed"),
            "range_start_inclusive": source["range_start_inclusive"],
            "range_end_inclusive": source["range_end_inclusive"],
            "tensor_prefix": f"double_blocks.{index}.",
            "tensor_count": 32,
        }

    source_inventory = dict(source_hashes)
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "upstream_revision": PINNED_REVISION,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "argv": list(sys.argv),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "mps",
            "activation_readback": "one packed final F32 tensor sidecar readback",
            "attention_adapter_calls": 2,
            "attention_adapter_calls_by_block": [1, 1],
            "block_invocations": ["block0", "block1"],
            "gpu_handoff": {
                "img": "block0.img.output passed directly as block1.img",
                "txt": "block0.txt.output passed directly as block1.txt",
                "same_mps_tensor_objects": True,
            },
            "terminal_tensor_count": len(records),
            "terminal_sidecar_bytes": len(raw_tensor_bytes),
        },
        "source": {
            "conditioning": {
                **CONDITIONING_SOURCE,
            },
            "blocks": block_source_metadata,
            "upstream_source_sha256": source_inventory,
        },
        "geometry": dict(GEOMETRY),
        "precision": dict(PRECISION),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "limitations": [
            "input.img is the seeded hidden video state saved by the text-conditioning fixture; this is not full video latent conditioning or img_in patch embedding",
            "PyTorch MPS F32 SDPA substitutes for Magi/Flash attention and does not establish their numerical parity",
            "weights use F32 parameter storage containing BF16-rounded values; this does not test physical BF16-resident parameter storage or native BF16 compute",
            "the declared whole-checkpoint HF LFS digest is not verified; source evidence is the pinned revision/header and the authenticated per-block byte-range payload hashes",
            "the admitted chain covers only actual block 0 followed by actual block 1, not full-model execution or media generation",
        ],
        "tensor_file": {
            "name": tensor_name,
            "nbytes": len(raw_tensor_bytes),
            "sha256": tensor_sha,
            "encoding": "f32-le",
        },
        "tensors": tensor_metadata,
    }
    validate_reference_document(document)
    validate_tensor_layout(tensor_metadata, len(raw_tensor_bytes))
    json_bytes = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_noreplace(tensor_path, raw_tensor_bytes)
    try:
        _publish_noreplace(output, json_bytes)
    except Exception:
        tensor_path.unlink(missing_ok=True)
        raise
    return {
        "status": "created",
        "output": str(output),
        "tensor_file": str(tensor_path),
        "tensor_file_sha256": tensor_sha,
        "tensor_count": len(tensor_metadata),
        "tensor_bytes": len(raw_tensor_bytes),
        "attention_adapter_calls": receipt["calls"],
        "block_handoff": "block0 MPS outputs reused as block1 inputs",
        "upstream_revision": PINNED_REVISION,
    }


def main(argv: list[str] | None = None) -> int:
    args = _fixture_parser().parse_args(argv)
    try:
        result = produce_fixture(args)
    except (OSError, ValueError, ReferenceError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
