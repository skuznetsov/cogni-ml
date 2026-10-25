#!/usr/bin/env python3
"""Produce a bounded MPS reference for actual Hy-Nomas blocks 0 through 3.

The producer reuses the qualified full-J synthetic MPS inputs, upstream ND
RoPE, F32 SDPA query chunks, and unchanged pinned MMDoubleStreamBlock source.
It loads four distinct authenticated block subsets, hands each block's MPS
image/text outputs directly to the next, and serializes only six original
inputs plus the eight final image/text outputs. This is not full generation.
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

import hy_nomas_full_j_reference as full_j_reference
import hy_nomas_pretrained_serial_reference as serial_reference
import hy_nomas_reference as block_reference


PINNED_REVISION = serial_reference.PINNED_REVISION
DEFAULT_UPSTREAM = serial_reference.DEFAULT_UPSTREAM
DEFAULT_BLOCK0_WEIGHTS = serial_reference.DEFAULT_BLOCK0_WEIGHTS
DEFAULT_BLOCK0_MANIFEST = serial_reference.DEFAULT_BLOCK0_MANIFEST
DEFAULT_BLOCK1_WEIGHTS = serial_reference.DEFAULT_BLOCK1_WEIGHTS
DEFAULT_BLOCK1_MANIFEST = serial_reference.DEFAULT_BLOCK1_MANIFEST
FOUR_BLOCK_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-four-block-20261025/weights")
DEFAULT_BLOCK2_WEIGHTS = FOUR_BLOCK_ROOT / "distilled-block2-f32.safetensors"
DEFAULT_BLOCK2_MANIFEST = Path(str(DEFAULT_BLOCK2_WEIGHTS) + ".manifest.json")
DEFAULT_BLOCK3_WEIGHTS = FOUR_BLOCK_ROOT / "distilled-block3-f32.safetensors"
DEFAULT_BLOCK3_MANIFEST = Path(str(DEFAULT_BLOCK3_WEIGHTS) + ".manifest.json")

SCHEMA = "hy-nomas-four-block-reference-v1"
SIDE_CAR_LIMIT = 512 * 1024 * 1024
JSON_LIMIT = 2 * 1024 * 1024
ATOL = serial_reference.ATOL
RTOL = serial_reference.RTOL
GEOMETRY = dict(full_j_reference.GEOMETRY)
PRECISION = dict(full_j_reference.PRECISION)
INPUT_GENERATION = full_j_reference.INPUT_GENERATION
VIDEO_TOKENS = full_j_reference.VIDEO_TOKENS
TEXT_TOKENS = full_j_reference.TEXT_TOKENS
TOTAL_TOKENS = full_j_reference.TOTAL_TOKENS
QUERY_CHUNK = full_j_reference.QUERY_CHUNK
ATTENTION_CHUNKS_PER_BLOCK = full_j_reference.ATTENTION_CHUNKS_PER_BLOCK
SOURCE_HEADER_SHA256 = "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200"
SOURCE_PAYLOAD_BYTES = 604_227_584
MAX_SUBSET_HEADER_BYTES = 1024 * 1024

# These exact source byte ranges were derived from the authenticated original
# safetensors header. They are intentionally not extrapolated from block index.
DYNAMIC_BLOCK_PINS: dict[str, dict[str, int]] = {
    "block2": {
        "block_index": 2,
        "range_start_inclusive": 8_219_001_328,
        "range_end_inclusive": 8_823_228_911,
        "payload_bytes": SOURCE_PAYLOAD_BYTES,
    },
    "block3": {
        "block_index": 3,
        "range_start_inclusive": 14_865_504_752,
        "range_end_inclusive": 15_469_732_335,
        "payload_bytes": SOURCE_PAYLOAD_BYTES,
    },
}

_INPUT_SHAPES = dict(full_j_reference._INPUT_SHAPES)
_CAPTURE_SHAPES = {
    "img.output": [1, VIDEO_TOKENS, 2048],
    "txt.output": [1, TEXT_TOKENS, 2048],
}
EXPECTED_TENSOR_SHAPES: dict[str, list[int]] = dict(_INPUT_SHAPES)
for _block_index in range(4):
    for _capture_name, _capture_shape in _CAPTURE_SHAPES.items():
        EXPECTED_TENSOR_SHAPES[f"block{_block_index}.{_capture_name}"] = _capture_shape


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
    if (not isinstance(shape, (list, tuple)) or not shape or len(shape) > 4
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise ReferenceError("invalid tensor shape")
    return math.prod(shape)


def _valid_basename(value: Any, suffix: str) -> bool:
    if not isinstance(value, str) or not value or Path(value).name != value:
        return False
    return value.endswith(suffix) and value not in (".", "..")


def validate_tensor_layout(tensors: dict[str, Any], sidecar_bytes: int) -> None:
    if type(sidecar_bytes) is not int or not 0 < sidecar_bytes <= SIDE_CAR_LIMIT:
        raise ReferenceError("tensor sidecar size outside 512 MiB limit")
    if not isinstance(tensors, dict) or set(tensors) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("tensor inventory is incomplete or unexpected")
    cursor = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        metadata = tensors.get(name)
        shape = EXPECTED_TENSOR_SHAPES[name]
        if not isinstance(metadata, dict) or metadata.get("dtype") != "F32":
            raise ReferenceError(f"invalid F32 tensor metadata: {name}")
        if metadata.get("shape") != shape:
            raise ReferenceError(f"tensor shape mismatch for {name}")
        nbytes = _element_count(shape) * 4
        if (type(metadata.get("offset_bytes")) is not int
                or metadata["offset_bytes"] != cursor
                or type(metadata.get("nbytes")) is not int
                or metadata["nbytes"] != nbytes):
            raise ReferenceError(f"noncontiguous or malformed tensor range/offset: {name}")
        if nbytes > sidecar_bytes - cursor:
            raise ReferenceError(f"tensor range exceeds sidecar: {name}")
        cursor += nbytes
    if cursor != sidecar_bytes:
        raise ReferenceError("tensor sidecar has trailing or missing bytes")


def _validate_dynamic_block_document(name: str, value: Any) -> None:
    pin = DYNAMIC_BLOCK_PINS[name]
    index = pin["block_index"]
    if not isinstance(value, dict):
        raise ReferenceError(f"{name} provenance is missing")
    expected = {
        "block_index": index,
        "runtime_layer_name": f"double_block_{index + 1}",
        "source_repository": serial_reference.SOURCE_REPOSITORY,
        "source_revision": serial_reference.SOURCE_CHECKPOINT_REVISION,
        "source_git_commit": PINNED_REVISION,
        "checkpoint_path": serial_reference.SOURCE_CHECKPOINT_PATH,
        "checkpoint_lfs_sha256": serial_reference.SOURCE_CHECKPOINT_LFS_SHA256,
        "source_header_sha256": SOURCE_HEADER_SHA256,
        "range_start_inclusive": pin["range_start_inclusive"],
        "range_end_inclusive": pin["range_end_inclusive"],
        "tensor_prefix": f"double_blocks.{index}.",
        "tensor_count": 32,
    }
    for key, expected_value in expected.items():
        if key == "block_index" and type(value.get(key)) is not int:
            raise ReferenceError(f"{name} pinned block_index must be an integer")
        if value.get(key) != expected_value:
            raise ReferenceError(f"{name} pinned block {key} mismatch")
    if (not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or Path(value["path"]).name != f"distilled-block{index}-f32.safetensors"):
        raise ReferenceError(f"{name} checkpoint path is invalid")
    if (not isinstance(value.get("manifest_path"), str)
            or not Path(value["manifest_path"]).is_absolute()
            or Path(value["manifest_path"]).name
            != f"distilled-block{index}-f32.safetensors.manifest.json"):
        raise ReferenceError(f"{name} manifest path is invalid")
    if type(value.get("file_bytes")) is not int or not (
            pin["payload_bytes"] <= value["file_bytes"]
            <= pin["payload_bytes"] + MAX_SUBSET_HEADER_BYTES):
        raise ReferenceError(f"{name} safetensors size is outside the bounded subset extent")
    for field in ("file_sha256", "manifest_sha256", "source_payload_sha256"):
        if (not isinstance(value.get(field), str)
                or re.fullmatch(r"[0-9a-f]{64}", value[field]) is None):
            raise ReferenceError(f"{name} {field} is malformed")


def validate_reference_document(document: dict[str, Any]) -> None:
    """Validate the four-block authority envelope without importing torch."""
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1):
        raise ReferenceError("missing or invalid four-block reference schema")
    upstream = document.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("revision") != PINNED_REVISION:
        raise ReferenceError("pinned upstream revision mismatch")
    if upstream.get("source_sha256") != block_reference.PINNED_SOURCE_SHA256:
        raise ReferenceError("pinned upstream source inventory mismatch")
    if document.get("geometry") != GEOMETRY:
        raise ReferenceError("full-J geometry mismatch")
    if document.get("input_generation") != INPUT_GENERATION:
        raise ReferenceError("synthetic input or upstream RoPE recipe mismatch")
    if document.get("precision") != PRECISION:
        raise ReferenceError("unsupported precision policy")
    tolerance = document.get("tolerance")
    if (not isinstance(tolerance, dict) or type(tolerance.get("atol")) not in (int, float)
            or type(tolerance.get("rtol")) not in (int, float)
            or float(tolerance["atol"]) != ATOL or float(tolerance["rtol"]) != RTOL):
        raise ReferenceError("numerical tolerance mismatch")

    expected_order = [f"block{index}" for index in range(4)]
    expected_handoff = {
        "img": "each block output passed directly as next block input",
        "txt": "each block output passed directly as next block input",
        "same_mps_tensor_objects": True,
    }
    producer = document.get("producer")
    if (not isinstance(producer, dict) or producer.get("device") != "mps"
            or producer.get("activation_readback")
            != "one packed final F32 tensor sidecar readback"
            or producer.get("attention_adapter")
            != "PyTorch MPS SDPA F32, query rows chunked at 256"
            or type(producer.get("attention_adapter_calls")) is not int
            or producer["attention_adapter_calls"] != 4
            or type(producer.get("attention_query_chunks_per_block")) is not int
            or producer["attention_query_chunks_per_block"] != ATTENTION_CHUNKS_PER_BLOCK
            or type(producer.get("attention_query_chunk_calls")) is not int
            or producer["attention_query_chunk_calls"] != 4 * ATTENTION_CHUNKS_PER_BLOCK
            or type(producer.get("max_attention_query_chunk")) is not int
            or producer["max_attention_query_chunk"] != QUERY_CHUNK
            or producer.get("block_invocations") != expected_order
            or producer.get("block_handoff") != expected_handoff
            or producer.get("captures_per_block") != ["img.output", "txt.output"]
            or type(producer.get("terminal_tensor_count")) is not int
            or producer["terminal_tensor_count"] != len(EXPECTED_TENSOR_SHAPES)):
        raise ReferenceError("producer execution receipt mismatch")

    source = document.get("source")
    if (not isinstance(source, dict) or source.get("block_order") != expected_order
            or not isinstance(source.get("blocks"), dict)
            or set(source["blocks"]) != set(expected_order)):
        raise ReferenceError("four-block source order/inventory mismatch")
    for name in expected_order[:2]:
        try:
            serial_reference._validate_block_document(name, source["blocks"][name])
        except (serial_reference.ReferenceError, KeyError, TypeError) as exc:
            raise ReferenceError(f"{name} pinned source metadata mismatch: {exc}") from None
    for name in expected_order[2:]:
        _validate_dynamic_block_document(name, source["blocks"][name])
    weight_digests: set[str] = set()
    manifest_digests: set[str] = set()
    payload_digests: set[str] = set()
    for name in expected_order:
        block = source["blocks"][name]
        weight_digest = block["file_sha256"]
        manifest_digest = block["manifest_sha256"]
        payload_digest = block["source_payload_sha256"]
        if (weight_digest in weight_digests or manifest_digest in manifest_digests
                or payload_digest in payload_digests):
            raise ReferenceError("four distinct block sources are required; alias rejected")
        weight_digests.add(weight_digest)
        manifest_digests.add(manifest_digest)
        payload_digests.add(payload_digest)

    tensor_file = document.get("tensor_file")
    if (not isinstance(tensor_file, dict)
            or not _valid_basename(tensor_file.get("name"), ".tensors.f32le")
            or tensor_file.get("encoding") != "f32-le"
            or type(tensor_file.get("nbytes")) is not int
            or not isinstance(tensor_file.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", tensor_file["sha256"]) is None):
        raise ReferenceError("tensor sidecar metadata is invalid")
    validate_tensor_layout(document.get("tensors"), tensor_file["nbytes"])
    if (type(producer.get("terminal_sidecar_bytes")) is not int
            or producer["terminal_sidecar_bytes"] != tensor_file["nbytes"]):
        raise ReferenceError("producer sidecar byte receipt mismatch")


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new reference JSON path in an existing scratch directory")
    paths = {
        0: (DEFAULT_BLOCK0_WEIGHTS, DEFAULT_BLOCK0_MANIFEST),
        1: (DEFAULT_BLOCK1_WEIGHTS, DEFAULT_BLOCK1_MANIFEST),
        2: (DEFAULT_BLOCK2_WEIGHTS, DEFAULT_BLOCK2_MANIFEST),
        3: (DEFAULT_BLOCK3_WEIGHTS, DEFAULT_BLOCK3_MANIFEST),
    }
    for index, (weights, manifest) in paths.items():
        parser.add_argument(f"--block{index}-weights", type=Path, default=weights)
        parser.add_argument(f"--block{index}-weights-sha256", required=True)
        parser.add_argument(f"--block{index}-manifest", type=Path, default=manifest)
        parser.add_argument(f"--block{index}-manifest-sha256", required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    return parser


def _validate_dynamic_manifest(
    manifest: dict[str, Any],
    *,
    name: str,
    filename: str,
    file_bytes: int,
    file_sha256: str,
) -> None:
    pin = DYNAMIC_BLOCK_PINS[name]
    index = pin["block_index"]
    output = manifest.get("output")
    source = manifest.get("source")
    subset = manifest.get("subset")
    if (manifest.get("format") != "hy-nomas-block-subset-provenance-v1"
            or not isinstance(output, dict) or not isinstance(source, dict)
            or not isinstance(subset, dict)):
        raise ReferenceError(f"{name} manifest format/provenance mismatch")
    if (output.get("filename") != filename
            or output.get("sha256") != file_sha256
            or type(output.get("size_bytes")) is not int
            or output.get("size_bytes") != file_bytes):
        raise ReferenceError(f"{name} manifest output pin mismatch")
    source_values = {
        "repository": serial_reference.SOURCE_REPOSITORY,
        "revision": serial_reference.SOURCE_CHECKPOINT_REVISION,
        "checkpoint_path": serial_reference.SOURCE_CHECKPOINT_PATH,
        "checkpoint_size_bytes": serial_reference.SOURCE_CHECKPOINT_BYTES,
        "checkpoint_lfs_sha256": serial_reference.SOURCE_CHECKPOINT_LFS_SHA256,
        "source_git_commit": PINNED_REVISION,
        "range_start_inclusive": pin["range_start_inclusive"],
        "range_end_inclusive": pin["range_end_inclusive"],
        "header_sha256_observed": SOURCE_HEADER_SHA256,
    }
    for key, expected in source_values.items():
        if key in (
            "checkpoint_size_bytes", "range_start_inclusive", "range_end_inclusive"
        ) and type(source.get(key)) is not int:
            raise ReferenceError(f"{name} manifest source {key} must be an integer")
        if source.get(key) != expected:
            raise ReferenceError(f"{name} manifest source {key} mismatch")
    payload_digest = source.get("payload_sha256_observed")
    if (not isinstance(payload_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", payload_digest) is None):
        raise ReferenceError(f"{name} manifest source payload digest is malformed")
    expected = block_reference._expected_block_keys(2048, 8192, 16)
    if (type(subset.get("block_index")) is not int
            or subset.get("block_index") != index
            or subset.get("tensor_prefix") != f"double_blocks.{index}."
            or type(subset.get("tensor_count")) is not int
            or subset.get("tensor_count") != 32
            or subset.get("dtype") != "F32"
            or subset.get("stored_bytes_preserved") is not True
            or subset.get("conversion_performed") is not False
            or type(subset.get("payload_bytes")) is not int
            or subset.get("payload_bytes") != pin["payload_bytes"]
            or subset.get("tensor_names")
            != serial_reference._expected_source_names(index, expected)):
        raise ReferenceError(f"{name} manifest tensor inventory/index mismatch")
    if type(file_bytes) is not int or not (pin["payload_bytes"] <= file_bytes
            <= pin["payload_bytes"] + MAX_SUBSET_HEADER_BYTES):
        raise ReferenceError(f"{name} subset file size exceeds its bounded header allowance")


def _trusted_dynamic_pair(
    *,
    name: str,
    weights: Path,
    weights_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[Path, str, Path, dict[str, Any]]:
    expected_weight_sha = _digest_arg(weights_sha256, f"{name} weights digest")
    expected_manifest_sha = _digest_arg(manifest_sha256, f"{name} manifest digest")
    if weights.is_symlink() or manifest_path.is_symlink():
        raise ReferenceError(f"{name} checkpoint and manifest symlinks are not admitted")
    weight_path = weights.resolve(strict=True)
    manifest_resolved = manifest_path.resolve(strict=True)
    for label, path in (("weights", weight_path), ("manifest", manifest_resolved)):
        if not path.is_file() or path.is_symlink():
            raise ReferenceError(f"{name} {label} must be a regular file")
    expected_name = f"distilled-block{DYNAMIC_BLOCK_PINS[name]['block_index']}-f32.safetensors"
    if weight_path.name != expected_name:
        raise ReferenceError(f"{name} checkpoint filename/index mismatch")
    file_bytes = weight_path.stat().st_size
    pin = DYNAMIC_BLOCK_PINS[name]
    if not (pin["payload_bytes"] <= file_bytes
            <= pin["payload_bytes"] + MAX_SUBSET_HEADER_BYTES):
        raise ReferenceError(f"{name} safetensors size/type mismatch")
    if sha256_file(weight_path) != expected_weight_sha:
        raise ReferenceError(f"{name} safetensors SHA-256 mismatch")
    if manifest_resolved.stat().st_size <= 0 or manifest_resolved.stat().st_size > JSON_LIMIT:
        raise ReferenceError(f"{name} manifest exceeds its size bound")
    if sha256_file(manifest_resolved) != expected_manifest_sha:
        raise ReferenceError(f"{name} manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError(f"{name} manifest is not bounded UTF-8 JSON") from None
    if not isinstance(manifest, dict):
        raise ReferenceError(f"{name} manifest root must be an object")
    _validate_dynamic_manifest(
        manifest,
        name=name,
        filename=weight_path.name,
        file_bytes=file_bytes,
        file_sha256=expected_weight_sha,
    )
    return weight_path, expected_weight_sha, manifest_resolved, manifest


def _source_block_metadata(
    weights: dict[str, tuple[Path, str, Path, dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (path, file_sha, manifest_path, manifest) in weights.items():
        source = manifest["source"]
        pins = serial_reference.BLOCK_PINS if name in serial_reference.BLOCK_PINS else DYNAMIC_BLOCK_PINS
        index = pins[name]["block_index"]
        result[name] = {
            "block_index": index,
            "runtime_layer_name": f"double_block_{index + 1}",
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "file_sha256": file_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
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
    return result


def _run_one_block(
    torch: Any,
    upstream: Path,
    helpers: tuple[Any, ...],
    state: dict[str, Any],
    index: int,
    device: Any,
    adapter: Any,
    inputs: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[Any, Any, Any]:
    block_type = block_reference._extract_pinned_block(
        upstream, torch, adapter, helpers[-1].apply_rotary_emb, helpers
    )
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

    attn_param = {
        "num_prompts": 1,
        "txt_tokens_per_block": TEXT_TOKENS,
        "prompt_video_boundaries_tokens": [0, VIDEO_TOKENS],
        "prompt_video_boundaries": [0, VIDEO_TOKENS],
        "chunk_size_tokens": 0,
        "seg_ar_attention_backend": "magi",
    }
    before_calls = receipt["calls"]
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
            txt_tokens_per_block=TEXT_TOKENS,
            prompt_video_boundaries=[0, VIDEO_TOKENS],
        )
    if receipt["calls"] != before_calls + 1:
        raise ReferenceError(f"block {index} did not make exactly one MPS SDPA adapter call")
    for stream, output in (("img", img_output), ("txt", txt_output)):
        shape = _CAPTURE_SHAPES[f"{stream}.output"]
        if output.device.type != "mps" or output.dtype != torch.float32 or list(output.shape) != shape:
            raise ReferenceError(f"block {index} {stream} output device/dtype/shape mismatch")
    return img_output, txt_output, block


def _serialize_tensors(torch: Any, records: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    if set(records) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("terminal tensor inventory differs from the admitted 14 names")
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
        nbytes = int(contiguous.numel()) * 4
        metadata[name] = {
            "dtype": "F32",
            "shape": list(contiguous.shape),
            "offset_bytes": offset,
            "nbytes": nbytes,
        }
        offset += nbytes
        flat.append(contiguous.reshape(-1))
        finite.append(torch.isfinite(contiguous).all().to(dtype=torch.float32).reshape(1))
    if not flat or offset > SIDE_CAR_LIMIT:
        raise ReferenceError("terminal sidecar exceeds the 512 MiB bound")
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
    owned_temporary = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        owned_temporary = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if owned_temporary:
            temporary.unlink(missing_ok=True)


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

    weights: dict[str, tuple[Path, str, Path, dict[str, Any]]] = {}
    for name, weight_arg, weight_sha_arg, manifest_arg, manifest_sha_arg in (
        ("block0", args.block0_weights, args.block0_weights_sha256,
         args.block0_manifest, args.block0_manifest_sha256),
        ("block1", args.block1_weights, args.block1_weights_sha256,
         args.block1_manifest, args.block1_manifest_sha256),
    ):
        weights[name] = serial_reference._trusted_weight_pair(
            name=name,
            weights=weight_arg,
            weights_sha256=weight_sha_arg,
            manifest_path=manifest_arg,
            manifest_sha256=manifest_sha_arg,
        )
    for name, weight_arg, weight_sha_arg, manifest_arg, manifest_sha_arg in (
        ("block2", args.block2_weights, args.block2_weights_sha256,
         args.block2_manifest, args.block2_manifest_sha256),
        ("block3", args.block3_weights, args.block3_weights_sha256,
         args.block3_manifest, args.block3_manifest_sha256),
    ):
        weights[name] = _trusted_dynamic_pair(
            name=name,
            weights=weight_arg,
            weights_sha256=weight_sha_arg,
            manifest_path=manifest_arg,
            manifest_sha256=manifest_sha_arg,
        )
    file_hashes = [weights[f"block{index}"][1] for index in range(4)]
    manifest_hashes = [sha256_file(weights[f"block{index}"][2]) for index in range(4)]
    if len(set(file_hashes)) != 4 or len(set(manifest_hashes)) != 4:
        raise ReferenceError("all four blocks must have distinct source and manifest digests")

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

    try:
        helpers = block_reference._install_upstream_modules(upstream)
        posemb_layers = helpers[-1]
        inputs = full_j_reference._generate_mps_inputs(torch, posemb_layers, device)
        records: dict[str, Any] = {
            "input.img": inputs["img"],
            "input.txt": inputs["txt"],
            "input.vec": inputs["vec"],
            "input.vec_txt": inputs["vec_txt"],
            "input.rope_cos": inputs["rope_cos"],
            "input.rope_sin": inputs["rope_sin"],
        }
        receipt: dict[str, Any] = {"calls": 0, "query_chunks": 0, "max_query_chunk": 0}
        adapter = full_j_reference._build_chunked_sdpa_adapter(torch, receipt)
        model_roots: list[Any] = []
        block_inputs = inputs
        for index in range(4):
            name = f"block{index}"
            state = serial_reference._load_block_state(torch, weights[name][0], index)
            img_output, txt_output, model = _run_one_block(
                torch, upstream, helpers, state, index, device, adapter, block_inputs, receipt
            )
            records[f"{name}.img.output"] = img_output
            records[f"{name}.txt.output"] = txt_output
            model_roots.append(model)
            block_inputs = {
                **inputs,
                "img": img_output,
                "txt": txt_output,
            }
        expected_chunks = 4 * ATTENTION_CHUNKS_PER_BLOCK
        if receipt["calls"] != 4 or receipt["query_chunks"] != expected_chunks:
            raise ReferenceError("four-block chain did not execute all admitted attention chunks")
        raw_tensor_bytes, tensor_metadata = _serialize_tensors(torch, records)
    except ReferenceError:
        raise
    except Exception as exc:
        raise ReferenceError(
            f"four-block full-J MPS oracle failed: {type(exc).__name__}: {exc}"
        ) from None

    tensor_sha = hashlib.sha256(raw_tensor_bytes).hexdigest()
    source_blocks = _source_block_metadata(weights)
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "upstream": {
            "revision": PINNED_REVISION,
            "source_sha256": source_hashes,
        },
        "producer": {
            "path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "input_recipe_path": str(Path(full_j_reference.__file__).resolve()),
            "input_recipe_source_sha256": sha256_file(Path(full_j_reference.__file__).resolve()),
            "argv": list(sys.argv),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "mps",
            "activation_readback": "one packed final F32 tensor sidecar readback",
            "attention_adapter": "PyTorch MPS SDPA F32, query rows chunked at 256",
            "attention_adapter_calls": receipt["calls"],
            "attention_query_chunks_per_block": ATTENTION_CHUNKS_PER_BLOCK,
            "attention_query_chunk_calls": receipt["query_chunks"],
            "max_attention_query_chunk": QUERY_CHUNK,
            "block_invocations": [f"block{index}" for index in range(4)],
            "block_handoff": {
                "img": "each block output passed directly as next block input",
                "txt": "each block output passed directly as next block input",
                "same_mps_tensor_objects": True,
            },
            "captures_per_block": ["img.output", "txt.output"],
            "terminal_tensor_count": len(records),
            "terminal_sidecar_bytes": len(raw_tensor_bytes),
        },
        "source": {
            "block_order": [f"block{index}" for index in range(4)],
            "blocks": source_blocks,
        },
        "geometry": dict(GEOMETRY),
        "input_generation": json.loads(json.dumps(INPUT_GENERATION)),
        "precision": dict(PRECISION),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "limitations": [
            "img, txt, vec, and vec_txt are deterministic synthetic MPS inputs, not a saved conditioning or full-generation path",
            "PyTorch MPS F32 SDPA with query chunks of 256 substitutes for Magi/Flash attention and does not establish their parity",
            "weights use F32 parameter storage containing BF16-rounded values; physical BF16-resident parameters and native BF16 compute are not tested",
            "the four-block chain covers actual blocks 0 through 3 only, not the remaining DiT blocks or media generation",
            "block 2 and block 3 source-file and manifest digests are invocation-supplied pins; source header, byte ranges, HF identity and tensor inventory are independently checked against this producer's pins",
            "the declared whole-checkpoint HF LFS digest is not verified; source evidence is the pinned revision/header and authenticated per-block byte-range manifests",
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
    if len(json_bytes) > JSON_LIMIT:
        raise ReferenceError("reference JSON exceeds the 2 MiB bound")
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
        "attention_query_chunk_calls": receipt["query_chunks"],
        "block_order": [f"block{index}" for index in range(4)],
        "upstream_revision": PINNED_REVISION,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    try:
        result = produce_fixture(args)
    except (OSError, ValueError, ReferenceError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
