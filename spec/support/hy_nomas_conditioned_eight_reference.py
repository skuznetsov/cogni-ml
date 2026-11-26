#!/usr/bin/env python3
"""Produce a bounded conditioned full-J reference for pretrained blocks 0..7.

This reuses the qualified conditioned-four pipeline and extends its real
MMDoubleStreamBlock chain to eight authenticated source blocks. The saved
Qwen state-26 crop feeds one complete masked refiner; synthetic video and the
pinned upstream ND RoPE then pass through blocks 0..7 in order. Only the five
conditioning/input tensors and each block's final image/text outputs are
captured. The MPS SDPA unpadding adapter is a scoped attention surrogate, not
Magi/CUDA parity or full media generation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SPEC_SUPPORT = ROOT / "spec" / "support"
EXAMPLES_SUPPORT = ROOT / "examples" / "support"
for _support in (SPEC_SUPPORT, EXAMPLES_SUPPORT):
    if str(_support) not in sys.path:
        sys.path.insert(0, str(_support))

import hy_nomas_block47_bf16_reference as block47_reference  # noqa: E402
import hy_nomas_conditioned_four_reference as four_conditioned  # noqa: E402
import hy_nomas_four_block_reference as four_reference  # noqa: E402
import hy_nomas_full_j_reference as full_j_reference  # noqa: E402
import hy_nomas_pretrained_serial_reference as serial_reference  # noqa: E402
import hy_nomas_reference as block_reference  # noqa: E402
import hy_nomas_text_conditioning_reference as text_reference  # noqa: E402


SCHEMA = "hy-nomas-conditioned-eight-reference-v1"
PINNED_REVISION = four_conditioned.PINNED_REVISION
DEFAULT_UPSTREAM = four_conditioned.DEFAULT_UPSTREAM
DEFAULT_TEXT_WEIGHTS = four_conditioned.DEFAULT_TEXT_WEIGHTS
DEFAULT_TEXT_MANIFEST = four_conditioned.DEFAULT_TEXT_MANIFEST
DEFAULT_QWEN_REFERENCE = four_conditioned.DEFAULT_QWEN_REFERENCE
DEFAULT_QWEN_SIDECAR = four_conditioned.DEFAULT_QWEN_SIDECAR
ATOL = four_conditioned.ATOL
RTOL = four_conditioned.RTOL
VIDEO_TOKENS = four_conditioned.VIDEO_TOKENS
TEXT_TOKENS = four_conditioned.TEXT_TOKENS
VALID_TEXT_TOKENS = four_conditioned.VALID_TEXT_TOKENS
PADDED_TEXT_TOKENS = four_conditioned.PADDED_TEXT_TOKENS
TOTAL_TOKENS = four_conditioned.TOTAL_TOKENS
VALID_JOINT_TOKENS = four_conditioned.VALID_JOINT_TOKENS
QUERY_CHUNK = four_conditioned.QUERY_CHUNK
ATTENTION_CHUNKS_PER_BLOCK = four_conditioned.ATTENTION_CHUNKS_PER_BLOCK
BLOCK_COUNT = 8
TOTAL_ATTENTION_CHUNKS = BLOCK_COUNT * ATTENTION_CHUNKS_PER_BLOCK
JSON_LIMIT = four_conditioned.JSON_LIMIT
SIDE_CAR_LIMIT = four_conditioned.SIDE_CAR_LIMIT
QWEN_SIDECAR_BYTES = four_conditioned.QWEN_SIDECAR_BYTES
GEOMETRY = copy.deepcopy(four_conditioned.GEOMETRY)
PRECISION = copy.deepcopy(four_conditioned.PRECISION)
INPUT_GENERATION = copy.deepcopy(four_conditioned.INPUT_GENERATION)
PINNED_UPSTREAM_SOURCE_SHA256 = dict(four_conditioned.PINNED_UPSTREAM_SOURCE_SHA256)

EXPECTED_TENSOR_SHAPES: dict[str, list[int]] = {
    name: list(shape)
    for name, shape in four_conditioned.EXPECTED_TENSOR_SHAPES.items()
    if name in {
        "input.img", "input.rope_cos", "input.rope_sin",
        "conditioning.projected_text", "conditioning.vec",
    }
}
for _index in range(BLOCK_COUNT):
    EXPECTED_TENSOR_SHAPES[f"block{_index}.img.output"] = [1, VIDEO_TOKENS, 2048]
    EXPECTED_TENSOR_SHAPES[f"block{_index}.txt.output"] = [1, TEXT_TOKENS, 2048]

SOURCE_HEADER_SHA256 = four_reference.SOURCE_HEADER_SHA256
SOURCE_PAYLOAD_BYTES = four_reference.SOURCE_PAYLOAD_BYTES
MAX_SUBSET_HEADER_BYTES = four_reference.MAX_SUBSET_HEADER_BYTES
EIGHT_BLOCK_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-conditioned-eight-20261025/weights"
)

# Blocks 2 and 3 are pinned to the already-produced authenticated four-block
# source capture, rather than accepting caller-defined hashes as trust roots.
BLOCK23_SOURCE_PINS: dict[int, dict[str, Any]] = {
    2: {
        "file_bytes": 604_231_488,
        "file_sha256": "436c2376ab736131ecc3e8374c8cafc99cf215e90a85be876f919021c9bc0b00",
        "manifest_sha256": "16568167da07946a7d14bd702238ea9824a69aaaa021797c0b56b322d222f8c8",
        "source_payload_sha256": "bc5bbbf6391e602dbc29b38e84d7c70061e00d3f643cd8f65c89970bd1840aeb",
        "range_start_inclusive": 8_219_001_328,
        "range_end_inclusive": 8_823_228_911,
    },
    3: {
        "file_bytes": 604_231_488,
        "file_sha256": "67ae4564ac33f4e143f58b2fc929df7e5263f77fa3b2157950dab30461ee40c1",
        "manifest_sha256": "c6418b2cb4bade3c83dd25cc6debde755c05c5d9bdd386c19e167f1a3ea709fd",
        "source_payload_sha256": "5150107b12fb863fead456179dcd97993a40236649d27fa463153fd76192062b",
        "range_start_inclusive": 14_865_504_752,
        "range_end_inclusive": 15_469_732_335,
    },
}

# Pins 4..7 are filled from the independently authenticated block-subset
# manifests. They bind both the caller-facing fixture validator and actual
# source admission; hashes are never learned from the fixture being checked.
BLOCK47_SOURCE_PINS: dict[int, dict[str, Any]] = {
    4: {
        "file_bytes": 604_231_488,
        "file_sha256": "08a7502785d0cac541f83a7931494eceaae8c7cef5ff7212b7710a92a5dcfb5f",
        "manifest_sha256": "e52d4e096f5cc972db3cca709398283898a82cb11889852002e180340b80f0ac",
        "source_payload_sha256": "cac4b32a7c08eb75b200193045df885cd4808233e4b4a4dfe916c92273582e97",
        "range_start_inclusive": 21_512_008_176,
        "range_end_inclusive": 22_116_235_759,
    },
    5: {
        "file_bytes": 604_231_488,
        "file_sha256": "140ffda85831042315ff0c21b9e6bc2df7ad523508aed0cd065c43e07cca5e3f",
        "manifest_sha256": "036980294835bf4fa7e803d834a25a2266c5dd4feb03398b8c1251c7dd404ad8",
        "source_payload_sha256": "393808d3db42ef0cc4edb5e898ad62795f9465684e76f91cc645cb771f7edefe",
        "range_start_inclusive": 28_158_511_600,
        "range_end_inclusive": 28_762_739_183,
    },
    6: {
        "file_bytes": 604_231_488,
        "file_sha256": "90c135deb5d0f0031ed5e69e6032667bdf4a91edbb87396203ea6880d2d73e22",
        "manifest_sha256": "9cdd4a9274eea986935d0c54b34e8412de2ba5239dee471d904142ec89dada77",
        "source_payload_sha256": "d7dfed1e1a41bb9d50093625f61625a30e2a139f0b19110b802cd034ebc1a01a",
        "range_start_inclusive": 31_179_649_520,
        "range_end_inclusive": 31_783_877_103,
    },
    7: {
        "file_bytes": 604_231_488,
        "file_sha256": "658bfc21ddd09a5ca03130e08fcdcece16fc4ce75bfcab5ec2c969caef5218d5",
        "manifest_sha256": "92d1216d02de9fa5eed098fe1ac73d21c767ed49aed60ea3fb395006011c86a2",
        "source_payload_sha256": "5d6cbc18d4b4ae5ac56187ca04042c84fad67e608a2c88fae07363d9099cd62f",
        "range_start_inclusive": 31_783_877_104,
        "range_end_inclusive": 32_388_104_687,
    },
}


class ReferenceError(RuntimeError):
    """A fail-closed input, provenance, execution, or fixture error."""


def _same_json_types(expected: Any, actual: Any) -> bool:
    return four_conditioned._same_json_types(expected, actual)


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReferenceError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _element_count(shape: list[int] | tuple[int, ...]) -> int:
    return four_conditioned._element_count(shape)


def validate_tensor_layout(tensors: dict[str, Any], sidecar_bytes: int) -> None:
    if type(sidecar_bytes) is not int or not 0 < sidecar_bytes <= SIDE_CAR_LIMIT:
        raise ReferenceError("eight-block tensor sidecar size outside 512 MiB limit")
    if not isinstance(tensors, dict) or set(tensors) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("eight-block tensor inventory is incomplete or unexpected")
    cursor = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        shape = EXPECTED_TENSOR_SHAPES[name]
        metadata = tensors.get(name)
        nbytes = _element_count(shape) * 4
        if (not isinstance(metadata, dict) or metadata.get("dtype") != "F32"
                or type(metadata.get("shape")) is not list
                or not _same_json_types(shape, metadata.get("shape"))
                or type(metadata.get("offset_bytes")) is not int
                or metadata["offset_bytes"] != cursor
                or type(metadata.get("nbytes")) is not int
                or metadata["nbytes"] != nbytes):
            raise ReferenceError(f"eight-block tensor range/shape mismatch: {name}")
        if nbytes > sidecar_bytes - cursor:
            raise ReferenceError(f"eight-block tensor range exceeds sidecar: {name}")
        cursor += nbytes
    if cursor != sidecar_bytes:
        raise ReferenceError("eight-block tensor sidecar has trailing or missing bytes")


def _validate_fixed_extra_block(name: str, value: Any, index: int,
                                pin: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ReferenceError(f"{name} source metadata is missing")
    expected: dict[str, Any] = {
        "block_index": index,
        "runtime_layer_name": f"double_block_{index + 1}",
        "file_bytes": pin["file_bytes"],
        "file_sha256": pin["file_sha256"],
        "manifest_sha256": pin["manifest_sha256"],
        "source_repository": serial_reference.SOURCE_REPOSITORY,
        "source_revision": serial_reference.SOURCE_CHECKPOINT_REVISION,
        "source_git_commit": PINNED_REVISION,
        "checkpoint_path": serial_reference.SOURCE_CHECKPOINT_PATH,
        "checkpoint_lfs_sha256": serial_reference.SOURCE_CHECKPOINT_LFS_SHA256,
        "source_payload_sha256": pin["source_payload_sha256"],
        "source_header_sha256": SOURCE_HEADER_SHA256,
        "range_start_inclusive": pin["range_start_inclusive"],
        "range_end_inclusive": pin["range_end_inclusive"],
        "tensor_prefix": f"double_blocks.{index}.",
        "tensor_count": 32,
    }
    for key, expected_value in expected.items():
        actual = value.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ReferenceError(f"{name} pinned source {key} mismatch")
    for field, basename in (
        ("path", f"distilled-block{index}-f32.safetensors"),
        ("manifest_path", f"distilled-block{index}-f32.safetensors.manifest.json"),
    ):
        path = value.get(field)
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or Path(path).name != basename):
            raise ReferenceError(f"{name} source {field} is invalid")


def _validate_block_sources(source: Any) -> None:
    order = [f"block{index}" for index in range(BLOCK_COUNT)]
    if (not isinstance(source, dict) or source.get("block_order") != order
            or not isinstance(source.get("blocks"), dict)
            or set(source["blocks"]) != set(order)):
        raise ReferenceError("conditioned eight-block order/inventory mismatch")
    blocks = source["blocks"]
    for name in ("block0", "block1"):
        try:
            serial_reference._validate_block_document(name, blocks[name])
        except (serial_reference.ReferenceError, KeyError, TypeError) as exc:
            raise ReferenceError(f"{name} source metadata mismatch: {exc}") from None
    for name in ("block2", "block3"):
        try:
            four_reference._validate_dynamic_block_document(name, blocks[name])
        except (four_reference.ReferenceError, KeyError, TypeError) as exc:
            raise ReferenceError(f"{name} source metadata mismatch: {exc}") from None
        index = int(name.removeprefix("block"))
        _validate_fixed_extra_block(name, blocks[name], index, BLOCK23_SOURCE_PINS[index])
    for index in range(4, 8):
        _validate_fixed_extra_block(
            f"block{index}", blocks[f"block{index}"], index,
            BLOCK47_SOURCE_PINS[index],
        )

    digests = {"file_sha256": set(), "manifest_sha256": set(),
               "source_payload_sha256": set()}
    for name in order:
        block = blocks[name]
        for field, seen in digests.items():
            digest = block.get(field)
            if not isinstance(digest, str) or digest in seen:
                raise ReferenceError("eight distinct block source identities are required")
            seen.add(digest)


def validate_reference_document(document: dict[str, Any]) -> None:
    """Validate the eight-block fixture envelope without importing Torch."""
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1):
        raise ReferenceError("missing or invalid conditioned eight-block schema")
    upstream = document.get("upstream")
    if (not isinstance(upstream, dict)
            or upstream.get("revision") != PINNED_REVISION
            or not _same_json_types(PINNED_UPSTREAM_SOURCE_SHA256,
                                    upstream.get("source_sha256"))):
        raise ReferenceError("conditioned pinned upstream source inventory mismatch")
    if not _same_json_types(GEOMETRY, document.get("geometry")):
        raise ReferenceError("conditioned full-J geometry mismatch")
    if not _same_json_types(INPUT_GENERATION, document.get("input_generation")):
        raise ReferenceError("conditioned video/RoPE generation recipe mismatch")
    if not _same_json_types(PRECISION, document.get("precision")):
        raise ReferenceError("conditioned precision policy mismatch")
    if not _same_json_types(text_reference.QWEN_SOURCE, document.get("qwen_source")):
        raise ReferenceError("conditioned Qwen source pin mismatch")
    mask = document.get("mask")
    if (not isinstance(mask, list) or len(mask) != TEXT_TOKENS
            or any(type(value) is not int for value in mask)
            or mask != [1] * VALID_TEXT_TOKENS + [0] * PADDED_TEXT_TOKENS):
        raise ReferenceError("conditioned Qwen crop mask mismatch")
    timestep = document.get("timestep")
    if type(timestep) not in (int, float) or float(timestep) != 500.0:
        raise ReferenceError("conditioned timestep mismatch")
    try:
        four_conditioned._validate_text_source(document.get("text_source"))
    except (four_conditioned.ReferenceError, KeyError, TypeError) as exc:
        raise ReferenceError(f"conditioned text source metadata mismatch: {exc}") from None
    _validate_block_sources(document.get("source"))

    expected_padding = {
        "kind": "unpad_zero_tokens",
        "valid_joint_prefix": VALID_JOINT_TOKENS,
        "valid_text_rows": VALID_TEXT_TOKENS,
        "padded_text_rows": PADDED_TEXT_TOKENS,
        "refiner_rows": TEXT_TOKENS,
        "padded_query_outputs": "attention result explicitly zero-filled",
    }
    if not _same_json_types(expected_padding, document.get("padding_policy")):
        raise ReferenceError("conditioned padding/unpadding policy mismatch")
    tolerance = document.get("tolerance")
    if (not isinstance(tolerance, dict)
            or type(tolerance.get("atol")) not in (int, float)
            or type(tolerance.get("rtol")) not in (int, float)
            or float(tolerance["atol"]) != ATOL
            or float(tolerance["rtol"]) != RTOL):
        raise ReferenceError("conditioned unchanged tolerance mismatch")

    expected_blocks = [f"block{index}" for index in range(BLOCK_COUNT)]
    producer = document.get("producer")
    expected_receipt = {
        "device": "mps",
        "activation_readback": "one packed final F32 tensor sidecar readback",
        "conditioning_invocations": 1,
        "refiner_rows": TEXT_TOKENS,
        "refiner_attention_calls": four_conditioned.REFINER_ATTENTION_CALLS,
        "block_invocations": expected_blocks,
        "attention_adapter": (
            "PyTorch MPS SDPA F32; valid joint prefix 5303; query rows chunked at 256"
        ),
        "attention_adapter_calls": BLOCK_COUNT,
        "attention_query_chunks_per_block": ATTENTION_CHUNKS_PER_BLOCK,
        "attention_query_chunk_calls": TOTAL_ATTENTION_CHUNKS,
        "padding_poison_control_calls": 1,
        "padding_poison_control_query_chunks": ATTENTION_CHUNKS_PER_BLOCK,
        "padding_poison_control_passed": True,
        "block_handoff": {
            "img": "each block output passed directly as next block input",
            "txt": "each block output passed directly as next block input",
            "same_mps_tensor_objects": True,
        },
        "captures_per_block": ["img.output", "txt.output"],
        "terminal_tensor_count": len(EXPECTED_TENSOR_SHAPES),
        "max_attention_query_chunk": QUERY_CHUNK,
    }
    if not isinstance(producer, dict):
        raise ReferenceError("conditioned eight-block producer receipt is missing")
    for key, expected in expected_receipt.items():
        if not _same_json_types(expected, producer.get(key)):
            raise ReferenceError(f"conditioned producer receipt mismatch: {key}")

    tensor_file = document.get("tensor_file")
    if (not isinstance(tensor_file, dict)
            or not isinstance(tensor_file.get("name"), str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+\.tensors\.f32le",
                            tensor_file["name"]) is None
            or tensor_file.get("encoding") != "f32-le"
            or type(tensor_file.get("nbytes")) is not int
            or not isinstance(tensor_file.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", tensor_file["sha256"]) is None):
        raise ReferenceError("conditioned eight-block tensor sidecar metadata is invalid")
    validate_tensor_layout(document.get("tensors"), tensor_file["nbytes"])
    if (type(producer.get("terminal_sidecar_bytes")) is not int
            or producer["terminal_sidecar_bytes"] != tensor_file["nbytes"]):
        raise ReferenceError("conditioned producer sidecar byte count mismatch")


def _validate_fixed_pair(name: str, pair: tuple[Path, str, Path, dict[str, Any]],
                         index: int, pin: dict[str, Any]) -> None:
    path, file_sha, manifest_path, manifest = pair
    if (path.stat().st_size != pin["file_bytes"]
            or file_sha != pin["file_sha256"]
            or four_conditioned.sha256_file(manifest_path) != pin["manifest_sha256"]
            or manifest["source"].get("payload_sha256_observed")
            != pin["source_payload_sha256"]
            or manifest["source"].get("range_start_inclusive")
            != pin["range_start_inclusive"]
            or manifest["source"].get("range_end_inclusive")
            != pin["range_end_inclusive"]):
        raise ReferenceError(f"{name} source bytes do not match the fixed eight-block pins")
    if manifest["source"].get("header_sha256_observed") != SOURCE_HEADER_SHA256:
        raise ReferenceError(f"{name} authenticated header digest mismatch")


def _validate_block47_pair(name: str, index: int, weights: Path,
                           weights_sha256: str, manifest_path: Path,
                           manifest_sha256: str) -> tuple[Path, str, Path, dict[str, Any]]:
    pin = BLOCK47_SOURCE_PINS[index]
    try:
        inventory = block47_reference.validate_source_pair(
            weights=weights,
            weights_sha256=weights_sha256,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            block_index=index,
        )
    except (block47_reference.ReferenceError, OSError, ValueError) as exc:
        raise ReferenceError(f"{name} authenticated source admission failed: {exc}") from None
    if (inventory.file_bytes != pin["file_bytes"]
            or inventory.file_sha256 != pin["file_sha256"]
            or inventory.manifest_sha256 != pin["manifest_sha256"]
            or inventory.payload_sha256 != pin["source_payload_sha256"]
            or inventory.manifest["source"].get("range_start_inclusive")
            != pin["range_start_inclusive"]
            or inventory.manifest["source"].get("range_end_inclusive")
            != pin["range_end_inclusive"]
            or inventory.manifest["source"].get("header_sha256_observed")
            != SOURCE_HEADER_SHA256):
        raise ReferenceError(f"{name} source identity differs from fixed producer pins")
    return (inventory.path, inventory.file_sha256,
            inventory.manifest_path, inventory.manifest)


def _source_block_metadata(
    weights: dict[str, tuple[Path, str, Path, dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (path, file_sha, manifest_path, manifest) in weights.items():
        source = manifest["source"]
        index = int(name.removeprefix("block"))
        result[name] = {
            "block_index": index,
            "runtime_layer_name": f"double_block_{index + 1}",
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "file_sha256": file_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": four_conditioned.sha256_file(manifest_path),
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


def _serialize_eight_tensors(torch: Any, records: dict[str, Any],
                             padding_control: Any) -> tuple[bytes, dict[str, Any], bool]:
    if set(records) != set(EXPECTED_TENSOR_SHAPES):
        raise ReferenceError("terminal conditioned eight-block tensor inventory mismatch")
    flat: list[Any] = []
    finite: list[Any] = []
    metadata: dict[str, Any] = {}
    offset = 0
    for name in sorted(EXPECTED_TENSOR_SHAPES):
        value = records[name]
        shape = EXPECTED_TENSOR_SHAPES[name]
        if (value.device.type != "mps" or value.dtype != torch.float32
                or list(value.shape) != shape):
            raise ReferenceError(f"terminal conditioned tensor device/dtype/shape mismatch: {name}")
        contiguous = value.detach().contiguous()
        nbytes = _element_count(shape) * 4
        metadata[name] = {
            "dtype": "F32", "shape": list(shape),
            "offset_bytes": offset, "nbytes": nbytes,
        }
        offset += nbytes
        flat.append(contiguous.reshape(-1))
        finite.append(torch.isfinite(contiguous).all().to(dtype=torch.float32).reshape(1))
    if offset != 489_070_592 or offset > SIDE_CAR_LIMIT or padding_control is None:
        raise ReferenceError("eight-block sidecar extent or padding control is invalid")
    packed = torch.cat(flat + finite + [padding_control]).to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    array = packed.numpy()
    value_count = offset // 4
    finite_stop = value_count + len(finite)
    if not bool((array[value_count:finite_stop] == 1.0).all()):
        raise ReferenceError("non-finite value in conditioned eight-block tensors")
    padding_passed = bool(array[finite_stop] == 1.0)
    if not padding_passed:
        raise ReferenceError("poisoned masked Q/K/V suffix changed valid attention")
    raw = array[:value_count].astype("<f4", copy=False).tobytes()
    if len(raw) != offset:
        raise ReferenceError("conditioned eight-block F32 sidecar extent mismatch")
    validate_tensor_layout(metadata, len(raw))
    return raw, metadata, padding_passed


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new conditioned reference JSON in an existing directory")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--qwen-reference", type=Path, default=DEFAULT_QWEN_REFERENCE)
    parser.add_argument("--qwen-sidecar", type=Path, default=DEFAULT_QWEN_SIDECAR)
    parser.add_argument("--text-weights", type=Path, default=DEFAULT_TEXT_WEIGHTS)
    parser.add_argument("--text-weights-sha256", required=True)
    parser.add_argument("--text-manifest", type=Path, default=DEFAULT_TEXT_MANIFEST)
    parser.add_argument("--text-manifest-sha256", required=True)
    defaults: dict[int, tuple[Path, Path]] = {
        0: (serial_reference.DEFAULT_BLOCK0_WEIGHTS,
            serial_reference.DEFAULT_BLOCK0_MANIFEST),
        1: (serial_reference.DEFAULT_BLOCK1_WEIGHTS,
            serial_reference.DEFAULT_BLOCK1_MANIFEST),
        2: (four_reference.DEFAULT_BLOCK2_WEIGHTS,
            four_reference.DEFAULT_BLOCK2_MANIFEST),
        3: (four_reference.DEFAULT_BLOCK3_WEIGHTS,
            four_reference.DEFAULT_BLOCK3_MANIFEST),
        **{
            index: (block47_reference.DEFAULT_WEIGHTS[index],
                    block47_reference.DEFAULT_MANIFESTS[index])
            for index in range(4, 8)
        },
    }
    for index, (weights, manifest) in defaults.items():
        parser.add_argument(f"--block{index}-weights", type=Path, default=weights)
        parser.add_argument(f"--block{index}-weights-sha256", required=True)
        parser.add_argument(f"--block{index}-manifest", type=Path, default=manifest)
        parser.add_argument(f"--block{index}-manifest-sha256", required=True)
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
        raise ReferenceError("conditioned eight-block sidecar already exists")

    text_path, text_manifest_path, _text_manifest = four_conditioned._verify_text_weights(
        args.text_weights, args.text_manifest,
        args.text_weights_sha256, args.text_manifest_sha256,
    )
    qwen_reference_path = args.qwen_reference
    qwen_sidecar_path = args.qwen_sidecar
    for label, path in (("Qwen reference", qwen_reference_path),
                        ("Qwen sidecar", qwen_sidecar_path)):
        if path.is_symlink():
            raise ReferenceError(f"{label} symlink is not admitted")
    qwen_reference_path = qwen_reference_path.resolve(strict=True)
    qwen_sidecar_path = qwen_sidecar_path.resolve(strict=True)
    if (not qwen_reference_path.is_file()
            or qwen_reference_path.stat().st_size > JSON_LIMIT
            or not qwen_sidecar_path.is_file()
            or qwen_sidecar_path.stat().st_size != QWEN_SIDECAR_BYTES):
        raise ReferenceError("Qwen fixture file type or extent mismatch")
    qwen_reference_sha = four_conditioned.sha256_file(qwen_reference_path)
    qwen_sidecar_sha = four_conditioned.sha256_file(qwen_sidecar_path)
    if (qwen_reference_sha != text_reference.QWEN_REFERENCE_SHA256
            or qwen_sidecar_sha != text_reference.QWEN_SIDECAR_SHA256):
        raise ReferenceError("Qwen reference or BF16 sidecar digest mismatch")
    try:
        qwen_document = json.loads(qwen_reference_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError("Qwen reference is invalid bounded UTF-8 JSON") from None
    valid_indices = text_reference.validate_qwen_reference(
        qwen_document,
        reference_sha256=qwen_reference_sha,
        sidecar_bytes=qwen_sidecar_path.stat().st_size,
        sidecar_sha256=qwen_sidecar_sha,
    )

    block_weights: dict[str, tuple[Path, str, Path, dict[str, Any]]] = {}
    for index in (0, 1):
        name = f"block{index}"
        block_weights[name] = serial_reference._trusted_weight_pair(
            name=name,
            weights=getattr(args, f"block{index}_weights"),
            weights_sha256=getattr(args, f"block{index}_weights_sha256"),
            manifest_path=getattr(args, f"block{index}_manifest"),
            manifest_sha256=getattr(args, f"block{index}_manifest_sha256"),
        )
    for index in (2, 3):
        name = f"block{index}"
        pair = four_reference._trusted_dynamic_pair(
            name=name,
            weights=getattr(args, f"block{index}_weights"),
            weights_sha256=getattr(args, f"block{index}_weights_sha256"),
            manifest_path=getattr(args, f"block{index}_manifest"),
            manifest_sha256=getattr(args, f"block{index}_manifest_sha256"),
        )
        _validate_fixed_pair(name, pair, index, BLOCK23_SOURCE_PINS[index])
        block_weights[name] = pair
    for index in range(4, 8):
        name = f"block{index}"
        block_weights[name] = _validate_block47_pair(
            name, index,
            getattr(args, f"block{index}_weights"),
            getattr(args, f"block{index}_weights_sha256"),
            getattr(args, f"block{index}_manifest"),
            getattr(args, f"block{index}_manifest_sha256"),
        )
    if len({block_weights[f"block{i}"][1] for i in range(8)}) != 8:
        raise ReferenceError("all eight block source file digests must be distinct")
    if len({four_conditioned.sha256_file(block_weights[f"block{i}"][2])
            for i in range(8)}) != 8:
        raise ReferenceError("all eight block manifest digests must be distinct")

    upstream = args.upstream.resolve(strict=True)
    source_hashes = four_conditioned._verify_upstream(upstream)

    import numpy as np
    import torch
    from safetensors import safe_open

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
    device = torch.device("mps")
    runtime_started = False
    try:
        runtime_started = True
        helpers = block_reference._install_upstream_modules(upstream)
        text_dir = upstream / "hymm/models/text_encoder"
        block_reference._namespace("hymm.models.text_encoder", text_dir)
        text_reference._install_text_attention(
            torch, valid_indices=valid_indices, precision="f32", capture={},
        )
        text_module = __import__("importlib").import_module(
            "hymm.models.text_encoder.token_refiner"
        )
        embed_layers = __import__("importlib").import_module(
            "hymm.models.dit.modules.embed_layers"
        )

        qwen_map = np.memmap(
            qwen_sidecar_path, mode="r", dtype="<u2", shape=(27, 1, 1388, 3584)
        )
        raw_rows = np.array(qwen_map[26, 0, 108:1388], copy=True)
        del qwen_map
        qwen_bf16 = torch.from_numpy(raw_rows).view(torch.bfloat16)
        del raw_rows
        qwen_full = qwen_bf16.unsqueeze(0).to(device=device, dtype=torch.float32)
        del qwen_bf16
        mask_values = [1] * VALID_TEXT_TOKENS + [0] * PADDED_TEXT_TOKENS
        full_mask = torch.tensor([mask_values], dtype=torch.bool, device=device)
        timestep = torch.tensor([500.0], dtype=torch.float32, device=device)

        text_state = text_reference._text_state(torch, text_path)
        time_state = text_reference._time_state(torch, text_path)
        refiner = text_module.SingleTokenRefinerPerBlock(
            in_channels=3584, hidden_size=2048, heads_num=16, depth=2,
            mlp_width_ratio=4.0, mlp_drop_rate=0.0, act_type="silu",
            qk_norm=False, qkv_bias=True, attn_mode="torch", seg_pos_emb_num=0,
            dtype=torch.float32, device=torch.device("meta"),
        )
        refiner.load_state_dict(text_state, strict=True, assign=True)
        refiner.to(device=device, dtype=torch.float32).eval()
        del text_state
        time_in = embed_layers.TimestepEmbedder(
            2048, torch.nn.SiLU, dtype=torch.float32, device=torch.device("meta")
        )
        time_in.load_state_dict(time_state, strict=True, assign=True)
        time_in.to(device=device, dtype=torch.float32).eval()
        del time_state
        block_reference._round_parameters_bf16_in_place(torch, refiner, device)
        block_reference._round_parameters_bf16_in_place(torch, time_in, device)

        records: dict[str, Any] = {}
        projected_hook = four_conditioned._capture_projected_text(
            refiner.individual_token_refiner, records
        )
        try:
            with torch.inference_mode():
                vec = time_in(timestep)
                _refiner_output = refiner(qwen_full, timestep, full_mask)
        finally:
            projected_hook.remove()
        projected_text = records.get("conditioning.projected_text")
        if (projected_text is None
                or tuple(projected_text.shape) != (1, TEXT_TOKENS, 2048)
                or tuple(vec.shape) != (1, 2048)
                or any(value.device.type != "mps" or value.dtype != torch.float32
                       for value in (projected_text, vec))):
            raise ReferenceError("full masked refiner conditioning output geometry/device mismatch")
        records["conditioning.vec"] = vec

        input_set = full_j_reference._generate_mps_inputs(torch, helpers[-1], device)
        img = input_set["img"]
        rope_cos = input_set["rope_cos"]
        rope_sin = input_set["rope_sin"]
        del input_set
        records["input.img"] = img
        records["input.rope_cos"] = rope_cos
        records["input.rope_sin"] = rope_sin

        receipt: dict[str, Any] = {
            "calls": 0,
            "query_chunks": 0,
            "max_query_chunk": 0,
            "padding_control": None,
            "padding_control_chunks": 0,
        }
        adapter = four_conditioned._build_chunked_unpad_adapter(torch, receipt)
        block_inputs = {
            "img": img,
            "txt": projected_text,
            "vec": vec,
            "vec_txt": vec,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
        }
        model_roots: list[Any] = []
        for index in range(BLOCK_COUNT):
            name = f"block{index}"
            state = serial_reference._load_block_state(
                torch, block_weights[name][0], index
            )
            img_output, txt_output, model = four_reference._run_one_block(
                torch, upstream, helpers, state, index, device, adapter,
                block_inputs, receipt,
            )
            records[f"{name}.img.output"] = img_output
            records[f"{name}.txt.output"] = txt_output
            model_roots.append(model)
            block_inputs = {**block_inputs, "img": img_output, "txt": txt_output}
        if (receipt["calls"] != BLOCK_COUNT
                or receipt["query_chunks"] != TOTAL_ATTENTION_CHUNKS
                or receipt["padding_control_chunks"] != ATTENTION_CHUNKS_PER_BLOCK):
            raise ReferenceError("conditioned eight-block attention receipt mismatch")
        raw_tensor_bytes, tensor_metadata, padding_passed = _serialize_eight_tensors(
            torch, records, receipt["padding_control"]
        )
        if not padding_passed:
            raise ReferenceError("masked Q/K/V poison falsifier failed")
    except BaseException:
        if runtime_started:
            try:
                torch.mps.synchronize()
            except BaseException:
                pass
        raise

    tensor_sha = hashlib.sha256(raw_tensor_bytes).hexdigest()
    text_source = {
        "path": str(text_path),
        "nbytes": text_path.stat().st_size,
        "sha256": text_reference.TEXT_WEIGHTS_SHA256,
        "manifest_path": str(text_manifest_path),
        "manifest_sha256": text_reference.TEXT_MANIFEST_SHA256,
        "tensor_count": 42,
        "dtype": "F32",
        **text_reference.TEXT_SOURCE,
        "source_payload_sha256": "8b60f66d65d54332ecd96aa89ba4fb6c4565cb11cefccee50f4f68138301b1a5",
        "source_header_sha256": SOURCE_HEADER_SHA256,
    }
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "upstream": {"revision": PINNED_REVISION, "source_sha256": source_hashes},
        "qwen_source": dict(text_reference.QWEN_SOURCE),
        "text_source": text_source,
        "source": {
            "block_order": [f"block{index}" for index in range(BLOCK_COUNT)],
            "blocks": _source_block_metadata(block_weights),
        },
        "geometry": dict(GEOMETRY),
        "input_generation": copy.deepcopy(INPUT_GENERATION),
        "precision": copy.deepcopy(PRECISION),
        "mask": mask_values,
        "timestep": 500.0,
        "padding_policy": {
            "kind": "unpad_zero_tokens",
            "valid_joint_prefix": VALID_JOINT_TOKENS,
            "valid_text_rows": VALID_TEXT_TOKENS,
            "padded_text_rows": PADDED_TEXT_TOKENS,
            "refiner_rows": TEXT_TOKENS,
            "padded_query_outputs": "attention result explicitly zero-filled",
        },
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "producer": {
            "path": str(Path(__file__).resolve()),
            "source_sha256": four_conditioned.sha256_file(Path(__file__).resolve()),
            "device": "mps",
            "activation_readback": "one packed final F32 tensor sidecar readback",
            "conditioning_invocations": 1,
            "refiner_rows": TEXT_TOKENS,
            "refiner_attention_calls": four_conditioned.REFINER_ATTENTION_CALLS,
            "block_invocations": [f"block{index}" for index in range(BLOCK_COUNT)],
            "attention_adapter": (
                "PyTorch MPS SDPA F32; valid joint prefix 5303; query rows chunked at 256"
            ),
            "attention_adapter_calls": receipt["calls"],
            "attention_query_chunks_per_block": ATTENTION_CHUNKS_PER_BLOCK,
            "attention_query_chunk_calls": receipt["query_chunks"],
            "max_attention_query_chunk": receipt["max_query_chunk"],
            "padding_poison_control_calls": 1,
            "padding_poison_control_query_chunks": receipt["padding_control_chunks"],
            "padding_poison_control_passed": padding_passed,
            "block_handoff": {
                "img": "each block output passed directly as next block input",
                "txt": "each block output passed directly as next block input",
                "same_mps_tensor_objects": True,
            },
            "captures_per_block": ["img.output", "txt.output"],
            "terminal_tensor_count": len(records),
            "terminal_sidecar_bytes": len(raw_tensor_bytes),
        },
        "limitations": [
            "Qwen state 26 and the 108:1388 crop come from the authenticated saved BF16 capture; Qwen is not rerun",
            "the full 1280-row refiner is evaluated; its MPS attention adapter unpads the known 23-row valid prefix",
            "video input is deterministic synthetic MPS F32 and all outputs stop after actual DiT block 7",
            "PyTorch MPS F32 SDPA with explicit prefix unpadding and zero-padded query outputs is not Magi/CUDA parity",
            "parameters use F32 storage containing BF16-rounded values; physical BF16 storage and native BF16 compute are not tested",
            "the eight-block chain is a bounded prefix probe, not full DiT execution or media generation",
            "the declared whole-checkpoint HF LFS digest is not verified; extracted block payloads and source ranges are authenticated individually",
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
    json_bytes = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(json_bytes) > JSON_LIMIT:
        raise ReferenceError("conditioned eight-block JSON exceeds 2 MiB")
    four_reference._publish_noreplace(tensor_path, raw_tensor_bytes)
    try:
        four_reference._publish_noreplace(output, json_bytes)
    except BaseException:
        tensor_path.unlink(missing_ok=True)
        raise
    return document


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    try:
        document = produce_fixture(args)
    except (OSError, ValueError, ReferenceError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({
        "fixture": str(args.output),
        "schema": SCHEMA,
        "sidecar": document["tensor_file"],
        "blocks": document["source"]["block_order"],
        "padding_poison_control_passed": (
            document["producer"]["padding_poison_control_passed"]
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
