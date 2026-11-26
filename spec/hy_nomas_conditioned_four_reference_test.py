#!/usr/bin/env python3
"""CPU-only admission tests for the conditioned four-block MPS oracle."""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_conditioned_four_reference as reference  # noqa: E402
import hy_nomas_four_block_reference as four_reference  # noqa: E402
import hy_nomas_pretrained_serial_reference as serial_reference  # noqa: E402
import hy_nomas_reference as block_reference  # noqa: E402
import hy_nomas_text_conditioning_reference as text_reference  # noqa: E402


def _tensor_layout():
    tensors = {}
    offset = 0
    for name, shape in sorted(reference.EXPECTED_TENSOR_SHAPES.items()):
        nbytes = reference._element_count(shape) * 4
        tensors[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "offset_bytes": offset,
            "nbytes": nbytes,
        }
        offset += nbytes
    return tensors, offset


def _block_document(index: int):
    name = f"block{index}"
    if index in (0, 1):
        pin = serial_reference.BLOCK_PINS[name]
        return {
            "block_index": index,
            "runtime_layer_name": f"double_block_{index + 1}",
            "path": f"/weights/distilled-block{index}-f32.safetensors",
            "file_bytes": pin["file_bytes"],
            "file_sha256": pin["file_sha256"],
            "manifest_path": f"/weights/distilled-block{index}-f32.safetensors.manifest.json",
            "manifest_sha256": pin["manifest_sha256"],
            "source_repository": serial_reference.SOURCE_REPOSITORY,
            "source_revision": serial_reference.SOURCE_CHECKPOINT_REVISION,
            "source_git_commit": serial_reference.PINNED_REVISION,
            "checkpoint_path": serial_reference.SOURCE_CHECKPOINT_PATH,
            "checkpoint_lfs_sha256": serial_reference.SOURCE_CHECKPOINT_LFS_SHA256,
            "source_payload_sha256": pin["source_payload_sha256"],
            "source_header_sha256": pin["source_header_sha256"],
            "range_start_inclusive": pin["range_start_inclusive"],
            "range_end_inclusive": pin["range_end_inclusive"],
            "tensor_prefix": f"double_blocks.{index}.",
            "tensor_count": 32,
        }
    pin = four_reference.DYNAMIC_BLOCK_PINS[name]
    return {
        "block_index": index,
        "runtime_layer_name": f"double_block_{index + 1}",
        "path": f"/weights/distilled-block{index}-f32.safetensors",
        "file_bytes": pin["payload_bytes"] + 4096,
        "file_sha256": "b" * 64 if index == 2 else "c" * 64,
        "manifest_path": f"/weights/distilled-block{index}-f32.safetensors.manifest.json",
        "manifest_sha256": "d" * 64 if index == 2 else "e" * 64,
        "source_repository": serial_reference.SOURCE_REPOSITORY,
        "source_revision": serial_reference.SOURCE_CHECKPOINT_REVISION,
        "source_git_commit": serial_reference.PINNED_REVISION,
        "checkpoint_path": serial_reference.SOURCE_CHECKPOINT_PATH,
        "checkpoint_lfs_sha256": serial_reference.SOURCE_CHECKPOINT_LFS_SHA256,
        "source_payload_sha256": "f" * 64 if index == 2 else "9" * 64,
        "source_header_sha256": four_reference.SOURCE_HEADER_SHA256,
        "range_start_inclusive": pin["range_start_inclusive"],
        "range_end_inclusive": pin["range_end_inclusive"],
        "tensor_prefix": f"double_blocks.{index}.",
        "tensor_count": 32,
    }


def valid_document():
    tensors, tensor_bytes = _tensor_layout()
    return {
        "schema": reference.SCHEMA,
        "schema_version": 1,
        "upstream": {
            "revision": reference.PINNED_REVISION,
            "source_sha256": dict(reference.PINNED_UPSTREAM_SOURCE_SHA256),
        },
        "qwen_source": copy.deepcopy(text_reference.QWEN_SOURCE),
        "text_source": {
            "path": str(reference.DEFAULT_TEXT_WEIGHTS),
            "nbytes": 583_316_840,
            "sha256": text_reference.TEXT_WEIGHTS_SHA256,
            "manifest_path": str(reference.DEFAULT_TEXT_MANIFEST),
            "manifest_sha256": text_reference.TEXT_MANIFEST_SHA256,
            "tensor_count": 42,
            "dtype": "F32",
            **text_reference.TEXT_SOURCE,
            "source_payload_sha256": "8b60f66d65d54332ecd96aa89ba4fb6c4565cb11cefccee50f4f68138301b1a5",
            "source_header_sha256": "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200",
        },
        "source": {
            "block_order": [f"block{i}" for i in range(4)],
            "blocks": {f"block{i}": _block_document(i) for i in range(4)},
        },
        "geometry": copy.deepcopy(reference.GEOMETRY),
        "input_generation": copy.deepcopy(reference.INPUT_GENERATION),
        "precision": copy.deepcopy(reference.PRECISION),
        "mask": [1] * reference.VALID_TEXT_TOKENS + [0] * reference.PADDED_TEXT_TOKENS,
        "timestep": 500.0,
        "padding_policy": {
            "kind": "unpad_zero_tokens",
            "valid_joint_prefix": reference.VALID_JOINT_TOKENS,
            "valid_text_rows": reference.VALID_TEXT_TOKENS,
            "padded_text_rows": reference.PADDED_TEXT_TOKENS,
            "refiner_rows": reference.TEXT_TOKENS,
            "padded_query_outputs": "attention result explicitly zero-filled",
        },
        "tolerance": {"atol": reference.ATOL, "rtol": reference.RTOL},
        "producer": {
            "device": "mps",
            "activation_readback": "one packed final F32 tensor sidecar readback",
            "conditioning_invocations": 1,
            "refiner_rows": reference.TEXT_TOKENS,
            "refiner_attention_calls": reference.REFINER_ATTENTION_CALLS,
            "block_invocations": [f"block{i}" for i in range(4)],
            "attention_adapter": "PyTorch MPS SDPA F32; valid joint prefix 5303; query rows chunked at 256",
            "attention_adapter_calls": 4,
            "attention_query_chunks_per_block": reference.ATTENTION_CHUNKS_PER_BLOCK,
            "attention_query_chunk_calls": 4 * reference.ATTENTION_CHUNKS_PER_BLOCK,
            "padding_poison_control_calls": 1,
            "padding_poison_control_query_chunks": reference.ATTENTION_CHUNKS_PER_BLOCK,
            "padding_poison_control_passed": True,
            "block_handoff": {
                "img": "each block output passed directly as next block input",
                "txt": "each block output passed directly as next block input",
                "same_mps_tensor_objects": True,
            },
            "captures_per_block": ["img.output", "txt.output"],
            "terminal_tensor_count": len(reference.EXPECTED_TENSOR_SHAPES),
            "max_attention_query_chunk": reference.QUERY_CHUNK,
            "terminal_sidecar_bytes": tensor_bytes,
        },
        "tensor_file": {
            "name": "conditioned-four-reference.tensors.f32le",
            "nbytes": tensor_bytes,
            "sha256": "a" * 64,
            "encoding": "f32-le",
        },
        "tensors": tensors,
    }


class HyNomasConditionedFourReferenceTests(unittest.TestCase):
    def test_accepts_authenticated_four_block_conditioned_envelope(self):
        document = valid_document()
        reference.validate_reference_document(document)
        self.assertEqual(len(reference.EXPECTED_TENSOR_SHAPES), 13)
        self.assertEqual(document["tensor_file"]["nbytes"], 274_112_512)
        self.assertEqual(
            reference.attention_geometry(5280, 1280, 23, 256),
            {
                "tensor_tokens": 6560,
                "valid_joint_tokens": 5303,
                "padded_text_tokens": 1257,
                "query_chunks": 21,
            },
        )

    def test_captures_only_conditioning_outputs_and_block_img_txt_outputs(self):
        names = set(reference.EXPECTED_TENSOR_SHAPES)
        self.assertEqual(
            {name for name in names if name.startswith("input.")},
            {"input.img", "input.rope_cos", "input.rope_sin"},
        )
        self.assertEqual(
            {name for name in names if name.startswith("conditioning.")},
            {"conditioning.projected_text", "conditioning.vec"},
        )
        for index in range(4):
            self.assertEqual(
                {name.removeprefix(f"block{index}.") for name in names
                 if name.startswith(f"block{index}.")},
                {"img.output", "txt.output"},
            )

    def test_rejects_block_order_index_alias_and_source_range_substitution(self):
        document = valid_document()
        document["source"]["block_order"] = ["block0", "block2", "block1", "block3"]
        with self.assertRaisesRegex(reference.ReferenceError, "order"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["source"]["blocks"]["block3"]["block_index"] = 2
        with self.assertRaisesRegex(reference.ReferenceError, "block_index"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["source"]["blocks"]["block3"]["source_payload_sha256"] = \
            document["source"]["blocks"]["block2"]["source_payload_sha256"]
        with self.assertRaisesRegex(reference.ReferenceError, "alias"):
            reference.validate_reference_document(document)

        for name, key in (("block2", "range_start_inclusive"),
                          ("block3", "range_end_inclusive")):
            document = valid_document()
            document["source"]["blocks"][name][key] += 1
            with self.subTest(name=name, key=key), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_rejects_conditioning_source_mask_and_padding_falsifiers(self):
        for mutation in (
            lambda doc: doc["qwen_source"].update(state_index=25),
            lambda doc: doc["text_source"].update(sha256="0" * 64),
            lambda doc: doc["text_source"].update(manifest_sha256="0" * 64),
            lambda doc: doc["mask"].__setitem__(0, 0),
            lambda doc: doc["padding_policy"].update(valid_joint_prefix=True),
            lambda doc: doc["producer"].update(padding_poison_control_passed=1),
        ):
            document = valid_document()
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_rejects_typed_metadata_tolerance_and_attention_receipt_changes(self):
        for mutation in (
            lambda doc: doc["geometry"].update(batches=True),
            lambda doc: doc["input_generation"]["rope"].update(use_real=1),
            lambda doc: doc["precision"].update(native_bf16_compute=1),
            lambda doc: doc["tolerance"].update(atol=1e-4),
            lambda doc: doc["producer"].update(attention_query_chunk_calls=80),
            lambda doc: doc["producer"].update(block_handoff={"img": "copied"}),
        ):
            document = valid_document()
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_rejects_missing_reordered_or_malformed_tensor_ranges(self):
        mutations = (
            lambda doc: doc["tensors"].pop("block3.txt.output"),
            lambda doc: doc["tensors"]["block1.img.output"].update(offset_bytes=4),
            lambda doc: doc["tensors"]["block2.txt.output"].update(shape=[True, 1280, 2048]),
            lambda doc: doc["tensor_file"].update(nbytes=doc["tensor_file"]["nbytes"] + 4),
            lambda doc: doc["tensor_file"].update(name="../escape.tensors.f32le"),
        )
        for mutation in mutations:
            document = valid_document()
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_attention_geometry_rejects_noninteger_or_impossible_shapes(self):
        for args in ((5280, 1280, 23, True), (5280, 1280, 1281, 256),
                     (5280, 1280, 23, 0), (0, 1280, 23, 256)):
            with self.subTest(args=args), self.assertRaises(reference.ReferenceError):
                reference.attention_geometry(*args)

    def test_cli_requires_all_weight_and_manifest_pins(self):
        parser = reference._build_cli()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parser.parse_args(["--output", "/tmp/reference.json"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
