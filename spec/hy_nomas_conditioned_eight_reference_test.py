#!/usr/bin/env python3
"""CPU-only admission tests for the conditioned eight-block MPS oracle."""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_conditioned_eight_reference as reference  # noqa: E402
import hy_nomas_conditioned_four_reference as four_conditioned  # noqa: E402
import hy_nomas_four_block_reference as four_reference  # noqa: E402
import hy_nomas_pretrained_serial_reference as serial_reference  # noqa: E402
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
    if index in (2, 3):
        pin = reference.BLOCK23_SOURCE_PINS[index]
    else:
        pin = reference.BLOCK47_SOURCE_PINS[index]
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
        "source_header_sha256": reference.SOURCE_HEADER_SHA256,
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
            "source_header_sha256": reference.SOURCE_HEADER_SHA256,
        },
        "source": {
            "block_order": [f"block{i}" for i in range(8)],
            "blocks": {f"block{i}": _block_document(i) for i in range(8)},
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
            "refiner_attention_calls": four_conditioned.REFINER_ATTENTION_CALLS,
            "block_invocations": [f"block{i}" for i in range(8)],
            "attention_adapter": (
                "PyTorch MPS SDPA F32; valid joint prefix 5303; query rows chunked at 256"
            ),
            "attention_adapter_calls": 8,
            "attention_query_chunks_per_block": reference.ATTENTION_CHUNKS_PER_BLOCK,
            "attention_query_chunk_calls": reference.TOTAL_ATTENTION_CHUNKS,
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
            "name": "conditioned-eight-reference.tensors.f32le",
            "nbytes": tensor_bytes,
            "sha256": "a" * 64,
            "encoding": "f32-le",
        },
        "tensors": tensors,
    }


class HyNomasConditionedEightReferenceTests(unittest.TestCase):
    def test_accepts_eight_distinct_authenticated_blocks_and_exact_sidecar(self):
        document = valid_document()
        reference.validate_reference_document(document)
        self.assertEqual(len(reference.EXPECTED_TENSOR_SHAPES), 21)
        self.assertEqual(document["tensor_file"]["nbytes"], 489_070_592)
        self.assertEqual(reference.TOTAL_ATTENTION_CHUNKS, 168)
        self.assertEqual(document["source"]["block_order"], [f"block{i}" for i in range(8)])

    def test_captures_only_five_inputs_and_final_img_txt_per_block(self):
        names = set(reference.EXPECTED_TENSOR_SHAPES)
        self.assertEqual(
            {name for name in names if name.startswith("input.")},
            {"input.img", "input.rope_cos", "input.rope_sin"},
        )
        self.assertEqual(
            {name for name in names if name.startswith("conditioning.")},
            {"conditioning.projected_text", "conditioning.vec"},
        )
        for index in range(8):
            self.assertEqual(
                {name.removeprefix(f"block{index}.") for name in names
                 if name.startswith(f"block{index}.")},
                {"img.output", "txt.output"},
            )
        self.assertFalse(any("attn." in name or "ff." in name for name in names))

    def test_rejects_order_index_alias_and_fixed_source_hash_substitution(self):
        document = valid_document()
        document["source"]["block_order"] = [f"block{i}" for i in range(1, 8)] + ["block0"]
        with self.assertRaisesRegex(reference.ReferenceError, "order"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["source"]["blocks"]["block7"]["block_index"] = 6
        with self.assertRaisesRegex(reference.ReferenceError, "block_index"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["source"]["blocks"]["block7"]["file_sha256"] = \
            document["source"]["blocks"]["block6"]["file_sha256"]
        with self.assertRaises(reference.ReferenceError):
            reference.validate_reference_document(document)

        document = valid_document()
        document["source"]["blocks"]["block5"]["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(reference.ReferenceError, "manifest_sha256"):
            reference.validate_reference_document(document)

    def test_rejects_wrong_authenticated_ranges_and_header_for_blocks_4_to_7(self):
        for index, field in ((4, "range_start_inclusive"),
                             (5, "range_end_inclusive"),
                             (6, "source_payload_sha256"),
                             (7, "source_header_sha256")):
            document = valid_document()
            if field.startswith("range"):
                document["source"]["blocks"][f"block{index}"][field] += 1
            else:
                document["source"]["blocks"][f"block{index}"][field] = "0" * 64
            with self.subTest(index=index, field=field), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_rejects_conditioning_padding_tolerance_and_chunk_falsifiers(self):
        mutations = (
            lambda doc: doc.update(schema="hy-nomas-conditioned-four-reference-v1"),
            lambda doc: doc["qwen_source"].update(state_index=25),
            lambda doc: doc["text_source"].update(sha256="0" * 64),
            lambda doc: doc["mask"].__setitem__(0, 0),
            lambda doc: doc["padding_policy"].update(valid_joint_prefix=True),
            lambda doc: doc["tolerance"].update(rtol=1e-4),
            lambda doc: doc["producer"].update(attention_query_chunk_calls=167),
            lambda doc: doc["producer"].update(padding_poison_control_passed=1),
        )
        for mutation in mutations:
            document = valid_document()
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_rejects_tensor_range_gaps_bool_dimensions_and_trailing_bytes(self):
        mutations = (
            lambda doc: doc["tensors"]["block7.txt.output"].update(offset_bytes=0),
            lambda doc: doc["tensors"]["block0.img.output"]["shape"].__setitem__(0, True),
            lambda doc: doc["tensor_file"].update(nbytes=489_070_596),
        )
        for mutation in mutations:
            document = valid_document()
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_cli_requires_each_block_file_and_manifest_pin(self):
        parser = reference._build_cli()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--output", "fixture.json"])

        argv = [
            "--output", "fixture.json",
            "--text-weights-sha256", "1" * 64,
            "--text-manifest-sha256", "2" * 64,
        ]
        for index in range(8):
            argv.extend((f"--block{index}-weights-sha256", f"{index + 1:x}" * 64,
                         f"--block{index}-manifest-sha256", f"{index + 2:x}" * 64))
        args = parser.parse_args(argv)
        self.assertEqual(args.block7_weights_sha256, "8" * 64)
        self.assertEqual(args.block7_manifest_sha256, "9" * 64)

        argv.remove("--block7-manifest-sha256")
        argv.remove("9" * 64)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
