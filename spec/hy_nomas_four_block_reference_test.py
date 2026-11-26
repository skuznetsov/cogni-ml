#!/usr/bin/env python3
"""CPU-only admission tests for the four-block Hy-Nomas MPS oracle."""

from __future__ import annotations

import ast
import copy
import contextlib
import io
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_four_block_reference as reference  # noqa: E402
import hy_nomas_pretrained_serial_reference as serial_reference  # noqa: E402
import hy_nomas_reference as block_reference  # noqa: E402


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


def _dynamic_file_bytes(index: int) -> int:
    return reference.DYNAMIC_BLOCK_PINS[f"block{index}"]["payload_bytes"] + 4096


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
    pin = reference.DYNAMIC_BLOCK_PINS[name]
    digit = str(index)
    return {
        "block_index": index,
        "runtime_layer_name": f"double_block_{index + 1}",
        "path": f"/weights/distilled-block{index}-f32.safetensors",
        "file_bytes": _dynamic_file_bytes(index),
        "file_sha256": digit * 64,
        "manifest_path": f"/weights/distilled-block{index}-f32.safetensors.manifest.json",
        "manifest_sha256": str(index + 1) * 64,
        "source_repository": serial_reference.SOURCE_REPOSITORY,
        "source_revision": serial_reference.SOURCE_CHECKPOINT_REVISION,
        "source_git_commit": serial_reference.PINNED_REVISION,
        "checkpoint_path": serial_reference.SOURCE_CHECKPOINT_PATH,
        "checkpoint_lfs_sha256": serial_reference.SOURCE_CHECKPOINT_LFS_SHA256,
        "source_payload_sha256": str(index + 2) * 64,
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
            "source_sha256": dict(block_reference.PINNED_SOURCE_SHA256),
        },
        "producer": {
            "device": "mps",
            "activation_readback": "one packed final F32 tensor sidecar readback",
            "attention_adapter": "PyTorch MPS SDPA F32, query rows chunked at 256",
            "attention_adapter_calls": 4,
            "attention_query_chunks_per_block": reference.ATTENTION_CHUNKS_PER_BLOCK,
            "attention_query_chunk_calls": 4 * reference.ATTENTION_CHUNKS_PER_BLOCK,
            "max_attention_query_chunk": reference.QUERY_CHUNK,
            "block_invocations": [f"block{i}" for i in range(4)],
            "block_handoff": {
                "img": "each block output passed directly as next block input",
                "txt": "each block output passed directly as next block input",
                "same_mps_tensor_objects": True,
            },
            "captures_per_block": ["img.output", "txt.output"],
            "terminal_tensor_count": len(reference.EXPECTED_TENSOR_SHAPES),
            "terminal_sidecar_bytes": tensor_bytes,
        },
        "source": {
            "block_order": [f"block{i}" for i in range(4)],
            "blocks": {f"block{i}": _block_document(i) for i in range(4)},
        },
        "geometry": copy.deepcopy(reference.GEOMETRY),
        "input_generation": copy.deepcopy(reference.INPUT_GENERATION),
        "precision": dict(reference.PRECISION),
        "tolerance": {"atol": reference.ATOL, "rtol": reference.RTOL},
        "tensor_file": {
            "name": "four-block-reference.tensors.f32le",
            "nbytes": tensor_bytes,
            "sha256": "a" * 64,
            "encoding": "f32-le",
        },
        "tensors": tensors,
    }


def valid_manifest(index: int):
    pin = reference.DYNAMIC_BLOCK_PINS[f"block{index}"]
    expected = block_reference._expected_block_keys(2048, 8192, 16)
    return {
        "format": "hy-nomas-block-subset-provenance-v1",
        "output": {
            "filename": f"distilled-block{index}-f32.safetensors",
            "sha256": str(index) * 64,
            "size_bytes": _dynamic_file_bytes(index),
        },
        "source": {
            "repository": serial_reference.SOURCE_REPOSITORY,
            "revision": serial_reference.SOURCE_CHECKPOINT_REVISION,
            "checkpoint_path": serial_reference.SOURCE_CHECKPOINT_PATH,
            "checkpoint_size_bytes": serial_reference.SOURCE_CHECKPOINT_BYTES,
            "checkpoint_lfs_sha256": serial_reference.SOURCE_CHECKPOINT_LFS_SHA256,
            "source_git_commit": serial_reference.PINNED_REVISION,
            "range_start_inclusive": pin["range_start_inclusive"],
            "range_end_inclusive": pin["range_end_inclusive"],
            "payload_sha256_observed": str(index + 2) * 64,
            "header_sha256_observed": reference.SOURCE_HEADER_SHA256,
        },
        "subset": {
            "block_index": index,
            "tensor_prefix": f"double_blocks.{index}.",
            "tensor_count": 32,
            "dtype": "F32",
            "stored_bytes_preserved": True,
            "conversion_performed": False,
            "payload_bytes": pin["payload_bytes"],
            "tensor_names": serial_reference._expected_source_names(index, expected),
        },
    }


class HyNomasFourBlockReferenceTests(unittest.TestCase):
    def test_accepts_complete_ordered_four_block_envelope(self):
        document = valid_document()
        reference.validate_reference_document(document)
        self.assertEqual(len(reference.EXPECTED_TENSOR_SHAPES), 14)
        self.assertEqual(document["tensor_file"]["nbytes"], 274_120_704)

    def test_captures_only_img_txt_outputs_for_each_block(self):
        names = set(reference.EXPECTED_TENSOR_SHAPES)
        self.assertEqual(
            {name for name in names if name.startswith("input.")},
            {
                "input.img", "input.txt", "input.vec", "input.vec_txt",
                "input.rope_cos", "input.rope_sin",
            },
        )
        for index in range(4):
            self.assertEqual(
                {name.removeprefix(f"block{index}.") for name in names
                 if name.startswith(f"block{index}.")},
                {"img.output", "txt.output"},
            )

    def test_rejects_source_order_and_index_substitution(self):
        document = valid_document()
        document["source"]["block_order"] = ["block0", "block2", "block1", "block3"]
        with self.assertRaisesRegex(reference.ReferenceError, "order"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["source"]["blocks"]["block3"]["block_index"] = 2
        with self.assertRaisesRegex(reference.ReferenceError, "block_index"):
            reference.validate_reference_document(document)

    def test_rejects_bad_block2_block3_source_ranges_or_header(self):
        for key, value in (
            ("range_start_inclusive", reference.DYNAMIC_BLOCK_PINS["block2"]["range_start_inclusive"] + 1),
            ("range_end_inclusive", reference.DYNAMIC_BLOCK_PINS["block3"]["range_end_inclusive"] - 1),
            ("source_header_sha256", "0" * 64),
        ):
            document = valid_document()
            target = "block3" if key == "range_end_inclusive" else "block2"
            document["source"]["blocks"][target][key] = value
            with self.subTest(key=key), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_rejects_bad_dynamic_manifest_inventory_or_source_identity(self):
        for mutation in (
            lambda manifest: manifest["subset"].update(block_index=1),
            lambda manifest: manifest["subset"]["tensor_names"].pop(),
            lambda manifest: manifest["source"].update(revision="0" * 40),
            lambda manifest: manifest["source"].update(header_sha256_observed="0" * 64),
        ):
            manifest = valid_manifest(2)
            mutation(manifest)
            with self.subTest(mutation=mutation), self.assertRaises(reference.ReferenceError):
                reference._validate_dynamic_manifest(
                    manifest,
                    name="block2",
                    filename="distilled-block2-f32.safetensors",
                    file_bytes=_dynamic_file_bytes(2),
                    file_sha256="2" * 64,
                )

    def test_rejects_aliases_tolerance_capture_receipt_or_nonfinite_policy(self):
        document = valid_document()
        document["source"]["blocks"]["block3"]["file_sha256"] = document["source"]["blocks"]["block2"]["file_sha256"]
        with self.assertRaisesRegex(reference.ReferenceError, "distinct|alias"):
            reference.validate_reference_document(document)

        for mutation in (
            lambda doc: doc["tolerance"].update(atol=1e-4),
            lambda doc: doc["producer"].update(attention_adapter_calls=3),
            lambda doc: doc["producer"].update(attention_query_chunks_per_block=25),
            lambda doc: doc["producer"].update(captures_per_block=["img.output"]),
        ):
            document = valid_document()
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_rejects_malformed_tensor_ranges_and_extra_capture(self):
        document = valid_document()
        document["tensors"]["block3.txt.output"]["offset_bytes"] += 4
        with self.assertRaisesRegex(reference.ReferenceError, "range|offset"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["tensors"]["block3.img.attn.q_rope"] = {
            "dtype": "F32", "shape": [1, 5280, 16, 128],
            "offset_bytes": 0, "nbytes": 43_253_760,
        }
        with self.assertRaisesRegex(reference.ReferenceError, "inventory"):
            reference.validate_reference_document(document)

    def test_cli_requires_all_four_explicit_weight_and_manifest_pins(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            reference._build_cli().parse_args(["--output", "/tmp/four-block-reference.json"])

    def test_metadata_import_path_does_not_import_torch(self):
        source_path = SUPPORT / "hy_nomas_four_block_reference.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(
            (alias.name == "torch" if isinstance(node, ast.Import)
             else node.module == "torch")
            for node in imports for alias in getattr(node, "names", ())
        ))


if __name__ == "__main__":
    unittest.main()
