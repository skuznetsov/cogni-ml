#!/usr/bin/env python3
"""CPU-only admission tests for the full-J Hy-Nomas MPS oracle."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_full_j_reference as reference  # noqa: E402
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


def valid_document():
    tensors, tensor_bytes = _tensor_layout()
    blocks = {}
    for name, index in (("block0", 0), ("block1", 1)):
        pin = serial_reference.BLOCK_PINS[name]
        blocks[name] = {
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
            "attention_adapter_calls": 2,
            "attention_query_chunks_per_block": reference.ATTENTION_CHUNKS_PER_BLOCK,
            "captures_per_block": [
                "img.attn.q_rope", "img.attn.k_rope", "img.output", "txt.output"
            ],
            "terminal_tensor_count": len(reference.EXPECTED_TENSOR_SHAPES),
        },
        "source": {"blocks": blocks},
        "geometry": dict(reference.GEOMETRY),
        "input_generation": copy.deepcopy(reference.INPUT_GENERATION),
        "precision": dict(reference.PRECISION),
        "tolerance": {"atol": reference.ATOL, "rtol": reference.RTOL},
        "tensor_file": {
            "name": "full-j-reference.tensors.f32le",
            "nbytes": tensor_bytes,
            "sha256": "a" * 64,
            "encoding": "f32-le",
        },
        "tensors": tensors,
    }


class HyNomasFullJReferenceTests(unittest.TestCase):
    def test_accepts_exact_full_j_inventory_and_geometry(self):
        document = valid_document()
        reference.validate_reference_document(document)
        self.assertEqual(len(reference.EXPECTED_TENSOR_SHAPES), 14)
        self.assertEqual(document["tensor_file"]["nbytes"], 339_656_704)

    def test_inventory_is_six_inputs_and_only_four_captures_per_block(self):
        names = set(reference.EXPECTED_TENSOR_SHAPES)
        self.assertEqual(
            {name for name in names if name.startswith("input.")},
            {
                "input.img", "input.txt", "input.vec", "input.vec_txt",
                "input.rope_cos", "input.rope_sin",
            },
        )
        for block_name in ("block0", "block1"):
            self.assertEqual(
                {name.removeprefix(f"{block_name}.") for name in names
                 if name.startswith(f"{block_name}.")},
                {"img.attn.q_rope", "img.attn.k_rope", "img.output", "txt.output"},
            )

    def test_rejects_geometry_formula_or_rope_substitution(self):
        for mutation in (
            lambda doc: doc["geometry"].update(rope_grid=[6, 20, 44]),
            lambda doc: doc["input_generation"]["phases"].update(img=0.12),
            lambda doc: doc["input_generation"]["rope"].update(axis_dims=[16, 56, 55]),
            lambda doc: doc["input_generation"]["rope"].update(device="cpu"),
        ):
            with self.subTest(mutation=mutation):
                document = valid_document()
                mutation(document)
                with self.assertRaises(reference.ReferenceError):
                    reference.validate_reference_document(document)

    def test_rejects_block_index_alias_or_source_pin_confusion(self):
        document = valid_document()
        document["source"]["blocks"]["block1"]["block_index"] = 0
        with self.assertRaisesRegex(reference.ReferenceError, "block_index"):
            reference.validate_reference_document(document)

        document = valid_document()
        block1 = document["source"]["blocks"]["block1"]
        block0 = document["source"]["blocks"]["block0"]
        block1["file_sha256"] = block0["file_sha256"]
        block1["manifest_sha256"] = block0["manifest_sha256"]
        with self.assertRaisesRegex(reference.ReferenceError, "distinct|block1"):
            reference.validate_reference_document(document)

    def test_rejects_capture_inventory_or_layout_growth(self):
        document = valid_document()
        del document["tensors"]["block1.img.attn.k_rope"]
        with self.assertRaisesRegex(reference.ReferenceError, "inventory"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["tensors"]["block0.img.output"]["offset_bytes"] += 4
        with self.assertRaisesRegex(reference.ReferenceError, "range|offset"):
            reference.validate_reference_document(document)

    def test_layout_and_sidecar_bounds_are_fail_closed(self):
        tensors, nbytes = _tensor_layout()
        reference.validate_tensor_layout(tensors, nbytes)
        with self.assertRaisesRegex(reference.ReferenceError, "512 MiB"):
            reference.validate_tensor_layout(tensors, reference.SIDE_CAR_LIMIT + 1)
        with self.assertRaisesRegex(reference.ReferenceError, "trailing|missing"):
            reference.validate_tensor_layout(tensors, nbytes + 4)

    def test_rejects_wrong_upstream_source_hash_inventory_and_chunk_receipt(self):
        document = valid_document()
        document["upstream"]["source_sha256"]["hymm/models/dit/nomas_cpt.py"] = "0" * 64
        with self.assertRaisesRegex(reference.ReferenceError, "source inventory"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["producer"]["attention_query_chunks_per_block"] -= 1
        with self.assertRaisesRegex(reference.ReferenceError, "receipt"):
            reference.validate_reference_document(document)

    def test_cli_requires_both_actual_weight_and_manifest_pins(self):
        with self.assertRaises(SystemExit):
            reference._build_cli().parse_args(["--output", "/tmp/full-j-reference.json"])


if __name__ == "__main__":
    unittest.main()
