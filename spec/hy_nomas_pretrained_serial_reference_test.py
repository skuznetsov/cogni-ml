#!/usr/bin/env python3
"""CPU-only admission tests for the pinned two-block Hy-Nomas oracle."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_pretrained_serial_reference as reference  # noqa: E402


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
        pin = reference.BLOCK_PINS[name]
        block = {
            "block_index": index,
            "runtime_layer_name": f"double_block_{index + 1}",
            "path": f"/weights/distilled-block{index}-f32.safetensors",
            "file_bytes": pin["file_bytes"],
            "file_sha256": pin["file_sha256"],
            "manifest_path": f"/weights/distilled-block{index}-f32.safetensors.manifest.json",
            "manifest_sha256": pin["manifest_sha256"],
            "source_repository": reference.SOURCE_REPOSITORY,
            "source_revision": reference.SOURCE_CHECKPOINT_REVISION,
            "source_git_commit": reference.PINNED_REVISION,
            "checkpoint_path": reference.SOURCE_CHECKPOINT_PATH,
            "checkpoint_lfs_sha256": reference.SOURCE_CHECKPOINT_LFS_SHA256,
            "source_payload_sha256": pin["source_payload_sha256"],
            "source_header_sha256": pin.get("source_header_sha256"),
            "range_start_inclusive": pin["range_start_inclusive"],
            "range_end_inclusive": pin["range_end_inclusive"],
            "tensor_prefix": f"double_blocks.{index}.",
            "tensor_count": 32,
        }
        blocks[name] = block
    return {
        "schema": reference.SCHEMA,
        "schema_version": 1,
        "upstream_revision": reference.PINNED_REVISION,
        "geometry": dict(reference.GEOMETRY),
        "precision": dict(reference.PRECISION),
        "tolerance": {"atol": reference.ATOL, "rtol": reference.RTOL},
        "producer": {
            "device": "mps",
            "activation_readback": "one packed final F32 tensor sidecar readback",
            "attention_adapter_calls": 2,
        },
        "source": {
            "conditioning": dict(reference.CONDITIONING_SOURCE),
            "blocks": blocks,
        },
        "tensor_file": {
            "name": "serial-reference.tensors.f32le",
            "nbytes": tensor_bytes,
            "sha256": "a" * 64,
            "encoding": "f32-le",
        },
        "tensors": tensors,
    }


def valid_block_manifest(index):
    name = f"block{index}"
    pin = reference.BLOCK_PINS[name]
    expected = reference.block_reference._expected_block_keys(2048, 8192, 16)
    source = {
        "repository": reference.SOURCE_REPOSITORY,
        "revision": reference.SOURCE_CHECKPOINT_REVISION,
        "checkpoint_path": reference.SOURCE_CHECKPOINT_PATH,
        "checkpoint_size_bytes": reference.SOURCE_CHECKPOINT_BYTES,
        "checkpoint_lfs_sha256": reference.SOURCE_CHECKPOINT_LFS_SHA256,
        "source_git_commit": reference.PINNED_REVISION,
        "range_start_inclusive": pin["range_start_inclusive"],
        "range_end_inclusive": pin["range_end_inclusive"],
        "payload_sha256_observed": pin["source_payload_sha256"],
    }
    if pin["source_header_sha256"] is not None:
        source["header_sha256_observed"] = pin["source_header_sha256"]
    subset = {
        "tensor_prefix": f"double_blocks.{index}.",
        "tensor_count": 32,
        "dtype": "F32",
        "stored_bytes_preserved": True,
        "conversion_performed": False,
        "payload_bytes": 604227584,
        "tensor_names": reference._expected_source_names(index, expected),
    }
    if index == 1:
        subset["block_index"] = 1
    return {
        "format": "hy-nomas-block-subset-provenance-v1",
        "output": {
            "filename": f"distilled-block{index}-f32.safetensors",
            "sha256": pin["file_sha256"],
            "size_bytes": pin["file_bytes"],
        },
        "source": source,
        "subset": subset,
    }


class HyNomasPretrainedSerialReferenceTests(unittest.TestCase):
    def test_accepts_only_complete_pinned_two_block_envelope(self):
        reference.validate_reference_document(valid_document())

    def test_rejects_stale_conditioning_and_wrong_upstream_revision(self):
        document = valid_document()
        document["source"]["conditioning"]["reference_sha256"] = "0" * 64
        with self.assertRaisesRegex(reference.ReferenceError, "conditioning"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["upstream_revision"] = "0" * 40
        with self.assertRaisesRegex(reference.ReferenceError, "revision"):
            reference.validate_reference_document(document)

    def test_rejects_block_index_prefix_or_source_range_substitution(self):
        for key, field, value in (
            ("block1", "block_index", 0),
            ("block1", "tensor_prefix", "double_blocks.0."),
            ("block1", "range_start_inclusive", 968270320),
        ):
            with self.subTest(key=key, field=field):
                document = valid_document()
                document["source"]["blocks"][key][field] = value
                with self.assertRaises(reference.ReferenceError):
                    reference.validate_reference_document(document)

    def test_rejects_identical_block_substitution_even_with_repacked_tensor_file(self):
        document = valid_document()
        block1 = document["source"]["blocks"]["block1"]
        block0 = document["source"]["blocks"]["block0"]
        block1["file_sha256"] = block0["file_sha256"]
        block1["manifest_sha256"] = block0["manifest_sha256"]
        with self.assertRaisesRegex(reference.ReferenceError, "distinct|block1"):
            reference.validate_reference_document(document)

    def test_rejects_missing_or_malformed_checkpoint_inventory(self):
        document = valid_document()
        del document["tensors"]["block1.joint.query"]
        with self.assertRaisesRegex(reference.ReferenceError, "inventory"):
            reference.validate_reference_document(document)

        document = valid_document()
        document["tensors"]["block1.img.output"]["offset_bytes"] += 4
        with self.assertRaisesRegex(reference.ReferenceError, "range|offset"):
            reference.validate_reference_document(document)

    def test_tensor_layout_enforces_bound_and_contiguous_ranges(self):
        tensors, tensor_bytes = _tensor_layout()
        reference.validate_tensor_layout(tensors, tensor_bytes)

        malformed = copy.deepcopy(tensors)
        malformed["block1.img.output"]["offset_bytes"] += 4
        with self.assertRaisesRegex(reference.ReferenceError, "range|offset"):
            reference.validate_tensor_layout(malformed, tensor_bytes)
        with self.assertRaisesRegex(reference.ReferenceError, "64 MiB"):
            reference.validate_tensor_layout(tensors, reference.SIDE_CAR_LIMIT + 1)

    def test_block_manifest_authenticates_the_distinct_index_and_source_header(self):
        for index in (0, 1):
            with self.subTest(index=index):
                reference._validate_block_manifest(
                    valid_block_manifest(index),
                    name=f"block{index}",
                    path=Path(f"/weights/distilled-block{index}-f32.safetensors"),
                    file_sha256=reference.BLOCK_PINS[f"block{index}"]["file_sha256"],
                )

        manifest = valid_block_manifest(1)
        manifest["source"]["header_sha256_observed"] = "0" * 64
        with self.assertRaisesRegex(reference.ReferenceError, "header"):
            reference._validate_block_manifest(
                manifest,
                name="block1",
                path=Path("/weights/distilled-block1-f32.safetensors"),
                file_sha256=reference.BLOCK_PINS["block1"]["file_sha256"],
            )

        manifest = valid_block_manifest(1)
        manifest["subset"]["tensor_names"].pop()
        with self.assertRaisesRegex(reference.ReferenceError, "inventory"):
            reference._validate_block_manifest(
                manifest,
                name="block1",
                path=Path("/weights/distilled-block1-f32.safetensors"),
                file_sha256=reference.BLOCK_PINS["block1"]["file_sha256"],
            )

    def test_rotary_receipt_accepts_full_trace_and_rejects_missing_or_duplicate_call(self):
        trace = {name: object() for name in reference.BLOCK_CAPTURE_NAMES}
        reference.validate_rotary_capture(trace, calls=1, block_index=0)

        for calls in (0, 2):
            with self.subTest(calls=calls):
                with self.assertRaisesRegex(reference.ReferenceError, "rotary"):
                    reference.validate_rotary_capture(trace, calls=calls, block_index=0)

        missing = dict(trace)
        del missing["img.attn.k_rope"]
        with self.assertRaisesRegex(reference.ReferenceError, "rotary"):
            reference.validate_rotary_capture(missing, calls=1, block_index=0)

    def test_block_capture_inventory_is_exactly_forty_per_block(self):
        self.assertEqual(len(reference.BLOCK_CAPTURE_NAMES), 40)
        self.assertEqual(
            {name for name in reference.EXPECTED_TENSOR_SHAPES if name.startswith("block0.")},
            {f"block0.{name}" for name in reference.BLOCK_CAPTURE_NAMES},
        )
        self.assertEqual(
            {name for name in reference.EXPECTED_TENSOR_SHAPES if name.startswith("block1.")},
            {f"block1.{name}" for name in reference.BLOCK_CAPTURE_NAMES},
        )

    def test_cli_requires_all_source_hash_pins(self):
        parser = reference._fixture_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--output", "/tmp/serial-reference.json"])


if __name__ == "__main__":
    unittest.main()
