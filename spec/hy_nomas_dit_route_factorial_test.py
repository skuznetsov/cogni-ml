#!/usr/bin/env python3
"""CPU-only admission and mask sensitivity tests for the direct DiT route probe."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_dit_route_factorial_reference as reference  # noqa: E402


def _stats(label: str, checkpoint: str) -> dict:
    return {
        "checkpoint": label,
        "elements": __import__("math").prod(reference.OUTPUT_SHAPES[checkpoint]),
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "rms_abs": 0.0,
        "outside_atol_rtol": 0,
        "all_finite": True,
        "passed": True,
        "atol": reference.ATOL,
        "rtol": reference.RTOL,
    }


def _valid_document() -> dict:
    tensors = {}
    cursor = 0
    for case in reference.CASE_ORDER:
        for checkpoint in reference.CHECKPOINTS:
            shape = reference.OUTPUT_SHAPES[checkpoint]
            nbytes = __import__("math").prod(shape) * 4
            name = f"{case}.{checkpoint}"
            tensors[name] = {
                "dtype": "F32",
                "source_dtype": "BF16",
                "shape": shape,
                "offset_bytes": cursor,
                "nbytes": nbytes,
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
                "checkpoint_boundary": checkpoint,
            }
            cursor += nbytes
    proof = reference.mask_proof()
    shard_records = [
        {"block": index, "path": f"/cache/block-{index}-bf16.safetensors", "bytes": 100,
         "sha256": hashlib.sha256(f"block-{index}".encode()).hexdigest(), "tensor_count": 32}
        for index in range(reference.BLOCKS)
    ]
    source_inputs = {
        "frozen_inputs": {"path": "/fixture/frozen-inputs.json", "sha256": reference.FROZEN_INPUTS_SHA256},
        "roi": {
            "manifest_path": "/fixture/source-v2-roi.json",
            "manifest_sha256": reference.ROI_MANIFEST_SHA256,
            "payload_path": "/fixture/source-v2-roi.prelude-euler.f32le",
            "payload_sha256": reference.ROI_PAYLOAD_SHA256,
            "capture_script_sha256": reference.ROI_CAPTURE_SCRIPT_SHA256,
        },
        "continuation": {
            "manifest_path": "/fixture/continuation.json",
            "manifest_sha256": reference.CONTINUATION_MANIFEST_SHA256,
            "payload_path": "/fixture/continuation.tensors.bin",
            "payload_sha256": reference.CONTINUATION_PAYLOAD_SHA256,
        },
        "accepted_parent_comparison_sha256": reference.ACCEPTED_COMPARISON_SHA256,
        "tensors": {},
    }
    for tensor_name, shape in reference.INPUT_SHAPES.items():
        source_offset, source_sha = reference.INPUT_SOURCE_RANGES[tensor_name]
        source_inputs["tensors"][tensor_name] = {
            "path": f"/fixture/{tensor_name}.f32le",
            "bytes": __import__("math").prod(shape) * 4,
            "sha256": source_sha,
            "dtype": "F32",
            "shape": shape,
            "source_range_offset": source_offset,
            "source_range_bytes": __import__("math").prod(shape) * 4,
        }
    for rope_name, rope_sha, offset in (
        ("rope_cos", reference.ROPE_COS_SHA256, 12_492_800),
        ("rope_sin", reference.ROPE_SIN_SHA256, 25_763_840),
    ):
        bank_bytes = reference.VIDEO_ROWS * reference.HEAD_DIM * 4
        source_inputs["continuation"][rope_name] = {
            "record": {"dtype": "F32", "shape": [6, reference.VIDEO_ROWS, reference.HEAD_DIM],
                       "offset": offset, "nbytes": 13_271_040, "sha256": rope_sha},
            "banks": {
                bank_name: {
                    "selected_bank": index,
                    "selected_range_offset": offset + index * bank_bytes,
                    "selected_range_bytes": bank_bytes,
                    "selected_range_sha256": hashlib.sha256(f"{rope_name}-{bank_name}".encode()).hexdigest(),
                }
                for bank_name, index in (("T0", 0), ("T1", 1))
            },
        }
    global_shard = {"path": "/cache/globals-bf16.safetensors", "bytes": 50,
                    "sha256": "c" * 64, "tensor_count": 4}
    return {
        "schema": reference.SCHEMA,
        "schema_version": reference.SCHEMA_VERSION,
        "upstream_revision": reference.PINNED_REVISION,
        "source": {
            "upstream_path": "/cached/upstream", "upstream_revision": reference.PINNED_REVISION,
            "source_hashes": {"hymm/models/dit/nomas_cpt.py": "d" * 64},
            "cache_index_path": "/cache/index.json", "cache_index_sha256": reference.CACHE_INDEX_SHA256,
            "block_shards": shard_records, "global_shard": global_shard,
            "total_weight_bytes": sum(shard["bytes"] for shard in shard_records) + global_shard["bytes"],
        },
        "source_inputs": source_inputs,
        "geometry": {
            "video_rows": reference.VIDEO_ROWS,
            "t0_valid_text_rows": reference.T0_VALID_TEXT_ROWS,
            "t1_valid_text_rows": reference.T1_VALID_TEXT_ROWS,
            "text_storage_rows": reference.TEXT_STORAGE_ROWS,
            "physical_cache_rows": reference.PHYSICAL_CACHE_ROWS,
            "history_rows": reference.HISTORY_ROWS,
            "query_rows": reference.QUERY_ROWS,
            "key_rows": reference.KEY_ROWS,
            "hidden": reference.HIDDEN,
            "heads": reference.HEADS,
            "head_dim": reference.HEAD_DIM,
            "blocks": reference.BLOCKS,
            "attention_layout": "compact",
        },
        "case_factors": copy.deepcopy(reference.FACTORS),
        "mask_proof": proof,
        "precision": reference.PRECISION_CONTRACT.copy(),
        "execution": {
            **reference.EXECUTION_FIXED,
            "parent_replay": {
                "M11": {
                    checkpoint: _stats(checkpoint, checkpoint)
                    for checkpoint in reference.CHECKPOINTS
                }
            },
            "source_aa": {
                checkpoint: _stats(checkpoint, checkpoint)
                for checkpoint in reference.CHECKPOINTS
            },
        },
        "payload": {
            "basename": "source.f32le",
            "bytes": cursor,
            "sha256": "b" * 64,
            "encoding": "f32le_bf16_expansion",
        },
        "tensors": tensors,
    }


class HyNomasDiTRouteFactorialTests(unittest.TestCase):
    def test_compact_geometry_and_route_mask_edge_counts(self):
        self.assertEqual(reference.HISTORY_ROWS, 4583)
        self.assertEqual(reference.QUERY_ROWS, 4544)
        self.assertEqual(reference.KEY_ROWS, 9127)
        queries, keys = reference.query_key_labels()
        self.assertEqual(len(queries), 4544)
        self.assertEqual(len(keys), 9127)
        self.assertEqual(keys[: reference.HISTORY_ROWS + reference.VIDEO_ROWS],
                         [0] * (reference.HISTORY_ROWS + reference.VIDEO_ROWS))
        self.assertEqual(keys[reference.HISTORY_ROWS + reference.VIDEO_ROWS :],
                         [0] * 49 + [1] * 83 + [2] * 92)

        proof = reference.mask_proof()
        self.assertEqual(proof["cases"]["M11"]["removed_edges_from_M11"],
                         {"action1": 0, "action2": 0})
        self.assertEqual(proof["cases"]["M00"]["removed_edges_from_M11"],
                         {"action1": 47061, "action2": 44160})
        self.assertEqual(proof["cases"]["M10"]["removed_edges_from_M11"],
                         {"action1": 0, "action2": 44160})
        self.assertEqual(proof["cases"]["M01"]["removed_edges_from_M11"],
                         {"action1": 47061, "action2": 0})
        self.assertEqual(proof["cases"]["M11_repeat"]["matrix_sha256"],
                         proof["cases"]["M11"]["matrix_sha256"])

    def test_mask_changes_only_selected_video_to_action_text_edges(self):
        baseline = reference.factorial_mask("M11")
        action2_off = reference.factorial_mask("M10")
        action1_off = reference.factorial_mask("M01")
        action2_start = reference.HISTORY_ROWS + reference.VIDEO_ROWS + 132
        action2_stop = reference.HISTORY_ROWS + reference.VIDEO_ROWS + 224
        action1_start = reference.HISTORY_ROWS + reference.VIDEO_ROWS + 49
        action1_stop = reference.HISTORY_ROWS + reference.VIDEO_ROWS + 132

        removed_action2 = 0
        removed_action1 = 0
        for row in range(reference.QUERY_ROWS):
            start = row * reference.KEY_ROWS
            base_row = baseline[start:start + reference.KEY_ROWS]
            other_row = action2_off[start:start + reference.KEY_ROWS]
            other_a1 = action1_off[start:start + reference.KEY_ROWS]
            if row < reference.VIDEO_ROWS:
                self.assertEqual(base_row[:action2_start], other_row[:action2_start])
                self.assertEqual(base_row[action2_stop:], other_row[action2_stop:])
                self.assertEqual(base_row[:action1_start], other_a1[:action1_start])
                self.assertEqual(base_row[action1_stop:], other_a1[action1_stop:])
                removed_action2 += sum(base_row[action2_start:action2_stop])
                removed_action1 += sum(base_row[action1_start:action1_stop])
            else:
                self.assertEqual(base_row, other_row)
                self.assertEqual(base_row, other_a1)
        self.assertEqual(removed_action2, 44_160)
        self.assertEqual(removed_action1, 47_061)

    def test_schema_accepts_complete_reference_and_rejects_factor_drift(self):
        document = _valid_document()
        reference.validate_reference_document(document)

        bad = copy.deepcopy(document)
        bad["case_factors"]["M10"]["action1_video_to_text"] = False
        with self.assertRaisesRegex(reference.ReferenceError, "factors"):
            reference.validate_reference_document(bad)

    def test_schema_rejects_missing_checkpoint_and_failed_parent_replay(self):
        document = _valid_document()
        del document["tensors"]["M01.block53.text_valid"]
        with self.assertRaisesRegex(reference.ReferenceError, "tensor inventory"):
            reference.validate_reference_document(document)

        document = _valid_document()
        document["execution"]["parent_replay"]["M11"]["velocity"]["passed"] = False
        with self.assertRaisesRegex(reference.ReferenceError, "comparison gate"):
            reference.validate_reference_document(document)

    def test_schema_rejects_failed_or_incomplete_aa_floor(self):
        document = _valid_document()
        document["execution"]["source_aa"]["block0.image"]["outside_atol_rtol"] = 1
        with self.assertRaisesRegex(reference.ReferenceError, "comparison gate"):
            reference.validate_reference_document(document)

        document = _valid_document()
        del document["execution"]["source_aa"]["block53.text_valid"]
        with self.assertRaisesRegex(reference.ReferenceError, "A/A checkpoint inventory"):
            reference.validate_reference_document(document)

    def test_comparator_detects_seeded_out_of_tolerance_mismatch(self):
        import numpy as np

        expected = np.zeros((16,), dtype=np.float32)
        equal = reference.compare_checkpoint("equal", expected, expected.copy())
        self.assertTrue(equal["passed"])
        within = expected.copy()
        within[3] = reference.ATOL * 0.5
        self.assertTrue(reference.compare_checkpoint("within", expected, within)["passed"])
        outside = expected.copy()
        outside[3] = reference.ATOL * 1.1
        result = reference.compare_checkpoint("seeded mismatch", expected, outside)
        self.assertFalse(result["passed"])
        self.assertEqual(result["outside_atol_rtol"], 1)

    def test_source_runtime_receipt_fails_closed_on_extra_readback(self):
        receipt = reference.RUNTIME_RECEIPT_FIXED.copy()
        self.assertEqual(reference.validate_runtime_receipt(receipt), receipt)
        receipt["intermediate_readbacks"] = 1
        with self.assertRaisesRegex(reference.ReferenceError, "runtime receipt"):
            reference.validate_runtime_receipt(receipt)

    def test_capture_inventory_rejects_missing_case_before_export(self):
        with self.assertRaisesRegex(reference.ReferenceError, "checkpoint inventory"):
            reference.validate_capture_map({})


if __name__ == "__main__":
    unittest.main()
