#!/usr/bin/env python3
"""CPU admission and comparator sensitivity tests for the paired refiner oracle."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_refiner_transfer_reference as reference  # noqa: E402


def _input_record(name: str, digest: str) -> dict:
    return {
        "manifest_path": f"input-{name}.json",
        "manifest_sha256": ("a" if name == "A" else "b") * 64,
        "payload_path": f"input-{name}.bf16le",
        "payload_bytes": reference.INPUT_BYTES,
        "payload_sha256": digest,
        "tensor_name": "input.qwen_bf16",
        "offset_bytes": 0,
        "nbytes": reference.INPUT_BYTES,
        "source_tensor_sha256": ("c" if name == "A" else "d") * 64,
        "source_shape": list(reference.QWEN_SHAPE),
        "source_dtype": "BF16",
        "normalized_tensor_sha256": digest,
    }


def _valid_document() -> dict:
    input_a = _input_record("A", "1" * 64)
    input_b = _input_record("B", "2" * 64)
    tensors = {}
    cursor = 0
    for case in reference.CASE_ORDER:
        for checkpoint in reference.CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            shape = reference._output_shape(case, checkpoint)
            nbytes = reference._output_nbytes(shape)
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
    return {
        "schema": reference.SCHEMA,
        "schema_version": reference.SCHEMA_VERSION,
        "upstream_revision": reference.PINNED_REVISION,
        "source": {
            "upstream_path": "/cached/upstream",
            "revision": reference.PINNED_REVISION,
            "source_files": copy.deepcopy(reference.PINNED_SOURCE_SHA256),
            "text_weights": {
                "path": "/weights/text-time-f32.safetensors",
                "sha256": "3" * 64,
                "bytes": 583316840,
                "manifest_path": "/weights/text-time-f32.safetensors.manifest.json",
                "manifest_sha256": "4" * 64,
                "tensor_count": 42,
                "stored_dtype": "F32",
                "loaded_dtype": "BF16",
                "conversion": "F32_to_BF16_RNE",
            },
            "source_checkpoint": {
                "repository": "tencent/Hy-Nomas",
                "checkpoint_path": "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
                "checkpoint_lfs_sha256": "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
                "checkpoint_lfs_sha256_status": "declared by pinned HF API; whole file not downloaded or verified",
                "source_snapshot_revision": "bac719a80c8d02f91c485fca804bb8221e149b70",
            },
            "module_bindings": {
                "refiner": "hymm.models.text_encoder.token_refiner.SingleTokenRefinerPerBlock",
                "time_embedder": "hymm.models.dit.modules.embed_layers.TimestepEmbedder",
            },
        },
        "geometry": copy.deepcopy(reference.GEOMETRY),
        "timestep": 1000.0,
        "mask": copy.deepcopy(reference.MASK_RECORD),
        "action_spans": copy.deepcopy(reference.ACTION_SPANS),
        "inputs": {
            "cases": {"A0": input_a, "A1": copy.deepcopy(input_a), "B": input_b},
            "prefix_checks": {
                "validated_before_padding_normalization": True,
                "unchanged_rows": [0, 132],
                "changed_rows": [132, 224],
                "action2_changed_bytes": 109584,
                "global_action1_sha256": "6" * 64,
                "A0_A1_same_normalized_qwen_sha256": "1" * 64,
            },
        },
        "normalization": copy.deepcopy(reference.NORMALIZATION),
        "precision": copy.deepcopy(reference.PRECISION),
        "execution": {
            **copy.deepcopy(reference.EXECUTION),
            "source_precision": copy.deepcopy(reference.SOURCE_PRECISION),
            "source_attention_query_chunks": 2,
            "source_a_a_floor": {
                checkpoint: {
                    "checkpoint": f"A0/A1.{checkpoint}",
                    "elements": reference._output_nbytes(
                        reference._output_shape("A0", checkpoint)
                    ) // 4,
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "rms_abs": 0.0,
                    "outside_atol_rtol": 0,
                    "all_finite": True,
                    "passed": True,
                    "atol": reference.ATOL,
                    "rtol": reference.RTOL,
                }
                for checkpoint in reference.CHECKPOINTS
            },
        },
        "payload": {
            "basename": "source.f32le",
            "bytes": cursor,
            "sha256": "7" * 64,
            "encoding": "f32le_bf16_expansion",
        },
        "tensors": tensors,
    }


class HyNomasRefinerTransferTests(unittest.TestCase):
    def test_pair_preserves_prefix_and_normalizes_padding_after_validation(self):
        base = bytearray(reference.INPUT_BYTES)
        edited = bytearray(base)
        row_bytes = reference.HIDDEN_SIZE * 2
        edited[132 * row_bytes : 132 * row_bytes + 2] = b"\x01\x00"
        edited[224 * row_bytes :] = b"\x02" * (len(edited) - 224 * row_bytes)

        result = reference.prepare_pair(bytes(base), bytes(edited))

        self.assertEqual(result["A0"], result["A1"])
        self.assertEqual(result["A0"][:132 * row_bytes], result["B"][:132 * row_bytes])
        self.assertNotEqual(result["A0"][132 * row_bytes : 224 * row_bytes],
                            result["B"][132 * row_bytes : 224 * row_bytes])
        self.assertEqual(result["A0"][224 * row_bytes :], bytes(len(base) - 224 * row_bytes))
        self.assertEqual(result["B"][224 * row_bytes :], bytes(len(base) - 224 * row_bytes))

    def test_pair_rejects_prefix_contamination_before_normalization(self):
        base = bytearray(reference.INPUT_BYTES)
        edited = bytearray(base)
        row_bytes = reference.HIDDEN_SIZE * 2
        edited[10 * row_bytes] = 1
        with self.assertRaisesRegex(reference.ReferenceError, "unchanged prefix"):
            reference.prepare_pair(bytes(base), bytes(edited))

    def test_pair_rejects_unchanged_action2(self):
        fixture = bytes(reference.INPUT_BYTES)
        with self.assertRaisesRegex(reference.ReferenceError, "Action2 intervention"):
            reference.prepare_pair(fixture, fixture)

    def test_pair_rejects_nonfinite_bf16_and_wrong_extent(self):
        finite = bytearray(reference.INPUT_BYTES)
        nonfinite = bytearray(finite)
        nonfinite[2:4] = b"\x80\x7f"
        with self.assertRaisesRegex(reference.ReferenceError, "nonfinite"):
            reference.prepare_pair(bytes(finite), bytes(nonfinite))
        with self.assertRaisesRegex(reference.ReferenceError, "extent"):
            reference.prepare_pair(bytes(finite[:-2]), bytes(finite))

    def test_reference_document_rejects_geometry_digest_and_layout_drift(self):
        document = _valid_document()
        reference.validate_reference_document(document)

        broken = copy.deepcopy(document)
        broken["geometry"]["valid_rows"] = 223
        with self.assertRaisesRegex(reference.ReferenceError, "geometry"):
            reference.validate_reference_document(broken)

        broken = copy.deepcopy(document)
        broken["source"]["text_weights"]["sha256"] = "not-a-digest"
        with self.assertRaisesRegex(reference.ReferenceError, "digest"):
            reference.validate_reference_document(broken)

        broken = copy.deepcopy(document)
        broken["inputs"]["cases"]["B"]["payload_bytes"] -= 2
        with self.assertRaisesRegex(reference.ReferenceError, "input case"):
            reference.validate_reference_document(broken)

        broken = copy.deepcopy(document)
        broken["mask"]["valid_true_prefix"] -= 1
        with self.assertRaisesRegex(reference.ReferenceError, "mask"):
            reference.validate_reference_document(broken)

    def test_reference_document_requires_complete_passing_source_aa_floor(self):
        document = _valid_document()
        reference.validate_reference_document(document)

        broken = copy.deepcopy(document)
        broken["execution"]["source_a_a_floor"] = {}
        with self.assertRaisesRegex(reference.ReferenceError, "checkpoint inventory"):
            reference.validate_reference_document(broken)

        broken = copy.deepcopy(document)
        broken["execution"]["source_a_a_floor"]["c"]["passed"] = False
        broken["execution"]["source_a_a_floor"]["c"]["outside_atol_rtol"] = 1
        with self.assertRaisesRegex(reference.ReferenceError, "failed tolerance"):
            reference.validate_reference_document(broken)

        broken = copy.deepcopy(document)
        broken["execution"]["source_a_a_floor"]["block1.output"]["elements"] -= 1
        with self.assertRaisesRegex(reference.ReferenceError, "failed tolerance"):
            reference.validate_reference_document(broken)

        broken = copy.deepcopy(document)
        broken["execution"]["source_a_a_floor"]["masked_mean"]["rtol"] *= 2
        with self.assertRaisesRegex(reference.ReferenceError, "failed tolerance"):
            reference.validate_reference_document(broken)

    def test_case_manifest_binds_whole_sidecar_selected_crop_and_optional_generation(self):
        binding = {
            "manifest_path": "/accepted/t1-inputs.json",
            "manifest_sha256": "1" * 64,
            "payload_basename": "t1-inputs.bin",
            "payload_bytes": 232000000,
            "payload_sha256": "2" * 64,
            "tensor_name": "input.qwen_bf16",
            "tensor_dtype": "BF16",
            "tensor_shape": list(reference.QWEN_SHAPE),
            "tensor_offset": 0,
            "tensor_nbytes": reference.INPUT_BYTES,
            "tensor_sha256": "a" * 64,
        }
        a = _case_manifest(binding)
        reference._validate_case_manifest(a)
        b = _case_manifest(None)
        reference._validate_case_manifest(b)

        broken = copy.deepcopy(a)
        broken["source_qwen_range"]["selected_shape"] = [1, 1388, reference.HIDDEN_SIZE]
        with self.assertRaisesRegex(reference.ReferenceError, "range metadata"):
            reference._validate_case_manifest(broken)

        broken = copy.deepcopy(b)
        broken["source_qwen_range"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(reference.ReferenceError, "whole-sidecar"):
            reference._validate_case_manifest(broken)

    def test_comparator_reports_seeded_mismatch(self):
        import numpy as np

        rng = np.random.default_rng(20261105)
        expected = rng.standard_normal(4096, dtype=np.float32)
        self.assertTrue(reference.compare_checkpoint("synthetic", expected,
                                                     expected.copy())["passed"])
        mismatch = expected.copy()
        mismatch[1777] += np.float32(1e-3)
        result = reference.compare_checkpoint("synthetic", expected, mismatch)
        self.assertFalse(result["passed"])
        self.assertGreater(result["outside_atol_rtol"], 0)
        with self.assertRaisesRegex(reference.ReferenceError, "outside atol/rtol"):
            reference.assert_checkpoint_close("synthetic", expected, mismatch)

    def test_comparator_accepts_exact_tolerance_and_rejects_just_over(self):
        import numpy as np

        expected = np.asarray([0.0], dtype=np.float32)
        exact = np.asarray([0.25], dtype=np.float32)
        at_limit = reference.compare_checkpoint(
            "synthetic-limit", expected, exact, atol=0.25, rtol=0.0,
        )
        self.assertTrue(at_limit["passed"])
        self.assertEqual(at_limit["outside_atol_rtol"], 0)

        just_over = np.asarray([0.25001], dtype=np.float32)
        over_limit = reference.compare_checkpoint(
            "synthetic-over-limit", expected, just_over, atol=0.25, rtol=0.0,
        )
        self.assertFalse(over_limit["passed"])
        self.assertEqual(over_limit["outside_atol_rtol"], 1)

    def test_input_payload_digest_is_required(self):
        payload = b"synthetic-payload"
        good = hashlib.sha256(payload).hexdigest()
        self.assertEqual(reference.require_digest(payload, good, "test"), good)
        with self.assertRaisesRegex(reference.ReferenceError, "digest mismatch"):
            reference.require_digest(payload, "0" * 64, "test")


def _case_manifest(generation_binding):
    selected_sha = "a" * 64
    sidecar_sha = "b" * 64
    normalized_sha = "c" * 64
    document = {
        "schema": reference.INPUT_SCHEMA,
        "schema_version": 1,
        "upstream_revision": reference.PINNED_REVISION,
        "source_qwen_manifest": {
            "path": "/qwen/b-qwen.json",
            "sha256": "d" * 64,
            "schema": "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference",
            "source_revision": "eb6ba55b40ada64977147b52e50a4145ee89ac4f",
        },
        "source_qwen_payload": {
            "basename": "b-qwen.bf16le",
            "bytes": reference.QWEN_SOURCE_BYTES,
            "sha256": sidecar_sha,
            "dtype": "BF16",
            "shape": list(reference.QWEN_SOURCE_SHAPE),
            "offset_bytes": 0,
        },
        "source_qwen_range": {
            "tensor_name": "capture.hidden_states",
            "dtype": "BF16",
            "shape": list(reference.QWEN_SOURCE_SHAPE),
            "offset_bytes": 0,
            "nbytes": reference.QWEN_SOURCE_BYTES,
            "sha256": sidecar_sha,
            "state_index": reference.STATE_INDEX,
            "crop_start": reference.CROP_START,
            "crop_stop": reference.RAW_SEQUENCE_LENGTH,
            "selected_shape": list(reference.QWEN_SHAPE),
            "selected_nbytes": reference.INPUT_BYTES,
            "selected_sha256": selected_sha,
        },
        "source_qwen": {
            "name": "input.qwen_bf16",
            "dtype": "BF16",
            "shape": list(reference.QWEN_SHAPE),
            "offset": 0,
            "nbytes": reference.INPUT_BYTES,
            "sha256": selected_sha,
        },
        "origin_manifest": {"path": "/qwen/b-origin.json", "sha256": "e" * 64},
        "normalization": {
            "valid_prefix_rows": reference.VALID_ROWS,
            "pad_start": reference.VALID_ROWS,
            "pad_stop": reference.TOKEN_COUNT,
            "pad_operation": "zero_after_prefix_validation",
            "normalized_sha256": normalized_sha,
        },
        "payload": {"basename": "b.bf16le", "bytes": reference.INPUT_BYTES,
                    "sha256": normalized_sha},
        "tensors": {"input.qwen_bf16": {
            "dtype": "BF16", "shape": list(reference.QWEN_SHAPE), "offset": 0,
            "nbytes": reference.INPUT_BYTES, "sha256": normalized_sha,
        }},
    }
    if generation_binding is not None:
        document["generation_input_binding"] = generation_binding
    return document


if __name__ == "__main__":
    unittest.main()
