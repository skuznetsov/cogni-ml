#!/usr/bin/env python3
"""CPU-only falsifier tests for authored soft-route bias admission and masks."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import numpy as np
import torch

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_dit_route_factorial_reference as base  # noqa: E402
import hy_nomas_dit_route_factorial_runtime as runtime  # noqa: E402
import hy_nomas_soft_route_reference as reference  # noqa: E402


def _fixture_document() -> dict:
    vectors = reference.expected_bias_vectors()
    return {
        "schema": reference.BIAS_SCHEMA,
        "geometry": {"video_rows": reference.VIDEO_ROWS, "query_rows": reference.QUERY_ROWS},
        "case_order": list(reference.CASE_ORDER),
        "reference_image": {"path": "/fixture/reference.png", "sha256": hashlib.sha256(b"image").hexdigest()},
        "maps": reference._expected_maps(),
        "biases": vectors,
        "bias_f32le_sha256": {case: reference._vector_sha256(values) for case, values in vectors.items()},
    }


class HyNomasSoftRouteTests(unittest.TestCase):
    def test_pinned_bias_manifest_authenticates_map_image_and_support(self):
        value = reference.authenticate_bias_manifest(
            reference.DEFAULT_BIAS_MANIFEST, reference.BIAS_SHA256,
        )
        support = reference.validate_bias_support(value["vectors"])
        self.assertEqual(value["sha256"], reference.BIAS_SHA256)
        self.assertEqual(support["nonzero_query_rows"], {
            "A1_low": 144, "A1_high": 144, "A2_low": 128, "A2_high": 128,
            "A1_sham": 144, "A2_sham": 128,
        })
        self.assertTrue(support["text_suffix_rows_zero"])

    def test_independent_map_reconstruction_and_shams_preserve_histogram(self):
        document = _fixture_document()
        vectors = reference.validate_bias_document(document, verify_reference_image=False)
        self.assertEqual(vectors, reference.expected_bias_vectors())
        for action in ("A1", "A2"):
            self.assertEqual(sorted(vectors[f"{action}_high"]), sorted(vectors[f"{action}_sham"]))
            self.assertEqual(sum(vectors[f"{action}_high"]), sum(vectors[f"{action}_sham"]))

    def test_manifest_rejects_geometry_maps_cases_and_vector_shape_drift(self):
        document = _fixture_document()
        document["geometry"]["query_rows"] -= 1
        with self.assertRaisesRegex(reference.SoftRouteError, "geometry"):
            reference.validate_bias_document(document, verify_reference_image=False)

        document = _fixture_document()
        document["maps"]["A2"]["radius_cells"] = [2, 2]
        with self.assertRaisesRegex(reference.SoftRouteError, "map parameters"):
            reference.validate_bias_document(document, verify_reference_image=False)

        document = _fixture_document()
        del document["biases"]["A2_low"]
        with self.assertRaisesRegex(reference.SoftRouteError, "case inventory"):
            reference.validate_bias_document(document, verify_reference_image=False)

        document = _fixture_document()
        document["biases"]["A1_low"].pop()
        with self.assertRaisesRegex(reference.SoftRouteError, "length"):
            reference.validate_bias_document(document, verify_reference_image=False)

    def test_manifest_rejects_nonfinite_non_bf16_and_wrong_map_values(self):
        action_row = 144 + 16 + 5
        document = _fixture_document()
        document["biases"]["A1_low"][action_row] = float("nan")
        with self.assertRaisesRegex(reference.SoftRouteError, "finite"):
            reference.validate_bias_document(document, verify_reference_image=False)

        document = _fixture_document()
        document["biases"]["A1_low"][action_row] = 0.1
        with self.assertRaisesRegex(reference.SoftRouteError, "BF16-expanded"):
            reference.validate_bias_document(document, verify_reference_image=False)

        document = _fixture_document()
        document["biases"]["A1_low"][action_row] = 0.0
        document["bias_f32le_sha256"]["A1_low"] = reference._vector_sha256(document["biases"]["A1_low"])
        with self.assertRaisesRegex(reference.SoftRouteError, "authored map/dose"):
            reference.validate_bias_document(document, verify_reference_image=False)

    def test_bias_support_is_video_only_and_targets_matching_action_keys(self):
        vectors = reference.expected_bias_vectors()
        proof = reference.validate_bias_support(vectors)
        qlabels, klabels = base.query_key_labels()
        self.assertEqual(proof["target_key_spans"]["A1"],
                         (base.HISTORY_ROWS + base.VIDEO_ROWS + 49, base.HISTORY_ROWS + base.VIDEO_ROWS + 132))
        self.assertEqual(proof["target_key_spans"]["A2"],
                         (base.HISTORY_ROWS + base.VIDEO_ROWS + 132, base.HISTORY_ROWS + base.VIDEO_ROWS + 224))
        self.assertEqual(
            qlabels[base.VIDEO_ROWS:],
            [0] * 49 + [1] * 83 + [2] * 92,
        )
        self.assertEqual(klabels[:base.HISTORY_ROWS + base.VIDEO_ROWS], [0] * (base.HISTORY_ROWS + base.VIDEO_ROWS))
        self.assertTrue(proof["text_suffix_rows_zero"])
        for case in reference.POSITIVE_CASES:
            action = case[:2]
            label = 1 if action == "A1" else 2
            self.assertTrue(all(qlabels[i] == label for i, v in enumerate(vectors[case]) if v != 0))

    def test_additive_chunk_preserves_mask_and_biases_only_target_action_keys(self):
        source = torch.ones((2, base.KEY_ROWS), dtype=torch.bool)
        source[0, 0] = False
        vector = np.zeros((base.QUERY_ROWS,), dtype=np.float32)
        vector[0] = 0.5
        output = runtime._soft_route_additive_chunk(torch, source, vector, "A1", 0)
        start, stop = runtime.SOFT_ACTION_KEY_SPANS["A1"]
        a2_start, a2_stop = runtime.SOFT_ACTION_KEY_SPANS["A2"]
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(torch.isneginf(output), ~source))
        self.assertTrue(torch.all(output[0, start:stop] == 0.5))
        self.assertTrue(torch.all(output[1, start:stop] == 0))
        self.assertTrue(torch.all(output[:, a2_start:a2_stop] == 0))
        self.assertTrue(torch.isneginf(output[0, 0]))

    def test_additive_chunk_rejects_mask_escape_bad_geometry_and_unrepresentable_bias(self):
        source = torch.ones((2, base.KEY_ROWS), dtype=torch.bool)
        vector = np.zeros((base.QUERY_ROWS,), dtype=np.float32)
        vector[0] = 0.5
        blocked = source.clone()
        start, stop = runtime.SOFT_ACTION_KEY_SPANS["A1"]
        blocked[0, start:stop] = False
        with self.assertRaisesRegex(RuntimeError, "escapes accepted M11"):
            runtime._soft_route_additive_chunk(torch, blocked, vector, "A1", 0)
        with self.assertRaisesRegex(RuntimeError, "geometry/dtype/device"):
            runtime._soft_route_additive_chunk(torch, torch.ones((2, 10), dtype=torch.bool), vector, "A1", 0)
        vector[0] = 0.1
        with self.assertRaisesRegex(RuntimeError, "BF16 representable"):
            runtime._soft_route_additive_chunk(torch, source, vector, "A1", 0)
        vector[0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            runtime._soft_route_additive_chunk(torch, source, vector, "A1", 0)

    def test_zero_cases_stay_boolean_and_runtime_receipt_detects_missing_arm(self):
        receipt = {
            "attention_calls": 54 * 9, "attention_query_chunks": 54 * 9 * 18,
            "t0_attention_calls": 54, "t1_attention_calls": 54 * 8,
            "explicit_terminal_synchronizations": 1, "intermediate_readbacks": 0,
            "scheduler_calls": 0, "decoder_calls": 0, "refiner_calls": 0,
            "weights_staged": 54, "arms_per_layer": 9, "one_layer_cache": True,
            "autocast": "mps BF16 full block and final head",
        }
        self.assertEqual(reference.validate_runtime_receipt(receipt), receipt)
        bad = copy.deepcopy(receipt)
        bad["t1_attention_calls"] -= 54
        with self.assertRaisesRegex(reference.SoftRouteError, "runtime receipt"):
            reference.validate_runtime_receipt(bad)

    def test_source_runtime_receipt_checks_legacy_five_and_soft_eight_arm_schedules(self):
        fixtures = (
            (runtime.CASES, 324, 5832, 270, 6, 270),
            (runtime.SOFT_ROUTE_CASES, 486, 8748, 432, 9, 432),
        )
        for cases, attention_calls, query_chunks, t1_calls, arms_per_layer, key_rows in fixtures:
            receipt = {
                "attention_calls": attention_calls,
                "attention_query_chunks": query_chunks,
                "t0_attention_calls": 54,
                "t1_attention_calls": t1_calls,
                "explicit_terminal_synchronizations": 1,
                "intermediate_readbacks": 0,
                "scheduler_calls": 0,
                "decoder_calls": 0,
                "refiner_calls": 0,
                "weights_staged": 54,
                "arms_per_layer": arms_per_layer,
                "one_layer_cache": True,
                "autocast": "mps BF16 full block and final head",
                "attention_row_inventory": {
                    "4640->4583": 54,
                    "9280->9127": key_rows,
                },
            }
            runtime.assert_runtime_receipt(receipt, cases)
            preterminal = copy.deepcopy(receipt)
            preterminal["explicit_terminal_synchronizations"] = 0
            runtime.assert_runtime_receipt(preterminal, cases, terminal_synchronizations=0)
            bad = copy.deepcopy(receipt)
            bad["attention_query_chunks"] += 18
            with self.assertRaisesRegex(RuntimeError, "receipt inventory"):
                runtime.assert_runtime_receipt(bad, cases)

    def test_exact_parent_replay_rejects_subtolerance_perturbation(self):
        expected = np.zeros((8,), dtype=np.float32)
        changed = expected.copy()
        changed[4] = np.float32(base.ATOL * 0.25)
        stats = base.compare_checkpoint("subtolerance parent perturbation", expected, changed)
        self.assertTrue(stats["passed"])
        self.assertGreater(stats["max_abs"], 0)
        with self.assertRaisesRegex(reference.SoftRouteError, "exact baseline replay"):
            reference._assert_exact_control("pinned parent", expected, changed, stats)

    def _small_replay_fixture(self, root: Path) -> tuple[dict, dict, dict, dict, dict]:
        shapes = {checkpoint: [4] for checkpoint in base.CHECKPOINTS}
        parent_payload = root / "parent.f32le"
        parent_payload.write_bytes(np.zeros((4,), dtype="<f4").tobytes())
        parent_tensors = {
            f"M11.{checkpoint}": {"offset_bytes": 0}
            for checkpoint in base.CHECKPOINTS
        }
        parent = {
            "document": {"payload": {"basename": parent_payload.name, "bytes": 16,
                                      "sha256": hashlib.sha256(parent_payload.read_bytes()).hexdigest()},
                         "tensors": parent_tensors},
            "payload_path": str(parent_payload), "path": str(root / "parent.json"),
            "sha256": reference.PARENT_SHA256,
        }
        roi_payload = root / "roi.f32le"
        roi_payload.write_bytes(parent_payload.read_bytes())
        admitted = {
            "roi_payload_path": str(roi_payload),
            "parent_checkpoints": {checkpoint: {"offset": 0} for checkpoint in base.CHECKPOINTS},
        }
        captures = {
            f"{case}.{checkpoint}": np.zeros(shapes[checkpoint], dtype=np.float32)
            for case in reference.CASE_ORDER for checkpoint in base.CHECKPOINTS
        }
        captures["zero.velocity"][0] = np.float32(0.001)
        captures["zero_repeat.velocity"] = captures["zero.velocity"].copy()
        bias_document = _fixture_document()
        bias = {
            "document": bias_document, "path": "/fixture/bias.json", "sha256": reference.BIAS_SHA256,
        }
        return parent, admitted, captures, bias, shapes

    def test_comparison_rejection_collects_all_parent_roi_aa_and_effect_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(base.OUTPUT_SHAPES, {checkpoint: [4] for checkpoint in base.CHECKPOINTS}):
                parent, admitted, captures, _, _ = self._small_replay_fixture(Path(directory))
                checked = reference.validate_capture_map(captures)
                with self.assertRaises(reference.ExactReplayRejected) as raised:
                    reference._comparison_gates(parent, admitted, checked)
        rejection = raised.exception
        self.assertEqual(set(rejection.gates["parent_replay"]["zero"]), set(base.CHECKPOINTS))
        self.assertEqual(set(rejection.gates["accepted_roi_replay"]), set(base.CHECKPOINTS))
        self.assertEqual(set(rejection.gates["source_aa"]), set(base.CHECKPOINTS))
        self.assertEqual(set(rejection.gates["intervention_effects"]), set(reference.POSITIVE_CASES))
        self.assertTrue(any(
            failure["control"] == "parent_replay" and failure["checkpoint"] == "velocity"
            for failure in rejection.failures
        ))

    def test_exact_replay_rejection_publishes_separate_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(base.OUTPUT_SHAPES, {checkpoint: [4] for checkpoint in base.CHECKPOINTS}):
                parent, admitted, captures, bias, _ = self._small_replay_fixture(root)
                checked = reference.validate_capture_map(captures)
                with self.assertRaises(reference.ExactReplayRejected) as raised:
                    reference._comparison_gates(parent, admitted, checked)
                receipt = runtime.expected_runtime_receipt(
                    runtime.SOFT_ROUTE_CASES, include_attention_row_inventory=False,
                )
                with (patch.object(base, "source_inputs_document", return_value={"fixture": "authenticated"}),
                      patch.object(base, "_validate_source"),
                      patch.object(base, "_validate_source_inputs")):
                    manifest, payload, document = reference.publish_rejected_diagnostic(
                        root / "rejected.json", parent=parent, bias=bias, source={"fixture": "source"},
                        admitted=admitted, captures=checked, receipt=receipt, rejection=raised.exception,
                        runtime_environment={"python_version": "3.12.0", "torch_version": "2.8.0",
                                             "mps_available": True},
                    )
                self.assertEqual(document["schema"], reference.DIAGNOSTIC_SCHEMA)
                self.assertNotEqual(document["schema"], reference.SCHEMA)
                self.assertEqual(document["status"], "rejected")
                self.assertEqual(document["payload"]["bytes"], payload.stat().st_size)
                self.assertEqual(len(document["tensors"]), len(reference.TENSOR_NAMES))
                self.assertEqual(json.loads(manifest.read_text())["rejection"]["failures"], raised.exception.failures)
                with (patch.object(base, "_validate_source"),
                      patch.object(base, "_validate_source_inputs")):
                    reference.validate_diagnostic_document(document)
                with self.assertRaisesRegex(reference.SoftRouteError, "exact replay rejection"):
                    reference.publish_rejected_diagnostic(
                        root / "must-not-exist.json", parent=parent, bias=bias, source={}, admitted=admitted,
                        captures=checked, receipt=receipt, rejection=reference.SoftRouteError("auth failure"),
                        runtime_environment={"python_version": "3.12.0", "torch_version": "2.8.0",
                                             "mps_available": True},
                    )
                self.assertFalse((root / "must-not-exist.json").exists())

    def test_admission_failure_does_not_publish_diagnostic_or_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error_output = io.StringIO()
            with redirect_stderr(error_output):
                status = reference.main([
                    "--bias-manifest", str(root / "missing-bias.json"),
                    "--output-manifest", str(root / "source.json"),
                    "--diagnostic-manifest", str(root / "diagnostic.json"),
                ])
            self.assertEqual(status, 2)
            self.assertIn('"status": "rejected"', error_output.getvalue())
            self.assertFalse((root / "source.json").exists())
            self.assertFalse((root / "source.tensors.f32le").exists())
            self.assertFalse((root / "diagnostic.json").exists())
            self.assertFalse((root / "diagnostic.tensors.f32le").exists())

    def test_real_comparison_output_roundtrips_flat_accepted_roi_schema(self):
        from spec.hy_nomas_dit_route_factorial_test import _valid_document

        template = _valid_document()
        input_document = template["source_inputs"]
        continuation = input_document["continuation"]
        admitted = {
            "frozen_inputs_path": input_document["frozen_inputs"]["path"],
            "frozen_inputs_sha256": input_document["frozen_inputs"]["sha256"],
            "roi_manifest_path": input_document["roi"]["manifest_path"],
            "roi_manifest_sha256": input_document["roi"]["manifest_sha256"],
            "roi_payload_path": input_document["roi"]["payload_path"],
            "roi_payload_sha256": input_document["roi"]["payload_sha256"],
            "continuation_manifest_path": continuation["manifest_path"],
            "continuation_manifest_sha256": continuation["manifest_sha256"],
            "continuation_payload_path": continuation["payload_path"],
            "continuation_payload_sha256": continuation["payload_sha256"],
            "accepted_parent_comparison_sha256": input_document["accepted_parent_comparison_sha256"],
            "inputs": input_document["tensors"],
            "rope": {"input.rope_cos": continuation["rope_cos"],
                     "input.rope_sin": continuation["rope_sin"]},
            "parent_checkpoints": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(base.OUTPUT_SHAPES, {checkpoint: [4] for checkpoint in base.CHECKPOINTS}):
                parent_bytes = bytearray()
                parent_tensors = {}
                for checkpoint in base.CHECKPOINTS:
                    offset = len(parent_bytes)
                    raw = np.zeros((4,), dtype="<f4").tobytes()
                    parent_bytes.extend(raw)
                    parent_tensors[f"M11.{checkpoint}"] = {"offset_bytes": offset}
                    admitted["parent_checkpoints"][checkpoint] = {"offset": offset}
                parent_payload = root / "parent.f32le"
                parent_payload.write_bytes(parent_bytes)
                roi_payload = root / "roi.f32le"
                roi_payload.write_bytes(parent_bytes)
                admitted["roi_payload_path"] = str(roi_payload)
                parent = {
                    "path": str(root / "parent.json"), "payload_path": str(parent_payload),
                    "sha256": reference.PARENT_SHA256,
                    "document": {"payload": {"basename": parent_payload.name, "bytes": len(parent_bytes),
                                               "sha256": hashlib.sha256(parent_bytes).hexdigest()},
                                 "tensors": parent_tensors},
                }
                captures = {
                    f"{case}.{checkpoint}": np.zeros((4,), dtype=np.float32)
                    for case in reference.CASE_ORDER for checkpoint in base.CHECKPOINTS
                }
                checked = reference.validate_capture_map(captures)
                gates = reference._comparison_gates(parent, admitted, checked)
                self.assertEqual(set(gates["accepted_roi_replay"]), set(base.CHECKPOINTS))
                self.assertTrue(all(gates["accepted_roi_replay"][checkpoint]["max_abs"] == 0
                                    for checkpoint in base.CHECKPOINTS))

                records = {}
                payload = bytearray()
                for case in reference.CASE_ORDER:
                    for checkpoint in base.CHECKPOINTS:
                        name = f"{case}.{checkpoint}"
                        raw = checked[name].astype("<f4", copy=False).tobytes()
                        records[name] = {
                            "dtype": "F32", "source_dtype": "BF16", "shape": [4],
                            "offset_bytes": len(payload), "nbytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "checkpoint_boundary": checkpoint,
                        }
                        payload.extend(raw)
                bias_document = _fixture_document()
                document = reference._render_document(
                    parent=parent,
                    bias={"document": bias_document, "path": "/fixture/bias.json",
                          "sha256": reference.BIAS_SHA256},
                    source=template["source"], admitted=admitted, gates=gates,
                    receipt=runtime.expected_runtime_receipt(
                        runtime.SOFT_ROUTE_CASES, include_attention_row_inventory=False,
                    ),
                    tensor_records=records, payload_basename="source.tensors.f32le",
                    payload_bytes=len(payload), payload_sha256=hashlib.sha256(payload).hexdigest(),
                )
                reference.validate_reference_document(document)

    def test_runtime_bias_api_accepts_six_vectors_or_all_eight_with_zero_controls(self):
        vectors = {
            case: np.asarray(values, dtype=np.float32)
            for case, values in reference.expected_bias_vectors().items()
        }
        positives = {case: values for case, values in vectors.items() if case not in reference.ZERO_CASES}
        self.assertEqual(
            set(runtime._normalize_soft_route_biases(positives, torch)),
            set(reference.POSITIVE_CASES),
        )
        self.assertEqual(
            set(runtime._normalize_soft_route_biases(vectors, torch)),
            set(reference.POSITIVE_CASES),
        )
        bad = copy.deepcopy(vectors)
        bad["zero"][0] = np.float32(0.5)
        with self.assertRaisesRegex(RuntimeError, "exact F32 zero"):
            runtime._normalize_soft_route_biases(bad, torch)
        with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
            runtime._normalize_soft_route_biases({"A1_low": vectors["A1_low"]}, torch)

    def test_seeded_numeric_mismatch_is_rejected_by_comparator(self):
        expected = np.zeros((8,), dtype=np.float32)
        self.assertTrue(base.compare_checkpoint("control", expected, expected.copy())["passed"])
        changed = expected.copy()
        changed[3] = base.ATOL * 1.1
        result = base.compare_checkpoint("seeded mismatch", expected, changed)
        self.assertFalse(result["passed"])
        self.assertEqual(result["outside_atol_rtol"], 1)


if __name__ == "__main__":
    unittest.main()
