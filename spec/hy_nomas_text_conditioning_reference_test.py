#!/usr/bin/env python3
"""CPU-only admission tests for the actual-weight Hy-Nomas text oracle."""

from contextlib import redirect_stderr
import copy
from io import StringIO
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_text_conditioning_reference as reference  # noqa: E402


def valid_document():
    return {
        "schema": reference.SCHEMA,
        "schema_version": 1,
        "upstream_revision": reference.PINNED_REVISION,
        "geometry": dict(reference.GEOMETRY),
        "precision": {
            "resident_weights": "BF16",
            "compute": "F32",
            "attention": "f32",
        },
        "qwen_source": dict(reference.QWEN_SOURCE),
        "mask": [1] * 23 + [0] * 1257,
        "timestep": 500.0,
        "compaction": {"passed": True, "comparisons": []},
    }


def valid_qwen_reference():
    return {
        "schema": "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference",
        "schema_version": 1,
        "provenance": {"activation_dtype": "bfloat16"},
        "capture": {"captured_state_count": 27},
        "sidecars": {
            "bf16": {
                "encoding": "bfloat16-le",
                "nbytes": 268_627_968,
                "sha256": reference.QWEN_SIDECAR_SHA256,
            }
        },
        "inputs": {"attention_mask": [1] * 131 + [0] * 1257},
    }


class HyNomasTextConditioningReferenceTests(unittest.TestCase):
    def test_reference_document_accepts_only_pinned_envelope(self):
        reference.validate_reference_document(valid_document())

        invalid_documents = []
        missing_schema = valid_document()
        del missing_schema["schema"]
        invalid_documents.append(missing_schema)

        wrong_revision = valid_document()
        wrong_revision["upstream_revision"] = "0" * 40
        invalid_documents.append(wrong_revision)

        wrong_geometry = valid_document()
        wrong_geometry["geometry"]["valid_tokens"] = 24
        invalid_documents.append(wrong_geometry)

        wrong_precision = valid_document()
        wrong_precision["precision"]["compute"] = "BF16"
        invalid_documents.append(wrong_precision)

        wrong_qwen = valid_document()
        wrong_qwen["qwen_source"]["state_index"] = 25
        invalid_documents.append(wrong_qwen)

        wrong_mask = valid_document()
        wrong_mask["mask"] = [True] * 23 + [0] * 1257
        invalid_documents.append(wrong_mask)

        wrong_timestep = valid_document()
        wrong_timestep["timestep"] = 501.0
        invalid_documents.append(wrong_timestep)

        failed_compaction = valid_document()
        failed_compaction["compaction"]["passed"] = False
        invalid_documents.append(failed_compaction)

        for document in invalid_documents:
            with self.subTest(document=document):
                with self.assertRaises(reference.ReferenceError):
                    reference.validate_reference_document(document)

    def test_qwen_reference_requires_exact_crop_and_sidecar_identity(self):
        mask = reference.validate_qwen_reference(
            valid_qwen_reference(),
            reference_sha256=reference.QWEN_REFERENCE_SHA256,
            sidecar_bytes=268_627_968,
            sidecar_sha256=reference.QWEN_SIDECAR_SHA256,
        )
        self.assertEqual(mask, list(range(23)))

        with self.assertRaisesRegex(reference.ReferenceError, "crop"):
            invalid = valid_qwen_reference()
            invalid["inputs"]["attention_mask"] = [1] * 132 + [0] * 1256
            reference.validate_qwen_reference(
                invalid,
                reference_sha256=reference.QWEN_REFERENCE_SHA256,
                sidecar_bytes=268_627_968,
                sidecar_sha256=reference.QWEN_SIDECAR_SHA256,
            )
        with self.assertRaisesRegex(reference.ReferenceError, "sidecar digest or byte count"):
            reference.validate_qwen_reference(
                valid_qwen_reference(),
                reference_sha256=reference.QWEN_REFERENCE_SHA256,
                sidecar_bytes=1,
                sidecar_sha256=reference.QWEN_SIDECAR_SHA256,
            )

    def test_tensor_layout_requires_contiguous_bounded_f32_ranges(self):
        tensors = {
            "second": {"dtype": "F32", "shape": [2], "offset_bytes": 4, "nbytes": 8},
            "first": {"dtype": "F32", "shape": [1], "offset_bytes": 0, "nbytes": 4},
        }
        reference.validate_tensor_layout(tensors, 12)

        malformed = copy.deepcopy(tensors)
        malformed["second"]["offset_bytes"] = "4"
        with self.assertRaisesRegex(reference.ReferenceError, "offset"):
            reference.validate_tensor_layout(malformed, 12)

        gap = copy.deepcopy(tensors)
        gap["second"]["offset_bytes"] = 8
        with self.assertRaisesRegex(reference.ReferenceError, "range|bytes"):
            reference.validate_tensor_layout(gap, 16)

        bad_shape = copy.deepcopy(tensors)
        bad_shape["first"]["shape"] = [True]
        with self.assertRaisesRegex(reference.ReferenceError, "shape"):
            reference.validate_tensor_layout(bad_shape, 12)

        with self.assertRaisesRegex(reference.ReferenceError, "outside limit"):
            reference.validate_tensor_layout(tensors, reference.SIDE_CAR_LIMIT + 1)

    def test_comparison_report_keeps_first_mismatch_and_sanitizes_nonfinite_values(self):
        comparisons = {"block0.norm1": {"shape": [2, 3]}}
        report = reference._comparison_reports(
            comparisons, [0.25, 1.0, 4.0, 1.5, 1.0]
        )[0]
        self.assertEqual(report["first_mismatch"], {
            "flat_index": 4,
            "coordinate": [1, 1],
            "actual_full": 1.5,
            "expected_compact": 1.0,
        })
        self.assertEqual(report["failures"], 1)

        nonfinite = reference._comparison_reports(
            comparisons, [float("inf"), 1.0, 0.0, float("nan"), 1.0]
        )[0]
        self.assertTrue(nonfinite["nonfinite"])
        self.assertIsNone(nonfinite["max_abs"])
        self.assertIsNone(nonfinite["first_mismatch"]["actual_full"])

    def test_all_authenticated_weight_digests_are_required_by_cli(self):
        parser = reference._fixture_parser()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--output", "/tmp/oracle.json"])
        parsed = parser.parse_args([
            "--output", "/tmp/oracle.json",
            "--text-weights-sha256", reference.TEXT_WEIGHTS_SHA256,
            "--text-manifest-sha256", reference.TEXT_MANIFEST_SHA256,
            "--block-weights-sha256", reference.BLOCK_WEIGHTS_SHA256,
            "--block-manifest-sha256", reference.BLOCK_MANIFEST_SHA256,
        ])
        self.assertEqual(parsed.attention_precision, "f32")


if __name__ == "__main__":
    unittest.main()
