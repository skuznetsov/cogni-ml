"""CPU-only contracts for the source/native T0 preblock comparator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_reference_preblock_compare as compare


class PreblockComparatorAdmissionTests(unittest.TestCase):
    def test_output_must_be_fresh_absolute_json_in_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            self.assertEqual(compare.output_path(output), output)
            output.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(compare.PreblockCompareError, "fresh"):
                compare.output_path(output)

    def test_hostile_source_schema_rejects_before_payload_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "source.json"
            raw = json.dumps({"schema": "attacker-controlled"}).encode("utf-8")
            manifest.write_bytes(raw)
            with self.assertRaisesRegex(compare.PreblockCompareError, "schema"):
                compare._validate_source_trace(
                    manifest,
                    expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_timeline_manifest_sha256="1" * 64,
                    native_validation={
                        "manifest_sha256": "2" * 64,
                        "payload_sha256": "3" * 64,
                    },
                )

    def test_each_precision_mode_requires_its_own_notes_and_consumed_dtypes(self):
        bf16 = {
            "mode": "source-bf16",
            "source_dtype_by_tensor": compare._BF16_SOURCE_TYPES,
            "semantic_notes": compare._BF16_SEMANTIC_NOTES,
        }
        f32 = {
            "mode": "source-f32-precision-control",
            "source_dtype_by_tensor": compare._F32_CONTROL_SOURCE_TYPES,
            "semantic_notes": compare._F32_CONTROL_SEMANTIC_NOTES,
            "prelude_precision_control": compare._PRELUDE_F32_CONTROL,
        }
        self.assertEqual(compare._validate_source_precision_profile(bf16), compare._BF16_SOURCE_TYPES)
        self.assertEqual(
            compare._validate_source_precision_profile(f32), compare._F32_CONTROL_SOURCE_TYPES
        )

        wrong_f32_note = dict(f32, semantic_notes=compare._BF16_SEMANTIC_NOTES)
        with self.assertRaisesRegex(compare.PreblockCompareError, "semantic notes"):
            compare._validate_source_precision_profile(wrong_f32_note)

        wrong_f32_dtype = dict(f32, source_dtype_by_tensor=compare._BF16_SOURCE_TYPES)
        with self.assertRaisesRegex(compare.PreblockCompareError, "consumed dtypes"):
            compare._validate_source_precision_profile(wrong_f32_dtype)

        wrong_bf16_note = dict(bf16, semantic_notes=compare._F32_CONTROL_SEMANTIC_NOTES)
        with self.assertRaisesRegex(compare.PreblockCompareError, "semantic notes"):
            compare._validate_source_precision_profile(wrong_bf16_note)


class MetricControlTests(unittest.TestCase):
    def test_aa_is_positive_and_deliberate_perturbation_is_detected(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        left = torch.arange(16, dtype=torch.float32)
        same = left.clone()
        perturbed = left.clone()
        perturbed[0] += 0.01

        def scalar_metrics(actual, expected):
            tensors = compare._metric_stat_tensors(
                torch, actual, expected, atol=2e-5, rtol=2e-5
            )
            return dict(zip(compare.METRIC_KEYS, torch.stack(
                [tensors[key] for key in compare.METRIC_KEYS]
            ).tolist()))

        positive = scalar_metrics(same, left)
        negative = scalar_metrics(perturbed, left)
        self.assertEqual(positive["mismatch_count"], 0.0)
        self.assertEqual(positive["finite_count"], 16.0)
        self.assertGreater(negative["mismatch_count"], 0.0)
        self.assertGreater(negative["max_abs"], 2e-5)


if __name__ == "__main__":
    unittest.main()
