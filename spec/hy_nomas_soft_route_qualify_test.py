#!/usr/bin/env python3
"""Synthetic falsifiers for the independent soft-route post-run checker."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np

SUPPORT = Path(__file__).resolve().parents[1] / "examples" / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_soft_route_qualify as qualify  # noqa: E402


class SoftRouteQualificationTests(unittest.TestCase):
    def test_manifest_comparison_distinguishes_boolean_from_integer(self):
        self.assertFalse(qualify._strict_json_equal({"rows": True}, {"rows": 1}))
        self.assertFalse(qualify._strict_json_equal({"dose": 1}, {"dose": 1.0}))
        self.assertTrue(qualify._strict_json_equal({"rows": [1, 2]}, {"rows": [1, 2]}))

    def test_upstream_path_binding_accepts_mount_alias_only_for_same_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "upstream"
            actual.mkdir()
            alias = root / "mounted-alias"
            alias.symlink_to(actual, target_is_directory=True)
            qualify._same_resolved_directory(str(actual), str(alias), "upstream")
            different = root / "other-upstream"
            different.mkdir()
            with self.assertRaisesRegex(qualify.QualificationError, "does not resolve"):
                qualify._same_resolved_directory(str(actual), str(different), "upstream")

    def _payload_case(self, values: tuple[float, ...] = (1.0, 2.0)):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "tiny.f32le"
        raw = struct.pack(f"<{len(values)}f", *values)
        path.write_bytes(raw)
        name = "zero.velocity"
        record = {
            "shape": [1, len(values)],
            "dtype": "F32",
            "source_dtype": "BF16",
            "offset_bytes": 0,
            "nbytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        arrays = qualify.validate_tensor_payload(
            path, {name: record}, [name], {name: (1, len(values))},
            record_fields=("shape", "dtype", "source_dtype", "offset_bytes", "nbytes", "sha256"),
            label="synthetic",
        )
        return temp, path, name, record, arrays

    def test_small_bf16_payload_validates_hash_shape_and_inventory(self):
        temp, path, name, record, arrays = self._payload_case()
        try:
            self.assertEqual(arrays[name].tolist(), [[1.0, 2.0]])
            self.assertEqual(path.stat().st_size, record["nbytes"])
        finally:
            temp.cleanup()

    def test_payload_rejects_extra_or_missing_tensor_inventory(self):
        temp, path, name, record, _ = self._payload_case()
        try:
            with self.assertRaisesRegex(qualify.QualificationError, "inventory"):
                qualify.validate_tensor_payload(
                    path, {name: record, "unexpected": record}, [name], {name: (1, 2)},
                    record_fields=tuple(record), label="synthetic",
                )
        finally:
            temp.cleanup()

    def test_payload_rejects_noncontiguous_or_boolean_shape_metadata(self):
        temp, path, name, record, _ = self._payload_case()
        try:
            bad_offset = dict(record, offset_bytes=4)
            with self.assertRaisesRegex(qualify.QualificationError, "offset"):
                qualify.validate_tensor_payload(
                    path, {name: bad_offset}, [name], {name: (1, 2)},
                    record_fields=tuple(record), label="synthetic",
                )
            bad_shape = dict(record, shape=[True, 2])
            with self.assertRaisesRegex(qualify.QualificationError, "shape"):
                qualify.validate_tensor_payload(
                    path, {name: bad_shape}, [name], {name: (1, 2)},
                    record_fields=tuple(record), label="synthetic",
                )
        finally:
            temp.cleanup()

    def test_payload_rejects_tampered_tensor_hash_and_trailing_bytes(self):
        temp, path, name, record, _ = self._payload_case()
        try:
            bad_hash = dict(record, sha256="0" * 64)
            with self.assertRaisesRegex(qualify.QualificationError, "range digest"):
                qualify.validate_tensor_payload(
                    path, {name: bad_hash}, [name], {name: (1, 2)},
                    record_fields=tuple(record), label="synthetic",
                )
            path.write_bytes(path.read_bytes() + b"junk")
            with self.assertRaisesRegex(qualify.QualificationError, "cover payload"):
                qualify.validate_tensor_payload(
                    path, {name: record}, [name], {name: (1, 2)},
                    record_fields=tuple(record), label="synthetic",
                )
        finally:
            temp.cleanup()

    def test_payload_rejects_non_bf16_expansion_and_nonfinite_values(self):
        for raw in (struct.pack("<f", 1.0001), struct.pack("<f", float("inf"))):
            with self.subTest(raw=raw):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "tiny.f32le"
                    path.write_bytes(raw)
                    name = "zero.velocity"
                    record = {
                        "shape": [1, 1], "dtype": "F32", "source_dtype": "BF16",
                        "offset_bytes": 0, "nbytes": 4, "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                    with self.assertRaises(qualify.QualificationError):
                        qualify.validate_tensor_payload(
                            path, {name: record}, [name], {name: (1, 1)},
                            record_fields=tuple(record), label="synthetic",
                        )

    def test_fixed_comparator_accepts_boundary_and_rejects_seeded_outlier(self):
        expected = np.asarray([1.0, 0.0], dtype=np.float32)
        boundary_inside = expected.copy()
        boundary_inside[0] += np.float32(3.999e-5)
        self.assertEqual(qualify.compare_arrays(expected, boundary_inside)["outside_atol_rtol"], 0)
        just_outside = expected.copy()
        just_outside[0] += np.float32(4.001e-5)
        result = qualify.compare_arrays(expected, just_outside)
        self.assertEqual(result["outside_atol_rtol"], 1)
        self.assertEqual(result["exact_bit_differences"], 1)

    def test_region_metrics_are_fixed_box_differences_and_normalized_by_zero(self):
        baseline = np.full(qualify.OUTPUT_SHAPES["velocity"], 2.0, dtype=np.float32)
        values = baseline.copy()
        values[0, :, 1:10, 0:4, 4:8] += 1.0
        values[0, :, 1:10, 0:9, 1:8] += 0.25
        values[0, :, 1:10, 2:8, 10:15] += 0.125
        result = qualify.region_metrics(values, baseline, "A1")
        self.assertEqual(result["part_proxy"]["frames"], [1, 10])
        self.assertEqual(result["part_proxy"]["spatial_cells"], 16)
        self.assertAlmostEqual(result["part_proxy"]["increment_rms"], 1.25)
        self.assertAlmostEqual(result["part_proxy"]["parity_allowance_rms"], 6.0e-5)
        self.assertAlmostEqual(result["part_proxy"]["baseline_normalized_rms"], 0.625)
        self.assertGreater(result["other_actor_roi"]["increment_rms"], 0.0)
        self.assertEqual(result["outside_actor_union"]["increment_rms"], 0.0)

    def test_zero_baseline_reports_null_normalization_not_nan_or_zero(self):
        baseline = np.zeros(qualify.OUTPUT_SHAPES["velocity"], dtype=np.float32)
        values = baseline.copy()
        values[0, 0, 1, 0, 4] = 1.0
        result = qualify.region_metrics(values, baseline, "A1")
        self.assertIsNone(result["part_proxy"]["baseline_normalized_rms"])

    def _gate_measurements(self, *, low: float, high: float, sham: float,
                           high_complement: float, sham_complement: float):
        def region(target: float, complement: float):
            return {
                "part_proxy": {"increment_rms": target, "parity_allowance_rms": 0.0},
                "actor_complement": {"increment_rms": complement},
            }

        arms = {
            "A1_low": {"A1": region(low, 1.0)},
            "A1_high": {"A1": region(high, high_complement)},
            "A1_sham": {"A1": region(sham, sham_complement)},
        }
        arms.update({
            "A2_low": {"A2": region(low, 1.0)},
            "A2_high": {"A2": region(high, high_complement)},
            "A2_sham": {"A2": region(sham, sham_complement)},
        })
        return arms

    def test_gate_requires_dose_response_target_advantage_and_localization(self):
        good = qualify.action_gate("A1", self._gate_measurements(
            low=0.2, high=0.6, sham=0.3, high_complement=0.2, sham_complement=0.5,
        ))
        self.assertTrue(good["passed"])
        self.assertTrue(all(good["checks"].values()))

        bad_localization = qualify.action_gate("A1", self._gate_measurements(
            low=0.2, high=0.6, sham=0.3, high_complement=0.8, sham_complement=0.2,
        ))
        self.assertFalse(bad_localization["passed"])
        self.assertFalse(bad_localization["checks"]["high_target_to_complement_ratio_exceeds_sham"])

    def test_gate_rejects_nonmonotone_dose_and_undefined_sham_denominator(self):
        nonmonotone = qualify.action_gate("A1", self._gate_measurements(
            low=0.7, high=0.6, sham=0.1, high_complement=0.2, sham_complement=0.1,
        ))
        self.assertFalse(nonmonotone["checks"]["dose_response_low_not_greater_than_high"])

        zero_denominator = qualify.action_gate("A1", self._gate_measurements(
            low=0.2, high=0.6, sham=0.1, high_complement=0.2, sham_complement=0.0,
        ))
        self.assertFalse(zero_denominator["checks"]["target_and_complement_denominators_defined"])
        self.assertFalse(zero_denominator["passed"])

    def test_gate_rejects_seeded_tiny_effect_and_exact_allowance_boundary(self):
        below_threshold = self._gate_measurements(
            low=1.0e-8, high=2.0e-8, sham=0.0,
            high_complement=1.0, sham_complement=2.0,
        )
        below_threshold["A1_high"]["A1"]["part_proxy"]["parity_allowance_rms"] = 3.0e-8
        tiny = qualify.action_gate("A1", below_threshold)
        self.assertTrue(tiny["checks"]["targeted_high_change_nonzero"])
        self.assertFalse(tiny["checks"]["high_target_effect_exceeds_parity_allowance"])
        self.assertFalse(tiny["passed"])

        at_threshold = self._gate_measurements(
            low=0.1, high=0.2, sham=0.05,
            high_complement=1.0, sham_complement=2.0,
        )
        at_threshold["A1_high"]["A1"]["part_proxy"]["parity_allowance_rms"] = 0.2
        boundary = qualify.action_gate("A1", at_threshold)
        self.assertFalse(boundary["checks"]["high_target_effect_exceeds_parity_allowance"])

    def test_numerical_gate_failure_propagates_per_artifact(self):
        source = self._gate_measurements(
            low=0.1, high=0.2, sham=0.05,
            high_complement=1.0, sham_complement=2.0,
        )
        native = self._gate_measurements(
            low=1.0e-8, high=2.0e-8, sham=0.0,
            high_complement=1.0, sham_complement=2.0,
        )
        native["A1_high"]["A1"]["part_proxy"]["parity_allowance_rms"] = 3.0e-8
        reports = qualify._build_action_gate_reports({"source": source, "native": native})
        self.assertTrue(reports["A1_soft_dose_localization"]["passed"])
        self.assertFalse(reports["A1_native_soft_dose_localization"]["passed"])
        self.assertEqual(
            [failure["name"] for failure in qualify._failed_numerical_gates(reports)],
            ["A1_native_soft_dose_localization"],
        )

    def test_failed_numerical_cli_writes_negative_report_and_exits_two(self):
        failed_report = {
            "qualification_status": "failed_numerical_gate",
            "qualified": False,
            "failed_gates": [{"name": "A1_native_soft_dose_localization", "failed_checks": [
                "high_target_effect_exceeds_parity_allowance",
            ]}],
            "internal_checks_passed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "negative.json"
            args = [
                "--source", "source.json", "--source-sha256", "0" * 64,
                "--native", "native.json", "--native-sha256", "1" * 64,
                "--parent", "parent.json", "--bias", "bias.json",
                "--report", str(report_path),
            ]
            with patch.object(qualify, "qualify", return_value=failed_report), \
                    redirect_stdout(io.StringIO()):
                exit_code = qualify.main(args)
            self.assertEqual(exit_code, 2)
            self.assertEqual(json.loads(report_path.read_text()), failed_report)

            report_path.unlink()
            with patch.object(qualify, "qualify", side_effect=qualify.QualificationError("parity failed")), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = qualify.main(args)
            self.assertEqual(exit_code, 2)
            self.assertFalse(report_path.exists())

    def test_gate_rejects_no_effect_even_when_all_arms_match(self):
        no_effect = qualify.action_gate("A1", self._gate_measurements(
            low=0.0, high=0.0, sham=0.0, high_complement=0.0, sham_complement=0.0,
        ))
        self.assertFalse(no_effect["checks"]["targeted_high_change_nonzero"])
        self.assertFalse(no_effect["checks"]["target_and_complement_denominators_defined"])
        self.assertFalse(no_effect["passed"])

    def test_bias_support_rejects_spatially_misplaced_row(self):
        vectors = {case: np.zeros(qualify.QUERY_ROWS, dtype=np.float32) for case in qualify.CASE_ORDER}
        a1_row = 144 + 1  # frame 1, h=0, w=1: inside Alex ROI
        a2_row = 12 * 144 + 2 * 16 + 10  # frame 12, h=2, w=10: inside Resume ROI
        for case in ("A1_low", "A1_high", "A1_sham"):
            vectors[case][a1_row] = 0.5
        for case in ("A2_low", "A2_high", "A2_sham"):
            vectors[case][a2_row] = 0.5
        qualify._validate_bias_support(vectors)
        vectors["A1_high"][a1_row] = 0.0
        vectors["A1_high"][0] = 0.5
        with self.assertRaisesRegex(qualify.QualificationError, "support"):
            qualify._validate_bias_support(vectors)


if __name__ == "__main__":
    unittest.main()
