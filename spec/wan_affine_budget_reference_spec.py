"""Small CPU contract checks for the independent affine reference writer."""

from __future__ import annotations

import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SUPPORT_PATH = Path(__file__).resolve().parents[1] / "examples" / "support"
sys.path.insert(0, str(SUPPORT_PATH))

import wan_affine_budget_reference as reference


class AffineBudgetReferenceSpec(unittest.TestCase):
    def test_affine_record_is_low_nibble_first_and_decodes_as_separate_f32_ops(self) -> None:
        source = np.arange(256, dtype=np.float32)
        record = reference._pack_affine_group(source)
        self.assertEqual(len(record), 136)
        scale, offset = np.frombuffer(record[128:], dtype="<f4")
        self.assertEqual(float(scale), 17.0)
        self.assertEqual(float(offset), 0.0)
        packed = np.frombuffer(record[:128], dtype=np.uint8)
        self.assertEqual(int(packed[0]), 0x00)
        self.assertEqual(int(packed[4]), 0x10)
        decoded = reference._decode_affine_group(record)
        self.assertEqual(decoded.dtype, np.float32)
        np.testing.assert_array_equal(decoded[::17], np.arange(0, 256, 17, dtype=np.float32))

    def test_constant_group_uses_zero_scale_and_exact_offset(self) -> None:
        source = np.full(256, -3.25, dtype=np.float32)
        record = reference._pack_affine_group(source)
        scale, offset = np.frombuffer(record[128:], dtype="<f4")
        self.assertEqual(float(scale), 0.0)
        self.assertEqual(float(offset), -3.25)
        np.testing.assert_array_equal(reference._decode_affine_group(record), source)

    def test_bf16_conversion_rounds_ties_to_even(self) -> None:
        values = np.asarray([1.0, 1.0 + 2**-8, 1.0 + 3 * 2**-8], dtype=np.float32)
        bits = reference.f32_to_bf16_bits(values)
        decoded = reference.bf16_bits_to_f32(bits)
        np.testing.assert_array_equal(decoded, np.asarray([1.0, 1.0, 1.0 + 2**-6], dtype=np.float32))

    def test_affine_decoder_rejects_invalid_scale_and_top_k_rejects_nan(self) -> None:
        record = bytearray(reference._pack_affine_group(np.arange(256, dtype=np.float32)))
        struct.pack_into("<f", record, 128, -1.0)
        with self.assertRaises(ValueError):
            reference._decode_affine_group(record)
        with self.assertRaises(ValueError):
            reference.top_indices(np.asarray([1.0, np.nan], dtype=np.float32), 1)

    def test_vector_affine_decoder_rejects_short_input_and_destination_overflow(self) -> None:
        with self.assertRaises(ValueError):
            reference.decode_affine_region(io.BytesIO(bytes(135)), 136, np.empty(256, dtype=np.float32))
        with self.assertRaises(ValueError):
            reference.decode_affine_region(io.BytesIO(bytes(136)), 136, np.empty(255, dtype=np.float32))

    def test_affine_encoder_rejects_nonfinite_source(self) -> None:
        source = np.zeros(256, dtype=np.float32)
        source[17] = np.inf
        with self.assertRaises(ValueError):
            reference._pack_affine_group(source)

    def test_vector_affine_wire_matches_scalar_golden_and_decodes(self) -> None:
        groups = np.stack(
            [np.arange(256, dtype=np.float32), np.full(256, -3.25, dtype=np.float32)]
        )
        wire, decoded = reference._pack_affine_groups(groups)
        scalar_wire = b"".join(reference._pack_affine_group(group) for group in groups)
        self.assertEqual(wire, scalar_wire)
        destination = np.empty(512, dtype=np.float32)
        reference.decode_affine_region(io.BytesIO(wire), len(wire), destination)
        np.testing.assert_array_equal(decoded.reshape(-1), destination)

    def test_top_k_breaks_boundary_ties_by_lower_index(self) -> None:
        scores = np.asarray([2.0, 7.0, 7.0, 4.0, 7.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(reference.top_indices(scores, 2), np.asarray([1, 2]))
        np.testing.assert_array_equal(reference.top_indices(scores, 3), np.asarray([1, 2, 4]))

    def test_candidate_payloads_exactly_match_q4k_budget(self) -> None:
        sparse = sum(
            (
                reference.AFFINE_BASE_BYTES,
                reference.SPARSE_COUNTS_BYTES,
                reference.SPARSE_OFFSETS_BYTES,
                reference.SPARSE_PATCH_BYTES,
            )
        )
        self.assertEqual(sparse, reference.Q4_K_BYTES)
        k_count, k_used, k_pad = reference.channel_capacity("k")
        n_count, n_used, n_pad = reference.channel_capacity("n")
        self.assertEqual((k_count, k_used + k_pad), (79, reference.Q4_K_BYTES))
        self.assertEqual((n_count, n_used + n_pad), (286, reference.Q4_K_BYTES))
        self.assertEqual(reference.FORMAT_DESCRIPTOR_BYTES, 16)

    def test_descriptor_is_a_real_little_endian_16_byte_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptor.bin"
            reference.write_descriptor(path, reference.FORMAT_IDS["sparse"])
            self.assertEqual(path.stat().st_size, 16)
            self.assertEqual(
                path.read_bytes(),
                struct.pack("<4I", 0x31425157, 1, reference.OUTPUT_DIM, reference.INPUT_DIM),
            )

    def test_joint_candidates_use_stable_low_index_ties_and_constant_ranges(self) -> None:
        values = np.full((1, reference.BLOCK_VALUES), np.float32(-2.5), dtype=np.float32)
        fits = reference.joint_fit_groups(values)
        self.assertEqual(
            reference.JOINT_RANGE_CANDIDATES,
            ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)),
        )
        self.assertEqual(int(fits["q1"]["candidate_id"][0]), 0)
        self.assertEqual(int(fits["q2"]["candidate_id"][0]), 0)
        np.testing.assert_array_equal(fits["q1"]["patch_positions"][0], [0])
        np.testing.assert_array_equal(fits["q2"]["patch_positions"][0], [0, 1])
        self.assertEqual(float(fits["q1"]["loss"][0]), 0.0)
        self.assertEqual(float(fits["q2"]["loss"][0]), 0.0)

    def test_joint_range_endpoint_ties_keep_first_signed_zero(self) -> None:
        values = np.zeros((1, reference.BLOCK_VALUES), dtype=np.float32)
        values[0, 0] = np.float32(-0.0)
        fits = reference.joint_fit_groups(values)
        self.assertEqual(int(fits["q1"]["lower"].view(np.uint32)[0]), 0x80000000)
        self.assertEqual(int(fits["q1"]["upper"].view(np.uint32)[0]), 0x00000000)

    def test_joint_one_patch_range_fit_repairs_high_outlier_stretched_group(self) -> None:
        values = np.concatenate(
            [np.linspace(0.0, 1.0, 255, dtype=np.float32), np.asarray([100.0], dtype=np.float32)]
        )
        values = reference.bf16_bits_to_f32(reference.f32_to_bf16_bits(values))
        fits = reference.joint_fit_groups(values.reshape(1, -1))
        q1 = fits["q1"]
        self.assertEqual(
            int(q1["candidate_id"][0]),
            int(np.argmin(q1["candidate_losses"][0])),
        )
        self.assertEqual(int(q1["patch_positions"][0, 0]), 255)
        self.assertGreater(float(q1["candidate_losses"][0, 0]), 80.0)
        self.assertLess(float(q1["loss"][0]), 0.2)

    def test_joint_two_patch_fit_handles_two_same_side_extrema(self) -> None:
        values = np.concatenate(
            [
                np.linspace(0.0, 1.0, 254, dtype=np.float32),
                np.asarray([100.0, 101.0], dtype=np.float32),
            ]
        )
        values = reference.bf16_bits_to_f32(reference.f32_to_bf16_bits(values))
        fits = reference.joint_fit_groups(values.reshape(1, -1))
        q2 = fits["q2"]
        self.assertEqual(int(q2["candidate_id"][0]), 5)
        np.testing.assert_array_equal(q2["patch_positions"][0], [255, 254])
        self.assertLess(float(q2["loss"][0]), 0.2)
        self.assertLessEqual(float(q2["loss"][0]), float(fits["q1"]["loss"][0]))

    def test_joint_two_patch_fit_trims_mixed_extrema(self) -> None:
        values = np.concatenate(
            [
                np.linspace(0.0, 1.0, 254, dtype=np.float32),
                np.asarray([-100.0, 100.0], dtype=np.float32),
            ]
        )
        values = reference.bf16_bits_to_f32(reference.f32_to_bf16_bits(values))
        fits = reference.joint_fit_groups(values.reshape(1, -1))
        q2 = fits["q2"]
        self.assertEqual(int(q2["candidate_id"][0]), 4)
        np.testing.assert_array_equal(q2["patch_positions"][0], [254, 255])
        self.assertLess(float(q2["loss"][0]), 0.2)

    def test_joint_quota_has_exact_patch_count_and_differs_from_global_top_k(self) -> None:
        q1 = np.asarray([3, 4, 2], dtype=np.uint16)
        q2 = np.asarray([[3, 7], [4, 6], [2, 5]], dtype=np.uint16)
        gains = np.asarray([9.0, 2.0, 1.0], dtype=np.float32)
        quota = reference.select_joint_quota_patches(q1, q2, gains, patch_count=4)
        self.assertEqual(int(quota["upgrade_count"]), 1)
        self.assertEqual(quota["selected_flat_indices"].size, 4)
        np.testing.assert_array_equal(quota["upgrade_groups"], [0])

        global_scores = np.zeros((3, reference.BLOCK_VALUES), dtype=np.float32)
        global_scores[0, 8:11] = [9.0, 8.0, 7.0]
        global_selected = reference.top_indices(global_scores.reshape(-1), 4)
        self.assertTrue(np.all(global_selected // reference.BLOCK_VALUES == 0))
        self.assertFalse(np.array_equal(global_selected, quota["selected_flat_indices"]))

    def test_joint_quota_randomized_selection_keeps_exact_budget_and_distinct_patches(self) -> None:
        rng = np.random.default_rng(0x51424954)
        groups = 24
        q1 = rng.integers(0, reference.BLOCK_VALUES, size=groups, dtype=np.uint16)
        q2 = np.empty((groups, 2), dtype=np.uint16)
        for group in range(groups):
            pair = rng.choice(reference.BLOCK_VALUES, size=2, replace=False)
            q2[group] = pair
        gains = rng.random(groups).astype(np.float32)
        quota = reference.select_joint_quota_patches(q1, q2, gains, patch_count=36)
        selected = quota["selected_flat_indices"]
        self.assertEqual(int(quota["upgrade_count"]), 12)
        self.assertEqual(selected.size, 36)
        self.assertEqual(np.unique(selected).size, 36)
        per_group = np.bincount(selected // reference.BLOCK_VALUES, minlength=groups)
        self.assertEqual(np.count_nonzero(per_group == 1), groups - 12)
        self.assertEqual(np.count_nonzero(per_group == 2), 12)

    def test_joint_quota_rejects_positions_outside_group_before_uint16_cast(self) -> None:
        with self.assertRaisesRegex(ValueError, "within each 256-value group"):
            reference.select_joint_quota_patches(
                np.asarray([-1], dtype=np.int64),
                np.asarray([[0, 1]], dtype=np.int64),
                np.asarray([1.0], dtype=np.float32),
                patch_count=1,
            )
        with self.assertRaisesRegex(ValueError, "within each 256-value group"):
            reference.select_joint_quota_patches(
                np.asarray([0], dtype=np.int64),
                np.asarray([[0, 256]], dtype=np.int64),
                np.asarray([1.0], dtype=np.float32),
                patch_count=1,
            )

    def test_joint_quota_rejects_duplicate_q2_positions_and_invalid_gains(self) -> None:
        with self.assertRaisesRegex(ValueError, "q2 positions must be distinct"):
            reference.select_joint_quota_patches(
                np.asarray([0], dtype=np.uint16),
                np.asarray([[0, 0]], dtype=np.uint16),
                np.asarray([1.0], dtype=np.float32),
                patch_count=1,
            )
        for gain in (-1.0, np.nan, np.inf):
            with self.subTest(gain=gain):
                with self.assertRaisesRegex(ValueError, "gain"):
                    reference.select_joint_quota_patches(
                        np.asarray([0], dtype=np.uint16),
                        np.asarray([[0, 1]], dtype=np.uint16),
                        np.asarray([gain], dtype=np.float32),
                        patch_count=1,
                    )

    def test_joint_patch_histograms_distinguish_256_groups_and_1024_records(self) -> None:
        selected = np.asarray(
            [0, 1, 2, 256 + 5, 512 + 7, 768 + 9, 768 + 10, 768 + 11, 1024 + 12],
            dtype=np.int64,
        )
        groups = reference.sparse_record_patch_histogram(selected, reference.BLOCK_VALUES)
        self.assertEqual(groups["values_per_group"], 256)
        self.assertEqual(groups["one_patch_groups"], 3)
        self.assertEqual(groups["more_than_two_patch_groups"], 2)
        self.assertEqual(groups["maximum_patches_per_group"], 3)

        records = reference.sparse_record_patch_histogram(selected)
        self.assertEqual(records["values_per_group"], 1024)
        self.assertEqual(records["one_patch_groups"], 1)
        self.assertEqual(records["more_than_two_patch_groups"], 1)
        self.assertEqual(records["maximum_patches_per_group"], 8)

    def test_joint_patch_histogram_rejects_duplicate_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reference.sparse_record_patch_histogram(np.asarray([7, 7], dtype=np.int64))

    def test_joint_activation_fit_does_not_read_heldout_rows(self) -> None:
        rows = np.ones((4, reference.BLOCK_VALUES), dtype=np.float32)
        rows[0] = np.linspace(0.1, 2.0, reference.BLOCK_VALUES, dtype=np.float32)
        rows[1] = np.linspace(2.0, 0.1, reference.BLOCK_VALUES, dtype=np.float32)
        fit_weights = reference.calibration_weights(rows, (0, 1))
        values = np.linspace(-1.0, 1.0, reference.BLOCK_VALUES, dtype=np.float32).reshape(1, -1)
        before = reference.joint_fit_groups(values, fit_weights)

        rows[2:] = np.float32(1000.0)
        same_train_weights = reference.calibration_weights(rows, (0, 1))
        after = reference.joint_fit_groups(values, same_train_weights)
        np.testing.assert_array_equal(fit_weights, same_train_weights)
        np.testing.assert_array_equal(before["q2"]["candidate_id"], after["q2"]["candidate_id"])
        np.testing.assert_array_equal(before["q2"]["patch_positions"], after["q2"]["patch_positions"])
        np.testing.assert_array_equal(before["q2"]["loss"], after["q2"]["loss"])


if __name__ == "__main__":
    unittest.main()
