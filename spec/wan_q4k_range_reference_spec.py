"""Small CPU contract checks for the independent Q4_K range reference."""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

import numpy as np

SUPPORT_PATH = Path(__file__).resolve().parents[1] / "examples" / "support"
sys.path.insert(0, str(SUPPORT_PATH))

import wan_q4k_range_reference as reference


class Q4KRangeReferenceSpec(unittest.TestCase):
    def test_hand_computed_minmax1_wire_record(self) -> None:
        source = np.zeros(reference.BLOCK_VALUES, dtype=np.float32)
        source[31] = np.float32(1.0)
        record = reference.pack_q4k_record(source, 1)
        self.assertEqual(len(record), reference.Q4_K_BLOCK_BYTES)
        self.assertEqual(record[:4], bytes.fromhex("56 14 00 00"))
        self.assertEqual(record[4:16], bytes.fromhex("ff ff ff ff 00 00 00 00 0f 0f 0f 0f"))
        self.assertEqual(record[16 + 31], 0x0F)
        self.assertEqual(sum(record[16:16 + 31]), 0)
        decoded = reference.decode_q4k_record(record)
        self.assertGreater(float(decoded[31]), 0.99)
        np.testing.assert_array_equal(decoded[np.arange(256) != 31], np.zeros(255, dtype=np.float32))

    def test_vector_writer_matches_independent_scalar_control(self) -> None:
        rng = np.random.default_rng(0x514B5246)
        values = rng.normal(size=(7, reference.BLOCK_VALUES)).astype(np.float32)
        for ranges in (1, 8):
            with self.subTest(ranges=ranges):
                vector = reference.pack_q4k_groups(values, ranges)
                scalar = b"".join(reference.pack_q4k_record(row, ranges) for row in values)
                self.assertEqual(vector, scalar)

    def test_all_zero_and_signed_zero_blocks_are_valid_canonical_zero_wire(self) -> None:
        for fill in (np.float32(0.0), np.float32(-0.0)):
            with self.subTest(fill=float(fill)):
                values = np.full(reference.BLOCK_VALUES, fill, dtype=np.float32)
                record = reference.pack_q4k_record(values, 8)
                self.assertEqual(record, bytes(reference.Q4_K_BLOCK_BYTES))
                np.testing.assert_array_equal(
                    reference.decode_q4k_record(record), np.zeros(reference.BLOCK_VALUES, dtype=np.float32)
                )
        self.assertEqual(reference.decode_q4k_record(bytes(144)).shape, (256,))

    def test_minmax1_broadcasts_one_fit_to_all_eight_scale_and_minimum_fields(self) -> None:
        source = np.linspace(-3.0, 11.0, reference.BLOCK_VALUES, dtype=np.float32)
        record = reference.pack_q4k_record(source, 1)
        _, _, scales, minima, _ = reference._decode_q4k_fields(record)
        self.assertEqual(np.unique(scales).size, 1)
        self.assertEqual(np.unique(minima).size, 1)
        self.assertEqual(int(scales[0]), 63)
        self.assertEqual(int(minima[0]), 63)

    def test_minmax8_preserves_eight_distinct_local_fits(self) -> None:
        groups = [
            np.linspace(-float(group), 2.0 * float(group), 32, dtype=np.float32)
            for group in range(1, 9)
        ]
        source = np.concatenate(groups)
        record = reference.pack_q4k_record(source, 8)
        _, _, scales, minima, _ = reference._decode_q4k_fields(record)
        self.assertEqual(np.unique(scales).size, 8)
        self.assertEqual(np.unique(minima).size, 8)

    def test_local_ranges_reduce_error_when_one_32_value_group_has_an_outlier(self) -> None:
        source = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), 8)
        source[31] = np.float32(100.0)
        decoded8 = reference.decode_q4k_record(reference.pack_q4k_record(source, 8))
        decoded1 = reference.decode_q4k_record(reference.pack_q4k_record(source, 1))
        error8 = np.mean(np.square(decoded8 - source, dtype=np.float32), dtype=np.float64)
        error1 = np.mean(np.square(decoded1 - source, dtype=np.float32), dtype=np.float64)
        self.assertLess(float(error8), float(error1))

    def test_permuting_values_within_groups_permutates_decode_without_changing_fit(self) -> None:
        source = np.linspace(-5.0, 7.0, reference.BLOCK_VALUES, dtype=np.float32)
        permutation = np.arange(reference.BLOCK_VALUES).reshape(8, 32)[:, ::-1].reshape(-1)
        original = reference.pack_q4k_record(source, 8)
        permuted = reference.pack_q4k_record(source[permutation], 8)
        original_decoded = reference.decode_q4k_record(original)
        permuted_decoded = reference.decode_q4k_record(permuted)
        np.testing.assert_array_equal(permuted_decoded, original_decoded[permutation])

    def test_f16_superscale_conversion_uses_ties_to_even(self) -> None:
        # The production path divides by 63 before its vectorized F16 cast.
        first_tie = reference._superscales(
            np.asarray([[np.float32(63.0 * 1.00048828125)]], dtype=np.float32), "test d"
        )[0]
        second_tie = reference._superscales(
            np.asarray([[np.float32(63.0 * 1.00146484375)]], dtype=np.float32), "test d"
        )[0]
        self.assertEqual(int(np.asarray([first_tie], dtype="<f2").view("<u2")[0]), 0x3C00)
        self.assertEqual(int(np.asarray([second_tie], dtype="<f2").view("<u2")[0]), 0x3C02)

    def test_six_bit_scale_and_minimum_rounding_uses_half_up(self) -> None:
        source = np.zeros(reference.BLOCK_VALUES, dtype=np.float32)
        source[7 * 32] = np.float32(0.0)
        source[7 * 32 + 1] = np.float32(945.0)  # raw scale 63 -> d is exactly 1
        source[1 * 32] = np.float32(-63.0)  # dmin is exactly 1
        source[2 * 32] = np.float32(-10.5)  # 10.5 / dmin rounds upward to 11
        record = reference.pack_q4k_record(source, 8)
        _, _, scales, minima, _ = reference._decode_q4k_fields(record)
        self.assertEqual(int(scales[0]), 0)
        self.assertEqual(int(scales[7]), 63)
        self.assertEqual(int(minima[1]), 63)
        self.assertEqual(int(minima[2]), 11)

    def test_six_bit_upper_bits_and_low_high_nibble_order_decode_from_wire(self) -> None:
        record = bytearray(reference.Q4_K_BLOCK_BYTES)
        struct.pack_into("<ee", record, 0, 1.0, 1.0)
        record[4 + 0] = 0x80  # scale group 4 high two bits: 2 << 4
        record[4 + 4] = 0x40  # minimum group 4 high two bits: 1 << 4
        record[4 + 8] = 0x53  # scale low four bits 3; minimum low four bits 5
        record[4 + 9] = 0x01  # scale group 5 is one
        record[16 + 64] = 0x02  # group 4 code 2 in low nibble
        record[16 + 64 + 1] = 0x10  # group 4 code 0, group 5 code 1
        decoded = reference.decode_q4k_record(record)
        _, _, scales, minima, codes = reference._decode_q4k_fields(record)
        self.assertEqual(int(scales[4]), 35)
        self.assertEqual(int(minima[4]), 21)
        self.assertEqual(int(codes[4, 0]), 2)
        self.assertEqual(float(decoded[128]), 49.0)
        self.assertEqual(float(decoded[129]), -21.0)
        self.assertEqual(float(decoded[5 * 32 + 1]), 1.0)

    def test_wire_perturbations_change_the_independently_decoded_element(self) -> None:
        record = bytearray(reference.Q4_K_BLOCK_BYTES)
        struct.pack_into("<ee", record, 0, 1.0, 0.0)
        record[4] = 1
        record[5] = 1
        record[16] = 0x21  # group 0 low nibble 1; group 1 high nibble 2
        decoded = reference.decode_q4k_record(record)
        self.assertEqual(float(decoded[0]), 1.0)
        self.assertEqual(float(decoded[32]), 2.0)
        record[16] = 0x31
        changed_code = reference.decode_q4k_record(record)
        self.assertEqual(float(changed_code[32]), 3.0)
        record[16] = 0x21
        record[4] = 2
        changed_scale = reference.decode_q4k_record(record)
        self.assertEqual(float(changed_scale[0]), 2.0)

    def test_scalar_and_vector_decoders_reject_short_or_nonfinite_payloads(self) -> None:
        with self.assertRaises(ValueError):
            reference.decode_q4k_record(bytes(143))
        with self.assertRaises(ValueError):
            reference.decode_q4k_records(bytes(145))
        nonfinite = bytearray(144)
        struct.pack_into("<e", nonfinite, 0, float("inf"))
        with self.assertRaises(ValueError):
            reference.decode_q4k_record(nonfinite)
        with self.assertRaises(ValueError):
            reference.decode_q4k_records(nonfinite)

    def test_serializer_rejects_nonfinite_input_and_f16_overflow_or_positive_underflow(self) -> None:
        values = np.zeros(reference.BLOCK_VALUES, dtype=np.float32)
        values[0] = np.float32(np.nan)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reference.pack_q4k_record(values, 8)

        overflow = np.zeros(reference.BLOCK_VALUES, dtype=np.float32)
        overflow[0] = np.float32(1.0e10)
        with self.assertRaisesRegex(ValueError, "overflows F16"):
            reference.pack_q4k_record(overflow, 1)

        scale_underflow = np.zeros(reference.BLOCK_VALUES, dtype=np.float32)
        scale_underflow[0] = np.float32(1.0e-6)
        with self.assertRaisesRegex(ValueError, "underflows"):
            reference.pack_q4k_record(scale_underflow, 1)

        min_underflow = np.zeros(reference.BLOCK_VALUES, dtype=np.float32)
        min_underflow[0] = np.float32(-1.0e-7)
        min_underflow[1] = np.float32(15.0)
        with self.assertRaisesRegex(ValueError, "dmin superscale underflows"):
            reference.pack_q4k_record(min_underflow, 1)


if __name__ == "__main__":
    unittest.main()
