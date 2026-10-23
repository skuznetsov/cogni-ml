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


if __name__ == "__main__":
    unittest.main()
