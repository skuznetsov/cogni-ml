"""CPU-only contract checks for the independent Q4_K plus F16 delta reference."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SUPPORT_PATH = Path(__file__).resolve().parents[1] / "examples" / "support"
sys.path.insert(0, str(SUPPORT_PATH))

import wan_q4k_delta_reference as reference


class Q4KDeltaReferenceSpec(unittest.TestCase):
    def test_top_patches_are_ranked_descending_with_lower_index_ties(self) -> None:
        scores = np.zeros(256, dtype=np.float32)
        scores[[3, 1, 2, 255]] = [9.0, 7.0, 7.0, 7.0]
        np.testing.assert_array_equal(reference.select_patch_positions(scores, 3), [3, 1, 2])

    def test_zero_and_full_quotas_have_deterministic_positions(self) -> None:
        scores = np.zeros(256, dtype=np.float32)
        self.assertEqual(reference.select_patch_positions(scores, 0).size, 0)
        np.testing.assert_array_equal(reference.select_patch_positions(scores, 256), np.arange(256))

    def test_sidecar_uses_rank_order_u8_position_then_little_endian_f16_delta(self) -> None:
        base = np.zeros(256, dtype=np.float32)
        original = np.zeros(256, dtype=np.float32)
        original[255] = np.float32(1.00146484375)  # F16 midpoint rounds to even 1.001953125.
        original[7] = np.float32(-1.00048828125)  # F16 midpoint rounds to even -1.0.
        wire = reference.pack_delta_block_scalar(original, base, 2)
        self.assertEqual(len(wire), 2 * reference.DELTA_RECORD_BYTES)
        self.assertEqual(wire, bytes([255]) + struct.pack("<e", 1.001953125) + bytes([7]) + struct.pack("<e", -1.0))
        np.testing.assert_array_equal(
            reference.decode_delta_block_scalar(base, wire, 2)[[7, 255]],
            np.asarray([-1.0, np.float32(1.001953125)], dtype=np.float32),
        )

    def test_delta_rounding_is_f16_ties_to_even_and_reconstruction_is_separate_f32_add(self) -> None:
        base = np.full(256, np.float32(2048.0), dtype=np.float32)
        original = base.copy()
        original[19] = np.float32(2048.0 + 2.0)
        wire = reference.pack_delta_block_scalar(original, base, 1)
        self.assertEqual(wire, bytes([19]) + struct.pack("<e", 2.0))
        expected = np.add(base[19], np.float32(np.float16(2.0)), dtype=np.float32)
        decoded = reference.decode_delta_block_scalar(base, wire, 1)
        self.assertEqual(decoded[19].view(np.uint32), expected.view(np.uint32))
        self.assertEqual(float(decoded[19]), 2050.0)

    def test_scalar_and_vector_paths_match_on_microfixture(self) -> None:
        original = np.zeros((5, 256), dtype=np.float32)
        base = np.linspace(-2.0, 3.0, 5 * 256, dtype=np.float32).reshape(5, 256)
        original[:] = base
        original[0, 255] += np.float32(4.0)
        original[1, 2] -= np.float32(7.0)
        original[2, 17] += np.float32(0.3333)
        original[4, 31] -= np.float32(0.625)

        wire, reconstructed = reference.pack_delta_blocks(original, base, 3)
        scalar_wire = b"".join(
            reference.pack_delta_block_scalar(original[row], base[row], 3)
            for row in range(original.shape[0])
        )
        self.assertEqual(wire, scalar_wire)
        scalar_reconstructed = np.stack(
            [
                reference.decode_delta_block_scalar(base[row], wire[row * 9 : (row + 1) * 9], 3)
                for row in range(original.shape[0])
            ]
        )
        np.testing.assert_array_equal(reconstructed, scalar_reconstructed)
        np.testing.assert_array_equal(reference.decode_delta_blocks(base, wire, 3), reconstructed)

    def test_zero_quota_has_empty_sidecar_and_preserves_the_base(self) -> None:
        original = np.ones((2, 256), dtype=np.float32)
        base = np.arange(512, dtype=np.float32).reshape(2, 256)
        wire, reconstructed = reference.pack_delta_blocks(original, base, 0)
        self.assertEqual(wire, b"")
        np.testing.assert_array_equal(reconstructed, base)
        np.testing.assert_array_equal(reference.decode_delta_blocks(base, wire, 0), base)

    def test_sidecar_decoder_rejects_wrong_size_duplicate_positions_and_nonfinite_delta(self) -> None:
        base = np.zeros(256, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exactly"):
            reference.decode_delta_block_scalar(base, bytes(2), 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reference.decode_delta_block_scalar(base, bytes([4]) + struct.pack("<e", 1.0) + bytes([4]) + struct.pack("<e", 2.0), 2)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reference.decode_delta_block_scalar(base, bytes([4]) + struct.pack("<e", float("inf")), 1)
        with self.assertRaisesRegex(ValueError, "multiple of"):
            reference.decode_delta_blocks(np.zeros((2, 256), dtype=np.float32), bytes(4), 1)

    def test_selection_and_packers_reject_nonfinite_and_invalid_shapes(self) -> None:
        scores = np.zeros(256, dtype=np.float32)
        scores[9] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reference.select_patch_positions(scores, 1)
        with self.assertRaises(ValueError):
            reference.select_patch_positions(np.zeros(256, dtype=np.float32), -1)
        with self.assertRaises(ValueError):
            reference.pack_delta_block_scalar(np.zeros(255, dtype=np.float32), np.zeros(255, dtype=np.float32), 1)
        bad = np.zeros(256, dtype=np.float32)
        bad[0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reference.pack_delta_block_scalar(bad, np.zeros(256, dtype=np.float32), 1)

    def test_descriptor_is_a_fixed_little_endian_16_byte_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptor.bin"
            reference.write_descriptor(path, reference.format_id_for_patches(4))
            self.assertEqual(path.stat().st_size, 16)
            self.assertEqual(
                path.read_bytes(),
                struct.pack("<4I", 0x31425157, 0x104, reference.OUTPUT_DIM, reference.INPUT_DIM),
            )


if __name__ == "__main__":
    unittest.main()
