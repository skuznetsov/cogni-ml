"""CPU-only geometry and admission tests for the Echo decoder seam oracle."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_decoder_boundary as oracle


class DecoderBoundaryTests(unittest.TestCase):
    def _receipt(self) -> dict:
        return {
            "schema": oracle.NATIVE_SCHEMA,
            "upstream_revision": oracle.block_reference.UPSTREAM_REVISION,
            "segments": 4,
            "frames": 480,
            "clean_latent_frames": 120,
            "latent_bank_shape": list(oracle.LATENT_SHAPE),
            "width": 512,
            "height": 288,
            "fps": 24,
            "steps_per_segment": 4,
            "shift": 5,
            "cfg": 1,
            "inputs": "/tmp/continuation-inputs.json",
            "inputs_sha256": "a" * 64,
            "prompt": "authenticated prompt",
            "continuation_experimental": True,
            "pixelumm": {
                "steps": 1,
                "window_latents": 10,
                "overlap_latents": 5,
                "global_windows": 23,
            },
            "diagnostics": {
                "latent_bank": "/tmp/latent-bank.f32le",
                "latent_bank_sha256": "b" * 64,
                "pixels": "/tmp/pixels.f32le",
                "pixels_sha256": "c" * 64,
            },
        }

    def test_global_seam_geometry_covers_120_240_360(self):
        expected = {
            120: ([25, 30], [100, 120], [100, 160], [120, 140]),
            240: ([55, 60], [220, 240], [220, 280], [240, 260]),
            360: ([85, 90], [340, 360], [340, 400], [360, 380]),
        }
        for seam, (latents, pixels, noise, compare) in expected.items():
            with self.subTest(seam=seam):
                geometry = oracle.seam_geometry(seam)
                self.assertEqual(geometry["latent_starts"], latents)
                self.assertEqual(geometry["pixel_starts"], pixels)
                self.assertEqual(geometry["noise_frame_range"], noise)
                self.assertEqual(geometry["compare_frame_range"], compare)
                self.assertEqual(geometry["local_compare_frame_range"], [20, 40])

    def test_geometry_rejects_non_seam_frame(self):
        for seam in (0, 121, 480, True):
            with self.subTest(seam=seam), self.assertRaises(oracle.DecoderBoundaryError):
                oracle.seam_geometry(seam)

    def test_ncthw_slice_selects_time_axis_before_flattening(self):
        tensor = oracle.torch.arange(20).reshape(1, 2, 10, 1, 1)
        selected = oracle.slice_ncthw_time(tensor, 2, 4)
        self.assertEqual(tuple(selected.shape), (1, 2, 2, 1, 1))
        self.assertEqual(selected.flatten().tolist(), [2, 3, 12, 13])
        self.assertNotEqual(selected.flatten().tolist(), tensor.flatten()[:4].tolist())

    def test_ncthw_slice_rejects_invalid_extent(self):
        tensor = oracle.torch.zeros((1, 3, 8, 2, 2))
        for start, stop in ((-1, 2), (3, 3), (0, 9)):
            with self.subTest(start=start, stop=stop), self.assertRaises(oracle.DecoderBoundaryError):
                oracle.slice_ncthw_time(tensor, start, stop)

    def test_native_receipt_rejects_changed_input_identity(self):
        with self.assertRaisesRegex(oracle.DecoderBoundaryError, "does not authenticate"):
            oracle.validate_native_receipt_document(
                self._receipt(), expected_inputs_sha256="d" * 64
            )

    def test_native_receipt_rejects_latent_hash_tampering(self):
        receipt = self._receipt()
        receipt["diagnostics"]["latent_bank_sha256"] = "bad-digest"
        with self.assertRaisesRegex(oracle.DecoderBoundaryError, "latent_bank_sha256"):
            oracle.validate_native_receipt_document(
                receipt, expected_inputs_sha256="a" * 64
            )

    def test_saved_tensor_hash_and_extent_are_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.f32le"
            path.write_bytes(b"abcd")
            digest = hashlib.sha256(b"abcd").hexdigest()
            self.assertEqual(
                oracle.authenticate_file(
                    path, expected_sha256=digest, expected_bytes=4, what="test artifact"
                ),
                path.resolve(),
            )
            with self.assertRaisesRegex(oracle.DecoderBoundaryError, "SHA-256 mismatch"):
                oracle.authenticate_file(
                    path, expected_sha256="0" * 64, expected_bytes=4, what="test artifact"
                )
            with self.assertRaisesRegex(oracle.DecoderBoundaryError, "byte extent"):
                oracle.authenticate_file(
                    path, expected_sha256=digest, expected_bytes=8, what="test artifact"
                )

    def test_display_conversion_matches_bf16_stage_boundaries(self):
        # This value makes the BF16-rounded multiply land exactly at a tie for
        # the subsequent add. Rounding only once after the combined F32
        # expression produces the adjacent BF16 value instead.
        raw = oracle.torch.tensor([0.0039065], dtype=oracle.torch.float32).reshape(1, 1, 1, 1, 1)
        display = oracle._display_range_bf16(raw)
        staged_bf16 = display.to(dtype=oracle.torch.bfloat16).contiguous()
        combined_bf16 = (raw * 0.5 + 0.5).to(dtype=oracle.torch.bfloat16).contiguous()
        self.assertEqual(staged_bf16.reshape(-1).view(oracle.torch.uint8).tolist(), [0x00, 0x3F])
        self.assertEqual(combined_bf16.reshape(-1).view(oracle.torch.uint8).tolist(), [0x01, 0x3F])
        self.assertEqual(display.flatten().tolist(), [0.5])
        self.assertEqual(display.dtype, oracle.torch.float32)


if __name__ == "__main__":
    unittest.main()
