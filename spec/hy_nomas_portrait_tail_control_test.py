from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

SUPPORT = Path(__file__).resolve().parent / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_portrait_tail_control as control  # noqa: E402


class TailGeometryTest(unittest.TestCase):
    def test_60_latent_tail_is_not_covered_but_90_latent_run_covers_tail(self):
        self.assertEqual(control.window_starts(60), tuple(range(0, 51, 5)))
        self.assertEqual(control.window_starts(90), tuple(range(0, 81, 5)))
        self.assertEqual(len(control.window_starts(60)), 11)
        self.assertEqual(len(control.window_starts(90)), 17)
        covered_60 = {frame for start in control.window_starts(60)
                      for frame in range(start, start + control.WINDOW_LATENTS)}
        covered_90 = {frame for start in control.window_starts(90)
                      for frame in range(start, start + control.WINDOW_LATENTS)}
        self.assertEqual(max(covered_60), 59)
        self.assertEqual(max(covered_90), 89)
        self.assertTrue(set(range(60, 90)).issubset(covered_90))


class CyclicExtensionTest(unittest.TestCase):
    @staticmethod
    def _fixture(channels=2, frames=12, frame_bytes=3):
        return b"".join(
            bytes(((channel * 53 + frame * 7 + offset) % 256
                   for offset in range(frame_bytes)))
            for channel in range(channels)
            for frame in range(frames)
        )

    def test_accepts_prefix_preserving_three_cycle_extension(self):
        original = self._fixture(frames=8)
        extended = control.make_cyclic_tail_bytes(
            original, channels=2, frames=8, frame_bytes=3,
            prefix_frames=8, cycle_start=4, cycle_frames=2, repeats=3,
        )
        self.assertEqual(len(extended), 2 * 14 * 3)
        control.validate_cyclic_tail_bytes(
            extended, channels=2, frames=14, frame_bytes=3,
            prefix=original, prefix_frames=8,
            cycle_start=4, cycle_frames=2, repeats=3,
        )

    def test_rejects_changed_prefix_or_noncyclic_extension(self):
        original = self._fixture(frames=8)
        extended = bytearray(control.make_cyclic_tail_bytes(
            original, channels=2, frames=8, frame_bytes=3,
            prefix_frames=8, cycle_start=4, cycle_frames=2, repeats=3,
        ))
        extended[1] ^= 1
        with self.assertRaisesRegex(control.ControlError, "prefix"):
            control.validate_cyclic_tail_bytes(
                bytes(extended), channels=2, frames=14, frame_bytes=3,
                prefix=original, prefix_frames=8,
                cycle_start=4, cycle_frames=2, repeats=3,
            )
        extended = bytearray(control.make_cyclic_tail_bytes(
            original, channels=2, frames=8, frame_bytes=3,
            prefix_frames=8, cycle_start=4, cycle_frames=2, repeats=3,
        ))
        extended[-1] ^= 1
        with self.assertRaisesRegex(control.ControlError, "cyclic extension"):
            control.validate_cyclic_tail_bytes(
                bytes(extended), channels=2, frames=14, frame_bytes=3,
                prefix=original, prefix_frames=8,
                cycle_start=4, cycle_frames=2, repeats=3,
            )

    def test_rejects_wrong_digest_and_extent(self):
        data = self._fixture(frames=8)
        with self.assertRaisesRegex(control.ControlError, "SHA-256"):
            control.validate_latent_artifact_bytes(data, "0" * 64, expected_bytes=len(data))
        with self.assertRaisesRegex(control.ControlError, "extent"):
            control.validate_latent_artifact_bytes(data, hashlib.sha256(data).hexdigest(),
                                                  expected_bytes=len(data) + 1)


class NativeReceiptTest(unittest.TestCase):
    def _receipt(self):
        return {
            "schema": control.NATIVE_SCHEMA,
            "executable_sha256": control.NATIVE_EXECUTABLE_SHA256,
            "latent_sha256": "a" * 64,
            "latent_bank_sha256": "a" * 64,
            "latent_bytes": control.LATENT90_BYTES,
            "latent_dtype": "F32 little-endian",
            "latent_shape": list(control.LATENT90_SHAPE),
            "latent_frames": 90,
            "latent_selected_frames": 90,
            "latent_prefix_start_frame": 0,
            "latent_bank_shape": list(control.LATENT90_SHAPE),
            "latent_bank_frames": 90,
            "frames": 360,
            "width": 512,
            "height": 288,
            "fps": 24,
            "output_dtype": "F32 little-endian planar RGB",
            "output": "pixels.f32le",
            "output_bytes": control.PIXELS360_BYTES,
            "output_sha256": "b" * 64,
            "output_shape": list(control.PIXELS360_SHAPE),
            "pixelumm": {"steps": 1, "window_latents": 10, "overlap_latents": 5,
                         "starts": list(control.WINDOW_STARTS)},
            "metrics": {
                "decoder_encode_allocations": 0,
                "decoder_encode_readbacks": 0,
                "decoder_preterminal_waits": 0,
                "decoder_waits": 1,
            },
        }

    def test_accepts_exact_90latent_receipt_and_rejects_bad_counters(self):
        receipt = self._receipt()
        control.validate_native90_receipt(receipt, receipt_sha256="c" * 64,
                                          expected_receipt_sha256="c" * 64,
                                          latent_sha256="a" * 64)
        for key, value in (("decoder_encode_allocations", 1),
                           ("decoder_encode_readbacks", 1),
                           ("decoder_preterminal_waits", 1),
                           ("decoder_waits", 2),
                           ("decoder_waits", True)):
            bad = self._receipt()
            bad["metrics"][key] = value
            with self.subTest(counter=key, value=value):
                with self.assertRaises(control.ControlError):
                    control.validate_native90_receipt(
                        bad, receipt_sha256="c" * 64,
                        expected_receipt_sha256="c" * 64,
                        latent_sha256="a" * 64,
                    )

    def test_rejects_receipt_digest_and_geometry_mismatches(self):
        receipt = self._receipt()
        with self.assertRaisesRegex(control.ControlError, "receipt SHA-256"):
            control.validate_native90_receipt(receipt, receipt_sha256="d" * 64,
                                              expected_receipt_sha256="c" * 64,
                                              latent_sha256="a" * 64)
        receipt = self._receipt()
        receipt["latent_frames"] = 60
        with self.assertRaisesRegex(control.ControlError, "geometry"):
            control.validate_native90_receipt(receipt, receipt_sha256="c" * 64,
                                              expected_receipt_sha256="c" * 64,
                                              latent_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
