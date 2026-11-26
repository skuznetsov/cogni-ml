"""Input-transport falsifiers, including BF16 halfway and signed values."""
from pathlib import Path
import hashlib
import io
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_t0_rgb_artifacts as control


class NormalizationTest(unittest.TestCase):
    def test_clamp_scale_and_signed_endpoints(self):
        actual = np.frombuffer(control.normalized_bf16(
            np.array([-1, 0, .25, .5, .75, 1, 2], dtype="<f4")), dtype="<u2")
        self.assertEqual(actual.tolist(), [0xBF80, 0xBF80, 0xBF00, 0, 0x3F00, 0x3F80, 0x3F80])

    def test_round_to_nearest_even_at_positive_and_negative_halfways(self):
        # Invert normalization for exact F32 halfways around +/-0.5.
        values = np.array([.7509765625, .7529296875, .2490234375, .2470703125], dtype="<f4")
        actual = np.frombuffer(control.normalized_bf16(values), dtype="<u2")
        self.assertEqual(actual.tolist(), [0x3F00, 0x3F02, 0xBF00, 0xBF02])

    def test_nonfinite_source_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                control.normalized_bf16(np.array([value], dtype="<f4"))

    def test_wrong_mode_layout_and_dtype_rejected_before_gather(self):
        bad = np.zeros((1, 3, 120, 1, 1), dtype="<f4")
        with self.assertRaisesRegex(ValueError, "mode"):
            next(control.expected_clip_chunks(bad, "generated"))
        for value in (bad, bad.astype("<f8")):
            with self.assertRaisesRegex(ValueError, "extent"):
                next(control.expected_clip_chunks(value, "moving"))


class FrameTransportTest(unittest.TestCase):
    def setUp(self):
        self.source = np.zeros((1, 3, 360, 1, 1), dtype="<f4")
        self.source[0, :, 118, 0, 0] = [.25, .75, 0]
        self.source[0, :, 119, 0, 0] = [.75, .25, 1]
        self.source[0, :, 120:, 0, 0] = .5

    def test_repeat119_and_channel_major_order(self):
        chunks = list(control._selected_chunks(self.source, "repeat-last"))
        self.assertEqual(chunks[:120], [bytes.fromhex("003f")] * 120)
        self.assertEqual(chunks[120:240], [bytes.fromhex("00bf")] * 120)
        self.assertEqual(chunks[240:], [bytes.fromhex("803f")] * 120)
        raw = b"".join(chunks)
        self.assertEqual(control._compare_chunks(io.BytesIO(raw), iter(chunks)),
                         hashlib.sha256(raw).hexdigest())

    def test_moving_selection_and_wrong118_wrong_t1_or_other_arm_rejected(self):
        moving = list(control._selected_chunks(self.source, "moving"))
        self.assertEqual(moving[0], bytes.fromhex("80bf"))
        self.assertEqual(moving[118], bytes.fromhex("00bf"))
        self.assertEqual(moving[119], bytes.fromhex("003f"))
        static = list(control._selected_chunks(self.source, "repeat-last"))
        wrong118 = b"".join(control.normalized_bf16(self.source[0, c, 118]) * 120
                             for c in range(3))
        wrong_t1 = b"".join(control.normalized_bf16(self.source[0, c, 120:240])
                              for c in range(3))
        swapped_channels = b"".join(static[120:240] + static[:120] + static[240:])
        for raw in (wrong118, wrong_t1, b"".join(moving), swapped_channels):
            with self.assertRaisesRegex(ValueError, "differs"):
                control._compare_chunks(io.BytesIO(raw), iter(static))

    def test_truncation_and_trailing_data_fail_closed(self):
        chunks = list(control._selected_chunks(self.source, "moving"))
        raw = b"".join(chunks)
        for bad in (raw[:-1], raw + b"\0"):
            with self.assertRaises(ValueError):
                control._compare_chunks(io.BytesIO(bad), iter(chunks))


if __name__ == "__main__":
    unittest.main()
