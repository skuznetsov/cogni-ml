"""CPU-only tests for terminal Hy-Nomas F32 export; no model inference."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wan = load_module("wan_s2v_export", EXAMPLES / "wan_s2v_export.py")
hy = load_module("hy_nomas_export", EXAMPLES / "hy_nomas_export.py")


class PixelRangeSpec(unittest.TestCase):
    def test_wan_default_keeps_minus_one_to_one_normalization(self):
        row = np.array([-1.0, 0.0, 1.0, 1.5], dtype=np.float32)
        frame = np.repeat(row.reshape(1, 1, 4), 3, axis=0)
        actual = wan.frame_to_rgb24(frame)
        np.testing.assert_array_equal(actual[0, :, 0],
                                      np.array([0, 127, 255, 255], dtype=np.uint8))

    def test_hy_uses_display_range_zero_to_one_multiply_then_clamp(self):
        row = np.array([-0.2, 0.0, 0.5, 1.0, 1.2], dtype=np.float32)
        frame = np.repeat(row.reshape(1, 1, 5), 3, axis=0)
        actual = wan.frame_to_rgb24(frame, "zero_to_one")
        np.testing.assert_array_equal(actual[0, :, 0],
                                      np.array([0, 0, 127, 255, 255], dtype=np.uint8))

    def test_frame_conversion_rejects_unknown_range_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            wan.frame_to_rgb24(np.zeros((3, 2, 2), dtype=np.float32), "guess")
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                wan.frame_to_rgb24(np.full((3, 2, 2), value, dtype=np.float32),
                                   "zero_to_one")


class HyTerminalExportSpec(unittest.TestCase):
    def _raw(self, root, array):
        path = root / "terminal.f32"
        np.asarray(array, dtype="<f4").tofile(path)
        return path

    def test_rejects_shape_byte_mismatch_and_latent_count_mismatch_before_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self._raw(root, np.zeros((3, 4, 16, 16), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Expected exactly"):
                hy.export_video(raw, root / "wrong.mp4", frames=3, height=16, width=16,
                                fps=24, latent_frames=1)
            with self.assertRaisesRegex(ValueError, "latent"):
                hy.export_video(raw, root / "wrong-latent.mp4", frames=4, height=16,
                                width=16, fps=24, latent_frames=2)
            self.assertFalse((root / "wrong.mp4").exists())
            self.assertFalse((root / "wrong-latent.mp4").exists())

    def test_nan_is_rejected_and_existing_output_is_not_clobbered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self._raw(root, np.full((3, 4, 16, 16), np.nan, dtype=np.float32))
            output = root / "existing.mp4"
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                hy.export_video(raw, output, frames=4, height=16, width=16, fps=24,
                                latent_frames=1)
            output.unlink()
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                hy.export_video(raw, output, frames=4, height=16, width=16, fps=24,
                                latent_frames=1)
            self.assertFalse(output.exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_small_completed_terminal_file_roundtrips_through_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # [B=1,C=3,T=4,H=16,W=16], display-range RGB.
            raw = self._raw(root, np.full((3, 4, 16, 16), 0.5, dtype=np.float32))
            output = root / "hy.mp4"
            manifest = hy.export_video(raw, output, frames=4, height=16, width=16,
                                       fps=24, latent_frames=1)
            self.assertTrue(output.is_file())
            self.assertEqual(manifest["schema"], "hy_nomas_terminal_export_v1")
            self.assertEqual(manifest["pixel_range"], "zero_to_one")
            self.assertEqual(manifest["latent_frames"], 1)
            self.assertFalse(manifest["neural_inference"])
            probe = shutil.which("ffprobe")
            if probe:
                result = subprocess.run(
                    [probe, "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                     "-of", "json", str(output)], check=True, capture_output=True,
                    text=True)
                stream = json.loads(result.stdout)["streams"][0]
                self.assertEqual((stream["width"], stream["height"]), (16, 16))
                self.assertEqual(stream["r_frame_rate"], "24/1")
                self.assertEqual(stream["nb_frames"], "4")


if __name__ == "__main__":
    unittest.main()
