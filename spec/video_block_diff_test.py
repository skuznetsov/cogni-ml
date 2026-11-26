"""CPU controls for the offline, streaming video-change diagnostic."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import video_block_diff as diff


class VideoBlockDiffTests(unittest.TestCase):
    def analyze(self, frames, **kwargs):
        return diff.analyze_frames(frames, fps=24.0, **kwargs)

    def test_identical_frames_have_zero_change(self):
        frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        result = self.analyze([frame] * 8, scale=2, block=4, lags=[1, 6])
        self.assertEqual(result["frames"], 8)
        for lag, expected in [("1", 7), ("6", 2)]:
            self.assertEqual(result["lags"][lag]["pairs"], expected)
            self.assertEqual(np.max(result["lags"][lag]["mean_block_delta"]), 0)
            self.assertTrue(all(row["roi_mean_delta"]["full"] == 0
                                for row in result["lags"][lag]["series"]))

    def test_translated_patch_is_localized_to_changed_blocks(self):
        before = np.zeros((16, 24, 3), dtype=np.uint8)
        before[:8, :8] = 255
        after = np.zeros_like(before)
        after[:8, 8:16] = 255
        result = self.analyze([before, after], scale=2, block=8, lags=[1],
                              rois={"moving": (0, 0, 16, 8), "still": (16, 0, 8, 16)})
        lag = result["lags"]["1"]
        np.testing.assert_array_equal(lag["mean_block_delta"], [[1, 1, 0], [0, 0, 0]])
        self.assertEqual(lag["series"][0]["roi_mean_delta"], {"moving": 1.0, "still": 0.0})

    def test_longer_lag_detects_accumulated_small_changes(self):
        frames = (np.full((8, 8, 3), value, dtype=np.uint8) for value in range(19))
        result = self.analyze(frames, scale=4, block=8, lags=[1, 6])
        short = result["lags"]["1"]["series"][0]["mean_delta"]
        long = result["lags"]["6"]["series"][0]["mean_delta"]
        self.assertAlmostEqual(long, 6 * short)
        self.assertEqual(result["lags"]["6"]["series"][0]["time_s"], 0.25)
        for lag in [1, 6]:
            for row in result["lags"][str(lag)]["series"]:
                self.assertAlmostEqual(row["mean_delta"], lag / 255)

    def test_partial_edge_blocks_are_weighted_by_actual_pixel_area(self):
        before = np.zeros((5, 7, 3), dtype=np.uint8)
        after = before.copy()
        after[4, 6] = 255
        result = self.analyze([before, after], scale=2, block=4, lags=[1])
        row = result["lags"]["1"]["series"][0]
        self.assertAlmostEqual(row["mean_delta"], 1 / 35, places=7)
        self.assertAlmostEqual(result["lags"]["1"]["mean_block_delta"][1][1], 1 / 3, places=7)

    def test_downsampling_can_hide_changes_and_scale_one_is_a_control(self):
        before = np.zeros((4, 4, 3), dtype=np.uint8)
        before[:, ::2] = 255
        after = before[:, ::-1].copy()
        coarse = self.analyze([before, after], scale=2, block=4, lags=[1])
        fine = self.analyze([before, after], scale=1, block=4, lags=[1])
        self.assertEqual(coarse["lags"]["1"]["series"][0]["mean_delta"], 0)
        self.assertEqual(fine["lags"]["1"]["series"][0]["mean_delta"], 1)

    def test_flicker_is_change_without_semantic_motion(self):
        frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in [0, 255]]
        result = self.analyze(frames, scale=2, block=4, lags=[1])
        self.assertEqual(result["lags"]["1"]["series"][0]["mean_delta"], 1)

    def test_malformed_frames_and_regions_are_rejected(self):
        valid = np.zeros((8, 8, 3), dtype=np.uint8)
        for frames in [[], [valid], [valid, valid[:4]], [valid.astype(np.float32)] * 2]:
            with self.subTest(shapes=[f.shape for f in frames]):
                with self.assertRaises(ValueError):
                    self.analyze(frames, scale=2, block=4, lags=[1])
        for roi in [(1, 0, 4, 4), (0, 0, 12, 4), (0, 0, 0, 4)]:
            with self.assertRaises(ValueError):
                self.analyze([valid, valid], scale=2, block=4, lags=[1], rois={"bad": roi})
        with self.assertRaises(ValueError):
            self.analyze([valid] * 6, scale=2, block=4, lags=[6])

    def test_truncated_raw_frame_is_not_accepted(self):
        with self.assertRaises(ValueError):
            list(diff.read_rgb_frames(io.BytesIO(bytes(8 * 8 * 3 + 1)), 8, 8))

    def test_short_pipe_reads_preserve_frame_boundaries(self):
        class ShortReads(io.BytesIO):
            def read(self, size):
                return super().read(min(size, 7))

        source = ShortReads(bytes(8 * 8 * 3) + bytes([255]) * (8 * 8 * 3))
        frames = list(diff.read_rgb_frames(source, 8, 8))
        self.assertEqual(len(frames), 2)
        self.assertTrue(np.all(frames[0] == 0))
        self.assertTrue(np.all(frames[1] == 255))

    def test_retained_image_storage_does_not_grow_with_video_length(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        result = self.analyze((frame for _ in range(100)), scale=2, block=4, lags=[1, 6])
        self.assertEqual(result["retained_coarse_frames_peak"], 7)


if __name__ == "__main__":
    unittest.main()
