"""CPU-only checks for the fixed-geometry T0 key-visibility diagnostic."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SUPPORT = Path(__file__).resolve().parent / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_portrait_dit_control as portrait  # noqa: E402
import hy_nomas_reference_continuation_control as source  # noqa: E402


class _FakeMask:
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device
        self.dtype = "bool"
        self.values = [True] * shape[1]

    def __setitem__(self, key, value):
        _, span = key
        for index in range(span.start or 0, span.stop or len(self.values)):
            self.values[index] = value


class _FakeTorch:
    bool = "bool"

    @staticmethod
    def ones(shape, *, dtype, device):
        if dtype != "bool":
            raise AssertionError("visibility mask must be boolean")
        return _FakeMask(shape, device)


class CompactT0HistoryMaskTest(unittest.TestCase):
    def test_keep_all_and_keep_latest_ten_have_identical_key_geometry(self) -> None:
        all_visible = source.compact_t0_history_visibility_layout(30, portrait.VALID)
        latest_ten = source.compact_t0_history_visibility_layout(10, portrait.VALID)

        self.assertEqual(all_visible["key_rows"], 8903)
        self.assertEqual(latest_ten["key_rows"], all_visible["key_rows"])
        self.assertEqual(all_visible["hidden_t0_video_prefix_rows"], 0)
        self.assertEqual(latest_ten["hidden_t0_video_prefix_rows"], 20 * 144)
        self.assertEqual(latest_ten["visible_t0_video_rows"], 10 * 144)
        for key in (
            "t0_video_rows",
            "t0_text_start",
            "t0_text_end",
            "t1_video_start",
            "t1_video_end",
        ):
            self.assertEqual(latest_ten[key], all_visible[key])
        self.assertEqual((latest_ten["t0_text_start"], latest_ten["t0_text_end"]), (4320, 4583))
        self.assertEqual((latest_ten["t1_video_start"], latest_ten["t1_video_end"]), (4583, 8903))

    def test_materialized_mask_hides_only_old_t0_video_prefix(self) -> None:
        all_visible_layout, all_visible = portrait.build_t0_history_attention_mask(
            _FakeTorch, "mps", 30)
        latest_ten_layout, latest_ten = portrait.build_t0_history_attention_mask(
            _FakeTorch, "mps", 10)

        self.assertEqual(all_visible.shape, latest_ten.shape)
        self.assertEqual(all_visible_layout["key_rows"], latest_ten_layout["key_rows"])
        self.assertTrue(all(all_visible.values))
        self.assertEqual(latest_ten.values[:2880], [False] * 2880)
        self.assertTrue(all(latest_ten.values[2880:]))

    def test_visibility_record_names_fixed_cache_and_untrained_scope(self) -> None:
        historical = portrait.history_visibility_record(None)
        all_true = portrait.history_visibility_record(30)
        ablated = portrait.history_visibility_record(10)

        self.assertEqual(historical["hidden_t0_video_key_rows"], 0)
        self.assertFalse(historical["explicit_all_true_mask"])
        self.assertTrue(all_true["explicit_all_true_mask"])
        self.assertEqual(ablated["hidden_t0_video_key_rows"], 2880)
        self.assertEqual(ablated["retained_t0_caption_key_rows"], portrait.VALID)
        self.assertEqual(ablated["compact_attention_key_rows"], all_true["compact_attention_key_rows"])
        self.assertTrue(ablated["t0_cache_content_unchanged"])
        self.assertTrue(ablated["t0_cache_writer_unchanged"])
        self.assertTrue(ablated["t1_rope_and_query_geometry_unchanged"])
        self.assertFalse(ablated["trained_stock_contract"])

    def test_rejects_unadmitted_keep_counts_and_caption_extents(self) -> None:
        for keep_latents in (0, 9, 20, 31):
            with self.subTest(keep_latents=keep_latents):
                with self.assertRaises(source.ControlError):
                    source.compact_t0_history_visibility_layout(keep_latents, portrait.VALID)
        for valid_tokens in (0, source.DIT_TEXT_ROWS + 1):
            with self.subTest(valid_tokens=valid_tokens):
                with self.assertRaises(source.ControlError):
                    source.compact_t0_history_visibility_layout(10, valid_tokens)


class ComparisonNativeAuthenticationTest(unittest.TestCase):
    def test_optional_native_segment_requires_exact_extent_and_digest(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "native-t1.f32le"
            path.write_bytes(bytes((1, 2, 3, 4)))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(source, "LATENT_SEGMENT_SHAPE", (1, 1, 1, 1, 1)):
                self.assertIsNone(portrait._admit_comparison_latents(None, None))
                admitted = portrait._admit_comparison_latents(path, digest)
                self.assertEqual(admitted["path"], path)
                self.assertEqual(admitted["sha256"], digest)
                self.assertEqual(admitted["shape"], [1, 1, 1, 1, 1])
                with self.assertRaisesRegex(ValueError, "path and lowercase SHA-256"):
                    portrait._admit_comparison_latents(path, None)
                with self.assertRaisesRegex(ValueError, "shape or digest"):
                    portrait._admit_comparison_latents(path, "0" * 64)
                path.write_bytes(b"short")
                short_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "shape or digest"):
                    portrait._admit_comparison_latents(path, short_digest)


if __name__ == "__main__":
    unittest.main()
