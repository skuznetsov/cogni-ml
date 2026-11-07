import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_portrait_history_decoder_control as control  # noqa: E402


class HistoryPrefixTest(unittest.TestCase):
    def test_caption_key_label_rejects_a_video_history_intervention(self):
        args = SimpleNamespace(keep_t0_latents=10, hide_t0_caption_keys=True)
        with self.assertRaisesRegex(control.base.ControlError, "caption.*30"):
            control.preflight(args)

    def test_current_t1_text_kv_metadata_requires_caption_hidden_keep_thirty(self):
        metadata = control.history_probe_metadata(30, True, True)
        self.assertEqual(metadata, {
            "keep_t0_latents": 30,
            "t0_hide_caption_keys": True,
            "t1_current_text_kv": True,
            "all_t0_video_keys_hidden": False,
            "all_t0_keys_hidden": False,
        })
        with self.assertRaisesRegex(control.base.ControlError, "requires hidden T0 caption keys"):
            control.history_probe_metadata(30, False, True)
        with self.assertRaisesRegex(control.base.ControlError, "requires all 30 T0 video latents"):
            control.history_probe_metadata(10, True, True)

    def test_all_t0_hidden_decoder_metadata_requires_caption_hidden_text_rescue(self):
        metadata = control.history_probe_metadata(0, True, True)
        self.assertEqual(metadata, {
            "keep_t0_latents": 0,
            "t0_hide_caption_keys": True,
            "t1_current_text_kv": True,
            "all_t0_video_keys_hidden": True,
            "all_t0_keys_hidden": True,
        })
        for hide_caption, current_text in ((False, False), (True, False), (False, True)):
            with self.subTest(hide_caption=hide_caption, current_text=current_text):
                with self.assertRaises(control.base.ControlError):
                    control.history_probe_metadata(0, hide_caption, current_text)

    @staticmethod
    def bank(frames):
        return b"".join(bytes([channel + frame % 17]) * (9 * 16 * 4)
                        for channel in range(32) for frame in range(frames))

    def test_t1_may_change_but_clean_t0_must_not(self):
        bank = self.bank(180)
        prefix = bytearray(control.channel_prefix(bank, frames=180, selected=60))
        prefix[30 * 9 * 16 * 4] ^= 1
        control.validate_arm_prefix(bytes(prefix), bank)
        prefix[0] ^= 1
        with self.assertRaisesRegex(control.base.ControlError, "clean T0"):
            control.validate_arm_prefix(bytes(prefix), bank)

    def test_rejects_wrong_extent(self):
        with self.assertRaisesRegex(control.base.ControlError, "extent"):
            control.validate_arm_prefix(b"short", self.bank(180))
        with self.assertRaisesRegex(control.base.ControlError, "geometry"):
            control.channel_prefix(b"short", frames=60, selected=30)

    def test_rejects_changed_decoder_noise(self):
        b = control.base
        receipt = {
            "pixel_noise_manifest_sha256": b.INPUT_MANIFEST_SHA256,
            "pixel_noise_record_sha256": b.NOISE_SHA256,
            "continuation_manifest_sha256": b.INPUT_MANIFEST_SHA256,
            "decoder_rope_cos_record_sha256": b.ROPE_COS_SHA256,
            "decoder_rope_sin_record_sha256": b.ROPE_SIN_SHA256,
            "decoder_weight_cache_sha256": b.CACHE_MANIFEST_SHA256,
            "stem_weights_sha256": b.STEM_SHA256,
            "head_weights_sha256": b.HEAD_SHA256,
        }
        control.validate_native_noise(receipt)
        receipt["pixel_noise_record_sha256"] = "0" * 64
        with self.assertRaisesRegex(control.base.ControlError, "noise, RoPE or weight"):
            control.validate_native_noise(receipt)


if __name__ == "__main__":
    unittest.main()
