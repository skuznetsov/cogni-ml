import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_portrait_history_decoder_control as control  # noqa: E402


class HistoryPrefixTest(unittest.TestCase):
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
