"""Reject counterfactual banks that do not preserve the original photo context."""
import hashlib
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_photo_pixelumm_control as control


class PhotoPixelControlTest(unittest.TestCase):
    def setUp(self):
        # Channel-dependent values expose an incorrect contiguous-half split.
        self.channels = [struct.pack("<f", float(c)) * (30 * 9 * 16) for c in range(32)]
        self.t0 = b"".join(self.channels)
        self.bank = b"".join(c + c for c in self.channels)
        self.sha = hashlib.sha256(self.t0).hexdigest()
        self.pin = patch.object(control, "T0_SHA", self.sha)
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def test_channel_strides_preserve_photo_and_repeat(self):
        self.assertEqual(control.segments(self.bank), (self.t0, self.t0))
        self.assertEqual(control.validate_bank(self.bank, "clean", self.sha)["t0_sha256"], self.sha)

    def test_generated_t0_substitution_is_rejected(self):
        bank = bytearray(self.bank)
        bank[:4] = struct.pack("<f", 1.0)
        with self.assertRaisesRegex(control.base.ControlError, "original photo T0"):
            control.validate_bank(bytes(bank), "step1", self.sha)

    def test_clean_control_cannot_contain_a_dit_estimate(self):
        bank = bytearray(self.bank)
        bank[30 * control.FRAME_BYTES:30 * control.FRAME_BYTES + 4] = struct.pack("<f", 1.0)
        _, t1 = control.segments(bytes(bank))
        with self.assertRaisesRegex(control.base.ControlError, "repeat original"):
            control.validate_bank(bytes(bank), "clean", hashlib.sha256(t1).hexdigest())

    def test_unrounded_and_nonfinite_values_are_rejected(self):
        for word in (0x3f800001, 0x7fc00000, 0x7f800000):
            with self.subTest(word=word):
                bank = struct.pack("<I", word) + self.bank[4:]
                with self.assertRaisesRegex(control.base.ControlError, "finite BF16"):
                    control.validate_bank(bank, "step1", self.sha)

    def test_wrong_extent_checkpoint_or_kind_is_rejected(self):
        for bank, kind, sha in ((self.bank[:-4], "clean", self.sha),
                                (self.bank, "step1", "0" * 64),
                                (self.bank, "velocity", self.sha),
                                (self.bank, "final", self.sha)):
            with self.subTest(kind=kind, size=len(bank)):
                with self.assertRaises(control.base.ControlError):
                    control.validate_bank(bank, kind, sha)

    def test_malformed_receipt_cannot_establish_bank_noise_join(self):
        # A malformed receipt cannot establish replay parity by naming a bank.
        with self.assertRaisesRegex(control.base.ControlError, "bank/noise"):
            control.validate_replay({"schema": "cogni_ml.hy_nomas_decode_latent/v1"}, "a" * 64, "b" * 64)

    def test_step_label_cannot_be_substituted_for_another_checkpoint(self):
        receipt = {"schema": "cogni_ml.hy_nomas_photo_dit_checkpoint_control/v1",
                   "passed": True, "source_capture_sha256": control.DIT_CAPTURE_SHA,
                   "atol": control.base.ATOL, "rtol": control.base.RTOL,
                   "native_x0_sha256": ["a" * 64, "b" * 64, "c" * 64, "d" * 64],
                   "artifacts": [{"name": "native-step1-paired60", "sha256": "e" * 64,
                                  "shape": [1, 32, 60, 9, 16]}]}
        control.validate_checkpoint_join(receipt, "step1", "e" * 64, "a" * 64)
        with self.assertRaisesRegex(control.base.ControlError, "named DiT step"):
            control.validate_checkpoint_join(receipt, "step2", "e" * 64, "a" * 64)
        with self.assertRaisesRegex(control.base.ControlError, "paired bank"):
            control.validate_checkpoint_join(receipt, "step1", "f" * 64, "a" * 64)


if __name__ == "__main__":
    unittest.main()
