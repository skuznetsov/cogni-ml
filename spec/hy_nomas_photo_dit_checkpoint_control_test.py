"""Admission tests require no PyTorch import or GPU allocation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_photo_dit_checkpoint_control as control


class CheckpointAdmissionTest(unittest.TestCase):
    def records(self):
        return [{"step": i+1, "sigma_f32": sigma, "dt_f32": dt,
                 **{kind: {"shape": list(control.SHAPE), "dtype": "BF16",
                           "encoding": "f32le_bf16_expansion"}
                    for kind in ("velocity_bf16", "x0_bf16", "after_euler_bf16")}}
                for i, (sigma, dt) in enumerate(zip(control.SIGMAS, control.DELTAS))]

    def test_exact_four_step_schedule_is_admitted(self):
        control.validate_steps(self.records())

    def test_shortest_f32_json_decimals_preserve_schedule_bits(self):
        records = self.records()
        records[2]["sigma_f32"] = 0.8333333
        records[1]["dt_f32"] = -0.10416669
        records[2]["dt_f32"] = -0.20833331
        control.validate_steps(records)

    def test_noisy_state_mislabel_and_schedule_changes_are_rejected(self):
        records = self.records()
        variants = []
        wrong_sigma = copy.deepcopy(records)
        wrong_sigma[1]["sigma_f32"] = 0.9
        variants.append(wrong_sigma)
        wrong_dtype = copy.deepcopy(records)
        wrong_dtype[0]["x0_bf16"]["dtype"] = "F32"
        variants.append(wrong_dtype)
        variants.extend([records[::-1], records[:1]])
        for records in variants:
            with self.subTest(records=records):
                with self.assertRaises(control.base.ControlError):
                    control.validate_steps(records)

    def test_bank_segment_uses_channel_strides(self):
        frame_bytes = 9 * 16 * 4
        bank = b"".join(bytes([c]) * (180 * frame_bytes) for c in range(32))
        expected = b"".join(bytes([c]) * (30 * frame_bytes) for c in range(32))
        self.assertEqual(control.bank_segment(bank, 30), expected)
        with self.assertRaises(control.base.ControlError):
            control.bank_segment(bank, 1)

    def test_records_cannot_substitute_for_same_chain_contract(self):
        with self.assertRaisesRegex(control.base.ControlError, "capture contract"):
            control.validate_native_capture({"reference_t1_step_captures": self.records()})


if __name__ == "__main__":
    unittest.main()
