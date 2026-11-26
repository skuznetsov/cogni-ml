"""CPU admission falsifiers for context-source decoder transport."""
from pathlib import Path
import copy
import hashlib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_t0_context_decoder_control as control


class ContextPrefixTest(unittest.TestCase):
    def setUp(self):
        self.bank = bytes(32 * 180 * 9 * 16 * 4)
        self.prefix = control.history.channel_prefix(self.bank, frames=180, selected=60)
        self.receipt = {
            "schema": "cogni_ml.hy_nomas_portrait_t0_history_probe/v1",
            "t0_context_mode": "generated", "inputs_sha256": control.base.INPUT_MANIFEST_SHA256,
            "upstream_revision": control.base.UPSTREAM_REVISION,
            "source_bf16": True, "segments": 2, "steps_per_segment": 4,
            "keep_t0_video_latents": 30, "t0_hide_caption_keys": False,
            "t1_current_text_kv": False,
            "output": {"latent_bank_sha256": hashlib.sha256(self.bank).hexdigest()},
            "metrics": {"chain_preterminal_allocations": 0, "chain_preterminal_readbacks": 0,
                        "chain_preterminal_waits": 0, "dit_terminal_waits": 1},
        }

    def test_same_arm_actual_bank_admitted(self):
        control.validate_dit_prefix(self.receipt, self.bank, self.prefix, "generated")

    def test_stale_prefix_rejected_even_with_other_valid_metadata(self):
        stale = b"\x01" + self.prefix[1:]
        with self.assertRaisesRegex(control.base.ControlError, "actual same-arm"):
            control.validate_dit_prefix(self.receipt, self.bank, stale, "generated")

    def test_bank_digest_and_geometry_fail_closed(self):
        for bank in (self.bank[:-4], b"\x01" + self.bank[1:]):
            with self.assertRaises(control.base.ControlError):
                control.validate_dit_prefix(self.receipt, bank, self.prefix, "generated")

    def test_mode_masks_precision_and_sync_mismatch_rejected(self):
        for key, value in (("t0_context_mode", "repeated-photo"), ("source_bf16", False),
                           ("segments", 3), ("t0_hide_caption_keys", True),
                           ("t1_current_text_kv", True), ("keep_t0_video_latents", 0)):
            receipt = copy.deepcopy(self.receipt)
            receipt[key] = value
            with self.assertRaises(control.base.ControlError):
                control.validate_dit_prefix(receipt, self.bank, self.prefix, "generated")
        receipt = copy.deepcopy(self.receipt)
        receipt["metrics"]["chain_preterminal_waits"] = 1
        with self.assertRaises(control.base.ControlError):
            control.validate_dit_prefix(receipt, self.bank, self.prefix, "generated")


if __name__ == "__main__":
    unittest.main()
