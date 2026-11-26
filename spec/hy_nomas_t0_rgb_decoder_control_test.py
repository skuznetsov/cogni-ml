"""Fail-closed same-arm DCAE/DiT/decoder joins, without GPU or model weights."""
from pathlib import Path
import copy
import hashlib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_t0_rgb_decoder_control as control


class SameArmTest(unittest.TestCase):
    def setUp(self):
        # Distinct channel/frame values catch accidental contiguous slicing.
        import struct
        self.bank = b"".join(struct.pack("<f", channel * 180 + frame) * (9 * 16)
                             for channel in range(32) for frame in range(180))
        self.prefix = control.history.channel_prefix(self.bank, frames=180, selected=60)
        self.dcae = control.latent_segment(self.bank, 0)
        def sha(value):
            return hashlib.sha256(value).hexdigest()
        self.receipt = {
            "schema": control.SCHEMA, "context_mode": "moving",
            "inputs_sha256": control.base.INPUT_MANIFEST_SHA256,
            "upstream_revision": control.base.UPSTREAM_REVISION,
            "source_bf16": True, "segments": 2, "steps_per_segment": 4,
            "output": {"latent_bank_sha256": sha(self.bank), "t0_sha256": sha(self.dcae),
                       "t1_sha256": sha(control.latent_segment(self.bank, 30)),
                       "prefix60_sha256": sha(self.prefix), "dcae_latent_f32_sha256": sha(self.dcae)},
            "metrics": {"chain_preterminal_allocations": 0, "chain_preterminal_readbacks": 0,
                        "chain_preterminal_waits": 0, "terminal_waits": 1},
        }

    def check(self, receipt=None, bank=None, prefix=None, dcae=None):
        control.validate_dit_prefix(self.receipt if receipt is None else receipt,
                                    self.bank if bank is None else bank,
                                    self.prefix if prefix is None else prefix,
                                    "moving", self.dcae if dcae is None else dcae)

    def test_actual_same_arm_join(self):
        self.check()

    def test_wrong_prefix_or_dcae_endpoint_rejected(self):
        for field, raw in (("prefix", self.prefix), ("dcae", self.dcae)):
            changed = b"\x01" + raw[1:]
            with self.assertRaises(control.base.ControlError):
                self.check(**{field: changed})

    def test_contiguous_bank_prefix_is_not_channel_major_prefix(self):
        with self.assertRaisesRegex(control.base.ControlError, "same-arm"):
            self.check(prefix=self.bank[:len(self.prefix)])

    def test_wrong_mode_precision_manifest_or_counters_rejected(self):
        for key, value in (("context_mode", "repeat-last"), ("source_bf16", False),
                           ("inputs_sha256", "0" * 64), ("segments", 3)):
            changed = copy.deepcopy(self.receipt)
            changed[key] = value
            with self.assertRaises(control.base.ControlError):
                self.check(receipt=changed)
        for key in self.receipt["metrics"]:
            changed = copy.deepcopy(self.receipt)
            changed["metrics"][key] += 1
            with self.assertRaisesRegex(control.base.ControlError, "counter"):
                self.check(receipt=changed)

    def test_bank_or_endpoint_digest_not_trusted_from_receipt(self):
        with self.assertRaisesRegex(control.base.ControlError, "bank digest"):
            self.check(bank=b"\x01" + self.bank[1:])
        changed = copy.deepcopy(self.receipt)
        changed["output"]["t1_sha256"] = "0" * 64
        with self.assertRaisesRegex(control.base.ControlError, "endpoint digest"):
            self.check(receipt=changed)


if __name__ == "__main__":
    unittest.main()
