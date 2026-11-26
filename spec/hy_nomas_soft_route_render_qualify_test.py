"""Seeded falsifiers for trajectory parity, replay, and inventory coverage."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "support"))
import hy_nomas_soft_route_render_qualify as q


class RenderQualificationTests(unittest.TestCase):
    def fixtures(self):
        source = {name: np.asarray([1.0, 2.0], dtype="<f4") for name in q.TENSOR_NAMES}
        native = {name: value.copy() for name, value in source.items()}
        accepted = {name: source[name].copy() for name in q.REPLAY_NAMES}
        velocities = {f"{case}.step1.velocity": source[f"{case}.step1.velocity"].copy() for case in q.CASES}
        return source, native, accepted, velocities

    def test_equal_outputs_pass_all_eighteen_comparisons(self):
        result = q.evaluate_arrays(*self.fixtures())
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["parity"]), 18)
        self.assertEqual(len(result["exact_replays"]), 22)

    def test_seeded_late_biased_trajectory_error_fails(self):
        items = self.fixtures()
        items[1]["A2_high.step3.latent"][1] += 0.015625
        result = q.evaluate_arrays(*items)
        self.assertEqual(result["failed_gates"], ["parity:A2_high.step3.latent"])

    def test_shared_wrong_zero_is_rejected_even_when_source_native_agree(self):
        items = self.fixtures()
        for index in (0, 1):
            items[index]["zero.step2.latent"][0] += 0.0078125
        result = q.evaluate_arrays(*items)
        self.assertFalse(result["passed"])
        self.assertTrue(all(x.startswith("exact-replay:") for x in result["failed_gates"]))

    def test_small_zero_replay_drift_is_rejected_inside_tolerance(self):
        items = self.fixtures()
        items[1]["zero.step4.latent"][0] += np.float32(1e-6)
        result = q.evaluate_arrays(*items)
        self.assertEqual(result["parity"]["zero.step4.latent"]["outside_atol_rtol"], 0)
        self.assertFalse(result["passed"])

    def test_wrong_biased_first_velocity_is_rejected_against_frozen_result(self):
        items = self.fixtures()
        for index in (0, 1):
            items[index]["A1_high.step1.velocity"][0] = 3.0
        self.assertFalse(q.evaluate_arrays(*items)["passed"])

    def test_missing_middle_step_fails_coverage(self):
        items = self.fixtures()
        del items[1]["A1_high.step2.latent"]
        with self.assertRaisesRegex(q.frozen.QualificationError, "inventory"):
            q.evaluate_arrays(*items)

    def test_replay_inventory_cannot_drop_an_intermediate_zero(self):
        items = self.fixtures()
        del items[2]["zero.step2.latent"]
        with self.assertRaisesRegex(q.frozen.QualificationError, "replay inventory"):
            q.evaluate_arrays(*items)

    def native_receipt(self):
        return {"execution": {"case_order": list(q.CASES), "steps": 4, "full_block_calls": 864,
                "t0_block_calls": 216, "t1_block_calls": 648, "refiner_calls": 4,
                "patch_calls": 12, "final_head_calls": 12, "euler_updates": 12,
                "command_chains": 1, "terminal_waits": 1, "named_allocations_during_encode": 0,
                "activation_readbacks_before_terminal_wait": 0, "postwait_tensor_readbacks": 18,
                "decoder_calls": 0, "observed_attention_calls": {name: 216 for name in ("T0", *q.CASES)}},
                "resource_accounting": {"actual_setup_named_bytes": 10,
                "peak_named_bytes": 11}, "max_named_bytes": 12}

    def test_second_wait_is_rejected(self):
        doc = self.native_receipt()
        q.validate_execution(doc, "native")
        doc["execution"]["terminal_waits"] = 2
        with self.assertRaisesRegex(q.frozen.QualificationError, "counter mismatch"):
            q.validate_execution(doc, "native")

    def test_intermediate_readback_is_rejected(self):
        doc = self.native_receipt()
        doc["execution"]["activation_readbacks_before_terminal_wait"] = 1
        with self.assertRaisesRegex(q.frozen.QualificationError, "counter mismatch"):
            q.validate_execution(doc, "native")

    def test_actual_named_buffers_may_not_exceed_the_plan(self):
        doc = self.native_receipt()
        doc["resource_accounting"]["actual_setup_named_bytes"] = 12
        with self.assertRaisesRegex(q.frozen.QualificationError, "budget mismatch"):
            q.validate_execution(doc, "native")

    def test_missing_middle_arm_attention_fails_even_when_declared_total_is_correct(self):
        doc = self.native_receipt()
        doc["execution"]["observed_attention_calls"]["A1_high"] = 215
        with self.assertRaisesRegex(q.frozen.QualificationError, "coverage mismatch"):
            q.validate_execution(doc, "native")


if __name__ == "__main__":
    unittest.main()
