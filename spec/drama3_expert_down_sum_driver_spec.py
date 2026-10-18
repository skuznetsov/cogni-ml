#!/usr/bin/env python3
"""Offline checks for the expert down-sum paired measurement driver."""

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/drama3/measure_host_pairs.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("measure_host_pairs_expert_down_sum", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


REQUIRED_ARGS = [
    "--binary", "worker",
    "--model", "model",
    "--output", "output",
    "--baseline", "baseline",
]


class ExpertDownSumDriverSpec(unittest.TestCase):
    def test_schedule_accepts_explicit_zero_baseline_and_binary_candidate(self):
        self.assertEqual(driver.parse_expert_down_sum_schedule(" 0, 1,0 "), ["0", "1", "0"])
        for invalid in ("", "1", "0,2", "0,true", "0,"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "include baseline 0"):
                    driver.parse_expert_down_sum_schedule(invalid)

    def test_schedule_option_is_exclusive_with_other_benchmark_schedules(self):
        parser = driver.build_parser()
        args = REQUIRED_ARGS + [
            "--expert-down-sum-schedule", "0,1",
            "--tower-ffn-pair-schedule", "0,1",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(args)
        self.assertEqual(error.exception.code, 2)

    def test_arm_controls_pin_clean_baseline_and_select_coalesced_dac(self):
        self.assertEqual(driver.CONTROLS["DRAMA3_EXPERT_DOWN_SUM_FUSION"], "0")
        for mode in ("0", "1"):
            controls = dict(
                driver.CONTROLS,
                DRAMA3_GENERATION_SCRATCH="0",
                DRAMA3_HOST_PROFILE="0",
            )
            driver.configure_expert_down_sum_arm(controls, mode)
            self.assertEqual(controls["DRAMA3_EXPERT_DOWN_SUM_FUSION"], mode)
            self.assertEqual(controls["DRAMA3_DAC_CONV_LAYOUT"], "coalesced")
            self.assertEqual(controls["DRAMA3_DAC_SCRATCH"], "0")
            self.assertEqual(controls["DRAMA3_GENERATION_SCRATCH"], "0")
            self.assertEqual(controls["DRAMA3_HOST_PROFILE"], "0")
            self.assertEqual(controls["DRAMA3_MEMORY_PROFILE"], "0")
            self.assertEqual(controls["DRAMA3_ALLOCATION_PROFILE"], "0")
            self.assertEqual(controls["DRAMA3_METAL_RESOURCES"], "0")
            for selector in (
                "DRAMA3_TOWER_EMBEDDING_FUSION",
                "DRAMA3_BACKBONE_DENSE_FUSION",
                "DRAMA3_EXPERT_PACKED_ROWS",
                "DRAMA3_TOWER_FFN_PAIR_FUSION",
            ):
                self.assertEqual(controls[selector], "0", selector)

    def test_frozen_report_selector_must_match_requested_arm_and_be_boolean(self):
        for mode in ("0", "1"):
            expected = mode == "1"
            driver.validate_expert_down_sum_fusion(
                {"resident_expert_down_sum_fusion": expected}, mode
            )
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(AssertionError, "differs from requested arm"):
                    driver.validate_expert_down_sum_fusion(
                        {"resident_expert_down_sum_fusion": not expected}, mode
                    )
        with self.assertRaisesRegex(AssertionError, "differs from requested arm"):
            driver.validate_expert_down_sum_fusion(
                {"resident_expert_down_sum_fusion": 1}, "1"
            )


if __name__ == "__main__":
    unittest.main()
