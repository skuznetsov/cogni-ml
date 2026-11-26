#!/usr/bin/env python3
"""Offline checks for the routed/shared SwiGLU paired measurement driver."""

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/drama3/measure_host_pairs.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("measure_host_pairs_expert_swiglu", SCRIPT)
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


class ExpertSwiGLUDriverSpec(unittest.TestCase):
    def test_schedule_requires_explicit_baseline_zero(self):
        self.assertEqual(driver.parse_expert_swiglu_schedule(" 0, 1,0 "), ["0", "1", "0"])
        for invalid in ("", "1", "0,2", "0,true", "0,"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "include baseline 0"):
                    driver.parse_expert_swiglu_schedule(invalid)

    def test_schedule_option_is_mutually_exclusive(self):
        parser = driver.build_parser()
        args = REQUIRED_ARGS + [
            "--expert-swiglu-schedule", "0,1",
            "--expert-down-sum-schedule", "0,1",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(args)
        self.assertEqual(error.exception.code, 2)

    def test_arm_controls_pin_other_experiments_off(self):
        self.assertEqual(driver.CONTROLS["DRAMA3_EXPERT_SWIGLU_FUSION"], "0")
        self.assertEqual(driver.CONTROLS["DRAMA3_PREFILL_CHUNK"], "1")
        for mode in ("0", "1"):
            controls = dict(
                driver.CONTROLS,
                DRAMA3_GENERATION_SCRATCH="0",
                DRAMA3_HOST_PROFILE="0",
            )
            driver.configure_expert_swiglu_arm(controls, mode)
            self.assertEqual(controls["DRAMA3_EXPERT_SWIGLU_FUSION"], mode)
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
                "DRAMA3_EXPERT_DOWN_SUM_FUSION",
            ):
                self.assertEqual(controls[selector], "0", selector)

    def test_frozen_selector_and_both_dispatch_counters_match_arm(self):
        backbone_rows = 29 + min(60, 64 - 1)
        expected_dispatches = backbone_rows * 40
        for mode in ("0", "1"):
            expected = mode == "1"
            report = {
                "resident_expert_swiglu_fusion": expected,
                "resident_expert_swiglu_fused_dispatches": expected_dispatches if expected else 0,
                "resident_dense_swiglu_fused_dispatches": expected_dispatches if expected else 0,
                "prompt_rows": 29,
                "frames": [[0] * 10 for _ in range(60)],
                "max_frames": 64,
            }
            driver.validate_expert_swiglu_fusion(report, mode)
            with self.subTest(mode=mode):
                bad = dict(report)
                bad["resident_expert_swiglu_fusion"] = not expected
                with self.assertRaisesRegex(AssertionError, "differs from requested arm"):
                    driver.validate_expert_swiglu_fusion(bad, mode)

        for bad_report in (
            {"resident_expert_swiglu_fusion": 1,
             "resident_expert_swiglu_fused_dispatches": 1,
             "resident_dense_swiglu_fused_dispatches": 1},
            {"resident_expert_swiglu_fusion": True,
             "resident_expert_swiglu_fused_dispatches": 0,
             "resident_dense_swiglu_fused_dispatches": 1},
            {"resident_expert_swiglu_fusion": True,
             "resident_expert_swiglu_fused_dispatches": 1,
             "resident_dense_swiglu_fused_dispatches": 0},
            {"resident_expert_swiglu_fusion": False,
             "resident_expert_swiglu_fused_dispatches": 1,
             "resident_dense_swiglu_fused_dispatches": 0},
        ):
            bad_report = {
                "prompt_rows": 29,
                "frames": [[0] * 10 for _ in range(60)],
                "max_frames": 64,
                **bad_report,
            }
            with self.subTest(report=bad_report):
                with self.assertRaises(AssertionError):
                    driver.validate_expert_swiglu_fusion(bad_report, "1" if bad_report["resident_expert_swiglu_fusion"] else "0")

        for count in (expected_dispatches - 1, expected_dispatches + 1):
            bad_report = {
                "resident_expert_swiglu_fusion": True,
                "resident_expert_swiglu_fused_dispatches": count,
                "resident_dense_swiglu_fused_dispatches": expected_dispatches,
                "prompt_rows": 29,
                "frames": [[0] * 10 for _ in range(60)],
                "max_frames": 64,
            }
            with self.subTest(count=count):
                with self.assertRaisesRegex(AssertionError, "expected 3560"):
                    driver.validate_expert_swiglu_fusion(bad_report, "1")


if __name__ == "__main__":
    unittest.main()
