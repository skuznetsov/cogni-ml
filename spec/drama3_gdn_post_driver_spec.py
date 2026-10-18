#!/usr/bin/env python3
"""Offline checks for the GDN post paired measurement driver."""

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/drama3/measure_host_pairs.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("measure_host_pairs_gdn_post", SCRIPT)
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


def report(mode, prompt_rows=5, frames=None, max_frames=64, dispatches=None):
    frames = [1, 2] if frames is None else frames
    expected = (prompt_rows + min(len(frames), max_frames - 1)) * 30
    if dispatches is None:
        dispatches = expected if mode == "1" else 0
    return {
        "backend": "metal_resident",
        "resident_gdn_post_fusion": mode == "1",
        "resident_gdn_post_fused_dispatches": dispatches,
        "prompt_rows": prompt_rows,
        "frames": frames,
        "max_frames": max_frames,
    }


def paired_report(mode="0", prompt_rows=5, frames=None, max_frames=64):
    result = report(mode, prompt_rows=prompt_rows, frames=frames, max_frames=max_frames)
    result.update({
        "text": driver.INPUT["text"],
        "seed": driver.INPUT["seed"],
        "temperature": driver.INPUT["temperature"],
        "top_k": driver.INPUT["top_k"],
        "top_p": driver.INPUT["top_p"],
        "stopped_on_eos": True,
        "truncated": False,
        "sampler_inputs": [0.1, 0.2],
        "resident_readbacks": {"terminal": 1},
        "resident_graph": {"submitted": 5, "completed": 5, "completion_waits": 5, "pending": 0},
        "resident_generation_scratch": {"enabled": False},
    })
    return result


class GDNPostDriverSpec(unittest.TestCase):
    def test_schedule_requires_zero_and_accepts_repeated_binary_arms(self):
        self.assertEqual(driver.parse_gdn_post_schedule(" 0, 1,1,0 "), ["0", "1", "1", "0"])
        for invalid in ("", "1", "0,2", "0,true", "0,"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "include baseline 0"):
                    driver.parse_gdn_post_schedule(invalid)

    def test_schedule_option_is_exclusive_with_other_benchmark_schedules(self):
        parser = driver.build_parser()
        args = REQUIRED_ARGS + [
            "--gdn-post-schedule", "0,1",
            "--expert-down-sum-schedule", "0,1",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(args)
        self.assertEqual(error.exception.code, 2)

    def test_arm_controls_pin_other_experiments_off_and_select_coalesced_dac(self):
        self.assertEqual(driver.CONTROLS["DRAMA3_GDN_POST_FUSION"], "0")
        for mode in ("0", "1"):
            controls = dict(
                driver.CONTROLS,
                DRAMA3_GENERATION_SCRATCH="0",
                DRAMA3_HOST_PROFILE="0",
            )
            driver.configure_gdn_post_arm(controls, mode)
            self.assertEqual(controls["DRAMA3_GDN_POST_FUSION"], mode)
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
                "DRAMA3_EXPERT_SWIGLU_FUSION",
            ):
                self.assertEqual(controls[selector], "0", selector)

    def test_frozen_selector_is_exact_boolean_matching_requested_arm(self):
        for mode in ("0", "1"):
            expected = mode == "1"
            driver.validate_gdn_post_fusion(
                report(mode), mode, expected_prompt_rows=5, expected_max_frames=64
            )
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(AssertionError, "differs from requested arm"):
                    driver.validate_gdn_post_fusion(
                        {**report(mode), "resident_gdn_post_fusion": not expected}, mode,
                        expected_prompt_rows=5, expected_max_frames=64,
                    )
        for invalid in (1, 0, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(AssertionError, "differs from requested arm"):
                    driver.validate_gdn_post_fusion(
                        {**report("1"), "resident_gdn_post_fusion": invalid}, "1",
                        expected_prompt_rows=5, expected_max_frames=64,
                    )

    def test_dispatch_count_requires_complete_candidate_coverage(self):
        # Early EOS: frames is shorter than max_frames and counts generated work.
        early_eos = report("1", prompt_rows=5, frames=[1, 2], max_frames=64)
        driver.validate_gdn_post_fusion(
            early_eos, "1", expected_prompt_rows=5, expected_max_frames=64
        )
        self.assertEqual(early_eos["resident_gdn_post_fused_dispatches"], (5 + 2) * 30)

        # Max-frame stop: only max_frames - 1 generated frame transitions are covered.
        at_limit = report("1", prompt_rows=5, frames=list(range(64)), max_frames=64)
        driver.validate_gdn_post_fusion(
            at_limit, "1", expected_prompt_rows=5, expected_max_frames=64
        )
        self.assertEqual(at_limit["resident_gdn_post_fused_dispatches"], (5 + 63) * 30)

        # A fallback on even one layer-row must fail despite selecting fusion.
        partial_fallback = report("1", prompt_rows=5, frames=[1, 2], max_frames=64, dispatches=209)
        with self.assertRaisesRegex(AssertionError, "expected 210"):
            driver.validate_gdn_post_fusion(
                partial_fallback, "1", expected_prompt_rows=5, expected_max_frames=64
            )

        # Adversarial reports cannot lower either operand of the coverage count.
        underreported_cap = report("1", prompt_rows=5, frames=list(range(20)), max_frames=10)
        with self.assertRaisesRegex(AssertionError, "max frames differ from requested input"):
            driver.validate_gdn_post_fusion(
                underreported_cap, "1", expected_prompt_rows=5, expected_max_frames=64
            )
        underreported_prompt = report("1", prompt_rows=4, frames=[1, 2], dispatches=180)
        with self.assertRaisesRegex(AssertionError, "prompt rows differ from paired baseline"):
            driver.validate_gdn_post_fusion(
                underreported_prompt, "1", expected_prompt_rows=5, expected_max_frames=64
            )

    def test_baseline_dispatch_count_is_zero_and_counter_must_be_integer_not_bool(self):
        driver.validate_gdn_post_fusion(
            report("0"), "0", expected_prompt_rows=5, expected_max_frames=64
        )
        with self.assertRaisesRegex(AssertionError, "baseline arm used"):
            driver.validate_gdn_post_fusion(
                report("0", dispatches=1), "0", expected_prompt_rows=5, expected_max_frames=64
            )
        with self.assertRaisesRegex(AssertionError, "invalid GDN post fused dispatch count"):
            driver.validate_gdn_post_fusion(
                report("1", dispatches=True), "1", expected_prompt_rows=5, expected_max_frames=64
            )

    def test_generic_pair_validation_rejects_reported_request_and_termination_drift(self):
        baseline_report = paired_report()
        baseline = {"report": baseline_report, "wav_sha256": "a" * 64}
        driver.validate(paired_report("1"), "a" * 64, baseline)

        # Cap termination is distinct from EOS and is pinned to the request.
        capped_report = paired_report(frames=list(range(driver.INPUT["max_frames"])))
        capped_report["stopped_on_eos"] = False
        capped_report["truncated"] = True
        capped_baseline = {"report": capped_report, "wav_sha256": "a" * 64}
        driver.validate(paired_report("1", frames=list(range(driver.INPUT["max_frames"])), max_frames=64)
                        | {"stopped_on_eos": False, "truncated": True}, "a" * 64, capped_baseline)
        impossible_termination = dict(capped_report, stopped_on_eos=True, truncated=False)
        with self.assertRaisesRegex(AssertionError, "EOS flag disagrees with frame count"):
            driver.validate(
                impossible_termination, "a" * 64,
                {"report": impossible_termination, "wav_sha256": "a" * 64},
            )

        # The baseline and candidate can collude through the same malformed
        # max_frames field; the pinned request remains the independent oracle.
        malformed_baseline = {"report": paired_report(max_frames=10), "wav_sha256": "a" * 64}
        with self.assertRaisesRegex(AssertionError, "max_frames"):
            driver.validate(paired_report("1", max_frames=10), "a" * 64, malformed_baseline)

        mutations = (
            ("prompt_rows", 4),
            ("stopped_on_eos", False),
            ("truncated", True),
            ("text", "different prompt"),
            ("seed", 2),
            ("temperature", 0.7),
            ("top_k", 29),
            ("top_p", 0.8),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                candidate = paired_report("1")
                candidate[key] = value
                with self.assertRaisesRegex(AssertionError, key):
                    driver.validate(candidate, "a" * 64, baseline)

        for key in ("max_frames", "prompt_rows", "stopped_on_eos", "truncated", "text", "seed",
                    "temperature", "top_k", "top_p"):
            with self.subTest(missing=key):
                candidate = paired_report("1")
                del candidate[key]
                with self.assertRaisesRegex(AssertionError, key):
                    driver.validate(candidate, "a" * 64, baseline)


if __name__ == "__main__":
    unittest.main()
