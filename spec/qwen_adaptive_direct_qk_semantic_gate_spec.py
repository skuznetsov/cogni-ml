from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_adaptive_direct_qk_semantic_gate.py"


class AdaptiveDirectQKSemanticGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "qwen_adaptive_direct_qk_semantic_gate", SCRIPT
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {SCRIPT}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def _record(self, *, prompt_tokens: int, prompt_sha256: str) -> dict[str, object]:
        observed_samples = 40
        steps = observed_samples + 1
        ids = list(range(100, 100 + observed_samples + 2))
        quality_steps = [
            {
                "phase": "warmup" if index == 0 else "sample",
                "index": -1 if index == 0 else index - 1,
                "baseline_top1_id": 1000 + index,
                "candidate_top1_id": 1000 + index,
                "baseline_top2_id": 2000 + index,
                "candidate_top2_id": 2000 + index,
                "baseline_top1_logit": 4.0,
                "candidate_top1_logit": 3.998 if index == 3 else 4.0,
                "baseline_top2_logit": 3.0,
                "candidate_top2_logit": 2.999 if index == 3 else 3.0,
                "baseline_margin": 1.0,
                "candidate_margin": 0.999 if index == 3 else 1.0,
                "first_logit_delta": 0.002 if index == 3 else 0.0,
                "second_logit_delta": 0.001 if index == 3 else 0.0,
                "margin_delta": 0.001 if index == 3 else 0.0,
                "ranked_top2_matches": 2,
                "top2_set_overlap": 2,
                "exact_top1_covered": True,
                "exact_top2_covered": True,
                "token_ecs": 1.0,
                "paired_logits_valid": True,
            }
            for index in range(steps)
        ]
        return {
            "schema": "qwen-adaptive-t8-decode-ab-v12",
            "comparison": "p4_stage1_bundle",
            "state_source": "real_prompt_prefill",
            "semantic_quality_valid": True,
            "quality_top2": True,
            "quality_top2_production_prefill": True,
            "semantic_coding_quality": True,
            "quality_scope": "real_prefix_production_top1_then_free_run_top2",
            "quality_measurement_valid": True,
            "semantic_trajectory_gate_passed": True,
            "strict_numeric_gate_passed": False,
            "prefill_boundary_mode": "production_top1",
            "semantic_task_scored": False,
            "timing_gate_valid": False,
            "ecs_basis": "output.weight",
            "ecs_interpretation": "static_output_row_cosine_token_proxy",
            "ecs_equal_ids_short_circuit_to_one": True,
            "quality_logit_tolerance": 0.0001,
            "release_build": True,
            "device": "Apple M2 Max",
            "model": "Qwen3.8-27B-Q4_K_M.gguf",
            "prompt_sha256": prompt_sha256,
            "resident_map": "p4;27=bf16,43=bf16,47=bf16,51=bf16",
            "effective_splitk": True,
            "route_certificate_kind": "policy_eligibility",
            "baseline_splitk_chunk": 64,
            "candidate_splitk_chunk": 64,
            "baseline_stage2": "forced_on",
            "candidate_stage2": "forced_on",
            "baseline_direct_qk": False,
            "candidate_direct_qk": True,
            "baseline_v_contiguous": False,
            "candidate_v_contiguous": True,
            "p4_stage1_admission": "forced_off_vs_forced_on",
            "candidate_p4_t8_owners": 12,
            "candidate_bf16_t8_owners": 4,
            "candidate_p4_direct_qk_owners": 12,
            "candidate_p4_v_contiguous_owners": 12,
            "prompt_tokens": prompt_tokens,
            "seeded_prefix_tokens": prompt_tokens,
            "prompt_repeats": 1,
            "samples": 256,
            "observed_samples": observed_samples,
            "requested_samples_completed": False,
            "termination_reason": f"eos_before_sample_{observed_samples}",
            "baseline_eos": True,
            "candidate_eos": True,
            "aligned_eos": True,
            "baseline_eos_step": len(ids) - 1,
            "candidate_eos_step": len(ids) - 1,
            "coding_completion": "aligned_eos",
            "max_seq": 16384,
            "pooled_scratch": True,
            "prefill_gc_guard": True,
            "quality_ranked_top2_matches": steps * 2,
            "quality_ranked_top2_count": steps * 2,
            "quality_min_set_overlap": 2,
            "quality_exact_top1_covered": steps,
            "quality_exact_top2_covered": steps,
            "quality_step_count": steps,
            "quality_steps": quality_steps,
            "quality_min_token_ecs": 1.0,
            "quality_aligned_step_count": steps,
            "quality_gate_passed": False,
            "quality_violations": [
                "prefill_top1_logit_delta=0.0025",
                "sample_3_first_logit_delta=0.002",
                "sample_3_second_logit_delta=0.001",
                "sample_3_margin_delta=0.001",
            ],
            "free_common_prefix": len(ids),
            "first_divergence_step": None,
            "prefill_logit_delta": 0.0025,
            "warm_logit_delta": 0.0,
            "quality_max_second_logit_delta": 0.002,
            "quality_max_margin_delta": 0.001,
            "baseline_output_ids": ids,
            "candidate_output_ids": ids,
            "baseline_text": "```crystal\nmodule Answer\nend\n```",
            "candidate_text": "```crystal\nmodule Answer\nend\n```",
        }

    @staticmethod
    def _external(passed: bool = True) -> dict[str, object]:
        return {"external_pass": passed, "exit_code": 0 if passed else 1}

    def _case(
        self, task_id: str, prompt_tokens: int, *, candidate_pass: bool = True
    ) -> dict[str, object]:
        return {
            "task_id": task_id,
            "record": self._record(
                prompt_tokens=prompt_tokens,
                prompt_sha256=(task_id.encode().hex() + "0" * 64)[:64],
            ),
            "baseline": self._external(),
            "candidate": self._external(candidate_pass),
        }

    def test_accepts_three_distinct_exact_trajectory_coding_cases_across_context_bands(
        self,
    ) -> None:
        report = self.module.evaluate_suite(
            [
                self._case("lower_bound", 7800),
                self._case("stable_unique", 11200),
                self._case("merge_ranges", 15100),
            ]
        )

        self.assertEqual("fixture_semantic_smoke_pass", report["verdict"])
        self.assertEqual([], report["violations"])
        self.assertTrue(report["semantic_fixture_gate_passed"])
        self.assertFalse(report["strict_numeric_gate_passed"])
        self.assertFalse(report["admission_eligible"])
        self.assertEqual(123, report["quality_step_count"])
        self.assertTrue(report["context_coverage"]["low"])
        self.assertTrue(report["context_coverage"]["high"])

    def test_rejects_top2_ecs_and_external_regressions_even_when_top1_ids_match(self) -> None:
        broken = self._case("stable_unique", 11200, candidate_pass=False)
        record = broken["record"]
        assert isinstance(record, dict)
        record["quality_ranked_top2_matches"] = 67
        record["quality_min_set_overlap"] = 1
        record["quality_min_token_ecs"] = 0.98
        quality_steps = record["quality_steps"]
        assert isinstance(quality_steps, list)
        assert isinstance(quality_steps[0], dict)
        quality_steps[0]["exact_top1_covered"] = False

        report = self.module.evaluate_suite(
            [
                self._case("lower_bound", 7800),
                broken,
                self._case("merge_ranges", 15100),
            ]
        )

        self.assertEqual("semantic_regression", report["verdict"])
        joined = "\n".join(report["violations"])
        self.assertIn("ranked top-2", joined)
        self.assertIn("top-2 set overlap", joined)
        self.assertIn("token ECS", joined)
        self.assertIn("exact top-1 is not covered", joined)
        self.assertIn("candidate external specs failed", joined)

    def test_rejects_wrong_route_truncation_and_non_numeric_quality_violations(self) -> None:
        bad = self._case("stable_unique", 11200)
        record = bad["record"]
        assert isinstance(record, dict)
        record["comparison"] = "p4_direct_qk"
        record["aligned_eos"] = False
        record["candidate_eos"] = False
        record["semantic_trajectory_gate_passed"] = False
        record["coding_completion"] = "unaligned_eos"
        record["termination_reason"] = "eos_before_sample_7"
        record["quality_violations"] = ["free_run_divergence_step=9"]

        report = self.module.evaluate_suite(
            [
                self._case("lower_bound", 7800),
                bad,
                self._case("merge_ranges", 15100),
            ]
        )

        self.assertEqual("invalid_evidence", report["verdict"])
        joined = "\n".join(report["violations"])
        self.assertIn("comparison", joined)
        self.assertIn("early EOS termination", joined)
        self.assertIn("non-numeric quality violation", joined)

    def test_rejects_sample_cap_and_non_pinned_fixture_alias(self) -> None:
        truncated = self._case("stable_unique", 11200)
        record = truncated["record"]
        assert isinstance(record, dict)
        record["samples"] = 96
        record["baseline_eos"] = False
        record["candidate_eos"] = False
        record["aligned_eos"] = False
        record["baseline_eos_step"] = None
        record["candidate_eos_step"] = None
        record["coding_completion"] = "sample_limit"
        record["requested_samples_completed"] = True
        record["termination_reason"] = "requested_samples_completed"

        alias = self._case("stable_unique_alias", 15100)
        report = self.module.evaluate_suite(
            [self._case("lower_bound", 7800), truncated, alias]
        )

        self.assertEqual("invalid_evidence", report["verdict"])
        joined = "\n".join(report["violations"])
        self.assertIn("samples must be at least 256", joined)
        self.assertIn("coding_completion must be aligned_eos", joined)
        self.assertIn("exactly the three pinned Crystal fixtures", joined)

    def test_extracts_exactly_one_v12_probe_record(self) -> None:
        record = self._record(prompt_tokens=7800, prompt_sha256="a" * 64)
        log = "noise\nQBIT_T8_DECODE_JSON=" + json.dumps(record) + "\n"

        self.assertEqual(record, self.module.extract_probe_record(log))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.module.extract_probe_record(log + log)


if __name__ == "__main__":
    unittest.main()
