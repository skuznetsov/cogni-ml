#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_qbit_quality_corpus.py"
SPEC = importlib.util.spec_from_file_location("qwen_qbit_quality_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SemanticJudgeTest(unittest.TestCase):
    def _quality_record(self) -> dict[str, object]:
        return {
            "schema": "qwen-qbit-quality-v1",
            "policy": "resident[p4;27=bf16,43=bf16,47=bf16,51=bf16]",
            "exact_text": "The final available count is 84.",
            "candidate_text": "The final available count is 84.",
            "exact_ids": [1, 2, 3, 4],
            "candidate_ids": [1, 2, 3, 4],
            "exact_ended_with_eos": True,
            "candidate_ended_with_eos": True,
            "retire_order_top1_matches": 4,
            "retire_order_top1_count": 4,
            "teacher_top2_ranked_matches": 6,
            "teacher_top2_ranked_count": 6,
            "teacher_top2_set_overlap": 6,
            "teacher_top2_set_overlap_count": 6,
            "teacher_exact_top1_covered": 3,
            "teacher_top2_steps": 3,
            "teacher_token_ecs_mean": 1.0,
            "prefix_raw_bytes": 100,
            "prefix_payload_bytes": 20,
        }

    def test_selects_named_cases_without_reordering_the_manifest(self) -> None:
        cases = [{"name": "first"}, {"name": "second"}, {"name": "third"}]

        selected = MODULE.select_cases(cases, ["third", "first"])

        self.assertEqual(["first", "third"], [case["name"] for case in selected])
        with self.assertRaisesRegex(ValueError, "unknown cases"):
            MODULE.select_cases(cases, ["missing"])

    def test_requires_facts_alternatives_and_minimum_length(self) -> None:
        rules = {
            "required_all": [r"\b84\b", r"reserved"],
            "required_any": [[r"available", r"unreserved"]],
            "forbidden": [r"\b85\b"],
            "min_words": 6,
        }

        self.assertEqual(
            [],
            MODULE.semantic_failures(
                "After the update, 84 units remain available and reserved stock stays separate.",
                rules,
            ),
        )
        failures = MODULE.semantic_failures(
            "After the update, 85 units remain available.",
            rules,
        )
        self.assertTrue(any("required_all" in failure for failure in failures))
        self.assertTrue(any("forbidden" in failure for failure in failures))

    def test_extracts_probe_json_from_guarded_runner_noise(self) -> None:
        log = """[PREFLIGHT] System memory free 80% > 12%
not json
QBIT_QUALITY_JSON={"schema":"qwen-qbit-quality-v1","policy":"p4"}
[EXIT: 0]
"""

        records = MODULE.extract_probe_records(log)

        self.assertEqual(1, len(records))
        self.assertEqual("p4", records[0]["policy"])

    def test_extracts_one_requested_generation_budget_from_the_probe_header(self) -> None:
        log = "  prompt=\"x\" requested_gen=256 observed_gen=223 max_seq=300\n"

        self.assertEqual(256, MODULE.extract_requested_generation(log))
        with self.assertRaisesRegex(ValueError, "one requested_gen"):
            MODULE.extract_requested_generation("requested_gen=128\nrequested_gen=256\n")

    def test_invalid_exact_baseline_is_not_reported_as_qbit_failure(self) -> None:
        case = {
            "name": "ledger",
            "min_exact_tokens": 4,
            "rules": {"required_all": [r"\b84\b"]},
        }
        records = [
            {
                "schema": "qwen-qbit-quality-v1",
                "policy": "p4",
                "exact_text": "The answer is unknown.",
                "candidate_text": "The answer is 84.",
                "exact_ids": [1, 2, 3, 4],
                "exact_ended_with_eos": False,
                "candidate_ended_with_eos": True,
            }
        ]

        result = MODULE.evaluate_case(case, records)

        self.assertEqual("invalid_baseline", result["status"])
        self.assertNotIn("policy_results", result)

    def test_scores_each_policy_without_scalarizing_the_quality_vector(self) -> None:
        case = {
            "name": "ledger",
            "min_exact_tokens": 4,
            "rules": {"required_all": [r"\b84\b"]},
        }
        base = {
            "schema": "qwen-qbit-quality-v1",
            "exact_text": "The final available count is 84.",
            "exact_ids": [1, 2, 3, 4],
            "candidate_ids": [1, 2, 3, 4],
            "exact_ended_with_eos": True,
            "candidate_ended_with_eos": True,
            "retire_order_top1_matches": 3,
            "retire_order_top1_count": 4,
            "teacher_top2_ranked_matches": 5,
            "teacher_top2_ranked_count": 6,
            "teacher_top2_set_overlap": 6,
            "teacher_top2_set_overlap_count": 6,
            "teacher_exact_top1_covered": 3,
            "teacher_top2_steps": 3,
            "teacher_token_ecs_mean": 0.9,
            "prefix_raw_bytes": 100,
            "prefix_payload_bytes": 20,
        }
        records = [
            {**base, "policy": "p4", "candidate_text": "The final available count is 84."},
            {**base, "policy": "adaptive[51:k0=bf16]", "candidate_text": "The final count is 83."},
        ]

        result = MODULE.evaluate_case(case, records)

        self.assertEqual("valid", result["status"])
        self.assertTrue(result["policy_results"]["p4"]["meaning_preserved"])
        self.assertFalse(result["policy_results"]["adaptive[51:k0=bf16]"]["meaning_preserved"])
        self.assertEqual(3, result["policy_results"]["p4"]["top1_matches"])
        self.assertEqual(5.0, result["policy_results"]["p4"]["density"])

    def test_rejects_a_candidate_that_contains_the_facts_but_did_not_reach_eos(self) -> None:
        case = {
            "name": "ledger",
            "min_exact_tokens": 4,
            "rules": {"required_all": [r"\b84\b"]},
        }
        record = {
            "schema": "qwen-qbit-quality-v1",
            "policy": "p4",
            "exact_text": "The final available count is 84.<|im_end|>",
            "candidate_text": "The final available count is 84, and",
            "exact_ids": [1, 2, 3, 4],
            "candidate_ids": [1, 2, 3, 4],
            "exact_ended_with_eos": True,
            "candidate_ended_with_eos": False,
            "retire_order_top1_matches": 4,
            "retire_order_top1_count": 4,
            "teacher_top2_ranked_matches": 6,
            "teacher_top2_ranked_count": 6,
            "teacher_top2_set_overlap": 6,
            "teacher_top2_set_overlap_count": 6,
            "teacher_exact_top1_covered": 3,
            "teacher_top2_steps": 3,
            "teacher_token_ecs_mean": 1.0,
            "prefix_raw_bytes": 100,
            "prefix_payload_bytes": 20,
        }

        result = MODULE.evaluate_case(case, [record])

        self.assertEqual("valid", result["status"])
        self.assertFalse(result["policy_results"]["p4"]["meaning_preserved"])
        self.assertIn("candidate did not reach EOS", result["policy_results"]["p4"]["semantic_failures"])

    def test_resident_mode_requires_gpu_ownership_proof(self) -> None:
        case = {
            "name": "ledger",
            "min_exact_tokens": 4,
            "rules": {"required_all": [r"\b84\b"]},
        }

        with self.assertRaisesRegex(ValueError, "resident GPU proof"):
            MODULE.evaluate_case(case, [self._quality_record()], require_resident=True)

        record = {
            **self._quality_record(),
            "execution_mode": "resident_gpu",
            "full_attention_layers": 16,
            "resident_layers": 16,
            "resident_f32_owner_layers": [],
            "resident_cache_consistent": True,
        }
        result = MODULE.evaluate_case(case, [record], require_resident=True)

        self.assertEqual("valid", result["status"])
        row = result["policy_results"][record["policy"]]
        self.assertEqual(16, row["resident_layers"])
        self.assertEqual([], row["resident_f32_owner_layers"])

        with self.assertRaisesRegex(ValueError, "invalid resident GPU proof"):
            MODULE.evaluate_case(
                case,
                [{**record, "resident_layers": 15}],
                require_resident=True,
            )
        with self.assertRaisesRegex(ValueError, "invalid resident GPU proof"):
            MODULE.evaluate_case(
                case,
                [{**record, "resident_f32_owner_layers": ""}],
                require_resident=True,
            )

    def test_resident_command_uses_only_the_runtime_map(self) -> None:
        command = MODULE.build_probe_command(
            run_safe=Path("safe"),
            binary=Path("resident-probe"),
            timeout=120,
            max_mem_mb=8192,
            generation=256,
            retire_chunk=32,
            model=None,
            policy_maps=("p4;27=bf16,43=bf16,47=bf16,51=bf16",),
            prompt="hello",
            resident=True,
        )

        self.assertIn("--resident-map=p4;27=bf16,43=bf16,47=bf16,51=bf16", command)
        self.assertNotIn("--precisions=4", command)
        self.assertFalse(any(value.startswith("--adaptive-map=") for value in command))


if __name__ == "__main__":
    unittest.main()
