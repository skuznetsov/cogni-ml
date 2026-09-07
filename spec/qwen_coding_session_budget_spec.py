import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "budget", Path(__file__).resolve().parents[1] / "scripts/qwen_coding_session_budget.py")
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)
PERF = ("route=greedy,total_ms=100,render_ms=1,load_ms=10,tokenize_ms=2,"
        "cache_lookup_ms=3,prefill_top1_ms=40,decode_body_ms=30,greedy_ms=70")
CALL = "💭 Calling LLM (3 messages, 6 tools)...\n💭 CogniQwen perf: " + PERF + "\n"
FINISHED = "✅ Agent finished\n"


class BudgetTest(unittest.TestCase):
    def test_identical_calls_count_but_final_summary_does_not(self):
        result = budget.summarize(CALL * 2 + FINISHED + "   CogniQwen perf: " + PERF, 250)
        self.assertEqual(result["call_count"], 2)
        self.assertEqual(result["provider_totals_ms"]["total_ms"], 200)
        self.assertEqual(result["provider_totals_ms"]["provider_unattributed_ms"], 28)
        self.assertEqual(result["outside_provider_ms"], 50)
        self.assertNotIn("greedy_ms", result["provider_totals_ms"])

    def test_ansi_is_supported(self):
        self.assertEqual(budget.summarize("\x1b[34m" + CALL + FINISHED + "\x1b[0m", 120)["call_count"], 1)

    def test_final_summary_alone_is_not_a_session(self):
        with self.assertRaises(ValueError):
            budget.summarize("CogniQwen perf: " + PERF, 120)

    def test_missing_or_duplicate_call_record_is_rejected(self):
        for text in (CALL + "💭 Calling LLM (3 messages)...", CALL.splitlines()[0] + "\n" + CALL,
                     CALL + "💭 CogniQwen perf: " + PERF):
            with self.subTest(text=text), self.assertRaises(ValueError):
                budget.summarize(text, 250)

    def test_nonfinite_and_negative_values_are_rejected(self):
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.summarize(CALL.replace("load_ms=10", "load_ms=" + value), 120)

    def test_nested_or_inconsistent_times_are_rejected(self):
        for text, wall in ((CALL.replace("total_ms=100", "total_ms=20"), 120), (CALL, 50)):
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                budget.summarize(text, wall)

    def test_wall_must_be_observed_positive_finite(self):
        for wall in (0, -1, float("nan"), float("inf")):
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                budget.summarize(CALL, wall)

    def test_duplicate_field_is_ambiguous_and_rejected(self):
        with self.assertRaises(ValueError):
            budget.summarize(CALL.rstrip() + ",total_ms=110\n", 120)

    def test_adapter_error_cannot_be_hidden_by_harness_finished(self):
        failed = (CALL + "💭 Calling LLM (4 messages, 6 tools)...\n"
                  "💭 CogniQwen perf: route=exception error=Exception\n"
                  "✅ Agent finished\n   CogniQwen perf: route=exception error=Exception\n")
        with self.assertRaisesRegex(ValueError, "missing performance fields"):
            budget.summarize(failed, 120)

    def test_truncation_after_complete_call_and_concatenated_sessions_reject(self):
        for text in (CALL, CALL + FINISHED + CALL + FINISHED, CALL + FINISHED * 2):
            with self.subTest(text=text), self.assertRaises(ValueError):
                budget.summarize(text, 250)


if __name__ == "__main__":
    unittest.main()
