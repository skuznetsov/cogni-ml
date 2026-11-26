"""CPU falsifiers for frozen-head input binding and score summaries."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parent / "support" / "hy_nomas_query_head_robustness.py"
SPEC = importlib.util.spec_from_file_location("hy_nomas_query_head_robustness", SOURCE)
robustness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(robustness)


class QueryHeadRobustnessTests(unittest.TestCase):
    def test_frozen_policy_pins_heads_frames_and_screen_without_tolerance_semantics(self):
        policy = robustness.frozen_policy()
        self.assertEqual(policy["heads"], [12, 15])
        self.assertEqual(policy["discovery_frame"], 0)
        self.assertEqual(policy["held_out_frames"], [1, 14, 29])
        self.assertEqual(policy["strict_diagonal_minimum"], 108)
        self.assertEqual(policy["positions_per_frame"], 144)
        robustness.validate_policy(policy)

        reselection = dict(policy, heads=[12, 14])
        with self.assertRaisesRegex(ValueError, "frozen policy"):
            robustness.validate_policy(reselection)
        relaxed = dict(policy, strict_diagonal_minimum=107)
        with self.assertRaisesRegex(ValueError, "frozen policy"):
            robustness.validate_policy(relaxed)

    def test_case_matrix_has_exact_frame_boundaries_and_roll_controls(self):
        cases = robustness.case_specs()
        self.assertEqual(
            [(case["name"], case["frame"], case["boundary"], case["key_roll"])
             for case in cases],
            [
                ("f00-pre", 0, "pre", 0),
                ("f01-pre", 1, "pre", 0),
                ("f01-post", 1, "post", 0),
                ("f14-pre", 14, "pre", 0),
                ("f14-post", 14, "post", 0),
                ("f29-pre", 29, "pre", 0),
                ("f29-post", 29, "post", 0),
                ("f14-pre-keyroll17", 14, "pre", 17),
                ("f14-post-keyroll17", 14, "post", 17),
            ],
        )
        policy = robustness.frozen_policy()
        robustness.validate_case_specs({"policy": policy, "cases": cases})

        wrong_rows = [dict(case) for case in cases]
        wrong_rows[1]["source_row_range"] = [0, 144]
        with self.assertRaisesRegex(ValueError, "case matrix"):
            robustness.validate_case_specs({"policy": policy, "cases": wrong_rows})

    def test_select_case_slices_the_named_frame_and_rolls_only_keys(self):
        shape = (1, 4320, 16, 128)
        tensor_bytes = int(np.prod(shape)) * 2
        q_raw = bytes(tensor_bytes)
        k_raw = bytearray(tensor_bytes)
        frame = 14
        frame_offset = frame * 144 * 16 * 128 * 2
        position_bytes = 16 * 128 * 2
        for position in range(144):
            # Distinct, finite BF16 values make the key-position roll observable.
            bf16_word = (0x3F80 + position).to_bytes(2, "little")
            start = frame_offset + position * position_bytes
            k_raw[start:start + position_bytes] = bf16_word * (position_bytes // 2)

        spec = next(case for case in robustness.case_specs()
                    if case["frame"] == 14 and case["boundary"] == "post"
                    and case["key_roll"] == 17)
        query, key = robustness.select_case(
            {"img.q_rope": q_raw, "img.k_rope": bytes(k_raw)}, spec)

        self.assertEqual(query.shape, (144, 16, 128))
        self.assertEqual(key.shape, (144, 16, 128))
        self.assertEqual(query.dtype, np.dtype("<f4"))
        source_words = np.frombuffer(k_raw, dtype="<u2").reshape(shape)
        expected_words = np.roll(source_words[0, frame * 144:(frame + 1) * 144], 17, axis=0)
        expected_key = (expected_words.astype("<u4") << 16).view("<f4")
        np.testing.assert_array_equal(key, expected_key)
        self.assertTrue(np.all(query == 0))

    def test_case_binding_rejects_forged_provenance_or_submitted_values(self):
        spec = robustness.case_specs()[0]
        query = np.zeros((144, 16, 128), dtype="<f4")
        key = np.zeros_like(query)
        provenance = {
            "anchor_manifest_sha256": "a" * 64,
            "source_receipt_sha256": "b" * 64,
            "source_artifact_sha256": "c" * 64,
            "frame": 0,
            "boundary": "pre",
        }
        admitted = {"document": {"provenance": provenance},
                    "query": query.copy(), "key": key.copy()}
        robustness.verify_case_binding(admitted, spec, query, key, provenance)

        wrong_provenance = dict(provenance, source_receipt_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "provenance"):
            robustness.verify_case_binding(admitted, spec, query, key, wrong_provenance)

        forged = {**admitted, "query": query.copy()}
        forged["query"][0, 0, 0] = np.float32(1)
        with self.assertRaisesRegex(ValueError, "authenticated source"):
            robustness.verify_case_binding(forged, spec, query, key, provenance)

    def test_score_summary_counts_each_frozen_head_independently(self):
        metrics = np.zeros((16, 144, 6), dtype="<f4")
        metrics[12, :108, 2] = 1.0
        metrics[15, :107, 2] = 1.0
        summary = robustness.score_summary(metrics)
        self.assertEqual(summary["12"]["strict_diagonal_count"], 108)
        self.assertTrue(summary["12"]["screen_passed"])
        self.assertEqual(summary["15"]["strict_diagonal_count"], 107)
        self.assertFalse(summary["15"]["screen_passed"])
        self.assertAlmostEqual(summary["12"]["fraction"], 0.75)

    def test_roll_score_columns_are_bit_exact_and_tied_argmax_is_recomputed(self):
        base = np.zeros((16, 144, 144), dtype="<f4")
        base[12, 0, 0] = 2.0
        base[12, 0, 140] = 2.0
        rolled = np.roll(base, 17, axis=-1)
        metrics = np.zeros((16, 144, 6), dtype="<f4")
        metrics[..., 4] = np.argmax(rolled, axis=-1)
        result = robustness.check_roll(base, rolled, metrics)
        self.assertTrue(result["passed"])
        self.assertTrue(result["scores_exact"])
        self.assertTrue(result["stable_argmax_exact"])
        # The tied maxima move from {0, 140} to {17, 13}; lower-index wins at 13.
        self.assertEqual(int(metrics[12, 0, 4]), 13)
        self.assertNotEqual(int(metrics[12, 0, 4]), (0 + 17) % 144)

    def test_roll_rejects_even_small_score_column_drift(self):
        base = np.zeros((16, 144, 144), dtype="<f4")
        rolled = np.roll(base, 17, axis=-1)
        rolled[0, 0, 0] = np.nextafter(np.float32(0), np.float32(1))
        metrics = np.zeros((16, 144, 6), dtype="<f4")
        metrics[..., 4] = np.argmax(rolled, axis=-1)
        result = robustness.check_roll(base, rolled, metrics)
        self.assertFalse(result["passed"])
        self.assertFalse(result["scores_exact"])

    def test_tied_argmax_cannot_be_approved_by_naive_index_shift(self):
        metrics = np.zeros((16, 144, 6), dtype="<f4")
        base = np.zeros((16, 144, 144), dtype="<f4")
        base[12, 0, 0] = 2.0
        base[12, 0, 140] = 2.0
        rolled = np.roll(base, 17, axis=-1)
        metrics[..., 4] = np.argmax(rolled, axis=-1)
        metrics[12, 0, 4] = 17  # Incorrectly shift the stable base winner at column 0.

        result = robustness.check_roll(base, rolled, metrics)
        self.assertTrue(result["scores_exact"])
        self.assertFalse(result["stable_argmax_exact"])
        self.assertFalse(result["passed"])
        self.assertEqual(int(np.argmax(rolled[12, 0])), 13)
        self.assertEqual(int(metrics[12, 0, 4]), 17)


if __name__ == "__main__":
    unittest.main()
