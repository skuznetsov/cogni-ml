"""CPU-only negative tests for DRAMA-3 late fixture validation helpers."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "analyze_late_compare.py"
SPEC = importlib.util.spec_from_file_location("drama3_analyze_late_compare", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class LateFixtureAnalysisSpec(unittest.TestCase):
    def test_readback_ledger_accepts_omitted_zero_but_rejects_extra_bytes(self):
        expected = {"diagnostic": 0, "control": 4, "terminal": 318 * 52 * 4}
        self.assertEqual(
            analysis._validated_readback_ledger(
                {"control": 4, "terminal": 318 * 52 * 4}, 318
            ),
            expected,
        )
        with self.assertRaisesRegex(analysis.ValidationError, "compact metrics receipt"):
            analysis._validated_readback_ledger(
                {"diagnostic": 4, "control": 4, "terminal": 318 * 52 * 4}, 318
            )
        with self.assertRaisesRegex(analysis.ValidationError, "unknown keys"):
            analysis._validated_readback_ledger(
                {"diagnostic": 0, "control": 4, "terminal": 318 * 52 * 4, "activation": 4},
                318,
            )

    def test_expected_coverage_pins_81_boundaries_and_six_moe_layers(self):
        for arm, expected_count in (("scalar", 313), ("sequence", 319)):
            names = analysis.expected_tensor_shapes(arm)
            boundaries = {
                name
                for name in names
                if name == "backbone.hidden"
                or name.endswith(".attention_residual")
                or name.endswith(".hidden")
            }
            self.assertEqual(len(names), expected_count)
            self.assertEqual(len(boundaries), 81)
            for layer in range(34, 40):
                prefix = f"backbone.layers.{layer}.mlp."
                self.assertIn(prefix + "canonical_expert_down_outputs", names)
                self.assertIn(prefix + "shared_weighted_output", names)
            self.assertEqual("backbone.layers.34.mlp.switch_inverse_order" in names, arm == "sequence")

    def test_native_matrix_pins_q4_arm_and_affine_down_metadata(self):
        cases = analysis._replay_specs()
        self.assertEqual(len(cases), 156)
        for arm in ("scalar", "sequence"):
            for layer in range(34, 40):
                stage = f"backbone.layers.{layer}.mlp.canonical_expert_down_outputs"
                for kind in ("expert_down_native", "expert_down_mlx_q4"):
                    case = cases[f"{kind}|{arm}|{layer}|{stage}"]
                    self.assertEqual(case["operands"], [
                        (f"backbone.layers.{layer}.mlp.canonical_expert_activation_outputs", "upstream_fixture"),
                        (f"backbone.layers.{layer}.mlp.canonical_expert_ids", "upstream_fixture"),
                    ])
                    self.assertEqual(case["route_ids"], f"backbone.layers.{layer}.mlp.canonical_expert_ids")
                    self.assertEqual(case["model_tensors"], [
                        f"backbone.layers.{layer}.mlp.switch_mlp.down_proj.weight",
                        f"backbone.layers.{layer}.mlp.switch_mlp.down_proj.scales",
                        f"backbone.layers.{layer}.mlp.switch_mlp.down_proj.biases",
                    ])

    def test_route_inverse_and_canonical_ids_are_bijective(self):
        ids = np.array([[7, 2, 5], [3, 1, 8]], dtype=np.int32)
        order = np.argsort(ids.reshape(-1), kind="stable")
        inverse = np.argsort(order, kind="stable").astype(np.int32)
        switch = ids.reshape(-1)[order]
        rank = np.argsort(ids, axis=1, kind="stable").astype(np.int32)
        canonical = np.take_along_axis(ids, rank, axis=1)
        got = analysis.validate_route_arrays(
            expert_ids=ids,
            switch_ids=switch,
            canonical_rank_order=rank,
            canonical_expert_ids=canonical,
            canonical_route_ids_from_switch=canonical.copy(),
            switch_inverse_order=inverse,
            number_of_experts=16,
        )
        self.assertEqual(got, {"rows": 2, "top_k": 3, "experts": 16})

    def test_route_guard_rejects_nonpermutation_or_bad_canonicalization(self):
        ids = np.array([[7, 2, 5], [3, 1, 8]], dtype=np.int32)
        order = np.argsort(ids.reshape(-1), kind="stable")
        inverse = np.argsort(order, kind="stable").astype(np.int32)
        switch = ids.reshape(-1)[order]
        rank = np.argsort(ids, axis=1, kind="stable").astype(np.int32)
        canonical = np.take_along_axis(ids, rank, axis=1)
        values = dict(
            expert_ids=ids,
            switch_ids=switch,
            canonical_rank_order=rank,
            canonical_expert_ids=canonical,
            canonical_route_ids_from_switch=canonical,
            switch_inverse_order=inverse,
            number_of_experts=16,
        )
        bad_inverse = dict(values, switch_inverse_order=np.zeros(6, dtype=np.int32))
        with self.assertRaisesRegex(analysis.ValidationError, "not a bijection"):
            analysis.validate_route_arrays(**bad_inverse)
        bad_ids = dict(values, canonical_route_ids_from_switch=canonical[:, ::-1])
        with self.assertRaisesRegex(analysis.ValidationError, "canonical_route_ids_from_switch"):
            analysis.validate_route_arrays(**bad_ids)

    def test_route_guard_rejects_equal_id_tie_permutation(self):
        ids = np.array([[2, 1, 3], [1, 4, 2]], dtype=np.int32)
        stable_order = np.argsort(ids.reshape(-1), kind="stable")
        wrong_order = stable_order.copy()
        wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
        switch = ids.reshape(-1)[wrong_order]
        inverse = np.argsort(wrong_order, kind="stable").astype(np.int32)
        rank = np.argsort(ids, axis=1, kind="stable").astype(np.int32)
        canonical = np.take_along_axis(ids, rank, axis=1)

        # This is still a bijection and it still restores the same expert IDs;
        # the invalid tie swap would silently attach a different token's output.
        self.assertTrue(np.array_equal(switch[inverse].reshape(ids.shape), ids))
        with self.assertRaisesRegex(analysis.ValidationError, "upstream stable expert sort"):
            analysis.validate_route_arrays(
                expert_ids=ids,
                switch_ids=switch,
                canonical_rank_order=rank,
                canonical_expert_ids=canonical,
                canonical_route_ids_from_switch=canonical.copy(),
                switch_inverse_order=inverse,
                number_of_experts=8,
            )

    def test_scalar_unflattened_routes_need_no_inverse(self):
        ids = np.array([[5, 1], [4, 0]], dtype=np.int32)
        rank = np.argsort(ids, axis=1, kind="stable").astype(np.int32)
        canonical = np.take_along_axis(ids, rank, axis=1)
        analysis.validate_route_arrays(
            expert_ids=ids,
            switch_ids=ids.copy(),
            canonical_rank_order=rank,
            canonical_expert_ids=canonical,
            canonical_route_ids_from_switch=canonical.copy(),
            switch_inverse_order=None,
            number_of_experts=8,
        )

    def test_canonical_float_views_are_exact_permutations_of_official_returns(self):
        ids = np.array([[7, 2, 5], [3, 1, 8]], dtype=np.int32)
        rank = np.argsort(ids, axis=1, kind="stable").astype(np.int32)
        original = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        original[0, 0, 0] = 0.0
        canonical = np.take_along_axis(original, np.broadcast_to(rank[..., None], original.shape), axis=1)

        # The sequence SwitchGLU projection and activation callbacks preserve
        # sorted route order; inverse restores original token/rank slots.
        order = np.argsort(ids.reshape(-1), kind="stable")
        inverse = np.argsort(order, kind="stable").astype(np.int32)
        sorted_values = original.reshape(6, 4)[order]
        analysis.validate_canonical_route_values(
            raw_values=sorted_values.reshape(6, 1, 4), canonical_values=canonical,
            rank_order=rank, inverse_order=inverse, rows=2, top_k=3,
        )

        # The scalar arm returns original route order and does not need an inverse.
        analysis.validate_canonical_route_values(
            raw_values=original[:, :, None, :], canonical_values=canonical,
            rank_order=rank, inverse_order=None, rows=2, top_k=3,
        )
        changed = canonical.copy()
        changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(analysis.ValidationError, "exact official-value permutation"):
            analysis.validate_canonical_route_values(
                raw_values=sorted_values.reshape(6, 1, 4), canonical_values=changed,
                rank_order=rank, inverse_order=inverse, rows=2, top_k=3,
            )

        # Numeric equality would treat signed zero as equal; payload equality must not.
        signed_zero = canonical.copy()
        signed_zero.view(np.uint32)[0, 0, 0] ^= np.uint32(0x80000000)
        with self.assertRaisesRegex(analysis.ValidationError, "exact official-value permutation"):
            analysis.validate_canonical_route_values(
                raw_values=sorted_values.reshape(6, 1, 4), canonical_values=signed_zero,
                rank_order=rank, inverse_order=inverse, rows=2, top_k=3,
            )

    def test_tensor_record_checks_shape_hash_and_finiteness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = np.array([[1.0, -2.0]], dtype="<f4").tobytes()
            (root / "x.f32").write_bytes(payload)
            record = {
                "name": "x",
                "file": "x.f32",
                "shape": [1, 2],
                "dtype": "float32",
                "bytes": len(payload),
                "sha256": analysis.hashlib.sha256(payload).hexdigest(),
            }
            analysis._validate_tensor_record(
                root, record, expected_name="x", expected_shape=(1, 2), expected_dtype="float32"
            )
            with self.assertRaisesRegex(analysis.ValidationError, "SHA-256 mismatch"):
                analysis._validate_tensor_record(
                    root,
                    dict(record, sha256="0" * 64),
                    expected_name="x",
                    expected_shape=(1, 2),
                    expected_dtype="float32",
                )
            nonfinite = np.array([[np.nan, 2.0]], dtype="<f4").tobytes()
            (root / "x.f32").write_bytes(nonfinite)
            record["sha256"] = analysis.hashlib.sha256(nonfinite).hexdigest()
            with self.assertRaisesRegex(analysis.ValidationError, "NaN or infinity"):
                analysis._validate_tensor_record(
                    root, record, expected_name="x", expected_shape=(1, 2), expected_dtype="float32"
                )

    def test_tensor_path_cannot_escape_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(analysis.ValidationError, "escapes fixture"):
                analysis._safe_child(root, "../outside.f32", "x")

    def test_earliest_bit_tolerance_and_nonfinite_cases_are_independent(self):
        def case(case_id, kind, arm, stage, *, bits=0, tolerance=0, nonfinite=0):
            return {
                "id": case_id, "kind": kind, "arm": arm, "stage": stage,
                "counts": {
                    "bit_mismatch_count": bits,
                    "tolerance_failure_count": tolerance,
                    "nonfinite_count": nonfinite,
                },
                "witnesses": {},
            }

        validated = [
            case("late-tolerance", "full_boundary", "scalar", "backbone.layers.1.attention_residual", tolerance=1),
            case("early-bit", "full_boundary", "scalar", "backbone.layers.0.attention_residual", bits=1),
            case("early-nonfinite", "full_boundary", "scalar", "backbone.layers.0.hidden", nonfinite=1),
            case("down-bit", "expert_down_native", "scalar", "backbone.layers.34.mlp.canonical_expert_down_outputs", bits=1),
            case("down-tolerance", "expert_down_native", "scalar", "backbone.layers.35.mlp.canonical_expert_down_outputs", tolerance=1),
        ]
        arms, replays = analysis._summarize_native_cases(validated)
        self.assertEqual(arms["scalar"]["first_boundary_bit"]["id"], "early-bit")
        self.assertEqual(arms["scalar"]["first_boundary_finite_tolerance"]["id"], "late-tolerance")
        self.assertEqual(arms["scalar"]["first_boundary_nonfinite"]["id"], "early-nonfinite")
        self.assertEqual(replays["scalar:expert_down_native"]["first_bit"]["id"], "down-bit")
        self.assertEqual(
            replays["scalar:expert_down_native"]["first_finite_tolerance"]["id"], "down-tolerance"
        )
        self.assertEqual(replays["scalar:expert_down_native"]["failing_layers_by_tolerance"], [35])


if __name__ == "__main__":
    unittest.main()
