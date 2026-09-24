"""Unit tests for the bounded Qwen3-VL layer-0 trace report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qwen3vl_text_layer_trace.py"
SPEC = importlib.util.spec_from_file_location("qwen3vl_text_layer_trace", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


class SequenceMetricSpec(unittest.TestCase):
    def test_counts_every_row_and_separates_dropped_prefix_from_retained_rows(self) -> None:
        expected = torch.zeros((1, 24, 2), dtype=torch.bfloat16)
        actual = expected.clone()
        for row, column in ((0, 0), (13, 1), (14, 0), (23, 1)):
            actual[0, row, column] = 1

        regions = TRACE._sequence_regions([1] * 24, drop_idx=14)
        metrics = TRACE._metrics(actual, expected, query_axis=1, row_regions=regions)

        self.assertEqual(4, metrics["exact_mismatches"])
        self.assertEqual(24, len(metrics["by_row"]))
        self.assertEqual([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [row["exact_mismatches"] for row in metrics["by_row"]])
        self.assertEqual(["prefix"] * 14 + ["retained"] * 10,
                         [row["region"] for row in metrics["by_row"]])
        self.assertEqual(2, metrics["region_totals"]["prefix"]["exact_mismatches"])
        self.assertEqual(2, metrics["region_totals"]["retained"]["exact_mismatches"])
        self.assertEqual(28, metrics["region_totals"]["prefix"]["elements"])
        self.assertEqual(20, metrics["region_totals"]["retained"]["elements"])

    def test_sequence_axis_uses_token_axis_for_q_norm_and_k_norm(self) -> None:
        self.assertEqual(1, TRACE._sequence_axis("layers.0.self_attn.q_norm"))
        self.assertEqual(1, TRACE._sequence_axis("layers.0.self_attn.k_norm"))
        self.assertEqual(1, TRACE._sequence_axis("layers.0.self_attn.q_proj"))

    def test_full_and_prefix_snapshots_compare_their_common_rows(self) -> None:
        full = torch.zeros((1, 24, 3), dtype=torch.bfloat16)
        prefix = torch.zeros((1, 2, 3), dtype=torch.bfloat16)
        full[0, 1, 2] = 1

        comparisons = TRACE._compare_snapshots(
            {"layers.0.input_layernorm": full},
            {"layers.0.input_layernorm": prefix},
            row_regions=["prefix"] * 14 + ["retained"] * 10,
        )

        metric = comparisons["layers.0.input_layernorm"]
        self.assertTrue(metric["comparable"])
        self.assertEqual([0, 1], [row["exact_mismatches"] for row in metric["by_row"]])
        self.assertEqual(2, metric["compared_sequence_rows"])

    def test_native_comparison_reports_first_boundary_and_every_row(self) -> None:
        official = {}
        native = {}
        for name in TRACE.NATIVE_BOUNDARY_ORDER:
            width = TRACE.NATIVE_BOUNDARY_WIDTHS[name]
            official[name] = torch.zeros((1, 24, width), dtype=torch.bfloat16)
            native[name] = torch.zeros((1, 24, width), dtype=torch.bfloat16)
        official["layers.0.self_attn.q_norm"] = torch.zeros((1, 24, 32, 128), dtype=torch.bfloat16)
        native["layers.0.self_attn.q_norm"] = torch.zeros((1, 24, 4096), dtype=torch.bfloat16)
        official["layers.0.self_attn.q_norm"][0, 13, 31, 127] = 1
        official["layers.0.self_attn.q_norm"][0, 14, 0, 0] = 1

        result = TRACE._compare_native_stages(
            official,
            native,
            row_regions=["prefix"] * 14 + ["retained"] * 10,
        )

        self.assertEqual("layers.0.self_attn.q_norm", result["first_divergent_stage"])
        self.assertEqual(
            list(TRACE.NATIVE_BOUNDARY_ORDER[:5]),
            result["exact_prefix_before_divergence"],
        )
        metric = result["stages"]["layers.0.self_attn.q_norm"]
        self.assertEqual(24, len(metric["by_row"]))
        self.assertEqual(1, metric["by_row"][13]["exact_mismatches"])
        self.assertEqual(1, metric["by_row"][14]["exact_mismatches"])
        self.assertEqual(1, metric["region_totals"]["prefix"]["exact_mismatches"])
        self.assertEqual(1, metric["region_totals"]["retained"]["exact_mismatches"])

    def test_retained_only_embedding_snapshot_keeps_retained_row_labels(self) -> None:
        expected = torch.zeros((1, 10, 2), dtype=torch.bfloat16)
        actual = expected.clone()
        actual[0, 0, 0] = 1
        regions = ["prefix"] * 14 + ["retained"] * 10

        comparisons = TRACE._compare_snapshots(
            {"pre_final_prompt_embeddings": actual},
            {"pre_final_prompt_embeddings": expected},
            row_regions=regions,
        )

        metric = comparisons["pre_final_prompt_embeddings"]
        self.assertEqual("retained", metric["by_row"][0]["region"])
        self.assertEqual(10, metric["region_totals"]["retained"]["rows"])
        self.assertEqual(1, metric["region_totals"]["retained"]["exact_mismatches"])
        self.assertNotIn("prefix", metric["region_totals"])

    def test_native_loader_rejects_a_tampered_stage_hash(self) -> None:
        fixture_sha = TRACE.EXPECTED_FIXTURE["payload_sha256"]
        manifest = {
            "schema": TRACE.NATIVE_TRACE_SCHEMA,
            "schema_version": 1,
            "prompt": "red cube",
            "model_revision": TRACE.EXPECTED_FIXTURE["model.revision_sha"],
            "fixture_payload_sha256": fixture_sha,
            "dtype": "bfloat16-le",
            "stages": {},
        }
        for name in TRACE.NATIVE_BOUNDARY_ORDER:
            shape = [24, TRACE.NATIVE_BOUNDARY_WIDTHS[name]]
            manifest["stages"][name] = {
                "filename": f"{name}.bf16le",
                "shape": shape,
                "nbytes": shape[0] * shape[1] * 2,
                "sha256": "0" * 64,
            }

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = TRACE.NATIVE_BOUNDARY_ORDER[0]
            (root / manifest["stages"][first]["filename"]).write_bytes(
                bytes(manifest["stages"][first]["nbytes"])
            )
            (root / "trace.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "failed its length or SHA-256 check"):
                TRACE._load_native_trace(root, torch=torch, fixture_payload_sha256=fixture_sha)

    def test_equal_input_operator_comparison_reports_zero_and_one_mismatch_by_region(self) -> None:
        expected = torch.zeros((1, 24, 4), dtype=torch.bfloat16)
        regions = ["prefix"] * 14 + ["retained"] * 10

        exact = TRACE._compare_equal_input_operator(
            "q_proj", expected.clone(), expected, row_regions=regions
        )
        self.assertEqual(0, exact["exact_mismatches"])
        self.assertEqual(14, exact["region_totals"]["prefix"]["rows"])
        self.assertEqual(10, exact["region_totals"]["retained"]["rows"])

        actual = expected.clone()
        actual[0, 17, 2] = 1
        one_mismatch = TRACE._compare_equal_input_operator(
            "q_proj", actual, expected, row_regions=regions
        )
        self.assertEqual(1, one_mismatch["exact_mismatches"])
        self.assertEqual(1, one_mismatch["by_row"][17]["exact_mismatches"])
        self.assertEqual("retained", one_mismatch["by_row"][17]["region"])
        self.assertEqual(1, one_mismatch["region_totals"]["retained"]["exact_mismatches"])
        self.assertEqual(0, one_mismatch["region_totals"]["prefix"]["exact_mismatches"])

    def test_equal_input_stage_validation_rejects_shape_and_dtype_drift(self) -> None:
        correct = torch.zeros((1, 24, 4), dtype=torch.bfloat16)
        TRACE._validate_equal_input_stage("input", correct, (1, 24, 4), torch=torch)

        with self.assertRaisesRegex(ValueError, "unexpected shape"):
            TRACE._validate_equal_input_stage(
                "input", torch.zeros((1, 23, 4), dtype=torch.bfloat16), (1, 24, 4), torch=torch
            )
        with self.assertRaisesRegex(ValueError, "must be bfloat16"):
            TRACE._validate_equal_input_stage(
                "input", torch.zeros((1, 24, 4), dtype=torch.float32), (1, 24, 4), torch=torch
            )

    def test_equal_input_replay_uses_gqa_causal_sdpa_math_from_native_inputs(self) -> None:
        q_proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        k_proj = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
        v_proj = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
        o_proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            q_proj.weight.copy_(torch.eye(8, dtype=torch.bfloat16))
            o_proj.weight.copy_(torch.eye(8, dtype=torch.bfloat16))
        attention = SimpleNamespace(
            num_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            scaling=2**-0.5,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
        )
        native = {
            "layers.0.input_layernorm": torch.tensor(
                [[list(range(1, 9)), list(range(9, 17))]], dtype=torch.bfloat16
            ),
            "layers.0.self_attn.q_proj": torch.tensor(
                [[list(range(1, 9)), list(range(9, 17))]], dtype=torch.bfloat16
            ),
            "post_rope_q": torch.zeros((1, 2, 8), dtype=torch.bfloat16),
            "post_rope_k": torch.zeros((1, 2, 4), dtype=torch.bfloat16),
            "layers.0.self_attn.v_proj": torch.tensor(
                [[[1, 0, 0, 2], [0, 1, 2, 0]]], dtype=torch.bfloat16
            ),
            # Zero Q/K makes causal row 0 select V0 and row 1 average V0,V1.
            # Query heads 0/1 share KV head 0; heads 2/3 share KV head 1.
            "attended": torch.tensor(
                [[[1, 0, 1, 0, 0, 2, 0, 2], [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]]],
                dtype=torch.bfloat16,
            ),
            "layers.0.self_attn.o_proj": torch.tensor(
                [[[1, 0, 1, 0, 0, 2, 0, 2], [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]]],
                dtype=torch.bfloat16,
            ),
        }

        replay = TRACE._equal_input_operator_replay(
            attention,
            native,
            attention_mask=torch.ones((1, 2), dtype=torch.int64),
            row_regions=["prefix", "retained"],
            torch=torch,
        )

        self.assertEqual("operator-local equal-input replay", replay["scope"])
        self.assertFalse(replay["full_chain_parity"])
        self.assertFalse(replay["image_quality"])
        self.assertEqual(4, replay["attention"]["query_heads"])
        self.assertEqual(2, replay["attention"]["key_value_heads"])
        self.assertEqual(2, replay["attention"]["kv_repeat_interleave"])
        self.assertEqual("MATH", replay["attention"]["backend"])
        self.assertIsNone(replay["attention"]["attn_mask"])
        self.assertEqual(
            {name: 0 for name in replay["operators"]},
            {
                name: operator["metrics"]["exact_mismatches"]
                for name, operator in replay["operators"].items()
            },
        )

        changed_native = {name: tensor.clone() for name, tensor in native.items()}
        changed_native["layers.0.self_attn.q_proj"][0, 1, 0] = 17
        one_mismatch = TRACE._equal_input_operator_replay(
            attention,
            changed_native,
            attention_mask=torch.ones((1, 2), dtype=torch.int64),
            row_regions=["prefix", "retained"],
            torch=torch,
        )
        self.assertEqual(1, one_mismatch["operators"]["q_proj"]["metrics"]["exact_mismatches"])
        self.assertEqual(
            "retained",
            one_mismatch["operators"]["q_proj"]["metrics"]["by_row"][1]["region"],
        )

        with self.assertRaisesRegex(ValueError, "all-visible attention mask"):
            TRACE._equal_input_operator_replay(
                attention,
                native,
                attention_mask=torch.tensor([[1, 0]], dtype=torch.int64),
                row_regions=["prefix", "retained"],
                torch=torch,
            )


if __name__ == "__main__":
    unittest.main()
