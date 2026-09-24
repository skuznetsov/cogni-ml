"""Unit tests for the bounded Qwen3-VL layer-0 trace report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
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


if __name__ == "__main__":
    unittest.main()
