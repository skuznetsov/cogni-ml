"""CPU semantic checks for the bounded PixelUMM local-attention adapter."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_pixelumm_reference as reference


UPSTREAM = Path(os.environ.get(
    "PIXELUMM_UPSTREAM",
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024",
))


class PixelUMMReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, _ = reference.load_upstream(UPSTREAM)

    def test_upstream_permutation_is_bijective_and_tile_major(self):
        grid = (2, 32, 32)
        plan, inverse, _ = reference.make_attention_plan(self.source, grid)
        permutation = torch.tensor(plan.permutation, dtype=torch.long)
        self.assertEqual(permutation.numel(), 2 * 32 * 32)
        self.assertTrue(torch.equal(inverse.index_select(0, permutation), torch.arange(permutation.numel())))
        coordinates = []
        for original in plan.permutation:
            h = (original // grid[2]) % grid[1]
            w = original % grid[2]
            coordinates.append((h // 8, w // 8))
        self.assertEqual(coordinates, sorted(coordinates))
        for region in plan.regions:
            positions = plan.permutation[region.query_start:region.query_stop]
            times_and_pixels = []
            for original in positions:
                t = original // (grid[1] * grid[2])
                h = (original // grid[2]) % grid[1]
                w = original % grid[2]
                times_and_pixels.append((t, h % 8, w % 8))
            self.assertEqual(times_and_pixels, sorted(times_and_pixels))

    def test_spatial_edges_shift_inward_and_all_times_remain_visible(self):
        grid = (2, 32, 32)
        plan, _, _ = reference.make_attention_plan(self.source, grid)
        first, last = plan.regions[0], plan.regions[-1]
        self.assertEqual((first.tile_h, first.tile_w), (0, 0))
        self.assertEqual((last.tile_h, last.tile_w), (3, 3))
        # Border centers are clamped to tile 1 or tile 2, yielding a 3x3
        # neighborhood. The temporal coordinate is deliberately absent from
        # the source local mask, so both frames are represented for every tile.
        self.assertEqual(len(first.key_positions), 2 * 3 * 3 * 8 * 8)
        self.assertEqual(len(last.key_positions), 2 * 3 * 3 * 8 * 8)
        for position in first.key_positions:
            original = plan.permutation[position]
            t = original // (grid[1] * grid[2])
            h = (original // grid[2]) % grid[1]
            w = original % grid[2]
            self.assertIn(t, (0, 1))
            self.assertLessEqual(abs(1 - h // 8), 1)
            self.assertLessEqual(abs(1 - w // 8), 1)
        observed_times = {
            plan.permutation[position] // (grid[1] * grid[2])
            for position in first.key_positions
        }
        self.assertEqual(observed_times, {0, 1})

    def test_gathered_attention_matches_independent_dense_masked_oracle(self):
        grid = (2, 16, 32)
        plan, _, _ = reference.make_attention_plan(self.source, grid)
        generator = torch.Generator().manual_seed(821)
        batch, count, q_heads, kv_heads, dim = 1, 2 * 16 * 32, 4, 2, 8
        q = torch.randn((batch, count, q_heads, dim), generator=generator)
        k = torch.randn((batch, count, kv_heads, dim), generator=generator)
        v = torch.randn((batch, count, kv_heads, dim), generator=generator)

        actual = reference.gathered_local_attention(q, k, v, plan)
        keys = k.repeat_interleave(q_heads // kv_heads, dim=2).transpose(1, 2)
        values = v.repeat_interleave(q_heads // kv_heads, dim=2).transpose(1, 2)
        queries = q.transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * (dim ** -0.5)
        mask = torch.zeros((count, count), dtype=torch.bool)
        for region in plan.regions:
            mask[region.query_start:region.query_stop, list(region.key_positions)] = True
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        expected = torch.matmul(torch.softmax(scores, dim=-1), values).transpose(1, 2)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_bf16_attention_math_is_not_recaptured_by_outer_autocast(self):
        grid = (2, 16, 32)
        plan, _, _ = reference.make_attention_plan(self.source, grid)
        generator = torch.Generator().manual_seed(192)
        count = 2 * 16 * 32
        q = torch.randn((1, count, 4, 8), generator=generator).to(torch.bfloat16)
        k = torch.randn((1, count, 2, 8), generator=generator).to(torch.bfloat16)
        v = torch.randn((1, count, 2, 8), generator=generator).to(torch.bfloat16)
        observed = []
        original_matmul = torch.matmul

        def recording_matmul(left, right):
            observed.append((left.dtype, right.dtype))
            return original_matmul(left, right)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with mock.patch("torch.matmul", side_effect=recording_matmul):
                output = reference.gathered_local_attention(q, k, v, plan)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(observed)
        self.assertEqual(set(observed), {(torch.float32, torch.float32)})

    def test_input_norm_diagnostics_match_source_and_preserve_forward(self):
        grid = (1, 16, 32)
        plan, _, _ = reference.make_attention_plan(self.source, grid)
        arch = self.source.GenExpertArch(
            hidden_size=16, intermediate_size=32, num_layers=1,
            num_heads=4, num_kv_heads=2, head_dim=8,
        )
        layer = self.source.GenLayer(arch).eval().to(dtype=torch.bfloat16)
        generator = torch.Generator().manual_seed(7821)
        count = 1 * 16 * 32
        permutation = torch.tensor(plan.permutation, dtype=torch.long)
        x = torch.randn((1, count, arch.hidden_size), generator=generator)
        x = x.to(torch.bfloat16).index_select(1, permutation)
        cos_original, sin_original = self.source.mrope_tables(arch, grid, "cpu")
        cos = cos_original.index_select(1, permutation)
        sin = sin_original.index_select(1, permutation)
        diagnostic_stages = (
            "input_norm_statistics", "input_norm_pre_gain", "input_norm",
            "q_proj", "k_proj", "q_norm", "k_norm", "q_rope", "k_rope",
            "q_norm_statistics", "k_norm_statistics", "q_norm_inverse", "k_norm_inverse",
            "q_ready", "k_ready",
        )

        with torch.inference_mode(), torch.autocast(
            device_type="cpu", dtype=torch.bfloat16
        ):
            baseline, _ = reference.run_layer_reference(layer, x, cos, sin, plan)
            observed, captures = reference.run_layer_reference(
                layer, x, cos, sin, plan, capture_stages=diagnostic_stages
            )
            x_f32 = x.float()
            expected_statistics = (
                x_f32.pow(2).mean(-1, keepdim=True) + layer.input_norm.eps
            )
            expected_pre_gain = (
                x_f32 * torch.rsqrt(expected_statistics)
            ).to(dtype=x.dtype)
            original_norm = layer.input_norm(x)

        expected_q_raw = captures["q_proj"].float()
        expected_q_stats = torch.cat((
            expected_q_raw[..., :4].pow(2).mean(-1, keepdim=True)
            + layer.attn.q_norm_t.eps,
            expected_q_raw[..., 4:].pow(2).mean(-1, keepdim=True)
            + layer.attn.q_norm_hw.eps,
        ), dim=-1)
        expected_k_raw = captures["k_proj"].float()
        expected_k_stats = torch.cat((
            expected_k_raw[..., :4].pow(2).mean(-1, keepdim=True)
            + layer.attn.k_norm_t.eps,
            expected_k_raw[..., 4:].pow(2).mean(-1, keepdim=True)
            + layer.attn.k_norm_hw.eps,
        ), dim=-1)

        self.assertEqual(tuple(captures["input_norm_statistics"].shape), (1, count, 1))
        self.assertEqual(captures["input_norm_statistics"].dtype, torch.float32)
        self.assertEqual(
            tuple(captures["input_norm_pre_gain"].shape),
            (1, count, arch.hidden_size),
        )
        self.assertEqual(captures["input_norm_pre_gain"].dtype, torch.bfloat16)
        self.assertEqual(tuple(captures["q_ready"].shape), (1, count, 4, 8))
        self.assertEqual(tuple(captures["k_ready"].shape), (1, count, 2, 8))
        self.assertEqual(captures["q_ready"].dtype, torch.bfloat16)
        self.assertEqual(captures["k_ready"].dtype, torch.bfloat16)
        for name, heads in (("q", 4), ("k", 2)):
            for suffix in ("statistics", "inverse"):
                observed_diagnostic = captures[f"{name}_norm_{suffix}"]
                self.assertEqual(tuple(observed_diagnostic.shape), (1, count, heads, 2))
                self.assertEqual(observed_diagnostic.dtype, torch.float32)
        expected_q_inverse = torch.rsqrt(expected_q_stats)
        expected_k_inverse = torch.rsqrt(expected_k_stats)
        torch.testing.assert_close(
            captures["q_norm_statistics"], expected_q_stats, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            captures["k_norm_statistics"], expected_k_stats, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            captures["q_norm_inverse"], expected_q_inverse, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            captures["k_norm_inverse"], expected_k_inverse, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            captures["input_norm_statistics"], expected_statistics, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            captures["input_norm_pre_gain"], expected_pre_gain, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            captures["input_norm"], original_norm, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            captures["q_ready"], captures["q_rope"].to(torch.bfloat16),
            atol=0.0, rtol=0.0,
        )
        torch.testing.assert_close(
            captures["k_ready"], captures["k_rope"].to(torch.bfloat16),
            atol=0.0, rtol=0.0,
        )
        torch.testing.assert_close(observed, baseline, atol=0.0, rtol=0.0)

    def test_whole_reduced_layer_matches_upstream_flex(self):
        # J=2048 mirrors the small semantic qualifier; reduced widths keep its
        # eager CPU FlexAttention dense score matrix bounded.
        grid = (2, 32, 32)
        plan, _, block_mask = reference.make_attention_plan(self.source, grid)
        arch = self.source.GenExpertArch(
            hidden_size=16, intermediate_size=32, num_layers=1,
            num_heads=4, num_kv_heads=2, head_dim=8,
        )
        layer = self.source.GenLayer(arch).eval()
        generator = torch.Generator().manual_seed(20261026)
        count = 2 * 32 * 32
        x_original = torch.randn((1, count, arch.hidden_size), generator=generator)
        permutation = torch.tensor(plan.permutation, dtype=torch.long)
        x = x_original.index_select(1, permutation)
        cos_original, sin_original = self.source.mrope_tables(arch, grid, "cpu")
        cos = cos_original.index_select(1, permutation)
        sin = sin_original.index_select(1, permutation)
        with torch.inference_mode():
            expected = layer(x, cos, sin, block_mask)
            actual, captures = reference.run_layer_reference(
                layer, x, cos, sin, plan, capture_stages=reference.ALL_STAGES
            )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(tuple(captures["q_norm"].shape), (1, count, 4, 8))
        self.assertEqual(captures["q_norm"].dtype, torch.float32)
        self.assertEqual(captures["q_rope"].dtype, torch.float32)
        self.assertEqual(captures["attention"].dtype, torch.float32)
        self.assertEqual(tuple(captures["output"].shape), (1, count, 16))

    def test_global_attention_geometries_are_rejected_before_scoring(self):
        with self.assertRaises(reference.ReferenceError):
            reference.make_attention_plan(self.source, (2, 24, 24))
        for grid in ((0, 32, 32), (2, -1, 32), (2, 32), (True, 32, 32)):
            with self.assertRaises(reference.ReferenceError):
                reference.make_attention_plan(self.source, grid)

    def test_full_target_uses_bounded_formula_without_flex_blockmask(self):
        with mock.patch.object(
            self.source, "local_attention_layout",
            side_effect=AssertionError("full target must not construct a Flex BlockMask"),
        ):
            plan, _, block_mask = reference.make_attention_plan(self.source, (6, 44, 80))
        self.assertIsNone(block_mask)
        self.assertEqual(len(plan.regions), 60)
        self.assertEqual(max(r.query_stop - r.query_start for r in plan.regions), 384)
        self.assertEqual(max(len(r.key_positions) for r in plan.regions), 3456)

    def test_two_block_full_reference_peak_estimate_fits_root_oracle_cap(self):
        peak = reference._estimated_peak_bytes(
            (6, 44, 80), self.source.QWEN3_8B_GEN, layers=2
        )
        self.assertEqual(peak, 6_399_526_912)
        self.assertLess(peak, int(9.5 * 1024**3))

    def test_fixture_binary_records_are_flat_contiguous_and_hash_bound(self):
        tensors = {
            "inputs": {"x": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)},
            "weights": {"w": torch.tensor([[1.0, -2.0]], dtype=torch.float32)},
            "captures": {"y": torch.tensor([0.5, 1.5], dtype=torch.bfloat16)},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            manifest = reference._write_fixture(
                path, metadata={"token_order": "tile_major"}, **tensors
            )
            payload = path.with_suffix(".bin").read_bytes()
            records = [record for group in ("inputs", "weights", "captures")
                       for record in manifest[group].values()]
            self.assertEqual(manifest["schema"], reference.SCHEMA)
            self.assertEqual(manifest["payload"], "fixture.bin")
            self.assertEqual(manifest["payload_bytes"], len(payload))
            self.assertEqual(manifest["payload_sha256"], hashlib.sha256(payload).hexdigest())
            cursor = 0
            for record in records:
                self.assertEqual(record["offset"], cursor)
                chunk = payload[cursor:cursor + record["nbytes"]]
                self.assertEqual(len(chunk), record["nbytes"])
                self.assertEqual(record["sha256"], hashlib.sha256(chunk).hexdigest())
                cursor += len(chunk)
            self.assertEqual(cursor, len(payload))


if __name__ == "__main__":
    unittest.main()
