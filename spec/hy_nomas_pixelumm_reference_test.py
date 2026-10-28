"""CPU semantic checks for the bounded PixelUMM local-attention adapter."""
from __future__ import annotations

import os
import hashlib
from io import BytesIO
from contextlib import redirect_stdout
from io import StringIO
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

    def test_all_stage_capture_budget_only_extends_to_exact_preview_geometry(self):
        self.assertEqual(reference.single_block_capture_stages((10, 18, 32), "all"), reference.ALL_STAGES)
        self.assertEqual(reference.single_block_capture_stages((10, 18, 32), "auto"), reference.FULL_TARGET_STAGES)
        self.assertEqual(reference.single_block_capture_stages((2, 32, 32), "auto"), reference.ALL_STAGES)
        for grid in ((6, 44, 80), (10, 32, 18), (10, 18, 33)):
            with self.assertRaisesRegex(reference.ReferenceError, "all-stage captures"):
                reference.single_block_capture_stages(grid, "all")
        with self.assertRaisesRegex(reference.ReferenceError, "capture mode"):
            reference.single_block_capture_stages((10, 18, 32), "unknown")

    def test_preview_oracle_rejects_a_different_mps_runtime_before_gpu_work(self):
        with mock.patch.object(reference.torch, "__version__", "2.8.0"):
            with mock.patch.object(reference.torch, "device") as device:
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(reference.ReferenceError, "requires torch 2.6.0"):
                        reference.make_fixture(
                            upstream="/unused", checkpoint="/unused",
                            output=Path(directory) / "preview.json",
                            grid=(10, 18, 32), device_name="mps",
                        )
                device.assert_not_called()
            reference.validate_preview_oracle_runtime((10, 18, 32), "cpu")
        with mock.patch.object(reference.torch, "__version__", "2.6.0"):
            reference.validate_preview_oracle_runtime((10, 18, 32), "mps")

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

    def test_four_block_full_reference_peak_estimate_fits_root_oracle_cap(self):
        peak = reference._estimated_peak_bytes(
            (6, 44, 80), self.source.QWEN3_8B_GEN, layers=4
        )
        self.assertEqual(peak, 8_555_432_960)
        self.assertLess(peak, int(9.5 * 1024**3))

    def test_four_block_schema_and_predecessor_edges(self):
        self.assertEqual(
            reference.FOUR_BLOCK_SCHEMA,
            "cogni_ml.pixelumm_four_block_reference/v1",
        )
        edges = [reference._block_io_edge(index) for index in range(4)]
        self.assertEqual(
            [edge["input_source"] for edge in edges],
            ["inputs.x", "captures.block0.output", "captures.block1.output",
             "captures.block2.output"],
        )
        self.assertEqual(
            [edge["output_key"] for edge in edges],
            [f"captures.block{index}.output" for index in range(4)],
        )

    def test_eight_block_full_reference_estimate_schema_and_predecessor_edges(self):
        base_peak = reference._estimated_peak_bytes(
            (6, 44, 80), self.source.QWEN3_8B_GEN, layers=8
        )
        full_peak = reference._estimated_peak_bytes(
            (6, 44, 80), self.source.QWEN3_8B_GEN, layers=8,
            capture_stages=reference.FULL_TARGET_STAGES,
        )
        small_all_peak = reference._estimated_peak_bytes(
            (2, 32, 32), self.source.QWEN3_8B_GEN, layers=8,
            capture_stages=reference.ALL_STAGES,
        )
        small_all_payload = reference._estimated_fixture_payload_bytes(
            (2, 32, 32), self.source.QWEN3_8B_GEN, layers=8,
            capture_stages=reference.ALL_STAGES,
        )
        full_boundary_payload = reference._estimated_fixture_payload_bytes(
            (6, 44, 80), self.source.QWEN3_8B_GEN, layers=8,
            capture_stages=reference.FULL_TARGET_STAGES,
        )
        self.assertEqual(base_peak, 12_867_245_056)
        self.assertEqual(full_peak, 15_635_485_696)
        self.assertEqual(small_all_peak, 8_957_170_688)
        self.assertEqual(small_all_payload, 6_841_110_528)
        self.assertGreater(small_all_payload, 4 * 1024**3)
        self.assertEqual(full_boundary_payload, 6_050_025_472)
        self.assertGreater(full_boundary_payload, 4 * 1024**3)
        self.assertEqual(
            reference.EIGHT_BLOCK_SCHEMA,
            "cogni_ml.pixelumm_eight_block_reference/v1",
        )
        edges = [reference._block_io_edge(index) for index in range(8)]
        self.assertEqual(
            [edge["input_source"] for edge in edges],
            ["inputs.x"] + [f"captures.block{index}.output" for index in range(7)],
        )
        self.assertEqual(
            [edge["output_key"] for edge in edges],
            [f"captures.block{index}.output" for index in range(8)],
        )

    def test_sixteen_block_schema_profiles_and_predecessor_edges(self):
        self.assertEqual(
            reference.SIXTEEN_BLOCK_SCHEMA,
            "cogni_ml.pixelumm_sixteen_block_reference/v1",
        )
        self.assertEqual(
            reference.REAL_INPUT_NOISE_OWNER,
            "hymm.models.vae.pixelumm_decoder.autoencoder."
            "PixelUMMDCAELatentDecoder._make_noise",
        )
        self.assertEqual(
            set(reference.REAL_INPUT_ARTIFACTS),
            {
                "pixelumm-real-input.json",
                "pixelumm-real-input-full-seed42.json",
                "pixelumm-real-input-full-seed43.json",
                "pixelumm-real-input-small-seed42.json",
                "pixelumm-real-input-small-seed43.json",
            },
        )
        self.assertEqual(reference.REAL_INPUT_ARTIFACTS["pixelumm-real-input.json"],
                         ("full", 42, "pixelumm-real-input.bin"))
        self.assertEqual(reference.REAL_INPUT_ARTIFACTS[
            "pixelumm-real-input-small-seed43.json"
        ], ("small", 43, "pixelumm-real-input-small-seed43.bin"))
        edges = [reference._block_io_edge(index) for index in range(16)]
        self.assertEqual(
            [edge["input_source"] for edge in edges],
            ["inputs.x"] + [f"captures.block{index}.output" for index in range(15)],
        )
        self.assertEqual(
            [edge["output_key"] for edge in edges],
            [f"captures.block{index}.output" for index in range(16)],
        )
        with self.assertRaises(reference.ReferenceError):
            reference._block_io_edge(36)

    def test_thirty_six_block_profile_is_full_grid_with_streamed_one_layer_peak(self):
        self.assertEqual(
            reference.THIRTY_SIX_BLOCK_SCHEMA,
            "cogni_ml.pixelumm_thirty_six_block_reference/v1",
        )
        self.assertEqual(reference._block_io_edge(35), {
            "input_source": "captures.block34.output",
            "output_key": "captures.block35.output",
        })
        with self.assertRaises(reference.ReferenceError):
            reference._block_io_edge(36)

        peak = reference._estimated_peak_bytes(
            (6, 44, 80), self.source.QWEN3_8B_GEN,
            layers=36, resident_layers=1,
            capture_stages=reference.FULL_TARGET_STAGES,
            full36_capture_policy="two_bank_survivors",
        )
        self.assertEqual(peak, 11_896_145_408)
        self.assertLess(peak + 194_641_920, 23 * 1024**3)
        self.assertEqual(17_778_656_768 - peak, 5_882_511_360)

        payload = reference._estimated_fixture_payload_bytes(
            (6, 44, 80), self.source.QWEN3_8B_GEN,
            layers=36, capture_stages=reference.FULL_TARGET_STAGES,
            full36_capture_policy="two_bank_survivors",
        )
        self.assertEqual(payload, 20_661_356_544)
        self.assertLess(payload, 32 * 1024**3)
        self.assertEqual(26_543_867_904 - payload, 5_882_511_360)

        for estimator in (reference._estimated_peak_bytes, reference._estimated_fixture_payload_bytes):
            kwargs = {
                "grid": (6, 44, 80),
                "arch": self.source.QWEN3_8B_GEN,
                "layers": 36,
                "capture_stages": reference.FULL_TARGET_STAGES,
            }
            if estimator is reference._estimated_peak_bytes:
                kwargs["resident_layers"] = 1
            with self.subTest(estimator=estimator.__name__, policy="missing"), self.assertRaises(
                reference.ReferenceError
            ):
                estimator(**kwargs)
            with self.subTest(estimator=estimator.__name__, policy="wrong"), self.assertRaises(
                reference.ReferenceError
            ):
                estimator(**kwargs, full36_capture_policy="all_boundaries")

        with self.assertRaises(reference.ReferenceError):
            reference._estimated_fixture_payload_bytes(
                (2, 32, 32), self.source.QWEN3_8B_GEN,
                layers=36, capture_stages=reference.FULL_TARGET_STAGES,
                full36_capture_policy="two_bank_survivors",
            )
        with self.assertRaises(reference.ReferenceError):
            reference._estimated_peak_bytes(
                (6, 44, 80), self.source.QWEN3_8B_GEN,
                layers=37, resident_layers=1,
                capture_stages=reference.FULL_TARGET_STAGES,
            )

    def test_full36_two_bank_capture_policy_is_exact_and_strict(self):
        self.assertEqual(reference.THIRTY_SIX_CAPTURE_POLICY, "two_bank_survivors")
        self.assertEqual(
            [reference._full36_capture_stages(index) for index in range(36)],
            [
                ("attention",) if index < 34 else ("attention", "output")
                for index in range(36)
            ],
        )
        expected = [
            f"block{index}.{stage}"
            for index in range(36)
            for stage in (("attention",) if index < 34 else ("attention", "output"))
        ]
        self.assertEqual(len(expected), 38)
        blocks = [
            {
                "index": index,
                "capture_keys": [
                    f"block{index}.{stage}"
                    for stage in (("attention",) if index < 34 else ("attention", "output"))
                ],
            }
            for index in range(36)
        ]
        reference._validate_full36_capture_inventory(
            policy="two_bank_survivors", capture_order=expected, blocks=blocks,
        )

        for policy in (None, "all_boundaries", "two_bank_survivors "):
            with self.subTest(policy=policy), self.assertRaises(reference.ReferenceError):
                reference._validate_full36_capture_inventory(
                    policy=policy, capture_order=expected, blocks=blocks,
                )

        negative_inventories = []
        missing_final_output = list(expected)
        missing_final_output.remove("block35.output")
        negative_inventories.append(missing_final_output)
        extra_early_output = list(expected)
        extra_early_output.insert(67, "block33.output")
        negative_inventories.append(extra_early_output)
        wrong_final_stage = list(expected)
        wrong_final_stage[-1] = "block35.down_proj"
        negative_inventories.append(wrong_final_stage)
        for capture_order in negative_inventories:
            with self.subTest(capture_order=capture_order[-3:]), self.assertRaises(
                reference.ReferenceError
            ):
                reference._validate_full36_capture_inventory(
                    policy="two_bank_survivors",
                    capture_order=capture_order, blocks=blocks,
                )

        missing_block_output = [dict(block) for block in blocks]
        missing_block_output[-1]["capture_keys"] = ["block35.attention"]
        with self.assertRaises(reference.ReferenceError):
            reference._validate_full36_capture_inventory(
                policy="two_bank_survivors", capture_order=expected,
                blocks=missing_block_output,
            )
        extra_block_stage = [dict(block) for block in blocks]
        extra_block_stage[33]["capture_keys"] = ["block33.attention", "block33.output"]
        with self.assertRaises(reference.ReferenceError):
            reference._validate_full36_capture_inventory(
                policy="two_bank_survivors", capture_order=expected,
                blocks=extra_block_stage,
            )

    def test_thirty_six_block_cli_requires_all_blocks_real_input_and_cache(self):
        argv = ["--upstream", "upstream", "--header", "header.json",
                "--output", "fixture.json", "--grid", "6", "44", "80",
                "--real-input-manifest", "/pinned/source/pixelumm-real-input.json",
                "--weight-cache-manifest", "/pinned/source/pixelumm-thirty-six-bf16-cache.json"]
        for index in range(36):
            argv += [f"--block{index}", f"block{index}.safetensors"]
            argv += [f"--receipt{index}", f"block{index}.receipt.json"]
        with mock.patch.object(
            reference, "make_thirty_six_block_fixture", return_value={"accepted": True}
        ) as make:
            with redirect_stdout(StringIO()):
                status = reference.main(argv)
        self.assertEqual(status, 0)
        self.assertEqual(len(make.call_args.kwargs["block_sources"]), 36)
        self.assertEqual(make.call_args.kwargs["block_sources"][-1],
                         (Path("block35.safetensors"), Path("block35.receipt.json")))
        self.assertEqual(
            make.call_args.kwargs["real_input_manifest"],
            Path("/pinned/source/pixelumm-real-input.json"),
        )
        self.assertEqual(
            make.call_args.kwargs["weight_cache_manifest"],
            Path("/pinned/source/pixelumm-thirty-six-bf16-cache.json"),
        )

        missing_cache = [arg for arg in argv if arg not in {
            "--weight-cache-manifest", "/pinned/source/pixelumm-thirty-six-bf16-cache.json",
        }]
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                reference.main(missing_cache)

        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                reference.main(argv + ["--block36", "block36.safetensors",
                                       "--receipt36", "block36.receipt.json"])

    def test_streamed_cache_weight_records_bind_all_468_tiny_records(self):
        names = {f"weight{index:02d}": (1,) for index in range(13)}
        schema_patch = mock.patch.object(
            reference.checkpoint_inventory, "block_schema", return_value=names
        )
        with tempfile.TemporaryDirectory() as directory, schema_patch:
            payload_path = Path(directory) / "cache.bin"
            payload = bytes(index % 256 for index in range(36 * 13 * 2))
            payload_path.write_bytes(payload)
            cache_weights = {}
            cache_blocks = {}
            cursor = 0
            for block in range(36):
                keys = [f"block{block}.{name}" for name in sorted(names)]
                cache_blocks[str(block)] = {
                    "selected_block": block,
                    "weight_keys": keys,
                }
                for key in keys:
                    chunk = payload[cursor:cursor + 2]
                    cache_weights[key] = {
                        "dtype": "BF16", "shape": [1], "offset": cursor,
                        "nbytes": 2, "sha256": hashlib.sha256(chunk).hexdigest(),
                    }
                    cursor += 2
            cache_manifest = {
                "blocks": cache_blocks,
                "weights": cache_weights,
                "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
            copied, copied_digest, records = BytesIO(), hashlib.sha256(), {}
            cursor = reference._write_cache_weight_records(
                copied, cache_manifest=cache_manifest,
                cache_payload_path=payload_path, records=records,
                payload_digest=copied_digest, start_offset=0,
            )
            self.assertEqual(cursor, len(payload))
            self.assertEqual(copied.getvalue(), payload)
            self.assertEqual(copied_digest.hexdigest(), hashlib.sha256(payload).hexdigest())
            self.assertEqual(len(records), 468)
            self.assertEqual(records["block35.weight12"]["offset"], len(payload) - 2)

            fixture_path = Path(directory) / "fixture.json"
            manifest = reference._write_fixture(
                fixture_path,
                metadata={"weight_cache_manifest_sha256": "a" * 64},
                inputs={"x": torch.tensor([1], dtype=torch.bfloat16)},
                weights={}, captures={"y": torch.tensor([2], dtype=torch.bfloat16)},
                schema=reference.THIRTY_SIX_BLOCK_SCHEMA,
                weight_cache=(cache_manifest, payload_path),
            )
            self.assertEqual(len(manifest["weights"]), 468)
            self.assertEqual(manifest["weights"]["block35.weight12"]["offset"],
                             manifest["payload_bytes"] - 4)

            payload_path.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))
            with self.assertRaises(reference.ReferenceError):
                reference._write_cache_weight_records(
                    BytesIO(), cache_manifest=cache_manifest,
                    cache_payload_path=payload_path, records={},
                    payload_digest=hashlib.sha256(), start_offset=0,
                )

    def test_loaded_cache_word_check_is_per_named_parameter_and_bitwise(self):
        names = {f"weight{index:02d}": (1,) for index in range(13)}
        words = torch.arange(13, dtype=torch.int16)
        model = torch.nn.Module()
        for index, name in enumerate(sorted(names)):
            value = words[index:index + 1].view(torch.bfloat16).clone()
            model.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
        keys = [f"block0.{name}" for name in sorted(names)]
        records = {
            key: {"dtype": "BF16", "shape": [1], "offset": index * 2, "nbytes": 2}
            for index, key in enumerate(keys)
        }
        cache_manifest = {
            "blocks": {"0": {"weight_keys": keys}},
            "weights": records,
        }
        with mock.patch.object(
            reference.checkpoint_inventory, "block_schema", return_value=names
        ):
            checks = reference._queue_cache_weight_word_checks(
                model, cache_manifest=cache_manifest, cache_words=words.clone(),
                block_index=0, device=torch.device("cpu"),
            )
            self.assertEqual(list(checks), keys)
            self.assertTrue(all(bool(check) for check in checks.values()))

            mismatched = words.clone()
            mismatched[-1] ^= 1
            failed = reference._queue_cache_weight_word_checks(
                model, cache_manifest=cache_manifest, cache_words=mismatched,
                block_index=0, device=torch.device("cpu"),
            )
            self.assertFalse(bool(failed[keys[-1]]))

            model.weight00 = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            with self.assertRaises(reference.ReferenceError):
                reference._queue_cache_weight_word_checks(
                    model, cache_manifest=cache_manifest, cache_words=words.clone(),
                    block_index=0, device=torch.device("cpu"),
                )

    def test_real_input_lineage_uses_authenticated_full_subset_file_bytes(self):
        checked_subsets = {
            "encoder": {
                "subset_bytes": 3_666_640_744,
                "source_f32_bytes": 3_666_620_032,
            },
            "stem": {
                "subset_bytes": 52_511_088,
                "source_f32_bytes": 52_510_720,
            },
        }
        lineage = {
            "encoder_subset_bytes": checked_subsets["encoder"]["subset_bytes"],
            "stem_subset_bytes": checked_subsets["stem"]["subset_bytes"],
        }
        reference._validate_real_input_subset_sizes(lineage, checked_subsets)

        # Producer lineage records complete safetensors file sizes, while the
        # receipt's source_f32_bytes field excludes the encoded header.
        payload_only = dict(lineage)
        payload_only["encoder_subset_bytes"] = checked_subsets["encoder"]["source_f32_bytes"]
        with self.assertRaises(reference.ReferenceError):
            reference._validate_real_input_subset_sizes(payload_only, checked_subsets)

    def test_real_input_manifest_path_must_be_canonical_and_non_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            manifest = root / "pixelumm-real-input.json"
            manifest.write_text("{}", encoding="utf-8")
            with mock.patch.object(reference, "SIXTEEN_SOURCE_DIR", root.resolve()):
                self.assertEqual(
                    reference._admitted_real_input_manifest_path(manifest.resolve()),
                    manifest.resolve(),
                )
                linked_manifest = root / "pixelumm-real-input-small-seed42.json"
                linked_manifest.symlink_to(manifest)
                with self.assertRaises(reference.ReferenceError):
                    reference._admitted_real_input_manifest_path(linked_manifest)

                linked_parent = Path(temporary) / "source-alias"
                linked_parent.symlink_to(root, target_is_directory=True)
                with self.assertRaises(reference.ReferenceError):
                    reference._admitted_real_input_manifest_path(
                        linked_parent / "pixelumm-real-input.json"
                    )

    def test_real_input_x_bf16_finiteness_guard_uses_cpu_payload_bytes(self):
        self.assertTrue(reference._bf16_bytes_are_finite(bytes.fromhex("803f00c0")))
        self.assertFalse(reference._bf16_bytes_are_finite(bytes.fromhex("807f")))
        self.assertFalse(reference._bf16_bytes_are_finite(bytes.fromhex("c17f")))
        self.assertFalse(reference._bf16_bytes_are_finite(bytes.fromhex("80")))

    def test_sixteen_block_cli_routes_only_complete_prefix_with_real_manifest(self):
        argv = ["--upstream", "upstream", "--header", "header.json",
                "--output", "fixture.json", "--grid", "2", "32", "32",
                "--real-input-manifest",
                "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261027-sixteen/"
                "source/pixelumm-real-input-small-seed42.json"]
        for index in range(16):
            argv += [f"--block{index}", f"block{index}.safetensors"]
            argv += [f"--receipt{index}", f"block{index}.receipt.json"]
        with mock.patch.object(
            reference, "make_sixteen_block_fixture", return_value={"accepted": True}
        ) as make_sixteen:
            with redirect_stdout(StringIO()):
                status = reference.main(argv)
        self.assertEqual(status, 0)
        self.assertEqual(len(make_sixteen.call_args.kwargs["block_sources"]), 16)
        self.assertEqual(
            make_sixteen.call_args.kwargs["real_input_manifest"],
            Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261027-sixteen/"
                 "source/pixelumm-real-input-small-seed42.json"),
        )

        missing_manifest = [arg for arg in argv if arg != "--real-input-manifest"
                            and "pixelumm-real-input-small-seed42.json" not in arg]
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                reference.main(missing_manifest)

        partial = argv[:-4]
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                reference.main(partial)

    def test_eight_block_cli_routes_only_a_complete_consecutive_prefix(self):
        argv = ["--upstream", "upstream", "--header", "header.json",
                "--output", "fixture.json", "--grid", "6", "44", "80"]
        for index in range(8):
            argv += [f"--block{index}", f"block{index}.safetensors"]
            argv += [f"--receipt{index}", f"block{index}.receipt.json"]
        with mock.patch.object(
            reference, "make_eight_block_fixture", return_value={"accepted": True}
        ) as make_eight:
            with redirect_stdout(StringIO()):
                status = reference.main(argv)
        self.assertEqual(status, 0)
        self.assertEqual(make_eight.call_args.kwargs["block7"], Path("block7.safetensors"))
        self.assertEqual(make_eight.call_args.kwargs["receipt7"], Path("block7.receipt.json"))

        partial = ["--upstream", "upstream", "--header", "header.json",
                   "--output", "fixture.json", "--grid", "6", "44", "80",
                   "--block0", "block0.safetensors", "--receipt0", "block0.receipt.json",
                   "--block1", "block1.safetensors", "--receipt1", "block1.receipt.json",
                   "--block4", "block4.safetensors", "--receipt4", "block4.receipt.json"]
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                reference.main(partial)

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
