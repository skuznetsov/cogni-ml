from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import dcae_v_decoder_reference as reference  # noqa: E402


class DCAEVDecoderReferenceContractTest(unittest.TestCase):
    def test_only_admitted_full_tiles_have_a_cache_oracle(self):
        self.assertEqual(
            reference.validate_latent_shape(reference.SMOKE_LATENT_SHAPE),
            reference.SMOKE_LATENT_SHAPE,
        )
        self.assertEqual(
            reference.validate_latent_shape(reference.FULL_LATENT_SHAPE),
            reference.FULL_LATENT_SHAPE,
        )
        with self.assertRaisesRegex(reference.ReferenceError, "whole ten-latent tiles"):
            reference.validate_latent_shape((1, 32, 21, 1, 1))
        with self.assertRaises(reference.ReferenceError):
            reference.validate_latent_shape((1, 16, 20, 1, 1))

    def test_seeded_smoke_latents_are_bf16_repeatable_and_change_across_tiles(self):
        first = reference.make_smoke_latents(torch, seed=17031)
        second = reference.make_smoke_latents(torch, seed=17031)
        self.assertEqual(tuple(first.shape), reference.SMOKE_LATENT_SHAPE)
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first[:, :, 0], first[:, :, 1]))
        self.assertFalse(torch.equal(first[:, :, :10], first[:, :, 10:]))

    def test_stage_trace_inventory_covers_both_smoke_tiles_in_execution_order(self):
        self.assertEqual(
            reference.trace_tensor_names((0, 1)),
            (
                "trace.project_in.tile0",
                "trace.stages.6.tile0",
                "trace.stages.5.tile0",
                "trace.stages.4.tile0",
                "trace.stages.3.tile0",
                "trace.stages.2.tile0",
                "trace.stages.1.tile0",
                "trace.project_out.tile0",
                "trace.project_in.tile1",
                "trace.stages.6.tile1",
                "trace.stages.5.tile1",
                "trace.stages.4.tile1",
                "trace.stages.3.tile1",
                "trace.stages.2.tile1",
                "trace.stages.1.tile1",
                "trace.project_out.tile1",
            ),
        )

    def test_full_fixture_defaults_to_output_only_stage_capture(self):
        self.assertEqual(reference.capture_policy(reference.SMOKE_LATENT_SHAPE), "stage_trace")
        self.assertEqual(reference.capture_policy(reference.FULL_LATENT_SHAPE), "output_only")


if __name__ == "__main__":
    unittest.main()
