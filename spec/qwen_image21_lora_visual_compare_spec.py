from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_image21_lora_visual_compare as MODULE  # noqa: E402


class AdapterLayer(nn.Module):
    def __init__(self, zero_b: bool = True):
        super().__init__()
        self.disable_adapters = False
        self.lora_A = nn.ModuleDict({"grad_probe": nn.Linear(4096, 4, bias=False)})
        self.lora_B = nn.ModuleDict({"grad_probe": nn.Linear(4, 4096, bias=False)})
        if zero_b:
            nn.init.zeros_(self.lora_B["grad_probe"].weight)


class TinyAttention(nn.Module):
    def __init__(self, zero_b: bool = True):
        super().__init__()
        self.to_q = AdapterLayer(zero_b=zero_b)


class TinyBlock(nn.Module):
    def __init__(self, zero_b: bool = True):
        super().__init__()
        self.attn = TinyAttention(zero_b=zero_b)


class TinyTransformer(nn.Module):
    def __init__(self, zero_b: bool = True):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(31)])
        self.transformer_blocks.append(TinyBlock(zero_b=zero_b))
        self.disable_calls = 0
        self.enable_calls = 0

    def disable_adapters(self):
        self.disable_calls += 1
        for module in self.modules():
            if isinstance(module, AdapterLayer):
                module.disable_adapters = True

    def enable_adapters(self):
        self.enable_calls += 1
        for module in self.modules():
            if isinstance(module, AdapterLayer):
                module.disable_adapters = False


class TinyBundle:
    def __init__(self, offset: float = 0.0):
        self.initial_target_latents = (
            torch.full((1, 256, 64), 2.0, dtype=torch.float32) + offset
        )
        self.encoder_hidden_states = torch.zeros((1, 10, 4096), dtype=torch.float32)
        self.encoder_hidden_states_mask = torch.ones((1, 10), dtype=torch.bool)
        self.encoder_img_mask = torch.zeros((1, 10), dtype=torch.bool)
        self.img_shapes = [[(1, 16, 16)]]
        self.prompt = "red cube" if offset == 0 else "castle"


class QwenImage21LoraVisualCompareSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = MODULE.probe.load_flow_match_schedule(
            ROOT / "spec/fixtures/qwen_image21_flow_match_diffusers.json"
        )

    def test_routes_use_full_four_step_baseline_and_coarse_then_base_hybrid(self):
        self.assertEqual((0, 1, 2, 3), MODULE.BASE4_STEP_INDICES)
        self.assertEqual((0, 2, 3), MODULE.HYBRID3_STEP_INDICES)
        MODULE._validate_schedule_contract(self.schedule)
        self.assertEqual(
            self.schedule.sigmas[2] - self.schedule.sigmas[0],
            MODULE._route_sigma_delta(self.schedule, 0, "coarse"),
        )

    def test_base4_uses_every_pinned_timestep_and_fp32_endpoint_accumulation(self):
        model = TinyTransformer()
        model.eval()
        bundle = TinyBundle()
        calls = []

        def velocity(_model, _bundle, state, timestep, _device, _dtype):
            self.assertTrue(torch.is_inference_mode_enabled())
            calls.append((timestep, state.dtype))
            return torch.full_like(state, 0.25, dtype=torch.bfloat16)

        with patch.object(MODULE.probe, "_forward_velocity", side_effect=velocity):
            result = MODULE.sample_base4(
                model,
                bundle,
                self.schedule,
                device=torch.device("cpu"),
                model_dtype=torch.float32,
            )

        self.assertEqual(list(self.schedule.timesteps), [item[0] for item in calls])
        self.assertTrue(all(dtype == torch.float32 for _, dtype in calls))
        self.assertEqual(torch.float32, result.final_latents.dtype)
        self.assertEqual(
            [0, 1, 2, 3], [step["schedule_index"] for step in result.steps]
        )
        expected = 2.0 + 0.25 * (self.schedule.sigmas[-1] - self.schedule.sigmas[0])
        self.assertTrue(
            torch.allclose(
                result.final_latents, torch.full_like(result.final_latents, expected)
            )
        )

    def test_hybrid3_disables_adapter_only_after_coarse_first_step_and_restores_it(
        self,
    ):
        model = TinyTransformer()
        model.eval()
        bundle = TinyBundle()
        calls = []

        def velocity(current_model, _bundle, state, timestep, _device, _dtype):
            layer = dict(current_model.named_modules())[MODULE.probe.TARGET_MODULE]
            enabled = not layer.disable_adapters
            calls.append((timestep, enabled, state.dtype))
            return torch.full_like(state, 1.0 if enabled else 0.5)

        with patch.object(MODULE.probe, "_forward_velocity", side_effect=velocity):
            result = MODULE.sample_hybrid3(
                model,
                bundle,
                self.schedule,
                device=torch.device("cpu"),
                model_dtype=torch.float32,
                adapter_label="trained",
            )

        self.assertEqual(
            [self.schedule.timesteps[index] for index in (0, 2, 3)],
            [item[0] for item in calls],
        )
        self.assertEqual([True, False, False], [item[1] for item in calls])
        self.assertTrue(all(item[2] == torch.float32 for item in calls))
        self.assertEqual(
            [True, False, False], [item["adapter_enabled"] for item in result.steps]
        )
        self.assertEqual(
            (0, 2, 3), tuple(item["schedule_index"] for item in result.steps)
        )
        self.assertEqual(1, model.disable_calls)
        self.assertEqual(1, model.enable_calls)
        self.assertFalse(
            dict(model.named_modules())[MODULE.probe.TARGET_MODULE].disable_adapters
        )

    def test_hybrid3_restores_adapter_if_a_tail_evaluation_raises(self):
        model = TinyTransformer()
        model.eval()
        bundle = TinyBundle()
        calls = 0

        def fail_on_first_tail(
            current_model, _bundle, state, timestep, _device, _dtype
        ):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.assertTrue(
                    dict(current_model.named_modules())[
                        MODULE.probe.TARGET_MODULE
                    ].disable_adapters
                )
                raise RuntimeError("injected tail failure")
            return torch.zeros_like(state)

        with patch.object(
            MODULE.probe, "_forward_velocity", side_effect=fail_on_first_tail
        ):
            with self.assertRaisesRegex(RuntimeError, "injected tail failure"):
                MODULE.sample_hybrid3(
                    model,
                    bundle,
                    self.schedule,
                    device=torch.device("cpu"),
                    model_dtype=torch.float32,
                    adapter_label="trained",
                )
        self.assertEqual(1, model.disable_calls)
        self.assertEqual(1, model.enable_calls)
        self.assertFalse(
            dict(model.named_modules())[MODULE.probe.TARGET_MODULE].disable_adapters
        )

    def test_hybrid3_fails_closed_when_adapter_cannot_be_disabled(self):
        model = TinyTransformer()
        model.eval()
        model.disable_adapters = None
        with self.assertRaisesRegex(ValueError, "disable.*adapter"):
            MODULE.sample_hybrid3(
                model,
                TinyBundle(),
                self.schedule,
                device=torch.device("cpu"),
                model_dtype=torch.float32,
                adapter_label="trained",
            )

    def test_seeded_zero_b_adapter_must_be_exactly_zero(self):
        self.assertTrue(MODULE._seeded_adapter_b_is_zero(TinyTransformer()))
        with self.assertRaisesRegex(ValueError, "zero-B"):
            MODULE._seeded_adapter_b_is_zero(TinyTransformer(zero_b=False))

    def test_seeded_first_coarse_update_checks_enabled_disabled_parity(self):
        bundle = TinyBundle()
        model = TinyTransformer()
        model.eval()
        with patch.object(
            MODULE.probe,
            "_forward_velocity",
            side_effect=lambda *_args: torch.full_like(
                bundle.initial_target_latents, 0.125
            ),
        ):
            result = MODULE.check_seeded_coarse_parity(
                model,
                bundle,
                self.schedule,
                device=torch.device("cpu"),
                model_dtype=torch.float32,
            )
        self.assertEqual(0.0, result["velocity_max_abs"])
        self.assertEqual(0.0, result["coarse_endpoint_rmse"])

        def wrong_velocity(current_model, _bundle, state, *_args):
            layer = dict(current_model.named_modules())[MODULE.probe.TARGET_MODULE]
            return torch.full_like(state, 0.25 if layer.disable_adapters else 0.5)

        with patch.object(
            MODULE.probe, "_forward_velocity", side_effect=wrong_velocity
        ):
            with self.assertRaisesRegex(ValueError, "seeded adapter.*base parity"):
                MODULE.check_seeded_coarse_parity(
                    model,
                    bundle,
                    self.schedule,
                    device=torch.device("cpu"),
                    model_dtype=torch.float32,
                )

    def test_latent_and_rgba_distance_metrics_are_finite_and_correct(self):
        left = torch.zeros((1, 256, 64), dtype=torch.float32)
        right = torch.ones_like(left)
        latent = MODULE.latent_distance(left, right)
        self.assertEqual(1.0, latent["mae"])
        self.assertEqual(1.0, latent["rmse"])
        self.assertEqual(1.0, latent["max_abs"])

        image_left = np.zeros((2, 2, 4), dtype=np.uint8)
        image_right = image_left.copy()
        image_right[0, 0, 0] = 4
        rgba = MODULE.rgba_distance(image_left, image_right)
        self.assertEqual(0.25, rgba["mae"])
        self.assertEqual(4.0, rgba["max_abs"])
        self.assertEqual(1, rgba["different_channel_values"])
        self.assertEqual(
            "raw integer RGBA levels on a 0-255 scale", rgba["channel_scale"]
        )

    def test_output_directory_must_be_new_and_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            output = root / "new-images"
            MODULE.validate_output_dir(output, repo_root=repo)
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "must not exist"):
                MODULE.validate_output_dir(output, repo_root=repo)
            with self.assertRaisesRegex(ValueError, "outside the repository"):
                MODULE.validate_output_dir(repo / "images", repo_root=repo)

    def test_vae_hash_verifier_rejects_substituted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            weights = root / "weights.safetensors"
            config.write_bytes(b"config")
            weights.write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "VAE config SHA256 mismatch"):
                MODULE.verify_vae_files(
                    config,
                    weights,
                    expected_config_sha256="0" * 64,
                    expected_weights_sha256="0" * 64,
                )

    def test_vae_revision_metadata_is_read_from_model_root_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vae_dir = root / "vae"
            vae_dir.mkdir()
            (vae_dir / "config.json").write_text(
                '{"z_dim":64,"out_channels":4,"scale_factor_spatial":16}',
                encoding="utf-8",
            )
            (vae_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
            metadata_dir = root / ".cache/huggingface/download/vae"
            metadata_dir.mkdir(parents=True)
            for filename in ("config.json", "diffusion_pytorch_model.safetensors"):
                (metadata_dir / f"{filename}.metadata").write_text(
                    f"{MODULE.probe.MODEL_REVISION}\ncontent-hash\n", encoding="utf-8"
                )
            with patch.object(
                MODULE,
                "verify_vae_files",
                return_value={"config_sha256": "a" * 64, "weights_sha256": "b" * 64},
            ):
                identity = MODULE.verify_vae_snapshot(root)
        self.assertEqual(MODULE.probe.MODEL_REVISION, identity["model_revision"])
        self.assertEqual("a" * 64, identity["config_sha256"])

    def test_main_help_exits_before_model_or_artifact_access(self):
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            MODULE.main(["--help"])
        self.assertEqual(0, raised.exception.code)
        self.assertIn("--output-dir", output.getvalue())


if __name__ == "__main__":
    unittest.main()
