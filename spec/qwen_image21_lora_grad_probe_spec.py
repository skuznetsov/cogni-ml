from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_image21_lora_grad_probe.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_lora_grad_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QwenImage21LoraGradProbeSpec(unittest.TestCase):
    def test_schedule_keeps_nested_nodes_and_model_time_distinct_from_sigma(self):
        schedule = MODULE.load_flow_match_schedule(
            ROOT / "spec/fixtures/qwen_image21_flow_match_diffusers.json"
        )
        self.assertEqual((0, 1), schedule.teacher_step_indices)
        self.assertEqual((0,), schedule.student_step_indices)
        self.assertEqual(1.0, schedule.sigmas[0])
        self.assertEqual(0.744611382484436, schedule.sigmas[1])
        self.assertEqual(0.4266734719276428, schedule.sigmas[2])
        self.assertEqual(1000.0, schedule.timesteps[0])
        self.assertEqual(744.6113891601562, schedule.timesteps[1])
        self.assertEqual(426.6734619140625, schedule.timesteps[2])
        self.assertNotEqual(schedule.timesteps[1] / 1000, schedule.sigmas[1])
        self.assertNotEqual(schedule.timesteps[2] / 1000, schedule.sigmas[2])

    def test_teacher_two_euler_updates_and_student_one_coarse_update(self):
        schedule = MODULE.load_flow_match_schedule(
            ROOT / "spec/fixtures/qwen_image21_flow_match_diffusers.json"
        )
        initial = torch.tensor([2.0])
        velocity_0 = torch.tensor([3.0])
        velocity_1 = torch.tensor([-1.0])
        student_velocity = torch.tensor([0.5])
        teacher = MODULE.euler_endpoint(
            initial, (velocity_0, velocity_1), schedule.sigmas[:3]
        )
        student = MODULE.euler_endpoint(
            initial, (student_velocity,), (schedule.sigmas[0], schedule.sigmas[2])
        )
        self.assertAlmostEqual(
            2.0 + (schedule.sigmas[1] - schedule.sigmas[0]) * 3.0
            + (schedule.sigmas[2] - schedule.sigmas[1]) * -1.0,
            teacher.item(),
            places=6,
        )
        self.assertAlmostEqual(
            2.0 + (schedule.sigmas[2] - schedule.sigmas[0]) * 0.5,
            student.item(),
            places=6,
        )

    def test_bundle_loader_hash_checks_and_forward_inputs_append_target_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_bundle(root)
            bundle = MODULE.load_conditioning_bundle(manifest, verify_pinned_bundle=False)
            self.assertEqual((1, 256, 64), tuple(bundle.initial_target_latents.shape))
            self.assertEqual((1, 10, 4096), tuple(bundle.encoder_hidden_states.shape))
            self.assertEqual((1, 10), tuple(bundle.encoder_hidden_states_mask.shape))
            self.assertEqual((1, 10), tuple(bundle.encoder_img_mask.shape))

            kwargs = MODULE.build_forward_kwargs(
                bundle,
                bundle.initial_target_latents,
                scheduler_timestep=744.6113891601562,
                device=torch.device("cpu"),
                model_dtype=torch.float32,
            )
            self.assertEqual(
                {
                    "hidden_states",
                    "encoder_hidden_states",
                    "encoder_hidden_states_mask",
                    "img_shapes",
                    "img_mask",
                    "timestep",
                    "attention_kwargs",
                    "return_dict",
                },
                set(kwargs),
            )
            self.assertEqual((1, 74), tuple(kwargs["img_mask"].shape))
            self.assertEqual(64, int(kwargs["img_mask"].sum()))
            self.assertEqual([[ (1, 16, 16) ]], kwargs["img_shapes"])
            self.assertEqual((1,), tuple(kwargs["timestep"].shape))
            self.assertAlmostEqual(0.7446113891601562, kwargs["timestep"].item(), places=7)
            bf16_kwargs = MODULE.build_forward_kwargs(
                bundle,
                bundle.initial_target_latents,
                scheduler_timestep=744.6113891601562,
                device=torch.device("cpu"),
                model_dtype=torch.bfloat16,
            )
            self.assertEqual(0.7421875, bf16_kwargs["timestep"].item())

            payload = root / "qwen_image21_conditioning.bin"
            payload.write_bytes(payload.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "payload.*(size|SHA256)"):
                MODULE.load_conditioning_bundle(manifest, verify_pinned_bundle=False)

    def test_unpinned_loader_accepts_structurally_valid_variable_text_length(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_bundle(Path(directory), text_tokens=37)
            bundle = MODULE.load_conditioning_bundle(manifest, verify_pinned_bundle=False)
            self.assertEqual((1, 37, 4096), tuple(bundle.encoder_hidden_states.shape))
            self.assertEqual((1, 37), tuple(bundle.encoder_hidden_states_mask.shape))
            kwargs = MODULE.build_forward_kwargs(
                bundle,
                bundle.initial_target_latents,
                scheduler_timestep=1000.0,
                device=torch.device("cpu"),
                model_dtype=torch.bfloat16,
            )
            self.assertEqual((1, 101), tuple(kwargs["img_mask"].shape))

    def test_variable_text_length_rejects_mismatched_mask_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_bundle(Path(directory), text_tokens=37)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tensors"]["encoder_hidden_states_mask"]["shape"] = [36]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tensor descriptor is invalid"):
                MODULE.load_conditioning_bundle(manifest_path, verify_pinned_bundle=False)

    def test_variable_text_length_rejects_zero_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_bundle(Path(directory), text_tokens=10)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["tensors"]["encoder_hidden_states"]["shape"] = [0, 4096]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "text token count"):
                MODULE.load_conditioning_bundle(manifest_path, verify_pinned_bundle=False)

    def test_rank4_probe_backpropagates_only_into_one_lora_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = MODULE.load_conditioning_bundle(
                self._write_bundle(Path(directory)), verify_pinned_bundle=False
            )
        model = TinyTransformer()
        result = MODULE.run_gradient_probe(
            model,
            bundle,
            MODULE.load_flow_match_schedule(
                ROOT / "spec/fixtures/qwen_image21_flow_match_diffusers.json"
            ),
            device=torch.device("cpu"),
            adapter_dtype=torch.float32,
        )
        self.assertEqual("transformer_blocks.31.attn.to_q", result.target_module)
        self.assertEqual(4, result.rank)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))
        self.assertGreater(result.loss, 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(result.adapter_gradient_norm)))
        self.assertGreater(result.adapter_gradient_norm, 0.0)
        self.assertEqual(3, len(model.calls))
        for actual, expected in zip(model.raw_timesteps, [1000.0, 744.6113891601562, 1000.0]):
            self.assertAlmostEqual(expected, actual, places=4)
        self.assertTrue(all(call["img_mask"].shape == (1, 74) for call in model.calls))
        self.assertTrue(all(call["img_mask"].sum().item() == 64 for call in model.calls))
        self.assertTrue(result.base_parameters_have_no_grad)
        self.assertFalse(result.optimizer_step_performed)
        self.assertTrue(all("lora_" in name for name in result.trainable_parameter_names))
        self.assertEqual(2, len(result.trainable_parameter_names))

    def test_diffusers_add_adapter_installs_rank4_on_exact_projection(self):
        from diffusers.loaders import PeftAdapterMixin

        class TinyDiffusersTransformer(TinyTransformer, PeftAdapterMixin):
            pass

        model = TinyDiffusersTransformer()
        self.assertIs(model.add_adapter.__func__, PeftAdapterMixin.add_adapter)
        trainable_names = MODULE._install_probe_adapter(
            model, adapter_dtype=torch.float32, device=torch.device("cpu")
        )
        target = dict(model.named_modules())[MODULE.TARGET_MODULE]
        self.assertEqual(2, len(trainable_names))
        self.assertEqual((4, 64), tuple(target.lora_A[MODULE.ADAPTER_NAME].weight.shape))
        self.assertEqual((64, 4), tuple(target.lora_B[MODULE.ADAPTER_NAME].weight.shape))
        self.assertEqual(MODULE.ADAPTER_NAME, model.active_adapters()[0])

    def test_adapter_initialization_is_deterministic_and_preserves_global_rng(self):
        from diffusers.loaders import PeftAdapterMixin

        class TinyDiffusersTransformer(TinyTransformer, PeftAdapterMixin):
            pass

        def initialized_parameters(prior_seed):
            torch.manual_seed(prior_seed)
            model = TinyDiffusersTransformer()
            rng_before = torch.random.get_rng_state().clone()
            trainable_names = MODULE._install_probe_adapter(
                model, adapter_dtype=torch.float32, device=torch.device("cpu")
            )
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            return {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if name in trainable_names
            }

        first = initialized_parameters(7)
        second = initialized_parameters(911)
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(first[name], second[name]))
        self.assertTrue(any(float(parameter.norm()) > 0.0 for parameter in first.values()))

    def test_incompatible_peft_version_is_rejected(self):
        for version in ("0.15.1", "0.17.9"):
            with self.subTest(version=version), self.assertRaisesRegex(
                RuntimeError, rf"PEFT >= 0\.18\.0.*{version}"
            ):
                MODULE.validate_peft_version(version)
        self.assertEqual("0.18.0", MODULE.validate_peft_version("0.18.0"))

    def test_hash_and_memory_preflights_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pinned-file"
            path.write_bytes(b"expected source")
            digest = hashlib.sha256(b"expected source").hexdigest()
            self.assertEqual(digest, MODULE.require_sha256(path, digest, "test source"))
            path.write_bytes(b"replaced source")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                MODULE.require_sha256(path, digest, "test source")

        with self.assertRaisesRegex(MemoryError, "MPS guard"):
            MODULE.guard_memory_budget(
                4 * MODULE.GIB,
                device_type="mps",
                device_limit_bytes=8 * MODULE.GIB,
                current_device_bytes=0,
                pressure_estimated_available_host_bytes=48 * MODULE.GIB,
                raw_reclaimable_host_bytes=40 * MODULE.GIB,
                total_host_bytes=64 * MODULE.GIB,
            )
        memory_guard = MODULE.guard_memory_budget(
            1 * MODULE.GIB,
            device_type="cpu",
            device_limit_bytes=None,
            current_device_bytes=0,
            pressure_estimated_available_host_bytes=40 * MODULE.GIB,
            raw_reclaimable_host_bytes=30 * MODULE.GIB,
            total_host_bytes=64 * MODULE.GIB,
        )
        self.assertEqual(40 * MODULE.GIB, memory_guard["pressure_estimated_available_host_bytes"])
        self.assertEqual(30 * MODULE.GIB, memory_guard["raw_reclaimable_host_bytes"])
        self.assertTrue(memory_guard["estimated_peak_not_guaranteed"])
        for pressure_available, raw_reclaimable in (
            (22 * MODULE.GIB, 30 * MODULE.GIB),
            (40 * MODULE.GIB, 12 * MODULE.GIB),
        ):
            with self.subTest(pressure_available=pressure_available, raw_reclaimable=raw_reclaimable):
                with self.assertRaisesRegex(MemoryError, "host-memory guard"):
                    MODULE.guard_memory_budget(
                        4 * MODULE.GIB,
                        device_type="cpu",
                        device_limit_bytes=None,
                        current_device_bytes=0,
                        pressure_estimated_available_host_bytes=pressure_available,
                        raw_reclaimable_host_bytes=raw_reclaimable,
                        total_host_bytes=64 * MODULE.GIB,
                    )
        pressure_available, raw_reclaimable, total = MODULE._parse_darwin_memory(
            "The system has 68719476736 (4194304 pages with a page size of 16384).\n"
            "System-wide memory free percentage: 67%\n",
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 10.\nPages inactive: 20.\nPages speculative: 3.\n"
            "Pages purgeable: 4.\n",
        )
        self.assertEqual(64 * MODULE.GIB * 66 // 100, pressure_available)
        self.assertEqual(37 * 16384, raw_reclaimable)
        self.assertEqual(64 * MODULE.GIB, total)

    def test_default_bundle_loader_fails_closed_on_unpinned_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_bundle(Path(directory))
            with self.assertRaisesRegex(ValueError, "conditioning manifest SHA256 mismatch"):
                MODULE.load_conditioning_bundle(manifest)

    def test_cli_rejects_unbudgeted_device_and_dtype_modes(self):
        for option, value in (("--device", "cpu"), ("--dtype", "float32")):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    MODULE.main(
                        ["--model-dir", "unused", "--conditioning", "unused", option, value]
                    )
                self.assertEqual(2, raised.exception.code)
        cap_calls = []
        fake_torch = SimpleNamespace(
            mps=SimpleNamespace(
                set_per_process_memory_fraction=lambda fraction: cap_calls.append(fraction)
            )
        )
        MODULE._set_device_memory_cap(fake_torch, SimpleNamespace(type="mps"), 0.70)
        self.assertEqual([0.70], cap_calls)
        pinned_direct_url = json.dumps(
            {
                "url": "https://github.com/huggingface/diffusers.git",
                "vcs_info": {"vcs": "git", "commit_id": MODULE.DIFFUSERS_COMMIT},
            }
        )
        self.assertEqual(
            MODULE.DIFFUSERS_COMMIT,
            MODULE.validate_diffusers_direct_url(pinned_direct_url),
        )
        for unpinned in (None, "{}", json.dumps({"vcs_info": {"vcs": "git", "commit_id": "main"}})):
            with self.subTest(direct_url=unpinned), self.assertRaises(RuntimeError):
                MODULE.validate_diffusers_direct_url(unpinned)

    def test_nonfinite_zero_and_frozen_base_gradients_are_rejected(self):
        schedule = MODULE.load_flow_match_schedule(
            ROOT / "spec/fixtures/qwen_image21_flow_match_diffusers.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle = MODULE.load_conditioning_bundle(
                self._write_bundle(Path(directory)), verify_pinned_bundle=False
            )
        for model, expected_message in (
            (NonFiniteTransformer(), "non-finite velocity"),
            (ZeroGradTransformer(), "gradient norm is zero"),
            (BaseGradientTransformer(), "frozen transformer base unexpectedly received a gradient"),
        ):
            with self.subTest(model=type(model).__name__), self.assertRaisesRegex(
                ValueError, expected_message
            ):
                MODULE.run_gradient_probe(
                    model,
                    bundle,
                    schedule,
                    device=torch.device("cpu"),
                    adapter_dtype=torch.float32,
                )

    def _write_bundle(self, root: Path, *, text_tokens: int = 10) -> Path:
        import hashlib
        import numpy as np

        tensors = {
            "encoder_hidden_states": np.zeros((text_tokens, 4096), dtype="<f4"),
            "encoder_hidden_states_mask": np.ones((text_tokens,), dtype=np.uint8),
            "encoder_img_mask": np.zeros((text_tokens,), dtype=np.uint8),
            "initial_target_latents": np.linspace(-1, 1, 256 * 64, dtype="<f4").reshape(256, 64),
        }
        payload = bytearray()
        descriptors = {}
        dtype_map = {
            "encoder_hidden_states": "float32-le",
            "encoder_hidden_states_mask": "uint8",
            "encoder_img_mask": "uint8",
            "initial_target_latents": "float32-le",
        }
        for name, tensor in tensors.items():
            content = tensor.tobytes(order="C")
            descriptors[name] = {
                "dtype": dtype_map[name],
                "shape": list(tensor.shape),
                "offset_bytes": len(payload),
                "nbytes": len(content),
            }
            payload.extend(content)
        payload_path = root / "qwen_image21_conditioning.bin"
        payload_path.write_bytes(payload)
        manifest = {
            "schema": "qwen-image21-conditioning",
            "schema_version": 1,
            "model": {
                "repo": "Qwen/Qwen-Image-2.1",
                "revision": MODULE.MODEL_REVISION,
            },
            "runtime": {"diffusers_commit": MODULE.DIFFUSERS_COMMIT},
            "prompt": "red cube",
            "image": {
                "width": 256,
                "height": 256,
                "latent_height": 16,
                "latent_width": 16,
                "img_shapes": [[1, 16, 16]],
            },
            "payload_file": payload_path.name,
            "payload_nbytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "tensors": descriptors,
        }
        manifest_path = root / "qwen_image21_conditioning.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(64, 64, bias=False)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([TinyBlock() for _ in range(32)])
        self.calls = []
        self.raw_timesteps = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        self.raw_timesteps.append(float(kwargs["timestep"].item() * 1000))
        velocity = self.transformer_blocks[31].attn.to_q(kwargs["hidden_states"])
        velocity = velocity + kwargs["timestep"].reshape(1, 1, 1)
        return (velocity,)


class NonFiniteTransformer(TinyTransformer):
    def forward(self, **kwargs):
        super().forward(**kwargs)
        return (torch.full_like(kwargs["hidden_states"], float("nan")),)


class ZeroGradTransformer(TinyTransformer):
    def forward(self, **kwargs):
        self.calls.append(kwargs)
        self.raw_timesteps.append(float(kwargs["timestep"].item() * 1000))
        velocity = self.transformer_blocks[31].attn.to_q(kwargs["hidden_states"]) * 0.0
        velocity = velocity + kwargs["timestep"].reshape(1, 1, 1)
        return (velocity,)


class BaseGradientTransformer(TinyTransformer):
    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        if len(self.calls) == 3:
            base = self.transformer_blocks[0].attn.to_q.weight
            base.grad = torch.ones_like(base)
        return result


if __name__ == "__main__":
    unittest.main()
