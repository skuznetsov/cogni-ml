from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_image21_lora_distill_control as MODULE  # noqa: E402


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
        self.calls = 0
        self.grad_modes = []

    def forward(self, **kwargs):
        self.calls += 1
        self.grad_modes.append(torch.is_grad_enabled())
        velocity = self.transformer_blocks[31].attn.to_q(kwargs["hidden_states"])
        velocity = velocity + kwargs["timestep"].reshape(1, 1, 1)
        return (velocity,)


class InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf"))


class NonFiniteGradientTransformer(TinyTransformer):
    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        return (InfiniteGradient.apply(result[0]),)


class BaseGradientLeakTransformer(TinyTransformer):
    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        if self.calls > 2:
            base = self.transformer_blocks[0].attn.to_q.weight
            base.grad = torch.ones_like(base)
        return result


class NonFinitePostStepTransformer(TinyTransformer):
    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        if self.calls >= 5:
            return (torch.full_like(result[0], float("nan")),)
        return result


def make_bundle():
    from qwen_image21_lora_grad_probe import ConditioningBundle

    return ConditioningBundle(
        manifest_path=Path("tiny-conditioning.json"),
        payload_sha256="tiny",
        prompt="tiny target",
        img_shapes=[[(1, 16, 16)]],
        encoder_hidden_states=torch.zeros((1, 10, 4096), dtype=torch.float32),
        encoder_hidden_states_mask=torch.ones((1, 10), dtype=torch.bool),
        encoder_img_mask=torch.zeros((1, 10), dtype=torch.bool),
        initial_target_latents=torch.linspace(-1.0, 1.0, 256 * 64).reshape(1, 256, 64),
    )


def make_schedule():
    return MODULE.probe.load_flow_match_schedule(
        ROOT / "spec/fixtures/qwen_image21_flow_match_diffusers.json"
    )


def deterministic_train(model, **overrides):
    torch.manual_seed(419)
    options = {
        "steps": 4,
        "learning_rate": 0.003,
        "max_grad_norm": 1.0,
    }
    options.update(overrides)
    return MODULE.train_distillation(
        model,
        make_bundle(),
        make_schedule(),
        device=torch.device("cpu"),
        adapter_dtype=torch.float32,
        memory_snapshot_fn=lambda _torch, _device: {"device_type": "cpu", "rss_bytes": 1},
        **options,
    )


class QwenImage21LoraDistillControlSpec(unittest.TestCase):
    def test_distillation_loss_decreases_and_only_adapter_changes(self):
        torch.manual_seed(31)
        model = TinyTransformer()
        base_before = [
            (parameter, parameter.detach().clone())
            for parameter in model.parameters()
        ]

        result = deterministic_train(model)

        self.assertLess(result.steps[-1].post_step_loss, result.steps[0].post_step_loss)
        self.assertEqual(4, len(result.steps))
        self.assertTrue(all(step.adapter_weights_finite for step in result.steps))
        self.assertTrue(all(step.post_step_loss_finite for step in result.steps))
        self.assertTrue(result.base_parameters_have_no_grad)
        self.assertTrue(all(step.memory_after["rss_bytes"] == 1 for step in result.steps))
        self.assertTrue(
            all(
                step.preclip_gradient_norm + 1e-6 >= step.postclip_gradient_norm
                for step in result.steps
            )
        )
        adapter_names = set(result.trainable_parameter_names)
        self.assertEqual(2, len(adapter_names))
        changed_adapters = set()
        for name, parameter in model.named_parameters():
            if name in adapter_names:
                if not torch.equal(parameter.detach(), result.initial_adapter_weights[name]):
                    changed_adapters.add(name)
        for parameter, before in base_before:
            self.assertTrue(torch.equal(parameter.detach(), before))
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        self.assertTrue(changed_adapters)

    def test_teacher_endpoint_is_computed_once_for_all_student_steps(self):
        model = TinyTransformer()
        original = MODULE._compute_teacher_endpoint
        with patch.object(MODULE, "_compute_teacher_endpoint", wraps=original) as teacher:
            result = deterministic_train(model, steps=3)

        teacher.assert_called_once()
        self.assertEqual(3 + 2 * 3, model.calls)
        self.assertEqual(3, len(result.steps))
        self.assertTrue(result.teacher_target_detached)

    def test_initial_and_final_losses_share_no_grad_evaluation_context(self):
        model = TinyTransformer()
        original = MODULE._evaluate_student_loss
        evaluated_losses = []
        events = []

        def record_evaluation(*args, **kwargs):
            events.append("evaluation")
            loss = original(*args, **kwargs)
            evaluated_losses.append(loss)
            return loss

        base_optimizer = torch.optim.AdamW

        class RecordingAdamW(base_optimizer):
            def step(self, *args, **kwargs):
                events.append("optimizer_step")
                return super().step(*args, **kwargs)

        with patch.object(
            MODULE, "_evaluate_student_loss", side_effect=record_evaluation
        ) as evaluate, patch.object(torch.optim, "AdamW", RecordingAdamW):
            result = deterministic_train(model, steps=2)

        self.assertEqual(3, evaluate.call_count)
        self.assertEqual(
            ["evaluation", "optimizer_step", "evaluation", "optimizer_step", "evaluation"],
            events,
        )
        self.assertEqual([False, False, False, True, False, True, False], model.grad_modes)
        self.assertEqual(3, len(evaluated_losses))
        self.assertAlmostEqual(
            result.initial_eval_loss,
            evaluated_losses[0],
            places=12,
        )
        self.assertAlmostEqual(result.final_eval_loss, evaluated_losses[-1], places=12)
        self.assertAlmostEqual(
            result.final_eval_loss, result.steps[-1].post_step_loss, places=12
        )
        self.assertAlmostEqual(
            result.no_grad_context_delta,
            result.final_eval_loss - result.initial_eval_loss,
            places=12,
        )
        self.assertLess(result.final_eval_loss, result.initial_eval_loss)

    def test_nonfinite_gradient_blocks_optimizer_step(self):
        base_optimizer = torch.optim.AdamW
        step_calls = []

        class RecordingAdamW(base_optimizer):
            def step(self, *args, **kwargs):
                step_calls.append(True)
                return super().step(*args, **kwargs)

        with patch.object(torch.optim, "AdamW", RecordingAdamW):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                deterministic_train(NonFiniteGradientTransformer(), steps=1)

        self.assertEqual([], step_calls)

    def test_base_gradient_leakage_blocks_optimizer_step(self):
        base_optimizer = torch.optim.AdamW
        step_calls = []

        class RecordingAdamW(base_optimizer):
            def step(self, *args, **kwargs):
                step_calls.append(True)
                return super().step(*args, **kwargs)

        with patch.object(torch.optim, "AdamW", RecordingAdamW):
            with self.assertRaisesRegex(ValueError, "base unexpectedly received a gradient"):
                deterministic_train(BaseGradientLeakTransformer(), steps=1)

        self.assertEqual([], step_calls)

    def test_miswired_adapter_allowlist_rejected_before_optimizer_step(self):
        model = TinyTransformer()
        optimizer_steps = []
        base_optimizer = torch.optim.AdamW

        class RecordingAdamW(base_optimizer):
            def step(self, *args, **kwargs):
                optimizer_steps.append(True)
                return super().step(*args, **kwargs)

        def miswire_base_as_adapter(transformer, **_kwargs):
            names = (
                "transformer_blocks.0.attn.to_q.weight",
                "transformer_blocks.31.attn.to_q.weight",
            )
            for name, parameter in transformer.named_parameters():
                if name in names:
                    parameter.requires_grad_(True)
            return names

        with patch.object(
            MODULE.probe, "_install_probe_adapter", side_effect=miswire_base_as_adapter
        ), patch.object(torch.optim, "AdamW", RecordingAdamW):
            with self.assertRaisesRegex(ValueError, "exact LoRA A/B"):
                deterministic_train(model, steps=1)

        self.assertEqual([], optimizer_steps)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_unfrozen_base_rejected_even_with_correct_adapter_names(self):
        model = TinyTransformer()
        original = MODULE.probe._install_probe_adapter

        def unfreeze_base(transformer, **kwargs):
            names = original(transformer, **kwargs)
            transformer.transformer_blocks[0].attn.to_q.weight.requires_grad_(True)
            return names

        with patch.object(
            MODULE.probe, "_install_probe_adapter", side_effect=unfreeze_base
        ):
            with self.assertRaisesRegex(ValueError, "base parameter unexpectedly trainable"):
                deterministic_train(model, steps=1)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if "lora_" not in name
            )
        )

    def test_nonfinite_post_step_weights_are_restored(self):
        base_optimizer = torch.optim.AdamW
        before_step = {}

        class CorruptingAdamW(base_optimizer):
            def step(self, *args, **kwargs):
                for group in self.param_groups:
                    for parameter in group["params"]:
                        before_step[parameter] = parameter.detach().clone()
                result = super().step(*args, **kwargs)
                self.param_groups[0]["params"][0].data.fill_(float("nan"))
                return result

        model = TinyTransformer()
        with patch.object(torch.optim, "AdamW", CorruptingAdamW):
            with self.assertRaisesRegex(ValueError, "non-finite adapter weights.*rolled back"):
                deterministic_train(model, steps=1)

        for parameter, expected in before_step.items():
            self.assertTrue(torch.equal(parameter.detach(), expected))

    def test_nonfinite_post_step_loss_restores_adapter_weights(self):
        base_optimizer = torch.optim.AdamW
        before_step = {}

        class TrackingAdamW(base_optimizer):
            def step(self, *args, **kwargs):
                for group in self.param_groups:
                    for parameter in group["params"]:
                        before_step[parameter] = parameter.detach().clone()
                return super().step(*args, **kwargs)

        model = NonFinitePostStepTransformer()
        events = []
        with patch.object(torch.optim, "AdamW", TrackingAdamW):
            with self.assertRaisesRegex(ValueError, "post-step student loss.*rolled back"):
                deterministic_train(model, steps=1, on_step=events.append)

        self.assertEqual([], events)
        for parameter, expected in before_step.items():
            self.assertTrue(torch.equal(parameter.detach(), expected))

    def test_bounded_cli_arguments_and_safe_optimizer_values(self):
        required = ["--model-dir", "model", "--conditioning", "conditioning.json"]
        self.assertEqual(2, MODULE.parse_args(required).steps)
        self.assertEqual(8, MODULE.parse_args(required + ["--steps", "8"]).steps)
        for option, value in (
            ("--steps", "0"),
            ("--steps", "9"),
            ("--learning-rate", "nan"),
            ("--learning-rate", "0"),
            ("--learning-rate", "1"),
            ("--learning-rate", "0.0101"),
            ("--grad-clip-norm", "inf"),
            ("--grad-clip-norm", "0"),
            ("--grad-clip-norm", "10.1"),
        ):
            with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    MODULE.parse_args(required + [option, value])


if __name__ == "__main__":
    unittest.main()
