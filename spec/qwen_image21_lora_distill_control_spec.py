from __future__ import annotations

import io
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
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


def make_bundle(
    *,
    manifest_path=Path("tiny-conditioning.json"),
    payload_sha256="tiny",
    prompt="tiny target",
    latent_offset=0.0,
):
    from qwen_image21_lora_grad_probe import ConditioningBundle

    return ConditioningBundle(
        manifest_path=Path(manifest_path),
        payload_sha256=payload_sha256,
        prompt=prompt,
        img_shapes=[[(1, 16, 16)]],
        encoder_hidden_states=torch.zeros((1, 10, 4096), dtype=torch.float32),
        encoder_hidden_states_mask=torch.ones((1, 10), dtype=torch.bool),
        encoder_img_mask=torch.zeros((1, 10), dtype=torch.bool),
        initial_target_latents=(
            torch.linspace(-1.0, 1.0, 256 * 64).reshape(1, 256, 64) + latent_offset
        ),
    )


def make_schedule():
    return MODULE.probe.load_flow_match_schedule(
        ROOT / "spec/fixtures/qwen_image21_flow_match_diffusers.json"
    )


def deterministic_train(
    model, *, bundle=None, heldout_bundle=None, schedule=None, **overrides
):
    torch.manual_seed(419)
    options = {
        "steps": 4,
        "learning_rate": 0.003,
        "max_grad_norm": 1.0,
    }
    options.update(overrides)
    if heldout_bundle is not None:
        options["heldout_bundle"] = heldout_bundle
    return MODULE.train_distillation(
        model,
        bundle or make_bundle(),
        schedule or make_schedule(),
        device=torch.device("cpu"),
        adapter_dtype=torch.float32,
        memory_snapshot_fn=lambda _torch, _device: {"device_type": "cpu", "rss_bytes": 1},
        **options,
    )


def write_manifest(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class QwenImage21LoraDistillControlSpec(unittest.TestCase):
    def _bundle_pair(self, root: Path, *, same_manifest=False, same_payload=False, same_prompt=False):
        train_path = write_manifest(root / "train.json", json.dumps({"source": "train"}))
        heldout_path = train_path if same_manifest else write_manifest(
            root / "heldout.json",
            json.dumps({"source": "train" if same_manifest else "heldout"}),
        )
        train = make_bundle(
            manifest_path=train_path,
            payload_sha256="shared-payload" if same_payload else "train-payload",
            prompt="tiny target",
        )
        heldout = make_bundle(
            manifest_path=heldout_path,
            payload_sha256="shared-payload" if same_payload else "heldout-payload",
            prompt="tiny target" if same_prompt else "heldout target",
            latent_offset=0.125,
        )
        return train, heldout

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

    def test_heldout_evaluation_uses_one_frozen_teacher_and_no_grad_same_context_losses(self):
        model = TinyTransformer()
        base_optimizer = torch.optim.AdamW
        events = []
        optimizer_parameter_ids = []
        teacher_bundles = []
        heldout_eval_snapshots = []
        original_evaluate = MODULE._evaluate_student_loss
        original_teacher = MODULE._compute_teacher_endpoint

        class RecordingAdamW(base_optimizer):
            def __init__(self, parameters, *args, **kwargs):
                parameters = list(parameters)
                optimizer_parameter_ids.extend(id(parameter) for parameter in parameters)
                super().__init__(parameters, *args, **kwargs)

            def step(self, *args, **kwargs):
                events.append("optimizer_step")
                return super().step(*args, **kwargs)

        def record_teacher(transformer, bundle, *args, **kwargs):
            teacher_bundles.append(bundle)
            return original_teacher(transformer, bundle, *args, **kwargs)

        def record_evaluate(transformer, bundle, *args, **kwargs):
            events.append("heldout_eval" if bundle is heldout else "train_eval")
            before = [parameter.detach().clone() for parameter in transformer.parameters()]
            loss = original_evaluate(transformer, bundle, *args, **kwargs)
            if bundle is heldout:
                self.assertTrue(all(parameter.grad is None for parameter in transformer.parameters()))
                self.assertTrue(
                    all(
                        torch.equal(parameter.detach(), prior)
                        for parameter, prior in zip(transformer.parameters(), before)
                    )
                )
                heldout_eval_snapshots.append(tuple(before))
            return loss

        with tempfile.TemporaryDirectory() as directory:
            train, heldout = self._bundle_pair(Path(directory))
            with patch.object(MODULE, "_compute_teacher_endpoint", side_effect=record_teacher), patch.object(
                MODULE, "_evaluate_student_loss", side_effect=record_evaluate
            ), patch.object(torch.optim, "AdamW", RecordingAdamW):
                result = deterministic_train(
                    model, bundle=train, heldout_bundle=heldout, steps=2
                )

        self.assertIsNotNone(result.heldout_evaluation)
        evaluation = result.heldout_evaluation
        self.assertTrue(evaluation.teacher_target_detached)
        self.assertEqual(heldout.payload_sha256, evaluation.conditioning_payload_sha256)
        self.assertTrue(torch.isfinite(torch.tensor(evaluation.initial_eval_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(evaluation.final_eval_loss)))
        self.assertAlmostEqual(
            evaluation.no_grad_context_delta,
            evaluation.final_eval_loss - evaluation.initial_eval_loss,
            places=12,
        )
        self.assertEqual([train, heldout], teacher_bundles)
        self.assertEqual(2, len(heldout_eval_snapshots))
        self.assertEqual(
            ["train_eval", "heldout_eval", "optimizer_step", "train_eval",
             "optimizer_step", "train_eval", "heldout_eval"],
            events,
        )
        self.assertEqual(2, len(optimizer_parameter_ids))
        optimizer_names = {
            name for name, parameter in model.named_parameters()
            if id(parameter) in set(optimizer_parameter_ids)
        }
        self.assertEqual(set(result.trainable_parameter_names), optimizer_names)
        self.assertEqual([False] * 6 + [True, False, True, False, False], model.grad_modes)

    def test_heldout_evaluation_detects_and_restores_adapter_mutation(self):
        model = TinyTransformer()
        original_evaluate = MODULE._evaluate_student_loss
        adapter_before = []
        optimizer_constructions = []
        base_optimizer = torch.optim.AdamW

        class RecordingAdamW(base_optimizer):
            def __init__(self, *args, **kwargs):
                optimizer_constructions.append(True)
                super().__init__(*args, **kwargs)

        def mutate_during_heldout(transformer, bundle, *args, **kwargs):
            if bundle is heldout:
                adapter_parameters = [
                    parameter
                    for name, parameter in transformer.named_parameters()
                    if ".lora_" in name
                ]
                adapter_before.extend(
                    (parameter, parameter.detach().clone())
                    for parameter in adapter_parameters
                )
                with torch.no_grad():
                    adapter_parameters[0].add_(1.0)
            return original_evaluate(transformer, bundle, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            train, heldout = self._bundle_pair(Path(directory))
            with patch.object(
                MODULE, "_evaluate_student_loss", side_effect=mutate_during_heldout
            ), patch.object(torch.optim, "AdamW", RecordingAdamW):
                with self.assertRaisesRegex(
                    ValueError, "held-out no-grad evaluation changed adapter weights.*restored"
                ):
                    deterministic_train(
                        model, bundle=train, heldout_bundle=heldout, steps=1
                    )

        self.assertEqual([], optimizer_constructions)
        self.assertEqual(2, len(adapter_before))
        for parameter, expected in adapter_before:
            self.assertTrue(torch.equal(parameter.detach(), expected))

    def test_heldout_rejects_same_manifest_payload_or_prompt_before_forward(self):
        cases = (
            ("same manifest", True, False, False, "different conditioning manifest"),
            ("same payload", False, True, False, "distinct conditioning payload"),
            ("same prompt", False, False, True, "distinct conditioning prompt"),
        )
        for label, same_manifest, same_payload, same_prompt, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                train, heldout = self._bundle_pair(
                    Path(directory),
                    same_manifest=same_manifest,
                    same_payload=same_payload,
                    same_prompt=same_prompt,
                )
                model = TinyTransformer()
                with self.assertRaisesRegex(ValueError, message):
                    deterministic_train(
                        model, bundle=train, heldout_bundle=heldout, steps=1
                    )
                self.assertEqual(0, model.calls)

    def test_heldout_rejects_matching_manifest_contents_even_at_another_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = write_manifest(root / "train.json", "same bytes")
            heldout_path = write_manifest(root / "copy.json", "same bytes")
            train = make_bundle(manifest_path=train_path, payload_sha256="train-payload")
            heldout = make_bundle(
                manifest_path=heldout_path,
                payload_sha256="heldout-payload",
                prompt="heldout target",
            )
            with self.assertRaisesRegex(ValueError, "reuse the training conditioning manifest"):
                deterministic_train(
                    TinyTransformer(), bundle=train, heldout_bundle=heldout, steps=1
                )

    def test_heldout_manifest_pin_is_rechecked_at_training_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            train, heldout = self._bundle_pair(Path(directory))
            heldout_pin = MODULE.probe.sha256_file(heldout.manifest_path)
            heldout.manifest_path.write_text('{"source":"rewritten"}', encoding="utf-8")
            model = TinyTransformer()
            with self.assertRaisesRegex(
                ValueError, "held-out conditioning manifest SHA256 mismatch"
            ):
                deterministic_train(
                    model,
                    bundle=train,
                    heldout_bundle=heldout,
                    heldout_manifest_sha256_pin=heldout_pin,
                    steps=1,
                )
            self.assertEqual(0, model.calls)

    def test_training_manifest_pin_can_be_rechecked_at_training_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            train, _heldout = self._bundle_pair(Path(directory))
            training_pin = MODULE.probe.sha256_file(train.manifest_path)
            train.manifest_path.write_text('{"source":"rewritten"}', encoding="utf-8")
            model = TinyTransformer()
            with self.assertRaisesRegex(ValueError, "conditioning manifest SHA256 mismatch"):
                deterministic_train(
                    model,
                    bundle=train,
                    conditioning_manifest_sha256_pin=training_pin,
                    steps=1,
                )
            self.assertEqual(0, model.calls)

    def test_heldout_geometry_and_schedule_must_match_the_pinned_control(self):
        with tempfile.TemporaryDirectory() as directory:
            train, heldout = self._bundle_pair(Path(directory))
            wrong_geometry = replace(
                heldout,
                initial_target_latents=torch.zeros((1, 255, 64), dtype=torch.float32),
            )
            model = TinyTransformer()
            with self.assertRaisesRegex(ValueError, "256-token target geometry"):
                deterministic_train(
                    model, bundle=train, heldout_bundle=wrong_geometry, steps=1
                )
            self.assertEqual(0, model.calls)

            schedule = replace(make_schedule(), sigmas=(1.0, 0.7, 0.4, 0.02, 0.0))
            with self.assertRaisesRegex(ValueError, "schedule nodes differ"):
                deterministic_train(
                    model,
                    bundle=train,
                    heldout_bundle=heldout,
                    schedule=schedule,
                    steps=1,
                )
            self.assertEqual(0, model.calls)

    def test_heldout_loader_pins_manifest_payload_and_provenance(self):
        required = ["--model-dir", "model", "--conditioning", "conditioning.json"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(required + ["--heldout-conditioning", "heldout.json"])

        manifest = {
            "model": {
                "repo": "Qwen/Qwen-Image-2.1",
                "revision": MODULE.probe.MODEL_REVISION,
            },
            "runtime": {
                "diffusers_commit": MODULE.probe.DIFFUSERS_COMMIT,
                "device": "cpu",
            },
            "noise": {
                "source_dtype": "bfloat16",
                "generator_device": "cpu",
                "seed": 7,
            },
            "image": {
                "width": 256,
                "height": 256,
                "latent_height": 16,
                "latent_width": 16,
                "img_shapes": [[1, 16, 16]],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "heldout.json"
            original_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
            manifest_path.write_bytes(original_bytes)
            manifest_sha = hashlib.sha256(original_bytes).hexdigest()
            expected_payload = "a" * 64
            args = MODULE.parse_args(
                required
                + [
                    "--heldout-conditioning", str(manifest_path),
                    "--heldout-manifest-sha256", manifest_sha,
                    "--heldout-payload-sha256", expected_payload,
                ]
            )
            self.assertEqual(manifest_path, args.heldout_conditioning)

            loaded = make_bundle(
                manifest_path=manifest_path,
                payload_sha256=expected_payload,
                prompt="heldout target",
            )
            with patch.object(
                MODULE.probe, "load_conditioning_bundle", return_value=loaded
            ) as load:
                self.assertIs(
                    loaded,
                    MODULE.load_heldout_conditioning_bundle(
                        manifest_path,
                        expected_manifest_sha256=manifest_sha,
                        expected_payload_sha256=expected_payload,
                    ),
                )
            load.assert_called_once_with(manifest_path, verify_pinned_bundle=False)

            def rewrite_manifest_during_load(*_args, **_kwargs):
                manifest_path.write_text(
                    json.dumps({**manifest, "concurrent_rewrite": True}),
                    encoding="utf-8",
                )
                return loaded

            with patch.object(
                MODULE.probe,
                "load_conditioning_bundle",
                side_effect=rewrite_manifest_during_load,
            ):
                with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                    MODULE.load_heldout_conditioning_bundle(
                        manifest_path,
                        expected_manifest_sha256=manifest_sha,
                        expected_payload_sha256=expected_payload,
                    )
            manifest_path.write_bytes(original_bytes)

            tampered_manifest = {**manifest, "runtime": {**manifest["runtime"], "device": "cuda"}}
            manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
            with patch.object(MODULE.probe, "load_conditioning_bundle") as load:
                with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                    MODULE.load_heldout_conditioning_bundle(
                        manifest_path,
                        expected_manifest_sha256=manifest_sha,
                        expected_payload_sha256=expected_payload,
                    )
                load.assert_not_called()

    def test_heldout_loader_rejects_payload_pin_and_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = write_manifest(Path(directory) / "heldout.json", "pinned manifest")
            manifest_sha = MODULE.probe.sha256_file(manifest_path)
            expected_payload = "a" * 64
            loaded = make_bundle(
                manifest_path=manifest_path,
                payload_sha256="b" * 64,
                prompt="heldout target",
            )
            with patch.object(MODULE.probe, "load_conditioning_bundle", return_value=loaded):
                with self.assertRaisesRegex(ValueError, "payload SHA256 mismatch"):
                    MODULE.load_heldout_conditioning_bundle(
                        manifest_path,
                        expected_manifest_sha256=manifest_sha,
                        expected_payload_sha256=expected_payload,
                    )

            invalid_manifest = {
                "model": {
                    "repo": "Qwen/Qwen-Image-2.1",
                    "revision": MODULE.probe.MODEL_REVISION,
                },
                "runtime": {
                    "diffusers_commit": MODULE.probe.DIFFUSERS_COMMIT,
                    "device": "cuda",
                },
                "noise": {
                    "source_dtype": "bfloat16",
                    "generator_device": "cpu",
                    "seed": 7,
                },
                "image": {"width": 256, "height": 256},
            }
            manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
            invalid_sha = MODULE.probe.sha256_file(manifest_path)
            loaded = make_bundle(
                manifest_path=manifest_path,
                payload_sha256=expected_payload,
                prompt="heldout target",
            )
            with patch.object(MODULE.probe, "load_conditioning_bundle", return_value=loaded):
                with self.assertRaisesRegex(ValueError, "CPU conditioning provenance"):
                    MODULE.load_heldout_conditioning_bundle(
                        manifest_path,
                        expected_manifest_sha256=invalid_sha,
                        expected_payload_sha256=expected_payload,
                    )

            manifest_path.write_text("[]", encoding="utf-8")
            non_object_sha = MODULE.probe.sha256_file(manifest_path)
            with patch.object(MODULE.probe, "load_conditioning_bundle", return_value=loaded):
                with self.assertRaisesRegex(ValueError, "invalid CPU conditioning provenance"):
                    MODULE.load_heldout_conditioning_bundle(
                        manifest_path,
                        expected_manifest_sha256=non_object_sha,
                        expected_payload_sha256=expected_payload,
                    )

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
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(required + ["--adapter-output", "new-checkpoint"])
        heldout_args = required + [
            "--heldout-conditioning", "heldout.json",
            "--heldout-manifest-sha256", "a" * 64,
            "--heldout-payload-sha256", "b" * 64,
            "--adapter-output", "new-checkpoint",
        ]
        self.assertEqual(Path("new-checkpoint"), MODULE.parse_args(heldout_args).adapter_output)
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
