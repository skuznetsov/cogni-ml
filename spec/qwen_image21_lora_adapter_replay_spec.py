from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_image21_lora_adapter_replay as MODULE  # noqa: E402


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(1, 1)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

        self.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(32)])
        self.requires_grad_(False)


class QwenImage21LoraAdapterReplaySpec(unittest.TestCase):
    def test_loss_comparison_reports_tolerated_non_bitwise_delta(self):
        comparison = MODULE.compare_loss(
            "heldout_final", 0.0439080, 0.04390815272927284
        )

        self.assertTrue(comparison["within_tolerance"])
        self.assertNotEqual(0.0, comparison["delta"])
        self.assertEqual(
            "absolute_delta <= max(abs_tol, rel_tol * abs(expected))",
            comparison["rule"],
        )
        self.assertFalse(comparison["bitwise_equality_required"])

    def test_bundle_prompts_must_be_exact_train_red_cube_and_hashed_heldout_castle(
        self,
    ):
        schedule = SimpleNamespace()
        train = SimpleNamespace(prompt="red cube")
        heldout = SimpleNamespace(prompt="A stone castle at dusk")
        with patch.object(MODULE.distill, "_validate_heldout_bundle"):
            MODULE.validate_conditioning_bundles(train, heldout, schedule)

            with self.assertRaisesRegex(ValueError, "training prompt must be exactly"):
                MODULE.validate_conditioning_bundles(
                    SimpleNamespace(prompt="red cube, blue sky"), heldout, schedule
                )
            with self.assertRaisesRegex(
                ValueError, "held-out prompt must identify a castle"
            ):
                MODULE.validate_conditioning_bundles(
                    train, SimpleNamespace(prompt="A stone tower at dusk"), schedule
                )

    def test_manifest_pin_mismatch_fails_before_model_snapshot_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / MODULE.adapter_checkpoint.MANIFEST_NAME).write_text(
                "changed manifest", encoding="utf-8"
            )
            args = SimpleNamespace(
                checkpoint_dir=checkpoint,
                checkpoint_manifest_sha256="a" * 64,
            )
            with (
                patch.object(MODULE.probe, "load_flow_match_schedule") as load_schedule,
                patch.object(MODULE.probe, "_load_transformer") as load_transformer,
            ):
                with self.assertRaisesRegex(
                    ValueError, "differs from the explicit pin"
                ):
                    MODULE.run_replay(args, torch_module=torch)
                load_schedule.assert_not_called()
                load_transformer.assert_not_called()

    def test_replay_checks_initial_and_final_losses_with_teachers_before_each_adapter(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model"
            model_path.mkdir()
            checkpoint_path = root / "checkpoint"
            checkpoint_path.mkdir()
            train_path = root / "train.json"
            heldout_path = root / "heldout.json"
            train_path.write_text("train manifest", encoding="utf-8")
            heldout_path.write_text("heldout manifest", encoding="utf-8")
            args = argparse.Namespace(
                model_dir=model_path,
                conditioning=train_path,
                heldout_conditioning=heldout_path,
                heldout_manifest_sha256="a" * 64,
                heldout_payload_sha256="b" * 64,
                checkpoint_dir=checkpoint_path,
                checkpoint_manifest_sha256=None,
                schedule=root / "schedule.json",
                mps_memory_fraction=0.70,
            )
            train_bundle = SimpleNamespace(
                manifest_path=train_path,
                payload_sha256="c" * 64,
                prompt="red cube",
            )
            heldout_bundle = SimpleNamespace(
                manifest_path=heldout_path,
                payload_sha256="d" * 64,
                prompt="A stone castle at dusk",
            )
            schedule = SimpleNamespace(fixture_sha256="e" * 64)
            expected = {
                "train_initial_eval_loss": 0.027024995535612106,
                "train_final_eval_loss": 0.026850158348679543,
                "heldout_initial_eval_loss": 0.04398445039987564,
                "heldout_final_eval_loss": 0.04390815272927284,
            }
            adapter_names = (
                f"{MODULE.probe.TARGET_MODULE}.lora_A.{MODULE.probe.ADAPTER_NAME}.weight",
                f"{MODULE.probe.TARGET_MODULE}.lora_B.{MODULE.probe.ADAPTER_NAME}.weight",
            )
            events = []
            loaded_models = []
            load_record = SimpleNamespace(
                manifest={"evaluation": expected},
                manifest_sha256="0" * 64,
                payload_sha256="1" * 64,
                trainable_parameter_names=adapter_names,
            )

            checkpoint_manifest_path = (
                checkpoint_path / MODULE.adapter_checkpoint.MANIFEST_NAME
            )
            checkpoint_manifest_path.write_bytes(b"pinned manifest fixture")
            checkpoint_manifest_sha256 = hashlib.sha256(
                checkpoint_manifest_path.read_bytes()
            ).hexdigest()
            args.checkpoint_manifest_sha256 = checkpoint_manifest_sha256
            load_record.manifest_sha256 = checkpoint_manifest_sha256

            def load_transformer(*_args, **_kwargs):
                model = TinyTransformer()
                model.eval()
                loaded_models.append(model)
                events.append(f"load_{len(loaded_models)}")
                return model

            def verify_snapshot(*_args, **_kwargs):
                events.append("verify_snapshot")
                return ({"_class_name": "PinnedTransformer"}, 12)

            def teacher(model, bundle, *_args, **_kwargs):
                model_index = loaded_models.index(model) + 1
                events.append(f"teacher_{model_index}_{bundle.prompt}")
                return torch.zeros(1), torch.zeros(1)

            def install_initial(model, **_kwargs):
                self.assertIs(loaded_models[0], model)
                events.append("install_seeded_adapter")
                return adapter_names

            observed_losses = iter(
                (
                    expected["train_initial_eval_loss"] + 2e-8,
                    expected["heldout_initial_eval_loss"] - 3e-8,
                    expected["train_final_eval_loss"] + 4e-7,
                    expected["heldout_final_eval_loss"] - 4e-7,
                )
            )

            def evaluate(model, bundle, *_args, **_kwargs):
                model_index = loaded_models.index(model) + 1
                phase = "initial" if model_index == 1 else "final"
                split = "train" if bundle is train_bundle else "heldout"
                events.append(f"evaluate_{phase}_{split}")
                return next(observed_losses)

            def bundle_digests(bundle, _label):
                if bundle is train_bundle:
                    return {
                        "manifest_sha256": MODULE.probe.CONDITIONING_MANIFEST_SHA256,
                        "payload_sha256": MODULE.probe.CONDITIONING_PAYLOAD_SHA256,
                    }
                return {
                    "manifest_sha256": args.heldout_manifest_sha256,
                    "payload_sha256": args.heldout_payload_sha256,
                }

            def guard_local_memory(*_args, **_kwargs):
                events.append("guard_memory")
                return {"device_type": "mps"}

            def load_adapter(verified_base, *_args, **_kwargs):
                self.assertIsInstance(
                    verified_base, MODULE.adapter_checkpoint.VerifiedBase
                )
                self.assertIs(loaded_models[1], verified_base.transformer)
                events.append("load_saved_adapter")
                return load_record

            with ExitStack() as patches:
                patches.enter_context(
                    patch.object(
                        MODULE.probe, "load_flow_match_schedule", return_value=schedule
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.probe,
                        "load_conditioning_bundle",
                        return_value=train_bundle,
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.distill,
                        "load_heldout_conditioning_bundle",
                        return_value=heldout_bundle,
                    )
                )
                patches.enter_context(
                    patch.object(MODULE.distill, "_validate_heldout_bundle")
                )
                patches.enter_context(
                    patch.object(
                        MODULE.probe,
                        "require_pinned_diffusers_installation",
                        return_value=MODULE.probe.DIFFUSERS_COMMIT,
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.probe,
                        "require_compatible_peft_installation",
                        return_value=("0.18.0", "/pinned/peft"),
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.probe,
                        "verify_snapshot_integrity",
                        side_effect=verify_snapshot,
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.probe,
                        "_resolve_device",
                        return_value=torch.device("mps"),
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.probe,
                        "_guard_local_memory",
                        side_effect=guard_local_memory,
                    )
                )
                patches.enter_context(
                    patch.object(MODULE.probe, "_set_device_memory_cap")
                )
                patches.enter_context(
                    patch.object(
                        MODULE.probe, "_load_transformer", side_effect=load_transformer
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.probe,
                        "_install_probe_adapter",
                        side_effect=install_initial,
                    )
                )
                patches.enter_context(
                    patch.object(MODULE.distill, "_require_exact_adapter_contract")
                )
                patches.enter_context(
                    patch.object(
                        MODULE.distill, "_compute_teacher_endpoint", side_effect=teacher
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.distill, "_evaluate_student_loss", side_effect=evaluate
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.adapter_checkpoint,
                        "_bundle_digests",
                        side_effect=bundle_digests,
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE.adapter_checkpoint,
                        "load_adapter",
                        side_effect=load_adapter,
                    )
                )
                patches.enter_context(
                    patch.object(
                        MODULE,
                        "_release_model_memory",
                        side_effect=lambda *_args: events.append("release"),
                    )
                )
                report = MODULE.run_replay(args, torch_module=torch)

            self.assertEqual("reference_replay_passed", report["status"])
            self.assertEqual(4, len(report["loss_comparisons"]))
            self.assertTrue(
                all(
                    item["within_tolerance"]
                    for item in report["loss_comparisons"].values()
                )
            )
            self.assertTrue(
                report["heldout_improvement_visibility"][
                    "tolerance_sum_below_improvement"
                ]
            )
            self.assertEqual(
                [
                    "verify_snapshot",
                    "guard_memory",
                    "load_1",
                    "teacher_1_red cube",
                    "teacher_1_A stone castle at dusk",
                    "install_seeded_adapter",
                    "evaluate_initial_train",
                    "evaluate_initial_heldout",
                    "release",
                    "verify_snapshot",
                    "guard_memory",
                    "load_2",
                    "teacher_2_red cube",
                    "teacher_2_A stone castle at dusk",
                    "load_saved_adapter",
                    "evaluate_final_train",
                    "evaluate_final_heldout",
                ],
                events,
            )


if __name__ == "__main__":
    unittest.main()
