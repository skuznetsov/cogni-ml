from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen_image21_lora_adapter_checkpoint as MODULE  # noqa: E402


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


class QwenImage21LoraAdapterCheckpointSpec(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_bundle = self._write_bundle("train", b"red-cube-train-payload")
        self.heldout_bundle = self._write_bundle("heldout", b"heldout-payload")
        self.flow_fixture = (
            ROOT / "spec/fixtures/qwen_image21_flow_match_diffusers.json"
        )
        self.output = self.root / "checkpoint"

    def _write_bundle(self, name: str, payload: bytes):
        payload_path = self.root / f"{name}.bin"
        payload_path.write_bytes(payload)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        manifest = {
            "schema": "qwen-image21-conditioning",
            "schema_version": 1,
            "model": {
                "repo": "Qwen/Qwen-Image-2.1",
                "revision": MODULE.probe.MODEL_REVISION,
            },
            "runtime": {"diffusers_commit": MODULE.probe.DIFFUSERS_COMMIT},
            "payload_file": payload_path.name,
            "payload_sha256": payload_sha256,
        }
        manifest_path = self.root / f"{name}.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return SimpleNamespace(
            manifest_path=manifest_path, payload_sha256=payload_sha256
        )

    def _trained_model(self):
        model = TinyTransformer()
        names = MODULE.probe._install_probe_adapter(
            model, adapter_dtype=torch.float32, device=torch.device("cpu")
        )
        with torch.no_grad():
            named = dict(model.named_parameters())
            named[names[0]].copy_(
                torch.linspace(-0.5, 0.5, named[names[0]].numel()).reshape_as(
                    named[names[0]]
                )
            )
            named[names[1]].copy_(
                torch.linspace(0.5, 1.5, named[names[1]].numel()).reshape_as(
                    named[names[1]]
                )
            )
        return model, names

    def _training_result(self, names, *, steps=2):
        heldout_evaluation = SimpleNamespace(
            conditioning_manifest_sha256=hashlib.sha256(
                self.heldout_bundle.manifest_path.read_bytes()
            ).hexdigest(),
            conditioning_payload_sha256=self.heldout_bundle.payload_sha256,
            initial_eval_loss=1.25,
            final_eval_loss=0.75,
            teacher_target_detached=True,
        )
        return SimpleNamespace(
            trainable_parameter_names=names,
            steps=tuple(object() for _ in range(steps)),
            initial_eval_loss=2.0,
            final_eval_loss=1.0,
            base_parameters_have_no_grad=True,
            teacher_target_detached=True,
            heldout_evaluation=heldout_evaluation,
        )

    def _export(self, model=None, names=None):
        if model is None:
            model, actual_names = self._trained_model()
            names = actual_names if names is None else names
        result = self._training_result(names)
        record = MODULE.export_adapter(
            model,
            self.output,
            train_bundle=self.train_bundle,
            heldout_bundle=self.heldout_bundle,
            flow_fixture_path=self.flow_fixture,
            training_result=result,
        )
        return model, names, record

    def _load(self, model=None):
        model = TinyTransformer() if model is None else model
        with (
            patch.object(
                MODULE.probe,
                "require_pinned_diffusers_installation",
                return_value=MODULE.probe.DIFFUSERS_COMMIT,
            ),
            patch.object(
                MODULE.probe,
                "verify_snapshot_integrity",
                return_value=({"config": "pinned"}, 123),
            ),
            patch.object(MODULE.probe, "_load_transformer", return_value=model),
        ):
            verified_base = MODULE.load_verified_base(
                self.root,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        record = MODULE.load_adapter(
            verified_base,
            self.output,
            train_bundle=self.train_bundle,
            heldout_bundle=self.heldout_bundle,
            flow_fixture_path=self.flow_fixture,
            device=torch.device("cpu"),
        )
        return model, record

    def test_round_trip_reloads_exact_rank4_fp32_adapter_and_reports_hashes(self):
        trained, names, exported = self._export()

        self.assertEqual(
            {"adapter.safetensors", "manifest.json"},
            {p.name for p in self.output.iterdir()},
        )
        self.assertEqual(
            MODULE.probe.MODEL_REVISION, exported.manifest["source_revision"]
        )
        self.assertEqual(
            MODULE.probe.DIFFUSERS_COMMIT, exported.manifest["diffusers_commit"]
        )
        self.assertEqual(2, exported.manifest["training"]["optimizer_steps"])
        self.assertEqual(
            MODULE.probe.ADAPTER_INIT_SEED,
            exported.manifest["training"]["adapter_init_seed"],
        )
        self.assertEqual(
            2.0, exported.manifest["evaluation"]["train_initial_eval_loss"]
        )
        self.assertEqual(
            0.75, exported.manifest["evaluation"]["heldout_final_eval_loss"]
        )
        self.assertEqual(1.0, exported.manifest["adapter"]["alpha_over_rank"])
        self.assertEqual(
            hashlib.sha256(self.train_bundle.manifest_path.read_bytes()).hexdigest(),
            exported.manifest["bundles"]["train"]["manifest_sha256"],
        )
        self.assertEqual(
            self.heldout_bundle.payload_sha256,
            exported.manifest["bundles"]["heldout"]["payload_sha256"],
        )
        self.assertEqual(
            MODULE.probe.FIXTURE_SHA256, exported.manifest["flow_fixture_sha256"]
        )
        self.assertEqual("float32", exported.manifest["tensors"][names[0]]["dtype"])
        self.assertEqual(
            hashlib.sha256((self.output / "manifest.json").read_bytes()).hexdigest(),
            exported.manifest_sha256,
        )

        reloaded, loaded = self._load()
        self.assertEqual(names, loaded.trainable_parameter_names)
        self.assertEqual(exported.payload_sha256, loaded.payload_sha256)
        self.assertEqual(exported.manifest_sha256, loaded.manifest_sha256)
        source = dict(trained.named_parameters())
        actual = dict(reloaded.named_parameters())
        for name in names:
            self.assertTrue(torch.equal(source[name], actual[name]), name)
            self.assertEqual(torch.float32, actual[name].dtype)
            self.assertTrue(actual[name].requires_grad)
        self.assertEqual(
            1.0,
            dict(reloaded.named_modules())[MODULE.probe.TARGET_MODULE].scaling[
                MODULE.probe.ADAPTER_NAME
            ],
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in actual.items()
                if name not in names
            )
        )

    def test_safe_loader_requires_verified_base_and_constructs_it_from_snapshot(self):
        self._export()
        events = []
        constructed = TinyTransformer()

        def verify(model_dir):
            events.append(("verify", Path(model_dir)))
            return (
                {
                    "_class_name": "QwenImage21Transformer2DModel",
                    "axes_dims_rope": [16, 56, 56],
                },
                123,
            )

        def require_diffusers():
            events.append(("runtime",))
            return MODULE.probe.DIFFUSERS_COMMIT

        def before_load(config, checkpoint_bytes):
            events.append(("guard", config, checkpoint_bytes))

        def load_transformer(model_dir, device, dtype):
            events.append(("load", Path(model_dir), device, dtype))
            return constructed

        with (
            patch.object(
                MODULE.probe,
                "require_pinned_diffusers_installation",
                side_effect=require_diffusers,
            ),
            patch.object(MODULE.probe, "verify_snapshot_integrity", side_effect=verify),
            patch.object(
                MODULE.probe, "_load_transformer", side_effect=load_transformer
            ),
        ):
            verified_base = MODULE.load_verified_base(
                self.root,
                device=torch.device("cpu"),
                dtype=torch.float32,
                before_load=before_load,
            )
            self.assertIs(constructed, verified_base.transformer)
            self.assertEqual(
                "QwenImage21Transformer2DModel", verified_base.config["_class_name"]
            )
            self.assertEqual((16, 56, 56), verified_base.config["axes_dims_rope"])
            with self.assertRaises(TypeError):
                verified_base.config["_class_name"] = "WrongModel"
            loaded = MODULE.load_adapter(
                verified_base,
                self.output,
                train_bundle=self.train_bundle,
                heldout_bundle=self.heldout_bundle,
                flow_fixture_path=self.flow_fixture,
                device=torch.device("cpu"),
            )

        self.assertEqual(["runtime", "verify", "guard", "load"], [e[0] for e in events])
        self.assertEqual(
            loaded.trainable_parameter_names, tuple(sorted(MODULE._EXPECTED_TENSORS))
        )

        # Identical module dimensions are not evidence of pinned base provenance.
        wrong_same_shape_base = TinyTransformer()
        with self.assertRaisesRegex(AttributeError, "immutable"):
            verified_base.transformer = wrong_same_shape_base
        with self.assertRaisesRegex(TypeError, "VerifiedBase"):
            MODULE.load_adapter(
                wrong_same_shape_base,
                self.output,
                train_bundle=self.train_bundle,
                heldout_bundle=self.heldout_bundle,
                flow_fixture_path=self.flow_fixture,
                device=torch.device("cpu"),
            )
        self.assertFalse(
            hasattr(
                dict(wrong_same_shape_base.named_modules())[MODULE.probe.TARGET_MODULE],
                "lora_A",
            )
        )

    def test_safe_loader_rejects_unpinned_diffusers_before_snapshot_or_model_load(self):
        verify = patch.object(MODULE.probe, "verify_snapshot_integrity")
        load = patch.object(MODULE.probe, "_load_transformer")
        with (
            patch.object(
                MODULE.probe,
                "require_pinned_diffusers_installation",
                return_value="0" * 40,
            ),
            verify as verify_snapshot,
            load as load_transformer,
        ):
            with self.assertRaisesRegex(RuntimeError, "Diffusers commit"):
                MODULE.load_verified_base(
                    self.root,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
        verify_snapshot.assert_not_called()
        load_transformer.assert_not_called()

    def test_adapter_loader_checks_compatible_peft_before_injection(self):
        self._export()
        model = TinyTransformer()
        with (
            patch.object(
                MODULE.probe,
                "require_pinned_diffusers_installation",
                return_value=MODULE.probe.DIFFUSERS_COMMIT,
            ),
            patch.object(
                MODULE.probe, "verify_snapshot_integrity", return_value=({}, 1)
            ),
            patch.object(MODULE.probe, "_load_transformer", return_value=model),
        ):
            verified_base = MODULE.load_verified_base(
                self.root,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        with (
            patch.object(
                MODULE.probe,
                "require_compatible_peft_installation",
                side_effect=RuntimeError("untrusted PEFT runtime"),
            ),
            self.assertRaisesRegex(RuntimeError, "untrusted PEFT runtime"),
        ):
            MODULE.load_adapter(
                verified_base,
                self.output,
                train_bundle=self.train_bundle,
                heldout_bundle=self.heldout_bundle,
                flow_fixture_path=self.flow_fixture,
                device=torch.device("cpu"),
            )
        self.assertFalse(
            hasattr(dict(model.named_modules())[MODULE.probe.TARGET_MODULE], "lora_A")
        )

    def test_export_rejects_existing_output(self):
        self._export()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self._export()

    def test_export_publishes_payload_before_manifest(self):
        model, names = self._trained_model()
        result = self._training_result(names)
        original_rename = MODULE.os.rename
        final_renames = []

        def recording_rename(source, destination):
            destination = Path(destination)
            if destination.parent == self.output:
                final_renames.append(destination.name)
            return original_rename(source, destination)

        with patch.object(MODULE.os, "rename", side_effect=recording_rename):
            MODULE.export_adapter(
                model,
                self.output,
                train_bundle=self.train_bundle,
                heldout_bundle=self.heldout_bundle,
                flow_fixture_path=self.flow_fixture,
                training_result=result,
            )
        self.assertEqual(["adapter.safetensors", "manifest.json"], final_renames)

    def test_load_rejects_incomplete_payload_manifest_pair(self):
        self._export()
        (self.output / "adapter.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete|missing"):
            self._load()

    def test_export_rejects_incomplete_trainable_pair(self):
        model, names = self._trained_model()
        with self.assertRaisesRegex(ValueError, "exact LoRA A/B"):
            self._export(model, names[:1])
        self.assertFalse(self.output.exists())

    def test_export_rejects_nonfinite_adapter_weights(self):
        model, names = self._trained_model()
        with torch.no_grad():
            dict(model.named_parameters())[names[0]][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._export(model, names)
        self.assertFalse(self.output.exists())

    def test_export_rejects_unverified_or_nonfinite_training_result(self):
        model, names = self._trained_model()
        result = self._training_result(names)
        result.heldout_evaluation = None
        with self.assertRaisesRegex(ValueError, "held-out evaluation"):
            MODULE.export_adapter(
                model,
                self.output,
                train_bundle=self.train_bundle,
                heldout_bundle=self.heldout_bundle,
                flow_fixture_path=self.flow_fixture,
                training_result=result,
            )

        result = self._training_result(names)
        result.final_eval_loss = float("nan")
        with self.assertRaisesRegex(ValueError, "final training loss.*finite"):
            MODULE.export_adapter(
                model,
                self.output,
                train_bundle=self.train_bundle,
                heldout_bundle=self.heldout_bundle,
                flow_fixture_path=self.flow_fixture,
                training_result=result,
            )

        result = self._training_result(names)
        result.base_parameters_have_no_grad = False
        with self.assertRaisesRegex(ValueError, "frozen base"):
            MODULE.export_adapter(
                model,
                self.output,
                train_bundle=self.train_bundle,
                heldout_bundle=self.heldout_bundle,
                flow_fixture_path=self.flow_fixture,
                training_result=result,
            )
        self.assertFalse(self.output.exists())

    def test_load_rejects_source_revision_and_bundle_hash_mismatch(self):
        self._export()
        manifest_path = self.output / "manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest["source_revision"] = "0" * 40
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "revision"):
            self._load()

        manifest_path.write_bytes(original)
        self.heldout_bundle.payload_sha256 = "0" * 64
        with self.assertRaisesRegex(
            ValueError, "heldout bundle payload SHA256 mismatch"
        ):
            self._load()

    def test_load_rejects_extra_tensor_even_with_matching_payload_hash(self):
        self._export()
        payload_path = self.output / "adapter.safetensors"
        tensors = load_file(payload_path, device="cpu")
        tensors["unexpected.weight"] = torch.ones((1,), dtype=torch.float32)
        save_file(tensors, payload_path)
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["payload"]["sha256"] = hashlib.sha256(
            payload_path.read_bytes()
        ).hexdigest()
        manifest["payload"]["nbytes"] = payload_path.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
            ValueError, "tensor set|unexpected tensor|extra tensor"
        ):
            self._load()

    def test_load_rejects_tensor_hash_mismatch(self):
        self._export()
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_name = next(iter(manifest["tensors"]))
        manifest["tensors"][first_name]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tensor SHA256 mismatch"):
            self._load()

    def test_load_rejects_malformed_manifest_and_malformed_safetensors(self):
        self._export()
        manifest_path = self.output / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        manifest_path.write_bytes(b"{")
        with self.assertRaisesRegex(ValueError, "cannot parse"):
            self._load()
        manifest_path.write_bytes(original_manifest)

        payload_path = self.output / "adapter.safetensors"
        payload_path.write_bytes(b"not a safetensors payload")
        manifest = json.loads(original_manifest)
        manifest["payload"]["sha256"] = hashlib.sha256(
            payload_path.read_bytes()
        ).hexdigest()
        manifest["payload"]["nbytes"] = payload_path.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "safetensors payload is malformed"):
            self._load()

    def test_export_rejects_changed_bundle_payload_and_flow_fixture(self):
        self.train_bundle.manifest_path.with_name("train.bin").write_bytes(b"changed")
        model, names = self._trained_model()
        with self.assertRaisesRegex(ValueError, "payload SHA256 mismatch"):
            self._export(model, names)
        self.assertFalse(self.output.exists())

        self.train_bundle = self._write_bundle("train2", b"valid-train-payload")
        bad_fixture = self.root / "bad-flow.json"
        bad_fixture.write_bytes(self.flow_fixture.read_bytes() + b" ")
        result = self._training_result(names, steps=1)
        with self.assertRaisesRegex(ValueError, "flow fixture SHA256 mismatch"):
            MODULE.export_adapter(
                model,
                self.output,
                train_bundle=self.train_bundle,
                heldout_bundle=self.heldout_bundle,
                flow_fixture_path=bad_fixture,
                training_result=result,
            )
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
