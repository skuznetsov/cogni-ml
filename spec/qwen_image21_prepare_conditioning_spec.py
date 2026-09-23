#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_image21_prepare_conditioning.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_prepare_conditioning", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QwenImage21PrepareConditioningSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.hidden = np.arange(2 * 4096, dtype=np.float32).reshape(2, 4096)
        self.attention_mask = np.array([True, True], dtype=np.bool_)
        self.image_mask = np.array([False, False], dtype=np.bool_)
        self.latents = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_bundle(self, **overrides):
        arguments = {
            "output_dir": self.output_dir,
            "prompt": "A small red kite above the ocean",
            "revision": "790c92633540aa0cb11d9abf19eb46d861714758",
            "width": 32,
            "height": 32,
            "seed": 7,
            "encoder_hidden_states": self.hidden,
            "encoder_hidden_states_mask": self.attention_mask,
            "encoder_img_mask": self.image_mask,
            "initial_target_latents": self.latents,
            "source_dtype": "bfloat16",
            "device": "cpu",
            "torch_version": "2.6.0",
            "diffusers_version": "0.37.0.dev0",
            "diffusers_commit": "8b3c707ebd3ec4881f4190cf42931da07eaf3b65",
        }
        arguments.update(overrides)
        return MODULE.write_conditioning_bundle(**arguments)

    @staticmethod
    def fake_torch(*, mps_available: bool, cuda_available: bool):
        return SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: mps_available)
            ),
            cuda=SimpleNamespace(is_available=lambda: cuda_available),
        )

    def test_auto_prefers_cpu_to_unvalidated_mps_but_keeps_cuda(self) -> None:
        mps_only = self.fake_torch(mps_available=True, cuda_available=False)
        self.assertEqual("cpu", MODULE._resolve_device(mps_only, "auto"))
        self.assertEqual("mps", MODULE._resolve_device(mps_only, "mps"))

        mps_and_cuda = self.fake_torch(mps_available=True, cuda_available=True)
        self.assertEqual("cuda", MODULE._resolve_device(mps_and_cuda, "auto"))

    def test_mps_eager_attention_is_opt_in_and_restricted_to_mps(self) -> None:
        self.assertEqual({}, MODULE._text_encoder_attention_kwargs("cpu", False))
        self.assertEqual({}, MODULE._text_encoder_attention_kwargs("cuda", False))
        with self.assertRaisesRegex(ValueError, "requires --mps-eager-attention"):
            MODULE._text_encoder_attention_kwargs("mps", False)

        with self.assertWarnsRegex(RuntimeWarning, "not yet quality-validated"):
            self.assertEqual(
                {"attn_implementation": "eager"},
                MODULE._text_encoder_attention_kwargs("mps", True),
            )

        for device in ("cpu", "cuda"):
            with self.subTest(device=device), self.assertRaisesRegex(
                ValueError, "requires --device mps"
            ):
                MODULE._text_encoder_attention_kwargs(device, True)

    def test_writes_versioned_manifest_and_exact_little_endian_payload_layout(self) -> None:
        manifest_path = self.write_bundle()
        self.assertEqual("qwen_image21_conditioning.json", manifest_path.name)
        payload_path = self.output_dir / "qwen_image21_conditioning.bin"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual("qwen-image21-conditioning", manifest["schema"])
        self.assertEqual(1, manifest["schema_version"])
        self.assertEqual("Qwen/Qwen-Image-2.1", manifest["model"]["repo"])
        self.assertEqual("790c92633540aa0cb11d9abf19eb46d861714758", manifest["model"]["revision"])
        self.assertEqual([[1, 2, 2]], manifest["image"]["img_shapes"])
        self.assertEqual(16, manifest["image"]["vae_scale_factor"])
        self.assertEqual(7, manifest["noise"]["seed"])
        self.assertEqual("cpu", manifest["noise"]["generator_device"])
        self.assertEqual(
            "8b3c707ebd3ec4881f4190cf42931da07eaf3b65",
            manifest["runtime"]["diffusers_commit"],
        )
        self.assertEqual("qwen_image21_conditioning.bin", manifest["payload_file"])

        tensors = manifest["tensors"]
        self.assertEqual(
            [
                "encoder_hidden_states",
                "encoder_hidden_states_mask",
                "encoder_img_mask",
                "initial_target_latents",
            ],
            list(tensors),
        )
        self.assertEqual([2, 4096], tensors["encoder_hidden_states"]["shape"])
        self.assertEqual("float32-le", tensors["encoder_hidden_states"]["dtype"])
        self.assertEqual([2], tensors["encoder_hidden_states_mask"]["shape"])
        self.assertEqual("uint8", tensors["encoder_img_mask"]["dtype"])
        self.assertEqual([4, 64], tensors["initial_target_latents"]["shape"])

        first = tensors["encoder_hidden_states"]
        attention = tensors["encoder_hidden_states_mask"]
        image = tensors["encoder_img_mask"]
        latents = tensors["initial_target_latents"]
        self.assertEqual(0, first["offset_bytes"])
        self.assertEqual(2 * 4096 * 4, first["nbytes"])
        self.assertEqual(first["offset_bytes"] + first["nbytes"], attention["offset_bytes"])
        self.assertEqual(attention["offset_bytes"] + attention["nbytes"], image["offset_bytes"])
        self.assertEqual(image["offset_bytes"] + image["nbytes"], latents["offset_bytes"])

        payload = payload_path.read_bytes()
        self.assertEqual(manifest["payload_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(0.0, struct.unpack_from("<f", payload, first["offset_bytes"])[0])
        self.assertEqual(b"\x01\x01", payload[attention["offset_bytes"] : image["offset_bytes"]])
        self.assertEqual(b"\x00\x00", payload[image["offset_bytes"] : latents["offset_bytes"]])
        self.assertEqual(0.0, struct.unpack_from("<f", payload, latents["offset_bytes"])[0])
        self.assertEqual(len(payload), latents["offset_bytes"] + latents["nbytes"])

    def test_rejects_wrong_context_or_target_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "4096"):
            self.write_bundle(encoder_hidden_states=np.zeros((2, 4095), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "must have shape"):
            self.write_bundle(initial_target_latents=np.zeros((3, 64), dtype=np.float32))

    def test_rejects_nonbinary_masks_nonfinite_values_and_bad_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            self.write_bundle(encoder_img_mask=np.array([0, 2], dtype=np.uint8))

        bad_hidden = self.hidden.copy()
        bad_hidden[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            self.write_bundle(encoder_hidden_states=bad_hidden)

        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            self.write_bundle(width=33)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.write_bundle(prompt="")
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            self.write_bundle(width=MODULE.MAX_IMAGE_SIDE + 32)

    def test_refuses_to_overwrite_an_existing_bundle(self) -> None:
        self.write_bundle()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.write_bundle()

    def test_model_directory_rejects_the_legacy_qwen_image_pipeline(self) -> None:
        model_dir = self.output_dir / "wrong-model"
        (model_dir / "text_encoder").mkdir(parents=True)
        (model_dir / "processor").mkdir()
        (model_dir / "model_index.json").write_text(
            json.dumps({
                "_class_name": "QwenImagePipeline",
                "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
            }),
            encoding="utf-8",
        )
        (model_dir / "text_encoder" / "config.json").write_text(
            json.dumps({"model_type": "qwen2_5_vl", "architectures": []}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "QwenImage21Pipeline"):
            MODULE._validate_model_directory(model_dir)


if __name__ == "__main__":
    unittest.main()
