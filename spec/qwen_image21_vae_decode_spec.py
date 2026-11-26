#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_image21_vae_decode.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_vae_decode", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QwenImage21VaeDecodeSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "qwen_image21_latents.json"
        self.payload_path = self.root / "qwen_image21_latents.bin"
        self.height = 2
        self.width = 2
        self.values = [
            float((y * self.width + x) * MODULE.CHANNELS + channel)
            for y in range(self.height)
            for x in range(self.width)
            for channel in range(MODULE.CHANNELS)
        ]
        self.payload_path.write_bytes(struct.pack(f"<{len(self.values)}f", *self.values))
        self.write_manifest()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def manifest(self) -> dict[str, object]:
        return {
            "format": MODULE.FORMAT,
            "model_id": MODULE.MODEL_ID,
            "layout": "tokens_hwc",
            "channels": MODULE.CHANNELS,
            "latent_height": self.height,
            "latent_width": self.width,
            "image_height": self.height * MODULE.SPATIAL_SCALE,
            "image_width": self.width * MODULE.SPATIAL_SCALE,
            "dtype": "float32-le",
            "scaling": "diffusers_normalized",
            "payload": self.payload_path.name,
            "payload_bytes": len(self.values) * 4,
        }

    def write_manifest(self, content: dict[str, object] | None = None) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest() if content is None else content), encoding="utf-8"
        )

    def test_unflattens_plain_spatial_tokens_to_vae_ncthw_without_patch_unpacking(self) -> None:
        bundle = MODULE.load_latent_bundle(self.manifest_path)
        vae_input = MODULE.tokens_hwc_to_vae_input(bundle.payload, self.height, self.width)

        self.assertEqual((1, 64, 1, 2, 2), vae_input.shape)
        for y in range(self.height):
            for x in range(self.width):
                token = y * self.width + x
                for channel in range(MODULE.CHANNELS):
                    self.assertEqual(
                        self.values[token * MODULE.CHANNELS + channel],
                        vae_input[0, channel, 0, y, x],
                    )

    def test_rejects_legacy_qwen_image_2x2_patch_layout(self) -> None:
        content = self.manifest()
        content["layout"] = "packed_2x2"
        self.write_manifest(content)

        with self.assertRaisesRegex(ValueError, "layout"):
            MODULE.load_latent_bundle(self.manifest_path)

    def test_rejects_truncated_or_extended_payload(self) -> None:
        self.payload_path.write_bytes(self.payload_path.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, "expected exactly"):
            MODULE.load_latent_bundle(self.manifest_path)

        self.payload_path.write_bytes(struct.pack(f"<{len(self.values)}f", *self.values) + b"x")
        with self.assertRaisesRegex(ValueError, "expected exactly"):
            MODULE.load_latent_bundle(self.manifest_path)

    def test_rejects_non_finite_latents(self) -> None:
        invalid = self.values.copy()
        invalid[17] = math.nan
        self.payload_path.write_bytes(struct.pack(f"<{len(invalid)}f", *invalid))

        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            MODULE.load_latent_bundle(self.manifest_path)

    def test_rejects_wrong_model_scaling_dimensions_and_path_traversal(self) -> None:
        content = self.manifest()
        content["model_id"] = "Qwen/Qwen-Image"
        self.write_manifest(content)
        with self.assertRaisesRegex(ValueError, "model_id"):
            MODULE.load_latent_bundle(self.manifest_path)

        content = self.manifest()
        content["scaling"] = "raw_vae_latents"
        self.write_manifest(content)
        with self.assertRaisesRegex(ValueError, "scaling"):
            MODULE.load_latent_bundle(self.manifest_path)

        content = self.manifest()
        content["latent_height"] = True
        self.write_manifest(content)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            MODULE.load_latent_bundle(self.manifest_path)

        content = self.manifest()
        content["payload"] = "../outside.f32le"
        self.write_manifest(content)
        with self.assertRaisesRegex(ValueError, "filename"):
            MODULE.load_latent_bundle(self.manifest_path)

    def test_rejects_dimension_and_declared_payload_byte_mismatches(self) -> None:
        content = self.manifest()
        content["image_width"] = self.width * MODULE.SPATIAL_SCALE + 16
        self.write_manifest(content)
        with self.assertRaisesRegex(ValueError, "image_width"):
            MODULE.load_latent_bundle(self.manifest_path)

        content = self.manifest()
        content["payload_bytes"] = len(self.values) * 4 + 4
        self.write_manifest(content)
        with self.assertRaisesRegex(ValueError, "payload_bytes"):
            MODULE.load_latent_bundle(self.manifest_path)

    def test_rejects_odd_or_excessive_latent_dimensions(self) -> None:
        content = self.manifest()
        content["latent_height"] = 3
        self.write_manifest(content)
        with self.assertRaisesRegex(ValueError, "both be even"):
            MODULE.load_latent_bundle(self.manifest_path)

        content = self.manifest()
        content["latent_height"] = MODULE.MAX_LATENT_SIDE + 1
        self.write_manifest(content)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            MODULE.load_latent_bundle(self.manifest_path)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("PIL") is not None,
        "PyTorch and Pillow are required for the decode-only CPU fixture",
    )
    def test_decode_uses_diffusers_unnormalization_and_writes_rgba_png(self) -> None:
        import torch
        from PIL import Image

        captured_latents: list[object] = []
        means = [1.25] * MODULE.CHANNELS
        stds = [2.0] * MODULE.CHANNELS

        class FakeVae:
            config = SimpleNamespace(
                z_dim=MODULE.CHANNELS,
                out_channels=4,
                scale_factor_spatial=MODULE.SPATIAL_SCALE,
                latents_mean=means,
                latents_std=stds,
            )

            def to(self, **_kwargs):
                return self

            def eval(self):
                return self

            def decode(self, latents, return_dict=False):
                self.return_dict = return_dict
                captured_latents.append(latents.detach().clone())
                decoded = torch.empty((1, 4, 1, 32, 32), dtype=torch.float32)
                decoded[:, 0].fill_(-1.0)
                decoded[:, 1].fill_(0.0)
                decoded[:, 2].fill_(1.0)
                decoded[:, 3].fill_(0.5)
                return (decoded,)

        vae = FakeVae()
        png_path = self.root / "decoded.png"
        with mock.patch.object(
            MODULE,
            "_load_local_vae",
            return_value=(torch, vae, "cpu", torch.float32),
        ):
            size = MODULE.decode_bundle(
                self.manifest_path, self.root / "local-model", png_path, device="cpu"
            )

        raw = torch.from_numpy(
            MODULE.tokens_hwc_to_vae_input(self.payload_path.read_bytes(), self.height, self.width)
        )
        expected_latents = raw * 2.0 + 1.25
        self.assertTrue(torch.equal(captured_latents[0], expected_latents))
        self.assertFalse(vae.return_dict)
        self.assertEqual(size, (32, 32))
        with Image.open(png_path) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.size, (32, 32))
            self.assertEqual(image.getpixel((0, 0)), (0, 128, 255, 191))

    @unittest.skipUnless(
        os.environ.get("QWEN_IMAGE21_VAE_MODEL_DIR"),
        "set QWEN_IMAGE21_VAE_MODEL_DIR to opt into the local-weight smoke test",
    )
    def test_local_vae_decode_smoke_when_opted_in(self) -> None:
        from PIL import Image

        self.payload_path.write_bytes(b"\x00" * (self.height * self.width * MODULE.CHANNELS * 4))
        self.write_manifest()
        output_path = Path(
            os.environ.get("QWEN_IMAGE21_VAE_SMOKE_OUTPUT", self.root / "vae-smoke.png")
        )
        size = MODULE.decode_bundle(
            self.manifest_path,
            os.environ["QWEN_IMAGE21_VAE_MODEL_DIR"],
            output_path,
            device="cpu",
            dtype_name="float32",
        )
        self.assertEqual(size, (32, 32))
        with Image.open(output_path) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.size, (32, 32))


if __name__ == "__main__":
    unittest.main()
