"""CPU-only admission checks for the connected RGB-history T0 source oracle."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SUPPORT = Path(__file__).resolve().parent / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_t0_rgb_dit_control as control  # noqa: E402


class RGBHistoryAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_receipt = {
            "schema": "cogni_ml.hy_nomas_decode_latent/v1",
            "upstream_revision": control.UPSTREAM_REVISION,
            "output_bytes": control.SOURCE_BYTES,
            "output_sha256": control.SOURCE_SHA256,
            "output_dtype": "F32 little-endian planar RGB",
            "output_shape": list(control.SOURCE_SHAPE),
            "frames": 360,
            "width": 512,
            "height": 288,
            "full_upstream_video_parity": False,
        }
        self.frozen = {
            "t0_initial_noise_sha256": "1" * 64,
            "t1_initial_noise_sha256": "2" * 64,
            "scene_qwen_bf16_sha256": "3" * 64,
            "t0_rope_cos_sha256": "4" * 64,
            "t0_rope_sin_sha256": "5" * 64,
            "t1_rope_cos_sha256": "6" * 64,
            "t1_rope_sin_sha256": "7" * 64,
            "t0_cache_epsilon_bf16_sha256": "8" * 64,
        }
        self.native_receipt = {
            "schema": "cogni_ml.hy_nomas_t0_rgb_history_probe/v1",
            "context_mode": "moving",
            "frozen_input_identity": copy.deepcopy(self.frozen),
            "metrics": {
                "chain_preterminal_allocations": 0,
                "chain_preterminal_readbacks": 0,
                "chain_preterminal_waits": 0,
                "terminal_waits": 1,
            },
            "output": {
                "latent_bank": "/run/latent-bank.f32le",
                "latent_bank_sha256": "9" * 64,
                "t0_sha256": "a" * 64,
                "t1_sha256": "b" * 64,
                "prefix60_sha256": "d" * 64,
                "normalized_clip_bf16": "/run/input.bf16le",
                "normalized_clip_bf16_sha256": control.rgb_artifacts.EXPECTED_CLIP_SHA256["moving"],
                "dcae_latent_f32": "/run/t0.f32le",
                "dcae_latent_f32_sha256": "a" * 64,
            },
        }

    def test_context_modes_are_explicit_and_closed(self) -> None:
        self.assertEqual(control.validate_context_mode("moving"), "moving")
        self.assertEqual(control.validate_context_mode("repeat-last"), "repeat-last")
        for invalid in ("generated", "static", "repeat", "MOVING", ""):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "context-mode"):
                control.validate_context_mode(invalid)

    def test_decoder_receipt_binds_exact_frozen_pixel_payload(self) -> None:
        contract = control.validate_decoder_receipt_document(self.source_receipt)
        self.assertEqual(contract["pixels_sha256"], control.SOURCE_SHA256)
        self.assertEqual(contract["shape"], list(control.SOURCE_SHAPE))
        for key, value in (
            ("output_bytes", control.SOURCE_BYTES - 4),
            ("output_sha256", "0" * 64),
            ("output_dtype", "F32 interleaved RGB"),
            ("full_upstream_video_parity", True),
        ):
            invalid = dict(self.source_receipt, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "decoder receipt"):
                control.validate_decoder_receipt_document(invalid)

    def test_native_receipt_requires_matching_mode_frozen_inputs_and_quiescent_chain(self) -> None:
        result = control.validate_native_receipt_document(
            self.native_receipt,
            mode="moving",
            frozen_input_identity=self.frozen,
            native_t1_sha256="b" * 64,
        )
        self.assertEqual(result["output"]["dcae_latent_f32_sha256"], "a" * 64)
        self.assertEqual(result["context_mode"], "moving")

        for mutation, pattern in (
            (lambda item: item.update(context_mode="repeat-last"), "context-mode"),
            (lambda item: item.update(frozen_input_identity={}), "frozen input identity"),
            (lambda item: item["metrics"].update(chain_preterminal_waits=1), "terminal chain"),
            (lambda item: item["metrics"].update(terminal_waits=2), "terminal chain"),
            (lambda item: item["output"].update(t1_sha256="d" * 64), "T1 digest"),
            (lambda item: item["output"].update(normalized_clip_bf16_sha256="c" * 64),
             "independent input expectation"),
        ):
            invalid = copy.deepcopy(self.native_receipt)
            mutation(invalid)
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                control.validate_native_receipt_document(
                    invalid,
                    mode="moving",
                    frozen_input_identity=self.frozen,
                    native_t1_sha256="b" * 64,
                )

    def test_native_artifact_admission_rechecks_180_frame_bank_and_logical_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = np.arange(np.prod(control.LATENT_BANK_SHAPE), dtype="<f4").reshape(
                control.LATENT_BANK_SHAPE
            )
            bank_path = root / "latent-bank.f32le"
            bank.tofile(bank_path)
            t0_path = root / "dcae-t0.f32le"
            t1_path = root / "t1.f32le"
            bank[:, :, :30].tofile(t0_path)
            bank[:, :, 30:60].tofile(t1_path)
            clip_path = root / "normalized.bf16le"
            with clip_path.open("wb") as stream:
                stream.truncate(control.CLIP_BYTES)

            output = {
                "latent_bank": str(bank_path),
                "latent_bank_sha256": control._sha256(bank_path),
                "t0_sha256": control._sha256(t0_path),
                "t1_sha256": control._sha256(t1_path),
                "prefix60_sha256": hashlib.sha256(
                    bank[:, :, :60].tobytes(order="C")
                ).hexdigest(),
                "normalized_clip_bf16": str(clip_path),
                "normalized_clip_bf16_sha256": control._sha256(clip_path),
                "dcae_latent_f32": str(t0_path),
                "dcae_latent_f32_sha256": control._sha256(t0_path),
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps({
                "schema": control.NATIVE_SCHEMA,
                "context_mode": "moving",
                "frozen_input_identity": self.frozen,
                "metrics": {
                    "chain_preterminal_allocations": 0,
                    "chain_preterminal_readbacks": 0,
                    "chain_preterminal_waits": 0,
                    "terminal_waits": 1,
                },
                "output": output,
            }))

            receipt_sha = control._sha256(receipt_path)
            with patch.dict(
                control.rgb_artifacts.EXPECTED_CLIP_SHA256,
                {"moving": output["normalized_clip_bf16_sha256"]},
            ):
                result = control.admit_native_artifacts(
                    receipt_path, receipt_sha, "moving", self.frozen,
                    t1_path, output["t1_sha256"],
                )
            self.assertEqual(result["context_mode"], "moving")
            self.assertNotEqual(output["prefix60_sha256"], output["latent_bank_sha256"])

            with bank_path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"bad!")
            with self.assertRaisesRegex(ValueError, "native latent bank extent or SHA-256"):
                with patch.dict(
                    control.rgb_artifacts.EXPECTED_CLIP_SHA256,
                    {"moving": output["normalized_clip_bf16_sha256"]},
                ):
                    control.admit_native_artifacts(
                        receipt_path, receipt_sha, "moving", self.frozen,
                        t1_path, output["t1_sha256"],
                    )


if __name__ == "__main__":
    unittest.main()
