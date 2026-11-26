"""CPU-only contract checks for the stock-generated T0 source oracle."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

SUPPORT = Path(__file__).resolve().parent / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_portrait_dit_control as portrait  # noqa: E402


class T0ContextControlTest(unittest.TestCase):
    def test_generated_plan_denoses_t0_before_writer_and_frozen_t1(self) -> None:
        plan = portrait.generated_context_plan()

        self.assertEqual(len(plan), 9)
        self.assertEqual(
            [call["kind"] for call in plan],
            ["denoise"] * 4 + ["writer"] + ["denoise"] * 4,
        )
        for step, call in enumerate(plan[:4]):
            with self.subTest(step=step):
                self.assertEqual(call["condition"], "t0_shared_scene")
                self.assertEqual(call["step"], step)
                self.assertEqual(call["cache_valid_len"], 0)
                self.assertFalse(call["update_cache"])
                self.assertTrue(call["cache_txt"])
                self.assertEqual(call["rope_bank"], 0)

        writer = plan[4]
        self.assertEqual(writer["condition"], "t0_caption")
        self.assertEqual(writer["cache_valid_len"], 0)
        self.assertTrue(writer["update_cache"])
        self.assertTrue(writer["cache_txt"])
        self.assertEqual(writer["rope_bank"], 0)

        for step, call in enumerate(plan[5:]):
            with self.subTest(t1_step=step):
                self.assertEqual(call["condition"], "t1_shared_scene")
                self.assertEqual(call["step"], step)
                self.assertEqual(call["cache_valid_len"], portrait.CACHE_ROWS)
                self.assertFalse(call["update_cache"])
                self.assertFalse(call["cache_txt"])
                self.assertEqual(call["rope_bank"], 1)

    def test_noise_slice_digest_hashes_only_requested_temporal_frames(self) -> None:
        # [B=1,C=2,T=4,H=1,W=1], channel-major contiguous F32 bytes.
        raw = bytes(range(8))
        with tempfile.TemporaryDirectory() as temp:
            payload = Path(temp) / "payload.bin"
            payload.write_bytes(raw)
            record = {"offset": 0, "shape": [1, 2, 4, 1, 1], "dtype": "U8"}

            actual = portrait.tensor_time_slice_sha256(payload, record, 1, 3, itemsize=1)
            expected = hashlib.sha256(bytes([1, 2, 5, 6])).hexdigest()
            self.assertEqual(actual, expected)

            changed_outside = bytearray(raw)
            changed_outside[0] ^= 0xFF
            payload.write_bytes(changed_outside)
            self.assertEqual(
                portrait.tensor_time_slice_sha256(payload, record, 1, 3, itemsize=1),
                expected,
            )

            changed_inside = bytearray(changed_outside)
            changed_inside[1] ^= 0xFF
            payload.write_bytes(changed_inside)
            self.assertNotEqual(
                portrait.tensor_time_slice_sha256(payload, record, 1, 3, itemsize=1),
                expected,
            )

    def test_native_endpoint_is_optional_but_path_and_digest_are_a_pair(self) -> None:
        self.assertIsNone(portrait.admit_native_endpoint(None, None, "T0"))
        with self.assertRaisesRegex(ValueError, "requires both"):
            portrait.admit_native_endpoint(Path("unused.f32le"), None, "T0")
        with self.assertRaisesRegex(ValueError, "requires both"):
            portrait.admit_native_endpoint(None, "0" * 64, "T0")

    def test_generated_context_rejects_history_mask_interventions(self) -> None:
        portrait.validate_t0_context("photo")
        portrait.validate_t0_context("generated")
        with self.assertRaisesRegex(ValueError, "ordinary unmasked"):
            portrait.validate_t0_context("generated", keep_latents=30)
        with self.assertRaisesRegex(ValueError, "ordinary unmasked"):
            portrait.validate_t0_context("generated", hide_caption=True)
        with self.assertRaisesRegex(ValueError, "ordinary unmasked"):
            portrait.validate_t0_context("generated", current_text_kv=True)

    def test_receipt_identity_covers_fixed_noise_text_rope_and_writer_epsilon(self) -> None:
        latent = bytes(range(240))
        qwen = b"fixed qwen state"
        rope_cos = bytes(range(24))
        rope_sin = bytes(range(24, 48))
        epsilon = bytes(range(12))
        with tempfile.TemporaryDirectory() as temp:
            payload = Path(temp) / "inputs.bin"
            payload.write_bytes(latent + qwen + rope_cos + rope_sin + epsilon)
            offset = len(latent)
            offset += len(qwen)
            cos_offset = offset
            offset += len(rope_cos)
            sin_offset = offset
            offset += len(rope_sin)
            epsilon_offset = offset
            identity = portrait.frozen_input_identity(dict(
                payload_path=payload,
                latent_record={"offset": 0, "shape": [1, 1, 60, 1, 1]},
                qwen_record={"sha256": hashlib.sha256(qwen).hexdigest()},
                rope_cos_record={"offset": cos_offset, "shape": [6, 1, 1]},
                rope_sin_record={"offset": sin_offset, "shape": [6, 1, 1]},
                cache_epsilon_record={"offset": epsilon_offset, "shape": [6, 1, 1, 1, 1, 1]},
            ))

        self.assertEqual(identity["t0_initial_noise_sha256"], hashlib.sha256(latent[:120]).hexdigest())
        self.assertEqual(identity["t1_initial_noise_sha256"], hashlib.sha256(latent[120:]).hexdigest())
        self.assertEqual(identity["scene_qwen_bf16_sha256"], hashlib.sha256(qwen).hexdigest())
        self.assertEqual(identity["t0_rope_cos_sha256"], hashlib.sha256(rope_cos[:4]).hexdigest())
        self.assertEqual(identity["t1_rope_cos_sha256"], hashlib.sha256(rope_cos[4:8]).hexdigest())
        self.assertEqual(identity["t0_rope_sin_sha256"], hashlib.sha256(rope_sin[:4]).hexdigest())
        self.assertEqual(identity["t1_rope_sin_sha256"], hashlib.sha256(rope_sin[4:8]).hexdigest())
        self.assertEqual(identity["t0_cache_epsilon_bf16_sha256"], hashlib.sha256(epsilon[:2]).hexdigest())

    def test_native_endpoint_requires_exact_segment_extent_and_sha(self) -> None:
        raw = bytes(portrait.LATENT_SEGMENT_BYTES)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "t0.f32le"
            path.write_bytes(raw)
            endpoint = portrait.admit_native_endpoint(path, digest, "T0")
            self.assertEqual(endpoint["sha256"], digest)
            self.assertEqual(endpoint["shape"], [1, 32, 30, 9, 16])

            with self.assertRaisesRegex(ValueError, "shape or digest differs"):
                portrait.admit_native_endpoint(path, "0" * 64, "T0")

            path.write_bytes(raw[:-4])
            with self.assertRaisesRegex(ValueError, "shape or digest differs"):
                portrait.admit_native_endpoint(path, digest, "T0")


if __name__ == "__main__":
    unittest.main()
