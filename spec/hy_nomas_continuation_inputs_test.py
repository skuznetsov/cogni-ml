#!/usr/bin/env python3
"""CPU-only contract tests for the six-segment continuation input bundle."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_continuation_inputs as inputs  # noqa: E402
import hy_nomas_generation_inputs as generation_inputs  # noqa: E402


class _FakeDevice:
    type = "mps"


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.device = _FakeDevice()
        self.slices = []
        self.contiguous_called = False

    def __getitem__(self, index):
        self.slices.append(index)
        shape = list(self.shape)
        for axis, item in enumerate(index):
            if isinstance(item, slice):
                start, stop, step = item.indices(shape[axis])
                shape[axis] = len(range(start, stop, step))
        return _FakeTensor(shape)

    def contiguous(self):
        self.contiguous_called = True
        return self


class _FakeTypedTensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.device = _FakeDevice()
        self.dtype = dtype

    def to(self, *, dtype):
        return _FakeTypedTensor(self.shape, dtype)


class _FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed


class _FakeTorchForLatents:
    float32 = object()

    def __init__(self):
        self.generator = None
        self.randn_args = None
        self.full_bank = None

    def Generator(self, *, device):
        self.generator = _FakeGenerator(device)
        return self.generator

    def randn(self, shape, *, generator, device, dtype):
        self.randn_args = (tuple(shape), generator, device, dtype)
        self.full_bank = _FakeTensor(shape)
        return self.full_bank


class _FakeTorchForCache:
    float32 = object()
    bfloat16 = object()

    def __init__(self):
        self.generators = []
        self.draw_calls = []

    def Generator(self, *, device):
        generator = _FakeGenerator(device)
        self.generators.append(generator)
        return generator

    def randn(self, shape, *, generator, device, dtype):
        self.draw_calls.append((tuple(shape), generator, device, dtype))
        return _FakeTypedTensor(shape, dtype)

    def stack(self, tensors, *, dim):
        self.stacked = list(tensors)
        self.stack_dim = dim
        return _FakeTypedTensor((len(tensors), *tensors[0].shape), tensors[0].dtype)


def _valid_document() -> dict:
    prompt = "authenticated continuation prompt"
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    records = {}
    offset = 0
    for name, dtype, shape in inputs.TENSOR_SPECS:
        size = inputs.tensor_nbytes(dtype, shape)
        records[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "offset": offset,
            "nbytes": size,
            "sha256": "a" * 64,
        }
        offset += size
    qwen_source = {
        "prompt_sha256": prompt_sha,
        "prompt_contract": {"prompt": prompt, "prompt_sha256": prompt_sha},
        "state_index": 26,
        "crop_tokens": 1280,
        "hidden_size": 3584,
        "upstream_revision": inputs.PINNED_REVISION,
    }
    qwen_bundle = {
        "manifest_path": "/tmp/inputs-detailed.json",
        "manifest_sha256": "b" * 64,
        "payload_path": "/tmp/inputs-detailed.tensors.bin",
        "payload_sha256": "c" * 64,
        "qwen_tensor_sha256": records["input.qwen_bf16"]["sha256"],
        "first_rope_cos_sha256": "d" * 64,
        "first_rope_sin_sha256": "e" * 64,
        "first_rope_atol": inputs.ROPE_ATOL,
        "first_rope_rtol": inputs.ROPE_RTOL,
    }
    return {
        "schema": inputs.SCHEMA,
        "schema_version": inputs.SCHEMA_VERSION,
        "payload": "continuation.tensors.bin",
        "payload_bytes": offset,
        "payload_sha256": "f" * 64,
        "tensors": records,
        "geometry": inputs._geometry(),
        "prompt": prompt,
        "qwen_source": qwen_source,
        "qwen_input_bundle": qwen_bundle,
        "noise_seeds": inputs._noise_seeds(),
        "cache_writer": inputs._cache_writer_contract(),
        "upstream_revision": inputs.PINNED_REVISION,
        "upstream_sources": copy.deepcopy(inputs.PINNED_UPSTREAM_SOURCE_SHA256),
        "producer": {
            "device": "mps",
            "mps_fallback": "disabled",
            "model_inference": False,
            "tensor_generation_device": "mps",
            "host_export": "one packed terminal MPS-to-host readback",
            "torch_version": "2.6.0",
            "python_version": "3.12.0",
        },
    }


class HyNomasQwenInputAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "alternate.json"
        self.payload_path = self.root / "alternate.tensors.bin"
        reference_path = self.root / "reference.json"
        sidecar_path = self.root / "reference.bf16le"
        qwen = bytes(range(8))
        sidecar = b"\0" * (26 * len(qwen)) + qwen
        sidecar_path.write_bytes(sidecar)
        prompt_contract = {"prompt": "synthetic admission fixture"}
        reference_path.write_text(json.dumps({
            "schema": "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference",
            "schema_version": 1,
            "hy_nomas_prompt": prompt_contract,
            "sidecars": {"bf16": {
                "path": sidecar_path.name,
                "sha256": hashlib.sha256(sidecar).hexdigest(),
                "nbytes": len(sidecar),
                "encoding": "bfloat16-le",
            }},
        }), encoding="utf-8")
        payload = qwen + b"cos0sin0"
        self.payload_path.write_bytes(payload)
        records = {}
        for name, offset, size in (("input.qwen_bf16", 0, 8),
                                   ("input.rope_cos", 8, 4), ("input.rope_sin", 12, 4)):
            records[name] = {
                "dtype": "BF16" if name == "input.qwen_bf16" else "F32",
                "shape": [1, 2, 2] if name == "input.qwen_bf16" else [1],
                "offset": offset,
                "nbytes": size,
                "sha256": hashlib.sha256(payload[offset:offset + size]).hexdigest(),
            }
        self.document = {
            "schema": "cogni_ml.hy_nomas_generation_inputs/v2",
            "upstream_revision": inputs.PINNED_REVISION,
            "payload": self.payload_path.name,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "tensors": records,
            "qwen_source": {
                "reference_path": str(reference_path),
                "reference_sha256": inputs._hash_file(reference_path),
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": hashlib.sha256(sidecar).hexdigest(),
                "sidecar_bytes": len(sidecar),
                "prompt_contract": prompt_contract,
                "state_index": 26,
                "raw_sequence_length": 2,
                "crop": [0, 2],
            },
        }
        self.write_manifest()
        # Generation-schema checks have their own suite. Keep this unit fixture
        # small while exercising continuation's real file, digest, and range checks.
        self.enterContext(mock.patch.object(generation_inputs, "validate_document"))
        self.enterContext(mock.patch.object(inputs, "QWEN_SHAPE", (1, 2, 2)))
        self.enterContext(mock.patch.object(
            inputs, "DEFAULT_QWEN_INPUTS", self.root / "removed-default" / "inputs.json"
        ))

    def write_manifest(self):
        self.path.write_text(json.dumps(self.document), encoding="utf-8")

    def admit(self):
        return inputs._admit_qwen_inputs(self.path, inputs._hash_file(self.path))

    def test_explicit_authenticated_bundle_does_not_require_historical_default(self):
        self.assertFalse(inputs.DEFAULT_QWEN_INPUTS.parent.exists())
        source = self.admit()
        self.assertEqual(source["path"], str(self.path))
        self.assertEqual(source["payload_sha256"], self.document["payload_sha256"])
        self.assertEqual(source["qwen_sidecar_sha256"],
                         self.document["tensors"]["input.qwen_bf16"]["sha256"])

    def test_default_location_retains_manifest_pin_through_a_directory_alias(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for default in (self.path, alias / self.path.name):
            with self.subTest(default=default), mock.patch.object(inputs, "DEFAULT_QWEN_INPUTS", default):
                with self.assertRaisesRegex(inputs.InputsError, "source manifest is not the pinned artifact"):
                    self.admit()

    def test_default_location_retains_payload_pin_after_manifest_authentication(self):
        with mock.patch.object(inputs, "DEFAULT_QWEN_INPUTS", self.path), \
                mock.patch.object(inputs, "_MANIFEST_SHA256", inputs._hash_file(self.path)):
            with self.assertRaisesRegex(inputs.InputsError, "source payload is not the pinned artifact"):
                self.admit()

    def test_alternate_still_requires_the_explicit_manifest_digest(self):
        for sha, message in (("0" * 64, "differs from the admitted digest"),
                             ("A" * 64, "64 lowercase hex characters")):
            with self.subTest(sha=sha), self.assertRaisesRegex(inputs.InputsError, message):
                inputs._admit_qwen_inputs(self.path, sha)

    def test_alternate_still_authenticates_the_payload_and_tensor_range(self):
        self.payload_path.write_bytes(b"x" * self.document["payload_bytes"])
        with self.assertRaisesRegex(inputs.InputsError, "aggregate payload digest mismatch"):
            self.admit()
        self.document["payload_sha256"] = inputs._hash_file(self.payload_path)
        self.document["tensors"]["input.qwen_bf16"]["offset"] = self.document["payload_bytes"]
        self.write_manifest()
        with self.assertRaisesRegex(inputs.InputsError, "source tensor range is outside its payload"):
            self.admit()

    def test_alternate_manifest_must_remain_a_canonical_regular_file(self):
        alias = self.root / "alternate-alias.json"
        alias.symlink_to(self.path)
        with self.assertRaisesRegex(inputs.InputsError, "canonical regular file"):
            inputs._admit_qwen_inputs(alias, inputs._hash_file(self.path))


class HyNomasContinuationInputsTests(unittest.TestCase):
    def test_named_tensor_inventory_and_exact_byte_budget(self):
        self.assertEqual(
            [name for name, _, _ in inputs.TENSOR_SPECS],
            [
                "input.latent_bank",
                "input.qwen_bf16",
                "input.rope_cos",
                "input.rope_sin",
                "cache.eps_bf16",
                "decoder.noise",
                "decoder.rope_cos",
                "decoder.rope_sin",
            ],
        )
        self.assertEqual(inputs.CACHE_EPS_BANK_SHAPE, (6, 1, 32, 30, 9, 16))
        self.assertEqual(inputs.tensor_nbytes("BF16", inputs.CACHE_EPS_BANK_SHAPE), 1_658_880)
        self.assertEqual(inputs.expected_payload_bytes(), 1_320_611_840)
        with self.assertRaisesRegex(inputs.InputsError, "unsupported tensor dtype"):
            inputs.tensor_nbytes("BF16_UNKNOWN", (1, 2))

    def test_one_seeded_720_latent_mps_draw_returns_only_first_180(self):
        fake_torch = _FakeTorchForLatents()
        device = _FakeDevice()

        bank = inputs._draw_latent_bank(fake_torch, device)

        self.assertEqual(fake_torch.generator.device, device)
        self.assertEqual(fake_torch.generator.seed, 42)
        self.assertEqual(fake_torch.randn_args[0], (1, 32, 720, 9, 16))
        self.assertEqual(fake_torch.randn_args[1], fake_torch.generator)
        self.assertEqual(fake_torch.randn_args[2], device)
        self.assertIs(fake_torch.randn_args[3], fake_torch.float32)
        self.assertEqual(bank.shape, (1, 32, 180, 9, 16))
        self.assertEqual(
            fake_torch.full_bank.slices,
            [(slice(None), slice(None), slice(None, inputs.LATENT_FRAMES, None), slice(None), slice(None))],
        )
        self.assertTrue(bank.contiguous_called)

    def test_cache_eps_uses_six_sequential_mps_draws_and_bf16_exports(self):
        fake_torch = _FakeTorchForCache()
        device = _FakeDevice()

        bank = inputs._draw_cache_eps_bank(fake_torch, device)

        self.assertEqual(len(fake_torch.generators), 1)
        self.assertEqual(fake_torch.generators[0].seed, 42)
        self.assertEqual(fake_torch.stack_dim, 0)
        self.assertEqual(len(fake_torch.draw_calls), 6)
        for shape, generator, draw_device, dtype in fake_torch.draw_calls:
            self.assertEqual(shape, (1, 32, 30, 9, 16))
            self.assertIs(generator, fake_torch.generators[0])
            self.assertIs(draw_device, device)
            self.assertIs(dtype, fake_torch.float32)
        self.assertEqual(bank.shape, (6, 1, 32, 30, 9, 16))
        self.assertIs(bank.dtype, fake_torch.bfloat16)

    def test_pinned_decoder_noise_call_covers_all_180_global_latents(self):
        calls = []

        class FakeDecoder:
            @staticmethod
            def _make_noise(owner, latent, start, stop):
                calls.append((owner.seed, latent, start, stop))
                return _FakeTensor(inputs.PIXEL_NOISE_SHAPE)

        class FakeAutoencoder:
            PixelUMMDCAELatentDecoder = FakeDecoder

        latent = _FakeTensor(inputs.LATENT_BANK_SHAPE)
        output = inputs._draw_decoder_noise(object(), FakeAutoencoder, latent)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (42, latent, 0, 180))
        self.assertEqual(output.shape, (1, 3, 720, 288, 512))

    def test_cache_writer_binds_four_step_demo_and_external_mps_eps(self):
        contract = inputs._cache_writer_contract()
        self.assertEqual(contract["resolved_override"], {"inference.infer_steps_video": 4})
        self.assertTrue(contract["preserve_kv_cache"])
        self.assertEqual(contract["fixed_initial_noise_timeline"]["latent_frames"], 720)
        self.assertEqual(contract["cache_at_step"], -1)
        self.assertEqual(contract["cache_noise_sigma"], 0.206897)
        self.assertEqual(contract["cache_timestep"], 206.897)
        self.assertFalse(contract["cache_eps_upstream_cpu_rng_parity"])
        self.assertEqual(
            [(row["global_latent_start"], row["global_latent_stop"]) for row in contract["cache_eps_draws"]],
            [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)],
        )

    def test_manifest_admission_rejects_unknown_precision_and_cache_rng_claim(self):
        document = _valid_document()
        inputs.validate_document(document)

        bad_precision = copy.deepcopy(document)
        bad_precision["tensors"]["cache.eps_bf16"]["dtype"] = "BF16?"
        with self.assertRaisesRegex(inputs.InputsError, "tensor record mismatch"):
            inputs.validate_document(bad_precision)

        false_parity = copy.deepcopy(document)
        false_parity["cache_writer"]["cache_eps_upstream_cpu_rng_parity"] = True
        with self.assertRaisesRegex(inputs.InputsError, "cache-writer configuration"):
            inputs.validate_document(false_parity)

        wrong_override = copy.deepcopy(document)
        wrong_override["cache_writer"]["resolved_override"]["inference.infer_steps_video"] = 8
        with self.assertRaisesRegex(inputs.InputsError, "cache-writer configuration"):
            inputs.validate_document(wrong_override)

        if inputs.DEFAULT_QWEN_INPUTS.is_file():
            with self.assertRaisesRegex(inputs.InputsError, "admitted digest"):
                inputs._admit_qwen_inputs(inputs.DEFAULT_QWEN_INPUTS, "0" * 64)

    def test_pinned_upstream_rope_offsets_match_direct_equations_and_v2_first_segment(self):
        try:
            import numpy as np
            import torch
        except ImportError as error:
            self.skipTest(f"CPU RoPE integration dependencies unavailable: {error}")
        if not inputs.DEFAULT_QWEN_INPUTS.is_file():
            self.skipTest("authenticated detailed v2 RoPE fixture is not mounted")

        upstream, _ = inputs._verify_upstream(inputs.DEFAULT_UPSTREAM)
        posemb_path = upstream / "hymm/models/dit/modules/posemb_layers.py"
        posemb = inputs._load_pinned_module("continuation_test_pinned_posemb", posemb_path)
        device = torch.device("cpu")
        cos_bank, sin_bank = inputs._draw_dit_rope_bank(torch, posemb, device)
        self.assertEqual(tuple(cos_bank.shape), (6, 4320, 128))
        self.assertEqual(tuple(sin_bank.shape), (6, 4320, 128))

        for index, offset in enumerate(inputs.ROPE_OFFSETS):
            expected_cos, expected_sin = posemb.get_nd_rotary_pos_embed(
                [16, 56, 56],
                (offset, 0, 0),
                (offset + 30, 9, 16),
                theta=256.0,
                use_real=True,
                theta_rescale_factor=1,
                device=device,
            )
            torch.testing.assert_close(cos_bank[index], expected_cos, rtol=0, atol=0)
            torch.testing.assert_close(sin_bank[index], expected_sin, rtol=0, atol=0)
        self.assertFalse(torch.equal(cos_bank[0], cos_bank[1]))
        self.assertFalse(torch.equal(sin_bank[0], sin_bank[1]))

        source = inputs._admit_qwen_inputs(inputs.DEFAULT_QWEN_INPUTS, inputs._MANIFEST_SHA256)
        for output, key in ((cos_bank[0], "cos"), (sin_bank[0], "sin")):
            record = source["rope_records"][key]
            raw = inputs._read_range_bytes(Path(source["payload_path"]), record["offset"], record["nbytes"])
            expected = np.frombuffer(raw, dtype="<f4").reshape(inputs.DIT_ROPE_SHAPE)
            self.assertTrue(
                np.allclose(output.numpy(), expected, atol=inputs.ROPE_ATOL, rtol=inputs.ROPE_RTOL),
                f"first segment {key} RoPE differs from detailed v2 fixture",
            )


if __name__ == "__main__":
    unittest.main()
