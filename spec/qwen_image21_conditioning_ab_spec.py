#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_image21_conditioning_ab.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_conditioning_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PREP_SCRIPT = ROOT / "scripts" / "qwen_image21_prepare_conditioning.py"
PREP_SPEC = importlib.util.spec_from_file_location(
    "qwen_image21_prepare_conditioning_for_ab_spec", PREP_SCRIPT
)
assert PREP_SPEC is not None and PREP_SPEC.loader is not None
PREP = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(PREP)


PROMPT = "red cube"
GENERIC_PROMPT = "a tiny lighthouse above a quiet harbor"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
REFERENCE_FIXTURE_SHA256 = "3edcd7bf7964237d649a43c35cd82f6d6bd7b15835fddec2b1fe42f3a89b1e07"
BF16_WORDS = np.array([0x3F00, 0xBF80, 0x4000, 0x0000], dtype="<u2")


class QwenImage21ConditioningABSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.baseline_dir = self.root / "baseline"
        self.native_dir = self.root / "native"
        self.native_dir.mkdir()
        self.output_dir = self.root / "native-ab"
        self.generic_reference_index = 0
        baseline_words = np.resize(
            np.array([0x3F80, 0xBF00, 0x4000, 0x0000], dtype="<u2"), 10 * 4096
        )
        self.hidden = (baseline_words.astype("<u4") << np.uint32(16)).view("<f4").reshape(10, 4096)
        self.original_embeddings_sha = MODULE.PINNED_REFERENCE_EMBEDDINGS_SHA256
        MODULE.PINNED_REFERENCE_EMBEDDINGS_SHA256 = hashlib.sha256(baseline_words.tobytes()).hexdigest()
        self.attention_mask = np.array([1] * 10, dtype=np.uint8)
        self.image_mask = np.array([0] * 10, dtype=np.uint8)
        self.latents = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)
        self.manifest_path = self._write_baseline()
        self.sidecar_path, self.sidecar_manifest_path = self._write_native_sidecar()

    def tearDown(self) -> None:
        MODULE.PINNED_REFERENCE_EMBEDDINGS_SHA256 = self.original_embeddings_sha
        self.temp_dir.cleanup()

    def _write_baseline(self) -> Path:
        return PREP.write_conditioning_bundle(
            output_dir=self.baseline_dir,
            prompt=PROMPT,
            revision=REVISION,
            width=32,
            height=32,
            seed=7,
            encoder_hidden_states=self.hidden,
            encoder_hidden_states_mask=self.attention_mask,
            encoder_img_mask=self.image_mask,
            initial_target_latents=self.latents,
            source_dtype="bfloat16",
            device="cpu",
            torch_version="synthetic",
            diffusers_version="synthetic",
        )

    def _write_native_sidecar(
        self,
        *,
        words: np.ndarray | None = None,
        prompt: str = PROMPT,
        revision: str = REVISION,
        fixture_sha256: str = REFERENCE_FIXTURE_SHA256,
        fixture_manifest_sha256: str | None = None,
        drop_idx: int = 14,
        shape: list[int] | None = None,
    ):
        sidecar_path = self.native_dir / "retained.bf16le"
        if words is None:
            words = np.resize(BF16_WORDS, 10 * 4096).astype("<u2", copy=False)
        words = np.asarray(words, dtype="<u2").reshape(-1)
        if shape is None:
            shape = [words.size // 4096, 4096]
        sidecar = words.tobytes()
        sidecar_path.write_bytes(sidecar)
        manifest = {
            "schema": "qwen3vl-retained-embeddings",
            "schema_version": 1,
            "prompt": prompt,
            "model_revision": revision,
            "fixture_payload_sha256": fixture_sha256,
            "drop_idx": drop_idx,
            "shape": shape,
            "dtype": "bfloat16-le",
            "nbytes": len(sidecar),
            "sha256": hashlib.sha256(sidecar).hexdigest(),
            "payload_file": sidecar_path.name,
        }
        if fixture_manifest_sha256 is not None:
            manifest["fixture_manifest_sha256"] = fixture_manifest_sha256
        sidecar_manifest_path = self.native_dir / "retained.bf16le.json"
        sidecar_manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return sidecar_path, sidecar_manifest_path

    def _write_generic_reference(self, *, prompt: str = GENERIC_PROMPT, raw_tokens: int = 18, drop_idx: int = 14):
        retained_rows = raw_tokens - drop_idx
        self.assertGreater(retained_rows, 0)
        ids = np.arange(raw_tokens, dtype="<i8").tobytes()
        mask = np.ones(raw_tokens, dtype="<i8").tobytes()
        embedding_words = np.resize(BF16_WORDS, retained_rows * 4096).astype("<u2")
        embedding = embedding_words.tobytes()
        tensors: dict[str, dict] = {}
        payload_parts: list[bytes] = []
        offset = 0

        def add_tensor(name: str, dtype: str, shape: list[int], data: bytes) -> None:
            nonlocal offset
            tensors[name] = {
                "dtype": dtype,
                "shape": shape,
                "offset_bytes": offset,
                "nbytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            payload_parts.append(data)
            offset += len(data)

        add_tensor("input_ids", "int64-le", [1, raw_tokens], ids)
        add_tensor("attention_mask", "int64-le", [1, raw_tokens], mask)
        add_tensor(
            "pre_final_rmsnorm_embeddings",
            "bfloat16-le",
            [1, retained_rows, 4096],
            embedding,
        )
        final_hidden_words = np.zeros((raw_tokens, 4096), dtype="<u2")
        final_hidden_words[drop_idx:] = embedding_words.reshape(retained_rows, 4096)
        for index in range(37):
            hidden = final_hidden_words if index == 36 else np.zeros_like(final_hidden_words)
            add_tensor(
                f"hidden_state_{index:03d}",
                "bfloat16-le",
                [1, raw_tokens, 4096],
                hidden.tobytes(),
            )

        payload = b"".join(payload_parts)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        reference_dir = self.root / "official-reference"
        reference_dir.mkdir(exist_ok=True)
        payload_path = reference_dir / "qwen_image21_text_reference.bin"
        payload_path.write_bytes(payload)
        manifest = {
            "schema": "qwen-image21-text-reference",
            "schema_version": 1,
            "model": {
                "repo": "Qwen/Qwen-Image-2.1",
                "revision_sha": REVISION,
                "revision_source": "argument_and_local_cache_metadata",
                "pipeline_class": "QwenImage21Pipeline",
                "text_encoder_class": "Qwen3VLForConditionalGeneration",
                "processor_class": "Qwen3VLProcessor",
            },
            "prompt": prompt,
            "tokenization": {
                "raw_template_text": (
                    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                ),
                "processor_kwargs": {"padding": True, "padding_side": "left", "return_tensors": "pt"},
                "tokenizer_truncation": False,
            },
            "sequence": {
                "max_sequence_length": retained_rows,
                "max_sequence_length_semantics": "post-drop validation guard only; no processor truncation",
                "actual_sequence_length": retained_rows,
                "raw_input_shape": [1, raw_tokens],
                "drop_idx": drop_idx,
            },
            "embedding": {
                "source": "QwenImage21Pipeline._get_qwen_prompt_embeds",
                "pre_final_rmsnorm": True,
                "rmsnorm_hook_observed_and_verified": True,
                "shape": [1, retained_rows, 4096],
                "source_dtype": "bfloat16",
                "expected_decoder_layer_count": 36,
                "expected_hidden_state_count": 37,
                "expected_hidden_state_count_source": "loaded text_encoder.config.text_config.num_hidden_layers + 1",
                "hidden_state_count": 37,
            },
            "runtime": {
                "source_dtype": "bfloat16",
                "device": "cpu",
                "diffusers_version": "0.41.0.dev0",
                "diffusers_commit": "8b3c707ebd3ec4881f4190cf42931da07eaf3b65",
                "official_source_file": "diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py",
            },
            "payload_file": payload_path.name,
            "payload_nbytes": len(payload),
            "payload_sha256": payload_sha256,
            "tensors": tensors,
        }
        manifest_path = reference_dir / "qwen_image21_text_reference.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        baseline_words = embedding_words
        expanded = (baseline_words.astype("<u4") << np.uint32(16)).view("<f4").reshape(retained_rows, 4096)
        self.baseline_dir = self.root / f"generic-baseline-{self.generic_reference_index}"
        self.generic_reference_index += 1
        self.manifest_path = PREP.write_conditioning_bundle(
            output_dir=self.baseline_dir,
            prompt=prompt,
            revision=REVISION,
            width=32,
            height=32,
            seed=7,
            encoder_hidden_states=expanded,
            encoder_hidden_states_mask=np.ones(retained_rows, dtype=np.uint8),
            encoder_img_mask=np.zeros(retained_rows, dtype=np.uint8),
            initial_target_latents=self.latents,
            source_dtype="bfloat16",
            device="cpu",
            torch_version="synthetic",
            diffusers_version="synthetic",
        )
        native_words = np.resize(
            np.array([0x3F80, 0xBF00, 0x4000, 0x0000], dtype="<u2"),
            retained_rows * 4096,
        )
        self.sidecar_path, self.sidecar_manifest_path = self._write_native_sidecar(
            words=native_words,
            prompt=prompt,
            revision=REVISION,
            fixture_sha256=payload_sha256,
            fixture_manifest_sha256=manifest_sha256,
            drop_idx=drop_idx,
            shape=[retained_rows, 4096],
        )
        return manifest_path, payload_path, payload_sha256, embedding_words, retained_rows, drop_idx

    def _read_manifest(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _mutate_native_manifest(self, key: str, value) -> None:
        manifest = self._read_manifest(self.sidecar_manifest_path)
        manifest[key] = value
        self.sidecar_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_replaces_only_embedding_bytes_and_records_diagnostic_provenance(self) -> None:
        source_manifest = self._read_manifest(self.manifest_path)
        source_payload_path = self.baseline_dir / source_manifest["payload_file"]
        source_payload = source_payload_path.read_bytes()
        result_manifest_path = MODULE.create_ab_bundle(
            self.baseline_dir, self.sidecar_manifest_path, self.output_dir
        )

        result_manifest = self._read_manifest(result_manifest_path)
        result_payload = (self.output_dir / result_manifest["payload_file"]).read_bytes()
        hidden = source_manifest["tensors"]["encoder_hidden_states"]
        start = hidden["offset_bytes"]
        end = start + hidden["nbytes"]
        self.assertEqual(source_payload[:start], result_payload[:start])
        self.assertNotEqual(source_payload[start:end], result_payload[start:end])
        self.assertEqual(source_payload[end:], result_payload[end:])
        native_words = np.frombuffer(self.sidecar_path.read_bytes(), dtype="<u2")
        expected_hidden = (native_words.astype("<u4") << np.uint32(16)).view("<f4")
        actual_hidden = np.frombuffer(result_payload[start:end], dtype="<f4")
        np.testing.assert_array_equal(actual_hidden, expected_hidden)
        self.assertEqual(
            hashlib.sha256(result_payload).hexdigest(), result_manifest["payload_sha256"]
        )
        self.assertEqual(source_payload, source_payload_path.read_bytes())
        self.assertEqual(source_manifest["payload_sha256"], result_manifest["conditioning_ab"]["baseline_payload_sha256"])
        self.assertNotEqual(
            source_manifest["payload_sha256"],
            result_manifest["conditioning_ab"]["native_reference_fixture_payload_sha256"],
        )
        self.assertEqual(
            REFERENCE_FIXTURE_SHA256,
            result_manifest["conditioning_ab"]["native_reference_fixture_payload_sha256"],
        )
        self.assertEqual(
            MODULE.PINNED_REFERENCE_EMBEDDINGS_SHA256,
            result_manifest["conditioning_ab"]["official_reference_embeddings_sha256"],
        )
        self.assertEqual("native-retained-embeddings", result_manifest["conditioning_ab"]["kind"])
        self.assertEqual(14, result_manifest["conditioning_ab"]["drop_idx"])
        self.assertEqual([10, 4096], result_manifest["conditioning_ab"]["shape"])
        self.assertEqual("bfloat16-le", result_manifest["conditioning_ab"]["source_dtype"])

    def test_opt_in_generic_prompt_validates_official_reference_and_dynamic_shape(self) -> None:
        reference_path, _, reference_sha, official_words, retained_rows, drop_idx = self._write_generic_reference()
        baseline_manifest = self._read_manifest(self.manifest_path)
        baseline_payload_path = self.baseline_dir / baseline_manifest["payload_file"]
        baseline_payload = baseline_payload_path.read_bytes()
        result_manifest_path = MODULE.create_ab_bundle(
            self.baseline_dir,
            self.sidecar_manifest_path,
            self.output_dir,
            reference_manifest_path=reference_path,
        )

        result_manifest = self._read_manifest(result_manifest_path)
        result_payload = (self.output_dir / result_manifest["payload_file"]).read_bytes()
        descriptor = baseline_manifest["tensors"]["encoder_hidden_states"]
        start = descriptor["offset_bytes"]
        end = start + descriptor["nbytes"]
        expected = (official_words.astype("<u4") << np.uint32(16)).view("<f4").tobytes()
        self.assertEqual(expected, baseline_payload[start:end])
        self.assertEqual(baseline_payload[:start], result_payload[:start])
        self.assertNotEqual(baseline_payload[start:end], result_payload[start:end])
        self.assertEqual(baseline_payload[end:], result_payload[end:])
        conditioning_ab = result_manifest["conditioning_ab"]
        self.assertEqual(GENERIC_PROMPT, result_manifest["prompt"])
        self.assertEqual(reference_sha, conditioning_ab["native_reference_fixture_payload_sha256"])
        self.assertEqual([retained_rows, 4096], conditioning_ab["shape"])
        self.assertEqual(drop_idx, conditioning_ab["drop_idx"])
        self.assertEqual(
            hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            conditioning_ab["official_reference_manifest_sha256"],
        )
        self.assertEqual(result_manifest["tensors"]["encoder_hidden_states"]["shape"], [retained_rows, 4096])

    def test_opt_in_generic_prompt_rejects_reference_or_sidecar_drift_before_output(self) -> None:
        reference_path, payload_path, reference_sha, _, retained_rows, drop_idx = self._write_generic_reference()
        original_payload = payload_path.read_bytes()
        payload_path.write_bytes(original_payload[:-1] + bytes([original_payload[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "reference payload SHA-256"):
            MODULE.create_ab_bundle(
                self.baseline_dir,
                self.sidecar_manifest_path,
                self.output_dir,
                reference_manifest_path=reference_path,
            )
        self.assertFalse(self.output_dir.exists())
        payload_path.write_bytes(original_payload)

        for key, value in (
            ("prompt", "a different prompt"),
            ("model_revision", "0" * 40),
            ("fixture_payload_sha256", "0" * 64),
            ("fixture_manifest_sha256", "0" * 64),
            ("drop_idx", drop_idx + 1),
            ("shape", [retained_rows + 1, 4096]),
            ("shape", [True, retained_rows, 4096]),
        ):
            with self.subTest(sidecar_field=key):
                self._write_native_sidecar(
                    words=np.resize(BF16_WORDS, retained_rows * 4096),
                    prompt=GENERIC_PROMPT,
                    revision=REVISION,
                    fixture_sha256=reference_sha,
                    drop_idx=drop_idx,
                    shape=[retained_rows, 4096],
                )
                self._mutate_native_manifest(key, value)
                output = self.root / f"generic-rejected-{key}"
                with self.assertRaises(ValueError):
                    MODULE.create_ab_bundle(
                        self.baseline_dir,
                        self.sidecar_manifest_path,
                        output,
                        reference_manifest_path=reference_path,
                    )
                self.assertFalse(output.exists())

    def test_generic_reference_requires_manifest_binding_and_capture_guards(self) -> None:
        reference_path, _, _, _, _, _ = self._write_generic_reference()
        original = reference_path.read_bytes()
        mutations = (
            ("model.processor_class", "WrongProcessor"),
            ("tokenization.tokenizer_truncation", True),
            ("tokenization.processor_kwargs.padding_side", "right"),
            ("sequence.max_sequence_length_semantics", "truncates before tokenization"),
            ("embedding.source", "unverified source"),
            ("embedding.rmsnorm_hook_observed_and_verified", False),
            ("embedding.hidden_state_count", 37.0),
            ("embedding.expected_hidden_state_count", 37.0),
            ("runtime.diffusers_commit", "0" * 40),
        )
        for dotted_key, value in mutations:
            with self.subTest(reference_field=dotted_key):
                manifest = json.loads(original)
                keys = dotted_key.split(".")
                target = manifest
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                reference_path.write_text(json.dumps(manifest), encoding="utf-8")
                output = self.root / f"generic-guard-{keys[-1]}"
                with self.assertRaises(ValueError):
                    MODULE.create_ab_bundle(
                        self.baseline_dir,
                        self.sidecar_manifest_path,
                        output,
                        reference_manifest_path=reference_path,
                    )
                self.assertFalse(output.exists())
        reference_path.write_bytes(original)

        reference_manifest, payload_path, _, _, _, _ = self._write_generic_reference()
        payload = bytearray(payload_path.read_bytes())
        descriptor = json.loads(reference_manifest.read_text(encoding="utf-8"))["tensors"]["hidden_state_036"]
        payload[descriptor["offset_bytes"] + 14 * 4096 * 2] ^= 1
        payload_path.write_bytes(payload)
        manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
        manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        hidden = manifest["tensors"]["hidden_state_036"]
        start = hidden["offset_bytes"]
        end = start + hidden["nbytes"]
        hidden["sha256"] = hashlib.sha256(payload[start:end]).hexdigest()
        reference_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        native_manifest = self._read_manifest(self.sidecar_manifest_path)
        native_manifest["fixture_payload_sha256"] = hashlib.sha256(payload).hexdigest()
        native_manifest["fixture_manifest_sha256"] = hashlib.sha256(
            reference_manifest.read_bytes()
        ).hexdigest()
        self.sidecar_manifest_path.write_text(json.dumps(native_manifest), encoding="utf-8")
        output = self.root / "generic-hidden-state-mismatch"
        with self.assertRaisesRegex(ValueError, "hidden_state_036.*pre-final-RMSNorm"):
            MODULE.create_ab_bundle(
                self.baseline_dir,
                self.sidecar_manifest_path,
                output,
                reference_manifest_path=reference_manifest,
            )
        self.assertFalse(output.exists())

    def test_generic_reference_max_sequence_length_is_post_drop(self) -> None:
        reference_path, _, _, _, _, _ = self._write_generic_reference(raw_tokens=18, drop_idx=14)
        # Synthetic reference has 18 raw tokens, but only four post-drop rows;
        # max_sequence_length=4 is valid according to the capture contract.
        MODULE.create_ab_bundle(
            self.baseline_dir,
            self.sidecar_manifest_path,
            self.output_dir,
            reference_manifest_path=reference_path,
        )

    def test_rejects_oversized_baseline_and_native_files_before_reading(self) -> None:
        baseline_manifest = self._read_manifest(self.manifest_path)
        baseline_payload_path = self.baseline_dir / baseline_manifest["payload_file"]
        with baseline_payload_path.open("ab") as stream:
            stream.truncate(MODULE.MAX_BASELINE_PAYLOAD_BYTES + 1)
        original_read_bytes = Path.read_bytes

        def reject_baseline_read(path: Path) -> bytes:
            if path == baseline_payload_path:
                raise AssertionError("oversized baseline payload was read before its size guard")
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", reject_baseline_read):
            with self.assertRaisesRegex(ValueError, "baseline conditioning payload exceeds"):
                MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, self.output_dir)
        self.assertFalse(self.output_dir.exists())
        # Restore the synthetic baseline, then make the native sidecar sparse
        # and oversized; stat-based rejection must happen before read_bytes().
        self.manifest_path.unlink()
        baseline_payload_path.unlink()
        self.manifest_path = self._write_baseline()
        with self.sidecar_path.open("ab") as stream:
            stream.truncate(self.sidecar_path.stat().st_size + 1)
        def reject_native_read(path: Path) -> bytes:
            if path == self.sidecar_path:
                raise AssertionError("native sidecar was read before its declared-size guard")
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", reject_native_read):
            with self.assertRaisesRegex(ValueError, "native BF16 sidecar length"):
                MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_noninteger_hidden_count_and_boolean_baseline_img_shapes(self) -> None:
        reference_path, _, _, _, _, _ = self._write_generic_reference()
        original_reference = reference_path.read_bytes()
        manifest = json.loads(original_reference)
        manifest["embedding"]["hidden_state_count"] = 37.0
        reference_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "model layer count"):
            MODULE.create_ab_bundle(
                self.baseline_dir,
                self.sidecar_manifest_path,
                self.output_dir,
                reference_manifest_path=reference_path,
            )
        self.assertFalse(self.output_dir.exists())

        reference_path.write_bytes(original_reference)
        manifest = self._read_manifest(self.manifest_path)
        manifest["image"]["img_shapes"] = [[True, 2, 2]]
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "latent geometry"):
            MODULE.create_ab_bundle(
                self.baseline_dir,
                self.sidecar_manifest_path,
                self.output_dir,
                reference_manifest_path=reference_path,
            )
        self.assertFalse(self.output_dir.exists())

    def test_opt_in_generic_prompt_rejects_official_embedding_mismatch(self) -> None:
        reference_path, _, _, _, _, _ = self._write_generic_reference()
        baseline_manifest = self._read_manifest(self.manifest_path)
        payload_path = self.baseline_dir / baseline_manifest["payload_file"]
        payload = bytearray(payload_path.read_bytes())
        descriptor = baseline_manifest["tensors"]["encoder_hidden_states"]
        payload[descriptor["offset_bytes"] : descriptor["offset_bytes"] + 4] = np.array(
            [2.0], dtype="<f4"
        ).tobytes()
        payload_path.write_bytes(payload)
        baseline_manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        self.manifest_path.write_text(json.dumps(baseline_manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "official reference embeddings"):
            MODULE.create_ab_bundle(
                self.baseline_dir,
                self.sidecar_manifest_path,
                self.output_dir,
                reference_manifest_path=reference_path,
            )
        self.assertFalse(self.output_dir.exists())

    def test_generic_reference_requires_bfloat16_hidden_states(self) -> None:
        reference_path, _, _, _, _, _ = self._write_generic_reference()
        manifest = json.loads(reference_path.read_text(encoding="utf-8"))
        manifest["tensors"]["hidden_state_000"]["dtype"] = "float32-le"
        reference_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "hidden state 'hidden_state_000' must use bfloat16-le"):
            MODULE.create_ab_bundle(
                self.baseline_dir,
                self.sidecar_manifest_path,
                self.output_dir,
                reference_manifest_path=reference_path,
            )
        self.assertFalse(self.output_dir.exists())

    def test_opt_in_generic_prompt_rejects_reference_metadata_and_symlink_inputs(self) -> None:
        reference_path, payload_path, _, _, _, _ = self._write_generic_reference()
        reference_manifest = reference_path
        original_manifest = reference_manifest.read_bytes()
        reference_fields = (
            ("schema", "wrong-schema"),
            ("model.revision_sha", "0" * 40),
            ("prompt", "different prompt"),
            ("sequence.drop_idx", 13),
            ("sequence.actual_sequence_length", 5),
            ("sequence.raw_input_shape", [True, 18]),
            ("embedding.shape", [True, 4, 4096]),
            ("tensors.input_ids.shape", [True, 18]),
        )
        for dotted_key, value in reference_fields:
            with self.subTest(reference_field=dotted_key):
                manifest = json.loads(original_manifest)
                keys = dotted_key.split(".")
                target = manifest
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                reference_manifest.write_text(json.dumps(manifest), encoding="utf-8")
                output = self.root / f"generic-invalid-reference-{keys[-1]}"
                with self.assertRaises(ValueError):
                    MODULE.create_ab_bundle(
                        self.baseline_dir,
                        self.sidecar_manifest_path,
                        output,
                        reference_manifest_path=reference_manifest,
                    )
                self.assertFalse(output.exists())
        reference_manifest.write_bytes(original_manifest)

        real_payload = payload_path.with_name("official-reference.real.bin")
        payload_path.rename(real_payload)
        try:
            payload_path.symlink_to(real_payload.name)
            with self.assertRaisesRegex(ValueError, "regular file"):
                MODULE.create_ab_bundle(
                    self.baseline_dir,
                    self.sidecar_manifest_path,
                    self.output_dir,
                    reference_manifest_path=reference_manifest,
                )
            self.assertFalse(self.output_dir.exists())
        finally:
            payload_path.unlink(missing_ok=True)
            real_payload.rename(payload_path)

        native_manifest_link = self.root / "native-manifest-link.json"
        native_manifest_link.symlink_to(self.sidecar_manifest_path)
        try:
            with self.assertRaisesRegex(ValueError, "native manifest must not be a symlink"):
                MODULE.create_ab_bundle(
                    self.baseline_dir,
                    native_manifest_link,
                    self.output_dir,
                    reference_manifest_path=reference_manifest,
                )
            self.assertFalse(self.output_dir.exists())
        finally:
            native_manifest_link.unlink(missing_ok=True)

    def test_cli_resolves_the_sidecar_as_a_manifest_sibling(self) -> None:
        # The subprocess uses a synthetic BF16 baseline; patch only its pinned
        # expected embedding digest, while the real CLI uses the official one.
        inline = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('ab_cli', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "module.PINNED_REFERENCE_EMBEDDINGS_SHA256 = sys.argv[2]; "
            "raise SystemExit(module.main(sys.argv[3:]))"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                inline,
                str(SCRIPT),
                MODULE.PINNED_REFERENCE_EMBEDDINGS_SHA256,
                "--baseline-bundle",
                str(self.baseline_dir),
                "--native-manifest",
                str(self.sidecar_manifest_path),
                "--output-dir",
                str(self.output_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("conditioning_ab_manifest=", completed.stdout)
        self.assertIn("no DiT/VAE inference", completed.stdout)
        self.assertTrue((self.output_dir / "qwen_image21_conditioning.json").is_file())

    def test_rejects_mismatched_native_provenance_before_creating_output(self) -> None:
        mismatches = (
            ("schema", "wrong-schema"),
            ("schema_version", 2),
            ("prompt", "a different prompt"),
            ("model_revision", "0" * 40),
            ("fixture_payload_sha256", "0" * 64),
            ("drop_idx", 13),
            ("shape", [9, 4096]),
            ("dtype", "float32-le"),
            ("nbytes", 1),
            ("sha256", "0" * 64),
        )
        for key, value in mismatches:
            with self.subTest(key=key):
                self._mutate_native_manifest(key, value)
                output = self.root / f"rejected-{key}"
                with self.assertRaises((ValueError, OSError)):
                    MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, output)
                self.assertFalse(output.exists())
                self._write_native_sidecar()

    def test_rejects_nonfinite_sidecar_values_and_payload_path_escape(self) -> None:
        invalid_words = np.resize(BF16_WORDS, 10 * 4096).astype("<u2", copy=True)
        invalid_words[0] = 0x7F80  # positive infinity in bfloat16
        self._write_native_sidecar(words=invalid_words)
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, self.output_dir)
        self.assertFalse(self.output_dir.exists())

        self._write_native_sidecar()
        self._mutate_native_manifest("payload_file", "../outside.bf16le")
        with self.assertRaisesRegex(ValueError, "payload_file"):
            MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_corrupt_baseline_and_existing_output(self) -> None:
        source_manifest = self._read_manifest(self.manifest_path)
        source_payload_path = self.baseline_dir / source_manifest["payload_file"]
        original_payload = source_payload_path.read_bytes()
        original_manifest_bytes = self.manifest_path.read_bytes()
        malformed_layout = self._read_manifest(self.manifest_path)
        malformed_layout["tensors"]["encoder_img_mask"]["offset_bytes"] += 1
        self.manifest_path.write_text(json.dumps(malformed_layout), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid offset or length"):
            MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, self.output_dir)
        self.assertFalse(self.output_dir.exists())
        self.manifest_path.write_bytes(original_manifest_bytes)

        source_payload_path.write_bytes(original_payload[:-1] + bytes([original_payload[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, self.output_dir)
        self.assertFalse(self.output_dir.exists())
        source_payload_path.write_bytes(original_payload)

        self.output_dir.mkdir()
        sentinel = self.output_dir / "keep.txt"
        sentinel.write_text("do not overwrite", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "output directory"):
            MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, self.output_dir)
        self.assertEqual("do not overwrite", sentinel.read_text(encoding="utf-8"))

    def test_rejects_baseline_embeddings_that_do_not_match_pinned_reference(self) -> None:
        baseline_manifest = self._read_manifest(self.manifest_path)
        payload_path = self.baseline_dir / baseline_manifest["payload_file"]
        payload = bytearray(payload_path.read_bytes())
        payload[0:4] = np.array([2.0], dtype="<f4").tobytes()
        payload_path.write_bytes(payload)
        baseline_manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        self.manifest_path.write_text(json.dumps(baseline_manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "official reference embeddings"):
            MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_non_cpu_bf16_baselines_and_inconsistent_latent_geometry(self) -> None:
        baseline_manifest = self._read_manifest(self.manifest_path)
        cases = (
            ("manifest.prompt", "a different prompt"),
            ("model.revision", "0" * 40),
            ("noise.source_dtype", "float32"),
            ("noise.generator_device", "mps"),
            ("runtime.device", "mps"),
            ("image.latent_height", 3),
            ("image.latent_width", 3),
            ("image.img_shapes", [[1, 3, 2]]),
        )
        for dotted_key, value in cases:
            with self.subTest(field=dotted_key):
                altered = self._read_manifest(self.manifest_path)
                section, field = dotted_key.split(".")
                if section == "manifest":
                    altered["prompt"] = value
                elif section == "model":
                    altered["model"][field] = value
                else:
                    altered[section][field] = value
                self.manifest_path.write_text(json.dumps(altered), encoding="utf-8")
                output = self.root / f"rejected-{section}-{field}"
                with self.assertRaisesRegex(ValueError, "baseline"):
                    MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, output)
                self.assertFalse(output.exists())
                self.manifest_path.write_text(json.dumps(baseline_manifest), encoding="utf-8")

    def test_rejects_invalid_retained_mask_and_image_placeholders(self) -> None:
        baseline_manifest = self._read_manifest(self.manifest_path)
        payload_path = self.baseline_dir / baseline_manifest["payload_file"]
        original_payload = payload_path.read_bytes()
        original_manifest = self.manifest_path.read_bytes()
        for tensor_name, first_byte, rejection in (
            ("encoder_hidden_states_mask", 0, "all ten text tokens valid"),
            ("encoder_hidden_states_mask", 1, "all ten text tokens valid"),
            ("encoder_img_mask", 1, "image placeholders"),
        ):
            with self.subTest(tensor=tensor_name, first_byte=first_byte):
                altered_payload = bytearray(original_payload)
                descriptor = baseline_manifest["tensors"][tensor_name]
                altered_payload[
                    descriptor["offset_bytes"] : descriptor["offset_bytes"] + descriptor["nbytes"]
                ] = bytes(descriptor["nbytes"])
                altered_payload[descriptor["offset_bytes"]] = first_byte
                payload_path.write_bytes(altered_payload)
                altered_manifest = dict(baseline_manifest)
                altered_manifest["payload_sha256"] = hashlib.sha256(altered_payload).hexdigest()
                self.manifest_path.write_text(json.dumps(altered_manifest), encoding="utf-8")
                output = self.root / f"rejected-{tensor_name}"
                with self.assertRaisesRegex(ValueError, rejection):
                    MODULE.create_ab_bundle(self.baseline_dir, self.sidecar_manifest_path, output)
                self.assertFalse(output.exists())
                payload_path.write_bytes(original_payload)
                self.manifest_path.write_bytes(original_manifest)


if __name__ == "__main__":
    unittest.main()
