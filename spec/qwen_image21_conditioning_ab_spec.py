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

    def _write_native_sidecar(self, *, words: np.ndarray | None = None):
        sidecar_path = self.native_dir / "retained.bf16le"
        if words is None:
            words = np.resize(BF16_WORDS, 10 * 4096).astype("<u2", copy=False)
        sidecar = words.tobytes()
        sidecar_path.write_bytes(sidecar)
        manifest = {
            "schema": "qwen3vl-retained-embeddings",
            "schema_version": 1,
            "prompt": PROMPT,
            "model_revision": REVISION,
            "fixture_payload_sha256": REFERENCE_FIXTURE_SHA256,
            "drop_idx": 14,
            "shape": [10, 4096],
            "dtype": "bfloat16-le",
            "nbytes": len(sidecar),
            "sha256": hashlib.sha256(sidecar).hexdigest(),
            "payload_file": sidecar_path.name,
        }
        sidecar_manifest_path = self.native_dir / "retained.bf16le.json"
        sidecar_manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return sidecar_path, sidecar_manifest_path

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
