#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_image21_cuda_reference.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_cuda_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenImage21CudaReferenceSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "fixture"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_tiny_fixture_is_a_hashed_mini_quad_with_exact_selected_modulation(self) -> None:
        manifest_path = MODULE.prepare_fixture(self.root, "tiny_mini_quad")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual("qwen-image21-cuda-block-fixture", manifest["schema"])
        self.assertEqual("tiny_mini_quad", manifest["fixture"]["kind"])
        self.assertEqual(6, manifest["fixture"]["tokens"])
        self.assertEqual([1, 2, 2], manifest["fixture"]["config"]["img_shapes"][0])
        self.assertEqual([6, 5], manifest["weights"]["mlp_out"]["shape"])

        inputs = MODULE._validate_inputs(self.root, manifest)
        np.testing.assert_array_equal(inputs["image_ids"], [-1, -1, 0, 0, 0, 0])
        np.testing.assert_array_equal(inputs["key_valid"], [1, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(inputs["target_token_mask"], [0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(
            inputs["modulation"],
            np.where(inputs["target_token_mask"][:, None].astype(bool), inputs["modulation_rows"][0], inputs["modulation_rows"][1]),
        )
        packed = (self.root / manifest["weights"]["to_q"]["packed_file"]).read_bytes()
        expected = np.asarray(MODULE._tiny_weights()["to_q"][0], dtype="<f4").tobytes()
        self.assertEqual(expected, packed)

    def test_real_fixture_uses_the_pinned_asset_and_executes_quad_attention(self) -> None:
        MODULE.prepare_fixture(self.root, "real_block0_mini_quad")
        manifest = MODULE._load_manifest(self.root)
        self.assertEqual(
            {
                "repo": MODULE.GGUF_REPO,
                "revision": MODULE.GGUF_REVISION,
                "filename": MODULE.GGUF_FILENAME,
                "size": MODULE.GGUF_SIZE,
                "sha256": MODULE.GGUF_SHA256,
                "block_index": 0,
            },
            manifest["model"],
        )
        self.assertEqual(4096, manifest["fixture"]["config"]["hidden_dim"])
        self.assertEqual(12288, manifest["fixture"]["config"]["intermediate_dim"])
        inputs = MODULE._validate_inputs(self.root, manifest)
        self.assertEqual(6, inputs["hidden"].shape[0])
        self.assertEqual(4, int((inputs["image_ids"] == 0).sum()))
        self.assertEqual(2, int((inputs["image_ids"] == -1).sum()))

    def _write_fake_reference_and_candidate(self) -> Path:
        MODULE.prepare_fixture(self.root, "tiny_mini_quad")
        manifest = MODULE._load_manifest(self.root)
        shapes = MODULE._checkpoint_shapes(manifest["fixture"])
        reference_dir = self.root / "reference"
        reference_dir.mkdir()
        reference = {}
        arrays = {}
        for index, name in enumerate(MODULE.CHECKPOINTS):
            count = int(np.prod(shapes[name]))
            values = np.linspace(-0.25, 0.25, count, dtype=np.float32).reshape(shapes[name]) + index * 0.001
            arrays[name] = values
            reference[name] = MODULE._save_tensor(self.root, f"reference/{name}.f32", values, "float32-le")
        chain_values = arrays["output"].copy()
        chain_reference = MODULE._save_tensor(
            self.root, "reference/device_chain_output.f32", chain_values, "float32-le"
        )
        manifest["reference"] = {
            "schema": MODULE.REFERENCE_SCHEMA,
            "schema_version": 1,
            "diffusers_commit": MODULE.DIFFUSERS_COMMIT,
            "checkpoints": reference,
            "guards": {
                "device_chain": "same_official_block_second_invocation_with_first_output_as_hidden_states",
                "device_chain_output": chain_reference,
            },
        }
        MODULE._write_json(self.root / "manifest.json", manifest)

        candidate_dir = self.root / "candidate"
        candidate_dir.mkdir()
        candidate = {}
        for name, values in arrays.items():
            candidate[name] = MODULE._save_tensor(candidate_dir, f"{name}.f32", values, "float32-le")
        chain_candidate = MODULE._save_tensor(
            candidate_dir, "device_chain_output.f32", chain_values, "float32-le"
        )
        fixture_bytes = (self.root / "manifest.json").read_bytes()
        MODULE._write_json(
            candidate_dir / "candidate.json",
            {
                "schema": MODULE.CANDIDATE_SCHEMA,
                "schema_version": 1,
                "fixture_manifest_sha256": MODULE._sha256(fixture_bytes),
                "checkpoints": candidate,
                "guards": {
                    "device_chain": "runner_b_input=runner_a.output_device_ptr",
                    "repeat_input_output_bitwise_equal": True,
                    "chain_output": chain_candidate,
                },
            },
        )
        return candidate_dir

    def test_comparator_accepts_exact_reference_then_rejects_seeded_checkpoint_defect(self) -> None:
        candidate_dir = self._write_fake_reference_and_candidate()
        report = MODULE.compare_fixture(self.root, candidate_dir)
        self.assertEqual(16, report["checkpoint_count"])
        self.assertEqual(2, report["guard_count"])
        self.assertEqual(0.0, report["max_abs"])

        candidate_manifest_path = candidate_dir / "candidate.json"
        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        descriptor = candidate_manifest["checkpoints"]["output"]
        candidate = MODULE._read_tensor(candidate_dir, descriptor, "candidate output")
        candidate[0, 0] += np.float32(2.0 * MODULE.TOLERANCES["tiny_max_abs"])
        candidate_manifest["checkpoints"]["output"] = MODULE._save_tensor(
            candidate_dir, descriptor["file"], candidate, "float32-le"
        )
        MODULE._write_json(candidate_manifest_path, candidate_manifest)
        with self.assertRaisesRegex(MODULE.ParityError, "output"):
            MODULE.compare_fixture(self.root, candidate_dir)

        # The separately chained runner output is a required parity guard.
        self.tearDown()
        self.setUp()
        candidate_dir = self._write_fake_reference_and_candidate()
        candidate_manifest_path = candidate_dir / "candidate.json"
        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        descriptor = candidate_manifest["guards"]["chain_output"]
        chain = MODULE._read_tensor(candidate_dir, descriptor, "candidate device-chain output")
        chain[0, 0] += np.float32(2.0 * MODULE.TOLERANCES["tiny_max_abs"])
        candidate_manifest["guards"]["chain_output"] = MODULE._save_tensor(
            candidate_dir, descriptor["file"], chain, "float32-le"
        )
        MODULE._write_json(candidate_manifest_path, candidate_manifest)
        with self.assertRaisesRegex(MODULE.ParityError, "device_chain_output"):
            MODULE.compare_fixture(self.root, candidate_dir)

    def test_comparator_rejects_tolerance_widening_missing_checkpoints_and_nonfinite(self) -> None:
        candidate_dir = self._write_fake_reference_and_candidate()
        fixture_manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        fixture_manifest["tolerances"]["tiny_max_abs"] = 0.1
        MODULE._write_json(self.root / "manifest.json", fixture_manifest)
        with self.assertRaisesRegex(MODULE.FixtureError, "tolerance policy"):
            MODULE.compare_fixture(self.root, candidate_dir)

        # Restore the fixture bytes and candidate binding before independent
        # corruption checks.
        self.tearDown()
        self.setUp()
        candidate_dir = self._write_fake_reference_and_candidate()
        candidate_manifest_path = candidate_dir / "candidate.json"
        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        del candidate_manifest["checkpoints"]["k_rope"]
        MODULE._write_json(candidate_manifest_path, candidate_manifest)
        with self.assertRaisesRegex(MODULE.FixtureError, "checkpoint set"):
            MODULE.compare_fixture(self.root, candidate_dir)

        # A separate valid binding with one NaN must be rejected as corrupt
        # data, not reported as a numerical mismatch.
        self.tearDown()
        self.setUp()
        candidate_dir = self._write_fake_reference_and_candidate()
        candidate_manifest_path = candidate_dir / "candidate.json"
        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        descriptor = candidate_manifest["checkpoints"]["output"]
        values = MODULE._read_tensor(candidate_dir, descriptor, "candidate output")
        values[0, 0] = np.nan
        payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
        file_path = candidate_dir / descriptor["file"]
        file_path.write_bytes(payload)
        descriptor["sha256"] = MODULE._sha256(payload)
        descriptor["nbytes"] = len(payload)
        MODULE._write_json(candidate_manifest_path, candidate_manifest)
        with self.assertRaisesRegex(MODULE.FixtureError, "non-finite"):
            MODULE.compare_fixture(self.root, candidate_dir)


if __name__ == "__main__":
    unittest.main()
