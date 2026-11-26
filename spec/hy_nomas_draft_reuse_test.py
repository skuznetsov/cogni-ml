"""Fail-closed CPU checks for frozen scene-input reuse and model selection."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/support"))
import hy_nomas_draft_inputs as inputs


class ModelSelectionTests(unittest.TestCase):
    def test_flags_must_be_complete_and_default_remains_base(self):
        self.assertEqual(inputs.model_identity(None, None)["dit_variant"], "original")
        for path, sha in ((Path("/tmp/missing"), None), (None, "a" * 64)):
            with self.assertRaises(inputs.InputReuseError):
                inputs.model_identity(path, sha)

    def test_lineage_and_index_are_both_bound_and_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.json"
            index.write_text("{}")
            lineage = root / "cinematic-lineage.json"
            lineage.write_text(json.dumps({"base_index_sha256": inputs.BASE_INDEX_SHA,
                "index": {"basename": "index.json", "sha256": inputs.digest(index)}}))
            expected = inputs.digest(lineage)
            identity = inputs.model_identity(lineage, expected)
            self.assertEqual(identity["cinematic_lineage"], str(lineage.resolve()))
            index.write_text('{"changed":true}')
            with self.assertRaises(inputs.InputReuseError):
                inputs.model_identity(lineage, expected)


class FrozenReuseTests(unittest.TestCase):
    def make_run(self, root):
        def write(name, value):
            path = root / name
            path.write_text(json.dumps(value))
            return path
        prompt = "exact native prompt"
        noise = {name: str(i) * 64 for i, name in enumerate(inputs.NOISE_NAMES, 1)}
        qwen = write("qwen.json", {})
        qsha = "e" * 64
        base = write("base.json", {"prompt": prompt, "tensors": {
            "input.qwen_bf16": {"sha256": qsha}}})
        payload = root / "payload.bin"
        payload.write_bytes(b"frozen activations")
        def add_payload(path):
            document = json.loads(path.read_text())
            document.update(payload="payload.bin", payload_bytes=payload.stat().st_size,
                            payload_sha256=inputs.digest(payload))
            path.write_text(json.dumps(document))
        add_payload(base)
        timeline = write("timeline.json", {"prompt": prompt,
            "qwen_input_bundle": {"manifest_path": str(base),
                "manifest_sha256": inputs.digest(base), "qwen_tensor_sha256": qsha},
            "qwen_source": {"reference_path": str(qwen), "reference_sha256": inputs.digest(qwen)},
            "tensors": {name: {"sha256": sha} for name, sha in noise.items()}})
        add_payload(timeline)
        receipt = write("receipt.json", {"inputs": str(timeline)})
        row = {"id": "scene", "status": "complete", "native_prompt": prompt,
            "inputs_sha256": inputs.digest(timeline), "noise_sha256": noise,
            "qwen_sha256": inputs.digest(qwen), "qwen_tensor_sha256": qsha,
            "receipt": "receipt.json"}
        suite = b"frozen suite bytes"
        (root / "suite.json").write_bytes(suite)
        report = write("results.json", {"status": "complete", "suite_sha256": hashlib.sha256(suite).hexdigest(),
            "shared_noise_sha256": noise, "scenes": [row]})
        return suite, report, timeline, row

    @mock.patch.object(inputs.results, "validate_completed_result")
    def test_reuse_uses_the_exact_saved_bundle_and_detects_later_changes(self, validator):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, report, timeline, row = self.make_run(root)
            frozen = inputs.FrozenInputs(report, suite, [{"id": "scene"}], {"scene": row["native_prompt"]})
            self.assertEqual(frozen.scene("scene")["continuation"], timeline.resolve())
            self.assertTrue(validator.call_args.kwargs["full_media_validation"])
            payload = root / "payload.bin"
            original = payload.read_bytes()
            payload.write_bytes(b"changed after admission")
            with self.assertRaisesRegex(inputs.InputReuseError, "payload"):
                frozen.scene("scene")
            payload.write_bytes(original)
            timeline.write_text('{}')
            with self.assertRaises(inputs.InputReuseError):
                frozen.scene("scene")

    @mock.patch.object(inputs.results, "validate_completed_result")
    def test_changed_payload_is_rejected_before_any_model_launch(self, _validator):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, report, _timeline, row = self.make_run(root)
            (root / "payload.bin").write_bytes(b"changed activations")
            with self.assertRaisesRegex(inputs.InputReuseError, "payload"):
                inputs.FrozenInputs(report, suite, [{"id": "scene"}], {"scene": row["native_prompt"]})

    @mock.patch.object(inputs.results, "validate_completed_result")
    def test_changed_suite_prompt_or_noise_fails_before_inference(self, _validator):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, report, _timeline, row = self.make_run(root)
            with self.assertRaises(inputs.InputReuseError):
                inputs.FrozenInputs(report, suite + b" ", [{"id": "scene"}], {"scene": row["native_prompt"]})
            with self.assertRaises(inputs.InputReuseError):
                inputs.FrozenInputs(report, suite, [{"id": "scene"}], {"scene": "another prompt"})
            data = json.loads(report.read_text())
            data["scenes"][0]["noise_sha256"][inputs.NOISE_NAMES[0]] = "f" * 64
            report.write_text(json.dumps(data))
            with self.assertRaises(inputs.InputReuseError):
                inputs.FrozenInputs(report, suite, [{"id": "scene"}], {"scene": row["native_prompt"]})


if __name__ == "__main__":
    unittest.main()
