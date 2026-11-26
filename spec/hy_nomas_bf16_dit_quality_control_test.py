"""CPU falsifiers for bounded BF16-control admission and no-clobber export."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_bf16_dit_quality_control as control


class ControlAdmissionTests(unittest.TestCase):
    def shard(self, root, descriptors, payload=b"\0" * 4):
        header = json.dumps(descriptors).encode()
        path = Path(root) / "weights.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        return path

    def test_non_bf16_weight_storage_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.shard(root, {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
            with self.assertRaisesRegex(control.ControlError, "BF16"):
                control._read_safetensors_header(path)

    def test_overlapping_tensor_extents_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.shard(root, {
                "w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                "b": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            })
            with self.assertRaisesRegex(control.ControlError, "overlap|gap"):
                control._read_safetensors_header(path)

    def test_weight_mutation_cannot_pass_its_trusted_digest(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.shard(root, {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(path.read_bytes()[:-1] + b"\1")
            with self.assertRaisesRegex(control.ControlError, "hash mismatch"):
                control._validate_cache_shard(path, digest, {"w": [2]})

    def test_invalid_process_caps_are_rejected(self):
        for value in ("nan", "inf", "0", "27"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    control._parse_args(["--validate-only", "--planned-cap-gib", value])

    def test_occupied_result_is_rejected_before_model_work(self):
        with tempfile.TemporaryDirectory() as root:
            result = Path(root) / "result.json"
            result.write_bytes(b"keep")
            with self.assertRaisesRegex(control.ControlError, "new|exist|fresh"):
                control._validate_output_target(result)
            self.assertEqual(result.read_bytes(), b"keep")

    def test_existing_latent_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            result = Path(root) / "result.json"
            latent = Path(root) / "result.latent.f32le"
            latent.write_bytes(b"keep")
            with self.assertRaisesRegex(control.ControlError, "new|exist|fresh"):
                control._publish(result, {"result": {}}, b"replacement")
            self.assertEqual(latent.read_bytes(), b"keep")
            self.assertFalse(result.exists())

    def test_fresh_publication_binds_payload_and_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            result = Path(root) / "result.json"
            payload = struct.pack("<f", 0.5)
            document = {"result": {"payload_sha256": hashlib.sha256(payload).hexdigest()}}
            latent, digest = control._publish(result, document, payload)
            self.assertEqual(latent.read_bytes(), payload)
            self.assertEqual(hashlib.sha256(result.read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads(result.read_text())["result"]["payload"], latent.name)


class RefinerBindingTests(unittest.TestCase):
    def fixture(self):
        module = types.ModuleType("hymm.models.text_encoder.token_refiner")
        exec(
            "class IndividualTokenRefinerBlock:\n"
            "    def forward(self):\n"
            "        return attention()\n"
            "class SingleTokenRefinerPerBlock:\n"
            "    pass\n",
            module.__dict__,
        )
        instance = module.SingleTokenRefinerPerBlock()
        instance.individual_token_refiner = types.SimpleNamespace(
            blocks=[module.IndividualTokenRefinerBlock() for _ in range(2)]
        )
        return module, instance

    def adapter(self, state):
        def attention():
            return state
        return attention

    def test_preflight_import_is_rebound_to_the_current_run(self):
        module, instance = self.fixture()
        stale, current = {}, {}
        module.attention = self.adapter(stale)
        adapter = self.adapter(current)
        with mock.patch.dict(sys.modules, {module.__name__: module}), mock.patch.object(
            control.block_reference, "_namespace"
        ):
            control._bind_refiner_attention_adapter(Path("/unused"), adapter, current, instance)
        for block in instance.individual_token_refiner.blocks:
            self.assertIs(block.forward(), current)

    def test_adapter_with_another_runs_state_is_rejected(self):
        module, instance = self.fixture()
        adapter = self.adapter({})
        with mock.patch.dict(sys.modules, {module.__name__: module}), mock.patch.object(
            control.block_reference, "_namespace"
        ):
            with self.assertRaisesRegex(control.ControlError, "state"):
                control._bind_refiner_attention_adapter(Path("/unused"), adapter, {}, instance)


if __name__ == "__main__":
    unittest.main()
