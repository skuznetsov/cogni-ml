#!/usr/bin/env python3
"""CPU-only admission tests for fixed Hy-Nomas generation inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_generation_inputs as inputs  # noqa: E402


def valid_document() -> dict:
    cursor = 0
    tensors = {}
    for name, dtype, shape in inputs.TENSOR_SPECS:
        nbytes = inputs.tensor_nbytes(dtype, shape)
        tensors[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "offset": cursor,
            "nbytes": nbytes,
            "sha256": "a" * 64,
        }
        cursor += nbytes
    return {
        "schema": inputs.SCHEMA,
        "schema_version": inputs.SCHEMA_VERSION,
        "payload": "preview.tensors.bin",
        "payload_bytes": cursor,
        "payload_sha256": "b" * 64,
        "tensors": tensors,
        "geometry": copy.deepcopy(inputs.GEOMETRY),
        "prompt": inputs.PROMPT,
        "qwen_source": copy.deepcopy(inputs.QWEN_SOURCE_RECORD),
        "noise_seeds": copy.deepcopy(inputs.NOISE_SEEDS),
        "upstream_revision": inputs.PINNED_REVISION,
        "upstream_sources": copy.deepcopy(inputs.PINNED_UPSTREAM_SOURCE_SHA256),
        "producer": {
            **copy.deepcopy(inputs.PRODUCER_RECORD),
            "torch_version": "2.6.0",
            "python_version": "3.12.0",
        },
    }


class HyNomasGenerationInputsTests(unittest.TestCase):
    def test_canonical_inventory_has_expected_payload_budget(self):
        self.assertEqual(
            [name for name, _, _ in inputs.TENSOR_SPECS],
            [
                "input.latent",
                "input.qwen_bf16",
                "input.rope_cos",
                "input.rope_sin",
                "decoder.noise",
                "decoder.rope_cos",
                "decoder.rope_sin",
            ],
        )
        self.assertEqual(inputs.expected_payload_bytes(), 232_386_560)
        self.assertEqual(inputs.GEOMETRY["qwen_valid_tokens"], 23)
        self.assertEqual(inputs.NOISE_SEEDS["pixel_noise_seeds"][0], 42 * 1_000_003)
        self.assertEqual(inputs.NOISE_SEEDS["pixel_noise_seeds"][-1], 42 * 1_000_003 + 29)

    def test_upstream_pin_matches_existing_dit_reference(self):
        self.assertEqual(
            inputs.PINNED_UPSTREAM_SOURCE_SHA256[
                "hymm/models/dit/modules/posemb_layers.py"
            ],
            "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa",
        )

    def test_valid_document_and_one_byte_truncation(self):
        document = valid_document()
        inputs.validate_document(document)
        with self.assertRaisesRegex(inputs.InputsError, "payload byte count"):
            inputs.validate_document(document, payload_bytes=document["payload_bytes"] - 1)

    def test_published_sorted_json_round_trip_preserves_binary_order(self):
        document = json.loads(json.dumps(valid_document(), sort_keys=True))
        self.assertNotEqual(list(document["tensors"]), [name for name, _, _ in inputs.TENSOR_SPECS])
        inputs.validate_document(document)
        document["tensors"]["input.latent"]["offset"] = 1
        with self.assertRaisesRegex(inputs.InputsError, "tensor record mismatch"):
            inputs.validate_document(document)

    def test_admission_rejects_shape_dtype_offset_seed_and_prompt_drift(self):
        mutations = (
            ("shape", lambda doc: doc["tensors"]["input.latent"].__setitem__("shape", [1, 32, 30, 9, 15])),
            ("dtype", lambda doc: doc["tensors"]["input.qwen_bf16"].__setitem__("dtype", "F32")),
            ("offset", lambda doc: doc["tensors"]["decoder.noise"].__setitem__("offset", 1)),
            ("seed", lambda doc: doc["noise_seeds"]["pixel_noise_seeds"].__setitem__(0, 42)),
            ("prompt", lambda doc: doc.__setitem__("prompt", "different prompt")),
            ("source digest", lambda doc: doc["upstream_sources"].__setitem__(
                "hymm/models/vae/pixelumm_decoder/model.py", "c" * 64
            )),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                document = valid_document()
                mutate(document)
                with self.assertRaises(inputs.InputsError):
                    inputs.validate_document(document)

    def test_output_paths_require_new_sibling_bundle_under_allowed_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, payload = inputs.validate_output_paths(
                root / "preview.json", allowed_root=root
            )
            self.assertEqual(manifest, root / "preview.json")
            self.assertEqual(payload, root / "preview.tensors.bin")

            payload.touch()
            with self.assertRaisesRegex(inputs.InputsError, "already exists"):
                inputs.validate_output_paths(root / "preview.json", allowed_root=root)


if __name__ == "__main__":
    unittest.main()
