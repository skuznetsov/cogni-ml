#!/usr/bin/env python3
"""CPU-only admission tests for fixed Hy-Nomas generation inputs."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_generation_inputs as inputs  # noqa: E402

PROMPT_SPEC_PATH = Path(__file__).resolve().with_name("qwen25_hy_nomas_prompt_reference_spec.py")
PROMPT_SPEC_INFO = importlib.util.spec_from_file_location(
    "qwen25_hy_nomas_prompt_reference_fixture", PROMPT_SPEC_PATH
)
if PROMPT_SPEC_INFO is None or PROMPT_SPEC_INFO.loader is None:
    raise RuntimeError("could not load the Qwen prompt fixture builder")
PROMPT_SPEC = importlib.util.module_from_spec(PROMPT_SPEC_INFO)
PROMPT_SPEC_INFO.loader.exec_module(PROMPT_SPEC)


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


def valid_dynamic_document() -> dict:
    contract, _ = PROMPT_SPEC._metadata_fixture()
    contract["producer_source_sha256"] = inputs._hash_file(
        inputs.ROOT / "examples/support/qwen25_hy_nomas_prompt_reference.py"
    )
    mask = list(contract["cropped_attention_mask"])
    prompt = contract["prompt"]
    source = {
        "reference_schema": inputs.QWEN_REFERENCE_SCHEMA,
        "reference_schema_version": inputs.QWEN_REFERENCE_SCHEMA_VERSION,
        "reference_path": "/tmp/qwen-reference.json",
        "reference_sha256": "a" * 64,
        "sidecar_path": "/tmp/qwen-reference.bf16le",
        "sidecar_sha256": "b" * 64,
        "sidecar_bytes": 27 * contract["raw_length"] * inputs.QWEN_HIDDEN_SIZE * 2,
        "sidecar_encoding": "bfloat16-le",
        "state_index": inputs.QWEN_PREFIX_STATE_INDEX,
        "crop": [contract["crop_start"], contract["raw_length"]],
        "crop_tokens": inputs.QWEN_BASE_SEQUENCE_LENGTH,
        "raw_sequence_length": contract["raw_length"],
        "hidden_size": inputs.QWEN_HIDDEN_SIZE,
        "valid_tokens": sum(mask),
        "cropped_attention_mask": mask,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_contract": contract,
        "upstream_revision": contract["upstream_revision"],
        "template_name": contract["template_name"],
        "template_sha256": contract["template_sha256"],
        "source_code_hashes": {
            "oracle_producer": inputs._hash_file(
                inputs.ROOT / "examples/support/qwen25_pretrained_reference.py"
            ),
            "prompt_helper": contract["producer_source_sha256"],
            "upstream_encoder": contract["upstream_source"]["encoder_source"]["sha256"],
            "upstream_constants": contract["upstream_source"]["constants_source"]["sha256"],
        },
    }
    document = valid_document()
    document["schema"] = inputs.DYNAMIC_SCHEMA
    document["schema_version"] = inputs.DYNAMIC_SCHEMA_VERSION
    document["prompt"] = prompt
    document["qwen_source"] = source
    document["geometry"]["qwen_crop"] = source["crop"]
    document["geometry"]["qwen_valid_tokens"] = sum(mask)
    return document


def valid_high_resolution_document() -> dict:
    document = valid_dynamic_document()
    document["schema"] = inputs.HIGH_RESOLUTION_SCHEMA
    document["schema_version"] = inputs.HIGH_RESOLUTION_SCHEMA_VERSION
    document["geometry"] = inputs._geometry_for_resolution(
        inputs.HIGH_RESOLUTION,
        valid_tokens=document["qwen_source"]["valid_tokens"],
        crop=document["qwen_source"]["crop"],
    )
    cursor = 0
    records = {}
    specs = inputs.tensor_specs_for_resolution(inputs.HIGH_RESOLUTION)
    for name, dtype, shape in specs:
        nbytes = inputs.tensor_nbytes(dtype, shape)
        records[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "offset": cursor,
            "nbytes": nbytes,
            "sha256": "a" * 64,
        }
        cursor += nbytes
    document["tensors"] = records
    document["payload_bytes"] = cursor
    return document


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

    def test_dynamic_prompt_bundle_admits_pinned_crop_and_mask_contract(self):
        document = valid_dynamic_document()
        inputs.validate_document(document)
        self.assertEqual(document["qwen_source"]["valid_tokens"], 7)
        self.assertEqual(document["geometry"]["qwen_crop"], [3, 1283])

    def test_opt_in_768x416_v3_has_derived_dit_decoder_and_payload_extents(self):
        document = valid_high_resolution_document()
        specs = inputs.tensor_specs_for_resolution(inputs.HIGH_RESOLUTION)
        shapes = {name: list(shape) for name, _, shape in specs}
        self.assertEqual(shapes["input.latent"], [1, 32, 30, 13, 24])
        self.assertEqual(shapes["input.rope_cos"], [9360, 128])
        self.assertEqual(shapes["decoder.noise"], [1, 3, 120, 416, 768])
        self.assertEqual(shapes["decoder.rope_cos"], [1, 12480, 1, 128])
        self.assertEqual(document["geometry"]["pixel_width"], 768)
        self.assertEqual(document["geometry"]["pixel_height"], 416)
        self.assertEqual(
            document["payload_bytes"], inputs.expected_payload_bytes(specs)
        )
        self.assertEqual(document["payload_bytes"], 492_800_000)
        inputs.validate_document(document)

    def test_opt_in_v3_rejects_geometry_and_tensor_extent_drift(self):
        cases = (
            (
                "decoder RoPE grid",
                lambda doc: doc["geometry"].__setitem__("decoder_rope_grid", [10, 18, 48]),
            ),
            (
                "pixel width",
                lambda doc: doc["geometry"].__setitem__("pixel_width", 512),
            ),
            (
                "latent tensor shape",
                lambda doc: doc["tensors"]["input.latent"].__setitem__(
                    "shape", [1, 32, 30, 9, 16]
                ),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                document = valid_high_resolution_document()
                mutate(document)
                with self.assertRaises(inputs.InputsError):
                    inputs.validate_document(document)

    def test_dynamic_prompt_bundle_rejects_mask_crop_and_source_drift(self):
        cases = (
            (
                "valid-token limit",
                lambda doc: self._set_dynamic_mask(doc, [1] * 1001 + [0] * 279),
                "valid-token count is outside",
            ),
            (
                "visible prefix",
                lambda doc: self._set_dynamic_mask(doc, [0] + doc["qwen_source"]["cropped_attention_mask"][1:]),
                "visible prefix",
            ),
            (
                "crop geometry",
                lambda doc: doc["qwen_source"]["crop"].__setitem__(0, 4),
                "crop geometry",
            ),
            (
                "source implementation",
                lambda doc: doc["qwen_source"]["source_code_hashes"].__setitem__(
                    "oracle_producer", "c" * 64
                ),
                "source-code hash binding mismatch: oracle_producer",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label):
                document = valid_dynamic_document()
                mutate(document)
                with self.assertRaisesRegex(inputs.InputsError, message):
                    inputs.validate_document(document)

    @staticmethod
    def _set_dynamic_mask(document: dict, mask: list[int]) -> None:
        source = document["qwen_source"]
        source["cropped_attention_mask"] = mask
        source["valid_tokens"] = sum(mask)
        source["prompt_contract"]["cropped_attention_mask"] = list(mask)
        source["prompt_contract"]["cropped_attention_mask_sha256"] = (
            inputs._prompt_reference_helper()._canonical_json_sha256(mask)
        )
        document["geometry"]["qwen_valid_tokens"] = sum(mask)

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
