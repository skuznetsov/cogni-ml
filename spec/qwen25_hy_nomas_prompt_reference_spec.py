#!/usr/bin/env python3
"""CPU-only tests for source-pinned HyNomas prompt rendering and tokenization."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples/support/qwen25_hy_nomas_prompt_reference.py"
SPEC = importlib.util.spec_from_file_location("qwen25_hy_nomas_prompt_reference", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the HyNomas prompt reference helper")
prompt_reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prompt_reference)


def _metadata_fixture(crop_start: int = 3) -> tuple[dict, dict]:
    raw_length = prompt_reference.BASE_SEQUENCE_LENGTH + crop_start
    marker_ids = [151644, 872, 198]
    marker_start = crop_start - len(marker_ids)
    token_ids = [17] * raw_length
    token_ids[marker_start:crop_start] = marker_ids
    visible = max(10, crop_start + 1)
    attention_mask = [1] * visible + [0] * (raw_length - visible)
    prompt = "A natural text-only prompt."
    messages = [
        {"role": "system", "content": "Pinned system instruction."},
        {"role": "user", "content": prompt},
    ]
    cropped_ids = token_ids[crop_start:]
    cropped_mask = attention_mask[crop_start:]
    assets = {
        name: {"nbytes": 1, "sha256": (str(index + 1) * 64)}
        for index, name in enumerate(("config.json", "tokenizer.json", "tokenizer_config.json"))
    }
    upstream = {
        "repository_revision": prompt_reference.UPSTREAM_REVISION,
        "encoder_source": {
            "path": "hymm/models/text_encoder/encoder.py",
            "sha256": prompt_reference.UPSTREAM_ENCODER_SHA256,
        },
        "constants_source": {
            "path": "configs/constants.py",
            "sha256": prompt_reference.UPSTREAM_CONSTANTS_SHA256,
        },
        "methods_executed_from_ast": [
            "apply_text_to_template", "calculate_crop_start", "text2tokens"
        ],
    }
    canonical_sha = prompt_reference._canonical_json_sha256
    raw_sha = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    metadata = {
        "schema": prompt_reference.PROMPT_SCHEMA,
        "schema_version": prompt_reference.PROMPT_SCHEMA_VERSION,
        "template_name": prompt_reference.TEMPLATE_NAME,
        "template_sha256": prompt_reference.VIDEO_JSON_TEMPLATE_SHA256,
        "prompt": prompt,
        "prompt_sha256": raw_sha(prompt),
        "rendered_prompt": "<|im_start|>user\nA natural text-only prompt.",
        "rendered_prompt_sha256": raw_sha("<|im_start|>user\nA natural text-only prompt."),
        "messages": messages,
        "marker": "<|im_start|>user\n",
        "marker_token_ids": marker_ids,
        "marker_start": marker_start,
        "crop_start": crop_start,
        "raw_length": raw_length,
        "text_length": prompt_reference.BASE_SEQUENCE_LENGTH,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "cropped_token_ids": cropped_ids,
        "cropped_attention_mask": cropped_mask,
        "raw_token_ids_sha256": canonical_sha(token_ids),
        "raw_mask_sha256": canonical_sha(attention_mask),
        "cropped_token_ids_sha256": canonical_sha(cropped_ids),
        "cropped_attention_mask_sha256": canonical_sha(cropped_mask),
        "padding": {"side": "right", "pad_token_id": 0},
        "tokenizer": {
            "repo_id": prompt_reference.MODEL_ID,
            "revision": prompt_reference.MODEL_REVISION,
            "class": "test.Tokenizer",
            "transformers_version": "4.57.3",
            "local_files_only": True,
            "trust_remote_code": False,
            "receipt_sha256": "a" * 64,
            "assets": assets,
            "runtime": {
                "transformers_version": prompt_reference.TOKENIZER_RUNTIME_TRANSFORMERS_VERSION,
                "python_version": "3.13.1",
                "transformers_sources": {"tokenization_utils.py": {"sha256": "b" * 64}},
            },
            "compatibility_note": (
                "Prompt tokenization used the recorded local Transformers and tokenizers runtime; "
                "the HyNomas model and full pipeline were not executed"
            ),
        },
        "tokenizer_assets": assets,
        "upstream_revision": prompt_reference.UPSTREAM_REVISION,
        "upstream_source": upstream,
        "producer_source_sha256": "f" * 64,
        "execution_scope": (
            "exact prompt/template/crop/tokenization methods extracted from the pinned HyNomas encoder AST; "
            "does not execute the HyNomas model or text encoder"
        ),
        "forbidden_multimodal_token_ids": dict(prompt_reference.REQUIRED_VIDEO_MARKER_IDS),
        "text_only": True,
    }
    positions = [list(range(raw_length)) for _ in range(3)]
    positions_f32 = [[float(value) for value in axis] for axis in positions]
    return metadata, {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "position_ids_f32": positions_f32,
        "position_ids_for_official_model": positions,
        "multimodal_position_shape": [3, 1, raw_length],
        "position_offset": 0,
    }


class HyNomasPromptReferenceSpec(unittest.TestCase):
    def test_accepts_maximum_authenticated_raw_width_and_suffix_crop(self):
        metadata, values = _metadata_fixture(prompt_reference.MAX_CROP_START)
        self.assertEqual(
            prompt_reference.validate_prompt_metadata(
                metadata,
                values["token_ids"],
                values["attention_mask"],
                values["position_ids_f32"],
                values["position_ids_for_official_model"],
                values["multimodal_position_shape"],
                values["position_offset"],
            ),
            prompt_reference.MAX_RAW_SEQUENCE_LENGTH,
        )

    def test_rejects_raw_mask_or_crop_mutations(self):
        metadata, values = _metadata_fixture()
        bad = copy.deepcopy(values["attention_mask"])
        bad[1] = 0
        with self.assertRaisesRegex(ValueError, "token arrays differ"):
            prompt_reference.validate_prompt_metadata(
                metadata,
                values["token_ids"],
                bad,
                values["position_ids_f32"],
                values["position_ids_for_official_model"],
                values["multimodal_position_shape"],
                values["position_offset"],
            )

        malformed = copy.deepcopy(metadata)
        malformed["crop_start"] += 1
        with self.assertRaisesRegex(ValueError, "raw prompt width"):
            prompt_reference.validate_prompt_metadata(
                malformed,
                values["token_ids"],
                values["attention_mask"],
                values["position_ids_f32"],
                values["position_ids_for_official_model"],
                values["multimodal_position_shape"],
                values["position_offset"],
            )

    def test_rejects_a_prior_exact_user_marker_occurrence(self):
        metadata, values = _metadata_fixture(crop_start=8)
        malformed = copy.deepcopy(metadata)
        malformed_values = copy.deepcopy(values)
        malformed_values["token_ids"][:3] = malformed["marker_token_ids"]
        malformed["token_ids"] = malformed_values["token_ids"]
        malformed["raw_token_ids_sha256"] = prompt_reference._canonical_json_sha256(
            malformed_values["token_ids"]
        )

        with self.assertRaisesRegex(ValueError, "first exact occurrence"):
            prompt_reference.validate_prompt_metadata(
                malformed,
                malformed_values["token_ids"],
                malformed_values["attention_mask"],
                malformed_values["position_ids_f32"],
                malformed_values["position_ids_for_official_model"],
                malformed_values["multimodal_position_shape"],
                malformed_values["position_offset"],
            )

    def test_rejects_untrusted_prompt_hash_source_revision_and_multimodal_ids(self):
        metadata, values = _metadata_fixture()
        cases = (
            ("prompt_sha256", "0" * 64, "prompt text digest"),
            ("upstream_revision", "main", "upstream revision"),
            ("template_sha256", "0" * 64, "template identity"),
        )
        for field, value, message in cases:
            malformed = copy.deepcopy(metadata)
            malformed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                prompt_reference.validate_prompt_metadata(
                    malformed,
                    values["token_ids"],
                    values["attention_mask"],
                    values["position_ids_f32"],
                    values["position_ids_for_official_model"],
                    values["multimodal_position_shape"],
                    values["position_offset"],
                )
        malformed = copy.deepcopy(metadata)
        malformed["forbidden_multimodal_token_ids"].pop("<|vision_pad|>")
        with self.assertRaisesRegex(ValueError, "text-only token inventory"):
            prompt_reference.validate_prompt_metadata(
                malformed,
                values["token_ids"],
                values["attention_mask"],
                values["position_ids_f32"],
                values["position_ids_for_official_model"],
                values["multimodal_position_shape"],
                values["position_offset"],
            )
        malformed = copy.deepcopy(metadata)
        malformed["forbidden_multimodal_token_ids"]["<|vision_pad|>"] = values["token_ids"][0]
        with self.assertRaisesRegex(ValueError, "text-only token inventory"):
            prompt_reference.validate_prompt_metadata(
                malformed,
                values["token_ids"],
                values["attention_mask"],
                values["position_ids_f32"],
                values["position_ids_for_official_model"],
                values["multimodal_position_shape"],
                values["position_offset"],
            )

    @unittest.skipUnless(
        os.environ.get("QWEN25_PROMPT_INTEGRATION") == "1",
        "set QWEN25_PROMPT_INTEGRATION=1 to exercise pinned local tokenizer assets",
    )
    def test_pinned_upstream_methods_and_tokenizer_produce_natural_prompt(self):
        checkpoint_root = Path(os.environ["QWEN25_PROMPT_CHECKPOINT_ROOT"])
        receipt = Path(os.environ["QWEN25_PROMPT_CHECKPOINT_RECEIPT"])
        upstream = Path(os.environ["QWEN25_HYNOMAS_ROOT"])
        prompt = "A yellow tram turns through a wet downtown intersection at dusk as pedestrians wait under umbrellas."
        with tempfile.TemporaryDirectory() as temporary:
            prompt_file = Path(temporary) / "prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            request, metadata = prompt_reference.build_prompt(
                checkpoint_root, receipt, upstream, prompt_file
            )
        self.assertEqual(metadata["template_name"], "li-dit-encode-video-json")
        self.assertEqual(metadata["crop_start"], 108)
        self.assertEqual(metadata["marker_token_ids"], [151644, 872, 198])
        self.assertEqual(
            {name: metadata["forbidden_multimodal_token_ids"][name] for name in prompt_reference.REQUIRED_VIDEO_MARKER_IDS},
            prompt_reference.REQUIRED_VIDEO_MARKER_IDS,
        )
        self.assertEqual(request["sequence_length"], 1388)
        self.assertEqual(metadata["text_length"], 1280)
        self.assertEqual(len(metadata["cropped_token_ids"]), 1280)
        self.assertEqual(metadata["tokenizer"]["revision"], prompt_reference.MODEL_REVISION)
        self.assertEqual(metadata["upstream_revision"], prompt_reference.UPSTREAM_REVISION)
        self.assertEqual(sum(request["attention_mask"]), 131)
        self.assertTrue(
            prompt_reference.validate_prompt_metadata(
                metadata,
                request["token_ids"],
                request["attention_mask"],
                request["position_ids_f32"],
                request["position_ids_for_official_model"],
                request["multimodal_position_shape"],
                request["position_offset"],
            )
        )


if __name__ == "__main__":
    unittest.main()
