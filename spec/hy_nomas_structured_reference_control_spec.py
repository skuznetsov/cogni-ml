"""CPU-only admission and source-range checks for structured continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_structured_reference_control as control


PINNED_911 = Path(control.base.single.DEFAULT_UPSTREAM)


def _profile_fixture() -> tuple[dict[str, object], dict[str, object], str, str]:
    prompt = (
        '{"Style":"Warm animation","segments":[{"start_time":"00:05.000",'
        '"end_time":"00:10.000","description_change":"Resume stays nearby",'
        '"contained_actions":[{"start_time":"00:00.500","end_time":"00:04.500",'
        '"action_description":"<Action1><|Subject1|> turns his head toward Resume</Action1>."}]}]}'
    )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    manifest_sha = "a" * 64
    payload_sha = "b" * 64
    qwen_source = {
        "prompt_contract": {"prompt": prompt},
        "prompt_sha256": prompt_sha,
        "valid_tokens": 317,
    }
    t1_qwen = {
        "qwen_source": qwen_source,
        "document": {"qwen_source": qwen_source},
        "payload_sha256": payload_sha,
    }
    profile = {
        "schema": control.PROFILE_SCHEMA,
        "conditioning_manifest_sha256": manifest_sha,
        "conditioning_payload_sha256": payload_sha,
        "prompt_sha256": prompt_sha,
        "prompt": prompt,
        "valid_text_tokens": 317,
        "geometry": {
            "run_segments": 2,
            "segment_index": 1,
            "segment_start_seconds": 5.0,
            "segment_end_seconds": 10.0,
            "latent_frames": 30,
            "tokens_per_frame": 144,
            "video_tokens": 4320,
            "text_length": 1280,
        },
        "token_contract": {
            "template_name": control.action_conditioning.TEMPLATE_NAME,
            "template_sha256": control.PROFILE_TEMPLATE_SHA256,
            "text_length": 1280,
            "valid_text_tokens": 317,
            "prompt_sha256": prompt_sha,
        },
        "source": {
            "upstream_revision": control.action_conditioning.UPSTREAM_REVISION,
            "source_sha256": control.action_conditioning.UPSTREAM_SOURCE_SHA256,
            "time_origin": "segment_local_seconds",
            "resolve_subject_ids": False,
            "action_wrapper_representation": "literal_text_markup_not_token_ids",
        },
        "text_offsets": {
            "head_end": 201,
            "action_start": 201,
            "action_end": 317,
            "valid_text_tokens": 317,
            "source_offsets_raw": [201, 317],
            "effective_offsets": [201, 317],
            "offset_method_endpoint_note": (
                "BasePipeline returns the raw attention-mask sum; pinned nomas_cpt clamps it "
                "with min(offset, valid current text query tokens), yielding the authenticated cropped-mask count"
            ),
        },
        "action": {
            "action_id": 1,
            "subject_id": "<|Subject1|>",
            "time_origin": "segment_local_seconds",
            "source_segment_start_seconds": 5.0,
            "source_segment_end_seconds": 10.0,
            "start_time_local_seconds": 0.5,
            "end_time_local_seconds": 4.5,
            "latent_start": 3,
            "latent_end": 27,
            "action_description": "<Action1><|Subject1|> turns his head toward Resume</Action1>.",
        },
        "video_windows": [[0, 3, -1], [3, 27, 0], [27, 30, -1]],
        "intra_segment_active": True,
        "labels": {
            "global": 0,
            "action": 1,
            "padding": -1,
            "video_gap_window_index": -1,
            "query_scope": "global_head_and_matching_action_window",
            "key_scope": "global_history_or_matching_action",
        },
    }
    return profile, t1_qwen, manifest_sha, payload_sha


class StructuredReferenceAdmissionTests(unittest.TestCase):
    def test_dit_rows_round_up_dynamic_caption_and_action_prefixes(self):
        self.assertEqual(control._round_dit_text_rows(421, 317), 448)
        self.assertEqual(control._round_dit_text_rows(421, 557), 576)
        self.assertEqual(control._round_dit_text_rows(1000, 1280), 1280)
        with self.assertRaisesRegex(control.StructuredControlError, "outside Qwen"):
            control._round_dit_text_rows(0, 317)
        with self.assertRaisesRegex(control.StructuredControlError, "outside Qwen"):
            control._round_dit_text_rows(421, 1281)

    @unittest.skipUnless(PINNED_911.is_dir(), "pinned 911 upstream checkout is not mounted")
    def test_action_profile_binds_t1_manifest_payload_prompt_and_source(self):
        profile, t1_qwen, manifest_sha, _payload_sha = _profile_fixture()
        admitted = control._validate_profile_bindings(
            profile,
            "c" * 64,
            t1_qwen,
            manifest_sha,
            PINNED_911,
        )
        self.assertEqual(admitted["valid_text_tokens"], 317)
        self.assertEqual(admitted["source_range_offsets"], [201, 317])
        self.assertEqual(admitted["video_windows"], [[0, 3, -1], [3, 27, 0], [27, 30, -1]])

        wrong_payload = dict(t1_qwen, payload_sha256="d" * 64)
        with self.assertRaisesRegex(control.StructuredControlError, "does not bind"):
            control._validate_profile_bindings(
                profile, "c" * 64, wrong_payload, manifest_sha, PINNED_911
            )
        with self.assertRaisesRegex(control.StructuredControlError, "does not bind"):
            control._validate_profile_bindings(
                profile, "c" * 64, t1_qwen, "e" * 64, PINNED_911
            )

        detached_action = dict(profile, action=dict(profile["action"]))
        detached_action["action"]["action_description"] = (
            "<Action1><|Subject1|> turns his head away from Resume</Action1>."
        )
        with self.assertRaisesRegex(control.StructuredControlError, "differs from the bound prompt action"):
            control._validate_profile_bindings(
                detached_action, "c" * 64, t1_qwen, manifest_sha, PINNED_911
            )

        detached_time = dict(profile, action=dict(profile["action"]))
        detached_time["action"]["start_time_local_seconds"] = 1.0
        with self.assertRaisesRegex(control.StructuredControlError, "differs from the bound prompt action"):
            control._validate_profile_bindings(
                detached_time, "c" * 64, t1_qwen, manifest_sha, PINNED_911
            )

        detached_offsets = dict(profile, text_offsets=dict(profile["text_offsets"]))
        detached_offsets["text_offsets"]["effective_offsets"] = [200, 317]
        with self.assertRaisesRegex(control.StructuredControlError, "text offsets"):
            control._validate_profile_bindings(
                detached_offsets, "c" * 64, t1_qwen, manifest_sha, PINNED_911
            )

    def test_t0_caption_has_one_scene_and_one_attributed_subject1_alex(self):
        control._validate_t0_subject1_alex_binding(
            json.dumps({
                "subject_list": [{
                    "subject_id": "<|Subject1|>",
                    "display_name": "Alex",
                    "subject_attribute": "adult man",
                }],
                "segments": [{"start_time": "00:00.000", "end_time": "00:05.000", "contained_actions": []}],
            })
        )
        invalid_prompts = [
            {"subject_list": [
                {"subject_id": "<|Subject1|>", "display_name": "Alex", "subject_attribute": "adult man"},
                {"subject_id": "<|Subject2|>", "display_name": "Resume", "subject_attribute": "paper"},
            ], "segments": [{"contained_actions": []}]},
            {"subject_list": [{
                "subject_id": "<|Subject1|>", "display_name": "Alex", "subject_attribute": "  "
            }], "segments": [{"contained_actions": []}]},
            {"subject_list": [{
                "subject_id": "<|Subject1|>", "display_name": "Alex", "subject_attribute": "adult man"
            }], "segments": [{"contained_actions": [{"action_description": "tail"}]}]},
            {"subject_list": [{
                "subject_id": "<|Subject1|>", "display_name": "Alex", "subject_attribute": "adult man"
            }], "segments": [{"contained_actions": []}, {"contained_actions": []}]},
        ]
        for prompt in invalid_prompts:
            with self.subTest(prompt=prompt):
                with self.assertRaises(control.StructuredControlError):
                    control._validate_t0_subject1_alex_binding(json.dumps(prompt))

    def test_dynamic_base_contract_restores_legacy_constants(self):
        before = (control.base.ACTION_VALID_TOKENS, control.base.DIT_TEXT_ROWS,
                  control.base.CACHE_CAPACITY)
        with control._bind_dynamic_base_contract(317, 448):
            self.assertEqual(control.base.ACTION_VALID_TOKENS, 317)
            self.assertEqual(control.base.DIT_TEXT_ROWS, 448)
            self.assertEqual(control.base.CACHE_CAPACITY, control.base.VIDEO_TOKENS + 448)
        self.assertEqual(
            (control.base.ACTION_VALID_TOKENS, control.base.DIT_TEXT_ROWS,
             control.base.CACHE_CAPACITY),
            before,
        )


class StructuredSourceRangeOracleTests(unittest.TestCase):
    def test_t0_dispatch_uses_captured_factory_after_base_is_patched(self):
        marker = object()
        legacy_dispatch = Mock(return_value=marker)
        captured_factory = Mock(return_value=legacy_dispatch)
        recursive_factory = Mock(side_effect=AssertionError("patched factory must not be reentered"))
        state = {"receipt": {}, "t0_valid_tokens": 3, "call": {"condition": "t0_caption"}}
        profile = {"tokens_per_frame": 144, "valid_text_tokens": 7,
                   "source_range_offsets": [2, 7], "video_windows": [[0, 30, 0]]}
        record = {"range_count": 1}
        with patch.object(control.base, "_cache_attention_dispatch_factory", recursive_factory), \
                patch.object(control, "_build_pinned_range_builder", return_value=(object(), "a" * 64, 1, 2)), \
                patch.object(control, "_source_mask_chunks_cpu", return_value=([], record)):
            dispatch = control._action_cache_attention_dispatch_factory(
                torch, state, profile, PINNED_911, legacy_factory=captured_factory,
            )
            self.assertIs(dispatch(None, None, None, None, None, 0, True, True), marker)
        captured_factory.assert_called_once_with(torch, state)
        legacy_dispatch.assert_called_once_with(None, None, None, None, None, 0, True, True)
        recursive_factory.assert_not_called()
        self.assertEqual(state["receipt"]["t0_unmasked_cache_attention_calls"], 1)

    @unittest.skipUnless(PINNED_911.is_dir(), "pinned 911 upstream checkout is not mounted")
    def test_source_range_union_matches_global_action_labels_and_legacy_control(self):
        torch.set_num_threads(1)
        builder, source_sha, _line_start, _line_end = control._build_pinned_range_builder(
            PINNED_911, torch
        )
        self.assertEqual(source_sha, "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee")
        chunks, record = control._source_mask_chunks_cpu(
            torch,
            builder,
            history_rows=3,
            image_rows=80,
            valid_text_rows=7,
            offsets=[2, 7],
            video_windows=[[0, 24, -1], [24, 72, 0], [72, 80, -1]],
            query_chunk=32,
        )
        source_mask = torch.cat(chunks, dim=0)
        query_labels = [0] * 24 + [1] * 48 + [0] * 8 + [0] * 2 + [1] * 5
        key_labels = [0] * 3 + [0] * 80 + [0] * 2 + [1] * 5
        semantic_mask = torch.tensor([
            [key == 0 or (query > 0 and key == query) for key in key_labels]
            for query in query_labels
        ], dtype=torch.bool)
        self.assertTrue(torch.equal(source_mask, semantic_mask))
        self.assertEqual(record["query_rows"], 87)
        self.assertEqual(record["key_rows"], 90)
        self.assertEqual(record["query_chunk"], 32)

        q = torch.arange(87 * 2 * 4, dtype=torch.float32).reshape(1, 2, 87, 4) / 97
        k = torch.arange(90 * 2 * 4, dtype=torch.float32).reshape(1, 2, 90, 4) / 83
        v = torch.sin(torch.arange(90 * 2 * 4, dtype=torch.float32)).reshape(1, 2, 90, 4)
        v[:, :, 3 + 80 + 2:3 + 80 + 7] = torch.tensor([128.0, -96.0, 64.0, -32.0])
        masked_chunks = [
            F.scaled_dot_product_attention(
                q[:, :, start:start + mask.shape[0]], k, v,
                attn_mask=mask[None, None, :, :], dropout_p=0.0, is_causal=False,
            )
            for start, mask in zip(range(0, 87, 32), chunks)
        ]
        masked = torch.cat(masked_chunks, dim=2)
        direct = F.scaled_dot_product_attention(
            q, k, v, attn_mask=semantic_mask[None, None], dropout_p=0.0, is_causal=False
        )
        legacy = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        self.assertTrue(torch.allclose(masked, direct, atol=2e-5, rtol=0.0))
        self.assertFalse(torch.allclose(masked, legacy, atol=2e-5, rtol=0.0))

        # The production adapter emits compact active queries then scatters into
        # a full padded query block; source padding is restored as exact zeros.
        padded = torch.zeros((1, 2, 89, 4), dtype=torch.float32)
        padded[:, :, :87] = masked
        self.assertTrue(torch.equal(padded[:, :, 87:], torch.zeros_like(padded[:, :, 87:])))


if __name__ == "__main__":
    unittest.main()
