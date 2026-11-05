"""CPU-only checks for the bounded T1 action prompt contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "support"))
import hy_nomas_action_conditioning as conditioning


def _prompt(start: str = "00:00.500", end: str = "00:04.500",
            action: str = "<Action1><|Subject1|> turns his head to the right and looks toward Resume</Action1>.",
            segment_start: str = "00:05.000") -> str:
    return json.dumps({
        "Style": "Warm hand-drawn character animation",
        "segments": [{
            "start_time": segment_start,
            "end_time": "00:10.000",
            "description_change": "Resume remains on the desk beside the laptop.",
            "contained_actions": [{
                "start_time": start,
                "end_time": end,
                "action_description": action,
            }],
        }],
    }, ensure_ascii=False)


def _v2_document(actions: list[dict[str, str]] | None = None) -> dict[str, object]:
    return {
        "Style": "Warm hand-drawn character animation",
        "segments": [{
            "start_time": "00:05.000",
            "end_time": "00:10.000",
            "description_change": "Resume looks back at Alex beside the laptop.",
            "contained_actions": actions or [
                {
                    "start_time": "00:00.500",
                    "end_time": "00:02.500",
                    "action_description": (
                        "<Action1><|Subject1|> turns his head from the laptop toward Resume. "
                        "His hands stay on the desk.</Action1>."
                    ),
                },
                {
                    "start_time": "00:03.000",
                    "end_time": "00:04.500",
                    "action_description": (
                        "<Action2><|Subject2|> looks back at Alex, raises its white-gloved hand, "
                        "bends its arm and points at its own printed chest.</Action2>."
                    ),
                },
            ],
        }],
    }


def _v2_prompt(actions: list[dict[str, str]] | None = None) -> str:
    return json.dumps(_v2_document(actions), ensure_ascii=False)


def _v2_reference_inputs() -> tuple[dict[str, object], dict[str, object]]:
    caption: dict[str, object] = {
        "Style": "Warm hand-drawn animation",
        "subject_list": [
            {"subject_id": "<|Subject1|>", "display_name": "Alex", "subject_attribute": "Teal shirt, seated left"},
            {"subject_id": "<|Subject2|>", "display_name": "Resume", "subject_attribute": "Cream paper, white gloves, standing right"},
        ],
        "segments": [{
            "start_time": "00:00.000",
            "end_time": "00:05.000",
            "visual_description_long": "Alex and Resume sit together at the desk.",
            "description_change": None,
            "contained_actions": [],
        }],
    }
    followup: dict[str, object] = {"segments": [{
        "start_time": "00:05.000",
        "end_time": "00:10.000",
        "visual_description_long": "Dropped after the first segment.",
        "description_change": "Resume looks back at Alex beside the laptop.",
        "contained_actions": [
            {
                "start_time": "00:05.500",
                "end_time": "00:07.500",
                "action_description": "<Action1><|Subject1|> turns his head toward Resume. His hands stay on the desk.</Action1>.",
            },
            {
                "start_time": "00:08.000",
                "end_time": "00:09.500",
                "action_description": "<Action2><|Subject2|> raises one white glove, taps its own chest once, then lowers the glove to its side.</Action2>.",
            },
        ],
    }]}
    return caption, followup


class _FakeTensor:
    def __init__(self, values: list[int]):
        self.values = values

    def squeeze(self, _dimension: int = 0) -> "_FakeTensor":
        return self

    def tolist(self) -> list[int]:
        return list(self.values)

    def sum(self) -> "_FakeScalar":
        return _FakeScalar(sum(self.values))


class _FakeScalar:
    def __init__(self, value: int):
        self.value = value

    def item(self) -> int:
        return self.value


class _FakeBatchedTokens:
    def __init__(self, rows: list[list[int]]):
        self.rows = rows

    def tolist(self) -> list[list[int]]:
        return [list(row) for row in self.rows]


class _FakeNoTruncationTokenizer:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids

    def apply_chat_template(self, _messages: list[object], **_kwargs: object) -> dict[str, _FakeBatchedTokens]:
        return {"input_ids": _FakeBatchedTokens([self.token_ids])}


class _FakeTemplateEncoder:
    @staticmethod
    def apply_text_to_template(prompt: str, template: object) -> list[object]:
        return [prompt, template]


class _FakeActionTextEncoder:
    def __init__(self, crop_start: int):
        self.prompt_template_video = {"crop_start": crop_start}
        self.crop_start = crop_start

    def text2tokens(self, prompt: list[str] | str, data_type: str = "video", max_length: int = 1280) -> dict[str, _FakeTensor]:
        value = prompt[0] if isinstance(prompt, list) else prompt
        encoded = [ord(char) + 1 for char in value[:max_length]]
        raw = [100 + index for index in range(self.crop_start)] + encoded
        raw = raw[:self.crop_start + max_length]
        mask = [1] * len(raw)
        raw.extend([0] * (self.crop_start + max_length - len(raw)))
        mask.extend([0] * (self.crop_start + max_length - len(mask)))
        return {"input_ids": _FakeTensor(raw), "attention_mask": _FakeTensor(mask)}


def _pinned_action_pipeline(upstream: Path) -> object:
    methods = conditioning._base_pipeline_methods(upstream / "hymm/pipeline/base.py")
    pipeline = methods()
    pipeline.args = conditioning.SimpleNamespace(
        text_len=1280,
        inference=conditioning.SimpleNamespace(video_fps=24),
    )
    pipeline.t_ratio = 4
    pipeline.text_encoder = _FakeActionTextEncoder(crop_start=108)
    return pipeline


class ActionPromptContractTests(unittest.TestCase):
    def test_accepts_one_local_subject_action_with_upstream_wrapper(self) -> None:
        action = conditioning.validate_t1_prompt(_prompt())

        self.assertEqual(action["action_id"], 1)
        self.assertEqual(action["subject_id"], "<|Subject1|>")
        self.assertEqual(action["start_time_local_seconds"], 0.5)
        self.assertEqual(action["end_time_local_seconds"], 4.5)
        self.assertIn("Resume", action["action_description"])

    def test_rejects_un_normalized_global_action_times(self) -> None:
        with self.assertRaisesRegex(conditioning.ActionConditioningError, "segment-local"):
            conditioning.validate_t1_prompt(_prompt("00:05.500", "00:09.500"))

    def test_rejects_bad_wrapper_or_subject_markup(self) -> None:
        malformed = [
            "<Action2><|Subject1|> turns his head</Action2>.",
            "<Action1>Alex turns his head</Action1>.",
            "<Action1><|Subject1|> turns his head</Action2>.",
            "<|Subject1|> turns his head without an Action wrapper",
        ]
        for description in malformed:
            with self.subTest(description=description):
                with self.assertRaises(conditioning.ActionConditioningError):
                    conditioning.validate_t1_prompt(_prompt(action=description))

    def test_rejects_wrong_origin_or_multiple_segments(self) -> None:
        with self.assertRaisesRegex(conditioning.ActionConditioningError, "T1 starts at 5"):
            conditioning.validate_t1_prompt(_prompt(segment_start="00:00.000"))

        document = json.loads(_prompt())
        document["segments"].append(dict(document["segments"][0]))
        with self.assertRaisesRegex(conditioning.ActionConditioningError, "one T1 segment"):
            conditioning.validate_t1_prompt(json.dumps(document))

    def test_source_offsets_match_native_visible_query_clamp(self) -> None:
        self.assertEqual(conditioning.clamp_source_action_offsets([12, 188], 80), [12, 80])
        with self.assertRaisesRegex(conditioning.ActionConditioningError, "token contract"):
            conditioning.clamp_source_action_offsets([12, 80], 0)

    def test_v2_accepts_two_ordered_subject_bound_actions(self) -> None:
        actions = conditioning.validate_t1_prompt_v2(_v2_prompt())

        self.assertEqual([action["action_id"] for action in actions], [1, 2])
        self.assertEqual([action["subject_id"] for action in actions], ["<|Subject1|>", "<|Subject2|>"])
        self.assertEqual([action["start_time_local_seconds"] for action in actions], [0.5, 3.0])
        self.assertEqual([action["end_time_local_seconds"] for action in actions], [2.5, 4.5])
        actual_prompt_actions = _v2_document()["segments"][0]["contained_actions"]
        actual_prompt_actions[1]["action_description"] = (
            "<Action2><|Subject2|> raises one white glove, taps its own chest once, "
            "then lowers the glove to its side.</Action2>."
        )
        self.assertEqual(
            [action["subject_id"] for action in conditioning.validate_t1_prompt_v2(_v2_prompt(actual_prompt_actions))],
            ["<|Subject1|>", "<|Subject2|>"],
        )

    def test_v2_rejects_reversed_overlapping_wrong_subject_and_truncated_actions(self) -> None:
        valid = _v2_document()["segments"][0]["contained_actions"]
        invalid_cases = [
            [valid[1], valid[0]],
            [valid[0] | {"end_time": "00:03.500"}, valid[1]],
            [valid[0] | {"action_description": "<Action1><|Subject2|> turns his head toward Resume</Action1>."}, valid[1]],
            [valid[0] | {"action_description": "<Action1><|Subject1|></Action1>."}, valid[1]],
            [valid[0], valid[1] | {"action_description": "<Action2><|Subject3|> looks at Alex, raises its white-gloved hand and points at its own chest.</Action2>."}],
            [valid[0], valid[1] | {"action_description": "<Action2><|Subject2|> looks back and raises its white-gloved hand, but the wrapper is truncated"}],
        ]
        for actions in invalid_cases:
            with self.subTest(actions=actions):
                with self.assertRaises(conditioning.ActionConditioningError):
                    conditioning.validate_t1_prompt_v2(_v2_prompt(actions))

    def test_v2_offsets_require_three_strict_visible_boundaries(self) -> None:
        self.assertEqual(conditioning.clamp_source_action_offsets_v2([18, 55, 140], 90), [18, 55, 90])
        for offsets, valid in (([18, 18, 90], 90), ([18, 55, 140], 54), ([10, 30], 90)):
            with self.subTest(offsets=offsets, valid=valid):
                with self.assertRaises(conditioning.ActionConditioningError):
                    conditioning.clamp_source_action_offsets_v2(offsets, valid)

    def test_v2_requires_recomputed_no_truncation_preflight(self) -> None:
        token_ids = [11, 12, 13, 14, 15]
        contract = {
            "contract": {
                "untruncated_preflight": {
                    "method": "apply_chat_template_without_truncation_or_padding",
                    "truncation_required": False,
                    "maximum_raw_length": 7,
                    "raw_length": 5,
                    "token_ids_sha256": conditioning._canonical_json_sha256(token_ids),
                },
            },
            "crop_start": 2,
            "text_length": 5,
            "raw_token_ids": token_ids + [0, 0],
            "raw_attention_mask": [1, 1, 1, 1, 1, 0, 0],
        }
        conditioning._verify_v2_untruncated_prompt(
            "prompt", contract, _FakeNoTruncationTokenizer(token_ids), _FakeTemplateEncoder(), {"template": []},
        )
        contract["contract"]["untruncated_preflight"]["truncation_required"] = True
        with self.assertRaisesRegex(conditioning.ActionConditioningError, "no-truncation"):
            conditioning._verify_v2_untruncated_prompt(
                "prompt", contract, _FakeNoTruncationTokenizer(token_ids), _FakeTemplateEncoder(), {"template": []},
            )

    def test_v2_windows_from_pinned_ast_tile_exactly_once_per_action(self) -> None:
        upstream = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1/upstream-a31")
        pipeline = _pinned_action_pipeline(upstream)
        prompt = _v2_prompt()
        inference_path = upstream / "hymm/inference.py"
        old_parser = conditioning._install_pinned_timestamp_parser(inference_path)
        try:
            source_offsets = pipeline._compute_action_text_offsets(prompt)
            windows = pipeline._compute_action_video_windows(prompt, 0, 30)
        finally:
            if old_parser is None:
                sys.modules.pop("hymm.inference", None)
            else:
                sys.modules["hymm.inference"] = old_parser

        actions = conditioning.validate_t1_prompt_v2(prompt)
        self.assertIsNotNone(source_offsets)
        self.assertEqual(len(source_offsets), 3)
        self.assertEqual(conditioning.clamp_source_action_offsets_v2(source_offsets, len(prompt)), [
            source_offsets[0], source_offsets[1], len(prompt),
        ])
        self.assertEqual(windows, [[0, 3, -1], [3, 15, 0], [15, 18, -1], [18, 27, 1], [27, 30, -1]])
        self.assertEqual(conditioning._verify_windows_v2(windows, actions), windows)
        for invalid in (
            [[0, 3, -1], [3, 15, 0], [15, 27, 1], [27, 30, -1]],
            [[0, 3, -1], [3, 15, 0], [15, 18, -1], [18, 27, 0], [27, 30, -1]],
            [[0, 3, -1], [3, 18, 0], [15, 18, -1], [18, 27, 1], [27, 30, -1]],
        ):
            with self.subTest(windows=invalid):
                with self.assertRaises(conditioning.ActionConditioningError):
                    conditioning._verify_windows_v2(invalid, actions)
        with self.assertRaises(conditioning.ActionConditioningError):
            conditioning._verify_windows_v2(windows, [actions[0], actions[1] | {"start_time_local_seconds": 1.0}])

    @unittest.skipUnless(
        Path("/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1/upstream-a31/.git").exists(),
        "pinned upstream checkout is not mounted",
    )
    def test_source_shaper_normalizes_raw_followup_times_before_serialization(self) -> None:
        upstream = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1/upstream-a31")
        caption = {
            "Style": "Warm hand-drawn animation",
            "subject_list": [{"subject_id": "<|Subject1|>", "display_name": "Alex"}],
            "segments": [{
                "start_time": "00:00.000",
                "end_time": "00:05.000",
                "visual_description_long": "Alex and Resume sit together at the desk.",
                "description_change": None,
                "contained_actions": [],
            }],
        }
        followup = {"segments": [{
            "start_time": "00:05.000",
            "end_time": "00:10.000",
            "visual_description_long": "Dropped after the first segment.",
            "description_change": "Resume stays beside the laptop.",
            "contained_actions": [{
                "start_time": "00:05.500",
                "end_time": "00:09.500",
                "action_description": "<Action1><|Subject1|> turns his head toward Resume</Action1>.",
            }],
        }]}

        t0, t1 = conditioning.shape_reference_prompts(upstream, caption, followup)
        action = conditioning.validate_t1_prompt(t1)

        self.assertEqual(action["start_time_local_seconds"], 0.5)
        self.assertEqual(action["end_time_local_seconds"], 4.5)
        self.assertNotIn("subject_list", json.loads(t1))
        self.assertIn("<|Subject1|>", t1)
        self.assertIn("subject_list", json.loads(t0))

    @unittest.skipUnless(
        Path("/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1/upstream-a31/.git").exists(),
        "pinned upstream checkout is not mounted",
    )
    def test_v2_source_shaper_checks_ordered_t0_roster_and_normalizes_two_actions(self) -> None:
        upstream = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1/upstream-a31")
        caption, followup = _v2_reference_inputs()

        t0, t1 = conditioning.shape_reference_prompts(upstream, caption, followup, profile_schema="v2")
        actions = conditioning.validate_t1_prompt_v2(t1)
        self.assertIn("subject_list", json.loads(t0))
        self.assertNotIn("subject_list", json.loads(t1))
        self.assertEqual([(action["start_time_local_seconds"], action["end_time_local_seconds"]) for action in actions], [
            (0.5, 2.5), (3.0, 4.5),
        ])

        for subjects in (
            caption["subject_list"][:1],
            list(reversed(caption["subject_list"])),
            caption["subject_list"][:1] + [caption["subject_list"][1] | {"subject_attribute": " "}],
        ):
            with self.subTest(subjects=subjects):
                with self.assertRaisesRegex(conditioning.ActionConditioningError, "ordered T0"):
                    conditioning.shape_reference_prompts(
                        upstream,
                        caption | {"subject_list": subjects},
                        followup,
                        profile_schema="v2",
                    )

    @unittest.skipUnless(
        Path("/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1/upstream-a31/.git").exists(),
        "pinned upstream checkout is not mounted",
    )
    def test_shaper_serialization_matches_ast_offsets_and_pretty_json_is_rejected(self) -> None:
        upstream = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1/upstream-a31")
        caption, followup = _v2_reference_inputs()
        _t0, shaped_t1 = conditioning.shape_reference_prompts(
            upstream, caption, followup, profile_schema="v2",
        )
        pipeline = _pinned_action_pipeline(upstream)
        old_parser = conditioning._install_pinned_timestamp_parser(upstream / "hymm/inference.py")
        try:
            shaped_offsets = pipeline._compute_action_text_offsets(shaped_t1)
            shaped_windows = pipeline._compute_action_video_windows(shaped_t1, 0, 30)
            pretty_prompt = json.dumps(json.loads(shaped_t1), ensure_ascii=False, indent=2)
            pretty_offsets = pipeline._compute_action_text_offsets(pretty_prompt)
        finally:
            if old_parser is None:
                sys.modules.pop("hymm.inference", None)
            else:
                sys.modules["hymm.inference"] = old_parser

        self.assertEqual(len(shaped_offsets), 3)
        effective = conditioning.clamp_source_action_offsets_v2(shaped_offsets, len(shaped_t1))
        self.assertLess(effective[0], effective[1])
        self.assertEqual(effective[2], len(shaped_t1))
        self.assertEqual(
            conditioning._verify_windows_v2(shaped_windows, conditioning.validate_t1_prompt_v2(shaped_t1)),
            shaped_windows,
        )
        self.assertEqual(pretty_offsets[:2], [1, 1])
        with self.assertRaisesRegex(conditioning.ActionConditioningError, "partition"):
            conditioning.clamp_source_action_offsets_v2(pretty_offsets, len(pretty_prompt))


if __name__ == "__main__":
    unittest.main()
