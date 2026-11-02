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


if __name__ == "__main__":
    unittest.main()
