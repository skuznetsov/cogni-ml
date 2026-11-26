"""CPU-only admission falsifiers for the reference-continuation control."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_reference_continuation_control as control


class ContinuationAdmissionTests(unittest.TestCase):
    def test_mask_admission_uses_values_and_prefix_geometry(self):
        mask = [1] * 369 + [0] * (control.QWEN_ROWS - 369)
        self.assertEqual(control._validate_mask(mask, "action"), 369)
        with self.assertRaisesRegex(control.ControlError, "valid prefix"):
            control._validate_mask([1] * 368 + [0, 1] + [0] * (control.QWEN_ROWS - 370), "action")
        with self.assertRaisesRegex(control.ControlError, "integer values"):
            control._validate_mask([True] + [1] * 368 + [0] * (control.QWEN_ROWS - 369), "action")
        with self.assertRaisesRegex(control.ControlError, "1280"):
            control._validate_mask([1, 0], "caption")

    def test_tensor_record_admission_rejects_geometry_and_byte_extent_drift(self):
        record = {
            "dtype": "F32",
            "shape": [1, 32, 180, 9, 16],
            "nbytes": 1 * 32 * 180 * 9 * 16 * 4,
            "offset": 0,
            "sha256": "a" * 64,
        }
        self.assertEqual(
            control._validate_tensor_record(record, "F32", [1, 32, 180, 9, 16]),
            record["nbytes"],
        )
        wrong_shape = {**record, "shape": [1, 32, 30, 9, 16]}
        with self.assertRaisesRegex(control.ControlError, "geometry"):
            control._validate_tensor_record(wrong_shape, "F32", [1, 32, 180, 9, 16])
        wrong_size = {**record, "nbytes": record["nbytes"] - 4}
        with self.assertRaisesRegex(control.ControlError, "extent"):
            control._validate_tensor_record(wrong_size, "F32", [1, 32, 180, 9, 16])

    def test_file_authentication_checks_bytes_and_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            # macOS may expose the temporary directory through a /var alias.
            # The production adapter intentionally rejects non-canonical paths.
            path = Path(temporary).resolve() / "latent-bank.f32le"
            payload = bytes(range(256)) * 16
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(control._authenticate_file(path, expected, "latent"), expected)
            with self.assertRaisesRegex(control.ControlError, "digest"):
                control._authenticate_file(path, "0" * 64, "latent")
            path.write_bytes(payload[:-4])
            with self.assertRaisesRegex(control.ControlError, "extent"):
                control._authenticate_file(path, expected, "latent", expected_bytes=len(payload))

    def test_source_digest_admission_detects_local_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "hymm" / "models" / "dit" / "nomas_cpt.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"pinned source bytes\n")
            expected = {"hymm/models/dit/nomas_cpt.py": hashlib.sha256(source.read_bytes()).hexdigest()}
            self.assertEqual(control._verify_source_files(root, expected), expected)
            source.write_bytes(b"locally drifted source bytes\n")
            with self.assertRaisesRegex(control.ControlError, "source digest"):
                control._verify_source_files(root, expected)

    def test_t0_selection_is_exact_first_clean_segment(self):
        self.assertEqual(control._validate_t0_selection([1, 32, 180, 9, 16], 0, 30), [0, 30])
        with self.assertRaisesRegex(control.ControlError, "selection"):
            control._validate_t0_selection([1, 32, 180, 9, 16], 30, 30)
        with self.assertRaisesRegex(control.ControlError, "geometry"):
            control._validate_t0_selection([1, 32, 60, 9, 16], 0, 30)

    def test_schedule_writes_caption_history_then_queries_with_action_text(self):
        plan = control._forward_plan(421, 369)
        control._validate_forward_plan(plan, 421, 369)
        self.assertEqual([item["kind"] for item in plan], ["writer"] + ["denoise"] * 4)
        self.assertEqual(plan[0]["condition"], "t0_caption")
        self.assertEqual(plan[0]["valid_text_tokens"], 421)
        self.assertTrue(plan[0]["cache_txt"])
        self.assertEqual(
            control._block_cache_contract(plan[0]),
            {"cache_valid_len": 0, "update_cache": True, "cache_txt": True},
        )
        self.assertTrue(all(item["condition"] == "t1_action" for item in plan[1:]))
        self.assertTrue(all(item["valid_text_tokens"] == 369 for item in plan[1:]))
        self.assertTrue(all(item["cache_valid_len"] == control.CACHE_CAPACITY for item in plan[1:]))
        self.assertTrue(all(item["cache_txt"] for item in plan[1:]))
        self.assertTrue(all(not item["update_cache"] for item in plan[1:]))
        self.assertTrue(all(
            control._block_cache_contract(item)
            == {"cache_valid_len": control.CACHE_CAPACITY, "update_cache": False, "cache_txt": True}
            for item in plan[1:]
        ))
        self.assertEqual(sum(item["update_cache"] for item in plan), 1)
        altered = [dict(item) for item in plan]
        altered[1]["cache_txt"] = False
        with self.assertRaisesRegex(control.ControlError, "schedule"):
            control._validate_forward_plan(altered, 421, 369)


if __name__ == "__main__":
    unittest.main()
