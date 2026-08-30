#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_qbit_coding_session_score.py"


class CodingSessionScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("qwen_qbit_coding_session_score", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {SCRIPT}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_extracts_crystal_fence_without_surrounding_prose(self) -> None:
        text = "Explanation.\n```crystal\nmodule Fixed\nend\n```\nDone."

        self.assertEqual("module Fixed\nend\n", self.module.extract_crystal_source(text))

    def test_requires_one_resident_quality_record(self) -> None:
        record = {
            "schema": "qwen-qbit-quality-v1",
            "execution_mode": "resident_gpu",
            "policy": "resident[p4]",
            "full_attention_layers": 16,
            "resident_layers": 16,
            "resident_f32_owner_layers": [],
            "resident_cache_consistent": True,
        }
        log = "noise\nQBIT_QUALITY_JSON=" + json.dumps(record) + "\n"

        self.assertEqual(record, self.module.extract_quality_record(log))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.module.extract_quality_record(log + "QBIT_QUALITY_JSON=" + json.dumps(record))

    def test_external_specs_distinguish_valid_and_invalid_generated_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            (project / "src").mkdir(parents=True)
            (project / "spec").mkdir()
            (project / "src" / "answer.cr").write_text(
                "module Answer\n  def self.value : Int32\n    1\n  end\nend\n",
                encoding="utf-8",
            )
            (project / "spec" / "answer_spec.cr").write_text(
                'require "spec"\nrequire "../src/answer"\n'
                'describe Answer do\n  it "returns two" do\n'
                '    Answer.value.should eq(2)\n  end\nend\n',
                encoding="utf-8",
            )
            hidden = root / "hidden_spec.cr"
            hidden.write_text(
                'require "spec"\nrequire "../src/answer"\n'
                'describe "hidden" do\n  it "rejects three" do\n'
                '    Answer.value.should_not eq(3)\n  end\nend\n',
                encoding="utf-8",
            )

            valid = self.module.run_external_specs(
                project=project,
                hidden_spec=hidden,
                source_path=Path("src/answer.cr"),
                generated_source="module Answer\n  def self.value : Int32\n    2\n  end\nend\n",
                output_dir=root / "valid",
                timeout=30,
            )
            invalid = self.module.run_external_specs(
                project=project,
                hidden_spec=hidden,
                source_path=Path("src/answer.cr"),
                generated_source="module Answer\n  def self.value : Int32\n    3\n  end\nend\n",
                output_dir=root / "invalid",
                timeout=30,
            )

            self.assertTrue(valid["external_pass"], valid["output_tail"])
            self.assertFalse(invalid["external_pass"])
            self.assertEqual(0, valid["exit_code"])
            self.assertNotEqual(0, invalid["exit_code"])

    def test_qbit_regression_requires_a_valid_exact_baseline(self) -> None:
        verdict = self.module.classify_verdict(
            exact={"external_pass": True},
            candidate={"external_pass": False},
        )
        invalid_baseline = self.module.classify_verdict(
            exact={"external_pass": False},
            candidate={"external_pass": False},
        )

        self.assertEqual("qbit_regression", verdict)
        self.assertEqual("invalid_exact_baseline", invalid_baseline)


if __name__ == "__main__":
    unittest.main()
