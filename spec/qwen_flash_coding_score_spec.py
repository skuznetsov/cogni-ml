from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_flash_coding_score.py"


class FlashCodingScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("qwen_flash_coding_score", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {SCRIPT}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def _config(self, **overrides: object) -> dict[str, object]:
        config: dict[str, object] = {
            "event": "config",
            "fixture": "chat_prompt_file_no_thinking",
            "control": False,
        }
        config.update(overrides)
        return config

    def _summary(self, **overrides: object) -> dict[str, object]:
        summary: dict[str, object] = {
            "event": "summary",
            "eos_stopping": True,
            "baseline_eos": True,
            "candidate_eos": True,
            "state_passed": False,
            "teacher_passed": False,
            "baseline_text": "module Answer\nend\n",
            "candidate_text": "module Answer\nend\n",
        }
        summary.update(overrides)
        return summary

    def test_extracts_one_config_and_summary_while_ignoring_probe_events(self) -> None:
        config = self._config()
        summary = self._summary()
        log = "\n".join(
            [
                "self_test=PASS",
                json.dumps(config),
                json.dumps({"event": "state", "label": "common_prefix"}),
                json.dumps({"event": "teacher", "step": 0}),
                json.dumps(summary),
            ]
        )

        extracted_config, extracted_summary = self.module.extract_flash_records(log)

        self.assertEqual(config, extracted_config)
        self.assertEqual(summary, extracted_summary)

    def test_rejects_missing_or_duplicate_required_records(self) -> None:
        config = json.dumps(self._config())
        summary = json.dumps(self._summary())

        with self.assertRaisesRegex(ValueError, "exactly one config"):
            self.module.extract_flash_records(config + "\n" + config)
        with self.assertRaisesRegex(ValueError, "exactly one summary"):
            self.module.extract_flash_records(config)
        with self.assertRaisesRegex(ValueError, "exactly one summary"):
            self.module.extract_flash_records(config + "\n" + summary + "\n" + summary)
        with self.assertRaisesRegex(ValueError, "exactly one config"):
            self.module.extract_flash_records(summary)

    def test_rejects_truncated_json_record(self) -> None:
        log = json.dumps(self._config()) + '\n{"event":"summary","baseline_eos":true'

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            self.module.extract_flash_records(log)

    def test_rejects_wrong_fixture_missing_eos_and_non_boolean_control(self) -> None:
        config = self._config()
        summary = self._summary()

        with self.assertRaisesRegex(ValueError, "fixture"):
            self.module.extract_flash_records(
                json.dumps({**config, "fixture": "raw_code_completion_fixed_token_count"})
                + "\n"
                + json.dumps(summary)
            )
        with self.assertRaisesRegex(ValueError, "EOS"):
            self.module.extract_flash_records(
                json.dumps(config) + "\n" + json.dumps({**summary, "candidate_eos": False})
            )
        with self.assertRaisesRegex(ValueError, "control"):
            self.module.extract_flash_records(
                json.dumps({**config, "control": "false"}) + "\n" + json.dumps(summary)
            )

    def test_seeded_wrong_implementation_fails_external_crystal_specs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            (project / "src").mkdir(parents=True)
            (project / "spec").mkdir()
            (project / "src" / "answer.cr").write_text(
                "module Answer\n"
                "  def self.stable_unique(values : Array(Int32)) : Array(Int32)\n"
                "    result = [] of Int32\n"
                "    values.each { |value| result << value unless result.includes?(value) }\n"
                "    result\n"
                "  end\n"
                "end\n",
                encoding="utf-8",
            )
            (project / "spec" / "answer_spec.cr").write_text(
                'require "spec"\nrequire "../src/answer"\n'
                'describe Answer do\n  it "keeps first occurrence order" do\n'
                '    Answer.stable_unique([3, 1, 3, 2, 1]).should eq([3, 1, 2])\n'
                "  end\nend\n",
                encoding="utf-8",
            )
            hidden = root / "hidden_spec.cr"
            hidden.write_text(
                'require "spec"\nrequire "../src/answer"\n'
                'describe "hidden stable_unique" do\n  it "keeps a lower-bound order case" do\n'
                '    Answer.stable_unique([2, 1, 2]).should eq([2, 1])\n'
                "  end\nend\n",
                encoding="utf-8",
            )

            baseline = self.module.run_external_specs(
                project=project,
                hidden_spec=hidden,
                source_path=Path("src/answer.cr"),
                generated_source=(project / "src" / "answer.cr").read_text(encoding="utf-8"),
                output_dir=root / "baseline",
                timeout=30,
            )
            wrong = self.module.run_external_specs(
                project=project,
                hidden_spec=hidden,
                source_path=Path("src/answer.cr"),
                generated_source=(
                    "module Answer\n"
                    "  def self.stable_unique(values : Array(Int32)) : Array(Int32)\n"
                    "    values.sort.uniq\n"
                    "  end\n"
                    "end\n"
                ),
                output_dir=root / "wrong",
                timeout=30,
            )

            self.assertTrue(baseline["external_pass"], baseline["output_tail"])
            self.assertFalse(wrong["external_pass"])

    def test_real_flash_fixture_oracles_reject_three_seeded_wrong_implementations(self) -> None:
        fixture_sources = {
            "stable_unique": (
                'require "set"\n'
                "module Answer\n"
                "  def self.stable_unique(values : Array(Int32)) : Array(Int32)\n"
                "    seen = Set(Int32).new\n"
                "    result = [] of Int32\n"
                "    values.each do |value|\n"
                "      unless seen.includes?(value)\n"
                "        seen << value\n"
                "        result << value\n"
                "      end\n"
                "    end\n"
                "    result\n"
                "  end\n"
                "end\n",
                "module Answer\n"
                "  def self.stable_unique(values : Array(Int32)) : Array(Int32)\n"
                "    [] of Int32\n"
                "  end\n"
                "end\n",
            ),
            "lower_bound": (
                "module Answer\n"
                "  def self.lower_bound(values : Array(Int32), target : Int32) : Int32\n"
                "    lo = 0\n"
                "    hi = values.size\n"
                "    while lo < hi\n"
                "      mid = lo + (hi - lo) // 2\n"
                "      if values[mid] < target\n"
                "        lo = mid + 1\n"
                "      else\n"
                "        hi = mid\n"
                "      end\n"
                "    end\n"
                "    lo\n"
                "  end\n"
                "end\n",
                "module Answer\n"
                "  def self.lower_bound(values : Array(Int32), target : Int32) : Int32\n"
                "    0\n"
                "  end\n"
                "end\n",
            ),
            "merge_ranges": (
                "module Answer\n"
                "  def self.merge_ranges(values : Array(Tuple(Int32, Int32))) : Array(Tuple(Int32, Int32))\n"
                "    merged = [] of Tuple(Int32, Int32)\n"
                "    values.sort_by { |range| range[0] }.each do |range|\n"
                "      if merged.empty? || range[0] > merged[-1][1]\n"
                "        merged << range\n"
                "      else\n"
                "        previous = merged[-1]\n"
                "        merged[-1] = {previous[0], Math.max(previous[1], range[1])}\n"
                "      end\n"
                "    end\n"
                "    merged\n"
                "  end\n"
                "end\n",
                "module Answer\n"
                "  def self.merge_ranges(values : Array(Tuple(Int32, Int32))) : Array(Tuple(Int32, Int32))\n"
                "    [] of Tuple(Int32, Int32)\n"
                "  end\n"
                "end\n",
            ),
        }

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            for name, (baseline_source, wrong_source) in fixture_sources.items():
                fixture = ROOT / "spec" / "fixtures" / "qwen_flash_coding" / name
                hidden = fixture / "check.cr"
                self.assertTrue(hidden.is_file(), hidden)

                baseline = self.module.run_external_specs(
                    project=fixture,
                    hidden_spec=hidden,
                    source_path=Path("src/answer.cr"),
                    generated_source=baseline_source,
                    output_dir=output_root / f"{name}-baseline",
                    timeout=30,
                )
                wrong = self.module.run_external_specs(
                    project=fixture,
                    hidden_spec=hidden,
                    source_path=Path("src/answer.cr"),
                    generated_source=wrong_source,
                    output_dir=output_root / f"{name}-wrong",
                    timeout=30,
                )

                self.assertTrue(baseline["external_pass"], f"{name}: {baseline['output_tail']}")
                self.assertFalse(wrong["external_pass"], name)
                self.assertIn("Failures:", wrong["output_tail"], name)
                self.assertNotIn("undefined method", wrong["output_tail"], name)

    def test_state_failure_is_diagnostic_and_baseline_is_distinguished(self) -> None:
        baseline = {"external_pass": True}
        candidate = {"external_pass": True}

        self.assertEqual(
            "flash_pass", self.module.classify_verdict(baseline=baseline, candidate=candidate)
        )
        self.assertEqual(
            "flash_regression",
            self.module.classify_verdict(
                baseline=baseline, candidate={"external_pass": False}
            ),
        )
        self.assertEqual(
            "invalid_flash_baseline",
            self.module.classify_verdict(
                baseline={"external_pass": False}, candidate={"external_pass": False}
            ),
        )
        self.assertEqual(
            "flash_control_pass",
            self.module.classify_verdict(
                baseline=baseline, candidate=candidate, control=True
            ),
        )


if __name__ == "__main__":
    unittest.main()
