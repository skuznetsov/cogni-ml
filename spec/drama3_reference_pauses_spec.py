"""CPU-only contracts for reference-pair waveform pause diagnostics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "compare_reference_pauses.py"
SPEC = importlib.util.spec_from_file_location("drama3_reference_pauses_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pauses = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pauses)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


TAPE_PREFIX_COUNT = 512 * 11
TAPE_PREFIX_BYTES = b"".join(struct.pack("<f", 0.25) for _ in range(TAPE_PREFIX_COUNT))
TAPE_PREFIX_SHA = _sha(TAPE_PREFIX_BYTES)


def _report_pair() -> tuple[dict, dict, dict, dict]:
    target = "Привет! Это тест."
    target_sha = _sha(target.encode("utf-8"))
    transcript = "We asked over twenty different people."
    transcript_sha = _sha(transcript.encode("utf-8"))
    wav_sha = "a" * 64
    code_sha = "b" * 64
    prompt_sha = "c" * 64
    frame = [248100, *range(10)]
    native_base = {
        "backend": "metal_resident",
        "text": target,
        "seed": 1,
        "temperature": 0.8,
        "top_k": 30,
        "top_p": 0.9,
        "max_frames": 512,
        "sampler_inputs": {
            "dtype": "float32-le",
            "uniform_count": TAPE_PREFIX_COUNT,
            "sha256": TAPE_PREFIX_SHA,
            "provenance": "host input tape uploaded once",
        },
        "frames": [frame, frame],
        "stopped_on_eos": True,
        "truncated": False,
        "sample_rate": 44100,
        "samples": 4096,
        "prompt_rows": 5,
        "phase_seconds": {"model_initialize": 1.0, "generation": 2.0},
        "resident_readbacks": {"control": 8, "diagnostic": 0},
        "resident_graph": {"depth": 2, "pending": 0},
    }
    mlx_base = {
        "format": "drama3-mlx-control-v1",
        "upstream": {
            "repository": "/repo/mlx-audio",
            "revision": "abc123",
            "worktree_clean": True,
        },
        "model": {
            "repository": "example/drama-3-mlx-q4",
            "revision": "def456",
            "config_sha256": "1" * 64,
            "index_sha256": "2" * 64,
            "sample_rate": 44100,
            "num_codebooks": 10,
            "codebook_size": 4096,
        },
        "input": {
            "text": target,
            "text_utf8_sha256": target_sha,
            "prompt_rows": 5,
            "prompt_ids": [1, 2, 3, 4, 5],
        },
        "settings": {
            "seed": 1,
            "max_frames": 512,
            "temperature": 0.8,
            "top_k": 30,
            "top_p": 0.9,
            "repetition_aware_retry": {
                "temperature": 1.0,
                "top_k": 30,
                "top_p": 0.95,
                "window_frames": 16,
            },
            "sampler": "shared_uniform_inverse_cdf_shadow",
            "draws_per_iteration": 11,
            "random_tape": {
                "path": "/tmp/tape.json",
                "sha256": "d" * 64,
                "uniform_count": 5632,
                "used_uniform_count": 33,
                "host_uploads": 1,
            },
        },
        "termination": {
            "status": "eos",
            "eos_detected": True,
            "last_sampled_main_token": 248046,
            "generation_iterations": 3,
            "returned_frames": 2,
            "max_frames": 512,
            "frame_cap_reached": False,
        },
        "output": {
            "frame_shape": [2, 11],
            "frames": [frame, frame],
            "samples": 4096,
            "wav_sha256": "e" * 64,
        },
        "timings_seconds": {"generation": 2.1, "decode_codes": 0.1},
        "report_path": "/tmp/mlx.json",
    }
    native_ref = dict(native_base)
    native_ref["prompt_rows"] = 8
    native_ref["prompt_ids"] = None
    native_ref["reference"] = {
        "audio_sha256": wav_sha,
        "text_sha256": transcript_sha,
        "sample_rate": 44100,
        "samples": 172167,
        "frames": 85,
        "text": transcript,
        "conditioning": "resident GPU DAC encoder and prompt",
        "host_code_read_before_prefill": False,
        "audio_row_start": 3,
    }
    mlx_ref = json.loads(json.dumps(mlx_base))
    mlx_ref["input"]["prompt_rows"] = 8
    mlx_ref["input"]["prompt_ids"] = [10, 11, 12, 13, 14, 15, 16, 17]
    mlx_ref["settings"]["random_tape"]["sha256"] = "d" * 64
    mlx_ref["settings"]["random_tape"]["used_uniform_count"] = 33
    mlx_ref["reference_voice"] = {
        "supplied": True,
        "audio": {
            "wav_sha256": wav_sha,
            "pcm_sha256": "f" * 64,
            "sample_rate": 44100,
            "channels": 1,
            "sample_width_bits": 16,
            "samples": 172167,
        },
        "transcript": transcript,
        "transcript_utf8_sha256": transcript_sha,
        "codes": {
            "shape": [10, 85],
            "layout": "codebook-major",
            "sha256": code_sha,
            "feature_lengths": [85],
        },
        "prompt_rows": {
            "path": "/tmp/mlx.json.prompt_rows.f32",
            "shape": [8, 11],
            "dtype": "float32-le",
            "file_sha256": prompt_sha,
            "rows_sha256": prompt_sha,
            "official_builder_byte_equal_after_generation": True,
        },
        "complete_conditioning": True,
    }
    native_no_ref = dict(native_base)
    native_no_ref["prompt_ids"] = [1, 2, 3, 4, 5]
    native_no_ref["reference"] = None
    mlx_no_ref = json.loads(json.dumps(mlx_base))
    mlx_no_ref["reference_voice"] = None
    return native_ref, native_no_ref, mlx_ref, mlx_no_ref


class QuietWindowMergeSpec(unittest.TestCase):
    def test_only_full_quiet_windows_form_internal_pause_runs(self) -> None:
        peaks = [100, 64, 0, 100, 64, 64, 0]
        rms = [20, 10, 0, 20, 10, 10, 0]
        summary = pauses.summarize_quiet_runs(
            peaks, rms, samples=6 * 22_050 + 5_000
        )

        self.assertEqual(summary["full_window_count"], 6)
        self.assertEqual(summary["quiet_full_window_count"], 4)
        self.assertEqual(
            [(run["start_window"], run["end_window_exclusive"], run["duration_seconds"])
             for run in summary["internal_pauses"]],
            [(1, 3, 1.0)],
        )
        self.assertEqual(
            [(run["start_window"], run["end_window_exclusive"])
             for run in summary["trailing_tails"]],
            [(4, 6)],
        )
        self.assertEqual(summary["partial_tail_window"]["full_window"], False)
        self.assertFalse(summary["partial_tail_window"]["quiet_full_window"])

    def test_half_second_and_partial_tail_do_not_combine_to_a_full_pause(self) -> None:
        summary = pauses.summarize_quiet_runs(
            [0, 0], [0, 0], samples=22_050 + 2_000
        )

        self.assertEqual(summary["quiet_full_window_count"], 1)
        self.assertEqual(summary["internal_pauses"], [])
        self.assertEqual(summary["leading_tails"], [])
        self.assertEqual(summary["trailing_tails"], [])
        self.assertTrue(summary["partial_tail_window"]["peak_threshold_met"])

    def test_partial_quiet_tail_alone_is_never_a_full_window_or_pause(self) -> None:
        summary = pauses.summarize_quiet_runs([0], [0], samples=5_000)

        self.assertEqual(summary["full_window_count"], 0)
        self.assertEqual(summary["quiet_full_window_count"], 0)
        self.assertEqual(summary["internal_pauses"], [])
        self.assertEqual(summary["leading_tails"], [])
        self.assertEqual(summary["trailing_tails"], [])
        self.assertEqual(summary["partial_tail_window"]["full_window"], False)
        self.assertEqual(summary["partial_tail_window"]["quiet_full_window"], False)

    def test_leading_trailing_and_whole_waveform_runs_are_separate(self) -> None:
        leading = pauses.summarize_quiet_runs([0, 0, 100, 100], [0] * 4, 4 * 22_050)
        trailing = pauses.summarize_quiet_runs([100, 100, 0, 0], [0] * 4, 4 * 22_050)
        all_quiet = pauses.summarize_quiet_runs([64, 64, 64], [0] * 3, 3 * 22_050)

        self.assertEqual(len(leading["leading_tails"]), 1)
        self.assertEqual(leading["internal_pauses"], [])
        self.assertEqual(len(trailing["trailing_tails"]), 1)
        self.assertEqual(trailing["internal_pauses"], [])
        self.assertEqual(len(all_quiet["leading_tails"]), 1)
        self.assertEqual(len(all_quiet["trailing_tails"]), 1)
        self.assertEqual(all_quiet["internal_pauses"], [])

    def test_metric_shapes_and_sample_geometry_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number"):
            pauses.summarize_quiet_runs([0, 0], [0], 22_050)
        with self.assertRaisesRegex(ValueError, "expected"):
            pauses.summarize_quiet_runs([0], [0], 22_051)
        with self.assertRaisesRegex(ValueError, "finite"):
            pauses.summarize_quiet_runs([float("nan")], [0], 22_050)


class ReferencePauseSchemaSpec(unittest.TestCase):
    def test_reports_validate_shared_text_settings_and_reference_prompt(self) -> None:
        reports = dict(zip(
            ("native-ru-reference", "native-ru-no-reference", "mlx-ru-reference", "mlx-ru-no-reference"),
            _report_pair(),
            strict=True,
        ))
        tape_sha = "d" * 64
        uniforms = [0.25] * TAPE_PREFIX_COUNT
        prefix_bytes = TAPE_PREFIX_BYTES
        proof = {
            "prefix_exact": True,
            "compared_uniforms": 512 * 11,
            "frames": 512,
            "stride": 11,
            "seed": 1,
            "tape_sha256": tape_sha,
            "prefix_f32_sha256": _sha(prefix_bytes),
            "producer": "ResidentGenerator#sampling_draws called through a probe wrapper",
            "production_source": "/repo/src/ml/audio/drama3/generator_resident.cr",
            "production_source_sha256": "e" * 64,
        }
        manifest = {
            "format": "drama3-reference-voice-manifest-v1",
            "input": {
                "wav_sha256": "a" * 64,
                "pcm_sha256": "f" * 64,
                "sample_rate": 44100,
                "channels": 1,
                "sample_width_bits": 16,
                "samples": 172167,
                "transcript_utf8_sha256": _sha(b"We asked over twenty different people."),
                "target_text_utf8_sha256": _sha("Привет! Это тест.".encode("utf-8")),
            },
            "codec": {"sha256": "f" * 64},
            "codes": {
                "shape": [10, 85],
                "layout": "codebook-major",
                "dtype": "int32",
                "sha256": "b" * 64,
            },
            "prompt": {
                "name": "prompt.rows",
                "shape": [8, 11],
                "dtype": "float32-le",
                "sha256": "c" * 64,
                "target_text_utf8_sha256": _sha("Привет! Это тест.".encode("utf-8")),
                "transcript_utf8_sha256": _sha(b"We asked over twenty different people."),
                "reference_codes_sha256": "b" * 64,
            },
        }
        encoder_report = {
            "format": "drama3-native-reference-comparison-v1",
            "waveform_sha256": "a" * 64,
            "samples": 172167,
            "reference_text_sha256": manifest["input"]["transcript_utf8_sha256"],
            "text_sha256": manifest["input"]["target_text_utf8_sha256"],
            "codec_sha256": "f" * 64,
            "code_ids_exact": True,
            "full_gate_passes": False,
            "first_failed_stage": "quantizer.downsample.1.convnext",
            "prompt": {"exact_required": True, "elementwise_exact": True, "passes": True,
                       "shape": [8, 11]},
        }
        tape = {"uniforms": uniforms}

        checked = pauses.validate_report_contract(
            reports,
            tape=tape,
            tape_sha256=tape_sha,
            tape_path=Path("/tmp/tape.json"),
            tape_proof=proof,
            fixture_manifest=manifest,
            native_encoder_report=encoder_report,
        )

        self.assertEqual(checked["target_text_utf8_sha256"], manifest["input"]["target_text_utf8_sha256"])
        self.assertEqual(checked["reference_prompt_sha256"], "c" * 64)
        self.assertEqual(checked["tape_prefix_uniforms"], 512 * 11)
        self.assertFalse(checked["native_encoder_gate"]["full_numeric_stage_gate_passes"])
        self.assertEqual(checked["native_encoder_gate"]["first_failed_stage"],
                         "quantizer.downsample.1.convnext")

    def test_rejects_wrong_format_text_tape_use_and_unverified_prompt(self) -> None:
        cases = ("format", "text", "used", "prompt", "upstream")
        for case in cases:
            with self.subTest(case=case):
                reports = dict(zip(
                    ("native-ru-reference", "native-ru-no-reference", "mlx-ru-reference", "mlx-ru-no-reference"),
                    _report_pair(), strict=True,
                ))
                tape_sha = "d" * 64
                uniforms = [0.25] * (512 * 11)
                proof = {
                    "prefix_exact": True, "compared_uniforms": 512 * 11,
                    "frames": 512, "stride": 11, "seed": 1,
                    "tape_sha256": tape_sha,
                    "prefix_f32_sha256": _sha(b"".join(struct.pack("<f", v) for v in uniforms)),
                    "producer": "ResidentGenerator#sampling_draws producer probe",
                    "production_source": "/repo/generator_resident.cr",
                    "production_source_sha256": "e" * 64,
                }
                manifest = {
                    "format": "drama3-reference-voice-manifest-v1",
                    "input": {
                        "wav_sha256": "a" * 64, "sample_rate": 44100,
                        "pcm_sha256": "f" * 64,
                        "channels": 1, "sample_width_bits": 16, "samples": 172167,
                        "transcript_utf8_sha256": _sha(b"We asked over twenty different people."),
                        "target_text_utf8_sha256": _sha("Привет! Это тест.".encode()),
                    },
                    "codec": {"sha256": "f" * 64},
                    "codes": {"shape": [10, 85], "layout": "codebook-major", "dtype": "int32", "sha256": "b" * 64},
                    "prompt": {"name": "prompt.rows", "shape": [8, 11], "dtype": "float32-le", "sha256": "c" * 64,
                               "target_text_utf8_sha256": _sha("Привет! Это тест.".encode()),
                               "transcript_utf8_sha256": _sha(b"We asked over twenty different people."),
                               "reference_codes_sha256": "b" * 64},
                }
                encoder_report = {
                    "format": "drama3-native-reference-comparison-v1", "waveform_sha256": "a" * 64,
                    "samples": 172167, "reference_text_sha256": manifest["input"]["transcript_utf8_sha256"],
                    "text_sha256": manifest["input"]["target_text_utf8_sha256"],
                    "codec_sha256": "f" * 64, "code_ids_exact": True,
                    "full_gate_passes": False,
                    "first_failed_stage": "quantizer.downsample.1.convnext",
                    "prompt": {"exact_required": True, "elementwise_exact": True, "passes": True,
                               "shape": [8, 11]},
                }
                tape = {"uniforms": uniforms}
                if case == "format":
                    reports["mlx-ru-no-reference"]["format"] = "wrong"
                elif case == "text":
                    reports["native-ru-no-reference"]["text"] = "different"
                elif case == "used":
                    reports["mlx-ru-reference"]["settings"]["random_tape"]["used_uniform_count"] = 32
                elif case == "upstream":
                    reports["mlx-ru-no-reference"]["upstream"]["revision"] = "changed"
                else:
                    encoder_report["prompt"]["elementwise_exact"] = False

                with self.assertRaises(ValueError):
                    pauses.validate_report_contract(
                        reports, tape=tape, tape_sha256=tape_sha,
                        tape_path=Path("/tmp/tape.json"), tape_proof=proof,
                        fixture_manifest=manifest, native_encoder_report=encoder_report,
                    )

    def test_tape_proof_requires_the_full_native_producer_prefix(self) -> None:
        uniforms = [0.5] * (512 * 11)
        good = {
            "prefix_exact": True, "compared_uniforms": 512 * 11,
            "frames": 512, "stride": 11, "seed": 1,
            "tape_sha256": "d" * 64,
            "prefix_f32_sha256": _sha(b"".join(struct.pack("<f", v) for v in uniforms)),
            "producer": "ResidentGenerator#sampling_draws",
            "production_source": "/repo/generator_resident.cr",
            "production_source_sha256": "e" * 64,
        }
        checked = pauses.validate_tape_proof({"uniforms": uniforms}, "d" * 64, good)
        self.assertEqual(checked["compared_uniforms"], 512 * 11)
        for change in (
            {"compared_uniforms": 512 * 11 - 1},
            {"seed": 2},
            {"prefix_exact": False},
            {"producer": "synthetic fixture"},
            {"prefix_f32_sha256": "0" * 64},
        ):
            proof = dict(good, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                pauses.validate_tape_proof({"uniforms": uniforms}, "d" * 64, proof)

    def test_each_native_run_must_bind_sampler_input_count_and_digest(self) -> None:
        for native_name in ("native-ru-reference", "native-ru-no-reference"):
            for field, invalid in (("uniform_count", TAPE_PREFIX_COUNT - 1),
                                   ("sha256", "0" * 64)):
                with self.subTest(native_run=native_name, field=field):
                    reports = dict(zip(
                        ("native-ru-reference", "native-ru-no-reference", "mlx-ru-reference", "mlx-ru-no-reference"),
                        _report_pair(), strict=True,
                    ))
                    reports[native_name]["sampler_inputs"][field] = invalid
                    uniforms = [0.25] * TAPE_PREFIX_COUNT
                    proof = {
                        "prefix_exact": True,
                        "compared_uniforms": TAPE_PREFIX_COUNT,
                        "frames": 512,
                        "stride": 11,
                        "seed": 1,
                        "tape_sha256": "d" * 64,
                        "prefix_f32_sha256": TAPE_PREFIX_SHA,
                        "producer": "ResidentGenerator#sampling_draws called through a probe wrapper",
                        "production_source": "/repo/src/ml/audio/drama3/generator_resident.cr",
                        "production_source_sha256": "e" * 64,
                    }
                    manifest = {
                        "format": "drama3-reference-voice-manifest-v1",
                        "input": {
                            "wav_sha256": "a" * 64,
                            "pcm_sha256": "f" * 64,
                            "sample_rate": 44100,
                            "channels": 1,
                            "sample_width_bits": 16,
                            "samples": 172167,
                            "transcript_utf8_sha256": _sha(b"We asked over twenty different people."),
                            "target_text_utf8_sha256": _sha("Привет! Это тест.".encode("utf-8")),
                        },
                        "codec": {"sha256": "f" * 64},
                        "codes": {"shape": [10, 85], "layout": "codebook-major", "dtype": "int32",
                                  "sha256": "b" * 64},
                        "prompt": {"name": "prompt.rows", "shape": [8, 11], "dtype": "float32-le",
                                   "sha256": "c" * 64,
                                   "target_text_utf8_sha256": _sha("Привет! Это тест.".encode("utf-8")),
                                   "transcript_utf8_sha256": _sha(b"We asked over twenty different people."),
                                   "reference_codes_sha256": "b" * 64},
                    }
                    encoder_report = {
                        "format": "drama3-native-reference-comparison-v1", "waveform_sha256": "a" * 64,
                        "samples": 172167, "reference_text_sha256": manifest["input"]["transcript_utf8_sha256"],
                        "text_sha256": manifest["input"]["target_text_utf8_sha256"],
                        "codec_sha256": "f" * 64, "code_ids_exact": True,
                        "full_gate_passes": False,
                        "first_failed_stage": "quantizer.downsample.1.convnext",
                        "prompt": {"exact_required": True, "elementwise_exact": True, "passes": True,
                                   "shape": [8, 11]},
                    }
                    with self.assertRaisesRegex(ValueError, "sampler_inputs"):
                        pauses.validate_report_contract(
                            reports, tape={"uniforms": uniforms}, tape_sha256="d" * 64,
                            tape_path=Path("/tmp/tape.json"), tape_proof=proof,
                            fixture_manifest=manifest, native_encoder_report=encoder_report,
                        )

    def test_prompt_file_gate_accepts_real_stage_array_and_checks_prompt_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            fixture_prompt = root / "fixture-prompt.f32"
            actual_prompt = root / "actual-prompt.f32"
            prompt_bytes = bytes(2 * 11 * 4)
            fixture_prompt.write_bytes(prompt_bytes)
            actual_prompt.write_bytes(prompt_bytes)
            prompt_sha = _sha(prompt_bytes)
            manifest = {
                "stage_files": "flat row-major little-endian F32; each stage shape is [time,channels]",
                "stages": [{
                    "name": "final.codes",
                    "shape": [3, 10],
                    "row_axis": "time",
                    "channel_axis": "codebook",
                    "dtype": "float32-le",
                    "file": "stage-final-codes.f32",
                    "sha256": "d" * 64,
                }],
                "prompt": {
                    "name": "prompt.rows",
                    "shape": [2, 11],
                    "dtype": "float32-le",
                    "file": fixture_prompt.name,
                    "sha256": prompt_sha,
                },
            }
            checked = {
                "reports": {
                    "mlx-ru-reference": {
                        "raw": {
                            "reference_voice": {
                                "prompt_rows": {"path": str(actual_prompt)},
                            },
                        },
                    },
                },
            }

            result = pauses._prompt_file_check(manifest_path, manifest, checked)

            self.assertTrue(result["byte_equal"])
            self.assertEqual(result["encoder_stage_manifest_entries"], 1)
            manifest["stages"][0]["dtype"] = "int32"
            with self.assertRaisesRegex(ValueError, "float32-le"):
                pauses._prompt_file_check(manifest_path, manifest, checked)

    def test_production_source_hash_is_bound_to_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "generator.cr"
            source.write_bytes(b"def sampling_draws; end\n")
            expected = _sha(source.read_bytes())

            checked = pauses.validate_source_hash(source, expected)
            self.assertEqual(checked["sha256"], expected)

            with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
                pauses.validate_source_hash(source, "0" * 64)


if __name__ == "__main__":
    unittest.main()
