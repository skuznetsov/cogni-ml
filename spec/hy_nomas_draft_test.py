"""Hostile CPU-only tests for the native draft suite launcher."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "examples" / "support"))

import hy_nomas_draft as draft
import hy_nomas_draft_results as results
import hy_nomas_draft_inputs as draft_inputs


SUITE_PATH = ROOT / "examples" / "data" / "hy_nomas_photorealistic_drafts.json"


def load_suite():
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def valid_guard_log(*, returncode=0, final_live=0):
    limits = {
        "event": "limits", "monotonic": 0.0,
        "runtime_floor_pct": 10, "runtime_floor_pct_exact": "10",
        "max_footprint_bytes": 30 * 2**30,
        "max_swap_growth_bytes": 64 * 2**20,
        "timeout_sec": 600, "memory_accounting": "file-cache",
        "headroom_formula": "printed free + file-backed (includes speculative)",
        "startup_rule": "memory_pressure -Q >= 10% and selected headroom >= 10% physical + footprint budget",
        "file_cache_limitation": "file-backed bytes are a reclaimability estimate, not guaranteed clean or immediately free memory",
        "limitation": "sampled limits cannot prevent between-poll spikes or cover deliberate setsid/setpgid escape",
    }
    initial = {
        "event": "sample", "monotonic": 1.0, "pressure_free_pct": 14,
        "free_spec_bytes": 2, "physical_bytes": 100, "swap_used_bytes": 10,
        "group_footprint_bytes": 0, "group_live_processes": 0,
        "speculative_bytes": 1, "file_backed_bytes": 8, "headroom_bytes": 9,
    }
    rows = [
        limits,
        initial,
        {"event": "started", "monotonic": 1.5, "pid": 123, "pgid": 123},
        {"event": "sample", "monotonic": 2.0,
         "pressure_free_pct": 14, "free_spec_bytes": 2,
         "physical_bytes": 100, "swap_used_bytes": 10,
         "group_footprint_bytes": 20, "group_live_processes": 1,
         "speculative_bytes": 1, "file_backed_bytes": 8, "headroom_bytes": 9},
        {"event": "sample", "monotonic": 3.0,
         "pressure_free_pct": 13, "free_spec_bytes": 3,
         "physical_bytes": 100, "swap_used_bytes": 15,
         "group_footprint_bytes": 30, "group_live_processes": final_live,
         "speculative_bytes": 1, "file_backed_bytes": 8, "headroom_bytes": 10},
        {"event": "finished", "monotonic": 3.5, "returncode": returncode},
    ]
    return rows


def valid_native_receipt(scene, prompt, pixels_sha256):
    return {
        "schema": results.RECEIPT_SCHEMA,
        "prompt": prompt,
        "inputs_sha256": "a" * 64,
        "full_upstream_video_parity": False,
        "segments": 2,
        "frames": 240,
        "clean_latent_frames": 60,
        "latent_bank_shape": [1, 32, 180, 9, 16],
        "width": 512,
        "height": 288,
        "fps": 24,
        "steps_per_segment": 4,
        "cache_precision": "F32 exact native K/V boundary; upstream cache is BF16",
        "noise_scope": "authenticated MPS-generated fixed banks; stock upstream CPU RNG value parity is false",
        "pixelumm": {"steps": 1, "window_latents": 10,
                     "overlap_latents": 5, "global_windows": 11},
        "metrics": {
            **{f"{phase}_{metric}": 0 for phase in ("dit", "decoder")
               for metric in ("encode_allocations", "encode_readbacks", "preterminal_waits")},
            "dit_waits": 1, "dit_segments": 22,
            "decoder_waits": 1, "decoder_segments": 44,
        },
        "diagnostics": {"pixels": "pixels.f32le", "pixels_sha256": pixels_sha256},
    }


class GuardReceiptTests(unittest.TestCase):
    def write_log(self, root: Path, rows) -> Path:
        path = root / "guard.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8")
        return path

    def test_limits_event_before_samples_does_not_corrupt_swap_growth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_log(Path(directory), valid_guard_log())
            summary = draft.guard_summary(path)
        self.assertEqual(summary["peak_footprint_gib"], 30 / 2**30)
        self.assertEqual(summary["min_headroom_gib"], 9 / 2**30)
        self.assertEqual(summary["swap_growth_bytes"], 5)

    def test_guard_failure_or_live_process_cannot_be_summarized_as_success(self):
        for rows, message in (
            (valid_guard_log(returncode=76), "certify"),
            (valid_guard_log(final_live=1), "certify"),
            ([{"event": "limits"}, {"event": "finished", "returncode": 0}], "certify"),
            (valid_guard_log(returncode=False), "certify"),
            (valid_guard_log(final_live=False), "certify"),
        ):
            with self.subTest(rows=rows[-1]):
                with tempfile.TemporaryDirectory() as directory:
                    path = self.write_log(Path(directory), rows)
                    with self.assertRaisesRegex(draft.DraftError, message):
                        draft.guard_summary(path)

    def test_malformed_or_boolean_sample_fields_fail_closed(self):
        mutations = []
        missing = valid_guard_log()
        del missing[3]["swap_used_bytes"]
        mutations.append(missing)
        boolean = valid_guard_log()
        boolean[3]["group_live_processes"] = False
        mutations.append(boolean)
        negative = valid_guard_log()
        negative[3]["group_footprint_bytes"] = -1
        mutations.append(negative)
        for rows in mutations:
            with self.subTest(sample=rows[3]):
                with tempfile.TemporaryDirectory() as directory:
                    path = self.write_log(Path(directory), rows)
                    with self.assertRaises(draft.DraftError):
                        draft.guard_summary(path)

    def test_invalid_json_line_is_a_guard_receipt_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guard.jsonl"
            path.write_text('{"event":"finished","returncode":0}\n{broken\n',
                            encoding="utf-8")
            with self.assertRaises(draft.DraftError):
                draft.guard_summary(path)


class PromptAndPreflightTests(unittest.TestCase):
    def test_reuse_skips_every_conditioning_and_noise_producer(self):
        runner = object.__new__(draft.Runner)
        runner.reuse = mock.Mock()
        bundle = {"qwen": Path("/frozen/qwen.json"), "inputs": Path("/frozen/inputs.json"),
                  "continuation": Path("/frozen/timeline.json")}
        runner.reuse.scene.return_value = bundle
        runner.command = mock.Mock(side_effect=AssertionError("input producer was rerun"))
        qwen, base, timeline, stages = runner.prepare_inputs("portrait", Path("/unused/prompt"))
        self.assertEqual((qwen, base, timeline, stages),
                         (bundle["qwen"], bundle["inputs"], bundle["continuation"], []))
        runner.command.assert_not_called()

    def test_prompt_wrapper_hash_is_the_exact_suite_conditioning_text(self):
        scene = load_suite()["scenes"][0]
        prompt = draft.native_prompt(scene)
        results.validate_native_prompt(prompt, scene["prompt"])
        self.assertEqual(draft.sha_bytes(prompt.encode("utf-8")),
                         hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        altered = copy.deepcopy(scene)
        altered["native_prompt"] = prompt.replace(scene["prompt"], scene["prompt"] + " changed")
        with self.assertRaises(results.DraftResultsError):
            draft.native_prompt(altered)

    def test_bad_plan_steps_are_rejected_before_the_native_model_command(self):
        scene = load_suite()["scenes"][0]
        prompt = draft.native_prompt(scene)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for name in ("prompts", "conditioning", "results", "logs", "tmp"):
                (run / name).mkdir()
            runner = object.__new__(draft.Runner)
            runner.run = run
            runner.prompts = {scene["id"]: prompt}
            runner.python = Path("/usr/bin/python3")
            runner.generator = Path("/tmp/native-generator")
            runner.upstream = Path("/tmp/upstream")
            runner.checkpoint = Path("/tmp/checkpoint")
            runner.reuse = None
            runner.model = draft_inputs.model_identity(None, None)
            runner.cinematic_args = []
            runner.env = {}
            runner.document = {"scenes": [{"id": scene["id"]}]}
            runner.refresh = lambda: None
            calls = []

            def fake_command(name, argv, cap_mib, timeout_sec):
                calls.append(name)
                if name.endswith("-plan"):
                    plan = {"event": "sequence_memory_plan", "segments": 2,
                            "steps_per_segment": 8, "frames": 240,
                            "named_budget_bytes": draft.NAMED_BUDGET_MIB * 1024**2,
                            "dit_peak_named_bytes": 1, "decoder_peak_named_bytes": 1,
                            "text_cache_policy": "once_per_scene",
                            "full_upstream_video_parity": False}
                    (run / "logs" / f"{name}.out").write_text(
                        json.dumps(plan) + "\n", encoding="utf-8")
                elif name.endswith("-inputs"):
                    output_path = Path(argv[argv.index("--output") + 1])
                    output_path.write_text(json.dumps({"tensors": {
                        "input.qwen_bf16": {"sha256": "a" * 64}}}), encoding="utf-8")
                elif name.endswith("-continuation"):
                    output_path = Path(argv[argv.index("--output") + 1])
                    tensors = {key: {"sha256": chr(ord("b") + index) * 64}
                               for index, key in enumerate(("input.latent_bank",
                                   "decoder.noise", "cache.eps_bf16"))}
                    output_path.write_text(json.dumps({"prompt": prompt, "tensors": tensors}),
                                           encoding="utf-8")
                elif name.endswith("-qwen"):
                    output_path = Path(argv[argv.index("--output") + 1])
                    output_path.write_text("{}", encoding="utf-8")
                elif name.endswith("-native"):
                    raise AssertionError("native model launched after a bad CPU plan")
                return {"seconds": 1.0, "peak_footprint_gib": 1.0,
                        "min_headroom_gib": 20.0}

            runner.command = fake_command
            with self.assertRaises(draft.DraftError):
                runner.render(scene, {"id": scene["id"]})
        self.assertNotIn(f"{scene['id']}-native", calls)

    def test_final_gallery_binds_the_current_mp4_to_its_recorded_hash(self):
        scene = load_suite()["scenes"][0]
        prompt = draft.native_prompt(scene)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / "results" / scene["id"]
            artifact_dir.mkdir(parents=True)
            video = artifact_dir / "draft.mp4"
            video.write_bytes(b"first valid video bytes")
            (artifact_dir / "pixels.f32le").write_bytes(b"small test pixels")
            pixel_sha = results.sha256_file(artifact_dir / "pixels.f32le")
            receipt = valid_native_receipt(scene, prompt, pixel_sha)
            receipt_path = artifact_dir / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            result = {
                "id": scene["id"], "status": "complete",
                "receipt": f"results/{scene['id']}/receipt.json",
                "video": f"results/{scene['id']}/draft.mp4",
                "native_prompt": prompt,
                "prompt_sha256": draft.sha_bytes(prompt.encode("utf-8")),
                "inputs_sha256": "a" * 64,
                "timings": {"total_scene_seconds": 1.0},
                "memory_peak_gib": 1.0,
                "pixels_sha256": pixel_sha,
                "video_sha256": results.sha256_file(video),
            }
            video.write_bytes(b"replacement video with the same advertised geometry")
            with mock.patch.object(results, "validate_rawpixels",
                                   return_value={"sha256": pixel_sha}), \
                    self.assertRaises(results.DraftResultsError):
                results.validate_completed_result(
                    scene, result, root, full_media_validation=True,
                    probe_function=lambda _path: {
                        "streams": [{"codec_type": "video", "width": 512,
                                     "height": 288, "r_frame_rate": "24/1",
                                     "avg_frame_rate": "24/1", "nb_frames": "240",
                                     "nb_read_frames": "240"}],
                        "format": {"duration": "10.0"}})

    def test_source_and_binary_snapshot_detect_mutation_between_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "generator"
            executable.write_bytes(b"native-binary-v1")
            runner = object.__new__(draft.Runner)
            runner.generator = executable
            runner.binary_sha = results.sha256_file(executable)
            runner.model = draft_inputs.model_identity(None, None)
            runner.source_shas = {
                "examples/hy_nomas_draft.py": results.sha256_file(ROOT / "examples/hy_nomas_draft.py")
            }
            runner.reuse = None
            runner.check_sources()
            executable.write_bytes(b"native-binary-v2")
            with self.assertRaisesRegex(draft.DraftError, "executable changed"):
                runner.check_sources()

    def test_scheduler_stops_at_first_scene_failure_and_preserves_later_pending(self):
        class Harness(draft.Runner):
            def refresh(self):
                return None

            def render(self, scene, row):
                self.visited.append(scene["id"])
                raise draft.DraftError("synthetic first-scene failure")

        runner = object.__new__(Harness)
        runner.scenes = [{"id": "first"}, {"id": "second"}]
        runner.document = {"scenes": [{"id": "first", "status": "pending"},
                                      {"id": "second", "status": "pending"}]}
        runner.visited = []
        with self.assertRaisesRegex(draft.DraftError, "synthetic"):
            runner.execute()
        self.assertEqual(runner.visited, ["first"])
        self.assertEqual(runner.document["status"], "incomplete")
        self.assertEqual(runner.document["scenes"][1]["status"], "pending")

    def test_final_validation_failure_cannot_leave_suite_marked_complete(self):
        class Harness(draft.Runner):
            def refresh(self):
                return None

            def render(self, scene, row):
                row.update(status="complete")

        runner = object.__new__(Harness)
        runner.scenes = [{"id": "first"}]
        runner.document = {"suite_sha256": "a" * 64,
                           "scenes": [{"id": "first", "status": "pending"}]}
        runner.run = Path("/tmp/run")
        runner.suite = {}
        with mock.patch.object(draft.results, "write_gallery",
                               side_effect=results.DraftResultsError("bad final media")):
            with self.assertRaises(results.DraftResultsError):
                runner.execute()
        self.assertNotEqual(runner.document["status"], "complete")


if __name__ == "__main__":
    unittest.main()
