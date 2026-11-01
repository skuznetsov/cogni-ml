"""Offline tests for Hy-Nomas draft receipts, media checks, and gallery HTML."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SUITE_PATH = EXAMPLES / "data" / "hy_nomas_photorealistic_drafts.json"
sys.path.insert(0, str(EXAMPLES / "support"))

import hy_nomas_draft_results as drafts


def load_suite():
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def make_native_prompt(scene):
    wrapper = {
        "Style": "Photorealistic real-world footage",
        "perspective_view": "third-person observer",
        "subject_list": [],
        "segments": [{
            "start_time": "00:00.000",
            "end_time": "00:10.000",
            "visual_description_long": scene["prompt"],
            "description_change": None,
            "contained_actions": [],
        }],
    }
    return json.dumps(wrapper, ensure_ascii=False, indent=2)


def make_receipt(scene, native_prompt, inputs_sha="a" * 64):
    return {
        "schema": drafts.RECEIPT_SCHEMA,
        "prompt": native_prompt,
        "inputs_sha256": inputs_sha,
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
        "pixelumm": {
            "steps": 1,
            "window_latents": 10,
            "overlap_latents": 5,
            "global_windows": 11,
        },
        "metrics": {
            "dit_encode_allocations": 0,
            "dit_encode_readbacks": 0,
            "dit_preterminal_waits": 0,
            "dit_waits": 1,
            "dit_segments": 22,
            "decoder_encode_allocations": 0,
            "decoder_encode_readbacks": 0,
            "decoder_preterminal_waits": 0,
            "decoder_waits": 1,
            "decoder_segments": 44,
        },
        "diagnostics": {
            "pixels": "/artifacts/results/scene/pixels.f32le",
            "pixels_sha256": "b" * 64,
        },
    }


def make_result(scene, native_prompt, receipt_rel="results/portrait/receipt.json"):
    prompt_sha = hashlib.sha256(native_prompt.encode("utf-8")).hexdigest()
    return {
        "id": scene["id"],
        "status": "completed",
        "video": "results/portrait/draft.mp4",
        "receipt": receipt_rel,
        "native_prompt": native_prompt,
        "prompt_sha256": prompt_sha,
        "inputs_sha256": "a" * 64,
        "pixels_sha256": "b" * 64,
        "video_sha256": "c" * 64,
        "timings": {"dit_seconds": 12.5, "decoder_seconds": 3.25},
        "memory_peak_gib": 28.7,
    }


def make_model(variant="original"):
    if variant == "original":
        return {
            "dit_variant": "original",
            "dit_weight_index_sha256": "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0",
            "cinematic_lineage": None,
            "cinematic_lineage_sha256": None,
        }
    return {
        "dit_variant": "cinematic-active-blocks",
        "dit_weight_index_sha256": "c" * 64,
        "cinematic_lineage": "/opt/mldata_ssd/cogni-ml/cinematic/cache.json",
        "cinematic_lineage_sha256": "d" * 64,
    }


def valid_probe():
    return {
        "streams": [{
            "codec_type": "video",
            "width": 512,
            "height": 288,
            "r_frame_rate": "24/1",
            "avg_frame_rate": "24/1",
            "nb_frames": "240",
            "nb_read_frames": "240",
        }],
        "format": {"duration": "10.000000"},
    }


class NativeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_suite()
        self.scene = self.suite["scenes"][0]
        self.prompt = make_native_prompt(self.scene)
        self.receipt = make_receipt(self.scene, self.prompt)

    def test_checked_in_suite_and_native_receipt_contract(self):
        self.assertEqual(len(drafts.validate_suite(self.suite)), 6)
        result = drafts.validate_metadata(self.receipt, self.prompt, "a" * 64)
        self.assertEqual(result["inputs_sha256"], "a" * 64)
        self.assertEqual(result["pixels_sha256"], "b" * 64)

    def test_model_identity_is_exact_structural_and_variant_aware(self):
        original = make_model()
        cinematic = make_model("cinematic")
        self.assertEqual(drafts.validate_model_identity(original, original), original)
        self.assertEqual(drafts.validate_model_identity(cinematic, cinematic), cinematic)
        with self.assertRaisesRegex(drafts.DraftResultsError, "model identity mismatch"):
            drafts.validate_model_identity(original, cinematic)

        invalid_models = []
        wrong_base_index = dict(original, dit_weight_index_sha256="e" * 64)
        invalid_models.append(wrong_base_index)
        missing_base_lineage = dict(original, cinematic_lineage_sha256="f" * 64)
        invalid_models.append(missing_base_lineage)
        relative_lineage = dict(cinematic, cinematic_lineage="cache.json")
        invalid_models.append(relative_lineage)
        bad_lineage_sha = dict(cinematic, cinematic_lineage_sha256="invalid")
        invalid_models.append(bad_lineage_sha)
        extra_field = dict(original, extra="not part of identity")
        invalid_models.append(extra_field)
        for model in invalid_models:
            with self.subTest(model=model):
                with self.assertRaises(drafts.DraftResultsError):
                    drafts.validate_model_identity(model, model)

    def test_expected_model_requires_matching_receipt_identity_but_legacy_call_is_unchanged(self):
        model = make_model("cinematic")
        self.assertEqual(
            drafts.validate_metadata(self.receipt, self.prompt, "a" * 64)["pixels_sha256"],
            "b" * 64)

        with self.assertRaisesRegex(drafts.DraftResultsError, "model identity"):
            drafts.validate_metadata(self.receipt, self.prompt, "a" * 64,
                                     expected_model=model)

        self.receipt.update(model)
        self.assertEqual(
            drafts.validate_metadata(self.receipt, self.prompt, "a" * 64,
                                     expected_model=model)["pixels_sha256"],
            "b" * 64)
        wrong_receipt = dict(self.receipt, dit_weight_index_sha256="e" * 64)
        with self.assertRaisesRegex(drafts.DraftResultsError, "model identity"):
            drafts.validate_metadata(wrong_receipt, self.prompt, "a" * 64,
                                     expected_model=model)

    def test_wrong_native_prompt_and_wrong_input_sha_are_rejected(self):
        wrong_prompt = self.prompt.replace("afternoon", "midnight")
        with self.assertRaisesRegex(drafts.DraftResultsError, "prompt"):
            drafts.validate_metadata(self.receipt, wrong_prompt, "a" * 64)
        with self.assertRaisesRegex(drafts.DraftResultsError, "inputs_sha256"):
            drafts.validate_metadata(self.receipt, self.prompt, "c" * 64)

    def test_visual_description_and_time_window_are_bound_to_suite(self):
        changed_scene = copy.deepcopy(self.scene)
        changed_scene["prompt"] += " Add a blue bird."
        with self.assertRaisesRegex(drafts.DraftResultsError, "visual description"):
            drafts.validate_native_prompt(self.prompt, changed_scene["prompt"])
        wrong_time = self.prompt.replace("00:10.000", "00:09.000")
        with self.assertRaisesRegex(drafts.DraftResultsError, "ten seconds"):
            drafts.validate_native_prompt(wrong_time, self.scene["prompt"])
        duplicate_style = self.prompt.replace(
            '"Style": "Photorealistic real-world footage",',
            '"Style": "Photorealistic real-world footage", "Style": "other",',
            1)
        with self.assertRaisesRegex(drafts.DraftResultsError, "duplicate JSON key"):
            drafts.validate_native_prompt(duplicate_style, self.scene["prompt"])

    def test_rejects_wrong_geometry_windows_and_noise_contract(self):
        wrong_frames = dict(self.receipt, frames=239)
        with self.assertRaisesRegex(drafts.DraftResultsError, "frames"):
            drafts.validate_metadata(wrong_frames, self.prompt, "a" * 64)
        wrong_windows = copy.deepcopy(self.receipt)
        wrong_windows["pixelumm"]["global_windows"] = 10
        with self.assertRaisesRegex(drafts.DraftResultsError, "global_windows"):
            drafts.validate_metadata(wrong_windows, self.prompt, "a" * 64)
        wrong_noise = dict(self.receipt, noise_scope="randomized noise per prompt")
        with self.assertRaisesRegex(drafts.DraftResultsError, "noise scope"):
            drafts.validate_metadata(wrong_noise, self.prompt, "a" * 64)

    def test_requires_zero_encode_costs_and_one_terminal_wait_per_phase(self):
        for key, bad in (("dit_encode_allocations", 1),
                         ("decoder_encode_readbacks", 1),
                         ("dit_preterminal_waits", 1),
                         ("decoder_waits", 2)):
            with self.subTest(key=key):
                wrong = copy.deepcopy(self.receipt)
                wrong["metrics"][key] = bad
                with self.assertRaises(drafts.DraftResultsError):
                    drafts.validate_metadata(wrong, self.prompt, "a" * 64)

    def test_suite_rejects_changed_noise_and_geometry_declarations(self):
        bad_noise = copy.deepcopy(self.suite)
        bad_noise["settings"]["noise"] = "a new random bank per prompt"
        with self.assertRaisesRegex(drafts.DraftResultsError, "setting 'noise'"):
            drafts.validate_suite(bad_noise)
        bad_geometry = copy.deepcopy(self.suite)
        bad_geometry["settings"]["segments"] = 3
        with self.assertRaisesRegex(drafts.DraftResultsError, "setting 'segments'"):
            drafts.validate_suite(bad_geometry)


class RawPixelsAndProbeTests(unittest.TestCase):
    def test_rawpixel_hash_and_injectable_small_size(self):
        payload = b"small synthetic tensor"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixels.f32le"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            result = drafts.validate_rawpixels(path, digest, expected_bytes=len(payload))
            self.assertEqual(result["bytes"], len(payload))
            with self.assertRaisesRegex(drafts.DraftResultsError, "SHA-256 mismatch"):
                drafts.validate_rawpixels(path, "0" * 64, expected_bytes=len(payload))
            with self.assertRaisesRegex(drafts.DraftResultsError, "byte count"):
                drafts.validate_rawpixels(path, digest, expected_bytes=len(payload) + 1)

    def test_production_raw_pixel_size_is_exact_without_a_large_fixture(self):
        drafts.validate_rawpixel_size(drafts.RAW_PIXELS_BYTES)
        with self.assertRaisesRegex(drafts.DraftResultsError, "byte count"):
            drafts.validate_rawpixel_size(drafts.RAW_PIXELS_BYTES - 4)

    def test_ffprobe_requires_actual_decoded_frame_count_geometry_and_duration(self):
        self.assertEqual(drafts.validate_ffprobe(valid_probe())["frames"], 240)
        mutations = (
            ("nb_read_frames", "239", "decoded frame count"),
            ("width", 640, "width"),
            ("avg_frame_rate", "25/1", "24 fps"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key):
                probe = copy.deepcopy(valid_probe())
                probe["streams"][0][key] = value
                with self.assertRaisesRegex(drafts.DraftResultsError, message):
                    drafts.validate_ffprobe(probe)
        wrong_duration = valid_probe()
        wrong_duration["format"]["duration"] = "9.96"
        with self.assertRaisesRegex(drafts.DraftResultsError, "10 seconds"):
            drafts.validate_ffprobe(wrong_duration)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg/ffprobe unavailable")
    def test_ffprobe_reads_a_real_local_mp4_and_counts_decoded_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "synthetic.mp4"
            subprocess.run(
                [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                 "color=c=black:s=512x288:r=24:d=10", "-frames:v", "240",
                 "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "37",
                 str(video)],
                check=True, capture_output=True, timeout=60)
            self.assertEqual(drafts.validate_ffprobe(drafts.ffprobe_mp4(video))["frames"], 240)


class GalleryTests(unittest.TestCase):
    def _completed_fixture(self, root):
        suite = load_suite()
        scene = suite["scenes"][0]
        native_prompt = make_native_prompt(scene)
        artifact_dir = root / "results" / scene["id"]
        artifact_dir.mkdir(parents=True)
        video = artifact_dir / "draft.mp4"
        video.write_bytes(b"synthetic mp4 bytes")
        receipt = make_receipt(scene, native_prompt)
        (artifact_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return suite, scene, make_result(scene, native_prompt), video

    def test_partial_gallery_shows_pending_failed_and_escaped_local_output(self):
        suite = load_suite()
        suite_bytes = SUITE_PATH.read_bytes()
        suite_sha = hashlib.sha256(suite_bytes).hexdigest()
        first = suite["scenes"][0]
        native_prompt = make_native_prompt(first)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results" / first["id"]
            result_dir.mkdir(parents=True)
            video = result_dir / "draft.mp4"
            video.write_bytes(b"fake-mp4-for-gallery-only")
            receipt = make_receipt(first, native_prompt)
            (result_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            failed_scene = suite["scenes"][1]
            rows = {
                "schema": drafts.RESULTS_SCHEMA,
                "suite_sha256": suite_sha,
                "scenes": [
                    make_result(first, native_prompt),
                    {"id": failed_scene["id"], "status": "failed",
                     "detail": "<script>alert('x')</script>"},
                ],
            }
            output = root / "index.html"
            report = drafts.render_gallery(suite, rows, output, root,
                                          suite_sha256=suite_sha)
            page = output.read_text(encoding="utf-8")
            self.assertEqual(report["scenes"][0]["status"], "completed")
            self.assertNotIn("model", report)
            self.assertIn('src="results/portrait/draft.mp4"', page)
            self.assertIn("fixed same-noise banks", page)
            self.assertIn("does not claim upstream end-to-end video parity", page)
            self.assertIn("Pending", page)
            self.assertIn("Failed", page)
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>alert", page)
            self.assertNotIn("https://", page)

    def test_gallery_binds_declared_model_and_shows_effective_cinematic_variant(self):
        suite = load_suite()
        scene = suite["scenes"][0]
        native_prompt = make_native_prompt(scene)
        suite_sha = hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()
        model = make_model("cinematic")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results" / scene["id"]
            result_dir.mkdir(parents=True)
            (result_dir / "draft.mp4").write_bytes(b"synthetic mp4 bytes")
            receipt = make_receipt(scene, native_prompt)
            receipt.update(model)
            (result_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            result = make_result(scene, native_prompt)
            result["receipt"] = f"results/{scene['id']}/receipt.json"
            document = {"schema": drafts.RESULTS_SCHEMA,
                        "suite_sha256": suite_sha, "model": model,
                        "scenes": [result]}

            output = root / "index.html"
            report = drafts.render_gallery(suite, document, output, root,
                                           suite_sha256=suite_sha)
            page = output.read_text(encoding="utf-8")
            self.assertEqual(report["scenes"][0]["status"], "completed")
            self.assertEqual(report["model"], model)
            self.assertIn("Cinematic active blocks BF16", page)
            self.assertNotIn("Weights: original BF16", page)

            receipt["cinematic_lineage_sha256"] = "e" * 64
            (result_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(drafts.DraftResultsError, "model identity"):
                drafts.render_gallery(suite, document, output, root,
                                      suite_sha256=suite_sha)

            malformed_document = dict(document, scenes=[])
            malformed_document["model"] = dict(model, extra="not an identity field")
            with self.assertRaisesRegex(drafts.DraftResultsError, "exactly the four"):
                drafts.render_gallery(suite, malformed_document, output, root,
                                      suite_sha256=suite_sha)

    def test_rejects_suite_digest_mismatch_and_unsafe_video_paths(self):
        suite = load_suite()
        first = suite["scenes"][0]
        prompt = make_native_prompt(first)
        suite_sha = hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results" / first["id"]
            result_dir.mkdir(parents=True)
            (result_dir / "draft.mp4").write_bytes(b"fake")
            (result_dir / "receipt.json").write_text(
                json.dumps(make_receipt(first, prompt)), encoding="utf-8")
            row = make_result(first, prompt)
            document = {"schema": drafts.RESULTS_SCHEMA,
                        "suite_sha256": suite_sha, "scenes": [row]}
            with self.assertRaisesRegex(drafts.DraftResultsError, "suite_sha256"):
                drafts.render_gallery(suite, document, root / "index.html", root,
                                      suite_sha256="0" * 64)
            row["video"] = "../outside.mp4"
            with self.assertRaisesRegex(drafts.DraftResultsError, "safe relative"):
                drafts.render_gallery(suite, document, root / "index.html", root,
                                      suite_sha256=suite_sha)

    def test_completed_row_requires_pixel_alias_and_syntactically_valid_video_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, scene, result, _video = self._completed_fixture(root)
            result["pixels_sha256"] = "d" * 64
            with self.assertRaisesRegex(drafts.DraftResultsError, "pixels_sha256 differs"):
                drafts.validate_completed_result(scene, result, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, scene, result, _video = self._completed_fixture(root)
            result.pop("pixels_sha256")
            with self.assertRaisesRegex(drafts.DraftResultsError, "result.pixels_sha256"):
                drafts.validate_completed_result(scene, result, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, scene, result, _video = self._completed_fixture(root)
            result.pop("video_sha256")
            with self.assertRaisesRegex(drafts.DraftResultsError, "result.video_sha256"):
                drafts.validate_completed_result(scene, result, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, scene, result, _video = self._completed_fixture(root)
            result["video_sha256"] = "not-a-hash"
            with self.assertRaisesRegex(drafts.DraftResultsError, "lowercase SHA-256"):
                drafts.validate_completed_result(scene, result, root)

    def test_full_media_validation_binds_current_video_and_raw_bytes_to_recorded_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, scene, result, video = self._completed_fixture(root)
            pixel_path = video.parent / "pixels.f32le"
            pixel_path.write_bytes(b"small raw pixels")
            pixel_sha = hashlib.sha256(pixel_path.read_bytes()).hexdigest()
            receipt_path = video.parent / "receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["diagnostics"]["pixels"] = str(pixel_path)
            receipt["diagnostics"]["pixels_sha256"] = pixel_sha
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            result["pixels_sha256"] = pixel_sha
            result["video_sha256"] = drafts.sha256_file(video)

            with mock.patch.object(
                    drafts, "validate_rawpixels",
                    return_value={"sha256": pixel_sha, "bytes": drafts.RAW_PIXELS_BYTES}):
                report = drafts.validate_completed_result(
                    scene, result, root, full_media_validation=True,
                    probe_function=lambda _path: valid_probe())
            self.assertEqual(report["video_sha256"], result["video_sha256"])

            video.write_bytes(b"substituted mp4 with plausible metadata")
            with mock.patch.object(
                    drafts, "validate_rawpixels",
                    return_value={"sha256": pixel_sha, "bytes": drafts.RAW_PIXELS_BYTES}), \
                    self.assertRaisesRegex(drafts.DraftResultsError, "current MP4 SHA-256"):
                drafts.validate_completed_result(
                    scene, result, root, full_media_validation=True,
                    probe_function=lambda _path: valid_probe())

    def test_final_validation_keeps_failure_visible_but_does_not_pass_partial_suite(self):
        suite = load_suite()
        suite_sha = hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()
        document = {
            "schema": drafts.RESULTS_SCHEMA,
            "suite_sha256": suite_sha,
            "scenes": [{"id": suite["scenes"][0]["id"], "status": "failed",
                        "detail": "decoder validation rejected the first take"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = drafts.render_gallery(
                suite, document, root / "index.html", root,
                full_media_validation=True, suite_sha256=suite_sha)
            self.assertIs(report["passed"], False)
            page = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn("decoder validation rejected", page)

    def test_final_cli_prints_a_deliberately_json_serializable_report(self):
        suite_bytes = SUITE_PATH.read_bytes()
        suite_sha = hashlib.sha256(suite_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite_path = root / "suite.json"
            suite_path.write_bytes(suite_bytes)
            scenes = []
            for scene in load_suite()["scenes"]:
                native_prompt = make_native_prompt(scene)
                artifact_dir = root / "results" / scene["id"]
                artifact_dir.mkdir(parents=True)
                video = artifact_dir / "draft.mp4"
                video.write_bytes(f"fake mp4 {scene['id']}".encode())
                pixels = artifact_dir / "pixels.f32le"
                pixels.write_bytes(f"fake pixels {scene['id']}".encode())
                pixels_sha = hashlib.sha256(pixels.read_bytes()).hexdigest()
                receipt = make_receipt(scene, native_prompt)
                receipt["diagnostics"].update(pixels=str(pixels), pixels_sha256=pixels_sha)
                receipt_rel = f"results/{scene['id']}/receipt.json"
                (artifact_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                row = make_result(scene, native_prompt, receipt_rel)
                row.update(video=f"results/{scene['id']}/draft.mp4",
                           pixels_sha256=pixels_sha,
                           video_sha256=drafts.sha256_file(video))
                scenes.append(row)

            results_path = root / "results.json"
            results_path.write_text(json.dumps({
                "schema": drafts.RESULTS_SCHEMA,
                "suite_sha256": suite_sha,
                "scenes": scenes,
            }), encoding="utf-8")
            output = io.StringIO()
            def small_pixel_check(path, expected_sha):
                return {"path": str(path), "bytes": drafts.RAW_PIXELS_BYTES,
                        "sha256": expected_sha}

            with mock.patch.object(drafts, "validate_rawpixels", side_effect=small_pixel_check), \
                    mock.patch.object(drafts, "ffprobe_mp4", return_value=valid_probe()), \
                    contextlib.redirect_stdout(output):
                exit_code = drafts.main([
                    "--suite", str(suite_path), "--results", str(results_path),
                    "--output", str(root / "index.html"), "--validate-final",
                ])
            self.assertEqual(exit_code, 0)
            report = json.loads(output.getvalue())
            self.assertIs(report["passed"], True)
            self.assertEqual(len(report["scenes"]), 6)
            self.assertTrue(all(isinstance(row["video_path"], str)
                                and isinstance(row["receipt_path"], str)
                                for row in report["scenes"]))


if __name__ == "__main__":
    unittest.main()
