"""CPU-only mutations for the bounded Hy-Nomas segment-edit video comparator."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_cache_qbit_video_compare as video
import hy_nomas_segment_edit_resume_reference as segment_edit
import hy_nomas_segment_edit_video_compare as comparison


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class SegmentEditVideoComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="hy-nomas-segment-edit-video-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index_path = self.root / "matrix-index.json"
        self.index_path.write_text("{}\n", encoding="utf-8")

        self.base_inputs = self.root / "base-inputs.json"
        self.base_inputs.write_bytes(b'{"schema":"test-base"}\n')
        self.base_inputs_sha = digest(self.base_inputs.read_bytes())
        self.executable = self.root / "hy_nomas_decode_latent"
        self.executable.write_bytes(b"test decoder executable")
        self.executable.chmod(0o700)

        self.noise_record_sha = digest(b"authenticated decoder.noise record")
        self.latent_record_sha = digest(b"authenticated input.latent_bank record")
        self.continuation_manifest = self.root / "continuation-inputs.json"
        manifest = {
            "schema": "cogni_ml.hy_nomas_continuation_inputs/v1",
            "payload": "continuation-inputs.bin",
            "payload_bytes": 8,
            "payload_sha256": digest(b"payload!"),
            "qwen_input_bundle": {
                "manifest_path": str(self.base_inputs),
                "manifest_sha256": self.base_inputs_sha,
            },
            "tensors": {
                "input.latent_bank": {
                    "dtype": "F32",
                    "shape": list(video.LATENT_BANK_SHAPE),
                    "offset": 0,
                    "nbytes": video._f32_extent(video.LATENT_BANK_SHAPE),
                    "sha256": self.latent_record_sha,
                },
                "decoder.noise": {
                    "dtype": "F32",
                    "shape": [1, 3, 720, 288, 512],
                    "offset": 4,
                    "nbytes": 4,
                    "sha256": self.noise_record_sha,
                },
            },
        }
        self.continuation_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.continuation_sha = digest(self.continuation_manifest.read_bytes())

        self.original_timeline = self._timeline_file("original", b"\x00\x00\x00\x00")
        self.edited_timeline = self._timeline_file("edited", b"\x00\x00\x80\x3f")
        self.matrix_report = self._matrix_report()
        self.original_decoder_dir = self._decoder_dir(
            "original", self.original_timeline, digest(b"original pixels")
        )
        self.edited_decoder_dir = self._decoder_dir(
            "edited", self.edited_timeline, digest(b"edited pixels")
        )

    def _timeline_file(self, name: str, first_word: bytes) -> Path:
        path = self.root / f"matrix-{name}" / "timeline.f32le"
        path.parent.mkdir()
        path.write_bytes(first_word + bytes(video._f32_extent(video.LATENT_BANK_SHAPE) - 4))
        return path

    def _matrix_artifact(self, timeline: Path, interval: list[int], *, edited: bool) -> dict:
        directory = timeline.parent
        sha = digest(timeline.read_bytes())
        receipt_path = directory / "receipt.json"
        receipt_path.write_text(json.dumps({
            "timeline_shape": list(video.LATENT_BANK_SHAPE),
            "start_segment": interval[0],
            "stop_after_segments": interval[1],
            "noise_edit": dict(segment_edit.NOISE_EDIT) if edited else None,
            "noise_scope": segment_edit.EDIT_NOISE_SCOPE if edited else "unchanged authenticated latent/epsilon banks; no reseeding",
            "state_input_sha256": digest(b"edited state") if edited else digest(b"parent state"),
            "diagnostics": {
                "timeline": str(timeline),
                "timeline_sha256": sha,
            },
        }), encoding="utf-8")
        return {
            "interval": interval,
            "receipt_path": str(receipt_path),
            "timeline": {"path": str(timeline), "bytes": timeline.stat().st_size, "sha256": sha},
        }

    def _matrix_report(self) -> dict:
        return {
            "schema": segment_edit.REPORT_SCHEMA,
            "passed": True,
            "noise_edit": dict(segment_edit.NOISE_EDIT),
            "identity": {
                "inputs": str(self.continuation_manifest),
                "inputs_sha256": self.continuation_sha,
                "conditioning_sha256": self.base_inputs_sha,
                "upstream_revision": "test-upstream-revision",
            },
            "source_bank": {
                "manifest_path": str(self.continuation_manifest),
                "manifest_sha256": self.continuation_sha,
                "record_sha256": self.latent_record_sha,
                "record_bytes": video._f32_extent(video.LATENT_BANK_SHAPE),
                "shape": list(video.LATENT_BANK_SHAPE),
            },
            "intervals": {arm: [list(interval) for interval in intervals]
                          for arm, intervals in segment_edit.ARM_INTERVALS.items()},
            "state_chains": {
                "original_control": {"state_input_sha256": digest(b"parent state")},
                "edited_restart_chain": [
                    {"state_input_sha256": digest(b"parent state")},
                    {"state_input_sha256": digest(b"edited state")},
                ],
            },
            "artifacts": {
                "original_control": [self._matrix_artifact(self.original_timeline, [1, 3], edited=False)],
                "edited_restart_chain": [
                    {"interval": [1, 2]},
                    self._matrix_artifact(self.edited_timeline, [2, 3], edited=True),
                ],
            },
        }

    def _decoder_receipt(self, timeline: Path, timeline_sha: str, output_sha: str) -> dict:
        geometry = video._admitted_geometry(90)
        return {
            "schema": video.DECODER_RECEIPT_SCHEMA,
            "upstream_revision": "test-upstream-revision",
            "executable": str(self.executable),
            "executable_sha256": digest(self.executable.read_bytes()),
            "inputs": str(self.base_inputs),
            "inputs_sha256": self.base_inputs_sha,
            "latent": str(timeline),
            "latent_sha256": timeline_sha,
            "latent_bank_sha256": timeline_sha,
            "latent_bytes": video._f32_extent(video.LATENT_BANK_SHAPE),
            "latent_bank_bytes": video._f32_extent(video.LATENT_BANK_SHAPE),
            "latent_dtype": "F32 little-endian",
            "latent_shape": list(geometry["latent_shape"]),
            "latent_frames": 90,
            "latent_selected_frames": 90,
            "latent_bank_shape": list(video.LATENT_BANK_SHAPE),
            "latent_bank_frames": 180,
            "latent_selection": geometry["latent_selection"],
            "decoder_weight_cache_sha256": digest(b"decoder cache"),
            "stem_weights_sha256": digest(b"stem weights"),
            "head_weights_sha256": digest(b"head weights"),
            "pixel_noise_source": str(self.continuation_manifest),
            "pixel_noise_manifest_sha256": self.continuation_sha,
            "pixel_noise_record_sha256": self.noise_record_sha,
            "pixel_noise_shape": list(geometry["pixel_shape"]),
            "pixel_noise_frames": 360,
            "pixel_noise_selection": geometry["noise_selection"],
            "pixelumm": geometry["settings"],
            "precision": comparison.EXPECTED_PRECISION,
            "frames": 360,
            "width": 512,
            "height": 288,
            "fps": 24,
            "output_dtype": "F32 little-endian planar RGB",
            "output": "unused",
            "output_bytes": geometry["pixel_bytes"],
            "output_sha256": output_sha,
            "output_shape": list(geometry["pixel_shape"]),
            "metrics": {
                "decoder_encode_allocations": 0,
                "decoder_encode_readbacks": 0,
                "decoder_preterminal_waits": 0,
                "decoder_waits": 1,
                "latent_prefix_gpu_gathered": 1,
            },
            "full_upstream_video_parity": False,
        }

    def _decoder_dir(self, name: str, timeline: Path, output_sha: str) -> Path:
        directory = self.root / f"decoder-{name}"
        directory.mkdir()
        receipt = self._decoder_receipt(timeline, digest(timeline.read_bytes()), output_sha)
        receipt["output"] = str(directory / "pixels.f32le")
        (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return directory

    def _compare(self, original_dir: Path | None = None, edited_dir: Path | None = None) -> dict:
        def region(start: int, stop: int) -> dict:
            return {
                "frames": [start, stop],
                "elements": (stop - start) * 3,
                "bitwise_equal": True,
                "bitwise_changed_elements": 0,
                "rmse": 0.0,
                "mean_absolute_error": 0.0,
                "max_absolute_error": 0.0,
            }

        with mock.patch.object(segment_edit, "compare_index", return_value=self.matrix_report) as fresh:
            with mock.patch.object(
                comparison, "_compare_pixel_files",
                return_value={
                    "prefix_integrity_passed": True,
                    "untouched_prefix": region(0, 100),
                    "a_overlap_tail": region(100, 120),
                    "regions": {
                        "untouched_prefix": region(0, 100),
                        "a_overlap_tail": region(100, 120),
                        "b": region(120, 240),
                        "c": region(240, 360),
                    },
                    "diagnostic_only": True,
                },
            ):
                report = comparison.compare_runs(
                    str(self.index_path),
                    str(original_dir or self.original_decoder_dir),
                    str(edited_dir or self.edited_decoder_dir),
                )
        fresh.assert_called_once_with(str(self.index_path.resolve()))
        return report

    def _rewrite_receipt(self, directory: Path, mutate) -> None:
        path = directory / "receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        mutate(receipt)
        path.write_text(json.dumps(receipt), encoding="utf-8")

    def test_compare_recomputes_matrix_and_selects_exact_original_and_edited_terminals(self):
        report = self._compare()
        self.assertTrue(report["pair_integrity_passed"])
        self.assertEqual(report["dit"]["selected_timelines"]["original_control"]["interval"], [1, 3])
        self.assertEqual(report["dit"]["selected_timelines"]["edited_restart_chain"]["interval"], [2, 3])
        self.assertFalse(report["full_upstream_video_parity"])
        self.assertEqual(report["pixels"]["original"]["sha256"], digest(b"original pixels"))
        self.assertEqual(report["decoder"]["output_sha256"]["edited"], digest(b"edited pixels"))

    def test_rejects_decoder_bound_to_wrong_arm_or_bank(self):
        self._rewrite_receipt(
            self.edited_decoder_dir,
            lambda receipt: receipt.update(latent=str(self.original_timeline)),
        )
        with self.assertRaisesRegex(ValueError, "selected edited_restart_chain timeline"):
            self._compare()

        # Restore the correct path, then mutate the hash while keeping the path.
        self._rewrite_receipt(
            self.edited_decoder_dir,
            lambda receipt: receipt.update(latent=str(self.edited_timeline),
                                           latent_bank_sha256=digest(b"wrong bank")),
        )
        with self.assertRaisesRegex(ValueError, "does not match its authenticated timeline"):
            self._compare()

    def test_rejects_decoder_noise_manifest_path_or_record_mismatch(self):
        self._rewrite_receipt(
            self.edited_decoder_dir,
            lambda receipt: receipt.update(pixel_noise_manifest_sha256=digest(b"other manifest")),
        )
        with self.assertRaisesRegex(ValueError, "decoder noise manifest does not match"):
            self._compare()

        self._rewrite_receipt(
            self.edited_decoder_dir,
            lambda receipt: receipt.update(pixel_noise_manifest_sha256=self.continuation_sha,
                                           pixel_noise_record_sha256=digest(b"other noise")),
        )
        with self.assertRaisesRegex(ValueError, "decoder.noise record does not match"):
            self._compare()

    def test_rejects_latent_and_pixel_geometry_mutations(self):
        mutations = [
            ("latent_shape", [1, 32, 180, 9, 16]),
            ("latent_bank_frames", 90),
            ("frames", 240),
            ("output_shape", [1, 3, 240, 288, 512]),
            ("pixel_noise_shape", [1, 3, 240, 288, 512]),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                self._rewrite_receipt(self.edited_decoder_dir,
                                      lambda receipt, k=key, v=value: receipt.update({k: v}))
                with self.assertRaises(ValueError):
                    self._compare()
                self._rewrite_receipt(self.edited_decoder_dir,
                                      lambda receipt, k=key: receipt.update({
                                          k: self._decoder_receipt(
                                              self.edited_timeline,
                                              digest(self.edited_timeline.read_bytes()),
                                              digest(b"edited pixels"),
                                          )[k]
                                      }))

    def test_rejects_decoder_executable_weights_precision_settings_and_windows(self):
        mutations = [
            ("executable_sha256", digest(b"other decoder")),
            ("decoder_weight_cache_sha256", digest(b"other cache")),
            ("stem_weights_sha256", digest(b"other stem")),
            ("precision", "F32 decoder"),
            ("pixelumm", {"steps": 1, "window_latents": 10, "overlap_latents": 5,
                          "starts": list(range(0, 81, 5))[:-1]}),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                self._rewrite_receipt(self.edited_decoder_dir,
                                      lambda receipt, k=key, v=value: receipt.update({k: v}))
                with self.assertRaises(ValueError):
                    self._compare()
                self._rewrite_receipt(self.edited_decoder_dir,
                                      lambda receipt, k=key: receipt.update({
                                          k: self._decoder_receipt(
                                              self.edited_timeline,
                                              digest(self.edited_timeline.read_bytes()),
                                              digest(b"edited pixels"),
                                          )[k]
                                      }))

    def test_rejects_decoder_allocation_or_wait_metric_mutations(self):
        for key, value in (("decoder_encode_allocations", 1),
                           ("decoder_encode_readbacks", 1),
                           ("decoder_preterminal_waits", 1),
                           ("decoder_waits", 0)):
            with self.subTest(key=key):
                self._rewrite_receipt(
                    self.edited_decoder_dir,
                    lambda receipt, k=key, v=value: receipt["metrics"].update({k: v}),
                )
                with self.assertRaises(ValueError):
                    self._compare()
                self._rewrite_receipt(
                    self.edited_decoder_dir,
                    lambda receipt, k=key: receipt["metrics"].update({k: 0 if k != "decoder_waits" else 1}),
                )

    def _write_small_pair(self, original_values: list[float], edited_values: list[float]) -> tuple[Path, Path, dict]:
        original = self.root / "pixels-original.f32le"
        edited = self.root / "pixels-edited.f32le"
        original_bytes = struct.pack("<" + "f" * len(original_values), *original_values)
        edited_bytes = struct.pack("<" + "f" * len(edited_values), *edited_values)
        original.write_bytes(original_bytes)
        edited.write_bytes(edited_bytes)
        regions = (
            ("untouched_prefix", 0, 2),
            ("a_overlap_tail", 2, 3),
            ("b", 3, 4),
        )
        result = comparison._compare_pixel_files(
            original,
            edited,
            original_sha256=digest(original_bytes),
            edited_sha256=digest(edited_bytes),
            expected_bytes=len(original_bytes),
            frames=4,
            height=1,
            width=2,
            channels=3,
            prefix_frames=2,
            regions=regions,
        )
        return original, edited, result

    def test_planar_rgb_prefix_and_frame_100_boundary_are_checked_per_channel(self):
        # NCTHW flattening is channel-major. Mutations hit G/B after the exact prefix.
        original = [0.0] * (3 * 4 * 2)
        edited = list(original)
        edited[(1 * 4 + 2) * 2 + 1] = 0.25  # G, frame 2: A overlap tail.
        edited[(2 * 4 + 3) * 2] = -0.5     # B, frame 3: B diagnostic region.
        _, _, result = self._write_small_pair(original, edited)
        self.assertTrue(result["prefix_integrity_passed"])
        self.assertTrue(result["regions"]["untouched_prefix"]["bitwise_equal"])
        self.assertEqual(result["regions"]["a_overlap_tail"]["bitwise_changed_elements"], 1)
        self.assertEqual(result["regions"]["b"]["bitwise_changed_elements"], 1)

    def test_rejects_prefix_mutation_in_green_or_blue_channel(self):
        original = [0.0] * (3 * 4 * 2)
        edited = list(original)
        edited[(2 * 4 + 1) * 2] = 0.1
        original_bytes = struct.pack("<" + "f" * len(original), *original)
        edited_bytes = struct.pack("<" + "f" * len(edited), *edited)
        original_path = self.root / "prefix-original.f32le"
        edited_path = self.root / "prefix-edited.f32le"
        original_path.write_bytes(original_bytes)
        edited_path.write_bytes(edited_bytes)
        with self.assertRaisesRegex(ValueError, r"prefix frames \[0, 2\) are not bitwise equal"):
            comparison._compare_pixel_files(
                original_path,
                edited_path,
                original_sha256=digest(original_bytes),
                edited_sha256=digest(edited_bytes),
                expected_bytes=len(original_bytes),
                frames=4,
                height=1,
                width=2,
                channels=3,
                prefix_frames=2,
                regions=(("untouched_prefix", 0, 2), ("a_overlap_tail", 2, 3), ("b", 3, 4)),
            )

    def test_pixel_stream_rejects_missing_nonfinite_and_receipt_hash_mismatch(self):
        original_bytes = struct.pack("<12f", *([0.0] * 12))
        edited_bytes = struct.pack("<12f", *([0.0] * 12))
        original = self.root / "small-original.f32le"
        edited = self.root / "small-edited.f32le"
        original.write_bytes(original_bytes)
        edited.write_bytes(edited_bytes)
        kwargs = dict(
            original_sha256=digest(original_bytes),
            edited_sha256=digest(edited_bytes),
            expected_bytes=len(original_bytes),
            frames=2,
            height=1,
            width=2,
            channels=3,
            prefix_frames=1,
            regions=(("untouched_prefix", 0, 1), ("a_overlap_tail", 1, 2)),
        )
        with self.assertRaisesRegex(ValueError, "cannot open original pixels"):
            comparison._compare_pixel_files(self.root / "missing.f32le", edited, **kwargs)

        edited.write_bytes(struct.pack("<12f", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, float("nan"), 0.0))
        kwargs["edited_sha256"] = digest(edited.read_bytes())
        with self.assertRaisesRegex(ValueError, "non-finite F32"):
            comparison._compare_pixel_files(original, edited, **kwargs)

        kwargs["edited_sha256"] = digest(edited_bytes)
        edited.write_bytes(bytes([edited_bytes[0] ^ 1]) + edited_bytes[1:])
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match its receipt"):
            comparison._compare_pixel_files(original, edited, **kwargs)

    def test_cli_writes_a_fresh_report_and_rejects_existing_output(self):
        output = self.root / "comparison.json"
        args = [
            "--edit-index", str(self.index_path),
            "--original-decoder-dir", str(self.original_decoder_dir),
            "--edited-decoder-dir", str(self.edited_decoder_dir),
            "--output", str(output),
        ]
        with mock.patch.object(comparison, "compare_runs", return_value={
            "schema": comparison.REPORT_SCHEMA,
            "pair_integrity_passed": True,
        }):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(comparison.main(args), 0)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema"], comparison.REPORT_SCHEMA)

        with mock.patch.object(comparison, "compare_runs", return_value={
            "schema": comparison.REPORT_SCHEMA,
            "pair_integrity_passed": True,
        }):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(comparison.main(args), 2)


if __name__ == "__main__":
    unittest.main()
