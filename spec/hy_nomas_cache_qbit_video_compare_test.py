"""CPU-only falsifiers for same-decoder QBit video provenance."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_cache_qbit_video_compare as comparison


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decoder_receipt(latent_sha: str, output_sha: str, *, latent_frames: int = 60) -> dict:
    geometry = comparison._admitted_geometry(latent_frames)
    return {
        "schema": comparison.DECODER_RECEIPT_SCHEMA,
        "upstream_revision": "upstream-test-revision",
        "executable": "/opt/test/hy_nomas_decode_latent",
        "executable_sha256": digest(b"decoder binary"),
        "inputs_sha256": comparison.BASE_INPUTS_SHA256,
        "latent_sha256": latent_sha,
        "latent_bank_sha256": latent_sha,
        "latent_bytes": 3_317_760,
        "latent_bank_bytes": 3_317_760,
        "latent_dtype": "F32 little-endian",
        "latent_shape": list(geometry["latent_shape"]),
        "latent_frames": latent_frames,
        "latent_selected_frames": latent_frames,
        "latent_bank_shape": [1, 32, 180, 9, 16],
        "latent_bank_frames": 180,
        "latent_selection": geometry["latent_selection"],
        "decoder_weight_cache_sha256": digest(b"decoder cache"),
        "stem_weights_sha256": digest(b"stem"),
        "head_weights_sha256": digest(b"head"),
        "pixel_noise_manifest_sha256": comparison.CONTINUATION_INPUTS_SHA256,
        "pixel_noise_record_sha256": digest(b"noise record"),
        "pixel_noise_source": "/scratch/noise-manifest.json",
        "pixel_noise_shape": list(geometry["pixel_shape"]),
        "pixel_noise_frames": geometry["pixel_frames"],
        "pixel_noise_selection": geometry["noise_selection"],
        "pixelumm": geometry["settings"],
        "frames": geometry["pixel_frames"],
        "width": 512,
        "height": 288,
        "fps": 24,
        "output_dtype": "F32 little-endian planar RGB",
        "precision": "F32 test decoder",
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


class CacheQBitVideoComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timeline_exact = digest(b"exact timeline")
        self.timeline_qbit = digest(b"qbit timeline")
        self.exact = decoder_receipt(self.timeline_exact, digest(b"exact pixels"))
        self.qbit = decoder_receipt(self.timeline_qbit, digest(b"qbit pixels"))

    def validate(self, *, latent_frames: int = 60) -> dict:
        return comparison.validate_decoder_receipts(
            self.exact, self.qbit, self.timeline_exact, self.timeline_qbit,
            comparison.CONTINUATION_INPUTS_SHA256, comparison.CONTINUATION_INPUTS_SHA256,
            latent_frames=latent_frames,
        )

    def test_accepts_linked_distinct_inputs_with_identical_decoder_identity(self):
        result = self.validate()
        self.assertTrue(result["decoder_pair_integrity_passed"])
        self.assertEqual(result["decoder_sha256"], self.exact["executable_sha256"])
        self.assertNotEqual(self.exact["inputs_sha256"], result["dit_continuation_manifest_sha256"])
        self.assertTrue(result["active_latent_treatment"])
        self.assertTrue(result["active_pixel_output"])
        extents = result["latent_extents"]
        self.assertEqual(extents["source_bank"]["f32_bytes"], {
            "exact": 3_317_760, "qbit": 3_317_760,
        })
        self.assertEqual(extents["selected_decoder_prefix"]["f32_bytes"], {
            "exact": 1_105_920, "qbit": 1_105_920,
        })

    def test_rejects_decoder_identity_mismatch(self):
        self.qbit["pixel_noise_record_sha256"] = digest(b"different noise")
        with self.assertRaisesRegex(ValueError, "pair identity mismatch"):
            self.validate()

    def test_rejects_decoder_base_manifest_outside_pinned_bundle(self):
        self.qbit["inputs_sha256"] = digest(b"different base inputs")
        with self.assertRaisesRegex(ValueError, "base input manifest differs from the pinned bundle"):
            self.validate()

    def test_rejects_boolean_in_place_of_numeric_decoder_setting(self):
        self.qbit["pixelumm"]["steps"] = True
        with self.assertRaisesRegex(ValueError, "PixelUMM setting steps is invalid"):
            self.validate()

    def test_rejects_decoder_bank_not_linked_to_its_timeline(self):
        self.qbit["latent_bank_sha256"] = digest(b"unrelated latent")
        with self.assertRaisesRegex(ValueError, "does not match the authenticated DiT timeline"):
            self.validate()

    def test_rejects_legacy_latent_bytes_misreported_as_selected_prefix_extent(self):
        self.exact["latent_bytes"] = 1_105_920
        with self.assertRaisesRegex(ValueError, "legacy latent_bytes must equal source-bank extent"):
            self.validate()

    def test_rejects_latent_bank_bytes_misreported_as_selected_prefix_extent(self):
        self.qbit["latent_bank_bytes"] = 1_105_920
        with self.assertRaisesRegex(ValueError, "latent_bank_bytes must equal source-bank extent"):
            self.validate()

    def test_rejects_source_bank_shape_reported_as_selected_prefix_shape(self):
        self.qbit["latent_shape"] = list(comparison.LATENT_BANK_SHAPE)
        with self.assertRaisesRegex(ValueError, "latent_shape does not match the admitted geometry"):
            self.validate()

    def test_accepts_admitted_90_latent_15_second_decode_geometry(self):
        exact_receipt = decoder_receipt(
            self.timeline_exact, digest(b"exact 15 second pixels"), latent_frames=90
        )
        qbit_receipt = decoder_receipt(
            self.timeline_qbit, digest(b"qbit 15 second pixels"), latent_frames=90
        )
        report = comparison.validate_decoder_receipts(
            exact_receipt, qbit_receipt, self.timeline_exact, self.timeline_qbit,
            comparison.CONTINUATION_INPUTS_SHA256, comparison.CONTINUATION_INPUTS_SHA256,
            latent_frames=90,
        )
        geometry = comparison._admitted_geometry(90)
        self.assertEqual(geometry["pixel_shape"], (1, 3, 360, 288, 512))
        self.assertEqual(geometry["pixel_bytes"], 637_009_920)
        self.assertEqual(geometry["settings"]["starts"], list(range(0, 81, 5)))
        self.assertEqual(report["latent_extents"]["selected_decoder_prefix"]["shape"], [1, 32, 90, 9, 16])
        self.assertEqual(report["latent_extents"]["selected_decoder_prefix"]["frames"], 90)
        self.assertEqual(report["latent_extents"]["selected_decoder_prefix"]["f32_bytes"], {
            "exact": 1_658_880, "qbit": 1_658_880,
        })
        self.assertEqual(report["pixel_noise"]["shape"], [1, 3, 360, 288, 512])

    def test_rejects_unadmitted_decoder_latent_frame_count(self):
        with self.assertRaisesRegex(ValueError, "60 or 90 frames"):
            self.validate(latent_frames=75)

    def test_90_frame_decode_rejects_mismatched_noise_and_bank(self):
        exact_receipt = decoder_receipt(self.timeline_exact, digest(b"exact"), latent_frames=90)
        qbit_receipt = decoder_receipt(self.timeline_qbit, digest(b"qbit"), latent_frames=90)
        qbit_receipt["pixel_noise_record_sha256"] = digest(b"other noise")
        with self.assertRaisesRegex(ValueError, "pair identity mismatch: pixel_noise_record_sha256"):
            comparison.validate_decoder_receipts(
                exact_receipt, qbit_receipt, self.timeline_exact, self.timeline_qbit,
                comparison.CONTINUATION_INPUTS_SHA256, comparison.CONTINUATION_INPUTS_SHA256,
                latent_frames=90,
            )

        qbit_receipt = decoder_receipt(self.timeline_qbit, digest(b"qbit"), latent_frames=90)
        qbit_receipt["latent_bank_sha256"] = digest(b"wrong source bank")
        with self.assertRaisesRegex(ValueError, "does not match the authenticated DiT timeline"):
            comparison.validate_decoder_receipts(
                exact_receipt, qbit_receipt, self.timeline_exact, self.timeline_qbit,
                comparison.CONTINUATION_INPUTS_SHA256, comparison.CONTINUATION_INPUTS_SHA256,
                latent_frames=90,
            )

    def test_90_frame_decode_rejects_wrong_output_shape_or_window_starts(self):
        exact_receipt = decoder_receipt(self.timeline_exact, digest(b"exact"), latent_frames=90)
        qbit_receipt = decoder_receipt(self.timeline_qbit, digest(b"qbit"), latent_frames=90)
        qbit_receipt["output_shape"] = [1, 3, 240, 288, 512]
        with self.assertRaisesRegex(ValueError, "output_shape does not match"):
            comparison.validate_decoder_receipts(
                exact_receipt, qbit_receipt, self.timeline_exact, self.timeline_qbit,
                comparison.CONTINUATION_INPUTS_SHA256, comparison.CONTINUATION_INPUTS_SHA256,
                latent_frames=90,
            )

        qbit_receipt = decoder_receipt(self.timeline_qbit, digest(b"qbit"), latent_frames=90)
        qbit_receipt["pixelumm"]["starts"].remove(80)
        with self.assertRaisesRegex(ValueError, "window starts are invalid"):
            comparison.validate_decoder_receipts(
                exact_receipt, qbit_receipt, self.timeline_exact, self.timeline_qbit,
                comparison.CONTINUATION_INPUTS_SHA256, comparison.CONTINUATION_INPUTS_SHA256,
                latent_frames=90,
            )

    def _compact_fixture(self, root: Path) -> tuple[dict, Path, Path]:
        timeline_bytes = comparison._f32_extent(comparison.LATENT_BANK_SHAPE)
        matrix_runs = {}
        decoder_dirs = {}
        for name, payload in (
            ("exact", bytes(timeline_bytes)),
            ("qbit", struct.pack("<f", 1.0) + bytes(timeline_bytes - 4)),
        ):
            matrix_dir = root / f"matrix-{name}-terminal"
            matrix_dir.mkdir()
            timeline_path = matrix_dir / "timeline.f32le"
            timeline_path.write_bytes(payload)
            timeline_sha = digest(payload)
            receipt_path = matrix_dir / "receipt.json"
            receipt_path.write_text(json.dumps({
                "timeline_shape": list(comparison.LATENT_BANK_SHAPE),
                "diagnostics": {
                    "timeline": str(timeline_path),
                    "timeline_sha256": timeline_sha,
                },
            }, sort_keys=True) + "\n", encoding="utf-8")
            matrix_runs[name] = {
                "receipt_path": str(receipt_path),
                "interval": [2, 3],
                "timeline": {
                    "path": str(timeline_path),
                    "bytes": timeline_bytes,
                    "sha256": timeline_sha,
                },
            }

            decoder_dir = root / f"decoder-{name}"
            decoder_dir.mkdir()
            receipt = decoder_receipt(timeline_sha, digest(f"{name} pixels".encode()), latent_frames=90)
            receipt["latent"] = str(timeline_path)
            receipt["output"] = str(decoder_dir / "pixels.f32le")
            (decoder_dir / "receipt.json").write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            decoder_dirs[name] = decoder_dir

        def chain(first_two: list[dict], terminal: dict) -> list[dict]:
            return first_two + [terminal]

        report = {
            "passed": True,
            "identity": {"inputs_sha256": comparison.CONTINUATION_INPUTS_SHA256},
            "source_bank": {
                "shape": list(comparison.LATENT_BANK_SHAPE),
                "record_bytes": timeline_bytes,
            },
            "intervals": {
                "exact_control": [[0, 3]],
                "exact_restart_chain": [[0, 1], [1, 2], [2, 3]],
                "qbit_control": [[0, 3]],
                "compact_restart_chain": [[0, 1], [1, 2], [2, 3]],
            },
            "artifacts": {
                "exact_restart_chain": chain([{}, {}], matrix_runs["exact"]),
                "compact_restart_chain": chain([{}, {}], matrix_runs["qbit"]),
            },
        }
        return report, decoder_dirs["exact"], decoder_dirs["qbit"]

    def test_compact_compare_binds_restart_chain_timelines_and_authenticates_15s_pixels(self):
        with tempfile.TemporaryDirectory(prefix="hy-nomas-compact-video-") as temp:
            root = Path(temp)
            matrix_report, exact_dir, qbit_dir = self._compact_fixture(root)
            pixel_checks = []

            def authenticate(directory, receipt, label, *, expected_bytes):
                pixel_checks.append((label, expected_bytes))
                return {"path": receipt["output"], "bytes": expected_bytes,
                        "sha256": receipt["output_sha256"], "finite_f32": True,
                        "finite_elements": expected_bytes // 4}

            with mock.patch.object(
                comparison.compact_comparison, "compare_index", return_value=matrix_report
            ), mock.patch.object(comparison, "authenticate_pixels", side_effect=authenticate):
                report = comparison.compare_compact_runs("/unused/matrix.json", str(exact_dir), str(qbit_dir))

            self.assertTrue(report["pair_integrity_passed"])
            self.assertTrue(report["dit_pair_integrity_passed"])
            self.assertIs(report["full_upstream_video_parity"], False)
            self.assertEqual(report["quality_verdict"], "not assessed; visual review is pending")
            self.assertEqual(report["decoder"]["timeline_sha256"], {
                "exact": matrix_report["artifacts"]["exact_restart_chain"][-1]["timeline"]["sha256"],
                "qbit": matrix_report["artifacts"]["compact_restart_chain"][-1]["timeline"]["sha256"],
            })
            self.assertEqual(pixel_checks, [("exact", 637_009_920), ("qbit", 637_009_920)])

    def test_compact_compare_rejects_decoder_bound_to_wrong_matrix_arm(self):
        with tempfile.TemporaryDirectory(prefix="hy-nomas-compact-video-") as temp:
            root = Path(temp)
            matrix_report, exact_dir, qbit_dir = self._compact_fixture(root)
            qbit_receipt_path = qbit_dir / "receipt.json"
            qbit_receipt = json.loads(qbit_receipt_path.read_text(encoding="utf-8"))
            qbit_receipt["latent"] = matrix_report["artifacts"]["exact_restart_chain"][-1]["timeline"]["path"]
            qbit_receipt_path.write_text(json.dumps(qbit_receipt, sort_keys=True) + "\n", encoding="utf-8")

            with mock.patch.object(
                comparison.compact_comparison, "compare_index", return_value=matrix_report
            ):
                with self.assertRaisesRegex(ValueError, "selected compact-matrix timeline"):
                    comparison.compare_compact_runs("/unused/matrix.json", str(exact_dir), str(qbit_dir))

    def test_rejects_noise_manifest_not_linked_to_dit_continuation_manifest(self):
        self.exact["pixel_noise_manifest_sha256"] = digest(b"wrong continuation manifest")
        with self.assertRaisesRegex(ValueError, "noise manifest does not match"):
            comparison.validate_decoder_receipts(
                self.exact, self.qbit, self.timeline_exact, self.timeline_qbit,
                comparison.CONTINUATION_INPUTS_SHA256, comparison.CONTINUATION_INPUTS_SHA256,
            )

    def test_rejects_modified_pixels_against_receipt_sha(self):
        with tempfile.TemporaryDirectory(prefix="hy-nomas-qbit-video-") as temp:
            directory = Path(temp)
            path = directory / "pixels.f32le"
            expected = b"\x00\x00\x00\x00"
            path.write_bytes(b"\x00\x00\x80\x3f")
            receipt = {"output": str(path), "output_bytes": len(expected),
                       "output_sha256": digest(expected)}
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match its receipt"):
                comparison.authenticate_pixels(directory.resolve(), receipt, "exact", expected_bytes=4)

    def test_rejects_nonfinite_pixels_even_when_sha_matches(self):
        with tempfile.TemporaryDirectory(prefix="hy-nomas-qbit-video-") as temp:
            directory = Path(temp)
            path = directory / "pixels.f32le"
            payload = b"\x00\x00\xc0\x7f"  # little-endian F32 NaN
            path.write_bytes(payload)
            receipt = {"output": str(path), "output_bytes": len(payload),
                       "output_sha256": digest(payload)}
            with self.assertRaisesRegex(ValueError, "non-finite F32"):
                comparison.authenticate_pixels(directory.resolve(), receipt, "exact", expected_bytes=4)

    def test_cli_writes_a_fresh_json_receipt(self):
        with tempfile.TemporaryDirectory(prefix="hy-nomas-qbit-video-") as temp:
            output = Path(temp) / "comparison.json"
            report = {"schema": comparison.REPORT_SCHEMA, "pair_integrity_passed": True}
            args = [
                "--exact-dit-dir", "/unused/exact-dit",
                "--qbit-dit-dir", "/unused/qbit-dit",
                "--exact-decoder-dir", "/unused/exact-decoder",
                "--qbit-decoder-dir", "/unused/qbit-decoder",
                "--output", str(output),
            ]
            with mock.patch.object(comparison, "compare_runs", return_value=report):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(comparison.main(args), 0)
            self.assertEqual(json.loads(output.read_text()), report)
            with mock.patch.object(comparison, "compare_runs", return_value=report):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(comparison.main(args), 2)

    def test_cli_selects_compact_index_mode_and_rejects_ambiguous_modes(self):
        with tempfile.TemporaryDirectory(prefix="hy-nomas-qbit-video-cli-") as temp:
            output = Path(temp) / "compact-comparison.json"
            report = {"schema": comparison.REPORT_SCHEMA, "pair_integrity_passed": True}
            args = [
                "--compact-index", "/unused/compact-matrix.json",
                "--exact-decoder-dir", "/unused/exact-decoder",
                "--qbit-decoder-dir", "/unused/qbit-decoder",
                "--output", str(output),
            ]
            with mock.patch.object(comparison, "compare_compact_runs", return_value=report):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(comparison.main(args), 0)
            self.assertEqual(json.loads(output.read_text()), report)

            ambiguous = args + ["--exact-dit-dir", "/unused/exact-dit"]
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    comparison.main(ambiguous)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
