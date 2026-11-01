"""CPU-only falsifiers for same-decoder QBit video provenance."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_cache_qbit_video_compare as comparison


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decoder_receipt(latent_sha: str, output_sha: str) -> dict:
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
        "latent_shape": [1, 32, 60, 9, 16],
        "latent_frames": 60,
        "latent_selected_frames": 60,
        "latent_bank_shape": [1, 32, 180, 9, 16],
        "latent_bank_frames": 180,
        "latent_selection": "first 60 global frames from each NCTHW channel using HyNomasTimelineOps::PrefixPlan",
        "decoder_weight_cache_sha256": digest(b"decoder cache"),
        "stem_weights_sha256": digest(b"stem"),
        "head_weights_sha256": digest(b"head"),
        "pixel_noise_manifest_sha256": comparison.CONTINUATION_INPUTS_SHA256,
        "pixel_noise_record_sha256": digest(b"noise record"),
        "pixel_noise_source": "/scratch/noise-manifest.json",
        "pixel_noise_shape": [1, 3, 240, 288, 512],
        "pixel_noise_frames": 240,
        "pixel_noise_selection": "first 240 global frames from each NCTHW channel using HyNomasTimelineOps::PrefixPlan",
        "pixelumm": {
            "steps": 1,
            "window_latents": 10,
            "overlap_latents": 5,
            "starts": list(range(0, 51, 5)),
        },
        "frames": 240,
        "width": 512,
        "height": 288,
        "fps": 24,
        "output_dtype": "F32 little-endian planar RGB",
        "precision": "F32 test decoder",
        "output_bytes": comparison.PIXEL_BYTES,
        "output_sha256": output_sha,
        "output_shape": [1, 3, 240, 288, 512],
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

    def validate(self) -> dict:
        return comparison.validate_decoder_receipts(
            self.exact, self.qbit, self.timeline_exact, self.timeline_qbit,
            comparison.CONTINUATION_INPUTS_SHA256, comparison.CONTINUATION_INPUTS_SHA256,
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


if __name__ == "__main__":
    unittest.main()
