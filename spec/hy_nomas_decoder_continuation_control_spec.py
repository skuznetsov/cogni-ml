#!/usr/bin/env python3
"""CPU-only admission tests for the exact 60-to-240 PixelUMM control."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SUPPORT = Path(__file__).resolve().parent / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_decoder_continuation_control as control


def valid_receipt() -> dict[str, object]:
    return {
        "schema": control.NATIVE_SCHEMA,
        "upstream_revision": control.UPSTREAM_REVISION,
        "executable": "/tmp/hy_nomas_decode_latent",
        "executable_sha256": control.EXPECTED_NATIVE_EXECUTABLE_SHA256,
        "precision": control.EXPECTED_PRECISION,
        "latent_sha256": "a" * 64,
        "latent_bank_sha256": "a" * 64,
        "latent": "/tmp/latent.f32le",
        "decoder_weight_cache_sha256": control.EXPECTED_NATIVE_CACHE_SHA256,
        "stem_weights_sha256": control.EXPECTED_NATIVE_STEM_SHA256,
        "head_weights_sha256": control.EXPECTED_NATIVE_HEAD_SHA256,
        "latent_bytes": control.LATENT_BYTES,
        "latent_bank_bytes": control.LATENT_BYTES,
        "latent_dtype": "F32 little-endian",
        "latent_shape": list(control.LATENT_SHAPE),
        "latent_frames": 60,
        "latent_selected_frames": 60,
        "latent_prefix_start_frame": 0,
        "latent_bank_shape": list(control.LATENT_SHAPE),
        "latent_bank_frames": 60,
        "latent_selection": "authenticated latent bank global frames [0, 60) consumed directly",
        "latent_boundary": control.EXPECTED_LATENT_BOUNDARY,
        "continuation_manifest_sha256": "b" * 64,
        "continuation_source": "/tmp/continuation.json",
        "pixel_noise_source": "/tmp/continuation.json",
        "pixel_noise_manifest_sha256": "b" * 64,
        "pixel_noise_record_sha256": "c" * 64,
        "pixel_noise_shape": list(control.NOISE_PREFIX_SHAPE),
        "pixel_noise_frames": 240,
        "pixel_noise_prefix_start_frame": 0,
        "pixel_noise_selection": control.EXPECTED_NOISE_SELECTION,
        "pixelumm": {
            "steps": 1,
            "window_latents": 10,
            "overlap_latents": 5,
            "starts": list(control.WINDOW_STARTS),
        },
        "frames": 240,
        "width": 512,
        "height": 288,
        "fps": 24,
        "output_dtype": "F32 little-endian planar RGB",
        "output": "/tmp/native-pixels.f32le",
        "output_bytes": control.RAW_BYTES,
        "output_sha256": "d" * 64,
        "output_shape": list(control.RAW_SHAPE),
        "full_upstream_video_parity": False,
    }


class ContinuationControlAdmissionSpec(unittest.TestCase):
    def test_admits_exact_saved_native_decode_contract(self) -> None:
        admitted = control.validate_native_decode_receipt_document(valid_receipt())
        self.assertEqual(admitted["latent_shape"], control.LATENT_SHAPE)
        self.assertEqual(admitted["noise_prefix_shape"], control.NOISE_PREFIX_SHAPE)
        self.assertEqual(admitted["window_starts"], control.WINDOW_STARTS)
        self.assertEqual(admitted["native_output_representation"], control.NATIVE_DISPLAY_REPRESENTATION)
        self.assertEqual(admitted["native_executable_sha256"], control.EXPECTED_NATIVE_EXECUTABLE_SHA256)
        self.assertFalse(admitted["full_upstream_video_parity"])

    def test_rejects_unpinned_native_executable(self) -> None:
        receipt = valid_receipt()
        receipt["executable_sha256"] = "e" * 64
        with self.assertRaisesRegex(control.DecoderControlError, "executable"):
            control.validate_native_decode_receipt_document(receipt)

    def test_rejects_receipt_executable_path_with_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native-executable"
            path.write_bytes(b"not the pinned decoder")
            with self.assertRaisesRegex(control.DecoderControlError, "bytes differ"):
                control._authenticate_native_executable(
                    path, control.EXPECTED_NATIVE_EXECUTABLE_SHA256
                )

    def test_rejects_caller_supplied_pixel_range_claim(self) -> None:
        receipt = valid_receipt()
        receipt["output_range"] = "[-1,1]"
        with self.assertRaisesRegex(control.DecoderControlError, "derived from the pinned native executable"):
            control.validate_native_decode_receipt_document(receipt)

    def test_bf16_raw_is_postprocessed_before_display_comparison(self) -> None:
        torch = control.torch
        raw = torch.tensor([-1.0, -0.25, 0.5, 1.0], dtype=torch.bfloat16, device="cpu")
        expected_display = torch.tensor([0.0, 0.375, 0.75, 1.0], dtype=torch.float32)

        display = control._display_pixels_from_bf16(raw)

        self.assertFalse(torch.equal(raw.float(), expected_display))
        self.assertEqual(display.dtype, torch.float32)
        torch.testing.assert_close(display, expected_display, atol=control.ATOL, rtol=control.RTOL)

    def test_display_conversion_rejects_f32_raw_without_bf16_boundary(self) -> None:
        torch = control.torch
        raw_f32 = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cpu")
        with self.assertRaisesRegex(control.DecoderControlError, "actual BF16 tensor"):
            control._display_pixels_from_bf16(raw_f32)

    def test_authenticates_fake_receipt_bytes_before_following_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(valid_receipt()), encoding="utf-8")
            raw = path.read_bytes()
            receipt, digest, admitted = control.authenticate_receipt(
                path, hashlib.sha256(raw).hexdigest()
            )
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(receipt["full_upstream_video_parity"], False)
            self.assertEqual(admitted["window_starts"], control.WINDOW_STARTS)

    def test_rejects_fake_receipt_with_wrong_caller_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(valid_receipt()), encoding="utf-8")
            with self.assertRaisesRegex(control.DecoderControlError, "caller authorization"):
                control.authenticate_receipt(path, "0" * 64)

    def test_rejects_upstream_parity_claim(self) -> None:
        receipt = valid_receipt()
        receipt["full_upstream_video_parity"] = True
        with self.assertRaisesRegex(control.DecoderControlError, "parity"):
            control.validate_native_decode_receipt_document(receipt)

    def test_rejects_changed_window_schedule(self) -> None:
        receipt = valid_receipt()
        receipt["pixelumm"] = dict(receipt["pixelumm"], starts=[0, 5, 10, 15, 20])
        with self.assertRaisesRegex(control.DecoderControlError, "window"):
            control.validate_native_decode_receipt_document(receipt)

    def test_rejects_noise_prefix_not_axis_two_global_prefix(self) -> None:
        receipt = valid_receipt()
        receipt["pixel_noise_selection"] = "first flattened 240 frames"
        with self.assertRaisesRegex(control.DecoderControlError, "prefix"):
            control.validate_native_decode_receipt_document(receipt)

    def test_rejects_latent_extent_mismatch(self) -> None:
        receipt = valid_receipt()
        receipt["latent_bytes"] = control.LATENT_BYTES - 4
        with self.assertRaisesRegex(control.DecoderControlError, "latent"):
            control.validate_native_decode_receipt_document(receipt)

    def test_rejects_noise_record_that_does_not_match_receipt(self) -> None:
        record = {
            "dtype": "F32",
            "shape": list(control.NOISE_SOURCE_SHAPE),
            "nbytes": control.NOISE_SOURCE_BYTES,
            "sha256": "e" * 64,
            "offset": 0,
        }
        with self.assertRaisesRegex(control.DecoderControlError, "noise record"):
            control.validate_noise_record(record, expected_sha256="c" * 64)

    def test_rejects_noise_record_that_is_only_the_240_frame_prefix(self) -> None:
        record = {
            "dtype": "F32",
            "shape": list(control.NOISE_PREFIX_SHAPE),
            "nbytes": control.RAW_BYTES,
            "sha256": "c" * 64,
            "offset": 0,
        }
        with self.assertRaisesRegex(control.DecoderControlError, "geometry"):
            control.validate_noise_record(record, expected_sha256="c" * 64)

    def test_rejects_weight_provenance_that_differs_from_receipt(self) -> None:
        receipt = valid_receipt()
        for field, values in (
            ("decoder_weight_cache_sha256", {
                "cache_manifest_sha256": "9" * 64,
                "stem_file_sha256": control.EXPECTED_NATIVE_STEM_SHA256,
                "head_file_sha256": control.EXPECTED_NATIVE_HEAD_SHA256,
            }),
            ("stem_weights_sha256", {
                "cache_manifest_sha256": control.EXPECTED_NATIVE_CACHE_SHA256,
                "stem_file_sha256": "9" * 64,
                "head_file_sha256": control.EXPECTED_NATIVE_HEAD_SHA256,
            }),
            ("head_weights_sha256", {
                "cache_manifest_sha256": control.EXPECTED_NATIVE_CACHE_SHA256,
                "stem_file_sha256": control.EXPECTED_NATIVE_STEM_SHA256,
                "head_file_sha256": "9" * 64,
            }),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                control.DecoderControlError, field
            ):
                control.validate_weight_identities(receipt, **values)

    def test_output_prefix_must_be_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "control"
            outputs = control.output_paths(prefix)
            self.assertFalse(any(path.exists() for path in outputs))
            outputs[0].touch()
            with self.assertRaisesRegex(control.DecoderControlError, "replace"):
                control.output_paths(prefix)


if __name__ == "__main__":
    unittest.main()
