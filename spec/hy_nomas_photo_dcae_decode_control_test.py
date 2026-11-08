import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_photo_dcae_decode_control as control  # noqa: E402


class PhotoDcaeDecodeAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema": "cogni_ml.hy_nomas_reference_image/v1",
            "schema_version": 1,
            "upstream_revision": control.DCAE_MODULE_REVISION,
            "checkpoint_sha256": control.DCAE_CHECKPOINT_SHA256,
            "encoder_safetensors_sha256": control.ENCODER_SHA256,
            "image_path": str(control.DEFAULT_PHOTO),
            "image_sha256": control.PHOTO_SHA256,
            "payload": "pixels.rgb8",
            "payload_bytes": control.REFERENCE_RGB8_BYTES,
            "payload_sha256": control.PREPARED_RGB8_SHA256,
            "width": 512,
            "height": 288,
            "frames": 120,
            "fit_policy": "exif-rgb-lanczos-pad-black-center",
        }
        self.native = {
            "schema": "cogni_ml.hy_nomas_native_continuation/v1",
            "reference_initialization": True,
            "technical_reference_segment_index": 0,
            "technical_reference_latent_frames": 30,
            "generated_segment_start_index": 1,
            "reference_manifest_sha256": control.REFERENCE_MANIFEST_SHA256,
            "reference_image_sha256": control.PHOTO_SHA256,
            "reference_payload_sha256": control.PREPARED_RGB8_SHA256,
            "reference_encoder_sha256": control.ENCODER_SHA256,
            "diagnostics": {"latent_bank_sha256": control.NATIVE_BANK_SHA256},
        }
        self.dcae = {
            "schema": "cogni_ml.hy_nomas_portrait_dcae_control/v1",
            "dcae_module_history_revision": control.DCAE_MODULE_REVISION,
            "dcae_module_sha256": control.DCAE_MODULE_SHA256,
            "recovered_encoder_sha256": control.ENCODER_SHA256,
            "reference_manifest_sha256": control.REFERENCE_MANIFEST_SHA256,
            "reference_payload_sha256": control.PREPARED_RGB8_SHA256,
            "native_receipt_sha256": control.NATIVE_RECEIPT_SHA256,
            "native_latent_bank_sha256": control.NATIVE_BANK_SHA256,
            "runtime": {"torch": "2.8.0", "device": "mps", "dtype": "BF16",
                        "mps_fallback": False, "autocast": False},
            "input": {"shape": [1, 3, 120, 288, 512], "dtype": "BF16",
                      "temporal_feature_cache": True, "tile_frames": 40},
            "latent": {"shape": [1, 32, 30, 9, 16], "dtype": "BF16",
                       "export_dtype": "F32", "sha256": control.T0_LATENT_SHA256},
            "comparison": {"atol": 2e-5, "rtol": 2e-5,
                           "bitwise_equal": True, "values": 138240,
                           "max_abs": 0.0, "unequal_f32_values": 0},
            "intermediate_readbacks": 0,
            "explicit_mps_synchronize_calls": 0,
            "terminal_transfer_count": 1,
        }

    def test_reference_manifest_binds_original_photo_and_rgb8_extent(self):
        control.validate_photo_manifest(
            self.manifest, control.REFERENCE_MANIFEST_SHA256,
            control.PHOTO_SHA256, control.REFERENCE_RGB8_BYTES, control.PREPARED_RGB8_SHA256,
        )
        with self.assertRaisesRegex(control.ControlError, "photo"):
            control.validate_photo_manifest(
                self.manifest, control.REFERENCE_MANIFEST_SHA256,
                "0" * 64, control.REFERENCE_RGB8_BYTES, control.PREPARED_RGB8_SHA256,
            )
        with self.assertRaisesRegex(control.ControlError, "payload"):
            control.validate_photo_manifest(
                self.manifest, control.REFERENCE_MANIFEST_SHA256,
                control.PHOTO_SHA256, control.REFERENCE_RGB8_BYTES - 1, control.PREPARED_RGB8_SHA256,
            )

    def test_t0_receipt_must_join_native_reference_and_original_photo(self):
        control.validate_t0_lineage(
            self.native, self.dcae, latent_bytes=552960,
            latent_sha256=control.T0_LATENT_SHA256,
            native_receipt_sha256=control.NATIVE_RECEIPT_SHA256,
        )
        bad_native = dict(self.native, reference_initialization=False)
        with self.assertRaisesRegex(control.ControlError, "reference"):
            control.validate_t0_lineage(
                bad_native, self.dcae, latent_bytes=552960,
                latent_sha256=control.T0_LATENT_SHA256,
                native_receipt_sha256=control.NATIVE_RECEIPT_SHA256,
            )
        bad_dcae = dict(self.dcae, reference_manifest_sha256="0" * 64)
        with self.assertRaisesRegex(control.ControlError, "lineage"):
            control.validate_t0_lineage(
                self.native, bad_dcae, latent_bytes=552960,
                latent_sha256=control.T0_LATENT_SHA256,
                native_receipt_sha256=control.NATIVE_RECEIPT_SHA256,
            )
        bad_latent = dict(self.dcae, latent=dict(self.dcae["latent"], sha256="0" * 64))
        with self.assertRaisesRegex(control.ControlError, "latent identity"):
            control.validate_t0_lineage(
                self.native, bad_latent, latent_bytes=552960,
                latent_sha256=control.T0_LATENT_SHA256,
                native_receipt_sha256=control.NATIVE_RECEIPT_SHA256,
            )
        with self.assertRaisesRegex(control.ControlError, "extent"):
            control.validate_t0_lineage(
                self.native, self.dcae, latent_bytes=552959,
                latent_sha256=control.T0_LATENT_SHA256,
                native_receipt_sha256=control.NATIVE_RECEIPT_SHA256,
            )

    def test_decoder_checkpoint_metadata_requires_exact_bf16_schema(self):
        expected = {"decoder.project.weight": [4, 2, 1, 3, 3]}
        valid = {"decoder.project.weight": {
            "shape": [4, 2, 1, 3, 3], "dtype": "BF16", "device": "cpu",
            "contiguous": True,
        }}
        result = control.validate_decoder_metadata(valid, expected)
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["bf16_bytes"], 4 * 2 * 1 * 3 * 3 * 2)
        wrong = {"decoder.project.weight": dict(valid["decoder.project.weight"], shape=[4, 2])}
        with self.assertRaisesRegex(control.ControlError, "shape"):
            control.validate_decoder_metadata(wrong, expected)
        wrong_dtype = {"decoder.project.weight": dict(valid["decoder.project.weight"], dtype="F32")}
        with self.assertRaisesRegex(control.ControlError, "BF16"):
            control.validate_decoder_metadata(wrong_dtype, expected)
        missing = {"decoder.other.weight": valid["decoder.project.weight"]}
        with self.assertRaisesRegex(control.ControlError, "inventory"):
            control.validate_decoder_metadata(missing, expected)


if __name__ == "__main__":
    unittest.main()
