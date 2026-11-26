"""Pure admission tests for the final-photo upstream DCAE cross-decoder control."""
import hashlib
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_photo_final_dcae_control as control


class FinalPhotoDCAEAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.t0_channels = [struct.pack("<f", float(channel)) * (30 * 9 * 16)
                            for channel in range(32)]
        self.t1_channels = [struct.pack("<f", float(channel + 64)) * (30 * 9 * 16)
                            for channel in range(32)]
        self.t0 = b"".join(self.t0_channels)
        self.t1 = b"".join(self.t1_channels)
        self.bank = b"".join(t0 + t1 for t0, t1 in zip(self.t0_channels, self.t1_channels))
        self.t0_sha = hashlib.sha256(self.t0).hexdigest()
        self.t1_sha = hashlib.sha256(self.t1).hexdigest()
        self.bank_sha = hashlib.sha256(self.bank).hexdigest()
        self.patches = [
            patch.object(control.photo, "T0_SHA", self.t0_sha),
            patch.object(control.photo, "FINAL_SHA", self.t1_sha),
            patch.object(control, "FINAL_X0_SHA256", self.t1_sha),
            patch.object(control, "FINAL_PAIRED_BANK_SHA256", self.bank_sha),
            patch.object(control, "CHECKPOINT_CONTROL_SHA256", "d" * 64),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def _valid_checkpoint(self, bank_path):
        row = [0.0, 0.0, 0.0, 0.0, 0.0]
        return {
            "schema": "cogni_ml.hy_nomas_photo_dit_checkpoint_control/v1",
            "passed": True,
            "source_capture_sha256": control.photo.DIT_CAPTURE_SHA,
            "atol": control.dcae.ATOL,
            "rtol": control.dcae.RTOL,
            "comparison_order": ["velocity_bf16", "x0_bf16", "after_euler_bf16"],
            "metric_order": ["max_abs", "rmse", "outside_tolerance", "unequal", "nonfinite"],
            "metrics": [[list(row) for _ in range(3)] for _ in range(4)],
            "native_x0_sha256": ["a" * 64, "b" * 64, "c" * 64, self.t1_sha],
            "step4_x0_equals_final_endpoint": True,
            "runtime": {"torch": "2.8.0", "device": "mps",
                        "intermediate_readbacks": 0,
                        "explicit_synchronize_calls": 0, "terminal_transfers": 1},
            "artifacts": [{"name": "native-step4-paired60", "sha256": self.bank_sha,
                           "shape": [1, 32, 60, 9, 16], "path": str(bank_path)}],
        }

    def _valid_pixel_receipt(self, bank_path, pixels_path, pixels_sha):
        base = control.photo.base
        return {
            "schema": "cogni_ml.hy_nomas_decode_latent/v1",
            "upstream_revision": base.UPSTREAM_REVISION,
            "latent_sha256": self.bank_sha,
            "latent_bank_sha256": self.bank_sha,
            "latent_shape": [1, 32, 60, 9, 16],
            "latent_bank_shape": [1, 32, 60, 9, 16],
            "latent_prefix_start_frame": 0,
            "pixel_noise_prefix_start_frame": 0,
            "continuation_manifest_sha256": base.INPUT_MANIFEST_SHA256,
            "pixel_noise_manifest_sha256": base.INPUT_MANIFEST_SHA256,
            "pixel_noise_record_sha256": base.NOISE_SHA256,
            "decoder_weight_cache_sha256": base.CACHE_MANIFEST_SHA256,
            "stem_weights_sha256": base.STEM_SHA256,
            "head_weights_sha256": base.HEAD_SHA256,
            "decoder_rope_cos_record_sha256": base.ROPE_COS_SHA256,
            "decoder_rope_sin_record_sha256": base.ROPE_SIN_SHA256,
            "pixel_noise_shape": list(base.NOISE_PREFIX_SHAPE),
            "pixelumm": {"steps": 1, "window_latents": 10,
                         "overlap_latents": 5, "starts": list(base.WINDOW_STARTS)},
            "output_shape": list(base.PIXELS_SHAPE),
            "output_sha256": pixels_sha,
            "latent": str(bank_path),
            "output": str(pixels_path),
            "metrics": {"decoder_encode_allocations": 0,
                        "decoder_encode_readbacks": 0,
                        "decoder_preterminal_waits": 0,
                        "decoder_waits": 1},
        }

    def test_final_bank_admission_rejects_wrong_hash_context_and_dtype(self):
        self.assertEqual(control.validate_final_bank(self.bank)["second_segment_sha256"], self.t1_sha)
        with self.assertRaisesRegex(control.ControlError, "final paired60 bank SHA"):
            control.validate_final_bank(self.bank[:-4] + b"xxxx")

        clean = b"".join(channel + channel for channel in self.t0_channels)
        with patch.object(control, "FINAL_PAIRED_BANK_SHA256", hashlib.sha256(clean).hexdigest()):
            with self.assertRaises(control.ControlError):
                control.validate_final_bank(clean)
        with self.assertRaisesRegex(control.ControlError, "final paired60 bank SHA"):
            control.validate_final_bank(self.t1)
        for word in (0x7FC00000, 0x3F800001):
            invalid = struct.pack("<I", word) + self.bank[4:]
            with self.subTest(word=word), patch.object(
                    control, "FINAL_PAIRED_BANK_SHA256", hashlib.sha256(invalid).hexdigest()):
                with self.assertRaisesRegex(control.ControlError, "finite BF16"):
                    control.validate_final_bank(invalid)

    def test_checkpoint_receipt_joins_exact_step4_bank_path_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "native-step4-paired60.f32le"
            bank_path.write_bytes(self.bank)
            receipt = self._valid_checkpoint(bank_path)
            admitted = control.validate_checkpoint_control(
                receipt, receipt_sha256="d" * 64, bank_sha256=self.bank_sha,
                bank_path=bank_path,
            )
            self.assertEqual(admitted["path"], str(bank_path))

            with self.assertRaisesRegex(control.ControlError, "checkpoint comparison SHA"):
                control.validate_checkpoint_control(
                    receipt, receipt_sha256="0" * 64, bank_sha256=self.bank_sha,
                    bank_path=bank_path,
                )

            wrong_step = dict(receipt, artifacts=[dict(receipt["artifacts"][0],
                                                        name="native-step3-paired60")])
            with self.assertRaisesRegex(control.ControlError, "paired bank"):
                control.validate_checkpoint_control(
                    wrong_step, receipt_sha256="d" * 64, bank_sha256=self.bank_sha,
                    bank_path=bank_path,
                )

            wrong_path = Path(directory) / "other-paired60.f32le"
            wrong_path.write_bytes(self.bank)
            wrong_receipt = self._valid_checkpoint(wrong_path)
            with self.assertRaisesRegex(control.ControlError, "path differs"):
                control.validate_checkpoint_control(
                    wrong_receipt, receipt_sha256="d" * 64, bank_sha256=self.bank_sha,
                    bank_path=bank_path,
                )

    def test_checkpoint_requires_capture_context_four_steps_and_clean_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "paired60.f32le"
            bank_path.write_bytes(self.bank)
            receipt = self._valid_checkpoint(bank_path)
            missing_context = dict(receipt, source_capture_sha256="0" * 64)
            with self.assertRaisesRegex(control.ControlError, "source/gate"):
                control.validate_checkpoint_control(
                    missing_context, receipt_sha256="d" * 64,
                    bank_sha256=self.bank_sha, bank_path=bank_path,
                )

            wrong_endpoint = dict(receipt, native_x0_sha256=["a" * 64, "b" * 64,
                                                             "c" * 64, "e" * 64])
            with self.assertRaisesRegex(control.ControlError, "named DiT step"):
                control.validate_checkpoint_control(
                    wrong_endpoint, receipt_sha256="d" * 64,
                    bank_sha256=self.bank_sha, bank_path=bank_path,
                )

            missing_step = dict(receipt, native_x0_sha256=receipt["native_x0_sha256"][:3])
            with self.assertRaisesRegex(control.ControlError, "named DiT step"):
                control.validate_checkpoint_control(
                    missing_step, receipt_sha256="d" * 64,
                    bank_sha256=self.bank_sha, bank_path=bank_path,
                )

            for metric in (2, 3, 4):
                with self.subTest(metric=receipt["metric_order"][metric]):
                    bad_metrics = [[list(row) for row in rows]
                                   for rows in receipt["metrics"]]
                    bad_metrics[3][1][metric] = 1.0
                    with self.assertRaisesRegex(control.ControlError, "outside tolerance"):
                        control.validate_checkpoint_control(
                            dict(receipt, metrics=bad_metrics), receipt_sha256="d" * 64,
                            bank_sha256=self.bank_sha, bank_path=bank_path,
                        )

            nonfinite_metrics = [[list(row) for row in rows] for rows in receipt["metrics"]]
            nonfinite_metrics[0][0][0] = math.inf
            with self.assertRaisesRegex(control.ControlError, "non-finite"):
                control.validate_checkpoint_control(
                    dict(receipt, metrics=nonfinite_metrics), receipt_sha256="d" * 64,
                    bank_sha256=self.bank_sha, bank_path=bank_path,
                )

    def test_pixel_receipt_binds_resources_paths_output_and_terminal_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "paired60.f32le"
            bank_path.write_bytes(self.bank)
            pixels_path = Path(directory) / "native-pixels.f32le"
            pixels_path.write_bytes(b"pixels")
            pixels_sha = hashlib.sha256(b"pixels").hexdigest()
            receipt = self._valid_pixel_receipt(bank_path, pixels_path, pixels_sha)
            validated = control.validate_pixelumm_receipt(
                receipt, bank_sha256=self.bank_sha, bank_path=bank_path,
                pixels_sha256=pixels_sha, pixels_path=pixels_path,
                pixels_bytes=math.prod(control.photo.base.PIXELS_SHAPE) * 4,
            )
            self.assertEqual(validated["sha256"], pixels_sha)

            wrong_resource = dict(receipt, stem_weights_sha256="0" * 64)
            with self.assertRaisesRegex(control.ControlError, "bank/noise/positions/weights"):
                control.validate_pixelumm_receipt(
                    wrong_resource, bank_sha256=self.bank_sha,
                    bank_path=bank_path, pixels_sha256=pixels_sha,
                    pixels_path=pixels_path,
                    pixels_bytes=math.prod(control.photo.base.PIXELS_SHAPE) * 4,
                )

            wrong_terminal = dict(receipt, metrics=dict(receipt["metrics"],
                                                        decoder_preterminal_waits=1))
            with self.assertRaisesRegex(control.ControlError, "preterminal GPU boundary"):
                control.validate_pixelumm_receipt(
                    wrong_terminal, bank_sha256=self.bank_sha,
                    bank_path=bank_path, pixels_sha256=pixels_sha,
                    pixels_path=pixels_path,
                    pixels_bytes=math.prod(control.photo.base.PIXELS_SHAPE) * 4,
                )

            with self.assertRaisesRegex(control.ControlError, "extent"):
                control.validate_pixelumm_receipt(
                    receipt, bank_sha256=self.bank_sha, bank_path=bank_path,
                    pixels_sha256=pixels_sha, pixels_path=pixels_path, pixels_bytes=6,
                )

    def test_final_memory_inventory_keeps_external_guard_and_opaque_boundary(self):
        estimate = control.final_memory_estimate({
            "external_cap_bytes": 18 * control.GIB,
            "physical_reserve_fraction": 0.10,
            "external_guard_required": True,
        })
        self.assertEqual(estimate["external_cap_bytes"], 18 * control.GIB)
        self.assertEqual(estimate["physical_reserve_fraction"], 0.10)
        self.assertFalse(estimate["opaque_mps_workspace_included"])
        inventory = estimate["named_inventory"]
        self.assertEqual(inventory["host_pixelumm_final120_f32_selected_crop_bytes"],
                         120 * 3 * 288 * 512 * 4)
        self.assertEqual(inventory["host_pixelumm_full240_f32_read_buffer_bytes"],
                         240 * 3 * 288 * 512 * 4)
        self.assertEqual(inventory["device_prefix120_rgb8_staging_bytes"],
                         120 * 288 * 512 * 3)
        self.assertEqual(inventory["device_combined_terminal_pack_bytes"],
                         2 * 120 * 288 * 512 * 3 + 16)
        self.assertEqual(inventory["device_final120_f32_cross_decoder_delta_bytes"],
                         120 * 3 * 288 * 512 * 4)
        self.assertGreaterEqual(estimate["named_bytes"], sum(inventory.values()))


if __name__ == "__main__":
    unittest.main()
