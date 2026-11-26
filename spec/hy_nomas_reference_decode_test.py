import sys
import tempfile
import unittest
from pathlib import Path


SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_reference_decode as reference_decode


def valid_receipt():
    digest_a = "a" * 64
    digest_b = "b" * 64
    digest_c = "c" * 64
    digest_d = "d" * 64
    return {
        "schema": reference_decode.NATIVE_SCHEMA,
        "operation": reference_decode.NATIVE_OPERATION,
        # Native PixelUMM code is pinned separately from the DCAE source run.
        "upstream_revision": "911cea6458a89c2663513ff7099561ec190dd491",
        "precision": reference_decode.NATIVE_PRECISION,
        "resolution": "512x288",
        "width": 512,
        "height": 288,
        "frames": 120,
        "fps": 24,
        "source_latent": {
            "path": "/tmp/native-full.safetensors",
            "sha256": reference_decode.DCAE_LATENT_FILE_SHA256,
            "tensor_key": "latent",
            "dtype": "BF16",
            "shape": [1, 32, 30, 9, 16],
            "nbytes": 276480,
        },
        "dcae_lineage": {
            "upstream_reference": {"path": "/tmp/upstream-full.json", "sha256": digest_a},
            "upstream_artifact": {"path": "/tmp/upstream-full.safetensors", "sha256": reference_decode.DCAE_UPSTREAM_ARTIFACT_SHA256},
            "native_receipt": {"path": "/tmp/native-full-receipt.json", "sha256": digest_b},
            "comparison": {"path": "/tmp/native-full-comparison.json", "sha256": digest_c},
            "bitwise_comparison": {"path": "/tmp/native-full-bitwise.json", "sha256": digest_d},
        },
        "decoder_inputs": {
            "manifest_path": "/tmp/inputs.json",
            "manifest_sha256": digest_a,
            "payload_sha256": digest_b,
            "prompt": "a reference scene",
            "noise": {
                "key": "decoder.noise",
                "dtype": "F32",
                "shape": [1, 3, 120, 288, 512],
                "nbytes": 212336640,
                "sha256": digest_c,
            },
        },
        "weights": {
            "decoder_cache_manifest_sha256": digest_a,
            "decoder_cache_payload_sha256": digest_b,
            "stem_subset_sha256": digest_c,
            "head_subset_sha256": digest_d,
        },
        "outputs": {
            "latent_f32": {
                "path": "/tmp/latent.f32le",
                "dtype": "F32LE",
                "shape": [1, 32, 30, 9, 16],
                "nbytes": 552960,
                "sha256": digest_a,
            },
            "raw_pixels": {
                "path": "/tmp/raw.f32le",
                "dtype": "F32LE",
                "shape": [1, 3, 120, 288, 512],
                "nbytes": 212336640,
                "sha256": digest_b,
            },
            "display_pixels": {
                "path": "/tmp/display.f32le",
                "dtype": "F32LE",
                "shape": [1, 3, 120, 288, 512],
                "nbytes": 212336640,
                "sha256": digest_c,
            },
        },
        "pixelumm": {
            "window_latents": 10,
            "overlap_latents": 5,
            "starts": [0, 5, 10, 15, 20],
            "prediction_count_per_window": 1,
        },
        "execution": {
            "device": "mps",
            "terminal_drain_confirmed": True,
            "exports_after_terminal_drain": True,
            "timings": {"decoder_wall_seconds": 1.0},
            "decoder_named_bytes": 1000,
            "saved_latent_source_bytes": 276480,
            "aggregate_named_bytes": 277480,
            "opaque_mps_private_allocation_visibility": "not observable through MetalBuffer instrumentation",
            "metrics": {
                "decoder_encode_allocations": 0,
                "decoder_encode_readbacks": 0,
                "decoder_preterminal_waits": 0,
                "decoder_waits": 1,
            },
        },
    }


class ReferenceDecodeReceiptTests(unittest.TestCase):
    def test_accepts_decoder_only_receipt_without_dit_fields(self):
        receipt = valid_receipt()
        self.assertEqual(
            reference_decode.PIXELUMM_UPSTREAM_REVISION,
            "911cea6458a89c2663513ff7099561ec190dd491",
        )
        self.assertEqual(
            reference_decode.DCAE_REFERENCE_REVISION,
            "a31c0fe0bcc493be188fd37ddba4ce2f674becb4",
        )
        self.assertNotEqual(
            reference_decode.PIXELUMM_UPSTREAM_REVISION,
            reference_decode.DCAE_REFERENCE_REVISION,
        )
        diagnostics = reference_decode.validate_native_receipt_document(
            receipt,
            expected_inputs_sha256=receipt["decoder_inputs"]["manifest_sha256"],
        )
        self.assertEqual(diagnostics["latent"], receipt["outputs"]["latent_f32"]["path"])
        self.assertEqual(diagnostics["raw_pixels_sha256"], receipt["outputs"]["raw_pixels"]["sha256"])
        self.assertNotIn("seed", receipt)
        self.assertNotIn("steps", receipt)

    def test_rejects_bad_expected_receipt_hashes(self):
        receipt = valid_receipt()
        receipt["source_latent"]["sha256"] = "not-a-hash"
        with self.assertRaisesRegex(reference_decode.ReferenceDecodeError, "SHA-256"):
            reference_decode.validate_native_receipt_document(
                receipt,
                expected_inputs_sha256=receipt["decoder_inputs"]["manifest_sha256"],
            )

    def test_rejects_unknown_receipt_schema(self):
        receipt = valid_receipt()
        receipt["schema"] = "cogni_ml.hy_nomas_native_preview/v1"
        with self.assertRaisesRegex(reference_decode.ReferenceDecodeError, "schema"):
            reference_decode.validate_native_receipt_document(
                receipt,
                expected_inputs_sha256=receipt["decoder_inputs"]["manifest_sha256"],
            )

    def test_rejects_non_default_latent_geometry(self):
        receipt = valid_receipt()
        receipt["source_latent"]["shape"] = [1, 32, 30, 13, 24]
        with self.assertRaisesRegex(reference_decode.ReferenceDecodeError, "shape"):
            reference_decode.validate_native_receipt_document(
                receipt,
                expected_inputs_sha256=receipt["decoder_inputs"]["manifest_sha256"],
            )

    def test_rejects_missing_decoder_noise_binding(self):
        receipt = valid_receipt()
        del receipt["decoder_inputs"]["noise"]
        with self.assertRaisesRegex(reference_decode.ReferenceDecodeError, "noise"):
            reference_decode.validate_native_receipt_document(
                receipt,
                expected_inputs_sha256=receipt["decoder_inputs"]["manifest_sha256"],
            )

    def test_rejects_no_terminal_drain_or_hidden_encode_activity(self):
        receipt = valid_receipt()
        receipt["execution"]["metrics"]["decoder_preterminal_waits"] = 1
        with self.assertRaisesRegex(reference_decode.ReferenceDecodeError, "preterminal"):
            reference_decode.validate_native_receipt_document(
                receipt,
                expected_inputs_sha256=receipt["decoder_inputs"]["manifest_sha256"],
            )

    def test_rejects_inconsistent_named_memory_receipt(self):
        receipt = valid_receipt()
        receipt["execution"]["aggregate_named_bytes"] += 1
        with self.assertRaisesRegex(reference_decode.ReferenceDecodeError, "aggregate_named_bytes"):
            reference_decode.validate_native_receipt_document(
                receipt,
                expected_inputs_sha256=receipt["decoder_inputs"]["manifest_sha256"],
            )

    def test_requires_exact_saved_bf16_to_f32_latent_expansion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "latent.bf16"
            output = root / "latent.f32le"
            source_bytes = bytes((0x80, 0x3F)) * (276480 // 2)
            source.write_bytes(source_bytes)
            output.write_bytes(bytes((0, 0, 0x80, 0x3F)) * (276480 // 2))
            source_info = {"path": str(source), "data_offset": 0, "nbytes": len(source_bytes)}
            reference_decode._verify_exact_bf16_to_f32_expansion(source_info, output)
            corrupted = bytearray(output.read_bytes())
            corrupted[-1] = 1
            output.write_bytes(corrupted)
            with self.assertRaisesRegex(reference_decode.ReferenceDecodeError, "exact GPU BF16-word expansion"):
                reference_decode._verify_exact_bf16_to_f32_expansion(source_info, output)


if __name__ == "__main__":
    unittest.main()
