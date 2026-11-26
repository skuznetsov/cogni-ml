"""CPU falsifiers for the physical BF16 patch comparison, using small extents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_bf16_patch_reference as oracle


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BF16PatchComparisonTests(unittest.TestCase):
    def setUp(self):
        # Exercise all descriptors and comparison branches without a full-J
        # fixture. This does not qualify native execution at these dimensions.
        small = patch.multiple(
            oracle, CHANNELS=2, FRAMES=1, HEIGHT=1, WIDTH=2, DIM=4, ROWS=2,
            INPUT_SHAPE=(1, 2, 1, 1, 2), OUTPUT_SHAPE=(1, 2, 4),
            INPUT_ELEMENTS=4, OUTPUT_ELEMENTS=8, INPUT_BYTES=16,
            OUTPUT_BF16_BYTES=16, OUTPUT_F32_BYTES=32,
        )
        small.start()
        self.addCleanup(small.stop)

    def fixture(self, directory, actual_word=0x3F80, expected_word=0x3F80,
                expanded_word=None):
        root = Path(directory).resolve()
        latent = struct.pack("<f", 0.5) * oracle.INPUT_ELEMENTS
        expected = struct.pack("<H", expected_word) * oracle.OUTPUT_ELEMENTS
        case_data = latent + expected
        case_payload = root / "case.tensors.bin"
        case_payload.write_bytes(case_data)
        case = {
            "schema": oracle.SCHEMA, "upstream_revision": oracle.PINNED_REVISION,
            "operation": oracle.OPERATION, "torch_device": "mps",
            "torch_input_dtype": "bfloat16", "torch_output_dtype": "bfloat16",
            "weight_index_sha256": oracle.capture_reference.PINNED_CACHE_INDEX_SHA256,
            "global_weights_sha256": "2" * 64, "capture_manifest_sha256": "3" * 64,
            "capture_latent_sha256": sha(latent),
            "geometry": {"input_shape": list(oracle.INPUT_SHAPE), "rows": 2,
                         "input_width": 2, "output_width": 4,
                         "patch_size": [1, 1, 1], "token_order": "T,H,W"},
            "weights": {"weight_name": "img_in.proj.weight",
                        "weight_shape": [4, 2, 1, 1, 1],
                        "bias_name": "img_in.proj.bias", "bias_shape": [4]},
            "input_nonzero_low16_count": 0, "input_rounding_changed_count": 0,
            "named_buffer_budget_bytes": 1024,
            "payload": case_payload.name, "payload_bytes": len(case_data),
            "payload_sha256": sha(case_data),
            "tensors": {
                "input.latent": self.record("F32", oracle.INPUT_SHAPE, 0, latent),
                "patch.expected_bf16": self.record("BF16", oracle.OUTPUT_SHAPE, len(latent), expected),
            },
        }
        case_path = root / "case.json"
        case_bytes = json.dumps(case).encode()
        case_path.write_bytes(case_bytes)
        actual = struct.pack("<H", actual_word) * oracle.OUTPUT_ELEMENTS
        expanded = struct.pack("<I", actual_word << 16 if expanded_word is None else expanded_word) * oracle.OUTPUT_ELEMENTS
        native_data = latent + actual + expanded
        native_payload = root / "native.tensors.bin"
        native_payload.write_bytes(native_data)
        native = {
            "schema": oracle.NATIVE_SCHEMA, "upstream_revision": oracle.PINNED_REVISION,
            "case_manifest": str(case_path), "case_manifest_sha256": sha(case_bytes),
            "case_payload_sha256": sha(case_data),
            "capture_manifest_sha256": case["capture_manifest_sha256"],
            "weight_index_sha256": case["weight_index_sha256"],
            "global_weights_sha256": case["global_weights_sha256"],
            "operation": oracle.OPERATION, "terminal_wait_count": 1,
            "input_rounding": "explicit Metal BF16 RNE in-place on F32 storage before production channel-to-token transpose",
            "input_nonzero_low16_count": 0, "input_rounding_changed_count": 0,
            "named_allocations_during_encode": 0, "activation_readbacks_during_encode": 0,
            "named_bytes": 128, "variant": "bf16_graph_ops",
            "geometry": {"batches": 1, "frames": 1, "height": 1, "width": 2,
                         "rows": 2, "input_width": 2, "output_width": 4},
            "payload": native_payload.name, "payload_bytes": len(native_data),
            "payload_sha256": sha(native_data),
            "tensors": {
                "patch.input_bf16_f32": self.record("F32", oracle.INPUT_SHAPE, 0, latent),
                "patch.output_bf16": self.record("BF16", oracle.OUTPUT_SHAPE, len(latent), actual),
                "patch.output_f32": self.record("F32", oracle.OUTPUT_SHAPE, len(latent) + len(actual), expanded),
            },
        }
        native_path = root / "native.json"
        native_bytes = json.dumps(native).encode()
        native_path.write_bytes(native_bytes)
        return argparse.Namespace(case=case_path, case_sha256=sha(case_bytes),
                                  native=native_path, native_sha256=sha(native_bytes))

    @staticmethod
    def record(dtype, shape, offset, data):
        return {"dtype": dtype, "shape": list(shape), "offset": offset,
                "nbytes": len(data), "sha256": sha(data)}

    def test_physical_bf16_and_expansion_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = oracle.compare_capture(self.fixture(directory))
            self.assertTrue(result["torch_mps_bf16_vs_native"]["allclose_atol_2e-5_rtol_2e-5"])
            self.assertTrue(result["native_f32_is_exact_bf16_expansion"])

    def test_one_bf16_ulp_cannot_be_hidden_by_the_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            result = oracle.compare_capture(self.fixture(directory, actual_word=0x3F81))
            self.assertFalse(result["torch_mps_bf16_vs_native"]["allclose_atol_2e-5_rtol_2e-5"])
            self.assertEqual(result["torch_mps_bf16_vs_native"]["outside_tolerance"], oracle.OUTPUT_ELEMENTS)

    def test_unrounded_f32_expansion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(oracle.ReferenceError, "exact physical BF16 expansion"):
                oracle.compare_capture(self.fixture(directory, expanded_word=0x3F800001))

    def test_equal_nonfinite_bytes_cannot_pass(self):
        for word in (0x7F80, 0xFF80, 0x7FC0):
            with self.subTest(word=word), tempfile.TemporaryDirectory() as directory:
                result = oracle.compare_capture(self.fixture(directory, word, word))
                self.assertFalse(result["torch_mps_bf16_vs_native"]["allclose_atol_2e-5_rtol_2e-5"])
                self.assertEqual(result["torch_mps_bf16_vs_native"]["actual_nonfinite"], oracle.OUTPUT_ELEMENTS)

    def test_mutated_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            payload = Path(directory) / "native.tensors.bin"
            data = bytearray(payload.read_bytes())
            data[0] ^= 1
            payload.write_bytes(data)
            with self.assertRaisesRegex(oracle.ReferenceError, "digest"):
                oracle.compare_capture(args)


if __name__ == "__main__":
    unittest.main()
