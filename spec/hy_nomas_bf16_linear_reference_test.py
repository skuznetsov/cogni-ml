"""CPU falsifiers for the physical BF16 linear comparison boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_bf16_linear_reference as oracle


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BF16LinearBoundaryTests(unittest.TestCase):
    def _case(self, directory: str, actual_word: int = 0x3F800000,
              expected_word: int = 0x3F800000):
        root = Path(directory).resolve()
        case_path, native_path = root / "case.json", root / "native.json"
        vec = struct.pack("<I", 0x3F000000) * oracle.INPUT_WIDTH
        expected = struct.pack("<I", expected_word) * oracle.OUTPUT_WIDTH
        data = vec + vec + expected
        records, offset = {}, 0
        for name, width, values in (("input.vec", oracle.INPUT_WIDTH, vec),
                                    ("linear.input", oracle.INPUT_WIDTH, vec),
                                    ("linear.expected", oracle.OUTPUT_WIDTH, expected)):
            records[name] = {"dtype": "F32", "shape": [1, width], "offset": offset,
                             "nbytes": len(values), "sha256": sha(values)}
            offset += len(values)
        payload = root / "case.tensors.bin"
        payload.write_bytes(data)
        case = {
            "schema": oracle.SCHEMA, "upstream_revision": oracle.PINNED_REVISION,
            "operation": oracle.OPERATION, "torch_device": "mps",
            "torch_input_dtype": "bfloat16", "torch_linear_output_dtype": "bfloat16",
            "geometry": {"rows": 1, "input_width": oracle.INPUT_WIDTH,
                         "output_width": oracle.OUTPUT_WIDTH},
            "weight_index_sha256": "1" * 64, "block_weights_sha256": "2" * 64,
            "payload": payload.name, "payload_bytes": len(data),
            "payload_sha256": sha(data), "tensors": records,
        }
        case_bytes = json.dumps(case).encode()
        case_path.write_bytes(case_bytes)
        actual = struct.pack("<I", actual_word) * oracle.OUTPUT_WIDTH
        native_payload = root / "native.tensors.bin"
        native_payload.write_bytes(actual)
        native = {
            "schema": oracle.NATIVE_SCHEMA, "case_manifest": str(case_path),
            "case_manifest_sha256": sha(case_bytes), "case_payload_sha256": sha(data),
            "weight_index_sha256": case["weight_index_sha256"],
            "block_weights_sha256": case["block_weights_sha256"],
            "upstream_revision": oracle.PINNED_REVISION,
            "operation": "resident BF16 img_mod.linear weights+bias; F32 storage input cast to BF16",
            "named_allocations_during_encode": 0, "activation_readbacks_during_encode": 0,
            "terminal_wait_count": 1, "geometry": case["geometry"],
            "variant": "bf16_graph_ops", "payload": native_payload.name,
            "payload_bytes": len(actual), "payload_sha256": sha(actual),
            "tensors": {"linear.output": {"dtype": "F32", "shape": [1, oracle.OUTPUT_WIDTH],
                        "offset": 0, "nbytes": len(actual), "sha256": sha(actual)}},
        }
        native_bytes = json.dumps(native).encode()
        native_path.write_bytes(native_bytes)
        return argparse.Namespace(case=case_path, case_sha256=sha(case_bytes),
                                  native=native_path, native_sha256=sha(native_bytes))

    def test_genuine_bf16_expansion_is_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            result = oracle.compare_capture(self._case(directory))
            self.assertTrue(result["bitwise_equal"])
            self.assertTrue(result["expanded_bf16_storage"])
            self.assertEqual(result["nonzero_low16_elements"], 0)
            self.assertEqual(result["different_elements"], 0)

    def test_eliminated_roundtrip_is_rejected_even_for_tiny_error(self):
        # This one-F32-ULP error would pass the ordinary numerical tolerance.
        # It still falsifies a claim that physical BF16 rounding occurred.
        with tempfile.TemporaryDirectory() as directory:
            result = oracle.compare_capture(self._case(directory, actual_word=0x3F800001))
            self.assertFalse(result["bitwise_equal"])
            self.assertFalse(result["expanded_bf16_storage"])
            self.assertEqual(result["nonzero_low16_elements"], oracle.OUTPUT_WIDTH)
            self.assertEqual(result["first_difference_flat_index"], 0)

    def test_nonfinite_values_cannot_pass_byte_equality(self):
        for word in (0x7F800000, 0xFF800000, 0x7FC00000):
            with self.subTest(word=word), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(oracle.ReferenceError, "finite"):
                    oracle.compare_capture(self._case(directory, word, word))

    def test_expected_non_bf16_values_invalidate_the_case(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(oracle.ReferenceError, "not representable"):
                oracle.compare_capture(self._case(directory, 0x3F800001, 0x3F800001))

    def test_payload_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._case(directory)
            payload = Path(directory) / "native.tensors.bin"
            data = bytearray(payload.read_bytes())
            data[0] ^= 1
            payload.write_bytes(data)
            with self.assertRaisesRegex(oracle.ReferenceError, "digest"):
                oracle.compare_capture(args)


if __name__ == "__main__":
    unittest.main()
