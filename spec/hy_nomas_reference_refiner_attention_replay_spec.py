"""Admission and provenance falsifiers for the GPU attention replay."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_reference_refiner_attention_replay as replay


def operands():
    records = {f"block0.{name}": dict(shape=[1, 1280, 2048], source_dtype="BF16",
                                     sha256="1"*64, offset=0, nbytes=4)
               for name in ("q", "k", "v")}
    return dict(valid_text_tokens=263,
                native_document=dict(qkv_bf16_control=True, tensors=copy.deepcopy(records)),
                source_document=dict(tensors=copy.deepcopy(records)),
                native_payload=b"same", source_payload=b"same")


class AttentionReplayTests(unittest.TestCase):
    def test_exact_operands_admitted_and_recorded(self):
        self.assertEqual(set(replay._validate_operands(operands())),
                         {"block0.q", "block0.k", "block0.v"})

    def test_hash_claim_does_not_replace_direct_byte_identity(self):
        inputs = operands()
        inputs["source_payload"] = b"else"
        with self.assertRaisesRegex(replay.Error, "source-exact bytes"):
            replay._validate_operands(inputs)

    def test_each_operand_shape_dtype_and_digest_is_checked(self):
        for name in ("q", "k", "v"):
            for key, value in (("shape", [1, 263, 2048]), ("source_dtype", "F32"),
                               ("sha256", "2"*64)):
                inputs = operands()
                inputs["source_document"]["tensors"][f"block0.{name}"][key] = value
                with self.subTest(name=name, key=key), self.assertRaises(replay.Error):
                    replay._validate_operands(inputs)

    def test_geometry_and_control_cannot_drift(self):
        inputs = operands()
        inputs["valid_text_tokens"] = 264
        with self.assertRaisesRegex(replay.Error, "V263"):
            replay._validate_operands(inputs)
        for flag in (False, 1, None, "true"):
            inputs = operands()
            inputs["native_document"]["qkv_bf16_control"] = flag
            with self.subTest(flag=flag), self.assertRaisesRegex(replay.Error, "QKV control"):
                replay._validate_operands(inputs)

    def test_vector_probabilities_cannot_be_consumed(self):
        class Query:
            shape = (1, 16, 7, 128)
        # None as torch makes an accidental operator call fail this falsifier.
        with self.assertRaisesRegex(replay.Error, "more than eight"):
            replay._general_math(None, Query(), None, None)


if __name__ == "__main__":
    unittest.main()
