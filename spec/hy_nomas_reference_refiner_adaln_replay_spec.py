"""CPU admission contracts for the MPS-only refiner adaLN replay."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_reference_refiner_adaln_replay as replay
import hy_nomas_reference_refiner_blocks_compare as blocks


class AdaLNReplayContractTests(unittest.TestCase):
    def test_candidate_inventory_separates_input_precision_projection_and_output_rounding(self):
        expected = (
            "native_f32_linear_raw_input",
            "native_f32_linear_bf16_rounded_input",
            "source_f32_linear_bf16_input",
            "native_bf16_linear_rounded_input",
            "source_bf16_linear_source_input",
            "native_f32_linear_raw_input_output_bf16_rounded",
        )
        self.assertEqual(replay.CANDIDATE_NAMES, expected)
        self.assertEqual((replay.ATOL, replay.RTOL), (blocks.ATOL, blocks.RTOL))
        self.assertEqual(replay.INPUT_SHAPE, (1, 2048))
        self.assertEqual(replay.OUTPUT_SHAPE, (1, 4096))
        self.assertEqual(replay.WEIGHT_SHAPE, (4096, 2048))
        self.assertEqual(replay.BIAS_SHAPE, (4096,))

    def test_checkpoint_names_bind_the_nested_upstream_refiner_module(self):
        prefix = "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1."
        self.assertEqual(replay.WEIGHT_NAME, prefix + "weight")
        self.assertEqual(replay.BIAS_NAME, prefix + "bias")

    def test_source_pins_require_complete_file_inventory_and_current_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = {}
            for name in replay.REQUIRED_UPSTREAM_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"pinned:{name}".encode()
                path.write_bytes(payload)
                pins[name] = hashlib.sha256(payload).hexdigest()
            source = dict(upstream_revision=replay.PINNED_REVISION,
                          upstream_source_sha256=pins)
            actual_root, actual_pins = replay._validate_source_file_pins(root, source)
            self.assertEqual(actual_root, root.resolve())
            self.assertEqual(actual_pins, pins)

            changed = root / replay.REQUIRED_UPSTREAM_FILES[0]
            changed.write_bytes(b"changed")
            with self.assertRaisesRegex(replay.Error, "digest mismatch"):
                replay._validate_source_file_pins(root, source)

    def test_source_pins_reject_missing_files_unsafe_names_and_wrong_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = {name: "a" * 64 for name in replay.REQUIRED_UPSTREAM_FILES}
            source = dict(upstream_revision=replay.PINNED_REVISION,
                          upstream_source_sha256=pins)
            with self.assertRaisesRegex(replay.Error, "missing or non-canonical"):
                replay._validate_source_file_pins(root, source)
            bad_pins = dict(pins)
            bad_pins.pop(replay.REQUIRED_UPSTREAM_FILES[0])
            with self.assertRaisesRegex(replay.Error, "inventory"):
                replay._validate_source_file_pins(root, dict(source, upstream_source_sha256=bad_pins))
            with self.assertRaisesRegex(replay.Error, "pinned to the admitted upstream revision"):
                replay._validate_source_file_pins(root, dict(source, upstream_revision="0" * 40))

    def test_global_checkpoint_bytes_must_equal_linked_source_identity(self):
        payload = b"authenticated source BF16 globals"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "globals-bf16.safetensors"
            path.write_bytes(payload)
            source = {"global_shard_sha256": digest}
            self.assertEqual(replay._authenticate_global_weights(path, digest, source), path.resolve())
            with self.assertRaisesRegex(replay.Error, "linked source-BF16"):
                replay._authenticate_global_weights(path, "0" * 64, source)
            path.write_bytes(b"different checkpoint bytes")
            with self.assertRaisesRegex(replay.Error, "bytes differ"):
                replay._authenticate_global_weights(path, digest, source)

    def test_operand_contract_keeps_native_source_silu_difference_observable(self):
        def record(shape, digest, **extra):
            return {"shape": list(shape), "sha256": digest, **extra}

        native = {"tensors": {
            "block0.input": record((1, 1280, 2048), "1" * 64),
            "c": record((1, 2048), "2" * 64),
            replay.INPUT: record(replay.INPUT_SHAPE, "3" * 64),
            replay.OUTPUT: record(replay.OUTPUT_SHAPE, "4" * 64),
        }}
        source = {"tensors": {
            "block0.input": record((1, 1280, 2048), "1" * 64),
            "c": record((1, 2048), "2" * 64),
            replay.INPUT: record(replay.INPUT_SHAPE, "5" * 64,
                                 source_dtype="BF16", source_shape=list(replay.INPUT_SHAPE),
                                 source_checkpoint=replay.INPUT),
            replay.OUTPUT: record(replay.OUTPUT_SHAPE, "6" * 64,
                                  source_dtype="BF16", source_shape=list(replay.OUTPUT_SHAPE),
                                  source_checkpoint=replay.OUTPUT),
        }}
        identity = replay._validate_operand_contract({
            "native_document": native, "source_document": source,
            "valid_text_tokens": 263,
        })
        self.assertFalse(identity["condition_silu_raw_hash_equal"])

        source["tensors"]["c"]["sha256"] = "7" * 64
        with self.assertRaisesRegex(replay.Error, "exact linked source/native input"):
            replay._validate_operand_contract({
                "native_document": native, "source_document": source,
                "valid_text_tokens": 263,
            })

    def test_source_operand_provenance_and_v263_scope_fail_closed(self):
        native = {"tensors": {
            "block0.input": {"sha256": "1" * 64}, "c": {"sha256": "2" * 64},
            replay.INPUT: {"shape": list(replay.INPUT_SHAPE), "sha256": "3" * 64},
            replay.OUTPUT: {"shape": list(replay.OUTPUT_SHAPE), "sha256": "4" * 64},
        }}
        source = {"tensors": {
            "block0.input": {"sha256": "1" * 64}, "c": {"sha256": "2" * 64},
            replay.INPUT: {"shape": list(replay.INPUT_SHAPE), "sha256": "5" * 64,
                           "source_dtype": "F32", "source_shape": list(replay.INPUT_SHAPE),
                           "source_checkpoint": replay.INPUT},
            replay.OUTPUT: {"shape": list(replay.OUTPUT_SHAPE), "sha256": "6" * 64,
                            "source_dtype": "BF16", "source_shape": list(replay.OUTPUT_SHAPE),
                            "source_checkpoint": replay.OUTPUT},
        }}
        inputs = {"native_document": native, "source_document": source,
                  "valid_text_tokens": 263}
        with self.assertRaisesRegex(replay.Error, "observed BF16 condition_silu"):
            replay._validate_operand_contract(inputs)
        source["tensors"][replay.INPUT]["source_dtype"] = "BF16"
        inputs["valid_text_tokens"] = 264
        with self.assertRaisesRegex(replay.Error, "T0 V263"):
            replay._validate_operand_contract(inputs)


if __name__ == "__main__":
    unittest.main()
