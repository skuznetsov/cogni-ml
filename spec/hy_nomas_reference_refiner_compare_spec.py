"""CPU checks for the refiner trace comparator and its BF16 mean falsifier."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_reference_refiner_compare as compare


class RefinerAdmissionTests(unittest.TestCase):
    def test_connected_prelude_controls_require_boolean_dependencies(self):
        defaults = dict(
            prelude_input_bf16_control=False,
            prelude_linear_bf16_control=False,
            prelude_silu_bf16_control=False,
            prelude_second_linear_bf16_control=False,
            prelude_add_bf16_control=False)
        document = dict(defaults, full_bf16_prelude_parity=False)
        self.assertEqual(compare._validate_prelude_controls(document), defaults)
        enabled = dict(prelude_input_bf16_control=True,
                       prelude_linear_bf16_control=True,
                       prelude_silu_bf16_control=True,
                       prelude_second_linear_bf16_control=True,
                       prelude_add_bf16_control=True)
        enabled_document = dict(enabled, full_bf16_prelude_parity=False)
        self.assertEqual(compare._validate_prelude_controls(enabled_document), enabled)
        with self.assertRaisesRegex(compare.Error, "boolean"):
            compare._validate_prelude_controls({})
        for name in enabled:
            with self.subTest(control=name), self.assertRaisesRegex(compare.Error, "boolean"):
                compare._validate_prelude_controls({**enabled_document, name: 1})
            missing = dict(enabled_document)
            missing.pop(name)
            with self.subTest(missing=name), self.assertRaisesRegex(compare.Error, "boolean"):
                compare._validate_prelude_controls(missing)
        for invalid in [dict(prelude_linear_bf16_control=True),
                        dict(prelude_silu_bf16_control=True),
                        dict(prelude_input_bf16_control=True, prelude_silu_bf16_control=True),
                        dict(prelude_second_linear_bf16_control=True),
                        dict(prelude_add_bf16_control=True),
                        {**enabled, "prelude_silu_bf16_control": False},
                        {**enabled, "prelude_second_linear_bf16_control": False}]:
            with self.subTest(controls=invalid), self.assertRaisesRegex(compare.Error, "requires"):
                compare._validate_prelude_controls({**document, **invalid})

    def test_connected_controls_reject_missing_or_promoted_full_parity(self):
        flags = dict.fromkeys(("prelude_input_bf16_control", "prelude_linear_bf16_control",
                               "prelude_silu_bf16_control", "prelude_second_linear_bf16_control",
                               "prelude_add_bf16_control"), False)
        for value in (True, 0, None, "false"):
            with self.subTest(value=value), self.assertRaisesRegex(compare.Error, "full BF16 prelude parity"):
                compare._validate_prelude_controls(dict(flags, full_bf16_prelude_parity=value))
        with self.assertRaisesRegex(compare.Error, "full BF16 prelude parity"):
            compare._validate_prelude_controls(flags)

    def test_controlled_boundary_inventory_is_cumulative_and_excludes_refiner_blocks(self):
        flags = ("prelude_input_bf16_control", "prelude_linear_bf16_control",
                 "prelude_silu_bf16_control", "prelude_second_linear_bf16_control",
                 "prelude_add_bf16_control")
        boundaries = ("masked_mean", "refiner_timestep_sinusoidal", "time_in.sinusoidal",
                      "refiner_timestep_mlp1", "context_linear1", "input_embed", "time_in.mlp1",
                      "refiner_timestep_silu", "context_silu", "time_in.silu",
                      "refiner_timestep", "context_embed", "vec", "c")
        for enabled_count, expected_count in enumerate((0, 3, 7, 10, 13, 14)):
            document = {**dict.fromkeys(flags, False), **dict.fromkeys(flags[:enabled_count], True),
                        "full_bf16_prelude_parity": False}
            controls = compare._validate_prelude_controls(document)
            names = compare._controlled_boundaries(controls)
            self.assertEqual(names, boundaries[:expected_count])
            self.assertEqual(len(names), expected_count)
            self.assertEqual(len(set(names)), expected_count)
            self.assertFalse(any(name.startswith("block") for name in names))
            self.assertEqual("c" in names, enabled_count == 5)
            for name in ("refiner_timestep", "context_embed", "vec"):
                self.assertEqual(name in names, enabled_count >= 4)

    def test_compiled_pins_cover_linear_kernel_and_activation_bridge(self):
        self.assertIn("src/ml/metal/kernels/shared/linear_mps.metal", compare.NATIVE_SOURCE_FILES)
        self.assertIn("src/ml/video/hy_nomas_mps_reductions_bridge.mm", compare.NATIVE_SOURCE_FILES)

    def test_native_trace_and_comparator_pin_inventories_match(self):
        root = Path(compare.__file__).resolve().parents[2]
        text = (root/"examples/support/hy_nomas_reference_refiner_trace.cr").read_text()
        inventory = text.split("SOURCE_FILES = [", 1)[1].split("\n  ]", 1)[0]
        self.assertEqual(tuple(re.findall(r'^    "([^"]+)",$', inventory, re.MULTILINE)),
                         compare.NATIVE_SOURCE_FILES)

    def test_manifest_schema_hash_and_duplicate_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            raw = b'{"schema":"wrong","schema_version":1}'
            path.write_bytes(raw)
            with self.assertRaisesRegex(compare.Error, "schema"):
                compare._load_document(path, hashlib.sha256(raw).hexdigest(), compare.NATIVE_SCHEMA)
            with self.assertRaisesRegex(compare.Error, "SHA-256"):
                compare._load_document(path, "0"*64, compare.NATIVE_SCHEMA)
            raw = b'{"schema":"one","schema":"two"}'
            path.write_bytes(raw)
            with self.assertRaisesRegex(compare.Error, "duplicate"):
                compare._load_document(path, hashlib.sha256(raw).hexdigest(), compare.NATIVE_SCHEMA)

    def test_tensor_order_and_shape_are_checked_before_payload_access(self):
        document = dict(tensor_order=list(reversed(compare.COMMON_ORDER)), tensors={})
        with self.assertRaisesRegex(compare.Error, "inventory"):
            compare._validate_payload(Path("/missing/trace.json"), document, native=True)
        with self.assertRaises(compare.Error):
            compare.shapes(True)
        with self.assertRaises(compare.Error):
            compare.shapes(0)

    def test_mean_replay_rejects_tolerated_operand_or_mask_differences(self):
        valid = 263
        prefix = b"\x00\x00\x80\x3f"*valid + b"\0"*(4*(1280-valid))
        native = dict(valid_text_tokens=valid, tensors={"input_compact_valid": {"sha256": "1"*64}})
        source = dict(tensors={"input_compact_valid": {"sha256": "1"*64},
                              "refiner_mask": {"sha256": hashlib.sha256(prefix).hexdigest()}})
        compare._validate_exact_mean_operands(native, source)
        source["tensors"]["input_compact_valid"]["sha256"] = "2"*64
        with self.assertRaisesRegex(compare.Error, "byte-identical"):
            compare._validate_exact_mean_operands(native, source)
        source["tensors"]["input_compact_valid"]["sha256"] = "1"*64
        source["tensors"]["refiner_mask"]["sha256"] = hashlib.sha256(prefix[::-1]).hexdigest()
        with self.assertRaisesRegex(compare.Error, "actual source refiner mask"):
            compare._validate_exact_mean_operands(native, source)

    def test_compiled_pins_require_complete_current_source_identity(self):
        root = Path(compare.__file__).resolve().parents[2]
        pins = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in compare.NATIVE_SOURCE_FILES}
        compare._validate_compiled_pins(pins)
        pins[compare.NATIVE_SOURCE_FILES[0]] = "0"*64
        with self.assertRaisesRegex(compare.Error, "reviewed working tree"):
            compare._validate_compiled_pins(pins)
        pins.pop(compare.NATIVE_SOURCE_FILES[0])
        with self.assertRaises(compare.Error):
            compare._validate_compiled_pins(pins)

    def test_native_payload_is_authenticated_and_sinusoid_alias_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            records = {}
            blocks = []
            offset = 0
            for name in compare.COMMON_ORDER:
                shape = compare.shapes(1)[name]
                block = b"\0" * (4 * math.prod(shape))
                records[name] = dict(dtype="F32", shape=list(shape), offset=offset,
                                     nbytes=len(block), sha256=hashlib.sha256(block).hexdigest())
                blocks.append(block)
                offset += len(block)
            records["time_in.sinusoidal"]["alias_of"] = "refiner_timestep_sinusoidal"
            payload = b"".join(blocks)
            (path.parent / "trace.tensors.bin").write_bytes(payload)
            document = dict(tensor_order=list(compare.COMMON_ORDER), tensors=records,
                            valid_text_tokens=1,
                            tensor_artifact=dict(basename="trace.tensors.bin", bytes=len(payload),
                                                 sha256=hashlib.sha256(payload).hexdigest()))
            self.assertEqual(compare._validate_payload(path, document, native=True), payload)
            records["time_in.sinusoidal"]["alias_of"] = "c"
            with self.assertRaisesRegex(compare.Error, "alias"):
                compare._validate_payload(path, document, native=True)
            records["time_in.sinusoidal"]["alias_of"] = "refiner_timestep_sinusoidal"
            (path.parent / "trace.tensors.bin").write_bytes(payload[:-1] + b"x")
            with self.assertRaisesRegex(compare.Error, "SHA-256"):
                compare._validate_payload(path, document, native=True)


class MeanReplayTests(unittest.TestCase):
    def test_bf16_mask_count_263_and_f32_count_are_distinguishable(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        compact = ((torch.arange(263, dtype=torch.float32) % 7) - 3).reshape(1, 263, 1)
        replay = compare._mean_replay(torch, compact, 263)
        self.assertEqual(replay["denominator"].item(), 264)
        source = replay["source_bf16_expression"].float()
        other = replay["round_sum_divide_f32_count"]
        self.assertTrue(torch.all(torch.abs(other-source) > compare.ATOL + compare.RTOL*torch.abs(source)))


if __name__ == "__main__":
    unittest.main()
