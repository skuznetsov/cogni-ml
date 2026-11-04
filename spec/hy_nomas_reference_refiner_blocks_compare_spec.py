"""CPU admission checks for the GPU-only two-block refiner comparator."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_reference_refiner_blocks_compare as compare


def document():
    records = {}
    offset = 0
    for name in compare.TENSOR_ORDER:
        if name == "block1.input":
            records[name] = {**records["block0.output"], "alias_of": "block0.output"}
            continue
        size = 4 * math.prod(compare.shapes()[name])
        records[name] = dict(dtype="F32", shape=list(compare.shapes()[name]),
                             offset=offset, nbytes=size, sha256="1" * 64)
        offset += size
    return dict(tensor_order=list(compare.TENSOR_ORDER), tensors=records), offset


class BlocksCompareTests(unittest.TestCase):
    def test_block_control_requires_an_explicit_bool_and_limits_coverage(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(compare.Error, "explicit bool"):
                compare._controlled_boundaries({compare.BLOCK_CONTROL_FIELD: value})
        with self.assertRaisesRegex(compare.Error, "explicit bool"):
            compare._controlled_boundaries({})
        self.assertEqual(compare._controlled_boundaries({compare.BLOCK_CONTROL_FIELD: False}), ())
        selected = compare._controlled_boundaries({compare.BLOCK_CONTROL_FIELD: True})
        self.assertEqual(len(selected), 10)
        self.assertIn("block0.norm1", selected)
        self.assertIn("block1.norm1", selected)
        self.assertNotIn("block0.norm2", selected)
        self.assertNotIn("block1.qkv", selected)

    def test_control_flags_are_strict_separate_and_qkv_requires_old_control(self):
        for value in (None, 0, 1, "true", [], {}):
            native = {compare.BLOCK_CONTROL_FIELD: False, compare.QKV_CONTROL_FIELD: value}
            with self.subTest(field=compare.QKV_CONTROL_FIELD, value=value), \
                    self.assertRaisesRegex(compare.Error, "explicit bool"):
                compare._controlled_outputs(native)
            native = {compare.BLOCK_CONTROL_FIELD: value, compare.QKV_CONTROL_FIELD: False}
            with self.subTest(field=compare.BLOCK_CONTROL_FIELD, value=value), \
                    self.assertRaisesRegex(compare.Error, "explicit bool"):
                compare._controlled_outputs(native)
        with self.assertRaisesRegex(compare.Error, "explicit bool"):
            compare._controlled_outputs({compare.BLOCK_CONTROL_FIELD: False})

        self.assertEqual(compare._controlled_outputs({
            compare.BLOCK_CONTROL_FIELD: False, compare.QKV_CONTROL_FIELD: False}), ((), ()))
        boundaries, qkv = compare._controlled_outputs({
            compare.BLOCK_CONTROL_FIELD: True, compare.QKV_CONTROL_FIELD: False})
        self.assertEqual(boundaries, compare.BLOCK_CONTROL_TENSORS)
        self.assertEqual(qkv, ())
        self.assertEqual(len(boundaries), 10)
        self.assertNotIn("block0.qkv", boundaries)
        self.assertNotIn("block1.qkv", boundaries)

        with self.assertRaisesRegex(compare.Error, "requires block-boundary"):
            compare._controlled_outputs({compare.BLOCK_CONTROL_FIELD: False,
                                         compare.QKV_CONTROL_FIELD: True})
        boundaries, qkv = compare._controlled_outputs({
            compare.BLOCK_CONTROL_FIELD: True, compare.QKV_CONTROL_FIELD: True})
        self.assertEqual(boundaries, compare.BLOCK_CONTROL_TENSORS)
        self.assertEqual(qkv, compare.QKV_CONTROL_TENSORS)
        self.assertEqual(qkv, ("block0.qkv", "block0.q", "block0.k", "block0.v",
                               "block1.qkv", "block1.q", "block1.k", "block1.v"))
        self.assertTrue(set(boundaries).isdisjoint(qkv))

    def test_qkv_prerequisite_includes_all_five_native_prelude_controls(self):
        prelude_flags = dict.fromkeys(("prelude_input_bf16_control", "prelude_linear_bf16_control",
                                       "prelude_silu_bf16_control", "prelude_second_linear_bf16_control",
                                       "prelude_add_bf16_control"), True)
        linked = {**prelude_flags, "full_bf16_prelude_parity": False}
        native = {**prelude_flags, compare.BLOCK_CONTROL_FIELD: True,
                  compare.QKV_CONTROL_FIELD: True}
        self.assertEqual(compare._validate_control_prerequisites(native, linked),
                         (compare.BLOCK_CONTROL_TENSORS, compare.QKV_CONTROL_TENSORS))

        missing_prelude_control = {**native, "prelude_add_bf16_control": False}
        with self.assertRaisesRegex(compare.Error, "native block prelude control"):
            compare._validate_control_prerequisites(missing_prelude_control, linked)
        incomplete_linked_prelude = {**linked, "prelude_add_bf16_control": False}
        with self.assertRaisesRegex(compare.Error, "native block prelude control"):
            compare._validate_control_prerequisites(native, incomplete_linked_prelude)

    def test_source_attention_control_is_legacy_default_off_and_geometry_bounded(self):
        base = {compare.BLOCK_CONTROL_FIELD: True, compare.QKV_CONTROL_FIELD: True,
                "valid_text_tokens": 263, "processed_tokens": 1280}
        self.assertFalse(compare._source_attention_control(base))
        self.assertFalse(compare._source_attention_control({**base,
            compare.SOURCE_ATTENTION_CONTROL_FIELD: False}))
        self.assertTrue(compare._source_attention_control({**base,
            compare.SOURCE_ATTENTION_CONTROL_FIELD: True}))

        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(compare.Error, "boolean"):
                compare._source_attention_control({**base,
                    compare.SOURCE_ATTENTION_CONTROL_FIELD: value})

        for field, value, message in (
            (compare.QKV_CONTROL_FIELD, False, "qkv_bf16_control"),
            (compare.BLOCK_CONTROL_FIELD, False, "requires block-boundary"),
            ("valid_text_tokens", 262, "valid_text_tokens"),
            ("processed_tokens", 1279, "processed_tokens"),
        ):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(compare.Error, message):
                compare._source_attention_control({**base, field: value,
                    compare.SOURCE_ATTENTION_CONTROL_FIELD: True})

    def test_source_attention_pin_inventory_is_conditional_and_exact(self):
        root = Path(__file__).resolve().parents[1]

        def pin(names):
            return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}

        base = pin(compare.NATIVE_SOURCE_FILES)
        enabled_names = compare.NATIVE_SOURCE_FILES + compare.SOURCE_ATTENTION_SOURCE_FILES
        enabled = pin(enabled_names)
        compare._validate_pins(base)
        compare._validate_pins(enabled, source_attention_control=True)
        with self.assertRaisesRegex(compare.Error, "inventory"):
            compare._validate_pins(base, source_attention_control=True)
        with self.assertRaisesRegex(compare.Error, "inventory"):
            compare._validate_pins(enabled)

    def test_source_attention_exact_gate_requires_complete_zero_error_valid_rows(self):
        def row(tensor):
            return dict(kind="source_attention_valid_exact", tensor=tensor, region="valid",
                        count=compare.SOURCE_ATTENTION_VALID_COUNT,
                        finite_count=compare.SOURCE_ATTENTION_VALID_COUNT,
                        max_abs=0.0, mae=0.0, rms=0.0, mismatch_count=0,
                        passed=True, atol=0.0, rtol=0.0)

        rows = [row(name) for name in compare.SOURCE_ATTENTION_TENSORS]
        self.assertEqual(compare._source_attention_checks_passed([], False), (None, None))
        self.assertEqual(compare._source_attention_checks_passed(rows, True), (True, True))

        block0_bad = [dict(rows[0], passed=False, max_abs=0.25, mismatch_count=1), rows[1]]
        self.assertEqual(compare._source_attention_checks_passed(block0_bad, True), (False, True))
        # Do not trust a forged `passed` flag if exact metrics show a mismatch.
        forged = [dict(rows[0], max_abs=0.25, mismatch_count=1), rows[1]]
        self.assertEqual(compare._source_attention_checks_passed(forged, True), (False, True))
        forged_zero_type = [dict(rows[0], max_abs=False), rows[1]]
        self.assertEqual(compare._source_attention_checks_passed(forged_zero_type, True), (False, True))
        truncated = [dict(rows[0], count=compare.SOURCE_ATTENTION_VALID_COUNT-1), rows[1]]
        self.assertEqual(compare._source_attention_checks_passed(truncated, True), (False, True))

        for label, malformed in (
            ("missing", rows[:1]),
            ("duplicate", rows + rows[:1]),
            ("reordered", list(reversed(rows))),
            ("wrong region", [dict(rows[0], region="full"), rows[1]]),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(compare.Error, "coverage"):
                compare._source_attention_checks_passed(malformed, True)
        with self.assertRaisesRegex(compare.Error, "coverage"):
            compare._source_attention_checks_passed(rows, False)

    def test_source_attention_zero_padding_gate_requires_both_full_padding_regions(self):
        rows = [dict(kind="source_attention_padding_zero_exact", tensor=name, region="padding",
                     count=compare.SOURCE_ATTENTION_PADDING_COUNT,
                     finite_count=compare.SOURCE_ATTENTION_PADDING_COUNT,
                     max_abs=0.0, mae=0.0, rms=0.0, mismatch_count=0,
                     passed=True, atol=0.0, rtol=0.0)
                for name in compare.SOURCE_ATTENTION_TENSORS]
        self.assertEqual(compare._source_attention_checks_passed(rows, True,
            kind="source_attention_padding_zero_exact", region="padding",
            expected_count=compare.SOURCE_ATTENTION_PADDING_COUNT), (True, True))
        changed = [rows[0], dict(rows[1], count=1)]
        self.assertEqual(compare._source_attention_checks_passed(changed, True,
            kind="source_attention_padding_zero_exact", region="padding",
            expected_count=compare.SOURCE_ATTENTION_PADDING_COUNT), (True, False))
        nonzero = [dict(rows[0], passed=False, max_abs=0.5, mismatch_count=1), rows[1]]
        self.assertEqual(compare._source_attention_checks_passed(nonzero, True,
            kind="source_attention_padding_zero_exact", region="padding",
            expected_count=compare.SOURCE_ATTENTION_PADDING_COUNT), (False, True))
        with self.assertRaisesRegex(compare.Error, "coverage"):
            compare._source_attention_checks_passed([], True,
                kind="source_attention_padding_zero_exact", region="padding",
                expected_count=compare.SOURCE_ATTENTION_PADDING_COUNT)

    def test_unselected_or_missing_representation_checks_cannot_pass_vacuously(self):
        groups = (
            (compare.BLOCK_CONTROL_TENSORS,
             ("native_controlled_bf16_representation", "controlled_boundary_source_exact")),
            (compare.QKV_CONTROL_TENSORS,
             ("native_controlled_qkv_bf16_representation", "controlled_qkv_source_exact")),
        )
        for selected, kinds in groups:
            for kind in kinds:
                with self.subTest(kind=kind, selected=False):
                    self.assertIsNone(compare._controlled_checks_passed([], (), kind))
                rows = [dict(kind=kind, tensor=name, passed=True) for name in selected]
                self.assertTrue(compare._controlled_checks_passed(rows, selected, kind))
                rows[-1]["passed"] = False
                self.assertFalse(compare._controlled_checks_passed(rows, selected, kind))
                for label, broken in (("all omitted", []), ("one omitted", rows[:-1]),
                                      ("duplicate", rows + rows[:1]),
                                      ("reordered", list(reversed(rows)))):
                    with self.subTest(kind=kind, failure=label), \
                            self.assertRaisesRegex(compare.Error, "coverage"):
                        compare._controlled_checks_passed(broken, selected, kind)
                with self.assertRaisesRegex(compare.Error, "coverage"):
                    compare._controlled_checks_passed(rows, (), kind)

    def test_full_padding_and_explicit_interblock_alias(self):
        doc, extent = document()
        self.assertEqual(len(compare.TENSOR_ORDER), 37)
        self.assertEqual(compare.shapes()["block0.input"], (1, 1280, 2048))
        self.assertEqual(compare.shapes()["block0.qkv"], (1, 1280, 3, 2048))
        self.assertEqual(compare._validate_layout(doc), extent)
        for key, value in (("offset", 1), ("sha256", "2" * 64), ("alias_of", "block1.output")):
            bad = copy.deepcopy(doc)
            bad["tensors"]["block1.input"][key] = value
            with self.subTest(key=key), self.assertRaises(compare.Error):
                compare._validate_layout(bad)

    def test_extent_shape_dtype_order_and_extra_alias_fail_closed(self):
        doc, _ = document()
        for key, value in (("offset", True), ("nbytes", 0), ("dtype", "BF16"),
                           ("shape", [1, 263, 2048]), ("alias_of", "c")):
            bad = copy.deepcopy(doc)
            bad["tensors"]["block0.norm1"][key] = value
            with self.subTest(key=key), self.assertRaises(compare.Error):
                compare._validate_layout(bad)
        doc["tensor_order"].reverse()
        with self.assertRaisesRegex(compare.Error, "inventory"):
            compare._validate_layout(doc)

    def test_first_divergence_uses_raw_execution_order_and_region(self):
        rows = [dict(kind="raw", region="full", tensor=name, passed=True)
                for name in compare.TENSOR_ORDER]
        rows.extend([dict(kind="native_bf16_rounding_view", region="full",
                          tensor="c", passed=False)])
        self.assertIsNone(compare.first_divergence(rows, "full"))
        for name in ("block1.output", "block0.norm1", "block0.condition_silu"):
            next(row for row in rows if row["kind"] == "raw" and row["tensor"] == name)["passed"] = False
        self.assertEqual(compare.first_divergence(rows, "full"), "block0.condition_silu")
        self.assertIsNone(compare.first_divergence(rows, "padding"))

    def test_operand_links_require_full_input_and_condition_identity(self):
        native, _ = document()
        source = copy.deepcopy(native)
        compare._validate_exact_inputs(native, source)
        for name in ("block0.input", "c"):
            bad = copy.deepcopy(source)
            bad["tensors"][name]["sha256"] = hashlib.sha256(name.encode()).hexdigest()
            with self.subTest(name=name), self.assertRaisesRegex(compare.Error, "byte-identical"):
                compare._validate_exact_inputs(native, bad)

    def test_strict_execution_flags_cannot_be_integer_attestations(self):
        expected = dict(encode_allocations=0, encode_readbacks=0,
                        preterminal_waits=0, terminal_waits=1)
        compare._require_fields(expected, expected, "test")
        with self.assertRaises(compare.Error):
            compare._require_fields(dict(expected, encode_allocations=False), expected, "test")

    def test_prefix_links_bind_final_output_and_reject_missing_checkpoint(self):
        payload = b"\x00" * 8192
        doc = {"tensors": {"block1.output": {"offset": 0}}}
        linked = {"tensors": {"block.txt_valid_f32": {"sha256": hashlib.sha256(payload).hexdigest()}}}
        links = (("block1.output", "block.txt_valid_f32"),)
        compare._validate_prefix_links(doc, payload, linked, 1, links)
        with self.assertRaisesRegex(compare.Error, "detached"):
            compare._validate_prefix_links(doc, b"\x01"+payload[1:], linked, 1, links)
        with self.assertRaisesRegex(compare.Error, "missing"):
            compare._validate_prefix_links(doc, payload, {"tensors": {}}, 1, links)

    def test_source_contract_rejects_dtype_shape_count_and_identity_drift(self):
        doc, _ = document()
        identity = dict(upstream_revision="a"*40, upstream_source_sha256={"token_refiner.py": "b"*64},
                        conditioning_manifest_sha256="c"*64)
        doc.update(device="mps", producer_dtype="BF16", full_text_refiner_encode_executed=True,
                   full_upstream_t0_preprocessing_parity=False, capture_boundary="after_block1_output",
                   blocks_executed=2, source=identity, aliases={"block1.input": "block0.output"},
                   unique_payload_tensor_order=[name for name in compare.TENSOR_ORDER if name != "block1.input"])
        doc["execution"] = dict(mode="source-bf16", device="mps", valid_text_tokens=263,
            processed_text_rows=1280, padded_text_rows=1017, refiner_block_forward_calls=2,
            refiner_attention_calls=2, observed_attention_calls=2, observed_gate_calls=4,
            dit_block_forward_calls=0, loaded_dit_blocks=0,
            cache_kv_written=False, terminal_cpu_exports=1, unique_payload_tensor_count=36,
            source_dtype_by_tensor=dict.fromkeys(compare.TENSOR_ORDER, "BF16"),
            export_dtype_by_tensor=dict.fromkeys(compare.TENSOR_ORDER, "F32"))
        for name, record in doc["tensors"].items():
            record.update(source_dtype="BF16", source_shape=([1,1280,16,128] if name.endswith((".q", ".k", ".v"))
                          else list(compare.shapes()[name])), source_checkpoint={"block0.input": "input_embed",
                          "block1.input": "block0.output"}.get(name, name))
        linked = {"source": copy.deepcopy(identity)}
        compare._validate_source_contract(doc, linked, 263)
        for target, key, value in (("execution", "terminal_cpu_exports", True),
                                   ("source", "conditioning_manifest_sha256", "d"*64),
                                   ("block0.q", "source_shape", [1,1280,2048]),
                                   ("block0.q", "source_dtype", "F32")):
            bad = copy.deepcopy(doc)
            (bad[target] if target in ("source", "execution") else bad["tensors"][target])[key] = value
            with self.subTest(target=target, key=key), self.assertRaises(compare.Error):
                compare._validate_source_contract(bad, linked, 263)


if __name__ == "__main__":
    unittest.main()
