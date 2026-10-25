#!/usr/bin/env python3
"""Offline admission tests for the bounded Hy-Nomas text subset extractor."""

import json
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_text_checkpoint_subset as subset  # noqa: E402


def official_text_tensors():
    result = {
        "time_in.mlp.0.bias": (2048,),
        "time_in.mlp.0.weight": (2048, 256),
        "time_in.mlp.2.bias": (2048,),
        "time_in.mlp.2.weight": (2048, 2048),
        "txt_in.c_embedder.linear_1.bias": (2048,),
        "txt_in.c_embedder.linear_1.weight": (2048, 3584),
        "txt_in.c_embedder.linear_2.bias": (2048,),
        "txt_in.c_embedder.linear_2.weight": (2048, 2048),
        "txt_in.input_embedder.bias": (2048,),
        "txt_in.input_embedder.weight": (2048, 3584),
        "txt_in.t_embedder.mlp.0.bias": (2048,),
        "txt_in.t_embedder.mlp.0.weight": (2048, 256),
        "txt_in.t_embedder.mlp.2.bias": (2048,),
        "txt_in.t_embedder.mlp.2.weight": (2048, 2048),
    }
    for block in range(2):
        root = f"txt_in.individual_token_refiner.blocks.{block}"
        result[f"{root}.adaLN_modulation.1.bias"] = (4096,)
        result[f"{root}.adaLN_modulation.1.weight"] = (4096, 2048)
        result[f"{root}.mlp.fc1.bias"] = (8192,)
        result[f"{root}.mlp.fc1.weight"] = (8192, 2048)
        result[f"{root}.mlp.fc2.bias"] = (2048,)
        result[f"{root}.mlp.fc2.weight"] = (2048, 8192)
        for norm in ("norm1", "norm2"):
            result[f"{root}.{norm}.bias"] = (2048,)
            result[f"{root}.{norm}.weight"] = (2048,)
        result[f"{root}.self_attn_proj.bias"] = (2048,)
        result[f"{root}.self_attn_proj.weight"] = (2048, 2048)
        result[f"{root}.self_attn_qkv.bias"] = (6144,)
        result[f"{root}.self_attn_qkv.weight"] = (6144, 2048)
    return result


def source_order():
    names = [
        "time_in.mlp.0.bias",
        "time_in.mlp.0.weight",
        "time_in.mlp.2.bias",
        "time_in.mlp.2.weight",
        "txt_in.c_embedder.linear_1.bias",
        "txt_in.c_embedder.linear_1.weight",
        "txt_in.c_embedder.linear_2.bias",
        "txt_in.c_embedder.linear_2.weight",
    ]
    for block in range(2):
        root = f"txt_in.individual_token_refiner.blocks.{block}"
        names.extend([
            f"{root}.adaLN_modulation.1.bias",
            f"{root}.adaLN_modulation.1.weight",
            f"{root}.mlp.fc1.bias",
            f"{root}.mlp.fc1.weight",
            f"{root}.mlp.fc2.bias",
            f"{root}.mlp.fc2.weight",
            f"{root}.norm1.bias",
            f"{root}.norm1.weight",
            f"{root}.norm2.bias",
            f"{root}.norm2.weight",
            f"{root}.self_attn_proj.bias",
            f"{root}.self_attn_proj.weight",
            f"{root}.self_attn_qkv.bias",
            f"{root}.self_attn_qkv.weight",
        ])
    names.extend([
        "txt_in.input_embedder.bias",
        "txt_in.input_embedder.weight",
        "txt_in.t_embedder.mlp.0.bias",
        "txt_in.t_embedder.mlp.0.weight",
        "txt_in.t_embedder.mlp.2.bias",
        "txt_in.t_embedder.mlp.2.weight",
    ])
    return names


def synthetic_header(*, omit=None, bad_shape=None, gap=None, bad_dtype=None):
    expected = official_text_tensors()
    cursor = subset.TEXT_DATA_START
    value = {"__metadata__": {"fixture": "offline"}}
    for name in source_order():
        if name == omit:
            continue
        shape = list(expected[name])
        if name == bad_shape:
            shape[-1] += 1
        dtype = "F16" if name == bad_dtype else "F32"
        size = 2 if dtype == "F16" else 4
        for dimension in expected[name]:
            size *= dimension
        start = cursor + (4 if name == gap else 0)
        end = start + size
        value[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, end],
        }
        cursor = end
    if omit is None and bad_shape is None and gap is None and bad_dtype is None:
        if cursor != subset.TEXT_DATA_END:
            raise AssertionError(f"fixture extent mismatch: {cursor}")
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return raw


class HyNomasTextCheckpointSubsetTests(unittest.TestCase):
    def test_exact_official_inventory_and_byte_extent(self):
        self.assertEqual(subset.EXPECTED_TENSORS, official_text_tensors())
        self.assertEqual(len(subset.EXPECTED_TENSORS), 42)
        self.assertEqual(sum(name.startswith("time_in.") for name in subset.EXPECTED_TENSORS), 4)
        self.assertEqual(sum(name.startswith("txt_in.") for name in subset.EXPECTED_TENSORS), 38)
        plan = subset._record_subset_header(synthetic_header())
        self.assertEqual(plan.range_start, subset.EXPECTED_RANGE_START)
        self.assertEqual(plan.range_end, subset.EXPECTED_RANGE_END)
        self.assertEqual(plan.payload_bytes, 583_311_360)
        self.assertEqual({record.name for record in plan.tensors}, set(official_text_tensors()))

    def test_missing_extra_bad_shape_dtype_and_noncontiguous_ranges_fail_closed(self):
        with self.assertRaisesRegex(subset.SubsetError, "inventory mismatch"):
            subset._record_subset_header(synthetic_header(omit="time_in.mlp.0.bias"))

        value = json.loads(synthetic_header())
        value["txt_in.unexpected.weight"] = {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
        }
        with self.assertRaisesRegex(subset.SubsetError, "inventory mismatch"):
            subset._record_subset_header(json.dumps(value).encode())

        with self.assertRaisesRegex(subset.SubsetError, "shape"):
            subset._record_subset_header(
                synthetic_header(bad_shape="txt_in.input_embedder.weight")
            )
        with self.assertRaisesRegex(subset.SubsetError, "dtype or shape"):
            subset._record_subset_header(
                synthetic_header(bad_dtype="txt_in.input_embedder.bias")
            )
        with self.assertRaisesRegex(subset.SubsetError, "extent|contiguous"):
            subset._record_subset_header(
                synthetic_header(gap="txt_in.individual_token_refiner.blocks.0.norm1.bias")
            )

    def test_rebased_header_has_exact_names_f32_shapes_and_contiguous_offsets(self):
        plan = subset._record_subset_header(synthetic_header())
        raw = subset.make_subset_header(plan)
        header_length = int.from_bytes(raw[:8], "little")
        self.assertEqual(header_length % 8, 0)
        header = json.loads(raw[8:8 + header_length])
        tensors = {name: descriptor for name, descriptor in header.items()
                   if name != "__metadata__"}
        self.assertEqual(set(tensors), set(official_text_tensors()))
        records = sorted(
            (descriptor["data_offsets"], descriptor["dtype"], descriptor["shape"])
            for descriptor in tensors.values()
        )
        self.assertTrue(all(dtype == "F32" for _, dtype, _ in records))
        self.assertEqual(records[0][0][0], 0)
        self.assertEqual(records[-1][0][1], subset.EXPECTED_TEXT_BYTES)
        self.assertTrue(all(left[0][1] == right[0][0]
                            for left, right in zip(records, records[1:])))

    def test_source_pin_and_transfer_budget_are_strict(self):
        self.assertEqual(subset.EXPECTED_HEADER_BYTES, 252_904)
        self.assertEqual(
            subset.EXPECTED_HEADER_SHA256,
            "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200",
        )
        self.assertEqual(
            (subset.EXPECTED_RANGE_START, subset.EXPECTED_RANGE_END),
            (33_630_679_920, 34_213_991_279),
        )
        with self.assertRaisesRegex(subset.SubsetError, "below"):
            subset.extract("/unused", max_payload_bytes=subset.EXPECTED_TEXT_BYTES - 1)
        with self.assertRaisesRegex(subset.SubsetError, "exceeds"):
            subset.extract("/unused", max_payload_bytes=subset.EXPECTED_TEXT_BYTES + 1)


if __name__ == "__main__":
    unittest.main()
