#!/usr/bin/env python3
"""Offline block-identity and byte-range falsifiers for the second subset."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_block1_checkpoint_subset as subset


def header():
    value = {}
    cursor = 1_572_244_992
    # Independently enumerate the official second block's dimensions.
    for stream in ("img", "txt"):
        relative = {"mod.linear.weight": [12288, 2048], "mod.linear.bias": [12288]}
        for projection in ("q", "k", "v", "proj"):
            relative[f"attn_{projection}.weight"] = [2048, 2048]
            relative[f"attn_{projection}.bias"] = [2048]
        for norm in ("q_norm", "k_norm"):
            relative[f"attn_{norm}.weight"] = [128]
        relative.update({"mlp.fc1.weight": [8192, 2048], "mlp.fc1.bias": [8192],
                         "mlp.fc2.weight": [2048, 8192], "mlp.fc2.bias": [2048]})
        for relative_name, shape in relative.items():
            size = 4
            for dimension in shape:
                size *= dimension
            value[f"double_blocks.1.{stream}_{relative_name}"] = {
                "dtype": "F32", "shape": shape, "data_offsets": [cursor, cursor + size]
            }
            cursor += size
    assert cursor == 2_176_472_576
    return value


class Block1SubsetTests(unittest.TestCase):
    def plan(self, value):
        return subset.record_subset_header(json.dumps(value).encode())

    def test_second_block_exact_inventory_range_and_rebase(self):
        value = header()
        self.assertEqual(set(value), set(subset.EXPECTED_TENSORS))
        plan = self.plan(value)
        self.assertEqual(plan.block_bytes, 604_227_584)
        self.assertEqual((plan.range_start, plan.range_end), (1_572_497_904, 2_176_725_487))
        raw = subset.make_subset_header(plan)
        size = struct.unpack("<Q", raw[:8])[0]
        saved = json.loads(raw[8:8 + size])
        self.assertEqual(size % 8, 0)
        self.assertEqual(saved["__metadata__"]["hy_nomas_block_index"], "1")
        saved.pop("__metadata__")
        self.assertEqual(set(saved), set(value))
        records = sorted(saved.values(), key=lambda item: item["data_offsets"][0])
        self.assertEqual(records[0]["data_offsets"][0], 0)
        self.assertEqual(records[-1]["data_offsets"][1], 604_227_584)
        self.assertTrue(all(a["data_offsets"][1] == b["data_offsets"][0]
                            for a, b in zip(records, records[1:])))

    def test_first_block_substitution_missing_extra_shape_dtype_and_gap_reject(self):
        value = header()
        first = {name.replace("double_blocks.1.", "double_blocks.0."): tensor
                 for name, tensor in value.items()}
        with self.assertRaisesRegex(subset.SubsetError, "inventory"):
            self.plan(first)
        name = "double_blocks.1.img_attn_q.weight"
        for mutation in ("missing", "extra", "shape", "dtype", "gap"):
            candidate = header()
            if mutation == "missing":
                del candidate[name]
            elif mutation == "extra":
                candidate["double_blocks.1.unexpected"] = candidate[name]
            elif mutation == "shape":
                candidate[name]["shape"] = [2048, 2049]
            elif mutation == "dtype":
                candidate[name]["dtype"] = "BF16"
            else:
                candidate[name]["data_offsets"] = [x + 4 for x in candidate[name]["data_offsets"]]
            with self.subTest(mutation=mutation), self.assertRaises(subset.SubsetError):
                self.plan(candidate)

    def test_exact_pinned_header_and_exact_transfer_budget_reject_before_network(self):
        self.assertEqual(subset.EXPECTED_HEADER_SHA256,
                         "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200")
        raw = json.dumps(header()).encode()
        raw += b" " * (subset.EXPECTED_HEADER_BYTES - len(raw))
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), subset.EXPECTED_HEADER_SHA256)
        with self.assertRaisesRegex(subset.SubsetError, "SHA-256"):
            subset.parse_safetensors_header(raw, subset.base.SOURCE_FILE_SIZE)
        for size in (True, 604_227_583, 604_227_585):
            with self.subTest(size=size), self.assertRaisesRegex(subset.SubsetError, "exact"):
                subset.extract("/unused", max_payload_bytes=size)

    def test_serialization_rechecks_mutated_plan(self):
        plan = self.plan(header())
        plan.tensors[0].dtype = "BF16"
        with self.assertRaisesRegex(subset.SubsetError, "descriptor"):
            subset.make_subset_header(plan)


if __name__ == "__main__":
    unittest.main()
