#!/usr/bin/env python3
"""CPU-only range and source-identity falsifiers for the block 2/3 subsets."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "spec" / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_block23_checkpoint_subset as subset  # noqa: E402


PINNED_HEADER = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-pretrained-chain-20261025/source/pinned-header.json"
)


def target_header(block_index):
    relative = {}
    for stream in ("img", "txt"):
        root = f"double_blocks.{block_index}.{stream}"
        relative[f"{root}_mod.linear.weight"] = [12288, 2048]
        relative[f"{root}_mod.linear.bias"] = [12288]
        for projection in ("q", "k", "v", "proj"):
            relative[f"{root}_attn_{projection}.weight"] = [2048, 2048]
            relative[f"{root}_attn_{projection}.bias"] = [2048]
        for norm in ("q_norm", "k_norm"):
            relative[f"{root}_attn_{norm}.weight"] = [128]
        relative[f"{root}_mlp.fc1.weight"] = [8192, 2048]
        relative[f"{root}_mlp.fc1.bias"] = [8192]
        relative[f"{root}_mlp.fc2.weight"] = [2048, 8192]
        relative[f"{root}_mlp.fc2.bias"] = [2048]

    cursor = subset.BLOCK_DATA_RANGES[block_index][0]
    result = {}
    # This deliberately uses the lexical name order, which differs from the
    # authenticated payload ordering; the parser must sort by data offsets.
    for name in sorted(relative):
        shape = relative[name]
        size = 4
        for dimension in shape:
            size *= dimension
        result[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [cursor, cursor + size]}
        cursor += size
    return result


class Block23CheckpointSubsetTests(unittest.TestCase):
    def plan(self, header, block_index):
        return subset.record_subset_header(json.dumps(header).encode(), block_index)

    def test_pinned_header_derives_exact_disjoint_block_ranges(self):
        self.assertEqual(subset.EXPECTED_HEADER_SHA256,
                         "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200")
        self.assertEqual(subset.BLOCK_DATA_RANGES, {
            2: (8_218_748_416, 8_822_976_000),
            3: (14_865_251_840, 15_469_479_424),
        })
        self.assertEqual(subset.BLOCK_HTTP_RANGES, {
            2: (8_219_001_328, 8_823_228_911),
            3: (14_865_504_752, 15_469_732_335),
        })
        if not PINNED_HEADER.is_file():
            self.skipTest("pinned header is not present on this host")
        raw = PINNED_HEADER.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), subset.EXPECTED_HEADER_SHA256)
        for block_index in (2, 3):
            plan = subset.parse_safetensors_header(raw, subset.base.SOURCE_FILE_SIZE, block_index)
            self.assertEqual(plan.block_bytes, subset.EXPECTED_BLOCK_BYTES)
            self.assertEqual((plan.range_start, plan.range_end), subset.BLOCK_HTTP_RANGES[block_index])
            self.assertEqual(len(plan.tensors), 32)
            self.assertTrue(all(record.dtype == "F32" for record in plan.tensors))
            self.assertEqual({record.name for record in plan.tensors},
                             set(subset.expected_tensors(block_index)))

    def test_each_block_inventory_rebases_in_source_data_order(self):
        for block_index in (2, 3):
            with self.subTest(block_index=block_index):
                value = target_header(block_index)
                self.assertEqual(set(value), set(subset.expected_tensors(block_index)))
                plan = self.plan(value, block_index)
                self.assertEqual(plan.block_bytes, subset.EXPECTED_BLOCK_BYTES)
                encoded = subset.make_subset_header(plan)
                size = struct.unpack("<Q", encoded[:8])[0]
                self.assertEqual(size % 8, 0)
                saved = json.loads(encoded[8:8 + size])
                self.assertEqual(saved["__metadata__"]["hy_nomas_block_index"], str(block_index))
                saved.pop("__metadata__")
                self.assertEqual(set(saved), set(value))
                records = sorted(saved.values(), key=lambda item: item["data_offsets"][0])
                self.assertEqual(records[0]["data_offsets"][0], 0)
                self.assertEqual(records[-1]["data_offsets"][1], subset.EXPECTED_BLOCK_BYTES)
                self.assertTrue(all(a["data_offsets"][1] == b["data_offsets"][0]
                                    for a, b in zip(records, records[1:])))

    def test_wrong_identity_inventory_shape_dtype_gap_and_overlap_reject(self):
        for block_index in (2, 3):
            good = target_header(block_index)
            wrong_index = 3 if block_index == 2 else 2
            wrong = {name.replace(f"double_blocks.{block_index}.",
                                  f"double_blocks.{wrong_index}.", 1): descriptor
                     for name, descriptor in good.items()}
            with self.subTest(block_index=block_index, mutation="wrong identity"), self.assertRaises(
                    subset.SubsetError):
                self.plan(wrong, block_index)

            name = f"double_blocks.{block_index}.img_attn_q.weight"
            for mutation in ("missing", "extra", "shape", "dtype", "gap", "overlap"):
                candidate = target_header(block_index)
                if mutation == "missing":
                    del candidate[name]
                elif mutation == "extra":
                    candidate[f"double_blocks.{block_index}.unexpected"] = candidate[name]
                elif mutation == "shape":
                    candidate[name]["shape"] = [2048, 2049]
                elif mutation == "dtype":
                    candidate[name]["dtype"] = "BF16"
                else:
                    start, end = candidate[name]["data_offsets"]
                    candidate[name]["data_offsets"] = ([start + 4, end + 4]
                                                        if mutation == "gap"
                                                        else [start - 4, end - 4])
                with self.subTest(block_index=block_index, mutation=mutation), self.assertRaises(
                        subset.SubsetError):
                    self.plan(candidate, block_index)

    def test_header_digest_source_size_and_transfer_budget_fail_closed(self):
        synthetic = json.dumps(target_header(2)).encode()
        with mock.patch.object(subset, "EXPECTED_HEADER_SHA256", hashlib.sha256(synthetic).hexdigest()), \
             mock.patch.object(subset.base, "parse_safetensors_header") as full_validator:
            with self.assertRaises(subset.SubsetError):
                subset.parse_safetensors_header(synthetic, subset.base.SOURCE_FILE_SIZE - 1, 2)
            with self.assertRaises(subset.SubsetError):
                subset.parse_safetensors_header(synthetic + b" ", subset.base.SOURCE_FILE_SIZE, 2)
            full_validator.assert_not_called()

        for index in (True, 0, 1, 4, "2"):
            with self.subTest(index=index), self.assertRaises(subset.SubsetError):
                subset.expected_tensors(index)
        for budget in (True, subset.EXPECTED_BLOCK_BYTES - 1,
                       subset.EXPECTED_BLOCK_BYTES + 1):
            with self.subTest(budget=budget), self.assertRaises(subset.SubsetError):
                subset.validate_transfer_budget(budget)

    def test_output_path_confined_to_designated_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(subset, "ALLOWED_OUTPUT_ROOT", root):
                self.assertEqual(subset.output_path(root / "block2.safetensors", 2),
                                 (root / "block2.safetensors").resolve())
                with self.assertRaises(subset.SubsetError):
                    subset.output_path(root.parent / "escape.safetensors", 2)


if __name__ == "__main__":
    unittest.main()
