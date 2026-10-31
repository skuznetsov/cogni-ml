#!/usr/bin/env python3
"""CPU-only admission falsifiers for the bounded Cinematic LoRA cache packer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "examples" / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_cinematic_cache as cache  # noqa: E402


def fake_tensor_records():
    records = []
    for key, (dtype, shape) in cache.expected_lora_key_shapes().items():
        records.append((key, cache.TensorRecord(dtype=dtype, shape=shape, start=0, end=0)))
    return records


def synthetic_base_index():
    return {
        "schema": cache.INDEX_SCHEMA,
        "schema_version": 1,
        "source": {"sha256": cache.BASE_SOURCE_SHA256, "bytes": cache.BASE_SOURCE_BYTES},
        "blocks": [
            {
                "block_index": index,
                "manifest": f"block-{index}-q4k.json",
                "manifest_sha256": f"{index + 1:064x}",
                "payload_sha256": f"{index + 55:064x}",
                "bf16": f"block-{index}-bf16.safetensors",
                "bf16_sha256": f"{index + 109:064x}",
            }
            for index in range(cache.BLOCK_COUNT)
        ],
        "globals": {"bf16": "globals-bf16.safetensors", "sha256": "f" * 64},
    }


def encoded(document):
    return (json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n").encode()


class HyNomasCinematicCacheTests(unittest.TestCase):
    def test_expected_inventory_is_exactly_918_pairs_and_separates_camera_off_skips(self):
        pairs = cache.validate_adapter_inventory(fake_tensor_records())
        self.assertEqual(len(pairs), 918)
        self.assertEqual(sum(pair.active for pair in pairs.values()), 756)
        self.assertEqual(sum(not pair.active for pair in pairs.values()), 162)
        skipped = cache.skipped_action_targets()
        self.assertEqual(len(skipped), 162)
        self.assertEqual(len(set(skipped)), 162)
        self.assertTrue(all(name.startswith("action_modules.") for name in skipped))
        self.assertFalse(any(name.endswith(".lora_A.weight") for name in skipped))

    def test_rejects_duplicate_targets_after_upstream_wrapper_normalization(self):
        records = fake_tensor_records()
        name, record = records[0]
        suffix = ".lora_A.weight" if name.endswith(".lora_A.weight") else ".lora_B.weight"
        stem = name[:-len(suffix)]
        parts = stem.split(".", 2)
        alias = f"{parts[0]}.{parts[1]}._checkpoint_wrapped_module.{parts[2]}{suffix}"
        records.append((alias, record))
        with self.assertRaisesRegex(cache.CinematicCacheError, "duplicate"):
            cache.validate_adapter_inventory(records)

    def test_rejects_missing_and_unexpected_adapter_targets(self):
        missing = fake_tensor_records()
        missing.pop()
        with self.assertRaisesRegex(cache.CinematicCacheError, "missing"):
            cache.validate_adapter_inventory(missing)

        unexpected = fake_tensor_records()
        unexpected.append((
            "double_blocks.54.img_attn_q.lora_A.weight",
            cache.TensorRecord("F32", (256, 2048), 0, 0),
        ))
        with self.assertRaisesRegex(cache.CinematicCacheError, "unexpected"):
            cache.validate_adapter_inventory(unexpected)

    def test_rejects_bad_dtype_and_malformed_rank_or_pair_shapes(self):
        wrong_rank = fake_tensor_records()
        name = next(key for key, _ in wrong_rank if key.endswith(".lora_A.weight"))
        wrong_rank = [
            (key, cache.TensorRecord(record.dtype, (255, *record.shape[1:]), 0, 0)
             if key == name else record)
            for key, record in wrong_rank
        ]
        with self.assertRaisesRegex(cache.CinematicCacheError, "shape"):
            cache.validate_adapter_inventory(wrong_rank)

        wrong_dtype = fake_tensor_records()
        name = wrong_dtype[0][0]
        wrong_dtype[0] = (name, cache.TensorRecord("BF16", wrong_dtype[0][1].shape, 0, 0))
        with self.assertRaisesRegex(cache.CinematicCacheError, "F32"):
            cache.validate_adapter_inventory(wrong_dtype)

    def test_config_admission_binds_hash_bytes_rank_alpha_and_targets(self):
        document = {
            "format_version": 1,
            "rank": 256,
            "alpha": 512.0,
            "dropout": 0.0,
            "targets": list(cache.CONFIG_TARGETS),
        }
        raw = encoded(document)
        digest = hashlib.sha256(raw).hexdigest()
        cache.validate_config_document(document, actual_sha256=digest, actual_bytes=len(raw),
                                       expected_sha256=digest, expected_bytes=len(raw))

        malformed = dict(document, rank=True)
        with self.assertRaisesRegex(cache.CinematicCacheError, "config"):
            cache.validate_config_document(malformed, actual_sha256=digest, actual_bytes=len(raw),
                                           expected_sha256=digest, expected_bytes=len(raw))

        wrong_targets = dict(document, targets=list(document["targets"][:-1]))
        with self.assertRaisesRegex(cache.CinematicCacheError, "target"):
            cache.validate_config_document(wrong_targets, actual_sha256=digest, actual_bytes=len(raw),
                                           expected_sha256=digest, expected_bytes=len(raw))

    def test_index_admission_rejects_wrong_source_or_index_pins(self):
        document = synthetic_base_index()
        raw = encoded(document)
        digest = hashlib.sha256(raw).hexdigest()
        cache.validate_base_index_document(
            document, actual_sha256=digest, expected_index_sha256=digest,
            expected_source_sha256=cache.BASE_SOURCE_SHA256,
            expected_source_bytes=cache.BASE_SOURCE_BYTES,
        )

        with self.assertRaisesRegex(cache.CinematicCacheError, "source"):
            changed_source = synthetic_base_index()
            changed_source["source"]["sha256"] = "0" * 64
            cache.validate_base_index_document(
                changed_source, actual_sha256=hashlib.sha256(encoded(changed_source)).hexdigest(),
                expected_index_sha256=hashlib.sha256(encoded(changed_source)).hexdigest(),
                expected_source_sha256=cache.BASE_SOURCE_SHA256,
                expected_source_bytes=cache.BASE_SOURCE_BYTES,
            )

        with self.assertRaisesRegex(cache.CinematicCacheError, "index SHA"):
            cache.validate_base_index_document(
                document, actual_sha256=digest, expected_index_sha256="0" * 64,
                expected_source_sha256=cache.BASE_SOURCE_SHA256,
                expected_source_bytes=cache.BASE_SOURCE_BYTES,
            )

    def test_index_admission_rejects_duplicate_or_out_of_order_blocks(self):
        for mutation in ("duplicate", "order"):
            with self.subTest(mutation=mutation):
                document = synthetic_base_index()
                if mutation == "duplicate":
                    document["blocks"][1]["block_index"] = 0
                else:
                    document["blocks"][1], document["blocks"][2] = (
                        document["blocks"][2], document["blocks"][1]
                    )
                raw = encoded(document)
                digest = hashlib.sha256(raw).hexdigest()
                with self.assertRaises(cache.CinematicCacheError):
                    cache.validate_base_index_document(
                        document, actual_sha256=digest, expected_index_sha256=digest,
                        expected_source_sha256=cache.BASE_SOURCE_SHA256,
                        expected_source_bytes=cache.BASE_SOURCE_BYTES,
                    )

    def test_output_directory_reservation_never_replaces_existing_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "candidate"
            existing.mkdir()
            sentinel = existing / "keep"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                cache.create_output_directory(existing, allowed_root=root)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_external_source_snapshot_rejects_changed_preflight_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pinned-input"
            path.write_bytes(b"before")
            snapshot = cache._file_snapshot(path, "test input")
            preflight = cache.Preflight(
                source_header=None,
                adapter_header=None,
                adapter_pairs={},
                base_index={},
                block_headers=(),
                globals_header=None,
                q4_manifests=(),
                external_snapshots=((path, snapshot, "test input"),),
            )
            path.write_bytes(b"after-and-longer")
            with self.assertRaisesRegex(cache.CinematicCacheError, "changed"):
                cache._validate_external_source_snapshots(preflight)

    def test_lineage_lists_all_skipped_action_targets_and_candidate_index_pin(self):
        lineage = cache.build_lineage_document(
            index_sha256="a" * 64,
            producer={"device": "mps", "producer": "test"},
        )
        cache.validate_lineage_document(lineage, expected_index_sha256="a" * 64)
        self.assertEqual(lineage["schema"], cache.LINEAGE_SCHEMA)
        self.assertEqual(lineage["index"], {"basename": "index.json", "sha256": "a" * 64})
        self.assertEqual(lineage["merge"]["active_targets"], 756)
        self.assertEqual(lineage["merge"]["skipped_action_targets"], 162)
        self.assertIs(lineage["merge"]["action_conditioning"], False)
        self.assertEqual(lineage["producer"]["skipped_action_targets"], cache.skipped_action_targets())


if __name__ == "__main__":
    unittest.main()
