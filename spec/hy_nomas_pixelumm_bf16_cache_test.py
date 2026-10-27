"""CPU-only structural checks for PixelUMM BF16 cache serialization."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import platform
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_pixelumm_bf16_cache as cache
import hy_nomas_pixelumm_checkpoint as checkpoint
import hy_nomas_pixelumm_reference as reference


def _toy_cache(directory: Path):
    shapes = {
        f"block{block}.weight{index:02d}": [1]
        for block in (0, 1) for index in range(13)
    }
    payload = directory / "toy.bin"
    records, observations = {}, {}
    aggregate = hashlib.sha256()
    offset = 0
    with payload.open("w+b") as stream:
        for index, key in enumerate(sorted(shapes)):
            tensor = torch.tensor([index + 0.5], dtype=torch.bfloat16)
            record, observed = cache._tensor_record(stream, aggregate, key, tensor, offset)
            records[key], observations[key] = record, observed
            offset += record["nbytes"]
    blocks = {}
    for slot in (0, 1):
        keys = sorted(key for key in shapes if key.startswith(f"block{slot}."))
        blocks[str(slot)] = {
            "selected_block": slot,
            "subset_file": f"block{slot}-f32.safetensors",
            "subset_bytes": 100,
            "subset_sha256": hashlib.sha256(f"subset{slot}".encode()).hexdigest(),
            "source_receipt_file": f"block{slot}-f32.receipt.json",
            "source_receipt_sha256": hashlib.sha256(f"receipt{slot}".encode()).hexdigest(),
            "source_extent": [slot * 52, (slot + 1) * 52],
            "source_f32_bytes": 52,
            "source_range_chunks": [{
                "start": slot * 52 + 8 + checkpoint.HEADER_BYTES,
                "end_inclusive": (slot + 1) * 52 + 8 + checkpoint.HEADER_BYTES - 1,
                "sha256": hashlib.sha256(f"chunk{slot}".encode()).hexdigest(),
            }],
            "weight_keys": keys,
            "bf16_bytes": 26,
        }
    manifest = {
        "schema": cache.SCHEMA,
        "upstream_revision": reference.UPSTREAM_REVISION,
        "upstream_model_path": reference.UPSTREAM_MODEL_PATH,
        "upstream_model_sha256": cache.UPSTREAM_MODEL_SHA256,
        "source_commit": checkpoint.SOURCE_COMMIT,
        "hf_revision": checkpoint.REVISION,
        "hf_source_path": checkpoint.SOURCE_PATH,
        "source_file_bytes": checkpoint.SOURCE_BYTES,
        "checkpoint_header_sha256": checkpoint.HEADER_SHA256,
        "stored_dtype": "F32", "runtime_dtype": "BF16",
        "full_file_hash_verified": False,
        "python_version": platform.python_version(),
        "conversion": {**cache.CONVERSION, "torch_version": torch.__version__},
        "blocks": blocks,
        "weights": records,
        "comparison": {
            "method": "source MPS BF16 tensor raw uint16 bytes equal persisted payload",
            "tensor_count": 26, "all_raw_uint16_equal": True,
            "weights": observations,
        },
        "payload": payload.name, "payload_bytes": offset,
        "payload_sha256": aggregate.hexdigest(),
    }
    return manifest, payload, shapes


class PixelUMMBF16CacheTest(unittest.TestCase):
    def test_cache_records_are_bf16_flat_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload, shapes = _toy_cache(Path(directory))
            result = cache.validate_manifest(manifest, payload, expected_shapes=shapes)
            self.assertEqual(result["tensor_count"], 26)
            self.assertEqual(result["payload_bytes"], 52)
            self.assertTrue(result["all_raw_uint16_equal"])

    def test_payload_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload, shapes = _toy_cache(Path(directory))
            damaged = bytearray(payload.read_bytes())
            damaged[0] ^= 1
            payload.write_bytes(damaged)
            with self.assertRaises(cache.CacheError):
                cache.validate_manifest(manifest, payload, expected_shapes=shapes)

    def test_source_conversion_claim_must_be_true_for_every_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload, shapes = _toy_cache(Path(directory))
            damaged = copy.deepcopy(manifest)
            key = sorted(shapes)[0]
            damaged["comparison"]["weights"][key]["raw_uint16_equal"] = False
            with self.assertRaises(cache.CacheError):
                cache.validate_manifest(damaged, payload, expected_shapes=shapes)

    def test_cache_must_contain_exactly_two_complete_block_inventories(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload, shapes = _toy_cache(Path(directory))
            damaged = copy.deepcopy(manifest)
            del damaged["blocks"]["1"]["source_range_chunks"]
            with self.assertRaises(cache.CacheError):
                cache.validate_manifest(damaged, payload, expected_shapes=shapes)

    def test_cache_runtime_and_rounding_provenance_are_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload, shapes = _toy_cache(Path(directory))
            damaged = copy.deepcopy(manifest)
            damaged["conversion"]["rounding"] = "truncate"
            with self.assertRaises(cache.CacheError):
                cache.validate_manifest(damaged, payload, expected_shapes=shapes)

    def test_well_formed_but_unpinned_upstream_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, payload, shapes = _toy_cache(Path(directory))
            damaged = copy.deepcopy(manifest)
            damaged["upstream_model_sha256"] = "0" * 64
            with self.assertRaises(cache.CacheError):
                cache.validate_manifest(damaged, payload, expected_shapes=shapes)

    def test_tensor_writer_rejects_non_bf16_sources(self):
        from io import BytesIO
        with self.assertRaises(cache.CacheError):
            cache._tensor_record(BytesIO(), hashlib.sha256(), "weight", torch.ones(1), 0)


if __name__ == "__main__":
    unittest.main()
