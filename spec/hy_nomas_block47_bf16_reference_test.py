#!/usr/bin/env python3
"""CPU-only admission tests for the block 4--7 MPS BF16 conversion oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "examples" / "support"
SPEC_SUPPORT = ROOT / "spec" / "support"
sys.path.insert(0, str(SUPPORT))
sys.path.insert(0, str(SPEC_SUPPORT))

import hy_nomas_block47_bf16_reference as reference  # noqa: E402


def _source_document(block_index: int, source_path: Path, source_digest: str) -> dict:
    subset = reference.subset
    names = sorted(reference.block_tensor_shapes(block_index))
    payload_bytes = reference.block_payload_bytes(block_index)
    return {
        "format": "hy-nomas-block-subset-provenance-v1",
        "source": {
            "repository": subset.base.HF_REPO,
            "revision": subset.base.HF_REVISION,
            "checkpoint_path": subset.base.CHECKPOINT_PATH,
            "checkpoint_size_bytes": subset.base.SOURCE_FILE_SIZE,
            "checkpoint_lfs_sha256": subset.base.SOURCE_LFS_SHA256,
            "checkpoint_lfs_sha256_status": (
                "declared by pinned HF API; whole file not downloaded or verified"
            ),
            "source_git_commit": subset.base.SOURCE_GIT_COMMIT,
            "header_sha256_observed": subset.EXPECTED_HEADER_SHA256,
            "range_start_inclusive": subset.BLOCK_HTTP_RANGES[block_index][0],
            "range_end_inclusive": subset.BLOCK_HTTP_RANGES[block_index][1],
            "authentication_precondition": "HTTPS/TLS to pinned HF revision and its HTTPS range redirects",
            "payload_sha256_observed": "a" * 64,
        },
        "subset": {
            "block_index": block_index,
            "tensor_prefix": f"double_blocks.{block_index}.",
            "tensor_count": 32,
            "tensor_names": names,
            "dtype": "F32",
            "stored_bytes_preserved": True,
            "conversion_performed": False,
            "payload_bytes": payload_bytes,
        },
        "output": {
            "filename": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": source_digest,
        },
    }


def _tiny_source_file(path: Path, block_index: int) -> tuple[dict, bytes]:
    shapes = reference.block_tensor_shapes(block_index)
    metadata = reference.source_header_metadata(block_index)
    tensors = {}
    payload = bytearray()
    offsets = {}
    for name in sorted(shapes):
        shape = shapes[name]
        nbytes = reference.shape_product(shape) * 4
        start = len(payload)
        payload.extend(bytes([len(tensors) + 1]) * nbytes)
        offsets[name] = (start, start + nbytes)
        tensors[name] = {"dtype": "F32", "shape": shape,
                         "data_offsets": [start, start + nbytes]}
    tensors["__metadata__"] = metadata
    header = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    encoded = struct.pack("<Q", len(header)) + header + bytes(payload)
    path.write_bytes(encoded)
    document = _source_document(block_index, path, hashlib.sha256(encoded).hexdigest())
    document["source"]["payload_sha256_observed"] = hashlib.sha256(payload).hexdigest()
    return document, encoded


class HyNomasBlock47Bf16ReferenceTests(unittest.TestCase):
    def test_all_four_exact_inventories_and_bounded_indices(self):
        for index in range(4, 8):
            shapes = reference.block_tensor_shapes(index)
            self.assertEqual(len(shapes), 32)
            self.assertTrue(all(n.startswith(f"double_blocks.{index}.") for n in shapes))
            self.assertEqual(reference.block_payload_bytes(index), 604_227_584)
            header, records = reference.build_bf16_header(index)
            self.assertEqual(sum(r["nbytes"] for r in records.values()), 302_113_792)
            self.assertEqual(set(records), set(shapes))
        for invalid in (True, 0, 2, 3, 8, "4"):
            with self.subTest(invalid=invalid), self.assertRaises(reference.ReferenceError):
                reference.block_tensor_shapes(invalid)

    def test_source_authentication_and_wrong_successor_before_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(4, 8):
                tiny = {n: [1 + (j % 2)] for j, n in enumerate(reference.block_tensor_shapes(index))}
                with mock.patch.dict(reference.BLOCK_TENSOR_SHAPES, {index: tiny}):
                    weights = root / f"block{index}.safetensors"
                    manifest_path = root / f"block{index}.safetensors.manifest.json"
                    doc, encoded = _tiny_source_file(weights, index)
                    data = (json.dumps(doc, sort_keys=True) + "\n").encode()
                    manifest_path.write_bytes(data)
                    args = dict(weights=weights, weights_sha256=hashlib.sha256(encoded).hexdigest(),
                                manifest_path=manifest_path, manifest_sha256=hashlib.sha256(data).hexdigest(),
                                block_index=index)
                    admitted = reference.validate_source_pair(**args)
                    self.assertEqual(admitted.block_index, index)
                    reference.validate_source_unchanged(admitted)
                    for mutation in (dict(weights_sha256="0"*64), dict(manifest_sha256="0"*64),
                                     dict(block_index=7 if index == 4 else 4)):
                        with self.subTest(index=index, mutation=mutation), self.assertRaises(reference.ReferenceError):
                            reference.validate_source_pair(**(args | mutation))
                    weights.write_bytes(encoded + b"changed")
                    with self.assertRaisesRegex(reference.ReferenceError, "changed after admission"):
                        reference.validate_source_unchanged(admitted)

    def test_cli_omitted_output_uses_the_bound_index_defaults_before_torch(self):
        args = reference._fixture_parser().parse_args([
            "--weights-sha256", "a"*64, "--manifest-sha256", "b"*64])
        self.assertEqual(args.block_index, 4)
        with mock.patch.object(reference, "_prepare_outputs", side_effect=RuntimeError("stop before admission")) as prepare:
            with self.assertRaisesRegex(RuntimeError, "stop before admission"):
                reference.produce_cache(args)
            prepare.assert_called_once_with(reference.DEFAULT_CACHE[4], reference.DEFAULT_REPORT[4])

    def test_cpu_import_does_not_import_torch(self):
        script = ("import sys; " + f"sys.path.insert(0, {str(SUPPORT)!r}); " +
                  "import hy_nomas_block47_bf16_reference; assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-B", "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
