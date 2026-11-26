#!/usr/bin/env python3
"""CPU-only admission tests for the block 2/3 MPS BF16 conversion oracle."""

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

import hy_nomas_block23_bf16_reference as reference  # noqa: E402


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


class HyNomasBlock23Bf16ReferenceTests(unittest.TestCase):
    def test_actual_block_inventories_and_range_identity_are_distinct(self):
        for block_index in (2, 3):
            shapes = reference.block_tensor_shapes(block_index)
            self.assertEqual(len(shapes), 32)
            self.assertEqual(set(shapes), set(reference.subset.expected_tensors(block_index)))
            self.assertTrue(all(name.startswith(f"double_blocks.{block_index}.")
                                for name in shapes))
            self.assertEqual(reference.block_payload_bytes(block_index), 604_227_584)
        self.assertNotEqual(reference.subset.BLOCK_HTTP_RANGES[2],
                            reference.subset.BLOCK_HTTP_RANGES[3])

    def test_cpu_validation_authenticates_file_manifest_header_and_all_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for block_index in (2, 3):
                with self.subTest(block_index=block_index):
                    # Keep this structural falsifier small while using the real
                    # production admission code and exact block name inventory.
                    tiny_shapes = {
                        name: [1 + (index % 2)]
                        for index, name in enumerate(reference.block_tensor_shapes(block_index))
                    }
                    with mock.patch.dict(reference.BLOCK_TENSOR_SHAPES,
                                         {block_index: tiny_shapes}):
                        weights = root / f"block{block_index}.safetensors"
                        manifest_path = root / f"block{block_index}.safetensors.manifest.json"
                        manifest, encoded = _tiny_source_file(weights, block_index)
                        # Regenerate manifest after the helper writes the file.
                        manifest["subset"]["payload_bytes"] = reference.block_payload_bytes(block_index)
                        manifest["output"]["size_bytes"] = len(encoded)
                        manifest["output"]["sha256"] = hashlib.sha256(encoded).hexdigest()
                        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
                        manifest_path.write_bytes(manifest_bytes)
                        file_sha = hashlib.sha256(encoded).hexdigest()
                        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
                        admitted = reference.validate_source_pair(
                            weights=weights,
                            weights_sha256=file_sha,
                            manifest_path=manifest_path,
                            manifest_sha256=manifest_sha,
                            block_index=block_index,
                        )
                        self.assertEqual(admitted.block_index, block_index)
                        self.assertEqual(len(admitted.tensors), 32)
                        self.assertEqual(admitted.payload_bytes, reference.block_payload_bytes(block_index))
                        reference.validate_source_unchanged(admitted)
                        weights.write_bytes(encoded + b"changed")
                        with self.assertRaisesRegex(reference.ReferenceError, "changed after admission"):
                            reference.validate_source_unchanged(admitted)

    def test_equal_shape_wrong_block_and_wrong_hashes_reject_before_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny_shapes = {name: [2] for name in reference.block_tensor_shapes(2)}
            patched = mock.patch.dict(reference.BLOCK_TENSOR_SHAPES, {2: tiny_shapes})
            with patched:
                weights = root / "block2.safetensors"
                manifest_path = root / "block2.safetensors.manifest.json"
                manifest, encoded = _tiny_source_file(weights, 2)
                manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
                manifest_path.write_bytes(manifest_bytes)
                file_sha = hashlib.sha256(encoded).hexdigest()
                manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

                for bad_file_sha, bad_manifest_sha, wrong_index in (
                    ("0" * 64, manifest_sha, 2),
                    (file_sha, "0" * 64, 2),
                    (file_sha, manifest_sha, 3),
                ):
                    with self.subTest(index=wrong_index, file=bad_file_sha[:4],
                                      manifest=bad_manifest_sha[:4]), self.assertRaises(
                                          reference.ReferenceError):
                        reference.validate_source_pair(
                            weights=weights,
                            weights_sha256=bad_file_sha,
                            manifest_path=manifest_path,
                            manifest_sha256=bad_manifest_sha,
                            block_index=wrong_index,
                        )

                mutations = (
                    ("source.range_start_inclusive", 0),
                    ("source.header_sha256_observed", "0" * 64),
                    ("source.payload_sha256_observed", "not-a-digest"),
                    ("subset.block_index", 3),
                    ("subset.tensor_prefix", "double_blocks.3."),
                    ("subset.conversion_performed", True),
                )
                for path, bad_value in mutations:
                    bad = json.loads(json.dumps(manifest))
                    section, key = path.split(".")
                    bad[section][key] = bad_value
                    raw = (json.dumps(bad, sort_keys=True) + "\n").encode()
                    manifest_path.write_bytes(raw)
                    with self.subTest(field=path), self.assertRaises(
                            reference.ReferenceError):
                        reference.validate_source_pair(
                            weights=weights,
                            weights_sha256=file_sha,
                            manifest_path=manifest_path,
                            manifest_sha256=hashlib.sha256(raw).hexdigest(),
                            block_index=2,
                        )

                malformed = json.loads(json.dumps(manifest))
                malformed["source"] = []
                raw = (json.dumps(malformed, sort_keys=True) + "\n").encode()
                manifest_path.write_bytes(raw)
                with self.assertRaises(reference.ReferenceError):
                    reference.validate_source_pair(
                        weights=weights,
                        weights_sha256=file_sha,
                        manifest_path=manifest_path,
                        manifest_sha256=hashlib.sha256(raw).hexdigest(),
                        block_index=2,
                    )

    def test_safetensors_rejects_duplicate_keys_dtype_shape_gaps_and_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny_shapes = {name: [1] for name in reference.block_tensor_shapes(2)}
            with mock.patch.dict(reference.BLOCK_TENSOR_SHAPES, {2: tiny_shapes}):
                good = root / "good.safetensors"
                _tiny_source_file(good, 2)
                self.assertEqual(len(reference.parse_source_inventory(good, 2).tensors), 32)
                raw = good.read_bytes()
                header_size = struct.unpack("<Q", raw[:8])[0]
                payload_start = 8 + header_size
                header = json.loads(raw[8:payload_start])

                variants = []
                duplicate = raw[8:payload_start].decode().rstrip()
                first_tensor = next(name for name in header if name != "__metadata__")
                variants.append(("duplicate", duplicate[:-1] + f',"{first_tensor}":{json.dumps(header[first_tensor])}' + "}"))
                wrong_dtype = json.loads(json.dumps(header))
                wrong_dtype[first_tensor]["dtype"] = "BF16"
                variants.append(("dtype", json.dumps(wrong_dtype, separators=(",", ":"))))
                bool_shape = json.loads(json.dumps(header))
                bool_shape[first_tensor]["shape"] = [True]
                variants.append(("bool-shape", json.dumps(bool_shape, separators=(",", ":"))))
                gap = json.loads(json.dumps(header))
                gap[first_tensor]["data_offsets"][0] = 1
                variants.append(("gap", json.dumps(gap, separators=(",", ":"))))

                for label, encoded_header in variants:
                    encoded = encoded_header.encode()
                    encoded += b" " * ((-len(encoded)) % 8)
                    path = root / f"{label}.safetensors"
                    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[payload_start:])
                    with self.subTest(mutation=label), self.assertRaises(
                            reference.ReferenceError):
                        reference.parse_source_inventory(path, 2)

                trailing = root / "trailing.safetensors"
                trailing.write_bytes(raw + b"x")
                with self.assertRaises(reference.ReferenceError):
                    reference.parse_source_inventory(trailing, 2)

    def test_bf16_header_rebases_exact_block_names_and_byte_extents(self):
        tiny = {name: [2] for name in reference.block_tensor_shapes(3)}
        with mock.patch.dict(reference.BLOCK_TENSOR_SHAPES, {3: tiny}):
            header, tensors = reference.build_bf16_header(3)
        header_size = struct.unpack("<Q", header[:8])[0]
        document = json.loads(header[8:8 + header_size])
        self.assertEqual(len(tensors), 32)
        self.assertEqual(set(document) - {"__metadata__"}, set(tiny))
        ranges = sorted(value["data_offsets"] for name, value in document.items()
                        if name != "__metadata__")
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 128)
        self.assertTrue(all(left[1] == right[0]
                            for left, right in zip(ranges, ranges[1:])))

    def test_cli_requires_caller_source_pins_and_only_accepts_blocks_2_or_3(self):
        base = ["--output", "cache.safetensors", "--report", "cache.json",
                "--weights", "source.safetensors", "--manifest", "source.manifest.json",
                "--weights-sha256", "a" * 64, "--manifest-sha256", "b" * 64]
        selected = reference._fixture_parser().parse_args(base + ["--block-index", "3"])
        self.assertEqual(selected.block_index, 3)
        self.assertEqual(reference._fixture_parser().parse_args(base).block_index, 2)
        for invalid in ("0", "1", "4", "-1"):
            with self.subTest(index=invalid), self.assertRaises(SystemExit):
                reference._fixture_parser().parse_args(base + ["--block-index", invalid])
        with self.assertRaises(SystemExit):
            reference._fixture_parser().parse_args(base[:-2])

    def test_module_cpu_validation_import_does_not_import_torch(self):
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(SUPPORT)!r}); "
            f"sys.path.insert(0, {str(SPEC_SUPPORT)!r}); "
            "import hy_nomas_block23_bf16_reference; "
            "assert 'torch' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_output_pair_publication_is_no_replace_and_rolls_back_partial_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "cache.safetensors"
            report = root / "cache.json"
            output_temp = root / ".cache.safetensors.partial"
            report_temp = root / ".cache.json.partial"
            output_temp.write_bytes(b"cache")
            report_temp.write_bytes(b"report")
            reference.publish_output_pair(output_temp, output, report_temp, report)
            self.assertEqual(output.read_bytes(), b"cache")
            self.assertEqual(report.read_bytes(), b"report")
            self.assertFalse(output_temp.exists())
            self.assertFalse(report_temp.exists())

            retry_output = root / "retry.safetensors"
            retry_report = root / "existing.json"
            retry_report.write_bytes(b"preserve")
            retry_output_temp = root / ".retry.safetensors.partial"
            retry_report_temp = root / ".existing.json.partial"
            retry_output_temp.write_bytes(b"must roll back")
            retry_report_temp.write_bytes(b"must not replace")
            with self.assertRaises(FileExistsError):
                reference.publish_output_pair(
                    retry_output_temp, retry_output, retry_report_temp, retry_report
                )
            self.assertFalse(retry_output.exists())
            self.assertEqual(retry_report.read_bytes(), b"preserve")
            self.assertFalse(retry_output_temp.exists())
            self.assertFalse(retry_report_temp.exists())

            existing_output = root / "kept.safetensors"
            existing_output.write_bytes(b"preserve output")
            second_report = root / "second.json"
            second_output_temp = root / ".kept.safetensors.partial"
            second_report_temp = root / ".second.json.partial"
            second_output_temp.write_bytes(b"must not replace")
            second_report_temp.write_bytes(b"unused")
            with self.assertRaises(FileExistsError):
                reference.publish_output_pair(
                    second_output_temp, existing_output, second_report_temp, second_report
                )
            self.assertEqual(existing_output.read_bytes(), b"preserve output")
            self.assertFalse(second_report.exists())


if __name__ == "__main__":
    unittest.main()
