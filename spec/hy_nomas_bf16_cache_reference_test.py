#!/usr/bin/env python3
"""CPU-only admission tests for the streamed Hy-Nomas BF16 cache oracle."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "examples" / "support"
SPEC_SUPPORT = ROOT / "spec" / "support"
sys.path.insert(0, str(SUPPORT))
sys.path.insert(0, str(SPEC_SUPPORT))
sys.path.insert(0, str(ROOT / "spec"))

import hy_nomas_bf16_cache_reference as cache_reference  # noqa: E402
from hy_nomas_pretrained_serial_reference_test import valid_document  # noqa: E402


def valid_serial_document():
    document = valid_document()
    document["producer"].update({
        "block_invocations": ["block0", "block1"],
        "attention_adapter_calls_by_block": [1, 1],
        "gpu_handoff": {"same_mps_tensor_objects": True},
    })
    return document


class HyNomasBf16CacheReferenceTests(unittest.TestCase):
    def test_accepts_pinned_serial_capture_envelope(self):
        document = valid_serial_document()
        cache_reference.validate_serial_reference_envelope(
            document,
            reference_sha256=cache_reference.SERIAL_REFERENCE_SHA256,
            sidecar_sha256=cache_reference.SERIAL_SIDECAR_SHA256,
            sidecar_nbytes=cache_reference.SERIAL_SIDECAR_BYTES,
        )

    def test_accepts_selected_block1_and_rejects_index_name_or_pin_confusion(self):
        document = valid_serial_document()
        cache_reference.validate_serial_reference_envelope(
            document,
            reference_sha256=cache_reference.SERIAL_REFERENCE_SHA256,
            sidecar_sha256=cache_reference.SERIAL_SIDECAR_SHA256,
            sidecar_nbytes=cache_reference.SERIAL_SIDECAR_BYTES,
            block_index=1,
        )

        for field, wrong_value in (
            ("block_index", 0),
            ("tensor_prefix", "double_blocks.0."),
            ("file_sha256", cache_reference.serial_reference.BLOCK_PINS["block0"]["file_sha256"]),
            ("manifest_sha256", cache_reference.serial_reference.BLOCK_PINS["block0"]["manifest_sha256"]),
        ):
            with self.subTest(field=field):
                confused = valid_serial_document()
                confused["source"]["blocks"]["block1"][field] = wrong_value
                with self.assertRaises(cache_reference.ReferenceError):
                    cache_reference.validate_serial_reference_envelope(
                        confused,
                        reference_sha256=cache_reference.SERIAL_REFERENCE_SHA256,
                        sidecar_sha256=cache_reference.SERIAL_SIDECAR_SHA256,
                        sidecar_nbytes=cache_reference.SERIAL_SIDECAR_BYTES,
                        block_index=1,
                    )

        for invalid_index in (True, -1, 2, "1"):
            with self.subTest(block_index=invalid_index), self.assertRaises(
                    cache_reference.ReferenceError):
                cache_reference.validate_serial_reference_envelope(
                    valid_serial_document(),
                    reference_sha256=cache_reference.SERIAL_REFERENCE_SHA256,
                    sidecar_sha256=cache_reference.SERIAL_SIDECAR_SHA256,
                    sidecar_nbytes=cache_reference.SERIAL_SIDECAR_BYTES,
                    block_index=invalid_index,
                )

    def test_rejects_changed_reference_hash_and_inherited_precision(self):
        document = valid_serial_document()
        with self.assertRaisesRegex(cache_reference.ReferenceError, "reference SHA"):
            cache_reference.validate_serial_reference_envelope(
                document,
                reference_sha256="0" * 64,
                sidecar_sha256=cache_reference.SERIAL_SIDECAR_SHA256,
                sidecar_nbytes=cache_reference.SERIAL_SIDECAR_BYTES,
            )

        document = valid_serial_document()
        document["precision"]["compute"] = "BF16"
        with self.assertRaisesRegex(cache_reference.ReferenceError, "precision"):
            cache_reference.validate_serial_reference_envelope(
                document,
                reference_sha256=cache_reference.SERIAL_REFERENCE_SHA256,
                sidecar_sha256=cache_reference.SERIAL_SIDECAR_SHA256,
                sidecar_nbytes=cache_reference.SERIAL_SIDECAR_BYTES,
            )

    def test_rejects_unpinned_sidecar_and_missing_block0_capture(self):
        document = valid_serial_document()
        with self.assertRaisesRegex(cache_reference.ReferenceError, "sidecar"):
            cache_reference.validate_serial_reference_envelope(
                document,
                reference_sha256=cache_reference.SERIAL_REFERENCE_SHA256,
                sidecar_sha256="1" * 64,
                sidecar_nbytes=cache_reference.SERIAL_SIDECAR_BYTES,
            )

        document = valid_serial_document()
        del document["tensors"]["block0.img.output"]
        with self.assertRaisesRegex(cache_reference.ReferenceError, "inventory"):
            cache_reference.validate_serial_reference_envelope(
                document,
                reference_sha256=cache_reference.SERIAL_REFERENCE_SHA256,
                sidecar_sha256=cache_reference.SERIAL_SIDECAR_SHA256,
                sidecar_nbytes=cache_reference.SERIAL_SIDECAR_BYTES,
            )

    def test_safetensors_header_has_exact_bf16_ranges_and_roundtrips(self):
        tensor_specs = {
            "double_blocks.0.first": ([2, 2], 8),
            "double_blocks.0.second": ([3], 6),
        }
        header, records = cache_reference.build_safetensors_header(tensor_specs)
        header_bytes = header[8:]
        self.assertEqual(struct.unpack("<Q", header[:8])[0], len(header_bytes))
        decoded = json.loads(header_bytes.rstrip(b" ").decode("utf-8"))
        self.assertEqual(decoded["double_blocks.0.first"]["dtype"], "BF16")
        self.assertEqual(decoded["double_blocks.0.first"]["shape"], [2, 2])
        self.assertEqual(decoded["double_blocks.0.first"]["data_offsets"], [0, 8])
        self.assertEqual(decoded["double_blocks.0.second"]["data_offsets"], [8, 14])
        self.assertEqual(records["double_blocks.0.second"]["nbytes"], 6)
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "small-bf16.safetensors"
            path.write_bytes(header + bytes(14))
            with safe_open(str(path), framework="pt", device="cpu") as artifact:
                self.assertEqual(set(artifact.keys()), set(tensor_specs))
                self.assertEqual(artifact.get_slice("double_blocks.0.first").get_dtype(), "BF16")
                self.assertEqual(artifact.get_slice("double_blocks.0.first").get_shape(), [2, 2])
                self.assertEqual(list(artifact.get_tensor("double_blocks.0.second").shape), [3])

    def test_rejects_invalid_bf16_shapes_and_extents(self):
        for specs in (
            {"double_blocks.0.x": ([0, 2], 0)},
            {"double_blocks.0.x": ([2, 2], 6)},
            {"double_blocks.0.x": ([True], 2)},
            {"__metadata__": ([1], 2)},
        ):
            with self.subTest(specs=specs), self.assertRaises(cache_reference.ReferenceError):
                cache_reference.build_safetensors_header(specs)

    def test_block0_inventory_preserves_all_source_names_and_shapes(self):
        shapes = cache_reference._block0_tensor_shapes()
        self.assertEqual(len(shapes), 32)
        self.assertEqual(set(shapes), set(cache_reference.serial_reference._expected_source_names(
            0,
            cache_reference.serial_reference.block_reference._expected_block_keys(
                2048, 8192, 16
            ),
        )))
        self.assertEqual(shapes["double_blocks.0.img_mod.linear.weight"], [12288, 2048])
        self.assertEqual(shapes["double_blocks.0.img_mlp.fc2.weight"], [2048, 8192])

    def test_block1_inventory_uses_distinct_authenticated_source_names(self):
        shapes = cache_reference._block_tensor_shapes(1)
        self.assertEqual(len(shapes), 32)
        self.assertTrue(all(name.startswith("double_blocks.1.") for name in shapes))
        self.assertEqual(shapes["double_blocks.1.img_mod.linear.weight"], [12288, 2048])
        self.assertEqual(shapes["double_blocks.1.img_mlp.fc2.weight"], [2048, 8192])
        self.assertEqual(set(cache_reference._block0_tensor_shapes()), {
            name.replace("double_blocks.1.", "double_blocks.0.") for name in shapes
        })

        class WrongBlockSource:
            def keys(self):
                return cache_reference._block0_tensor_shapes()

        with self.assertRaisesRegex(cache_reference.ReferenceError, "block1 source tensor inventory"):
            cache_reference._validate_source_inventory(
                WrongBlockSource(), shapes, block_index=1
            )

    def test_selected_capture_hashes_bind_to_requested_block(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")
        document = valid_serial_document()
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "serial.tensors.f32le"
            with sidecar.open("wb") as stream:
                stream.truncate(cache_reference.SERIAL_SIDECAR_BYTES)
            for index in (0, 1):
                captures = cache_reference._capture_hashes(document, sidecar, index)
                self.assertEqual(len(captures), 40)
                self.assertTrue(all(name.startswith(f"block{index}.") for name in captures))
                self.assertFalse(any(name.startswith(f"block{1-index}.") for name in captures))

    def test_cli_keeps_block0_default_and_accepts_generic_block1_selection(self):
        base = [
            "--output", "cache.safetensors",
            "--report", "cache.json",
            "--reference-sha256", cache_reference.SERIAL_REFERENCE_SHA256,
        ]
        legacy = cache_reference._fixture_parser().parse_args(base + [
            "--block0-weights-sha256", cache_reference.serial_reference.BLOCK_PINS["block0"]["file_sha256"],
            "--block0-manifest-sha256", cache_reference.serial_reference.BLOCK_PINS["block0"]["manifest_sha256"],
        ])
        self.assertEqual(legacy.block_index, 0)
        self.assertIsNone(legacy.block_weights)
        selected = cache_reference._fixture_parser().parse_args(base + [
            "--block-index", "1",
            "--block-weights-sha256", cache_reference.serial_reference.BLOCK_PINS["block1"]["file_sha256"],
            "--block-manifest-sha256", cache_reference.serial_reference.BLOCK_PINS["block1"]["manifest_sha256"],
        ])
        self.assertEqual(selected.block_index, 1)
        with self.assertRaises(SystemExit):
            cache_reference._fixture_parser().parse_args(base + [
                "--block-index", "2",
                "--block-weights-sha256", "0" * 64,
                "--block-manifest-sha256", "0" * 64,
            ])

    def test_publication_refuses_existing_or_symlink_destinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing.json"
            existing.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                cache_reference.publish_noreplace(existing, b"replace")
            self.assertEqual(existing.read_bytes(), b"keep")

            target = root / "target.json"
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(cache_reference.ReferenceError):
                cache_reference.publish_noreplace(link, b"data")


if __name__ == "__main__":
    unittest.main()
