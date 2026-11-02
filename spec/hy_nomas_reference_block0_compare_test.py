#!/usr/bin/env python3
"""CPU-only tests for streamed source/native block-0 numeric comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))

import hy_nomas_reference_block0_compare as compare  # noqa: E402
import hy_nomas_reference_cache as oracle  # noqa: E402
import hy_nomas_reference as source_reference  # noqa: E402
import hy_nomas_reference_preqkv_compare as preqkv_compare  # noqa: E402


def write_safetensors(
    path: Path,
    records: list[tuple[str, str, tuple[int, ...], bytes]],
    *,
    metadata: dict[str, str] | None = None,
) -> str:
    header: dict[str, object] = {}
    offset = 0
    for name, dtype, shape, raw in records:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    header["__metadata__"] = metadata if metadata is not None else {"format": "test"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-8 - len(encoded)) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _name, _dtype, _shape, raw in records:
            stream.write(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Block0CompareTests(unittest.TestCase):
    def _pair(self, directory: Path, candidate_values: tuple[float, ...]):
        source_path = directory / "source.safetensors"
        candidate_path = directory / "candidate.safetensors"
        source_sha = write_safetensors(source_path, [
            ("a", "BF16", (2,), struct.pack("<HH", 0x3F80, 0x4000)),
            ("b", "BF16", (2,), struct.pack("<HH", 0x4040, 0x4080)),
        ])
        candidate_sha = write_safetensors(candidate_path, [
            ("joined", "F32", (4,), struct.pack("<4f", *candidate_values)),
        ])
        source = compare.TensorArtifact(source_path, expected_sha256=source_sha,
                                        expected_names=("a", "b"))
        candidate = compare.TensorArtifact(candidate_path, expected_sha256=candidate_sha,
                                           expected_names=("joined",))
        return source, candidate

    def test_streamed_comparison_decodes_bf16_and_joins_source_ranges(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            source, candidate = self._pair(Path(temporary), (1.0, 2.0, 3.0, 4.0))
            result = compare.compare_tensor(
                candidate,
                "joined",
                [compare._tensor_ref(source, "a"), compare._tensor_ref(source, "b")],
                chunk_elements=3,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["mismatch_count"], 0)
            self.assertEqual(result["max_abs_error"], 0.0)

    def test_numeric_mismatch_fails_even_when_artifact_hashes_are_valid(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            source, candidate = self._pair(Path(temporary), (1.0, 2.0, 3.0, 4.25))
            result = compare.compare_tensor(
                candidate,
                "joined",
                [compare._tensor_ref(source, "a"), compare._tensor_ref(source, "b")],
                chunk_elements=2,
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["mismatch_count"], 1)
            self.assertEqual(result["max_abs_error"], 0.25)

    def test_rejects_tolerance_wider_than_declared_parity_threshold(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            source, candidate = self._pair(Path(temporary), (1.0, 2.0, 3.0, 4.0))
            for kwargs in ({"atol": 2.0001e-5}, {"rtol": 2.0001e-5}, {"atol": float("nan")},
                           {"rtol": float("inf")}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                        compare.CompareError, "no wider than 2e-5"):
                    compare.compare_tensor(
                        candidate, "joined",
                        [compare._tensor_ref(source, "a"), compare._tensor_ref(source, "b")],
                        **kwargs,
                    )

    def test_rejects_nonfinite_numeric_capture_values(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            source, candidate = self._pair(Path(temporary), (1.0, 2.0, 3.0, float("nan")))
            with self.assertRaisesRegex(compare.CompareError, "non-finite"):
                compare.compare_tensor(
                    candidate, "joined",
                    [compare._tensor_ref(source, "a"), compare._tensor_ref(source, "b")],
                )

    def test_artifact_hash_and_exact_inventory_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.safetensors"
            digest = write_safetensors(path, [
                ("x", "F32", (1,), struct.pack("<f", 1.0)),
            ])
            with self.assertRaisesRegex(compare.CompareError, "SHA-256"):
                compare.TensorArtifact(path, expected_sha256="0" * 64,
                                       expected_names=("x",))
            with self.assertRaisesRegex(compare.CompareError, "tensor names"):
                compare.TensorArtifact(path, expected_sha256=digest,
                                       expected_names=("missing",))

    def test_rejects_corrupt_shape_byte_extent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corrupt.safetensors"
            write_safetensors(path, [
                ("x", "F32", (1,), struct.pack("<f", 1.0)),
            ])
            raw = bytearray(path.read_bytes())
            header_bytes = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8:8 + header_bytes].decode().rstrip(" "))
            header["x"]["shape"] = [2]
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * ((-8 - len(encoded)) % 8)
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<f", 1.0))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(compare.CompareError, "extent"):
                compare.TensorArtifact(path, expected_sha256=digest, expected_names=("x",))

    def _receipt_pair(self, directory: Path):
        source_path = directory / "source.safetensors"
        native_path = directory / "native.safetensors"
        source_path.write_bytes(b"source artifact")
        native_path.write_bytes(b"native artifact")
        sha = "a" * 64
        source = {
            "schema": compare.SOURCE_SCHEMA,
            "operation": "upstream_block0_first_chunk_kv_cache_append",
            "upstream_revision": compare.UPSTREAM_REVISION,
            "upstream_source_root": "/pinned/source",
            "upstream_source_file_sha256": "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee",
            "upstream_dependency_sha256": dict(source_reference.PINNED_SOURCE_SHA256),
            "inference_dtype_source_sha256": dict(oracle.INFERENCE_DTYPE_SOURCE_SHA256),
            "weight_source": {
                "sha256": compare.BLOCK_WEIGHTS_SHA256,
                "source_dtype": "BF16",
                "tensor_prefix": "double_blocks.0.",
                "tensor_count": 32,
                "parameter_dtype": "BF16",
            },
            "parameter_dtype_basis": list(compare.PARAMETER_DTYPE_BASIS),
            "precision": {"mode": "bf16"},
            "scope": "pinned source block capture",
            "cache_append": {
                "cache_valid_len": 0,
                "update_cache": True,
                "cache_tokens": 4320 + 529,
                "attention_executed": False,
                "cache_tensor_order": "image tokens followed by valid text prefix",
            },
            "input_bundle": {
                "manifest_sha256": "b" * 64,
                "payload_sha256": "c" * 64,
                "reference_latent_artifact_sha256": "d" * 64,
                "timeline_manifest_sha256": "e" * 64,
                "valid_text_tokens": 529,
            },
            "tensor_artifact": {
                "path": source_path.name,
                "sha256": sha,
                "tensor_count": 14,
                "tensors": {name: {} for name in (*compare.PRE_CACHE_NAMES, "cache.k_bf16", "cache.v_bf16")},
            },
        }
        native = {
            "schema": compare.NATIVE_SCHEMA,
            "schema_version": 1,
            "native_revision": compare.UPSTREAM_REVISION,
            "source_revision": compare.NATIVE_IMPLEMENTATION_REVISION,
            "scope": compare.NATIVE_SCOPE,
            "full_upstream_t0_preprocessing_parity": False,
            "history_cache_write_executed": False,
            "cache_tensor_semantics": compare.NATIVE_CACHE_SEMANTICS,
            "input_bundle_manifest_sha256": "b" * 64,
            "input_bundle_payload_sha256": "c" * 64,
            "reference_latent_artifact_sha256": "d" * 64,
            "timeline_manifest_sha256": "e" * 64,
            "valid_text_tokens": 529,
            "text_storage_tokens": 1280,
            "text_padding_rows": 1280 - 529,
            "text_padding_value": 0,
            "weights_sha256": compare.BLOCK_WEIGHTS_SHA256,
            "precision_control": "bf16-boundary-control",
            "tensor_artifact": {
                "basename": native_path.name,
                "sha256": sha,
                "bytes": native_path.stat().st_size,
                "dtype": "F32",
                "tensors": [],
            },
            "named_buffer_accounting": {
                "input_buffers": 1,
                "block_weight_residents": 2,
                "block_scratch": 3,
                "mps_weight_stage": 4,
                "compact_attention_buffers": 5,
                "qk_rms_square_and_mean_buffers": 6,
                "total_named_bytes": 21,
                "opaque_mpsgraph_private_bytes": "not measured; excluded from named Metal-buffer total",
            },
            "execution_counters": {
                "encode_allocations": 0,
                "encode_readbacks": 0,
                "preterminal_waits": 0,
                "terminal_waits": 1,
            },
        }
        return source, native, source_path, native_path

    def test_receipts_must_bind_same_inputs_and_valid_append_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, native, source_path, native_path = self._receipt_pair(Path(temporary))
            # The source writer serializes JSON with sort_keys=True, so map
            # ordering is lexical even though its capture tensor order differs.
            source["tensor_artifact"]["tensors"] = dict(
                sorted(source["tensor_artifact"]["tensors"].items())
            )
            compare.validate_receipt_pair(
                source, native, source_path=source_path, native_path=native_path,
                precision="BF16BoundaryControl", valid_text_tokens=529,
                input_manifest_sha256="b" * 64, input_payload_sha256="c" * 64,
                latent_artifact_sha256="d" * 64, timeline_manifest_sha256="e" * 64,
                weights_sha256=compare.BLOCK_WEIGHTS_SHA256,
            )
            native["history_cache_write_executed"] = True
            with self.assertRaisesRegex(compare.CompareError, "prospective cache"):
                compare.validate_receipt_pair(
                    source, native, source_path=source_path, native_path=native_path,
                    precision="BF16BoundaryControl", valid_text_tokens=529,
                    input_manifest_sha256="b" * 64, input_payload_sha256="c" * 64,
                    latent_artifact_sha256="d" * 64, timeline_manifest_sha256="e" * 64,
                    weights_sha256=compare.BLOCK_WEIGHTS_SHA256,
                )

    def test_receipt_comparison_rejects_input_or_precision_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, native, source_path, native_path = self._receipt_pair(Path(temporary))
            native["precision_control"] = "f32-control"
            with self.assertRaisesRegex(compare.CompareError, "precision"):
                compare.validate_receipt_pair(
                    source, native, source_path=source_path, native_path=native_path,
                    precision="BF16BoundaryControl", valid_text_tokens=529,
                    input_manifest_sha256="b" * 64, input_payload_sha256="c" * 64,
                    latent_artifact_sha256="d" * 64, timeline_manifest_sha256="e" * 64,
                    weights_sha256=compare.BLOCK_WEIGHTS_SHA256,
                )

    def test_source_receipt_pins_dependency_hashes_and_dtype_basis(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, native, source_path, native_path = self._receipt_pair(Path(temporary))
            source["upstream_dependency_sha256"]["hymm/models/dit/nomas_cpt.py"] = "f" * 64
            with self.assertRaisesRegex(compare.CompareError, "dependency hashes"):
                compare.validate_receipt_pair(
                    source, native, source_path=source_path, native_path=native_path,
                    precision="BF16BoundaryControl", valid_text_tokens=529,
                    input_manifest_sha256="b" * 64, input_payload_sha256="c" * 64,
                    latent_artifact_sha256="d" * 64, timeline_manifest_sha256="e" * 64,
                    weights_sha256=compare.BLOCK_WEIGHTS_SHA256,
                )
            source, native, source_path, native_path = self._receipt_pair(Path(temporary))
            source["parameter_dtype_basis"][0] = "unverified claim"
            with self.assertRaisesRegex(compare.CompareError, "dtype basis"):
                compare.validate_receipt_pair(
                    source, native, source_path=source_path, native_path=native_path,
                    precision="BF16BoundaryControl", valid_text_tokens=529,
                    input_manifest_sha256="b" * 64, input_payload_sha256="c" * 64,
                    latent_artifact_sha256="d" * 64, timeline_manifest_sha256="e" * 64,
                    weights_sha256=compare.BLOCK_WEIGHTS_SHA256,
                )

    def test_native_execution_counters_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, native, source_path, native_path = self._receipt_pair(Path(temporary))
            native["execution_counters"]["encode_readbacks"] = 1
            with self.assertRaisesRegex(compare.CompareError, "terminal-drain"):
                compare.validate_receipt_pair(
                    source, native, source_path=source_path, native_path=native_path,
                    precision="BF16BoundaryControl", valid_text_tokens=529,
                    input_manifest_sha256="b" * 64, input_payload_sha256="c" * 64,
                    latent_artifact_sha256="d" * 64, timeline_manifest_sha256="e" * 64,
                    weights_sha256=compare.BLOCK_WEIGHTS_SHA256,
                )


class PreQKVComparatorTests(unittest.TestCase):
    @staticmethod
    def _write_artifact(path: Path, *, native: bool, mismatch: str | None = None) -> compare.TensorArtifact:
        records = []
        for name in preqkv_compare.NAMES:
            if native:
                value = (1.0, 2.25) if name == mismatch else (1.0, 2.0)
                raw = struct.pack("<2f", *value)
                records.append((name, "F32", (2,), raw))
            else:
                records.append((name, "BF16", (2,), struct.pack("<HH", 0x3F80, 0x4000)))
        digest = write_safetensors(path, records)
        return compare.TensorArtifact(path, expected_sha256=digest,
                                      expected_names=preqkv_compare.NAMES)

    def test_compares_all_eight_in_order_and_reports_first_divergence(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._write_artifact(root / "source.safetensors", native=False)
            native = self._write_artifact(
                root / "native.safetensors", native=True, mismatch="img.qkv_input"
            )
            result = preqkv_compare.compare_named_tensors(source, native)
            self.assertFalse(result["passed"])
            self.assertEqual(result["tensor_count"], 8)
            self.assertEqual(result["first_differing_tensor"], "img.qkv_input")

    def test_rejects_wider_tolerance(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._write_artifact(root / "source.safetensors", native=False)
            native = self._write_artifact(root / "native.safetensors", native=True)
            with self.assertRaisesRegex(preqkv_compare.PreQKVCompareError, "no wider than 2e-5"):
                preqkv_compare.compare_named_tensors(source, native, atol=3e-5)

    def test_parent_artifact_and_receipt_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_receipt = root / "parent.json"
            parent_artifact = root / "parent.safetensors"
            parent_receipt.write_text("parent receipt\n")
            parent_artifact.write_bytes(b"parent artifact")
            receipt = {
                "schema": preqkv_compare.SOURCE_SCHEMA,
                "schema_version": 1,
                "upstream_revision": preqkv_compare.UPSTREAM_REVISION,
                "scope": preqkv_compare.SOURCE_SCOPE,
                "precision_control": "F32Control",
                "input_bundle_manifest_sha256": "1" * 64,
                "input_bundle_payload_sha256": "2" * 64,
                "reference_latent_artifact_sha256": "3" * 64,
                "timeline_manifest_sha256": "4" * 64,
                "valid_text_tokens": 3,
                "weights_sha256": compare.BLOCK_WEIGHTS_SHA256,
                "parent_source_receipt_sha256": "0" * 64,
                "parent_source_artifact_sha256": "0" * 64,
                "tensor_artifact": {},
            }
            identities = {
                "input_bundle_manifest_sha256": "1" * 64,
                "input_bundle_payload_sha256": "2" * 64,
                "reference_latent_artifact_sha256": "3" * 64,
                "timeline_manifest_sha256": "4" * 64,
                "weights_sha256": compare.BLOCK_WEIGHTS_SHA256,
            }
            with self.assertRaisesRegex(preqkv_compare.PreQKVCompareError, "exact parent"):
                preqkv_compare._validate_companion_receipt(
                    receipt, native=False, tensor_path=root / "unused.safetensors",
                    parent_receipt_path=parent_receipt,
                    parent_artifact_path=parent_artifact,
                    precision="F32Control", valid_text_tokens=3, identities=identities,
                )

    def test_admits_final_native_metadata_and_source_bf16_companion_contract(self):
        """Exercise receipt, metadata, order, dtype, shape, and parent binding together."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = {
                "input_bundle_manifest_sha256": "1" * 64,
                "input_bundle_payload_sha256": "2" * 64,
                "reference_latent_artifact_sha256": "3" * 64,
                "timeline_manifest_sha256": "4" * 64,
                "weights_sha256": compare.BLOCK_WEIGHTS_SHA256,
            }
            tiny_shapes = {name: (1, 2) for name in preqkv_compare.NAMES}

            def make_parent(prefix: str) -> tuple[Path, Path]:
                receipt_path = root / f"{prefix}-parent.json"
                artifact_path = root / f"{prefix}-parent.safetensors"
                receipt_path.write_text(f"{prefix} parent receipt\n", encoding="utf-8")
                artifact_path.write_bytes(f"{prefix} parent artifact\n".encode())
                return receipt_path, artifact_path

            def make_companion(*, native: bool, prefix: str):
                parent_receipt, parent_artifact = make_parent(prefix)
                parent_receipt_sha = hashlib.sha256(parent_receipt.read_bytes()).hexdigest()
                parent_artifact_sha = hashlib.sha256(parent_artifact.read_bytes()).hexdigest()
                precision = "BF16BoundaryControl"
                source_tensor_dtypes = preqkv_compare._expected_source_boundary_dtypes(precision)
                tensor_dtypes = (
                    {name: "F32" for name in preqkv_compare.NAMES}
                    if native else source_tensor_dtypes
                )
                disk_dtype = (
                    "F32" if native else
                    (next(iter(set(tensor_dtypes.values())))
                     if len(set(tensor_dtypes.values())) == 1 else "MIXED")
                )
                if native:
                    metadata = {
                        "format": preqkv_compare.NATIVE_SCHEMA,
                        "native_revision": preqkv_compare.UPSTREAM_REVISION,
                        "source_revision": preqkv_compare.NATIVE_SOURCE_REVISION,
                        "precision_control": "BF16BoundaryControl",
                        "parent_native_receipt_sha256": parent_receipt_sha,
                        "parent_native_artifact_sha256": parent_artifact_sha,
                    }
                else:
                    metadata = {
                        "format": preqkv_compare.SOURCE_SCHEMA,
                        "upstream_revision": preqkv_compare.UPSTREAM_REVISION,
                        "precision_control": precision,
                        "parent_source_receipt_sha256": parent_receipt_sha,
                        "parent_source_artifact_sha256": parent_artifact_sha,
                    }
                raw_by_name = {
                    name: (
                        struct.pack("<2f", 1.25, 1.25)
                        if tensor_dtypes[name] == "F32"
                        else struct.pack("<2H", 0x3FA0, 0x3FA0)
                    )
                    for name in preqkv_compare.NAMES
                }
                artifact_path = root / f"{prefix}-preqkv.safetensors"
                records = [
                    (name, tensor_dtypes[name], tiny_shapes[name], raw_by_name[name])
                    for name in preqkv_compare.NAMES
                ]
                artifact_sha = write_safetensors(artifact_path, records, metadata=metadata)
                receipt = {
                    "schema": preqkv_compare.NATIVE_SCHEMA if native else preqkv_compare.SOURCE_SCHEMA,
                    "schema_version": 1,
                    "scope": preqkv_compare.NATIVE_SCOPE if native else preqkv_compare.SOURCE_SCOPE,
                    "precision_control": precision,
                    **identities,
                    "valid_text_tokens": 1,
                    "weights_sha256": identities["weights_sha256"],
                    "tensor_artifact": {
                        "basename": artifact_path.name,
                        "sha256": artifact_sha,
                        "bytes": artifact_path.stat().st_size,
                        "dtype": disk_dtype,
                        "tensors": [
                            {
                                "name": name,
                                "dtype": tensor_dtypes[name],
                                "shape": list(tiny_shapes[name]),
                                "nbytes": len(raw_by_name[name]),
                                "sha256": hashlib.sha256(raw_by_name[name]).hexdigest(),
                            }
                            for name in preqkv_compare.NAMES
                        ],
                    },
                }
                if native:
                    receipt.update({
                        "native_revision": preqkv_compare.UPSTREAM_REVISION,
                        "source_revision": preqkv_compare.NATIVE_SOURCE_REVISION,
                        "parent_native_receipt_sha256": parent_receipt_sha,
                        "parent_native_artifact_sha256": parent_artifact_sha,
                    })
                else:
                    receipt.update({
                        "upstream_revision": preqkv_compare.UPSTREAM_REVISION,
                        "parent_source_receipt_sha256": parent_receipt_sha,
                        "parent_source_artifact_sha256": parent_artifact_sha,
                    })
                receipt_path = root / f"{prefix}-preqkv.json"
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                return receipt, artifact_path, receipt_path, parent_receipt, parent_artifact

            with patch.object(preqkv_compare, "_tensor_shapes", return_value=tiny_shapes):
                source = make_companion(native=False, prefix="source")
                native = make_companion(native=True, prefix="native")
                source_artifact = preqkv_compare._validate_companion_receipt(
                    source[0], native=False, tensor_path=source[1],
                    parent_receipt_path=source[3], parent_artifact_path=source[4],
                    precision="BF16BoundaryControl", valid_text_tokens=1, identities=identities,
                )
                native_artifact = preqkv_compare._validate_companion_receipt(
                    native[0], native=True, tensor_path=native[1],
                    parent_receipt_path=native[3], parent_artifact_path=native[4],
                    precision="BF16BoundaryControl", valid_text_tokens=1, identities=identities,
                )
                self.assertEqual(source_artifact.tensors[preqkv_compare.NAMES[0]]["dtype"], "BF16")
                self.assertEqual(source_artifact.tensors["img.norm1"]["dtype"], "F32")
                self.assertEqual(source_artifact.tensors["txt.qkv_input"]["dtype"], "F32")
                self.assertEqual(native_artifact.tensors[preqkv_compare.NAMES[0]]["dtype"], "F32")

                for prefix, is_native, fixture in (
                    ("source", False, source),
                    ("native", True, native),
                ):
                    receipt, artifact_path, _receipt_path, parent_receipt, parent_artifact = fixture

                    boolean_extent_receipt = json.loads(json.dumps(receipt))
                    boolean_extent_receipt["tensor_artifact"]["tensors"][0]["shape"][0] = True
                    with self.assertRaisesRegex(
                            preqkv_compare.PreQKVCompareError,
                            "receipt tensor record mismatch"):
                        preqkv_compare._validate_companion_receipt(
                            boolean_extent_receipt, native=is_native, tensor_path=artifact_path,
                            parent_receipt_path=parent_receipt,
                            parent_artifact_path=parent_artifact,
                            precision="BF16BoundaryControl", valid_text_tokens=1,
                            identities=identities,
                        )

                    # A receipt basename must resolve to the caller-selected file, and
                    # the original CLI leaf must be checked before path resolution.
                    symlink_dir = root / f"{prefix}-symlink-input"
                    symlink_dir.mkdir()
                    symlink_input = symlink_dir / artifact_path.name
                    symlink_input.symlink_to(artifact_path)
                    with self.assertRaisesRegex(
                            preqkv_compare.PreQKVCompareError,
                            "absolute regular non-symlink file"):
                        preqkv_compare._validate_companion_receipt(
                            receipt, native=is_native, tensor_path=symlink_input,
                            parent_receipt_path=parent_receipt,
                            parent_artifact_path=parent_artifact,
                            precision="BF16BoundaryControl", valid_text_tokens=1,
                            identities=identities,
                        )

                    caller_path = root / f"{prefix}-caller.safetensors"
                    caller_path.write_bytes(b"caller-selected regular file\n")
                    with self.assertRaisesRegex(
                            preqkv_compare.PreQKVCompareError,
                            "different tensor artifact"):
                        preqkv_compare._validate_companion_receipt(
                            receipt, native=is_native, tensor_path=caller_path,
                            parent_receipt_path=parent_receipt,
                            parent_artifact_path=parent_artifact,
                            precision="BF16BoundaryControl", valid_text_tokens=1,
                            identities=identities,
                        )

                    parent_receipt_link = root / f"{prefix}-parent-receipt-link.json"
                    parent_receipt_link.symlink_to(parent_receipt)
                    with self.assertRaisesRegex(
                            preqkv_compare.PreQKVCompareError,
                            "absolute regular non-symlink file"):
                        preqkv_compare._validate_companion_receipt(
                            receipt, native=is_native, tensor_path=artifact_path,
                            parent_receipt_path=parent_receipt_link,
                            parent_artifact_path=parent_artifact,
                            precision="BF16BoundaryControl", valid_text_tokens=1,
                            identities=identities,
                        )

                    parent_artifact_link = root / f"{prefix}-parent-artifact-link.safetensors"
                    parent_artifact_link.symlink_to(parent_artifact)
                    with self.assertRaisesRegex(
                            preqkv_compare.PreQKVCompareError,
                            "absolute regular non-symlink file"):
                        preqkv_compare._validate_companion_receipt(
                            receipt, native=is_native, tensor_path=artifact_path,
                            parent_receipt_path=parent_receipt,
                            parent_artifact_path=parent_artifact_link,
                            precision="BF16BoundaryControl", valid_text_tokens=1,
                            identities=identities,
                        )

                wrong_metadata = dict(native_artifact.metadata)
                wrong_metadata["source_revision"] = "0" * 40
                wrong_path = root / "native-wrong-metadata.safetensors"
                wrong_sha = write_safetensors(
                    wrong_path,
                    [(name, "F32", tiny_shapes[name], struct.pack("<2f", 1.25, 1.25))
                     for name in preqkv_compare.NAMES],
                    metadata=wrong_metadata,
                )
                wrong_receipt = dict(native[0])
                wrong_receipt["tensor_artifact"] = dict(native[0]["tensor_artifact"])
                wrong_receipt["tensor_artifact"]["basename"] = wrong_path.name
                wrong_receipt["tensor_artifact"]["sha256"] = wrong_sha
                wrong_receipt["tensor_artifact"]["bytes"] = wrong_path.stat().st_size
                wrong_receipt["tensor_artifact"]["tensors"] = [
                    {
                        "name": name,
                        "dtype": "F32",
                        "shape": list(tiny_shapes[name]),
                        "nbytes": 8,
                        "sha256": hashlib.sha256(struct.pack("<2f", 1.25, 1.25)).hexdigest(),
                    }
                    for name in preqkv_compare.NAMES
                ]
                with self.assertRaisesRegex(
                        preqkv_compare.PreQKVCompareError,
                        "safetensors metadata does not bind"):
                    preqkv_compare._validate_companion_receipt(
                        wrong_receipt, native=True, tensor_path=wrong_path,
                        parent_receipt_path=native[3], parent_artifact_path=native[4],
                        precision="BF16BoundaryControl", valid_text_tokens=1, identities=identities,
                    )


if __name__ == "__main__":
    unittest.main()
