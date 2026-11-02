from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from spec.support import hy_nomas_reference_cache as cache


_LATENT_BYTES = 276_480
_EXPECTED_LATENT_SHA256 = hashlib.sha256(bytes(_LATENT_BYTES)).hexdigest()
_TIMELINE_SHA256 = "b" * 64


def _digest_zeros(size: int) -> str:
    digest = hashlib.sha256()
    chunk = bytes(1024 * 1024)
    while size:
        amount = min(size, len(chunk))
        digest.update(chunk[:amount])
        size -= amount
    return digest.hexdigest()


def _write_safetensors_latent(path: Path, body: bytes) -> str:
    header = json.dumps({
        "latent": {
            "dtype": "BF16",
            "shape": [1, 32, 30, 9, 16],
            "data_offsets": [0, _LATENT_BYTES],
        },
    }, separators=(",", ":")).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.write(body)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReferenceCacheManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="hy-nomas-cache-validator-")
        cls.root = Path(cls.temp.name)
        cls.source_latent = cls.root / "reference.safetensors"
        cls.reference_sha256 = _write_safetensors_latent(
            cls.source_latent, bytes(_LATENT_BYTES)
        )
        cls.payload_template = cls.root / "template.tensors.bin"
        cls.manifest_template = cls.root / "template.json"
        cls.document = cls._make_fixture(cls.payload_template, cls.manifest_template)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    @staticmethod
    def _make_fixture(payload_path: Path, manifest_path: Path) -> dict:
        records: dict[str, dict[str, object]] = {}
        offset = 0
        payload_digest = hashlib.sha256()
        zero_chunk = bytes(1024 * 1024)
        with payload_path.open("xb") as stream:
            for name, dtype, shape in cache._expected_tensors(3):
                nbytes = cache._expected_nbytes(dtype, shape)
                range_digest = hashlib.sha256()
                remaining = nbytes
                while remaining:
                    amount = min(remaining, len(zero_chunk))
                    part = zero_chunk[:amount]
                    stream.write(part)
                    payload_digest.update(part)
                    range_digest.update(part)
                    remaining -= amount
                records[name] = {
                    "dtype": dtype,
                    "shape": list(shape),
                    "offset": offset,
                    "nbytes": nbytes,
                    "sha256": range_digest.hexdigest(),
                }
                offset += nbytes
        document = {
            "schema": cache.SCHEMA,
            "schema_version": cache.SCHEMA_VERSION,
            "upstream_revision": cache.UPSTREAM_REVISION,
            "scope": cache.SCOPE,
            "preprocessing": cache.PREPROCESSING,
            "reference_latent_artifact_sha256": ReferenceCacheManifestTest.reference_sha256,
            "timeline_manifest_sha256": _TIMELINE_SHA256,
            "valid_text_tokens": 3,
            "cache_writer": cache.CACHE_WRITER,
            "payload": manifest_path.with_suffix(".tensors.bin").name,
            "payload_bytes": offset,
            "payload_sha256": payload_digest.hexdigest(),
            "tensors": records,
        }
        manifest_path.write_text(json.dumps(document, separators=(",", ":")) + "\n")
        return document

    def setUp(self) -> None:
        self.case_root = Path(tempfile.mkdtemp(prefix="case-", dir=self.root))
        self.manifest = self.case_root / "fixture.json"
        self.payload = self.manifest.with_suffix(".tensors.bin")
        self.document = json.loads(json.dumps(self.__class__.document))
        self.document["payload"] = self.payload.name
        os.link(self.payload_template, self.payload)
        os.link(self.source_latent, self.case_root / "reference.safetensors")
        self.latent_path = self.case_root / "reference.safetensors"
        self._write_document()

    def tearDown(self) -> None:
        for path in self.case_root.iterdir():
            path.unlink()
        self.case_root.rmdir()

    def _write_document(self) -> None:
        self.manifest.write_text(json.dumps(self.document, separators=(",", ":")) + "\n")

    def _validate(self, **kwargs):
        reference_latent_sha256 = kwargs.pop(
            "expected_reference_latent_artifact_sha256", self.reference_sha256
        )
        timeline_sha256 = kwargs.pop(
            "expected_timeline_manifest_sha256", _TIMELINE_SHA256
        )
        return cache.validate_reference_cache_manifest(
            self.manifest,
            reference_latent_path=self.latent_path,
            expected_reference_latent_artifact_sha256=reference_latent_sha256,
            expected_timeline_manifest_sha256=timeline_sha256,
            **kwargs,
        )

    def test_accepts_authenticated_dynamic_text_extent(self) -> None:
        result = self._validate(expected_manifest_sha256=cache._hash_file(self.manifest))
        self.assertEqual(result["valid_text_tokens"], 3)
        self.assertEqual(result["tensor_count"], 9)
        self.assertEqual(
            result["tensor_order"],
            [name for name, _, _ in cache._expected_tensors(3)],
        )
        self.assertEqual(result["reference_latent_tensor_sha256"], _EXPECTED_LATENT_SHA256)

    def test_rejects_unexpected_schema_revision(self) -> None:
        self.document["upstream_revision"] = cache.TIMELINE_UPSTREAM_REVISION
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "schema, version, or upstream revision"):
            self._validate()

    def test_rejects_bool_text_extent(self) -> None:
        self.document["valid_text_tokens"] = True
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "valid_text_tokens"):
            self._validate()

    def test_rejects_missing_noisy_input(self) -> None:
        del self.document["tensors"]["cache.noisy_latent_f32"]
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "inventory names or insertion order"):
            self._validate()

    def test_rejects_wrong_tensor_shape_or_dtype(self) -> None:
        self.document["tensors"]["block.img_f32"]["dtype"] = "BF16"
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "schema, offset, extent, or digest"):
            self._validate()

    def test_rejects_wrong_tensor_shape(self) -> None:
        self.document["tensors"]["block.img_f32"]["shape"] = [1, 4320, 2047]
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "schema, offset, extent, or digest"):
            self._validate()

    def test_rejects_bool_offset(self) -> None:
        self.document["tensors"]["reference.latent_bf16"]["offset"] = False
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "schema, offset, extent, or digest"):
            self._validate()

    def test_rejects_noncontiguous_tensor_ranges(self) -> None:
        self.document["tensors"]["block.img_f32"]["offset"] += 2
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "schema, offset, extent, or digest"):
            self._validate()

    def test_rejects_wrong_timeline_authorization(self) -> None:
        with self.assertRaisesRegex(cache.ReferenceCacheError, "timeline manifest identity"):
            self._validate(expected_timeline_manifest_sha256="c" * 64)

    def test_rejects_payload_digest_mismatch(self) -> None:
        self.document["payload_sha256"] = "a" * 64
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "payload SHA-256 mismatch"):
            self._validate()

    def test_rejects_tensor_bytes_that_differ_from_payload_record(self) -> None:
        self.document["tensors"]["block.img_f32"]["sha256"] = "a" * 64
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "payload range SHA-256 mismatch"):
            self._validate()

    def test_rejects_duplicate_manifest_field(self) -> None:
        raw = self.manifest.read_text()
        self.manifest.write_text(raw[:-2] + ',"schema":"duplicate"}\n')
        with self.assertRaisesRegex(cache.ReferenceCacheError, "duplicate JSON field"):
            self._validate()

    def test_rejects_latent_record_that_does_not_bind_source_bytes(self) -> None:
        alternate = self.case_root / "alternate.safetensors"
        alternate_sha = _write_safetensors_latent(alternate, b"\x01\x00" * (_LATENT_BYTES // 2))
        self.document["reference_latent_artifact_sha256"] = alternate_sha
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "latent record digest differs"):
            cache.validate_reference_cache_manifest(
                self.manifest,
                reference_latent_path=alternate,
                expected_reference_latent_artifact_sha256=alternate_sha,
                expected_timeline_manifest_sha256=_TIMELINE_SHA256,
            )

    def test_rejects_malformed_source_safetensors_header(self) -> None:
        malformed = self.case_root / "malformed.safetensors"
        malformed.write_bytes(struct.pack("<Q", 100) + b"not-json")
        malformed_sha256 = hashlib.sha256(malformed.read_bytes()).hexdigest()
        self.document["reference_latent_artifact_sha256"] = malformed_sha256
        self._write_document()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "header size is invalid"):
            cache.validate_reference_cache_manifest(
                self.manifest,
                reference_latent_path=malformed,
                expected_reference_latent_artifact_sha256=malformed_sha256,
                expected_timeline_manifest_sha256=_TIMELINE_SHA256,
            )


class ReferencePreQKVContractTest(unittest.TestCase):
    def test_order_and_shapes_include_dynamic_valid_text_prefix(self) -> None:
        shapes = cache._expected_preqkv_shapes(529)
        self.assertEqual(tuple(shapes), cache.PREQKV_NAMES)
        self.assertEqual(shapes["img.mod_silu"], (1, 2048))
        self.assertEqual(shapes["img.mod_params"], (1, 12_288))
        self.assertEqual(shapes["img.norm1"], (1, 4320, 2048))
        self.assertEqual(shapes["img.qkv_input"], (1, 4320, 2048))
        self.assertEqual(shapes["txt.mod_silu"], (1, 2048))
        self.assertEqual(shapes["txt.mod_params"], (1, 12_288))
        self.assertEqual(shapes["txt.norm1"], (1, 529, 2048))
        self.assertEqual(shapes["txt.qkv_input"], (1, 529, 2048))

    def test_rejects_invalid_dynamic_text_prefix(self) -> None:
        for extent in (True, 0, 1281, 1.0):
            with self.subTest(extent=extent), self.assertRaisesRegex(
                    cache.ReferenceCacheError, "text extent"):
                cache._expected_preqkv_shapes(extent)

    def test_actual_hook_interleaving_is_canonicalized_before_serialization(self) -> None:
        hook_order = (
            "img.mod_silu", "img.mod_params", "txt.mod_silu", "txt.mod_params",
            "img.norm1", "img.qkv_input", "txt.norm1", "txt.qkv_input",
        )
        capture = {name: name for name in hook_order}
        self.assertNotEqual(tuple(capture), cache.PREQKV_NAMES)
        cache._canonicalize_preqkv_capture(capture)
        self.assertEqual(tuple(capture), cache.PREQKV_NAMES)
        self.assertEqual(capture["txt.qkv_input"], "txt.qkv_input")

    def test_canonicalizer_rejects_missing_and_unexpected_boundaries(self) -> None:
        missing = {name: name for name in cache.PREQKV_NAMES[:-1]}
        with self.assertRaisesRegex(cache.ReferenceCacheError, "inventory mismatch"):
            cache._canonicalize_preqkv_capture(missing)
        unexpected = {name: name for name in cache.PREQKV_NAMES}
        unexpected["extra"] = object()
        with self.assertRaisesRegex(cache.ReferenceCacheError, "inventory mismatch"):
            cache._canonicalize_preqkv_capture(unexpected)

    def test_type_verifier_rejects_missing_or_reordered_boundary(self) -> None:
        shapes = cache._expected_preqkv_shapes(3)
        torch_types = SimpleNamespace(bfloat16="bf16", float32="f32")
        capture = {
            name: SimpleNamespace(shape=shape, dtype="f32")
            for name, shape in shapes.items()
        }
        cache._verify_typed_preqkv_capture(
            capture, torch_types, dtype="f32", text_tokens=3
        )
        missing = dict(capture)
        missing.pop("img.mod_silu")
        with self.assertRaisesRegex(cache.ReferenceCacheError, "actual_inventory"):
            cache._verify_typed_preqkv_capture(
                missing, torch_types, dtype="f32", text_tokens=3
            )

    def test_bf16_mode_uses_mixed_source_boundary_dtypes(self) -> None:
        shapes = cache._expected_preqkv_shapes(3)
        torch_types = SimpleNamespace(bfloat16="bf16", float32="f32")
        capture = {
            name: SimpleNamespace(
                shape=shape,
                dtype=("bf16" if name in cache.PREQKV_BF16_BOUNDARIES else "f32"),
            )
            for name, shape in shapes.items()
        }
        cache._verify_typed_preqkv_capture(
            capture, torch_types, dtype="bf16", text_tokens=3
        )

        capture["img.norm1"] = SimpleNamespace(shape=shapes["img.norm1"], dtype="bf16")
        with self.assertRaisesRegex(
                cache.ReferenceCacheError,
                r"img\.norm1 expected .*actual_inventory=.*img.norm1.*dtype=bf16"):
            cache._verify_typed_preqkv_capture(
                capture, torch_types, dtype="bf16", text_tokens=3
            )

    def test_capture_requires_both_explicit_companion_outputs(self) -> None:
        with self.assertRaisesRegex(cache.ReferenceCacheError, "requires explicit"):
            cache.run_reference_block0_cache_oracle(
                "/unused", reference_latent_path="/unused",
                expected_reference_latent_artifact_sha256="0" * 64,
                expected_timeline_manifest_sha256="1" * 64,
                expected_manifest_sha256="2" * 64,
                output_safetensors="/unused-main.safetensors",
                receipt_path="/unused-main.json", capture_preqkv=True,
                preqkv_output_safetensors="/unused-companion.safetensors",
            )

if __name__ == "__main__":
    unittest.main()
