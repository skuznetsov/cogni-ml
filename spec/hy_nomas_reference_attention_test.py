#!/usr/bin/env python3
"""CPU-only contract and numeric tests for the first-T0 attention oracle."""
from __future__ import annotations

import hashlib
import json
import ast
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))

import hy_nomas_reference_attention as attention  # noqa: E402
import hy_nomas_reference_block0_compare as block0_compare  # noqa: E402


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
    header["__metadata__"] = metadata or {"format": "test"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-8 - len(encoded)) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _name, _dtype, _shape, raw in records:
            stream.write(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReferenceAttentionTests(unittest.TestCase):
    def _receipt(self, *, source: bool) -> dict:
        schema = attention.SOURCE_SCHEMA if source else attention.NATIVE_SCHEMA
        attention_inputs = [
            {
                "name": name,
                "dtype": "BF16",
                "shape": list(attention.OUTPUT_SHAPE),
                "nbytes": attention.ELEMENTS * 2,
                "sha256": char * 64,
            }
            for name, char in zip(attention.INPUT_TENSOR_NAMES, ("a", "b", "c"))
        ]
        output_dtype = "BF16" if source else "F32"
        tensors = [
            {
                "name": name,
                "dtype": dtype,
                "shape": list(attention.OUTPUT_SHAPE),
                "nbytes": attention.ELEMENTS * (2 if dtype == "BF16" else 4),
                "sha256": char * 64,
            }
            for name, dtype, char in zip(
                attention.TENSOR_NAMES,
                ("BF16", "BF16", "BF16", output_dtype),
                ("a", "b", "c", "d"),
            )
        ]
        document = {
            "schema": schema,
            "schema_version": attention.SCHEMA_VERSION,
            "upstream_revision": attention.UPSTREAM_REVISION,
            "scope": attention.SOURCE_SCOPE if source else attention.NATIVE_SCOPE,
            "full_upstream_t0_preprocessing_parity": False,
            "attention_executed": True,
            "cache_append_executed": False,
            "device": "mps",
            "input_fixtures": {
                "source_input_receipt_sha256": "1" * 64,
                "source_input_artifact_sha256": "2" * 64,
                "source_history_receipt_sha256": "3" * 64,
                "source_history_artifact_sha256": "4" * 64,
            },
            "attention": {
                "query_tokens": attention.TOKENS,
                "kv_tokens": attention.TOKENS,
                "heads": attention.HEADS,
                "head_dim": attention.HEAD_DIM,
                "input_dtype": "BF16",
                "output_dtype": "BF16" if source else output_dtype,
                "layout": attention.LAYOUT,
                "attention_mask": "none",
                "dropout_p": 0.0,
                "is_causal": False,
            },
            "attention_inputs": attention_inputs,
            "tensor_artifact": {
                "basename": "attention.safetensors",
                "sha256": "e" * 64,
                "bytes": sum(record["nbytes"] for record in tensors) + 512,
                "tensors": tensors,
            },
        }
        if source:
            document.update({
                "upstream_source_file_sha256": attention.UPSTREAM_SOURCE_FILE_SHA256,
                "cache_function_source_sha256": attention.CACHE_FUNCTION_SOURCE_SHA256,
                "torch_branch_body_ast_sha256": attention.TORCH_BRANCH_BODY_AST_SHA256,
                "qkv_fixture_provenance": attention.TRUSTED_FIXTURE_PROVENANCE,
                "device": "mps",
                "cpu_fallback_disabled": True,
                "attention_backend": "torch.nn.functional.scaled_dot_product_attention",
            })
        else:
            accounting = {
                "input_buffers": 1,
                "block_weight_residents": 1,
                "block_scratch": 1,
                "mps_weight_stage": 1,
                "compact_attention_buffers": 1,
                "qk_rms_square_and_mean_buffers": 1,
                "reference_bf16_buffers": 1,
                "reference_history_buffers": attention.ELEMENTS * 12,
                "reference_attention_buffers": attention.ELEMENTS * 4,
            }
            accounting["total_named_bytes"] = sum(accounting.values())
            document.update({
                "native_revision": "2e95375d" + "0" * 32,
                "source_attention_receipt_sha256": "5" * 64,
                "attention_backend": "native_mps",
                "named_buffer_accounting": accounting,
                "encode_allocations": 0,
                "encode_readbacks": 0,
                "preterminal_waits": 0,
                "terminal_waits": 1,
            })
        return document

    def test_pinned_torch_branch_is_extracted_from_actual_source_ast(self):
        upstream_root = Path(attention.history.cache_oracle.UPSTREAM_ROOT_DEFAULT)
        source_file = upstream_root / "hymm/models/dit/nomas_cpt.py"
        if not source_file.exists() and not source_file.is_symlink():
            self.skipTest("external pinned upstream source fixture is absent")
        module, identity = attention._load_torch_attention_branch(
            upstream_root
        )
        self.assertEqual(identity["torch_branch_body_ast_sha256"],
                         attention.TORCH_BRANCH_BODY_AST_SHA256)
        self.assertIn("scaled_dot_product_attention", ast_dump := ast.dump(
            module, include_attributes=False
        ))
        self.assertIn("dropout_p=0.0", ast.unparse(module))
        self.assertIn("is_causal=False", ast.unparse(module))

    def test_source_receipt_contract_rejects_scope_and_ast_tampering(self):
        valid = self._receipt(source=True)
        attention._validate_attention_receipt(valid, source=True)

        bad_scope = self._receipt(source=True)
        bad_scope["full_upstream_t0_preprocessing_parity"] = True
        with self.assertRaises(attention.AttentionProbeError):
            attention._validate_attention_receipt(bad_scope, source=True)

        bad_ast = self._receipt(source=True)
        bad_ast["torch_branch_body_ast_sha256"] = "0" * 64
        with self.assertRaises(attention.AttentionProbeError):
            attention._validate_attention_receipt(bad_ast, source=True)

        bad_hash = self._receipt(source=True)
        bad_hash["attention_inputs"][0]["sha256"] = "0" * 64
        with self.assertRaises(attention.AttentionProbeError):
            attention._validate_attention_receipt(bad_hash, source=True)

    def test_receipts_reject_broad_scope_and_padded_5600_row_geometry(self):
        old_scopes = {
            True: (
                "pinned upstream torch attention branch on MPS using a trusted source block-0 "
                "BF16 preattention Q fixture and physical first-T0 BF16 stored K/V; no cache "
                "append or full upstream T0 handoff"
            ),
            False: (
                "one native block-0 attention operation using the same Q and physical first-T0 "
                "BF16 stored K/V; no cache append, full upstream T0 handoff, rest block, other "
                "layers, T1, or generation"
            ),
        }
        self.assertEqual(
            attention.SOURCE_SCOPE,
            "isolated active-prefix invocation of the pinned upstream torch attention branch on MPS "
            "using trusted block-0 BF16 Q and physical first-T0 BF16 stored K/V; 4849 active rows; "
            "no padded Torch, Flash3, cache append, or full upstream T0 handoff parity",
        )
        self.assertEqual(
            attention.NATIVE_SCOPE,
            "one native block-0 active-prefix attention operation using the same Q and physical "
            "first-T0 BF16 stored K/V; 4849 active rows; no padded Torch, Flash3, cache append, "
            "full upstream T0 handoff, rest block, other layers, T1, or generation parity",
        )
        for source, old_scope in old_scopes.items():
            broad = self._receipt(source=source)
            broad["scope"] = old_scope
            with self.assertRaises(attention.AttentionProbeError):
                attention._validate_attention_receipt(broad, source=source)

            padded = self._receipt(source=source)
            padded_elements = 5600 * attention.HEADS * attention.HEAD_DIM
            padded["attention"]["query_tokens"] = 5600
            padded["attention"]["kv_tokens"] = 5600
            for record in padded["attention_inputs"]:
                record["shape"] = [1, 5600, attention.HEADS, attention.HEAD_DIM]
                record["nbytes"] = padded_elements * 2
            for record in padded["tensor_artifact"]["tensors"]:
                record["shape"] = [1, 5600, attention.HEADS, attention.HEAD_DIM]
                record["nbytes"] = padded_elements * (2 if record["dtype"] == "BF16" else 4)
            padded["tensor_artifact"]["bytes"] = sum(
                record["nbytes"] for record in padded["tensor_artifact"]["tensors"]
            ) + 512
            with self.assertRaises(attention.AttentionProbeError):
                attention._validate_attention_receipt(padded, source=source)

    def test_native_receipt_requires_exact_reference_qkv_identity(self):
        valid = self._receipt(source=False)
        attention._validate_attention_receipt(valid, source=False)

        bad_q = self._receipt(source=False)
        bad_q["attention_inputs"][0]["dtype"] = "F32"
        with self.assertRaises(attention.AttentionProbeError):
            attention._validate_attention_receipt(bad_q, source=False)

        bool_shape = self._receipt(source=False)
        bool_shape["attention_inputs"][0]["shape"][0] = True
        with self.assertRaises(attention.AttentionProbeError):
            attention._validate_attention_receipt(bool_shape, source=False)

        short_artifact = self._receipt(source=False)
        short_artifact["tensor_artifact"]["bytes"] = sum(
            record["nbytes"] for record in short_artifact["tensor_artifact"]["tensors"]
        )
        with self.assertRaises(attention.AttentionProbeError):
            attention._validate_attention_receipt(short_artifact, source=False)

        bad_counters = self._receipt(source=False)
        bad_counters["terminal_waits"] = 0
        with self.assertRaises(attention.AttentionProbeError):
            attention._validate_attention_receipt(bad_counters, source=False)

        bad_output = self._receipt(source=False)
        bad_output["attention"]["output_dtype"] = "F16"
        with self.assertRaises(attention.AttentionProbeError):
            attention._validate_attention_receipt(bad_output, source=False)

    def test_numeric_compare_admits_only_fixed_tolerance_known_values(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected_path = directory / "expected.safetensors"
            candidate_path = directory / "candidate.safetensors"
            expected_sha = write_safetensors(expected_path, [
                ("attention.output_bf16", "BF16", (2,), struct.pack("<HH", 0, 0x3F80)),
            ])
            candidate_sha = write_safetensors(candidate_path, [
                ("attention.output_bf16", "F32", (2,), struct.pack("<2f", 2 ** -20, 1.0)),
            ])
            expected = block0_compare.TensorArtifact(
                expected_path, expected_sha256=expected_sha,
                expected_names=("attention.output_bf16",),
            )
            candidate = block0_compare.TensorArtifact(
                candidate_path, expected_sha256=candidate_sha,
                expected_names=("attention.output_bf16",),
            )
            result = attention._compare_output_tensor(expected, candidate)
            self.assertTrue(result["passed"])
            self.assertEqual(result["mismatch_count"], 0)

            bf16_path = directory / "candidate-bf16.safetensors"
            bf16_sha = write_safetensors(bf16_path, [
                ("attention.output_bf16", "BF16", (2,), struct.pack("<HH", 0, 0x3F80)),
            ])
            bf16_candidate = block0_compare.TensorArtifact(
                bf16_path, expected_sha256=bf16_sha,
                expected_names=("attention.output_bf16",),
            )
            bf16_result = attention._compare_output_tensor(expected, bf16_candidate)
            self.assertTrue(bf16_result["passed"])
            self.assertEqual(bf16_result["mismatch_count"], 0)

            bad_path = directory / "bad.safetensors"
            bad_sha = write_safetensors(bad_path, [
                ("attention.output_bf16", "F32", (2,), struct.pack("<2f", 2 ** -20, 0.01)),
            ])
            bad = block0_compare.TensorArtifact(
                bad_path, expected_sha256=bad_sha,
                expected_names=("attention.output_bf16",),
            )
            failed = attention._compare_output_tensor(expected, bad)
            self.assertFalse(failed["passed"])
            self.assertEqual(failed["mismatch_count"], 1)

    def test_f32_candidate_output_must_be_exact_bf16_expansion(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-expansion.safetensors"
            digest = write_safetensors(path, [
                ("attention.output_bf16", "F32", (1,), struct.pack("<f", 1.0001)),
            ])
            artifact = block0_compare.TensorArtifact(
                path, expected_sha256=digest,
                expected_names=("attention.output_bf16",),
            )
            with self.assertRaises(attention.AttentionProbeError):
                attention._validate_output_representation(artifact, source=False)


if __name__ == "__main__":
    unittest.main()
