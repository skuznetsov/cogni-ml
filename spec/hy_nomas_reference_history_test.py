#!/usr/bin/env python3
"""CPU-only receipt-contract tests for the standalone history store/read oracle."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))

import hy_nomas_reference_history as history  # noqa: E402


class ReferenceHistoryContractTests(unittest.TestCase):
    def _native_receipt(self, valid_text_tokens: int = 1) -> dict:
        rows = history.IMAGE_TOKENS + valid_text_tokens
        persisted = rows * history.HIDDEN * 4
        read_stage = rows * history.HIDDEN * 8
        accounting = {
            "input_buffers": 101,
            "block_weight_residents": 202,
            "block_scratch": 303,
            "mps_weight_stage": 404,
            "compact_attention_buffers": 505,
            "qk_rms_square_and_mean_buffers": 606,
            "reference_bf16_buffers": 707,
            "reference_history_buffers": rows * history.HIDDEN * 12,
            "opaque_mpsgraph_private_bytes": "not measured; excluded from named Metal-buffer total",
        }
        accounting["total_named_bytes"] = sum(
            value for key, value in accounting.items() if key != "opaque_mpsgraph_private_bytes"
        )
        tensor_records = []
        for name, dtype in zip(history.NATIVE_TENSOR_NAMES, ("BF16", "BF16", "F32", "F32")):
            tensor_records.append({
                "name": name,
                "dtype": dtype,
                "shape": [1, rows, history.HEADS, history.HEAD_DIM],
                "nbytes": rows * history.HIDDEN * (2 if dtype == "BF16" else 4),
                "sha256": "a" * 64,
            })
        return {
            "schema": history.NATIVE_SCHEMA,
            "schema_version": history.SCHEMA_VERSION,
            "native_revision": history.UPSTREAM_REVISION,
            "source_revision": history.SOURCE_REVISION,
            "scope": history.NATIVE_SCOPE,
            "full_upstream_t0_preprocessing_parity": False,
            "history_cache_write_executed": True,
            "history_cache_read_executed": True,
            "attention_executed": False,
            "input_bundle_manifest_sha256": "b" * 64,
            "input_bundle_payload_sha256": "c" * 64,
            "reference_latent_artifact_sha256": "d" * 64,
            "timeline_manifest_sha256": "e" * 64,
            "valid_text_tokens": valid_text_tokens,
            "text_storage_tokens": history.TEXT_STORAGE_TOKENS,
            "weights_sha256": history.BLOCK_WEIGHTS_SHA256,
            "precision_control": history.REFERENCE_BF16_PRECISION,
            "bf16_linear_variant": history.BF16_LINEAR_VARIANT,
            "history": {
                "storage_dtype": "BF16",
                "read_dtype": "F32",
                "layout": history.LAYOUT,
                "cache_valid_len": 0,
                "cache_txt": True,
                "stored_tokens": rows,
                "cache_writes": 1,
                "cache_reads": 1,
                "persisted_bytes": persisted,
                "read_stage_bytes": read_stage,
            },
            "tensor_artifact": {
                "basename": "history.safetensors",
                "sha256": "f" * 64,
                "bytes": sum(record["nbytes"] for record in tensor_records),
                "tensors": tensor_records,
            },
            "named_buffer_accounting": accounting,
            "execution_counters": {
                "encode_allocations": 0,
                "encode_readbacks": 0,
                "preterminal_waits": 0,
                "terminal_waits": 1,
            },
        }

    def _source_history(self, valid_text_tokens: int = 1) -> dict:
        rows = history.IMAGE_TOKENS + valid_text_tokens
        return {
            "storage_dtype": "BF16",
            "read_dtype": "F32",
            "consumer_input_dtype": "BF16",
            "layout": history.LAYOUT,
            "cache_valid_len": 0,
            "cache_txt": True,
            "stored_tokens": rows,
            "cache_writes": 1,
            "cache_reads": 2,
            "persisted_bytes": rows * history.HIDDEN * 4,
            "read_stage_bytes": rows * history.HIDDEN * 8,
            "poisoned_read_only_probe": {
                "executed": True,
                "cache_valid_len": rows,
                "update_cache": False,
                "cache_txt": True,
                "current_qkv_poison": "zeroed in place after T0 store",
                "attention_consumer_intercepted": True,
                "history_prefix_matches_stored": True,
                "current_suffix_matches_poisoned_inputs": True,
                "history_prefix_differs_from_poisoned_current": True,
                "cache_unchanged": True,
                "history_k_f32_sha256": "9" * 64,
                "history_v_f32_sha256": "8" * 64,
            },
        }

    def test_native_t0_contract_admits_exact_store_and_read_scope(self):
        receipt = self._native_receipt()
        self.assertEqual(history._validate_native_receipt_contract(receipt), 4321)

    def test_native_contract_rejects_adapter_only_or_prospective_receipts(self):
        bad_cases = []
        old_receipt = self._native_receipt()
        old_receipt["history_cache_write_executed"] = False
        bad_cases.append(old_receipt)
        no_read = self._native_receipt()
        no_read["history_cache_read_executed"] = False
        bad_cases.append(no_read)
        prospective = self._native_receipt()
        prospective["cache_tensor_semantics"] = "prospective concatenated K/V before any continuation-history append"
        bad_cases.append(prospective)
        for receipt in bad_cases:
            with self.subTest(receipt=receipt):
                with self.assertRaises(history.HistoryProbeError):
                    history._validate_native_receipt_contract(receipt)

    def test_native_contract_rejects_attention_and_accounting_claim_drift(self):
        bad_attention = self._native_receipt()
        bad_attention["attention_executed"] = True
        with self.assertRaises(history.HistoryProbeError):
            history._validate_native_receipt_contract(bad_attention)
        bad_bytes = self._native_receipt()
        bad_bytes["history"]["persisted_bytes"] += 1
        with self.assertRaises(history.HistoryProbeError):
            history._validate_native_receipt_contract(bad_bytes)
        bool_counter = self._native_receipt()
        bool_counter["history"]["cache_writes"] = True
        with self.assertRaises(history.HistoryProbeError):
            history._validate_native_receipt_contract(bool_counter)
        bool_shape = self._native_receipt()
        bool_shape["tensor_artifact"]["tensors"][0]["shape"][0] = True
        with self.assertRaises(history.HistoryProbeError):
            history._validate_native_receipt_contract(bool_shape)
        bad_accounting = self._native_receipt()
        bad_accounting["named_buffer_accounting"]["total_named_bytes"] += 1
        with self.assertRaises(history.HistoryProbeError):
            history._validate_native_receipt_contract(bad_accounting)
        underreported_history = self._native_receipt()
        accounting = underreported_history["named_buffer_accounting"]
        original_history_bytes = accounting["reference_history_buffers"]
        accounting["reference_history_buffers"] = 1
        accounting["total_named_bytes"] += 1 - original_history_bytes
        with self.assertRaises(history.HistoryProbeError):
            history._validate_native_receipt_contract(underreported_history)

    def test_source_probe_requires_a_real_read_only_poisoned_prefix(self):
        valid = self._source_history()
        history._validate_history_semantics(valid, rows=4321, native=False)
        for field, value in (
            ("history_prefix_matches_stored", False),
            ("current_suffix_matches_poisoned_inputs", False),
            ("history_prefix_differs_from_poisoned_current", False),
            ("cache_unchanged", False),
        ):
            candidate = self._source_history()
            candidate["poisoned_read_only_probe"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(history.HistoryProbeError):
                    history._validate_history_semantics(candidate, rows=4321, native=False)

    def test_source_probe_rejects_wrong_history_extent_and_read_count(self):
        bad_extent = self._source_history()
        bad_extent["poisoned_read_only_probe"]["cache_valid_len"] = 4320
        with self.assertRaises(history.HistoryProbeError):
            history._validate_history_semantics(bad_extent, rows=4321, native=False)
        bad_count = self._source_history()
        bad_count["cache_reads"] = 1
        with self.assertRaises(history.HistoryProbeError):
            history._validate_history_semantics(bad_count, rows=4321, native=False)

    def test_comparator_refuses_any_tolerance_change_before_reading_artifacts(self):
        with self.assertRaisesRegex(history.HistoryProbeError, "unchanged atol=rtol=2e-5"):
            history.compare_native_history(
                "/missing/source.json", "/missing/source.safetensors",
                "/missing/native.json", "/missing/native.safetensors",
                atol=2.1e-5,
            )
        with self.assertRaisesRegex(history.HistoryProbeError, "unchanged atol=rtol=2e-5"):
            history.compare_native_history(
                "/missing/source.json", "/missing/source.safetensors",
                "/missing/native.json", "/missing/native.safetensors",
                rtol=1e-5,
            )

    def test_token_extent_rejects_booleans_and_empty_text(self):
        for value in (True, 0, 1281, "1"):
            with self.subTest(value=value):
                with self.assertRaises(history.HistoryProbeError):
                    history._expected_shape(value)


if __name__ == "__main__":
    unittest.main()
