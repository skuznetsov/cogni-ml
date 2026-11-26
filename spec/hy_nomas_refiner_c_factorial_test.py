#!/usr/bin/env python3
"""CPU-only admission and sensitivity tests for the c/text factorial oracle."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_refiner_c_factorial_reference as factorial  # noqa: E402
import hy_nomas_refiner_transfer_reference as transfer  # noqa: E402


PARENT_SHA = "e" * 64


def _input_record(label: str, digest: str) -> dict:
    return {
        "manifest_path": f"input-{label}.json",
        "manifest_sha256": ("a" if label == "A" else "b") * 64,
        "payload_path": f"input-{label}.bf16le",
        "payload_bytes": transfer.INPUT_BYTES,
        "payload_sha256": digest,
        "tensor_name": "input.qwen_bf16",
        "offset_bytes": 0,
        "nbytes": transfer.INPUT_BYTES,
        "source_tensor_sha256": ("c" if label == "A" else "d") * 64,
        "source_shape": list(transfer.QWEN_SHAPE),
        "source_dtype": "BF16",
        "normalized_tensor_sha256": digest,
    }


def _stats(label: str, elements: int) -> dict:
    return {
        "checkpoint": label,
        "elements": elements,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "rms_abs": 0.0,
        "outside_atol_rtol": 0,
        "all_finite": True,
        "passed": True,
        "atol": factorial.ATOL,
        "rtol": factorial.RTOL,
    }


def _parent_fixture(directory: Path) -> tuple[dict, Path]:
    input_a = _input_record("A", "1" * 64)
    input_b = _input_record("B", "2" * 64)
    tensors = {}
    chunks = []
    cursor = 0
    for case in transfer.CASE_ORDER:
        for checkpoint in transfer.CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            shape = transfer._output_shape(case, checkpoint)
            raw = bytes(transfer._output_nbytes(shape))
            tensors[name] = {
                "dtype": "F32",
                "source_dtype": "BF16",
                "shape": shape,
                "offset_bytes": cursor,
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint_boundary": checkpoint,
            }
            chunks.append(raw)
            cursor += len(raw)
    parent_payload = b"".join(chunks)
    payload_path = directory / "parent.f32le"
    payload_path.write_bytes(parent_payload)
    document = {
        "schema": transfer.SCHEMA,
        "schema_version": transfer.SCHEMA_VERSION,
        "upstream_revision": transfer.PINNED_REVISION,
        "source": {
            "upstream_path": "/cached/upstream",
            "revision": transfer.PINNED_REVISION,
            "source_files": copy.deepcopy(transfer.PINNED_SOURCE_SHA256),
            "text_weights": {
                "path": "/weights/text-time-f32.safetensors",
                "sha256": "3" * 64,
                "bytes": 583316840,
                "manifest_path": "/weights/text-time-f32.safetensors.manifest.json",
                "manifest_sha256": "4" * 64,
                "tensor_count": 42,
                "stored_dtype": "F32",
                "loaded_dtype": "BF16",
                "conversion": "F32_to_BF16_RNE",
            },
            "source_checkpoint": {
                "repository": "tencent/Hy-Nomas",
                "checkpoint_path": "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
                "checkpoint_lfs_sha256": "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
                "checkpoint_lfs_sha256_status": "declared by pinned HF API; whole file not downloaded or verified",
                "source_snapshot_revision": "bac719a80c8d02f91c485fca804bb8221e149b70",
            },
            "module_bindings": {
                "refiner": "hymm.models.text_encoder.token_refiner.SingleTokenRefinerPerBlock",
                "time_embedder": "hymm.models.dit.modules.embed_layers.TimestepEmbedder",
            },
        },
        "geometry": copy.deepcopy(transfer.GEOMETRY),
        "timestep": 1000.0,
        "mask": copy.deepcopy(transfer.MASK_RECORD),
        "action_spans": copy.deepcopy(transfer.ACTION_SPANS),
        "inputs": {
            "cases": {"A0": input_a, "A1": copy.deepcopy(input_a), "B": input_b},
            "prefix_checks": {
                "validated_before_padding_normalization": True,
                "unchanged_rows": [0, 132],
                "changed_rows": [132, 224],
                "action2_changed_bytes": 109584,
                "global_action1_sha256": "5" * 64,
                "A0_A1_same_normalized_qwen_sha256": "1" * 64,
            },
        },
        "normalization": copy.deepcopy(transfer.NORMALIZATION),
        "precision": copy.deepcopy(transfer.PRECISION),
        "execution": {
            **copy.deepcopy(transfer.EXECUTION),
            "source_precision": copy.deepcopy(transfer.SOURCE_PRECISION),
            "source_attention_query_chunks": 2,
            "source_a_a_floor": {
                checkpoint: _stats(f"A0/A1.{checkpoint}", transfer._output_elements(checkpoint))
                for checkpoint in transfer.CHECKPOINTS
            },
        },
        "payload": {
            "basename": payload_path.name,
            "bytes": len(parent_payload),
            "sha256": hashlib.sha256(parent_payload).hexdigest(),
            "encoding": "f32le_bf16_expansion",
        },
        "tensors": tensors,
    }
    transfer.validate_reference_document(document)
    return document, payload_path


def _factorial_fixture(directory: Path, parent: dict) -> tuple[dict, Path]:
    tensors = {}
    chunks = []
    cursor = 0
    for case in factorial.CASE_ORDER:
        for checkpoint in factorial.CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            shape = factorial._record_shape(checkpoint)
            raw = bytes(factorial._elements(checkpoint) * 4)
            tensors[name] = {
                "dtype": "F32",
                "source_dtype": "BF16",
                "shape": shape,
                "offset_bytes": cursor,
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint_boundary": checkpoint,
            }
            chunks.append(raw)
            cursor += len(raw)
    raw_payload = b"".join(chunks)
    payload_path = directory / "child.factorial.f32le"
    payload_path.write_bytes(raw_payload)
    parent_path = (directory / "parent.json").resolve()
    document = {
        "schema": factorial.SCHEMA,
        "schema_version": factorial.SCHEMA_VERSION,
        "upstream_revision": parent["upstream_revision"],
        "parent_reference": {"path": str(parent_path), "sha256": PARENT_SHA},
        "source": copy.deepcopy(parent["source"]),
        "geometry": copy.deepcopy(parent["geometry"]),
        "timestep": parent["timestep"],
        "mask": copy.deepcopy(parent["mask"]),
        "action_spans": copy.deepcopy(parent["action_spans"]),
        "normalization": copy.deepcopy(parent["normalization"]),
        "precision": copy.deepcopy(parent["precision"]),
        "execution": {
            "case_order": list(factorial.CASE_ORDER),
            "scope": "refiner only; no full DiT",
            "source_precision": copy.deepcopy(parent["execution"]["source_precision"]),
            "source_attention_query_chunks": 2,
            "parent_replay": {
                case: {
                    checkpoint: _stats(
                        f"parent.{case}.{checkpoint}", factorial._elements(checkpoint),
                    ) for checkpoint in factorial.CHECKPOINTS
                } for case in factorial.PARENT_CASES
            },
            "source_aa": {
                checkpoint: _stats(f"A0/A1.{checkpoint}", factorial._elements(checkpoint))
                for checkpoint in factorial.CHECKPOINTS
            },
        },
        "case_factors": copy.deepcopy(factorial.FACTORS),
        "payload": {
            "basename": payload_path.name,
            "bytes": len(raw_payload),
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
            "encoding": "f32le_bf16_expansion",
        },
        "tensors": tensors,
    }
    return document, payload_path


def _mutate_tensor(document: dict, path: Path, tensor_name: str, value: float) -> None:
    record = document["tensors"][tensor_name]
    payload = bytearray(path.read_bytes())
    payload[record["offset_bytes"]:record["offset_bytes"] + 4] = struct.pack("<f", value)
    path.write_bytes(payload)
    raw = bytes(payload[record["offset_bytes"]:record["offset_bytes"] + record["nbytes"]])
    record["sha256"] = hashlib.sha256(raw).hexdigest()
    document["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()


class HyNomasRefinerCFactorialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hy-nomas-c-factorial-test-")
        self.directory = Path(self.temp.name)
        self.parent, self.parent_payload = _parent_fixture(self.directory)
        self.document, self.payload = _factorial_fixture(self.directory, self.parent)
        self.parent_path = (self.directory / "parent.json").resolve()

    def tearDown(self):
        self.temp.cleanup()

    def _validate(self):
        factorial.validate_reference_document(
            self.document, self.parent, self.parent_path, PARENT_SHA,
        )

    def _validate_payload(self):
        factorial.validate_reference_payload(
            self.document, self.payload, self.parent, self.parent_payload,
            self.parent_path, PARENT_SHA,
        )

    def test_valid_factorial_document_and_payload_admit(self):
        self._validate()
        self._validate_payload()

    def test_rejects_factor_map_drift_or_missing_case(self):
        self.document["case_factors"]["X10"]["c_case"] = "B"
        with self.assertRaisesRegex(factorial.ReferenceError, "factor map mismatch"):
            self._validate()
        self.document["case_factors"] = copy.deepcopy(factorial.FACTORS)
        del self.document["case_factors"]["X01"]
        with self.assertRaisesRegex(factorial.ReferenceError, "case inventory"):
            self._validate()

    def test_rejects_missing_checkpoint_and_parent_replay_case(self):
        del self.document["tensors"]["X01.block1.output"]
        with self.assertRaisesRegex(factorial.ReferenceError, "tensor inventory"):
            self._validate()
        self.document["tensors"] = _factorial_fixture(self.directory, self.parent)[0]["tensors"]
        del self.document["execution"]["parent_replay"]["B"]
        with self.assertRaisesRegex(factorial.ReferenceError, "case inventory"):
            self._validate()

    def test_rejects_parent_digest_and_incomplete_or_failed_source_aa(self):
        self.document["parent_reference"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(factorial.ReferenceError, "parent reference binding"):
            self._validate()
        self.document["parent_reference"]["sha256"] = PARENT_SHA
        self.document["execution"]["source_aa"] = {}
        with self.assertRaisesRegex(factorial.ReferenceError, "source A/A checkpoint inventory"):
            self._validate()
        self.document["execution"]["source_aa"] = {
            checkpoint: _stats(f"A0/A1.{checkpoint}", factorial._elements(checkpoint))
            for checkpoint in factorial.CHECKPOINTS
        }
        self.document["execution"]["source_aa"]["c"]["passed"] = False
        self.document["execution"]["source_aa"]["c"]["outside_atol_rtol"] = 1
        with self.assertRaisesRegex(factorial.ReferenceError, "failed tolerance"):
            self._validate()

    def test_payload_validator_rejects_seeded_noop_mismatch(self):
        _mutate_tensor(self.document, self.payload, "A1.c", 1.0)
        with self.assertRaisesRegex(factorial.ReferenceError, "source A/A statistics"):
            self._validate_payload()

    def test_payload_validator_rejects_seeded_parent_replay_mismatch(self):
        _mutate_tensor(self.document, self.payload, "B.c", 1.0)
        with self.assertRaisesRegex(factorial.ReferenceError, "parent replay statistics"):
            self._validate_payload()

    def test_comparator_detects_seeded_numeric_mismatch(self):
        import numpy as np

        rng = np.random.default_rng(20261105)
        expected = rng.standard_normal(2048, dtype=np.float32)
        self.assertTrue(transfer.compare_checkpoint("seed", expected, expected.copy())["passed"])
        changed = expected.copy()
        changed[1027] += np.float32(1e-2)
        stats = transfer.compare_checkpoint("seeded-defect", expected, changed)
        self.assertFalse(stats["passed"])
        self.assertGreater(stats["outside_atol_rtol"], 0)


if __name__ == "__main__":
    unittest.main()
