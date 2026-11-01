"""CPU falsifiers for the reduced GPU segment-noise fork probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_segment_edit_reference as oracle


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fork_contract_sha() -> str:
    base_sha = _sha(oracle.BASE_CONTRACT.encode())
    document = {
        "schema": "cogni_ml.hy_nomas_edited_sequence_boundary/v1",
        "base_contract_sha256": base_sha,
        "noise_edit": oracle.NOISE_EDIT,
    }
    raw = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
    return _sha(raw)


class SegmentEditReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.receipt_path = self.root / "receipt.json"
        self.write_valid_case()

    def write_artifact(self, name: str, payload: bytes) -> dict:
        filename, elements, shape = oracle.ARTIFACTS[name]
        self.assertEqual(len(payload), elements * oracle.F32_BYTES)
        (self.root / filename).write_bytes(payload)
        return {
            "path": filename,
            "sha256": _sha(payload),
            "bytes": len(payload),
            "shape": shape,
        }

    def write_receipt(self):
        self.receipt_path.write_text(
            json.dumps(self.receipt, separators=(",", ":")), encoding="utf-8"
        )

    def rewrite_artifact(self, name: str, payload: bytes):
        self.receipt["artifacts"][name] = self.write_artifact(name, payload)
        self.write_receipt()

    def write_valid_case(self):
        initial = oracle._initial_timeline()
        after_a = oracle._expected_after_a(initial)
        artifacts = {
            "after_a_timeline": after_a,
            "timeline": oracle._expected_after_b(after_a),
            "b_noise": oracle._gather_segment(initial, oracle.SOURCE_SEGMENT),
            "b_cosine": oracle._rope_slice(
                oracle._rope_bank("cosine"), oracle.TARGET_SEGMENT
            ),
            "b_sine": oracle._rope_slice(
                oracle._rope_bank("sine"), oracle.TARGET_SEGMENT
            ),
            "c_noise": oracle._gather_segment(initial, 2),
            "c_cosine": oracle._rope_slice(oracle._rope_bank("cosine"), 2),
            "c_sine": oracle._rope_slice(oracle._rope_bank("sine"), 2),
        }
        self.receipt = {
            "schema": oracle.RECEIPT_SCHEMA,
            "upstream": oracle.UPSTREAM,
            "scope": (
                "synthetic GPU after-A boundary, B-from-F gather/scatter, and C prepare; "
                "no neural model or decoder"
            ),
            "resident_segments": oracle.RESIDENT_SEGMENTS,
            "run_segments": oracle.RUN_SEGMENTS,
            "checkpoint_boundary_next_segment": oracle.TARGET_SEGMENT,
            "target_segment": oracle.TARGET_SEGMENT,
            "source_segment": oracle.SOURCE_SEGMENT,
            "noise_edit": oracle.NOISE_EDIT,
            "noise_edit_identity_sha256": _sha(json.dumps(
                oracle.NOISE_EDIT, separators=(",", ":"), ensure_ascii=False
            ).encode()),
            "boundary_contract_sha256": _sha(oracle.BASE_CONTRACT.encode()),
            "fork_contract_sha256": _fork_contract_sha(),
            "checks": dict(oracle.CHECKS),
            "metrics": dict(oracle.METRICS),
            "artifacts": {
                name: self.write_artifact(name, payload)
                for name, payload in artifacts.items()
            },
            "passed": True,
        }
        self.write_receipt()

    def test_valid_receipt_authenticates_b_fork_c_continuation_and_lineage(self):
        report = oracle.validate_probe_directory(self.root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["segments"]["target"], 1)
        self.assertEqual(report["segments"]["noise_source"], 5)
        self.assertEqual(report["segments"]["resident"], 6)
        self.assertEqual(report["checks"]["clean_a_retained"], True)
        self.assertEqual(report["checks"]["target_b_scatter_only_to_b"], True)
        self.assertEqual(report["checks"]["c_noise_uses_c"], True)
        self.assertEqual(report["metrics"]["terminal_waits"], 1)

    def test_b_noise_must_come_from_f_not_original_b(self):
        initial = oracle._initial_timeline()
        self.rewrite_artifact(
            "b_noise", oracle._gather_segment(initial, oracle.TARGET_SEGMENT)
        )
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "F donor"):
            oracle.validate_probe_directory(self.root)

    def test_b_rope_must_remain_target_rope(self):
        self.rewrite_artifact(
            "b_cosine",
            oracle._rope_slice(oracle._rope_bank("cosine"), oracle.SOURCE_SEGMENT),
        )
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "global RoPE"):
            oracle.validate_probe_directory(self.root)

    def test_b_scatter_must_preserve_c_and_future_suffix(self):
        timeline = bytearray(oracle._expected_after_b(
            oracle._expected_after_a(oracle._initial_timeline())
        ))
        # Change one future F value and update the hash to isolate scatter
        # semantics from transport integrity.
        index = (oracle.SOURCE_SEGMENT * oracle.SEGMENT_FRAMES * oracle.SPATIAL) * oracle.F32_BYTES
        struct.pack_into("<f", timeline, index, 123.0)
        self.rewrite_artifact("timeline", bytes(timeline))
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "scatter regenerated B only"):
            oracle.validate_probe_directory(self.root)

    def test_after_a_snapshot_is_a_separate_exact_prefix_artifact(self):
        after_a = bytearray(oracle._expected_after_a(oracle._initial_timeline()))
        struct.pack_into("<f", after_a, 0, 123.0)
        self.rewrite_artifact("after_a_timeline", bytes(after_a))
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "after-A snapshot"):
            oracle.validate_probe_directory(self.root)

    def test_c_noise_must_use_its_own_timeline_slice(self):
        self.rewrite_artifact(
            "c_noise", oracle._gather_segment(oracle._initial_timeline(), oracle.SOURCE_SEGMENT)
        )
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "C must gather"):
            oracle.validate_probe_directory(self.root)

    def test_c_rope_must_use_global_c_slice(self):
        self.rewrite_artifact(
            "c_sine", oracle._rope_slice(oracle._rope_bank("sine"), oracle.TARGET_SEGMENT)
        )
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "C prepare"):
            oracle.validate_probe_directory(self.root)

    def test_nonzero_preterminal_wait_is_rejected(self):
        self.receipt["metrics"]["preterminal_waits"] = 1
        self.write_receipt()
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "metrics"):
            oracle.validate_probe_directory(self.root)

    def test_missing_fork_identity_is_rejected(self):
        self.receipt["fork_contract_sha256"] = self.receipt["boundary_contract_sha256"]
        self.write_receipt()
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "fork lineage"):
            oracle.validate_probe_directory(self.root)

    def test_artifact_hash_mismatch_is_rejected(self):
        path = self.root / oracle.ARTIFACTS["b_noise"][0]
        path.write_bytes(path.read_bytes()[:-4] + b"\0\0\0\0")
        with self.assertRaisesRegex(oracle.SegmentEditReferenceError, "SHA-256"):
            oracle.validate_probe_directory(self.root)


if __name__ == "__main__":
    unittest.main()
