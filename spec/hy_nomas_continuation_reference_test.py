#!/usr/bin/env python3
"""CPU-only checks for the pinned Hy-Nomas continuation cache oracle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_continuation_reference as oracle


def _write_f32(path: Path, tensor: torch.Tensor) -> str:
    tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContinuationReferenceTests(unittest.TestCase):
    def _report(self, root: Path) -> dict:
        shapes = {
            "query": oracle.INPUT_SHAPE,
            "key": oracle.INPUT_SHAPE,
            "value": oracle.INPUT_SHAPE,
            "output": oracle.OUTPUT_SHAPE,
            "cache_key": oracle.CACHE_SHAPE,
            "cache_value": oracle.CACHE_SHAPE,
        }
        artifacts = {}
        for name, shape in shapes.items():
            path = root / f"{name}.f32le"
            tensor = torch.zeros(shape, dtype=torch.float32)
            digest = _write_f32(path, tensor)
            artifacts[name] = {
                "path": path.name,
                "sha256": digest,
                "shape": list(shape),
            }
        return {
            "schema": oracle.NATIVE_REPORT_SCHEMA,
            "schema_version": 2,
            "segments": 3,
            "layers": 54,
            "denoise_forwards_per_segment": 4,
            "writer_forwards": 2,
            "forward_count": 14,
            "video_tokens": 2,
            "valid_text_tokens": 1,
            "active_query_rows": 3,
            "storage_rows": 4,
            "heads": 2,
            "head_dim": 2,
            "text_cache_policy": "once_per_scene",
            "artifacts": artifacts,
        }

    def test_native_report_authenticates_exact_artifact_geometry_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "native.json"
            document = self._report(root)
            admitted = oracle.validate_native_report_document(document, report_path)
            self.assertEqual(set(admitted["artifacts"]), set(oracle.ARTIFACT_SHAPES))

            malformed = json.loads(json.dumps(document))
            malformed["text_cache_policy"] = "per_segment"
            with self.assertRaisesRegex(oracle.ReferenceError, "text_cache_policy"):
                oracle.validate_native_report_document(malformed, report_path)

            malformed = json.loads(json.dumps(document))
            malformed["artifacts"]["query"]["shape"][-2] = 3
            with self.assertRaisesRegex(oracle.ReferenceError, "query.*shape"):
                oracle.validate_native_report_document(malformed, report_path)

            malformed = json.loads(json.dumps(document))
            malformed["artifacts"]["value"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(oracle.ReferenceError, "value.*SHA-256"):
                oracle.validate_native_report_document(malformed, report_path)

    def test_native_report_rejects_artifact_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._report(root)
            document["artifacts"]["query"]["path"] = "../query.f32le"
            with self.assertRaisesRegex(oracle.ReferenceError, "sibling basename"):
                oracle.validate_native_report_document(document, root / "native.json")

    def test_schedule_caches_scene_text_once_and_keeps_writer_text_transient(self):
        attention, source = oracle.load_upstream_attention()
        self.assertEqual(source["revision"], oracle.UPSTREAM_REVISION)

        # Zero Q makes each SDPA result the mean of visible V rows. Unique per-
        # forward text values distinguish missing, duplicated, or persistent
        # later text; the fourth row poisons accidental padded-row inclusion.
        query = torch.zeros((14, 4, 1, 1), dtype=torch.float32)
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        for forward in range(14):
            key[forward, 0, 0, 0] = 10 + forward
            key[forward, 1, 0, 0] = 20 + forward
            key[forward, 2, 0, 0] = 100 + forward
            key[forward, 3, 0, 0] = 999_999
            value[forward, 0, 0, 0] = 100 + forward
            value[forward, 1, 0, 0] = 200 + forward
            value[forward, 2, 0, 0] = 1_000 + forward
            value[forward, 3, 0, 0] = 999_999

        outputs, final_key, final_value = oracle.run_one_layer(
            query, key, value, attention, device="cpu"
        )
        self.assertEqual(tuple(outputs.shape), (14, 3, 1, 1))
        self.assertEqual(tuple(final_key.shape), (5, 1, 1))
        self.assertEqual(tuple(final_value.shape), (5, 1, 1))

        # First writer sees video+text. Later denoise reads cached scene text
        # plus current video. The second writer attends current text once but
        # appends only current video. The final chunk is read-only.
        self.assertAlmostEqual(outputs[4, 0, 0, 0].item(), 1300 / 3 + 4, places=4)
        self.assertAlmostEqual(outputs[5, 2, 0, 0].item(), 1622 / 5, places=4)
        self.assertAlmostEqual(outputs[9, 2, 0, 0].item(), 2639 / 6, places=4)
        self.assertAlmostEqual(outputs[10, 2, 0, 0].item(), 1950 / 7, places=4)
        self.assertEqual(
            final_key[:, 0, 0].tolist(), [14.0, 24.0, 104.0, 19.0, 29.0]
        )
        self.assertEqual(
            final_value[:, 0, 0].tolist(), [104.0, 204.0, 1004.0, 109.0, 209.0]
        )

    def test_pinned_reference_source_is_exact_and_has_cpu_torch_path(self):
        attention, metadata = oracle.load_upstream_attention()
        self.assertTrue(callable(attention))
        self.assertEqual(metadata["source_sha256"], oracle.UPSTREAM_SOURCE_SHA256)
        self.assertEqual(metadata["cache_storage_sha256"], oracle.CACHE_STORAGE_SHA256)


if __name__ == "__main__":
    unittest.main()
