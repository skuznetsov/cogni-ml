"""CPU falsifiers for admission of the real first continuation boundary."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_boundary_attention_reference as oracle


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BoundaryCaptureAdmissionTests(unittest.TestCase):
    def _fixture(self, directory: str):
        # Preserve the actual tensor inventory; shrink only tensor extents so
        # the malformed-input tests require no models, Torch, or GPU work.
        root = Path(directory)
        shapes = {name: (1,) for name in oracle._SHAPES}
        data, records = bytearray(), {}
        for name in shapes:
            value = b"\0" * 4
            records[name] = {"dtype": "F32", "shape": [1], "offset": len(data),
                             "nbytes": 4, "sha256": sha(value)}
            data.extend(value)
        payload = root / "capture.tensors.bin"
        payload.write_bytes(data)
        capture = {key: "unused" for key in oracle._CAPTURE_TOP_KEYS}
        capture.update({
            "schema": oracle.CAPTURE_SCHEMA,
            "upstream_revision": oracle.PINNED_REVISION,
            "geometry": copy.deepcopy(oracle.GEOMETRY),
            "conditioning_sha256": oracle.CONDITIONING_SHA256,
            "preview_receipt_sha256": oracle.RECEIPT_SHA256,
            "preview_latent_bank_sha256": oracle.LATENT_BANK_SHA256,
            "payload": payload.name, "payload_bytes": len(data),
            "payload_sha256": sha(data), "tensors": records,
        })
        return root / "capture.json", capture, shapes

    def _validate(self, path, capture, shapes, digest=None):
        raw = json.dumps(capture).encode()
        path.write_bytes(raw)
        with mock.patch.object(oracle, "_SHAPES", shapes):
            return oracle._validate_capture(capture, path, digest or sha(raw))

    def test_exact_inventory_and_authenticated_extent_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, capture, shapes = self._fixture(directory)
            payload, records = self._validate(path, capture, shapes)
            self.assertEqual(payload.name, "capture.tensors.bin")
            self.assertEqual(set(records), set(shapes))

    def test_changed_history_geometry_and_saved_prefix_identity_rejected(self):
        changes = (
            lambda c: c["geometry"].update(current_rows=4877),
            lambda c: c["geometry"].update(history_rows=4320),
            lambda c: c.update(preview_latent_bank_sha256="0" * 64),
            lambda c: c.update(conditioning_sha256="0" * 64),
        )
        with tempfile.TemporaryDirectory() as directory:
            path, original, shapes = self._fixture(directory)
            for change in changes:
                capture = copy.deepcopy(original)
                change(capture)
                with self.subTest(change=change), self.assertRaises(oracle.BoundaryReferenceError):
                    self._validate(path, capture, shapes)

    def test_wrong_manifest_digest_and_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, capture, shapes = self._fixture(directory)
            with self.assertRaises(oracle.BoundaryReferenceError):
                self._validate(path, capture, shapes, "0" * 64)
            capture["tensors"].pop("history.value")
            with self.assertRaises(oracle.BoundaryReferenceError):
                self._validate(path, capture, shapes)

    def test_overlapping_ranges_and_tensor_digest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, original, shapes = self._fixture(directory)
            for field, value in (("offset", 0), ("sha256", "0" * 64)):
                capture = copy.deepcopy(original)
                capture["tensors"]["history.key"][field] = value
                with self.subTest(field=field), self.assertRaises(oracle.BoundaryReferenceError):
                    self._validate(path, capture, shapes)

    def test_payload_changes_and_trailing_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, capture, shapes = self._fixture(directory)
            payload = path.with_name(capture["payload"])
            original = payload.read_bytes()
            for altered in (b"x" + original[1:], original + b"x", original[:-1]):
                payload.write_bytes(altered)
                with self.subTest(size=len(altered)), self.assertRaises(oracle.BoundaryReferenceError):
                    self._validate(path, capture, shapes)

    def test_nested_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(oracle.BoundaryReferenceError, "duplicate JSON key"):
            oracle._json_no_duplicates(b'{"history":{"rows":4320,"rows":4877}}', "capture")


if __name__ == "__main__":
    unittest.main()
