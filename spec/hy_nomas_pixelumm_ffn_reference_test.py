"""CPU-only falsifiers for the saved-attention FFN reference corridor."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_pixelumm_ffn_reference as reference


def parent(full=False):
    count = 21120 if full else 2048
    document = {
        "schema": "cogni_ml.pixelumm_thirty_six_block_reference/v1" if full else "cogni_ml.pixelumm_block_reference/v1",
        "upstream_revision": reference.UPSTREAM_REVISION,
        "upstream_model_sha256": reference.MODEL_SHA,
        "comparison": dict(reference.COMPARISON), "token_order": "tile_major", "tile_tokens": 8,
        "grid": [6, 44, 80] if full else [2, 32, 32],
        "checkpoint_subset_sha256": reference.SUBSET_SHA,
        "payload_sha256": "a" * 64, "payload": "parent.bin",
        "runtime_dtype": "BF16", "full36_capture_policy": "two_bank_survivors",
        "input_provenance": {"kind": "real_latent_xt_pixelumm_stem"},
        "blocks": [{"index": i, "checkpoint_subset_sha256": reference.SUBSET_SHA} for i in range(36)],
        "inputs": {}, "weights": {}, "captures": {},
    }
    cursor = 0

    def record(shape):
        nonlocal cursor
        nbytes = 2
        for n in shape:
            nbytes *= n
        value = {"dtype": "BF16", "shape": shape, "offset": cursor,
                 "nbytes": nbytes, "sha256": "b" * 64}
        cursor += nbytes
        return value

    prefix = "block0." if full else ""
    document["inputs"]["x"] = record([1, count, 4096])
    document["captures"][prefix + "attention"] = record([1, count, 32, 128])
    for name, shape in reference.WEIGHTS.items():
        document["weights"][prefix + name] = record(shape)
    document["payload_bytes"] = cursor
    return document


class FFNReferenceTests(unittest.TestCase):
    def test_true_small_and_full_rows_are_retained(self):
        for full, count in ((False, 2048), (True, 21120)):
            actual, records = reference.validate_parent(parent(full))
            self.assertEqual(actual, count)
            self.assertEqual(records["x"]["shape"], [1, count, 4096])
            self.assertEqual(len(records), 7)

    def test_full_grid_rejects_repeated_small_capture(self):
        value = parent(True)
        value["captures"]["block0.attention"]["shape"][1] = 2048
        with self.assertRaises(reference.FFNReferenceError):
            reference.validate_parent(value)

    def test_tolerance_source_and_layout_do_not_widen(self):
        for key, replacement in (("comparison", {"atol": 1e-3, "rtol": 2e-5}),
                                 ("upstream_revision", "wrong"),
                                 ("upstream_model_sha256", "a" * 64),
                                 ("token_order", "thw"), ("tile_tokens", 16)):
            value = parent(True)
            value[key] = replacement
            with self.assertRaises(reference.FFNReferenceError):
                reference.validate_parent(value)

    def test_full_real_input_and_block_zero_identity_required(self):
        original = parent(True)
        for key in ("input_provenance", "blocks", "full36_capture_policy"):
            value = copy.deepcopy(original)
            del value[key]
            with self.assertRaises(reference.FFNReferenceError):
                reference.validate_parent(value)
        original["blocks"][0]["checkpoint_subset_sha256"] = "a" * 64
        with self.assertRaises(reference.FFNReferenceError):
            reference.validate_parent(original)

    def test_record_extent_overlap_hash_and_precision_rejected(self):
        for key, replacement in (("offset", 0), ("nbytes", 1), ("sha256", "z" * 64),
                                 ("dtype", "F16"), ("shape", [1, 2048, 4096])):
            value = parent(True)
            value["weights"]["block0.mlp.gate_proj.weight"][key] = replacement
            with self.assertRaises(reference.FFNReferenceError):
                reference.validate_parent(value)

    def test_metadata_identity_fails_before_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.json"
            path.write_text(json.dumps(parent()))
            with self.assertRaisesRegex(reference.FFNReferenceError, "trusted metadata"):
                reference.load_parent(path, "a" * 64)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            value = parent()
            value["payload"] = "../escape.bin"
            path.write_text(json.dumps(value))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(reference.FFNReferenceError, "sibling"):
                reference.load_parent(path, digest)

    def test_metadata_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.json"
            path.write_text(json.dumps(parent()))
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(reference.FFNReferenceError, "regular"):
                reference.load_parent(link, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
