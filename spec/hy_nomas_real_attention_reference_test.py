"""CPU-only falsifiers for the real-input Hy-Nomas diagnostic admission."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_real_attention_reference as oracle


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RealAttentionAdmissionTests(unittest.TestCase):
    def test_comparison_summary_preserves_integer_counts_above_f32_precision(self):
        class Tensor:
            def __init__(self, value):
                self.value = value

            def __truediv__(self, divisor):
                return Tensor(self.value / divisor)

            def sqrt(self):
                return Tensor(math.sqrt(self.value))

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.value

            def item(self):
                return self.value

        class Torch:
            @staticmethod
            def stack(values):
                return Tensor([value.value for value in values])

        comparison = oracle._Comparison.__new__(oracle._Comparison)
        comparison.torch = Torch
        comparison.name = "large_ffn"
        comparison.expected_count = comparison.seen = 40_000_000
        comparison.max_abs = Tensor(0.125)
        comparison.sum_abs = comparison.sum_sq = comparison.max_rel = Tensor(1.0)
        comparison.bad_count = Tensor(2**24 + 3)
        comparison.first_bad = Tensor(2**25 + 5)
        comparison.all_finite = Tensor(True)
        result = comparison.finish()
        self.assertEqual(result["outside_atol_rtol"], 2**24 + 3)
        self.assertEqual(result["first_failure_flat_index"], 2**25 + 5)
        self.assertFalse(result["allclose_atol_2e-5_rtol_2e-5"])

    def test_bf16_provenance_rounding_covers_ties_signs_and_subnormals(self):
        words = (0, 0x80000000, 0x3F808000, 0x3F818000,
                 0xBF808000, 0xBF818000, 0x3F807FFF, 0x3F808001,
                 0x00008000, 0x00008001, 0x80008000, 0x80008001)
        expected = (0, 0x80000000, 0x3F800000, 0x3F820000,
                    0xBF800000, 0xBF820000, 0x3F800000, 0x3F810000,
                    0, 0x00010000, 0x80000000, 0x80010000)
        source = struct.pack("<12I", *words)
        self.assertEqual(oracle._bf16_rne_roundtrip_f32_bytes(source),
                         struct.pack("<12I", *expected))

    def test_bf16_provenance_rejects_nonfinite_overflow_and_partial_words(self):
        for word in (0x7F800000, 0xFF800000, 0x7FC00000, 0x7F7FFFFF):
            with self.subTest(word=word), self.assertRaises(oracle.ReferenceError):
                oracle._bf16_rne_roundtrip_f32_bytes(struct.pack("<I", word))
        with self.assertRaises(oracle.ReferenceError):
            oracle._bf16_rne_roundtrip_f32_bytes(b"\0")

    def test_latent_prefix_is_sliced_per_channel_before_rounding(self):
        # A flat prefix would contain channel 0's unused future frames and
        # omit the other channels. Mark them differently to reject that bug.
        frames, prefix_frames, spatial = 180, 30, 9 * 16
        per_channel = []
        expected = []
        for channel in range(32):
            word = struct.pack("<f", float(channel + 1))
            valid = word * (prefix_frames * spatial)
            per_channel.append(valid + struct.pack("<f", -10.0) *
                               ((frames - prefix_frames) * spatial))
            expected.append(valid)
        bank = b"".join(per_channel)
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "bank.bin"
            payload.write_bytes(b"offset!!" + bank)
            record = {"dtype": "F32", "shape": [1, 32, 180, 9, 16],
                      "offset": 8, "nbytes": len(bank), "sha256": sha(bank)}
            self.assertEqual(oracle._first_segment_latent_roundtrip_sha256(payload, record),
                             sha(b"".join(expected)))
            for change in ({"dtype": "BF16"}, {"shape": [1, 180, 32, 9, 16]},
                           {"offset": 9}, {"nbytes": len(bank) - 4}):
                with self.subTest(change=change), self.assertRaises(oracle.ReferenceError):
                    oracle._first_segment_latent_roundtrip_sha256(payload, record | change)

    def test_captured_latent_must_match_authenticated_rounded_prefix(self):
        capture = {"tensors": {"input.latent": {"sha256": "0" * 64}}}
        with self.assertRaisesRegex(oracle.ReferenceError, "latent differs"):
            oracle._validate_capture_aliases(capture, Path("unused"), {
                "first_segment_latent_bf16_roundtrip_sha256": "1" * 64,
            })

    def test_duplicate_keys_rejected_at_any_depth(self):
        for document in (b'{"schema":1,"schema":2}', b'{"tensors":{"x":1,"x":2}}'):
            with self.subTest(document=document), self.assertRaisesRegex(
                oracle.ReferenceError, "duplicate JSON key"
            ):
                oracle._json_no_duplicates(document, "capture")

    def test_non_object_or_invalid_json_rejected(self):
        for document in (b"[]", b"null", b"{", b"\xff"):
            with self.subTest(document=document), self.assertRaises(oracle.ReferenceError):
                oracle._json_no_duplicates(document, "capture")

    def test_range_hashing_handles_unaligned_chunk_boundaries(self):
        data = b"abcdefghijklmnopqrs"
        ranges = [("a", 0, 3, sha(data[:3])), ("b", 3, 14, sha(data[3:14])),
                  ("c", 14, 19, sha(data[14:]))]
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "tensors.bin"
            payload.write_bytes(data)
            with mock.patch.object(oracle, "HASH_CHUNK", 7):
                oracle._authenticate_ranges(payload, len(data), sha(data), ranges)

    def test_tensor_digest_rejected_even_when_whole_payload_digest_matches(self):
        data = b"abcdefgh"
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "tensors.bin"
            payload.write_bytes(data)
            with self.assertRaisesRegex(oracle.ReferenceError, "tensor digest mismatch: b"):
                oracle._authenticate_ranges(payload, 8, sha(data), [
                    ("a", 0, 4, sha(data[:4])), ("b", 4, 8, "0" * 64),
                ])

    def test_payload_mutation_and_trailing_bytes_rejected(self):
        data = b"abcdefgh"
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "tensors.bin"
            for changed in (b"Abcdefgh", data + b"x", data[:-1]):
                with self.subTest(changed=changed), self.assertRaises(oracle.ReferenceError):
                    payload.write_bytes(changed)
                    oracle._authenticate_ranges(payload, 8, sha(data), [
                        ("all", 0, 8, sha(data)),
                    ])

    def _capture(self, directory: str) -> tuple[Path, dict, dict]:
        # Retain the actual checkpoint inventory, but shrink each tensor to one
        # scalar so admission failures need neither torch nor model allocations.
        shapes = {name: (1,) for name in oracle._capture_shapes()}
        records = {}
        data = bytearray()
        for name in shapes:
            records[name] = {"dtype": "F32", "shape": [1], "offset": len(data),
                             "nbytes": 4, "sha256": sha(b"\0" * 4)}
            data.extend(b"\0" * 4)
        path = Path(directory) / "capture.json"
        payload = path.with_name("capture.tensors.bin")
        payload.write_bytes(data)
        document = {key: "unused" for key in oracle.EXPECTED_CAPTURE_KEYS}
        document.update({
            "schema": oracle.CAPTURE_SCHEMA, "geometry": copy.deepcopy(oracle.GEOMETRY),
            "timestep": 1000, "terminal_wait_count": 1,
            "named_allocations_during_encode": 0,
            "activation_readbacks_before_terminal_wait": 0,
            "scope": oracle.EXPECTED_CAPTURE_SCOPE,
            "precision": "BF16 resident weights; F32 activations and MPS attention control",
            "payload": payload.name, "payload_bytes": len(data),
            "payload_sha256": sha(data), "tensors": records,
        })
        return path, document, shapes

    def _validate(self, path: Path, document: dict, shapes: dict):
        raw = json.dumps(document).encode()
        path.write_bytes(raw)
        with mock.patch.object(oracle, "_capture_shapes", return_value=shapes):
            return oracle._validate_flat_capture(document, path, sha(raw))

    def test_exact_inventory_and_layout_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document, shapes = self._capture(directory)
            payload, names = self._validate(path, document, shapes)
            self.assertEqual(payload.name, "capture.tensors.bin")
            self.assertEqual(set(names), set(shapes))
            self.assertEqual(len(names), 99)

    def test_changed_geometry_or_execution_scope_rejected(self):
        mutations = (
            lambda d: d["geometry"].update(valid_text_tokens=556),
            lambda d: d.update(terminal_wait_count=2),
            lambda d: d.update(activation_readbacks_before_terminal_wait=1),
            lambda d: d.update(named_allocations_during_encode=1),
            lambda d: d.update(timestep=999),
        )
        with tempfile.TemporaryDirectory() as directory:
            path, original, shapes = self._capture(directory)
            for change in mutations:
                document = copy.deepcopy(original)
                change(document)
                with self.subTest(document=document["geometry"]), self.assertRaises(
                    oracle.ReferenceError
                ):
                    self._validate(path, document, shapes)

    def test_missing_checkpoint_or_overlapping_extent_rejected(self):
        mutations = (
            lambda d: d["tensors"].pop("block.img.output"),
            lambda d: d["tensors"]["input.txt"].update(offset=0),
            lambda d: d["tensors"]["input.txt"].update(nbytes=8),
            lambda d: d["tensors"]["input.txt"].update(shape=[2]),
            lambda d: d["tensors"]["input.txt"].update(dtype="BF16"),
            lambda d: d.update(payload="../capture.tensors.bin"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path, original, shapes = self._capture(directory)
            for change in mutations:
                document = copy.deepcopy(original)
                change(document)
                with self.subTest(change=change), self.assertRaises(oracle.ReferenceError):
                    self._validate(path, document, shapes)

    def test_manifest_digest_required_even_with_valid_tensor_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document, shapes = self._capture(directory)
            self._validate(path, document, shapes)
            with self.assertRaisesRegex(oracle.ReferenceError, "manifest digest"):
                oracle._validate_flat_capture(document, path, "0" * 64)

    def test_report_never_overwrites_existing_file_or_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            existing.write_text("user data")
            link = Path(directory) / "report.json"
            link.symlink_to(existing)
            for path in (existing, link):
                with self.subTest(path=path), self.assertRaises(oracle.ReferenceError):
                    oracle._write_json_fresh(path, {"status": "completed"})
            self.assertEqual(existing.read_text(), "user data")

    def test_report_publication_does_not_clobber_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            real_fsync = os.fsync

            def competing_writer(fd):
                destination.write_text("concurrent report")
                return real_fsync(fd)

            with mock.patch.object(oracle.os, "fsync", side_effect=competing_writer):
                with self.assertRaises((oracle.ReferenceError, FileExistsError)):
                    oracle._write_json_fresh(destination, {"status": "completed"})
            self.assertEqual(destination.read_text(), "concurrent report")
            self.assertEqual(sorted(path.name for path in destination.parent.iterdir()),
                             ["report.json"])


if __name__ == "__main__":
    unittest.main()
