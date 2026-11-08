#!/usr/bin/env python3
"""CPU-only tests for the chunked DCAE-V decoder artifact comparator."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))

import dcae_v_decoder_compare as compare  # noqa: E402


TEST_TMP_ROOT = Path(os.environ.get("DCAE_V_DECODER_COMPARE_TEST_TMPDIR", tempfile.gettempdir()))
LATENT_SHAPE = (1, 32, 20, 1, 1)
OUTPUT_SHAPE = (1, 3, 80, 32, 32)
TRACE_SHAPES = {
    "project_in": (1, 1024, 10, 1, 1),
    "stages.6": (1, 1024, 10, 1, 1),
    "stages.5": (1, 1024, 40, 1, 1),
    "stages.4": (1, 1024, 40, 2, 2),
    "stages.3": (1, 512, 40, 4, 4),
    "stages.2": (1, 512, 40, 8, 8),
    "stages.1": (1, 256, 40, 16, 16),
    "project_out": (1, 3, 40, 32, 32),
}


def _bf16_bytes(shape: tuple[int, ...], word: int) -> bytes:
    return struct.pack("<H", word) * math.prod(shape)


def write_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, ...], bytes]]) -> str:
    header: dict[str, object] = {}
    offset = 0
    payloads = []
    for name, (shape, raw) in tensors.items():
        expected_size = 2 * math.prod(shape)
        if len(raw) != expected_size:
            raise AssertionError(f"invalid test fixture byte count for {name}")
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payloads.append(raw)
    header["__metadata__"] = {"format": "dcae-v-decoder-test"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-8 - len(encoded)) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for raw in payloads:
            stream.write(raw)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_tensors(*, output_word: int = 0x3F80, input_word: int = 0x3F00,
                  nonfinite_output: bool = False) -> dict[str, tuple[tuple[int, ...], bytes]]:
    tensors = {
        "input.latent": (LATENT_SHAPE, _bf16_bytes(LATENT_SHAPE, input_word)),
        "output.pixels": (
            OUTPUT_SHAPE,
            _bf16_bytes(OUTPUT_SHAPE, 0x7FC0 if nonfinite_output else output_word),
        ),
    }
    for boundary, shape in TRACE_SHAPES.items():
        tensors[f"trace.{boundary}.tile0"] = (shape, _bf16_bytes(shape, 0x0000))
    return tensors


def make_pair(directory: Path, *, candidate_output_word: int = 0x3F80,
              reference_output_word: int = 0x3F80,
              candidate_input_word: int = 0x3F00,
              candidate_nonfinite_output: bool = False,
              candidate_tensors: dict[str, tuple[tuple[int, ...], bytes]] | None = None):
    reference_path = directory / "reference.safetensors"
    candidate_path = directory / "candidate.safetensors"
    reference_sha = write_safetensors(
        reference_path, smoke_tensors(output_word=reference_output_word)
    )
    candidate = candidate_tensors if candidate_tensors is not None else smoke_tensors(
        output_word=candidate_output_word,
        input_word=candidate_input_word,
        nonfinite_output=candidate_nonfinite_output,
    )
    candidate_sha = write_safetensors(candidate_path, candidate)
    return reference_path, reference_sha, candidate_path, candidate_sha


class DCAEVDecoderCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TEST_TMP_ROOT.is_dir():
            raise unittest.SkipTest(f"external test root is unavailable: {TEST_TMP_ROOT}")

    def _temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT)

    def _compare(self, pair, *, chunk_elements=16384):
        reference_path, reference_sha, candidate_path, candidate_sha = pair
        return compare.compare_artifacts(
            reference_path,
            reference_sha,
            candidate_path,
            candidate_sha,
            chunk_elements=chunk_elements,
        )

    def test_real_spatial_causal_prefix_schema_is_output_only(self):
        for frames in (10, 20):
            tensors = {
                "input.latent": {"shape": (1, 32, frames, 9, 16)},
                "output.pixels": {"shape": (1, 3, frames * 4, 288, 512)},
            }
            compare._validate_decoder_schema(SimpleNamespace(tensors=tensors))
            tensors["trace.project_in.tile0"] = {"shape": (1, 1024, 10, 9, 16)}
            with self.assertRaises(compare.CompareError):
                compare._validate_decoder_schema(SimpleNamespace(tensors=tensors))

    def test_real_spatial_prefix_rejects_wrong_extent_and_partial_tile(self):
        for latent, output in (
            ((1, 32, 10, 9, 16), (1, 3, 80, 288, 512)),
            ((1, 32, 15, 9, 16), (1, 3, 60, 288, 512)),
            ((1, 32, 10, 16, 9), (1, 3, 40, 512, 288)),
        ):
            with self.assertRaises(compare.CompareError):
                compare._validate_decoder_schema(SimpleNamespace(tensors={
                    "input.latent": {"shape": latent},
                    "output.pixels": {"shape": output},
                }))

    def test_exact_bf16_smoke_pair_passes_with_small_chunks(self):
        with self._temporary_directory() as temporary:
            result = self._compare(make_pair(Path(temporary)), chunk_elements=257)
            self.assertTrue(result["passed"])
            self.assertEqual(result["total_failed_count"], 0)
            self.assertEqual(result["tensors"]["output.pixels"]["bitwise_mismatch_count"], 0)
            self.assertEqual(result["tensors"]["output.pixels"]["max_abs_error"], 0.0)

    def test_numeric_output_difference_exceeding_tolerance_fails(self):
        with self._temporary_directory() as temporary:
            result = self._compare(make_pair(Path(temporary), candidate_output_word=0x3F81))
            self.assertFalse(result["passed"])
            self.assertGreater(result["total_failed_count"], 0)
            self.assertEqual(result["tensors"]["output.pixels"]["bitwise_mismatch_count"],
                             OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]
                             * OUTPUT_SHAPE[3] * OUTPUT_SHAPE[4])
            self.assertGreater(result["tensors"]["output.pixels"]["max_abs_error"], 2e-5)

    def test_numeric_difference_within_absolute_tolerance_passes_despite_bf16_bit_difference(self):
        with self._temporary_directory() as temporary:
            result = self._compare(make_pair(
                Path(temporary),
                candidate_output_word=0x0001,
                reference_output_word=0x0000,
            ))
            output = result["tensors"]["output.pixels"]
            self.assertTrue(result["passed"])
            self.assertEqual(output["bitwise_mismatch_count"], output["elements"])
            self.assertEqual(output["failed_count"], 0)
            self.assertLessEqual(output["max_abs_error"], 2e-5)

    def test_input_latent_requires_bitwise_identity(self):
        with self._temporary_directory() as temporary:
            result = self._compare(make_pair(Path(temporary), candidate_input_word=0x3F01))
            self.assertFalse(result["passed"])
            self.assertGreater(result["tensors"]["input.latent"]["bitwise_mismatch_count"], 0)
            self.assertGreater(result["tensors"]["input.latent"]["failed_count"], 0)

    def test_nonfinite_capture_is_reported_as_a_failure(self):
        with self._temporary_directory() as temporary:
            result = self._compare(make_pair(Path(temporary), candidate_nonfinite_output=True))
            output = result["tensors"]["output.pixels"]
            self.assertFalse(result["passed"])
            self.assertEqual(output["nonfinite_count"], OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
                             * OUTPUT_SHAPE[2] * OUTPUT_SHAPE[3] * OUTPUT_SHAPE[4])
            self.assertEqual(output["failed_count"], output["nonfinite_count"])

    def test_first_failed_trace_uses_tile_and_upstream_execution_order(self):
        with self._temporary_directory() as temporary:
            changed = smoke_tensors()
            for name in ("trace.project_in.tile0", "trace.stages.6.tile0"):
                shape, _raw = changed[name]
                changed[name] = (shape, _bf16_bytes(shape, 0x3F80))
            result = self._compare(make_pair(
                Path(temporary), candidate_tensors=changed
            ))
            self.assertEqual(result["first_failed_trace"], "trace.project_in.tile0")
            self.assertEqual(result["tensor_order"][2:4], [
                "trace.project_in.tile0", "trace.stages.6.tile0"
            ])

    def test_tensor_schema_and_caller_sha_are_authenticated(self):
        with self._temporary_directory() as temporary:
            directory = Path(temporary)
            pair = make_pair(directory)
            with self.assertRaisesRegex(compare.CompareError, "SHA-256 differs"):
                compare.compare_artifacts(pair[0], "0" * 64, pair[2], pair[3])

            changed = smoke_tensors()
            del changed["trace.project_in.tile0"]
            malformed_dir = directory / "wrong-schema"
            malformed_dir.mkdir()
            malformed_pair = make_pair(malformed_dir, candidate_tensors=changed)
            with self.assertRaisesRegex(compare.CompareError, "tensor names differ"):
                self._compare(malformed_pair)

    def test_rejects_corrupt_header_even_when_its_caller_hash_matches(self):
        with self._temporary_directory() as temporary:
            directory = Path(temporary)
            pair = make_pair(directory)
            corrupt_path = directory / "corrupt.safetensors"
            corrupt_path.write_bytes(b"not a safetensors artifact")
            corrupt_sha = hashlib.sha256(corrupt_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(compare.CompareError, "safetensors header extent"):
                compare.compare_artifacts(pair[0], pair[1], corrupt_path, corrupt_sha)

    def test_cli_returns_nonzero_and_writes_json_report_on_mismatch(self):
        with self._temporary_directory() as temporary:
            pair = make_pair(Path(temporary), candidate_output_word=0x3F81)
            report_path = Path(temporary) / "compare.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = compare.main([
                    "--reference", str(pair[0]), "--reference-sha256", pair[1],
                    "--candidate", str(pair[2]), "--candidate-sha256", pair[3],
                    "--report", str(report_path), "--chunk-elements", "8191",
                ])
            self.assertEqual(code, 1)
            summary = json.loads(stdout.getvalue())
            self.assertFalse(summary["passed"])
            self.assertTrue(report_path.is_file())
            self.assertEqual(json.loads(report_path.read_text())["candidate_sha256"], pair[3])


if __name__ == "__main__":
    unittest.main()
