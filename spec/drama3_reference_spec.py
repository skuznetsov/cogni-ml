"""Adversarial contract checks for pinned DRAMA-3 reference fixtures."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spec.support.drama3.reference_compare import (
    TraceFormatError,
    compare_bundles,
)


FIXTURE = Path(__file__).parent / "support/drama3/reference_fixture"


def write_bundle(path: Path, traces: list[dict[str, object]], *, include_format: bool = True) -> None:
    bundle: dict[str, object] = {"traces": traces}
    if include_format:
        bundle["format"] = "drama3-reference-traces-v1"
    path.write_text(json.dumps(bundle, allow_nan=True), encoding="utf-8")


class Drama3ReferenceSpec(unittest.TestCase):
    def read_reference(self) -> dict[str, object]:
        return json.loads((FIXTURE / "reference_traces.json").read_text(encoding="utf-8"))

    def test_fixture_lineage_and_artifact_hashes_are_pinned(self) -> None:
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["not_released_checkpoint_weights"])
        self.assertEqual(
            manifest["upstream"]["mlx-audio"]["revision"],
            "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7",
        )
        self.assertEqual(
            manifest["upstream"]["fish-speech"]["revision"],
            "541566932abbaebcb0a40e562a1f846738383f38",
        )
        self.assertEqual(len(manifest["reference_files"]), 4)
        for name, expected in manifest["artifacts_sha256"].items():
            digest = hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, name)
        traces = self.read_reference()["traces"]
        self.assertEqual(len(traces), manifest["reference_trace_count"])
        self.assertEqual(len({trace["name"] for trace in traces}), len(traces))

    def test_fixture_compares_and_prefix_selects_backbone_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actual = Path(directory) / "actual.json"
            actual.write_bytes((FIXTURE / "reference_traces.json").read_bytes())
            result = compare_bundles(FIXTURE / "reference_traces.json", actual)
            self.assertEqual(result["status"], "match")
            self.assertEqual(result["trace_count"], 255)
            backbone = compare_bundles(
                FIXTURE / "reference_traces.json", actual, prefixes=("backbone",)
            )
            self.assertEqual(backbone["trace_count"], 27)

    def test_seeded_numerical_defect_is_rejected(self) -> None:
        traces = self.read_reference()["traces"]
        traces[0]["values"][0] += 0.25
        with tempfile.TemporaryDirectory() as directory:
            actual = Path(directory) / "defect.json"
            write_bundle(actual, traces)
            with self.assertRaisesRegex(AssertionError, "differs"):
                compare_bundles(FIXTURE / "reference_traces.json", actual)

    def test_float32_shortest_decimal_roundtrips_before_comparison(self) -> None:
        reference = [{"name": "roundtrip", "shape": [1], "dtype": "float32", "values": [-8.824492]}]
        actual = [{
            "name": "roundtrip",
            "shape": [1],
            "dtype": "float32",
            "values": [-8.824492454528809],
        }]
        with tempfile.TemporaryDirectory() as directory:
            expected_path = Path(directory) / "reference.json"
            actual_path = Path(directory) / "actual.json"
            write_bundle(expected_path, reference)
            write_bundle(actual_path, actual)
            result = compare_bundles(expected_path, actual_path)
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_malformed_nonfinite_missing_and_unmatched_traces_are_rejected(self) -> None:
        original = self.read_reference()["traces"]
        cases: list[tuple[str, list[dict[str, object]], bool, type[Exception]]] = []

        bad_shape = json.loads(json.dumps(original))
        bad_shape[0]["shape"] = [len(bad_shape[0]["values"]) + 1]
        cases.append(("shape", bad_shape, True, TraceFormatError))

        nonfinite = json.loads(json.dumps(original))
        nonfinite[0]["values"][0] = float("nan")
        cases.append(("nonfinite", nonfinite, True, TraceFormatError))

        missing = json.loads(json.dumps(original[:-1]))
        cases.append(("missing", missing, True, TraceFormatError))

        unmatched = json.loads(json.dumps(original))
        unmatched[0]["name"] = "unexpected.trace"
        cases.append(("unmatched", unmatched, True, TraceFormatError))

        cases.append(("missing format", original, False, TraceFormatError))

        for label, traces, include_format, error_type in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                actual = Path(directory) / "actual.json"
                write_bundle(actual, traces, include_format=include_format)
                with self.assertRaises(error_type):
                    compare_bundles(FIXTURE / "reference_traces.json", actual)

    def test_affine_quantization_fixtures_cover_four_and_eight_bit(self) -> None:
        values = json.loads((FIXTURE / "quantization_expected.json").read_text())
        self.assertEqual(set(values), {"affine_fixture.q4", "affine_fixture.q8"})
        for item in values.values():
            self.assertEqual(item["group_size"], 64)
            self.assertEqual(len(item["dense"]), 2)
            self.assertTrue(all(len(row) == 64 for row in item["dense"]))
            self.assertEqual(len(item["matvec"]), 2)


if __name__ == "__main__":
    unittest.main()
