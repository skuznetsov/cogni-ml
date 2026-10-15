"""Focused negative controls for the pinned dense projection comparator."""

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "drama3" / "dense_reference.py"
SPEC = importlib.util.spec_from_file_location("drama3_dense_reference_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dense_reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense_reference)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n",
                    encoding="utf-8")


def fixture(directory: Path) -> tuple[Path, Path, Path, Path]:
    hashes = {name: f"{index:064x}" for index, name in enumerate(
        sorted(dense_reference.REQUIRED_SOURCE_HASHES), 1
    )}
    input_path = directory / "dense-input.json"
    input_bundle = {
        "format": dense_reference.INPUT_FORMAT,
        "projection": dense_reference.PROJECTION,
        "input_dtype": "float32",
        "input_shape": [dense_reference.INPUT_WIDTH],
        "rows": [{"row": 0, "values": [0.0] * dense_reference.INPUT_WIDTH}],
        "source_model": dense_reference.MODEL_REPOSITORY,
        "source_revision": dense_reference.MODEL_REVISION,
        "source_hashes": hashes,
    }
    write_json(input_path, input_bundle)
    input_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()

    projections = {
        name: {"shape": [width], "dtype": "float32", "values": [0.25] * width}
        for name, width in dense_reference.PROJECTION_SIZES.items()
    }
    reference_path = directory / "dense-reference.json"
    reference = {
        "format": dense_reference.REFERENCE_FORMAT,
        "projection": dense_reference.PROJECTION,
        "source_model": dense_reference.MODEL_REPOSITORY,
        "source_revision": dense_reference.MODEL_REVISION,
        "upstream_revision": dense_reference.UPSTREAM_REVISION,
        "source_hashes": hashes,
        "input_fixture": {"path": input_path.name, "sha256": input_digest},
        "rows": [{"row": 0, "projections": projections}],
    }
    write_json(reference_path, reference)

    metadata = {
        name: {"shape": [width, dense_reference.INPUT_WIDTH], "dtype": "BF16"}
        for name, width in dense_reference.PROJECTION_SIZES.items()
    }
    native_base = {
        "format": dense_reference.NATIVE_FORMAT,
        "projection": dense_reference.PROJECTION,
        "model": "/models/drama-3",
        "input_fixture": input_path.name,
        "input_fixture_sha256": input_digest,
        "weight_metadata": metadata,
        "rows": [{"row": 0, "projections": projections}],
    }
    tree_path, simd_path = directory / "tree.json", directory / "simd.json"
    write_json(tree_path, {**native_base, "mode": "tree"})
    write_json(simd_path, {**native_base, "mode": "simd"})
    return reference_path, tree_path, simd_path, input_path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def next_float32(value: float) -> float:
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    return struct.unpack("!f", struct.pack("!I", bits + 1))[0]


class DenseReferenceControls(unittest.TestCase):
    def test_accepts_valid_real_shape_projection_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, tree, simd, _ = fixture(Path(temporary))
            report = dense_reference.compare(reference, tree, simd)

        self.assertEqual(report["status"], "match")
        self.assertTrue(report["tree_simd_bitwise_equal"])
        self.assertEqual(report["rows"], [0])
        self.assertEqual(report["value_counts"], dense_reference.PROJECTION_SIZES)

    def test_rejects_candidate_value_changed_by_one_float32_ulp(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, tree, simd, _ = fixture(Path(temporary))
            candidate = load_json(simd)
            candidate["rows"][0]["projections"]["qkv"]["values"][0] = next_float32(0.25)
            write_json(simd, candidate)

            with self.assertRaisesRegex(AssertionError, "tree/simd mismatch"):
                dense_reference.compare(reference, tree, simd)

    def test_rejects_tree_and_simd_that_agree_but_exceed_upstream_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, tree, simd, _ = fixture(Path(temporary))
            for path in (tree, simd):
                native = load_json(path)
                native["rows"][0]["projections"]["qkv"]["values"][0] = 0.5
                write_json(path, native)

            with self.assertRaisesRegex(AssertionError, "upstream mismatch"):
                dense_reference.compare(reference, tree, simd)

    def test_rejects_widened_tolerances(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, tree, simd, _ = fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "tolerances may only be tightened"):
                dense_reference.compare(
                    reference, tree, simd,
                    atol=dense_reference.DEFAULT_ATOL * 2,
                    rtol=dense_reference.DEFAULT_RTOL,
                )

    def test_rejects_stale_input_fixture_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, tree, simd, input_path = fixture(Path(temporary))
            input_bundle = load_json(input_path)
            input_bundle["rows"][0]["values"][0] = 1.0
            write_json(input_path, input_bundle)

            with self.assertRaisesRegex(ValueError, "fixture bytes do not match reference hash"):
                dense_reference.compare(reference, tree, simd)

    def test_rejects_wrong_checkpoint_and_upstream_revisions(self):
        for field in ("source_revision", "upstream_revision"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                reference, tree, simd, _ = fixture(Path(temporary))
                manifest = load_json(reference)
                manifest[field] = "0" * 40
                write_json(reference, manifest)

                with self.assertRaisesRegex(ValueError, "model or source revision differs"):
                    dense_reference.compare(reference, tree, simd)

    def test_rejects_wrong_model_path_and_projection_weight_metadata(self):
        mutations = (
            ("model path", lambda native: native.__setitem__("model", "/models/other"),
             "different model paths"),
            ("weight dtype", lambda native: native["weight_metadata"]["qkv"].__setitem__("dtype", "F32"),
             "expected pinned dense BF16 projection weight shapes"),
            ("weight shape", lambda native: native["weight_metadata"]["z"].__setitem__("shape", [1, 2048]),
             "expected pinned dense BF16 projection weight shapes"),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                reference, tree, simd, _ = fixture(Path(temporary))
                candidate = load_json(simd)
                mutate(candidate)
                write_json(simd, candidate)

                with self.assertRaisesRegex(ValueError, message):
                    dense_reference.compare(reference, tree, simd)


if __name__ == "__main__":
    unittest.main()
