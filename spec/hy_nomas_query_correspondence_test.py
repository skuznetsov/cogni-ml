"""CPU falsifiers for the GPU query-correspondence screening diagnostic."""
import importlib.util
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parent / "support" / "hy_nomas_query_correspondence_reference.py"
CAPTURE_SOURCE = SOURCE.with_name("hy_nomas_query_capture.py")


class QueryCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("query_correspondence", SOURCE)
        self.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reference)

    def test_synthetic_controls_are_distinct_and_lossless_bf16(self):
        positive, shuffled, ties = self.reference.control_arrays()
        for query, key in (positive, shuffled, ties):
            self.assertEqual(query.shape, (144, 16, 128))
            self.assertTrue(np.all(query.view("<u4") & 0xFFFF == 0))
            self.assertTrue(np.all(key.view("<u4") & 0xFFFF == 0))
        query, key = positive
        scores = np.einsum("phd,qhd->hpq", query, key) / np.sqrt(128)
        self.assertTrue(np.all(scores.argmax(axis=-1) == np.arange(144)))
        query, key = shuffled
        scores = np.einsum("phd,qhd->hpq", query, key) / np.sqrt(128)
        self.assertTrue(np.all(scores.argmax(axis=-1) != np.arange(144)))
        self.assertTrue(np.all(ties[0] == 0))

    def fixture(self, root):
        query, key = self.reference.control_arrays()[0]
        return self.reference.publish_input(root / "input.json", query, key,
                                            {"kind": "synthetic_positive"})

    def test_input_authentication_rejects_changed_payload_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            path, digest = self.fixture(Path(temp))
            admitted = self.reference.load_input(path, digest)
            self.assertEqual(admitted["query"].shape, (144, 16, 128))
            payload = path.with_suffix(".f32le")
            raw = bytearray(payload.read_bytes())
            raw[0] ^= 1
            payload.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "digest"):
                self.reference.load_input(path, digest)

    def test_numerically_equal_invalid_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            query, key = self.reference.control_arrays()[0]
            query[0, 0, 0] = np.nextafter(query[0, 0, 0], np.float32(2))
            with self.assertRaisesRegex(ValueError, "BF16"):
                self.reference.publish_input(Path(temp) / "bad.json", query, key, {})

    def test_shapes_and_nonfinite_values_fail_before_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            query, key = self.reference.control_arrays()[0]
            with self.assertRaisesRegex(ValueError, "shape"):
                self.reference.publish_input(Path(temp) / "bad.json", query[:-1], key, {})
            query[0, 0, 0] = np.inf
            with self.assertRaisesRegex(ValueError, "finite"):
                self.reference.publish_input(Path(temp) / "bad.json", query, key, {})
            self.assertFalse((Path(temp) / "bad.json").exists())

    def test_existing_output_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, digest = self.fixture(Path(temp))
            query, key = self.reference.control_arrays()[0]
            with self.assertRaisesRegex(ValueError, "exist"):
                self.reference.publish_input(path, query, key, {})
            alias = Path(temp) / "alias.json"
            alias.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.reference.load_input(alias, digest)

    def test_ties_do_not_establish_strict_self_correspondence(self):
        scores = np.zeros((16, 144, 144), dtype="<f4")
        metrics = np.zeros((16, 144, 6), dtype="<f4")
        metrics[:, :, 5] = 1 / 144
        summary = self.reference.summarize(scores, metrics)
        self.assertEqual(summary["all"]["strict_self_max_fraction"], 0)
        self.assertAlmostEqual(summary["all"]["diagonal_argmax_fraction"], 1 / 144)
        self.assertEqual(summary["all"]["strictly_greater_count_mean"], 0)

    def test_comparator_detects_errors_and_exact_discrete_mismatch(self):
        values = np.ones((16, 144, 6), dtype="<f4")
        self.assertTrue(self.reference.compare(values, values, discrete_columns=(3, 4))["passed"])
        bad = values.copy()
        bad[0, 0, 0] += 0.001
        self.assertFalse(self.reference.compare(bad, values, discrete_columns=(3, 4))["passed"])
        bad = values.copy()
        bad[0, 0, 3] += 0.000001
        self.assertFalse(self.reference.compare(bad, values, discrete_columns=(3, 4))["passed"])

    def test_shared_wrong_metrics_fail_independent_score_audit(self):
        scores = np.zeros((16, 144, 144), dtype="<f4")
        metrics = np.zeros((16, 144, 6), dtype="<f4")
        metrics[..., 5] = 1 / 144
        self.assertTrue(self.reference.audit_metrics(scores, metrics)["passed"])
        metrics[..., 2] = 0.1
        self.assertFalse(self.reference.audit_metrics(scores, metrics)["passed"])

    def test_consumed_input_identity_is_checked_separately_from_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            path, digest = self.fixture(Path(temp))
            admitted = self.reference.load_input(path, digest)
            document = {"input_payload_sha256": admitted["document"]["payload"]["sha256"],
                        "input_tensors": copy.deepcopy(admitted["document"]["tensors"])}
            self.reference.validate_consumed_input(document, admitted)
            document["input_tensors"]["query"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "tensor identity"):
                self.reference.validate_consumed_input(document, admitted)

    def test_model_capture_claim_requires_source_bytes_beyond_provenance_labels(self):
        spec = importlib.util.spec_from_file_location("query_capture", CAPTURE_SOURCE)
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, digest = self.fixture(root)
            with self.assertRaisesRegex(capture.QueryCaptureError, "source T0 block-0"):
                capture.verify_published_input(path, digest)
            query, key = self.reference.control_arrays()[0]
            path, digest = self.reference.publish_input(root / "relabeled.json", query, key, {
                "kind": capture.CAPTURE_KIND, "scope": capture.CAPTURE_SCOPE,
                "boundary": capture.CAPTURE_BOUNDARY, "noise_scope": capture.NOISE_SCOPE,
            })
            with self.assertRaisesRegex(capture.QueryCaptureError, "preblock lineage"):
                capture.verify_published_input(path, digest)

    def test_wrong_model_pins_fail_before_output_or_model_initialization(self):
        spec = importlib.util.spec_from_file_location("query_capture_pins", CAPTURE_SOURCE)
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(capture.QueryCaptureError, "caller pins"):
                capture.run_capture(
                    input_manifest=root / "absent.json", input_manifest_sha256="0" * 64,
                    input_payload_sha256=capture.DEFAULT_INPUT_PAYLOAD_SHA256,
                    derived_directory=root / "derived", source_tensors=root / "source.safetensors",
                    source_receipt=root / "source.json", report=root / "query.json",
                    weights_path=root / "absent.safetensors", weights_sha256="0" * 64,
                    upstream_root=root / "upstream",
                )
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
