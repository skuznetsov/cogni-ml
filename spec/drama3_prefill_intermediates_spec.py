"""CPU-only checks for the upstream full-prefill intermediate exporter."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "prefill_intermediates.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("drama3_prefill_intermediates", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
prefill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prefill)


class NumpyMx:
    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def argsort(values, axis=-1):
        return np.argsort(values, axis=axis)

    @staticmethod
    def take_along_axis(values, indices, axis):
        return np.take_along_axis(values, indices, axis=axis)

    @staticmethod
    def expand_dims(values, axis):
        return np.expand_dims(values, axis=axis)

    @staticmethod
    def broadcast_to(values, shape):
        return np.broadcast_to(values, shape)


class PrefillIntermediatesSpec(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.values = np.arange(12, dtype="<f4").reshape(3, 4)

    def tearDown(self):
        self.temp.cleanup()

    def test_reads_json_embeddings_and_prefix_rows(self):
        path = self.root / "inputs.json"
        raw = json.dumps({"input_embeddings": self.values.tolist()}).encode("utf-8")
        path.write_bytes(raw)

        values, metadata = prefill.read_input_embeddings(
            inputs_path=path,
            input_raw_path=None,
            input_width=None,
            expected_width=4,
            rows=2,
        )

        np.testing.assert_array_equal(values, self.values[:2])
        self.assertEqual(values.dtype, np.dtype("<f4"))
        self.assertEqual(metadata["available_rows"], 3)
        self.assertEqual(metadata["used_rows"], 2)
        self.assertEqual(
            metadata["used_f32_sha256"],
            prefill._sha256_bytes(self.values[:2].tobytes()),
        )

    def test_reads_little_endian_raw_embeddings(self):
        path = self.root / "inputs.f32"
        raw = self.values.tobytes(order="C")
        path.write_bytes(raw)

        values, metadata = prefill.read_input_embeddings(
            inputs_path=None,
            input_raw_path=path,
            input_width=4,
            expected_width=4,
        )

        np.testing.assert_array_equal(values, self.values)
        self.assertEqual(metadata["sha256"], prefill._sha256_bytes(raw))
        self.assertEqual(metadata["format"], "raw.little_endian.float32")

    def test_rejects_invalid_raw_width_and_nonfinite_inputs(self):
        raw_path = self.root / "bad.f32"
        raw_path.write_bytes(b"\0\0\0")
        with self.assertRaises(prefill.CaptureError):
            prefill.read_input_embeddings(
                inputs_path=None,
                input_raw_path=raw_path,
                input_width=4,
                expected_width=4,
            )

        json_path = self.root / "nonfinite.json"
        json_path.write_text(json.dumps({"input_embeddings": [[float("nan")]]}))
        with self.assertRaises(prefill.CaptureError):
            prefill.read_input_embeddings(
                inputs_path=json_path,
                input_raw_path=None,
                input_width=None,
                expected_width=1,
            )

    def test_scalar_and_sequence_rows_keep_route_layouts_verbatim(self):
        rows, hidden, top_k, channels = 3, 4, 2, 5
        scalar_hidden = [
            np.full((1, 1, hidden), row, dtype=np.float32) for row in range(rows)
        ]
        joined_hidden = prefill.PrefillCapture._rows_array(
            NumpyMx, "scalar", scalar_hidden, rows
        )
        self.assertEqual(joined_hidden.shape, (rows, hidden))

        scalar_routes = [
            np.full((1, 1, top_k, 1, channels), row, dtype=np.float32)
            for row in range(rows)
        ]
        joined_routes = prefill.PrefillCapture._rows_array(
            NumpyMx, "scalar", scalar_routes, rows
        )
        self.assertEqual(joined_routes.shape, (rows, top_k, 1, channels))

        sequence_routes = np.zeros((rows * top_k, channels), dtype=np.float32)
        kept_routes = prefill.PrefillCapture._rows_array(
            NumpyMx, "sequence", [sequence_routes], rows
        )
        np.testing.assert_array_equal(kept_routes, sequence_routes)

    def test_native_probe_names_and_raw_tensor_files(self):
        capture = prefill.PrefillCapture.__new__(prefill.PrefillCapture)
        capture.moe_layer = 39
        self.assertEqual(
            capture._moe_tensor_name("routed_sum_official"),
            "backbone.layers.39.mlp.routed_sum",
        )
        self.assertEqual(
            capture._moe_tensor_name("moe_output_official"),
            "backbone.layers.39.mlp.moe_output",
        )

        output = self.root / "capture"
        output.mkdir()
        tensors = {
            "backbone.layers.39.mlp.expert_ids": np.array([[2, 7]], dtype=np.int32),
            "backbone.hidden": np.array([[1.0, -2.0]], dtype=np.float32),
        }
        records = prefill._write_arm(output, "sequence", tensors)
        self.assertEqual([record["name"] for record in records], sorted(tensors))
        for record in records:
            data = (output / record["file"]).read_bytes()
            self.assertEqual(len(data), record["bytes"])
            self.assertEqual(prefill._sha256_bytes(data), record["sha256"])
            expected_dtype = (
                np.dtype("<i4")
                if record["dtype"] == "int32"
                else np.dtype("<f4")
            )
            self.assertEqual(len(data), int(np.prod(record["shape"])) * expected_dtype.itemsize)

    def test_sequence_route_sort_is_reconstructed_and_canonicalized_by_expert_id(self):
        rows, top_k = 2, 2
        expert_ids = np.array([[5, 1], [7, 2]], dtype=np.int32)
        selected_weights = np.array([[0.7, 0.3], [0.8, 0.2]], dtype=np.float32)
        selected_probabilities = selected_weights.copy()
        # Pinned upstream sorts flattened IDs as [1, 2, 5, 7]. Its inverse
        # order maps each original token/rank slot to that sorted projection.
        switch_ids = np.array([1, 2, 5, 7], dtype=np.int32)
        inverse_order = np.array([2, 0, 3, 1], dtype=np.int32)
        sorted_projection = np.array(
            [[[10, 11]], [[20, 21]], [[50, 51]], [[70, 71]]], dtype=np.float32
        )
        sequence = {
            "backbone.layers.39.mlp.expert_ids": expert_ids,
            "backbone.layers.39.mlp.switch_expert_ids": switch_ids,
            "backbone.layers.39.mlp.switch_inverse_order": inverse_order,
            "backbone.layers.39.mlp.selected_weights": selected_weights,
            "backbone.layers.39.mlp.selected_probabilities": selected_probabilities,
            "backbone.layers.39.mlp.expert_gate_outputs": sorted_projection,
            "backbone.layers.39.mlp.expert_up_outputs": sorted_projection + 100,
            "backbone.layers.39.mlp.expert_activation_outputs": sorted_projection + 200,
            "backbone.layers.39.mlp.expert_down_outputs": np.arange(
                rows * top_k * 3, dtype=np.float32
            ).reshape(rows, top_k, 3),
            "backbone.layers.39.mlp.expert_weighted_outputs": np.arange(
                rows * top_k * 3, dtype=np.float32
            ).reshape(rows, top_k, 3)
            + 20,
        }
        canonical_sequence = prefill._canonicalize_moe_routes(
            NumpyMx, sequence, rows=rows, top_k=top_k, layer=39
        )

        np.testing.assert_array_equal(
            canonical_sequence["backbone.layers.39.mlp.canonical_rank_order"],
            np.array([[1, 0], [1, 0]]),
        )
        expected_ids = np.array([[1, 5], [2, 7]], dtype=np.int32)
        np.testing.assert_array_equal(
            canonical_sequence["backbone.layers.39.mlp.canonical_expert_ids"],
            expected_ids,
        )
        np.testing.assert_array_equal(
            canonical_sequence[
                "backbone.layers.39.mlp.canonical_route_ids_from_switch"
            ],
            expected_ids,
        )
        np.testing.assert_array_equal(
            canonical_sequence[
                "backbone.layers.39.mlp.canonical_expert_gate_outputs"
            ],
            np.array([[[10, 11], [50, 51]], [[20, 21], [70, 71]]]),
        )
        np.testing.assert_array_equal(
            canonical_sequence[
                "backbone.layers.39.mlp.canonical_selected_weights"
            ],
            np.array([[0.3, 0.7], [0.2, 0.8]], dtype=np.float32),
        )

        scalar = {
            "backbone.layers.39.mlp.expert_ids": expert_ids,
            "backbone.layers.39.mlp.switch_expert_ids": expert_ids.copy(),
            "backbone.layers.39.mlp.selected_weights": selected_weights,
            "backbone.layers.39.mlp.selected_probabilities": selected_probabilities,
            "backbone.layers.39.mlp.expert_gate_outputs": np.array(
                [[[50, 51], [10, 11]], [[70, 71], [20, 21]]], dtype=np.float32
            )[:, :, None, :],
            "backbone.layers.39.mlp.expert_up_outputs": np.array(
                [[[150, 151], [110, 111]], [[170, 171], [120, 121]]], dtype=np.float32
            )[:, :, None, :],
            "backbone.layers.39.mlp.expert_activation_outputs": np.array(
                [[[250, 251], [210, 211]], [[270, 271], [220, 221]]],
                dtype=np.float32,
            )[:, :, None, :],
            "backbone.layers.39.mlp.expert_down_outputs": sequence[
                "backbone.layers.39.mlp.expert_down_outputs"
            ],
            "backbone.layers.39.mlp.expert_weighted_outputs": sequence[
                "backbone.layers.39.mlp.expert_weighted_outputs"
            ],
        }
        canonical_scalar = prefill._canonicalize_moe_routes(
            NumpyMx, scalar, rows=rows, top_k=top_k, layer=39
        )
        for stage in (
            "canonical_expert_ids",
            "canonical_selected_weights",
            "canonical_expert_gate_outputs",
            "canonical_expert_up_outputs",
            "canonical_expert_activation_outputs",
            "canonical_expert_down_outputs",
            "canonical_expert_weighted_outputs",
        ):
            name = "backbone.layers.39.mlp." + stage
            np.testing.assert_array_equal(canonical_sequence[name], canonical_scalar[name])


if __name__ == "__main__":
    unittest.main()
