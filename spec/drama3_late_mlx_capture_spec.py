"""CPU-only checks for the late-layer upstream diagnostic contract."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "late_mlx_capture.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("drama3_late_mlx_capture", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class LateMlxCaptureSpec(unittest.TestCase):
    def test_default_and_explicit_layer_ranges(self):
        self.assertEqual(capture.DEFAULT_SELECTED_LAYERS, tuple(range(34, 40)))
        self.assertEqual(capture.parse_layers("34-36,39"), (34, 35, 36, 39))

    def test_layer_parser_rejects_duplicates_and_bad_ranges(self):
        for value in ("", "34-", "35-34", "34,34", "-1"):
            with self.subTest(value=value), self.assertRaises(
                capture.argparse.ArgumentTypeError
            ):
                capture.parse_layers(value)

    def test_contract_names_official_shared_and_canonical_stages(self):
        contract = capture.capture_contract((34, 35, 36, 37, 38, 39))
        self.assertEqual(contract["selected_moe_layers"], list(range(34, 40)))
        self.assertEqual(
            contract["decoder_boundaries"], ["attention_residual", "hidden"]
        )
        for stage in (
            "shared_gate_output",
            "shared_up_output",
            "shared_activation_output",
            "shared_expert_output",
            "shared_gate_logits",
            "shared_weighted_output",
            "canonical_expert_down_outputs",
            "canonical_expert_weighted_outputs",
            "canonical_selected_probabilities",
            "canonical_selected_weights",
        ):
            self.assertIn(stage, contract["moe_stages"])
        self.assertNotIn("canonical_expert_down_outputs", capture._REQUIRED_MOE_STAGES)
        self.assertIn("shared_activation_output", capture._REQUIRED_MOE_STAGES)

    def test_used_input_export_is_exact_and_hash_bound(self):
        embeddings = np.arange(24, dtype="<f4").reshape(3, 8)[:2]
        metadata = {"used_f32_sha256": capture._prefill._sha256_bytes(embeddings.tobytes())}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            updated = capture.write_used_input(output, embeddings, metadata)
            relative = updated["used_f32_file"]
            self.assertEqual(relative, "input/used_embeddings.f32")
            payload = (output / relative).read_bytes()
            self.assertEqual(payload, embeddings.tobytes(order="C"))
            self.assertEqual(updated["used_f32_bytes"], len(payload))
            self.assertEqual(
                capture._prefill._sha256_bytes(payload), updated["used_f32_sha256"]
            )
            self.assertNotIn("used_f32_file", metadata)

    def test_arm_tensors_maps_required_official_returns_and_rejects_missing_ones(self):
        capture_instance = object.__new__(capture.LateLayerCapture)
        capture_instance.hidden_size = 1
        capture_instance.selected_layers = (34,)
        capture_instance._layer_calls = {"scalar": {}}
        capture_instance.refs = {"scalar": {}}
        capture_instance.moe_refs = {"scalar": {34: []}}
        capture_instance.top_k = {34: 1}

        for missing_stage in ("routed_sum", "moe_output"):
            with self.subTest(missing_stage=missing_stage):
                record = {"layer": 34}
                for stage in capture._REQUIRED_MOE_STAGES:
                    record[capture._MOE_RECORD_KEYS.get(stage, stage)] = object()
                record.pop(capture._MOE_RECORD_KEYS[missing_stage])
                capture_instance.moe_refs["scalar"][34] = [record]

                with self.assertRaisesRegex(
                    capture._prefill.CaptureError,
                    rf"layer 34 omitted selected MoE stages: \['{missing_stage}'\]",
                ):
                    capture_instance.arm_tensors(
                        np,
                        "scalar",
                        [np.zeros((1, 1, 1), dtype=np.float32)],
                        rows=1,
                        layer_count=0,
                    )


if __name__ == "__main__":
    unittest.main()
