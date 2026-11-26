"""CPU falsifiers for the bounded upstream diagnostic; no neural execution."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_upstream_activation_recording as recording


class RecordingTests(unittest.TestCase):
    def test_authenticated_model_paths_are_serializable_but_opaque_objects_are_not(self):
        model = dict(upstream_path=Path("/source"), upstream_revision="a" * 40,
                     source_hashes={"block.py": "b" * 64}, cache_index_path=Path("/index.json"),
                     cache_shards=[{"path": Path("/block.safetensors")}],
                     global_shard={"path": Path("/global.safetensors")}, total_weight_bytes=42)
        result = recording.model_bindings(model)
        self.assertEqual(result["cache_shards"][0]["path"], "/block.safetensors")
        json.dumps(result, allow_nan=False)
        model["global_shard"] = object()
        with self.assertRaisesRegex(TypeError, "unsupported model provenance"):
            recording.model_bindings(model)

    def test_plan_does_not_authenticate_or_launch_model(self):
        with patch.object(recording.render, "authenticate_inputs") as inputs, \
                patch.object(recording.render, "_run_source") as launch, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(recording.main(["--plan-only"]), 0)
        inputs.assert_not_called()
        launch.assert_not_called()
        plan = recording.plan()
        self.assertEqual(plan["block_calls"], 432)
        self.assertLess(plan["block_batch_bytes"], plan["max_batch_bytes"])
        self.assertLess(plan["all_block_bytes"], plan["max_total_bytes"])

    def test_manifest_pin_and_payload_corruption_fail_before_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "source.bin"
            payload.write_bytes(b"authenticated")
            manifest = root / "source.json"
            document = {"payload": {"basename": payload.name, "bytes": payload.stat().st_size,
                                    "sha256": recording.sha256(payload)}}
            manifest.write_text(json.dumps(document))
            with patch.object(recording, "BASELINE_SHA256", recording.sha256(manifest)), \
                    patch.object(recording, "BASELINE_PAYLOAD_SHA256", recording.sha256(payload)):
                self.assertEqual(recording.authenticate_baseline(manifest)[1], payload)
                payload.write_bytes(b"unauthorized!")
                with self.assertRaisesRegex(ValueError, "payload missing or changed"):
                    recording.authenticate_baseline(manifest)
            with self.assertRaisesRegex(ValueError, "manifest missing or changed"):
                recording.authenticate_baseline(manifest)

    def test_manifest_payload_escape_rejected_even_in_pinned_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "source.json"
            manifest.write_text(json.dumps({"payload": {"basename": "../outside", "bytes": 1,
                                                        "sha256": "a" * 64}}))
            with patch.object(recording, "BASELINE_SHA256", recording.sha256(manifest)):
                with self.assertRaisesRegex(ValueError, "sibling basename"):
                    recording.authenticate_baseline(manifest)

    def test_shape_or_precision_promotion_rejected_before_snapshot(self):
        writer = SimpleNamespace(record=unittest.mock.Mock())
        observer = recording.Observer(writer, "A1_high")
        observer.torch = SimpleNamespace(float32="F32", bfloat16="BF16")
        observer._context("step1.T0.block00")
        for value in (SimpleNamespace(shape=(1,), dtype="BF16"),
                      SimpleNamespace(shape=(1, 4320, 2048), dtype="F32")):
            with self.assertRaisesRegex(ValueError, "shape mismatch|dtype mismatch"):
                observer.keep("img.input", value)
        writer.record.assert_not_called()

    def test_missing_block_rejected_even_when_call_counter_matches(self):
        observer = recording.Observer(None, "A1_high")
        observer.block_calls = 432
        observer.expected = {f"step1.T0.block00.{name}" for name in recording.block_shapes()}
        with self.assertRaisesRegex(ValueError, "skipped or repeated"):
            observer.expected_scope(pilot=False)

    def test_prelude_module_named_blocks_is_not_a_dit_block_context(self):
        observer = recording.Observer(None, "A1_high")
        observer.block_calls = 2
        observer.expected = {f"step1.{arm}.block00.{name}"
                             for arm in ("T0", "A1_high") for name in recording.block_shapes()}
        observer.expected.add("step1.prelude.txt_in.individual_token_refiner.blocks.0.norm1")
        self.assertEqual(observer.expected_scope(pilot=True), observer.expected)

    def test_signed_zero_difference_is_not_exact_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "latents.f32le"
            np.zeros(4, dtype="<f4").tofile(payload)
            document = {"tensors": {f"A1_high.step{step}.latent": {
                "shape": [1], "offset_bytes": (step - 1) * 4} for step in range(1, 5)}}
            captures = {name: np.zeros(1, dtype="<f4") for name in document["tensors"]}
            self.assertEqual(len(recording.compare_replay(captures, document, payload, "A1_high")), 4)
            captures["A1_high.step3.latent"][0] = -0.0
            with self.assertRaisesRegex(ValueError, "changed the accepted"):
                recording.compare_replay(captures, document, payload, "A1_high")

    def test_single_arm_capture_inventory_keeps_only_shared_prelude(self):
        render = recording.render
        names = {name for name in render.TENSOR_ORDER if name.startswith("A1_high.")
                 or name in ("zero.step1.patch", "zero.step1.text_valid", "zero.step1.vector")}
        captures = {name: np.zeros(render.TENSOR_SHAPES[name], dtype="<f4") for name in names}
        self.assertEqual(set(render.validate_captures(captures, cases=("A1_high",))), names)
        captures["zero.step2.latent"] = np.zeros(render.LATENT_SHAPE, dtype="<f4")
        with self.assertRaisesRegex(render.RenderReferenceError, "inventory mismatch"):
            render.validate_captures(captures, cases=("A1_high",))
        for cases in ((), ("A1_high", "A1_high"), ("unknown",)):
            with self.assertRaisesRegex(render.RenderReferenceError, "case selection mismatch"):
                render.validate_captures({}, cases=cases)


if __name__ == "__main__":
    unittest.main()
