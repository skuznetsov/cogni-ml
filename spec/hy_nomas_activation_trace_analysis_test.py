"""CPU falsifiers for selected-record activation trace analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))
import hy_nomas_activation_trace as trace_writer  # noqa: E402
import hy_nomas_activation_trace_analysis as analysis  # noqa: E402


class _Device:
    type = "mps"

    def __eq__(self, other):
        return getattr(other, "type", None) == self.type


class _FakeTensor:
    layout = "torch.strided"

    def __init__(self, words: np.ndarray):
        self.words = words
        self.device = _Device()
        self.dtype = "torch.bfloat16"

    @property
    def shape(self):
        return self.words.shape

    def numel(self):
        return self.words.size

    def is_floating_point(self):
        return True

    def detach(self):
        return self

    def contiguous(self):
        return self

    def clone(self):
        return _FakeTensor(self.words.copy())

    def _activation_trace_raw_bytes(self, dtype: str) -> bytes:
        if dtype != "bfloat16":
            raise AssertionError(dtype)
        return self.words.astype("<u2", copy=False).tobytes(order="C")


def _bf16_words(value: float, shape: tuple[int, ...]) -> np.ndarray:
    f32 = np.full(shape, value, dtype="<f4")
    bits = f32.view("<u4")
    return (bits >> 16).astype("<u2")


def _trace(path: Path, *, omit: set[str] | None = None, finish: bool = True) -> list[str]:
    arm = "A1_high"
    context = "step1.A1_high.block13.img."
    names = [context + suffix for suffix in ("input", "after_attention", "output")]
    latent_name = "step1.A1_high.head.after_euler"
    all_names = names + [latent_name]
    omitted = omit or set()
    writer = trace_writer.ActivationTraceWriter(
        path,
        max_total_bytes=1024 * 1024,
        max_batch_bytes=1024 * 1024,
        min_free_bytes=0,
        metadata={"arm": arm, "diagnostic_only": True},
    )
    image_shape = (1, 30 * 9 * 16, 2)
    image_values = {"input": 2.0, "after_attention": 3.0, "output": 5.0}
    for name in names:
        if name in omitted:
            continue
        value_name = name.rsplit(".", 1)[-1]
        writer.record(name, _FakeTensor(_bf16_words(image_values[value_name], image_shape)))
    if latent_name not in omitted:
        latent = np.empty((1, 32, 30, 9, 16), dtype="<u2")
        for t in range(30):
            latent[:, :, t, :, :] = _bf16_words(float(t), (1, 32, 1, 9, 16))[:, :, 0]
        writer.record(latent_name, _FakeTensor(latent))
    included = [name for name in all_names if name not in omitted]
    if finish:
        writer.finish(included)
    else:
        writer.flush()
    return included


def _default_trace(path: Path) -> list[str]:
    arm = "A1_high"
    writer = trace_writer.ActivationTraceWriter(
        path,
        max_total_bytes=8 * 1024 * 1024,
        max_batch_bytes=8 * 1024 * 1024,
        min_free_bytes=0,
        metadata={"arm": arm, "diagnostic_only": True},
    )
    names = []
    image_shape = (1, 30 * 9 * 16, 2)
    for step in analysis.DEFAULT_STEPS:
        for layer in analysis.DEFAULT_LAYERS:
            prefix = f"step{step}.{arm}.block{layer:02d}.img."
            for boundary, value in (("input", 2.0), ("after_attention", 3.0), ("output", 5.0)):
                name = prefix + boundary
                writer.record(name, _FakeTensor(_bf16_words(value, image_shape)))
                names.append(name)
        latent_name = f"step{step}.{arm}.head.after_euler"
        latent = np.empty((1, 32, 30, 9, 16), dtype="<u2")
        for t in range(30):
            latent[:, :, t, :, :] = _bf16_words(float(t), (1, 32, 1, 9, 16))[:, :, 0]
        writer.record(latent_name, _FakeTensor(latent))
        names.append(latent_name)
    writer.finish(names)
    return names


class ActivationTraceAnalysisSpec(unittest.TestCase):
    def test_bfloat16_reader_expands_bits_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            names = _trace(path)
            reader = analysis.SelectedTraceReader(path)
            record_name = next(name for name in names if name.endswith("img.input"))
            actual = reader.read_f32(record_name, expected_shape=(1, 30 * 9 * 16, 2))
            original_words = _bf16_words(2.0, (1, 30 * 9 * 16, 2))
            expected_bits = original_words.astype("<u4") << 16
            self.assertTrue(np.array_equal(actual.view("<u4"), expected_bits))

    def test_known_residual_math_and_region_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            _trace(path)
            result = analysis.analyze_trace(path, layers=(13,), steps=(1,))
            self.assertEqual(result["validation_scope"], "selected_only")
            self.assertEqual(result["selection"]["arm"], "A1_high")
            self.assertEqual(result["trace_index_sha256"], analysis.sha256_file(path / "index.json"))
            self.assertIn("coarse", result["regions"]["Alex ROI"]["label"])
            self.assertIn("not optical flow", result["definitions"]["temporal_adjacent_latent_rms"])
            row = result["block_metrics"][0]
            self.assertEqual((row["step"], row["layer"], row["arm"]), (1, 13, "A1_high"))
            for region in ("Alex ROI", "Resume ROI", "rest"):
                metric = row["regions"][region]
                self.assertAlmostEqual(metric["input_rms"], 2.0, places=6)
                self.assertAlmostEqual(metric["attention_residual_rms"], 1.0, places=6)
                self.assertAlmostEqual(metric["ffn_residual_rms"], 2.0, places=6)
                self.assertAlmostEqual(metric["relative_update"], 1.5, places=6)
            latent = result["temporal_latent_metrics"][0]
            for region in ("Alex ROI", "Resume ROI", "rest"):
                self.assertAlmostEqual(latent["regions"][region]["rms"], 1.0, places=6)

    def test_default_scope_selects_all_requested_blocks_and_euler_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            names = _default_trace(path)
            result = analysis.analyze_trace(path)
            self.assertEqual(result["selection"]["layers"], [0, 13, 26, 39, 53])
            self.assertEqual(result["selection"]["steps"], [1, 2, 3, 4])
            self.assertEqual(result["selection"]["arm"], "A1_high")
            self.assertEqual(result["selection"]["selected_record_count"], 64)
            self.assertEqual(len(result["block_metrics"]), 20)
            self.assertEqual(len(result["temporal_latent_metrics"]), 4)
            self.assertEqual(len(result["selected_record_hashes"]), len(names))

    def test_missing_selected_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            missing = "step1.A1_high.block13.img.output"
            _trace(path, omit={missing})
            with self.assertRaisesRegex(ValueError, "missing.*record|record.*missing"):
                analysis.analyze_trace(path, layers=(13,), steps=(1,))

    def test_selected_range_hash_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            _trace(path)
            index = json.loads((path / "index.json").read_text())
            name = "step1.A1_high.block13.img.after_attention"
            record = next(item for item in index["records"] if item["name"] == name)
            shard = path / record["shard"]
            with shard.open("r+b") as handle:
                handle.seek(record["offset"])
                first = handle.read(1)
                handle.seek(record["offset"])
                handle.write(bytes([first[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "record hash mismatch"):
                analysis.analyze_trace(path, layers=(13,), steps=(1,))

    def test_selected_shard_truncation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            _trace(path)
            index = json.loads((path / "index.json").read_text())
            shard = path / index["records"][0]["shard"]
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "shard.*size|truncat"):
                analysis.analyze_trace(path, layers=(13,), steps=(1,))

    def test_selected_shape_mismatch_with_same_element_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            _trace(path)
            index_path = path / "index.json"
            index = json.loads(index_path.read_text())
            record = next(item for item in index["records"]
                          if item["name"] == "step1.A1_high.block13.img.input")
            record["shape"] = [1, 2160, 4]
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "expected shape|shape mismatch|wrong shape"):
                analysis.analyze_trace(path, layers=(13,), steps=(1,))

    def test_incomplete_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            _trace(path, finish=False)
            with self.assertRaisesRegex(ValueError, "complete|incomplete"):
                analysis.analyze_trace(path, layers=(13,), steps=(1,))

    def test_cli_output_is_fresh_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_path = root / "run" / "activations"
            trace_path.parent.mkdir()
            _trace(trace_path)
            output = trace_path.parent / "analysis.json"
            self.assertEqual(analysis.main(["--trace", str(trace_path), "--output", str(output),
                                            "--layers", "13", "--steps", "1"]), 0)
            saved = output.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "overwrite|exists"):
                analysis.main(["--trace", str(trace_path), "--output", str(output),
                               "--layers", "13", "--steps", "1"])
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
