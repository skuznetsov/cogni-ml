"""CPU-only falsifiers for metadata-only MLX memory diagnostics."""

from __future__ import annotations

import gc
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/drama3/mlx_memory.py"
# Fake bound methods expose the same schedule global as pinned fish_omni.py.
PREFILL_CHUNK_SIZE = 1
SPEC = importlib.util.spec_from_file_location("drama3_mlx_memory_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)


class MetadataArray:
    def __init__(self, shape, nbytes):
        self.shape, self.nbytes, self.dtype = tuple(shape), nbytes, "float32"

    def item(self):
        raise AssertionError("tensor readback is forbidden")

    def __array__(self, *args, **kwargs):
        raise AssertionError("tensor readback is forbidden")


def fake_mx():
    calls = []

    def evaluate(*args, **kwargs):
        # Keep only scalars: the diagnostic itself must not retain array handles.
        calls.append((len(args), tuple(kwargs)))
        return "original-eval-return"

    return SimpleNamespace(
        eval=evaluate,
        get_active_memory=lambda: 100,
        get_cache_memory=lambda: 20,
        get_peak_memory=lambda: 150,
    ), calls


class MlxMemorySpec(unittest.TestCase):
    def test_header_inventory_preserves_packed_bytes_and_counts_f32_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = {
                "backbone.a": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
                "tower.b": {"dtype": "F32", "shape": [2], "data_offsets": [12, 20]},
                "backbone.packed": {"dtype": "U32", "shape": [3], "data_offsets": [20, 32]},
                "other.unused": {"dtype": "F16", "shape": [1], "data_offsets": [32, 34]},
            }
            raw = json.dumps(header).encode()
            (root / "s.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(34))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {name: "s.safetensors" for name in header},
            }))
            report = memory.core_weight_inventory(root)
            self.assertEqual(report["tensor_count"], 3)
            self.assertEqual(report["stored_bytes"], 32)
            self.assertEqual(report["f32_core_bytes"], 44)
            self.assertEqual(report["packed_u32_bytes"], 12)
            self.assertIn("not measured residency", report["scope"])

    def test_inventory_rejects_missing_tensor_and_shard_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"backbone.a": "../outside.safetensors"},
            }))
            with self.assertRaisesRegex(ValueError, "shard"):
                memory.core_weight_inventory(root)
            raw = b"{}"
            (root / "s.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"backbone.a": "s.safetensors"},
            }))
            with self.assertRaisesRegex(ValueError, "missing"):
                memory.core_weight_inventory(root)

    def test_cache_metadata_counts_backing_capacity_not_valid_prefix(self):
        # Real KVCache.state slices to offset; its keys/values retain capacity.
        class KV:
            keys = MetadataArray((1, 2, 512, 256), 1048576)
            values = MetadataArray((1, 2, 512, 256), 1048576)
            offset = 269

            @property
            def state(self):
                raise AssertionError("do not create valid-prefix array views")

        linear = SimpleNamespace(cache=[None, MetadataArray((1, 32, 128, 128), 2097152)])
        report = memory.cache_inventory({"backbone": [KV(), linear]})
        self.assertEqual(report["backbone"]["array_bytes"], 4194304)
        self.assertEqual(report["backbone"]["layers"][0]["arrays"][0]["shape"][2], 512)
        self.assertEqual(report["backbone"]["layers"][0]["offset"], 269)

    def test_prefill_delegates_unchanged_and_observes_only_existing_evals(self):
        mx, calls = fake_mx()
        original_eval = mx.eval
        output = io.StringIO()
        trace = memory.MemoryTrace(mx, output, budget_bytes=1000, rss_reader=lambda: 999)
        caches = [SimpleNamespace(keys=None, values=None, offset=0)]
        prompt = SimpleNamespace(shape=(1, 3, 11))
        result = object()

        class Model:
            def prefill(self, actual_prompt, backbone, tower):
                self_args = (actual_prompt is prompt, backbone is caches, tower == [])
                assert all(self_args)
                for row in range(3):
                    caches[0].offset = row + 1
                    assert mx.eval("hidden", {"cache": row}) == "original-eval-return"
                return result

        self.assertIs(trace.prefill(Model(), prompt, caches, [], stage="prefill_A"), result)
        self.assertIs(mx.eval, original_eval)
        self.assertEqual(len(calls), 3)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r["completed_rows"] for r in records], [0, 1, 2, 3])
        self.assertEqual(records[-1]["mlx_active_bytes"], 100)
        self.assertEqual(records[-1]["mlx_cached_bytes"], 20)
        self.assertEqual(records[-1]["mlx_peak_active_bytes"], 150)
        self.assertEqual(records[-1]["process_rss_bytes"], 999)

    def test_budget_excess_stops_before_next_row_and_restores_eval(self):
        mx, calls = fake_mx()
        original_eval = mx.eval
        output = io.StringIO()
        trace = memory.MemoryTrace(mx, output, budget_bytes=200, rss_reader=lambda: 0)

        class Model:
            def prefill(self, *args):
                mx.get_active_memory = lambda: 190
                mx.eval("row1")  # 190 active +20 cached exceeds 200.
                raise AssertionError("the next row must not be encoded")

        with self.assertRaises(memory.MemoryBudgetExceeded):
            trace.prefill(Model(), SimpleNamespace(shape=(1, 3, 11)), [], [], stage="prefill")
        self.assertIs(mx.eval, original_eval)
        self.assertEqual(len(calls), 1)
        self.assertTrue(json.loads(output.getvalue().splitlines()[-1])["budget_exceeded"])

    def test_preexisting_budget_excess_never_calls_model(self):
        mx, calls = fake_mx()
        trace = memory.MemoryTrace(mx, io.StringIO(), budget_bytes=119, rss_reader=lambda: 0)

        class Model:
            def prefill(self, *args):
                raise AssertionError("over-budget prefill must not run")

        with self.assertRaises(memory.MemoryBudgetExceeded):
            trace.prefill(Model(), SimpleNamespace(shape=(1, 1, 11)), [], [], stage="prefill")
        self.assertEqual(calls, [])

    def test_transient_peak_excess_stops_even_after_active_memory_falls(self):
        mx, _ = fake_mx()
        output = io.StringIO()
        mx.get_peak_memory = lambda: 201
        trace = memory.MemoryTrace(mx, output, budget_bytes=200, rss_reader=lambda: 0)
        with self.assertRaises(memory.MemoryBudgetExceeded):
            trace.snapshot("after_eval")
        record = json.loads(output.getvalue())
        self.assertEqual(record["mlx_active_bytes"] + record["mlx_cached_bytes"], 120)
        self.assertEqual(record["mlx_peak_active_bytes"], 201)
        self.assertTrue(record["budget_exceeded"])

    def test_chunk_schedule_is_checked_before_any_model_work(self):
        mx, calls = fake_mx()
        original_eval = mx.eval
        output = io.StringIO()
        trace = memory.MemoryTrace(mx, output, rss_reader=lambda: 0)

        class Model:
            def prefill(self, *args):
                raise AssertionError("chunked prefill must not run")

        global PREFILL_CHUNK_SIZE
        try:
            PREFILL_CHUNK_SIZE = 128
            with self.assertRaisesRegex(ValueError, "PREFILL_CHUNK_SIZE=1"):
                trace.prefill(Model(), SimpleNamespace(shape=(1, 269, 11)), [], [], stage="prefill")
        finally:
            PREFILL_CHUNK_SIZE = 1
        with self.assertRaisesRegex(ValueError, "PREFILL_CHUNK_SIZE=1"):
            trace.prefill(SimpleNamespace(prefill=lambda *args: None),
                          SimpleNamespace(shape=(1, 269, 11)), [], [], stage="prefill")
        self.assertIs(mx.eval, original_eval)
        self.assertEqual(calls, [])
        self.assertEqual(output.getvalue(), "")

    def test_exception_and_unexpected_eval_count_do_not_leave_hook_installed(self):
        mx, _ = fake_mx()
        original_eval = mx.eval
        trace = memory.MemoryTrace(mx, io.StringIO(), rss_reader=lambda: 0)

        class FailingModel:
            def prefill(self, *args):
                mx.eval("row1")
                raise RuntimeError("original failure")

        with self.assertRaisesRegex(RuntimeError, "original failure"):
            trace.prefill(FailingModel(), SimpleNamespace(shape=(1, 3, 11)), [], [], stage="prefill")
        self.assertIs(mx.eval, original_eval)

        class DriftedModel:
            def prefill(self, *args):
                mx.eval("row1")

        with self.assertRaisesRegex(ValueError, "evaluation count"):
            trace.prefill(DriftedModel(), SimpleNamespace(shape=(1, 3, 11)), [], [], stage="prefill")
        self.assertIs(mx.eval, original_eval)

    def test_records_do_not_keep_array_handles_alive(self):
        mx, _ = fake_mx()
        output = io.StringIO()
        trace = memory.MemoryTrace(mx, output, rss_reader=lambda: 0)
        value = MetadataArray((2, 3), 24)
        reference = weakref.ref(value)
        cache = SimpleNamespace(cache=[value])
        trace.snapshot("row", {"backbone": [cache]})
        cache.cache.clear()
        del value
        gc.collect()
        self.assertIsNone(reference())

    def test_diagnostic_module_does_not_import_model_or_gpu_runtime(self):
        self.assertNotIn("mlx.core", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("mlx_audio", sys.modules)


if __name__ == "__main__":
    unittest.main()
