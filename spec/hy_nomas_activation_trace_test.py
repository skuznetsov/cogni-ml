"""CPU-only safety tests for the bounded MPS activation trace writer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "spec" / "support" / "hy_nomas_activation_trace.py"
SPEC = importlib.util.spec_from_file_location("hy_nomas_activation_trace", MODULE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE}")
trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace)


class _Device:
    type = "mps"

    def __init__(self, index: int = 0):
        self.index = index

    def __eq__(self, other):
        return isinstance(other, _Device) and self.index == other.index

    def __str__(self):
        return f"mps:{self.index}"


class _FakeTensor:
    """A narrow tensor double: it reports MPS but stores CPU bytes for tests."""

    layout = "torch.strided"

    def __init__(self, array: np.ndarray, dtype_name: str, *, clone_state=None):
        self.array = array
        self.dtype = f"torch.{dtype_name}"
        self.device = _Device()
        self.clone_state = clone_state or {"clones": 0, "cpu_exports": 0}

    @property
    def shape(self):
        return self.array.shape

    def numel(self):
        return self.array.size

    def is_floating_point(self):
        return True

    def detach(self):
        return self

    def contiguous(self):
        return self

    def clone(self):
        self.clone_state["clones"] += 1
        return _FakeTensor(self.array.copy(), self.dtype.removeprefix("torch."),
                           clone_state=self.clone_state)

    def cpu(self):
        self.clone_state["cpu_exports"] += 1
        return self

    def numpy(self):
        return self.array

    def _activation_trace_raw_bytes(self, dtype: str) -> bytes:
        """Exercise the raw BF16 path without requiring PyTorch in this test."""
        self.cpu()
        if dtype == "bfloat16":
            return self.array.astype("<u2", copy=False).tobytes(order="C")
        return self.array.tobytes(order="C")


def _tensor(values, dtype="float32"):
    if dtype == "float32":
        array = np.asarray(values, dtype="<f4")
    elif dtype == "float16":
        array = np.asarray(values, dtype="<f2")
    elif dtype == "bfloat16":
        array = np.asarray(values, dtype="<u2")
    else:
        raise AssertionError(dtype)
    return _FakeTensor(array, dtype)


def _writer(directory: Path, **kwargs):
    return trace.ActivationTraceWriter(
        directory,
        max_total_bytes=kwargs.pop("max_total_bytes", 1024),
        max_batch_bytes=kwargs.pop("max_batch_bytes", 1024),
        min_free_bytes=kwargs.pop("min_free_bytes", 0),
        metadata=kwargs.pop("metadata", {"purpose": "unit-test"}),
        **kwargs,
    )


class ActivationTraceWriterSpec(unittest.TestCase):
    def test_import_does_not_require_torch(self):
        code = (
            "import builtins, importlib.util, sys\n"
            "class BlockTorch:\n"
            "  def find_spec(self, fullname, path=None, target=None):\n"
            "    if fullname == 'torch' or fullname.startswith('torch.'):\n"
            "      raise AssertionError('torch imported eagerly')\n"
            "sys.meta_path.insert(0, BlockTorch())\n"
            f"spec = importlib.util.spec_from_file_location('trace_no_torch', {str(MODULE)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "assert hasattr(mod, 'ActivationTraceWriter')\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_batch_quota_is_checked_before_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensor = _tensor([1, 2, 3, 4])
            writer = _writer(Path(tmp) / "trace", max_batch_bytes=8)
            with self.assertRaisesRegex(ValueError, "batch.*limit|batch.*quota"):
                writer.record("block/0", tensor)
            self.assertEqual(tensor.clone_state["clones"], 0)
            self.assertEqual(tensor.clone_state["cpu_exports"], 0)
            self.assertEqual(writer.pending_bytes, 0)

    def test_default_device_policy_rejects_cpu_before_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = _writer(Path(tmp) / "trace")
            tensor = _tensor([1.0])
            tensor.device = type("CPUDevice", (), {"type": "cpu"})()
            with self.assertRaisesRegex(ValueError, "MPS"):
                writer.record("block/0/output", tensor)
            self.assertEqual(tensor.clone_state["clones"], 0)

    def test_constructor_refuses_to_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            path.mkdir()
            sentinel = path / "preserve.txt"
            sentinel.write_text("existing evidence")
            with self.assertRaises(FileExistsError):
                _writer(path)
            self.assertEqual(sentinel.read_text(), "existing evidence")

    def test_record_snapshots_before_source_mutation_and_exports_only_on_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensor = _tensor([1.0, -2.5, 3.25])
            source_bytes = tensor.array.tobytes()
            writer = _writer(Path(tmp) / "trace", metadata={"run": "cpu-test"})
            writer.record("block/0/output", tensor)
            tensor.array[:] = 99
            self.assertEqual(tensor.clone_state["clones"], 1)
            self.assertEqual(tensor.clone_state["cpu_exports"], 0)

            writer.finish(["block/0/output"])
            self.assertEqual(tensor.clone_state["cpu_exports"], 1)
            index = trace.validate_trace(Path(tmp) / "trace")
            self.assertTrue(index["complete"])
            self.assertEqual(index["metadata"], {"run": "cpu-test"})
            item = index["records"][0]
            with (Path(tmp) / "trace" / item["shard"]).open("rb") as shard:
                shard.seek(item["offset"])
                self.assertEqual(shard.read(item["nbytes"]), source_bytes)

    def test_bfloat16_is_written_as_original_uint16_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensor = _tensor([0x3F80, 0xC020, 0x0001], "bfloat16")
            expected = tensor.array.astype("<u2", copy=False).tobytes()
            writer = _writer(Path(tmp) / "trace")
            writer.record("block/0/bf16", tensor)
            writer.finish(["block/0/bf16"])
            index = trace.validate_trace(Path(tmp) / "trace")
            item = index["records"][0]
            self.assertEqual(item["dtype"], "bfloat16")
            self.assertEqual(item["nbytes"], len(expected))
            with (Path(tmp) / "trace" / item["shard"]).open("rb") as shard:
                self.assertEqual(shard.read(), expected)

    def test_duplicate_name_is_rejected_even_after_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = _writer(Path(tmp) / "trace")
            first = _tensor([1.0])
            writer.record("block/0/output", first)
            writer.flush()
            second = _tensor([2.0])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                writer.record("block/0/output", second)
            self.assertEqual(second.clone_state["clones"], 0)

    def test_total_quota_is_checked_before_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = _writer(Path(tmp) / "trace", max_total_bytes=4)
            tensor = _tensor([1.0, 2.0])
            with self.assertRaisesRegex(ValueError, "total.*limit|total.*quota"):
                writer.record("block/0/output", tensor)
            self.assertEqual(tensor.clone_state["clones"], 0)

    def test_disk_reserve_rejects_before_creating_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            writer = _writer(path, min_free_bytes=100)
            writer.record("block/0/output", _tensor([1.0]))
            disk = type("DiskUsage", (), {"total": 1000, "used": 950, "free": 50})()
            with mock.patch.object(trace.shutil, "disk_usage", return_value=disk):
                with self.assertRaisesRegex(OSError, "free space|reserve"):
                    writer.flush()
            self.assertFalse(list(path.glob("batch-*.bin")))
            self.assertTrue((path / ".incomplete").is_file())

    def test_validator_detects_truncation_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            writer = _writer(path)
            writer.record("block/0/output", _tensor([1.0, 2.0]))
            writer.finish(["block/0/output"])
            index = json.loads((path / "index.json").read_text())
            shard = path / index["shards"][0]["name"]
            original = shard.read_bytes()

            shard.write_bytes(original[:-1])
            with self.assertRaisesRegex(ValueError, "size|truncat|hash"):
                trace.validate_trace(path)

            shard.write_bytes(original)
            shard.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaisesRegex(ValueError, "hash"):
                trace.validate_trace(path)

    def test_validator_enforces_declared_batch_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            writer = _writer(path, max_batch_bytes=16)
            writer.record("block/0/output", _tensor([1.0, 2.0]))
            writer.finish(["block/0/output"])
            index_path = path / "index.json"
            index = json.loads(index_path.read_text())
            index["limits"]["max_batch_bytes"] = 4
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "batch byte limit"):
                trace.validate_trace(path)

    def test_incomplete_trace_is_inspectable_but_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            writer = _writer(path)
            writer.record("block/0/output", _tensor([1.0]))
            writer.flush()
            self.assertFalse(trace.validate_trace(path, require_complete=False)["complete"])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                trace.validate_trace(path)
            writer.finish(["block/0/output"])
            self.assertTrue(trace.validate_trace(path)["complete"])

    def test_scope_mismatch_cannot_mark_trace_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            writer = _writer(path)
            writer.record("block/0/output", _tensor([1.0]))
            with self.assertRaisesRegex(ValueError, "scope|expected|missing"):
                writer.finish(["block/0/input"])
            self.assertTrue((path / ".incomplete").exists())
            self.assertFalse(trace.validate_trace(path, require_complete=False)["complete"])

    def test_validator_rejects_path_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            writer = _writer(path)
            writer.record("block/0/output", _tensor([1.0]))
            writer.finish(["block/0/output"])
            index_path = path / "index.json"
            index = json.loads(index_path.read_text())
            index["records"][0]["shard"] = "../outside.bin"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "shard|path|safe"):
                trace.validate_trace(path)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            writer = _writer(path)
            writer.record("block/0/output", _tensor([1.0]))
            writer.finish(["block/0/output"])
            (path / "extra.bin").symlink_to(path / "index.json")
            with self.assertRaisesRegex(ValueError, "symlink|unexpected"):
                trace.validate_trace(path)


if __name__ == "__main__":
    unittest.main()
