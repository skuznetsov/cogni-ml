"""CPU-only end-to-end checks of the mapping harness's owned-process guards.

Pressure readings are injected; no model weights or GPU workloads are used.
The real run_safe wrapper and process monitor execute a tiny fixture program.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
MODULE_SPEC = importlib.util.spec_from_file_location(
    "mapping_benchmark", ROOT / "scripts/drama3_mapping_benchmark.py")
benchmark = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(benchmark)


@unittest.skipUnless(sys.platform == "darwin", "requires Darwin mmap/mincore/msync")
class MappingBenchmarkGuardTest(unittest.TestCase):
    def exercise(self, mode, cache_state="nonresident"):
        with tempfile.TemporaryDirectory(prefix="drama3-harness-contract-") as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            for index in range(6):
                (model / f"{index}.safetensors").write_bytes(bytes(range(256)) * 64)
            paths = benchmark.model_files(model)
            marker = root / "child.pid"
            fixture = root / "fixture.py"
            fixture.write_text(
                f"#!{sys.executable}\n"
                "import json, os, pathlib, sys, time\n"
                f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                + ("time.sleep(60)\n" if mode == "during" else "time.sleep(0.1)\n")
                + "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "out.write_bytes(b'fixture WAV bytes')\n"
                "report = {'text':'Hello.', 'seed':0, 'prompt_ids':[1,2], 'frames':[[3]],\n"
                " 'stopped_on_eos':True, 'sample_rate':44100, 'samples':4,\n"
                " 'backend':'metal_resident', 'resident_readbacks':{'control':1,'terminal':4},\n"
                " 'resident_graph':{'depth':2,'submitted':3,'completed':3,\n"
                " 'completion_waits':3,'max_pending':2,'pending':0,'retained_bytes':0}}\n"
                "pathlib.Path(str(out) + '.json').write_text(json.dumps(report))\n")
            fixture.chmod(0o700)
            baseline = {"text": "Hello.", "seed": 0, "prompt_ids": [1, 2],
                        "frames": [[3]], "stopped_on_eos": True, "sample_rate": 44100,
                        "samples": 4, "backend": "metal_resident",
                        "resident_readbacks": {"control": 1, "terminal": 4}}
            identity = benchmark.binary_identity(fixture)
            doc = {"binary": {"path": str(fixture), **identity},
                   "source_sha256_before": benchmark.sha256_paths(benchmark.source_files()),
                   "rows": [], "current_arm": None}
            args = SimpleNamespace(model=model, binary=fixture)

            def pressure():
                row = doc.get("current_arm") or {}
                if mode == "final" and "outer_monitor_wall_seconds" in row:
                    return benchmark.RESERVE_STOP
                if mode == "during" and marker.exists():
                    return benchmark.RESERVE_STOP
                return 75

            real_query = benchmark.query_cache

            def cache_query(vm, model_paths):
                snapshot = real_query(vm, model_paths)
                if mode == "prelaunch" and "preparation_seconds" in (doc.get("current_arm") or {}):
                    snapshot.update(all_resident=False, resident_pages=5)
                return snapshot

            with patch.object(benchmark, "free_percent", pressure), \
                    patch.object(benchmark, "query_cache", side_effect=cache_query), \
                    patch.object(benchmark, "gpu_utilization_sample", return_value=None), \
                    patch.object(benchmark, "host_cpu_aggregate", return_value=None):
                call = lambda: benchmark.run_arm(
                    args, benchmark.DarwinVM(), paths, benchmark.snapshot_model(paths),
                    baseline, b"fixture WAV bytes", root, cache_state, 2, 0,
                    "private", doc, root / "progress.json")
                if mode == "normal":
                    row = call()
                    self.assertTrue(row["passed"])
                    self.assertGreater(row["process_seconds"], 0)
                elif mode == "prelaunch":
                    with self.assertRaises(benchmark.CacheStateError):
                        call()
                    row = doc["rows"][0]
                    self.assertFalse(row["passed"])
                    self.assertEqual(row["cache_immediately_before_launch"]["resident_pages"], 5)
                else:
                    with self.assertRaises(benchmark.HeadroomBlocked):
                        call()
                    row = doc["rows"][0]
                    self.assertFalse(row["passed"])
                    self.assertTrue(row["reserve_violation"])
                    self.assertEqual(row["status"], "aborted_memory_reserve")
                    self.assertFalse(benchmark.evaluate_gate(
                        doc["rows"], ["nonresident", "resident"], [0, 2], 4)["promoted"])
                    if mode == "final":
                        self.assertEqual(row["post_process_free_percent"], benchmark.RESERVE_STOP)
                self.assertIsNone(doc["current_arm"])
                saved = json.loads((root / "progress.json").read_text())
                self.assertEqual(saved["rows"][0]["passed"], mode == "normal")
            if mode == "prelaunch":
                self.assertFalse(marker.exists(), "rejected cache state launched a child")
                return
            self.assertTrue(marker.exists(), "fixture child never started")
            pid = int(marker.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_normal_completion(self):
        self.exercise("normal")

    def test_resident_completion_after_helper_cleanup(self):
        self.exercise("normal", cache_state="resident")

    def test_reserve_abort_stops_owned_child(self):
        self.exercise("during")

    def test_final_sample_invalidates_completed_child(self):
        self.exercise("final")

    def test_partial_cache_immediately_before_launch_starts_no_child(self):
        self.exercise("prelaunch", cache_state="resident")


class MappingBenchmarkWarmPreparationTest(unittest.TestCase):
    def repair_exercise(self, converges):
        paths = [Path(f"{index}.safetensors") for index in range(6)]
        live = set()
        repaired = []
        self_assert = self

        class FakeVM:
            def map_file(self, path):
                live.add(path)
                return path, 16

            def unmap(self, address, length):
                live.remove(address)

            def touch_all(self, address, length, progress):
                self_assert.assertEqual(live, set(paths))
                progress()

            def missing_ranges(self, address, length):
                return [(0, 8)] if address == paths[0] else []

            def touch_ranges(self, address, ranges, progress):
                self_assert.assertEqual(live, set(paths))
                repaired.append((address, ranges))
                progress()
                return 8

        zero = {"zero_resident": True, "all_resident": False,
                "resident_pages": 0, "total_pages": 6}
        warm = {"zero_resident": False, "all_resident": True,
                "resident_pages": 6, "total_pages": 6}
        partial = {"zero_resident": False, "all_resident": False,
                   "resident_pages": 5, "total_pages": 6}
        snapshots = [zero, partial, warm, warm] if converges else [zero] + [partial] * 4
        record = {}
        with patch.object(benchmark, "query_cache", side_effect=snapshots), \
                patch.object(benchmark, "require_no_open_model_files"), \
                patch.object(benchmark, "free_percent", return_value=75):
            call = lambda: benchmark.prepare_cache(FakeVM(), paths, "resident", [], record)
            if converges:
                call()
            else:
                with self.assertRaises(benchmark.CacheStateError):
                    call()
        self.assertFalse(live)
        count = 1 if converges else 2
        self.assertEqual(repaired, [(paths[0], [(0, 8)])] * count)
        self.assertEqual(len(record["warm_repair_passes"]), count)
        self.assertTrue(all(p["read_bytes"] == 8 for p in record["warm_repair_passes"]))

    def test_missing_page_repair_converges_then_releases_helpers(self):
        self.repair_exercise(True)

    def test_stagnating_repair_stops_after_two_passes(self):
        self.repair_exercise(False)

    def test_missing_ranges_clip_the_last_partial_page(self):
        vm = object.__new__(benchmark.DarwinVM)
        vm.page_size = 4
        with patch.object(vm, "residency_vector", return_value=[0, 1, 0, 0, 1, 0]):
            self.assertEqual(vm.missing_ranges(100, 22), [(0, 4), (8, 8), (20, 2)])

    def exercise(self, fail_during_touch=False):
        paths = [Path(f"{index}.safetensors") for index in range(6)]
        live = set()
        touches = []

        class FakeVM:
            def map_file(self, path):
                live.add(path)
                return path, 16

            def unmap(self, address, length):
                live.remove(address)

            def touch_all(self, address, length, progress):
                self_assert.assertEqual(live, set(paths))
                touches.append(address)
                if fail_during_touch:
                    raise benchmark.HeadroomBlocked("injected reserve exit")
                progress()

        self_assert = self
        zero = {"zero_resident": True, "all_resident": False,
                "resident_pages": 0, "total_pages": 6}
        warm = {"zero_resident": False, "all_resident": True,
                "resident_pages": 6, "total_pages": 6}
        partial = {"zero_resident": False, "all_resident": False,
                   "resident_pages": 5, "total_pages": 6}
        snapshots = [zero, partial] if fail_during_touch else [zero, warm, partial]
        record = {}
        with patch.object(benchmark, "query_cache", side_effect=snapshots), \
                patch.object(benchmark, "require_no_open_model_files"), \
                patch.object(benchmark, "free_percent", return_value=75):
            with self.assertRaises(benchmark.HeadroomBlocked if fail_during_touch
                                   else benchmark.CacheStateError):
                benchmark.prepare_cache(FakeVM(), paths, "resident", [], record)
        self.assertFalse(live, "helper mappings leaked on a rejected preparation")
        self.assertEqual(len(touches), 1 if fail_during_touch else 6)
        key = "cache_after_prepare_failure" if fail_during_touch else "cache_after_prepare"
        self.assertEqual(record[key], partial)
        self.assertEqual(record["cache_before_prepare"], zero)

    def test_rejects_partial_residency_after_all_helpers_released(self):
        self.exercise()

    def test_reserve_exit_releases_all_helpers_and_records_snapshot(self):
        self.exercise(fail_during_touch=True)


if __name__ == "__main__":
    unittest.main()
