"""CPU falsifiers for the isolated FFN frequency/timing admission rule."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load("ffn_stage_analysis_test_target", "examples/support/analyze_pixelumm_ffn_stage_timing.py")
P = load("frequency_test_fixture_reused", "spec/pixelumm_frequency_profile_test.py")


def inputs():
    old_report = P.report_document()
    launch = P.launch_document(old_report)
    summary, samples = P.observer_inputs(old_report, launch)
    launch["schema"] = A.LAUNCH_SCHEMA
    report = {"schema": A.REPORT_SCHEMA, "pass": True, "correctness_pass": True,
              "execution_pass": True, "timing_isolation_pass": True,
              "production_promoted": False, "performance_promoted": False,
              "full_generation_executed": False, "schedule": [], "gpu_roots": []}
    report["protocol"] = {"tokens": 5760, "hidden": 4096, "ffn": 12288, "block": 0,
                           "repeat_count_per_sample": 128, "terminal_waits": 1, "window": "w0",
                           "candidate_topology": "stage_outputs", "api_phase_roots_enabled": False,
                           "no_generation": True}
    report["parity"] = {"atol": 2e-5, "rtol": 2e-5}
    metrics = []
    for slot in range(57):
        elements = (A.PRODUCT_ELEMENTS if slot % 2 == 0 else A.DOWN_ELEMENTS) if slot < 20 else (
            0 if slot < 54 else A.PRODUCT_ELEMENTS if slot < 56 else A.DOWN_ELEMENTS)
        metrics.append({"slot": slot, "elements": elements, "different_words": 0,
                        "failures": 1 if slot == 56 else 0, "nonfinite": 1 if slot == 56 else 0})
    report["execution"] = {"encoding_named_allocation_delta": 0, "preterminal_readbacks": 0,
                            "total_readbacks_after_wait": 1, "wait_count": 1,
                            "maximum_named_bytes": 2 * 1024**3, "named_bytes": 2047869840,
                            "named_resources": {"all_fixture_buffers": 2047869840},
                            "baseline_operation_counts": [641] * 6 + [2],
                            "candidate_graph_operation_counts": A.GRAPH_COUNTS,
                            "metrics": metrics, "metric_failures": [], "execution_failures": [],
                            "api_profile_pass": True}
    report["api_profile"] = []
    for phase in range(6):
        operations = {}
        for name in ("graph_encode", "commit_and_continue", "t5_linear_encode"):
            count = (3203 if name == "graph_encode" else 9999 if name == "commit_and_continue" else 0) if phase == 5 else 0
            operations[name] = {"count": count, "wall_total_ns": 10 if count else 0,
                                "nested_count": 2 if count else 0,
                                "nested_wall_total_ns": 3 if count else 0,
                                "exclusive_wall_total_ns": (7 if name == "graph_encode" else 10) if count else 0,
                                "thread_cpu_total_ns": 5 if count else 0,
                                "nested_thread_cpu_total_ns": 2 if count else 0,
                                "exclusive_thread_cpu_total_ns": (3 if name == "graph_encode" else 5) if count else 0}
        report["api_profile"].append({"phase_id": phase, "operations": operations})
    for index, (label, mode, warmup) in enumerate(A.SCHEDULE):
        start = 12.0 + index * 32
        duration = 28.0 if mode == "B" else 30.0
        report["gpu_roots"].append({"sequence": index, "completed": True, "has_error": False,
                                    "gpu_start_seconds": start, "gpu_end_seconds": start + duration})
        report["schedule"].append({"label": label, "mode": mode, "warmup": warmup,
                                   "repeats": 128, "first_root": index, "end_root": index + 1,
                                   "gpu_root_count": 1, "gpu_root_union_seconds": duration,
                                   "first_gpu_start_seconds": start,
                                   "last_gpu_end_seconds": start + duration,
                                   "outside_root_overlap_seconds": 0.0,
                                   "encode_wall_seconds": duration - 1})

    def anchor(seconds):
        monotonic = round(seconds * 1e9)
        return {"implementation": "mach_absolute_time()", "monotonic_before_ns": monotonic - 500,
                "monotonic_after_ns": monotonic + 500,
                "unix_ns": P.ANCHOR_UNIX + monotonic - P.ANCHOR_MONO}

    report["clock_anchors"] = {"before_encode": anchor(11), "after_terminal_wait": anchor(333)}
    return report, launch, summary, samples


class FFNStageTimingAnalysisTests(unittest.TestCase):
    def test_both_orders_qualify_but_do_not_promote_production(self):
        result = A.analyze_documents(*inputs())
        self.assertTrue(result["all_pairs_frequency_qualified"])
        self.assertTrue(result["full_block_trial_admission"]["passed"])
        self.assertEqual([row["order"] for row in result["pairs"]], ["AB", "BA", "BA", "AB"])
        self.assertAlmostEqual(result["qualified_median_gpu_root_union_gain"], 1 - 28 / 30)
        self.assertFalse(result["production_promoted"])
        self.assertFalse(result["full_block_speedup_established"])

    def test_frequency_mismatch_keeps_only_unqualified_contrast(self):
        report, launch, summary, samples = inputs()
        start = P.ANCHOR_UNIX // 1_000_000 + 98_000
        end = start + 28_000
        for row in samples:
            if start <= row["sample_window_start_unix_ms"] < end:
                row["gpu_active_frequency_mhz"] = 1200.0
        P.refresh_sample_integrity(summary, samples)
        result = A.analyze_documents(report, launch, summary, samples)
        self.assertFalse(result["all_pairs_frequency_qualified"])
        self.assertIsNone(result["qualified_median_gpu_root_union_gain"])
        self.assertFalse(result["full_block_trial_admission"]["passed"])
        self.assertGreater(result["median_observed_gpu_root_union_gain"], 0)

    def test_one_regressing_pair_rejects_a_favorable_median(self):
        report, launch, summary, samples = inputs()
        root, row = report["gpu_roots"][9], report["schedule"][9]
        root["gpu_end_seconds"] = root["gpu_start_seconds"] + 31
        row.update(gpu_root_union_seconds=31, last_gpu_end_seconds=root["gpu_end_seconds"])
        result = A.analyze_documents(report, launch, summary, samples)
        self.assertTrue(result["all_pairs_frequency_qualified"])
        self.assertGreater(result["qualified_median_gpu_root_union_gain"], 0.03)
        self.assertFalse(result["full_block_trial_admission"]["passed"])

    def test_rejects_aliasing_ranges_missing_roots_bad_controls_and_forged_union(self):
        mutations = [
            lambda r: r["schedule"][3].update(first_root=2),
            lambda r: r["gpu_roots"][4].update(completed=False),
            lambda r: r["schedule"][5].update(warmup=True),
            lambda r: r["schedule"][7].update(gpu_root_union_seconds=1),
            lambda r: r.update(correctness_pass=False),
            lambda r: r["gpu_roots"][0].update(gpu_start_seconds=1),
            lambda r: r["clock_anchors"]["after_terminal_wait"].update(unix_ns=P.ANCHOR_UNIX),
            lambda r: r["execution"]["metrics"][0].update(failures=1),
            lambda r: r["execution"]["metrics"][56].update(nonfinite=0),
            lambda r: r["execution"].update(wait_count=2),
            lambda r: r["execution"].update(baseline_operation_counts=[1] * 7),
            lambda r: r["parity"].update(atol=3e-5),
            lambda r: r["execution"].update(api_profile_pass=False),
            lambda r: r["api_profile"][5]["operations"]["graph_encode"].update(count=3),
            lambda r: r["api_profile"][0]["operations"]["graph_encode"].update(count=1),
            lambda r: r["api_profile"][5]["operations"]["commit_and_continue"].update(exclusive_wall_total_ns=7),
            lambda r: r["api_profile"][5]["operations"]["commit_and_continue"].update(nested_count=1),
        ]
        for mutate in mutations:
            report, launch, summary, samples = inputs()
            mutate(report)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                A.analyze_documents(report, launch, summary, samples)

    def test_interior_coverage_and_active_residency_are_required(self):
        for field, value in (("gpu_active_residency_pct", 97.0), ("gpu_active_frequency_mhz", 0.0)):
            report, launch, summary, samples = inputs()
            for row in samples:
                row[field] = value
            P.refresh_sample_integrity(summary, samples)
            result = A.analyze_documents(report, launch, summary, samples)
            self.assertFalse(result["all_pairs_frequency_qualified"])
        report, launch, summary, samples = inputs()
        # Narrow one sample without changing its operation count; its frequency
        # window can no longer contain five independent interior observations.
        report["gpu_roots"][3]["gpu_end_seconds"] -= 22
        report["schedule"][3]["gpu_root_union_seconds"] -= 22
        report["schedule"][3]["last_gpu_end_seconds"] -= 22
        result = A.analyze_documents(report, launch, summary, samples)
        self.assertFalse(result["all_pairs_frequency_qualified"])

    def test_unchanged_two_percent_gate_is_not_relaxed(self):
        report, launch, summary, samples = inputs()
        launch["frequency_gate"]["max_mean_relative_difference"] = 0.025
        with self.assertRaises(ValueError):
            A.analyze_documents(report, launch, summary, samples)

    def test_unbound_observer_receipt_cannot_admit_a_favorable_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("summary.json", "samples.jsonl", "powermetrics.txt")]
            for path in paths:
                path.write_text(path.name)
            completion = {"artifact_sha256": {}}
            result = A.analyze_documents(*inputs())
            self.assertTrue(result["full_block_trial_admission"]["passed"])
            A.bind_observer_artifacts(result, completion, *paths)
            self.assertFalse(result["full_block_trial_admission"]["passed"])
            self.assertFalse(result["receipt_bound_frequency_gate_pass"])
            self.assertTrue(result["frequency_labels_are_descriptive_only"])
            for path in paths:
                completion["artifact_sha256"][f"power/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
            result = A.analyze_documents(*inputs())
            A.bind_observer_artifacts(result, completion, *paths)
            self.assertTrue(result["full_block_trial_admission"]["passed"])
            self.assertTrue(result["receipt_bound_frequency_gate_pass"])
            completion["artifact_sha256"].pop("power/samples.jsonl")
            with self.assertRaisesRegex(ValueError, "partial"):
                A.bind_observer_artifacts(result, completion, *paths)
            completion["artifact_sha256"]["power/samples.jsonl"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "observer completion digest differs"):
                A.bind_observer_artifacts(result, completion, *paths)

    def test_artifact_receipt_rejects_changed_report_and_parity_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, launch, _, _ = inputs()
            parity_path, guard_path, binary = (root / name for name in ("parity.json", "guard.jsonl", "binary"))
            binary.write_bytes(b"fixture binary")
            limits = {"event": "limits", "max_footprint_bytes": 8 * 1024**3,
                      "max_swap_growth_bytes": 64 * 1024**2, "runtime_floor_pct": 10,
                      "runtime_floor_pct_exact": "10", "timeout_sec": 600, "memory_accounting": "file-cache"}
            started = {"event": "started", "pid": 1234, "pgid": 1234}
            guard_path.write_text("\n".join(json.dumps(row) for row in
                                           (limits, started, {"event": "finished", "returncode": 0})) + "\n")
            sources = {str(ROOT / f"source-{i}"): f"{i:064x}" for i in range(1, 15)}
            correctness_entry = "examples/pixelumm_ffn_graph_probe.cr"
            sources[str(ROOT / correctness_entry)] = hashlib.sha256((ROOT / correctness_entry).read_bytes()).hexdigest()
            mixed_sources = {str(Path(path).relative_to(ROOT)) if i else path: value
                             for i, (path, value) in enumerate(sources.items())}
            parity = {"schema": "cogni_ml.pixelumm_ffn_graph_probe/v1", "pass": True,
                      "probe_completed": True, "numerical_pass": True, "execution_pass": True,
                      "protocol": {"candidate_topology": "stage_outputs", "stage_outputs_are_mandatory_targets": True,
                                   "output_only_front_stage_outputs": ["graph.gate", "graph.up", "graph.silu", "graph.product"]},
                      "fixture": {"path": "/fixture.json", "metadata_sha256": "a" * 64,
                                  "payload_sha256": "b" * 64, "weight_cache_metadata_sha256": "c" * 64,
                                  "weight_cache_payload_sha256": "d" * 64}, "source_sha256": mixed_sources}
            parity_path.write_text(json.dumps(parity))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            report["parity"].update(path=str(parity_path), sha256=digest(parity_path), schema=parity["schema"],
                                    candidate_topology="stage_outputs", source_sha256=sources,
                                    fixture_path="/fixture.json", fixture_metadata_sha256="a" * 64,
                                    fixture_payload_sha256="b" * 64, weight_cache_metadata_sha256="c" * 64,
                                    weight_cache_payload_sha256="d" * 64)
            report["memory_guard"] = {"log_path": str(guard_path), "limits": limits, "started": started}
            launch.update(native_argv=[str(binary), "--weight-cache", "/cache.json"], binary_sha256=digest(binary),
                          fixture_pins={str(parity_path): digest(parity_path), "/fixture.json": "a" * 64, "/cache.json": "c" * 64},
                          sources_sha256={str(Path(path).relative_to(ROOT)): value for path, value in sources.items()
                                          if path != str(ROOT / correctness_entry)})
            report_path, launch_path, completion_path = (root / name for name in ("report.json", "launch.json", "completion.json"))
            report_path.write_text(json.dumps(report))
            launch_path.write_text(json.dumps(launch))
            completion = {"returncode": 0, "observer_returncode": 0, "failure": None, "sources_unchanged": True,
                          "artifact_sha256": {path.name: digest(path) for path in (report_path, launch_path, guard_path)}}
            completion_path.write_text(json.dumps(completion))
            A.validate_artifacts(report_path, launch_path, completion_path)
            original_parity = parity_path.read_bytes()
            for bad_sources in (dict(mixed_sources, **{"source-1": "1" * 64}),
                                dict(mixed_sources, **{"../outside-checkout": "0" * 64})):
                parity_path.write_text(json.dumps(dict(parity, source_sha256=bad_sources)))
                report["parity"]["sha256"] = digest(parity_path)
                launch["fixture_pins"][str(parity_path)] = digest(parity_path)
                report_path.write_text(json.dumps(report))
                launch_path.write_text(json.dumps(launch))
                completion["artifact_sha256"].update({path.name: digest(path) for path in (report_path, launch_path)})
                completion_path.write_text(json.dumps(completion))
                with self.assertRaises(ValueError):
                    A.validate_artifacts(report_path, launch_path, completion_path)
            parity_path.write_bytes(original_parity)
            report["parity"]["sha256"] = digest(parity_path)
            launch["fixture_pins"][str(parity_path)] = digest(parity_path)
            report_path.write_text(json.dumps(report))
            launch_path.write_text(json.dumps(launch))
            completion["artifact_sha256"].update({path.name: digest(path) for path in (report_path, launch_path)})
            completion_path.write_text(json.dumps(completion))
            original = report_path.read_bytes()
            report_path.write_text(json.dumps(dict(report, harmless_extra=True)))
            with self.assertRaisesRegex(ValueError, "completion digest"):
                A.validate_artifacts(report_path, launch_path, completion_path)
            report_path.write_bytes(original)
            parity_path.write_text(json.dumps(dict(parity, pass_changed=False)))
            with self.assertRaisesRegex(ValueError, "parity report digest"):
                A.validate_artifacts(report_path, launch_path, completion_path)


if __name__ == "__main__":
    unittest.main()
