"""CPU-only hostile controls for the PixelUMM order/profile analyzer."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "examples/support/analyze_pixelumm_order_profile.py"
SPEC = importlib.util.spec_from_file_location("pixelumm_order_analysis", ANALYZER_PATH)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def capture_order() -> list[str]:
    return [f"block{slot}.attention" if slot < 34 else name
            for slot in range(36)
            for name in ((f"block{slot}.attention",) if slot < 34 else
                         (f"block{slot}.attention", f"block{slot}.output"))] + [
        "head.final_norm_tile_major", "head.restored_thw", "head.head_norm_thw",
        "head.linear_thw", "head.output_thw",
    ]


def root_rows() -> list[dict[str, object]]:
    return [
        {"sequence": 0, "completed": True, "has_error": False,
         "gpu_start_seconds": 1.000000003, "gpu_end_seconds": 1.050000003},
        {"sequence": 1, "completed": True, "has_error": False,
         "gpu_start_seconds": 1.010000003, "gpu_end_seconds": 1.060000003},
    ]


def trace_document() -> dict[str, object]:
    rows = root_rows()
    roots = []
    for index, row in enumerate(rows):
        tracked = 1_000_000_000 if index == 0 else 1_010_000_003
        completion = 1_070_000_000 if index == 0 else 1_080_000_000
        start = float(row["gpu_start_seconds"])
        end = float(row["gpu_end_seconds"])
        roots.append({
            "sequence": index,
            "tracked_or_observed_mach_ticks": tracked,
            "completion_observed_mach_ticks": completion,
            "gpu_start_time_bits": struct.unpack("=Q", struct.pack("=d", start))[0],
            "gpu_end_time_bits": struct.unpack("=Q", struct.pack("=d", end))[0],
            "status_at_track_or_enable": 0,
            "final_status": 4,
            "has_error": 0,
            "completion_source": 1,
            "gpu_start_seconds": start,
            "gpu_end_seconds": end,
        })
    return {
        "info": {"owner_thread_id": 73, "mach_timebase_numer": 1,
                 "mach_timebase_denom": 1, "trace_start_mach_ticks": 990_000_000,
                 "event_capacity": 8192, "event_count": 1, "root_count": 2, "failed": 0},
        "events": [{
            "serial": 1, "begin_mach_ticks": 1_000_000_004, "end_mach_ticks": 1_005_000_000,
            "begin_thread_cpu_ns": 10_000_000, "end_thread_cpu_ns": 15_000_000,
            "api_scope_serial": 3, "parent_graph_serial": 2,
            "before_root_sequence": 0, "after_root_sequence": 1, "phase": 2,
            "graph_depth": 1, "before_root_status": 0,
            "before_root_status_after_call": 4, "after_root_status": 0, "flags": 15,
        }],
        "roots": roots,
    }


def op_stats(count: int, *, nested_count: int = 0, nested_wall: int = 0,
             nested_cpu: int = 0, wall_each: int = 100_000,
             cpu_each: int = 60_000) -> dict[str, int]:
    if count == 0:
        return dict.fromkeys(ANALYZER.API_FIELDS, 0)
    wall = count * wall_each
    cpu = count * cpu_each
    return {
        "count": count, "wall_total_ns": wall, "wall_max_ns": wall_each,
        "thread_cpu_total_ns": cpu, "thread_cpu_max_ns": cpu_each,
        "nested_count": nested_count, "nested_wall_total_ns": nested_wall,
        "nested_thread_cpu_total_ns": nested_cpu,
        "exclusive_wall_total_ns": wall - nested_wall,
        "exclusive_thread_cpu_total_ns": cpu - nested_cpu,
    }


def api_document() -> dict[str, object]:
    graph_counts = {0: 5, 1: 324, 2: 6480, 3: 108, 4: 0, 5: 108}
    pixel_names = {0: "unassigned_head", 1: "self_prepare", 2: "local_attention",
                   3: "projection_residual", 4: "unused_cross_attention", 5: "ffn"}
    commit_count = {2: 1}
    phases = []
    for phase_id, phase_name in ANALYZER.API_PHASES.items():
        nested = phase_id == 2
        graph = op_stats(graph_counts[phase_id], nested_count=int(nested),
                          nested_wall=5_000_000 if nested else 0,
                          nested_cpu=5_000_000 if nested else 0)
        commit = op_stats(commit_count.get(phase_id, 0), nested_count=int(nested),
                           nested_wall=5_000_000 if nested else 0,
                           nested_cpu=5_000_000 if nested else 0,
                           wall_each=6_000_000, cpu_each=6_000_000)
        commit["exclusive_wall_total_ns"] = commit["wall_total_ns"]
        commit["exclusive_thread_cpu_total_ns"] = commit["thread_cpu_total_ns"]
        phases.append({
            "pixelumm_phase": {0: "unassigned_head", 1: "self_prepare", 2: "local_attention",
                               3: "projection_residual", 4: "unused_cross_attention", 5: "ffn"}[phase_id],
            "source_phase_id": phase_id, "source_phase_name": phase_name,
            "operations": {"graph_encode": graph, "commit_and_continue": commit,
                           "t5_linear_encode": op_stats(0)},
        })
    return {"source_schema": "MPSAPIProfile WAN phase IDs", "phases": phases}


def profile_document(enabled: bool) -> dict[str, object]:
    markers = ANALYZER.MARKERS
    cursor = 1
    blocks = []
    all_wall = {phase: 0.0 for phase in ANALYZER.PHASES}
    all_cpu = {phase: 0.0 for phase in ANALYZER.PHASES}
    expected_candidates: list[list[str]] = [[] for _ in range(2)]
    for slot in range(36):
        local_root = cursor
        marker_rows = []
        for marker_index, name in enumerate(markers):
            if slot == 0 and marker_index == len(markers) - 1:
                local_root = 2
            marker_rows.append({"name": name, "elapsed_seconds": marker_index * 0.01,
                                "thread_cpu_seconds": marker_index * 0.005,
                                "root_count": local_root})
        wall = {}
        cpu = {}
        for index, phase in enumerate(ANALYZER.PHASES):
            wall[phase] = 0.01
            cpu[phase] = 0.005
            all_wall[phase] += wall[phase]
            all_cpu[phase] += cpu[phase]
            start_count = marker_rows[index]["root_count"]
            end_count = marker_rows[index + 1]["root_count"]
            for root_index in range(start_count - 1, end_count):
                expected_candidates[root_index].append(f"block_{slot}/{phase}")
        blocks.append({"slot": slot, "attention_tile_count": 60, "markers": marker_rows,
                       "phase_wall_seconds": wall, "phase_thread_cpu_seconds": cpu})
        cursor = marker_rows[-1]["root_count"]

    head_start = cursor
    for root_index in range(head_start - 1, 2):
        expected_candidates[root_index].append("head")
    roots = []
    candidate_sums = {f"block_{slot}/{phase}": 0.0 for slot in range(36) for phase in ANALYZER.PHASES}
    candidate_sums["head"] = 0.0
    definite_sums = dict.fromkeys(candidate_sums, 0.0)
    for index, row in enumerate(root_rows()):
        duration = float(row["gpu_end_seconds"]) - float(row["gpu_start_seconds"])
        candidates = expected_candidates[index]
        roots.append({**row, "duration_seconds": duration, "candidate_phases": candidates})
        for label in candidates:
            candidate_sums[label] += duration
        if len(candidates) == 1:
            definite_sums[candidates[0]] += duration
    metrics = ANALYZER._root_metrics([
        (float(row["gpu_start_seconds"]), float(row["gpu_end_seconds"])) for row in root_rows()
    ])
    phase_root_profile = {"root_count": 2, "root_duration_seconds": metrics["root_duration_seconds"],
                          "envelope_seconds": metrics["envelope_seconds"],
                          "union_seconds": metrics["union_seconds"],
                          "overlap_seconds": metrics["overlap_seconds"],
                          "phase_union_seconds": {"whole_chain": metrics["union_seconds"]}}
    return {
        "schema": ANALYZER.PROFILE, "block_count": 36,
        "ffn_graph_topology": "separate_gate_up", "fused_kv_gather": enabled,
        "scale_softmax": enabled, "blocks": blocks,
        "aggregate_phase_wall_seconds": all_wall,
        "aggregate_phase_thread_cpu_seconds": all_cpu,
        "head_root_range": {"start_root_count": head_start, "end_root_count": 2,
                            "candidate_root_indices_inclusive": [head_start - 1, 1]},
        "root_candidates": expected_candidates,
        "phase_candidate_root_duration_sum_seconds": candidate_sums,
        "phase_definite_root_duration_sum_seconds": definite_sums,
        "phase_root_duration_sum_semantics": "Candidate root-duration sums are non-additive: shared roots may appear in multiple phase candidates and root GPU intervals may overlap; they are not per-phase GPU work.",
        "root_profile": phase_root_profile, "roots": roots,
        "api_profile": api_document(),
    }


def comparison_rows() -> list[dict[str, object]]:
    rows = []
    for name in capture_order():
        elements = ANALYZER.TOKENS * (3072 if name in ("head.linear_thw", "head.output_thw") else 4096)
        rows.append({"stage": name, "elements": elements, "failures": 0, "different_words": 0,
                     "max_abs": 0.0, "finite": True, "actual_sha256": sha(name),
                     "first_failure_index": None, "first_expected": None, "first_actual": None})
    return rows


def memory_events() -> list[dict[str, object]]:
    phases = []
    timestamp = 1_000_000_000
    order = capture_order()
    for request in range(8):
        for name in ("request_before_encode", "request_after_terminal_wait", "request_after_profile_extraction"):
            phases.append((name, request))
        for stage in order:
            head = stage.startswith("head.")
            prefix = "integrated_head_" if head else ""
            phases.extend(((f"before_{prefix}compare_buffers", request),
                           (f"{prefix}compare_buffers_ready", request),
                           (f"{prefix}compare_stage_done", request)))
        phases.extend((("request_after_diagnostic_comparisons", request),
                       ("after_chain_release", request),
                       ("before_post_diagnostic_gc", request),
                       ("after_post_diagnostic_gc", request)))
    events = []
    for sequence, (phase, request) in enumerate(phases, 1):
        timestamp += 1_000_000
        events.append({"schema": "cogni_ml.pixelumm_memory_phase/v1", "sequence": sequence,
                       "timestamp_clock": "CLOCK_MONOTONIC", "monotonic_ns": timestamp,
                       "guard_timestamp_clock": "mach_absolute_time", "guard_monotonic_ns": timestamp,
                       "phase": phase, "request": request})
    return events


def request_document(index: int, enabled: bool, profiled: bool) -> dict[str, object]:
    rows = root_rows()
    metrics = ANALYZER._root_metrics([
        (float(row["gpu_start_seconds"]), float(row["gpu_end_seconds"])) for row in rows
    ])
    return {
        "kind": ["a_warm_off_0", "a_warm_off_1", "a_profile_0", "a_profile_1",
                 "a_profile_2", "a_profile_3", "a_late_off_0", "a_late_off_1"][index],
        "request": index, "passed": True, "input_sha256": sha("same input A"),
        "head_input_is_block35_output": True, "blocks_encode_seconds": 1.0,
        "head_encode_seconds": 0.01, "encode_seconds": 1.01,
        "observed_encode_span_seconds": 1.1, "terminal_wait_seconds": 0.5,
        "total_seconds": 1.6, "gpu_elapsed_seconds": metrics["root_duration_seconds"],
        "command_segments": 2, "terminal_wait_count": 1, "named_allocations_during_encode": 0,
        "activation_readbacks_before_terminal_wait": 0, "diagnostic_readbacks_after_terminal_wait": 43,
        "diagnostic_cpu_scratch_bytes": 123456, "head_before_blocks_rejected": index == 0,
        "early_release_rejected": index == 0, "early_read_rejected": index == 0,
        "early_reuse_rejected": index == 0, "early_poison_rejected": index == 0,
        "shared_scratch_poisoned_before_request": index < 2,
        "comparisons": comparison_rows(), "profiling": profiled,
        "early_profile_rejected": profiled, "shared_gate_up": False,
        "shared_gate_up_block_count": 0, "shared_gate_up_prepared_block_count": 0,
        "combined_attention": enabled, "fused_kv_gather_block_count": 36 if enabled else 0,
        "scale_softmax_block_count": 36 if enabled else 0,
        "scale_softmax_prepared_block_count": 36, "early_attention_change_rejected": index == 0,
        "early_mode_change_rejected": False,
        "unsupported_profile_rejected_without_submission": index == 0,
        "missing_api_profile_rejected_without_submission": index == 0,
        "per_block_encode_host_seconds": [0.001] * 36,
        "root_profile": {"root_count": 2, "root_duration_seconds": metrics["root_duration_seconds"],
                         "envelope_seconds": metrics["envelope_seconds"],
                         "union_seconds": metrics["union_seconds"],
                         "overlap_seconds": metrics["overlap_seconds"], "roots": rows},
        "phase_profile": profile_document(enabled) if profiled else None,
        "continuation_trace": trace_document() if profiled else None,
    }


def report_document(reverse: bool) -> dict[str, object]:
    modes = [True, False, False, True, True, False, False, True]
    if reverse:
        modes = [not mode for mode in modes]
    profiles = [False, False, True, True, True, True, False, False]
    schedule = ["a_warm_off_0", "a_warm_off_1", "a_profile_0", "a_profile_1",
                "a_profile_2", "a_profile_3", "a_late_off_0", "a_late_off_1"]
    return {
        "schema": ANALYZER.REPORT, "generation": False,
        "scope": "native_36_blocks_plus_global_head_in_one_gpu_chain",
        "profile_combined_attention": True, "compare_combined_attention": False,
        "profile_variance": False, "profile_shared_gate_up": False,
        "compare_shared_gate_up": False, "reverse_combined_attention": reverse,
        "correctness_only": False, "atol": 2e-5, "rtol": 2e-5,
        "grid": [6, 44, 80], "block_indices": list(range(36)),
        "output_shape": [1, 3, 24, 704, 1280], "passed": True, "failure": None,
        "cleanup_quarantined": False, "repeats": 1, "expected_request_count": 8,
        "duplicate_head_input_bytes": 0, "shared_block_workspace": True,
        "persistent_block_output_banks": 2, "head_scratch_aliases_block_workspace": False,
        "reference_sha256": sha("r"), "alternate_reference_sha256": sha("ra"),
        "head_reference_sha256": sha("h"), "head_alternate_reference_sha256": sha("ha"),
        "bf16_cache_sha256": sha("cache"), "named_bytes": 600,
        "block_named_bytes": 300, "head_owned_named_bytes": 200,
        "head_weight_named_bytes": 100, "named_budget_bytes": 1024,
        "compared_capture_order": capture_order(),
        "combined_attention_schedule": [
            {"kind": schedule[i], "combined_attention": modes[i], "alternate": False,
             "poisoned": i < 2, "profiling": profiles[i]} for i in range(8)],
        "shared_gate_up_schedule": [
            {"kind": schedule[i], "shared_gate_up": False, "profiling": profiles[i]} for i in range(8)],
        "requests": [request_document(i, modes[i], profiles[i]) for i in range(8)],
        "memory_phases": memory_events(),
    }


class PixelUMMOrderAnalysisTests(unittest.TestCase):
    def test_graph_aggregate_allows_multiple_children_per_call(self) -> None:
        stats = op_stats(1, nested_count=2, nested_wall=2, nested_cpu=2,
                         wall_each=100, cpu_each=100)
        ANALYZER._validate_api_operation_stats(stats, "graph_encode", 0)
        commit_stats = dict(stats)
        commit_stats["exclusive_wall_total_ns"] = commit_stats["wall_total_ns"]
        commit_stats["exclusive_thread_cpu_total_ns"] = commit_stats["thread_cpu_total_ns"]
        with self.assertRaises(ValueError):
            ANALYZER._validate_api_operation_stats(commit_stats, "commit_and_continue", 0)

    def test_normal_and_reverse_reports_recompute(self) -> None:
        normal_raw = report_document(False)
        reverse_raw = report_document(True)
        normal_bytes = json.dumps(normal_raw, sort_keys=True).encode()
        reverse_bytes = json.dumps(reverse_raw, sort_keys=True).encode()
        runs = [
            ANALYZER.analyze_report(normal_raw, hashlib.sha256(normal_bytes).hexdigest(), "normal.json"),
            ANALYZER.analyze_report(reverse_raw, hashlib.sha256(reverse_bytes).hexdigest(), "reverse.json"),
        ]
        summary = ANALYZER.analyze_reports(runs)
        self.assertEqual(summary["schema"], "cogni_ml.pixelumm_order_analysis/v1")
        self.assertEqual(summary["lifecycle_scope"]["terminal_waits_per_request"], 1)
        self.assertEqual(summary["numerical_scope"]["captures_per_request"], 43)
        self.assertEqual(summary["runs"][0]["memory_event_gaps"]["comparison_stage_gap_counts"],
                         {"readback_setup": 43, "compare_and_digest": 43})
        profiled = summary["runs"][0]["phase_localization"]["profiled_requests"]
        self.assertEqual(len(profiled), 4)
        self.assertEqual(profiled[0]["per_block_phase_markers"][0]["slot"], 0)
        self.assertIn("Syntactic parent/depth/flag checks only",
                      summary["runs"][0]["phase_localization"]["continuation_trace"][
                          "parent_graph_validation_scope"])
        api = summary["runs"][0]["phase_localization"]["by_mode"]["baseline"][
            "api_profile_inclusive_and_exclusive_ns"]["self_prepare"]["graph_encode"]
        self.assertEqual(api["wall_max_ns"], 100_000)
        self.assertEqual(api["wall_total_ns"], 2 * 324 * 100_000)

    def assert_rejected(self, mutate) -> None:
        raw = copy.deepcopy(report_document(False))
        mutate(raw)
        with self.assertRaises(ValueError):
            ANALYZER.analyze_report(raw, sha("mutant"), "mutant.json")

    def test_corruption_controls_fail_closed(self) -> None:
        self.assert_rejected(lambda doc: doc.__setitem__("schema", "wrong/v1"))
        self.assert_rejected(lambda doc: doc["requests"][2]["comparisons"][0].__setitem__("finite", False))
        self.assert_rejected(lambda doc: doc["requests"][2].__setitem__("terminal_wait_count", 2))
        self.assert_rejected(lambda doc: doc["requests"][2].__setitem__("named_allocations_during_encode", 1))
        self.assert_rejected(lambda doc: doc["requests"][2]["phase_profile"]["blocks"][0]["markers"][1].__setitem__("root_count", 2))
        self.assert_rejected(lambda doc: doc["requests"][2]["phase_profile"]["api_profile"]["phases"][2]["operations"]["graph_encode"].__setitem__("count", 1))
        self.assert_rejected(lambda doc: doc["requests"][2]["continuation_trace"]["roots"][0].__setitem__("gpu_start_seconds", 1.2))
        self.assert_rejected(lambda doc: doc["memory_phases"][2].__setitem__("guard_monotonic_ns", 1))
        self.assert_rejected(lambda doc: doc["requests"][2]["phase_profile"]["api_profile"]["phases"][1]["operations"]["graph_encode"].__setitem__("wall_max_ns", 1))
        self.assert_rejected(lambda doc: doc["requests"][2]["phase_profile"]["api_profile"]["phases"][1]["operations"]["graph_encode"].__setitem__("thread_cpu_max_ns", 1))
        self.assert_rejected(lambda doc: doc["requests"][2].__setitem__("fused_kv_gather_block_count", 36))
        self.assert_rejected(lambda doc: doc["requests"][2]["phase_profile"].__setitem__("scale_softmax", True))
        self.assert_rejected(lambda doc: doc["requests"][2]["continuation_trace"]["events"][0].__setitem__("graph_depth", 0))
        self.assert_rejected(lambda doc: doc["requests"][2]["phase_profile"]["api_profile"]["phases"][2]["operations"]["commit_and_continue"].__setitem__("nested_count", 0))
        self.assert_rejected(lambda doc: doc["requests"][2]["continuation_trace"]["roots"][0].__setitem__("gpu_start_time_bits", struct.unpack("=Q", struct.pack("=d", 1.2))[0]))

    def test_cli_output_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            normal = Path(directory) / "normal.json"
            reverse = Path(directory) / "reverse.json"
            output = Path(directory) / "analysis.json"
            normal.write_text(json.dumps(report_document(False)), encoding="utf-8")
            reverse.write_text(json.dumps(report_document(True)), encoding="utf-8")
            args = ["--report", str(normal), "--report", str(reverse), "--output", str(output)]
            self.assertEqual(ANALYZER.main(args), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema"],
                             "cogni_ml.pixelumm_order_analysis/v1")
            self.assertEqual(ANALYZER.main(args), 2)


if __name__ == "__main__":
    unittest.main()
