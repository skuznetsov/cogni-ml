#!/usr/bin/env python3
"""Focused CPU-only rejection tests for the Q4_K evidence analyzer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples/support/wan_q4k_range_results.py"
SPEC = importlib.util.spec_from_file_location("wan_q4k_range_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(results)


def valid_metric(count: int) -> dict:
    return {
        "compared_values": count,
        "bit_differences": 0,
        "tolerance_violations": 0,
        "nonfinite": 0,
        "max_abs": 0.0,
        "max_ratio": 0.0,
        "rmse": 0.0,
        "relative_l2": 0.0,
        "cosine": 1.0,
    }


def timing_records(rounds: int = 12, reverse_first: bool = False,
                   manifest_sha256: str = "a" * 64) -> list[dict]:
    records = []
    for mode in results.MODES:
        for round_id in (-2, -1, *range(rounds)):
            order = list(results.ARMS[max(round_id, 0) % len(results.ARMS):])
            order += list(results.ARMS[:max(round_id, 0) % len(results.ARMS)])
            reverse = (round_id == -1 if round_id < 0 else round_id >= rounds // 2)
            reverse = reverse != reverse_first
            if reverse:
                order.reverse()
            for position, arm in enumerate(order):
                records.append({
                    "schema": "wan_q4k_range_timing_v1",
                    "mode": mode,
                    "manifest_sha256": manifest_sha256,
                    "rows": 4,
                    "terminal_waits": 1,
                    "inter_operator_host_waits": 0,
                    "activation_readbacks": 0,
                    "encode_metal_buffer_allocations": 0,
                    "round": round_id,
                    "position": position,
                    "warmup": round_id < 0,
                    "variant": arm,
                    "gpu_seconds": 0.001,
                    "wall_seconds": 0.002,
                    "encode_seconds": 0.0001,
                })
    return records


def valid_guard_events() -> list[dict]:
    physical = 64 * 1024**3
    sample = {
        "event": "sample",
        "pressure_free_pct": 80,
        "physical_bytes": physical,
        "headroom_bytes": 20 * 1024**3,
        "group_footprint_bytes": 1024**3,
        "swap_used_bytes": 1_000_000_000,
    }
    limits = {
        "event": "limits",
        "runtime_floor_pct": 15,
        "max_footprint_bytes": 4096 * 1024**2,
        "max_swap_growth_bytes": 64 * 1024**2,
        "timeout_sec": 300,
        "memory_accounting": "file-cache",
        "headroom_formula": "printed free + file-backed (includes speculative)",
        "startup_rule": "memory_pressure -Q >= 15% and selected headroom >= 15% physical + footprint budget",
    }
    return [limits, dict(sample), {"event": "started"}, dict(sample),
            {"event": "finished", "returncode": 0}]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def complete_synthetic_probe(base: Path) -> str:
    """Create a full but tiny receipt tree to exercise read_probe acceptance."""
    rows, din, dout = 4, 5120, 13824
    fixture = base / "fixture-r4"
    fixture.mkdir(parents=True)
    binary = base / "probe-v1"
    binary.write_bytes(b"synthetic native executable identity")
    source_fixture = base.parent / "wan-quantized-projection-20261022" / "fixture-4480-v3"
    source_fixture.mkdir(parents=True)
    source_manifest = source_fixture / "manifest.json"
    source_manifest.write_text('{"schema":"synthetic-source-fixture-v1"}\n')

    output_manifest = fixture / "manifest.json"
    output_manifest.write_text(json.dumps({
        "source_fixture": {"manifest_sha256": results.sha(source_manifest)},
    }, sort_keys=True) + "\n")
    manifest_sha = results.sha(output_manifest)

    metric = lambda count, **changes: {
        **valid_metric(count), **changes,
    }
    setup = {
        "schema": "wan_q4k_range_setup_v1",
        "manifest_sha256": manifest_sha,
        "rows": rows,
        "din": din,
        "dout": dout,
        "unique_source_rows": 4,
        "real_video_derived": False,
        "variants": list(results.ARMS),
        "tolerance": "finite && abs(actual-ref)<=2e-5+2e-5*abs(ref)",
        "payload_bytes_per_arm": 39_813_120,
        "common_descriptor_bytes_per_arm": 16,
        "package_bytes_per_arm": 39_813_136,
        "f32_stage_bytes": din * dout * 4,
        "packing_workspace_bytes": 0,
        "explicit_buffer_bytes_during_pack": 1024**3,
    }
    records = [setup]
    for arm in results.ARMS[1:]:
        records.append({
            "schema": "wan_q4k_range_packing_v1",
            "variant": arm,
            "gpu_seconds": 0.01,
            "wall_seconds": 0.02,
            "encode_seconds": 0.001,
            "payload_word_differences": 0,
            "encode_metal_buffer_allocations": 0,
            "activation_readbacks": 0,
            "terminal_waits": 1,
            "byte_validation_outside_packing_interval": True,
        })
    records.append({
        "schema": "wan_q4k_range_bf16_control_v1",
        "decoder": metric(din * dout),
        "projection": metric(rows * dout),
    })
    for phase in ("before_timing", "after_timing"):
        for arm in results.ARMS:
            lossy = [metric(dout, bit_differences=dout, tolerance_violations=dout,
                            rmse=0.1, relative_l2=0.1, cosine=0.99)
                     for _ in range(4)]
            records.append({
                "schema": "wan_q4k_range_parity_v1",
                "phase": phase,
                "variant": arm,
                "decoder": metric(din * dout),
                "projection": metric(rows * dout),
                "lossy_vs_bf16_per_source_row": lossy,
                "calibration_rows": [],
                "unique_evaluation_rows": [0, 1, 2, 3],
            })
    records.extend(timing_records(manifest_sha256=manifest_sha))
    records.append({
        "schema": "wan_q4k_range_controls_v1",
        "corruption_rejected": True,
        "corruption": metric(rows * dout, bit_differences=2,
                              tolerance_violations=2, nonfinite=1),
        "final_status_zero": True,
    })

    stdout = base / "probe-r4.stdout.log"
    stdout.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    stderr = base / "probe-r4.stderr.log"
    stderr.write_text("")
    guard = base / "probe-r4.guard.jsonl"
    guard.write_text("".join(json.dumps(event, sort_keys=True) + "\n"
                           for event in valid_guard_events()))
    build_intent_path = base / "build-v1.intent.json"
    build_receipt_path = base / "build-v1.completion.json"
    build_intent = {
        "schema": "wan_q4k_range_intent_v1",
        "mode": "build",
        "tag": "v1",
        "command": ["crystal", "build", "-o", str(binary)],
        "files": {str(binary): results.sha(binary)},
    }
    write_json(build_intent_path, build_intent)
    build_receipt = {
        "schema": "wan_q4k_range_completion_v1",
        "returncode": 0,
        "changed_files": [],
        "intent_sha256": results.sha(build_intent_path),
        "sources": {},
        "binary_sha256": results.sha(binary),
    }
    write_json(build_receipt_path, build_receipt)

    oracle_intent_path = base / "oracle-r4.intent.json"
    oracle_receipt_path = base / "oracle-r4.completion.json"
    oracle_intent = {
        "schema": "wan_q4k_range_intent_v1",
        "mode": "oracle",
        "tag": "r4",
        "command": ["python", "reference.py", "--fixture-dir", str(source_fixture),
                    "--output-dir", str(fixture), "--rows", str(rows)],
        "files": {str(source_manifest): results.sha(source_manifest)},
    }
    write_json(oracle_intent_path, oracle_intent)
    oracle_receipt = {
        "schema": "wan_q4k_range_completion_v1",
        "returncode": 0,
        "changed_files": [],
        "intent_sha256": results.sha(oracle_intent_path),
        "sources": {},
        "produced_manifest_sha256": manifest_sha,
    }
    write_json(oracle_receipt_path, oracle_receipt)

    probe_intent_path = base / "probe-r4.intent.json"
    probe_receipt_path = base / "probe-r4.completion.json"
    files = {
        str(binary): results.sha(binary),
        str(output_manifest): results.sha(output_manifest),
        str(build_receipt_path): results.sha(build_receipt_path),
        str(oracle_receipt_path): results.sha(oracle_receipt_path),
        str(source_manifest): results.sha(source_manifest),
    }
    probe_intent = {
        "schema": "wan_q4k_range_intent_v1",
        "mode": "probe",
        "tag": "r4",
        "command": [str(binary), f"--fixture={fixture}", "--rounds=12"],
        "reverse_first": False,
        "files": files,
        "memory_accounting": "file-cache",
        "reserve_pct": 15,
        "max_footprint_mib": 4096,
        "max_swap_growth_mib": 64,
        "timeout_sec": 300,
    }
    write_json(probe_intent_path, probe_intent)
    probe_receipt = {
        "schema": "wan_q4k_range_completion_v1",
        "returncode": 0,
        "changed_files": [],
        "intent_sha256": results.sha(probe_intent_path),
        "stdout_sha256": results.sha(stdout),
        "stderr_sha256": results.sha(stderr),
        "guard_sha256": results.sha(guard),
        "sources": {},
        "binary_sha256": results.sha(binary),
    }
    write_json(probe_receipt_path, probe_receipt)
    return "r4"


class ResultAnalyzerSpec(unittest.TestCase):
    def test_inventory_count_matches_actual_12_round_sequence(self):
        timing_count = len(timing_records())
        self.assertEqual(timing_count, 84)
        self.assertEqual(results.expected_probe_record_count(timing_count), 95)

    def test_complete_synthetic_receipt_and_record_stream_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            tag = complete_synthetic_probe(evidence)
            accepted = results.read_probe(evidence, tag)
            self.assertEqual(accepted["setup"]["rows"], 4)
            self.assertTrue(accepted["all_decoder_projection_checks_bit_exact"])

    def test_metric_requires_full_element_count_and_bounded_counters(self):
        metric = valid_metric(17)
        results.validate_metric(metric, 17, "complete")
        metric["compared_values"] = 16
        with self.assertRaises(ValueError):
            results.validate_metric(metric, 17, "truncated")
        metric = valid_metric(17)
        metric["nonfinite"] = 18
        with self.assertRaises(ValueError):
            results.validate_metric(metric, 17, "impossible counter")

    def test_timing_accepts_exact_schedule_and_rejects_missing_warmup(self):
        records = timing_records()
        summary = results.validate_timing_records(
            records, {"rows": 4, "manifest_sha256": "a" * 64}, 12, False)
        self.assertEqual(summary["projection"]["arms"]["q4_k"]["gpu_seconds"]["count"], 12)
        with self.assertRaises(ValueError):
            results.validate_timing_records(
                records[1:], {"rows": 4, "manifest_sha256": "a" * 64}, 12, False)

    def test_timing_rejects_order_wait_and_nan_mutations(self):
        records = timing_records()
        records[0]["variant"] = "q4_k_minmax1"
        with self.assertRaises(ValueError):
            results.validate_timing_records(
                records, {"rows": 4, "manifest_sha256": "a" * 64}, 12, False)
        records = timing_records()
        records[0]["terminal_waits"] = 2
        with self.assertRaises(ValueError):
            results.validate_timing_records(
                records, {"rows": 4, "manifest_sha256": "a" * 64}, 12, False)
        records = timing_records()
        records[0]["gpu_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            results.validate_timing_records(
                records, {"rows": 4, "manifest_sha256": "a" * 64}, 12, False)

    def test_source_binding_rejects_missing_or_changed_repo_pin(self):
        source = str(ROOT / "src/fake.cr")
        results.validate_source_binding({source: "b" * 64}, {source: "b" * 64})
        with self.assertRaises(ValueError):
            results.validate_source_binding({source: "b" * 64}, {})
        with self.assertRaises(ValueError):
            results.validate_source_binding({source: "b" * 64}, {source: "c" * 64})

    def test_corruption_control_requires_exact_positive_failure_signature(self):
        record = {
            "corruption": {
                **valid_metric(20),
                "bit_differences": 2,
                "tolerance_violations": 2,
                "nonfinite": 1,
            },
            "corruption_rejected": True,
            "final_status_zero": True,
        }
        results.validate_corruption_control(record, 20)
        record["corruption"]["nonfinite"] = 0
        with self.assertRaises(ValueError):
            results.validate_corruption_control(record, 20)

    def test_guard_accepts_policy_and_rejects_runtime_or_startup_breach(self):
        intent = {
            "memory_accounting": "file-cache",
            "reserve_pct": 15,
            "max_footprint_mib": 4096,
            "max_swap_growth_mib": 64,
            "timeout_sec": 300,
        }
        memory = results.validate_guard(valid_guard_events(), intent)
        self.assertEqual(memory["sampled_min_pressure_free_pct"], 80)

        events = valid_guard_events()
        events[3]["pressure_free_pct"] = 14
        with self.assertRaises(ValueError):
            results.validate_guard(events, intent)

        events = valid_guard_events()
        events[1]["headroom_bytes"] = 13 * 1024**3
        with self.assertRaises(ValueError):
            results.validate_guard(events, intent)


if __name__ == "__main__":
    unittest.main()
