"""CPU-only validation tests for the PixelUMM recovery matrix."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "support" / "pixelumm_recovery_matrix.py"
SPEC = importlib.util.spec_from_file_location("pixelumm_recovery_matrix", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)

RECOVERY_CLOCK = "CLOCK_MONOTONIC_RAW(4)"


def valid_passes(order: str = "short-long") -> list[dict[str, object]]:
    sequence = [("warmup", 0)] + [
        (label, 1 if label == "short" else 120) for label in order.split("-")
    ]
    result = []
    epoch = 1_800_000_000_000
    previous_finished_ms = 0
    monotonic_ns = 100_000_000_000
    previous_finished_ns = 0
    for output_index, (label, requested) in enumerate(sequence):
        observed = 0 if requested == 0 else requested + 0.25
        if output_index:
            epoch = previous_finished_ms + int(observed * 1_000)
            monotonic_ns = previous_finished_ns + int(observed * 1_000_000_000)
        finished_ms = epoch + 8_000
        finished_ns = monotonic_ns + 8_000_000_000
        result.append({
            "label": label,
            "requested_recovery_seconds": requested,
            "observed_recovery_seconds": observed,
            "wall_seconds": 8.0,
            "encode_seconds": 5.0,
            "terminal_wait_seconds": 1.0,
            "gpu_seconds": 3.0 + output_index,
            "segments": 3,
            "waits": 1,
            "encode_allocations": 0,
            "encode_readbacks": 0,
            "preterminal_waits": 0,
            "thermal_before": 0,
            "thermal_after": 1,
            "output_index": output_index,
            "started_unix_ms": epoch,
            "finished_unix_ms": finished_ms,
            "started_monotonic_ns": monotonic_ns,
            "finished_monotonic_ns": finished_ns,
        })
        previous_finished_ms = finished_ms
        previous_finished_ns = finished_ns
    return result


class PassValidationSpec(unittest.TestCase):
    def test_accepts_warmup_then_both_recovery_orders(self):
        for order in ("short-long", "long-short"):
            with self.subTest(order=order):
                got = matrix._validate_passes(valid_passes(order), order, "cell")
                self.assertEqual(list(got), ["warmup", *order.split("-")])

    def test_rejects_wrong_pass_order_and_output_index(self):
        passes = valid_passes("short-long")
        with self.assertRaisesRegex(matrix.ValidationError, "pass order"):
            matrix._validate_passes(passes, "long-short", "cell")

        passes = valid_passes()
        passes[1]["output_index"] = 2
        with self.assertRaisesRegex(matrix.ValidationError, "output_index"):
            matrix._validate_passes(passes, "short-long", "cell")

    def test_rejects_underreported_recovery_and_inconsistent_timing(self):
        passes = valid_passes()
        passes[1]["observed_recovery_seconds"] = 0.99
        with self.assertRaisesRegex(matrix.ValidationError, "observed recovery"):
            matrix._validate_passes(passes, "short-long", "cell")

        passes = valid_passes()
        passes[2]["wall_seconds"] = 2.0
        with self.assertRaisesRegex(matrix.ValidationError, "shorter than encode"):
            matrix._validate_passes(passes, "short-long", "cell")

    def test_rejects_nonmonotonic_clock_and_thermal_out_of_range(self):
        passes = valid_passes()
        passes[2]["started_monotonic_ns"] = passes[1]["finished_monotonic_ns"] - 1
        with self.assertRaisesRegex(matrix.ValidationError, "monotonic"):
            matrix._validate_passes(passes, "short-long", "cell")

        passes = valid_passes()
        passes[0]["thermal_after"] = 4
        with self.assertRaisesRegex(matrix.ValidationError, "thermal_after"):
            matrix._validate_passes(passes, "short-long", "cell")

    def test_rejects_overslept_treatments_and_bad_derived_recovery_interval(self):
        for label, value in (("short", 5.01), ("long", 125.01)):
            passes = valid_passes()
            pass_index = [entry["label"] for entry in passes].index(label)
            prior_finished = passes[pass_index - 1]["finished_monotonic_ns"]
            passes[pass_index]["started_monotonic_ns"] = prior_finished + int(value * 1e9)
            passes[pass_index]["observed_recovery_seconds"] = value
            passes[pass_index]["finished_monotonic_ns"] = (
                passes[pass_index]["started_monotonic_ns"] + 8_000_000_000
            )
            with self.subTest(label=label), self.assertRaisesRegex(matrix.ValidationError, "exceeded"):
                matrix._validate_passes(passes, "short-long", "cell")

        passes = valid_passes()
        passes[1]["observed_recovery_seconds"] += 0.000002
        with self.assertRaisesRegex(matrix.ValidationError, "disagrees with the monotonic"):
            matrix._validate_passes(passes, "short-long", "cell")

    def test_rejects_wall_duration_that_disagrees_with_monotonic_clock(self):
        passes = valid_passes()
        passes[1]["finished_monotonic_ns"] += 20_000_000
        with self.assertRaisesRegex(matrix.ValidationError, "wall_seconds disagrees"):
            matrix._validate_passes(passes, "short-long", "cell")

    def test_preserves_optional_raw_clock_and_wall_gap_fields(self):
        passes = valid_passes()
        passes[1]["raw_clock_trace_ns"] = [108_000_000_000, 109_250_000_000]
        passes[1]["wall_gap_seconds"] = 1.25
        got = matrix._validate_passes(passes, "short-long", "cell")
        self.assertEqual(got["short"]["raw_clock_trace_ns"], [108_000_000_000, 109_250_000_000])
        self.assertEqual(got["short"]["wall_gap_seconds"], 1.25)

    def test_rejects_nonzero_encode_allocations_readbacks_or_preterminal_waits(self):
        for field in ("encode_allocations", "encode_readbacks", "preterminal_waits"):
            passes = valid_passes()
            passes[1][field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(matrix.ValidationError, field):
                matrix._validate_passes(passes, "short-long", "cell")


class OutputValidationSpec(unittest.TestCase):
    def test_rehashes_all_three_output_files_and_rejects_digest_mismatch(self):
        payload = b"pinned output fixture"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            outputs = []
            for index, label in enumerate(("warmup", "short", "long")):
                path = output_dir / f"output-{index}.bin"
                path.write_bytes(payload)
                outputs.append({
                    "label": label,
                    "output_index": index,
                    "path": str(path),
                    "sha256": digest,
                    "upstream_bitwise_equal": True,
                    "exported_unix_ms": 1_800_000_500_000,
                })
            with patch.object(matrix, "OUTPUT_BYTES", len(payload)), \
                    patch.object(matrix, "OUTPUT_SHA256", digest):
                matrix._validate_outputs(outputs, output_dir, ("warmup", "short", "long"), "cell",
                                         1_800_000_300_000)
                outputs[2]["sha256"] = "0" * 64
                with self.assertRaisesRegex(matrix.ValidationError, "pinned upstream oracle"):
                    matrix._validate_outputs(outputs, output_dir,
                                             ("warmup", "short", "long"), "cell",
                                             1_800_000_300_000)
                outputs[2]["sha256"] = digest
                outputs[2]["exported_unix_ms"] = 1_800_000_200_000
                with self.assertRaisesRegex(matrix.ValidationError, "before all passes"):
                    matrix._validate_outputs(outputs, output_dir,
                                             ("warmup", "short", "long"), "cell",
                                             1_800_000_300_000)

    def test_rejects_output_order_and_paths_outside_cell(self):
        outputs = [
            {"label": label, "output_index": index, "path": "/tmp/a", "sha256": "0" * 64,
             "upstream_bitwise_equal": True, "exported_unix_ms": 1_800_000_000_000}
            for index, label in enumerate(("warmup", "short", "long"))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            outputs[1]["label"] = "long"
            with self.assertRaisesRegex(matrix.ValidationError, "output labels/order"):
                matrix._validate_outputs(outputs, output_dir, ("warmup", "short", "long"), "cell", 1)

            outputs[1]["label"] = "short"
            with patch.object(matrix, "OUTPUT_BYTES", 1), patch.object(matrix, "OUTPUT_SHA256", "0" * 64):
                with self.assertRaisesRegex(matrix.ValidationError, "missing or symlinked"):
                    matrix._validate_outputs(outputs, output_dir,
                                             ("warmup", "short", "long"), "cell", 1)


class ReceiptValidationSpec(unittest.TestCase):
    def test_requires_pinned_identity_geometry_budget_and_owner_set(self):
        receipt = valid_receipt()
        receipt["decoder_named_bytes"] = receipt["named_budget_bytes"] + 1
        with self.assertRaisesRegex(matrix.ValidationError, "24 GiB budget"):
            matrix._validate_receipt(receipt, "short-long", Path("/tmp/output"), "cell")

        receipt = valid_receipt()
        receipt["reused_owner_ids"][-1] = receipt["reused_owner_ids"][0]
        with self.assertRaisesRegex(matrix.ValidationError, "owner identity"):
            matrix._validate_receipt(receipt, "short-long", Path("/tmp/output"), "cell")

        receipt = valid_receipt()
        receipt["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(matrix.ValidationError, "input_sha256"):
            matrix._validate_receipt(receipt, "short-long", Path("/tmp/output"), "cell")

    def test_requires_exact_recovery_clock_and_preserves_adjusted_wall_diagnostics(self):
        receipt = valid_receipt()
        receipt["passes"][1]["observed_recovery_adjusted_seconds"] = 1.275
        receipt["passes"][1]["observed_recovery_wall_seconds"] = 1.25
        with patch.object(matrix, "_validate_outputs"):
            validated = matrix._validate_receipt(receipt, "short-long", Path("/tmp/output"), "cell")
        self.assertEqual(validated["recovery_clock"], RECOVERY_CLOCK)
        self.assertEqual(
            validated["passes"]["short"]["observed_recovery_adjusted_seconds"], 1.275
        )
        self.assertEqual(
            validated["passes"]["short"]["observed_recovery_wall_seconds"], 1.25
        )

        for bad_clock in ("CLOCK_MONOTONIC(6)", None):
            malformed = valid_receipt()
            if bad_clock is None:
                del malformed["recovery_clock"]
            else:
                malformed["recovery_clock"] = bad_clock
            with self.subTest(recovery_clock=bad_clock), patch.object(matrix, "_validate_outputs"):
                with self.assertRaisesRegex(matrix.ValidationError, "recovery_clock"):
                    matrix._validate_receipt(
                        malformed, "short-long", Path("/tmp/output"), "cell"
                    )

    def test_shared_identity_includes_recovery_clock(self):
        shared = {
            "input_sha256": "1" * 64,
            "latent_sha256": "2" * 64,
            "noise_sha256": "3" * 64,
            "oracle_report_sha256": "4" * 64,
            "oracle_display_sha256": "5" * 64,
            "decoder_cache_sha256": "6" * 64,
            "geometry": "768x416",
            "frames": 120,
            "raw_capture": False,
            "atol": 2e-5,
            "rtol": 2e-5,
            "named_budget_bytes": 24 * matrix.GIB,
        }
        cells = [
            {"receipt": {**shared, "recovery_clock": RECOVERY_CLOCK}},
            {"receipt": {**shared, "recovery_clock": "CLOCK_MONOTONIC(6)"}},
        ]
        with self.assertRaisesRegex(matrix.ValidationError, "shared receipt field recovery_clock"):
            matrix._validate_shared_identity(cells)


def valid_receipt() -> dict[str, object]:
    return {
        "schema": "cogni_ml.pixelumm_recovery_probe/v1",
        "recovery_clock": RECOVERY_CLOCK,
        **matrix.PINNED,
        "oracle_display_sha256": matrix.OUTPUT_SHA256,
        "decoder_cache_sha256": "a" * 64,
        "geometry": "768x416",
        "frames": 120,
        "raw_capture": False,
        "dit_constructed": False,
        "atol": 2e-5,
        "rtol": 2e-5,
        "named_budget_bytes": 24 * matrix.GIB,
        "decoder_named_bytes": 8 * matrix.GIB,
        "reused_owner_ids": list(range(1, 43)),
        "recovery_order": "short-long",
        "passes": valid_passes(),
        "outputs": [],
        "timings": {"decoder_setup_seconds": 2.5},
    }


class GuardAndIdentitySpec(unittest.TestCase):
    def _valid_guard(self, path: Path):
        source = {
            "manifest_sha256": "a" * 64,
            "matrix_script_sha256": "b" * 64,
            "probe_source_sha256": "c" * 64,
        }
        launch = {
            "binary": {"path": "/tmp/probe", "sha256": "d" * 64},
            "source": source,
        }
        item = {"cell_index": 1, "cell_name": "cell-01-short-long", "mode": "short-long"}
        limits = matrix.MATRIX.GUARD_LIMITS
        physical = 64 * matrix.GIB
        cap = limits["max_footprint_mib"] * matrix.MIB
        floor = (physical + 9) // 10
        events = [
            {
                "event": "limits", "runtime_floor_pct": 10, "runtime_floor_pct_exact": "10",
                "max_footprint_bytes": cap,
                "max_swap_growth_bytes": limits["max_swap_growth_mib"] * matrix.MIB,
                "timeout_sec": limits["timeout_sec"], "memory_accounting": "file-cache",
            },
            {
                "event": "sample", "pressure_free_pct": 80, "physical_bytes": physical,
                "headroom_bytes": floor + cap, "group_footprint_bytes": cap // 2,
                "swap_used_bytes": 10 * matrix.MIB,
            },
            {"event": "started"},
            {
                "event": "sample", "pressure_free_pct": 80, "physical_bytes": physical,
                "headroom_bytes": floor, "group_footprint_bytes": cap // 2,
                "swap_used_bytes": 10 * matrix.MIB,
            },
            {"event": "finished", "returncode": 0},
            {
                "event": "matrix_cell_identity", "cell_index": 1,
                "cell_name": "cell-01-short-long", "mode": "short-long",
                "binary_path": "/tmp/probe", "binary_sha256": "d" * 64,
                "source_manifest_sha256": "a" * 64, "matrix_script_sha256": "b" * 64,
                "probe_source_sha256": "c" * 64,
            },
        ]
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        completion = {"guard_sha256": matrix._sha256_file(path), "returncode": 0}
        return path, launch, item, completion

    def test_reuses_completed_guard_validator_and_rejects_wrong_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, launch, item, completion = self._valid_guard(Path(temporary) / "guard.jsonl")
            matrix.MATRIX._load_guard(path, launch, item, completion)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            events[0]["max_swap_growth_bytes"] -= 1
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            completion["guard_sha256"] = matrix._sha256_file(path)
            with self.assertRaisesRegex(matrix.ValidationError, "guard limit mismatch"):
                matrix.MATRIX._load_guard(path, launch, item, completion)

    def test_rejects_guard_identity_mismatch_and_live_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, launch, item, completion = self._valid_guard(Path(temporary) / "guard.jsonl")
            events = [json.loads(line) for line in path.read_text().splitlines()]
            events[-1]["mode"] = "long-short"
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            completion["guard_sha256"] = matrix._sha256_file(path)
            with self.assertRaisesRegex(matrix.ValidationError, "identity mismatch: mode"):
                matrix.MATRIX._load_guard(path, launch, item, completion)

        launch = {"binary": {"path": "/tmp/probe", "sha256": "a" * 64},
                  "source": {"files": {"f": "b" * 64}}}
        with patch.object(matrix, "_binary_identity", return_value=launch["binary"]), \
                patch.object(matrix, "_source_identity", return_value={"files": {"f": "c" * 64}}):
            with self.assertRaisesRegex(matrix.ValidationError, "source identity changed"):
                matrix._check_live_identities(launch)


class PlanSpec(unittest.TestCase):
    def test_plan_only_has_two_fresh_cells_fixed_order_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "not-created"
            source = {
                "manifest_sha256": "a" * 64,
                "files": {},
                "matrix_script_sha256": "b" * 64,
                "probe_source_sha256": "c" * 64,
                "guard_source_sha256": "d" * 64,
            }
            binary = {"path": "/tmp/probe", "sha256": "e" * 64, "size_bytes": 1,
                      "device": 1, "inode": 1, "mode": 0o755}
            with patch.object(matrix, "_binary_identity", return_value=binary), \
                    patch.object(matrix, "_output_root_candidate", return_value=output_root), \
                    patch.object(matrix, "_source_identity", return_value=source):
                plan = matrix.dry_run_plan("/tmp/probe", str(output_root))
            self.assertFalse(output_root.exists())
            self.assertEqual(
                [cell["recovery_order"] for cell in plan["launch"]["schedule"]],
                ["short-long", "long-short"],
            )
            self.assertEqual(len(plan["cells"]), 2)
            self.assertIn("--recovery-order", plan["cells"][0]["guard_command"][-4:])

    def test_manifest_requires_probe_bridge_and_launcher_sources(self):
        required = set(matrix.SOURCE_FILES)
        self.assertIn("examples/pixelumm_recovery_probe.cr", required)
        self.assertIn("examples/support/pixelumm_saved_latent.cr", required)
        self.assertIn("examples/support/pixelumm_recovery_host.mm", required)
        self.assertIn("examples/support/pixelumm_recovery_timing.cr", required)
        self.assertIn("examples/support/pixelumm_recovery_matrix.py", required)

    def test_timing_helper_is_hashed_and_validated_in_source_manifest(self):
        files = {
            path: hashlib.sha256(path.encode("utf-8")).hexdigest()
            for path in matrix.SOURCE_FILES
        }
        source = {
            "files": files,
            "manifest_sha256": matrix._sha256_bytes(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "matrix_script_sha256": files["examples/support/pixelumm_after_dit_matrix.py"],
            "recovery_script_sha256": files["examples/support/pixelumm_recovery_matrix.py"],
            "probe_source_sha256": files["examples/pixelumm_recovery_probe.cr"],
            "fixture_source_sha256": files["examples/support/pixelumm_saved_latent.cr"],
            "host_observer_sha256": files["examples/support/pixelumm_recovery_host.mm"],
            "recovery_timing_sha256": files["examples/support/pixelumm_recovery_timing.cr"],
            "guard_source_sha256": files["examples/support/wan_memory_guard.py"],
        }
        matrix._validate_source_manifest(source)

        missing_timing = dict(source)
        missing_timing["files"] = dict(files)
        missing_timing["files"].pop("examples/support/pixelumm_recovery_timing.cr")
        missing_timing["manifest_sha256"] = matrix._sha256_bytes(
            json.dumps(missing_timing["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        with self.assertRaisesRegex(matrix.ValidationError, "omits a required source"):
            matrix._validate_source_manifest(missing_timing)

        mismatched_timing = dict(source)
        mismatched_timing["recovery_timing_sha256"] = "0" * 64
        with self.assertRaisesRegex(matrix.ValidationError, "recovery_timing_sha256"):
            matrix._validate_source_manifest(mismatched_timing)

    def test_records_failure_if_launch_manifest_creation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "matrix"
            plan = {"launch": {"output_root": str(output_root)}}
            write_new_json = matrix._write_new_json

            def fail_launch_manifest(path, value):
                if Path(path).name == "launch.json":
                    raise OSError("fixture write failure")
                return write_new_json(path, value)

            with patch.object(matrix, "dry_run_plan", return_value=plan), \
                    patch.object(matrix, "_write_new_json", side_effect=fail_launch_manifest):
                with self.assertRaisesRegex(OSError, "fixture write failure"):
                    matrix.launch_matrix("/tmp/probe", str(output_root))
            failure = json.loads((output_root / "failure.json").read_text(encoding="utf-8"))
            self.assertFalse(failure["passed"])
            self.assertEqual(failure["completed_cells"], 0)

    def test_chunked_external_recovery_records_complete_wait_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wait.jsonl"
            path.write_text("", encoding="utf-8")
            item = {"cell_index": 1, "cell_name": "cell-01-short-long",
                    "recovery_order": "short-long"}
            clock = [50.0]

            def sleep_for(seconds):
                self.assertLessEqual(seconds, 30.0)
                clock[0] += seconds

            actual = matrix._wait_for_recovery(
                path, item, sleep_fn=sleep_for, monotonic_fn=lambda: clock[0]
            )
            self.assertEqual(actual, 120.0)
            events = matrix._validate_wait_log(path, item, 120, actual)
            self.assertEqual(sum(event["event"] == "recovery_wait_progress" for event in events), 4)


if __name__ == "__main__":
    unittest.main()
