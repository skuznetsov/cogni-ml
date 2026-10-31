"""CPU-only fail-closed tests for the PixelUMM lifecycle matrix analyzer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "support" / "pixelumm_after_dit_matrix.py"
SPEC = importlib.util.spec_from_file_location("pixelumm_after_dit_matrix", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
analysis = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)
ORIGINAL_SHA256_FILE = analysis.sha256_file


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _guard_events(index: int, mode: str, source_manifest_sha256: str) -> list[dict]:
    physical = 64 * analysis.GIB
    return [
        {
            "event": "limits",
            "monotonic": 1.0,
            "runtime_floor_pct": 10,
            "runtime_floor_pct_exact": "10",
            "max_footprint_bytes": analysis.GUARD_LIMITS["max_footprint_mib"] * analysis.MIB,
            "max_swap_growth_bytes": analysis.GUARD_LIMITS["max_swap_growth_mib"] * analysis.MIB,
            "timeout_sec": analysis.GUARD_LIMITS["timeout_sec"],
            "memory_accounting": "file-cache",
        },
        {
            "event": "sample",
            "monotonic": 2.0,
            "pressure_free_pct": 48,
            "physical_bytes": physical,
            "headroom_bytes": 40 * analysis.GIB,
            "group_footprint_bytes": 0,
            "group_live_processes": 0,
            "swap_used_bytes": 4 * analysis.GIB,
            "file_backed_bytes": 30 * analysis.GIB,
            "free_spec_bytes": 10 * analysis.GIB,
            "speculative_bytes": 100 * analysis.MIB,
        },
        {"event": "started", "monotonic": 2.5, "pid": 9000 + index, "pgid": 9000 + index},
        {
            "event": "sample",
            "monotonic": 3.0,
            "pressure_free_pct": 45,
            "physical_bytes": physical,
            "headroom_bytes": 32 * analysis.GIB,
            "group_footprint_bytes": 20 * analysis.GIB,
            "group_live_processes": 1,
            "swap_used_bytes": 4 * analysis.GIB + 4 * analysis.MIB,
            "file_backed_bytes": 28 * analysis.GIB,
            "free_spec_bytes": 4 * analysis.GIB,
            "speculative_bytes": 100 * analysis.MIB,
        },
        {"event": "finished", "monotonic": 4.0, "returncode": 0},
        {
            "event": "matrix_cell_identity",
            "cell_index": index - 1,
            "cell_name": f"cell-{index:02d}-{mode}",
            "mode": mode,
            "binary_path": "/opt/mldata_ssd/probe",
            "binary_sha256": _digest("binary"),
            "source_manifest_sha256": source_manifest_sha256,
            "matrix_script_sha256": _digest("matrix-script"),
            "probe_source_sha256": _digest("probe-source"),
        },
    ]


def _pass(label: str, wall: float, encode: float, wait: float, gpu: float) -> dict:
    return {
        "label": label,
        "wall_seconds": wall,
        "encode_seconds": encode,
        "terminal_wait_seconds": wait,
        "gpu_seconds": gpu,
        "segments": 11734 if label == "cold" else 11692,
        "waits": 1,
        "encode_allocations": 0,
        "encode_readbacks": 0,
        "preterminal_waits": 0,
    }


def _receipt(mode: str, cell_index: int, cold: float, warm: float,
             output_dir: Path) -> dict:
    dit_named = 25_305_268_836
    decoder_named = 19_570_000_000
    outputs = {}
    for label in ("cold", "warm"):
        path = output_dir / f"{label}-display.f32le"
        with path.open("xb") as stream:
            stream.truncate(analysis.OUTPUT_BYTES)
        outputs[label] = {
            "path": str(path),
            "sha256": analysis.OUTPUT_SHA256,
            "upstream_bitwise_equal": True,
        }
    receipt = {
        "schema": "cogni_ml.pixelumm_after_dit_probe/v1",
        "mode": mode,
        "input_sha256": analysis.PINNED["input_sha256"],
        "latent_sha256": analysis.PINNED["latent_sha256"],
        "noise_sha256": analysis.PINNED["noise_sha256"],
        "oracle_report_sha256": analysis.PINNED["oracle_report_sha256"],
        "oracle_display_sha256": analysis.OUTPUT_SHA256,
        "decoder_cache_sha256": _digest("decoder-cache"),
        "cinematic_lineage_sha256": _digest("cinematic-lineage"),
        "geometry": "768x416",
        "frames": 120,
        "raw_capture": False,
        "named_budget_bytes": 24 * analysis.GIB,
        "dit_named_bytes": dit_named,
        "decoder_named_bytes": decoder_named,
        "reused_owner_ids": list(range(1, 43)),
        "passes": [
            _pass("cold", cold, cold - 2.0, 2.0, cold - 1.0),
            _pass("warm", warm, warm - 2.0, 2.0, warm - 1.0),
        ],
        "timings": {"dit_wall_seconds": 63.0, "dit_gpu_seconds": 62.0}
        if mode == "after-dit" else {},
        "outputs": outputs,
        "atol": 2e-5,
        "rtol": 2e-5,
    }
    return receipt


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(directory: Path) -> Path:
    root = directory / "matrix"
    root.mkdir()
    source_files = {
        "examples/support/pixelumm_after_dit_matrix.py": _digest("matrix-script"),
        "examples/pixelumm_after_dit_probe.cr": _digest("probe-source"),
        "examples/support/wan_memory_guard.py": _digest("guard-source"),
        "Makefile": _digest("makefile"),
    }
    manifest = hashlib.sha256(json.dumps(source_files, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    schedule = [
        {"cell_index": index, "cell_name": f"cell-{index + 1:02d}-{mode}", "mode": mode}
        for index, mode in enumerate(analysis.ORDER)
    ]
    launch = {
        "schema": "cogni_ml.pixelumm_after_dit_matrix_launch/v1",
        "output_root": str(root),
        "pause_seconds_between_cells": 30,
        "guard_limits": analysis.GUARD_LIMITS,
        "schedule": schedule,
        "source": {
            "files": source_files,
            "manifest_sha256": manifest,
            "matrix_script_sha256": source_files["examples/support/pixelumm_after_dit_matrix.py"],
            "probe_source_sha256": source_files["examples/pixelumm_after_dit_probe.cr"],
            "guard_source_sha256": source_files["examples/support/wan_memory_guard.py"],
        },
        "binary": {"path": "/opt/mldata_ssd/probe", "sha256": _digest("binary")},
    }
    _write_json(root / "launch.json", launch)
    times = [(100.0, 90.0), (110.0, 97.0), (120.0, 95.0), (105.0, 92.0)]
    for item, (cold, warm) in zip(schedule, times, strict=True):
        cell_dir = root / item["cell_name"]
        output_dir = cell_dir / "output"
        output_dir.mkdir(parents=True)
        _write_json(cell_dir / "intent.json", {
            "schema": "cogni_ml.pixelumm_after_dit_matrix_cell_intent/v1",
            **item,
            "binary": launch["binary"],
            "source_manifest_sha256": manifest,
        })
        receipt_path = output_dir / "receipt.json"
        _write_json(receipt_path, _receipt(item["mode"], item["cell_index"] + 1,
                                           cold, warm, output_dir))
        stdout = cell_dir / "stdout.log"
        stderr = cell_dir / "stderr.log"
        stdout.write_text("probe output\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        guard = cell_dir / "guard.jsonl"
        guard.write_text("".join(json.dumps(event, sort_keys=True) + "\n"
                                   for event in _guard_events(item["cell_index"] + 1,
                                                              item["mode"], manifest)), encoding="utf-8")
        completion = {
            "schema": "cogni_ml.pixelumm_after_dit_matrix_cell_completion/v1",
            "cell_index": item["cell_index"],
            "cell_name": item["cell_name"],
            "mode": item["mode"],
            "returncode": 0,
            "launch_error": None,
            "binary_sha256": launch["binary"]["sha256"],
            "source_manifest_sha256": manifest,
            "guard_sha256": analysis.sha256_file(guard),
            "stdout_sha256": analysis.sha256_file(stdout),
            "stderr_sha256": analysis.sha256_file(stderr),
            "receipt_sha256": analysis.sha256_file(receipt_path),
        }
        _write_json(cell_dir / "completion.json", completion)
    return root


def _fake_output_hash(path: Path | str) -> str:
    path = Path(path)
    if path.name in {"cold-display.f32le", "warm-display.f32le"}:
        return analysis.OUTPUT_SHA256
    return ORIGINAL_SHA256_FILE(path)


class PixelUMMAfterDiTMatrixSpec(unittest.TestCase):
    def test_accepts_complete_guarded_abba_and_reports_individual_samples_and_ratios(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _fixture(Path(temporary))
            with mock.patch.object(analysis, "sha256_file", side_effect=_fake_output_hash):
                report = analysis.analyze_matrix(root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["order"], list(analysis.ORDER))
        self.assertEqual(len(report["cells"]), 4)
        cold = report["ratios"]["cold"]["wall_seconds"]
        self.assertEqual(cold["standalone_samples"], [100.0, 105.0])
        self.assertEqual(cold["after_dit_samples"], [110.0, 120.0])
        self.assertAlmostEqual(cold["after_dit_over_standalone"], 115.0 / 102.5)
        self.assertEqual(len(cold["adjacent_pair_ratios"]), 2)
        self.assertNotEqual(report["cells"][0]["passes"]["cold"]["segments"],
                            report["cells"][0]["passes"]["warm"]["segments"])

    def test_rejects_missing_cell_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _fixture(Path(temporary))
            (root / "cell-02-after-dit" / "output" / "receipt.json").unlink()
            with mock.patch.object(analysis, "sha256_file", side_effect=_fake_output_hash):
                with self.assertRaisesRegex(analysis.ValidationError, "receipt is missing or changed"):
                    analysis.analyze_matrix(root)

    def test_rejects_changed_saved_latent_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _fixture(Path(temporary))
            receipt_path = root / "cell-03-after-dit" / "output" / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["latent_sha256"] = _digest("different-latent")
            _write_json(receipt_path, receipt)
            completion_path = root / "cell-03-after-dit" / "completion.json"
            completion = json.loads(completion_path.read_text())
            completion["receipt_sha256"] = analysis.sha256_file(receipt_path)
            _write_json(completion_path, completion)
            with mock.patch.object(analysis, "sha256_file", side_effect=_fake_output_hash):
                with self.assertRaisesRegex(analysis.ValidationError, "latent_sha256 mismatch"):
                    analysis.analyze_matrix(root)

    def test_rejects_extra_wait_even_when_process_and_guard_succeeded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _fixture(Path(temporary))
            receipt_path = root / "cell-02-after-dit" / "output" / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["passes"][1]["waits"] = 2
            _write_json(receipt_path, receipt)
            completion_path = root / "cell-02-after-dit" / "completion.json"
            completion = json.loads(completion_path.read_text())
            completion["receipt_sha256"] = analysis.sha256_file(receipt_path)
            _write_json(completion_path, completion)
            with mock.patch.object(analysis, "sha256_file", side_effect=_fake_output_hash):
                with self.assertRaisesRegex(analysis.ValidationError, "exactly one terminal wait"):
                    analysis.analyze_matrix(root)

    def test_rejects_guard_reserve_breach(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _fixture(Path(temporary))
            cell = root / "cell-01-standalone"
            guard_path = cell / "guard.jsonl"
            events = [json.loads(line) for line in guard_path.read_text().splitlines()]
            events[3]["pressure_free_pct"] = 9
            guard_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n"
                                                 for event in events), encoding="utf-8")
            completion_path = cell / "completion.json"
            completion = json.loads(completion_path.read_text())
            completion["guard_sha256"] = analysis.sha256_file(guard_path)
            _write_json(completion_path, completion)
            with mock.patch.object(analysis, "sha256_file", side_effect=_fake_output_hash):
                with self.assertRaisesRegex(analysis.ValidationError, "pressure fell below 10%"):
                    analysis.analyze_matrix(root)


if __name__ == "__main__":
    unittest.main()
