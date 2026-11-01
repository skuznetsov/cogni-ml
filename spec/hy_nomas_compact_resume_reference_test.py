"""CPU-only falsifiers for repeated compact continuation provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_compact_resume_reference as comparison


SEGMENTS = 3
STEPS = 4
TIMELINE_SHAPE = [1, 32, 120, 9, 16]
STEP_SHAPE = [1, 32, 30, 9, 16]
SEGMENT_LATENTS = 30
FRAME_VALUES = 9 * 16
F32_BYTES = 4


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return _sha_file(path)


def _f32_fill(count: int, value: float) -> bytes:
    return struct.pack("<f", value) * count


def _slice_time(payload: bytes, shape: list[int], start: int, stop: int) -> bytes:
    channels, frames, height, width = shape[1], shape[2], shape[3], shape[4]
    channel_stride = frames * height * width * F32_BYTES
    frame_stride = height * width * F32_BYTES
    return b"".join(
        payload[channel * channel_stride + start * frame_stride:
                channel * channel_stride + stop * frame_stride]
        for channel in range(channels)
    )


def _timeline(source: bytes, stop: int, treatment: str) -> bytes:
    result = bytearray(source)
    frames = TIMELINE_SHAPE[2]
    segment_values = SEGMENT_LATENTS * FRAME_VALUES
    for segment in range(stop):
        value = _step_value(segment, STEPS - 1, treatment)
        block = _f32_fill(segment_values, value)
        for channel in range(TIMELINE_SHAPE[1]):
            start_value = channel * frames * FRAME_VALUES + segment * segment_values
            start = start_value * F32_BYTES
            result[start:start + len(block)] = block
    return bytes(result)


def _step_value(segment: int, step: int, treatment: str) -> float:
    value = 0.125 + 0.2 * segment + 0.01 * step
    if treatment == "qbit" and segment > 0:
        value += 0.001 * segment
    return value


def _write_manifest(root: Path) -> tuple[Path, str, Path, str, bytes]:
    values = 32 * TIMELINE_SHAPE[2] * FRAME_VALUES
    bank = _f32_fill(values, -0.0625)
    payload_path = root / "inputs.tensors.bin"
    record_offset = 16
    payload = b"HY-NOMAS-INPUT\0\0" + bank + b"END"
    payload_sha = _write(payload_path, payload)
    manifest = {
        "schema": "cogni_ml.hy_nomas_continuation_inputs/v1",
        "payload": payload_path.name,
        "payload_bytes": len(payload),
        "payload_sha256": payload_sha,
        "tensors": {
            "input.latent_bank": {
                "dtype": "F32",
                "offset": record_offset,
                "nbytes": len(bank),
                "sha256": _sha_bytes(bank),
                "shape": TIMELINE_SHAPE,
            }
        },
    }
    manifest_path = root / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, _sha_file(manifest_path), payload_path, payload_sha, bank


def _write_run(
    root: Path,
    *,
    name: str,
    pid: int,
    manifest_path: Path,
    inputs_sha: str,
    payload_sha: str,
    bank: bytes,
    treatment: str,
    state_format: str,
    start: int,
    stop: int,
    parent_state_sha: str = "",
    state_bytes: int | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    qbit = treatment == "qbit" or state_format == "qbit"
    diagnostics: dict[str, str] = {}
    timeline_payload = _timeline(bank, stop, treatment)
    timeline_path = run_dir / "timeline.f32le"
    diagnostics["timeline"] = str(timeline_path)
    diagnostics["timeline_sha256"] = _write(timeline_path, timeline_payload)
    for segment in range(start, stop):
        for step in range(STEPS):
            key = f"segment{segment}-step{step + 1}"
            step_payload = _f32_fill(
                32 * SEGMENT_LATENTS * FRAME_VALUES,
                _step_value(segment, step, treatment),
            )
            artifact = run_dir / f"{key}.f32le"
            diagnostics[key] = str(artifact)
            diagnostics[f"{key}_sha256"] = _write(artifact, step_payload)

    state_sha = ""
    checkpoint_file_bytes = 0
    logical_packed_kv_bytes = 0
    codec: dict[str, dict[str, int]] = {}
    if qbit and state_format == "qbit":
        if start > 0:
            codec["restore"] = {
                "invalid_rows": 0,
                "logical_packed_bytes": 100,
                "exact_bytes": 1_000,
            }
        if stop < SEGMENTS:
            codec["pack"] = {
                "invalid_rows": 0,
                "logical_packed_bytes": 100,
                "exact_bytes": 1_000,
                "logical_packed_kv_bytes": 100,
            }
            logical_packed_kv_bytes = 100
    elif qbit:
        for boundary in range(start + 1, stop):
            codec[f"roundtrip{boundary}"] = {
                "invalid_rows": 0,
                "logical_packed_bytes": 100,
                "exact_bytes": 1_000,
                "logical_packed_kv_bytes": 100,
            }

    if state_bytes is not None:
        suffix = "hnqc" if state_format == "qbit" else "hnrs"
        state_path = run_dir / f"boundary.{suffix}"
        state_payload = (f"{state_format}-state-{stop}".encode("ascii")
                         + bytes(max(0, state_bytes - len(f"{state_format}-state-{stop}"))))
        state_sha = _write(state_path, state_payload)
        checkpoint_file_bytes = len(state_payload)
        diagnostics["state"] = str(state_path)
        diagnostics["state_sha256"] = state_sha

    contract_map = {
        str(boundary): _sha_bytes(f"boundary-contract-{boundary}".encode())
        for boundary in range(1, SEGMENTS)
    }
    if start == 0:
        mode = "control" if stop == SEGMENTS else "save"
        contract_sha = ""
    else:
        mode = "resume"
        contract_sha = contract_map[str(start)]
    receipt = {
        "schema": "cogni_ml.hy_nomas_compact_sequence_probe/v1",
        "mode": mode,
        "pid": pid,
        "executable_sha256": _sha_bytes(b"one frozen executable"),
        "inputs": str(manifest_path),
        "inputs_sha256": inputs_sha,
        "payload_sha256": payload_sha,
        "conditioning_sha256": _sha_bytes(b"same conditioning"),
        "model_index_sha256": _sha_bytes(b"same model index"),
        "contract_sha256": contract_sha,
        "boundary_contract_sha256": contract_map,
        "state_input_sha256": parent_state_sha,
        "state_format": state_format,
        "cache_roundtrip": "qbit" if treatment == "qbit" and state_format == "exact" else "exact",
        "qbit_roundtrip_boundaries": (
            list(range(1, SEGMENTS)) if treatment == "qbit" and state_format == "exact" else []
        ),
        "upstream_revision": "911cea6458a89c2663513ff7099561ec190dd491",
        "segments": SEGMENTS,
        "start_segment": start,
        "stop_after_segments": stop,
        "resume_next_segment": start,
        "steps_per_segment": STEPS,
        "valid_text_tokens": 557,
        "timeline_shape": TIMELINE_SHAPE,
        "step_shape": STEP_SHAPE,
        "checkpoint_file_bytes": checkpoint_file_bytes,
        "logical_packed_kv_bytes": logical_packed_kv_bytes,
        "resident_f32_kv_bytes": 1_000,
        "checkpoint_payload_bytes": (
            len(bank) + 1_000 if state_bytes is not None else 0
        ),
        "codec": codec,
        "compact_pack_boundary": stop if qbit and state_format == "qbit" and stop < SEGMENTS else 0,
        "metrics": {
            "encode_allocations": 0,
            "encode_readbacks": 0,
            "preterminal_waits": 0,
            "command_segments": 20,
            "terminal_waits": 1,
        },
        "diagnostics": diagnostics,
        "decoder": False,
        "full_generation": False,
        "full_upstream_video_parity": False,
    }
    (run_dir / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def _fixture(root: Path) -> Path:
    manifest_path, inputs_sha, payload_path, payload_sha, bank = _write_manifest(root)
    state_shas: dict[str, list[str]] = {"exact": [], "qbit": []}
    pids = iter(range(21001, 21020))

    def write_arm(
        key: str,
        treatment: str,
        state_format: str,
        intervals: list[tuple[int, int]],
    ) -> dict:
        runs = []
        parent_sha = ""
        previous = ""
        for index, (start, stop) in enumerate(intervals):
            name = f"{key}-run-{index}"
            needs_state = stop < SEGMENTS
            state_size = (300 if state_format == "qbit" else 1_600) if needs_state else None
            run_dir = _write_run(
                root,
                name=name,
                pid=next(pids),
                manifest_path=manifest_path,
                inputs_sha=inputs_sha,
                payload_sha=payload_sha,
                bank=bank,
                treatment=treatment,
                state_format=state_format,
                start=start,
                stop=stop,
                parent_state_sha=parent_sha,
                state_bytes=state_size,
            )
            receipt_path = run_dir / "receipt.json"
            runs.append({"receipt_path": str(receipt_path), "interval": [start, stop]})
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if needs_state:
                parent_sha = receipt["diagnostics"]["state_sha256"]
            previous = str(receipt_path)
        return {"runs": runs}

    arms = {
        "exact_control": write_arm("exact-control", "exact", "exact", [(0, SEGMENTS)]),
        "exact_restart_chain": write_arm(
            "exact-chain", "exact", "exact", [(0, 1), (1, 2), (2, 3)]
        ),
        "qbit_control": write_arm("qbit-control", "qbit", "exact", [(0, SEGMENTS)]),
        "compact_restart_chain": write_arm(
            "compact-chain", "qbit", "qbit", [(0, 1), (1, 2), (2, 3)]
        ),
    }
    index = {
        "schema": "cogni_ml.hy_nomas_compact_resume_matrix/v1",
        "segments": SEGMENTS,
        "steps_per_segment": STEPS,
        "arms": arms,
        "source_bank": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": inputs_sha,
            "payload_path": str(payload_path),
            "payload_sha256": payload_sha,
            "record_offset": 16,
            "nbytes": len(bank),
            "shape": TIMELINE_SHAPE,
        },
    }
    index_path = root / "matrix.json"
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n", encoding="utf-8")
    return index_path


def _read_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_index(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _edit_receipt(run_entry: dict, edit) -> None:
    receipt_path = Path(run_entry["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    edit(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_artifact_sha(run_entry: dict, key: str) -> None:
    receipt_path = Path(run_entry["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    path = Path(receipt["diagnostics"][key])
    receipt["diagnostics"][f"{key}_sha256"] = _sha_file(path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")


class CompactResumeReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hy-nomas-compact-resume-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index_path = _fixture(self.root)

    def index(self) -> dict:
        return _read_index(self.index_path)

    def compare(self) -> dict:
        return comparison.compare_index(self.index_path)

    def test_accepts_exact_and_compact_chains_with_separate_lossy_diagnostics(self) -> None:
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertTrue(report["exact_restart"]["all_steps_bitwise_equal"])
        self.assertTrue(report["exact_restart"]["timeline_bitwise_equal"])
        self.assertTrue(report["compact_restart"]["all_steps_bitwise_equal"])
        self.assertTrue(report["compact_restart"]["timeline_bitwise_equal"])
        self.assertEqual(set(report["lossy_vs_exact"]["segments"]), {"1", "2"})
        self.assertFalse(report["lossy_vs_exact"]["segments"]["1"]["steps"]["1"]["within_tolerance"])
        self.assertFalse(report["lossy_vs_exact"]["segments"]["2"]["steps"]["4"]["within_tolerance"])
        self.assertEqual(report["tolerance"], {"atol": 2e-5, "rtol": 2e-5})

    def test_rejects_a_stale_or_mismatched_interval_entry(self) -> None:
        index = self.index()
        index["arms"]["compact_restart_chain"]["runs"][1]["interval"] = [0, 2]
        _write_index(self.index_path, index)
        with self.assertRaisesRegex(ValueError, "interval"):
            self.compare()

    def test_rejects_a_gap_or_duplicate_interval_in_restart_chain(self) -> None:
        index = self.index()
        index["arms"]["exact_restart_chain"]["runs"][1]["interval"] = [2, 3]
        _write_index(self.index_path, index)
        with self.assertRaisesRegex(ValueError, "contiguous|interval"):
            self.compare()

    def test_rejects_a_parent_state_sha_not_linked_to_prior_checkpoint(self) -> None:
        run = self.index()["arms"]["compact_restart_chain"]["runs"][1]
        _edit_receipt(run, lambda receipt: receipt.update(state_input_sha256="a" * 64))
        with self.assertRaisesRegex(ValueError, "state.*match|chain"):
            self.compare()

    def test_rejects_a_restart_timeline_that_mutates_its_generated_prefix(self) -> None:
        run = self.index()["arms"]["compact_restart_chain"]["runs"][1]
        receipt = json.loads(Path(run["receipt_path"]).read_text(encoding="utf-8"))
        timeline = Path(receipt["diagnostics"]["timeline"])
        with timeline.open("r+b") as stream:
            stream.write(struct.pack("<f", 99.0))
        _refresh_artifact_sha(run, "timeline")
        with self.assertRaisesRegex(ValueError, "already-generated prefix"):
            self.compare()

    def test_rejects_a_stale_step_even_when_its_digest_is_refreshed(self) -> None:
        run = self.index()["arms"]["compact_restart_chain"]["runs"][1]
        receipt = json.loads(Path(run["receipt_path"]).read_text(encoding="utf-8"))
        step = Path(receipt["diagnostics"]["segment1-step2"])
        with step.open("r+b") as stream:
            stream.write(struct.pack("<f", 99.0))
        _refresh_artifact_sha(run, "segment1-step2")
        with self.assertRaisesRegex(ValueError, "step|bitwise"):
            self.compare()

    def test_rejects_timeline_that_mutates_unexecuted_source_noise(self) -> None:
        run = self.index()["arms"]["qbit_control"]["runs"][0]
        receipt = json.loads(Path(run["receipt_path"]).read_text(encoding="utf-8"))
        timeline = Path(receipt["diagnostics"]["timeline"])
        with timeline.open("r+b") as stream:
            frame_offset = 3 * SEGMENT_LATENTS * FRAME_VALUES * F32_BYTES
            stream.seek(frame_offset)
            stream.write(struct.pack("<f", 99.0))
        _refresh_artifact_sha(run, "timeline")
        with self.assertRaisesRegex(ValueError, "future|source bank|unexecuted"):
            self.compare()

    def test_rejects_conditioning_or_noise_identity_mismatch(self) -> None:
        run = self.index()["arms"]["compact_restart_chain"]["runs"][1]
        _edit_receipt(run, lambda receipt: receipt.update(conditioning_sha256="b" * 64))
        with self.assertRaisesRegex(ValueError, "identity|conditioning"):
            self.compare()

    def test_rejects_reused_process_identity(self) -> None:
        index = self.index()
        source = index["arms"]["exact_control"]["runs"][0]
        target = index["arms"]["qbit_control"]["runs"][0]
        source_receipt = json.loads(Path(source["receipt_path"]).read_text(encoding="utf-8"))
        _edit_receipt(target, lambda receipt: receipt.update(pid=source_receipt["pid"]))
        with self.assertRaisesRegex(ValueError, "distinct process IDs|fresh processes"):
            self.compare()

    def test_rejects_uninstrumented_encode_allocation_or_extra_wait(self) -> None:
        run = self.index()["arms"]["compact_restart_chain"]["runs"][2]
        _edit_receipt(
            run,
            lambda receipt: receipt["metrics"].update(encode_allocations=1),
        )
        with self.assertRaisesRegex(ValueError, "encode_allocations"):
            self.compare()

    def test_rejects_multiple_terminal_waits(self) -> None:
        run = self.index()["arms"]["compact_restart_chain"]["runs"][2]
        _edit_receipt(run, lambda receipt: receipt["metrics"].update(terminal_waits=2))
        with self.assertRaisesRegex(ValueError, "exactly one terminal wait"):
            self.compare()

    def test_rejects_qbit_roundtrip_label_without_boundary_codec_evidence(self) -> None:
        run = self.index()["arms"]["qbit_control"]["runs"][0]
        _edit_receipt(
            run,
            lambda receipt: receipt.update(qbit_roundtrip_boundaries=[]),
        )
        with self.assertRaisesRegex(ValueError, "roundtrip|codec"):
            self.compare()

    def test_rejects_qbit_control_without_opt_in_roundtrip_mode(self) -> None:
        run = self.index()["arms"]["qbit_control"]["runs"][0]
        _edit_receipt(run, lambda receipt: receipt.update(cache_roundtrip="exact"))
        with self.assertRaisesRegex(ValueError, "cache_roundtrip"):
            self.compare()

    def test_rejects_source_record_extent_confused_with_a_prefix_extent(self) -> None:
        index = self.index()
        index["source_bank"]["nbytes"] = 32 * 60 * 9 * 16 * F32_BYTES
        _write_index(self.index_path, index)
        with self.assertRaisesRegex(ValueError, "source bank record metadata|extent"):
            self.compare()

    def test_rejects_checkpoint_file_extent_confused_with_other_byte_accounting(self) -> None:
        run = self.index()["arms"]["compact_restart_chain"]["runs"][0]
        _edit_receipt(
            run,
            lambda receipt: receipt.update(checkpoint_file_bytes=receipt["resident_f32_kv_bytes"]),
        )
        with self.assertRaisesRegex(ValueError, "checkpoint.*bytes|extent"):
            self.compare()

    def test_rejects_nonfinite_capture_even_with_updated_digest(self) -> None:
        run = self.index()["arms"]["exact_restart_chain"]["runs"][1]
        receipt = json.loads(Path(run["receipt_path"]).read_text(encoding="utf-8"))
        capture = Path(receipt["diagnostics"]["segment1-step1"])
        with capture.open("r+b") as stream:
            stream.write(struct.pack("<f", float("nan")))
        _refresh_artifact_sha(run, "segment1-step1")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.compare()

    def test_cli_writes_a_fresh_machine_readable_report(self) -> None:
        output = self.root / "report.json"
        self.assertEqual(
            comparison.main(["--index", str(self.index_path), "--output", str(output)]),
            0,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], comparison.REPORT_SCHEMA)
        self.assertTrue(report["passed"])
        self.assertEqual(
            comparison.main(["--index", str(self.index_path), "--output", str(output)]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
