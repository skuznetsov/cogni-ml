"""CPU-only falsifiers for the native Hy-Nomas segment-edit resume matrix."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_segment_edit_resume_reference as comparison


SEGMENTS = 3
STEPS = 4
RESIDENT_SEGMENTS = 6
TARGET_SEGMENT = 1
SOURCE_SEGMENT = 5
CHANNELS = 32
SEGMENT_FRAMES = 30
TOTAL_FRAMES = RESIDENT_SEGMENTS * SEGMENT_FRAMES
HEIGHT = 9
WIDTH = 16
FRAME_VALUES = HEIGHT * WIDTH
SEGMENT_VALUES = SEGMENT_FRAMES * FRAME_VALUES
TIMELINE_SHAPE = [1, CHANNELS, TOTAL_FRAMES, HEIGHT, WIDTH]
STEP_SHAPE = [1, CHANNELS, SEGMENT_FRAMES, HEIGHT, WIDTH]
F32_BYTES = 4
TIMELINE_BYTES = CHANNELS * TOTAL_FRAMES * FRAME_VALUES * F32_BYTES
STEP_BYTES = CHANNELS * SEGMENT_VALUES * F32_BYTES
NOISE_EDIT = {
    "schema": "cogni_ml.hy_nomas_segment_noise_edit/v1",
    "run_segments": SEGMENTS,
    "target_segment": TARGET_SEGMENT,
    "source_segment": SOURCE_SEGMENT,
    "resident_segments": RESIDENT_SEGMENTS,
}


def _sha(payload: bytes) -> str:
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


def _fill(count: int, value: float) -> bytes:
    return struct.pack("<f", value) * count


def _source_bank() -> bytes:
    # F32 is channel-major. Distinct resident segments make donor/suffix
    # corruption visible while retaining a compact synthetic fixture.
    one_segment = [
        _fill(SEGMENT_VALUES, -0.5 + segment * 0.0625)
        for segment in range(RESIDENT_SEGMENTS)
    ]
    return b"".join(one_segment[segment] for _channel in range(CHANNELS)
                    for segment in range(RESIDENT_SEGMENTS))


def _step_value(segment: int, step: int, edited: bool) -> float:
    value = 0.125 + 0.15 * segment + 0.01 * step
    if edited and segment >= TARGET_SEGMENT:
        value += 0.025 * (segment - TARGET_SEGMENT + 1)
    return value


def _timeline(bank: bytes, stop: int, edited: bool) -> bytes:
    result = bytearray(bank)
    block_bytes = SEGMENT_VALUES * F32_BYTES
    for channel in range(CHANNELS):
        for segment in range(stop):
            start = (channel * TOTAL_FRAMES + segment * SEGMENT_FRAMES) * FRAME_VALUES * F32_BYTES
            result[start:start + block_bytes] = _fill(
                SEGMENT_VALUES, _step_value(segment, STEPS - 1, edited)
            )
    return bytes(result)


def _base_boundary_contracts() -> dict[str, str]:
    return {
        str(boundary): _sha(f"base-contract-{boundary}".encode("ascii"))
        for boundary in range(1, SEGMENTS)
    }


def _edited_boundary_contracts(base: dict[str, str]) -> dict[str, str]:
    fork_contract = {
        "schema": "cogni_ml.hy_nomas_edited_sequence_boundary/v1",
        "base_contract_sha256": base["2"],
        "noise_edit": NOISE_EDIT,
    }
    encoded = json.dumps(
        fork_contract, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {"1": base["1"], "2": _sha(encoded)}


def _write_run(
    root: Path,
    *,
    name: str,
    pid: int,
    manifest_path: Path,
    inputs_sha: str,
    payload_sha: str,
    bank: bytes,
    start: int,
    stop: int,
    edited: bool,
    parent_state_sha: str = "",
    base_contracts: dict[str, str],
    state_payload: bytes | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    diagnostics: dict[str, str] = {}
    timeline = _timeline(bank, stop, edited)
    timeline_path = run_dir / "timeline.f32le"
    diagnostics["timeline"] = str(timeline_path)
    diagnostics["timeline_sha256"] = _write(timeline_path, timeline)
    for segment in range(start, stop):
        for step in range(STEPS):
            key = f"segment{segment}-step{step + 1}"
            payload = _fill(
                CHANNELS * SEGMENT_VALUES,
                _step_value(segment, step, edited),
            )
            artifact = run_dir / f"{key}.f32le"
            diagnostics[key] = str(artifact)
            diagnostics[f"{key}_sha256"] = _write(artifact, payload)

    checkpoint_file_bytes = 0
    checkpoint_payload_bytes = 0
    if state_payload is not None:
        state_path = run_dir / "boundary.hnrs"
        diagnostics["state"] = str(state_path)
        diagnostics["state_sha256"] = _write(state_path, state_payload)
        checkpoint_file_bytes = len(state_payload)
        checkpoint_payload_bytes = len(bank) + 1024

    boundary_contracts = _edited_boundary_contracts(base_contracts) if edited else base_contracts
    mode = "save" if start == 0 else "resume"
    receipt = {
        "schema": (
            "cogni_ml.hy_nomas_segment_edit_probe/v1"
            if edited else "cogni_ml.hy_nomas_compact_sequence_probe/v1"
        ),
        "mode": mode,
        "pid": pid,
        "executable_sha256": _sha(b"same release executable"),
        "inputs": str(manifest_path),
        "inputs_sha256": inputs_sha,
        "payload_sha256": payload_sha,
        "conditioning_sha256": _sha(b"same shared scene conditioning"),
        "model_index_sha256": _sha(b"same BF16 model index"),
        "contract_sha256": "" if start == 0 else boundary_contracts[str(start)],
        "boundary_contract_sha256": boundary_contracts,
        "state_input_sha256": parent_state_sha,
        "state_format": "exact",
        "cache_roundtrip": "exact",
        "qbit_roundtrip_boundaries": [],
        "compact_pack_boundary": 0,
        "upstream_revision": "fixture-upstream-revision",
        "segments": SEGMENTS,
        "start_segment": start,
        "stop_after_segments": stop,
        "resume_next_segment": start,
        "steps_per_segment": STEPS,
        "valid_text_tokens": 12,
        "timeline_shape": TIMELINE_SHAPE,
        "step_shape": STEP_SHAPE,
        "checkpoint_file_bytes": checkpoint_file_bytes,
        "logical_packed_kv_bytes": 0,
        "resident_f32_kv_bytes": 1024,
        "checkpoint_payload_bytes": checkpoint_payload_bytes,
        "codec": {},
        "metrics": {
            "encode_allocations": 0,
            "encode_readbacks": 0,
            "preterminal_waits": 0,
            "command_segments": 20,
            "terminal_waits": 1,
        },
        "diagnostics": diagnostics,
        "noise_edit": NOISE_EDIT if edited else None,
        "invalidation": ({
            "latent_frames": [30, 90],
            "decoder_pixel_frames": [100, 360],
            "prior_segment_overlap_pixel_frames": [100, 120],
            "units": "half-open global frame ranges; regenerate every descendant",
        } if edited else None),
        "noise_scope": (
            "one initial-noise gather from an unused donor in the authenticated state; "
            "target RoPE/cache epsilon and shared conditioning unchanged"
            if edited else
            "unchanged authenticated latent/epsilon banks; no reseeding"
        ),
        "metric_scope": "instrumented MetalBuffer methods; opaque MPS resources and terminal I/O excluded",
        "decoder": False,
        "full_generation": False,
        "full_upstream_video_parity": False,
    }
    receipt_path = run_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt_path


def _fixture(root: Path) -> Path:
    bank = _source_bank()
    payload_path = root / "inputs.tensors.bin"
    payload = b"HY-NOMAS-INPUT\0\0" + bank + b"END"
    payload_sha = _write(payload_path, payload)
    record_offset = 16
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
                "sha256": _sha(bank),
                "shape": TIMELINE_SHAPE,
            }
        },
    }
    manifest_path = root / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    inputs_sha = _sha_file(manifest_path)
    base_contracts = _base_boundary_contracts()

    parent_a_path = _write_run(
        root, name="parent-a", pid=31001, manifest_path=manifest_path,
        inputs_sha=inputs_sha, payload_sha=payload_sha, bank=bank, start=0, stop=1,
        edited=False, base_contracts=base_contracts,
        state_payload=b"native-produced-after-A-state".ljust(1024, b"\0"),
    )
    parent_a_receipt = json.loads(parent_a_path.read_text(encoding="utf-8"))
    parent_a_sha = parent_a_receipt["diagnostics"]["state_sha256"]

    original_path = _write_run(
        root, name="original-control", pid=31002, manifest_path=manifest_path,
        inputs_sha=inputs_sha, payload_sha=payload_sha, bank=bank, start=1, stop=3,
        edited=False, parent_state_sha=parent_a_sha, base_contracts=base_contracts,
    )
    edited_control_path = _write_run(
        root, name="edited-control", pid=31003, manifest_path=manifest_path,
        inputs_sha=inputs_sha, payload_sha=payload_sha, bank=bank, start=1, stop=3,
        edited=True, parent_state_sha=parent_a_sha, base_contracts=base_contracts,
    )
    edited_b_path = _write_run(
        root, name="edited-b-checkpoint", pid=31004, manifest_path=manifest_path,
        inputs_sha=inputs_sha, payload_sha=payload_sha, bank=bank, start=1, stop=2,
        edited=True, parent_state_sha=parent_a_sha, base_contracts=base_contracts,
        state_payload=b"native-produced-after-edited-B-state".ljust(1024, b"\0"),
    )
    edited_b_receipt = json.loads(edited_b_path.read_text(encoding="utf-8"))
    edited_b_sha = edited_b_receipt["diagnostics"]["state_sha256"]
    edited_c_path = _write_run(
        root, name="edited-c-restart", pid=31005, manifest_path=manifest_path,
        inputs_sha=inputs_sha, payload_sha=payload_sha, bank=bank, start=2, stop=3,
        edited=True, parent_state_sha=edited_b_sha, base_contracts=base_contracts,
    )

    index = {
        "schema": "cogni_ml.hy_nomas_segment_edit_resume_matrix/v1",
        "segments": SEGMENTS,
        "steps_per_segment": STEPS,
        "noise_edit": NOISE_EDIT,
        "arms": {
            "parent_a": {"runs": [{"receipt_path": str(parent_a_path), "interval": [0, 1]}]},
            "original_control": {"runs": [{"receipt_path": str(original_path), "interval": [1, 3]}]},
            "edited_control": {"runs": [{"receipt_path": str(edited_control_path), "interval": [1, 3]}]},
            "edited_restart_chain": {"runs": [
                {"receipt_path": str(edited_b_path), "interval": [1, 2]},
                {"receipt_path": str(edited_c_path), "interval": [2, 3]},
            ]},
        },
        "source_bank": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": inputs_sha,
            "payload_path": str(payload_path),
            "payload_sha256": payload_sha,
            "record_offset": record_offset,
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


def _entry(index_path: Path, arm: str, run_index: int = 0) -> dict:
    return _read_index(index_path)["arms"][arm]["runs"][run_index]


def _edit_receipt(entry: dict, edit) -> None:
    path = Path(entry["receipt_path"])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    edit(receipt)
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_artifact_sha(entry: dict, name: str) -> None:
    path = Path(entry["receipt_path"])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    artifact = Path(receipt["diagnostics"][name])
    receipt["diagnostics"][f"{name}_sha256"] = _sha_file(artifact)
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")


class SegmentEditResumeReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hy-nomas-segment-edit-resume-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index_path = _fixture(self.root)

    def compare(self) -> dict:
        return comparison.compare_index(self.index_path)

    def test_accepts_parent_control_edited_control_and_exact_restart_matrix(self) -> None:
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertTrue(report["a_prefix_and_future"]["a_prefix_bitwise_unchanged"])
        self.assertTrue(report["a_prefix_and_future"]["future_suffix_matches_source_bank"])
        self.assertTrue(report["edited_restart"]["within_tolerance"])
        self.assertTrue(report["edited_restart"]["all_steps_within_tolerance"])
        self.assertTrue(report["edited_restart"]["all_steps_bitwise_equal"])
        self.assertTrue(report["edit_effect"]["b_changed_from_original_control"])
        self.assertTrue(report["edit_effect"]["c_changed_from_original_control"])
        self.assertEqual(report["tolerance"], {"atol": 2e-5, "rtol": 2e-5})
        self.assertIn("does not establish upstream/full-pixel quality", report["scope"])
        self.assertIn("checkpoint", report["checkpoint_validation"])

    def test_cli_writes_the_machine_readable_matrix_report(self) -> None:
        output = self.root / "report.json"
        with redirect_stdout(io.StringIO()):
            status = comparison.main([
                "--index", str(self.index_path), "--output", str(output),
            ])
        self.assertEqual(status, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], "cogni_ml.hy_nomas_segment_edit_resume_reference/v1")
        self.assertTrue(report["passed"])

    def test_rejects_a_wrong_parent_after_b_state_digest(self) -> None:
        entry = _entry(self.index_path, "edited_restart_chain", 1)
        _edit_receipt(entry, lambda receipt: receipt.update(state_input_sha256="f" * 64))
        with self.assertRaisesRegex(ValueError, "parent.*state|state.*parent"):
            self.compare()

    def test_rejects_an_unedited_after_b_contract_on_edited_restart(self) -> None:
        index = _read_index(self.index_path)
        base_contract = json.loads(
            Path(index["arms"]["original_control"]["runs"][0]["receipt_path"])
            .read_text(encoding="utf-8")
        )["boundary_contract_sha256"]["2"]
        entry = index["arms"]["edited_restart_chain"]["runs"][1]

        def old_contract(receipt: dict) -> None:
            receipt["boundary_contract_sha256"]["2"] = base_contract
            receipt["contract_sha256"] = base_contract

        _edit_receipt(entry, old_contract)
        with self.assertRaisesRegex(ValueError, "edited.*contract|fork.*contract|boundary contract"):
            self.compare()

    def test_rejects_a_changed_a_prefix_even_with_a_refreshed_artifact_hash(self) -> None:
        entry = _entry(self.index_path, "edited_control")
        receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8"))
        timeline = Path(receipt["diagnostics"]["timeline"])
        with timeline.open("r+b") as stream:
            stream.write(struct.pack("<f", 99.0))
        _refresh_artifact_sha(entry, "timeline")
        with self.assertRaisesRegex(ValueError, "A.*prefix|prefix.*A"):
            self.compare()

    def test_rejects_mutated_future_donor_suffix_even_with_a_refreshed_hash(self) -> None:
        entry = _entry(self.index_path, "edited_restart_chain", 0)
        receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8"))
        timeline = Path(receipt["diagnostics"]["timeline"])
        donor_offset = SOURCE_SEGMENT * SEGMENT_FRAMES * FRAME_VALUES * F32_BYTES
        with timeline.open("r+b") as stream:
            stream.seek(donor_offset)
            stream.write(struct.pack("<f", 99.0))
        _refresh_artifact_sha(entry, "timeline")
        with self.assertRaisesRegex(ValueError, "future|source.bank|suffix|donor"):
            self.compare()

    def test_rejects_restart_output_mismatch_beyond_existing_tolerance(self) -> None:
        entry = _entry(self.index_path, "edited_restart_chain", 1)
        receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8"))
        step = Path(receipt["diagnostics"]["segment2-step3"])
        with step.open("r+b") as stream:
            stream.write(struct.pack("<f", 90.0))
        _refresh_artifact_sha(entry, "segment2-step3")
        with self.assertRaisesRegex(ValueError, "tolerance|restart outputs"):
            self.compare()

    def test_rejects_a_vacuous_descendant_c_even_when_b_changes(self) -> None:
        original_entry = _entry(self.index_path, "original_control")
        edited_entry = _entry(self.index_path, "edited_control")
        original_receipt = json.loads(Path(original_entry["receipt_path"]).read_text(encoding="utf-8"))
        edited_receipt = json.loads(Path(edited_entry["receipt_path"]).read_text(encoding="utf-8"))
        for step in range(1, STEPS + 1):
            key = f"segment2-step{step}"
            Path(original_receipt["diagnostics"][key]).write_bytes(
                Path(edited_receipt["diagnostics"][key]).read_bytes()
            )
            _refresh_artifact_sha(original_entry, key)

        original_timeline = Path(original_receipt["diagnostics"]["timeline"])
        edited_timeline = Path(edited_receipt["diagnostics"]["timeline"])
        original = bytearray(original_timeline.read_bytes())
        edited = edited_timeline.read_bytes()
        segment_bytes = SEGMENT_VALUES * F32_BYTES
        for channel in range(CHANNELS):
            offset = (channel * TOTAL_FRAMES + 2 * SEGMENT_FRAMES) * FRAME_VALUES * F32_BYTES
            original[offset:offset + segment_bytes] = edited[offset:offset + segment_bytes]
        original_timeline.write_bytes(original)
        _refresh_artifact_sha(original_entry, "timeline")
        with self.assertRaisesRegex(ValueError, "descendant C output change"):
            self.compare()

    def test_rejects_encode_readback_or_extra_terminal_wait_metrics(self) -> None:
        entry = _entry(self.index_path, "edited_control")

        def violate(receipt: dict) -> None:
            receipt["metrics"]["encode_readbacks"] = 1

        _edit_receipt(entry, violate)
        with self.assertRaisesRegex(ValueError, "encode_readbacks"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
