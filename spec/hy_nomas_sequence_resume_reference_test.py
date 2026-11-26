"""CPU-only tests for bounded Hy-Nomas cross-process parity receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_sequence_resume_reference as resume_reference


INPUTS_SHA = hashlib.sha256(b"test continuation input manifest").hexdigest()
CONTRACT_SHA = hashlib.sha256(b"test continuation contract").hexdigest()
STATE_SHA_PLACEHOLDER = hashlib.sha256(b"state pending").hexdigest()
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(directory: Path, name: str, filename: str, payload: bytes) -> tuple[str, str]:
    path = directory / filename
    path.write_bytes(payload)
    return str(path), _digest(path)


def _fixture(root: Path) -> dict[str, Path]:
    directories = {mode: root / mode for mode in ("control", "save", "resume")}
    for directory in directories.values():
        directory.mkdir()

    timeline = bytes(resume_reference.TIMELINE_BYTES)
    step = struct.pack("<f", 0.125) * (resume_reference.STEP_BYTES // 4)
    state = bytes(resume_reference.STATE_PAYLOAD_OFFSET + 16)
    state_path, state_sha = _write_artifact(
        directories["save"], "state", resume_reference.STATE_FILENAME, state
    )

    receipts = {}
    for mode, directory in directories.items():
        timeline_path, timeline_sha = _write_artifact(directory, "timeline", "timeline.f32le", timeline)
        diagnostics = {"timeline": timeline_path, "timeline_sha256": timeline_sha}
        if mode == "save":
            diagnostics.update({"state": state_path, "state_sha256": state_sha})
        else:
            for step_name in resume_reference.STEP_NAMES:
                path, digest = _write_artifact(directory, step_name, f"{step_name}.f32le", step)
                diagnostics[step_name] = path
                diagnostics[f"{step_name}_sha256"] = digest
        receipts[mode] = {
            "schema": resume_reference.RECEIPT_SCHEMA,
            "mode": mode,
            "pid": {"control": 1101, "save": 1102, "resume": 1103}[mode],
            "inputs": "/scratch/continuation-inputs.json",
            "inputs_sha256": INPUTS_SHA,
            "contract_sha256": CONTRACT_SHA,
            "state_input_sha256": state_sha if mode == "resume" else "",
            "upstream_revision": UPSTREAM_REVISION,
            "segments": 2,
            "resume_next_segment": 1,
            "steps_per_segment": 4,
            "checkpoint_payload_bytes": 16,
            "capture_bytes": 0 if mode == "save" else 4 * resume_reference.STEP_BYTES,
            "dit_peak_named_bytes": 1024,
            "peak_named_bytes": 1024 if mode == "save" else 1024 + 4 * resume_reference.STEP_BYTES,
            "timeline_shape": [1, 32, 180, 9, 16],
            "step_shape": [1, 32, 30, 9, 16],
            "metrics": {
                "encode_allocations": 0,
                "encode_readbacks": 0,
                "preterminal_waits": 0,
                "command_segments": 1,
                "terminal_waits": 1,
            },
            "diagnostics": diagnostics,
            "decoder": False,
            "full_generation": False,
        }
        (directory / "receipt.json").write_text(
            json.dumps(receipts[mode], sort_keys=True) + "\n", encoding="utf-8"
        )
    return directories


def _edit_receipt(directory: Path, edit) -> None:
    path = directory / "receipt.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_artifact_sha(directory: Path, artifact_name: str) -> None:
    _edit_receipt(
        directory,
        lambda document: document["diagnostics"].__setitem__(
            f"{artifact_name}_sha256",
            _digest(Path(document["diagnostics"][artifact_name])),
        ),
    )


class SequenceResumeReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hy-nomas-resume-reference-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directories = _fixture(self.root)

    def compare(self) -> dict:
        return resume_reference.compare_runs(
            self.directories["control"], self.directories["save"], self.directories["resume"]
        )

    def test_accepts_authenticated_distinct_process_parity(self) -> None:
        report = self.compare()
        self.assertTrue(report["passed"])
        self.assertTrue(report["finite"])
        self.assertTrue(report["a_prefix"]["control_save_bitwise_equal"])
        self.assertTrue(report["a_prefix"]["control_resume_bitwise_equal"])
        self.assertTrue(report["timeline_control_resume"]["bitwise_equal"])
        self.assertEqual(set(report["b_steps_control_resume"]), set(resume_reference.STEP_NAMES))
        self.assertTrue(all(item["within_tolerance"] for item in report["b_steps_control_resume"].values()))

    def test_rejects_nonfinite_step_even_when_its_digest_matches(self) -> None:
        path = self.directories["resume"] / "b-step1.f32le"
        payload = bytearray(path.read_bytes())
        payload[0:4] = struct.pack("<f", float("nan"))
        path.write_bytes(payload)
        _refresh_artifact_sha(self.directories["resume"], "b-step1")
        with self.assertRaisesRegex(resume_reference.ResumeReferenceError, "non-finite"):
            self.compare()

    def test_rejects_missing_step_artifact(self) -> None:
        (self.directories["control"] / "b-step4.f32le").unlink()
        with self.assertRaisesRegex(resume_reference.ResumeReferenceError, "cannot read control b-step4"):
            self.compare()

    def test_rejects_mutated_step_even_when_its_digest_matches(self) -> None:
        path = self.directories["resume"] / "b-step3.f32le"
        payload = bytearray(path.read_bytes())
        payload[0:4] = struct.pack("<f", 0.25)
        path.write_bytes(payload)
        _refresh_artifact_sha(self.directories["resume"], "b-step3")
        with self.assertRaisesRegex(resume_reference.ResumeReferenceError, "b-step3 values exceed tolerance"):
            self.compare()

    def test_rejects_mutated_step_with_stale_digest(self) -> None:
        path = self.directories["resume"] / "b-step2.f32le"
        payload = bytearray(path.read_bytes())
        payload[0:4] = struct.pack("<f", 0.25)
        path.write_bytes(payload)
        with self.assertRaisesRegex(resume_reference.ResumeReferenceError, "SHA-256 does not match"):
            self.compare()

    def test_rejects_reused_process_identity(self) -> None:
        _edit_receipt(
            self.directories["resume"],
            lambda document: document.__setitem__("pid", 1101),
        )
        with self.assertRaisesRegex(resume_reference.ResumeReferenceError, "distinct process IDs"):
            self.compare()

    def test_rejects_resume_state_not_linked_to_saved_bundle(self) -> None:
        _edit_receipt(
            self.directories["resume"],
            lambda document: document.__setitem__("state_input_sha256", STATE_SHA_PLACEHOLDER),
        )
        with self.assertRaisesRegex(resume_reference.ResumeReferenceError, "does not match the saved boundary"):
            self.compare()

    def test_rejects_cross_run_contract_mismatch(self) -> None:
        _edit_receipt(
            self.directories["resume"],
            lambda document: document.__setitem__(
                "contract_sha256", hashlib.sha256(b"different contract").hexdigest()
            ),
        )
        with self.assertRaisesRegex(resume_reference.ResumeReferenceError, "identical contract_sha256"):
            self.compare()

    def test_rejects_cross_run_input_mismatch(self) -> None:
        _edit_receipt(
            self.directories["resume"],
            lambda document: document.__setitem__(
                "inputs_sha256", hashlib.sha256(b"different input manifest").hexdigest()
            ),
        )
        with self.assertRaisesRegex(resume_reference.ResumeReferenceError, "identical inputs_sha256"):
            self.compare()

    def test_reads_artifacts_in_bounded_chunks_without_retaining_state_bundle(self) -> None:
        read_sizes = []
        state_retain_flags = []
        real_fdopen = resume_reference.os.fdopen
        real_read_artifact = resume_reference._read_artifact

        class ReadSizeGuard:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def read(self, size=-1):
                read_sizes.append(size)
                if type(size) is not int or size < 0 or size > 1024 * 1024:
                    raise AssertionError(f"artifact read exceeded 1 MiB: {size}")
                return self.stream.read(size)

            def __getattr__(self, key):
                return getattr(self.stream, key)

        def bounded_fdopen(*args, **kwargs):
            return ReadSizeGuard(real_fdopen(*args, **kwargs))

        def observe_state_retention(*args, **kwargs):
            if len(args) >= 3 and args[2] == "state":
                state_retain_flags.append(kwargs.get("retain_payload", True))
            return real_read_artifact(*args, **kwargs)

        with (
            mock.patch.object(resume_reference.os, "fdopen", side_effect=bounded_fdopen),
            mock.patch.object(resume_reference, "_read_artifact", side_effect=observe_state_retention),
        ):
            report = self.compare()
        self.assertTrue(report["passed"])
        self.assertTrue(read_sizes)
        self.assertLessEqual(max(read_sizes), 1024 * 1024)
        self.assertEqual(state_retain_flags, [False])

    def test_cli_writes_fresh_machine_readable_report(self) -> None:
        output = self.root / "report.json"
        result = resume_reference.main(
            [
                "--control-dir", str(self.directories["control"]),
                "--save-dir", str(self.directories["save"]),
                "--resume-dir", str(self.directories["resume"]),
                "--output", str(output),
            ]
        )
        self.assertEqual(result, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], resume_reference.REPORT_SCHEMA)
        self.assertTrue(report["passed"])
        self.assertEqual(
            resume_reference.main(
                [
                    "--control-dir", str(self.directories["control"]),
                    "--save-dir", str(self.directories["save"]),
                    "--resume-dir", str(self.directories["resume"]),
                    "--output", str(output),
                ]
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
