"""CPU-only hostile fixtures for the native Hy-Nomas prompt-edit matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_prompt_edit_resume_reference as comparison


SEGMENTS = 3
STEPS = 4
RESIDENT_SEGMENTS = 6
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
BASE_PROMPT_SHA = hashlib.sha256(b"original prompt").hexdigest()
EDIT_PROMPT_SHA = hashlib.sha256(b"replacement prompt").hexdigest()
PROMPT_INPUTS_SHA = hashlib.sha256(b"conditioned input manifest").hexdigest()
PROMPT_PAYLOAD_SHA = hashlib.sha256(b"conditioned input payload").hexdigest()
PROMPT_INPUTS_PATH = "/fixtures/prompt-inputs-v2.json"
PROMPT_PAYLOAD_PATH = "/fixtures/prompt-inputs-v2.tensors.bin"
IDENTITY = {
    "schema": comparison.EDIT_IDENTITY_SCHEMA,
    "run_segments": 3,
    "target_segment": 1,
    "conditioning_manifest_sha256": PROMPT_INPUTS_SHA,
    "conditioning_payload_sha256": PROMPT_PAYLOAD_SHA,
    "prompt_sha256": EDIT_PROMPT_SHA,
    "valid_text_tokens": 31,
    "following_prompt": "original_as_new_text_turn",
}
ORIGINAL_TOKENS = 23
EDITED_TOKENS = 31


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return _sha(payload)


def _float_payload(count: int, value: float) -> bytes:
    return struct.pack("<f", value) * count


def _source_bank() -> bytes:
    values = [
        _float_payload(SEGMENT_VALUES, -0.5 + segment * 0.0625)
        for segment in range(RESIDENT_SEGMENTS)
    ]
    return b"".join(values[segment] for _channel in range(CHANNELS)
                    for segment in range(RESIDENT_SEGMENTS))


def _value(segment: int, step: int, edited: bool) -> float:
    value = 0.125 + 0.15 * segment + 0.01 * step
    if edited and segment >= 1:
        value += 0.05 * (segment - 1 + 1)
    return value


def _timeline(bank: bytes, stop: int, edited: bool) -> bytes:
    result = bytearray(bank)
    frame_block_bytes = SEGMENT_VALUES * F32_BYTES
    for channel in range(CHANNELS):
        for segment in range(stop):
            start = (channel * TOTAL_FRAMES + segment * SEGMENT_FRAMES) * FRAME_VALUES * F32_BYTES
            result[start:start + frame_block_bytes] = _float_payload(
                SEGMENT_VALUES, _value(segment, STEPS - 1, edited)
            )
    return bytes(result)


def _base_contract_document(
    *,
    executable_sha: str,
    inputs_sha: str,
    payload_sha: str,
    conditioning_sha: str,
    model_sha: str,
    next_segment: int,
) -> dict:
    return {
        "schema": "cogni_ml.hy_nomas_sequence_boundary/v1",
        "boundary": "after_noisy_writer",
        "upstream": "fixture-upstream-revision",
        "implementation_sha256": comparison._current_implementation_sha256(),
        "executable_sha256": executable_sha,
        "crystal": "1.16.3",
        "model_index_sha256": model_sha,
        "input_manifest_sha256": inputs_sha,
        "input_payload_sha256": payload_sha,
        "conditioning_manifest_sha256": conditioning_sha,
        "segments": 3,
        "next_segment": next_segment,
        "steps": 4,
        "shift": 5,
        "text_cache_policy": "once_per_scene",
        "history_dtype": "F32",
        "valid_text_tokens": ORIGINAL_TOKENS,
        "storage_text_tokens": 64,
        "video_tokens": 4320,
        "layers": 54,
        "heads": 16,
        "head_dim": 128,
        "timeline_shape": TIMELINE_SHAPE,
        "cache_sigma": 0.206897,
    }


def _base_contracts(**identity) -> tuple[dict[str, str], dict[str, dict]]:
    documents = {
        str(boundary): _base_contract_document(next_segment=boundary, **identity)
        for boundary in (1, 2)
    }
    strings = {
        boundary: json.dumps(document, separators=(",", ":"), ensure_ascii=False)
        for boundary, document in documents.items()
    }
    hashes = {boundary: _sha(value.encode()) for boundary, value in strings.items()}
    return hashes, documents


def _hnrs_file(path: Path, contract_document: dict, payload_bytes: int) -> str:
    contract = json.dumps(contract_document, separators=(",", ":"), ensure_ascii=False)
    payload = bytes(payload_bytes)
    record = {
        "name": "fixture.timeline",
        "nbytes": payload_bytes,
        "offset": 0,
        "sha256": _sha(payload),
    }
    header = json.dumps(
        {
            "schema": comparison.HNRS_SCHEMA,
            "schema_version": 1,
            "contract": contract,
            "records": [record],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if len(header) > comparison.HNRS_MAX_HEADER_BYTES:
        raise AssertionError("fixture HNRS header exceeds its fixed header area")
    encoded = (
        comparison.HNRS_MAGIC
        + struct.pack("<I", len(header))
        + header
        + bytes(comparison.HNRS_MAX_HEADER_BYTES - len(header))
        + payload
    )
    return _write(path, encoded)


def _fixture(root: Path) -> Path:
    bank = _source_bank()
    payload_path = root / "inputs.tensors.bin"
    payload = b"HY-NOMAS-INPUT\0\0" + bank + b"END"
    payload_sha = _write(payload_path, payload)
    record_offset = len(b"HY-NOMAS-INPUT\0\0")
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
        "qwen_source": {
            "prompt_contract": {"prompt": "original prompt"},
            "prompt_sha256": BASE_PROMPT_SHA,
            "valid_tokens": ORIGINAL_TOKENS,
        },
        "prompt": "original prompt",
    }
    manifest_path = root / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    inputs_sha = _sha(manifest_path.read_bytes())
    conditioning_sha = _sha(b"original conditioning")
    model_sha = _sha(b"BF16 model index")
    executable_sha = _sha(b"same current native executable")
    identity = {
        "executable_sha": executable_sha,
        "inputs_sha": inputs_sha,
        "payload_sha": payload_sha,
        "conditioning_sha": conditioning_sha,
        "model_sha": model_sha,
    }
    contracts, documents = _base_contracts(**identity)
    prompt_inputs = {
        "manifest_path": PROMPT_INPUTS_PATH,
        "manifest_sha256": PROMPT_INPUTS_SHA,
        "payload_path": PROMPT_PAYLOAD_PATH,
        "payload_sha256": PROMPT_PAYLOAD_SHA,
        "prompt_sha256": EDIT_PROMPT_SHA,
        "valid_text_tokens": EDITED_TOKENS,
    }
    edited_contracts = {
        "1": contracts["1"],
        "2": comparison._canonical_prompt_edit_contract_sha(contracts["2"], IDENTITY),
    }

    def write_run(name: str, pid: int, interval: tuple[int, int], edited: bool,
                  *, parent_state_sha256: str = "", stop_state: bool = False,
                  boundary_contracts: dict[str, str] | None = None,
                  state_contract_document: dict | None = None) -> Path:
        start, stop = interval
        directory = root / name
        directory.mkdir()
        diagnostics: dict[str, str] = {}
        timeline_path = directory / "timeline.f32le"
        diagnostics["timeline"] = str(timeline_path)
        diagnostics["timeline_sha256"] = _write(timeline_path, _timeline(bank, stop, edited))
        for segment in range(start, stop):
            for step in range(1, STEPS + 1):
                key = f"segment{segment}-step{step}"
                artifact_path = directory / f"{key}.f32le"
                diagnostics[key] = str(artifact_path)
                diagnostics[f"{key}_sha256"] = _write(
                    artifact_path,
                    _float_payload(CHANNELS * SEGMENT_VALUES, _value(segment, step - 1, edited)),
                )
        checkpoint_file_bytes = 0
        checkpoint_payload_bytes = 0
        if stop_state:
            state_path = directory / "boundary.hnrs"
            diagnostics["state"] = str(state_path)
            checkpoint_payload_bytes = TIMELINE_BYTES + 16
            state_sha = _hnrs_file(state_path, state_contract_document, checkpoint_payload_bytes)
            diagnostics["state_sha256"] = state_sha
            checkpoint_file_bytes = comparison.HNRS_PAYLOAD_OFFSET + checkpoint_payload_bytes
        treatment_contracts = boundary_contracts or (edited_contracts if edited else contracts)
        receipt = {
            "schema": comparison.EDIT_RECEIPT_SCHEMA if edited else comparison.BASE_RECEIPT_SCHEMA,
            "mode": "save" if start == 0 else "resume",
            "pid": pid,
            "executable_sha256": executable_sha,
            "inputs": str(manifest_path),
            "inputs_sha256": inputs_sha,
            "payload_sha256": payload_sha,
            "conditioning_sha256": conditioning_sha,
            "model_index_sha256": model_sha,
            "upstream_revision": "fixture-upstream-revision",
            "segments": 3,
            "start_segment": start,
            "stop_after_segments": stop,
            "resume_next_segment": start,
            "steps_per_segment": STEPS,
            "segment_text_tokens": (
                [ORIGINAL_TOKENS, EDITED_TOKENS, ORIGINAL_TOKENS]
                if edited else [ORIGINAL_TOKENS] * SEGMENTS
            ),
            "segment_cache_text": [True, True, True] if edited else [True, False, False],
            "valid_text_tokens": ORIGINAL_TOKENS,
            "prompt_edit": IDENTITY if edited else None,
            "prompt_inputs": PROMPT_INPUTS_PATH if edited else "",
            "prompt_inputs_sha256": PROMPT_INPUTS_SHA if edited else "",
            "noise_edit": None,
            "invalidation": comparison.PROMPT_INVALIDATION if edited else None,
            "noise_scope": comparison.UNCHANGED_NOISE_SCOPE,
            "timeline_shape": TIMELINE_SHAPE,
            "step_shape": STEP_SHAPE,
            "contract_sha256": "" if start == 0 else treatment_contracts[str(start)],
            "boundary_contract_sha256": treatment_contracts,
            "state_input_sha256": parent_state_sha256,
            "state_format": "exact",
            "cache_roundtrip": "exact",
            "qbit_roundtrip_boundaries": [],
            "compact_pack_boundary": 0,
            "checkpoint_file_bytes": checkpoint_file_bytes,
            "logical_packed_kv_bytes": 0,
            "resident_f32_kv_bytes": TIMELINE_BYTES,
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
            "decoder": False,
            "full_generation": False,
            "full_upstream_video_parity": False,
        }
        receipt_path = directory / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        return receipt_path

    parent = write_run(
        "parent-a", 31001, (0, 1), False, stop_state=True,
        boundary_contracts=contracts, state_contract_document=documents["1"],
    )
    parent_receipt = json.loads(parent.read_text())
    original = write_run(
        "original-control", 31002, (1, 3), False,
        parent_state_sha256=parent_receipt["diagnostics"]["state_sha256"],
        boundary_contracts=contracts,
    )
    edited = write_run(
        "edited-control", 31003, (1, 3), True,
        parent_state_sha256=parent_receipt["diagnostics"]["state_sha256"],
        boundary_contracts=edited_contracts,
    )
    edited_b = write_run(
        "edited-restart-b", 31004, (1, 2), True,
        parent_state_sha256=parent_receipt["diagnostics"]["state_sha256"],
        stop_state=True, boundary_contracts=edited_contracts,
        state_contract_document={
            "schema": comparison.EDIT_BOUNDARY_SCHEMA,
            "base_contract_sha256": contracts["2"],
            "prompt_edit": IDENTITY,
        },
    )
    edited_b_receipt = json.loads(edited_b.read_text())
    edited_c = write_run(
        "edited-restart-c", 31005, (2, 3), True,
        parent_state_sha256=edited_b_receipt["diagnostics"]["state_sha256"],
        boundary_contracts=edited_contracts,
    )
    index = {
        "schema": comparison.INDEX_SCHEMA,
        "segments": 3,
        "steps_per_segment": 4,
        "prompt_edit": IDENTITY,
        "source_bank": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": inputs_sha,
            "payload_path": str(payload_path),
            "payload_sha256": payload_sha,
            "record_offset": record_offset,
            "nbytes": len(bank),
            "shape": TIMELINE_SHAPE,
        },
        "prompt_inputs": {
            "manifest_path": PROMPT_INPUTS_PATH,
            "manifest_sha256": PROMPT_INPUTS_SHA,
            "payload_path": PROMPT_PAYLOAD_PATH,
            "payload_sha256": PROMPT_PAYLOAD_SHA,
        },
        "arms": {
            "parent_a": {"runs": [{"receipt_path": str(parent), "interval": [0, 1]}]},
            "original_control": {"runs": [{"receipt_path": str(original), "interval": [1, 3]}]},
            "edited_control": {"runs": [{"receipt_path": str(edited), "interval": [1, 3]}]},
            "edited_restart_chain": {"runs": [
                {"receipt_path": str(edited_b), "interval": [1, 2]},
                {"receipt_path": str(edited_c), "interval": [2, 3]},
            ]},
        },
    }
    index_path = root / "matrix-index.json"
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n", encoding="utf-8")
    return index_path


def _mutate_receipt(index_path: Path, arm: str, run_index: int, mutation) -> None:
    index = json.loads(index_path.read_text())
    receipt_path = Path(index["arms"][arm]["runs"][run_index]["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    mutation(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")


class PromptEditResumeReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="prompt-edit-reference-"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index_path = _fixture(self.root)
        self.prompt_inputs = {
            "manifest_path": PROMPT_INPUTS_PATH,
            "manifest_sha256": PROMPT_INPUTS_SHA,
            "manifest_bytes": 1,
            "payload_path": PROMPT_PAYLOAD_PATH,
            "payload_sha256": PROMPT_PAYLOAD_SHA,
            "payload_bytes": 1,
            "prompt_sha256": EDIT_PROMPT_SHA,
            "prompt": "replacement prompt",
            "valid_text_tokens": EDITED_TOKENS,
            "document": {},
            "qwen_crop_sha256": _sha(b"authenticated state26 crop"),
        }

    def _compare(self) -> dict:
        original_prompt_sha = BASE_PROMPT_SHA

        def fake_base_prompt(source: dict) -> str:
            source["original_valid_text_tokens"] = ORIGINAL_TOKENS
            source["original_prompt_sha256"] = original_prompt_sha
            return original_prompt_sha

        with mock.patch.object(comparison, "_base_prompt_sha256", side_effect=fake_base_prompt), \
             mock.patch.object(comparison, "_load_prompt_inputs", return_value=self.prompt_inputs):
            return comparison.compare_index(self.index_path)

    def test_complete_five_interval_matrix_passes(self) -> None:
        report = self._compare()
        self.assertTrue(report["passed"])
        self.assertTrue(report["contracts"]["after_a_preserved"])
        self.assertTrue(report["contracts"]["after_b_binds_prompt_edit"])
        self.assertTrue(report["state_chains"]["fresh_parent_a_used_by_controls"])
        self.assertTrue(report["state_chains"]["edited_b_linked_to_c_restart"])
        self.assertTrue(report["edit_effect"]["b_changed_from_original_control"])
        self.assertTrue(report["edit_effect"]["c_changed_from_original_control"])
        self.assertTrue(report["edited_restart"]["within_tolerance"])
        self.assertEqual(report["text_profiles"]["query_workspace_text_tokens"], EDITED_TOKENS)

    def test_stale_after_a_checkpoint_cannot_replace_fresh_parent_state(self) -> None:
        _mutate_receipt(
            self.index_path,
            "edited_control",
            0,
            lambda receipt: receipt.__setitem__("state_input_sha256", _sha(b"stale after-A")),
        )
        with self.assertRaisesRegex(comparison.PromptEditResumeReferenceError, "fresh same-executable parent A"):
            self._compare()

    def test_after_b_header_must_bind_the_full_prompt_identity(self) -> None:
        index = json.loads(self.index_path.read_text())
        receipt_path = Path(index["arms"]["edited_restart_chain"]["runs"][0]["receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        state_path = Path(receipt["diagnostics"]["state"])
        altered = dict(IDENTITY)
        altered["prompt_sha256"] = BASE_PROMPT_SHA
        parent_receipt_path = Path(index["arms"]["parent_a"]["runs"][0]["receipt_path"])
        parent_receipt = json.loads(parent_receipt_path.read_text())
        base_contracts, _ = _base_contracts(
            executable_sha=parent_receipt["executable_sha256"],
            inputs_sha=parent_receipt["inputs_sha256"],
            payload_sha=parent_receipt["payload_sha256"],
            conditioning_sha=parent_receipt["conditioning_sha256"],
            model_sha=parent_receipt["model_index_sha256"],
        )
        bad_state_sha = _hnrs_file(
            state_path,
            {
                "schema": comparison.EDIT_BOUNDARY_SCHEMA,
                "base_contract_sha256": base_contracts["2"],
                "prompt_edit": altered,
            },
            TIMELINE_BYTES + 16,
        )
        # Keep the matrix internally digest-consistent so rejection is about identity.
        _mutate_receipt(
            self.index_path,
            "edited_restart_chain",
            0,
            lambda value: value["diagnostics"].__setitem__("state_sha256", bad_state_sha),
        )
        with self.assertRaisesRegex(comparison.PromptEditResumeReferenceError, "HNRS header contract"):
            self._compare()

    def test_coordinated_fake_after_b_contract_cannot_replace_authenticated_base(self) -> None:
        index = json.loads(self.index_path.read_text())
        fake_base2_sha = _sha(b"wrong after-B base contract / wrong executable")
        fake_edited2_sha = comparison._canonical_prompt_edit_contract_sha(
            fake_base2_sha, IDENTITY
        )

        def receipt_path(arm: str, run_index: int = 0) -> Path:
            return Path(index["arms"][arm]["runs"][run_index]["receipt_path"])

        def set_boundary2(arm: str, value: str, run_index: int = 0) -> dict:
            path = receipt_path(arm, run_index)
            receipt = json.loads(path.read_text())
            receipt["boundary_contract_sha256"]["2"] = value
            path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
            return receipt

        set_boundary2("parent_a", fake_base2_sha)
        set_boundary2("original_control", fake_base2_sha)
        set_boundary2("edited_control", fake_edited2_sha)
        edited_b_receipt = set_boundary2("edited_restart_chain", fake_edited2_sha, 0)
        edited_c_receipt = set_boundary2("edited_restart_chain", fake_edited2_sha, 1)

        state_path = Path(edited_b_receipt["diagnostics"]["state"])
        fake_state_sha = _hnrs_file(
            state_path,
            {
                "schema": comparison.EDIT_BOUNDARY_SCHEMA,
                "base_contract_sha256": fake_base2_sha,
                "prompt_edit": IDENTITY,
            },
            TIMELINE_BYTES + 16,
        )
        edited_b_path = receipt_path("edited_restart_chain", 0)
        edited_b_receipt["diagnostics"]["state_sha256"] = fake_state_sha
        edited_b_path.write_text(
            json.dumps(edited_b_receipt, sort_keys=True) + "\n", encoding="utf-8"
        )

        edited_c_path = receipt_path("edited_restart_chain", 1)
        edited_c_receipt["state_input_sha256"] = fake_state_sha
        edited_c_receipt["contract_sha256"] = fake_edited2_sha
        edited_c_path.write_text(
            json.dumps(edited_c_receipt, sort_keys=True) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(
            comparison.PromptEditResumeReferenceError,
            "base after-B contract digest is not reconstructed",
        ):
            self._compare()

    def test_segment_text_counts_and_cache_turn_schedule_are_exact(self) -> None:
        _mutate_receipt(
            self.index_path,
            "edited_control",
            0,
            lambda receipt: receipt["segment_cache_text"].__setitem__(2, False),
        )
        with self.assertRaisesRegex(comparison.PromptEditResumeReferenceError, "segment_cache_text"):
            self._compare()

    def test_edit_must_change_b_and_c_and_preserve_a_bytes(self) -> None:
        index = json.loads(self.index_path.read_text())
        receipt_path = Path(index["arms"]["edited_control"]["runs"][0]["receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        for segment in (1, 2):
            for step in range(1, STEPS + 1):
                key = f"segment{segment}-step{step}"
                path = Path(receipt["diagnostics"][key])
                payload = _float_payload(CHANNELS * SEGMENT_VALUES, _value(segment, step - 1, False))
                receipt["diagnostics"][f"{key}_sha256"] = _write(path, payload)
        timeline_path = Path(receipt["diagnostics"]["timeline"])
        receipt["diagnostics"]["timeline_sha256"] = _write(timeline_path, _timeline(_source_bank(), 3, False))
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(comparison.PromptEditResumeReferenceError, "nonvacuous B output change"):
            self._compare()

    def test_split_parity_rejects_drift_beyond_two_e_minus_five(self) -> None:
        index = json.loads(self.index_path.read_text())
        receipt_path = Path(index["arms"]["edited_restart_chain"]["runs"][1]["receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        key = "segment2-step4"
        artifact = Path(receipt["diagnostics"][key])
        payload = _float_payload(CHANNELS * SEGMENT_VALUES, _value(2, 3, True) + 0.01)
        receipt["diagnostics"][f"{key}_sha256"] = _write(artifact, payload)
        timeline_path = Path(receipt["diagnostics"]["timeline"])
        timeline = bytearray(_timeline(_source_bank(), 3, True))
        block_bytes = SEGMENT_VALUES * F32_BYTES
        changed_segment = _float_payload(CHANNELS * SEGMENT_VALUES, _value(2, 3, True) + 0.01)
        channel_bytes = SEGMENT_FRAMES * FRAME_VALUES * F32_BYTES
        for channel in range(CHANNELS):
            src = channel * block_bytes
            dst = (channel * TOTAL_FRAMES + 2 * SEGMENT_FRAMES) * FRAME_VALUES * F32_BYTES
            timeline[dst:dst + channel_bytes] = changed_segment[src:src + channel_bytes]
        receipt["diagnostics"]["timeline_sha256"] = _write(timeline_path, bytes(timeline))
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(comparison.PromptEditResumeReferenceError, "exceed the existing 2e-5 tolerance"):
            self._compare()

    def test_preterminal_wait_is_rejected(self) -> None:
        _mutate_receipt(
            self.index_path,
            "edited_restart_chain",
            1,
            lambda receipt: receipt["metrics"].__setitem__("preterminal_waits", 1),
        )
        with self.assertRaisesRegex(comparison.PromptEditResumeReferenceError, "preterminal_waits"):
            self._compare()

    def test_conditioned_qwen_auth_uses_dynamic_prompt_metadata_helper(self) -> None:
        reference_path = self.root / "conditioned-qwen.json"
        sidecar_path = self.root / "conditioned-qwen.bf16le"
        sidecar_bytes = b"fixture-sidecar"
        sidecar_sha = _write(sidecar_path, sidecar_bytes)
        crop_bytes = b"authenticated state26 crop bytes"
        contract = {"schema": "fixture.dynamic-prompt-contract", "prompt": "replacement prompt"}
        source = {
            "reference_path": str(reference_path),
            "reference_sha256": "0" * 64,
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sidecar_sha,
            "sidecar_bytes": len(sidecar_bytes),
            "prompt_contract": contract,
            "valid_tokens": EDITED_TOKENS,
            "state_index": 26,
            "crop_tokens": 1280,
            "hidden_size": 3584,
            "crop": [1, 1281],
            "raw_sequence_length": 1281,
            "cropped_attention_mask": [1] * EDITED_TOKENS + [0] * (1280 - EDITED_TOKENS),
        }
        reference_document = {
            "hy_nomas_prompt": contract,
            "sidecars": {
                "bf16": {
                    "path": sidecar_path.name,
                    "sha256": sidecar_sha,
                    "nbytes": len(sidecar_bytes),
                    "encoding": "bfloat16-le",
                }
            },
            "inputs": {
                "token_ids": [1, 2],
                "attention_mask": [1, 1],
                "position_ids_f32": [],
                "position_ids_for_official_model": [],
                "multimodal_position_shape": [],
                "position_offset": 0,
            },
        }
        reference_raw = json.dumps(reference_document).encode()
        reference_path.write_bytes(reference_raw)
        source["reference_sha256"] = _sha(reference_raw)
        qwen_record = {
            "dtype": "BF16",
            "shape": [1, 1280, 3584],
            "nbytes": 1280 * 3584 * 2,
            "sha256": _sha(crop_bytes),
        }
        validate_prompt_metadata = mock.Mock(return_value=1281)
        qwen_helper = SimpleNamespace(_validate_manifest=mock.Mock())
        with mock.patch.object(comparison.generation_inputs, "_load_qwen_prefix_reference", return_value=qwen_helper), \
             mock.patch.object(
                 comparison.generation_inputs,
                 "_prompt_reference_helper",
                 return_value=SimpleNamespace(validate_prompt_metadata=validate_prompt_metadata),
             ), \
             mock.patch.object(
                 comparison.compact,
                 "_read_hashed_file",
                 side_effect=[(reference_raw, len(reference_raw)), (crop_bytes, len(crop_bytes))],
             ):
            self.assertEqual(
                comparison._authenticate_qwen_reference_crop(source, qwen_record),
                qwen_record["sha256"],
            )
        validate_prompt_metadata.assert_called_once()
        self.assertEqual(validate_prompt_metadata.call_args.args[0], contract)

    def test_conditioned_generation_payload_hashes_each_tensor(self) -> None:
        payload_path = self.root / "small-conditioned-payload.bin"
        payload = b"tensor-data"
        payload_path.write_bytes(payload)
        document = {
            "payload_bytes": len(payload),
            "tensors": {
                "input.test": {
                    "nbytes": len(payload),
                    "sha256": _sha(payload),
                }
            },
        }
        with mock.patch.object(
            comparison.generation_inputs, "_resolution_for_schema", return_value="fixture"
        ), mock.patch.object(
            comparison.generation_inputs,
            "tensor_specs_for_resolution",
            return_value=(("input.test", "F32", [1]),),
        ):
            comparison._validate_generation_payload(payload_path, document, _sha(payload), len(payload))
            with self.assertRaisesRegex(comparison.PromptEditResumeReferenceError, "aggregate payload SHA"):
                comparison._validate_generation_payload(payload_path, document, "0" * 64, len(payload))

    def test_base_continuation_payload_checks_every_tensor_digest(self) -> None:
        payload_path = self.root / "small-base-payload.bin"
        payload = b"base-tensor"
        payload_path.write_bytes(payload)
        document = {
            "payload_bytes": len(payload),
            "payload_sha256": _sha(payload),
            "tensors": {"fixture.tensor": {"nbytes": len(payload), "sha256": _sha(payload)}},
        }
        with mock.patch.object(
            comparison.continuation_inputs,
            "TENSOR_SPECS",
            (("fixture.tensor", "F32", (1,)),),
        ):
            comparison._validate_continuation_payload(payload_path, document)
            document["tensors"]["fixture.tensor"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(comparison.PromptEditResumeReferenceError, "tensor digest mismatch"):
                comparison._validate_continuation_payload(payload_path, document)


if __name__ == "__main__":
    unittest.main()
