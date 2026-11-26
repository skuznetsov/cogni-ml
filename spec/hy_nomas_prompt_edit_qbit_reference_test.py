"""CPU-only adversarial fixtures for compact QBit prompt-edit resume evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_prompt_edit_qbit_reference as comparison


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tempdir() -> tempfile.TemporaryDirectory[str]:
    tmpdir = os.environ.get("TMPDIR")
    selected = tmpdir if tmpdir and Path(tmpdir).is_dir() else None
    return tempfile.TemporaryDirectory(prefix="hy-nomas-prompt-qbit-test-", dir=selected)


def _stats_doc(inventory: dict[str, int]) -> dict:
    records = []
    total_logical = total_exact = total_rows = 0
    for name in sorted(set(inventory) - {"timeline"}):
        exact = inventory[name]
        rows = exact // comparison.ROW_SLOT_BYTES
        payload = rows * 64
        logical = rows * comparison.HEADER_BYTES + payload
        records.append({
            "name": name,
            "count": rows * comparison.ROW_VALUES,
            "rows": rows,
            "logical_packed_bytes": logical,
            "exact_bytes": exact,
            "payload_bytes": payload,
            "tier_counts": {"p4": rows, "p5": 0, "p8": 0, "f32": 0},
            "invalid_rows": 0,
        })
        total_logical += logical
        total_exact += exact
        total_rows += rows
    return {
        "records": records,
        "logical_packed_bytes": total_logical,
        "exact_bytes": total_exact,
        "tier_counts": {"p4": total_rows, "p5": 0, "p8": 0, "f32": 0},
        "invalid_rows": 0,
    }


def _compact_record(rows: int, *, bad_tier: bool = False) -> bytes:
    headers = bytearray()
    payload = bytearray()
    cursor = 0
    for row in range(rows):
        headers.extend(struct.pack(
            "<ffII", 0.0, 1.0, cursor, 99 if bad_tier else 4
        ))
        payload.extend(bytes(64))
        cursor += 64
    return bytes(headers + payload)


def _checkpoint_fixture(
    path: Path,
    *,
    base_contract: str = '{"schema":"fixture-base","next_segment":1}',
    wrapper_identity: str | None = None,
    manifest_identity: str | None = None,
    outer_schema: str = comparison.HNRS_SCHEMA,
    bad_tier: bool = False,
) -> tuple[bytes, dict[str, int]]:
    inventory = {
        "history.layer0.key": 1024,
        "history.layer0.value": 1024,
        "timeline": 64,
    }
    stats = _stats_doc(inventory)
    identity = comparison._qbit_identity_json()
    wrapper = {
        "schema": comparison.QBIT_CONTRACT_SCHEMA,
        "base_contract": base_contract,
        "identity_json": wrapper_identity or identity,
    }
    manifest = {
        "schema": comparison.QBIT_MANIFEST_SCHEMA,
        "identity_json": manifest_identity or identity,
        "timeline_bytes": inventory["timeline"],
        "stats_json": json.dumps(stats, separators=(",", ":")),
    }
    payloads = {
        "codec.manifest": json.dumps(manifest, separators=(",", ":")).encode(),
        "history.layer0.key": _compact_record(2, bad_tier=bad_tier),
        "history.layer0.value": _compact_record(2),
        "timeline": bytes(inventory["timeline"]),
    }
    records = []
    cursor = 0
    for name, payload in sorted(payloads.items()):
        records.append({
            "name": name,
            "nbytes": len(payload),
            "offset": cursor,
            "sha256": _sha(payload),
        })
        cursor += len(payload)
    header = {
        "schema": outer_schema,
        "schema_version": 1,
        "contract": json.dumps(wrapper, separators=(",", ":")),
        "records": records,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    if len(header_bytes) > comparison.HNRS_MAX_HEADER_BYTES:
        raise AssertionError("fixture HNRS header exceeds the production bound")
    prefix = (
        comparison.HNRS_MAGIC
        + struct.pack("<I", len(header_bytes))
        + header_bytes
    )
    prefix += bytes(comparison.HNRS_PAYLOAD_OFFSET - len(prefix))
    payload = b"".join(payloads[name] for name in sorted(payloads))
    raw = prefix + payload
    path.write_bytes(raw)
    return raw, inventory


def _parse_fixture(path: Path, raw: bytes, inventory: dict[str, int]) -> dict:
    return comparison._parse_qbit_checkpoint(
        path,
        expected_sha256=_sha(raw),
        expected_file_bytes=len(raw),
        expected_payload_bytes=len(raw) - comparison.HNRS_PAYLOAD_OFFSET,
        expected_base_contract='{"schema":"fixture-base","next_segment":1}',
        expected_inventory=inventory,
        expected_timeline_sha256=_sha(bytes(inventory["timeline"])),
        label="fixture",
    )


def _exact_report() -> dict:
    return {
        "source_bank": {
            "manifest_path": "/fixtures/original-inputs.json",
            "manifest_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "original_valid_text_tokens": 557,
            "original_prompt_sha256": "c" * 64,
        },
        "identity": {
            "conditioning_sha256": "d" * 64,
            "model_index_sha256": "e" * 64,
            "upstream_revision": "fixture-upstream",
            "executable_sha256": "f" * 64,
        },
    }


PROMPT_IDENTITY = {
    "schema": "cogni_ml.hy_nomas_segment_prompt_edit/v1",
    "run_segments": 3,
    "target_segment": 1,
    "conditioning_manifest_sha256": "1" * 64,
    "conditioning_payload_sha256": "2" * 64,
    "prompt_sha256": "3" * 64,
    "valid_text_tokens": 585,
    "following_prompt": "original_as_new_text_turn",
}
PROMPT_INPUTS = {
    "manifest_path": "/fixtures/prompt-inputs-v2.json",
    "manifest_sha256": PROMPT_IDENTITY["conditioning_manifest_sha256"],
}


def _receipt(arm: str, interval: tuple[int, int]) -> dict:
    start, stop = interval
    edited = arm != "parent_qbit_a"
    receipt = {
        "schema": comparison.QBIT_RECEIPT_SCHEMA if edited else comparison.BASE_RECEIPT_SCHEMA,
        "mode": "save" if start == 0 else "resume",
        "segments": 3,
        "start_segment": start,
        "stop_after_segments": stop,
        "resume_next_segment": start,
        "steps_per_segment": comparison.STEPS_PER_SEGMENT,
        "pid": {"parent_qbit_a": 11, "qbit_control": 12, "qbit_restart_chain": 13}[arm],
        "executable_sha256": "f" * 64,
        "inputs": "/fixtures/original-inputs.json",
        "inputs_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "conditioning_sha256": "d" * 64,
        "model_index_sha256": "e" * 64,
        "upstream_revision": "fixture-upstream",
        "segment_text_tokens": comparison.TEXT_PROFILE if edited else comparison.BASE_TEXT_PROFILE,
        "segment_cache_text": comparison.CACHE_PROFILE if edited else comparison.BASE_CACHE_PROFILE,
        "valid_text_tokens": 557,
        "prompt_edit": PROMPT_IDENTITY if edited else None,
        "prompt_inputs": PROMPT_INPUTS["manifest_path"] if edited else "",
        "prompt_inputs_sha256": PROMPT_INPUTS["manifest_sha256"] if edited else "",
        "invalidation": comparison.prompt.PROMPT_INVALIDATION if edited else None,
        "noise_edit": None,
        "noise_scope": comparison.prompt.UNCHANGED_NOISE_SCOPE,
        "timeline_shape": comparison.TIMELINE_SHAPE,
        "step_shape": comparison.STEP_SHAPE,
        "state_format": "qbit",
        "cache_roundtrip": "exact" if arm == "parent_qbit_a" or arm == "qbit_restart_chain" else "qbit",
        "qbit_roundtrip_boundaries": [2] if arm == "qbit_control" else [],
        "compact_pack_boundary": 1 if arm == "parent_qbit_a" else 2 if arm == "qbit_restart_chain" and stop == 2 else 0,
        "codec": {"pack": {}} if arm == "parent_qbit_a" else (
            {"restore": {}, "roundtrip2": {}} if arm == "qbit_control" else
            {"restore": {}, "pack": {}} if stop == 2 else {"restore": {}}
        ),
        "boundary_contract_sha256": {"1": "6" * 64, "2": "7" * 64},
        "contract_sha256": "" if start == 0 else "6" * 64 if start == 1 else "7" * 64,
        "state_input_sha256": "" if start == 0 else "8" * 64,
        "metrics": {
            "encode_allocations": 0,
            "encode_readbacks": 0,
            "preterminal_waits": 0,
            "command_segments": 4,
            "terminal_waits": 1,
        },
        "checkpoint_file_bytes": 8_000_000 if stop < 3 else 0,
        "checkpoint_payload_bytes": 20_000_000 if stop < 3 else 0,
        "resident_f32_kv_bytes": 100_000_000,
        "logical_packed_kv_bytes": 1 if stop < 3 else 0,
        "decoder": False,
        "full_generation": False,
        "full_upstream_video_parity": False,
    }
    return receipt


class QBitCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = _tempdir()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "boundary.hnqc"
        self.raw, self.inventory = _checkpoint_fixture(self.path)

    def test_authenticates_compact_hnrs_records_and_qbit_inventory(self):
        parsed = _parse_fixture(self.path, self.raw, self.inventory)
        self.assertEqual(parsed["record_count"], 4)
        self.assertEqual(parsed["cache_records"], 2)
        self.assertEqual(parsed["codec_stats"]["tier_counts"]["p4"], 4)
        self.assertEqual(parsed["payload_bytes"], len(self.raw) - comparison.HNRS_PAYLOAD_OFFSET)
        self.assertNotEqual(parsed["payload_bytes"], sum(self.inventory.values()))

    def test_rejects_wrong_after_a_base_contract(self):
        with self.assertRaisesRegex(ValueError, "stale or different base boundary"):
            comparison._parse_qbit_checkpoint(
                self.path,
                expected_sha256=_sha(self.raw),
                expected_file_bytes=len(self.raw),
                expected_payload_bytes=len(self.raw) - comparison.HNRS_PAYLOAD_OFFSET,
                expected_base_contract='{"schema":"wrong-base","next_segment":1}',
                expected_inventory=self.inventory,
                expected_timeline_sha256=_sha(bytes(self.inventory["timeline"])),
                label="fixture",
            )

    def test_reconstructs_fresh_after_a_contract_for_qbit_executable(self):
        fields = comparison.prompt.BASE_CONTRACT_FIELD_ORDER
        contract = {key: f"fixture-{key}" for key in fields}
        contract["next_segment"] = 1
        contract["executable_sha256"] = "f" * 64
        exact = {
            "state_chains": {"parent_a": {"state": {"contract_document": contract}}},
            "source_bank": {"original_valid_text_tokens": 557},
        }
        qbit_receipt = {"executable_sha256": "9" * 64}
        with mock.patch.object(comparison.prompt, "_validate_base_contract_document") as validate:
            _doc, raw, after_a_sha, after_b_sha = comparison._reconstruct_qbit_base_contract(
                exact, qbit_receipt
            )
        self.assertEqual(validate.call_args.args[0]["executable_sha256"], "9" * 64)
        self.assertEqual(tuple(json.loads(raw)), fields)
        self.assertEqual(after_a_sha, _sha(raw.encode()))
        self.assertNotEqual(after_a_sha, after_b_sha)

    def test_rejects_payload_digest_change(self):
        changed = self.raw[:-1] + bytes([self.raw[-1] ^ 1])
        self.path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            _parse_fixture(self.path, self.raw, self.inventory)

    def test_rejects_wrong_hnrs_prefix(self):
        raw, inventory = _checkpoint_fixture(self.path, outer_schema="wrong-hnrs-schema")
        with self.assertRaisesRegex(ValueError, "HNRS schema is unsupported"):
            _parse_fixture(self.path, raw, inventory)

    def test_rejects_stale_qbit_codec_identity(self):
        raw, inventory = _checkpoint_fixture(
            self.path, manifest_identity="{}"
        )
        with self.assertRaisesRegex(ValueError, "codec manifest identity mismatch"):
            _parse_fixture(self.path, raw, inventory)

    def test_rejects_unexpected_cache_inventory(self):
        wrong_inventory = dict(self.inventory, **{"history.layer0.extra": 1024})
        with self.assertRaisesRegex(ValueError, "inventory does not match"):
            comparison._parse_qbit_checkpoint(
                self.path,
                expected_sha256=_sha(self.raw),
                expected_file_bytes=len(self.raw),
                expected_payload_bytes=len(self.raw) - comparison.HNRS_PAYLOAD_OFFSET,
                expected_base_contract='{"schema":"fixture-base","next_segment":1}',
                expected_inventory=wrong_inventory,
                expected_timeline_sha256=_sha(bytes(self.inventory["timeline"])),
                label="fixture",
            )

    def test_rejects_invalid_row_tier_even_with_rehashed_container(self):
        raw, inventory = _checkpoint_fixture(self.path, bad_tier=True)
        with self.assertRaisesRegex(ValueError, "row tier is unsupported"):
            _parse_fixture(self.path, raw, inventory)

    def test_rejects_changed_checkpoint_bytes_after_parsing(self):
        parsed = _parse_fixture(self.path, self.raw, self.inventory)
        self.assertEqual(parsed["base_contract_sha256"], _sha(b'{"schema":"fixture-base","next_segment":1}'))
        self.path.write_bytes(self.raw[:-1] + bytes([self.raw[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            _parse_fixture(self.path, self.raw, self.inventory)


class ReceiptAndLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exact = _exact_report()

    def _validate(self, receipt: dict, arm: str, interval: tuple[int, int]) -> None:
        comparison._validate_receipt(
            receipt, arm, interval, self.exact, PROMPT_IDENTITY, PROMPT_INPUTS,
            self.exact["source_bank"]["original_prompt_sha256"],
        )

    def test_accepts_parent_a_and_edited_qbit_receipts(self):
        self._validate(_receipt("parent_qbit_a", (0, 1)), "parent_qbit_a", (0, 1))
        self._validate(_receipt("qbit_control", (1, 3)), "qbit_control", (1, 3))

    def test_rejects_wrong_prompt_identity(self):
        receipt = _receipt("qbit_control", (1, 3))
        receipt["prompt_edit"] = dict(PROMPT_IDENTITY, prompt_sha256="9" * 64)
        with self.assertRaisesRegex(ValueError, "prompt_edit identity"):
            self._validate(receipt, "qbit_control", (1, 3))

    def test_rejects_qbit_control_with_exact_roundtrip_flag(self):
        receipt = _receipt("qbit_control", (1, 3))
        receipt["cache_roundtrip"] = "exact"
        with self.assertRaisesRegex(ValueError, "cache_roundtrip is incorrect"):
            self._validate(receipt, "qbit_control", (1, 3))

    def test_rejects_parent_with_prompt_edit_claim(self):
        receipt = _receipt("parent_qbit_a", (0, 1))
        receipt["prompt_edit"] = PROMPT_IDENTITY
        with self.assertRaisesRegex(ValueError, "parent A cannot claim a prompt edit"):
            self._validate(receipt, "parent_qbit_a", (0, 1))

    def test_rejects_logical_inventory_mismatch_in_checkpoint_receipt(self):
        receipt = _receipt("parent_qbit_a", (0, 1))
        inventory = comparison._expected_inventory_from_contract(
            {"layers": 54, "heads": 16, "head_dim": 128, "video_tokens": 4320},
            1, comparison.BASE_TEXT_PROFILE, comparison.BASE_CACHE_PROFILE,
        )
        receipt["checkpoint_payload_bytes"] = sum(inventory.values()) - 1
        validated = comparison._validate_receipt(
            receipt, "parent_qbit_a", (0, 1), self.exact,
            PROMPT_IDENTITY, PROMPT_INPUTS,
            self.exact["source_bank"]["original_prompt_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "logical checkpoint payload"):
            comparison._validate_logical_checkpoint_inventory(
                validated, inventory, "parent_qbit_a"
            )

    def test_rejects_wrong_state_link(self):
        stats_a = {"stats": "after-a"}
        stats_b = {"stats": "after-b"}
        parent_state = {"sha256": "a" * 64, "base_contract_sha256": "1" * 64, "codec_stats": stats_a}
        edited_state = {"sha256": "b" * 64, "base_contract_sha256": "2" * 64, "codec_stats": stats_b}
        arms = {
            "parent_qbit_a": [{"state": parent_state, "receipt": {"boundary_contract_sha256": {"1": "1" * 64}}}],
            "qbit_control": [{
                "state": None,
                "receipt": {"state_input_sha256": "c" * 64, "contract_sha256": "1" * 64},
                "codec_stats": {"restore": stats_a},
            }],
            "qbit_restart_chain": [
                {
                    "state": edited_state,
                    "receipt": {
                        "state_input_sha256": "a" * 64,
                        "contract_sha256": "1" * 64,
                        "boundary_contract_sha256": {"2": "2" * 64},
                    },
                    "codec_stats": {"restore": stats_a},
                },
                {
                    "state": None,
                    "receipt": {
                        "state_input_sha256": "b" * 64,
                        "contract_sha256": "2" * 64,
                    },
                    "codec_stats": {"restore": stats_b},
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "does not consume the fresh same-matrix after-A"):
            comparison._validate_state_lineage(arms)

    def test_rejects_stale_source_pinned_executable(self):
        with _tempdir() as temp:
            root = Path(temp)
            run = root / "run"
            real = run / "real"
            build = run / "build"
            real.mkdir(parents=True)
            build.mkdir()
            index_path = real / "matrix-index.json"
            index_path.write_text("{}")
            exact_sha = "a" * 64
            runner = run / "run_qbit_matrix.py"
            executable = build / "hy_nomas_compact_sequence_probe"
            runner.write_bytes(b"runner")
            executable.write_bytes(b"fresh binary")
            sources = {
                relative: _sha((ROOT / relative).read_bytes())
                for relative in comparison.PINNED_SOURCE_PATHS
            }
            pins = {
                "head": "1" * 40,
                "executable_sha256": _sha(b"fresh binary"),
                "sources": sources,
                "exact_index_sha256": exact_sha,
                "runner_sha256": _sha(b"runner"),
            }
            (run / "source-pins.json").write_text(json.dumps(pins))
            self.assertTrue(comparison._load_source_pins(index_path, exact_sha)["executable_file_verified"])
            executable.write_bytes(b"stale binary")
            with self.assertRaisesRegex(ValueError, "executable bytes differ"):
                comparison._load_source_pins(index_path, exact_sha)


class CaptureAndComparatorTests(unittest.TestCase):
    def test_allows_generated_b_and_c_but_rejects_a_or_untouched_suffix_changes(self):
        exact = bytes(comparison.TIMELINE_BYTES)
        frame_bytes = comparison.TIMELINE_SHAPE[3] * comparison.TIMELINE_SHAPE[4] * 4

        def generated(payload: bytes, segment: int) -> bytes:
            changed = bytearray(payload)
            changed[segment * comparison.SEGMENT_LATENTS * frame_bytes] = segment + 1
            return bytes(changed)

        parent = {"arm": "parent_qbit_a", "interval": (0, 1), "timeline": exact}
        arms = {
            "parent_qbit_a": [parent],
            "qbit_control": [{
                "arm": "qbit_control", "interval": (1, 3),
                "timeline": generated(generated(exact, 1), 2),
            }],
            "qbit_restart_chain": [
                {
                    "arm": "qbit_restart_chain", "interval": (1, 2),
                    "timeline": generated(exact, 1),
                },
                {
                    "arm": "qbit_restart_chain", "interval": (2, 3),
                    "timeline": generated(exact, 2),
                },
            ],
        }
        report = comparison._validate_latent_a_and_future(arms, exact)
        self.assertTrue(report["a_latent_bitwise_equal_to_prior_exact"])
        self.assertEqual(
            report["untouched_suffix_checks"]["qbit_control:1-3"]["start_frame"],
            3 * comparison.SEGMENT_LATENTS,
        )
        self.assertEqual(
            report["untouched_suffix_checks"]["qbit_restart_chain:1-2"]["start_frame"],
            2 * comparison.SEGMENT_LATENTS,
        )
        changed_a = bytearray(exact)
        changed_a[0] = 1
        arms["qbit_control"][0]["timeline"] = bytes(changed_a)
        with self.assertRaisesRegex(ValueError, "changed the immutable A latent"):
            comparison._validate_latent_a_and_future(arms, exact)
        arms["qbit_control"][0]["timeline"] = generated(
            generated(generated(exact, 1), 2), 3
        )
        with self.assertRaisesRegex(ValueError, "untouched future latent/noise"):
            comparison._validate_latent_a_and_future(arms, exact)

    def test_reports_full_restart_capture_coverage_and_tolerance(self):
        zero_timeline = bytes(comparison.TIMELINE_BYTES)
        zero_step = bytes(comparison.STEP_BYTES)

        def run(interval: tuple[int, int]) -> dict:
            return {
                "interval": interval,
                "timeline": zero_timeline,
                "steps": {
                    f"segment{segment}-step{step}": (zero_step, {})
                    for segment in range(*interval)
                    for step in range(1, comparison.STEPS_PER_SEGMENT + 1)
                },
            }

        result = comparison._compare_qbit_restart({
            "qbit_control": [run((1, 3))],
            "qbit_restart_chain": [run((1, 2)), run((2, 3))],
        })
        self.assertTrue(result["within_tolerance"])
        self.assertEqual(result["step_capture_count"], 8)
        self.assertEqual(result["segment_timeline_count"], 2)
        self.assertEqual(result["full_timeline_count"], 1)
        self.assertEqual(result["tolerance"], {"atol": 2e-5, "rtol": 2e-5})

    def test_requires_nonvacuous_qbit_b_and_c_against_exact_original(self):
        zero_timeline = bytes(comparison.TIMELINE_BYTES)
        zero_step = bytes(comparison.STEP_BYTES)
        original_meta = {
            "timeline": {
                "path": "/fixtures/original-timeline.f32le",
                "bytes": comparison.TIMELINE_BYTES,
                "sha256": _sha(zero_timeline),
            },
            "steps": {},
        }
        for segment in (1, 2):
            for step in range(1, comparison.STEPS_PER_SEGMENT + 1):
                key = f"segment{segment}-step{step}"
                original_meta["steps"][key] = {
                    "path": f"/fixtures/{key}.f32le",
                    "bytes": comparison.STEP_BYTES,
                    "sha256": _sha(zero_step),
                }
        qbit_steps = {
            f"segment{segment}-step{step}": (zero_step, {})
            for segment in (1, 2)
            for step in range(1, comparison.STEPS_PER_SEGMENT + 1)
        }
        with _tempdir() as temp:
            root = Path(temp)
            original_meta["timeline"]["path"] = str(root / "original-timeline.f32le")
            (root / "original-timeline.f32le").write_bytes(zero_timeline)
            for key in original_meta["steps"]:
                path = root / f"{key}.f32le"
                path.write_bytes(zero_step)
                original_meta["steps"][key]["path"] = str(path)
            qbit = {
                "timeline": zero_timeline,
                "steps": qbit_steps,
            }
            report = {"artifacts": {"original_control": [original_meta]}}
            with self.assertRaisesRegex(ValueError, "B output is indistinguishable"):
                comparison._compare_qbit_prompt_effect(
                    {"qbit_control": [qbit]}, report
                )
            changed = bytearray(zero_step)
            changed[:4] = struct.pack("<f", 1.0)
            qbit["steps"]["segment1-step1"] = (bytes(changed), {})
            qbit["steps"]["segment2-step1"] = (bytes(changed), {})
            result = comparison._compare_qbit_prompt_effect(
                {"qbit_control": [qbit]}, report
            )
            self.assertTrue(result["b_differs_from_prior_exact_original"])
            self.assertTrue(result["c_differs_from_prior_exact_original"])
            self.assertIn("not causal isolation", result["scope"])


if __name__ == "__main__":
    unittest.main()
