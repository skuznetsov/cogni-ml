"""CPU-only admission and provenance falsifiers; never constructs a model on MPS."""
from __future__ import annotations

import hashlib
from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_upstream_trace_decode as decode
import hy_nomas_pixelumm_checkpoint as checkpoint
import hy_nomas_upstream_activation_recording as recorder
import hy_nomas_soft_route_render_reference as render


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _replay(arm: str, latent_path: Path, latent_sha: str) -> dict:
    return {
        "schema": decode.RECORDER_SCHEMA,
        "pilot": False,
        "all_euler_boundaries_exact": True,
        "trace_hashes_validated": True,
        "gates": [{"name": f"{arm}.step{step}.latent", "bitwise_equal": True,
                   "max_abs_error": 0.0} for step in range(1, 5)],
        "runtime": {"device": "mps", "total_block_invocations": 432,
                    "diagnostic_activation_exports": {
                        "block_calls": 432, "source_math_device": "mps",
                        "exports_add_diagnostic_waits": True}},
        "final_latent": {"path": str(latent_path), "sha256": latent_sha,
                         "shape": list(decode.LATENT_SHAPE), "dtype": "float32"},
    }


def _recording_metadata() -> dict:
    source_hashes = {
        decode.real_input.UPSTREAM_MODEL_RELATIVE: decode.UPSTREAM_MODEL_SHA256,
        decode.real_input.PIXELUMM_NOISE_RELATIVE: decode.AUTOENCODER_SOURCE_SHA256,
    }
    return {
        "schema": decode.RECORDER_SCHEMA,
        "arm": "A1_high",
        "diagnostic_only": True,
        "upstream_revision": decode.UPSTREAM_REVISION,
        "plan": {"steps": 4, "blocks": 54, "block_calls": 432,
                 "recording_exports_are_diagnostic": True},
        "accepted_source": {"path": str(recorder.BASELINE.resolve()),
                            "sha256": recorder.BASELINE_SHA256},
        "model": {"upstream_revision": decode.UPSTREAM_REVISION,
                  "upstream_path": str(decode.UPSTREAM_ROOT.resolve()),
                  "source_hashes": source_hashes},
        "diagnostic_sources": recorder.diagnostic_source_bindings(),
        "render_helper_sha256": decode._sha256_file(Path(render.__file__)),
        "recorder_sha256": decode._sha256_file(Path(recorder.__file__)),
    }


class UpstreamTraceDecodeTests(unittest.TestCase):
    def test_plan_is_static_and_names_bounded_sources_and_memory_cap(self):
        plan = decode.plan_document()
        self.assertFalse(plan["model_launched"])
        self.assertEqual(plan["windows"]["starts"], [0, 5, 10, 15, 20])
        self.assertEqual(plan["weights"]["per_tensor_hashes_required"], 468)
        self.assertFalse(plan["weights"]["full_cache_mapped"])
        self.assertEqual(decode.MAX_ESTIMATED_PROCESS_BYTES, 10 * decode.GIB)
        memory = plan["memory_plan"]
        self.assertEqual(memory["admission_memory_cap_bytes"], 10 * decode.GIB)
        self.assertEqual(memory["minimum_host_reserve_fraction"], 0.10)
        self.assertGreater(memory["estimated_total_bytes"], 0)
        self.assertIn("noise_and_stem_head_cpu_device_allowance_bytes", memory)
        self.assertIn("raw_display_blender_export_scratch_bytes", memory)

    def test_replay_admission_requires_nonpilot_exact_gates_and_nested_mps_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final-latent.f32le"
            digest = "a" * 64
            valid = _replay("A1_high", path, digest)
            decode.validate_replay_document(valid, arm="A1_high", final_latent_path=path,
                                            expected_latent_sha256=digest)
            changed = json.loads(json.dumps(valid))
            changed["runtime"]["diagnostic_activation_exports"]["source_math_device"] = "cpu"
            with self.assertRaisesRegex(decode.DecodeError, "runtime receipt"):
                decode.validate_replay_document(changed, arm="A1_high", final_latent_path=path,
                                                expected_latent_sha256=digest)
            changed = json.loads(json.dumps(valid))
            changed["gates"][2]["max_abs_error"] = 1e-7
            with self.assertRaisesRegex(decode.DecodeError, "Euler replay gate"):
                decode.validate_replay_document(changed, arm="A1_high", final_latent_path=path,
                                                expected_latent_sha256=digest)
            changed = json.loads(json.dumps(valid))
            changed["gates"][0]["max_abs_error"] = False
            with self.assertRaisesRegex(decode.DecodeError, "Euler replay gate"):
                decode.validate_replay_document(changed, arm="A1_high", final_latent_path=path,
                                                expected_latent_sha256=digest)
            changed = json.loads(json.dumps(valid))
            changed["trace_hashes_validated"] = False
            with self.assertRaisesRegex(decode.DecodeError, "trace hashes"):
                decode.validate_replay_document(changed, arm="A1_high", final_latent_path=path,
                                                expected_latent_sha256=digest)
            changed = json.loads(json.dumps(valid))
            changed["runtime"]["diagnostic_activation_exports"]["exports_add_diagnostic_waits"] = False
            with self.assertRaisesRegex(decode.DecodeError, "runtime receipt"):
                decode.validate_replay_document(changed, arm="A1_high", final_latent_path=path,
                                                expected_latent_sha256=digest)

    def test_replay_binds_trace_record_and_byte_counts_to_authenticated_index(self):
        replay = {"trace_records": 7, "trace_payload_bytes": 2048}
        index = {"records": [{}] * 7, "total_bytes": 2048}
        decode._validate_replay_trace_counts(replay, index)
        replay["trace_records"] = True
        with self.assertRaisesRegex(decode.DecodeError, "record/byte counts"):
            decode._validate_replay_trace_counts(replay, index)
        replay["trace_records"] = 6
        with self.assertRaisesRegex(decode.DecodeError, "record/byte counts"):
            decode._validate_replay_trace_counts(replay, index)
        replay["trace_records"] = 7
        replay["trace_payload_bytes"] = 2049
        with self.assertRaisesRegex(decode.DecodeError, "record/byte counts"):
            decode._validate_replay_trace_counts(replay, index)

    def test_final_latent_is_exact_expansion_of_authenticated_after_euler_record(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            activation_dir = run / "activations"
            activation_dir.mkdir()
            words = np.zeros(decode.LATENT_BYTES // 4, dtype="<u2").tobytes()
            shard_path = activation_dir / "batch-000000.bin"
            shard_path.write_bytes(words)
            latent_path = run / "final-latent.f32le"
            latent_bytes = decode.expand_bf16_le_to_f32_bytes(words)
            latent_path.write_bytes(latent_bytes)
            digest = _sha(latent_bytes)
            index = {"records": [{"name": "step4.A1_high.head.after_euler",
                                  "dtype": "bfloat16", "shape": list(decode.LATENT_SHAPE),
                                  "nbytes": len(words), "offset": 0,
                                  "shard": shard_path.name, "sha256": _sha(words)}]}
            replay = _replay("A1_high", latent_path, digest)
            result_path, record, snapshot, identity = decode._validate_final_latent(
                run, index, replay, arm="A1_high", expected_latent_sha256=digest,
            )
            self.assertEqual(result_path, latent_path)
            self.assertEqual(record["name"], "step4.A1_high.head.after_euler")
            self.assertEqual(snapshot, latent_bytes)
            self.assertEqual(identity[2], decode.LATENT_BYTES)
            latent_path.write_bytes(bytes(len(latent_bytes)))
            # The admitted CPU snapshot remains the exact payload already checked.
            self.assertEqual(snapshot, latent_bytes)
            with self.assertRaisesRegex(decode.DecodeError, "final latent.*changed after admission"):
                decode._assert_file_identity(latent_path, identity, "final latent")

    def test_rehashed_same_shape_latent_still_must_match_authenticated_trace_before_mps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / "recordings" / "case"
            activation_dir = run / "activations"
            activation_dir.mkdir(parents=True)
            words = np.zeros(decode.LATENT_BYTES // 4, dtype="<u2").tobytes()
            shard_path = activation_dir / "batch-000000.bin"
            shard_path.write_bytes(words)
            latent_path = run / "final-latent.f32le"
            altered = bytearray(decode.expand_bf16_le_to_f32_bytes(words))
            altered[:4] = np.asarray([1.0], dtype="<f4").tobytes()
            altered_bytes = bytes(altered)
            latent_path.write_bytes(altered_bytes)
            latent_sha = _sha(altered_bytes)
            index = {
                "schema": decode.TRACE_SCHEMA,
                "version": 1,
                "complete": True,
                "metadata": _recording_metadata(),
                "records": [{"name": "step4.A1_high.head.after_euler",
                             "dtype": "bfloat16", "shape": list(decode.LATENT_SHAPE),
                             "nbytes": len(words), "offset": 0,
                             "shard": shard_path.name, "sha256": _sha(words)}],
                "total_bytes": len(words),
            }
            index_bytes = json.dumps(index).encode("utf-8")
            (activation_dir / "index.json").write_bytes(index_bytes)
            replay = _replay("A1_high", latent_path, latent_sha)
            replay.update({"trace_records": 1, "trace_payload_bytes": len(words)})
            replay_bytes = json.dumps(replay).encode("utf-8")
            (run / "replay.json").write_bytes(replay_bytes)

            argv = [
                "--execute", "--recording-dir", str(run),
                "--index-sha256", _sha(index_bytes),
                "--replay-sha256", _sha(replay_bytes),
                "--latent-sha256", latent_sha,
                "--output-dir", str(root / "output"),
            ]
            stderr = StringIO()
            with (
                patch.object(decode, "RUN_ROOT", root),
                patch.object(decode.activation_trace, "validate_trace", return_value=index),
                patch.object(decode, "_authenticate_weight_bundle",
                             side_effect=AssertionError("weight admission reached")) as weights,
                patch.object(decode, "_verify_upstream_sources",
                             side_effect=AssertionError("source admission reached")) as sources,
                patch.object(decode, "_validate_runtime",
                             side_effect=AssertionError("MPS admission reached")) as runtime,
                patch.object(decode, "run_decode",
                             side_effect=AssertionError("decoder execution reached")) as execute,
                redirect_stderr(stderr),
            ):
                result = decode.main(argv)

            self.assertEqual(result, 2)
            self.assertIn("not the exact BF16 trace expansion", stderr.getvalue())
            weights.assert_not_called()
            sources.assert_not_called()
            runtime.assert_not_called()
            execute.assert_not_called()

    def test_replacement_after_admission_is_rejected_before_runtime_or_mps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latent = root / "final-latent.f32le"
            noise = root / "anchor.bin"
            latent.write_bytes(b"admitted latent")
            noise.write_bytes(b"admitted noise")
            _, latent_identity = decode._read_regular_snapshot(latent, "latent")
            _, noise_identity = decode._read_regular_snapshot(noise, "noise")
            admission = {"latent_path": latent, "latent_file_identity": latent_identity,
                         "noise": {"payload_path": str(noise),
                                   "payload_file_identity": noise_identity}}
            replacement = root / "replacement.bin"
            replacement.write_bytes(b"unauthorized latent")
            os.replace(replacement, latent)
            with patch.object(decode, "_validate_runtime", side_effect=AssertionError("MPS reached")) as runtime:
                with self.assertRaisesRegex(decode.DecodeError, "final latent changed after admission"):
                    decode.run_decode(admission=admission, weights={}, source_hashes={},
                                      output=root / "unused-output")
            runtime.assert_not_called()

    def test_metadata_binds_recorder_source_inventory_and_rejects_wrong_model_source(self):
        metadata = _recording_metadata()
        index = {"schema": decode.TRACE_SCHEMA, "version": 1, "complete": True,
                 "metadata": metadata}
        self.assertIs(decode.validate_recording_metadata(index, arm="A1_high"), metadata)
        # The real trace records the resolved volume path; legacy /opt is an alias
        # to the same pinned checkout and is accepted only after strict resolution.
        metadata["model"]["upstream_path"] = str(decode.real_input.UPSTREAM_ROOT)
        self.assertIs(decode.validate_recording_metadata(index, arm="A1_high"), metadata)
        self.assertNotIn(decode.real_input.DCAE_ENCODER_RELATIVE,
                         metadata["model"]["source_hashes"])
        metadata["model"]["source_hashes"][decode.real_input.PIXELUMM_NOISE_RELATIVE] = "0" * 64
        with self.assertRaisesRegex(decode.DecodeError, "source-code digest mismatch"):
            decode.validate_recording_metadata(index, arm="A1_high")

    def test_pinned_decoder_sources_are_separately_authenticated_from_recorder_hashes(self):
        hashes = decode._verify_upstream_sources(decode.UPSTREAM_ROOT)
        self.assertEqual(hashes[decode.real_input.DCAE_ENCODER_RELATIVE],
                         decode.DCAE_SOURCE_SHA256)
        self.assertEqual(hashes[decode.real_input.UPSTREAM_MODEL_RELATIVE],
                         decode.UPSTREAM_MODEL_SHA256)
        self.assertEqual(hashes[decode.real_input.PIXELUMM_NOISE_RELATIVE],
                         decode.AUTOENCODER_SOURCE_SHA256)

    def test_cache_structure_covers_468_tensor_ranges_and_rejects_overlap(self):
        manifest: dict = {"blocks": {}, "weights": {}}
        header: dict = {}
        cache_cursor = 0
        source_cursor = 0
        for block_index in range(36):
            schema = checkpoint.block_schema(block_index)
            source_start = source_cursor
            for name, shape in sorted(schema.items()):
                count = int(np.prod(shape))
                bf16_bytes = count * 2
                f32_bytes = count * 4
                manifest["weights"][f"block{block_index}.{name}"] = {
                    "dtype": "BF16", "shape": list(shape), "nbytes": bf16_bytes,
                    "offset": cache_cursor, "sha256": "a" * 64,
                }
                header[f"decoder.layers.{block_index}.{name}"] = {
                    "data_offsets": [source_cursor, source_cursor + f32_bytes],
                }
                cache_cursor += bf16_bytes
                source_cursor += f32_bytes
            manifest["blocks"][str(block_index)] = {
                "selected_block": block_index,
                "source_file_sha256": decode.SOURCE_CHECKPOINT_SHA256,
                "conversion_device": "cpu", "bf16_bytes": decode.CACHE_BLOCK_BYTES,
                "source_f32_bytes": 2 * decode.CACHE_BLOCK_BYTES,
                "source_data_offsets": [source_start, source_cursor],
            }
        self.assertEqual(cache_cursor, decode.RECOVERY_CACHE_BYTES)
        decode._validate_cache_manifest_structure(manifest, header, cache_cursor)
        first = sorted(manifest["weights"])[0]
        manifest["weights"][first]["offset"] += 1
        with self.assertRaisesRegex(decode.DecodeError, "metadata mismatch"):
            decode._validate_cache_manifest_structure(manifest, header, cache_cursor)

    def test_memory_plan_rejects_less_than_reserve_and_exposes_overbound(self):
        arch = SimpleNamespace(hidden_size=4096, intermediate_size=12288, num_heads=32,
                               num_kv_heads=8, head_dim=128)
        plan = decode._memory_plan(arch, physical_bytes=64 * decode.GIB)
        self.assertLess(plan["estimated_total_bytes"], plan["admission_memory_cap_bytes"])
        self.assertGreater(plan["diagnostic_capture_policy_overbound_bytes"],
                           plan["single_block_no_capture_reference_bytes"])
        allocated = (64 * decode.GIB - plan["estimated_total_bytes"]
                     - plan["required_reserve_bytes"] + 1)
        with self.assertRaisesRegex(decode.DecodeError, "10% host reserve"):
            decode._memory_plan(arch, physical_bytes=64 * decode.GIB,
                                allocated_bytes=allocated)


if __name__ == "__main__":
    unittest.main()
