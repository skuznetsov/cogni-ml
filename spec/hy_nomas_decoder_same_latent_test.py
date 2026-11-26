"""CPU-only admission and geometry tests for the same-latent decoder oracle."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_decoder_same_latent as oracle


class SameLatentAdmissionTests(unittest.TestCase):
    @contextmanager
    def _stubbed_main(self):
        receipt_bytes = b"test receipt bytes"
        receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
        output_json = Path("/tmp/oracle.json")
        output_paths = (
            output_json,
            output_json.with_suffix(".f32le"),
            output_json.with_name("oracle-first-window.f32le"),
            output_json.with_name("oracle-display.f32le"),
        )
        args = [
            "--native-receipt", "/tmp/receipt.json",
            "--native-receipt-sha256", receipt_sha,
            "--inputs", "/tmp/inputs.json",
            "--inputs-sha256", "a" * 64,
        ]
        with (
            patch.object(oracle, "_canonical_file", side_effect=lambda path, _: Path(path)),
            patch.object(oracle, "_read_json", return_value=({"test": True}, receipt_bytes)),
            patch.object(oracle, "validate_native_receipt_document", return_value={"contract": True}),
            patch.object(oracle, "_validate_inputs", return_value=({}, Path("/tmp/payload.bin"), {})),
            patch.object(oracle, "_resolve_geometry", return_value=oracle.DEFAULT_GEOMETRY),
            patch.object(oracle, "_preflight_artifacts") as artifact_preflight,
            patch.object(oracle, "_output_paths", return_value=output_paths) as output_preflight,
            patch.object(oracle, "_validate_execution_runtime") as runtime_preflight,
            patch.object(oracle, "_admission_summary", return_value={"admitted": True}),
        ):
            yield args, artifact_preflight, output_preflight, runtime_preflight

    def _receipt(self, resolution: str = oracle.DEFAULT_RESOLUTION) -> dict:
        geometry = oracle._geometry_for_resolution(resolution)
        return {
            "schema": oracle.NATIVE_SCHEMA,
            "upstream_revision": oracle.block_reference.UPSTREAM_REVISION,
            "precision": oracle.NATIVE_PRECISION,
            "dit_weights": "bf16",
            "dit_quantization_experimental": False,
            "seed": 42,
            "frames": 120,
            "width": geometry.width,
            "height": geometry.height,
            "fps": 24,
            "steps": 4,
            "shift": 5,
            "cfg": 1,
            "camera_conditioning": False,
            "action_conditioning": False,
            "prompt": "authenticated prompt",
            "inputs_sha256": "a" * 64,
            "inputs": "/tmp/inputs.json",
            "output": "/tmp/display.f32le",
            "output_bytes": geometry.raw_bytes,
            "output_sha256": "d" * 64,
            "pixelumm": {
                "steps": 1,
                "window_latents": 10,
                "overlap_latents": 5,
                "starts": [0, 5, 10, 15, 20],
            },
            "diagnostics": {
                "latent": "/tmp/latent.f32le",
                "latent_sha256": "b" * 64,
                "raw_pixels": "/tmp/raw.f32le",
                "raw_pixels_sha256": "c" * 64,
            },
        }

    def _manifest(self, resolution: str = oracle.DEFAULT_RESOLUTION,
                  schema: str | None = None) -> dict:
        geometry = oracle._geometry_for_resolution(resolution)
        if schema is None:
            schema = (
                oracle.generation_inputs.HIGH_RESOLUTION_SCHEMA
                if resolution == oracle.HIGH_RESOLUTION
                else oracle.generation_inputs.SCHEMA
            )
        tensors = {
            name: {
                "dtype": dtype,
                "shape": list(shape),
                "nbytes": oracle.generation_inputs.tensor_nbytes(dtype, shape),
            }
            for name, dtype, shape in oracle.generation_inputs.tensor_specs_for_resolution(
                resolution
            )
        }
        return {
            "schema": schema,
            "geometry": {
                "dit_latent_shape": list(geometry.latent_shape),
                "decoder_latent_grid": [
                    30, geometry.latent_height, geometry.latent_width,
                ],
                "decoder_noise_shape": list(geometry.noise_shape),
                "decoder_rope_grid": list(geometry.token_grid),
                "decoder_rope_shape": [1, 10 * 2 * geometry.latent_height * 2 * geometry.latent_width, 1, 128],
                "decoder_local_attention_tile_pixels": 128,
                "decoder_local_attention_tile_tokens": 8,
                "decoder_noise_piece_frames": 4,
                "decoder_noise_piece_latents": 1,
            },
            "tensors": tensors,
        }

    def test_receipt_rejects_unknown_precision(self):
        receipt = self._receipt()
        receipt["precision"] = "bf16-ish"
        with self.assertRaisesRegex(oracle.SameLatentError, "precision is unknown"):
            oracle.validate_native_receipt_document(
                receipt, expected_inputs_sha256="a" * 64
            )

    def test_receipt_rejects_bad_authorized_input_digest(self):
        with self.assertRaisesRegex(oracle.SameLatentError, "does not authenticate"):
            oracle.validate_native_receipt_document(
                self._receipt(), expected_inputs_sha256="not-a-digest"
            )

    def test_default_receipt_and_v1_geometry_remain_compatible(self):
        contract = oracle.validate_native_receipt_document(
            self._receipt(), expected_inputs_sha256="a" * 64
        )
        geometry = oracle._resolve_geometry(contract, self._manifest())
        self.assertEqual(geometry, oracle.DEFAULT_GEOMETRY)
        self.assertEqual(contract["display_pixels"], "/tmp/display.f32le")
        self.assertEqual(contract["display_pixels_bytes"], oracle.RAW_BYTES)
        self.assertEqual(contract["display_pixels_sha256"], "d" * 64)
        self.assertEqual(geometry.token_grid, (10, 18, 32))

    def test_v2_dynamic_default_bundle_remains_at_512x288(self):
        contract = oracle.validate_native_receipt_document(
            self._receipt(), expected_inputs_sha256="a" * 64
        )
        geometry = oracle._resolve_geometry(
            contract,
            self._manifest(schema=oracle.generation_inputs.DYNAMIC_SCHEMA),
        )
        self.assertEqual(geometry, oracle.DEFAULT_GEOMETRY)

    def test_high_resolution_v3_uses_exact_partial_tile_geometry(self):
        receipt = oracle.validate_native_receipt_document(
            self._receipt(oracle.HIGH_RESOLUTION), expected_inputs_sha256="a" * 64
        )
        geometry = oracle._resolve_geometry(
            receipt, self._manifest(oracle.HIGH_RESOLUTION)
        )
        self.assertEqual(geometry.latent_shape, (1, 32, 30, 13, 24))
        self.assertEqual(geometry.noise_shape, (1, 3, 120, 416, 768))
        self.assertEqual(geometry.raw_shape, (1, 3, 120, 416, 768))
        self.assertEqual(geometry.first_window_shape, (1, 3, 40, 416, 768))
        self.assertEqual(geometry.token_grid, (10, 26, 48))
        self.assertEqual(geometry.raw_bytes, 460_062_720)

        plan, _, _ = oracle.block_reference.make_attention_plan(
            None, geometry.token_grid, tile=8
        )
        self.assertEqual(len(plan.regions), 24)
        bottom = [region for region in plan.regions if region.tile_h == 3]
        self.assertEqual(len(bottom), 6)
        self.assertTrue(all(region.query_stop - region.query_start == 160 for region in bottom))
        self.assertEqual((bottom[-1].tile_h, bottom[-1].tile_w), (3, 5))
        self.assertEqual(max(plan.permutation), 12_479)

    def test_receipt_resolution_must_match_v3_input_geometry(self):
        default_receipt = oracle.validate_native_receipt_document(
            self._receipt(), expected_inputs_sha256="a" * 64
        )
        high_manifest = self._manifest(oracle.HIGH_RESOLUTION)
        with self.assertRaisesRegex(oracle.SameLatentError, "resolution differs"):
            oracle._resolve_geometry(default_receipt, high_manifest)

        high_receipt = oracle.validate_native_receipt_document(
            self._receipt(oracle.HIGH_RESOLUTION), expected_inputs_sha256="a" * 64
        )
        with self.assertRaisesRegex(oracle.SameLatentError, "resolution differs"):
            oracle._resolve_geometry(high_receipt, self._manifest())

    def test_input_schema_and_partial_bottom_tile_must_match_exactly(self):
        manifest = self._manifest(oracle.HIGH_RESOLUTION)
        manifest["geometry"]["decoder_rope_grid"] = [10, 26, 47]
        with self.assertRaisesRegex(oracle.SameLatentError, "geometry differs"):
            oracle._validate_decoder_noise_geometry(manifest)

        manifest = self._manifest(oracle.HIGH_RESOLUTION)
        manifest["tensors"]["decoder.rope_cos"]["shape"] = [1, 12_220, 1, 128]
        with self.assertRaisesRegex(oracle.SameLatentError, "shape/dtype differs"):
            oracle._geometry_for_input_manifest(manifest)

        manifest = self._manifest(oracle.HIGH_RESOLUTION)
        manifest["schema"] = []
        with self.assertRaisesRegex(oracle.SameLatentError, "schema is malformed"):
            oracle._geometry_for_input_manifest(manifest)

    def test_receipt_requires_authenticated_top_level_display_output(self):
        receipt = self._receipt(oracle.HIGH_RESOLUTION)
        receipt["output_sha256"] = "malformed"
        with self.assertRaisesRegex(oracle.SameLatentError, "display output SHA-256"):
            oracle.validate_native_receipt_document(
                receipt, expected_inputs_sha256="a" * 64
            )
        receipt = self._receipt(oracle.HIGH_RESOLUTION)
        receipt["output_bytes"] -= 4
        with self.assertRaisesRegex(oracle.SameLatentError, "display output byte extent"):
            oracle.validate_native_receipt_document(
                receipt, expected_inputs_sha256="a" * 64
            )

    def test_display_payload_digest_is_checked_against_the_authenticated_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            display = Path(directory) / "pixels.f32le"
            display.write_bytes(b"display")
            digest = hashlib.sha256(b"display").hexdigest()
            self.assertEqual(
                oracle._regular_artifact(str(display), "native display pixels", digest, 7),
                display.resolve(),
            )
            display.write_bytes(b"changed")
            with self.assertRaisesRegex(oracle.SameLatentError, "SHA-256 mismatch"):
                oracle._regular_artifact(str(display), "native display pixels", digest, 7)

    def test_payload_digest_mismatch_is_rejected_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "small.tensors.bin"
            payload.write_bytes(b"valid")
            digest = hashlib.sha256(b"valid").hexdigest()
            document = {
                "tensors": {
                    "decoder.noise": {
                        "offset": 0, "nbytes": 5, "sha256": digest,
                    },
                },
                "payload_sha256": digest,
            }
            with (
                patch.object(oracle.generation_inputs, "validate_document"),
                patch.object(oracle.generation_inputs, "TENSOR_SPECS", (("decoder.noise", "F32", (1,)),)),
            ):
                oracle._validate_payload_file(document, payload)
                payload.write_bytes(b"other")
                with self.assertRaisesRegex(oracle.SameLatentError, "tensor digest mismatch"):
                    oracle._validate_payload_file(document, payload)

    def test_full_and_first_window_comparison_extents_are_exact_for_both_sizes(self):
        self.assertEqual(
            oracle._comparison_element_count(oracle.RAW_SHAPE, oracle.RAW_SHAPE),
            oracle.RAW_BYTES // 4,
        )
        high = oracle.HIGH_RESOLUTION_GEOMETRY
        self.assertEqual(
            oracle._comparison_element_count(
                high.raw_shape, high.raw_shape, geometry=high
            ),
            high.raw_bytes // 4,
        )
        self.assertEqual(
            oracle._comparison_element_count(
                high.first_window_shape, high.raw_shape,
                high.first_prefix_elements, geometry=high,
            ),
            high.first_prefix_elements,
        )
        with self.assertRaisesRegex(oracle.SameLatentError, "first-window comparison geometry"):
            oracle._comparison_element_count(
                (1, 3, 39, 416, 768), high.raw_shape,
                high.first_prefix_elements, geometry=high,
            )

    def test_first_window_prefix_slices_time_before_flattening(self):
        actual = oracle.torch.arange(8).reshape(1, 2, 4, 1, 1)
        expected = oracle.torch.arange(10).reshape(1, 2, 5, 1, 1)
        actual_prefix, expected_prefix = oracle._first_prefix_tensors(
            actual, expected, frames=2
        )
        self.assertEqual(actual_prefix.flatten().tolist(), [0, 1, 4, 5])
        self.assertEqual(expected_prefix.flatten().tolist(), [0, 1, 5, 6])
        self.assertNotEqual(actual_prefix.flatten().tolist(), actual.flatten()[:4].tolist())

    def test_output_admission_reserves_display_path_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "results"
            output_dir.mkdir()
            candidate = output_dir / "decoder.json"
            with patch.object(oracle, "SCRATCH_ROOT", root):
                paths = oracle._output_paths(candidate)
                self.assertEqual(paths[2].name, "decoder-first-window.f32le")
                self.assertEqual(paths[3].name, "decoder-display.f32le")
                paths[3].touch()
                with self.assertRaisesRegex(oracle.SameLatentError, "refusing to replace"):
                    oracle._output_paths(candidate)

    def test_non_admission_requires_output_before_artifact_preflight(self):
        with self._stubbed_main() as (args, artifact_preflight, output_preflight, runtime_preflight):
            with redirect_stderr(io.StringIO()):
                status = oracle.main(args)
        self.assertEqual(status, 2)
        output_preflight.assert_not_called()
        runtime_preflight.assert_not_called()
        artifact_preflight.assert_not_called()

    def test_non_admission_output_failure_precedes_artifact_preflight(self):
        with self._stubbed_main() as (args, artifact_preflight, output_preflight, runtime_preflight):
            args.extend(("--output", "/tmp/not-used.json"))
            output_preflight.side_effect = oracle.SameLatentError("invalid output root")
            with redirect_stderr(io.StringIO()):
                status = oracle.main(args)
        self.assertEqual(status, 2)
        output_preflight.assert_called_once_with(Path("/tmp/not-used.json"))
        runtime_preflight.assert_not_called()
        artifact_preflight.assert_not_called()

    def test_non_admission_runtime_failure_precedes_artifact_preflight(self):
        with self._stubbed_main() as (args, artifact_preflight, output_preflight, runtime_preflight):
            args.extend(("--output", "/tmp/valid.json"))
            runtime_preflight.side_effect = oracle.SameLatentError("wrong pinned Torch runtime")
            with redirect_stderr(io.StringIO()):
                status = oracle.main(args)
        self.assertEqual(status, 2)
        output_preflight.assert_called_once_with(Path("/tmp/valid.json"))
        runtime_preflight.assert_called_once_with(oracle.DEFAULT_GEOMETRY)
        artifact_preflight.assert_not_called()

    def test_admission_only_skips_output_and_mps_runtime_preflight(self):
        with self._stubbed_main() as (args, artifact_preflight, output_preflight, runtime_preflight):
            args.append("--admission-only")
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                status = oracle.main(args)
        self.assertEqual(status, 0)
        output_preflight.assert_not_called()
        runtime_preflight.assert_not_called()
        artifact_preflight.assert_called_once()

    def test_high_resolution_runtime_rejects_unpinned_torch_before_mps_check(self):
        with (
            patch.object(oracle.torch, "__version__", "2.8.0"),
            patch.object(oracle.torch.backends.mps, "is_available") as mps_available,
        ):
            with self.assertRaisesRegex(oracle.SameLatentError, "requires torch 2.6.0"):
                oracle._validate_execution_runtime(oracle.HIGH_RESOLUTION_GEOMETRY)
        mps_available.assert_not_called()

    def test_canonical_input_path_rejects_symlink_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "input.json"
            target.write_text("{}", encoding="utf-8")
            alias = root / "alias.json"
            alias.symlink_to(target)
            self.assertEqual(oracle._canonical_file(target, "test input"), target.resolve())
            with self.assertRaisesRegex(oracle.SameLatentError, "non-symlink"):
                oracle._canonical_file(alias, "test input")

    def test_cpu_artifact_preflight_authenticates_raw_and_display_before_return(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"

            def regular(path_value, what, *_):
                events.append(what)
                return Path(path_value)

            with (
                patch.object(oracle, "_validate_upstream_source_metadata",
                             side_effect=lambda: events.append("source") or {"revision": "pinned"}),
                patch.object(oracle, "_regular_artifact", side_effect=regular),
                patch.object(oracle, "_authenticate_stem",
                             side_effect=lambda: events.append("stem") or (b"header", path, path)),
                patch.object(oracle, "_authenticate_head",
                             side_effect=lambda _: events.append("head") or {"subset_path": path}),
                patch.object(oracle, "_authenticate_full36_cache",
                             side_effect=lambda: events.append("cache") or ({}, path, path, "d" * 64, "e" * 64)),
                patch.object(oracle, "_canonical_file", side_effect=lambda p, _: Path(p)),
            ):
                admitted = oracle._preflight_artifacts(
                    receipt={"inputs_sha256": "a" * 64},
                    diagnostics={
                        "latent": str(path), "latent_sha256": "b" * 64,
                        "raw_pixels": str(path), "raw_pixels_sha256": "c" * 64,
                        "display_pixels": str(path), "display_pixels_sha256": "f" * 64,
                    },
                    input_document={"payload_sha256": "e" * 64},
                    input_payload_path=path,
                )
        self.assertEqual(
            events,
            ["source", "native latent", "native raw pixels", "native display pixels",
             "stem", "head", "cache"],
        )
        self.assertEqual(admitted["native_display_path"], path)
        self.assertEqual(admitted["source_metadata"]["revision"], "pinned")

    @unittest.skipUnless(
        (oracle.checkpoint_inventory.ROOT / "source" / "pixelumm-header.json").is_file()
        and (oracle.head_reference.SOURCE_DIR / "head-f32.safetensors").is_file(),
        "pinned checkpoint artifacts are not mounted",
    )
    def test_current_pinned_source_stem_and_head_headers_without_model(self):
        source_metadata = oracle._validate_upstream_source_metadata()
        raw_header, stem_path, _ = oracle._authenticate_stem()
        head = oracle._authenticate_head(raw_header)
        self.assertEqual(source_metadata["revision"], oracle.block_reference.UPSTREAM_REVISION)
        self.assertTrue(stem_path.is_absolute())
        self.assertTrue(head["subset_path"].is_absolute())
        self.assertTrue(head["receipt_path"].is_absolute())

    def test_memory_preflight_counts_raw_and_display_comparison_buffers(self):
        gib = oracle.GIB
        bound = oracle._comparison_buffer_bound_bytes(oracle.HIGH_RESOLUTION_GEOMETRY)
        admitted = oracle._memory_budget(
            physical_bytes=64 * gib,
            driver_allocated_bytes=4 * gib,
            reference_peak_bytes=5 * gib,
            cache_resident_allowance_bytes=13 * gib,
            comparison_buffer_bytes=bound,
        )
        self.assertEqual(admitted["raw_display_comparison_buffer_bound_bytes"], bound)
        self.assertEqual(
            admitted["estimated_peak_bytes"],
            5 * gib + 13 * gib + bound + oracle.EXTRA_GPU_RESERVE_BYTES,
        )
        self.assertEqual(admitted["required_reserve_bytes"], (64 * gib) // 10)
        with self.assertRaisesRegex(oracle.SameLatentError, "10% physical-memory reserve"):
            oracle._memory_budget(
                physical_bytes=20 * gib,
                driver_allocated_bytes=4 * gib,
                reference_peak_bytes=5 * gib,
                cache_resident_allowance_bytes=10 * gib,
                comparison_buffer_bytes=bound,
            )


if __name__ == "__main__":
    unittest.main()
