"""CPU-only admission and geometry tests for the same-latent decoder oracle."""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_decoder_same_latent as oracle


class SameLatentAdmissionTests(unittest.TestCase):
    def _receipt(self) -> dict:
        return {
            "schema": oracle.NATIVE_SCHEMA,
            "upstream_revision": oracle.block_reference.UPSTREAM_REVISION,
            "precision": oracle.NATIVE_PRECISION,
            "dit_weights": "bf16",
            "dit_quantization_experimental": False,
            "seed": 42,
            "frames": 120,
            "width": 512,
            "height": 288,
            "fps": 24,
            "steps": 4,
            "shift": 5,
            "cfg": 1,
            "camera_conditioning": False,
            "action_conditioning": False,
            "prompt": "authenticated prompt",
            "inputs_sha256": "a" * 64,
            "inputs": "/tmp/inputs.json",
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

    def test_decoder_input_geometry_and_noise_shape_are_pinned(self):
        manifest = {
            "geometry": {
                "dit_latent_shape": list(oracle.LATENT_SHAPE),
                "decoder_noise_shape": list(oracle.NOISE_SHAPE),
                "decoder_rope_grid": list(oracle.GRID),
                "decoder_local_attention_tile_pixels": 128,
                "decoder_local_attention_tile_tokens": 8,
                "decoder_noise_piece_frames": 4,
                "decoder_noise_piece_latents": 1,
            },
            "tensors": {
                "decoder.noise": {
                    "dtype": "F32",
                    "shape": list(oracle.NOISE_SHAPE),
                    "nbytes": oracle.NOISE_BYTES,
                },
            },
        }
        self.assertEqual(
            oracle._validate_decoder_noise_geometry(manifest),
            manifest["tensors"]["decoder.noise"],
        )
        malformed = {**manifest, "tensors": dict(manifest["tensors"])}
        malformed["tensors"]["decoder.noise"] = {
            **manifest["tensors"]["decoder.noise"],
            "shape": [1, 3, 120, 288, 511],
        }
        with self.assertRaisesRegex(oracle.SameLatentError, "noise tensor shape/dtype"):
            oracle._validate_decoder_noise_geometry(malformed)

    def test_full_and_first_window_comparison_extents_are_exact(self):
        self.assertEqual(
            oracle._comparison_element_count(oracle.RAW_SHAPE, oracle.RAW_SHAPE),
            oracle.RAW_BYTES // 4,
        )
        self.assertEqual(
            oracle._comparison_element_count(
                oracle.FIRST_WINDOW_SHAPE, oracle.RAW_SHAPE,
                oracle.FIRST_PREFIX_ELEMENTS,
            ),
            oracle.FIRST_PREFIX_ELEMENTS,
        )
        with self.assertRaisesRegex(oracle.SameLatentError, "first-window comparison geometry"):
            oracle._comparison_element_count(
                oracle.FIRST_WINDOW_SHAPE, oracle.RAW_SHAPE,
                oracle.FIRST_PREFIX_ELEMENTS - 1,
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

    def test_output_admission_reserves_optional_first_window_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "results"
            output_dir.mkdir()
            candidate = output_dir / "decoder.json"
            with patch.object(oracle, "SCRATCH_ROOT", root):
                paths = oracle._output_paths(candidate)
                self.assertEqual(paths[2].name, "decoder-first-window.f32le")
                paths[2].touch()
                with self.assertRaisesRegex(oracle.SameLatentError, "refusing to replace"):
                    oracle._output_paths(candidate)

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

    def test_cpu_artifact_preflight_orders_all_provenance_before_return(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            with (
                patch.object(oracle, "_validate_upstream_source_metadata",
                             side_effect=lambda: events.append("source") or {"revision": "pinned"}),
                patch.object(oracle, "_regular_artifact",
                             side_effect=lambda *args: events.append("native artifact") or path),
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
                    diagnostics={"latent": str(path), "latent_sha256": "b" * 64,
                                 "raw_pixels": str(path), "raw_pixels_sha256": "c" * 64},
                    input_document={"payload_sha256": "e" * 64},
                    input_payload_path=path,
                )
        self.assertEqual(events, ["source", "native artifact", "native artifact", "stem", "head", "cache"])
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

    def test_memory_preflight_counts_mapped_cache_before_reserve(self):
        gib = oracle.GIB
        admitted = oracle._memory_budget(
            physical_bytes=64 * gib,
            driver_allocated_bytes=4 * gib,
            reference_peak_bytes=5 * gib,
            cache_resident_allowance_bytes=13 * gib,
        )
        self.assertEqual(admitted["estimated_peak_bytes"], 5 * gib + 13 * gib + oracle.EXTRA_GPU_RESERVE_BYTES)
        self.assertEqual(admitted["required_reserve_bytes"], (64 * gib) // 10)
        with self.assertRaisesRegex(oracle.SameLatentError, "10% physical-memory reserve"):
            oracle._memory_budget(
                physical_bytes=20 * gib,
                driver_allocated_bytes=4 * gib,
                reference_peak_bytes=5 * gib,
                cache_resident_allowance_bytes=10 * gib,
            )


if __name__ == "__main__":
    unittest.main()
