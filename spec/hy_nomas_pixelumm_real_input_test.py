"""CPU-only contract and admission checks for the real-input producer."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_pixelumm_real_input as producer


class PixelUMMRealInputProducerTest(unittest.TestCase):
    def test_only_two_admitted_profiles_and_seeds_are_selectable(self):
        full = producer.PROFILES["full"]
        self.assertEqual(full.grid, (6, 44, 80))
        self.assertEqual(full.frames, 24)
        self.assertEqual(full.crop, (0, 8, 1280, 712))
        self.assertEqual(full.latent_shape, (1, 32, 6, 22, 40))
        self.assertEqual(full.xt_shape, (1, 3, 24, 704, 1280))

        small = producer.PROFILES["small"]
        self.assertEqual(small.grid, (2, 32, 32))
        self.assertEqual(small.frames, 8)
        self.assertEqual(small.crop, (384, 104, 896, 616))
        self.assertEqual(small.latent_shape, (1, 32, 2, 16, 16))
        self.assertEqual(small.xt_shape, (1, 3, 8, 512, 512))

        for profile in ("full", "small"):
            for seed in (42, 43):
                self.assertEqual(
                    producer.payload_basename(producer.manifest_basename(profile, seed)),
                    producer.manifest_basename(profile, seed).removesuffix(".json") + ".bin",
                )
        self.assertEqual(producer.manifest_basename("full", 42), "pixelumm-real-input.json")
        with self.assertRaises(producer.RealInputError):
            producer.manifest_basename("small", 44)
        with self.assertRaises(producer.RealInputError):
            producer.manifest_basename("wide", 42)

    def test_pair_payload_extents_are_exact_and_contiguous_by_contract(self):
        full = producer.PROFILES["full"]
        extents = (
            producer.expected_tensor_bytes("BF16", full.latent_shape),
            producer.expected_tensor_bytes("F32", full.xt_shape),
            producer.expected_tensor_bytes("BF16", full.x_shape),
        )
        self.assertEqual(extents, (337_920, 259_522_560, 173_015_040))
        self.assertEqual(sum(extents), 432_875_520)

        small = producer.PROFILES["small"]
        small_extents = (
            producer.expected_tensor_bytes("BF16", small.latent_shape),
            producer.expected_tensor_bytes("F32", small.xt_shape),
            producer.expected_tensor_bytes("BF16", small.x_shape),
        )
        self.assertEqual(small_extents, (32_768, 25_165_824, 16_777_216))
        self.assertEqual(sum(small_extents), 41_975_808)

    def test_manifest_profile_metadata_binds_profile_and_noise_seed(self):
        metadata = producer._manifest_profile_metadata(producer.PROFILES["small"], 43)
        self.assertEqual(metadata, {
            "schema": "cogni_ml.pixelumm_real_input/v1",
            "profile": "small",
            "noise_seed": 43,
            "grid": [2, 32, 32],
            "latent_shape": [1, 32, 2, 16, 16],
            "xt_shape": [1, 3, 8, 512, 512],
            "x_shape": [1, 2048, 4096],
            "token_order": "tile_major",
        })
        with self.assertRaises(producer.RealInputError):
            producer._manifest_profile_metadata(producer.PROFILES["full"], True)

    def test_static_memory_estimate_fits_each_default_allocator_cap(self):
        for profile in producer.PROFILES.values():
            estimate = producer.estimate_mps_peak_bytes(profile)
            self.assertGreater(estimate, producer.ENCODER_F32_BYTES // 2)
            self.assertLessEqual(estimate, profile.default_mps_bytes)

    def test_memory_preflight_enforces_allocator_cap_and_ten_percent_floor(self):
        gib = producer.GIB
        fresh = producer.memory_preflight(
            physical_bytes=64 * gib,
            driver_allocated_bytes=0,
            recommended_max_bytes=52 * gib,
            estimated_peak_bytes=18 * gib,
            max_mps_bytes=24 * gib,
        )
        self.assertEqual(fresh["driver_allocated_bytes"], 0)
        admitted = producer.memory_preflight(
            physical_bytes=64 * gib,
            driver_allocated_bytes=4 * gib,
            recommended_max_bytes=52 * gib,
            estimated_peak_bytes=18 * gib,
            max_mps_bytes=24 * gib,
        )
        self.assertEqual(admitted["required_floor_bytes"], (64 * gib) // 10)
        self.assertEqual(admitted["mps_memory_fraction"], 24 / 52)
        with self.assertRaisesRegex(producer.RealInputError, "floor"):
            producer.memory_preflight(
                physical_bytes=64 * gib,
                driver_allocated_bytes=36 * gib,
                recommended_max_bytes=52 * gib,
                estimated_peak_bytes=18 * gib,
                max_mps_bytes=24 * gib,
            )
        with self.assertRaisesRegex(producer.RealInputError, "estimated MPS peak"):
            producer.memory_preflight(
                physical_bytes=64 * gib,
                driver_allocated_bytes=4 * gib,
                recommended_max_bytes=52 * gib,
                estimated_peak_bytes=25 * gib,
                max_mps_bytes=24 * gib,
            )
        with self.assertRaisesRegex(producer.RealInputError, "driver allocated bytes"):
            producer.memory_preflight(
                physical_bytes=64 * gib,
                driver_allocated_bytes=-1,
                recommended_max_bytes=52 * gib,
                estimated_peak_bytes=18 * gib,
                max_mps_bytes=24 * gib,
            )

    def test_named_subset_compares_payload_bytes_but_records_full_file_bytes(self):
        payload_bytes = 3_666_620_032
        file_bytes = payload_bytes + 1_328
        self.assertEqual(
            producer._named_subset_file_bytes({
                "selected_subset": "encoder",
                "source_f32_bytes": payload_bytes,
                "subset_bytes": file_bytes,
            }, "encoder", payload_bytes),
            file_bytes,
        )
        with self.assertRaisesRegex(producer.RealInputError, "F32 payload extent"):
            producer._named_subset_file_bytes({
                "selected_subset": "encoder",
                "source_f32_bytes": payload_bytes - 4,
                "subset_bytes": file_bytes,
            }, "encoder", payload_bytes)

    def test_stem_raw_hash_contract_excludes_safetensors_header(self):
        self.assertEqual(
            producer.STEM_RAW_KEYS,
            (
                "decoder.cond_proj.bias", "decoder.cond_proj.weight",
                "decoder.embed.bias", "decoder.embed.weight",
            ),
        )
        self.assertEqual(producer.REAL_INPUT_SCHEMA, "cogni_ml.pixelumm_real_input/v1")
        self.assertEqual(producer.UPSTREAM_REVISION, "911cea6458a89c2663513ff7099561ec190dd491")

    def test_pinned_upstream_noise_method_owner_and_lineage(self):
        _, _, noise_source = producer._load_modules(producer.UPSTREAM_ROOT)
        owner = getattr(noise_source, "PixelUMMDCAELatentDecoder")
        self.assertTrue(callable(getattr(owner, "_make_noise", None)))

        digest = "a" * 64
        lineage = producer._lineage(
            producer.PROFILES["small"], 42,
            {
                "dcae_encoder_source_sha256": digest,
                "pixelumm_model_source_sha256": digest,
                "pixelumm_noise_source_sha256": digest,
            },
            {
                "encoder_subset_sha256": digest,
                "encoder_receipt_sha256": digest,
                "encoder_subset_bytes": 3_666_621_360,
                "stem_subset_sha256": digest,
                "stem_receipt_sha256": digest,
                "stem_subset_bytes": 52_512_048,
                "stem_weights_sha256": digest,
            },
            digest, digest,
        )
        self.assertEqual(
            lineage["xt_noise_source"],
            "hymm.models.vae.pixelumm_decoder.autoencoder."
            "PixelUMMDCAELatentDecoder._make_noise",
        )

    def test_terminal_cpu_payload_readback_rejects_nonfinite_latent_xt_and_x(self):
        for name, dtype in (
            ("latent", producer.torch.bfloat16),
            ("xt", producer.torch.float32),
            ("x", producer.torch.bfloat16),
        ):
            finite = producer.torch.tensor([1.0, -2.0], dtype=dtype)
            chunks = list(producer._tensor_raw_chunks(finite, name))
            self.assertEqual(sum(chunk.nbytes for chunk in chunks), finite.numel() * finite.element_size())

            invalid = producer.torch.tensor([1.0, float("nan")], dtype=dtype)
            with self.subTest(tensor=name):
                with self.assertRaisesRegex(
                    producer.RealInputError, f"{name} tensor contains non-finite values"
                ):
                    list(producer._tensor_raw_chunks(invalid, name))


if __name__ == "__main__":
    unittest.main()
