"""CPU admission and state-schedule falsifiers for the two-segment control."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_two_segment_quality_control as control


def writer_contract() -> dict:
    return {
        "cache_at_step": -1,
        "cache_noise_sigma": control.WRITER_SIGMA,
        "cache_timestep": control.WRITER_TIMESTEP,
        "cache_latent_source": "final denoised segment latent selected by cache_at_step=-1",
        "cache_write_formula": "((1-sigma)*cache_latents.float()+sigma*cache_eps.float()).to(cache_latents.dtype)",
        "cache_eps_tensor": "cache.eps_bf16",
        "cache_eps_dtype": "BF16",
        "cache_eps_seed": 42,
        "cache_eps_generator_device": "mps",
        "cache_eps_sample_dtype": "F32 then cast to BF16",
        "cache_eps_upstream_cpu_rng_parity": False,
        "cache_eps_draws": [
            {
                "segment_index": index,
                "draw_order": index,
                "seed": 42,
                "generator_device": "mps",
                "global_latent_start": index * control.SEGMENT_LATENTS,
                "global_latent_stop": (index + 1) * control.SEGMENT_LATENTS,
                "shape": [1, 32, control.SEGMENT_LATENTS, 9, 16],
                "dtype": "BF16",
            }
            for index in range(6)
        ],
    }


class TwoSegmentStateTests(unittest.TestCase):
    def test_reusable_f32_block_gets_cache_adapter_in_its_fresh_namespace(self):
        def make_block_type(name):
            namespace = {"__name__": name}
            exec(
                "def forward_with_cache(self):\n"
                "    return causal_attention_with_cache_mp\n",
                namespace,
            )
            return type(
                name,
                (),
                {"forward_with_cache": namespace["forward_with_cache"]},
            )

        source_block = make_block_type("source_block")()
        reusable_f32_block = make_block_type("reusable_f32_block")()
        self.assertIsNot(
            source_block.forward_with_cache.__globals__,
            reusable_f32_block.forward_with_cache.__globals__,
        )
        with self.assertRaises(NameError):
            reusable_f32_block.forward_with_cache()

        cache_function = object()
        control._bind_runtime_cache_functions(
            [source_block], reusable_f32_block, cache_function
        )
        self.assertIs(source_block.forward_with_cache(), cache_function)
        self.assertIs(reusable_f32_block.forward_with_cache(), cache_function)

    def test_writer_noisy_latent_is_bf16_rounded_in_both_modes(self):
        class FakeTorch:
            bfloat16 = "bf16"
            float32 = "f32"

        class FakeTensor:
            def __init__(self, value, dtype, calls):
                self.value = value
                self.dtype = dtype
                self.calls = calls

            def to(self, dtype):
                self.calls.append(dtype)
                if dtype == FakeTorch.bfloat16:
                    # Coarse fake quantization makes the required boundary visible.
                    value = round(self.value * 4) / 4
                else:
                    value = self.value
                return FakeTensor(value, dtype, self.calls)

        clean = FakeTensor(0.375, FakeTorch.float32, [])
        noisy = FakeTensor(0.62, FakeTorch.float32, [])

        source_calls = []
        source_input = FakeTensor(noisy.value, noisy.dtype, source_calls)
        source_output = control._round_cache_writer_input(
            FakeTorch, source_input, FakeTorch.bfloat16
        )
        self.assertEqual(source_output.dtype, FakeTorch.bfloat16)
        self.assertEqual(source_output.value, 0.5)
        self.assertEqual(source_calls, [FakeTorch.bfloat16])

        native_calls = []
        native_input = FakeTensor(noisy.value, noisy.dtype, native_calls)
        native_output = control._round_cache_writer_input(
            FakeTorch, native_input, FakeTorch.float32
        )
        self.assertEqual(native_output.dtype, FakeTorch.float32)
        self.assertEqual(native_output.value, 0.5)
        self.assertEqual(native_calls, [FakeTorch.bfloat16, FakeTorch.float32])

        # The helper rounds only the computed noised value; the clean segment
        # remains available unchanged for terminal output and timeline scatter.
        self.assertEqual((clean.value, clean.dtype), (0.375, FakeTorch.float32))
        self.assertEqual((noisy.value, noisy.dtype), (0.62, FakeTorch.float32))

    def test_writer_rounding_rejects_unadmitted_dtype(self):
        class FakeTorch:
            bfloat16 = "bf16"
            float32 = "f32"

        class FakeTensor:
            def to(self, dtype):
                return self

        with self.assertRaisesRegex(control.ControlError, "output dtype"):
            control._round_cache_writer_input(FakeTorch, FakeTensor(), "f64")

    def test_model_control_runs_inside_inference_mode_before_impl(self):
        state = {"active": False}

        class FakeInferenceMode:
            def __enter__(self):
                state["active"] = True

            def __exit__(self, exc_type, exc_value, traceback):
                state["active"] = False

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        def implementation(*args):
            self.assertTrue(state["active"])
            return {"status": "inference"}, b""

        with patch.object(control, "_run_control_impl", implementation):
            result = control._run_control({}, object(), FakeTorch, object(), "test")
        self.assertEqual(result, ({"status": "inference"}, b""))
        self.assertFalse(state["active"])

    def test_exact_forward_plan_is_nine_calls_with_text_cached_once(self):
        plan = control._forward_plan()
        control._validate_forward_plan(plan)
        self.assertEqual(len(plan), 9)
        self.assertEqual(
            [call["kind"] for call in plan],
            ["denoise"] * 4 + ["writer"] + ["denoise"] * 4,
        )
        self.assertTrue(all(call["cache_txt"] for call in plan[:5]))
        self.assertTrue(all(not call["cache_txt"] for call in plan[5:]))
        self.assertEqual([call["rope_bank"] for call in plan], [0] * 5 + [1] * 4)
        self.assertTrue(
            all(
                call["cache_valid_len"] == control.CACHE_ROWS_PER_SOURCE_CHUNK
                for call in plan[5:]
            )
        )
        self.assertEqual(sum(call["update_cache"] for call in plan), 1)

    def test_plan_rejects_second_text_write_or_wrong_history_offset(self):
        plan = control._forward_plan()
        plan[-1]["cache_txt"] = True
        with self.assertRaisesRegex(control.ControlError, "schedule"):
            control._validate_forward_plan(plan)
        plan = control._forward_plan()
        plan[5]["cache_valid_len"] -= 19
        with self.assertRaisesRegex(control.ControlError, "schedule"):
            control._validate_forward_plan(plan)

    def test_geometry_preserves_padded_masked_cache_and_60_latent_output(self):
        geometry = control._geometry()
        self.assertEqual(geometry["terminal_latent_shape"], [1, 32, 60, 9, 16])
        self.assertEqual(geometry["upstream_cache_writer_rows"], 4896)
        self.assertEqual(geometry["upstream_cache_writer_valid_rows"], 4877)
        self.assertTrue(geometry["source_padded_cache_rows_are_masked"])
        self.assertEqual(geometry["upstream_cache_capacity_rows"], 4896)
        self.assertEqual(geometry["maximum_attention_key_rows"], 9216)
        self.assertEqual(
            geometry["persistent_cache_storage_bytes"]["bf16_total_bytes"],
            2_166_098_112,
        )
        self.assertEqual(
            geometry["persistent_cache_storage_bytes"]["f32_total_bytes"],
            4_331_931_840,
        )

    def test_history_and_current_attention_extent_are_not_cache_capacity(self):
        self.assertEqual(control._expected_attention_key_rows(0, True), 4896)
        self.assertEqual(control._expected_attention_key_rows(4896, False), 9216)
        for invalid in (-1, 1, 4877, 9792):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(control.ControlError, "history length"):
                    control._expected_attention_key_rows(invalid, False)
        with self.assertRaisesRegex(control.ControlError, "state types"):
            control._expected_attention_key_rows(4896, 0)

    def test_allocated_cache_and_mask_use_one_written_segment(self):
        class FakeTorch:
            bool = object()

            @staticmethod
            def zeros(shape, dtype, device):
                return {"shape": tuple(shape), "dtype": dtype, "device": device}

        caches, masks = control._new_caches(FakeTorch, "f32", "cpu")
        self.assertEqual(len(caches), 54)
        self.assertEqual(len(masks), 54)
        self.assertEqual(caches[0]["k"]["shape"], (1, 4896, 16, 128))
        self.assertEqual(caches[0]["v"]["shape"], (1, 4896, 16, 128))
        self.assertEqual(masks[0]["shape"], (1, 4896))


class TwoSegmentArtifactAdmissionTests(unittest.TestCase):
    def test_writer_contract_requires_final_segment_latent_and_saved_epsilon(self):
        contract = writer_contract()
        self.assertIs(control._validate_writer_contract(contract), contract)
        contract["cache_timestep"] += 1
        with self.assertRaisesRegex(control.ControlError, "writer metadata"):
            control._validate_writer_contract(contract)

    def test_writer_contract_requires_six_ordered_epsilon_draws(self):
        contract = writer_contract()
        contract["cache_eps_draws"][1]["global_latent_start"] = 31
        with self.assertRaisesRegex(control.ControlError, "draw order/geometry"):
            control._validate_writer_contract(contract)

    def test_conditioning_hashes_temporal_latent_segments_in_channel_major_order(self):
        latent_bytes = 1 * 32 * 180 * 9 * 16 * 4
        rope_bank_bytes = control.VIDEO_TOKENS * control.HEAD_DIM * 4
        eps_bank_bytes = 1 * 32 * 30 * 9 * 16 * 2
        offsets = {
            "input.latent_bank": 0,
            "input.rope_cos": latent_bytes,
            "input.rope_sin": latent_bytes + 6 * rope_bank_bytes,
            "cache.eps_bf16": latent_bytes + 12 * rope_bank_bytes,
        }
        total_bytes = offsets["cache.eps_bf16"] + 6 * eps_bank_bytes
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "zeros.bin"
            with payload.open("wb") as stream:
                stream.truncate(total_bytes)
            records = {
                "input.latent_bank": {
                    "dtype": "F32",
                    "shape": [1, 32, 180, 9, 16],
                    "offset": offsets["input.latent_bank"],
                    "sha256": "latent-record",
                },
                "input.rope_cos": {
                    "dtype": "F32",
                    "shape": [6, control.VIDEO_TOKENS, control.HEAD_DIM],
                    "offset": offsets["input.rope_cos"],
                    "sha256": "cos-record",
                },
                "input.rope_sin": {
                    "dtype": "F32",
                    "shape": [6, control.VIDEO_TOKENS, control.HEAD_DIM],
                    "offset": offsets["input.rope_sin"],
                    "sha256": "sin-record",
                },
                "cache.eps_bf16": {
                    "dtype": "BF16",
                    "shape": [6, 1, 32, 30, 9, 16],
                    "offset": offsets["cache.eps_bf16"],
                    "sha256": "eps-record",
                },
            }
            admitted = {
                "conditioning": {"tensors": records, "cache_writer": writer_contract()},
                "base_qwen": {"valid_text_mask": [True] * 557 + [False] * (1280 - 557)},
                "conditioning_payload": str(payload),
            }
            identities = control._validate_conditioning_records(admitted)
        expected_latent = hashlib.sha256(bytes(32 * 30 * 9 * 16 * 4)).hexdigest()
        self.assertEqual(identities["latent_segment_sha256"], [expected_latent] * 2)
        self.assertEqual(len(identities["rope_cos_bank_sha256"]), 2)
        self.assertEqual(
            identities["cache_epsilon_segment_sha256"],
            [hashlib.sha256(bytes(eps_bank_bytes)).hexdigest()] * 2,
        )

    def test_conditioning_admission_rejects_nonprefix_text_mask(self):
        mask = [True] * 556 + [False] + [True] + [False] * (1280 - 558)
        with self.assertRaisesRegex(control.ControlError, "text mask"):
            control._validate_text_mask(mask)


class PinnedCacheBodySmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        control.single.DEFAULT_UPSTREAM.is_dir(),
        "pinned upstream checkout is required for cache-body CPU smoke",
    )
    def test_pinned_no_cache_writer_reader_body_executes_in_both_layouts(self):
        torch = importlib.import_module("torch")
        upstream = control.single.DEFAULT_UPSTREAM
        control.block_reference._verify_upstream(upstream)

        expected_physical_rows = [4, 4, 6]
        expected_masks = [
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False, True, True],
        ]
        for layout, expected_sdpa_rows in (
            ("padded", [4, 4, 6]),
            ("compact", [3, 3, 5]),
        ):
            with self.subTest(layout=layout):
                receipt = control._validate_pinned_cache_contracts(
                    torch, upstream, "native-f32", layout
                )

                self.assertTrue(receipt["executed"])
                self.assertEqual(receipt["device"], "cpu")
                self.assertTrue(receipt["block_forward_bound_to_cache_function"])
                self.assertEqual(receipt["cache_function_package"], "hymm.models.dit")
                self.assertEqual(receipt["branches"], ["no_cache", "writer", "reader"])
                self.assertEqual(receipt["attention_calls"], 3)
                self.assertEqual(receipt["attention_layout"], layout)
                self.assertEqual(receipt["observed_key_rows"], expected_physical_rows)
                self.assertEqual(receipt["observed_sdpa_key_rows"], expected_sdpa_rows)
                self.assertEqual(receipt["observed_kv_masks"], expected_masks)
                self.assertEqual(
                    receipt["attention_row_inventory"],
                    [
                        {
                            "physical_key_rows": 4,
                            "sdpa_key_rows": expected_sdpa_rows[0],
                            "calls": 2,
                        },
                        {
                            "physical_key_rows": 6,
                            "sdpa_key_rows": expected_sdpa_rows[2],
                            "calls": 1,
                        },
                    ],
                )

    def test_padded_and_compact_dispatch_are_equivalent_on_tiny_cpu_geometry(self):
        torch = importlib.import_module("torch")
        geometry = control.CPU_CACHE_SMOKE_GEOMETRY
        q = torch.arange(1, 1 + 4 * 2 * 4, dtype=torch.float32).reshape(1, 4, 2, 4) / 17
        k = torch.arange(1, 1 + 6 * 2 * 4, dtype=torch.float32).reshape(1, 6, 2, 4) / 13
        v = torch.arange(1, 1 + 6 * 2 * 4, dtype=torch.float32).reshape(1, 6, 2, 4) / 7
        # Give masked padding large values so a mistaken retained row cannot
        # pass by being numerically negligible.
        k[:, 3] = 1000
        v[:, 3] = -1000
        q_mask = torch.ones((1, 4), dtype=torch.bool)

        for cache_valid_len, cache_txt, update_cache, kv_mask in (
            (0, True, False, [True, True, True, False]),
            (4, False, False, [True, True, True, False, True, True]),
        ):
            outputs = {}
            physical_rows = 4 if cache_valid_len == 0 else 6
            for layout in ("padded", "compact"):
                state = {
                    "dtype": torch.float32,
                    "device_type": "cpu",
                    "cpu_smoke_geometry": geometry,
                    "attention_layout": layout,
                    "call": {
                        "cache_valid_len": cache_valid_len,
                        "update_cache": update_cache,
                        "cache_txt": cache_txt,
                    },
                    "receipt": {
                        "attention_calls": 0,
                        "attention_query_chunks": 0,
                    },
                }
                dispatch = control._cache_attention_dispatch_factory(torch, state)
                outputs[layout] = dispatch(
                    q,
                    k[:, :physical_rows],
                    v[:, :physical_rows],
                    q_mask,
                    torch.tensor([kv_mask], dtype=torch.bool),
                    cache_valid_len,
                    update_cache,
                    cache_txt,
                )
            torch.testing.assert_close(
                outputs["compact"], outputs["padded"], rtol=1e-6, atol=1e-6
            )


class AttentionLayoutAdmissionTests(unittest.TestCase):
    def test_attention_layout_defaults_to_legacy_padded_receipt(self):
        self.assertEqual(control.QUERY_CHUNK, 256)
        self.assertEqual(
            control._parse_args(["--validate-only"]).attention_layout, "padded"
        )
        self.assertEqual(
            control._parse_args(
                ["--validate-only", "--attention-layout", "compact"]
            ).attention_layout,
            "compact",
        )

    def test_attention_layout_rejects_unadmitted_value(self):
        with self.assertRaises(SystemExit):
            control._parse_args(["--validate-only", "--attention-layout", "masked"])


if __name__ == "__main__":
    unittest.main()
