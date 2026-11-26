"""CPU-only contracts for the source T0 preblock trace wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_reference_preblock_control as control


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type="mps")

    def to(self, *, dtype: str):
        return _FakeTensor(self.shape, dtype)


class _HookHandle:
    def __init__(self, remove):
        self._remove = remove

    def remove(self):
        self._remove()


class _HookModule:
    def __init__(self):
        self.pre_hooks = []
        self.post_hooks = []

    def register_forward_pre_hook(self, hook):
        self.pre_hooks.append(hook)
        return _HookHandle(lambda: self.pre_hooks.remove(hook))

    def register_forward_hook(self, hook):
        self.post_hooks.append(hook)
        return _HookHandle(lambda: self.post_hooks.remove(hook))

    def __call__(self, *args):
        for hook in tuple(self.pre_hooks):
            replacement = hook(self, args)
            if replacement is not None:
                args = tuple(replacement)
        output = args[0] if args else object()
        for hook in tuple(self.post_hooks):
            hook(self, args, output)
        return output


class PreblockLayoutTests(unittest.TestCase):
    def _valid_snapshot(self, valid_text_tokens: int = 57, precision_control_f32: bool = False):
        shapes = control._expected_shapes(valid_text_tokens)
        source_types = control._source_types(precision_control_f32)
        return {
            name: _FakeTensor(
                shape,
                "torch.bfloat16" if source_types[name] == "BF16" else "torch.float32",
            )
            for name, shape in shapes.items()
        }

    def test_exact_nine_tensor_geometry_and_source_dtypes_are_admitted(self):
        snapshot = self._valid_snapshot()
        shapes = control.validate_snapshot_layout(snapshot, 57, require_mps=True)
        self.assertEqual(tuple(shapes), control.TENSOR_ORDER)
        self.assertEqual(shapes["block.txt_valid_f32"], (1, 57, 2048))
        self.assertEqual(shapes["block.img_f32"], (1, 4320, 2048))
        self.assertEqual(control._source_types(), control._SOURCE_TYPES)

    def test_precision_flag_is_opt_in_and_single_use(self):
        common = [
            "--conditioning-manifest", "/tmp/conditioning.json",
            "--conditioning-manifest-sha256", "0" * 64,
            "--conditioning-payload-sha256", "1" * 64,
            "--t1-qwen-inputs", "/tmp/t1-qwen.json",
            "--t1-qwen-inputs-sha256", "2" * 64,
            "--action-profile", "/tmp/action-profile.json",
            "--action-profile-sha256", "3" * 64,
            "--plan-only",
        ]
        self.assertFalse(control._parse_args(common).precision_control_f32)
        self.assertTrue(control._parse_args(common + ["--precision-control-f32"]).precision_control_f32)
        with self.assertRaisesRegex(control.PreblockTraceError, "only once"):
            control._parse_args(common + ["--precision-control-f32", "--precision-control-f32"])

    def test_f32_precision_control_changes_only_prelude_outputs(self):
        source_types = control._source_types(precision_control_f32=True)
        self.assertEqual(
            [name for name in control.TENSOR_ORDER
             if source_types[name] != control._SOURCE_TYPES[name]],
            list(control._F32_PRELUDE_NAMES),
        )
        shapes = control.validate_snapshot_layout(
            self._valid_snapshot(263, precision_control_f32=True), 263,
            precision_control_f32=True,
        )
        self.assertEqual(shapes["block.txt_valid_f32"], (1, 263, 2048))
        with self.assertRaisesRegex(control.PreblockTraceError, "source dtype"):
            control.validate_snapshot_layout(
                self._valid_snapshot(263), 263, precision_control_f32=True,
            )

    def test_missing_tensor_wrong_geometry_or_source_dtype_rejects(self):
        snapshot = self._valid_snapshot()
        snapshot.pop("block.vec_txt_f32")
        with self.assertRaisesRegex(control.PreblockTraceError, "inventory or order"):
            control.validate_snapshot_layout(snapshot, 57)

        snapshot = self._valid_snapshot()
        snapshot["block.img_f32"] = _FakeTensor((1, 4320, 2047), "torch.bfloat16")
        with self.assertRaisesRegex(control.PreblockTraceError, "geometry"):
            control.validate_snapshot_layout(snapshot, 57)

        snapshot = self._valid_snapshot()
        snapshot["block.vec_f32"] = _FakeTensor((1, 2048), "torch.float32")
        with self.assertRaisesRegex(control.PreblockTraceError, "source dtype"):
            control.validate_snapshot_layout(snapshot, 57)

    def test_missing_or_reordered_preblock_stage_rejects(self):
        snapshot = self._valid_snapshot()
        with self.assertRaisesRegex(control.PreblockTraceError, "stages"):
            control.validate_snapshot_layout(snapshot, 57, stage_order=("time_in", "img_in", "block0"))
        with self.assertRaisesRegex(control.PreblockTraceError, "stages"):
            control.validate_snapshot_layout(
                snapshot, 57, stage_order=("txt_in", "time_in", "img_in", "block0")
            )

    def test_output_manifest_and_payload_must_both_be_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            manifest, payload = control.output_paths(output)
            self.assertEqual(manifest, output)
            self.assertEqual(payload, Path(directory) / "trace.tensors.bin")

            payload.write_bytes(b"occupied")
            with self.assertRaisesRegex(control.PreblockTraceError, "fresh files"):
                control.output_paths(output)

            payload.unlink()
            output.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(control.PreblockTraceError, "fresh files"):
                control.output_paths(output)


class RuntimeSentinelTests(unittest.TestCase):
    def _runtime(self):
        def original_load_blocks(*_args, **_kwargs):
            return ["original-block"]

        def original_bind(*_args, **_kwargs):
            return "original-bind"

        def original_caches(*_args, **_kwargs):
            return "original-caches"

        def original_clean(*_args, **_kwargs):
            return "original-clean"

        def original_load_qwen(*_args, **_kwargs):
            return "original-qwen"

        def original_load_epsilon(*_args, **_kwargs):
            return "original-epsilon"

        def original_install_refiner_attention(_torch, state, *_args, **_kwargs):
            return state, "original-adapter"

        def original_prelude(*_args, **_kwargs):
            return {name: _HookModule() for name in ("time_in", "txt_in", "img_in")}

        runtime = SimpleNamespace(
            single=SimpleNamespace(_load_all_blocks=original_load_blocks),
            two_segment=SimpleNamespace(_bind_runtime_cache_functions=original_bind),
            _new_caches=original_caches,
            _load_t0_clean=original_clean,
            _load_qwen=original_load_qwen,
            _load_cache_epsilon=original_load_epsilon,
            attention_reference=SimpleNamespace(_load_prelude_weights=original_prelude),
        )
        runtime.attention_reference._install_refiner_attention = original_install_refiner_attention
        originals = (
            runtime.single._load_all_blocks,
            runtime.two_segment._bind_runtime_cache_functions,
            runtime._new_caches,
            runtime._load_t0_clean,
            runtime._load_qwen,
            runtime._load_cache_epsilon,
            runtime.attention_reference._install_refiner_attention,
            runtime.attention_reference._load_prelude_weights,
        )
        return runtime, originals

    def test_sentinel_stops_before_block_and_all_runtime_patches_restore(self):
        runtime, originals = self._runtime()
        capture = {"events": []}
        with self.assertRaises(control._StopBeforeBlock0):
            with control._install_runtime_patches(runtime, capture):
                blocks = runtime.single._load_all_blocks()
                self.assertEqual(len(blocks), 1)
                self.assertEqual(runtime._new_caches(), ([unittest.mock.ANY], [unittest.mock.ANY]))
                modules = runtime.attention_reference._load_prelude_weights()
                modules["time_in"]("timestep")
                modules["txt_in"]("qwen")
                modules["img_in"]("latent")
                self.assertIs(capture["txt_in"]["consumed_args"][0], capture["txt_in"]["args"][0])
                self.assertIs(capture["img_in"]["consumed_args"][0], capture["img_in"]["args"][0])
                self.assertEqual(runtime._load_qwen(), "original-qwen")
                self.assertEqual(runtime._load_cache_epsilon(), "original-epsilon")
                self.assertEqual(
                    runtime.attention_reference._install_refiner_attention(
                        None, {"mode": "bf16"},
                    )[0]["mode"],
                    "bf16",
                )
                self.assertEqual(runtime._load_t0_clean(), "original-clean")
                self.assertEqual(capture["clean_t0"], "original-clean")
                blocks[0](img="preblock")

        restored = (
            runtime.single._load_all_blocks,
            runtime.two_segment._bind_runtime_cache_functions,
            runtime._new_caches,
            runtime._load_t0_clean,
            runtime._load_qwen,
            runtime._load_cache_epsilon,
            runtime.attention_reference._install_refiner_attention,
            runtime.attention_reference._load_prelude_weights,
        )
        self.assertEqual(restored, originals)
        self.assertEqual(capture["events"], ["time_in", "txt_in", "img_in", "block0"])
        self.assertEqual(capture["block_loader_calls"], 1)
        self.assertEqual(capture["sentinel_calls"], 1)
        self.assertEqual(capture["t0_qwen_load_calls"], 1)
        self.assertEqual(capture["cache_epsilon_load_calls"], 1)
        self.assertTrue(all(not module.pre_hooks and not module.post_hooks
                            for module in modules.values()))


class PrecisionManifestTests(unittest.TestCase):
    def _runtime(self):
        return RuntimeSentinelTests()._runtime()

    def _admitted(self):
        digest = "a" * 64
        return {
            "t0_qwen": {
                "manifest_sha256": digest,
                "payload_sha256": digest,
                "qwen_record": {"sha256": digest},
            },
            "t1_qwen": {
                "manifest_sha256": digest,
                "payload_sha256": digest,
                "qwen_record": {"sha256": digest},
            },
            "action_profile": {"profile_sha256": digest},
            "source_hashes": {},
            "payload_sha256": digest,
            "sha256": digest,
            "cache_index_sha256": digest,
            "global_shard": {"sha256": digest},
            "upstream_revision": "test-revision",
            "t0_valid_tokens": 263,
            "dit_text_rows": 320,
            "forward_plan": [{"kind": "writer"}],
        }

    def test_manifest_keeps_default_and_f32_precision_modes_distinct(self):
        args = SimpleNamespace(conditioning_manifest_sha256="b" * 64)
        admitted = self._admitted()
        default = control._manifest(args, admitted, "trace.tensors.bin", b"payload", {})
        f32_control = control._manifest(
            args, admitted, "trace.tensors.bin", b"payload", {}, precision_control_f32=True,
        )
        self.assertEqual(default["execution"]["mode"], "source-bf16")
        self.assertEqual(default["execution"]["source_dtype_by_tensor"], control._SOURCE_TYPES)
        self.assertNotIn("prelude_precision_control", default["execution"])
        self.assertEqual(f32_control["execution"]["mode"], "source-f32-precision-control")
        self.assertEqual(
            [name for name in control.TENSOR_ORDER
             if f32_control["execution"]["source_dtype_by_tensor"][name] != control._SOURCE_TYPES[name]],
            list(control._F32_PRELUDE_NAMES),
        )
        self.assertIn("cache-epsilon input", " ".join(
            f32_control["execution"]["prelude_precision_control"].values()
        ))

    def test_f32_override_casts_live_inputs_after_bf16_load_and_restores_runtime(self):
        runtime, originals = self._runtime()
        capture = {"events": []}
        observed = {}
        source_qwen = _FakeTensor((1, 1280, 3584), "torch.bfloat16")
        source_latent = _FakeTensor((1, 32, 30, 9, 16), "torch.bfloat16")
        source_time = _FakeTensor((1,), "torch.float32")
        source_admitted = object()
        source_epsilon = object()
        source_mask = object()
        torch = SimpleNamespace(float32="torch.float32")

        def load_prelude(_torch, _path, _upstream, _helpers, _device, dtype):
            observed["prelude_dtype"] = dtype
            return {name: _HookModule() for name in ("time_in", "txt_in", "img_in")}

        def load_qwen(*_args, **_kwargs):
            return source_qwen

        def load_epsilon(*_args, **_kwargs):
            return source_epsilon

        runtime.attention_reference._load_prelude_weights = load_prelude
        runtime._load_qwen = load_qwen
        runtime._load_cache_epsilon = load_epsilon
        state = {"mode": "bf16"}
        with control._install_runtime_patches(
            runtime, capture, precision_control_f32=True,
            expected_t0_qwen=source_qwen, expected_admitted=source_admitted,
        ):
            runtime.attention_reference._install_refiner_attention(torch, state)
            modules = runtime.attention_reference._load_prelude_weights(
                torch, "path", "upstream", (), "mps", "torch.bfloat16",
            )
            self.assertEqual(state["mode"], "f32")
            self.assertEqual(observed["prelude_dtype"], "torch.float32")
            self.assertIs(runtime._load_qwen(torch, None, source_qwen, "mps"), source_qwen)
            self.assertIs(runtime._load_cache_epsilon(torch, None, source_admitted, "mps"), source_epsilon)
            modules["time_in"](source_time)
            modules["txt_in"](source_qwen, source_time, source_mask)
            modules["img_in"](source_latent)

            self.assertIs(capture["t0_qwen"], source_qwen)
            self.assertIs(capture["cache_epsilon"], source_epsilon)
            self.assertIs(capture["txt_in"]["args"][0], source_qwen)
            self.assertIs(capture["img_in"]["args"][0], source_latent)
            for name in ("txt_in", "img_in"):
                record = capture[name]
                self.assertIs(record["consumed_args"][0], record["cast_input"])
                self.assertEqual(record["consumed_args"][0].dtype, "torch.float32")
            self.assertEqual(capture["txt_in"]["args"][0].dtype, "torch.bfloat16")
            self.assertEqual(capture["img_in"]["args"][0].dtype, "torch.bfloat16")
            self.assertIs(capture["time_in"]["consumed_args"][0], source_time)
            self.assertEqual(capture["events"], ["time_in", "txt_in", "img_in"])

        self.assertIs(runtime._load_qwen, load_qwen)
        self.assertIs(runtime._load_cache_epsilon, load_epsilon)
        self.assertIs(runtime.attention_reference._load_prelude_weights, load_prelude)
        self.assertIs(runtime.attention_reference._install_refiner_attention, originals[6])
        self.assertTrue(all(not module.pre_hooks and not module.post_hooks
                            for module in modules.values()))


class WriterBindingTests(unittest.TestCase):
    def _capture(self, precision_control_f32: bool = False):
        valid = 263
        dit_rows = 320
        qwen = _FakeTensor((1, 1280, 3584), "torch.bfloat16")
        time_input = _FakeTensor((1,), "torch.float32")
        time_output = _FakeTensor((1, 2048), "torch.float32" if precision_control_f32 else "torch.bfloat16")
        text_mask_input = object()
        text_dtype = "torch.float32" if precision_control_f32 else "torch.bfloat16"
        text_output = _FakeTensor((1, 1280, 2048), text_dtype)
        text_input = (
            _FakeTensor((1, 1280, 3584), "torch.float32")
            if precision_control_f32 else qwen
        )
        text_record = {
            "args": (qwen, time_input, text_mask_input),
            "consumed_args": (text_input, time_input, text_mask_input),
            "cast_input": text_input if precision_control_f32 else None,
            "output": text_output,
        }
        noisy = _FakeTensor((1, 32, 30, 9, 16), "torch.bfloat16")
        image_input = (
            _FakeTensor((1, 32, 30, 9, 16), "torch.float32")
            if precision_control_f32 else noisy
        )
        image_dtype = "torch.float32" if precision_control_f32 else "torch.bfloat16"
        image_output = _FakeTensor((1, 4320, 2048), image_dtype)
        vector = time_output
        image_record = {
            "args": (noisy,),
            "consumed_args": (image_input,),
            "cast_input": image_input if precision_control_f32 else None,
            "output": image_output,
        }
        stages = {
            "time_in": {"args": (time_input,), "consumed_args": (time_input,), "output": time_output},
            "txt_in": text_record,
            "img_in": image_record,
        }
        clean = _FakeTensor((1, 32, 30, 9, 16), "torch.bfloat16")
        text_valid = _FakeTensor((1, valid, 2048), text_dtype)
        block_kwargs = {
            "cache_valid_len": 0,
            "update_cache": True,
            "cache_txt": True,
            "attn_param": None,
            "is_flash": False,
            "cache_attn_mode": "torch",
            "txt_tokens_per_block": dit_rows,
            "prompt_video_boundaries": [0, 4320],
            "kv_cache": {},
            "cache_mask": object(),
            "vec": vector,
            "vec_txt": vector,
            "img": image_output,
            "txt": _FakeTensor((1, dit_rows, 2048), text_dtype),
            "freqs_cis": (
                _FakeTensor((4320, 128), "torch.float32"),
                _FakeTensor((4320, 128), "torch.float32"),
            ),
            "text_mask": _FakeTensor((1, dit_rows), "torch.bool"),
        }
        capture = {
            **stages,
            "t0_qwen": qwen,
            "t0_qwen_load_calls": 1,
            "cache_epsilon": noisy,
            "cache_epsilon_load_calls": 1,
            "block_kwargs": block_kwargs,
        }
        snapshot = {
            "reference.latent_bf16": clean,
            "cache.noisy_latent_f32": noisy,
            "block.img_f32": image_output,
            "block.txt_valid_f32": text_valid,
            "block.vec_f32": vector,
            "block.vec_txt_f32": vector,
            "block.rope_cos_f32": block_kwargs["freqs_cis"][0],
            "block.rope_sin_f32": block_kwargs["freqs_cis"][1],
            "cache.timestep_f32": time_input,
        }
        admitted = {
            "forward_plan": [{
                "cache_valid_len": 0,
                "update_cache": True,
                "cache_txt": True,
            }],
            "dit_text_rows": dit_rows,
        }
        return capture, snapshot, admitted

    def test_default_and_f32_writer_bindings_accept_exact_t0_objects(self):
        for precision_control_f32 in (False, True):
            with self.subTest(precision_control_f32=precision_control_f32):
                capture, snapshot, admitted = self._capture(precision_control_f32)
                control._validate_writer_call(
                    capture, snapshot, admitted,
                    precision_control_f32=precision_control_f32,
                )

    def test_writer_binding_rejects_reloaded_qwen_or_epsilon(self):
        capture, snapshot, admitted = self._capture()
        capture["txt_in"]["args"] = (
            _FakeTensor((1, 1280, 3584), "torch.bfloat16"),
            capture["time_in"]["args"][0],
            object(),
        )
        with self.assertRaisesRegex(control.PreblockTraceError, "loaded T0 Qwen"):
            control._validate_writer_call(capture, snapshot, admitted)

        capture, snapshot, admitted = self._capture()
        capture["cache_epsilon_load_calls"] = 2
        with self.assertRaisesRegex(control.PreblockTraceError, "loaded exactly once"):
            control._validate_writer_call(capture, snapshot, admitted)


if __name__ == "__main__":
    unittest.main()
