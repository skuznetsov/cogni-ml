"""CPU-only contracts for the source T0 refiner-intermediate trace."""

from __future__ import annotations

from pathlib import Path
import inspect
import sys
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_reference_refiner_trace as control  # noqa: E402


class _FakeTensor:
    def __init__(self, shape, dtype, *, device_type="mps", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type)
        self._contiguous = contiguous

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        shape = list(self.shape)
        for dimension, selector in enumerate(key):
            if isinstance(selector, slice):
                start = 0 if selector.start is None else selector.start
                stop = shape[dimension] if selector.stop is None else selector.stop
                shape[dimension] = max(0, stop - start)
            elif isinstance(selector, int):
                shape[dimension] = None
        return _FakeTensor(tuple(value for value in shape if value is not None), self.dtype)

    def to(self, *, dtype):
        return _FakeTensor(self.shape, dtype)

    def detach(self):
        return self

    def contiguous(self):
        return self

    def is_contiguous(self):
        return self._contiguous


class _HookHandle:
    def __init__(self, remove):
        self._remove = remove

    def remove(self):
        self._remove()


class _Module:
    def __init__(self, forward=None):
        self.forward = forward or (lambda *_args: _FakeTensor((1, 2048), "torch.bfloat16"))
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
        output = self.forward(*args)
        for hook in tuple(self.post_hooks):
            hook(self, args, output)
        return output


def _fake_source(dtype="torch.bfloat16"):
    embed_layers = SimpleNamespace(__file__="", timestep_embedding=None)

    def timestep_embedding(timestep, _dimension, *_args, **_kwargs):
        return _FakeTensor((1, 256), "torch.float32")

    embed_layers.timestep_embedding = timestep_embedding

    def linear(name):
        return _Module(lambda _value: _FakeTensor((1, 2048), dtype))

    def timestep_embedder():
        mlp = [linear("mlp0"), linear("mlp1"), linear("mlp2")]

        def forward(timestep):
            raw = embed_layers.timestep_embedding(timestep, 256)
            return mlp[2](mlp[1](mlp[0](raw.to(dtype=dtype))))

        module = _Module(forward)
        module.mlp = mlp
        return module

    time_in = timestep_embedder()
    refiner_time = timestep_embedder()
    c_linear1 = linear("context1")
    c_silu = linear("context_silu")
    c_linear2 = linear("context2")
    c_embedder = _Module(lambda mean: c_linear2(c_silu(c_linear1(mean))))
    c_embedder.linear_1 = c_linear1
    c_embedder.act_1 = c_silu
    c_embedder.linear_2 = c_linear2
    input_embedder = _Module(
        lambda _x: _FakeTensor((1, control.base.QWEN_ROWS, 2048), dtype)
    )
    received = {}
    individual = _Module(
        lambda _x, _c, _mask: _FakeTensor((1, control.base.QWEN_ROWS, 2048), dtype)
    )

    def text_forward(x, timestep, mask):
        refiner_time(timestep)
        masked_mean = _FakeTensor((1, 3584), dtype)
        c_embedder(masked_mean)
        embedded = input_embedder(x)
        consumed_c = _FakeTensor((1, 2048), dtype)
        received["c"] = consumed_c
        return individual(embedded, consumed_c, mask)

    txt_in = _Module(text_forward)
    txt_in.t_embedder = refiner_time
    txt_in.c_embedder = c_embedder
    txt_in.input_embedder = input_embedder
    txt_in.individual_token_refiner = individual
    modules = {
        "time_in": time_in,
        "txt_in": txt_in,
        "img_in": _Module(),
        "final_layer": _Module(),
    }
    all_modules = [time_in, txt_in, refiner_time, *time_in.mlp, *refiner_time.mlp,
                   c_embedder, c_linear1, c_silu, c_linear2, input_embedder, individual]
    return modules, embed_layers, all_modules, received


class SnapshotContractTests(unittest.TestCase):
    def _snapshot(self, valid=263, f32=False):
        shapes = {
            **control.preblock._expected_shapes(valid),
            **control._expected_checkpoint_shapes(valid),
        }
        types = control._all_source_types(f32)
        return {
            name: _FakeTensor(shapes[name], {
                "BF16": "torch.bfloat16", "F32": "torch.float32", "BOOL": "torch.bool",
            }[types[name]])
            for name in control.TENSOR_ORDER
        }

    def test_exact_old_nine_plus_native_sixteen_plus_source_only_inventory(self):
        snapshot = self._snapshot()
        shapes = control.validate_snapshot_layout(snapshot, 263)
        self.assertEqual(tuple(snapshot), control.TENSOR_ORDER)
        self.assertEqual(control.TENSOR_ORDER[:9], control.preblock.TENSOR_ORDER)
        self.assertEqual(control.TENSOR_ORDER[9:25], control.COMMON_TENSOR_ORDER)
        self.assertEqual(control.TENSOR_ORDER[25:], control.SOURCE_ONLY_TENSOR_ORDER)
        self.assertEqual(control.SOURCE_ONLY_TENSOR_ORDER, (
            "refiner_timestep_sinusoidal_raw", "time_in.sinusoidal_raw", "refiner_mask",
        ))
        self.assertEqual(shapes["input_compact_valid"], (1, 263, 3584))
        self.assertEqual(shapes["input_embed"], (1, 263, 2048))
        self.assertEqual(shapes["refiner_mask"], (1, control.base.QWEN_ROWS))

    def test_source_bf16_and_opt_in_f32_checkpoint_types(self):
        bf16 = self._snapshot(263)
        self.assertEqual(control._dtype_name(bf16["masked_mean"]), "BF16")
        self.assertEqual(control._dtype_name(bf16["refiner_timestep_sinusoidal_raw"]), "F32")
        f32 = self._snapshot(263, f32=True)
        self.assertEqual(control._dtype_name(f32["masked_mean"]), "F32")
        self.assertEqual(control._dtype_name(f32["input_compact_valid"]), "F32")
        control.validate_snapshot_layout(f32, 263, precision_control_f32=True)
        with self.assertRaisesRegex(control.RefinerTraceError, "source dtype"):
            control.validate_snapshot_layout(bf16, 263, precision_control_f32=True)

    def test_missing_extra_reordered_shape_dtype_or_event_rejects(self):
        snapshot = self._snapshot()
        snapshot.pop("masked_mean")
        with self.assertRaisesRegex(control.RefinerTraceError, "inventory or order"):
            control.validate_snapshot_layout(snapshot, 263)

        snapshot = self._snapshot()
        snapshot["c"] = _FakeTensor((1, 2047), "torch.bfloat16")
        with self.assertRaisesRegex(control.RefinerTraceError, "geometry"):
            control.validate_snapshot_layout(snapshot, 263)

        snapshot = self._snapshot()
        snapshot["masked_mean"] = _FakeTensor((1, 3584), "torch.float32")
        with self.assertRaisesRegex(control.RefinerTraceError, "source dtype"):
            control.validate_snapshot_layout(snapshot, 263)

        snapshot = self._snapshot()
        with self.assertRaisesRegex(control.RefinerTraceError, "capture order"):
            control.validate_snapshot_layout(snapshot, 263, event_order=tuple(reversed(control.TRACE_EVENT_ORDER)))

    def test_single_terminal_copy_packer_has_no_scalar_readback(self):
        source = inspect.getsource(control._pack_payload)
        self.assertEqual(source.count(".cpu()"), 1)
        self.assertNotIn(".item(", source)

    def test_exported_refiner_mask_must_equal_valid_prefix(self):
        valid = 263
        expected = b"\x00\x00\x80\x3f" * valid + b"\x00" * (
            4 * (control.base.QWEN_ROWS - valid)
        )
        record = {
            "dtype": "F32", "shape": [1, control.base.QWEN_ROWS],
            "offset": 0, "nbytes": 4 * control.base.QWEN_ROWS,
        }
        control._validate_refiner_mask_payload(expected, {"refiner_mask": record}, valid)

        wrong = bytearray(expected)
        wrong[valid * 4:valid * 4 + 4] = b"\x00\x00\x80\x3f"
        with self.assertRaisesRegex(control.RefinerTraceError, "authenticated valid prefix"):
            control._validate_refiner_mask_payload(bytes(wrong), {"refiner_mask": record}, valid)

        with self.assertRaisesRegex(control.RefinerTraceError, "tensor contract"):
            control._validate_refiner_mask_payload(expected, {
                "refiner_mask": {**record, "dtype": "BOOL"},
            }, valid)


class HookLifecycleTests(unittest.TestCase):
    def _runtime(self, temp_root, *, dtype="torch.bfloat16", wrong_origin=False):
        modules, embed_layers, all_modules, received = _fake_source(dtype)
        upstream = Path(temp_root)
        source_path = upstream / "hymm/models/dit/modules/embed_layers.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("# fake source path for CPU hook tests\n", encoding="utf-8")
        actual_path = source_path if not wrong_origin else source_path.with_name("wrong.py")
        if wrong_origin:
            actual_path.write_text("# alternate fake source path\n", encoding="utf-8")
        embed_layers.__file__ = str(actual_path)
        name = control._TIME_EMBED_LAYERS
        old_module = sys.modules.get(name)
        sys.modules[name] = embed_layers

        def loader():
            return modules

        runtime = SimpleNamespace(attention_reference=SimpleNamespace(_load_prelude_weights=loader))
        return runtime, modules, embed_layers, all_modules, received, upstream, name, old_module

    def _run_source_calls(self, modules, embed_layers, dtype="torch.bfloat16", mask=None):
        timestep = _FakeTensor((1,), "torch.float32")
        qwen = _FakeTensor((1, control.base.QWEN_ROWS, 3584), dtype)
        if mask is None:
            mask = _FakeTensor((1, control.base.QWEN_ROWS), "torch.bool")
        vec = modules["time_in"](timestep)
        output = modules["txt_in"](qwen, timestep, mask)
        return timestep, qwen, mask, vec, output

    def _restore_module(self, name, previous):
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    def test_real_hook_tree_captures_actual_consumer_operands_and_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, modules, embed_layers, all_modules, received, upstream, name, previous = self._runtime(directory)
            original_embedding = embed_layers.timestep_embedding
            trace = {}
            try:
                with control._install_refiner_trace_runtime(runtime, trace, 263, upstream):
                    loaded = runtime.attention_reference._load_prelude_weights()
                    self.assertIs(loaded, modules)
                    timestep, qwen, mask, vec, text_output = self._run_source_calls(modules, embed_layers)
                    self.assertIs(trace["checkpoints"]["timestep"], timestep)
                    self.assertIs(trace["checkpoints"]["vec"], vec)
                    self.assertIs(trace["input_compact_full"], qwen)
                    self.assertIs(trace["refiner_mask_full"], mask)
                    self.assertIs(trace["checkpoints"]["refiner_mask"], mask)
                    self.assertIs(trace["checkpoints"]["c"], received["c"])
                    self.assertEqual(tuple(trace["events"]), control.TRACE_EVENT_ORDER)
                    clean = _FakeTensor(control.base.LATENT_SEGMENT_SHAPE, "torch.bfloat16")
                    noisy = _FakeTensor(control.base.LATENT_SEGMENT_SHAPE, "torch.bfloat16")
                    img_tokens = _FakeTensor(
                        (1, control.base.VIDEO_TOKENS, control.base.DIM), "torch.bfloat16"
                    )
                    text_tokens = _FakeTensor(
                        (1, control.base.QWEN_ROWS, control.base.DIM), "torch.bfloat16"
                    )
                    cos = _FakeTensor((control.base.VIDEO_TOKENS, control.base.HEAD_DIM), "torch.float32")
                    sin = _FakeTensor((control.base.VIDEO_TOKENS, control.base.HEAD_DIM), "torch.float32")
                    preblock_capture = {
                        "events": ["time_in", "txt_in", "img_in", "block0"],
                        "block_loader_calls": 1,
                        "sentinel_calls": 1,
                        "clean_t0": clean,
                        "time_in": {"args": (timestep,), "output": vec},
                        "txt_in": {
                            "args": (qwen, timestep, mask),
                            "consumed_args": (qwen, timestep, mask),
                            "output": text_output,
                        },
                        "img_in": {"args": (noisy,), "output": img_tokens},
                        "block_kwargs": {
                            "freqs_cis": (cos, sin), "txt": text_tokens,
                            "img": img_tokens, "vec": vec, "vec_txt": vec,
                        },
                    }
                    snapshot = control._snapshot_from_capture(preblock_capture, trace, 263)
                    control.validate_snapshot_layout(snapshot, 263, require_mps=True)
                    self.assertEqual(tuple(snapshot), control.TENSOR_ORDER)
                    self.assertIs(snapshot["cache.noisy_latent_f32"], noisy)
                    self.assertIs(snapshot["c"], received["c"])
                    self.assertIs(snapshot["refiner_mask"], mask)
                self.assertIs(embed_layers.timestep_embedding, original_embedding)
                self.assertTrue(all(not module.pre_hooks and not module.post_hooks for module in all_modules))
            finally:
                self._restore_module(name, previous)

    def test_hooks_and_global_function_restore_when_trace_scope_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _modules, embed_layers, all_modules, _received, upstream, name, previous = self._runtime(directory)
            original_embedding = embed_layers.timestep_embedding
            trace = {}
            try:
                with self.assertRaisesRegex(RuntimeError, "sentinel test"):
                    with control._install_refiner_trace_runtime(runtime, trace, 263, upstream):
                        runtime.attention_reference._load_prelude_weights()
                        raise RuntimeError("sentinel test")
                self.assertIs(embed_layers.timestep_embedding, original_embedding)
                self.assertTrue(all(not module.pre_hooks and not module.post_hooks for module in all_modules))
            finally:
                self._restore_module(name, previous)

    def test_wrong_pinned_source_origin_fails_closed_before_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _modules, embed_layers, all_modules, _received, upstream, name, previous = self._runtime(
                directory, wrong_origin=True,
            )
            original_embedding = embed_layers.timestep_embedding
            trace = {}
            try:
                with control._install_refiner_trace_runtime(runtime, trace, 263, upstream):
                    with self.assertRaisesRegex(control.RefinerTraceError, "authenticated pinned upstream"):
                        runtime.attention_reference._load_prelude_weights()
                self.assertIs(embed_layers.timestep_embedding, original_embedding)
                self.assertTrue(all(not module.pre_hooks and not module.post_hooks for module in all_modules))
            finally:
                self._restore_module(name, previous)

    def test_late_hook_registration_failure_removes_earlier_hooks_and_restores_function(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, modules, embed_layers, all_modules, _received, upstream, name, previous = self._runtime(directory)
            original_embedding = embed_layers.timestep_embedding

            def fail_registration(_hook):
                raise RuntimeError("late hook registration sentinel")

            modules["txt_in"].individual_token_refiner.register_forward_pre_hook = fail_registration
            trace = {}
            try:
                with control._install_refiner_trace_runtime(runtime, trace, 263, upstream):
                    with self.assertRaisesRegex(RuntimeError, "late hook registration sentinel"):
                        runtime.attention_reference._load_prelude_weights()
                self.assertIs(embed_layers.timestep_embedding, original_embedding)
                self.assertTrue(all(not module.pre_hooks and not module.post_hooks for module in all_modules))
            finally:
                self._restore_module(name, previous)

    def test_missing_checkpoint_fails_closed_on_snapshot(self):
        qwen = object()
        mask = object()
        trace = {
            "input_compact_full": qwen,
            "refiner_mask_full": mask,
            "loader_calls": 1,
            "active_sinusoid": None,
            "sinusoid_calls": {"time_in": 1, "refiner": 1},
            "events": list(control.TRACE_EVENT_ORDER[:-1]),
            "checkpoints": {name: object() for name in control.TRACE_EVENT_ORDER[:-1]},
        }
        with self.assertRaisesRegex(control.RefinerTraceError, "missing, duplicated, or out of order"):
            control._snapshot_from_capture({"txt_in": {"consumed_args": (qwen, object(), mask)}}, trace, 263)

    def test_source_mask_capture_rejects_bad_shape_dtype_device_and_contiguity(self):
        invalid_masks = (
            (_FakeTensor((1, control.base.QWEN_ROWS - 1), "torch.bool"), "shape"),
            (_FakeTensor((1, control.base.QWEN_ROWS), "torch.float32"), "boolean"),
            (_FakeTensor((1, control.base.QWEN_ROWS), "torch.bool", device_type="cpu"), "MPS"),
            (_FakeTensor((1, control.base.QWEN_ROWS), "torch.bool", contiguous=False), "contiguous"),
        )
        for mask, reason in invalid_masks:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                runtime, modules, _embed_layers, _all_modules, _received, upstream, name, previous = (
                    self._runtime(directory)
                )
                trace = {}
                try:
                    with control._install_refiner_trace_runtime(runtime, trace, 263, upstream):
                        runtime.attention_reference._load_prelude_weights()
                        timestep = _FakeTensor((1,), "torch.float32")
                        qwen = _FakeTensor((1, control.base.QWEN_ROWS, 3584), "torch.bfloat16")
                        with self.assertRaisesRegex(control.RefinerTraceError, reason):
                            modules["txt_in"](qwen, timestep, mask)
                finally:
                    self._restore_module(name, previous)

    def test_snapshot_binds_mask_to_actual_consumed_txt_in_arg_two(self):
        qwen = object()
        captured_mask = object()
        consumed_mask = object()
        trace = {
            "input_compact_full": qwen,
            "refiner_mask_full": captured_mask,
        }
        capture = {"txt_in": {"consumed_args": (qwen, object(), consumed_mask)}}
        with self.assertRaisesRegex(control.RefinerTraceError, "exact consumed txt_in argument 2"):
            control._snapshot_from_capture(capture, trace, 263)


if __name__ == "__main__":
    unittest.main()
