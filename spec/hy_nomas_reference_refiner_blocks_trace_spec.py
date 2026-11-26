"""CPU-only contract checks for the source T0 refiner-block trace."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import hy_nomas_reference_refiner_blocks_trace as control  # noqa: E402


class _FakeTensor:
    def __init__(self, shape, dtype="torch.bfloat16", device_type="mps"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = type("Device", (), {"type": device_type})()

    def detach(self):
        return self


class _FakeHookModule:
    def __init__(self):
        self.forward_hooks = []
        self.forward_pre_hooks = []

    def register_forward_hook(self, hook):
        self.forward_hooks.append(hook)
        return object()

    def register_forward_pre_hook(self, hook):
        self.forward_pre_hooks.append(hook)
        return object()


class BlockTraceContractTests(unittest.TestCase):
    def _lifecycle_fixture(self, root: Path):
        source_file = root / "hymm" / "models" / "text_encoder" / "token_refiner.py"
        source_file.parent.mkdir(parents=True)
        source_file.touch()
        upstream = root

        def adapter(q, k, v, drop_rate=0, attn_mask=None, causal=False):
            return _FakeTensor((1, control.TEXT_ROWS, control.HIDDEN_DIM))

        def apply_gate(x, gate=None, tanh=False):
            return x

        token_module = ModuleType(control._TOKEN_REFINER_MODULE)
        token_module.__file__ = str(source_file)
        token_module.attention = adapter
        token_module.apply_gate = apply_gate

        class Block(_FakeHookModule):
            def __init__(self):
                super().__init__()
                self.adaLN_modulation = [_FakeHookModule(), _FakeHookModule()]
                self.norm1 = _FakeHookModule()
                self.self_attn_qkv = _FakeHookModule()
                self.self_attn_proj = _FakeHookModule()
                self.norm2 = _FakeHookModule()
                self.mlp = SimpleNamespace(
                    fc1=_FakeHookModule(), act=_FakeHookModule(), fc2=_FakeHookModule(),
                )

        individual = _FakeHookModule()
        individual.blocks = [Block(), Block()]

        def fake_forward(_self):
            return None

        for block in individual.blocks:
            block.forward = FunctionType(
                fake_forward.__code__, token_module.__dict__, "forward",
            ).__get__(block, type(block))
        refiner = _FakeHookModule()
        refiner.input_embedder = _FakeHookModule()
        refiner.individual_token_refiner = individual

        single = SimpleNamespace(token_module=token_module)

        def bind_refiner_attention_adapter(
            _upstream, bound_adapter, _adapter_state, module_instance=None,
        ):
            single.token_module.attention = bound_adapter
            return single.token_module

        single._bind_refiner_attention_adapter = bind_refiner_attention_adapter
        state = {"attention_calls": 0, "gate_calls": 0}
        trace_state = {"hook_handles": [], "checkpoints": {"c": object()}}
        return upstream, token_module, refiner, single, adapter, apply_gate, state, trace_state

    def test_attention_observer_is_installed_after_rebind_and_restored(self):
        with TemporaryDirectory() as directory:
            (upstream, token_module, refiner, single, adapter, original_gate,
             state, trace_state) = self._lifecycle_fixture(Path(directory))
            original_binder = single._bind_refiner_attention_adapter
            with patch.dict(sys.modules, {control._TOKEN_REFINER_MODULE: token_module}):
                control._install_block_hooks(
                    {"txt_in": refiner}, trace_state, state, upstream=upstream,
                )
                with control._observe_refiner_adapter_binding(single, state):
                    # The initial bind has no module instance; the final bind
                    # must replace that adapter with our observer afterward.
                    single._bind_refiner_attention_adapter(upstream, adapter, object())
                    self.assertIs(token_module.attention, adapter)
                    single._bind_refiner_attention_adapter(
                        upstream, adapter, object(), module_instance=refiner,
                    )
                    self.assertIsNot(token_module.attention, adapter)
                    self.assertTrue(state["attention_hook_installed"])

                    q = _FakeTensor((1, control.TEXT_ROWS, control.HEADS, control.HEAD_DIM))
                    k = _FakeTensor((1, control.TEXT_ROWS, control.HEADS, control.HEAD_DIM))
                    v = _FakeTensor((1, control.TEXT_ROWS, control.HEADS, control.HEAD_DIM))
                    mask = _FakeTensor((1, control.TEXT_ROWS), dtype="torch.bool")
                    for _ in range(2):
                        token_module.attention(q, k, v, attn_mask=mask)
                    x = _FakeTensor((1, control.TEXT_ROWS, control.HIDDEN_DIM))
                    gate = _FakeTensor((1, control.HIDDEN_DIM))
                    for _ in range(4):
                        token_module.apply_gate(x, gate)

                    self.assertEqual(state["attention_calls"], 2)
                    self.assertEqual(state["gate_calls"], 4)
                    self.assertEqual(
                        state["events"],
                        [
                            "block0.q", "block0.k", "block0.v", "block0.attention",
                            "block1.q", "block1.k", "block1.v", "block1.attention",
                            "block0.gate_msa", "block0.gate_mlp",
                            "block1.gate_msa", "block1.gate_mlp",
                        ],
                    )

                self.assertIs(token_module.attention, adapter)
                self.assertIs(token_module.apply_gate, original_gate)
                self.assertIs(single._bind_refiner_attention_adapter, original_binder)

    def test_preflight_binding_is_transparent_before_hook_setup_and_late_foreign_binding_rejects(self):
        with TemporaryDirectory() as directory:
            (upstream, token_module, refiner, single, adapter, original_gate,
             state, trace_state) = self._lifecycle_fixture(Path(directory))
            with patch.dict(sys.modules, {control._TOKEN_REFINER_MODULE: token_module}):
                with control._observe_refiner_adapter_binding(single, state):
                    preflight_result = single._bind_refiner_attention_adapter(
                        upstream, adapter, object(), module_instance=object(),
                    )
                    self.assertIs(preflight_result, token_module)
                    self.assertIs(token_module.attention, adapter)
                    self.assertNotIn("refiner_instance", state)

                    control._install_block_hooks(
                        {"txt_in": refiner}, trace_state, state, upstream=upstream,
                    )
                    single._bind_refiner_attention_adapter(
                        upstream, adapter, object(), module_instance=refiner,
                    )
                    self.assertTrue(state["attention_hook_installed"])
                    with self.assertRaisesRegex(
                        control.BlockTraceError, "another refiner instance",
                    ):
                        single._bind_refiner_attention_adapter(
                            upstream, adapter, object(), module_instance=object(),
                        )

                self.assertIs(token_module.attention, adapter)
                self.assertIs(token_module.apply_gate, original_gate)

    def test_attention_and_gate_globals_restore_when_bound_forward_fails(self):
        with TemporaryDirectory() as directory:
            (upstream, token_module, refiner, single, adapter, original_gate,
             state, trace_state) = self._lifecycle_fixture(Path(directory))
            original_binder = single._bind_refiner_attention_adapter
            with patch.dict(sys.modules, {control._TOKEN_REFINER_MODULE: token_module}):
                control._install_block_hooks(
                    {"txt_in": refiner}, trace_state, state, upstream=upstream,
                )
                with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
                    with control._observe_refiner_adapter_binding(single, state):
                        single._bind_refiner_attention_adapter(
                            upstream, adapter, object(), module_instance=refiner,
                        )
                        raise RuntimeError("synthetic forward failure")

                self.assertIs(token_module.attention, adapter)
                self.assertIs(token_module.apply_gate, original_gate)
                self.assertIs(single._bind_refiner_attention_adapter, original_binder)

    def test_attention_and_gate_count_errors_report_separate_observed_counts(self):
        with self.assertRaisesRegex(control.BlockTraceError, "attention call count.*observed 1, expected 2"):
            control._block_snapshot(
                {"values": {}, "attention_calls": 1, "gate_calls": 4,
                 "attention_hook_installed": True},
                {},
            )
        with self.assertRaisesRegex(control.BlockTraceError, "gate call count.*observed 3, expected 4"):
            control._block_snapshot(
                {"values": {}, "attention_calls": 2, "gate_calls": 3,
                 "attention_hook_installed": True},
                {},
            )

    def test_canonical_name_order_and_full_padded_shapes(self):
        names = ["block0.input", "c"]
        block_stages = (
            "condition_silu", "adaLN", "gate_msa", "gate_mlp", "norm1", "qkv",
            "q", "k", "v", "attention", "proj", "residual1", "norm2",
            "mlp_fc1", "mlp_silu", "mlp_fc2", "output",
        )
        names.extend(f"block0.{stage}" for stage in block_stages)
        names.append("block1.input")
        names.extend(f"block1.{stage}" for stage in block_stages)

        self.assertEqual(control.BLOCK_TENSOR_ORDER, tuple(names))
        self.assertEqual(len(control.BLOCK_TENSOR_ORDER), 37)
        shapes = control.expected_shapes(valid_text_tokens=263)
        self.assertEqual(set(shapes), set(names))
        self.assertEqual(shapes["block0.input"], (1, 1280, 2048))
        self.assertEqual(shapes["c"], (1, 2048))
        self.assertEqual(shapes["block0.qkv"], (1, 1280, 3, 2048))
        for block in (0, 1):
            prefix = f"block{block}."
            self.assertEqual(shapes[prefix + "q"], (1, 1280, 2048))
            self.assertEqual(shapes[prefix + "k"], (1, 1280, 2048))
            self.assertEqual(shapes[prefix + "v"], (1, 1280, 2048))
            self.assertEqual(shapes[prefix + "attention"], (1, 1280, 2048))
            self.assertEqual(shapes[prefix + "mlp_fc1"], (1, 1280, 8192))
            self.assertEqual(shapes[prefix + "mlp_silu"], (1, 1280, 8192))
            self.assertEqual(shapes[prefix + "output"], (1, 1280, 2048))
        self.assertEqual(shapes["block1.input"], (1, 1280, 2048))

    def test_source_bf16_snapshot_must_have_exact_order_shape_and_device(self):
        shapes = control.expected_shapes(valid_text_tokens=263)
        snapshot = {
            name: _FakeTensor(shape)
            for name, shape in shapes.items()
        }
        control.validate_snapshot(snapshot, valid_text_tokens=263)

        reordered = dict(reversed(tuple(snapshot.items())))
        with self.assertRaisesRegex(control.BlockTraceError, "inventory or order"):
            control.validate_snapshot(reordered, valid_text_tokens=263)

        wrong_shape = dict(snapshot)
        wrong_shape["block0.q"] = _FakeTensor((1, 263, 2048))
        with self.assertRaisesRegex(control.BlockTraceError, "shape"):
            control.validate_snapshot(wrong_shape, valid_text_tokens=263)

        wrong_dtype = dict(snapshot)
        wrong_dtype["block1.mlp_silu"] = _FakeTensor(
            shapes["block1.mlp_silu"], dtype="torch.float32",
        )
        with self.assertRaisesRegex(control.BlockTraceError, "dtype"):
            control.validate_snapshot(wrong_dtype, valid_text_tokens=263)

        wrong_device = dict(snapshot)
        wrong_device["block0.attention"] = _FakeTensor(
            shapes["block0.attention"], device_type="cpu",
        )
        with self.assertRaisesRegex(control.BlockTraceError, "MPS"):
            control.validate_snapshot(wrong_device, valid_text_tokens=263)

    def test_block_input_aliases_are_identity_checked_without_duplicate_storage(self):
        block0_output = _FakeTensor((1, 1280, 2048))
        captured = {
            "block0.input": _FakeTensor((1, 1280, 2048)),
            "c": _FakeTensor((1, 2048)),
            "block0.output": block0_output,
            "block1.input": block0_output,
        }
        control.validate_input_aliases(captured)
        with self.assertRaisesRegex(control.BlockTraceError, "block0.output"):
            control.validate_input_aliases({})

        captured["block1.input"] = _FakeTensor((1, 1280, 2048))
        with self.assertRaisesRegex(control.BlockTraceError, "block0.output"):
            control.validate_input_aliases(captured)

    def test_packer_defers_the_only_device_readback_until_terminal_pack(self):
        source = __import__("inspect").getsource(control.pack_snapshot)
        self.assertEqual(source.count(".cpu()"), 1)
        self.assertNotIn(".item(", source)

    def test_manifest_distinguishes_source_dtype_from_export_dtype(self):
        shapes = control.expected_shapes(valid_text_tokens=263)
        records = {
            name: {
                "dtype": "F32", "source_dtype": "BF16", "shape": list(shapes[name]),
                "offset": 0, "nbytes": 1, "sha256": "c" * 64,
            }
            for name in control.BLOCK_TENSOR_ORDER
        }
        records["block1.input"] = {
            **records["block0.output"], "alias_of": "block0.output",
        }
        manifest = control.manifest_contract(
            source_manifest={
                "schema": "cogni_ml.hy_nomas_reference_refiner_source_trace/v1",
                "payload_sha256": "a" * 64,
                "payload": "prelude.tensors.bin",
                "valid_text_tokens": 263,
                "source": {"upstream_revision": control.PINNED_REVISION},
                "execution": {"mode": "source-bf16", "device": "mps"},
            },
            source_manifest_sha256="b" * 64,
            payload_name="blocks.tensors.bin",
            payload=b"x",
            records=records,
            valid_text_tokens=263,
        )
        self.assertEqual(manifest["schema"], control.SCHEMA)
        self.assertEqual(manifest["source_refiner_manifest_sha256"], "b" * 64)
        self.assertEqual(manifest["source_refiner_payload_sha256"], "a" * 64)
        self.assertEqual(manifest["processed_tokens"], 1280)
        self.assertEqual(manifest["valid_text_tokens"], 263)
        self.assertEqual(manifest["padding_tokens"], 1017)
        self.assertEqual(manifest["execution"]["mode"], "source-bf16")
        self.assertEqual(manifest["execution"]["device"], "mps")
        self.assertEqual(manifest["tensors"]["block0.input"]["source_dtype"], "BF16")
        self.assertEqual(manifest["tensors"]["block0.input"]["dtype"], "F32")


if __name__ == "__main__":
    unittest.main()
