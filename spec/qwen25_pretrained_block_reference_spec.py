#!/usr/bin/env python3
"""CPU-only authentication and indexing tests for the pretrained block oracle."""

from __future__ import annotations

import hashlib
import ast
import importlib.util
import json
import math
import struct
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples/support/qwen25_pretrained_block_reference.py"
SPEC = importlib.util.spec_from_file_location("qwen25_pretrained_block_reference", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the pretrained block reference producer")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class Qwen25PretrainedBlockReferenceSpec(unittest.TestCase):
    def test_authenticates_exact_manifest_and_sidecar_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sidecar_bytes = bytes.fromhex("0102030405060708")
            sidecar_path = root / "prefix.bf16le"
            sidecar_path.write_bytes(sidecar_bytes)
            document = {
                "sidecars": {
                    "bf16": {
                        "path": sidecar_path.name,
                        "encoding": "bfloat16-le",
                        "nbytes": len(sidecar_bytes),
                        "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                    }
                }
            }
            manifest_bytes = json.dumps(document, sort_keys=True).encode("utf-8")
            manifest_path = root / "prefix.json"
            manifest_path.write_bytes(manifest_bytes)
            trusted_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

            with mock.patch.object(reference, "TRUSTED_PREFIX_SHA256S", frozenset({trusted_sha256})):
                snapshot = reference._read_prefix_snapshot(
                    manifest_path,
                    trusted_sha256,
                    lambda value: self.assertEqual(value, document),
                )

            self.assertEqual(snapshot["manifest_bytes"], manifest_bytes)
            self.assertEqual(snapshot["manifest_sha256"], trusted_sha256)
            self.assertEqual(snapshot["sidecar_bytes"], sidecar_bytes)
            self.assertEqual(snapshot["sidecar_sha256"], document["sidecars"]["bf16"]["sha256"])

            # Mutations after authentication cannot change the consumed snapshot.
            manifest_path.write_text("{}", encoding="utf-8")
            sidecar_path.write_bytes(b"replacement")
            self.assertEqual(snapshot["manifest_bytes"], manifest_bytes)
            self.assertEqual(snapshot["sidecar_bytes"], sidecar_bytes)

    def test_rejects_wrong_trusted_digest_before_manifest_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "prefix.json"
            manifest_path.write_text("{}", encoding="utf-8")
            validator_called = False

            def validator(_document):
                nonlocal validator_called
                validator_called = True

            with mock.patch.object(reference, "TRUSTED_PREFIX_SHA256S", frozenset({"a" * 64})):
                with self.assertRaisesRegex(ValueError, "trusted prefix manifest digest"):
                    reference._read_prefix_snapshot(manifest_path, "b" * 64, validator)
            self.assertFalse(validator_called)

    def test_rejects_sidecar_digest_drift_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sidecar_bytes = b"\0\0"
            sidecar_path = root / "prefix.bf16le"
            sidecar_path.write_bytes(sidecar_bytes)
            document = {
                "sidecars": {
                    "bf16": {
                        "path": sidecar_path.name,
                        "encoding": "bfloat16-le",
                        "nbytes": len(sidecar_bytes),
                        "sha256": "0" * 64,
                    }
                }
            }

            for mutate in (
                lambda value: value["sidecars"]["bf16"].update(sha256="0" * 64),
                lambda value: value["sidecars"]["bf16"].update(path="../prefix.bf16le"),
            ):
                candidate = json.loads(json.dumps(document))
                mutate(candidate)
                encoded = json.dumps(candidate, sort_keys=True).encode("utf-8")
                manifest_path = root / "prefix.json"
                manifest_path.write_bytes(encoded)
                digest = hashlib.sha256(encoded).hexdigest()
                with mock.patch.object(reference, "TRUSTED_PREFIX_SHA256S", frozenset({digest})):
                    with self.assertRaises(ValueError):
                        reference._read_prefix_snapshot(manifest_path, digest, lambda _value: None)

    def test_trusted_prefix_digests_include_the_authenticated_s1280_fixture(self):
        self.assertIn(reference.TRUSTED_PREFIX_SHA256, reference.TRUSTED_PREFIX_SHA256S)
        self.assertEqual(
            reference.TRUSTED_PREFIX_S1280_SHA256,
            "b78616cc1dc3b1deb1215b06ecc49fbf60e3b38477f24aec2a0ccd60da745d3a",
        )
        self.assertIn(reference.TRUSTED_PREFIX_S1280_SHA256, reference.TRUSTED_PREFIX_SHA256S)

    def test_layer_index_is_bounded_to_the_authenticated_prefix(self):
        self.assertEqual(reference._validate_layer_index(0), 0)
        self.assertEqual(reference._validate_layer_index(24), 24)
        self.assertEqual(reference._validate_layer_index(25), 25)
        for value in (-1, 26, True, 24.0, "24"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reference._validate_layer_index(value)

    def test_diagnostic_inputs_preserve_prefix_rotary_contract(self):
        producer = reference._load_pretrained_producer()
        request = producer._validate_inputs(
            [11, 12, 13, 0], [1, 1, 1, 0],
            producer.REAL_TEXT_GEOMETRY["vocab_size"], 7,
            producer.REAL_TEXT_GEOMETRY["hidden_size"],
        )
        prefix_inputs = dict(request)
        prefix_inputs["inv_freq_f32"] = [0.5] * 64

        diagnostic = reference._diagnostic_inputs(request, prefix_inputs)

        self.assertEqual(diagnostic["multimodal_position_shape"], [3, 1, 4])
        self.assertEqual(diagnostic["position_ids_f32"], request["position_ids_f32"])
        self.assertEqual(diagnostic["position_ids_for_official_model"], request["position_ids_for_official_model"])
        self.assertEqual(diagnostic["inv_freq_f32"], [0.5] * 64)
        self.assertEqual(
            set(diagnostic),
            {
                "token_ids", "attention_mask", "position_ids_f32",
                "position_ids_for_official_model", "multimodal_position_shape",
                "position_offset", "inv_freq_f32",
            },
        )

        changed = dict(prefix_inputs)
        changed["multimodal_position_shape"] = [3, 1, 32]
        with self.assertRaisesRegex(ValueError, "multimodal_position_shape"):
            reference._diagnostic_inputs(request, changed)

        changed = dict(prefix_inputs)
        changed["inv_freq_f32"] = [float("nan")] * 64
        with self.assertRaisesRegex(ValueError, "inverse frequencies"):
            reference._diagnostic_inputs(request, changed)

    def test_config_generated_inverse_frequencies_must_match_prefix_inputs(self):
        prefix_values = [0.5 / (index + 1) for index in range(64)]
        reference._verify_config_generated_inv_freq(prefix_values, list(prefix_values))

        changed = list(prefix_values)
        changed[24] = math.nextafter(changed[24], math.inf)
        with self.assertRaisesRegex(ValueError, "differs from prefix input at 24"):
            reference._verify_config_generated_inv_freq(prefix_values, changed)

        with self.assertRaisesRegex(ValueError, "geometry is invalid"):
            reference._verify_config_generated_inv_freq(prefix_values[:-1], prefix_values[:-1])

    def test_rms_trace_requires_the_pinned_qwen2_norm_source_hash(self):
        source = reference.RMS_SOURCE_FILE
        self.assertEqual(
            reference._verified_rms_source_sha256(
                {source: {"sha256": reference.RMS_SOURCE_SHA256}}
            ),
            reference.RMS_SOURCE_SHA256,
        )
        with self.assertRaisesRegex(RuntimeError, "source record is missing or changed"):
            reference._verified_rms_source_sha256({source: {"sha256": "0" * 64}})

    def test_rms_trace_names_and_dtypes_follow_non24_layer_index(self):
        stages = {name: object() for name in reference.RMS_TRACE_STAGE_SUFFIXES}
        entries = reference._qualified_rms_trace_entries(0, stages)

        self.assertEqual(
            [name for name, _, _ in entries],
            [f"layers.0.input_layernorm.{name}" for name in reference.RMS_TRACE_STAGE_SUFFIXES],
        )
        self.assertEqual(
            [dtype for _, _, dtype in entries],
            ["float32", "float32", "float32", "float32", "bfloat16"],
        )
        self.assertEqual(
            set(reference._rms_trace_shapes(0, 2, 4)),
            {name for name, _, _ in entries},
        )
        with self.assertRaisesRegex(RuntimeError, "inventory differs"):
            reference._qualified_rms_trace_entries(0, {"variance_f32": object()})

    def test_post_attention_rms_site_is_selectable_and_authenticates_residual_input(self):
        common = ["--checkpoint-root", "/checkpoint", "--checkpoint-receipt", "/receipt",
                  "--prefix-reference", "/prefix.json", "--prefix-reference-sha256", "a" * 64,
                  "--output", "/output.json"]
        default_args = reference._parse_args(common)
        post_args = reference._parse_args(common + ["--rms-site", "post-attention"])

        self.assertEqual(default_args.rms_site, "input")
        self.assertEqual(reference._normalize_rms_site(default_args.rms_site), "input_layernorm")
        self.assertEqual(
            reference._normalize_rms_site(post_args.rms_site), "post_attention_layernorm"
        )

        stages = {name: object() for name in reference.RMS_TRACE_STAGE_SUFFIXES}
        entries = reference._qualified_rms_trace_entries(
            0, stages, "post_attention_layernorm"
        )
        self.assertEqual(
            [name for name, _, _ in entries],
            [f"layers.0.post_attention_layernorm.{name}" for name in reference.RMS_TRACE_STAGE_SUFFIXES],
        )
        self.assertEqual(
            set(reference._rms_trace_shapes(0, 2, 4, "post_attention_layernorm")),
            {name for name, _, _ in entries},
        )

        prefix_input = bytes.fromhex("01020304")
        residual_input = bytes.fromhex("05060708")
        self.assertEqual(
            reference._validate_rms_input_binding(
                "post_attention_layernorm", residual_input, prefix_input, residual_input, 0
            ),
            hashlib.sha256(residual_input).hexdigest(),
        )
        with self.assertRaisesRegex(RuntimeError, "differs from captured residual_after_attention"):
            reference._validate_rms_input_binding(
                "post_attention_layernorm", prefix_input, prefix_input, residual_input, 0
            )
        self.assertEqual(
            reference._validate_rms_input_binding(
                "input_layernorm", prefix_input, prefix_input, residual_input, 0
            ),
            hashlib.sha256(prefix_input).hexdigest(),
        )

    def test_rms_dispatch_observer_records_actual_cpu_ops_without_replacing_them(self):
        import torch

        class RMSNormCall(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.variance_epsilon = 1e-6
                self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)

        module = RMSNormCall()
        original_forward = module.forward
        hidden_states = torch.tensor([[[1, -2, 3, 4], [4, 3, -2, 1]]], dtype=torch.bfloat16)
        expected_shapes = reference._rms_trace_shapes(24, 2, 4)

        with torch.inference_mode():
            with reference._observe_official_rmsnorm(module, torch) as captured:
                actual = module(hidden_states)

        self.assertIs(module.forward.__func__, original_forward.__func__)
        self.assertIs(module.forward.__self__, module)
        self.assertEqual(actual.dtype, torch.bfloat16)
        reference._validate_rms_trace(captured, expected_shapes, torch, require_mps=False)
        self.assertEqual(set(captured), set(reference.RMS_TRACE_STAGE_SUFFIXES) | {"input_bf16"})
        self.assertTrue(torch.equal(captured["normalized_bf16"], actual))

    def test_rms_trace_source_is_the_selected_official_norm_class(self):
        pinned_class = type(
            "Qwen2RMSNorm",
            (),
            {"__module__": "transformers.models.qwen2.modeling_qwen2"},
        )
        selected = pinned_class()
        source_map = {
            reference.RMS_SOURCE_FILE: {"sha256": reference.RMS_SOURCE_SHA256}
        }
        reference._validate_official_rmsnorm_instance(selected, pinned_class, source_map)

        with self.assertRaisesRegex(RuntimeError, "not the imported official"):
            reference._validate_official_rmsnorm_instance(object(), pinned_class, source_map)

    def test_mixed_diagnostic_sidecars_have_explicit_f32_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "rms-trace.json"
            bf16_payload = bytes.fromhex("01020304")
            f32_payload = struct.pack("<2f", 1.25, -0.5)
            document = {"schema": "test"}

            manifest_path, bf16_path = reference._write_diagnostic(
                output_path,
                document,
                {"bf16": bf16_payload, "f32": f32_payload},
            )

            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(bf16_path.read_bytes(), bf16_payload)
            f32_path = Path(temporary) / "rms-trace.json.f32le"
            self.assertEqual(f32_path.read_bytes(), f32_payload)
            self.assertEqual(saved["sidecars"]["f32"]["encoding"], "float32-le")
            self.assertEqual(saved["sidecars"]["f32"]["byte_order"], "little")
            self.assertEqual(saved["sidecars"]["f32"]["storage_dtype"], "float32")
            self.assertEqual(saved["sidecars"]["f32"]["nbytes"], len(f32_payload))
            self.assertEqual(saved["sidecars"]["f32"]["sha256"], hashlib.sha256(f32_payload).hexdigest())

    def test_selected_weight_digests_must_match_the_prefix_certificate_for_layer0(self):
        producer = reference._load_pretrained_producer()
        names = (
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
        sources = []
        for name in names:
            shape = producer._expected_weight_shape(name)
            sources.append({
                "key": f"model.layers.0.{name}",
                "shard": "model-00004-of-00005.safetensors",
                "dtype": "BF16",
                "shape": shape,
                "nbytes": math.prod(shape) * 2,
                "tensor_sha256": "a" * 64,
            })
        document = {"weight_sources": [dict(source) for source in sources]}
        reference._validate_layer_weight_digests(sources, document, 0)

        sources[3]["tensor_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "trusted prefix"):
            reference._validate_layer_weight_digests(sources, document, 0)

    def test_forward_path_has_one_terminal_mps_synchronization(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        run = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run"
        )
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        sync_calls = [
            node for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize"
        ]
        self.assertEqual(len(sync_calls), 1)
        sync = sync_calls[0]
        self.assertIsInstance(sync.func.value, ast.Attribute)
        self.assertEqual(sync.func.value.attr, "mps")
        capture = next(
            node for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_capture_layer"
        )
        post_wait_readback = next(
            node for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "_tensor_bf16_bytes"
        )
        self.assertLess(capture.lineno, sync.lineno)
        self.assertGreater(post_wait_readback.lineno, sync.lineno)

    def test_extracts_indexed_input_and_expected_output_without_repacking(self):
        producer = reference._load_pretrained_producer()
        hidden_size = producer.REAL_TEXT_GEOMETRY["hidden_size"]
        states = [index.to_bytes(2, "little") * hidden_size for index in range(27)]
        document = {
            "inputs": {
                "token_ids": [1],
                "attention_mask": [1],
                "position_offset": 0,
            },
            "capture": {
                "hidden_states": {
                    "shape": [27, 1, 1, hidden_size],
                }
            },
        }
        snapshot = {"document": document, "sidecar_bytes": b"".join(states)}

        for layer_index in (0, 24):
            with self.subTest(layer_index=layer_index):
                input_state, expected_output, full_shape, request = reference._prefix_state_bytes(
                    snapshot, layer_index, producer
                )

                self.assertEqual(input_state, states[layer_index])
                self.assertEqual(expected_output, states[layer_index + 1])
                self.assertEqual(full_shape, [27, 1, 1, hidden_size])
                self.assertEqual(request["sequence_length"], 1)
        with self.assertRaisesRegex(ValueError, "layer index"):
            reference._prefix_state_bytes(snapshot, 26, producer)


if __name__ == "__main__":
    unittest.main()
