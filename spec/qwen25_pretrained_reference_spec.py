#!/usr/bin/env python3
"""CPU-only contract tests for the Qwen2.5-VL pretrained prefix producer."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples/support/qwen25_pretrained_reference.py"
SPEC = importlib.util.spec_from_file_location("qwen25_pretrained_reference", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the Qwen pretrained reference producer")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def _valid_manifest(directory: Path) -> dict:
    hidden_shape = [reference.PREFIX_LAYER_COUNT + 1, 1, 1, reference.REAL_TEXT_GEOMETRY["hidden_size"]]
    sidecar_bytes = b"\x00\x00" * __import__("math").prod(hidden_shape)
    sidecar_path = directory / "prefix.bf16le"
    sidecar_path.write_bytes(sidecar_bytes)
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()
    checkpoint_file = {
        "path": "model.safetensors-00001-of-00004.safetensors",
        "nbytes": 2,
        "sha256": "a" * 64,
        "hf_lfs_sha256": None,
    }
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
    weight_sources = [
        {
            "key": f"model.layers.{layer}.{name}",
            "shard": checkpoint_file["path"],
            "dtype": "BF16",
            "shape": reference._expected_weight_shape(name),
            "nbytes": __import__("math").prod(reference._expected_weight_shape(name)) * 2,
            "tensor_sha256": "b" * 64,
        }
        for layer in range(reference.PREFIX_LAYER_COUNT)
        for name in names
    ]
    return {
        "schema": reference.SCHEMA,
        "schema_version": reference.SCHEMA_VERSION,
        "checkpoint": {
            "repo_id": reference.MODEL_ID,
            "revision": reference.MODEL_REVISION,
            "receipt_sha256": "e" * 64,
            "config_sha256": "c" * 64,
            "index_sha256": "d" * 64,
            "verified_files": {
                "config.json": {
                    "path": "config.json",
                    "nbytes": 1,
                    "sha256": "c" * 64,
                    "hf_lfs_sha256": None,
                },
                "model.safetensors.index.json": {
                    "path": "model.safetensors.index.json",
                    "nbytes": 1,
                    "sha256": "d" * 64,
                    "hf_lfs_sha256": None,
                },
                checkpoint_file["path"]: checkpoint_file,
            },
        },
        "config": {
            "hidden_size": reference.REAL_TEXT_GEOMETRY["hidden_size"],
            "num_hidden_layers": reference.SOURCE_LAYER_COUNT,
            "sequence_length": 1,
            "attention_implementation": "sdpa",
        },
        "inputs": {
            "token_ids": [1],
            "attention_mask": [1],
            "position_offset": 0,
            "position_ids_f32": [[0.0], [0.0], [0.0]],
            "position_ids_for_official_model": [[0], [0], [0]],
            "inv_freq_f32": [1.0] * (reference.REAL_TEXT_GEOMETRY["head_dim"] // 2),
        },
        "capture": {
            "source_layer_count": reference.SOURCE_LAYER_COUNT,
            "prefix_layer_count": reference.PREFIX_LAYER_COUNT,
            "captured_state_count": reference.PREFIX_LAYER_COUNT + 1,
            "state_indexing": "0=token_embedding; i+1=decoder_layer_i_output",
            "hidden_states": {
                "dtype": "bfloat16",
                "shape": hidden_shape,
                "storage": {
                    "sidecar": "bf16",
                    "offset_bytes": 0,
                    "nbytes": len(sidecar_bytes),
                },
            },
            "embedding_source_key": reference.EMBEDDING_KEY,
            "embedding_source_shard": checkpoint_file["path"],
            "embedding_unique_rows": [{
                "key": reference.EMBEDDING_KEY,
                "row": 1,
                "shard": checkpoint_file["path"],
                "dtype": "BF16",
                "shape": [reference.REAL_TEXT_GEOMETRY["hidden_size"]],
                "nbytes": reference.REAL_TEXT_GEOMETRY["hidden_size"] * 2,
                "tensor_sha256": "3" * 64,
            }],
        },
        "sidecars": {
            "bf16": {
                "path": sidecar_path.name,
                "encoding": reference.SIDECAR_ENCODING,
                "nbytes": len(sidecar_bytes),
                "sha256": sidecar_sha,
            }
        },
        "weight_sources": weight_sources,
        "provenance": {
            "producer_source_sha256": "e" * 64,
            "upstream_helper_source_sha256": "f" * 64,
            "transformers_sources": {"modeling.py": {"sha256": "1" * 64}},
            "torch_sources": {"functional.py": {"sha256": "2" * 64}},
            "device": "mps",
            "activation_dtype": "bfloat16",
            "comparison_tolerance": {"rtol": reference.RTOL, "atol": reference.ATOL},
            "hidden_states_minus3_mapping": (
                "source-derived: after decoder layer 25 equals hidden_states[-3] for the pinned 28-layer "
                "Qwen2_5_VLTextModel.forward tuple; this fixture runs only layers 0-25"
            ),
            "full_tuple_produced": False,
        },
    }


class Qwen25PretrainedReferenceSpec(unittest.TestCase):
    def test_accepts_short_right_padded_explicit_token_ids(self):
        request = reference._validate_inputs([10, 20, 0, 0], [1, 1, 0, 0], 100, 9, 32)
        self.assertEqual(request["token_ids"], [10, 20, 0, 0])
        self.assertEqual(request["position_ids_f32"], [[9.0, 10.0, 11.0, 12.0]] * 3)

    def test_rejects_malformed_or_non_right_padded_inputs(self):
        bad_requests = (
            ([True], [1], 100, 0, 32),
            ([100], [1], 100, 0, 32),
            ([1, 2, 3], [1, 0, 1], 100, 0, 32),
            ([1], [0], 100, 0, 32),
            ([1], [1], 100, -1, 32),
            ([1], [1], 100, 128000, 32),
        )
        for request in bad_requests:
            with self.subTest(request=request), self.assertRaises(ValueError):
                reference._validate_inputs(*request)

    def test_explicit_ids_keep_1280_limit_while_natural_path_has_bounded_extension(self):
        ids = [1] * (reference.MAX_SEQUENCE_LENGTH + 1)
        mask = [1] * len(ids)
        with self.assertRaisesRegex(ValueError, "1280-token limit"):
            reference._validate_inputs(ids, mask, 100, 0, 32)
        request = reference._validate_inputs(
            ids,
            mask,
            100,
            0,
            32,
            reference.MAX_NATURAL_SEQUENCE_LENGTH,
        )
        self.assertEqual(request["sequence_length"], reference.MAX_SEQUENCE_LENGTH + 1)
        self.assertEqual(request["multimodal_position_shape"], [3, 1, len(ids)])

    def test_prompt_cli_rejects_mixed_or_incomplete_input_modes_before_runtime(self):
        common = [
            "--checkpoint-root", "/missing/checkpoint",
            "--checkpoint-receipt", "/missing/receipt.json",
            "--output", "/tmp/qwen25-never-created.json",
        ]
        with self.assertRaisesRegex(SystemExit, "must be supplied together"):
            reference.main(common + ["--hy-nomas-prompt-file", "/tmp/prompt.txt"])
        with self.assertRaisesRegex(SystemExit, "does not accept --token-ids"):
            reference.main(common + [
                "--hy-nomas-prompt-file", "/tmp/prompt.txt",
                "--hy-nomas-root", "/tmp/hy-nomas",
                "--token-ids", "1,2",
            ])

    def test_rejects_receipt_with_wrong_revision_or_path_traversal(self):
        receipt = {
            "repo_id": reference.MODEL_ID,
            "revision": reference.MODEL_REVISION,
            "files": {
                "config.json": {"nbytes": 1, "sha256": "a" * 64, "hf_lfs_sha256": None},
                "model.safetensors.index.json": {"nbytes": 1, "sha256": "b" * 64, "hf_lfs_sha256": None},
            },
        }
        self.assertEqual(len(reference._validate_receipt_metadata(receipt)), 2)
        malformed = copy.deepcopy(receipt)
        malformed["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "revision"):
            reference._validate_receipt_metadata(malformed)
        malformed = copy.deepcopy(receipt)
        malformed["files"]["../config.json"] = malformed["files"].pop("config.json")
        with self.assertRaisesRegex(ValueError, "basenames"):
            reference._validate_receipt_metadata(malformed)

    def test_rejects_receipt_hash_or_size_drift(self):
        receipt = {
            "repo_id": reference.MODEL_ID,
            "revision": reference.MODEL_REVISION,
            "files": {
                "config.json": {"nbytes": 1, "sha256": "z" * 64, "hf_lfs_sha256": None},
                "model.safetensors.index.json": {"nbytes": 1, "sha256": "b" * 64, "hf_lfs_sha256": None},
            },
        }
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            reference._validate_receipt_metadata(receipt)
        receipt["files"]["config.json"]["sha256"] = "a" * 64
        receipt["files"]["config.json"]["nbytes"] = True
        with self.assertRaisesRegex(ValueError, "size"):
            reference._validate_receipt_metadata(receipt)

    def test_validates_real_text_geometry_and_rejects_drift(self):
        config = {
            "model_type": "qwen2_5_vl",
            "text_config": {
                **reference.REAL_TEXT_GEOMETRY,
                "rope_scaling": {"rope_type": "default", "mrope_section": [16, 24, 24]},
            },
        }
        self.assertEqual(reference._text_config(config)["hidden_size"], 3584)
        config["text_config"]["num_key_value_heads"] = 8
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            reference._text_config(config)

    def test_accepts_pinned_flat_top_level_model_config(self):
        config = {
            "model_type": "qwen2_5_vl",
            **reference.REAL_TEXT_GEOMETRY,
            "head_dim": None,
            "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
        }
        text = reference._text_config(config)
        self.assertEqual(text["hidden_size"], 3584)
        self.assertIsNone(text["head_dim"])

    def test_materializes_official_transformers_config_without_head_dim(self):
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

        values = dict(reference.REAL_TEXT_GEOMETRY)
        values["rope_scaling"] = {"type": "mrope", "mrope_section": [16, 24, 24]}
        config = reference._parse_text_config(
            types.SimpleNamespace(Qwen2_5_VLTextConfig=Qwen2_5_VLTextConfig),
            values,
        )
        config.num_hidden_layers = reference.SOURCE_LAYER_COUNT
        self.assertFalse(hasattr(config, "head_dim"))

        record = reference._materialize_config_record(config, sequence_length=4)

        for key, expected in reference.REAL_TEXT_GEOMETRY.items():
            self.assertEqual(record[key], expected, key)
        self.assertEqual(record["rope_scaling"]["mrope_section"], [16, 24, 24])
        self.assertEqual(record["sequence_length"], 4)
        self.assertEqual(record["attention_implementation"], "sdpa")

    def test_config_metadata_preflight_rejects_missing_field(self):
        config = types.SimpleNamespace(**{
            key: value
            for key, value in reference.REAL_TEXT_GEOMETRY.items()
            if key != "head_dim"
        })
        config.rope_scaling = {"type": "default", "mrope_section": [16, 24, 24]}
        config._attn_implementation = "sdpa"
        del config.hidden_act
        with self.assertRaisesRegex(RuntimeError, "missing hidden_act"):
            reference._materialize_config_record(config, sequence_length=4)

    def test_rejects_a_prefix_count_that_does_not_match_26_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            document = _valid_manifest(directory)
            reference._validate_manifest(document, directory / "manifest.json")
            malformed = copy.deepcopy(document)
            malformed["capture"]["prefix_layer_count"] = 25
            with self.assertRaisesRegex(ValueError, "prefix layer count"):
                reference._validate_manifest(malformed)

    def test_rejects_missing_or_duplicate_layer_weight_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            document = _valid_manifest(directory)
            document["weight_sources"].pop()
            with self.assertRaisesRegex(ValueError, "all 26 layers"):
                reference._validate_manifest(document)
            document = _valid_manifest(directory)
            document["weight_sources"][-1] = copy.deepcopy(document["weight_sources"][0])
            with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
                reference._validate_manifest(document)

    def test_enforces_15_percent_mps_memory_reserve(self):
        reference._require_memory_budget(700, 100, 1000)
        with self.assertRaisesRegex(MemoryError, "15% reserve"):
            reference._require_memory_budget(751, 100, 1000)
        with self.assertRaises(ValueError):
            reference._require_memory_budget(True, 0, 1000)

    def test_prefix_runner_has_one_terminal_sync_and_packed_readback(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        run = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_prefix"
        )
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        sync_calls = [
            node for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "synchronize"
        ]
        self.assertEqual(len(sync_calls), 1)
        self.assertIsInstance(sync_calls[0].func.value, ast.Attribute)
        self.assertEqual(sync_calls[0].func.value.attr, "mps")
        stack_call = next(
            node for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "stack"
        )
        append_call = next(
            node for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sidecar_writer"
        )
        self.assertLess(stack_call.lineno, sync_calls[0].lineno)
        self.assertLess(sync_calls[0].lineno, append_call.lineno)
        cpu_transfers = [
            node for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
            and any(
                keyword.arg == "device"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "cpu"
                for keyword in node.keywords
            )
        ]
        self.assertEqual(len(cpu_transfers), 1, "only small rotary metadata may read back separately")
        decoder_loop = next(
            node for node in ast.walk(run)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
            and node.iter.args
            and isinstance(node.iter.args[0], ast.Name)
            and node.iter.args[0].id == "layers"
        )
        loop_calls = [node for node in ast.walk(decoder_loop) if isinstance(node, ast.Call)]
        self.assertFalse(any(
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"synchronize", "to"}
            for node in loop_calls
        ))
        self.assertFalse(any(
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sidecar_writer"
            for node in loop_calls
        ))

    def test_prefix_preparation_converts_serializable_checkpoint_root_to_path(self):
        class PreparationReached(Exception):
            pass

        class FakeDevice:
            def __init__(self, device):
                self.device = device

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint_root = directory / "checkpoint"
            checkpoint_root.mkdir()
            receipt_path = directory / "receipt.json"
            receipt_path.write_text("{}", encoding="utf-8")
            args = types.SimpleNamespace(
                checkpoint_root=checkpoint_root,
                checkpoint_receipt=receipt_path,
                token_ids=[1],
                attention_mask=[1],
                position_offset=0,
            )
            text_config = dict(reference.REAL_TEXT_GEOMETRY)
            checkpoint = {
                "root": str(checkpoint_root),
                "index_weight_map": {reference.EMBEDDING_KEY: "embedding.safetensors"},
            }
            config_fields = {
                key: value
                for key, value in reference.REAL_TEXT_GEOMETRY.items()
                if key != "head_dim"
            }
            config = types.SimpleNamespace(**config_fields)
            config.rope_scaling = {"type": "default", "mrope_section": [16, 24, 24]}
            config._attn_implementation = "sdpa"
            modeling = types.SimpleNamespace(
                Qwen2_5_VLDecoderLayer=lambda _config, layer_idx: {"layer_idx": layer_idx}
            )
            helper = types.SimpleNamespace(
                _import_official=lambda _torch: (modeling, object(), {}, "mock-transformers"),
                _verify_torch_sources=lambda _torch: {},
            )
            fake_torch = types.ModuleType("torch")
            fake_torch.backends = types.SimpleNamespace(
                mps=types.SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
            )
            fake_torch.mps = types.SimpleNamespace(
                current_allocated_memory=lambda: 0,
                recommended_max_memory=lambda: 1,
            )
            fake_torch.device = FakeDevice
            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = object()

            expected_shapes = {"test.weight": (1,)}
            observed_roots = []
            materialized_records = []

            materialize_config_record = reference._materialize_config_record

            def record_config(config_value, sequence_length, **kwargs):
                record = materialize_config_record(config_value, sequence_length, **kwargs)
                materialized_records.append(record)
                return record

            def record_embedding_root(root, *_args):
                self.assertEqual(len(materialized_records), 1)
                self.assertIsInstance(root, Path)
                observed_roots.append(root)
                return object(), object(), []

            def record_layer_root(_index, _layer, _shapes, root, *_args):
                self.assertIsInstance(root, Path)
                observed_roots.append(root)
                raise PreparationReached

            with (
                mock.patch.dict(
                    os.environ,
                    {"PYTORCH_ENABLE_MPS_FALLBACK": "0"},
                ),
                mock.patch.dict(
                    sys.modules,
                    {"torch": fake_torch, "safetensors": fake_safetensors},
                ),
                mock.patch.object(
                    reference,
                    "_checkpoint_inputs",
                    return_value=(text_config, {}, checkpoint),
                ),
                mock.patch.object(reference, "_text_config", return_value=text_config),
                mock.patch.object(reference, "_load_upstream_helper", return_value=helper),
                mock.patch.object(reference, "_parse_text_config", return_value=config),
                mock.patch.object(reference, "_materialize_config_record", side_effect=record_config),
                mock.patch.object(reference, "_expected_weight_names", return_value=expected_shapes),
                mock.patch.object(
                    reference,
                    "_estimate_mps_bytes",
                    return_value={"estimated_peak_mps_bytes": 0},
                ),
                mock.patch.object(reference, "_require_memory_budget"),
                mock.patch.object(
                    reference,
                    "_load_selected_embedding_rows",
                    side_effect=record_embedding_root,
                ),
                mock.patch.object(
                    reference,
                    "_load_decoder_layer",
                    side_effect=record_layer_root,
                ),
            ):
                with self.assertRaises(PreparationReached):
                    reference._run_prefix(args, sidecar_writer=None)

                self.assertEqual(observed_roots, [checkpoint_root, checkpoint_root])
                self.assertTrue(all(isinstance(root, Path) for root in observed_roots))
                self.assertEqual(checkpoint["root"], str(checkpoint_root))

                # A missing config field is rejected before even selected embedding
                # rows are copied to MPS. This keeps metadata failures cheap.
                del config.hidden_act
                materialized_records.clear()
                observed_roots.clear()
                with self.assertRaisesRegex(RuntimeError, "missing hidden_act"):
                    reference._run_prefix(args, sidecar_writer=None)
                self.assertEqual(materialized_records, [])
                self.assertEqual(observed_roots, [])


if __name__ == "__main__":
    unittest.main()
