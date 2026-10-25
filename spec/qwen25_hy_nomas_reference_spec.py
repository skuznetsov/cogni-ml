"""CPU-only contract tests for the tiny upstream BF16 oracle schema."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples/support/qwen25_hy_nomas_reference.py"
SPEC = importlib.util.spec_from_file_location("qwen25_hy_nomas_reference", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def tensor_record(shape: list[int], dtype: str, values=None):
    count = 1
    for dimension in shape:
        count *= dimension
    if values is None:
        default = False if dtype == "bool" else 0
        values = [default] * count
    return {"dtype": dtype, "shape": shape, "values": values}


def fixture(layer_count: int = 1):
    config = reference._default_config(layer_count)
    config.update(
        {
            "sequence_length": 4,
            "binary_mask": [1, 1, 1, 0],
            "position_offset": 7,
        }
    )
    inputs = {
        "hidden_states": tensor_record([1, 4, 24], "bfloat16"),
        "binary_mask": tensor_record([1, 4], "int64", [1, 1, 1, 0]),
        "position_ids": tensor_record([4], "float32", [7.0, 8.0, 9.0, 10.0]),
        "multimodal_position_ids": tensor_record(
            [3, 1, 4], "int64", [7, 8, 9, 10] * 3
        ),
        "inv_freq": tensor_record([6], "float32"),
        "rotary_cos": tensor_record([3, 1, 4, 12], "bfloat16"),
        "rotary_sin": tensor_record([3, 1, 4, 12], "bfloat16"),
        "causal_attention_mask": tensor_record(
            [1, 1, 4, 4],
            "bool",
            [True, False, False, False,
             True, True, False, False,
             True, True, True, False,
             True, True, True, False],
        ),
        "additive_attention_bias": tensor_record(
            [1, 1, 4, 4],
            "bfloat16",
            [
                0.0, -3.3895313892515355e38, -3.3895313892515355e38,
                -3.3895313892515355e38,
                0.0, 0.0, -3.3895313892515355e38, -3.3895313892515355e38,
                0.0, 0.0, 0.0, -3.3895313892515355e38,
                0.0, 0.0, 0.0, -3.3895313892515355e38,
            ],
        ),
    }


    weight_shapes = {
        "input_layernorm.weight": [24],
        "self_attn.q_proj.weight": [24, 24],
        "self_attn.q_proj.bias": [24],
        "self_attn.k_proj.weight": [12, 24],
        "self_attn.k_proj.bias": [12],
        "self_attn.v_proj.weight": [12, 24],
        "self_attn.v_proj.bias": [12],
        "self_attn.o_proj.weight": [24, 24],
        "post_attention_layernorm.weight": [24],
        "mlp.gate_proj.weight": [48, 24],
        "mlp.up_proj.weight": [48, 24],
        "mlp.down_proj.weight": [24, 48],
    }
    stage_shapes = {
        "input_layernorm": [1, 4, 24],
        "self_attn.q_proj": [1, 4, 24],
        "self_attn.k_proj": [1, 4, 12],
        "self_attn.v_proj": [1, 4, 12],
        "post_rope_q": [1, 2, 4, 12],
        "post_rope_k": [1, 1, 4, 12],
        "attended": [1, 4, 24],
        "self_attn.o_proj": [1, 4, 24],
        "residual_after_attention": [1, 4, 24],
        "post_attention_layernorm": [1, 4, 24],
        "mlp.gate_proj": [1, 4, 48],
        "mlp.up_proj": [1, 4, 48],
        "silu_product": [1, 4, 48],
        "mlp.down_proj": [1, 4, 24],
        "hidden_states_out": [1, 4, 24],
    }
    layers = []
    for index in range(layer_count):
        layers.append(
            {
                "index": index,
                "initialization_seed": 20261024 + index,
                "weights": {
                    f"layers.{index}.{name}": tensor_record(
                        shape,
                        "bfloat16",
                        reference._expected_rms_gain_values("official_ones", 24)
                        if name in {
                            "input_layernorm.weight",
                            "post_attention_layernorm.weight",
                        }
                        else None,
                    )
                    for name, shape in weight_shapes.items()
                },
                "stages": {
                    (name if name in {"post_rope_q", "post_rope_k", "attended",
                                      "residual_after_attention", "silu_product",
                                      "hidden_states_out"}
                     else f"layers.{index}.{name}"): tensor_record(shape, "bfloat16")
                    for name, shape in stage_shapes.items()
                },
            }
        )
    return {
        "schema": reference.SCHEMA,
        "schema_version": reference.SCHEMA_VERSION,
        "config": config,
        "inputs": inputs,
        "layers": layers,
    }


def sidecar_fixture(directory: Path):
    document = fixture()
    payload = bytearray()

    def externalize(value):
        if isinstance(value, dict):
            if value.get("dtype") == "bfloat16" and "values" in value:
                offset = len(payload)
                for number in value["values"]:
                    bits = struct.unpack("<I", struct.pack("<f", float(number)))[0]
                    payload.extend(struct.pack("<H", bits >> 16))
                nbytes = len(payload) - offset
                del value["values"]
                value["storage"] = {
                    "sidecar": "bf16",
                    "offset_bytes": offset,
                    "nbytes": nbytes,
                }
            else:
                for child in value.values():
                    externalize(child)
        elif isinstance(value, list):
            for child in value:
                externalize(child)

    externalize(document)
    sidecar_path = directory / "fixture.bf16le"
    sidecar_path.write_bytes(payload)
    document["sidecars"] = {
        "bf16": {
            "path": sidecar_path.name,
            "encoding": reference.SIDECAR_ENCODING,
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    }
    return document, directory / "fixture.json", sidecar_path


class Qwen25HyNomasReferenceSpec(unittest.TestCase):
    def test_accepts_one_and_two_layer_fixtures(self):
        for layer_count in (1, 2):
            with self.subTest(layer_count=layer_count):
                reference._validate_fixture(fixture(layer_count))

    def test_rejects_non_bf16_decoded_weight_values(self):
        with self.assertRaisesRegex(ValueError, "exact F32 decode"):
            reference._record_from_values([1], [1.0001], "bfloat16")

    def test_rejects_stage_shape_drift(self):
        document = fixture()
        key = "layers.0.self_attn.q_proj"
        document["layers"][0]["stages"][key]["shape"] = [1, 4, 23]
        with self.assertRaises(ValueError):
            reference._validate_fixture(document)

    def test_rejects_layer_weight_aliasing_by_key(self):
        document = fixture(2)
        layer = document["layers"][1]
        layer["weights"] = copy.deepcopy(document["layers"][0]["weights"])
        with self.assertRaisesRegex(ValueError, "unexpected weight set"):
            reference._validate_fixture(document)

    def test_rejects_mismatched_position_offset(self):
        document = fixture()
        document["inputs"]["position_ids"]["values"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "position ids"):
            reference._validate_fixture(document)

    def test_rejects_non_right_padded_mask(self):
        document = fixture()
        document["inputs"]["binary_mask"]["values"] = [1, 0, 1, 0]
        document["config"]["binary_mask"] = [1, 0, 1, 0]
        with self.assertRaisesRegex(ValueError, "right padded"):
            reference._validate_fixture(document)

    def test_rejects_bias_that_does_not_match_official_bool_mask(self):
        document = fixture()
        document["inputs"]["additive_attention_bias"]["values"][0] = (
            -3.3895313892515355e38
        )
        with self.assertRaisesRegex(ValueError, "attention bias"):
            reference._validate_fixture(document)

    def test_accepts_explicit_nonunit_rms_gain_weights(self):
        document = fixture(2)
        document["config"]["rms_gain_mode"] = "nonunit_pattern_v1"
        expected = reference._expected_rms_gain_values("nonunit_pattern_v1", 24)
        for layer in document["layers"]:
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                layer["weights"][f"layers.{layer['index']}.{name}"]["values"] = expected
        reference._validate_fixture(document)

    def test_rejects_nonunit_mode_with_default_unit_weights(self):
        document = fixture()
        document["config"]["rms_gain_mode"] = "nonunit_pattern_v1"
        with self.assertRaisesRegex(ValueError, "RMS gain"):
            reference._validate_fixture(document)

    def test_accepts_contiguous_checksums_bf16_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, manifest_path, _ = sidecar_fixture(Path(temporary))
            reference._validate_fixture(document, manifest_path)

    def test_requires_manifest_path_for_sidecar_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, _, _ = sidecar_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "manifest path"):
                reference._validate_fixture(document)

    def test_rejects_sidecar_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, manifest_path, _ = sidecar_fixture(Path(temporary))
            document["sidecars"]["bf16"]["path"] = "../fixture.bf16le"
            with self.assertRaisesRegex(ValueError, "sibling basename"):
                reference._validate_fixture(document, manifest_path)

    def test_rejects_sidecar_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            document, manifest_path, sidecar_path = sidecar_fixture(directory)
            outside = directory.parent / f"{directory.name}-outside.bf16le"
            outside.write_bytes(sidecar_path.read_bytes())
            sidecar_path.unlink()
            sidecar_path.symlink_to(outside)
            try:
                with self.assertRaisesRegex(ValueError, "non-symlink"):
                    reference._validate_fixture(document, manifest_path)
            finally:
                outside.unlink(missing_ok=True)

    def test_rejects_bad_sidecar_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, manifest_path, _ = sidecar_fixture(Path(temporary))
            document["sidecars"]["bf16"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                reference._validate_fixture(document, manifest_path)

    def test_rejects_sidecar_size_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, manifest_path, _ = sidecar_fixture(Path(temporary))
            document["sidecars"]["bf16"]["nbytes"] += 2
            with self.assertRaisesRegex(ValueError, "file size"):
                reference._validate_fixture(document, manifest_path)

    def test_rejects_sidecar_range_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, manifest_path, _ = sidecar_fixture(Path(temporary))
            record = document["inputs"]["hidden_states"]
            record["storage"]["offset_bytes"] = 2
            with self.assertRaisesRegex(ValueError, "ranges overlap or leave"):
                reference._validate_fixture(document, manifest_path)

    def test_rejects_unreferenced_sidecar_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, manifest_path, sidecar_path = sidecar_fixture(Path(temporary))
            payload = sidecar_path.read_bytes() + b"\x00\x00"
            sidecar_path.write_bytes(payload)
            descriptor = document["sidecars"]["bf16"]
            descriptor["nbytes"] = len(payload)
            descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(ValueError, "unreferenced bytes"):
                reference._validate_fixture(document, manifest_path)

    def test_rejects_nonfinite_bf16_sidecar_words(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, manifest_path, sidecar_path = sidecar_fixture(Path(temporary))
            payload = bytearray(sidecar_path.read_bytes())
            payload[0:2] = b"\x80\x7f"  # BF16 positive infinity.
            sidecar_path.write_bytes(payload)
            document["sidecars"]["bf16"]["sha256"] = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(ValueError, "non-finite"):
                reference._validate_fixture(document, manifest_path)

    def test_rejects_records_with_both_inline_and_sidecar_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            document, manifest_path, _ = sidecar_fixture(Path(temporary))
            record = document["inputs"]["hidden_states"]
            record["values"] = [0.0] * 96
            with self.assertRaisesRegex(ValueError, "exactly one"):
                reference._validate_fixture(document, manifest_path)

    def test_rejects_tensor_record_over_sidecar_limit(self):
        record = {
            "dtype": "bfloat16",
            "shape": [reference.MAX_SIDECAR_BYTES // 2 + 1],
            "storage": {
                "sidecar": "bf16",
                "offset_bytes": 0,
                "nbytes": reference.MAX_SIDECAR_BYTES + 2,
            },
        }
        with self.assertRaisesRegex(ValueError, "512 MiB"):
            reference._validate_tensor_record(record, "bfloat16", [])

    def test_real_geometry_profile_has_pinned_7b_dimensions(self):
        config = reference._default_config(1, "qwen25-vl-7b")
        self.assertEqual(config["hidden_size"], 3584)
        self.assertEqual(config["intermediate_size"], 18944)
        self.assertEqual(config["num_attention_heads"], 28)
        self.assertEqual(config["num_key_value_heads"], 4)
        self.assertEqual(config["head_dim"], 128)
        self.assertEqual(config["num_hidden_layers"], 1)
        self.assertEqual(config["rms_norm_eps"], 1e-6)
        self.assertEqual(config["rope_theta"], 1_000_000.0)
        self.assertEqual(config["max_position_embeddings"], 128000)
        self.assertEqual(config["rope_scaling"]["mrope_section"], [16, 24, 24])
        self.assertEqual(
            reference.REAL_GEOMETRY_SOURCE["revision"],
            "eb6ba55b40ada64977147b52e50a4145ee89ac4f",
        )
        self.assertEqual(reference.REAL_GEOMETRY_SOURCE["source_num_hidden_layers"], 28)

    def test_rejects_real_geometry_without_sidecar(self):
        document = fixture()
        document["config"].update(reference._default_config(1, "qwen25-vl-7b"))
        document["config"].update(
            {"sequence_length": 4, "binary_mask": [1, 1, 1, 0], "position_offset": 7}
        )
        with self.assertRaisesRegex(ValueError, "requires its BF16 sidecar"):
            reference._validate_fixture(document)


if __name__ == "__main__":
    unittest.main()
