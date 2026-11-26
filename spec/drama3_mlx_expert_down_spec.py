"""CPU falsifiers for the pinned MLX expert-down probe contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import copy
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "probe_mlx_expert_down.py"
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("drama3_mlx_expert_down_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


MODEL = {
    "repository": "cguldogan/drama-3-mlx-q4",
    "revision": "5331ab509359c0e4a7e353a19e255ddf3e8152a8",
}
UPSTREAM = {
    "repository": "https://github.com/cguldogan/mlx-audio",
    "revision": "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7",
}
MODEL_CONFIG = {
    "hidden_size": 2048,
    "moe_intermediate_size": 512,
    "num_experts": 256,
    "num_experts_per_tok": 8,
}
SOURCE_HASH_NAMES = (
    "config.json",
    "model.safetensors.index.json",
    "mlx_audio/lm/models/qwen3_5.py",
    "mlx_audio/lm/models/switch_layers.py",
    "mlx_audio/utils.py",
    "scripts/drama3/replay_mlx_backbone.py",
    "scripts/drama3/replay_mlx_moe.py",
    "input_bundle",
    "target_moe_weights",
)
SOURCE_HASHES = {name: f"{index:064x}" for index, name in enumerate(SOURCE_HASH_NAMES, 1)}
EXPERT_IDS = [159, 194, 245, 154, 230, 191, 171, 108]


def activation_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def make_fixture() -> tuple[dict, bytes]:
    values = [0.0] * MODEL_CONFIG["hidden_size"]
    activation_sha = hashlib.sha256(activation_bytes(values)).hexdigest()
    fixture = {
        "format": "drama3-moe-input-v1",
        "coordinate": {"row": 1, "layer": 39},
        "activation_name": "post_attention_layernorm",
        "activation_shape": [MODEL_CONFIG["hidden_size"]],
        "activation_dtype": "float32",
        "activation": values,
        "activation_sha256": activation_sha,
        "model": copy.deepcopy(MODEL),
        "upstream": copy.deepcopy(UPSTREAM),
        "source_hashes": copy.deepcopy(SOURCE_HASHES),
    }
    return fixture, json_bytes(fixture)


def make_upstream() -> tuple[dict, dict, bytes]:
    fixture, raw_fixture = make_fixture()
    input_sha = hashlib.sha256(raw_fixture).hexdigest()
    expert_ids = list(EXPERT_IDS)
    input_record = {
        "format": fixture["format"],
        "path": "moe-input.json",
        "sha256": input_sha,
        "activation_sha256": fixture["activation_sha256"],
        "shape": fixture["activation_shape"],
        "dtype": fixture["activation_dtype"],
        "activation_name": fixture["activation_name"],
    }
    rank_map = [{"rank": rank, "expert_id": expert_id}
                for rank, expert_id in enumerate(expert_ids)]
    upstream = {
        "format": "drama3-moe-layer-diagnostic-v1",
        "mode": "full_run",
        "coordinate": {"row": 1, "layer": 39},
        "model": copy.deepcopy(MODEL),
        "upstream": copy.deepcopy(UPSTREAM),
        "runtime": {"mlx": "0.31.1", "device": "Device(gpu, 0)"},
        "model_config": copy.deepcopy(MODEL_CONFIG),
        "capture_control": {"passed": True},
        "source_hashes": copy.deepcopy(SOURCE_HASHES),
        "input": input_record,
        "expert_rank_to_id": rank_map,
        "stages": {
            "normalized_input": {
                "shape": [2048], "dtype": "float32",
                "capture_kind": "upstream_sparse_argument",
                "values": list(fixture["activation"]),
            },
            "expert_ids_argpartition_order": {
                "shape": [8], "dtype": "int32", "capture_kind": "upstream_local",
                "values": list(expert_ids),
            },
            "expert_activation_outputs": {
                "shape": [8, 512], "dtype": "float32",
                "capture_kind": "upstream_swiglu_return",
                "expert_ids": list(expert_ids),
                "values": [0.125] * (8 * 512),
            },
            "expert_down_outputs": {
                "shape": [8, 2048], "dtype": "float32",
                "capture_kind": "upstream_switch_mlp_return",
                "expert_ids": list(expert_ids),
                "values": [0.25] * (8 * 2048),
            },
        },
    }
    ablation = {
        "format": "drama3-moe-ablation-v1",
        "coordinate": {"row": 1, "layer": 39},
        "model": copy.deepcopy(MODEL),
        "upstream": copy.deepcopy(UPSTREAM),
        "capture_control": {"passed": True},
        "native_down_replay_control": {"passed": True},
        "model_weight_provenance": {
            "target_layer": 39,
            "target_moe_weights_sha256": SOURCE_HASHES["target_moe_weights"],
            "tensor_count": 14,
            "model_path": "/models/drama-3",
            "config_sha256": SOURCE_HASHES["config.json"],
            "index_sha256": SOURCE_HASHES["model.safetensors.index.json"],
            "verified_before_gpu": True,
        },
        "input_hashes": {
            "upstream_trace_sha256": "a" * 64,
            "native_trace_sha256": "b" * 64,
            "upstream_input_bundle_sha256": input_sha,
            "upstream_activation_sha256": fixture["activation_sha256"],
            "native_fixture_sha256": input_sha,
            "input_fixture_bytes_equal": True,
            "normalized_input_bitwise_equal": True,
        },
    }
    return upstream, ablation, raw_fixture


def swap_rank_ids(upstream: dict) -> None:
    first, second = upstream["expert_rank_to_id"][:2]
    first["expert_id"], second["expert_id"] = second["expert_id"], first["expert_id"]


class ExpertDownProbeContract(unittest.TestCase):
    def test_accepts_real_shape_ordered_capture_and_provenance(self):
        upstream, ablation, raw_fixture = make_upstream()

        validated = probe.validate_probe_inputs(upstream, ablation)
        fixture_result = probe.validate_fixture_binding(
            upstream, json.loads(raw_fixture), raw_fixture
        )

        self.assertEqual(validated["route_ids"], EXPERT_IDS)
        self.assertEqual(validated["expert108_rank"], EXPERT_IDS.index(108))
        self.assertEqual(len(validated["activation_values"]), 8 * 512)
        self.assertEqual(len(validated["upstream_down_values"]), 8 * 2048)
        target_activation_sha = hashlib.sha256(activation_bytes([0.125] * 512)).hexdigest()
        self.assertEqual(validated["target_expert_activation_sha256"], target_activation_sha)
        self.assertNotEqual(validated["target_expert_activation_sha256"], validated["activation_sha256"])
        self.assertIsInstance(fixture_result, dict)

    def test_rejects_reordered_rank_map_or_stage_ids(self):
        mutations = (
            ("rank map", swap_rank_ids),
            ("activation IDs", lambda upstream: upstream["stages"]["expert_activation_outputs"]["expert_ids"].reverse()),
            ("down IDs", lambda upstream: upstream["stages"]["expert_down_outputs"]["expert_ids"].reverse()),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                upstream, ablation, _ = make_upstream()
                mutate(upstream)
                with self.assertRaises(ValueError):
                    probe.validate_probe_inputs(upstream, ablation)

    def test_rejects_nonfinite_values_wrong_shapes_and_untrusted_capture(self):
        mutations = (
            ("activation NaN", lambda upstream: upstream["stages"]["expert_activation_outputs"]["values"].__setitem__(0, float("nan"))),
            ("down infinity", lambda upstream: upstream["stages"]["expert_down_outputs"]["values"].__setitem__(0, float("inf"))),
            ("activation shape", lambda upstream: upstream["stages"]["expert_activation_outputs"].__setitem__("shape", [8, 511])),
            ("down shape", lambda upstream: upstream["stages"]["expert_down_outputs"].__setitem__("shape", [8, 2047])),
            ("activation capture", lambda upstream: upstream["stages"]["expert_activation_outputs"].__setitem__("capture_kind", "diagnostic_recompute")),
            ("down capture", lambda upstream: upstream["stages"]["expert_down_outputs"].__setitem__("capture_kind", None)),
            ("capture control", lambda upstream: upstream["capture_control"].__setitem__("passed", False)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                upstream, ablation, _ = make_upstream()
                mutate(upstream)
                with self.assertRaises(ValueError):
                    probe.validate_probe_inputs(upstream, ablation)

    def test_rejects_unpinned_model_upstream_runtime_and_provenance(self):
        mutations = (
            ("model revision", lambda upstream, ablation: upstream["model"].__setitem__("revision", "0" * 40)),
            ("upstream revision", lambda upstream, ablation: upstream["upstream"].__setitem__("revision", "0" * 40)),
            ("MLX runtime", lambda upstream, ablation: upstream["runtime"].__setitem__("mlx", "0.31.0")),
            ("device", lambda upstream, ablation: upstream["runtime"].__setitem__("device", "Device(cpu, 0)")),
            ("source hash", lambda upstream, ablation: upstream["source_hashes"].__setitem__("target_moe_weights", "0" * 64)),
            ("target digest", lambda upstream, ablation: ablation["model_weight_provenance"].__setitem__("target_moe_weights_sha256", "0" * 64)),
            ("config digest", lambda upstream, ablation: ablation["model_weight_provenance"].__setitem__("config_sha256", "0" * 64)),
            ("tensor count", lambda upstream, ablation: ablation["model_weight_provenance"].__setitem__("tensor_count", 13)),
            ("pre-GPU verification", lambda upstream, ablation: ablation["model_weight_provenance"].__setitem__("verified_before_gpu", False)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                upstream, ablation, _ = make_upstream()
                mutate(upstream, ablation)
                with self.assertRaises(ValueError):
                    probe.validate_probe_inputs(upstream, ablation)

    def test_rejects_different_input_identity_and_failed_replay_control(self):
        mutations = (
            ("input bytes", lambda upstream, ablation: ablation["input_hashes"].__setitem__("input_fixture_bytes_equal", False)),
            ("normalized activation", lambda upstream, ablation: ablation["input_hashes"].__setitem__("normalized_input_bitwise_equal", False)),
            ("activation hash", lambda upstream, ablation: ablation["input_hashes"].__setitem__("upstream_activation_sha256", "0" * 64)),
            ("input fixture hash", lambda upstream, ablation: ablation["input_hashes"].__setitem__("upstream_input_bundle_sha256", "0" * 64)),
            ("malformed trace hash", lambda upstream, ablation: ablation["input_hashes"].__setitem__("upstream_trace_sha256", "z" * 64)),
            ("native capture", lambda upstream, ablation: ablation["capture_control"].__setitem__("passed", False)),
            ("native replay", lambda upstream, ablation: ablation["native_down_replay_control"].__setitem__("passed", False)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                upstream, ablation, _ = make_upstream()
                mutate(upstream, ablation)
                with self.assertRaises(ValueError):
                    probe.validate_probe_inputs(upstream, ablation)

    def test_fixture_binding_rejects_stale_raw_bytes_and_activation_digest(self):
        upstream, _, raw_fixture = make_upstream()
        fixture = json.loads(raw_fixture)

        stale_raw = raw_fixture + b" "
        with self.assertRaises(ValueError):
            probe.validate_fixture_binding(upstream, fixture, stale_raw)

        altered_fixture = dict(fixture)
        altered_fixture["activation"] = list(fixture["activation"])
        altered_fixture["activation"][0] = 1.0
        with self.assertRaises(ValueError):
            probe.validate_fixture_binding(upstream, altered_fixture, raw_fixture)

    def test_model_source_validator_binds_native_replay_scripts_to_local_bytes(self):
        replay_sources = (
            "scripts/drama3/replay_mlx_backbone.py",
            "scripts/drama3/replay_mlx_moe.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            mlx_root = base / "mlx-audio"
            model_dir = base / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_bytes(b"{\"config\":true}\n")
            (model_dir / "model.safetensors.index.json").write_bytes(b"{\"index\":true}\n")

            hashes = copy.deepcopy(SOURCE_HASHES)
            hashes["config.json"] = hashlib.sha256(
                (model_dir / "config.json").read_bytes()
            ).hexdigest()
            hashes["model.safetensors.index.json"] = hashlib.sha256(
                (model_dir / "model.safetensors.index.json").read_bytes()
            ).hexdigest()
            for relative in (
                "mlx_audio/lm/models/qwen3_5.py",
                "mlx_audio/lm/models/switch_layers.py",
                "mlx_audio/utils.py",
            ):
                source_path = mlx_root / relative
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_bytes((relative + "\n").encode())
                hashes[relative] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            for relative in replay_sources:
                local_source = ROOT / relative
                hashes[relative] = hashlib.sha256(local_source.read_bytes()).hexdigest()

            fake_git = SimpleNamespace(stdout=f"{probe.UPSTREAM_REVISION}\n")
            with patch.object(probe.subprocess, "run", return_value=fake_git):
                valid = probe._validate_model_sources(mlx_root, model_dir, hashes)
                self.assertEqual(valid["config_sha256"], hashes["config.json"])
                for relative in replay_sources:
                    with self.subTest(source=relative):
                        tampered = dict(hashes)
                        tampered[relative] = "0" * 64
                        with self.assertRaises(ValueError):
                            probe._validate_model_sources(mlx_root, model_dir, tampered)

    def test_fixed_comparison_tolerance_reports_a_known_outlier(self):
        report = probe.compare_float32([0.0, 1.0], [0.0, 1.0 + 0.01])
        self.assertFalse(report["tolerance_passed"])
        self.assertEqual(report["failure_count"], 1)
        self.assertEqual(report["failure_indices"], [1])
        self.assertEqual(report["worst_index"], 1)

    def test_comparison_rejects_widened_tolerances(self):
        with self.assertRaises(ValueError):
            probe.compare_float32([1.0], [1.0], atol=1e-4, rtol=3e-4)
        with self.assertRaises(ValueError):
            probe.compare_float32([1.0], [1.0], atol=6e-5, rtol=4e-4)

    def test_bitwise_capture_detects_one_float32_ulp_even_within_tolerance(self):
        one_ulp = struct.unpack("!f", struct.pack("!I", struct.unpack("!I", struct.pack("!f", 0.25))[0] + 1))[0]
        report = probe.compare_float32([one_ulp], [0.25])
        self.assertFalse(report["bitwise_equal"])
        self.assertTrue(report["tolerance_passed"])
        self.assertEqual(report["failure_count"], 0)

    def test_execution_guard_rejects_read_before_terminal_eval_and_repeat_eval(self):
        calls: list[tuple[object, ...]] = []
        mx = SimpleNamespace(eval=lambda *arrays: calls.append(arrays))
        guard = probe.ExecutionGuard()
        array = SimpleNamespace(tolist=lambda: [1.0, 2.0])

        with self.assertRaises(RuntimeError):
            guard.readback("before", array)
        self.assertEqual(calls, [])

        guard.terminal_eval(mx, array)
        with self.assertRaises(RuntimeError):
            guard.terminal_eval(mx, array)
        self.assertEqual(calls, [(array,)])
        self.assertEqual(guard.readback("after", array), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
