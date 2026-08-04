#!/usr/bin/env python3
"""Export a source-pinned CPU/F32 oracle for TRELLIS.2 CFG branches.

This executes the real ``FlowEulerCfgSampler._inference_model`` method through
its upstream MRO.  It isolates the three classifier-free-guidance branches and
deliberately excludes the sampler loop, guidance interval/rescale, RNG, model
weights, and every accelerator backend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/flow-euler-cfg-oracle/v1"
GENERATOR_REPO_PATH = "tools/trellis2_oracle/export_flow_euler_cfg.py"

SHAPE = [2, 4]
T = 0.6180339887498948
MIXED_STRENGTH = 1.7
X_T = [
    1.0,
    -2.0,
    0.5,
    3.0,
    -4.0,
    0.25,
    8.0,
    -0.125,
]
PRED_POS = [
    0.125,
    -16.743309020996094,
    1000.25,
    -0.0009765625,
    -85.5,
    0.3333333432674408,
    4096.0,
    -7.25,
]
PRED_NEG = [
    -0.25,
    124.7040023803711,
    -999.75,
    0.001953125,
    31.75,
    -0.6666666865348816,
    -2048.0,
    12.5,
]
POS_COND = [0.25, -0.5]
NEG_COND = [-0.75, 1.25]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32le_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        values.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()


def f32_bits(values: np.ndarray) -> list[int]:
    return values.astype(np.float32, copy=False).reshape(-1).view(np.uint32).tolist()


def tensor_payload(values: np.ndarray) -> dict[str, object]:
    materialized = values.astype(np.float32, copy=False)
    return {
        "shape": list(materialized.shape),
        "dtype": "float32",
        "values": materialized.tolist(),
        "f32le_sha256": f32le_sha256(materialized),
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_pins(trellis_root: Path) -> dict[str, Path]:
    actual_commit = subprocess.run(
        ["git", "-C", str(trellis_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != TRELLIS_PIN:
        raise RuntimeError(
            f"pinned TRELLIS.2 revision drift: expected {TRELLIS_PIN}, got {actual_commit}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(trellis_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("pinned TRELLIS.2 checkout must be clean")

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    expected = {"python": PYTHON_PIN, "torch": TORCH_PIN, "numpy": NUMPY_PIN}
    if versions != expected:
        raise RuntimeError(f"pinned package drift: expected {expected}, got {versions}")

    sampler_root = trellis_root / "trellis2/pipelines/samplers"
    sources = {
        "base": sampler_root / "base.py",
        "cfg": sampler_root / "classifier_free_guidance_mixin.py",
        "interval": sampler_root / "guidance_interval_mixin.py",
        "flow_euler": sampler_root / "flow_euler.py",
    }
    expected_hashes = {
        "base": "be8530b55ea66ac58e8ab23d650f463636dd52cf7e39c7c9c66f69bf72e6a0d1",
        "cfg": "1780182cd7d3c7af3f82b7904aa9c40eb3f632064d77456565c5acccd9598bee",
        "interval": "633f97c48a811a835d3b894b3e0de794407f60774f6c60a2c6a61e7c0351c6f2",
        "flow_euler": "b4bd235874adfc47fd3bce3d596249b1ccfa6644983a9b4562c9295a463bc0fd",
    }
    for name, path in sources.items():
        actual = sha256(path)
        if actual != expected_hashes[name]:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_hashes[name]}, got {actual}"
            )
    return sources


def load_cfg_sampler(trellis_root: Path):
    easydict = types.ModuleType("easydict")

    class EasyDict(dict):
        __getattr__ = dict.__getitem__

    easydict.EasyDict = EasyDict
    sys.modules["easydict"] = easydict

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

    package_paths = {
        "trellis2": trellis_root / "trellis2",
        "trellis2.pipelines": trellis_root / "trellis2/pipelines",
        "trellis2.pipelines.samplers": trellis_root / "trellis2/pipelines/samplers",
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    sampler_root = package_paths["trellis2.pipelines.samplers"]
    load_module("trellis2.pipelines.samplers.base", sampler_root / "base.py")
    load_module(
        "trellis2.pipelines.samplers.classifier_free_guidance_mixin",
        sampler_root / "classifier_free_guidance_mixin.py",
    )
    load_module(
        "trellis2.pipelines.samplers.guidance_interval_mixin",
        sampler_root / "guidance_interval_mixin.py",
    )
    flow_euler = load_module(
        "trellis2.pipelines.samplers.flow_euler", sampler_root / "flow_euler.py"
    )
    return flow_euler.FlowEulerCfgSampler


def scalar_source_mix(
    pred_pos: np.ndarray, pred_neg: np.ndarray, strength: float
) -> np.ndarray:
    """Independent source-order scalar F32 reference."""
    pos_scale = np.float32(strength)
    neg_scale = np.float32(1.0 - strength)
    output = np.empty_like(pred_pos, dtype=np.float32)
    pos_flat = pred_pos.reshape(-1)
    neg_flat = pred_neg.reshape(-1)
    out_flat = output.reshape(-1)
    for index in range(pos_flat.size):
        positive = np.float32(pos_scale * np.float32(pos_flat[index]))
        negative = np.float32(neg_scale * np.float32(neg_flat[index]))
        out_flat[index] = np.float32(positive + negative)
    return output


def build_fixture(trellis_root: Path, generator_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    sources = require_pins(trellis_root)
    sampler_class = load_cfg_sampler(trellis_root)
    sampler = sampler_class(1e-5)

    x_t = torch.tensor(X_T, dtype=torch.float32).reshape(SHAPE)
    pred_pos = torch.tensor(PRED_POS, dtype=torch.float32).reshape(SHAPE)
    pred_neg = torch.tensor(PRED_NEG, dtype=torch.float32).reshape(SHAPE)
    pos_cond = torch.tensor(POS_COND, dtype=torch.float32)
    neg_cond = torch.tensor(NEG_COND, dtype=torch.float32)
    originals = [value.clone() for value in (x_t, pred_pos, pred_neg, pos_cond, neg_cond)]

    def execute_case(name: str, strength: float) -> dict[str, object]:
        calls: list[dict[str, object]] = []

        def model(actual_x, model_t, actual_cond, **kwargs):
            if kwargs:
                raise AssertionError(f"unexpected model kwargs: {kwargs}")
            if actual_x is not x_t:
                raise AssertionError("CFG path replaced x_t before model inference")
            if actual_cond is pos_cond:
                condition = "positive"
                prediction = pred_pos
            elif actual_cond is neg_cond:
                condition = "negative"
                prediction = pred_neg
            else:
                raise AssertionError("CFG path replaced the condition carrier")
            if list(model_t.shape) != [SHAPE[0]]:
                raise AssertionError(f"unexpected model timestep shape: {model_t.shape}")
            if model_t.dtype != torch.float32 or model_t.device.type != "cpu":
                raise AssertionError("model timestep must be CPU Float32")
            calls.append(
                {
                    "index": len(calls),
                    "condition": condition,
                    "x_t_identity_preserved": True,
                    "model_t": tensor_payload(model_t.detach().cpu().numpy()),
                    "prediction": tensor_payload(prediction.detach().cpu().numpy()),
                }
            )
            return prediction

        output = sampler._inference_model(
            model,
            x_t,
            T,
            pos_cond,
            neg_cond,
            guidance_strength=strength,
            guidance_rescale=0.0,
        )
        return {
            "name": name,
            "guidance_strength": strength,
            "call_count": len(calls),
            "call_order": [call["condition"] for call in calls],
            "calls": calls,
            "output": tensor_payload(output.detach().cpu().numpy()),
        }

    cases = [
        execute_case("positive_only", 1.0),
        execute_case("negative_only", 0.0),
        execute_case("mixed", MIXED_STRENGTH),
    ]
    expected_orders = [["positive"], ["negative"], ["positive", "negative"]]
    for case, expected_order in zip(cases, expected_orders):
        if case["call_order"] != expected_order:
            raise AssertionError(
                f"source CFG call-order drift for {case['name']}: {case['call_order']}"
            )

    mixed = np.asarray(cases[2]["output"]["values"], dtype=np.float32)
    scalar_mixed = scalar_source_mix(
        pred_pos.detach().cpu().numpy(),
        pred_neg.detach().cpu().numpy(),
        MIXED_STRENGTH,
    )
    if not np.array_equal(mixed, scalar_mixed):
        raise AssertionError("independent source-order scalar reference drift")

    reassociated = (
        pred_neg + MIXED_STRENGTH * (pred_pos - pred_neg)
    ).detach().cpu().numpy()
    differing = np.flatnonzero(mixed.reshape(-1) != reassociated.reshape(-1))
    if differing.size == 0:
        raise AssertionError("fixture failed to expose reassociated CFG arithmetic")

    for current, original in zip(
        (x_t, pred_pos, pred_neg, pos_cond, neg_cond), originals
    ):
        if not torch.equal(current, original):
            raise AssertionError("source CFG path mutated a fixture input")

    source_payload = {
        name: {"path": str(path.relative_to(trellis_root)), "sha256": sha256(path)}
        for name, path in sources.items()
    }
    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "https://github.com/microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": "cpu",
            "dtype": "float32 state, predictions, model timestep, and CFG output",
            "weights": "none",
            "network": "none",
            "generator": GENERATOR_REPO_PATH,
            "generator_sha256": sha256(generator_path),
            "sources": source_payload,
        },
        "contract": {
            "owner": "ClassifierFreeGuidanceSamplerMixin._inference_model",
            "entrypoint": "trellis2.pipelines.samplers.flow_euler.FlowEulerCfgSampler._inference_model",
            "mro": [owner.__name__ for owner in sampler_class.__mro__],
            "branch_1": "guidance_strength == 1 calls only the positive condition",
            "branch_0": "guidance_strength == 0 calls only the negative condition",
            "branch_mixed": "positive call, then negative call, then strength*positive + (1-strength)*negative",
            "model_timestep": "each model call receives float32[batch] = 1000 * normalized_t",
            "condition_carriers": "opaque positive and negative objects forwarded by identity",
            "guidance_rescale": "explicitly 0.0; standard-deviation rescale branch not executed",
            "arithmetic_scope": "one CPU/F32 fixture seals source order, not general cross-platform parity",
        },
        "fixture": {
            "shape": SHAPE,
            "normalized_t": T,
            "expected_model_t_f32": float(np.float32(1000.0 * T)),
            "mixed_guidance_strength": MIXED_STRENGTH,
        },
        "inputs": {
            "x_t": tensor_payload(x_t.detach().cpu().numpy()),
            "pred_pos": tensor_payload(pred_pos.detach().cpu().numpy()),
            "pred_neg": tensor_payload(pred_neg.detach().cpu().numpy()),
            "pos_cond": tensor_payload(pos_cond.detach().cpu().numpy()),
            "neg_cond": tensor_payload(neg_cond.detach().cpu().numpy()),
            "all_unchanged": True,
        },
        "cases": cases,
        "independent_scalar_reference": {
            "formula": "f32(f32(f32(strength)*positive) + f32(f32(1-strength)*negative))",
            "output": tensor_payload(scalar_mixed),
            "max_abs_error": float(np.max(np.abs(mixed - scalar_mixed))),
        },
        "counterfactual_reassociated_mix": {
            "formula": "negative + strength*(positive-negative)",
            "output": tensor_payload(reassociated),
            "differing_element_count": int(differing.size),
            "first_differing_flat_index": int(differing[0]),
            "source_f32_bits": f32_bits(mixed),
            "counterfactual_f32_bits": f32_bits(reassociated),
            "max_abs_difference": float(np.max(np.abs(mixed - reassociated))),
        },
        "rejected_scope": [
            "Crystal CFG runtime or public API",
            "guidance interval or standard-deviation guidance rescale execution",
            "Euler step or repeated sampler execution",
            "random-noise or cross-device seed reproducibility",
            "production defaults, pipeline, sparse-structure flow, or decoder",
            "real weights, model scale, BF16, F16, quantization, GPU, or Metal",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.trellis_root.resolve(), Path(__file__).resolve())
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
