#!/usr/bin/env python3
"""Export a source-pinned CPU oracle for one TRELLIS.2 flow Euler step.

This executes the real upstream ``FlowEulerSampler.sample_once`` with a tiny
fixed-velocity model.  It deliberately excludes the multi-step scheduler, CFG,
random noise generation, model weights, and every GPU backend.
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
SCHEMA = "cogni-ml/trellis2/flow-euler-step-oracle/v1"
GENERATOR_REPO_PATH = "tools/trellis2_oracle/export_flow_euler_step.py"

SHAPE = [2, 3, 2, 2]
SIGMA_MIN = 0.125
PRODUCTION_SIGMA_MIN = 1e-5
T = 0.75
T_PREV = 0.25
TOLERANCE = 1e-6

X_T = [
    -1.5, -0.75, 0.25, 1.25,
    0.5, 1.75, -1.0, 0.125,
    2.0, -0.25, 0.875, -1.25,
    -0.375, 1.5, -2.0, 0.625,
    1.125, -0.875, 0.375, 2.25,
    -1.75, 0.75, -0.125, 1.0,
]
VELOCITY = [
    0.25, -0.5, 1.0, -1.5,
    0.75, 0.125, -0.25, 2.0,
    -1.0, 0.5, 1.25, -0.75,
    1.5, -1.25, 0.375, 0.625,
    -0.125, 1.75, -2.0, 0.875,
    0.5, -0.375, 1.0, -1.5,
]
COND_SHAPE = [2, 2, 3]
COND = [
    0.5, -1.0, 1.5,
    2.0, -0.25, 0.75,
    -1.5, 0.125, 1.0,
    0.375, 2.25, -0.75,
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32le_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        values.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_flow_euler_sampler(trellis_root: Path):
    # Avoid importing TRELLIS.2's model package.  These two stubs replace only
    # progress/display helpers that sample_once does not semantically use.
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
    return flow_euler.FlowEulerSampler


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

    sources = {
        "base": trellis_root / "trellis2/pipelines/samplers/base.py",
        "flow_euler": trellis_root / "trellis2/pipelines/samplers/flow_euler.py",
        "cfg": trellis_root / "trellis2/pipelines/samplers/classifier_free_guidance_mixin.py",
        "interval": trellis_root / "trellis2/pipelines/samplers/guidance_interval_mixin.py",
        "pipeline": trellis_root / "trellis2/pipelines/trellis2_image_to_3d.py",
        "trainer": trellis_root / "trellis2/trainers/flow_matching/flow_matching.py",
        "production_config": trellis_root / "configs/gen/ss_flow_img_dit_1_3B_64_bf16.json",
    }
    expected_hashes = {
        "base": "be8530b55ea66ac58e8ab23d650f463636dd52cf7e39c7c9c66f69bf72e6a0d1",
        "flow_euler": "b4bd235874adfc47fd3bce3d596249b1ccfa6644983a9b4562c9295a463bc0fd",
        "cfg": "1780182cd7d3c7af3f82b7904aa9c40eb3f632064d77456565c5acccd9598bee",
        "interval": "633f97c48a811a835d3b894b3e0de794407f60774f6c60a2c6a61e7c0351c6f2",
        "pipeline": "e2addfca672354284b23d1541a8f49228d5a727d49220fcb8512cca2cdd38ce9",
        "trainer": "09da19de08fc95315d3588d7ca078128e03914d692f20fa2a2ba58038c0cb563",
        "production_config": "6128e63a7bd77db798c649d08fd05ac6b86f3fa7a0d4a405008ccf6cf29945c0",
    }
    for name, path in sources.items():
        actual = sha256(path)
        if actual != expected_hashes[name]:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_hashes[name]}, got {actual}"
            )
    return sources


def scalar_f32_reference(
    x_t: np.ndarray,
    velocity: np.ndarray,
    sigma_min: float,
    t: float,
    t_prev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent scalar loop with explicit F32 state arithmetic."""
    sigma = np.float32(sigma_min)
    raw_t = np.float32(t)
    dt = np.float32(t - t_prev)
    one_minus_sigma = np.float32(np.float32(1.0) - sigma)
    noise_scale = np.float32(sigma + np.float32(one_minus_sigma * raw_t))
    x_prev = np.empty_like(x_t, dtype=np.float32)
    x_0 = np.empty_like(x_t, dtype=np.float32)
    x_flat = x_t.reshape(-1)
    v_flat = velocity.reshape(-1)
    prev_flat = x_prev.reshape(-1)
    x0_flat = x_0.reshape(-1)
    for index in range(x_flat.size):
        x = np.float32(x_flat[index])
        v = np.float32(v_flat[index])
        prev_flat[index] = np.float32(x - np.float32(dt * v))
        x0_flat[index] = np.float32(
            np.float32(one_minus_sigma * x) - np.float32(noise_scale * v)
        )
    return x_prev, x_0


def tensor_payload(values: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(values.shape),
        "dtype": "float32",
        "values": values.tolist(),
        "f32le_sha256": f32le_sha256(values),
    }


def build_fixture(trellis_root: Path, generator_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    flow_euler_sampler = load_flow_euler_sampler(trellis_root)

    x_t = torch.tensor(X_T, dtype=torch.float32).reshape(SHAPE)
    velocity = torch.tensor(VELOCITY, dtype=torch.float32).reshape(SHAPE)
    cond = torch.tensor(COND, dtype=torch.float32).reshape(COND_SHAPE)
    calls: list[dict[str, object]] = []

    def fixed_model(actual_x, model_t, actual_cond, **kwargs):
        if actual_x is not x_t:
            raise AssertionError("upstream sampler did not forward x_t by identity")
        if actual_cond is not cond:
            raise AssertionError("upstream sampler did not forward cond by identity")
        if kwargs:
            raise AssertionError(f"unexpected model kwargs: {kwargs}")
        calls.append(
            {
                "x_identity_preserved": True,
                "cond_identity_preserved": True,
                "t_shape": list(model_t.shape),
                "t_dtype": str(model_t.dtype).removeprefix("torch."),
                "t_device": str(model_t.device),
                "t_values": model_t.detach().cpu().tolist(),
            }
        )
        return velocity.clone()

    sampler = flow_euler_sampler(SIGMA_MIN)
    source = sampler.sample_once(fixed_model, x_t, T, T_PREV, cond)
    if len(calls) != 1:
        raise AssertionError(f"expected one model call, got {len(calls)}")
    if calls[0]["t_values"] != [1000.0 * T] * SHAPE[0]:
        raise AssertionError("model timestep scaling drift")

    x_np = x_t.numpy()
    velocity_np = velocity.numpy()
    scalar_prev, scalar_x0 = scalar_f32_reference(
        x_np, velocity_np, SIGMA_MIN, T, T_PREV
    )
    source_prev = source.pred_x_prev.detach().cpu().numpy()
    source_x0 = source.pred_x_0.detach().cpu().numpy()
    max_prev_error = float(np.max(np.abs(source_prev - scalar_prev)))
    max_x0_error = float(np.max(np.abs(source_x0 - scalar_x0)))
    if max(max_prev_error, max_x0_error) > TOLERANCE:
        raise AssertionError(
            f"scalar reference drift: prev={max_prev_error}, x0={max_x0_error}"
        )

    source_payload = {
        name: {
            "path": str(path.relative_to(trellis_root)),
            "sha256": sha256(path),
        }
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
            "dtype": "float32",
            "weights": "none",
            "network": "none",
            "generator": GENERATOR_REPO_PATH,
            "generator_sha256": sha256(generator_path),
            "sources": source_payload,
        },
        "contract": {
            "owner": "FlowEulerSampler.sample_once",
            "entrypoint": "trellis2.pipelines.samplers.flow_euler.FlowEulerSampler.sample_once",
            "model_timestep": "float32[batch] = 1000 * normalized_t",
            "state_timestep": "fixture uses normalized scalar t in [0,1]",
            "source_validation": "shape equality only; no finite, range, or time-order validation",
            "arithmetic_scope": "binary-exact fixture coefficients; general precision drift remains unsealed",
            "x0_formula": "(1-sigma_min)*x_t - (sigma_min+(1-sigma_min)*t)*pred_v",
            "euler_formula": "x_t - (t-t_prev)*pred_v",
            "pred_eps_computed_but_not_returned": True,
            "condition_carrier": "opaque cond forwarded unchanged to model",
            "fixture_condition_shape_only": COND_SHAPE,
            "production_sigma_min": PRODUCTION_SIGMA_MIN,
            "rng_owner": "Trellis2ImageTo3DPipeline.run calls torch.manual_seed once; sample_sparse_structure then calls torch.randn",
            "initial_noise_in_fixture": "explicit x_t values; pipeline RNG not executed",
        },
        "inventory_not_executed": {
            "schedule": "np.linspace(1,0,steps+1), then rescale_t*t/(1+(rescale_t-1)*t)",
            "cfg": "strength 1 positive-only; 0 negative-only; otherwise strength*positive+(1-strength)*negative",
            "guidance_interval": "inclusive normalized-t interval; outside forces positive-only strength 1",
            "guidance_rescale": "non-special CFG branch only; per-batch torch.std default correction; no zero-std guard",
        },
        "fixture": {
            "shape": SHAPE,
            "sigma_min": SIGMA_MIN,
            "t": T,
            "t_prev": T_PREV,
            "model_t": 1000.0 * T,
        },
        "inputs": {
            "x_t": tensor_payload(x_np),
            "pred_v": tensor_payload(velocity_np),
            "cond": tensor_payload(cond.numpy()),
        },
        "model_probe": {
            "call_count": len(calls),
            "calls": calls,
        },
        "expected": {
            "pred_x_prev": tensor_payload(source_prev),
            "pred_x_0": tensor_payload(source_x0),
        },
        "independent_scalar_reference": {
            "max_abs_error_pred_x_prev": max_prev_error,
            "max_abs_error_pred_x_0": max_x0_error,
            "tolerance": TOLERANCE,
            "pred_x_prev_f32le_sha256": f32le_sha256(scalar_prev),
            "pred_x_0_f32le_sha256": f32le_sha256(scalar_x0),
        },
        "rejected_scope": [
            "sampler schedule loop or repeated-step drift",
            "CFG, guidance interval, or guidance rescale execution",
            "random-noise or cross-device seed reproducibility",
            "production sampler class or defaults from external pipeline.json",
            "sparse-structure flow model, decoder, or complete stage",
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
