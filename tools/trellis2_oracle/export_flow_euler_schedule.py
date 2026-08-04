#!/usr/bin/env python3
"""Export a source-pinned repeated TRELLIS.2 flow Euler CPU oracle.

This executes the real upstream ``FlowEulerSampler.sample`` with a tiny,
deterministic pointwise model.  The fixture isolates the NumPy Float64 schedule
from the Float32 model-timestep tensor and state tensors.  It deliberately does
not add a Crystal sampler loop or execute CFG, RNG, weights, GPU, or Metal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from export_flow_euler_step import (
    f32le_sha256,
    load_flow_euler_sampler,
    require_pins,
    sha256,
)


SCHEMA = "cogni-ml/trellis2/flow-euler-schedule-oracle/v1"
GENERATOR_REPO_PATH = "tools/trellis2_oracle/export_flow_euler_schedule.py"
SUPPORT_REPO_PATH = "tools/trellis2_oracle/export_flow_euler_step.py"

SHAPE = [2, 2]
STEPS = 3
RESCALE_T = 1.7
SIGMA_MIN = 0.125
MODEL_X_SCALE = 0.25
MODEL_T_SCALE = 1.0 / 1024.0

NOISE = [1.0, -2.0, 0.5, 3.0]
COND = [0.125, -0.25, 0.375, -0.5]


def f64le_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        values.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()


def f32_bits(value: float | np.float32) -> int:
    return int(np.asarray(np.float32(value)).view(np.uint32))


def f64_bits(value: float | np.float64) -> int:
    return int(np.asarray(np.float64(value)).view(np.uint64))


def f64_bits_hex(value: float | np.float64) -> str:
    # JSON has no unsigned integer type.  A fixed-width string keeps all 64
    # bits readable by Crystal instead of routing high words through Float64.
    return f"0x{f64_bits(value):016x}"


def tensor_payload(values: np.ndarray) -> dict[str, object]:
    payload = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(payload.shape),
        "dtype": "float32",
        "values": payload.tolist(),
        "f32le_sha256": f32le_sha256(payload),
    }


def schedule_payload(values: np.ndarray) -> dict[str, object]:
    payload = np.asarray(values, dtype=np.float64)
    return {
        "dtype": "float64",
        "values": payload.tolist(),
        "f64_bits_hex": [f64_bits_hex(value) for value in payload],
        "f64le_sha256": f64le_sha256(payload),
    }


def source_schedule() -> np.ndarray:
    values = np.linspace(1, 0, STEPS + 1, dtype=np.float64)
    return RESCALE_T * values / (1 + (RESCALE_T - 1) * values)


def independent_schedule_reference() -> np.ndarray:
    # Construct the unscaled grid independently from np.linspace.  For this
    # bounded fixture it must agree bit-for-bit with the pinned source path.
    values = np.asarray(
        [1.0 - index / float(STEPS) for index in range(STEPS + 1)],
        dtype=np.float64,
    )
    return np.asarray(
        [
            RESCALE_T * value / (1.0 + (RESCALE_T - 1.0) * value)
            for value in values
        ],
        dtype=np.float64,
    )


def scalar_velocity(
    x_t: np.ndarray,
    model_t: np.float32,
    cond: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(x_t, dtype=np.float32)
    for index in np.ndindex(x_t.shape):
        x_term = np.float32(np.float32(x_t[index]) * np.float32(MODEL_X_SCALE))
        t_term = np.float32(model_t * np.float32(MODEL_T_SCALE))
        result[index] = np.float32(
            np.float32(x_term + t_term) + np.float32(cond[index])
        )
    return result


def scalar_step(
    x_t: np.ndarray,
    pred_v: np.ndarray,
    t: np.float64,
    t_prev: np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    dt = np.float32(np.float64(t) - np.float64(t_prev))
    one_minus_sigma = np.float32(1.0 - SIGMA_MIN)
    noise_scale = np.float32(
        SIGMA_MIN + (1.0 - SIGMA_MIN) * np.float64(t)
    )
    pred_x_prev = np.empty_like(x_t, dtype=np.float32)
    pred_x_0 = np.empty_like(x_t, dtype=np.float32)
    for index in np.ndindex(x_t.shape):
        x = np.float32(x_t[index])
        velocity = np.float32(pred_v[index])
        pred_x_prev[index] = np.float32(
            x - np.float32(dt * velocity)
        )
        pred_x_0[index] = np.float32(
            np.float32(one_minus_sigma * x)
            - np.float32(noise_scale * velocity)
        )
    return pred_x_prev, pred_x_0


def independent_scalar_reference(
    noise: np.ndarray,
    cond: np.ndarray,
    schedule: np.ndarray,
) -> tuple[list[dict[str, np.ndarray | np.float32]], np.ndarray]:
    state = noise.copy()
    steps: list[dict[str, np.ndarray | np.float32]] = []
    for t, t_prev in zip(schedule[:-1], schedule[1:]):
        model_t = np.float32(1000.0 * np.float64(t))
        velocity = scalar_velocity(state, model_t, cond)
        pred_x_prev, pred_x_0 = scalar_step(state, velocity, t, t_prev)
        steps.append(
            {
                "x_t": state.copy(),
                "model_t": model_t,
                "pred_v": velocity,
                "pred_x_prev": pred_x_prev.copy(),
                "pred_x_0": pred_x_0.copy(),
            }
        )
        state = pred_x_prev
    return steps, state


def build_fixture(trellis_root: Path, generator_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    flow_euler_sampler = load_flow_euler_sampler(trellis_root)

    schedule = source_schedule()
    independent_schedule = independent_schedule_reference()
    if not np.array_equal(schedule, independent_schedule):
        raise AssertionError("independent Float64 schedule reference drift")

    noise = torch.tensor(NOISE, dtype=torch.float32).reshape(SHAPE)
    cond = torch.tensor(COND, dtype=torch.float32).reshape(SHAPE)
    noise_before = noise.clone()
    cond_before = cond.clone()
    x_refs: list[torch.Tensor] = []
    cond_refs: list[torch.Tensor] = []
    calls: list[dict[str, object]] = []

    def pointwise_model(actual_x, model_t, actual_cond, **kwargs):
        if kwargs:
            raise AssertionError(f"unexpected model kwargs: {kwargs}")
        if actual_cond is not cond:
            raise AssertionError("upstream sampler replaced the condition object")
        if list(model_t.shape) != [SHAPE[0]]:
            raise AssertionError(f"unexpected model timestep shape: {model_t.shape}")
        if model_t.dtype != torch.float32 or model_t.device.type != "cpu":
            raise AssertionError("model timestep must be CPU Float32")
        velocity = (
            actual_x * MODEL_X_SCALE
            + model_t.reshape(SHAPE[0], 1) * MODEL_T_SCALE
            + actual_cond
        )
        x_refs.append(actual_x)
        cond_refs.append(actual_cond)
        calls.append(
            {
                "x_t": tensor_payload(actual_x.detach().cpu().numpy()),
                "model_t": tensor_payload(model_t.detach().cpu().numpy()),
                "pred_v": tensor_payload(velocity.detach().cpu().numpy()),
            }
        )
        return velocity

    sampler = flow_euler_sampler(SIGMA_MIN)
    source = sampler.sample(
        pointwise_model,
        noise,
        cond,
        steps=STEPS,
        rescale_t=RESCALE_T,
        verbose=False,
    )
    if len(calls) != STEPS:
        raise AssertionError(f"expected {STEPS} model calls, got {len(calls)}")
    if x_refs[0] is not noise:
        raise AssertionError("first sampler call did not receive noise by identity")
    for index in range(1, STEPS):
        if x_refs[index] is not source.pred_x_t[index - 1]:
            raise AssertionError("sampler did not transport prior output by identity")
    if any(actual_cond is not cond for actual_cond in cond_refs):
        raise AssertionError("condition identity changed between sampler steps")
    if source.samples is not source.pred_x_t[-1]:
        raise AssertionError("final sample does not alias the last recorded step")
    if not torch.equal(noise, noise_before) or not torch.equal(cond, cond_before):
        raise AssertionError("source sampler mutated fixture inputs")

    reference_steps, reference_final = independent_scalar_reference(
        noise.detach().cpu().numpy(),
        cond.detach().cpu().numpy(),
        independent_schedule,
    )
    step_payloads: list[dict[str, object]] = []
    max_reference_error = 0.0
    for index, (t, t_prev) in enumerate(zip(schedule[:-1], schedule[1:])):
        source_prev = source.pred_x_t[index].detach().cpu().numpy()
        source_x_0 = source.pred_x_0[index].detach().cpu().numpy()
        reference = reference_steps[index]
        reference_prev = np.asarray(reference["pred_x_prev"], dtype=np.float32)
        reference_x_0 = np.asarray(reference["pred_x_0"], dtype=np.float32)
        if not np.array_equal(source_prev, reference_prev):
            raise AssertionError(f"scalar pred_x_prev drift at step {index}")
        if not np.array_equal(source_x_0, reference_x_0):
            raise AssertionError(f"scalar pred_x_0 drift at step {index}")
        expected_model_t = np.float32(1000.0 * np.float64(t))
        actual_model_t = np.asarray(calls[index]["model_t"]["values"], dtype=np.float32)
        if not np.array_equal(
            actual_model_t,
            np.full([SHAPE[0]], expected_model_t, dtype=np.float32),
        ):
            raise AssertionError(f"model timestep drift at step {index}")
        max_reference_error = max(
            max_reference_error,
            float(np.max(np.abs(source_prev - reference_prev))),
            float(np.max(np.abs(source_x_0 - reference_x_0))),
        )
        step_payloads.append(
            {
                "index": index,
                "t": float(t),
                "t_f64_bits_hex": f64_bits_hex(t),
                "t_prev": float(t_prev),
                "t_prev_f64_bits_hex": f64_bits_hex(t_prev),
                "dt_f32_after_f64_subtract": float(np.float32(t - t_prev)),
                "dt_f32_bits": f32_bits(np.float32(t - t_prev)),
                "x_identity": "initial_noise" if index == 0 else "prior_pred_x_prev",
                "cond_identity_preserved": True,
                "x_t": calls[index]["x_t"],
                "model_t": calls[index]["model_t"],
                "pred_v": calls[index]["pred_v"],
                "pred_x_prev": tensor_payload(source_prev),
                "pred_x_0": tensor_payload(source_x_0),
            }
        )

    source_final = source.samples.detach().cpu().numpy()
    if not np.array_equal(source_final, reference_final):
        raise AssertionError("independent final sample drift")

    narrowed_schedule = schedule.astype(np.float32).astype(np.float64)
    narrowed_state = noise.clone()

    def unrecorded_model(actual_x, model_t, actual_cond, **kwargs):
        if kwargs or actual_cond is not cond:
            raise AssertionError("counterfactual model boundary drift")
        return (
            actual_x * MODEL_X_SCALE
            + model_t.reshape(SHAPE[0], 1) * MODEL_T_SCALE
            + actual_cond
        )

    for t, t_prev in zip(narrowed_schedule[:-1], narrowed_schedule[1:]):
        narrowed_state = sampler.sample_once(
            unrecorded_model,
            narrowed_state,
            float(t),
            float(t_prev),
            cond,
        ).pred_x_prev
    narrowed_final = narrowed_state.detach().cpu().numpy()
    differing = np.flatnonzero(source_final.reshape(-1) != narrowed_final.reshape(-1))
    if differing.size == 0:
        raise AssertionError("fixture failed to expose pre-narrowed F32 schedule drift")

    dt_comparison = []
    separating_pairs = 0
    for index, (t, t_prev) in enumerate(zip(schedule[:-1], schedule[1:])):
        source_dt = np.float32(t - t_prev)
        narrowed_dt = np.float32(t) - np.float32(t_prev)
        differs = f32_bits(source_dt) != f32_bits(narrowed_dt)
        separating_pairs += int(differs)
        dt_comparison.append(
            {
                "index": index,
                "source_f32_after_f64_subtract": float(source_dt),
                "source_f32_bits": f32_bits(source_dt),
                "pre_narrowed_f32_subtract": float(narrowed_dt),
                "pre_narrowed_f32_bits": f32_bits(narrowed_dt),
                "differs": differs,
            }
        )
    if separating_pairs < 1:
        raise AssertionError("fixture schedule does not separate subtraction order")

    source_payload = {
        name: {
            "path": str(path.relative_to(trellis_root)),
            "sha256": sha256(path),
        }
        for name, path in sources.items()
    }
    support_path = generator_path.with_name("export_flow_euler_step.py")
    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "https://github.com/microsoft/TRELLIS.2",
            "commit": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": "cpu",
            "dtype": "float32 state/model timestep; float64 schedule",
            "weights": "none",
            "network": "none",
            "generator": GENERATOR_REPO_PATH,
            "generator_sha256": sha256(generator_path),
            "support_module": SUPPORT_REPO_PATH,
            "support_module_sha256": sha256(support_path),
            "sources": source_payload,
        },
        "contract": {
            "owner": "FlowEulerSampler.sample",
            "entrypoint": "trellis2.pipelines.samplers.flow_euler.FlowEulerSampler.sample",
            "schedule": "np.linspace(1,0,steps+1) in float64, then rescale_t*t/(1+(rescale_t-1)*t)",
            "model_timestep": "each float64 normalized t is materialized as float32[batch] = 1000*t",
            "state_scalar_boundary": "t/t_prev remain Python float64 until source scalar expressions meet float32 tensors",
            "model_formula": "pred_v = x_t*0.25 + model_t*(1/1024) + cond",
            "condition_carrier": "same opaque condition object on every call",
            "state_transport": "first call receives noise; later calls receive prior pred_x_prev by identity",
            "rng": "none; initial noise is explicit",
            "progress_display": "tqdm wrapper replaced by an identity iterator; schedule iteration is unchanged",
            "fixture_purpose": "exposes endpoint pre-narrowing; the dt table separately isolates subtraction order",
            "counterfactual_effect": "pre-narrowing changes both model-time materialization and state coefficients",
            "production_defaults": "not claimed; steps=3 and rescale_t=1.7 are separating test values",
        },
        "fixture": {
            "shape": SHAPE,
            "steps": STEPS,
            "rescale_t": RESCALE_T,
            "sigma_min": SIGMA_MIN,
            "model_x_scale": MODEL_X_SCALE,
            "model_t_scale": MODEL_T_SCALE,
        },
        "inputs": {
            "noise": tensor_payload(noise.detach().cpu().numpy()),
            "cond": tensor_payload(cond.detach().cpu().numpy()),
            "noise_unchanged": True,
            "cond_unchanged": True,
        },
        "schedule": schedule_payload(schedule),
        "model_probe": {
            "call_count": len(calls),
            "same_condition_every_call": True,
            "first_x_is_noise": True,
            "later_x_is_prior_output": True,
            "final_is_last_pred_x_prev": True,
        },
        "steps": step_payloads,
        "expected": {
            "samples": tensor_payload(source_final),
            "pred_x_t_count": len(source.pred_x_t),
            "pred_x_0_count": len(source.pred_x_0),
        },
        "independent_scalar_reference": {
            "schedule_f64le_sha256": f64le_sha256(independent_schedule),
            "max_abs_error": max_reference_error,
            "samples_f32le_sha256": f32le_sha256(reference_final),
        },
        "counterfactual_pre_narrowed_f32_schedule": {
            "schedule_values": narrowed_schedule.tolist(),
            "schedule_f32_bits": [f32_bits(value) for value in schedule],
            "dt_comparison": dt_comparison,
            "separating_pair_count": separating_pairs,
            "samples": tensor_payload(narrowed_final),
            "differing_element_count": int(differing.size),
            "first_differing_flat_index": int(differing[0]),
            "max_abs_difference": float(np.max(np.abs(source_final - narrowed_final))),
        },
        "rejected_scope": [
            "Crystal repeated-step sampler runtime or schedule API",
            "CFG, guidance interval, or guidance rescale execution",
            "random-noise or cross-device seed reproducibility",
            "production sampler defaults, pipeline, sparse-structure flow, or decoder",
            "aggregate process, stage, retained, RSS, native, or peak memory",
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
        json.dumps(fixture, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
