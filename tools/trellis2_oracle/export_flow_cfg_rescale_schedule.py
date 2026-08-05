#!/usr/bin/env python3
"""Export the source-pinned repeated TRELLIS.2 CFG-rescale oracle.

The exporter executes the real FlowEulerCfgSampler.sample MRO on a tiny,
deterministic CPU/F32 fixture. It binds the Float64 schedule, per-step CFG
call order, x0-space guidance rescale, and Euler histories. It deliberately
does not add a Crystal sampler loop, load weights, use RNG, or touch an
accelerator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from export_flow_euler_cfg import load_cfg_sampler, require_pins, sha256


SCHEMA = "cogni-ml/trellis2/flow-cfg-rescale-schedule-oracle/v1"
GENERATOR_REPO_PATH = "tools/trellis2_oracle/export_flow_cfg_rescale_schedule.py"
SUPPORT_REPO_PATH = "tools/trellis2_oracle/export_flow_euler_cfg.py"

TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
SHAPE = [2, 2, 2]
STEPS = 3
RESCALE_T = 1.7
SIGMA_MIN = 1e-5
GUIDANCE_STRENGTH = 1.7
GUIDANCE_RESCALE = 0.35
MODEL_X_SCALE = 0.25
MODEL_T_SCALE = 1.0 / 1024.0

NOISE = [
    1.0,
    -2.0,
    0.5,
    3.0,
    -0.75,
    1.25,
    2.5,
    -3.5,
]
POS_COND = [
    0.125,
    -0.25,
    0.375,
    -0.5,
    0.625,
    -0.75,
    0.875,
    -1.0,
]
NEG_COND = [
    -0.5,
    0.25,
    -0.75,
    0.375,
    -1.25,
    0.625,
    -1.5,
    0.875,
]


def f32le_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def f64le_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def f32_bits(value: float | np.float32) -> int:
    return int(np.asarray(np.float32(value)).view(np.uint32))


def f64_bits(value: float | np.float64) -> int:
    return int(np.asarray(np.float64(value)).view(np.uint64))


def f64_bits_hex(value: float | np.float64) -> str:
    return f"0x{f64_bits(value):016x}"


def tensor_payload(value: torch.Tensor | np.ndarray) -> dict[str, object]:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise AssertionError(
                f"oracle tensor must remain CPU Float32, got {value.device}/{value.dtype}"
            )
        array = value.detach().cpu().numpy()
    else:
        array = value
    array = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "dtype": "float32",
        "values": array.tolist(),
        "f32le_sha256": f32le_sha256(array),
    }


def same_f32_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_bits = left.detach().cpu().numpy().astype(np.float32).view(np.uint32)
    right_bits = right.detach().cpu().numpy().astype(np.float32).view(np.uint32)
    return np.array_equal(left_bits, right_bits)


def scalar_sample_std(values: np.ndarray) -> np.ndarray:
    """Independent two-pass per-batch sample std with explicit F32 steps."""
    source = np.asarray(values, dtype=np.float32)
    output = np.empty((source.shape[0], 1, 1), dtype=np.float32)
    for batch in range(source.shape[0]):
        row = source[batch].reshape(-1)
        total = np.float32(0.0)
        for item in row:
            total = np.float32(total + np.float32(item))
        mean = np.float32(total / np.float32(row.size))
        squared = np.float32(0.0)
        for item in row:
            delta = np.float32(np.float32(item) - mean)
            squared = np.float32(squared + np.float32(delta * delta))
        variance = np.float32(squared / np.float32(row.size - 1))
        output[batch, 0, 0] = np.float32(np.sqrt(variance, dtype=np.float32))
    return output


def scalar_model(
    state: np.ndarray,
    model_t: np.float32,
    condition: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(state, dtype=np.float32)
    state_flat = np.asarray(state, dtype=np.float32).reshape(-1)
    condition_flat = np.asarray(condition, dtype=np.float32).reshape(-1)
    result_flat = result.reshape(-1)
    model_term = np.float32(model_t * np.float32(MODEL_T_SCALE))
    for index, value in enumerate(state_flat):
        x_term = np.float32(np.float32(value) * np.float32(MODEL_X_SCALE))
        result_flat[index] = np.float32(
            np.float32(x_term + model_term) + np.float32(condition_flat[index])
        )
    return result


def scalar_pred_to_x0(
    state: np.ndarray,
    normalized_t: np.float64,
    prediction: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(state, dtype=np.float32)
    one_minus_sigma = np.float32(1.0 - SIGMA_MIN)
    noise_scale = np.float32(
        SIGMA_MIN + (1.0 - SIGMA_MIN) * np.float64(normalized_t)
    )
    for index in np.ndindex(state.shape):
        x = np.float32(state[index])
        pred = np.float32(prediction[index])
        result[index] = np.float32(
            np.float32(one_minus_sigma * x)
            - np.float32(noise_scale * pred)
        )
    return result


def scalar_x0_to_pred(
    state: np.ndarray,
    normalized_t: np.float64,
    x0: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(state, dtype=np.float32)
    one_minus_sigma = np.float32(1.0 - SIGMA_MIN)
    noise_scale = np.float32(
        SIGMA_MIN + (1.0 - SIGMA_MIN) * np.float64(normalized_t)
    )
    for index in np.ndindex(state.shape):
        x = np.float32(state[index])
        numerator = np.float32(np.float32(one_minus_sigma * x) - x0[index])
        result[index] = np.float32(numerator / noise_scale)
    return result


def scalar_cfg_mix(
    positive: np.ndarray,
    negative: np.ndarray,
    strength: float,
) -> np.ndarray:
    result = np.empty_like(positive, dtype=np.float32)
    positive_scale = np.float32(strength)
    negative_scale = np.float32(1.0 - strength)
    for index in np.ndindex(result.shape):
        positive_term = np.float32(positive_scale * positive[index])
        negative_term = np.float32(negative_scale * negative[index])
        result[index] = np.float32(positive_term + negative_term)
    return result


def scalar_step_reference(
    noise: np.ndarray,
    positive_condition: np.ndarray,
    negative_condition: np.ndarray,
    schedule: np.ndarray,
    rescale_space: str = "x0",
) -> tuple[list[dict[str, np.ndarray | np.float32]], np.ndarray]:
    state = np.asarray(noise, dtype=np.float32).copy()
    steps: list[dict[str, np.ndarray | np.float32]] = []
    for t, t_prev in zip(schedule[:-1], schedule[1:]):
        model_t = np.float32(1000.0 * np.float64(t))
        pred_positive = scalar_model(state, model_t, positive_condition)
        pred_negative = scalar_model(state, model_t, negative_condition)
        pred_cfg = scalar_cfg_mix(pred_positive, pred_negative, GUIDANCE_STRENGTH)
        x0_positive = scalar_pred_to_x0(state, t, pred_positive)
        x0_cfg = scalar_pred_to_x0(state, t, pred_cfg)
        std_positive = scalar_sample_std(
            x0_positive if rescale_space == "x0" else pred_positive
        )
        std_cfg = scalar_sample_std(x0_cfg if rescale_space == "x0" else pred_cfg)
        if rescale_space == "x0":
            rescaled = np.empty_like(x0_cfg, dtype=np.float32)
            for index in np.ndindex(rescaled.shape):
                rescaled[index] = np.float32(
                    x0_cfg[index]
                    * np.float32(
                        std_positive[index[0], 0, 0] / std_cfg[index[0], 0, 0]
                    )
                )
            x0_blend = np.empty_like(x0_cfg, dtype=np.float32)
            for index in np.ndindex(x0_blend.shape):
                x0_blend[index] = np.float32(
                    np.float32(GUIDANCE_RESCALE) * rescaled[index]
                    + np.float32(1.0 - GUIDANCE_RESCALE) * x0_cfg[index]
                )
            prediction = scalar_x0_to_pred(state, t, x0_blend)
        elif rescale_space == "prediction":
            rescaled = np.empty_like(pred_cfg, dtype=np.float32)
            for index in np.ndindex(rescaled.shape):
                rescaled[index] = np.float32(
                    pred_cfg[index]
                    * np.float32(
                        std_positive[index[0], 0, 0] / std_cfg[index[0], 0, 0]
                    )
                )
            prediction = np.empty_like(pred_cfg, dtype=np.float32)
            for index in np.ndindex(prediction.shape):
                prediction[index] = np.float32(
                    np.float32(GUIDANCE_RESCALE) * rescaled[index]
                    + np.float32(1.0 - GUIDANCE_RESCALE) * pred_cfg[index]
                )
            x0_blend = scalar_pred_to_x0(state, t, prediction)
        else:
            raise ValueError(f"unknown rescale space: {rescale_space}")

        dt = np.float32(np.float64(t) - np.float64(t_prev))
        pred_x_prev = np.empty_like(state, dtype=np.float32)
        for index in np.ndindex(state.shape):
            pred_x_prev[index] = np.float32(
                np.float32(state[index]) - np.float32(dt * prediction[index])
            )
        pred_x_0 = scalar_pred_to_x0(state, t, prediction)
        steps.append(
            {
                "x_t": state.copy(),
                "model_t": model_t,
                "pred_positive": pred_positive,
                "pred_negative": pred_negative,
                "pred_cfg": pred_cfg,
                "x0_positive": x0_positive,
                "x0_cfg": x0_cfg,
                "std_positive": std_positive,
                "std_cfg": std_cfg,
                "rescaled": rescaled,
                "x0_blend": x0_blend,
                "pred_v": prediction,
                "pred_x_prev": pred_x_prev,
                "pred_x_0": pred_x_0,
            }
        )
        state = pred_x_prev
    return steps, state


def source_schedule() -> np.ndarray:
    values = np.linspace(1, 0, STEPS + 1, dtype=np.float64)
    return RESCALE_T * values / (1 + (RESCALE_T - 1) * values)


def build_fixture(trellis_root: Path, generator_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    sources = require_pins(trellis_root)
    sampler_base = load_cfg_sampler(trellis_root)

    class TracedSampler(sampler_base):
        def __init__(self, sigma_min: float):
            super().__init__(sigma_min)
            self.current_step = -1
            self.records: list[dict[str, object]] = []
            self.internal: list[dict[str, object]] = []
            self.expected_state: torch.Tensor | None = None
            self.trace_positive: torch.Tensor | None = None

        def sample_once(self, model, x_t, t, t_prev, cond=None, **kwargs):
            self.current_step += 1
            self.expected_state = x_t
            self.trace_positive = None
            self.records.append(
                {
                    "index": self.current_step,
                    "x_identity": (
                        "initial_noise"
                        if self.current_step == 0
                        else "prior_pred_x_prev"
                    ),
                    "calls": [],
                    "conversion_trace": [],
                    "conversion_times": [],
                }
            )
            self.internal.append({"x_t": x_t})
            output = super().sample_once(model, x_t, t, t_prev, cond, **kwargs)
            self.records[-1]["pred_x_prev"] = tensor_payload(output.pred_x_prev)
            self.records[-1]["pred_x_0"] = tensor_payload(output.pred_x_0)
            self.internal[-1]["pred_x_prev"] = output.pred_x_prev
            self.internal[-1]["pred_x_0"] = output.pred_x_0
            return output

        def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
            output = super()._get_model_prediction(model, x_t, t, cond, **kwargs)
            self.records[-1]["pred_v"] = tensor_payload(output[2])
            self.internal[-1]["pred_v"] = output[2]
            return output

        def _pred_to_xstart(self, x_t, t, pred):
            role = "positive" if pred is self.trace_positive else "cfg"
            output = super()._pred_to_xstart(x_t, t, pred)
            self.records[-1]["conversion_trace"].append(
                f"pred_to_xstart:{role}"
            )
            self.records[-1]["conversion_times"].append(float(t))
            self.internal[-1][f"pred_{role}"] = pred
            self.internal[-1][f"x0_{role}"] = output
            return output

        def _xstart_to_pred(self, x_t, t, x_0):
            self.records[-1]["conversion_trace"].append("xstart_to_pred:blend")
            self.records[-1]["conversion_times"].append(float(t))
            self.internal[-1]["x0_blend"] = x_0
            return super()._xstart_to_pred(x_t, t, x_0)

    noise = torch.tensor(NOISE, dtype=torch.float32).reshape(SHAPE)
    positive_condition = torch.tensor(POS_COND, dtype=torch.float32).reshape(SHAPE)
    negative_condition = torch.tensor(NEG_COND, dtype=torch.float32).reshape(SHAPE)
    noise_before = noise.clone()
    positive_before = positive_condition.clone()
    negative_before = negative_condition.clone()
    sampler = TracedSampler(SIGMA_MIN)

    def pointwise_model(actual_x, model_t, actual_cond, **kwargs):
        if kwargs:
            raise AssertionError(f"unexpected model kwargs: {kwargs}")
        if sampler.expected_state is None or actual_x is not sampler.expected_state:
            raise AssertionError("sampler replaced the state before model inference")
        if actual_cond is positive_condition:
            condition = "positive"
            condition_tensor = positive_condition
        elif actual_cond is negative_condition:
            condition = "negative"
            condition_tensor = negative_condition
        else:
            raise AssertionError("sampler replaced the condition carrier")
        if model_t.dtype != torch.float32 or model_t.device.type != "cpu":
            raise AssertionError("model timestep must be CPU Float32")
        if list(model_t.shape) != [SHAPE[0]]:
            raise AssertionError(f"unexpected model timestep shape: {model_t.shape}")
        prediction = (
            actual_x * MODEL_X_SCALE
            + model_t.reshape(SHAPE[0], 1, 1) * MODEL_T_SCALE
            + condition_tensor
        )
        if condition == "positive":
            sampler.trace_positive = prediction
        sampler.records[-1]["calls"].append(
            {
                "index": len(sampler.records[-1]["calls"]),
                "condition": condition,
                "x_t_identity_preserved": True,
                "condition_identity_preserved": True,
                "model_t": tensor_payload(model_t),
                "prediction": tensor_payload(prediction),
            }
        )
        sampler.internal[-1][f"pred_{condition}"] = prediction
        return prediction

    schedule = source_schedule()
    with torch.no_grad():
        source = sampler.sample(
            pointwise_model,
            noise,
            positive_condition,
            negative_condition,
            steps=STEPS,
            rescale_t=RESCALE_T,
            guidance_strength=GUIDANCE_STRENGTH,
            guidance_rescale=GUIDANCE_RESCALE,
            verbose=False,
        )

    if len(sampler.records) != STEPS or len(sampler.internal) != STEPS:
        raise AssertionError("repeated CFG-rescale loop changed step count")
    for index, record in enumerate(sampler.records):
        if len(record["calls"]) != 2:
            raise AssertionError(f"step {index} did not make two CFG calls")
        if [call["condition"] for call in record["calls"]] != [
            "positive",
            "negative",
        ]:
            raise AssertionError(f"step {index} changed positive-negative order")
        if record["conversion_trace"] != [
            "pred_to_xstart:positive",
            "pred_to_xstart:cfg",
            "xstart_to_pred:blend",
        ]:
            raise AssertionError(f"step {index} changed x0-space conversion order")
        if any(value != float(schedule[index]) for value in record["conversion_times"]):
            raise AssertionError(f"step {index} changed normalized conversion time")
        internal = sampler.internal[index]
        x0_positive = internal["x0_positive"]
        x0_cfg = internal["x0_cfg"]
        std_positive = x0_positive.std(
            dim=list(range(1, x0_positive.ndim)), keepdim=True
        )
        std_cfg = x0_cfg.std(dim=list(range(1, x0_cfg.ndim)), keepdim=True)
        x0_rescaled = x0_cfg * (std_positive / std_cfg)
        x0_blend = GUIDANCE_RESCALE * x0_rescaled + (1 - GUIDANCE_RESCALE) * x0_cfg
        if not same_f32_bits(x0_blend, internal["x0_blend"]):
            raise AssertionError(f"step {index} source-order x0 blend reconstruction drift")
        record["intermediates"] = {
            "pred_positive": tensor_payload(internal["pred_positive"]),
            "pred_negative": tensor_payload(internal["pred_negative"]),
            "pred_cfg": tensor_payload(internal["pred_cfg"]),
            "x0_positive": tensor_payload(x0_positive),
            "x0_cfg": tensor_payload(x0_cfg),
            "std_positive": tensor_payload(std_positive),
            "std_cfg": tensor_payload(std_cfg),
            "x0_rescaled": tensor_payload(x0_rescaled),
            "x0_blend": tensor_payload(internal["x0_blend"]),
        }
        record["t"] = float(schedule[index])
        record["t_f64_bits_hex"] = f64_bits_hex(schedule[index])
        record["t_prev"] = float(schedule[index + 1])
        record["t_prev_f64_bits_hex"] = f64_bits_hex(schedule[index + 1])
        record["dt_f32_after_f64_subtract"] = float(
            np.float32(schedule[index] - schedule[index + 1])
        )
        record["dt_f32_bits"] = f32_bits(
            np.float32(schedule[index] - schedule[index + 1])
        )
        record["x_t"] = tensor_payload(internal["x_t"])
        record["pred_v"] = tensor_payload(internal["pred_v"])
        record["pred_x_prev"] = tensor_payload(internal["pred_x_prev"])
        record["pred_x_0"] = tensor_payload(internal["pred_x_0"])
        record["cond_identity_preserved"] = True

    if source.samples is not source.pred_x_t[-1]:
        raise AssertionError("final sample does not alias last Euler history entry")
    if not torch.equal(noise, noise_before):
        raise AssertionError("sampler mutated initial noise")
    if not torch.equal(positive_condition, positive_before):
        raise AssertionError("sampler mutated positive condition")
    if not torch.equal(negative_condition, negative_before):
        raise AssertionError("sampler mutated negative condition")
    for index in range(1, STEPS):
        if sampler.internal[index]["x_t"] is not source.pred_x_t[index - 1]:
            raise AssertionError(f"step {index} lost prior-state identity")

    noise_np = np.asarray(NOISE, dtype=np.float32).reshape(SHAPE)
    positive_np = np.asarray(POS_COND, dtype=np.float32).reshape(SHAPE)
    negative_np = np.asarray(NEG_COND, dtype=np.float32).reshape(SHAPE)
    reference_steps, reference_final = scalar_step_reference(
        noise_np,
        positive_np,
        negative_np,
        schedule,
        rescale_space="x0",
    )
    source_final = source.samples.detach().cpu().numpy()
    max_abs_error = 0.0
    for index, reference in enumerate(reference_steps):
        for name in ("pred_v", "pred_x_prev", "pred_x_0"):
            actual = np.asarray(
                sampler.internal[index][name].detach().cpu().numpy()
            )
            expected = np.asarray(reference[name], dtype=np.float32)
            max_abs_error = max(
                max_abs_error,
                float(np.max(np.abs(actual - expected))),
            )
    max_abs_error = max(
        max_abs_error,
        float(np.max(np.abs(source_final - reference_final))),
    )
    if max_abs_error > 2e-3:
        raise AssertionError(f"scalar repeated CFG-rescale reference drift: {max_abs_error}")

    narrowed_schedule = schedule.astype(np.float32).astype(np.float64)
    _, narrowed_final = scalar_step_reference(
        noise_np,
        positive_np,
        negative_np,
        narrowed_schedule,
        rescale_space="x0",
    )
    _, prediction_final = scalar_step_reference(
        noise_np,
        positive_np,
        negative_np,
        schedule,
        rescale_space="prediction",
    )
    narrowed_differing = np.flatnonzero(
        source_final.reshape(-1) != narrowed_final.reshape(-1)
    )
    prediction_differing = np.flatnonzero(
        source_final.reshape(-1) != prediction_final.reshape(-1)
    )
    if narrowed_differing.size == 0:
        raise AssertionError("endpoint pre-narrowing counterfactual no longer differs")
    if prediction_differing.size == 0:
        raise AssertionError("prediction-space rescale counterfactual no longer differs")

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
            "threads": 1,
            "dtype": "Float64 schedule/controls; Float32 state/model/CFG tensors",
            "weights": "none",
            "network": "none",
            "generator": GENERATOR_REPO_PATH,
            "generator_sha256": sha256(generator_path),
            "support": SUPPORT_REPO_PATH,
            "support_sha256": sha256(generator_path.parent / "export_flow_euler_cfg.py"),
            "sources": source_payload,
        },
        "contract": {
            "owner": "FlowEulerSampler.sample",
            "entrypoint": "trellis2.pipelines.samplers.flow_euler.FlowEulerCfgSampler.sample",
            "mro": [owner.__name__ for owner in sampler_base.__mro__],
            "schedule": "np.linspace(1,0,steps+1) in Float64, then rescale_t*t/(1+(rescale_t-1)*t)",
            "model_timestep": "each normalized Float64 t becomes fresh CPU Float32[batch] = 1000*t",
            "cfg": "each step calls positive then negative and mixes strength*positive + (1-strength)*negative",
            "rescale": "positive guidance_rescale converts mixed and positive prediction to x0, rescales per batch over non-batch axes, blends x0, then converts back to prediction",
            "euler": "pred_x_prev = x_t - (t-t_prev)*pred_v; pred_x_0 uses normalized Float64 t at the step boundary",
            "identity": "initial state and later prior pred_x_prev are forwarded by identity; positive/negative condition carriers remain identical",
            "fixture_purpose": "bind cross-step order before any Crystal loop composition",
            "counterfactuals": "endpoint pre-narrowing and prediction-space rescale both produce different final histories",
            "production_defaults": "not claimed; steps=3, rescale_t=1.7, guidance_strength=1.7, guidance_rescale=0.35 are separating values",
        },
        "fixture": {
            "shape": SHAPE,
            "steps": STEPS,
            "rescale_t": RESCALE_T,
            "sigma_min": SIGMA_MIN,
            "guidance_strength": GUIDANCE_STRENGTH,
            "guidance_rescale": GUIDANCE_RESCALE,
            "model_x_scale": MODEL_X_SCALE,
            "model_t_scale": MODEL_T_SCALE,
        },
        "inputs": {
            "noise": tensor_payload(noise_np),
            "positive_condition": tensor_payload(positive_np),
            "negative_condition": tensor_payload(negative_np),
            "all_unchanged": True,
        },
        "schedule": {
            "dtype": "float64",
            "values": schedule.tolist(),
            "f64_bits_hex": [f64_bits_hex(value) for value in schedule],
            "f64le_sha256": f64le_sha256(schedule),
        },
        "steps": sampler.records,
        "expected": {
            "pred_x_t_count": len(source.pred_x_t),
            "pred_x_0_count": len(source.pred_x_0),
            "samples": tensor_payload(source_final),
        },
        "independent_scalar_reference": {
            "kind": "explicit scalar Float32 model, CFG, x0-space sample-std rescale, and Euler loop",
            "schedule_f64le_sha256": f64le_sha256(schedule),
            "max_abs_error": max_abs_error,
            "samples_f32le_sha256": f32le_sha256(reference_final),
        },
        "counterfactuals": {
            "endpoint_pre_narrowed_schedule": {
                "schedule_values": narrowed_schedule.tolist(),
                "schedule_f32_bits": [f32_bits(value) for value in schedule],
                "samples": tensor_payload(narrowed_final),
                "differing_element_count": int(narrowed_differing.size),
                "first_differing_flat_index": int(narrowed_differing[0]),
                "max_abs_difference": float(np.max(np.abs(source_final - narrowed_final))),
            },
            "prediction_space_rescale": {
                "formula": "std over prediction tensors, blend prediction, then derive x0 only for history",
                "samples": tensor_payload(prediction_final),
                "differing_element_count": int(prediction_differing.size),
                "first_differing_flat_index": int(prediction_differing[0]),
                "max_abs_difference": float(np.max(np.abs(source_final - prediction_final))),
            },
        },
        "rejected_scope": [
            "Crystal repeated CFG-rescale sampler loop or public runtime API",
            "guidance interval, RNG replay, pipeline defaults, or production stages",
            "real weights, model scale, BF16, F16, quantization, GPU, or Metal",
            "aggregate process, stage, retained, RSS, native, or peak memory",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.trellis_root.resolve(), Path(__file__).resolve())
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
