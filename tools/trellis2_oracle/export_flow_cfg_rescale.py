#!/usr/bin/env python3
"""Export the pinned TRELLIS.2 CPU/F32 CFG-rescale oracle.

The exporter executes the real ``FlowEulerCfgSampler._inference_model`` MRO.
It covers only guidance-rescale routing and arithmetic; it does not execute a
sampler loop, load weights, use RNG, access the network, or touch an accelerator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from export_flow_euler_cfg import (
    NEG_COND,
    POS_COND,
    PRED_NEG,
    PRED_POS,
    TRELLIS_PIN,
    X_T,
    load_cfg_sampler,
    require_pins,
    sha256,
)


SCHEMA = "cogni-ml/trellis2/flow-cfg-rescale-oracle/v1"
GENERATOR_REPO_PATH = "tools/trellis2_oracle/export_flow_cfg_rescale.py"
SUPPORT_REPO_PATH = "tools/trellis2_oracle/export_flow_euler_cfg.py"
SHAPE = [2, 2, 2]
T = 0.6180339887498948
SIGMA_MIN = 1e-5


def f32le_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def f32_class(value: np.float32) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "positive_infinity"
    if np.isneginf(value):
        return "negative_infinity"
    return "finite"


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
    flat = array.reshape(-1)
    classes = [f32_class(item) for item in flat]
    all_finite = all(item == "finite" for item in classes)
    payload: dict[str, object] = {
        "shape": list(array.shape),
        "dtype": "float32",
        "all_finite": all_finite,
        "f32le_sha256": f32le_sha256(array),
    }
    if all_finite:
        payload["values"] = array.tolist()
    else:
        payload["f32_bits_hex"] = [
            f"0x{int(bits):08x}" for bits in flat.view(np.uint32)
        ]
        payload["classes"] = classes
    return payload


def scalar_payload(value: float) -> dict[str, object]:
    bits = np.asarray(np.float64(value)).view(np.uint64).item()
    if math.isnan(value):
        category = "nan"
    elif value == math.inf:
        category = "positive_infinity"
    elif value == -math.inf:
        category = "negative_infinity"
    else:
        category = "finite"
    payload: dict[str, object] = {
        "class": category,
        "f64_bits_hex": f"0x{bits:016x}",
    }
    if category == "finite":
        payload["value"] = value
    return payload


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


def build_fixture(trellis_root: Path, generator_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    sources = require_pins(trellis_root)
    sampler_base = load_cfg_sampler(trellis_root)

    class TracedSampler(sampler_base):
        def _pred_to_xstart(self, x_t, t, pred):
            role = "positive" if pred is self.trace_positive else "cfg"
            output = super()._pred_to_xstart(x_t, t, pred)
            self.conversion_trace.append(f"pred_to_xstart:{role}")
            self.intermediates[f"pred_{role}"] = pred
            self.intermediates[f"x0_{role}"] = output
            self.conversion_times.append(t)
            return output

        def _xstart_to_pred(self, x_t, t, x_0):
            self.conversion_trace.append("xstart_to_pred:blend")
            self.intermediates["x0_blend"] = x_0
            self.conversion_times.append(t)
            return super()._xstart_to_pred(x_t, t, x_0)

    sampler = TracedSampler(SIGMA_MIN)
    pos_cond = torch.tensor(POS_COND, dtype=torch.float32)
    neg_cond = torch.tensor(NEG_COND, dtype=torch.float32)
    standard = {
        "x_t": torch.tensor(X_T, dtype=torch.float32).reshape(SHAPE),
        "positive": torch.tensor(PRED_POS, dtype=torch.float32).reshape(SHAPE),
        "negative": torch.tensor(PRED_NEG, dtype=torch.float32).reshape(SHAPE),
    }
    zero_x = torch.zeros(SHAPE, dtype=torch.float32)
    variance_positive = torch.tensor(
        [0.25, 0.5, 1.0, 2.0, 0.125, 0.375, 0.75, 1.5],
        dtype=torch.float32,
    ).reshape(SHAPE)
    zero_target = torch.ones(SHAPE, dtype=torch.float32)
    near_target = torch.tensor(
        [
            1.0,
            1.0 + 2.0**-22,
            1.0 - 2.0**-22,
            1.0 + 2.0**-21,
            1.0,
            1.0 + 2.0**-21,
            1.0 - 2.0**-21,
            1.0 + 2.0**-20,
        ],
        dtype=torch.float32,
    ).reshape(SHAPE)

    case_specs = [
        ("positive_only_ignores_rescale", 1.0, math.inf, standard),
        ("negative_only_ignores_rescale", 0.0, math.nan, standard),
        ("mixed_zero_bypass", 1.7, 0.0, standard),
        ("mixed_negative_bypass", 1.7, -0.5, standard),
        ("mixed_nan_bypass", 1.7, math.nan, standard),
        ("mixed_finite_rescale", 1.7, 0.35, standard),
        ("mixed_unclamped_rescale", 1.7, 1.7, standard),
        ("mixed_positive_infinity", 1.7, math.inf, standard),
        (
            "mixed_zero_cfg_std",
            2.0,
            0.35,
            {
                "x_t": zero_x,
                "positive": variance_positive,
                "negative": 2.0 * variance_positive - zero_target,
            },
        ),
        (
            "mixed_near_zero_cfg_std",
            2.0,
            0.35,
            {
                "x_t": zero_x,
                "positive": variance_positive,
                "negative": 2.0 * variance_positive - near_target,
            },
        ),
    ]

    cases: list[dict[str, object]] = []
    for name, strength, rescale, tensors in case_specs:
        x_t = tensors["x_t"]
        positive = tensors["positive"]
        negative = tensors["negative"]
        originals = [item.clone() for item in (x_t, positive, negative, pos_cond, neg_cond)]
        calls: list[dict[str, object]] = []
        sampler.trace_positive = positive
        sampler.conversion_trace = []
        sampler.conversion_times = []
        sampler.intermediates = {}

        def model(actual_x, model_t, actual_cond, **kwargs):
            if kwargs:
                raise AssertionError(f"unexpected model kwargs: {kwargs}")
            if actual_x is not x_t:
                raise AssertionError("CFG-rescale path replaced x_t")
            if actual_cond is pos_cond:
                condition = "positive"
                prediction = positive
            elif actual_cond is neg_cond:
                condition = "negative"
                prediction = negative
            else:
                raise AssertionError("CFG-rescale path replaced condition carrier")
            if model_t.dtype != torch.float32 or model_t.device.type != "cpu":
                raise AssertionError("model timestep must be CPU Float32")
            if list(model_t.shape) != [SHAPE[0]]:
                raise AssertionError(f"unexpected model timestep shape: {model_t.shape}")
            calls.append(
                {
                    "condition": condition,
                    "x_t_identity_preserved": True,
                    "condition_identity_preserved": True,
                    "model_t": tensor_payload(model_t),
                }
            )
            return prediction

        with torch.no_grad():
            output = sampler._inference_model(
                model,
                x_t,
                T,
                pos_cond,
                neg_cond,
                guidance_strength=strength,
                guidance_rescale=rescale,
            )

        special = strength == 1.0 or strength == 0.0
        enters_rescale = (not special) and rescale > 0
        expected_order = ["positive"] if strength == 1.0 else ["negative"] if strength == 0.0 else ["positive", "negative"]
        if [call["condition"] for call in calls] != expected_order:
            raise AssertionError(f"provider order drift for {name}")
        expected_trace = (
            ["pred_to_xstart:positive", "pred_to_xstart:cfg", "xstart_to_pred:blend"]
            if enters_rescale
            else []
        )
        if sampler.conversion_trace != expected_trace:
            raise AssertionError(f"conversion routing drift for {name}")
        if any(value != T for value in sampler.conversion_times):
            raise AssertionError("conversion did not receive normalized Float64 time")

        identity = (
            "positive_prediction"
            if output is positive
            else "negative_prediction"
            if output is negative
            else "owned_mixed"
        )
        if special and identity == "owned_mixed":
            raise AssertionError("special CFG branch lost provider identity")
        if not special and identity != "owned_mixed":
            raise AssertionError("mixed CFG branch aliased a provider output")

        intermediates: dict[str, object] = {}
        if enters_rescale:
            pred_cfg = sampler.intermediates["pred_cfg"]
            x0_positive = sampler.intermediates["x0_positive"]
            x0_cfg = sampler.intermediates["x0_cfg"]
            std_dims = list(range(1, x0_positive.ndim))
            std_positive = x0_positive.std(dim=std_dims, keepdim=True)
            std_cfg = x0_cfg.std(dim=std_dims, keepdim=True)
            x0_rescaled = x0_cfg * (std_positive / std_cfg)
            x0_blend_reference = rescale * x0_rescaled + (1 - rescale) * x0_cfg
            if not same_f32_bits(x0_blend_reference, sampler.intermediates["x0_blend"]):
                raise AssertionError(f"source-order blend reconstruction drift for {name}")
            intermediates = {
                "pred_cfg": tensor_payload(pred_cfg),
                "x0_positive": tensor_payload(x0_positive),
                "x0_cfg": tensor_payload(x0_cfg),
                "std_positive": tensor_payload(std_positive),
                "std_cfg": tensor_payload(std_cfg),
                "x0_rescaled": tensor_payload(x0_rescaled),
                "x0_blend": tensor_payload(sampler.intermediates["x0_blend"]),
            }

        for current, original in zip((x_t, positive, negative, pos_cond, neg_cond), originals):
            if not torch.equal(current, original):
                raise AssertionError(f"source path mutated an input for {name}")

        cases.append(
            {
                "name": name,
                "guidance_strength": strength,
                "guidance_rescale": scalar_payload(rescale),
                "rescale_branch_entered": enters_rescale,
                "call_order": [call["condition"] for call in calls],
                "calls": calls,
                "conversion_trace": sampler.conversion_trace,
                "output_identity": identity,
                "inputs": {
                    "x_t": tensor_payload(x_t),
                    "positive": tensor_payload(positive),
                    "negative": tensor_payload(negative),
                    "all_unchanged": True,
                },
                "intermediates": intermediates,
                "output": tensor_payload(output),
            }
        )

    by_name = {case["name"]: case for case in cases}
    finite_case = by_name["mixed_finite_rescale"]
    finite_intermediates = finite_case["intermediates"]
    x0_positive = np.asarray(finite_intermediates["x0_positive"]["values"], dtype=np.float32)
    x0_cfg = np.asarray(finite_intermediates["x0_cfg"]["values"], dtype=np.float32)
    scalar_std_positive = scalar_sample_std(x0_positive)
    scalar_std_cfg = scalar_sample_std(x0_cfg)
    torch_std_positive = np.asarray(finite_intermediates["std_positive"]["values"], dtype=np.float32)
    torch_std_cfg = np.asarray(finite_intermediates["std_cfg"]["values"], dtype=np.float32)
    scalar_std_error = max(
        float(np.max(np.abs(scalar_std_positive - torch_std_positive))),
        float(np.max(np.abs(scalar_std_cfg - torch_std_cfg))),
    )
    if scalar_std_error > 1e-3:
        raise AssertionError("independent scalar sample-std reference diverged")

    # Counterfactuals deliberately target plausible but wrong ports.
    x_t = standard["x_t"]
    positive = standard["positive"]
    pred_cfg = finite_intermediates["pred_cfg"]
    pred_cfg_tensor = torch.from_numpy(np.asarray(pred_cfg["values"], dtype=np.float32))
    x0_positive_tensor = torch.from_numpy(x0_positive)
    x0_cfg_tensor = torch.from_numpy(x0_cfg)
    global_std_positive = x0_positive_tensor.std(keepdim=True)
    global_std_cfg = x0_cfg_tensor.std(keepdim=True)
    global_blend = 0.35 * x0_cfg_tensor * (global_std_positive / global_std_cfg) + 0.65 * x0_cfg_tensor
    global_output = sampler_base(SIGMA_MIN)._xstart_to_pred(x_t, T, global_blend)
    population_std_positive = x0_positive_tensor.std(dim=[1, 2], correction=0, keepdim=True)
    population_std_cfg = x0_cfg_tensor.std(dim=[1, 2], correction=0, keepdim=True)
    pred_std_positive = positive.std(dim=[1, 2], keepdim=True)
    pred_std_cfg = pred_cfg_tensor.std(dim=[1, 2], keepdim=True)
    prediction_space = 0.35 * pred_cfg_tensor * (pred_std_positive / pred_std_cfg) + 0.65 * pred_cfg_tensor

    unclamped = by_name["mixed_unclamped_rescale"]
    unclamped_i = unclamped["intermediates"]
    unclamped_x_t = torch.from_numpy(np.asarray(unclamped["inputs"]["x_t"]["values"], dtype=np.float32))
    unclamped_x0_rescaled = torch.from_numpy(np.asarray(unclamped_i["x0_rescaled"]["values"], dtype=np.float32))
    clamp_one_output = sampler_base(SIGMA_MIN)._xstart_to_pred(unclamped_x_t, T, unclamped_x0_rescaled)

    if finite_case["output"]["f32le_sha256"] == tensor_payload(global_output)["f32le_sha256"]:
        raise AssertionError("global-axis counterfactual no longer differs")
    if finite_case["output"]["f32le_sha256"] == tensor_payload(prediction_space)["f32le_sha256"]:
        raise AssertionError("prediction-space counterfactual no longer differs")
    if unclamped["output"]["f32le_sha256"] == tensor_payload(clamp_one_output)["f32le_sha256"]:
        raise AssertionError("clamp-to-one counterfactual no longer differs")

    zero_std = by_name["mixed_zero_cfg_std"]["intermediates"]["std_cfg"]
    if zero_std["values"] != [[[0.0]], [[0.0]]]:
        raise AssertionError("zero-variance fixture is not exact")
    near_std = by_name["mixed_near_zero_cfg_std"]["intermediates"]["std_cfg"]
    near_values = np.asarray(near_std["values"], dtype=np.float32)
    if not near_std["all_finite"] or np.any(near_values == np.float32(0.0)):
        raise AssertionError("near-zero fixture collapsed to zero")

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
            "threads": 1,
            "dtype": "Float64 controls and normalized time; Float32 tensors",
            "weights": "none",
            "network": "none",
            "generator": GENERATOR_REPO_PATH,
            "generator_sha256": sha256(generator_path),
            "support": SUPPORT_REPO_PATH,
            "support_sha256": sha256(generator_path.parent / "export_flow_euler_cfg.py"),
            "sources": source_payload,
        },
        "contract": {
            "owner": "ClassifierFreeGuidanceSamplerMixin._inference_model",
            "entrypoint": "trellis2.pipelines.samplers.flow_euler.FlowEulerCfgSampler._inference_model",
            "mro": [owner.__name__ for owner in sampler_base.__mro__],
            "rescale_gate": "guidance_strength is mixed and guidance_rescale > 0",
            "formula": "x0_pos=pred_to_xstart(pos); x0_cfg=pred_to_xstart(cfg); std over all non-batch axes with default correction=1 and keepdim; x0_rescaled=x0_cfg*(std_pos/std_cfg); x0=r*x0_rescaled+(1-r)*x0_cfg; pred=xstart_to_pred(x0)",
            "validation": "no clamp, epsilon floor, or finite-value validation in pinned source",
            "std_axes": [1, 2],
            "std_correction": 1,
            "std_keepdim": True,
            "normalized_t": T,
            "sigma_min": SIGMA_MIN,
            "model_timestep": "fresh CPU Float32[batch] = 1000*normalized_t",
        },
        "cases": cases,
        "independent_reference": {
            "kind": "two-pass scalar Float32 per-batch sample standard deviation",
            "std_positive": tensor_payload(scalar_std_positive),
            "std_cfg": tensor_payload(scalar_std_cfg),
            "max_abs_error_vs_torch": scalar_std_error,
        },
        "counterfactuals": {
            "global_axes_output": tensor_payload(global_output),
            "population_std_positive": tensor_payload(population_std_positive),
            "population_std_cfg": tensor_payload(population_std_cfg),
            "population_ratio_note": "correction factor cancels from std_pos/std_cfg when both reduce the same finite element count; intermediate std values still differ",
            "prediction_space_output": tensor_payload(prediction_space),
            "clamp_unclamped_rescale_to_one_output": tensor_payload(clamp_one_output),
        },
        "rejected_scope": [
            "Crystal guidance-rescale runtime or public API",
            "guidance-interval routing or repeated sampler execution",
            "random-noise or seed reproducibility",
            "production defaults, pipeline, sparse flow, decoder, or mesh",
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
        json.dumps(fixture, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
