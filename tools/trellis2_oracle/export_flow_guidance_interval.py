#!/usr/bin/env python3
"""Export a source-pinned TRELLIS.2 guidance-interval routing oracle.

This executes the real ``FlowEulerGuidanceIntervalSampler._inference_model``
method through its upstream MRO.  It isolates the inclusive normalized-time
interval predicate and the outside-interval strength-1 override.  It does not
execute a sampler loop, guidance rescale, weights, RNG, or an accelerator.
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
    MIXED_STRENGTH,
    NEG_COND,
    POS_COND,
    PRED_NEG,
    PRED_POS,
    SHAPE,
    TRELLIS_PIN,
    X_T,
    f32_bits,
    load_cfg_sampler,
    require_pins,
    scalar_source_mix,
    sha256,
    tensor_payload,
)


SCHEMA = "cogni-ml/trellis2/flow-guidance-interval-oracle/v1"
GENERATOR_REPO_PATH = "tools/trellis2_oracle/export_flow_guidance_interval.py"
SUPPORT_REPO_PATH = "tools/trellis2_oracle/export_flow_euler_cfg.py"

LOWER = 0.25
UPPER = 0.75
MIDPOINT = 0.5


def f64_bits_hex(value: float) -> str:
    bits = np.asarray(np.float64(value)).view(np.uint64).item()
    return f"0x{bits:016x}"


def f32le_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def load_interval_sampler(trellis_root: Path):
    # The support loader installs only the pinned sampler modules and its tiny
    # dependency stubs.  Read the interval class from the same loaded module so
    # this exporter executes the actual upstream MRO rather than a replica.
    load_cfg_sampler(trellis_root)
    module = sys.modules["trellis2.pipelines.samplers.flow_euler"]
    return module.FlowEulerGuidanceIntervalSampler


def build_fixture(trellis_root: Path, generator_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    sources = require_pins(trellis_root)
    sampler_class = load_interval_sampler(trellis_root)
    sampler = sampler_class(1e-5)

    x_t = torch.tensor(X_T, dtype=torch.float32).reshape(SHAPE)
    pred_pos = torch.tensor(PRED_POS, dtype=torch.float32).reshape(SHAPE)
    pred_neg = torch.tensor(PRED_NEG, dtype=torch.float32).reshape(SHAPE)
    pos_cond = torch.tensor(POS_COND, dtype=torch.float32)
    neg_cond = torch.tensor(NEG_COND, dtype=torch.float32)
    originals = [value.clone() for value in (x_t, pred_pos, pred_neg, pos_cond, neg_cond)]

    interval = (LOWER, UPPER)
    case_inputs = [
        ("below_lower", math.nextafter(LOWER, -math.inf), False),
        ("lower_boundary", LOWER, True),
        ("inside", MIDPOINT, True),
        ("upper_boundary", UPPER, True),
        ("above_upper", math.nextafter(UPPER, math.inf), False),
    ]

    def execute_case(name: str, normalized_t: float, inside: bool) -> dict[str, object]:
        calls: list[dict[str, object]] = []

        def model(actual_x, model_t, actual_cond, **kwargs):
            if kwargs:
                raise AssertionError(f"unexpected model kwargs: {kwargs}")
            if actual_x is not x_t:
                raise AssertionError("interval path replaced x_t before model inference")
            if actual_cond is pos_cond:
                condition = "positive"
                prediction = pred_pos
            elif actual_cond is neg_cond:
                condition = "negative"
                prediction = pred_neg
            else:
                raise AssertionError("interval path replaced the condition carrier")
            if list(model_t.shape) != [SHAPE[0]]:
                raise AssertionError(f"unexpected model timestep shape: {model_t.shape}")
            if model_t.dtype != torch.float32 or model_t.device.type != "cpu":
                raise AssertionError("model timestep must be CPU Float32")
            calls.append(
                {
                    "index": len(calls),
                    "condition": condition,
                    "x_t_identity_preserved": True,
                    "condition_identity_preserved": True,
                    "model_t": tensor_payload(model_t.detach().cpu().numpy()),
                    "prediction": tensor_payload(prediction.detach().cpu().numpy()),
                }
            )
            return prediction

        output = sampler._inference_model(
            model,
            x_t,
            normalized_t,
            pos_cond,
            neg_cond=neg_cond,
            guidance_strength=MIXED_STRENGTH,
            guidance_interval=interval,
            guidance_rescale=0.0,
        )
        expected_order = ["positive", "negative"] if inside else ["positive"]
        actual_order = [call["condition"] for call in calls]
        if actual_order != expected_order:
            raise AssertionError(
                f"source interval routing drift for {name}: {actual_order}"
            )
        expected_output = None if inside else pred_pos
        if expected_output is not None and output is not expected_output:
            raise AssertionError("outside interval did not preserve positive output identity")
        if inside and (output is pred_pos or output is pred_neg):
            raise AssertionError("inside interval did not produce an owned mixed output")
        return {
            "name": name,
            "normalized_t": normalized_t,
            "normalized_t_f64_bits_hex": f64_bits_hex(normalized_t),
            "inside_inclusive_interval": inside,
            "requested_guidance_strength": MIXED_STRENGTH,
            "effective_guidance_strength": MIXED_STRENGTH if inside else 1.0,
            "call_count": len(calls),
            "call_order": actual_order,
            "output_identity": "owned_mixed" if inside else "positive_prediction",
            "calls": calls,
            "output": tensor_payload(output.detach().cpu().numpy()),
        }

    cases = [execute_case(*case) for case in case_inputs]

    scalar_mixed = scalar_source_mix(
        pred_pos.detach().cpu().numpy(),
        pred_neg.detach().cpu().numpy(),
        MIXED_STRENGTH,
    )
    for case in cases:
        actual = np.asarray(case["output"]["values"], dtype=np.float32)
        expected = scalar_mixed if case["inside_inclusive_interval"] else pred_pos.numpy()
        if not np.array_equal(actual, expected):
            raise AssertionError(f"independent routing/output drift for {case['name']}")

    # The adjacent Float64 points deliberately narrow to the same model-time
    # F32 values as the exact boundaries.  Routing must still differ.
    paired_names = [("below_lower", "lower_boundary"), ("above_upper", "upper_boundary")]
    by_name = {case["name"]: case for case in cases}
    for outside_name, boundary_name in paired_names:
        outside_t = np.asarray(
            by_name[outside_name]["calls"][0]["model_t"]["values"], dtype=np.float32
        )
        boundary_t = np.asarray(
            by_name[boundary_name]["calls"][0]["model_t"]["values"], dtype=np.float32
        )
        if not np.array_equal(outside_t, boundary_t):
            raise AssertionError("fixture no longer separates Float64 routing from F32 model time")
        if by_name[outside_name]["call_order"] == by_name[boundary_name]["call_order"]:
            raise AssertionError("fixture failed to expose inclusive-boundary routing")

    for current, original in zip(
        (x_t, pred_pos, pred_neg, pos_cond, neg_cond), originals
    ):
        if not torch.equal(current, original):
            raise AssertionError("source guidance-interval path mutated a fixture input")

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
            "dtype": "Float64 normalized time; Float32 state, predictions, model timestep, and output",
            "weights": "none",
            "network": "none",
            "generator": GENERATOR_REPO_PATH,
            "generator_sha256": sha256(generator_path),
            "support": SUPPORT_REPO_PATH,
            "support_sha256": sha256(generator_path.parent / "export_flow_euler_cfg.py"),
            "sources": source_payload,
        },
        "contract": {
            "owner": "GuidanceIntervalSamplerMixin._inference_model",
            "entrypoint": "trellis2.pipelines.samplers.flow_euler.FlowEulerGuidanceIntervalSampler._inference_model",
            "mro": [owner.__name__ for owner in sampler_class.__mro__],
            "predicate": "guidance_interval[0] <= normalized_t <= guidance_interval[1]",
            "inside": "forward requested guidance strength to classifier-free guidance",
            "outside": "force guidance strength to 1 before classifier-free guidance",
            "model_timestep": "base flow receives normalized Float64 t and creates CPU Float32[batch] = 1000*t",
            "condition_carriers": "opaque positive and negative objects forwarded by identity",
            "guidance_rescale": "explicitly 0.0; standard-deviation rescale branch not executed",
        },
        "fixture": {
            "shape": SHAPE,
            "guidance_interval": [LOWER, UPPER],
            "requested_guidance_strength": MIXED_STRENGTH,
            "case_count": len(cases),
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
        "independent_reference": {
            "predicate": "lower <= normalized_t and normalized_t <= upper",
            "outside_effective_guidance_strength": 1.0,
            "inside_output": tensor_payload(scalar_mixed),
            "inside_output_f32_bits": f32_bits(scalar_mixed),
        },
        "counterfactuals": {
            "strict_interval_mismatches": ["lower_boundary", "upper_boundary"],
            "missing_outside_override_mismatches": ["below_lower", "above_upper"],
            "model_t_domain_mismatches": ["lower_boundary", "inside", "upper_boundary"],
            "outside_negative_output_sha256": f32le_sha256(pred_neg.detach().cpu().numpy()),
        },
        "rejected_scope": [
            "Crystal guidance-interval runtime or public API",
            "standard-deviation guidance rescale execution",
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
