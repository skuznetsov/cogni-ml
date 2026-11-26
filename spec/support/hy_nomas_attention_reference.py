#!/usr/bin/env python3
"""Emit same-input MPS references for isolated Hy-Nomas attention operations.

This producer consumes a hash-pinned single-block fixture and the exact
checkpoint subset named by that fixture. It does not instantiate or execute a
Hy-Nomas block. Projection, RMSNorm, RoPE, and attention operations are
evaluated independently on frozen fixture tensors on MPS. The result is a
diagnostic artifact, not an alternate production oracle or Magi parity claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
REFERENCE_KIND = "hy_nomas_same_input_attention_diagnostic"
SOURCE_VARIANTS = {
    "f32": "F32",
    "bf16_weights_roundtrip_f32_compute": "BF16",
}
PROJECTIONS = ("q", "k", "v", "proj")
ATTENTION_CHECKPOINTS = (
    "attention_replay.query",
    "attention_replay.key",
    "attention_replay.value",
    "attention_replay.attention_f32",
)


class DiagnosticError(Exception):
    """A bounded, safe-to-report rejection of an input artifact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_object_without_duplicate_keys)
    except DiagnosticError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DiagnosticError(f"invalid {context} JSON: {exc}") from None
    if not isinstance(value, dict):
        raise DiagnosticError(f"{context} JSON must be an object")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _record_shape(record: Any, *, context: str) -> tuple[int, ...]:
    if not isinstance(record, dict) or set(record) != {"shape", "values"}:
        raise DiagnosticError(f"{context} must be a tensor record with shape and values")
    shape = record["shape"]
    values = record["values"]
    if not isinstance(shape, list) or not shape or not isinstance(values, list):
        raise DiagnosticError(f"{context} has an invalid shape or values list")
    count = 1
    normalized = []
    for dimension in shape:
        if not _is_int(dimension) or dimension <= 0:
            raise DiagnosticError(f"{context} has an invalid shape dimension")
        normalized.append(dimension)
        count *= dimension
    if count != len(values):
        raise DiagnosticError(f"{context} value count does not match its shape")
    return tuple(normalized)


def _validate_record_values(record: dict[str, Any], *, context: str) -> None:
    _record_shape(record, context=context)
    for value in record["values"]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DiagnosticError(f"{context} contains a non-numeric value")
        try:
            f64 = float(value)
        except (OverflowError, ValueError):
            raise DiagnosticError(f"{context} contains a value outside F64 range") from None
        if not math.isfinite(f64):
            raise DiagnosticError(f"{context} contains a non-finite value")
        if abs(f64) > 3.4028234663852886e38:
            raise DiagnosticError(f"{context} contains a value outside F32 range")


def _selected_fixture(document: dict[str, Any]) -> dict[str, Any]:
    """Keep only frozen tensors consumed by this producer before torch import."""
    if document.get("schema_version") != 2:
        raise DiagnosticError("source fixture must use schema_version 2")
    if document.get("scope", {}).get("attention_precision") != "bf16-boundaries":
        raise DiagnosticError("source fixture must be the BF16-boundary control")
    if document.get("scope", {}).get("reference_kind") != (
        "upstream_block_bf16_boundary_control_mps_sdpa"
    ):
        raise DiagnosticError("source fixture is outside the admitted boundary-control scope")
    provenance = document.get("provenance")
    config = document.get("config")
    cases = document.get("cases")
    if not isinstance(provenance, dict) or not isinstance(config, dict):
        raise DiagnosticError("source fixture is missing provenance or config")
    if provenance.get("revision") != PINNED_REVISION:
        raise DiagnosticError("source fixture upstream revision is not pinned")
    if not isinstance(cases, list) or {case.get("name") for case in cases} != {
        "broadcast",
        "per_token",
    }:
        raise DiagnosticError("source fixture must contain broadcast and per_token cases")

    selected_cases = []
    for case in cases:
        if not isinstance(case, dict):
            raise DiagnosticError("source fixture case must be an object")
        name = case["name"]
        variants = case.get("variants")
        inputs = case.get("inputs")
        if not isinstance(variants, dict) or set(variants) != set(SOURCE_VARIANTS):
            raise DiagnosticError(f"{name} must have both admitted weight variants")
        if not isinstance(inputs, dict) or not {"video_cos", "video_sin"} <= set(inputs):
            raise DiagnosticError(f"{name} is missing frozen video RoPE frequencies")
        for key in ("video_cos", "video_sin"):
            _validate_record_values(inputs[key], context=f"{name}.inputs.{key}")

        selected_variants = []
        for variant_name, resident_dtype in SOURCE_VARIANTS.items():
            variant = variants[variant_name]
            if not isinstance(variant, dict) or variant.get("resident_dtype") != resident_dtype:
                raise DiagnosticError(f"{name}/{variant_name} resident dtype mismatch")
            checkpoints = variant.get("checkpoints")
            replay = variant.get("attention_replay")
            if not isinstance(checkpoints, dict) or not isinstance(replay, dict):
                raise DiagnosticError(f"{name}/{variant_name} is missing frozen checkpoints")

            required = set()
            for stream in ("img", "txt"):
                required.add(f"{stream}.attn.modulated")
                required.add(f"{stream}.attn.raw")
                for projection in PROJECTIONS:
                    required.add(f"{stream}.attn.{projection if projection != 'proj' else 'projected'}")
                for norm in ("q", "k"):
                    required.add(f"{stream}.attn.{norm}_norm")
            required.update({"img.attn.q_rope", "img.attn.k_rope"})
            if not required <= set(checkpoints):
                raise DiagnosticError(
                    f"{name}/{variant_name} is missing checkpoints: "
                    f"{sorted(required - set(checkpoints))}"
                )
            if not set(ATTENTION_CHECKPOINTS[0:3]) <= {
                f"attention_replay.{key}" for key in replay
            }:
                raise DiagnosticError(f"{name}/{variant_name} is missing replay Q/K/V")
            if not {"query", "key", "value", "attention_f32"} <= set(replay):
                raise DiagnosticError(f"{name}/{variant_name} is missing replay tensors")

            selected_checkpoints = {}
            for key in sorted(required):
                _validate_record_values(checkpoints[key], context=f"{name}/{variant_name}/{key}")
                selected_checkpoints[key] = checkpoints[key]
            selected_replay = {}
            for key in ("query", "key", "value", "attention_f32"):
                _validate_record_values(replay[key], context=f"{name}/{variant_name}/replay.{key}")
                selected_replay[key] = replay[key]

            selected_variants.append(
                {
                    "name": variant_name,
                    "resident_dtype": resident_dtype,
                    "checkpoints": selected_checkpoints,
                    "attention_replay": selected_replay,
                }
            )
        selected_cases.append(
            {
                "name": name,
                "config_overrides": case.get("config_overrides", {}),
                "inputs": {key: inputs[key] for key in ("video_cos", "video_sin")},
                "variants": selected_variants,
            }
        )

    return {
        "config": config,
        "provenance": provenance,
        "source": document.get("source"),
        "weights_are_synthetic": document.get("weights_are_synthetic"),
        "cases": selected_cases,
    }


def _selected_fixture_with_weight_provenance(document: dict[str, Any]) -> dict[str, Any]:
    selected = _selected_fixture(document)
    if selected["weights_are_synthetic"] is True:
        hashes = set()
        for case in document["cases"]:
            weights = case["variants"]["f32"].get("weights_file", {})
            if isinstance(weights, dict) and isinstance(weights.get("sha256"), str):
                hashes.add(weights["sha256"])
        if len(hashes) != 1:
            raise DiagnosticError("synthetic fixture does not name one common F32 weight sidecar")
        selected["source_checkpoint_sha256"] = next(iter(hashes))
        selected["source_dtype"] = "F32"
    else:
        source = selected.get("source")
        if not isinstance(source, dict):
            raise DiagnosticError("trusted real fixture is missing source checkpoint provenance")
        selected["source_checkpoint_sha256"] = source.get("file_sha256")
        selected["source_dtype"] = source.get("source_dtype")
    return selected


def _expected_keys(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    hidden = config.get("dim")
    ffn = config.get("ffn_dim")
    heads = config.get("heads")
    if not all(_is_int(value) and value > 0 for value in (hidden, ffn, heads)):
        raise DiagnosticError("fixture config must contain positive dim, ffn_dim, and heads")
    if hidden % heads:
        raise DiagnosticError("fixture dim must be divisible by heads")
    expected: dict[str, tuple[int, ...]] = {}
    for stream in ("img", "txt"):
        expected[f"{stream}_mod.linear.weight"] = (6 * hidden, hidden)
        expected[f"{stream}_mod.linear.bias"] = (6 * hidden,)
        for projection in PROJECTIONS:
            prefix = f"{stream}_attn_{projection}"
            expected[f"{prefix}.weight"] = (hidden, hidden)
            expected[f"{prefix}.bias"] = (hidden,)
        for norm in ("q_norm", "k_norm"):
            expected[f"{stream}_attn_{norm}.weight"] = (hidden // heads,)
        expected[f"{stream}_mlp.fc1.weight"] = (ffn, hidden)
        expected[f"{stream}_mlp.fc1.bias"] = (ffn,)
        expected[f"{stream}_mlp.fc2.weight"] = (hidden, ffn)
        expected[f"{stream}_mlp.fc2.bias"] = (hidden,)
    return expected


def _verify_weights_file(
    weights: Path,
    config: dict[str, Any],
    source_dtype: str,
) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise DiagnosticError("checkpoint diagnostics require safetensors") from exc
    expected = _expected_keys(config)
    source_keys = {f"double_blocks.0.{name}": shape for name, shape in expected.items()}
    try:
        with safe_open(str(weights), framework="pt", device="cpu") as source:
            if set(source.keys()) != set(source_keys):
                raise DiagnosticError("weights file must contain exactly the expected block subset")
            for key, shape in source_keys.items():
                tensor_slice = source.get_slice(key)
                if tuple(tensor_slice.get_shape()) != shape:
                    raise DiagnosticError(f"checkpoint shape mismatch for {key}")
                if tensor_slice.get_dtype() != source_dtype:
                    raise DiagnosticError(f"checkpoint dtype mismatch for {key}")
    except DiagnosticError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise DiagnosticError(f"could not validate checkpoint subset: {exc}") from None


def _input_tensor(torch: Any, record: dict[str, Any], device: Any) -> Any:
    return torch.tensor(record["values"], dtype=torch.float32, device=device).reshape(
        tuple(record["shape"])
    )


def _weight_tensor(
    torch: Any,
    source: Any,
    prefix: str,
    name: str,
    resident_dtype: str,
    device: Any,
) -> Any:
    key = f"double_blocks.0.{prefix}.{name}"
    value = source.get_tensor(key)
    value = value.to(device=device, dtype=torch.float32)
    if resident_dtype == "BF16":
        value = value.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    return value


def _capture_operation(
    torch: Any,
    capture: dict[str, Any],
    name: str,
    kind: str,
    inputs: dict[str, Any],
    expected: Any,
    *,
    origin_expected: Any | None = None,
    weight_prefix: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    keys = {input_name: f"{name}.input.{input_name}" for input_name in inputs}
    for input_name, tensor in inputs.items():
        capture[keys[input_name]] = tensor
    expected_key = f"{name}.expected"
    rounded_key = f"{name}.rounded_expected"
    capture[expected_key] = expected
    capture[rounded_key] = expected.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    operation: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "_input_keys": keys,
        "_expected_key": expected_key,
        "_rounded_expected_key": rounded_key,
    }
    if origin_expected is not None:
        origin_key = f"{name}.origin_expected"
        capture[origin_key] = origin_expected
        operation["_origin_expected_key"] = origin_key
    if weight_prefix is not None:
        operation["weight_prefix"] = weight_prefix
    if metadata:
        operation.update(metadata)
    return operation


def _linear_output(
    torch: Any,
    functional: Any,
    x: Any,
    weight: Any,
    bias: Any | None,
    *,
    projection: str,
    config: dict[str, Any],
) -> Any:
    result = functional.linear(x, weight, bias)
    if projection in ("q", "k", "v"):
        result = result.reshape(
            result.shape[0], result.shape[1], config["heads"], config["head_dim"]
        )
    return result


def _linear_operations(
    torch: Any,
    source: Any,
    selected_case: dict[str, Any],
    variant: dict[str, Any],
    config: dict[str, Any],
    device: Any,
    capture: dict[str, Any],
) -> list[dict[str, Any]]:
    functional = torch.nn.functional
    checkpoints = variant["checkpoints"]
    operations = []
    for stream in ("img", "txt"):
        for projection in PROJECTIONS:
            prefix = f"{stream}_attn_{projection}"
            if projection == "proj":
                x_record = checkpoints[f"{stream}.attn.raw"]
                x = _input_tensor(torch, x_record, device)
                origin = checkpoints[f"{stream}.attn.projected"]
            else:
                x_record = checkpoints[f"{stream}.attn.modulated"]
                x = _input_tensor(torch, x_record, device)
                origin = checkpoints[f"{stream}.attn.{projection}"]
            weight = _weight_tensor(torch, source, prefix, "weight", variant["resident_dtype"], device)
            bias = _weight_tensor(torch, source, prefix, "bias", variant["resident_dtype"], device)
            base = f"{stream}.attn.{projection}.linear"
            expected = _linear_output(
                torch,
                functional,
                x,
                weight,
                bias,
                projection=projection,
                config=config,
            )
            operations.append(
                _capture_operation(
                    torch,
                    capture,
                    base,
                    "linear",
                    {"x": x},
                    expected,
                    origin_expected=_input_tensor(torch, origin, device),
                    weight_prefix=prefix,
                    metadata={"bias_mode": "normal", "reference_bias_mode": "fused_linear"},
                )
            )

            no_bias = _linear_output(
                torch,
                functional,
                x,
                weight,
                None,
                projection=projection,
                config=config,
            )
            operations.append(
                _capture_operation(
                    torch,
                    capture,
                    f"{base}_nobias",
                    "linear",
                    {"x": x},
                    no_bias,
                    weight_prefix=prefix,
                    metadata={"bias_mode": "none", "reference_bias_mode": "none"},
                )
            )
            explicit_bias = _linear_output(
                torch, functional, x, weight, None,
                projection="proj", config=config,
            ) + bias
            if projection in ("q", "k", "v"):
                explicit_bias = explicit_bias.reshape(
                    explicit_bias.shape[0], explicit_bias.shape[1],
                    config["heads"], config["head_dim"],
                )
            operations.append(
                _capture_operation(
                    torch,
                    capture,
                    f"{base}_explicit_bias",
                    "linear",
                    {"x": x},
                    explicit_bias,
                    weight_prefix=prefix,
                    metadata={
                        "bias_mode": "normal",
                        "reference_bias_mode": "explicit_add_after_linear",
                    },
                )
            )
            del weight, bias, expected, no_bias, explicit_bias, x
    return operations


def _norm_and_rope_operations(
    torch: Any,
    norm_layers: Any,
    posemb_layers: Any,
    source: Any,
    selected_case: dict[str, Any],
    variant: dict[str, Any],
    config: dict[str, Any],
    device: Any,
    capture: dict[str, Any],
) -> list[dict[str, Any]]:
    checkpoints = variant["checkpoints"]
    operations = []
    for stream in ("img", "txt"):
        for norm_name in ("q", "k"):
            prefix = f"{stream}_attn_{norm_name}_norm"
            x = _input_tensor(torch, checkpoints[f"{stream}.attn.{norm_name}"], device)
            weight = _weight_tensor(torch, source, prefix, "weight", variant["resident_dtype"], device)
            norm_class = norm_layers("rms")
            norm = norm_class(
                config["head_dim"], elementwise_affine=True,
                eps=config["norm_eps"], device=device, dtype=torch.float32,
            )
            with torch.no_grad():
                norm.weight.copy_(weight)
            expected = norm(x)
            operations.append(
                _capture_operation(
                    torch,
                    capture,
                    f"{stream}.attn.{norm_name}.rms",
                    "rms",
                    {"x": x},
                    expected,
                    origin_expected=_input_tensor(
                        torch, checkpoints[f"{stream}.attn.{norm_name}_norm"], device
                    ),
                    weight_prefix=prefix,
                    metadata={"eps": config["norm_eps"]},
                )
            )
            del x, weight, norm, expected

    img_q = _input_tensor(torch, checkpoints["img.attn.q_norm"], device)
    img_k = _input_tensor(torch, checkpoints["img.attn.k_norm"], device)
    cos = _input_tensor(torch, selected_case["inputs"]["video_cos"], device)
    sin = _input_tensor(torch, selected_case["inputs"]["video_sin"], device)
    expected_q, expected_k = posemb_layers.apply_rotary_emb(
        img_q, img_k, (cos, sin), head_first=False
    )
    for label, x, expected in (
        ("q", img_q, expected_q),
        ("k", img_k, expected_k),
    ):
        operations.append(
            _capture_operation(
                torch,
                capture,
                f"img.attn.{label}.rope",
                "rope",
                {"x": x, "cos": cos, "sin": sin},
                expected,
                origin_expected=_input_tensor(
                    torch, checkpoints[f"img.attn.{label}_rope"], device
                ),
            )
        )
    return operations


def _qk_scores(torch: Any, q: Any, k: Any, *, scale: float) -> Any:
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 3, 1)
    return (torch.matmul(qh, kh) * scale).permute(0, 2, 1, 3).contiguous()


def _pv_output(torch: Any, probabilities: Any, value: Any) -> Any:
    output = torch.matmul(
        probabilities.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
    )
    return output.permute(0, 2, 1, 3).contiguous()


def _online_from_scores(torch: Any, scores: Any, value: Any) -> Any:
    """GPU-only explicit online-softmax recurrence over frozen scores."""
    key_count = scores.shape[-1]
    maximum = None
    running_sum = torch.zeros_like(scores[..., 0])
    accumulator = torch.zeros_like(value[:, 0, :, :]).unsqueeze(1).expand(
        value.shape[0], scores.shape[1], value.shape[2], value.shape[3]
    ).clone()
    for key_index in range(key_count):
        score = scores[..., key_index]
        next_maximum = score if maximum is None else torch.maximum(maximum, score)
        correction = (
            torch.zeros_like(score)
            if maximum is None
            else torch.exp(maximum - next_maximum)
        )
        probability = torch.exp(score - next_maximum)
        running_sum = running_sum * correction + probability
        accumulator = (
            accumulator * correction.unsqueeze(-1)
            + probability.unsqueeze(-1) * value[:, key_index, :, :].unsqueeze(1)
        )
        maximum = next_maximum
    return accumulator / running_sum.unsqueeze(-1)


def _attention_operations(
    torch: Any,
    variant: dict[str, Any],
    config: dict[str, Any],
    device: Any,
    capture: dict[str, Any],
) -> list[dict[str, Any]]:
    replay = variant["attention_replay"]
    q = _input_tensor(torch, replay["query"], device)
    k = _input_tensor(torch, replay["key"], device)
    v = _input_tensor(torch, replay["value"], device)
    capture["canonical_qkv.q"] = q
    capture["canonical_qkv.k"] = k
    capture["canonical_qkv.v"] = v
    scale = 1.0 / math.sqrt(float(config["head_dim"]))
    sqrt_scale = math.sqrt(scale)
    q_scaled = q * scale
    q_split = q * sqrt_scale
    k_split = k * sqrt_scale

    score_modes = (
        ("postscale", q, k, scale, "post_matmul"),
        ("query_prescale", q_scaled, k, 1.0, "query_prescaled"),
        ("split_prescale", q_split, k_split, 1.0, "split_sqrt_prescaled"),
    )
    operations = []
    scores_by_mode = {}
    probs_by_mode = {}
    output_by_mode = {}
    for mode, q_input, k_input, op_scale, scale_mode in score_modes:
        scores = _qk_scores(torch, q_input, k_input, scale=op_scale)
        scores_by_mode[mode] = scores
        operations.append(
            _capture_operation(
                torch,
                capture,
                f"qk.{mode}",
                "qk",
                {"q": q_input, "k": k_input},
                scores,
                metadata={"scale": op_scale, "scale_mode": scale_mode},
            )
        )
        probabilities = torch.softmax(scores, dim=-1)
        probs_by_mode[mode] = probabilities
        operations.append(
            _capture_operation(
                torch,
                capture,
                f"softmax.{mode}",
                "softmax",
                {"scores": scores},
                probabilities,
            )
        )
        output = _pv_output(torch, probabilities, v)
        output_by_mode[mode] = output
        operations.append(
            _capture_operation(
                torch,
                capture,
                f"pv.{mode}",
                "pv",
                {"probabilities": probabilities, "v": v},
                output,
            )
        )
        if mode == "postscale":
            online = _online_from_scores(torch, scores, v)
            operations.append(
                _capture_operation(
                    torch,
                    capture,
                    "online_scores.postscale",
                    "online_scores",
                    {"scores": scores, "v": v},
                    online,
                )
            )

    qh = q.permute(0, 2, 1, 3).contiguous()
    kh = k.permute(0, 2, 1, 3).contiguous()
    vh = v.permute(0, 2, 1, 3).contiguous()
    attention = torch.nn.functional.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    ).permute(0, 2, 1, 3).contiguous()
    operations.append(
        _capture_operation(
            torch,
            capture,
            "attention.canonical_sdpa",
            "attention",
            {"q": q, "k": k, "v": v},
            attention,
            origin_expected=_input_tensor(torch, replay["attention_f32"], device),
            metadata={"reference_mode": "canonical_sdpa", "scale": scale},
        )
    )

    # PyTorch 2.8's MPS math SDPA exposes the exact probability tensor used by
    # the canonical output. This is the discriminating control for separating
    # score/softmax differences from the PV reduction.
    if torch.__version__.split("+")[0] != "2.8.0":
        raise DiagnosticError("the MPS math SDPA probability control requires PyTorch 2.8.0")
    try:
        mps_math_sdpa = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default
        schema = str(mps_math_sdpa._schema)
        expected_schema_fragment = (
            "(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, "
            "float dropout_p=0., bool is_causal=False, Tensor? dropout_mask=None, "
            "*, float? scale=None) -> (Tensor, Tensor)"
        )
        if expected_schema_fragment not in schema:
            raise DiagnosticError("PyTorch MPS math SDPA operator schema changed")
        math_output_h, math_probabilities_h = mps_math_sdpa(
            qh,
            kh,
            vh,
            None,
            0.0,
            False,
            None,
            scale=None,
        )
    except DiagnosticError:
        raise
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise DiagnosticError(f"PyTorch MPS math SDPA probability control is unavailable: {exc}") from None

    expected_probability_shape = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
    if tuple(math_output_h.shape) != tuple(qh.shape):
        raise DiagnosticError("PyTorch MPS math SDPA returned an unexpected output layout")
    if tuple(math_probabilities_h.shape) != expected_probability_shape:
        raise DiagnosticError("PyTorch MPS math SDPA returned an unexpected probability layout")
    if math_output_h.device.type != "mps" or math_probabilities_h.device.type != "mps":
        raise DiagnosticError("PyTorch MPS math SDPA probability control escaped MPS")
    math_output = math_output_h.permute(0, 2, 1, 3).contiguous()
    math_probabilities = math_probabilities_h.permute(0, 2, 1, 3).contiguous()
    operations.append(
        _capture_operation(
            torch,
            capture,
            "attention.mps_math_sdpa",
            "attention",
            {"q": q, "k": k, "v": v},
            math_output,
            origin_expected=attention,
            metadata={"reference_mode": "aten_mps_math_sdpa_with_probabilities", "scale": scale},
        )
    )
    operations.append(
        _capture_operation(
            torch,
            capture,
            "softmax.mps_math_probabilities",
            "softmax",
            {"scores": scores_by_mode["postscale"]},
            math_probabilities,
            metadata={"reference_mode": "aten_mps_math_sdpa_probabilities"},
        )
    )
    operations.append(
        _capture_operation(
            torch,
            capture,
            "pv.mps_math_probabilities",
            "pv",
            {"probabilities": math_probabilities, "v": v},
            math_output,
            origin_expected=attention,
            metadata={"reference_mode": "aten_mps_math_sdpa_probabilities"},
        )
    )
    for mode, q_input, k_input, op_scale, scale_mode in score_modes:
        operations.append(
            _capture_operation(
                torch,
                capture,
                    f"attention.explicit.{mode}",
                "attention",
                {"q": q_input, "k": k_input, "v": v},
                output_by_mode[mode],
                metadata={
                    "reference_mode": "explicit_qk_softmax_pv",
                    "scale_mode": scale_mode,
                    "scale": op_scale,
                },
            )
        )
    return operations


def _finalize_operations(
    operations: list[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    finalized = []
    for operation in operations:
        result = {
            key: value
            for key, value in operation.items()
            if not key.startswith("_")
        }
        result["inputs"] = {
            name: records[key]
            for name, key in operation["_input_keys"].items()
        }
        result["expected"] = records[operation["_expected_key"]]
        result["rounded_expected"] = records[operation["_rounded_expected_key"]]
        origin_key = operation.get("_origin_expected_key")
        if origin_key is not None:
            result["origin_expected"] = records[origin_key]
        finalized.append(result)
    return finalized


def _case_config(base: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    config = dict(base)
    config.update(case["config_overrides"])
    img_shape = case["variants"][0]["checkpoints"]["img.attn.modulated"]["shape"]
    txt_shape = case["variants"][0]["checkpoints"]["txt.attn.modulated"]["shape"]
    config.update(
        {
            "batches": img_shape[0],
            "video_tokens": img_shape[1],
            "text_tokens": txt_shape[1],
        }
    )
    required = ("batches", "video_tokens", "text_tokens", "dim", "heads", "head_dim", "ffn_dim")
    for key in required:
        if key not in config or not _is_int(config[key]) or config[key] <= 0:
            raise DiagnosticError(f"case config is missing positive {key}")
    if config["dim"] != config["heads"] * config["head_dim"]:
        raise DiagnosticError("case config dim does not equal heads times head_dim")
    config.setdefault("norm_eps", 1e-6)
    return {key: config[key] for key in (*required, "norm_eps")}


def _build_case(
    torch: Any,
    norm_layers: Any,
    posemb_layers: Any,
    source: Any,
    selected_case: dict[str, Any],
    variant: dict[str, Any],
    config: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    capture: dict[str, Any] = {}
    with torch.inference_mode():
        operations = _linear_operations(
            torch, source, selected_case, variant, config, device, capture
        )
        operations.extend(
            _norm_and_rope_operations(
                torch,
                norm_layers,
                posemb_layers,
                source,
                selected_case,
                variant,
                config,
                device,
                capture,
            )
        )
        operations.extend(
            _attention_operations(torch, variant, config, device, capture)
        )
    # Reuse the established single-transfer MPS checkpoint capture helper.
    import hy_nomas_reference as oracle

    captured = oracle._tensor_records(capture, torch)
    finalized = _finalize_operations(operations, captured)
    return {
        "label": f"{selected_case['name']}/{variant['name']}",
        "source_case": selected_case["name"],
        "source_variant": variant["name"],
        "resident_dtype": variant["resident_dtype"],
        "canonical_qkv": {
            name: captured[f"canonical_qkv.{name}"] for name in ("q", "k", "v")
        },
        "config": config,
        "operations": finalized,
    }


def _write_fresh_json(output: Path, value: dict[str, Any]) -> None:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise DiagnosticError("--output must be a fresh path")
    if output.suffix.lower() != ".json":
        raise DiagnosticError("--output must name a .json file")
    if not output.parent.is_dir():
        raise DiagnosticError("--output parent directory must already exist")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise DiagnosticError("temporary output path already exists")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        # link() publishes atomically and fails rather than replacing a racing file.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_fresh_output(output: Path) -> None:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise DiagnosticError("--output must be a fresh path")
    if output.suffix.lower() != ".json":
        raise DiagnosticError("--output must name a .json file")
    if not output.parent.is_dir():
        raise DiagnosticError("--output parent directory must already exist")


def _structure_self_test() -> None:
    record = {"shape": [2, 3], "values": [0.0] * 6}
    if _record_shape(record, context="self-test") != (2, 3):
        raise AssertionError("valid tensor record was rejected")
    for bad in (
        {"shape": [2, 3], "values": [0.0] * 5},
        {"shape": [0], "values": []},
        {"shape": [2], "values": [float("nan"), 0.0]},
    ):
        try:
            _validate_record_values(bad, context="self-test")
        except DiagnosticError:
            continue
        raise AssertionError("invalid tensor record was admitted")
    try:
        json.loads('{"x":1,"x":2}', object_pairs_hook=_object_without_duplicate_keys)
    except DiagnosticError:
        pass
    else:
        raise AssertionError("duplicate JSON keys were admitted")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="trusted frozen oracle JSON")
    parser.add_argument("--fixture-sha256", help="SHA-256 of exact fixture JSON bytes")
    parser.add_argument("--weights", type=Path, help="exact original block safetensors subset")
    parser.add_argument("--weights-sha256", help="SHA-256 of exact checkpoint subset bytes")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=DEFAULT_UPSTREAM,
        help="pinned Hy-Nomas checkout at revision " + PINNED_REVISION,
    )
    parser.add_argument("--output", type=Path, help="new diagnostic JSON path")
    parser.add_argument(
        "--structure-self-test",
        action="store_true",
        help="run offline record/schema guards without importing PyTorch or using MPS",
    )
    args = parser.parse_args(argv)
    if args.structure_self_test:
        return args
    for name in ("fixture", "fixture_sha256", "weights", "weights_sha256", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    for name in ("fixture_sha256", "weights_sha256"):
        if re.fullmatch(r"[0-9a-fA-F]{64}", getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be exactly 64 hexadecimal characters")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.structure_self_test:
        _structure_self_test()
        print(json.dumps({"structure_self_test": "pass"}, sort_keys=True))
        return 0

    fixture_path = args.fixture.resolve(strict=True)
    weights_path = args.weights.resolve(strict=True)
    _validate_fresh_output(args.output)
    if not fixture_path.is_file() or fixture_path.suffix.lower() != ".json":
        raise DiagnosticError("--fixture must name a regular .json file")
    if not weights_path.is_file() or weights_path.suffix.lower() != ".safetensors":
        raise DiagnosticError("--weights must name a regular .safetensors file")
    fixture_sha256 = _sha256(fixture_path)
    weights_sha256 = _sha256(weights_path)
    if fixture_sha256 != args.fixture_sha256.lower():
        raise DiagnosticError("fixture SHA-256 does not match --fixture-sha256")
    if weights_sha256 != args.weights_sha256.lower():
        raise DiagnosticError("checkpoint SHA-256 does not match --weights-sha256")

    # Parse/filter the large fixture before importing torch, then release all
    # unused fixture checkpoints so host memory is not duplicated by MPS setup.
    document = _read_json(fixture_path, context="source fixture")
    selected = _selected_fixture_with_weight_provenance(document)
    if selected["source_checkpoint_sha256"] != weights_sha256:
        raise DiagnosticError("checkpoint SHA-256 does not match the source fixture provenance")
    upstream_provenance = selected["provenance"]
    if upstream_provenance.get("revision") != PINNED_REVISION:
        raise DiagnosticError("fixture provenance is not from the pinned upstream revision")
    del document
    gc.collect()

    # Import the existing pinned-source verifier only after filtering. It has no
    # top-level torch import and preserves the oracle's exact source hash gate.
    support_dir = Path(__file__).resolve().parent
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))
    import hy_nomas_reference as oracle

    upstream = args.upstream.resolve()
    upstream_hashes = oracle._verify_upstream(upstream)
    fixture_hashes = upstream_provenance.get("file_sha256")
    if fixture_hashes != upstream_hashes:
        raise DiagnosticError("fixture pinned-source hashes do not match the selected upstream")

    source_dtype = selected["source_dtype"]
    if source_dtype not in ("F32", "BF16"):
        raise DiagnosticError("fixture source dtype must be F32 or BF16")
    base_config = selected["config"]
    # The subset metadata can be checked without loading its tensor payloads.
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise DiagnosticError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
    except ImportError as exc:
        raise DiagnosticError("PyTorch is required to emit the MPS diagnostic") from exc
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise DiagnosticError("PyTorch MPS is required; CPU fallback is forbidden")
    device = torch.device("mps")

    first_config = _case_config(base_config, selected["cases"][0])
    _verify_weights_file(weights_path, first_config, source_dtype)
    helpers = oracle._install_upstream_modules(upstream)
    norm_layers = helpers[1]
    posemb_layers = helpers[-1]
    cases = []
    try:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            for selected_case in selected["cases"]:
                case_config = _case_config(base_config, selected_case)
                if case_config != first_config:
                    _verify_weights_file(weights_path, case_config, source_dtype)
                for variant in selected_case["variants"]:
                    cases.append(
                        _build_case(
                            torch,
                            norm_layers,
                            posemb_layers,
                            source,
                            selected_case,
                            variant,
                            case_config,
                            device,
                        )
                    )
    except DiagnosticError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise DiagnosticError(f"MPS diagnostic reference failed: {exc}") from None

    output = {
        "schema_version": 1,
        "kind": REFERENCE_KIND,
        "upstream_revision": PINNED_REVISION,
        "source_fixture_sha256": fixture_sha256,
        "source_checkpoint_sha256": weights_sha256,
        "source_dtype": source_dtype,
        "weights_are_synthetic": selected["weights_are_synthetic"],
        "provenance": {
            "upstream_revision": PINNED_REVISION,
            "upstream_source_sha256": upstream_hashes,
            "source_fixture_provenance": upstream_provenance,
            "reference_device": "mps",
            "compute_dtype": "F32",
            "activation_storage": "F32",
            "canonical_qkv_boundary": "BF16_RNE_roundtrip_F32",
            "effective_weight_policy": "source_to_F32_or_source_to_BF16_RNE_to_F32_on_MPS",
            "weight_rounding": "resident BF16 control casts MPS F32 weights through BF16 back to F32",
            "attention_reference": "PyTorch MPS SDPA plus explicit diagnostic decompositions; not Magi parity",
            "producer": Path(__file__).name,
            "torch_version": torch.__version__,
        },
        "scope": {
            "source_reference_kind": "upstream_block_bf16_boundary_control_mps_sdpa",
            "single_full_segment": True,
            "all_text_valid": True,
            "history_or_cache": False,
            "mask": False,
            "full_generation": False,
        },
        "config": base_config,
        "cases": cases,
    }
    _write_fresh_json(args.output, output)
    print(
        json.dumps(
            {
                "status": "created",
                "output": str(args.output.resolve()),
                "cases": len(cases),
                "upstream_revision": PINNED_REVISION,
                "device": "mps",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        raise SystemExit(str(exc)) from None
