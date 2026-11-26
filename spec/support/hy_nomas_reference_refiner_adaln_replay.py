#!/usr/bin/env python3
"""Replay one captured text-refiner adaLN projection on MPS.

This diagnostic authenticates the existing native/source block traces, then
recomputes only block 0's 2048-to-4096 adaLN projection from its captured
condition_silu operand and exact BF16 global checkpoint weights. It performs
no refiner/block forward, no attention, and no full generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping

try:
    from . import hy_nomas_reference_refiner_blocks_compare as blocks
except ImportError:
    import hy_nomas_reference_refiner_blocks_compare as blocks

shared = blocks.shared
Error = blocks.Error
SCHEMA = "cogni_ml.hy_nomas_reference_refiner_adaln_replay/v1"
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
ATOL, RTOL = blocks.ATOL, blocks.RTOL
SEED = 20261103
INPUT = "block0.condition_silu"
OUTPUT = "block0.adaLN"
INPUT_SHAPE = (1, 2048)
OUTPUT_SHAPE = (1, 4096)
WEIGHT_NAME = "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight"
BIAS_NAME = "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.bias"
WEIGHT_SHAPE = (4096, 2048)
BIAS_SHAPE = (4096,)
REQUIRED_UPSTREAM_FILES = (
    "hymm/models/dit/nomas_cpt.py",
    "hymm/models/dit/modules/activation_layers.py",
    "hymm/models/dit/modules/norm_layers.py",
    "hymm/models/dit/modules/modulate_layers.py",
    "hymm/models/dit/modules/mlp_layers.py",
    "hymm/models/dit/modules/posemb_layers.py",
    "hymm/models/dit/modules/attention/__init__.py",
    "hymm/models/dit/modules/attention/magi_attn.py",
    "hymm/utils/helpers.py",
    "hymm/models/text_encoder/token_refiner.py",
    "hymm/models/dit/modules/embed_layers.py",
    "hymm/models/dit/modules/int8_stored_bf16_linear.py",
    "hymm/models/vae/pixelumm_decoder/model.py",
    "hymm/models/vae/pixelumm_decoder/autoencoder.py",
    "hymm/pipeline/causal_ar.py",
    "configs/experiment/inference.yaml",
    "hymm/models/dit/modules/kv_cache_storage.py",
    "hymm/pipeline/solvers/euler.py",
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CANDIDATE_NAMES = (
    "native_f32_linear_raw_input",
    "native_f32_linear_bf16_rounded_input",
    "source_f32_linear_bf16_input",
    "native_bf16_linear_rounded_input",
    "source_bf16_linear_source_input",
    "native_f32_linear_raw_input_output_bf16_rounded",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Error(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_absolute_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise Error(f"{label} must be an absolute regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise Error(f"cannot resolve {label}: {error}") from None


def _validate_source_file_pins(upstream_arg: str | Path,
                               source_identity: Mapping[str, Any]) -> tuple[Path, dict[str, str]]:
    upstream = Path(upstream_arg).expanduser()
    if not upstream.is_absolute() or upstream.is_symlink() or not upstream.is_dir():
        raise Error("--upstream must be an absolute regular non-symlink directory")
    upstream = upstream.resolve(strict=True)
    if source_identity.get("upstream_revision") != PINNED_REVISION:
        raise Error("source refiner block is not pinned to the admitted upstream revision")
    pins = source_identity.get("upstream_source_sha256")
    if type(pins) is not dict or set(pins) != set(REQUIRED_UPSTREAM_FILES):
        raise Error("source upstream pin inventory differs from the admitted refiner source set")
    for relative in REQUIRED_UPSTREAM_FILES:
        digest = pins.get(relative)
        if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
            raise Error(f"invalid upstream source pin: {relative}")
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts or "." in rel.parts:
            raise Error(f"unsafe upstream source path: {relative}")
        path = upstream.joinpath(*rel.parts)
        if path.is_symlink() or not path.is_file():
            raise Error(f"pinned upstream source is missing or non-canonical: {relative}")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(upstream)
        except (OSError, RuntimeError, ValueError):
            raise Error(f"pinned upstream source escapes its checkout: {relative}") from None
        if _sha256_file(resolved) != digest:
            raise Error(f"pinned upstream source digest mismatch: {relative}")
    return upstream, dict(pins)


def _authenticate_global_weights(path_arg: str | Path, expected_sha256: str,
                                source_identity: Mapping[str, Any]) -> Path:
    expected = source_identity.get("global_shard_sha256")
    if (not isinstance(expected, str) or HEX_SHA256.fullmatch(expected) is None
            or not isinstance(expected_sha256, str) or HEX_SHA256.fullmatch(expected_sha256) is None
            or expected_sha256 != expected):
        raise Error("global checkpoint digest is not the linked source-BF16 checkpoint identity")
    path = _canonical_absolute_file(path_arg, "global BF16 checkpoint")
    if _sha256_file(path) != expected:
        raise Error("global BF16 checkpoint bytes differ from the linked source identity")
    return path


def _validate_operand_contract(inputs: Mapping[str, Any]) -> dict[str, Any]:
    native = inputs["native_document"]
    source = inputs["source_document"]
    valid = inputs["valid_text_tokens"]
    _require(valid == 263, "adaLN replay is scoped to the admitted T0 V263 input")
    for kind, document in (("native", native), ("source", source)):
        for name, shape in ((INPUT, INPUT_SHAPE), (OUTPUT, OUTPUT_SHAPE)):
            record = document["tensors"].get(name)
            _require(isinstance(record, dict) and record.get("shape") == list(shape),
                     f"{kind} replay tensor has an unexpected shape: {name}")
    for name in ("block0.input", "c"):
        _require(native["tensors"][name]["sha256"] == source["tensors"][name]["sha256"],
                 f"adaLN replay requires exact linked source/native input identity: {name}")
    source_input = source["tensors"][INPUT]
    source_output = source["tensors"][OUTPUT]
    _require(source_input.get("source_dtype") == "BF16"
             and source_input.get("source_checkpoint") == INPUT
             and source_input.get("source_shape") == list(INPUT_SHAPE),
             "source adaLN input is not the observed BF16 condition_silu checkpoint")
    _require(source_output.get("source_dtype") == "BF16"
             and source_output.get("source_checkpoint") == OUTPUT
             and source_output.get("source_shape") == list(OUTPUT_SHAPE),
             "source adaLN output is not the observed BF16 checkpoint")
    # Do not require the native/source activation hashes to agree. Their raw
    # and BF16-rounded comparisons are the discriminator under investigation.
    return {
        "native_condition_silu_sha256": native["tensors"][INPUT]["sha256"],
        "source_condition_silu_sha256": source_input["sha256"],
        "native_adaLN_sha256": native["tensors"][OUTPUT]["sha256"],
        "source_adaLN_sha256": source_output["sha256"],
        "condition_silu_raw_hash_equal": (
            native["tensors"][INPUT]["sha256"] == source_input["sha256"]
        ),
    }


def validate_inputs(args: Any) -> dict[str, Any]:
    """Admit block/prelude/cache inputs and bind exact source files/weights."""
    admitted = blocks.validate_inputs(args)
    operand_identity = _validate_operand_contract(admitted)
    source_identity = admitted["prelude_admission"]["source_document"].get("source", {})
    if (not isinstance(source_identity.get("global_shard_sha256"), str)
            or HEX_SHA256.fullmatch(source_identity["global_shard_sha256"]) is None):
        raise Error("linked source prelude lacks the authenticated global checkpoint identity")
    upstream, source_pins = _validate_source_file_pins(args.upstream,
                                                        admitted["source_document"]["source"])
    weights = _authenticate_global_weights(args.global_weights, args.global_weights_sha256,
                                            source_identity)
    return {
        **admitted,
        "operand_identity": operand_identity,
        "source_identity": source_identity,
        "source_pins": source_pins,
        "upstream_path": upstream,
        "global_weights_path": weights,
        "global_weights_sha256": args.global_weights_sha256,
    }


def _load_projection_weights(torch: Any, path: Path, device: Any) -> tuple[Any, Any]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise Error("safetensors is required to load the pinned BF16 adaLN weights") from error
    try:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
            if WEIGHT_NAME not in keys or BIAS_NAME not in keys:
                raise Error("pinned global checkpoint is missing the block-0 adaLN linear tensors")
            weight = checkpoint.get_tensor(WEIGHT_NAME)
            bias = checkpoint.get_tensor(BIAS_NAME)
    except Error:
        raise
    except Exception as error:
        raise Error(f"could not read pinned BF16 adaLN weights: {error}") from None
    if (weight.dtype != torch.bfloat16 or tuple(weight.shape) != WEIGHT_SHAPE
            or bias.dtype != torch.bfloat16 or tuple(bias.shape) != BIAS_SHAPE):
        raise Error("pinned BF16 adaLN weight or bias dtype/shape mismatch")
    return weight.to(device=device).contiguous(), bias.to(device=device).contiguous()


def _run_mps(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    version = str(torch.__version__).split("+", 1)[0]
    if not version.startswith("2.8."):
        raise Error(f"adaLN replay requires PyTorch 2.8.x; found {torch.__version__}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise Error("adaLN replay requires an available PyTorch MPS device")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
        raise Error("MPS fallback must be disabled for the adaLN replay")

    device = torch.device("mps")
    native_doc = inputs["native_document"]
    source_doc = inputs["source_document"]
    native_input = shared._load_tensor(
        torch, np, inputs["native_payload"], native_doc["tensors"][INPUT], device,
    )
    source_input = shared._load_tensor(
        torch, np, inputs["source_payload"], source_doc["tensors"][INPUT], device,
    )
    native_observed = shared._load_tensor(
        torch, np, inputs["native_payload"], native_doc["tensors"][OUTPUT], device,
    )
    source_observed = shared._load_tensor(
        torch, np, inputs["source_payload"], source_doc["tensors"][OUTPUT], device,
    )
    weight_bf16, bias_bf16 = _load_projection_weights(
        torch, inputs["global_weights_path"], device,
    )
    weight_f32 = weight_bf16.to(dtype=torch.float32)
    bias_f32 = bias_bf16.to(dtype=torch.float32)
    functional = torch.nn.functional
    queued: list[tuple[str, Any, Any, Any, float, float]] = []

    def compare(name: str, actual: Any, expected: Any, *, atol: float = ATOL,
                rtol: float = RTOL) -> None:
        if actual.device.type != "mps" or expected.device.type != "mps":
            raise Error("adaLN metric escaped MPS")
        queued.append((name, shared._metric_stat_tensors(
            torch, actual, expected, atol=atol, rtol=rtol,
        ), actual, expected, atol, rtol))

    with torch.inference_mode():
        native_input_bf16 = native_input.to(dtype=torch.bfloat16)
        source_input_bf16 = source_input.to(dtype=torch.bfloat16)
        native_input_bf16_f32 = native_input_bf16.to(dtype=torch.float32)
        source_input_bf16_f32 = source_input_bf16.to(dtype=torch.float32)
        native_f32 = functional.linear(native_input, weight_f32, bias_f32)
        native_f32_no_bias = functional.linear(native_input, weight_f32, None)
        native_f32_separate_bias = native_f32_no_bias + bias_f32
        native_rounded_f32 = functional.linear(native_input_bf16_f32, weight_f32, bias_f32)
        native_rounded_no_bias = functional.linear(native_input_bf16_f32, weight_f32, None)
        native_rounded_separate_bias = native_rounded_no_bias + bias_f32
        source_f32 = functional.linear(source_input, weight_f32, bias_f32)
        native_bf16 = functional.linear(native_input_bf16, weight_bf16, bias_bf16)
        source_bf16 = functional.linear(source_input_bf16, weight_bf16, bias_bf16)
        native_f32_output_bf16 = native_f32.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        native_rounded_output_bf16 = native_rounded_f32.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        native_rounded_separate_bias_output_bf16 = (
            native_rounded_separate_bias.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        )

        if any(output.dtype != torch.float32 for output in (
                native_f32, native_f32_no_bias, native_f32_separate_bias,
                native_rounded_f32, native_rounded_no_bias, native_rounded_separate_bias,
                source_f32)):
            raise Error("F32 adaLN candidate returned an unexpected dtype")
        if any(output.dtype != torch.bfloat16 for output in (native_bf16, source_bf16)):
            raise Error("BF16 functional.linear candidate returned an unexpected dtype")
        for name, tensor in (("native F32 linear", native_f32),
                             ("native BF16 linear", native_bf16),
                             ("source BF16 linear", source_bf16)):
            if tuple(tensor.shape) != OUTPUT_SHAPE:
                raise Error(f"{name} returned an unexpected adaLN shape")

        # Operand diagnostics distinguish a raw activation mismatch from one
        # that survives conversion to the source BF16 representation.
        compare("input_raw_native_vs_source", native_input, source_input)
        compare("input_bf16_rounded_native_vs_source_exact",
                native_input_bf16_f32, source_input, atol=0.0, rtol=0.0)
        compare("input_source_bf16_roundtrip_exact",
                source_input_bf16_f32, source_input, atol=0.0, rtol=0.0)
        compare("observed_native_adaLN_vs_source", native_observed, source_observed)
        compare("observed_native_adaLN_bf16_round_vs_source_exact",
                native_observed.to(dtype=torch.bfloat16).to(dtype=torch.float32),
                source_observed, atol=0.0, rtol=0.0)

        # F32 candidates measure native functional.linear faithfulness,
        # input-rounding impact, and a separate-bias control.
        compare("native_f32_linear_raw_input_vs_native", native_f32, native_observed)
        compare("native_f32_linear_raw_input_vs_source", native_f32, source_observed)
        compare("native_f32_linear_bf16_rounded_input_vs_source",
                native_rounded_f32, source_observed)
        compare("native_f32_linear_bf16_rounded_input_output_bf16_round_vs_source",
                native_rounded_output_bf16, source_observed)
        compare("native_f32_linear_bf16_rounded_input_output_bf16_round_vs_source_exact",
                native_rounded_output_bf16, source_observed, atol=0.0, rtol=0.0)
        compare("native_f32_rounded_input_bias_epilogue_vs_separate_bias_exact",
                native_rounded_f32, native_rounded_separate_bias, atol=0.0, rtol=0.0)
        compare("native_f32_rounded_input_separate_bias_output_bf16_round_vs_source_exact",
                native_rounded_separate_bias_output_bf16, source_observed, atol=0.0, rtol=0.0)
        compare("source_f32_linear_bf16_input_vs_source", source_f32, source_observed)
        compare("f32_addmm_bias_epilogue_vs_separate_bias_exact",
                native_f32, native_f32_separate_bias, atol=0.0, rtol=0.0)
        compare("f32_raw_linear_output_bf16_round_vs_source_exact",
                native_f32_output_bf16, source_observed, atol=0.0, rtol=0.0)
        compare("f32_raw_linear_output_bf16_round_vs_native",
                native_f32_output_bf16, native_observed)

        # The BF16 paths use the exact source file parameters and report both
        # source-reference reproduction and native-connected behavior.
        compare("native_bf16_linear_rounded_input_vs_source",
                native_bf16.to(dtype=torch.float32), source_observed)
        compare("native_bf16_linear_rounded_input_vs_native",
                native_bf16.to(dtype=torch.float32), native_observed)
        compare("source_bf16_linear_source_input_vs_source",
                source_bf16.to(dtype=torch.float32), source_observed)
        compare("source_bf16_linear_source_input_vs_native",
                source_bf16.to(dtype=torch.float32), native_observed)

        # A/A and a deterministic seeded perturbation exercise the same MPS
        # metric path without introducing host-generated neural values.
        control = torch.arange(16, device=device, dtype=torch.float32)
        perturbation = torch.sin(control * 0.37 + float(SEED)) * 0.01
        compare("sensitivity_a_vs_a", control, control, atol=0.0, rtol=0.0)
        compare("sensitivity_seeded_perturbation", control + perturbation,
                control, atol=0.0, rtol=0.0)

    flat = [stats[key] for _name, stats, _actual, _expected, _atol, _rtol in queued
            for key in shared.METRIC_KEYS]
    # All neural outputs and metric reductions remain on MPS until this one
    # packed terminal export.
    values = torch.stack(flat).detach().cpu().tolist()
    stride = len(shared.METRIC_KEYS)
    rows = [dict(kind=name, **shared._stats_record(
        values[index*stride:(index+1)*stride], atol=atol, rtol=rtol,
    )) for index, (name, _stats, _actual, _expected, atol, rtol) in enumerate(queued)]
    by_kind = {row["kind"]: row for row in rows}
    input_raw = by_kind["input_raw_native_vs_source"]
    input_rounded = by_kind["input_bf16_rounded_native_vs_source_exact"]
    sensitivity_ok = (by_kind["sensitivity_a_vs_a"]["passed"]
                      and not by_kind["sensitivity_seeded_perturbation"]["passed"])
    return {
        "device": "mps",
        "torch_version": torch.__version__,
        "numeric_tolerance": {"atol": ATOL, "rtol": RTOL},
        "raw_native_vs_source_input_failure_count": input_raw["mismatch_count"],
        "raw_native_vs_source_adaLN_failure_count": by_kind[
            "observed_native_adaLN_vs_source"]["mismatch_count"],
        "bf16_rounded_operand_mismatch_count": input_rounded["mismatch_count"],
        "bf16_rounded_operand_exact_equal": input_rounded["passed"],
        "raw_condition_silu_hash_equal": inputs["operand_identity"]["condition_silu_raw_hash_equal"],
        "source_input_bf16_roundtrip_passed": by_kind["input_source_bf16_roundtrip_exact"]["passed"],
        "source_bf16_linear_replay_passed": by_kind[
            "source_bf16_linear_source_input_vs_source"]["passed"],
        "f32_raw_linear_native_faithful": by_kind[
            "native_f32_linear_raw_input_vs_native"]["passed"],
        "bf16_output_round_mismatch_count": by_kind[
            "f32_raw_linear_output_bf16_round_vs_source_exact"]["mismatch_count"],
        "native_adaLN_bf16_round_mismatch_count": by_kind[
            "observed_native_adaLN_bf16_round_vs_source_exact"]["mismatch_count"],
        "native_rounded_f32_output_bf16_round_mismatch_count": by_kind[
            "native_f32_linear_bf16_rounded_input_output_bf16_round_vs_source_exact"]["mismatch_count"],
        "f32_bias_epilogue_exact_equal": by_kind[
            "f32_addmm_bias_epilogue_vs_separate_bias_exact"]["passed"],
        "sensitivity_controls_passed": sensitivity_ok,
        "terminal_cpu_exports": 1,
        "upstream_revision": PINNED_REVISION,
        "source_pins_verified": len(inputs["source_pins"]),
        "checkpoint_tensor_names": {"weight": WEIGHT_NAME, "bias": BIAS_NAME},
        "checkpoint_tensor_shapes": {"weight": list(WEIGHT_SHAPE), "bias": list(BIAS_SHAPE)},
        "candidate_inventory": list(CANDIDATE_NAMES),
        "rows": rows,
        "interpretation": (
            "One authenticated T0 V263 block-0 condition_silu operand and exact source BF16 "
            "global checkpoint weights. The source BF16 replay is a same-MPS functional.linear "
            "control; native F32 candidates independently measure captured-native faithfulness, "
            "input rounding, output rounding, and separate-bias effects. Source checkpoint rows "
            "remain the only observed source activations. No full refiner/block execution, "
            "attention, K/V, DiT, motion, CUDA, or generation claim follows."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "native", "native-sha256", "source", "source-sha256", "native-refiner",
        "native-refiner-sha256", "source-refiner", "source-refiner-sha256",
        "native-preblock", "native-preblock-sha256", "reference-latent",
        "reference-latent-sha256", "timeline-sha256", "upstream", "global-weights",
        "global-weights-sha256", "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        output = shared.output_path(args.output)
        inputs = validate_inputs(args)
        result = _run_mps(inputs)
        document = dict(
            schema=SCHEMA,
            schema_version=1,
            native_manifest_sha256=args.native_sha256,
            native_payload_sha256=inputs["native_document"]["tensor_artifact"]["sha256"],
            source_manifest_sha256=args.source_sha256,
            source_payload_sha256=inputs["source_document"]["payload_sha256"],
            native_refiner_manifest_sha256=args.native_refiner_sha256,
            source_refiner_manifest_sha256=args.source_refiner_sha256,
            native_preblock_manifest_sha256=args.native_preblock_sha256,
            reference_latent_artifact_sha256=args.reference_latent_sha256,
            timeline_manifest_sha256=args.timeline_sha256,
            global_weights_sha256=inputs["global_weights_sha256"],
            global_weights_basename=inputs["global_weights_path"].name,
            source_conditioning_manifest_sha256=inputs["source_identity"][
                "conditioning_manifest_sha256"],
            source_upstream_source_sha256=inputs["source_pins"],
            operand_identity=inputs["operand_identity"],
            **result,
        )
        shared._publish_fresh(output, document)
    except (Error, shared.native_cache.ReferenceCacheError, OSError, ValueError,
            RuntimeError, ImportError) as error:
        print(json.dumps(dict(schema=SCHEMA, status="rejected", reason=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(dict(
        schema=SCHEMA,
        output=str(output),
        raw_native_vs_source_input_failure_count=result["raw_native_vs_source_input_failure_count"],
        raw_native_vs_source_adaLN_failure_count=result["raw_native_vs_source_adaLN_failure_count"],
        bf16_rounded_operand_mismatch_count=result["bf16_rounded_operand_mismatch_count"],
        bf16_rounded_operand_exact_equal=result["bf16_rounded_operand_exact_equal"],
        native_adaLN_bf16_round_mismatch_count=result["native_adaLN_bf16_round_mismatch_count"],
        native_rounded_f32_output_bf16_round_mismatch_count=(
            result["native_rounded_f32_output_bf16_round_mismatch_count"]
        ),
        f32_raw_linear_native_faithful=result["f32_raw_linear_native_faithful"],
        sensitivity_controls_passed=result["sensitivity_controls_passed"],
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
