#!/usr/bin/env python3
"""Replay matched block-0 Q/K/V on MPS without a refiner/model forward.

The original MPS adapter uses query chunks 256 + 7. PyTorch 2.8's last
seven queries use its vector kernel, whose second return is not probabilities.
Only the general (Q > 8) math operator's second return is consumed here.
Explicit QK/softmax/PV tensors are diagnostic decompositions, not observations
of the fused source backend. Their faithfulness is measured, never assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

try:
    from . import hy_nomas_reference_refiner_adaln_replay as admission
except ImportError:
    import hy_nomas_reference_refiner_adaln_replay as admission

blocks, shared, Error = admission.blocks, admission.shared, admission.Error
SCHEMA = "cogni_ml.hy_nomas_refiner_attention_replay/v1"
ATOL, RTOL = blocks.ATOL, blocks.RTOL
VALID, STORED, HEADS, DIM, CHUNK = 263, 1280, 16, 128, 256
GEOMETRY = dict(batches=1, query_rows=VALID, key_rows=VALID, heads=HEADS, head_dim=DIM)
SOURCE_MASK = "compacted valid prefix; no SDPA mask"
PROBABILITIES_SCOPE = "actual general-263 operator; tail7 is a counterfactual to the source vector kernel"
EXPLICIT_SCOPE = "separate MPS ops, not captured backend internals"
ARGUMENTS = (
    "native", "native-sha256", "source", "source-sha256", "native-refiner",
    "native-refiner-sha256", "source-refiner", "source-refiner-sha256",
    "native-preblock", "native-preblock-sha256", "reference-latent",
    "reference-latent-sha256", "timeline-sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Error(message)


def _record_bytes(payload: bytes, record: Mapping[str, Any]) -> bytes:
    return payload[record["offset"]:record["offset"] + record["nbytes"]]


def _validate_operands(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(inputs["valid_text_tokens"] == VALID, "attention replay requires T0 V263")
    native, source = inputs["native_document"], inputs["source_document"]
    _require(native.get("qkv_bf16_control") is True, "attention replay requires QKV control")
    identity = {}
    for name in ("block0.q", "block0.k", "block0.v"):
        nr, sr = native["tensors"][name], source["tensors"][name]
        for label, record in (("native", nr), ("source", sr)):
            _require(record.get("shape") == [1, STORED, HEADS*DIM],
                     f"{label} QKV shape differs: {name}")
        _require(sr.get("source_dtype") == "BF16", f"source QKV is not BF16: {name}")
        _require(nr["sha256"] == sr["sha256"] and
                 _record_bytes(inputs["native_payload"], nr) ==
                 _record_bytes(inputs["source_payload"], sr),
                 f"QKV must be source-exact bytes: {name}")
        identity[name] = sr["sha256"]
    return identity


def validate_inputs(args: Any) -> dict[str, Any]:
    inputs = blocks.validate_inputs(args)
    identity = _validate_operands(inputs)
    upstream, pins = admission._validate_source_file_pins(args.upstream,
                                                         inputs["source_document"]["source"])
    return {**inputs, "qkv_identity": identity, "source_pins": pins,
            "upstream_path": upstream}


def _general_math(torch: Any, q: Any, k: Any, v: Any) -> tuple[Any, Any]:
    # The vector path returns an uninitialized second tensor. Reject that
    # geometry rather than mistaking it for a softmax observation.
    _require(q.shape[-2] > 8, "general probability witness requires more than eight queries")
    return torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(
        q, k, v, None, 0.0, False, None, scale=None,
    )


def _chunked_sdpa(torch: Any, q: Any, k: Any, v: Any) -> Any:
    return torch.cat([
        torch.nn.functional.scaled_dot_product_attention(
            q[:, :, start:min(start+CHUNK, VALID)], k, v,
            attn_mask=None, dropout_p=0.0, is_causal=False,
        ) for start in range(0, VALID, CHUNK)
    ], dim=2)


def _run_mps(inputs: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    import numpy as np
    import torch

    _require(str(torch.__version__).split("+", 1)[0].startswith("2.8."),
             "attention replay requires PyTorch 2.8.x")
    _require(torch.backends.mps.is_available(), "attention replay requires MPS")
    _require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") in ("", "0"),
             "MPS fallback must be disabled")
    device = torch.device("mps")
    tensors: dict[str, Any] = {}
    queued = []

    def compare(name: str, actual: Any, expected: Any, *, exact: bool = False) -> None:
        _require(actual.device.type == expected.device.type == "mps", "metric escaped MPS")
        atol, rtol = (0.0, 0.0) if exact else (ATOL, RTOL)
        queued.append((name, shared._metric_stat_tensors(
            torch, actual.to(torch.float32), expected.to(torch.float32), atol=atol, rtol=rtol,
        ), atol, rtol))

    def stream(tensor: Any) -> Any:
        return tensor.permute(0, 2, 1, 3).contiguous().to(torch.float32)

    def load(kind: str, name: str) -> Any:
        return shared._load_tensor(torch, np, inputs[f"{kind}_payload"],
                                   inputs[f"{kind}_document"]["tensors"][name], device)

    with torch.inference_mode():
        q, k, v = [load("native", f"block0.{name}")[:, :VALID].reshape(
            1, VALID, HEADS, DIM).permute(0, 2, 1, 3).contiguous() for name in ("q", "k", "v")]
        qb, kb, vb = [value.to(torch.bfloat16) for value in (q, k, v)]
        for name, value in zip(("q", "k", "v"), (q, k, v)):
            tensors[name] = stream(value)
            compare(f"{name}_bf16_roundtrip", value, value.to(torch.bfloat16).to(torch.float32), exact=True)

        source_observed = load("source", "block0.attention").reshape(1, STORED, HEADS, DIM)
        native_observed = load("native", "block0.attention").reshape(1, STORED, HEADS, DIM)
        tensors["observed_source"] = source_observed[:, :VALID].contiguous()
        tensors["observed_native"] = native_observed[:, :VALID].contiguous()
        source_chunked = _chunked_sdpa(torch, qb, kb, vb)
        source_chunked_aa = _chunked_sdpa(torch, qb, kb, vb)
        f32_chunked = _chunked_sdpa(torch, q, k, v)
        source_general, source_p = _general_math(torch, qb, kb, vb)
        f32_general, f32_p = _general_math(torch, q, k, v)
        main_context, main_p = _general_math(torch, qb[:, :, :CHUNK], kb, vb)
        _require(tuple(source_p.shape) == (1, HEADS, VALID, VALID)
                 and source_p.dtype == torch.bfloat16, "general BF16 operator P contract differs")
        _require(tuple(main_p.shape) == (1, HEADS, CHUNK, VALID), "main-chunk probability shape differs")

        scale = 1.0/math.sqrt(DIM)
        # Separate operations expose logical BF16 boundaries from the pinned
        # PyTorch MPSGraph implementation; they are not hidden backend capture.
        bf_raw = torch.matmul(qb, kb.transpose(-1, -2))
        bf_scaled = (bf_raw.to(torch.float32)*scale).to(torch.bfloat16)
        bf_p = torch.softmax(bf_scaled, dim=-1)
        bf_pv = torch.matmul(bf_p, vb)
        f32_raw = torch.matmul(q, k.transpose(-1, -2))
        f32_scaled = f32_raw*scale
        f32_softmax = torch.softmax(f32_scaled, dim=-1)
        f32_pv = torch.matmul(f32_softmax, v)
        common_pv = torch.matmul(source_p.to(torch.float32), v)
        # MPSGraph may fuse the cast/scale corridor. These distinct candidate
        # boundaries must reproduce actual source P before explaining it.
        scaled_only_bf16 = f32_scaled.to(torch.bfloat16)
        scaled_only_p = torch.softmax(scaled_only_bf16, dim=-1)
        scaled_only_pv = torch.matmul(scaled_only_p, vb)
        scaled_only_f32_softmax = torch.softmax(scaled_only_bf16.to(torch.float32), dim=-1).to(torch.bfloat16)
        raw_round_only_p = torch.softmax(bf_raw.to(torch.float32)*scale, dim=-1).to(torch.bfloat16)
        for name, value in (
            ("source_qk_raw", bf_raw), ("source_qk_scaled", bf_scaled),
            ("source_softmax", bf_p), ("source_pv", bf_pv),
            ("source_probabilities", source_p), ("source_sdpa_general", source_general),
            ("source_sdpa_chunked", source_chunked),
            ("source_main_probabilities", main_p), ("source_main_context", main_context),
            ("f32_qk_raw", f32_raw), ("f32_qk_scaled", f32_scaled),
            ("f32_softmax", f32_softmax), ("f32_pv", f32_pv),
            ("f32_sdpa_general", f32_general), ("f32_sdpa_chunked", f32_chunked),
            ("f32_pv_source_probabilities", common_pv),
            ("scaled_only_bf16", scaled_only_bf16), ("scaled_only_p", scaled_only_p),
            ("scaled_only_pv", scaled_only_pv),
            ("scaled_only_f32_softmax_p", scaled_only_f32_softmax),
            ("raw_round_only_p", raw_round_only_p),
        ):
            tensors[name] = stream(value)

        compare("source_chunked_vs_observed", stream(source_chunked), tensors["observed_source"])
        compare("source_chunked_a_vs_a", source_chunked, source_chunked_aa, exact=True)
        compare("source_general_vs_observed", stream(source_general), tensors["observed_source"])
        compare("source_general_vs_chunked_main256", source_general[:, :, :CHUNK], source_chunked[:, :, :CHUNK])
        compare("source_general_vs_chunked_tail7", source_general[:, :, CHUNK:], source_chunked[:, :, CHUNK:])
        compare("source_main_general_vs_chunked_main", main_context, source_chunked[:, :, :CHUNK], exact=True)
        compare("source_full_vs_main_probabilities", source_p[:, :, :CHUNK], main_p, exact=True)
        compare("explicit_bf16_softmax_vs_general_p", bf_p, source_p)
        compare("explicit_bf16_pv_vs_general", bf_pv, source_general)
        compare("explicit_bf16_softmax_vs_main_p", bf_p[:, :, :CHUNK], main_p)
        compare("explicit_bf16_pv_vs_observed_main", stream(bf_pv)[:, :CHUNK], tensors["observed_source"][:, :CHUNK])
        compare("f32_general_vs_observed_native", stream(f32_general), tensors["observed_native"])
        compare("f32_explicit_pv_vs_observed_native", stream(f32_pv), tensors["observed_native"])
        compare("f32_chunked_vs_general", f32_chunked, f32_general)
        compare("f32_explicit_softmax_vs_general_p", f32_softmax, f32_p)
        compare("f32_raw_vs_bf16_raw", f32_raw, bf_raw)
        compare("f32_raw_bf16_round_vs_bf16_raw", f32_raw.to(torch.bfloat16), bf_raw, exact=True)
        compare("f32_scaled_vs_bf16_scaled", f32_scaled, bf_scaled)
        compare("f32_softmax_vs_source_general_p", f32_softmax, source_p)
        compare("common_pv_f32_vs_source_general", common_pv, source_general)
        compare("common_pv_f32_bf16_round_vs_source_general", common_pv.to(torch.bfloat16), source_general, exact=True)
        compare("scaled_only_p_vs_general_p", scaled_only_p, source_p, exact=True)
        compare("scaled_only_pv_vs_general", scaled_only_pv, source_general, exact=True)
        compare("scaled_only_pv_vs_observed_main", stream(scaled_only_pv)[:, :CHUNK], tensors["observed_source"][:, :CHUNK], exact=True)
        compare("scaled_only_f32_softmax_vs_general_p", scaled_only_f32_softmax, source_p, exact=True)
        compare("raw_round_only_p_vs_general_p", raw_round_only_p, source_p, exact=True)
        compare("raw_round_only_p_vs_general_p_main", raw_round_only_p[:, :, :CHUNK], main_p)
        compare("f32_p_bf16_round_vs_general_p", f32_softmax.to(torch.bfloat16), source_p, exact=True)
        compare("source_padding_zero", source_observed[:, VALID:], torch.zeros_like(source_observed[:, VALID:]), exact=True)
        compare("native_padding_zero", native_observed[:, VALID:], torch.zeros_like(native_observed[:, VALID:]), exact=True)
        compare("general_p_row_sum", source_p.to(torch.float32).sum(-1), torch.ones((1, HEADS, VALID), device=device))
        control = torch.arange(16, device=device, dtype=torch.float32)
        compare("sensitivity_a_vs_a", control, control, exact=True)
        compare("sensitivity_perturbation", control+0.01, control, exact=True)

        # One packed terminal export contains tensors and GPU-reduced metrics.
        offsets, parts, cursor = {}, [], 0
        for name, value in tensors.items():
            _require(value.device.type == "mps" and value.dtype == torch.float32,
                     f"packet tensor escaped F32 MPS: {name}")
            count = value.numel()
            offsets[name] = (cursor, count, list(value.shape))
            parts.append(value.flatten())
            cursor += count
        metric_values = torch.stack([stats[key] for _, stats, _, _ in queued for key in shared.METRIC_KEYS])
        packed = torch.cat([*parts, metric_values]).detach().cpu().numpy()

    payload = packed[:cursor].astype("<f4", copy=False).tobytes()
    records = {}
    for name, (start, count, shape) in offsets.items():
        data = payload[start*4:(start+count)*4]
        records[name] = dict(dtype="F32", shape=shape, offset=start*4, nbytes=count*4,
                             sha256=hashlib.sha256(data).hexdigest())
    values, stride = packed[cursor:].tolist(), len(shared.METRIC_KEYS)
    rows = [dict(kind=name, **shared._stats_record(values[i*stride:(i+1)*stride], atol=atol, rtol=rtol))
            for i, (name, _, atol, rtol) in enumerate(queued)]
    by_name = {row["kind"]: row for row in rows}
    _require(by_name["source_chunked_vs_observed"]["passed"], "source adapter replay no longer reproduces the observed attention")
    _require(by_name["source_chunked_a_vs_a"]["passed"], "source adapter A/A differs")
    _require(by_name["f32_general_vs_observed_native"]["passed"], "F32 replay no longer reproduces the native attention")
    _require(by_name["sensitivity_a_vs_a"]["passed"] and not by_name["sensitivity_perturbation"]["passed"],
             "metric sensitivity failed")
    return dict(
        device="mps", torch_version=str(torch.__version__), macos_version=platform.mac_ver()[0],
        os_build=platform.version(), numeric_tolerance=dict(atol=ATOL, rtol=RTOL),
        geometry=GEOMETRY, source_exact_qkv=True, qkv_identity=inputs["qkv_identity"],
        source_query_chunks=[256, 7], source_mask=SOURCE_MASK,
        source_probabilities_scope=PROBABILITIES_SCOPE,
        explicit_stages_scope=EXPLICIT_SCOPE,
        terminal_cpu_exports=1, rows=rows, tensor_order=list(records), tensors=records,
        full_generation_executed=False, attention_production_changed=False,
    ), payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*ARGUMENTS, "upstream", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        output = shared.output_path(args.output)
        payload_path = output.with_suffix(".tensors.bin")
        _require(not payload_path.exists(), "packet payload already exists")
        inputs = validate_inputs(args)
        result, payload = _run_mps(inputs)
        document = dict(schema=SCHEMA, schema_version=1,
                        input_manifest_sha256={name: getattr(args, name.replace("-", "_")) for name in ARGUMENTS if name.endswith("sha256")},
                        upstream_revision=admission.PINNED_REVISION,
                        source_upstream_source_sha256=inputs["source_pins"],
                        tensor_artifact=dict(basename=payload_path.name, bytes=len(payload),
                                             sha256=hashlib.sha256(payload).hexdigest()), **result)
        with payload_path.open("xb") as stream:
            stream.write(payload)
        shared._publish_fresh(output, document)
    except (Error, shared.native_cache.ReferenceCacheError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(json.dumps(dict(schema=SCHEMA, status="rejected", reason=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(dict(schema=SCHEMA, output=str(output), bytes=len(payload),
                         source_adapter_reproduced=True, terminal_cpu_exports=1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
