#!/usr/bin/env python3
"""Trace Qwen3-VL text layer zero against the pinned BF16 fixture.

The model is loaded only from a local checkpoint. The trace compares the full
fixture sequence with its first two tokens under the checkpoint's default
attention implementation and eager attention, without changing repository
parity thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


HIDDEN_SIZE = 4096
EXPECTED_FIXTURE = {
    "schema": "qwen-image21-text-reference",
    "schema_version": 1,
    "model.repo": "Qwen/Qwen-Image-2.1",
    "model.revision_sha": "790c92633540aa0cb11d9abf19eb46d861714758",
    "prompt": "red cube",
    "payload_file": "qwen_image21_text_reference.bin",
    "payload_nbytes": 7_356_992,
    "payload_sha256": "3edcd7bf7964237d649a43c35cd82f6d6bd7b15835fddec2b1fe42f3a89b1e07",
}
EXPECTED_TENSORS = {
    "input_ids": (
        "int64-le", (1, 24), 0,
        "c1afceabeb24e6f5e44814caa83134046d433bd99a0e482a580cb8da6bc0a707",
    ),
    "attention_mask": (
        "int64-le", (1, 24), 192,
        "d335b3bc1461ed30f75513e7c17ed90d15c756ef02e8911aee0c66b1bddaf425",
    ),
    "mm_token_type_ids": (
        "int64-le", (1, 24), 384,
        "5d89f056865052bcb89c910d2d62872e029fb273c3db03f8968a52a41593c1b5",
    ),
    "hidden_state_000": (
        "bfloat16-le", (1, 24, HIDDEN_SIZE), 82_496,
        "eea2ebdeb9adc4db8a20e4ee846c683f56010a9a7f74a7dd722f6f8ede08f438",
    ),
    "hidden_state_001": (
        "bfloat16-le", (1, 24, HIDDEN_SIZE), 279_104,
        "c047cd3e0448a64397bc781fbf7fb3cf78dfd97753989f91efade6ce488ef71b",
    ),
    "pre_final_rmsnorm_embeddings": (
        "bfloat16-le", (1, 10, HIDDEN_SIZE), 576,
        "4de67b800ab96fb9d71b5eecf0b6c20c61758450bd98b1e8d97e9707fdf8b388",
    ),
}
TRACE_MODULES = (
    "embed_tokens",
    "layers.0.input_layernorm",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.q_norm",
    "layers.0.self_attn.k_norm",
    "layers.0.self_attn.o_proj",
    "layers.0.post_attention_layernorm",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
    "layers.0",
)
INTERMEDIATE_MODULES = (
    "embed_tokens",
    "layer0_input",
    "layers.0.input_layernorm",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.q_norm",
    "layers.0.self_attn.k_norm",
    "layers.0.self_attn.o_proj",
    "layers.0.post_attention_layernorm",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
    "layers.0",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes, *, overwrite: bool) -> None:
    with path.open("wb" if overwrite else "xb") as output_file:
        output_file.write(data)


def _raw_tensor_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().to(device="cpu").contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_sha256(tensor: Any) -> str:
    return _sha256(_raw_tensor_bytes(tensor))


def _snapshot(tensor: Any) -> Any:
    return tensor.detach().to(device="cpu").contiguous().clone()


def _bf16_bits(tensor: Any) -> list[int]:
    import torch

    value = tensor.detach().to(device="cpu").contiguous()
    if value.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 tensor for exact bit dump; got {value.dtype}")
    return value.view(torch.uint16).reshape(-1).tolist()


def _bf16_le_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().to(device="cpu").contiguous()
    if value.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 tensor for exact bit dump; got {value.dtype}")
    return value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()


def _bf16_values(tensor: Any) -> list[float]:
    return tensor.detach().to(device="cpu", dtype=__import__("torch").float32).reshape(-1).tolist()


def _metrics(actual: Any, expected: Any, *, query_axis: int | None = None) -> dict[str, Any]:
    import torch

    left = actual.detach().to(device="cpu").contiguous()
    right = expected.detach().to(device="cpu").contiguous()
    if left.shape != right.shape:
        return {
            "comparable": False,
            "actual_shape": list(left.shape),
            "expected_shape": list(right.shape),
        }

    if torch.is_floating_point(left) and not torch.isfinite(left).all().item():
        raise ValueError("actual tensor contains non-finite values")
    if torch.is_floating_point(right) and not torch.isfinite(right).all().item():
        raise ValueError("expected tensor contains non-finite values")

    if left.dtype == torch.bfloat16 and right.dtype == torch.bfloat16:
        mismatch = left.view(torch.uint16) != right.view(torch.uint16)
    else:
        mismatch = left != right
    error = (left.to(torch.float64) - right.to(torch.float64)).abs()
    squared_error = error.square().sum().item()
    squared_expected = right.to(torch.float64).square().sum().item()
    result: dict[str, Any] = {
        "comparable": True,
        "shape": list(left.shape),
        "actual_dtype": str(left.dtype),
        "expected_dtype": str(right.dtype),
        "exact_mismatches": int(mismatch.sum().item()),
        "elements": int(left.numel()),
        "max_abs_error": float(error.max().item()) if error.numel() else 0.0,
        "rmse": float((squared_error / max(1, left.numel())) ** 0.5),
        "relative_rms": float((squared_error / (squared_expected + 1e-30)) ** 0.5),
        "actual_sha256": _tensor_sha256(left),
        "expected_sha256": _tensor_sha256(right),
    }
    if query_axis is not None and left.ndim > query_axis and left.shape[query_axis] >= 2:
        by_query = []
        for query_index in range(2):
            selection = [slice(None)] * left.ndim
            selection[query_axis] = query_index
            selected_mismatch = mismatch[tuple(selection)]
            selected_error = error[tuple(selection)]
            by_query.append(
                {
                    "query_index": query_index,
                    "exact_mismatches": int(selected_mismatch.sum().item()),
                    "elements": int(selected_mismatch.numel()),
                    "max_abs_error": float(selected_error.max().item()) if selected_error.numel() else 0.0,
                }
            )
        result["by_query"] = by_query
    return result


def _load_fixture(fixture_dir: Path, torch: Any) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    fixture_root = fixture_dir.resolve(strict=True)
    if not fixture_root.is_dir():
        raise ValueError(f"fixture path is not a directory: {fixture_root}")
    manifest_path = fixture_root / "qwen_image21_text_reference.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def expect_manifest_value(path: tuple[str, ...], expected: Any) -> None:
        value: Any = manifest
        for field in path:
            if not isinstance(value, dict) or field not in value:
                raise ValueError(f"fixture manifest is missing {'.'.join(path)}")
            value = value[field]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"fixture manifest {'.'.join(path)} does not match the pinned fixture")

    for field, expected in EXPECTED_FIXTURE.items():
        path = tuple(field.split("."))
        expect_manifest_value(path, expected)
    for field, expected in (
        (("sequence", "raw_input_shape"), [1, 24]),
        (("sequence", "actual_sequence_length"), 10),
        (("sequence", "drop_idx"), 14),
    ):
        expect_manifest_value(field, expected)

    payload_name = manifest["payload_file"]
    relative_payload = Path(payload_name)
    if relative_payload.is_absolute() or len(relative_payload.parts) != 1 or relative_payload.name in ("", ".", ".."):
        raise ValueError("fixture payload path must be a file directly inside the fixture directory")
    payload_path = (fixture_root / relative_payload).resolve(strict=True)
    try:
        payload_path.relative_to(fixture_root)
    except ValueError as exc:
        raise ValueError("fixture payload resolves outside the fixture directory") from exc
    payload = payload_path.read_bytes()
    if len(payload) != EXPECTED_FIXTURE["payload_nbytes"]:
        raise ValueError("fixture payload byte count does not match manifest")
    if _sha256(payload) != EXPECTED_FIXTURE["payload_sha256"]:
        raise ValueError("fixture payload SHA-256 does not match manifest")

    dtype_info = {
        "int64-le": (torch.int64, 8),
        "bfloat16-le": (torch.bfloat16, 2),
        "float32-le": (torch.float32, 4),
    }
    if sys.byteorder != "little":
        raise RuntimeError("pinned fixture tensors are little-endian; trace loader requires a little-endian host")
    tensor_descriptors = manifest.get("tensors")
    if not isinstance(tensor_descriptors, dict):
        raise ValueError("fixture manifest tensors must be an object")
    occupied_ranges: list[tuple[int, int, str]] = []

    def read_tensor(name: str) -> Any:
        descriptor = tensor_descriptors.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"fixture manifest is missing tensor descriptor {name}")
        expected_dtype, expected_shape, expected_offset, expected_sha256 = EXPECTED_TENSORS[name]
        dtype_name = descriptor.get("dtype")
        shape = descriptor.get("shape")
        if dtype_name != expected_dtype or not isinstance(shape, list):
            raise ValueError(f"fixture tensor {name} has an unexpected dtype or shape")
        if any(type(dimension) is not int for dimension in shape) or tuple(shape) != expected_shape:
            raise ValueError(f"fixture tensor {name} has an unexpected dtype or shape")
        dtype, itemsize = dtype_info[expected_dtype]
        element_count = 1
        for dimension in expected_shape:
            element_count *= dimension
        expected_nbytes = element_count * itemsize
        start = descriptor.get("offset_bytes")
        nbytes = descriptor.get("nbytes")
        if (
            type(start) is not int
            or start != expected_offset
            or type(nbytes) is not int
            or nbytes != expected_nbytes
        ):
            raise ValueError(f"fixture tensor {name} has an invalid byte range")
        end = start + nbytes
        if end > len(payload):
            raise ValueError(f"fixture tensor {name} extends beyond the payload")
        data = payload[start:end]
        if (
            len(data) != nbytes
            or descriptor.get("sha256") != expected_sha256
            or _sha256(data) != expected_sha256
        ):
            raise ValueError(f"fixture tensor {name} failed its length or SHA-256 check")
        occupied_ranges.append((start, end, name))
        value = torch.frombuffer(bytearray(data), dtype=dtype)
        return value.reshape(expected_shape).clone()

    tensors = {
        "input_ids": read_tensor("input_ids"),
        "attention_mask": read_tensor("attention_mask"),
        "mm_token_type_ids": read_tensor("mm_token_type_ids"),
        "hidden_state_000": read_tensor("hidden_state_000"),
        "hidden_state_001": read_tensor("hidden_state_001"),
        "pre_final_rmsnorm_embeddings": read_tensor("pre_final_rmsnorm_embeddings"),
    }
    occupied_ranges.sort()
    for previous, current in zip(occupied_ranges, occupied_ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"fixture tensor byte ranges overlap: {previous[2]} and {current[2]}")
    return manifest, payload, tensors


def _config_implementations(model: Any, language_model: Any, attention: Any) -> dict[str, Any]:
    return {
        "conditional_generation_config": getattr(model.config, "_attn_implementation", None),
        "text_config": getattr(model.config.text_config, "_attn_implementation", None),
        "language_model_config": getattr(language_model.config, "_attn_implementation", None),
        "layer0_attention_config": getattr(attention.config, "_attn_implementation", None),
    }


def _set_attention_implementation(model: Any, language_model: Any, implementation: str) -> None:
    configs = [
        model.config,
        getattr(model.config, "text_config", None),
        language_model.config,
    ]
    for config in configs:
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = implementation


def _first_two(tensor: Any, *, sequence_axis: int = 1) -> Any:
    selection = [slice(None)] * tensor.ndim
    selection[sequence_axis] = slice(0, 2)
    return tensor[tuple(selection)]


def _run_trace(
    model: Any,
    tensors: dict[str, Any],
    *,
    token_count: int,
    drop_idx: int,
    device: str,
) -> dict[str, Any]:
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    language_model = model.model.language_model
    attention = language_model.layers[0].self_attn
    captured: dict[str, Any] = {}
    handles = []

    def tensor_hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                captured[name] = _snapshot(_first_two(output))
        return hook

    modules = dict(language_model.named_modules())
    for name in TRACE_MODULES:
        module = modules.get(name)
        if module is None:
            raise RuntimeError(f"expected model module is missing: {name}")
        handles.append(module.register_forward_hook(tensor_hook(name)))

    def capture_layer0_input(_module: Any, inputs: Any) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["layer0_input"] = _snapshot(_first_two(inputs[0]))

    handles.append(language_model.layers[0].register_forward_pre_hook(capture_layer0_input))

    def capture_attention_call(_module: Any, args: Any, kwargs: dict[str, Any]) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        position_embeddings = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
        mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
        if hidden is not None:
            captured["attention_input"] = _snapshot(_first_two(hidden))
        if position_embeddings is not None:
            captured["position_cos"] = _snapshot(_first_two(position_embeddings[0], sequence_axis=1))
            captured["position_sin"] = _snapshot(_first_two(position_embeddings[1], sequence_axis=1))
        if mask is not None:
            captured["attention_mask_2x2"] = _snapshot(mask[..., :2, :2])
        else:
            captured["attention_mask_2x2"] = None
        captured["attention_is_causal"] = bool(getattr(attention, "is_causal", False))

    handles.append(attention.register_forward_pre_hook(capture_attention_call, with_kwargs=True))

    def capture_attention_result(_module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, (tuple, list)):
            if output and isinstance(output[0], torch.Tensor):
                captured["self_attn_output"] = _snapshot(_first_two(output[0]))
            if len(output) > 1 and isinstance(output[1], torch.Tensor):
                captured["attention_probabilities"] = _snapshot(output[1][..., :2, :2])
            else:
                captured["attention_probabilities"] = None

    handles.append(attention.register_forward_hook(capture_attention_result))

    def capture_final_norm(_module: Any, inputs: Any, output: Any) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["final_norm_input_full"] = _snapshot(inputs[0])
        if isinstance(output, torch.Tensor):
            captured["final_norm_output_full"] = _snapshot(output)

    handles.append(language_model.norm.register_forward_hook(capture_final_norm))

    inputs: dict[str, Any] = {}
    for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
        value = tensors[name][:, :token_count].to(device=device)
        inputs[name] = value

    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) < 2:
        raise RuntimeError("the text encoder did not return the first decoder-layer hidden state")
    captured["encoder_hidden_state_000"] = _snapshot(_first_two(hidden_states[0]))
    captured["encoder_hidden_state_001"] = _snapshot(_first_two(hidden_states[1]))
    captured["encoder_last_hidden_state"] = _snapshot(_first_two(outputs.hidden_states[-1]))
    captured["encoder_hidden_state_count"] = len(hidden_states)
    if token_count > drop_idx and "final_norm_input_full" in captured:
        visible = tensors["attention_mask"][:, :token_count].to(device="cpu").bool()[0]
        selected = captured["final_norm_input_full"][0][visible][drop_idx:]
        captured["pre_final_prompt_embeddings"] = _snapshot(selected.unsqueeze(0))
        captured["final_norm_input_equals_output"] = bool(
            torch.equal(captured["final_norm_input_full"], captured["final_norm_output_full"])
        )

    # Reconstruct the first two rows of the exact eager attention scores from
    # the hooked BF16 projections and positional embeddings. The same Q/K/V
    # tensors feed both the SDPA and eager module calls.
    q_norm = captured["layers.0.self_attn.q_norm"]
    k_norm = captured["layers.0.self_attn.k_norm"]
    v_proj = captured["layers.0.self_attn.v_proj"]
    cos = captured["position_cos"]
    sin = captured["position_sin"]
    query = q_norm.transpose(1, 2)
    key = k_norm.transpose(1, 2)
    query, key = modeling_qwen3_vl.apply_rotary_pos_emb(query, key, cos, sin)
    key_states = modeling_qwen3_vl.repeat_kv(key, attention.num_key_value_groups)
    value_states = v_proj.reshape(v_proj.shape[0], v_proj.shape[1], -1, attention.head_dim).transpose(1, 2)
    value_states = modeling_qwen3_vl.repeat_kv(value_states, attention.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * attention.scaling
    mask_2x2 = captured["attention_mask_2x2"]
    if mask_2x2 is not None:
        scores = scores + mask_2x2
    elif captured["attention_is_causal"]:
        # SDPA commonly represents causality with is_causal=True and no
        # materialized mask; eager receives the equivalent additive mask.
        future_keys = torch.triu(
            torch.ones((2, 2), dtype=torch.bool, device=scores.device), diagonal=1
        )
        causal_mask = torch.zeros((1, 1, 2, 2), dtype=scores.dtype, device=scores.device)
        causal_mask.masked_fill_(future_keys, torch.finfo(scores.dtype).min)
        scores = scores + causal_mask
    probabilities = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    captured["reconstructed_attention_scores"] = _snapshot(scores)
    captured["reconstructed_attention_probabilities"] = _snapshot(probabilities)
    captured["reconstructed_attention_head_output"] = _snapshot(torch.matmul(probabilities, value_states))

    if captured["attention_probabilities"] is not None:
        captured["eager_probabilities_equal_reconstruction"] = bool(
            torch.equal(captured["attention_probabilities"], captured["reconstructed_attention_probabilities"])
        )
    else:
        captured["eager_probabilities_equal_reconstruction"] = None
    captured["layer0_hook_equals_encoder_hidden_state_001"] = bool(
        torch.equal(captured["layers.0"], captured["encoder_hidden_state_001"])
    )
    return captured


def _summarize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, tensor in snapshot.items():
        if not hasattr(tensor, "shape"):
            summary[name] = tensor
            continue
        entry: dict[str, Any] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": _tensor_sha256(tensor),
        }
        if name in ("reconstructed_attention_scores", "reconstructed_attention_probabilities"):
            entry["values"] = _bf16_values(tensor)
            entry["bf16_bits"] = _bf16_bits(tensor)
        summary[name] = entry
    return summary


def _compare_snapshots(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name in sorted(left.keys() & right.keys()):
        a, b = left[name], right[name]
        if hasattr(a, "shape") and hasattr(b, "shape"):
            query_axis = 2 if name in (
                "attention_probabilities",
                "reconstructed_attention_scores",
                "reconstructed_attention_probabilities",
                "reconstructed_attention_head_output",
            ) else 1
            comparisons[name] = _metrics(a, b, query_axis=query_axis)
        elif hasattr(a, "shape") or hasattr(b, "shape"):
            comparisons[name] = {
                "left": _summarize_snapshot({name: a})[name] if hasattr(a, "shape") else a,
                "right": _summarize_snapshot({name: b})[name] if hasattr(b, "shape") else b,
                "equal": False,
            }
        elif a != b:
            comparisons[name] = {"left": a, "right": b, "equal": False}
    return comparisons


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output or BF16 sidecar")
    parser.add_argument(
        "--dump-layer0-bf16",
        action="store_true",
        help="write each run's first-two-token layer-0 output as a compact BF16 little-endian sidecar",
    )
    parser.add_argument(
        "--prefix2-only",
        action="store_true",
        help="run only the default-attention two-token trace instead of the default four-run comparison",
    )
    parser.add_argument(
        "--dump-intermediates-bf16",
        action="store_true",
        help="write selected first-two-token BF16 module outputs from the default-attention run",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    if args.model_dir is None and os.environ.get("QWEN3VL_TEXT_ENCODER_DIR"):
        args.model_dir = Path(os.environ["QWEN3VL_TEXT_ENCODER_DIR"])
    if args.fixture_dir is None and os.environ.get("QWEN3VL_TEXT_REFERENCE_DIR"):
        args.fixture_dir = Path(os.environ["QWEN3VL_TEXT_REFERENCE_DIR"])
    if args.model_dir is None:
        parser.error("provide --model-dir or QWEN3VL_TEXT_ENCODER_DIR")
    if args.fixture_dir is None:
        parser.error("provide --fixture-dir or QWEN3VL_TEXT_REFERENCE_DIR")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists (pass --overwrite to replace it): {args.output}")
    return args


def main() -> int:
    args = _parse_args()
    import torch
    import transformers
    from transformers import Qwen3VLForConditionalGeneration

    manifest, payload, fixture = _load_fixture(args.fixture_dir, torch)
    if fixture["input_ids"].shape[0] != 1 or fixture["input_ids"].shape[1] < 2:
        raise ValueError("trace expects one fixture row with at least two input tokens")
    if fixture["hidden_state_001"].shape[-1] != HIDDEN_SIZE:
        raise ValueError("fixture hidden size does not match this diagnostic")
    if not torch.all(fixture["attention_mask"][:, :2] == 1):
        raise ValueError("fixture first two tokens must both be attended")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=args.device)
    model.eval()
    language_model = model.model.language_model
    attention = language_model.layers[0].self_attn
    original_implementations = _config_implementations(model, language_model, attention)
    default_implementation = original_implementations["layer0_attention_config"]
    if not isinstance(default_implementation, str):
        raise RuntimeError(f"could not read actual layer0 _attn_implementation: {original_implementations}")

    fixture_layer0 = fixture["hidden_state_001"][:, :2].to(device="cpu")
    run_records: dict[str, dict[str, Any]] = {}
    run_tensors: dict[str, dict[str, Any]] = {}
    if args.prefix2_only:
        run_plan = [("default_prefix2", default_implementation, 2)]
    else:
        run_plan = [
            ("default_full", default_implementation, fixture["input_ids"].shape[1]),
            ("default_prefix2", default_implementation, 2),
        ]
        if default_implementation != "eager":
            run_plan.extend(
                [
                    ("eager_full", "eager", fixture["input_ids"].shape[1]),
                    ("eager_prefix2", "eager", 2),
                ]
            )

    planned_sidecars: list[Path] = []
    if args.dump_intermediates_bf16:
        run_name = "default_prefix2" if args.prefix2_only else "default_full"
        planned_sidecars.append(
            args.output.with_name(f"{args.output.stem}.{run_name}.intermediates_first2.bf16le")
        )
    if args.dump_layer0_bf16:
        planned_sidecars.extend(
            args.output.with_name(f"{args.output.stem}.{name}.layer0_first2.bf16le")
            for name, _, _ in run_plan
        )
    if not args.overwrite:
        existing_sidecars = [sidecar for sidecar in planned_sidecars if sidecar.exists()]
        if existing_sidecars:
            raise FileExistsError(
                "BF16 sidecar already exists (pass --overwrite to replace it): "
                + ", ".join(str(sidecar) for sidecar in existing_sidecars)
            )

    for name, implementation, token_count in run_plan:
        _set_attention_implementation(model, language_model, implementation)
        actual_implementations = _config_implementations(model, language_model, attention)
        if actual_implementations["layer0_attention_config"] != implementation:
            raise RuntimeError(
                f"failed to select {implementation!r} for layer0 attention: {actual_implementations}"
            )
        print(f"running {name}: tokens={token_count}, implementations={actual_implementations}", flush=True)
        tensors = _run_trace(
            model,
            fixture,
            token_count=token_count,
            drop_idx=int(manifest["sequence"]["drop_idx"]),
            device=args.device,
        )
        run_tensors[name] = tensors
        run_records[name] = {
            "requested_implementation": implementation,
            "actual_implementations": actual_implementations,
            "input_token_count": token_count,
            "first_two_input_ids": fixture["input_ids"][0, :2].tolist(),
            "first_two_attention_mask": fixture["attention_mask"][0, :2].tolist(),
            "first_two_mm_token_type_ids": fixture["mm_token_type_ids"][0, :2].tolist(),
            "fixture_layer0_metrics": _metrics(tensors["layers.0"], fixture_layer0, query_axis=1),
            "fixture_hidden_state_001_metrics": _metrics(
                tensors["encoder_hidden_state_001"], fixture_layer0, query_axis=1
            ),
            "pre_final_prompt_metrics": (
                _metrics(
                    tensors["pre_final_prompt_embeddings"],
                    fixture["pre_final_rmsnorm_embeddings"],
                    query_axis=1,
                )
                if "pre_final_prompt_embeddings" in tensors
                else None
            ),
            "final_norm_input_equals_output": tensors.get("final_norm_input_equals_output"),
            "layer0_hook_equals_encoder_hidden_state_001": tensors[
                "layer0_hook_equals_encoder_hidden_state_001"
            ],
            "eager_probabilities_equal_reconstruction": tensors[
                "eager_probabilities_equal_reconstruction"
            ],
            "trace": _summarize_snapshot(tensors),
        }

    comparisons: dict[str, Any] = {}
    for left, right in (
        ("default_full", "default_prefix2"),
        ("eager_full", "eager_prefix2"),
        ("default_full", "eager_full"),
        ("default_prefix2", "eager_prefix2"),
    ):
        if left in run_tensors and right in run_tensors:
            comparisons[f"{left}_vs_{right}"] = _compare_snapshots(run_tensors[left], run_tensors[right])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_sidecar: dict[str, Any] | None = None
    if args.dump_intermediates_bf16:
        run_name = "default_prefix2" if args.prefix2_only else "default_full"
        snapshot = run_tensors[run_name]
        sidecar = args.output.with_name(f"{args.output.stem}.{run_name}.intermediates_first2.bf16le")
        packed = bytearray()
        tensor_manifest: dict[str, Any] = {}
        for module_name in INTERMEDIATE_MODULES:
            tensor = snapshot.get(module_name)
            if tensor is None:
                raise RuntimeError(f"selected run {run_name} did not capture required module output {module_name!r}")
            chunk = _bf16_le_bytes(tensor)
            tensor_manifest[module_name] = {
                "dtype": "bfloat16-le",
                "shape": list(tensor.shape),
                "offset_bytes": len(packed),
                "nbytes": len(chunk),
                "sha256": _sha256(chunk),
            }
            packed.extend(chunk)
        if sidecar.exists() and not args.overwrite:
            raise FileExistsError(f"BF16 intermediate sidecar already exists: {sidecar}")
        _write_bytes(sidecar, packed, overwrite=args.overwrite)
        intermediate_sidecar = {
            "run": run_name,
            "requested_implementation": run_records[run_name]["requested_implementation"],
            "actual_implementations": run_records[run_name]["actual_implementations"],
            "path": str(sidecar),
            "dtype": "concatenated-bfloat16-le",
            "nbytes": len(packed),
            "sha256": _sha256(bytes(packed)),
            "tensors": tensor_manifest,
        }
    if args.dump_layer0_bf16:
        for name, snapshot in run_tensors.items():
            tensor = snapshot["layers.0"]
            sidecar = args.output.with_name(f"{args.output.stem}.{name}.layer0_first2.bf16le")
            if sidecar.exists() and not args.overwrite:
                raise FileExistsError(f"BF16 sidecar already exists: {sidecar}")
            _write_bytes(sidecar, _bf16_le_bytes(tensor), overwrite=args.overwrite)
            run_records[name]["layer0_first_two_bf16_sidecar"] = {
                "path": str(sidecar),
                "dtype": "bfloat16-le",
                "shape": list(tensor.shape),
                "nbytes": sidecar.stat().st_size,
                "sha256": _sha256(sidecar.read_bytes()),
            }

    result = {
        "schema": "qwen3vl-text-layer-trace",
        "schema_version": 1,
        "model_dir": str(args.model_dir),
        "fixture_dir": str(args.fixture_dir),
        "fixture": {
            "repo": manifest["model"].get("repo"),
            "revision_sha": manifest["model"].get("revision_sha"),
            "payload_sha256": _sha256(payload),
            "raw_input_shape": manifest["sequence"]["raw_input_shape"],
            "fixture_hidden_state_001_shape": manifest["tensors"]["hidden_state_001"]["shape"],
            "fixture_hidden_state_001_sha256": manifest["tensors"]["hidden_state_001"]["sha256"],
            "hidden_state_001_first_two_sha256": _tensor_sha256(fixture_layer0),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": args.device,
            "model_dtype": str(next(model.parameters()).dtype),
            "original_implementations": original_implementations,
            "default_layer0_attention_implementation": default_implementation,
        },
        "runs": run_records,
        "comparisons": comparisons,
        "intermediate_sidecar": intermediate_sidecar,
        "interpretation": {
            "hidden_state_001_mapping": (
                "For this Transformers output, hidden_states[1] is checked against the direct layer-0 module hook; "
                "the fixture comparison separately checks whether its hidden_state_001 values match."
            ),
            "attention_scores": (
                "The trace stores the first-two-token BF16 score matrix reconstructed from the hooked BF16 Q/K "
                "projections, Qwen rotary embeddings, and causal mask (including the is_causal=True/no-mask SDPA form) "
                "using Transformers eager score operations. "
                "For eager runs it also checks the returned attention probabilities against the reconstruction; "
                "SDPA itself does not return its internal probabilities."
            ),
        },
    }
    result_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as output_file:
        output_file.write(result_json)
    print(f"wrote {args.output}", flush=True)
    for name, record in run_records.items():
        metrics = record["fixture_layer0_metrics"]
        print(
            f"{name}: fixture_mismatches={metrics['exact_mismatches']}/{metrics['elements']} "
            f"first-token={metrics['by_query'][0]['exact_mismatches']} "
            f"second-token={metrics['by_query'][1]['exact_mismatches']} "
            f"max_abs={metrics['max_abs_error']:.9g} rmse={metrics['rmse']:.9g} "
            f"relative_rms={metrics['relative_rms']:.9g} "
            f"hook_is_hidden_state_001={record['layer0_hook_equals_encoder_hidden_state_001']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # Keep diagnostic failure visible and attributable.
        print(f"qwen3vl_text_layer_trace: error: {exc}", file=sys.stderr)
        raise
