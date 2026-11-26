#!/usr/bin/env python3
"""Capture a text-only Qwen-Image 2.1 encoder reference without denoising.

The generator invokes the pinned Diffusers ``QwenImage21Pipeline`` prompt
encoder with its local processor and text encoder. It writes the raw processor
inputs, official prefix drop index, returned pre-final-RMSNorm embeddings, and
all returned hidden states into a versioned little-endian payload.

The maximum sequence length is a post-tokenization guard only. The official
processor call is left-padded and untruncated, as in the pinned pipeline.
Model files are loaded with ``local_files_only=True``; there is no network
fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "qwen-image21-text-reference"
SCHEMA_VERSION = 1
MODEL_REPO = "Qwen/Qwen-Image-2.1"
PIPELINE_CLASS = "QwenImage21Pipeline"
TEXT_ENCODER_CLASS = "Qwen3VLForConditionalGeneration"
PROCESSOR_CLASS = "Qwen3VLProcessor"
CONTEXT_DIM = 4096
PINNED_DIFFUSERS_COMMIT = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
MANIFEST_NAME = "qwen_image21_text_reference.json"
PAYLOAD_NAME = "qwen_image21_text_reference.bin"
REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SOURCE_DTYPES = {"bfloat16", "float16", "float32"}


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _expected_hidden_state_count(text_encoder: Any) -> int:
    config = getattr(text_encoder, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        text_config = config
    decoder_layer_count = getattr(text_config, "num_hidden_layers", None)
    if not _is_positive_int(decoder_layer_count):
        raise RuntimeError(
            "cannot derive expected hidden-state count from loaded text_encoder.config"
        )
    # Transformers returns the embedding output followed by one state for each
    # decoder layer when output_hidden_states=True.
    return decoder_layer_count + 1


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("tensor value must have a shape")
    return tuple(int(dimension) for dimension in shape)


def _captured_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return "unknown"
    return str(dtype).removeprefix("torch.")


def _cpu_copy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().to(device="cpu").clone()
    import numpy as np

    return np.array(value, copy=True)


def _to_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu")
        if _captured_dtype(value) == "bfloat16":
            raise ValueError("bfloat16 is not valid for token IDs or masks")
        value = value.numpy()
    return np.asarray(value)


def _values_equal(left: Any, right: Any) -> bool:
    if hasattr(left, "detach") or hasattr(right, "detach"):
        import torch

        if not hasattr(left, "detach"):
            left = torch.as_tensor(left)
        if not hasattr(right, "detach"):
            right = torch.as_tensor(right)
        return bool(torch.equal(left.detach().to(device="cpu"), right.detach().to(device="cpu")))
    import numpy as np

    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _masked_after_drop(hidden_state: Any, attention_mask: Any, drop_idx: int) -> Any:
    if hasattr(hidden_state, "detach"):
        import torch

        mask = attention_mask
        if not hasattr(mask, "detach"):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=hidden_state.device).bool()
        selected = hidden_state[mask]
        return selected[drop_idx:].unsqueeze(0)

    import numpy as np

    hidden = np.asarray(hidden_state)
    mask = np.asarray(attention_mask).astype(bool)
    return hidden[mask][drop_idx:][None, ...]


def _validated_arrays(
    *,
    input_ids: Any,
    attention_mask: Any,
    mm_token_type_ids: Any | None,
    drop_idx: int,
    max_sequence_length: int,
    expected_hidden_state_count: int,
    pre_final_rmsnorm_embeddings: Any,
    layer_hidden_states: list[Any] | tuple[Any, ...],
) -> int:
    import numpy as np

    ids_shape = _shape(input_ids)
    mask_shape = _shape(attention_mask)
    if len(ids_shape) != 2 or ids_shape[0] != 1 or ids_shape[1] == 0:
        raise ValueError("input_ids must have batch-one shape [1, raw_sequence_length]")
    if mask_shape != ids_shape:
        raise ValueError(f"attention_mask shape must match input_ids shape {ids_shape}; got {mask_shape}")
    ids = _to_numpy(input_ids)
    mask = _to_numpy(attention_mask)
    if ids.dtype.kind not in "iu":
        raise ValueError("input_ids must contain integer token IDs")
    if mask.dtype.kind not in "iub" or not bool(np.logical_or(mask == 0, mask == 1).all()):
        raise ValueError("attention_mask must contain only integer or boolean 0/1 values")
    if mm_token_type_ids is not None:
        if _shape(mm_token_type_ids) != ids_shape:
            raise ValueError("mm_token_type_ids shape must match input_ids shape")
        if _to_numpy(mm_token_type_ids).dtype.kind not in "iu":
            raise ValueError("mm_token_type_ids must contain integer values")

    if isinstance(drop_idx, bool) or not isinstance(drop_idx, int) or drop_idx < 0:
        raise ValueError("drop_idx must be a non-negative integer")
    if not _is_positive_int(max_sequence_length):
        raise ValueError("max_sequence_length must be a positive integer")
    valid_token_count = int(mask.astype(bool).sum())
    if drop_idx >= valid_token_count:
        raise ValueError("drop_idx must be smaller than the attended input token count")
    actual_sequence_length = valid_token_count - drop_idx
    if actual_sequence_length > max_sequence_length:
        raise ValueError(
            "post-drop sequence length "
            f"{actual_sequence_length} exceeds max_sequence_length guard {max_sequence_length}"
        )

    embed_shape = _shape(pre_final_rmsnorm_embeddings)
    expected_embedding_shape = (1, actual_sequence_length, CONTEXT_DIM)
    if embed_shape != expected_embedding_shape:
        raise ValueError(
            "pre_final_rmsnorm_embeddings must have shape "
            f"{expected_embedding_shape}; got {embed_shape}"
        )
    if not layer_hidden_states:
        raise ValueError("layer_hidden_states must include the embedding state and decoder layer states")
    if not _is_positive_int(expected_hidden_state_count):
        raise ValueError("expected_hidden_state_count must be a positive integer")
    if len(layer_hidden_states) != expected_hidden_state_count:
        raise ValueError(
            f"expected {expected_hidden_state_count} hidden states from the loaded text config; "
            f"got {len(layer_hidden_states)}"
        )
    expected_layer_shape = (1, ids_shape[1], CONTEXT_DIM)
    for index, hidden_state in enumerate(layer_hidden_states):
        if _shape(hidden_state) != expected_layer_shape:
            raise ValueError(
                f"layer_hidden_states[{index}] must have shape {expected_layer_shape}; "
                f"got {_shape(hidden_state)}"
            )

    expected_embeddings = _masked_after_drop(layer_hidden_states[-1], attention_mask, drop_idx)
    if not _values_equal(pre_final_rmsnorm_embeddings, expected_embeddings):
        raise ValueError(
            "returned prompt embeddings do not equal the attended final hidden state after drop_idx"
        )
    return actual_sequence_length


class _RecordingProcessor:
    """Transparent processor wrapper that records the official pipeline call."""

    def __init__(self, processor: Any):
        self._processor = processor
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._processor(*args, **kwargs)
        texts = kwargs.get("text")
        if isinstance(texts, str):
            texts = [texts]
        text_ids = getattr(result, "input_ids", None)
        attention_mask = getattr(result, "attention_mask", None)
        if text_ids is None or attention_mask is None:
            raise RuntimeError("official processor output lacks input_ids or attention_mask")
        mm_token_type_ids = getattr(result, "mm_token_type_ids", None)
        self.calls.append(
            {
                "texts": list(texts) if texts is not None else None,
                "kwargs": {key: value for key, value in kwargs.items() if key != "text"},
                "input_ids": _cpu_copy(text_ids),
                "attention_mask": _cpu_copy(attention_mask),
                "mm_token_type_ids": (
                    _cpu_copy(mm_token_type_ids) if mm_token_type_ids is not None else None
                ),
            }
        )
        return result


def capture_pipeline_reference(
    pipeline: Any,
    prompt: str,
    *,
    max_sequence_length: int,
    device: Any,
    source_dtype: str,
) -> dict[str, Any]:
    """Capture one text-only call through ``QwenImage21Pipeline`` itself.

    The encoder and RMSNorm hooks observe the official forward call and are
    always removed, including when processing or validation raises.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if source_dtype not in SOURCE_DTYPES:
        raise ValueError("source_dtype must be bfloat16, float16, or float32")
    if not _is_positive_int(max_sequence_length):
        raise ValueError("max_sequence_length must be a positive integer")

    original_processor = pipeline.processor
    recording_processor = _RecordingProcessor(original_processor)
    processor_replaced = False
    encoder_handle = None
    norm_handle = None
    captured_encoder_outputs: list[Any] = []
    norm_observations: list[tuple[Any, Any]] = []
    expected_hidden_state_count = _expected_hidden_state_count(pipeline.text_encoder)

    text_model = getattr(pipeline.text_encoder.model, "language_model", pipeline.text_encoder.model)
    norm_module = getattr(text_model, "norm", None)
    if norm_module is None or not hasattr(norm_module, "register_forward_hook"):
        raise RuntimeError("Qwen3-VL text model does not expose its final RMSNorm module")

    def capture_encoder_output(_module: Any, _args: Any, output: Any) -> None:
        captured_encoder_outputs.append(output)

    def observe_final_norm(_module: Any, args: Any, output: Any) -> None:
        if args:
            norm_observations.append((args[0], output))

    try:
        pipeline.processor = recording_processor
        processor_replaced = True
        # These hooks are registered before the pipeline's own RMSNorm hook.
        # The norm observer records both the pre-norm input and the ordinary
        # normalized output; the encoder hook sees the final hooked result.
        norm_handle = norm_module.register_forward_hook(observe_final_norm)
        encoder_handle = pipeline.text_encoder.register_forward_hook(capture_encoder_output)
        outputs = pipeline._get_qwen_prompt_embeds(
            prompt=prompt,
            image=None,
            device=device,
        )
    finally:
        if encoder_handle is not None:
            encoder_handle.remove()
        if norm_handle is not None:
            norm_handle.remove()
        if processor_replaced:
            pipeline.processor = original_processor

    if len(recording_processor.calls) != 1:
        raise RuntimeError(
            f"expected one official processor call; observed {len(recording_processor.calls)}"
        )
    call = recording_processor.calls[0]
    expected_processor_kwargs = {"padding", "padding_side", "return_tensors"}
    if set(call["kwargs"]) != expected_processor_kwargs:
        raise RuntimeError(
            "pinned text-only processor arguments changed: "
            f"expected {sorted(expected_processor_kwargs)}, got {sorted(call['kwargs'])}"
        )
    if call["kwargs"] != {"padding": True, "padding_side": "left", "return_tensors": "pt"}:
        raise RuntimeError(f"pinned text-only processor settings changed: {call['kwargs']}")
    if call["texts"] is None or len(call["texts"]) != 1:
        raise RuntimeError("official text-only processor call must contain one raw template string")
    if len(captured_encoder_outputs) != 1:
        raise RuntimeError(
            f"expected one text-encoder forward; observed {len(captured_encoder_outputs)}"
        )
    if len(norm_observations) != 1:
        raise RuntimeError(f"expected one final RMSNorm call; observed {len(norm_observations)}")
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("official prompt encoder must return embeddings, attention mask, and image-pad mask")

    output_hidden_states = getattr(captured_encoder_outputs[0], "hidden_states", None)
    if not output_hidden_states:
        raise RuntimeError("Qwen3-VL forward did not return per-layer hidden states")
    hidden_states = tuple(output_hidden_states)
    if len(hidden_states) != expected_hidden_state_count:
        raise RuntimeError(
            f"expected {expected_hidden_state_count} hidden states from the loaded text config; "
            f"got {len(hidden_states)}"
        )
    norm_input, norm_output = norm_observations[0]
    if not _values_equal(hidden_states[-1], norm_input):
        raise RuntimeError(
            "final hidden state is not the final RMSNorm input; pre-final-RMSNorm capture is not established"
        )
    if _values_equal(norm_input, norm_output):
        raise RuntimeError(
            "final RMSNorm output matched its input, so the pre-final-RMSNorm guard was not discriminating"
        )

    prompt_embeddings, _returned_mask, _image_pad_mask = outputs
    attention_mask = call["attention_mask"]
    drop_idx = pipeline._drop_idx
    expected_embeddings = _masked_after_drop(hidden_states[-1], attention_mask, drop_idx)
    if not _values_equal(prompt_embeddings, expected_embeddings):
        raise RuntimeError(
            "official returned embeddings differ from the attended final hidden state after drop_idx"
        )
    actual_sequence_length = int(_to_numpy(attention_mask).astype(bool).sum()) - int(drop_idx)
    if actual_sequence_length > max_sequence_length:
        raise ValueError(
            "post-drop sequence length "
            f"{actual_sequence_length} exceeds max_sequence_length guard {max_sequence_length}"
        )

    return {
        "prompt": prompt,
        "raw_template_text": call["texts"][0],
        "processor_kwargs": call["kwargs"],
        "input_ids": call["input_ids"],
        "attention_mask": attention_mask,
        "mm_token_type_ids": call["mm_token_type_ids"],
        "drop_idx": int(drop_idx),
        "expected_hidden_state_count": expected_hidden_state_count,
        "max_sequence_length": max_sequence_length,
        "source_dtype": source_dtype,
        "pre_final_rmsnorm_embeddings": prompt_embeddings,
        "layer_hidden_states": hidden_states,
    }


def _check_revision(revision: str) -> str:
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise ValueError("model revision must be a full 40-character Hugging Face commit SHA")
    return revision.lower()


def _read_cached_revision(model_dir: Path) -> str | None:
    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    revisions: set[str] = set()
    if cache_dir.is_dir():
        for metadata in cache_dir.rglob("*.metadata"):
            try:
                lines = metadata.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                raise ValueError(f"cannot read model cache metadata {metadata}: {exc}") from exc
            if lines and REVISION_RE.fullmatch(lines[0]):
                revisions.add(lines[0].lower())
    if len(revisions) > 1:
        raise ValueError(f"local model files come from different revisions: {sorted(revisions)}")
    return next(iter(revisions), None)


def _resolve_revision(model_dir: Path, requested_revision: str | None) -> tuple[str, str]:
    cached_revision = _read_cached_revision(model_dir)
    if requested_revision is None:
        if cached_revision is None:
            raise ValueError(
                "could not determine local model revision; pass the full --revision commit SHA"
            )
        return cached_revision, "local_cache_metadata"
    revision = _check_revision(requested_revision)
    if cached_revision is not None and cached_revision != revision:
        raise ValueError(
            f"requested revision {revision} differs from local model cache revision {cached_revision}"
        )
    return revision, "argument" if cached_revision is None else "argument_and_local_cache_metadata"


def _validate_model_directory(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise ValueError(f"local model directory does not exist: {model_dir}")
    index_path = model_dir / "model_index.json"
    config_path = model_dir / "text_encoder" / "config.json"
    processor_dir = model_dir / "processor"
    if not index_path.is_file() or not config_path.is_file() or not processor_dir.is_dir():
        raise ValueError("expected local model_index.json, text_encoder/config.json, and processor/")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read local Qwen-Image 2.1 config: {exc}") from exc
    if index.get("_class_name") != PIPELINE_CLASS:
        raise ValueError(f"model_index.json must select {PIPELINE_CLASS}")
    if index.get("processor") != ["transformers", PROCESSOR_CLASS]:
        raise ValueError(f"model_index.json must select {PROCESSOR_CLASS}")
    if index.get("text_encoder") != ["transformers", TEXT_ENCODER_CLASS]:
        raise ValueError(f"model_index.json must select {TEXT_ENCODER_CLASS}")
    if config.get("architectures") != [TEXT_ENCODER_CLASS]:
        raise ValueError(f"text_encoder/config.json must describe {TEXT_ENCODER_CLASS}")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("text_encoder/config.json must describe Qwen3-VL")
    if config.get("text_config", {}).get("hidden_size") != CONTEXT_DIM:
        raise ValueError(f"Qwen3-VL text hidden_size must be {CONTEXT_DIM}")


def _installed_diffusers_commit() -> str | None:
    try:
        direct_url = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
        if not direct_url:
            return None
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if isinstance(commit, str) and REVISION_RE.fullmatch(commit):
            return commit.lower()
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        return None
    return None


def _serialize_tensor(name: str, value: Any) -> tuple[bytes, str, list[int], str]:
    import numpy as np

    shape = list(_shape(value))
    source_dtype = _captured_dtype(value)
    is_integer_tensor = name in {"input_ids", "attention_mask", "mm_token_type_ids"}
    if is_integer_tensor:
        array = _to_numpy(value)
        if array.dtype.kind not in "iub":
            raise ValueError(f"{name} must contain integer values")
        if name == "attention_mask" and not bool(
            np.logical_or(array == 0, array == 1).all()
        ):
            raise ValueError("attention_mask must contain only 0/1 values")
        if array.dtype.kind == "u" and array.size and int(array.max()) > (2**63 - 1):
            raise ValueError(f"{name} contains a value outside the int64 payload range")
        payload = np.ascontiguousarray(array, dtype="<i8").tobytes(order="C")
        return payload, "int64-le", shape, source_dtype

    if source_dtype == "bfloat16":
        if not hasattr(value, "detach"):
            raise ValueError(f"{name}: NumPy has no native bfloat16 dtype")
        import torch

        bits = value.detach().to(device="cpu").contiguous().view(torch.uint16).numpy()
        payload = np.ascontiguousarray(bits, dtype="<u2").tobytes(order="C")
        return payload, "bfloat16-le", shape, source_dtype

    if source_dtype not in {"float16", "float32"}:
        raise ValueError(f"{name} must have float16, bfloat16, or float32 dtype; got {source_dtype}")
    array = _to_numpy(value)
    if array.dtype.kind != "f":
        raise ValueError(f"{name} must contain floating-point values")
    payload_dtype = "<f2" if source_dtype == "float16" else "<f4"
    payload = np.ascontiguousarray(array, dtype=payload_dtype).tobytes(order="C")
    return payload, f"{source_dtype}-le", shape, source_dtype


def write_reference_bundle(
    *,
    output_dir: str | Path,
    prompt: str,
    model_revision: str,
    raw_template_text: str,
    max_sequence_length: int,
    source_dtype: str,
    device: str,
    runtime: dict[str, Any],
    input_ids: Any,
    attention_mask: Any,
    drop_idx: int,
    pre_final_rmsnorm_embeddings: Any,
    layer_hidden_states: list[Any] | tuple[Any, ...],
    expected_hidden_state_count: int,
    mm_token_type_ids: Any | None = None,
    processor_kwargs: dict[str, Any] | None = None,
    model_revision_source: str = "local_cache_metadata",
) -> Path:
    """Validate and write the raw text encoder reference payload and manifest."""
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not isinstance(raw_template_text, str):
        raise ValueError("raw_template_text must be a string")
    revision = _check_revision(model_revision)
    if source_dtype not in SOURCE_DTYPES:
        raise ValueError("source_dtype must be bfloat16, float16, or float32")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a metadata dictionary")
    if runtime.get("diffusers_commit") != PINNED_DIFFUSERS_COMMIT:
        raise ValueError(
            "runtime.diffusers_commit must match pinned official source "
            f"{PINNED_DIFFUSERS_COMMIT}"
        )
    if not isinstance(runtime.get("diffusers_version"), str):
        raise ValueError("runtime.diffusers_version is required")
    actual_sequence_length = _validated_arrays(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        drop_idx=drop_idx,
        max_sequence_length=max_sequence_length,
        expected_hidden_state_count=expected_hidden_state_count,
        pre_final_rmsnorm_embeddings=pre_final_rmsnorm_embeddings,
        layer_hidden_states=layer_hidden_states,
    )

    tensor_values: list[tuple[str, Any]] = [
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
    ]
    if mm_token_type_ids is not None:
        tensor_values.append(("mm_token_type_ids", mm_token_type_ids))
    tensor_values.append(("pre_final_rmsnorm_embeddings", pre_final_rmsnorm_embeddings))
    tensor_values.extend(
        (f"hidden_state_{index:03d}", hidden_state)
        for index, hidden_state in enumerate(layer_hidden_states)
    )

    payload = bytearray()
    tensors: dict[str, dict[str, Any]] = {}
    for name, value in tensor_values:
        data, dtype, shape, captured_dtype = _serialize_tensor(name, value)
        tensors[name] = {
            "dtype": dtype,
            "captured_dtype": captured_dtype,
            "shape": shape,
            "offset_bytes": len(payload),
            "nbytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        payload.extend(data)

    ids_shape = list(_shape(input_ids))
    embeddings_shape = list(_shape(pre_final_rmsnorm_embeddings))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    payload_path = output_dir / PAYLOAD_NAME
    if manifest_path.exists() or payload_path.exists():
        raise FileExistsError(
            f"text reference already exists in {output_dir}; choose an empty output directory"
        )

    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "model": {
            "repo": MODEL_REPO,
            "revision_sha": revision,
            "revision_source": model_revision_source,
            "pipeline_class": PIPELINE_CLASS,
            "text_encoder_class": TEXT_ENCODER_CLASS,
            "processor_class": PROCESSOR_CLASS,
        },
        "prompt": prompt,
        "tokenization": {
            "raw_template_text": raw_template_text,
            "processor_kwargs": processor_kwargs or {
                "padding": True,
                "padding_side": "left",
                "return_tensors": "pt",
            },
            "tokenizer_truncation": False,
        },
        "sequence": {
            "max_sequence_length": max_sequence_length,
            "max_sequence_length_semantics": "post-drop validation guard only; no processor truncation",
            "actual_sequence_length": actual_sequence_length,
            "raw_input_shape": ids_shape,
            "drop_idx": int(drop_idx),
        },
        "embedding": {
            "source": "QwenImage21Pipeline._get_qwen_prompt_embeds",
            "pre_final_rmsnorm": True,
            "rmsnorm_hook_observed_and_verified": True,
            "shape": embeddings_shape,
            "source_dtype": source_dtype,
            "expected_decoder_layer_count": expected_hidden_state_count - 1,
            "expected_hidden_state_count": expected_hidden_state_count,
            "expected_hidden_state_count_source": (
                "loaded text_encoder.config.text_config.num_hidden_layers + 1"
            ),
            "hidden_state_count": len(layer_hidden_states),
        },
        "runtime": {
            "source_dtype": source_dtype,
            "device": device,
            **runtime,
        },
        "payload_file": PAYLOAD_NAME,
        "payload_nbytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "tensors": tensors,
    }

    payload_created = False
    manifest_created = False
    try:
        with payload_path.open("xb") as payload_file:
            payload_created = True
            payload_file.write(payload)
        with manifest_path.open("x", encoding="utf-8") as manifest_file:
            manifest_created = True
            json.dump(manifest, manifest_file, indent=2, ensure_ascii=False, allow_nan=False)
            manifest_file.write("\n")
    except Exception:
        # Remove only files created by this invocation; never touch a preexisting
        # manifest or payload if another writer won the exclusive-create race.
        if manifest_created:
            try:
                manifest_path.unlink()
            except OSError:
                pass
        if payload_created:
            try:
                payload_path.unlink()
            except OSError:
                pass
        raise
    return manifest_path


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    return requested


def _load_local_pipeline(model_dir: Path, device: str, source_dtype: str):
    try:
        import torch
        import diffusers
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "text reference capture requires PyTorch, Transformers with Qwen3-VL, and "
            "Diffusers with QwenImage21Pipeline"
        ) from exc

    diffusers_commit = _installed_diffusers_commit()
    if diffusers_commit != PINNED_DIFFUSERS_COMMIT:
        raise RuntimeError(
            "installed Diffusers source must be pinned at "
            f"{PINNED_DIFFUSERS_COMMIT}; observed {diffusers_commit or 'unknown'}"
        )
    resolved_device = _resolve_device(torch, device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[source_dtype]
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_dir / "processor"), local_files_only=True
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir / "text_encoder"),
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not load local Qwen3-VL processor/text encoder (no network fallback): {exc}"
        ) from exc
    text_encoder = text_encoder.to(device=resolved_device)
    text_encoder.eval()
    pipeline = QwenImage21Pipeline(
        scheduler=None,
        vae=None,
        text_encoder=text_encoder,
        processor=processor,
        transformer=None,
    )
    return torch, diffusers, transformers, pipeline, resolved_device


def _peak_rss_bytes() -> int | None:
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # macOS reports bytes; Linux and the other supported Unix hosts report KiB.
    return peak if sys.platform == "darwin" else peak * 1024


def prepare_text_reference(
    *,
    model_dir: str | Path,
    output_dir: str | Path,
    prompt: str,
    revision: str | None = None,
    device: str = "auto",
    source_dtype: str = "bfloat16",
    max_sequence_length: int = 1024,
) -> Path:
    """Load local Qwen3-VL and write a text-only native-parity reference."""
    if source_dtype not in SOURCE_DTYPES:
        raise ValueError("source_dtype must be bfloat16, float16, or float32")
    if not _is_positive_int(max_sequence_length):
        raise ValueError("max_sequence_length must be a positive integer")
    model_root = Path(model_dir)
    _validate_model_directory(model_root)
    model_revision, revision_source = _resolve_revision(model_root, revision)

    load_started = time.perf_counter()
    torch, diffusers, transformers, pipeline, device_name = _load_local_pipeline(
        model_root, device, source_dtype
    )
    load_seconds = time.perf_counter() - load_started

    capture_started = time.perf_counter()
    try:
        with torch.inference_mode():
            reference = capture_pipeline_reference(
                pipeline,
                prompt,
                max_sequence_length=max_sequence_length,
                device=torch.device(device_name),
                source_dtype=source_dtype,
            )
    except Exception as exc:
        raise RuntimeError(f"official QwenImage21 text reference capture failed: {exc}") from exc
    capture_seconds = time.perf_counter() - capture_started
    runtime = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "diffusers_version": str(diffusers.__version__),
        "diffusers_commit": _installed_diffusers_commit(),
        "official_source_file": "diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py",
        "load_seconds": load_seconds,
        "capture_seconds": capture_seconds,
        "peak_rss_bytes_before_serialization": _peak_rss_bytes(),
    }
    return write_reference_bundle(
        output_dir=output_dir,
        model_revision=model_revision,
        device=device_name,
        runtime=runtime,
        model_revision_source=revision_source,
        **reference,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="local Qwen-Image 2.1 snapshot directory")
    parser.add_argument("--output-dir", required=True, help="new or empty reference output directory")
    parser.add_argument("--prompt", required=True, help="one text prompt to encode")
    parser.add_argument("--revision", help="full model commit SHA; inferred from local cache metadata when available")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=tuple(sorted(SOURCE_DTYPES)), default="bfloat16")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=1024,
        help="post-drop sequence-length guard only; the official processor call is not truncated",
    )
    args = parser.parse_args(argv)
    try:
        manifest_path = prepare_text_reference(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            prompt=args.prompt,
            revision=args.revision,
            device=args.device,
            source_dtype=args.dtype,
            max_sequence_length=args.max_sequence_length,
        )
    except Exception as exc:
        print(f"qwen_image21_text_reference: error: {exc}", file=sys.stderr)
        return 1
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
