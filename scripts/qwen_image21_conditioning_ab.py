#!/usr/bin/env python3
"""Create a checksummed Qwen-Image 2.1 bundle with native retained embeddings.

This is a diagnostic bundle transform only. It replaces the retained
``encoder_hidden_states`` tensor in an existing CPU-BF16 conditioning bundle
with BF16 values from a pinned-metadata, checksummed native sidecar, expanded to the
bundle's required float32-le representation. Masks and seeded latents remain
byte-identical. It does not run the DiT, scheduler, or VAE.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


BUNDLE_SCHEMA = "qwen-image21-conditioning"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "qwen_image21_conditioning.json"
BUNDLE_PAYLOAD_NAME = "qwen_image21_conditioning.bin"
MAX_IMAGE_SIDE = 4096
BUNDLE_TENSOR_NAMES = (
    "encoder_hidden_states",
    "encoder_hidden_states_mask",
    "encoder_img_mask",
    "initial_target_latents",
)
NATIVE_SCHEMA = "qwen3vl-retained-embeddings"
NATIVE_SCHEMA_VERSION = 1
NATIVE_DROP_IDX = 14
CONTEXT_DIM = 4096
NATIVE_SHAPE = [10, CONTEXT_DIM]
NATIVE_DTYPE = "bfloat16-le"
PINNED_REFERENCE_FIXTURE_SHA256 = "3edcd7bf7964237d649a43c35cd82f6d6bd7b15835fddec2b1fe42f3a89b1e07"
# SHA of the BF16 pre_final_rmsnorm_embeddings tensor inside that fixture,
# not the fixture payload SHA or the conditioning bundle's F32 payload SHA.
PINNED_REFERENCE_EMBEDDINGS_SHA256 = "4de67b800ab96fb9d71b5eecf0b6c20c61758450bd98b1e8d97e9707fdf8b388"
PINNED_PROMPT = "red cube"
PINNED_MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REFERENCE_SCHEMA = "qwen-image21-text-reference"
REFERENCE_SCHEMA_VERSION = 1
MAX_REFERENCE_RAW_TOKENS = 256
MAX_REFERENCE_PAYLOAD_BYTES = (
    MAX_REFERENCE_RAW_TOKENS * CONTEXT_DIM * 4 * (37 + 1)
    + MAX_REFERENCE_RAW_TOKENS * 24
)
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_BASELINE_PAYLOAD_BYTES = (
    (MAX_IMAGE_SIDE // 16) * (MAX_IMAGE_SIDE // 16) * 64 * 4
    + MAX_REFERENCE_RAW_TOKENS * CONTEXT_DIM * 4
    + MAX_REFERENCE_RAW_TOKENS * 2
)
PINNED_DIFFUSERS_COMMIT = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = _read_regular_file(path, description, max_bytes=MAX_MANIFEST_BYTES)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value, raw


def _read_regular_file(
    path: Path,
    description: str,
    *,
    max_bytes: int,
    expected_bytes: int | None = None,
) -> bytes:
    """Read one bounded regular file without following a symlink at open time."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{description} is missing, unreadable, or is a symlink: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{description} is missing or is not a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError(f"{description} exceeds the {max_bytes}-byte memory guard")
        if expected_bytes is not None and metadata.st_size != expected_bytes:
            raise ValueError(f"{description} length does not match its declared size")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
        if len(data) != metadata.st_size or len(data) > max_bytes:
            raise ValueError(f"{description} changed size while being read or exceeds its memory guard")
        return data
    finally:
        os.close(descriptor)


def _safe_sibling(directory: Path, raw_name: Any, description: str) -> Path:
    if (
        not isinstance(raw_name, str)
        or not raw_name
        or raw_name in {".", ".."}
        or "/" in raw_name
        or "\\" in raw_name
        or PurePosixPath(raw_name).name != raw_name
    ):
        raise ValueError(f"{description} payload_file must be a sibling basename")
    path = directory / raw_name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} payload is missing or is not a regular file")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_baseline(
    bundle_dir: Path,
    *,
    expected_prompt: str = PINNED_PROMPT,
    expected_revision: str = PINNED_MODEL_REVISION,
    retained_rows: int = NATIVE_SHAPE[0],
    official_reference_bf16: bytes | None = None,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    if not bundle_dir.is_dir():
        raise ValueError(f"baseline bundle directory does not exist: {bundle_dir}")
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"baseline manifest is missing or is not a regular file: {manifest_path}")
    manifest, _ = _read_json(manifest_path, "baseline conditioning manifest")
    schema_version = manifest.get("schema_version")
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or not _is_int(schema_version)
        or schema_version != BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("baseline is not a supported qwen-image21-conditioning bundle")

    model = manifest.get("model")
    if (
        not isinstance(model, dict)
        or model.get("repo") != "Qwen/Qwen-Image-2.1"
        or not isinstance(model.get("revision"), str)
        or not REVISION_RE.fullmatch(model["revision"])
        or model["revision"] != expected_revision
    ):
        raise ValueError("baseline model must identify a pinned Qwen-Image 2.1 revision")
    prompt = manifest.get("prompt")
    if prompt != expected_prompt:
        raise ValueError("baseline prompt does not match the selected official text reference")

    payload_path = _safe_sibling(bundle_dir, manifest.get("payload_file"), "baseline")
    if payload_path.name != BUNDLE_PAYLOAD_NAME:
        raise ValueError(f"baseline payload_file must name {BUNDLE_PAYLOAD_NAME!r}")
    payload_nbytes = manifest.get("payload_nbytes")
    if (
        not _is_int(payload_nbytes)
        or payload_nbytes <= 0
        or payload_nbytes > MAX_BASELINE_PAYLOAD_BYTES
    ):
        raise ValueError(
            f"baseline conditioning payload metadata is invalid or exceeds the "
            f"{MAX_BASELINE_PAYLOAD_BYTES}-byte memory guard"
        )
    try:
        payload = _read_regular_file(
            payload_path,
            "baseline conditioning payload",
            max_bytes=MAX_BASELINE_PAYLOAD_BYTES,
            expected_bytes=payload_nbytes,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read baseline conditioning payload: {exc}") from exc
    expected_sha = manifest.get("payload_sha256")
    if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
        raise ValueError("baseline manifest has an invalid payload SHA-256")
    if payload_nbytes != len(payload):
        raise ValueError("baseline payload length does not match its manifest")
    if _sha256(payload) != expected_sha:
        raise ValueError("baseline payload SHA-256 does not match its manifest")

    image = manifest.get("image")
    if not isinstance(image, dict):
        raise ValueError("baseline manifest is missing image metadata")
    width, height = image.get("width"), image.get("height")
    img_shapes = image.get("img_shapes")
    if (
        not _is_int(width)
        or not _is_int(height)
        or width <= 0
        or height <= 0
        or width > MAX_IMAGE_SIDE
        or height > MAX_IMAGE_SIDE
        or width % 32
        or height % 32
        or not _is_int(image.get("vae_scale_factor"))
        or image.get("vae_scale_factor") != 16
    ):
        raise ValueError("baseline image dimensions or VAE scale factor are invalid")
    latent_height, latent_width = height // 16, width // 16
    if (
        not _is_int(image.get("latent_height"))
        or image.get("latent_height") != latent_height
        or not _is_int(image.get("latent_width"))
        or image.get("latent_width") != latent_width
        or not isinstance(img_shapes, list)
        or len(img_shapes) != 1
        or not isinstance(img_shapes[0], list)
        or len(img_shapes[0]) != 3
        or any(not _is_int(dimension) for dimension in img_shapes[0])
        or img_shapes != [[1, latent_height, latent_width]]
    ):
        raise ValueError("baseline latent geometry metadata does not match image dimensions")
    noise = manifest.get("noise")
    runtime = manifest.get("runtime")
    if (
        not isinstance(noise, dict)
        or noise.get("source_dtype") != "bfloat16"
        or noise.get("generator_device") != "cpu"
        or not isinstance(runtime, dict)
        or runtime.get("device") != "cpu"
    ):
        raise ValueError("baseline must declare CPU BF16 conditioning and CPU noise generation")
    latent_shape = [latent_height * latent_width, 64]
    context_shape = [retained_rows, CONTEXT_DIM]
    shapes_and_types = {
        "encoder_hidden_states": ("float32-le", context_shape),
        "encoder_hidden_states_mask": ("uint8", [retained_rows]),
        "encoder_img_mask": ("uint8", [retained_rows]),
        "initial_target_latents": ("float32-le", latent_shape),
    }
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(BUNDLE_TENSOR_NAMES):
        raise ValueError("baseline tensor descriptors do not match the conditioning bundle layout")

    expected_offset = 0
    descriptors: dict[str, dict[str, Any]] = {}
    for name in BUNDLE_TENSOR_NAMES:
        descriptor = tensors.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"baseline tensor descriptor is invalid: {name}")
        expected_dtype, expected_shape = shapes_and_types[name]
        shape = descriptor.get("shape")
        if (
            descriptor.get("dtype") != expected_dtype
            or shape != expected_shape
            or not isinstance(shape, list)
            or any(not _is_int(dim) or dim <= 0 for dim in shape)
        ):
            raise ValueError(f"baseline tensor descriptor has the wrong shape or dtype: {name}")
        expected_nbytes = int(np.prod(expected_shape, dtype=np.int64)) * (1 if expected_dtype == "uint8" else 4)
        if (
            not _is_int(descriptor.get("offset_bytes"))
            or descriptor["offset_bytes"] != expected_offset
            or not _is_int(descriptor.get("nbytes"))
            or descriptor["nbytes"] != expected_nbytes
        ):
            raise ValueError(f"baseline tensor descriptor has an invalid offset or length: {name}")
        descriptors[name] = descriptor
        expected_offset += expected_nbytes
    if expected_offset != len(payload):
        raise ValueError("baseline tensor layout does not exactly cover its payload")

    hidden_desc = descriptors["encoder_hidden_states"]
    hidden_start = hidden_desc["offset_bytes"]
    hidden_end = hidden_start + hidden_desc["nbytes"]
    hidden_values = np.frombuffer(payload[hidden_start:hidden_end], dtype="<f4")
    if not bool(np.isfinite(hidden_values).all()):
        raise ValueError("baseline encoder_hidden_states must contain only finite values")
    hidden_bits = hidden_values.view("<u4")
    if bool((hidden_bits & np.uint32(0xFFFF)).any()):
        raise ValueError("baseline embeddings are not exact BF16 values expanded to float32")
    baseline_bf16 = (hidden_bits >> np.uint32(16)).astype("<u2").tobytes()
    if official_reference_bf16 is None:
        if _sha256(baseline_bf16) != PINNED_REFERENCE_EMBEDDINGS_SHA256:
            raise ValueError("baseline embeddings do not match the pinned official reference embeddings")
    else:
        if len(official_reference_bf16) != len(baseline_bf16):
            raise ValueError("official reference embedding size differs from the baseline retained shape")
        official_words = np.frombuffer(official_reference_bf16, dtype="<u2")
        official_bits = official_words.astype("<u4") << np.uint32(16)
        official_expanded = official_bits.astype("<u4", copy=False).tobytes()
        if payload[hidden_start:hidden_end] != official_expanded:
            raise ValueError("baseline embeddings do not match the official reference embeddings")
    masks: dict[str, np.ndarray] = {}
    for name in ("encoder_hidden_states_mask", "encoder_img_mask"):
        desc = descriptors[name]
        mask = np.frombuffer(
            payload[desc["offset_bytes"] : desc["offset_bytes"] + desc["nbytes"]],
            dtype=np.uint8,
        )
        if not bool(np.logical_or(mask == 0, mask == 1).all()):
            raise ValueError(f"baseline {name} values must be 0 or 1")
        masks[name] = mask
    if not bool(masks["encoder_hidden_states_mask"].all()):
        if official_reference_bf16 is None:
            raise ValueError("pinned retained conditioning must mark all ten text tokens valid")
        raise ValueError("retained conditioning must mark all text tokens valid")
    if bool(masks["encoder_img_mask"].any()):
        raise ValueError("baseline text-to-image conditioning contains image placeholders")
    latent_desc = descriptors["initial_target_latents"]
    latent_values = np.frombuffer(
        payload[
            latent_desc["offset_bytes"] : latent_desc["offset_bytes"] + latent_desc["nbytes"]
        ],
        dtype="<f4",
    )
    if not bool(np.isfinite(latent_values).all()):
        raise ValueError("baseline initial_target_latents must contain only finite values")
    return manifest, payload, descriptors


def _reference_tensor_bytes(
    manifest: dict[str, Any],
    payload: bytes,
    *,
    name: str,
    dtype: str,
    shape: list[int],
    itemsize: int,
) -> bytes:
    tensors = manifest.get("tensors")
    descriptor = tensors.get(name) if isinstance(tensors, dict) else None
    if not isinstance(descriptor, dict):
        raise ValueError(f"official reference is missing tensor descriptor {name!r}")
    expected_nbytes = int(np.prod(shape, dtype=np.int64)) * itemsize
    descriptor_shape = descriptor.get("shape")
    if (
        descriptor.get("dtype") != dtype
        or not isinstance(descriptor_shape, list)
        or any(not _is_int(dim) for dim in descriptor_shape)
        or descriptor_shape != shape
    ):
        raise ValueError(f"official reference tensor {name!r} has the wrong dtype or shape")
    offset, nbytes, sha = (
        descriptor.get("offset_bytes"),
        descriptor.get("nbytes"),
        descriptor.get("sha256"),
    )
    if (
        not _is_int(offset)
        or not _is_int(nbytes)
        or offset < 0
        or nbytes != expected_nbytes
        or offset > len(payload)
        or nbytes > len(payload) - offset
        or not isinstance(sha, str)
        or not SHA256_RE.fullmatch(sha)
    ):
        raise ValueError(f"official reference tensor {name!r} has an invalid byte range or checksum")
    data = payload[offset : offset + nbytes]
    if _sha256(data) != sha:
        raise ValueError(f"official reference tensor {name!r} SHA-256 does not match its manifest")
    return data


def _expected_official_template(prompt: str) -> str:
    return (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _validate_official_reference(reference_manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(reference_manifest_path).expanduser()
    manifest, reference_manifest_bytes = _read_json(manifest_path, "official text-reference manifest")
    schema_version = manifest.get("schema_version")
    if (
        manifest.get("schema") != REFERENCE_SCHEMA
        or not _is_int(schema_version)
        or schema_version != REFERENCE_SCHEMA_VERSION
    ):
        raise ValueError("official reference schema/version is unsupported")

    model = manifest.get("model")
    if (
        not isinstance(model, dict)
        or model.get("repo") != "Qwen/Qwen-Image-2.1"
        or model.get("revision_sha") != PINNED_MODEL_REVISION
        or model.get("pipeline_class") != "QwenImage21Pipeline"
        or model.get("text_encoder_class") != "Qwen3VLForConditionalGeneration"
        or model.get("processor_class") != "Qwen3VLProcessor"
        or model.get("revision_source") not in {
            "local_cache_metadata",
            "argument",
            "argument_and_local_cache_metadata",
        }
    ):
        raise ValueError("official reference model classes and revision must match the pinned Qwen-Image 2.1 capture")
    prompt = manifest.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("official reference prompt must be a non-empty string")
    tokenization = manifest.get("tokenization")
    if not isinstance(tokenization, dict) or not isinstance(tokenization.get("raw_template_text"), str):
        raise ValueError("official reference tokenization metadata is invalid")
    if tokenization.get("raw_template_text") != _expected_official_template(prompt):
        raise ValueError("official reference template must contain the exact prompt in the expected system/user/assistant format")
    if (
        tokenization.get("processor_kwargs")
        != {"padding": True, "padding_side": "left", "return_tensors": "pt"}
        or tokenization.get("tokenizer_truncation") is not False
    ):
        raise ValueError("official reference processor metadata must record left padding and disabled tokenizer truncation")

    sequence = manifest.get("sequence")
    if not isinstance(sequence, dict):
        raise ValueError("official reference sequence metadata is missing")
    raw_shape = sequence.get("raw_input_shape")
    raw_tokens = (
        raw_shape[1]
        if (
            isinstance(raw_shape, list)
            and len(raw_shape) == 2
            and _is_int(raw_shape[0])
            and raw_shape[0] == 1
        )
        else None
    )
    retained_rows = sequence.get("actual_sequence_length")
    drop_idx = sequence.get("drop_idx")
    max_sequence_length = sequence.get("max_sequence_length")
    if (
        not _is_int(raw_tokens)
        or raw_tokens <= 0
        or raw_tokens > MAX_REFERENCE_RAW_TOKENS
        or not _is_int(retained_rows)
        or retained_rows <= 0
        or retained_rows > raw_tokens
        or not _is_int(drop_idx)
        or drop_idx < 0
        or not _is_int(max_sequence_length)
        or max_sequence_length <= 0
        or retained_rows > max_sequence_length
        or drop_idx != NATIVE_DROP_IDX
        or sequence.get("max_sequence_length_semantics")
        != "post-drop validation guard only; no processor truncation"
    ):
        raise ValueError(
            f"official reference counts must be valid, drop_idx must be {NATIVE_DROP_IDX}, and raw tokens must not exceed {MAX_REFERENCE_RAW_TOKENS}"
        )

    embedding = manifest.get("embedding")
    embedding_shape = embedding.get("shape") if isinstance(embedding, dict) else None
    hidden_state_count = embedding.get("hidden_state_count") if isinstance(embedding, dict) else None
    expected_hidden_state_count = embedding.get("expected_hidden_state_count") if isinstance(embedding, dict) else None
    expected_decoder_layer_count = embedding.get("expected_decoder_layer_count") if isinstance(embedding, dict) else None
    if (
        not isinstance(embedding, dict)
        or embedding.get("source") != "QwenImage21Pipeline._get_qwen_prompt_embeds"
        or embedding.get("pre_final_rmsnorm") is not True
        or embedding.get("rmsnorm_hook_observed_and_verified") is not True
        or not isinstance(embedding_shape, list)
        or any(not _is_int(dim) for dim in embedding_shape)
        or embedding_shape != [1, retained_rows, CONTEXT_DIM]
        or embedding.get("source_dtype") != "bfloat16"
        or not _is_int(hidden_state_count)
        or hidden_state_count != 37
        or not _is_int(expected_hidden_state_count)
        or expected_hidden_state_count != 37
        or not _is_int(expected_decoder_layer_count)
        or expected_decoder_layer_count != 36
        or embedding.get("expected_hidden_state_count_source")
        != "loaded text_encoder.config.text_config.num_hidden_layers + 1"
    ):
        raise ValueError("official reference retained embedding shape or model layer count is invalid")

    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("source_dtype") != "bfloat16"
        or runtime.get("device") != "cpu"
        or not isinstance(runtime.get("diffusers_version"), str)
        or not runtime.get("diffusers_version")
        or runtime.get("diffusers_commit") != PINNED_DIFFUSERS_COMMIT
        or runtime.get("official_source_file")
        != "diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py"
    ):
        raise ValueError("official reference runtime metadata does not identify the pinned CPU/Diffusers capture")

    expected_payload_sha = manifest.get("payload_sha256")
    payload_nbytes = manifest.get("payload_nbytes")
    if (
        manifest.get("payload_file") != "qwen_image21_text_reference.bin"
        or not isinstance(expected_payload_sha, str)
        or not SHA256_RE.fullmatch(expected_payload_sha)
        or not _is_int(payload_nbytes)
        or payload_nbytes <= 0
        or payload_nbytes > MAX_REFERENCE_PAYLOAD_BYTES
    ):
        raise ValueError("official reference payload metadata is invalid or exceeds the memory guard")
    payload_path = _safe_sibling(
        manifest_path.parent, manifest.get("payload_file"), "official reference"
    )
    payload = _read_regular_file(
        payload_path,
        "official reference payload",
        max_bytes=MAX_REFERENCE_PAYLOAD_BYTES,
        expected_bytes=payload_nbytes,
    )
    if _sha256(payload) != expected_payload_sha:
        raise ValueError("official reference payload SHA-256 does not match its manifest")

    _reference_tensor_bytes(
        manifest,
        payload,
        name="input_ids",
        dtype="int64-le",
        shape=[1, raw_tokens],
        itemsize=8,
    )
    mask_bytes = _reference_tensor_bytes(
        manifest,
        payload,
        name="attention_mask",
        dtype="int64-le",
        shape=[1, raw_tokens],
        itemsize=8,
    )
    mask = np.frombuffer(mask_bytes, dtype="<i8")
    if not bool(np.logical_or(mask == 0, mask == 1).all()):
        raise ValueError("official reference attention_mask must contain only 0/1 values")
    if int(np.count_nonzero(mask)) - drop_idx != retained_rows:
        raise ValueError("official reference retained length does not match attention_mask and drop_idx")

    official_embedding_bf16 = _reference_tensor_bytes(
        manifest,
        payload,
        name="pre_final_rmsnorm_embeddings",
        dtype="bfloat16-le",
        shape=[1, retained_rows, CONTEXT_DIM],
        itemsize=2,
    )
    tensors = manifest["tensors"]
    expected_names = {
        "input_ids",
        "attention_mask",
        "pre_final_rmsnorm_embeddings",
        *(f"hidden_state_{index:03d}" for index in range(37)),
    }
    if "mm_token_type_ids" in tensors:
        _reference_tensor_bytes(
            manifest,
            payload,
            name="mm_token_type_ids",
            dtype="int64-le",
            shape=[1, raw_tokens],
            itemsize=8,
        )
        expected_names.add("mm_token_type_ids")
    if set(tensors) != expected_names:
        raise ValueError("official reference tensor names do not match its declared 37 hidden states")
    for name in expected_names:
        if name.startswith("hidden_state_"):
            descriptor = tensors[name]
            if not isinstance(descriptor, dict) or descriptor.get("dtype") != "bfloat16-le":
                raise ValueError(f"official reference hidden state {name!r} must use bfloat16-le")
    ranges: list[tuple[int, int]] = []
    for name in expected_names:
        descriptor = tensors[name]
        if name.startswith("hidden_state_"):
            _reference_tensor_bytes(
                manifest,
                payload,
                name=name,
                dtype="bfloat16-le",
                shape=[1, raw_tokens, CONTEXT_DIM],
                itemsize=2,
            )
        descriptor = tensors[name]
        ranges.append((descriptor["offset_bytes"], descriptor["offset_bytes"] + descriptor["nbytes"]))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("official reference tensor payload ranges must be contiguous and non-overlapping")
        cursor = end
    if cursor != len(payload):
        raise ValueError("official reference tensor descriptors do not cover the complete payload")
    final_hidden_bytes = _reference_tensor_bytes(
        manifest,
        payload,
        name="hidden_state_036",
        dtype="bfloat16-le",
        shape=[1, raw_tokens, CONTEXT_DIM],
        itemsize=2,
    )
    final_hidden_words = np.frombuffer(final_hidden_bytes, dtype="<u2")
    final_hidden_values = (final_hidden_words.astype("<u4") << np.uint32(16)).view("<f4")
    attended_final_hidden = final_hidden_values.reshape(raw_tokens, CONTEXT_DIM)[mask.astype(bool)][drop_idx:]
    official_embedding_words = np.frombuffer(official_embedding_bf16, dtype="<u2")
    official_embedding_values = (
        (official_embedding_words.astype("<u4") << np.uint32(16))
        .view("<f4")
        .reshape(retained_rows, CONTEXT_DIM)
    )
    if (
        not bool(np.isfinite(attended_final_hidden).all())
        or not np.array_equal(attended_final_hidden, official_embedding_values)
    ):
        raise ValueError("official reference hidden_state_036 retained rows differ from pre-final-RMSNorm embeddings")
    return {
        "prompt": prompt,
        "model_revision": PINNED_MODEL_REVISION,
        "raw_tokens": raw_tokens,
        "retained_rows": retained_rows,
        "drop_idx": drop_idx,
        "payload_sha256": expected_payload_sha,
        "manifest_sha256": _sha256(reference_manifest_bytes),
        "official_embedding_bf16": official_embedding_bf16,
    }


def _validate_native_sidecar(
    manifest_path: Path,
    native_manifest: dict[str, Any],
    native_manifest_bytes: bytes,
    baseline: dict[str, Any],
    reference: dict[str, Any] | None = None,
) -> tuple[bytes, str]:
    schema_version = native_manifest.get("schema_version")
    if (
        native_manifest.get("schema") != NATIVE_SCHEMA
        or not _is_int(schema_version)
        or schema_version != NATIVE_SCHEMA_VERSION
    ):
        raise ValueError("native manifest schema/version is unsupported")
    for name in ("prompt", "model_revision"):
        expected = baseline["prompt"] if name == "prompt" else baseline["model"]["revision"]
        if native_manifest.get(name) != expected:
            raise ValueError(f"native manifest {name} does not match the baseline bundle")
    expected_fixture_sha = (
        reference["payload_sha256"] if reference is not None else PINNED_REFERENCE_FIXTURE_SHA256
    )
    if native_manifest.get("fixture_payload_sha256") != expected_fixture_sha:
        raise ValueError("native manifest fixture_payload_sha256 does not match the selected text-reference fixture")
    if reference is not None and native_manifest.get("fixture_manifest_sha256") != reference["manifest_sha256"]:
        raise ValueError("native manifest fixture_manifest_sha256 does not match the selected official text-reference manifest")
    drop_idx = native_manifest.get("drop_idx")
    expected_drop_idx = reference["drop_idx"] if reference is not None else NATIVE_DROP_IDX
    if not _is_int(drop_idx) or drop_idx != expected_drop_idx:
        raise ValueError(f"native manifest drop_idx must be {expected_drop_idx}")
    expected_shape = (
        [reference["retained_rows"], CONTEXT_DIM] if reference is not None else NATIVE_SHAPE
    )
    native_shape = native_manifest.get("shape")
    if (
        not isinstance(native_shape, list)
        or any(not _is_int(dim) for dim in native_shape)
        or native_shape != expected_shape
    ):
        raise ValueError(f"native manifest shape must be {expected_shape}")
    if native_manifest.get("dtype") != NATIVE_DTYPE:
        raise ValueError(f"native manifest dtype must be {NATIVE_DTYPE!r}")
    expected_nbytes = expected_shape[0] * expected_shape[1] * 2
    if not _is_int(native_manifest.get("nbytes")) or native_manifest["nbytes"] != expected_nbytes:
        raise ValueError(f"native manifest nbytes must be {expected_nbytes}")
    expected_sha = native_manifest.get("sha256")
    if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
        raise ValueError("native manifest has an invalid sidecar SHA-256")

    sidecar_path = _safe_sibling(manifest_path.parent, native_manifest.get("payload_file"), "native")
    try:
        sidecar = _read_regular_file(
            sidecar_path,
            "native BF16 sidecar",
            max_bytes=MAX_REFERENCE_RAW_TOKENS * CONTEXT_DIM * 2,
            expected_bytes=expected_nbytes,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read native BF16 sidecar: {exc}") from exc
    if len(sidecar) != expected_nbytes:
        raise ValueError("native BF16 sidecar length does not match its manifest")
    if _sha256(sidecar) != expected_sha:
        raise ValueError("native BF16 sidecar SHA-256 does not match its manifest")

    # BF16 little-endian is the high 16 bits of each little-endian Float32 word.
    words = np.frombuffer(sidecar, dtype="<u2")
    expanded_bits = words.astype("<u4") << np.uint32(16)
    values = expanded_bits.view("<f4")
    if values.size != expected_shape[0] * expected_shape[1] or not bool(np.isfinite(values).all()):
        raise ValueError("native BF16 sidecar must expand to finite retained embeddings")
    return np.ascontiguousarray(values, dtype="<f4").tobytes(order="C"), _sha256(native_manifest_bytes)


def create_ab_bundle(
    baseline_bundle_dir: str | Path,
    native_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    reference_manifest_path: str | Path | None = None,
) -> Path:
    """Clone a validated baseline bundle, replacing only its embedding bytes."""
    baseline_dir = Path(baseline_bundle_dir).expanduser().resolve(strict=True)
    native_requested = Path(native_manifest_path).expanduser()
    if native_requested.is_symlink():
        raise ValueError("native manifest must not be a symlink")
    native_path = native_requested.resolve(strict=True)
    reference = (
        _validate_official_reference(reference_manifest_path)
        if reference_manifest_path is not None
        else None
    )
    if reference is None:
        baseline_manifest, baseline_payload, descriptors = _validate_baseline(baseline_dir)
    else:
        baseline_manifest, baseline_payload, descriptors = _validate_baseline(
            baseline_dir,
            expected_prompt=reference["prompt"],
            expected_revision=reference["model_revision"],
            retained_rows=reference["retained_rows"],
            official_reference_bf16=reference["official_embedding_bf16"],
        )
        if baseline_manifest["prompt"] != reference["prompt"]:
            raise ValueError("baseline prompt does not match the selected official text reference")
    native_manifest, native_manifest_bytes = _read_json(native_path, "native embedding manifest")
    native_hidden_bytes, native_manifest_sha256 = _validate_native_sidecar(
        native_path, native_manifest, native_manifest_bytes, baseline_manifest, reference
    )

    output_requested = Path(output_dir).expanduser()
    try:
        output_parent = output_requested.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"output parent directory does not exist: {output_requested.parent}") from exc
    if not output_parent.is_dir():
        raise ValueError(f"output parent is not a directory: {output_parent}")
    target_dir = output_parent / output_requested.name
    if os.path.lexists(target_dir):
        raise FileExistsError(f"output directory already exists; refusing to overwrite: {target_dir}")
    if target_dir == baseline_dir or baseline_dir in target_dir.parents:
        raise ValueError("output directory must not be inside the baseline bundle")

    output_payload = bytearray(baseline_payload)
    hidden_desc = descriptors["encoder_hidden_states"]
    start = hidden_desc["offset_bytes"]
    end = start + hidden_desc["nbytes"]
    if len(native_hidden_bytes) != end - start:
        raise ValueError("native embedding byte length differs from the baseline tensor layout")
    output_payload[start:end] = native_hidden_bytes
    output_payload_bytes = bytes(output_payload)

    output_manifest = copy.deepcopy(baseline_manifest)
    output_manifest["payload_sha256"] = _sha256(output_payload_bytes)
    output_manifest["payload_nbytes"] = len(output_payload_bytes)
    output_manifest["conditioning_ab"] = {
        "kind": "native-retained-embeddings",
        "baseline_payload_sha256": baseline_manifest["payload_sha256"],
        "native_reference_fixture_payload_sha256": native_manifest["fixture_payload_sha256"],
        "official_reference_embeddings_sha256": (
            _sha256(reference["official_embedding_bf16"])
            if reference is not None
            else PINNED_REFERENCE_EMBEDDINGS_SHA256
        ),
        **(
            {"official_reference_manifest_sha256": reference["manifest_sha256"]}
            if reference is not None
            else {}
        ),
        "native_manifest_sha256": native_manifest_sha256,
        "native_sidecar_sha256": native_manifest["sha256"],
        "model_revision": native_manifest["model_revision"],
        "drop_idx": native_manifest["drop_idx"],
        "shape": list(native_manifest["shape"]),
        "source_dtype": NATIVE_DTYPE,
        "target_dtype": "float32-le",
        "replaced_tensor": "encoder_hidden_states",
        "preserved_tensors": [
            "encoder_hidden_states_mask",
            "encoder_img_mask",
            "initial_target_latents",
        ],
    }

    # Create the destination directory exclusively, then publish payload first
    # and manifest last. Existing output paths and files are never overwritten.
    # If an I/O failure interrupts publication, retain the newly created partial
    # directory for inspection rather than deleting paths that could have been
    # concurrently replaced by another process.
    target_dir.mkdir()
    payload_path = target_dir / BUNDLE_PAYLOAD_NAME
    manifest_path = target_dir / BUNDLE_MANIFEST_NAME
    with payload_path.open("xb") as stream:
        stream.write(output_payload_bytes)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(output_manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", type=Path, required=True, help="validated baseline conditioning bundle directory")
    parser.add_argument("--native-manifest", type=Path, required=True, help="retained-embedding metadata JSON; payload_file resolves as its sibling")
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help="opt in to validate a non-default official text-reference manifest and its payload",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="new, non-existing directory for the A/B conditioning bundle")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = create_ab_bundle(
            args.baseline_bundle,
            args.native_manifest,
            args.output_dir,
            reference_manifest_path=args.reference_manifest,
        )
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"qwen_image21_conditioning_ab: {exc}", file=sys.stderr)
        return 2
    manifest, _ = _read_json(result, "written A/B conditioning manifest")
    print(f"conditioning_ab_manifest={result}")
    print(f"conditioning_ab_payload_sha256={manifest['payload_sha256']}")
    print("validation=payload hash/layout, official reference binding, native checksums and finite values; masks and initial latents preserved byte-for-byte")
    print("scope=conditioning bundle only; no DiT/VAE inference or image-quality claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
