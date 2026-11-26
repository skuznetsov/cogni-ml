#!/usr/bin/env python3
"""Capture a real BF16 prefix of the official Qwen2.5-VL text decoder.

The producer reads a pinned local Hugging Face checkpoint, selects only the
token embedding rows referenced by the input IDs and the first 26 text decoder
layers, and executes those layers through the pinned Transformers SDPA path on
MPS.  It writes one small JSON manifest and one BF16 sidecar containing the
embedding output followed by every decoder output.  No model weights are
copied into the fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any


SCHEMA = "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference"
SCHEMA_VERSION = 1
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "eb6ba55b40ada64977147b52e50a4145ee89ac4f"
SOURCE_LAYER_COUNT = 28
PREFIX_LAYER_COUNT = 26
EMBEDDING_KEY = "model.embed_tokens.weight"
DEFAULT_TOKEN_IDS = (151644, 872, 151645, 0)
DEFAULT_ATTENTION_MASK = (1, 1, 1, 0)
MAX_SEQUENCE_LENGTH = 1280
MAX_NATURAL_SEQUENCE_LENGTH = 1536
MAX_SIDECAR_BYTES = 512 * 1024 * 1024
SIDECAR_ENCODING = "bfloat16-le"
RTOL = 2.0e-5
ATOL = 2.0e-5
MPS_RESERVE_FRACTION = 0.15

REAL_TEXT_GEOMETRY = {
    "vocab_size": 152064,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "num_hidden_layers": SOURCE_LAYER_COUNT,
    "head_dim": 128,
    "max_position_embeddings": 128000,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "hidden_act": "silu",
}

UPSTREAM_HELPER_PATH = Path(__file__).resolve().with_name(
    "qwen25_hy_nomas_reference.py"
)
UPSTREAM_PROMPT_HELPER_PATH = Path(__file__).resolve().with_name(
    "qwen25_hy_nomas_prompt_reference.py"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _expected_weight_shape(local_name: str) -> list[int]:
    hidden = REAL_TEXT_GEOMETRY["hidden_size"]
    intermediate = REAL_TEXT_GEOMETRY["intermediate_size"]
    kv_width = REAL_TEXT_GEOMETRY["num_key_value_heads"] * REAL_TEXT_GEOMETRY["head_dim"]
    shapes = {
        "input_layernorm.weight": [hidden],
        "self_attn.q_proj.weight": [hidden, hidden],
        "self_attn.q_proj.bias": [hidden],
        "self_attn.k_proj.weight": [kv_width, hidden],
        "self_attn.k_proj.bias": [kv_width],
        "self_attn.v_proj.weight": [kv_width, hidden],
        "self_attn.v_proj.bias": [kv_width],
        "self_attn.o_proj.weight": [hidden, hidden],
        "post_attention_layernorm.weight": [hidden],
        "mlp.gate_proj.weight": [intermediate, hidden],
        "mlp.up_proj.weight": [intermediate, hidden],
        "mlp.down_proj.weight": [hidden, intermediate],
    }
    try:
        return shapes[local_name]
    except KeyError as error:
        raise ValueError(f"unexpected Qwen2.5-VL decoder parameter: {local_name}") from error


def _validate_inputs(
    token_ids: list[int] | tuple[int, ...],
    attention_mask: list[int] | tuple[int, ...],
    vocab_size: int,
    position_offset: int,
    hidden_size: int,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> dict[str, Any]:
    if not isinstance(token_ids, (list, tuple)) or not token_ids:
        raise ValueError("token IDs must be a nonempty integer sequence")
    if (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or max_sequence_length < 1
    ):
        raise ValueError("sequence length limit must be a positive integer")
    if len(token_ids) > max_sequence_length:
        raise ValueError(f"sequence length exceeds the {max_sequence_length}-token limit")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
        raise ValueError("token IDs must be integers")
    if any(token < 0 or token >= vocab_size for token in token_ids):
        raise ValueError("token ID is outside the checkpoint vocabulary")
    if not isinstance(attention_mask, (list, tuple)) or len(attention_mask) != len(token_ids):
        raise ValueError("attention mask length must match the token IDs")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
        for value in attention_mask
    ):
        raise ValueError("attention mask must contain only integer 0/1 values")
    if attention_mask[0] != 1 or not any(attention_mask):
        raise ValueError("attention mask must expose at least the first token")
    first_padding = next(
        (index for index, value in enumerate(attention_mask) if value == 0),
        len(attention_mask),
    )
    if any(attention_mask[first_padding:]):
        raise ValueError("attention mask must be right padded")
    if any(not any(attention_mask[:query + 1]) for query in range(len(token_ids))):
        raise ValueError("causal attention contains an empty query row")
    if isinstance(position_offset, bool) or not isinstance(position_offset, int) or position_offset < 0:
        raise ValueError("position offset must be a nonnegative integer")
    if position_offset + len(token_ids) > 128000:
        raise ValueError("position IDs exceed the checkpoint context limit")
    return {
        "token_ids": list(token_ids),
        "attention_mask": list(attention_mask),
        "position_ids_f32": [
            [float(position_offset + index) for index in range(len(token_ids))]
            for _ in range(3)
        ],
        "position_ids_for_official_model": [
            [position_offset + index for index in range(len(token_ids))]
            for _ in range(3)
        ],
        "multimodal_position_shape": [3, 1, len(token_ids)],
        "position_offset": position_offset,
        "hidden_size": hidden_size,
        "sequence_length": len(token_ids),
    }


def _validate_receipt_metadata(receipt: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(receipt, dict):
        raise ValueError("checkpoint receipt must be a JSON object")
    if receipt.get("repo_id") != MODEL_ID or receipt.get("revision") != MODEL_REVISION:
        raise ValueError("checkpoint receipt does not pin the requested Qwen model revision")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("checkpoint receipt has no file records")
    normalized: dict[str, dict[str, Any]] = {}
    for filename, metadata in files.items():
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
        ):
            raise ValueError("checkpoint receipt filenames must be sibling basenames")
        if not isinstance(metadata, dict):
            raise ValueError(f"checkpoint receipt metadata is invalid for {filename}")
        nbytes = metadata.get("nbytes")
        sha256 = metadata.get("sha256")
        lfs_sha256 = metadata.get("hf_lfs_sha256")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
            raise ValueError(f"checkpoint receipt size is invalid for {filename}")
        if not _is_sha256(sha256):
            raise ValueError(f"checkpoint receipt SHA-256 is invalid for {filename}")
        if lfs_sha256 is not None and not _is_sha256(lfs_sha256):
            raise ValueError(f"checkpoint LFS SHA-256 is invalid for {filename}")
        if lfs_sha256 is not None and lfs_sha256 != sha256:
            raise ValueError(f"checkpoint LFS SHA-256 differs from file SHA-256 for {filename}")
        normalized[filename] = {
            "nbytes": nbytes,
            "sha256": sha256,
            "hf_lfs_sha256": lfs_sha256,
        }
    for required in ("config.json", "model.safetensors.index.json"):
        if required not in normalized:
            raise ValueError(f"checkpoint receipt is missing {required}")
    return normalized


def _verify_receipt_files(
    checkpoint_root: Path, receipt_path: Path, receipt: dict[str, Any], required_files: set[str]
) -> dict[str, Any]:
    root = checkpoint_root.resolve(strict=True)
    receipt_files = _validate_receipt_metadata(receipt)
    records = {}
    for filename in sorted(required_files):
        if filename not in receipt_files:
            raise ValueError(f"checkpoint receipt is missing selected file {filename}")
        path = root / filename
        try:
            resolved = path.resolve(strict=True)
            stat = path.stat()
        except OSError as error:
            raise ValueError(f"checkpoint file is missing or inaccessible: {filename}") from error
        if not resolved.is_file():
            raise ValueError(f"checkpoint file is not a regular file: {filename}")
        expected = receipt_files[filename]
        if stat.st_size != expected["nbytes"]:
            raise ValueError(f"checkpoint file size does not match receipt: {filename}")
        actual_sha256 = _sha256_file(resolved)
        if actual_sha256 != expected["sha256"]:
            raise ValueError(f"checkpoint SHA-256 does not match receipt: {filename}")
        if expected["hf_lfs_sha256"] is not None and actual_sha256 != expected["hf_lfs_sha256"]:
            raise ValueError(f"checkpoint LFS SHA-256 does not match file: {filename}")
        records[filename] = {
            "path": filename,
            "nbytes": stat.st_size,
            "sha256": actual_sha256,
            "hf_lfs_sha256": expected["hf_lfs_sha256"],
        }
    return {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "root": str(root),
        "receipt_path": str(receipt_path.resolve(strict=True)),
        "receipt_sha256": _sha256_file(receipt_path),
        "verified_files": records,
    }


def _text_config(model_config: Any) -> dict[str, Any]:
    if not isinstance(model_config, dict) or model_config.get("model_type") != "qwen2_5_vl":
        raise ValueError("checkpoint config is not Qwen2.5-VL")
    text = model_config.get("text_config")
    if text is None:
        # The pinned official Qwen2.5-VL revision stores text geometry at the
        # top level; later multimodal checkpoints may nest it under text_config.
        text = model_config
    if not isinstance(text, dict):
        raise ValueError("checkpoint config has no usable text geometry")
    for key, expected in REAL_TEXT_GEOMETRY.items():
        actual = text.get(key)
        if actual is None and key == "head_dim":
            heads = text.get("num_attention_heads")
            actual = text.get("hidden_size", 0) // heads if isinstance(heads, int) and heads else None
        if actual != expected:
            raise ValueError(f"checkpoint text config field {key} does not match the pinned model")
    rope_scaling = text.get("rope_scaling")
    if not isinstance(rope_scaling, dict) or rope_scaling.get("mrope_section") != [16, 24, 24]:
        raise ValueError("checkpoint text config does not contain the pinned multimodal RoPE sections")
    return text


def _validate_manifest(document: Any, manifest_path: Path | None = None) -> None:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("fixture schema identity is invalid")
    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("repo_id") != MODEL_ID or checkpoint.get("revision") != MODEL_REVISION:
        raise ValueError("fixture checkpoint provenance is invalid")
    verified_files = checkpoint.get("verified_files")
    if not isinstance(verified_files, dict) or not {
        "config.json", "model.safetensors.index.json"
    }.issubset(verified_files):
        raise ValueError("fixture checkpoint file provenance is incomplete")
    for filename, entry in verified_files.items():
        if not isinstance(entry, dict) or entry.get("path") != filename:
            raise ValueError("fixture checkpoint file metadata is malformed")
        if not _is_sha256(entry.get("sha256")):
            raise ValueError("fixture checkpoint file hash is malformed")
        if isinstance(entry.get("nbytes"), bool) or not isinstance(entry.get("nbytes"), int) or entry["nbytes"] <= 0:
            raise ValueError("fixture checkpoint file size is malformed")
    if not _is_sha256(checkpoint.get("receipt_sha256")):
        raise ValueError("fixture checkpoint receipt hash is malformed")
    if checkpoint.get("config_sha256") != verified_files["config.json"]["sha256"]:
        raise ValueError("fixture checkpoint config hash does not match its verified file")
    if checkpoint.get("index_sha256") != verified_files["model.safetensors.index.json"]["sha256"]:
        raise ValueError("fixture checkpoint index hash does not match its verified file")
    capture = document.get("capture")
    config = document.get("config")
    inputs = document.get("inputs")
    if not isinstance(capture, dict) or not isinstance(config, dict) or not isinstance(inputs, dict):
        raise ValueError("fixture is missing capture, config, or input metadata")
    if capture.get("source_layer_count") != SOURCE_LAYER_COUNT or capture.get("prefix_layer_count") != PREFIX_LAYER_COUNT:
        raise ValueError("fixture prefix layer count is invalid")
    if capture.get("captured_state_count") != PREFIX_LAYER_COUNT + 1:
        raise ValueError("fixture must contain the embedding output and all 26 layer outputs")
    if capture.get("state_indexing") != "0=token_embedding; i+1=decoder_layer_i_output":
        raise ValueError("fixture hidden-state indexing convention is invalid")
    if capture.get("embedding_source_key") != EMBEDDING_KEY:
        raise ValueError("fixture embedding source key is invalid")
    if config.get("hidden_size") != REAL_TEXT_GEOMETRY["hidden_size"] or config.get("num_hidden_layers") != SOURCE_LAYER_COUNT:
        raise ValueError("fixture model geometry is invalid")
    if config.get("attention_implementation") != "sdpa":
        raise ValueError("fixture must use the official SDPA attention implementation")
    token_ids = inputs.get("token_ids")
    mask = inputs.get("attention_mask")
    prompt_contract = document.get("hy_nomas_prompt")
    sequence_limit = (
        MAX_NATURAL_SEQUENCE_LENGTH if prompt_contract is not None else MAX_SEQUENCE_LENGTH
    )
    request = _validate_inputs(
        token_ids,
        mask,
        REAL_TEXT_GEOMETRY["vocab_size"],
        inputs.get("position_offset"),
        REAL_TEXT_GEOMETRY["hidden_size"],
        sequence_limit,
    )
    if prompt_contract is not None:
        _load_upstream_prompt_helper().validate_prompt_metadata(
            prompt_contract,
            token_ids,
            mask,
            inputs.get("position_ids_f32"),
            inputs.get("position_ids_for_official_model"),
            inputs.get("multimodal_position_shape"),
            inputs.get("position_offset"),
        )
    if request["sequence_length"] != config.get("sequence_length"):
        raise ValueError("fixture sequence length does not match its inputs")
    if inputs.get("position_ids_f32") != request["position_ids_f32"]:
        raise ValueError("fixture F32 position IDs do not match the configured offset")
    if inputs.get("position_ids_for_official_model") != request["position_ids_for_official_model"]:
        raise ValueError("fixture official-model positions do not match the configured offset")
    inv_freq = inputs.get("inv_freq_f32")
    if not isinstance(inv_freq, list) or len(inv_freq) != REAL_TEXT_GEOMETRY["head_dim"] // 2:
        raise ValueError("fixture F32 inverse frequencies are missing or malformed")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in inv_freq):
        raise ValueError("fixture F32 inverse frequencies contain invalid values")
    outputs = capture.get("hidden_states")
    if not isinstance(outputs, dict) or outputs.get("dtype") != "bfloat16":
        raise ValueError("fixture hidden-state sidecar tensor is missing or not BF16")
    expected_shape = [PREFIX_LAYER_COUNT + 1, 1, request["sequence_length"], REAL_TEXT_GEOMETRY["hidden_size"]]
    if outputs.get("shape") != expected_shape:
        raise ValueError("fixture hidden-state tensor does not match the declared 26-layer prefix")
    storage = outputs.get("storage")
    if not isinstance(storage, dict) or storage.get("sidecar") != "bf16":
        raise ValueError("fixture hidden-state tensor does not reference the BF16 sidecar")
    if storage.get("offset_bytes") != 0 or storage.get("nbytes") != math.prod(expected_shape) * 2:
        raise ValueError("fixture hidden-state sidecar extent is invalid")
    sidecars = document.get("sidecars")
    if not isinstance(sidecars, dict) or set(sidecars) != {"bf16"}:
        raise ValueError("fixture must declare exactly one BF16 sidecar")
    descriptor = sidecars["bf16"]
    if not isinstance(descriptor, dict) or descriptor.get("encoding") != SIDECAR_ENCODING:
        raise ValueError("fixture BF16 sidecar metadata is invalid")
    if descriptor.get("nbytes") != storage["nbytes"] or not _is_sha256(descriptor.get("sha256")):
        raise ValueError("fixture BF16 sidecar size or hash is invalid")
    if manifest_path is not None:
        _, verified = _verify_fixture_sidecar(descriptor, manifest_path)
        if verified["sha256"] != descriptor["sha256"]:
            raise ValueError("fixture BF16 sidecar hash is invalid")
    weight_sources = document.get("weight_sources")
    expected_weight_count = PREFIX_LAYER_COUNT * 12
    if not isinstance(weight_sources, list) or len(weight_sources) != expected_weight_count:
        raise ValueError("fixture must declare provenance for all 26 layers' selected weights")
    expected_keys = {
        f"model.layers.{layer}.{name}"
        for layer in range(PREFIX_LAYER_COUNT)
        for name in (
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
    }
    seen_keys = set()
    for source in weight_sources:
        if not isinstance(source, dict):
            raise ValueError("fixture weight provenance entry is malformed")
        key = source.get("key")
        if key not in expected_keys or key in seen_keys:
            raise ValueError("fixture weight provenance keys are incomplete or duplicated")
        seen_keys.add(key)
        if source.get("dtype") != "BF16" or not _is_sha256(source.get("tensor_sha256")):
            raise ValueError("fixture weight tensor provenance is malformed")
        if source.get("shard") not in verified_files:
            raise ValueError("fixture weight provenance references an unverified shard")
    if seen_keys != expected_keys:
        raise ValueError("fixture weight provenance does not cover the 26-layer prefix")
    for source in weight_sources:
        local_name = source["key"].split(".", 3)[-1]
        expected_shape = _expected_weight_shape(local_name)
        if source.get("shape") != expected_shape or source.get("nbytes") != math.prod(expected_shape) * 2:
            raise ValueError("fixture weight tensor shape or byte count is invalid")
    embedding_rows = capture.get("embedding_unique_rows")
    unique_token_ids = sorted(set(token_ids))
    if not isinstance(embedding_rows, list) or [row.get("row") for row in embedding_rows if isinstance(row, dict)] != unique_token_ids:
        raise ValueError("fixture selected embedding row provenance is incomplete")
    if capture.get("embedding_source_shard") not in verified_files:
        raise ValueError("fixture embedding source shard is not verified")
    for row in embedding_rows:
        if (
            not isinstance(row, dict)
            or row.get("key") != EMBEDDING_KEY
            or row.get("shard") != capture.get("embedding_source_shard")
            or row.get("dtype") != "BF16"
            or row.get("shape") != [REAL_TEXT_GEOMETRY["hidden_size"]]
            or row.get("nbytes") != REAL_TEXT_GEOMETRY["hidden_size"] * 2
            or not _is_sha256(row.get("tensor_sha256"))
        ):
            raise ValueError("fixture selected embedding row provenance is malformed")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("fixture runtime provenance is missing")
    for key in ("producer_source_sha256", "upstream_helper_source_sha256"):
        if not _is_sha256(provenance.get(key)):
            raise ValueError(f"fixture source hash is malformed: {key}")
    if not isinstance(provenance.get("transformers_sources"), dict) or not provenance["transformers_sources"]:
        raise ValueError("fixture pinned Transformers source hashes are missing")
    if not isinstance(provenance.get("torch_sources"), dict) or not provenance["torch_sources"]:
        raise ValueError("fixture pinned PyTorch source hashes are missing")
    if provenance.get("device") != "mps" or provenance.get("activation_dtype") != "bfloat16":
        raise ValueError("fixture runtime was not captured as BF16 on MPS")
    if provenance.get("comparison_tolerance") != {"rtol": RTOL, "atol": ATOL}:
        raise ValueError("fixture comparison tolerance is invalid")
    if provenance.get("full_tuple_produced") is not False:
        raise ValueError("fixture must not claim production of the full 28-layer hidden-state tuple")
    if provenance.get("hidden_states_minus3_mapping") != (
        "source-derived: after decoder layer 25 equals hidden_states[-3] for the pinned 28-layer "
        "Qwen2_5_VLTextModel.forward tuple; this fixture runs only layers 0-25"
    ):
        raise ValueError("fixture hidden_states[-3] correspondence is missing or overclaimed")


def _verify_fixture_sidecar(descriptor: Any, manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "encoding", "nbytes", "sha256"}:
        raise ValueError("BF16 sidecar descriptor fields are invalid")
    name = descriptor.get("path")
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("BF16 sidecar path must be a sibling basename")
    if descriptor.get("encoding") != SIDECAR_ENCODING:
        raise ValueError("BF16 sidecar encoding is unsupported")
    nbytes = descriptor.get("nbytes")
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or not 0 < nbytes <= MAX_SIDECAR_BYTES or nbytes % 2:
        raise ValueError("BF16 sidecar size is outside its supported range")
    expected_sha256 = descriptor.get("sha256")
    if not _is_sha256(expected_sha256):
        raise ValueError("BF16 sidecar SHA-256 is malformed")
    root = manifest_path.parent.resolve(strict=True)
    sidecar_path = root / name
    if sidecar_path.is_symlink() or not sidecar_path.is_file() or sidecar_path.resolve(strict=True).parent != root:
        raise ValueError("BF16 sidecar must be a regular sibling file")
    if sidecar_path.stat().st_size != nbytes:
        raise ValueError("BF16 sidecar file size does not match its manifest")
    actual_sha256 = _sha256_file(sidecar_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("BF16 sidecar SHA-256 does not match its manifest")
    return sidecar_path, {"sha256": actual_sha256, "nbytes": nbytes}


def _load_upstream_helper() -> Any:
    spec = importlib.util.spec_from_file_location(
        "qwen25_hy_nomas_reference_for_pretrained", UPSTREAM_HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned Qwen reference verification helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_upstream_prompt_helper() -> Any:
    spec = importlib.util.spec_from_file_location(
        "qwen25_hy_nomas_prompt_reference", UPSTREAM_PROMPT_HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned HyNomas prompt helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_text_config(configuration: Any, values: dict[str, Any]) -> Any:
    config = configuration.Qwen2_5_VLTextConfig(
        vocab_size=values["vocab_size"],
        hidden_size=values["hidden_size"],
        intermediate_size=values["intermediate_size"],
        num_hidden_layers=values["num_hidden_layers"],
        num_attention_heads=values["num_attention_heads"],
        num_key_value_heads=values["num_key_value_heads"],
        hidden_act=values["hidden_act"],
        max_position_embeddings=values["max_position_embeddings"],
        rms_norm_eps=values["rms_norm_eps"],
        rope_theta=values["rope_theta"],
        rope_scaling=dict(values["rope_scaling"]),
        attention_dropout=float(values.get("attention_dropout", 0.0)),
        use_cache=False,
        attn_implementation="sdpa",
    )
    config._attn_implementation = "sdpa"
    return config


def _materialize_config_record(
    config: Any,
    sequence_length: int,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> dict[str, Any]:
    """Validate and freeze the official text geometry before loading weights."""
    integer_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "max_position_embeddings",
    )
    record: dict[str, Any] = {}
    for key in integer_fields:
        try:
            value = getattr(config, key)
        except AttributeError as error:
            raise RuntimeError(f"official text config is missing {key}") from error
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"official text config field {key} is invalid")
        record[key] = value

    hidden_size = record["hidden_size"]
    attention_heads = record["num_attention_heads"]
    if hidden_size % attention_heads:
        raise RuntimeError("official text config hidden size is not divisible by attention heads")
    derived_head_dim = hidden_size // attention_heads
    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is not None and (
        isinstance(configured_head_dim, bool)
        or not isinstance(configured_head_dim, int)
        or configured_head_dim != derived_head_dim
    ):
        raise RuntimeError("official text config head_dim disagrees with hidden size and attention heads")
    if derived_head_dim != REAL_TEXT_GEOMETRY["head_dim"]:
        raise RuntimeError("official text config derived head_dim does not match the pinned model")
    record["head_dim"] = derived_head_dim

    for key in ("rms_norm_eps", "rope_theta"):
        try:
            value = getattr(config, key)
        except AttributeError as error:
            raise RuntimeError(f"official text config is missing {key}") from error
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"official text config field {key} is invalid")
        record[key] = value
    try:
        hidden_act = config.hidden_act
    except AttributeError as error:
        raise RuntimeError("official text config is missing hidden_act") from error
    if not isinstance(hidden_act, str) or not hidden_act:
        raise RuntimeError("official text config field hidden_act is invalid")
    record["hidden_act"] = hidden_act

    for key, expected in REAL_TEXT_GEOMETRY.items():
        if record.get(key) != expected:
            raise RuntimeError(f"official text config field {key} does not match the pinned model")

    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict) or rope_scaling.get("mrope_section") != [16, 24, 24]:
        raise RuntimeError("official text config does not contain the pinned multimodal RoPE sections")
    record["rope_scaling"] = dict(rope_scaling)

    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or not 0 < sequence_length <= max_sequence_length
    ):
        raise RuntimeError("official text config sequence length is outside the supported range")
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("official text config must use SDPA attention")
    record["sequence_length"] = sequence_length
    record["attention_implementation"] = "sdpa"
    return record


def _expected_weight_names(layer: Any) -> dict[str, tuple[int, ...]]:
    result = {name: tuple(int(value) for value in parameter.shape) for name, parameter in layer.named_parameters()}
    required = {
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    if set(result) != required:
        raise RuntimeError(f"unexpected pinned decoder layer parameters: {sorted(result)}")
    return result


def _hash_bf16_tensor(tensor: Any, torch: Any) -> str:
    if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16:
        raise RuntimeError("checkpoint source tensor must remain CPU BF16 during provenance hashing")
    words = tensor.detach().contiguous().view(torch.uint16).numpy()
    return hashlib.sha256(memoryview(words).cast("B")).hexdigest()


def _estimate_mps_bytes(
    layer_shapes: dict[str, tuple[int, ...]],
    hidden_size: int,
    intermediate_size: int,
    sequence_length: int,
    unique_embedding_rows: int,
) -> dict[str, int]:
    layer_weight_bytes = sum(math.prod(shape) * 2 for shape in layer_shapes.values())
    weights_bytes = layer_weight_bytes * PREFIX_LAYER_COUNT
    embedding_bytes = unique_embedding_rows * hidden_size * 2
    state_capture_bytes = (PREFIX_LAYER_COUNT + 1) * sequence_length * hidden_size * 2
    layer_workspace_bytes = sequence_length * (4 * hidden_size + 4 * intermediate_size) * 2
    estimate = weights_bytes + embedding_bytes + state_capture_bytes * 2 + layer_workspace_bytes + 64 * 1024 * 1024
    return {
        "layer_weights_bytes": weights_bytes,
        "selected_embedding_rows_bytes": embedding_bytes,
        "hidden_state_capture_and_pack_bytes": state_capture_bytes * 2,
        "estimated_layer_workspace_bytes": layer_workspace_bytes,
        "fixed_runtime_slack_bytes": 64 * 1024 * 1024,
        "estimated_peak_mps_bytes": estimate,
    }


def _require_memory_budget(estimated_bytes: int, current_bytes: int, recommended_max_bytes: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (estimated_bytes, current_bytes, recommended_max_bytes)):
        raise ValueError("MPS memory budget values must be nonnegative integers")
    if recommended_max_bytes <= 0:
        raise RuntimeError("MPS did not report a usable recommended memory limit")
    allowed_bytes = int(recommended_max_bytes * (1.0 - MPS_RESERVE_FRACTION))
    if current_bytes + estimated_bytes > allowed_bytes:
        raise MemoryError(
            f"estimated prefix needs {estimated_bytes + current_bytes} bytes, above the "
            f"{allowed_bytes}-byte MPS limit that preserves a {int(MPS_RESERVE_FRACTION * 100)}% reserve"
        )


def _checkpoint_inputs(checkpoint_root: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = checkpoint_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint root is not a directory")
    receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_files = _validate_receipt_metadata(receipt_data)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = _text_config(model_config)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint safetensors index has no weight_map")
    selected_shards = {"config.json", "model.safetensors.index.json"}
    for layer_index in range(PREFIX_LAYER_COUNT):
        prefix = f"model.layers.{layer_index}."
        for name in (
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            key = prefix + name
            shard = weight_map.get(key)
            if not isinstance(shard, str) or shard not in receipt_files:
                raise ValueError(f"checkpoint index or receipt is missing selected tensor {key}")
            selected_shards.add(shard)
    embedding_shard = weight_map.get(EMBEDDING_KEY)
    if not isinstance(embedding_shard, str) or embedding_shard not in receipt_files:
        raise ValueError("checkpoint index or receipt is missing the token embedding tensor")
    selected_shards.add(embedding_shard)
    if any(not name.endswith(".safetensors") for name in selected_shards - {"config.json", "model.safetensors.index.json"}):
        raise ValueError("checkpoint index references a non-safetensors weight shard")
    all_shards = {name for name in receipt_files if name.endswith(".safetensors")}
    if not all_shards:
        raise ValueError("checkpoint receipt has no safetensors shard records")
    source = _verify_receipt_files(
        root,
        receipt_path,
        receipt_data,
        all_shards | {"config.json", "model.safetensors.index.json"},
    )
    source["index_weight_map"] = weight_map
    source["text_config"] = text_config
    source["selected_shards"] = sorted(selected_shards - {"config.json", "model.safetensors.index.json"})
    source["index_sha256"] = source["verified_files"]["model.safetensors.index.json"]["sha256"]
    source["config_sha256"] = source["verified_files"]["config.json"]["sha256"]
    return model_config, index, source


def _load_selected_embedding_rows(
    checkpoint_root: Path,
    shard_name: str,
    token_ids: list[int],
    hidden_size: int,
    torch: Any,
    safe_open: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    unique_ids = sorted(set(token_ids))
    remapped_ids = [unique_ids.index(token_id) for token_id in token_ids]
    source_records = []
    row_bytes = bytearray()
    key = EMBEDDING_KEY
    with safe_open(str(checkpoint_root / shard_name), framework="pt", device="cpu") as checkpoint:
        try:
            tensor_slice = checkpoint.get_slice(key)
        except Exception as error:
            raise RuntimeError("could not access the token embedding tensor by selected rows") from error
        if list(tensor_slice.get_shape()) != [REAL_TEXT_GEOMETRY["vocab_size"], hidden_size]:
            raise RuntimeError("checkpoint token embedding shape does not match the pinned model")
        for token_id in unique_ids:
            row = tensor_slice[token_id]
            if row.dtype != torch.bfloat16 or tuple(row.shape) != (hidden_size,):
                raise RuntimeError("selected token embedding row is not BF16 with the pinned width")
            words = row.contiguous().view(torch.uint16).numpy()
            raw = memoryview(words).cast("B")
            row_sha256 = hashlib.sha256(raw).hexdigest()
            row_bytes.extend(raw)
            source_records.append({
                "key": key,
                "row": token_id,
                "shard": shard_name,
                "dtype": "BF16",
                "shape": [hidden_size],
                "nbytes": len(raw),
                "tensor_sha256": row_sha256,
            })
    cpu_rows = torch.frombuffer(row_bytes, dtype=torch.uint16).view(torch.bfloat16)
    cpu_rows = cpu_rows.reshape(len(unique_ids), hidden_size)
    mps_rows = cpu_rows.to(device="mps")
    mps_remapped_ids = torch.tensor(remapped_ids, device="mps", dtype=torch.int64)
    return mps_rows, mps_remapped_ids, source_records


def _load_decoder_layer(
    layer_index: int,
    layer: Any,
    expected_shapes: dict[str, tuple[int, ...]],
    checkpoint_root: Path,
    weight_map: dict[str, str],
    torch: Any,
    safe_open: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    names_by_shard: dict[str, list[tuple[str, str]]] = {}
    for name in expected_shapes:
        key = f"model.layers.{layer_index}.{name}"
        shard = weight_map.get(key)
        if not isinstance(shard, str):
            raise ValueError(f"checkpoint index is missing layer tensor {key}")
        names_by_shard.setdefault(shard, []).append((name, key))
    state_dict = {}
    provenance = []
    for shard, entries in names_by_shard.items():
        with safe_open(str(checkpoint_root / shard), framework="pt", device="cpu") as checkpoint:
            for name, key in entries:
                source_tensor = checkpoint.get_tensor(key)
                if source_tensor.dtype != torch.bfloat16 or tuple(source_tensor.shape) != expected_shapes[name]:
                    raise RuntimeError(f"checkpoint tensor dtype or shape mismatch for {key}")
                source_sha256 = _hash_bf16_tensor(source_tensor, torch)
                device_tensor = source_tensor.to(device="mps")
                if device_tensor.device.type != "mps" or device_tensor.dtype != torch.bfloat16:
                    raise RuntimeError(f"checkpoint tensor did not remain BF16 while moving to MPS: {key}")
                state_dict[name] = device_tensor
                provenance.append({
                    "key": key,
                    "shard": shard,
                    "dtype": "BF16",
                    "shape": list(expected_shapes[name]),
                    "nbytes": source_tensor.numel() * 2,
                    "tensor_sha256": source_sha256,
                })
                del source_tensor, device_tensor
    result = layer.load_state_dict(state_dict, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"official decoder layer {layer_index} state_dict did not load strictly")
    for parameter in layer.parameters():
        if parameter.device.type != "mps" or parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"decoder layer {layer_index} has a non-BF16-MPS parameter")
    layer.eval()
    del state_dict
    return layer, provenance


def _run_prefix(args: argparse.Namespace, sidecar_writer: Any) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if sys.byteorder != "little":
        raise RuntimeError("checkpoint BF16 source preparation requires a little-endian host")

    model_config_json, index, checkpoint = _checkpoint_inputs(
        args.checkpoint_root, args.checkpoint_receipt
    )
    # Checkpoint provenance keeps this root JSON-serializable, but checkpoint
    # readers need a Path for safe shard resolution.
    checkpoint_root = Path(checkpoint["root"])
    text_values = _text_config(model_config_json)
    prompt_contract = None
    prompt_file = getattr(args, "hy_nomas_prompt_file", None)
    upstream_root = getattr(args, "hy_nomas_root", None)
    if (prompt_file is None) != (upstream_root is None):
        raise ValueError("--hy-nomas-prompt-file and --hy-nomas-root must be supplied together")
    if prompt_file is not None:
        if getattr(args, "position_offset", 0) != 0:
            raise ValueError("HyNomas video prompts use the original zero position offset")
        prompt_request, prompt_contract = _load_upstream_prompt_helper().build_prompt(
            args.checkpoint_root,
            args.checkpoint_receipt,
            upstream_root,
            prompt_file,
        )
        token_ids = prompt_request["token_ids"]
        attention_mask = prompt_request["attention_mask"]
        position_offset = 0
        sequence_limit = MAX_NATURAL_SEQUENCE_LENGTH
    else:
        prompt_request = None
        token_ids = args.token_ids
        attention_mask = args.attention_mask
        position_offset = args.position_offset
        sequence_limit = MAX_SEQUENCE_LENGTH
    request = _validate_inputs(
        token_ids,
        attention_mask,
        int(text_values["vocab_size"]),
        position_offset,
        int(text_values["hidden_size"]),
        sequence_limit,
    )
    if prompt_request is not None:
        for key in (
            "position_ids_f32",
            "position_ids_for_official_model",
            "multimodal_position_shape",
            "sequence_length",
        ):
            if request[key] != prompt_request[key]:
                raise ValueError(f"HyNomas prompt request has inconsistent {key}")
    weight_map = checkpoint["index_weight_map"]
    embedding_shard = weight_map[EMBEDDING_KEY]

    import torch
    from safetensors import safe_open

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("this pretrained oracle requires MPS; CPU fallback is forbidden")
    helper = _load_upstream_helper()
    modeling, configuration, transformers_sources, transformers_version = helper._import_official(torch)
    torch_sources = helper._verify_torch_sources(torch)
    config = _parse_text_config(configuration, text_values)
    config.num_hidden_layers = SOURCE_LAYER_COUNT
    config.use_cache = False
    config._attn_implementation = "sdpa"
    # Materialize every fixture metadata field before constructing/uploading
    # the resident decoder prefix. In particular, pinned Transformers config
    # versions omit head_dim, so record the validated derived value here.
    config_record = _materialize_config_record(
        config,
        request["sequence_length"],
        max_sequence_length=sequence_limit,
    )

    with torch.device("meta"):
        shape_probe = modeling.Qwen2_5_VLDecoderLayer(config, layer_idx=0)
    expected_shapes = _expected_weight_names(shape_probe)
    mps_estimate = _estimate_mps_bytes(
        expected_shapes,
        config.hidden_size,
        config.intermediate_size,
        request["sequence_length"],
        len(set(token_ids)),
    )
    _require_memory_budget(
        mps_estimate["estimated_peak_mps_bytes"],
        int(torch.mps.current_allocated_memory()),
        int(torch.mps.recommended_max_memory()),
    )
    del shape_probe

    device = torch.device("mps")
    embedding_rows, remapped_ids, embedding_row_sources = _load_selected_embedding_rows(
        checkpoint_root,
        embedding_shard,
        token_ids,
        config.hidden_size,
        torch,
        safe_open,
    )
    layers = []
    weight_sources = []
    for layer_index in range(PREFIX_LAYER_COUNT):
        with torch.device("meta"):
            layer = modeling.Qwen2_5_VLDecoderLayer(config, layer_idx=layer_index)
        shapes = _expected_weight_names(layer)
        if shapes != expected_shapes:
            raise RuntimeError("official decoder layer parameter geometry changed within the prefix")
        layer, layer_provenance = _load_decoder_layer(
            layer_index,
            layer,
            expected_shapes,
            checkpoint_root,
            weight_map,
            torch,
            safe_open,
        )
        layers.append(layer)
        weight_sources.extend(layer_provenance)
        if (layer_index + 1) % 4 == 0 or layer_index + 1 == PREFIX_LAYER_COUNT:
            print(
                f"loaded checkpoint decoder weights {layer_index + 1}/{PREFIX_LAYER_COUNT}",
                file=sys.stderr,
                flush=True,
            )

    # No forward operation begins until all 26 official decoder modules and
    # their checkpoint tensors are already MPS-resident.
    with torch.inference_mode():
        input_hidden_states = torch.nn.functional.embedding(remapped_ids, embedding_rows).unsqueeze(0)
    if input_hidden_states.device.type != "mps" or input_hidden_states.dtype != torch.bfloat16:
        raise RuntimeError("checkpoint embedding output must be BF16 on MPS")
    del embedding_rows, remapped_ids

    with torch.device(device):
        rotary = modeling.Qwen2_5_VLRotaryEmbedding(config=config, device=device)
    rotary.eval()
    token_count = request["sequence_length"]
    position_values = [position_offset + index for index in range(token_count)]
    multimodal_positions = torch.tensor(
        [position_values, position_values, position_values], device=device, dtype=torch.int64
    ).reshape(3, 1, token_count)
    binary_mask = torch.tensor(attention_mask, device=device, dtype=torch.int64).reshape(1, -1)
    cache_position = torch.arange(token_count, device=device, dtype=torch.int64)

    with torch.inference_mode():
        cos, sin = rotary(input_hidden_states, multimodal_positions)
        causal_mask = modeling.create_causal_mask(
            config=config,
            input_embeds=input_hidden_states,
            attention_mask=binary_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=None,
        )
        if causal_mask is None:
            if any(value == 0 for value in attention_mask):
                raise RuntimeError("official causal-mask helper omitted a required padded-key mask")
            causal_mask_mode = "official implicit causal mask for all-visible input"
        else:
            if not isinstance(causal_mask, torch.Tensor):
                raise RuntimeError("official causal-mask helper returned an unsupported mask value")
            if causal_mask.device.type != "mps" or causal_mask.dtype != torch.bool:
                raise RuntimeError("official Qwen2.5-VL mask must be bool on MPS")
            expected_mask_shape = (1, 1, token_count, token_count)
            if tuple(causal_mask.shape) != expected_mask_shape:
                raise RuntimeError(f"official causal mask shape changed: {tuple(causal_mask.shape)}")
            causal_mask_mode = "official bool mask from create_causal_mask"
        hidden_states = input_hidden_states
        captured_states = [input_hidden_states]
        for layer_index, layer in enumerate(layers):
            output = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = output[0]
            if hidden_states.device.type != "mps" or hidden_states.dtype != torch.bfloat16:
                raise RuntimeError("official decoder output escaped BF16 MPS")
            captured_states.append(hidden_states)
            if (layer_index + 1) % 4 == 0 or layer_index + 1 == PREFIX_LAYER_COUNT:
                print(
                    f"computed official decoder prefix {layer_index + 1}/{PREFIX_LAYER_COUNT}",
                    file=sys.stderr,
                    flush=True,
                )
        if len(captured_states) != PREFIX_LAYER_COUNT + 1:
            raise RuntimeError("official decoder did not produce all 26 prefix outputs")
        packed_states = torch.stack(captured_states, dim=0).contiguous()
        if tuple(packed_states.shape) != (
            PREFIX_LAYER_COUNT + 1,
            1,
            token_count,
            config.hidden_size,
        ):
            raise RuntimeError("packed prefix states have an unexpected shape")
        if packed_states.device.type != "mps" or packed_states.dtype != torch.bfloat16:
            raise RuntimeError("packed prefix states must remain BF16 on MPS before readback")
        del captured_states, hidden_states, output, input_hidden_states
        # This is the sole explicit device wait. One packed activation tensor is
        # read back only after the entire 26-layer prefix has completed.
        torch.mps.synchronize()
        hidden_state_record = sidecar_writer.append(packed_states, torch)
        inv_freq = rotary.inv_freq.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1).tolist()

    if len(inv_freq) != config_record["head_dim"] // 2 or any(not math.isfinite(value) for value in inv_freq):
        raise RuntimeError("official rotary inverse frequencies are not finite F32 values")
    if len(weight_sources) != PREFIX_LAYER_COUNT * 12:
        raise RuntimeError("checkpoint prefix did not load the expected number of weight tensors")

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            key: value
            for key, value in checkpoint.items()
            if key not in {"index_weight_map", "text_config"}
        },
        "config": config_record,
        "inputs": {
            key: request[key]
            for key in (
                "token_ids",
                "attention_mask",
                "position_ids_f32",
                "position_ids_for_official_model",
                "multimodal_position_shape",
                "position_offset",
            )
        } | {"inv_freq_f32": [float(value) for value in inv_freq]},
        **({"hy_nomas_prompt": prompt_contract} if prompt_contract is not None else {}),
        "capture": {
            "source_layer_count": SOURCE_LAYER_COUNT,
            "prefix_layer_count": PREFIX_LAYER_COUNT,
            "captured_state_count": PREFIX_LAYER_COUNT + 1,
            "state_indexing": "0=token_embedding; i+1=decoder_layer_i_output",
            "hidden_states": hidden_state_record,
            "embedding_source_key": EMBEDDING_KEY,
            "embedding_source_shard": embedding_shard,
            "embedding_unique_rows": embedding_row_sources,
        },
        "weight_sources": weight_sources,
        "shard_provenance": {
            "selected_safetensors_shards": checkpoint["selected_shards"],
            "verified_safetensors_shards": sorted(
                filename for filename in checkpoint["verified_files"] if filename.endswith(".safetensors")
            ),
            "source_layer_range_inclusive": [0, PREFIX_LAYER_COUNT - 1],
            "excluded": ["vision tower weights", "decoder layers 26-27", "final norm", "language-model head"],
        },
        "provenance": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers_version,
            "torch_sources": torch_sources,
            "transformers_sources": transformers_sources,
            "producer_source_sha256": _sha256_file(Path(__file__).resolve()),
            "upstream_helper_source_sha256": _sha256_file(UPSTREAM_HELPER_PATH),
            "official_decoder_layer_class": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer",
            "official_embedding_operation": "torch.nn.functional.embedding over selected checkpoint BF16 rows on MPS",
            "official_mask_helper": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.create_causal_mask",
            "attention_dispatcher": "transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS['sdpa']",
            "causal_mask_mode": causal_mask_mode,
            "attention_implementation": "sdpa",
            "device": "mps",
            "activation_dtype": "bfloat16",
            "position_dtype": "float32 in fixture; official text model passes 3-axis int64 IDs only to rotary and None to decoder layers",
            "inv_freq_dtype": "float32",
            "weight_source_dtype": "bfloat16; no F32 expansion",
            "mps_memory_estimate": mps_estimate,
            "mps_reserved_fraction": MPS_RESERVE_FRACTION,
            "comparison_tolerance": {"rtol": RTOL, "atol": ATOL},
            "hidden_states_minus3_mapping": (
                "source-derived: after decoder layer 25 equals hidden_states[-3] for the pinned 28-layer "
                "Qwen2_5_VLTextModel.forward tuple; this fixture runs only layers 0-25"
            ),
            "full_tuple_produced": False,
            "explicit_mps_synchronizations": 1,
            "hidden_state_readback": "one packed [27,1,S,3584] BF16 tensor after the terminal synchronization",
        },
        "sidecars": {"bf16": {
            "path": sidecar_writer.path.name,
            "encoding": SIDECAR_ENCODING,
            "nbytes": hidden_state_record["storage"]["nbytes"],
            "sha256": "0" * 64,
        }},
    }


def _parse_int_list(value: str) -> list[int]:
    try:
        result = [int(token.strip()) for token in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(not token for token in value.split(",")):
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated integer list")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON manifest path")
    parser.add_argument("--token-ids", type=_parse_int_list)
    parser.add_argument(
        "--attention-mask",
        type=_parse_int_list,
        help="integer 0/1 mask matching --token-ids; visible tokens must precede right padding",
    )
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument(
        "--hy-nomas-prompt-file",
        type=Path,
        help="UTF-8 prompt text; runs the pinned HyNomas video JSON template and tokenizer",
    )
    parser.add_argument(
        "--hy-nomas-root",
        type=Path,
        help="pinned HyNomas source checkout required with --hy-nomas-prompt-file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if (args.hy_nomas_prompt_file is None) != (args.hy_nomas_root is None):
        raise SystemExit("--hy-nomas-prompt-file and --hy-nomas-root must be supplied together")
    if args.hy_nomas_prompt_file is not None and args.position_offset != 0:
        raise SystemExit("HyNomas video prompts use the original zero position offset")
    if args.hy_nomas_prompt_file is not None:
        if args.token_ids is not None or args.attention_mask is not None:
            raise SystemExit("HyNomas prompt mode does not accept --token-ids or --attention-mask")
    else:
        if args.token_ids is None:
            args.token_ids = list(DEFAULT_TOKEN_IDS)
        if args.attention_mask is None:
            args.attention_mask = list(DEFAULT_ATTENTION_MASK)
    output = args.output.expanduser()
    if output.is_symlink():
        raise SystemExit(f"refusing symlink fixture output: {output}")
    output = output.resolve()
    sidecar_path = output.with_name(f"{output.stem}.bf16le")
    if output.exists() or sidecar_path.exists() or sidecar_path.is_symlink():
        raise SystemExit("refusing to replace an existing manifest or BF16 sidecar")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    helper = _load_upstream_helper()
    writer = helper._BF16SidecarWriter(sidecar_path)
    try:
        fixture = _run_prefix(args, writer)
        sidecar = writer.finish()
        fixture["sidecars"]["bf16"] = sidecar
        _validate_manifest(fixture, output)
        serialized = json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except BaseException:
        writer.abort()
        raise
    print(json.dumps({
        "manifest": str(output),
        "sidecar": str(sidecar_path),
        "schema": SCHEMA,
        "source_layers": SOURCE_LAYER_COUNT,
        "prefix_layers": PREFIX_LAYER_COUNT,
        "captured_states": PREFIX_LAYER_COUNT + 1,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
