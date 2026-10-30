#!/usr/bin/env python3
"""Prepare authenticated, fixed first-segment Hy-Nomas input tensors on MPS.

This prepares inputs only. It does not construct a model or run inference. All
random generation, RoPE construction, source-order cropping, and layout
permutation happen on MPS; one packed terminal copy exports the tensor payload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SPEC_SUPPORT = ROOT / "spec" / "support"
if str(SPEC_SUPPORT) not in sys.path:
    sys.path.insert(0, str(SPEC_SUPPORT))

SCHEMA = "cogni_ml.hy_nomas_generation_inputs/v1"
SCHEMA_VERSION = 1
DYNAMIC_SCHEMA = "cogni_ml.hy_nomas_generation_inputs/v2"
DYNAMIC_SCHEMA_VERSION = 2
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
OUTPUT_ROOT = Path("/opt/mldata_ssd")
QWEN_REFERENCE = Path(
    "/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024/fixtures/prefix-natural-tram-v1.json"
)
QWEN_SIDECAR = QWEN_REFERENCE.with_suffix(".bf16le")
QWEN_REFERENCE_SHA256 = "eb64e5c31bc2178436b2b7dd411af45dea7fa43a8e3ff37a146bc411fac03ba1"
QWEN_SIDECAR_SHA256 = "e7ebc9a784124afc20503435d84d42e0110d7c02054d3fc63fd586ec69ed4995"
QWEN_SIDECAR_BYTES = 268_627_968
QWEN_PROMPT_SHA256 = "91246a3eaf20efb68e83f9c6fe1e91afaedd8013a77971ebd74546fe2ed1e5a9"
QWEN_REFERENCE_SCHEMA = "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference"
QWEN_REFERENCE_SCHEMA_VERSION = 1
QWEN_PREFIX_STATE_INDEX = 26
QWEN_BASE_SEQUENCE_LENGTH = 1280
QWEN_HIDDEN_SIZE = 3584
MAX_DYNAMIC_VALID_TOKENS = 1000
PROMPT = (
    "A yellow tram turns through a wet downtown intersection at dusk as pedestrians "
    "wait under umbrellas."
)

LATENT_SHAPE = (1, 32, 30, 9, 16)
QWEN_SHAPE = (1, 1280, 3584)
DIT_ROPE_GRID = (30, 9, 16)
DIT_ROPE_SHAPE = (4320, 128)
PIXEL_ROPE_GRID = (10, 18, 32)
PIXEL_ROPE_SHAPE = (1, 5760, 1, 128)
PIXEL_NOISE_SHAPE = (1, 3, 120, 288, 512)
LATENT_SEED = 42
PIXEL_NOISE_SEEDS = tuple(LATENT_SEED * 1_000_003 + index for index in range(30))

GEOMETRY = {
    "dit_latent_shape": list(LATENT_SHAPE),
    "dit_rope_grid": list(DIT_ROPE_GRID),
    "dit_rope_axis_dims": [16, 56, 56],
    "dit_rope_theta": 256.0,
    "qwen_shape": list(QWEN_SHAPE),
    "qwen_state_index": 26,
    "qwen_crop": [108, 1388],
    "qwen_valid_tokens": 23,
    "decoder_latent_grid": [30, 9, 16],
    "decoder_noise_shape": list(PIXEL_NOISE_SHAPE),
    "decoder_noise_piece_latents": 1,
    "decoder_noise_piece_frames": 4,
    "decoder_rope_grid": list(PIXEL_ROPE_GRID),
    "decoder_rope_shape": list(PIXEL_ROPE_SHAPE),
    "decoder_rope_sections": [64, 32, 32],
    "decoder_rope_theta": [1_000_000.0, 10_000.0, 10_000.0],
    "decoder_local_attention_tile_tokens": 8,
    "decoder_local_attention_tile_pixels": 128,
}
NOISE_SEEDS = {
    "latent_seed": LATENT_SEED,
    "pixel_noise_seed_formula": "seed * 1000003 + latent_frame_index",
    "pixel_noise_seeds": list(PIXEL_NOISE_SEEDS),
}
QWEN_SOURCE_RECORD = {
    "reference_path": str(QWEN_REFERENCE),
    "reference_sha256": QWEN_REFERENCE_SHA256,
    "sidecar_path": str(QWEN_SIDECAR),
    "sidecar_sha256": QWEN_SIDECAR_SHA256,
    "sidecar_bytes": QWEN_SIDECAR_BYTES,
    "sidecar_encoding": "bfloat16-le",
    "state_index": 26,
    "crop": [108, 1388],
    "crop_tokens": 1280,
    "hidden_size": 3584,
    "valid_tokens": 23,
    "prompt_sha256": QWEN_PROMPT_SHA256,
}
UPSTREAM_SOURCE_PATHS = (
    "hymm/models/dit/modules/posemb_layers.py",
    "hymm/models/vae/pixelumm_decoder/model.py",
    "hymm/models/vae/pixelumm_decoder/autoencoder.py",
)
PINNED_UPSTREAM_SOURCE_SHA256 = {
    "hymm/models/dit/modules/posemb_layers.py": "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa",
    "hymm/models/vae/pixelumm_decoder/model.py": "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec",
    "hymm/models/vae/pixelumm_decoder/autoencoder.py": "65e0d77b9f2e9a3ca457d6a51bfdb239f7b9dc3fc8a0ad014025c36d6be9f0cf",
}
PRODUCER_RECORD = {
    "device": "mps",
    "mps_fallback": "disabled",
    "model_inference": False,
    "tensor_generation_device": "mps",
    "host_export": "one packed terminal MPS-to-host readback",
}
TENSOR_SPECS = (
    ("input.latent", "F32", LATENT_SHAPE),
    ("input.qwen_bf16", "BF16", QWEN_SHAPE),
    ("input.rope_cos", "F32", DIT_ROPE_SHAPE),
    ("input.rope_sin", "F32", DIT_ROPE_SHAPE),
    ("decoder.noise", "F32", PIXEL_NOISE_SHAPE),
    ("decoder.rope_cos", "F32", PIXEL_ROPE_SHAPE),
    ("decoder.rope_sin", "F32", PIXEL_ROPE_SHAPE),
)
_TENSOR_WIDTH = {"F32": 4, "BF16": 2}
_JSON_KEYS = {
    "schema", "schema_version", "payload", "payload_bytes", "payload_sha256",
    "tensors", "geometry", "prompt", "qwen_source", "noise_seeds",
    "upstream_revision", "upstream_sources", "producer",
}
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DYNAMIC_QWEN_SOURCE_KEYS = {
    "reference_schema", "reference_schema_version", "reference_path", "reference_sha256",
    "sidecar_path", "sidecar_sha256", "sidecar_bytes", "sidecar_encoding",
    "state_index", "crop", "crop_tokens", "raw_sequence_length", "hidden_size",
    "valid_tokens", "cropped_attention_mask", "prompt_sha256", "prompt_contract",
    "upstream_revision", "template_name", "template_sha256", "source_code_hashes",
}
_DYNAMIC_SOURCE_CODE_HASH_KEYS = {
    "oracle_producer", "prompt_helper", "upstream_encoder", "upstream_constants",
}
_PROMPT_REFERENCE_HELPER: Any = None


class InputsError(RuntimeError):
    """An input, provenance, output-admission, or MPS preparation failure."""


def tensor_nbytes(dtype: str, shape: tuple[int, ...] | list[int]) -> int:
    if dtype not in _TENSOR_WIDTH:
        raise InputsError(f"unsupported tensor dtype: {dtype}")
    if (not isinstance(shape, (tuple, list)) or not shape
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise InputsError("tensor shape must contain positive integers")
    return math.prod(shape) * _TENSOR_WIDTH[dtype]


def expected_payload_bytes() -> int:
    return sum(tensor_nbytes(dtype, shape) for _, dtype, shape in TENSOR_SPECS)


def _same_json_types(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return (type(actual) is dict and set(actual) == set(expected)
                and all(_same_json_types(value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (type(actual) is list and len(actual) == len(expected)
                and all(_same_json_types(left, right) for left, right in zip(expected, actual)))
    return type(actual) is type(expected) and actual == expected


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _HEX_DIGEST.fullmatch(value) is not None


def _prompt_reference_helper() -> Any:
    global _PROMPT_REFERENCE_HELPER
    if _PROMPT_REFERENCE_HELPER is None:
        helper_path = ROOT / "examples" / "support" / "qwen25_hy_nomas_prompt_reference.py"
        spec = importlib.util.spec_from_file_location(
            "qwen25_hy_nomas_prompt_reference_for_generation_inputs", helper_path
        )
        if spec is None or spec.loader is None:
            raise InputsError("cannot load the pinned Qwen prompt contract validator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PROMPT_REFERENCE_HELPER = module
    return _PROMPT_REFERENCE_HELPER


def _validate_dynamic_qwen_source(source: Any, prompt: Any) -> int:
    if type(source) is not dict or set(source) != _DYNAMIC_QWEN_SOURCE_KEYS:
        raise InputsError("dynamic Qwen source record fields are malformed")
    helper = _prompt_reference_helper()
    if (source.get("reference_schema") != QWEN_REFERENCE_SCHEMA
            or type(source.get("reference_schema_version")) is not int
            or source["reference_schema_version"] != QWEN_REFERENCE_SCHEMA_VERSION):
        raise InputsError("dynamic Qwen reference schema mismatch")
    for key in ("reference_path", "sidecar_path"):
        value = source.get(key)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise InputsError(f"dynamic Qwen {key} must be an absolute path")
    for key in ("reference_sha256", "sidecar_sha256", "prompt_sha256"):
        if not _valid_digest(source.get(key)):
            raise InputsError(f"dynamic Qwen {key} is malformed")
    if source.get("sidecar_encoding") != "bfloat16-le":
        raise InputsError("dynamic Qwen sidecar encoding mismatch")
    sidecar_bytes = source.get("sidecar_bytes")
    if (type(sidecar_bytes) is not int or sidecar_bytes <= 0
            or sidecar_bytes > 512 * 1024 * 1024 or sidecar_bytes % 2):
        raise InputsError("dynamic Qwen sidecar byte count is invalid")
    contract = source.get("prompt_contract")
    if type(contract) is not dict or contract.get("prompt") != prompt or type(prompt) is not str:
        raise InputsError("dynamic Qwen prompt contract does not bind the bundle prompt")
    expected_prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if (source["prompt_sha256"] != expected_prompt_sha
            or contract.get("prompt_sha256") != expected_prompt_sha):
        raise InputsError("dynamic Qwen prompt digest does not match the bundle prompt")
    if (contract.get("schema") != helper.PROMPT_SCHEMA
            or type(contract.get("schema_version")) is not int
            or contract["schema_version"] != helper.PROMPT_SCHEMA_VERSION):
        raise InputsError("dynamic Qwen prompt contract schema mismatch")
    if (source.get("upstream_revision") != helper.UPSTREAM_REVISION
            or contract.get("upstream_revision") != helper.UPSTREAM_REVISION):
        raise InputsError("dynamic Qwen upstream revision mismatch")
    if (source.get("template_name") != helper.TEMPLATE_NAME
            or contract.get("template_name") != helper.TEMPLATE_NAME
            or source.get("template_sha256") != helper.VIDEO_JSON_TEMPLATE_SHA256
            or contract.get("template_sha256") != helper.VIDEO_JSON_TEMPLATE_SHA256):
        raise InputsError("dynamic Qwen template identity mismatch")
    if source.get("state_index") != QWEN_PREFIX_STATE_INDEX:
        raise InputsError("dynamic Qwen state index mismatch")
    if source.get("crop_tokens") != QWEN_BASE_SEQUENCE_LENGTH:
        raise InputsError("dynamic Qwen crop width mismatch")
    if source.get("hidden_size") != QWEN_HIDDEN_SIZE:
        raise InputsError("dynamic Qwen hidden size mismatch")

    crop_start = contract.get("crop_start")
    raw_length = contract.get("raw_length")
    if (type(crop_start) is not int or not 0 < crop_start <= helper.MAX_CROP_START
            or type(raw_length) is not int
            or raw_length != crop_start + QWEN_BASE_SEQUENCE_LENGTH
            or source.get("raw_sequence_length") != raw_length
            or source.get("crop") != [crop_start, raw_length]):
        raise InputsError("dynamic Qwen crop geometry does not match its prompt contract")
    if sidecar_bytes != (QWEN_PREFIX_STATE_INDEX + 1) * raw_length * QWEN_HIDDEN_SIZE * 2:
        raise InputsError("dynamic Qwen sidecar size does not match its hidden-state geometry")

    mask = source.get("cropped_attention_mask")
    if (not isinstance(mask, list) or len(mask) != QWEN_BASE_SEQUENCE_LENGTH
            or any(type(value) is not int or value not in (0, 1) for value in mask)):
        raise InputsError("dynamic Qwen cropped mask must contain 1280 integer values")
    first_padding = next((index for index, value in enumerate(mask) if value == 0), len(mask))
    if mask[0] != 1 or any(mask[first_padding:]):
        raise InputsError("dynamic Qwen cropped mask must be a visible prefix followed by padding")
    valid_tokens = sum(mask)
    if (not 1 <= valid_tokens <= MAX_DYNAMIC_VALID_TOKENS
            or source.get("valid_tokens") != valid_tokens):
        raise InputsError("dynamic Qwen valid-token count is outside the admitted range")
    if contract.get("cropped_attention_mask") != mask:
        raise InputsError("dynamic Qwen mask differs from its prompt contract")
    if "untruncated_preflight" not in contract:
        raise InputsError("dynamic Qwen prompt has no untruncated tokenization preflight")
    token_ids = contract.get("token_ids")
    attention_mask = contract.get("attention_mask")
    if not isinstance(token_ids, list) or not isinstance(attention_mask, list):
        raise InputsError("dynamic Qwen prompt token arrays are missing")
    positions = [list(range(raw_length)) for _ in range(3)]
    positions_f32 = [[float(value) for value in axis] for axis in positions]
    try:
        helper.validate_prompt_metadata(
            contract,
            token_ids,
            attention_mask,
            positions_f32,
            positions,
            [3, 1, raw_length],
            0,
        )
    except Exception as error:
        raise InputsError(f"dynamic Qwen prompt contract rejected: {error}") from error

    code_hashes = source.get("source_code_hashes")
    upstream = contract.get("upstream_source")
    if (type(code_hashes) is not dict or set(code_hashes) != _DYNAMIC_SOURCE_CODE_HASH_KEYS
            or type(upstream) is not dict):
        raise InputsError("dynamic Qwen source-code hash inventory is malformed")
    expected_hashes = {
        "oracle_producer": _hash_file(
            ROOT / "examples" / "support" / "qwen25_pretrained_reference.py"
        ),
        "prompt_helper": contract.get("producer_source_sha256"),
        "upstream_encoder": upstream.get("encoder_source", {}).get("sha256")
        if type(upstream.get("encoder_source")) is dict else None,
        "upstream_constants": upstream.get("constants_source", {}).get("sha256")
        if type(upstream.get("constants_source")) is dict else None,
    }
    for key, value in expected_hashes.items():
        if not _valid_digest(value) or code_hashes.get(key) != value:
            raise InputsError(f"dynamic Qwen source-code hash binding mismatch: {key}")
    if code_hashes["prompt_helper"] != _hash_file(
        ROOT / "examples" / "support" / "qwen25_hy_nomas_prompt_reference.py"
    ):
        raise InputsError("dynamic Qwen prompt-helper digest differs from the admitted implementation")
    if (code_hashes["upstream_encoder"] != helper.UPSTREAM_ENCODER_SHA256
            or code_hashes["upstream_constants"] != helper.UPSTREAM_CONSTANTS_SHA256):
        raise InputsError("dynamic Qwen upstream source digests differ from pinned sources")
    return valid_tokens


def _validate_basename(value: Any, suffix: str) -> bool:
    return (isinstance(value, str) and value not in ("", ".", "..")
            and Path(value).name == value and value.endswith(suffix))


def validate_document(document: Mapping[str, Any], *, payload_bytes: int | None = None) -> None:
    """Validate fixed geometry and exact contiguous tensor inventory without torch."""
    if type(document) is not dict or set(document) != _JSON_KEYS:
        raise InputsError("generation-input document keys are malformed")
    schema = document.get("schema")
    version = document.get("schema_version")
    legacy = schema == SCHEMA and type(version) is int and version == SCHEMA_VERSION
    dynamic = schema == DYNAMIC_SCHEMA and type(version) is int and version == DYNAMIC_SCHEMA_VERSION
    if not legacy and not dynamic:
        raise InputsError("generation-input schema mismatch")
    if not _validate_basename(document.get("payload"), ".tensors.bin"):
        raise InputsError("payload must be a sibling .tensors.bin basename")
    if not _valid_digest(document.get("payload_sha256")):
        raise InputsError("payload SHA-256 is malformed")
    if legacy:
        if not _same_json_types(GEOMETRY, document.get("geometry")):
            raise InputsError("fixed generation-input geometry mismatch")
        if document.get("prompt") != PROMPT or type(document.get("prompt")) is not str:
            raise InputsError("fixed preview prompt mismatch")
        if not _same_json_types(QWEN_SOURCE_RECORD, document.get("qwen_source")):
            raise InputsError("authenticated Qwen source record mismatch")
    else:
        if type(document.get("prompt")) is not str:
            raise InputsError("dynamic preview prompt must be text")
        valid_tokens = _validate_dynamic_qwen_source(
            document.get("qwen_source"), document["prompt"]
        )
        expected_geometry = dict(GEOMETRY)
        expected_geometry["qwen_valid_tokens"] = valid_tokens
        expected_geometry["qwen_crop"] = document["qwen_source"]["crop"]
        if not _same_json_types(expected_geometry, document.get("geometry")):
            raise InputsError("dynamic generation-input geometry does not match the Qwen mask")
    if not _same_json_types(NOISE_SEEDS, document.get("noise_seeds")):
        raise InputsError("fixed noise seed inventory mismatch")
    if document.get("upstream_revision") != PINNED_REVISION:
        raise InputsError("pinned upstream revision mismatch")
    sources = document.get("upstream_sources")
    if not _same_json_types(PINNED_UPSTREAM_SOURCE_SHA256, sources):
        raise InputsError("pinned upstream source digest inventory mismatch")
    producer = document.get("producer")
    producer_keys = set(PRODUCER_RECORD) | {"torch_version", "python_version"}
    if (type(producer) is not dict or set(producer) != producer_keys
            or any(producer.get(key) != value for key, value in PRODUCER_RECORD.items())
            or any(type(producer.get(key)) is not str or not producer[key]
                   for key in ("torch_version", "python_version"))):
        raise InputsError("input producer scope/runtime record mismatch")

    records = document.get("tensors")
    expected_names = [name for name, _, _ in TENSOR_SPECS]
    # JSON object order is not the packed tensor order (publication sorts keys).
    # The source-order offset checks below establish the exact binary layout.
    if type(records) is not dict or set(records) != set(expected_names):
        raise InputsError("tensor inventory is incomplete")
    cursor = 0
    for name, dtype, shape in TENSOR_SPECS:
        record = records[name]
        expected_nbytes = tensor_nbytes(dtype, shape)
        if (type(record) is not dict or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}
                or record.get("dtype") != dtype
                or not _same_json_types(list(shape), record.get("shape"))
                or type(record.get("offset")) is not int or record["offset"] != cursor
                or type(record.get("nbytes")) is not int or record["nbytes"] != expected_nbytes
                or not _valid_digest(record.get("sha256"))):
            raise InputsError(f"tensor record mismatch: {name}")
        cursor += expected_nbytes
    if type(document.get("payload_bytes")) is not int or document["payload_bytes"] != cursor:
        raise InputsError("payload byte count does not match the tensor inventory")
    if payload_bytes is not None and (type(payload_bytes) is not int or payload_bytes != cursor):
        raise InputsError("payload byte count mismatch")


def validate_payload(document: Mapping[str, Any], payload: bytes | bytearray | memoryview) -> None:
    """Check the packed sidecar and each record digest on CPU after terminal export."""
    validate_document(document, payload_bytes=len(payload))
    view = memoryview(payload)
    if hashlib.sha256(view).hexdigest() != document["payload_sha256"]:
        raise InputsError("aggregate payload digest mismatch")
    for name, _, _ in TENSOR_SPECS:
        record = document["tensors"][name]
        start = record["offset"]
        stop = start + record["nbytes"]
        if hashlib.sha256(view[start:stop]).hexdigest() != record["sha256"]:
            raise InputsError(f"tensor payload digest mismatch: {name}")


def validate_output_paths(output_arg: Path | str, *, allowed_root: Path = OUTPUT_ROOT) -> tuple[Path, Path]:
    """Admit a new JSON/payload pair beneath an existing output root."""
    output = Path(output_arg).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.suffix.lower() != ".json":
        raise InputsError("--output must have a .json suffix")
    try:
        root = allowed_root.resolve(strict=True)
        canonical_parent = output.parent.resolve(strict=True)
    except OSError:
        raise InputsError("output root and parent must already exist") from None
    if output.parent != canonical_parent or (canonical_parent != root and root not in canonical_parent.parents):
        raise InputsError("output parent must be canonical and beneath /opt/mldata_ssd")
    manifest = canonical_parent / output.name
    payload = canonical_parent / f"{output.stem}.tensors.bin"
    if os.path.lexists(manifest) or os.path.lexists(payload):
        raise InputsError("output manifest or sibling payload already exists")
    return manifest, payload


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_qwen_prefix_reference() -> Any:
    helper_path = ROOT / "examples" / "support" / "qwen25_pretrained_reference.py"
    spec = importlib.util.spec_from_file_location(
        "qwen25_pretrained_reference_for_generation_inputs", helper_path
    )
    if spec is None or spec.loader is None:
        raise InputsError("cannot load the authenticated Qwen prefix verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qwen_source(q4_reference: Any, text_reference: Any) -> tuple[dict[str, Any], Path, tuple]:
    try:
        document, sidecar, reference_sha, sidecar_sha, reference_identity, sidecar_identity = (
            q4_reference._validate_qwen_source(QWEN_REFERENCE, QWEN_SIDECAR)
        )
    except Exception as error:
        raise InputsError(f"authenticated Qwen source rejected: {error}") from error
    prompt = document.get("hy_nomas_prompt")
    if (type(prompt) is not dict or prompt.get("prompt") != PROMPT
            or hashlib.sha256(PROMPT.encode("utf-8")).hexdigest() != QWEN_PROMPT_SHA256):
        raise InputsError("Qwen reference prompt does not match the fixed preview prompt")
    try:
        valid_indices = text_reference.validate_qwen_reference(
            document,
            reference_sha256=reference_sha,
            sidecar_bytes=sidecar.stat().st_size,
            sidecar_sha256=sidecar_sha,
        )
    except Exception as error:
        raise InputsError(f"Qwen crop/mask admission failed: {error}") from error
    if valid_indices != list(range(23)):
        raise InputsError("Qwen crop must contain the authenticated 23-row valid prefix")
    if (reference_sha != QWEN_REFERENCE_SHA256 or sidecar_sha != QWEN_SIDECAR_SHA256
            or sidecar.stat().st_size != QWEN_SIDECAR_BYTES):
        raise InputsError("Qwen source identity changed after authentication")
    return document, sidecar, (reference_identity, sidecar_identity)


def _qwen_source_from_reference(
    q4_reference: Any,
    reference_arg: Path | str,
    prompt_file: Path | str,
) -> tuple[dict[str, Any], Path, tuple, dict[str, Any]]:
    reference_path = Path(reference_arg).expanduser()
    try:
        reference_path = q4_reference._regular_file(reference_path, "Qwen reference manifest")
        reference_identity = q4_reference._identity(reference_path)
        if reference_identity[2] <= 0 or reference_identity[2] > q4_reference.JSON_LIMIT:
            raise InputsError("Qwen reference manifest exceeds its size bound")
        reference_bytes = reference_path.read_bytes()
        if q4_reference._identity(reference_path) != reference_identity:
            raise InputsError("Qwen reference changed while reading its manifest")
        reference_digest = hashlib.sha256(reference_bytes).hexdigest()
        document = json.loads(reference_bytes.decode("utf-8"))
        qwen_reference = _load_qwen_prefix_reference()
        qwen_reference._validate_manifest(document)
        descriptor = document["sidecars"]["bf16"]
        sidecar, verified_sidecar = qwen_reference._verify_fixture_sidecar(
            descriptor, reference_path
        )
        sidecar_identity = q4_reference._identity(sidecar)
        prompt_helper = _prompt_reference_helper()
        supplied_prompt, supplied_prompt_bytes = prompt_helper._read_prompt_file(
            Path(prompt_file)
        )
    except InputsError:
        raise
    except Exception as error:
        raise InputsError(f"authenticated Qwen reference rejected: {error}") from error

    contract = document.get("hy_nomas_prompt")
    if type(contract) is not dict:
        raise InputsError("Qwen reference does not contain a natural-prompt contract")
    if contract.get("prompt") != supplied_prompt:
        raise InputsError("prompt file differs from the authenticated Qwen reference")
    prompt_digest = hashlib.sha256(supplied_prompt_bytes).hexdigest()
    if contract.get("prompt_sha256") != prompt_digest:
        raise InputsError("Qwen reference prompt digest does not match the prompt file")
    if "untruncated_preflight" not in contract:
        raise InputsError("Qwen reference was produced without an untruncated prompt preflight")

    try:
        input_data = document["inputs"]
        prompt_helper.validate_prompt_metadata(
            contract,
            input_data["token_ids"],
            input_data["attention_mask"],
            input_data["position_ids_f32"],
            input_data["position_ids_for_official_model"],
            input_data["multimodal_position_shape"],
            input_data["position_offset"],
        )
    except Exception as error:
        raise InputsError(f"Qwen prompt contract rejected: {error}") from error

    mask = contract.get("cropped_attention_mask")
    if not isinstance(mask, list) or len(mask) != QWEN_BASE_SEQUENCE_LENGTH:
        raise InputsError("Qwen reference crop mask must have exactly 1280 positions")
    valid_tokens = sum(value for value in mask if type(value) is int and value in (0, 1))
    if not 1 <= valid_tokens <= MAX_DYNAMIC_VALID_TOKENS or any(
        type(value) is not int or value not in (0, 1) for value in mask
    ):
        raise InputsError("Qwen prompt valid-token count must be between 1 and 1000")

    capture = document.get("capture")
    hidden_states = capture.get("hidden_states") if type(capture) is dict else None
    shape = hidden_states.get("shape") if type(hidden_states) is dict else None
    raw_length = contract.get("raw_length")
    crop_start = contract.get("crop_start")
    if (not isinstance(shape, list) or len(shape) != 4
            or shape != [QWEN_PREFIX_STATE_INDEX + 1, 1, raw_length, QWEN_HIDDEN_SIZE]
            or type(crop_start) is not int or shape[2] != crop_start + QWEN_BASE_SEQUENCE_LENGTH):
        raise InputsError("Qwen reference hidden-state geometry does not match its prompt crop")

    provenance = document.get("provenance")
    upstream = contract.get("upstream_source")
    try:
        source_code_hashes = {
            "oracle_producer": provenance["producer_source_sha256"],
            "prompt_helper": contract["producer_source_sha256"],
            "upstream_encoder": upstream["encoder_source"]["sha256"],
            "upstream_constants": upstream["constants_source"]["sha256"],
        }
    except (TypeError, KeyError):
        raise InputsError("Qwen reference source-code hashes are incomplete") from None
    if any(not _valid_digest(value) for value in source_code_hashes.values()):
        raise InputsError("Qwen reference source-code hash is malformed")

    source_record = {
        "reference_schema": document.get("schema"),
        "reference_schema_version": document.get("schema_version"),
        "reference_path": str(reference_path),
        "reference_sha256": reference_digest,
        "sidecar_path": str(sidecar.resolve(strict=True)),
        "sidecar_sha256": verified_sidecar["sha256"],
        "sidecar_bytes": verified_sidecar["nbytes"],
        "sidecar_encoding": "bfloat16-le",
        "state_index": QWEN_PREFIX_STATE_INDEX,
        "crop": [crop_start, crop_start + QWEN_BASE_SEQUENCE_LENGTH],
        "crop_tokens": QWEN_BASE_SEQUENCE_LENGTH,
        "raw_sequence_length": raw_length,
        "hidden_size": QWEN_HIDDEN_SIZE,
        "valid_tokens": valid_tokens,
        "cropped_attention_mask": list(mask),
        "prompt_sha256": prompt_digest,
        "prompt_contract": contract,
        "upstream_revision": contract.get("upstream_revision"),
        "template_name": contract.get("template_name"),
        "template_sha256": contract.get("template_sha256"),
        "source_code_hashes": source_code_hashes,
    }
    _validate_dynamic_qwen_source(source_record, supplied_prompt)
    if (q4_reference._identity(reference_path) != reference_identity
            or q4_reference._identity(sidecar) != sidecar_identity):
        raise InputsError("Qwen sidecar changed while the reference was admitted")
    return document, sidecar, (reference_identity, sidecar_identity), source_record


def _verify_autoencoder_source(upstream: Path, pixel_reference: Any) -> tuple[Any, dict[str, str]]:
    try:
        expected_revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if expected_revision != PINNED_REVISION:
            raise InputsError("PixelUMM input helper revision mismatch")
        relative_model = "hymm/models/vae/pixelumm_decoder/model.py"
        relative_autoencoder = "hymm/models/vae/pixelumm_decoder/autoencoder.py"
        for relative in (relative_model, relative_autoencoder):
            status = subprocess.check_output(
                ["git", "-C", str(upstream), "status", "--porcelain", "--", relative],
                text=True, stderr=subprocess.DEVNULL,
            )
            if status.strip():
                raise InputsError(f"pinned PixelUMM source has local changes: {relative}")
            if _hash_file(upstream / relative) != PINNED_UPSTREAM_SOURCE_SHA256[relative]:
                raise InputsError(f"pinned PixelUMM source digest mismatch: {relative}")

        # Import only after both executable source files have been authenticated.
        source_module, model_sha = pixel_reference.load_upstream(upstream, verify_revision=True)
        if model_sha != PINNED_UPSTREAM_SOURCE_SHA256[relative_model]:
            raise InputsError("PixelUMM model.py digest mismatch")
        autoencoder_path = upstream / relative_autoencoder
        autoencoder_sha = PINNED_UPSTREAM_SOURCE_SHA256[relative_autoencoder]
        autoencoder_module = importlib.import_module(
            "hymm.models.vae.pixelumm_decoder.autoencoder"
        )
        if Path(autoencoder_module.__file__).resolve() != autoencoder_path.resolve():
            raise InputsError("Python imported a different PixelUMM autoencoder.py")
        return (source_module, autoencoder_module), {
            relative_model: model_sha,
            relative_autoencoder: autoencoder_sha,
        }
    except InputsError:
        raise
    except Exception as error:
        raise InputsError(f"pinned PixelUMM input helpers rejected: {error}") from error


def _read_qwen_crop_on_mps(
    torch: Any,
    np: Any,
    sidecar: Path,
    device: Any,
    *,
    raw_sequence_length: int = 1388,
    crop_start: int = 108,
) -> Any:
    if (type(raw_sequence_length) is not int or type(crop_start) is not int
            or not 108 <= crop_start <= 256
            or raw_sequence_length != crop_start + QWEN_BASE_SEQUENCE_LENGTH):
        raise InputsError("Qwen source crop does not match the pinned 1280-token suffix")
    # Host use is limited to authenticated source-byte transport. State selection
    # and the requested crop itself are performed after moving state 26 to MPS.
    mapping = np.memmap(
        sidecar,
        mode="r",
        dtype="<u2",
        shape=(QWEN_PREFIX_STATE_INDEX + 1, 1, raw_sequence_length, QWEN_HIDDEN_SIZE),
    )
    state26_raw = np.array(mapping[26], copy=True)
    del mapping
    state26_cpu = torch.from_numpy(state26_raw).view(torch.bfloat16)
    del state26_raw
    state26 = state26_cpu.to(device=device, dtype=torch.bfloat16)
    del state26_cpu
    crop = state26[:, crop_start:raw_sequence_length, :].contiguous()
    del state26
    if crop.device.type != "mps" or crop.dtype is not torch.bfloat16 or tuple(crop.shape) != QWEN_SHAPE:
        raise InputsError("Qwen crop did not preserve MPS BF16 source layout")
    return crop


def _generate_tensors(torch: Any, helpers: tuple, pixel_modules: tuple, device: Any) -> dict[str, Any]:
    posemb_layers = helpers[-1]
    source_model, autoencoder = pixel_modules
    with torch.inference_mode():
        latent_generator = torch.Generator(device=device)
        latent_generator.manual_seed(LATENT_SEED)
        latent = torch.randn(
            LATENT_SHAPE, generator=latent_generator, device=device, dtype=torch.float32
        )
        rope_cos, rope_sin = posemb_layers.get_nd_rotary_pos_embed(
            [16, 56, 56], DIT_ROPE_GRID, theta=256.0, use_real=True,
            theta_rescale_factor=1, device=device,
        )

        # Call the exact pinned upstream method without constructing a decoder,
        # checkpoint, or model. _make_noise reads only self.seed and latent shape.
        pixel_noise = autoencoder.PixelUMMDCAELatentDecoder._make_noise(
            SimpleNamespace(seed=LATENT_SEED), latent, 0, LATENT_SHAPE[2]
        )

        pixel_cos, pixel_sin = source_model.mrope_tables(
            source_model.QWEN3_8B_GEN, PIXEL_ROPE_GRID, device
        )
        layout = source_model.local_attention_layout(
            PIXEL_ROPE_GRID, 8, device
        )
        if layout is None:
            raise InputsError("pinned local_attention_layout returned no tile permutation")
        permutation, inverse, block_mask = layout
        pixel_cos = pixel_cos[:, permutation]
        pixel_sin = pixel_sin[:, permutation]
        del inverse, block_mask, layout, permutation

    tensors = {
        "input.latent": latent,
        "input.rope_cos": rope_cos,
        "input.rope_sin": rope_sin,
        "decoder.noise": pixel_noise,
        "decoder.rope_cos": pixel_cos,
        "decoder.rope_sin": pixel_sin,
    }
    expected = {name: (dtype, shape) for name, dtype, shape in TENSOR_SPECS}
    for name, tensor in tensors.items():
        dtype, shape = expected[name]
        wanted_dtype = torch.float32 if dtype == "F32" else torch.bfloat16
        if (tensor.device.type != "mps" or tensor.dtype is not wanted_dtype
                or tuple(tensor.shape) != shape):
            raise InputsError(f"generated MPS tensor shape/dtype mismatch: {name}")
    return tensors


def _pack_mps_tensors(torch: Any, tensors: Mapping[str, Any]) -> bytes:
    expected_names = [name for name, _, _ in TENSOR_SPECS]
    if list(tensors) != expected_names:
        raise InputsError("MPS tensor mapping must match canonical inventory order")
    byte_views = []
    for name, dtype, shape in TENSOR_SPECS:
        tensor = tensors[name]
        expected_dtype = torch.float32 if dtype == "F32" else torch.bfloat16
        if (tensor.device.type != "mps" or tensor.dtype is not expected_dtype
                or tuple(tensor.shape) != shape):
            raise InputsError(f"cannot export non-MPS or malformed tensor: {name}")
        contiguous = tensor.contiguous()
        byte_views.append(contiguous.view(torch.uint8).reshape(-1))
    packed_mps = torch.cat(byte_views, dim=0)
    if packed_mps.device.type != "mps" or packed_mps.dtype is not torch.uint8:
        raise InputsError("packed tensor payload must remain an MPS byte tensor")
    # This is the sole MPS-to-host transfer and terminal synchronization.
    packed_cpu = packed_mps.cpu()
    return packed_cpu.numpy().tobytes()


def _records_from_payload(payload: bytes) -> dict[str, dict[str, Any]]:
    if len(payload) != expected_payload_bytes():
        raise InputsError("packed payload extent differs from canonical inventory")
    view = memoryview(payload)
    records: dict[str, dict[str, Any]] = {}
    cursor = 0
    for name, dtype, shape in TENSOR_SPECS:
        nbytes = tensor_nbytes(dtype, shape)
        records[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "offset": cursor,
            "nbytes": nbytes,
            "sha256": hashlib.sha256(view[cursor:cursor + nbytes]).hexdigest(),
        }
        cursor += nbytes
    return records


def _build_document(payload: bytes, payload_name: str, source_hashes: Mapping[str, str],
                    torch: Any, *, dynamic_qwen_source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    records = _records_from_payload(payload)
    dynamic = dynamic_qwen_source is not None
    geometry = dict(GEOMETRY)
    if dynamic:
        crop = dynamic_qwen_source["crop"]
        geometry["qwen_valid_tokens"] = dynamic_qwen_source["valid_tokens"]
        geometry["qwen_crop"] = crop
    document = {
        "schema": DYNAMIC_SCHEMA if dynamic else SCHEMA,
        "schema_version": DYNAMIC_SCHEMA_VERSION if dynamic else SCHEMA_VERSION,
        "payload": payload_name,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "tensors": records,
        "geometry": geometry,
        "prompt": dynamic_qwen_source["prompt_contract"]["prompt"] if dynamic else PROMPT,
        "qwen_source": dict(dynamic_qwen_source) if dynamic else QWEN_SOURCE_RECORD,
        "noise_seeds": NOISE_SEEDS,
        "upstream_revision": PINNED_REVISION,
        "upstream_sources": dict(source_hashes),
        "producer": {
            **PRODUCER_RECORD,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
        },
    }
    validate_document(document, payload_bytes=len(payload))
    validate_payload(document, payload)
    return document


def _publish_noreplace(path: Path, data: bytes) -> None:
    """Publish one file using a fsynced temporary and atomic no-replace link."""
    if os.path.lexists(path):
        raise InputsError(f"output already exists: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_bundle(output: Path, payload_path: Path, document: dict[str, Any], payload: bytes) -> None:
    if os.path.lexists(output) or os.path.lexists(payload_path):
        raise InputsError("output manifest or sibling payload already exists")
    published_identity = None
    try:
        _publish_noreplace(payload_path, payload)
        stat = payload_path.stat()
        published_identity = (stat.st_dev, stat.st_ino)
        encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        _publish_noreplace(output, encoded)
    except Exception:
        if published_identity is not None and os.path.lexists(payload_path):
            stat = payload_path.stat()
            if (stat.st_dev, stat.st_ino) == published_identity:
                payload_path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new manifest path beneath /opt/mldata_ssd")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM,
                        help=f"pinned source checkout at {PINNED_REVISION}")
    parser.add_argument(
        "--qwen-reference",
        type=Path,
        help="authenticated dynamic Qwen prefix manifest (requires --prompt-file)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="the exact UTF-8 prompt used to produce --qwen-reference",
    )
    return parser


def produce_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise InputsError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    output, payload_path = validate_output_paths(args.output)

    import numpy as np
    import torch

    try:
        import hy_nomas_q4_dit_reference as q4_reference
        import hy_nomas_text_conditioning_reference as text_reference
        import hy_nomas_pixelumm_reference as pixel_reference
        import hy_nomas_reference as dit_reference
    except Exception as error:
        raise InputsError(f"cannot import authenticated input helpers: {error}") from error

    reference_arg = getattr(args, "qwen_reference", None)
    prompt_file = getattr(args, "prompt_file", None)
    if (reference_arg is None) != (prompt_file is None):
        raise InputsError("--qwen-reference and --prompt-file must be supplied together")
    dynamic_qwen_source = None
    if reference_arg is None:
        qwen_document, qwen_sidecar, identities = _qwen_source(q4_reference, text_reference)
        raw_sequence_length = 1388
        crop_start = 108
    else:
        qwen_document, qwen_sidecar, identities, dynamic_qwen_source = (
            _qwen_source_from_reference(q4_reference, reference_arg, prompt_file)
        )
        raw_sequence_length = dynamic_qwen_source["raw_sequence_length"]
        crop_start = dynamic_qwen_source["crop"][0]
    upstream = Path(args.upstream).expanduser().resolve(strict=True)
    try:
        source_hashes = dit_reference._verify_upstream(upstream)
        helpers = dit_reference._install_upstream_modules(upstream)
        pixel_modules, pixel_hashes = _verify_autoencoder_source(upstream, pixel_reference)
    except InputsError:
        raise
    except Exception as error:
        raise InputsError(f"pinned upstream source admission failed: {error}") from error
    source_hashes = {
        "hymm/models/dit/modules/posemb_layers.py": source_hashes[
            "hymm/models/dit/modules/posemb_layers.py"
        ],
        **pixel_hashes,
    }
    if source_hashes != PINNED_UPSTREAM_SOURCE_SHA256:
        raise InputsError("verified upstream source inventory differs from the pinned input sources")

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise InputsError("MPS is required; CPU tensor fallback is forbidden")
    device = torch.device("mps")

    qwen = _read_qwen_crop_on_mps(
        torch, np, qwen_sidecar, device,
        raw_sequence_length=raw_sequence_length,
        crop_start=crop_start,
    )
    if q4_reference._identity(qwen_sidecar) != identities[1]:
        raise InputsError("authenticated Qwen BF16 sidecar changed during MPS transfer")
    tensors = _generate_tensors(torch, helpers, pixel_modules, device)
    tensors["input.qwen_bf16"] = qwen
    ordered_tensors = {name: tensors[name] for name, _, _ in TENSOR_SPECS}
    payload = _pack_mps_tensors(torch, ordered_tensors)
    reference_path = QWEN_REFERENCE if dynamic_qwen_source is None else Path(
        dynamic_qwen_source["reference_path"]
    )
    if q4_reference._identity(reference_path) != identities[0]:
        raise InputsError("authenticated Qwen reference changed during input preparation")
    document = _build_document(
        payload,
        payload_path.name,
        source_hashes,
        torch,
        dynamic_qwen_source=dynamic_qwen_source,
    )
    _publish_bundle(output, payload_path, document, payload)
    return {
        "manifest": str(output),
        "payload": str(payload_path),
        "payload_bytes": len(payload),
        "payload_sha256": document["payload_sha256"],
        "tensor_count": len(document["tensors"]),
        "upstream_revision": PINNED_REVISION,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = produce_inputs(args)
    except (InputsError, OSError, subprocess.SubprocessError) as error:
        print(f"generation-input preparation rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
