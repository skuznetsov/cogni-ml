#!/usr/bin/env python3
"""Authenticate and compare the bounded Hy-Nomas prompt-edit resume matrix.

This CPU-only comparator verifies conditioning provenance, launcher receipts,
HNRS checkpoint headers, and F32 captures. It does not execute the model,
decode a video, or establish full-upstream video parity or visual quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any

import hy_nomas_compact_resume_reference as compact
import hy_nomas_continuation_inputs as continuation_inputs
import hy_nomas_generation_inputs as generation_inputs
import hy_nomas_sequence_resume_reference as resume


INDEX_SCHEMA = "cogni_ml.hy_nomas_segment_prompt_edit_resume_matrix/v1"
BASE_RECEIPT_SCHEMA = "cogni_ml.hy_nomas_compact_sequence_probe/v1"
EDIT_RECEIPT_SCHEMA = "cogni_ml.hy_nomas_segment_prompt_edit_probe/v1"
EDIT_IDENTITY_SCHEMA = "cogni_ml.hy_nomas_segment_prompt_edit/v1"
EDIT_BOUNDARY_SCHEMA = "cogni_ml.hy_nomas_prompt_edited_sequence_boundary/v1"
REPORT_SCHEMA = "cogni_ml.hy_nomas_segment_prompt_edit_resume_reference/v1"
HNRS_SCHEMA = "hy-nomas-resume-bundle-v1"
HNRS_MAGIC = b"HNRSB01\0"
HNRS_MAX_HEADER_BYTES = 1024 * 1024
HNRS_PAYLOAD_OFFSET = len(HNRS_MAGIC) + 4 + HNRS_MAX_HEADER_BYTES
IMPLEMENTATION_SOURCE_PATHS = (
    "examples/support/hy_nomas_sequence.cr",
    "src/ml/video/hy_nomas_segment_edit.cr",
    "src/ml/video/hy_nomas_continuation_history.cr",
    "src/ml/video/hy_nomas_timeline_ops.cr",
    "src/ml/video/kernels/hy_nomas_timeline_ops.metal",
    "src/ml/video/hy_nomas_euler.cr",
    "src/ml/video/kernels/hy_nomas_euler.metal",
    "src/ml/video/hy_nomas_block_mps.cr",
    "src/ml/video/hy_nomas_serial_mps.cr",
    "src/ml/video/hy_nomas_history_attention.cr",
    "src/ml/video/hy_nomas_history_attention_bridge.mm",
    "src/ml/video/hy_nomas_mps_bf16_linear_bridge.mm",
    "src/ml/video/hy_nomas_bf16_stack.cr",
)
BASE_CONTRACT_FIELD_ORDER = (
    "schema",
    "boundary",
    "upstream",
    "implementation_sha256",
    "executable_sha256",
    "crystal",
    "model_index_sha256",
    "input_manifest_sha256",
    "input_payload_sha256",
    "conditioning_manifest_sha256",
    "segments",
    "next_segment",
    "steps",
    "shift",
    "text_cache_policy",
    "history_dtype",
    "valid_text_tokens",
    "storage_text_tokens",
    "video_tokens",
    "layers",
    "heads",
    "head_dim",
    "timeline_shape",
    "cache_sigma",
)
SEGMENTS = 3
RESIDENT_SEGMENTS = 6
TARGET_SEGMENT = 1
CHANNELS = 32
SEGMENT_LATENTS = 30
HEIGHT = 9
WIDTH = 16
STEPS_PER_SEGMENT = 4
F32_BYTES = 4
TIMELINE_SHAPE = [1, CHANNELS, RESIDENT_SEGMENTS * SEGMENT_LATENTS, HEIGHT, WIDTH]
STEP_SHAPE = [1, CHANNELS, SEGMENT_LATENTS, HEIGHT, WIDTH]
TIMELINE_BYTES = math.prod(TIMELINE_SHAPE) * F32_BYTES
STEP_BYTES = math.prod(STEP_SHAPE) * F32_BYTES
ATOL = RTOL = 2e-5
ARM_INTERVALS = {
    "parent_a": [(0, 1)],
    "original_control": [(1, 3)],
    "edited_control": [(1, 3)],
    "edited_restart_chain": [(1, 2), (2, 3)],
}
IDENTITY_FIELDS = (
    "executable_sha256",
    "inputs",
    "inputs_sha256",
    "payload_sha256",
    "conditioning_sha256",
    "model_index_sha256",
    "upstream_revision",
    "segments",
    "steps_per_segment",
    "timeline_shape",
    "step_shape",
)
METRIC_FIELDS = {
    "encode_allocations",
    "encode_readbacks",
    "preterminal_waits",
    "command_segments",
    "terminal_waits",
}
PROMPT_INVALIDATION = {
    "latent_frames": [TARGET_SEGMENT * SEGMENT_LATENTS, SEGMENTS * SEGMENT_LATENTS],
    "decoder_pixel_frames": [TARGET_SEGMENT * 120 - 20, SEGMENTS * 120],
    "prior_segment_overlap_pixel_frames": [TARGET_SEGMENT * 120 - 20, TARGET_SEGMENT * 120],
    "units": "half-open global frame ranges; regenerate every descendant",
}
UNCHANGED_NOISE_SCOPE = "unchanged authenticated latent/epsilon banks; no reseeding"


class PromptEditResumeReferenceError(compact.CompactResumeReferenceError):
    """A prompt-edit matrix, receipt, checkpoint, or capture violates its contract."""


def _fail(message: str) -> None:
    raise PromptEditResumeReferenceError(message)


def _canonical_prompt_edit_contract_sha(
    base_contract_sha256: str, identity: dict[str, Any]
) -> str:
    # Match Crystal's insertion-ordered JSON in HyNomasSegmentPromptEdit#bind_contract.
    contract = {
        "schema": EDIT_BOUNDARY_SCHEMA,
        "base_contract_sha256": base_contract_sha256,
        "prompt_edit": identity,
    }
    raw = json.dumps(contract, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _base_contract_sha_for_next_segment(
    authenticated_contract: dict[str, Any], next_segment: int
) -> str:
    """Recreate Crystal's base boundary contract from an authenticated sibling."""
    if tuple(authenticated_contract) != BASE_CONTRACT_FIELD_ORDER:
        _fail("base HNRS contract fields or Crystal insertion order are unsupported")
    contract = dict(authenticated_contract)
    contract["next_segment"] = next_segment
    raw = json.dumps(contract, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _current_implementation_sha256() -> str:
    """Mirror the source-byte inventory compiled into HyNomasSequence contracts."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    try:
        for relative in IMPLEMENTATION_SOURCE_PATHS:
            digest.update((root / relative).read_bytes())
    except OSError as error:
        raise PromptEditResumeReferenceError(
            "cannot read the current Hy-Nomas implementation source inventory"
        ) from error
    return digest.hexdigest()


def _prompt_edit_identity(
    *, manifest_sha256: str, payload_sha256: str, prompt_sha256: str,
    valid_text_tokens: int,
) -> dict[str, Any]:
    return {
        "schema": EDIT_IDENTITY_SCHEMA,
        "run_segments": SEGMENTS,
        "target_segment": TARGET_SEGMENT,
        "conditioning_manifest_sha256": manifest_sha256,
        "conditioning_payload_sha256": payload_sha256,
        "prompt_sha256": prompt_sha256,
        "valid_text_tokens": valid_text_tokens,
        "following_prompt": "original_as_new_text_turn",
    }


def _load_prompt_inputs(index_record: Any) -> dict[str, Any]:
    """Authenticate the separate conditioned v2 generation-input bundle."""
    expected_keys = {"manifest_path", "manifest_sha256", "payload_path", "payload_sha256"}
    if type(index_record) is not dict or set(index_record) != expected_keys:
        _fail("prompt_inputs must name exactly the conditioned manifest and payload")
    manifest_path = compact._absolute_file(index_record["manifest_path"], "prompt_inputs manifest")
    manifest_sha = compact._sha(index_record["manifest_sha256"], "prompt_inputs manifest_sha256")
    payload_path = compact._absolute_file(index_record["payload_path"], "prompt_inputs payload")
    payload_sha = compact._sha(index_record["payload_sha256"], "prompt_inputs payload_sha256")
    if manifest_path.parent != payload_path.parent:
        _fail("prompt_inputs payload must be beside its manifest")

    document = resume._read_json(manifest_path, label="prompt_inputs manifest")
    if document.get("schema") != generation_inputs.DYNAMIC_SCHEMA or document.get("schema_version") != generation_inputs.DYNAMIC_SCHEMA_VERSION:
        _fail("prompt_inputs must use the authenticated generation_inputs v2 schema")
    if payload_path.name != document.get("payload"):
        _fail("prompt_inputs payload path does not match its manifest")
    _, manifest_bytes = compact._read_hashed_file(
        manifest_path, manifest_sha, "prompt_inputs manifest", retain_payload=False
    )
    payload_bytes = compact._int(document.get("payload_bytes"), "prompt_inputs payload_bytes", minimum=1)
    if document.get("payload_sha256") != payload_sha:
        _fail("prompt_inputs payload SHA differs from its manifest")
    try:
        generation_inputs.validate_document(document, payload_bytes=payload_bytes)
    except Exception as error:
        raise PromptEditResumeReferenceError(f"prompt_inputs manifest rejected: {error}") from error

    _validate_generation_payload(payload_path, document, payload_sha, payload_bytes)
    source = document.get("qwen_source")
    prompt = document.get("prompt")
    if type(source) is not dict or type(prompt) is not str:
        _fail("prompt_inputs must contain a conditioned prompt and Qwen source")
    prompt_sha = compact._sha(source.get("prompt_sha256"), "prompt_inputs prompt_sha256")
    if prompt_sha != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        _fail("prompt_inputs prompt SHA does not match its text")
    valid_tokens = compact._int(source.get("valid_tokens"), "prompt_inputs valid_tokens", minimum=1)
    if valid_tokens > 1000:
        _fail("prompt_inputs valid_tokens exceeds the admitted Qwen text bound")

    qwen_record = document["tensors"]["input.qwen_bf16"]
    crop_sha = _authenticate_qwen_reference_crop(source, qwen_record)
    if crop_sha != qwen_record["sha256"]:
        _fail("prompt_inputs Qwen record differs from the authenticated state26 crop")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_bytes": manifest_bytes,
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "payload_bytes": payload_bytes,
        "prompt_sha256": prompt_sha,
        "prompt": prompt,
        "valid_text_tokens": valid_tokens,
        "document": document,
        "qwen_crop_sha256": crop_sha,
    }


def _validate_generation_payload(
    path: Path, document: dict[str, Any], expected_sha256: str, expected_bytes: int
) -> None:
    """Stream-authenticate the aggregate and every v2 tensor record without a 300 MiB copy."""
    specs = generation_inputs.tensor_specs_for_resolution(
        generation_inputs._resolution_for_schema(document)
    )
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            _fail("prompt_inputs payload must be a regular file with the declared extent")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or resume._stat_identity(opened) != resume._stat_identity(before):
                _fail("prompt_inputs payload changed before it was opened")
            aggregate = hashlib.sha256()
            total = 0
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                for name, _dtype, _shape in specs:
                    record = document["tensors"][name]
                    record_hash = hashlib.sha256()
                    remaining = record["nbytes"]
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            _fail(f"prompt_inputs payload is truncated in tensor {name}")
                        aggregate.update(chunk)
                        record_hash.update(chunk)
                        total += len(chunk)
                        remaining -= len(chunk)
                    if record_hash.hexdigest() != record["sha256"]:
                        _fail(f"prompt_inputs tensor digest mismatch: {name}")
                if stream.read(1):
                    _fail("prompt_inputs payload has bytes beyond its tensor inventory")
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.stat(path, follow_symlinks=False)
    except PromptEditResumeReferenceError:
        raise
    except OSError as error:
        raise PromptEditResumeReferenceError("cannot read prompt_inputs payload") from error
    if (resume._stat_identity(before) != resume._stat_identity(after_open)
            or resume._stat_identity(before) != resume._stat_identity(after_path)
            or total != expected_bytes):
        _fail("prompt_inputs payload changed while being authenticated")
    if aggregate.hexdigest() != expected_sha256:
        _fail("prompt_inputs aggregate payload SHA mismatch")


def _authenticate_qwen_reference_crop(source: dict[str, Any], qwen_record: dict[str, Any]) -> str:
    """Bind the conditioned Qwen tensor to a validated reference's state26 crop."""
    reference_path = compact._absolute_file(source.get("reference_path"), "Qwen reference")
    reference_sha = compact._sha(source.get("reference_sha256"), "Qwen reference_sha256")
    sidecar_path = compact._absolute_file(source.get("sidecar_path"), "Qwen hidden-state sidecar")
    sidecar_sha = compact._sha(source.get("sidecar_sha256"), "Qwen sidecar_sha256")
    sidecar_bytes = compact._int(source.get("sidecar_bytes"), "Qwen sidecar_bytes", minimum=1)
    if sidecar_path.parent != reference_path.parent:
        _fail("Qwen sidecar must be beside its reference manifest")

    reference_raw, _ = compact._read_hashed_file(
        reference_path, reference_sha, "Qwen reference", retain_payload=True
    )
    try:
        reference_document = json.loads(
            reference_raw.decode("utf-8"),
            object_pairs_hook=resume._unique_object,
            parse_constant=resume._reject_json_constant,
        )
    except Exception as error:
        raise PromptEditResumeReferenceError("Qwen reference manifest is malformed") from error
    try:
        qwen_reference = generation_inputs._load_qwen_prefix_reference()
        prompt_reference = generation_inputs._prompt_reference_helper()
        qwen_reference._validate_manifest(reference_document, reference_path)
        descriptor = reference_document["sidecars"]["bf16"]
        expected_sidecar_name = descriptor["path"]
        if (type(expected_sidecar_name) is not str
                or (reference_path.parent / expected_sidecar_name).resolve(strict=True) != sidecar_path
                or descriptor.get("sha256") != sidecar_sha
                or descriptor.get("nbytes") != sidecar_bytes
                or descriptor.get("encoding") != "bfloat16-le"):
            _fail("Qwen reference descriptor does not bind the admitted sidecar")
        if reference_document.get("hy_nomas_prompt") != source.get("prompt_contract"):
            _fail("Qwen reference does not bind the conditioned prompt contract")
        inputs = reference_document["inputs"]
        raw_length = prompt_reference.validate_prompt_metadata(
            source["prompt_contract"],
            inputs["token_ids"],
            inputs["attention_mask"],
            inputs["position_ids_f32"],
            inputs["position_ids_for_official_model"],
            inputs["multimodal_position_shape"],
            inputs["position_offset"],
        )
    except PromptEditResumeReferenceError:
        raise
    except Exception as error:
        raise PromptEditResumeReferenceError(f"Qwen reference provenance rejected: {error}") from error

    valid_tokens = compact._int(source.get("valid_tokens"), "Qwen valid_tokens", minimum=1)
    mask = source.get("cropped_attention_mask")
    if (type(mask) is not list or len(mask) != 1280
            or any(type(value) is not int or value not in (0, 1) for value in mask)
            or sum(mask) != valid_tokens or mask != [1] * valid_tokens + [0] * (1280 - valid_tokens)):
        _fail("Qwen reference does not expose the exact authenticated visible text prefix")
    if (source.get("state_index") != 26 or source.get("crop_tokens") != 1280
            or source.get("hidden_size") != 3584):
        _fail("Qwen source must use the admitted 1280-row state26 crop geometry")
    crop = source.get("crop")
    if type(raw_length) is not int or raw_length != source.get("raw_sequence_length"):
        _fail("Qwen prompt metadata raw sequence length differs from the source record")
    if (type(crop) is not list or len(crop) != 2
            or any(type(value) is not int for value in crop)
            or crop[1] != raw_length or crop[1] - crop[0] != 1280):
        _fail("Qwen source crop geometry is malformed")
    offset = (26 * raw_length + crop[0]) * 3584 * 2
    expected_nbytes = compact._int(qwen_record.get("nbytes"), "Qwen tensor nbytes", minimum=1)
    if (qwen_record.get("dtype") != "BF16" or qwen_record.get("shape") != [1, 1280, 3584]
            or expected_nbytes != 1280 * 3584 * 2):
        _fail("prompt_inputs Qwen tensor geometry is unsupported")
    crop_payload, _ = compact._read_hashed_file(
        sidecar_path,
        sidecar_sha,
        "Qwen hidden-state sidecar",
        expected_bytes=sidecar_bytes,
        retain_payload=False,
        retain_range=(offset, expected_nbytes),
    )
    return hashlib.sha256(crop_payload).hexdigest()


def _base_prompt_sha256(source: dict[str, Any]) -> str:
    """Authenticate the original continuation tensors and its copied Qwen input."""
    manifest_path = Path(source["manifest_path"])
    document = resume._read_json(manifest_path, label="base continuation input manifest")
    qwen_source = document.get("qwen_source")
    if (type(qwen_source) is not dict or type(document.get("prompt")) is not str
            or not document["prompt"]):
        _fail("base continuation manifest must identify its original Qwen prompt")
    prompt_contract = qwen_source.get("prompt_contract")
    if type(prompt_contract) is not dict or prompt_contract.get("prompt") != document["prompt"]:
        _fail("base continuation prompt does not match its Qwen prompt contract")
    prompt_sha256 = compact._sha(qwen_source.get("prompt_sha256"), "base prompt_sha256")
    if prompt_sha256 != hashlib.sha256(document["prompt"].encode("utf-8")).hexdigest():
        _fail("base continuation prompt digest does not match its text")
    try:
        continuation_inputs.validate_document(
            document, payload_bytes=source["payload_bytes"]
        )
    except Exception as error:
        raise PromptEditResumeReferenceError(
            f"base continuation manifest rejected: {error}"
        ) from error
    if (document["payload_sha256"] != source["payload_sha256"]
            or document["payload"] != Path(source["payload_path"]).name):
        _fail("base continuation payload path or SHA differs from source_bank")

    _validate_continuation_payload(Path(source["payload_path"]), document)
    bundle = document["qwen_input_bundle"]
    try:
        admitted_qwen = continuation_inputs._admit_qwen_inputs(
            bundle["manifest_path"], bundle["manifest_sha256"]
        )
    except Exception as error:
        raise PromptEditResumeReferenceError(
            f"base Qwen input bundle rejected: {error}"
        ) from error
    qwen_document = admitted_qwen["document"]
    expected_bundle_values = {
        "manifest_path": admitted_qwen["path"],
        "manifest_sha256": admitted_qwen["manifest_sha256"],
        "payload_path": admitted_qwen["payload_path"],
        "payload_sha256": admitted_qwen["payload_sha256"],
        "qwen_tensor_sha256": admitted_qwen["qwen_record"]["sha256"],
        "first_rope_cos_sha256": admitted_qwen["rope_records"]["cos"]["sha256"],
        "first_rope_sin_sha256": admitted_qwen["rope_records"]["sin"]["sha256"],
        "first_rope_atol": continuation_inputs.ROPE_ATOL,
        "first_rope_rtol": continuation_inputs.ROPE_RTOL,
    }
    if bundle != expected_bundle_values:
        _fail("base continuation Qwen bundle fields do not match the authenticated v2 bundle")
    if (qwen_source != qwen_document.get("qwen_source")
            or document["prompt"] != qwen_document.get("prompt")
            or document["tensors"]["input.qwen_bf16"]["sha256"]
            != admitted_qwen["qwen_record"]["sha256"]):
        _fail("base continuation Qwen tensor or prompt differs from its authenticated v2 bundle")
    valid_tokens = compact._int(qwen_source.get("valid_tokens"), "base valid_text_tokens", minimum=1)
    if valid_tokens > 1000:
        _fail("base valid_text_tokens exceeds the admitted dynamic Qwen bound")
    source["original_valid_text_tokens"] = valid_tokens
    source["original_prompt_sha256"] = prompt_sha256
    source["continuation_tensor_digests_authenticated"] = [
        name for name, _dtype, _shape in continuation_inputs.TENSOR_SPECS
    ]
    source["base_qwen_input_bundle"] = {
        "manifest_path": admitted_qwen["path"],
        "manifest_sha256": admitted_qwen["manifest_sha256"],
        "payload_path": admitted_qwen["payload_path"],
        "payload_sha256": admitted_qwen["payload_sha256"],
        "qwen_tensor_sha256": admitted_qwen["qwen_record"]["sha256"],
        "state_index": qwen_source["state_index"],
        "crop_tokens": qwen_source["crop_tokens"],
    }
    return prompt_sha256


def _validate_continuation_payload(path: Path, document: dict[str, Any]) -> None:
    """Stream-check the complete continuation payload and each inventoried tensor."""
    expected_bytes = compact._int(
        document.get("payload_bytes"), "base continuation payload_bytes", minimum=1
    )
    expected_sha = compact._sha(
        document.get("payload_sha256"), "base continuation payload_sha256"
    )
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            _fail("base continuation payload must be a regular file with its declared extent")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or resume._stat_identity(opened) != resume._stat_identity(before):
                _fail("base continuation payload changed before it was opened")
            aggregate = hashlib.sha256()
            total = 0
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                for name, _dtype, _shape in continuation_inputs.TENSOR_SPECS:
                    record = document["tensors"][name]
                    record_hash = hashlib.sha256()
                    remaining = record["nbytes"]
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            _fail(f"base continuation payload is truncated in tensor {name}")
                        aggregate.update(chunk)
                        record_hash.update(chunk)
                        total += len(chunk)
                        remaining -= len(chunk)
                    if record_hash.hexdigest() != record["sha256"]:
                        _fail(f"base continuation tensor digest mismatch: {name}")
                if stream.read(1):
                    _fail("base continuation payload has bytes beyond its tensor inventory")
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.stat(path, follow_symlinks=False)
    except PromptEditResumeReferenceError:
        raise
    except OSError as error:
        raise PromptEditResumeReferenceError("cannot read base continuation payload") from error
    if (resume._stat_identity(before) != resume._stat_identity(after_open)
            or resume._stat_identity(before) != resume._stat_identity(after_path)
            or total != expected_bytes):
        _fail("base continuation payload changed while being authenticated")
    if aggregate.hexdigest() != expected_sha:
        _fail("base continuation aggregate payload SHA mismatch")


def _validate_receipt(
    receipt: dict[str, Any],
    arm: str,
    interval: tuple[int, int],
    source: dict[str, Any],
    prompt_inputs: dict[str, Any],
    base_prompt_sha256: str,
) -> None:
    label = f"{arm} receipt"
    start, stop = interval
    edited = arm in ("edited_control", "edited_restart_chain")
    expected_schema = EDIT_RECEIPT_SCHEMA if edited else BASE_RECEIPT_SCHEMA
    if receipt.get("schema") != expected_schema:
        _fail(f"{label} schema does not match its prompt-edit treatment")
    if receipt.get("mode") != ("save" if start == 0 else "resume"):
        _fail(f"{label} mode does not match its interval")

    for key, expected in (
        ("segments", SEGMENTS),
        ("start_segment", start),
        ("stop_after_segments", stop),
        ("resume_next_segment", start),
        ("steps_per_segment", STEPS_PER_SEGMENT),
    ):
        if type(receipt.get(key)) is not int or receipt[key] != expected:
            _fail(f"{label} {key} does not match its admitted interval")
    if type(receipt.get("pid")) is not int or receipt["pid"] <= 0:
        _fail(f"{label} pid must be a positive integer")
    for key in (
        "executable_sha256", "inputs_sha256", "payload_sha256",
        "conditioning_sha256", "model_index_sha256",
    ):
        compact._sha(receipt.get(key), f"{label} {key}")
    if type(receipt.get("inputs")) is not str or not receipt["inputs"]:
        _fail(f"{label} inputs must name its authenticated base manifest")
    if type(receipt.get("upstream_revision")) is not str or not receipt["upstream_revision"]:
        _fail(f"{label} upstream_revision must be non-empty")

    expected_text_tokens = (
        [source["original_valid_text_tokens"], prompt_inputs["valid_text_tokens"],
         source["original_valid_text_tokens"]]
        if edited else [source["original_valid_text_tokens"]] * SEGMENTS
    )
    expected_cache_text = [True, True, True] if edited else [True, False, False]
    raw_text_tokens = receipt.get("segment_text_tokens")
    if (type(raw_text_tokens) is not list or len(raw_text_tokens) != SEGMENTS
            or any(type(value) is not int or value <= 0 for value in raw_text_tokens)
            or raw_text_tokens != expected_text_tokens):
        _fail(f"{label} segment_text_tokens do not implement the admitted per-segment prompts")
    raw_cache_text = receipt.get("segment_cache_text")
    if (type(raw_cache_text) is not list or len(raw_cache_text) != SEGMENTS
            or any(type(value) is not bool for value in raw_cache_text)
            or raw_cache_text != expected_cache_text):
        _fail(f"{label} segment_cache_text does not implement the upstream text-turn schedule")
    if (type(receipt.get("valid_text_tokens")) is not int
            or receipt["valid_text_tokens"] != source["original_valid_text_tokens"]):
        _fail(f"{label} base valid_text_tokens must remain bound to the original conditioning")

    if edited:
        identity = _prompt_edit_identity(
            manifest_sha256=prompt_inputs["manifest_sha256"],
            payload_sha256=prompt_inputs["payload_sha256"],
            prompt_sha256=prompt_inputs["prompt_sha256"],
            valid_text_tokens=prompt_inputs["valid_text_tokens"],
        )
        if receipt.get("prompt_edit") != identity:
            _fail(f"{label} prompt_edit identity does not match the authenticated conditioned inputs")
        if receipt.get("prompt_inputs") != prompt_inputs["manifest_path"]:
            _fail(f"{label} prompt_inputs path does not match the authenticated conditioned manifest")
        if receipt.get("prompt_inputs_sha256") != prompt_inputs["manifest_sha256"]:
            _fail(f"{label} prompt_inputs_sha256 does not match its authenticated manifest")
        if prompt_inputs["prompt_sha256"] == base_prompt_sha256:
            _fail("edited prompt SHA is identical to the original prompt")
        expected_invalidation = PROMPT_INVALIDATION
        if receipt.get("invalidation") != expected_invalidation:
            _fail(f"{label} invalidation ranges do not cover B and every descendant")
    else:
        if receipt.get("prompt_edit") is not None:
            _fail(f"{label} cannot claim a prompt edit")
        if receipt.get("prompt_inputs") != "" or receipt.get("prompt_inputs_sha256") != "":
            _fail(f"{label} cannot claim conditioned prompt inputs")
        if receipt.get("invalidation") is not None:
            _fail(f"{label} unedited control cannot claim prompt-edit invalidation")
    if receipt.get("noise_edit") is not None:
        _fail(f"{label} prompt-only matrix cannot claim a noise edit")
    if receipt.get("noise_scope") != UNCHANGED_NOISE_SCOPE:
        _fail(f"{label} must preserve the authenticated latent and epsilon banks")

    for key, expected in (("timeline_shape", TIMELINE_SHAPE), ("step_shape", STEP_SHAPE)):
        value = receipt.get(key)
        if type(value) is not list or any(type(dim) is not int for dim in value) or value != expected:
            _fail(f"{label} {key} does not match the six-bank DiT geometry")

    if receipt.get("state_format") != "exact" or receipt.get("cache_roundtrip") != "exact":
        _fail(f"{label} must use exact checkpoint and cache-roundtrip treatment")
    if receipt.get("qbit_roundtrip_boundaries") != []:
        _fail(f"{label} cannot report QBit cache roundtrips")
    if type(receipt.get("compact_pack_boundary")) is not int or receipt["compact_pack_boundary"] != 0:
        _fail(f"{label} cannot report compact checkpoint packing")
    codec = receipt.get("codec")
    if type(codec) is not dict or codec:
        _fail(f"{label} cannot report codec operations")

    boundary_map = receipt.get("boundary_contract_sha256")
    if type(boundary_map) is not dict or set(boundary_map) != {"1", "2"}:
        _fail(f"{label} boundary contract map must contain boundaries 1 and 2")
    for boundary, digest in boundary_map.items():
        compact._sha(digest, f"{label} boundary contract {boundary}")
    contract = receipt.get("contract_sha256")
    if start == 0:
        if contract != "":
            _fail(f"{label} start-zero contract_sha256 must be empty")
    elif compact._sha(contract, f"{label} contract_sha256") != boundary_map[str(start)]:
        _fail(f"{label} contract SHA does not match its start boundary")
    state_input_sha = receipt.get("state_input_sha256")
    if start == 0:
        if state_input_sha != "":
            _fail(f"{label} start-zero interval cannot consume a state")
    else:
        compact._sha(state_input_sha, f"{label} state_input_sha256")

    metrics = receipt.get("metrics")
    if type(metrics) is not dict or set(metrics) != METRIC_FIELDS:
        _fail(f"{label} metrics must contain exactly the five instrumented counters")
    for key, value in metrics.items():
        compact._int(value, f"{label} metric {key}")
    for key in ("encode_allocations", "encode_readbacks", "preterminal_waits"):
        if metrics[key] != 0:
            _fail(f"{label} metric {key} must be zero")
    if metrics["terminal_waits"] != 1:
        _fail(f"{label} must report exactly one terminal drain")

    checkpoint_file_bytes = compact._int(
        receipt.get("checkpoint_file_bytes"), f"{label} checkpoint_file_bytes"
    )
    logical_packed = compact._int(
        receipt.get("logical_packed_kv_bytes"), f"{label} logical_packed_kv_bytes"
    )
    compact._int(receipt.get("resident_f32_kv_bytes"), f"{label} resident_f32_kv_bytes", minimum=1)
    checkpoint_payload_bytes = compact._int(
        receipt.get("checkpoint_payload_bytes"), f"{label} checkpoint_payload_bytes"
    )
    if stop < SEGMENTS:
        if (checkpoint_file_bytes != HNRS_PAYLOAD_OFFSET + checkpoint_payload_bytes
                or checkpoint_payload_bytes <= TIMELINE_BYTES):
            _fail(f"{label} checkpoint metadata omits its bounded HNRS payload")
    elif checkpoint_file_bytes != 0 or checkpoint_payload_bytes != 0:
        _fail(f"{label} terminal interval cannot claim an output checkpoint")
    if logical_packed != 0:
        _fail(f"{label} exact treatment cannot report packed K/V bytes")
    if receipt.get("decoder") is not False or receipt.get("full_generation") is not False:
        _fail(f"{label} must remain a DiT-only interval probe")
    if receipt.get("full_upstream_video_parity") is not False:
        _fail(f"{label} cannot claim full upstream video parity")

    receipt_manifest = compact._absolute_file(receipt["inputs"], f"{label} base input manifest")
    if receipt_manifest != Path(source["manifest_path"]):
        _fail(f"{label} base input manifest does not match the authenticated source bank")
    if receipt["inputs_sha256"] != source["manifest_sha256"]:
        _fail(f"{label} inputs_sha256 does not match the original continuation manifest")
    if receipt["payload_sha256"] != source["payload_sha256"]:
        _fail(f"{label} payload_sha256 does not match the original continuation payload")


def _parse_hnrs_header(
    path: Path, expected_sha256: str, expected_file_bytes: int,
    expected_payload_bytes: int, label: str,
) -> dict[str, Any]:
    """Read and validate the fixed HNRS header while authenticating the whole file."""
    try:
        prefix, total_bytes = compact._read_hashed_file(
            path,
            expected_sha256,
            f"{label} HNRS checkpoint",
            expected_bytes=expected_file_bytes,
            retain_payload=False,
            retain_range=(0, HNRS_PAYLOAD_OFFSET),
        )
    except compact.CompactResumeReferenceError as error:
        raise PromptEditResumeReferenceError(str(error)) from error
    if len(prefix) != HNRS_PAYLOAD_OFFSET or total_bytes != expected_file_bytes:
        _fail(f"{label} checkpoint HNRS header extent mismatch")
    if prefix[:len(HNRS_MAGIC)] != HNRS_MAGIC:
        _fail(f"{label} checkpoint HNRS magic mismatch")
    header_length = struct.unpack_from("<I", prefix, len(HNRS_MAGIC))[0]
    if not 0 < header_length <= HNRS_MAX_HEADER_BYTES:
        _fail(f"{label} checkpoint HNRS header length is invalid")
    header_start = len(HNRS_MAGIC) + 4
    header_stop = header_start + header_length
    raw_header = prefix[header_start:header_stop]
    if any(prefix[header_stop:HNRS_PAYLOAD_OFFSET]):
        _fail(f"{label} checkpoint HNRS header padding is not zero")
    try:
        header = json.loads(
            raw_header.decode("utf-8"),
            object_pairs_hook=resume._unique_object,
            parse_constant=resume._reject_json_constant,
        )
    except Exception as error:
        raise PromptEditResumeReferenceError(f"{label} HNRS header JSON is malformed") from error
    if type(header) is not dict or set(header) != {"schema", "schema_version", "contract", "records"}:
        _fail(f"{label} HNRS header fields are unsupported")
    if header.get("schema") != HNRS_SCHEMA or type(header.get("schema_version")) is not int or header["schema_version"] != 1:
        _fail(f"{label} HNRS schema is unsupported")
    contract = header.get("contract")
    if type(contract) is not str or not contract:
        _fail(f"{label} HNRS contract is missing")
    records = header.get("records")
    if type(records) is not list or not records:
        _fail(f"{label} HNRS record inventory is empty")
    names: list[str] = []
    cursor = 0
    for record in records:
        if type(record) is not dict or set(record) != {"name", "nbytes", "offset", "sha256"}:
            _fail(f"{label} HNRS record fields are malformed")
        name = record.get("name")
        nbytes = record.get("nbytes")
        offset = record.get("offset")
        if type(name) is not str or not name or "\x00" in name:
            _fail(f"{label} HNRS record name is malformed")
        if type(nbytes) is not int or nbytes < 0 or type(offset) is not int or offset != cursor:
            _fail(f"{label} HNRS record extent is malformed")
        compact._sha(record.get("sha256"), f"{label} HNRS record SHA")
        names.append(name)
        cursor += nbytes
    if names != sorted(names) or len(names) != len(set(names)):
        _fail(f"{label} HNRS record names are not unique canonical order")
    if cursor != expected_payload_bytes or HNRS_PAYLOAD_OFFSET + cursor != expected_file_bytes:
        _fail(f"{label} HNRS record extents do not match the saved payload")
    try:
        contract_document = json.loads(
            contract,
            object_pairs_hook=resume._unique_object,
            parse_constant=resume._reject_json_constant,
        )
    except Exception as error:
        raise PromptEditResumeReferenceError(f"{label} HNRS contract JSON is malformed") from error
    if type(contract_document) is not dict:
        _fail(f"{label} HNRS contract must be an object")
    return {
        "sha256": expected_sha256,
        "file_bytes": total_bytes,
        "payload_bytes": cursor,
        "contract": contract,
        "contract_sha256": hashlib.sha256(contract.encode("utf-8")).hexdigest(),
        "contract_document": contract_document,
        "records": records,
    }


def _load_run(
    arm: str,
    entry: dict[str, Any],
    expected_interval: tuple[int, int],
    source: dict[str, Any],
    prompt_inputs: dict[str, Any],
    base_prompt_sha256: str,
) -> dict[str, Any]:
    if type(entry) is not dict or set(entry) != {"receipt_path", "interval"}:
        _fail(f"{arm} run entry must contain only receipt_path and interval")
    receipt_path = compact._absolute_file(entry["receipt_path"], f"{arm} receipt_path")
    if receipt_path.name != "receipt.json":
        _fail(f"{arm} receipt_path must name receipt.json")
    directory = resume._directory(receipt_path.parent, f"{arm} output directory")
    if receipt_path.parent != directory:
        _fail(f"{arm} receipt_path must use its canonical output directory")
    raw_interval = entry["interval"]
    if type(raw_interval) is not list or len(raw_interval) != 2:
        _fail(f"{arm} interval must be a two-integer [start, stop] list")
    interval = (
        compact._int(raw_interval[0], f"{arm} interval start"),
        compact._int(raw_interval[1], f"{arm} interval stop", minimum=1),
    )
    if interval != expected_interval:
        _fail(f"{arm} interval must be {list(expected_interval)}")
    receipt = resume._read_json(receipt_path, label=f"{arm} receipt")
    _validate_receipt(receipt, arm, interval, source, prompt_inputs, base_prompt_sha256)

    start, stop = interval
    diagnostics = receipt.get("diagnostics")
    expected_artifacts = compact._artifact_names(start, stop, SEGMENTS, "exact")
    if type(diagnostics) is not dict or set(diagnostics) != expected_artifacts:
        _fail(f"{arm} diagnostics do not name exactly the interval artifacts")
    timeline, timeline_meta = resume._read_artifact(
        directory, diagnostics, "timeline", "timeline.f32le", TIMELINE_BYTES,
        arm, retain_payload=True,
    )
    resume._finite(timeline, f"{arm} timeline")
    steps: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for segment in range(start, stop):
        for step in range(1, STEPS_PER_SEGMENT + 1):
            key = f"segment{segment}-step{step}"
            payload, metadata = resume._read_artifact(
                directory, diagnostics, key, f"{key}.f32le", STEP_BYTES,
                arm, retain_payload=True,
            )
            resume._finite(payload, f"{arm} {key}")
            steps[key] = (payload, metadata)
        final_step = steps[f"segment{segment}-step{STEPS_PER_SEGMENT}"][0]
        timeline_segment = compact._time_slice(
            timeline, TIMELINE_SHAPE,
            segment * SEGMENT_LATENTS, (segment + 1) * SEGMENT_LATENTS,
        )
        if timeline_segment != final_step:
            _fail(f"{arm} timeline does not contain final segment {segment} step capture")

    state = None
    if stop < SEGMENTS:
        try:
            _, state_meta = resume._read_artifact(
                directory,
                diagnostics,
                "state",
                "boundary.hnrs",
                receipt["checkpoint_file_bytes"],
                arm,
                retain_payload=False,
            )
        except resume.ResumeReferenceError as error:
            raise PromptEditResumeReferenceError(
                f"{arm} checkpoint file byte extent or digest mismatch: {error}"
            ) from error
        state = _parse_hnrs_header(
            Path(state_meta["path"]),
            diagnostics["state_sha256"],
            receipt["checkpoint_file_bytes"],
            receipt["checkpoint_payload_bytes"],
            arm,
        )
        expected_contract = receipt["boundary_contract_sha256"][str(stop)]
        if state["contract_sha256"] != expected_contract:
            _fail(f"{arm} HNRS header contract does not match its after-segment boundary")
    return {
        "arm": arm,
        "interval": interval,
        "directory": directory,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "timeline": timeline,
        "timeline_meta": timeline_meta,
        "steps": steps,
        "state": state,
    }


def _load_arms(
    index: dict[str, Any], source: dict[str, Any], prompt_inputs: dict[str, Any],
    base_prompt_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    raw_arms = index.get("arms")
    if type(raw_arms) is not dict or set(raw_arms) != set(ARM_INTERVALS):
        _fail("index arms must contain exactly parent_a, original_control, edited_control, and edited_restart_chain")
    arms: dict[str, list[dict[str, Any]]] = {}
    directories: set[Path] = set()
    pids: set[int] = set()
    for arm, intervals in ARM_INTERVALS.items():
        document = raw_arms[arm]
        if type(document) is not dict or set(document) != {"runs"} or type(document["runs"]) is not list:
            _fail(f"{arm} must contain a runs list")
        if len(document["runs"]) != len(intervals):
            _fail(f"{arm} run count does not match its required intervals")
        runs = []
        for entry, interval in zip(document["runs"], intervals):
            run = _load_run(
                arm, entry, interval, source, prompt_inputs, base_prompt_sha256
            )
            if run["directory"] in directories:
                _fail("every matrix interval must use a distinct output directory")
            directories.add(run["directory"])
            pid = run["receipt"]["pid"]
            if pid in pids:
                _fail("every matrix interval must use a distinct process ID")
            pids.add(pid)
            runs.append(run)
        arms[arm] = runs
    return arms


def _validate_identities(
    arms: dict[str, list[dict[str, Any]]], source: dict[str, Any]
) -> dict[str, Any]:
    reference = arms["parent_a"][0]["receipt"]
    for arm_runs in arms.values():
        for run in arm_runs:
            receipt = run["receipt"]
            for key in IDENTITY_FIELDS:
                if receipt.get(key) != reference.get(key):
                    _fail(f"all matrix runs must share identity field {key}")
            if compact._absolute_file(receipt["inputs"], "receipt base input manifest") != Path(source["manifest_path"]):
                _fail("all matrix runs must name the authenticated original continuation manifest")
            if receipt["inputs_sha256"] != source["manifest_sha256"]:
                _fail("all matrix runs must share the authenticated original input manifest digest")
            if receipt["payload_sha256"] != source["payload_sha256"]:
                _fail("all matrix runs must share the authenticated original input payload digest")
    return {
        "executable_sha256": reference["executable_sha256"],
        "inputs": reference["inputs"],
        "inputs_sha256": reference["inputs_sha256"],
        "payload_sha256": reference["payload_sha256"],
        "conditioning_sha256": reference["conditioning_sha256"],
        "model_index_sha256": reference["model_index_sha256"],
        "upstream_revision": reference["upstream_revision"],
        "distinct_processes": True,
        "pids": {
            arm: [run["receipt"]["pid"] for run in runs]
            for arm, runs in arms.items()
        },
    }


def _edited_boundary_contracts(
    base_contracts: dict[str, str], identity: dict[str, Any]
) -> dict[str, str]:
    return {
        "1": base_contracts["1"],
        "2": _canonical_prompt_edit_contract_sha(base_contracts["2"], identity),
    }


def _validate_contracts(
    arms: dict[str, list[dict[str, Any]]], prompt_inputs: dict[str, Any]
) -> dict[str, Any]:
    base_contracts = arms["original_control"][0]["receipt"]["boundary_contract_sha256"]
    if arms["parent_a"][0]["receipt"]["boundary_contract_sha256"] != base_contracts:
        _fail("fresh parent A and original control do not share the base boundary contracts")
    identity = _prompt_edit_identity(
        manifest_sha256=prompt_inputs["manifest_sha256"],
        payload_sha256=prompt_inputs["payload_sha256"],
        prompt_sha256=prompt_inputs["prompt_sha256"],
        valid_text_tokens=prompt_inputs["valid_text_tokens"],
    )
    edited_contracts = _edited_boundary_contracts(base_contracts, identity)
    edited_runs = arms["edited_control"] + arms["edited_restart_chain"]
    for run in edited_runs:
        if run["receipt"]["boundary_contract_sha256"] != edited_contracts:
            _fail("edited boundary contract must preserve after-A and bind the new prompt after-B")
    implementation_sha256 = _current_implementation_sha256()
    parent_a = arms["parent_a"][0]
    parent_state = parent_a["state"]
    if parent_state is None:
        _fail("fresh parent A must save its authenticated after-A base contract")
    after_a_contract = parent_state["contract_document"]
    _validate_base_contract_document(
        after_a_contract,
        parent_a["receipt"],
        next_segment=1,
        original_valid_text_tokens=parent_a["receipt"]["segment_text_tokens"][0],
        expected_implementation_sha256=implementation_sha256,
    )
    if parent_state["contract_sha256"] != base_contracts["1"]:
        _fail("base after-A receipt contract digest does not match the authenticated parent checkpoint")
    reconstructed_after_b_sha256 = _base_contract_sha_for_next_segment(
        after_a_contract, 2
    )
    if base_contracts["2"] != reconstructed_after_b_sha256:
        _fail("base after-B contract digest is not reconstructed from the authenticated after-A contract")
    for run in arms["edited_restart_chain"][:1]:
        state_contract = run["state"]["contract_document"] if run["state"] else None
        if type(state_contract) is not dict or set(state_contract) != {
            "schema", "base_contract_sha256", "prompt_edit"
        }:
            _fail("edited after-B checkpoint must contain the prompt-edited boundary contract")
        if (state_contract.get("schema") != EDIT_BOUNDARY_SCHEMA
                or state_contract.get("base_contract_sha256") != base_contracts["2"]
                or state_contract.get("prompt_edit") != identity):
            _fail("edited after-B HNRS contract does not bind the prompt identity")
    return {
        "base": dict(base_contracts),
        "edited": edited_contracts,
        "after_a_preserved": True,
        "after_b_binds_prompt_edit": True,
        "hnrs_contract_headers_parsed": True,
        "current_implementation_sha256": implementation_sha256,
    }


def _validate_base_contract_document(
    contract: dict[str, Any], receipt: dict[str, Any], *, next_segment: int,
    original_valid_text_tokens: int, expected_implementation_sha256: str,
) -> None:
    if tuple(contract) != BASE_CONTRACT_FIELD_ORDER:
        _fail("fresh after-A HNRS base contract fields or Crystal insertion order are unsupported")
    expected = {
        "schema": "cogni_ml.hy_nomas_sequence_boundary/v1",
        "boundary": "after_noisy_writer",
        "executable_sha256": receipt["executable_sha256"],
        "input_manifest_sha256": receipt["inputs_sha256"],
        "input_payload_sha256": receipt["payload_sha256"],
        "conditioning_manifest_sha256": receipt["conditioning_sha256"],
        "model_index_sha256": receipt["model_index_sha256"],
        "upstream": receipt["upstream_revision"],
        "segments": SEGMENTS,
        "next_segment": next_segment,
        "steps": STEPS_PER_SEGMENT,
        "text_cache_policy": "once_per_scene",
        "history_dtype": "F32",
        "valid_text_tokens": original_valid_text_tokens,
        "storage_text_tokens": ((original_valid_text_tokens + 63) // 64) * 64,
        "video_tokens": 4320,
        "layers": 54,
        "heads": 16,
        "head_dim": 128,
        "timeline_shape": TIMELINE_SHAPE,
        "shift": 5,
        "cache_sigma": 0.206897,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            _fail(f"fresh after-A HNRS base contract does not match receipt field {key}")
    implementation_sha = compact._sha(
        contract.get("implementation_sha256"), "HNRS implementation_sha256"
    )
    if implementation_sha != expected_implementation_sha256:
        _fail("fresh after-A HNRS source digest does not match the current native implementation")


def _validate_state_chains(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    parent_a = arms["parent_a"][0]
    original = arms["original_control"][0]
    edited_control = arms["edited_control"][0]
    edited_chain = arms["edited_restart_chain"]
    parent_state = parent_a["state"]
    if parent_state is None:
        _fail("fresh parent A must save the exact after-A checkpoint")
    if parent_state["contract_document"].get("next_segment") != 1:
        _fail("fresh parent A checkpoint is not bound to the after-A boundary")
    for run in (original, edited_control, edited_chain[0]):
        if run["receipt"]["state_input_sha256"] != parent_state["sha256"]:
            _fail("start-after-A run must consume the fresh same-executable parent A checkpoint")
        if run["receipt"]["contract_sha256"] != parent_state["contract_sha256"]:
            _fail("start-after-A run contract does not match the fresh parent A HNRS contract")
    edited_b_state = edited_chain[0]["state"]
    if edited_b_state is None:
        _fail("edited B checkpoint interval must save its after-B state")
    if edited_b_state["contract_document"].get("schema") != EDIT_BOUNDARY_SCHEMA:
        _fail("edited B checkpoint does not carry the prompt-specific boundary schema")
    if edited_chain[1]["receipt"]["state_input_sha256"] != edited_b_state["sha256"]:
        _fail("edited C restart parent state does not match the edited B checkpoint")
    if edited_chain[1]["receipt"]["contract_sha256"] != edited_b_state["contract_sha256"]:
        _fail("edited C restart contract does not match the edited B HNRS contract")
    return {
        "parent_a": {
            "interval": [0, 1],
            "state_input_sha256": parent_a["receipt"]["state_input_sha256"],
            "state": parent_state,
        },
        "original_control": {
            "interval": [1, 3],
            "state_input_sha256": original["receipt"]["state_input_sha256"],
        },
        "edited_control": {
            "interval": [1, 3],
            "state_input_sha256": edited_control["receipt"]["state_input_sha256"],
        },
        "edited_restart_chain": [
            {
                "interval": list(run["interval"]),
                "state_input_sha256": run["receipt"]["state_input_sha256"],
                "state": run["state"],
            }
            for run in edited_chain
        ],
        "fresh_parent_a_used_by_controls": True,
        "edited_b_linked_to_c_restart": True,
    }


def _validate_timeline_prefix_and_suffix(
    arms: dict[str, list[dict[str, Any]]], source_bank: bytes
) -> dict[str, Any]:
    parent_a = arms["parent_a"][0]
    baseline_a = compact._time_slice(
        parent_a["timeline"], TIMELINE_SHAPE, 0, SEGMENT_LATENTS
    )
    for arm_runs in arms.values():
        for run in arm_runs:
            actual_a = compact._time_slice(
                run["timeline"], TIMELINE_SHAPE, 0, SEGMENT_LATENTS
            )
            if actual_a != baseline_a:
                _fail(f"{run['arm']} timeline changed already-generated A bytes")
            stop = run["interval"][1]
            actual_suffix = compact._time_slice(
                run["timeline"], TIMELINE_SHAPE,
                stop * SEGMENT_LATENTS, TIMELINE_SHAPE[2],
            )
            source_suffix = compact._time_slice(
                source_bank, TIMELINE_SHAPE,
                stop * SEGMENT_LATENTS, TIMELINE_SHAPE[2],
            )
            if actual_suffix != source_suffix:
                _fail(f"{run['arm']} mutated the future/source-bank suffix")
    return {
        "a_prefix_bitwise_unchanged": True,
        "a_prefix_bytes": len(baseline_a),
        "future_suffix_matches_source_bank": True,
    }


def _segment_timeline(run: dict[str, Any], segment: int) -> bytes:
    start = segment * SEGMENT_LATENTS
    return compact._time_slice(run["timeline"], TIMELINE_SHAPE, start, start + SEGMENT_LATENTS)


def _compare_segment(
    left: dict[str, Any], right: dict[str, Any], segment: int, label: str
) -> tuple[dict[str, Any], bool]:
    steps: dict[str, Any] = {}
    changed = False
    for step in range(1, STEPS_PER_SEGMENT + 1):
        key = f"segment{segment}-step{step}"
        result = resume._compare_f32(
            left["steps"][key][0], right["steps"][key][0], f"{label} {key}"
        )
        steps[str(step)] = result
        changed = changed or not result["bitwise_equal"]
    timeline = resume._compare_f32(
        _segment_timeline(left, segment), _segment_timeline(right, segment),
        f"{label} timeline segment {segment}",
    )
    changed = changed or not timeline["bitwise_equal"]
    return {"steps": steps, "timeline": timeline}, changed


def _compare_edit_effect(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    original = arms["original_control"][0]
    edited = arms["edited_control"][0]
    changed_by_segment: dict[str, bool] = {}
    comparisons: dict[str, Any] = {}
    for segment in (TARGET_SEGMENT, TARGET_SEGMENT + 1):
        report, changed = _compare_segment(
            original, edited, segment, "prompt-edited control vs original control"
        )
        comparisons[str(segment)] = report
        changed_by_segment[str(segment)] = changed
    if not changed_by_segment[str(TARGET_SEGMENT)]:
        _fail("prompt edit produced no nonvacuous B output change")
    if not changed_by_segment[str(TARGET_SEGMENT + 1)]:
        _fail("prompt edit produced no nonvacuous C descendant output change")
    return {
        "b_changed_from_original_control": changed_by_segment[str(TARGET_SEGMENT)],
        "c_changed_from_original_control": changed_by_segment[str(TARGET_SEGMENT + 1)],
        "segments": comparisons,
        "scope": "observed bitwise B/C output differences only; no quality claim",
    }


def _compare_edited_restart(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    control = arms["edited_control"][0]
    restart_by_segment = {
        segment: run
        for run in arms["edited_restart_chain"]
        for segment in range(*run["interval"])
    }
    per_segment: dict[str, Any] = {}
    all_steps_bitwise = True
    all_steps_within = True
    all_timeline_bitwise = True
    all_timeline_within = True
    for segment in (TARGET_SEGMENT, TARGET_SEGMENT + 1):
        comparison, _ = _compare_segment(
            control, restart_by_segment[segment], segment,
            "edited uninterrupted vs split restart",
        )
        step_results = list(comparison["steps"].values())
        all_steps_bitwise = all_steps_bitwise and all(
            result["bitwise_equal"] for result in step_results
        )
        all_steps_within = all_steps_within and all(
            result["within_tolerance"] for result in step_results
        )
        timeline_result = comparison["timeline"]
        all_timeline_bitwise = all_timeline_bitwise and timeline_result["bitwise_equal"]
        all_timeline_within = all_timeline_within and timeline_result["within_tolerance"]
        per_segment[str(segment)] = comparison
    full_timeline = resume._compare_f32(
        control["timeline"], arms["edited_restart_chain"][-1]["timeline"],
        "edited uninterrupted vs split restart full timeline",
    )
    if not all_steps_within or not all_timeline_within or not full_timeline["within_tolerance"]:
        _fail("edited uninterrupted vs split restart outputs exceed the existing 2e-5 tolerance")
    return {
        "within_tolerance": True,
        "all_steps_within_tolerance": all_steps_within,
        "all_steps_bitwise_equal": all_steps_bitwise,
        "all_timeline_segments_within_tolerance": all_timeline_within,
        "all_timeline_segments_bitwise_equal": all_timeline_bitwise,
        "full_timeline": full_timeline,
        "segments": per_segment,
    }


def _artifact_report(arms: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        arm: [
            {
                "interval": list(run["interval"]),
                "receipt_path": str(run["receipt_path"]),
                "timeline": run["timeline_meta"],
                "steps": {
                    key: metadata
                    for key, (_, metadata) in sorted(run["steps"].items())
                },
                "state": run["state"],
            }
            for run in runs
        ]
        for arm, runs in arms.items()
    }


def compare_index(index_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Authenticate and compare the five-run prompt-edit matrix."""
    path = compact._absolute_file(str(Path(index_path).absolute()), "matrix index")
    index = resume._read_json(path, label="matrix index")
    expected_keys = {
        "schema", "segments", "steps_per_segment", "prompt_edit", "arms",
        "source_bank", "prompt_inputs",
    }
    if set(index) != expected_keys or index.get("schema") != INDEX_SCHEMA:
        _fail("matrix index schema or key inventory is unsupported")
    if type(index.get("segments")) is not int or index["segments"] != SEGMENTS:
        _fail("matrix must contain exactly the three active A/B/C segments")
    if type(index.get("steps_per_segment")) is not int or index["steps_per_segment"] != STEPS_PER_SEGMENT:
        _fail("matrix must include all four Euler steps per segment")

    source_bank, source_metadata = compact._load_source_bank(index)
    if source_metadata["shape"] != TIMELINE_SHAPE:
        _fail("source bank must contain six resident segments of admitted DiT geometry")
    if source_metadata["manifest_path"] != str(Path(index["source_bank"]["manifest_path"]).resolve(strict=True)):
        _fail("source bank manifest path is not canonical")
    original_prompt_sha = _base_prompt_sha256(source_metadata)
    prompt_inputs = _load_prompt_inputs(index["prompt_inputs"])
    expected_identity = _prompt_edit_identity(
        manifest_sha256=prompt_inputs["manifest_sha256"],
        payload_sha256=prompt_inputs["payload_sha256"],
        prompt_sha256=prompt_inputs["prompt_sha256"],
        valid_text_tokens=prompt_inputs["valid_text_tokens"],
    )
    if index.get("prompt_edit") != expected_identity:
        _fail("matrix prompt_edit identity does not match the authenticated conditioned inputs")
    if prompt_inputs["prompt_sha256"] == original_prompt_sha:
        _fail("matrix prompt inputs repeat the original prompt rather than editing it")
    if type(index.get("prompt_edit")) is not dict or set(index["prompt_edit"]) != set(expected_identity):
        _fail("matrix prompt_edit identity fields are unsupported")
    original_valid_tokens = compact._int(
        source_metadata.get("original_valid_text_tokens"),
        "original valid_text_tokens",
        minimum=1,
    )
    source_metadata["original_prompt_sha256"] = original_prompt_sha
    source_metadata["original_valid_text_tokens"] = original_valid_tokens

    arms = _load_arms(index, source_metadata, prompt_inputs, original_prompt_sha)
    identity = _validate_identities(arms, source_metadata)
    contracts = _validate_contracts(arms, prompt_inputs)
    state_chains = _validate_state_chains(arms)
    prefix_and_future = _validate_timeline_prefix_and_suffix(arms, source_bank)
    edit_effect = _compare_edit_effect(arms)
    edited_restart = _compare_edited_restart(arms)

    return {
        "schema": REPORT_SCHEMA,
        "passed": True,
        "scope": (
            "three-segment native DiT prompt-edit continuation; authenticates the separate conditioned "
            "Qwen v2 input bundle and state26 crop, then compares receipts, HNRS contract headers, "
            "and F32 captures; it authenticates ancillary tensors in the conditioned bundle but does "
            "not introspect which tensors the executable consumed or establish upstream/full-pixel quality"
        ),
        "checkpoint_validation": (
            "Authenticates each complete HNRS file SHA, fixed header envelope, record extents, contract "
            "identity and parent-state linkage; it does not hash each embedded KV record or decode KV/timeline payload semantics."
        ),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "identity": identity,
        "source_bank": source_metadata,
        "prompt_edit": expected_identity,
        "prompt_inputs": {
            "manifest_path": prompt_inputs["manifest_path"],
            "manifest_sha256": prompt_inputs["manifest_sha256"],
            "payload_path": prompt_inputs["payload_path"],
            "payload_sha256": prompt_inputs["payload_sha256"],
            "prompt_sha256": prompt_inputs["prompt_sha256"],
            "valid_text_tokens": prompt_inputs["valid_text_tokens"],
            "qwen_crop_sha256": prompt_inputs["qwen_crop_sha256"],
            "qwen_state_index": 26,
            "qwen_crop_tokens": 1280,
        },
        "text_profiles": {
            "receipt_valid_text_tokens": original_valid_tokens,
            "original": {
                "segment_text_tokens": [original_valid_tokens] * SEGMENTS,
                "segment_cache_text": [True, False, False],
            },
            "edited": {
                "segment_text_tokens": [original_valid_tokens, prompt_inputs["valid_text_tokens"], original_valid_tokens],
                "segment_cache_text": [True, True, True],
            },
            "query_workspace_text_tokens": max(original_valid_tokens, prompt_inputs["valid_text_tokens"]),
        },
        "intervals": {
            arm: [list(run["interval"]) for run in runs]
            for arm, runs in arms.items()
        },
        "metrics": {
            arm: [run["receipt"]["metrics"] for run in runs]
            for arm, runs in arms.items()
        },
        "contracts": contracts,
        "state_chains": state_chains,
        "a_prefix_and_future": prefix_and_future,
        "edit_effect": edit_effect,
        "edited_restart": edited_restart,
        "artifacts": _artifact_report(arms),
        "finite": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Five-run prompt-edit matrix JSON")
    parser.add_argument("--output", required=True, help="Fresh JSON report path")
    args = parser.parse_args(argv)
    try:
        report = compare_index(args.index)
        output = Path(args.output).absolute()
        resume._write_fresh_json(output, report)
    except (OSError, ValueError) as error:
        print(f"prompt-edit resume reference rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": report["passed"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
