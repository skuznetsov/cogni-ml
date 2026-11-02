#!/usr/bin/env python3
"""Export a CPU-only action span profile from a pinned Hy-Nomas prompt contract."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Optional


UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
UPSTREAM_SOURCE_SHA256 = {
    "hymm/pipeline/base.py": "77a039514ff494cb496a194c60ca3270f09c15b194bda25448a036a6fcc79582",
    "hymm/inference.py": "7b0a8152c9ab5c1682dc7e4aee41d7fb3b2f3416873b3b4a6795a45e65b82e6d",
    "hymm/models/text_encoder/encoder.py": "aa84b366d6e0a5cf9c73e591e6827746e8cab97852a112d1b4783dc7c2d507f1",
    "hymm/models/dit/nomas_cpt.py": "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee",
    "configs/constants.py": "b4ab0055d0a2fade687f012f0e305915f5305df58dc82e2ad8bde25ef4236af9",
}
PROFILE_SCHEMA = "cogni_ml.hy_nomas_action_conditioning/v1"
TEMPLATE_NAME = "li-dit-encode-video-json"
TEMPLATE_SHA256 = "3e7b69bdf2a057bf2d6a8322dfb8ad619e686e4e8240b65f29e94492bee0c4a6"
TEXT_LENGTH = 1280
MAX_VALID_TEXT_TOKENS = 1000
SEGMENT_START_SECONDS = 5.0
SEGMENT_END_SECONDS = 10.0
SEGMENT_LATENT_FRAMES = 30
TOKENS_PER_FRAME = 144
VIDEO_FPS = 24
TEMPORAL_RATIO = 4
MAX_INPUT_BYTES = 16 * 1024 * 1024
SHA_RE = re.compile(r"\A[0-9a-f]{64}\Z")
ACTION_WRAPPER_RE = re.compile(r"\A<Action1>(?P<body>.*?)</Action1>\.\Z", re.DOTALL)
VIDEO_JSON_TEMPLATE_SHA256 = TEMPLATE_SHA256


class ActionConditioningError(ValueError):
    """The input does not satisfy the bounded action-conditioning contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ActionConditioningError("timestamps must be numeric seconds or canonical strings")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ActionConditioningError("timestamps must be canonical strings")
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(value)
    except (TypeError, ValueError) as error:
        raise ActionConditioningError("invalid action timestamp") from error


def _format_relative_timestamp(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def validate_t1_prompt(prompt: str) -> dict[str, Any]:
    """Validate the serialized, segment-local, one-action T1 prompt."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > 64 * 1024:
        raise ActionConditioningError("T1 prompt must be nonempty UTF-8 JSON under 64 KiB")
    try:
        document = json.loads(prompt)
    except (json.JSONDecodeError, UnicodeEncodeError) as error:
        raise ActionConditioningError("T1 prompt is not valid JSON") from error
    if not isinstance(document, dict):
        raise ActionConditioningError("T1 prompt root must be an object")
    segments = document.get("segments")
    if not isinstance(segments, list) or len(segments) != 1 or not isinstance(segments[0], dict):
        raise ActionConditioningError("T1 prompt must contain exactly one T1 segment")
    segment = segments[0]
    try:
        segment_start = _parse_timestamp(segment.get("start_time"))
        segment_end = _parse_timestamp(segment.get("end_time"))
    except ActionConditioningError as error:
        raise ActionConditioningError("T1 segment has invalid global time boundaries") from error
    if segment_start != SEGMENT_START_SECONDS or segment_end != SEGMENT_END_SECONDS:
        raise ActionConditioningError("T1 starts at 5s and ends at 10s in the global timeline")

    actions = segment.get("contained_actions")
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        raise ActionConditioningError("T1 must contain exactly one contained action")
    action = actions[0]
    if set(action) != {"start_time", "end_time", "action_description"}:
        raise ActionConditioningError("action must contain only start_time, end_time, and action_description")
    for key in ("start_time", "end_time"):
        if not isinstance(action[key], str):
            raise ActionConditioningError("action timestamps must be segment-local strings")
        parsed = _parse_timestamp(action[key])
        if action[key] != _format_relative_timestamp(parsed):
            raise ActionConditioningError("action timestamps must use normalized segment-local format")
    action_start = _parse_timestamp(action["start_time"])
    action_end = _parse_timestamp(action["end_time"])
    segment_duration = segment_end - segment_start
    if not (0.0 <= action_start < action_end <= segment_duration):
        raise ActionConditioningError("action timestamps must be segment-local and bounded to 0–5 seconds")

    description = action.get("action_description")
    if not isinstance(description, str):
        raise ActionConditioningError("action_description must be text")
    match = ACTION_WRAPPER_RE.fullmatch(description)
    if match is None or description.count("<Action") != 1 or description.count("</Action") != 1:
        raise ActionConditioningError("action description must use one paired literal <Action1> wrapper")
    body = match.group("body")
    if "<|Subject1|>" not in body or "<|Subject2|>" in body:
        raise ActionConditioningError("Action1 must reference only the upstream <|Subject1|> identifier")
    if re.search(r"\bhead\b", body, flags=re.IGNORECASE) is None or re.search(
        r"\bturn(?:s|ed|ing)?\b", body, flags=re.IGNORECASE
    ) is None:
        raise ActionConditioningError("the bounded experiment requires Alex's head-turn action")
    if "Resume" not in prompt:
        raise ActionConditioningError("Resume must remain present as T1 scene context")
    return {
        "action_id": 1,
        "subject_id": "<|Subject1|>",
        "start_time_local_seconds": action_start,
        "end_time_local_seconds": action_end,
        "action_description": description,
        "time_origin": "segment_local_seconds",
        "source_segment_start_seconds": segment_start,
        "source_segment_end_seconds": segment_end,
    }


def _upstream_source_record(upstream_root: Path) -> tuple[Path, dict[str, str]]:
    root = upstream_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ActionConditioningError("upstream source root is not a directory")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ActionConditioningError("upstream source checkout is not the pinned a31 revision")
    hashes: dict[str, str] = {}
    for relative, expected in UPSTREAM_SOURCE_SHA256.items():
        source_path = root / relative
        actual = _sha256_file(source_path)
        if actual != expected:
            raise ActionConditioningError(f"pinned upstream source digest does not match: {relative}")
        hashes[relative] = actual
    return root, hashes


def _exec_ast_nodes(path: Path, wanted_functions: set[str], wanted_assignments: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected: list[ast.stmt] = []
    found_functions: set[str] = set()
    found_assignments: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            selected.append(node)
            found_functions.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & wanted_assignments:
                selected.append(node)
                found_assignments.update(names & wanted_assignments)
    if found_functions != wanted_functions or found_assignments != wanted_assignments:
        raise ActionConditioningError(f"upstream AST is missing expected helpers from {path.name}")
    extracted = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace


def _shape_helpers(upstream_root: Path) -> dict[str, Any]:
    root, _ = _upstream_source_record(upstream_root)
    inference_path = root / "hymm/inference.py"
    namespace: dict[str, Any] = {
        "Any": Any,
        "Dict": dict,
        "List": list,
        "Optional": Optional,
        "Tuple": tuple,
        "json": json,
        "re": re,
    }
    return _exec_ast_nodes(
        inference_path,
        {
            "_remove_keys_from_dict",
            "_keep_keys_in_dict",
            "_parse_timestamp",
            "_resolve_subject_ids",
            "_format_relative_timestamp",
            "_normalize_contained_action_timestamps",
            "_strip_camera_text_blocks_for_camera_segment",
            "_build_whole_item_prompts",
        },
        {
            "PROMPT_SEPARATOR",
            "_CAMERA_TEXT_BLOCK_RE",
            "_BENCH_META_TOP_LEVEL_KEYS",
            "_BENCH_META_SEGMENT_KEYS",
            "_BENCH_STRIP_KEYS",
        },
        namespace,
    )


def shape_reference_prompts(upstream_root: Path, caption: dict[str, Any], followup: dict[str, Any]) -> tuple[str, str]:
    """Shape T0 and T1 using the exact pinned inference AST (CPU only)."""
    if not isinstance(caption, dict) or not isinstance(followup, dict):
        raise ActionConditioningError("caption and follow-up inputs must be JSON objects")
    caption_segments = caption.get("segments")
    followup_segments = followup.get("segments")
    if not isinstance(caption_segments, list) or len(caption_segments) != 1:
        raise ActionConditioningError("caption input must contain one T0 segment")
    if not isinstance(followup_segments, list) or len(followup_segments) != 1:
        raise ActionConditioningError("follow-up input must contain one T1 segment")
    combined_item = {key: value for key, value in caption.items() if key != "segments"}
    combined_item["segments"] = [caption_segments[0], followup_segments[0]]
    helpers = _shape_helpers(Path(upstream_root))
    prompts, count = helpers["_build_whole_item_prompts"](
        combined_item,
        [caption_segments[0], followup_segments[0]],
        keep_keywords=None,
        resolve_subject_ids=False,
        global_only_in_first_segment=True,
        no_desc_long_after_first=True,
        normalize_action_timestamp=True,
    )
    shaped = prompts.split(helpers["PROMPT_SEPARATOR"])
    if count != 2 or len(shaped) != 2:
        raise ActionConditioningError("pinned prompt shaper did not produce exactly T0 and T1")
    return shaped[0], shaped[1]


def _text_encoder_methods(encoder_path: Path) -> type:
    module = ast.parse(encoder_path.read_text(encoding="utf-8"), filename=str(encoder_path))
    source_class = next((node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "TextEncoder"), None)
    if source_class is None:
        raise ActionConditioningError("pinned encoder source does not define TextEncoder")
    wanted = {"apply_text_to_template", "calculate_crop_start", "text2tokens"}
    methods = [node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in methods} != wanted or len(methods) != len(wanted):
        raise ActionConditioningError("pinned TextEncoder prompt methods do not match the expected contract")
    extracted = ast.Module(body=[ast.ClassDef(name="PinnedTextEncoderMethods", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace: dict[str, Any] = {"deepcopy": copy.deepcopy}
    exec(compile(extracted, str(encoder_path), "exec"), namespace)
    return namespace["PinnedTextEncoderMethods"]


def _video_json_template(constants_path: Path) -> dict[str, Any]:
    module = ast.parse(constants_path.read_text(encoding="utf-8"), filename=str(constants_path))
    assignments = [
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id.startswith("PROMPT_TEMPLATE") for target in node.targets)
    ]
    namespace: dict[str, Any] = {}
    extracted = ast.Module(body=assignments, type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(constants_path), "exec"), namespace)
    template = namespace.get("PROMPT_TEMPLATE", {}).get(TEMPLATE_NAME)
    if not isinstance(template, dict) or template.get("crop_start") != -1:
        raise ActionConditioningError("pinned video JSON template is missing or has an unexpected crop prepass")
    if _canonical_json_sha256(template["template"]) != TEMPLATE_SHA256:
        raise ActionConditioningError("pinned video JSON template digest does not match")
    return copy.deepcopy(template)


def _base_pipeline_methods(base_path: Path) -> type:
    module = ast.parse(base_path.read_text(encoding="utf-8"), filename=str(base_path))
    source_class = next((node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "BasePipeline"), None)
    if source_class is None:
        raise ActionConditioningError("pinned base.py does not define BasePipeline")
    wanted = {"_compute_action_text_offsets", "_compute_action_video_windows"}
    methods = [node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in methods} != wanted or len(methods) != len(wanted):
        raise ActionConditioningError("pinned BasePipeline action methods do not match the expected contract")
    extracted = ast.Module(body=[ast.ClassDef(name="PinnedBasePipelineActionMethods", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace: dict[str, Any] = {"List": list, "Optional": Optional}
    exec(compile(extracted, str(base_path), "exec"), namespace)
    return namespace["PinnedBasePipelineActionMethods"]


def _install_pinned_timestamp_parser(inference_path: Path):
    namespace = {"Any": Any}
    _exec_ast_nodes(inference_path, {"_parse_timestamp"}, set(), namespace)
    fake_module = ModuleType("hymm.inference")
    fake_module._parse_timestamp = namespace["_parse_timestamp"]
    old = sys.modules.get("hymm.inference")
    sys.modules["hymm.inference"] = fake_module
    return old


def _token_contract(document: dict[str, Any], tokenizer_root: Path) -> tuple[Any, dict[str, Any]]:
    qwen = document.get("qwen_source")
    geometry = document.get("geometry")
    if not isinstance(qwen, dict) or not isinstance(geometry, dict):
        raise ActionConditioningError("conditioning inputs omit the Qwen source or geometry record")
    contract = qwen.get("prompt_contract")
    if not isinstance(contract, dict) or contract.get("schema") != "cogni_ml.hy_nomas_qwen25_video_prompt_reference":
        raise ActionConditioningError("conditioning inputs omit the exact Qwen prompt contract")
    prompt = document.get("prompt")
    if not isinstance(prompt, str) or prompt != contract.get("prompt"):
        raise ActionConditioningError("conditioning prompt differs from its Qwen prompt contract")
    prompt_sha = _sha256_bytes(prompt.encode("utf-8"))
    if prompt_sha != contract.get("prompt_sha256") or prompt_sha != qwen.get("prompt_sha256"):
        raise ActionConditioningError("conditioning prompt SHA-256 differs from its Qwen contract")
    action = validate_t1_prompt(prompt)

    crop_start = contract.get("crop_start")
    text_length = contract.get("text_length")
    if isinstance(crop_start, bool) or not isinstance(crop_start, int) or crop_start <= 0:
        raise ActionConditioningError("Qwen prompt contract crop_start is invalid")
    if text_length != TEXT_LENGTH or contract.get("template_name") != TEMPLATE_NAME or contract.get("template_sha256") != TEMPLATE_SHA256:
        raise ActionConditioningError("Qwen prompt contract does not use the pinned video JSON template and 1280-token crop")
    expected_crop = [crop_start, crop_start + text_length]
    if qwen.get("crop") != expected_crop or geometry.get("qwen_crop") != expected_crop:
        raise ActionConditioningError("Qwen crop values disagree across conditioning metadata")
    if qwen.get("crop_tokens") != text_length or qwen.get("valid_tokens") != geometry.get("qwen_valid_tokens"):
        raise ActionConditioningError("Qwen valid token and crop metadata disagree")

    raw_ids = contract.get("token_ids")
    raw_mask = contract.get("attention_mask")
    cropped_ids = contract.get("cropped_token_ids")
    cropped_mask = contract.get("cropped_attention_mask")
    if not all(isinstance(value, list) for value in (raw_ids, raw_mask, cropped_ids, cropped_mask)):
        raise ActionConditioningError("Qwen prompt contract must retain the actual token IDs and masks")
    if len(raw_ids) != text_length + crop_start or len(raw_mask) != len(raw_ids):
        raise ActionConditioningError("raw Qwen token and attention mask extents do not match crop geometry")
    if len(cropped_ids) != text_length or len(cropped_mask) != text_length:
        raise ActionConditioningError("cropped Qwen token and attention mask extents must be 1280")
    if raw_ids[crop_start:] != cropped_ids or raw_mask[crop_start:] != cropped_mask:
        raise ActionConditioningError("Qwen cropped IDs or attention mask differ from the raw suffix")
    if _canonical_json_sha256(raw_ids) != contract.get("raw_token_ids_sha256"):
        raise ActionConditioningError("raw Qwen token IDs do not match their recorded digest")
    if _canonical_json_sha256(raw_mask) != contract.get("raw_mask_sha256"):
        raise ActionConditioningError("raw Qwen attention mask does not match its recorded digest")
    if _canonical_json_sha256(cropped_ids) != contract.get("cropped_token_ids_sha256"):
        raise ActionConditioningError("cropped Qwen token IDs do not match their recorded digest")
    if _canonical_json_sha256(cropped_mask) != contract.get("cropped_attention_mask_sha256"):
        raise ActionConditioningError("cropped Qwen attention mask does not match its recorded digest")
    if any(value not in (0, 1) for value in cropped_mask):
        raise ActionConditioningError("cropped Qwen attention mask must contain only zero and one")
    valid_tokens = sum(cropped_mask)
    if not 1 <= valid_tokens <= MAX_VALID_TEXT_TOKENS or valid_tokens != qwen.get("valid_tokens"):
        raise ActionConditioningError("valid text token count must match the cropped mask and stay at or below 1000")
    first_padding = next((index for index, value in enumerate(cropped_mask) if value == 0), len(cropped_mask))
    if any(cropped_mask[first_padding:]):
        raise ActionConditioningError("cropped Qwen attention mask must be right padded")

    prompt_contract_tokenizer = contract.get("tokenizer")
    padding = contract.get("padding")
    assets = contract.get("tokenizer_assets")
    if not isinstance(prompt_contract_tokenizer, dict) or not isinstance(padding, dict) or not isinstance(assets, dict):
        raise ActionConditioningError("Qwen contract omits tokenizer identity or assets")
    pad_token_id = padding.get("pad_token_id")
    if (padding.get("side") != "right" or isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, int) or pad_token_id < 0):
        raise ActionConditioningError("Qwen contract padding metadata is inconsistent")
    root = tokenizer_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ActionConditioningError("tokenizer root is not a directory")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        record = assets.get(name)
        asset_path = root / name
        if not isinstance(record, dict) or not asset_path.is_file() or _sha256_file(asset_path) != record.get("sha256"):
            raise ActionConditioningError(f"local tokenizer asset does not match the prompt contract: {name}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import transformers
        import tokenizers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ActionConditioningError("local tokenizer runtime is unavailable") from error
    runtime = prompt_contract_tokenizer.get("runtime", {})
    if str(transformers.__version__) != str(runtime.get("transformers_version")):
        raise ActionConditioningError("installed Transformers version differs from the prompt contract")
    if str(tokenizers.__version__) != str(runtime.get("tokenizers", {}).get("version")):
        raise ActionConditioningError("installed tokenizers version differs from the prompt contract")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False, padding_side="right"
        )
    except Exception as error:
        raise ActionConditioningError("cannot load the authenticated local Qwen tokenizer") from error
    if tokenizer.padding_side != padding.get("side") or tokenizer.pad_token_id != pad_token_id:
        raise ActionConditioningError("loaded Qwen tokenizer padding differs from the prompt contract")
    return tokenizer, {
        "contract": contract,
        "action": action,
        "crop_start": crop_start,
        "text_length": text_length,
        "valid_text_tokens": valid_tokens,
        "raw_token_ids": raw_ids,
        "raw_attention_mask": raw_mask,
        "cropped_token_ids": cropped_ids,
        "cropped_attention_mask": cropped_mask,
        "tokenizer_assets": assets,
        "tokenizer": prompt_contract_tokenizer,
    }


def _derive_action_offsets_and_windows(
    upstream_root: Path, prompt: str, tokenizer: Any, contract: dict[str, Any],
) -> tuple[list[int], list[list[int]] | None, list[int], list[int]]:
    root, _ = _upstream_source_record(upstream_root)
    template = _video_json_template(root / "configs/constants.py")
    encoder_type = _text_encoder_methods(root / "hymm/models/text_encoder/encoder.py")
    text_encoder = encoder_type()
    text_encoder.text_encoder_type = "qwen-2.5vl-7b"
    text_encoder.tokenizer = tokenizer
    text_encoder.use_template = True
    text_encoder.use_video_template = True
    text_encoder.prompt_template_video = template
    try:
        text_encoder.text2tokens("a photo of a cat", data_type="video", max_length=256)
    except Exception as error:
        raise ActionConditioningError("pinned TextEncoder crop prepass failed") from error
    crop_start = text_encoder.prompt_template_video.get("crop_start")
    if crop_start != contract["crop_start"]:
        raise ActionConditioningError("pinned TextEncoder crop prepass differs from the conditioning contract")

    pipeline_type = _base_pipeline_methods(root / "hymm/pipeline/base.py")
    pipeline = pipeline_type()
    pipeline.args = SimpleNamespace(
        text_len=contract["text_length"],
        inference=SimpleNamespace(video_fps=VIDEO_FPS),
    )
    pipeline.text_encoder = text_encoder
    pipeline.t_ratio = TEMPORAL_RATIO
    try:
        raw_ids = text_encoder.text2tokens([prompt], data_type="video", max_length=TEXT_LENGTH)["input_ids"].squeeze(0).tolist()
        raw_mask = text_encoder.text2tokens([prompt], data_type="video", max_length=TEXT_LENGTH)["attention_mask"].squeeze(0).tolist()
        if raw_ids != contract["raw_token_ids"] or raw_mask != contract["raw_attention_mask"]:
            raise ActionConditioningError("pinned tokenizer output differs from the actual Qwen prompt contract IDs")
        source_offsets = pipeline._compute_action_text_offsets(prompt)
        if not isinstance(source_offsets, list) or len(source_offsets) != 2:
            raise ActionConditioningError("upstream action offset helper did not return one head/action boundary pair")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in source_offsets):
            raise ActionConditioningError("upstream action text offsets must be integer token positions")

        inference_path = root / "hymm/inference.py"
        old_parser = _install_pinned_timestamp_parser(inference_path)
        try:
            windows = pipeline._compute_action_video_windows(prompt, 0, SEGMENT_LATENT_FRAMES)
        finally:
            if old_parser is None:
                sys.modules.pop("hymm.inference", None)
            else:
                sys.modules["hymm.inference"] = old_parser
    except ActionConditioningError:
        raise
    except Exception as error:
        raise ActionConditioningError("pinned upstream action offset/window helper failed") from error
    return list(source_offsets), windows, [int(value) for value in raw_ids], [int(value) for value in raw_mask]


def clamp_source_action_offsets(source_offsets: list[int], valid_text_tokens: int) -> list[int]:
    """Apply the exact visible-query endpoint clamp used by pinned nomas_cpt."""
    if (not isinstance(source_offsets, list) or len(source_offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   or value > TEXT_LENGTH for value in source_offsets)
            or isinstance(valid_text_tokens, bool) or not isinstance(valid_text_tokens, int)
            or not 1 <= valid_text_tokens <= MAX_VALID_TEXT_TOKENS):
        raise ActionConditioningError("action text offsets do not fit the bounded token contract")
    return [min(offset, valid_text_tokens) for offset in source_offsets]


def _verify_windows(windows: list[list[int]] | None, action: dict[str, Any]) -> tuple[bool, list[list[int]] | None]:
    if windows is None:
        full = action["start_time_local_seconds"] == 0.0 and action["end_time_local_seconds"] == 5.0
        if not full:
            raise ActionConditioningError("bounded action windows are missing for a non-full-segment action")
        return False, None
    if not isinstance(windows, list) or not windows:
        raise ActionConditioningError("upstream action windows must tile the 30-frame segment")
    cursor = 0
    normalized: list[list[int]] = []
    action_windows = 0
    for window in windows:
        if not isinstance(window, list) or len(window) != 3:
            raise ActionConditioningError("upstream action window is malformed")
        start, end, action_index = window
        if any(isinstance(value, bool) or not isinstance(value, int) for value in window):
            raise ActionConditioningError("upstream action windows must use integer frame bounds and IDs")
        if start != cursor or end <= start or end > SEGMENT_LATENT_FRAMES:
            raise ActionConditioningError("upstream action windows overlap, have gaps, or exceed the segment")
        if action_index not in (-1, 0):
            raise ActionConditioningError("exactly one action window with upstream action index 0 is admitted")
        action_windows += int(action_index == 0)
        normalized.append([start, end, action_index])
        cursor = end
    if cursor != SEGMENT_LATENT_FRAMES or action_windows != 1:
        raise ActionConditioningError("upstream action windows do not cover the segment and one action")
    return True, normalized


def export_action_conditioning(
    conditioning_inputs: Path,
    conditioning_sha256: str,
    upstream_root: Path,
    tokenizer_root: Path,
) -> dict[str, Any]:
    if not SHA_RE.fullmatch(conditioning_sha256):
        raise ActionConditioningError("conditioning manifest SHA-256 must be lowercase hexadecimal")
    input_path = conditioning_inputs.expanduser().resolve(strict=True)
    if not input_path.is_file() or input_path.stat().st_size > MAX_INPUT_BYTES:
        raise ActionConditioningError("conditioning inputs must be a regular JSON manifest under 16 MiB")
    raw = input_path.read_bytes()
    if _sha256_bytes(raw) != conditioning_sha256:
        raise ActionConditioningError("conditioning manifest SHA-256 differs from --conditioning-sha")
    try:
        document = json.loads(raw.decode("utf-8", errors="strict"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ActionConditioningError("conditioning inputs are not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ActionConditioningError("conditioning inputs root must be a JSON object")
    payload_sha256 = document.get("payload_sha256")
    if not isinstance(payload_sha256, str) or not SHA_RE.fullmatch(payload_sha256):
        raise ActionConditioningError("conditioning payload SHA-256 is missing or malformed")

    root, source_hashes = _upstream_source_record(upstream_root)
    prompt = document.get("prompt")
    if not isinstance(prompt, str):
        raise ActionConditioningError("conditioning prompt is missing")
    tokenizer, contract = _token_contract(document, tokenizer_root)
    source_offsets, windows, token_ids, attention_mask = _derive_action_offsets_and_windows(
        root, prompt, tokenizer, contract
    )
    valid_tokens = contract["valid_text_tokens"]
    effective_offsets = clamp_source_action_offsets(source_offsets, valid_tokens)
    head_end, action_end = effective_offsets
    intra_segment_active, windows = _verify_windows(windows, contract["action"])
    if action_end != valid_tokens or head_end <= 0 or head_end >= action_end:
        raise ActionConditioningError("action text span is empty or exceeds the 1000-token native limit")
    return {
        "schema": PROFILE_SCHEMA,
        "conditioning_manifest_sha256": conditioning_sha256,
        "conditioning_payload_sha256": payload_sha256,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "prompt": prompt,
        "valid_text_tokens": contract["valid_text_tokens"],
        "geometry": {
            "run_segments": 2,
            "segment_index": 1,
            "segment_start_seconds": SEGMENT_START_SECONDS,
            "segment_end_seconds": SEGMENT_END_SECONDS,
            "latent_frames": SEGMENT_LATENT_FRAMES,
            "tokens_per_frame": TOKENS_PER_FRAME,
            "video_tokens": SEGMENT_LATENT_FRAMES * TOKENS_PER_FRAME,
            "text_length": TEXT_LENGTH,
        },
        "token_contract": {
            "schema": contract["contract"].get("schema"),
            "schema_version": contract["contract"].get("schema_version"),
            "prompt_sha256": contract["contract"].get("prompt_sha256"),
            "crop_start": contract["crop_start"],
            "text_length": contract["text_length"],
            "template_name": TEMPLATE_NAME,
            "template_sha256": TEMPLATE_SHA256,
            "valid_text_tokens": contract["valid_text_tokens"],
            "raw_token_ids_sha256": _canonical_json_sha256(token_ids),
            "raw_attention_mask_sha256": _canonical_json_sha256(attention_mask),
            "cropped_token_ids_sha256": _canonical_json_sha256(contract["cropped_token_ids"]),
            "cropped_attention_mask_sha256": _canonical_json_sha256(contract["cropped_attention_mask"]),
            "tokenizer_assets": contract["tokenizer_assets"],
            "tokenizer_identity": {
                "repo_id": contract["tokenizer"].get("repo_id"),
                "revision": contract["tokenizer"].get("revision"),
                "transformers_version": contract["tokenizer"].get("runtime", {}).get("transformers_version"),
                "tokenizers_version": contract["tokenizer"].get("runtime", {}).get("tokenizers", {}).get("version"),
            },
        },
        "source": {
            "upstream_revision": UPSTREAM_REVISION,
            "source_sha256": source_hashes,
            "offset_method": "BasePipeline._compute_action_text_offsets (AST-extracted from pinned base.py)",
            "video_window_method": "BasePipeline._compute_action_video_windows (AST-extracted from pinned base.py)",
            "text_tokenizer_methods": ["TextEncoder.apply_text_to_template", "TextEncoder.text2tokens"],
            "offset_clamp_method": "nomas_cpt.py clamps every source offset with min(offset, valid_current_text_query_tokens)",
            "time_normalizer": "hymm.inference._normalize_contained_action_timestamps (AST-extracted)",
            "time_origin": "segment_local_seconds",
            "resolve_subject_ids": False,
            "action_wrapper_representation": "literal_text_markup_not_token_ids",
        },
        "action": {
            **contract["action"],
            "latent_start": next((int(window[0]) for window in windows or [] if window[2] == 0), 0),
            "latent_end": next((int(window[1]) for window in windows or [] if window[2] == 0), SEGMENT_LATENT_FRAMES),
        },
        "text_offsets": {
            "head_end": head_end,
            "action_start": head_end,
            "action_end": action_end,
            "valid_text_tokens": action_end,
            "source_offsets_raw": source_offsets,
            "effective_offsets": effective_offsets,
            "offset_method_endpoint_note": "BasePipeline returns the raw attention-mask sum; pinned nomas_cpt clamps it with min(offset, valid current text query tokens), yielding the authenticated cropped-mask count",
        },
        "video_windows": windows,
        "intra_segment_active": intra_segment_active,
        "labels": {
            "global": 0,
            "action": 1,
            "padding": -1,
            "video_gap_window_index": -1,
            "query_scope": "global_head_and_matching_action_window",
            "key_scope": "global_history_or_matching_action",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioning-inputs", required=True, type=Path)
    parser.add_argument("--conditioning-sha", required=True)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--tokenizer-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        profile = export_action_conditioning(
            args.conditioning_inputs,
            args.conditioning_sha,
            args.upstream,
            args.tokenizer_root,
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(profile, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        output.write_bytes(encoded)
    except (ActionConditioningError, OSError, subprocess.SubprocessError) as error:
        print(f"action conditioning export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(output),
        "profile_sha256": _sha256_bytes(encoded),
        "conditioning_manifest_sha256": profile["conditioning_manifest_sha256"],
        "conditioning_payload_sha256": profile["conditioning_payload_sha256"],
        "prompt_sha256": profile["prompt_sha256"],
        "valid_text_tokens": profile["valid_text_tokens"],
        "text_offsets": profile["text_offsets"],
        "video_windows": profile["video_windows"],
        "intra_segment_active": profile["intra_segment_active"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
