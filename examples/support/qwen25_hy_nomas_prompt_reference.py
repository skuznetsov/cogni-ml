#!/usr/bin/env python3
"""Produce CPU-only, source-pinned HyNomas Qwen video-prompt inputs."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any


UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_ENCODER_SHA256 = "aa84b366d6e0a5cf9c73e591e6827746e8cab97852a112d1b4783dc7c2d507f1"
UPSTREAM_CONSTANTS_SHA256 = "b4ab0055d0a2fade687f012f0e305915f5305df58dc82e2ad8bde25ef4236af9"
VIDEO_JSON_TEMPLATE_SHA256 = "3e7b69bdf2a057bf2d6a8322dfb8ad619e686e4e8240b65f29e94492bee0c4a6"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "eb6ba55b40ada64977147b52e50a4145ee89ac4f"
TOKENIZER_RUNTIME_TRANSFORMERS_VERSION = "4.57.3"
TEMPLATE_NAME = "li-dit-encode-video-json"
PROMPT_SCHEMA = "cogni_ml.hy_nomas_qwen25_video_prompt_reference"
PROMPT_SCHEMA_VERSION = 1
BASE_SEQUENCE_LENGTH = 1280
MAX_CROP_START = 256
MAX_RAW_SEQUENCE_LENGTH = BASE_SEQUENCE_LENGTH + MAX_CROP_START
MAX_PROMPT_FILE_BYTES = 64 * 1024
VIDEO_MARKERS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|image|>",
    "<|video|>",
)
REQUIRED_VIDEO_MARKER_IDS = {
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
}
TOKENIZER_ASSET_NAMES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _verify_upstream_sources(upstream_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = upstream_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("HyNomas source root is not a directory")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("HyNomas source checkout is not the pinned revision")
    encoder_path = root / "hymm/models/text_encoder/encoder.py"
    constants_path = root / "configs/constants.py"
    if _sha256_file(encoder_path) != UPSTREAM_ENCODER_SHA256:
        raise ValueError("pinned HyNomas encoder source digest does not match")
    if _sha256_file(constants_path) != UPSTREAM_CONSTANTS_SHA256:
        raise ValueError("pinned HyNomas prompt constants digest does not match")
    source_record = {
        "repository_revision": revision,
        "encoder_source": {
            "path": "hymm/models/text_encoder/encoder.py",
            "sha256": UPSTREAM_ENCODER_SHA256,
        },
        "constants_source": {
            "path": "configs/constants.py",
            "sha256": UPSTREAM_CONSTANTS_SHA256,
        },
        "methods_executed_from_ast": [
            "apply_text_to_template",
            "calculate_crop_start",
            "text2tokens",
        ],
    }
    return encoder_path, constants_path, source_record


def _tokenizer_runtime_sources(tokenizer: Any, transformers: Any, tokenizers: Any) -> dict[str, Any]:
    transformers_root = Path(transformers.__file__).resolve().parent
    tokenizer_root = Path(tokenizers.__file__).resolve().parent
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    objects = (AutoTokenizer, type(tokenizer), PreTrainedTokenizerBase, PreTrainedTokenizerFast)
    files: dict[str, Any] = {}
    for obj in objects:
        source_path_text = inspect.getsourcefile(obj)
        if source_path_text is None:
            raise RuntimeError(f"could not locate tokenizer implementation source for {obj}")
        source_path = Path(source_path_text).resolve(strict=True)
        try:
            relative = source_path.relative_to(transformers_root).as_posix()
        except ValueError as error:
            raise RuntimeError("tokenizer source escaped the installed Transformers package") from error
        files[relative] = {"path": str(source_path), "sha256": _sha256_file(source_path)}
    tokenizer_runtime: dict[str, Any] = {
        "version": str(tokenizers.__version__),
        "package_source": {
            "path": str(Path(tokenizers.__file__).resolve()),
            "sha256": _sha256_file(Path(tokenizers.__file__).resolve()),
        },
    }
    native_candidates = sorted(
        list(tokenizer_root.glob("*_tokenizers*.so"))
        + list(tokenizer_root.glob("*_tokenizers*.dylib"))
        + list(tokenizer_root.glob("_tokenizers*.so"))
        + list(tokenizer_root.glob("_tokenizers*.dylib"))
    )
    if native_candidates:
        native = native_candidates[0]
        tokenizer_runtime["native_extension"] = {
            "path": str(native),
            "sha256": _sha256_file(native),
        }
    return {
        "python_version": platform.python_version(),
        "transformers_version": str(transformers.__version__),
        "transformers_sources": files,
        "tokenizers": tokenizer_runtime,
    }


def _source_methods(encoder_path: Path) -> type:
    module = ast.parse(encoder_path.read_text(encoding="utf-8"), filename=str(encoder_path))
    wanted = {"apply_text_to_template", "calculate_crop_start", "text2tokens"}
    source_class = next(
        (
            node for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "TextEncoder"
        ),
        None,
    )
    if source_class is None:
        raise ValueError("pinned HyNomas encoder does not define TextEncoder")
    methods = [
        node for node in source_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    if {node.name for node in methods} != wanted or len(methods) != len(wanted):
        raise ValueError("pinned HyNomas encoder does not expose the expected prompt methods")
    class_node = ast.ClassDef(
        name="PinnedHyNomasTextEncoderMethods",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    extracted = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace: dict[str, Any] = {"deepcopy": __import__("copy").deepcopy}
    exec(compile(extracted, str(encoder_path), "exec"), namespace)
    return namespace["PinnedHyNomasTextEncoderMethods"]


def _source_templates(constants_path: Path) -> dict[str, Any]:
    module = ast.parse(constants_path.read_text(encoding="utf-8"), filename=str(constants_path))
    wanted = {
        "PROMPT_TEMPLATE_ENCODE_VIDEO_JSON",
        "PROMPT_TEMPLATE_ENCODE_IMAGE_JSON",
        "PROMPT_TEMPLATE",
    }
    assignments: dict[str, ast.Assign] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        for name in names:
            if name in wanted:
                assignments[name] = node
    if set(assignments) != wanted:
        raise ValueError("pinned HyNomas constants do not define the expected JSON templates")
    # The two JSON lists and PROMPT_TEMPLATE map refer to other module globals.
    all_template_assignments = [
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id.startswith("PROMPT_TEMPLATE")
            for target in node.targets
        )
    ]
    selected = ast.Module(body=all_template_assignments, type_ignores=[])
    ast.fix_missing_locations(selected)
    namespace: dict[str, Any] = {}
    exec(compile(selected, str(constants_path), "exec"), namespace)
    templates = namespace.get("PROMPT_TEMPLATE")
    video_json = namespace.get("PROMPT_TEMPLATE_ENCODE_VIDEO_JSON")
    if not isinstance(templates, dict) or not isinstance(video_json, list):
        raise ValueError("pinned HyNomas video JSON prompt template is malformed")
    selected_template = templates.get(TEMPLATE_NAME)
    if not isinstance(selected_template, dict) or selected_template.get("template") != video_json:
        raise ValueError("video JSON template differs from the pinned constants map")
    if selected_template.get("crop_start") != -1:
        raise ValueError("pinned JSON video template must use the dynamic crop prepass")
    return templates


def _read_prompt_file(prompt_file: Path) -> tuple[str, bytes]:
    path = prompt_file.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("prompt input must be a regular UTF-8 text file")
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_PROMPT_FILE_BYTES:
        raise ValueError("prompt file must contain between 1 byte and 64 KiB")
    if b"\x00" in raw:
        raise ValueError("prompt file must not contain NUL bytes")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("prompt file is not valid UTF-8") from error
    if not text.strip():
        raise ValueError("natural prompt must not be empty or whitespace-only")
    return text, raw


def _receipt_files(receipt_path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    raw = receipt_path.expanduser().resolve(strict=True).read_bytes()
    receipt = json.loads(raw.decode("utf-8"))
    if not isinstance(receipt, dict) or receipt.get("repo_id") != MODEL_ID or receipt.get("revision") != MODEL_REVISION:
        raise ValueError("tokenizer receipt does not pin the requested Qwen model revision")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise ValueError("tokenizer receipt has no file records")
    return files, _sha256_bytes(raw)


def _verify_tokenizer_assets(checkpoint_root: Path, receipt_path: Path) -> tuple[dict[str, Any], str]:
    root = checkpoint_root.expanduser().resolve(strict=True)
    receipt_files, receipt_sha = _receipt_files(receipt_path)
    assets: dict[str, Any] = {}
    for name in TOKENIZER_ASSET_NAMES:
        candidate = root / name
        if not candidate.exists():
            continue
        record = receipt_files.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"checkpoint receipt does not authenticate tokenizer asset {name}")
        size = candidate.stat().st_size
        digest = _sha256_file(candidate)
        if record.get("nbytes") != size or record.get("sha256") != digest:
            raise ValueError(f"tokenizer asset {name} differs from its checkpoint receipt")
        assets[name] = {"nbytes": size, "sha256": digest}
    for required in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        if required not in assets:
            raise ValueError(f"pinned tokenizer asset is missing: {required}")
    return assets, receipt_sha


def _encoding_rows(encoding: Any, key: str) -> list[int]:
    value = encoding[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError(f"upstream tokenizer returned unexpected {key} shape")
    row = value[0]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in row):
        raise ValueError(f"upstream tokenizer returned non-integer {key} values")
    return row


def build_prompt(
    checkpoint_root: Path,
    checkpoint_receipt: Path,
    upstream_root: Path,
    prompt_file: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run pinned upstream prompt rendering/tokenization with the local tokenizer only."""
    encoder_path, constants_path, upstream_record = _verify_upstream_sources(upstream_root)
    templates = _source_templates(constants_path)
    tokenizer_assets, receipt_sha = _verify_tokenizer_assets(checkpoint_root, checkpoint_receipt)
    prompt, prompt_bytes = _read_prompt_file(prompt_file)

    from transformers import AutoTokenizer

    import transformers
    import tokenizers

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if str(transformers.__version__) != TOKENIZER_RUNTIME_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "tokenizer runtime version differs from the pinned reference environment"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_root.expanduser().resolve(strict=True)),
        local_files_only=True,
        trust_remote_code=False,
        padding_side="right",
    )
    if tokenizer.padding_side != "right":
        raise RuntimeError("Qwen tokenizer did not preserve HyNomas right padding")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("pinned Qwen tokenizer has no padding token ID")
    tokenizer_runtime = _tokenizer_runtime_sources(tokenizer, transformers, tokenizers)

    methods = _source_methods(encoder_path)
    encoder = methods()
    encoder.text_encoder_type = "qwen-2.5vl-7b"
    encoder.tokenizer = tokenizer
    encoder.use_template = True
    encoder.use_video_template = True
    encoder.prompt_template = templates["li-dit-encode-image-json"]
    encoder.prompt_template_video = templates[TEMPLATE_NAME]

    # TextEncoder.__init__ computes this dynamic video crop using a fixed dummy
    # prompt before processing user prompts. Keep that constructor behavior.
    encoder.text2tokens("a photo of a cat", data_type="video")
    crop_start = encoder.prompt_template_video.get("crop_start")
    if isinstance(crop_start, bool) or not isinstance(crop_start, int) or not 0 < crop_start <= MAX_CROP_START:
        raise ValueError("upstream dynamic video crop is missing or outside the supported bound")

    marker = "<|im_start|>user\n"
    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    if not isinstance(marker_ids, list) or not marker_ids or any(
        isinstance(item, bool) or not isinstance(item, int) for item in marker_ids
    ):
        raise ValueError("Qwen user-header marker did not tokenize to integer IDs")
    marker_start = crop_start - len(marker_ids)
    if marker_start < 0:
        raise ValueError("dynamic crop is shorter than the Qwen user-header marker")

    messages = encoder.apply_text_to_template(prompt, templates[TEMPLATE_NAME]["template"])
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered_prompt, str):
        raise ValueError("Qwen tokenizer did not return a rendered chat string")

    # This call is the exact extracted upstream text2tokens implementation.
    encoding = encoder.text2tokens([prompt], data_type="video", max_length=BASE_SEQUENCE_LENGTH)
    token_ids = _encoding_rows(encoding, "input_ids")
    attention_mask = _encoding_rows(encoding, "attention_mask")
    expected_raw_length = BASE_SEQUENCE_LENGTH + crop_start
    if len(token_ids) != expected_raw_length or len(attention_mask) != expected_raw_length:
        raise ValueError("upstream tokenizer raw width does not equal 1280 plus crop_start")
    if token_ids[marker_start:crop_start] != marker_ids:
        raise ValueError("raw prompt does not contain the pinned user-header marker at crop_start")
    first_marker_start = next(
        (
            index
            for index in range(len(token_ids) - len(marker_ids) + 1)
            if token_ids[index : index + len(marker_ids)] == marker_ids
        ),
        None,
    )
    if first_marker_start != marker_start:
        raise ValueError("raw prompt user-header marker is not its first exact occurrence")
    if attention_mask[0] != 1 or any(
        value not in (0, 1) for value in attention_mask
    ):
        raise ValueError("upstream tokenizer returned an invalid attention mask")
    first_padding = next(
        (index for index, value in enumerate(attention_mask) if value == 0),
        len(attention_mask),
    )
    if any(attention_mask[first_padding:]):
        raise ValueError("upstream tokenizer did not right-pad the prompt")
    if attention_mask[crop_start] != 1:
        raise ValueError("cropped video prompt has no visible first token")

    forbidden_ids: dict[str, int] = {}
    for marker_text in VIDEO_MARKERS:
        marker_id = tokenizer.convert_tokens_to_ids(marker_text)
        if (
            isinstance(marker_id, int)
            and marker_id >= 0
            and marker_id != tokenizer.unk_token_id
        ):
            if marker_id in token_ids:
                raise ValueError(f"text-only prompt contains a multimodal token: {marker_text}")
            forbidden_ids[marker_text] = marker_id
    if any(forbidden_ids.get(name) != marker_id for name, marker_id in REQUIRED_VIDEO_MARKER_IDS.items()):
        raise ValueError("pinned Qwen tokenizer multimodal marker IDs do not match the native contract")

    cropped_ids = token_ids[crop_start:]
    cropped_mask = attention_mask[crop_start:]
    if len(cropped_ids) != BASE_SEQUENCE_LENGTH or len(cropped_mask) != BASE_SEQUENCE_LENGTH:
        raise ValueError("HyNomas suffix crop does not have the configured visible width")

    request = {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "position_offset": 0,
        "position_ids_f32": [
            [float(index) for index in range(expected_raw_length)] for _ in range(3)
        ],
        "position_ids_for_official_model": [
            [index for index in range(expected_raw_length)] for _ in range(3)
        ],
        "multimodal_position_shape": [3, 1, expected_raw_length],
        "sequence_length": expected_raw_length,
    }
    metadata = {
        "schema": PROMPT_SCHEMA,
        "schema_version": PROMPT_SCHEMA_VERSION,
        "template_name": TEMPLATE_NAME,
        "template_sha256": VIDEO_JSON_TEMPLATE_SHA256,
        "prompt": prompt,
        "prompt_sha256": _sha256_bytes(prompt_bytes),
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": _sha256_bytes(rendered_prompt.encode("utf-8")),
        "messages": messages,
        "marker": marker,
        "marker_token_ids": marker_ids,
        "marker_start": marker_start,
        "crop_start": crop_start,
        "raw_length": expected_raw_length,
        "text_length": BASE_SEQUENCE_LENGTH,
        "sequence_operation": "decoder runs on full raw width; output hidden states and attention mask use suffix [crop_start:]",
        "constructor_prepass": {
            "prompt": "a photo of a cat",
            "max_length": 256,
            "padding": "max_length",
            "truncation": True,
            "marker_search": "first exact marker-token subsequence",
        },
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "cropped_attention_mask": cropped_mask,
        "cropped_token_ids": cropped_ids,
        "raw_token_ids_sha256": _canonical_json_sha256(token_ids),
        "raw_mask_sha256": _canonical_json_sha256(attention_mask),
        "cropped_token_ids_sha256": _canonical_json_sha256(cropped_ids),
        "cropped_attention_mask_sha256": _canonical_json_sha256(cropped_mask),
        "padding": {
            "side": tokenizer.padding_side,
            "pad_token_id": int(tokenizer.pad_token_id),
        },
        "forbidden_multimodal_token_ids": forbidden_ids,
        "tokenizer": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "transformers_version": str(transformers.__version__),
            "local_files_only": True,
            "trust_remote_code": False,
            "receipt_sha256": receipt_sha,
            "assets": tokenizer_assets,
            "runtime": tokenizer_runtime,
            "compatibility_note": (
                "Prompt tokenization used the recorded local Transformers and tokenizers runtime; "
                "the HyNomas model and full pipeline were not executed"
            ),
        },
        "tokenizer_assets": tokenizer_assets,
        "upstream_revision": upstream_record["repository_revision"],
        "upstream_source": upstream_record,
        "producer_source_sha256": _sha256_file(Path(__file__).resolve()),
        "execution_scope": (
            "exact prompt/template/crop/tokenization methods extracted from the pinned HyNomas encoder AST; "
            "does not execute the HyNomas model or text encoder"
        ),
        "text_positions": {
            "source": "pinned Qwen2_5_VLTextModel default cache positions",
            "position_offset": 0,
            "axes": 3,
            "includes_right_padding": True,
            "cropping_resets_positions": False,
        },
        "text_only": True,
    }
    return request, metadata


def validate_prompt_metadata(
    metadata: Any,
    token_ids: Any,
    attention_mask: Any,
    position_ids_f32: Any,
    position_ids_for_official_model: Any,
    multimodal_position_shape: Any,
    position_offset: Any,
) -> int:
    """Validate fixture binding for a stored natural-prompt contract; return raw width."""
    if not isinstance(metadata, dict) or metadata.get("schema") != PROMPT_SCHEMA or metadata.get("schema_version") != PROMPT_SCHEMA_VERSION:
        raise ValueError("HyNomas prompt contract schema is invalid")
    if metadata.get("template_name") != TEMPLATE_NAME or metadata.get("template_sha256") != VIDEO_JSON_TEMPLATE_SHA256:
        raise ValueError("HyNomas video JSON template identity is invalid")
    crop_start = metadata.get("crop_start")
    raw_length = metadata.get("raw_length")
    final_length = metadata.get("text_length")
    if isinstance(crop_start, bool) or not isinstance(crop_start, int) or not 0 < crop_start <= MAX_CROP_START:
        raise ValueError("HyNomas crop_start is outside the pinned range")
    if raw_length != BASE_SEQUENCE_LENGTH + crop_start or raw_length > MAX_RAW_SEQUENCE_LENGTH:
        raise ValueError("HyNomas raw prompt width is invalid")
    if final_length != BASE_SEQUENCE_LENGTH:
        raise ValueError("HyNomas cropped prompt width is invalid")
    if not isinstance(token_ids, list) or len(token_ids) != raw_length:
        raise ValueError("HyNomas raw token IDs do not match the producer inputs")
    if not isinstance(attention_mask, list) or len(attention_mask) != raw_length:
        raise ValueError("HyNomas raw attention mask does not match the producer inputs")
    if metadata.get("token_ids") != token_ids or metadata.get("attention_mask") != attention_mask:
        raise ValueError("HyNomas prompt token arrays differ from the producer inputs")
    if metadata.get("cropped_token_ids") != token_ids[crop_start:]:
        raise ValueError("HyNomas cropped token IDs are not the raw suffix")
    if metadata.get("cropped_attention_mask") != attention_mask[crop_start:]:
        raise ValueError("HyNomas cropped mask is not the raw mask suffix")
    if not attention_mask[crop_start]:
        raise ValueError("HyNomas crop begins in right padding")
    if position_offset != 0:
        raise ValueError("HyNomas prompt positions must retain the original zero offset")
    positions_i64 = [list(range(raw_length)) for _ in range(3)]
    positions_f32 = [[float(value) for value in axis] for axis in positions_i64]
    if position_ids_for_official_model != positions_i64 or position_ids_f32 != positions_f32:
        raise ValueError("HyNomas raw positions must match the un-cropped official text positions")
    if multimodal_position_shape != [3, 1, raw_length]:
        raise ValueError("HyNomas multimodal position shape is invalid")
    crop_marker = metadata.get("marker")
    marker_ids = metadata.get("marker_token_ids")
    marker_start = metadata.get("marker_start")
    if crop_marker != "<|im_start|>user\n" or not isinstance(marker_ids, list) or not marker_ids:
        raise ValueError("HyNomas crop marker metadata is invalid")
    if marker_start != crop_start - len(marker_ids) or marker_start < 0:
        raise ValueError("HyNomas crop marker offset is invalid")
    if token_ids[marker_start:crop_start] != marker_ids:
        raise ValueError("HyNomas raw IDs do not contain the declared marker at crop_start")
    first_marker_start = next(
        (
            index
            for index in range(len(token_ids) - len(marker_ids) + 1)
            if token_ids[index : index + len(marker_ids)] == marker_ids
        ),
        None,
    )
    if first_marker_start != marker_start:
        raise ValueError("HyNomas user-header marker is not its first exact occurrence")
    if not metadata.get("text_only"):
        raise ValueError("HyNomas prompt contract is not marked text-only")
    producer_sha = metadata.get("producer_source_sha256")
    if not isinstance(producer_sha, str) or re.fullmatch(r"[0-9a-f]{64}", producer_sha) is None:
        raise ValueError("HyNomas prompt producer source digest is malformed")
    if metadata.get("execution_scope") != (
        "exact prompt/template/crop/tokenization methods extracted from the pinned HyNomas encoder AST; "
        "does not execute the HyNomas model or text encoder"
    ):
        raise ValueError("HyNomas prompt execution scope is invalid")
    prompt_text = metadata.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("HyNomas original prompt text is missing")
    if metadata.get("prompt_sha256") != _sha256_bytes(prompt_text.encode("utf-8")):
        raise ValueError("HyNomas original prompt text digest is invalid")
    rendered = metadata.get("rendered_prompt")
    if not isinstance(rendered, str) or metadata.get("rendered_prompt_sha256") != _sha256_bytes(rendered.encode("utf-8")):
        raise ValueError("HyNomas rendered prompt digest is invalid")
    messages = metadata.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not all(isinstance(item, dict) for item in messages)
        or messages[0].get("role") != "system"
        or not isinstance(messages[0].get("content"), str)
        or messages[1].get("role") != "user"
        or messages[1].get("content") != prompt_text
    ):
        raise ValueError("HyNomas rendered messages do not bind the original prompt")
    if metadata.get("raw_token_ids_sha256") != _canonical_json_sha256(token_ids):
        raise ValueError("HyNomas raw token ID digest is invalid")
    if metadata.get("raw_mask_sha256") != _canonical_json_sha256(attention_mask):
        raise ValueError("HyNomas raw attention-mask digest is invalid")
    if metadata.get("cropped_token_ids_sha256") != _canonical_json_sha256(token_ids[crop_start:]):
        raise ValueError("HyNomas cropped token ID digest is invalid")
    if metadata.get("cropped_attention_mask_sha256") != _canonical_json_sha256(attention_mask[crop_start:]):
        raise ValueError("HyNomas cropped attention-mask digest is invalid")
    tokenizer = metadata.get("tokenizer")
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("repo_id") != MODEL_ID
        or tokenizer.get("revision") != MODEL_REVISION
    ):
        raise ValueError("HyNomas tokenizer provenance is invalid")
    runtime = tokenizer.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("transformers_version") != TOKENIZER_RUNTIME_TRANSFORMERS_VERSION
        or not isinstance(runtime.get("python_version"), str)
        or not isinstance(runtime.get("transformers_sources"), dict)
        or not runtime["transformers_sources"]
        or tokenizer.get("compatibility_note") != (
            "Prompt tokenization used the recorded local Transformers and tokenizers runtime; "
            "the HyNomas model and full pipeline were not executed"
        )
    ):
        raise ValueError("HyNomas tokenizer runtime provenance is incomplete")
    for source in runtime["transformers_sources"].values():
        if not isinstance(source, dict) or re.fullmatch(
            r"[0-9a-f]{64}", str(source.get("sha256", ""))
        ) is None:
            raise ValueError("HyNomas tokenizer runtime source digest is malformed")
    assets = metadata.get("tokenizer_assets")
    if not isinstance(assets, dict) or not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(assets):
        raise ValueError("HyNomas tokenizer asset provenance is incomplete")
    if tokenizer.get("assets") != assets:
        raise ValueError("HyNomas tokenizer asset records disagree")
    for record in assets.values():
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            or isinstance(record.get("nbytes"), bool)
            or not isinstance(record.get("nbytes"), int)
            or record["nbytes"] <= 0
        ):
            raise ValueError("HyNomas tokenizer asset digest is malformed")
    upstream = metadata.get("upstream_source")
    if not isinstance(upstream, dict) or upstream.get("repository_revision") != UPSTREAM_REVISION:
        raise ValueError("HyNomas upstream source revision is invalid")
    if metadata.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("HyNomas upstream revision field is invalid")
    expected_sources = {
        "encoder_source": UPSTREAM_ENCODER_SHA256,
        "constants_source": UPSTREAM_CONSTANTS_SHA256,
    }
    for key, digest in expected_sources.items():
        source = upstream.get(key)
        expected_path = (
            "hymm/models/text_encoder/encoder.py"
            if key == "encoder_source"
            else "configs/constants.py"
        )
        if (
            not isinstance(source, dict)
            or source.get("sha256") != digest
            or source.get("path") != expected_path
        ):
            raise ValueError(f"HyNomas upstream source digest is invalid: {key}")
    if upstream.get("methods_executed_from_ast") != [
        "apply_text_to_template", "calculate_crop_start", "text2tokens"
    ]:
        raise ValueError("HyNomas AST method inventory is invalid")
    forbidden_ids = metadata.get("forbidden_multimodal_token_ids")
    if not isinstance(forbidden_ids, dict) or any(
        forbidden_ids.get(name) != marker_id
        for name, marker_id in REQUIRED_VIDEO_MARKER_IDS.items()
    ) or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value in token_ids
        for value in forbidden_ids.values()
    ):
        raise ValueError("HyNomas text-only token inventory is invalid")
    return raw_length
