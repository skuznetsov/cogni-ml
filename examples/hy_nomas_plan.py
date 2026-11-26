#!/usr/bin/env python3
"""Refine Hy-Nomas scene briefs with one explicit GPT-6 Luna API call.

Without ``--execute`` this validates and summarizes the exact bounded request;
it does not call the API or write an output suite. Successful execution writes
a fresh suite and provenance receipt without replacing existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-6-luna"
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 12_000
REQUEST_TIMEOUT_SECONDS = 120
MAX_SUITE_BYTES = 512 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_SUITE_BYTES = 512 * 1024
MAX_SCENES = 12
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SUITE_SCHEMA = "cogni_ml.hy_nomas_draft_suite/v1"
RECEIPT_SCHEMA = "cogni_ml.hy_nomas_prompt_planner_receipt/v1"
STYLE = "Photorealistic real-world footage"
PERSPECTIVE = "third-person observer"

EXPECTED_SETTINGS = {
    "segments": 2,
    "seconds": 10,
    "width": 512,
    "height": 288,
    "fps": 24,
    "steps_per_segment": 4,
    "weights": "original BF16",
    "history": "F32 once-per-scene text cache",
    "noise": "The same authenticated seed-42 noise banks for every scene; prompt-only variation.",
    "selection": "One first take per scene, including visible failures.",
}

SYSTEM_PROMPT = f"""You refine short first-take briefs for a native video generator.
Return only the JSON object required by the supplied strict schema. Preserve
every scene id exactly and keep the scene order. Use style {STYLE!r} and
perspective {PERSPECTIVE!r} exactly. Each visual_description must describe one
photorealistic, continuous ten-second real-world shot that can use the same
prompt throughout two linked five-second segments. Keep motion simple and
physically plausible, make the main subject and camera behavior explicit, and
avoid cuts, transitions, scene changes, action-tail annotations, or a sequence
of many separate actions. Refine the supplied brief without inventing a claim
that the scene will succeed. Subjects must have concise visible descriptions;
use one subject entry for a landscape or product when no person or animal is
present. Checks must be short, observable review criteria, not quality claims."""

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scenes"],
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id", "title", "style", "perspective", "subjects",
                    "visual_description", "checks",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "style": {"type": "string", "enum": [STYLE]},
                    "perspective": {"type": "string", "enum": [PERSPECTIVE]},
                    "subjects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "description"],
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                    "visual_description": {"type": "string"},
                    "checks": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


class PlannerError(ValueError):
    """A bounded input, API, or generated-plan rejection."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        # Never forward an authorization header away from the fixed endpoint.
        raise PlannerError("Responses API redirect rejected")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlannerError("duplicate JSON key")
        result[key] = value
    return result


def _parse_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except PlannerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise PlannerError(f"{label} is not valid UTF-8 JSON") from None


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent) + "\n").encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise PlannerError("JSON document cannot be serialized safely") from None


def _text(value: Any, where: str, *, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise PlannerError(f"{where} must be a string no longer than {maximum} characters")
    if not allow_empty and not value.strip():
        raise PlannerError(f"{where} must not be empty")
    if "|||" in value or any(ord(ch) < 32 for ch in value):
        raise PlannerError(f"{where} contains a reserved delimiter or control character")
    return value


def validate_input_suite(suite: Any) -> list[dict[str, Any]]:
    if not isinstance(suite, dict) or suite.get("schema") != SUITE_SCHEMA:
        raise PlannerError("unsupported Hy-Nomas draft suite schema")
    if not isinstance(suite.get("id"), str) or not suite["id"].strip():
        raise PlannerError("suite id is required")
    if not isinstance(suite.get("title"), str) or not suite["title"].strip():
        raise PlannerError("suite title is required")
    settings = suite.get("settings")
    if not isinstance(settings, dict):
        raise PlannerError("suite settings must be an object")
    for key, expected in EXPECTED_SETTINGS.items():
        if settings.get(key) != expected:
            raise PlannerError(f"suite setting {key!r} differs from the admitted native profile")

    scenes = suite.get("scenes")
    if not isinstance(scenes, list) or not 1 <= len(scenes) <= MAX_SCENES:
        raise PlannerError(f"suite must contain between 1 and {MAX_SCENES} scenes")
    ids: set[str] = set()
    for index, scene in enumerate(scenes):
        where = f"suite.scenes[{index}]"
        if not isinstance(scene, dict):
            raise PlannerError(f"{where} must be an object")
        scene_id = scene.get("id")
        if not isinstance(scene_id, str) or not SCENE_ID_RE.fullmatch(scene_id):
            raise PlannerError(f"{where}.id is invalid")
        if scene_id in ids:
            raise PlannerError(f"duplicate scene id: {scene_id}")
        ids.add(scene_id)
        _text(scene.get("title"), f"{where}.title", maximum=120)
        _text(scene.get("prompt"), f"{where}.prompt", maximum=6000)
        checks = scene.get("checks")
        if not isinstance(checks, list) or not 1 <= len(checks) <= 12:
            raise PlannerError(f"{where}.checks must contain 1..12 strings")
        for check_index, check in enumerate(checks):
            _text(check, f"{where}.checks[{check_index}]", maximum=240)
    return scenes


def load_suite(path: Path) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file():
        raise PlannerError("suite path must name an existing regular file, not a symlink")
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_SUITE_BYTES + 1)
    except OSError:
        raise PlannerError("could not read suite file") from None
    if len(raw) > MAX_SUITE_BYTES:
        raise PlannerError("suite file exceeds the size limit")
    suite = _parse_json(raw, "suite")
    scenes = validate_input_suite(suite)
    return suite, raw, scenes


def build_request(suite: Mapping[str, Any], scenes: list[dict[str, Any]], effort: str) -> dict[str, Any]:
    if effort not in EFFORTS:
        raise PlannerError("unsupported reasoning effort")
    briefs = {
        "suite_id": suite["id"],
        "title": suite["title"],
        "native_settings": suite["settings"],
        "scenes": [
            {
                "id": scene["id"],
                "title": scene["title"],
                "first_take_brief": scene["prompt"],
                "current_review_checks": scene["checks"],
            }
            for scene in scenes
        ],
    }
    user_text = (
        "Refine each first-take brief for the native profile. Keep scene IDs and "
        "order exact; do not add, remove, merge, split, or reorder scenes. Keep "
        "the current native settings unchanged. Return one plan for every scene.\n\n"
        + json.dumps(briefs, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "model": MODEL,
        "store": False,
        "reasoning": {"effort": effort},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "hy_nomas_photorealistic_scene_plan",
                "strict": True,
                "schema": PLAN_SCHEMA,
            },
        },
    }


def _validate_plan_text(value: Any, where: str, *, maximum: int) -> str:
    result = _text(value, where, maximum=maximum)
    if "<|" in result or "|>" in result:
        raise PlannerError(f"{where} contains a reserved model token")
    return result


def validate_model_plan(plan: Any, input_scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(plan, dict) or set(plan) != {"scenes"}:
        raise PlannerError("model plan must contain only the scenes field")
    scenes = plan["scenes"]
    if not isinstance(scenes, list) or len(scenes) != len(input_scenes):
        raise PlannerError("model plan scene count differs from the input suite")
    validated: list[dict[str, Any]] = []
    scene_keys = {
        "id", "title", "style", "perspective", "subjects", "visual_description", "checks",
    }
    for index, (scene, original) in enumerate(zip(scenes, input_scenes, strict=True)):
        where = f"model.scenes[{index}]"
        if not isinstance(scene, dict) or set(scene) != scene_keys:
            raise PlannerError(f"{where} has an unexpected field inventory")
        if scene.get("id") != original["id"]:
            raise PlannerError(f"{where}.id/order differs from the source suite")
        title = _validate_plan_text(scene.get("title"), f"{where}.title", maximum=120)
        if scene.get("style") != STYLE or scene.get("perspective") != PERSPECTIVE:
            raise PlannerError(f"{where} changed the canonical style or perspective")
        visual = _validate_plan_text(scene.get("visual_description"),
                                     f"{where}.visual_description", maximum=4000)
        # Style is typed above. Shot continuity and intent require a human
        # review; matching English keywords would not establish either one.
        subjects = scene.get("subjects")
        if not isinstance(subjects, list) or not 1 <= len(subjects) <= 8:
            raise PlannerError(f"{where}.subjects must contain 1..8 entries")
        clean_subjects: list[dict[str, str]] = []
        for subject_index, subject in enumerate(subjects):
            subject_where = f"{where}.subjects[{subject_index}]"
            if not isinstance(subject, dict) or set(subject) != {"name", "description"}:
                raise PlannerError(f"{subject_where} has an unexpected field inventory")
            clean_subjects.append({
                "name": _validate_plan_text(subject.get("name"), f"{subject_where}.name", maximum=100),
                "description": _validate_plan_text(subject.get("description"),
                                                    f"{subject_where}.description", maximum=600),
            })
        checks = scene.get("checks")
        if not isinstance(checks, list) or not 1 <= len(checks) <= 8:
            raise PlannerError(f"{where}.checks must contain 1..8 entries")
        clean_checks = [
            _validate_plan_text(check, f"{where}.checks[{i}]", maximum=200)
            for i, check in enumerate(checks)
        ]
        validated.append({
            "id": original["id"],
            "title": title,
            "style": STYLE,
            "perspective": PERSPECTIVE,
            "subjects": clean_subjects,
            "visual_description": visual,
            "checks": clean_checks,
        })
    return validated


def _native_prompt(scene: Mapping[str, Any]) -> str:
    subjects = [
        {
            "subject_id": f"<|Subject{index}|>",
            "display_name": subject["name"],
            "subject_attribute": subject["description"],
        }
        for index, subject in enumerate(scene["subjects"], start=1)
    ]
    document = {
        "Style": STYLE,
        "perspective_view": PERSPECTIVE,
        "subject_list": subjects,
        "segments": [{
            "start_time": "00:00.000",
            "end_time": "00:10.000",
            "visual_description_long": scene["visual_description"],
            "description_change": None,
            "contained_actions": [],
        }],
    }
    return json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)


def make_output_suite(suite: Mapping[str, Any], plan: list[dict[str, Any]]) -> dict[str, Any]:
    output = json.loads(json.dumps(suite, ensure_ascii=False))
    updated_scenes: list[dict[str, Any]] = []
    for source, refined in zip(suite["scenes"], plan, strict=True):
        scene = dict(source)
        scene["title"] = refined["title"]
        scene["prompt"] = refined["visual_description"]
        scene["checks"] = refined["checks"]
        scene["native_prompt"] = _native_prompt(refined)
        updated_scenes.append(scene)
    output["scenes"] = updated_scenes
    if output.get("settings") != suite.get("settings"):
        raise PlannerError("planner changed suite settings")
    return output


def _extract_output_text(response: Mapping[str, Any]) -> str:
    if response.get("status") != "completed":
        raise PlannerError("Responses API did not complete the request")
    output = response.get("output")
    if not isinstance(output, list):
        raise PlannerError("Responses API response has no output list")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "refusal":
            raise PlannerError("Responses API refused the planning request")
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal" or "refusal" in part:
                raise PlannerError("Responses API refused the planning request")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if len(texts) != 1 or not texts[0].strip():
        raise PlannerError("Responses API response did not contain one structured output")
    return texts[0]


def _read_limited(response: Any, maximum: int) -> bytes:
    result = bytearray()
    while len(result) <= maximum:
        chunk = response.read(min(64 * 1024, maximum + 1 - len(result)))
        if not chunk:
            break
        result.extend(chunk)
    if len(result) > maximum:
        raise PlannerError("Responses API response exceeds the size limit")
    return bytes(result)


def _api_request(payload: Mapping[str, Any], api_key: str,
                 opener: Callable[..., Any] | None = None) -> tuple[Any, bytes]:
    if not api_key:
        raise PlannerError("OPENAI_API_KEY is required with --execute")
    if api_key != api_key.strip() or any(ch in api_key for ch in "\r\n"):
        raise PlannerError("OPENAI_API_KEY has invalid whitespace")
    body = _json_bytes(payload)
    if len(body) > MAX_REQUEST_BYTES:
        raise PlannerError("Responses API request exceeds the size limit")
    request = Request(
        API_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    open_url = opener or build_opener(NoRedirect()).open
    try:
        with open_url(request, timeout=REQUEST_TIMEOUT_SECONDS) as result:
            status = getattr(result, "status", 200)
            if status != 200:
                raise PlannerError(f"Responses API returned HTTP status {status}")
            raw = _read_limited(result, MAX_RESPONSE_BYTES)
    except HTTPError as error:
        raise PlannerError(f"Responses API returned HTTP status {error.code}") from None
    except (URLError, TimeoutError, OSError):
        raise PlannerError("Responses API request failed") from None
    response = _parse_json(raw, "Responses API response")
    if not isinstance(response, dict):
        raise PlannerError("Responses API response must be an object")
    return response, raw


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise PlannerError("Responses API response is missing token usage")
    result: dict[str, int] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(name)
        if type(value) is not int or value < 0:
            raise PlannerError("Responses API token usage is malformed")
        result[name] = value
    return result


def _response_metadata(response: Mapping[str, Any]) -> tuple[str, str, dict[str, int]]:
    response_id = response.get("id")
    model = response.get("model")
    if not isinstance(response_id, str) or not response_id or len(response_id) > 256:
        raise PlannerError("Responses API response id is missing")
    if not isinstance(model, str) or not (model == MODEL or model.startswith(MODEL + "-")):
        raise PlannerError("Responses API returned a model outside the requested Luna family")
    return response_id, model, _usage(response)


def _safe_output_path(value: str, label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.name in ("", ".", "..") or candidate.is_symlink():
        raise PlannerError(f"{label} must be a fresh file path, not a symlink")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError:
        raise PlannerError(f"{label} parent must already exist") from None
    if not parent.is_dir():
        raise PlannerError(f"{label} parent must be a directory")
    return parent / candidate.name


def _fresh_paths(output_value: str, receipt_value: str | None) -> tuple[Path, Path]:
    output = _safe_output_path(output_value, "output")
    receipt_default = output.with_name(f"{output.stem}.receipt.json")
    receipt = _safe_output_path(receipt_value, "receipt") if receipt_value else receipt_default
    if output == receipt:
        raise PlannerError("output and receipt paths must differ")
    for path, label in ((output, "output"), (receipt, "receipt")):
        if path.exists() or path.is_symlink():
            raise PlannerError(f"{label} already exists; refusing to overwrite")
    return output, receipt


def _write_fresh(path: Path, payload: bytes) -> tuple[int, int]:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise PlannerError(f"output already exists; refusing to overwrite: {path.name}") from None
    except OSError:
        raise PlannerError(f"could not create output file: {path.name}") from None
    identity: tuple[int, int] | None = None
    try:
        stat = os.fstat(fd)
        identity = (stat.st_dev, stat.st_ino)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return identity
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        if identity is not None:
            try:
                stat = path.stat()
                if (stat.st_dev, stat.st_ino) == identity:
                    path.unlink()
            except OSError:
                pass
        raise PlannerError(f"could not write output file: {path.name}") from None


def _remove_if_same(path: Path, identity: tuple[int, int]) -> None:
    try:
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) == identity:
            path.unlink()
    except OSError:
        pass


def _source_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        raise PlannerError("could not hash planner source") from None


def run(suite_path: Path, output_value: str, receipt_value: str | None, effort: str,
        execute: bool, *, api_key: str | None = None,
        opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    suite, suite_bytes, input_scenes = load_suite(suite_path)
    output_path, receipt_path = _fresh_paths(output_value, receipt_value)
    response_path = receipt_path.with_name(f"{receipt_path.stem}.response.json")
    if response_path in (output_path, receipt_path) or response_path.exists() or response_path.is_symlink():
        raise PlannerError("response artifact must be fresh and distinct from outputs")
    request_payload = build_request(suite, input_scenes, effort)
    request_bytes = _json_bytes(request_payload)
    if len(request_bytes) > MAX_REQUEST_BYTES:
        raise PlannerError("Responses API request exceeds the size limit")
    summary = {
        "status": "request_validated_not_sent",
        "api_call": False,
        "model": MODEL,
        "effort": effort,
        "scene_count": len(input_scenes),
        "scene_ids": [scene["id"] for scene in input_scenes],
        "request_bytes": len(request_bytes),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "output": str(output_path),
        "receipt": str(receipt_path),
        "response": str(response_path),
    }
    if not execute:
        return summary

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    response, response_bytes = _api_request(request_payload, api_key or "", opener)
    # Keep the bounded API body even if subsequent validation rejects it.
    # Request headers and credentials are never part of this artifact.
    _write_fresh(response_path, response_bytes)
    output_text = _extract_output_text(response)
    plan = _parse_json(output_text.encode("utf-8"), "model plan")
    refined = validate_model_plan(plan, input_scenes)
    output_suite = make_output_suite(suite, refined)
    output_bytes = _json_bytes(output_suite, indent=2)
    if len(output_bytes) > MAX_OUTPUT_SUITE_BYTES:
        raise PlannerError("refined suite exceeds the output size limit")

    response_id, actual_model, usage = _response_metadata(response)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "model_requested": MODEL,
        "model_returned": actual_model,
        "response_id": response_id,
        "reasoning_effort": effort,
        "usage": usage,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "response_artifact": str(response_path),
        "source_suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "output_suite_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "planner_source_sha256": _source_sha256(),
        "scene_ids": [scene["id"] for scene in input_scenes],
    }
    receipt_bytes = _json_bytes(receipt, indent=2)
    output_identity = _write_fresh(output_path, output_bytes)
    try:
        _write_fresh(receipt_path, receipt_bytes)
    except Exception:
        _remove_if_same(output_path, output_identity)
        raise
    summary.update({
        "status": "completed",
        "api_call": True,
        "model_returned": actual_model,
        "response_id": response_id,
        "usage": usage,
        "output_suite_sha256": receipt["output_suite_sha256"],
    })
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", help="fresh adjacent provenance receipt; default is OUTPUT stem.receipt.json")
    parser.add_argument("--effort", choices=EFFORTS, default=DEFAULT_EFFORT)
    parser.add_argument("--execute", action="store_true", help="send one paid Responses API request")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args.suite, args.output, args.receipt, args.effort, args.execute)
    except PlannerError as error:
        print(f"hy_nomas_plan: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
