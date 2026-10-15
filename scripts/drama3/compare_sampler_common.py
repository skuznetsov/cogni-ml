#!/usr/bin/env python3
"""Prepare and compare controlled common-randomness DRAMA-3 sampler probes.

The prepare command reads retained native and pinned-MLX common-prefix logits;
it does not import MLX or execute model math. The compare command validates a
GPU probe bundle and reports fixed-tolerance probability/logit comparisons,
candidate-set differences, and tie ambiguity. Neither command claims to replay
MLX's seeded random stream.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACE_FORMAT = "drama3-generation-common-prefix-v1"
FIXTURE_FORMAT = "drama3-sampler-common-input-v1"
PROBE_FORMAT = "drama3-sampler-common-output-v1"
REPORT_FORMAT = "drama3-sampler-common-comparison-v1"
ATOL = 6e-5
RTOL = 3e-4
DEFAULT_STEPS = (0, 128, 192, 256)
DRAW_VALUES = (0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DRAW_STRIDE = 11
DRAW_SLOTS = {
    "main": 0,
    "main_retry": 1,
    **{f"tower_head_{head}": head + 2 for head in range(9)},
}
NORMAL = (0.8, 30, 0.9)
RETRY = (1.0, 30, 0.95)
NOISE_SEED = 0xD3A3
NOISE_MODULUS = 997
MODEL_CONFIG_KEYS = (
    "hidden_size",
    "main_vocab_size",
    "tower_dim",
    "tower_vocab_size",
    "num_codebooks",
)
MODEL_FILE_KEYS = ("config.json", "model.safetensors.index.json")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
STAGE_RE = re.compile(
    r"generation\.step([0-9]+)\.(backbone_hidden|main_logits|tower_hidden|tower_logits\.([0-8]))\Z"
)
PINNED_MLX_ROOT = Path(
    "/Users/sergey/.cache/cogni-ml/drama3-reference-20261014/mlx-audio"
)
PINNED_MLX_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
PINNED_MODEL = {
    "repository": "cguldogan/drama-3-mlx-q4",
    "revision": "5331ab509359c0e4a7e353a19e255ddf3e8152a8",
}
PINNED_UPSTREAM = {
    "repository": "https://github.com/cguldogan/mlx-audio",
    "revision": PINNED_MLX_REVISION,
}
PINNED_TOWER_SOURCE = Path("mlx_audio/tts/models/fish_omni/tower.py")
PINNED_TOWER_SHA256 = "a184fcc388a70d389ec912c531eb3bbdcc0372be638560d1dd9a9ee1a1ecdba0"


class InvalidBundle(ValueError):
    """An input trace, fixture, or GPU result violates its declared schema."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidBundle(message)


def _canonical_sampling_contract() -> dict[str, Any]:
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "normal": {"temperature": NORMAL[0], "top_k": NORMAL[1], "top_p": NORMAL[2]},
        "main_retry": {"temperature": RETRY[0], "top_k": RETRY[1], "top_p": RETRY[2]},
        "draw_semantics": "shared fixed boundary uniforms for native inverse-CDF diagnostics; when a native random tape is supplied, its actual per-step/per-call uniform is appended and shared across native/upstream logits arms",
        "noise_semantics": "common deterministic open-interval uniforms for the Gumbel-max diagnostic; they do not replay an MLX seed or random stream",
        "noise_seed": NOISE_SEED,
        "noise_formula": "u[d,r] = F32(((((d+1)*73 + (r+1)*151 + seed) mod 997) + 0.5) / 997)",
        "upstream_filter": "F32 logits / max(temperature,1e-5); argpartition top-k; sort selected logits descending; full-vocabulary logsumexp; drop the token crossing top_p; retain rank 0; categorical is diagnosed by common Gumbel-max",
        "native_filter": "resident Metal dr3_sample with stable lower-token-ID top-k tie-break and inverse-CDF draw",
        "top_k_tie_policy": "native uses lower token ID; MLX argpartition tie membership is not API-guaranteed and must be reported separately",
    }


def _same_typed_json_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (type(actual) is dict and actual.keys() == expected.keys() and
                all(_same_typed_json_value(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, float):
        return type(actual) is float and math.isfinite(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _read_json_once(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidBundle(f"{path}: invalid UTF-8 JSON") from error
    return raw, {"sha256": digest, "value": value}


def _f32(value: Any, where: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{where}: expected finite numeric F32",
    )
    try:
        packed = struct.pack("<f", float(value))
    except (OverflowError, struct.error) as error:
        raise InvalidBundle(f"{where}: outside F32 range") from error
    result = struct.unpack("<f", packed)[0]
    _require(math.isfinite(result), f"{where}: expected finite F32")
    return result


def _strict_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{where}: expected integer")
    if minimum is not None:
        _require(value >= minimum, f"{where}: must be >= {minimum}")
    return value


def _sha(value: Any, where: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
             f"{where}: expected lowercase SHA-256")
    return value


def _number_list(value: Any, where: str, expected: int) -> list[float]:
    _require(isinstance(value, list) and len(value) == expected,
             f"{where}: expected {expected} values")
    return [_f32(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _int_list(value: Any, where: str, expected: int | None = None) -> list[int]:
    _require(isinstance(value, list), f"{where}: expected array")
    if expected is not None:
        _require(len(value) == expected, f"{where}: expected {expected} values")
    return [_strict_int(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _read_trace(path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, envelope = _read_json_once(path)
    bundle = envelope["value"]
    _require(isinstance(bundle, dict), f"{path}: expected JSON object")
    _require(bundle.get("format") == TRACE_FORMAT, f"{path}: unexpected trace format")
    source_report = bundle.get("source_report")
    _require(isinstance(source_report, dict), f"{path}: missing source_report")
    _require(isinstance(source_report.get("path"), str) and source_report["path"],
             f"{path}: invalid source_report.path")
    _sha(source_report.get("sha256"), f"{path}.source_report.sha256")

    prompt_ids = _int_list(bundle.get("prompt_ids"), f"{path}.prompt_ids")
    _require(bool(prompt_ids), f"{path}.prompt_ids must not be empty")
    prompt_rows = _strict_int(bundle.get("prompt_rows"), f"{path}.prompt_rows", minimum=1)
    _require(prompt_rows == len(prompt_ids), f"{path}: prompt_rows/prompt_ids mismatch")
    frames_raw = bundle.get("frames")
    _require(isinstance(frames_raw, list) and bool(frames_raw), f"{path}.frames: expected nonempty array")
    frames = [_int_list(row, f"{path}.frames[{index}]", 11) for index, row in enumerate(frames_raw)]
    frame_count = _strict_int(bundle.get("frame_count"), f"{path}.frame_count", minimum=1)
    _require(frame_count == len(frames), f"{path}: frame_count/frames mismatch")

    config_raw = bundle.get("model_config")
    _require(isinstance(config_raw, dict), f"{path}.model_config: expected object")
    config: dict[str, int] = {}
    for key in MODEL_CONFIG_KEYS:
        config[key] = _strict_int(config_raw.get(key), f"{path}.model_config.{key}", minimum=1)
    _require(config["num_codebooks"] == 10, f"{path}: expected ten codebooks")
    _require(config["main_vocab_size"] == 4097 and config["tower_vocab_size"] == 4096,
             f"{path}: unexpected sampler vocabulary sizes")

    files_raw = bundle.get("model_files")
    _require(isinstance(files_raw, dict), f"{path}.model_files: expected object")
    model_files = {key: _sha(files_raw.get(key), f"{path}.model_files.{key}") for key in MODEL_FILE_KEYS}
    execution = bundle.get("execution")
    _require(isinstance(execution, dict), f"{path}.execution: expected object")

    captures = bundle.get("capture_steps")
    _require(isinstance(captures, list), f"{path}.capture_steps: expected array")
    capture_steps = [_strict_int(item, f"{path}.capture_steps[{index}]", minimum=0)
                     for index, item in enumerate(captures)]
    _require(capture_steps == sorted(set(capture_steps)),
             f"{path}.capture_steps: must be sorted and unique")

    traces_raw = bundle.get("traces")
    _require(isinstance(traces_raw, list), f"{path}.traces: expected array")
    traces: dict[str, list[float]] = {}
    seen_stage_names: set[str] = set()
    for index, trace in enumerate(traces_raw):
        where = f"{path}.traces[{index}]"
        _require(isinstance(trace, dict), f"{where}: expected object")
        name = trace.get("name")
        _require(isinstance(name, str), f"{where}.name: expected string")
        match = STAGE_RE.fullmatch(name)
        _require(match is not None, f"{where}.name: unsupported stage {name!r}")
        _require(trace.get("dtype") == "float32", f"{where}.dtype: expected float32")
        shape = _int_list(trace.get("shape"), f"{where}.shape", 1)
        stage_kind = match.group(2)
        expected_width = {
            "backbone_hidden": config["hidden_size"],
            "main_logits": config["main_vocab_size"],
            "tower_hidden": config["tower_dim"],
        }.get(stage_kind, config["tower_vocab_size"])
        _require(shape[0] == expected_width, f"{where}.shape: expected [{expected_width}]")
        values = _number_list(trace.get("values"), f"{where}.values", expected_width)
        _require(name not in seen_stage_names, f"{where}.name: duplicate stage")
        seen_stage_names.add(name)
        step = int(match.group(1))
        _require(step in capture_steps, f"{where}.name: stage absent from capture_steps")
        if stage_kind in ("main_logits",) or stage_kind.startswith("tower_logits."):
            traces[name] = values

    normalized = {
        "arm": arm,
        "path": str(path.resolve()),
        "sha256": envelope["sha256"],
        "source_report": source_report,
        "prompt_ids": prompt_ids,
        "prompt_rows": prompt_rows,
        "frames": frames,
        "frame_count": frame_count,
        "model_config": config,
        "model_files": model_files,
        "capture_steps": capture_steps,
        "execution": execution,
        "model": bundle.get("model"),
        "upstream": bundle.get("upstream"),
        "source_hashes": bundle.get("source_hashes"),
        "generation_parameters": bundle.get("generation_parameters"),
        "traces": traces,
    }
    return normalized, bundle


def _read_and_validate_source_report(native: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    left = native["source_report"]
    right = upstream["source_report"]
    _require(left == right, "native/upstream traces refer to different source reports")
    report_path = Path(left["path"])
    raw, envelope = _read_json_once(report_path)
    _require(envelope["sha256"] == left["sha256"], "source report SHA-256 mismatch")
    report = envelope["value"]
    _require(isinstance(report, dict), "source report: expected JSON object")
    _require(report.get("prompt_ids") == native["prompt_ids"], "source report prompt IDs mismatch")
    _require(report.get("frames") == native["frames"], "source report frame rows mismatch")
    _require(report.get("prompt_rows") == native["prompt_rows"], "source report prompt row count mismatch")
    _require(report.get("seed") == 1, "source report seed mismatch")
    _require(_f32(report.get("temperature"), "source_report.temperature") == _f32(NORMAL[0], "temperature"),
             "source report temperature mismatch")
    _require(report.get("top_k") == NORMAL[1], "source report top_k mismatch")
    _require(_f32(report.get("top_p"), "source_report.top_p") == _f32(NORMAL[2], "top_p"),
             "source report top_p mismatch")
    return {"path": str(report_path.resolve()), "sha256": envelope["sha256"], "value": report}


def _pinned_sampler_metadata(mlx_root: Path) -> dict[str, str]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(mlx_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise InvalidBundle(f"cannot resolve pinned MLX-audio revision at {mlx_root}") from error
    _require(revision == PINNED_MLX_REVISION, f"pinned MLX-audio revision mismatch: {revision}")
    source_path = mlx_root / PINNED_TOWER_SOURCE
    source_bytes = source_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    _require(source_sha == PINNED_TOWER_SHA256, f"pinned tower.py hash mismatch: {source_sha}")
    try:
        source = source_bytes.decode("utf-8")
        tree = ast.parse(source, filename=str(source_path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise InvalidBundle("pinned tower.py cannot be parsed") from error
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "sample_top_k_top_p"),
        None,
    )
    _require(function is not None, "pinned tower.py is missing sample_top_k_top_p")
    segment = ast.get_source_segment(source, function)
    _require(segment is not None, "cannot extract pinned sampler source")
    return {
        "repository": PINNED_UPSTREAM["repository"],
        "revision": revision,
        "source": PINNED_TOWER_SOURCE.as_posix(),
        "source_sha256": source_sha,
        "function_sha256": hashlib.sha256(segment.encode("utf-8")).hexdigest(),
    }


def _noise_rows(k: int, count: int = len(DRAW_VALUES)) -> list[list[float]]:
    rows: list[list[float]] = []
    for draw_index in range(count):
        row: list[float] = []
        for rank in range(k):
            numerator = ((draw_index + 1) * 73 + (rank + 1) * 151 + NOISE_SEED) % NOISE_MODULUS
            value = _f32((numerator + 0.5) / NOISE_MODULUS, "noise_uniform")
            _require(0.0 < value < 1.0, "generated Gumbel noise uniform is outside open interval")
            row.append(value)
        rows.append(row)
    return rows


def _case(
    *,
    name: str,
    role: str,
    source_arm: str,
    step: int | None,
    case_kind: str,
    logits: list[float],
    settings: tuple[float, int, float],
    native_draw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    temperature, top_k, top_p = settings
    draws = [_f32(value, "draw") for value in DRAW_VALUES]
    normalized_draw = None
    if native_draw is not None:
        normalized_draw = dict(native_draw)
        normalized_draw["uniform"] = _f32(native_draw["uniform"], f"{name}.native_draw.uniform")
        draws.append(normalized_draw["uniform"])
    case = {
        "name": name,
        "role": role,
        "source_arm": source_arm,
        "step": step,
        "case_kind": case_kind,
        "logits": logits,
        "temperature": _f32(temperature, f"{name}.temperature"),
        "top_k": top_k,
        "top_p": _f32(top_p, f"{name}.top_p"),
        "draws": draws,
        "noise_uniforms": _noise_rows(top_k, len(draws)),
    }
    if normalized_draw is not None:
        case["native_draw"] = normalized_draw
    return case


def _read_native_random_tape(path: Path) -> tuple[dict[str, Any], list[float]]:
    raw, envelope = _read_json_once(path)
    tape = envelope["value"]
    _require(isinstance(tape, dict), f"{path}: random tape must be an object")
    _require(tape.get("format") == "drama3-random-tape-v1", f"{path}: random tape format mismatch")
    seed = _strict_int(tape.get("seed"), f"{path}.seed", minimum=0)
    _require(seed == 1, f"{path}: expected seed-1 Crystal random tape")
    frames = _strict_int(tape.get("frames"), f"{path}.frames", minimum=1)
    stride = _strict_int(tape.get("stride"), f"{path}.stride", minimum=1)
    _require(stride == DRAW_STRIDE, f"{path}: expected {DRAW_STRIDE} sampler draws per frame")
    expected_generator = "Crystal Random.new(seed), rand.to_f32; main, unconditional retry, nine residual heads"
    _require(tape.get("generator") == expected_generator, f"{path}: random tape generator/order mismatch")
    raw_uniforms = tape.get("uniforms")
    _require(isinstance(raw_uniforms, list), f"{path}.uniforms: expected array")
    _require(len(raw_uniforms) == frames * stride,
             f"{path}: uniform count does not equal frames*stride")
    uniforms = [_f32(value, f"{path}.uniforms[{index}]")
                for index, value in enumerate(raw_uniforms)]
    _require(all(0.0 <= value <= 1.0 for value in uniforms),
             f"{path}.uniforms: values must lie in [0,1]")
    return ({
        "path": str(path.resolve()),
        "sha256": envelope["sha256"],
        "format": tape["format"],
        "seed": seed,
        "frames": frames,
        "stride": stride,
        "generator": tape["generator"],
    }, uniforms)


def _native_draw_record(tape: dict[str, Any], uniforms: list[float], step: int,
                        call: str) -> dict[str, Any]:
    _require(call in DRAW_SLOTS, f"unsupported native sampler call {call!r}")
    slot = DRAW_SLOTS[call]
    flat_index = step * DRAW_STRIDE + slot
    _require(flat_index < len(uniforms), f"native random tape has no draw for step {step}, slot {slot}")
    return {
        "tape_sha256": tape["sha256"],
        "step": step,
        "slot": slot,
        "call": call,
        "flat_index": flat_index,
        "uniform": uniforms[flat_index],
    }


def _native_token_mapping(report: dict[str, Any], expected_config_sha256: str) -> dict[str, int]:
    model_path = report.get("model_path")
    _require(isinstance(model_path, str) and model_path,
             "source report must identify model_path for RAS token mapping")
    config_path = Path(model_path) / "config.json"
    _, envelope = _read_json_once(config_path)
    _require(envelope["sha256"] == expected_config_sha256,
             "source model config hash differs from trace checkpoint manifest")
    config = envelope["value"]
    _require(isinstance(config, dict), "source model config must be an object")
    eos = _strict_int(config.get("eos_token_id"), "config.eos_token_id", minimum=0)
    semantic_begin = _strict_int(config.get("semantic_start_token_id"),
                                 "config.semantic_start_token_id", minimum=0)
    semantic_end = _strict_int(config.get("semantic_end_token_id"),
                               "config.semantic_end_token_id", minimum=semantic_begin)
    _require(eos < semantic_begin, "EOS token must precede the semantic token range")
    return {"eos_token_id": eos, "semantic_start_token_id": semantic_begin,
            "semantic_end_token_id": semantic_end}


def _logit_stats(actual: list[float], reference: list[float]) -> dict[str, Any]:
    violations = 0
    max_abs = 0.0
    max_ratio = 0.0
    max_index = 0
    for index, (left, right) in enumerate(zip(actual, reference)):
        delta = abs(left - right)
        allowed = ATOL + RTOL * abs(right)
        ratio = delta / allowed if allowed > 0 else math.inf
        if delta > max_abs:
            max_abs, max_ratio, max_index = delta, ratio, index
        if delta > allowed:
            violations += 1
    return {
        "count": len(actual),
        "violations": violations,
        "within_tolerance": violations == 0,
        "max_abs": max_abs,
        "max_tolerance_ratio": max_ratio,
        "max_index": max_index,
        "atol": ATOL,
        "rtol": RTOL,
    }


def _selected_steps(native: dict[str, Any], upstream: dict[str, Any], requested: Iterable[int] | None,
                    all_common_steps: bool) -> list[int]:
    common = sorted(set(native["capture_steps"]) & set(upstream["capture_steps"]))
    if all_common_steps:
        selected = [step for step in common if step <= min(native["frame_count"], upstream["frame_count"])]
    elif requested is not None:
        selected = sorted(set(requested))
    else:
        selected = list(DEFAULT_STEPS)
    _require(bool(selected), "no sampling steps selected")
    for step in selected:
        _strict_int(step, "selected step", minimum=0)
        _require(step in common, f"step {step} is not present in both trace bundles")
        _require(step <= min(native["frame_count"], upstream["frame_count"]),
                 f"step {step} is outside the common continuation prefix")
        for arm, bundle in (("native", native), ("upstream", upstream)):
            _require(f"generation.step{step}.main_logits" in bundle["traces"],
                     f"{arm}: missing main logits at step {step}")
            if step < bundle["frame_count"]:
                for head in range(9):
                    _require(f"generation.step{step}.tower_logits.{head}" in bundle["traces"],
                             f"{arm}: missing tower head {head} at step {step}")
    return selected


def build_fixture(native_trace: Path, upstream_trace: Path, *, steps: Iterable[int] | None = None,
                  all_common_steps: bool = False, mlx_root: Path = PINNED_MLX_ROOT,
                  random_tape_path: Path | None = None) -> dict[str, Any]:
    native, _ = _read_trace(native_trace, "native")
    upstream, _ = _read_trace(upstream_trace, "upstream")
    _require(native["source_report"] == upstream["source_report"], "source report identity mismatch")
    _require(native["prompt_ids"] == upstream["prompt_ids"], "prompt IDs differ")
    _require(native["prompt_rows"] == upstream["prompt_rows"], "prompt row counts differ")
    _require(native["frames"] == upstream["frames"], "teacher-forced frame rows differ")
    _require(native["model_config"] == upstream["model_config"], "model configuration differs")
    _require(native["model_files"] == upstream["model_files"], "model checkpoint manifest differs")
    _require(upstream["model"] == PINNED_MODEL, "upstream model identity mismatch")
    _require(upstream["upstream"] == PINNED_UPSTREAM, "upstream repository/revision mismatch")
    _require(isinstance(upstream["source_hashes"], dict), "upstream source_hashes missing")
    _require(upstream["source_hashes"].get("fish_omni_tower") == PINNED_TOWER_SHA256,
             "trace tower.py source hash mismatch")
    _require(upstream["generation_parameters"] == {"seed": 1, "temperature": NORMAL[0],
                                                     "top_k": NORMAL[1], "top_p": NORMAL[2]},
             "upstream generation parameters mismatch")
    _require(upstream["execution"].get("device") == "Device(gpu, 0)",
             "upstream common-prefix trace was not recorded on the default GPU")
    _require(upstream["execution"].get("float_checkpoint_weights_promoted_to_float32") is True,
             "upstream trace is not the declared F32 replay")
    report = _read_and_validate_source_report(native, upstream)
    sampler_meta = _pinned_sampler_metadata(mlx_root)
    selected = _selected_steps(native, upstream, steps, all_common_steps)

    random_tape = None
    tape_uniforms: list[float] | None = None
    token_mapping = None
    ras_checks: list[dict[str, Any]] = []
    if random_tape_path is not None:
        random_tape, tape_uniforms = _read_native_random_tape(random_tape_path)
        sampled_steps = [step for step in selected if step < native["frame_count"]]
        _require(not sampled_steps or max(sampled_steps) < random_tape["frames"],
                 "random tape is shorter than the selected generated-frame prefix")
        token_mapping = _native_token_mapping(report["value"], native["model_files"]["config.json"])
        report_frames = report["value"].get("frames")
        _require(isinstance(report_frames, list) and len(report_frames) == native["frame_count"],
                 "random-tape RAS check requires the full reported native frame sequence")
        for step in sampled_steps:
            _require(isinstance(report_frames[step], list) and len(report_frames[step]) == 11,
                     f"source report frame {step} has an invalid code width")
            history = (native["prompt_ids"][-16:]
                       + [row[0] for row in report_frames[:step]])[-16:]
            ras_checks.append({
                "step": step,
                "window_size": 16,
                "window_main_tokens": history,
                "reported_main_token": _strict_int(report_frames[step][0],
                                                    f"source_report.frames[{step}][0]", minimum=0),
                "main_case": f"native.step{step}.main",
                "retry_case": f"native.step{step}.main_retry",
                "main_draw_index": step * DRAW_STRIDE + DRAW_SLOTS["main"],
                "retry_draw_index": step * DRAW_STRIDE + DRAW_SLOTS["main_retry"],
            })

    case_list: list[dict[str, Any]] = []
    pair_list: list[dict[str, Any]] = []
    logit_pairs: list[dict[str, Any]] = []
    for step in selected:
        roles: list[tuple[str, str, tuple[float, int, float]]] = [
            ("main", "main", NORMAL),
            ("main_retry", "main", RETRY),
        ]
        if step < native["frame_count"]:
            roles.extend((f"tower_head_{head}", f"tower_head_{head}", NORMAL) for head in range(9))
        seen_logit_role: set[str] = set()
        for case_suffix, role, settings in roles:
            stage = f"generation.step{step}.main_logits" if role == "main" else (
                f"generation.step{step}.tower_logits.{int(role.rsplit('_', 1)[1])}"
            )
            native_logits = native["traces"][stage]
            upstream_logits = upstream["traces"][stage]
            native_name = f"native.step{step}.{case_suffix}"
            upstream_name = f"upstream.step{step}.{case_suffix}"
            native_draw = None
            if tape_uniforms is not None and step < native["frame_count"]:
                native_draw = _native_draw_record(random_tape, tape_uniforms, step, case_suffix)
            case_list.append(_case(name=native_name, role=role, source_arm="native", step=step,
                                   case_kind="trace", logits=native_logits, settings=settings,
                                   native_draw=native_draw))
            case_list.append(_case(name=upstream_name, role=role, source_arm="upstream", step=step,
                                   case_kind="trace", logits=upstream_logits, settings=settings,
                                   native_draw=native_draw))
            pair_list.append({
                "name": f"step{step}.{case_suffix}",
                "role": role,
                "step": step,
                "native_case": native_name,
                "upstream_case": upstream_name,
                "settings": {"temperature": _f32(settings[0], "temperature"),
                             "top_k": settings[1], "top_p": _f32(settings[2], "top_p")},
            })
            if role not in seen_logit_role:
                logit_pairs.append({
                    "name": f"step{step}.{role}",
                    "role": role,
                    "step": step,
                    "stage": stage,
                    "comparison": _logit_stats(native_logits, upstream_logits),
                })
                seen_logit_role.add(role)

    # A controlled exact-tie case distinguishes the native lower-ID tie-break
    # from MLX argpartition's unspecified boundary-tie identity. It is not a
    # model-derived row and is explicitly tagged as synthetic.
    tie_logits = [_f32(10.0 if index < 5 else 0.0, "synthetic.tie_logit") for index in range(64)]
    case_list.append(_case(
        name="synthetic.topk_boundary_tie",
        role="main",
        source_arm="synthetic",
        step=None,
        case_kind="synthetic_topk_boundary_tie",
        logits=tie_logits,
        settings=(1.0, 4, 1.0),
    ))

    source_report_sha = report["sha256"]
    source_files = {
        "native": {"path": native["path"], "sha256": native["sha256"]},
        "upstream": {"path": upstream["path"], "sha256": upstream["sha256"]},
    }
    return {
        "format": FIXTURE_FORMAT,
        "provenance": {
            "source_report": report,
            "source_traces": source_files,
            "model": PINNED_MODEL,
            "upstream": PINNED_UPSTREAM,
            "model_files": native["model_files"],
            "upstream_source_hashes": upstream["source_hashes"],
            "sampler_source": sampler_meta,
            "native_sampler_source": "src/ml/audio/drama3/kernels/resident.metal::dr3_sample",
        "native_sampler_source_sha256": hashlib.sha256(
            (ROOT / "src/ml/audio/drama3/kernels/resident.metal").read_bytes()
        ).hexdigest(),
            "seeded_rng_parity_claimed": False,
            "native_random_tape": random_tape,
        },
        "sampling_contract": _canonical_sampling_contract(),
        "generation": {
            "prompt_rows": native["prompt_rows"],
            "frame_count": native["frame_count"],
            "capture_steps": selected,
            "model_config": native["model_config"],
            "input_report_sha256": source_report_sha,
            "ras_token_mapping": token_mapping,
            "ras_window_size": 16 if random_tape is not None else None,
            "ras_checks": ras_checks,
        },
        "input_logit_comparisons": logit_pairs,
        "pairs": pair_list,
        "cases": case_list,
    }


def _flat_float_array(value: Any, where: str, count: int) -> list[float]:
    return _number_list(value, where, count)


def _validate_case_input(case: Any, where: str) -> dict[str, Any]:
    _require(isinstance(case, dict), f"{where}: expected object")
    name = case.get("name")
    _require(isinstance(name, str) and bool(name), f"{where}.name: expected nonempty string")
    role = case.get("role")
    _require(isinstance(role, str) and bool(role), f"{where}.role: expected nonempty string")
    source_arm = case.get("source_arm")
    _require(source_arm in ("native", "upstream", "synthetic"), f"{where}.source_arm: invalid value")
    step = case.get("step")
    _require(step is None or (isinstance(step, int) and not isinstance(step, bool) and step >= 0),
             f"{where}.step: invalid value")
    kind = case.get("case_kind")
    _require(kind in ("trace", "synthetic_sampler_vector", "synthetic_topk_boundary_tie"),
             f"{where}.case_kind: invalid value")
    logits = case.get("logits")
    _require(isinstance(logits, list) and len(logits) >= 2, f"{where}.logits: expected vector")
    logits_f32 = [_f32(value, f"{where}.logits[{i}]") for i, value in enumerate(logits)]
    temperature = _f32(case.get("temperature"), f"{where}.temperature")
    _require(temperature > 0, f"{where}.temperature: must be positive")
    top_k = _strict_int(case.get("top_k"), f"{where}.top_k", minimum=1)
    _require(top_k <= len(logits_f32), f"{where}.top_k exceeds vocabulary")
    top_p = _f32(case.get("top_p"), f"{where}.top_p")
    _require(0 < top_p <= 1, f"{where}.top_p outside (0,1]")
    native_draw_raw = case.get("native_draw")
    has_native_draw = native_draw_raw is not None
    draws = _number_list(case.get("draws"), f"{where}.draws",
                         len(DRAW_VALUES) + int(has_native_draw))
    _require(draws[:len(DRAW_VALUES)] == [_f32(value, "draw") for value in DRAW_VALUES],
             f"{where}.draws: boundary schedule mismatch")
    native_draw = None
    if has_native_draw:
        _require(isinstance(native_draw_raw, dict), f"{where}.native_draw: expected object")
        tape_sha = _sha(native_draw_raw.get("tape_sha256"), f"{where}.native_draw.tape_sha256")
        draw_step = _strict_int(native_draw_raw.get("step"), f"{where}.native_draw.step", minimum=0)
        slot = _strict_int(native_draw_raw.get("slot"), f"{where}.native_draw.slot", minimum=0)
        call = native_draw_raw.get("call")
        _require(isinstance(call, str) and call in DRAW_SLOTS, f"{where}.native_draw.call: invalid sampler slot")
        _require(slot == DRAW_SLOTS[call], f"{where}.native_draw.slot/call mismatch")
        flat_index = _strict_int(native_draw_raw.get("flat_index"),
                                 f"{where}.native_draw.flat_index", minimum=0)
        _require(draw_step == step, f"{where}.native_draw.step differs from case step")
        _require(flat_index == draw_step * DRAW_STRIDE + slot,
                 f"{where}.native_draw.flat_index disagrees with step/slot")
        _require(source_arm in ("native", "upstream") and kind == "trace",
                 f"{where}.native_draw is valid only for trace-backed model arms")
        if call == "main":
            _require(name.endswith(".main"), f"{where}.native_draw.call disagrees with case name")
        elif call == "main_retry":
            _require(name.endswith(".main_retry"), f"{where}.native_draw.call disagrees with case name")
        else:
            _require(name.endswith(f".{call}"), f"{where}.native_draw.call disagrees with case name")
        uniform = _f32(native_draw_raw.get("uniform"), f"{where}.native_draw.uniform")
        _require(uniform == draws[-1], f"{where}.native_draw.uniform differs from final draw")
        native_draw = {"tape_sha256": tape_sha, "step": draw_step, "slot": slot,
                       "call": call, "flat_index": flat_index, "uniform": uniform}
    noise = case.get("noise_uniforms")
    _require(isinstance(noise, list) and len(noise) == len(draws),
             f"{where}.noise_uniforms: expected one row per draw")
    noise_rows: list[list[float]] = []
    for index, row in enumerate(noise):
        values = _number_list(row, f"{where}.noise_uniforms[{index}]", top_k)
        _require(all(0 < value < 1 for value in values),
                 f"{where}.noise_uniforms[{index}]: Gumbel uniforms must be in open interval (0,1)")
        noise_rows.append(values)
    return {
        "name": name, "role": role, "source_arm": source_arm, "step": step, "case_kind": kind,
        "logits": logits_f32, "vocab_size": len(logits_f32), "temperature": temperature,
        "top_k": top_k, "top_p": top_p, "draws": draws, "noise_uniforms": noise_rows,
        "native_draw": native_draw,
    }


def _validate_sampling_contract(fixture: dict[str, Any]) -> dict[str, tuple[float, int, float]]:
    raw = fixture.get("sampling_contract")
    expected = _canonical_sampling_contract()
    _require(_same_typed_json_value(raw, expected),
             "fixture.sampling_contract is malformed or differs from the canonical sampler contract")
    return {
        "normal": (_f32(NORMAL[0], "sampling_contract.normal.temperature"), NORMAL[1],
                   _f32(NORMAL[2], "sampling_contract.normal.top_p")),
        "main_retry": (_f32(RETRY[0], "sampling_contract.main_retry.temperature"), RETRY[1],
                       _f32(RETRY[2], "sampling_contract.main_retry.top_p")),
    }


def _validate_case_sampling_settings(
    case: dict[str, Any], contract: dict[str, tuple[float, int, float]], where: str
) -> tuple[float, int, float]:
    if case["case_kind"] == "trace":
        _require(case["source_arm"] in ("native", "upstream") and case["step"] is not None,
                 f"{where}: trace case needs a model arm and step")
        call = case["name"].split(".")[-1]
        _require(case["name"] == f"{case['source_arm']}.step{case['step']}.{call}",
                 f"{where}.name: trace case identity is not canonical")
        if call == "main":
            _require(case["role"] == "main", f"{where}.role disagrees with main sampler call")
            policy = "normal"
        elif call == "main_retry":
            _require(case["role"] == "main", f"{where}.role disagrees with retry sampler call")
            policy = "main_retry"
        elif re.fullmatch(r"tower_head_[0-8]", call):
            _require(case["role"] == call, f"{where}.role disagrees with tower sampler call")
            policy = "normal"
        else:
            raise InvalidBundle(f"{where}.name: unknown trace sampler call")
        expected = contract[policy]
    elif case["case_kind"] == "synthetic_topk_boundary_tie":
        expected = (_f32(1.0, "synthetic.temperature"), 4, _f32(1.0, "synthetic.top_p"))
        _require(
            case["name"] == "synthetic.topk_boundary_tie" and case["source_arm"] == "synthetic"
            and case["role"] == "main" and case["step"] is None,
            f"{where}: synthetic sampler identity is not canonical",
        )
    else:
        expected = contract["normal"]
        _require(
            case["name"] == "synthetic.unit_vector" and case["source_arm"] == "synthetic"
            and case["role"] == "main" and case["step"] is None,
            f"{where}: synthetic sampler identity is not canonical",
        )

    actual = (case["temperature"], case["top_k"], case["top_p"])
    _require(actual == expected,
             f"{where}: sampler settings disagree with canonical "
             f"{policy if case['case_kind'] == 'trace' else 'synthetic'} contract")
    return expected


def _validate_source_trace_bindings(
    fixture: dict[str, Any], cases: dict[str, dict[str, Any]]
) -> None:
    """Bind every promotable trace case to the exact retained source trace bytes."""
    trace_cases = {name: case for name, case in cases.items() if case["case_kind"] == "trace"}
    if not trace_cases:
        return

    generation = fixture.get("generation")
    _require(type(generation) is dict,
             "trace cases require generation metadata and hash-bound source traces")
    provenance = fixture.get("provenance")
    _require(type(provenance) is dict, "trace cases require fixture provenance")
    source_traces = provenance.get("source_traces")
    _require(type(source_traces) is dict and set(source_traces) == {"native", "upstream"},
             "trace cases require native and upstream source trace references")

    loaded: dict[str, dict[str, Any]] = {}
    for arm in ("native", "upstream"):
        reference = source_traces[arm]
        _require(type(reference) is dict and set(reference) == {"path", "sha256"},
                 f"fixture.provenance.source_traces.{arm}: expected path and SHA-256")
        path_raw = reference.get("path")
        _require(isinstance(path_raw, str) and bool(path_raw),
                 f"fixture.provenance.source_traces.{arm}.path is missing")
        expected_sha = _sha(reference.get("sha256"),
                            f"fixture.provenance.source_traces.{arm}.sha256")
        source, source_bundle = _read_trace(Path(path_raw), arm)
        _require(source["sha256"] == expected_sha,
                 f"fixture {arm} source trace SHA-256 mismatch")
        _require(Path(path_raw).resolve() == Path(source["path"]),
                 f"fixture {arm} source trace path does not resolve consistently")
        del source_bundle
        loaded[arm] = source

    native = loaded["native"]
    upstream = loaded["upstream"]
    for key in ("source_report", "prompt_ids", "prompt_rows", "frames", "frame_count",
                "model_config", "model_files"):
        _require(native[key] == upstream[key],
                 f"native/upstream source traces disagree at {key}")
    _require(upstream["model"] == PINNED_MODEL, "source trace model identity mismatch")
    _require(upstream["upstream"] == PINNED_UPSTREAM, "source trace upstream revision mismatch")
    _require(isinstance(upstream["source_hashes"], dict) and
             upstream["source_hashes"].get("fish_omni_tower") == PINNED_TOWER_SHA256,
             "source trace pinned tower.py hash mismatch")
    _require(upstream["generation_parameters"] == {
        "seed": 1, "temperature": NORMAL[0], "top_k": NORMAL[1], "top_p": NORMAL[2]
    }, "source trace generation parameters mismatch")
    _require(upstream["execution"].get("device") == "Device(gpu, 0)" and
             upstream["execution"].get("float_checkpoint_weights_promoted_to_float32") is True,
             "source trace is not the declared rowwise F32 GPU replay")
    _require(provenance.get("model") == upstream["model"] and
             provenance.get("upstream") == upstream["upstream"] and
             provenance.get("model_files") == native["model_files"] and
             provenance.get("upstream_source_hashes") == upstream["source_hashes"],
             "fixture model provenance differs from hash-verified source traces")

    source_report = _read_and_validate_source_report(native, upstream)
    source_report_provenance = provenance.get("source_report")
    _require(_same_typed_json_value(source_report_provenance, source_report),
             "fixture source report provenance differs from the hash-verified report")
    _require(generation.get("input_report_sha256") == source_report["sha256"],
             "fixture generation input report hash differs from source traces")
    _require(generation.get("frame_count") == native["frame_count"] and
             generation.get("prompt_rows") == native["prompt_rows"] and
             generation.get("model_config") == native["model_config"],
             "fixture generation metadata differs from source traces")

    capture_steps = _int_list(generation.get("capture_steps"), "generation.capture_steps")
    _require(bool(capture_steps) and capture_steps == sorted(set(capture_steps)),
             "generation.capture_steps must be nonempty, sorted, and unique")
    case_steps = sorted({case["step"] for case in trace_cases.values()})
    _require(case_steps == capture_steps,
             "generation.capture_steps do not match trace-backed sampler cases")
    for step in capture_steps:
        _require(step <= native["frame_count"],
                 f"generation capture step {step} exceeds the source continuation prefix")
        _require(step in native["capture_steps"] and step in upstream["capture_steps"],
                 f"generation capture step {step} is absent from a source trace")

    expected_logit_pairs: list[dict[str, Any]] = []
    case_roles = sorted({(case["step"], case["role"]) for case in trace_cases.values()})
    for step, role in case_roles:
        _require(step is not None, "trace case has no source prediction step")
        _require(role == "main" or step < native["frame_count"],
                 f"{role} is not available at terminal step {step}")
        if role == "main" or re.fullmatch(r"tower_head_[0-8]", role):
            stage = (f"generation.step{step}.main_logits" if role == "main" else
                     f"generation.step{step}.tower_logits.{role.rsplit('_', 1)[1]}")
        else:
            raise InvalidBundle(f"trace case has unsupported source role {role!r}")
        native_logits = native["traces"].get(stage)
        upstream_logits = upstream["traces"].get(stage)
        _require(native_logits is not None and upstream_logits is not None,
                 f"source traces are missing {stage}")
        expected_logit_pairs.append({
            "name": f"step{step}.{role}",
            "role": role,
            "step": step,
            "stage": stage,
            "comparison": _logit_stats(native_logits, upstream_logits),
        })

    for name, case in trace_cases.items():
        stage = (f"generation.step{case['step']}.main_logits" if case["role"] == "main" else
                 f"generation.step{case['step']}.tower_logits.{case['role'].rsplit('_', 1)[1]}")
        source_logits = loaded[case["source_arm"]]["traces"].get(stage)
        _require(source_logits is not None,
                 f"{name}: stage {stage} is absent from the hash-verified source trace")
        _require(case["logits"] == source_logits,
                 f"{name}: logits differ from the hash-verified source trace stage")

    logit_comparisons = fixture.get("input_logit_comparisons")
    _require(_same_typed_json_value(logit_comparisons, expected_logit_pairs),
             "fixture input logit comparisons differ from source-trace recomputation")


def _pair_settings(value: Any, where: str) -> tuple[float, int, float]:
    _require(type(value) is dict and set(value) == {"temperature", "top_k", "top_p"},
             f"{where}: expected temperature/top_k/top_p settings")
    return (
        _f32(value.get("temperature"), f"{where}.temperature"),
        _strict_int(value.get("top_k"), f"{where}.top_k", minimum=1),
        _f32(value.get("top_p"), f"{where}.top_p"),
    )


def _validate_fixture_pairs(
    fixture: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    case_settings: dict[str, tuple[float, int, float]],
) -> None:
    pairs = fixture.get("pairs", [])
    _require(isinstance(pairs, list), "fixture.pairs: expected array")
    expected_by_name: dict[str, tuple[str, str]] = {}
    trace_cases = {name: case for name, case in cases.items() if case["case_kind"] == "trace"}
    enforce_complete = bool(trace_cases)
    if enforce_complete:
        _require(bool(pairs), "trace cases require complete native/upstream source pairs")
        _require("generation" in fixture,
                 "trace cases require generation metadata and source-trace provenance")
    if enforce_complete:
        for name, case in trace_cases.items():
            if case["source_arm"] == "native":
                suffix = name.split(".")[-1]
                upstream_name = "upstream." + name[len("native."):]
                _require(upstream_name in trace_cases,
                         f"fixture.pairs: missing upstream counterpart for {name}")
                _require(trace_cases[upstream_name]["role"] == case["role"]
                         and trace_cases[upstream_name]["step"] == case["step"]
                         and case_settings[upstream_name] == case_settings[name],
                         f"fixture.pairs: paired cases disagree for {name}")
                expected_by_name[f"step{case['step']}.{suffix}"] = (name, upstream_name)
            else:
                native_name = "native." + name[len("upstream."):]
                _require(native_name in trace_cases,
                         f"fixture.pairs: missing native counterpart for {name}")

    seen_names: set[str] = set()
    seen_cases: set[str] = set()
    for index, pair in enumerate(pairs):
        where = f"fixture.pairs[{index}]"
        _require(type(pair) is dict, f"{where}: expected object")
        name = pair.get("name")
        _require(isinstance(name, str) and name not in seen_names,
                 f"{where}.name: missing or duplicate pair name")
        native_name = pair.get("native_case")
        upstream_name = pair.get("upstream_case")
        _require(isinstance(native_name, str) and isinstance(upstream_name, str)
                 and native_name in trace_cases and upstream_name in trace_cases,
                 f"{where}: references missing trace cases")
        native = trace_cases[native_name]
        upstream = trace_cases[upstream_name]
        _require(native["source_arm"] == "native" and upstream["source_arm"] == "upstream",
                 f"{where}: pair arms are reversed or invalid")
        _require(native_name == f"native.{upstream_name[len('upstream.'):]}",
                 f"{where}: paired case identities disagree")
        step = _strict_int(pair.get("step"), f"{where}.step", minimum=0)
        role = pair.get("role")
        _require(step == native["step"] == upstream["step"] and role == native["role"] == upstream["role"],
                 f"{where}: pair step/role disagrees with its cases")
        call = native_name.split(".")[-1]
        _require(name == f"step{step}.{call}", f"{where}.name disagrees with paired case identity")
        settings = _pair_settings(pair.get("settings"), f"{where}.settings")
        _require(settings == case_settings[native_name] == case_settings[upstream_name],
                 f"{where}.settings disagree with canonical sampling contract or paired cases")
        seen_names.add(name)
        seen_cases.update((native_name, upstream_name))

    if enforce_complete:
        _require(seen_names == set(expected_by_name),
                 "fixture.pairs do not cover every native/upstream trace case")
        expected_cases = {name for pair in expected_by_name.values() for name in pair}
        _require(seen_cases == expected_cases,
                 "fixture.pairs omit or reuse native/upstream trace cases")


def _ranked_logits(case: dict[str, Any]) -> list[float]:
    # Exact ties are intentionally retained as equal values; the returned list
    # is only a score multiset and does not choose a token identity.
    scaled = [_f32(value / case["temperature"], "scaled logit") for value in case["logits"]]
    return sorted(scaled, reverse=True)


def _tie_info(case: dict[str, Any]) -> dict[str, Any]:
    ranked = _ranked_logits(case)
    k = case["top_k"]
    groups: list[dict[str, Any]] = []
    start = 0
    while start < k:
        end = start + 1
        while end < len(ranked) and ranked[end] == ranked[start]:
            end += 1
        if end - start > 1:
            groups.append({"rank_start": start, "rank_end_exclusive": end,
                           "scaled_logit": ranked[start], "crosses_top_k": start < k < end})
        start = end
    return {
        "top_k_boundary_tie": k < len(ranked) and ranked[k - 1] == ranked[k],
        "top_k_tie_groups": groups,
    }


def _cdf_checks(values: list[float], where: str, *, upper: float = 1.0001) -> dict[str, Any]:
    """Report arithmetic CDF invariant failures without discarding valid F32 data.

    Shape and finiteness are checked before this function. The historical
    monotonicity and range thresholds remain unchanged; violations are
    carried into the comparison report so probability, ID, and RAS checks can
    still run on the same terminal output.
    """
    minimum = -1e-7
    monotonic_tolerance = 1e-7
    previous: float | None = None
    violations: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if value < minimum:
            violations.append({
                "kind": "below_minimum",
                "index": index,
                "value": value,
                "minimum": minimum,
            })
        if previous is not None and value + monotonic_tolerance < previous:
            violations.append({
                "kind": "decrease_exceeds_tolerance",
                "index": index,
                "previous_index": index - 1,
                "previous": previous,
                "value": value,
                "decrease": previous - value,
                "tolerance": monotonic_tolerance,
            })
        if value > upper:
            violations.append({
                "kind": "above_maximum",
                "index": index,
                "value": value,
                "maximum": upper,
            })
        previous = value
    return {
        "status": "fail" if violations else "pass",
        "monotonic_tolerance": monotonic_tolerance,
        "minimum": minimum,
        "maximum": upper,
        "violations": violations,
    }


def _expected_softmax(case: dict[str, Any], top_ids: list[int], mask: list[bool]) -> dict[str, Any]:
    """Independent high-precision check of serialized top-k softmax diagnostics."""
    scaled = [_f32(value / case["temperature"], "expected scaled logit") for value in case["logits"]]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    denominator = math.fsum(weights)
    full_probabilities = [weights[token] / denominator for token in top_ids]
    full_cdf: list[float] = []
    total = 0.0
    for probability in full_probabilities:
        total += probability
        full_cdf.append(total)

    retained_mass = math.fsum(weights[token] for token, keep in zip(top_ids, mask) if keep)
    conditional_probabilities = [
        weights[token] / retained_mass if keep else 0.0
        for token, keep in zip(top_ids, mask)
    ]
    conditional_cdf: list[float] = []
    cumulative = 0.0
    for probability, keep in zip(conditional_probabilities, mask):
        if keep:
            cumulative += probability
            conditional_cdf.append(cumulative)
        else:
            conditional_cdf.append(0.0)
    return {
        "normalization": denominator,
        "full_probabilities": full_probabilities,
        "full_cdf": full_cdf,
        "retained_conditional_probabilities": conditional_probabilities,
        "retained_conditional_cdf": conditional_cdf,
    }


def _array_stats(actual: list[float], reference: list[float]) -> dict[str, Any]:
    _require(len(actual) == len(reference), "internal probability comparison length mismatch")
    return _logit_stats(actual, reference)


def _id_vector(value: Any, where: str, count: int, vocab: int, *, unique: bool) -> list[int]:
    ids = _int_list(value, where, count)
    if unique:
        _require(len(set(ids)) == len(ids), f"{where}: duplicate token ID")
    _require(all(0 <= token < vocab for token in ids), f"{where}: token ID outside vocabulary")
    return ids


def _validate_algorithm_record(raw: Any, case: dict[str, Any], prefix: str, where: str) -> dict[str, Any]:
    _require(isinstance(raw, dict), f"{where}: expected object")
    k = case["top_k"]
    vocab = case["vocab_size"]
    fields = {
        "sample_ids": (len(case["draws"]), "ids"),
        "topk_ids": (k, "ids"),
        "topk_scaled_logits": (k, "numbers"),
        "kept_mask": (k, "bools"),
        "kept_count": (None, "int"),
        "full_vocab_normalization": (None, "number"),
        "full_probabilities": (k, "numbers"),
        "full_cdf": (k, "numbers"),
        "retained_conditional_probabilities": (k, "numbers"),
        "retained_conditional_cdf": (k, "numbers"),
        "shared_gumbel_ids": (len(case["draws"]), "ids"),
    }
    normalized: dict[str, Any] = {}
    for field, (length, kind) in fields.items():
        value = raw.get(f"{prefix}_{field}")
        label = f"{where}.{prefix}_{field}"
        if kind == "ids":
            normalized[field] = _id_vector(value, label, int(length), vocab,
                                           unique=field == "topk_ids")
        elif kind == "bools":
            _require(isinstance(value, list) and len(value) == int(length), f"{label}: expected {length} booleans")
            _require(all(isinstance(item, bool) for item in value), f"{label}: expected booleans")
            normalized[field] = list(value)
        elif kind == "int":
            normalized[field] = _strict_int(value, label, minimum=1)
        elif kind == "number":
            normalized[field] = _f32(value, label)
        else:
            assert length is not None
            normalized[field] = _number_list(value, label, int(length))

    mask = normalized["kept_mask"]
    count = normalized["kept_count"]
    _require(sum(mask) == count, f"{where}.{prefix}_kept_count disagrees with kept_mask")
    _require(count > 0, f"{where}.{prefix}_kept_count must be positive")
    _require(mask[0] is True, f"{where}.{prefix}_kept_mask must retain rank 0")
    seen_drop = False
    for index, keep in enumerate(mask):
        if not keep:
            seen_drop = True
        _require(not (seen_drop and keep), f"{where}.{prefix}_kept_mask must be a retained prefix")

    top_ids = normalized["topk_ids"]
    scaled_values = [_f32(case["logits"][token] / case["temperature"], f"{where}.scaled[{token}]")
                     for token in top_ids]
    _require(
        _array_stats(normalized["topk_scaled_logits"], scaled_values)["within_tolerance"],
        f"{where}.{prefix}_topk_scaled_logits disagree with the input logits",
    )
    _require(all(scaled_values[i] >= scaled_values[i + 1] for i in range(k - 1)),
             f"{where}.{prefix}_topk_ids are not sorted by descending scaled logit")
    expected_scores = _ranked_logits(case)[:k]
    _require(sorted(scaled_values, reverse=True) == expected_scores,
             f"{where}.{prefix}_topk_ids are not a valid top-k score multiset")
    if prefix == "native":
        stable_expected = sorted(
            range(vocab),
            key=lambda token: (-_f32(case["logits"][token] / case["temperature"], "native top-k"), token),
        )[:k]
        _require(top_ids == stable_expected,
                 f"{where}.native_topk_ids violate stable lower-ID top-k membership/order")

    full_p = normalized["full_probabilities"]
    full_cdf = normalized["full_cdf"]
    cond_p = normalized["retained_conditional_probabilities"]
    cond_cdf = normalized["retained_conditional_cdf"]
    _require(all(0.0 <= item <= 1.0 for item in full_p), f"{where}.{prefix}_full_probabilities outside [0,1]")
    _require(all(0.0 <= item <= 1.0 for item in cond_p),
             f"{where}.{prefix}_retained_conditional_probabilities outside [0,1]")
    cdf_invariant_checks = {
        "full_cdf": _cdf_checks(full_cdf, f"{where}.{prefix}_full_cdf"),
        "retained_conditional_cdf": _cdf_checks(
            cond_cdf[:count], f"{where}.{prefix}_retained_conditional_cdf"
        ),
    }
    running = 0.0
    for index, value in enumerate(full_p):
        running += value
        _require(abs(full_cdf[index] - running) <= 2e-4,
                 f"{where}.{prefix}_full_cdf disagrees with full probabilities at rank {index}")
    running_conditional = 0.0
    for index in range(count):
        running_conditional += cond_p[index]
        _require(abs(cond_cdf[index] - running_conditional) <= 2e-4,
                 f"{where}.{prefix}_retained_conditional_cdf disagrees at rank {index}")
    _require(abs(sum(cond_p[:count]) - 1.0) <= 2e-4,
             f"{where}.{prefix}_retained_conditional_probabilities do not sum to one")
    _require(abs(cond_cdf[count - 1] - 1.0) <= 2e-4,
             f"{where}.{prefix}_retained_conditional_cdf does not end at one")
    _require(all(value == 0.0 for value in cond_p[count:]),
             f"{where}.{prefix}_retained_conditional_probabilities must zero dropped ranks")
    _require(all(value == 0.0 for value in cond_cdf[count:]),
             f"{where}.{prefix}_retained_conditional_cdf must zero dropped ranks")
    for index in range(k):
        expected_keep = (index == 0) or (full_cdf[index] <= case["top_p"])
        _require(mask[index] is expected_keep,
                 f"{where}.{prefix}_kept_mask disagrees with full CDF/top_p at rank {index}")

    expected = _expected_softmax(case, top_ids, mask)
    normalization_stats = _logit_stats(
        [normalized["full_vocab_normalization"]], [expected["normalization"]]
    )
    _require(normalization_stats["within_tolerance"],
             f"{where}.{prefix}_full_vocab_normalization disagrees with the input logits")
    math_stats = {
        field: _array_stats(normalized[field], expected[field])
        for field in (
            "full_probabilities",
            "full_cdf",
            "retained_conditional_probabilities",
            "retained_conditional_cdf",
        )
    }
    for field, stats in math_stats.items():
        _require(stats["within_tolerance"],
                 f"{where}.{prefix}_{field} disagrees with the input logits")

    expected_samples: list[int] = []
    kept_weights = [
        math.exp(
            _f32(case["logits"][token] / case["temperature"], "native sample scaled logit")
            - max(_ranked_logits(case))
        )
        for token, keep in zip(top_ids, mask)
        if keep
    ]
    kept_ids = [token for token, keep in zip(top_ids, mask) if keep]
    kept_total = math.fsum(kept_weights)
    for draw in case["draws"]:
        if draw == 1.0:
            expected_samples.append(kept_ids[-1])
            continue
        threshold = draw * kept_total
        cumulative = 0.0
        selected = kept_ids[-1]
        for token, weight in zip(kept_ids, kept_weights):
            cumulative += weight
            if threshold < cumulative:
                selected = token
                break
        expected_samples.append(selected)
    sample_mismatches = [
        index for index, (actual, expected_id) in enumerate(
            zip(normalized["sample_ids"], expected_samples)
        ) if actual != expected_id
    ]
    _require(not sample_mismatches,
             f"{where}.{prefix}_sample_ids disagree with fixed-draw inverse-CDF reference at draws {sample_mismatches[:4]}")

    expected_gumbel: list[int] = []
    gumbel_ambiguous_draws: list[int] = []
    maximum_scaled = max(_ranked_logits(case))
    for draw_index, uniforms in enumerate(case["noise_uniforms"]):
        scores: list[tuple[float, int, int]] = []
        for rank, (token, keep, uniform) in enumerate(zip(top_ids, mask, uniforms)):
            if not keep:
                continue
            scaled = _f32(case["logits"][token] / case["temperature"], "gumbel scaled logit")
            gumbel = _f32(-math.log(-math.log(uniform)), "gumbel reference")
            scores.append((_f32(scaled + gumbel, "gumbel score"), -rank, token))
        scores.sort(reverse=True)
        expected_gumbel.append(scores[0][2])
        if len(scores) > 1 and scores[0][0] - scores[1][0] <= 2e-6 * max(1.0, abs(scores[0][0])):
            gumbel_ambiguous_draws.append(draw_index)
    gumbel_mismatches = [
        index for index, (actual, expected_id) in enumerate(
            zip(normalized["shared_gumbel_ids"], expected_gumbel)
        ) if actual != expected_id and index not in gumbel_ambiguous_draws
    ]
    _require(not gumbel_mismatches,
             f"{where}.{prefix}_shared_gumbel_ids disagree with common-noise reference at draws {gumbel_mismatches[:4]}")
    retained_ids = {token for token, keep in zip(top_ids, mask) if keep}
    _require(all(token in retained_ids for token in normalized["sample_ids"]),
             f"{where}.{prefix}_sample_ids selected a dropped token")
    _require(all(token in retained_ids for token in normalized["shared_gumbel_ids"]),
             f"{where}.{prefix}_shared_gumbel_ids selected a dropped token")
    normalized["input_math_checks"] = {
        "normalization": normalization_stats,
        "probabilities_and_cdfs": math_stats,
        "cdf_invariants": cdf_invariant_checks,
        "inverse_cdf_draws_checked": len(expected_samples),
        "gumbel_draws_checked": len(expected_gumbel) - len(gumbel_ambiguous_draws),
        "gumbel_numerically_ambiguous_draws": gumbel_ambiguous_draws,
    }
    return normalized


def _validate_probe_lineage(probe: dict[str, Any], fixture: dict[str, Any]) -> None:
    _require(probe.get("sampler_kernel") == "dr3_sample", "probe did not invoke the production sampler kernel")
    _require(probe.get("diagnostic_kernel") == "dr3_sampler_compare", "probe diagnostic kernel mismatch")
    provenance = fixture.get("provenance")
    _require(isinstance(provenance, dict), "fixture.provenance: expected object")
    expected_sampler_sha = hashlib.sha256(
        (ROOT / "src/ml/audio/drama3/kernels/resident.metal").read_bytes()
    ).hexdigest()
    expected_diagnostic_sha = hashlib.sha256(
        (ROOT / "examples/support/drama3_sampler_compare.metal").read_bytes()
    ).hexdigest()
    _require(provenance.get("native_sampler_source_sha256") == expected_sampler_sha,
             "fixture native sampler source hash is stale")
    _require(probe.get("resident_metal_source_sha256") == expected_sampler_sha,
             "probe production Metal source hash mismatch")
    _require(probe.get("diagnostic_metal_source_sha256") == expected_diagnostic_sha,
             "probe diagnostic Metal source hash mismatch")


def _validate_queue_ledger(probe: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    queue = probe.get("queue")
    _require(isinstance(queue, dict), "probe.queue: expected object")
    before = queue.get("graph_before_work")
    pre_drain = queue.get("graph_before_final_drain")
    drained = queue.get("graph_after_final_drain")
    after_reads = queue.get("graph_after_terminal_reads")
    before_reads = queue.get("readbacks_before_final_drain")
    final_reads = queue.get("readbacks_after_terminal_reads")
    for name, value in (("graph_before_work", before), ("graph_before_final_drain", pre_drain),
                        ("graph_after_final_drain", drained), ("graph_after_terminal_reads", after_reads),
                        ("readbacks_before_final_drain", before_reads),
                        ("readbacks_after_terminal_reads", final_reads)):
        _require(isinstance(value, dict), f"probe.queue.{name}: expected object")
    _require(before.get("depth") == 2 and before.get("submitted") == 0 and
             before.get("completion_waits") == 0 and before.get("pending") == 0,
             "probe did not start with a clean depth-2 graph")
    _require(all(value == 0 for value in before_reads.values()),
             "probe performed a readback before the final drain")
    _require(pre_drain.get("submitted") == 1 and pre_drain.get("pending") == 1 and
             pre_drain.get("completion_waits") == 0 and queue.get("submissions_before_final_drain") == 1 and
             queue.get("predrain_waits") == 0,
             "probe submitted something other than one queued terminal batch")
    _require(drained.get("submitted") == 1 and drained.get("completed") == 1 and
             drained.get("pending") == 0 and drained.get("completion_waits") == 1 and
             queue.get("final_drain_waits") == 1,
             "probe did not drain exactly once after the terminal submission")
    for key in ("submitted", "completed", "completion_waits"):
        _require(after_reads.get(key) == drained.get(key),
                 f"terminal reads changed graph {key}")
    _require(after_reads.get("pending") == 0, "terminal reads left queued work")

    diagnostic_bytes = sum((7 * case["top_k"] + 3 + len(case["draws"])) * 4 for case in cases)
    terminal_bytes = sum(len(case["draws"]) * 4 for case in cases)
    control_reads = len(cases) * 2
    _require(queue.get("diagnostic_bytes") == diagnostic_bytes,
             "diagnostic terminal byte ledger mismatch")
    _require(queue.get("terminal_bytes") == terminal_bytes,
             "production-sampler terminal byte ledger mismatch")
    _require(queue.get("finite_status_reads_after_drain") == control_reads and
             queue.get("finite_status_readback_bytes") == control_reads * 4,
             "finite-status readback ledger mismatch")
    _require(final_reads.get("diagnostic") == before_reads.get("diagnostic", 0) + diagnostic_bytes,
             "diagnostic readback ledger mismatch")
    _require(final_reads.get("terminal") == before_reads.get("terminal", 0) + terminal_bytes,
             "terminal readback ledger mismatch")
    _require(final_reads.get("control") == before_reads.get("control", 0) + control_reads * 4,
             "control readback ledger mismatch")
    return {
        "submissions_before_final_drain": queue.get("submissions_before_final_drain"),
        "predrain_waits": queue.get("predrain_waits"),
        "final_drain_waits": queue.get("final_drain_waits"),
        "diagnostic_bytes": diagnostic_bytes,
        "terminal_bytes": terminal_bytes,
        "finite_status_reads_after_drain": control_reads,
    }


def _pair_probability_stats(left: dict[str, Any], right: dict[str, Any], field: str) -> dict[str, Any]:
    return _array_stats(left[field], right[field])


def _validate_random_tape_lineage(
    fixture: dict[str, Any], input_cases: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    provenance = fixture.get("provenance")
    _require(isinstance(provenance, dict), "fixture.provenance: expected object")
    tape_meta = provenance.get("native_random_tape")
    if tape_meta is None:
        _require(all(case["native_draw"] is None for case in input_cases.values()),
                 "fixture has native draws without native_random_tape provenance")
        generation = fixture.get("generation", {})
        _require(isinstance(generation, dict), "fixture.generation: expected object")
        _require(generation.get("ras_checks", []) == [],
                 "fixture has RAS checks without native_random_tape provenance")
        return None

    generation = fixture.get("generation")
    _require(isinstance(generation, dict), "fixture.generation: expected object")
    _require(isinstance(tape_meta, dict), "fixture.provenance.native_random_tape: expected object")
    tape_path_raw = tape_meta.get("path")
    _require(isinstance(tape_path_raw, str) and tape_path_raw,
             "native random tape provenance path is missing")
    tape, uniforms = _read_native_random_tape(Path(tape_path_raw))
    for key in ("path", "sha256", "format", "seed", "frames", "stride", "generator"):
        _require(tape.get(key) == tape_meta.get(key),
                 f"native random tape provenance mismatch at {key}")

    source_meta = provenance.get("source_report")
    _require(isinstance(source_meta, dict), "fixture source_report provenance is missing")
    source_path_raw = source_meta.get("path")
    _require(isinstance(source_path_raw, str) and source_path_raw,
             "fixture source_report path is missing")
    _, source_envelope = _read_json_once(Path(source_path_raw))
    _require(source_envelope["sha256"] == _sha(source_meta.get("sha256"), "source_report.sha256"),
             "source report SHA-256 mismatch")
    source_report = source_envelope["value"]
    _require(source_meta.get("value") == source_report,
             "fixture source_report value differs from the referenced source bytes")
    _require(isinstance(source_report, dict), "source report must be an object")
    prompt_ids = _int_list(source_report.get("prompt_ids"), "source_report.prompt_ids")
    frames_raw = source_report.get("frames")
    _require(isinstance(frames_raw, list), "source_report.frames: expected array")
    frames = [_int_list(row, f"source_report.frames[{index}]", 11)
              for index, row in enumerate(frames_raw)]
    frame_count = _strict_int(generation.get("frame_count"), "generation.frame_count", minimum=1)
    _require(len(frames) == frame_count, "source report frame count differs from fixture")
    _require(tape["frames"] >= frame_count,
             "native random tape is shorter than the source generation report")
    capture_steps = _int_list(generation.get("capture_steps"), "generation.capture_steps")
    _require(capture_steps == sorted(set(capture_steps)),
             "generation.capture_steps must be sorted and unique")

    for case in input_cases.values():
        expected_draw = None
        if case["case_kind"] == "trace" and case["step"] is not None and case["step"] < frame_count:
            name_parts = case["name"].split(".")
            _require(len(name_parts) == 3 and name_parts[0] == case["source_arm"] and
                     name_parts[1] == f"step{case['step']}",
                     f"{case['name']}: malformed random-tape case identity")
            call = name_parts[2]
            _require(call in DRAW_SLOTS, f"{case['name']}: unknown sampler call")
            if call.startswith("tower_head_"):
                _require(case["role"] == call, f"{case['name']}: role/call mismatch")
            else:
                _require(case["role"] == "main", f"{case['name']}: role/call mismatch")
            expected_draw = _native_draw_record(tape, uniforms, case["step"], call)
        _require(case["native_draw"] == expected_draw,
                 f"{case['name']}: native draw does not match the referenced random tape")

    model_files = provenance.get("model_files")
    _require(isinstance(model_files, dict), "fixture model_files provenance is missing")
    config_sha = _sha(model_files.get("config.json"), "model_files.config.json")
    token_mapping = _native_token_mapping(source_report, config_sha)
    _require(generation.get("ras_token_mapping") == token_mapping,
             "fixture RAS token mapping differs from the pinned model config")
    _require(generation.get("ras_window_size") == 16,
             "fixture RAS window size must be 16")

    expected_checks = []
    for step in capture_steps:
        if step >= frame_count:
            continue
        history = (prompt_ids[-16:] + [row[0] for row in frames[:step]])[-16:]
        expected_checks.append({
            "step": step,
            "window_size": 16,
            "window_main_tokens": history,
            "reported_main_token": frames[step][0],
            "main_case": f"native.step{step}.main",
            "retry_case": f"native.step{step}.main_retry",
            "main_draw_index": step * DRAW_STRIDE + DRAW_SLOTS["main"],
            "retry_draw_index": step * DRAW_STRIDE + DRAW_SLOTS["main_retry"],
        })
    _require(generation.get("ras_checks") == expected_checks,
             "fixture RAS checks differ from the referenced source report")
    required_names = {name for check in expected_checks
                      for name in (check["main_case"], check["retry_case"])}
    _require(required_names <= set(input_cases),
             "fixture omits a native main/retry case required for RAS replay")
    for step in capture_steps:
        if step > frame_count:
            continue
        calls = ["main", "main_retry"]
        if step < frame_count:
            calls.extend(f"tower_head_{head}" for head in range(9))
        for arm in ("native", "upstream"):
            expected = {f"{arm}.step{step}.{call}" for call in calls}
            _require(expected <= set(input_cases),
                     f"fixture omits random-tape sampler cases at {arm} step {step}")
    return {
        "tape": tape,
        "uniforms": uniforms,
        "source_report": source_report,
        "token_mapping": token_mapping,
        "checks": expected_checks,
    }


def _candidate_token(index: int, token_mapping: dict[str, int], where: str) -> int:
    _strict_int(index, where, minimum=0)
    if index == 0:
        return token_mapping["eos_token_id"]
    token = token_mapping["semantic_start_token_id"] + index - 1
    _require(token <= token_mapping["semantic_end_token_id"],
             f"{where}: candidate index maps outside the semantic token band")
    return token


def _run_native_ras_replay(
    context: dict[str, Any] | None,
    input_cases: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if context is None:
        return {"status": "not_applicable", "check_count": 0, "failures": [], "checks": []}
    token_mapping = context["token_mapping"]
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for check in context["checks"]:
        main_case = input_cases[check["main_case"]]
        retry_case = input_cases[check["retry_case"]]
        _require(main_case["source_arm"] == retry_case["source_arm"] == "native",
                 f"RAS step {check['step']}: expected native sampler cases")
        _require(main_case["native_draw"] is not None and retry_case["native_draw"] is not None,
                 f"RAS step {check['step']}: missing actual native draw")
        main_record = outputs[check["main_case"]]["native"]
        retry_record = outputs[check["retry_case"]]["native"]
        main_index = main_record["sample_ids"][-1]
        retry_index = retry_record["sample_ids"][-1]
        first_token = _candidate_token(main_index, token_mapping, f"RAS step {check['step']} main")
        retry_token = _candidate_token(retry_index, token_mapping, f"RAS step {check['step']} retry")
        history = check["window_main_tokens"]
        is_semantic = (token_mapping["semantic_start_token_id"] <= first_token <=
                       token_mapping["semantic_end_token_id"])
        retry_used = is_semantic and first_token in history
        selected_token = retry_token if retry_used else first_token
        passed = selected_token == check["reported_main_token"]
        main_draw = main_case["native_draw"]
        retry_draw = retry_case["native_draw"]
        results.append({
            "step": check["step"],
            "random_tape_sha256": context["tape"]["sha256"],
            "history_window_size": 16,
            "history_main_tokens": history,
            "main_draw": {"slot": main_draw["slot"], "flat_index": main_draw["flat_index"],
                          "uniform": main_draw["uniform"], "candidate_index": main_index,
                          "token_id": first_token},
            "retry_draw": {"slot": retry_draw["slot"], "flat_index": retry_draw["flat_index"],
                           "uniform": retry_draw["uniform"], "candidate_index": retry_index,
                           "token_id": retry_token},
            "retry_triggered": retry_used,
            "selected_token_id": selected_token,
            "reported_main_token_id": check["reported_main_token"],
            "matches_reported_frame": passed,
        })
        if not passed:
            failures.append(
                f"step {check['step']}: RAS replay selected {selected_token}, "
                f"reported frame contains {check['reported_main_token']}"
            )
    return {
        "status": "fail" if failures else "pass",
        "check_count": len(results),
        "failures": failures,
        "checks": results,
    }


def compare_probe(fixture: dict[str, Any], probe: dict[str, Any], *, fixture_sha256: str) -> dict[str, Any]:
    _require(fixture.get("format") == FIXTURE_FORMAT, "fixture format mismatch")
    contract_settings = _validate_sampling_contract(fixture)
    _require(probe.get("format") == PROBE_FORMAT, "probe output format mismatch")
    _require(_sha(probe.get("fixture_sha256"), "probe.fixture_sha256") == fixture_sha256,
             "probe output is bound to a different fixture byte sequence")
    _validate_probe_lineage(probe, fixture)
    raw_cases = fixture.get("cases")
    _require(isinstance(raw_cases, list) and bool(raw_cases), "fixture.cases: expected nonempty array")
    input_cases: dict[str, dict[str, Any]] = {}
    case_settings: dict[str, tuple[float, int, float]] = {}
    for index, raw_case in enumerate(raw_cases):
        case = _validate_case_input(raw_case, f"fixture.cases[{index}]")
        _require(case["name"] not in input_cases, f"fixture.cases[{index}].name: duplicate case")
        input_cases[case["name"]] = case
        case_settings[case["name"]] = _validate_case_sampling_settings(
            case, contract_settings, f"fixture.cases[{index}]"
        )
    _validate_source_trace_bindings(fixture, input_cases)
    _validate_fixture_pairs(fixture, input_cases, case_settings)

    raw_output_cases = probe.get("cases")
    _require(isinstance(raw_output_cases, list), "probe.cases: expected array")
    outputs: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_output_cases):
        where = f"probe.cases[{index}]"
        _require(isinstance(raw, dict), f"{where}: expected object")
        name = raw.get("name")
        _require(isinstance(name, str) and name in input_cases, f"{where}.name: unknown case")
        _require(name not in outputs, f"{where}.name: duplicate case")
        case = input_cases[name]
        for key in ("role", "source_arm", "step", "case_kind"):
            _require(raw.get(key) == case[key], f"{where}.{key}: metadata mismatch")
        _require(_f32(raw.get("temperature"), f"{where}.temperature") == case["temperature"],
                 f"{where}.temperature: mismatch")
        _require(_strict_int(raw.get("top_k"), f"{where}.top_k", minimum=1) == case["top_k"],
                 f"{where}.top_k: mismatch")
        _require(_f32(raw.get("top_p"), f"{where}.top_p") == case["top_p"],
                 f"{where}.top_p: mismatch")
        _require(_number_list(raw.get("draws"), f"{where}.draws", len(case["draws"])) == case["draws"],
                 f"{where}.draws: mismatch")
        outputs[name] = {
            "native": _validate_algorithm_record(raw, case, "native", where),
            "mlx": _validate_algorithm_record(raw, case, "mlx", where),
        }
    missing = sorted(set(input_cases) - set(outputs))
    extra = sorted(set(outputs) - set(input_cases))
    _require(not missing and not extra, f"probe cases do not match fixture: missing={missing[:4]} extra={extra[:4]}")
    queue_summary = _validate_queue_ledger(probe, list(input_cases.values()))
    random_tape_context = _validate_random_tape_lineage(fixture, input_cases)

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    strict_all_case_failures: list[str] = []
    real_trace_case_failures: list[str] = []
    tie_cases = 0
    real_trace_case_count = 0
    synthetic_case_count = 0
    strict_id_mismatch_count = 0
    for name, case in input_cases.items():
        pair = outputs[name]
        native = pair["native"]
        mlx = pair["mlx"]
        is_real_trace = case["case_kind"] == "trace"
        real_trace_case_count += int(is_real_trace)
        synthetic_case_count += int(not is_real_trace)
        tie = _tie_info(case)
        if tie["top_k_boundary_tie"]:
            tie_cases += 1
        topk_equal = native["topk_ids"] == mlx["topk_ids"]
        topk_set_equal = set(native["topk_ids"]) == set(mlx["topk_ids"])
        topk_tie_ambiguous = (not topk_set_equal and tie["top_k_boundary_tie"])
        if not topk_set_equal and not topk_tie_ambiguous:
            failures.append(f"{name}: native/MLX top-k sets differ without a boundary tie")
            strict_all_case_failures.append(f"{name}: top-k sets differ without a boundary tie")
            if is_real_trace:
                real_trace_case_failures.append(f"{name}: top-k sets differ without a boundary tie")

        native_kept = {token for token, keep in zip(native["topk_ids"], native["kept_mask"]) if keep}
        mlx_kept = {token for token, keep in zip(mlx["topk_ids"], mlx["kept_mask"]) if keep}
        kept_diff = native_kept.symmetric_difference(mlx_kept)
        kept_tie_ambiguous = bool(kept_diff) and len({case["logits"][token] for token in kept_diff}) == 1
        if kept_diff and not kept_tie_ambiguous:
            failures.append(f"{name}: retained candidate IDs differ outside an equal-logit tie group")
            strict_all_case_failures.append(f"{name}: retained candidate IDs differ outside an equal-logit tie group")
            if is_real_trace:
                real_trace_case_failures.append(f"{name}: retained candidate IDs differ outside an equal-logit tie group")

        full_prob_stats = _pair_probability_stats(native, mlx, "full_probabilities")
        full_cdf_stats = _pair_probability_stats(native, mlx, "full_cdf")
        conditional_stats = _pair_probability_stats(
            native, mlx, "retained_conditional_probabilities"
        )
        conditional_cdf_stats = _pair_probability_stats(native, mlx, "retained_conditional_cdf")
        for label, stats in (("full probabilities", full_prob_stats), ("full CDF", full_cdf_stats),
                             ("retained conditional probabilities", conditional_stats),
                             ("retained conditional CDF", conditional_cdf_stats)):
            if not stats["within_tolerance"]:
                failures.append(f"{name}: native/MLX {label} exceed fixed tolerance")
                strict_all_case_failures.append(f"{name}: native/MLX {label} exceed fixed tolerance")
                if is_real_trace:
                    real_trace_case_failures.append(f"{name}: native/MLX {label} exceed fixed tolerance")

        cdf_arithmetic_failures: list[dict[str, Any]] = []
        for sampler_name, sampler_record in (("native", native), ("mlx", mlx)):
            checks = sampler_record["input_math_checks"].get("cdf_invariants")
            _require(isinstance(checks, dict), f"{name}: missing {sampler_name} CDF invariant checks")
            for field, check in checks.items():
                _require(isinstance(check, dict) and isinstance(check.get("violations"), list),
                         f"{name}: malformed {sampler_name} {field} invariant checks")
                for violation in check["violations"]:
                    _require(isinstance(violation, dict),
                             f"{name}: malformed {sampler_name} {field} invariant violation")
                    detail = {
                        "sampler": sampler_name,
                        "field": field,
                        **violation,
                    }
                    cdf_arithmetic_failures.append(detail)
                    strict_failure = (
                        f"{name}: {sampler_name} {field} arithmetic invariant "
                        f"{violation.get('kind')} at rank {violation.get('index')}"
                    )
                    strict_all_case_failures.append(strict_failure)
                    if is_real_trace:
                        failures.append(strict_failure)
                        real_trace_case_failures.append(strict_failure)

        gumbel_equal = native["shared_gumbel_ids"] == mlx["shared_gumbel_ids"]
        exact_ids = {
            "topk_order_equal": topk_equal,
            "topk_set_equal": topk_set_equal,
            "kept_mask_equal_by_rank": native["kept_mask"] == mlx["kept_mask"],
            "kept_candidate_order_equal": [token for token, keep in zip(native["topk_ids"], native["kept_mask"])
                                           if keep] ==
                                           [token for token, keep in zip(mlx["topk_ids"], mlx["kept_mask"])
                                            if keep],
            "inverse_cdf_sample_ids_equal": native["sample_ids"] == mlx["sample_ids"],
            "shared_gumbel_ids_equal": gumbel_equal,
        }
        mismatched_id_fields = [field for field, equal in exact_ids.items() if not equal]
        if mismatched_id_fields:
            strict_id_mismatch_count += 1
            strict_all_case_failures.append(
                f"{name}: exact sampler IDs differ in {', '.join(mismatched_id_fields)}"
            )
            if is_real_trace:
                real_trace_case_failures.append(
                    f"{name}: exact sampler IDs differ in {', '.join(mismatched_id_fields)}"
                )
        results.append({
            "name": name,
            "role": case["role"],
            "source_arm": case["source_arm"],
            "step": case["step"],
            "case_kind": case["case_kind"],
            "tie_analysis": tie,
            "native_topk_ids": native["topk_ids"],
            "mlx_topk_ids": mlx["topk_ids"],
            "topk_order_equal": topk_equal,
            "topk_set_equal": topk_set_equal,
            "topk_identity_ambiguity_due_to_exact_boundary_tie": topk_tie_ambiguous,
            "native_kept_ids": [token for token, keep in zip(native["topk_ids"], native["kept_mask"]) if keep],
            "mlx_kept_ids": [token for token, keep in zip(mlx["topk_ids"], mlx["kept_mask"]) if keep],
            "kept_set_equal": native_kept == mlx_kept,
            "kept_identity_ambiguity_due_to_equal_logit_tie": kept_tie_ambiguous,
            "native_sample_ids": native["sample_ids"],
            "mlx_sample_ids": mlx["sample_ids"],
            "exact_id_comparison": exact_ids,
            "exact_id_mismatches": mismatched_id_fields,
            "native_shared_gumbel_ids": native["shared_gumbel_ids"],
            "mlx_shared_gumbel_ids": mlx["shared_gumbel_ids"],
            "shared_gumbel_ids_equal": gumbel_equal,
            "native_input_math_checks": native["input_math_checks"],
            "mlx_input_math_checks": mlx["input_math_checks"],
            "cdf_arithmetic_invariant_failures": cdf_arithmetic_failures,
            "full_probability_comparison": full_prob_stats,
            "full_cdf_comparison": full_cdf_stats,
            "retained_conditional_probability_comparison": conditional_stats,
            "retained_conditional_cdf_comparison": conditional_cdf_stats,
        })

    input_logit_pairs = fixture.get("input_logit_comparisons")
    _require(isinstance(input_logit_pairs, list), "fixture.input_logit_comparisons: expected array")
    for row in input_logit_pairs:
        if not isinstance(row, dict) or not isinstance(row.get("comparison"), dict):
            raise InvalidBundle("fixture has malformed input logit comparison")
        if not row["comparison"].get("within_tolerance", False):
            failure = f"{row.get('name')}: native/upstream input logits exceed fixed tolerance"
            failures.append(failure)
            strict_all_case_failures.append(failure)
            real_trace_case_failures.append(failure)

    paired_probabilities: list[dict[str, Any]] = []
    for pair_index, pair_raw in enumerate(fixture.get("pairs", [])):
        where = f"fixture.pairs[{pair_index}]"
        _require(isinstance(pair_raw, dict), f"{where}: expected object")
        native_name = pair_raw.get("native_case")
        upstream_name = pair_raw.get("upstream_case")
        _require(native_name in outputs and upstream_name in outputs, f"{where}: references missing case")
        per_sampler: dict[str, Any] = {}
        for sampler in ("native", "mlx"):
            native_probs = outputs[native_name][sampler]["full_probabilities"]
            upstream_probs = outputs[upstream_name][sampler]["full_probabilities"]
            stats = _array_stats(native_probs, upstream_probs)
            per_sampler[sampler] = stats
            if not stats["within_tolerance"]:
                failure = f"{pair_raw.get('name')}: paired {sampler} probabilities exceed fixed tolerance"
                failures.append(failure)
                strict_all_case_failures.append(failure)
                real_trace_case_failures.append(failure)
        paired_probabilities.append({
            "name": pair_raw.get("name"),
            "role": pair_raw.get("role"),
            "step": pair_raw.get("step"),
            "native_case": native_name,
            "upstream_case": upstream_name,
            "rank_aligned_topk_probability_comparison": per_sampler,
        })

    ras_replay = _run_native_ras_replay(random_tape_context, input_cases, outputs)
    ras_failures = [f"native RAS: {failure}" for failure in ras_replay["failures"]]

    supported_status = ("not_applicable" if real_trace_case_count == 0 else
                        "fail" if real_trace_case_failures else "pass")
    strict_status = "fail" if strict_all_case_failures else "pass"
    report_failures = [*real_trace_case_failures, *ras_failures]
    overall_status = ("not_applicable" if supported_status == "not_applicable" and
                      ras_replay["status"] == "not_applicable" else
                      "fail" if report_failures else "pass")
    return {
        "format": REPORT_FORMAT,
        "status": overall_status,
        "status_scope": "supported_real_trace_sampler_cases_plus_optional_native_ras_replay",
        "supported_real_trace": {
            "status": supported_status,
            "case_count": real_trace_case_count,
            "failures": real_trace_case_failures,
        },
        "native_ras_replay": ras_replay,
        "strict_all_cases_parity": {
            "status": strict_status,
            "exact_id_parity": strict_id_mismatch_count == 0,
            "failure_count": len(strict_all_case_failures),
            "failures": strict_all_case_failures,
        },
        "fixture_sha256": fixture_sha256,
        "tolerance": {"atol": ATOL, "rtol": RTOL, "reference_direction": "upstream for input logits; second arm for pair summaries"},
        "scope": {
            "source": "same logits vectors from retained native and pinned MLX F32 rowwise traces",
            "randomness": "fixed common scalar draws and independent common Gumbel uniforms; not seeded MLX RNG parity",
            "probabilities": "full-vocabulary normalization is consumed by the GPU probe; serialized probability/CDF vectors cover the sorted top-k candidates",
            "ties": "native lower-token-ID tie rule is explicit; MLX argpartition boundary-tie token identity is reported as ambiguous",
        },
        "input_logit_comparisons": input_logit_pairs,
        "sampler_cases": results,
        "paired_probability_comparisons": paired_probabilities,
        "summary": {
            "case_count": len(results),
            "real_trace_case_count": real_trace_case_count,
            "synthetic_case_count": synthetic_case_count,
            "cases_with_exact_id_mismatches": strict_id_mismatch_count,
            "strict_all_case_failure_count": len(strict_all_case_failures),
            "supported_real_trace_failure_count": len(real_trace_case_failures),
            "cdf_arithmetic_invariant_failure_count": sum(
                len(row["cdf_arithmetic_invariant_failures"]) for row in results
            ),
            "synthetic_strict_failure_count": sum(
                failure.split(":", 1)[0] in input_cases and
                input_cases[failure.split(":", 1)[0]]["case_kind"] != "trace"
                for failure in strict_all_case_failures
            ),
            "paired_source_vectors": len(paired_probabilities),
            "exact_topk_order_matches": sum(row["topk_order_equal"] for row in results),
            "exact_topk_set_matches": sum(row["topk_set_equal"] for row in results),
            "boundary_tie_cases": tie_cases,
            "shared_gumbel_id_matches": sum(row["shared_gumbel_ids_equal"] for row in results),
            "failure_count": len(report_failures),
            "native_ras_check_count": ras_replay["check_count"],
            "native_ras_failure_count": len(ras_replay["failures"]),
        },
        "queue_validation": queue_summary,
        "failures": report_failures,
    }


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    with path.open("xb") as stream:
        stream.write(payload)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _prepare(args: argparse.Namespace) -> int:
    steps = args.step if args.step else None
    fixture = build_fixture(args.native_trace, args.upstream_trace, steps=steps,
                            all_common_steps=args.all_common_steps, mlx_root=args.mlx_root,
                            random_tape_path=args.random_tape)
    payload = _json_bytes(fixture)
    _write_new(args.output, payload)
    logit_failures = [row["name"] for row in fixture["input_logit_comparisons"]
                      if not row["comparison"]["within_tolerance"]]
    print(json.dumps({
        "status": "fail" if logit_failures else "pass",
        "fixture": str(args.output.resolve()),
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "case_count": len(fixture["cases"]),
        "paired_case_count": len(fixture["pairs"]),
        "capture_steps": fixture["generation"]["capture_steps"],
        "native_random_tape": fixture["provenance"]["native_random_tape"],
        "ras_check_count": len(fixture["generation"]["ras_checks"]),
        "input_logit_failures": logit_failures,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
    }, sort_keys=True))
    return 1 if logit_failures else 0


def _compare(args: argparse.Namespace) -> int:
    fixture_raw, fixture_envelope = _read_json_once(args.fixture)
    probe_raw, probe_envelope = _read_json_once(args.probe_output)
    report = compare_probe(fixture_envelope["value"], probe_envelope["value"],
                           fixture_sha256=fixture_envelope["sha256"])
    report["artifacts"] = {
        "fixture": {"path": str(args.fixture.resolve()), "sha256": fixture_envelope["sha256"]},
        "probe_output": {"path": str(args.probe_output.resolve()), "sha256": probe_envelope["sha256"]},
    }
    payload = _json_bytes(report)
    _write_new(args.output, payload)
    print(json.dumps({
        "status": report["status"],
        "report": str(args.output.resolve()),
        "report_sha256": hashlib.sha256(payload).hexdigest(),
        "summary": report["summary"],
        "failures": report["failures"][:12],
    }, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="build input cases from retained native/upstream traces")
    prepare.add_argument("--native-trace", type=Path, required=True)
    prepare.add_argument("--upstream-trace", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--mlx-root", type=Path, default=PINNED_MLX_ROOT)
    prepare.add_argument("--random-tape", type=Path,
                         help="optional seed-1 native sampler tape for actual-U and RAS checks")
    prepare.add_argument("--step", type=int, action="append", help="captured prediction step; repeatable")
    prepare.add_argument("--all-common-steps", action="store_true",
                         help="use every captured common step, including dense tail windows")
    compare = commands.add_parser("compare", help="validate/compare one GPU common-sampler result bundle")
    compare.add_argument("--fixture", type=Path, required=True)
    compare.add_argument("--probe-output", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.all_common_steps and args.step:
                raise InvalidBundle("--step and --all-common-steps are mutually exclusive")
            return _prepare(args)
        return _compare(args)
    except (InvalidBundle, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
