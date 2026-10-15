#!/usr/bin/env python3
"""Strict CPU comparison for native and MLX common-prefix generation traces."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


TRACE_FORMAT = "drama3-generation-common-prefix-v1"
REPORT_FORMAT = "drama3-generation-replay-comparison-v1"
ATOL = 6e-5
RTOL = 3e-4
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 640)
TAIL_WINDOW_FRAMES = 128
STAGE_NAME = re.compile(r"generation\.step([0-9]+)\.(backbone_hidden|main_logits|tower_hidden|tower_logits\.([0-8]))\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHARED_MODEL_CONFIG = ("hidden_size", "main_vocab_size", "tower_dim", "tower_vocab_size", "num_codebooks")
MODEL_FILES = ("config.json", "model.safetensors.index.json")


class InvalidTrace(ValueError):
    """An input is malformed or does not describe the same replay."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidTrace(message)


def _strict_int(value: Any, where: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{where}: expected integer")
    return value


def _strict_int_list(value: Any, where: str, *, nonempty: bool = True) -> list[int]:
    _require(isinstance(value, list), f"{where}: expected array")
    _require(not nonempty or bool(value), f"{where}: must not be empty")
    return [_strict_int(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _float32(value: Any, where: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{where}: expected numeric F32 value")
    number = float(value)
    _require(math.isfinite(number), f"{where}: non-finite F32 value")
    try:
        packed = struct.pack("<f", number)
    except (OverflowError, struct.error) as error:
        raise InvalidTrace(f"{where}: value is outside F32 range") from error
    roundtrip = struct.unpack("<f", packed)[0]
    # Crystal's JSON encoder may print a Float32 using its shortest decimal
    # form (for example, 0.1). Normalize both sides back to the declared wire
    # dtype before comparing.
    return roundtrip


def _expected_steps(frame_count: int) -> list[int]:
    tail_start = max(0, frame_count - TAIL_WINDOW_FRAMES)
    return sorted({step for step in CHECKPOINTS if step <= frame_count}
                  | set(range(tail_start, frame_count + 1)))


def _validate_capture_policy(value: Any, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where}.capture_policy: expected object")
    _require(set(value) == {"checkpoints", "tail_window_frames", "tail_inclusive"},
             f"{where}.capture_policy: unexpected fields")
    checkpoints = _strict_int_list(value.get("checkpoints"), f"{where}.capture_policy.checkpoints")
    _require(checkpoints == list(CHECKPOINTS), f"{where}.capture_policy.checkpoints: mismatch")
    tail_window = _strict_int(value.get("tail_window_frames"),
                              f"{where}.capture_policy.tail_window_frames")
    _require(tail_window == TAIL_WINDOW_FRAMES,
             f"{where}.capture_policy.tail_window_frames: expected {TAIL_WINDOW_FRAMES}")
    tail_inclusive = value.get("tail_inclusive")
    _require(tail_inclusive is True, f"{where}.capture_policy.tail_inclusive: expected true")
    return {
        "checkpoints": checkpoints,
        "tail_window_frames": tail_window,
        "tail_inclusive": tail_inclusive,
    }


def _expected_names(steps: list[int], frame_count: int) -> set[str]:
    names: set[str] = set()
    for step in steps:
        names.add(f"generation.step{step}.backbone_hidden")
        names.add(f"generation.step{step}.main_logits")
        if step < frame_count:
            names.add(f"generation.step{step}.tower_hidden")
            names.update(f"generation.step{step}.tower_logits.{head}" for head in range(9))
    return names


def _model_metadata(bundle: dict[str, Any], where: str) -> tuple[dict[str, int], dict[str, str]]:
    raw_config = bundle.get("model_config")
    _require(isinstance(raw_config, dict), f"{where}.model_config: expected object")
    model_config: dict[str, int] = {}
    for name in SHARED_MODEL_CONFIG:
        dimension = _strict_int(raw_config.get(name), f"{where}.model_config.{name}")
        _require(dimension > 0, f"{where}.model_config.{name}: must be positive")
        model_config[name] = dimension
    _require(model_config["num_codebooks"] == 10, f"{where}.model_config.num_codebooks: expected ten")

    raw_files = bundle.get("model_files")
    _require(isinstance(raw_files, dict), f"{where}.model_files: expected object")
    model_files: dict[str, str] = {}
    for name in MODEL_FILES:
        digest = raw_files.get(name)
        _require(isinstance(digest, str) and SHA256.fullmatch(digest),
                 f"{where}.model_files.{name}: expected lowercase SHA-256")
        model_files[name] = digest
    return model_config, model_files


def _expected_stage_length(name: str, model_config: dict[str, int]) -> int:
    if name.endswith(".backbone_hidden"):
        return model_config["hidden_size"]
    if name.endswith(".main_logits"):
        return model_config["main_vocab_size"]
    if name.endswith(".tower_hidden"):
        return model_config["tower_dim"]
    if ".tower_logits." in name:
        return model_config["tower_vocab_size"]
    raise AssertionError(f"unhandled stage name: {name}")


def _validate_bundle(bundle: Any, where: str) -> dict[str, Any]:
    _require(isinstance(bundle, dict), f"{where}: expected JSON object")
    _require(bundle.get("format") == TRACE_FORMAT, f"{where}: unexpected trace format")
    source_report = bundle.get("source_report")
    _require(isinstance(source_report, dict), f"{where}.source_report: expected object")
    _require(isinstance(source_report.get("path"), str) and bool(source_report["path"]),
             f"{where}.source_report.path: expected non-empty string")
    _require(isinstance(source_report.get("sha256"), str) and SHA256.fullmatch(source_report["sha256"]),
             f"{where}.source_report.sha256: expected lowercase SHA-256")

    prompt_ids = _strict_int_list(bundle.get("prompt_ids"), f"{where}.prompt_ids")
    prompt_rows = _strict_int(bundle.get("prompt_rows"), f"{where}.prompt_rows")
    _require(prompt_rows == len(prompt_ids), f"{where}: prompt_rows does not match prompt_ids")

    raw_frames = bundle.get("frames")
    _require(isinstance(raw_frames, list) and bool(raw_frames), f"{where}.frames: expected non-empty array")
    frames: list[list[int]] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        frame = _strict_int_list(raw_frame, f"{where}.frames[{frame_index}]")
        _require(len(frame) == 11, f"{where}.frames[{frame_index}]: expected 11 token IDs")
        frames.append(frame)
    frame_count = _strict_int(bundle.get("frame_count"), f"{where}.frame_count")
    _require(frame_count == len(frames), f"{where}: frame_count does not match frames")

    source_generation = bundle.get("source_generation")
    _require(isinstance(source_generation, dict), f"{where}.source_generation: expected object")
    truncated = source_generation.get("truncated")
    stopped_on_eos = source_generation.get("stopped_on_eos")
    _require(isinstance(truncated, bool) and isinstance(stopped_on_eos, bool),
             f"{where}.source_generation: termination flags must be booleans")
    _require(truncated != stopped_on_eos, f"{where}.source_generation: termination flags disagree")
    expected_terminal = {
        "index": frame_count,
        "measured_in_generation": not truncated,
        "interpretation": (
            "counterfactual_after_truncated_prefix" if truncated
            else "actual_terminal_prediction_after_reported_frames"
        ),
    }
    terminal = bundle.get("terminal_step")
    _require(isinstance(terminal, dict), f"{where}.terminal_step: expected object")
    _require(terminal == expected_terminal, f"{where}.terminal_step: inconsistent terminal-step metadata")

    capture_policy = _validate_capture_policy(bundle.get("capture_policy"), where)
    capture_steps = _strict_int_list(bundle.get("capture_steps"), f"{where}.capture_steps")
    _require(capture_steps == _expected_steps(frame_count),
             f"{where}.capture_steps: missing, extra, or unordered checkpoint")

    model_config, model_files = _model_metadata(bundle, where)
    execution = bundle.get("execution")
    _require(isinstance(execution, dict), f"{where}.execution: expected object")

    traces = bundle.get("traces")
    _require(isinstance(traces, list), f"{where}.traces: expected array")
    indexed: dict[str, dict[str, Any]] = {}
    for trace_index, trace in enumerate(traces):
        label = f"{where}.traces[{trace_index}]"
        _require(isinstance(trace, dict), f"{label}: expected object")
        name = trace.get("name")
        _require(isinstance(name, str) and STAGE_NAME.fullmatch(name), f"{label}.name: invalid stage name")
        _require(name not in indexed, f"{label}.name: duplicate stage")
        match = STAGE_NAME.fullmatch(name)
        assert match is not None
        step = int(match.group(1))
        _require(step in capture_steps, f"{label}.name: stage for uncaptured step")
        _require(trace.get("dtype") == "float32", f"{label}.dtype: expected float32")
        shape = _strict_int_list(trace.get("shape"), f"{label}.shape")
        _require(len(shape) == 1 and shape[0] > 0, f"{label}.shape: expected a positive flat vector")
        expected_length = _expected_stage_length(name, model_config)
        _require(shape[0] == expected_length,
                 f"{label}.shape: expected [{expected_length}], found {shape}")
        values = trace.get("values")
        _require(isinstance(values, list), f"{label}.values: expected array")
        _require(shape[0] == len(values), f"{label}: shape/value count mismatch")
        normalized = [_float32(value, f"{label}.values[{i}]") for i, value in enumerate(values)]
        indexed[name] = {"name": name, "step": step, "shape": shape, "values": normalized}

    expected_names = _expected_names(capture_steps, frame_count)
    actual_names = set(indexed)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    _require(not missing and not unexpected,
             f"{where}.traces: missing={missing[:4]} unexpected={unexpected[:4]}")

    return {
        "source_report": source_report,
        "prompt_ids": prompt_ids,
        "prompt_rows": prompt_rows,
        "frames": frames,
        "frame_count": frame_count,
        "capture_policy": capture_policy,
        "capture_steps": capture_steps,
        "terminal_step": expected_terminal,
        "source_generation": source_generation,
        "model_config": model_config,
        "model_files": model_files,
        "execution": execution,
        "provenance": {
            key: bundle[key] for key in ("model", "upstream", "source_hashes", "model_config", "model_files")
            if key in bundle
        },
        "traces": indexed,
    }


def _unravel(index: int, shape: list[int]) -> list[int]:
    coordinate = [0] * len(shape)
    remainder = index
    for dimension in range(len(shape) - 1, -1, -1):
        coordinate[dimension] = remainder % shape[dimension]
        remainder //= shape[dimension]
    return coordinate


def _metrics(records: list[tuple[str, list[int], list[float], list[float]]]) -> dict[str, Any]:
    count = sum(len(actual) for _, _, actual, _ in records)
    _require(count > 0, "internal comparator error: empty metric input")
    sum_sq = 0.0
    ref_sq = 0.0
    max_abs = -1.0
    worst: tuple[str, list[int], int, float, float, float] | None = None
    failures = 0
    for name, shape, actual_values, reference_values in records:
        for index, (actual, reference) in enumerate(zip(actual_values, reference_values, strict=True)):
            error = abs(actual - reference)
            allowed = ATOL + RTOL * abs(reference)
            sum_sq += (actual - reference) ** 2
            ref_sq += reference * reference
            if error > allowed:
                failures += 1
            if error > max_abs:
                max_abs = error
                worst = (name, _unravel(index, shape), index, reference, actual, allowed)
    assert worst is not None
    relative_l2 = math.sqrt(sum_sq) / max(math.sqrt(ref_sq), 1e-30)
    stage_name, coordinate, flat_index, reference, actual, allowed = worst
    return {
        "value_count": count,
        "out_of_tolerance_count": failures,
        "max_abs": max_abs,
        "rms": math.sqrt(sum_sq / count),
        "relative_l2": relative_l2,
        "relative_l2_denominator_floor": 1e-30,
        "worst": {
            "stage": stage_name,
            "coordinate": coordinate,
            "flat_index": flat_index,
            "reference": reference,
            "actual": actual,
            "allowed_error": allowed,
        },
    }


def _logit_diagnostic(reference: list[float], actual: list[float]) -> dict[str, Any]:
    size = min(30, len(reference))
    ref_order = sorted(range(len(reference)), key=lambda index: (-reference[index], index))
    actual_order = sorted(range(len(actual)), key=lambda index: (-actual[index], index))
    ref_top = ref_order[:size]
    actual_top = actual_order[:size]
    overlap = len(set(ref_top) & set(actual_top))
    return {
        "top1_equal": ref_order[0] == actual_order[0],
        "top30_overlap_count": overlap,
        "top30_size": size,
        "top30_overlap_fraction": overlap / size,
        "diagnostic_only": True,
    }


def compare_bundles(reference_bundle: Any, actual_bundle: Any) -> dict[str, Any]:
    reference = _validate_bundle(reference_bundle, "reference")
    actual = _validate_bundle(actual_bundle, "actual")
    for key in ("source_report", "prompt_ids", "prompt_rows", "frames", "frame_count",
                "capture_policy", "capture_steps", "terminal_step", "source_generation",
                "model_config", "model_files"):
        if key == "source_report":
            same = reference[key]["sha256"] == actual[key]["sha256"]
        else:
            same = reference[key] == actual[key]
        _require(same, f"reference/actual identity mismatch: {key}")

    frame_count = reference["frame_count"]
    terminal_index = reference["terminal_step"]["index"]
    counterfactual = not reference["terminal_step"]["measured_in_generation"]
    per_step: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    included_records: list[tuple[str, list[int], list[float], list[float]]] = []
    counterfactual_records: list[tuple[str, list[int], list[float], list[float]]] = []

    for step in reference["capture_steps"]:
        names = sorted(name for name, record in reference["traces"].items() if record["step"] == step)
        included = not (counterfactual and step == terminal_index)
        step_records: list[tuple[str, list[int], list[float], list[float]]] = []
        step_stages: list[dict[str, Any]] = []
        for name in names:
            expected_record = reference["traces"][name]
            actual_record = actual["traces"][name]
            _require(expected_record["shape"] == actual_record["shape"],
                     f"reference/actual shape mismatch for {name}")
            record = (name, expected_record["shape"], actual_record["values"], expected_record["values"])
            step_records.append(record)
            if included:
                included_records.append(record)
            else:
                counterfactual_records.append(record)
            metric = _metrics([record])
            stage_result = {
                "name": name,
                "step": step,
                "included_in_parity": included,
                **metric,
            }
            if name.endswith(".main_logits") or ".tower_logits." in name:
                stage_result["logit_ranking"] = _logit_diagnostic(
                    expected_record["values"], actual_record["values"]
                )
            step_stages.append(stage_result)
            stage_reports.append(stage_result)
        per_step.append({
            "step": step,
            "is_terminal": step == terminal_index,
            "measured_in_generation": included,
            "included_in_parity": included,
            "metrics": _metrics(step_records),
            "stages": step_stages,
        })

    parity = _metrics(included_records)
    counterfactual_metrics = _metrics(counterfactual_records) if counterfactual_records else None
    status = "pass" if parity["out_of_tolerance_count"] == 0 else "mismatch"
    return {
        "format": REPORT_FORMAT,
        "status": status,
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "comparison_scope": {
            "kind": "numeric_common_prefix_trajectory",
            "note": (
                "Execution modes may differ and are reported for interpretation. "
                "This comparison does not attest model shard bytes or establish actual sampling/EOS parity."
            ),
        },
        "reference_execution": reference["execution"],
        "actual_execution": actual["execution"],
        "reference_provenance": reference["provenance"],
        "actual_provenance": actual["provenance"],
        "identity": {
            "source_report_sha256": reference["source_report"]["sha256"],
            "prompt_rows": reference["prompt_rows"],
            "prompt_ids": reference["prompt_ids"],
            "frame_count": frame_count,
            "capture_policy": reference["capture_policy"],
            "capture_steps": reference["capture_steps"],
            "terminal_step": reference["terminal_step"],
        },
        "parity_scope": {
            "actual_generation_steps": [step for step in reference["capture_steps"]
                                        if not (counterfactual and step == terminal_index)],
            "counterfactual_terminal_step": terminal_index if counterfactual else None,
            "counterfactual_terminal_excluded_from_pass_fail": counterfactual,
            "note": (
                "The final prediction after a truncated prefix is reported diagnostically, "
                "but excluded from actual-generation parity."
                if counterfactual else
                "The terminal prediction follows the reported EOS-terminated generation."
            ),
        },
        "parity_metrics": parity,
        "counterfactual_terminal_metrics": counterfactual_metrics,
        "steps": per_step,
        "stages": stage_reports,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise InvalidTrace(f"{label}: duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise InvalidTrace(f"{label}: invalid JSON numeric constant {token}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidTrace(f"{label}: cannot read JSON: {error}") from error
    _require(isinstance(value, dict), f"{label}: expected JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.reference.resolve(), args.actual.resolve()}:
        print(json.dumps({"format": REPORT_FORMAT, "status": "invalid",
                          "error": "output path must differ from both input paths"}), file=sys.stderr)
        return 2
    if args.output.exists():
        print(json.dumps({"format": REPORT_FORMAT, "status": "invalid",
                          "error": f"output already exists: {args.output}"}), file=sys.stderr)
        return 2
    try:
        report = compare_bundles(_read_json(args.reference, "reference"),
                                 _read_json(args.actual, "actual"))
    except InvalidTrace as error:
        print(json.dumps({"format": REPORT_FORMAT, "status": "invalid", "error": str(error)}),
              file=sys.stderr)
        return 2
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as output_file:
            output_file.write(serialized)
    except OSError as error:
        print(json.dumps({"format": REPORT_FORMAT, "status": "invalid", "error": str(error)}),
              file=sys.stderr)
        return 2
    print(json.dumps({"format": REPORT_FORMAT, "status": report["status"],
                      "output": str(args.output),
                      "out_of_tolerance_count": report["parity_metrics"]["out_of_tolerance_count"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
