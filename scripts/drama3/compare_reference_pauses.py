#!/usr/bin/env python3
"""GPU diagnostics for completed native/MLX reference-voice WAV pairs.

This tool compares generated token sequences and measures quiet regions in the
already completed WAV files. It does not synthesize audio, run ASR, or make a
claim about perceived voice quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
import wave
from pathlib import Path
from typing import Any


REPORT_FORMAT = "drama3-reference-pause-diagnostics-v1"
MLX_FORMAT = "drama3-mlx-control-v1"
NATIVE_FORMAT = "metal_resident"
FIXTURE_FORMAT = "drama3-reference-voice-manifest-v1"
ENCODER_FORMAT = "drama3-native-reference-comparison-v1"
WINDOW_SAMPLES = 22_050
QUIET_PEAK_PCM16 = 64
MIN_PAUSE_WINDOWS = 2
FRAME_LIMIT = 512
DRAW_STRIDE = 11
TAPE_PREFIX_COUNT = FRAME_LIMIT * DRAW_STRIDE
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
LANGUAGES = ("ru", "en")
REPORT_NAMES = (
    "native-ru-reference",
    "native-ru-no-reference",
    "mlx-ru-reference",
    "mlx-ru-no-reference",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def report_names(language: str = "ru") -> tuple[str, str, str, str]:
    """Return report names for the explicitly selected filename namespace."""
    _require(language in LANGUAGES, "language: expected ru or en")
    return (
        f"native-{language}-reference",
        f"native-{language}-no-reference",
        f"mlx-{language}-reference",
        f"mlx-{language}-no-reference",
    )


def report_paths(run_dir: Path, language: str = "ru") -> dict[str, Path]:
    """Return the required report file paths for the selected language namespace."""
    return {
        name: run_dir / (f"{name}.wav.json" if name.startswith("native-") else f"{name}.json")
        for name in report_names(language)
    }


def _obj(value: Any, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where}: expected JSON object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    _require(isinstance(value, list), f"{where}: expected array")
    return value


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{where}: expected integer")
    if minimum is not None:
        _require(value >= minimum, f"{where}: expected integer >= {minimum}")
    return value


def _finite(value: Any, where: str, *, minimum: float | None = None) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{where}: expected number")
    result = float(value)
    _require(math.isfinite(result), f"{where}: expected finite number")
    if minimum is not None:
        _require(result >= minimum, f"{where}: expected number >= {minimum}")
    return result


def _sha(value: Any, where: str) -> str:
    _require(isinstance(value, str) and SHA256.fullmatch(value) is not None,
             f"{where}: expected lowercase SHA-256")
    return value


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def validate_source_hash(path: Path, expected_sha256: str) -> dict[str, str]:
    """Bind the producer-prefix proof to the source file named by its report."""
    _sha(expected_sha256, "production source SHA-256")
    _require(path.is_file(), f"production source is missing: {path}")
    actual = _hash_file(path)
    _require(actual == expected_sha256, "production source SHA-256 mismatch")
    return {"path": str(path.resolve()), "sha256": actual}


def _load_json(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{where}: cannot read valid UTF-8 JSON from {path}: {error}") from error
    return _obj(value, where)


def _check_text_sha(text: Any, digest: Any, where: str) -> str:
    _require(isinstance(text, str), f"{where}.text: expected string")
    actual = _hash_bytes(text.encode("utf-8"))
    _require(_sha(digest, f"{where}.text_utf8_sha256") == actual,
             f"{where}: text UTF-8 SHA-256 mismatch")
    return actual


def _normalize_frames(value: Any, where: str) -> list[list[int]]:
    frames = _array(value, f"{where}.frames")
    _require(bool(frames), f"{where}.frames: expected non-empty generated output")
    normalized: list[list[int]] = []
    for i, raw in enumerate(frames):
        row = _array(raw, f"{where}.frames[{i}]")
        _require(len(row) == 11, f"{where}.frames[{i}]: expected main token plus ten codebooks")
        ids = [_int(item, f"{where}.frames[{i}][{j}]", minimum=0) for j, item in enumerate(row)]
        _require(all(item < 4096 for item in ids[1:]),
                 f"{where}.frames[{i}]: codebook ID outside [0,4095]")
        normalized.append(ids)
    return normalized


def _validate_generation_termination(
    *, frames: int, max_frames: Any, stopped_on_eos: Any, truncated: Any,
    where: str,
) -> None:
    _require(_int(max_frames, f"{where}.max_frames") == FRAME_LIMIT,
             f"{where}.max_frames: expected {FRAME_LIMIT}")
    _require(isinstance(stopped_on_eos, bool), f"{where}.stopped_on_eos: expected boolean")
    _require(isinstance(truncated, bool), f"{where}.truncated: expected boolean")
    _require(stopped_on_eos != truncated, f"{where}: EOS/truncation flags must be mutually exclusive")
    if truncated:
        _require(frames == FRAME_LIMIT, f"{where}: truncated output must contain exactly {FRAME_LIMIT} frames")
    else:
        _require(0 < frames < FRAME_LIMIT, f"{where}: EOS output must be shorter than the frame cap")


def _native_report(report: Any, where: str) -> dict[str, Any]:
    value = _obj(report, where)
    _require(value.get("backend") == NATIVE_FORMAT, f"{where}.backend: expected {NATIVE_FORMAT}")
    _require(isinstance(value.get("text"), str), f"{where}.text: expected string")
    _require(_int(value.get("seed"), f"{where}.seed") == 1, f"{where}.seed: expected 1")
    _require(_finite(value.get("temperature"), f"{where}.temperature") == 0.8,
             f"{where}.temperature: expected 0.8")
    _require(_int(value.get("top_k"), f"{where}.top_k") == 30, f"{where}.top_k: expected 30")
    _require(_finite(value.get("top_p"), f"{where}.top_p") == 0.9, f"{where}.top_p: expected 0.9")
    _require(_int(value.get("max_frames"), f"{where}.max_frames") == FRAME_LIMIT,
             f"{where}.max_frames: expected {FRAME_LIMIT}")
    frames = _normalize_frames(value.get("frames"), where)
    _validate_generation_termination(
        frames=len(frames), max_frames=value.get("max_frames"),
        stopped_on_eos=value.get("stopped_on_eos"), truncated=value.get("truncated"), where=where,
    )
    sample_rate = _int(value.get("sample_rate"), f"{where}.sample_rate")
    samples = _int(value.get("samples"), f"{where}.samples")
    _require(sample_rate == 44100, f"{where}.sample_rate: expected 44100")
    _require(samples == len(frames) * 2048, f"{where}.samples: expected frames * 2048")
    _require(_int(value.get("prompt_rows"), f"{where}.prompt_rows", minimum=1) > 0,
             f"{where}.prompt_rows: expected positive value")
    sampler_inputs = _obj(value.get("sampler_inputs"), f"{where}.sampler_inputs")
    _require(sampler_inputs.get("dtype") == "float32-le",
             f"{where}.sampler_inputs.dtype: expected float32-le")
    _require(_int(sampler_inputs.get("uniform_count"), f"{where}.sampler_inputs.uniform_count") == TAPE_PREFIX_COUNT,
             f"{where}.sampler_inputs.uniform_count: expected {TAPE_PREFIX_COUNT}")
    sampler_inputs_sha = _sha(sampler_inputs.get("sha256"), f"{where}.sampler_inputs.sha256")
    _require(sampler_inputs.get("provenance") == "host input tape uploaded once",
             f"{where}.sampler_inputs.provenance: expected one host tape upload")
    timings = _obj(value.get("phase_seconds"), f"{where}.phase_seconds")
    for name, seconds in timings.items():
        _finite(seconds, f"{where}.phase_seconds.{name}", minimum=0.0)
    readbacks = _obj(value.get("resident_readbacks"), f"{where}.resident_readbacks")
    for name, count in readbacks.items():
        _int(count, f"{where}.resident_readbacks.{name}", minimum=0)
    _obj(value.get("resident_graph"), f"{where}.resident_graph")
    prompt_ids = value.get("prompt_ids")
    if prompt_ids is not None:
        prompt_ids = [_int(x, f"{where}.prompt_ids[{i}]", minimum=0)
                      for i, x in enumerate(_array(prompt_ids, f"{where}.prompt_ids"))]
    reference = value.get("reference")
    if reference is not None:
        reference = _obj(reference, f"{where}.reference")
    return {
        "raw": value,
        "frames": frames,
        "prompt_ids": prompt_ids,
        "reference": reference,
        "sampler_inputs_sha256": sampler_inputs_sha,
    }


def _mlx_report(report: Any, where: str) -> dict[str, Any]:
    value = _obj(report, where)
    _require(value.get("format") == MLX_FORMAT, f"{where}.format: expected {MLX_FORMAT}")
    input_info = _obj(value.get("input"), f"{where}.input")
    text_sha = _check_text_sha(input_info.get("text"), input_info.get("text_utf8_sha256"), f"{where}.input")
    prompt_ids = [_int(x, f"{where}.input.prompt_ids[{i}]", minimum=0)
                  for i, x in enumerate(_array(input_info.get("prompt_ids"), f"{where}.input.prompt_ids"))]
    _require(_int(input_info.get("prompt_rows"), f"{where}.input.prompt_rows") == len(prompt_ids),
             f"{where}.input.prompt_rows: does not match prompt ID count")

    settings = _obj(value.get("settings"), f"{where}.settings")
    _require(_int(settings.get("seed"), f"{where}.settings.seed") == 1,
             f"{where}.settings.seed: expected 1")
    _require(_int(settings.get("max_frames"), f"{where}.settings.max_frames") == FRAME_LIMIT,
             f"{where}.settings.max_frames: expected {FRAME_LIMIT}")
    _require(_finite(settings.get("temperature"), f"{where}.settings.temperature") == 0.8,
             f"{where}.settings.temperature: expected 0.8")
    _require(_int(settings.get("top_k"), f"{where}.settings.top_k") == 30,
             f"{where}.settings.top_k: expected 30")
    _require(_finite(settings.get("top_p"), f"{where}.settings.top_p") == 0.9,
             f"{where}.settings.top_p: expected 0.9")
    _require(settings.get("sampler") == "shared_uniform_inverse_cdf_shadow",
             f"{where}.settings.sampler: expected shared-uniform inverse-CDF shadow tape")
    _require(_int(settings.get("draws_per_iteration"), f"{where}.settings.draws_per_iteration") == DRAW_STRIDE,
             f"{where}.settings.draws_per_iteration: expected {DRAW_STRIDE}")
    retry = _obj(settings.get("repetition_aware_retry"), f"{where}.settings.repetition_aware_retry")
    _require(_finite(retry.get("temperature"), f"{where}.settings.repetition_aware_retry.temperature") == 1.0,
             f"{where}.settings.repetition_aware_retry.temperature: expected 1.0")
    _require(_int(retry.get("top_k"), f"{where}.settings.repetition_aware_retry.top_k") == 30,
             f"{where}.settings.repetition_aware_retry.top_k: expected 30")
    _require(_finite(retry.get("top_p"), f"{where}.settings.repetition_aware_retry.top_p") == 0.95,
             f"{where}.settings.repetition_aware_retry.top_p: expected 0.95")
    _require(_int(retry.get("window_frames"), f"{where}.settings.repetition_aware_retry.window_frames") == 16,
             f"{where}.settings.repetition_aware_retry.window_frames: expected 16")
    tape = _obj(settings.get("random_tape"), f"{where}.settings.random_tape")
    _sha(tape.get("sha256"), f"{where}.settings.random_tape.sha256")
    tape_count = _int(tape.get("uniform_count"), f"{where}.settings.random_tape.uniform_count", minimum=1)
    used_count = _int(tape.get("used_uniform_count"), f"{where}.settings.random_tape.used_uniform_count", minimum=1)
    _require(used_count <= tape_count, f"{where}.settings.random_tape: used count exceeds tape length")
    _require(_int(tape.get("host_uploads"), f"{where}.settings.random_tape.host_uploads") == 1,
             f"{where}.settings.random_tape.host_uploads: expected one upload")

    termination = _obj(value.get("termination"), f"{where}.termination")
    iterations = _int(termination.get("generation_iterations"), f"{where}.termination.generation_iterations", minimum=1)
    returned = _int(termination.get("returned_frames"), f"{where}.termination.returned_frames", minimum=1)
    _require(iterations <= FRAME_LIMIT, f"{where}.termination: iterations exceed frame cap")
    _require(used_count == iterations * DRAW_STRIDE,
             f"{where}.settings.random_tape.used_uniform_count: must equal iterations * {DRAW_STRIDE}")
    status = termination.get("status")
    is_eos = status == "eos"
    is_frame_cap = status in ("frame_cap", "max_frames")
    _require(is_eos or is_frame_cap,
             f"{where}.termination.status: expected eos, frame_cap, or max_frames")
    _require(returned == iterations - 1 if is_eos else returned == iterations,
             f"{where}.termination: returned frame count conflicts with status")
    eos_detected = termination.get("eos_detected")
    cap_reached = termination.get("frame_cap_reached")
    _require(isinstance(eos_detected, bool) and isinstance(cap_reached, bool),
             f"{where}.termination: expected boolean EOS and cap flags")
    _require(eos_detected == is_eos and cap_reached == is_frame_cap,
             f"{where}.termination: status and termination flags disagree")
    _require(_int(termination.get("max_frames"), f"{where}.termination.max_frames") == FRAME_LIMIT,
             f"{where}.termination.max_frames: expected {FRAME_LIMIT}")

    output = _obj(value.get("output"), f"{where}.output")
    frames = _normalize_frames(output.get("frames"), f"{where}.output")
    shape = _array(output.get("frame_shape"), f"{where}.output.frame_shape")
    _require(shape == [len(frames), 11], f"{where}.output.frame_shape: inconsistent with frames")
    _require(len(frames) == returned, f"{where}.output: frame count disagrees with termination")
    _validate_generation_termination(
        frames=len(frames), max_frames=termination.get("max_frames"),
        stopped_on_eos=is_eos, truncated=is_frame_cap, where=where,
    )
    _require(_int(output.get("samples"), f"{where}.output.samples") == len(frames) * 2048,
             f"{where}.output.samples: expected frames * 2048")
    _sha(output.get("wav_sha256"), f"{where}.output.wav_sha256")
    timings = _obj(value.get("timings_seconds"), f"{where}.timings_seconds")
    for name, seconds in timings.items():
        _finite(seconds, f"{where}.timings_seconds.{name}", minimum=0.0)
    return {
        "raw": value, "text_sha256": text_sha, "prompt_ids": prompt_ids,
        "frames": frames, "tape": tape, "status": status,
        "iterations": iterations, "returned_frames": returned,
    }


def _checked_prefix_bytes(uniforms: Any) -> bytes:
    values = _array(uniforms, "tape.uniforms")
    _require(len(values) >= TAPE_PREFIX_COUNT,
             f"tape.uniforms: expected at least {TAPE_PREFIX_COUNT} values")
    packed = bytearray()
    for index, value in enumerate(values):
        number = _finite(value, f"tape.uniforms[{index}]")
        _require(0.0 <= number <= 1.0, f"tape.uniforms[{index}]: expected value in [0,1]")
        try:
            packed.extend(struct.pack("<f", number))
        except (OverflowError, struct.error) as error:
            raise ValueError(f"tape.uniforms[{index}]: outside F32 range") from error
    return bytes(packed)


def validate_tape_proof(tape: Any, tape_sha256: str, proof: Any) -> dict[str, Any]:
    """Fail closed unless the complete 512x11 native producer prefix is proven."""
    tape_obj = _obj(tape, "tape")
    _sha(tape_sha256, "tape SHA-256")
    proof_obj = _obj(proof, "tape proof")
    prefix = _checked_prefix_bytes(tape_obj.get("uniforms"))
    uniforms = tape_obj["uniforms"]
    _require(proof_obj.get("prefix_exact") is True, "tape proof.prefix_exact: expected true")
    _require(_int(proof_obj.get("compared_uniforms"), "tape proof.compared_uniforms") == TAPE_PREFIX_COUNT,
             f"tape proof.compared_uniforms: expected {TAPE_PREFIX_COUNT}")
    _require(_int(proof_obj.get("frames"), "tape proof.frames") == FRAME_LIMIT,
             f"tape proof.frames: expected {FRAME_LIMIT}")
    _require(_int(proof_obj.get("stride"), "tape proof.stride") == DRAW_STRIDE,
             f"tape proof.stride: expected {DRAW_STRIDE}")
    _require(_int(proof_obj.get("seed"), "tape proof.seed") == 1, "tape proof.seed: expected 1")
    _require(_sha(proof_obj.get("tape_sha256"), "tape proof.tape_sha256") == tape_sha256,
             "tape proof: tape SHA-256 mismatch")
    prefix_sha = _hash_bytes(prefix[:TAPE_PREFIX_COUNT * 4])
    _require(_sha(proof_obj.get("prefix_f32_sha256"), "tape proof.prefix_f32_sha256") == prefix_sha,
             "tape proof: F32 producer prefix SHA-256 mismatch")
    producer = proof_obj.get("producer")
    _require(isinstance(producer, str) and "ResidentGenerator#sampling_draws" in producer,
             "tape proof.producer: expected ResidentGenerator#sampling_draws producer")
    source = proof_obj.get("production_source")
    _require(isinstance(source, str) and source.strip(), "tape proof.production_source: expected source path")
    source_sha = _sha(proof_obj.get("production_source_sha256"),
                      "tape proof.production_source_sha256")
    return {
        "seed": 1,
        "frames": FRAME_LIMIT,
        "stride": DRAW_STRIDE,
        "compared_uniforms": TAPE_PREFIX_COUNT,
        "tape_uniform_count": len(uniforms),
        "tape_sha256": tape_sha256,
        "prefix_f32_sha256": prefix_sha,
        "producer": producer,
        "production_source": source,
        "production_source_sha256": source_sha,
    }


def validate_report_contract(
    reports: dict[str, Any], *, tape: Any, tape_sha256: str,
    tape_path: Path, tape_proof: Any, fixture_manifest: Any,
    native_encoder_report: Any,
    language: str = "ru",
) -> dict[str, Any]:
    """Validate four reports, shared settings, tape lineage, and reference gates."""
    names = report_names(language)
    native_ref_name, native_plain_name, mlx_ref_name, mlx_plain_name = names
    _require(set(reports) == set(names), "reports: expected all four reference/no-reference reports")
    native_ref = _native_report(reports[native_ref_name], native_ref_name)
    native_plain = _native_report(reports[native_plain_name], native_plain_name)
    mlx_ref = _mlx_report(reports[mlx_ref_name], mlx_ref_name)
    mlx_plain = _mlx_report(reports[mlx_plain_name], mlx_plain_name)

    texts = [native_ref["raw"]["text"], native_plain["raw"]["text"],
             mlx_ref["raw"]["input"]["text"], mlx_plain["raw"]["input"]["text"]]
    _require(len(set(texts)) == 1, "reports: target text differs across the four runs")
    target_text_sha = _hash_bytes(texts[0].encode("utf-8"))
    _require(mlx_ref["text_sha256"] == mlx_plain["text_sha256"] == target_text_sha,
             "MLX reports: target text UTF-8 hash mismatch")
    mlx_upstream = _obj(mlx_ref["raw"].get("upstream"), f"{mlx_ref_name}.upstream")
    plain_upstream = _obj(mlx_plain["raw"].get("upstream"), f"{mlx_plain_name}.upstream")
    for where, upstream in ((f"{mlx_ref_name}.upstream", mlx_upstream),
                            (f"{mlx_plain_name}.upstream", plain_upstream)):
        _require(isinstance(upstream.get("repository"), str) and upstream["repository"],
                 f"{where}.repository: expected non-empty path")
        _require(isinstance(upstream.get("revision"), str) and upstream["revision"],
                 f"{where}.revision: expected non-empty revision")
        _require(upstream.get("worktree_clean") is True,
                 f"{where}.worktree_clean: expected true")
    _require(mlx_upstream == plain_upstream,
             "MLX reports: pinned upstream source differs between reference conditions")
    mlx_model = _obj(mlx_ref["raw"].get("model"), f"{mlx_ref_name}.model")
    plain_model = _obj(mlx_plain["raw"].get("model"), f"{mlx_plain_name}.model")
    _require(mlx_model == plain_model,
             "MLX reports: model identity differs between reference conditions")
    for key in ("repository", "revision", "config_sha256", "index_sha256"):
        _require(isinstance(mlx_model.get(key), str) and mlx_model[key],
                 f"mlx model.{key}: expected non-empty identity")
    _require(_int(mlx_model.get("sample_rate"), "mlx model.sample_rate") == 44100,
             "mlx model.sample_rate: expected 44100")
    _require(_int(mlx_model.get("num_codebooks"), "mlx model.num_codebooks") == 10 and
             _int(mlx_model.get("codebook_size"), "mlx model.codebook_size") == 4096,
             "mlx model: expected ten codebooks with 4096 entries")
    for name, checked in ((mlx_ref_name, mlx_ref), (mlx_plain_name, mlx_plain)):
        _require(checked["tape"]["sha256"] == tape_sha256, f"{name}: random tape SHA-256 mismatch")
        _require(checked["tape"]["uniform_count"] == len(_array(_obj(tape, "tape").get("uniforms"), "tape.uniforms")),
                 f"{name}: random tape length differs from tape file")
        _require(checked["tape"]["uniform_count"] >= TAPE_PREFIX_COUNT,
                 f"{name}: random tape is shorter than the proved 512-frame prefix")
    proof_summary = validate_tape_proof(tape, tape_sha256, tape_proof)
    for name, checked in ((native_ref_name, native_ref),
                          (native_plain_name, native_plain)):
        _require(
            checked["sampler_inputs_sha256"] == proof_summary["prefix_f32_sha256"],
            f"{name}.sampler_inputs.sha256: does not match the proved first {TAPE_PREFIX_COUNT} GPU tape uniforms",
        )

    manifest = _obj(fixture_manifest, "fixture manifest")
    _require(manifest.get("format") == FIXTURE_FORMAT, "fixture manifest: unexpected format")
    fixture_input = _obj(manifest.get("input"), "fixture manifest.input")
    fixture_wav_sha = _sha(fixture_input.get("wav_sha256"), "fixture manifest.input.wav_sha256")
    transcript_sha = _sha(fixture_input.get("transcript_utf8_sha256"),
                          "fixture manifest.input.transcript_utf8_sha256")
    _require(_sha(fixture_input.get("target_text_utf8_sha256"),
                  "fixture manifest.input.target_text_utf8_sha256") == target_text_sha,
             "fixture manifest: target text hash differs from generation reports")
    _require(_int(fixture_input.get("sample_rate"), "fixture manifest.input.sample_rate") == 44100,
             "fixture manifest.input.sample_rate: expected 44100")
    _require(_int(fixture_input.get("channels"), "fixture manifest.input.channels") == 1,
             "fixture manifest.input.channels: expected mono")
    _require(_int(fixture_input.get("sample_width_bits"), "fixture manifest.input.sample_width_bits") == 16,
             "fixture manifest.input.sample_width_bits: expected PCM16")
    reference_samples = _int(fixture_input.get("samples"), "fixture manifest.input.samples", minimum=1)
    codes = _obj(manifest.get("codes"), "fixture manifest.codes")
    code_sha = _sha(codes.get("sha256"), "fixture manifest.codes.sha256")
    _require(codes.get("shape") == [10, (reference_samples + 2047) // 2048],
             "fixture manifest.codes.shape: expected [10, ceil(samples/2048)]")
    _require(codes.get("layout") == "codebook-major", "fixture manifest.codes.layout: expected codebook-major")
    _require(codes.get("dtype") == "int32", "fixture manifest.codes.dtype: expected int32")
    codec = _obj(manifest.get("codec"), "fixture manifest.codec")
    codec_sha = _sha(codec.get("sha256"), "fixture manifest.codec.sha256")

    prompt = _obj(manifest.get("prompt"), "fixture manifest.prompt")
    prompt_sha = _sha(prompt.get("sha256"), "fixture manifest.prompt.sha256")
    prompt_shape = _array(prompt.get("shape"), "fixture manifest.prompt.shape")
    _require(len(prompt_shape) == 2 and all(_int(x, "fixture manifest.prompt.shape[]", minimum=1) > 0 for x in prompt_shape),
             "fixture manifest.prompt.shape: expected positive [rows,11]")
    _require(prompt_shape[1] == 11, "fixture manifest.prompt.shape: expected 11 columns")
    _require(prompt.get("name") == "prompt.rows", "fixture manifest.prompt.name: expected prompt.rows")
    _require(prompt.get("dtype") == "float32-le", "fixture manifest.prompt.dtype: expected float32-le")
    _require(_sha(prompt.get("target_text_utf8_sha256"), "fixture manifest.prompt.target_text_utf8_sha256") == target_text_sha,
             "fixture manifest.prompt: target text hash mismatch")
    _require(_sha(prompt.get("transcript_utf8_sha256"), "fixture manifest.prompt.transcript_utf8_sha256") == transcript_sha,
             "fixture manifest.prompt: transcript hash mismatch")
    _require(_sha(prompt.get("reference_codes_sha256"), "fixture manifest.prompt.reference_codes_sha256") == code_sha,
             "fixture manifest.prompt: reference code hash mismatch")

    encoder = _obj(native_encoder_report, "native encoder report")
    _require(encoder.get("format") == ENCODER_FORMAT, "native encoder report: unexpected format")
    _require(_sha(encoder.get("waveform_sha256"), "native encoder report.waveform_sha256") == fixture_wav_sha,
             "native encoder report: reference WAV hash mismatch")
    _require(_int(encoder.get("samples"), "native encoder report.samples") == reference_samples,
             "native encoder report: sample count mismatch")
    _require(_sha(encoder.get("reference_text_sha256"), "native encoder report.reference_text_sha256") == transcript_sha,
             "native encoder report: transcript hash mismatch")
    _require(_sha(encoder.get("text_sha256"), "native encoder report.text_sha256") == target_text_sha,
             "native encoder report: target text hash mismatch")
    _require(_sha(encoder.get("codec_sha256"), "native encoder report.codec_sha256") == codec_sha,
             "native encoder report: codec weight SHA-256 differs from fixture")
    _require(encoder.get("code_ids_exact") is True, "native encoder report: reference code IDs are not exact")
    native_prompt = _obj(encoder.get("prompt"), "native encoder report.prompt")
    _require(native_prompt.get("exact_required") is True and native_prompt.get("elementwise_exact") is True
             and native_prompt.get("passes") is True,
             "native encoder report: reference prompt is not verified elementwise exact")
    _require(native_prompt.get("shape") == prompt_shape, "native encoder report: prompt shape mismatch")
    numeric_gate = encoder.get("full_gate_passes")
    _require(isinstance(numeric_gate, bool),
             "native encoder report.full_gate_passes: expected explicit boolean")
    first_failed_stage = encoder.get("first_failed_stage")
    if numeric_gate:
        _require(first_failed_stage is None,
                 "native encoder report.first_failed_stage: expected null when numeric gate passes")
    else:
        _require(isinstance(first_failed_stage, str) and first_failed_stage,
                 "native encoder report.first_failed_stage: required when numeric gate fails")

    mlx_voice = _obj(mlx_ref["raw"].get("reference_voice"), f"{mlx_ref_name}.reference_voice")
    _require(mlx_voice.get("supplied") is True and mlx_voice.get("complete_conditioning") is True,
             f"{mlx_ref_name}: reference voice was not fully supplied")
    audio = _obj(mlx_voice.get("audio"), f"{mlx_ref_name}.reference_voice.audio")
    _require(_sha(audio.get("wav_sha256"), "MLX reference WAV SHA") == fixture_wav_sha,
             "MLX reference WAV hash differs from fixture")
    _require(_sha(audio.get("pcm_sha256"), "MLX reference PCM SHA") ==
             _sha(fixture_input.get("pcm_sha256"), "fixture manifest.input.pcm_sha256"),
             "MLX reference PCM payload hash differs from fixture")
    for key, expected in (("sample_rate", 44100), ("channels", 1), ("sample_width_bits", 16), ("samples", reference_samples)):
        _require(_int(audio.get(key), f"MLX reference audio.{key}") == expected,
                 f"MLX reference audio.{key}: expected {expected}")
    _require(_sha(mlx_voice.get("transcript_utf8_sha256"), "MLX reference transcript SHA") == transcript_sha,
             "MLX reference transcript hash differs from fixture")
    _require(isinstance(mlx_voice.get("transcript"), str) and
             _hash_bytes(mlx_voice["transcript"].encode("utf-8")) == transcript_sha,
             "MLX reference transcript text hash mismatch")
    mlx_codes = _obj(mlx_voice.get("codes"), "MLX reference codes")
    _require(mlx_codes.get("shape") == [10, codes["shape"][1]] and
             mlx_codes.get("layout") == "codebook-major" and
             _sha(mlx_codes.get("sha256"), "MLX reference code SHA") == code_sha,
             "MLX reference codes differ from fixture")
    prompt_export = _obj(mlx_voice.get("prompt_rows"), "MLX reference prompt export")
    _require(prompt_export.get("shape") == prompt_shape and prompt_export.get("dtype") == "float32-le",
             "MLX reference prompt export shape/dtype differs from fixture")
    _require(_sha(prompt_export.get("file_sha256"), "MLX prompt export SHA") == prompt_sha and
             _sha(prompt_export.get("rows_sha256"), "MLX prompt rows SHA") == prompt_sha,
             "MLX actual prompt export SHA differs from fixture prompt rows")
    _require(mlx_voice.get("prompt_rows", {}).get("official_builder_byte_equal_after_generation") is True,
             "MLX reference prompt was not verified against the official prompt builder")
    _require(mlx_ref["raw"]["input"]["prompt_rows"] == prompt_shape[0],
             "MLX reference prompt row count differs from fixture")
    _require(mlx_plain["raw"]["reference_voice"] is None,
             "MLX no-reference run unexpectedly has reference conditioning")

    native_reference = native_ref["reference"]
    _require(native_reference is not None, "native reference run is missing reference metadata")
    _require(_sha(native_reference.get("audio_sha256"), "native reference WAV SHA") == fixture_wav_sha,
             "native reference WAV hash differs from fixture")
    _require(_sha(native_reference.get("text_sha256"), "native reference transcript SHA") == transcript_sha,
             "native reference transcript hash differs from fixture")
    _require(_int(native_reference.get("sample_rate"), "native reference sample rate") == 44100 and
             _int(native_reference.get("samples"), "native reference samples") == reference_samples,
             "native reference audio geometry differs from fixture")
    _require(_int(native_reference.get("frames"), "native reference frames") == codes["shape"][1],
             "native reference frame count differs from fixture codes")
    _require(isinstance(native_reference.get("text"), str) and
             _hash_bytes(native_reference["text"].encode("utf-8")) == transcript_sha,
             "native reference transcript text hash mismatch")
    _require(native_reference.get("host_code_read_before_prefill") is False,
             "native reference path reports host code read before prefill")
    _require(native_ref["raw"]["prompt_rows"] == prompt_shape[0],
             "native reference prompt row count differs from fixture")
    _require(native_plain["reference"] is None, "native no-reference run unexpectedly has reference conditioning")
    _require(native_plain["raw"]["prompt_rows"] == len(native_plain["prompt_ids"] or []),
             "native no-reference prompt row count differs from prompt IDs")
    _require(native_plain["prompt_ids"] == mlx_plain["prompt_ids"],
             "native and MLX no-reference prompt IDs differ")

    return {
        "target_text_utf8_sha256": target_text_sha,
        "reference_wav_sha256": fixture_wav_sha,
        "reference_transcript_utf8_sha256": transcript_sha,
        "reference_samples": reference_samples,
        "reference_codes_sha256": code_sha,
        "reference_codec_sha256": codec_sha,
        "reference_prompt_sha256": prompt_sha,
        "reference_prompt_shape": prompt_shape,
        "mlx_upstream": mlx_upstream,
        "mlx_model": mlx_model,
        "tape_path": str(tape_path),
        "tape_prefix_uniforms": proof_summary["compared_uniforms"],
        "tape_proof": proof_summary,
        "native_encoder_gate": {
            "code_ids_exact": True,
            "prompt_exact": True,
            "full_numeric_stage_gate_passes": numeric_gate,
            "first_failed_stage": first_failed_stage,
            "interpretation": (
                "a separate native encoder probe matched the fixture's terminal code IDs and prompt rows; "
                "the generation report's reference metadata is validated independently, and the full encoder "
                "numeric intermediate-stage gate is reported separately"
            ),
        },
        "reports": dict(zip(names, (native_ref, native_plain, mlx_ref, mlx_plain), strict=True)),
    }


def summarize_quiet_runs(
    peaks: list[float], rms: list[float], samples: int, *,
    sample_rate: int = 44100, window_samples: int = WINDOW_SAMPLES,
    quiet_peak_pcm16: int = QUIET_PEAK_PCM16,
    min_pause_seconds: float = 1.0,
) -> dict[str, Any]:
    """Merge consecutive quiet full windows and separate internal pauses/tails."""
    _require(len(peaks) == len(rms), "peaks and RMS arrays must have the same number of windows")
    _int(samples, "samples", minimum=1)
    _int(sample_rate, "sample_rate", minimum=1)
    _int(window_samples, "window_samples", minimum=1)
    _int(quiet_peak_pcm16, "quiet_peak_pcm16", minimum=0)
    min_pause_seconds = _finite(min_pause_seconds, "min_pause_seconds", minimum=0.0)
    full_count = samples // window_samples
    partial_samples = samples % window_samples
    expected_count = full_count + (1 if partial_samples else 0)
    _require(len(peaks) == expected_count,
             f"metric array length: expected {expected_count} windows for sample geometry")
    normalized_peaks = [_finite(x, f"peaks[{i}]", minimum=0.0) for i, x in enumerate(peaks)]
    normalized_rms = [_finite(x, f"rms[{i}]", minimum=0.0) for i, x in enumerate(rms)]
    quiet = [i < full_count and normalized_peaks[i] <= quiet_peak_pcm16 for i in range(len(peaks))]

    runs: list[tuple[int, int]] = []
    i = 0
    while i < full_count:
        if not quiet[i]:
            i += 1
            continue
        start = i
        i += 1
        while i < full_count and quiet[i]:
            i += 1
        end = i
        if (end - start) * window_samples / sample_rate >= min_pause_seconds:
            runs.append((start, end))

    def as_run(start: int, end: int) -> dict[str, Any]:
        return {
            "start_window": start,
            "end_window_exclusive": end,
            "start_seconds": round(start * window_samples / sample_rate, 6),
            "end_seconds": round(end * window_samples / sample_rate, 6),
            "duration_seconds": round((end - start) * window_samples / sample_rate, 6),
            "window_count": end - start,
        }

    leading = [as_run(start, end) for start, end in runs if start == 0]
    trailing = [as_run(start, end) for start, end in runs if end == full_count]
    internal = [as_run(start, end) for start, end in runs if start > 0 and end < full_count]
    partial: dict[str, Any] | None = None
    if partial_samples:
        index = full_count
        peak = normalized_peaks[index]
        partial = {
            "start_seconds": round(full_count * window_samples / sample_rate, 6),
            "duration_seconds": round(partial_samples / sample_rate, 6),
            "peak_pcm16": peak,
            "rms_pcm16": normalized_rms[index],
            "peak_threshold_met": peak <= quiet_peak_pcm16,
            "full_window": False,
            "quiet_full_window": False,
        }

    return {
        "sample_rate": sample_rate,
        "window_samples": window_samples,
        "window_seconds": window_samples / sample_rate,
        "quiet_peak_threshold_pcm16": quiet_peak_pcm16,
        "minimum_pause_seconds": min_pause_seconds,
        "full_window_count": full_count,
        "quiet_full_window_count": sum(quiet[:full_count]),
        "internal_pauses": internal,
        "leading_tails": leading,
        "trailing_tails": trailing,
        "partial_tail_window": partial,
    }


def _read_pcm16_wav(path: Path, expected_sha: str | None, expected_samples: int,
                    where: str) -> tuple[bytes, dict[str, Any]]:
    wav_sha = _hash_file(path)
    if expected_sha is not None:
        _require(wav_sha == expected_sha, f"{where}: WAV file SHA-256 mismatch")
    try:
        with wave.open(str(path), "rb") as stream:
            params = stream.getparams()
            frames = stream.readframes(params.nframes)
    except (OSError, wave.Error) as error:
        raise ValueError(f"{where}: cannot read WAV: {error}") from error
    _require(params.nchannels == 1, f"{where}: expected mono WAV")
    _require(params.sampwidth == 2, f"{where}: expected signed PCM16 WAV")
    _require(params.framerate == 44100, f"{where}: expected 44100 Hz WAV")
    _require(params.comptype == "NONE", f"{where}: expected uncompressed PCM WAV")
    _require(params.nframes == expected_samples, f"{where}: sample count differs from report")
    _require(len(frames) == expected_samples * 2, f"{where}: truncated PCM payload")
    return frames, {
        "path": str(path),
        "wav_sha256": wav_sha,
        "pcm_bytes_sha256": _hash_bytes(frames),
        "sample_rate": params.framerate,
        "channels": params.nchannels,
        "sample_width_bits": params.sampwidth * 8,
        "samples": params.nframes,
        "duration_seconds": params.nframes / params.framerate,
    }


def _gpu_waveform_metrics(pcm_bytes: bytes, samples: int, mx: Any, np: Any) -> dict[str, Any]:
    """Upload PCM16 integers unchanged, calculate all amplitude metrics on MLX GPU."""
    int16 = np.frombuffer(pcm_bytes, dtype="<i2")
    raw = mx.array(int16)
    signal = raw.astype(mx.float32)
    window_count = (samples + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
    pad = window_count * WINDOW_SAMPLES - samples
    if pad:
        signal_for_windows = mx.concatenate([signal, mx.zeros((pad,), dtype=mx.float32)], axis=0)
    else:
        signal_for_windows = signal
    windows = signal_for_windows.reshape((window_count, WINDOW_SAMPLES))
    abs_windows = mx.abs(windows)
    full_count = samples // WINDOW_SAMPLES
    full_abs = abs_windows[:full_count]
    full_square = mx.square(windows[:full_count])
    full_peaks = mx.max(full_abs, axis=1) if full_count else mx.zeros((0,), dtype=mx.float32)
    full_rms = mx.sqrt(mx.mean(full_square, axis=1)) if full_count else mx.zeros((0,), dtype=mx.float32)
    if samples % WINDOW_SAMPLES:
        partial = signal[full_count * WINDOW_SAMPLES:]
        partial_peak = mx.max(mx.abs(partial))
        partial_rms = mx.sqrt(mx.mean(mx.square(partial)))
        peaks = mx.concatenate([full_peaks, mx.reshape(partial_peak, (1,))], axis=0)
        rms = mx.concatenate([full_rms, mx.reshape(partial_rms, (1,))], axis=0)
    else:
        peaks, rms = full_peaks, full_rms
    waveform_peak = mx.max(mx.abs(signal))
    waveform_rms = mx.sqrt(mx.mean(mx.square(signal)))
    mx.eval(peaks, rms, waveform_peak, waveform_rms)
    return {
        "window_peaks_pcm16": np.asarray(peaks).astype(float).tolist(),
        "window_rms_pcm16": np.asarray(rms).astype(float).tolist(),
        "peak_pcm16": float(np.asarray(waveform_peak)),
        "rms_pcm16": float(np.asarray(waveform_rms)),
        "gpu": str(mx.default_device()),
        "upload_dtype": "signed-int16; values preserved exactly before GPU cast to float32",
        "metric_dtype": "float32",
    }


def _gpu_frame_comparison(native_frames: list[list[int]], mlx_frames: list[list[int]], mx: Any, np: Any) -> dict[str, Any]:
    n = mx.array(np.asarray(native_frames, dtype=np.int32))
    m = mx.array(np.asarray(mlx_frames, dtype=np.int32))
    common_len = min(len(native_frames), len(mlx_frames))

    def compare_columns(start: int, end: int) -> dict[str, Any]:
        row_equal = mx.all(n[:common_len, start:end] == m[:common_len, start:end], axis=1)
        mx.eval(row_equal)
        same = np.asarray(row_equal).astype(bool).tolist()
        prefix = next((i for i, equal in enumerate(same) if not equal), common_len)
        return {
            "common_prefix_frames": prefix,
            "common_prefix_fraction_of_shorter": prefix / common_len if common_len else 0.0,
            "common_prefix_exact": prefix == common_len,
            "all_frames_exact": len(native_frames) == len(mlx_frames) and prefix == common_len,
            "first_mismatch_frame": None if prefix == common_len else prefix,
            "comparison": "exact elementwise integer equality on MLX GPU",
        }

    return {
        "native_frames": len(native_frames),
        "mlx_frames": len(mlx_frames),
        "main_token": compare_columns(0, 1),
        "codebooks": compare_columns(1, 11),
        "all_tokens": compare_columns(0, 11),
    }


def _prompt_file_check(
    manifest_path: Path, manifest: dict[str, Any], checked: dict[str, Any], *, language: str = "ru",
) -> dict[str, Any]:
    _require(isinstance(manifest.get("stage_files"), str) and manifest["stage_files"],
             "fixture manifest.stage_files: expected format description")
    stages = _array(manifest.get("stages"), "fixture manifest.stages")
    _require(bool(stages), "fixture manifest.stages: expected non-empty stage list")
    stage_names: set[str] = set()
    for index, raw_stage in enumerate(stages):
        where = f"fixture manifest.stages[{index}]"
        stage = _obj(raw_stage, where)
        name = stage.get("name")
        _require(isinstance(name, str) and name, f"{where}.name: expected non-empty string")
        _require(name not in stage_names, f"{where}.name: duplicate stage {name}")
        stage_names.add(name)
        shape = _array(stage.get("shape"), f"{where}.shape")
        _require(len(shape) == 2 and all(_int(x, f"{where}.shape[]", minimum=1) > 0 for x in shape),
                 f"{where}.shape: expected positive [rows,channels]")
        _require(stage.get("dtype") == "float32-le", f"{where}.dtype: expected float32-le")
        _require(isinstance(stage.get("row_axis"), str) and stage["row_axis"],
                 f"{where}.row_axis: expected non-empty string")
        _require(isinstance(stage.get("channel_axis"), str) and stage["channel_axis"],
                 f"{where}.channel_axis: expected non-empty string")
        _sha(stage.get("sha256"), f"{where}.sha256")
        filename = stage.get("file")
        _require(isinstance(filename, str) and filename,
                 f"{where}.file: expected relative path")
        stage_path = (manifest_path.parent / filename).resolve()
        _require(manifest_path.parent.resolve() in stage_path.parents,
                 f"{where}.file: resolved path escaped manifest directory")
    prompt_meta = _obj(manifest.get("prompt"), "fixture manifest.prompt")
    relative = prompt_meta.get("file")
    _require(isinstance(relative, str) and relative, "fixture manifest.prompt.file: expected relative path")
    prompt_path = (manifest_path.parent / relative).resolve()
    _require(manifest_path.parent.resolve() in prompt_path.parents,
             "fixture manifest.prompt.file: resolved file escaped manifest directory")
    expected_sha = _sha(prompt_meta.get("sha256"), "fixture manifest.prompt.sha256")
    _require(_hash_file(prompt_path) == expected_sha, "fixture prompt file SHA-256 mismatch")
    _require(prompt_path.stat().st_size == prompt_meta["shape"][0] * prompt_meta["shape"][1] * 4,
             "fixture prompt file byte length differs from declared F32 shape")
    mlx_ref_name = report_names(language)[2]
    actual_meta = checked["reports"][mlx_ref_name]["raw"]["reference_voice"]["prompt_rows"]
    actual_path = Path(actual_meta["path"]).resolve()
    _require(actual_path.is_file(), f"MLX prompt export is missing: {actual_path}")
    actual_sha = _hash_file(actual_path)
    _require(actual_sha == expected_sha, "actual MLX prompt export file SHA differs from fixture")
    _require(actual_path.stat().st_size == prompt_path.stat().st_size,
             "actual MLX prompt export byte length differs from fixture")
    return {
        "fixture_path": str(prompt_path),
        "actual_mlx_prompt_path": str(actual_path),
        "shape": prompt_meta["shape"],
        "dtype": "float32-le",
        "sha256": expected_sha,
        "file_bytes": prompt_path.stat().st_size,
        "byte_equal": True,
        "official_builder_byte_equal_after_generation": True,
        "encoder_stage_manifest_entries": len(stages),
    }


def _load_all_contracts(
    run_dir: Path, tape_path: Path, proof_path: Path,
    encoder_report_path: Path | None = None,
    *, language: str = "ru",
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    reports: dict[str, Any] = {}
    paths = report_paths(run_dir, language)
    hashes: dict[str, str] = {}
    for name, report_path in paths.items():
        reports[name] = _load_json(report_path, name)
        hashes[str(report_path)] = _hash_file(report_path)

    fixture_path = run_dir / "mlx-encoder" / "manifest.json"
    fixture_manifest = _load_json(fixture_path, "fixture manifest")
    encoder_path = (encoder_report_path or (run_dir / "native-encoder-depth2-v2.json")).resolve()
    native_encoder_report = _load_json(encoder_path, "native encoder report")
    tape = _load_json(tape_path, "random tape")
    proof = _load_json(proof_path, "random tape producer proof")
    tape_sha = _hash_file(tape_path)
    hashes.update({
        str(fixture_path): _hash_file(fixture_path),
        str(encoder_path): _hash_file(encoder_path),
        str(tape_path): tape_sha,
        str(proof_path): _hash_file(proof_path),
    })
    checked = validate_report_contract(
        reports, tape=tape, tape_sha256=tape_sha, tape_path=tape_path,
        tape_proof=proof, fixture_manifest=fixture_manifest,
        native_encoder_report=native_encoder_report, language=language,
    )
    manifest_sha = _hash_file(fixture_path)
    _require(_sha(native_encoder_report.get("manifest_sha256"),
                  "native encoder report.manifest_sha256") == manifest_sha,
             "native encoder report: fixture manifest SHA-256 mismatch")
    source_path = Path(checked["tape_proof"]["production_source"]).resolve()
    checked["tape_production_source"] = validate_source_hash(
        source_path, checked["tape_proof"]["production_source_sha256"]
    )
    fixture_input = fixture_manifest["input"]
    reference_wav_raw_path = fixture_input.get("wav_path")
    _require(isinstance(reference_wav_raw_path, str) and reference_wav_raw_path,
             "fixture manifest.input.wav_path: expected path")
    reference_wav_path = Path(reference_wav_raw_path)
    if not reference_wav_path.is_absolute():
        reference_wav_path = fixture_path.parent / reference_wav_path
    reference_pcm, reference_wav_meta = _read_pcm16_wav(
        reference_wav_path, expected_sha=checked["reference_wav_sha256"],
        expected_samples=checked["reference_samples"], where="reference input WAV",
    )
    expected_pcm_sha = _sha(fixture_input.get("pcm_sha256"), "fixture manifest.input.pcm_sha256")
    reference_pcm_sha = _hash_bytes(reference_pcm)
    _require(reference_pcm_sha == expected_pcm_sha,
             "reference input PCM payload SHA-256 differs from fixture")
    checked["reference_audio_file"] = reference_wav_meta
    prompt_gate = _prompt_file_check(fixture_path, fixture_manifest, checked, language=language)
    checked["prompt_gate"] = prompt_gate
    hashes[str(source_path)] = checked["tape_production_source"]["sha256"]
    hashes[str(reference_wav_path.resolve())] = reference_wav_meta["wav_sha256"]
    hashes[prompt_gate["fixture_path"]] = prompt_gate["sha256"]
    hashes[prompt_gate["actual_mlx_prompt_path"]] = prompt_gate["sha256"]
    return checked, paths, hashes


def build_report(run_dir: Path, random_tape: Path, tape_proof: Path, *, output: Path,
                 encoder_report_path: Path | None = None, language: str = "ru") -> dict[str, Any]:
    run_dir = run_dir.resolve()
    random_tape = random_tape.resolve()
    tape_proof = tape_proof.resolve()
    names = report_names(language)
    native_ref_name, native_plain_name, mlx_ref_name, mlx_plain_name = names
    checked, report_paths, hashes = _load_all_contracts(
        run_dir, random_tape, tape_proof, encoder_report_path, language=language
    )
    report_docs = {name: checked["reports"][name]["raw"] for name in names}

    wav_paths = {name: run_dir / f"{name}.wav" for name in names}
    wav_meta: dict[str, dict[str, Any]] = {}
    pcm_data: dict[str, bytes] = {}
    for name in names:
        if name.startswith("native"):
            report_info = checked["reports"][name]["raw"]
            expected_samples = report_info["samples"]
            expected_wav_sha = None
        else:
            report_info = checked["reports"][name]["raw"]["output"]
            expected_samples = report_info["samples"]
            expected_wav_sha = report_info["wav_sha256"]
        raw, metadata = _read_pcm16_wav(
            wav_paths[name], expected_sha=expected_wav_sha,
            expected_samples=expected_samples, where=name,
        )
        wav_meta[name] = metadata
        pcm_data[name] = raw
        hashes[str(wav_paths[name])] = metadata["wav_sha256"]

    # GPU libraries are intentionally imported only after all metadata, hashes,
    # and fail-closed schema checks have succeeded.
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError as error:
        raise RuntimeError("MLX and NumPy are required for GPU waveform/code analysis") from error
    device = mx.default_device()
    _require("gpu" in str(device).lower(), f"expected MLX GPU device, found {device}")

    waveform_summary: dict[str, Any] = {}
    for name in names:
        metrics = _gpu_waveform_metrics(pcm_data[name], wav_meta[name]["samples"], mx, np)
        metrics["quiet_windows"] = summarize_quiet_runs(
            metrics["window_peaks_pcm16"], metrics["window_rms_pcm16"],
            samples=wav_meta[name]["samples"],
        )
        metrics["wav"] = wav_meta[name]
        waveform_summary[name] = metrics

    native_ref = checked["reports"][native_ref_name]
    native_plain = checked["reports"][native_plain_name]
    mlx_ref = checked["reports"][mlx_ref_name]
    mlx_plain = checked["reports"][mlx_plain_name]
    pair_comparisons = {
        "reference": _gpu_frame_comparison(native_ref["frames"], mlx_ref["frames"], mx, np),
        "no_reference": _gpu_frame_comparison(native_plain["frames"], mlx_plain["frames"], mx, np),
    }
    prompt_ids_match = native_plain["prompt_ids"] == mlx_plain["prompt_ids"]

    read_ledger = {
        "native_reference": report_docs[native_ref_name].get("resident_readbacks"),
        "native_no_reference": report_docs[native_plain_name].get("resident_readbacks"),
        "mlx_reference": report_docs[mlx_ref_name].get("runtime", {}).get("host_readback_scope"),
        "mlx_no_reference": report_docs[mlx_plain_name].get("runtime", {}).get("host_readback_scope"),
    }
    phases = {
        name: report_docs[name].get("phase_seconds") if name.startswith("native")
        else report_docs[name].get("timings_seconds")
        for name in names
    }
    output_report = {
        "format": REPORT_FORMAT,
        "claim_scope": (
            "Completed-WAV GPU diagnostics and exact token-prefix comparison for the supplied runs. "
            "Quiet windows are a waveform amplitude statistic; this report is not ASR, hearing certification, "
            "a voice-quality judgment, or a causal claim about conditioning."
        ),
        "configuration": {
            "target_text_utf8_sha256": checked["target_text_utf8_sha256"],
            "seed": 1,
            "temperature": 0.8,
            "top_k": 30,
            "top_p": 0.9,
            "max_frames": FRAME_LIMIT,
            "random_tape": checked["tape_proof"],
            "reference_voice": {
                "wav_sha256": checked["reference_wav_sha256"],
                "pcm_sha256": checked["reference_audio_file"]["pcm_bytes_sha256"],
                "transcript_utf8_sha256": checked["reference_transcript_utf8_sha256"],
                "samples": checked["reference_samples"],
                "codes_sha256": checked["reference_codes_sha256"],
                "prompt_fixture_and_actual_mlx_sha256": checked["reference_prompt_sha256"],
                "prompt_shape": checked["reference_prompt_shape"],
                "prompt_gate": checked["prompt_gate"],
                "upstream": {
                    "repository": checked["mlx_upstream"]["repository"],
                    "revision": checked["mlx_upstream"]["revision"],
                    "worktree_clean": checked["mlx_upstream"]["worktree_clean"],
                    "model_repository": checked["mlx_model"]["repository"],
                    "model_revision": checked["mlx_model"]["revision"],
                    "model_config_sha256": checked["mlx_model"]["config_sha256"],
                    "model_index_sha256": checked["mlx_model"]["index_sha256"],
                    "codec_sha256": checked["reference_codec_sha256"],
                },
                "native_encoder_numeric_gate": checked["native_encoder_gate"],
            },
        },
        "input_files": {
            "reports_and_evidence": [
                {"path": path, "sha256": digest} for path, digest in sorted(hashes.items())
            ],
            "random_tape": {"path": str(random_tape), "sha256": hashes[str(random_tape)]},
            "tape_proof": {"path": str(tape_proof), "sha256": hashes[str(tape_proof)]},
        },
        "generation": {
            "reference": {
                "native_frames": len(native_ref["frames"]),
                "mlx_frames": len(mlx_ref["frames"]),
                "native_stopped_on_eos": native_ref["raw"]["stopped_on_eos"],
                "mlx_status": mlx_ref["status"],
                "mlx_iterations": mlx_ref["iterations"],
                "mlx_returned_frames": mlx_ref["returned_frames"],
                "main_token_comparison": pair_comparisons["reference"]["main_token"],
                "codebook_comparison": pair_comparisons["reference"]["codebooks"],
                "all_token_comparison": pair_comparisons["reference"]["all_tokens"],
            },
            "no_reference": {
                "native_frames": len(native_plain["frames"]),
                "mlx_frames": len(mlx_plain["frames"]),
                "native_stopped_on_eos": native_plain["raw"]["stopped_on_eos"],
                "mlx_status": mlx_plain["status"],
                "mlx_iterations": mlx_plain["iterations"],
                "mlx_returned_frames": mlx_plain["returned_frames"],
                "main_token_comparison": pair_comparisons["no_reference"]["main_token"],
                "codebook_comparison": pair_comparisons["no_reference"]["codebooks"],
                "all_token_comparison": pair_comparisons["no_reference"]["all_tokens"],
                "prompt_ids_exact": prompt_ids_match,
                "prompt_rows": len(native_plain["prompt_ids"] or []),
            },
        },
        "waveforms": waveform_summary,
        "phase_timings_seconds": phases,
        "readback_ledger": read_ledger,
        "precision_scope": {
            "waveform_input": "exact little-endian signed PCM16 uploaded to MLX GPU; cast to F32 on GPU",
            "waveform_metrics": "F32 MLX GPU max-absolute and RMS reductions; terminal scalar/window metrics read after evaluation",
            "quiet_rule": f"full {WINDOW_SAMPLES}-sample windows only; peak <= {QUIET_PEAK_PCM16} PCM16; adjacent qualifying windows merged; minimum run 1.0s",
            "partial_tail": "reported diagnostically, excluded from full-window quiet counts and pause runs",
            "code_tokens": "exact integer row equality evaluated on MLX GPU; all 11 IDs compared",
            "tolerance": "none for token IDs or prompt file bytes; waveform metrics are F32 summaries",
        },
    }
    return output_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language", choices=LANGUAGES, default="ru",
        help="select the report filename namespace (default: ru)",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--random-tape", type=Path, required=True)
    parser.add_argument("--tape-proof", type=Path, required=True)
    parser.add_argument("--encoder-report", type=Path,
                        help="native encoder comparison report (default: RUN_DIR/native-encoder-depth2-v2.json)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_report(
            args.run_dir, args.random_tape, args.tape_proof,
            output=args.output, encoder_report_path=args.encoder_report, language=args.language,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, RuntimeError, OSError) as error:
        print(f"reference pause analysis failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(args.output),
        "format": result["format"],
        "reference_exact_codebook_frames": result["generation"]["reference"]["codebook_comparison"]["common_prefix_frames"],
        "no_reference_exact_codebook_frames": result["generation"]["no_reference"]["codebook_comparison"]["common_prefix_frames"],
        "gpu": next(iter(result["waveforms"].values()))["gpu"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
