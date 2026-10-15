#!/usr/bin/env python3
"""Export step-zero MLX prompt diagnostics as a native comparator fixture.

This is a terminal-only packaging tool. It does not import MLX, load model
weights, or run inference; its two F32 arrays come from a completed NPZ capture.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

import numpy as np


FORMAT = "drama3-prefill-reference-v1"
UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
PROMPT_SOURCE_SHA256 = "c2f9c80ed2e4ba6c73a3806e3fdbd94f5cea80c75cce450a003e4e71d17f25c2"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
REQUIRED_SOURCE_PATHS = (
    "mlx_audio/tts/models/fish_omni/prompt.py",
    "scripts/drama3/generate_mlx_control.py",
    "scripts/drama3/replay_mlx_generation.py",
)
STAGE_CONTRACT = (
    {
        "key": "step_00000_prompt_backbone_hidden",
        "name": "prompt.backbone.hidden",
        "shape": [1, 2048],
        "kind": "prompt.backbone.hidden",
        "file": "prompt.backbone.hidden.f32",
    },
    {
        "key": "step_00000_main",
        "name": "prompt.main_logits",
        "shape": [1, 4097],
        "sampler": "main",
        "file": "prompt.main_logits.f32",
    },
)
BASE_MODEL_FILES = {
    "codec.safetensors",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ExportError(ValueError):
    """Input evidence is missing, malformed, or not linked to this fixture."""


def _fail(message):
    raise ExportError(message)


def _object(value, label):
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _string(value, label):
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _utc_timestamp(value, label):
    from datetime import datetime, timedelta
    value = _string(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        _fail(f"{label} must be an ISO timestamp: {error}")
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        _fail(f"{label} must include a UTC timezone")
    return value


def _integer(value, label):
    if type(value) is not int:
        _fail(f"{label} must be an integer")
    return value


def _sha256(value, label):
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_file(value, label):
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        _fail(f"cannot resolve {label}: {error}")
    if not path.is_file():
        _fail(f"{label} is not a regular file: {path}")
    return path


def _output_path(value):
    try:
        raw = Path(value).expanduser()
        path = Path(os.path.abspath(raw))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        _fail(f"invalid output path: {error}")
    if os.path.lexists(path):
        _fail(f"output already exists: {path}")
    try:
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        _fail(f"output parent does not exist: {error}")
    if not parent.is_dir() or path.name in ("", ".", ".."):
        _fail("output must name a new directory inside an existing directory")
    return parent / path.name


def _read_json(path, label):
    try:
        contents = path.read_bytes()
        return _object(json.loads(contents), label), hashlib.sha256(contents).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read {label}: {error}")


def _safe_model_filename(value, label):
    name = _string(value, label)
    if (PurePosixPath(name).name != name or "/" in name or "\\" in name
            or "\x00" in name or name in (".", "..")):
        _fail(f"{label} must be a model filename, got {name!r}")
    return name


def _model_paths(checkpoint):
    model_dir_value = _string(checkpoint.get("model_dir"), "checkpoint model_dir")
    try:
        model_dir = Path(model_dir_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        _fail(f"cannot resolve checkpoint model_dir: {error}")
    if not model_dir.is_dir():
        _fail("checkpoint model_dir is not a directory")

    files = checkpoint.get("files")
    if not isinstance(files, list) or not files:
        _fail("checkpoint files must be a non-empty array")
    records = {}
    for index, entry in enumerate(files):
        entry = _object(entry, f"checkpoint files[{index}]")
        name = _safe_model_filename(entry.get("file"), f"checkpoint files[{index}].file")
        if name in records:
            _fail(f"duplicate checkpoint file {name}")
        digest = _sha256(entry.get("sha256"), f"checkpoint {name} sha256")
        size = _integer(entry.get("bytes"), f"checkpoint {name} bytes")
        mtime = _integer(entry.get("mtime_ns"), f"checkpoint {name} mtime_ns")
        if size < 0 or mtime < 0:
            _fail(f"checkpoint {name} size and mtime must be non-negative")
        path = model_dir / name
        try:
            if path.is_symlink() or not path.is_file():
                _fail(f"checkpoint file is missing or not regular: {name}")
            stat = path.stat()
        except OSError as error:
            _fail(f"cannot stat checkpoint file {name}: {error}")
        if stat.st_size != size or stat.st_mtime_ns != mtime:
            _fail(f"checkpoint manifest metadata is stale for {name}")
        records[name] = {"file": name, "sha256": digest, "bytes": size, "mtime_ns": mtime}

    if not BASE_MODEL_FILES.issubset(records):
        _fail("checkpoint manifest is missing a required config/index/tokenizer/codec file")
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config, config_digest = _read_json(config_path, "checkpoint config.json")
    index, index_digest = _read_json(index_path, "checkpoint model index")
    for name, actual in (("config.json", config_digest),
                         ("model.safetensors.index.json", index_digest)):
        if actual != records[name]["sha256"]:
            _fail(f"checkpoint SHA-256 mismatch for {name}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        _fail("checkpoint index must contain a non-empty weight_map")
    shards = {
        _safe_model_filename(value, "checkpoint indexed shard")
        for value in weight_map.values()
    }
    if not shards or any(not name.endswith(".safetensors") for name in shards):
        _fail("checkpoint index contains no valid safetensors shards")
    if set(records) != BASE_MODEL_FILES | shards:
        _fail("checkpoint manifest must cover exactly config/index/tokenizer/codec and indexed shards")
    if len(shards) != 5 or any(not name.startswith("model-") for name in shards):
        _fail("checkpoint index must identify the five pinned model shards")
    return model_dir, records, config, sorted(shards)


def _validate_report(report, report_path, model_dir, records, config, shards):
    if report.get("format") != "drama3-mlx-control-v1":
        _fail("report format is not drama3-mlx-control-v1")
    _utc_timestamp(report.get("created_utc"), "report created_utc")
    upstream = _object(report.get("upstream"), "report upstream")
    if upstream.get("revision") != UPSTREAM_REVISION or upstream.get("worktree_clean") is not True:
        _fail("report upstream must be the pinned clean MLX-audio revision")
    upstream_repository = _string(upstream.get("repository"), "report upstream repository")
    if not Path(upstream_repository).is_absolute():
        _fail("report upstream repository must be an absolute path")

    source_hashes = _object(report.get("source_hashes"), "report source_hashes")
    for source in REQUIRED_SOURCE_PATHS:
        if source not in source_hashes:
            _fail(f"report source_hashes is missing {source}")
    for source, digest in source_hashes.items():
        source_name = _string(source, "source hash path")
        pure = PurePosixPath(source_name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in source_name or "\x00" in source_name:
            _fail(f"unsafe source hash path {source_name!r}")
        _sha256(digest, f"source hash for {source_name}")
    if source_hashes[REQUIRED_SOURCE_PATHS[0]] != PROMPT_SOURCE_SHA256:
        _fail("report Fish Omni prompt.py source hash differs from the pinned source")

    model = _object(report.get("model"), "report model")
    if model.get("repository") != MODEL_REPOSITORY or model.get("revision") != MODEL_REVISION:
        _fail("report model repository/revision is not the pinned checkpoint")
    if model.get("model_type") != "fish_omni":
        _fail("report model_type is not fish_omni")
    model_path_value = _string(model.get("path"), "report model path")
    try:
        report_model_path = Path(model_path_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        _fail(f"cannot resolve report model path: {error}")
    if report_model_path != model_dir:
        _fail("report model path does not match checkpoint manifest model_dir")

    config_digest = records["config.json"]["sha256"]
    index_digest = records["model.safetensors.index.json"]["sha256"]
    if model.get("config_sha256") != config_digest or model.get("index_sha256") != index_digest:
        _fail("report config/index hashes do not match the checkpoint manifest")
    report_shards = model.get("indexed_shards")
    if (not isinstance(report_shards, list)
            or any(not isinstance(name, str) for name in report_shards)
            or sorted(report_shards) != shards):
        _fail("report indexed_shards do not match the checkpoint index")

    decoder = _object(config.get("audio_decoder_config"), "config audio_decoder_config")
    text_config = _object(config.get("text_config"), "config text_config")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or any(not isinstance(name, str) for name in architectures):
        _fail("config architectures must be an array of strings")
    if (config.get("model_type") != "fish_omni" or "FishOmniModel" not in architectures
            or decoder.get("num_codebooks") != 10 or decoder.get("vocab_size") != 4096
            or text_config.get("hidden_size") != 2048 or config.get("sample_rate") != 44100
            or config.get("semantic_start_token_id") != 248099 or config.get("eos_token_id") != 248046):
        _fail("checkpoint config does not match the pinned Fish Omni prompt dimensions/tokens")
    if (model.get("num_codebooks") != decoder.get("num_codebooks")
            or model.get("codebook_size") != decoder.get("vocab_size")
            or model.get("sample_rate") != config.get("sample_rate")
            or model.get("semantic_begin") != config.get("semantic_start_token_id")
            or model.get("eos_id") != config.get("eos_token_id")):
        _fail("report model dimensions/tokens do not match checkpoint config.json")

    input_data = _object(report.get("input"), "report input")
    if (type(input_data.get("speaker_id")) is not int or input_data.get("speaker_id") != 0
            or input_data.get("speaker_tag") != "<|speaker:0|>"):
        _fail("report input must use the single supported speaker 0")
    text = _string(input_data.get("text"), "report input text")
    try:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        _fail("report input text is not valid UTF-8")
    if input_data.get("text_utf8_sha256") != text_hash:
        _fail("report text UTF-8 hash does not match its text")
    voice = _object(report.get("reference_voice"), "report reference_voice")
    if voice.get("supplied") is not True or voice.get("complete_conditioning") is not True:
        _fail("report must contain a supplied, completely conditioned reference voice")
    audio = _object(voice.get("audio"), "report reference audio")
    wav_hash = _sha256(audio.get("wav_sha256"), "reference WAV hash")
    transcript = _string(voice.get("transcript"), "reference transcript")
    try:
        transcript_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        _fail("report transcript is not valid UTF-8")
    if voice.get("transcript_utf8_sha256") != transcript_hash:
        _fail("report transcript UTF-8 hash does not match its text")

    termination = _object(report.get("termination"), "report termination")
    if (termination.get("status") != "eos" or termination.get("eos_detected") is not True
            or termination.get("frame_cap_reached") is not False):
        _fail("report must show an EOS-completed run below the frame cap")

    settings = _object(report.get("settings"), "report settings")
    capture_steps = settings.get("capture_steps")
    if (not isinstance(capture_steps, list)
            or any(type(step) is not int for step in capture_steps)
            or 0 not in capture_steps):
        _fail("report did not request step-zero capture")

    capture = _object(report.get("sampler_capture"), "report sampler_capture")
    capture_name = _string(capture.get("path"), "sampler_capture path")
    try:
        capture_path = Path(capture_name).expanduser()
        if not capture_path.is_absolute():
            capture_path = report_path.parent / capture_path
        capture_path = capture_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        _fail(f"cannot resolve sampler capture: {error}")
    if not capture_path.is_file():
        _fail("sampler capture is not a regular file")
    capture_digest = _sha256(capture.get("sha256"), "sampler_capture sha256")
    try:
        actual_capture_digest = _hash_file(capture_path)
    except OSError as error:
        _fail(f"cannot hash sampler capture: {error}")
    if actual_capture_digest != capture_digest:
        _fail("sampler capture SHA-256 does not match report")
    entries = capture.get("entries")
    if not isinstance(entries, list):
        _fail("sampler_capture entries must be an array")
    by_key = {}
    for index, entry in enumerate(entries):
        entry = _object(entry, f"sampler_capture entries[{index}]")
        key = _string(entry.get("key"), f"sampler_capture entries[{index}].key")
        if key in by_key:
            _fail(f"duplicate sampler capture entry {key}")
        by_key[key] = entry
    for stage in STAGE_CONTRACT:
        entry = by_key.get(stage["key"])
        if entry is None:
            _fail(f"report is missing step-zero capture {stage['key']}")
        shape = entry.get("shape")
        if (type(entry.get("step")) is not int or entry.get("step") != 0
                or not isinstance(shape, list) or any(type(dim) is not int for dim in shape)
                or shape != stage["shape"]
                or entry.get("dtype") != "float32"):
            _fail(f"report capture metadata does not match {stage['name']}")
        if "kind" in stage and entry.get("kind") != stage["kind"]:
            _fail(f"report capture kind does not match {stage['name']}")
        if "sampler" in stage and entry.get("sampler") != stage["sampler"]:
            _fail(f"report capture sampler does not match {stage['name']}")

    return {
        "text_sha256": text_hash,
        "wav_sha256": wav_hash,
        "transcript_sha256": transcript_hash,
        "capture_path": capture_path,
        "capture_sha256": capture_digest,
        "source_identity": {
            "repository": upstream_repository,
            "revision": UPSTREAM_REVISION,
            "worktree_clean": True,
            "source_hashes": source_hashes,
            "source_hashes_sha256": hashlib.sha256(
                json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
    }


def _load_capture(capture_path, expected_sha256):
    arrays = []
    try:
        with capture_path.open("rb") as source_file:
            before = _hash_stream(source_file)
            if before != expected_sha256:
                _fail("sampler capture changed after its report hash was checked")
            source_file.seek(0)
            archive = np.load(source_file, allow_pickle=False)
            with archive:
                keys = set(archive.files)
                for stage in STAGE_CONTRACT:
                    if stage["key"] not in keys:
                        _fail(f"sampler NPZ is missing {stage['key']}")
                    source = np.asarray(archive[stage["key"]])
                    if source.shape != tuple(stage["shape"]):
                        _fail(f"sampler NPZ {stage['key']} shape is {source.shape}, expected {stage['shape']}")
                    if source.dtype.kind != "f" or source.dtype.itemsize != 4:
                        _fail(f"sampler NPZ {stage['key']} must contain float32")
                    if not np.isfinite(source).all():
                        _fail(f"sampler NPZ {stage['key']} contains non-finite values")
                    arrays.append(np.array(source, dtype="<f4", order="C", copy=True))
            source_file.seek(0)
            after = _hash_stream(source_file)
            if after != before:
                _fail("sampler capture changed while being read")
    except ExportError:
        raise
    except (OSError, ValueError, EOFError, KeyError, TypeError, zipfile.BadZipFile) as error:
        _fail(f"cannot read sampler NPZ: {error}")
    return arrays


def _hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def export_prefill_reference(report, checkpoint_manifest, output_dir):
    """Validate inputs, then write a new v1 terminal-prefill fixture directory."""
    output_path = _output_path(output_dir)
    report_path = _input_file(report, "report")
    checkpoint_path = _input_file(checkpoint_manifest, "checkpoint manifest")
    report_data, report_digest = _read_json(report_path, "report")
    checkpoint_data, checkpoint_digest = _read_json(checkpoint_path, "checkpoint manifest")
    checkpoint_recorded = _utc_timestamp(
        checkpoint_data.get("recorded_at_utc"), "checkpoint recorded_at_utc"
    )
    model_dir, records, config, shards = _model_paths(checkpoint_data)
    identity = _validate_report(report_data, report_path, model_dir, records, config, shards)
    arrays = _load_capture(identity["capture_path"], identity["capture_sha256"])
    stage_entries = []
    for stage, values in zip(STAGE_CONTRACT, arrays):
        raw = values.tobytes(order="C")
        stage_entries.append({
            "name": stage["name"],
            "shape": stage["shape"],
            "row_axis": "time",
            "channel_axis": "features",
            "dtype": "float32-le",
            "file": stage["file"],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "capture_key": stage["key"],
        })

    manifest = {
        "format": FORMAT,
        "upstream_revision": UPSTREAM_REVISION,
        "source_identity": identity["source_identity"],
        "text_sha256": identity["text_sha256"],
        "wav_sha256": identity["wav_sha256"],
        "transcript_sha256": identity["transcript_sha256"],
        "model_files": [records[name] for name in sorted(records)],
        "checkpoint_manifest": str(checkpoint_path),
        "checkpoint_manifest_sha256": checkpoint_digest,
        "checkpoint_recorded_at_utc": checkpoint_recorded,
        "model_identity": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "model_dir": str(model_dir),
            "config_sha256": records["config.json"]["sha256"],
            "index_sha256": records["model.safetensors.index.json"]["sha256"],
        },
        "report": str(report_path),
        "report_sha256": report_digest,
        "capture": str(identity["capture_path"]),
        "capture_sha256": identity["capture_sha256"],
        "stages": stage_entries,
        "scope": "Repacked terminal upstream prefill diagnostic arrays; no native activations read",
    }

    created_files = []
    created_directory = False
    try:
        output_path.mkdir(exist_ok=False)
        created_directory = True
        for stage, values in zip(STAGE_CONTRACT, arrays):
            path = output_path / stage["file"]
            with path.open("xb") as target:
                created_files.append(path)
                target.write(values.tobytes(order="C"))
        manifest_path = output_path / "manifest.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as target:
            created_files.append(manifest_path)
            json.dump(manifest, target, ensure_ascii=False, indent=2)
            target.write("\n")
    except OSError as error:
        for path in created_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if created_directory:
            try:
                output_path.rmdir()
            except OSError:
                pass
        _fail(f"cannot write prefill fixture: {error}")
    return output_path / "manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="completed drama3-mlx-control-v1 report")
    parser.add_argument("--checkpoint-manifest", required=True, help="fresh local checkpoint hash manifest")
    parser.add_argument("--output-dir", required=True, help="new directory for raw F32 stages and manifest")
    args = parser.parse_args(argv)
    try:
        manifest = export_prefill_reference(args.report, args.checkpoint_manifest, args.output_dir)
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
