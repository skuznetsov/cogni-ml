#!/usr/bin/env python3
"""Emit an upstream MLX full-code DAC activation reference.

This diagnostic loads the pinned codec checkpoint without dtype conversion,
decodes a complete code stream through the upstream DAC.decode API, and keeps
the selected intermediate MLX arrays on-device until terminal export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path


EXPECTED_MLX_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
EXPECTED_CODEC_SHA256 = "543445a2ebc5224d2af5956d422f88c2cd21f4018c1a26d81b51ec231bb630f2"
NUM_CODEBOOKS = 10
EXPECTED_STAGE_COUNT = 58
SAMPLE_RATE = 44_100
FRAME_LENGTH = 2_048
CODEBOOK_LIMITS = (4_096,) + (1_024,) * 9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_book_major_codes(codes):
    """Validate exactly ten `[frames]` codebook rows and return plain lists."""
    if not isinstance(codes, (list, tuple)) or len(codes) != NUM_CODEBOOKS:
        raise ValueError("codes must contain exactly 10 codebooks")
    normalized = []
    expected_frames = None
    for book_index, book in enumerate(codes):
        if not isinstance(book, (list, tuple)):
            raise ValueError(f"book {book_index} must be an array of codes")
        if expected_frames is None:
            expected_frames = len(book)
            if expected_frames == 0:
                raise ValueError("codebooks must have the same non-zero frame count")
        elif len(book) != expected_frames:
            raise ValueError("codebooks must have the same non-zero frame count")
        limit = CODEBOOK_LIMITS[book_index]
        row = []
        for frame_index, code in enumerate(book):
            if type(code) is not int:
                raise ValueError(
                    f"book {book_index} frame {frame_index} code must be an integer"
                )
            if not 0 <= code < limit:
                raise ValueError(
                    f"book {book_index} code {code} is outside [0, {limit - 1}]"
                )
            row.append(code)
        normalized.append(row)
    return normalized


def _validate_declared_shape(mapping, key, expected, description):
    declared = mapping.get(key)
    if declared is not None and declared != expected:
        raise ValueError(
            f"{description} {key} {declared!r} does not match payload shape {expected!r}"
        )


def _book_major_from_frame_rows(rows, mapping, description):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{description} frames must be a non-empty array")
    books = [[] for _ in range(NUM_CODEBOOKS)]
    for frame_index, frame in enumerate(rows):
        if not isinstance(frame, list) or len(frame) != NUM_CODEBOOKS + 1:
            raise ValueError(
                f"{description} frame {frame_index} must contain 11 integers "
                "[main,c0,...,c9]"
            )
        if any(type(code) is not int for code in frame):
            raise ValueError(
                f"{description} frame {frame_index} must contain only integers"
            )
        for book_index, code in enumerate(frame[1:]):
            books[book_index].append(code)
    _validate_declared_shape(
        mapping,
        "frame_shape",
        [len(rows), NUM_CODEBOOKS + 1],
        description,
    )
    return validate_book_major_codes(books)


def _codes_from_mapping(mapping, layout):
    if not isinstance(mapping, dict):
        raise ValueError(f"{layout} must be a JSON object")
    if "frames" in mapping:
        return (
            _book_major_from_frame_rows(mapping["frames"], mapping, layout),
            layout + "_time_major_frames",
        )
    for key in ("codes", "generated_codes"):
        if key in mapping:
            codes = validate_book_major_codes(mapping[key])
            _validate_declared_shape(
                mapping,
                "codes_shape",
                [NUM_CODEBOOKS, len(codes[0])],
                layout,
            )
            return codes, "book_major_codes"
    raise ValueError(f"{layout} has neither frames nor book-major codes")


def parse_codes_document(document, chunk_index=None):
    """Read supported native/MLX JSON reports into a validated `[10,T]` list."""
    if not isinstance(document, dict):
        raise ValueError("codes report must be a JSON object")
    if chunk_index is not None:
        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError("--chunk-index must be a non-negative integer")
        chunks = document.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("--chunk-index is only valid for a chunked report")
        if chunk_index >= len(chunks):
            raise ValueError(
                f"chunk index {chunk_index} is outside the {len(chunks)} available chunks"
            )
        chunk = chunks[chunk_index]
        if not isinstance(chunk, dict):
            raise ValueError(f"chunk {chunk_index} must be a JSON object")
        output = chunk.get("output", chunk)
        codes, layout = _codes_from_mapping(output, "mlx_chunk")
        return codes, "mlx_chunk_time_major_frames" if layout == "mlx_chunk_time_major_frames" else layout

    if "chunks" in document and "frames" not in document:
        output = document.get("output")
        if not isinstance(output, dict) or not ("frames" in output or "codes" in output):
            raise ValueError("chunked report requires an explicit --chunk-index")

    if "frames" in document:
        return _codes_from_mapping(document, "native")
    if "output" in document:
        output = document["output"]
        codes, layout = _codes_from_mapping(output, "mlx")
        if layout == "mlx_time_major_frames":
            return codes, layout
        return codes, layout
    if "codes" in document or "generated_codes" in document:
        codes, _ = _codes_from_mapping(document, "book-major")
        return codes, "book_major_codes"
    raise ValueError("codes report must contain native frames or MLX output frames/codes")


def load_codes_json(path: Path, chunk_index=None):
    try:
        with path.open("r", encoding="utf-8") as source:
            document = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid codes JSON: {error}") from error
    codes, layout = parse_codes_document(document, chunk_index=chunk_index)
    return codes, layout


def require_new_output_dir(path: Path):
    if os.path.lexists(str(path)):
        raise ValueError(f"output directory must be new: {path}")
    if not path.parent.is_dir():
        raise ValueError(f"output directory parent must already exist: {path.parent}")


def inspect_upstream(mlx_root: Path, codec_path: Path):
    """Fail closed before importing MLX unless both immutable inputs match."""
    source = mlx_root / "mlx_audio/codec/models/fish_s1_dac/fish_s1_dac.py"
    if not source.is_file():
        raise ValueError(f"pinned MLX source file is missing: {source}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(mlx_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(mlx_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot verify MLX upstream checkout: {error}") from error
    if revision != EXPECTED_MLX_REVISION:
        raise ValueError(
            f"MLX revision mismatch: expected {EXPECTED_MLX_REVISION}, got {revision}"
        )
    if status.strip():
        raise ValueError("MLX upstream checkout must be clean before import")
    codec_sha = sha256_file(codec_path)
    if codec_sha != EXPECTED_CODEC_SHA256:
        raise ValueError(
            f"codec SHA-256 mismatch: expected {EXPECTED_CODEC_SHA256}, got {codec_sha}"
        )
    return {
        "revision": revision,
        "worktree_clean": True,
        "source_file": source.relative_to(mlx_root).as_posix(),
        "source_sha256": sha256_file(source),
        "codec_sha256": codec_sha,
    }


class _ActivationTracer:
    """Temporarily wrap upstream methods and retain only actual call outputs."""

    def __init__(self):
        self.values = {}
        self.layouts = {}
        self._targets = {}
        self._patches = []

    def _register(self, cls, method_name, target, stage_name, layout):
        key = (cls, method_name)
        target_map = self._targets.get(key)
        if target_map is None:
            original = getattr(cls, method_name)
            target_map = {}

            def wrapped(instance, *args, **kwargs):
                value = original(instance, *args, **kwargs)
                target = target_map.get(id(instance))
                if target is not None:
                    self._capture(target[0], value, target[1])
                return value

            setattr(cls, method_name, wrapped)
            self._targets[key] = target_map
            self._patches.append((cls, method_name, original))
        if id(target) in target_map:
            raise ValueError(f"duplicate traced upstream target for stage {stage_name}")
        target_map[id(target)] = (stage_name, layout)

    def add_module(self, module, stage_name, layout="nct"):
        self._register(type(module), "__call__", module, stage_name, layout)

    def add_method(self, obj, method_name, stage_name, layout):
        self._register(type(obj), method_name, obj, stage_name, layout)

    def _capture(self, stage_name, value, layout):
        if stage_name in self.values:
            raise ValueError(f"upstream stage was called more than once: {stage_name}")
        if not hasattr(value, "shape") or len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError(
                f"stage {stage_name} must return one rank-3 batch, got "
                f"{getattr(value, 'shape', None)}"
            )
        if layout not in ("nct", "ntc"):
            raise ValueError(f"unsupported stage layout {layout!r}")
        self.values[stage_name] = value
        self.layouts[stage_name] = layout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for cls, method_name, original in reversed(self._patches):
            setattr(cls, method_name, original)
        return False


def _expected_stage_names(model):
    names = []
    quantizer = model.quantizer
    semantic_books = quantizer.semantic_quantizer.quantizers
    residual_books = quantizer.quantizer.quantizers
    books = list(semantic_books) + list(residual_books)
    if len(books) != NUM_CODEBOOKS:
        raise ValueError(f"upstream codec exposes {len(books)} codebooks, expected 10")
    for book_index in range(NUM_CODEBOOKS):
        names.extend(
            [
                f"quantizer.book.{book_index}.embedding",
                f"quantizer.book.{book_index}.projection",
            ]
        )
    names.append("quantizer.codebook_sum")
    layers = getattr(quantizer.post_module, "layers", None)
    if not isinstance(layers, list) or len(layers) != 8:
        raise ValueError("upstream codec post_module must expose exactly 8 transformer layers")
    names.extend(f"quantizer.post_module.layers.{index}" for index in range(8))
    names.append("quantizer.post_module")
    for stage in range(len(quantizer.upsample)):
        names.extend(
            [
                f"quantizer.upsample.{stage}.deconv",
                f"quantizer.upsample.{stage}.convnext",
            ]
        )
    decoder = model.decoder
    names.append("decoder.model.0")
    for stage in range(len(decoder.model) - 4):
        block = decoder.model[stage + 1]
        for substage in range(len(block.block)):
            names.append(f"decoder.model.{stage + 1}.block.{substage}")
    final_index = len(decoder.model) - 3
    names.extend(
        [
            f"decoder.model.{final_index}",
            f"decoder.model.{final_index + 1}",
            "waveform",
        ]
    )
    if len(names) != EXPECTED_STAGE_COUNT:
        raise ValueError(
            f"upstream codec exposes {len(names)} trace stages, "
            f"expected {EXPECTED_STAGE_COUNT}"
        )
    return books, names


def _install_trace_wrappers(model, tracer):
    quantizer = model.quantizer
    books, expected_names = _expected_stage_names(model)
    for book_index, book in enumerate(books):
        tracer.add_method(book, "embed_code", f"quantizer.book.{book_index}.embedding", "ntc")
        tracer.add_module(book.out_proj, f"quantizer.book.{book_index}.projection", "nct")
    tracer.add_method(
        quantizer,
        "_codes_to_latents",
        "quantizer.codebook_sum",
        "nct",
    )
    for index, layer in enumerate(quantizer.post_module.layers):
        tracer.add_module(layer, f"quantizer.post_module.layers.{index}", "ntc")
    tracer.add_module(quantizer.post_module, "quantizer.post_module", "nct")
    for stage, block in enumerate(quantizer.upsample):
        tracer.add_module(block[0], f"quantizer.upsample.{stage}.deconv", "nct")
        tracer.add_module(block[1], f"quantizer.upsample.{stage}.convnext", "nct")
    decoder = model.decoder
    tracer.add_module(decoder.model[0], "decoder.model.0", "nct")
    for stage in range(len(decoder.model) - 4):
        block = decoder.model[stage + 1]
        for substage, layer in enumerate(block.block):
            tracer.add_module(
                layer,
                f"decoder.model.{stage + 1}.block.{substage}",
                "nct",
            )
    final_index = len(decoder.model) - 3
    tracer.add_module(decoder.model[final_index], f"decoder.model.{final_index}", "nct")
    tracer.add_module(
        decoder.model[final_index + 1], f"decoder.model.{final_index + 1}", "nct"
    )
    tracer.add_module(decoder.model[final_index + 2], "waveform", "nct")
    return expected_names


def _stage_time_major(value, layout, stage_name):
    if str(value.dtype).split(".")[-1] != "float32":
        raise ValueError(f"stage {stage_name} is not F32: {value.dtype}")
    if layout == "ntc":
        return value[0]
    return value[0].transpose(1, 0)


def _write_array_exclusive(path: Path, values):
    with path.open("xb") as output:
        values.tofile(output)


def _terminal_export(
    args,
    upstream,
    codes,
    input_layout,
    input_sha256,
    checkpoint_dtype_counts,
    model,
    mx,
    np,
):
    output_dir = args.output_dir
    output_dir.mkdir()
    code_count = len(codes[0])
    code_tensor = mx.array(np.asarray(codes, dtype=np.int32)[None, :, :])
    feature_lengths = mx.array([code_count], dtype=mx.int32)

    # Establish the normal upstream result separately. Evaluation materializes
    # the full output on GPU but does not copy activations to host.
    baseline_audio, _ = model.decode(code_tensor, feature_lengths)
    mx.eval(baseline_audio)

    tracer = _ActivationTracer()
    traced_audio = None
    trace_started = time.perf_counter()
    with tracer:
        expected_names = _install_trace_wrappers(model, tracer)
        traced_audio, _ = model.decode(code_tensor, feature_lengths)
    if list(tracer.values) != expected_names:
        missing = [name for name in expected_names if name not in tracer.values]
        extra = [name for name in tracer.values if name not in expected_names]
        raise ValueError(f"trace stage set/order mismatch; missing={missing}, extra={extra}")

    comparison = {
        "shape_equal": tuple(baseline_audio.shape) == tuple(traced_audio.shape),
        "dtype_equal": str(baseline_audio.dtype) == str(traced_audio.dtype),
    }
    if comparison["shape_equal"] and comparison["dtype_equal"]:
        difference = mx.abs(traced_audio - baseline_audio)
        equal = mx.all(traced_audio == baseline_audio)
        max_abs = mx.max(difference)
        mismatched = mx.sum((traced_audio != baseline_audio).astype(mx.int32))
        terminal_values = [equal, max_abs, mismatched]
    else:
        difference = None
        equal = None
        max_abs = None
        mismatched = None
        terminal_values = []

    # This is the sole activation export boundary: resolve every retained
    # stage/output first, then perform terminal GPU-to-host serialization.
    stage_views = {
        name: _stage_time_major(tracer.values[name], tracer.layouts[name], name)
        for name in expected_names
    }
    mx.eval(*stage_views.values(), traced_audio, *terminal_values)

    if comparison["shape_equal"] and comparison["dtype_equal"]:
        comparison.update(
            {
                "elementwise_exact": bool(equal.item()),
                "max_abs_error": float(max_abs.item()),
                "mismatched_values": int(mismatched.item()),
            }
        )
    else:
        comparison.update(
            {
                "elementwise_exact": False,
                "max_abs_error": None,
                "mismatched_values": None,
            }
        )
    comparison["status"] = "PASS" if comparison["elementwise_exact"] else "FAIL"
    comparison["official_decode_shape"] = list(baseline_audio.shape)
    comparison["traced_decode_shape"] = list(traced_audio.shape)
    traced_decode_and_gpu_comparison_seconds = time.perf_counter() - trace_started

    stage_entries = []
    waveform_array = None
    for index, name in enumerate(expected_names):
        values = np.asarray(stage_views[name], dtype=np.dtype("<f4"))
        if values.ndim != 2:
            raise ValueError(f"terminal stage {name} is not [time,channels]: {values.shape}")
        if name == "waveform":
            if values.shape[1] != 1:
                raise ValueError(f"waveform must be mono [samples,1], got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError("terminal waveform contains non-finite values")
            filename = "waveform.f32"
            waveform_array = values
        else:
            label = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
            filename = f"stage_{index:03d}_{label}.f32"
        destination = output_dir / filename
        _write_array_exclusive(destination, values)
        stage_entries.append(
            {
                "name": name,
                "rows": int(values.shape[0]),
                "channels": int(values.shape[1]),
                "count": int(values.size),
                "file": filename,
                "sha256": sha256_file(destination),
            }
        )
    if waveform_array is None:
        raise ValueError("trace did not export a waveform stage")

    clipped = np.clip(waveform_array[:, 0], np.float32(-1.0), np.float32(1.0))
    scaled = (clipped * np.float32(32767.0)).astype(np.float32)
    pcm = np.rint(scaled).astype("<i2")
    wav_path = output_dir / "reference.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    waveform_sha = sha256_file(output_dir / "waveform.f32")
    wav_sha = sha256_file(wav_path)

    report = {
        "format": "drama3-codec-cross-reference-v1",
        "input": {
            "path": str(args.codes_json.resolve()),
            "sha256": input_sha256,
            "layout": input_layout,
            "codebook_major": True,
            "codebook_count": NUM_CODEBOOKS,
            "frames": code_count,
            "chunk_index": args.chunk_index,
        },
        "upstream": {
            "repository": str(args.mlx_root.resolve()),
            "revision": upstream["revision"],
            "worktree_clean": upstream["worktree_clean"],
            "source_file": upstream["source_file"],
            "source_sha256": upstream["source_sha256"],
        },
        "codec": {
            "path": str(args.codec.resolve()),
            "sha256": upstream["codec_sha256"],
            "checkpoint_floating_dtype": "float32 (loaded without conversion)",
            "checkpoint_tensor_dtype_counts": checkpoint_dtype_counts,
            "official_build_ae_and_sanitize_load": True,
        },
        "decode": {
            "api": "official DAC.decode(indices, feature_lengths)",
            "sample_rate": SAMPLE_RATE,
            "frame_length_samples": FRAME_LENGTH,
            "traced_decode_and_gpu_comparison_seconds": traced_decode_and_gpu_comparison_seconds,
            "official_decode_comparison": comparison,
        },
        "host_readback": {
            "mid_decode_activation_readback": False,
            "activation_export": "single terminal export after full upstream decode",
            "terminal_gpu_comparison": True,
        },
        "frames": code_count,
        "codes": codes,
        "stages": stage_entries,
        "waveform": {
            "shape": [int(waveform_array.shape[0]), 1],
            "dtype": "float32",
            "layout": "samples,channels",
            "file": "waveform.f32",
            "sha256": waveform_sha,
            "wav_file": "reference.wav",
            "wav_sha256": wav_sha,
            "sample_rate": SAMPLE_RATE,
            "samples": int(waveform_array.shape[0]),
            "pcm16_conversion": "clip float32 to [-1,1], multiply float32 by 32767, ties-to-even round, little-endian signed PCM16",
        },
    }
    report_path = output_dir / "reference.json"
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return report


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--codes-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    args.codec = args.codec.expanduser().resolve()
    args.mlx_root = args.mlx_root.expanduser().resolve()
    args.codes_json = args.codes_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().absolute()
    try:
        if not args.codec.is_file():
            raise ValueError(f"codec checkpoint is not a file: {args.codec}")
        if not args.codes_json.is_file():
            raise ValueError(f"codes report is not a file: {args.codes_json}")
        require_new_output_dir(args.output_dir)
        codes, input_layout = load_codes_json(args.codes_json, args.chunk_index)
        upstream = inspect_upstream(args.mlx_root, args.codec)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    # Import only after the pinned source tree and codec bytes have passed the
    # preflight checks above.
    sys.path.insert(0, str(args.mlx_root))
    import mlx.core as mx
    import numpy as np
    from mlx_audio.codec.models.fish_s1_dac.fish_s1_dac import build_ae

    try:
        mx.set_default_device(mx.gpu)
        if mx.default_device() != mx.gpu:
            raise RuntimeError(f"failed to select MLX GPU: {mx.default_device()}")
        model = build_ae()
        original_weights = mx.load(str(args.codec))
        checkpoint_dtype_counts = {}
        for tensor in original_weights.values():
            dtype_name = str(tensor.dtype).split(".")[-1]
            checkpoint_dtype_counts[dtype_name] = checkpoint_dtype_counts.get(dtype_name, 0) + 1
            if ("float" in dtype_name or "bfloat" in dtype_name) and dtype_name != "float32":
                raise RuntimeError(
                    f"codec checkpoint floating dtype must remain original F32, got {dtype_name}"
                )
        weights = model.sanitize(original_weights)
        model.load_weights(list(weights.items()), strict=False)
        del weights
        del original_weights
        model.eval()
        mx.eval(model.parameters())
        if model.sample_rate != SAMPLE_RATE or model.frame_length != FRAME_LENGTH:
            raise RuntimeError(
                f"upstream codec geometry mismatch: sample_rate={model.sample_rate}, "
                f"frame_length={model.frame_length}"
            )
        report = _terminal_export(
            args,
            upstream,
            codes,
            input_layout,
            sha256_file(args.codes_json),
            checkpoint_dtype_counts,
            model,
            mx,
            np,
        )
    except Exception as error:
        raise SystemExit(f"codec cross-reference failed: {error}") from error
    print(
        json.dumps(
            {
                "format": report["format"],
                "output_dir": str(args.output_dir),
                "frames": report["frames"],
                "stage_count": len(report["stages"]),
                "official_decode_status": report["decode"]["official_decode_comparison"]["status"],
            },
            sort_keys=True,
        )
    )
    if report["decode"]["official_decode_comparison"]["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
