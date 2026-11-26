#!/usr/bin/env python3
"""Capture a pinned upstream MLX DAC reference-voice encoder trace.

The upstream DAC.encode implementation is left intact. A temporary observer
retains intermediate MLX arrays on device, compares a complete observed encode
against an unobserved encode, and exports flat time-major F32 files only after
the GPU equality gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any, Callable


SAMPLE_RATE = 44_100
FRAME_LENGTH = 2_048
MAX_SAMPLES = SAMPLE_RATE * 10
NUM_CODEBOOKS = 10


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pcm16_wav(path: Path) -> dict[str, Any]:
    """Read one strict mono PCM16 44.1 kHz WAV without changing its samples."""
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            compression = source.getcomptype()
            pcm_bytes = source.readframes(frame_count)
    except (OSError, wave.Error) as error:
        raise ValueError(f"cannot read reference WAV: {error}") from error

    if compression != "NONE":
        raise ValueError("reference WAV must use uncompressed PCM")
    if channels != 1:
        raise ValueError(f"reference WAV must be mono, got {channels} channels")
    if sample_width != 2:
        raise ValueError(f"reference WAV must be PCM 16-bit, got {sample_width * 8}-bit")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"reference WAV sample rate must be 44100 Hz, got {sample_rate}")
    if frame_count < 1 or not pcm_bytes:
        raise ValueError("reference WAV must contain non-empty audio")
    if frame_count > MAX_SAMPLES:
        raise ValueError("reference WAV must be no longer than 10 seconds")
    if len(pcm_bytes) != frame_count * 2:
        raise ValueError("reference WAV PCM byte count does not match its frame count")

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "samples": frame_count,
        "pcm_bytes": pcm_bytes,
        "wav_sha256": sha256_file(path),
        "pcm_sha256": sha256_bytes(pcm_bytes),
    }


def validate_transcript(transcript: str) -> str:
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("reference transcript must be non-empty")
    return transcript


def stage_to_time_major_f32(value: Any, layout: str, stage_name: str):
    """Convert a CPU value to the manifest's little-endian [time,channels] form."""
    import numpy as np

    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"stage {stage_name} contains non-finite values")
    if layout == "nct":
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(f"stage {stage_name} nct layout requires [1,channels,time], got {array.shape}")
        array = array[0].T
    elif layout == "ntc":
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(f"stage {stage_name} ntc layout requires [1,time,channels], got {array.shape}")
        array = array[0]
    elif layout != "tc" or array.ndim != 2:
        raise ValueError(f"stage {stage_name} has unsupported layout {layout!r} for shape {array.shape}")
    return np.asarray(array, dtype=np.dtype("<f4"), order="C")


def _stage_to_time_major_mx(mx: Any, value: Any, layout: str, stage_name: str):
    if layout == "nct":
        if len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError(f"stage {stage_name} nct layout requires [1,channels,time], got {value.shape}")
        return value[0].transpose(1, 0)
    if layout == "ntc":
        if len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError(f"stage {stage_name} ntc layout requires [1,time,channels], got {value.shape}")
        return value[0]
    if layout == "tc" and len(value.shape) == 2:
        return value
    raise ValueError(f"stage {stage_name} has unsupported layout {layout!r} for shape {value.shape}")


class _EncoderTrace:
    """Temporary class-method wrappers that retain selected MLX arrays."""

    def __init__(self, mx: Any):
        self.mx = mx
        self.values: dict[str, Any] = {}
        self.layouts: dict[str, str] = {}
        self._targets: dict[tuple[type, str], dict[int, tuple[Any, ...]]] = {}
        self._patches: list[tuple[type, str, Any]] = []

    def _register(
        self,
        obj: Any,
        method_name: str,
        *,
        output_name: str | None = None,
        output_layout: str | None = None,
        aliases: tuple[str, ...] = (),
        input_name: str | None = None,
        input_layout: str | None = None,
        observer: Callable[[Any, tuple[Any, ...], Any], None] | None = None,
    ) -> None:
        key = (type(obj), method_name)
        target_map = self._targets.get(key)
        if target_map is None:
            original = getattr(type(obj), method_name)
            target_map = {}

            def wrapped(instance, *args, **kwargs):
                target = target_map.get(id(instance))
                if target is not None and target[3] is not None:
                    if not args:
                        raise ValueError("traced module call has no first input")
                    self.capture(target[3], args[0], target[4])
                result = original(instance, *args, **kwargs)
                if target is not None:
                    callback = target[5]
                    if callback is not None:
                        callback(args, result)
                    if target[0] is not None:
                        self.capture(target[0], result, target[1])
                        for alias in target[2]:
                            self.capture(alias, result, target[1])
                return result

            setattr(type(obj), method_name, wrapped)
            self._targets[key] = target_map
            self._patches.append((type(obj), method_name, original))
        if id(obj) in target_map:
            raise ValueError(f"duplicate trace target for stage {output_name or input_name}")
        target_map[id(obj)] = (
            output_name,
            output_layout,
            aliases,
            input_name,
            input_layout,
            observer,
        )

    def add_module(
        self,
        module: Any,
        stage_name: str,
        layout: str = "nct",
        *,
        aliases: tuple[str, ...] = (),
        input_name: str | None = None,
        input_layout: str | None = None,
    ) -> None:
        self._register(
            module,
            "__call__",
            output_name=stage_name,
            output_layout=layout,
            aliases=aliases,
            input_name=input_name,
            input_layout=input_layout,
        )

    def add_method(
        self,
        obj: Any,
        method_name: str,
        *,
        stage_name: str | None = None,
        layout: str | None = None,
        observer: Callable[[tuple[Any, ...], Any], None] | None = None,
    ) -> None:
        self._register(
            obj,
            method_name,
            output_name=stage_name,
            output_layout=layout,
            observer=observer,
        )

    def capture(self, name: str, value: Any, layout: str) -> None:
        if name in self.values:
            raise ValueError(f"upstream stage was called more than once: {name}")
        if not hasattr(value, "shape"):
            raise ValueError(f"stage {name} did not return an array")
        if layout not in ("nct", "ntc", "tc"):
            raise ValueError(f"unsupported trace layout {layout!r}")
        self.values[name] = value
        self.layouts[name] = layout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for cls, method_name, original in reversed(self._patches):
            setattr(cls, method_name, original)
        return False


def _quantizer_book_trace(trace: _EncoderTrace, book: Any, label: str) -> None:
    trace.add_module(book.in_proj, f"{label}.in_proj", "nct")

    def observe(latents_and_kwargs: tuple[Any, ...], result: Any) -> None:
        latents = latents_and_kwargs[0]
        _, indices = result
        if len(latents.shape) != 3 or latents.shape[0] != 1:
            raise ValueError(f"{label} input must be [1,channels,time], got {latents.shape}")
        # Mirror the pinned upstream decode_latents distance formula on GPU for
        # diagnostic columns. The selected IDs themselves come from its result.
        encodings = latents.transpose(0, 2, 1).reshape((-1, latents.shape[1]))
        codebook = book.codebook.weight
        enc_norm = encodings / trace.mx.maximum(
            trace.mx.sqrt(trace.mx.sum(trace.mx.power(encodings, 2), axis=1, keepdims=True)),
            1e-12,
        )
        code_norm = codebook / trace.mx.maximum(
            trace.mx.sqrt(trace.mx.sum(trace.mx.power(codebook, 2), axis=1, keepdims=True)),
            1e-12,
        )
        distances = (
            trace.mx.sum(trace.mx.power(enc_norm, 2), axis=1, keepdims=True)
            - 2 * (enc_norm @ code_norm.T)
            + trace.mx.sum(trace.mx.power(code_norm, 2), axis=1, keepdims=True).T
        )
        two_smallest = trace.mx.sort(distances, axis=1)[:, :2]
        margin = two_smallest[:, 1:2] - two_smallest[:, 0:1]
        trace.capture(f"{label}.normalized_input", enc_norm, "tc")
        trace.capture(f"{label}.distance", distances, "tc")
        trace.capture(f"{label}.top2_margin", margin, "tc")
        trace.capture(f"{label}.selected_ids", indices[0].reshape((-1, 1)).astype(trace.mx.float32), "tc")

    trace.add_method(book, "decode_latents", observer=observe)


def _install_encoder_trace(model: Any, trace: _EncoderTrace) -> list[str]:
    expected: list[str] = ["audio.padded"]
    encoder = model.encoder
    for index, block in enumerate(encoder.block):
        if hasattr(block, "block") and isinstance(block.block, (list, tuple)):
            for inner_index, inner in enumerate(block.block):
                trace.add_module(inner, f"encoder.block.{index}.block.{inner_index}")
                expected.append(f"encoder.block.{index}.block.{inner_index}")
        trace.add_module(block, f"encoder.block.{index}")
        expected.append(f"encoder.block.{index}")
    trace.add_module(
        encoder,
        "encoder.output",
        input_name="audio.padded",
        input_layout="nct",
    )
    expected.append("encoder.output")

    quantizer = model.quantizer
    for index, (conv, convnext) in enumerate(quantizer.downsample):
        conv_name = f"quantizer.downsample.{index}.conv"
        convnext_name = f"quantizer.downsample.{index}.convnext"
        trace.add_module(conv, conv_name)
        trace.add_module(convnext, convnext_name)
        expected.extend([conv_name, convnext_name])

    pre = quantizer.pre_module
    trace.add_module(
        pre,
        "quantizer.pre_module",
        aliases=("quantizer.semantic.input",),
    )
    expected.extend(["quantizer.pre_module", "quantizer.semantic.input"])
    for name in ("input_proj", "look_ahead_conv"):
        module = getattr(pre, name, None)
        if module is not None:
            stage = f"quantizer.pre_module.{name}"
            trace.add_module(module, stage, "ntc")
            expected.append(stage)
    for index, layer in enumerate(getattr(pre, "layers", ())):
        stage = f"quantizer.pre_module.layer.{index}"
        trace.add_module(layer, stage, "ntc")
        expected.append(stage)
    norm = getattr(pre, "norm", None)
    if norm is not None:
        trace.add_module(norm, "quantizer.pre_module.norm", "ntc")
        expected.append("quantizer.pre_module.norm")
    output_proj = getattr(pre, "output_proj", None)
    if output_proj is not None:
        trace.add_module(output_proj, "quantizer.pre_module.output_proj", "ntc")
        expected.append("quantizer.pre_module.output_proj")

    groups = (
        ("quantizer.semantic", quantizer.semantic_quantizer.quantizers),
        ("quantizer.residual", quantizer.quantizer.quantizers),
    )
    for prefix, books in groups:
        for index, book in enumerate(books):
            label = f"{prefix}.{index}"
            _quantizer_book_trace(trace, book, label)
            expected.extend(
                [
                    f"{label}.in_proj",
                    f"{label}.normalized_input",
                    f"{label}.distance",
                    f"{label}.top2_margin",
                    f"{label}.selected_ids",
                ]
            )
    return expected


def _load_codec_helper():
    path = Path(__file__).resolve().with_name("codec_cross_reference.py")
    spec = importlib.util.spec_from_file_location("drama3_codec_cross_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load codec pin helper: {path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _checkpoint_dtype_counts(weights: dict[Any, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in weights.values():
        dtype_name = str(tensor.dtype).split(".")[-1]
        counts[dtype_name] = counts.get(dtype_name, 0) + 1
        if ("float" in dtype_name or "bfloat" in dtype_name) and dtype_name != "float32":
            raise ValueError(f"codec checkpoint floating dtype must remain original F32, got {dtype_name}")
    return counts


def _expected_stage_names(model: Any) -> list[str]:
    names = ["audio.padded"]
    for index, block in enumerate(model.encoder.block):
        if hasattr(block, "block") and isinstance(block.block, (list, tuple)):
            names.extend(f"encoder.block.{index}.block.{inner}" for inner in range(len(block.block)))
        names.append(f"encoder.block.{index}")
    names.append("encoder.output")
    for index in range(len(model.quantizer.downsample)):
        names.extend(
            [f"quantizer.downsample.{index}.conv", f"quantizer.downsample.{index}.convnext"]
        )
    names.extend(
        [
            "quantizer.pre_module",
            "quantizer.semantic.input",
            "quantizer.pre_module.input_proj",
            "quantizer.pre_module.look_ahead_conv",
        ]
    )
    names.extend(
        f"quantizer.pre_module.layer.{index}"
        for index in range(len(model.quantizer.pre_module.layers))
    )
    names.extend(["quantizer.pre_module.norm", "quantizer.pre_module.output_proj"])
    for prefix, books in (
        ("quantizer.semantic", model.quantizer.semantic_quantizer.quantizers),
        ("quantizer.residual", model.quantizer.quantizer.quantizers),
    ):
        for index in range(len(books)):
            label = f"{prefix}.{index}"
            names.extend(
                [
                    f"{label}.in_proj",
                    f"{label}.normalized_input",
                    f"{label}.distance",
                    f"{label}.top2_margin",
                    f"{label}.selected_ids",
                ]
            )
    names.append("final.codes")
    return names


def _write_stage(
    output_dir: Path,
    index: int,
    name: str,
    values: Any,
    np: Any,
    *,
    row_axis: str = "time",
    channel_axis: str | None = None,
) -> dict[str, Any]:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"terminal stage {name} must be finite [time,channels], got {values.shape}")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    filename = f"stage_{index:03d}_{safe_name}.f32"
    destination = output_dir / filename
    with destination.open("xb") as output:
        values.tofile(output)
    return {
        "name": name,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "row_axis": row_axis,
        "channel_axis": channel_axis or ("codebook" if name == "final.codes" else "features"),
        "dtype": "float32-le",
        "file": filename,
        "sha256": sha256_file(destination),
    }


def _terminal_export(
    *, args: argparse.Namespace, upstream: dict[str, Any], wav: dict[str, Any],
    transcript: str, checkpoint_dtype_counts: dict[str, int], model: Any,
    mx: Any, np: Any, input_audio: Any, prompt_builder: Any = None,
    tokenizer: Any = None, semantic_begin: int | None = None,
    codebook_size: int | None = None,
) -> dict[str, Any]:
    baseline_codes, baseline_lengths = model.encode(input_audio)

    trace = _EncoderTrace(mx)
    with trace:
        expected = _install_encoder_trace(model, trace)
        observed_codes, observed_lengths = model.encode(input_audio)
    trace.capture(
        "final.codes",
        observed_codes[0].transpose(1, 0).astype(mx.float32),
        "tc",
    )

    expected_names = _expected_stage_names(model)
    missing = sorted(set(expected_names) - set(trace.values))
    extra = sorted(set(trace.values) - set(expected_names))
    if missing or extra:
        raise ValueError(f"trace stage set mismatch; missing={missing}, extra={extra}")
    if (set(expected) | {"final.codes"}) != set(expected_names):
        raise ValueError("trace installer and expected stage names disagree")

    shape_equal = tuple(baseline_codes.shape) == tuple(observed_codes.shape)
    length_shape_equal = tuple(baseline_lengths.shape) == tuple(observed_lengths.shape)
    codes_equal = mx.all(baseline_codes == observed_codes) if shape_equal else mx.array(False)
    lengths_equal = mx.all(baseline_lengths == observed_lengths) if length_shape_equal else mx.array(False)
    mismatch_count = (
        mx.sum((baseline_codes != observed_codes).astype(mx.int32))
        if shape_equal
        else mx.array(1, dtype=mx.int32)
    )
    mx.eval(codes_equal, lengths_equal, mismatch_count)
    comparison = {
        "shape_equal": shape_equal,
        "feature_length_shape_equal": length_shape_equal,
        "codes_elementwise_exact": bool(codes_equal.item()),
        "feature_lengths_elementwise_exact": bool(lengths_equal.item()),
        "mismatched_code_values": int(mismatch_count.item()),
        "status": "PASS" if bool(codes_equal.item()) and bool(lengths_equal.item()) else "FAIL",
    }
    if comparison["status"] != "PASS":
        raise ValueError(f"observed-vs-unobserved GPU encode equality failed: {comparison}")

    time_major = {
        name: _stage_to_time_major_mx(mx, trace.values[name], trace.layouts[name], name)
        for name in expected_names
    }
    # This is the terminal GPU-to-host boundary: all codec calls and the full
    # code equality check have completed before any stage bytes are exported.
    mx.eval(*time_major.values(), observed_codes, observed_lengths)
    codes_np = np.asarray(observed_codes[0], dtype=np.int32)
    lengths_np = np.asarray(observed_lengths, dtype=np.int32)
    if codes_np.ndim != 2 or codes_np.shape[0] != NUM_CODEBOOKS:
        raise ValueError(f"upstream DAC returned invalid code shape {codes_np.shape}")
    frames = int(codes_np.shape[1])
    expected_frames = (wav["samples"] + FRAME_LENGTH - 1) // FRAME_LENGTH
    if frames != expected_frames:
        raise ValueError(f"upstream code frames {frames} do not match PCM geometry {expected_frames}")
    if lengths_np.reshape(-1).tolist() != [frames]:
        raise ValueError(f"upstream feature lengths do not match code frames: {lengths_np.tolist()}")
    if not np.issubdtype(codes_np.dtype, np.integer):
        raise ValueError(f"upstream code IDs are not integers: {codes_np.dtype}")
    for book_index, limit in enumerate((4096,) + (1024,) * 9):
        row = codes_np[book_index]
        if row.size and (int(row.min()) < 0 or int(row.max()) >= limit):
            raise ValueError(f"upstream code IDs for book {book_index} are out of range")

    args.output_dir.mkdir()
    stages = []
    for index, name in enumerate(expected_names):
        values = np.asarray(time_major[name], dtype=np.dtype("<f4"), order="C")
        if not np.isfinite(values).all():
            raise ValueError(f"terminal stage {name} contains non-finite values")
        if name.endswith(".selected_ids"):
            if not np.array_equal(values, np.rint(values)):
                raise ValueError(f"selected IDs for {name} are not exact F32 integers")
        stages.append(_write_stage(args.output_dir, index, name, values, np))

    codes_sha = sha256_bytes(np.asarray(codes_np, dtype="<i4", order="C").tobytes())
    prompt_entry = None
    if args.text is not None:
        if prompt_builder is None or tokenizer is None or semantic_begin is None or codebook_size is None:
            raise ValueError("--text prompt capture is missing the official tokenizer or model metadata")
        prompt_rows = prompt_builder(
            args.text,
            tokenizer,
            num_codebooks=NUM_CODEBOOKS,
            codebook_size=codebook_size,
            semantic_start=semantic_begin,
            prompt_texts=[transcript],
            prompt_tokens=[codes_np.astype(np.int64, copy=False)],
        )
        prompt_f32 = np.asarray(prompt_rows, dtype=np.dtype("<f4"), order="C")
        prompt_entry = _write_stage(
            args.output_dir,
            len(stages),
            "prompt.rows",
            prompt_f32,
            np,
            row_axis="prompt_row",
            channel_axis="main_plus_10_codebooks",
        )
        prompt_entry["official_builder"] = "mlx_audio.tts.models.fish_omni.prompt.build_prompt"
        prompt_entry["target_text_utf8_sha256"] = sha256_bytes(args.text.encode("utf-8"))
        prompt_entry["transcript_utf8_sha256"] = sha256_bytes(transcript.encode("utf-8"))
        prompt_entry["reference_codes_sha256"] = codes_sha

    manifest = {
        "format": "drama3-reference-voice-manifest-v1",
        "input": {
            "wav_path": str(args.wav.resolve()),
            "wav_sha256": wav["wav_sha256"],
            "pcm_sha256": wav["pcm_sha256"],
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "sample_width_bits": 16,
            "samples": int(wav["samples"]),
            "duration_seconds": wav["samples"] / SAMPLE_RATE,
            "pcm16_normalization": "upload exact signed little-endian int16 values as F32; divide by 32768.0 on the MLX GPU",
            "transcript_utf8_sha256": sha256_bytes(transcript.encode("utf-8")),
            "target_text_utf8_sha256": (
                sha256_bytes(args.text.encode("utf-8")) if args.text is not None else None
            ),
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
        "encode": {
            "api": "official DAC.encode(audio)",
            "unobserved_vs_observed_gpu_code_comparison": comparison,
            "feature_lengths": lengths_np.reshape(-1).tolist(),
            "audio_frame_length_samples": FRAME_LENGTH,
            "host_readback": "one terminal export after full observed encode and exact GPU code comparison",
        },
        "codes": {
            "shape": [NUM_CODEBOOKS, frames],
            "layout": "codebook-major",
            "dtype": "int32",
            "sha256": codes_sha,
            "source": "final.codes stage, exact integer IDs represented as F32",
        },
        "stages": stages,
        "prompt": prompt_entry,
        "stage_files": "flat row-major little-endian F32; each stage shape is [time,channels]",
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True, help="Pinned F32 codec.safetensors checkpoint")
    parser.add_argument("--mlx-root", type=Path, required=True, help="Pinned clean mlx-audio checkout")
    parser.add_argument("--wav", type=Path, required=True, help="Canonical mono PCM16 44.1 kHz WAV")
    parser.add_argument("--transcript", required=True, help="Exact transcript for the reference audio")
    parser.add_argument(
        "--text",
        help="Optional target text; builds and exports the official reference-conditioned prompt",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for manifest and stage files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.codec = args.codec.expanduser().resolve()
    args.mlx_root = args.mlx_root.expanduser().resolve()
    args.wav = args.wav.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().absolute()
    try:
        transcript = validate_transcript(args.transcript)
        if args.text is not None and not args.text.strip():
            raise ValueError("target text must be non-empty")
        if not args.codec.is_file():
            raise ValueError(f"codec checkpoint is not a file: {args.codec}")
        if not args.wav.is_file():
            raise ValueError(f"reference WAV is not a file: {args.wav}")
        if args.output_dir.exists():
            raise ValueError(f"output directory must be new: {args.output_dir}")
        if not args.output_dir.parent.is_dir():
            raise ValueError(f"output directory parent must already exist: {args.output_dir.parent}")
        wav = read_pcm16_wav(args.wav)
        helper = _load_codec_helper()
        upstream = helper.inspect_upstream(args.mlx_root, args.codec)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    # Import MLX/upstream only after all immutable-input and WAV checks pass.
    sys.path.insert(0, str(args.mlx_root))
    import mlx.core as mx
    import numpy as np
    from mlx_audio.codec.models.fish_s1_dac.fish_s1_dac import build_ae
    from mlx_audio.tts.models.fish_omni.prompt import validate_text

    try:
        validate_text(transcript, "reference 1")
        if args.text is not None:
            validate_text(args.text, "text")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    prompt_builder = None
    tokenizer = None
    semantic_begin = None
    codebook_size = None
    if args.text is not None:
        model_dir = args.codec.parent
        config_path = model_dir / "config.json"
        tokenizer_path = model_dir / "tokenizer.json"
        if not config_path.is_file() or not tokenizer_path.is_file():
            raise SystemExit(
                "--text requires config.json and tokenizer.json beside the codec checkpoint"
            )
        try:
            model_config = json.loads(config_path.read_bytes())
            semantic_begin = int(model_config["semantic_start_token_id"])
            codebook_size = int(model_config["audio_decoder_config"]["vocab_size"])
            if model_config.get("model_type") != "fish_omni" or codebook_size != 4096:
                raise ValueError("prompt checkpoint must be a compatible Fish Omni model")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid prompt model config: {error}") from error
        from tokenizers import Tokenizer
        from mlx_audio.tts.models.fish_omni.prompt import build_prompt as prompt_builder

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    try:
        mx.set_default_device(mx.gpu)
        if mx.default_device() != mx.gpu:
            raise RuntimeError(f"failed to select MLX GPU: {mx.default_device()}")
        model = build_ae()
        original_weights = mx.load(str(args.codec))
        dtype_counts = _checkpoint_dtype_counts(original_weights)
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
        pcm_i16 = np.frombuffer(wav["pcm_bytes"], dtype=np.dtype("<i2"))
        pcm_f32_upload = pcm_i16.astype(np.float32)
        input_audio = mx.array(pcm_f32_upload[None, None, :], dtype=mx.float32) / np.float32(32768.0)
        manifest = _terminal_export(
            args=args,
            upstream=upstream,
            wav=wav,
            transcript=transcript,
            checkpoint_dtype_counts=dtype_counts,
            model=model,
            mx=mx,
            np=np,
            input_audio=input_audio,
            prompt_builder=prompt_builder,
            tokenizer=tokenizer,
            semantic_begin=semantic_begin,
            codebook_size=codebook_size,
        )
    except Exception as error:
        raise SystemExit(f"reference voice encoder capture failed: {error}") from error

    print(
        json.dumps(
            {
                "format": manifest["format"],
                "output_dir": str(args.output_dir),
                "frames": manifest["codes"]["shape"][1],
                "stage_count": len(manifest["stages"]),
                "encode_comparison": manifest["encode"]["unobserved_vs_observed_gpu_code_comparison"]["status"],
                "manifest": str(args.output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
