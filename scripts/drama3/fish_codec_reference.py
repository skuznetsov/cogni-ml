#!/usr/bin/env python3
"""Export full-code activation references from the pinned original Fish DAC."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path


EXPECTED_FISH_REVISION = "541566932abbaebcb0a40e562a1f846738383f38"
EXPECTED_CODEC_SHA256 = "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"
EXPECTED_DAC_PACKAGE_VERSION = "1.0.0"
NUM_CODEBOOKS = 10
EXPECTED_STAGE_COUNT = 58
SAMPLE_RATE = 44_100
FRAME_LENGTH = 2_048
CODEBOOK_LIMITS = (4_096,) + (1_024,) * 9
FISH_SOURCE_FILE = "fish_speech/models/dac/inference.py"
DAC_MODEL_SOURCE_FILE = "fish_speech/models/dac/modded_dac.py"
RVQ_SOURCE_FILE = "fish_speech/models/dac/rvq.py"
CONFIG_FILE = "fish_speech/configs/modded_dac_vq.yaml"


def _load_common():
    path = Path(__file__).with_name("codec_cross_reference.py")
    spec = importlib.util.spec_from_file_location("_drama3_codec_cross_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared codec reference helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COMMON = _load_common()
sha256_file = _COMMON.sha256_file
validate_book_major_codes = _COMMON.validate_book_major_codes
parse_codes_document = _COMMON.parse_codes_document
load_codes_json = _COMMON.load_codes_json
require_new_output_dir = _COMMON.require_new_output_dir


def validate_pin_metadata(revision, worktree_status, codec_sha256):
    """Fail closed on a moved/dirty Fish checkout or a different checkpoint."""
    if revision != EXPECTED_FISH_REVISION:
        raise ValueError(
            f"Fish revision mismatch: expected {EXPECTED_FISH_REVISION}, got {revision}"
        )
    if worktree_status.strip():
        raise ValueError("Fish upstream checkout must be clean before import")
    if codec_sha256 != EXPECTED_CODEC_SHA256:
        raise ValueError(
            f"codec SHA-256 mismatch: expected {EXPECTED_CODEC_SHA256}, got {codec_sha256}"
        )


def inspect_upstream(fish_root: Path, codec_path: Path):
    """Verify the pinned source, clean worktree, config, and codec before imports."""
    source = fish_root / FISH_SOURCE_FILE
    dac_model_source = fish_root / DAC_MODEL_SOURCE_FILE
    rvq_source = fish_root / RVQ_SOURCE_FILE
    config = fish_root / CONFIG_FILE
    for path in (source, dac_model_source, rvq_source, config):
        if not path.is_file():
            raise ValueError(f"pinned Fish input is missing: {path}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(fish_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(fish_root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot verify Fish upstream checkout: {error}") from error
    codec_sha = sha256_file(codec_path)
    validate_pin_metadata(revision, status, codec_sha)
    return {
        "revision": revision,
        "worktree_clean": True,
        "source_file": FISH_SOURCE_FILE,
        "source_sha256": sha256_file(source),
        "source_files": [
            {"file": FISH_SOURCE_FILE, "sha256": sha256_file(source)},
            {"file": DAC_MODEL_SOURCE_FILE, "sha256": sha256_file(dac_model_source)},
            {"file": RVQ_SOURCE_FILE, "sha256": sha256_file(rvq_source)},
        ],
        "config_file": CONFIG_FILE,
        "config_sha256": sha256_file(config),
        "codec_sha256": codec_sha,
    }


def _state_dict_mismatches(expected, supplied):
    """Return strict key and shape mismatches without changing either mapping."""
    expected_keys = set(expected)
    supplied_keys = set(supplied)
    missing = sorted(expected_keys - supplied_keys)
    unexpected = sorted(supplied_keys - expected_keys)
    shape_mismatches = []
    for name in sorted(expected_keys & supplied_keys):
        expected_shape = tuple(expected[name].shape)
        supplied_shape = tuple(supplied[name].shape)
        if expected_shape != supplied_shape:
            shape_mismatches.append(
                {
                    "name": name,
                    "expected": list(expected_shape),
                    "supplied": list(supplied_shape),
                }
            )
    return {"missing": missing, "unexpected": unexpected, "shape_mismatches": shape_mismatches}


def _module_sequence(value, description):
    try:
        sequence = list(value)
    except TypeError as error:
        raise ValueError(f"upstream codec {description} is not iterable") from error
    return sequence


def _device_matches(actual, requested):
    """Compare MPS's optional zero index while keeping CUDA index explicit."""
    if actual.type != requested.type:
        return False
    if requested.index is not None:
        return actual.index == requested.index
    return actual.index in (None, 0)


def _expected_stage_names(model):
    """Validate the original DAC graph and return its 58 ordered trace stages."""
    quantizer = model.quantizer
    semantic_books = _module_sequence(
        quantizer.semantic_quantizer.quantizers, "semantic codebooks"
    )
    residual_books = _module_sequence(quantizer.quantizer.quantizers, "residual codebooks")
    books = semantic_books + residual_books
    if len(books) != NUM_CODEBOOKS:
        raise ValueError(f"upstream codec exposes {len(books)} codebooks, expected 10")
    for index, book in enumerate(books):
        if not hasattr(book, "codebook") or not hasattr(book, "out_proj"):
            raise ValueError(f"upstream codebook {index} lacks codebook or out_proj")
    names = []
    for index in range(NUM_CODEBOOKS):
        names.extend(
            [f"quantizer.book.{index}.embedding", f"quantizer.book.{index}.projection"]
        )
    names.append("quantizer.codebook_sum")
    layers = _module_sequence(quantizer.post_module.layers, "post_module.layers")
    if len(layers) != 8:
        raise ValueError("upstream codec post_module must expose exactly 8 transformer layers")
    names.extend(f"quantizer.post_module.layers.{index}" for index in range(8))
    names.append("quantizer.post_module")
    upsample = _module_sequence(quantizer.upsample, "quantizer.upsample")
    if len(upsample) != 2:
        raise ValueError(f"upstream codec exposes {len(upsample)} upsample blocks, expected 2")
    for stage, block in enumerate(upsample):
        if len(block) != 2:
            raise ValueError(f"upstream upsample block {stage} must have two modules")
        names.extend(
            [f"quantizer.upsample.{stage}.deconv", f"quantizer.upsample.{stage}.convnext"]
        )
    decoder = model.decoder
    decoder_layers = _module_sequence(decoder.model, "decoder.model")
    if len(decoder_layers) != 8:
        raise ValueError(f"upstream decoder exposes {len(decoder_layers)} layers, expected 8")
    names.append("decoder.model.0")
    for stage in range(4):
        block = decoder_layers[stage + 1]
        block_layers = _module_sequence(block.block, f"decoder block {stage}")
        if len(block_layers) != 5:
            raise ValueError(f"upstream decoder block {stage} must expose five stages")
        names.extend(
            f"decoder.model.{stage + 1}.block.{substage}"
            for substage in range(len(block_layers))
        )
    names.extend(["decoder.model.5", "decoder.model.6", "waveform"])
    if len(names) != EXPECTED_STAGE_COUNT:
        raise ValueError(f"upstream codec exposes {len(names)} trace stages, expected 58")
    return books, names


class _FishTrace:
    """Capture actual outputs from a single official Fish DAC decode call."""

    def __init__(self):
        self.values = {}
        self.layouts = {}
        self.handles = []
        self.restorers = []

    def capture(self, name, value, layout):
        if name in self.values:
            raise ValueError(f"upstream stage was called more than once: {name}")
        if not hasattr(value, "shape") or len(value.shape) != 3 or value.shape[0] != 1:
            raise ValueError(
                f"stage {name} must return one rank-3 batch, got "
                f"{getattr(value, 'shape', None)}"
            )
        if layout not in ("nct", "ntc"):
            raise ValueError(f"unsupported stage layout {layout!r}")
        self.values[name] = value.detach().clone()
        self.layouts[name] = layout

    def add_module(self, module, name, layout="nct"):
        def hook(_module, _inputs, output):
            self.capture(name, output, layout)

        self.handles.append(module.register_forward_hook(hook))

    def add_pre_input(self, module, name, layout="nct"):
        def hook(_module, inputs):
            if not inputs:
                raise ValueError(f"stage {name} received no input")
            self.capture(name, inputs[0], layout)

        self.handles.append(module.register_forward_pre_hook(hook))

    def add_method_capture(self, module, attribute, name, layout):
        """Wrap one upstream instance method without patching a shared function."""
        original = getattr(module, attribute)
        namespace = module.__dict__
        had_instance_value = attribute in namespace
        previous_instance_value = namespace.get(attribute)

        def traced_method(*args, **kwargs):
            result = original(*args, **kwargs)
            self.capture(name, result, layout)
            return result

        setattr(module, attribute, traced_method)
        self.restorers.append(
            (module, attribute, had_instance_value, previous_instance_value)
        )

    def install(self, model):
        books, expected_names = _expected_stage_names(model)
        for index, book in enumerate(books):
            self.add_method_capture(
                book,
                "embed_code",
                f"quantizer.book.{index}.embedding",
                "ntc",
            )
            self.add_module(book.out_proj, f"quantizer.book.{index}.projection", "nct")
        quantizer = model.quantizer
        self.add_pre_input(quantizer.post_module, "quantizer.codebook_sum", "nct")
        for index, layer in enumerate(quantizer.post_module.layers):
            self.add_module(layer, f"quantizer.post_module.layers.{index}", "ntc")
        self.add_module(quantizer.post_module, "quantizer.post_module", "nct")
        for stage, block in enumerate(quantizer.upsample):
            self.add_module(block[0], f"quantizer.upsample.{stage}.deconv", "nct")
            self.add_module(block[1], f"quantizer.upsample.{stage}.convnext", "nct")
        decoder = model.decoder
        self.add_module(decoder.model[0], "decoder.model.0", "nct")
        for stage in range(4):
            for substage, layer in enumerate(decoder.model[stage + 1].block):
                self.add_module(
                    layer, f"decoder.model.{stage + 1}.block.{substage}", "nct"
                )
        self.add_module(decoder.model[5], "decoder.model.5", "nct")
        self.add_module(decoder.model[6], "decoder.model.6", "nct")
        self.add_module(decoder.model[7], "waveform", "nct")
        return expected_names

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, attribute, had_instance_value, previous_instance_value in reversed(
            self.restorers
        ):
            if had_instance_value:
                setattr(module, attribute, previous_instance_value)
            else:
                delattr(module, attribute)
        self.restorers.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def _new_mps_guard(torch, target_device):
    """Reject host tensors and scalar extraction inside the traced model call."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class DeviceOnlyMode(TorchDispatchMode):
        def _walk(self, value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from self._walk(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    yield from self._walk(item)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(getattr(func, "_schema", func))
            if "_local_scalar_dense" in name:
                raise RuntimeError(f"host scalar extraction during decode: {name}")
            for tensor in self._walk((args, kwargs)):
                if not _device_matches(tensor.device, target_device):
                    raise RuntimeError(
                        f"non-{target_device} tensor entered decode op {name}: {tensor.device}"
                    )
            result = func(*args, **kwargs)
            for tensor in self._walk(result):
                if not _device_matches(tensor.device, target_device):
                    raise RuntimeError(
                        f"decode op {name} produced non-{target_device} tensor: {tensor.device}"
                    )
            return result

    return DeviceOnlyMode()


def _dtype_counts(tensors):
    counts = {}
    for tensor in tensors:
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _checkpoint_state_dict(torch, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("codec checkpoint must contain a state dictionary")
    if "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("codec checkpoint state_dict must be a mapping")
    if any("generator" in key for key in state):
        state = {
            key.replace("generator.", ""): value
            for key, value in state.items()
            if "generator." in key
        }
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("codec checkpoint state dictionary contains non-tensor values")
    return state


def _nonpersistent_buffer_names(model):
    return {
        f"{name}.{buffer}" if name else buffer
        for name, module in model.named_modules()
        for buffer in module._non_persistent_buffers_set
    }


def _load_model(args, fish_root, torch):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = fish_root / CONFIG_FILE
    model = instantiate(OmegaConf.load(str(config_path)))
    state = _checkpoint_state_dict(torch, args.codec)
    nonpersistent = _nonpersistent_buffer_names(model)
    state = {key: value for key, value in state.items() if key not in nonpersistent}
    expected = model.state_dict()
    mismatches = _state_dict_mismatches(expected, state)
    if any(mismatches.values()):
        raise RuntimeError(
            "strict codec checkpoint mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    float_dtypes = sorted(
        {
            str(value.dtype).removeprefix("torch.")
            for value in state.values()
            if value.is_floating_point()
        }
    )
    if float_dtypes != ["float32"]:
        raise RuntimeError(f"checkpoint floating tensors must remain F32, got {float_dtypes}")
    load_result = model.load_state_dict(state, strict=True, assign=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "strict codec load returned missing/unexpected keys: "
            f"{load_result.missing_keys}, {load_result.unexpected_keys}"
        )
    model.eval()
    # Keep the original parametrized weight-norm graph. assign=True preserves
    # checkpoint F32 parameters while leaving constructed BF16 buffers intact.
    model.to(device=args.device)
    parameter_dtypes = _dtype_counts(parameter for parameter in model.parameters())
    buffer_dtypes = _dtype_counts(buffer for buffer in model.buffers())
    misplaced = [
        str(tensor.device)
        for tensor in list(model.parameters()) + list(model.buffers())
        if not _device_matches(tensor.device, torch.device(args.device))
    ]
    if misplaced:
        raise RuntimeError(f"model parameters/buffers are not all on {args.device}: {misplaced[:8]}")
    return model, _dtype_counts(state.values()), parameter_dtypes, buffer_dtypes


def _capture_manifest(args, codes, input_layout, input_sha256, upstream, checkpoint_dtype_counts,
                      parameter_dtype_counts, buffer_dtype_counts, quantize_source):
    stages = []
    waveform = None
    terminal_arrays = {}
    for index, name in enumerate(args.expected_names):
        tensor = args.trace.values[name]
        if str(tensor.dtype).removeprefix("torch.") != "float32":
            raise RuntimeError(f"stage {name} is not F32: {tensor.dtype}")
        if args.trace.layouts[name] == "ntc":
            terminal = tensor[0].contiguous()
        else:
            terminal = tensor[0].transpose(0, 1).contiguous()
        if len(terminal.shape) != 2:
            raise RuntimeError(f"stage {name} is not [time,channels]: {tuple(terminal.shape)}")
        terminal_arrays[name] = terminal
    # The first host activation conversion occurs here, after both full decodes
    # and the terminal GPU equality check have completed.
    args.output_dir.mkdir()
    for index, name in enumerate(args.expected_names):
        values = terminal_arrays[name].detach().cpu().numpy().astype("<f4", copy=False)
        if values.ndim != 2:
            raise RuntimeError(f"terminal stage {name} is not [time,channels]")
        if name == "waveform":
            if values.shape[1] != 1:
                raise RuntimeError(f"waveform must be mono [samples,1], got {values.shape}")
            if not args.numpy.isfinite(values).all():
                raise RuntimeError("terminal waveform contains non-finite values")
            filename = "waveform.f32"
            waveform = values
        else:
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
            filename = f"stage_{index:03d}_{safe_name}.f32"
        destination = args.output_dir / filename
        with destination.open("xb") as output:
            values.tofile(output)
        stages.append(
            {
                "name": name,
                "rows": int(values.shape[0]),
                "channels": int(values.shape[1]),
                "count": int(values.size),
                "file": filename,
                "sha256": sha256_file(destination),
            }
        )
    if waveform is None:
        raise RuntimeError("trace did not export a waveform stage")
    clipped = args.numpy.clip(waveform[:, 0], args.numpy.float32(-1), args.numpy.float32(1))
    scaled = (clipped * args.numpy.float32(32767)).astype(args.numpy.float32)
    pcm = args.numpy.rint(scaled).astype("<i2")
    wav_path = args.output_dir / "reference.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    waveform_path = args.output_dir / "waveform.f32"
    comparison = args.comparison
    report = {
        "format": "drama3-fish-codec-cross-reference-v1",
        "input": {
            "path": str(args.codes_json.resolve()),
            "sha256": input_sha256,
            "layout": input_layout,
            "codebook_major": True,
            "codebook_count": NUM_CODEBOOKS,
            "frames": len(codes[0]),
            "chunk_index": args.chunk_index,
        },
        "upstream": {
            "repository": str(args.fish_root.resolve()),
            "revision": upstream["revision"],
            "worktree_clean": upstream["worktree_clean"],
            "source_file": upstream["source_file"],
            "source_sha256": upstream["source_sha256"],
            "source_files": upstream["source_files"],
            "loader_api": "fish_speech.models.dac.inference.load_model",
            "loader_state_assign": True,
            "loader_precision_override": "omitted to preserve checkpoint F32 and model buffer dtypes",
            "config_file": upstream["config_file"],
            "config_sha256": upstream["config_sha256"],
        },
        "codec": {
            "path": str(args.codec.resolve()),
            "sha256": upstream["codec_sha256"],
            "implementation": "Fish DAC with descript-audio-codec",
            "package_version": EXPECTED_DAC_PACKAGE_VERSION,
            "dac_source_file": quantize_source["file"],
            "dac_source_sha256": quantize_source["sha256"],
            "checkpoint_floating_dtype": "float32 (loaded without conversion)",
            "checkpoint_tensor_dtype_counts": checkpoint_dtype_counts,
            "parameter_dtype_counts_after_load": parameter_dtype_counts,
            "buffer_dtype_counts_after_load": buffer_dtype_counts,
            "strict_state_dict": True,
            "state_dict_assign": True,
            "weight_norm_folded": False,
            "weight_norm_preserved": True,
        },
        "decode": {
            "api": "official Fish DAC.from_indices(indices)",
            "device": str(args.device),
            "sample_rate": SAMPLE_RATE,
            "frame_length_samples": FRAME_LENGTH,
            "decode_and_terminal_comparison_seconds": args.decode_seconds,
            "official_decode_comparison": comparison,
        },
        "host_readback": {
            "mid_decode_activation_readback": False,
            "mid_decode_scalar_readback": False,
            "activation_export": "single terminal export after full upstream decode",
            "terminal_gpu_comparison": True,
        },
        "frames": len(codes[0]),
        "codes": codes,
        "stages": stages,
        "waveform": {
            "shape": [int(waveform.shape[0]), 1],
            "dtype": "float32",
            "layout": "samples,channels",
            "file": "waveform.f32",
            "sha256": sha256_file(waveform_path),
            "wav_file": "reference.wav",
            "wav_sha256": sha256_file(wav_path),
            "sample_rate": SAMPLE_RATE,
            "samples": int(waveform.shape[0]),
            "pcm16_conversion": "clip float32 to [-1,1], multiply float32 by 32767, ties-to-even round, little-endian signed PCM16",
        },
    }
    report_path = args.output_dir / "reference.json"
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return report


def _write_failure_report(args, codes, input_layout, input_sha256, upstream,
                          checkpoint_dtype_counts, parameter_dtype_counts,
                          buffer_dtype_counts, quantize_source, comparison):
    """Persist a machine-readable FAIL without exporting mismatching fixtures."""
    args.output_dir.mkdir()
    report = {
        "format": "drama3-fish-codec-cross-reference-v1",
        "status": "FAIL",
        "input": {
            "path": str(args.codes_json.resolve()),
            "sha256": input_sha256,
            "layout": input_layout,
            "codebook_major": True,
            "codebook_count": NUM_CODEBOOKS,
            "frames": len(codes[0]),
            "chunk_index": args.chunk_index,
        },
        "upstream": {
            "repository": str(args.fish_root.resolve()),
            "revision": upstream["revision"],
            "worktree_clean": upstream["worktree_clean"],
            "source_file": upstream["source_file"],
            "source_sha256": upstream["source_sha256"],
            "source_files": upstream["source_files"],
            "loader_api": "fish_speech.models.dac.inference.load_model",
            "loader_state_assign": True,
            "loader_precision_override": "omitted to preserve checkpoint F32 and model buffer dtypes",
            "config_file": upstream["config_file"],
            "config_sha256": upstream["config_sha256"],
        },
        "codec": {
            "path": str(args.codec.resolve()),
            "sha256": upstream["codec_sha256"],
            "implementation": "Fish DAC with descript-audio-codec",
            "package_version": EXPECTED_DAC_PACKAGE_VERSION,
            "dac_source_file": quantize_source["file"],
            "dac_source_sha256": quantize_source["sha256"],
            "checkpoint_tensor_dtype_counts": checkpoint_dtype_counts,
            "parameter_dtype_counts_after_load": parameter_dtype_counts,
            "buffer_dtype_counts_after_load": buffer_dtype_counts,
            "strict_state_dict": True,
            "state_dict_assign": True,
            "weight_norm_folded": False,
            "weight_norm_preserved": True,
        },
        "decode": {
            "api": "official Fish DAC.from_indices(indices)",
            "device": str(args.device),
            "sample_rate": SAMPLE_RATE,
            "frame_length_samples": FRAME_LENGTH,
            "decode_and_terminal_comparison_seconds": args.decode_seconds,
            "official_decode_comparison": comparison,
        },
        "host_readback": {
            "mid_decode_activation_readback": False,
            "mid_decode_scalar_readback": False,
            "activation_export": "not exported because official decode comparison failed",
            "terminal_gpu_comparison": True,
        },
        "frames": len(codes[0]),
        "codes": codes,
        "stages": [],
    }
    report_path = args.output_dir / "reference.json"
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fish-root", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codes-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    args.fish_root = args.fish_root.expanduser().resolve()
    args.codec = args.codec.expanduser().resolve()
    args.codes_json = args.codes_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().absolute()
    try:
        if not args.codec.is_file():
            raise ValueError(f"codec checkpoint is not a file: {args.codec}")
        if not args.codes_json.is_file():
            raise ValueError(f"codes report is not a file: {args.codes_json}")
        require_new_output_dir(args.output_dir)
        input_sha256 = sha256_file(args.codes_json)
        codes, input_layout = load_codes_json(args.codes_json, args.chunk_index)
        upstream = inspect_upstream(args.fish_root, args.codec)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    sys.path.insert(0, str(args.fish_root))
    import torch
    import numpy as np
    from importlib.metadata import version

    try:
        package_version = version("descript-audio-codec")
    except Exception as error:
        raise SystemExit(f"cannot identify descript-audio-codec version: {error}") from error
    if package_version != EXPECTED_DAC_PACKAGE_VERSION:
        raise SystemExit(
            f"descript-audio-codec version mismatch: expected "
            f"{EXPECTED_DAC_PACKAGE_VERSION}, got {package_version}"
        )
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS is unavailable; refusing a CPU fallback")
    elif not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing a CPU fallback")
    elif args.device == "cuda":
        torch.cuda.set_device(0)
        args.device = torch.device("cuda", 0)
    if args.device == "mps":
        args.device = torch.device("mps")
    args.torch = torch
    args.numpy = np

    import dac.nn.quantize as quantize_module
    from fish_speech.models.dac.modded_dac import DAC

    if not issubclass(DAC, torch.nn.Module):
        raise SystemExit("pinned Fish DAC class is not a torch.nn.Module")
    try:
        model, checkpoint_dtype_counts, parameter_dtype_counts, buffer_dtype_counts = _load_model(
            args, args.fish_root, torch
        )
    except Exception as error:
        raise SystemExit(f"cannot load pinned Fish DAC checkpoint: {error}") from error
    if model.sample_rate != SAMPLE_RATE or model.frame_length != FRAME_LENGTH:
        raise SystemExit(
            f"Fish codec geometry mismatch: sample_rate={model.sample_rate}, "
            f"frame_length={model.frame_length}"
        )
    try:
        quantize_source_path = Path(sys.modules[quantize_module.__name__].__file__).resolve()
        package_root = Path(quantize_source_path.parents[2]).resolve()
        quantize_source = {
            "file": quantize_source_path.relative_to(package_root).as_posix(),
            "sha256": sha256_file(quantize_source_path),
        }
    except (AttributeError, ValueError, IndexError) as error:
        raise SystemExit(f"cannot identify installed DAC quantizer source: {error}") from error

    codes_tensor = torch.tensor(codes, dtype=torch.long, device="cpu").unsqueeze(0)
    input_tensor = codes_tensor.to(device=args.device)
    baseline_input = input_tensor.clone()
    trace_input = input_tensor.clone()
    started = time.perf_counter()
    try:
        baseline_guard = _new_mps_guard(torch, args.device)
        with torch.inference_mode(), torch.device(args.device), baseline_guard:
            baseline_audio = model.from_indices(baseline_input)
        trace = _FishTrace()
        with trace:
            expected_names = trace.install(model)
            guard = _new_mps_guard(torch, args.device)
            with torch.inference_mode(), torch.device(args.device), guard:
                traced_audio = model.from_indices(trace_input)
        if list(trace.values) != expected_names:
            missing = [name for name in expected_names if name not in trace.values]
            extra = [name for name in trace.values if name not in expected_names]
            raise RuntimeError(
                f"trace stage set/order mismatch; missing={missing}, extra={extra}"
            )
        shape_equal = tuple(baseline_audio.shape) == tuple(traced_audio.shape)
        dtype_equal = baseline_audio.dtype == traced_audio.dtype
        if not _device_matches(baseline_audio.device, args.device) or not _device_matches(
            traced_audio.device, args.device
        ):
            raise RuntimeError(
                "Fish DAC returned an output outside the selected GPU: "
                f"baseline={baseline_audio.device}, traced={traced_audio.device}"
            )
        if shape_equal and dtype_equal:
            exact = bool(torch.equal(baseline_audio, traced_audio))
            difference = (baseline_audio - traced_audio).abs()
            max_abs_error = float(difference.max().item())
            mismatched_values = int((baseline_audio != traced_audio).sum().item())
        else:
            exact = False
            max_abs_error = None
            mismatched_values = None
        comparison = {
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "elementwise_exact": exact,
            "max_abs_error": max_abs_error,
            "mismatched_values": mismatched_values,
            "status": "PASS" if exact else "FAIL",
            "official_decode_shape": list(baseline_audio.shape),
            "traced_decode_shape": list(traced_audio.shape),
        }
        args.decode_seconds = time.perf_counter() - started
        if not exact:
            _write_failure_report(
                args,
                codes,
                input_layout,
                input_sha256,
                upstream,
                checkpoint_dtype_counts,
                parameter_dtype_counts,
                buffer_dtype_counts,
                quantize_source,
                comparison,
            )
            raise RuntimeError(
                "traced Fish decode differs from the unobserved official from_indices "
                f"result: {json.dumps(comparison, sort_keys=True)}"
            )
        args.expected_names = expected_names
        args.trace = trace
        args.comparison = comparison
        report = _capture_manifest(
            args,
            codes,
            input_layout,
            input_sha256,
            upstream,
            checkpoint_dtype_counts,
            parameter_dtype_counts,
            buffer_dtype_counts,
            quantize_source,
        )
    finally:
        if "trace" in locals():
            trace.close()
    print(json.dumps({"status": "PASS", "reference": str(args.output_dir / "reference.json"),
                      "frames": len(codes[0]), "stages": EXPECTED_STAGE_COUNT,
                      "official_decode_comparison": report["decode"]["official_decode_comparison"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
