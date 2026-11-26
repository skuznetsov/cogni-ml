#!/usr/bin/env python3
"""Emit deterministic MLX codec tensors and stage outputs for native parity tests.

Reference implementation: mlx-audio fish_s1_dac.py, e7488935ec2ebfafd38ef2b58e290d79f3d05dc7.
This is test-only; production decode remains in Crystal.
"""

import argparse
import json
import math
import wave

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.codec.models.fish_s1_dac.fish_s1_dac import (
    DAC,
    DownsampleResidualVectorQuantize,
    ModelArgs,
    WindowLimitedTransformer,
    build_ae,
)


SYNTHETIC_CONFIG = {
    "latent_channels": 8,
    "codebook_dim": 2,
    "semantic_codebook_size": 5,
    "residual_codebooks": 2,
    "residual_codebook_size": 4,
    "post_layers": 1,
    "transformer_heads": 2,
    "transformer_head_dim": 4,
    "transformer_ff_dim": 12,
    "transformer_window": 2,
    # Distinct rates discriminate the reversed decoder stage ordering.
    "quantizer_upsample_rates": [2, 3],
    "decoder_channels": 8,
    "decoder_rates": [2],
    "norm_eps": 1.0e-5,
    "rope_base": 10_000.0,
}


def make_synthetic_model():
    transformer = WindowLimitedTransformer(
        config=ModelArgs(
            block_size=16,
            n_layer=SYNTHETIC_CONFIG["post_layers"],
            n_head=SYNTHETIC_CONFIG["transformer_heads"],
            dim=SYNTHETIC_CONFIG["latent_channels"],
            intermediate_size=SYNTHETIC_CONFIG["transformer_ff_dim"],
            head_dim=SYNTHETIC_CONFIG["transformer_head_dim"],
            norm_eps=SYNTHETIC_CONFIG["norm_eps"],
            rope_base=SYNTHETIC_CONFIG["rope_base"],
            channels_first=True,
        ),
        input_dim=SYNTHETIC_CONFIG["latent_channels"],
        window_size=SYNTHETIC_CONFIG["transformer_window"],
        causal=True,
    )
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=SYNTHETIC_CONFIG["latent_channels"],
        n_codebooks=SYNTHETIC_CONFIG["residual_codebooks"],
        codebook_size=SYNTHETIC_CONFIG["residual_codebook_size"],
        codebook_dim=SYNTHETIC_CONFIG["codebook_dim"],
        semantic_codebook_size=SYNTHETIC_CONFIG["semantic_codebook_size"],
        downsample_factor=tuple(SYNTHETIC_CONFIG["quantizer_upsample_rates"]),
        downsample_dims=(
            SYNTHETIC_CONFIG["latent_channels"],
        )
        * len(SYNTHETIC_CONFIG["quantizer_upsample_rates"]),
        post_module=transformer,
    )
    return DAC(
        encoder_dim=2,
        encoder_rates=[1],
        latent_dim=SYNTHETIC_CONFIG["latent_channels"],
        decoder_dim=SYNTHETIC_CONFIG["decoder_channels"],
        decoder_rates=SYNTHETIC_CONFIG["decoder_rates"],
        quantizer=quantizer,
        causal=True,
    )


def deterministic_array(name, shape):
    count = math.prod(shape)
    tag = sum(ord(char) for char in name) % 23
    values = (((np.arange(count, dtype=np.int64) * 17 + tag * 7) % 31) - 15).astype(np.float32)
    values *= np.float32(0.002)
    if name.endswith(".weight_g"):
        values = np.float32(0.25) + (np.arange(count, dtype=np.float32) % 5) * np.float32(0.01)
    elif name.endswith(".alpha"):
        values = np.float32(0.7) + (np.arange(count, dtype=np.float32) % 7) * np.float32(0.03)
    elif name.endswith(".gamma"):
        values = np.float32(0.015) + (np.arange(count, dtype=np.float32) % 5) * np.float32(0.001)
    elif name.endswith(".norm.weight") or name.endswith(".attention_norm.weight") or name.endswith(".ffn_norm.weight"):
        values = np.float32(0.9) + (np.arange(count, dtype=np.float32) % 7) * np.float32(0.01)
    elif name.endswith(".codebook.weight"):
        values = np.float32(0.1) + values
    return values.reshape(shape)


def flatten_time_major(value):
    mx.eval(value)
    array = np.asarray(value[0].transpose(1, 0), dtype=np.float32)
    return array.reshape(-1).tolist()


def flatten_waveform(value):
    mx.eval(value)
    return np.asarray(value[0, 0], dtype=np.float32).reshape(-1).tolist()


def decode_with_trace(model, codes, capture_stages=True):
    indices = mx.array(np.asarray(codes, dtype=np.int32)[None, :, :])
    traces = {}

    def save(name, value, waveform=False):
        # Evaluating each stage releases its temporary graph before the next
        # large DAC block. Full native generations need only the final PCM, so
        # avoid retaining every intermediate activation in Python lists.
        if capture_stages:
            flattened = flatten_waveform(value) if waveform else flatten_time_major(value)
            traces[name] = flattened
        else:
            mx.eval(value)

    x = model.quantizer._codes_to_latents(indices)
    save("quantizer.codebook_sum", x)
    x = model.quantizer.post_module(x)
    save("quantizer.post_module", x)
    for stage, block in enumerate(model.quantizer.upsample):
        x = block[0](x)
        save(f"quantizer.upsample.{stage}.deconv", x)
        x = block[1](x)
        save(f"quantizer.upsample.{stage}.convnext", x)

    decoder = model.decoder
    x = decoder.model[0](x)
    save("decoder.model.0", x)
    for stage in range(len(decoder.model) - 4):
        block = decoder.model[stage + 1]
        for substage, layer in enumerate(block.block):
            x = layer(x)
            save(f"decoder.model.{stage + 1}.block.{substage}", x)
    final_index = len(decoder.model) - 3
    x = decoder.model[final_index](x)
    save(f"decoder.model.{final_index}", x)
    x = decoder.model[final_index + 1](x)
    save(f"decoder.model.{final_index + 1}", x)
    x = decoder.model[final_index + 2](x)
    save("waveform", x, waveform=True)
    waveform = np.asarray(x[0, 0], dtype=np.float32).reshape(-1)
    return traces, waveform


def read_native_codes(path):
    """Convert native frame rows [main, c0, ..., c9] to DAC book-major ids."""
    with open(path, "r", encoding="utf-8") as source:
        sidecar = json.load(source)
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("frames"), list):
        raise ValueError("--codes-json must be a native generation report with a frames array")
    frames = sidecar["frames"]
    if not frames:
        raise ValueError("native generation report contains no audio frames")

    codebooks = [[] for _ in range(10)]
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, list) or len(frame) != 11:
            raise ValueError(
                f"native frame {frame_index} must contain [main, c0, ..., c9]"
            )
        if any(type(code) is not int for code in frame):
            raise ValueError(f"native frame {frame_index} contains a non-integer code")
        for book_index, code in enumerate(frame[1:]):
            codebooks[book_index].append(code)
    return codebooks


def pcm16_samples(waveform):
    """Match the native writer's F32 clip, scale, and ties-to-even rounding."""
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("reference waveform must be non-empty and finite")
    clipped = np.clip(samples, np.float32(-1.0), np.float32(1.0))
    scaled = (clipped * np.float32(32767.0)).astype(np.float32)
    # Crystal Number#round defaults to Float32 round_even (ties-to-even).
    return np.rint(scaled).astype("<i2")


def write_reference_wav(path, waveform):
    pcm = pcm16_samples(waveform)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(44_100)
        output.writeframes(pcm.tobytes())
    return pcm


def compare_wav(path, reference_pcm):
    with wave.open(str(path), "rb") as native:
        if native.getnchannels() != 1 or native.getsampwidth() != 2:
            raise ValueError("--compare-wav must be mono PCM16")
        if native.getframerate() != 44_100 or native.getcomptype() != "NONE":
            raise ValueError("--compare-wav must be uncompressed 44100 Hz PCM")
        native_pcm = np.frombuffer(native.readframes(native.getnframes()), dtype="<i2")
    if native_pcm.size != reference_pcm.size:
        raise ValueError(
            f"WAV sample count mismatch: native={native_pcm.size}, reference={reference_pcm.size}"
        )
    delta = reference_pcm.astype(np.int32) - native_pcm.astype(np.int32)
    max_abs_pcm_error = int(np.max(np.abs(delta)))
    rms_pcm_error = float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))
    return {
        "samples": int(reference_pcm.size),
        "exact_samples": int(np.count_nonzero(delta == 0)),
        "max_abs_pcm_error": max_abs_pcm_error,
        "rms_pcm_error": rms_pcm_error,
        "max_abs_error_normalized_units": max_abs_pcm_error / 32767.0,
        "rms_error_normalized_units": rms_pcm_error / 32767.0,
        "one_pcm_lsb_normalized_units": 1.0 / 32767.0,
        "within_one_pcm_lsb": max_abs_pcm_error <= 1,
    }


def raw_weights(parameters):
    wn_stems = {name[:-9] for name in parameters if name.endswith(".weight_g")}
    result = {}
    for name, array in parameters.items():
        if name.endswith(".weight_g"):
            key = name[:-9] + ".conv.parametrizations.weight.original0"
        elif name.endswith(".weight_v"):
            key = name[:-9] + ".conv.parametrizations.weight.original1"
        elif name.endswith(".bias") and name[:-5] in wn_stems:
            key = name[:-5] + ".conv.bias"
        else:
            key = name
        result[key] = {
            "shape": list(array.shape),
            "values": np.asarray(array, dtype=np.float32).reshape(-1).astype(np.float64).tolist(),
        }
    return result


def build_payload(mode, codec_path=None, codes_json=None, wav_output=None, compare_wav_path=None):
    if mode == "synthetic":
        if codes_json is not None:
            raise ValueError("--codes-json is supported only in real mode")
        model = make_synthetic_model()
        parameters = {}
        for name, initial in tree_flatten(model.parameters()):
            if name.startswith("encoder.") or ".conv.conv." in name:
                continue
            array = deterministic_array(name, initial.shape)
            parameters[name] = mx.array(array)
        model.load_weights(list(parameters.items()), strict=False)
        mx.eval(model.parameters())
        codes = [[2, 4, -5], [3, 7, 0], [1, 2, 5]]
        config = SYNTHETIC_CONFIG
        weights = raw_weights(parameters)
    else:
        if codec_path is None:
            raise ValueError("--codec is required for real mode")
        model = build_ae()
        weights = model.sanitize(mx.load(codec_path))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        if codes_json is None:
            codes = [[(book * 73 + 1) % 4096 for _ in range(2)] for book in range(10)]
            for book in range(1, 10):
                codes[book] = [(book * 73 + 1) % 1024, (book * 73 + 74) % 1024]
        else:
            codes = read_native_codes(codes_json)
        config = None
        # The production codec has ~1.87 GB of F32 weights. The Crystal side
        # mmap-loads the original safetensors file directly; serializing those
        # tensors through JSON would multiply peak memory without adding evidence.
        weights = {}

    capture_stages = codes_json is None
    traces, waveform = decode_with_trace(model, codes, capture_stages)
    payload = {"mode": mode, "config": config, "codes": codes, "weights": weights, "traces": traces}
    if wav_output is not None:
        reference_pcm = write_reference_wav(wav_output, waveform)
        payload["wav_output"] = wav_output
        if compare_wav_path is not None:
            payload["wav_comparison"] = compare_wav(compare_wav_path, reference_pcm)
    elif compare_wav_path is not None:
        raise ValueError("--compare-wav requires --wav-output")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["synthetic", "real"])
    parser.add_argument("--codec")
    parser.add_argument("--output", required=True)
    parser.add_argument("--codes-json", help="native generation report containing [main, c0, ..., c9] frame rows")
    parser.add_argument("--wav-output", help="write reference PCM16 mono WAV")
    parser.add_argument("--compare-wav", help="compare output samples with this native WAV")
    args = parser.parse_args()
    if args.compare_wav is not None and args.wav_output is None:
        parser.error("--compare-wav requires --wav-output")
    payload = build_payload(args.mode, args.codec, args.codes_json, args.wav_output, args.compare_wav)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(payload, output, separators=(",", ":"), allow_nan=False)
    if "wav_comparison" in payload:
        print(json.dumps(payload["wav_comparison"], sort_keys=True))


if __name__ == "__main__":
    main()
