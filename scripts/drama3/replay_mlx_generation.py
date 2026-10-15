#!/usr/bin/env python3
"""Replay a native Drama 3 frame prefix through pinned upstream MLX.

The generation report is the immutable teacher-forcing input: this script does
not sample a new sequence. It loads the pinned MLX-audio Fish Omni backbone,
main head, and codebook tower, then records selected predictions along the
same autoregressive prefix. Prompt prefill is forced to one row at a time to
match the native replay schedule and avoid chunk-size-dependent GEMM paths.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np


MLX_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
TRACE_FORMAT = "drama3-generation-common-prefix-v1"
CAPTURE_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 640)
TAIL_WINDOW_FRAMES = 128
CORE_PREFIXES = ("backbone.", "lm_head.", "tower.")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a JSON integer")
    return value


def read_generation_report(path: pathlib.Path) -> tuple[dict[str, Any], bytes, str]:
    raw = path.read_bytes()
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid generation report {path}: {error}") from error
    if not isinstance(report, dict):
        raise TypeError("generation report must be a JSON object")
    return report, raw, sha256_bytes(raw)


def selected_core_map(weight_map: dict[str, str]) -> dict[str, str]:
    selected = {
        name: shard
        for name, shard in weight_map.items()
        if name.startswith(CORE_PREFIXES)
    }
    if not selected:
        raise ValueError("checkpoint index contains no backbone/lm_head/tower tensors")
    return selected


def parameter_names(tree: Any, prefix: str = "") -> set[str]:
    """Flatten MLX's nested ``Module.parameters()`` tree to checkpoint names."""
    names: set[str] = set()
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            names.update(parameter_names(value, child))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            child = f"{prefix}.{index}" if prefix else str(index)
            names.update(parameter_names(value, child))
    else:
        if not prefix:
            raise ValueError("parameter tree contains an unnamed leaf")
        names.add(prefix)
    return names


def read_inputs(
    report: dict[str, Any],
    *,
    vocab_size: int,
    semantic_begin: int,
    semantic_end: int,
    eos_id: int,
    prompt_vocab_size: int,
    max_cache_rows: int = 1024,
) -> tuple[list[int], list[list[int]], int]:
    prompt_ids_raw = report.get("prompt_ids")
    if not isinstance(prompt_ids_raw, list) or not prompt_ids_raw:
        raise ValueError("generation report prompt_ids must be a non-empty array")
    prompt_ids = [strict_int(value, f"prompt_ids[{i}]") for i, value in enumerate(prompt_ids_raw)]
    if any(token < 0 or token >= prompt_vocab_size for token in prompt_ids):
        raise ValueError("generation report contains a prompt token outside the backbone vocabulary")

    prompt_rows = strict_int(report.get("prompt_rows"), "prompt_rows")
    if prompt_rows != len(prompt_ids):
        raise ValueError(f"prompt_rows={prompt_rows} does not match {len(prompt_ids)} prompt IDs")

    raw_frames = report.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("generation report frames must be a non-empty array")
    frames: list[list[int]] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, list) or len(raw_frame) != 11:
            raise ValueError(f"frames[{frame_index}] must contain main plus ten codebook IDs")
        frame = [strict_int(value, f"frames[{frame_index}][{column}]") for column, value in enumerate(raw_frame)]
        main = frame[0]
        if main == eos_id or not semantic_begin <= main <= semantic_end:
            raise ValueError(f"frames[{frame_index}] main token is outside the semantic range")
        if frame[1] != main - semantic_begin:
            raise ValueError(f"frames[{frame_index}] c0 does not equal main - semantic_begin")
        if any(code < 0 or code >= vocab_size for code in frame[1:]):
            raise ValueError(f"frames[{frame_index}] has a code outside the tower vocabulary")
        frames.append(frame)

    if prompt_rows + len(frames) > max_cache_rows:
        raise ValueError(
            f"rowwise replay needs {prompt_rows + len(frames)} cache rows; limit is {max_cache_rows}"
        )
    return prompt_ids, frames, prompt_rows


def capture_steps(frame_count: int) -> list[int]:
    tail_start = max(0, frame_count - TAIL_WINDOW_FRAMES)
    tail_steps = range(tail_start, frame_count + 1)
    return sorted({
        step
        for step in (*CAPTURE_CHECKPOINTS, *tail_steps, frame_count)
        if step <= frame_count
    })


def stage_record(name: str, shape: list[int], value: Any) -> dict[str, Any]:
    data = np.asarray(value, dtype=np.float32)
    if list(data.shape) != shape:
        raise ValueError(f"{name}: got shape {list(data.shape)}, expected {shape}")
    if not np.isfinite(data).all():
        raise FloatingPointError(f"{name}: non-finite MLX reference activation")
    return {
        "name": name,
        "shape": shape,
        "dtype": "float32",
        "values": np.ascontiguousarray(data).reshape(-1).tolist(),
    }


def load_core_weights(model: Any, model_dir: pathlib.Path, selected: dict[str, str], mx: Any) -> int:
    """Load indexed core tensors shardwise, promoting floating values to F32."""
    import torch
    from safetensors import safe_open

    expected = parameter_names(model.parameters())
    indexed = set(selected)
    if expected != indexed:
        raise ValueError(
            "indexed Fish Omni parameters differ from instantiated model: "
            f"missing={sorted(expected - indexed)[:8]}, extra={sorted(indexed - expected)[:8]}"
        )

    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)

    loaded = 0
    for shard in sorted(by_shard):
        shard_path = model_dir / shard
        names = sorted(by_shard[shard])
        if not shard_path.is_file():
            raise FileNotFoundError(f"indexed checkpoint shard is missing: {shard_path}")
        tensors_for_model = []
        with safe_open(str(shard_path), framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            missing = [name for name in names if name not in available]
            if missing:
                raise ValueError(f"{shard}: indexed core tensors missing: {missing[:8]}")
            for name in names:
                tensor = tensors.get_tensor(name)
                values = tensor.float().numpy() if tensor.is_floating_point() else tensor.numpy()
                tensors_for_model.append((name, mx.array(values)))
                del tensor, values
        model.load_weights(tensors_for_model, strict=False)
        loaded += len(tensors_for_model)
        print(f"loaded {loaded}/{len(selected)} core tensors from {shard}", file=sys.stderr)
        del tensors_for_model
        mx.clear_cache()

    if loaded != len(selected):
        raise AssertionError(f"loaded {loaded} core tensors; expected {len(selected)}")
    return loaded


def run_replay(
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
    generation_path: pathlib.Path,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    generation_path = generation_path.resolve()
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing replay output: {output_path}")

    import mlx.core as mx
    import torch
    from safetensors import __version__ as safetensors_version

    report, report_bytes, report_digest = read_generation_report(generation_path)
    if report.get("backend") != "metal_resident":
        raise ValueError("common-prefix replay requires a native metal_resident generation report")

    actual_revision = subprocess.check_output(
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != MLX_REVISION:
        raise RuntimeError(f"MLX-audio source is {actual_revision}; expected {MLX_REVISION}")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    selected = selected_core_map(index["weight_map"])

    sys.path.insert(0, str(mlx_root))
    from mlx_audio.tts.models.fish_omni import fish_omni, tower as tower_module
    from mlx_audio.tts.models.fish_omni.config import ModelConfig
    from mlx_audio.utils import apply_quantization

    mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"failed to select MLX GPU device: {mx.default_device()}")

    model_config = ModelConfig.from_dict(config)
    audio = model_config.audio_decoder_config
    text_config = config["text_config"]
    hidden_size = int(text_config["hidden_size"])
    main_vocab_size = 1 + int(config["semantic_end_token_id"]) - int(config["semantic_start_token_id"]) + 1
    if audio.num_codebooks != 10 or audio.vocab_size <= 0:
        raise ValueError("this replay requires the pinned ten-codebook Fish Omni tower")
    if int(text_config["vocab_size"]) <= 0 or main_vocab_size != model_config.semantic_end_token_id - model_config.semantic_start_token_id + 2:
        raise ValueError("invalid model vocabulary/configuration dimensions")

    prompt_ids, frames, prompt_rows = read_inputs(
        report,
        vocab_size=audio.vocab_size,
        semantic_begin=model_config.semantic_start_token_id,
        semantic_end=model_config.semantic_end_token_id,
        eos_id=model_config.eos_token_id,
        prompt_vocab_size=int(text_config["vocab_size"]),
    )
    frame_count = len(frames)
    steps = capture_steps(frame_count)
    if int(report.get("frame_count", frame_count)) != frame_count:
        raise ValueError("generation report frame_count disagrees with frames array")
    truncated = report.get("truncated")
    stopped_on_eos = report.get("stopped_on_eos")
    if not isinstance(truncated, bool) or not isinstance(stopped_on_eos, bool):
        raise ValueError("generation report must include boolean truncated and stopped_on_eos flags")
    if truncated == stopped_on_eos:
        raise ValueError("generation report termination flags are inconsistent")
    terminal_step = {
        "index": frame_count,
        "measured_in_generation": not truncated,
        "interpretation": (
            "counterfactual_after_truncated_prefix"
            if truncated
            else "actual_terminal_prediction_after_reported_frames"
        ),
    }

    model = fish_omni.Model(model_config)
    apply_quantization(model, config, selected)
    weight_count = load_core_weights(model, model_dir, selected, mx)
    model.eval()

    old_chunk_size = fish_omni.PREFILL_CHUNK_SIZE
    fish_omni.PREFILL_CHUNK_SIZE = 1
    original_sampler = tower_module.sample_top_k_top_p
    original_tower_call = tower_module.TowerLayer.__call__
    active_step: list[int | None] = [None]
    active_logits: list[list[Any] | None] = [None]
    target_codes: list[list[int] | None] = [None]
    sampler_call_index = [0]
    terminal_tower_hidden: list[Any | None] = [None]
    output_refs: list[tuple[str, list[int], Any]] = []
    step_output_refs: dict[int, list[Any]] = {}
    decoded_code_refs: list[tuple[int, Any]] = []
    cache_eval_seconds = 0.0
    prefill_started = time.monotonic()

    try:
        backbone_cache = model.backbone.make_cache()
        tower_cache = model.tower.make_cache()
        prompt = np.zeros((1, prompt_rows, audio.num_codebooks + 1), dtype=np.int32)
        prompt[0, :, 0] = np.asarray(prompt_ids, dtype=np.int32)
        prompt_mx = mx.array(prompt, dtype=mx.int32)
        hidden = model.prefill(prompt_mx, backbone_cache, tower_cache)
        prefill_seconds = time.monotonic() - prefill_started

        def save_hidden_step(step: int, value: Any) -> Any:
            logits = model.lm_head(value)[:, -1]
            if step in steps:
                output_refs.append((f"generation.step{step}.backbone_hidden", [hidden_size], value[0, -1]))
                output_refs.append((f"generation.step{step}.main_logits", [main_vocab_size], logits[0]))
                step_output_refs[step] = [value[0, -1], logits[0]]
            return logits

        # Capture direct return values from the official tower layer, and force
        # its official sampler to the corresponding known continuation codes.
        def traced_tower_call(
            self: Any,
            hidden_value: Any,
            codes: Any,
            semantic: Any,
            mask: Any,
            cache: Any,
            shared_input: Any = None,
        ) -> Any:
            result = original_tower_call(
                self, hidden_value, codes, semantic, mask, cache, shared_input
            )
            if active_step[0] is not None and self is model.tower.residual_layers[-1]:
                terminal_tower_hidden[0] = result[0, -1]
            return result

        def forced_sampler(logits: Any, temperature: float, top_k: int, top_p: float) -> Any:
            if active_step[0] is None or target_codes[0] is None:
                raise RuntimeError("tower sampler ran outside a teacher-forced decode step")
            index = sampler_call_index[0]
            if index >= 9:
                raise RuntimeError("tower decode invoked more than nine codebook heads")
            sampler_call_index[0] += 1
            if active_logits[0] is not None:
                active_logits[0].append(logits)
            target = target_codes[0][index]
            return mx.array([target], dtype=mx.int32)

        tower_module.TowerLayer.__call__ = traced_tower_call
        tower_module.sample_top_k_top_p = forced_sampler

        generation_started = time.monotonic()
        # Step 0 predicts the first generated frame from the final prompt row.
        save_hidden_step(0, hidden)
        previous_frame_values = [prompt_ids[-1]] + [0] * audio.num_codebooks
        previous_frame = mx.array(previous_frame_values, dtype=mx.int32)[None, None, :]

        for step, frame_values in enumerate(frames):
            selected_step = step in steps
            logits_refs: list[Any] | None = [] if selected_step else None
            active_step[0] = step
            active_logits[0] = logits_refs
            target_codes[0] = frame_values[2:]
            sampler_call_index[0] = 0
            terminal_tower_hidden[0] = None

            main = mx.array([[frame_values[0]]], dtype=mx.int32)
            codes = model.tower.decode_step(
                main,
                hidden,
                previous_frame,
                tower_cache,
                temperature=float(report["temperature"]),
                top_k=int(report["top_k"]),
                top_p=float(report["top_p"]),
            )
            if sampler_call_index[0] != 9:
                raise AssertionError(
                    f"step {step}: official tower called {sampler_call_index[0]} heads; expected 9"
                )
            if selected_step:
                if logits_refs is None or len(logits_refs) != 9:
                    raise AssertionError(f"step {step}: captured {0 if logits_refs is None else len(logits_refs)} tower heads")
                tower_hidden = terminal_tower_hidden[0]
                if tower_hidden is None:
                    raise AssertionError(f"step {step}: final tower layer output was not captured")
                output_refs.append((f"generation.step{step}.tower_hidden", [audio.dim], tower_hidden))
                for head_index, logits in enumerate(logits_refs):
                    output_refs.append(
                        (f"generation.step{step}.tower_logits.{head_index}", [audio.vocab_size], logits[0])
                    )
            decoded_code_refs.append((step, codes))

            # Realize the current row's caches before advancing; no array is
            # converted to a host value until the final readback phase.
            eval_started = time.monotonic()
            mx.eval(
                codes,
                *step_output_refs.get(step, []),
                *([] if not selected_step else [terminal_tower_hidden[0], *logits_refs]),
                [cache.state for cache in backbone_cache],
                [cache.state for cache in tower_cache],
            )
            cache_eval_seconds += time.monotonic() - eval_started

            active_step[0] = None
            active_logits[0] = None
            target_codes[0] = None
            current_frame = mx.array(frame_values, dtype=mx.int32)[None, None, :]
            hidden = model._backbone_step(current_frame, backbone_cache)
            previous_frame = current_frame
            eval_started = time.monotonic()
            mx.eval(hidden, [cache.state for cache in backbone_cache])
            cache_eval_seconds += time.monotonic() - eval_started

            next_step = step + 1
            if next_step < frame_count:
                save_hidden_step(next_step, hidden)

        # The last generated frame is consumed to obtain the next-main-token
        # prediction. For capped reports this is explicitly counterfactual.
        if frame_count in steps:
            save_hidden_step(frame_count, hidden)
        generation_seconds = time.monotonic() - generation_started
        terminal_mx_eval_started = time.monotonic()
        mx.eval(
            hidden,
            *[value for _, _, value in output_refs],
            *[value for _, value in decoded_code_refs],
            [cache.state for cache in backbone_cache],
            [cache.state for cache in tower_cache],
        )
        cache_eval_seconds += time.monotonic() - terminal_mx_eval_started
    finally:
        tower_module.sample_top_k_top_p = original_sampler
        tower_module.TowerLayer.__call__ = original_tower_call
        fish_omni.PREFILL_CHUNK_SIZE = old_chunk_size

    # Only now are MLX arrays transferred to NumPy/host memory.
    read_started = time.monotonic()
    traces = [stage_record(name, shape, value) for name, shape, value in output_refs]
    traces.sort(key=lambda item: item["name"])
    decoded: list[list[int]] = []
    for step, value in decoded_code_refs:
        codes = np.asarray(value, dtype=np.int32).reshape(-1)
        expected = np.asarray(frames[step][1:], dtype=np.int32)
        if codes.shape != (audio.num_codebooks,) or not np.array_equal(codes, expected):
            raise AssertionError(f"step {step}: forced tower codes did not round-trip to the fixture")
        decoded.append(codes.tolist())
    readback_seconds = time.monotonic() - read_started

    expected_count = len(steps) * 2 + sum(step < frame_count for step in steps) * 10
    if len(traces) != expected_count:
        raise AssertionError(f"emitted {len(traces)} stages; expected {expected_count}")
    if len({trace["name"] for trace in traces}) != len(traces):
        raise AssertionError("duplicate generation trace stage name")

    source_files = {
        "fish_omni": mlx_root / "mlx_audio/tts/models/fish_omni/fish_omni.py",
        "fish_omni_tower": mlx_root / "mlx_audio/tts/models/fish_omni/tower.py",
        "fish_omni_config": mlx_root / "mlx_audio/tts/models/fish_omni/config.py",
        "qwen3_5": mlx_root / "mlx_audio/lm/models/qwen3_5.py",
        "cache": mlx_root / "mlx_audio/lm/models/cache.py",
        "utils": mlx_root / "mlx_audio/utils.py",
    }
    source_hashes = {name: sha256_file(path) for name, path in source_files.items()}
    source_hashes["replay_script"] = sha256_file(pathlib.Path(__file__).resolve())
    model_files = {
        "config.json": sha256_file(config_path),
        "model.safetensors.index.json": sha256_file(index_path),
    }
    source_generation = {
        key: report[key]
        for key in ("elapsed_seconds", "stopped_on_eos", "truncated")
        if key in report
    }
    if isinstance(report.get("phase_seconds"), dict) and "generation" in report["phase_seconds"]:
        source_generation["generation_seconds"] = report["phase_seconds"]["generation"]

    bundle = {
        "format": TRACE_FORMAT,
        "source_report": {"path": str(generation_path), "sha256": report_digest},
        "source_generation": source_generation,
        "prompt_ids": prompt_ids,
        "prompt_rows": prompt_rows,
        "frames": frames,
        "frame_count": frame_count,
        "capture_steps": steps,
        "capture_policy": {
            "checkpoints": list(CAPTURE_CHECKPOINTS),
            "tail_window_frames": TAIL_WINDOW_FRAMES,
            "tail_inclusive": True,
        },
        "terminal_step": terminal_step,
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "upstream": {
            "repository": "https://github.com/cguldogan/mlx-audio",
            "revision": MLX_REVISION,
        },
        "source_hashes": source_hashes,
        "model_files": model_files,
        "model_config": {
            "hidden_size": hidden_size,
            "main_vocab_size": main_vocab_size,
            "tower_dim": int(audio.dim),
            "tower_vocab_size": int(audio.vocab_size),
            "num_codebooks": int(audio.num_codebooks),
            "loaded_core_tensor_count": weight_count,
            "loaded_core_shards": sorted(set(selected.values())),
            "shard_sha256_recorded": False,
        },
        "execution": {
            "backend": "mlx_reference",
            "device": str(mx.default_device()),
            "prefill_schedule": "rowwise",
            "frame_schedule": "teacher_forced_autoregressive",
            "sampling": "disabled; official tower sampler returns fixture code IDs",
            "float_checkpoint_weights_promoted_to_float32": True,
        },
        "generation_parameters": {
            key: report[key]
            for key in ("seed", "temperature", "top_k", "top_p")
            if key in report
        },
        "timings": {
            "scope": "diagnostic teacher-forced replay; not generation latency",
            "prefill_seconds": prefill_seconds,
            "teacher_forced_replay_seconds": generation_seconds,
            "cache_eval_seconds": cache_eval_seconds,
            "terminal_readback_seconds": readback_seconds,
        },
        "readbacks": {
            "terminal_array_count": len(output_refs) + len(decoded_code_refs),
            "terminal_float32_values": sum(len(trace["values"]) for trace in traces),
            "forced_code_values": len(decoded),
        },
        "forced_code_roundtrip": decoded,
        "traces": traces,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing trace: {output_path}")
    output_bytes = json.dumps(bundle, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    output_path.write_bytes(output_bytes)
    manifest = {
        "kind": "pinned_mlx_fish_omni_common_prefix_replay",
        "trace": {"path": output_path.name, "sha256": sha256_bytes(output_bytes)},
        "source_report": bundle["source_report"],
        "model": bundle["model"],
        "upstream": bundle["upstream"],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "safetensors": safetensors_version,
        },
        "source_hashes": source_hashes,
        "model_files": model_files,
        "loaded_core_tensor_count": weight_count,
        "loaded_core_shards": sorted(set(selected.values())),
        "trace_count": len(traces),
        "trace_names": [trace["name"] for trace in traces],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--generation", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    for field in ("mlx_root", "model_dir", "generation", "output"):
        setattr(args, field, getattr(args, field).resolve())
    manifest = run_replay(args.mlx_root, args.model_dir, args.generation, args.output)
    print(json.dumps({"trace": str(args.output), "trace_count": manifest["trace_count"], "manifest": str(args.output) + ".manifest.json"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
