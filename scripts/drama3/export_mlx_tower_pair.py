#!/usr/bin/env python3
"""Export pinned MLX tower traces and pair inputs for the native fusion probe.

The common prefix is taken from an immutable native metal_resident generation
report. Prompt prefill uses the official one-row MLX path. The selected tower
prefill trace is cache row 85; the selected decode uses that completed prompt
cache, the report's first generated main token, and deterministic top-1
sampling for every residual codebook. Arrays stay on MLX until the final JSON
export phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
from typing import Any

import numpy as np


UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
GENERATION_SHA256 = "ba22da53fa60afc34e80b23594ca311c6eb5fba74c5eed47605e87499bbf1fcf"
CONFIG_SHA256 = "9d52aa7960ba723efcfe47d2ec03f52c5e22e84a6c2a7d773a845db1bf48563c"
INDEX_SHA256 = "da812cc5b08188ccef048b9074e4cf70c8a1896859c5b492f374d7833e63b855"
REPLAY_HELPER_SHA256 = "33a1468038b1a97ec9d144665a4d0572a9fed9c9ab544ab2f97bc64c6cd31cad"
PREFILL_TOWER_ROW = 85
PAIR_CASES = (
    (
        "shared0-prefill",
        "tower.shared_layers.0.transformer_layer.feed_forward",
        "tower.prefill.row85.shared_layers.0",
    ),
    (
        "residual5-decode",
        "tower.residual_layers.5.transformer_layer.feed_forward",
        "tower.decode.row0.residual_layers.5",
    ),
    (
        "residual8-decode",
        "tower.residual_layers.8.transformer_layer.feed_forward",
        "tower.decode.row0.residual_layers.8",
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_tensor_hash(model_dir: pathlib.Path, index: dict[str, Any]) -> str:
    """Match WeightStore's exact BF16 name/dtype/shape/raw-payload serializer."""
    import torch
    from safetensors import safe_open

    weight_map = index["weight_map"]
    names = sorted(
        {
            f"{prefix}.{projection}.weight"
            for _, prefix, _ in PAIR_CASES
            for projection in ("w1", "w3")
        }
    )
    digest = hashlib.sha256()
    for name in names:
        shard = weight_map.get(name)
        if not isinstance(shard, str):
            raise ValueError(f"checkpoint index has no shard for {name}")
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as tensors:
            tensor = tensors.get_tensor(name)
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (6144, 2048):
            raise ValueError(f"{name}: expected BF16 [6144,2048], got {tensor.dtype} {tuple(tensor.shape)}")
        payload = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0torch.bfloat16\0")
        digest.update(struct.pack("<qq", *tensor.shape))
        digest.update(payload)
    return digest.hexdigest()


class Capture:
    def __init__(self, mx: Any, tower_module: Any, model: Any):
        self.mx = mx
        self.tower_module = tower_module
        self.model = model
        self.mode: str | None = None
        self.row: int | None = None
        self.arrays: dict[str, tuple[list[int], Any]] = {}
        self.layer_parts: dict[int, tuple[str, int]] = {}
        self.norm_suffixes: dict[int, tuple[int, str]] = {}

        for group, layers in (
            ("shared_layers", model.tower.shared_layers),
            ("residual_layers", model.tower.residual_layers),
        ):
            for index, layer in enumerate(layers):
                self.layer_parts[id(layer)] = (group, index)
                self.norm_suffixes[id(layer.enorm)] = (id(layer), "enorm")
                self.norm_suffixes[id(layer.hnorm)] = (id(layer), "hnorm")
                self.norm_suffixes[id(layer.final_layernorm)] = (id(layer), "output")
                transformer = layer.transformer_layer
                self.norm_suffixes[id(transformer.attention_norm)] = (id(layer), "attention_norm")
                self.norm_suffixes[id(transformer.ffn_norm)] = (id(layer), "ffn_norm")
                attention = transformer.attention
                self.norm_suffixes[id(attention.q_norm)] = (id(layer), "attention.q_norm")
                self.norm_suffixes[id(attention.k_norm)] = (id(layer), "attention.k_norm")

        self.originals: dict[tuple[Any, str], Any] = {}

    def active(self) -> bool:
        return self.mode is not None and self.row is not None

    def prefix(self, layer: Any) -> str:
        group, index = self.layer_parts[id(layer)]
        return f"tower.{self.mode}.row{self.row}.{group}.{index}"

    def put(self, name: str, value: Any, shape: list[int] | None = None) -> None:
        if not self.active():
            return
        actual_shape = list(value.shape) if shape is None else shape
        if name in self.arrays:
            raise ValueError(f"duplicate tower trace name {name}")
        self.arrays[name] = (actual_shape, value)

    def remember(self, owner: Any, method: str) -> Any:
        key = (owner, method)
        if key not in self.originals:
            self.originals[key] = getattr(owner, method)
        return self.originals[key]

    def install(self) -> None:
        tower = self.tower_module
        audio_decoder = type(self.model.tower)

        original_norm = self.remember(tower.ZeroCenteredRMSNorm, "__call__")

        def traced_norm(norm: Any, x: Any) -> Any:
            out = original_norm(norm, x)
            entry = self.norm_suffixes.get(id(norm))
            if entry is not None and self.active():
                layer_id, suffix = entry
                layer = self._layer_by_id[layer_id]
                prefix = self.prefix(layer)
                shape = list(out.shape)
                trace_out = out
                if suffix.endswith("q_norm") or suffix.endswith("k_norm"):
                    # MLX normalizes the current token as [batch,time,heads,dim],
                    # while the native trace contract is [batch,heads,time,dim].
                    # These probes have one current token even when the KV cache
                    # already has history; transpose the captured view itself,
                    # without changing the tensor returned to attention.
                    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
                        raise ValueError(
                            f"{prefix}.{suffix}: expected one current token before cache update, got {shape}"
                        )
                    shape = [shape[0], shape[2], shape[1], shape[3]]
                    trace_out = out.transpose(0, 2, 1, 3)
                self.put(f"{prefix}.{suffix}", trace_out, shape)
            return out

        tower.ZeroCenteredRMSNorm.__call__ = traced_norm

        original_mlp = self.remember(tower.TowerMLP, "__call__")

        def traced_mlp(mlp: Any, x: Any) -> Any:
            if not self.active():
                return original_mlp(mlp, x)
            layer = self._layer_by_mlp[id(mlp)]
            prefix = self.prefix(layer) + ".feed_forward"
            up = mlp.w1(x)
            self.put(prefix + ".w1", up)
            activated = tower.nn.silu(up)
            self.put(prefix + ".silu", activated)
            gate = mlp.w3(x)
            self.put(prefix + ".w3", gate)
            product = activated * gate
            self.put(prefix + ".product", product)
            output = mlp.w2(product)
            self.put(prefix + ".w2", output)
            return output

        tower.TowerMLP.__call__ = traced_mlp

        original_attention = self.remember(tower.TowerAttention, "__call__")

        def traced_attention(attention: Any, x: Any, mask: Any, cache: Any) -> Any:
            if not self.active():
                return original_attention(attention, x, mask, cache)
            layer = self._layer_by_attention[id(attention)]
            prefix = self.prefix(layer) + ".attention"
            batch, length, _ = x.shape
            expected_cache_offset = (
                PREFILL_TOWER_ROW if self.mode == "prefill" else PREFILL_TOWER_ROW + 1
            )
            if length != 1 or int(cache.offset) != expected_cache_offset:
                raise ValueError(
                    f"{prefix}: expected one current token at cache offset {expected_cache_offset}, "
                    f"got length={length}, offset={cache.offset}"
                )
            q, gate, k, v = self.mx.split(
                attention.wqkv(x),
                [attention.q_width, 2 * attention.q_width, 2 * attention.q_width + attention.kv_width],
                axis=-1,
            )
            q = q.reshape(batch, length, attention.n_head, attention.head_dim)
            gate = gate.reshape(batch, length, attention.n_head, attention.head_dim)
            k = k.reshape(batch, length, attention.n_local_heads, attention.head_dim)
            v = v.reshape(batch, length, attention.n_local_heads, attention.head_dim)
            self.put(prefix + ".q_projected", q)
            self.put(prefix + ".gate", gate)
            self.put(prefix + ".k_projected", k)
            self.put(prefix + ".v_projected", v)
            q = attention.q_norm(q)
            k = attention.k_norm(k)
            q = attention.rope(q.transpose(0, 2, 1, 3), offset=cache.offset)
            k = attention.rope(k.transpose(0, 2, 1, 3), offset=cache.offset)
            self.put(prefix + ".q_rope", q)
            self.put(prefix + ".k_rope", k)
            v = v.transpose(0, 2, 1, 3)
            k, v = cache.update_and_fetch(k, v)
            output = self.mx.fast.scaled_dot_product_attention(q, k, v, scale=attention.scale, mask=mask)
            output = output.transpose(0, 2, 1, 3).reshape(batch, length, attention.q_width)
            gated = output * self.mx.sigmoid(gate.reshape(batch, length, attention.q_width))
            self.put(prefix + ".gated_context", gated)
            result = attention.wo(gated)
            self.put(prefix + ".output", result)
            return result

        tower.TowerAttention.__call__ = traced_attention

        original_transformer = self.remember(tower.TowerTransformerLayer, "__call__")

        def traced_transformer(transformer: Any, x: Any, mask: Any, cache: Any) -> Any:
            if not self.active():
                return original_transformer(transformer, x, mask, cache)
            layer = self._layer_by_transformer[id(transformer)]
            prefix = self.prefix(layer)
            h = x + transformer.attention(transformer.attention_norm(x), mask, cache)
            self.put(prefix + ".attention_residual", h)
            residual = h + transformer.feed_forward(transformer.ffn_norm(h))
            self.put(prefix + ".ffn_residual", residual)
            return residual

        tower.TowerTransformerLayer.__call__ = traced_transformer

        original_layer = self.remember(tower.TowerLayer, "__call__")

        def traced_layer(layer: Any, hidden: Any, codes: Any, semantic: Any, mask: Any,
                         cache: Any, shared_input: Any = None) -> Any:
            if not self.active():
                return original_layer(layer, hidden, codes, semantic, mask, cache, shared_input)
            prefix = self.prefix(layer)
            embedding = layer.semantic_embedding if layer.shared else layer.codebook_embedding
            e = embedding(self.mx.clip(codes, 0, layer.vocab_size - 1))
            e = self.mx.where(semantic[..., None], e, self.mx.zeros_like(e))
            if shared_input is not None:
                e = e + shared_input
            self.put(prefix + ".embedding", e)
            combined = self.mx.concatenate([layer.enorm(e), layer.hnorm(hidden)], axis=-1)
            self.put(prefix + ".eh_concat", combined)
            mixed = layer.eh_proj(combined)
            self.put(prefix + ".eh_proj", mixed)
            result = layer.final_layernorm(layer.transformer_layer(mixed, mask, cache))
            if not layer.shared:
                residual_index = self.layer_parts[id(layer)][1]
                logits = self.model.tower.codebook_heads[residual_index](result)[:, -1]
                self.put(f"tower.{self.mode}.row{self.row}.head_logits.{residual_index}", logits)
                if self.mode == "decode" and residual_index == len(self.model.tower.residual_layers) - 1:
                    self.put(f"tower.{self.mode}.row{self.row}.hidden_output", result)
            return result

        tower.TowerLayer.__call__ = traced_layer

        original_prefill = self.remember(audio_decoder, "prefill")

        original_run_shared = self.remember(audio_decoder, "run_shared")

        def traced_run_shared(decoder: Any, hidden: Any, c0: Any, semantic: Any,
                              frames: Any, mask: Any, caches: Any) -> Any:
            if not self.active():
                return original_run_shared(decoder, hidden, c0, semantic, frames, mask, caches)
            unseen = decoder.embedding_sum(
                frames,
                1 + decoder.config.vq_backbone_codebooks,
                decoder.config.num_codebooks,
            )
            unseen = self.mx.where(
                decoder.is_semantic(frames[..., 0])[..., None],
                unseen,
                self.mx.zeros_like(unseen),
            )
            self.put(f"tower.{self.mode}.row{self.row}.unseen_embedding", unseen)
            for index, layer in enumerate(decoder.shared_layers):
                hidden = layer(
                    hidden,
                    c0,
                    semantic,
                    mask,
                    caches[index],
                    unseen if index == 0 else None,
                )
            return hidden

        audio_decoder.run_shared = traced_run_shared

        def traced_prefill(decoder: Any, hidden: Any, frames: Any, next_frames: Any, caches: Any) -> Any:
            offset = int(caches[0].offset)
            if offset != PREFILL_TOWER_ROW:
                return original_prefill(decoder, hidden, frames, next_frames, caches)
            self.mode, self.row = "prefill", PREFILL_TOWER_ROW
            try:
                prefix = f"tower.prefill.row{PREFILL_TOWER_ROW}"
                self.put(prefix + ".hidden_input", hidden)
                result = original_prefill(decoder, hidden, frames, next_frames, caches)
                self.put(prefix + ".hidden_output", result)
                return result
            finally:
                self.mode, self.row = None, None

        audio_decoder.prefill = traced_prefill

        original_decode = self.remember(audio_decoder, "decode_step")

        def traced_decode(decoder: Any, main: Any, hidden: Any, frames: Any, caches: Any,
                          *, temperature: float, top_k: int, top_p: float) -> Any:
            self.mode, self.row = "decode", 0
            try:
                self.put("tower.decode.row0.hidden_input", hidden)
                return original_decode(decoder, main, hidden, frames, caches,
                                       temperature=temperature, top_k=top_k, top_p=top_p)
            finally:
                self.mode, self.row = None, None

        audio_decoder.decode_step = traced_decode

        original_sampler = self.remember(tower, "sample_top_k_top_p")

        def top1_sampler(logits: Any, temperature: float, top_k: int, top_p: float) -> Any:
            if self.mode == "decode":
                return self.mx.argmax(logits.astype(self.mx.float32), axis=-1)
            return original_sampler(logits, temperature, top_k, top_p)

        tower.sample_top_k_top_p = top1_sampler

        self._layer_by_id = self._all_layers
        self._layer_by_mlp = {
            id(layer.transformer_layer.feed_forward): layer for layer in self._all_layers.values()
        }
        self._layer_by_attention = {
            id(layer.transformer_layer.attention): layer for layer in self._all_layers.values()
        }
        self._layer_by_transformer = {
            id(layer.transformer_layer): layer for layer in self._all_layers.values()
        }

    def restore(self) -> None:
        for (owner, method), original in reversed(list(self.originals.items())):
            setattr(owner, method, original)


def setup_capture(mx: Any, tower_module: Any, model: Any) -> Capture:
    capture = Capture(mx, tower_module, model)
    capture._all_layers = {
        id(layer): layer
        for layer in [*model.tower.shared_layers, *model.tower.residual_layers]
    }
    capture.install()
    return capture


def tensor_record(name: str, shape: list[int], value: Any) -> dict[str, Any]:
    data = np.asarray(value, dtype=np.float32)
    if list(data.shape) != shape:
        raise ValueError(f"{name}: got shape {list(data.shape)}, expected {shape}")
    if not np.isfinite(data).all():
        raise FloatingPointError(f"{name}: non-finite upstream activation")
    return {
        "name": name,
        "shape": shape,
        "dtype": "float32",
        "values": np.ascontiguousarray(data).reshape(-1).tolist(),
    }


def gpu_float_control(mx: Any, left: Any, right: Any) -> tuple[bool, bool]:
    """Compare float32 bit patterns and finiteness on GPU; read back scalars only."""
    if left.shape != right.shape:
        return False, False
    left_f32 = left.astype(mx.float32)
    right_f32 = right.astype(mx.float32)
    equal_bits = mx.all(
        mx.view(left_f32, mx.uint32, stream=mx.gpu)
        == mx.view(right_f32, mx.uint32, stream=mx.gpu),
        stream=mx.gpu,
    )
    finite = mx.all(mx.isfinite(left_f32, stream=mx.gpu), stream=mx.gpu) & mx.all(
        mx.isfinite(right_f32, stream=mx.gpu), stream=mx.gpu
    )
    mx.eval(equal_bits, finite)
    finite_value = bool(finite.item())
    return bool(equal_bits.item()) and finite_value, finite_value


def gpu_int_control(mx: Any, left: Any, right: Any) -> bool:
    """Compare integer outputs on GPU; read back only the scalar predicate."""
    if left.shape != right.shape:
        return False
    equal = mx.all(left == right, stream=mx.gpu)
    mx.eval(equal)
    return bool(equal.item())


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Import the already-pinned loader/replay helpers beside this script.
    from replay_mlx_generation import (
        CORE_PREFIXES,
        MODEL_REVISION as REPLAY_MODEL_REVISION,
        MODEL_REPOSITORY as REPLAY_MODEL_REPOSITORY,
        MLX_REVISION as REPLAY_UPSTREAM_REVISION,
        load_core_weights,
        parameter_names,
        read_generation_report,
        read_inputs,
        selected_core_map,
    )

    del CORE_PREFIXES, parameter_names
    if REPLAY_UPSTREAM_REVISION != UPSTREAM_REVISION:
        raise AssertionError("replay helper's upstream revision changed")
    if REPLAY_MODEL_REPOSITORY != MODEL_REPOSITORY or REPLAY_MODEL_REVISION != MODEL_REVISION:
        raise AssertionError("replay helper's model pin changed")
    replay_helper_path = pathlib.Path(__file__).resolve().with_name("replay_mlx_generation.py")
    replay_helper_sha = sha256_file(replay_helper_path)
    if replay_helper_sha != REPLAY_HELPER_SHA256:
        raise RuntimeError(f"replay helper changed: {replay_helper_sha} != pinned {REPLAY_HELPER_SHA256}")

    mlx_root = args.mlx_root.resolve()
    model_dir = args.model_dir.resolve()
    generation_path = args.generation_report.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing fixture: {output_path}")

    report, report_bytes, report_sha = read_generation_report(generation_path)
    if report_sha != GENERATION_SHA256:
        raise ValueError(f"generation report changed: {report_sha} != pinned {GENERATION_SHA256}")
    if report.get("backend") != "metal_resident" or report.get("prompt_rows") != 87:
        raise ValueError("expected the retained 87-row native metal_resident generation report")

    actual_revision = subprocess.check_output(
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != UPSTREAM_REVISION:
        raise RuntimeError(f"MLX-audio source is {actual_revision}; expected {UPSTREAM_REVISION}")
    dirty = subprocess.check_output(
        ["git", "-C", str(mlx_root), "status", "--porcelain", "--untracked-files=normal"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("pinned MLX-audio worktree has local changes")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config_sha, index_sha = sha256_file(config_path), sha256_file(index_path)
    if config_sha != CONFIG_SHA256 or index_sha != INDEX_SHA256:
        raise RuntimeError("cached model config/index hash differs from the established pin")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    projection_sha = projection_tensor_hash(model_dir, index)
    selected = selected_core_map(index["weight_map"])

    sys.path.insert(0, str(mlx_root))
    import mlx.core as mx
    from mlx_audio.tts.models.fish_omni import fish_omni, tower as tower_module
    from mlx_audio.tts.models.fish_omni.config import ModelConfig
    from mlx_audio.utils import apply_quantization

    mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"failed to select MLX GPU: {mx.default_device()}")
    model_config = ModelConfig.from_dict(config)
    audio = model_config.audio_decoder_config
    prompt_ids, frames, prompt_rows = read_inputs(
        report,
        vocab_size=audio.vocab_size,
        semantic_begin=model_config.semantic_start_token_id,
        semantic_end=model_config.semantic_end_token_id,
        eos_id=model_config.eos_token_id,
        prompt_vocab_size=int(config["text_config"]["vocab_size"]),
    )
    if prompt_rows != 87 or len(frames) < 1 or audio.num_codebooks != 10:
        raise ValueError("generation fixture no longer has the selected tower geometry")

    model = fish_omni.Model(model_config)
    apply_quantization(model, config, selected)
    loaded = load_core_weights(model, model_dir, selected, mx)
    model.eval()
    old_chunk_size = fish_omni.PREFILL_CHUNK_SIZE
    fish_omni.PREFILL_CHUNK_SIZE = 1

    capture = setup_capture(mx, tower_module, model)
    hidden_rows: list[Any] = []
    original_backbone_step = model._backbone_step

    def traced_backbone_step(row: Any, cache: Any) -> Any:
        hidden = original_backbone_step(row, cache)
        # Called once per prompt row because prefill is pinned to chunk size 1.
        if len(hidden_rows) < prompt_rows:
            hidden_rows.append(hidden[0, -1])
        return hidden

    model._backbone_step = traced_backbone_step
    try:
        backbone_cache = model.backbone.make_cache()
        tower_cache = model.tower.make_cache()
        prompt = np.zeros((1, prompt_rows, audio.num_codebooks + 1), dtype=np.int32)
        prompt[0, :, 0] = np.asarray(prompt_ids, dtype=np.int32)
        prompt_mx = mx.array(prompt, dtype=mx.int32)
        final_hidden = model.prefill(prompt_mx, backbone_cache, tower_cache)
        if len(hidden_rows) != prompt_rows:
            raise AssertionError(f"captured {len(hidden_rows)} backbone rows; expected {prompt_rows}")
        if int(tower_cache[0].offset) != prompt_rows - 1:
            raise AssertionError(f"prompt tower cache offset is {tower_cache[0].offset}; expected 86")

        # Decode the first actual report main token from the same prompt cache.
        decode_main = int(frames[0][0])
        previous_values = [prompt_ids[-1]] + [0] * audio.num_codebooks
        previous_frame = mx.array(previous_values, dtype=mx.int32)[None, None, :]
        decoded_codes = model.tower.decode_step(
            mx.array([[decode_main]], dtype=mx.int32),
            final_hidden,
            previous_frame,
            tower_cache,
            temperature=float(report["temperature"]),
            top_k=int(report["top_k"]),
            top_p=float(report["top_p"]),
        )
        decoded_frame = mx.concatenate(
            [mx.array([[[decode_main]]], dtype=mx.int32), decoded_codes], axis=-1
        )
        if int(tower_cache[0].offset) != prompt_rows:
            raise AssertionError(f"decode tower cache offset is {tower_cache[0].offset}; expected 87")

        trace_values = [value for _, (_, value) in capture.arrays.items()]
        mx.eval(*hidden_rows, final_hidden, decoded_frame, *trace_values,
                [cache.state for cache in backbone_cache], [cache.state for cache in tower_cache])
    finally:
        capture.restore()
        model._backbone_step = original_backbone_step
        fish_omni.PREFILL_CHUNK_SIZE = old_chunk_size

    # Re-run only the captured tower rows through the restored upstream
    # methods. This is an observation control for the instrumentation above;
    # it reuses already-captured backbone hidden rows and does not run the
    # backbone a second time.
    control_layer_outputs: dict[str, Any] = {}
    control_phase = "prefill"
    original_layer_call = tower_module.TowerLayer.__call__
    original_sampler = tower_module.sample_top_k_top_p

    def observe_original_layer(layer: Any, hidden: Any, codes: Any, semantic: Any, mask: Any,
                               cache: Any, shared_input: Any = None) -> Any:
        result = original_layer_call(layer, hidden, codes, semantic, mask, cache, shared_input)
        if not layer.shared and layer is model.tower.residual_layers[-1]:
            control_layer_outputs[control_phase] = result
        return result

    tower_module.TowerLayer.__call__ = observe_original_layer
    tower_module.sample_top_k_top_p = lambda logits, temperature, top_k, top_p: mx.argmax(
        logits.astype(mx.float32), axis=-1
    )
    try:
        control_cache = model.tower.make_cache()
        selected_prefill_output = None
        for row in range(prompt_rows - 1):
            hidden_row = mx.array(hidden_rows[row], dtype=mx.float32)[None, None, :]
            result = model.tower.prefill(
                hidden_row,
                prompt_mx[:, row : row + 1],
                prompt_mx[:, row + 1 : row + 2],
                control_cache,
            )
            if row == PREFILL_TOWER_ROW:
                selected_prefill_output = result

        control_decode_cache = control_cache
        control_phase = "decode"
        control_decode_codes = model.tower.decode_step(
            mx.array([[decode_main]], dtype=mx.int32),
            mx.array(hidden_rows[-1], dtype=mx.float32)[None, None, :],
            previous_frame,
            control_decode_cache,
            temperature=float(report["temperature"]),
            top_k=int(report["top_k"]),
            top_p=float(report["top_p"]),
        )
        control_decode_frame = mx.concatenate(
            [mx.array([[[decode_main]]], dtype=mx.int32), control_decode_codes], axis=-1
        )
        mx.eval(selected_prefill_output, control_layer_outputs["prefill"], control_decode_frame)
    finally:
        tower_module.TowerLayer.__call__ = original_layer_call
        tower_module.sample_top_k_top_p = original_sampler

    if selected_prefill_output is None:
        raise AssertionError("unhooked selected prefill control was not produced")
    captured_prefill_output = capture.arrays[
        f"tower.prefill.row{PREFILL_TOWER_ROW}.hidden_output"
    ][1]
    captured_decode_output = capture.arrays["tower.decode.row0.hidden_output"][1]
    control_prefill_bits_equal, control_prefill_finite = gpu_float_control(
        mx, selected_prefill_output, captured_prefill_output
    )
    control_prefill_layer_bits_equal, control_prefill_layer_finite = gpu_float_control(
        mx, control_layer_outputs["prefill"], captured_prefill_output
    )
    control_decode_layer_bits_equal, control_decode_layer_finite = gpu_float_control(
        mx, control_layer_outputs["decode"], captured_decode_output
    )
    control_decode_frame_equal = gpu_int_control(mx, control_decode_frame, decoded_frame)
    if not (
        control_prefill_bits_equal
        and control_prefill_finite
        and control_prefill_layer_bits_equal
        and control_prefill_layer_finite
        and control_decode_frame_equal
    ):
        raise AssertionError(
            "trace instrumentation changed the upstream prefill output or greedy decode frame: "
            f"prefill={control_prefill_bits_equal}, layer={control_prefill_layer_bits_equal}, "
            f"decode_frame={control_decode_frame_equal}"
        )

    traces = [
        tensor_record(name, shape, value)
        for name, (shape, value) in sorted(capture.arrays.items())
    ]
    trace_by_name = {item["name"]: item for item in traces}
    required_prefixes = [
        "tower.prefill.row85.shared_layers.0",
        *[f"tower.decode.row0.residual_layers.{index}" for index in range(9)],
    ]
    for prefix in required_prefixes:
        for suffix in (
            ".ffn_norm",
            ".feed_forward.w1",
            ".feed_forward.w3",
            ".feed_forward.silu",
            ".feed_forward.product",
            ".feed_forward.w2",
            ".ffn_residual",
            ".output",
        ):
            if prefix + suffix not in trace_by_name:
                raise AssertionError(f"missing required tower boundary {prefix + suffix}")
    if not (control_decode_layer_bits_equal and control_decode_layer_finite):
        raise AssertionError("unhooked decode final hidden output differs from instrumented trace")
    if len(hidden_rows) != 87:
        raise AssertionError("prompt hidden row capture count changed")

    cases = []
    for label, weight_prefix, trace_prefix in PAIR_CASES:
        input_trace = trace_by_name[trace_prefix + ".ffn_norm"]
        w1_trace = trace_by_name[trace_prefix + ".feed_forward.w1"]
        w3_trace = trace_by_name[trace_prefix + ".feed_forward.w3"]
        if input_trace["shape"] != [1, 1, 2048] or w1_trace["shape"] != [1, 1, 6144] or w3_trace["shape"] != [1, 1, 6144]:
            raise AssertionError(f"benchmark case geometry changed for {label}")
        cases.append(
            {
                "label": label,
                "weight_prefix": weight_prefix,
                "input": {"shape": [2048], "dtype": "float32", "values": input_trace["values"]},
                "projections": {
                    "w1": {"shape": [6144], "dtype": "float32", "values": w1_trace["values"]},
                    "w3": {"shape": [6144], "dtype": "float32", "values": w3_trace["values"]},
                },
            }
        )

    hidden_export = [
        tensor_record(f"generation.prompt.row{index}.backbone_hidden", [2048], value)["values"]
        for index, value in enumerate(hidden_rows)
    ]
    prompt_frames = [[token, *([0] * audio.num_codebooks)] for token in prompt_ids]
    decoded_values = np.asarray(decoded_frame, dtype=np.int32).reshape(-1).tolist()
    if len(decoded_values) != 11 or any(not isinstance(value, int) for value in decoded_values):
        raise AssertionError("top-1 MLX decode frame has an unexpected shape")

    upstream_files = {
        "tower.py": mlx_root / "mlx_audio/tts/models/fish_omni/tower.py",
        "fish_omni.py": mlx_root / "mlx_audio/tts/models/fish_omni/fish_omni.py",
        "config.py": mlx_root / "mlx_audio/tts/models/fish_omni/config.py",
    }
    fixture = {
        "format": "drama3-tower-pair-fixture-v1",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "upstream_revision": UPSTREAM_REVISION,
        "source_hashes": {
            "config.json": config_sha,
            "model.safetensors.index.json": index_sha,
            "actual_projection_tensors": projection_sha,
            "exporter": sha256_file(pathlib.Path(__file__).resolve()),
            "replay_helper": replay_helper_sha,
            "upstream_files": {name: sha256_file(path) for name, path in upstream_files.items()},
        },
        "provenance": {
            "generation_report": str(generation_path),
            "generation_report_sha256": report_sha,
            "generation_report_backend": report["backend"],
            "prompt_rows": prompt_rows,
            "report_frame_count": len(frames),
            "upstream_repository": "https://github.com/cguldogan/mlx-audio",
            "upstream_revision": UPSTREAM_REVISION,
            "model_path": str(model_dir),
            "loaded_core_tensor_count": loaded,
            "selected_prefill_tower_row": PREFILL_TOWER_ROW,
            "prefill_cache_rows": prompt_rows - 1,
            "decode_policy": "actual first report main token; actual final prompt frame; residual codebooks use MLX argmax(top1), matching native context.argmax",
            "captured_boundaries": "all available non-cache native TowerResident trace_value sites for one final prompt teacher_force row and first decode row, including all layer head_logits",
            "instrumentation_control": {
                "unhooked_prefill_hidden_output_bit_identical": control_prefill_bits_equal,
                "unhooked_prefill_hidden_output_finite": control_prefill_finite,
                "unhooked_prefill_final_layer_bit_identical": control_prefill_layer_bits_equal,
                "unhooked_prefill_final_layer_finite": control_prefill_layer_finite,
                "unhooked_decode_final_layer_bit_identical": control_decode_layer_bits_equal,
                "unhooked_decode_final_layer_finite": control_decode_layer_finite,
                "unhooked_greedy_decode_frame_identical": control_decode_frame_equal,
                "control_backbone_reruns": 0,
            },
        },
        "replay": {
            "prefill_hidden_rows": hidden_export,
            "prompt_frames": prompt_frames,
            "decode_main": decode_main,
            "decode_previous_frame": previous_values,
            "decode_reference_frame": decoded_values,
            "decode_policy": "actual first report main token; MLX argmax on each residual head",
        },
        "cases": cases,
        "traces": traces,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(fixture, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return {
        "output": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "trace_count": len(traces),
        "hidden_rows": len(hidden_export),
        "decoded_frame": decoded_values,
        "projection_sha256": projection_sha,
        "generation_report_sha256": report_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--generation-report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
