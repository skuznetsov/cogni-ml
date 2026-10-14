#!/usr/bin/env python3
"""Export deterministic tiny traces from the pinned DRAMA-3 reference sources.

This creates a small synthetic fixture, not a copy of a released model.  The
MLX text model and Fish-Speech audio tower are instantiated from the pinned
upstream checkouts supplied on the command line and receive generated FP32
weights.  Keeping this script outside those checkouts makes source provenance
and the native-vs-reference boundary explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import pathlib
import platform
import sys
import types
from typing import Any

import numpy as np
import safetensors.numpy


MLX_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
FISH_REVISION = "541566932abbaebcb0a40e562a1f846738383f38"
SEED = 340219

TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 8,
    "intermediate_size": 8,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "rms_norm_eps": 1e-6,
    "vocab_size": 32,
    "max_position_embeddings": 8,
    "linear_num_value_heads": 2,
    "linear_num_key_heads": 1,
    "linear_key_head_dim": 4,
    "linear_value_head_dim": 4,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "shared_expert_intermediate_size": 8,
    "moe_intermediate_size": 8,
    "norm_topk_prob": True,
    # Partial rotary width 2 is the smallest positive even rotary width.
    "rope_parameters": {
        "type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
    },
}

TOWER_CONFIG = {
    "dim": 8,
    "intermediate_size": 8,
    "n_head": 2,
    "n_local_heads": 1,
    "head_dim": 4,
    "norm_eps": 1e-6,
    "rope_base": 10000.0,
    "rotary_percent": 0.5,
    "vocab_size": 4,
    "num_codebooks": 4,
    "shared_enrichment_layers": 1,
    "vq_backbone_codebooks": 1,
    "n_layer": 1,
    "decoder_type": "codebook_mtp_shared",
    "rotary_interleaved": True,
    "attention_output_gate": True,
    "layernorm_zero_centered_gamma": True,
    "use_moe": False,
    "mtp_use_swa": False,
    "attention_qkv_bias": False,
    "attention_o_bias": False,
    "dropout": 0.0,
    "vq_feature_placement": "backbone_partial",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return np.asarray(values, dtype=np.float32).reshape(-1)


def record(traces: list[dict[str, Any]], name: str, shape: list[int], value: Any) -> None:
    data = flatten(value)
    expected = math.prod(shape)
    if data.size != expected:
        raise ValueError(f"{name}: {data.size} values do not fit shape {shape}")
    if not np.isfinite(data).all():
        raise FloatingPointError(f"{name}: non-finite reference activation")
    traces.append(
        {
            "name": name,
            "shape": shape,
            "dtype": "float32",
            "values": data.tolist(),
        }
    )


def deterministic_array(rng: np.random.Generator, shape: tuple[int, ...], name: str) -> np.ndarray:
    if name.endswith(".A_log"):
        return rng.normal(-0.15, 0.025, shape).astype(np.float32)
    if name.endswith(".dt_bias"):
        return rng.normal(0.05, 0.025, shape).astype(np.float32)
    if any(token in name for token in ("norm.weight", "enorm", "hnorm", "final_layernorm")):
        if name.startswith("tower."):
            return rng.normal(0.0, 0.025, shape).astype(np.float32)
        return rng.normal(1.0, 0.025, shape).astype(np.float32)
    return rng.normal(0.0, 0.075, shape).astype(np.float32)


def ensure_source_pin(root: pathlib.Path, revision: str) -> None:
    import subprocess

    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise RuntimeError(f"{root} is at {actual}, expected pinned revision {revision}")


def export_backbone(mlx_root: pathlib.Path, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[list[float]], list[list[float]]]:
    sys.path.insert(0, str(mlx_root))
    from mlx.utils import tree_flatten, tree_unflatten
    import mlx.core as mx
    from mlx_audio.lm.models import qwen3_5
    from mlx_audio.lm.models.qwen3_5 import Qwen3_5TextModel, TextModelArgs

    source = inspect.getsource(qwen3_5.DecoderLayer.__call__)
    original_layer_call = qwen3_5.DecoderLayer.__call__
    required = (
        "r = self.linear_attn(self.input_layernorm(x), mask, cache)",
        "r = self.self_attn(self.input_layernorm(x), mask, cache)",
        "h = x + r",
        "out = h + self.mlp(self.post_attention_layernorm(h))",
    )
    if not all(fragment in source for fragment in required):
        raise RuntimeError("pinned DecoderLayer implementation no longer matches trace wrapper")

    args = TextModelArgs(**TEXT_CONFIG)
    model = Qwen3_5TextModel(args)
    original = tree_flatten(model.parameters())
    prepared: list[tuple[str, Any]] = []
    weights: dict[str, np.ndarray] = {}
    for name, value in original:
        weight = deterministic_array(rng, tuple(value.shape), f"backbone.{name}")
        prepared.append((name, mx.array(weight)))
        weights[f"backbone.{name}"] = weight
    model.update(tree_unflatten(prepared))

    embeddings = rng.normal(0.0, 0.3, (3, TEXT_CONFIG["hidden_size"])).astype(np.float32)
    traces: list[dict[str, Any]] = []
    layer_ids = {id(layer): index for index, layer in enumerate(model.layers)}

    # This wrapper exposes the two residual boundaries in the exact pinned
    # DecoderLayer sequence while continuing to call its actual MLX submodules.
    def traced_layer(self: Any, x: Any, mask: Any = None, cache: Any = None) -> Any:
        if self.is_linear:
            residual = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            residual = self.self_attn(self.input_layernorm(x), mask, cache)
        attention_residual = x + residual
        index = layer_ids[id(self)]
        row = current_row[0]
        record(
            traces,
            f"backbone.row{row}.layers.{index}.attention_residual",
            [TEXT_CONFIG["hidden_size"]],
            attention_residual,
        )
        hidden = attention_residual + self.mlp(
            self.post_attention_layernorm(attention_residual)
        )
        record(
            traces,
            f"backbone.row{row}.layers.{index}.hidden",
            [TEXT_CONFIG["hidden_size"]],
            hidden,
        )
        return hidden

    current_row = [0]
    qwen3_5.DecoderLayer.__call__ = traced_layer
    cache = model.make_cache()
    outputs: list[list[float]] = []
    for row in range(len(embeddings)):
        current_row[0] = row
        output = model(
            inputs=mx.zeros((1, 1), dtype=mx.int32),
            cache=cache,
            input_embeddings=mx.array(embeddings[row][None, None, :]),
        )
        mx.eval(output)
        record(
            traces,
            f"backbone.row{row}.hidden",
            [TEXT_CONFIG["hidden_size"]],
            output,
        )
        outputs.append(np.asarray(output).reshape(-1).astype(np.float32).tolist())
    # Restore the pinned implementation for any later in-process use.
    qwen3_5.DecoderLayer.__call__ = original_layer_call
    return weights, traces, embeddings.tolist(), outputs


def build_torch_config(config_module: Any) -> Any:
    from types import SimpleNamespace

    decoder = config_module.AudioDecoderConfig(
        dim=TOWER_CONFIG["dim"],
        intermediate_size=TOWER_CONFIG["intermediate_size"],
        n_head=TOWER_CONFIG["n_head"],
        n_local_heads=TOWER_CONFIG["n_local_heads"],
        head_dim=TOWER_CONFIG["head_dim"],
        norm_eps=TOWER_CONFIG["norm_eps"],
        rope_base=TOWER_CONFIG["rope_base"],
        rotary_percent=TOWER_CONFIG["rotary_percent"],
        vocab_size=TOWER_CONFIG["vocab_size"],
        num_codebooks=TOWER_CONFIG["num_codebooks"],
        shared_enrichment_layers=TOWER_CONFIG["shared_enrichment_layers"],
        vq_backbone_codebooks=TOWER_CONFIG["vq_backbone_codebooks"],
    )
    return SimpleNamespace(
        audio_decoder=decoder,
        semantic_start_token_id=20,
        semantic_end_token_id=23,
        eos_token_id=24,
    )


def export_tower(fish_root: pathlib.Path, rng: np.random.Generator, hidden_rows: list[list[float]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    sys.path.insert(0, str(fish_root))
    try:
        import loguru  # noqa: F401
    except ImportError:
        loguru = types.ModuleType("loguru")
        loguru.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
        sys.modules["loguru"] = loguru
    import torch
    import fish_speech.models.text2semantic.fish_omni.tower as tower_module
    from fish_speech.models.text2semantic.fish_omni import config as fish_config
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder,
        TowerAttention,
        TowerLayer,
        TowerMLP,
        TowerTransformerLayer,
        load_audio_decoder_weights,
    )
    prefill_source = inspect.getsource(FishOmniAudioDecoder.prefill)
    decode_source = inspect.getsource(FishOmniAudioDecoder.decode_step)
    if not all(fragment in prefill_source for fragment in ("self.run_shared", "self.residual_layers", "positions")):
        raise RuntimeError("pinned FishOmniAudioDecoder.prefill no longer matches trace wrapper")
    if not all(fragment in decode_source for fragment in ("self.run_shared", "self.residual_layers", "self.codebook_heads")):
        raise RuntimeError("pinned FishOmniAudioDecoder.decode_step no longer matches trace wrapper")

    torch.set_num_threads(1)
    model = FishOmniAudioDecoder(build_torch_config(fish_config), dtype=torch.float32, device="cpu").eval()
    checkpoint_prefixes = (
        ("codebook_embeddings.", "audio_decoder.codebook_embeddings."),
        ("shared_layers.", "audio_decoder_mtp_shared.layers."),
        ("residual_layers.", "audio_decoder_mtp.layers."),
        ("codebook_heads.", "audio_decoder_mtp.heads."),
    )
    checkpoint = []
    target_to_native: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        value = deterministic_array(rng, tuple(parameter.shape), f"tower.{name}")
        target = next((prefix + name[len(stem):] for stem, prefix in checkpoint_prefixes if name.startswith(stem)), None)
        if target is None:
            raise RuntimeError(f"no official tower checkpoint prefix for {name}")
        checkpoint.append((target, torch.from_numpy(value.copy())))
        target_to_native[name] = f"tower.{name}"
    # Exercise the actual official checkpoint mapper before saving native-facing names.
    load_audio_decoder_weights(model, checkpoint)
    weights = {
        target_to_native[name]: parameter.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, parameter in model.named_parameters()
    }

    traces: list[dict[str, Any]] = []
    def trace(name: str, value: Any, shape: list[int] | None = None) -> None:
        tensor = value.detach() if hasattr(value, "detach") else value
        actual_shape = shape or list(tensor.shape)
        # Fish-Speech uses [T,D] inside a one-row decode; the Crystal callback
        # preserves the corresponding singleton batch/time axes as [1,1,D].
        if shape is None and tensor.ndim == 2 and tensor.shape[0] == 1:
            actual_shape = [1, 1, int(tensor.shape[-1])]
        record(traces, name, actual_shape, tensor)

    # Guard the stage-aware wrappers against upstream source drift.
    layer_source = inspect.getsource(TowerLayer.forward)
    transformer_source = inspect.getsource(TowerTransformerLayer.forward)
    attention_source = inspect.getsource(TowerAttention.forward)
    mlp_source = inspect.getsource(TowerMLP.forward)
    for source, fragments, label in (
        (layer_source, ("self.enorm(e)", "self.hnorm(hidden)", "self.eh_proj", "self.transformer_layer", "self.final_layernorm"), "TowerLayer"),
        (transformer_source, ("self.attention_norm", "self.attention(", "self.ffn_norm", "self.feed_forward"), "TowerTransformerLayer"),
        (attention_source, ("self.wqkv(x)", "self.q_norm", "self.k_norm", "self.rotate", "grouped_attention", "self.wo"), "TowerAttention"),
        (mlp_source, ("self.w1(x)", "self.w3(x)", "self.w2("), "TowerMLP"),
    ):
        if not all(fragment in source for fragment in fragments):
            raise RuntimeError(f"pinned {label} implementation no longer matches trace wrapper")

    def tower_layer(self: Any, hidden: Any, codes: Any, semantic: Any, positions: Any, shared_input: Any = None) -> Any:
        embedding_module = self.semantic_embedding if self.shared else self.codebook_embedding
        codes = codes.clamp(0, embedding_module.num_embeddings - 1)
        embedding = torch.where(semantic[:, None], embedding_module(codes), 0.0)
        if shared_input is not None:
            embedding = embedding + shared_input
        prefix = active_prefix[0]
        index = active_layer[0]
        part = "shared_layers" if self.shared else "residual_layers"
        name = f"{prefix}.{part}.{index}"
        trace(f"{name}.embedding", embedding)
        enorm = self.enorm(embedding)
        hnorm = self.hnorm(hidden)
        trace(f"{name}.enorm", enorm)
        trace(f"{name}.hnorm", hnorm)
        concat = torch.cat((enorm, hnorm), dim=-1)
        trace(f"{name}.eh_concat", concat)
        mixed = self.eh_proj(concat)
        trace(f"{name}.eh_proj", mixed)
        transformed = self.transformer_layer(mixed, positions)
        output = self.final_layernorm(transformed)
        trace(f"{name}.output", output)
        active_layer[0] += 1
        return output.to(mixed.dtype)

    def transformer_layer(self: Any, x: Any, positions: Any) -> Any:
        residual = x.float() if self.attention.wqkv.weight.dtype == torch.float16 else x
        prefix = active_prefix[0]
        name = active_layer_name[0]
        attention_input = self.attention_norm(residual)
        trace(f"{name}.attention_norm", attention_input)
        attention = self.attention(attention_input.to(self.attention.wqkv.weight.dtype), positions)
        residual = residual + attention
        trace(f"{name}.attention_residual", residual)
        ffn_input = self.ffn_norm(residual)
        trace(f"{name}.ffn_norm", ffn_input)
        feed = self.feed_forward(ffn_input.to(self.attention.wqkv.weight.dtype))
        residual = residual + feed
        trace(f"{name}.ffn_residual", residual)
        return residual

    def tower_attention(self: Any, x: Any, positions: Any) -> Any:
        cfg = self.config
        length = len(positions)
        q_width, kv_width = cfg.n_head * cfg.head_dim, cfg.n_local_heads * cfg.head_dim
        q, gate, k, v = self.wqkv(x).split((q_width, q_width, kv_width, kv_width), -1)
        prefix, name = active_prefix[0], active_layer_name[0] + ".attention"
        trace(f"{name}.q_projected", q.reshape(1, length, cfg.n_head, cfg.head_dim))
        trace(f"{name}.gate", gate.reshape(1, length, cfg.n_head, cfg.head_dim))
        trace(f"{name}.k_projected", k.reshape(1, length, cfg.n_local_heads, cfg.head_dim))
        trace(f"{name}.v_projected", v.reshape(1, length, cfg.n_local_heads, cfg.head_dim))
        q = self.q_norm(q.reshape(length, cfg.n_head, cfg.head_dim))
        k = self.k_norm(k.reshape(length, cfg.n_local_heads, cfg.head_dim))
        trace(f"{name}.q_norm", q.transpose(0, 1).unsqueeze(0))
        trace(f"{name}.k_norm", k.transpose(0, 1).unsqueeze(0))
        q = self.rotate(q, positions).transpose(0, 1).unsqueeze(0)
        k = self.rotate(k, positions).transpose(0, 1).unsqueeze(0)
        v = v.reshape(length, cfg.n_local_heads, cfg.head_dim).transpose(0, 1).unsqueeze(0)
        trace(f"{name}.q_rope", q)
        trace(f"{name}.k_rope", k)
        if self.k_cache is not None:
            if self.static_kv_len is not None:
                end = self.static_kv_len
            else:
                end = int(positions[-1]) + 1
                assert end <= self.k_cache.shape[2], "tower cache capacity exceeded"
            self.k_cache.index_copy_(2, positions, k)
            self.v_cache.index_copy_(2, positions, v)
            k, v = self.k_cache[:, :, :end], self.v_cache[:, :, :end]
            key_positions = torch.arange(end, device=x.device)
        else:
            key_positions = positions
        trace(f"{name}.key_cache", k)
        trace(f"{name}.value_cache", v)
        mask = key_positions[None, :] <= positions[:, None]
        context = tower_module.grouped_attention(q, k, v, mask)
        context = context.squeeze(0).transpose(0, 1).reshape(length, q_width)
        gated = context * gate.sigmoid()
        trace(f"{name}.gated_context", gated.reshape(1, length, q_width))
        output = self.wo(gated)
        trace(f"{name}.output", output.reshape(1, length, cfg.dim))
        return output

    def tower_mlp(self: Any, x: Any) -> Any:
        up = self.w1(x)
        gate = self.w3(x)
        silu = torch.nn.functional.silu(up)
        product = silu * gate
        prefix = active_layer_name[0] + ".feed_forward"
        trace(f"{prefix}.w1", up)
        trace(f"{prefix}.w3", gate)
        trace(f"{prefix}.silu", silu)
        trace(f"{prefix}.product", product)
        output = self.w2(product)
        trace(f"{prefix}.w2", output)
        return output

    active_prefix = [""]
    active_layer = [0]
    active_layer_name = [""]

    original_layer_forward = TowerLayer.forward
    original_transformer_forward = TowerTransformerLayer.forward
    original_attention_forward = TowerAttention.forward
    original_mlp_forward = TowerMLP.forward
    TowerLayer.forward = tower_layer
    TowerTransformerLayer.forward = transformer_layer
    TowerAttention.forward = tower_attention
    TowerMLP.forward = tower_mlp

    # Explicit stage counters identify the current module index for nested layers.
    # run_shared and decode_step visit shared index 0 before residual index 0;
    # the layer wrapper gets the official ModuleList index from this mapping.
    layer_indices = {id(layer): i for i, layer in enumerate(model.shared_layers)}
    layer_indices.update({id(layer): i for i, layer in enumerate(model.residual_layers)})
    last_decode_state: list[Any] = [None]
    # Wrap the layer entry to select the right layer name around each official call.
    active_layer[0] = 0
    # Python hook-free approach: infer index from embedding module identity.
    old_tower_layer = TowerLayer.forward
    def named_layer(self: Any, hidden: Any, codes: Any, semantic: Any, positions: Any, shared_input: Any = None) -> Any:
        active_layer[0] = layer_indices[id(self)]
        active_layer_name[0] = f"{active_prefix[0]}.{'shared_layers' if self.shared else 'residual_layers'}.{active_layer[0]}"
        result = old_tower_layer(self, hidden, codes, semantic, positions, shared_input)
        if active_prefix[0].startswith("tower.decode.") and not self.shared:
            last_decode_state[0] = result
        return result
    TowerLayer.forward = named_layer

    inputs = {
        "input_embeddings": [
            [0.18, -0.13, 0.24, -0.05, 0.31, 0.07, -0.19, 0.11],
            [-0.08, 0.21, 0.03, 0.27, -0.14, 0.16, 0.05, -0.22],
            [0.04, -0.17, 0.29, 0.12, 0.08, -0.25, 0.19, 0.02],
        ],
        "prefill_previous_frame": [21, 1, 2, 0, 3],
        "prefill_next_frame": [22, 2, 1, 3, 0],
        "decode_previous_frame": [22, 2, 1, 3, 0],
        "decode_main": 21,
        "sample_policy": "argmax",
    }
    hidden = torch.tensor(hidden_rows[0], dtype=torch.float32).reshape(1, -1)
    decode_hidden = torch.tensor(hidden_rows[1], dtype=torch.float32).reshape(1, -1)
    previous = torch.tensor([inputs["prefill_previous_frame"]], dtype=torch.long)
    next_frame = torch.tensor([inputs["prefill_next_frame"]], dtype=torch.long)
    model.setup_caches(max_seq_len=4)
    active_prefix[0] = "tower.prefill.row0"
    active_layer[0] = 0
    record(traces, "tower.prefill.row0.hidden_input", [1, 1, TOWER_CONFIG["dim"]], hidden)
    semantic = model.is_semantic(next_frame[:, 0])
    unseen = model.embedding_sum(
        previous,
        1 + TOWER_CONFIG["vq_backbone_codebooks"],
        TOWER_CONFIG["num_codebooks"],
    )
    unseen = torch.where(model.is_semantic(previous[:, 0])[:, None], unseen, 0.0)
    record(traces, "tower.prefill.row0.unseen_embedding", [1, 1, TOWER_CONFIG["dim"]], unseen)
    with torch.no_grad():
        # The official prefill updates the same cache/layer state and discards
        # logits. The additional head reads make teacher-forced logits visible.
        positions = torch.tensor([0], dtype=torch.long)
        state = model.run_shared(hidden, next_frame[:, 1], semantic, positions, previous)
        for index, layer in enumerate(model.residual_layers):
            state = layer(state, next_frame[:, index + 1], semantic, positions)
            logits = model.codebook_heads[index](state)
            record(traces, f"tower.prefill.row0.head_logits.{index}", [1, TOWER_CONFIG["vocab_size"]], logits)
        record(traces, "tower.prefill.row0.hidden_output", [1, 1, TOWER_CONFIG["dim"]], state)

        # Decode chooses the desired main token and each desired residual code
        # by deterministic argmax. Official decode_step owns the inference path.
        sample_index = [0]
        original_sampler = tower_module.sample_top_k_top_p
        def deterministic_sampler(logits: Any, temperature: Any, top_k: Any, top_p: Any, *, check_finite: bool = True) -> Any:
            call_index = sample_index[0]
            sample_index[0] += 1
            if call_index >= 2:
                head_index = call_index - 2
                record(
                    traces,
                    f"tower.decode.row0.head_logits.{head_index}",
                    [1, TOWER_CONFIG["vocab_size"]],
                    logits,
                )
            return logits.argmax(-1)
        tower_module.sample_top_k_top_p = deterministic_sampler
        main_logits = torch.full((1, 32), -1000.0, dtype=torch.float32)
        main_logits[0, inputs["decode_main"]] = 1000.0
        active_prefix[0] = "tower.decode.row0"
        active_layer[0] = 0
        record(traces, "tower.decode.row0.hidden_input", [1, 1, TOWER_CONFIG["dim"]], decode_hidden)
        decode_previous = torch.tensor([inputs["decode_previous_frame"]], dtype=torch.long)
        decode_unseen = model.embedding_sum(
            decode_previous,
            1 + TOWER_CONFIG["vq_backbone_codebooks"],
            TOWER_CONFIG["num_codebooks"],
        )
        decode_unseen = torch.where(model.is_semantic(decode_previous[:, 0])[:, None], decode_unseen, 0.0)
        record(traces, "tower.decode.row0.unseen_embedding", [1, 1, TOWER_CONFIG["dim"]], decode_unseen)
        decode_result = model.decode_step(
            main_logits,
            decode_hidden,
            torch.tensor([1], dtype=torch.long),
            torch.tensor([inputs["decode_previous_frame"]], dtype=torch.long),
            torch.tensor([[20]], dtype=torch.long),
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            check_finite=True,
        )
        tower_module.sample_top_k_top_p = original_sampler
        inputs["reference_decoded_frame"] = decode_result.reshape(-1).tolist()
        record(
            traces,
            "tower.decode.row0.hidden_output",
            [1, 1, TOWER_CONFIG["dim"]],
            last_decode_state[0],
        )

    TowerLayer.forward = original_layer_forward
    TowerTransformerLayer.forward = original_transformer_forward
    TowerAttention.forward = original_attention_forward
    TowerMLP.forward = original_mlp_forward
    return weights, traces, inputs


def add_quant_fixtures(weights: dict[str, np.ndarray]) -> dict[str, Any]:
    # Keep this fixture tied to the upstream representation rather than a
    # second hand-written packer.  In particular, MLX stores several low-bit
    # values in each UInt32 word (8 per word for q4, 4 for q8).
    import mlx.core as mx

    expected: dict[str, Any] = {}
    for bits in (4, 8):
        rows, cols = 2, 64
        source = np.stack(
            [
                np.linspace(-0.7, 0.8, cols, dtype=np.float32),
                np.cos(np.linspace(-2.0, 2.0, cols, dtype=np.float32)).astype(np.float32) * 0.4,
            ]
        )
        packed, scales, biases = mx.quantize(
            mx.array(source), group_size=cols, bits=bits, mode="affine"
        )
        dequantized = mx.dequantize(
            packed, scales, biases, group_size=cols, bits=bits
        )
        x = np.linspace(-0.5, 0.5, cols, dtype=np.float32)
        product = mx.quantized_matmul(
            mx.array(x[None, :]),
            packed,
            scales,
            biases,
            transpose=True,
            group_size=cols,
            bits=bits,
            mode="affine",
        )
        mx.eval(packed, scales, biases, dequantized, product)
        name = f"affine_fixture.q{bits}"
        weights[f"{name}.weight"] = np.asarray(packed, dtype=np.uint32)
        weights[f"{name}.scales"] = np.asarray(scales, dtype=np.float32)
        weights[f"{name}.biases"] = np.asarray(biases, dtype=np.float32)
        expected[name] = {
            "bits": bits,
            "group_size": 64,
            "input": x.tolist(),
            "dense": np.asarray(dequantized, dtype=np.float32).tolist(),
            "matvec": np.asarray(product, dtype=np.float32).reshape(-1).tolist(),
        }
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-root", type=pathlib.Path, required=True)
    parser.add_argument("--fish-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.mlx_root = args.mlx_root.resolve()
    args.fish_root = args.fish_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_source_pin(args.mlx_root, MLX_REVISION)
    ensure_source_pin(args.fish_root, FISH_REVISION)
    rng = np.random.default_rng(SEED)

    backbone_weights, backbone_traces, embeddings, backbone_outputs = export_backbone(args.mlx_root, rng)
    tower_weights, tower_traces, tower_inputs = export_tower(args.fish_root, rng, backbone_outputs)
    weights = {**backbone_weights, **tower_weights}
    quant_expected = add_quant_fixtures(weights)
    config = {
        "model_type": "drama3_tiny_reference_fixture",
        "text_config": TEXT_CONFIG,
        "audio_decoder_config": TOWER_CONFIG,
        "semantic_start_token_id": 20,
        "semantic_end_token_id": 23,
        "eos_token_id": 24,
        "quantization": {
            "affine_fixture.q4": {"bits": 4, "group_size": 64},
            "affine_fixture.q8": {"bits": 8, "group_size": 64},
        },
    }
    inputs = {
        **tower_inputs,
        "input_embeddings": embeddings,
        # These are the actual final-normalized MLX backbone rows passed to
        # the pinned PyTorch tower oracle. Keeping them in the bundle lets a
        # native tower trace reproduce the identical sensitive inputs.
        "backbone_hidden_rows": backbone_outputs,
        "tower_hidden_rows": backbone_outputs,
    }
    traces = [*backbone_traces, *tower_traces]
    traces.sort(key=lambda item: item["name"])
    safetensors.numpy.save_file(weights, args.output / "model.safetensors", metadata={"format": "pt"})
    (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.output / "inputs.json").write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")
    (args.output / "reference_traces.json").write_text(
        json.dumps({"format": "drama3-reference-traces-v1", "traces": traces}, separators=(",", ":")) + "\n"
    )
    (args.output / "quantization_expected.json").write_text(json.dumps(quant_expected, indent=2, sort_keys=True) + "\n")

    source_files = {
        "mlx_qwen3_5": args.mlx_root / "mlx_audio/lm/models/qwen3_5.py",
        "fish_tower": args.fish_root / "fish_speech/models/text2semantic/fish_omni/tower.py",
        "fish_layers": args.fish_root / "fish_speech/models/text2semantic/fish_omni/layers.py",
        "fish_config": args.fish_root / "fish_speech/models/text2semantic/fish_omni/config.py",
    }
    from importlib.metadata import version
    import torch

    manifest = {
        "fixture_kind": "deterministic_synthetic_tiny_reference",
        "seed": SEED,
        "not_released_checkpoint_weights": True,
        "upstream": {
            "mlx-audio": {
                "repository": "https://github.com/cguldogan/mlx-audio",
                "revision": MLX_REVISION,
            },
            "fish-speech": {
                "repository": "https://github.com/fishaudio/fish-speech",
                "revision": FISH_REVISION,
            },
        },
        "reference_files": {
            name: {
                "path": path.relative_to(args.mlx_root if name == "mlx_qwen3_5" else args.fish_root).as_posix(),
                "sha256": sha256(path),
            }
            for name, path in source_files.items()
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mlx": version("mlx"),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
        },
        "artifacts_sha256": {
            name: sha256(args.output / name)
            for name in ("model.safetensors", "config.json", "inputs.json", "reference_traces.json", "quantization_expected.json")
        },
        "reference_trace_count": len(traces),
        "reference_trace_names": [item["name"] for item in traces],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
