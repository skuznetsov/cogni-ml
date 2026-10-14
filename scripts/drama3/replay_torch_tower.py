#!/usr/bin/env python3
"""Replay selected checkpoint tower weights through pinned Fish-Speech PyTorch.

The input bundle is produced alongside native Q4 traces. Only indexed
``tower.*`` tensors are read from the two relevant checkpoint shards; the
backbone and codec are never instantiated. This model's tower weights are
stored as BF16 dense tensors, so the official Fish-Speech loader maps them to
its FP32 reference module without a checkpoint-wide conversion.
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
import types
from typing import Any, Iterable

import numpy as np


FISH_REVISION = "541566932abbaebcb0a40e562a1f846738383f38"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
TRACE_FORMAT = "drama3-reference-traces-v1"
CHECKPOINT_PREFIXES = (
    ("codebook_embeddings.", "audio_decoder.codebook_embeddings."),
    ("shared_layers.", "audio_decoder_mtp_shared.layers."),
    ("residual_layers.", "audio_decoder_mtp.layers."),
    ("codebook_heads.", "audio_decoder_mtp.heads."),
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_tower_map(weight_map: dict[str, str]) -> dict[str, str]:
    selected = {name: shard for name, shard in weight_map.items() if name.startswith("tower.")}
    if not selected:
        raise ValueError("checkpoint index contains no tower.* tensors")
    return selected


def checkpoint_name(name: str) -> str:
    if not name.startswith("tower."):
        raise ValueError(f"not a tower tensor: {name}")
    native_name = name[len("tower."):]
    target = next(
        (prefix + native_name[len(stem):] for stem, prefix in CHECKPOINT_PREFIXES if native_name.startswith(stem)),
        None,
    )
    if target is None:
        raise ValueError(f"no official Fish-Speech checkpoint prefix for {name}")
    return target


def record(traces: list[dict[str, Any]], name: str, shape: list[int], value: Any) -> None:
    tensor = value.detach() if hasattr(value, "detach") else value
    data = np.asarray(tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor, dtype=np.float32).reshape(-1)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{name}: {data.size} values do not fit shape {shape}")
    if not np.isfinite(data).all():
        raise FloatingPointError(f"{name}: non-finite reference activation")
    traces.append({"name": name, "shape": shape, "dtype": "float32", "values": data.tolist()})


def trace_shape(value: Any) -> list[int]:
    shape = list(value.shape)
    if len(shape) == 2 and shape[0] == 1:
        return [1, 1, int(shape[-1])]
    return [int(dim) for dim in shape]


def load_selected_weights(
    model: Any,
    model_dir: pathlib.Path,
    selected: dict[str, str],
    torch: Any,
) -> None:
    """Feed only selected tower tensors to the pinned official mapper."""
    from safetensors import safe_open
    from fish_speech.models.text2semantic.fish_omni.tower import load_audio_decoder_weights

    expected = dict(model.named_parameters())
    mapped = {checkpoint_name(name) for name in selected}
    expected_checkpoint_names = set()
    for name in expected:
        prefix = next((old for old, _ in CHECKPOINT_PREFIXES if name.startswith(old)), None)
        if prefix is None:
            raise ValueError(f"official parameter has no checkpoint prefix: {name}")
        official = next(new for old, new in CHECKPOINT_PREFIXES if old == prefix)
        expected_checkpoint_names.add(official + name[len(prefix):])
    if mapped != expected_checkpoint_names:
        raise ValueError(
            "indexed tower weights differ from official parameters: "
            f"missing={sorted(expected_checkpoint_names - mapped)[:5]}, "
            f"extra={sorted(mapped - expected_checkpoint_names)[:5]}"
        )

    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)

    def weights() -> Iterable[tuple[str, Any]]:
        loaded = 0
        for shard in sorted(by_shard):
            shard_path = model_dir / shard
            with safe_open(str(shard_path), framework="pt", device="cpu") as tensors:
                for name in sorted(by_shard[shard]):
                    value = tensors.get_tensor(name)
                    if value.dtype != torch.bfloat16:
                        raise TypeError(f"{name}: expected checkpoint BF16, found {value.dtype}")
                    # Preserve the exact BF16 value while satisfying the official
                    # FP32 CPU module and avoiding NumPy's unsupported BF16 dtype.
                    yield checkpoint_name(name), value.float()
                    loaded += 1
                    del value
            print(f"read {loaded}/{len(selected)} tower tensors from {shard}", file=sys.stderr)
        if loaded != len(selected):
            raise AssertionError(f"read {loaded} tensors, expected {len(selected)}")

    load_audio_decoder_weights(model, weights())


def run_replay(
    fish_root: pathlib.Path,
    model_dir: pathlib.Path,
    inputs_path: pathlib.Path,
    output_path: pathlib.Path,
    hash_shards: bool = False,
) -> dict[str, Any]:
    import torch
    from safetensors import __version__ as safetensors_version

    sys.path.insert(0, str(fish_root))
    try:
        import loguru  # noqa: F401
    except ImportError:
        loguru = types.ModuleType("loguru")
        loguru.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
        sys.modules["loguru"] = loguru

    from fish_speech.models.text2semantic.fish_omni import config as fish_config
    import fish_speech.models.text2semantic.fish_omni.tower as tower_module
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder,
        TowerAttention,
        TowerLayer,
        TowerMLP,
        TowerTransformerLayer,
    )

    actual_revision = subprocess.check_output(
        ["git", "-C", str(fish_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != FISH_REVISION:
        raise RuntimeError(f"Fish-Speech source is {actual_revision}; expected {FISH_REVISION}")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    selected = selected_tower_map(index["weight_map"])
    audio_config = fish_config.AudioDecoderConfig.from_dict(config_data["audio_decoder_config"])
    model_config = types.SimpleNamespace(
        audio_decoder=audio_config,
        semantic_start_token_id=int(config_data["semantic_start_token_id"]),
        semantic_end_token_id=int(config_data["semantic_end_token_id"]),
        eos_token_id=int(config_data["eos_token_id"]),
    )
    torch.set_num_threads(1)
    model = FishOmniAudioDecoder(model_config, dtype=torch.float32, device="cpu").eval()
    load_selected_weights(model, model_dir, selected, torch)

    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    hidden_rows = np.asarray(inputs["tower_hidden_rows"], dtype=np.float32)
    expected_hidden_shape = (2, audio_config.dim)
    if hidden_rows.shape != expected_hidden_shape:
        raise ValueError(f"tower_hidden_rows has shape {hidden_rows.shape}; expected {expected_hidden_shape}")
    if not np.isfinite(hidden_rows).all():
        raise FloatingPointError("tower_hidden_rows contain non-finite values")

    traces: list[dict[str, Any]] = []
    active_prefix = [""]
    active_layer = [0]
    active_layer_name = [""]
    last_decode_state: list[Any] = [None]

    def trace(name: str, value: Any, shape: list[int] | None = None) -> None:
        record(traces, name, shape or trace_shape(value), value)

    def tower_layer(self: Any, hidden: Any, codes: Any, semantic: Any, positions: Any, shared_input: Any = None) -> Any:
        embedding_module = self.semantic_embedding if self.shared else self.codebook_embedding
        codes = codes.clamp(0, embedding_module.num_embeddings - 1)
        embedding = torch.where(semantic[:, None], embedding_module(codes), 0.0)
        if shared_input is not None:
            embedding = embedding + shared_input
        prefix = active_layer_name[0]
        trace(f"{prefix}.embedding", embedding)
        enorm = self.enorm(embedding)
        hnorm = self.hnorm(hidden)
        trace(f"{prefix}.enorm", enorm)
        trace(f"{prefix}.hnorm", hnorm)
        combined = torch.cat((enorm, hnorm), dim=-1)
        trace(f"{prefix}.eh_concat", combined)
        mixed = self.eh_proj(combined)
        trace(f"{prefix}.eh_proj", mixed)
        transformed = self.transformer_layer(mixed, positions)
        output = self.final_layernorm(transformed)
        trace(f"{prefix}.output", output)
        if active_prefix[0].startswith("tower.decode.") and not self.shared:
            last_decode_state[0] = output
        return output.to(mixed.dtype)

    def transformer_layer(self: Any, x: Any, positions: Any) -> Any:
        residual = x.float() if self.attention.wqkv.weight.dtype == torch.float16 else x
        prefix = active_layer_name[0]
        attention_input = self.attention_norm(residual)
        trace(f"{prefix}.attention_norm", attention_input)
        attention = self.attention(attention_input.to(self.attention.wqkv.weight.dtype), positions)
        residual = residual + attention
        trace(f"{prefix}.attention_residual", residual)
        ffn_input = self.ffn_norm(residual)
        trace(f"{prefix}.ffn_norm", ffn_input)
        feed = self.feed_forward(ffn_input.to(self.attention.wqkv.weight.dtype))
        residual = residual + feed
        trace(f"{prefix}.ffn_residual", residual)
        return residual

    def tower_attention(self: Any, x: Any, positions: Any) -> Any:
        cfg = self.config
        length = len(positions)
        q_width, kv_width = cfg.n_head * cfg.head_dim, cfg.n_local_heads * cfg.head_dim
        q, gate, k, v = self.wqkv(x).split((q_width, q_width, kv_width, kv_width), -1)
        prefix = active_layer_name[0] + ".attention"
        trace(f"{prefix}.q_projected", q.reshape(1, length, cfg.n_head, cfg.head_dim))
        trace(f"{prefix}.gate", gate.reshape(1, length, cfg.n_head, cfg.head_dim))
        trace(f"{prefix}.k_projected", k.reshape(1, length, cfg.n_local_heads, cfg.head_dim))
        trace(f"{prefix}.v_projected", v.reshape(1, length, cfg.n_local_heads, cfg.head_dim))
        q = self.q_norm(q.reshape(length, cfg.n_head, cfg.head_dim))
        k = self.k_norm(k.reshape(length, cfg.n_local_heads, cfg.head_dim))
        trace(f"{prefix}.q_norm", q.transpose(0, 1).unsqueeze(0))
        trace(f"{prefix}.k_norm", k.transpose(0, 1).unsqueeze(0))
        q = self.rotate(q, positions).transpose(0, 1).unsqueeze(0)
        k = self.rotate(k, positions).transpose(0, 1).unsqueeze(0)
        v = v.reshape(length, cfg.n_local_heads, cfg.head_dim).transpose(0, 1).unsqueeze(0)
        trace(f"{prefix}.q_rope", q)
        trace(f"{prefix}.k_rope", k)
        if self.k_cache is not None:
            end = int(positions[-1]) + 1
            if end > self.k_cache.shape[2]:
                raise AssertionError("tower cache capacity exceeded")
            self.k_cache.index_copy_(2, positions, k)
            self.v_cache.index_copy_(2, positions, v)
            k, v = self.k_cache[:, :, :end], self.v_cache[:, :, :end]
            key_positions = torch.arange(end, device=x.device)
        else:
            key_positions = positions
        trace(f"{prefix}.key_cache", k)
        trace(f"{prefix}.value_cache", v)
        mask = key_positions[None, :] <= positions[:, None]
        context = tower_module.grouped_attention(q, k, v, mask)
        context = context.squeeze(0).transpose(0, 1).reshape(length, q_width)
        gated = context * gate.sigmoid()
        trace(f"{prefix}.gated_context", gated.reshape(1, length, q_width))
        output = self.wo(gated)
        trace(f"{prefix}.output", output.reshape(1, length, cfg.dim))
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

    original_layer_forward = TowerLayer.forward
    original_transformer_forward = TowerTransformerLayer.forward
    original_attention_forward = TowerAttention.forward
    original_mlp_forward = TowerMLP.forward
    original_sampler = tower_module.sample_top_k_top_p
    layer_indices = {id(layer): i for i, layer in enumerate(model.shared_layers)}
    layer_indices.update({id(layer): i for i, layer in enumerate(model.residual_layers)})

    def named_layer(self: Any, hidden: Any, codes: Any, semantic: Any, positions: Any, shared_input: Any = None) -> Any:
        active_layer[0] = layer_indices[id(self)]
        part = "shared_layers" if self.shared else "residual_layers"
        active_layer_name[0] = f"{active_prefix[0]}.{part}.{active_layer[0]}"
        return tower_layer(self, hidden, codes, semantic, positions, shared_input)

    TowerLayer.forward = named_layer
    TowerTransformerLayer.forward = transformer_layer
    TowerAttention.forward = tower_attention
    TowerMLP.forward = tower_mlp

    previous = torch.tensor([inputs["prefill_previous_frame"]], dtype=torch.long)
    next_frame = torch.tensor([inputs["prefill_next_frame"]], dtype=torch.long)
    positions = torch.tensor([0], dtype=torch.long)
    hidden = torch.tensor(hidden_rows[0][None, :], dtype=torch.float32)
    model.setup_caches(max_seq_len=2)
    try:
        with torch.no_grad():
            active_prefix[0] = "tower.prefill.row0"
            trace("tower.prefill.row0.hidden_input", hidden)
            semantic = model.is_semantic(next_frame[:, 0])
            unseen = model.embedding_sum(
                previous,
                1 + audio_config.vq_backbone_codebooks,
                audio_config.num_codebooks,
            )
            unseen = torch.where(model.is_semantic(previous[:, 0])[:, None], unseen, 0.0)
            trace("tower.prefill.row0.unseen_embedding", unseen)
            state = model.run_shared(hidden, next_frame[:, 1], semantic, positions, previous)
            for index, layer in enumerate(model.residual_layers):
                active_layer_name[0] = f"tower.prefill.row0.residual_layers.{index}"
                state = layer(state, next_frame[:, index + 1], semantic, positions)
                logits = model.codebook_heads[index](state)
                trace(f"tower.prefill.row0.head_logits.{index}", logits, [1, audio_config.vocab_size])
            trace("tower.prefill.row0.hidden_output", state)

            decode_main = int(inputs["decode_main"])
            text_vocab_size = int(config_data["text_config"]["vocab_size"])
            main_logits = torch.full((1, text_vocab_size), -1000.0, dtype=torch.float32)
            main_logits[0, decode_main] = 1000.0
            decode_previous = torch.tensor([inputs["decode_previous_frame"]], dtype=torch.long)
            decode_hidden = torch.tensor(hidden_rows[1][None, :], dtype=torch.float32)
            history = torch.full((1, 1), model.eos_id, dtype=torch.long)
            sample_index = [0]

            def deterministic_sampler(logits: Any, temperature: Any, top_k: Any, top_p: Any, *, check_finite: bool = True) -> Any:
                call_index = sample_index[0]
                sample_index[0] += 1
                if call_index >= 2:
                    head_index = call_index - 2
                    trace(
                        f"tower.decode.row0.head_logits.{head_index}",
                        logits,
                        [1, audio_config.vocab_size],
                    )
                return logits.argmax(-1)

            tower_module.sample_top_k_top_p = deterministic_sampler
            active_prefix[0] = "tower.decode.row0"
            trace("tower.decode.row0.hidden_input", decode_hidden)
            decode_unseen = model.embedding_sum(
                decode_previous,
                1 + audio_config.vq_backbone_codebooks,
                audio_config.num_codebooks,
            )
            decode_unseen = torch.where(
                model.is_semantic(decode_previous[:, 0])[:, None], decode_unseen, 0.0
            )
            trace("tower.decode.row0.unseen_embedding", decode_unseen)
            decoded = model.decode_step(
                main_logits,
                decode_hidden,
                torch.tensor([1], dtype=torch.long),
                decode_previous,
                history,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                check_finite=True,
            )
            trace("tower.decode.row0.hidden_output", last_decode_state[0])
    finally:
        TowerLayer.forward = original_layer_forward
        TowerTransformerLayer.forward = original_transformer_forward
        TowerAttention.forward = original_attention_forward
        TowerMLP.forward = original_mlp_forward
        tower_module.sample_top_k_top_p = original_sampler

    traces.sort(key=lambda item: item["name"])
    if len(traces) != 726:
        raise AssertionError(f"emitted {len(traces)} traces; expected 726")
    decoded_codes = decoded.detach().cpu().numpy().reshape(-1)[1:].astype(np.int32).tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"format": TRACE_FORMAT, "traces": traces, "reference_decoded_frame": decoded_codes},
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )

    shards = sorted(set(selected.values()))
    model_files = {
        "config.json": sha256_file(config_path),
        "model.safetensors.index.json": sha256_file(index_path),
    }
    if hash_shards:
        model_files.update({shard: sha256_file(model_dir / shard) for shard in shards})
    source_files = {
        "tower": fish_root / "fish_speech/models/text2semantic/fish_omni/tower.py",
        "layers": fish_root / "fish_speech/models/text2semantic/fish_omni/layers.py",
        "config": fish_root / "fish_speech/models/text2semantic/fish_omni/config.py",
    }
    manifest = {
        "kind": "actual_pinned_q4_checkpoint_tower_replay",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "upstream": {
            "fish-speech": {
                "repository": "https://github.com/fishaudio/fish-speech",
                "revision": FISH_REVISION,
            }
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "safetensors": safetensors_version,
        },
        "inputs": {"path": inputs_path.name, "sha256": sha256_file(inputs_path)},
        "model_files": model_files,
        "shard_sha256_recorded": hash_shards,
        "reference_files": {
            name: {"path": path.relative_to(fish_root).as_posix(), "sha256": sha256_file(path)}
            for name, path in source_files.items()
        },
        "tower_tensor_count": len(selected),
        "tower_shards": shards,
        "trace_count": len(traces),
        "trace_sha256": sha256_file(output_path),
        "trace_names": [trace_item["name"] for trace_item in traces],
        "native_decoded_frame_comparison": {
            "reference_decoded_frame": decoded_codes,
            "input_path": inputs_path.name,
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fish-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--inputs", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="list selected tower shards without loading weights")
    parser.add_argument("--hash-shards", action="store_true", help="hash selected multi-gigabyte shards after replay")
    args = parser.parse_args()
    args.fish_root = args.fish_root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.inputs = args.inputs.resolve()
    args.output = args.output.resolve()
    if args.plan_only:
        index = json.loads((args.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        selected = selected_tower_map(index["weight_map"])
        print(json.dumps({"tower_tensor_count": len(selected), "tower_shards": sorted(set(selected.values()))}))
        return 0
    manifest = run_replay(args.fish_root, args.model_dir, args.inputs, args.output, args.hash_shards)
    print(json.dumps({"trace_count": manifest["trace_count"], "manifest": str(args.output) + ".manifest.json"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
