"""Bounded pinned-source DiT execution for the direct-route factorial.

The only intervention is the supplied boolean attention mask. One reusable
BF16 block and one layer-local history cache serve six independent GPU state
banks. This module performs no refiner, scheduler update, or decoder work.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

VIDEO = 4320
TEXT = 320
DIM = 2048
HEADS = 16
HEAD_DIM = 128
CAPACITY = VIDEO + TEXT
T0_TEXT = 263
T1_TEXT = 224
HISTORY = VIDEO + T0_TEXT
QUERIES = VIDEO + T1_TEXT
KEYS = HISTORY + QUERIES
CHUNK = 256
CASES = ("M11", "M00", "M10", "M01", "M11_repeat")


def _dispatch_factory(torch: Any, state: dict[str, Any], masks: dict[str, Any],
                      legacy_factory: Any) -> Any:
    legacy = legacy_factory(torch, state)

    def dispatch(q_full: Any, full_k: Any, full_v: Any, q_mask: Any,
                 kv_mask: Any, cache_valid_len: int, update_cache: bool,
                 cache_txt: bool) -> Any:
        call = state["call"]
        if call["condition"] == "t0_caption":
            state["receipt"]["t0_attention_calls"] += 1
            return legacy(q_full, full_k, full_v, q_mask, kv_mask,
                          cache_valid_len, update_cache, cache_txt)
        physical_q = (1, CAPACITY, HEADS, HEAD_DIM)
        physical_k = (1, CAPACITY * 2, HEADS, HEAD_DIM)
        if (call["condition"] != "t1_action" or call["valid_text_tokens"] != T1_TEXT
                or cache_valid_len != CAPACITY or update_cache is not False
                or cache_txt is not True or tuple(q_full.shape) != physical_q
                or tuple(full_k.shape) != physical_k
                or tuple(full_v.shape) != physical_k
                or tuple(q_mask.shape) != (1, CAPACITY)
                or tuple(kv_mask.shape) != (1, 2 * CAPACITY)
                or any(t.device.type != "mps" or t.dtype != torch.bfloat16
                       for t in (q_full, full_k, full_v))
                or any(t.device != q_full.device or t.dtype != torch.bool
                       for t in (q_mask, kv_mask))):
            raise RuntimeError("T1 route attention escaped the pinned BF16 geometry")
        q = q_full[:, :QUERIES].permute(0, 2, 1, 3).contiguous()
        # Keep history and all video keys global. Only action text edges differ.
        compact_k = torch.cat((full_k[:, :HISTORY], full_k[:, CAPACITY:CAPACITY + QUERIES]), dim=1)
        compact_v = torch.cat((full_v[:, :HISTORY], full_v[:, CAPACITY:CAPACITY + QUERIES]), dim=1)
        keys = compact_k.permute(0, 2, 1, 3).contiguous()
        values = compact_v.permute(0, 2, 1, 3).contiguous()
        result = torch.zeros(physical_q, device=q_full.device, dtype=torch.bfloat16)
        chunks = masks[state["case"]]
        for index, start in enumerate(range(0, QUERIES, CHUNK)):
            stop = min(start + CHUNK, QUERIES)
            attended = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, start:stop], keys, values,
                attn_mask=chunks[index][None, None, :, :],
                dropout_p=0.0, is_causal=False,
            )
            result[:, start:stop].copy_(attended.permute(0, 2, 1, 3))
        receipt = state["receipt"]
        receipt["attention_calls"] += 1
        receipt["attention_query_chunks"] += len(chunks)
        receipt["t1_attention_calls"] += 1
        inventory = receipt.setdefault("attention_row_inventory", {})
        inventory[f"{2 * CAPACITY}->{KEYS}"] = inventory.get(f"{2 * CAPACITY}->{KEYS}", 0) + 1
        return result

    return dispatch


def run(admitted: dict[str, Any], cache_index_path: str | Path,
        upstream_path: str | Path, masks: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return 25 CPU arrays only after the sole explicit terminal MPS wait."""
    import numpy as np

    model = admitted["model"]  # Full shard/source authentication precedes MPS use.
    upstream = Path(upstream_path).resolve(strict=True)
    if (Path(model["upstream_path"]) != upstream
            or Path(model["cache_index_path"]) != Path(cache_index_path).resolve(strict=True)
            or len(model["cache_shards"]) != 54):
        raise RuntimeError("runtime model receipt differs from admitted sources")
    support = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-final-euler-20261104.c7zj7lnM")
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    base = importlib.import_module("replay_source_autocast_full_base")
    base.single._configure_mps_runtime()
    import torch
    from safetensors import safe_open

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU model fallback is forbidden")
    device, dtype = torch.device("mps"), torch.bfloat16
    # The accepted source wrapper supplies the unchanged pinned cache body.
    base.DIT_TEXT_ROWS = TEXT
    base.CACHE_CAPACITY = CAPACITY
    helpers = base.block_reference._install_upstream_modules(upstream)
    regular_state = {"mode": "bf16", "compare": base.single._NoopCompare(),
                     "receipt": {"attention_calls": 0, "attention_query_chunks": 0}}
    regular_attention = base.attention_reference._build_attention_adapter(torch, regular_state)
    block_type = base.block_reference._extract_pinned_block(
        upstream, torch, regular_attention, helpers[-1].apply_rotary_emb, helpers,
    )
    block = block_type(
        hidden_size=DIM, heads_num=HEADS, mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh", attn_mode="magi_mp_causal", qk_norm=True,
        qk_norm_type="rms", qkv_bias=True, dtype=dtype, device=torch.device("meta"),
    )
    block.to_empty(device=device).eval()
    expected = base.cache_reference._BLOCK_RELATIVE_SHAPES
    if set(block.state_dict()) != set(expected):
        raise RuntimeError("reusable source block tensor inventory mismatch")
    final_type = sys.modules["hymm.models.dit.modules.mlp_layers"].FinalLayerPerBlock
    final = final_type(DIM, 1, 32, helpers[0]("silu"), device=torch.device("meta"), dtype=dtype)
    globals_path = Path(model["global_shard"]["path"])
    with safe_open(str(globals_path), framework="pt", device="cpu") as source:
        final_state = {name: source.get_tensor(f"final_layer.{name}") for name in final.state_dict()}
        if any(t.dtype != dtype or tuple(t.shape) != tuple(final.state_dict()[name].shape)
               for name, t in final_state.items()):
            raise RuntimeError("source final-head BF16 weights mismatch")
        final.load_state_dict(final_state, strict=True, assign=True)
        final.to(device=device, dtype=dtype).eval()
    del final_state

    def input_tensor(name: str) -> Any:
        record = admitted["inputs"][name]
        array = np.fromfile(record["path"], dtype="<f4").reshape(record["shape"])
        return torch.from_numpy(array).to(device=device, dtype=dtype)

    def text_tensor(name: str, valid: int) -> Any:
        result = torch.zeros((1, TEXT, DIM), device=device, dtype=dtype)
        result[:, :valid].copy_(input_tensor(name))
        return result

    def rope_tensor(name: str, bank: str) -> Any:
        record = admitted["rope"][name]["banks"][bank]
        with Path(admitted["continuation_payload_path"]).open("rb") as stream:
            stream.seek(record["selected_range_offset"])
            payload = stream.read(record["selected_range_bytes"])
        if len(payload) != VIDEO * HEAD_DIM * 4:
            raise RuntimeError("selected RoPE range changed")
        array = np.frombuffer(payload, dtype="<f4").copy().reshape(VIDEO, HEAD_DIM)
        return torch.from_numpy(array).to(device=device, dtype=torch.float32)

    receipt = {"attention_calls": 0, "attention_query_chunks": 0,
               "t0_attention_calls": 0, "t1_attention_calls": 0,
               "explicit_terminal_synchronizations": 0, "intermediate_readbacks": 0,
               "scheduler_calls": 0, "decoder_calls": 0, "refiner_calls": 0,
               "weights_staged": 0, "arms_per_layer": 6, "one_layer_cache": True,
               "autocast": "mps BF16 full block and final head"}
    device_masks = {}
    for case in CASES[:4]:
        chunks = masks[case]
        if len(chunks) != 18:
            raise RuntimeError("source route mask requires exactly 18 chunks")
        for index, chunk in enumerate(chunks):
            if (tuple(chunk.shape) != (min(CHUNK, QUERIES - index * CHUNK), KEYS)
                    or chunk.dtype != torch.bool or chunk.device.type != "cpu"):
                raise RuntimeError("source CPU route mask has wrong shape/dtype/device")
        device_masks[case] = [chunk.to(device=device) for chunk in chunks]
    device_masks["M11_repeat"] = device_masks["M11"]
    state = {"dtype": dtype, "device_type": "mps", "call": None,
             "case": None, "attention_layout": "compact", "t0_valid_tokens": T0_TEXT,
             "receipt": receipt}
    original_factory = base._cache_attention_dispatch_factory
    try:
        base._cache_attention_dispatch_factory = lambda torch_, state_: _dispatch_factory(
            torch_, state_, device_masks, original_factory,
        )
        cache_function = base._build_pinned_cache_function(torch, upstream, state)
    finally:
        base._cache_attention_dispatch_factory = original_factory
    base.two_segment._bind_runtime_cache_functions([block], None, cache_function)
    cache = {key: torch.zeros((1, CAPACITY, HEADS, HEAD_DIM), device=device, dtype=dtype)
             for key in ("k", "v")}
    cache_mask = torch.zeros((1, CAPACITY), device=device, dtype=torch.bool)
    writer = [input_tensor("t0.patch"), text_tensor("t0.text_valid", T0_TEXT)]
    initial = [input_tensor("t1.step1.patch"), text_tensor("t1.step1.text_valid", T1_TEXT)]
    arms = {case: [value.clone() for value in initial] for case in CASES}
    vectors = [input_tensor("t0.vector"), input_tensor("t1.step1.vector")]
    ropes = [(rope_tensor("input.rope_cos", bank), rope_tensor("input.rope_sin", bank))
             for bank in ("T0", "T1")]
    text_masks = [torch.arange(TEXT, device=device)[None, :] < valid
                  for valid in (T0_TEXT, T1_TEXT)]
    captures: dict[str, Any] = {}
    cache_dir = Path(model["cache_index_path"]).parent

    def forward(values: list[Any], index: int) -> list[Any]:
        image, text = block(
            img=values[0], txt=values[1], vec=vectors[index], vec_txt=vectors[index],
            freqs_cis=ropes[index], text_mask=text_masks[index], attn_param=None,
            is_flash=False, txt_tokens_per_block=TEXT,
            prompt_video_boundaries=[0, VIDEO], kv_cache=cache,
            cache_valid_len=0 if index == 0 else CAPACITY,
            update_cache=index == 0, cache_txt=True,
            cache_attn_mode="torch", cache_mask=cache_mask,
        )
        if any(t.device.type != "mps" or t.dtype != dtype for t in (image, text)):
            raise RuntimeError("source block output escaped MPS BF16")
        return [image, text]

    with torch.inference_mode(), torch.autocast("mps", dtype=dtype):
        for layer, entry in enumerate(model["cache_index"]["blocks"]):
            path = base.single._canonical_file(cache_dir / entry["bf16"], "BF16 block shard")
            if base.single._file_identity(path) != model["cache_identities"][str(path)]:
                raise RuntimeError("BF16 shard changed after authentication")
            with safe_open(str(path), framework="pt", device="cpu") as source:
                if set(source.keys()) != {f"double_blocks.{layer}.{name}" for name in expected}:
                    raise RuntimeError("BF16 block source inventory changed")
                # Reuse GPU storage; Metal's ordered stream preserves the preceding
                # six calls before these new weights become visible. No CPU wait.
                for name, target in block.state_dict().items():
                    value = source.get_tensor(f"double_blocks.{layer}.{name}")
                    if value.dtype != dtype or list(value.shape) != expected[name]:
                        raise RuntimeError("BF16 block source shape/dtype changed")
                    target.copy_(value, non_blocking=True)
            if base.single._file_identity(path) != model["cache_identities"][str(path)]:
                raise RuntimeError("BF16 shard changed while staging")
            receipt["weights_staged"] += 1
            cache_mask.zero_()
            state["call"] = {"condition": "t0_caption", "valid_text_tokens": T0_TEXT,
                             "cache_valid_len": 0, "update_cache": True, "cache_txt": True}
            writer = forward(writer, 0)
            state["call"] = {"condition": "t1_action", "valid_text_tokens": T1_TEXT,
                             "cache_valid_len": CAPACITY, "update_cache": False, "cache_txt": True}
            for case in CASES:
                state["case"] = case
                arms[case] = forward(arms[case], 1)
                if layer in (0, 53):
                    captures[f"{case}.block{layer}.image"] = arms[case][0].clone()
                    captures[f"{case}.block{layer}.text_valid"] = arms[case][1][:, :T1_TEXT].contiguous().clone()
        for case in CASES:
            prediction = final(arms[case][0], vectors[1])
            if prediction.dtype != dtype or tuple(prediction.shape) != (1, VIDEO, 32):
                raise RuntimeError("source final-head dtype/shape mismatch")
            captures[f"{case}.velocity"] = prediction.reshape(1, 30, 9, 16, 32).permute(0, 4, 1, 2, 3).contiguous()
    state["call"] = None
    if receipt["attention_calls"] != 324 or receipt["attention_query_chunks"] != 5832:
        raise RuntimeError("source factorial attention call inventory mismatch")
    if receipt.pop("attention_row_inventory", None) != {"4640->4583": 54, "9280->9127": 270}:
        raise RuntimeError("source factorial compact attention row inventory mismatch")
    # Include lossless promotions in the GPU phase, then export one packed
    # array. No conversion kernels or per-checkpoint transfers follow the wait.
    layout = [(name, tuple(tensor.shape), tensor.numel()) for name, tensor in captures.items()]
    packed = torch.cat([tensor.float().reshape(-1) for tensor in captures.values()])
    torch.mps.synchronize()
    receipt["explicit_terminal_synchronizations"] += 1
    host = packed.cpu().numpy()
    outputs = {}
    cursor = 0
    for name, shape, count in layout:
        outputs[name] = host[cursor:cursor + count].reshape(shape)
        cursor += count
    if cursor != host.size:
        raise RuntimeError("terminal source export inventory mismatch")
    return outputs, receipt
