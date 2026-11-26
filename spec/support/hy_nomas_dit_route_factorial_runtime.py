"""Bounded pinned-source DiT execution for direct-route and soft-route probes.

The default intervention is the supplied Boolean attention mask. The optional
soft-route path adds BF16 logit priors within an authenticated Boolean support.
One reusable BF16 block and one layer-local history cache serve each state bank.
This module performs no refiner, scheduler update, or decoder work.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any, Mapping

VIDEO = 4320
TEXT = 320
DIM = 2048
HEADS = 16
HEAD_DIM = 128
BLOCKS = 54
CAPACITY = VIDEO + TEXT
T0_TEXT = 263
T1_TEXT = 224
HISTORY = VIDEO + T0_TEXT
QUERIES = VIDEO + T1_TEXT
KEYS = HISTORY + QUERIES
CHUNK = 256
CASES = ("M11", "M00", "M10", "M01", "M11_repeat")
SOFT_ROUTE_CASES = (
    "zero", "A1_low", "A1_high", "A2_low", "A2_high", "A1_sham", "A2_sham", "zero_repeat",
)
SOFT_ACTION_KEY_SPANS = {
    "A1": (HISTORY + VIDEO + 49, HISTORY + VIDEO + 132),
    "A2": (HISTORY + VIDEO + 132, HISTORY + VIDEO + 224),
}


def expected_runtime_receipt(cases: tuple[str, ...], *, terminal_synchronizations: int = 1,
                             include_attention_row_inventory: bool = True) -> dict[str, Any]:
    """Build the exact counter receipt for either supported route experiment."""
    cases = tuple(cases)
    if cases not in (CASES, SOFT_ROUTE_CASES) or terminal_synchronizations not in (0, 1):
        raise RuntimeError("unsupported runtime receipt case inventory or synchronization count")
    t1_calls = BLOCKS * len(cases)
    receipt = {
        "attention_calls": BLOCKS + t1_calls,
        "attention_query_chunks": (BLOCKS + t1_calls) * 18,
        "t0_attention_calls": BLOCKS,
        "t1_attention_calls": t1_calls,
        "explicit_terminal_synchronizations": terminal_synchronizations,
        "intermediate_readbacks": 0,
        "scheduler_calls": 0,
        "decoder_calls": 0,
        "refiner_calls": 0,
        "weights_staged": BLOCKS,
        "arms_per_layer": 1 + len(cases),
        "one_layer_cache": True,
        "autocast": "mps BF16 full block and final head",
    }
    if include_attention_row_inventory:
        receipt["attention_row_inventory"] = {
            "4640->4583": BLOCKS,
            "9280->9127": t1_calls,
        }
    return receipt


def assert_runtime_receipt(receipt: Any, cases: tuple[str, ...], *, terminal_synchronizations: int = 1,
                           include_attention_row_inventory: bool = True) -> None:
    """Fail unless every source execution counter matches its frozen schedule."""
    if type(receipt) is not dict or receipt != expected_runtime_receipt(
            cases, terminal_synchronizations=terminal_synchronizations,
            include_attention_row_inventory=include_attention_row_inventory):
        raise RuntimeError("source runtime receipt inventory mismatch")


def _soft_route_additive_chunk(torch: Any, source: Any, bias: Any,
                               action: str, query_start: int) -> Any:
    """Return one BF16 additive chunk with exactly the supplied Boolean support."""
    import numpy as np

    if (action not in SOFT_ACTION_KEY_SPANS or source.device.type != "cpu"
            or source.dtype != torch.bool or len(source.shape) != 2
            or int(source.shape[1]) != KEYS):
        raise RuntimeError("soft-route source chunk geometry/dtype/device mismatch")
    array = np.asarray(bias)
    if array.shape != (QUERIES,) or array.dtype != np.dtype("float32") or not np.isfinite(array).all():
        raise RuntimeError("soft-route bias vector shape/dtype/nonfinite mismatch")
    values = torch.from_numpy(array.copy())
    if not torch.equal(values.to(dtype=torch.bfloat16).to(dtype=torch.float32), values):
        raise RuntimeError("soft-route bias is not exactly BF16 representable")
    rows = int(source.shape[0])
    if query_start < 0 or query_start + rows > QUERIES:
        raise RuntimeError("soft-route chunk query bounds mismatch")
    additive = torch.zeros(tuple(source.shape), dtype=torch.bfloat16, device="cpu")
    additive.masked_fill_(~source, float("-inf"))
    key_start, key_stop = SOFT_ACTION_KEY_SPANS[action]
    video_stop = min(query_start + rows, VIDEO)
    if query_start < video_stop:
        local_stop = video_stop - query_start
        target_allowed = source[:local_stop, key_start:key_stop]
        row_bias = values[query_start:video_stop].to(dtype=torch.bfloat16)
        nonzero = row_bias != 0
        if torch.any(nonzero & ~target_allowed.all(dim=1)):
            raise RuntimeError("soft-route bias escapes accepted M11 action support")
        expanded = row_bias[:, None].expand(-1, key_stop - key_start)
        additive[:local_stop, key_start:key_stop] = torch.where(
            target_allowed, expanded, torch.full_like(expanded, float("-inf")),
        )
    if not torch.equal(torch.isneginf(additive), ~source):
        raise RuntimeError("soft-route additive mask changed hard M11 support")
    return additive


def _normalize_soft_route_biases(biases: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    """Accept six intervention vectors or all eight cases with exact zero controls."""
    import numpy as np

    keys = set(biases)
    positive_cases = set(SOFT_ROUTE_CASES) - {"zero", "zero_repeat"}
    if keys == set(SOFT_ROUTE_CASES):
        for case in ("zero", "zero_repeat"):
            control = np.asarray(biases[case])
            if (control.shape != (QUERIES,) or control.dtype != np.dtype("float32")
                    or not np.isfinite(control).all() or np.any(control != 0)):
                raise RuntimeError(f"soft-route control bias must be an exact F32 zero vector: {case}")
    elif keys != positive_cases:
        raise RuntimeError("soft-route bias inventory mismatch")

    normalized: dict[str, Any] = {}
    for case in SOFT_ROUTE_CASES[1:-1]:
        array = np.asarray(biases[case])
        if (array.shape != (QUERIES,) or array.dtype != np.dtype("float32")
                or not np.isfinite(array).all()):
            raise RuntimeError(f"soft-route bias vector shape/dtype/nonfinite mismatch: {case}")
        rounded = array.astype(np.float32, copy=False)
        roundtrip = torch.from_numpy(rounded.copy()).to(dtype=torch.bfloat16).to(dtype=torch.float32).numpy()
        if not np.array_equal(roundtrip, rounded):
            raise RuntimeError(f"soft-route bias is not exactly BF16 representable: {case}")
        normalized[case] = array
    return normalized


def _soft_route_device_masks(torch: Any, source_masks: Mapping[str, Any], cases: tuple[str, ...],
                             biases: Mapping[str, Any], device: Any) -> dict[str, list[Any]]:
    """Prepare one BF16 additive M11 mask per positive case; zero stays Boolean."""
    if tuple(cases) != SOFT_ROUTE_CASES:
        raise RuntimeError("soft-route case inventory mismatch")
    positive_biases = _normalize_soft_route_biases(biases, torch)
    if "M11" not in source_masks or len(source_masks["M11"]) != 18:
        raise RuntimeError("soft-route requires the 18 accepted M11 mask chunks")
    source_chunks = source_masks["M11"]
    for index, chunk in enumerate(source_chunks):
        expected_rows = min(CHUNK, QUERIES - index * CHUNK)
        if (tuple(chunk.shape) != (expected_rows, KEYS) or chunk.dtype != torch.bool
                or chunk.device.type != "cpu"):
            raise RuntimeError("soft-route M11 source mask geometry/dtype/device mismatch")

    device_masks: dict[str, list[Any]] = {}
    device_masks["zero"] = [chunk.to(device=device) for chunk in source_chunks]
    device_masks["zero_repeat"] = device_masks["zero"]
    for case, array in positive_biases.items():
        action = case[:2]
        if action not in SOFT_ACTION_KEY_SPANS:
            raise RuntimeError(f"soft-route action key span missing: {case}")
        chunks: list[Any] = []
        for index, source in enumerate(source_chunks):
            q_start = index * CHUNK
            try:
                additive = _soft_route_additive_chunk(torch, source, array, action, q_start)
            except RuntimeError as error:
                raise RuntimeError(f"{error}: {case}") from None
            chunks.append(additive.to(device=device))
        device_masks[case] = chunks
    return device_masks


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
        upstream_path: str | Path, masks: dict[str, Any], *,
        cases: tuple[str, ...] | None = None,
        biases: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return captured CPU arrays only after the sole explicit terminal MPS wait.

    The default remains the original five-case boolean route factorial. The
    optional soft-route path uses the accepted M11 support, with BF16 additive
    masks for its positive-bias cases and Boolean M11 masks for zero controls.
    """
    import numpy as np

    route_cases = CASES if cases is None else tuple(cases)
    if biases is None:
        if route_cases != CASES:
            raise RuntimeError("custom source cases require an explicit bias mapping")
    elif route_cases != SOFT_ROUTE_CASES:
        raise RuntimeError("soft-route source case order mismatch")

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
               "weights_staged": 0, "arms_per_layer": len(route_cases) + 1, "one_layer_cache": True,
               "autocast": "mps BF16 full block and final head"}
    if biases is None:
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
    else:
        device_masks = _soft_route_device_masks(torch, masks, route_cases, biases, device)
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
    arms = {case: [value.clone() for value in initial] for case in route_cases}
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
            # all T1 arms before these new weights become visible. No CPU wait.
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
            for case in route_cases:
                state["case"] = case
                arms[case] = forward(arms[case], 1)
                if layer in (0, 53):
                    captures[f"{case}.block{layer}.image"] = arms[case][0].clone()
                    captures[f"{case}.block{layer}.text_valid"] = arms[case][1][:, :T1_TEXT].contiguous().clone()
        for case in route_cases:
            prediction = final(arms[case][0], vectors[1])
            if prediction.dtype != dtype or tuple(prediction.shape) != (1, VIDEO, 32):
                raise RuntimeError("source final-head dtype/shape mismatch")
            captures[f"{case}.velocity"] = prediction.reshape(1, 30, 9, 16, 32).permute(0, 4, 1, 2, 3).contiguous()
    state["call"] = None
    assert_runtime_receipt(receipt, route_cases, terminal_synchronizations=0)
    # Include lossless promotions in the GPU phase, then export one packed
    # array. No conversion kernels or per-checkpoint transfers follow the wait.
    layout = [(name, tuple(tensor.shape), tensor.numel()) for name, tensor in captures.items()]
    packed = torch.cat([tensor.float().reshape(-1) for tensor in captures.values()])
    torch.mps.synchronize()
    receipt["explicit_terminal_synchronizations"] += 1
    assert_runtime_receipt(receipt, route_cases)
    receipt.pop("attention_row_inventory")
    host = packed.cpu().numpy()
    outputs = {}
    cursor = 0
    for name, shape, count in layout:
        outputs[name] = host[cursor:cursor + count].reshape(shape)
        cursor += count
    if cursor != host.size:
        raise RuntimeError("terminal source export inventory mismatch")
    return outputs, receipt


def run_additive_mask_preflight() -> dict[str, Any]:
    """Check MPS BF16 additive-mask support and exact zero-bias equivalence."""
    import math
    import numpy as np
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the additive-mask preflight")
    device = torch.device("mps")
    dtype = torch.bfloat16
    # Zero logits make the Boolean control a closed-form average over allowed
    # values. This avoids saturated inputs and gives a positive-bias oracle.
    q = torch.zeros((1, 2, 4, 4), dtype=dtype, device=device)
    k = torch.zeros_like(q)
    v = torch.eye(4, dtype=dtype)[None, None].expand(1, 2, 4, 4).contiguous().to(device=device)
    allowed = torch.tensor(
        [[True, True, False, True], [True, True, True, False],
         [True, False, True, True], [False, True, True, True]],
        dtype=torch.bool,
    )
    zero_additive = torch.zeros(allowed.shape, dtype=dtype)
    zero_additive.masked_fill_(~allowed, float("-inf"))
    biased = zero_additive.clone()
    bias_value = torch.tensor(math.log(2.0), dtype=dtype).item()
    biased[1, 2] = torch.tensor(bias_value, dtype=dtype)
    allowed_device = allowed[None, None].to(device=device)
    zero_device = zero_additive[None, None].to(device=device)
    biased_device = biased[None, None].to(device=device)
    boolean = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed_device, dropout_p=0.0, is_causal=False,
    )
    additive_zero = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=zero_device, dropout_p=0.0, is_causal=False,
    )
    additive_positive = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=biased_device, dropout_p=0.0, is_causal=False,
    )
    packed = torch.cat((boolean.float().reshape(-1), additive_zero.float().reshape(-1),
                        additive_positive.float().reshape(-1)))
    expected_bias = math.exp(bias_value)
    expected_denominator = 2.0 + expected_bias
    expected_row = torch.tensor(
        [1.0 / expected_denominator, 1.0 / expected_denominator,
         expected_bias / expected_denominator, 0.0], dtype=dtype,
    ).float().numpy()
    torch.mps.synchronize()
    host = packed.cpu().numpy()
    count = boolean.numel()
    source = host[:count]
    zero = host[count:2 * count]
    positive = host[2 * count:]
    if not np.array_equal(source, zero):
        raise RuntimeError("zero BF16 additive mask differs from the Boolean MPS control")
    positive = positive.reshape(tuple(additive_positive.shape))
    if not np.array_equal(positive[0, 0, 1], expected_row):
        raise RuntimeError("positive BF16 additive output differs from the analytic softmax fixture")
    return {
        "mps_available": True, "dtype": "BF16", "shape": [1, 2, 4, 4],
        "boolean_zero_exact": True, "positive_bias_changes_output": True,
        "positive_bias_expected_row": expected_row.tolist(),
        "explicit_terminal_synchronizations": 1,
    }
