#!/usr/bin/env python3
"""Bounded pretrained PixelUMM decoder references with local attention.

The helper imports pinned upstream GenLayer/RoPE/layout code, loads only the
authenticated subset for the requested decoder block or consecutive prefix,
and replaces FlexAttention with per-spatial-tile gathered QK/softmax/PV. Fixture
inputs are controlled synthetic BF16 hidden states, not full decoder activations.
The full target exports only per-block attention and output boundaries.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from contextlib import nullcontext
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable, Mapping

import torch

import hy_nomas_pixelumm_checkpoint as checkpoint_inventory

UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_MODEL_PATH = "hymm/models/vae/pixelumm_decoder/model.py"
CHECKPOINT_SUBSET_SHA256 = "ae82bb41c121a955493f5b1502dc512ac8aaba05a6cec67e98edeaedfe720485"
CHECKPOINT_SUBSET_BYTES = 771_787_128
CHECKPOINT_PREFIX = "decoder.layers.0."
SCHEMA = "cogni_ml.pixelumm_block_reference/v1"
TWO_BLOCK_SCHEMA = "cogni_ml.pixelumm_two_block_reference/v1"
FOUR_BLOCK_SCHEMA = "cogni_ml.pixelumm_four_block_reference/v1"
EIGHT_BLOCK_SCHEMA = "cogni_ml.pixelumm_eight_block_reference/v1"
TILE_TOKENS = 8
ATOL = RTOL = 2e-5

ALL_STAGES = (
    "input_norm_statistics", "input_norm_pre_gain", "input_norm",
    "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "q_rope", "k_rope",
    "q_norm_statistics", "k_norm_statistics", "q_norm_inverse", "k_norm_inverse",
    "q_ready", "k_ready",
    "attention", "o_proj", "attn_residual",
    "post_attn_norm", "gate_proj", "up_proj", "silu_gate", "product",
    "down_proj", "output",
)
FULL_TARGET_STAGES = ("attention", "output")


class ReferenceError(ValueError):
    """The requested source, checkpoint, geometry, or fixture is unsupported."""


@dataclass(frozen=True)
class TileRegion:
    tile_h: int
    tile_w: int
    query_start: int
    query_stop: int
    key_positions: tuple[int, ...]


@dataclass(frozen=True)
class AttentionPlan:
    grid: tuple[int, int, int]
    tile: int
    permutation: tuple[int, ...]
    regions: tuple[TileRegion, ...]
    permutation_sha256: str


def _check_grid(grid: Iterable[int], tile: int = TILE_TOKENS) -> tuple[int, int, int]:
    values = tuple(grid)
    if len(values) != 3 or any(type(value) is not int or value <= 0 for value in values):
        raise ReferenceError("grid must contain three positive integers [T,H,W]")
    if type(tile) is not int or tile <= 0:
        raise ReferenceError("tile size must be a positive integer")
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upstream_permutation_formula(
    grid: tuple[int, int, int], tile: int,
) -> torch.Tensor:
    """The pinned helper's sorting formula, without constructing its Flex mask.

    At the 21120-token target, constructing a diagnostic Flex BlockMask may
    evaluate a quadratic mask domain. The formula below is copied from the
    pinned source's ``local_attention_layout`` permutation setup; small grids
    are checked directly against that helper in ``make_attention_plan``.
    """
    frames, height, width = grid
    tiles_w = (width + tile - 1) // tile
    t, h, w = torch.meshgrid(
        *(torch.arange(n, device="cpu") for n in grid), indexing="ij"
    )
    tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
    order = (((tile_h * tiles_w + tile_w) * frames + t.flatten()) * tile + (h % tile).flatten()) * tile
    return torch.argsort(order + (w % tile).flatten())


def load_upstream(upstream: Path | str, *, verify_revision: bool = True):
    """Import the pinned source module without constructing its full 36-layer model."""
    root = Path(upstream).expanduser().resolve()
    source_file = root / UPSTREAM_MODEL_PATH
    if not source_file.is_file():
        raise ReferenceError(f"missing upstream source file: {source_file}")
    if verify_revision:
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            changed = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--", UPSTREAM_MODEL_PATH],
                text=True, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ReferenceError("cannot verify pinned upstream git source") from error
        if revision != UPSTREAM_REVISION or changed.strip():
            raise ReferenceError("upstream revision or model.py worktree identity mismatch")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        module = importlib.import_module("hymm.models.vae.pixelumm_decoder.model")
    except Exception as error:  # preserve a compact CLI error at the outer boundary
        raise ReferenceError(f"cannot import pinned PixelUMM model module: {error}") from error
    imported_path = Path(module.__file__).resolve()
    if imported_path != source_file.resolve():
        raise ReferenceError("Python imported a different PixelUMM model.py")
    return module, _sha256_file(source_file)


def make_attention_plan(source, grid: Iterable[int], tile: int = TILE_TOKENS) -> tuple[AttentionPlan, torch.Tensor, object]:
    """Use upstream local_attention_layout for permutation and mask construction.

    Returns (plan, inverse, source_block_mask). Grids whose source semantics are
    global attention are rejected so this adapter can never accidentally create
    a dense target score matrix.
    """
    grid = _check_grid(grid, tile)
    _, height, width = grid
    tiles_h = (height + tile - 1) // tile
    tiles_w = (width + tile - 1) // tile
    if tiles_h <= 3 and tiles_w <= 3:
        raise ReferenceError(
            "local adapter requires more than 3 tiles on at least one spatial axis"
        )
    count = math.prod(grid)
    if count <= 2048:
        layout = source.local_attention_layout(grid, tile, "cpu")
        if layout is None:
            raise ReferenceError("upstream helper reported global attention for a local grid")
        perm, inverse, block_mask = layout
        formula_perm = _upstream_permutation_formula(grid, tile)
        if not torch.equal(perm, formula_perm):
            raise ReferenceError("local permutation formula disagrees with pinned source helper")
    else:
        # Never ask eager Flex to build a full-target BlockMask. The native and
        # reference paths consume only these static bounded tile neighborhoods.
        perm = _upstream_permutation_formula(grid, tile)
        inverse = torch.argsort(perm)
        block_mask = None
    if perm.device.type != "cpu" or tuple(perm.shape) != (count,):
        raise ReferenceError("upstream layout returned an unexpected permutation")
    permutation = tuple(int(value) for value in perm.tolist())
    if sorted(permutation) != list(range(count)):
        raise ReferenceError("upstream token permutation is not a bijection")

    tile_coordinates: list[tuple[int, int]] = []
    for original in permutation:
        h = (original // width) % height
        w = original % width
        tile_coordinates.append((h // tile, w // tile))

    regions: list[TileRegion] = []
    cursor = 0
    for tile_h in range(tiles_h):
        for tile_w in range(tiles_w):
            q_positions = [
                position for position, coordinate in enumerate(tile_coordinates)
                if coordinate == (tile_h, tile_w)
            ]
            if not q_positions:
                raise ReferenceError("upstream permutation omitted a spatial tile")
            if q_positions != list(range(q_positions[0], q_positions[-1] + 1)):
                raise ReferenceError("upstream tile-major query positions are not contiguous")
            center_h = min(max(tile_h, 1), max(tiles_h - 2, 1))
            center_w = min(max(tile_w, 1), max(tiles_w - 2, 1))
            key_positions = tuple(
                position for position, (key_h, key_w) in enumerate(tile_coordinates)
                if abs(center_h - key_h) <= 1 and abs(center_w - key_w) <= 1
            )
            regions.append(TileRegion(
                tile_h=tile_h, tile_w=tile_w,
                query_start=q_positions[0], query_stop=q_positions[-1] + 1,
                key_positions=key_positions,
            ))
            cursor += len(q_positions)
    if cursor != count:
        raise ReferenceError("spatial tile plan does not cover every token")
    perm_bytes = torch.tensor(permutation, dtype=torch.int32).numpy().tobytes()
    plan = AttentionPlan(
        grid=grid, tile=tile, permutation=permutation,
        regions=tuple(regions), permutation_sha256=hashlib.sha256(perm_bytes).hexdigest(),
    )
    return plan, inverse, block_mask


def gathered_local_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: AttentionPlan,
) -> torch.Tensor:
    """Per-spatial-tile QK/softmax/PV with physically gathered, ordered KV rows.

    q/k/v use [B,J,heads,D], all in the layer's projected dtype. Keys for each
    tile are selected in the upstream permutation order, including every time
    position; only spatial tile distance is constrained.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ReferenceError("q, k, and v must have shape [B,J,heads,D]")
    batch, count, q_heads, dim = q.shape
    if count != math.prod(plan.grid) or k.shape[:2] != (batch, count) or v.shape != k.shape:
        raise ReferenceError("q/k/v shapes do not match the attention plan")
    if k.shape[-1] != dim or q_heads % k.shape[2] or q.device != k.device or k.device != v.device:
        raise ReferenceError("incompatible attention heads, devices, or dimensions")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ReferenceError("q/k/v projected dtypes must match")
    group = q_heads // k.shape[2]
    output = torch.empty_like(q)
    scale = dim ** -0.5
    device = q.device
    f32_context = (
        torch.autocast(device_type=device.type, enabled=False)
        if device.type in {"cpu", "mps"} else nullcontext()
    )
    device_regions = [
        (
            region,
            torch.arange(region.query_start, region.query_stop, dtype=torch.long, device=device),
            torch.tensor(region.key_positions, dtype=torch.long, device=device),
        )
        for region in plan.regions
    ]
    with f32_context:
        for region, query_idx, key_idx in device_regions:
            # Indices were statically built and copied before the first tile's
            # math; no nonzero/item-based discovery or host readback occurs in
            # the MPS attention loop.
            q_tile = q.index_select(1, query_idx).transpose(1, 2).float()
            k_tile = k.index_select(1, key_idx).repeat_interleave(group, dim=2).transpose(1, 2).float()
            v_tile = v.index_select(1, key_idx).repeat_interleave(group, dim=2).transpose(1, 2).float()
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1))
            scores.mul_(scale)
            probabilities = torch.softmax(scores, dim=-1)
            attended = torch.matmul(probabilities, v_tile).to(dtype=q.dtype)
            output[:, region.query_start:region.query_stop] = attended.transpose(1, 2)
    return output


def _prepare_qk_with_stages(attention, raw, cos, sin, norm_t, norm_hw, dtype):
    normalized = attention._split_norm(raw.float(), norm_t, norm_hw)
    rotated = attention_source_apply_mrope(attention, normalized, cos, sin)
    return rotated.to(dtype), normalized, rotated


def _qk_norm_diagnostics(raw: torch.Tensor, norm_t, norm_hw):
    """Report source RMSNorm statistics and inverse roots without feeding forward.

    The upstream QK path independently applies RMSNorm to the first and second
    half of each head. Keep this diagnostic as separate eager ops so it cannot
    change the original ``_split_norm``/RoPE path being compared.
    """
    temporal, spatial = raw.float().chunk(2, dim=-1)
    statistics = torch.cat((
        temporal.pow(2).mean(-1, keepdim=True) + norm_t.eps,
        spatial.pow(2).mean(-1, keepdim=True) + norm_hw.eps,
    ), dim=-1)
    inverse = torch.rsqrt(statistics)
    return statistics, inverse


def attention_source_apply_mrope(attention, normalized, cos, sin):
    # Call the exact source function rather than maintaining a second RoPE impl.
    source = sys.modules.get("hymm.models.vae.pixelumm_decoder.model")
    if source is None:
        raise ReferenceError("pinned upstream module is not loaded")
    return source.apply_mrope(normalized, cos, sin, attention.arch.rope_sections)


def run_layer_reference(
    layer,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: AttentionPlan,
    *,
    capture_stages: Iterable[str] = (),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run upstream GenLayer equations, substituting only bounded local attention."""
    requested = set(capture_stages)
    unknown = requested - set(ALL_STAGES)
    if unknown:
        raise ReferenceError(f"unknown capture stage(s): {', '.join(sorted(unknown))}")
    if x.ndim != 3 or x.shape[1] != math.prod(plan.grid):
        raise ReferenceError("x must have shape [B,J,hidden] for the attention plan")
    captures: dict[str, torch.Tensor] = {}
    attention = layer.attn
    batch, count, _ = x.shape

    if requested & {"input_norm_statistics", "input_norm_pre_gain"}:
        input_f32 = x.float()
        input_norm_statistics = (
            input_f32.pow(2).mean(-1, keepdim=True) + layer.input_norm.eps
        )
        if "input_norm_statistics" in requested:
            captures["input_norm_statistics"] = input_norm_statistics
        if "input_norm_pre_gain" in requested:
            captures["input_norm_pre_gain"] = (
                input_f32 * torch.rsqrt(input_norm_statistics)
            ).to(dtype=x.dtype)
        del input_f32, input_norm_statistics
    normalized_input = layer.input_norm(x)
    if "input_norm" in requested:
        captures["input_norm"] = normalized_input
    q_raw = attention.q_proj(normalized_input).view(
        batch, count, attention.arch.num_heads, attention.arch.head_dim
    )
    k_raw = attention.k_proj(normalized_input).view(
        batch, count, attention.arch.num_kv_heads, attention.arch.head_dim
    )
    v = attention.v_proj(normalized_input).view(
        batch, count, attention.arch.num_kv_heads, attention.arch.head_dim
    )
    if "q_proj" in requested:
        captures["q_proj"] = q_raw
    if "k_proj" in requested:
        captures["k_proj"] = k_raw
    if "v_proj" in requested:
        captures["v_proj"] = v

    # These captures inspect the exact two 64-wide RMSNorm domains per Q/K
    # head. They are computed independently and never replace the source
    # _prepare_qk/_split_norm results used below.
    if requested & {"q_norm_statistics", "q_norm_inverse"}:
        q_statistics, q_inverse = _qk_norm_diagnostics(
            q_raw, attention.q_norm_t, attention.q_norm_hw
        )
        if "q_norm_statistics" in requested:
            captures["q_norm_statistics"] = q_statistics
        if "q_norm_inverse" in requested:
            captures["q_norm_inverse"] = q_inverse
    if requested & {"k_norm_statistics", "k_norm_inverse"}:
        k_statistics, k_inverse = _qk_norm_diagnostics(
            k_raw, attention.k_norm_t, attention.k_norm_hw
        )
        if "k_norm_statistics" in requested:
            captures["k_norm_statistics"] = k_statistics
        if "k_norm_inverse" in requested:
            captures["k_norm_inverse"] = k_inverse

    capture_qk_steps = bool(requested & {
        "q_norm", "k_norm", "q_rope", "k_rope", "q_ready", "k_ready",
    })
    if capture_qk_steps:
        q, q_norm, q_rope = _prepare_qk_with_stages(
            attention, q_raw, cos, sin, attention.q_norm_t, attention.q_norm_hw, v.dtype
        )
        k, k_norm, k_rope = _prepare_qk_with_stages(
            attention, k_raw, cos, sin, attention.k_norm_t, attention.k_norm_hw, v.dtype
        )
        if "q_norm" in requested:
            captures["q_norm"] = q_norm
        if "k_norm" in requested:
            captures["k_norm"] = k_norm
        if "q_rope" in requested:
            captures["q_rope"] = q_rope
        if "k_rope" in requested:
            captures["k_rope"] = k_rope
        if "q_ready" in requested:
            captures["q_ready"] = q
        if "k_ready" in requested:
            captures["k_ready"] = k
    else:
        q = attention._prepare_qk(
            q_raw, cos, sin, attention.q_norm_t, attention.q_norm_hw, v.dtype
        )
        k = attention._prepare_qk(
            k_raw, cos, sin, attention.k_norm_t, attention.k_norm_hw, v.dtype
        )

    attended = gathered_local_attention(q, k, v, plan)
    if "attention" in requested:
        captures["attention"] = attended
    projected_attention = attention.o_proj(
        attended.reshape(batch, count, attention.arch.num_heads * attention.arch.head_dim)
    )
    if "o_proj" in requested:
        captures["o_proj"] = projected_attention
    residual = x + projected_attention
    if "attn_residual" in requested:
        captures["attn_residual"] = residual
    normalized_mlp = layer.post_attn_norm(residual)
    if "post_attn_norm" in requested:
        captures["post_attn_norm"] = normalized_mlp
    gate = layer.mlp.gate_proj(normalized_mlp)
    up = layer.mlp.up_proj(normalized_mlp)
    if "gate_proj" in requested:
        captures["gate_proj"] = gate
    if "up_proj" in requested:
        captures["up_proj"] = up
    silu_gate = torch.nn.functional.silu(gate)
    product = silu_gate * up
    if "silu_gate" in requested:
        captures["silu_gate"] = silu_gate
    if "product" in requested:
        captures["product"] = product
    down = layer.mlp.down_proj(product)
    if "down_proj" in requested:
        captures["down_proj"] = down
    output = residual + down
    if "output" in requested:
        captures["output"] = output
    return output, captures


def _load_layer(source, checkpoint: Path, device: torch.device):
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required to load the bounded block subset") from error
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_SUBSET_BYTES:
        raise ReferenceError("layer-zero subset file is missing or has an unexpected size")
    actual_sha = _sha256_file(checkpoint)
    if actual_sha != CHECKPOINT_SUBSET_SHA256:
        raise ReferenceError("layer-zero subset SHA-256 does not match the pinned inventory")

    arch = source.QWEN3_8B_GEN
    with torch.device("meta"):
        layer = source.GenLayer(arch)
    expected = set(layer.state_dict())
    state: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
            original_keys = set(file.keys())
            expected_keys = {CHECKPOINT_PREFIX + key for key in expected}
            if original_keys != expected_keys:
                raise ReferenceError("checkpoint subset does not contain exactly GenLayer 0")
            for original in sorted(original_keys):
                name = original.removeprefix(CHECKPOINT_PREFIX)
                tensor = file.get_tensor(original)
                if tensor.dtype != torch.float32 or tuple(tensor.shape) != tuple(layer.state_dict()[name].shape):
                    raise ReferenceError(f"checkpoint tensor shape/dtype mismatch: {original}")
                state[name] = tensor
        layer.load_state_dict(state, strict=True, assign=True)
    except ReferenceError:
        raise
    except Exception as error:
        raise ReferenceError(f"cannot load checkpoint layer zero: {error}") from error
    del state
    layer.eval()
    layer.requires_grad_(False)
    # Source preset stores F32 checkpoint weights but runs the decoder in BF16.
    # Transfer the checkpoint subset first, then cast it on the selected device.
    layer.to(device=device)
    layer.to(dtype=torch.bfloat16)
    return layer, actual_sha


def _physical_memory_bytes() -> int:
    try:
        value = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(value)
    except (OSError, ValueError, subprocess.CalledProcessError):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            raise ReferenceError("cannot determine physical memory for the MPS safety floor")


def _capture_bytes_per_token(stage: str, arch) -> int:
    """Storage bytes for one token in a captured intermediate stage."""
    hidden = arch.hidden_size
    q_width = arch.num_heads * arch.head_dim
    kv_width = arch.num_kv_heads * arch.head_dim
    ffn = arch.intermediate_size
    widths = {
        "input_norm_statistics": 4,
        "input_norm_pre_gain": hidden * 2,
        "input_norm": hidden * 2,
        "q_proj": q_width * 2,
        "k_proj": kv_width * 2,
        "v_proj": kv_width * 2,
        "q_norm": q_width * 4,
        "k_norm": kv_width * 4,
        "q_rope": q_width * 4,
        "k_rope": kv_width * 4,
        "q_norm_statistics": arch.num_heads * 2 * 4,
        "k_norm_statistics": arch.num_kv_heads * 2 * 4,
        "q_norm_inverse": arch.num_heads * 2 * 4,
        "k_norm_inverse": arch.num_kv_heads * 2 * 4,
        "q_ready": q_width * 2,
        "k_ready": kv_width * 2,
        "attention": q_width * 2,
        "o_proj": hidden * 2,
        "attn_residual": hidden * 2,
        "post_attn_norm": hidden * 2,
        "gate_proj": ffn * 2,
        "up_proj": ffn * 2,
        "silu_gate": ffn * 2,
        "product": ffn * 2,
        "down_proj": hidden * 2,
        "output": hidden * 2,
    }
    try:
        return widths[stage]
    except KeyError as error:
        raise ReferenceError(f"unknown PixelUMM capture stage: {stage}") from error


def _estimated_peak_bytes(
    grid: tuple[int, int, int], arch, *, layers: int = 1,
    capture_stages: Iterable[str] | None = None,
) -> int:
    if type(layers) is not int or layers not in {1, 2, 4, 8}:
        raise ReferenceError("reference memory estimate supports one, two, four, or eight resident blocks")
    tokens = math.prod(grid)
    h = arch.hidden_size
    f = arch.intermediate_size
    block_weights_f32 = 771_785_728
    block_weights_bf16 = 385_892_864
    hidden = tokens * h * 2
    ffn = tokens * f * 2
    _, height, width = grid
    tiles_h = (height + TILE_TOKENS - 1) // TILE_TOKENS
    tiles_w = (width + TILE_TOKENS - 1) // TILE_TOKENS
    max_q = min(TILE_TOKENS, height) * min(TILE_TOKENS, width) * grid[0]
    max_k = min(3, tiles_h) * min(3, tiles_w) * min(TILE_TOKENS, height) * min(TILE_TOKENS, width) * grid[0]
    score = arch.num_heads * max_q * max_k * 4
    # Loading any later block briefly retains all earlier BF16 blocks, the new
    # block in F32, and its BF16 copy. Thus the conversion peak is one F32
    # block plus `layers` BF16 blocks. Extra hidden slots conservatively cover
    # each retained block-boundary capture while FFN and local scores remain
    # sequential.
    weight_bytes = block_weights_f32 + layers * block_weights_bf16
    hidden_slots = 7 + 4 * (layers - 1)
    captured_bytes = 0
    if capture_stages is not None:
        stages = tuple(capture_stages)
        if not set(stages) <= set(ALL_STAGES):
            raise ReferenceError("memory estimate received an unsupported capture stage")
        captured_bytes = layers * tokens * sum(
            _capture_bytes_per_token(stage, arch) for stage in stages
        )
    return (weight_bytes + hidden_slots * hidden + 4 * ffn + 2 * score
            + 512 * 1024 * 1024 + captured_bytes)


def _estimated_fixture_payload_bytes(
    grid: tuple[int, int, int], arch, *, layers: int,
    capture_stages: Iterable[str],
) -> int:
    """Exact serialized extent for the pinned BF16 inputs, weights and captures."""
    if type(layers) is not int or layers not in {1, 2, 4, 8}:
        raise ReferenceError("fixture payload estimate supports one, two, four, or eight blocks")
    tokens = math.prod(_check_grid(grid))
    stages = tuple(capture_stages)
    if not set(stages) <= set(ALL_STAGES):
        raise ReferenceError("fixture payload estimate received an unsupported capture stage")
    input_bytes = tokens * (arch.hidden_size * 2 + 2 * arch.head_dim * 4)
    weights = layers * 385_892_864
    captures = layers * tokens * sum(
        _capture_bytes_per_token(stage, arch) for stage in stages
    )
    return input_bytes + weights + captures


def mps_memory_floor(
    device: torch.device, grid: tuple[int, int, int], arch, *, layers: int = 1,
    capture_stages: Iterable[str] | None = None,
) -> dict[str, int | bool]:
    if device.type != "mps":
        return {"applies": False}
    total = _physical_memory_bytes()
    allocated = int(torch.mps.driver_allocated_memory())
    estimated = _estimated_peak_bytes(
        grid, arch, layers=layers, capture_stages=capture_stages
    )
    floor = total // 10
    if total - allocated - estimated < floor:
        raise ReferenceError(
            "MPS reference preflight would leave less than the required 10% physical-memory floor"
        )
    return {
        "applies": True, "physical_bytes": total, "driver_allocated_bytes": allocated,
        "estimated_peak_bytes": estimated, "required_floor_bytes": floor,
        "resident_blocks": layers,
    }


def _tensor_dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.bfloat16:
        return "BF16"
    if tensor.dtype == torch.float32:
        return "F32"
    if tensor.dtype == torch.int32:
        return "I32"
    raise ReferenceError(f"fixture does not support tensor dtype {tensor.dtype}")


def _write_fixture(path: Path, *, metadata: Mapping, inputs: Mapping[str, torch.Tensor],
                   weights: Mapping[str, torch.Tensor], captures: Mapping[str, torch.Tensor],
                   schema: str = SCHEMA) -> dict:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = path.with_suffix(".bin")
    if path.exists() or path.is_symlink() or payload_path.exists() or payload_path.is_symlink():
        raise ReferenceError("refusing to replace an existing reference JSON or payload")
    payload_digest = hashlib.sha256()
    payload_bytes = 0
    manifest = {
        **metadata,
        "schema": schema,
        "payload": payload_path.name,
        "inputs": {}, "weights": {}, "captures": {},
    }
    with payload_path.open("xb") as stream:
        for group_name, tensors in (("inputs", inputs), ("weights", weights), ("captures", captures)):
            for name, tensor in tensors.items():
                cpu = tensor.detach().contiguous().cpu()
                dtype_name = _tensor_dtype_name(cpu)
                array = cpu.view(torch.uint8).reshape(-1).numpy()
                byte_view = memoryview(array).cast("B")
                nbytes = len(byte_view)
                expected_bytes = cpu.numel() * cpu.element_size()
                if nbytes != expected_bytes:
                    raise ReferenceError(f"unexpected serialized tensor extent for {name}")
                record_digest = hashlib.sha256()
                chunk_size = 8 * 1024 * 1024
                for first in range(0, nbytes, chunk_size):
                    chunk = byte_view[first:first + chunk_size]
                    stream.write(chunk)
                    payload_digest.update(chunk)
                    record_digest.update(chunk)
                manifest[group_name][name] = {
                    "dtype": dtype_name,
                    "shape": list(cpu.shape),
                    "offset": payload_bytes,
                    "nbytes": nbytes,
                    "sha256": record_digest.hexdigest(),
                }
                payload_bytes += nbytes
        stream.flush()
        os.fsync(stream.fileno())
    manifest["payload_bytes"] = payload_bytes
    manifest["payload_sha256"] = payload_digest.hexdigest()
    manifest["capture_order"] = list(captures)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def _read_receipt(path: Path) -> tuple[dict, str]:
    path = path.expanduser().resolve()
    try:
        raw = path.read_bytes()
        value = checkpoint_inventory.parse_header(raw)
    except (OSError, checkpoint_inventory.InventoryError) as error:
        raise ReferenceError(f"cannot read block subset receipt: {path.name}") from error
    return value, hashlib.sha256(raw).hexdigest()


def _load_layer_for_block(
    source, checkpoint: Path, receipt_path: Path, header_path: Path,
    device: torch.device, block_index: int,
):
    """Load one individually receipted source block into the pinned GenLayer."""
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required to load bounded block subsets") from error
    checkpoint = checkpoint.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    header_path = header_path.expanduser().resolve()
    source_root = checkpoint_inventory.ROOT.resolve()
    for path in (checkpoint, receipt_path, header_path):
        if path.parent != (source_root / "source").resolve():
            raise ReferenceError("block reference source files must be in the task source directory")
    try:
        raw_header = header_path.read_bytes()
        receipt, receipt_sha = _read_receipt(receipt_path)
        checked = checkpoint_inventory.validate_block_receipt(
            raw_header, receipt, checkpoint, block_index
        )
    except (OSError, checkpoint_inventory.InventoryError) as error:
        raise ReferenceError(f"block-{block_index} source receipt rejected: {error}") from error

    arch = source.QWEN3_8B_GEN
    with torch.device("meta"):
        layer = source.GenLayer(arch)
    expected = set(layer.state_dict())
    prefix = f"decoder.layers.{block_index}."
    state: dict[str, torch.Tensor] = {}
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
            original_keys = set(file.keys())
            expected_keys = {prefix + key for key in expected}
            if original_keys != expected_keys:
                raise ReferenceError(f"block-{block_index} file does not contain exactly 13 expected tensors")
            for original in sorted(original_keys):
                name = original.removeprefix(prefix)
                tensor = file.get_tensor(original)
                if (tensor.dtype != torch.float32
                        or tuple(tensor.shape) != tuple(layer.state_dict()[name].shape)):
                    raise ReferenceError(f"block-{block_index} checkpoint tensor shape/dtype mismatch: {original}")
                state[name] = tensor
        layer.load_state_dict(state, strict=True, assign=True)
    except ReferenceError:
        raise
    except Exception as error:
        raise ReferenceError(f"cannot load checkpoint block {block_index}: {error}") from error
    del state
    layer.eval()
    layer.requires_grad_(False)
    # Match the pinned source runtime order: transfer F32, then cast on device.
    layer.to(device=device)
    layer.to(dtype=torch.bfloat16)
    return layer, checked, receipt_sha


def make_fixture(
    *, upstream: Path | str, checkpoint: Path | str, output: Path | str,
    grid: Iterable[int], device_name: str = "mps", capture_mode: str = "auto",
    seed: int = 20261026,
) -> dict:
    output_path = Path(output).expanduser().absolute()
    payload_path = output_path.with_suffix(".bin")
    if (output_path.exists() or output_path.is_symlink()
            or payload_path.exists() or payload_path.is_symlink()):
        raise ReferenceError("refusing to replace an existing reference JSON or payload")
    grid = _check_grid(grid)
    if device_name not in {"cpu", "mps"}:
        raise ReferenceError("device must be cpu or mps")
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is not available")
    if capture_mode not in {"auto", "all", "boundary"}:
        raise ReferenceError("capture mode must be auto, all, or boundary")
    count = math.prod(grid)
    if count <= 2048:
        chosen = ALL_STAGES if capture_mode in {"auto", "all"} else FULL_TARGET_STAGES
    else:
        chosen = FULL_TARGET_STAGES if capture_mode in {"auto", "boundary"} else ALL_STAGES
    if count > 2048 and chosen == ALL_STAGES:
        raise ReferenceError("all-stage captures are restricted to at most 2048 tokens")

    source, source_sha = load_upstream(upstream)
    plan, _, _ = make_attention_plan(source, grid)
    arch = source.QWEN3_8B_GEN
    memory = mps_memory_floor(
        device, grid, arch, capture_stages=chosen
    )
    layer, checkpoint_sha = _load_layer(source, Path(checkpoint), device)

    # Generate a reproducible synthetic hidden state on CPU before any GPU math.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_original = torch.randn((1, count, arch.hidden_size), generator=generator, dtype=torch.float32)
    x_original = x_original.to(device=device).to(dtype=torch.bfloat16)
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    x = x_original.index_select(1, permutation)
    del x_original
    cos, sin = source.mrope_tables(arch, grid, device)
    cos = cos.index_select(1, permutation)
    sin = sin.index_select(1, permutation)
    with torch.inference_mode():
        # The upstream decoder runs this layer after moving the complete model
        # to BF16. Explicit autocast mirrors its BF16 inference context.
        context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        with context:
            result, captures = run_layer_reference(
                layer, x, cos, sin, plan, capture_stages=chosen
            )
    if device.type == "mps":
        # One terminal synchronization before any artifact readback.
        torch.mps.synchronize()

    state = {name: parameter.detach() for name, parameter in layer.named_parameters()}
    if len(state) != 13:
        raise ReferenceError("loaded layer parameter inventory changed")
    inputs = {"x": x, "cos": cos, "sin": sin}
    meta = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_path": UPSTREAM_MODEL_PATH,
        "upstream_model_sha256": source_sha,
        "checkpoint_subset_sha256": checkpoint_sha,
        "grid": list(grid),
        "tile_tokens": TILE_TOKENS,
        "token_order": "tile_major",
        "permutation_sha256": plan.permutation_sha256,
        "runtime_dtype": "BF16",
        "input_generator": {"distribution": "standard_normal", "seed": seed, "cast": "BF16"},
        "source_config": asdict(arch),
        "attention_adapter": "gathered_spatial_tiles_qk_softmax_pv_fp32_accum",
        "gqa_group_size": arch.num_heads // arch.num_kv_heads,
        "comparison": {"atol": ATOL, "rtol": RTOL},
        "memory_preflight": memory,
        "captures_are_tile_major": True,
    }
    fixture = _write_fixture(
        Path(output), metadata=meta, inputs=inputs, weights=state, captures=captures
    )
    return {
        "fixture": str(Path(output).expanduser().resolve()),
        "payload": str(Path(output).expanduser().resolve().with_suffix(".bin")),
        "grid": list(grid), "tokens": count, "capture_order": fixture["capture_order"],
        "payload_bytes": fixture["payload_bytes"],
        "estimated_payload_bytes": _estimated_fixture_payload_bytes(
            grid, arch, layers=1, capture_stages=chosen
        ),
        "payload_sha256": fixture["payload_sha256"],
    }


def _block_io_edge(index: int) -> dict[str, str]:
    """Describe the predecessor edge for one of the admitted block slots."""
    if type(index) is not int or not 0 <= index < 8:
        raise ReferenceError("block I/O edges are defined for block indices zero through seven")
    return {
        "input_source": "inputs.x" if index == 0 else f"captures.block{index - 1}.output",
        "output_key": f"captures.block{index}.output",
    }


def _make_block_sequence_fixture(
    *, upstream: Path | str,
    block_sources: Iterable[tuple[Path | str, Path | str]],
    header: Path | str, output: Path | str, grid: Iterable[int],
    schema: str, device_name: str = "mps", capture_mode: str = "auto",
    seed: int = 20261026,
) -> dict:
    """Execute an admitted block sequence on one device and retain its edges."""
    block_sources = tuple(block_sources)
    if len(block_sources) not in {2, 4, 8}:
        raise ReferenceError("reference fixture requires exactly two, four, or eight consecutive blocks")
    expected_schema = {
        2: TWO_BLOCK_SCHEMA,
        4: FOUR_BLOCK_SCHEMA,
        8: EIGHT_BLOCK_SCHEMA,
    }[len(block_sources)]
    if schema != expected_schema:
        raise ReferenceError("block count and reference schema do not match")
    output_path = Path(output).expanduser().absolute()
    payload_path = output_path.with_suffix(".bin")
    if output_path.parent.resolve() != (checkpoint_inventory.ROOT / "results").resolve():
        raise ReferenceError("multi-block reference output must be in the task results directory")
    if (output_path.exists() or output_path.is_symlink()
            or payload_path.exists() or payload_path.is_symlink()):
        raise ReferenceError("refusing to replace an existing multi-block reference JSON or payload")
    grid = _check_grid(grid)
    if device_name not in {"cpu", "mps"}:
        raise ReferenceError("device must be cpu or mps")
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is not available")
    if capture_mode not in {"auto", "all", "boundary"}:
        raise ReferenceError("capture mode must be auto, all, or boundary")
    count = math.prod(grid)
    if count <= 2048:
        chosen = ALL_STAGES if capture_mode in {"auto", "all"} else FULL_TARGET_STAGES
    else:
        chosen = FULL_TARGET_STAGES if capture_mode in {"auto", "boundary"} else ALL_STAGES
    if count > 2048 and chosen == ALL_STAGES:
        raise ReferenceError("all-stage captures are restricted to at most 2048 tokens")

    source, source_sha = load_upstream(upstream)
    plan, _, _ = make_attention_plan(source, grid)
    arch = source.QWEN3_8B_GEN
    memory = mps_memory_floor(
        device, grid, arch, layers=len(block_sources), capture_stages=chosen
    )
    header_path = Path(header)
    layers = []
    subset_provenance = []
    for index, (checkpoint_path, receipt_path) in enumerate(block_sources):
        layer, checked, receipt_sha = _load_layer_for_block(
            source, Path(checkpoint_path), Path(receipt_path), header_path, device, index
        )
        layers.append(layer)
        subset_provenance.append({
            "index": index,
            "checkpoint_subset_sha256": checked["subset_sha256"],
            "checkpoint_subset_bytes": checked["subset_bytes"],
            "checkpoint_subset_file": checked["subset_file"],
            "source_receipt_sha256": receipt_sha,
            "source_extent": checked["block_data_offsets"],
            "legacy_extent_derived_from_pinned_header": checked[
                "legacy_extent_derived_from_pinned_header"
            ],
        })

    # Generate one initial prefix and one tile-major RoPE table for all blocks.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_original = torch.randn((1, count, arch.hidden_size), generator=generator, dtype=torch.float32)
    x_original = x_original.to(device=device).to(dtype=torch.bfloat16)
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    x = x_original.index_select(1, permutation)
    del x_original
    cos, sin = source.mrope_tables(arch, grid, device)
    cos = cos.index_select(1, permutation)
    sin = sin.index_select(1, permutation)

    block_captures: dict[str, torch.Tensor] = {}
    current = x
    with torch.inference_mode():
        context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        with context:
            for index, layer in enumerate(layers):
                result, captures = run_layer_reference(
                    layer, current, cos, sin, plan, capture_stages=chosen
                )
                for stage in chosen:
                    block_captures[f"block{index}.{stage}"] = captures[stage]
                # No inter-block device-to-host handoff occurs: this tensor
                # feeds the next layer directly on the selected device.
                current = result
    if device.type == "mps":
        # Single terminal synchronization after the entire decoder prefix.
        torch.mps.synchronize()

    weights: dict[str, torch.Tensor] = {}
    blocks_metadata = []
    for index, layer in enumerate(layers):
        prefix = f"block{index}."
        state = {name: parameter.detach() for name, parameter in layer.named_parameters()}
        if len(state) != 13:
            raise ReferenceError(f"loaded block-{index} parameter inventory changed")
        for name in sorted(state):
            weights[prefix + name] = state[name]
        weight_keys = [prefix + name for name in sorted(state)]
        capture_keys = [f"block{index}.{stage}" for stage in chosen]
        blocks_metadata.append({
            "index": index,
            "checkpoint_subset_sha256": subset_provenance[index]["checkpoint_subset_sha256"],
            "checkpoint_subset_bytes": subset_provenance[index]["checkpoint_subset_bytes"],
            "source_receipt_sha256": subset_provenance[index]["source_receipt_sha256"],
            "source_extent": subset_provenance[index]["source_extent"],
            "weight_prefix": prefix,
            "weight_keys": weight_keys,
            "capture_keys": capture_keys,
            **_block_io_edge(index),
        })
    metadata = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_path": UPSTREAM_MODEL_PATH,
        "upstream_model_sha256": source_sha,
        "hf_revision": checkpoint_inventory.REVISION,
        "hf_source_path": checkpoint_inventory.SOURCE_PATH,
        "checkpoint_header_sha256": checkpoint_inventory.HEADER_SHA256,
        "blocks": blocks_metadata,
        "grid": list(grid),
        "tile_tokens": TILE_TOKENS,
        "token_order": "tile_major",
        "permutation_sha256": plan.permutation_sha256,
        "runtime_dtype": "BF16",
        "input_generator": {"distribution": "standard_normal", "seed": seed, "cast": "BF16"},
        "source_config": asdict(arch),
        "attention_adapter": "gathered_spatial_tiles_qk_softmax_pv_fp32_accum",
        "gqa_group_size": arch.num_heads // arch.num_kv_heads,
        "comparison": {"atol": ATOL, "rtol": RTOL},
        "memory_preflight": memory,
        "captures_are_tile_major": True,
        "block_execution": "same_device_output_to_next_input",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    fixture = _write_fixture(
        output_path, metadata=metadata, inputs={"x": x, "cos": cos, "sin": sin},
        weights=weights, captures=block_captures, schema=schema,
    )
    return {
        "fixture": str(output_path.resolve()),
        "payload": str(payload_path.resolve()),
        "grid": list(grid), "tokens": count, "capture_order": fixture["capture_order"],
        "payload_bytes": fixture["payload_bytes"],
        "estimated_payload_bytes": _estimated_fixture_payload_bytes(
            grid, arch, layers=len(block_sources), capture_stages=chosen
        ),
        "payload_sha256": fixture["payload_sha256"],
        "blocks": subset_provenance,
        "estimated_peak_bytes": memory.get("estimated_peak_bytes"),
    }


def make_two_block_fixture(
    *, upstream: Path | str, block0: Path | str, block1: Path | str,
    receipt0: Path | str, receipt1: Path | str, header: Path | str,
    output: Path | str, grid: Iterable[int], device_name: str = "mps",
    capture_mode: str = "auto", seed: int = 20261026,
) -> dict:
    """Backward-compatible pair entry point and schema."""
    return _make_block_sequence_fixture(
        upstream=upstream,
        block_sources=((block0, receipt0), (block1, receipt1)),
        header=header, output=output, grid=grid, schema=TWO_BLOCK_SCHEMA,
        device_name=device_name, capture_mode=capture_mode, seed=seed,
    )


def make_four_block_fixture(
    *, upstream: Path | str,
    block0: Path | str, block1: Path | str, block2: Path | str, block3: Path | str,
    receipt0: Path | str, receipt1: Path | str, receipt2: Path | str, receipt3: Path | str,
    header: Path | str, output: Path | str, grid: Iterable[int],
    device_name: str = "mps", capture_mode: str = "auto", seed: int = 20261026,
) -> dict:
    """Run blocks zero through three sequentially on one device."""
    return _make_block_sequence_fixture(
        upstream=upstream,
        block_sources=(
            (block0, receipt0), (block1, receipt1),
            (block2, receipt2), (block3, receipt3),
        ),
        header=header, output=output, grid=grid, schema=FOUR_BLOCK_SCHEMA,
        device_name=device_name, capture_mode=capture_mode, seed=seed,
    )


def make_eight_block_fixture(
    *, upstream: Path | str,
    block0: Path | str, block1: Path | str, block2: Path | str, block3: Path | str,
    block4: Path | str, block5: Path | str, block6: Path | str, block7: Path | str,
    receipt0: Path | str, receipt1: Path | str, receipt2: Path | str, receipt3: Path | str,
    receipt4: Path | str, receipt5: Path | str, receipt6: Path | str, receipt7: Path | str,
    header: Path | str, output: Path | str, grid: Iterable[int],
    device_name: str = "mps", capture_mode: str = "auto", seed: int = 20261026,
) -> dict:
    """Run blocks zero through seven sequentially on one device."""
    return _make_block_sequence_fixture(
        upstream=upstream,
        block_sources=(
            (block0, receipt0), (block1, receipt1),
            (block2, receipt2), (block3, receipt3),
            (block4, receipt4), (block5, receipt5),
            (block6, receipt6), (block7, receipt7),
        ),
        header=header, output=output, grid=grid, schema=EIGHT_BLOCK_SCHEMA,
        device_name=device_name, capture_mode=capture_mode, seed=seed,
    )


def geometry_report(upstream: Path | str, grid: Iterable[int]) -> dict:
    source, source_sha = load_upstream(upstream)
    plan, _, _ = make_attention_plan(source, grid)
    max_q = max(region.query_stop - region.query_start for region in plan.regions)
    max_k = max(len(region.key_positions) for region in plan.regions)
    return {
        "schema": "cogni_ml.pixelumm_block_geometry/v1",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_sha256": source_sha,
        "grid": list(plan.grid), "tile_tokens": plan.tile,
        "tokens": math.prod(plan.grid), "tiles": len(plan.regions),
        "max_query_tokens_per_tile": max_q, "max_key_tokens_per_tile": max_k,
        "permutation_sha256": plan.permutation_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--block0", type=Path)
    parser.add_argument("--block1", type=Path)
    parser.add_argument("--block2", type=Path)
    parser.add_argument("--block3", type=Path)
    parser.add_argument("--block4", type=Path)
    parser.add_argument("--block5", type=Path)
    parser.add_argument("--block6", type=Path)
    parser.add_argument("--block7", type=Path)
    parser.add_argument("--receipt0", type=Path)
    parser.add_argument("--receipt1", type=Path)
    parser.add_argument("--receipt2", type=Path)
    parser.add_argument("--receipt3", type=Path)
    parser.add_argument("--receipt4", type=Path)
    parser.add_argument("--receipt5", type=Path)
    parser.add_argument("--receipt6", type=Path)
    parser.add_argument("--receipt7", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid", required=True, type=int, nargs=3, metavar=("T", "H", "W"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--capture", choices=("auto", "all", "boundary"), default="auto")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.geometry_only:
            print(json.dumps(geometry_report(args.upstream, args.grid), sort_keys=True))
            return 0
        block_paths = tuple(getattr(args, f"block{index}") for index in range(8))
        receipt_paths = tuple(getattr(args, f"receipt{index}") for index in range(8))
        if any(path is not None for path in (*block_paths, *receipt_paths)):
            if args.checkpoint is not None:
                parser.error("--checkpoint cannot be combined with block inputs")
            if args.header is None or args.output is None:
                parser.error("block reference requires --header and --output")
            present = []
            for index, (subset, receipt) in enumerate(zip(block_paths, receipt_paths)):
                if (subset is None) != (receipt is None):
                    parser.error(f"block {index} requires both --block{index} and --receipt{index}")
                if subset is not None:
                    present.append(index)
            count = len(present)
            if count not in {2, 4, 8} or present != list(range(count)):
                parser.error("block inputs must be consecutive indices 0-1, 0-3, or 0-7")
            common = {
                "upstream": args.upstream, "header": args.header,
                "output": args.output, "grid": args.grid,
                "device_name": args.device, "capture_mode": args.capture,
                "seed": args.seed,
            }
            if count == 2:
                report = make_two_block_fixture(
                    **common,
                    block0=block_paths[0], block1=block_paths[1],
                    receipt0=receipt_paths[0], receipt1=receipt_paths[1],
                )
            elif count == 4:
                report = make_four_block_fixture(
                    **common,
                    **{f"block{i}": block_paths[i] for i in range(4)},
                    **{f"receipt{i}": receipt_paths[i] for i in range(4)},
                )
            else:
                report = make_eight_block_fixture(
                    **common,
                    **{f"block{i}": block_paths[i] for i in range(8)},
                    **{f"receipt{i}": receipt_paths[i] for i in range(8)},
                )
        else:
            if args.checkpoint is None or args.output is None:
                parser.error("--checkpoint and --output are required unless --geometry-only or pair inputs are used")
            report = make_fixture(
                upstream=args.upstream, checkpoint=args.checkpoint, output=args.output,
                grid=args.grid, device_name=args.device, capture_mode=args.capture, seed=args.seed,
            )
        print(json.dumps(report, sort_keys=True))
        return 0
    except ReferenceError as error:
        print(f"PixelUMM reference rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
