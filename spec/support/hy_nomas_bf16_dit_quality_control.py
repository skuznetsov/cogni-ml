#!/usr/bin/env python3
"""Run a bounded, source-pinned five-second Hy-Nomas DiT arithmetic control.

The default `source-bf16` mode preserves the BF16 arithmetic control. The
`native-f32` mode uses pinned upstream model bodies with F32 activations and one
reusable F32 DiT block, while reproducing native BF16 Euler rounding boundaries.
Both modes execute one 30-latent-frame segment for four steps with authenticated
Echo noise, Qwen crop, first RoPE bank, and original BF16 checkpoint weights.
The only replaced operation is attention dispatch: chunked PyTorch MPS SDPA
admits the valid 557 text rows while keeping the full 1280-row Qwen/refiner
tensors. Neither mode establishes Magi/CUDA parity, Qwen inference, cache
history, PixelUMM decoding, or a complete video.

The CPU-only --validate-only path imports PyTorch and the pinned Euler
dependencies, then constructs the four-step schedule on CPU; it does not use
an MPS device or allocate model weights. A full run requires an external
guarded launcher; --planned-cap-gib records the planned process cap (at most
26 GiB) but does not implement an OS memory limit.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_continuation_inputs as continuation_inputs  # noqa: E402
import hy_nomas_q4_dit_reference as cache_reference  # noqa: E402
import hy_nomas_real_attention_reference as attention_reference  # noqa: E402
import hy_nomas_reference as block_reference  # noqa: E402


SCHEMA = "cogni_ml.hy_nomas_bf16_dit_quality_control/v1"
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
CONDITIONING_SHA256 = "2807e11814f07975c807f367b0a4f15e3c6bdde1eb62ce9bda851814d09a6e47"
CACHE_INDEX_SHA256 = "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0"
EULER_RELATIVE = "hymm/pipeline/solvers/euler.py"
EULER_SHA256 = "e7996f947b46835f97b66059e0f43019b744ecec5cda380432124f70e169e59a"
DEFAULT_CONDITIONING = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-echo-20261030/conditioning/"
    "continuation-inputs-echo-20s.json"
)
DEFAULT_CACHE_INDEX = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/weights/q4-cache-v2/index.json"
)
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
MAX_PLANNED_CAP_GIB = 26.0
DEFAULT_PLANNED_CAP_GIB = 26.0
EXECUTION_MODES = ("source-bf16", "native-f32")
SEGMENT_INDEX = 0
SEGMENT_LATENTS = 30
STEPS = 4
FLOW_SHIFT = 5.0
VIDEO_TOKENS = 30 * 9 * 16
TEXT_TOKENS = 576
VALID_TEXT_TOKENS = 557
TEXT_RMS_TOKENS = 1280
DIM = 2048
HEADS = 16
HEAD_DIM = 128
QUERY_CHUNK = attention_reference.QUERY_CHUNK
HASH_CHUNK = 8 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class ControlError(RuntimeError):
    """Fail-closed artifact admission, source binding, or execution error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = stream.read(min(remaining, HASH_CHUNK))
            if not chunk:
                raise ControlError(f"truncated authenticated tensor range: {path.name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _canonical_file(path_arg: str | Path, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ControlError(f"{label} does not exist: {path}") from None
    if path.is_symlink() or not resolved.is_file():
        raise ControlError(f"{label} must be a non-symlink regular file: {path}")
    return resolved


def _canonical_dir(path_arg: str | Path, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ControlError(f"{label} does not exist: {path}") from None
    if path.is_symlink() or not resolved.is_dir():
        raise ControlError(f"{label} must be a non-symlink directory: {path}")
    return resolved


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    """Read only the bounded JSON header; full file bytes are SHA-checked separately."""
    with path.open("rb") as stream:
        raw_len = stream.read(8)
        if len(raw_len) != 8:
            raise ControlError(f"safetensors header is truncated: {path.name}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 1 or header_len > 64 * 1024 * 1024:
            raise ControlError(
                f"safetensors header length is outside its bound: {path.name}"
            )
        header_raw = stream.read(header_len)
    if len(header_raw) != header_len:
        raise ControlError(f"safetensors header bytes are truncated: {path.name}")
    try:
        header = attention_reference._json_no_duplicates(
            header_raw, f"{path.name} header"
        )
    except Exception as error:
        raise ControlError(f"invalid safetensors header {path.name}: {error}") from None
    data_start = 8 + header_len
    file_bytes = path.stat().st_size
    ranges: list[tuple[int, int, str]] = []
    for name, record in header.items():
        if name == "__metadata__":
            if type(record) is not dict:
                raise ControlError(f"malformed safetensors metadata: {path.name}")
            continue
        if (
            type(record) is not dict
            or set(record) != {"dtype", "shape", "data_offsets"}
            or record.get("dtype") != "BF16"
        ):
            raise ControlError(
                f"safetensors tensor is not the admitted BF16 form: {name}"
            )
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if (
            type(shape) is not list
            or not shape
            or any(type(value) is not int or value <= 0 for value in shape)
            or type(offsets) is not list
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ControlError(f"malformed BF16 tensor descriptor: {name}")
        start, stop = offsets
        if start < 0 or stop < start or stop - start != math.prod(shape) * 2:
            raise ControlError(f"BF16 tensor byte extent mismatch: {name}")
        ranges.append((start, stop, name))
    ranges.sort()
    cursor = 0
    for start, stop, name in ranges:
        if start != cursor or stop > file_bytes - data_start:
            raise ControlError(
                f"safetensors tensor extents overlap or leave a gap: {name}"
            )
        cursor = stop
    if data_start + cursor != file_bytes:
        raise ControlError(
            f"safetensors data extent differs from file size: {path.name}"
        )
    return header, file_bytes


def _validate_cache_shard(
    path: Path,
    expected_sha: str,
    expected_names_shapes: dict[str, list[int]] | None,
    *,
    global_file: bool = False,
) -> dict[str, Any]:
    path = _canonical_file(path, "BF16 cache shard")
    identity = _file_identity(path)
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ControlError(f"BF16 cache shard hash mismatch: {path.name}")
    header, file_bytes = _read_safetensors_header(path)
    tensors = {
        name: record for name, record in header.items() if name != "__metadata__"
    }
    if global_file:
        expected_counts = cache_reference._GLOBAL_EXPECTED_COUNTS
        actual_counts = {
            prefix: sum(name.startswith(prefix) for name in tensors)
            for prefix in expected_counts
        }
        if actual_counts != expected_counts or sum(actual_counts.values()) != len(
            tensors
        ):
            raise ControlError(
                "global BF16 tensor inventory differs from pinned modules"
            )
    elif expected_names_shapes is not None:
        if set(tensors) != set(expected_names_shapes):
            raise ControlError(f"BF16 block tensor inventory mismatch: {path.name}")
        for name, shape in expected_names_shapes.items():
            if tensors[name]["shape"] != shape:
                raise ControlError(f"BF16 block tensor shape mismatch: {name}")
    if _file_identity(path) != identity:
        raise ControlError(f"BF16 cache shard changed during validation: {path.name}")
    return {
        "path": str(path),
        "bytes": file_bytes,
        "sha256": actual_sha,
        "tensor_count": len(tensors),
        "identity": identity,
    }


def _host_memory_snapshot() -> dict[str, Any]:
    try:
        import psutil
    except ImportError:
        return {"available": False, "source": "psutil unavailable"}
    memory = psutil.virtual_memory()
    return {
        "available": True,
        "source": "psutil.virtual_memory",
        "total_bytes": int(memory.total),
        "available_bytes": int(memory.available),
        "used_bytes": int(memory.used),
        "percent_used": float(memory.percent),
    }


def _authenticate(args: argparse.Namespace) -> dict[str, Any]:
    conditioning_path = _canonical_file(
        args.conditioning_manifest, "Echo continuation manifest"
    )
    if _sha256(conditioning_path) != CONDITIONING_SHA256:
        raise ControlError("Echo continuation manifest differs from the pinned source")
    cache_index = _canonical_file(args.cache_index, "BF16 cache index")
    if _sha256(cache_index) != CACHE_INDEX_SHA256:
        raise ControlError("BF16 cache index differs from its pinned SHA-256")
    try:
        index = attention_reference._json_no_duplicates(
            cache_index.read_bytes(), "BF16 cache index"
        )
        cache_reference.validate_cache_index_document(index)
    except Exception as error:
        raise ControlError(f"BF16 cache index rejected: {error}") from None

    # Reuse the full authenticated Echo/Qwen/range/RoPE and index admission
    # path from the real-attention reference. Its capture-only arguments are
    # bound here to the pinned first shard and globals from the same index.
    first = index["blocks"][0]
    global_path = cache_index.parent / index["globals"]["bf16"]
    synthetic_binding = {
        "conditioning_manifest": str(conditioning_path),
        "conditioning_sha256": CONDITIONING_SHA256,
        "cache_index": str(cache_index),
        "cache_index_sha256": CACHE_INDEX_SHA256,
        "weight_index_sha256": CACHE_INDEX_SHA256,
        "block_weights": str(cache_index.parent / first["bf16"]),
        "block_weights_sha256": first["bf16_sha256"],
        "global_weights": str(global_path),
        "global_weights_sha256": index["globals"]["sha256"],
    }
    try:
        source_info, prelude_inputs = attention_reference._authenticate_sources(
            synthetic_binding
        )
    except Exception as error:
        raise ControlError(
            f"authenticated Echo source admission failed: {error}"
        ) from None

    shard_records: list[dict[str, Any]] = []
    cache_identities: dict[str, tuple[int, int, int, int, int]] = {}
    total_weight_bytes = 0
    for block_index, entry in enumerate(index["blocks"]):
        block_path = cache_index.parent / entry["bf16"]
        expected = {
            name: shape
            for name, shape in cache_reference.expected_block_shapes(
                block_index
            ).items()
        }
        record = _validate_cache_shard(block_path, entry["bf16_sha256"], expected)
        if record["tensor_count"] != len(expected):
            raise ControlError(
                f"block {block_index} must contain exactly 32 BF16 tensors"
            )
        shard_records.append(record)
        cache_identities[str(block_path)] = record["identity"]
        total_weight_bytes += record["bytes"]
    global_record = _validate_cache_shard(
        global_path,
        index["globals"]["sha256"],
        None,
        global_file=True,
    )
    cache_identities[str(global_path)] = global_record["identity"]
    total_weight_bytes += global_record["bytes"]

    upstream = _canonical_dir(args.upstream, "pinned upstream checkout")
    try:
        upstream_revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if upstream_revision != PINNED_REVISION:
            raise ControlError("pinned upstream revision mismatch")
        source_hashes = block_reference._verify_upstream(upstream)
        prelude_hashes = attention_reference._verify_prelude_sources(upstream)
        _, continuation_hashes = continuation_inputs._verify_upstream(upstream)
    except ControlError:
        raise
    except Exception as error:
        raise ControlError(
            f"pinned upstream source admission failed: {error}"
        ) from None
    euler_path = upstream / EULER_RELATIVE
    euler_sha = _sha256(euler_path)
    if euler_sha != EULER_SHA256:
        raise ControlError("pinned upstream EulerSolver source digest mismatch")

    condition = attention_reference._json_no_duplicates(
        conditioning_path.read_bytes(),
        "Echo continuation manifest",
    )
    payload = _canonical_file(
        conditioning_path.parent / condition["payload"], "Echo payload"
    )
    base_manifest = _canonical_file(
        condition["qwen_input_bundle"]["manifest_path"],
        "base Qwen manifest",
    )
    attention_reference._json_no_duplicates(
        base_manifest.read_bytes(), "base Qwen manifest"
    )
    latent = condition["tensors"]["input.latent_bank"]
    rope_cos = condition["tensors"]["input.rope_cos"]
    rope_sin = condition["tensors"]["input.rope_sin"]
    latent_prefix_bytes = SEGMENT_LATENTS * 32 * 9 * 16 * 4
    rope_segment_bytes = VIDEO_TOKENS * HEAD_DIM * 4
    if (
        latent["dtype"] != "F32"
        or latent["shape"] != [1, 32, 180, 9, 16]
        or rope_cos["dtype"] != "F32"
        or rope_sin["dtype"] != "F32"
        or rope_cos["shape"] != [6, VIDEO_TOKENS, HEAD_DIM]
        or rope_sin["shape"] != [6, VIDEO_TOKENS, HEAD_DIM]
        or latent_prefix_bytes > latent["nbytes"]
        or rope_segment_bytes * 6 != rope_cos["nbytes"]
        or rope_segment_bytes * 6 != rope_sin["nbytes"]
    ):
        raise ControlError("authenticated first-segment latent/RoPE geometry mismatch")
    first_rope_digests = {}
    for key, record in (("cos", rope_cos), ("sin", rope_sin)):
        digest = _sha256_range(payload, record["offset"], rope_segment_bytes)
        if digest != condition["qwen_input_bundle"][f"first_rope_{key}_sha256"]:
            raise ControlError(
                f"Echo continuation RoPE bank 0 differs from its Qwen binding: {key}"
            )
        first_rope_digests[key] = digest

    return {
        "conditioning_path": str(conditioning_path),
        "conditioning": condition,
        "conditioning_payload": str(payload),
        "base_qwen": prelude_inputs,
        "source_info": source_info,
        "cache_index_path": str(cache_index),
        "cache_index": index,
        "cache_shards": shard_records,
        "global_shard": global_record,
        "cache_identities": cache_identities,
        "total_weight_bytes": total_weight_bytes,
        "upstream_path": str(upstream),
        "upstream_revision": upstream_revision,
        "upstream_source_sha256": source_hashes,
        "prelude_source_sha256": prelude_hashes,
        "continuation_upstream_source_sha256": continuation_hashes,
        "euler_source_sha256": euler_sha,
        "latent_record": latent,
        "rope_cos_record": rope_cos,
        "rope_sin_record": rope_sin,
        "first_rope_digests": first_rope_digests,
    }


class _NoopCompare:
    """Adapter sink; the quality control retains no activation checkpoints."""

    def update(self, _name: str, _tensor: Any) -> None:
        return None


def _assert_cache_sources_unchanged(admitted: dict[str, Any]) -> None:
    for path, identity in admitted["cache_identities"].items():
        if _file_identity(Path(path)) != identity:
            raise ControlError(
                f"authenticated BF16 shard changed before load: {Path(path).name}"
            )


def _load_all_blocks(
    torch: Any,
    admitted: dict[str, Any],
    helpers: tuple[Any, ...],
    attention_adapter: Any,
    device: Any,
) -> list[Any]:
    from safetensors import safe_open

    block_type = block_reference._extract_pinned_block(
        Path(admitted["upstream_path"]),
        torch,
        attention_adapter,
        helpers[-1].apply_rotary_emb,
        helpers,
    )
    blocks = []
    cache_dir = Path(admitted["cache_index_path"]).parent
    for block_index, entry in enumerate(admitted["cache_index"]["blocks"]):
        _assert_cache_sources_unchanged(admitted)
        path = _canonical_file(
            cache_dir / entry["bf16"], f"block {block_index} BF16 weights"
        )
        expected = cache_reference._BLOCK_RELATIVE_SHAPES
        source_to_relative = {
            f"double_blocks.{block_index}.{key}": key for key in expected
        }
        try:
            with safe_open(str(path), framework="pt", device="cpu") as source:
                if set(source.keys()) != set(source_to_relative):
                    raise ControlError(
                        f"block {block_index} has an unexpected tensor inventory"
                    )
                state = {}
                for source_name, relative_name in source_to_relative.items():
                    value = source.get_tensor(source_name)
                    shape = expected[relative_name]
                    if value.dtype != torch.bfloat16 or list(value.shape) != shape:
                        raise ControlError(
                            f"BF16 tensor dtype/shape mismatch: {source_name}"
                        )
                    state[relative_name] = value
                block = block_type(
                    hidden_size=DIM,
                    heads_num=HEADS,
                    mlp_width_ratio=4.0,
                    mlp_act_type="gelu_tanh",
                    attn_mode="magi_mp_causal",
                    qk_norm=True,
                    qk_norm_type="rms",
                    qkv_bias=True,
                    dtype=torch.bfloat16,
                    device=torch.device("meta"),
                )
                if set(block.state_dict()) != set(state):
                    raise ControlError(
                        f"pinned block {block_index} module inventory mismatch"
                    )
                block.load_state_dict(state, strict=True, assign=True)
                block.to(device=device, dtype=torch.bfloat16).eval()
                del state
        except ControlError:
            raise
        except Exception as error:
            raise ControlError(
                f"could not load pinned BF16 block {block_index}: {error}"
            ) from None
        if _file_identity(path) != admitted["cache_identities"][str(path)]:
            raise ControlError(f"BF16 block changed while loading: {path.name}")
        blocks.append(block)
    if len(blocks) != 54:
        raise ControlError("all 54 BF16 DiT blocks must remain resident")
    return blocks


def _new_reusable_f32_block(
    torch: Any,
    upstream: Path,
    helpers: tuple[Any, ...],
    attention_adapter: Any,
    device: Any,
) -> Any:
    block_type = block_reference._extract_pinned_block(
        upstream, torch, attention_adapter, helpers[-1].apply_rotary_emb, helpers
    )
    try:
        block = block_type(
            hidden_size=DIM,
            heads_num=HEADS,
            mlp_width_ratio=4.0,
            mlp_act_type="gelu_tanh",
            attn_mode="magi_mp_causal",
            qk_norm=True,
            qk_norm_type="rms",
            qkv_bias=True,
            dtype=torch.float32,
            device=torch.device("meta"),
        )
        block.to_empty(device=device)
        block.eval()
    except Exception as error:
        raise ControlError(
            f"could not allocate reusable pinned F32 block on MPS: {error}"
        ) from None
    return block


def _prepare_f32_block_copy_plan(
    torch: Any, source_blocks: list[Any], target_block: Any
) -> tuple[list[list[tuple[str, Any, Any]]], dict[str, int]]:
    """Bind index-ordered MPS BF16 sources to stable MPS F32 parameter storage."""
    if len(source_blocks) != 54:
        raise ControlError("F32 block reuse requires all 54 BF16 source blocks")
    target_state = target_block.state_dict()
    expected_names = set(cache_reference._BLOCK_RELATIVE_SHAPES)
    if set(target_state) != expected_names or len(target_state) != 32:
        raise ControlError("reusable F32 block parameter inventory mismatch")

    storage_pointers: dict[str, int] = {}
    for name, target in target_state.items():
        if target.dtype is not torch.float32 or target.device.type != "mps":
            raise ControlError(f"reusable F32 block tensor has wrong storage: {name}")
        pointer = int(target.data_ptr())
        if pointer == 0:
            raise ControlError(f"reusable F32 block tensor has no MPS storage: {name}")
        storage_pointers[name] = pointer

    plans: list[list[tuple[str, Any, Any]]] = []
    for block_index, source_block in enumerate(source_blocks):
        source_state = source_block.state_dict()
        if set(source_state) != expected_names or len(source_state) != 32:
            raise ControlError(
                f"BF16 source block {block_index} parameter inventory mismatch"
            )
        plan: list[tuple[str, Any, Any]] = []
        for name in sorted(expected_names):
            source = source_state[name]
            target = target_state[name]
            if (
                source.dtype is not torch.bfloat16
                or source.device.type != "mps"
                or target.dtype is not torch.float32
                or target.device.type != "mps"
                or tuple(source.shape) != tuple(target.shape)
            ):
                raise ControlError(
                    f"BF16-to-F32 MPS copy contract mismatch: block {block_index} {name}"
                )
            plan.append((name, target, source))
        plans.append(plan)
    return plans, storage_pointers


def _copy_source_block_into_f32(
    torch: Any,
    plan: list[tuple[str, Any, Any]],
    storage_pointers: dict[str, int],
    block_index: int,
) -> None:
    # Both endpoints are already MPS-resident. copy_ converts exact BF16 source
    # values into the reusable F32 parameter storage without host staging.
    try:
        for _name, target, source in plan:
            target.copy_(source)
    except Exception as error:
        raise ControlError(
            f"MPS BF16-to-F32 copy failed for source block {block_index}: {error}"
        ) from None
    for name, target, _source in plan:
        if target.dtype is not torch.float32:
            raise ControlError(f"F32 block tensor dtype changed during copy: {name}")
        if int(target.data_ptr()) != storage_pointers[name]:
            raise ControlError(f"F32 block tensor storage moved during copy: {name}")


def _load_segment_tensor(
    torch: Any,
    np: Any,
    payload_path: Path,
    record: dict[str, Any],
    *,
    segment_axis: int | None,
    device: Any,
    dtype: Any,
) -> Any:
    shape = tuple(record["shape"])
    raw = np.memmap(
        payload_path, dtype="<f4", mode="r", offset=record["offset"], shape=shape
    )
    if segment_axis is None:
        value = np.array(raw, copy=True)
    else:
        selector = [slice(None)] * len(shape)
        selector[segment_axis] = slice(
            SEGMENT_INDEX * SEGMENT_LATENTS, (SEGMENT_INDEX + 1) * SEGMENT_LATENTS
        )
        value = np.array(raw[tuple(selector)], copy=True)
    del raw
    return torch.from_numpy(value).to(device=device, dtype=dtype)


def _load_rope_segment(
    torch: Any,
    np: Any,
    conditioning_payload: Path,
    key: str,
    record: dict[str, Any],
    device: Any,
) -> Any:
    raw = np.memmap(
        conditioning_payload,
        dtype="<f4",
        mode="r",
        offset=record["offset"],
        shape=tuple(record["shape"]),
    )
    value = np.array(raw[SEGMENT_INDEX], copy=True)
    del raw
    tensor = torch.from_numpy(value).to(device=device, dtype=torch.float32)
    if tuple(tensor.shape) != (VIDEO_TOKENS, HEAD_DIM):
        raise ControlError(f"first-segment RoPE {key} shape mismatch")
    return tensor


def _load_qwen(torch: Any, np: Any, admitted: dict[str, Any], device: Any) -> Any:
    record = admitted["base_qwen"]["qwen_record"]
    payload = Path(admitted["base_qwen"]["qwen_payload"])
    raw = np.memmap(
        payload,
        dtype="<u2",
        mode="r",
        offset=record["offset"],
        shape=tuple(record["shape"]),
    )
    cpu = np.array(raw, copy=True)
    del raw
    value = torch.from_numpy(cpu).view(torch.bfloat16).to(device=device)
    if tuple(value.shape) != (1, TEXT_RMS_TOKENS, 3584):
        raise ControlError("authenticated Qwen state is not the full 1280 x 3584 crop")
    return value


def _import_euler_solver(upstream: Path) -> Any:
    import importlib.util

    path = upstream / EULER_RELATIVE
    spec = importlib.util.spec_from_file_location("hy_nomas_pinned_quality_euler", path)
    if spec is None or spec.loader is None:
        raise ControlError("could not import the pinned EulerSolver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        sys.modules.pop(spec.name, None)
        raise ControlError(
            f"pinned EulerSolver dependency unavailable: {error}"
        ) from None
    except Exception as error:
        sys.modules.pop(spec.name, None)
        raise ControlError(
            f"pinned EulerSolver import failed: {type(error).__name__}: {error}"
        ) from None
    return module.EulerSolver


def _mode_dtype(torch: Any, mode: str) -> Any:
    if mode == "source-bf16":
        return torch.bfloat16
    if mode == "native-f32":
        return torch.float32
    raise ControlError(f"unsupported arithmetic mode: {mode}")


def _prepare_cpu_scheduler(upstream: Path, mode: str) -> tuple[Any, Any, str]:
    try:
        import torch
        import diffusers
    except ImportError as error:
        raise ControlError(
            f"CPU Euler scheduler dependency unavailable: {error}"
        ) from None
    EulerSolver = _import_euler_solver(upstream)
    try:
        inspect.signature(EulerSolver).bind(
            num_train_timesteps=1000,
            shift=FLOW_SHIFT,
            reverse=True,
            solver="euler",
        )
        scheduler = EulerSolver(
            num_train_timesteps=1000,
            shift=FLOW_SHIFT,
            reverse=True,
            solver="euler",
        )
        inspect.signature(scheduler.set_timesteps).bind(
            STEPS, device=torch.device("cpu")
        )
        scheduler.set_timesteps(STEPS, device=torch.device("cpu"))
    except ControlError:
        raise
    except Exception as error:
        raise ControlError(
            "pinned EulerSolver CPU schedule construction failed: "
            f"{type(error).__name__}: {error}"
        ) from None
    tensors = (scheduler.timesteps, scheduler.sigmas)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ControlError("pinned EulerSolver schedule escaped CPU")
    if tuple(scheduler.timesteps.shape) != (STEPS,) or tuple(
        scheduler.sigmas.shape
    ) != (STEPS + 1,):
        raise ControlError("pinned EulerSolver returned an unexpected schedule extent")
    _validate_pinned_call_contracts(torch, upstream, mode)
    version = getattr(diffusers, "__version__", None)
    if not isinstance(version, str) or not version:
        raise ControlError("diffusers package did not report its version")
    return torch, scheduler, version


def _bind_refiner_attention_adapter(
    upstream: Path, adapter: Any, state: dict[str, Any], module_instance: Any = None
) -> Any:
    """Rebind the pinned refiner's imported attention symbol to this run's adapter."""
    import importlib

    module_name = "hymm.models.text_encoder.token_refiner"
    block_reference._namespace(
        "hymm.models.text_encoder", upstream / "hymm/models/text_encoder"
    )
    module = importlib.import_module(module_name)
    # token_refiner imports `attention` by value. Replacing sys.modules for the
    # attention module later does not update that already-imported global.
    module.attention = adapter
    block_type = module.IndividualTokenRefinerBlock
    forward = block_type.forward
    if (
        forward.__globals__ is not module.__dict__
        or forward.__globals__.get("attention") is not adapter
    ):
        raise ControlError(
            "pinned refiner forward is not bound to this run's attention adapter"
        )
    closure = dict(zip(adapter.__code__.co_freevars, adapter.__closure__ or ()))
    if "state" not in closure or closure["state"].cell_contents is not state:
        raise ControlError("pinned refiner adapter is not bound to this run's state")

    if module_instance is not None:
        if type(module_instance) is not module.SingleTokenRefinerPerBlock:
            raise ControlError("loaded pinned refiner instance has an unexpected class")
        blocks = module_instance.individual_token_refiner.blocks
        if len(blocks) != 2 or any(type(block) is not block_type for block in blocks):
            raise ControlError("loaded pinned refiner block inventory changed")
        if any(
            block.forward.__func__.__globals__.get("attention") is not adapter
            for block in blocks
        ):
            raise ControlError(
                "loaded pinned refiner blocks are not bound to this run's adapter"
            )
    return module


def _validate_pinned_call_contracts(torch: Any, upstream: Path, mode: str) -> None:
    """Bind every direct model call against the actual pinned forward signatures."""
    try:
        dtype = _mode_dtype(torch, mode)
        adapter_mode = "bf16" if mode == "source-bf16" else "f32"
        helpers = block_reference._install_upstream_modules(upstream)
        attention_state: dict[str, Any] = {
            "mode": adapter_mode,
            "compare": _NoopCompare(),
            "receipt": {"attention_calls": 0, "attention_query_chunks": 0},
        }
        adapter = attention_reference._build_attention_adapter(torch, attention_state)
        block_type = block_reference._extract_pinned_block(
            upstream, torch, adapter, helpers[-1].apply_rotary_emb, helpers
        )
        block = block_type(
            hidden_size=DIM,
            heads_num=HEADS,
            mlp_width_ratio=4.0,
            mlp_act_type="gelu_tanh",
            attn_mode="magi_mp_causal",
            qk_norm=True,
            qk_norm_type="rms",
            qkv_bias=True,
            dtype=dtype,
            device=torch.device("meta"),
        )
        refiner_state: dict[str, Any] = {
            "mode": adapter_mode,
            "valid_text_tokens": VALID_TEXT_TOKENS,
            "attention_block": 0,
            "attention_query_chunks": 0,
            "compare": _NoopCompare(),
        }
        _, refiner_attention = attention_reference._install_refiner_attention(
            torch, refiner_state
        )
        _bind_refiner_attention_adapter(upstream, refiner_attention, refiner_state)
        modules = attention_reference._build_prelude_modules(
            torch, upstream, helpers, dtype
        )
        _bind_refiner_attention_adapter(
            upstream, refiner_attention, refiner_state, modules["txt_in"]
        )
        block_kwargs = {
            "img": None,
            "txt": None,
            "vec": None,
            "vec_txt": None,
            "freqs_cis": None,
            "text_mask": None,
            "attn_param": {
                "num_prompts": 1,
                "txt_tokens_per_block": TEXT_TOKENS,
                "prompt_video_boundaries_tokens": [0, VIDEO_TOKENS],
                "prompt_video_boundaries": [0, VIDEO_TOKENS],
                "chunk_size_tokens": 0,
                "seg_ar_attention_backend": "magi",
            },
            "is_flash": False,
            "txt_tokens_per_block": TEXT_TOKENS,
            "prompt_video_boundaries": [0, VIDEO_TOKENS],
        }
        calls = (
            ("PatchEmbed.forward", modules["img_in"].forward, {"x": None}),
            ("TimestepEmbedder.forward", modules["time_in"].forward, {"t": None}),
            (
                "SingleTokenRefinerPerBlock.forward",
                modules["txt_in"].forward,
                {"x": None, "t": None, "mask": None},
            ),
            (
                "FinalLayerPerBlock.forward",
                modules["final_layer"].forward,
                {"x": None, "c": None},
            ),
            ("MMDoubleStreamBlock.forward", block.forward, block_kwargs),
        )
        for label, method, kwargs in calls:
            try:
                inspect.signature(method).bind(**kwargs)
            except TypeError as error:
                raise ControlError(
                    f"pinned call contract rejected {label}: {error}"
                ) from None
        try:
            # These adapter invocations mirror the calls made inside the
            # pinned MMDoubleStreamBlock and SingleTokenRefinerPerBlock. Bind
            # them here so a signature drift is rejected before loading GPU
            # weights or starting the four-step control.
            inspect.signature(adapter).bind(
                (None, None),
                (None, None),
                (None, None),
                img_q_len=VIDEO_TOKENS,
                img_kv_len=VIDEO_TOKENS,
                text_mask=None,
                attn_mode="magi_mp_causal",
                attn_param=block_kwargs["attn_param"],
            )
            inspect.signature(refiner_attention).bind(None, None, None, attn_mask=None)
        except TypeError as error:
            raise ControlError(
                f"pinned attention adapter call contract rejected: {error}"
            ) from None
    except ControlError:
        raise
    except Exception as error:
        raise ControlError(
            "pinned module call-contract preflight failed: "
            f"{type(error).__name__}: {error}"
        ) from None


def _schedule_record(scheduler: Any) -> dict[str, list[float]]:
    dt = scheduler.sigmas[1:] - scheduler.sigmas[:-1]
    return {
        "timesteps_f32": [float(value) for value in scheduler.timesteps.tolist()],
        "sigmas_f32": [float(value) for value in scheduler.sigmas.tolist()],
        "dt_f32": [float(value) for value in dt.tolist()],
    }


def _configure_mps_runtime() -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in ("", "0"):
        raise ControlError("MPS fallback must be disabled")
    if os.environ.get("HY_TEXT_MLP_FC2_FIXED_M", "0") not in ("", "0"):
        raise ControlError("the fixed-M text MLP path must be disabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HY_TEXT_MLP_FC2_FIXED_M"] = "0"


def _run_control(
    admitted: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
    scheduler: Any,
    diffusers_version: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        import numpy as np
    except ImportError as error:
        raise ControlError(f"MPS NumPy dependency unavailable: {error}") from None
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ControlError("a PyTorch MPS device is required")
    if not hasattr(torch, "bfloat16"):
        raise ControlError("this PyTorch build has no BF16 dtype")

    device = torch.device("mps")
    dtype = _mode_dtype(torch, args.mode)
    native_f32 = args.mode == "native-f32"
    adapter_mode = "f32" if native_f32 else "bf16"
    memory_before = {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory())
        if hasattr(torch.mps, "current_allocated_memory")
        else None,
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory())
        if hasattr(torch.mps, "driver_allocated_memory")
        else None,
        "planned_cap_gib": args.planned_cap_gib,
        "planned_cap_bytes": int(args.planned_cap_gib * (1024**3)),
        "cap_is_os_enforced": False,
    }

    helpers = block_reference._install_upstream_modules(Path(admitted["upstream_path"]))
    attention_state: dict[str, Any] = {
        "mode": adapter_mode,
        "compare": _NoopCompare(),
        "receipt": {"attention_calls": 0, "attention_query_chunks": 0},
    }
    attention_adapter = attention_reference._build_attention_adapter(
        torch, attention_state
    )
    refiner_state: dict[str, Any] = {
        "mode": adapter_mode,
        "valid_text_tokens": VALID_TEXT_TOKENS,
        "attention_block": 0,
        "attention_query_chunks": 0,
        "compare": _NoopCompare(),
    }
    _, refiner_attention = attention_reference._install_refiner_attention(
        torch, refiner_state
    )
    _bind_refiner_attention_adapter(
        Path(admitted["upstream_path"]), refiner_attention, refiner_state
    )
    model_globals = attention_reference._load_prelude_weights(
        torch,
        Path(admitted["global_shard"]["path"]),
        Path(admitted["upstream_path"]),
        helpers,
        device,
        dtype,
    )
    _bind_refiner_attention_adapter(
        Path(admitted["upstream_path"]),
        refiner_attention,
        refiner_state,
        model_globals["txt_in"],
    )
    blocks = _load_all_blocks(torch, admitted, helpers, attention_adapter, device)
    f32_block = None
    f32_copy_plans = None
    f32_storage_pointers = None
    if native_f32:
        f32_block = _new_reusable_f32_block(
            torch,
            Path(admitted["upstream_path"]),
            helpers,
            attention_adapter,
            device,
        )
        f32_copy_plans, f32_storage_pointers = _prepare_f32_block_copy_plan(
            torch, blocks, f32_block
        )

    payload_path = Path(admitted["conditioning_payload"])
    latent_record = admitted["latent_record"]
    latent = _load_segment_tensor(
        torch,
        np,
        payload_path,
        latent_record,
        segment_axis=2,
        device=device,
        dtype=dtype,
    )
    if tuple(latent.shape) != (1, 32, SEGMENT_LATENTS, 9, 16):
        raise ControlError("first Echo segment latent shape mismatch")
    if native_f32:
        # Native Metal rounds the initial F32 noise to BF16, then expands it
        # back to F32 for the pinned F32 model body.
        latent = latent.to(torch.bfloat16).to(torch.float32)
    qwen = _load_qwen(torch, np, admitted, device)
    if native_f32:
        qwen = qwen.to(dtype=torch.float32)
    full_mask = torch.tensor(
        [admitted["base_qwen"]["valid_text_mask"]],
        dtype=torch.bool,
        device=device,
    )
    if tuple(full_mask.shape) != (1, TEXT_RMS_TOKENS):
        raise ControlError("Qwen valid mask must retain all 1280 rows")
    rope_cos = _load_rope_segment(
        torch,
        np,
        payload_path,
        "cos",
        admitted["rope_cos_record"],
        device,
    )
    rope_sin = _load_rope_segment(
        torch,
        np,
        payload_path,
        "sin",
        admitted["rope_sin_record"],
        device,
    )

    # Keep scheduler construction and sigma differences on CPU exactly as the
    # source does; transfer the small fixed schedule once. Calling scheduler.step
    # would read a device timestep back for its Python step-index search.
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    sigmas_cpu = scheduler.sigmas
    dts = (sigmas_cpu[1:] - sigmas_cpu[:-1]).to(device=device, dtype=torch.float32)

    attn_param = {
        "num_prompts": 1,
        "txt_tokens_per_block": TEXT_TOKENS,
        "prompt_video_boundaries_tokens": [0, VIDEO_TOKENS],
        "prompt_video_boundaries": [0, VIDEO_TOKENS],
        "chunk_size_tokens": 0,
        "seg_ar_attention_backend": "magi",
    }
    device = torch.device("mps")
    with torch.inference_mode():
        for step_index in range(STEPS):
            timestep = timesteps[step_index].reshape(1)
            vector = model_globals["time_in"](timestep)
            refiner_state["attention_block"] = 0
            text_full = model_globals["txt_in"](qwen, timestep, full_mask)
            if refiner_state["attention_block"] != 2:
                raise ControlError(
                    "pinned refiner did not execute both attention blocks"
                )
            if tuple(text_full.shape) != (1, TEXT_RMS_TOKENS, DIM):
                raise ControlError("full-row refiner output shape mismatch")
            text = text_full[:, :TEXT_TOKENS].contiguous()
            image = model_globals["img_in"](latent)
            if tuple(image.shape) != (1, VIDEO_TOKENS, DIM):
                raise ControlError("PatchEmbed output shape mismatch")

            for block_index, source_block in enumerate(blocks):
                if native_f32:
                    assert f32_block is not None
                    assert f32_copy_plans is not None
                    assert f32_storage_pointers is not None
                    _copy_source_block_into_f32(
                        torch,
                        f32_copy_plans[block_index],
                        f32_storage_pointers,
                        block_index,
                    )
                    block = f32_block
                else:
                    block = source_block
                image, text = block(
                    img=image,
                    txt=text,
                    vec=vector,
                    vec_txt=vector,
                    freqs_cis=(rope_cos, rope_sin),
                    text_mask=None,
                    attn_param=attn_param,
                    is_flash=False,
                    txt_tokens_per_block=TEXT_TOKENS,
                    prompt_video_boundaries=[0, VIDEO_TOKENS],
                )
            prediction_tokens = model_globals["final_layer"](image, vector)
            if tuple(prediction_tokens.shape) != (1, VIDEO_TOKENS, 32):
                raise ControlError("final head output shape mismatch")
            prediction = prediction_tokens.reshape(1, SEGMENT_LATENTS, 9, 16, 32)
            prediction = prediction.permute(0, 4, 1, 2, 3).contiguous()

            # Pinned EulerSolver.step equations, with its already-known index
            # made explicit to avoid its host-side .item() lookup each step.
            if native_f32:
                # Metal rounds the model velocity to BF16, expands it, performs
                # separate F32 multiply/add operations, then stores BF16 state
                # expanded back to F32 for the next model step.
                velocity = prediction.to(torch.bfloat16).to(torch.float32)
                delta = velocity * dts[step_index]
                updated = latent + delta
                latent = updated.to(torch.bfloat16).to(torch.float32)
                del velocity, delta, updated
            else:
                # source-bf16 mode: sample.float() + model_output.float() * dt,
                # followed by the pipeline BF16 latent boundary.
                latent = (latent.float() + prediction.float() * dts[step_index]).to(
                    dtype
                )
            del image, text, text_full, vector, prediction_tokens, prediction

    # One terminal MPS-to-host export. The output values are BF16 Euler state
    # rounded back to F32 for the small evidence payload.
    terminal = latent.detach().float().contiguous().reshape(-1).cpu().numpy()
    if (
        terminal.size != 1 * 32 * SEGMENT_LATENTS * 9 * 16
        or not np.isfinite(terminal).all()
    ):
        raise ControlError("terminal latent has invalid extent or nonfinite values")
    payload_bytes = np.asarray(terminal, dtype="<f4").tobytes(order="C")
    attention_calls = int(attention_state["receipt"]["attention_calls"])
    attention_chunks = int(attention_state["receipt"]["attention_query_chunks"])
    refiner_chunks = int(refiner_state["attention_query_chunks"])
    if (
        attention_calls != STEPS * 54
        or attention_chunks
        != STEPS * 54 * math.ceil((VIDEO_TOKENS + VALID_TEXT_TOKENS) / QUERY_CHUNK)
        or refiner_chunks != STEPS * 2 * math.ceil(VALID_TEXT_TOKENS / QUERY_CHUNK)
    ):
        raise ControlError(
            "attention call/chunk inventory differs from the admitted one-segment run"
        )
    summary = {
        "device": "mps",
        "mode": args.mode,
        "parameter_activation_dtype": "F32" if native_f32 else "BF16",
        "attention": "chunked PyTorch MPS SDPA; valid prefix 557; full refiner extent 1280",
        "steps": STEPS,
        "flow_shift": FLOW_SHIFT,
        "cfg_scale_video": 1.0,
        "block_invocations": len(blocks) * STEPS,
        "attention_calls": attention_calls,
        "attention_query_chunks": attention_chunks,
        "refiner_attention_calls": STEPS * 2,
        "refiner_query_chunks": refiner_chunks,
        "activation_readbacks_before_terminal_export": 0,
        "terminal_export_count": 1,
        "terminal_values": int(terminal.size),
        "terminal_dtype": (
            "F32 expansion of native BF16-rounded latent state"
            if native_f32
            else "F32 expansion of BF16 latent state"
        ),
        "reused_f32_block_invocations": STEPS * 54 if native_f32 else 0,
        "resident_bf16_source_block_count": len(blocks),
        "stable_f32_block_parameter_storage": bool(native_f32),
        "terminal_finite": True,
        "diffusers_version": diffusers_version,
        "schedule": _schedule_record(scheduler),
        "memory_before_weights": memory_before,
    }
    return summary, payload_bytes


def _manifest(
    admitted: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
    payload: bytes,
    payload_name: str,
) -> dict[str, Any]:
    native_f32 = args.mode == "native-f32"
    precision_boundaries = (
        {
            "latent_input": "F32 source values rounded to BF16 then expanded to F32",
            "patch_embed": "BF16 checkpoint weights expanded to F32; F32 activations",
            "text_refiner": "BF16 checkpoint weights expanded to F32; full 1280-row F32 path",
            "dit_blocks": "54 exact BF16 source blocks resident on MPS; one stable F32 block receives per-index MPS copy_ casts",
            "final_head": "BF16 checkpoint weights expanded to F32; F32 activations",
            "euler": "velocity rounded to BF16; separate F32 multiply/add; latent rounded to BF16 each step then expanded to F32",
        }
        if native_f32
        else {
            "latent_input": "F32 source values cast once to BF16",
            "patch_embed": "BF16 weights and BF16 activation boundary",
            "text_refiner": "BF16 weights and activation boundaries; full 1280 rows retained",
            "dit_blocks": "all 54 exact BF16 checkpoint shards resident on MPS; BF16 activations",
            "final_head": "BF16 weights and activation boundary",
            "euler": "pinned F32 sample/velocity update then BF16 state boundary per step",
        }
    )
    limitations = [
        "Chunked MPS SDPA replaces the pinned Magi dispatch; no CUDA or Magi parity is claimed.",
        "No cache history, second segment, camera/action input, classifier-free guidance branch, or complete video is run.",
        "The result is a matched initial-segment latent arithmetic control; it does not prove an eye-motion cause or fix.",
        "PixelUMM decoding is not run by this control.",
    ]
    if native_f32:
        limitations.insert(
            0,
            "native-f32 is an F32 activation arithmetic control using BF16 checkpoint weights and native BF16 Euler rounding; it is not a stock-F32 checkpoint or output claim.",
        )
    else:
        limitations.insert(
            0,
            "source-bf16 is a BF16 arithmetic control for one five-second latent segment, not full video generation.",
        )
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "scope": "one Echo segment: first 30 latent frames, four denoise steps, 54 DiT blocks",
        "result": {
            "payload": payload_name,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "dtype": "F32",
            "value_semantics": (
                "final native BF16-rounded latent values expanded exactly to F32"
                if native_f32
                else "final BF16 latent values expanded exactly to F32"
            ),
            "shape": [1, 32, SEGMENT_LATENTS, 9, 16],
        },
        "source": {
            "conditioning_manifest": admitted["conditioning_path"],
            "conditioning_manifest_sha256": CONDITIONING_SHA256,
            "conditioning_payload_sha256": admitted["conditioning"]["payload_sha256"],
            "base_qwen_manifest_sha256": admitted["source_info"][
                "base_qwen_manifest_sha256"
            ],
            "base_qwen_payload_sha256": admitted["source_info"][
                "base_qwen_payload_sha256"
            ],
            "qwen_tensor_sha256": admitted["base_qwen"]["qwen_tensor_sha256"],
            "first_segment_latent_bf16_roundtrip_sha256": admitted["source_info"][
                "first_segment_latent_bf16_roundtrip_sha256"
            ],
            "first_segment_rope_cos_sha256": admitted["first_rope_digests"]["cos"],
            "first_segment_rope_sin_sha256": admitted["first_rope_digests"]["sin"],
            "continuation_rope_cos_bank_sha256": admitted["rope_cos_record"]["sha256"],
            "continuation_rope_sin_bank_sha256": admitted["rope_sin_record"]["sha256"],
            "cache_index": admitted["cache_index_path"],
            "cache_index_sha256": CACHE_INDEX_SHA256,
            "global_weights": admitted["global_shard"],
            "block_shards": [
                {key: value for key, value in record.items() if key != "identity"}
                for record in admitted["cache_shards"]
            ],
            "upstream_revision": admitted["upstream_revision"],
            "upstream_source_sha256": admitted["upstream_source_sha256"],
            "prelude_source_sha256": admitted["prelude_source_sha256"],
            "continuation_upstream_source_sha256": admitted[
                "continuation_upstream_source_sha256"
            ],
            "euler_source_sha256": admitted["euler_source_sha256"],
            "total_resident_checkpoint_bytes": admitted["total_weight_bytes"],
        },
        "configuration": {
            "mode": args.mode,
            "segment_index": SEGMENT_INDEX,
            "segment_latents": SEGMENT_LATENTS,
            "latent_shape": [1, 32, SEGMENT_LATENTS, 9, 16],
            "video_tokens": VIDEO_TOKENS,
            "qwen_rows": TEXT_RMS_TOKENS,
            "qwen_valid_prefix_rows": VALID_TEXT_TOKENS,
            "dit_text_rows": TEXT_TOKENS,
            "rope_bank_index": 0,
            "steps": STEPS,
            "shift": FLOW_SHIFT,
            "cfg_scale_video": 1.0,
            "precision_boundaries": precision_boundaries,
            "attention_adapter": {
                "implementation": "PyTorch MPS scaled_dot_product_attention",
                "query_chunk": QUERY_CHUNK,
                "valid_text_prefix": VALID_TEXT_TOKENS,
                "magi_cuda_parity": False,
            },
            "planned_process_cap_gib": args.planned_cap_gib,
            "planned_process_cap_os_enforced_by_script": False,
        },
        "execution": summary,
        "limitations": limitations,
    }


def _publish(
    output: Path, document: dict[str, Any], payload: bytes
) -> tuple[Path, str]:
    output = output.expanduser()
    if not output.is_absolute():
        raise ControlError("--output must be an absolute manifest path")
    if output.suffix.lower() != ".json" or not output.parent.is_dir():
        raise ControlError("--output must be a .json file in an existing directory")
    if output.is_symlink():
        raise ControlError("--output manifest path must not be a symlink")
    payload_path = output.with_name(f"{output.stem}.latent.f32le")
    if os.path.lexists(output) or os.path.lexists(payload_path):
        raise ControlError("result manifest and latent payload must both be new files")
    document["result"]["payload"] = payload_path.name
    encoded = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    try:
        continuation_inputs._publish_noreplace(payload_path, payload)
        stat = payload_path.stat()
        payload_identity = (stat.st_dev, stat.st_ino)
        try:
            continuation_inputs._publish_noreplace(output, encoded)
        except Exception:
            current = payload_path.stat()
            if (current.st_dev, current.st_ino) == payload_identity:
                payload_path.unlink(missing_ok=True)
            raise
    except Exception as error:
        raise ControlError(
            f"could not publish fresh BF16 control artifacts: {error}"
        ) from None
    return payload_path, hashlib.sha256(encoded).hexdigest()


def _validate_output_target(output: Path) -> Path:
    output = output.expanduser()
    if not output.is_absolute():
        raise ControlError("--output must be an absolute manifest path")
    if output.suffix.lower() != ".json" or not output.parent.is_dir():
        raise ControlError("--output must be a .json file in an existing directory")
    payload_path = output.with_name(f"{output.stem}.latent.f32le")
    if output.is_symlink() or payload_path.is_symlink():
        raise ControlError("--output manifest and payload paths must not be symlinks")
    if os.path.lexists(output) or os.path.lexists(payload_path):
        raise ControlError("result manifest and latent payload must both be new files")
    return output


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=EXECUTION_MODES,
        default="source-bf16",
        help="arithmetic control (default: source-bf16)",
    )
    parser.add_argument(
        "--conditioning-manifest", type=Path, default=DEFAULT_CONDITIONING
    )
    parser.add_argument("--cache-index", type=Path, default=DEFAULT_CACHE_INDEX)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--planned-cap-gib", type=float, default=DEFAULT_PLANNED_CAP_GIB
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="new result JSON under an existing directory; required for MPS run",
    )
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.planned_cap_gib)
        or args.planned_cap_gib <= 0
        or args.planned_cap_gib > MAX_PLANNED_CAP_GIB
    ):
        parser.error(f"--planned-cap-gib must be in (0, {MAX_PLANNED_CAP_GIB}] GiB")
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only is selected")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if not args.validate_only:
            args.output = _validate_output_target(args.output)
            _configure_mps_runtime()
        admitted = _authenticate(args)
        torch, scheduler, diffusers_version = _prepare_cpu_scheduler(
            Path(admitted["upstream_path"]), args.mode
        )
        memory_plan = {
            "source_weight_bytes": admitted["total_weight_bytes"],
            "source_weight_gib": admitted["total_weight_bytes"] / (1024**3),
            "planned_cap_gib": args.planned_cap_gib,
            "host_memory_before_mps": _host_memory_snapshot(),
            "activation_estimate_gib": 1.5,
            "python_runtime_estimate_gib": 1.0,
            "estimate_note": "Weights are exact shard file bytes; activation/runtime estimates are planning values, not measured peak.",
        }
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "status": "validated_only",
                        "mode": args.mode,
                        "torch_imported": True,
                        "torch_version": str(torch.__version__),
                        "diffusers_version": diffusers_version,
                        "scheduler_device": "cpu",
                        "mps_device_used": False,
                        "model_launched": False,
                        "scheduler": _schedule_record(scheduler),
                        "authenticated_sources": admitted["source_info"],
                        "cache_shard_count": len(admitted["cache_shards"]),
                        "cache_weight_bytes": admitted["total_weight_bytes"],
                        "upstream_revision": admitted["upstream_revision"],
                        "memory_plan": memory_plan,
                        "scope": (
                            "one 30-latent-frame segment, four steps, full 54-block "
                            f"{args.mode} arithmetic control"
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0
        summary, payload = _run_control(
            admitted, args, torch, scheduler, diffusers_version
        )
        doc = _manifest(admitted, args, summary, payload, "pending")
        payload_path, manifest_sha = _publish(args.output, doc, payload)
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "completed",
                    "output": str(args.output),
                    "manifest_sha256": manifest_sha,
                    "payload": str(payload_path),
                    "payload_sha256": doc["result"]["payload_sha256"],
                    "payload_bytes": len(payload),
                    "block_invocations": summary["block_invocations"],
                    "terminal_export_count": summary["terminal_export_count"],
                    "scope": doc["scope"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (ControlError, OSError, ValueError, RuntimeError) as error:
        print(f"Hy-Nomas DiT arithmetic control rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
