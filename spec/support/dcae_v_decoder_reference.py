"""Guarded upstream Hy-Nomas DCAE-V decoder oracle.

Plan mode admits only the pinned upstream module, original BF16 checkpoint,
and one of two fixed latent geometries using CPU metadata and a meta-device
decoder schema. Execute mode materializes only the original decoder on MPS,
keeps the upstream ten-latent causal feature cache across tiles, and exports
the input, selected stage boundaries, and final pixels in one terminal device
transfer. This diagnostic is not part of native timing or memory results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SUPPORT = Path(__file__).resolve().parent
REPO_ROOT = SUPPORT.parents[1]
EXAMPLE_SUPPORT = REPO_ROOT / "examples" / "support"
for entry in (str(EXAMPLE_SUPPORT), str(SUPPORT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import dcae_v_reference as artifact  # noqa: E402
import hy_nomas_photo_dcae_decode_control as dcae  # noqa: E402


SCHEMA = "cogni_ml.dcae_v_decoder_upstream_reference/v1"
VARIANT = "dc-ae-v-1.0-f32t4c32-bf16"
SMOKE_LATENT_SHAPE = (1, 32, 20, 1, 1)
FULL_LATENT_SHAPE = (1, 32, 60, 9, 16)
TILE_LATENTS = 10
TILE_FRAMES = 40
SPATIAL_COMPRESSION = 32
TEMPORAL_COMPRESSION = 4
ATOL = RTOL = 2e-5
SMOKE_SEED = 17031
GIB = 1024**3
MIB = 1024**2
EXTERNAL_CAP_BYTES = 18 * GIB
PHYSICAL_RESERVE_FRACTION = 0.10
MAX_ARTIFACT_BYTES = 2 * GIB
MAX_MANIFEST_BYTES = 4 * MIB

DEFAULT_UPSTREAM = dcae.DEFAULT_UPSTREAM
DEFAULT_WEIGHTS = dcae.DEFAULT_WEIGHTS
DEFAULT_OUTPUT_DIR = (
    dcae.DEFAULT_ROOT / "dcae-v-decoder-upstream-reference-v1"
)


class ReferenceError(ValueError):
    """A source, input, runtime, or oracle artifact violates its contract."""


def validate_latent_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Admit the small two-tile cache probe or the exact paired final bank."""
    if type(shape) not in (tuple, list) or any(type(dimension) is not int for dimension in shape):
        raise ReferenceError("latent shape must contain integer dimensions")
    candidate = tuple(shape)
    if candidate not in (SMOKE_LATENT_SHAPE, FULL_LATENT_SHAPE):
        if len(candidate) == 5 and candidate[0:2] == (1, 32) and candidate[2] % TILE_LATENTS:
            raise ReferenceError("latent input must contain whole ten-latent tiles")
        raise ReferenceError(
            "latent shape must be the seeded smoke shape [1,32,20,1,1] "
            "or the paired-bank shape [1,32,60,9,16]"
        )
    return candidate


def output_shape(latent_shape: Sequence[int]) -> tuple[int, ...]:
    """Return the pinned decoder output geometry for an admitted latent shape."""
    batch, _channels, frames, height, width = validate_latent_shape(latent_shape)
    return (
        batch,
        3,
        frames * TEMPORAL_COMPRESSION,
        height * SPATIAL_COMPRESSION,
        width * SPATIAL_COMPRESSION,
    )


def capture_policy(latent_shape: Sequence[int]) -> str:
    """Keep all scale boundaries for smoke and output-only for the full bank."""
    shape = validate_latent_shape(latent_shape)
    return "stage_trace" if shape == SMOKE_LATENT_SHAPE else "output_only"


def trace_tensor_names(tile_indices: Iterable[int]) -> tuple[str, ...]:
    """Name stage-boundary tensors in upstream decoder execution order."""
    boundaries = (
        "project_in",
        "stages.6",
        "stages.5",
        "stages.4",
        "stages.3",
        "stages.2",
        "stages.1",
        "project_out",
    )
    names: list[str] = []
    seen: set[int] = set()
    for tile in tile_indices:
        if type(tile) is not int or tile < 0 or tile in seen:
            raise ReferenceError("trace tile indices must be unique nonnegative integers")
        seen.add(tile)
        names.extend(f"trace.{boundary}.tile{tile}" for boundary in boundaries)
    return tuple(names)


def parse_trace_tiles(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of trace tile indices without duplicates."""
    try:
        result = tuple(int(part, 10) for part in value.split(","))
    except ValueError:
        raise ReferenceError("trace tiles must be comma-separated nonnegative integers") from None
    if not result or any(part < 0 for part in result) or len(set(result)) != len(result):
        raise ReferenceError("trace tiles must be a nonempty list of unique nonnegative integers")
    return result


def make_smoke_latents(torch: Any, *, seed: int = SMOKE_SEED) -> Any:
    """Create a deterministic BF16 latent whose two tiles and frames differ."""
    if type(seed) is not int or seed < 0:
        raise ReferenceError("smoke seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(SMOKE_LATENT_SHAPE, generator=generator, dtype=torch.float32)
    return values.to(dtype=torch.bfloat16).contiguous()


def _tensor_bytes_sha256(torch: Any, tensor: Any) -> str:
    if tensor.device.type != "cpu" or tensor.dtype is not torch.bfloat16:
        raise ReferenceError("input digest requires a CPU BF16 tensor")
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def decoder_cache_bytes(latent_shape: Sequence[int]) -> int:
    """Count one BF16 last-frame cache for every chunk-causal decoder conv."""
    batch, _channels, _frames, latent_h, latent_w = validate_latent_shape(latent_shape)
    # (input channels, spatial multiplier, conv count). Each conv stores one
    # preceding temporal frame at its input geometry. The list follows the
    # pinned project-in, stages 6..1, and project-out graph.
    cache_specs = (
        (32, 1, 1),
        (1024, 1, 8),   # stage 6 residual convs
        (1024, 1, 9),   # stage 5 upsample + residual convs
        (1024, 1, 1),   # stage 4 upsample
        (1024, 2, 8),   # stage 4 residual convs
        (1024, 2, 1),   # stage 3 upsample
        (512, 4, 8),    # stage 3 residual convs
        (512, 4, 1),    # stage 2 upsample
        (512, 8, 20),   # stage 2 residual convs
        (512, 8, 1),    # stage 1 upsample
        (256, 16, 10),  # stage 1 residual convs
        (256, 16, 1),   # project out
    )
    values = sum(
        batch * channels * (latent_h * scale) * (latent_w * scale) * count
        for channels, scale, count in cache_specs
    )
    return values * 2


def memory_estimate(latent_shape: Sequence[int]) -> dict[str, Any]:
    """Report named BF16 buffers; this is not an opaque MPS peak bound."""
    shape = validate_latent_shape(latent_shape)
    output = output_shape(shape)
    batch, _channels, _frames, height, width = shape
    full_stage1 = batch * 256 * TILE_FRAMES * (height * 16) * (width * 16) * 2
    return {
        "decoder_bf16_weight_bytes": dcae.DECODER_BF16_BYTES,
        "input_bf16_bytes": math.prod(shape) * 2,
        "persistent_feature_cache_bf16_bytes": decoder_cache_bytes(shape),
        "full_stage1_tile_boundary_bf16_bytes": full_stage1,
        "output_bf16_bytes": math.prod(output) * 2,
        "external_cap_bytes": EXTERNAL_CAP_BYTES,
        "physical_memory_reserve_fraction": PHYSICAL_RESERVE_FRACTION,
        "external_guard_required": True,
        "opaque_mps_workspace_included": False,
    }


def _read_f32le_latents(
    torch: Any, path: Path, expected_sha256: str, shape: Sequence[int]
) -> tuple[Any, dict[str, Any]]:
    """Read an explicitly pinned little-endian F32 BCTHW input on the CPU."""
    import numpy as np

    latent_shape = validate_latent_shape(shape)
    if latent_shape != FULL_LATENT_SHAPE:
        raise ReferenceError("external F32 latents must use the paired-bank shape [1,32,60,9,16]")
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ReferenceError("latent SHA-256 must be 64 lowercase hexadecimal characters")
    path = Path(path).expanduser()
    try:
        identity = artifact._identity(path)
    except artifact.ReferenceError as exc:
        raise ReferenceError(str(exc)) from None
    expected_bytes = math.prod(latent_shape) * 4
    if identity[2] != expected_bytes:
        raise ReferenceError(f"latent input byte extent mismatch: {identity[2]} != {expected_bytes}")
    observed_sha256 = artifact.sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ReferenceError("latent input SHA-256 differs from caller pin")
    raw = path.read_bytes()
    if artifact._identity(path) != identity or hashlib.sha256(raw).hexdigest() != observed_sha256:
        raise ReferenceError("latent input changed while it was read")
    values = np.frombuffer(raw, dtype="<f4").copy()
    if not np.isfinite(values).all():
        raise ReferenceError("latent input contains non-finite F32 values")
    latent = torch.from_numpy(values.reshape(latent_shape)).to(dtype=torch.bfloat16).contiguous()
    return latent, {
        "kind": "caller_pinned_f32le",
        "path": str(path.resolve(strict=True)),
        "source_dtype": "F32LE",
        "source_bytes": expected_bytes,
        "source_sha256": observed_sha256,
        "shape": list(latent_shape),
        "decoder_dtype": "BF16",
        "decoder_bf16_sha256": _tensor_bytes_sha256(torch, latent),
        "conversion": "F32LE to BF16 on CPU before one H2D input transfer",
    }


def _resolve_output_dir(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ReferenceError("output-dir must be an absolute path")
    if path.exists() or path.is_symlink():
        raise ReferenceError("output-dir must name a fresh path")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError:
        raise ReferenceError("output-dir parent must already exist") from None
    if not parent.is_dir() or not path.name or path.name in {".", ".."}:
        raise ReferenceError("output-dir parent must be an existing directory")
    resolved = parent / path.name
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise ReferenceError("oracle artifacts must be written outside the repository")
    if resolved.exists() or resolved.is_symlink():
        raise ReferenceError("output-dir appeared during admission")
    return resolved


def _load_input(args: argparse.Namespace, torch: Any) -> tuple[Any, dict[str, Any], tuple[int, ...]]:
    if args.latent_f32_path is None:
        if args.latent_sha256 is not None or args.latent_shape is not None:
            raise ReferenceError("latent path, SHA-256, and shape must be supplied together")
        latent = make_smoke_latents(torch, seed=args.seed)
        shape = validate_latent_shape(tuple(latent.shape))
        source = {
            "kind": "seeded_bf16_smoke",
            "seed": args.seed,
            "shape": list(shape),
            "dtype": "BF16",
            "bf16_sha256": _tensor_bytes_sha256(torch, latent),
            "generator": "torch.Generator(device=cpu), torch.randn(float32), cast to BF16",
        }
        return latent, source, shape
    if args.latent_sha256 is None or args.latent_shape is None:
        raise ReferenceError("latent path, SHA-256, and shape must be supplied together")
    shape = validate_latent_shape(tuple(args.latent_shape))
    latent, source = _read_f32le_latents(torch, args.latent_f32_path, args.latent_sha256, shape)
    return latent, source, shape


def _selected_trace_tiles(
    shape: Sequence[int], requested: Sequence[int] | None
) -> tuple[int, ...]:
    policy = capture_policy(shape)
    tile_count = shape[2] // TILE_LATENTS
    if policy == "output_only":
        if requested is not None:
            raise ReferenceError("the full paired-bank fixture is output-only to bound trace memory")
        return ()
    selected = tuple(range(tile_count)) if requested is None else tuple(requested)
    if not selected or any(type(tile) is not int or tile < 0 or tile >= tile_count for tile in selected):
        raise ReferenceError("trace tile index is outside the admitted latent input")
    if len(set(selected)) != len(selected):
        raise ReferenceError("trace tile indices must be unique")
    return selected


def preflight(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    """Authenticate pinned CPU inputs and derive the upstream decoder schema."""
    input_cpu, input_source, latent_shape = _load_input(args, torch)
    selected_tiles = _selected_trace_tiles(latent_shape, args.trace_tiles)
    output_dir = _resolve_output_dir(args.output_dir)

    upstream_root = Path(args.upstream).expanduser().resolve(strict=True)
    if dcae._git_head(upstream_root) != dcae.UPSTREAM_REVISION:
        raise ReferenceError("upstream checkout revision differs from the pinned Hy-Nomas source")
    module_path = dcae._regular_path(
        upstream_root / "hymm/models/vae/autoencoder_dcae_v.py", "upstream DCAE module"
    )
    module_identity = artifact._identity(module_path)
    module_sha256 = artifact.sha256_file(module_path)
    if module_sha256 != dcae.DCAE_MODULE_SHA256:
        raise ReferenceError("upstream DCAE module SHA-256 differs from the pin")

    weights_path = dcae._regular_path(args.weights, "original BF16 DCAE checkpoint")
    weights_identity = artifact._identity(weights_path)
    if weights_identity[2] != dcae.DCAE_CHECKPOINT_BYTES:
        raise ReferenceError("original BF16 DCAE checkpoint byte extent differs from the pin")
    weights_sha256 = artifact.sha256_file(weights_path)
    if weights_sha256 != dcae.DCAE_CHECKPOINT_SHA256:
        raise ReferenceError("original BF16 DCAE checkpoint SHA-256 differs from the pin")

    try:
        decoder_state, schema = dcae._read_checkpoint_metadata(weights_path, module_path, torch)
    except (dcae.ControlError, ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        raise ReferenceError(f"original decoder schema admission failed: {exc}") from None
    config = schema["config"]
    if (
        config.latent_channels != 32
        or config.decode_temporal_tile_size != TILE_FRAMES
        or config.decode_temporal_tile_latent_size != TILE_LATENTS
        or config.decode_temporal_tile_overlap_factor != 0
        or config.use_feature_cache is not True
        or config.num_pad_frames != 0
    ):
        raise ReferenceError("upstream decoder tiling/cache configuration differs from the oracle contract")
    if tuple(config.decoder.width_list) != (128, 256, 512, 512, 1024, 1024, 1024):
        raise ReferenceError("upstream decoder width schedule differs from the pinned variant")
    if tuple(config.decoder.depth_list) != (0, 5, 10, 4, 4, 4, 4):
        raise ReferenceError("upstream decoder depth schedule differs from the pinned variant")
    if artifact._identity(module_path) != module_identity:
        raise ReferenceError("upstream DCAE module changed during schema admission")
    if artifact._identity(weights_path) != weights_identity:
        raise ReferenceError("original DCAE checkpoint changed during schema admission")

    return {
        "input_cpu": input_cpu,
        "input_source": input_source,
        "latent_shape": latent_shape,
        "selected_trace_tiles": selected_tiles,
        "output_dir": output_dir,
        "upstream_root": upstream_root,
        "module_path": module_path,
        "module_sha256": module_sha256,
        "module_identity": module_identity,
        "weights_path": weights_path,
        "weights_sha256": weights_sha256,
        "weights_identity": weights_identity,
        "decoder_state": decoder_state,
        "schema": schema,
        "torch_version": torch.__version__,
        "memory_estimate": memory_estimate(latent_shape),
    }


def _trace_modules(decoder: Any) -> tuple[tuple[str, Any], ...]:
    """Select project boundaries and nonempty stages 6 through 1."""
    if len(decoder.stages) != 7:
        raise ReferenceError("upstream decoder stage count differs from the pinned seven-stage graph")
    modules = [("project_in", decoder.project_in)]
    modules.extend((f"stages.{stage}", decoder.stages[stage]) for stage in range(6, 0, -1))
    modules.append(("project_out", decoder.project_out))
    return tuple(modules)


def _pack_terminal(torch: Any, tensors: Mapping[str, Any]) -> tuple[bytes, list[dict[str, Any]]]:
    """Pack all MPS BF16 tensors into one byte tensor and make one CPU transfer."""
    if not tensors:
        raise ReferenceError("terminal transfer requires at least one tensor")
    pieces = []
    inventory: list[dict[str, Any]] = []
    offset = 0
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or name in {item["name"] for item in inventory}:
            raise ReferenceError("terminal tensor names must be unique nonempty strings")
        if tensor.device.type != "mps" or tensor.dtype is not torch.bfloat16:
            raise ReferenceError(f"terminal tensor is not MPS BF16: {name}")
        contiguous = tensor.detach().contiguous()
        nbytes = contiguous.numel() * 2
        pieces.append(contiguous.view(torch.uint8).reshape(-1))
        inventory.append({
            "name": name,
            "shape": [int(dimension) for dimension in contiguous.shape],
            "dtype": "BF16",
            "nbytes": nbytes,
            "offset": offset,
        })
        offset += nbytes
    packed = torch.cat(pieces, dim=0).contiguous()
    raw = packed.cpu().numpy().tobytes()
    if len(raw) != offset:
        raise ReferenceError("terminal pack byte extent differs from its tensor inventory")
    return raw, inventory


def _unpack_terminal(torch: Any, raw: bytes, inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconstruct CPU BF16 safetensors payloads after the single terminal readback."""
    import numpy as np

    result: dict[str, Any] = {}
    cursor = 0
    for item in inventory:
        name = item["name"]
        shape = tuple(item["shape"])
        nbytes = int(item["nbytes"])
        offset = int(item["offset"])
        if offset != cursor or nbytes != math.prod(shape) * 2 or cursor + nbytes > len(raw):
            raise ReferenceError(f"invalid terminal tensor range: {name}")
        words = np.frombuffer(raw, dtype="<u2", count=nbytes // 2, offset=cursor).copy()
        tensor = torch.from_numpy(words).view(torch.bfloat16).reshape(shape).contiguous()
        result[name] = tensor
        cursor += nbytes
    if cursor != len(raw):
        raise ReferenceError("terminal pack has unassigned trailing bytes")
    return result


def _write_artifacts(
    admitted: Mapping[str, Any],
    torch: Any,
    cpu_tensors: dict[str, Any],
    inventory: list[dict[str, Any]],
    *,
    decode_enqueue_seconds: float,
    through_terminal_seconds: float,
) -> dict[str, Any]:
    output_dir = Path(admitted["output_dir"])
    artifact_path = output_dir / "decoder-reference.safetensors"
    recipe_path = output_dir / "recipe.json"
    output_dir.mkdir(parents=False, exist_ok=False)
    try:
        artifact_bytes, artifact_sha256 = artifact._write_safetensors_atomic(
            artifact_path,
            cpu_tensors,
            {
                "reference_schema": SCHEMA,
                "upstream_module_sha256": admitted["module_sha256"],
                "checkpoint_sha256": admitted["weights_sha256"],
                "latent_bf16_sha256": admitted["input_source"]["decoder_bf16_sha256"]
                if "decoder_bf16_sha256" in admitted["input_source"]
                else admitted["input_source"]["bf16_sha256"],
            },
        )
        tensor_records = {
            item["name"]: {
                "shape": item["shape"],
                "dtype": item["dtype"],
                "nbytes": item["nbytes"],
                "offset": item["offset"],
            }
            for item in inventory
        }
        recipe = {
            "schema": SCHEMA,
            "variant": VARIANT,
            "upstream": {
                "repository": "Tencent-Hunyuan/Hy-Nomas",
                "revision": dcae.UPSTREAM_REVISION,
                "module_path": str(admitted["module_path"]),
                "module_sha256": admitted["module_sha256"],
            },
            "checkpoint": {
                "path": str(admitted["weights_path"]),
                "sha256": admitted["weights_sha256"],
                "bytes": dcae.DCAE_CHECKPOINT_BYTES,
                "repository_revision": artifact.CHECKPOINT_REVISION,
                "load_policy": "torch.load(weights_only=True,mmap=True,map_location=cpu)",
                "materialized_component": "decoder only",
                "weight_dtype": "BF16",
                "weight_tensor_count": dcae.DECODER_TENSOR_COUNT,
                "weight_payload_bytes": dcae.DECODER_BF16_BYTES,
            },
            "input": admitted["input_source"],
            "decode": {
                "input_layout": "BCTHW",
                "output_layout": "BCTHW",
                "input_shape": list(admitted["latent_shape"]),
                "output_shape": list(output_shape(admitted["latent_shape"])),
                "tile_latents": TILE_LATENTS,
                "tile_frames": TILE_FRAMES,
                "tile_count": admitted["latent_shape"][2] // TILE_LATENTS,
                "overlap_latents": 0,
                "temporal_feature_cache": True,
                "cache_key_prefix": "decoder.",
                "short_final_tile_supported": False,
                "capture_policy": capture_policy(admitted["latent_shape"]),
                "captured_tiles": list(admitted["selected_trace_tiles"]),
                "captured_boundaries": [
                    "decoder.project_in",
                    "decoder.stages.6..1",
                    "decoder.project_out",
                ] if admitted["selected_trace_tiles"] else [],
            },
            "execution": {
                "backend": "pytorch_mps",
                "torch_version": torch.__version__,
                "device": "mps",
                "dtype": "BF16",
                "cpu_fallback_enabled": False,
                "intermediate_device_to_host_readbacks": 0,
                "explicit_mps_synchronize_calls": 0,
                "terminal_transfer_count": 1,
                "seconds_decode_enqueue": decode_enqueue_seconds,
                "seconds_through_terminal": through_terminal_seconds,
            },
            "comparison": {
                "atol": ATOL,
                "rtol": RTOL,
                "interpretation": "BF16 value parity; mismatches are outside the declared tolerance",
                "tensor_keys": list(tensor_records),
            },
            "memory_estimate": admitted["memory_estimate"],
            "artifact": {
                "path": str(artifact_path),
                "bytes": artifact_bytes,
                "sha256": artifact_sha256,
                "tensors": tensor_records,
            },
        }
        artifact._write_json_atomic(recipe_path, recipe)
        return {
            "admission": "passed",
            "model_launched": True,
            "mps_work_started": True,
            "schema": SCHEMA,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "recipe_path": str(recipe_path),
            "output_shape": list(output_shape(admitted["latent_shape"])),
            "tensor_count": len(tensor_records),
            "terminal_transfer_count": 1,
            "intermediate_readbacks": 0,
            "explicit_mps_synchronize_calls": 0,
            "memory_estimate": admitted["memory_estimate"],
        }
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def execute_reference(admitted: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Run the pinned upstream decoder on MPS and export one packed readback."""
    if torch.__version__ != "2.8.0" or not torch.backends.mps.is_available():
        raise ReferenceError("execution requires the pinned Torch 2.8.0 MPS runtime")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be exactly '0'")
    if not torch.backends.mps.is_built():
        raise ReferenceError("this Torch build has no MPS backend")
    if artifact._identity(admitted["weights_path"]) != admitted["weights_identity"]:
        raise ReferenceError("original decoder checkpoint changed after admission")
    if artifact._identity(admitted["module_path"]) != admitted["module_identity"]:
        raise ReferenceError("upstream decoder source changed after admission")

    upstream = admitted["schema"]["upstream"]
    config = admitted["schema"]["config"]
    with torch.device("meta"):
        decoder = upstream.DCAEVDecoder(config.decoder)
    decoder.to(dtype=torch.bfloat16)
    decoder.to_empty(device="mps")
    local_state = {
        name.removeprefix("decoder."): tensor
        for name, tensor in admitted["decoder_state"].items()
    }
    try:
        decoder.load_state_dict(local_state, strict=True)
    except Exception as exc:
        raise ReferenceError(f"upstream decoder rejected its pinned BF16 checkpoint: {exc}") from None
    del local_state, admitted["decoder_state"]
    if artifact._identity(admitted["module_path"]) != admitted["module_identity"]:
        raise ReferenceError("upstream decoder source changed while weights were loaded")
    decoder.eval()

    latent = admitted["input_cpu"].to(device="mps", dtype=torch.bfloat16).contiguous()
    expected_output_shape = output_shape(admitted["latent_shape"])
    feature_cache: dict[str, Any] = {}
    selected_tiles = admitted["selected_trace_tiles"]
    traces: dict[str, Any] = {}
    active_tile: int | None = None
    handles = []

    if selected_tiles:
        for boundary, module in _trace_modules(decoder):
            def capture(_module: Any, _inputs: Any, output: Any, *, key: str = boundary) -> None:
                if active_tile not in selected_tiles:
                    return
                if not isinstance(output, torch.Tensor):
                    raise ReferenceError(f"upstream boundary did not return a tensor: {key}")
                if output.device.type != "mps" or output.dtype is not torch.bfloat16:
                    raise ReferenceError(f"upstream boundary is not MPS BF16: {key}")
                trace_name = f"trace.{key}.tile{active_tile}"
                if trace_name in traces:
                    raise ReferenceError(f"upstream boundary was captured twice: {trace_name}")
                traces[trace_name] = output.detach().contiguous().clone()

            handles.append(module.register_forward_hook(capture))

    decoded_tiles = []
    started = time.monotonic()
    try:
        with torch.no_grad():
            for tile_index, start in enumerate(range(0, latent.shape[2], TILE_LATENTS)):
                active_tile = tile_index
                tile = latent[:, :, start : start + TILE_LATENTS].contiguous()
                if tile.shape[2] != TILE_LATENTS:
                    raise ReferenceError("upstream cached decode requires whole ten-latent tiles")
                decoded = decoder(tile, feature_cache, "decoder.")
                expected_tile_shape = (
                    expected_output_shape[0],
                    expected_output_shape[1],
                    TILE_FRAMES,
                    expected_output_shape[3],
                    expected_output_shape[4],
                )
                if tuple(decoded.shape) != expected_tile_shape or decoded.dtype is not torch.bfloat16:
                    raise ReferenceError(
                        f"upstream decoder tile violates geometry/dtype: {tuple(decoded.shape)}, {decoded.dtype}"
                    )
                decoded_tiles.append(decoded)
    finally:
        for handle in handles:
            handle.remove()
    decode_enqueue_seconds = time.monotonic() - started
    if selected_tiles:
        expected_names = trace_tensor_names(selected_tiles)
        if set(traces) != set(expected_names):
            raise ReferenceError("upstream decoder stage trace is incomplete")

    decoded = torch.cat(decoded_tiles, dim=2).contiguous()
    if tuple(decoded.shape) != expected_output_shape or decoded.dtype is not torch.bfloat16:
        raise ReferenceError("joined upstream decoder output violates the admitted geometry/dtype")

    terminal_tensors: dict[str, Any] = {
        "input.latent": latent,
        "output.pixels": decoded,
    }
    for name in trace_tensor_names(selected_tiles):
        terminal_tensors[name] = traces[name]
    terminal, inventory = _pack_terminal(torch, terminal_tensors)
    through_terminal_seconds = time.monotonic() - started
    cpu_tensors = _unpack_terminal(torch, terminal, inventory)
    traces.clear()
    del terminal, terminal_tensors, decoded_tiles, decoded, latent, feature_cache
    return _write_artifacts(
        admitted,
        torch,
        cpu_tensors,
        inventory,
        decode_enqueue_seconds=decode_enqueue_seconds,
        through_terminal_seconds=through_terminal_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent-f32-path", type=Path)
    parser.add_argument("--latent-sha256")
    parser.add_argument("--latent-shape", type=int, nargs=5, metavar=("B", "C", "T", "H", "W"))
    parser.add_argument("--seed", type=int, default=SMOKE_SEED)
    parser.add_argument("--trace-tiles", type=parse_trace_tiles)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="admit CPU metadata without MPS execution")
    mode.add_argument("--execute", action="store_true", help="run the pinned upstream decoder on MPS")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        import torch

        admitted = preflight(args, torch)
        if args.execute:
            result = execute_reference(admitted, torch)
        else:
            result = {
                "admission": "passed",
                "model_launched": False,
                "mps_work_started": False,
                "schema": SCHEMA,
                "input": admitted["input_source"],
                "capture_policy": capture_policy(admitted["latent_shape"]),
                "trace_tiles": list(admitted["selected_trace_tiles"]),
                "output_shape": list(output_shape(admitted["latent_shape"])),
                "checkpoint_sha256": admitted["weights_sha256"],
                "upstream_module_sha256": admitted["module_sha256"],
                "memory_estimate": admitted["memory_estimate"],
                "output_dir": str(admitted["output_dir"]),
            }
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ReferenceError, dcae.ControlError, artifact.ReferenceError, ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f"DCAE-V decoder reference failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
