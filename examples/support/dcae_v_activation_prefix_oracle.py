#!/usr/bin/env python3
"""Export two pinned upstream DCAE-V stage-2 Conv3D entry activations.

This oracle executes only the original decoder prefix through
``decoder.stages.2.op_list.1.main.conv1`` for the first two retained
prefix20 tiles. A forward pre-hook captures the target activation and OLD
causal cache, then raises before the target ConvLayer3d runs. The tile-0
target cache is seeded with the exact reshape/slice used by the pinned
upstream ConvLayer3d before tile 1. No later decoder modules execute.

The result is separate evidence for upstream-prefix parity. It does not run or
compare the target Conv3D operator on native inputs; the reduction oracle owns
that identical-input operator comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


SUPPORT = Path(__file__).resolve().parent
REPO_ROOT = SUPPORT.parents[1]
SPEC_SUPPORT = REPO_ROOT / "spec" / "support"
for entry in (str(SUPPORT), str(SPEC_SUPPORT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import dcae_v_reference as artifact  # noqa: E402
import hy_nomas_photo_dcae_decode_control as dcae  # noqa: E402


SCHEMA = "cogni_ml.dcae_v_activation_prefix_upstream/v1"
VARIANT = "dc-ae-v-1.0-f32t4c32-bf16"
TARGET_CONV = "decoder.stages.2.op_list.1.main.conv1.conv"
TARGET_CACHE_KEY = "decoder.stages.2.op_list.1.main.conv1."
TILE_COUNT = 2
TILE_LATENTS = 10
TILE_FRAMES = 40
LATENT_SHAPE = (1, 32, 20, 9, 16)
ACTIVATION_SHAPE = (1, 512, 40, 72, 128)
CACHE_SHAPE = (1, 512, 1, 72, 128)
ATOL = RTOL = 2e-5

DEFAULT_REFERENCE = Path(
    "/opt/mldata_ssd/cogni-ml/dcae-conv-profile-20261108.dwteqxm5/"
    "upstream-prefix20.safetensors"
)
REFERENCE_SHA256 = "db96ee9247e69b338e3ce8d4ea349404208d6d50d6ba022548751b566632bc4e"
REFERENCE_INPUT_SHA256 = (
    "43081cf72cc0feda5edff7531f449209a542a2fb8d5ba8ae15d3067a06462a3c"
)
REFERENCE_OUTPUT_SHA256 = (
    "1ea898048e2cc31bd82baf0c85bf4586f0d99137d14c8ba6b75f7d780dd0b9fe"
)
REFERENCE_LINEAGE_PATH = Path(
    "/opt/mldata_ssd/cogni-ml/dcae-conv-profile-20261108.dwteqxm5/"
    "prefix-lineage-verified.json"
)
REFERENCE_LINEAGE_SHA256 = (
    "7961bd5f4ed32793e83ddd5eb5e982c49a3958d647f5718046ce3f867eaa256c"
)
REFERENCE_SOURCE_PATH = Path(
    "/opt/mldata_ssd/cogni-ml/dcae-native-decoder-20261108.t9_k76ut/"
    "upstream-full/decoder-reference.safetensors"
)
REFERENCE_SOURCE_SHA256 = (
    "0e873cba80446ef26920469ef3c1700f6e1686d021fb35b973872de4e383c2c7"
)

DEFAULT_NATIVE_WEIGHTS = Path(
    "/opt/mldata_ssd/models/hy-nomas/dcae-v-native-decoder-ebbee483/"
    "decoder-bf16.safetensors"
)
NATIVE_WEIGHTS_SHA256 = (
    "f53c94b06685027b8131b7f52cab519edee1e51287d93d8fdac6a17506f8881b"
)
DEFAULT_OUTPUT_DIR = Path(
    "/opt/mldata_ssd/cogni-ml/dcae-real-stage2-20261108.9Enu7v29/upstream-helper"
)
PREFIX_MODULES = (
    "project_in",
    "stages.6",
    "stages.5",
    "stages.4",
    "stages.3",
    "stages.2.op_list.0",
)
CASE_FILES = {
    "invalid": ("invalid-input.bf16", "invalid-cache.bf16"),
    "valid": ("valid-input.bf16", "valid-cache.bf16"),
}
GIB = 1024**3
MIB = 1024**2
EXTERNAL_CAP_BYTES = 18 * GIB
PHYSICAL_RESERVE_FRACTION = 0.10
DEFAULT_TIMEOUT_SECONDS = 1200


class OracleError(ValueError):
    """A pinned input, upstream boundary, runtime, or output was rejected."""


class _TargetEntryReached(Exception):
    """Private control flow used to stop before the target module executes."""


def activation_prefix_fields() -> dict[str, Any]:
    """Return the exact shared contract consumed by native prefix comparison."""
    return {
        "reference_path": str(DEFAULT_REFERENCE),
        "reference_sha256": REFERENCE_SHA256,
        "target_conv": TARGET_CONV,
        "tile_count": TILE_COUNT,
        "activation_shape": list(ACTIVATION_SHAPE),
        "cache_shape": list(CACHE_SHAPE),
    }


def validate_bcthw_shape(
    shape: Sequence[int], expected: Sequence[int], label: str
) -> tuple[int, ...]:
    """Validate a positive fixed-rank BCTHW shape without importing Torch."""
    if type(shape) not in (tuple, list) or any(
        type(dim) is not int or dim <= 0 for dim in shape
    ):
        raise OracleError(f"{label} shape must contain positive integer dimensions")
    result = tuple(shape)
    if result != tuple(expected):
        raise OracleError(
            f"{label} shape differs from the pinned boundary: {result} != {tuple(expected)}"
        )
    return result


def terminal_inventory() -> tuple[dict[str, Any], ...]:
    """Declare the full activation/cache tensors in deterministic transfer order."""
    records = []
    for case in ("invalid", "valid"):
        input_name, cache_name = CASE_FILES[case]
        records.extend(
            (
                {
                    "case": case,
                    "name": input_name,
                    "shape": list(ACTIVATION_SHAPE),
                    "dtype": "BF16",
                },
                {
                    "case": case,
                    "name": cache_name,
                    "shape": list(CACHE_SHAPE),
                    "dtype": "BF16",
                },
            )
        )
    return tuple(records)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _file_identity(path: Path, label: str) -> tuple[int, int, int, int]:
    try:
        identity = artifact._identity(path)
    except (OSError, artifact.ReferenceError) as exc:
        raise OracleError(f"cannot identify {label}: {exc}") from None
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a regular non-symlink file: {path}")
    return identity


def _sha256_file(
    path: Path, expected: str, label: str
) -> tuple[str, tuple[int, int, int, int]]:
    identity = _file_identity(path, label)
    observed = artifact.sha256_file(path)
    if observed != expected:
        raise OracleError(f"{label} SHA-256 differs from the pinned value")
    if _file_identity(path, label) != identity:
        raise OracleError(f"{label} changed while its SHA-256 was computed")
    return observed, identity


def _read_lineage(path: Path) -> dict[str, Any]:
    _sha256_file(path, REFERENCE_LINEAGE_SHA256, "prefix lineage audit")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleError(f"cannot parse prefix lineage audit: {exc}") from None
    if type(document) is not dict or document.get("passed") is not True:
        raise OracleError("prefix lineage audit is not a passing object")
    if document.get("source_path") != str(REFERENCE_SOURCE_PATH):
        raise OracleError("prefix lineage source path differs from its pin")
    if document.get("source_sha256") != REFERENCE_SOURCE_SHA256:
        raise OracleError("prefix lineage source hash differs from its pin")
    selected = None
    for row in document.get("prefixes", []):
        if (
            row.get("path") == str(DEFAULT_REFERENCE)
            and row.get("sha256") == REFERENCE_SHA256
        ):
            selected = row
            break
    if selected is None or selected.get("passed") is not True:
        raise OracleError("prefix20 entry is absent or failed in the lineage audit")
    inputs = [
        tensor
        for tensor in selected.get("tensors", [])
        if tensor.get("name") == "input.latent"
    ]
    if len(inputs) != 1:
        raise OracleError("prefix20 latent lineage entry is missing or ambiguous")
    record = inputs[0]
    if (
        record.get("source_shape") != [1, 32, 60, 9, 16]
        or record.get("prefix_shape") != list(LATENT_SHAPE)
        or record.get("compared_bytes") != math.prod(LATENT_SHAPE) * 2
        or record.get("source_causal_prefix_sha256") != REFERENCE_INPUT_SHA256
        or record.get("prefix_payload_sha256") != REFERENCE_INPUT_SHA256
        or record.get("passed") is not True
    ):
        raise OracleError(
            "prefix20 latent is not byte-lineaged to the pinned full reference"
        )
    outputs = [
        tensor
        for tensor in selected.get("tensors", [])
        if tensor.get("name") == "output.pixels"
    ]
    if (
        len(outputs) != 1
        or outputs[0].get("prefix_payload_sha256") != REFERENCE_OUTPUT_SHA256
    ):
        raise OracleError(
            "prefix20 pixel reference lineage differs from the pinned evidence"
        )
    return {
        "document": document,
        "prefix": selected,
        "input": record,
        "output": outputs[0],
    }


def _load_prefix_latent(torch: Any, path: Path) -> tuple[Any, dict[str, Any]]:
    """Authenticate the retained latent payload and return a CPU BF16 tensor."""
    _sha256_file(path, REFERENCE_SHA256, "retained prefix20 safetensors")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise OracleError(
            f"safetensors is required for the retained prefix fixture: {exc}"
        ) from None
    identity = _file_identity(path, "retained prefix20 safetensors")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as archive:
            if set(archive.keys()) != {"input.latent", "output.pixels"}:
                raise OracleError(
                    "prefix20 reference tensor inventory differs from its pin"
                )
            latent = archive.get_tensor("input.latent")
            pixels = archive.get_tensor("output.pixels")
    except OracleError:
        raise
    except Exception as exc:
        raise OracleError(f"cannot read retained prefix20 tensors: {exc}") from None
    validate_bcthw_shape(tuple(latent.shape), LATENT_SHAPE, "prefix20 latent")
    if (
        latent.dtype is not torch.bfloat16
        or latent.device.type != "cpu"
        or not latent.is_contiguous()
    ):
        raise OracleError("prefix20 latent must be contiguous CPU BF16")
    if (
        tuple(pixels.shape) != (1, 3, 80, 288, 512)
        or pixels.dtype is not torch.bfloat16
    ):
        raise OracleError(
            "prefix20 output pixels differ from the retained fixture geometry"
        )
    latent_digest = hashlib.sha256(
        latent.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    if latent_digest != REFERENCE_INPUT_SHA256:
        raise OracleError(
            "prefix20 latent payload differs from the pinned causal prefix"
        )
    pixel_digest = hashlib.sha256(
        pixels.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    if pixel_digest != REFERENCE_OUTPUT_SHA256:
        raise OracleError("prefix20 pixels differ from the pinned retained reference")
    if not bool(torch.isfinite(latent).all()) or not bool(torch.isfinite(pixels).all()):
        raise OracleError("retained prefix20 fixture contains non-finite values")
    if _file_identity(path, "retained prefix20 safetensors") != identity:
        raise OracleError("retained prefix20 fixture changed while it was read")
    return latent.contiguous(), {
        "path": str(path.resolve(strict=True)),
        "sha256": REFERENCE_SHA256,
        "tensor": "input.latent",
        "tensor_shape": list(LATENT_SHAPE),
        "tensor_dtype": "BF16",
        "tensor_sha256": latent_digest,
        "pixels_tensor_sha256": pixel_digest,
        "lineage_audit_path": str(REFERENCE_LINEAGE_PATH),
        "lineage_audit_sha256": REFERENCE_LINEAGE_SHA256,
        "full_source_reference_path": str(REFERENCE_SOURCE_PATH),
        "full_source_reference_sha256": REFERENCE_SOURCE_SHA256,
        "lineage_scope": "The latent and pixel payloads are byte-exact prefixes of a retained full upstream reference; this oracle independently reruns pinned upstream prefix activations.",
    }


def _resolve_output_dir(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise OracleError("output-dir must be an absolute path")
    if path.exists() or path.is_symlink():
        raise OracleError("output-dir must name a fresh path")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError:
        raise OracleError("output-dir parent must already exist") from None
    if not parent.is_dir() or not path.name or path.name in {".", ".."}:
        raise OracleError("output-dir parent must be an existing directory")
    resolved = parent / path.name
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise OracleError("oracle artifacts must be written outside the repository")
    if resolved.exists() or resolved.is_symlink():
        raise OracleError("output-dir appeared during admission")
    return resolved


def _selected_state_names(decoder_state: Mapping[str, Any]) -> tuple[str, ...]:
    selected = tuple(
        name
        for name in decoder_state
        if name.startswith("decoder." + PREFIX_MODULES[0] + ".")
        or any(
            name.startswith("decoder." + prefix + ".") for prefix in PREFIX_MODULES[1:]
        )
    )
    if not selected:
        raise OracleError(
            "no original checkpoint tensors matched the executed prefix modules"
        )
    return selected


def _authenticate_native_weight_cache(
    path: Path,
    decoder_state: Mapping[str, Any],
    selected_names: Sequence[str],
    torch: Any,
) -> dict[str, Any]:
    """Pin the full native decoder cache and bit-join every loaded tensor."""
    _sha256_file(path, NATIVE_WEIGHTS_SHA256, "native BF16 decoder weight cache")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise OracleError(
            f"safetensors is required to authenticate native decoder weights: {exc}"
        ) from None
    identity = _file_identity(path, "native BF16 decoder weight cache")
    source_names = set(decoder_state)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as archive:
            if set(archive.keys()) != source_names:
                raise OracleError(
                    "native cache tensor inventory differs from the original decoder state"
                )
            joined_bytes = 0
            for name in selected_names:
                original = decoder_state[name]
                cached = archive.get_tensor(name)
                if (
                    not isinstance(original, torch.Tensor)
                    or original.dtype is not torch.bfloat16
                    or cached.dtype is not torch.bfloat16
                    or tuple(cached.shape) != tuple(original.shape)
                    or not torch.equal(
                        original.contiguous().view(torch.uint16),
                        cached.contiguous().view(torch.uint16),
                    )
                ):
                    raise OracleError(
                        f"native cached BF16 tensor differs bytewise from original checkpoint: {name}"
                    )
                joined_bytes += original.numel() * 2
                del cached
    except OracleError:
        raise
    except Exception as exc:
        raise OracleError(
            f"cannot authenticate native decoder weight cache: {exc}"
        ) from None
    if _file_identity(path, "native BF16 decoder weight cache") != identity:
        raise OracleError(
            "native decoder weight cache changed while selected tensors were checked"
        )
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": NATIVE_WEIGHTS_SHA256,
        "bytes": identity[2],
        "tensor_count": len(source_names),
        "selected_prefix_tensor_count": len(selected_names),
        "selected_prefix_bytes_bitwise_joined_to_original_checkpoint": joined_bytes,
        "comparison": "uint16 payload equality for every materialized prefix state tensor",
    }


def _memory_plan(prefix_weight_bytes: int) -> dict[str, Any]:
    activation_bytes = math.prod(ACTIVATION_SHAPE) * 2
    cache_bytes = math.prod(CACHE_SHAPE) * 2
    input_bytes = math.prod(LATENT_SHAPE) * 2
    exports = activation_bytes * TILE_COUNT + cache_bytes * TILE_COUNT
    return {
        "original_checkpoint_mmap_bytes": dcae.DCAE_CHECKPOINT_BYTES,
        "original_decoder_total_bf16_weight_bytes": dcae.DECODER_BF16_BYTES,
        "mps_materialized_prefix_weight_bytes": prefix_weight_bytes,
        "mps_materialized_prefix_modules": list(PREFIX_MODULES),
        "target_conv_and_later_decoder_weights_materialized_on_mps": False,
        "input_bf16_bytes": input_bytes,
        "one_activation_bf16_bytes": activation_bytes,
        "two_activation_capture_bf16_bytes": activation_bytes * TILE_COUNT,
        "two_old_cache_capture_bf16_bytes": cache_bytes * TILE_COUNT,
        "terminal_export_pack_bf16_bytes": exports,
        "external_process_group_cap_bytes": EXTERNAL_CAP_BYTES,
        "physical_memory_reserve_fraction": PHYSICAL_RESERVE_FRACTION,
        "external_guard_required": True,
        "opaque_mps_workspace_included": False,
        "warning": "Named tensors are not a peak bound; run only under the external 18 GiB process cap and 10% physical-memory reserve.",
    }


def preflight(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    """Authenticate source, fixture and weight lineage without allocating MPS."""
    reference_path = Path(args.reference).expanduser()
    if reference_path.resolve(strict=True) != DEFAULT_REFERENCE.resolve(strict=True):
        raise OracleError("only the pinned retained prefix20 fixture path is admitted")
    lineage = _read_lineage(REFERENCE_LINEAGE_PATH)
    _sha256_file(
        REFERENCE_SOURCE_PATH,
        REFERENCE_SOURCE_SHA256,
        "full upstream decoder reference",
    )

    upstream_root = Path(args.upstream).expanduser().resolve(strict=True)
    if dcae._git_head(upstream_root) != dcae.UPSTREAM_REVISION:
        raise OracleError(
            "upstream checkout revision differs from the pinned Hy-Nomas source"
        )
    module_path = dcae._regular_path(
        upstream_root / "hymm/models/vae/autoencoder_dcae_v.py", "upstream DCAE module"
    )
    module_identity = artifact._identity(module_path)
    module_sha256 = artifact.sha256_file(module_path)
    if module_sha256 != dcae.DCAE_MODULE_SHA256:
        raise OracleError("upstream DCAE module SHA-256 differs from its pin")

    weights_path = dcae._regular_path(args.weights, "original BF16 DCAE checkpoint")
    weights_identity = artifact._identity(weights_path)
    if weights_identity[2] != dcae.DCAE_CHECKPOINT_BYTES:
        raise OracleError(
            "original BF16 DCAE checkpoint byte extent differs from its pin"
        )
    weights_sha256 = artifact.sha256_file(weights_path)
    if weights_sha256 != dcae.DCAE_CHECKPOINT_SHA256:
        raise OracleError("original BF16 DCAE checkpoint SHA-256 differs from its pin")
    if artifact._identity(weights_path) != weights_identity:
        raise OracleError(
            "original BF16 DCAE checkpoint changed while its SHA-256 was computed"
        )

    try:
        decoder_state, schema = dcae._read_checkpoint_metadata(
            weights_path, module_path, torch
        )
    except (
        dcae.ControlError,
        artifact.ReferenceError,
        ValueError,
        OSError,
        KeyError,
        TypeError,
        RuntimeError,
    ) as exc:
        raise OracleError(f"original decoder schema admission failed: {exc}") from None
    config = schema["config"]
    if (
        config.latent_channels != 32
        or config.decode_temporal_tile_size != TILE_FRAMES
        or config.decode_temporal_tile_latent_size != TILE_LATENTS
        or config.decode_temporal_tile_overlap_factor != 0
        or config.use_feature_cache is not True
        or config.num_pad_frames != 0
    ):
        raise OracleError(
            "upstream decoder tile/cache config differs from the two-tile prefix contract"
        )
    if tuple(config.decoder.width_list) != (128, 256, 512, 512, 1024, 1024, 1024):
        raise OracleError(
            "upstream decoder width schedule differs from the pinned variant"
        )
    if tuple(config.decoder.depth_list) != (0, 5, 10, 4, 4, 4, 4):
        raise OracleError(
            "upstream decoder depth schedule differs from the pinned variant"
        )
    if (
        artifact._identity(module_path) != module_identity
        or artifact._identity(weights_path) != weights_identity
    ):
        raise OracleError(
            "upstream source or original checkpoint changed during schema admission"
        )

    selected_names = _selected_state_names(decoder_state)
    native_weight_cache = _authenticate_native_weight_cache(
        Path(args.native_weights).expanduser(), decoder_state, selected_names, torch
    )
    selected_weight_bytes = sum(
        decoder_state[name].numel() * 2 for name in selected_names
    )
    if selected_weight_bytes <= 0:
        raise OracleError("executed prefix has no authenticated BF16 weights")

    input_cpu, input_source = _load_prefix_latent(torch, reference_path)
    output_dir = _resolve_output_dir(Path(args.output_dir))
    if sys.byteorder != "little":
        raise OracleError("raw BF16 outputs require a little-endian host")

    return {
        "input_cpu": input_cpu,
        "input_source": input_source,
        "lineage": lineage,
        "output_dir": output_dir,
        "upstream_root": upstream_root,
        "module_path": module_path,
        "module_identity": module_identity,
        "module_sha256": module_sha256,
        "weights_path": weights_path,
        "weights_identity": weights_identity,
        "weights_sha256": weights_sha256,
        "decoder_state": decoder_state,
        "schema": schema,
        "selected_names": selected_names,
        "selected_weight_bytes": selected_weight_bytes,
        "native_weight_cache": native_weight_cache,
        "torch_version": torch.__version__,
        "timeout_seconds": args.timeout_seconds,
        "memory_plan": _memory_plan(selected_weight_bytes),
    }


def _materialize_prefix_modules(
    admitted: dict[str, Any], decoder: Any, torch: Any
) -> None:
    """Move only modules executed before the target entry to MPS."""
    decoder.to(dtype=torch.bfloat16)
    state = admitted["decoder_state"]
    for module_path in PREFIX_MODULES:
        module = decoder.get_submodule(module_path)
        module.to_empty(device="mps")
        prefix = f"decoder.{module_path}."
        local_state = {
            name.removeprefix(prefix): tensor
            for name, tensor in state.items()
            if name.startswith(prefix)
        }
        if not local_state:
            raise OracleError(
                f"no original checkpoint state was selected for {module_path}"
            )
        try:
            module.load_state_dict(local_state, strict=True)
        except Exception as exc:
            raise OracleError(
                f"upstream prefix module rejected its original BF16 weights ({module_path}): {exc}"
            ) from None
        del local_state
    if any(
        parameter.device.type != "mps"
        for module_path in PREFIX_MODULES
        for parameter in decoder.get_submodule(module_path).parameters()
    ):
        raise OracleError(
            "an executed prefix module parameter was not materialized on MPS"
        )
    # Keep all unused stages, including the target ConvLayer3d's parameters,
    # on meta. Their forward methods are unreachable after the target pre-hook.
    del admitted["decoder_state"]
    decoder.eval()


def _target_internal_cache(
    torch: Any,
    activation: Any,
    chunk_length: int,
    temporal_pad: int,
    *,
    expected_shape: Sequence[int] = ACTIVATION_SHAPE,
) -> Any:
    """Match upstream ConvLayer3d's cache write before the target Conv3D."""
    if activation.dtype is not torch.bfloat16:
        raise OracleError("target activation must be BF16")
    if tuple(activation.shape) != tuple(expected_shape):
        raise OracleError(
            f"target activation shape differs from the pin: {tuple(activation.shape)}"
        )
    batch, channels, frames, height, width = activation.shape
    if chunk_length != TILE_FRAMES or frames % chunk_length or temporal_pad != 1:
        raise OracleError(
            "target ConvLayer3d chunk or temporal padding differs from the pinned contract"
        )
    chunks = activation.reshape(
        batch, channels, frames // chunk_length, chunk_length, height, width
    ).transpose(1, 2)
    return chunks[:, -1:, :, -temporal_pad:].clone()


def _target_cache_bcthw(
    torch: Any,
    internal_cache: Any,
    *,
    expected_shape: Sequence[int] = CACHE_SHAPE,
) -> Any:
    """Convert upstream [B,chunks,C,tp,H,W] storage to comparator BCTHW."""
    shape = tuple(expected_shape)
    if len(shape) != 5:
        raise OracleError("target cache shape must use BCTHW layout")
    expected_internal = (shape[0], 1, shape[1], shape[2], shape[3], shape[4])
    if tuple(internal_cache.shape) != expected_internal:
        raise OracleError(
            f"target old cache shape differs from upstream chunk storage: {tuple(internal_cache.shape)}"
        )
    if internal_cache.dtype is not torch.bfloat16:
        raise OracleError("target old cache must be BF16")
    return internal_cache[:, 0].contiguous()


def _run_two_tiles(
    admitted: dict[str, Any], torch: Any
) -> tuple[dict[str, Any], int, float]:
    """Execute the unmodified upstream prefix twice, stopping at target entry."""
    if (
        torch.__version__ != "2.8.0"
        or not torch.backends.mps.is_available()
        or not torch.backends.mps.is_built()
    ):
        raise OracleError("execution requires the pinned Torch 2.8.0 MPS runtime")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise OracleError("PYTORCH_ENABLE_MPS_FALLBACK must be exactly '0'")
    if (
        _file_identity(admitted["module_path"], "upstream DCAE module")
        != admitted["module_identity"]
    ):
        raise OracleError("upstream DCAE module changed after admission")
    if (
        _file_identity(admitted["weights_path"], "original BF16 DCAE checkpoint")
        != admitted["weights_identity"]
    ):
        raise OracleError("original DCAE checkpoint changed after admission")

    upstream = admitted["schema"]["upstream"]
    config = admitted["schema"]["config"]
    with torch.device("meta"):
        decoder = upstream.DCAEVDecoder(config.decoder)
    _materialize_prefix_modules(admitted, decoder, torch)
    if len(decoder.stages) != 7 or len(decoder.stages[2].op_list) < 2:
        raise OracleError(
            "pinned decoder stage-2 operation layout differs from target path"
        )
    target = decoder.stages[2].op_list[1].main.conv1
    if target.causal_chunk_length != TILE_FRAMES or tuple(target.conv.padding) != (
        1,
        1,
        1,
    ):
        raise OracleError(
            "target ConvLayer3d temporal chunk/padding differs from the pinned contract"
        )

    input_mps = (
        admitted["input_cpu"].to(device="mps", dtype=torch.bfloat16).contiguous()
    )
    feature_cache: dict[str, Any] = {}
    captures: dict[str, Any] = {}
    active_tile = -1
    seeded_cache: Any | None = None

    def stop_before_target(_module: Any, inputs: Any) -> None:
        nonlocal seeded_cache
        if len(inputs) < 3:
            raise OracleError(
                "target ConvLayer3d did not receive the upstream cache arguments"
            )
        activation, current_cache, feature_key = inputs[:3]
        if feature_key != TARGET_CACHE_KEY:
            raise OracleError(
                f"target cache key differs from pinned upstream path: {feature_key!r}"
            )
        if not isinstance(current_cache, dict):
            raise OracleError("target feature cache is not the upstream dictionary")
        if active_tile not in (0, 1):
            raise OracleError("target hook fired outside the first two prefix tiles")
        case = "invalid" if active_tile == 0 else "valid"
        if case in captures:
            raise OracleError(f"target activation captured more than once for {case}")
        if activation.device.type != "mps" or activation.dtype is not torch.bfloat16:
            raise OracleError("target activation is not MPS BF16")
        validate_bcthw_shape(
            tuple(activation.shape), ACTIVATION_SHAPE, "target Conv3D input"
        )
        old_cache = current_cache.get(feature_key)
        if active_tile == 0:
            if old_cache is not None:
                raise OracleError("tile 0 unexpectedly has an old target causal cache")
            exported_cache = torch.zeros(
                CACHE_SHAPE, dtype=torch.bfloat16, device="mps"
            )
            captures[case] = {
                "input": activation.detach().contiguous(),
                "cache": exported_cache,
                "old_cache_present": False,
            }
            # Pinned source ConvLayer3d.forward reshapes to [B,chunks,C,chunk,H,W]
            # and writes x[:, -1:, :, -tp:].clone() before executing Conv3D.
            seeded_cache = _target_internal_cache(
                torch,
                activation,
                int(target.causal_chunk_length),
                int(target.conv.padding[0]),
            )
            current_cache[feature_key] = seeded_cache
        else:
            if old_cache is None or old_cache is not seeded_cache:
                raise OracleError(
                    "tile 1 target cache is not the exact cache seeded from tile 0"
                )
            exported_cache = _target_cache_bcthw(torch, old_cache)
            captures[case] = {
                "input": activation.detach().contiguous(),
                "cache": exported_cache,
                "old_cache_present": True,
            }
        raise _TargetEntryReached

    handle = target.register_forward_pre_hook(stop_before_target)
    start = time.monotonic()
    try:
        with torch.no_grad():
            for tile_index in range(TILE_COUNT):
                active_tile = tile_index
                tile = input_mps[
                    :, :, tile_index * TILE_LATENTS : (tile_index + 1) * TILE_LATENTS
                ].contiguous()
                x = decoder.project_in(tile, feature_cache, "decoder.project_in.")
                for stage_id in (6, 5, 4, 3):
                    stage = decoder.stages[stage_id]
                    if not stage.op_list:
                        raise OracleError(
                            f"executed upstream prefix stage {stage_id} is empty"
                        )
                    x = stage(x, feature_cache, f"decoder.stages.{stage_id}.")
                stage2 = decoder.stages[2]
                x = stage2.op_list[0](x, feature_cache, "decoder.stages.2.op_list.0.")
                try:
                    stage2.op_list[1](x, feature_cache, "decoder.stages.2.op_list.1.")
                except _TargetEntryReached:
                    continue
                raise OracleError("target ConvLayer3d pre-hook did not stop the prefix")
    finally:
        handle.remove()
    enqueue_seconds = time.monotonic() - start
    if set(captures) != {"invalid", "valid"}:
        raise OracleError("upstream target prefix did not capture exactly two tiles")
    for case, record in captures.items():
        validate_bcthw_shape(
            tuple(record["input"].shape), ACTIVATION_SHAPE, f"{case} target activation"
        )
        validate_bcthw_shape(
            tuple(record["cache"].shape), CACHE_SHAPE, f"{case} old cache"
        )
        if record["input"].device.type != "mps" or record["cache"].device.type != "mps":
            raise OracleError(f"{case} capture left MPS before the terminal transfer")
        if (
            record["input"].dtype is not torch.bfloat16
            or record["cache"].dtype is not torch.bfloat16
        ):
            raise OracleError(f"{case} capture is not BF16")
    if (
        captures["invalid"]["old_cache_present"]
        or not captures["valid"]["old_cache_present"]
    ):
        raise OracleError(
            "tile cache-presence semantics differ from the causal prefix contract"
        )
    return (
        captures,
        sum(value.numel() * 2 for value in feature_cache.values()),
        enqueue_seconds,
    )


def _terminal_transfer(
    torch: Any, captures: Mapping[str, Mapping[str, Any]]
) -> tuple[bytes, list[dict[str, Any]], float]:
    """Pack four full MPS captures and perform one terminal host transfer."""
    pieces = []
    inventory = []
    offset = 0
    for case in ("invalid", "valid"):
        for field, name, expected_shape in (
            ("input", CASE_FILES[case][0], ACTIVATION_SHAPE),
            ("cache", CASE_FILES[case][1], CACHE_SHAPE),
        ):
            tensor = captures[case][field]
            if tensor.device.type != "mps" or tensor.dtype is not torch.bfloat16:
                raise OracleError(f"terminal tensor must be MPS BF16: {name}")
            if tuple(tensor.shape) != expected_shape:
                raise OracleError(f"terminal tensor shape differs from pin: {name}")
            contiguous = tensor.detach().contiguous()
            nbytes = contiguous.numel() * 2
            pieces.append(contiguous.view(torch.uint8).reshape(-1))
            inventory.append(
                {
                    "name": name,
                    "case": case,
                    "shape": list(expected_shape),
                    "dtype": "BF16",
                    "nbytes": nbytes,
                    "offset": offset,
                }
            )
            offset += nbytes
    packed = torch.cat(pieces, dim=0).contiguous()
    if (
        packed.device.type != "mps"
        or packed.dtype is not torch.uint8
        or packed.numel() != offset
    ):
        raise OracleError(
            "terminal MPS pack differs from its declared tensor inventory"
        )
    start = time.monotonic()
    raw = packed.cpu().numpy().tobytes()
    terminal_seconds = time.monotonic() - start
    if len(raw) != offset:
        raise OracleError(
            "terminal host buffer byte extent differs from its tensor inventory"
        )
    return raw, inventory, terminal_seconds


def _write_new_payload(path: Path, payload: memoryview) -> tuple[int, str]:
    digest = hashlib.sha256()
    digest.update(payload)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise OracleError(f"cannot write new output file {path}: {exc}") from None
    return len(payload), digest.hexdigest()


def _write_output(
    admitted: Mapping[str, Any],
    raw: bytes,
    inventory: Sequence[Mapping[str, Any]],
    torch: Any,
    *,
    cache_bytes: int,
    enqueue_seconds: float,
    terminal_seconds: float,
) -> dict[str, Any]:
    output_dir = Path(admitted["output_dir"])
    output_dir.mkdir(parents=False, exist_ok=False)
    records: dict[str, Any] = {}
    try:
        for item in inventory:
            name = item["name"]
            start = int(item["offset"])
            end = start + int(item["nbytes"])
            if start < 0 or end > len(raw):
                raise OracleError(f"terminal byte range is invalid for {name}")
            nbytes, sha256 = _write_new_payload(
                output_dir / name, memoryview(raw)[start:end]
            )
            records[name] = {
                "shape": item["shape"],
                "dtype": item["dtype"],
                "bytes": nbytes,
                "sha256": sha256,
            }

        fields = activation_prefix_fields()
        report = {
            "schema": SCHEMA,
            **fields,
            "weights_sha256": NATIVE_WEIGHTS_SHA256,
            "upstream_sha256": admitted["module_sha256"],
            "variant": VARIANT,
            "upstream": {
                "repository": "Tencent-Hunyuan/Hy-Nomas",
                "revision": dcae.UPSTREAM_REVISION,
                "checkout": str(admitted["upstream_root"]),
                "module_path": str(admitted["module_path"]),
                "module_sha256": admitted["module_sha256"],
            },
            "checkpoint": {
                "path": str(admitted["weights_path"]),
                "sha256": admitted["weights_sha256"],
                "bytes": dcae.DCAE_CHECKPOINT_BYTES,
                "load_policy": "torch.load(weights_only=True,mmap=True,map_location=cpu)",
            },
            "native_weight_cache": admitted["native_weight_cache"],
            "prefix_weight_load": {
                "materialized_prefix_modules": list(PREFIX_MODULES),
                "materialized_tensor_count": len(admitted["selected_names"]),
                "materialized_bf16_bytes": admitted["selected_weight_bytes"],
                "target_conv_weights_materialized": False,
                "encoder_weights_materialized": False,
                "later_decoder_weights_materialized": False,
                "source": "original BF16 checkpoint; every loaded prefix tensor was uint16-bitwise joined to the pinned native decoder cache",
            },
            "reference": admitted["input_source"],
            "causal_tiles": {
                "latent_input_shape": list(LATENT_SHAPE),
                "latent_tile_length": TILE_LATENTS,
                "decoder_tile_frames": TILE_FRAMES,
                "tile_indices": [0, 1],
                "cache_key": TARGET_CACHE_KEY,
                "invalid_case": "tile 0 has no old target cache; upstream substitutes a zero left pad",
                "valid_case": "tile 1 exports the tile-0 target input last frame stored in upstream chunk-cache layout",
                "cache_seed_semantics": "reshape(B,C,T//chunk,chunk,H,W).transpose(1,2)[:, -1:, :, -tp:].clone()",
                "stop_boundary": "forward pre-hook raises before target ConvLayer3d.forward; no later decoder operation executes",
            },
            "execution": {
                "backend": "pytorch_mps",
                "torch_version": torch.__version__,
                "device": "mps",
                "dtype": "BF16",
                "cpu_fallback_enabled": False,
                "prefix_tiles_enqueued": TILE_COUNT,
                "intermediate_device_to_host_readbacks": 0,
                "terminal_tensor_transfers": 1,
                "explicit_mps_synchronize_calls": 0,
                "seconds_prefix_enqueue": enqueue_seconds,
                "seconds_terminal_transfer": terminal_seconds,
                "seconds_through_terminal": enqueue_seconds + terminal_seconds,
                "configured_timeout_seconds": admitted["timeout_seconds"],
            },
            "operator_scope": {
                "upstream_prefix_vs_native_prefix": "full BF16 activation/cache byte comparison is performed by the root comparator",
                "upstream_operator_on_identical_native_input": "not performed by this script; the separate reduction oracle owns that comparison",
                "atol": ATOL,
                "rtol": RTOL,
            },
            "memory_estimate": {
                **admitted["memory_plan"],
                "observed_persistent_feature_cache_bf16_bytes_after_two_prefixes": cache_bytes,
            },
            "artifacts": records,
            "decay": "Invalidate if any pinned source, checkpoint, native weight cache, retained reference, lineage audit, target key, geometry, or runtime changes.",
        }
        report_path = output_dir / "upstream-prefix.json"
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Keep any partial output as failed evidence; absence of the manifest
        # means the comparator cannot authenticate this directory.
        raise
    return {
        "admission": "passed",
        "model_launched": True,
        "mps_work_started": True,
        "schema": SCHEMA,
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "upstream-prefix.json"),
        "artifact_names": list(records),
        "terminal_tensor_transfers": 1,
        "intermediate_readbacks": 0,
        "explicit_mps_synchronize_calls": 0,
        "memory_estimate": report["memory_estimate"],
    }


def execute(admitted: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Run two upstream prefix tiles and export all four tensors at once."""
    if artifact._identity(admitted["module_path"]) != admitted["module_identity"]:
        raise OracleError("upstream DCAE source changed after admission")
    if artifact._identity(admitted["weights_path"]) != admitted["weights_identity"]:
        raise OracleError("original decoder checkpoint changed after admission")
    captures, cache_bytes, enqueue_seconds = _run_two_tiles(admitted, torch)
    raw, inventory, terminal_seconds = _terminal_transfer(torch, captures)
    # Hashing and filesystem output happen only after the sole MPS->CPU transfer.
    result = _write_output(
        admitted,
        raw,
        inventory,
        torch,
        cache_bytes=cache_bytes,
        enqueue_seconds=enqueue_seconds,
        terminal_seconds=terminal_seconds,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=dcae.DEFAULT_UPSTREAM)
    parser.add_argument("--weights", type=Path, default=dcae.DEFAULT_WEIGHTS)
    parser.add_argument("--native-weights", type=Path, default=DEFAULT_NATIVE_WEIGHTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the bounded MPS prefix after full admission",
    )
    parser.add_argument(
        "--admitted",
        action="store_true",
        help="assert external 18 GiB/10%% memory guard is active",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if type(args.timeout_seconds) is not int or args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    if args.execute and not args.admitted:
        parser.error(
            "--execute requires --admitted after external memory-cap and reserve admission"
        )
    if not args.execute and args.admitted:
        parser.error("--admitted is only meaningful with --execute")
    if args.execute:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch is required for the prefix oracle: {exc}") from None
    admitted = preflight(args, torch)
    if args.execute:
        result = execute(admitted, torch)
    else:
        result = {
            "admission": "passed",
            "model_launched": False,
            "mps_work_started": False,
            "schema": SCHEMA,
            "input": admitted["input_source"],
            "native_weight_cache": admitted["native_weight_cache"],
            "mps_materialized_prefix_weight_bytes": admitted["selected_weight_bytes"],
            "memory_estimate": admitted["memory_plan"],
            "output_dir": str(admitted["output_dir"]),
            "execute_command_requires_external_guard": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
