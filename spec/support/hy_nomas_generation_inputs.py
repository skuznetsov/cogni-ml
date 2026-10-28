#!/usr/bin/env python3
"""Prepare authenticated, fixed first-segment Hy-Nomas input tensors on MPS.

This prepares inputs only. It does not construct a model or run inference. All
random generation, RoPE construction, source-order cropping, and layout
permutation happen on MPS; one packed terminal copy exports the tensor payload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SPEC_SUPPORT = ROOT / "spec" / "support"
if str(SPEC_SUPPORT) not in sys.path:
    sys.path.insert(0, str(SPEC_SUPPORT))

SCHEMA = "cogni_ml.hy_nomas_generation_inputs/v1"
SCHEMA_VERSION = 1
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
OUTPUT_ROOT = Path("/opt/mldata_ssd")
QWEN_REFERENCE = Path(
    "/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024/fixtures/prefix-natural-tram-v1.json"
)
QWEN_SIDECAR = QWEN_REFERENCE.with_suffix(".bf16le")
QWEN_REFERENCE_SHA256 = "eb64e5c31bc2178436b2b7dd411af45dea7fa43a8e3ff37a146bc411fac03ba1"
QWEN_SIDECAR_SHA256 = "e7ebc9a784124afc20503435d84d42e0110d7c02054d3fc63fd586ec69ed4995"
QWEN_SIDECAR_BYTES = 268_627_968
QWEN_PROMPT_SHA256 = "91246a3eaf20efb68e83f9c6fe1e91afaedd8013a77971ebd74546fe2ed1e5a9"
PROMPT = (
    "A yellow tram turns through a wet downtown intersection at dusk as pedestrians "
    "wait under umbrellas."
)

LATENT_SHAPE = (1, 32, 30, 9, 16)
QWEN_SHAPE = (1, 1280, 3584)
DIT_ROPE_GRID = (30, 9, 16)
DIT_ROPE_SHAPE = (4320, 128)
PIXEL_ROPE_GRID = (10, 18, 32)
PIXEL_ROPE_SHAPE = (1, 5760, 1, 128)
PIXEL_NOISE_SHAPE = (1, 3, 120, 288, 512)
LATENT_SEED = 42
PIXEL_NOISE_SEEDS = tuple(LATENT_SEED * 1_000_003 + index for index in range(30))

GEOMETRY = {
    "dit_latent_shape": list(LATENT_SHAPE),
    "dit_rope_grid": list(DIT_ROPE_GRID),
    "dit_rope_axis_dims": [16, 56, 56],
    "dit_rope_theta": 256.0,
    "qwen_shape": list(QWEN_SHAPE),
    "qwen_state_index": 26,
    "qwen_crop": [108, 1388],
    "qwen_valid_tokens": 23,
    "decoder_latent_grid": [30, 9, 16],
    "decoder_noise_shape": list(PIXEL_NOISE_SHAPE),
    "decoder_noise_piece_latents": 1,
    "decoder_noise_piece_frames": 4,
    "decoder_rope_grid": list(PIXEL_ROPE_GRID),
    "decoder_rope_shape": list(PIXEL_ROPE_SHAPE),
    "decoder_rope_sections": [64, 32, 32],
    "decoder_rope_theta": [1_000_000.0, 10_000.0, 10_000.0],
    "decoder_local_attention_tile_tokens": 8,
    "decoder_local_attention_tile_pixels": 128,
}
NOISE_SEEDS = {
    "latent_seed": LATENT_SEED,
    "pixel_noise_seed_formula": "seed * 1000003 + latent_frame_index",
    "pixel_noise_seeds": list(PIXEL_NOISE_SEEDS),
}
QWEN_SOURCE_RECORD = {
    "reference_path": str(QWEN_REFERENCE),
    "reference_sha256": QWEN_REFERENCE_SHA256,
    "sidecar_path": str(QWEN_SIDECAR),
    "sidecar_sha256": QWEN_SIDECAR_SHA256,
    "sidecar_bytes": QWEN_SIDECAR_BYTES,
    "sidecar_encoding": "bfloat16-le",
    "state_index": 26,
    "crop": [108, 1388],
    "crop_tokens": 1280,
    "hidden_size": 3584,
    "valid_tokens": 23,
    "prompt_sha256": QWEN_PROMPT_SHA256,
}
UPSTREAM_SOURCE_PATHS = (
    "hymm/models/dit/modules/posemb_layers.py",
    "hymm/models/vae/pixelumm_decoder/model.py",
    "hymm/models/vae/pixelumm_decoder/autoencoder.py",
)
PINNED_UPSTREAM_SOURCE_SHA256 = {
    "hymm/models/dit/modules/posemb_layers.py": "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa",
    "hymm/models/vae/pixelumm_decoder/model.py": "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec",
    "hymm/models/vae/pixelumm_decoder/autoencoder.py": "65e0d77b9f2e9a3ca457d6a51bfdb239f7b9dc3fc8a0ad014025c36d6be9f0cf",
}
PRODUCER_RECORD = {
    "device": "mps",
    "mps_fallback": "disabled",
    "model_inference": False,
    "tensor_generation_device": "mps",
    "host_export": "one packed terminal MPS-to-host readback",
}
TENSOR_SPECS = (
    ("input.latent", "F32", LATENT_SHAPE),
    ("input.qwen_bf16", "BF16", QWEN_SHAPE),
    ("input.rope_cos", "F32", DIT_ROPE_SHAPE),
    ("input.rope_sin", "F32", DIT_ROPE_SHAPE),
    ("decoder.noise", "F32", PIXEL_NOISE_SHAPE),
    ("decoder.rope_cos", "F32", PIXEL_ROPE_SHAPE),
    ("decoder.rope_sin", "F32", PIXEL_ROPE_SHAPE),
)
_TENSOR_WIDTH = {"F32": 4, "BF16": 2}
_JSON_KEYS = {
    "schema", "schema_version", "payload", "payload_bytes", "payload_sha256",
    "tensors", "geometry", "prompt", "qwen_source", "noise_seeds",
    "upstream_revision", "upstream_sources", "producer",
}
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class InputsError(RuntimeError):
    """An input, provenance, output-admission, or MPS preparation failure."""


def tensor_nbytes(dtype: str, shape: tuple[int, ...] | list[int]) -> int:
    if dtype not in _TENSOR_WIDTH:
        raise InputsError(f"unsupported tensor dtype: {dtype}")
    if (not isinstance(shape, (tuple, list)) or not shape
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise InputsError("tensor shape must contain positive integers")
    return math.prod(shape) * _TENSOR_WIDTH[dtype]


def expected_payload_bytes() -> int:
    return sum(tensor_nbytes(dtype, shape) for _, dtype, shape in TENSOR_SPECS)


def _same_json_types(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return (type(actual) is dict and set(actual) == set(expected)
                and all(_same_json_types(value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (type(actual) is list and len(actual) == len(expected)
                and all(_same_json_types(left, right) for left, right in zip(expected, actual)))
    return type(actual) is type(expected) and actual == expected


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _HEX_DIGEST.fullmatch(value) is not None


def _validate_basename(value: Any, suffix: str) -> bool:
    return (isinstance(value, str) and value not in ("", ".", "..")
            and Path(value).name == value and value.endswith(suffix))


def validate_document(document: Mapping[str, Any], *, payload_bytes: int | None = None) -> None:
    """Validate fixed geometry and exact contiguous tensor inventory without torch."""
    if type(document) is not dict or set(document) != _JSON_KEYS:
        raise InputsError("generation-input document keys are malformed")
    if document.get("schema") != SCHEMA or type(document.get("schema_version")) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise InputsError("generation-input schema mismatch")
    if not _validate_basename(document.get("payload"), ".tensors.bin"):
        raise InputsError("payload must be a sibling .tensors.bin basename")
    if not _valid_digest(document.get("payload_sha256")):
        raise InputsError("payload SHA-256 is malformed")
    if not _same_json_types(GEOMETRY, document.get("geometry")):
        raise InputsError("fixed generation-input geometry mismatch")
    if document.get("prompt") != PROMPT or type(document.get("prompt")) is not str:
        raise InputsError("fixed preview prompt mismatch")
    if not _same_json_types(QWEN_SOURCE_RECORD, document.get("qwen_source")):
        raise InputsError("authenticated Qwen source record mismatch")
    if not _same_json_types(NOISE_SEEDS, document.get("noise_seeds")):
        raise InputsError("fixed noise seed inventory mismatch")
    if document.get("upstream_revision") != PINNED_REVISION:
        raise InputsError("pinned upstream revision mismatch")
    sources = document.get("upstream_sources")
    if not _same_json_types(PINNED_UPSTREAM_SOURCE_SHA256, sources):
        raise InputsError("pinned upstream source digest inventory mismatch")
    producer = document.get("producer")
    producer_keys = set(PRODUCER_RECORD) | {"torch_version", "python_version"}
    if (type(producer) is not dict or set(producer) != producer_keys
            or any(producer.get(key) != value for key, value in PRODUCER_RECORD.items())
            or any(type(producer.get(key)) is not str or not producer[key]
                   for key in ("torch_version", "python_version"))):
        raise InputsError("input producer scope/runtime record mismatch")

    records = document.get("tensors")
    expected_names = [name for name, _, _ in TENSOR_SPECS]
    # JSON object order is not the packed tensor order (publication sorts keys).
    # The source-order offset checks below establish the exact binary layout.
    if type(records) is not dict or set(records) != set(expected_names):
        raise InputsError("tensor inventory is incomplete")
    cursor = 0
    for name, dtype, shape in TENSOR_SPECS:
        record = records[name]
        expected_nbytes = tensor_nbytes(dtype, shape)
        if (type(record) is not dict or set(record) != {"dtype", "shape", "offset", "nbytes", "sha256"}
                or record.get("dtype") != dtype
                or not _same_json_types(list(shape), record.get("shape"))
                or type(record.get("offset")) is not int or record["offset"] != cursor
                or type(record.get("nbytes")) is not int or record["nbytes"] != expected_nbytes
                or not _valid_digest(record.get("sha256"))):
            raise InputsError(f"tensor record mismatch: {name}")
        cursor += expected_nbytes
    if type(document.get("payload_bytes")) is not int or document["payload_bytes"] != cursor:
        raise InputsError("payload byte count does not match the tensor inventory")
    if payload_bytes is not None and (type(payload_bytes) is not int or payload_bytes != cursor):
        raise InputsError("payload byte count mismatch")


def validate_payload(document: Mapping[str, Any], payload: bytes | bytearray | memoryview) -> None:
    """Check the packed sidecar and each record digest on CPU after terminal export."""
    validate_document(document, payload_bytes=len(payload))
    view = memoryview(payload)
    if hashlib.sha256(view).hexdigest() != document["payload_sha256"]:
        raise InputsError("aggregate payload digest mismatch")
    for name, _, _ in TENSOR_SPECS:
        record = document["tensors"][name]
        start = record["offset"]
        stop = start + record["nbytes"]
        if hashlib.sha256(view[start:stop]).hexdigest() != record["sha256"]:
            raise InputsError(f"tensor payload digest mismatch: {name}")


def validate_output_paths(output_arg: Path | str, *, allowed_root: Path = OUTPUT_ROOT) -> tuple[Path, Path]:
    """Admit a new JSON/payload pair beneath an existing output root."""
    output = Path(output_arg).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.suffix.lower() != ".json":
        raise InputsError("--output must have a .json suffix")
    try:
        root = allowed_root.resolve(strict=True)
        canonical_parent = output.parent.resolve(strict=True)
    except OSError:
        raise InputsError("output root and parent must already exist") from None
    if output.parent != canonical_parent or (canonical_parent != root and root not in canonical_parent.parents):
        raise InputsError("output parent must be canonical and beneath /opt/mldata_ssd")
    manifest = canonical_parent / output.name
    payload = canonical_parent / f"{output.stem}.tensors.bin"
    if os.path.lexists(manifest) or os.path.lexists(payload):
        raise InputsError("output manifest or sibling payload already exists")
    return manifest, payload


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qwen_source(q4_reference: Any, text_reference: Any) -> tuple[dict[str, Any], Path, tuple]:
    try:
        document, sidecar, reference_sha, sidecar_sha, reference_identity, sidecar_identity = (
            q4_reference._validate_qwen_source(QWEN_REFERENCE, QWEN_SIDECAR)
        )
    except Exception as error:
        raise InputsError(f"authenticated Qwen source rejected: {error}") from error
    prompt = document.get("hy_nomas_prompt")
    if (type(prompt) is not dict or prompt.get("prompt") != PROMPT
            or hashlib.sha256(PROMPT.encode("utf-8")).hexdigest() != QWEN_PROMPT_SHA256):
        raise InputsError("Qwen reference prompt does not match the fixed preview prompt")
    try:
        valid_indices = text_reference.validate_qwen_reference(
            document,
            reference_sha256=reference_sha,
            sidecar_bytes=sidecar.stat().st_size,
            sidecar_sha256=sidecar_sha,
        )
    except Exception as error:
        raise InputsError(f"Qwen crop/mask admission failed: {error}") from error
    if valid_indices != list(range(23)):
        raise InputsError("Qwen crop must contain the authenticated 23-row valid prefix")
    if (reference_sha != QWEN_REFERENCE_SHA256 or sidecar_sha != QWEN_SIDECAR_SHA256
            or sidecar.stat().st_size != QWEN_SIDECAR_BYTES):
        raise InputsError("Qwen source identity changed after authentication")
    return document, sidecar, (reference_identity, sidecar_identity)


def _verify_autoencoder_source(upstream: Path, pixel_reference: Any) -> tuple[Any, dict[str, str]]:
    try:
        expected_revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if expected_revision != PINNED_REVISION:
            raise InputsError("PixelUMM input helper revision mismatch")
        relative_model = "hymm/models/vae/pixelumm_decoder/model.py"
        relative_autoencoder = "hymm/models/vae/pixelumm_decoder/autoencoder.py"
        for relative in (relative_model, relative_autoencoder):
            status = subprocess.check_output(
                ["git", "-C", str(upstream), "status", "--porcelain", "--", relative],
                text=True, stderr=subprocess.DEVNULL,
            )
            if status.strip():
                raise InputsError(f"pinned PixelUMM source has local changes: {relative}")
            if _hash_file(upstream / relative) != PINNED_UPSTREAM_SOURCE_SHA256[relative]:
                raise InputsError(f"pinned PixelUMM source digest mismatch: {relative}")

        # Import only after both executable source files have been authenticated.
        source_module, model_sha = pixel_reference.load_upstream(upstream, verify_revision=True)
        if model_sha != PINNED_UPSTREAM_SOURCE_SHA256[relative_model]:
            raise InputsError("PixelUMM model.py digest mismatch")
        autoencoder_path = upstream / relative_autoencoder
        autoencoder_sha = PINNED_UPSTREAM_SOURCE_SHA256[relative_autoencoder]
        autoencoder_module = importlib.import_module(
            "hymm.models.vae.pixelumm_decoder.autoencoder"
        )
        if Path(autoencoder_module.__file__).resolve() != autoencoder_path.resolve():
            raise InputsError("Python imported a different PixelUMM autoencoder.py")
        return (source_module, autoencoder_module), {
            relative_model: model_sha,
            relative_autoencoder: autoencoder_sha,
        }
    except InputsError:
        raise
    except Exception as error:
        raise InputsError(f"pinned PixelUMM input helpers rejected: {error}") from error


def _read_qwen_crop_on_mps(torch: Any, np: Any, sidecar: Path, device: Any) -> Any:
    # Host use is limited to authenticated source-byte transport. State selection
    # and the requested crop itself are performed after moving state 26 to MPS.
    mapping = np.memmap(sidecar, mode="r", dtype="<u2", shape=(27, 1, 1388, 3584))
    state26_raw = np.array(mapping[26], copy=True)
    del mapping
    state26_cpu = torch.from_numpy(state26_raw).view(torch.bfloat16)
    del state26_raw
    state26 = state26_cpu.to(device=device, dtype=torch.bfloat16)
    del state26_cpu
    crop = state26[:, 108:1388, :].contiguous()
    del state26
    if crop.device.type != "mps" or crop.dtype is not torch.bfloat16 or tuple(crop.shape) != QWEN_SHAPE:
        raise InputsError("Qwen crop did not preserve MPS BF16 source layout")
    return crop


def _generate_tensors(torch: Any, helpers: tuple, pixel_modules: tuple, device: Any) -> dict[str, Any]:
    posemb_layers = helpers[-1]
    source_model, autoencoder = pixel_modules
    with torch.inference_mode():
        latent_generator = torch.Generator(device=device)
        latent_generator.manual_seed(LATENT_SEED)
        latent = torch.randn(
            LATENT_SHAPE, generator=latent_generator, device=device, dtype=torch.float32
        )
        rope_cos, rope_sin = posemb_layers.get_nd_rotary_pos_embed(
            [16, 56, 56], DIT_ROPE_GRID, theta=256.0, use_real=True,
            theta_rescale_factor=1, device=device,
        )

        # Call the exact pinned upstream method without constructing a decoder,
        # checkpoint, or model. _make_noise reads only self.seed and latent shape.
        pixel_noise = autoencoder.PixelUMMDCAELatentDecoder._make_noise(
            SimpleNamespace(seed=LATENT_SEED), latent, 0, LATENT_SHAPE[2]
        )

        pixel_cos, pixel_sin = source_model.mrope_tables(
            source_model.QWEN3_8B_GEN, PIXEL_ROPE_GRID, device
        )
        layout = source_model.local_attention_layout(
            PIXEL_ROPE_GRID, 8, device
        )
        if layout is None:
            raise InputsError("pinned local_attention_layout returned no tile permutation")
        permutation, inverse, block_mask = layout
        pixel_cos = pixel_cos[:, permutation]
        pixel_sin = pixel_sin[:, permutation]
        del inverse, block_mask, layout, permutation

    tensors = {
        "input.latent": latent,
        "input.rope_cos": rope_cos,
        "input.rope_sin": rope_sin,
        "decoder.noise": pixel_noise,
        "decoder.rope_cos": pixel_cos,
        "decoder.rope_sin": pixel_sin,
    }
    expected = {name: (dtype, shape) for name, dtype, shape in TENSOR_SPECS}
    for name, tensor in tensors.items():
        dtype, shape = expected[name]
        wanted_dtype = torch.float32 if dtype == "F32" else torch.bfloat16
        if (tensor.device.type != "mps" or tensor.dtype is not wanted_dtype
                or tuple(tensor.shape) != shape):
            raise InputsError(f"generated MPS tensor shape/dtype mismatch: {name}")
    return tensors


def _pack_mps_tensors(torch: Any, tensors: Mapping[str, Any]) -> bytes:
    expected_names = [name for name, _, _ in TENSOR_SPECS]
    if list(tensors) != expected_names:
        raise InputsError("MPS tensor mapping must match canonical inventory order")
    byte_views = []
    for name, dtype, shape in TENSOR_SPECS:
        tensor = tensors[name]
        expected_dtype = torch.float32 if dtype == "F32" else torch.bfloat16
        if (tensor.device.type != "mps" or tensor.dtype is not expected_dtype
                or tuple(tensor.shape) != shape):
            raise InputsError(f"cannot export non-MPS or malformed tensor: {name}")
        contiguous = tensor.contiguous()
        byte_views.append(contiguous.view(torch.uint8).reshape(-1))
    packed_mps = torch.cat(byte_views, dim=0)
    if packed_mps.device.type != "mps" or packed_mps.dtype is not torch.uint8:
        raise InputsError("packed tensor payload must remain an MPS byte tensor")
    # This is the sole MPS-to-host transfer and terminal synchronization.
    packed_cpu = packed_mps.cpu()
    return packed_cpu.numpy().tobytes()


def _records_from_payload(payload: bytes) -> dict[str, dict[str, Any]]:
    if len(payload) != expected_payload_bytes():
        raise InputsError("packed payload extent differs from canonical inventory")
    view = memoryview(payload)
    records: dict[str, dict[str, Any]] = {}
    cursor = 0
    for name, dtype, shape in TENSOR_SPECS:
        nbytes = tensor_nbytes(dtype, shape)
        records[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "offset": cursor,
            "nbytes": nbytes,
            "sha256": hashlib.sha256(view[cursor:cursor + nbytes]).hexdigest(),
        }
        cursor += nbytes
    return records


def _build_document(payload: bytes, payload_name: str, source_hashes: Mapping[str, str],
                    torch: Any) -> dict[str, Any]:
    records = _records_from_payload(payload)
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "payload": payload_name,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "tensors": records,
        "geometry": GEOMETRY,
        "prompt": PROMPT,
        "qwen_source": QWEN_SOURCE_RECORD,
        "noise_seeds": NOISE_SEEDS,
        "upstream_revision": PINNED_REVISION,
        "upstream_sources": dict(source_hashes),
        "producer": {
            **PRODUCER_RECORD,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
        },
    }
    validate_document(document, payload_bytes=len(payload))
    validate_payload(document, payload)
    return document


def _publish_noreplace(path: Path, data: bytes) -> None:
    """Publish one file using a fsynced temporary and atomic no-replace link."""
    if os.path.lexists(path):
        raise InputsError(f"output already exists: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_bundle(output: Path, payload_path: Path, document: dict[str, Any], payload: bytes) -> None:
    if os.path.lexists(output) or os.path.lexists(payload_path):
        raise InputsError("output manifest or sibling payload already exists")
    published_identity = None
    try:
        _publish_noreplace(payload_path, payload)
        stat = payload_path.stat()
        published_identity = (stat.st_dev, stat.st_ino)
        encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        _publish_noreplace(output, encoded)
    except Exception:
        if published_identity is not None and os.path.lexists(payload_path):
            stat = payload_path.stat()
            if (stat.st_dev, stat.st_ino) == published_identity:
                payload_path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new manifest path beneath /opt/mldata_ssd")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM,
                        help=f"pinned source checkout at {PINNED_REVISION}")
    return parser


def produce_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise InputsError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    output, payload_path = validate_output_paths(args.output)

    import numpy as np
    import torch

    try:
        import hy_nomas_q4_dit_reference as q4_reference
        import hy_nomas_text_conditioning_reference as text_reference
        import hy_nomas_pixelumm_reference as pixel_reference
        import hy_nomas_reference as dit_reference
    except Exception as error:
        raise InputsError(f"cannot import authenticated input helpers: {error}") from error

    qwen_document, qwen_sidecar, identities = _qwen_source(q4_reference, text_reference)
    upstream = Path(args.upstream).expanduser().resolve(strict=True)
    try:
        source_hashes = dit_reference._verify_upstream(upstream)
        helpers = dit_reference._install_upstream_modules(upstream)
        pixel_modules, pixel_hashes = _verify_autoencoder_source(upstream, pixel_reference)
    except InputsError:
        raise
    except Exception as error:
        raise InputsError(f"pinned upstream source admission failed: {error}") from error
    source_hashes = {
        "hymm/models/dit/modules/posemb_layers.py": source_hashes[
            "hymm/models/dit/modules/posemb_layers.py"
        ],
        **pixel_hashes,
    }
    if source_hashes != PINNED_UPSTREAM_SOURCE_SHA256:
        raise InputsError("verified upstream source inventory differs from the pinned input sources")

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise InputsError("MPS is required; CPU tensor fallback is forbidden")
    device = torch.device("mps")

    qwen = _read_qwen_crop_on_mps(torch, np, qwen_sidecar, device)
    if q4_reference._identity(qwen_sidecar) != identities[1]:
        raise InputsError("authenticated Qwen BF16 sidecar changed during MPS transfer")
    tensors = _generate_tensors(torch, helpers, pixel_modules, device)
    tensors["input.qwen_bf16"] = qwen
    ordered_tensors = {name: tensors[name] for name, _, _ in TENSOR_SPECS}
    payload = _pack_mps_tensors(torch, ordered_tensors)
    if q4_reference._identity(QWEN_REFERENCE) != identities[0]:
        raise InputsError("authenticated Qwen reference changed during input preparation")
    document = _build_document(payload, payload_path.name, source_hashes, torch)
    _publish_bundle(output, payload_path, document, payload)
    return {
        "manifest": str(output),
        "payload": str(payload_path),
        "payload_bytes": len(payload),
        "payload_sha256": document["payload_sha256"],
        "tensor_count": len(document["tensors"]),
        "upstream_revision": PINNED_REVISION,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = produce_inputs(args)
    except (InputsError, OSError, subprocess.SubprocessError) as error:
        print(f"generation-input preparation rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
