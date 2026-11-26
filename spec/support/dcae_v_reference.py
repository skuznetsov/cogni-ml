"""Upstream DCAE-V image encoder oracle and artifact comparison helpers.

This diagnostic path follows the immutable Hy-Nomas reference-image helper,
loads only the original encoder weights, and keeps the model out of the native
measurement path. Tensor captures are BF16 safetensors so Crystal's existing
``CheckpointFile`` can inspect them without a PyTorch dependency.
"""

from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SUPPORT = ROOT / "examples" / "support"
if str(EXAMPLE_SUPPORT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SUPPORT))

import dcae_v_checkpoint as checkpoint  # noqa: E402


SCHEMA = "hy-nomas-dcae-v-upstream-reference-v1"
UPSTREAM_COMMIT = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
CHECKPOINT_REVISION = "bb6a8ca451bfe84947f0ea1c6d51957f983ffffe"
CHECKPOINT_SHA256 = checkpoint.PINNED_SOURCE_SHA256
CHECKPOINT_BYTES = checkpoint.PINNED_SOURCE_BYTES
VARIANT = "dc-ae-v-1.0-f32t4c32-bf16"
SPATIAL_COMPRESSION = 32
TEMPORAL_COMPRESSION = 4
TILE_FRAMES = 40
STAGE_CAPTURE = "trace.stages.5"
INPUT_KEY = "input.clip"
LATENT_KEY = "latent"
ATOL = 2e-5
RTOL = 2e-5
MAX_ARTIFACT_BYTES = 2 * 1024**3
MAX_MANIFEST_BYTES = 4 * 1024**2


class ReferenceError(ValueError):
    """An upstream reference fixture or comparison violates its contract."""


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identity(path: Path) -> tuple[int, int, int, int]:
    if path.is_symlink():
        raise ReferenceError(f"symlink is not admitted: {path}")
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReferenceError(f"cannot stat {path}: {exc}") from None
    if not stat.S_ISREG(info.st_mode):
        raise ReferenceError(f"expected a regular file: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_product(shape: list[int]) -> int:
    if not isinstance(shape, list) or not shape:
        raise ReferenceError("tensor shape must be a nonempty list")
    result = 1
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise ReferenceError("tensor shape dimensions must be positive integers")
        result *= dimension
        if result > MAX_ARTIFACT_BYTES:
            raise ReferenceError("tensor shape exceeds the reference artifact bound")
    return result


def _capture_shapes(frames: int, height: int, width: int, tile_frames: int) -> dict[str, list[int]]:
    if any(type(value) is not int for value in (frames, height, width, tile_frames)):
        raise ReferenceError("reference geometry must use integer dimensions")
    if frames <= 0 or frames % TEMPORAL_COMPRESSION:
        raise ReferenceError("reference frame count must be positive and divisible by four")
    if height < SPATIAL_COMPRESSION or width < SPATIAL_COMPRESSION:
        raise ReferenceError("reference dimensions must be at least 32x32")
    if height % SPATIAL_COMPRESSION or width % SPATIAL_COMPRESSION:
        raise ReferenceError("reference dimensions must be multiples of 32")
    if tile_frames != TILE_FRAMES:
        raise ReferenceError("the pinned DCAE-V encoder requires 40-frame tiles")
    if tile_frames % TEMPORAL_COMPRESSION:
        raise ReferenceError("tile length must be divisible by the temporal factor")
    tile_count = math.ceil(frames / tile_frames)
    result = {INPUT_KEY: [1, 3, frames, height, width]}
    for tile in range(tile_count):
        tile_count_here = min(tile_frames, frames - tile * tile_frames)
        if tile_count_here % TEMPORAL_COMPRESSION:
            raise ReferenceError("the final DCAE-V tile must be divisible by four frames")
        result[f"{STAGE_CAPTURE}.tile{tile}"] = [
            1, 1024, tile_count_here // TEMPORAL_COMPRESSION,
            height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION,
        ]
    result[LATENT_KEY] = [
        1, 32, frames // TEMPORAL_COMPRESSION,
        height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION,
    ]
    return result


def fit_image_to_frame(image: Any, size: tuple[int, int]) -> Any:
    """Match ``hymm.utils.image_resize.fit_image_to_frame`` exactly."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ReferenceError(f"Pillow is required for image preparation: {exc}") from None
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(
        image,
        size,
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )


def reference_clip_from_image(
    image_path: Path,
    *,
    height: int,
    width: int,
    frames: int = 120,
    device: str = "cpu",
    dtype: Any | None = None,
) -> Any:
    """Create the upstream BF16 BCTHW tensor: fit, normalize, then repeat pixels."""
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ReferenceError(f"PyTorch, NumPy and Pillow are required: {exc}") from None
    if dtype is None:
        dtype = torch.bfloat16
    height = height // SPATIAL_COMPRESSION * SPATIAL_COMPRESSION
    width = width // SPATIAL_COMPRESSION * SPATIAL_COMPRESSION
    _capture_shapes(frames, height, width, TILE_FRAMES)
    image_path = Path(image_path).expanduser()
    _identity(image_path)
    with Image.open(image_path) as source:
        image = fit_image_to_frame(source, (width, height))
        pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    # Keep these in-place dtype boundaries and operation order: they mirror
    # the immutable upstream `encode_reference_image` helper.
    pixels = pixels.to(device=device, dtype=dtype).div_(127.5).sub_(1)
    clip = pixels[None, :, None].repeat(1, 1, frames, 1, 1)
    expected = (1, 3, frames, height, width)
    if tuple(clip.shape) != expected:
        raise ReferenceError(f"prepared input shape mismatch: {tuple(clip.shape)} != {expected}")
    return clip.contiguous()


def _load_module(path: Path) -> Any:
    path = path.resolve(strict=True)
    module_name = "_hy_nomas_dcae_v_reference_module"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReferenceError(f"cannot import upstream DCAE-V module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ReferenceError(f"cannot import upstream DCAE-V module: {exc}") from None
    return module


def _validate_checkpoint_source(
    path: Path, *, expected_sha256: str, expected_bytes: int
) -> tuple[tuple[int, int, int, int], str]:
    if not _valid_sha256(expected_sha256):
        raise ReferenceError("caller must provide the pinned checkpoint SHA-256")
    if expected_sha256 != CHECKPOINT_SHA256:
        raise ReferenceError("checkpoint SHA-256 differs from the pinned Hy-Nomas source")
    if type(expected_bytes) is not int or expected_bytes != CHECKPOINT_BYTES:
        raise ReferenceError("checkpoint byte count differs from the pinned Hy-Nomas source")
    path = Path(path).expanduser()
    identity = _identity(path)
    if identity[2] != expected_bytes:
        raise ReferenceError(f"checkpoint byte count mismatch: {identity[2]} != {expected_bytes}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ReferenceError("checkpoint SHA-256 differs from caller pin")
    if _identity(path) != identity:
        raise ReferenceError("checkpoint changed while its SHA-256 was computed")
    return identity, observed_sha256


def _load_encoder_from_original_checkpoint(
    checkpoint_path: Path,
    *,
    expected_shapes: dict[str, list[int]],
    module_path: Path,
    device: str,
) -> tuple[Any, Any, Any, str]:
    """Instantiate upstream model with only its encoder materialized on MPS."""
    try:
        import torch
    except ImportError as exc:
        raise ReferenceError(f"PyTorch is required for the upstream oracle: {exc}") from None
    if device != "mps":
        raise ReferenceError("the DCAE-V upstream oracle only admits the MPS device")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled for the oracle")
    if not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is unavailable; no CPU fallback is admitted")
    module_path = Path(module_path).expanduser()
    module_identity = _identity(module_path)
    module_sha256 = sha256_file(module_path)
    upstream = _load_module(module_path)
    config = upstream.dc_ae_v_f32t4c32_config()
    with torch.device("meta"):
        model = upstream.DCAEV(config)
    model.encoder.to(dtype=torch.bfloat16)
    model.encoder.to_empty(device=device)
    try:
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise ReferenceError(f"safe memory-mapped checkpoint load failed: {exc}") from None
    state, _selection = checkpoint._select_state_dict(checkpoint_payload)
    encoder_state = checkpoint.validate_encoder_state(state, expected_shapes, torch)
    local_names = {name.removeprefix("encoder."): value for name, value in encoder_state.items()}
    try:
        model.encoder.load_state_dict(local_names, strict=True)
    except Exception as exc:
        raise ReferenceError(f"upstream encoder rejected the original checkpoint: {exc}") from None
    del encoder_state, local_names, state, checkpoint_payload
    if _identity(module_path) != module_identity or sha256_file(module_path) != module_sha256:
        raise ReferenceError("upstream DCAE-V source changed while the oracle was initialized")
    model.eval()
    return torch, model, config, module_sha256


def _write_safetensors_atomic(
    output_path: Path, tensors: dict[str, Any], metadata: dict[str, str]
) -> tuple[int, str]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ReferenceError(f"safetensors is required for the oracle artifact: {exc}") from None
    output_path = Path(output_path).expanduser()
    parent = output_path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ReferenceError("oracle artifact parent must be an existing directory")
    output_path = parent / output_path.name
    if output_path.exists() or output_path.is_symlink():
        raise ReferenceError(f"refusing to overwrite oracle artifact: {output_path}")
    fd, name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".partial", dir=parent)
    os.close(fd)
    temporary = Path(name)
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        size = _identity(temporary)[2]
        if size <= 0 or size > MAX_ARTIFACT_BYTES:
            raise ReferenceError(f"oracle artifact size is outside its bound: {size}")
        digest = sha256_file(temporary)
        with safe_open(str(temporary), framework="pt", device="cpu") as archive:
            if set(archive.keys()) != set(tensors):
                raise ReferenceError("oracle artifact tensor inventory changed during serialization")
        if output_path.exists():
            raise ReferenceError("oracle artifact path appeared during serialization")
        os.replace(temporary, output_path)
        return size, digest
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    parent = path.parent.resolve(strict=True)
    path = parent / path.name
    if path.exists() or path.is_symlink():
        raise ReferenceError(f"refusing to overwrite manifest: {path}")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ReferenceError("oracle manifest exceeds its size bound")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise ReferenceError("manifest path appeared during serialization")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_reference_manifest(
    *,
    image_path: Path,
    image_sha256: str,
    image_bytes: int,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    checkpoint_revision: str,
    module_sha256: str,
    output_path: Path,
    output_bytes: int,
    output_sha256: str,
    frames: int,
    height: int,
    width: int,
    tile_frames: int = TILE_FRAMES,
    torch_version: str,
) -> dict[str, Any]:
    shapes = _capture_shapes(frames, height, width, tile_frames)
    if checkpoint_revision != CHECKPOINT_REVISION:
        raise ReferenceError("checkpoint revision differs from the pinned Hy-Nomas source")
    if checkpoint_sha256 != CHECKPOINT_SHA256 or checkpoint_bytes != CHECKPOINT_BYTES:
        raise ReferenceError("checkpoint provenance differs from the pinned Hy-Nomas source")
    if not all(_valid_sha256(digest) for digest in (image_sha256, checkpoint_sha256, module_sha256, output_sha256)):
        raise ReferenceError("manifest requires valid observed SHA-256 digests")
    return {
        "schema": SCHEMA,
        "upstream": {
            "repository": "Tencent-Hunyuan/Hy-Nomas",
            "reference_image_revision": UPSTREAM_COMMIT,
            "vae_source_sha256": module_sha256,
            "reference_image_helper": "hymm.pipeline.reference_image.encode_reference_image",
            "resize_helper": "hymm.utils.image_resize.fit_image_to_frame",
        },
        "checkpoint": {
            "repository_revision": checkpoint_revision,
            "variant": VARIANT,
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
            "load_policy": "torch.load(weights_only=True,mmap=True,map_location=cpu)",
            "encoder_only_materialization": True,
        },
        "input": {
            "image_filename": Path(image_path).name,
            "image_bytes": image_bytes,
            "image_sha256": image_sha256,
            "fit": {
                "exif_transpose": True,
                "mode": "RGB",
                "method": "PIL.ImageOps.pad",
                "resampling": "LANCZOS",
                "centering": [0.5, 0.5],
                "padding_rgb": [0, 0, 0],
            },
            "height": height,
            "width": width,
            "frames": frames,
            "fps": 24,
            "dtype": "BF16",
            "normalization": "uint8 BF16 pixels: div_(127.5), then sub_(1), in upstream order",
            "repeat": "identical pixel frame repeated in pixel space before DCAE-V encode",
            "tile_frames": tile_frames,
            "temporal_feature_cache": True,
        },
        "execution": {
            "device": "mps",
            "torch_version": torch_version,
            "dtype": "BF16",
            "cpu_fallback_enabled": False,
            "latent_postprocess": "identity: upstream _LatentDist.mode returns the encoder output",
            "captured_boundaries": [STAGE_CAPTURE, LATENT_KEY],
            "capture_semantics": "diagnostic hooks read back stage5 once per upstream 40-frame tile",
        },
        "comparison": {
            "atol": ATOL,
            "rtol": RTOL,
            "input_key": INPUT_KEY,
            "tensor_keys": list(shapes),
        },
        "artifact": {
            "filename": Path(output_path).name,
            "bytes": output_bytes,
            "sha256": output_sha256,
            "tensors": {
                name: {"shape": shape, "dtype": "BF16"}
                for name, shape in shapes.items()
            },
        },
    }


def generate_upstream_reference(
    *,
    image_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path,
    height: int,
    width: int,
    expected_checkpoint_sha256: str,
    expected_checkpoint_bytes: int,
    checkpoint_revision: str = CHECKPOINT_REVISION,
    frames: int = 120,
    tile_frames: int = TILE_FRAMES,
    device: str = "mps",
    module_path: Path,
) -> dict[str, Any]:
    """Run the pinned upstream encoder and emit the diagnostic capture artifact."""
    height = height // SPATIAL_COMPRESSION * SPATIAL_COMPRESSION
    width = width // SPATIAL_COMPRESSION * SPATIAL_COMPRESSION
    shapes = _capture_shapes(frames, height, width, tile_frames)
    if checkpoint_revision != CHECKPOINT_REVISION:
        raise ReferenceError("checkpoint revision differs from the pinned Hy-Nomas source")
    module_path = Path(module_path).expanduser()
    module_identity = _identity(module_path)
    image_path = Path(image_path).expanduser()
    image_identity = _identity(image_path)
    image_sha256 = sha256_file(image_path)
    if _identity(image_path) != image_identity:
        raise ReferenceError("image changed while its SHA-256 was computed")
    checkpoint_identity, checkpoint_sha256 = _validate_checkpoint_source(
        Path(checkpoint_path),
        expected_sha256=expected_checkpoint_sha256,
        expected_bytes=expected_checkpoint_bytes,
    )
    expected_shapes, upstream_metadata = checkpoint.upstream_encoder_shapes(module_path)
    torch, model, config, module_sha256 = _load_encoder_from_original_checkpoint(
        Path(checkpoint_path),
        expected_shapes=expected_shapes,
        module_path=module_path,
        device=device,
    )
    if module_sha256 != upstream_metadata["module_sha256"]:
        raise ReferenceError("DCAE-V module changed between schema and model initialization")
    clip = reference_clip_from_image(
        image_path,
        height=height,
        width=width,
        frames=frames,
        device=device,
        dtype=torch.bfloat16,
    )
    stage_outputs: list[Any] = []

    def capture_stage5(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise ReferenceError("upstream stage 5 did not return a tensor")
        stage_outputs.append(output.detach().contiguous().cpu())

    hook = model.encoder.stages[5].register_forward_hook(capture_stage5)
    try:
        with torch.no_grad():
            latent = model.encode(clip)
    finally:
        hook.remove()
    expected_names = set(shapes)
    if tuple(latent.shape) != tuple(shapes[LATENT_KEY]) or latent.dtype is not torch.bfloat16:
        raise ReferenceError(
            f"upstream latent mismatch: shape={tuple(latent.shape)} dtype={latent.dtype}"
        )
    tile_count = math.ceil(frames / tile_frames)
    if len(stage_outputs) != tile_count:
        raise ReferenceError(f"stage5 capture count mismatch: {len(stage_outputs)} != {tile_count}")
    tensors = {
        INPUT_KEY: clip.detach().contiguous().cpu(),
        LATENT_KEY: latent.detach().contiguous().cpu(),
    }
    for index, value in enumerate(stage_outputs):
        name = f"{STAGE_CAPTURE}.tile{index}"
        if name not in expected_names or tuple(value.shape) != tuple(shapes[name]) or value.dtype is not torch.bfloat16:
            raise ReferenceError(f"upstream stage5 capture does not match schema: {name}")
        tensors[name] = value
    if set(tensors) != expected_names:
        raise ReferenceError("upstream output tensor set differs from capture contract")
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise ReferenceError(f"upstream reference contains non-finite values: {name}")
    if _identity(image_path) != image_identity or sha256_file(image_path) != image_sha256:
        raise ReferenceError("image changed while the upstream reference was generated")
    if _identity(Path(checkpoint_path)) != checkpoint_identity:
        raise ReferenceError("checkpoint changed while the upstream reference was generated")
    if _identity(module_path) != module_identity or sha256_file(module_path) != module_sha256:
        raise ReferenceError("upstream DCAE-V source changed while the reference was generated")
    artifact_bytes, artifact_sha256 = _write_safetensors_atomic(
        output_path,
        tensors,
        {
            "format": "pt",
            "reference_schema": SCHEMA,
            "upstream_revision": UPSTREAM_COMMIT,
            "checkpoint_sha256": checkpoint_sha256,
            "input_sha256": image_sha256,
        },
    )
    try:
        manifest = build_reference_manifest(
            image_path=image_path,
            image_sha256=image_sha256,
            image_bytes=image_identity[2],
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_bytes=checkpoint_identity[2],
            checkpoint_revision=checkpoint_revision,
            module_sha256=module_sha256,
            output_path=output_path,
            output_bytes=artifact_bytes,
            output_sha256=artifact_sha256,
            frames=frames,
            height=height,
            width=width,
            tile_frames=tile_frames,
            torch_version=torch.__version__,
        )
        _write_json_atomic(manifest_path, manifest)
    except BaseException:
        Path(output_path).unlink(missing_ok=True)
        raise
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    identity = _identity(path)
    if identity[2] <= 0 or identity[2] > MAX_MANIFEST_BYTES:
        raise ReferenceError("reference manifest size is outside its bound")
    raw = path.read_bytes()
    if len(raw) != identity[2] or _identity(path) != identity:
        raise ReferenceError("reference manifest changed while being read")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReferenceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=lambda _: (_ for _ in ()).throw(ReferenceError("non-finite JSON number")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceError(f"reference manifest is not valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ReferenceError("reference manifest root must be an object")
    return value


def validate_manifest_document(document: dict[str, Any], *, artifact_path: Path) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ReferenceError("reference manifest schema mismatch")
    upstream = document.get("upstream")
    source = document.get("checkpoint")
    inputs = document.get("input")
    execution = document.get("execution")
    comparison = document.get("comparison")
    artifact = document.get("artifact")
    if not all(isinstance(value, dict) for value in (upstream, source, inputs, execution, comparison, artifact)):
        raise ReferenceError("reference manifest sections are missing")
    if upstream.get("repository") != "Tencent-Hunyuan/Hy-Nomas" or upstream.get("reference_image_revision") != UPSTREAM_COMMIT:
        raise ReferenceError("reference-image upstream revision mismatch")
    if upstream.get("reference_image_helper") != "hymm.pipeline.reference_image.encode_reference_image":
        raise ReferenceError("reference-image helper provenance mismatch")
    if upstream.get("resize_helper") != "hymm.utils.image_resize.fit_image_to_frame":
        raise ReferenceError("reference-image resize provenance mismatch")
    if not _valid_sha256(upstream.get("vae_source_sha256")):
        raise ReferenceError("upstream VAE source digest is invalid")
    if source.get("repository_revision") != CHECKPOINT_REVISION or source.get("variant") != VARIANT:
        raise ReferenceError("DCAE-V checkpoint revision or variant mismatch")
    if not _valid_sha256(source.get("sha256")) or type(source.get("bytes")) is not int or source["bytes"] <= 0:
        raise ReferenceError("DCAE-V checkpoint provenance is invalid")
    if source.get("sha256") != CHECKPOINT_SHA256 or source.get("bytes") != CHECKPOINT_BYTES:
        raise ReferenceError("manifest checkpoint digest or size differs from the pinned source")
    if source.get("load_policy") != "torch.load(weights_only=True,mmap=True,map_location=cpu)" or source.get("encoder_only_materialization") is not True:
        raise ReferenceError("DCAE-V checkpoint load policy mismatch")
    for key in ("image_sha256",):
        if not _valid_sha256(inputs.get(key)):
            raise ReferenceError(f"reference input {key} is invalid")
    if type(inputs.get("image_bytes")) is not int or inputs["image_bytes"] <= 0:
        raise ReferenceError("reference input image size is invalid")
    fit = inputs.get("fit")
    if fit != {
        "exif_transpose": True,
        "mode": "RGB",
        "method": "PIL.ImageOps.pad",
        "resampling": "LANCZOS",
        "centering": [0.5, 0.5],
        "padding_rgb": [0, 0, 0],
    }:
        raise ReferenceError("reference image fit parameters mismatch")
    height, width, frames, tile_frames = (
        inputs.get("height"), inputs.get("width"), inputs.get("frames"), inputs.get("tile_frames")
    )
    shapes = _capture_shapes(frames, height, width, tile_frames)
    if inputs.get("fps") != 24 or inputs.get("dtype") != "BF16":
        raise ReferenceError("reference input media or dtype contract mismatch")
    if inputs.get("normalization") != "uint8 BF16 pixels: div_(127.5), then sub_(1), in upstream order":
        raise ReferenceError("reference input normalization mismatch")
    if inputs.get("repeat") != "identical pixel frame repeated in pixel space before DCAE-V encode" or inputs.get("temporal_feature_cache") is not True:
        raise ReferenceError("reference image temporal encoding contract mismatch")
    if (
        execution.get("device") != "mps"
        or execution.get("dtype") != "BF16"
        or execution.get("cpu_fallback_enabled") is not False
        or execution.get("latent_postprocess") != "identity: upstream _LatentDist.mode returns the encoder output"
    ):
        raise ReferenceError("reference execution must be the upstream MPS BF16 path")
    if not isinstance(execution.get("torch_version"), str) or not execution["torch_version"]:
        raise ReferenceError("reference PyTorch version provenance is missing")
    if execution.get("captured_boundaries") != [STAGE_CAPTURE, LATENT_KEY]:
        raise ReferenceError("reference capture boundaries mismatch")
    if comparison.get("atol") != ATOL or comparison.get("rtol") != RTOL:
        raise ReferenceError("reference comparison tolerance mismatch")
    if comparison.get("input_key") != INPUT_KEY or comparison.get("tensor_keys") != list(shapes):
        raise ReferenceError("reference comparison key contract mismatch")
    artifact_path = Path(artifact_path)
    if artifact.get("filename") != artifact_path.name:
        raise ReferenceError("reference artifact filename mismatch")
    if type(artifact.get("bytes")) is not int or artifact["bytes"] <= 0 or artifact["bytes"] > MAX_ARTIFACT_BYTES:
        raise ReferenceError("reference artifact byte count is outside its bound")
    if not _valid_sha256(artifact.get("sha256")):
        raise ReferenceError("reference artifact digest is invalid")
    expected_records = {name: {"shape": shape, "dtype": "BF16"} for name, shape in shapes.items()}
    if artifact.get("tensors") != expected_records:
        raise ReferenceError("reference artifact tensor manifest differs from the pinned shape contract")
    return document


def load_reference_artifact(weights_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    weights_path = Path(weights_path).expanduser()
    manifest_path = Path(manifest_path).expanduser()
    weights_identity = _identity(weights_path)
    if weights_identity[2] <= 0 or weights_identity[2] > MAX_ARTIFACT_BYTES:
        raise ReferenceError("reference artifact size is outside its bound")
    manifest = validate_manifest_document(_load_json(manifest_path), artifact_path=weights_path)
    if weights_identity[2] != manifest["artifact"]["bytes"]:
        raise ReferenceError("reference artifact byte count differs from manifest")
    if sha256_file(weights_path) != manifest["artifact"]["sha256"]:
        raise ReferenceError("reference artifact SHA-256 differs from manifest")
    if _identity(weights_path) != weights_identity:
        raise ReferenceError("reference artifact changed while it was authenticated")
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise ReferenceError(f"PyTorch and safetensors are required: {exc}") from None
    tensors: dict[str, Any] = {}
    try:
        with safe_open(str(weights_path), framework="pt", device="cpu") as archive:
            records = manifest["artifact"]["tensors"]
            if set(archive.keys()) != set(records):
                raise ReferenceError("reference safetensors tensor names differ from manifest")
            for name, record in records.items():
                tensor = archive.get_tensor(name)
                if tensor.dtype is not torch.bfloat16 or list(tensor.shape) != record["shape"]:
                    raise ReferenceError(f"reference safetensors tensor shape or dtype mismatch: {name}")
                if not bool(torch.isfinite(tensor).all()):
                    raise ReferenceError(f"reference safetensors tensor has non-finite values: {name}")
                tensors[name] = tensor
    except ReferenceError:
        raise
    except Exception as exc:
        raise ReferenceError(f"cannot read reference safetensors header or payload: {exc}") from None
    if _identity(weights_path) != weights_identity:
        raise ReferenceError("reference artifact changed while it was read")
    return manifest, tensors


def compare_tensor(reference: Any, candidate: Any, *, name: str, exact: bool = False) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ReferenceError(f"PyTorch is required for tensor comparison: {exc}") from None
    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise ReferenceError(f"comparison values must be tensors: {name}")
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ReferenceError(f"comparison tensor shape mismatch for {name}: {tuple(candidate.shape)} != {tuple(reference.shape)}")
    if reference.dtype is not torch.bfloat16 or candidate.dtype is not torch.bfloat16:
        raise ReferenceError(f"comparison tensor dtype must be BF16 for {name}")
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(candidate).all()):
        raise ReferenceError(f"comparison tensor contains non-finite values: {name}")
    if exact:
        # Compare the BF16 payload words so signed zero is not treated as equal.
        mismatches = int(torch.count_nonzero(reference.view(torch.int16) != candidate.view(torch.int16)).item())
        return {
            "name": name,
            "mode": "bitwise-value-equality",
            "elements": reference.numel(),
            "mismatches": mismatches,
            "max_abs_error": 0.0 if mismatches == 0 else float((reference.float() - candidate.float()).abs().max().item()),
            "passed": mismatches == 0,
        }
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    absolute_error = (candidate_f32 - reference_f32).abs()
    allowed = ATOL + RTOL * reference_f32.abs()
    failed = absolute_error > allowed
    mismatches = int(torch.count_nonzero(failed).item())
    return {
        "name": name,
        "mode": "allclose",
        "atol": ATOL,
        "rtol": RTOL,
        "elements": reference.numel(),
        "mismatches": mismatches,
        "max_abs_error": float(absolute_error.max().item()) if reference.numel() else 0.0,
        "max_scaled_error": float((absolute_error / allowed).max().item()) if reference.numel() else 0.0,
        "passed": mismatches == 0,
    }


def compare_artifacts(
    reference_path: Path,
    reference_manifest_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    manifest, reference = load_reference_artifact(reference_path, reference_manifest_path)
    candidate_path = Path(candidate_path).expanduser()
    candidate_identity = _identity(candidate_path)
    if candidate_identity[2] <= 0 or candidate_identity[2] > MAX_ARTIFACT_BYTES:
        raise ReferenceError("native candidate artifact size is outside its bound")
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise ReferenceError(f"PyTorch and safetensors are required: {exc}") from None
    candidate: dict[str, Any] = {}
    try:
        with safe_open(str(candidate_path), framework="pt", device="cpu") as archive:
            if set(archive.keys()) != set(reference):
                raise ReferenceError("native candidate tensor names differ from the upstream reference")
            for name in reference:
                value = archive.get_tensor(name)
                if value.dtype is not torch.bfloat16 or tuple(value.shape) != tuple(reference[name].shape):
                    raise ReferenceError(f"native candidate shape or dtype mismatch: {name}")
                candidate[name] = value
    except ReferenceError:
        raise
    except Exception as exc:
        raise ReferenceError(f"cannot read native candidate safetensors: {exc}") from None
    if _identity(candidate_path) != candidate_identity:
        raise ReferenceError("native candidate artifact changed while it was read")
    candidate_sha256 = sha256_file(candidate_path)
    if _identity(candidate_path) != candidate_identity:
        raise ReferenceError("native candidate artifact changed while it was hashed")
    comparisons = []
    for name in manifest["comparison"]["tensor_keys"]:
        comparisons.append(compare_tensor(reference[name], candidate[name], name=name, exact=(name == INPUT_KEY)))
    return {
        "schema": "hy-nomas-dcae-v-comparison-v1",
        "upstream_reference_sha256": manifest["artifact"]["sha256"],
        "candidate_filename": candidate_path.name,
        "candidate_bytes": candidate_identity[2],
        "candidate_sha256": candidate_sha256,
        "atol": ATOL,
        "rtol": RTOL,
        "comparisons": comparisons,
        "all_passed": all(record["passed"] for record in comparisons),
    }


def write_comparison_report(path: Path, report: dict[str, Any]) -> None:
    _write_json_atomic(path, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="run the upstream BF16 MPS DCAE-V oracle")
    generate.add_argument("--image", required=True, type=Path)
    generate.add_argument("--checkpoint", required=True, type=Path)
    generate.add_argument("--checkpoint-sha256", required=True)
    generate.add_argument("--checkpoint-bytes", required=True, type=int)
    generate.add_argument("--checkpoint-revision", default=CHECKPOINT_REVISION)
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--manifest", required=True, type=Path)
    generate.add_argument("--height", required=True, type=int)
    generate.add_argument("--width", required=True, type=int)
    generate.add_argument("--frames", default=120, type=int)
    generate.add_argument("--tile-frames", default=TILE_FRAMES, type=int)
    generate.add_argument("--device", default="mps", choices=("mps",))
    generate.add_argument("--upstream-module", required=True, type=Path)
    compare = commands.add_parser("compare", help="compare a native BF16 capture with the upstream oracle")
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--manifest", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    exit_code = 0
    try:
        if args.command == "generate":
            manifest = generate_upstream_reference(
                image_path=args.image,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                manifest_path=args.manifest,
                height=args.height,
                width=args.width,
                expected_checkpoint_sha256=args.checkpoint_sha256,
                expected_checkpoint_bytes=args.checkpoint_bytes,
                checkpoint_revision=args.checkpoint_revision,
                frames=args.frames,
                tile_frames=args.tile_frames,
                device=args.device,
                module_path=args.upstream_module,
            )
            result = {
                "schema": manifest["schema"],
                "artifact": manifest["artifact"],
                "manifest_path": str(args.manifest),
            }
        else:
            result = compare_artifacts(args.reference, args.manifest, args.candidate)
            write_comparison_report(args.report, result)
            if not result["all_passed"]:
                exit_code = 1
            result = {**result, "report_path": str(args.report)}
    except (ReferenceError, checkpoint.CheckpointError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
