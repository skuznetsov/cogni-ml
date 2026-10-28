#!/usr/bin/env python3
"""Produce a pinned MPS oracle for the native PixelUMM stem and output assembly.

The oracle covers only the source stem at its first window and the source
temporal merger/display conversion over five deterministic BF16 windows. It
does not load decoder blocks or the output head.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import mmap
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "spec" / "support") not in sys.path:
    sys.path.insert(0, str(ROOT / "spec" / "support"))

import hy_nomas_generation_inputs as generation_inputs  # noqa: E402


SCHEMA = "cogni_ml.pixelumm_generation_reference/v1"
SCHEMA_VERSION = 1
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = generation_inputs.DEFAULT_UPSTREAM
DEFAULT_STEM = Path(
    "/Volumes/Nata's Data/MLData/cogni-ml/hy-nomas-pixelumm-20261026/source/stem-f32.safetensors"
)
STEM_SHA256 = "7eb73ebf42f6681e21870dc2d2d4b592846ba15ed5bf5f9a6c7a83d39c2d8d04"
MODEL_PATH = "hymm/models/vae/pixelumm_decoder/model.py"
STREAMING_PATH = "hymm/models/vae/pixelumm_decoder/streaming.py"
POSTPROCESS_PATH = "hymm/pipeline/base.py"
SOURCE_SHA256 = {
    MODEL_PATH: "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec",
    STREAMING_PATH: "ae88a4ad1903ac5db3c96d2e171cead07279f60a89180dae8cd83ed45da63c7b",
    POSTPROCESS_PATH: "77a039514ff494cb496a194c60ca3270f09c15b194bda25448a036a6fcc79582",
}
TENSORS = (
    ("stem.output_bf16", "BF16", (1, 5760, 4096)),
    ("assembly.output_f32", "F32", (1, 3, 120, 288, 512)),
)
STEM_WEIGHTS = {
    "decoder.embed.weight": (4096, 3072),
    "decoder.embed.bias": (4096,),
    "decoder.cond_proj.weight": (16384, 32),
    "decoder.cond_proj.bias": (16384,),
}
STEM_WINDOW_START = 0
WINDOW_STARTS = (0, 5, 10, 15, 20)
WINDOW_FRAMES = 40
PIXEL_FRAMES = 120
PIXEL_HEIGHT = 288
PIXEL_WIDTH = 512
ATOL = RTOL = 2e-5
OUTPUT_ROOT = Path("/opt/mldata_ssd")
_DTYPE_WIDTH = {"BF16": 2, "F32": 4}


class ReferenceError(RuntimeError):
    """Input, source, weight, device, or output admission failed."""


def tensor_nbytes(dtype: str, shape: tuple[int, ...]) -> int:
    return _DTYPE_WIDTH[dtype] * _product(shape)


def _product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def synthetic_value(index: int, window_index: int) -> float:
    """F32 value before the required RNE BF16 cast; index is local CTHW order."""
    return (((index * 37 + window_index * 911) % 8192) - 4096) / 1024.0


def synthetic_window(torch: Any, window_index: int, device: Any):
    count = 3 * WINDOW_FRAMES * PIXEL_HEIGHT * PIXEL_WIDTH
    # The period is exactly 8192. Keep MPS integer operations on this small
    # range; applying int64 remainder after multiplying the full flattened
    # index loses low bits on MPS for indices above 2**24.
    period = torch.arange(8192, device=device, dtype=torch.int64)
    values = torch.remainder(period * 37 + window_index * 911, 8192) - 4096
    values = (values.to(torch.float32) / 1024.0).to(torch.bfloat16)
    values = values.repeat(count // 8192)
    return values.reshape(1, 3, WINDOW_FRAMES, PIXEL_HEIGHT, PIXEL_WIDTH)


def postprocess_bf16(torch: Any, video: Any):
    """Match the pinned source's BF16 multiply/add/clamp boundary, then expand."""
    return video.mul(0.5).add(0.5).clamp(0, 1).float()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ReferenceError(f"{label} must not be a symlink")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ReferenceError(f"{label} does not exist") from error
    if not resolved.is_file() or resolved != path.expanduser().absolute():
        raise ReferenceError(f"{label} must be a canonical absolute regular file")
    return resolved


def _validate_input_bundle(manifest_path: Path, expected_sha256: str) -> tuple[dict, Path]:
    manifest_path = _canonical_file(manifest_path, "generation input manifest")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ReferenceError("generation input manifest exceeds the 1 MiB bound")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ReferenceError("generation input manifest SHA-256 mismatch")
    try:
        document = json.loads(raw)
        payload_name = document["payload"]
        generation_inputs.validate_document(document)
    except (ValueError, KeyError, TypeError) as error:
        raise ReferenceError("generation input manifest is malformed") from error
    except Exception as error:
        raise ReferenceError(f"generation input manifest rejected: {error}") from error
    payload_path = manifest_path.parent / payload_name
    if payload_path.is_symlink() or not payload_path.is_file() or payload_path.resolve() != payload_path:
        raise ReferenceError("generation input payload must be a canonical regular sibling")
    try:
        generation_inputs.validate_document(document, payload_bytes=payload_path.stat().st_size)
    except Exception as error:
        raise ReferenceError(f"generation input manifest rejected: {error}") from error
    # Authenticate every payload byte and tensor record before creating GPU
    # inputs. The strict producer manifest fixes contiguous ranges in order.
    aggregate = hashlib.sha256()
    records = sorted(document["tensors"].items(), key=lambda item: item[1]["offset"])
    with payload_path.open("rb") as stream:
        for name, record in records:
            digest = hashlib.sha256()
            remaining = record["nbytes"]
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ReferenceError(f"generation payload is truncated at {name}")
                aggregate.update(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ReferenceError(f"generation tensor digest mismatch: {name}")
    if aggregate.hexdigest() != document["payload_sha256"]:
        raise ReferenceError("generation aggregate payload SHA-256 mismatch")
    return document, payload_path


def _load_input_tensor(torch: Any, payload_path: Path, record: Mapping[str, Any], device: Any):
    dtype = torch.float32 if record["dtype"] == "F32" else torch.bfloat16
    stream = payload_path.open("rb")
    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    # Keep the stream alive for platforms that tie mmap lifetime to the fd.
    # ACCESS_COPY prevents writes to the authenticated source file.
    try:
        byte_view = torch.frombuffer(mapping, dtype=torch.uint8)
        raw = byte_view.narrow(0, record["offset"], record["nbytes"])
        cpu = raw.view(dtype).reshape(record["shape"])
        result = cpu.to(device=device, copy=True)
        del cpu, raw, byte_view
        return result
    finally:
        mapping.close()
        stream.close()


def _verify_upstream_files(upstream: Path) -> Path:
    upstream = upstream.expanduser().resolve(strict=True)
    if not upstream.is_dir():
        raise ReferenceError("upstream source root must be a directory")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if revision != PINNED_REVISION:
            raise ReferenceError("upstream revision differs from the pinned source")
        for relative, expected in SOURCE_SHA256.items():
            status = subprocess.check_output(
                ["git", "-C", str(upstream), "status", "--porcelain", "--", relative],
                text=True, stderr=subprocess.DEVNULL,
            )
            path = upstream / relative
            if status.strip() or _sha_file(path) != expected:
                raise ReferenceError(f"pinned upstream source changed: {relative}")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReferenceError("cannot authenticate the pinned upstream checkout") from error
    return upstream


def _verify_upstream(upstream: Path) -> tuple[Any, Any]:
    import hy_nomas_pixelumm_reference as pixelumm_reference

    upstream = _verify_upstream_files(upstream)

    source, model_sha = pixelumm_reference.load_upstream(upstream, verify_revision=True)
    if model_sha != SOURCE_SHA256[MODEL_PATH]:
        raise ReferenceError("PixelUMM model source identity mismatch")
    streaming = importlib.import_module("hymm.models.vae.pixelumm_decoder.streaming")
    imported = Path(streaming.__file__).resolve()
    if imported != (upstream / STREAMING_PATH).resolve():
        raise ReferenceError("Python imported a different PixelUMM streaming module")
    return source, streaming


def _load_stem_weights(torch: Any, stem_path: Path, expected_sha256: str, device: Any) -> dict:
    stem_path = _canonical_file(stem_path, "stem checkpoint")
    if _sha_file(stem_path) != expected_sha256:
        raise ReferenceError("stem checkpoint SHA-256 mismatch")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required to read the pinned stem subset") from error
    result = {}
    with safe_open(str(stem_path), framework="pt", device="cpu") as checkpoint:
        if set(checkpoint.keys()) != set(STEM_WEIGHTS):
            raise ReferenceError("stem checkpoint tensor inventory mismatch")
        for name, shape in STEM_WEIGHTS.items():
            tensor = checkpoint.get_tensor(name)
            if tuple(tensor.shape) != shape or tensor.dtype is not torch.float32:
                raise ReferenceError(f"stem checkpoint dtype/shape mismatch: {name}")
            result[name] = tensor.to(device=device, dtype=torch.bfloat16).contiguous()
            del tensor
    return result


def _reference_stem(torch: Any, source: Any, plan: Any, latent: Any,
                    noise: Any, weights: Mapping[str, Any]):
    import torch.nn.functional as functional

    device = latent.device
    permutation = torch.tensor(plan.permutation, device=device, dtype=torch.long)
    start = STEM_WINDOW_START
    xt = noise.narrow(2, start * 4, WINDOW_FRAMES).to(torch.bfloat16)
    latent_window = latent.narrow(2, start, 10).to(torch.bfloat16)
    with torch.inference_mode(), torch.autocast(device_type="mps", dtype=torch.bfloat16):
        patches = source.patchify(xt)
        embedded = functional.linear(
            patches, weights["decoder.embed.weight"], weights["decoder.embed.bias"]
        )
        condition_input = latent_window.permute(0, 2, 3, 4, 1)
        condition = functional.linear(
            condition_input,
            weights["decoder.cond_proj.weight"],
            weights["decoder.cond_proj.bias"],
        )
        condition = condition.view(1, 10, 9, 16, 2, 2, 4096)
        condition = condition.permute(0, 1, 2, 4, 3, 5, 6).reshape(1, 5760, 4096)
        tokens = (embedded + condition).index_select(1, permutation).contiguous()
    if tokens.device.type != "mps" or tokens.dtype is not torch.bfloat16 or tuple(tokens.shape) != TENSORS[0][2]:
        raise ReferenceError("source stem returned an unexpected device, dtype, or shape")
    return tokens


def _reference_assembly(torch: Any, streaming: Any, device: Any):
    merged = None
    for window_index, start_latent in enumerate(WINDOW_STARTS):
        part = synthetic_window(torch, window_index, device)
        if merged is None:
            merged = part
        else:
            merged = streaming._merge_overlapping(
                [merged, part], [0, start_latent * 4], dim=2
            )
    if merged is None or tuple(merged.shape) != TENSORS[1][2] or merged.dtype is not torch.bfloat16:
        raise ReferenceError("pinned temporal merger returned an unexpected shape or dtype")
    # Exact pinned BF16 `_postprocess_vae_video` expression from
    # hymm/pipeline/base.py; expand only after both rounded operations.
    pixels = postprocess_bf16(torch, merged)
    if pixels.device.type != "mps" or pixels.dtype is not torch.float32:
        raise ReferenceError("display conversion returned an unexpected device or dtype")
    return pixels


def _tensor_bytes(tensor: Any, torch: Any, stream, aggregate, offset: int) -> dict:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype is torch.bfloat16:
        width = 2
        dtype = "BF16"
    elif cpu.dtype is torch.float32:
        width = 4
        dtype = "F32"
    else:
        raise ReferenceError(f"unsupported oracle output dtype: {cpu.dtype}")
    shape = list(cpu.shape)
    expected = _product(shape) * width
    byte_array = cpu.view(torch.uint8).numpy()
    view = memoryview(byte_array).cast("B")
    if len(view) != expected:
        raise ReferenceError("oracle tensor byte extent mismatch")
    record_digest = hashlib.sha256()
    for first in range(0, expected, 8 * 1024 * 1024):
        chunk = view[first:min(first + 8 * 1024 * 1024, expected)]
        stream.write(chunk)
        aggregate.update(chunk)
        record_digest.update(chunk)
    view.release()
    del byte_array, cpu
    return {
        "dtype": dtype,
        "shape": shape,
        "offset": offset,
        "nbytes": expected,
        "sha256": record_digest.hexdigest(),
    }


def _publish_manifest(path: Path, document: Mapping[str, Any]) -> None:
    encoded = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_output(output: Path) -> tuple[Path, Path]:
    output = output.expanduser()
    if not output.is_absolute() or output.suffix.lower() != ".json":
        raise ReferenceError("--output must be an absolute .json path")
    try:
        parent = output.parent.resolve(strict=True)
        allowed = OUTPUT_ROOT.resolve(strict=True)
    except OSError as error:
        raise ReferenceError("output parent and /opt/mldata_ssd must already exist") from error
    if output.parent != parent or (parent != allowed and allowed not in parent.parents):
        raise ReferenceError("oracle output must be beneath canonical /opt/mldata_ssd")
    manifest = parent / output.name
    payload = parent / f"{output.stem}.tensors.bin"
    if os.path.lexists(manifest) or os.path.lexists(payload):
        raise ReferenceError("refusing to replace an existing oracle manifest or payload")
    return manifest, payload


def produce(args: argparse.Namespace) -> dict:
    if sys.byteorder != "little":
        raise ReferenceError("oracle payload encoding requires a little-endian host")
    manifest_path, payload_path = _validate_output(args.output)
    document, input_payload = _validate_input_bundle(args.inputs, args.inputs_sha256)
    if args.stem_sha256 != STEM_SHA256:
        raise ReferenceError("only the pinned stem checkpoint SHA-256 is admitted")

    try:
        import torch
    except ImportError as error:
        raise ReferenceError("PyTorch is required for the MPS reference") from error
    if not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is unavailable")
    import hy_nomas_pixelumm_reference as pixelumm_reference

    device = torch.device("mps")
    source, streaming = _verify_upstream(args.upstream)
    generation_input_records = document["tensors"]
    latent = _load_input_tensor(torch, input_payload, generation_input_records["input.latent"], device)
    noise = _load_input_tensor(torch, input_payload, generation_input_records["decoder.noise"], device)
    if tuple(latent.shape) != generation_inputs.LATENT_SHAPE or tuple(noise.shape) != generation_inputs.PIXEL_NOISE_SHAPE:
        raise ReferenceError("generation input device tensors do not match the pinned geometry")
    plan, _, _ = pixelumm_reference.make_attention_plan(source, (10, 18, 32), 8)
    weights = _load_stem_weights(torch, args.stem, args.stem_sha256, device)
    stem = _reference_stem(torch, source, plan, latent, noise, weights)
    pixels = _reference_assembly(torch, streaming, device)
    torch.mps.synchronize()

    aggregate = hashlib.sha256()
    records = {}
    payload_bytes = 0
    try:
        with payload_path.open("xb") as stream:
            records[TENSORS[0][0]] = _tensor_bytes(stem, torch, stream, aggregate, payload_bytes)
            payload_bytes += records[TENSORS[0][0]]["nbytes"]
            records[TENSORS[1][0]] = _tensor_bytes(pixels, torch, stream, aggregate, payload_bytes)
            payload_bytes += records[TENSORS[1][0]]["nbytes"]
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        payload_path.unlink(missing_ok=True)
        raise

    expected_bytes = sum(tensor_nbytes(dtype, shape) for _, dtype, shape in TENSORS)
    if payload_bytes != expected_bytes:
        payload_path.unlink(missing_ok=True)
        raise ReferenceError("oracle payload size differs from the fixed tensor inventory")
    output_document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "payload": payload_path.name,
        "payload_bytes": payload_bytes,
        "payload_sha256": aggregate.hexdigest(),
        "tensors": records,
        "upstream_revision": PINNED_REVISION,
        "upstream_sources": SOURCE_SHA256,
        "inputs_sha256": args.inputs_sha256,
        "input_payload_sha256": document["payload_sha256"],
        "input_payload_bytes": document["payload_bytes"],
        "stem_checkpoint_sha256": args.stem_sha256,
        "stem_window_start_latent": STEM_WINDOW_START,
        "stem_runtime_dtype": "BF16",
        "assembly_window_starts_latent": list(WINDOW_STARTS),
        "synthetic_window_formula": {
            "index_space": "flattened local CTHW in each [1,3,40,288,512] tensor",
            "window_index": "zero-based in output order 0..4",
            "f32_before_bf16": "(((index*37 + window_index*911) % 8192) - 4096) / 1024",
            "mps_construction": "compute one exact 8192-value period, then repeat 2160 times; avoid int64 modulo on full flattened indices",
            "cast": "PyTorch MPS RNE BF16 before source temporal merge",
        },
        "display_conversion": "BF16 round after multiply by 0.5, BF16 round after add 0.5, clamp to [0,1], then expand to F32",
        "comparison": {"atol": ATOL, "rtol": RTOL, "require_finite": True},
        "producer": {
            "device": "mps",
            "mps_fallback": False,
            "decoder_blocks_loaded": False,
            "head_loaded": False,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
        },
    }
    try:
        _publish_manifest(manifest_path, output_document)
    except Exception:
        payload_path.unlink(missing_ok=True)
        raise
    return {
        "manifest": str(manifest_path),
        "payload": str(payload_path),
        "manifest_sha256": _sha_file(manifest_path),
        "payload_sha256": aggregate.hexdigest(),
        "payload_bytes": payload_bytes,
        "schema": SCHEMA,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--inputs", required=True, type=Path,
                        help="authenticated generation-input manifest")
    result.add_argument("--inputs-sha256", required=True,
                        help="trusted SHA-256 of the generation-input manifest")
    result.add_argument("--stem", type=Path, default=DEFAULT_STEM,
                        help="pinned F32 stem subset")
    result.add_argument("--stem-sha256", default=STEM_SHA256,
                        help="trusted SHA-256 of the pinned stem subset")
    result.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM,
                        help=f"pinned upstream checkout at {PINNED_REVISION}")
    result.add_argument("--output", required=True, type=Path,
                        help="new oracle manifest path beneath /opt/mldata_ssd")
    result.add_argument("--self-test", action="store_true",
                        help="check the deterministic CPU-side synthetic-window contract")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        expected = (-4.0, -3.9638671875, -3.927734375, -3.8916015625)
        actual = tuple(synthetic_value(index, 0) for index in range(4))
        if actual != expected:
            raise ReferenceError("synthetic window formula self-test failed")
        print(json.dumps({"self_test": "passed", "schema": SCHEMA,
                          "tensors": [
                              {"name": name, "dtype": dtype, "shape": list(shape),
                               "nbytes": tensor_nbytes(dtype, shape)}
                              for name, dtype, shape in TENSORS
                          ]}, sort_keys=True))
        return 0
    print(json.dumps(produce(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReferenceError as error:
        print(f"pixelumm_generation_reference: {error}", file=sys.stderr)
        raise SystemExit(2)
