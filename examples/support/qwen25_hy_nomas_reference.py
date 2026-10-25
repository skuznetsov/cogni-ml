#!/usr/bin/env python3
"""Capture a tiny BF16 Qwen2.5-VL text-decoder fixture from installed Transformers.

This invokes the installed, pinned Hugging Face decoder, rotary embedding,
causal-mask helper, and SDPA dispatcher. It creates no checkpoint or full model.
Only deliberately tiny synthetic layers are written as JSON; real-width weight
payloads require a separate binary sidecar format.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import struct
import sys
import tempfile
from typing import Any


SCHEMA = "cogni_ml.qwen25_vl_text_decoder_hidden_state_reference"
SCHEMA_VERSION = 1
PINNED_HY_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
DEFAULT_OUTPUT_ROOT = Path("/opt/mldata_ssd/cogni-ml/qwen25-hy-nomas-20261024")
PINNED_TRANSFORMERS_VERSION = "4.57.3"
PINNED_TORCH_VERSION = "2.8.0"

# These are the official installed source files that define the decoder,
# configuration, RoPE, RMSNorm, activation, SDPA dispatcher, and mask path.
PINNED_TRANSFORMERS_SOURCE_SHA256 = {
    "models/qwen2_5_vl/modeling_qwen2_5_vl.py": "f5051523300145f66de080765c06bd8080350c15692c096d2c38e73fe9d03099",
    "models/qwen2_5_vl/configuration_qwen2_5_vl.py": "f9bb5d84f18cfdb506d8439cb5b0f07920e41d2b4d93a3945113c2359cfa48cb",
    "models/qwen2/modeling_qwen2.py": "08742a7c82a8713bc22f632f01c97b7eca3ef0e947180e7d6c9beec4897c4ff8",
    "modeling_rope_utils.py": "55cc0c8cb76f592ab178b4662adcbdf4ad5012bb0b4249c0a96603141f18e9ac",
    "modeling_utils.py": "bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d",
    "integrations/sdpa_attention.py": "dc5abe49a98dec3b9026739dfbf2e9a8f3e5272b2916b3c2d404727ac931a013",
    "masking_utils.py": "7a963feed8173b8265298dd24350c166222ce6b4719442fce48ab414d0478a6a",
    "activations.py": "3dd5a81b1e5e0c5371256f00efec19df52255420318737db1e03430a98cf4264",
}
PINNED_TORCH_FUNCTIONAL_SHA256 = "2b6eb4aff305990cff9e32c6ff40ff82c1fec4b5ceba1f43540146a8eb64e952"
TRANSFORMERS_RELEASE_SOURCE_COMPARISON = {
    "origin": "official Hugging Face Transformers GitHub release tags",
    "versions": ["4.57.1", "4.57.3"],
    "files": {
        "activations.py": {
            "4.57.1": "3dd5a81b1e5e0c5371256f00efec19df52255420318737db1e03430a98cf4264",
            "4.57.3": "3dd5a81b1e5e0c5371256f00efec19df52255420318737db1e03430a98cf4264",
        },
        "integrations/sdpa_attention.py": {
            "4.57.1": "dc5abe49a98dec3b9026739dfbf2e9a8f3e5272b2916b3c2d404727ac931a013",
            "4.57.3": "dc5abe49a98dec3b9026739dfbf2e9a8f3e5272b2916b3c2d404727ac931a013",
        },
        "masking_utils.py": {
            "4.57.1": "7a963feed8173b8265298dd24350c166222ce6b4719442fce48ab414d0478a6a",
            "4.57.3": "7a963feed8173b8265298dd24350c166222ce6b4719442fce48ab414d0478a6a",
        },
        "modeling_rope_utils.py": {
            "4.57.1": "55cc0c8cb76f592ab178b4662adcbdf4ad5012bb0b4249c0a96603141f18e9ac",
            "4.57.3": "55cc0c8cb76f592ab178b4662adcbdf4ad5012bb0b4249c0a96603141f18e9ac",
        },
        "modeling_utils.py": {
            "4.57.1": "bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d",
            "4.57.3": "bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d",
        },
        "models/qwen2/modeling_qwen2.py": {
            "4.57.1": "a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb",
            "4.57.3": "08742a7c82a8713bc22f632f01c97b7eca3ef0e947180e7d6c9beec4897c4ff8",
        },
        "models/qwen2_5_vl/configuration_qwen2_5_vl.py": {
            "4.57.1": "f9bb5d84f18cfdb506d8439cb5b0f07920e41d2b4d93a3945113c2359cfa48cb",
            "4.57.3": "f9bb5d84f18cfdb506d8439cb5b0f07920e41d2b4d93a3945113c2359cfa48cb",
        },
        "models/qwen2_5_vl/modeling_qwen2_5_vl.py": {
            "4.57.1": "f5051523300145f66de080765c06bd8080350c15692c096d2c38e73fe9d03099",
            "4.57.3": "f5051523300145f66de080765c06bd8080350c15692c096d2c38e73fe9d03099",
        },
    },
    "review": "All active Qwen2.5-VL, rotary, activation, mask, and SDPA files are byte-identical. The sole selected-file difference is the unrelated Qwen2Model.forward decorator in models/qwen2/modeling_qwen2.py; Qwen2RMSNorm and Qwen2MLP definitions are unchanged.",
}

DEFAULT_CONFIG = {
    "hidden_size": 24,
    "intermediate_size": 48,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 12,
    "num_hidden_layers": 1,
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "rope_scaling": {"rope_type": "default", "mrope_section": [2, 2, 2]},
    "hidden_act": "silu",
    "attention_implementation": "sdpa",
    "rms_gain_mode": "official_ones",
    "geometry": "tiny",
}

RMS_GAIN_MODES = ("official_ones", "nonunit_pattern_v1")

STAGE_NAMES = (
    "layers.{i}.input_layernorm",
    "layers.{i}.self_attn.q_proj",
    "layers.{i}.self_attn.k_proj",
    "layers.{i}.self_attn.v_proj",
    "post_rope_q",
    "post_rope_k",
    "attended",
    "layers.{i}.self_attn.o_proj",
    "residual_after_attention",
    "layers.{i}.post_attention_layernorm",
    "layers.{i}.mlp.gate_proj",
    "layers.{i}.mlp.up_proj",
    "silu_product",
    "layers.{i}.mlp.down_proj",
    "hidden_states_out",
)

WEIGHT_NAMES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

MAX_JSON_VALUES = 250_000
MAX_TINY_SEQUENCE = 16
MAX_TINY_LAYERS = 2
MAX_SIDECAR_BYTES = 512 * 1024 * 1024
SIDECAR_ENCODING = "bfloat16-le"

REAL_GEOMETRY = {
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_hidden_layers": 1,
    "max_position_embeddings": 128000,
    "rope_scaling": {"rope_type": "default", "mrope_section": [16, 24, 24]},
}
REAL_GEOMETRY_SOURCE = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "revision": "eb6ba55b40ada64977147b52e50a4145ee89ac4f",
    "config_url": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/eb6ba55b40ada64977147b52e50a4145ee89ac4f/config.json",
    "source_num_hidden_layers": 28,
    "scope": "per-layer geometry copied; one synthetic decoder layer captured; no config file or pretrained weights loaded",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _BF16SidecarWriter:
    """Stream BF16 tensor bytes to a bounded sibling sidecar."""

    def __init__(self, path: Path):
        if sys.byteorder != "little":
            raise RuntimeError("BF16 sidecar production requires a little-endian host")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".qwen25-bf16-", suffix=".tmp", dir=self.path.parent
        )
        self._temporary_path = Path(temporary)
        self._stream = os.fdopen(descriptor, "wb")
        self._digest = hashlib.sha256()
        self._nbytes = 0
        self._published = False
        self._closed = False

    def append(self, tensor: Any, torch: Any) -> dict[str, Any]:
        if tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
            raise RuntimeError("only BF16 MPS tensors may be written to the sidecar")
        shape = [int(dimension) for dimension in tensor.shape]
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("sidecar tensor shape must contain positive dimensions")
        expected_nbytes = math.prod(shape) * 2
        if self._nbytes + expected_nbytes > MAX_SIDECAR_BYTES:
            raise ValueError("BF16 sidecar exceeds its 512 MiB limit")

        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        words = cpu_tensor.view(torch.uint16).numpy()
        raw = memoryview(words).cast("B")
        if raw.nbytes != expected_nbytes:
            raise RuntimeError("BF16 sidecar byte count does not match tensor shape")
        written = self._stream.write(raw)
        if written != expected_nbytes:
            raise OSError("short write while producing BF16 sidecar")
        self._digest.update(raw)
        offset_bytes = self._nbytes
        self._nbytes += expected_nbytes
        return {
            "dtype": "bfloat16",
            "shape": shape,
            "storage": {
                "sidecar": "bf16",
                "offset_bytes": offset_bytes,
                "nbytes": expected_nbytes,
            },
        }

    def finish(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("BF16 sidecar writer is already closed")
        if not 0 < self._nbytes <= MAX_SIDECAR_BYTES:
            raise ValueError("BF16 sidecar is empty or exceeds its 512 MiB limit")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._closed = True
        os.link(self._temporary_path, self.path)
        self._published = True
        self._temporary_path.unlink()
        return {
            "path": self.path.name,
            "encoding": SIDECAR_ENCODING,
            "nbytes": self._nbytes,
            "sha256": self._digest.hexdigest(),
        }

    def abort(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True
        self._temporary_path.unlink(missing_ok=True)
        if self._published:
            self.path.unlink(missing_ok=True)


def _verify_upstream(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"upstream path is not a directory: {resolved}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(resolved), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not read the pinned Hy-Nomas revision") from error
    if revision != PINNED_HY_REVISION:
        raise RuntimeError(
            f"Hy-Nomas revision mismatch: {revision} != {PINNED_HY_REVISION}"
        )
    pin_files = (Path("pyproject.toml"), Path("docker/Dockerfile"))
    pin_metadata = {}
    declared_versions = set()
    for relative in pin_files:
        source = resolved / relative
        content = source.read_text(encoding="utf-8")
        matches = set(re.findall(r"transformers==([0-9.]+)", content))
        if len(matches) != 1:
            raise RuntimeError(f"could not establish one Transformers pin in {source}")
        declared_versions.update(matches)
        pin_metadata[relative.as_posix()] = {
            "path": str(source),
            "sha256": _sha256_file(source),
            "transformers_version": next(iter(matches)),
        }
    if len(declared_versions) != 1:
        raise RuntimeError(f"Hy-Nomas has inconsistent Transformers pins: {sorted(declared_versions)}")
    return {
        "revision": revision,
        "path": str(resolved),
        "transformers_version": next(iter(declared_versions)),
        "pin_sources": pin_metadata,
    }


def _verify_transformers_sources(transformers_package: Any) -> dict[str, dict[str, str]]:
    root = Path(transformers_package.__file__).resolve().parent
    actual: dict[str, dict[str, str]] = {}
    for relative, expected in PINNED_TRANSFORMERS_SOURCE_SHA256.items():
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"pinned Transformers source is missing: {source}")
        digest = _sha256_file(source)
        if digest != expected:
            raise RuntimeError(
                f"Transformers source hash mismatch for {relative}: {digest} != {expected}"
            )
        actual[relative] = {"path": str(source), "sha256": digest}
    return actual


def _verify_torch_sources(torch: Any) -> dict[str, dict[str, str]]:
    source = Path(torch.nn.functional.__file__).resolve()
    digest = _sha256_file(source)
    if digest != PINNED_TORCH_FUNCTIONAL_SHA256:
        raise RuntimeError(
            f"PyTorch functional source hash mismatch: {digest} != {PINNED_TORCH_FUNCTIONAL_SHA256}"
        )
    return {"torch/nn/functional.py": {"path": str(source), "sha256": digest}}


def _record_from_values(
    shape: list[int] | tuple[int, ...], values: list[Any], dtype: str
) -> dict[str, Any]:
    """Build a JSON tensor record, preserving source dtype and flat row-major order."""
    normalized_shape = [int(dimension) for dimension in shape]
    if not normalized_shape or any(dimension <= 0 for dimension in normalized_shape):
        raise ValueError("tensor shape must contain positive dimensions")
    expected_values = math.prod(normalized_shape)
    if len(values) != expected_values:
        raise ValueError(
            f"tensor value count {len(values)} does not match shape {normalized_shape}"
        )
    if dtype not in {"bfloat16", "float32", "int64", "bool"}:
        raise ValueError(f"unsupported tensor source dtype: {dtype}")
    clean_values: list[int | float | bool] = []
    for value in values:
        if dtype == "bool":
            if not isinstance(value, bool):
                raise ValueError("bool tensor record contains a non-bool value")
            clean_values.append(value)
        elif dtype == "int64":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("int64 tensor record contains a non-integer value")
            clean_values.append(value)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("floating tensor record contains a non-number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("tensor record contains a non-finite value")
            if dtype == "bfloat16":
                try:
                    bits = struct.unpack("!I", struct.pack("!f", number))[0]
                except OverflowError as error:
                    raise ValueError("BF16 decoded value exceeds F32 range") from error
                if bits & 0xFFFF:
                    raise ValueError("BF16 tensor record value is not an exact F32 decode")
            clean_values.append(number)
    return {"dtype": dtype, "shape": normalized_shape, "values": clean_values}


def _tensor_record(
    tensor: Any, torch: Any, sidecar_writer: _BF16SidecarWriter | None = None
) -> dict[str, Any]:
    """Copy a completed MPS capture to inline JSON or streamed BF16 sidecar storage."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("captured value is not a torch.Tensor")
    if tensor.device.type != "mps":
        raise RuntimeError(f"oracle tensor escaped MPS: {tensor.device}")
    source_dtype = str(tensor.dtype).removeprefix("torch.")
    if source_dtype == "bfloat16":
        if sidecar_writer is not None:
            return sidecar_writer.append(tensor, torch)
        values = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .reshape(-1)
            .tolist()
        )
    elif source_dtype in {"float32", "int64", "bool"}:
        values = tensor.detach().to(device="cpu").contiguous().reshape(-1).tolist()
    else:
        raise TypeError(f"unsupported captured tensor dtype: {tensor.dtype}")
    return _record_from_values(tuple(tensor.shape), values, source_dtype)


def _additive_bias_from_mask(causal_mask: Any, torch: Any) -> Any:
    """Build the native graph's BF16 additive bias on MPS; preserve HF's bool mask."""
    if causal_mask.device.type != "mps" or causal_mask.dtype != torch.bool:
        raise RuntimeError("official SDPA causal mask must be a bool MPS tensor")
    bias = torch.zeros(causal_mask.shape, device=causal_mask.device, dtype=torch.bfloat16)
    return bias.masked_fill(~causal_mask, torch.finfo(torch.bfloat16).min)


def _validate_tensor_record(
    record: Any,
    expected_dtype: str,
    external_ranges: list[tuple[int, int]],
) -> None:
    if not isinstance(record, dict) or record.get("dtype") != expected_dtype:
        raise ValueError("tensor record is missing or has the wrong dtype")
    shape = record.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
               for dimension in shape)
    ):
        raise ValueError("tensor shape must contain positive integer dimensions")
    expected_nbytes = math.prod(shape) * 2 if expected_dtype == "bfloat16" else None
    if set(record) == {"dtype", "shape", "values"}:
        if not isinstance(record["values"], list):
            raise ValueError("inline tensor values must be a list")
        _record_from_values(shape, record["values"], expected_dtype)
        return
    if set(record) != {"dtype", "shape", "storage"}:
        raise ValueError("tensor record must contain exactly one of values or storage")
    if expected_dtype != "bfloat16" or not isinstance(record["storage"], dict):
        raise ValueError("only BF16 tensors may use binary sidecar storage")
    storage = record["storage"]
    if set(storage) != {"sidecar", "offset_bytes", "nbytes"}:
        raise ValueError("BF16 storage descriptor fields are invalid")
    offset = storage.get("offset_bytes")
    nbytes = storage.get("nbytes")
    if storage.get("sidecar") != "bf16":
        raise ValueError("BF16 tensor references an unknown sidecar")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset % 2:
        raise ValueError("BF16 sidecar offset must be a nonnegative aligned integer")
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes != expected_nbytes:
        raise ValueError("BF16 sidecar byte count does not match tensor shape")
    if offset + nbytes > MAX_SIDECAR_BYTES:
        raise ValueError("BF16 tensor range exceeds the 512 MiB sidecar limit")
    external_ranges.append((offset, offset + nbytes))


def _verify_bf16_sidecar(
    descriptor: Any, manifest_path: Path | None
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path", "encoding", "nbytes", "sha256"
    }:
        raise ValueError("BF16 sidecar manifest fields are invalid")
    relative_name = descriptor.get("path")
    if (
        not isinstance(relative_name, str)
        or not relative_name
        or relative_name in {".", ".."}
        or "/" in relative_name
        or "\\" in relative_name
        or "\x00" in relative_name
    ):
        raise ValueError("BF16 sidecar path must be a simple sibling basename")
    if descriptor.get("encoding") != SIDECAR_ENCODING:
        raise ValueError("BF16 sidecar encoding is unsupported")
    nbytes = descriptor.get("nbytes")
    if (
        isinstance(nbytes, bool)
        or not isinstance(nbytes, int)
        or not 0 < nbytes <= MAX_SIDECAR_BYTES
        or nbytes % 2
    ):
        raise ValueError("BF16 sidecar size is outside its even 512 MiB limit")
    expected_hash = descriptor.get("sha256")
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("BF16 sidecar SHA-256 must be 64 lowercase hexadecimal characters")
    if manifest_path is None:
        raise ValueError("manifest path is required to resolve BF16 sidecar storage")
    root = manifest_path.parent.resolve(strict=True)
    sidecar_path = root / relative_name
    try:
        sidecar_stat = sidecar_path.lstat()
    except OSError as error:
        raise ValueError("BF16 sidecar file is missing or inaccessible") from error
    if stat.S_ISLNK(sidecar_stat.st_mode) or not stat.S_ISREG(sidecar_stat.st_mode):
        raise ValueError("BF16 sidecar must be a regular non-symlink file")
    resolved_path = sidecar_path.resolve(strict=True)
    if resolved_path.parent != root:
        raise ValueError("BF16 sidecar resolves outside the manifest directory")
    if sidecar_stat.st_size != nbytes:
        raise ValueError("BF16 sidecar file size does not match its manifest")

    digest = hashlib.sha256()
    try:
        with resolved_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                if len(chunk) % 2:
                    raise ValueError("BF16 sidecar contains a partial word")
                digest.update(chunk)
                try:
                    numpy = importlib.import_module("numpy")
                    words = numpy.frombuffer(chunk, dtype="<u2")
                    has_nonfinite = bool(
                        numpy.any((words & numpy.uint16(0x7F80)) == numpy.uint16(0x7F80))
                    )
                except ImportError:
                    has_nonfinite = any(
                        int.from_bytes(chunk[offset : offset + 2], "little") & 0x7F80
                        == 0x7F80
                        for offset in range(0, len(chunk), 2)
                    )
                if has_nonfinite:
                    raise ValueError("BF16 sidecar contains a non-finite value")
    except OSError as error:
        raise ValueError("could not read BF16 sidecar") from error
    if digest.hexdigest() != expected_hash:
        raise ValueError("BF16 sidecar SHA-256 does not match its manifest")
    return resolved_path, descriptor


def _small_record_values(record: dict[str, Any], sidecar_path: Path | None) -> list[Any]:
    if "values" in record:
        return record["values"]
    storage = record["storage"]
    value_count = math.prod(record["shape"])
    if value_count > 4096:
        raise ValueError("semantic validation attempted to materialize a large sidecar tensor")
    if sidecar_path is None:
        raise ValueError("BF16 sidecar path is required to read tensor values")
    with sidecar_path.open("rb") as source:
        source.seek(storage["offset_bytes"])
        raw = source.read(storage["nbytes"])
    if len(raw) != storage["nbytes"]:
        raise ValueError("BF16 sidecar tensor range is truncated")
    values = []
    for offset in range(0, len(raw), 2):
        bf16_bits = int.from_bytes(raw[offset : offset + 2], "little")
        f32_bits = bf16_bits << 16
        values.append(struct.unpack("<f", struct.pack("<I", f32_bits))[0])
    return values


def _validate_fixture(document: dict[str, Any], manifest_path: Path | None = None) -> None:
    if document.get("schema") != SCHEMA or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("fixture schema identity is invalid")
    layers = document.get("layers")
    config = document.get("config")
    if not isinstance(layers, list) or not layers:
        raise ValueError("fixture must contain at least one decoder layer")
    geometry = config.get("geometry", "tiny") if isinstance(config, dict) else None
    if geometry == "tiny":
        if len(layers) > MAX_TINY_LAYERS:
            raise ValueError("tiny fixture must contain at most two decoder layers")
    elif geometry == "qwen25-vl-7b":
        if len(layers) != 1:
            raise ValueError("Qwen2.5-VL-7B profile must contain one decoder layer")
    else:
        raise ValueError("fixture geometry profile is missing or unsupported")
    if not isinstance(config, dict) or config.get("num_hidden_layers") != len(layers):
        raise ValueError("fixture config layer count does not match its layers")
    if config.get("attention_implementation") != "sdpa":
        raise ValueError("fixture must use the official SDPA attention dispatcher")
    sidecar_path = None
    sidecar_descriptor = None
    if "sidecars" in document:
        sidecars = document["sidecars"]
        if not isinstance(sidecars, dict) or set(sidecars) != {"bf16"}:
            raise ValueError("fixture may declare exactly one BF16 sidecar")
        sidecar_path, sidecar_descriptor = _verify_bf16_sidecar(
            sidecars["bf16"], manifest_path
        )
    external_ranges: list[tuple[int, int]] = []
    rms_gain_mode = config.get("rms_gain_mode")
    if rms_gain_mode not in RMS_GAIN_MODES:
        raise ValueError("fixture RMS gain mode is missing or unsupported")
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("fixture inputs are missing")
    required_input_dtypes = (
        ("hidden_states", "bfloat16"),
        ("binary_mask", "int64"),
        ("position_ids", "float32"),
        ("multimodal_position_ids", "int64"),
        ("inv_freq", "float32"),
        ("rotary_cos", "bfloat16"),
        ("rotary_sin", "bfloat16"),
        ("causal_attention_mask", "bool"),
        ("additive_attention_bias", "bfloat16"),
    )
    for key, dtype in required_input_dtypes:
        record = inputs.get(key)
        if not isinstance(record, dict) or record.get("dtype") != dtype:
            raise ValueError(f"fixture input {key} is missing or has the wrong dtype")
        _validate_tensor_record(record, dtype, external_ranges)
    hidden_size = config.get("hidden_size")
    intermediate_size = config.get("intermediate_size")
    num_heads = config.get("num_attention_heads")
    num_kv_heads = config.get("num_key_value_heads")
    head_dim = config.get("head_dim")
    sequence_length = config.get("sequence_length")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (
        hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, sequence_length
    )):
        raise ValueError("fixture config has invalid tensor dimensions")
    if hidden_size != num_heads * head_dim or num_heads % num_kv_heads:
        raise ValueError("fixture config has inconsistent attention dimensions")
    if geometry == "tiny" and sequence_length > MAX_TINY_SEQUENCE:
        raise ValueError(f"tiny fixture sequence length exceeds {MAX_TINY_SEQUENCE}")
    if geometry == "qwen25-vl-7b":
        for key, expected in REAL_GEOMETRY.items():
            if config.get(key) != expected:
                raise ValueError(f"real geometry config field {key} does not match Qwen2.5-VL-7B")
        if sequence_length != 4:
            raise ValueError("Qwen2.5-VL-7B profile uses exactly four input tokens")
        if sidecar_descriptor is None:
            raise ValueError("Qwen2.5-VL-7B fixture requires its BF16 sidecar")
    expected_input_shapes = {
        "hidden_states": [1, sequence_length, hidden_size],
        "binary_mask": [1, sequence_length],
        "position_ids": [sequence_length],
        "multimodal_position_ids": [3, 1, sequence_length],
        "inv_freq": [head_dim // 2],
        "rotary_cos": [3, 1, sequence_length, head_dim],
        "rotary_sin": [3, 1, sequence_length, head_dim],
        "causal_attention_mask": [1, 1, sequence_length, sequence_length],
        "additive_attention_bias": [1, 1, sequence_length, sequence_length],
    }
    for key, expected_shape in expected_input_shapes.items():
        if inputs[key].get("shape") != expected_shape:
            raise ValueError(f"fixture input {key} has shape {inputs[key].get('shape')}, expected {expected_shape}")
    mask_values = _small_record_values(inputs["binary_mask"], sidecar_path)
    if any(value not in (0, 1) for value in mask_values) or not any(mask_values):
        raise ValueError("fixture binary mask must contain at least one visible token")
    first_padding_index = next(
        (index for index, value in enumerate(mask_values) if value == 0), len(mask_values)
    )
    if mask_values[0] != 1 or any(mask_values[first_padding_index:]):
        raise ValueError("fixture binary mask must be right padded with its first token visible")
    if config.get("binary_mask") != mask_values:
        raise ValueError("fixture config mask does not match the serialized input mask")
    position_offset = config.get("position_offset")
    expected_positions = [position_offset + index for index in range(sequence_length)]
    if _small_record_values(inputs["position_ids"], sidecar_path) != [float(value) for value in expected_positions]:
        raise ValueError("fixture F32 position ids do not match the configured offset")
    if _small_record_values(inputs["multimodal_position_ids"], sidecar_path) != expected_positions * 3:
        raise ValueError("fixture multimodal position ids do not match the configured offset")
    expected_sdpa_mask = [
        bool(mask_values[key] and key <= query)
        for query in range(sequence_length)
        for key in range(sequence_length)
    ]
    sdpa_mask_values = _small_record_values(inputs["causal_attention_mask"], sidecar_path)
    if sdpa_mask_values != expected_sdpa_mask:
        raise ValueError("fixture bool SDPA mask does not match right-padded causal visibility")
    if any(
        not any(sdpa_mask_values[row * sequence_length : (row + 1) * sequence_length])
        for row in range(sequence_length)
    ):
        raise ValueError("fixture contains a fully masked causal query row")
    minimum_bf16 = -3.3895313892515355e38
    expected_additive_bias = [
        0.0 if allowed else minimum_bf16 for allowed in expected_sdpa_mask
    ]
    if _small_record_values(inputs["additive_attention_bias"], sidecar_path) != expected_additive_bias:
        raise ValueError("fixture BF16 attention bias does not match its bool SDPA mask")
    all_values = sum(
        len(record["values"])
        for record in _walk_tensor_records(document)
        if "values" in record
    )
    if all_values > MAX_JSON_VALUES:
        raise ValueError("fixture exceeds the inline JSON tensor-value budget")
    seen_seeds: set[int] = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("index") != index:
            raise ValueError("fixture layer order is invalid")
        seed = layer.get("initialization_seed")
        if not isinstance(seed, int) or seed in seen_seeds:
            raise ValueError("fixture decoder layers must have distinct integer seeds")
        seen_seeds.add(seed)
        weights = layer.get("weights")
        stages = layer.get("stages")
        if not isinstance(weights, dict) or not isinstance(stages, dict):
            raise ValueError(f"fixture layer {index} is missing weights or stages")
        if set(stages) != {name.format(i=index) for name in STAGE_NAMES}:
            raise ValueError(f"fixture layer {index} has an unexpected stage set")
        if set(weights) != {f"layers.{index}.{name}" for name in WEIGHT_NAMES}:
            raise ValueError(f"fixture layer {index} has an unexpected weight set")
        for record in (*weights.values(), *stages.values()):
            if not isinstance(record, dict) or record.get("dtype") != "bfloat16":
                raise ValueError(f"fixture layer {index} contains a non-BF16 weight or stage")
            _validate_tensor_record(record, "bfloat16", external_ranges)
        expected_stage_shapes = {
            f"layers.{index}.input_layernorm": [1, sequence_length, hidden_size],
            f"layers.{index}.self_attn.q_proj": [1, sequence_length, hidden_size],
            f"layers.{index}.self_attn.k_proj": [1, sequence_length, num_kv_heads * head_dim],
            f"layers.{index}.self_attn.v_proj": [1, sequence_length, num_kv_heads * head_dim],
            "post_rope_q": [1, num_heads, sequence_length, head_dim],
            "post_rope_k": [1, num_kv_heads, sequence_length, head_dim],
            "attended": [1, sequence_length, hidden_size],
            f"layers.{index}.self_attn.o_proj": [1, sequence_length, hidden_size],
            "residual_after_attention": [1, sequence_length, hidden_size],
            f"layers.{index}.post_attention_layernorm": [1, sequence_length, hidden_size],
            f"layers.{index}.mlp.gate_proj": [1, sequence_length, intermediate_size],
            f"layers.{index}.mlp.up_proj": [1, sequence_length, intermediate_size],
            "silu_product": [1, sequence_length, intermediate_size],
            f"layers.{index}.mlp.down_proj": [1, sequence_length, hidden_size],
            "hidden_states_out": [1, sequence_length, hidden_size],
        }
        for key, expected_shape in expected_stage_shapes.items():
            if stages[key].get("shape") != expected_shape:
                raise ValueError(f"fixture stage {key} has shape {stages[key].get('shape')}, expected {expected_shape}")
        expected_weight_shapes = {
            f"layers.{index}.input_layernorm.weight": [hidden_size],
            f"layers.{index}.self_attn.q_proj.weight": [hidden_size, hidden_size],
            f"layers.{index}.self_attn.q_proj.bias": [hidden_size],
            f"layers.{index}.self_attn.k_proj.weight": [num_kv_heads * head_dim, hidden_size],
            f"layers.{index}.self_attn.k_proj.bias": [num_kv_heads * head_dim],
            f"layers.{index}.self_attn.v_proj.weight": [num_kv_heads * head_dim, hidden_size],
            f"layers.{index}.self_attn.v_proj.bias": [num_kv_heads * head_dim],
            f"layers.{index}.self_attn.o_proj.weight": [hidden_size, hidden_size],
            f"layers.{index}.post_attention_layernorm.weight": [hidden_size],
            f"layers.{index}.mlp.gate_proj.weight": [intermediate_size, hidden_size],
            f"layers.{index}.mlp.up_proj.weight": [intermediate_size, hidden_size],
            f"layers.{index}.mlp.down_proj.weight": [hidden_size, intermediate_size],
        }
        for key, expected_shape in expected_weight_shapes.items():
            if weights[key].get("shape") != expected_shape:
                raise ValueError(f"fixture weight {key} has shape {weights[key].get('shape')}, expected {expected_shape}")
        expected_gains = _expected_rms_gain_values(rms_gain_mode, hidden_size)
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            key = f"layers.{index}.{name}"
            if _small_record_values(weights[key], sidecar_path) != expected_gains:
                raise ValueError(f"fixture RMS gain {key} does not match mode {rms_gain_mode}")

    if sidecar_descriptor is None:
        if external_ranges:
            raise ValueError("external BF16 records require a declared sidecar")
    else:
        cursor = 0
        for start, end in sorted(external_ranges):
            if start != cursor:
                raise ValueError("BF16 sidecar tensor ranges overlap or leave unreferenced bytes")
            cursor = end
        if cursor != sidecar_descriptor["nbytes"]:
            raise ValueError("BF16 sidecar contains unreferenced bytes")


def _walk_tensor_records(value: Any):
    if isinstance(value, dict):
        if {"dtype", "shape"}.issubset(value) and (
            "values" in value or "storage" in value
        ):
            yield value
        else:
            for child in value.values():
                yield from _walk_tensor_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_tensor_records(child)


def _import_official(torch: Any) -> tuple[Any, Any, dict[str, dict[str, str]], str]:
    transformers = importlib.import_module("transformers")
    version = importlib.metadata.version("transformers")
    if version != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Transformers version mismatch: {version} != {PINNED_TRANSFORMERS_VERSION}"
        )
    torch_version = str(torch.__version__).split("+", 1)[0]
    if torch_version != PINNED_TORCH_VERSION:
        raise RuntimeError(f"PyTorch version mismatch: {torch_version} != {PINNED_TORCH_VERSION}")

    sources = _verify_transformers_sources(transformers)

    # This isolated text-only process does not use torchvision. The installed
    # optional torchvision package has an incompatible fake-op registration;
    # disabling its import leaves official text decoder code unchanged.
    import_utils = importlib.import_module("transformers.utils.import_utils")
    import_utils._torchvision_available = False
    modeling = importlib.import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
    configuration = importlib.import_module(
        "transformers.models.qwen2_5_vl.configuration_qwen2_5_vl"
    )
    return modeling, configuration, sources, version


def _default_config(layer_count: int, geometry: str = "tiny") -> dict[str, Any]:
    if geometry not in {"tiny", "qwen25-vl-7b"}:
        raise ValueError(f"unsupported geometry profile: {geometry}")
    if geometry == "qwen25-vl-7b" and layer_count != 1:
        raise ValueError("Qwen2.5-VL-7B geometry profile supports exactly one decoder layer")
    config = dict(DEFAULT_CONFIG)
    config["rope_scaling"] = dict(DEFAULT_CONFIG["rope_scaling"])
    config["rope_scaling"]["mrope_section"] = list(
        DEFAULT_CONFIG["rope_scaling"]["mrope_section"]
    )
    config["num_hidden_layers"] = layer_count
    config["geometry"] = geometry
    if geometry == "qwen25-vl-7b":
        config.update(REAL_GEOMETRY)
        config["rope_scaling"] = dict(REAL_GEOMETRY["rope_scaling"])
        config["geometry"] = geometry
    return config


def _expected_rms_gain_values(mode: str, hidden_size: int) -> list[float]:
    if mode == "official_ones":
        return [1.0] * hidden_size
    if mode == "nonunit_pattern_v1":
        return [0.625 + (index % 7) * 0.125 for index in range(hidden_size)]
    raise ValueError(f"unsupported RMS gain mode: {mode}")


def _set_nonunit_rms_gains(layer: Any, hidden_size: int, torch: Any) -> None:
    values = _expected_rms_gain_values("nonunit_pattern_v1", hidden_size)
    gain = torch.tensor(values, device="mps", dtype=torch.float32).to(dtype=torch.bfloat16)
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(gain)
        layer.post_attention_layernorm.weight.copy_(gain)


def _new_text_config(configuration: Any, values: dict[str, Any]) -> Any:
    config = configuration.Qwen2_5_VLTextConfig(
        vocab_size=64,
        hidden_size=values["hidden_size"],
        intermediate_size=values["intermediate_size"],
        num_hidden_layers=values["num_hidden_layers"],
        num_attention_heads=values["num_attention_heads"],
        num_key_value_heads=values["num_key_value_heads"],
        hidden_act=values["hidden_act"],
        max_position_embeddings=values["max_position_embeddings"],
        rms_norm_eps=values["rms_norm_eps"],
        rope_theta=values["rope_theta"],
        rope_scaling=values["rope_scaling"],
        attention_dropout=0.0,
        use_cache=False,
        attn_implementation=values["attention_implementation"],
    )
    config._attn_implementation = values["attention_implementation"]
    return config


def _snapshot(tensor: Any, torch: Any) -> Any:
    if tensor.device.type != "mps":
        raise RuntimeError(f"decoder intermediate escaped MPS: {tensor.device}")
    return tensor.detach().clone()


def _capture_layer(
    layer: Any,
    layer_index: int,
    hidden_states: Any,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: tuple[Any, Any],
    modeling: Any,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    prefix = f"layers.{layer_index}."
    captured: dict[str, Any] = {"hidden_states_in": _snapshot(hidden_states, torch)}
    handles = []

    def capture_output(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                captured[name] = _snapshot(output, torch)
        return hook

    for name, module in (
        (prefix + "input_layernorm", layer.input_layernorm),
        (prefix + "self_attn.q_proj", layer.self_attn.q_proj),
        (prefix + "self_attn.k_proj", layer.self_attn.k_proj),
        (prefix + "self_attn.v_proj", layer.self_attn.v_proj),
        (prefix + "self_attn.o_proj", layer.self_attn.o_proj),
        (prefix + "post_attention_layernorm", layer.post_attention_layernorm),
        (prefix + "mlp.gate_proj", layer.mlp.gate_proj),
        (prefix + "mlp.up_proj", layer.mlp.up_proj),
        (prefix + "mlp.down_proj", layer.mlp.down_proj),
    ):
        handles.append(module.register_forward_hook(capture_output(name)))

    def capture_residual(_module: Any, inputs: Any) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["residual_after_attention"] = _snapshot(inputs[0], torch)

    def capture_attended(_module: Any, inputs: Any) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["attended"] = _snapshot(inputs[0], torch)

    def capture_silu_product(_module: Any, inputs: Any) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["silu_product"] = _snapshot(inputs[0], torch)

    handles.extend(
        (
            layer.post_attention_layernorm.register_forward_pre_hook(capture_residual),
            layer.self_attn.o_proj.register_forward_pre_hook(capture_attended),
            layer.mlp.down_proj.register_forward_pre_hook(capture_silu_product),
        )
    )

    original_rope = modeling.apply_multimodal_rotary_pos_emb

    def capture_rope(query: Any, key: Any, *args: Any, **kwargs: Any):
        query_rotated, key_rotated = original_rope(query, key, *args, **kwargs)
        captured["post_rope_q"] = _snapshot(query_rotated, torch)
        captured["post_rope_k"] = _snapshot(key_rotated, torch)
        return query_rotated, key_rotated

    modeling.apply_multimodal_rotary_pos_emb = capture_rope
    try:
        outputs = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            output_attentions=False,
            use_cache=False,
        )
    finally:
        modeling.apply_multimodal_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()

    output = outputs[0]
    captured["hidden_states_out"] = _snapshot(output, torch)
    required_capture = {
        "hidden_states_in",
        "post_rope_q",
        "post_rope_k",
        "attended",
        "residual_after_attention",
        "silu_product",
    }
    if missing := required_capture.difference(captured):
        raise RuntimeError(f"official decoder did not reach expected capture points: {sorted(missing)}")
    return output, captured


def _parameter_records(
    layer: Any,
    layer_index: int,
    torch: Any,
    sidecar_writer: _BF16SidecarWriter | None = None,
) -> dict[str, dict[str, Any]]:
    expected = {
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    named = dict(layer.named_parameters())
    if set(named) != expected:
        raise RuntimeError(f"unexpected official decoder parameter set: {sorted(named)}")
    result = {}
    for name, parameter in sorted(named.items()):
        if parameter.device.type != "mps" or parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"expected BF16 MPS parameter {name}, got {parameter.device}/{parameter.dtype}")
        result[f"layers.{layer_index}.{name}"] = _tensor_record(
            parameter, torch, sidecar_writer
        )
    return result


def _manifest_tensor_records(
    captures: list[dict[str, Any]],
    layers: list[Any],
    input_hidden_states: Any,
    binary_mask: Any,
    position_ids_f32: Any,
    multimodal_position_ids: Any,
    rotary: Any,
    cos: Any,
    sin: Any,
    causal_mask: Any,
    additive_attention_bias: Any,
    config: dict[str, Any],
    upstream: dict[str, Any],
    torch: Any,
    torch_sources: dict[str, dict[str, str]],
    transformers_sources: dict[str, dict[str, str]],
    sidecar_writer: _BF16SidecarWriter | None = None,
) -> dict[str, Any]:
    inputs = {
        "hidden_states": _tensor_record(input_hidden_states, torch, sidecar_writer),
        "binary_mask": _tensor_record(binary_mask, torch),
        "position_ids": _tensor_record(position_ids_f32, torch),
        "multimodal_position_ids": _tensor_record(multimodal_position_ids, torch),
        "inv_freq": _tensor_record(rotary.inv_freq, torch),
        "rotary_cos": _tensor_record(cos, torch, sidecar_writer),
        "rotary_sin": _tensor_record(sin, torch, sidecar_writer),
        "causal_attention_mask": _tensor_record(causal_mask, torch),
        "additive_attention_bias": _tensor_record(
            additive_attention_bias, torch, sidecar_writer
        ),
    }
    layer_records = []
    for index, (layer, capture) in enumerate(zip(layers, captures, strict=True)):
        stage_names = {name.format(i=index) for name in STAGE_NAMES}
        stages = {
            key: _tensor_record(capture[key], torch, sidecar_writer)
            for key in sorted(stage_names)
        }
        layer_records.append(
            {
                "index": index,
                "initialization_seed": 20261024 + index,
                "weights": _parameter_records(layer, index, torch, sidecar_writer),
                "stages": stages,
            }
        )
    fixture = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "upstream": {
            "repository": "Hy-Nomas",
            "revision": upstream["revision"],
            "checkout": upstream["path"],
            "declared_transformers_version": upstream["transformers_version"],
            "transformers_pin_sources": upstream["pin_sources"],
            "role": "related pinned source context; decoder implementation is installed Transformers",
        },
        "provenance": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "transformers_version": PINNED_TRANSFORMERS_VERSION,
            "torch_sources": torch_sources,
            "transformers_sources": transformers_sources,
            "transformers_release_source_comparison": TRANSFORMERS_RELEASE_SOURCE_COMPARISON,
            "fixture_producer_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "transformers_pin_mismatch": {
                "hy_nomas_declares": upstream["transformers_version"],
                "oracle_runtime": PINNED_TRANSFORMERS_VERSION,
                "matches": upstream["transformers_version"] == PINNED_TRANSFORMERS_VERSION,
            },
            "model_config": (
                "synthetic Qwen2_5_VLTextConfig; geometry fields copied from the pinned official Qwen2.5-VL-7B config; no config file or pretrained weights loaded"
                if config.get("geometry") == "qwen25-vl-7b"
                else "synthetic Qwen2_5_VLTextConfig; no pretrained config or weights loaded"
            ),
            "geometry_profile_source": (
                REAL_GEOMETRY_SOURCE
                if config.get("geometry") == "qwen25-vl-7b"
                else {"profile": "tiny synthetic dimensions"}
            ),
            "official_class": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer",
            "attention_dispatcher": "transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS['sdpa']",
            "additive_attention_bias": "derived on MPS from the captured bool SDPA mask: 0 where True, BF16 minimum where False; not passed to the HF oracle",
            "device": "mps",
            "activation_dtype": "bfloat16",
            "rms_gain_mode": config["rms_gain_mode"],
            "rms_gain_setup": (
                "official constructor initialized both RMSNorm gains to one"
                if config["rms_gain_mode"] == "official_ones"
                else "MPS setup copied the deterministic nonunit BF16 pattern into both RMSNorm gains before the official forward"
            ),
            "fallback": "PYTORCH_ENABLE_MPS_FALLBACK=0",
            "synthetic_weights": True,
            "weight_initialization": "official PyTorch module constructors on MPS; per-layer seeds; parameters cast to BF16 on MPS",
            "torchvision_import": "disabled process-locally for this text-only oracle due incompatible optional package registration; no torchvision code is used",
            "parity_scope": "synthetic bounded text decoder only; no checkpoint weights or image/video tower",
        },
        "config": config,
        "tensor_value_encoding": (
            "BF16 tensor records reference one raw little-endian sidecar; other tensor records use row-major flattened JSON values"
            if sidecar_writer is not None
            else "row-major flattened JSON numbers; BF16 sources decoded exactly to F32"
        ),
        "inputs": inputs,
        "layers": layer_records,
    }
    return fixture


def _generate(
    args: argparse.Namespace,
    sidecar_writer: _BF16SidecarWriter | None = None,
) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if args.layers not in (1, 2):
        raise ValueError("only one- or two-layer decoder captures are supported")
    if args.geometry == "qwen25-vl-7b":
        if args.layers != 1 or args.sequence_length != 4:
            raise ValueError("Qwen2.5-VL-7B geometry requires one layer and four tokens")
        if sidecar_writer is None:
            raise ValueError("Qwen2.5-VL-7B geometry requires BF16 sidecar output")
    elif sidecar_writer is not None:
        raise ValueError("BF16 sidecar output is reserved for real geometry fixtures")
    if not 1 <= args.sequence_length <= MAX_TINY_SEQUENCE:
        raise ValueError(f"sequence length must be in 1..{MAX_TINY_SEQUENCE}")
    mask_values = [int(value) for value in args.mask.split(",")]
    if len(mask_values) != args.sequence_length or any(value not in (0, 1) for value in mask_values):
        raise ValueError("--mask must provide one comma-separated 0/1 value per token")
    if not any(mask_values):
        raise ValueError("attention mask must expose at least one key token")
    first_padding_index = next(
        (index for index, value in enumerate(mask_values) if value == 0), len(mask_values)
    )
    if mask_values[0] != 1 or any(mask_values[first_padding_index:]):
        raise ValueError("--mask must be right padded with its first token visible")

    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("this oracle requires MPS; CPU fallback is forbidden")
    modeling, configuration, transformers_sources, transformers_version = _import_official(torch)
    torch_sources = _verify_torch_sources(torch)
    upstream = _verify_upstream(args.upstream)
    if upstream["revision"] != PINNED_HY_REVISION:
        raise RuntimeError("pinned Hy-Nomas source verification failed")

    config_values = _default_config(args.layers, args.geometry)
    config_object = _new_text_config(configuration, config_values)
    device = torch.device("mps")
    seed = 20261025
    torch.manual_seed(seed)

    # The official position embedding module owns an FP32 inv_freq buffer.
    # Keep it FP32 while decoder Parameters are converted to BF16 below.
    with torch.device(device):
        rotary = modeling.Qwen2_5_VLRotaryEmbedding(config=config_object, device=device)
    rotary.eval()

    layers = []
    for index in range(args.layers):
        torch.manual_seed(20261024 + index)
        with torch.device(device):
            layer = modeling.Qwen2_5_VLDecoderLayer(config_object, layer_idx=index)
        layer.to(device=device)
        for parameter in layer.parameters():
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
        if args.rms_gain_mode == "nonunit_pattern_v1":
            _set_nonunit_rms_gains(layer, config_values["hidden_size"], torch)
        layer.eval()
        layers.append(layer)

    torch.manual_seed(seed)
    input_hidden_states = (
        torch.randn(
            (1, args.sequence_length, config_values["hidden_size"]),
            device=device,
            dtype=torch.float32,
        )
        .mul_(0.125)
        .to(dtype=torch.bfloat16)
    )
    binary_mask = torch.tensor(mask_values, device=device, dtype=torch.int64).reshape(1, -1)
    position_ids_f32 = torch.arange(args.sequence_length, device=device, dtype=torch.float32)
    position_ids = torch.arange(args.sequence_length, device=device, dtype=torch.int64)
    position_ids_f32 = position_ids_f32 + args.position_offset
    position_ids = position_ids + args.position_offset
    multimodal_position_ids = position_ids.reshape(1, 1, -1).expand(3, 1, -1).contiguous()
    cache_position = torch.arange(args.sequence_length, device=device, dtype=torch.int64)

    with torch.inference_mode():
        cos, sin = rotary(input_hidden_states, multimodal_position_ids)
        if rotary.inv_freq.dtype != torch.float32:
            raise RuntimeError("official rotary inv_freq must remain FP32")
        # This is the same official Transformers mask helper used by
        # Qwen2_5_VLTextModel.forward; it consumes the binary key mask.
        causal_mask = modeling.create_causal_mask(
            config=config_object,
            input_embeds=input_hidden_states,
            attention_mask=binary_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=None,
        )
        if not isinstance(causal_mask, torch.Tensor):
            raise RuntimeError("official causal mask helper did not return a mask for padded input")
        additive_attention_bias = _additive_bias_from_mask(causal_mask, torch)
        hidden_states = input_hidden_states
        captures: list[dict[str, Any]] = []
        for index, layer in enumerate(layers):
            hidden_states, captured = _capture_layer(
                layer,
                index,
                hidden_states,
                causal_mask,
                multimodal_position_ids,
                (cos, sin),
                modeling,
                torch,
            )
            captures.append(captured)
        torch.mps.synchronize()

    config_values["num_hidden_layers"] = len(layers)
    config_values["sequence_length"] = args.sequence_length
    config_values["binary_mask"] = mask_values
    config_values["position_offset"] = args.position_offset
    config_values["rms_gain_mode"] = args.rms_gain_mode
    config_values["synthetic_input_seed"] = seed
    fixture = _manifest_tensor_records(
        captures,
        layers,
        input_hidden_states,
        binary_mask,
        position_ids_f32,
        multimodal_position_ids,
        rotary,
        cos,
        sin,
        causal_mask,
        additive_attention_bias,
        config_values,
        upstream,
        torch,
        torch_sources,
        transformers_sources,
        sidecar_writer,
    )
    fixture["provenance"]["transformers_version"] = transformers_version
    if sidecar_writer is None:
        _validate_fixture(fixture)
    return fixture


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "tiny.json",
        help=f"new JSON output under {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--layers", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--geometry",
        choices=("tiny", "qwen25-vl-7b"),
        default="tiny",
        help="fixed real Qwen2.5-VL-7B layer geometry; still uses synthetic weights",
    )
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--mask", default="1,1,1,0", help="one binary value per token")
    parser.add_argument("--position-offset", type=int, default=7)
    parser.add_argument("--rms-gain-mode", choices=RMS_GAIN_MODES, default="official_ones")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    requested_output = args.output.expanduser()
    if requested_output.is_symlink():
        raise SystemExit(f"refusing symlink fixture output: {requested_output}")
    output = requested_output.resolve()
    allowed_root = DEFAULT_OUTPUT_ROOT.resolve()
    try:
        output.relative_to(allowed_root)
    except ValueError as error:
        raise SystemExit(f"fixture output must be under {allowed_root}") from error
    if output.exists():
        raise SystemExit(f"refusing to replace existing fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = DEFAULT_OUTPUT_ROOT / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary_root)
    os.environ["TEMP"] = str(temporary_root)
    os.environ["TMP"] = str(temporary_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sidecar_writer = None
    try:
        if args.geometry == "qwen25-vl-7b":
            sidecar_path = output.with_name(f"{output.stem}.bf16le")
            if sidecar_path.exists() or sidecar_path.is_symlink():
                raise SystemExit(f"refusing to replace existing BF16 sidecar: {sidecar_path}")
            sidecar_writer = _BF16SidecarWriter(sidecar_path)
        fixture = _generate(args, sidecar_writer)
        if sidecar_writer is not None:
            fixture["sidecars"] = {"bf16": sidecar_writer.finish()}
            _validate_fixture(fixture, output)
        serialized = json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except BaseException:
        if sidecar_writer is not None:
            sidecar_writer.abort()
        raise
    summary = {"fixture": str(output), "schema": SCHEMA, "layers": len(fixture["layers"])}
    if "sidecars" in fixture:
        summary["sidecars"] = fixture["sidecars"]
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
