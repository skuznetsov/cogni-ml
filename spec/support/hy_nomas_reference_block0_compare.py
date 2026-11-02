#!/usr/bin/env python3
"""Compare saved source and native block-0 pre-cache captures numerically."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Iterable

try:
    from . import hy_nomas_reference_cache as cache_oracle
    from . import hy_nomas_reference as source_reference
except ImportError:  # direct import by repository test modules
    import hy_nomas_reference_cache as cache_oracle
    import hy_nomas_reference as source_reference


SOURCE_SCHEMA = "cogni_ml.hy_nomas_reference_block0_kv_oracle/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_reference_block0_probe/v1"
UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
NATIVE_IMPLEMENTATION_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
BLOCK_WEIGHTS_SHA256 = "01be7388fa4267f1f566c3ced405418f94d0ed2b4ee969cc880fb2518ab7c87b"
NATIVE_SCOPE = (
    "one standalone native block-0 forward using native-exported T0 preblock inputs; "
    "not a complete upstream T0 handoff"
)
NATIVE_CACHE_SEMANTICS = "prospective concatenated K/V before any continuation-history append"
IMAGE_TOKENS = 4320
HEADS = 16
HEAD_DIM = 128
CAPTURE_NAMES = (
    "img.q_projection", "img.k_projection", "img.v_projection",
    "img.q_qknorm", "img.k_qknorm", "img.q_rope", "img.k_rope",
    "txt.q_projection", "txt.k_projection", "txt.v_projection",
    "txt.q_qknorm", "txt.k_qknorm",
    "cache.k_native_f32", "cache.v_native_f32",
)
SOURCE_CACHE_NAMES = {"cache.k_native_f32": "cache.k_bf16", "cache.v_native_f32": "cache.v_bf16"}
PRE_CACHE_NAMES = CAPTURE_NAMES[:12]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HEADER_LIMIT = 16 * 1024 * 1024
MAX_ATOL = 2e-5
MAX_RTOL = 2e-5
PARAMETER_DTYPE_BASIS = [
    "inference.yaml sets dit.master_weight_type=bf16",
    "BasePipeline maps master_weight_type to torch.bfloat16 target_dtype",
    "BasePipeline moves the loaded transformer to target_dtype",
    "CausalAR casts reference latents to target_dtype before block inputs",
]
NATIVE_ACCOUNTING_KEYS = {
    "input_buffers", "block_weight_residents", "block_scratch", "mps_weight_stage",
    "compact_attention_buffers", "qk_rms_square_and_mean_buffers", "total_named_bytes",
    "opaque_mpsgraph_private_bytes",
}
NATIVE_EXECUTION_COUNTERS = {
    "encode_allocations", "encode_readbacks", "preterminal_waits", "terminal_waits",
}


class CompareError(ValueError):
    """A receipt, tensor artifact, or numerical comparison is invalid."""


def _regular_file_path(path: str | Path, description: str, *,
                       require_absolute: bool = True) -> Path:
    """Admit a regular file without first resolving away a symlink leaf."""
    source = Path(path).expanduser()
    if require_absolute and not source.is_absolute():
        raise CompareError(f"{description} must be an absolute regular non-symlink file")
    try:
        if source.is_symlink() or not source.is_file():
            raise CompareError(f"{description} must be an absolute regular non-symlink file")
        return source.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CompareError(f"{description} must be an absolute regular non-symlink file") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompareError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise CompareError("receipt must be an absolute regular non-symlink file")
    raw = source.read_bytes()
    if len(raw) > 1024 * 1024:
        raise CompareError("receipt exceeds the 1 MiB limit")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompareError("receipt JSON is invalid") from error
    if type(document) is not dict:
        raise CompareError("receipt must be a JSON object")
    return document


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TensorArtifact:
    """Validated safetensors ranges with identity checks around streamed reads."""

    def __init__(self, path: str | Path, *, expected_sha256: str,
                 expected_names: Iterable[str]):
        if not _is_sha256(expected_sha256):
            raise CompareError("tensor artifact SHA-256 is malformed")
        self.path = Path(path).expanduser()
        if (not self.path.is_absolute() or self.path.is_symlink()
                or not self.path.is_file()):
            raise CompareError("tensor artifact must be an absolute regular non-symlink file")
        self.path = self.path.resolve(strict=True)
        self._identity = self._stat_identity()
        if self._identity[2] < 8:
            raise CompareError("safetensors artifact is truncated")
        actual_sha = _hash_file(self.path)
        if actual_sha != expected_sha256:
            raise CompareError("tensor artifact SHA-256 differs from receipt or caller pin")
        self.sha256 = actual_sha
        self.tensors: dict[str, dict[str, Any]] = {}
        with self.path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise CompareError("safetensors header prefix is truncated")
            header_bytes = int.from_bytes(prefix, "little", signed=False)
            if header_bytes < 2 or header_bytes > _HEADER_LIMIT or 8 + header_bytes > self._identity[2]:
                raise CompareError("safetensors header extent is invalid")
            raw_header = stream.read(header_bytes)
            try:
                header = json.loads(raw_header.decode("utf-8").rstrip(" "),
                                    object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CompareError("safetensors header JSON is invalid") from error
            if type(header) is not dict:
                raise CompareError("safetensors header must be a JSON object")
            metadata = header.pop("__metadata__", {})
            if type(metadata) is not dict or any(
                    type(key) is not str or type(value) is not str
                    for key, value in metadata.items()):
                raise CompareError("safetensors metadata must map strings to strings")
            names = set(header)
            expected = set(expected_names)
            if names != expected:
                raise CompareError(
                    f"safetensors tensor names differ; missing={sorted(expected - names)}, "
                    f"extra={sorted(names - expected)}"
                )
            self.metadata = metadata
            self.data_offset = 8 + header_bytes
            ranges: list[tuple[int, int, str]] = []
            for name, record in header.items():
                if type(name) is not str or type(record) is not dict or set(record) != {
                        "dtype", "shape", "data_offsets"}:
                    raise CompareError(f"invalid safetensors record: {name}")
                dtype = record["dtype"]
                shape = record["shape"]
                offsets = record["data_offsets"]
                if dtype not in ("F32", "BF16") or type(dtype) is not str:
                    raise CompareError(f"unsupported safetensors dtype: {name}")
                if (type(shape) is not list or not shape
                        or any(type(extent) is not int or extent <= 0 for extent in shape)):
                    raise CompareError(f"invalid safetensors shape: {name}")
                if (type(offsets) is not list or len(offsets) != 2
                        or any(type(offset) is not int for offset in offsets)
                        or offsets[0] < 0 or offsets[1] < offsets[0]):
                    raise CompareError(f"invalid safetensors data range: {name}")
                count = math.prod(shape)
                expected_bytes = count * (4 if dtype == "F32" else 2)
                if offsets[1] - offsets[0] != expected_bytes:
                    raise CompareError(f"safetensors byte extent does not match shape: {name}")
                self.tensors[name] = {
                    "dtype": dtype, "shape": tuple(shape), "count": count,
                    "offset": offsets[0], "nbytes": expected_bytes,
                }
                ranges.append((offsets[0], offsets[1], name))
            ranges.sort()
            cursor = 0
            for start, end, name in ranges:
                if start != cursor:
                    raise CompareError(f"safetensors data ranges overlap or have gaps: {name}")
                cursor = end
            if self.data_offset + cursor != self._identity[2]:
                raise CompareError("safetensors ranges do not cover the complete data section")
        self.check_identity()

    def _stat_identity(self) -> tuple[int, int, int, int, int]:
        info = self.path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise CompareError("tensor artifact is not a regular file")
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def check_identity(self) -> None:
        if self._stat_identity() != self._identity:
            raise CompareError("tensor artifact changed while being compared")

    def read_values(self, name: str, start: int, count: int):
        import numpy as np

        self.check_identity()
        record = self.tensors[name]
        if start < 0 or count < 0 or start + count > record["count"]:
            raise CompareError(f"tensor read is outside the admitted range: {name}")
        itemsize = 4 if record["dtype"] == "F32" else 2
        byte_offset = self.data_offset + record["offset"] + start * itemsize
        byte_count = count * itemsize
        with self.path.open("rb") as stream:
            stream.seek(byte_offset)
            raw = stream.read(byte_count)
        self.check_identity()
        if len(raw) != byte_count:
            raise CompareError(f"safetensors tensor range was truncated: {name}")
        if record["dtype"] == "F32":
            return np.frombuffer(raw, dtype="<f4")
        words = np.frombuffer(raw, dtype="<u2").astype("<u4")
        return np.left_shift(words, 16).view("<f4")

    def tensor_sha256(self, name: str) -> str:
        digest = hashlib.sha256()
        record = self.tensors[name]
        byte_offset = self.data_offset + record["offset"]
        remaining = record["nbytes"]
        with self.path.open("rb") as stream:
            stream.seek(byte_offset)
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise CompareError(f"safetensors tensor range was truncated: {name}")
                digest.update(chunk)
                remaining -= len(chunk)
        self.check_identity()
        return digest.hexdigest()


def _tensor_ref(artifact: TensorArtifact, name: str) -> tuple[TensorArtifact, str, int, int]:
    record = artifact.tensors[name]
    return artifact, name, 0, record["count"]


def _read_expected_chunk(segments: list[tuple[TensorArtifact, str, int, int]],
                         cursor: list[int], count: int):
    import numpy as np

    pieces = []
    remaining = count
    while remaining:
        segment_index, segment_offset = cursor
        if segment_index >= len(segments):
            raise CompareError("comparison source segments ended before the candidate tensor")
        artifact, name, _start, segment_count = segments[segment_index]
        available = segment_count - segment_offset
        take = min(remaining, available)
        pieces.append(artifact.read_values(name, segment_offset, take))
        remaining -= take
        segment_offset += take
        if segment_offset == segment_count:
            cursor[0] += 1
            cursor[1] = 0
        else:
            cursor[1] = segment_offset
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def compare_tensor(candidate: TensorArtifact, candidate_name: str,
                   expected_segments: list[tuple[TensorArtifact, str, int, int]], *,
                   atol: float = 2e-5, rtol: float = 2e-5,
                   chunk_elements: int = 262_144) -> dict[str, Any]:
    """Compare one candidate tensor against one or more flattened source ranges."""
    import numpy as np

    if (type(atol) not in (int, float) or type(rtol) not in (int, float)
            or not math.isfinite(atol) or not math.isfinite(rtol)
            or atol < 0 or rtol < 0 or atol > MAX_ATOL or rtol > MAX_RTOL
            or type(chunk_elements) is not int or chunk_elements <= 0):
        raise CompareError(
            "comparison tolerances must be finite, nonnegative, and no wider than 2e-5; "
            "chunk extent must be a positive integer"
        )
    if candidate.tensors[candidate_name]["dtype"] != "F32":
        raise CompareError("native comparison artifact tensors must be F32 containers")
    candidate_count = candidate.tensors[candidate_name]["count"]
    if sum(segment[3] for segment in expected_segments) != candidate_count:
        raise CompareError(f"source and candidate element counts differ: {candidate_name}")
    cursor = [0, 0]
    compared = 0
    max_abs = 0.0
    squared_error_sum = 0.0
    mismatch_count = 0
    while compared < candidate_count:
        amount = min(chunk_elements, candidate_count - compared)
        actual = candidate.read_values(candidate_name, compared, amount)
        expected = _read_expected_chunk(expected_segments, cursor, amount)
        if not bool(np.isfinite(actual).all()) or not bool(np.isfinite(expected).all()):
            raise CompareError(f"non-finite value in numerical capture: {candidate_name}")
        difference = np.abs(actual - expected)
        if difference.size:
            max_abs = max(max_abs, float(difference.max()))
        squared_error_sum += float(np.square(difference, dtype=np.float64).sum(dtype=np.float64))
        threshold = atol + rtol * np.abs(expected)
        mismatch_count += int(np.count_nonzero(difference > threshold))
        compared += amount
    rms = math.sqrt(squared_error_sum / candidate_count) if candidate_count else 0.0
    return {
        "tensor": candidate_name,
        "elements": candidate_count,
        "max_abs_error": max_abs,
        "rms_error": rms,
        "mismatch_count": mismatch_count,
        "atol": atol,
        "rtol": rtol,
        "passed": mismatch_count == 0,
    }


def _same_shape(left: TensorArtifact, left_name: str,
                right: TensorArtifact, right_name: str) -> None:
    if left.tensors[left_name]["shape"] != right.tensors[right_name]["shape"]:
        raise CompareError(f"capture tensor shape mismatch: {left_name} / {right_name}")


def compare_capture_artifacts(source: TensorArtifact, candidate: TensorArtifact, *,
                              precision: str, valid_text_tokens: int,
                              atol: float = 2e-5, rtol: float = 2e-5,
                              chunk_elements: int = 262_144) -> dict[str, Any]:
    """Compare all stage captures; use source pre-cache stages for F32 cache joins."""
    if precision not in ("BF16BoundaryControl", "F32Control"):
        raise CompareError("candidate precision must be BF16BoundaryControl or F32Control")
    if type(valid_text_tokens) is not int or not 1 <= valid_text_tokens <= 1280:
        raise CompareError("valid_text_tokens must be an integer in [1, 1280]")
    if set(candidate.tensors) != set(CAPTURE_NAMES):
        raise CompareError("native capture artifact tensor inventory is not exact")
    expected_source_names = set(CAPTURE_NAMES[:12]) | {"cache.k_bf16", "cache.v_bf16"}
    if set(source.tensors) != expected_source_names:
        raise CompareError("source oracle artifact tensor inventory is not exact")
    image_shape = (1, IMAGE_TOKENS, HEADS, HEAD_DIM)
    text_shape = (1, valid_text_tokens, HEADS, HEAD_DIM)
    cache_shape = (1, IMAGE_TOKENS + valid_text_tokens, HEADS, HEAD_DIM)
    for name in CAPTURE_NAMES:
        if name.startswith("img."):
            wanted_shape = image_shape
        elif name.startswith("txt."):
            wanted_shape = text_shape
        else:
            wanted_shape = cache_shape
        if candidate.tensors[name]["shape"] != wanted_shape:
            raise CompareError(f"native capture shape differs from authenticated extent: {name}")
        if candidate.tensors[name]["dtype"] != "F32":
            raise CompareError(f"native capture must be an F32 artifact: {name}")
    for name in CAPTURE_NAMES[:12]:
        if source.tensors[name]["shape"] != candidate.tensors[name]["shape"]:
            raise CompareError(f"source/native capture shape mismatch: {name}")
        wanted_dtype = "BF16" if precision == "BF16BoundaryControl" else "F32"
        if source.tensors[name]["dtype"] != wanted_dtype:
            raise CompareError(
                f"source capture dtype does not match {precision}: {name}"
            )
    for native_name, source_name in SOURCE_CACHE_NAMES.items():
        if source.tensors[source_name]["shape"] != cache_shape:
            raise CompareError(f"source cache shape differs from valid append extent: {source_name}")
        if source.tensors[source_name]["dtype"] != "BF16":
            raise CompareError(f"source cache capture must be BF16: {source_name}")

    results: list[dict[str, Any]] = []
    for name in PRE_CACHE_NAMES:
        results.append(compare_tensor(
            candidate, name, [_tensor_ref(source, name)], atol=atol, rtol=rtol,
            chunk_elements=chunk_elements,
        ))

    if precision == "BF16BoundaryControl":
        for native_name, source_name in SOURCE_CACHE_NAMES.items():
            results.append(compare_tensor(
                candidate, native_name, [_tensor_ref(source, source_name)],
                atol=atol, rtol=rtol, chunk_elements=chunk_elements,
            ))
    else:
        # F32Control source cache tensors are already rounded to BF16 by the
        # upstream cache writer. Compare native F32 prospective K/V to their
        # upstream pre-storage constituents instead of mixing storage points.
        segments_by_cache = {
            "cache.k_native_f32": [
                _tensor_ref(source, "img.k_rope"),
                _tensor_ref(source, "txt.k_qknorm"),
            ],
            "cache.v_native_f32": [
                _tensor_ref(source, "img.v_projection"),
                _tensor_ref(source, "txt.v_projection"),
            ],
        }
        for native_name, segments in segments_by_cache.items():
            results.append(compare_tensor(
                candidate, native_name, segments, atol=atol, rtol=rtol,
                chunk_elements=chunk_elements,
            ))

    source.check_identity()
    candidate.check_identity()
    return {
        "precision": precision,
        "valid_text_tokens": valid_text_tokens,
        "atol": atol,
        "rtol": rtol,
        "tensor_count": len(results),
        "passed": all(result["passed"] for result in results),
        "tensors": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-safetensors", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--native-safetensors", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--precision", choices=("BF16BoundaryControl", "F32Control"), required=True)
    parser.add_argument("--valid-text-tokens", type=int, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--input-payload-sha256", required=True)
    parser.add_argument("--latent-artifact-sha256", required=True)
    parser.add_argument("--timeline-manifest-sha256", required=True)
    parser.add_argument("--weights-sha256", default=BLOCK_WEIGHTS_SHA256)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--chunk-elements", type=int, default=262_144)
    args = parser.parse_args(argv)
    # Receipt validation is implemented separately from tensor comparison so
    # numeric results can never be inherited from a producer's pass flag.
    source_receipt = _read_json(args.source_receipt)
    native_receipt = _read_json(args.native_receipt)
    validate_receipt_pair(
        source_receipt, native_receipt,
        source_path=args.source_safetensors, native_path=args.native_safetensors,
        precision=args.precision, valid_text_tokens=args.valid_text_tokens,
        input_manifest_sha256=args.input_manifest_sha256,
        input_payload_sha256=args.input_payload_sha256,
        latent_artifact_sha256=args.latent_artifact_sha256,
        timeline_manifest_sha256=args.timeline_manifest_sha256,
        weights_sha256=args.weights_sha256,
    )
    source_sha = source_receipt["tensor_artifact"]["sha256"]
    native_sha = native_receipt["tensor_artifact"]["sha256"]
    source = TensorArtifact(args.source_safetensors, expected_sha256=source_sha,
                            expected_names=(*PRE_CACHE_NAMES, "cache.k_bf16", "cache.v_bf16"))
    candidate = TensorArtifact(args.native_safetensors, expected_sha256=native_sha,
                               expected_names=CAPTURE_NAMES)
    validate_tensor_artifact_records(source_receipt, source, native=False,
                                     valid_text_tokens=args.valid_text_tokens,
                                     precision=args.precision)
    validate_tensor_artifact_records(native_receipt, candidate, native=True,
                                     valid_text_tokens=args.valid_text_tokens,
                                     precision=args.precision)
    result = compare_capture_artifacts(
        source, candidate, precision=args.precision,
        valid_text_tokens=args.valid_text_tokens, atol=args.atol, rtol=args.rtol,
        chunk_elements=args.chunk_elements,
    )
    result.update({
        "source_artifact_sha256": source.sha256,
        "native_artifact_sha256": candidate.sha256,
        "input_manifest_sha256": args.input_manifest_sha256,
        "input_payload_sha256": args.input_payload_sha256,
        "latent_artifact_sha256": args.latent_artifact_sha256,
        "timeline_manifest_sha256": args.timeline_manifest_sha256,
        "weights_sha256": args.weights_sha256,
    })
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


def validate_receipt_pair(source: dict[str, Any], native: dict[str, Any], *,
                          source_path: str | Path, native_path: str | Path,
                          precision: str, valid_text_tokens: int,
                          input_manifest_sha256: str, input_payload_sha256: str,
                          latent_artifact_sha256: str, timeline_manifest_sha256: str,
                          weights_sha256: str) -> None:
    """Bind source/native numeric artifacts to the same admitted inputs."""
    for label, digest in (
            ("input manifest", input_manifest_sha256),
            ("input payload", input_payload_sha256),
            ("latent artifact", latent_artifact_sha256),
            ("timeline manifest", timeline_manifest_sha256),
            ("block weights", weights_sha256)):
        if not _is_sha256(digest):
            raise CompareError(f"{label} SHA-256 is malformed")
    if weights_sha256 != BLOCK_WEIGHTS_SHA256:
        raise CompareError("comparison must use the pinned indexed BF16 block-0 weights")
    expected_source_keys = {
        "schema", "operation", "upstream_revision", "upstream_source_root",
        "upstream_source_file_sha256", "upstream_dependency_sha256",
        "inference_dtype_source_sha256", "weight_source", "parameter_dtype_basis",
        "precision", "scope", "cache_append", "input_bundle", "tensor_artifact",
    }
    if set(source) != expected_source_keys:
        raise CompareError("source receipt fields differ from the pinned oracle schema")
    # Source receipt contract is emitted by hy_nomas_reference_cache.py.
    if source.get("schema") != SOURCE_SCHEMA or source.get("operation") != "upstream_block0_first_chunk_kv_cache_append":
        raise CompareError("source receipt schema or operation mismatch")
    if source.get("upstream_source_file_sha256") != "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee":
        raise CompareError("source receipt does not bind the pinned block-0 source file")
    source_mode = {"BF16BoundaryControl": "bf16", "F32Control": "f32-control"}[precision]
    source_precision = source.get("precision")
    if type(source_precision) is not dict or source_precision.get("mode") != source_mode:
        raise CompareError("source receipt precision does not match the requested comparison")
    if source.get("upstream_revision") != UPSTREAM_REVISION:
        raise CompareError("source receipt is not from the pinned upstream revision")
    if source.get("upstream_dependency_sha256") != source_reference.PINNED_SOURCE_SHA256:
        raise CompareError("source receipt upstream dependency hashes differ from pinned source")
    if source.get("inference_dtype_source_sha256") != cache_oracle.INFERENCE_DTYPE_SOURCE_SHA256:
        raise CompareError("source receipt inference dtype source hashes differ from pinned source")
    if source.get("parameter_dtype_basis") != PARAMETER_DTYPE_BASIS:
        raise CompareError("source receipt parameter dtype basis differs from pinned inference path")
    source_weights = source.get("weight_source")
    if (type(source_weights) is not dict or source_weights.get("sha256") != weights_sha256
            or source_weights.get("source_dtype") != "BF16"
            or source_weights.get("tensor_prefix") != "double_blocks.0."
            or source_weights.get("tensor_count") != 32):
        raise CompareError("source receipt block weight identity mismatch")
    expected_source_param_dtype = "BF16" if precision == "BF16BoundaryControl" else "F32"
    if source_weights.get("parameter_dtype") != expected_source_param_dtype:
        raise CompareError("source receipt parameter dtype differs from requested control")
    source_input = source.get("input_bundle")
    if type(source_input) is not dict:
        raise CompareError("source receipt has no authenticated input bundle identity")
    source_identity = {
        "manifest_sha256": input_manifest_sha256,
        "payload_sha256": input_payload_sha256,
        "reference_latent_artifact_sha256": latent_artifact_sha256,
        "timeline_manifest_sha256": timeline_manifest_sha256,
        "valid_text_tokens": valid_text_tokens,
    }
    for field, expected in source_identity.items():
        actual = source_input.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise CompareError(f"source receipt input identity mismatch: {field}")
    cache_append = source.get("cache_append")
    if (type(cache_append) is not dict
            or type(cache_append.get("cache_valid_len")) is not int
            or cache_append["cache_valid_len"] != 0
            or cache_append.get("update_cache") is not True
            or type(cache_append.get("cache_tokens")) is not int
            or cache_append.get("cache_tokens") != IMAGE_TOKENS + valid_text_tokens
            or cache_append.get("attention_executed") is not False
            or cache_append.get("cache_tensor_order") != "image tokens followed by valid text prefix"):
        raise CompareError("source receipt does not describe the admitted first cache append boundary")
    source_artifact = source.get("tensor_artifact")
    source_names = (*PRE_CACHE_NAMES, "cache.k_bf16", "cache.v_bf16")
    if (type(source_artifact) is not dict or set(source_artifact) != {
            "path", "sha256", "tensor_count", "tensors"}
            or not _is_sha256(source_artifact.get("sha256"))
            or type(source_artifact.get("tensor_count")) is not int
            or source_artifact["tensor_count"] != len(source_names)
            or type(source_artifact.get("tensors")) is not dict
            or set(source_artifact["tensors"]) != set(source_names)):
        raise CompareError("source receipt tensor artifact identity is invalid")
    if _receipt_artifact_path(source_artifact, Path(source_path), "path") != Path(source_path).resolve(strict=True):
        raise CompareError("source receipt names a different tensor artifact")

    # Native receipt schema is deliberately strict about scope and identity;
    # no producer-side numerical pass claims are consumed.
    expected_native_keys = {
        "schema", "schema_version", "native_revision", "source_revision", "scope",
        "full_upstream_t0_preprocessing_parity", "history_cache_write_executed",
        "cache_tensor_semantics", "input_bundle_manifest_sha256",
        "input_bundle_payload_sha256", "reference_latent_artifact_sha256",
        "timeline_manifest_sha256", "valid_text_tokens", "text_storage_tokens",
        "text_padding_rows", "text_padding_value", "weights_sha256",
        "precision_control", "tensor_artifact", "named_buffer_accounting",
        "execution_counters",
    }
    if set(native) != expected_native_keys:
        raise CompareError("native receipt fields differ from the probe schema")
    if native.get("schema") != NATIVE_SCHEMA:
        raise CompareError("native receipt schema mismatch")
    if type(native.get("schema_version")) is not int or native["schema_version"] != 1:
        raise CompareError("native receipt schema version mismatch")
    if (native.get("native_revision") != UPSTREAM_REVISION
            or native.get("source_revision") != NATIVE_IMPLEMENTATION_REVISION):
        raise CompareError("native receipt revision identity mismatch")
    if (native.get("scope") != NATIVE_SCOPE
            or native.get("full_upstream_t0_preprocessing_parity") is not False):
        raise CompareError("native receipt overstates the upstream preprocessing scope")
    native_precision = {
        "BF16BoundaryControl": "bf16-boundary-control",
        "F32Control": "f32-control",
    }[precision]
    if native.get("precision_control") != native_precision:
        raise CompareError("native receipt precision control mismatch")
    if native.get("history_cache_write_executed") is not False:
        raise CompareError("native receipt does not identify prospective cache values only")
    if native.get("cache_tensor_semantics") != NATIVE_CACHE_SEMANTICS:
        raise CompareError("native receipt cache tensor semantics mismatch")
    native_identity = {
        "input_bundle_manifest_sha256": input_manifest_sha256,
        "input_bundle_payload_sha256": input_payload_sha256,
        "reference_latent_artifact_sha256": latent_artifact_sha256,
        "timeline_manifest_sha256": timeline_manifest_sha256,
        "valid_text_tokens": valid_text_tokens,
        "text_storage_tokens": 1280,
        "weights_sha256": weights_sha256,
    }
    for field, expected in native_identity.items():
        actual = native.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise CompareError(f"native receipt input identity mismatch: {field}")
    if (type(native.get("text_padding_rows")) is not int
            or native["text_padding_rows"] != 1280 - valid_text_tokens
            or type(native.get("text_padding_value")) is not int
            or native["text_padding_value"] != 0):
        raise CompareError("native receipt text padding contract mismatch")
    accounting = native.get("named_buffer_accounting")
    if type(accounting) is not dict or set(accounting) != NATIVE_ACCOUNTING_KEYS:
        raise CompareError("native receipt named buffer accounting fields are malformed")
    accounting_fields = NATIVE_ACCOUNTING_KEYS - {"opaque_mpsgraph_private_bytes"}
    if (any(type(accounting.get(name)) is not int or accounting[name] < 0
            for name in accounting_fields)
            or accounting["total_named_bytes"] != sum(
                accounting[name] for name in accounting_fields - {"total_named_bytes"}
            )
            or accounting.get("opaque_mpsgraph_private_bytes")
            != "not measured; excluded from named Metal-buffer total"):
        raise CompareError("native receipt named buffer totals are inconsistent")
    counters = native.get("execution_counters")
    if type(counters) is not dict or set(counters) != NATIVE_EXECUTION_COUNTERS:
        raise CompareError("native receipt execution counter fields are malformed")
    if any(type(counters.get(name)) is not int or counters[name] < 0 for name in NATIVE_EXECUTION_COUNTERS):
        raise CompareError("native receipt execution counters must be nonnegative integers")
    if (counters["encode_allocations"] != 0 or counters["encode_readbacks"] != 0
            or counters["preterminal_waits"] != 0 or counters["terminal_waits"] != 1):
        raise CompareError("native block-0 execution counters violate terminal-drain contract")
    native_artifact = native.get("tensor_artifact")
    if (type(native_artifact) is not dict or set(native_artifact) != {
            "basename", "sha256", "bytes", "dtype", "tensors"}
            or not _is_sha256(native_artifact.get("sha256"))
            or type(native_artifact.get("bytes")) is not int
            or native_artifact.get("dtype") != "F32"
            or type(native_artifact.get("tensors")) is not list):
        raise CompareError("native receipt tensor artifact identity is invalid")
    if _receipt_artifact_path(native_artifact, Path(native_path), "basename") != Path(native_path).resolve(strict=True):
        raise CompareError("native receipt names a different tensor artifact")


def validate_tensor_artifact_records(receipt: dict[str, Any], artifact: TensorArtifact, *,
                                     native: bool, valid_text_tokens: int,
                                     precision: str) -> None:
    """Check per-tensor receipt metadata against the actual artifact bytes."""
    image_shape = [1, IMAGE_TOKENS, HEADS, HEAD_DIM]
    text_shape = [1, valid_text_tokens, HEADS, HEAD_DIM]
    cache_shape = [1, IMAGE_TOKENS + valid_text_tokens, HEADS, HEAD_DIM]
    source_artifact = receipt["tensor_artifact"]
    if native:
        if source_artifact["bytes"] != artifact.path.stat().st_size:
            raise CompareError("native artifact byte count differs from its receipt")
        records = source_artifact["tensors"]
        if [record.get("name") for record in records if type(record) is dict] != list(CAPTURE_NAMES):
            raise CompareError("native receipt tensor order or inventory mismatch")
        if len(records) != len(CAPTURE_NAMES):
            raise CompareError("native receipt tensor inventory length mismatch")
        normalized = {record["name"]: record for record in records}
    else:
        normalized = source_artifact["tensors"]
        if set(normalized) != set((*PRE_CACHE_NAMES, "cache.k_bf16", "cache.v_bf16")):
            raise CompareError("source receipt tensor order or inventory mismatch")

    for name, tensor in artifact.tensors.items():
        if name.startswith("img."):
            expected_shape = image_shape
        elif name.startswith("txt."):
            expected_shape = text_shape
        else:
            expected_shape = cache_shape
        record = normalized[name]
        if native:
            required_record_keys = {"name", "shape", "nbytes", "sha256"}
            expected_dtype = "F32"
        else:
            required_record_keys = {"dtype", "shape", "nbytes", "sha256"}
            expected_dtype = "BF16" if name.endswith("bf16") or name.endswith("_bf16") else (
                "F32" if precision == "F32Control" else "BF16"
            )
        if type(record) is not dict or set(record) != required_record_keys:
            raise CompareError(f"receipt tensor record fields are invalid: {name}")
        if (tensor["shape"] != tuple(expected_shape)
                or type(record.get("shape")) is not list
                or any(type(extent) is not int for extent in record["shape"])
                or record.get("shape") != expected_shape
                or type(record.get("nbytes")) is not int
                or record["nbytes"] != tensor["nbytes"]
                or not _is_sha256(record.get("sha256"))
                or tensor["dtype"] != expected_dtype):
            raise CompareError(f"receipt tensor shape, dtype, extent, or hash mismatch: {name}")
        if not native and record.get("dtype") != expected_dtype:
            raise CompareError(f"source receipt dtype mismatch: {name}")
        if artifact.tensor_sha256(name) != record["sha256"]:
            raise CompareError(f"receipt tensor SHA-256 mismatch: {name}")
    artifact.check_identity()


def _receipt_artifact_path(artifact: dict[str, Any], cli_path: Path, field: str) -> Path:
    recorded = artifact.get(field)
    if type(recorded) is not str or not recorded:
        raise CompareError("receipt tensor artifact path is missing")
    cli_file = _regular_file_path(cli_path, "CLI tensor artifact", require_absolute=False)
    path = Path(recorded)
    if field == "basename" and (
            path.is_absolute() or path.name != recorded or "/" in recorded or "\\" in recorded
            or recorded in (".", "..")):
        raise CompareError("native receipt tensor artifact must use a safe basename")
    if not path.is_absolute():
        path = cli_file.parent / path
    return _regular_file_path(path, "receipt tensor artifact")


if __name__ == "__main__":
    raise SystemExit(main())
