#!/usr/bin/env python3
"""Strictly compare optional source/native pre-QKV diagnostic companions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from . import hy_nomas_reference_block0_compare as main_compare
    from . import hy_nomas_reference_cache as source_oracle
except ImportError:  # direct import by repository test modules
    import hy_nomas_reference_block0_compare as main_compare
    import hy_nomas_reference_cache as source_oracle


SOURCE_SCHEMA = source_oracle.PREQKV_SCHEMA
NATIVE_SCHEMA = "cogni_ml.hy_nomas_reference_block0_preqkv_probe/v1"
SOURCE_SCOPE = source_oracle.PREQKV_SCOPE
NATIVE_SCOPE = (
    "diagnostic pre-QKV boundaries from one standalone native block-0 forward "
    "using authenticated native preblock inputs; not a complete upstream T0 handoff"
)
NATIVE_PREATTENTION_SCOPE = (
    "diagnostic pre-QKV boundaries from one standalone native block-0 pre-attention forward "
    "using authenticated native preblock inputs; not a complete upstream T0 handoff"
)
UPSTREAM_REVISION = main_compare.UPSTREAM_REVISION
NATIVE_SOURCE_REVISION = main_compare.NATIVE_IMPLEMENTATION_REVISION
NAMES = source_oracle.PREQKV_NAMES
SOURCE_KEYS = {
    "schema", "schema_version", "upstream_revision", "scope", "precision_control",
    "input_bundle_manifest_sha256", "input_bundle_payload_sha256",
    "reference_latent_artifact_sha256", "timeline_manifest_sha256",
    "valid_text_tokens", "weights_sha256", "parent_source_receipt_sha256",
    "parent_source_artifact_sha256", "tensor_artifact",
}
NATIVE_KEYS = {
    "schema", "schema_version", "native_revision", "source_revision", "scope",
    "precision_control", "input_bundle_manifest_sha256", "input_bundle_payload_sha256",
    "reference_latent_artifact_sha256", "timeline_manifest_sha256",
    "valid_text_tokens", "weights_sha256", "parent_native_receipt_sha256",
    "parent_native_artifact_sha256", "tensor_artifact",
}
NATIVE_PREATTENTION_KEYS = NATIVE_KEYS | {"bf16_linear_variant"}
_ARTIFACT_KEYS = {"basename", "sha256", "bytes", "dtype", "tensors"}
_SHA_FIELDS = (
    "input_bundle_manifest_sha256", "input_bundle_payload_sha256",
    "reference_latent_artifact_sha256", "timeline_manifest_sha256", "weights_sha256",
)


class PreQKVCompareError(ValueError):
    """A pre-QKV companion receipt, artifact, or numeric comparison is invalid."""


def _expected_source_precision(precision: str) -> tuple[str, str]:
    if precision == "BF16BoundaryControl":
        return "BF16BoundaryControl", "BF16"
    if precision == "F32Control":
        return "F32Control", "F32"
    raise PreQKVCompareError("precision must be BF16BoundaryControl or F32Control")


def _expected_source_boundary_dtypes(precision: str) -> dict[str, str]:
    _control, uniform_dtype = _expected_source_precision(precision)
    if precision == "BF16BoundaryControl":
        return {
            name: "BF16" if name in source_oracle.PREQKV_BF16_BOUNDARIES else "F32"
            for name in NAMES
        }
    return {name: uniform_dtype for name in NAMES}


def _expected_native_precision(precision: str, native_precision: str | None = None) -> str:
    if native_precision == main_compare.REFERENCE_BF16_PRECISION:
        if precision != "BF16BoundaryControl":
            raise PreQKVCompareError("reference-bf16-preattention requires BF16 source precision")
        return main_compare.REFERENCE_BF16_PRECISION
    if native_precision is not None:
        raise PreQKVCompareError("unsupported native precision admission")
    return {
        "BF16BoundaryControl": "BF16BoundaryControl",
        "F32Control": "F32Control",
    }[precision]


def _tensor_shapes(valid_text_tokens: int) -> dict[str, tuple[int, ...]]:
    try:
        return source_oracle._expected_preqkv_shapes(valid_text_tokens)
    except source_oracle.ReferenceCacheError as error:
        raise PreQKVCompareError(str(error)) from error


def _read_receipt(path: str | Path) -> dict[str, Any]:
    try:
        return main_compare._read_json(path)
    except main_compare.CompareError as error:
        raise PreQKVCompareError(str(error)) from error


def _validate_companion_receipt(
    receipt: dict[str, Any], *, native: bool, tensor_path: str | Path,
    parent_receipt_path: str | Path, parent_artifact_path: str | Path,
    precision: str, valid_text_tokens: int, identities: dict[str, str],
    native_precision: str | None = None, native_variant: str | None = None,
) -> main_compare.TensorArtifact:
    shapes = _tensor_shapes(valid_text_tokens)
    expected_precision, source_dtype = _expected_source_precision(precision)
    expected_keys = (
        (NATIVE_PREATTENTION_KEYS
         if native_precision == main_compare.REFERENCE_BF16_PRECISION else NATIVE_KEYS)
        if native else SOURCE_KEYS
    )
    if set(receipt) != expected_keys:
        raise PreQKVCompareError("pre-QKV receipt fields differ from its pinned schema")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise PreQKVCompareError("pre-QKV receipt schema version mismatch")
    expected_scope = (
        (NATIVE_PREATTENTION_SCOPE
         if native_precision == main_compare.REFERENCE_BF16_PRECISION else NATIVE_SCOPE)
        if native else SOURCE_SCOPE
    )
    if receipt.get("scope") != expected_scope:
        raise PreQKVCompareError("pre-QKV receipt scope mismatch")
    if native:
        if (receipt.get("schema") != NATIVE_SCHEMA
                or receipt.get("native_revision") != UPSTREAM_REVISION
                or receipt.get("source_revision") != NATIVE_SOURCE_REVISION
                or receipt.get("precision_control") != _expected_native_precision(precision, native_precision)):
            raise PreQKVCompareError("native pre-QKV receipt schema, revision, or precision mismatch")
        if native_precision == main_compare.REFERENCE_BF16_PRECISION:
            if (type(receipt.get("bf16_linear_variant")) is not str
                    or receipt["bf16_linear_variant"] not in main_compare.REFERENCE_BF16_VARIANTS
                    or receipt.get("bf16_linear_variant") != native_variant):
                raise PreQKVCompareError("native pre-QKV linear variant differs from its parent receipt")
        elif native_variant is not None:
            raise PreQKVCompareError("legacy native pre-QKV receipt has an unexpected linear variant")
        parent_receipt_field = "parent_native_receipt_sha256"
        parent_artifact_field = "parent_native_artifact_sha256"
        expected_artifact_dtype = "F32"
        expected_tensor_dtypes = {name: "F32" for name in NAMES}
    else:
        if (receipt.get("schema") != SOURCE_SCHEMA
                or receipt.get("upstream_revision") != UPSTREAM_REVISION
                or receipt.get("precision_control") != expected_precision):
            raise PreQKVCompareError("source pre-QKV receipt schema, revision, or precision mismatch")
        parent_receipt_field = "parent_source_receipt_sha256"
        parent_artifact_field = "parent_source_artifact_sha256"
        expected_tensor_dtypes = _expected_source_boundary_dtypes(precision)
        expected_artifact_dtype = (
            source_oracle.PREQKV_MIXED_ARTIFACT_DTYPE
            if len(set(expected_tensor_dtypes.values())) > 1
            else source_dtype
        )

    for field in _SHA_FIELDS:
        expected = identities[field]
        actual = receipt.get(field)
        if (not main_compare._is_sha256(expected)
                or type(actual) is not str or actual != expected):
            raise PreQKVCompareError(f"pre-QKV receipt identity mismatch: {field}")
    if type(receipt.get("valid_text_tokens")) is not int or receipt["valid_text_tokens"] != valid_text_tokens:
        raise PreQKVCompareError("pre-QKV receipt valid text extent mismatch")

    try:
        parent_receipt = main_compare._regular_file_path(
            parent_receipt_path, "pre-QKV parent receipt"
        )
        parent_artifact = main_compare._regular_file_path(
            parent_artifact_path, "pre-QKV parent artifact"
        )
        parent_receipt_sha = main_compare._hash_file(parent_receipt)
        parent_artifact_sha = main_compare._hash_file(parent_artifact)
    except main_compare.CompareError as error:
        raise PreQKVCompareError(str(error)) from error
    except (OSError, RuntimeError) as error:
        raise PreQKVCompareError("pre-QKV parent source/native output is unavailable") from error
    if (receipt.get(parent_receipt_field) != parent_receipt_sha
            or receipt.get(parent_artifact_field) != parent_artifact_sha):
        raise PreQKVCompareError("pre-QKV receipt does not bind its exact parent receipt and artifact")

    artifact_record = receipt.get("tensor_artifact")
    if (type(artifact_record) is not dict or set(artifact_record) != _ARTIFACT_KEYS
            or not main_compare._is_sha256(artifact_record.get("sha256"))
            or type(artifact_record.get("bytes")) is not int
            or artifact_record.get("dtype") != expected_artifact_dtype
            or type(artifact_record.get("tensors")) is not list):
        raise PreQKVCompareError("pre-QKV tensor artifact identity is invalid")
    try:
        caller_artifact = main_compare._regular_file_path(
            tensor_path, "CLI tensor artifact", require_absolute=False
        )
        artifact_path = main_compare._receipt_artifact_path(
            artifact_record, Path(tensor_path).expanduser(), "basename"
        )
        if artifact_path != caller_artifact:
            raise main_compare.CompareError(
                "receipt names a different tensor artifact from the caller-selected file"
            )
        artifact = main_compare.TensorArtifact(
            artifact_path,
            expected_sha256=artifact_record["sha256"],
            expected_names=NAMES,
        )
    except main_compare.CompareError as error:
        raise PreQKVCompareError(str(error)) from error
    if artifact_record["bytes"] != artifact.path.stat().st_size:
        raise PreQKVCompareError("pre-QKV tensor artifact byte count mismatch")
    if native:
        expected_metadata = {
            "format": NATIVE_SCHEMA,
            "native_revision": UPSTREAM_REVISION,
            "source_revision": NATIVE_SOURCE_REVISION,
            "precision_control": _expected_native_precision(precision),
            "parent_native_receipt_sha256": parent_receipt_sha,
            "parent_native_artifact_sha256": parent_artifact_sha,
        }
        if native_precision == main_compare.REFERENCE_BF16_PRECISION:
            expected_metadata["precision_control"] = main_compare.REFERENCE_BF16_PRECISION
            expected_metadata["bf16_linear_variant"] = native_variant or ""
    else:
        expected_metadata = {
            "format": SOURCE_SCHEMA,
            "upstream_revision": UPSTREAM_REVISION,
            "precision_control": expected_precision,
            "parent_source_receipt_sha256": parent_receipt_sha,
            "parent_source_artifact_sha256": parent_artifact_sha,
        }
    if artifact.metadata != expected_metadata:
        raise PreQKVCompareError("pre-QKV safetensors metadata does not bind its receipt and parent")

    records = artifact_record["tensors"]
    if [record.get("name") for record in records if type(record) is dict] != list(NAMES):
        raise PreQKVCompareError("pre-QKV tensor record order or inventory mismatch")
    if len(records) != len(NAMES):
        raise PreQKVCompareError("pre-QKV tensor record count mismatch")
    for name, record in zip(NAMES, records):
        expected_shape = list(shapes[name])
        expected_dtype = expected_tensor_dtypes[name]
        expected_bytes = math.prod(expected_shape) * (4 if expected_dtype == "F32" else 2)
        tensor = artifact.tensors[name]
        if (type(record) is not dict or set(record) != {"name", "dtype", "shape", "nbytes", "sha256"}
                or record.get("name") != name or record.get("dtype") != expected_dtype
                or type(record.get("shape")) is not list
                or any(type(extent) is not int for extent in record["shape"])
                or record.get("shape") != expected_shape or type(record.get("nbytes")) is not int
                or record["nbytes"] != expected_bytes or tensor["shape"] != tuple(expected_shape)
                or tensor["dtype"] != expected_dtype or record.get("sha256") != artifact.tensor_sha256(name)):
            raise PreQKVCompareError(f"pre-QKV receipt tensor record mismatch: {name}")
    return artifact


def compare_named_tensors(
    source: main_compare.TensorArtifact,
    native: main_compare.TensorArtifact,
    *, names: tuple[str, ...] = NAMES,
    atol: float = 2e-5, rtol: float = 2e-5,
    chunk_elements: int = 262_144,
) -> dict[str, Any]:
    """Compare a named sequence and identify its first failing boundary."""
    if tuple(names) != NAMES:
        raise PreQKVCompareError("pre-QKV comparison requires the exact eight-boundary sequence")
    results: list[dict[str, Any]] = []
    for name in names:
        if name not in source.tensors or name not in native.tensors:
            raise PreQKVCompareError(f"comparison tensor is missing: {name}")
        if source.tensors[name]["shape"] != native.tensors[name]["shape"]:
            raise PreQKVCompareError(f"source/native pre-QKV shape mismatch: {name}")
        try:
            result = main_compare.compare_tensor(
                native, name, [main_compare._tensor_ref(source, name)],
                atol=atol, rtol=rtol, chunk_elements=chunk_elements,
            )
        except main_compare.CompareError as error:
            raise PreQKVCompareError(str(error)) from error
        results.append(result)
    first = next((result["tensor"] for result in results if not result["passed"]), None)
    return {
        "tensor_count": len(results),
        "passed": all(result["passed"] for result in results),
        "first_differing_tensor": first,
        "tensors": results,
    }


def compare_companions(
    *, source_safetensors: str | Path, source_receipt_path: str | Path,
    native_safetensors: str | Path, native_receipt_path: str | Path,
    source_parent_safetensors: str | Path, source_parent_receipt_path: str | Path,
    native_parent_safetensors: str | Path, native_parent_receipt_path: str | Path,
    precision: str, valid_text_tokens: int, input_manifest_sha256: str,
    native_precision: str | None = None,
    input_payload_sha256: str, latent_artifact_sha256: str,
    timeline_manifest_sha256: str, weights_sha256: str = main_compare.BLOCK_WEIGHTS_SHA256,
    atol: float = 2e-5, rtol: float = 2e-5,
    chunk_elements: int = 262_144,
) -> dict[str, Any]:
    """Validate both companion lineage chains and compare all eight boundaries."""
    _expected_source_precision(precision)
    _expected_native_precision(precision, native_precision)
    identities = {
        "input_bundle_manifest_sha256": input_manifest_sha256,
        "input_bundle_payload_sha256": input_payload_sha256,
        "reference_latent_artifact_sha256": latent_artifact_sha256,
        "timeline_manifest_sha256": timeline_manifest_sha256,
        "weights_sha256": weights_sha256,
    }
    # The main artifact/receipt pair is authenticated first, then each companion
    # must bind to these exact on-disk parents.
    source_parent = _read_receipt(source_parent_receipt_path)
    native_parent = _read_receipt(native_parent_receipt_path)
    try:
        main_compare.validate_receipt_pair(
            source_parent, native_parent,
            source_path=source_parent_safetensors,
            native_path=native_parent_safetensors,
            precision=precision,
            native_precision=native_precision,
            valid_text_tokens=valid_text_tokens,
            input_manifest_sha256=input_manifest_sha256,
            input_payload_sha256=input_payload_sha256,
            latent_artifact_sha256=latent_artifact_sha256,
            timeline_manifest_sha256=timeline_manifest_sha256,
            weights_sha256=weights_sha256,
        )
        source_parent_artifact = main_compare.TensorArtifact(
            source_parent_safetensors,
            expected_sha256=source_parent["tensor_artifact"]["sha256"],
            expected_names=(*main_compare.PRE_CACHE_NAMES, "cache.k_bf16", "cache.v_bf16"),
        )
        native_parent_artifact = main_compare.TensorArtifact(
            native_parent_safetensors,
            expected_sha256=native_parent["tensor_artifact"]["sha256"],
            expected_names=main_compare.CAPTURE_NAMES,
        )
        main_compare.validate_tensor_artifact_records(
            source_parent, source_parent_artifact, native=False,
            valid_text_tokens=valid_text_tokens, precision=precision,
        )
        main_compare.validate_tensor_artifact_records(
            native_parent, native_parent_artifact, native=True,
            valid_text_tokens=valid_text_tokens, precision=precision,
        )
    except main_compare.CompareError as error:
        raise PreQKVCompareError(f"parent v1 validation failed: {error}") from error

    source_receipt = _read_receipt(source_receipt_path)
    native_receipt = _read_receipt(native_receipt_path)
    native_variant = (
        native_parent.get("bf16_linear_variant")
        if native_precision == main_compare.REFERENCE_BF16_PRECISION else None
    )
    source = _validate_companion_receipt(
        source_receipt, native=False, tensor_path=source_safetensors,
        parent_receipt_path=source_parent_receipt_path,
        parent_artifact_path=source_parent_safetensors,
        precision=precision, valid_text_tokens=valid_text_tokens, identities=identities,
        native_precision=native_precision,
    )
    native = _validate_companion_receipt(
        native_receipt, native=True, tensor_path=native_safetensors,
        parent_receipt_path=native_parent_receipt_path,
        parent_artifact_path=native_parent_safetensors,
        precision=precision, valid_text_tokens=valid_text_tokens, identities=identities,
        native_precision=native_precision, native_variant=native_variant,
    )

    numeric = compare_named_tensors(
        source, native, atol=atol, rtol=rtol, chunk_elements=chunk_elements,
    )
    return {
        "schema": "cogni_ml.hy_nomas_reference_block0_preqkv_comparison/v1",
        "precision_control": precision,
        "native_precision": native_precision or precision,
        **numeric,
        "source_artifact_sha256": source.sha256,
        "native_artifact_sha256": native.sha256,
        "source_parent_receipt_sha256": main_compare._hash_file(Path(source_parent_receipt_path)),
        "native_parent_receipt_sha256": main_compare._hash_file(Path(native_parent_receipt_path)),
        **({"bf16_linear_variant": native_variant} if native_variant is not None else {}),
        "identities": identities,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-safetensors", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--native-safetensors", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--source-parent-safetensors", type=Path, required=True)
    parser.add_argument("--source-parent-receipt", type=Path, required=True)
    parser.add_argument("--native-parent-safetensors", type=Path, required=True)
    parser.add_argument("--native-parent-receipt", type=Path, required=True)
    parser.add_argument("--precision", choices=("BF16BoundaryControl", "F32Control"), required=True)
    parser.add_argument(
        "--native-precision", choices=(main_compare.REFERENCE_BF16_PRECISION,), default=None,
        help="explicitly admit the reference-bf16-preattention native path",
    )
    parser.add_argument("--valid-text-tokens", type=int, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--input-payload-sha256", required=True)
    parser.add_argument("--latent-artifact-sha256", required=True)
    parser.add_argument("--timeline-manifest-sha256", required=True)
    parser.add_argument("--weights-sha256", default=main_compare.BLOCK_WEIGHTS_SHA256)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--chunk-elements", type=int, default=262_144)
    args = parser.parse_args(argv)
    result = compare_companions(
        source_safetensors=args.source_safetensors,
        source_receipt_path=args.source_receipt,
        native_safetensors=args.native_safetensors,
        native_receipt_path=args.native_receipt,
        source_parent_safetensors=args.source_parent_safetensors,
        source_parent_receipt_path=args.source_parent_receipt,
        native_parent_safetensors=args.native_parent_safetensors,
        native_parent_receipt_path=args.native_parent_receipt,
        precision=args.precision,
        native_precision=args.native_precision,
        valid_text_tokens=args.valid_text_tokens,
        input_manifest_sha256=args.input_manifest_sha256,
        input_payload_sha256=args.input_payload_sha256,
        latent_artifact_sha256=args.latent_artifact_sha256,
        timeline_manifest_sha256=args.timeline_manifest_sha256,
        weights_sha256=args.weights_sha256,
        atol=args.atol,
        rtol=args.rtol,
        chunk_elements=args.chunk_elements,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
