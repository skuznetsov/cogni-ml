#!/usr/bin/env python3
"""Compare an isolated active-prefix BF16 attention invocation.

The source run executes only the pinned ``attn_mode == 'torch'`` branch body on
already-cropped 4849-row fixtures. Its Q comes from the prior trusted BF16
preattention fixture; its K/V come from that fixture's physical first-T0 cache
contents, independently replayed and recorded by the source-history helper.
This does not execute the enclosing function's padding or unpadding logic, claim
the padded Torch or Flash3 geometry, or establish full T0 preprocessing parity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable

try:
    from . import hy_nomas_reference_block0_compare as block0_compare
    from . import hy_nomas_reference_history as history
except ImportError:  # direct import by repository test modules
    import hy_nomas_reference_block0_compare as block0_compare
    import hy_nomas_reference_history as history


SOURCE_SCHEMA = "cogni_ml.hy_nomas_reference_attention_source/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_reference_attention_probe/v1"
COMPARISON_SCHEMA = "cogni_ml.hy_nomas_reference_attention_comparison/v1"
SCHEMA_VERSION = 1
UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
UPSTREAM_SOURCE_FILE_SHA256 = history.UPSTREAM_SOURCE_FILE_SHA256
CACHE_FUNCTION_SOURCE_SHA256 = history.CACHE_FUNCTION_SOURCE_SHA256
TORCH_BRANCH_BODY_AST_SHA256 = "973b9ed2905efb3ffe1a2dbff352b47b7f12a71e26f55599d7f1e8241f08fb85"
ATOL = 2e-5
RTOL = 2e-5
IMAGE_TOKENS = 4320
VALID_TEXT_TOKENS = 529
TOKENS = IMAGE_TOKENS + VALID_TEXT_TOKENS
HEADS = 16
HEAD_DIM = 128
ELEMENTS = TOKENS * HEADS * HEAD_DIM
OUTPUT_SHAPE = (1, TOKENS, HEADS, HEAD_DIM)
LAYOUT = "[1,tokens,heads,head_dim]; image then valid text"
INPUT_TENSOR_NAMES = (
    "attention.q_bf16", "attention.k_stored_bf16", "attention.v_stored_bf16",
)
TENSOR_NAMES = (*INPUT_TENSOR_NAMES, "attention.output_bf16")
SOURCE_SCOPE = (
    "isolated active-prefix invocation of the pinned upstream torch attention branch on MPS using "
    "trusted block-0 BF16 Q and physical first-T0 BF16 stored K/V; 4849 active rows; no padded "
    "Torch, Flash3, cache append, or full upstream T0 handoff parity"
)
NATIVE_SCOPE = (
    "one native block-0 active-prefix attention operation using the same Q and physical first-T0 "
    "BF16 stored K/V; 4849 active rows; no padded Torch, Flash3, cache append, full upstream T0 "
    "handoff, rest block, other layers, T1, or generation parity"
)
TRUSTED_FIXTURE_PROVENANCE = (
    "Q is assembled from the prior pinned source block-0 BF16 preattention fixture; "
    "its receipt and artifact hashes are bound here, but this attention oracle does "
    "not independently attest how that Q fixture was produced"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


class AttentionProbeError(ValueError):
    """An attention input, receipt, artifact, or numerical comparison is invalid."""


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _matches_exact_json(value: object, expected: object) -> bool:
    """Compare receipt fields without JSON's bool/int equality loophole."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _matches_exact_json(value[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _matches_exact_json(actual, wanted)
            for actual, wanted in zip(value, expected)
        )
    return value == expected


def _source_branch_match(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (
        isinstance(test.left, ast.Name)
        and test.left.id == "attn_mode"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "torch"
    )


def _load_torch_attention_branch(
    upstream_root: str | Path = history.cache_oracle.UPSTREAM_ROOT_DEFAULT,
) -> tuple[ast.Module, dict[str, str]]:
    """Extract and pin the exact torch branch body from the upstream cache AST."""
    root = history._regular_path(upstream_root, "pinned upstream source root", directory=True)
    source_path = root / "hymm/models/dit/nomas_cpt.py"
    try:
        if source_path.is_symlink() or not source_path.is_file():
            raise AttentionProbeError("pinned upstream cache source must be a regular file")
        source_text = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AttentionProbeError("cannot read pinned upstream cache source") from error
    source_file_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_file_sha != UPSTREAM_SOURCE_FILE_SHA256:
        raise AttentionProbeError("pinned upstream cache source file digest changed")
    try:
        tree = ast.parse(source_text, filename=str(source_path))
    except SyntaxError as error:
        raise AttentionProbeError("pinned upstream cache source is not valid Python") from error
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "causal_attention_with_cache_mp"
    ]
    if len(functions) != 1:
        raise AttentionProbeError("pinned source must define one cache-attention function")
    function = functions[0]
    function_source = ast.get_source_segment(source_text, function)
    if (not function_source
            or hashlib.sha256(function_source.encode("utf-8")).hexdigest()
            != CACHE_FUNCTION_SOURCE_SHA256):
        raise AttentionProbeError("pinned cache function AST source digest changed")
    branches = [node for node in ast.walk(function) if _source_branch_match(node)]
    if len(branches) != 1:
        raise AttentionProbeError("pinned cache function must contain one torch attention branch")
    branch = branches[0]
    module = ast.fix_missing_locations(ast.Module(body=branch.body, type_ignores=[]))
    branch_digest = hashlib.sha256(
        ast.dump(module, include_attributes=False).encode("utf-8")
    ).hexdigest()
    if branch_digest != TORCH_BRANCH_BODY_AST_SHA256:
        raise AttentionProbeError("pinned torch attention branch AST changed")
    identity = {
        "source_file_sha256": source_file_sha,
        "cache_function_source_sha256": CACHE_FUNCTION_SOURCE_SHA256,
        "torch_branch_body_ast_sha256": branch_digest,
    }
    return module, identity


def _validate_attention_receipt(receipt: dict[str, Any], *, source: bool) -> None:
    """Validate the closed source or native receipt contract without file IO."""
    if type(receipt) is not dict:
        raise AttentionProbeError("attention receipt must be a JSON object")
    common = {
        "schema", "schema_version", "upstream_revision", "scope",
        "full_upstream_t0_preprocessing_parity", "attention_executed",
        "cache_append_executed", "device", "input_fixtures", "attention_backend",
        "attention", "attention_inputs", "tensor_artifact",
    }
    source_only = {
        "upstream_source_file_sha256", "cache_function_source_sha256",
        "torch_branch_body_ast_sha256", "qkv_fixture_provenance",
        "cpu_fallback_disabled",
    }
    native_only = {
        "native_revision", "source_attention_receipt_sha256",
        "named_buffer_accounting", "encode_allocations", "encode_readbacks",
        "preterminal_waits", "terminal_waits",
    }
    expected_keys = common | (source_only if source else native_only)
    if set(receipt) != expected_keys:
        raise AttentionProbeError("attention receipt fields differ from the pinned schema")
    schema = SOURCE_SCHEMA if source else NATIVE_SCHEMA
    scope = SOURCE_SCOPE if source else NATIVE_SCOPE
    if (receipt.get("schema") != schema
            or type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("upstream_revision") != UPSTREAM_REVISION
            or receipt.get("scope") != scope
            or receipt.get("full_upstream_t0_preprocessing_parity") is not False
            or receipt.get("attention_executed") is not True
            or receipt.get("cache_append_executed") is not False
            or receipt.get("device") != "mps"):
        raise AttentionProbeError("attention receipt overstates or misidentifies its scope")
    if source:
        if (receipt.get("upstream_source_file_sha256") != UPSTREAM_SOURCE_FILE_SHA256
                or receipt.get("cache_function_source_sha256") != CACHE_FUNCTION_SOURCE_SHA256
                or receipt.get("torch_branch_body_ast_sha256") != TORCH_BRANCH_BODY_AST_SHA256
                or receipt.get("qkv_fixture_provenance") != TRUSTED_FIXTURE_PROVENANCE
                or receipt.get("cpu_fallback_disabled") is not True
                or receipt.get("attention_backend")
                != "torch.nn.functional.scaled_dot_product_attention"):
            raise AttentionProbeError("source receipt does not pin the actual upstream torch branch")
    else:
        if (type(receipt.get("native_revision")) is not str
                or _GIT_SHA.fullmatch(receipt["native_revision"]) is None
                or not _is_sha256(receipt.get("source_attention_receipt_sha256"))
                or receipt.get("attention_backend") != "native_mps"):
            raise AttentionProbeError("native receipt has an invalid revision, source binding, or attention backend")
        accounting_keys = {
            "input_buffers", "block_weight_residents", "block_scratch", "mps_weight_stage",
            "compact_attention_buffers", "qk_rms_square_and_mean_buffers",
            "reference_bf16_buffers", "reference_history_buffers",
            "reference_attention_buffers", "total_named_bytes",
        }
        accounting = receipt.get("named_buffer_accounting")
        if (type(accounting) is not dict or set(accounting) != accounting_keys
                or any(type(accounting.get(key)) is not int or accounting[key] < 0
                       for key in accounting_keys)
                or accounting["reference_bf16_buffers"] <= 0
                or accounting["reference_history_buffers"] != ELEMENTS * 12
                or accounting["reference_attention_buffers"] != ELEMENTS * 4
                or accounting["total_named_bytes"] != sum(
                    accounting[key] for key in accounting_keys - {"total_named_bytes"}
                )):
            raise AttentionProbeError("native named-buffer accounting is malformed or inconsistent")
        counters = (
            "encode_allocations", "encode_readbacks", "preterminal_waits", "terminal_waits",
        )
        if (any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in counters)
                or receipt["encode_allocations"] != 0
                or receipt["encode_readbacks"] != 0
                or receipt["preterminal_waits"] != 0
                or receipt["terminal_waits"] != 1):
            raise AttentionProbeError("native execution counters violate the terminal-drain contract")

    fixture_keys = {
        "source_input_receipt_sha256", "source_input_artifact_sha256",
        "source_history_receipt_sha256", "source_history_artifact_sha256",
    }
    fixtures = receipt.get("input_fixtures")
    if (type(fixtures) is not dict or set(fixtures) != fixture_keys
            or any(not _is_sha256(fixtures.get(key)) for key in fixture_keys)):
        raise AttentionProbeError("attention receipt input fixture identities are malformed")

    attention = receipt.get("attention")
    attention_keys = {
        "query_tokens", "kv_tokens", "heads", "head_dim", "input_dtype",
        "output_dtype", "layout", "attention_mask", "dropout_p", "is_causal",
    }
    expected_attention = {
        "query_tokens": TOKENS,
        "kv_tokens": TOKENS,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "input_dtype": "BF16",
        "output_dtype": "BF16" if source else None,
        "layout": LAYOUT,
        "attention_mask": "none",
        "dropout_p": 0.0,
        "is_causal": False,
    }
    if type(attention) is not dict or set(attention) != attention_keys:
        raise AttentionProbeError("attention receipt execution contract is malformed")
    if not source and attention.get("output_dtype") not in ("BF16", "F32"):
        raise AttentionProbeError("native output dtype must be BF16 or exact BF16-expanded F32")
    if not source:
        expected_attention["output_dtype"] = attention.get("output_dtype")
    if not _matches_exact_json(attention, expected_attention):
        raise AttentionProbeError("attention receipt has the wrong shape, dtype, or SDPA semantics")

    input_records = receipt.get("attention_inputs")
    if type(input_records) is not list or len(input_records) != len(INPUT_TENSOR_NAMES):
        raise AttentionProbeError("attention receipt Q/K/V inventory is malformed")
    for name, record in zip(INPUT_TENSOR_NAMES, input_records):
        expected_record = {
            "name": name,
            "dtype": "BF16",
            "shape": list(OUTPUT_SHAPE),
            "nbytes": ELEMENTS * 2,
        }
        if (type(record) is not dict
                or set(record) != set(expected_record) | {"sha256"}
                or any(not _matches_exact_json(record.get(key), value)
                       for key, value in expected_record.items())
                or type(record.get("nbytes")) is not int
                or not _is_sha256(record.get("sha256"))):
            raise AttentionProbeError(f"attention receipt has an invalid Q/K/V record: {name}")

    artifact = receipt.get("tensor_artifact")
    if (type(artifact) is not dict or set(artifact) != {"basename", "sha256", "bytes", "tensors"}
            or type(artifact.get("basename")) is not str
            or not artifact["basename"] or Path(artifact["basename"]).name != artifact["basename"]
            or artifact["basename"] in (".", "..")
            or "/" in artifact["basename"] or "\\" in artifact["basename"]
            or not _is_sha256(artifact.get("sha256"))
            or type(artifact.get("bytes")) is not int or artifact["bytes"] <= 0
            or type(artifact.get("tensors")) is not list
            or len(artifact["tensors"]) != len(TENSOR_NAMES)):
        raise AttentionProbeError("attention tensor artifact receipt identity is malformed")
    expected_dtypes = ["BF16", "BF16", "BF16", "BF16" if source else attention["output_dtype"]]
    total_bytes = 0
    for name, dtype, record, input_record in zip(
            TENSOR_NAMES, expected_dtypes, artifact["tensors"],
            (*input_records, None)):
        expected_record = {
            "name": name,
            "dtype": dtype,
            "shape": list(OUTPUT_SHAPE),
            "nbytes": ELEMENTS * (2 if dtype == "BF16" else 4),
        }
        if (type(record) is not dict
                or set(record) != set(expected_record) | {"sha256"}
                or any(not _matches_exact_json(record.get(key), value)
                       for key, value in expected_record.items())
                or type(record.get("nbytes")) is not int
                or not _is_sha256(record.get("sha256"))):
            raise AttentionProbeError(f"attention receipt has an invalid tensor record: {name}")
        if input_record is not None and record["sha256"] != input_record["sha256"]:
            raise AttentionProbeError(f"attention Q/K/V digest differs across receipt inventories: {name}")
        total_bytes += record["nbytes"]
    if (artifact["bytes"] <= total_bytes
            or artifact["bytes"] > total_bytes + block0_compare._HEADER_LIMIT + 8):
        raise AttentionProbeError("attention artifact file size is outside its tensor/header extent")


def _validate_attention_artifact(
    receipt: dict[str, Any], artifact_path: str | Path, *, source: bool,
) -> block0_compare.TensorArtifact:
    _validate_attention_receipt(receipt, source=source)
    try:
        artifact = block0_compare.TensorArtifact(
            artifact_path,
            expected_sha256=receipt["tensor_artifact"]["sha256"],
            expected_names=TENSOR_NAMES,
        )
        bound = block0_compare._receipt_artifact_path(
            receipt["tensor_artifact"], artifact.path, "basename"
        )
    except block0_compare.CompareError as error:
        raise AttentionProbeError(f"attention tensor artifact is invalid: {error}") from error
    if bound != artifact.path:
        raise AttentionProbeError("attention receipt names a different tensor artifact")
    if (receipt["tensor_artifact"]["bytes"] != artifact.path.stat().st_size
            or len(receipt["tensor_artifact"]["tensors"]) != len(TENSOR_NAMES)):
        raise AttentionProbeError("attention artifact size or inventory differs from its receipt")
    expected_metadata = _artifact_metadata(receipt, source=source)
    if artifact.metadata != expected_metadata:
        raise AttentionProbeError("attention safetensors metadata differs from its receipt")
    for name, record in zip(TENSOR_NAMES, receipt["tensor_artifact"]["tensors"]):
        tensor = artifact.tensors[name]
        if (tensor["dtype"] != record["dtype"]
                or tensor["shape"] != OUTPUT_SHAPE
                or tensor["nbytes"] != record["nbytes"]
                or artifact.tensor_sha256(name) != record["sha256"]):
            raise AttentionProbeError(f"attention tensor bytes differ from the receipt: {name}")
    _validate_output_representation(artifact, source=source)
    artifact.check_identity()
    return artifact


def _artifact_metadata(receipt: dict[str, Any], *, source: bool) -> dict[str, str]:
    metadata = {
        "format": SOURCE_SCHEMA if source else NATIVE_SCHEMA,
        "upstream_revision": UPSTREAM_REVISION,
        **receipt["input_fixtures"],
    }
    if source:
        metadata["torch_branch_body_ast_sha256"] = TORCH_BRANCH_BODY_AST_SHA256
    return metadata


def _validate_output_representation(
    artifact: block0_compare.TensorArtifact, *, source: bool,
) -> None:
    record = artifact.tensors["attention.output_bf16"]
    if source:
        if record["dtype"] != "BF16":
            raise AttentionProbeError("pinned upstream MPS SDPA control must produce BF16 output")
        return
    if record["dtype"] == "BF16":
        return
    if record["dtype"] != "F32":
        raise AttentionProbeError("native attention output must be BF16 or F32")
    try:
        import numpy as np
    except ImportError as error:
        raise AttentionProbeError("NumPy is required to validate F32 BF16 expansion") from error
    values = artifact.read_values("attention.output_bf16", 0, record["count"])
    words = np.asarray(values, dtype="<f4").view("<u4")
    if np.any((words & np.uint32(0xFFFF)) != 0):
        raise AttentionProbeError("native F32 output is not an exact BF16 expansion")


def _compare_output_tensor(
    source: block0_compare.TensorArtifact,
    native: block0_compare.TensorArtifact,
    *,
    chunk_elements: int = 262_144,
) -> dict[str, Any]:
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise AttentionProbeError("chunk_elements must be a positive integer")
    if (source.tensors.get("attention.output_bf16", {}).get("dtype") != "BF16"
            or native.tensors.get("attention.output_bf16", {}).get("dtype") not in ("BF16", "F32")
            or source.tensors["attention.output_bf16"]["shape"]
            != native.tensors["attention.output_bf16"]["shape"]):
        raise AttentionProbeError("source/native attention output layouts differ")
    try:
        import numpy as np

        source_record = source.tensors["attention.output_bf16"]
        native_record = native.tensors["attention.output_bf16"]
        count = source_record["count"]
        compared = 0
        max_abs = 0.0
        squared_error_sum = 0.0
        mismatch_count = 0
        while compared < count:
            amount = min(chunk_elements, count - compared)
            expected = source.read_values("attention.output_bf16", compared, amount)
            actual = native.read_values("attention.output_bf16", compared, amount)
            if not bool(np.isfinite(actual).all()) or not bool(np.isfinite(expected).all()):
                raise AttentionProbeError("non-finite value in source/native attention output")
            difference = np.abs(actual - expected)
            if difference.size:
                max_abs = max(max_abs, float(difference.max()))
            squared_error_sum += float(
                np.square(difference, dtype=np.float64).sum(dtype=np.float64)
            )
            mismatch_count += int(np.count_nonzero(difference > ATOL + RTOL * np.abs(expected)))
            compared += amount
        return {
            "tensor": "attention.output_bf16",
            "elements": count,
            "max_abs_error": max_abs,
            "rms_error": math.sqrt(squared_error_sum / count) if count else 0.0,
            "mismatch_count": mismatch_count,
            "atol": ATOL,
            "rtol": RTOL,
            "passed": mismatch_count == 0,
        }
    except ImportError as error:
        raise AttentionProbeError("NumPy is required for attention output comparison") from error
    except block0_compare.CompareError as error:
        raise AttentionProbeError(f"attention numeric comparison failed: {error}") from error


def _tensor_concat_sha256(artifact: block0_compare.TensorArtifact, names: Iterable[str]) -> str:
    """Hash exact tensor bytes in logical concatenation order without materializing them."""
    digest = hashlib.sha256()
    for name in names:
        record = artifact.tensors[name]
        start = artifact.data_offset + record["offset"]
        remaining = record["nbytes"]
        with artifact.path.open("rb") as stream:
            stream.seek(start)
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise AttentionProbeError(f"truncated attention input fixture tensor: {name}")
                digest.update(chunk)
                remaining -= len(chunk)
        artifact.check_identity()
    return digest.hexdigest()


def _validate_fixtures(
    source_input_receipt_path: str | Path,
    source_input_artifact_path: str | Path,
    source_history_receipt_path: str | Path,
    source_history_artifact_path: str | Path,
) -> tuple[dict[str, Any], block0_compare.TensorArtifact,
           dict[str, Any], block0_compare.TensorArtifact, dict[str, str]]:
    try:
        input_receipt, input_artifact, _bundle, input_receipt_sha, input_artifact_sha = (
            history._validate_source_input(
                source_input_receipt_path,
                source_input_artifact_path,
                support=block0_compare.source_reference,
            )
        )
        history_receipt, history_artifact = history._validate_source_history(
            source_history_receipt_path, source_history_artifact_path
        )
    except history.HistoryProbeError as error:
        raise AttentionProbeError(f"source Q/K/V fixture validation failed: {error}") from error
    fixture = history_receipt["source_input"]
    if (fixture["source_receipt_sha256"] != input_receipt_sha
            or fixture["source_artifact_sha256"] != input_artifact_sha):
        raise AttentionProbeError("source history was captured from a different preattention fixture")
    if fixture["valid_text_tokens"] != VALID_TEXT_TOKENS:
        raise AttentionProbeError("source attention fixture must have exactly 529 valid text tokens")
    for fixture_key, input_key in (
        ("input_bundle_manifest_sha256", "manifest_sha256"),
        ("input_bundle_payload_sha256", "payload_sha256"),
        ("reference_latent_artifact_sha256", "reference_latent_artifact_sha256"),
        ("timeline_manifest_sha256", "timeline_manifest_sha256"),
        ("weights_sha256", None),
    ):
        expected = input_receipt["weight_source"]["sha256"] if input_key is None else input_receipt["input_bundle"][input_key]
        if fixture[fixture_key] != expected:
            raise AttentionProbeError(f"source history fixture identity differs: {fixture_key}")
    for input_name, history_name in (
        ("cache.k_bf16", "history.k_bf16"),
        ("cache.v_bf16", "history.v_bf16"),
    ):
        if input_artifact.tensor_sha256(input_name) != history_artifact.tensor_sha256(history_name):
            raise AttentionProbeError(f"physical first-T0 source cache bytes differ: {input_name}")
    identities = {
        "source_input_receipt_sha256": input_receipt_sha,
        "source_input_artifact_sha256": input_artifact_sha,
        "source_history_receipt_sha256": history._hash_file(Path(source_history_receipt_path)),
        "source_history_artifact_sha256": history_artifact.sha256,
    }
    return input_receipt, input_artifact, history_receipt, history_artifact, identities


def _validate_source_attention_inputs(
    source_attention: block0_compare.TensorArtifact,
    input_artifact: block0_compare.TensorArtifact,
    history_artifact: block0_compare.TensorArtifact,
) -> None:
    source_q_sha = _tensor_concat_sha256(input_artifact, ("img.q_rope", "txt.q_qknorm"))
    if source_attention.tensor_sha256("attention.q_bf16") != source_q_sha:
        raise AttentionProbeError("source attention Q differs from the pinned BF16 Q fixture")
    for name, input_name, history_name in (
        ("attention.k_stored_bf16", "cache.k_bf16", "history.k_bf16"),
        ("attention.v_stored_bf16", "cache.v_bf16", "history.v_bf16"),
    ):
        source_hash = source_attention.tensor_sha256(name)
        if (source_hash != input_artifact.tensor_sha256(input_name)
                or source_hash != history_artifact.tensor_sha256(history_name)):
            raise AttentionProbeError(f"source attention did not consume the physical BF16 {name}")


def _read_bf16_to_device(
    artifact: block0_compare.TensorArtifact,
    name: str,
    *,
    torch: Any,
    device: Any,
) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise AttentionProbeError("NumPy is required to materialize pinned BF16 inputs") from error
    record = artifact.tensors[name]
    if record["dtype"] != "BF16":
        raise AttentionProbeError(f"attention source tensor must be stored BF16: {name}")
    values = artifact.read_values(name, 0, record["count"])
    if not bool(np.isfinite(values).all()):
        raise AttentionProbeError(f"attention source tensor contains non-finite values: {name}")
    host = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).reshape(record["shape"])
    return host.to(device=device, dtype=torch.bfloat16)


def _validate_output_paths(tensor_path: str | Path, receipt_path: str | Path) -> tuple[Path, Path]:
    try:
        return history._validate_new_outputs(tensor_path, receipt_path)
    except history.HistoryProbeError as error:
        raise AttentionProbeError(str(error)) from error


def _save_attention_artifact(
    tensors: dict[str, Any],
    *,
    tensor_path: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise AttentionProbeError("safetensors is required to write attention output") from error
    stage = Path(tempfile.mkdtemp(prefix=f".{tensor_path.stem}.stage-", dir=tensor_path.parent))
    staged_tensor = stage / tensor_path.name
    staged_receipt = stage / receipt_path.name
    try:
        # Stack the four MPS tensors and transfer once, after the SDPA result is complete.
        device_stack = torch.stack(tuple(tensors[name] for name in TENSOR_NAMES), dim=0)
        host_stack = device_stack.detach().contiguous().to(device="cpu").contiguous()
        cpu_tensors = {
            name: value.contiguous()
            for name, value in zip(TENSOR_NAMES, host_stack.unbind(0))
        }
        if any(value.dtype != torch.bfloat16 for value in cpu_tensors.values()):
            raise AttentionProbeError("source attention oracle requires BF16 Q/K/V and output")
        save_file(cpu_tensors, str(staged_tensor), metadata=_artifact_metadata(receipt, source=True))
        tensor_records = []
        for name in TENSOR_NAMES:
            value = cpu_tensors[name]
            raw = value.view(torch.uint8).numpy().tobytes()
            tensor_records.append({
                "name": name,
                "dtype": "BF16",
                "shape": list(value.shape),
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        receipt["attention_inputs"] = tensor_records[:len(INPUT_TENSOR_NAMES)]
        receipt["tensor_artifact"] = {
            "basename": tensor_path.name,
            "sha256": history._hash_file(staged_tensor),
            "bytes": staged_tensor.stat().st_size,
            "tensors": tensor_records,
        }
        _validate_attention_receipt(receipt, source=True)
        with staged_receipt.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
        if tensor_path.exists() or receipt_path.exists():
            raise AttentionProbeError("attention output appeared during artifact creation")
        os.link(staged_tensor, tensor_path)
        published = tensor_path.stat(follow_symlinks=False)
        try:
            os.link(staged_receipt, receipt_path)
        except OSError:
            try:
                current = tensor_path.stat(follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (published.st_dev, published.st_ino):
                tensor_path.unlink()
            raise
    except AttentionProbeError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise AttentionProbeError(f"cannot write source attention artifact: {error}") from error
    finally:
        for child in stage.iterdir():
            child.unlink(missing_ok=True)
        stage.rmdir()
    return receipt


def run_source_attention(
    source_input_receipt_path: str | Path,
    source_input_artifact_path: str | Path,
    source_history_receipt_path: str | Path,
    source_history_artifact_path: str | Path,
    *,
    upstream_root: str | Path = history.cache_oracle.UPSTREAM_ROOT_DEFAULT,
    output_safetensors: str | Path,
    output_receipt: str | Path,
) -> dict[str, Any]:
    """Run the pinned Torch branch body on cropped 4849-row MPS BF16 Q/K/V.

    The enclosing source function's padding and unpadding control flow is not
    executed; this is not a claim about padded Torch or Flash3 geometry.
    """
    tensor_path, receipt_path = _validate_output_paths(output_safetensors, output_receipt)
    input_receipt, input_artifact, history_receipt, history_artifact, identities = _validate_fixtures(
        source_input_receipt_path,
        source_input_artifact_path,
        source_history_receipt_path,
        source_history_artifact_path,
    )
    branch, branch_identity = _load_torch_attention_branch(upstream_root)
    if branch_identity["source_file_sha256"] != UPSTREAM_SOURCE_FILE_SHA256:
        raise AttentionProbeError("upstream source file pin mismatch")

    # PyTorch reads this switch during MPS dispatch; set it before importing torch.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
    except ImportError as error:
        raise AttentionProbeError("PyTorch with MPS support is required for source attention") from error
    if not torch.backends.mps.is_available():
        raise AttentionProbeError("MPS is unavailable; the source oracle has no CPU fallback")
    device = torch.device("mps")
    q_img = _read_bf16_to_device(input_artifact, "img.q_rope", torch=torch, device=device)
    q_txt = _read_bf16_to_device(input_artifact, "txt.q_qknorm", torch=torch, device=device)
    key = _read_bf16_to_device(history_artifact, "history.k_bf16", torch=torch, device=device)
    value = _read_bf16_to_device(history_artifact, "history.v_bf16", torch=torch, device=device)
    # These fixtures are already cropped to image tokens plus valid text rows;
    # execute the branch body without the enclosing function's pad/unpad path.
    q = torch.cat((q_img, q_txt), dim=1).contiguous()
    if any(tuple(tensor.shape) != OUTPUT_SHAPE for tensor in (q, key, value)):
        raise AttentionProbeError("source Q/K/V do not have the pinned [1,4849,16,128] shape")
    if any(tensor.dtype != torch.bfloat16 or tensor.device.type != "mps"
           for tensor in (q, key, value)):
        raise AttentionProbeError("source attention Q/K/V must remain BF16 on MPS")

    namespace = {
        "F": torch.nn.functional,
        "torch": torch,
        "q_full": q,
        "full_k": key,
        "full_v": value,
        "kv_mask": None,
        "compressed_storage": False,
        "kv_cache": {},
    }
    try:
        with torch.no_grad():
            exec(compile(branch, "<pinned-nomas-torch-attention-branch>", "exec"), namespace)
            output = namespace.get("hidden")
    except Exception as error:
        raise AttentionProbeError(f"pinned upstream MPS BF16 SDPA branch failed: {error}") from error
    if (not isinstance(output, torch.Tensor)
            or tuple(output.shape) != OUTPUT_SHAPE
            or output.dtype != torch.bfloat16
            or output.device.type != "mps"):
        raise AttentionProbeError("pinned upstream branch did not return BF16 MPS output [1,4849,16,128]")

    receipt: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_source_file_sha256": branch_identity["source_file_sha256"],
        "cache_function_source_sha256": branch_identity["cache_function_source_sha256"],
        "torch_branch_body_ast_sha256": branch_identity["torch_branch_body_ast_sha256"],
        "scope": SOURCE_SCOPE,
        "full_upstream_t0_preprocessing_parity": False,
        "qkv_fixture_provenance": TRUSTED_FIXTURE_PROVENANCE,
        "attention_executed": True,
        "cache_append_executed": False,
        "device": "mps",
        "cpu_fallback_disabled": True,
        "input_fixtures": identities,
        "attention_backend": "torch.nn.functional.scaled_dot_product_attention",
        "attention": {
            "query_tokens": TOKENS,
            "kv_tokens": TOKENS,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "input_dtype": "BF16",
            "output_dtype": "BF16",
            "layout": LAYOUT,
            "attention_mask": "none",
            "dropout_p": 0.0,
            "is_causal": False,
        },
        "attention_inputs": [],
        "tensor_artifact": {},
    }
    del input_receipt, history_receipt
    return _save_attention_artifact(
        {
            "attention.q_bf16": q,
            "attention.k_stored_bf16": key,
            "attention.v_stored_bf16": value,
            "attention.output_bf16": output,
        },
        tensor_path=tensor_path,
        receipt_path=receipt_path,
        receipt=receipt,
        torch=torch,
    )


def compare_native_attention(
    source_attention_receipt_path: str | Path,
    source_attention_artifact_path: str | Path,
    native_receipt_path: str | Path,
    native_artifact_path: str | Path,
    source_input_receipt_path: str | Path,
    source_input_artifact_path: str | Path,
    source_history_receipt_path: str | Path,
    source_history_artifact_path: str | Path,
    *,
    output_path: str | Path | None = None,
    chunk_elements: int = 262_144,
) -> dict[str, Any]:
    """Validate identical source/native QKV and strictly compare attention output."""
    try:
        source_receipt = block0_compare._read_json(source_attention_receipt_path)
        native_receipt = block0_compare._read_json(native_receipt_path)
    except block0_compare.CompareError as error:
        raise AttentionProbeError(f"attention receipt is invalid: {error}") from error
    source_artifact = _validate_attention_artifact(
        source_receipt, source_attention_artifact_path, source=True
    )
    native_artifact = _validate_attention_artifact(
        native_receipt, native_artifact_path, source=False
    )
    _input_receipt, input_artifact, _history_receipt, history_artifact, identities = _validate_fixtures(
        source_input_receipt_path,
        source_input_artifact_path,
        source_history_receipt_path,
        source_history_artifact_path,
    )
    if source_receipt["input_fixtures"] != identities:
        raise AttentionProbeError("source attention receipt input fixture identities changed")
    if native_receipt["input_fixtures"] != identities:
        raise AttentionProbeError("native attention did not bind the same source fixtures")
    if native_receipt["source_attention_receipt_sha256"] != history._hash_file(
            Path(source_attention_receipt_path)):
        raise AttentionProbeError("native attention receipt is bound to a different source attention receipt")
    _validate_source_attention_inputs(source_artifact, input_artifact, history_artifact)

    qkv_results = []
    for name in INPUT_TENSOR_NAMES:
        source_hash = source_artifact.tensor_sha256(name)
        native_hash = native_artifact.tensor_sha256(name)
        if source_hash != native_hash:
            raise AttentionProbeError(f"source/native attention input bytes differ: {name}")
        qkv_results.append({"tensor": name, "sha256": source_hash, "byte_exact": True})
    output_result = _compare_output_tensor(
        source_artifact, native_artifact, chunk_elements=chunk_elements
    )
    result = {
        "schema": COMPARISON_SCHEMA,
        "passed": output_result["passed"],
        "scope": NATIVE_SCOPE,
        "input_fixture_identities": identities,
        "source_attention_receipt_sha256": history._hash_file(Path(source_attention_receipt_path)),
        "source_attention_artifact_sha256": source_artifact.sha256,
        "native_attention_receipt_sha256": history._hash_file(Path(native_receipt_path)),
        "native_attention_artifact_sha256": native_artifact.sha256,
        "upstream_source_file_sha256": UPSTREAM_SOURCE_FILE_SHA256,
        "cache_function_source_sha256": CACHE_FUNCTION_SOURCE_SHA256,
        "torch_branch_body_ast_sha256": TORCH_BRANCH_BODY_AST_SHA256,
        "shape": list(OUTPUT_SHAPE),
        "atol": ATOL,
        "rtol": RTOL,
        "qkv_byte_exact": qkv_results,
        "output_comparison": output_result,
    }
    source_artifact.check_identity()
    native_artifact.check_identity()
    if output_path is not None:
        _write_fresh_json(output_path, result)
    return result


def _write_fresh_json(path: str | Path, document: dict[str, Any]) -> Path:
    output = Path(path).expanduser()
    if (not output.is_absolute() or output.suffix.lower() != ".json"
            or output.is_symlink() or output.exists() or not output.parent.is_dir()):
        raise AttentionProbeError("comparison output must be a new absolute .json file")
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{output.name}.stage-", dir=output.parent
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.link(staged, output)
    except OSError as error:
        raise AttentionProbeError(f"cannot publish comparison output: {error}") from error
    finally:
        staged.unlink(missing_ok=True)
    return output


def _add_fixture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-input-receipt", type=Path, required=True)
    parser.add_argument("--source-input-artifact", type=Path, required=True)
    parser.add_argument("--source-history-receipt", type=Path, required=True)
    parser.add_argument("--source-history-artifact", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=history.cache_oracle.UPSTREAM_ROOT_DEFAULT)
    parser.add_argument("--source-attention-receipt", type=Path)
    parser.add_argument("--source-attention-artifact", type=Path)
    parser.add_argument("--native-attention-receipt", type=Path)
    parser.add_argument("--native-attention-artifact", type=Path)
    parser.add_argument("--output-safetensors", type=Path)
    parser.add_argument("--output-receipt", type=Path)
    parser.add_argument("--output-comparison", type=Path)
    parser.add_argument("--chunk-elements", type=int, default=262_144)
    _add_fixture_arguments(parser)
    args = parser.parse_args(argv)
    producing = args.output_safetensors is not None or args.output_receipt is not None
    comparing = any((args.source_attention_receipt, args.source_attention_artifact,
                     args.native_attention_receipt, args.native_attention_artifact,
                     args.output_comparison))
    if producing == comparing:
        parser.error("select exactly one mode: paired --output-* or source/native attention receipts")
    try:
        if producing:
            if (args.output_safetensors is None or args.output_receipt is None
                    or any((args.source_attention_receipt, args.source_attention_artifact,
                            args.native_attention_receipt, args.native_attention_artifact,
                            args.output_comparison))):
                parser.error("source mode requires both output paths and no comparison paths")
            run_source_attention(
                args.source_input_receipt,
                args.source_input_artifact,
                args.source_history_receipt,
                args.source_history_artifact,
                upstream_root=args.upstream,
                output_safetensors=args.output_safetensors,
                output_receipt=args.output_receipt,
            )
            print(json.dumps({"passed": True, "schema": SOURCE_SCHEMA}, sort_keys=True))
            return 0
        if any(value is None for value in (
                args.source_attention_receipt, args.source_attention_artifact,
                args.native_attention_receipt, args.native_attention_artifact)):
            parser.error("compare mode requires source/native attention receipt and artifact pairs")
        result = compare_native_attention(
            args.source_attention_receipt,
            args.source_attention_artifact,
            args.native_attention_receipt,
            args.native_attention_artifact,
            args.source_input_receipt,
            args.source_input_artifact,
            args.source_history_receipt,
            args.source_history_artifact,
            output_path=args.output_comparison,
            chunk_elements=args.chunk_elements,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 1
    except AttentionProbeError as error:
        print(f"attention-oracle error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
