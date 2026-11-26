#!/usr/bin/env python3
"""Capture the pinned block-0 BF16 cache writer and reader, then compare native history."""
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
import types
from typing import Any

try:
    from . import hy_nomas_reference_block0_compare as block0_compare
    from . import hy_nomas_reference_cache as cache_oracle
except ImportError:  # direct import by repository test modules
    import hy_nomas_reference_block0_compare as block0_compare
    import hy_nomas_reference_cache as cache_oracle


SOURCE_SCHEMA = "cogni_ml.hy_nomas_reference_history_source/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_reference_history_probe/v1"
SCHEMA_VERSION = 1
UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
SOURCE_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_SOURCE_FILE_SHA256 = "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee"
CACHE_FUNCTION_SOURCE_SHA256 = "ccf791d9eacb943d5e655f108aba61d27928cdf5d018ecb7dceb85d4d68adbef"
BLOCK_WEIGHTS_SHA256 = "01be7388fa4267f1f566c3ced405418f94d0ed2b4ee969cc880fb2518ab7c87b"
IMAGE_TOKENS = 4320
TEXT_STORAGE_TOKENS = 1280
HEADS = 16
HEAD_DIM = 128
HIDDEN = HEADS * HEAD_DIM
LAYOUT = "[1,tokens,heads,head_dim]; image then valid text"
SOURCE_SCOPE = (
    "actual pinned causal_attention_with_cache_mp T0 store/read over a prior BF16 source block-0 "
    "Q/K/V capture supplied as a trusted fixture; this oracle binds its receipt/artifact hashes "
    "but does not independently attest how that capture was produced; torch attention consumer "
    "intercepted before attention math; not a complete upstream T0 handoff"
)
NATIVE_SCOPE = (
    "one standalone native block-0 BF16 pre-attention forward with first T0 history store/read; "
    "not a complete upstream T0 handoff"
)
REFERENCE_BF16_PRECISION = "reference-bf16-history"
BF16_LINEAR_VARIANT = "bf16-graph"
SOURCE_INPUT_SCOPE = (
    "Pinned upstream MMDoubleStreamBlock.forward_with_cache on the exact native preblock input "
    "tensors; captures block-0 K/V at first-chunk cache append. Does not claim parity for "
    "upstream timeline, noise, patch, or text-refiner preprocessing."
)
SOURCE_TENSOR_NAMES = (
    "history.k_bf16", "history.v_bf16", "history.k_read_f32", "history.v_read_f32",
)
NATIVE_TENSOR_NAMES = SOURCE_TENSOR_NAMES
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HEADER_LIMIT = 16 * 1024 * 1024
ATOL = 2e-5
RTOL = 2e-5
ACCOUNTING_KEYS = {
    "input_buffers", "block_weight_residents", "block_scratch", "mps_weight_stage",
    "compact_attention_buffers", "qk_rms_square_and_mean_buffers", "total_named_bytes",
    "reference_bf16_buffers", "reference_history_buffers", "opaque_mpsgraph_private_bytes",
}
EXECUTION_COUNTERS = {
    "encode_allocations", "encode_readbacks", "preterminal_waits", "terminal_waits",
}


class HistoryProbeError(ValueError):
    """A history receipt, source fixture, or tensor artifact failed admission."""


class _AttentionConsumerIntercepted(Exception):
    """Control flow after the pinned cache function reaches its attention consumer."""

    def __init__(self, phase: str):
        super().__init__(phase)
        self.phase = phase


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _matches_exact_json(value: object, expected: object) -> bool:
    """Compare receipt values without JSON's bool/int equality loophole."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _matches_exact_json(value[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _matches_exact_json(actual, wanted) for actual, wanted in zip(value, expected)
        )
    return value == expected


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_path(value: str | Path, what: str, *, directory: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise HistoryProbeError(f"{what} must be absolute and must not be a symlink")
    if directory:
        if not path.is_dir():
            raise HistoryProbeError(f"{what} must be a directory")
    elif not path.is_file():
        raise HistoryProbeError(f"{what} must be a regular file")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise HistoryProbeError(f"cannot resolve {what}") from error


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        return block0_compare._read_json(path)
    except block0_compare.CompareError as error:
        raise HistoryProbeError(str(error)) from error


def _expected_shape(valid_text_tokens: int) -> tuple[int, int, int, int]:
    if type(valid_text_tokens) is not int or not 1 <= valid_text_tokens <= TEXT_STORAGE_TOKENS:
        raise HistoryProbeError("valid_text_tokens must be an integer in [1, 1280]")
    return (1, IMAGE_TOKENS + valid_text_tokens, HEADS, HEAD_DIM)


def _history_accounting(rows: int) -> dict[str, int]:
    return {
        "persisted_bytes": rows * HIDDEN * 4,
        "read_stage_bytes": rows * HIDDEN * 8,
    }


def _validate_source_input(
    receipt_path: str | Path,
    artifact_path: str | Path,
    *,
    support: Any,
) -> tuple[dict[str, Any], block0_compare.TensorArtifact, dict[str, Any], str, str]:
    receipt_file = _regular_path(receipt_path, "source input receipt")
    artifact_file = _regular_path(artifact_path, "source input tensor artifact")
    source = _read_json(receipt_file)
    required = {
        "schema", "operation", "upstream_revision", "upstream_source_root",
        "upstream_source_file_sha256", "upstream_dependency_sha256",
        "inference_dtype_source_sha256", "weight_source", "parameter_dtype_basis",
        "precision", "scope", "cache_append", "input_bundle", "tensor_artifact",
    }
    if set(source) != required:
        raise HistoryProbeError("old source receipt fields differ from the pinned cache oracle schema")
    if (source.get("schema") != block0_compare.SOURCE_SCHEMA
            or source.get("operation") != "upstream_block0_first_chunk_kv_cache_append"
            or source.get("upstream_revision") != UPSTREAM_REVISION
            or source.get("upstream_source_file_sha256")
            != support.PINNED_SOURCE_SHA256["hymm/models/dit/nomas_cpt.py"]):
        raise HistoryProbeError("old source receipt does not identify the pinned upstream cache capture")
    if source.get("upstream_dependency_sha256") != support.PINNED_SOURCE_SHA256:
        raise HistoryProbeError("old source receipt dependency hashes differ from the pinned source set")
    if source.get("inference_dtype_source_sha256") != cache_oracle.INFERENCE_DTYPE_SOURCE_SHA256:
        raise HistoryProbeError("old source receipt inference dtype hashes differ from the pinned source set")
    if source.get("parameter_dtype_basis") != block0_compare.PARAMETER_DTYPE_BASIS:
        raise HistoryProbeError("old source receipt parameter dtype basis changed")
    if source.get("scope") != SOURCE_INPUT_SCOPE:
        raise HistoryProbeError("old source receipt scope differs from the admitted preblock fixture")
    precision = source.get("precision")
    if (type(precision) is not dict or precision.get("mode") != "bf16"
            or precision.get("block_parameters") != "BF16"
            or precision.get("cache_storage") != "BF16"):
        raise HistoryProbeError("old source artifact must be the authenticated BF16 source capture")
    weight = source.get("weight_source")
    if (type(weight) is not dict or weight.get("sha256") != BLOCK_WEIGHTS_SHA256
            or weight.get("source_dtype") != "BF16" or weight.get("parameter_dtype") != "BF16"
            or weight.get("tensor_prefix") != "double_blocks.0." or weight.get("tensor_count") != 32):
        raise HistoryProbeError("old source receipt block-0 weight identity mismatch")
    input_bundle = source.get("input_bundle")
    if type(input_bundle) is not dict:
        raise HistoryProbeError("old source receipt has no input bundle identity")
    for field in (
        "manifest_sha256", "payload_sha256", "reference_latent_artifact_sha256",
        "timeline_manifest_sha256",
    ):
        if not _is_sha256(input_bundle.get(field)):
            raise HistoryProbeError(f"old source receipt has a malformed {field}")
    valid = input_bundle.get("valid_text_tokens")
    _expected_shape(valid)
    append = source.get("cache_append")
    if (type(append) is not dict or set(append) != {
            "cache_valid_len", "update_cache", "cache_txt", "attention_executed",
            "cache_tensor_order", "cache_tokens"}
            or type(append.get("cache_valid_len")) is not int or append["cache_valid_len"] != 0
            or append.get("update_cache") is not True or append.get("cache_txt") is not True
            or append.get("attention_executed") is not False
            or append.get("cache_tensor_order") != "image tokens followed by valid text prefix"
            or type(append.get("cache_tokens")) is not int
            or append["cache_tokens"] != IMAGE_TOKENS + valid):
        raise HistoryProbeError("old source receipt is not the expected adapter-only T0 input capture")

    source_artifact = source.get("tensor_artifact")
    source_names = (*block0_compare.PRE_CACHE_NAMES, "cache.k_bf16", "cache.v_bf16")
    if (type(source_artifact) is not dict or set(source_artifact) != {
            "path", "sha256", "tensor_count", "tensors"}
            or not _is_sha256(source_artifact.get("sha256"))
            or type(source_artifact.get("tensor_count")) is not int
            or source_artifact["tensor_count"] != len(source_names)
            or type(source_artifact.get("tensors")) is not dict
            or set(source_artifact["tensors"]) != set(source_names)):
        raise HistoryProbeError("old source receipt tensor inventory is invalid")
    try:
        receipt_artifact = block0_compare._receipt_artifact_path(
            source_artifact, artifact_file, "path"
        )
    except block0_compare.CompareError as error:
        raise HistoryProbeError(str(error)) from error
    if receipt_artifact != artifact_file:
        raise HistoryProbeError("old source receipt names a different input artifact")
    try:
        artifact = block0_compare.TensorArtifact(
            artifact_file, expected_sha256=source_artifact["sha256"],
            expected_names=source_names,
        )
        block0_compare.validate_tensor_artifact_records(
            source, artifact, native=False, valid_text_tokens=valid,
            precision="BF16BoundaryControl",
        )
    except block0_compare.CompareError as error:
        raise HistoryProbeError(str(error)) from error
    if artifact.metadata.get("format") != block0_compare.SOURCE_SCHEMA:
        raise HistoryProbeError("old source safetensors metadata has the wrong format")
    if artifact.metadata.get("upstream_revision") != UPSTREAM_REVISION:
        raise HistoryProbeError("old source safetensors metadata has the wrong revision")
    if artifact.metadata.get("weight_sha256") != BLOCK_WEIGHTS_SHA256:
        raise HistoryProbeError("old source safetensors metadata has the wrong weight pin")
    return (
        source, artifact, input_bundle, _hash_file(receipt_file),
        source_artifact["sha256"],
    )


def _load_function_from_pinned_ast(root: Path, support: Any, torch: Any, consumer: Any) -> tuple[Any, dict[str, str]]:
    source_path = root / "hymm/models/dit/nomas_cpt.py"
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "causal_attention_with_cache_mp"
    ]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise HistoryProbeError("pinned source must define exactly one synchronous cache function")
    function_node = functions[0]
    function_source = ast.get_source_segment(source_text, function_node)
    if not function_source:
        raise HistoryProbeError("cannot recover the pinned cache-function source text")
    function_sha = hashlib.sha256(function_source.encode("utf-8")).hexdigest()
    if function_sha != CACHE_FUNCTION_SOURCE_SHA256:
        raise HistoryProbeError("pinned cache function AST source digest changed")

    # Supply only the direct block imports needed by the dense, SP1 torch path.
    for name, directory in {
        "hymm": root / "hymm",
        "hymm.models": root / "hymm/models",
        "hymm.models.dit": root / "hymm/models/dit",
        "hymm.models.dit.modules": root / "hymm/models/dit/modules",
    }.items():
        support._namespace(name, directory)
    cache_module_name = "hymm.models.dit.modules.kv_cache_storage"
    cache_module = sys.modules.get(cache_module_name)
    if cache_module is None or Path(getattr(cache_module, "__file__", "")).resolve() != (
            root / "hymm/models/dit/modules/kv_cache_storage.py").resolve():
        support._load_source_module(
            cache_module_name, root / "hymm/models/dit/modules/kv_cache_storage.py"
        )

    def no_sequence_parallel() -> bool:
        return False

    def reject_collective(*_args: Any, **_kwargs: Any) -> Any:
        raise HistoryProbeError("isolated source cache function unexpectedly requested a collective")

    namespace: dict[str, Any] = {
        "__name__": "hymm.models.dit.nomas_cpt_history_pinned",
        "__package__": "hymm.models.dit",
        "torch": torch,
        "F": consumer,
        "Dict": dict,
        "Optional": __import__("typing").Optional,
        "List": list,
        "get_sequence_parallel_state": no_sequence_parallel,
        "nccl_info": types.SimpleNamespace(sp_group=None, rank_within_spgroup=0),
        "all_to_all_4D": reject_collective,
        "all_gather": reject_collective,
        "ARCHIVED_HISTORY_SAMPLINGS": frozenset(),
        "HISTORY_SAMPLINGS": frozenset(),
        "_HAS_AR_VIS_PROFILER": False,
        "_ar_profile_fa_func": None,
    }
    wrapper = ast.Module(body=[function_node], type_ignores=[])
    exec(compile(wrapper, str(source_path), "exec"), namespace)
    function = namespace.get("causal_attention_with_cache_mp")
    if not callable(function) or function.__code__.co_filename != str(source_path):
        raise HistoryProbeError("compiled cache function is not the pinned source AST")
    function_identity = {
        "name": "causal_attention_with_cache_mp",
        "source_file": "hymm/models/dit/nomas_cpt.py",
        "source_file_sha256": _hash_file(source_path),
        "function_source_sha256": function_sha,
    }
    if function_identity["source_file_sha256"] != UPSTREAM_SOURCE_FILE_SHA256:
        raise HistoryProbeError("pinned cache function source file digest changed")
    return function, function_identity


def _read_source_tensor_to_device(
    artifact: block0_compare.TensorArtifact, name: str, *, torch: Any, device: Any,
) -> Any:
    import numpy as np

    record = artifact.tensors[name]
    values = artifact.read_values(name, 0, record["count"])
    if not bool(np.isfinite(values).all()):
        raise HistoryProbeError(f"old source input tensor contains non-finite values: {name}")
    host = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).reshape(record["shape"])
    if host.dtype != torch.float32 or tuple(host.shape) != record["shape"]:
        raise HistoryProbeError(f"old source input tensor could not be materialized: {name}")
    return host.to(device=device, dtype=torch.bfloat16)


def _tensor_sha256(tensor: Any) -> str:
    cpu = tensor.detach().contiguous().to(device="cpu")
    raw = cpu.view(__import__("torch").uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _validate_new_outputs(tensor_path: str | Path, receipt_path: str | Path) -> tuple[Path, Path]:
    paths = []
    for value, suffix, what in (
        (tensor_path, ".safetensors", "source history tensor artifact"),
        (receipt_path, ".json", "source history receipt"),
    ):
        path = Path(value).expanduser()
        if not path.is_absolute() or path.suffix.lower() != suffix:
            raise HistoryProbeError(f"{what} output must be an absolute {suffix} path")
        if path.is_symlink() or path.exists() or not path.parent.is_dir():
            raise HistoryProbeError(f"refusing to replace or create output at unsafe path: {path}")
        paths.append(path.parent.resolve(strict=True) / path.name)
    if paths[0] == paths[1] or paths[0].parent != paths[1].parent:
        raise HistoryProbeError("source history artifact and receipt must be distinct siblings")
    return paths[0], paths[1]


def _save_history_artifact(
    tensors: dict[str, Any], *, output_path: Path, receipt_path: Path,
    receipt: dict[str, Any], torch: Any, artifact_metadata: dict[str, str],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    stage = Path(tempfile.mkdtemp(prefix=f".{output_path.stem}.stage-", dir=output_path.parent))
    staged_artifact = stage / output_path.name
    staged_receipt = stage / receipt_path.name
    cpu_tensors = {name: value.detach().contiguous().to(device="cpu") for name, value in tensors.items()}
    try:
        save_file(cpu_tensors, str(staged_artifact), metadata=artifact_metadata)
        tensor_records = []
        for name in SOURCE_TENSOR_NAMES:
            tensor = cpu_tensors[name]
            raw = tensor.view(torch.uint8).numpy().tobytes()
            tensor_records.append({
                "name": name,
                "dtype": "BF16" if tensor.dtype == torch.bfloat16 else "F32",
                "shape": list(tensor.shape),
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        receipt["tensor_artifact"] = {
            "basename": output_path.name,
            "sha256": _hash_file(staged_artifact),
            "bytes": staged_artifact.stat().st_size,
            "tensors": tensor_records,
        }
        with staged_receipt.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
        if output_path.exists() or receipt_path.exists():
            raise HistoryProbeError("source history output appeared during artifact creation")
        os.link(staged_artifact, output_path)
        published = output_path.stat(follow_symlinks=False)
        try:
            os.link(staged_receipt, receipt_path)
        except OSError:
            try:
                current = output_path.stat(follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (published.st_dev, published.st_ino):
                output_path.unlink()
            raise
    finally:
        for child in stage.iterdir():
            child.unlink(missing_ok=True)
        stage.rmdir()
    receipt["receipt_path"] = str(receipt_path)
    receipt["tensor_artifact"]["path"] = str(output_path)
    return receipt


def run_upstream_history_probe(
    source_receipt_path: str | Path,
    source_artifact_path: str | Path,
    *,
    upstream_root: str | Path = cache_oracle.UPSTREAM_ROOT_DEFAULT,
    output_safetensors: str | Path,
    receipt_path: str | Path,
    device_name: str = "mps",
) -> dict[str, Any]:
    """Execute the exact pinned cache function on prior BF16 source Q/K/V captures."""
    if device_name != "mps":
        raise HistoryProbeError("the pinned source history oracle requires --device mps")
    output_path, receipt_file = _validate_new_outputs(output_safetensors, receipt_path)
    support = cache_oracle._load_support_oracle()
    source, input_artifact, input_bundle, source_receipt_sha, source_artifact_sha = _validate_source_input(
        source_receipt_path, source_artifact_path, support=support
    )
    source_root, upstream_hashes, upstream_revision = cache_oracle._verify_upstream_root(
        upstream_root, support
    )
    if upstream_revision != UPSTREAM_REVISION:
        raise HistoryProbeError("pinned source revision changed after verification")
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in (None, "0"):
        raise HistoryProbeError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise HistoryProbeError("the pinned source history oracle requires MPS; CPU fallback is forbidden")
    device = torch.device("mps")
    tokens = IMAGE_TOKENS + input_bundle["valid_text_tokens"]
    shape = (1, tokens, HEADS, HEAD_DIM)
    source_map = {
        "img.q_rope": "img.q_rope", "txt.q_qknorm": "txt.q_qknorm",
        "img.k_rope": "img.k_rope", "txt.k_qknorm": "txt.k_qknorm",
        "img.v_projection": "img.v_projection", "txt.v_projection": "txt.v_projection",
    }
    qkv = {
        name: _read_source_tensor_to_device(input_artifact, source_name, torch=torch, device=device)
        for name, source_name in source_map.items()
    }
    for name, tensor in qkv.items():
        if tensor.dtype != torch.bfloat16 or tensor.device.type != "mps":
            raise HistoryProbeError(f"source Q/K/V input is not MPS BF16: {name}")
    old_k = _read_source_tensor_to_device(input_artifact, "cache.k_bf16", torch=torch, device=device)
    old_v = _read_source_tensor_to_device(input_artifact, "cache.v_bf16", torch=torch, device=device)
    cache_buffers = {
        "k": torch.empty(shape, dtype=torch.bfloat16, device=device),
        "v": torch.empty(shape, dtype=torch.bfloat16, device=device),
    }

    active_phase = {"name": ""}
    consumer_captures: dict[str, tuple[Any, Any]] = {}

    def intercept_sdpa(q: Any, k: Any, v: Any, **_kwargs: Any) -> Any:
        phase = active_phase["name"]
        if phase not in ("t0_store_read", "poisoned_read_only") or phase in consumer_captures:
            raise HistoryProbeError("pinned source attention consumer was invoked an unexpected number of times")
        if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4
                or k.shape != v.shape or k.shape[0] != 1
                or k.shape[1] != HEADS or k.shape[3] != HEAD_DIM):
            raise HistoryProbeError("pinned source attention consumer received an unexpected K/V layout")
        canonical_k = k.transpose(1, 2).contiguous().detach().clone()
        canonical_v = v.transpose(1, 2).contiguous().detach().clone()
        consumer_captures[phase] = (canonical_k.to(torch.float32), canonical_v.to(torch.float32))
        raise _AttentionConsumerIntercepted(phase)

    consumer = types.SimpleNamespace(scaled_dot_product_attention=intercept_sdpa)
    cache_function, cache_function_identity = _load_function_from_pinned_ast(
        source_root, support, torch, consumer
    )

    def invoke(phase: str, *, q: Any, k: Any, v: Any, valid_len: int, update_cache: bool) -> None:
        active_phase["name"] = phase
        try:
            with torch.inference_mode():
                cache_function(
                    q=q, k=k, v=v, kv_cache=cache_buffers,
                    cache_valid_len=valid_len, attn_mode="torch", text_mask=None,
                    update_cache=update_cache, cache_txt=True, cache_mask=None,
                    seg_ar_attention_backend="magi",
                )
        except _AttentionConsumerIntercepted as caught:
            if caught.phase != phase:
                raise HistoryProbeError("cache function reached an unexpected consumer phase") from caught
        else:
            raise HistoryProbeError("pinned cache function did not reach the intercepted attention consumer")
        finally:
            active_phase["name"] = ""

    source_q = (qkv["img.q_rope"], qkv["txt.q_qknorm"])
    source_k = (qkv["img.k_rope"], qkv["txt.k_qknorm"])
    source_v = (qkv["img.v_projection"], qkv["txt.v_projection"])
    invoke("t0_store_read", q=source_q, k=source_k, v=source_v, valid_len=0, update_cache=True)
    if set(consumer_captures) != {"t0_store_read"}:
        raise HistoryProbeError("T0 store/read did not produce exactly one consumer capture")
    stored_k = cache_buffers["k"][:, :tokens].detach().clone()
    stored_v = cache_buffers["v"][:, :tokens].detach().clone()
    read_t0_k, read_t0_v = consumer_captures["t0_store_read"]
    if tuple(stored_k.shape) != shape or tuple(stored_v.shape) != shape:
        raise HistoryProbeError("pinned source cache write extent differs from the admitted T0 sequence")
    if not torch.equal(stored_k, old_k) or not torch.equal(stored_v, old_v):
        raise HistoryProbeError("actual pinned cache function store differs from the old source cache capture")
    if (not torch.equal(read_t0_k, stored_k.to(torch.float32))
            or not torch.equal(read_t0_v, stored_v.to(torch.float32))):
        raise HistoryProbeError("T0 attention consumer did not read the persisted BF16 cache bytes")

    saved_k = cache_buffers["k"].detach().clone()
    saved_v = cache_buffers["v"].detach().clone()
    for tensor in (*source_q, *source_k, *source_v):
        tensor.zero_()
    invoke(
        "poisoned_read_only", q=source_q, k=source_k, v=source_v,
        valid_len=tokens, update_cache=False,
    )
    if set(consumer_captures) != {"t0_store_read", "poisoned_read_only"}:
        raise HistoryProbeError("source reader poison probe did not run exactly once")
    probe_full_k, probe_full_v = consumer_captures["poisoned_read_only"]
    if tuple(probe_full_k.shape) != (1, tokens * 2, HEADS, HEAD_DIM):
        raise HistoryProbeError("poison probe consumer K extent is unexpected")
    probe_prefix_k = probe_full_k[:, :tokens]
    probe_prefix_v = probe_full_v[:, :tokens]
    probe_current_k = probe_full_k[:, tokens:]
    probe_current_v = probe_full_v[:, tokens:]
    if (not torch.equal(probe_prefix_k, stored_k.to(torch.float32))
            or not torch.equal(probe_prefix_v, stored_v.to(torch.float32))):
        raise HistoryProbeError("read-only consumer prefix differs from persisted cache history")
    if (bool(torch.count_nonzero(probe_current_k).item())
            or bool(torch.count_nonzero(probe_current_v).item())):
        raise HistoryProbeError("read-only consumer current K/V suffix differs from zeroed input tensors")
    if (torch.equal(probe_prefix_k, probe_current_k)
            or torch.equal(probe_prefix_v, probe_current_v)):
        raise HistoryProbeError("poisoned current K/V did not distinguish the cached history prefix")
    if not torch.equal(cache_buffers["k"], saved_k) or not torch.equal(cache_buffers["v"], saved_v):
        raise HistoryProbeError("read-only source cache call mutated persistent K/V")
    if not bool(torch.isfinite(stored_k).all().item()) or not bool(torch.isfinite(stored_v).all().item()):
        raise HistoryProbeError("source cache store contains non-finite values")

    rows = tokens
    history = {
        "storage_dtype": "BF16",
        "read_dtype": "F32",
        "consumer_input_dtype": "BF16",
        "layout": LAYOUT,
        "cache_valid_len": 0,
        "cache_txt": True,
        "stored_tokens": rows,
        "cache_writes": 1,
        "cache_reads": 2,
        **_history_accounting(rows),
        "poisoned_read_only_probe": {
            "executed": True,
            "cache_valid_len": rows,
            "update_cache": False,
            "cache_txt": True,
            "current_qkv_poison": "zeroed in place after T0 store",
            "attention_consumer_intercepted": True,
            "history_prefix_matches_stored": True,
            "current_suffix_matches_poisoned_inputs": True,
            "history_prefix_differs_from_poisoned_current": True,
            "cache_unchanged": True,
            "history_k_f32_sha256": _tensor_sha256(probe_prefix_k),
            "history_v_f32_sha256": _tensor_sha256(probe_prefix_v),
        },
    }
    source_identity = {
        "source_receipt_sha256": source_receipt_sha,
        "source_artifact_sha256": source_artifact_sha,
        "input_bundle_manifest_sha256": input_bundle["manifest_sha256"],
        "input_bundle_payload_sha256": input_bundle["payload_sha256"],
        "reference_latent_artifact_sha256": input_bundle["reference_latent_artifact_sha256"],
        "timeline_manifest_sha256": input_bundle["timeline_manifest_sha256"],
        "valid_text_tokens": input_bundle["valid_text_tokens"],
        "weights_sha256": source["weight_source"]["sha256"],
    }
    receipt: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "upstream_revision": upstream_revision,
        "source_revision": SOURCE_REVISION,
        "scope": SOURCE_SCOPE,
        "full_upstream_t0_preprocessing_parity": False,
        "history_cache_write_executed": True,
        "history_cache_read_executed": True,
        "attention_executed": False,
        "source_input": source_identity,
        "cache_function": {
            **cache_function_identity,
            "executed": True,
            "total_function_invocations": 2,
            "store_invocations": 1,
            "consumer_read_invocations": 2,
            "poisoned_read_only_invocations": 1,
            "attention_consumer": "torch.nn.functional.scaled_dot_product_attention",
            "attention_consumer_intercepted": True,
            "attention_executed": False,
        },
        "history": history,
    }
    tensors = {
        "history.k_bf16": stored_k,
        "history.v_bf16": stored_v,
        "history.k_read_f32": read_t0_k,
        "history.v_read_f32": read_t0_v,
    }
    artifact_metadata = {
        "format": SOURCE_SCHEMA,
        "upstream_revision": upstream_revision,
        "cache_function_source_sha256": cache_function_identity["function_source_sha256"],
        "source_receipt_sha256": source_receipt_sha,
        "source_artifact_sha256": source_artifact_sha,
    }
    if (_hash_file(Path(source_receipt_path).expanduser()) != source_receipt_sha
            or _hash_file(input_artifact.path) != source_artifact_sha):
        raise HistoryProbeError("authenticated old source input changed during the cache run")
    result = _save_history_artifact(
        tensors, output_path=output_path, receipt_path=receipt_file,
        receipt=receipt, torch=torch, artifact_metadata=artifact_metadata,
    )
    del qkv, input_artifact, cache_buffers, stored_k, stored_v, tensors
    return result


def _validate_tensor_records(
    artifact: block0_compare.TensorArtifact, *, names: tuple[str, ...], rows: int,
    receipt_artifact: dict[str, Any], expected_metadata: dict[str, str],
) -> None:
    if (type(receipt_artifact) is not dict or set(receipt_artifact) != {
            "basename", "sha256", "bytes", "tensors"}
            or not _is_sha256(receipt_artifact.get("sha256"))
            or type(receipt_artifact.get("bytes")) is not int
            or receipt_artifact["bytes"] != artifact.path.stat().st_size
            or type(receipt_artifact.get("tensors")) is not list):
        raise HistoryProbeError("history tensor artifact receipt identity is invalid")
    try:
        bound_path = block0_compare._receipt_artifact_path(receipt_artifact, artifact.path, "basename")
    except block0_compare.CompareError as error:
        raise HistoryProbeError(str(error)) from error
    if bound_path != artifact.path:
        raise HistoryProbeError("history receipt names a different tensor artifact")
    expected_shapes = {name: (1, rows, HEADS, HEAD_DIM) for name in names}
    expected_dtypes = {
        "history.k_bf16": "BF16", "history.v_bf16": "BF16",
        "history.k_read_f32": "F32", "history.v_read_f32": "F32",
    }
    records = receipt_artifact["tensors"]
    if len(records) != len(names) or [r.get("name") if type(r) is dict else None for r in records] != list(names):
        raise HistoryProbeError("history tensor receipt order or inventory mismatch")
    if set(artifact.tensors) != set(names):
        raise HistoryProbeError("history safetensors inventory mismatch")
    for name, record in zip(names, records):
        if (type(record) is not dict or set(record) != {"name", "dtype", "shape", "nbytes", "sha256"}
                or record.get("name") != name or record.get("dtype") != expected_dtypes[name]
                or not _matches_exact_json(record.get("shape"), list(expected_shapes[name]))
                or type(record.get("nbytes")) is not int
                or record["nbytes"] != math.prod(expected_shapes[name]) * (2 if expected_dtypes[name] == "BF16" else 4)
                or not _is_sha256(record.get("sha256"))):
            raise HistoryProbeError(f"history tensor receipt record is invalid: {name}")
        actual = artifact.tensors[name]
        if (actual["dtype"] != expected_dtypes[name]
                or actual["shape"] != expected_shapes[name]
                or actual["nbytes"] != record["nbytes"]
                or artifact.tensor_sha256(name) != record["sha256"]):
            raise HistoryProbeError(f"history tensor metadata or digest mismatch: {name}")
    if artifact.metadata != expected_metadata:
        raise HistoryProbeError("history artifact safetensors metadata mismatch")
    if artifact.sha256 != receipt_artifact["sha256"]:
        raise HistoryProbeError("history artifact SHA-256 mismatch")
    artifact.check_identity()


def _validate_history_semantics(history: Any, *, rows: int, native: bool) -> None:
    if type(history) is not dict:
        raise HistoryProbeError("history receipt must contain a history object")
    expected = {
        "storage_dtype": "BF16", "read_dtype": "F32", "layout": LAYOUT,
        "cache_valid_len": 0, "cache_txt": True, "stored_tokens": rows,
        "cache_writes": 1, "cache_reads": 1 if native else 2,
        **_history_accounting(rows),
    }
    if native:
        if set(history) != set(expected):
            raise HistoryProbeError("native history fields differ from the history schema")
        candidate = history
    else:
        expected_source = {**expected, "consumer_input_dtype": "BF16", "cache_reads": 2,
                           "poisoned_read_only_probe": None}
        if set(history) != set(expected_source):
            raise HistoryProbeError("source history fields differ from the source schema")
        probe = history.get("poisoned_read_only_probe")
        probe_keys = {
            "executed", "cache_valid_len", "update_cache", "cache_txt", "current_qkv_poison",
            "attention_consumer_intercepted", "history_prefix_matches_stored",
            "current_suffix_matches_poisoned_inputs",
            "history_prefix_differs_from_poisoned_current", "cache_unchanged",
            "history_k_f32_sha256", "history_v_f32_sha256",
        }
        if (type(probe) is not dict or set(probe) != probe_keys
                or probe.get("executed") is not True or probe.get("cache_valid_len") != rows
                or type(probe.get("cache_valid_len")) is not int
                or probe.get("update_cache") is not False or probe.get("cache_txt") is not True
                or probe.get("current_qkv_poison") != "zeroed in place after T0 store"
                or probe.get("attention_consumer_intercepted") is not True
                or probe.get("history_prefix_matches_stored") is not True
                or probe.get("current_suffix_matches_poisoned_inputs") is not True
                or probe.get("history_prefix_differs_from_poisoned_current") is not True
                or probe.get("cache_unchanged") is not True
                or not _is_sha256(probe.get("history_k_f32_sha256"))
                or not _is_sha256(probe.get("history_v_f32_sha256"))):
            raise HistoryProbeError("source history poisoned read-only probe is not admitted")
        candidate = {key: history[key] for key in expected}
    if not _matches_exact_json(candidate, expected):
        raise HistoryProbeError("history store/read dtype, layout, extent, or accounting mismatch")


def _validate_source_history(
    receipt_path: str | Path,
    artifact_path: str | Path,
) -> tuple[dict[str, Any], block0_compare.TensorArtifact]:
    receipt = _read_json(receipt_path)
    artifact_file = _regular_path(artifact_path, "source history tensor artifact")
    expected_keys = {
        "schema", "schema_version", "upstream_revision", "source_revision", "scope",
        "full_upstream_t0_preprocessing_parity", "history_cache_write_executed",
        "history_cache_read_executed", "attention_executed", "source_input",
        "cache_function", "history", "tensor_artifact",
    }
    if set(receipt) != expected_keys:
        raise HistoryProbeError("source history receipt fields differ from the pinned schema")
    if (receipt.get("schema") != SOURCE_SCHEMA
            or type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("upstream_revision") != UPSTREAM_REVISION
            or receipt.get("source_revision") != SOURCE_REVISION
            or receipt.get("scope") != SOURCE_SCOPE
            or receipt.get("full_upstream_t0_preprocessing_parity") is not False
            or receipt.get("history_cache_write_executed") is not True
            or receipt.get("history_cache_read_executed") is not True
            or receipt.get("attention_executed") is not False):
        raise HistoryProbeError("source history receipt overstates or misidentifies its execution scope")
    source_input = receipt.get("source_input")
    source_input_keys = {
        "source_receipt_sha256", "source_artifact_sha256", "input_bundle_manifest_sha256",
        "input_bundle_payload_sha256", "reference_latent_artifact_sha256",
        "timeline_manifest_sha256", "valid_text_tokens", "weights_sha256",
    }
    if type(source_input) is not dict or set(source_input) != source_input_keys:
        raise HistoryProbeError("source history receipt input identity is malformed")
    for key in source_input_keys - {"valid_text_tokens"}:
        if not _is_sha256(source_input.get(key)):
            raise HistoryProbeError(f"source history receipt has malformed identity: {key}")
    rows = _expected_shape(source_input.get("valid_text_tokens"))[1]
    if source_input.get("weights_sha256") != BLOCK_WEIGHTS_SHA256:
        raise HistoryProbeError("source history weight pin mismatch")
    function = receipt.get("cache_function")
    expected_function_keys = {
        "name", "source_file", "source_file_sha256", "function_source_sha256",
        "executed", "total_function_invocations", "store_invocations",
        "consumer_read_invocations", "poisoned_read_only_invocations",
        "attention_consumer", "attention_consumer_intercepted", "attention_executed",
    }
    if (type(function) is not dict or set(function) != expected_function_keys
            or function.get("name") != "causal_attention_with_cache_mp"
            or function.get("source_file") != "hymm/models/dit/nomas_cpt.py"
            or function.get("source_file_sha256") != UPSTREAM_SOURCE_FILE_SHA256
            or function.get("function_source_sha256") != CACHE_FUNCTION_SOURCE_SHA256
            or function.get("executed") is not True
            or type(function.get("total_function_invocations")) is not int
            or function["total_function_invocations"] != 2
            or type(function.get("store_invocations")) is not int or function["store_invocations"] != 1
            or type(function.get("consumer_read_invocations")) is not int
            or function["consumer_read_invocations"] != 2
            or type(function.get("poisoned_read_only_invocations")) is not int
            or function["poisoned_read_only_invocations"] != 1
            or function.get("attention_consumer") != "torch.nn.functional.scaled_dot_product_attention"
            or function.get("attention_consumer_intercepted") is not True
            or function.get("attention_executed") is not False):
        raise HistoryProbeError("source receipt does not attest the pinned cache function and intercepted consumer")
    _validate_history_semantics(receipt.get("history"), rows=rows, native=False)
    try:
        artifact = block0_compare.TensorArtifact(
            artifact_file, expected_sha256=receipt["tensor_artifact"]["sha256"],
            expected_names=SOURCE_TENSOR_NAMES,
        )
    except (block0_compare.CompareError, KeyError, TypeError) as error:
        raise HistoryProbeError(f"source history artifact is invalid: {error}") from error
    _validate_tensor_records(
        artifact, names=SOURCE_TENSOR_NAMES, rows=rows,
        receipt_artifact=receipt.get("tensor_artifact"),
        expected_metadata={
            "format": SOURCE_SCHEMA,
            "upstream_revision": UPSTREAM_REVISION,
            "cache_function_source_sha256": function["function_source_sha256"],
            "source_receipt_sha256": source_input["source_receipt_sha256"],
            "source_artifact_sha256": source_input["source_artifact_sha256"],
        },
    )
    poison_probe = receipt["history"]["poisoned_read_only_probe"]
    if (poison_probe["history_k_f32_sha256"] != artifact.tensor_sha256("history.k_read_f32")
            or poison_probe["history_v_f32_sha256"] != artifact.tensor_sha256("history.v_read_f32")):
        raise HistoryProbeError("poisoned reader prefix digests differ from the captured T0 cache read")
    metadata = artifact.metadata
    if (metadata.get("upstream_revision") != UPSTREAM_REVISION
            or metadata.get("cache_function_source_sha256") != function["function_source_sha256"]
            or metadata.get("source_receipt_sha256") != source_input["source_receipt_sha256"]
            or metadata.get("source_artifact_sha256") != source_input["source_artifact_sha256"]):
        raise HistoryProbeError("source history artifact metadata is not bound to its receipt")
    return receipt, artifact


def _validate_native_history(
    receipt_path: str | Path,
    artifact_path: str | Path,
) -> tuple[dict[str, Any], block0_compare.TensorArtifact]:
    receipt = _read_json(receipt_path)
    artifact_file = _regular_path(artifact_path, "native history tensor artifact")
    rows = _validate_native_receipt_contract(receipt)
    try:
        artifact = block0_compare.TensorArtifact(
            artifact_file, expected_sha256=receipt["tensor_artifact"]["sha256"],
            expected_names=NATIVE_TENSOR_NAMES,
        )
    except (block0_compare.CompareError, KeyError, TypeError) as error:
        raise HistoryProbeError(f"native history artifact is invalid: {error}") from error
    _validate_tensor_records(
        artifact, names=NATIVE_TENSOR_NAMES, rows=rows,
        receipt_artifact=receipt.get("tensor_artifact"), expected_metadata={},
    )
    return receipt, artifact


def _validate_native_receipt_contract(receipt: dict[str, Any]) -> int:
    expected_keys = {
        "schema", "schema_version", "native_revision", "source_revision", "scope",
        "full_upstream_t0_preprocessing_parity", "history_cache_write_executed",
        "history_cache_read_executed", "attention_executed",
        "input_bundle_manifest_sha256", "input_bundle_payload_sha256",
        "reference_latent_artifact_sha256", "timeline_manifest_sha256", "valid_text_tokens",
        "text_storage_tokens", "weights_sha256", "precision_control", "bf16_linear_variant",
        "history", "tensor_artifact", "named_buffer_accounting", "execution_counters",
    }
    if set(receipt) != expected_keys:
        raise HistoryProbeError("native history receipt fields differ from the new history schema")
    if (receipt.get("schema") != NATIVE_SCHEMA
            or type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("native_revision") != UPSTREAM_REVISION
            or receipt.get("source_revision") != SOURCE_REVISION
            or receipt.get("scope") != NATIVE_SCOPE
            or receipt.get("full_upstream_t0_preprocessing_parity") is not False
            or receipt.get("history_cache_write_executed") is not True
            or receipt.get("history_cache_read_executed") is not True
            or receipt.get("attention_executed") is not False):
        raise HistoryProbeError("native history receipt overstates or misidentifies its execution scope")
    if (receipt.get("precision_control") != REFERENCE_BF16_PRECISION
            or receipt.get("bf16_linear_variant") != BF16_LINEAR_VARIANT):
        raise HistoryProbeError("native history precision control or BF16 linear variant mismatch")
    rows = _expected_shape(receipt.get("valid_text_tokens"))[1]
    identity = {
        "input_bundle_manifest_sha256": None,
        "input_bundle_payload_sha256": None,
        "reference_latent_artifact_sha256": None,
        "timeline_manifest_sha256": None,
    }
    for key in identity:
        if not _is_sha256(receipt.get(key)):
            raise HistoryProbeError(f"native history receipt has malformed identity: {key}")
    if (type(receipt.get("text_storage_tokens")) is not int
            or receipt["text_storage_tokens"] != TEXT_STORAGE_TOKENS
            or receipt.get("weights_sha256") != BLOCK_WEIGHTS_SHA256):
        raise HistoryProbeError("native history text storage extent or weight pin mismatch")
    _validate_history_semantics(receipt.get("history"), rows=rows, native=True)
    accounting = receipt.get("named_buffer_accounting")
    if type(accounting) is not dict or set(accounting) != ACCOUNTING_KEYS:
        raise HistoryProbeError("native history named buffer accounting keys are malformed")
    numeric = ACCOUNTING_KEYS - {"opaque_mpsgraph_private_bytes"}
    if (any(type(accounting.get(key)) is not int or accounting[key] < 0 for key in numeric)
            or accounting["reference_bf16_buffers"] <= 0
            or accounting["reference_history_buffers"] != rows * HIDDEN * 12
            or accounting["total_named_bytes"] != sum(
                accounting[key] for key in numeric - {"total_named_bytes"}
            )
            or accounting.get("opaque_mpsgraph_private_bytes")
            != "not measured; excluded from named Metal-buffer total"):
        raise HistoryProbeError("native history named buffer accounting is inconsistent")
    counters = receipt.get("execution_counters")
    if (type(counters) is not dict or set(counters) != EXECUTION_COUNTERS
            or any(type(counters.get(key)) is not int or counters[key] < 0 for key in EXECUTION_COUNTERS)
            or counters["encode_allocations"] != 0 or counters["encode_readbacks"] != 0
            or counters["preterminal_waits"] != 0 or counters["terminal_waits"] != 1):
        raise HistoryProbeError("native history execution counters violate the standalone terminal-drain contract")
    artifact_receipt = receipt.get("tensor_artifact")
    if (type(artifact_receipt) is not dict or set(artifact_receipt) != {
            "basename", "sha256", "bytes", "tensors"}
            or not _is_sha256(artifact_receipt.get("sha256"))
            or type(artifact_receipt.get("bytes")) is not int
            or type(artifact_receipt.get("tensors")) is not list):
        raise HistoryProbeError("native history tensor artifact receipt identity is malformed")
    records = artifact_receipt["tensors"]
    if (len(records) != len(NATIVE_TENSOR_NAMES)
            or [record.get("name") if type(record) is dict else None for record in records]
            != list(NATIVE_TENSOR_NAMES)):
        raise HistoryProbeError("native history tensor artifact inventory or order mismatch")
    for record, name in zip(records, NATIVE_TENSOR_NAMES):
        dtype = "BF16" if name.endswith("bf16") else "F32"
        extent = math.prod((1, rows, HEADS, HEAD_DIM)) * (2 if dtype == "BF16" else 4)
        if (type(record) is not dict or set(record) != {"name", "dtype", "shape", "nbytes", "sha256"}
                or record.get("name") != name or record.get("dtype") != dtype
                or not _matches_exact_json(record.get("shape"), [1, rows, HEADS, HEAD_DIM])
                or type(record.get("nbytes")) is not int or record["nbytes"] != extent
                or not _is_sha256(record.get("sha256"))):
            raise HistoryProbeError(f"native history tensor receipt record is invalid: {name}")
    return rows


def compare_native_history(
    source_history_receipt_path: str | Path,
    source_history_artifact_path: str | Path,
    native_receipt_path: str | Path,
    native_artifact_path: str | Path,
    *,
    output_path: str | Path | None = None,
    atol: float = ATOL,
    rtol: float = RTOL,
    chunk_elements: int = 262_144,
) -> dict[str, Any]:
    """Strictly compare source and native stored BF16 history and loaded F32 reads."""
    if type(atol) not in (int, float) or type(rtol) not in (int, float):
        raise HistoryProbeError("comparison tolerances must be numeric")
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol != ATOL or rtol != RTOL:
        raise HistoryProbeError("history comparison requires unchanged atol=rtol=2e-5")
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise HistoryProbeError("chunk_elements must be a positive integer")
    source, source_artifact = _validate_source_history(
        source_history_receipt_path, source_history_artifact_path
    )
    native, native_artifact = _validate_native_history(native_receipt_path, native_artifact_path)
    source_identity = source["source_input"]
    for native_field, source_field in (
        ("input_bundle_manifest_sha256", "input_bundle_manifest_sha256"),
        ("input_bundle_payload_sha256", "input_bundle_payload_sha256"),
        ("reference_latent_artifact_sha256", "reference_latent_artifact_sha256"),
        ("timeline_manifest_sha256", "timeline_manifest_sha256"),
        ("valid_text_tokens", "valid_text_tokens"),
        ("weights_sha256", "weights_sha256"),
    ):
        actual = native[native_field]
        expected = source_identity[source_field]
        if type(actual) is not type(expected) or actual != expected:
            raise HistoryProbeError(f"native/source history identity mismatch: {native_field}")
    if source["source_input"]["weights_sha256"] != BLOCK_WEIGHTS_SHA256:
        raise HistoryProbeError("source history did not bind the indexed BF16 weights")

    storage_results = []
    for name in ("history.k_bf16", "history.v_bf16"):
        if (source_artifact.tensors[name]["dtype"] != "BF16"
                or native_artifact.tensors[name]["dtype"] != "BF16"
                or source_artifact.tensors[name]["shape"] != native_artifact.tensors[name]["shape"]):
            raise HistoryProbeError(f"physical BF16 history tensor layout mismatch: {name}")
        source_hash = source_artifact.tensor_sha256(name)
        native_hash = native_artifact.tensor_sha256(name)
        if source_hash != native_hash:
            raise HistoryProbeError(f"physical BF16 stored cache bytes differ: {name}")
        storage_results.append({"tensor": name, "sha256": source_hash, "byte_exact": True})

    read_results = []
    for name in ("history.k_read_f32", "history.v_read_f32"):
        read_results.append(block0_compare.compare_tensor(
            native_artifact, name,
            [(source_artifact, name, 0, source_artifact.tensors[name]["count"])],
            atol=atol, rtol=rtol, chunk_elements=chunk_elements,
        ))
    source_artifact.check_identity()
    native_artifact.check_identity()
    result = {
        "schema": "cogni_ml.hy_nomas_reference_history_comparison/v1",
        "passed": all(record["passed"] for record in read_results),
        "precision_control": REFERENCE_BF16_PRECISION,
        "valid_text_tokens": source_identity["valid_text_tokens"],
        "atol": atol,
        "rtol": rtol,
        "source_artifact_sha256": source_artifact.sha256,
        "native_artifact_sha256": native_artifact.sha256,
        "source_input_receipt_sha256": source_identity["source_receipt_sha256"],
        "source_input_artifact_sha256": source_identity["source_artifact_sha256"],
        "input_bundle_manifest_sha256": source_identity["input_bundle_manifest_sha256"],
        "input_bundle_payload_sha256": source_identity["input_bundle_payload_sha256"],
        "reference_latent_artifact_sha256": source_identity["reference_latent_artifact_sha256"],
        "timeline_manifest_sha256": source_identity["timeline_manifest_sha256"],
        "weights_sha256": source_identity["weights_sha256"],
        "physical_bf16_storage": storage_results,
        "read_value_comparisons": read_results,
        "source_poisoned_read_only_probe": source["history"]["poisoned_read_only_probe"],
    }
    if output_path is not None:
        output = Path(output_path).expanduser()
        if (not output.is_absolute() or output.suffix.lower() != ".json"
                or output.is_symlink() or output.exists() or not output.parent.is_dir()):
            raise HistoryProbeError("comparison output must be a new absolute .json file")
        staged = output.parent / f".{output.name}.stage-{os.getpid()}"
        try:
            with staged.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, sort_keys=True, indent=2)
                stream.write("\n")
            os.link(staged, output)
        finally:
            staged.unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-receipt", type=Path,
                      help="old adapter-only BF16 source receipt used as immutable Q/K/V input")
    mode.add_argument("--compare-native", type=Path,
                      help="native history receipt to compare against a fresh source history receipt")
    parser.add_argument("--source-artifact", type=Path,
                        help="old source Q/K/V artifact; required with --source-receipt")
    parser.add_argument("--upstream", type=Path, default=cache_oracle.UPSTREAM_ROOT_DEFAULT)
    parser.add_argument("--output-safetensors", type=Path)
    parser.add_argument("--output-receipt", type=Path)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--source-history-receipt", type=Path)
    parser.add_argument("--source-history-artifact", type=Path)
    parser.add_argument("--native-artifact", type=Path)
    parser.add_argument("--output-comparison", type=Path)
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--chunk-elements", type=int, default=262_144)
    args = parser.parse_args(argv)
    if args.source_receipt is not None:
        if (args.source_artifact is None or args.output_safetensors is None
                or args.output_receipt is None or args.source_history_receipt is not None
                or args.source_history_artifact is not None or args.native_artifact is not None
                or args.output_comparison is not None):
            parser.error("source mode requires --source-artifact, --output-safetensors, and --output-receipt only")
        result = run_upstream_history_probe(
            args.source_receipt, args.source_artifact,
            upstream_root=args.upstream,
            output_safetensors=args.output_safetensors,
            receipt_path=args.output_receipt,
            device_name=args.device,
        )
        print(json.dumps({
            "schema": result["schema"], "receipt_path": result["receipt_path"],
            "tensor_artifact": result["tensor_artifact"], "history": result["history"],
            "cache_function": result["cache_function"],
        }, sort_keys=True))
        return 0
    if (args.source_history_receipt is None or args.source_history_artifact is None
            or args.native_artifact is None or args.output_comparison is None
            or args.source_artifact is not None or args.output_safetensors is not None
            or args.output_receipt is not None):
        parser.error("--compare-native requires both source-history files, --native-artifact, and --output-comparison")
    result = compare_native_history(
        args.source_history_receipt, args.source_history_artifact,
        args.compare_native, args.native_artifact,
        output_path=args.output_comparison,
        atol=args.atol, rtol=args.rtol, chunk_elements=args.chunk_elements,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
