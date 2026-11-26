#!/usr/bin/env python3
"""Create a bounded MPS reference for the pinned Hy-Nomas timestep embedding."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Any


PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
SOURCE_RELATIVE_PATH = "hymm/models/dit/modules/embed_layers.py"
SOURCE_SHA256 = "5b055e383bbf532ea5aa49ee506031d96bea3e3d5b8da3a104d4bd875c2d0af3"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
DIM = 256
MAX_PERIOD = 10_000
TIMESTEPS = [-1.0, 0.0, 0.125, 1.0, 499.5, 500.0, 999.25, 1000.0]
SCHEMA = "hy-nomas-timestep-embedding-reference-v1"
MAX_OUTPUT_BYTES = 1024 * 1024


class ReferenceError(RuntimeError):
    """A fail-closed input, provenance, or MPS execution error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference_document(document: dict[str, Any]) -> None:
    """Validate the compact frequency-basis and dynamic-output fixture."""
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1):
        raise ReferenceError("missing or invalid timestep-reference schema")
    if document.get("upstream_revision") != PINNED_REVISION:
        raise ReferenceError("timestep-reference upstream revision mismatch")

    source = document.get("source")
    if not isinstance(source, dict):
        raise ReferenceError("timestep-reference source provenance is missing")
    if (source.get("git_commit") != PINNED_REVISION
            or source.get("path") != SOURCE_RELATIVE_PATH
            or source.get("sha256") != SOURCE_SHA256
            or source.get("function") != "timestep_embedding"):
        raise ReferenceError("timestep-reference source identity mismatch")

    spec = document.get("spec")
    if not isinstance(spec, dict):
        raise ReferenceError("timestep-reference spec is missing")
    timesteps = spec.get("timesteps")
    if (type(spec.get("dim")) is not int or spec.get("dim") != DIM
            or type(spec.get("max_period")) is not int or spec.get("max_period") != MAX_PERIOD
            or not isinstance(timesteps, list) or timesteps != TIMESTEPS
            or any(type(value) is not float for value in timesteps)):
        raise ReferenceError("timestep-reference dimensions or timestep set mismatch")

    basis = document.get("frequency_basis")
    bits = basis.get("bits") if isinstance(basis, dict) else None
    if (not isinstance(basis, dict) or basis.get("dtype") != "F32"
            or basis.get("shape") != [DIM // 2] or not isinstance(bits, list)
            or len(bits) != DIM // 2
            or basis.get("captured_from") != "the single CPU torch.exp result before upstream .to(t.device)"
            or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in bits)):
        raise ReferenceError("timestep-reference frequency basis is malformed")

    expected = document.get("expected")
    values = expected.get("values") if isinstance(expected, dict) else None
    if (not isinstance(expected, dict) or expected.get("dtype") != "F32"
            or expected.get("shape") != [len(TIMESTEPS), DIM]
            or expected.get("computed_on") != "MPS by the pinned upstream function for every listed timestep"
            or not isinstance(values, list) or len(values) != len(TIMESTEPS)):
        raise ReferenceError("timestep-reference output tensor metadata mismatch")
    for row in values:
        if (not isinstance(row, list) or len(row) != DIM
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in row)):
            raise ReferenceError("timestep-reference output values are malformed")

    producer = document.get("producer")
    if (not isinstance(producer, dict) or producer.get("device") != "mps"
            or producer.get("activation_dtype") != "F32"
            or producer.get("mps_fallback") != "disabled"
            or not isinstance(producer.get("python"), str)
            or not isinstance(producer.get("torch_version"), str)
            or not isinstance(producer.get("torch_git_version"), str)
            or not isinstance(producer.get("argv"), list)
            or any(not isinstance(value, str) for value in producer["argv"])):
        raise ReferenceError("timestep-reference producer policy mismatch")
    for field in ("source_sha256",):
        digest = producer.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ReferenceError(f"invalid producer {field}")


def _extract_function(source_bytes: bytes, function_name: str = "timestep_embedding") -> ast.Module:
    """Return only the selected upstream function, preserving its AST body."""
    try:
        tree = ast.parse(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError, ValueError, RecursionError):
        raise ReferenceError("pinned embed_layers.py is not valid UTF-8 Python") from None
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise ReferenceError(f"pinned source must define exactly one {function_name} function")
    function = matches[0]
    positional = [argument.arg for argument in function.args.args]
    if positional != ["t", "dim", "max_period"] or len(function.args.defaults) != 1:
        raise ReferenceError("pinned timestep_embedding signature changed")
    module = ast.Module(body=[function], type_ignores=[])
    return ast.fix_missing_locations(module)


class _ExpCapture:
    """Delegate torch operations and record the exact CPU exp result once."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self.results: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def exp(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        if self.results:
            raise ReferenceError("upstream timestep embedding called torch.exp more than once")
        result = self._torch.exp(value, *args, **kwargs)
        if result.device.type != "cpu" or result.dtype != self._torch.float32 or tuple(result.shape) != (DIM // 2,):
            raise ReferenceError("upstream frequency basis must be one CPU F32 vector")
        self.results.append(result.detach().clone())
        return result


def _git_revision(upstream: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        raise ReferenceError("cannot verify pinned upstream Git revision") from None
    revision = result.stdout.strip()
    if revision != PINNED_REVISION:
        raise ReferenceError(f"upstream Git revision mismatch: {revision}")
    return revision


def _publish_noreplace(path: Path, payload: bytes) -> None:
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ReferenceError("timestep-reference JSON exceeds the 1 MiB bound")
    if not path.parent.is_dir():
        raise ReferenceError(f"output parent must already exist: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"timestep-reference output already exists: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fixture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", required=True, type=Path,
                        help="new JSON output path in an existing directory")
    return parser


def produce_fixture(args: argparse.Namespace) -> dict[str, Any]:
    """Run one exact upstream embedding call on MPS and publish JSON only."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    requested_output = Path(args.output)
    if requested_output.is_symlink():
        raise ReferenceError("--output cannot be a symlink")
    output = requested_output.resolve()
    if output.suffix.lower() != ".json" or output.exists() or output.is_symlink():
        raise ReferenceError("--output must be a new .json file")
    if not output.parent.is_dir():
        raise ReferenceError("--output parent must already exist")

    upstream = args.upstream.resolve(strict=True)
    revision = _git_revision(upstream)
    source_path = upstream / SOURCE_RELATIVE_PATH
    if source_path.is_symlink() or not source_path.is_file():
        raise ReferenceError("pinned timestep source must be a regular file")
    source_bytes = source_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise ReferenceError(f"pinned timestep source digest mismatch: {source_digest}")
    function_ast = _extract_function(source_bytes)

    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    device = torch.device("mps")
    capture = _ExpCapture(torch)
    namespace: dict[str, Any] = {"math": math, "torch": capture}
    code = compile(function_ast, str(source_path), "exec")
    exec(code, namespace)
    timestep_embedding = namespace["timestep_embedding"]

    timestep_tensor = torch.tensor(TIMESTEPS, dtype=torch.float32, device=device)
    with torch.inference_mode():
        output_tensor = timestep_embedding(timestep_tensor, DIM, MAX_PERIOD)
    if output_tensor.device.type != "mps" or output_tensor.dtype != torch.float32:
        raise ReferenceError("upstream timestep outputs must remain F32 on MPS")
    if tuple(output_tensor.shape) != (len(TIMESTEPS), DIM):
        raise ReferenceError(f"upstream timestep output shape mismatch: {tuple(output_tensor.shape)}")
    if len(capture.results) != 1:
        raise ReferenceError("upstream timestep frequency basis was not captured exactly once")

    # The captured exp result is source-native CPU work: upstream constructs
    # arange/exp on CPU, then copies only the positional basis to t.device.
    # The embedding values themselves are copied back exactly once, after all
    # eight timesteps have completed on MPS.
    frequency_values = capture.results[0].tolist()
    if any(not math.isfinite(value) for value in frequency_values):
        raise ReferenceError("upstream frequency basis contains a nonfinite value")
    frequency_bits = [
        struct.unpack("<I", struct.pack("<f", value))[0]
        for value in frequency_values
    ]
    output_cpu = output_tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    expected_values = output_cpu.tolist()
    if any(not math.isfinite(value) for row in expected_values for value in row):
        raise ReferenceError("MPS timestep output contains a nonfinite value")

    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "upstream_revision": revision,
        "source": {
            "git_commit": revision,
            "path": SOURCE_RELATIVE_PATH,
            "sha256": source_digest,
            "function": "timestep_embedding",
            "execution": "single exact FunctionDef AST extracted from the pinned file",
        },
        "producer": {
            "path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "argv": list(sys.argv),
            "python": sys.version.split()[0],
            "torch_version": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "device": "mps",
            "activation_dtype": "F32",
            "mps_fallback": "disabled",
        },
        "spec": {
            "dim": DIM,
            "max_period": MAX_PERIOD,
            "timesteps": TIMESTEPS,
        },
        "frequency_basis": {
            "dtype": "F32",
            "shape": [DIM // 2],
            "bits": frequency_bits,
            "captured_from": "the single CPU torch.exp result before upstream .to(t.device)",
        },
        "expected": {
            "dtype": "F32",
            "shape": [len(TIMESTEPS), DIM],
            "values": expected_values,
            "computed_on": "MPS by the pinned upstream function for every listed timestep",
        },
        "provenance_note": (
            "The positional frequency basis is source-derived CPU F32 data; all eight "
            "timestep multiplications and trigonometric outputs were evaluated dynamically "
            "on MPS. No timestep activation constants were used."
        ),
    }
    validate_reference_document(document)
    payload = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _publish_noreplace(output, payload)
    return {
        "status": "created",
        "output": str(output),
        "frequency_count": len(frequency_bits),
        "timestep_count": len(TIMESTEPS),
        "output_shape": [len(TIMESTEPS), DIM],
        "upstream_revision": revision,
        "source_sha256": source_digest,
    }


def main(argv: list[str] | None = None) -> int:
    args = _fixture_parser().parse_args(argv)
    try:
        result = produce_fixture(args)
    except (OSError, ValueError, ReferenceError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
