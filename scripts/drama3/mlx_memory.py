#!/usr/bin/env python3
"""Inspect DRAMA core weight headers without importing or launching a model.

MemoryTrace is an opt-in, single-threaded diagnostic for the pinned one-row MLX
prefill. It observes existing evaluations; it is not a memory-leak fix or a
performance benchmark. MLX and model modules are deliberately not imported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any, Callable, TextIO


CORE_PREFIXES = ("backbone.", "lm_head.", "tower.")
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
    "U32": 4, "I32": 4, "U64": 8, "I64": 8,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
}


def core_weight_inventory(model_dir: Path) -> dict[str, Any]:
    """Read only index JSON and safetensors headers, never tensor payloads."""
    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_bytes())
    selected = {name: shard for name, shard in index["weight_map"].items()
                if name.startswith(CORE_PREFIXES)}
    if not selected:
        raise ValueError("checkpoint has no indexed core tensors")
    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"invalid checkpoint shard for {name}")
        by_shard.setdefault(shard, []).append(name)
    stored_bytes = f32_bytes = packed_bytes = 0
    for shard, names in sorted(by_shard.items()):
        path = model_dir / shard
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors header: {shard}")
            length = struct.unpack("<Q", prefix)[0]
            if not 1 <= length <= min(64 * 1024 * 1024, path.stat().st_size - 8):
                raise ValueError(f"invalid safetensors header size: {shard}")
            header = json.loads(stream.read(length))
        payload_bytes = path.stat().st_size - 8 - length
        for name in names:
            if name not in header:
                raise ValueError(f"indexed tensor missing from {shard}: {name}")
            tensor = header[name]
            dtype, shape = tensor["dtype"], tensor["shape"]
            if dtype not in DTYPE_BYTES or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"unsupported tensor metadata: {name}")
            first, last = tensor["data_offsets"]
            size = math.prod(shape) * DTYPE_BYTES[dtype]
            if not (0 <= first <= last <= payload_bytes) or last - first != size:
                raise ValueError(f"invalid tensor byte range: {name}")
            stored_bytes += size
            f32_bytes += math.prod(shape) * 4 if dtype in ("BF16", "F16", "F32", "F64") else size
            if dtype == "U32":
                packed_bytes += size
    return {
        "format": "drama3-core-weight-inventory-v1",
        "scope": "Header arithmetic for the F32 core loader; not measured residency or peak memory.",
        "tensor_count": len(selected), "stored_bytes": stored_bytes,
        "f32_core_bytes": f32_bytes, "packed_u32_bytes": packed_bytes,
    }


def cache_inventory(groups: dict[str, Any]) -> dict[str, Any]:
    """Count logical backing arrays without slicing states or reading values.

    These bytes are metadata, not unique physical allocations: views/aliases and
    allocator scratch can make the actual process footprint different.
    """
    result = {}
    for name, caches in groups.items():
        layers = []
        total = 0
        for cache in caches:
            if hasattr(cache, "keys") and hasattr(cache, "values"):
                values = (cache.keys, cache.values)
            elif hasattr(cache, "cache"):
                values = cache.cache
            else:
                raise TypeError(f"unsupported cache type: {type(cache).__name__}")
            arrays = [{"shape": list(v.shape), "dtype": str(v.dtype), "nbytes": int(v.nbytes)}
                      for v in values if v is not None]
            layer = {"type": type(cache).__name__, "arrays": arrays}
            offset = getattr(cache, "offset", None)
            if type(offset) is int:
                layer["offset"] = offset
            layers.append(layer)
            total += sum(v["nbytes"] for v in arrays)
        result[name] = {"array_bytes": total, "layers": layers}
    return result


def process_rss_bytes() -> int:
    """Current process RSS via the host OS; does not inspect GPU values.

    This diagnostic subprocess adds host overhead. Do not use traced timings as
    performance results. RSS includes mappings and is not private/physical GPU
    allocation accounting.
    """
    return int(subprocess.check_output(
        ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())], text=True,
    ).strip()) * 1024


class MemoryBudgetExceeded(RuntimeError):
    """An observed boundary exceeded the explicit diagnostic budget."""


class MemoryTrace:
    def __init__(self, mx: Any, stream: TextIO, *, budget_bytes: int | None = None,
                 rss_reader: Callable[[], int] = process_rss_bytes):
        if budget_bytes is not None and (type(budget_bytes) is not int or budget_bytes <= 0):
            raise ValueError("budget_bytes must be a positive integer")
        self.mx, self.stream, self.budget_bytes = mx, stream, budget_bytes
        self.rss_reader = rss_reader
        self._in_prefill = False

    def snapshot(self, stage: str, groups: dict[str, Any] | None = None,
                 *, completed_rows: int | None = None) -> None:
        # Counters are host metadata. No eval, synchronize, array conversion,
        # scalar .item(), cache clearing, or allocator policy changes occur here.
        active = int(self.mx.get_active_memory())
        cached = int(self.mx.get_cache_memory())
        peak = int(self.mx.get_peak_memory())
        rss = self.rss_reader()
        # MLX peak is cumulative since process start (this tracer never resets
        # it). Include transient active allocations released before this sample.
        exceeded = self.budget_bytes is not None and max(active + cached, peak, rss) > self.budget_bytes
        record = {
            "format": "drama3-mlx-memory-v1", "stage": stage, "pid": os.getpid(),
            "monotonic_seconds": time.monotonic(), "completed_rows": completed_rows,
            "mlx_active_bytes": active, "mlx_cached_bytes": cached,
            "mlx_peak_active_bytes": peak,
            "process_rss_bytes": rss, "caches": cache_inventory(groups or {}),
            "budget_bytes": self.budget_bytes, "budget_exceeded": exceeded,
        }
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        if exceeded:
            raise MemoryBudgetExceeded(f"memory budget exceeded at {stage}, row {completed_rows}")

    def prefill(self, model: Any, prompt: Any, backbone: Any, tower: Any, *, stage: str) -> Any:
        """Delegate to pinned rowwise upstream prefill, observing its mx.eval.

        Only for an isolated, single-threaded diagnostic. Require the pinned
        method's PREFILL_CHUNK_SIZE global to be 1 before entering the model.
        The final evaluation-count check detects additional schedule drift. The
        budget is checked before entry and after existing evals, including MLX's
        cumulative peak; it cannot interrupt a peak inside one evaluation.
        External run_safe pressure guards remain
        necessary. Exceptions restore the original eval function.
        """
        if self._in_prefill:
            raise ValueError("nested prefill memory tracing is unsupported")
        prefill = model.prefill
        function = getattr(prefill, "__func__", None)
        globals_ = getattr(function, "__globals__", {})
        chunk_size = globals_.get("PREFILL_CHUNK_SIZE")
        if type(chunk_size) is not int or chunk_size != 1:
            raise ValueError("memory tracing requires pinned prefill with PREFILL_CHUNK_SIZE=1")
        rows = int(prompt.shape[1])
        if rows < 1:
            raise ValueError("prefill needs a nonempty prompt")
        groups = {"backbone": backbone, "tower": tower}
        self.snapshot(stage, groups, completed_rows=0)
        original_eval = self.mx.eval
        completed = 0

        def observed_eval(*args: Any, **kwargs: Any) -> Any:
            nonlocal completed
            result = original_eval(*args, **kwargs)
            completed += 1
            if completed > rows:
                raise ValueError("upstream prefill evaluation count exceeds prompt rows")
            self.snapshot(stage, groups, completed_rows=completed)
            return result

        self._in_prefill = True
        self.mx.eval = observed_eval
        try:
            result = prefill(prompt, backbone, tower)
            if completed != rows:
                raise ValueError(f"upstream prefill evaluation count {completed} != {rows}")
            return result
        finally:
            self.mx.eval = original_eval
            self._in_prefill = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(core_weight_inventory(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
