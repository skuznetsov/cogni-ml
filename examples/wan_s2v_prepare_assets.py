#!/usr/bin/env python3
"""Export user-downloaded Wan .pth tensors as exact SafeTensors bytes.

Setup-only I/O, no neural inference, precision change or checkpoint download.
Uses restricted PyTorch loading and maps source storage instead of keeping two
whole checkpoints resident. Run with the project's memory/deadline guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import torch

DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    before = args.source.stat()
    state = torch.load(args.source, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Expected a nonempty flat tensor state_dict")
    descriptors = {}
    offset = 0
    for name, tensor in sorted(state.items()):
        if tensor.dtype not in DTYPES or not tensor.is_contiguous() or tensor.layout != torch.strided:
            raise ValueError(f"Unsupported dtype/layout for {name}: {tensor.dtype}/{tensor.layout}")
        size = tensor.numel() * tensor.element_size()
        descriptors[name] = dict(dtype=DTYPES[tensor.dtype], shape=list(tensor.shape), data_offsets=[offset, offset + size])
        offset += size
    source_sha = digest(args.source)
    header = {"__metadata__": {"source_revision": args.revision, "source_sha256": source_sha,
                               "conversion": "exact original contiguous tensor bytes"}, **descriptors}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(args.destination.suffix + ".partial")
    created = False
    try:
        with temporary.open("xb") as output:
            created = True
            output.write(struct.pack("<Q", len(raw)))
            output.write(raw)
            for name in sorted(state):
                tensor = state[name].detach()
                # Reinterpretation only. No BF16 -> F32 conversion or tensor math.
                data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
                for start in range(0, len(data), 16 * 1024 * 1024):
                    output.write(data[start:start + 16 * 1024 * 1024])
            output.flush()
            os.fsync(output.fileno())
        after = args.source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError("Source checkpoint changed during conversion")
        os.link(temporary, args.destination)  # Refuse to overwrite a concurrent result.
        temporary.unlink()
        report = dict(source=str(args.source), source_sha256=source_sha, revision=args.revision,
                      destination=str(args.destination), destination_sha256=digest(args.destination),
                      tensors=len(state), payload_bytes=offset, dtypes=sorted({DTYPES[t.dtype] for t in state.values()}))
        args.destination.with_suffix(".conversion.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        if created:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
