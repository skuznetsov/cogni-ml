#!/usr/bin/env python3
"""Extract Qwen3.6 MTP tensors from remote HF safetensors shards.

The official Qwen3.6 checkpoints keep the built-in multi-token predictor under
`mtp.*`. Common GGUF conversions currently drop those tensors. This utility
range-downloads only the MTP byte ranges and repacks them into a compact
MTP-only safetensors sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from huggingface_hub import hf_hub_url


DEFAULT_REPO = "Qwen/Qwen3.6-27B"
DEFAULT_OUT = (
    Path.home()
    / ".cache"
    / "cogni-ml"
    / "qwen36_mtp"
    / "Qwen3.6-27B-mtp.safetensors"
)


@dataclass(frozen=True)
class TensorRef:
    name: str
    shard: str
    dtype: str
    shape: list[int]
    source_start: int
    source_end: int

    @property
    def nbytes(self) -> int:
        return self.source_end - self.source_start


def range_read(url: str, start: int, end_inclusive: int) -> bytes:
    expected = end_inclusive - start + 1
    if shutil.which("curl"):
        args = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "--range",
            f"{start}-{end_inclusive}",
            "-sS",
            url,
        ]
        for attempt in range(3):
            try:
                data = subprocess.check_output(args)
                if len(data) != expected:
                    raise RuntimeError(f"range read returned {len(data)} bytes, expected {expected}")
                return data
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end_inclusive}",
            "Accept-Encoding": "identity",
            "User-Agent": "cogni-ml-mtp-sidecar/1",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            break
        except Exception:
            if attempt == 2:
                raise
    if len(data) != expected:
        raise RuntimeError(f"range read returned {len(data)} bytes, expected {expected}")
    return data


def url_read(url: str) -> bytes:
    if shutil.which("curl"):
        return subprocess.check_output(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--retry-all-errors",
                "-sS",
                url,
            ]
        )
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def range_copy(url: str, start: int, end_inclusive: int, out: BinaryIO) -> None:
    expected = end_inclusive - start + 1
    if shutil.which("curl"):
        base_pos = out.tell()
        args = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "--range",
            f"{start}-{end_inclusive}",
            "-sS",
            url,
        ]
        for attempt in range(3):
            proc = subprocess.Popen(args, stdout=subprocess.PIPE)
            assert proc.stdout is not None
            shutil.copyfileobj(proc.stdout, out, length=8 * 1024 * 1024)
            rc = proc.wait()
            if rc == 0 and out.tell() - base_pos == expected:
                return
            if attempt == 2:
                got = out.tell() - base_pos
                raise RuntimeError(f"curl range download failed with exit code {rc}, bytes={got}, expected={expected}")
            out.seek(base_pos)
            out.truncate()
            out.flush()

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end_inclusive}",
            "Accept-Encoding": "identity",
            "User-Agent": "cogni-ml-mtp-sidecar/1",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                shutil.copyfileobj(resp, out, length=8 * 1024 * 1024)
            return
        except Exception:
            if attempt == 2:
                raise


def load_safetensors_header(repo_id: str, shard: str, revision: str | None) -> dict:
    url = hf_hub_url(repo_id=repo_id, filename=shard, revision=revision)
    header_len = struct.unpack("<Q", range_read(url, 0, 7))[0]
    header = range_read(url, 8, 8 + header_len - 1)
    return json.loads(header), header_len


def discover_mtp_tensors(repo_id: str, revision: str | None) -> list[TensorRef]:
    rev = revision or "main"
    index_url = f"https://huggingface.co/{repo_id}/raw/{rev}/model.safetensors.index.json"
    index = json.loads(url_read(index_url))

    mtp_by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        if name.startswith("mtp."):
            mtp_by_shard.setdefault(shard, []).append(name)

    refs: list[TensorRef] = []
    for shard, names in sorted(mtp_by_shard.items()):
        header, header_len = load_safetensors_header(repo_id, shard, revision)
        data_base = 8 + header_len
        for name in sorted(names):
            info = header[name]
            start, end = info["data_offsets"]
            refs.append(
                TensorRef(
                    name=name,
                    shard=shard,
                    dtype=info["dtype"],
                    shape=[int(v) for v in info["shape"]],
                    source_start=data_base + int(start),
                    source_end=data_base + int(end),
                )
            )
    return refs


def build_sidecar_header(refs: list[TensorRef], repo_id: str, revision: str | None) -> bytes:
    offset = 0
    header: dict[str, object] = {
        "__metadata__": {
            "format": "pt",
            "source_repo": repo_id,
            "source_revision": revision or "main",
            "extractor": "scripts/qwen36_extract_mtp_sidecar.py",
            "tensor_count": str(len(refs)),
        }
    }
    for ref in refs:
        header[ref.name] = {
            "dtype": ref.dtype,
            "shape": ref.shape,
            "data_offsets": [offset, offset + ref.nbytes],
        }
        offset += ref.nbytes

    # Safetensors permits whitespace padding in the JSON header. Padding keeps
    # the data region 8-byte aligned for BF16/F32 consumers.
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (8 - ((8 + len(raw)) % 8)) % 8
    return raw + b" " * pad


def write_sidecar(repo_id: str, revision: str | None, refs: list[TensorRef], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = build_sidecar_header(refs, repo_id, revision)
    shard_urls = {
        ref.shard: hf_hub_url(repo_id=repo_id, filename=ref.shard, revision=revision)
        for ref in refs
    }

    total = sum(ref.nbytes for ref in refs)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(struct.pack("<Q", len(header)))
        tmp.write(header)
        done = 0
        for ref in refs:
            print(
                f"extract {ref.name} {ref.nbytes / (1024 * 1024):.2f} MiB "
                f"from {ref.shard}",
                file=sys.stderr,
            )
            range_copy(shard_urls[ref.shard], ref.source_start, ref.source_end - 1, tmp)
            done += ref.nbytes
            print(f"  progress {done / (1024 * 1024):.2f}/{total / (1024 * 1024):.2f} MiB", file=sys.stderr)
    os.replace(tmp_path, out_path)


def print_summary(refs: list[TensorRef]) -> None:
    print(f"MTP tensors: {len(refs)}")
    total = 0
    for ref in refs:
        total += ref.nbytes
        shape = "x".join(str(v) for v in ref.shape)
        print(f"{ref.name:55s} {ref.dtype:5s} {shape:18s} {ref.nbytes / (1024 * 1024):8.2f} MiB {ref.shard}")
    print(f"total: {total / (1024 * 1024):.2f} MiB")


def verify_sidecar(path: Path) -> None:
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - defensive local tool check
        print(f"warning: safetensors verification unavailable: {exc}", file=sys.stderr)
        return

    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
    mtp_keys = [k for k in keys if k.startswith("mtp.")]
    if len(mtp_keys) != len(keys) or not mtp_keys:
        raise RuntimeError(f"unexpected sidecar keys: {keys[:5]} ... total={len(keys)}")
    print(f"verified sidecar: {path} ({len(mtp_keys)} tensors)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--revision", default=None, help="HF revision/commit (default: main)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output safetensors path (default: {DEFAULT_OUT})")
    parser.add_argument("--dry-run", action="store_true", help="print tensor table without downloading tensor payloads")
    args = parser.parse_args()

    refs = discover_mtp_tensors(args.repo, args.revision)
    print_summary(refs)
    if args.dry_run:
        return 0
    write_sidecar(args.repo, args.revision, refs, args.out)
    verify_sidecar(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
