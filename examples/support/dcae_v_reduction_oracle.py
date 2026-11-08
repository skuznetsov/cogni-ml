#!/usr/bin/env python3
"""Separate pinned-upstream MPS operator control; never native timing."""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

UPSTREAM = Path("/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8/hymm/models/vae/autoencoder_dcae_v.py")
UPSTREAM_SHA = "38163ebead9a6bb643d27843022c1757c45c6b8638c334a476aee5b373edf8c1"


def load(torch, path, dtype, shape):
    raw = bytearray(path.read_bytes())
    size = 4 if dtype == torch.int32 else 2
    import math
    if len(raw) != math.prod(shape) * size:
        raise ValueError(f"Wrong tensor byte length: {path}")
    bits = torch.frombuffer(raw, dtype=dtype)
    if dtype == torch.uint16:
        bits = bits.view(torch.bfloat16)
    return bits.reshape(shape).to("mps")


def save(torch, tensor, path):
    if path.exists():
        raise FileExistsError(path)
    if tensor.device.type != "cpu":
        raise RuntimeError("Exports must use the already completed terminal transfer")
    bits = tensor.detach().contiguous().view(torch.uint16).reshape(-1)
    path.write_bytes(ctypes.string_at(bits.data_ptr(), bits.numel() * 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--admitted", action="store_true")
    args = parser.parse_args()
    if not args.admitted:
        parser.error("Separate guard admission is required; native chain must already have drained")
    if sys.byteorder != "little":
        raise ValueError("Raw fixture requires little endian")
    source = UPSTREAM.read_bytes()
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA:
        raise ValueError("Pinned upstream source changed")
    m = json.loads((args.root / "manifest.json").read_text())
    shape = tuple(m["shape"])
    if shape not in ((1, 512, 40, 3, 5), (1, 512, 40, 72, 128)):
        raise ValueError("Unadmitted operator geometry")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    import torch.nn.functional as functional
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_grad_enabled(False)
    # Execute the two unmodified pinned upstream function ASTs, without
    # importing model packages or constructing/loading the full autoencoder.
    names = {"_conv3d_split_channel", "_custom_conv3d"}
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError("Upstream operator functions missing")
    scope = {"torch": torch, "F": functional, "Optional": Optional, "Sequence": Sequence}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(UPSTREAM), "exec"), scope)
    conv = scope["_custom_conv3d"]
    root = args.root
    cache_shape = (1, 512, 1, shape[3], shape[4])
    real_prefix = m.get("activation_prefix")
    if real_prefix:
        if shape != (1, 512, 40, 72, 128) or real_prefix["tile_count"] != 2:
            raise ValueError("Unadmitted real-activation prefix")
        inputs = [load(torch, root / f"{case}-input.bf16", torch.uint16, shape)
                  for case in ("invalid", "valid")]
        caches = [load(torch, root / f"{case}-cache.bf16", torch.uint16, cache_shape)
                  for case in ("invalid", "valid")]
    else:
        x = load(torch, root / "input.bf16", torch.uint16, shape)
        cache = load(torch, root / "cache.bf16", torch.uint16, cache_shape)
        inputs, caches = [x, x], [cache, cache]
    weight = load(torch, root / "weight.bf16", torch.uint16, (512, 512, 3, 3, 3))
    bias = load(torch, root / "bias.bf16", torch.uint16, (1, 512, 1, 1, 1))
    zeros = torch.zeros(cache_shape, dtype=torch.bfloat16, device="mps")
    outputs = []
    for index, case in enumerate(("invalid", "valid")):
        x = inputs[index]
        left = caches[index] if index else zeros
        indices = load(torch, root / f"{case}-indices.u32", torch.int32, (m["samples"],)).to(torch.int64)
        joined = torch.cat((left, x, zeros), dim=2)
        pre = conv(joined, weight, None, (1, 1, 1), (0, 1, 1), (1, 1, 1), 1)
        post = pre + bias
        # The exact upstream call supplies bias to Conv3D. Verify its lowering
        # rather than assuming it equals the exposed separate BF16 boundary.
        biased = conv(joined, weight, bias.reshape(512), (1, 1, 1), (0, 1, 1), (1, 1, 1), 1)
        if tuple(pre.shape) != shape or pre.dtype != torch.bfloat16 or pre.device.type != "mps":
            raise RuntimeError("Upstream operator dtype/shape/device mismatch")
        sampled_pre = torch.index_select(pre.reshape(-1), 0, indices)
        sampled_post = torch.index_select(post.reshape(-1), 0, indices)
        sampled_biased = torch.index_select(biased.reshape(-1), 0, indices)
        if any(t.device.type != "mps" for t in (sampled_pre, sampled_post, sampled_biased)):
            raise RuntimeError("Sampling left the GPU")
        outputs.append((case, sampled_pre, sampled_post, sampled_biased))
    # One terminal transfer after both GPU operator cases. This diagnostic
    # process is separate from the native chain and never native timing.
    packed = torch.stack([tensor for _, *values in outputs for tensor in values]).to("cpu")
    for index, (case, _, _, _) in enumerate(outputs):
        for offset, name in enumerate(("pre", "post", "biased")):
            save(torch, packed[index * 3 + offset], root / f"torch-{case}-{name}.bf16")
        index_path = root / f"torch-{case}-indices.u32"
        if index_path.exists():
            raise FileExistsError(index_path)
        index_path.write_bytes((root / f"{case}-indices.u32").read_bytes())
    report = {"schema": "dcae_conv3d_reduction_upstream_operator/v1", "torch_version": torch.__version__,
              "upstream_path": str(UPSTREAM), "upstream_sha256": UPSTREAM_SHA,
              "executed_functions": sorted(names), "shape": shape, "device": "mps", "cpu_fallback": False,
              "terminal_tensor_transfers": 1, "explicit_mps_synchronizations": 0,
              "activation_prefix": real_prefix,
              "scope": "Pinned upstream Conv3D function on the identical native-exported input/cache, with BF16 causal concat and separate bias; same Apple backend, sampled evidence only; no native timing, prefix authentication or full-model inference"}
    target = root / "upstream-operator.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
