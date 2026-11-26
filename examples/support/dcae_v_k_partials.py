#!/usr/bin/env python3
"""Bounded stage-2 Conv3D probe. All reductions and sampling stay on MPS."""

from __future__ import annotations

import argparse
from array import array
import ctypes
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys

from dcae_v_reduction_oracle import UPSTREAM_SHA, pinned_conv3d, read_tensor


CASES = ("invalid", "valid")
SHAPE = (1, 512, 40, 72, 128)
CACHE = (1, 512, 1, 72, 128)
WEIGHT = (512, 512, 3, 3, 3)
SAMPLES = 8192
TILE_WIDTH = 64
TILE_COUNT = math.prod(WEIGHT[1:]) // TILE_WIDTH
# Include three temporal taps and an adjacent pair in the middle tap.
SPARSE_TILES = (33, 106, 107, 182)
SINGLETON_CHANNEL_STRIDE = 8
SHADER = Path(__file__).with_suffix(".metal")
PRIOR_EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / ".landmark/evidence/2026-11-08-dcae-real-stage2-reduction.json"
)


def fixture_pins(root):
    prior = json.loads(PRIOR_EVIDENCE.read_text())
    manifest_pin = prior["raw_artifacts"]["manifest"]["sha256"]
    manifest = (root.parent / "artifact-manifest.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != manifest_pin:
        raise ValueError("The prior authenticated artifact manifest has changed")
    real_pins = {
        Path(record["path"]).name: record["sha256"]
        for record in json.loads(manifest)["artifacts"]
        if Path(record["path"]).parent.name == "real"
    }
    names = ["weight.bf16"]
    for case in CASES:
        names += [
            f"{case}-input.bf16",
            f"{case}-cache.bf16",
            f"{case}-indices.u32",
            f"{case}-mps-f32.f32",
            f"{case}-order0-lanes1-tree0.pre.bf16",
            f"torch-{case}-pre.bf16",
        ]
    return manifest_pin, {name: real_pins[name] for name in names}


def host_fixture(torch, root, pins):
    files = {"weight.bf16": (torch.uint16, WEIGHT)}
    for case in CASES:
        indices = f"{case}-indices.u32"
        raw = (root / indices).read_bytes()
        coordinates = array("I", raw)
        if (
            hashlib.sha256(raw).hexdigest() != pins[indices]
            or len(coordinates) != SAMPLES
            or max(coordinates) >= math.prod(SHAPE)
        ):
            raise ValueError(f"Invalid sampled coordinates: {indices}")
        files.update(
            {
                f"{case}-input.bf16": (torch.uint16, SHAPE),
                f"{case}-cache.bf16": (torch.uint16, CACHE),
                indices: (torch.int32, (SAMPLES,)),
            }
        )
    return {
        name: read_tensor(torch, root / name, dtype, shape, sha256=pins[name])
        for name, (dtype, shape) in files.items()
    }


def native_samples(torch, kernels, joined, weight, indices, tiles, partials):
    local_partials, merge_partials = kernels
    local = torch.empty(
        (2, len(SPARSE_TILES), SAMPLES), dtype=torch.float32, device="mps"
    )
    full = torch.empty((4, SAMPLES), dtype=torch.float32, device="mps")
    if not all(t.is_contiguous() for t in (joined, weight, indices, tiles)):
        raise ValueError("Metal probe requires dense NCTHW/OIktkhkw storage")
    local_partials(
        joined,
        weight,
        indices,
        tiles,
        partials,
        local,
        threads=SAMPLES * TILE_COUNT,
    )
    # Both merge orders read each exact same FP32 partial vector.
    merge_partials(partials, full, threads=2 * SAMPLES)
    return full, local


def upstream_sample(torch, conv, joined, weight, indices):
    out = conv(joined, weight, None, (1, 1, 1), (0, 1, 1), (1, 1, 1), 1)
    if (
        tuple(out.shape) != SHAPE
        or out.dtype != weight.dtype
        or out.device.type != "mps"
    ):
        raise RuntimeError("Upstream Conv3D shape/dtype/device changed")
    # Keep only sampled rows alive across full-shape calls.
    return torch.index_select(out.reshape(-1), 0, indices)


def sum_trees(parts):
    """One representative per commutative four-leaf binary FP32 add tree."""
    ids = range(4)
    for i, j in combinations(ids, 2):
        k, last = (a for a in ids if a not in (i, j))
        first = parts[i] + parts[j]
        yield f"chain-{i}{j}{k}{last}", (first + parts[k]) + parts[last]
        yield f"chain-{i}{j}{last}{k}", (first + parts[last]) + parts[k]
    for j in (1, 2, 3):
        k, last = (a for a in (1, 2, 3) if a != j)
        yield f"pair-0{j}{k}{last}", (parts[0] + parts[j]) + (parts[k] + parts[last])


def run_case(torch, conv, kernels, data, case, tiles, weights, partials):
    cache = data[f"{case}-cache.bf16"]
    joined = torch.cat(
        (cache, data[f"{case}-input.bf16"], torch.zeros_like(cache)), dim=2
    )
    indices = data[f"{case}-indices.u32"]
    full, local = native_samples(
        torch, kernels, joined, weights[0][0], indices, tiles, partials
    )
    rows = {}

    def observe(name, fp32, bf16=None):
        rows[f"{name}.f32"] = fp32
        rows[f"{name}.rounded"] = fp32.to(torch.bfloat16).float()
        if bf16 is not None:
            rows[f"{name}.bf16"] = bf16.float()

    for i, policy in enumerate(("ff", "fr", "rf", "rr")):
        observe(f"full.native.{policy}", full[i])
    for arm in range(len(SPARSE_TILES)):
        observe(f"local.{arm}.native.f", local[0, arm])
        observe(f"local.{arm}.native.r", local[1, arm])

    joined_f32 = joined.float()
    arms = ("full", *(f"local.{i}" for i in range(len(SPARSE_TILES))), "union")
    for name, (w_bf16, w_f32) in zip(arms, weights, strict=True):
        raw = upstream_sample(torch, conv, joined_f32, w_f32, indices)
        bf16 = upstream_sample(torch, conv, joined, w_bf16, indices)
        observe(f"{name}.upstream", raw, bf16)

    # Freeze the separately observed upstream partials as F32, without an
    # intervening BF16 cast, CPU transfer or upstream convolution invocation.
    parts = tuple(rows[f"local.{i}.upstream.f32"] for i in range(len(SPARSE_TILES)))
    for tree, result in sum_trees(parts):
        observe(f"union.merge.{tree}", result)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path, help="Existing scratch directory")
    parser.add_argument("--admitted", action="store_true")
    args = parser.parse_args()
    if not args.admitted:
        parser.error("Launch through the cache-aware memory guard")
    if sys.byteorder != "little":
        parser.error("The retained tensor fixtures are little endian")
    meta_path = args.output / "k-partials.json"
    raw_path = args.output / "k-partials.f32"
    if not args.output.is_dir() or meta_path.exists() or raw_path.exists():
        parser.error("Expected an existing scratch directory with no previous outputs")

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    if not torch.backends.mps.is_available() or torch.__version__ != "2.8.0":
        raise RuntimeError("Requires the pinned Torch 2.8.0 MPS runtime")
    torch.set_grad_enabled(False)
    manifest_pin, pins = fixture_pins(args.fixture)
    host = host_fixture(torch, args.fixture, pins)
    host["tiles"] = torch.tensor(SPARSE_TILES, dtype=torch.int32)
    host["witness"] = torch.tensor([1.0, 2.0**-24, -1.0], dtype=torch.float32)
    conv, names = pinned_conv3d(torch)
    shader = SHADER.read_text()
    library = torch.mps.compile_shader(shader)
    kernels = (library.local_partials, library.merge_partials)
    mask_weights, arithmetic_witness = library.mask_weights, library.arithmetic_witness

    # The pinned backend wraps host pointers for these non-blocking copies.
    # Keep host ownership until the terminal readout has drained the stream.
    device = {name: value.to("mps", non_blocking=True) for name, value in host.items()}
    tiles, weight = device["tiles"], device["weight.bf16"]
    witness = torch.empty(4, dtype=torch.float32, device="mps")
    arithmetic_witness(device["witness"], witness, threads=1)
    weights = [(weight, weight.float())]
    for arm in range(len(SPARSE_TILES) + 1):
        masked = torch.empty_like(weight)
        mask_weights(
            weight, tiles, masked, arm, threads=weight.numel(), arg_casts={3: "int32"}
        )
        weights.append((masked, masked.float()))

    # Reuse this scratch only after its previous reads have been enqueued on
    # the same MPS stream.
    partials = torch.empty((2, SAMPLES, TILE_COUNT), dtype=torch.float32, device="mps")
    cases = []
    for case in CASES:
        cases.append(
            run_case(torch, conv, kernels, device, case, tiles, weights, partials)
        )
        print(
            f"Enqueued {case}: two FMA orders and six upstream arms per dtype",
            flush=True,
        )
    columns = tuple(cases[0])
    if any(tuple(rows) != columns for rows in cases) or any(
        v.device.type != "mps"
        or v.dtype != torch.float32
        or tuple(v.shape) != (SAMPLES,)
        for rows in cases
        for v in rows.values()
    ):
        raise RuntimeError("Unexpected GPU observations")
    observations = torch.stack(
        [rows[name] for rows in cases for name in columns]
    ).flatten()
    packed = torch.cat((observations, witness)).to("cpu")
    host.clear()
    measured = packed[-4:].tolist()
    if measured != [0.0, 2.0**-24, 0.0, 2.0**-24]:
        raise RuntimeError(f"Reduction order control failed: {measured}")

    payload = ctypes.string_at(packed.data_ptr(), observations.numel() * 4)
    with raw_path.open("xb") as stream:
        stream.write(payload)
    report = {
        "schema": "dcae_k_partials_raw/v1",
        "cases": CASES,
        "columns": columns,
        "samples": SAMPLES,
        "shape": SHAPE,
        "tile_width": TILE_WIDTH,
        "tile_count": TILE_COUNT,
        "sparse_tiles": SPARSE_TILES,
        "singleton_channel_stride": SINGLETON_CHANNEL_STRIDE,
        "prior_manifest_sha256": manifest_pin,
        "fixture_sha256": pins,
        "observations_sha256": hashlib.sha256(payload).hexdigest(),
        "shader_sha256": hashlib.sha256(shader.encode()).hexdigest(),
        "upstream_sha256": UPSTREAM_SHA,
        "upstream_functions": names,
        "torch_version": torch.__version__,
        "arithmetic_witness": measured,
        "terminal_tensor_transfers": 1,
        "explicit_mps_synchronizations": 0,
        "cpu_fallback": False,
    }
    with meta_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(payload)} observation bytes after the sole terminal readout",
        flush=True,
    )


if __name__ == "__main__":
    main()
