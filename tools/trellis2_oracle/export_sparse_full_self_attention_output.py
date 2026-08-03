#!/usr/bin/env python3
"""Export pinned TRELLIS.2 self/full attention composition boundaries.

The oracle executes the real SparseMultiHeadAttention.forward orchestration on
tiny synthetic CPU/F32 data. TRELLIS.2 has no CPU sparse-attention backend, so
the backend call is replaced with the pinned dense naive formula evaluated per
canonical sparse batch. One pass replaces the final output projection with
identity to retain the admitted pre-output boundary; a second pass executes a
frozen asymmetric biased to_out projection for the next bounded seam.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


TRELLIS_PIN = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
PYTHON_PIN = "3.11.9"
TORCH_PIN = "2.9.0"
NUMPY_PIN = "2.1.3"
SCHEMA = "cogni-ml/trellis2/sparse-full-self-attention-output-oracle/v1"

SOURCE_PINS = {
    "sparse_full_attention": (
        "trellis2/modules/sparse/attention/full_attn.py",
        "bee0c32089f060c8136292f41a2cc7a952a8679a8b6d113d14c772f2c681e520",
    ),
    "sparse_attention_modules": (
        "trellis2/modules/sparse/attention/modules.py",
        "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf",
    ),
    "sparse_rope": (
        "trellis2/modules/sparse/attention/rope.py",
        "0525164901c3f1c885e961b747c856b654a4d6840882f34827677fb017dcec00",
    ),
    "sparse_basic": (
        "trellis2/modules/sparse/basic.py",
        "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
    ),
    "sparse_config": (
        "trellis2/modules/sparse/config.py",
        "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b",
    ),
    "dense_full_attention": (
        "trellis2/modules/attention/full_attn.py",
        "64c43354780dcbc3dcf7612ac5e53d6e21c2081234ea63cd329a77f4185dadfc",
    ),
    "structured_flow": (
        "trellis2/models/structured_latent_flow.py",
        "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021",
    ),
}

BATCH_SIZE = 3
SPATIAL_SHAPE = [3, 3, 3]
SEQUENCE_LENGTHS = [2, 0, 2]
NUM_HEADS = 2
HEAD_DIM = 8
CHANNELS = NUM_HEADS * HEAD_DIM
ROPE_FREQ = [1.0, 10000.0]
COORDINATES = [
    [0, 0, 1, 2],
    [0, 2, 0, 1],
    [2, 1, 2, 0],
    [2, 2, 1, 1],
]


def patterned_f32(size: int, modulus: int, center: int, divisor: float) -> np.ndarray:
    values = np.arange(size, dtype=np.int64)
    return ((values % modulus) - center).astype(np.float32) / np.float32(divisor)


INPUT_FEATURES = patterned_f32(len(COORDINATES) * CHANNELS, 29, 14, 7.0).reshape(
    len(COORDINATES), CHANNELS
)
QKV_WEIGHT = patterned_f32(3 * CHANNELS * CHANNELS, 37, 18, 31.0).reshape(
    3 * CHANNELS, CHANNELS
)
QKV_BIAS = patterned_f32(3 * CHANNELS, 17, 8, 19.0)
Q_GAMMA = patterned_f32(NUM_HEADS * HEAD_DIM, 13, 5, 7.0).reshape(
    NUM_HEADS, HEAD_DIM
) + np.float32(1.0)
K_GAMMA = patterned_f32(NUM_HEADS * HEAD_DIM, 11, 4, 9.0).reshape(
    NUM_HEADS, HEAD_DIM
) + np.float32(0.75)
OUT_WEIGHT = patterned_f32(CHANNELS * CHANNELS, 31, 15, 23.0).reshape(
    CHANNELS, CHANNELS
)
OUT_BIAS = patterned_f32(CHANNELS, 13, 6, 17.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def little_endian_sha(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(
        values.astype(dtype, copy=False).tobytes(order="C")
    ).hexdigest()


def require_pins(trellis_root: Path) -> dict[str, Path]:
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    expected = {
        "python": PYTHON_PIN,
        "torch": TORCH_PIN,
        "numpy": NUMPY_PIN,
    }
    if versions != expected:
        raise RuntimeError(f"pinned package drift: expected {expected}, got {versions}")

    sources = {}
    for name, (relative_path, expected_hash) in SOURCE_PINS.items():
        source = trellis_root / relative_path
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned {name} source drift: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        sources[name] = source

    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(trellis_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to verify pinned TRELLIS.2 commit") from error
    if actual_commit != TRELLIS_PIN:
        raise RuntimeError(
            f"pinned TRELLIS.2 commit drift: expected {TRELLIS_PIN}, "
            f"got {actual_commit}"
        )
    return sources


def load_upstream(trellis_root: Path):
    sys.path.insert(0, str(trellis_root))
    sparse_config = importlib.import_module("trellis2.modules.sparse.config")
    sparse_config.set_conv_backend("none")
    if sparse_config.CONV != "none":
        raise AssertionError("failed to disable the sparse convolution backend")
    sparse_module = importlib.import_module("trellis2.modules.sparse")
    attention_module = importlib.import_module(
        "trellis2.modules.sparse.attention.modules"
    )
    dense_attention = importlib.import_module("trellis2.modules.attention.full_attn")
    return sparse_module, attention_module, dense_attention._naive_sdpa


def explicit_stable_attention(qkv: torch.Tensor) -> torch.Tensor:
    q, k, v = qkv.unbind(dim=1)
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    weights = shifted.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return torch.matmul(weights, v).permute(1, 0, 2)


def build_fixture(trellis_root: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    sources = require_pins(trellis_root)
    sparse_module, attention_module, dense_naive = load_upstream(trellis_root)

    bias_independence = attention_module.SparseMultiHeadAttention(
        channels=CHANNELS,
        num_heads=NUM_HEADS,
        type="self",
        attn_mode="full",
        qkv_bias=False,
    ).cpu().eval()
    if bias_independence.to_qkv.bias is not None:
        raise AssertionError("qkv_bias=False must disable only the QKV bias")
    if bias_independence.to_out.bias is None:
        raise AssertionError("to_out bias must remain independent from qkv_bias")

    coordinate_array = np.asarray(COORDINATES, dtype=np.int32)
    coordinates = torch.from_numpy(coordinate_array.copy())
    input_tensor = sparse_module.SparseTensor(
        torch.from_numpy(INPUT_FEATURES.copy()),
        coordinates,
        shape=torch.Size([BATCH_SIZE, CHANNELS]),
    )
    actual_lengths = [item.stop - item.start for item in input_tensor.layout]
    if actual_lengths != SEQUENCE_LENGTHS:
        raise AssertionError(
            f"pinned sparse layout drift: expected {SEQUENCE_LENGTHS}, got {actual_lengths}"
        )

    attention = attention_module.SparseMultiHeadAttention(
        channels=CHANNELS,
        num_heads=NUM_HEADS,
        type="self",
        attn_mode="full",
        qkv_bias=True,
        use_rope=True,
        rope_freq=tuple(ROPE_FREQ),
        qk_rms_norm=True,
    ).cpu().eval()
    attention.requires_grad_(False)
    if any(parameter.requires_grad for parameter in attention.parameters()):
        raise AssertionError("oracle attention parameters must be frozen")
    with torch.no_grad():
        attention.to_qkv.weight.copy_(torch.from_numpy(QKV_WEIGHT.copy()))
        attention.to_qkv.bias.copy_(torch.from_numpy(QKV_BIAS.copy()))
        attention.q_rms_norm.gamma.copy_(torch.from_numpy(Q_GAMMA.copy()))
        attention.k_rms_norm.gamma.copy_(torch.from_numpy(K_GAMMA.copy()))
        attention.to_out.weight.copy_(torch.from_numpy(OUT_WEIGHT.copy()))
        attention.to_out.bias.copy_(torch.from_numpy(OUT_BIAS.copy()))

    # Record named stages by invoking the exact upstream helpers in forward order.
    with torch.inference_mode():
        projected = attention._linear(attention.to_qkv, input_tensor)
        projected = attention._fused_pre(projected, num_fused=3)
        q, k, v = projected.unbind(dim=-3)
        q = attention.q_rms_norm(q)
        k = attention.k_rms_norm(k)
        normalized = projected.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
        q, k = attention.rope(q, k)
        roped = projected.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))

    captured: dict[str, list[torch.Tensor]] = {"qkv": [], "attention": []}

    def injected_cpu_full_attention(qkv):
        captured["qkv"].append(qkv.feats.detach().clone())
        output = torch.empty(
            (qkv.feats.shape[0], NUM_HEADS, HEAD_DIM), dtype=torch.float32
        )
        for batch_slice in qkv.layout:
            batch_qkv = qkv.feats[batch_slice]
            if batch_qkv.shape[0] == 0:
                continue
            manual = explicit_stable_attention(batch_qkv)
            q, k, v = batch_qkv.unbind(dim=1)
            dense = dense_naive(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0]
            torch.testing.assert_close(manual, dense, rtol=1e-6, atol=1e-6)
            output[batch_slice] = dense
        captured["attention"].append(output.detach().clone())
        return qkv.replace(output)

    input_before = input_tensor.feats.detach().clone()
    original_backend = attention_module.sparse_scaled_dot_product_attention
    attention_module.sparse_scaled_dot_product_attention = injected_cpu_full_attention
    output_projection = attention.to_out
    attention.to_out = nn.Identity()
    try:
        with torch.inference_mode():
            pre_output = attention(input_tensor)
            attention.to_out = output_projection
            final_output = attention(input_tensor)
    finally:
        attention.to_out = output_projection
        attention_module.sparse_scaled_dot_product_attention = original_backend

    if len(captured["qkv"]) != 2 or len(captured["attention"]) != 2:
        raise AssertionError("expected exactly two upstream forward captures")
    if not all(torch.equal(value, roped.feats) for value in captured["qkv"]):
        raise AssertionError("real forward QKV differs from named upstream stages")
    if not torch.equal(
        pre_output.feats, captured["attention"][0].reshape(-1, CHANNELS)
    ):
        raise AssertionError("identity pre-output boundary changed attention output")
    if not torch.equal(captured["attention"][1], captured["attention"][0]):
        raise AssertionError("repeated upstream attention context drift")
    expected_final = torch.nn.functional.linear(
        pre_output.feats,
        torch.from_numpy(OUT_WEIGHT.copy()),
        torch.from_numpy(OUT_BIAS.copy()),
    )
    if not torch.equal(final_output.feats, expected_final):
        raise AssertionError("real forward to_out differs from frozen biased Linear")
    if torch.equal(final_output.feats, pre_output.feats):
        raise AssertionError("final output projection must be non-identity")
    if not torch.equal(input_tensor.feats, input_before):
        raise AssertionError("pinned upstream mutated self-attention input")
    if pre_output.coords is not input_tensor.coords:
        raise AssertionError("pinned upstream pre-output coordinate-object drift")
    if final_output.coords is not input_tensor.coords:
        raise AssertionError("pinned upstream coordinate-object drift")
    if not torch.equal(normalized.feats[:, 2], projected.feats[:, 2]):
        raise AssertionError("Q/K normalization changed V")
    if not torch.equal(roped.feats[:, 2], projected.feats[:, 2]):
        raise AssertionError("RoPE changed V")
    if not torch.equal(roped.feats[..., 6:], normalized.feats[..., 6:]):
        raise AssertionError("RoPE changed the D=8 identity tail")

    arrays = {
        "projected_qkv": projected.feats.detach().cpu().numpy(),
        "normalized_qkv": normalized.feats.detach().cpu().numpy(),
        "roped_qkv": roped.feats.detach().cpu().numpy(),
        "attention_output": pre_output.feats.detach().cpu().numpy(),
        "final_output": final_output.feats.detach().cpu().numpy(),
    }
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise AssertionError("composition reference produced a non-finite value")

    return {
        "schema": SCHEMA,
        "provenance": {
            "repository": "microsoft/TRELLIS.2",
            "commit": TRELLIS_PIN,
            "sources": {
                name: {"path": SOURCE_PINS[name][0], "sha256": sha256(path)}
                for name, path in sources.items()
            },
            "generator": (
                "tools/trellis2_oracle/"
                "export_sparse_full_self_attention_output.py"
            ),
            "python_version": PYTHON_PIN,
            "torch_version": TORCH_PIN,
            "numpy_version": NUMPY_PIN,
            "network": "none",
            "device": "cpu",
            "weights": "synthetic",
            "sparse_backend": "injected dense block-diagonal CPU reference",
            "upstream_forward_executed": True,
            "upstream_sparse_backend_executed": False,
            "upstream_dense_reference_executed": True,
            "output_projection": "identity substitution",
            "final_output_projection": "frozen asymmetric biased Linear",
        },
        "contract": {
            "consumer": "SparseMultiHeadAttention.forward self/full path",
            "order": [
                "to_qkv",
                "packed [N, 3, H, D] view",
                "Q/K RMS normalization",
                "3D RoPE",
                "block-diagonal full attention",
            ],
            "final_order": [
                "to_qkv",
                "packed [N, 3, H, D] view",
                "Q/K RMS normalization",
                "3D RoPE",
                "block-diagonal full attention",
                "flatten heads to [N, C]",
                "biased to_out Linear(C, C)",
            ],
            "boundary": "pre-output projection",
            "final_boundary": "post-output projection",
            "qk_rms_norm": True,
            "use_rope": True,
            "not_sparse_backend_parity": True,
            "sequence_lengths": SEQUENCE_LENGTHS,
            "coordinate_object_reused": True,
            "final_coordinate_object_reused": True,
            "input_unchanged": True,
            "output_projection_non_identity": True,
            "qkv_bias_independent_from_to_out_bias": True,
            "value_component_unchanged_before_attention": True,
            "identity_tail_unchanged_before_attention": True,
            "local_mode": "bounded graphless frozen-parameter CPU/F32 inference",
        },
        "input": {
            "batch_size": BATCH_SIZE,
            "spatial_shape": SPATIAL_SHAPE,
            "coordinates": COORDINATES,
            "coordinates_i32le_sha256": little_endian_sha(coordinate_array, "<i4"),
            "features": INPUT_FEATURES.tolist(),
            "features_f32le_sha256": little_endian_sha(INPUT_FEATURES, "<f4"),
        },
        "attention": {
            "channels": CHANNELS,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_bias": True,
            "qkv_weight": QKV_WEIGHT.tolist(),
            "qkv_weight_f32le_sha256": little_endian_sha(QKV_WEIGHT, "<f4"),
            "qkv_bias_values": QKV_BIAS.tolist(),
            "qkv_bias_f32le_sha256": little_endian_sha(QKV_BIAS, "<f4"),
            "q_gamma": Q_GAMMA.tolist(),
            "q_gamma_f32le_sha256": little_endian_sha(Q_GAMMA, "<f4"),
            "k_gamma": K_GAMMA.tolist(),
            "k_gamma_f32le_sha256": little_endian_sha(K_GAMMA, "<f4"),
            "to_out_bias": True,
            "to_out_weight": OUT_WEIGHT.tolist(),
            "to_out_weight_f32le_sha256": little_endian_sha(OUT_WEIGHT, "<f4"),
            "to_out_bias_values": OUT_BIAS.tolist(),
            "to_out_bias_f32le_sha256": little_endian_sha(OUT_BIAS, "<f4"),
            "rope_freq": ROPE_FREQ,
            "scale": 1.0 / math.sqrt(HEAD_DIM),
        },
        "stages": {
            name: {
                "features": array.tolist(),
                "features_f32le_sha256": little_endian_sha(array, "<f4"),
            }
            for name, array in arrays.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--trellis-root", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.trellis_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(fixture, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
