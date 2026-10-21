#!/usr/bin/env python3
"""Export a bounded audio K/V oracle from the pinned original Wan modules.

Only one injector's K/V projections and K RMS normalization run. The original
BF16 checkpoint tensors are converted to F32 on MPS, matching the native MPS
projection policy. CPU work is limited to source/fixture I/O and final export;
this script is not a generation pipeline or an inference timing benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "spec" / "support"))
import wan_s2v_reference as upstream_reference
import torch
from safetensors import safe_open

DIM, HEADS, FRAMES, TOKENS = 5120, 40, 21, 5
ROWS = FRAMES * TOKENS
WEIGHTS_SHA256 = "c172519063c1c34fb36f7f930429ca7db655a0e335524d7487525db1df5ad220"
INPUT_MANIFEST_SHA256 = "fb7943d637bc8a6853707cd2ed284595e6744c4368083c36993fd1dd26d78da9"
SELECTED = {
    "k.weight": [DIM, DIM],
    "k.bias": [DIM],
    "v.weight": [DIM, DIM],
    "v.bias": [DIM],
    "norm_k.weight": [DIM],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=upstream_reference.DEFAULT_UPSTREAM)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.upstream = args.upstream.resolve()
    args.weights = args.weights.resolve()
    args.input_manifest = args.input_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise RuntimeError("Oracle output directory must be new")
    upstream_reference._verify_sources(args.upstream)
    if sha256_file(args.weights) != WEIGHTS_SHA256:
        raise RuntimeError("Pinned BF16 checkpoint subset SHA-256 mismatch")
    if sha256_file(args.input_manifest) != INPUT_MANIFEST_SHA256:
        raise RuntimeError("Pinned input fixture SHA-256 mismatch")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU fallback is forbidden")
    if os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] != "0":
        raise RuntimeError("MPS CPU fallback must be disabled")
    _, audio_module, _, _ = upstream_reference._install_import_shims(args.upstream)
    # Meta construction avoids allocating the unused Q/O projections. Their
    # forward methods are never invoked, nor is a complete model instantiated.
    with torch.device("meta"):
        injector = audio_module.AudioCrossAttention(dim=DIM, num_heads=HEADS, qk_norm=True)
    bindings = []
    with safe_open(str(args.weights), framework="pt", device="cpu") as checkpoint:
        for target, shape in SELECTED.items():
            source = f"audio_injector.injector.0.{target}"
            view = checkpoint.get_slice(source)
            if view.get_shape() != shape or view.get_dtype() != "BF16":
                raise RuntimeError(f"Unexpected checkpoint tensor: {source}")
            value = checkpoint.get_tensor(source).to("mps").to(torch.float32)
            module_name, parameter_name = target.rsplit(".", 1)
            setattr(getattr(injector, module_name), parameter_name,
                    torch.nn.Parameter(value, requires_grad=False))
            bindings.append({"source": source, "shape": shape, "dtype": "BF16"})

    fixture = json.loads(args.input_manifest.read_text())
    original = fixture["inputs"]["local_audio"]
    if len(original) != 2 * TOKENS * DIM:
        raise RuntimeError("Original local-audio geometry changed")
    with torch.inference_mode():
        base = torch.tensor(original[:TOKENS * DIM], dtype=torch.float32,
                            device="mps").view(1, TOKENS, DIM)
        # Reuse the pinned real-width fixture, with GPU-generated row offsets
        # to exercise all 105 rows rather than 21 identical copies.
        offsets = torch.sin(torch.arange(ROWS, device="mps", dtype=torch.float32) * 0.37) / 64
        local_audio = base.repeat(1, FRAMES, 1) + offsets.view(1, ROWS, 1)
        raw_k = injector.k(local_audio)
        normalized_k = injector.norm_k(raw_k)
        v = injector.v(local_audio)
        results = {"local_audio": local_audio, "raw_k": raw_k,
                   "normalized_k": normalized_k, "v": v}
        for name, result in results.items():
            if result.shape != (1, ROWS, DIM) or result.dtype != torch.float32:
                raise RuntimeError(f"Unexpected oracle result geometry/dtype: {name}")
            if not bool(torch.isfinite(result).all().item()):
                raise RuntimeError(f"Nonfinite upstream result: {name}")
        torch.mps.synchronize()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    files, artifacts = {}, {}
    for name, result in results.items():
        path = args.output_dir / f"{name}.f32"
        # Export is the explicit fixture boundary; no activation transfer is
        # added to the native model path or either timed benchmark arm.
        payload = result.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
        with path.open("xb") as output:
            output.write(payload)
        files[name] = str(path)
        artifacts[name] = {"bytes": len(payload), "sha256": sha256_file(path)}
    manifest = {
        "schema": "wan_s2v_audio_kv_reference_v1",
        "rows": ROWS, "dim": DIM, "heads": HEADS,
        "batches": 1, "frames": FRAMES, "tokens_per_frame": TOKENS,
        "dtype": "F32", "weight_dtype": "BF16", "eps": injector.norm_k.eps,
        "files": files, "artifacts": artifacts,
        "weights_path": str(args.weights), "weights_sha256": WEIGHTS_SHA256,
        "weights_bytes": args.weights.stat().st_size,
        "weight_bindings": bindings,
        "input_manifest": str(args.input_manifest), "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "input_derivation": "first five original rows repeated 21 times plus sin(row*0.37)/64 on MPS",
        "upstream_revision": upstream_reference.REVISION,
        "upstream_sources": {name: {"path": str(args.upstream / name), "sha256": digest}
                             for name, digest in upstream_reference.SOURCE_SHA256.items()},
        "checkpoint_repo": upstream_reference.CHECKPOINT_REPO,
        "checkpoint_revision": upstream_reference.CHECKPOINT_REVISION,
        "device": "mps", "cpu_fallback": False, "torch_version": torch.__version__,
        "generator": {"path": str(Path(__file__).resolve()),
                      "sha256": sha256_file(Path(__file__).resolve())},
        "tolerance": {"absolute": 2e-5, "relative": 2e-5, "require_finite": True},
        "scope": "original injector 0 K/raw V and post-RMS K, not full audio injection or generation",
    }
    path = args.output_dir / "reference.json"
    with path.open("x") as output:
        json.dump(manifest, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"event": "audio_kv_reference_exported", "manifest": str(path),
                      "manifest_sha256": sha256_file(path), "rows": ROWS, "dim": DIM,
                      "device": "mps", "cpu_fallback": False}))


if __name__ == "__main__":
    main()
