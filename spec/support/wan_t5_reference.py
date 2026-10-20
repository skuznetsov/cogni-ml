#!/usr/bin/env python3
"""Pinned Wan UMT5 encoder oracle; neural operators run on MPS in F32.

Copyright 2024-2025 The Alibaba Wan Team Authors. Apache-2.0.
The native BF16-storage mode is compared to the same F32 arithmetic policy as
the admitted Wan block. This does not certify upstream BF16 arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from safetensors.torch import load_file, save_file

REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
SOURCE_HASH = "8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--long", action="store_true")
    p.add_argument("--production-width", action="store_true")
    p.add_argument("--attention-diagnostics", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--full-context", action="store_true",
                   help="Qualify the actual 24-layer, 512-token checkpoint with a compact trace sidecar")
    args = p.parse_args()
    if args.full_context and not args.checkpoint:
        p.error("--full-context requires the original BF16 --checkpoint")
    source = args.upstream / "wan_modules_t5.py"
    text = source.read_text()
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_HASH:
        raise RuntimeError("Pinned UMT5 source changed")
    # Import only the real encoder definitions. The unused wrapper's default
    # CUDA-device expression would initialize CUDA while importing on an MPS host.
    text = text.split("class T5EncoderModel:")[0]
    text = text.replace("from .tokenizers import HuggingfaceTokenizer", "# unused tokenizer wrapper")
    if args.attention_diagnostics:
        source_expression = """        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)"""
        diagnostic_expression = """        attn = torch.einsum('binc,bjnc->bnij', q, k)
        if getattr(self, "_wan_capture_attention_diagnostics", False):
            self._wan_attention_logits = attn.detach()
            self._wan_attention_bias = attn_bias.detach()
        attn = attn + attn_bias
        if getattr(self, "_wan_capture_attention_diagnostics", False):
            self._wan_attention_scores = attn.detach()
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        if getattr(self, "_wan_capture_attention_diagnostics", False):
            self._wan_attention_probs = attn.detach()"""
        if source_expression not in text:
            raise RuntimeError("Pinned T5 attention expression changed; diagnostics cannot be injected")
        text = text.replace(source_expression, diagnostic_expression, 1)
    namespace = {"__name__": "wan_t5_pinned_reference"}
    exec(compile(text, str(source), "exec"), namespace)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU neural fallback is forbidden")
    production = args.production_width or bool(args.checkpoint)
    cfg = dict(vocab=256384 if args.checkpoint else 96,
               dim=4096 if production else 32,
               ffn_dim=10240 if production else 64,
               heads=64 if production else 4,
               layers=1 if production else 2, buckets=32, max_distance=128,
               batches=2, tokens=4 if production else (132 if args.long else 9), eps=1e-6)
    if args.full_context:
        cfg.update(layers=24, batches=1, tokens=512)
    torch.manual_seed(793)
    model_cls = namespace["T5Encoder"]
    kwargs = dict(vocab=cfg["vocab"], dim=cfg["dim"], dim_attn=cfg["dim"],
                  dim_ffn=cfg["ffn_dim"], num_heads=cfg["heads"], num_layers=cfg["layers"],
                  num_buckets=cfg["buckets"], shared_pos=False, dropout=0.0)
    if args.checkpoint:
        with torch.device("meta"):
            model = model_cls(**kwargs)
        with __import__("safetensors").safe_open(str(args.checkpoint), framework="pt", device="cpu") as f:
            state = {name: f.get_tensor(name) for name in model.state_dict()}
        dtype = {t.dtype for t in state.values()}
        if dtype != {torch.bfloat16}:
            raise RuntimeError(f"Expected original BF16 UMT5 tensors, got {dtype}")
        # Keep the checkpoint in its original BF16 representation on CPU.
        # The model is uploaded in BF16 and only then expanded to F32 on MPS;
        # constructing a CPU F32 mirror needlessly doubles checkpoint RSS.
        model.load_state_dict(state, assign=True)
    else:
        model = model_cls(**kwargs)
        state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    model = model.eval().requires_grad_(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weight_path = args.checkpoint.resolve() if args.full_context else args.output.with_suffix(".safetensors")
    if not args.full_context:
        save_file(state, str(weight_path), metadata={"upstream_revision": REVISION, "arithmetic": "F32"})
    if args.checkpoint:
        # The model now owns the same BF16 CPU tensors through assign=True.
        # Drop the temporary mapping before the MPS upload, then perform the
        # required BF16->F32 conversion on-device rather than on CPU.
        del state
        model = model.to(device="mps")
        model = model.to(dtype=torch.float32)
    else:
        model = model.to(device="mps", dtype=torch.float32)
    ids = (torch.arange(cfg["batches"] * cfg["tokens"]).reshape(cfg["batches"], cfg["tokens"]) * 7 + 3) % min(cfg["vocab"], 96)
    mask = torch.ones_like(ids)
    mask[0, -2:] = 0
    if cfg["tokens"] > 4 and cfg["batches"] > 1:
        mask[1, 2] = 0  # Exercise a hole, not only right padding.
    captured = {}
    hooks = []
    def capture(module, name, pre=False):
        def hook(_m, inputs, output=None):
            value = inputs[0] if pre else output
            captured[name] = value.detach().clone()
        hooks.append(module.register_forward_pre_hook(hook) if pre else module.register_forward_hook(hook))

    def capture_norm(module, name, capture_scalars=False):
        if capture_scalars:
            def pre_hook(norm, inputs):
                value = inputs[0]
                # Repeat the pinned T5LayerNorm scalar path on the exact tensor
                # presented to forward. Preserve pow(2) and mean as distinct
                # MPS operations so the square diagnostic can be compared
                # independently from the reduction tree.
                squared = value.float().pow(2)
                mean_square = squared.mean(dim=-1, keepdim=True)
                inverse_rms = torch.rsqrt(mean_square + norm.eps)
                captured[name + ".input"] = value.detach().clone()
                captured[name + ".square"] = squared.detach().clone()
                captured[name + ".mean_square"] = mean_square.detach().clone()
                captured[name + ".inverse_rms"] = inverse_rms.detach().clone()
            hooks.append(module.register_forward_pre_hook(pre_hook))
        capture(module, name)

    capture(model.token_embedding, "embedding")
    for i, block in enumerate(model.blocks):
        prefix = f"blocks.{i}."
        if args.full_context and i > 0:
            capture(block, prefix + "residual")
            continue
        capture_norm(block.norm1, prefix + "norm1", i == 0)
        capture_norm(block.norm2, prefix + "norm2", i == 0)
        for name in ("q", "k", "v"):
            capture(getattr(block.attn, name), prefix + name)
        if args.attention_diagnostics and i == 0:
            block.attn._wan_capture_attention_diagnostics = True
        capture(block.attn.o, prefix + "attention", pre=True)
        capture(block.attn.o, prefix + "attn_projection")
        capture(block.ffn.gate, prefix + "gate")
        capture(block.ffn.gate[0], prefix + "gate_pre")
        capture(block.ffn.fc1, prefix + "fc1")
        capture(block.ffn.fc2, prefix + "product", pre=True)
        capture(block.ffn, prefix + "ffn")
        def block_hook(_module, inputs, output, prefix=prefix):
            captured[prefix + "attn_residual"] = inputs[0] + captured[prefix + "attn_projection"]
            captured[prefix + "residual"] = output.detach().clone()
        hooks.append(block.register_forward_hook(block_hook))
    with torch.inference_mode():
        output = model(ids.to("mps"), mask.to("mps"))
        captured["output"] = output
        captured["masked_context"] = output * mask.to("mps").unsqueeze(-1)
    if args.attention_diagnostics:
        first_attention = model.blocks[0].attn
        for name, attribute in (("attention_logits", "_wan_attention_logits"),
                                ("attention_bias", "_wan_attention_bias"),
                                ("attention_scores", "_wan_attention_scores"),
                                ("attention_probs", "_wan_attention_probs")):
            value = getattr(first_attention, attribute, None)
            if value is None:
                raise RuntimeError(f"Pinned attention diagnostic {attribute} was not captured")
            captured[f"blocks.0.{name}"] = value
    torch.mps.synchronize()  # Terminal diagnostic boundary only.
    def flat(t):
        return t.detach().cpu().float().contiguous().reshape(-1).tolist()
    expected_path = None
    expected_sha256 = None
    if args.full_context:
        expected_path = args.output.with_suffix(".expected.safetensors")
        save_file({name: value.detach().cpu().float().contiguous().clone()
                   for name, value in captured.items()}, str(expected_path),
                  metadata={"upstream_revision": REVISION, "source_sha256": SOURCE_HASH,
                            "arithmetic": "F32", "device": "mps"})
        with expected_path.open("rb") as stream:
            expected_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    with weight_path.open("rb") as stream:
        weight_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(revision=REVISION, source_sha256=SOURCE_HASH, config=cfg,
                  device="mps", arithmetic="F32", torch_version=torch.__version__,
                  weights=str(weight_path), storage="BF16" if args.checkpoint else "F32",
                  weight_sha256=weight_sha256,
                  ids=ids.reshape(-1).tolist(), mask=mask.reshape(-1).tolist(),
                  expected=None if args.full_context else {name: flat(value) for name, value in captured.items()},
                  expected_safetensors=str(expected_path) if expected_path else None,
                  expected_safetensors_sha256=expected_sha256,
                  trace_shapes={name: list(value.shape) for name, value in captured.items()})
    args.output.write_text(json.dumps(report, allow_nan=False))
    print(json.dumps({"fixture": str(args.output), "arrays": len(captured), "config": cfg}))


if __name__ == "__main__":
    main()
