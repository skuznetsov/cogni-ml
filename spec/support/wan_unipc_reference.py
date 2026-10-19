#!/usr/bin/env python3
"""Execute pinned Wan Flow-UniPC on MPS, exporting terminal diagnostics only.

The missing Diffusers package supplies configuration plumbing, replaced here by
a minimal constructor decorator. Solver methods are the actual pinned source.
Small schedule matrices are constructed/solved on CPU like setup coefficients;
model outputs, samples, correctors, predictors and history remain on MPS.
CPU sigma views are cloned before MPS consumption: PyTorch 2.8.0 MPS otherwise
reuses the first scalar at the shared storage address, ignoring its view offset.
"""
from __future__ import annotations
import argparse
import ast
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import torch

REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
SOURCE_SHA256 = "0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9"


def register_to_config(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        bound = inspect.signature(fn).bind(self, *args, **kwargs)
        bound.apply_defaults()
        self.config = SimpleNamespace(**{k: v for k, v in bound.arguments.items() if k != "self"})
        return fn(self, *args, **kwargs)
    return wrapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()
    source = args.upstream / "wan_utils_fm_solvers_unipc.py"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Pinned UniPC source changed")
    module = ast.parse(raw)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "FlowUniPCMultistepScheduler")
    cls.bases = []
    cls.body = [n for n in cls.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_compatibles" for t in n.targets))]
    text = ast.unparse(ast.Module(body=[cls], type_ignores=[]))
    # R and b contain schedule constants only. Keep their tiny linear system on
    # CPU, avoiding unsupported MPS solve and any GPU -> CPU tensor readback.
    for expression in ("torch.tensor(rks, device=device)", "torch.tensor(b, device=device)"):
        text = text.replace(expression, expression.replace("device=device", "device='cpu'"))
    text = text.replace("self.sigmas[self.step_index]", "self.sigmas[self.step_index].clone()")
    namespace = dict(torch=torch, np=np, math=__import__("math"), register_to_config=register_to_config,
                     List=list, Optional=__import__("typing").Optional, Union=__import__("typing").Union,
                     Tuple=tuple, SchedulerMixin=object, SchedulerOutput=lambda **kw: SimpleNamespace(**kw))
    exec(compile(text, str(source), "exec"), namespace)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    scheduler = namespace["FlowUniPCMultistepScheduler"]()
    scheduler.set_timesteps(args.steps, device="cpu", shift=3)
    count = 257
    torch.manual_seed(1917)
    initial = torch.randn(1, 1, 1, 1, count, device="mps")
    conditional = [torch.randn_like(initial) for _ in range(args.steps)]
    unconditional = [torch.randn_like(initial) for _ in range(args.steps)]
    captured = []
    sample = initial
    with torch.inference_mode():
        for index, timestep in enumerate(scheduler.timesteps):
            model = unconditional[index] + 4.5 * (conditional[index] - unconditional[index])
            sample = scheduler.step(model, timestep, sample).prev_sample
            captured.append(dict(guidance=model.clone(), current=scheduler.model_outputs[-1].clone(),
                                 corrected=scheduler.last_sample.clone(), output=sample.clone()))
    torch.mps.synchronize()
    def flat(t):
        return t.detach().cpu().contiguous().flatten().tolist()
    report = dict(revision=REVISION, source_sha256=SOURCE_SHA256, torch_version=torch.__version__,
                  device="mps", scalar_setup="CPU F32 schedule matrices; no activation fallback",
                  adapters=["CPU-only schedule matrix solve", "independent CPU sigma scalar storage for MPS"],
                  steps=args.steps, count=count, sigmas=scheduler.sigmas.tolist(),
                  timesteps=scheduler.timesteps.tolist(), initial=flat(initial),
                  conditional=[flat(t) for t in conditional], unconditional=[flat(t) for t in unconditional],
                  expected=[{k: flat(v) for k, v in step.items()} for step in captured])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False))
    print(json.dumps(dict(fixture=str(args.output), steps=args.steps, arrays=4 * args.steps)))


if __name__ == "__main__":
    main()
