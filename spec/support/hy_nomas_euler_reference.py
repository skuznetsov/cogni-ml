#!/usr/bin/env python3
"""Produce a tiny MPS oracle fixture with the pinned Hy-Nomas EulerSolver."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import torch


UPSTREAM_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
SOLVER_RELATIVE_PATH = Path("hymm/pipeline/solvers/euler.py")
SOLVER_SHA256 = "e7996f947b46835f97b66059e0f43019b744ecec5cda380432124f70e169e59a"
SSD_ROOT = Path("/opt/mldata_ssd")


def source_identity() -> tuple[Path, str]:
    root = UPSTREAM_ROOT.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("pinned upstream root must resolve to a checkout directory")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"upstream revision mismatch: {revision}")
    git_root = Path(subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()).resolve(strict=True)
    if git_root != root:
        raise ValueError("pinned upstream path does not resolve to the checkout root")
    solver_path = root / SOLVER_RELATIVE_PATH
    if solver_path.is_symlink() or solver_path.resolve(strict=True) != solver_path or not solver_path.is_file():
        raise ValueError("pinned EulerSolver source is missing or symlinked")
    actual_sha = hashlib.sha256(solver_path.read_bytes()).hexdigest()
    if actual_sha != SOLVER_SHA256:
        raise ValueError(f"EulerSolver SHA-256 mismatch: {actual_sha}")
    return solver_path, actual_sha


def import_euler_solver(solver_path: Path):
    spec = importlib.util.spec_from_file_location("hy_nomas_pinned_euler", solver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned EulerSolver module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EulerSolver


def raw_inputs(count: int) -> tuple[list[float], list[float]]:
    # Non-BF16-exact fractions make both explicit cast boundaries observable.
    initial = [(((i * 29) % 71) - 35) / 37.0 + ((i % 3) - 1) * 0.00073 for i in range(count)]
    velocity = [(((i * 13 + 5) % 67) - 33) / 41.0 + ((i % 4) - 1) * 0.00091 for i in range(count)]
    return initial, velocity


def produce(count: int, solver_path: Path, output: Path) -> dict[str, object]:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("the independent Euler fixture requires an available PyTorch MPS device")
    if not hasattr(torch, "bfloat16"):
        raise RuntimeError("this PyTorch build does not expose bfloat16")

    raw_initial, raw_velocity = raw_inputs(count)
    device = torch.device("mps")
    initial = torch.tensor(raw_initial, dtype=torch.float32, device=device)
    velocity = torch.tensor(raw_velocity, dtype=torch.float32, device=device)
    sample = initial.to(torch.bfloat16)
    initial_boundary = sample.float()
    model_output = velocity.to(torch.bfloat16)

    EulerSolver = import_euler_solver(solver_path)
    scheduler = EulerSolver(num_train_timesteps=1000, shift=5.0, reverse=True, solver="euler")
    scheduler.set_timesteps(8, device=device)
    timesteps = scheduler.timesteps.float()
    if len(timesteps) != 8:
        raise RuntimeError("pinned EulerSolver produced an unexpected timestep count")

    for timestep in scheduler.timesteps:
        sample = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            return_dict=False,
        )[0].to(torch.bfloat16)
    expected_final = sample.float()

    # Export the boundary state, schedule and final state in one terminal MPS
    # transfer. All slicing and JSON conversion happens on the host afterward.
    packed_oracle = torch.cat((initial_boundary.reshape(-1), timesteps.reshape(-1), expected_final.reshape(-1)))
    packed_host = packed_oracle.detach().cpu().tolist()
    initial_boundary_values = packed_host[:count]
    timestep_values = packed_host[count:count + 8]
    expected_final_values = packed_host[count + 8:]

    document = {
        "schema": "cogni_ml.hy_nomas_euler_probe_fixture/v1",
        "upstream_revision": UPSTREAM_REVISION,
        "oracle": {"path": SOLVER_RELATIVE_PATH.as_posix(), "sha256": SOLVER_SHA256},
        "steps": 8,
        "shift": 5.0,
        "num_train_timesteps": 1000,
        "target_dtype": "bfloat16",
        "initial_latent_f32": raw_initial,
        "expected_initial_latent_bf16_f32": initial_boundary_values,
        "velocity_f32": raw_velocity,
        "expected_timesteps_f32": timestep_values,
        "expected_final_latent_f32": expected_final_values,
    }
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

    canonical_output = output.resolve(strict=False)
    canonical_root = SSD_ROOT.resolve(strict=True)
    if (
        not output.is_absolute()
        or canonical_output != output
        or canonical_root not in output.parents
        or not output.parent.is_dir()
        or output.parent.resolve(strict=True) != output.parent
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("--output must be a new canonical file under /opt/mldata_ssd in an existing directory")
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        import os

        os.fsync(stream.fileno())

    return {
        "schema": "cogni_ml.hy_nomas_euler_reference_result/v1",
        "fixture": str(output),
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_solver_sha256": SOLVER_SHA256,
        "device": "mps",
        "element_count": count,
        "steps": 8,
        "shift": 5.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--elements", type=int, choices=(32, 256), default=32)
    args = parser.parse_args()
    solver_path, _ = source_identity()
    print(json.dumps(produce(args.elements, solver_path, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
