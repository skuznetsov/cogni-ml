"""Generate a static Qwen-Image 2.1 schedule fixture with pinned Diffusers.

Example:
  python spec/support/qwen_image21_flow_match_reference.py \
    --config /path/to/pinned/scheduler_config.json

This is a fixture-generation tool only; specs read the checked-in JSON and do
not import Diffusers or access the network.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
from importlib import metadata
from pathlib import Path

import numpy as np

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift


MODEL_REPO = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
DIFFUSERS_COMMIT = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
SCHEDULER_CONFIG_SHA256 = "5895f3a167c14a967fe9ac70c64924ae5acc79799e0679fd12907e594a713cd1"
SCHEDULER_SOURCE_SHA256 = "1af27be5b2f92b7d139d3c50239be5ce3eafc4a7eddf59c2d30689f8ec31e93d"
PIPELINE_SOURCE_SHA256 = "6985b2f1f25e8dd09ef85c867ebcad184b981f10c0b4f80757ca9cc135b8fa5e"
CASES = ((4, 256), (20, 2304), (24, 2304), (40, 2304))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(obj: object) -> Path:
    path = inspect.getsourcefile(obj)
    if path is None:
        raise RuntimeError(f"cannot locate source for {obj!r}")
    return Path(path)


def make_case(config: dict[str, object], steps: int, image_seq_len: int) -> dict[str, object]:
    mu = calculate_shift(
        image_seq_len,
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
    pipeline_sigmas = np.linspace(1.0, 1.0 / steps, steps)
    scheduler.set_timesteps(sigmas=pipeline_sigmas, mu=mu, device="cpu")
    return {
        "steps": steps,
        "image_seq_len": image_seq_len,
        "mu": mu,
        "sigmas": scheduler.sigmas.cpu().tolist(),
        "timesteps": scheduler.timesteps.cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="local scheduler_config.json from the pinned model revision")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    if config_sha != SCHEDULER_CONFIG_SHA256:
        raise SystemExit(
            f"scheduler config sha256 {config_sha} does not match pinned revision "
            f"{MODEL_REVISION} ({SCHEDULER_CONFIG_SHA256})"
        )

    config = json.loads(config_bytes)
    if config.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise SystemExit("pinned config does not name FlowMatchEulerDiscreteScheduler")

    direct_url = metadata.distribution("diffusers").read_text("direct_url.json")
    if not direct_url:
        raise SystemExit("installed Diffusers package has no direct_url.json source provenance")
    installed_source = json.loads(direct_url)
    diffusers_commit = installed_source.get("vcs_info", {}).get("commit_id")
    if diffusers_commit != DIFFUSERS_COMMIT:
        raise SystemExit(
            f"installed Diffusers commit {diffusers_commit!r} does not match "
            f"pinned source {DIFFUSERS_COMMIT}"
        )

    scheduler_source = source_path(FlowMatchEulerDiscreteScheduler)
    pipeline_source = source_path(calculate_shift)
    scheduler_source_sha = sha256(scheduler_source)
    pipeline_source_sha = sha256(pipeline_source)
    if scheduler_source_sha != SCHEDULER_SOURCE_SHA256 or pipeline_source_sha != PIPELINE_SOURCE_SHA256:
        raise SystemExit("installed Diffusers source differs from the pinned fixture generator")
    fixture = {
        "provenance": {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "scheduler_config_sha256": config_sha,
            "diffusers_commit": diffusers_commit,
            "diffusers_version": metadata.version("diffusers"),
            "scheduler_source_sha256": scheduler_source_sha,
            "qwen_image21_pipeline_source_sha256": pipeline_source_sha,
            "numpy_version": np.__version__,
            "config_diffusers_version": config.get("_diffusers_version"),
            "generation_path": (
                "Qwen-Image 2.1 calculate_shift + np.linspace, then "
                "FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=..., mu=...)"
            ),
        },
        "schedules": [
            make_case(config, steps, image_seq_len)
            for steps, image_seq_len in CASES
        ],
    }
    print(json.dumps(fixture, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
