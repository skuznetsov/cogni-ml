#!/usr/bin/env python3
"""Render a frozen prompt suite with the bounded native Hy-Nomas pipeline.

Prompt planning is a separate, editable step (see hy_nomas_plan.py). This
launcher never calls a cloud API. It runs one guarded process at a time and
stops on the first failure, retaining logs and already completed first takes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "examples" / "support"
sys.path.insert(0, str(SUPPORT))
import hy_nomas_draft_results as results

DEFAULT_PYTHON = Path("/opt/mldata_ssd/cogni-ml/drama3-fish-reference-20261015/venv/bin/python")
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024")
DEFAULT_CHECKPOINT = Path("/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024")
NAMED_BUDGET_MIB = 27_648
NATIVE_CAP_MIB = 30_720
FLOOR_PCT = "10"
SOURCE_PATHS = (
    "examples/hy_nomas_draft.py",
    "examples/support/hy_nomas_draft_results.py",
    "examples/hy_nomas_sequence_generate.cr",
    "examples/support/hy_nomas_sequence.cr",
    "examples/support/hy_nomas_generation.cr",
    "examples/support/qwen25_pretrained_reference.py",
    "examples/support/qwen25_hy_nomas_prompt_reference.py",
    "spec/support/hy_nomas_generation_inputs.py",
    "spec/support/hy_nomas_continuation_inputs.py",
    "examples/support/wan_memory_guard.py",
    "examples/hy_nomas_export.py",
    "examples/wan_s2v_export.py",
)


class DraftError(RuntimeError):
    """A bounded stage or its evidence failed."""


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any, *, fresh: bool = False) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if fresh:
        with path.open("x", encoding="utf-8") as output:
            output.write(encoded)
    else:
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("x", encoding="utf-8") as output:
            output.write(encoded)
        temporary.replace(path)


def native_prompt(scene: dict[str, Any]) -> str:
    prompt = scene.get("native_prompt")
    if prompt is None:
        prompt = json.dumps({
            "Style": "Photorealistic real-world footage",
            "perspective_view": "third-person observer",
            "subject_list": [],
            "segments": [{
                "start_time": "00:00.000", "end_time": "00:10.000",
                "visual_description_long": scene["prompt"],
                "description_change": None, "contained_actions": [],
            }],
        }, ensure_ascii=False, indent=2)
    results.validate_native_prompt(prompt, scene["prompt"])
    if len(prompt.encode("utf-8")) > 64 * 1024:
        raise DraftError("native prompt exceeds the upstream 64 KiB bound")
    return prompt


def guard_summary(path: Path) -> dict[str, Any]:
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, ValueError):
        raise DraftError(f"guard receipt is unreadable or malformed: {path}") from None
    if any(not isinstance(row, dict) or not isinstance(row.get("event"), str) for row in records):
        raise DraftError(f"guard receipt has malformed records: {path}")
    samples = [row for row in records if row.get("event") == "sample"]
    if (not records or records[-1].get("event") != "finished"
            or type(records[-1].get("returncode")) is not int
            or records[-1]["returncode"] != 0 or not samples
            or type(samples[-1].get("group_live_processes")) is not int
            or samples[-1]["group_live_processes"] != 0):
        raise DraftError(f"guard did not certify successful cleanup: {path}")
    for row in samples:
        for key in ("group_footprint_bytes", "headroom_bytes", "swap_used_bytes", "group_live_processes"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise DraftError(f"guard sample has an invalid {key}: {path}")
    return {
        "peak_footprint_gib": max(row["group_footprint_bytes"] for row in samples) / 2**30,
        "min_headroom_gib": min(row["headroom_bytes"] for row in samples) / 2**30,
        "swap_growth_bytes": max(row["swap_used_bytes"] for row in samples)
        - samples[0]["swap_used_bytes"],
    }


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.suite_bytes = args.suite.resolve(strict=True).read_bytes()
        self.suite = json.loads(self.suite_bytes)
        self.scenes = results.validate_suite(self.suite)
        if len(self.scenes) > 6:
            raise DraftError("the initial bounded suite supports at most six scenes")
        # Validate every prompt before creating a run or starting a model.
        self.prompts = {scene["id"]: native_prompt(scene) for scene in self.scenes}
        self.generator = args.generator.resolve(strict=True)
        self.binary_sha = results.sha256_file(self.generator)
        self.python = args.reference_python.expanduser().absolute()
        if not self.python.is_file() or not os.access(self.python, os.X_OK):
            raise DraftError("reference Python must be an executable file")
        self.upstream = args.upstream.resolve(strict=True)
        self.checkpoint = args.checkpoint.resolve(strict=True)
        self.run = args.run_dir.expanduser().absolute()
        scratch = Path("/opt/mldata_ssd").resolve(strict=True)
        if self.run.exists() or not self.run.parent.resolve(strict=True).is_relative_to(scratch):
            raise DraftError("run directory must be fresh and below /opt/mldata_ssd")
        if shutil.disk_usage(scratch).free < 24 * 2**30:
            raise DraftError("at least 24 GiB of scratch disk headroom is required")
        self.run.mkdir()
        self.run = self.run.resolve()
        for name in ("prompts", "conditioning", "results", "logs", "tmp"):
            (self.run / name).mkdir()
        (self.run / "suite.json").write_bytes(self.suite_bytes)
        self.env = dict(os.environ, TMPDIR=str(self.run / "tmp"),
                        PYTHONDONTWRITEBYTECODE="1", PYTORCH_ENABLE_MPS_FALLBACK="0")
        # No child of the local renderer needs the cloud planner credential.
        self.env.pop("OPENAI_API_KEY", None)
        self.source_shas = {name: results.sha256_file(ROOT / name) for name in SOURCE_PATHS}
        self.document: dict[str, Any] = {
            "schema": results.RESULTS_SCHEMA,
            "status": "running",
            "suite_sha256": sha_bytes(self.suite_bytes),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "generator": str(self.generator), "generator_sha256": self.binary_sha,
            "selected_source_sha256": self.source_shas,
            "source_scope": "Selected orchestration/producer snapshots and observed executable hash; automatic build correspondence is not proven.",
            "conditioning_scope": "Pinned upstream Qwen MPS state-26 capture; native DiT and PixelUMM.",
            "guard_policy": {"memory_accounting": "file-cache", "runtime_floor_pct": 10,
                             "native_cap_mib": NATIVE_CAP_MIB, "max_swap_growth_mib": 64},
            "scenes": [{"id": scene["id"], "status": "pending"} for scene in self.scenes],
        }
        write_json(self.run / "run.json", {key: value for key, value in self.document.items()
                                          if key != "scenes"}, fresh=True)
        self.refresh()

    def refresh(self) -> None:
        write_json(self.run / "results.json", self.document)
        results.write_gallery(self.suite, self.document, self.run / "index.html", self.run,
                              suite_sha256=self.document["suite_sha256"])

    def check_sources(self) -> None:
        if results.sha256_file(self.generator) != self.binary_sha:
            raise DraftError("native executable changed during the suite")
        for name, expected in self.source_shas.items():
            if results.sha256_file(ROOT / name) != expected:
                raise DraftError(f"producer source changed during the suite: {name}")

    def command(self, name: str, argv: list[str], cap_mib: int,
                timeout_sec: int) -> dict[str, Any]:
        self.check_sources()
        log = self.run / "logs" / f"{name}.jsonl"
        output = self.run / "logs" / f"{name}.out"
        guarded = [sys.executable, "-B", str(SUPPORT / "wan_memory_guard.py"),
                   "--memory-accounting", "file-cache", "--runtime-floor-pct", FLOOR_PCT,
                   "--max-footprint-mib", str(cap_mib), "--max-swap-growth-mib", "64",
                   "--timeout-sec", str(timeout_sec), "--log", str(log), "--", *argv]
        record = {"argv": guarded, "cwd": str(ROOT),
                  "started_at": datetime.now(timezone.utc).isoformat()}
        record_path = self.run / "logs" / f"{name}-command.json"
        write_json(record_path, record, fresh=True)
        print(json.dumps({"scene_stage": name, "status": "running"}), flush=True)
        started = time.monotonic()
        with output.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(guarded, cwd=ROOT, env=self.env,
                                       stdout=stream, stderr=subprocess.STDOUT)
        record.update(returncode=completed.returncode, seconds=time.monotonic() - started)
        write_json(record_path, record)
        if completed.returncode:
            raise DraftError(f"{name} failed with exit {completed.returncode}; see {output}")
        record.update(guard_summary(log))
        self.check_sources()
        write_json(record_path, record)
        print(json.dumps({"scene_stage": name, "status": "finished",
                          "seconds": record["seconds"],
                          "peak_gib": record["peak_footprint_gib"]}), flush=True)
        return record

    def render(self, scene: dict[str, Any], row: dict[str, Any]) -> None:
        scene_id = scene["id"]
        row.update(status="running", native_prompt=self.prompts[scene_id],
                   prompt_sha256=sha_bytes(self.prompts[scene_id].encode("utf-8")))
        self.refresh()
        started = time.monotonic()
        prompt = self.run / "prompts" / f"{scene_id}.json"
        prompt.write_text(self.prompts[scene_id], encoding="utf-8")
        condition = self.run / "conditioning"
        qwen = condition / f"{scene_id}-qwen.json"
        inputs = condition / f"{scene_id}-inputs.json"
        continuation = condition / f"{scene_id}-continuation.json"
        py = [str(self.python), "-B"]
        stages = []
        stages.append(self.command(f"{scene_id}-qwen", [*py,
            str(SUPPORT / "qwen25_pretrained_reference.py"),
            "--checkpoint-root", str(self.checkpoint / "model"),
            "--checkpoint-receipt", str(self.checkpoint / "logs/checkpoint-download.json"),
            "--output", str(qwen), "--hy-nomas-prompt-file", str(prompt),
            "--hy-nomas-root", str(self.upstream)], 15_360, 600))
        stages.append(self.command(f"{scene_id}-inputs", [*py,
            str(ROOT / "spec/support/hy_nomas_generation_inputs.py"),
            "--output", str(inputs), "--qwen-reference", str(qwen),
            "--prompt-file", str(prompt)], 4096, 180))
        stages.append(self.command(f"{scene_id}-continuation", [*py,
            str(ROOT / "spec/support/hy_nomas_continuation_inputs.py"),
            "--output", str(continuation), "--qwen-inputs", str(inputs),
            "--qwen-inputs-sha256", results.sha256_file(inputs),
            "--upstream", str(self.upstream)], 15_360, 600))
        prepared = json.loads(continuation.read_text())
        if prepared.get("prompt") != row["native_prompt"]:
            raise DraftError("prepared continuation is conditioned on another prompt")
        noise = {name: prepared["tensors"][name]["sha256"]
                 for name in ("input.latent_bank", "decoder.noise", "cache.eps_bf16")}
        prior_noise = self.document.get("shared_noise_sha256")
        if prior_noise is not None and noise != prior_noise:
            raise DraftError("scene noise banks differ from the controlled suite")
        self.document["shared_noise_sha256"] = noise
        qwen_tensor_sha = json.loads(inputs.read_text())["tensors"]["input.qwen_bf16"]["sha256"]
        for previous in self.document["scenes"]:
            if (previous.get("qwen_tensor_sha256") == qwen_tensor_sha
                    and previous.get("native_prompt") != row["native_prompt"]):
                raise DraftError("different prompts reused an identical Qwen conditioning tensor")
        row.update(noise_sha256=noise, qwen_tensor_sha256=qwen_tensor_sha)
        input_sha = results.sha256_file(continuation)
        row.update(inputs_sha256=input_sha, qwen_sha256=results.sha256_file(qwen))
        native = [str(self.generator), "--inputs", str(continuation),
                  "--inputs-sha", input_sha, "--budget-mib", str(NAMED_BUDGET_MIB),
                  "--segments", "2"]
        self.command(f"{scene_id}-plan", [*native, "--plan-only"], 4096, 180)
        plans = []
        for line in (self.run / "logs" / f"{scene_id}-plan.out").read_text().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("event") == "sequence_memory_plan":
                plans.append(value)
        expected_plan = {"segments": 2, "frames": 240, "steps_per_segment": 4,
                         "named_budget_bytes": NAMED_BUDGET_MIB * 1024**2}
        if len(plans) != 1 or any(type(plans[0].get(key)) is not int or plans[0][key] != value
                                  for key, value in expected_plan.items()):
            raise DraftError("native plan did not authenticate the two-segment four-step profile")
        if (plans[0].get("full_upstream_video_parity") is not False
                or plans[0].get("text_cache_policy") != "once_per_scene"):
            raise DraftError("native plan changed the admitted cache/parity policy")
        for key in ("dit_peak_named_bytes", "decoder_peak_named_bytes"):
            value = plans[0].get(key)
            if type(value) is not int or not 0 < value <= NAMED_BUDGET_MIB * 1024**2:
                raise DraftError(f"native plan has an invalid or excessive {key}")
        row["memory_plan"] = plans[0]
        self.refresh()
        output = self.run / "results" / scene_id
        output.mkdir()
        native_record = self.command(f"{scene_id}-native",
            [*native, "--output-dir", str(output)], NATIVE_CAP_MIB, 1200)
        stages.append(native_record)
        receipt_path = output / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        metadata = results.validate_metadata(receipt, row["native_prompt"], input_sha)
        raw = results.validate_rawpixels(Path(metadata["pixels"]), metadata["pixels_sha256"])
        video = output / "draft.mp4"
        stages.append(self.command(f"{scene_id}-export", [*py,
            str(ROOT / "examples/hy_nomas_export.py"), "--input", str(raw["path"]),
            "--output", str(video), "--frames", "240", "--latent-frames", "60",
            "--height", "288", "--width", "512", "--fps", "24"], 4096, 180))
        probe = results.ffprobe_mp4(video)
        row["media"] = results.validate_ffprobe(probe)
        write_json(output / "ffprobe.json", probe, fresh=True)
        row.update(status="complete", receipt=f"results/{scene_id}/receipt.json",
                   video=f"results/{scene_id}/draft.mp4", pixels_sha256=raw["sha256"],
                   video_sha256=results.sha256_file(video),
                   timings={**receipt["timings"], "total_scene_seconds": time.monotonic() - started,
                            "conditioning_seconds": sum(stage["seconds"] for stage in stages[:3])},
                   memory_peak_gib=max(stage["peak_footprint_gib"] for stage in stages),
                   min_headroom_gib=min(stage["min_headroom_gib"] for stage in stages))
        self.refresh()
        print(json.dumps({"scene": scene_id, "status": "complete", "video": str(video),
                          "seconds": row["timings"]["total_scene_seconds"]}), flush=True)

    def execute(self) -> None:
        for scene, row in zip(self.scenes, self.document["scenes"], strict=True):
            try:
                self.render(scene, row)
            except Exception as error:
                row.update(status="failed", error=str(error))
                self.document["status"] = "incomplete"
                self.refresh()
                raise
        # Publish completion only after the final full-media gate succeeds.
        # Preserve a failure-visible partial gallery if that gate rejects.
        candidate = {**self.document, "status": "complete",
                     "finished_at": datetime.now(timezone.utc).isoformat()}
        try:
            results.write_gallery(self.suite, candidate, self.run / "index.html", self.run,
                                  suite_sha256=self.document["suite_sha256"], full_media_validation=True)
        except Exception:
            self.document["status"] = "incomplete"
            self.refresh()
            raise
        self.document = candidate
        write_json(self.run / "results.json", self.document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path,
                        default=ROOT / "examples/data/hy_nomas_photorealistic_drafts.json")
    parser.add_argument("--run-dir", required=True, type=Path, help="Fresh directory on /opt/mldata_ssd")
    parser.add_argument("--generator", required=True, type=Path, help="Fresh native sequence executable")
    parser.add_argument("--reference-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    try:
        runner = Runner(args)
        runner.execute()
    except (DraftError, results.DraftResultsError, OSError, ValueError) as error:
        print(f"Draft suite incomplete: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "gallery": str(runner.run / "index.html")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
