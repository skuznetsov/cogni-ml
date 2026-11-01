"""Authenticate model selection and reuse frozen drafter inputs without inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import hy_nomas_draft_results as results

BASE_INDEX_SHA = results.BASE_DIT_INDEX_SHA256
NOISE_NAMES = ("input.latent_bank", "decoder.noise", "cache.eps_bf16")


class InputReuseError(ValueError):
    """Frozen evidence or an explicitly selected model changed."""


def digest(path: Path) -> str:
    return results.sha256_file(path)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise InputReuseError(message)


def read_object(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 * 2**20,
            f"expected a bounded regular JSON file: {path}")
    value = json.loads(path.read_text(), object_pairs_hook=results._unique_json_object)
    require(isinstance(value, dict), f"JSON must contain an object: {path}")
    return value


def model_identity(lineage: Path | None, sha: str | None) -> dict[str, Any]:
    require(bool(lineage) == bool(sha), "Cinematic cache path and SHA must be supplied together")
    if lineage is None:
        return {"dit_variant": "original", "dit_weight_index_sha256": BASE_INDEX_SHA,
                "cinematic_lineage": None, "cinematic_lineage_sha256": None}
    require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
            "Cinematic lineage SHA must contain 64 lowercase hex digits")
    require(not lineage.is_symlink(), "Cinematic lineage must not be a symlink")
    path = lineage.expanduser().resolve(strict=True)
    document = read_object(path)
    require(digest(path) == sha, "Cinematic lineage SHA mismatch")
    require(document.get("base_index_sha256") == BASE_INDEX_SHA,
            "Cinematic lineage has another base index")
    index = document.get("index", {})
    require(isinstance(index, dict) and index.get("basename") == "index.json",
            "Cinematic index basename mismatch")
    index_path = path.parent / "index.json"
    read_object(index_path)
    require(digest(index_path) == index.get("sha256"), "Cinematic index SHA mismatch")
    # Native admission additionally verifies the complete lineage and weight files.
    return {"dit_variant": "cinematic-active-blocks", "dit_weight_index_sha256": digest(index_path),
            "cinematic_lineage": str(path), "cinematic_lineage_sha256": sha}


class FrozenInputs:
    """Bind a completed run to exact prompt, noise, and Qwen input manifests.

    Payloads remain at their original paths and are authenticated by native plan
    admission and again by native execution. No producer or RNG is invoked here.
    """
    def __init__(self, report: Path, suite_bytes: bytes, scenes: list[dict[str, Any]],
                 prompts: dict[str, str]) -> None:
        self.report = report.resolve(strict=True)
        self.root = self.report.parent
        self.fingerprints: dict[Path, str] = {}
        document = self.remember(self.report)
        suite_path = self.root / "suite.json"
        require(suite_path.read_bytes() == suite_bytes, "reused run suite bytes differ")
        self.fingerprints[suite_path] = digest(suite_path)
        require(document.get("status") == "complete", "reused run is incomplete")
        require(document.get("suite_sha256") == hashlib.sha256(suite_bytes).hexdigest(),
                "reused run suite SHA mismatch")
        rows = document.get("scenes", [])
        require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
                "reused run scenes must be objects")
        require([row.get("id") for row in rows] == [scene["id"] for scene in scenes],
                "reused run must have the exact ordered scene suite")
        self.bundles: dict[str, dict[str, Any]] = {}
        self.payloads: dict[str, dict[Path, str]] = {}
        for scene, row in zip(scenes, rows, strict=True):
            scene_id = scene["id"]
            results.validate_completed_result(scene, row, self.root, full_media_validation=True,
                expected_model=document.get("model"))
            require(row.get("native_prompt") == prompts[scene_id], "reused native prompt differs")
            receipt_path = results._path_under(self.root, row.get("receipt"), "reused receipt")
            receipt = self.remember(receipt_path)
            timeline_path = self.local_input(receipt.get("inputs"))
            timeline = self.remember(timeline_path, row["inputs_sha256"])
            require(timeline.get("prompt") == prompts[scene_id], "reused continuation prompt differs")
            bundle = timeline["qwen_input_bundle"]
            base_path = self.local_input(bundle["manifest_path"])
            base = self.remember(base_path, bundle["manifest_sha256"])
            self.payloads[scene_id] = {}
            for manifest_path, manifest in ((timeline_path, timeline), (base_path, base)):
                payload = self.local_input(str(manifest_path.parent / manifest["payload"]))
                require(payload.stat().st_size == manifest["payload_bytes"]
                        and digest(payload) == manifest["payload_sha256"],
                        "reused payload bytes or SHA mismatch")
                self.payloads[scene_id][payload] = manifest["payload_sha256"]
            qwen_path = self.local_input(timeline["qwen_source"]["reference_path"])
            self.remember(qwen_path, row["qwen_sha256"])
            require(timeline["qwen_source"]["reference_sha256"] == row["qwen_sha256"],
                    "reused Qwen source SHA differs")
            qwen_sha = base["tensors"]["input.qwen_bf16"]["sha256"]
            require(base.get("prompt") == prompts[scene_id]
                    and qwen_sha == row.get("qwen_tensor_sha256") == bundle["qwen_tensor_sha256"],
                    "reused Qwen conditioning differs")
            noise = {name: timeline["tensors"][name]["sha256"] for name in NOISE_NAMES}
            require(noise == row.get("noise_sha256") == document.get("shared_noise_sha256"),
                    "reused noise banks differ")
            self.bundles[scene_id] = {"continuation": timeline_path, "inputs": base_path,
                "qwen": qwen_path, "inputs_sha256": row["inputs_sha256"],
                "qwen_sha256": row["qwen_sha256"], "qwen_tensor_sha256": qwen_sha,
                "noise_sha256": noise}
        self.source = {"results": str(self.report), "results_sha256": self.fingerprints[self.report],
                       "scope": "Exact saved input reuse; archived media is a historical control, not a fresh timing pair."}

    def local_input(self, value: Any) -> Path:
        require(isinstance(value, str) and Path(value).is_absolute(), "reused input must be absolute")
        path = Path(value).resolve(strict=True)
        require(path.is_relative_to(self.root), "reused input escapes its original run")
        return path

    def remember(self, path: Path, expected: str | None = None) -> dict[str, Any]:
        document = read_object(path)
        actual = digest(path)
        require(expected is None or actual == expected, f"reused manifest SHA mismatch: {path}")
        self.fingerprints[path] = actual
        return document

    def check(self) -> None:
        for path, expected in self.fingerprints.items():
            require(digest(path) == expected, f"reused source changed: {path}")

    def scene(self, scene_id: str) -> dict[str, Any]:
        self.check()
        for path, expected in self.payloads[scene_id].items():
            require(digest(path) == expected, f"reused payload changed: {path}")
        return self.bundles[scene_id]
