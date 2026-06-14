#!/usr/bin/env python3
"""Attach exact-fallback base route artifacts to a mixed route plan.

This is an offline handoff helper for the mixed fast/exact path. It takes an
existing route-plan JSONL plus a prepared base artifact map and writes a new
plan where only selected_route=base_exact windows receive base_route_artifact
values. Variant-fast windows and unsafe rejected variant artifacts are not
rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any


SUMMARY_KIND = "diffusion_gemma_mixed_route_plan_summary_v1"
WINDOW_KIND = "diffusion_gemma_mixed_route_plan_window_v1"
MAP_KEYS = ("SUITE_BASE_ROUTE_ARTIFACT_MAP", "CERT_BASE_ROUTE_ARTIFACT_MAP")


def parse_map(raw: str, label: str) -> dict[tuple[int, int], str]:
    result: dict[tuple[int, int], str] = {}
    if not raw:
        return result
    for entry in raw.replace(",", " ").split():
        try:
            window_raw, path = entry.split("=", 1)
            prompt_raw, canvas_raw = window_raw.split(":", 1)
            window = (int(prompt_raw), int(canvas_raw))
        except ValueError as exc:
            raise SystemExit(f"{label} entry must be prompt:canvas=PATH, got {entry!r}") from exc
        if window[0] < 0 or window[1] < 0:
            raise SystemExit(f"{label} window must be non-negative, got {window_raw!r}")
        if not path:
            raise SystemExit(f"{label} path must not be empty for {window_raw}")
        if window in result:
            raise SystemExit(f"{label} duplicate window {window_raw}")
        result[window] = path
    return result


def extract_map_from_prepare_log(path: Path) -> str:
    found: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in MAP_KEYS:
                found[key] = value
                continue
            if key == "artifact_suite_promotion decision":
                # Promotion summary rows may contain shell-quoted base_map=...
                for part in shlex.split(line):
                    if part.startswith("base_map="):
                        value = part.split("=", 1)[1]
                        if value:
                            found["base_map"] = value
    for key in MAP_KEYS + ("base_map",):
        if key in found and found[key]:
            return found[key]
    raise SystemExit(f"no base route artifact map found in {path}")


def load_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] | None = None
    windows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
            kind = row.get("kind")
            if kind == SUMMARY_KIND:
                if summary is not None:
                    raise SystemExit("route plan has multiple summary rows")
                summary = row
            elif kind == WINDOW_KIND:
                windows.append(row)
            else:
                raise SystemExit(f"{path}:{lineno}: unsupported route-plan row kind: {kind!r}")
    if summary is None:
        raise SystemExit("route plan missing summary row")
    expected = int(summary.get("windows", -1))
    if expected != len(windows):
        raise SystemExit(f"route-plan window count mismatch: summary={expected} rows={len(windows)}")
    seen: set[tuple[int, int]] = set()
    for row in windows:
        key = window_key(row)
        if key in seen:
            raise SystemExit(f"duplicate route-plan window {key[0]}:{key[1]}")
        seen.add(key)
        if row.get("selected_route") not in {"variant_fast", "base_exact"}:
            raise SystemExit(f"unsupported selected_route for {key[0]}:{key[1]}: {row.get('selected_route')!r}")
    return summary, windows


def window_key(row: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(row["prompt_token"]), int(row["canvas_token"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("route-plan window row has invalid prompt_token/canvas_token") from exc


def validate_map(
    base_map: dict[tuple[int, int], str],
    windows: list[dict[str, Any]],
    require_artifacts: bool,
) -> None:
    known = {window_key(row): row for row in windows}
    for key, artifact in base_map.items():
        row = known.get(key)
        if row is None:
            raise SystemExit(f"base artifact map contains window not present in route plan: {key[0]}:{key[1]}")
        if row.get("selected_route") != "base_exact":
            raise SystemExit(f"base artifact map contains non-fallback window: {key[0]}:{key[1]}")
        if require_artifacts and not os.path.isfile(artifact):
            raise SystemExit(f"base artifact missing for {key[0]}:{key[1]}: {artifact}")


def write_plan(path: Path, summary: dict[str, Any], windows: list[dict[str, Any]]) -> None:
    parent = path.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
        for row in windows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("route_plan", type=Path, help="source mixed route-plan JSONL")
    parser.add_argument("--out", type=Path, required=True, help="output route-plan JSONL")
    parser.add_argument("--base-map", default="", help="prompt:canvas=PATH entries for base_exact windows")
    parser.add_argument("--prepare-log", type=Path, help="prepare/promotion stdout containing SUITE_BASE_ROUTE_ARTIFACT_MAP")
    parser.add_argument("--allow-missing-fallbacks", action="store_true", help="do not require every base_exact window to be present in the map")
    parser.add_argument("--no-require-artifacts", action="store_true", help="do not check mapped artifact paths exist")
    parser.add_argument("--keep-fallback-variant-artifacts", action="store_true", help="preserve diagnostic fallback variant_route_artifact values")
    parser.add_argument("--overwrite", action="store_true", help="allow --out to overwrite the source route plan")
    args = parser.parse_args()

    if args.route_plan.resolve() == args.out.resolve() and not args.overwrite:
        raise SystemExit("--out must differ from route_plan unless --overwrite is set")

    base_map: dict[tuple[int, int], str] = {}
    if args.prepare_log:
        base_map.update(parse_map(extract_map_from_prepare_log(args.prepare_log), "--prepare-log"))
    if args.base_map:
        explicit = parse_map(args.base_map, "--base-map")
        overlap = set(base_map).intersection(explicit)
        if overlap:
            first = sorted(overlap)[0]
            raise SystemExit(f"duplicate base artifact map window from inputs: {first[0]}:{first[1]}")
        base_map.update(explicit)
    if not base_map:
        raise SystemExit("--base-map or --prepare-log must provide at least one fallback artifact")

    summary, windows = load_plan(args.route_plan)
    validate_map(base_map, windows, require_artifacts=not args.no_require_artifacts)

    fallback_keys = [window_key(row) for row in windows if row.get("selected_route") == "base_exact"]
    if not fallback_keys:
        raise SystemExit("route plan contains no base_exact fallback windows")
    missing = [key for key in fallback_keys if key not in base_map]
    if missing and not args.allow_missing_fallbacks:
        first = missing[0]
        raise SystemExit(f"base artifact map missing fallback window {first[0]}:{first[1]}")

    attached = 0
    out_windows: list[dict[str, Any]] = []
    for row in windows:
        out = dict(row)
        key = window_key(out)
        if out.get("selected_route") == "base_exact" and key in base_map:
            out["base_route_artifact"] = base_map[key]
            if not args.keep_fallback_variant_artifacts:
                out["variant_route_artifact"] = ""
            attached += 1
        out_windows.append(out)
    if attached == 0:
        raise SystemExit("no fallback artifacts were attached")

    write_plan(args.out, summary, out_windows)
    print(
        "attached_fallback_artifacts "
        f"source={args.route_plan} out={args.out} attached={attached} "
        f"fallback_windows={len(fallback_keys)} require_artifacts={not args.no_require_artifacts}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
