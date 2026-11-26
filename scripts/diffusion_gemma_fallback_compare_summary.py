#!/usr/bin/env python3
"""Compare selected-vs-foreign fallback replay route-plan rows.

The selected replay plan may contain the full mixed suite while the foreign
replay plan usually contains only fallback windows. Compare only common
prompt:canvas windows so the decision is about the fallback corridor, not the
accepted fast windows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from diffusion_gemma_output_cert_atlas import group_rows, read_cert, truthy


SUMMARY_KIND = "diffusion_gemma_mixed_route_plan_summary_v1"
WINDOW_KIND = "diffusion_gemma_mixed_route_plan_window_v1"


def die(message: str) -> None:
    raise SystemExit(message)


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        die(f"window {window_key(row)} has invalid {key}")
        raise AssertionError from exc


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def window_key(row: dict[str, Any]) -> str:
    try:
        return f"{int(row['prompt_token'])}:{int(row['canvas_token'])}"
    except (KeyError, TypeError, ValueError) as exc:
        die(f"route-plan window row has invalid prompt/canvas token: {row!r}")
        raise AssertionError from exc


def key_parts(key: str) -> tuple[int, int]:
    prompt, canvas = key.split(":", 1)
    return int(prompt), int(canvas)


def selected_cost(row: dict[str, Any]) -> float:
    return as_float(row, "mixed_variant_ms")


def observed_cost(row: dict[str, Any]) -> float:
    if "observed_variant_ms" in row:
        return as_float(row, "observed_variant_ms")
    return selected_cost(row)


def parse_child_output_cert_log(child_log: str) -> str:
    if not child_log:
        return ""
    path = Path(child_log)
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line.startswith("output_cert_log="):
                continue
            return line.split("=", 1)[1]
    return ""


def foreign_cert_status(row: dict[str, Any]) -> dict[str, Any]:
    key = window_key(row)
    prompt_token, canvas_token = key.split(":", 1)
    child_log = str(row.get("child_log", "") or "")
    cert_log = parse_child_output_cert_log(child_log)
    result: dict[str, Any] = {
        "foreign_cert_status": "missing_output_cert_log",
        "foreign_cert_log": cert_log,
        "foreign_cert_argmax_matches": math.nan,
        "foreign_cert_rows": math.nan,
    }
    if not child_log or not Path(child_log).is_file():
        result["foreign_cert_status"] = "missing_child_log"
        return result
    if not cert_log or not Path(cert_log).is_file():
        return result

    rows, _, _ = read_cert(Path(cert_log))
    window_rows = group_rows(rows).get((prompt_token, canvas_token), [])
    if not window_rows:
        result["foreign_cert_status"] = "missing_cert_window"
        result["foreign_cert_log"] = cert_log
        return result
    argmax_matches = sum(1 for cert_row in window_rows if truthy(cert_row.get("argmax_match", "")))
    result.update(
        {
            "foreign_cert_status": "argmax_match" if argmax_matches == len(window_rows) else "argmax_fail",
            "foreign_cert_log": cert_log,
            "foreign_cert_argmax_matches": float(argmax_matches),
            "foreign_cert_rows": float(len(window_rows)),
        }
    )
    return result


def load_windows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        die(f"route plan not found: {path}")
    windows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                die(f"{path} invalid JSONL at line {lineno}: {exc}")
            if row.get("kind") != WINDOW_KIND:
                continue
            key = window_key(row)
            if key in windows:
                die(f"{path} contains duplicate window {key}")
            windows[key] = row
    if not windows:
        die(f"{path} contains no {WINDOW_KIND} rows")
    return windows


def load_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not path.is_file():
        die(f"route plan not found: {path}")
    summary: dict[str, Any] | None = None
    windows: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                die(f"{path} invalid JSONL at line {lineno}: {exc}")
            kind = row.get("kind")
            if kind == SUMMARY_KIND:
                if summary is not None:
                    die(f"{path} contains multiple summary rows")
                summary = row
            elif kind == WINDOW_KIND:
                key = window_key(row)
                if key in by_key:
                    die(f"{path} contains duplicate window {key}")
                windows.append(row)
                by_key[key] = row
    if summary is None:
        die(f"{path} contains no {SUMMARY_KIND} row")
    expected = int(summary.get("windows", -1))
    if expected != len(windows):
        die(f"{path} route-plan window count mismatch: summary={expected} rows={len(windows)}")
    if not windows:
        die(f"{path} contains no {WINDOW_KIND} rows")
    return summary, windows, by_key


def compare(selected_path: Path, foreign_path: Path, min_foreign_speedup: float, inspect_foreign_cert: bool = False) -> list[dict[str, Any]]:
    selected = load_windows(selected_path)
    foreign = load_windows(foreign_path)
    common = sorted(set(selected) & set(foreign), key=lambda key: tuple(map(int, key.split(":"))))
    if not common:
        die("selected and foreign route plans have no common windows")

    rows: list[dict[str, Any]] = []
    for key in common:
        selected_row = selected[key]
        foreign_row = foreign[key]
        selected_ms = selected_cost(selected_row)
        foreign_ms = selected_cost(foreign_row)
        ratio = selected_ms / foreign_ms if foreign_ms > 0 else math.inf
        delta = selected_ms - foreign_ms
        if ratio > 1.0:
            winner = "foreign"
        elif ratio < 1.0:
            winner = "selected"
        else:
            winner = "tie"
        out = {
            "window": key,
            "winner": winner,
            "foreign_vs_selected": ratio,
            "delta_ms": delta,
            "selected_ms": selected_ms,
            "foreign_ms": foreign_ms,
            "min_foreign_speedup": min_foreign_speedup,
            "threshold_pass": ratio >= min_foreign_speedup,
            "selected_route": selected_row.get("selected_route", ""),
            "foreign_route": foreign_row.get("selected_route", ""),
            "selected_status": selected_row.get("status", ""),
            "foreign_status": foreign_row.get("status", ""),
            "selected_reason": selected_row.get("reason", ""),
            "foreign_reason": foreign_row.get("reason", ""),
            "selected_observed_ms": observed_cost(selected_row),
            "foreign_observed_ms": observed_cost(foreign_row),
            "foreign_cert_status": "",
            "foreign_cert_log": "",
            "foreign_cert_argmax_matches": math.nan,
            "foreign_cert_rows": math.nan,
        }
        if inspect_foreign_cert:
            out.update(foreign_cert_status(foreign_row))
        rows.append(out)
    return rows


def foreign_cert_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        promotable = (
            row.get("selected_route") == "base_exact" and
            row.get("foreign_route") == "variant_fast" and
            bool(row.get("threshold_pass"))
        )
        if promotable and row.get("foreign_cert_status") != "argmax_match":
            failures.append(row)
    return failures


def normalize_window(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    selected_route = str(out.get("selected_route", ""))
    if selected_route == "variant_fast":
        out.setdefault("variant_env_role", "variant")
        out.setdefault("selected_route_artifact_arm", "variant")
        out.setdefault("selected_route_artifact_env_role", out["variant_env_role"])
    elif selected_route == "base_exact":
        out.setdefault("variant_env_role", "base")
        out.setdefault("selected_route_artifact_arm", "base")
        out.setdefault("selected_route_artifact_env_role", out["variant_env_role"])
    else:
        die(f"unsupported selected_route for promoted plan window {window_key(out)}: {selected_route!r}")
    for key in ("variant_env_role", "selected_route_artifact_arm", "selected_route_artifact_env_role"):
        if out.get(key) not in {"base", "variant"}:
            die(f"promoted plan window {window_key(out)} has invalid {key}={out.get(key)!r}")
    if selected_route == "variant_fast" and not out.get("variant_route_artifact"):
        die(f"promoted variant_fast window {window_key(out)} requires variant_route_artifact")
    return out


def promoted_summary(template: dict[str, Any], windows: list[dict[str, Any]], reason: str, min_foreign_speedup: float, promoted_count: int) -> dict[str, Any]:
    base_ms = sum(as_float(row, "base_ms") for row in windows)
    mixed_ms = sum(selected_cost(row) for row in windows)
    unsafe_ms = sum(observed_cost(row) for row in windows)
    candidate_count = sum(1 for row in windows if row.get("selected_route") == "variant_fast")
    fallback_count = sum(1 for row in windows if row.get("selected_route") == "base_exact")
    window_speedups = [
        as_float(row, "base_ms") / selected_cost(row)
        for row in windows
        if selected_cost(row) > 0
    ]
    summary = dict(template)
    summary.update(
        {
            "kind": SUMMARY_KIND,
            "decision": "candidate" if fallback_count == 0 else "mixed_candidate",
            "reason": reason,
            "windows": len(windows),
            "candidate_windows": candidate_count,
            "fallback_windows": fallback_count,
            "base_ms": base_ms,
            "unsafe_variant_ms": unsafe_ms,
            "mixed_variant_ms": mixed_ms,
            "unsafe_speedup": base_ms / unsafe_ms if unsafe_ms > 0 else math.inf,
            "mixed_speedup": base_ms / mixed_ms if mixed_ms > 0 else math.inf,
            "min_window_speedup": min(window_speedups) if window_speedups else math.inf,
            "min_foreign_speedup": min_foreign_speedup,
            "foreign_promoted_windows": promoted_count,
        }
    )
    return summary


def write_promoted_route_plan(selected_path: Path, foreign_path: Path, min_foreign_speedup: float, out_path: Path) -> int:
    selected_summary, selected_windows, _ = load_plan(selected_path)
    _, _, foreign_by_key = load_plan(foreign_path)
    promoted_windows: list[dict[str, Any]] = []
    promoted_count = 0
    for selected_row in selected_windows:
        key = window_key(selected_row)
        foreign_row = foreign_by_key.get(key)
        if foreign_row and selected_row.get("selected_route") == "base_exact":
            selected_ms = selected_cost(selected_row)
            foreign_ms = selected_cost(foreign_row)
            ratio = selected_ms / foreign_ms if foreign_ms > 0 else math.inf
            if ratio >= min_foreign_speedup:
                if foreign_row.get("selected_route") != "variant_fast":
                    die(f"foreign promoted window {key} must be selected_route=variant_fast")
                replacement = normalize_window(foreign_row)
                replacement["promoted_from_selected_route"] = selected_row.get("selected_route", "")
                replacement["promoted_selected_child_log"] = selected_row.get("child_log", "")
                replacement["promoted_foreign_child_log"] = foreign_row.get("child_log", "")
                replacement["promoted_foreign_vs_selected"] = ratio
                replacement["reason"] = f"foreign_replay_promoted:{foreign_row.get('reason', '')}"
                promoted_windows.append(replacement)
                promoted_count += 1
                continue
        promoted_windows.append(normalize_window(selected_row))
    if promoted_count == 0:
        die("no selected base_exact windows were promoted from the foreign plan")

    summary = promoted_summary(
        selected_summary,
        promoted_windows,
        "foreign_fallback_replay_promoted",
        min_foreign_speedup,
        promoted_count,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
        for row in promoted_windows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return promoted_count


def aggregate(rows: list[dict[str, Any]], min_foreign_speedup: float) -> dict[str, Any]:
    selected_ms = sum(float(row["selected_ms"]) for row in rows)
    foreign_ms = sum(float(row["foreign_ms"]) for row in rows)
    ratio = selected_ms / foreign_ms if foreign_ms > 0 else math.inf
    if ratio > 1.0:
        winner = "foreign"
    elif ratio < 1.0:
        winner = "selected"
    else:
        winner = "tie"
    return {
        "window": "ALL_COMMON",
        "winner": winner,
        "foreign_vs_selected": ratio,
        "delta_ms": selected_ms - foreign_ms,
        "selected_ms": selected_ms,
        "foreign_ms": foreign_ms,
        "min_foreign_speedup": min_foreign_speedup,
        "threshold_pass": ratio >= min_foreign_speedup,
        "selected_route": "",
        "foreign_route": "",
        "selected_status": "",
        "foreign_status": "",
        "selected_reason": "",
        "foreign_reason": "",
        "selected_observed_ms": math.nan,
        "foreign_observed_ms": math.nan,
        "foreign_cert_status": "",
        "foreign_cert_log": "",
        "foreign_cert_argmax_matches": math.nan,
        "foreign_cert_rows": math.nan,
    }


FIELDS = [
    "window",
    "winner",
    "foreign_vs_selected",
    "delta_ms",
    "selected_ms",
    "foreign_ms",
    "min_foreign_speedup",
    "threshold_pass",
    "selected_route",
    "foreign_route",
    "selected_status",
    "foreign_status",
    "selected_reason",
    "foreign_reason",
    "selected_observed_ms",
    "foreign_observed_ms",
    "foreign_cert_status",
    "foreign_cert_log",
    "foreign_cert_argmax_matches",
    "foreign_cert_rows",
]


def print_tsv(rows: list[dict[str, Any]]) -> None:
    print("\t".join(FIELDS))
    for row in rows:
        values: list[str] = []
        for field in FIELDS:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        print("\t".join(values))


def print_text(rows: list[dict[str, Any]], selected_path: Path, foreign_path: Path, min_foreign_speedup: float) -> None:
    total = aggregate(rows, min_foreign_speedup)
    print("DiffusionGemma fallback replay compare")
    print(f"  selected_plan={selected_path}")
    print(f"  foreign_plan={foreign_path}")
    print(
        "  aggregate common_windows=%d winner=%s foreign_vs_selected=%s "
        "min_foreign_speedup=%s threshold_pass=%s selected_ms=%s foreign_ms=%s delta_ms=%s"
        % (
            len(rows),
            total["winner"],
            fmt(float(total["foreign_vs_selected"])),
            fmt(float(total["min_foreign_speedup"])),
            str(total["threshold_pass"]).lower(),
            fmt(float(total["selected_ms"])),
            fmt(float(total["foreign_ms"])),
            fmt(float(total["delta_ms"])),
        )
    )
    print("  windows:")
    for row in rows:
        print(
            "    %s winner=%s foreign_vs_selected=%s threshold_pass=%s "
            "selected=%sms(%s/%s) foreign=%sms(%s/%s) delta_ms=%s cert=%s"
            % (
                row["window"],
                row["winner"],
                fmt(float(row["foreign_vs_selected"])),
                str(row["threshold_pass"]).lower(),
                fmt(float(row["selected_ms"])),
                row["selected_route"],
                row["selected_status"],
                fmt(float(row["foreign_ms"])),
                row["foreign_route"],
                row["foreign_status"],
                fmt(float(row["delta_ms"])),
                row.get("foreign_cert_status", "") or "not_checked",
            )
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-route-plan", required=True, type=Path)
    parser.add_argument("--foreign-route-plan", required=True, type=Path)
    parser.add_argument("--min-foreign-speedup", type=float, default=1.0)
    parser.add_argument("--require-foreign", "--require-foreign-pass", dest="require_foreign", action="store_true")
    parser.add_argument("--require-foreign-cert", action="store_true", help="require argmax output-cert evidence for threshold-passing foreign promotions")
    parser.add_argument("--promoted-route-plan-out", type=Path, help="write selected plan with threshold-passing base_exact windows replaced by foreign variant_fast rows")
    parser.add_argument("--tsv", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_foreign_speedup) or args.min_foreign_speedup <= 0.0:
        die("--min-foreign-speedup must be a positive finite number")

    rows = compare(args.selected_route_plan, args.foreign_route_plan, args.min_foreign_speedup, inspect_foreign_cert=args.require_foreign_cert)
    summary = aggregate(rows, args.min_foreign_speedup)
    output_rows = [summary] + rows
    if args.tsv:
        print_tsv(output_rows)
    else:
        print_text(rows, args.selected_route_plan, args.foreign_route_plan, args.min_foreign_speedup)
    if args.require_foreign and not summary["threshold_pass"]:
        print(
            "fallback compare rejected: foreign_vs_selected=%s below min_foreign_speedup=%s"
            % (fmt(float(summary["foreign_vs_selected"])), fmt(float(args.min_foreign_speedup))),
            file=sys.stderr,
        )
        return 4
    if args.require_foreign_cert:
        cert_failures = foreign_cert_failures(rows)
        if cert_failures:
            details = ", ".join(
                "%s status=%s cert=%s"
                % (row["window"], row.get("foreign_cert_status", ""), row.get("foreign_cert_log", ""))
                for row in cert_failures
            )
            print(f"fallback compare rejected: foreign certificate guard failed: {details}", file=sys.stderr)
            return 4
    if args.promoted_route_plan_out:
        promoted_count = write_promoted_route_plan(
            args.selected_route_plan,
            args.foreign_route_plan,
            args.min_foreign_speedup,
            args.promoted_route_plan_out,
        )
        print(f"promoted_route_plan_out={args.promoted_route_plan_out} promoted_windows={promoted_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
