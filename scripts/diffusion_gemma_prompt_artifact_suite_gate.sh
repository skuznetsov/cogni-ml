#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_prompt_artifact_suite_gate_$(date +%Y%m%d%H%M%S)}"

token_windows="${TOKEN_WINDOWS:-}"
base_map="${SUITE_BASE_ROUTE_ARTIFACT_MAP:-${CERT_BASE_ROUTE_ARTIFACT_MAP:-}}"
variant_map="${SUITE_VARIANT_ROUTE_ARTIFACT_MAP:-${CERT_VARIANT_ROUTE_ARTIFACT_MAP:-}}"
suite_min_total_speedup="${SUITE_MIN_TOTAL_SPEEDUP:-${MIN_TOTAL_SPEEDUP:-1.10}}"
window_min_total_speedup="${SUITE_WINDOW_MIN_TOTAL_SPEEDUP:-}"
suite_child_logs="${SUITE_CHILD_LOGS:-}"
suite_compatibility_audit="${SUITE_COMPATIBILITY_AUDIT:-0}"
suite_mixed_fallback_gate="${SUITE_MIXED_FALLBACK_GATE:-0}"
suite_run_abba_on_cert_fail="${SUITE_RUN_ABBA_ON_CERT_FAIL:-}"

usage() {
  cat <<'EOF'
Usage: diffusion_gemma_prompt_artifact_suite_gate.sh

Runs the certified prompt gate once per route-artifact window, then aggregates
the measured ABBA total_ms rows and child gate_metric phase rows across the
whole suite.

Important environment knobs:
  TOKEN_WINDOWS                         prompt:canvas windows to measure
  SUITE_BASE_ROUTE_ARTIFACT_MAP=SPEC    optional base routes, prompt:canvas=PATH
  SUITE_VARIANT_ROUTE_ARTIFACT_MAP=SPEC optional variant routes, prompt:canvas=PATH
  SUITE_CHILD_LOGS="A B ..."            aggregate existing child gate.stdout logs without rerun
  SUITE_MIN_TOTAL_SPEEDUP=F             aggregate speedup floor, default MIN_TOTAL_SPEEDUP or 1.10
  SUITE_WINDOW_MIN_TOTAL_SPEEDUP=F      per-window child gate floor, default aggregate floor,
                                        or 1.0 for explicit mixed fallback gates
  SUITE_COMPATIBILITY_AUDIT=1           continue through child rejects and report a mixed exact-fallback audit
  SUITE_MIXED_FALLBACK_GATE=1           emit mixed_candidate when the certified fast/exact fallback policy passes
  SUITE_RUN_ABBA_ON_CERT_FAIL=1         in audit mode, collect timing even when output cert fails

All ordinary diffusion_gemma_prompt_certified_variant_gate.sh environment knobs
are inherited. The suite wrapper overrides TOKEN_WINDOWS, LOG_DIR,
CERT_*_ROUTE_ARTIFACT_MAP, ABBA_*_ROUTE_ARTIFACT, and MIN_TOTAL_SPEEDUP for
each child window.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

bool_enabled() {
  case "$1" in
    1|true|yes|on) return 0 ;;
    0|false|no|off|"") return 1 ;;
    *) die "invalid boolean value: $1" ;;
  esac
}

if bool_enabled "$suite_mixed_fallback_gate"; then
  mixed_gate_enabled=1
else
  mixed_gate_enabled=0
fi
if bool_enabled "$suite_compatibility_audit" || [[ "$mixed_gate_enabled" == "1" ]]; then
  audit_enabled=1
else
  audit_enabled=0
fi
if [[ -z "$suite_run_abba_on_cert_fail" ]]; then
  suite_run_abba_on_cert_fail="$audit_enabled"
fi
bool_enabled "$suite_run_abba_on_cert_fail" >/dev/null || true
if [[ -z "$window_min_total_speedup" ]]; then
  if [[ "$mixed_gate_enabled" == "1" ]]; then
    window_min_total_speedup="1.0"
  else
    window_min_total_speedup="$suite_min_total_speedup"
  fi
fi

if [[ -z "$suite_child_logs" ]]; then
  [[ -n "$token_windows" ]] || die "TOKEN_WINDOWS is required"
  if [[ -z "$base_map" && -z "$variant_map" ]]; then
    die "SUITE_BASE_ROUTE_ARTIFACT_MAP or SUITE_VARIANT_ROUTE_ARTIFACT_MAP is required"
  fi
fi

mkdir -p "$log_dir"

suite_spec="$log_dir/suite_windows.tsv"
child_logs=()
if [[ -n "$suite_child_logs" ]]; then
  for child_log in $suite_child_logs; do
    [[ -f "$child_log" ]] || die "SUITE_CHILD_LOGS entry not found: $child_log"
    child_logs+=("$child_log")
  done
  ((${#child_logs[@]} > 0)) || die "SUITE_CHILD_LOGS must contain at least one log"
  : >"$suite_spec"
else
  python3 - "$token_windows" "$base_map" "$variant_map" >"$suite_spec" <<'PY'
import sys

token_windows_raw, base_raw, variant_raw = sys.argv[1:4]

def parse_windows(raw):
    windows = []
    for entry in raw.replace(",", " ").split():
        try:
            p, c = entry.split(":", 1)
        except ValueError:
            raise SystemExit(f"TOKEN_WINDOWS entry must be prompt:canvas, got {entry!r}")
        windows.append((int(p), int(c)))
    if not windows:
        raise SystemExit("TOKEN_WINDOWS must contain at least one entry")
    if len(set(windows)) != len(windows):
        raise SystemExit("TOKEN_WINDOWS contains duplicate entries")
    return windows

def parse_map(raw, label):
    result = {}
    if not raw:
        return result
    for entry in raw.replace(",", " ").split():
        try:
            window_raw, path = entry.split("=", 1)
            p, c = window_raw.split(":", 1)
        except ValueError:
            raise SystemExit(f"{label} entry must be prompt:canvas=PATH, got {entry!r}")
        if not path:
            raise SystemExit(f"{label} path must not be empty for {window_raw}")
        window = (int(p), int(c))
        if window in result:
            raise SystemExit(f"{label} duplicate window {window_raw}")
        result[window] = path
    return result

windows = parse_windows(token_windows_raw)
base_map = parse_map(base_raw, "SUITE_BASE_ROUTE_ARTIFACT_MAP")
variant_map = parse_map(variant_raw, "SUITE_VARIANT_ROUTE_ARTIFACT_MAP")
for label, mapping in (("SUITE_BASE_ROUTE_ARTIFACT_MAP", base_map), ("SUITE_VARIANT_ROUTE_ARTIFACT_MAP", variant_map)):
    for window in mapping:
        if window not in windows:
            raise SystemExit(f"{label} contains window {window[0]}:{window[1]} not present in TOKEN_WINDOWS")
    if mapping:
        for window in windows:
            if window not in mapping:
                raise SystemExit(f"{label} missing window {window[0]}:{window[1]}")

for index, (prompt, canvas) in enumerate(windows):
    print(f"{index}\t{prompt}\t{canvas}\t{base_map.get((prompt, canvas), '-')}\t{variant_map.get((prompt, canvas), '-')}")
PY
fi

printf 'log_dir=%s\n' "$log_dir"
printf 'token_windows=%s\n' "${token_windows:-<child_logs>}"
printf 'suite_child_logs=%s\n' "${suite_child_logs:-<empty>}"
printf 'suite_min_total_speedup=%s\n' "$suite_min_total_speedup"
printf 'suite_window_min_total_speedup=%s\n' "$window_min_total_speedup"
printf 'suite_compatibility_audit=%s\n' "$audit_enabled"
printf 'suite_mixed_fallback_gate=%s\n' "$mixed_gate_enabled"
printf 'suite_run_abba_on_cert_fail=%s\n' "$suite_run_abba_on_cert_fail"
printf 'suite_spec=%s\n' "$suite_spec"

if [[ -z "$suite_child_logs" ]]; then
  while IFS=$'\t' read -r index prompt canvas base_path variant_path; do
    [[ "$base_path" == "-" ]] && base_path=""
    [[ "$variant_path" == "-" ]] && variant_path=""
    child_dir="$log_dir/window_${index}_p${prompt}_c${canvas}"
    child_stdout="$child_dir/gate.stdout"
    mkdir -p "$child_dir"

    cert_base_map=""
    cert_variant_map=""
    if [[ -n "$base_path" ]]; then
      cert_base_map="${prompt}:${canvas}=${base_path}"
    fi
    if [[ -n "$variant_path" ]]; then
      cert_variant_map="${prompt}:${canvas}=${variant_path}"
    fi

    set +e
    LOG_DIR="$child_dir" \
    TOKEN_WINDOWS="${prompt}:${canvas}" \
    ABBA_PROMPT_TOKEN="$prompt" \
    CERT_BASE_ROUTE_ARTIFACT_MAP="$cert_base_map" \
    CERT_VARIANT_ROUTE_ARTIFACT_MAP="$cert_variant_map" \
    ABBA_BASE_ROUTE_ARTIFACT="$base_path" \
    ABBA_VARIANT_ROUTE_ARTIFACT="$variant_path" \
    MIN_TOTAL_SPEEDUP="$window_min_total_speedup" \
    RUN_ABBA_ON_CERT_FAIL="$suite_run_abba_on_cert_fail" \
    "$repo_root/scripts/diffusion_gemma_prompt_certified_variant_gate.sh" >"$child_stdout" 2>&1
    child_rc=$?
    set -e

    printf 'suite_child index=%s prompt_token=%s canvas_token=%s rc=%s log=%s\n' \
      "$index" "$prompt" "$canvas" "$child_rc" "$child_stdout"
    if (( child_rc != 0 )); then
      if [[ "$audit_enabled" == "1" && "$child_rc" -eq 4 ]]; then
        tail -20 "$child_stdout" || true
        printf 'suite_child_audit_continue index=%s prompt_token=%s canvas_token=%s child_rc=%s log=%s\n' \
          "$index" "$prompt" "$canvas" "$child_rc" "$child_stdout"
        child_logs+=("$child_stdout")
        continue
      fi
      tail -20 "$child_stdout" || true
      printf 'artifact_suite_gate decision=reject reason=child_failed index=%s prompt_token=%s canvas_token=%s child_rc=%s child_log=%s\n' \
        "$index" "$prompt" "$canvas" "$child_rc" "$child_stdout"
      exit 4
    fi
    child_logs+=("$child_stdout")
  done <"$suite_spec"
fi

python3 - "$audit_enabled" "$mixed_gate_enabled" "$suite_min_total_speedup" "$window_min_total_speedup" "${child_logs[@]}" <<'PY'
import math
import re
import sys

audit_mode = sys.argv[1] == "1"
mixed_gate = sys.argv[2] == "1"
aggregate_threshold = float(sys.argv[3])
window_threshold = float(sys.argv[4])
logs = sys.argv[5:]
total_base = 0.0
total_variant = 0.0
mixed_variant_total = 0.0
min_speedup = math.inf
rows = []
phase_rows = {}

def parse_fields(line):
    fields = {}
    for part in line.split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields

def parse_gate_metric(line, log):
    metric_fields = parse_fields(line)
    try:
        kind = metric_fields["kind"]
        metric = metric_fields["metric"]
        metric_base = float(metric_fields["base_ms"])
        metric_variant = float(metric_fields["variant_ms"])
        metric_range = float(metric_fields.get("range_over_delta", "nan"))
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"malformed gate_metric in {log}: {line}") from exc
    return kind, metric, metric_base, metric_variant, metric_range

def choose_total_metric(text, log):
    totals = {}
    for line in text:
        if not line.startswith("gate_metric "):
            continue
        kind, metric, metric_base, metric_variant, _ = parse_gate_metric(line, log)
        if metric == "total_ms":
            totals[kind] = (metric_base, metric_variant)
    for kind in ("effective_trimmed_summary", "effective_summary", "trimmed_summary", "summary"):
        if kind in totals:
            return kind, totals[kind][0], totals[kind][1]
    return None

for log in logs:
    text = open(log, encoding="utf-8", errors="replace").read().splitlines()
    decision = next((line for line in reversed(text) if line.startswith("certified_variant_gate decision=")), None)
    if decision is None:
        raise SystemExit(f"missing child decision in {log}")
    fields = parse_fields(decision)
    child_decision = fields.get("decision", "")
    accepted = child_decision.startswith("candidate")
    reason = fields.get("reason", "accepted" if accepted else "rejected")
    total_metric_kind = fields.get("total_summary_kind", "decision")
    if accepted:
        try:
            threshold_speedup = float(fields["total_speedup"])
            base = float(fields["base_ms"])
            variant = float(fields["variant_ms"])
        except KeyError as exc:
            raise SystemExit(f"missing {exc.args[0]} in {log}: {decision}") from exc
        mixed_variant = variant
        status = "candidate"
    elif audit_mode:
        chosen = choose_total_metric(text, log)
        if chosen is None:
            raise SystemExit(f"missing total_ms gate_metric for rejected audit child in {log}: {decision}")
        total_metric_kind, base, variant = chosen
        mixed_variant = base
        threshold_speedup = 1.0
        status = "fallback_exact"
    else:
        raise SystemExit(f"child did not produce a candidate decision in {log}: {decision}")
    observed_speedup = base / variant if variant > 0 else math.inf
    mixed_window_speedup = base / mixed_variant if mixed_variant > 0 else math.inf
    prompt = canvas = "n/a"
    m = re.search(r"window_(\d+)_p(-?\d+)_c(-?\d+)/gate\.stdout$", log)
    index = m.group(1) if m else "n/a"
    if m:
        prompt, canvas = m.group(2), m.group(3)
    total_base += base
    total_variant += variant
    mixed_variant_total += mixed_variant
    min_speedup = min(min_speedup, threshold_speedup)
    rows.append((index, prompt, canvas, status, reason, total_metric_kind, observed_speedup, mixed_window_speedup, base, variant, mixed_variant, log))
    for line in text:
        if not line.startswith("gate_metric "):
            continue
        kind, metric, metric_base, metric_variant, metric_range = parse_gate_metric(line, log)
        bucket = phase_rows.setdefault(
            (kind, metric),
            {"base": 0.0, "variant": 0.0, "max_range": 0.0, "windows": 0},
        )
        bucket["base"] += metric_base
        bucket["variant"] += metric_variant
        bucket["windows"] += 1
        if math.isfinite(metric_range):
            bucket["max_range"] = max(bucket["max_range"], metric_range)

for index, prompt, canvas, status, reason, total_metric_kind, observed_speedup, mixed_window_speedup, base, variant, mixed_variant, log in rows:
    print(
        "suite_window "
        f"index={index} prompt_token={prompt} canvas_token={canvas} status={status} reason={reason} "
        f"total_speedup={observed_speedup:.6f} base_ms={base:.6f} variant_ms={variant:.6f} log={log}"
    )
    if audit_mode:
        print(
            "suite_compat_window "
            f"index={index} prompt_token={prompt} canvas_token={canvas} status={status} reason={reason} "
            f"timing_kind={total_metric_kind} base_ms={base:.6f} observed_variant_ms={variant:.6f} "
            f"mixed_variant_ms={mixed_variant:.6f} observed_speedup={observed_speedup:.6f} "
            f"mixed_speedup={mixed_window_speedup:.6f} log={log}"
        )

dominant = None
for (kind, metric), bucket in sorted(phase_rows.items()):
    base = bucket["base"]
    variant = bucket["variant"]
    speedup = base / variant if variant > 0 else math.inf
    delta = base - variant
    print(
        "suite_metric "
        f"kind={kind} metric={metric} windows={bucket['windows']} "
        f"base_ms={base:.6f} variant_ms={variant:.6f} "
        f"speedup={speedup:.6f} delta_ms={delta:.6f} "
        f"max_child_range_over_delta={bucket['max_range']:.6f}"
    )
    if metric != "total_ms" and delta > 0 and (dominant is None or delta > dominant[0]):
        dominant = (delta, kind, metric, base, variant, speedup)

if dominant:
    delta, kind, metric, base, variant, speedup = dominant
    print(
        "suite_dominant_delta "
        f"kind={kind} metric={metric} delta_ms={delta:.6f} "
        f"base_ms={base:.6f} variant_ms={variant:.6f} speedup={speedup:.6f}"
    )

aggregate_speedup = total_base / total_variant if total_variant > 0 else math.inf
mixed_speedup = total_base / mixed_variant_total if mixed_variant_total > 0 else math.inf
print(
    "suite_summary "
    f"windows={len(rows)} base_ms={total_base:.6f} variant_ms={total_variant:.6f} "
    f"aggregate_speedup={aggregate_speedup:.6f} min_window_speedup={min_speedup:.6f} "
    f"min_total_speedup={aggregate_threshold:.6f} window_min_total_speedup={window_threshold:.6f}"
)
if audit_mode:
    fallback_count = sum(1 for row in rows if row[3] == "fallback_exact")
    candidate_count = len(rows) - fallback_count
    mixed_decision = "candidate" if (
        math.isfinite(mixed_speedup)
        and mixed_speedup >= aggregate_threshold
        and math.isfinite(min_speedup)
        and min_speedup >= window_threshold
    ) else "reject"
    print(
        "suite_compat_summary "
        f"windows={len(rows)} candidate_windows={candidate_count} fallback_windows={fallback_count} "
        f"base_ms={total_base:.6f} unsafe_variant_ms={total_variant:.6f} "
        f"mixed_variant_ms={mixed_variant_total:.6f} unsafe_speedup={aggregate_speedup:.6f} "
        f"mixed_speedup={mixed_speedup:.6f} min_window_speedup={min_speedup:.6f} "
        f"min_total_speedup={aggregate_threshold:.6f} window_min_total_speedup={window_threshold:.6f}"
    )
    if mixed_gate:
        if mixed_decision == "candidate":
            print(
                "artifact_suite_gate decision=mixed_candidate "
                f"mixed_speedup={mixed_speedup:.6f} unsafe_speedup={aggregate_speedup:.6f} "
                f"candidate_windows={candidate_count} fallback_windows={fallback_count} "
                f"min_total_speedup={aggregate_threshold:.6f} window_min_total_speedup={window_threshold:.6f}"
            )
            raise SystemExit(0)
        print(
            "artifact_suite_gate decision=reject reason=mixed_fallback_below_threshold "
            f"mixed_speedup={mixed_speedup:.6f} min_total_speedup={aggregate_threshold:.6f} "
            f"min_window_speedup={min_speedup:.6f} window_min_total_speedup={window_threshold:.6f} "
            f"fallback_windows={fallback_count}"
        )
        raise SystemExit(4)
    print(
        "artifact_suite_gate decision=audit_only "
        f"mixed_decision={mixed_decision} mixed_speedup={mixed_speedup:.6f} "
        f"unsafe_speedup={aggregate_speedup:.6f} fallback_windows={fallback_count} "
        f"reason=compatibility_audit_not_promotion"
    )
    raise SystemExit(0)
if not math.isfinite(aggregate_speedup) or aggregate_speedup < aggregate_threshold:
    print(
        "artifact_suite_gate decision=reject reason=aggregate_speed_below_threshold "
        f"aggregate_speedup={aggregate_speedup:.6f} min_total_speedup={aggregate_threshold:.6f}"
    )
    raise SystemExit(4)
if not math.isfinite(min_speedup) or min_speedup < window_threshold:
    print(
        "artifact_suite_gate decision=reject reason=window_speed_below_threshold "
        f"min_window_speedup={min_speedup:.6f} window_min_total_speedup={window_threshold:.6f}"
    )
    raise SystemExit(4)
print(
    "artifact_suite_gate decision=candidate "
    f"aggregate_speedup={aggregate_speedup:.6f} min_window_speedup={min_speedup:.6f} "
    f"min_total_speedup={aggregate_threshold:.6f} window_min_total_speedup={window_threshold:.6f}"
)
PY
