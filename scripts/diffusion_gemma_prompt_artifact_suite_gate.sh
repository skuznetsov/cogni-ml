#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_prompt_artifact_suite_gate_$(date +%Y%m%d%H%M%S)}"

token_windows="${TOKEN_WINDOWS:-}"
base_map="${SUITE_BASE_ROUTE_ARTIFACT_MAP:-${CERT_BASE_ROUTE_ARTIFACT_MAP:-}}"
variant_map="${SUITE_VARIANT_ROUTE_ARTIFACT_MAP:-${CERT_VARIANT_ROUTE_ARTIFACT_MAP:-}}"
suite_min_total_speedup="${SUITE_MIN_TOTAL_SPEEDUP:-${MIN_TOTAL_SPEEDUP:-1.10}}"
window_min_total_speedup="${SUITE_WINDOW_MIN_TOTAL_SPEEDUP:-$suite_min_total_speedup}"

usage() {
  cat <<'EOF'
Usage: diffusion_gemma_prompt_artifact_suite_gate.sh

Runs the certified prompt gate once per route-artifact window, then aggregates
the measured ABBA total_ms rows across the whole suite.

Important environment knobs:
  TOKEN_WINDOWS                         prompt:canvas windows to measure
  SUITE_BASE_ROUTE_ARTIFACT_MAP=SPEC    optional base routes, prompt:canvas=PATH
  SUITE_VARIANT_ROUTE_ARTIFACT_MAP=SPEC optional variant routes, prompt:canvas=PATH
  SUITE_MIN_TOTAL_SPEEDUP=F             aggregate speedup floor, default MIN_TOTAL_SPEEDUP or 1.10
  SUITE_WINDOW_MIN_TOTAL_SPEEDUP=F      per-window child gate floor, default aggregate floor

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

[[ -n "$token_windows" ]] || die "TOKEN_WINDOWS is required"
if [[ -z "$base_map" && -z "$variant_map" ]]; then
  die "SUITE_BASE_ROUTE_ARTIFACT_MAP or SUITE_VARIANT_ROUTE_ARTIFACT_MAP is required"
fi

mkdir -p "$log_dir"

suite_spec="$log_dir/suite_windows.tsv"
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

printf 'log_dir=%s\n' "$log_dir"
printf 'token_windows=%s\n' "$token_windows"
printf 'suite_min_total_speedup=%s\n' "$suite_min_total_speedup"
printf 'suite_window_min_total_speedup=%s\n' "$window_min_total_speedup"
printf 'suite_spec=%s\n' "$suite_spec"

child_logs=()
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
  "$repo_root/scripts/diffusion_gemma_prompt_certified_variant_gate.sh" >"$child_stdout" 2>&1
  child_rc=$?
  set -e

  printf 'suite_child index=%s prompt_token=%s canvas_token=%s rc=%s log=%s\n' \
    "$index" "$prompt" "$canvas" "$child_rc" "$child_stdout"
  if (( child_rc != 0 )); then
    tail -20 "$child_stdout" || true
    printf 'artifact_suite_gate decision=reject reason=child_failed index=%s prompt_token=%s canvas_token=%s child_rc=%s child_log=%s\n' \
      "$index" "$prompt" "$canvas" "$child_rc" "$child_stdout"
    exit 4
  fi
  child_logs+=("$child_stdout")
done <"$suite_spec"

python3 - "$suite_min_total_speedup" "${child_logs[@]}" <<'PY'
import math
import re
import sys

threshold = float(sys.argv[1])
logs = sys.argv[2:]
total_base = 0.0
total_variant = 0.0
min_speedup = math.inf
rows = []

for log in logs:
    text = open(log, encoding="utf-8", errors="replace").read().splitlines()
    decision = next((line for line in reversed(text) if line.startswith("certified_variant_gate decision=")), None)
    if decision is None:
        raise SystemExit(f"missing child decision in {log}")
    fields = {}
    for part in decision.split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    try:
        speedup = float(fields["total_speedup"])
        base = float(fields["base_ms"])
        variant = float(fields["variant_ms"])
    except KeyError as exc:
        raise SystemExit(f"missing {exc.args[0]} in {log}: {decision}") from exc
    prompt = canvas = "n/a"
    m = re.search(r"window_(\d+)_p(-?\d+)_c(-?\d+)/gate\.stdout$", log)
    index = m.group(1) if m else "n/a"
    if m:
        prompt, canvas = m.group(2), m.group(3)
    total_base += base
    total_variant += variant
    min_speedup = min(min_speedup, speedup)
    rows.append((index, prompt, canvas, speedup, base, variant, log))

for index, prompt, canvas, speedup, base, variant, log in rows:
    print(
        "suite_window "
        f"index={index} prompt_token={prompt} canvas_token={canvas} "
        f"total_speedup={speedup:.6f} base_ms={base:.6f} variant_ms={variant:.6f} log={log}"
    )

aggregate_speedup = total_base / total_variant if total_variant > 0 else math.inf
print(
    "suite_summary "
    f"windows={len(rows)} base_ms={total_base:.6f} variant_ms={total_variant:.6f} "
    f"aggregate_speedup={aggregate_speedup:.6f} min_window_speedup={min_speedup:.6f} "
    f"min_total_speedup={threshold:.6f}"
)
if not math.isfinite(aggregate_speedup) or aggregate_speedup < threshold:
    print(
        "artifact_suite_gate decision=reject reason=aggregate_speed_below_threshold "
        f"aggregate_speedup={aggregate_speedup:.6f} min_total_speedup={threshold:.6f}"
    )
    raise SystemExit(4)
if not math.isfinite(min_speedup) or min_speedup < threshold:
    print(
        "artifact_suite_gate decision=reject reason=window_speed_below_threshold "
        f"min_window_speedup={min_speedup:.6f} min_total_speedup={threshold:.6f}"
    )
    raise SystemExit(4)
print(
    "artifact_suite_gate decision=candidate "
    f"aggregate_speedup={aggregate_speedup:.6f} min_window_speedup={min_speedup:.6f} "
    f"min_total_speedup={threshold:.6f}"
)
PY
