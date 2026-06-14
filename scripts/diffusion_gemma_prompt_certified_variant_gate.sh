#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

default_base_env="DIFFUSION_GEMMA_C8_RESIDENT_DECODE_POLICY=1 DIFFUSION_GEMMA_PROMPT_CACHE_POLICY=1"
default_variant_env="$default_base_env DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_GRAPH=1 DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_BATCH_GRAPH_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_GATHER=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_GATHER_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_REDUCE=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_REDUCE_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_PRENORM=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_PRENORM_MAX_CANVAS=16 DIFFUSION_GEMMA_FFN_RESIDENT_SCRATCH=1 DIFFUSION_GEMMA_FFN_RESIDENT_GRAPH_CACHE=1 DIFFUSION_GEMMA_PROMPT_CONTEXT_METAL_ROWS=1 DIFFUSION_GEMMA_PROMPT_ATTENTION_RESIDUAL_CONTEXT_BUFFER=1"

model="${DIFFUSION_GEMMA_MODEL:-$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
crystal_bin="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
bridge_o="${COGNI_ML_BRIDGE_O:-}"
bin_dir="${BIN_DIR:-/tmp}"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_prompt_certified_variant_gate_$(date +%Y%m%d%H%M%S)}"

base_env="${BASE_ENV:-$default_base_env}"
variant_env="${VARIANT_ENV:-$default_variant_env}"
prompt_len="${PROMPT_LEN:-16}"
canvas_len="${CANVAS_LEN:-8}"
max_layers="${MAX_LAYERS:-30}"
token_windows="${TOKEN_WINDOWS:-1:0,17:100,257:1000,4096:8192}"
certificate_mode="${CERTIFICATE_MODE:-bounded}"
candidate_count="${CANDIDATE_COUNT:-1024}"
candidate_offsets="${CANDIDATE_OFFSETS:-0,1024,8192,32768,65536,131072}"
candidate_stride="${CANDIDATE_STRIDE:-1}"
max_candidate_row_size="${MAX_CANDIDATE_ROW_SIZE:-8192}"

cert_require_argmax="${CERT_REQUIRE_ARGMAX:-1}"
cert_require_sampled="${CERT_REQUIRE_SAMPLED:-0}"
min_base_logit_margin="${MIN_BASE_LOGIT_MARGIN:-}"
min_variant_logit_margin="${MIN_VARIANT_LOGIT_MARGIN:-}"
max_logit_delta="${MAX_LOGIT_DELTA:-}"

abba_warmups="${ABBA_WARMUPS:-1}"
abba_repeats="${ABBA_REPEATS:-3}"
abba_trim_per_arm="${ABBA_TRIM_PER_ARM:-1}"
abba_sequence="${ABBA_SEQUENCE:-base variant variant base}"
abba_mirror_sequence="${ABBA_MIRROR_SEQUENCE:-variant base base variant}"
abba_route_capture_amortize_uses="${ABBA_ROUTE_CAPTURE_AMORTIZE_USES:-1}"
abba_use_effective_total="${ABBA_USE_EFFECTIVE_TOTAL:-1}"
min_total_speedup="${MIN_TOTAL_SPEEDUP:-1.10}"
run_abba_on_cert_fail="${RUN_ABBA_ON_CERT_FAIL:-0}"
checksum_tolerance="${CHECKSUM_TOLERANCE:-}"
check_quiet="${CHECK_QUIET:-0}"

cert_source="$repo_root/scripts/diffusion_gemma_prompt_output_cert_probe.cr"
abba_source="$repo_root/scripts/diffusion_gemma_prompt_cache_abba.cr"
cert_bin="${CERT_BIN:-$bin_dir/diffusion_gemma_prompt_output_cert_probe}"
abba_bin="${ABBA_BIN:-$bin_dir/diffusion_gemma_prompt_cache_abba}"

usage() {
  cat <<'EOF'
Usage: diffusion_gemma_prompt_certified_variant_gate.sh

Runs a fail-closed promotion gate for the approximate prompt-MoE route:
  1. output certificate over TOKEN_WINDOWS,
  2. mirrored prompt-cache ABBA speed measurement,
  3. single decision row.

Important environment knobs:
  BASE_ENV / VARIANT_ENV          arm envs passed to both probes
  PROMPT_LEN / CANVAS_LEN         synthetic shape, defaults 16 / 8
  MAX_LAYERS                      full-depth default 30
  TOKEN_WINDOWS                   prompt:canvas starts, default four windows
  CERTIFICATE_MODE                bounded, full-vocab-top1-metal/cpu, full-vocab-top2-metal/cpu
  CANDIDATE_COUNT                 candidate ids per span, default 1024
  CANDIDATE_OFFSETS               span offsets from each canvas token, default six bands
  CANDIDATE_STRIDE                token stride inside each span, default 1
  MAX_CANDIDATE_ROW_SIZE          merged per-row cap, default 8192
  CERT_REQUIRE_SAMPLED=1          require deterministic sampled-token stability
  MIN_BASE_LOGIT_MARGIN=F         optional certificate margin floor
  MIN_VARIANT_LOGIT_MARGIN=F      optional certificate margin floor
  MAX_LOGIT_DELTA=F               optional candidate-logit delta ceiling
  MIN_TOTAL_SPEEDUP=F             ABBA total_ms speedup floor, default 1.10
  ABBA_BASE_REPLAY_ROUTES=1       pre-capture/replay base prompt MoE routes in ABBA
  ABBA_VARIANT_REPLAY_ROUTES=1    pre-capture/replay variant prompt MoE routes in ABBA
  ABBA_ROUTE_CAPTURE_AMORTIZE_USES=N
                                    charge route capture over N replayed prompt-cache uses, default 1
  ABBA_USE_EFFECTIVE_TOTAL=1       when replay is enabled, gate speed on capture-amortized total
  RUN_ABBA_ON_CERT_FAIL=1         run ABBA even when certificate rejects
  CHECK_QUIET=1                   poll the existing quiet gate before running
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

reject() {
  local reason="$1"
  shift || true
  printf 'certified_variant_gate decision=reject reason=%s' "$reason"
  while (($#)); do
    printf ' %s' "$1"
    shift
  done
  printf '\n'
  exit 4
}

bool_enabled() {
  case "$1" in
    1|true|yes|on) return 0 ;;
    0|false|no|off|"") return 1 ;;
    *) die "invalid boolean value: $1" ;;
  esac
}

validate_uint() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    die "$name must be a non-negative integer"
  fi
}

validate_positive_uint() {
  local name="$1"
  local value="$2"
  validate_uint "$name" "$value"
  if [[ "$value" -lt 1 ]]; then
    die "$name must be positive"
  fi
}

validate_env_tokens() {
  local label="$1"
  local raw="$2"
  local token
  for token in $raw; do
    if [[ ! "$token" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      die "$label env token must be KEY=VALUE, got $token"
    fi
  done
}

build_probe() {
  local source="$1"
  local bin="$2"
  if [[ -x "$bin" && "$source" -ot "$bin" && "${FORCE_BUILD:-0}" != "1" ]]; then
    return
  fi

  if [[ -z "$bridge_o" ]]; then
    if [[ -f "$repo_root/build/bridge.o" ]]; then
      bridge_o="$repo_root/build/bridge.o"
    else
      bridge_o="/Users/sergey/Projects/Crystal/cogni-ml/build/bridge.o"
    fi
  fi
  [[ -f "$bridge_o" ]] || die "bridge object not found: $bridge_o"
  mkdir -p "$(dirname "$bin")"
  "$crystal_bin" build --release "$source" \
    -o "$bin" \
    --error-trace \
    --link-flags="$bridge_o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
}

extract_metric() {
  local output="$1"
  local metric="$2"
  local kind="$3"
  awk -F'\t' -v metric="$metric" -v kind="$kind" '
    $1 == kind && $2 == metric {
      print $3 "\t" $4 "\t" $5 "\t" $6 "\t" $8
      found = 1
    }
    END { if (!found) exit 1 }
  ' "$output"
}

print_metric_row() {
  local output="$1"
  local metric="$2"
  local kind="$3"
  local values
  if ! values="$(extract_metric "$output" "$metric" "$kind")"; then
    printf 'gate_metric metric=%s status=missing\n' "$metric"
    return 1
  fi
  local base_ms variant_ms speedup delta range_over_delta
  IFS=$'\t' read -r base_ms variant_ms speedup delta range_over_delta <<<"$values"
  printf 'gate_metric kind=%s metric=%s base_ms=%s variant_ms=%s speedup=%s delta_ms=%s range_over_delta=%s\n' \
    "$kind" "$metric" "$base_ms" "$variant_ms" "$speedup" "$delta" "$range_over_delta"
}

float_ge() {
  python3 - "$1" "$2" <<'PY'
import math
import sys

value = float(sys.argv[1])
threshold = float(sys.argv[2])
sys.exit(0 if math.isfinite(value) and value >= threshold else 1)
PY
}

validate_positive_uint PROMPT_LEN "$prompt_len"
validate_positive_uint CANVAS_LEN "$canvas_len"
validate_positive_uint MAX_LAYERS "$max_layers"
validate_positive_uint CANDIDATE_COUNT "$candidate_count"
validate_positive_uint CANDIDATE_STRIDE "$candidate_stride"
validate_uint MAX_CANDIDATE_ROW_SIZE "$max_candidate_row_size"
validate_uint ABBA_WARMUPS "$abba_warmups"
validate_positive_uint ABBA_REPEATS "$abba_repeats"
validate_uint ABBA_TRIM_PER_ARM "$abba_trim_per_arm"
validate_positive_uint ABBA_ROUTE_CAPTURE_AMORTIZE_USES "$abba_route_capture_amortize_uses"
case "$certificate_mode" in
  bounded|full-vocab-top1-metal|full-vocab-top1-cpu|full-vocab-top2-metal|full-vocab-top2-cpu) ;;
  *) die "CERTIFICATE_MODE must be bounded, full-vocab-top1-metal/cpu, or full-vocab-top2-metal/cpu" ;;
esac
validate_env_tokens BASE_ENV "$base_env"
validate_env_tokens VARIANT_ENV "$variant_env"
[[ -f "$model" ]] || die "model not found: $model"

mkdir -p "$log_dir"

printf 'log_dir=%s\n' "$log_dir"
printf 'model=%s\n' "$model"
printf 'prompt_len=%s\n' "$prompt_len"
printf 'canvas_len=%s\n' "$canvas_len"
printf 'max_layers=%s\n' "$max_layers"
printf 'token_windows=%s\n' "$token_windows"
printf 'certificate_mode=%s\n' "$certificate_mode"
printf 'candidate_count=%s\n' "$candidate_count"
printf 'candidate_offsets=%s\n' "$candidate_offsets"
printf 'candidate_stride=%s\n' "$candidate_stride"
printf 'max_candidate_row_size=%s\n' "$max_candidate_row_size"
printf 'base_env=%s\n' "$base_env"
printf 'variant_env=%s\n' "$variant_env"
printf 'min_total_speedup=%s\n' "$min_total_speedup"

if bool_enabled "$check_quiet"; then
  quiet_log="$log_dir/quiet_gate.log"
  if ! "$repo_root/scripts/diffusion_gemma_quiet_gate_check.sh" >"$quiet_log" 2>&1; then
    printf 'quiet_gate status=fail log=%s\n' "$quiet_log"
    reject "quiet_gate_failed" "log=$quiet_log"
  fi
  printf 'quiet_gate status=ok log=%s\n' "$quiet_log"
fi

build_probe "$cert_source" "$cert_bin"
build_probe "$abba_source" "$abba_bin"

cert_out="$log_dir/output_cert.tsv"
cert_err="$log_dir/output_cert.stderr"
cert_args=(
  --model "$model"
  --prompt-len "$prompt_len"
  --canvas-len "$canvas_len"
  --max-layers "$max_layers"
  --certificate-mode "$certificate_mode"
  --candidate-count "$candidate_count"
  --candidate-offsets "$candidate_offsets"
  --candidate-stride "$candidate_stride"
  --max-candidate-row-size "$max_candidate_row_size"
  --token-windows "$token_windows"
  --base-env "$base_env"
  --variant-env "$variant_env"
)
if bool_enabled "$cert_require_argmax"; then
  cert_args+=(--require-argmax-match)
fi
if bool_enabled "$cert_require_sampled"; then
  cert_args+=(--require-sampled-match)
fi
if [[ -n "$min_base_logit_margin" ]]; then
  cert_args+=(--min-base-logit-margin "$min_base_logit_margin")
fi
if [[ -n "$min_variant_logit_margin" ]]; then
  cert_args+=(--min-variant-logit-margin "$min_variant_logit_margin")
fi
if [[ -n "$max_logit_delta" ]]; then
  cert_args+=(--max-logit-delta "$max_logit_delta")
fi

set +e
"$cert_bin" "${cert_args[@]}" >"$cert_out" 2>"$cert_err"
cert_rc=$?
set -e

cert_status="$(grep -E '^output_cert ' "$cert_err" | tail -1 || true)"
cert_aggregate="$(grep -E '^aggregate_summary' "$cert_out" | tail -1 || true)"
printf 'output_cert_rc=%s\n' "$cert_rc"
printf 'output_cert_log=%s\n' "$cert_out"
printf 'output_cert_stderr=%s\n' "$cert_err"
[[ -n "$cert_status" ]] && printf '%s\n' "$cert_status"
[[ -n "$cert_aggregate" ]] && printf '%s\n' "$cert_aggregate"

if (( cert_rc != 0 )) && ! bool_enabled "$run_abba_on_cert_fail"; then
  reject "certificate_failed" "cert_rc=$cert_rc" "cert_log=$cert_out" "cert_stderr=$cert_err"
fi

abba_out="$log_dir/prompt_cache_abba.tsv"
abba_err="$log_dir/prompt_cache_abba.stderr"
abba_args=(
  --model "$model"
  --prompt-len "$prompt_len"
  --canvas-len "$canvas_len"
  --prompt-token 1
  --max-layers "$max_layers"
  --warmups "$abba_warmups"
  --repeats "$abba_repeats"
  --trim-per-arm "$abba_trim_per_arm"
  --sequence "$abba_sequence"
  --mirror-sequence "$abba_mirror_sequence"
  --route-capture-amortize-uses "$abba_route_capture_amortize_uses"
  --base-env "$base_env"
  --variant-env "$variant_env"
  --full-routes
  --materialize-final-rows
)
if [[ -n "$checksum_tolerance" ]]; then
  abba_args+=(--checksum-tolerance "$checksum_tolerance")
fi
if bool_enabled "${ABBA_BASE_REPLAY_ROUTES:-0}"; then
  abba_args+=(--base-replay-routes)
fi
if bool_enabled "${ABBA_VARIANT_REPLAY_ROUTES:-0}"; then
  abba_args+=(--variant-replay-routes)
fi

set +e
"$abba_bin" "${abba_args[@]}" >"$abba_out" 2>"$abba_err"
abba_rc=$?
set -e

printf 'prompt_cache_abba_rc=%s\n' "$abba_rc"
printf 'prompt_cache_abba_log=%s\n' "$abba_out"
printf 'prompt_cache_abba_stderr=%s\n' "$abba_err"
if (( abba_rc != 0 )); then
  reject "abba_failed" "abba_rc=$abba_rc" "abba_log=$abba_out" "abba_stderr=$abba_err"
fi

summary_kind="summary"
if [[ "$abba_trim_per_arm" -gt 0 ]]; then
  summary_kind="trimmed_summary"
fi
total_summary_kind="$summary_kind"
route_replay_enabled=0
if bool_enabled "${ABBA_BASE_REPLAY_ROUTES:-0}"; then
  route_replay_enabled=1
fi
if bool_enabled "${ABBA_VARIANT_REPLAY_ROUTES:-0}"; then
  route_replay_enabled=1
fi
if [[ "$route_replay_enabled" == "1" ]] && bool_enabled "$abba_use_effective_total"; then
  if [[ "$abba_trim_per_arm" -gt 0 ]]; then
    total_summary_kind="effective_trimmed_summary"
  else
    total_summary_kind="effective_summary"
  fi
fi

print_metric_row "$abba_out" total_ms "$summary_kind"
if [[ "$total_summary_kind" != "$summary_kind" ]]; then
  print_metric_row "$abba_out" total_ms "$total_summary_kind" || reject "missing_effective_total" "abba_log=$abba_out"
fi
print_metric_row "$abba_out" materialize_ms "$summary_kind" || true
print_metric_row "$abba_out" materialize_moe_ffn_ms "$summary_kind" || true
print_metric_row "$abba_out" materialize_moe_grouped_gate_up_ms "$summary_kind" || true
print_metric_row "$abba_out" materialize_moe_grouped_down_ms "$summary_kind" || true
grep -E '^checksum_summary' "$abba_out" | tail -1 || true

if (( cert_rc != 0 )); then
  reject "certificate_failed_after_abba" "cert_rc=$cert_rc" "cert_log=$cert_out" "cert_stderr=$cert_err" "abba_log=$abba_out"
fi

total_values="$(extract_metric "$abba_out" total_ms "$total_summary_kind")" || reject "missing_total_speedup" "summary_kind=$total_summary_kind" "abba_log=$abba_out"
IFS=$'\t' read -r total_base_ms total_variant_ms total_speedup total_delta_ms total_range_over_delta <<<"$total_values"
if ! float_ge "$total_speedup" "$min_total_speedup"; then
  reject "speed_below_threshold" \
    "total_speedup=$total_speedup" \
    "min_total_speedup=$min_total_speedup" \
    "base_ms=$total_base_ms" \
    "variant_ms=$total_variant_ms" \
    "total_summary_kind=$total_summary_kind" \
    "route_capture_amortize_uses=$abba_route_capture_amortize_uses" \
    "abba_log=$abba_out"
fi

decision="candidate_argmax_only"
if bool_enabled "$cert_require_sampled"; then
  decision="candidate_sampled"
fi
if [[ "$certificate_mode" == full-vocab-top1-* ]]; then
  decision="candidate_full_vocab_argmax_only"
elif [[ "$certificate_mode" == full-vocab-top2-* ]]; then
  decision="candidate_full_vocab_margin_argmax_only"
fi
printf 'certified_variant_gate decision=%s total_speedup=%s min_total_speedup=%s base_ms=%s variant_ms=%s total_summary_kind=%s route_capture_amortize_uses=%s cert_rc=%s abba_rc=%s log_dir=%s\n' \
  "$decision" "$total_speedup" "$min_total_speedup" "$total_base_ms" "$total_variant_ms" "$total_summary_kind" "$abba_route_capture_amortize_uses" "$cert_rc" "$abba_rc" "$log_dir"
