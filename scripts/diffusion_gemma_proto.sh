#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model="${DIFFUSION_GEMMA_MODEL:-$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
prompt="${PROMPT:-Write one concise sentence about why local verification matters.}"
mode="${MODE:-smoke}"
case "$mode" in
  smoke) default_steps=4 ;;
  fast-quality) default_steps=10 ;;
  quality) default_steps=16 ;;
  *) printf 'error: MODE must be smoke, fast-quality, or quality, got %s\n' "$mode" >&2; exit 2 ;;
esac
steps="${STEPS:-$default_steps}"
predict="${N_PREDICT:--1}"
blocks="${BLOCKS:-1}"
seed="${SEED:-123}"
ngl="${NGL:-99}"
ctx="${CTX:-512}"
ubatch="${UBATCH:-512}"
batch="${BATCH:-}"
fa="${FA:-on}"
kv_cache="${KV_CACHE:-on}"
cache_type_k="${CACHE_TYPE_K:-}"
cache_type_v="${CACHE_TYPE_V:-}"
swa_full="${SWA_FULL:-0}"
no_host="${NO_HOST:-0}"
repack="${REPACK:-auto}"
diffusion_algorithm="${DIFFUSION_ALGORITHM:-}"
eb_t_min="${EB_T_MIN:-}"
eb_t_max="${EB_T_MAX:-}"
eb_entropy_bound="${EB_ENTROPY_BOUND:-}"
eb_stability="${EB_STABILITY:-}"
eb_confidence="${EB_CONFIDENCE:-}"
raw_output="${RAW_OUTPUT:-1}"
log_dir="${LOG_DIR:-/tmp}"

find_cli() {
  if [[ -n "${LLAMA_DIFFUSION_CLI:-}" ]]; then
    printf '%s\n' "$LLAMA_DIFFUSION_CLI"
    return
  fi

  local candidates=(
    "$HOME/SrcArchives/AI/llama.cpp/build/bin/llama-diffusion-cli"
    "$HOME/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg/bin/llama-diffusion-cli"
    "$repo_root/../llama.cpp/build/bin/llama-diffusion-cli"
  )
  local cli
  for cli in "${candidates[@]}"; do
    if [[ -x "$cli" ]] && "$cli" --help 2>&1 | grep -q -- '--diffusion-eb'; then
      printf '%s\n' "$cli"
      return
    fi
  done

  printf 'error: no llama-diffusion-cli with --diffusion-eb found; set LLAMA_DIFFUSION_CLI\n' >&2
  return 2
}

cli="$(find_cli)"
mkdir -p "$log_dir"
stamp="$(date +%Y%m%d%H%M%S)"
log="$log_dir/diffusiongemma_proto_${kv_cache}_steps${steps}_${stamp}.log"

printf 'mode=%s\n' "$mode"
printf 'cli=%s\n' "$cli"
printf 'model=%s\n' "$model"
printf 'log=%s\n' "$log"
printf 'ctx=%s\n' "$ctx"
printf 'batch=%s\n' "${batch:-auto}"
printf 'ubatch=%s\n' "$ubatch"
printf 'fa=%s\n' "$fa"
printf 'kv_cache=%s\n' "$kv_cache"
printf 'cache_type_k=%s\n' "${cache_type_k:-auto}"
printf 'cache_type_v=%s\n' "${cache_type_v:-auto}"
printf 'diffusion_algorithm=%s\n' "${diffusion_algorithm:-auto}"
printf 'eb_t_min=%s\n' "${eb_t_min:-auto}"
printf 'eb_t_max=%s\n' "${eb_t_max:-auto}"
printf 'eb_entropy_bound=%s\n' "${eb_entropy_bound:-auto}"
printf 'eb_stability=%s\n' "${eb_stability:-auto}"
printf 'eb_confidence=%s\n' "${eb_confidence:-auto}"
printf 'raw_output=%s\n' "$raw_output"

if [[ ! -f "$model" ]]; then
  printf 'error: model not found: %s\n' "$model" >&2
  exit 2
fi

args=(
  -m "$model"
  -p "$prompt"
  -n "$predict"
  --diffusion-blocks "$blocks"
  --diffusion-eb on
  --diffusion-eb-max-steps "$steps"
  --diffusion-kv-cache "$kv_cache"
  -ngl "$ngl"
  -fa "$fa"
  -c "$ctx"
  -ub "$ubatch"
  -s "$seed"
)

if [[ -n "$batch" ]]; then
  args+=(-b "$batch")
fi
if [[ -n "$cache_type_k" ]]; then
  args+=(-ctk "$cache_type_k")
fi
if [[ -n "$cache_type_v" ]]; then
  args+=(-ctv "$cache_type_v")
fi
if [[ -n "$diffusion_algorithm" ]]; then
  args+=(--diffusion-algorithm "$diffusion_algorithm")
fi
if [[ -n "$eb_t_min" ]]; then
  args+=(--diffusion-eb-t-min "$eb_t_min")
fi
if [[ -n "$eb_t_max" ]]; then
  args+=(--diffusion-eb-t-max "$eb_t_max")
fi
if [[ -n "$eb_entropy_bound" ]]; then
  args+=(--diffusion-eb-entropy-bound "$eb_entropy_bound")
fi
if [[ -n "$eb_stability" ]]; then
  args+=(--diffusion-eb-stability "$eb_stability")
fi
if [[ -n "$eb_confidence" ]]; then
  args+=(--diffusion-eb-confidence "$eb_confidence")
fi
if [[ "$swa_full" == "1" ]]; then
  args+=(--swa-full)
fi
case "$no_host" in
  0) ;;
  1) args+=(--no-host) ;;
  *) printf 'error: NO_HOST must be 0 or 1, got %s\n' "$no_host" >&2; exit 2 ;;
esac
case "$repack" in
  auto|on) ;;
  off) args+=(--no-repack) ;;
  *) printf 'error: REPACK must be auto, on, or off, got %s\n' "$repack" >&2; exit 2 ;;
esac
case "$raw_output" in
  0|1) ;;
  *) printf 'error: RAW_OUTPUT must be 0 or 1, got %s\n' "$raw_output" >&2; exit 2 ;;
esac

if [[ "$raw_output" == "1" ]]; then
  /usr/bin/time -p "$cli" "${args[@]}" 2>&1 | tee "$log"
else
  /usr/bin/time -p "$cli" "${args[@]}" >"$log" 2>&1
fi

python3 - "$log" "$steps" <<'PY'
import re
import sys
path = sys.argv[1]
expected_steps = int(sys.argv[2])
text = open(path, errors="replace").read()
fatal_patterns = [
    r"ggml_metal_graph_compute: backend is in error state",
    r"llama_decode: failed",
    r"diffusion_generate_entropy_bound: failed",
    r"command buffer .* failed",
]
for pattern in fatal_patterns:
    if re.search(pattern, text):
        print(f"diffusion_gemma_proto_result runtime_error=1 pattern={pattern!r} log={path}")
        sys.exit(5)
m = re.search(r"total time: ([0-9.]+)ms, time per step: ([0-9.]+)ms \((\d+) steps", text)
if m:
    actual_steps = int(m.group(3))
    if actual_steps < expected_steps:
        print(f"diffusion_gemma_proto_result incomplete_steps=1 actual_steps={actual_steps} expected_steps={expected_steps} log={path}")
        sys.exit(6)
    print(f"diffusion_gemma_proto_result total_ms={m.group(1)} step_ms={m.group(2)} steps={m.group(3)} log={path}")
else:
    print(f"diffusion_gemma_proto_result parse_error=1 log={path}")
    sys.exit(3)
PY

answer="$(python3 "$repo_root/scripts/diffusion_gemma_extract_answer.py" "$log")"
if [[ -n "$answer" ]]; then
  printf 'diffusion_gemma_answer_status=ok\n'
  printf 'diffusion_gemma_answer=%s\n' "$answer"
else
  printf 'diffusion_gemma_answer_status=no_clean_answer\n'
fi
