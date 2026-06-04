#!/usr/bin/env bash
set -euo pipefail

# Sequential CogniGemma row-prefill A/B runner.
# Default mode intentionally sets GEMMA4_ROW_PREFILL_ALLOW_GEMM=1 because the
# exact fallback clamps chunks to <=8 and is not comparable to prior pp results.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

profile_bin="${GEMMA4_PREFILL_AB_BIN:-/tmp/gemma4_metal_decode_profile_ab}"
crystal_bin="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
crystal_cache="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_gemma4_prefill_ab_build}"
link_flags="${GEMMA4_PREFILL_AB_LINK_FLAGS:-${repo_root}/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++}"
run_safe="${GEMMA4_PREFILL_AB_RUN_SAFE:-${repo_root}/scripts/run_safe.sh}"
build_flags="${GEMMA4_PREFILL_AB_BUILD_FLAGS:-}"
force_build="${GEMMA4_PREFILL_AB_FORCE_BUILD:-0}"

pps="${GEMMA4_PREFILL_AB_PPS:-64,256,1024}"
runs="${GEMMA4_PREFILL_AB_RUNS:-3}"
warmups="${GEMMA4_PREFILL_AB_WARMUPS:-1}"
timeout_sec="${GEMMA4_PREFILL_AB_TIMEOUT:-420}"
max_mem_mb="${GEMMA4_PREFILL_AB_MAX_MEM_MB:-8192}"
min_free_pct="${COGNI_RUN_SAFE_MIN_FREE_PCT:-12}"
log_dir="${GEMMA4_PREFILL_AB_LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/gemma4-prefill-ab.XXXXXX")}"
modes="${GEMMA4_PREFILL_AB_MODES:-default,q4pair}"
allow_busy="${GEMMA4_PREFILL_AB_ALLOW_BUSY:-0}"
busy_pattern="${GEMMA4_PREFILL_AB_BUSY_PATTERN:-crystal spec|/tmp/adamas|regression_tests/run_combined|gemma4_metal_decode_profile|benchmark_qwen|qwen35}"

usage() {
  cat <<'EOF'
usage: scripts/gemma4_prefill_ab.sh

Environment:
  GEMMA4_PREFILL_AB_PPS=64,256,1024       prompt lengths to test
  GEMMA4_PREFILL_AB_MODES=default,q4pair  modes: default,q4pair,fullpair,allpair
  GEMMA4_PREFILL_AB_RUNS=3                measured runs per row
  GEMMA4_PREFILL_AB_WARMUPS=1             warmup runs per row
  GEMMA4_PREFILL_AB_FORCE_BUILD=1         rebuild profile binary
  GEMMA4_PREFILL_AB_BIN=/tmp/...          profile binary path
  GEMMA4_PREFILL_AB_LOG_DIR=/tmp/...      keep logs in a chosen directory
  GEMMA4_PREFILL_AB_ALLOW_BUSY=1          skip other-heavy-process preflight
  COGNI_RUN_SAFE_MIN_FREE_PCT=12          system memory-pressure kill threshold

Notes:
  - Runs are sequential by design. Do not wrap this script in parallel runners.
  - By default, it refuses to run beside obvious Crystal/Metal/Adamas jobs.
  - GEMMA4_ROW_PREFILL_ALLOW_GEMM=1 is forced for comparable high-throughput pp.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

check_busy_host() {
  local self_pid="$$"
  local matches
  matches="$(
    ps -axo pid=,rss=,args= |
      awk -v self="${self_pid}" '$1 != self {print}' |
      grep -E "${busy_pattern}" |
      grep -v -E "grep -E|scripts/gemma4_prefill_ab.sh|${profile_bin//\//\\/}$" || true
  )"
  if [[ -n "${matches}" ]]; then
    echo "refusing to run Gemma prefill A/B while other heavy jobs are active:" >&2
    echo "${matches}" >&2
    echo "Set GEMMA4_PREFILL_AB_ALLOW_BUSY=1 to override after checking memory pressure." >&2
    exit 3
  fi
}

if [[ "${allow_busy}" != "1" ]]; then
  check_busy_host
fi

if [[ ! -f "${repo_root}/build/bridge.o" && "${link_flags}" == *"build/bridge.o"* ]]; then
  echo "missing ${repo_root}/build/bridge.o; build bridge.o first or set GEMMA4_PREFILL_AB_LINK_FLAGS" >&2
  exit 2
fi

if [[ "${force_build}" == "1" || ! -x "${profile_bin}" ]]; then
  echo "building ${profile_bin}" >&2
  (
    cd "${repo_root}"
    CRYSTAL_CACHE_DIR="${crystal_cache}" COGNI_RUN_SAFE_MIN_FREE_PCT="${min_free_pct}" \
      "${run_safe}" "${crystal_bin}" 240 "${max_mem_mb}" \
      build bin/gemma4_metal_decode_profile.cr ${build_flags} \
      -o "${profile_bin}" --error-trace --link-flags="${link_flags}"
  )
fi

mkdir -p "${log_dir}"

tokens_for_pp() {
  python3 - "$1" <<'PY'
import sys
n=int(sys.argv[1])
print(",".join(str(i) for i in range(42, 42+n)))
PY
}

env_for_mode() {
  case "$1" in
    default) printf '%s\n' ;;
    q4pair) printf '%s\n' "GEMMA4_ROW_PREFILL_Q4_PAIR_FFN=1" ;;
    fullpair) printf '%s\n' "GEMMA4_ROW_PREFILL_ATTN_GQA_PAIR_FULL=1" ;;
    allpair)
      printf '%s\n' "GEMMA4_ROW_PREFILL_Q4_PAIR_FFN=1"
      printf '%s\n' "GEMMA4_ROW_PREFILL_ATTN_GQA_PAIR_FULL=1"
      ;;
    *)
      echo "unknown mode: $1" >&2
      exit 2
      ;;
  esac
}

parse_log() {
  python3 - "$1" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
def grab(pattern, default=""):
    m = re.search(pattern, text)
    return m.group(1) if m else default
print("\t".join([
    grab(r"prefill_p50_ms=([0-9.]+)", "NA"),
    grab(r"prefill_p50_tok_s=([0-9.]+)", "NA"),
    grab(r"decode_ms_per_token_p50=([0-9.]+)", "NA"),
]))
PY
}

printf "pp\tmode\tprefill_p50_ms\tprefill_tok_s\tdecode_ms_per_tok\tlog\n"
IFS=',' read -r -a pp_items <<<"${pps}"
IFS=',' read -r -a mode_items <<<"${modes}"

for pp in "${pp_items[@]}"; do
  pp="${pp//[[:space:]]/}"
  [[ -z "${pp}" ]] && continue
  tokens="$(tokens_for_pp "${pp}")"
  max_seq=$((pp * 2))
  for mode in "${mode_items[@]}"; do
    mode="${mode//[[:space:]]/}"
    [[ -z "${mode}" ]] && continue
    log="${log_dir}/pp${pp}_${mode}.log"
    echo "RUN pp=${pp} mode=${mode} log=${log}" >&2
    env_args=(
      COGNI_RUN_SAFE_MIN_FREE_PCT="${min_free_pct}"
      GEMMA4_ROW_PREFILL_ALLOW_GEMM=1
    )
    while IFS= read -r kv; do
      [[ -n "${kv}" ]] && env_args+=("${kv}")
    done < <(env_for_mode "${mode}")
    env "${env_args[@]}" "${run_safe}" "${profile_bin}" "${timeout_sec}" "${max_mem_mb}" \
      --tokens "${tokens}" \
      --prefill-mode rows \
      --prefill-chunk "${pp}" \
      --generate 1 \
      --body-only \
      --prefill-no-head \
      --runs "${runs}" \
      --warmups "${warmups}" \
      --max-seq "${max_seq}" >"${log}" 2>&1
    metrics="$(parse_log "${log}")"
    printf "%s\t%s\t%s\t%s\n" "${pp}" "${mode}" "${metrics}" "${log}"
  done
done

echo "log_dir=${log_dir}" >&2
