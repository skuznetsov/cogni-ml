#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model="${DIFFUSION_GEMMA_MODEL:-$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
server="${LLAMA_DIFFUSION_GEMMA_SERVER:-$HOME/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg/bin/llama-diffusion-gemma-server}"
log_dir="${LOG_DIR:-/tmp}"
ngl="${NGL:-99}"
fa="${FA:-1}"
maxtok="${MAXTOK:-512}"

mkdir -p "$log_dir"
stamp="$(date +%Y%m%d%H%M%S)"
stderr_log="$log_dir/diffusiongemma_server_${stamp}.stderr.log"

printf 'server=%s\n' "$server"
printf 'model=%s\n' "$model"
printf 'stderr_log=%s\n' "$stderr_log"
printf 'ngl=%s\n' "$ngl"
printf 'fa=%s\n' "$fa"
printf 'maxtok=%s\n' "$maxtok"

if [[ ! -x "$server" ]]; then
  printf 'error: server not executable: %s\n' "$server" >&2
  exit 2
fi
if [[ ! -f "$model" ]]; then
  printf 'error: model not found: %s\n' "$model" >&2
  exit 2
fi

coproc DG_SERVER {
  NGL="$ngl" FA="$fa" MAXTOK="$maxtok" "$server" "$model" 2>"$stderr_log"
}

server_pid="$DG_SERVER_PID"
cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ready=""
if ! IFS= read -r -t "${READY_TIMEOUT:-120}" ready <&"${DG_SERVER[0]}"; then
  printf 'diffusion_gemma_server_smoke_result ready_timeout=1 stderr_log=%s\n' "$stderr_log"
  exit 3
fi

printf 'server_line=%s\n' "$ready"
if [[ ! "$ready" =~ ^READY[[:space:]][0-9]+$ ]]; then
  printf 'diffusion_gemma_server_smoke_result bad_ready=1 stderr_log=%s\n' "$stderr_log"
  exit 4
fi

printf 'QUIT\n' >&"${DG_SERVER[1]}"
exec {DG_SERVER[1]}>&-
wait "$server_pid"
trap - EXIT

printf 'diffusion_gemma_server_smoke_result ready=1 stderr_log=%s\n' "$stderr_log"
