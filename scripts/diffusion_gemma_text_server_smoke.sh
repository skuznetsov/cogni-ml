#!/usr/bin/env bash
set -euo pipefail

model="${DIFFUSION_GEMMA_MODEL:-$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
server="${LLAMA_DIFFUSION_GEMMA_SERVER:-$HOME/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg/bin/llama-diffusion-gemma-server}"
log_dir="${LOG_DIR:-/tmp}"
ngl="${NGL:-99}"
fa="${FA:-1}"
maxtok="${MAXTOK:-512}"
text_steps="${TEXT_STEPS:-10}"
seed="${SEED:-123}"
request_steps_env="${REQUEST_STEPS:-}"

prompts=(
  "Write one concise sentence about why local verification matters."
  "Write one concise sentence about why resident inference helps."
)

if [[ "$#" -gt 0 ]]; then
  prompts=("$@")
fi

mkdir -p "$log_dir"
stamp="$(date +%Y%m%d%H%M%S)"
stderr_log="$log_dir/diffusiongemma_text_server_${stamp}.stderr.log"
reply_log="$log_dir/diffusiongemma_text_server_${stamp}.replies.log"

printf 'server=%s\n' "$server"
printf 'model=%s\n' "$model"
printf 'stderr_log=%s\n' "$stderr_log"
printf 'reply_log=%s\n' "$reply_log"
printf 'text_steps=%s\n' "$text_steps"
if [[ -n "$request_steps_env" ]]; then
  printf 'request_steps=%s\n' "$request_steps_env"
fi

if [[ ! -x "$server" ]]; then
  printf 'error: server not executable: %s\n' "$server" >&2
  exit 2
fi
if [[ ! -f "$model" ]]; then
  printf 'error: model not found: %s\n' "$model" >&2
  exit 2
fi
request_steps=()
if [[ -n "$request_steps_env" ]]; then
  read -r -a request_steps <<<"$request_steps_env"
  if [[ "${#request_steps[@]}" -ne "${#prompts[@]}" ]]; then
    printf 'error: REQUEST_STEPS count (%s) must match prompt count (%s)\n' "${#request_steps[@]}" "${#prompts[@]}" >&2
    exit 2
  fi
  for step in "${request_steps[@]}"; do
    if [[ ! "$step" =~ ^[1-9][0-9]*$ ]] || (( step > 128 )); then
      printf 'error: invalid REQUEST_STEPS value: %s\n' "$step" >&2
      exit 2
    fi
  done
fi

coproc DG_TEXT_SERVER {
  NGL="$ngl" FA="$fa" MAXTOK="$maxtok" TEXT_STEPS="$text_steps" TEXT_KVCACHE=1 SEED="$seed" \
    "$server" "$model" 2>"$stderr_log"
}
server_pid="$DG_TEXT_SERVER_PID"

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ready=""
if ! IFS= read -r -t "${READY_TIMEOUT:-180}" ready <&"${DG_TEXT_SERVER[0]}"; then
  printf 'diffusion_gemma_text_server_smoke_result ready_timeout=1 stderr_log=%s\n' "$stderr_log"
  exit 3
fi
printf 'server_line=%s\n' "$ready"
if [[ ! "$ready" =~ ^READY[[:space:]][0-9]+$ ]]; then
  printf 'diffusion_gemma_text_server_smoke_result bad_ready=1 stderr_log=%s\n' "$stderr_log"
  exit 4
fi

: >"$reply_log"
for i in "${!prompts[@]}"; do
  prompt="${prompts[$i]}"
  if [[ "${#request_steps[@]}" -gt 0 ]]; then
    printf 'TEXT\tsteps=%s\t%s\n' "${request_steps[$i]}" "$prompt" >&"${DG_TEXT_SERVER[1]}"
  else
    printf 'TEXT\t%s\n' "$prompt" >&"${DG_TEXT_SERVER[1]}"
  fi
  reply=""
  if ! IFS= read -r -t "${TEXT_TIMEOUT:-180}" reply <&"${DG_TEXT_SERVER[0]}"; then
    printf 'diffusion_gemma_text_server_smoke_result text_timeout=1 stderr_log=%s\n' "$stderr_log"
    exit 5
  fi
  printf '%s\n' "$reply" | tee -a "$reply_log"
done

printf 'QUIT\n' >&"${DG_TEXT_SERVER[1]}"
exec {DG_TEXT_SERVER[1]}>&-
wait "$server_pid"
trap - EXIT

python3 - "$reply_log" "$stderr_log" <<'PY'
import pathlib
import sys

reply_path = pathlib.Path(sys.argv[1])
stderr_path = pathlib.Path(sys.argv[2])
rows = reply_path.read_text(errors="replace").splitlines()
ok = [r for r in rows if r.startswith("TEXT_OK\t")]
ready_count = stderr_path.read_text(errors="replace").count("diffusion-gemma-server ready")

print(
    "diffusion_gemma_text_server_smoke_result "
    f"text_ok={len(ok)}/{len(rows)} "
    f"server_ready_count={ready_count} "
    f"reply_log={reply_path} stderr_log={stderr_path}"
)
if len(ok) != len(rows) or ready_count != 1:
    sys.exit(6)
PY

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/diffusion_gemma_validate_text_replies.py" "$reply_log"
