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
steps_list="${STEPS_LIST:-8 10 12}"
prompt="${PROMPT:-Write one concise sentence about why local verification matters.}"
require_all_clean="${REQUIRE_ALL_CLEAN:-0}"

mkdir -p "$log_dir"
stamp="$(date +%Y%m%d%H%M%S)"
stderr_log="$log_dir/diffusiongemma_text_steps_${stamp}.stderr.log"
reply_log="$log_dir/diffusiongemma_text_steps_${stamp}.replies.log"

printf 'server=%s\n' "$server"
printf 'model=%s\n' "$model"
printf 'stderr_log=%s\n' "$stderr_log"
printf 'reply_log=%s\n' "$reply_log"
printf 'steps_list=%s\n' "$steps_list"
printf 'require_all_clean=%s\n' "$require_all_clean"

if [[ ! -x "$server" ]]; then
  printf 'error: server not executable: %s\n' "$server" >&2
  exit 2
fi
if [[ ! -f "$model" ]]; then
  printf 'error: model not found: %s\n' "$model" >&2
  exit 2
fi

read -r -a steps <<<"$steps_list"
if [[ "${#steps[@]}" -eq 0 ]]; then
  printf 'error: STEPS_LIST is empty\n' >&2
  exit 2
fi
for step in "${steps[@]}"; do
  if [[ ! "$step" =~ ^[1-9][0-9]*$ ]] || (( step > 128 )); then
    printf 'error: invalid STEPS_LIST value: %s\n' "$step" >&2
    exit 2
  fi
done

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
  printf 'diffusion_gemma_text_steps_sweep_result ready_timeout=1 stderr_log=%s\n' "$stderr_log"
  exit 3
fi
printf 'server_line=%s\n' "$ready"
if [[ ! "$ready" =~ ^READY[[:space:]][0-9]+$ ]]; then
  printf 'diffusion_gemma_text_steps_sweep_result bad_ready=1 stderr_log=%s\n' "$stderr_log"
  exit 4
fi

: >"$reply_log"
for step in "${steps[@]}"; do
  printf 'TEXT\tsteps=%s\t%s\n' "$step" "$prompt" >&"${DG_TEXT_SERVER[1]}"
  reply=""
  if ! IFS= read -r -t "${TEXT_TIMEOUT:-180}" reply <&"${DG_TEXT_SERVER[0]}"; then
    printf 'diffusion_gemma_text_steps_sweep_result text_timeout=1 step=%s stderr_log=%s\n' "$step" "$stderr_log"
    exit 5
  fi
  printf '%s\n' "$reply" | tee -a "$reply_log"
done

printf 'QUIT\n' >&"${DG_TEXT_SERVER[1]}"
exec {DG_TEXT_SERVER[1]}>&-
wait "$server_pid"
trap - EXIT

python3 - "$reply_log" "$stderr_log" "$require_all_clean" <<'PY'
import pathlib
import sys

reply_path = pathlib.Path(sys.argv[1])
stderr_path = pathlib.Path(sys.argv[2])
require_all_clean = sys.argv[3] == "1"
rows = reply_path.read_text(errors="replace").splitlines()
ready_count = stderr_path.read_text(errors="replace").count("diffusion-gemma-server ready")
ok = 0
no_clean = 0
errors = 0
for row in rows:
    parts = row.split("\t")
    tag = parts[0]
    fields = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    if tag == "TEXT_OK":
        ok += 1
        status = "ok"
    elif tag == "TEXT_NO_CLEAN":
        no_clean += 1
        status = "no_clean"
    else:
        errors += 1
        status = "error"
    print(
        "diffusion_gemma_text_steps_row "
        f"status={status} steps={fields.get('steps', '?')} total_ms={fields.get('total_ms', '?')}"
    )

print(
    "diffusion_gemma_text_steps_sweep_result "
    f"text_ok={ok}/{len(rows)} no_clean={no_clean}/{len(rows)} errors={errors} "
    f"server_ready_count={ready_count} reply_log={reply_path} stderr_log={stderr_path}"
)
if errors or ready_count != 1 or (require_all_clean and ok != len(rows)):
    sys.exit(6)
PY

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/diffusion_gemma_validate_text_replies.py" --allow-no-clean "$reply_log"
