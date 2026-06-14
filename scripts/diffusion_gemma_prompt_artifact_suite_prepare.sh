#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

default_base_env="DIFFUSION_GEMMA_C8_RESIDENT_DECODE_POLICY=1 DIFFUSION_GEMMA_PROMPT_CACHE_POLICY=1"
default_variant_env="$default_base_env DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_GRAPH=1 DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_BATCH_GRAPH_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_GATHER=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_GATHER_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_REDUCE=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_REDUCE_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_PRENORM=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_PRENORM_MAX_CANVAS=16 DIFFUSION_GEMMA_FFN_RESIDENT_SCRATCH=1 DIFFUSION_GEMMA_FFN_RESIDENT_GRAPH_CACHE=1 DIFFUSION_GEMMA_PROMPT_CONTEXT_METAL_ROWS=1 DIFFUSION_GEMMA_PROMPT_ATTENTION_RESIDUAL_CONTEXT_BUFFER=1"

model="${DIFFUSION_GEMMA_MODEL:-$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
crystal_bin="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
bridge_o="${COGNI_ML_BRIDGE_O:-}"
bin_dir="${BIN_DIR:-/tmp}"
log_dir="${LOG_DIR:-/tmp/diffusiongemma_prompt_artifact_suite_prepare_$(date +%Y%m%d%H%M%S)}"
artifact_dir="${SUITE_ARTIFACT_DIR:-$log_dir/artifacts}"

base_env="${BASE_ENV:-$default_base_env}"
variant_env="${VARIANT_ENV:-$default_variant_env}"
prompt_len="${PROMPT_LEN:-16}"
canvas_len="${CANVAS_LEN:-8}"
max_layers="${MAX_LAYERS:-30}"
token_windows="${TOKEN_WINDOWS:-}"
artifact_arms="${SUITE_ARTIFACT_ARMS:-variant}"
overwrite="${SUITE_ARTIFACT_OVERWRITE:-0}"

abba_source="$repo_root/scripts/diffusion_gemma_prompt_cache_abba.cr"
abba_bin="${ABBA_BIN:-$bin_dir/diffusion_gemma_prompt_cache_abba}"

usage() {
  cat <<'EOF'
Usage: diffusion_gemma_prompt_artifact_suite_prepare.sh

Captures v2 prompt-route artifacts for TOKEN_WINDOWS entries in one shared
model load and emits ready-to-use SUITE_*_ROUTE_ARTIFACT_MAP lines for the
suite gate.

Important environment knobs:
  TOKEN_WINDOWS                    prompt:canvas windows to capture
  SUITE_ARTIFACT_ARMS=variant      variant, base, or base,variant
  SUITE_ARTIFACT_DIR=PATH          artifact output directory
  SUITE_ARTIFACT_OVERWRITE=1       overwrite existing artifacts

The usual DIFFUSION_GEMMA_MODEL, PROMPT_LEN, CANVAS_LEN, MAX_LAYERS,
BASE_ENV, VARIANT_ENV, BIN_DIR, CRYSTAL_BIN, and COGNI_ML_BRIDGE_O knobs are
honored. The canvas token is part of the emitted map key; route capture itself
is bound to the prompt token, prompt length, arm env, and model fingerprint.
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

build_abba() {
  if [[ -x "$abba_bin" && "$abba_source" -ot "$abba_bin" && "${FORCE_BUILD:-0}" != "1" ]]; then
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
  mkdir -p "$(dirname "$abba_bin")"
  "$crystal_bin" build --release "$abba_source" \
    -o "$abba_bin" \
    --error-trace \
    --link-flags="$bridge_o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
}

validate_positive_uint PROMPT_LEN "$prompt_len"
validate_positive_uint CANVAS_LEN "$canvas_len"
validate_positive_uint MAX_LAYERS "$max_layers"
validate_env_tokens BASE_ENV "$base_env"
validate_env_tokens VARIANT_ENV "$variant_env"
[[ -n "$token_windows" ]] || die "TOKEN_WINDOWS is required"
[[ -f "$model" ]] || die "model not found: $model"
bool_enabled "$overwrite" >/dev/null || true

case ",${artifact_arms// /}," in
  *,base,*|*,variant,*) ;;
  *) die "SUITE_ARTIFACT_ARMS must include base, variant, or both" ;;
esac
if [[ "$artifact_arms" != "base" && "$artifact_arms" != "variant" && "$artifact_arms" != "base,variant" && "$artifact_arms" != "variant,base" ]]; then
  die "SUITE_ARTIFACT_ARMS must be variant, base, base,variant, or variant,base"
fi

mkdir -p "$log_dir" "$artifact_dir"
build_abba

suite_spec="$log_dir/suite_windows.tsv"
python3 - "$token_windows" >"$suite_spec" <<'PY'
import sys

windows = []
for entry in sys.argv[1].replace(",", " ").split():
    try:
        prompt, canvas = entry.split(":", 1)
    except ValueError:
        raise SystemExit(f"TOKEN_WINDOWS entry must be prompt:canvas, got {entry!r}")
    prompt = int(prompt)
    canvas = int(canvas)
    if prompt < 0 or canvas < 0:
        raise SystemExit("TOKEN_WINDOWS prompt and canvas tokens must be non-negative")
    windows.append((prompt, canvas))
if not windows:
    raise SystemExit("TOKEN_WINDOWS must contain at least one entry")
if len(set(windows)) != len(windows):
    raise SystemExit("TOKEN_WINDOWS contains duplicate entries")
for index, (prompt, canvas) in enumerate(windows):
    print(f"{index}\t{prompt}\t{canvas}")
PY

printf 'log_dir=%s\n' "$log_dir"
printf 'artifact_dir=%s\n' "$artifact_dir"
printf 'token_windows=%s\n' "$token_windows"
printf 'artifact_arms=%s\n' "$artifact_arms"
printf 'suite_spec=%s\n' "$suite_spec"

base_map_parts=()
variant_map_parts=()
while IFS=$'\t' read -r index prompt canvas; do
  window_key="${prompt}:${canvas}"
  base_artifact="$artifact_dir/base_p${prompt}_c${canvas}_pl${prompt_len}_l${max_layers}.tsv"
  variant_artifact="$artifact_dir/variant_p${prompt}_c${canvas}_pl${prompt_len}_l${max_layers}.tsv"
  if [[ ",$artifact_arms," == *",base,"* ]]; then
    if [[ -e "$base_artifact" ]] && ! bool_enabled "$overwrite"; then
      die "artifact already exists: $base_artifact; set SUITE_ARTIFACT_OVERWRITE=1"
    fi
    base_map_parts+=("${window_key}=${base_artifact}")
  fi
  if [[ ",$artifact_arms," == *",variant,"* ]]; then
    if [[ -e "$variant_artifact" ]] && ! bool_enabled "$overwrite"; then
      die "artifact already exists: $variant_artifact; set SUITE_ARTIFACT_OVERWRITE=1"
    fi
    variant_map_parts+=("${window_key}=${variant_artifact}")
  fi
  printf 'suite_artifact_target index=%s prompt_token=%s canvas_token=%s base=%s variant=%s\n' \
    "$index" "$prompt" "$canvas" \
    "$(if [[ ",$artifact_arms," == *",base,"* ]]; then printf '%s' "$base_artifact"; else printf '<none>'; fi)" \
    "$(if [[ ",$artifact_arms," == *",variant,"* ]]; then printf '%s' "$variant_artifact"; else printf '<none>'; fi)"
done <"$suite_spec"

args=(
  --model "$model"
  --prompt-len "$prompt_len"
  --canvas-len "$canvas_len"
  --max-layers "$max_layers"
  --base-env "$base_env"
  --variant-env "$variant_env"
  --full-routes
  --capture-route-artifacts-only
)

if ((${#base_map_parts[@]} > 0)); then
  base_map="$(IFS=,; printf '%s' "${base_map_parts[*]}")"
  args+=(--write-base-route-artifact-map "$base_map")
fi
if ((${#variant_map_parts[@]} > 0)); then
  variant_map="$(IFS=,; printf '%s' "${variant_map_parts[*]}")"
  args+=(--write-variant-route-artifact-map "$variant_map")
fi

capture_out="$log_dir/artifact_capture.tsv"
capture_err="$log_dir/artifact_capture.stderr"
"$abba_bin" "${args[@]}" >"$capture_out" 2>"$capture_err"
printf 'suite_artifact_capture log=%s stderr=%s\n' "$capture_out" "$capture_err"

if ((${#base_map_parts[@]} > 0)); then
  printf 'SUITE_BASE_ROUTE_ARTIFACT_MAP=%s\n' "$base_map"
  printf 'CERT_BASE_ROUTE_ARTIFACT_MAP=%s\n' "$base_map"
fi
if ((${#variant_map_parts[@]} > 0)); then
  printf 'SUITE_VARIANT_ROUTE_ARTIFACT_MAP=%s\n' "$variant_map"
  printf 'CERT_VARIANT_ROUTE_ARTIFACT_MAP=%s\n' "$variant_map"
fi
