#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

model="${DIFFUSION_GEMMA_MODEL:-$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
tokenizer="${DIFFUSION_GEMMA_LLAMA_TOKENIZE:-$HOME/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg/bin/llama-tokenize}"
crystal_bin="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
bin="${DIFFUSION_GEMMA_SPARSE_LOOP_BIN:-/tmp/diffusion_gemma_sparse_loop_perf}"
bridge_o="${COGNI_ML_BRIDGE_O:-}"
timeout_bin="${TIMEOUT_BIN:-$(command -v timeout || command -v gtimeout || true)}"

canvas="${CANVAS:-Hello}"
candidates="${CANDIDATES:-Hello|world}"
steps="${STEPS:-1}"
warmups="${WARMUPS:-1}"
repeats="${REPEATS:-2}"
cache_warmups="${CACHE_WARMUPS:-0}"
cache_repeats="${CACHE_REPEATS:-1}"
max_layers="${MAX_LAYERS:-1}"
timeout_seconds="${TIMEOUT_SECONDS:-60}"
materialize_prompt_final_rows="${MATERIALIZE_PROMPT_FINAL_ROWS:-0}"
include_large_prompt="${INCLUDE_LARGE_PROMPT:-0}"
out="${OUT:-${TMPDIR:-/tmp}/diffusion_gemma_prompt_perf_$(date +%Y%m%d%H%M%S).tsv}"

short_prompt="${SHORT_PROMPT:-Say:}"
medium_prompt="${MEDIUM_PROMPT:-You are testing the native DiffusionGemma bounded sparse denoising prototype. Pick the best one-token continuation from the supplied candidates.}"
long_prompt="${LONG_PROMPT:-You are testing the native DiffusionGemma bounded sparse denoising prototype with a larger prompt. The prompt is intentionally longer than the default smoke so that prompt-cache construction, attention cache reuse, and sparse canvas update timings can be separated. We want a practical local engineering signal, not a quality claim.}"
large_prompt="${LARGE_PROMPT:-You are testing the native DiffusionGemma bounded sparse denoising prototype with a substantially larger prompt for local performance attribution. This case repeats the same engineering constraints in plain text: preserve exact prompt-cache semantics, measure prompt projection subphases, avoid product-quality claims, and keep the sparse canvas row small enough for safe local runs. The goal is to expose scaling behavior in prompt-cache construction after the short, medium, and long smoke cases no longer show a single dominant bucket. The benchmark should identify whether prompt projection matmul, per-row normalization, projection assembly, RoPE application, materialization, or decode context becomes dominant as the prompt grows. This is a bounded probe on one layer and one canvas row, not a full generation benchmark.}"

if [[ -z "$timeout_bin" ]]; then
  echo "timeout command not found; set TIMEOUT_BIN=/path/to/timeout" >&2
  exit 2
fi

for path_var in model tokenizer; do
  path="${!path_var}"
  if [[ ! -f "$path" ]]; then
    echo "${path_var} not found: $path" >&2
    exit 2
  fi
done

for numeric in steps warmups repeats cache_warmups cache_repeats max_layers timeout_seconds; do
  value="${!numeric}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "${numeric^^} must be a non-negative integer" >&2
    exit 2
  fi
done

if [[ "$steps" -lt 1 || "$repeats" -lt 1 || "$cache_repeats" -lt 1 || "$max_layers" -lt 1 || "$timeout_seconds" -lt 1 ]]; then
  echo "STEPS, REPEATS, CACHE_REPEATS, MAX_LAYERS, and TIMEOUT_SECONDS must be positive" >&2
  exit 2
fi

case "$materialize_prompt_final_rows" in
  0|false|no) materialize_prompt_final_rows=0 ;;
  1|true|yes) materialize_prompt_final_rows=1 ;;
  *)
    echo "MATERIALIZE_PROMPT_FINAL_ROWS must be 0 or 1" >&2
    exit 2
    ;;
esac

case "$include_large_prompt" in
  0|false|no) include_large_prompt=0 ;;
  1|true|yes) include_large_prompt=1 ;;
  *)
    echo "INCLUDE_LARGE_PROMPT must be 0 or 1" >&2
    exit 2
    ;;
esac

if [[ ! -x "$bin" ]]; then
  if [[ -z "$bridge_o" ]]; then
    if [[ -f "$repo_root/build/bridge.o" ]]; then
      bridge_o="$repo_root/build/bridge.o"
    else
      bridge_o="/Users/sergey/Projects/Crystal/cogni-ml/build/bridge.o"
    fi
  fi

  if [[ ! -f "$bridge_o" ]]; then
    echo "bridge object not found: $bridge_o" >&2
    echo "Set COGNI_ML_BRIDGE_O=/path/to/bridge.o or build the repo bridge first." >&2
    exit 2
  fi

  "$crystal_bin" build "$repo_root/scripts/diffusion_gemma_sparse_loop_smoke.cr" \
    -o "$bin" \
    --link-flags="$bridge_o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
fi

mkdir -p "$(dirname "$out")"
printf 'status\tcase\tprompt_bytes\tprompt_len\tcanvas_len\tcandidate_count\tload_ms\tprompt_route_ms\tprompt_projection_backend\tprompt_cache_ms\tprompt_projection_ms\tprompt_projection_norm_ms\tprompt_projection_matmul_ms\tprompt_projection_assemble_ms\tprompt_projection_copy_ms\tprompt_projection_head_norm_ms\tprompt_projection_q_norm_ms\tprompt_projection_k_norm_ms\tprompt_projection_v_norm_ms\tprompt_projection_rope_ms\tprompt_materialize_ms\tprompt_cache_ms_samples\tprompt_projection_ms_samples\tprompt_projection_norm_ms_samples\tprompt_projection_matmul_ms_samples\tprompt_projection_assemble_ms_samples\tprompt_projection_copy_ms_samples\tprompt_projection_head_norm_ms_samples\tprompt_projection_q_norm_ms_samples\tprompt_projection_k_norm_ms_samples\tprompt_projection_v_norm_ms_samples\tprompt_projection_rope_ms_samples\tprompt_materialize_ms_samples\tprompt_cache_tokens_per_ms\tloop_ms_median\tloop_ms_samples\tloop_prediction_ms\tloop_decode_stack_ms\tloop_decode_qkv_ms\tloop_decode_context_ms\tloop_decode_attention_out_ms\tloop_decode_shared_ffn_ms\tloop_decode_moe_ffn_ms\tloop_decode_combine_scale_ms\tloop_output_head_ms\tloop_update_ms\tloop_regenerate_ms\tloop_proposal_ms\tloop_prediction_ms_samples\tloop_decode_stack_ms_samples\tloop_decode_qkv_ms_samples\tloop_decode_context_ms_samples\tloop_decode_attention_out_ms_samples\tloop_decode_shared_ffn_ms_samples\tloop_decode_moe_ffn_ms_samples\tloop_decode_combine_scale_ms_samples\tloop_output_head_ms_samples\tloop_update_ms_samples\tloop_regenerate_ms_samples\tloop_proposal_ms_samples\tchosen\tprobs\tartifact\terr_artifact\n' >"$out"

append_failed_row() {
  local status="$1"
  local name="$2"
  local bytes="$3"
  local artifact="$4"
  local err_artifact="$5"
  awk -v status="$status" -v case="$name" -v bytes="$bytes" -v artifact="$artifact" -v err_artifact="$err_artifact" 'BEGIN {
    printf "%s\t%s\t%s", status, case, bytes
    for (i = 4; i <= 62; i++) {
      printf "\t"
    }
    printf "\t%s\t%s\n", artifact, err_artifact
  }' >>"$out"
}

run_case() {
  local name="$1"
  local prompt="$2"
  local tmp err bytes rc
  local extra_args=()
  tmp="$(mktemp "${TMPDIR:-/tmp}/dg_prompt_perf_${name}.tsv.XXXXXX")"
  err="$(mktemp "${TMPDIR:-/tmp}/dg_prompt_perf_${name}.err.XXXXXX")"
  bytes="$(printf '%s' "$prompt" | wc -c | tr -d ' ')"
  if [[ "$materialize_prompt_final_rows" -eq 1 ]]; then
    extra_args+=(--materialize-prompt-final-rows)
  fi

  set +e
  "$timeout_bin" "$timeout_seconds" "$bin" \
    --model "$model" \
    --llama-tokenize "$tokenizer" \
    --prompt "$prompt" \
    --canvas "$canvas" \
    --candidate-texts "$candidates" \
    --steps "$steps" \
    --warmups "$warmups" \
    --repeats "$repeats" \
    --cache-warmups "$cache_warmups" \
    --cache-repeats "$cache_repeats" \
    --max-layers "$max_layers" \
    --format tsv \
    --decode-canvas-text \
    "${extra_args[@]}" \
    >"$tmp" 2>"$err"
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    awk -F '\t' -v case="$name" -v bytes="$bytes" -v artifact="$tmp" -v err_artifact="$err" '
      NR == 1 {
        for (i = 1; i <= NF; i++) {
          h[$i] = i
        }
        next
      }
      NR == 2 {
        found = 1
        print "ok" "\t" case "\t" bytes "\t" $h["prompt_len"] "\t" $h["canvas_len"] "\t" $h["candidate_count"] "\t" $h["load_ms"] "\t" $h["prompt_route_ms"] "\t" $h["prompt_projection_backend"] "\t" $h["prompt_cache_ms"] "\t" $h["prompt_projection_ms"] "\t" $h["prompt_projection_norm_ms"] "\t" $h["prompt_projection_matmul_ms"] "\t" $h["prompt_projection_assemble_ms"] "\t" $h["prompt_projection_copy_ms"] "\t" $h["prompt_projection_head_norm_ms"] "\t" $h["prompt_projection_q_norm_ms"] "\t" $h["prompt_projection_k_norm_ms"] "\t" $h["prompt_projection_v_norm_ms"] "\t" $h["prompt_projection_rope_ms"] "\t" $h["prompt_materialize_ms"] "\t" $h["prompt_cache_ms_samples"] "\t" $h["prompt_projection_ms_samples"] "\t" $h["prompt_projection_norm_ms_samples"] "\t" $h["prompt_projection_matmul_ms_samples"] "\t" $h["prompt_projection_assemble_ms_samples"] "\t" $h["prompt_projection_copy_ms_samples"] "\t" $h["prompt_projection_head_norm_ms_samples"] "\t" $h["prompt_projection_q_norm_ms_samples"] "\t" $h["prompt_projection_k_norm_ms_samples"] "\t" $h["prompt_projection_v_norm_ms_samples"] "\t" $h["prompt_projection_rope_ms_samples"] "\t" $h["prompt_materialize_ms_samples"] "\t" $h["prompt_cache_tokens_per_ms"] "\t" $h["loop_ms_median"] "\t" $h["loop_ms_samples"] "\t" $h["loop_prediction_ms"] "\t" $h["loop_decode_stack_ms"] "\t" $h["loop_decode_qkv_ms"] "\t" $h["loop_decode_context_ms"] "\t" $h["loop_decode_attention_out_ms"] "\t" $h["loop_decode_shared_ffn_ms"] "\t" $h["loop_decode_moe_ffn_ms"] "\t" $h["loop_decode_combine_scale_ms"] "\t" $h["loop_output_head_ms"] "\t" $h["loop_update_ms"] "\t" $h["loop_regenerate_ms"] "\t" $h["loop_proposal_ms"] "\t" $h["loop_prediction_ms_samples"] "\t" $h["loop_decode_stack_ms_samples"] "\t" $h["loop_decode_qkv_ms_samples"] "\t" $h["loop_decode_context_ms_samples"] "\t" $h["loop_decode_attention_out_ms_samples"] "\t" $h["loop_decode_shared_ffn_ms_samples"] "\t" $h["loop_decode_moe_ffn_ms_samples"] "\t" $h["loop_decode_combine_scale_ms_samples"] "\t" $h["loop_output_head_ms_samples"] "\t" $h["loop_update_ms_samples"] "\t" $h["loop_regenerate_ms_samples"] "\t" $h["loop_proposal_ms_samples"] "\t" $h["last_chosen_texts"] "\t" $h["last_argmax_probabilities"] "\t" artifact "\t" err_artifact
      }
      END {
        if (!found) {
          exit 3
        }
      }
    ' "$tmp" >>"$out" || append_failed_row failed "$name" "$bytes" "$tmp" "$err"
  elif [[ "$rc" -eq 124 ]]; then
    append_failed_row timeout "$name" "$bytes" "$tmp" "$err"
  else
    append_failed_row failed "$name" "$bytes" "$tmp" "$err"
  fi
}

run_case short "$short_prompt"
run_case medium "$medium_prompt"
run_case long "$long_prompt"
if [[ "$include_large_prompt" -eq 1 ]]; then
  run_case large "$large_prompt"
fi

cat "$out"
