#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

probe_bin="${QWEN35_WARM_PROBE_BIN:-/tmp/qwen35_warm_request_probe_matrix}"
crystal_cache="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_warm_probe_cache}"
link_flags="${QWEN35_MATRIX_LINK_FLAGS:-${repo_root}/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++}"
build_flags="${QWEN35_MATRIX_BUILD_FLAGS:---release}"

gen="${QWEN35_MATRIX_GEN:-16}"
requests="${QWEN35_MATRIX_REQUESTS:-3}"
warmups="${QWEN35_MATRIX_WARMUPS:-1}"
max_seq="${QWEN35_MATRIX_MAX_SEQ:-256}"
resident_states="${QWEN35_MATRIX_RESIDENT_STATES:-0}"
reuse_request_state="${QWEN35_MATRIX_REUSE_REQUEST_STATE:-0}"
artifact_block="${QWEN35_MATRIX_ARTIFACT_BLOCK:-8}"
serving_continuation="${QWEN35_MATRIX_SERVING_CONTINUATION:-0}"
serving_direct_miss="${QWEN35_MATRIX_SERVING_DIRECT_MISS:-0}"
serving_active_cursor="${QWEN35_MATRIX_SERVING_ACTIVE_CURSOR:-0}"
prompt_limit="${QWEN35_MATRIX_PROMPT_LIMIT:-0}"
matrix_mode="${QWEN35_MATRIX_MODE:---prompt-cache-fast-forward}"
keep_logs="${QWEN35_MATRIX_KEEP_LOGS:-0}"
force_build="${QWEN35_MATRIX_FORCE_BUILD:-0}"

case "${matrix_mode}" in
  --prompt-cache-fast-forward|prompt-cache-fast-forward|fast-forward)
    mode_arg="--prompt-cache-fast-forward"
    mode_label="fast_forward"
    codec_list="${QWEN35_MATRIX_CODECS:-raw recurrent-bf16}"
    live_kv_list="${QWEN35_MATRIX_LIVE_KV:-0}"
    ;;
  --prompt-cache-replay|prompt-cache-replay|replay)
    mode_arg="--prompt-cache-replay"
    mode_label="replay"
    codec_list="${QWEN35_MATRIX_CODECS:-raw recurrent-bf16}"
    live_kv_list="${QWEN35_MATRIX_LIVE_KV:-0}"
    ;;
  --prompt-cache-direct-output|prompt-cache-direct-output|direct-output|direct_output)
    mode_arg="--prompt-cache-direct-output"
    mode_label="direct_output"
    codec_list="${QWEN35_MATRIX_CODECS:-direct}"
    live_kv_list="${QWEN35_MATRIX_LIVE_KV:-na}"
    ;;
  --prompt-cache-serving-route|prompt-cache-serving-route|serving-route|serving_route)
    mode_arg="--prompt-cache-serving-route"
    mode_label="serving_route"
    codec_list="${QWEN35_MATRIX_CODECS:-raw recurrent-bf16}"
    live_kv_list="${QWEN35_MATRIX_LIVE_KV:-0}"
    ;;
  *)
    echo "invalid QWEN35_MATRIX_MODE: ${matrix_mode}" >&2
    echo "expected fast-forward, replay, direct-output, serving-route, or the matching --prompt-cache-* flag" >&2
    exit 2
    ;;
esac

if [[ "${serving_active_cursor}" == "1" && "${serving_continuation}" != "1" ]]; then
  echo "QWEN35_MATRIX_SERVING_ACTIVE_CURSOR=1 requires QWEN35_MATRIX_SERVING_CONTINUATION=1" >&2
  exit 2
fi

if [[ "${force_build}" == "1" || ! -x "${probe_bin}" ]]; then
  if [[ ! -f "${repo_root}/build/bridge.o" && "${link_flags}" == *"build/bridge.o"* ]]; then
    echo "missing ${repo_root}/build/bridge.o; build the bridge first or set QWEN35_MATRIX_LINK_FLAGS" >&2
    exit 2
  fi
  echo "building ${probe_bin}" >&2
  (
    cd "${repo_root}"
    CRYSTAL_CACHE_DIR="${crystal_cache}" crystal build bin/qwen35_warm_request_probe.cr \
      ${build_flags} \
      -o "${probe_bin}" \
      --link-flags="${link_flags}"
  )
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-cache-matrix.XXXXXX")"
if [[ "${keep_logs}" != "1" ]]; then
  trap 'rm -rf "${tmp_dir}"' EXIT
else
  echo "keeping logs in ${tmp_dir}" >&2
fi

prompt_ids=(
  "fact"
  "code"
  "json"
  "repeat"
  "cache"
)

prompts=(
  "The capital of France is"
  "Write a Crystal function that checks whether a string is a palindrome. Return only code."
  "Return a compact JSON object with keys city, country, language for Tokyo."
  "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"
  "Explain in three concise steps why caching a validated KV state can reduce latency for a local LLM session."
)

extract_field() {
  local field="$1"
  local file="$2"
  awk -v wanted="${field}" '
    /aggregate:/ {
      for (i = 1; i <= NF; i++) {
        prefix = wanted "="
        if (index($i, prefix) == 1) {
          print substr($i, length(prefix) + 1)
          found = 1
        }
      }
    }
    END { if (!found) exit 1 }
  ' "${file}"
}

extract_request_field() {
  local field="$1"
  local file="$2"
  awk -v wanted="${field}" '
    /request [0-9]+ summary:/ {
      for (i = 1; i <= NF; i++) {
        prefix = wanted "="
        if (index($i, prefix) == 1) {
          print substr($i, length(prefix) + 1)
          found = 1
          exit
        }
      }
    }
    END { if (!found) exit 1 }
  ' "${file}" || true
}

printf "prompt_id\tmode\tcodec\tlive_kv\tavg_total_ms\tp50_total_ms\tavg_ms_per_tok\tp50_restore_ms\tp50_prefill_ms\tp50_decode_ms\tprompt_tokens\toutput_tokens\troutes\tlog\n"

prompt_count="${#prompts[@]}"
if [[ "${prompt_limit}" =~ ^[0-9]+$ && "${prompt_limit}" -gt 0 && "${prompt_limit}" -lt "${prompt_count}" ]]; then
  prompt_count="${prompt_limit}"
fi

for ((idx = 0; idx < prompt_count; idx++)); do
  prompt_id="${prompt_ids[$idx]}"
  prompt="${prompts[$idx]}"
  for codec in ${codec_list}; do
    for live_kv in ${live_kv_list}; do
      log_file="${tmp_dir}/${prompt_id}.${mode_label}.${codec}.livekv${live_kv}.log"
      cmd=(
        "${probe_bin}"
        "${mode_arg}"
        "--gen=${gen}"
        "--requests=${requests}"
        "--warmups=${warmups}"
        "--max-seq=${max_seq}"
        "--resident-states=${resident_states}"
        "--quiet"
      )
      if [[ "${mode_label}" != "direct_output" && "${codec}" != "raw" ]]; then
        cmd+=("--artifact-codec=${codec}" "--artifact-codec-block=${artifact_block}")
      fi
      if [[ "${mode_label}" == "direct_output" ]]; then
        if [[ "${live_kv}" != "na" ]]; then
          echo "invalid QWEN35_MATRIX_LIVE_KV for direct-output: ${live_kv}; expected na" >&2
          exit 2
        fi
      elif [[ "${live_kv}" == "1" ]]; then
        cmd+=("--live-kv-artifacts")
      elif [[ "${live_kv}" != "0" ]]; then
        echo "invalid QWEN35_MATRIX_LIVE_KV value: ${live_kv}; expected 0 or 1" >&2
        exit 2
      fi
      if [[ "${reuse_request_state}" == "1" ]]; then
        cmd+=("--reuse-request-state")
      elif [[ "${reuse_request_state}" != "0" ]]; then
        echo "invalid QWEN35_MATRIX_REUSE_REQUEST_STATE value: ${reuse_request_state}; expected 0 or 1" >&2
        exit 2
      fi
      if [[ "${mode_label}" == "serving_route" ]]; then
        if [[ "${serving_continuation}" == "1" ]]; then
          cmd+=("--serving-route-continuation")
        elif [[ "${serving_continuation}" != "0" ]]; then
          echo "invalid QWEN35_MATRIX_SERVING_CONTINUATION value: ${serving_continuation}; expected 0 or 1" >&2
          exit 2
        fi
        if [[ "${serving_direct_miss}" == "1" ]]; then
          cmd+=("--serving-route-direct-miss")
        elif [[ "${serving_direct_miss}" != "0" ]]; then
          echo "invalid QWEN35_MATRIX_SERVING_DIRECT_MISS value: ${serving_direct_miss}; expected 0 or 1" >&2
          exit 2
        fi
        if [[ "${serving_active_cursor}" == "1" ]]; then
          cmd+=("--serving-route-active-cursor")
        elif [[ "${serving_active_cursor}" != "0" ]]; then
          echo "invalid QWEN35_MATRIX_SERVING_ACTIVE_CURSOR value: ${serving_active_cursor}; expected 0 or 1" >&2
          exit 2
        fi
      fi
      cmd+=("${prompt}")

      "${cmd[@]}" >"${log_file}"

      avg_total="$(extract_field avg_total_ms "${log_file}")"
      p50_total="$(extract_field p50_total_ms "${log_file}")"
      avg_ms_per_tok="$(extract_field avg_ms_per_tok "${log_file}")"
      p50_restore="$(extract_field p50_restore_ms "${log_file}")"
      p50_prefill="$(extract_field p50_prefill_ms "${log_file}")"
      p50_decode="$(extract_field p50_decode_ms "${log_file}")"
      routes="$(extract_field routes "${log_file}")"
      prompt_tokens="$(extract_request_field prompt_tokens "${log_file}")"
      output_tokens="$(extract_request_field output_tokens "${log_file}")"

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${prompt_id}" \
        "${mode_label}" \
        "${codec}" \
        "${live_kv}" \
        "${avg_total}" \
        "${p50_total}" \
        "${avg_ms_per_tok}" \
        "${p50_restore}" \
        "${p50_prefill}" \
        "${p50_decode}" \
        "${prompt_tokens:-?}" \
        "${output_tokens:-?}" \
        "${routes:-?}" \
        "${log_file}"
    done
  done
done
