#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

probe_bin="${QWEN35_SERVING_MATRIX_BIN:-/tmp/qwen35_warm_serving_route_matrix}"
crystal_cache="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_serving_route_matrix_cache}"
link_flags="${QWEN35_SERVING_MATRIX_LINK_FLAGS:-${repo_root}/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++}"
build_flags="${QWEN35_SERVING_MATRIX_BUILD_FLAGS:---release}"

modes="${QWEN35_SERVING_MATRIX_MODES:-greedy source_replay direct_output serving_terminal serving_direct_miss serving_continuation active_cursor fast_forward_live}"
gen="${QWEN35_SERVING_MATRIX_GEN:-16}"
cached_gen="${QWEN35_SERVING_MATRIX_CACHED_GEN:-${gen}}"
requests="${QWEN35_SERVING_MATRIX_REQUESTS:-3}"
warmups="${QWEN35_SERVING_MATRIX_WARMUPS:-1}"
max_seq="${QWEN35_SERVING_MATRIX_MAX_SEQ:-256}"
artifact_codec="${QWEN35_SERVING_MATRIX_ARTIFACT_CODEC:-recurrent-bf16}"
artifact_block="${QWEN35_SERVING_MATRIX_ARTIFACT_BLOCK:-8}"
prompt_limit="${QWEN35_SERVING_MATRIX_PROMPT_LIMIT:-0}"
keep_logs="${QWEN35_SERVING_MATRIX_KEEP_LOGS:-0}"
force_build="${QWEN35_SERVING_MATRIX_FORCE_BUILD:-0}"

if [[ "${force_build}" == "1" || ! -x "${probe_bin}" ]]; then
  if [[ ! -f "${repo_root}/build/bridge.o" && "${link_flags}" == *"build/bridge.o"* ]]; then
    echo "missing ${repo_root}/build/bridge.o; build the bridge first or set QWEN35_SERVING_MATRIX_LINK_FLAGS" >&2
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

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-serving-matrix.XXXXXX")"
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

mode_args() {
  local mode="$1"
  case "${mode}" in
    greedy)
      ;;
    source_replay)
      printf '%s\0' --source-replay
      ;;
    direct_output)
      printf '%s\0' --prompt-cache-direct-output
      ;;
    serving_terminal)
      printf '%s\0' --prompt-cache-serving-route
      ;;
    serving_direct_miss)
      printf '%s\0' --prompt-cache-serving-route --serving-route-direct-miss
      ;;
    serving_continuation)
      printf '%s\0' --prompt-cache-serving-route --serving-route-continuation --artifact-codec "${artifact_codec}" --artifact-codec-block "${artifact_block}" --live-kv-artifacts
      ;;
    active_cursor)
      printf '%s\0' --prompt-cache-serving-route --serving-route-continuation --serving-route-active-cursor --artifact-codec "${artifact_codec}" --artifact-codec-block "${artifact_block}" --live-kv-artifacts --reuse-request-state
      ;;
    fast_forward_live)
      printf '%s\0' --prompt-cache-fast-forward --artifact-codec "${artifact_codec}" --artifact-codec-block "${artifact_block}" --live-kv-artifacts --reuse-request-state
      ;;
    *)
      echo "invalid mode: ${mode}" >&2
      echo "valid modes: greedy source_replay direct_output serving_terminal serving_direct_miss serving_continuation active_cursor fast_forward_live" >&2
      exit 2
      ;;
  esac
}

printf "prompt_id\tmode\tavg_total_ms\tp50_total_ms\tavg_ms_per_tok\tp50_restore_ms\tp50_prefill_ms\tp50_decode_ms\tprompt_tokens\toutput_tokens\troutes\tlog\n"

prompt_count="${#prompts[@]}"
if [[ "${prompt_limit}" =~ ^[0-9]+$ && "${prompt_limit}" -gt 0 && "${prompt_limit}" -lt "${prompt_count}" ]]; then
  prompt_count="${prompt_limit}"
fi

for ((idx = 0; idx < prompt_count; idx++)); do
  prompt_id="${prompt_ids[$idx]}"
  prompt="${prompts[$idx]}"
  for mode in ${modes}; do
    log_file="${tmp_dir}/${prompt_id}.${mode}.log"
    cmd=(
      "${probe_bin}"
      "--gen" "${gen}"
      "--cached-gen" "${cached_gen}"
      "--requests" "${requests}"
      "--warmups" "${warmups}"
      "--max-seq" "${max_seq}"
      "--quiet"
    )
    while IFS= read -r -d '' arg; do
      cmd+=("${arg}")
    done < <(mode_args "${mode}")
    cmd+=("${prompt}")

    "${cmd[@]}" >"${log_file}" 2>&1

    avg_total="$(extract_field avg_total_ms "${log_file}")"
    p50_total="$(extract_field p50_total_ms "${log_file}")"
    avg_ms_per_tok="$(extract_field avg_ms_per_tok "${log_file}")"
    p50_restore="$(extract_field p50_restore_ms "${log_file}")"
    p50_prefill="$(extract_field p50_prefill_ms "${log_file}")"
    p50_decode="$(extract_field p50_decode_ms "${log_file}")"
    routes="$(extract_field routes "${log_file}")"
    prompt_tokens="$(extract_request_field prompt_tokens "${log_file}")"
    output_tokens="$(extract_request_field output_tokens "${log_file}")"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${prompt_id}" \
      "${mode}" \
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
