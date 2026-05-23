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
codec_list="${QWEN35_MATRIX_CODECS:-raw recurrent-bf16}"
live_kv_list="${QWEN35_MATRIX_LIVE_KV:-0}"
prompt_limit="${QWEN35_MATRIX_PROMPT_LIMIT:-0}"
mode="${QWEN35_MATRIX_MODE:---prompt-cache-fast-forward}"
keep_logs="${QWEN35_MATRIX_KEEP_LOGS:-0}"
force_build="${QWEN35_MATRIX_FORCE_BUILD:-0}"

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
        split($i, pair, "=")
        if (pair[1] == wanted) {
          print pair[2]
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
        split($i, pair, "=")
        if (pair[1] == wanted) {
          print pair[2]
          found = 1
          exit
        }
      }
    }
    END { if (!found) exit 1 }
  ' "${file}" || true
}

printf "prompt_id\tcodec\tlive_kv\tavg_total_ms\tp50_total_ms\tavg_ms_per_tok\tp50_restore_ms\tp50_prefill_ms\tp50_decode_ms\tprompt_tokens\toutput_tokens\tlog\n"

prompt_count="${#prompts[@]}"
if [[ "${prompt_limit}" =~ ^[0-9]+$ && "${prompt_limit}" -gt 0 && "${prompt_limit}" -lt "${prompt_count}" ]]; then
  prompt_count="${prompt_limit}"
fi

for ((idx = 0; idx < prompt_count; idx++)); do
  prompt_id="${prompt_ids[$idx]}"
  prompt="${prompts[$idx]}"
  for codec in ${codec_list}; do
    for live_kv in ${live_kv_list}; do
      log_file="${tmp_dir}/${prompt_id}.${codec}.livekv${live_kv}.log"
      cmd=(
        "${probe_bin}"
        "${mode}"
        "--gen=${gen}"
        "--requests=${requests}"
        "--warmups=${warmups}"
        "--max-seq=${max_seq}"
        "--resident-states=${resident_states}"
        "--quiet"
      )
      if [[ "${codec}" != "raw" ]]; then
        cmd+=("--artifact-codec=${codec}" "--artifact-codec-block=${artifact_block}")
      fi
      if [[ "${live_kv}" == "1" ]]; then
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
      cmd+=("${prompt}")

      "${cmd[@]}" >"${log_file}"

      avg_total="$(extract_field avg_total_ms "${log_file}")"
      p50_total="$(extract_field p50_total_ms "${log_file}")"
      avg_ms_per_tok="$(extract_field avg_ms_per_tok "${log_file}")"
      p50_restore="$(extract_field p50_restore_ms "${log_file}")"
      p50_prefill="$(extract_field p50_prefill_ms "${log_file}")"
      p50_decode="$(extract_field p50_decode_ms "${log_file}")"
      prompt_tokens="$(extract_request_field prompt_tokens "${log_file}")"
      output_tokens="$(extract_request_field output_tokens "${log_file}")"

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${prompt_id}" \
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
        "${log_file}"
    done
  done
done
