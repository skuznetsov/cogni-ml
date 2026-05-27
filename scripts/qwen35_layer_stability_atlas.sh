#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

probe_bin="${QWEN35_LAYER_STABILITY_ATLAS_BIN:-/tmp/qwen35_layer_stability_probe}"
crystal_cache="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_layer_stability_cache}"
link_flags="${QWEN35_LAYER_STABILITY_LINK_FLAGS:-${repo_root}/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++}"
build_flags="${QWEN35_LAYER_STABILITY_BUILD_FLAGS:---release}"
force_build="${QWEN35_LAYER_STABILITY_FORCE_BUILD:-0}"
gen="${QWEN35_LAYER_STABILITY_GEN:-4}"
max_seq="${QWEN35_LAYER_STABILITY_MAX_SEQ:-256}"
prompt_limit="${QWEN35_LAYER_STABILITY_PROMPT_LIMIT:-0}"
keep_logs="${QWEN35_LAYER_STABILITY_KEEP_LOGS:-0}"

if [[ "${force_build}" == "1" || ! -x "${probe_bin}" ]]; then
  if [[ ! -f "${repo_root}/build/bridge.o" && "${link_flags}" == *"build/bridge.o"* ]]; then
    echo "missing ${repo_root}/build/bridge.o; build the bridge first or set QWEN35_LAYER_STABILITY_LINK_FLAGS" >&2
    exit 2
  fi
  echo "building ${probe_bin}" >&2
  (
    cd "${repo_root}"
    CRYSTAL_CACHE_DIR="${crystal_cache}" crystal build bin/qwen35_layer_stability_probe.cr \
      ${build_flags} \
      -o "${probe_bin}" \
      --link-flags="${link_flags}"
  )
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-layer-stability.XXXXXX")"
if [[ "${keep_logs}" != "1" ]]; then
  trap 'rm -rf "${tmp_dir}"' EXIT
else
  echo "keeping logs in ${tmp_dir}" >&2
fi

prompt_ids=(facts code json repeat reasoning)
prompts=(
  "The capital of France is"
  "Write a Crystal function that returns the maximum of two Int32 values. Return only code."
  "Return a compact JSON object with keys city, country, language for Tokyo."
  "alpha one beta two alpha one beta two alpha one beta"
  "A farmer has 12 apples, gives away 5, then buys twice as many as remain. How many apples now? Think briefly."
)

printf 'prompt_id\tstep\tpos\tinput_token\tfinal_top1\tstable_from_layer\ttop1_changes\tfinal_logit\tprompt_tokens\tunique_rate\trepeat_rate\tbigram_repeat_rate\tadjacent_repeat_rate\n'

count="${#prompt_ids[@]}"
if [[ "${prompt_limit}" -gt 0 && "${prompt_limit}" -lt "${count}" ]]; then
  count="${prompt_limit}"
fi

for ((i = 0; i < count; i++)); do
  prompt_id="${prompt_ids[$i]}"
  prompt="${prompts[$i]}"
  log="${tmp_dir}/${prompt_id}.log"
  "${probe_bin}" --prompt "${prompt}" --gen "${gen}" --max-seq "${max_seq}" >"${log}" 2>&1
  awk -v prompt_id="${prompt_id}" '
    BEGIN { found = 0 }
    $1 == "summary" {
      print prompt_id "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $8 "\t" $9 "\t" $10 "\t" $11 "\t" $12 "\t" $13
      found = 1
    }
    END { if (!found) exit 1 }
  ' "${log}"
done
