#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

probe_bin="${QWEN35_ROUTE_CAL_PROBE_BIN:-/tmp/qwen35_route_memory_learn_probe}"
route_cli="${QWEN35_ROUTE_CAL_CLI_BIN:-/tmp/qwen35_proposal_route_memory}"
crystal_cache="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_route_cal_cache}"
link_flags="${QWEN35_ROUTE_CAL_LINK_FLAGS:-${repo_root}/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++}"
build_flags="${QWEN35_ROUTE_CAL_BUILD_FLAGS:---release}"
force_build="${QWEN35_ROUTE_CAL_FORCE_BUILD:-0}"

gen="${QWEN35_ROUTE_CAL_GEN:-16}"
gamma="${QWEN35_ROUTE_CAL_GAMMA:-4}"
draft_split="${QWEN35_ROUTE_CAL_DRAFT_SPLIT:-1}"
rank="${QWEN35_ROUTE_CAL_RANK:-16}"
updown_rank="${QWEN35_ROUTE_CAL_UPDOWN_RANK:-4}"
updown_layers="${QWEN35_ROUTE_CAL_UPDOWN_LAYERS:-0,2,4}"
tokens="${QWEN35_ROUTE_CAL_TOKENS:-256}"
calib_tokens="${QWEN35_ROUTE_CAL_CALIB_TOKENS:-999}"
pca_iters="${QWEN35_ROUTE_CAL_PCA_ITERS:-8}"
min_gain_ms="${QWEN35_ROUTE_CAL_MIN_GAIN_MS:-50}"
seed_baseline="${QWEN35_ROUTE_CAL_SEED_BASELINE:-1}"
dry_run="${QWEN35_ROUTE_CAL_DRY_RUN:-0}"
keep_logs="${QWEN35_ROUTE_CAL_KEEP_LOGS:-1}"
route_root="${QWEN35_ROUTE_CAL_ROOT:-${TMPDIR:-/tmp}/qwen35-proposal-route-cache}"
prompts_file="${QWEN35_ROUTE_CAL_PROMPTS_FILE:-}"
model_arg=()
[[ -n "${QWEN35_ROUTE_CAL_MODEL:-}" ]] && model_arg=(--model "${QWEN35_ROUTE_CAL_MODEL}")

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-route-cal.XXXXXX")"
if [[ "${keep_logs}" != "1" ]]; then
  trap 'rm -rf "${tmp_dir}"' EXIT
else
  echo "keeping logs in ${tmp_dir}" >&2
fi

if [[ ! -f "${repo_root}/build/bridge.o" && "${link_flags}" == *"build/bridge.o"* ]]; then
  echo "missing ${repo_root}/build/bridge.o; build the bridge first or set QWEN35_ROUTE_CAL_LINK_FLAGS" >&2
  exit 2
fi

if [[ "${force_build}" == "1" || ! -x "${probe_bin}" ]]; then
  echo "building ${probe_bin}" >&2
  (cd "${repo_root}" && CRYSTAL_CACHE_DIR="${crystal_cache}" crystal build bin/qwen35_deltanet_fixed_basis_probe.cr ${build_flags} -o "${probe_bin}" --link-flags="${link_flags}")
fi
if [[ "${force_build}" == "1" || ! -x "${route_cli}" ]]; then
  echo "building ${route_cli}" >&2
  (cd "${repo_root}" && CRYSTAL_CACHE_DIR="${crystal_cache}" crystal build bin/qwen35_proposal_route_memory.cr ${build_flags} -o "${route_cli}" --link-flags="${link_flags}")
fi

if [[ -z "${prompts_file}" ]]; then
  prompts_file="${tmp_dir}/default_prompts.txt"
  cat > "${prompts_file}" <<'EOF'
code_square::<|im_start|>system\nYou are a concise coding assistant.<|im_end|>\n<|im_start|>user\nWrite a tiny Crystal function that returns the square of an Int32. Explain in one sentence.<|im_end|>\n<|im_start|>assistant\n
code_clamp::<|im_start|>system\nYou are a precise Crystal programmer.<|im_end|>\n<|im_start|>user\nImplement a Crystal method `clamp(x : Int32, lo : Int32, hi : Int32) : Int32` and show one example call.<|im_end|>\n<|im_start|>assistant\n
science_bandwidth::<|im_start|>system\nYou are a concise technical explainer.<|im_end|>\n<|im_start|>user\nExplain in three sentences why memory bandwidth can limit local LLM decoding speed.<|im_end|>\n<|im_start|>assistant\n
EOF
fi

summary="${tmp_dir}/summary.tsv"
printf "name\tdecision\tbaseline_accept\tupdown_accept\tbaseline_ms\tupdown_ms\tgain_ms\tbaseline_rej\tupdown_rej\tbaseline_parity\tupdown_parity\tlog_dir\n" > "${summary}"

parse_metric() {
  local log="$1"
  python3 - "$log" <<'PY'
import pathlib,re,sys
text=pathlib.Path(sys.argv[1]).read_text(errors='replace')
line=next((ln for ln in text.splitlines() if ln.startswith('self_spec_gpu_pipeline layers=')), '')
if not line:
    print('PARSE_FAIL\t0\t0\t0\tfalse')
    raise SystemExit(0)
def g(pat, default=''):
    m=re.search(pat,line)
    return m.group(1) if m else default
print('\t'.join([g(r'accept_rate=([0-9.]+)%','0'), g(r'rejections=([0-9]+)','0'), g(r'overlap_ms=([0-9.]+)','0'), g(r'parity=([^ ]+)','false')]))
PY
}

while IFS= read -r raw || [[ -n "${raw}" ]]; do
  [[ -z "${raw//[[:space:]]/}" || "${raw}" == \#* ]] && continue
  if [[ "${raw}" != *"::"* ]]; then
    echo "bad prompt line, expected NAME::TEXT: ${raw}" >&2
    exit 2
  fi
  name="${raw%%::*}"
  escaped_prompt="${raw#*::}"
  prompt="$(printf '%b' "${escaped_prompt}")"
  safe_name="$(printf '%s' "${name}" | tr -c 'A-Za-z0-9_.-' '_')"
  base_log="${tmp_dir}/${safe_name}.baseline.log"
  up_log="${tmp_dir}/${safe_name}.updown.log"

  common=("${model_arg[@]}" --prompt "${prompt}" --prompt-name "${safe_name}" --prompt-as-prefix --tokens="${tokens}" --calib-tokens="${calib_tokens}" --ranks="${rank}" --basis=pca --pca-iters="${pca_iters}" --simulate-logits-rank="${rank}" --simulate-logits-layers="${updown_layers}" --simulate-generate="${gen}" --simulate-self-spec-gpu-pipeline-gammas="${gamma}" --simulate-self-spec-gpu-pipeline-draft-splits="${draft_split}")
  updown=(--simulate-self-spec-gpu-pipeline-draft-updown="${updown_rank}" --simulate-self-spec-gpu-pipeline-draft-updown-layers="${updown_layers}")

  echo "CAL ${safe_name} baseline" >&2
  if [[ "${dry_run}" == "1" ]]; then
    printf 'DRY baseline:' >&2; printf ' %q' "${probe_bin}" "${common[@]}" >&2; printf '\n' >&2
    base_metrics=$'0\t0\t0\ttrue'
  else
    "${repo_root}/scripts/run_safe.sh" "${probe_bin}" 260 12000 "${common[@]}" > "${base_log}" 2>&1
    base_metrics="$(parse_metric "${base_log}")"
  fi

  echo "CAL ${safe_name} updown" >&2
  if [[ "${dry_run}" == "1" ]]; then
    printf 'DRY updown:' >&2; printf ' %q' "${probe_bin}" "${common[@]}" "${updown[@]}" >&2; printf '\n' >&2
    up_metrics=$'0\t0\t0\ttrue'
  else
    "${repo_root}/scripts/run_safe.sh" "${probe_bin}" 260 12000 "${common[@]}" "${updown[@]}" > "${up_log}" 2>&1
    up_metrics="$(parse_metric "${up_log}")"
  fi

  decision_line="$(python3 - "${base_metrics}" "${up_metrics}" "${min_gain_ms}" <<'PY'
import sys
b=sys.argv[1].split('\t'); u=sys.argv[2].split('\t'); min_gain=float(sys.argv[3])
ba,br,bms,bp=float(b[0]),int(b[1]),float(b[2]),b[3]
ua,ur,ums,up=float(u[0]),int(u[1]),float(u[2]),u[3]
gain=bms-ums
ok=(bp=='true' and up=='true' and ua>=ba and gain>=min_gain)
print('\t'.join(['pca_updown' if ok else 'baseline', f'{ba:.2f}', f'{ua:.2f}', f'{bms:.3f}', f'{ums:.3f}', f'{gain:.3f}', str(br), str(ur), bp, up]))
PY
)"
  decision="${decision_line%%$'\t'*}"

  if [[ "${dry_run}" != "1" ]]; then
    if [[ "${decision}" == "pca_updown" ]]; then
      "${route_cli}" "${model_arg[@]}" --root "${route_root}" --prompt "${prompt}" --route-key "${safe_name}" --route pca_updown --rank "${updown_rank}" --layers "${updown_layers}" --trigger "offline-calibration" --evidence "${decision_line}" >/dev/null
    elif [[ "${seed_baseline}" == "1" ]]; then
      "${route_cli}" "${model_arg[@]}" --root "${route_root}" --prompt "${prompt}" --route-key "${safe_name}" --route baseline --trigger "offline-calibration" --evidence "${decision_line}" >/dev/null
    fi
  fi

  printf "%s\t%s\t%s\n" "${safe_name}" "${decision_line}" "${tmp_dir}" >> "${summary}"
done < "${prompts_file}"

cat "${summary}"
echo "route_root=${route_root}" >&2
echo "log_dir=${tmp_dir}" >&2
