#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crystal_bin="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
out="${DIFFUSION_GEMMA_SPARSE_LOOP_BIN:-/tmp/diffusion_gemma_sparse_loop_smoke}"
lengths_arg="${LENGTHS:-1 2 4 8}"
summary="${SUMMARY:-0}"

case "$summary" in
  0|1) ;;
  *)
    echo "SUMMARY must be 0 or 1" >&2
    exit 2
    ;;
esac

bridge_o="${COGNI_ML_BRIDGE_O:-}"
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

read -r -a lengths <<<"${lengths_arg//,/ }"
if [[ "${#lengths[@]}" -eq 0 ]]; then
  echo "LENGTHS is empty" >&2
  exit 2
fi
for length in "${lengths[@]}"; do
  if [[ ! "$length" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid prompt length: $length" >&2
    exit 2
  fi
done

"$crystal_bin" build "$repo_root/scripts/diffusion_gemma_sparse_loop_smoke.cr" \
  -o "$out" \
  --link-flags="$bridge_o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"

lengths_csv="$(IFS=,; echo "${lengths[*]}")"
if [[ "$summary" == "0" ]]; then
  exec "$out" --prompt-lengths "$lengths_csv" --format tsv "$@"
fi

summary_dir="${SUMMARY_DIR:-${TMPDIR:-/tmp}}"
mkdir -p "$summary_dir"
stamp="$(date +%Y%m%d%H%M%S)"
raw_tsv="${SUMMARY_RAW_TSV:-$summary_dir/diffusion_gemma_sparse_loop_prompt_sweep_${stamp}.tsv}"
summary_tsv="${SUMMARY_TSV:-$summary_dir/diffusion_gemma_sparse_loop_prompt_sweep_${stamp}.summary.tsv}"

"$out" --prompt-lengths "$lengths_csv" --format tsv "$@" >"$raw_tsv"
"$repo_root/scripts/diffusion_gemma_sparse_loop_summarize.py" "$raw_tsv" >"$summary_tsv"

printf 'raw_tsv=%s\n' "$raw_tsv"
printf 'summary_tsv=%s\n' "$summary_tsv"
cat "$summary_tsv"
