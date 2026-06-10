#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crystal_bin="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
out="${DIFFUSION_GEMMA_SPARSE_LOOP_BIN:-/tmp/diffusion_gemma_sparse_loop_smoke}"
counts_arg="${COUNTS:-1 2 4 8}"

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

read -r -a counts <<<"${counts_arg//,/ }"
if [[ "${#counts[@]}" -eq 0 ]]; then
  echo "COUNTS is empty" >&2
  exit 2
fi
for count in "${counts[@]}"; do
  if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid candidate count: $count" >&2
    exit 2
  fi
done

"$crystal_bin" build "$repo_root/scripts/diffusion_gemma_sparse_loop_smoke.cr" \
  -o "$out" \
  --link-flags="$bridge_o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"

counts_csv="$(IFS=,; echo "${counts[*]}")"
exec "$out" --candidate-counts "$counts_csv" --format tsv "$@"
