#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" -gt 0 ]]; then
  export PROMPT="$*"
fi

export MODE="${MODE:-fast-quality}"
export RAW_OUTPUT="${RAW_OUTPUT:-0}"

exec "$repo_root/scripts/diffusion_gemma_proto.sh"
