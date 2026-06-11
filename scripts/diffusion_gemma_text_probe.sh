#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

prompt="${PROMPT:-Say:}"
canvas="${CANVAS:-Hello}"
candidates="${CANDIDATES:-Hello|world}"
steps="${STEPS:-1}"
warmups="${WARMUPS:-1}"
repeats="${REPEATS:-2}"
format="${FORMAT:-tsv}"

case "$format" in
  keyvalue|tsv) ;;
  *)
    echo "FORMAT must be keyvalue or tsv" >&2
    exit 2
    ;;
esac

if [[ -z "$prompt" ]]; then
  echo "PROMPT must not be empty" >&2
  exit 2
fi

if [[ -z "$canvas" ]]; then
  echo "CANVAS must not be empty" >&2
  exit 2
fi

if [[ -z "$candidates" || "$candidates" == *"||"* || "$candidates" == "|"* || "$candidates" == *"|" ]]; then
  echo "CANDIDATES must be a pipe-separated list with no empty entries" >&2
  exit 2
fi

for numeric in steps warmups repeats; do
  value="${!numeric}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "${numeric^^} must be a non-negative integer" >&2
    exit 2
  fi
done

if [[ "$steps" -lt 1 ]]; then
  echo "STEPS must be positive" >&2
  exit 2
fi

if [[ "$repeats" -lt 1 ]]; then
  echo "REPEATS must be positive" >&2
  exit 2
fi

exec "$repo_root/scripts/diffusion_gemma_sparse_loop_smoke.sh" \
  --prompt "$prompt" \
  --canvas "$canvas" \
  --candidate-texts "$candidates" \
  --steps "$steps" \
  --warmups "$warmups" \
  --repeats "$repeats" \
  --format "$format" \
  --decode-canvas-text \
  "$@"
