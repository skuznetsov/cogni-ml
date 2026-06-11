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
expect_chosen="${EXPECT_CHOSEN:-}"
smoke_runner="${SMOKE_RUNNER:-$repo_root/scripts/diffusion_gemma_sparse_loop_smoke.sh}"

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

if [[ -n "$expect_chosen" && "$format" != "tsv" ]]; then
  echo "EXPECT_CHOSEN requires FORMAT=tsv" >&2
  exit 2
fi

smoke_cmd=(
  "$smoke_runner"
  --prompt "$prompt"
  --canvas "$canvas"
  --candidate-texts "$candidates"
  --steps "$steps"
  --warmups "$warmups"
  --repeats "$repeats"
  --format "$format"
  --decode-canvas-text
  "$@"
)

if [[ -z "$expect_chosen" ]]; then
  exec "${smoke_cmd[@]}"
fi

tmp_out="$(mktemp "${TMPDIR:-/tmp}/diffusion_gemma_text_probe.XXXXXX.tsv")"
trap 'rm -f "$tmp_out"' EXIT

"${smoke_cmd[@]}" >"$tmp_out"
cat "$tmp_out"

awk -F '\t' -v expected="$expect_chosen" '
  NR == 1 {
    for (i = 1; i <= NF; i++) {
      h[$i] = i
    }
    if (!("last_chosen_texts" in h)) {
      print "last_chosen_texts column missing" > "/dev/stderr"
      exit 2
    }
    next
  }
  NR == 2 {
    found = 1
    if ($h["last_chosen_texts"] != expected) {
      printf("expected last_chosen_texts=%s, got %s\n", expected, $h["last_chosen_texts"]) > "/dev/stderr"
      exit 2
    }
    next
  }
  END {
    if (!found) {
      print "no data row emitted" > "/dev/stderr"
      exit 2
    }
  }
' "$tmp_out"
