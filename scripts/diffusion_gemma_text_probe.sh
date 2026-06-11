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
expect_min_prob="${EXPECT_MIN_PROB:-}"
expect_canvas_len="${EXPECT_CANVAS_LEN:-}"
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

if [[ -n "$expect_canvas_len" ]]; then
  if [[ ! "$expect_canvas_len" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECT_CANVAS_LEN must be a positive integer" >&2
    exit 2
  fi
fi

if [[ -n "$expect_min_prob" ]]; then
  if [[ ! "$expect_min_prob" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "EXPECT_MIN_PROB must be a number in [0, 1]" >&2
    exit 2
  fi
  if ! awk -v value="$expect_min_prob" 'BEGIN { exit !(value >= 0 && value <= 1) }'; then
    echo "EXPECT_MIN_PROB must be a number in [0, 1]" >&2
    exit 2
  fi
fi

if [[ -n "$expect_chosen$expect_min_prob$expect_canvas_len" && "$format" != "tsv" ]]; then
  echo "EXPECT_CHOSEN/EXPECT_MIN_PROB/EXPECT_CANVAS_LEN require FORMAT=tsv" >&2
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

if [[ -z "$expect_chosen$expect_min_prob$expect_canvas_len" ]]; then
  exec "${smoke_cmd[@]}"
fi

tmp_out="$(mktemp "${TMPDIR:-/tmp}/diffusion_gemma_text_probe.XXXXXX")"
trap 'rm -f "$tmp_out"' EXIT

"${smoke_cmd[@]}" >"$tmp_out"
cat "$tmp_out"

awk -F '\t' -v expected="$expect_chosen" -v min_prob="$expect_min_prob" -v expected_canvas_len="$expect_canvas_len" '
  NR == 1 {
    for (i = 1; i <= NF; i++) {
      h[$i] = i
    }
    if (expected_canvas_len != "" && !("canvas_len" in h)) {
      print "canvas_len column missing" > "/dev/stderr"
      exit 2
    }
    if (expected != "" && !("last_chosen_texts" in h)) {
      print "last_chosen_texts column missing" > "/dev/stderr"
      exit 2
    }
    if (min_prob != "" && !("last_argmax_probabilities" in h)) {
      print "last_argmax_probabilities column missing" > "/dev/stderr"
      exit 2
    }
    next
  }
  NR == 2 {
    found = 1
    if (expected_canvas_len != "" && $h["canvas_len"] != expected_canvas_len) {
      printf("expected canvas_len=%s, got %s\n", expected_canvas_len, $h["canvas_len"]) > "/dev/stderr"
      exit 2
    }
    if (expected != "" && $h["last_chosen_texts"] != expected) {
      printf("expected last_chosen_texts=%s, got %s\n", expected, $h["last_chosen_texts"]) > "/dev/stderr"
      exit 2
    }
    if (min_prob != "") {
      count = split($h["last_argmax_probabilities"], probs, ",")
      for (i = 1; i <= count; i++) {
        if ((probs[i] + 0) < (min_prob + 0)) {
          printf("expected every last_argmax_probability >= %s, got %s\n", min_prob, probs[i]) > "/dev/stderr"
          exit 2
        }
      }
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
