#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SAFE="$ROOT/scripts/run_safe.sh"
CRYSTAL_BIN="${CRYSTAL_BIN:-/opt/homebrew/bin/crystal}"
LLAMA_BENCH="${LLAMA_BENCH:-$HOME/SrcArchives/AI/llama.cpp/build/bin/llama-bench}"
MODEL="${GEMMA4_MODEL:-$HOME/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf}"
PROFILE_BIN="${GEMMA4_PROFILE_BIN:-/tmp/gemma4_metal_decode_profile_body_bench}"
OUT_DIR="${OUT_DIR:-/tmp/gemma4_vs_llama_body_$(date +%Y%m%d%H%M%S)}"
PROMPT=256
GEN=32
REPS=3
WARMUPS=1
MAX_SEQ=512
PREFILL_CHUNK=512
TIMEOUT_SEC=420
MAX_RSS_MB=8000
RUN_TIMEOUT_MS=80000
LLAMA_THREADS=8
LLAMA_FA=1

usage() {
  cat <<'USAGE'
usage: scripts/gemma4_vs_llama_body_bench.sh [options]

Matched Gemma4 body-throughput bench. llama-bench tg does not compute logits;
this script compares it against CogniGemma --body-only decode, not top1 mode.

Options:
  --prompt N         Prompt tokens for pp (default: 256)
  --gen N            Generated/body decode tokens for tg (default: 32)
  --reps N           Measured repetitions (default: 3)
  --warmups N        Cogni warmups (default: 1)
  --max-seq N        Cogni max sequence (default: 512)
  --prefill-chunk N  Cogni row-prefill chunk (default: 512)
  --model PATH       Gemma4 GGUF path
  --profile-bin PATH Cogni profile binary path
  --llama-bench PATH llama-bench path
  --out-dir PATH     Output log directory
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="$2"; shift 2 ;;
    --gen) GEN="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --warmups) WARMUPS="$2"; shift 2 ;;
    --max-seq) MAX_SEQ="$2"; shift 2 ;;
    --prefill-chunk) PREFILL_CHUNK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --profile-bin) PROFILE_BIN="$2"; shift 2 ;;
    --llama-bench) LLAMA_BENCH="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"
[[ -x "$LLAMA_BENCH" ]] || { echo "llama-bench not executable: $LLAMA_BENCH" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

if [[ ! -x "$PROFILE_BIN" ]]; then
  echo "building profile binary: $PROFILE_BIN" >&2
  CRYSTAL_CACHE_DIR="${CRYSTAL_CACHE_DIR:-/tmp/cogni_ml_gemma4_body_bench_build}" \
    "$RUN_SAFE" "$CRYSTAL_BIN" "$TIMEOUT_SEC" "$MAX_RSS_MB" \
    build "$ROOT/bin/gemma4_metal_decode_profile.cr" \
    -o "$PROFILE_BIN" \
    --link-flags="$ROOT/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"
fi

TOKENS="$(python3 - <<PY
p=$PROMPT
print(','.join(str(42+i) for i in range(p)))
PY
)"

llama_log="$OUT_DIR/llama_p${PROMPT}_n${GEN}.log"
cogni_pp_log="$OUT_DIR/cogni_pp${PROMPT}_body.log"
cogni_tg_log="$OUT_DIR/cogni_tg${GEN}_body.log"

COGNI_RUN_SAFE_MIN_FREE_PCT=12 "$RUN_SAFE" "$LLAMA_BENCH" "$TIMEOUT_SEC" "$RUN_TIMEOUT_MS" \
  -m "$MODEL" -p "$PROMPT" -n "$GEN" -r "$REPS" -ngl 99 -fa "$LLAMA_FA" -t "$LLAMA_THREADS" -o json \
  > "$llama_log" 2>&1

COGNI_RUN_SAFE_MIN_FREE_PCT=12 GEMMA4_ROW_PREFILL_ALLOW_GEMM=1 "$RUN_SAFE" "$PROFILE_BIN" "$TIMEOUT_SEC" "$RUN_TIMEOUT_MS" \
  --model "$MODEL" --tokens "$TOKENS" --generate 1 --max-seq "$MAX_SEQ" --runs "$REPS" --warmups "$WARMUPS" \
  --prefill-mode rows --prefill-chunk "$PREFILL_CHUNK" --body-only --prefill-no-head \
  > "$cogni_pp_log" 2>&1

COGNI_RUN_SAFE_MIN_FREE_PCT=12 "$RUN_SAFE" "$PROFILE_BIN" "$TIMEOUT_SEC" "$RUN_TIMEOUT_MS" \
  --model "$MODEL" --tokens 42 --decode-only-seed 42 --generate "$GEN" --max-seq "$MAX_SEQ" --runs "$REPS" --warmups "$WARMUPS" \
  --prefill-mode rows --prefill-chunk "$PREFILL_CHUNK" --body-only \
  > "$cogni_tg_log" 2>&1

python3 - "$llama_log" "$cogni_pp_log" "$cogni_tg_log" <<'PY'
import json, re, sys
llama_log, cogni_pp_log, cogni_tg_log = sys.argv[1:]

def read(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

def llama_rows(path):
    s = read(path)
    if '=== STDOUT ===' in s:
        s = s.split('=== STDOUT ===', 1)[1].split('=== STDERR ===', 1)[0]
    a = s.find('[')
    b = s.find('\n]', a)
    if b >= 0:
        b += 1
    else:
        b = s.rfind(']')
    if a < 0 or b < a:
        raise SystemExit(f"cannot find JSON array in {path}")
    rows = json.loads(s[a:b+1])
    pp = next((r for r in rows if r.get('n_prompt', 0) > 0 and r.get('n_gen', 0) == 0), None)
    tg = next((r for r in rows if r.get('n_prompt', 0) == 0 and r.get('n_gen', 0) > 0), None)
    return pp, tg

def metric(path, key):
    m = re.search(rf'{re.escape(key)}=([^\s]+)', read(path))
    return float(m.group(1)) if m else float('nan')

pp, tg = llama_rows(llama_log)
llama_pp = float(pp['avg_ts']) if pp else float('nan')
llama_tg = float(tg['avg_ts']) if tg else float('nan')
cogni_pp = metric(cogni_pp_log, 'prefill_p50_tok_s')
cogni_tg = metric(cogni_tg_log, 'decode_p50_tok_s')
print('engine\tpp_tok_s\ttg_body_tok_s')
print(f'llama.cpp\t{llama_pp:.3f}\t{llama_tg:.3f}')
print(f'CogniGemma\t{cogni_pp:.3f}\t{cogni_tg:.3f}')
if llama_pp == llama_pp and llama_tg == llama_tg:
    print(f'ratio_cogni_over_llama\t{cogni_pp/llama_pp:.3f}\t{cogni_tg/llama_tg:.3f}')
PY

echo "logs=$OUT_DIR" >&2
