#!/usr/bin/env bash
# Measure one bounded QBit Native block in an isolated clickhouse-local store.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SAFE="${COGNI_RUN_SAFE:-$ROOT/scripts/run_safe.sh}"
CLICKHOUSE_BIN="${CLICKHOUSE_BIN:-/Users/sergey/.local/bin/clickhouse}"
MAX_INPUT_MIB="${QWEN_QBIT_CH_MAX_INPUT_MIB:-256}"
MAX_TREE_MIB="${QWEN_QBIT_CH_MAX_TREE_MIB:-2048}"
TIMEOUT_SEC="${QWEN_QBIT_CH_TIMEOUT_SEC:-120}"
MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-12}"
READ_REPEATS="${QWEN_QBIT_CH_READ_REPEATS:-5}"

usage() {
  echo "Usage: $0 PATH_TO_QBIT_NATIVE_BLOCK" >&2
}

if [[ "$#" -ne 1 ]]; then
  usage
  exit 2
fi
if [[ ! -x "$RUN_SAFE" ]]; then
  echo "safe runner is not executable: $RUN_SAFE" >&2
  exit 2
fi
if [[ ! -x "$CLICKHOUSE_BIN" ]]; then
  echo "ClickHouse binary is not executable: $CLICKHOUSE_BIN" >&2
  exit 2
fi
if ! [[ "$MAX_INPUT_MIB" =~ ^[1-9][0-9]*$ && "$MAX_TREE_MIB" =~ ^[1-9][0-9]*$ && "$TIMEOUT_SEC" =~ ^[1-9][0-9]*$ && "$READ_REPEATS" =~ ^[1-9][0-9]*$ && "$MIN_FREE_PCT" =~ ^[0-9]+$ ]]; then
  echo "QBit ClickHouse limits must be integers (zero is allowed only for the memory-pressure floor)" >&2
  exit 2
fi
if ((MAX_INPUT_MIB > 1024 || MAX_TREE_MIB > 8192 || TIMEOUT_SEC > 600 || READ_REPEATS > 20 || MIN_FREE_PCT > 99)); then
  echo "QBit ClickHouse limits exceed the diagnostic safety envelope" >&2
  exit 2
fi

INPUT_ARG="$1"
if [[ ! -f "$INPUT_ARG" ]]; then
  echo "QBit Native block not found: $INPUT_ARG" >&2
  exit 2
fi
INPUT_DIR="$(cd "$(dirname "$INPUT_ARG")" && pwd)"
INPUT_PATH="$INPUT_DIR/$(basename "$INPUT_ARG")"
if [[ "$(uname -s)" == "Darwin" ]]; then
  INPUT_BYTES="$(stat -f %z "$INPUT_PATH")"
else
  INPUT_BYTES="$(stat -c %s "$INPUT_PATH")"
fi
MAX_INPUT_BYTES=$((MAX_INPUT_MIB * 1024 * 1024))
if [[ "$INPUT_BYTES" -le 0 || "$INPUT_BYTES" -gt "$MAX_INPUT_BYTES" ]]; then
  echo "QBit Native block size $INPUT_BYTES is outside 1..$MAX_INPUT_BYTES bytes" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d /private/tmp/cogni-qbit-ch.XXXXXX)"
cleanup() {
  case "$WORK_DIR" in
    /private/tmp/cogni-qbit-ch.*|/tmp/cogni-qbit-ch.*) rm -rf -- "$WORK_DIR" ;;
    *) echo "refusing to remove unexpected work directory: $WORK_DIR" >&2 ;;
  esac
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

STORE_PATH="$WORK_DIR/store"
ROUNDTRIP_PATH="$WORK_DIR/roundtrip.native"
READ_LOG="$WORK_DIR/read.log"
READ_TIMES_PATH="$WORK_DIR/read-times.txt"
PART_STATS_PATH="$WORK_DIR/part-stats.tsv"
mkdir -p "$STORE_PATH"

SCHEMA="cache_id UInt64, layer Int32, kind UInt8, tile UInt32, value_count UInt16, mean Float32, sigma Float32, codes QBit(Int8, 1024)"

run_clickhouse() {
  RUN_SAFE_PASSTHROUGH_STDIO=1 \
    COGNI_RUN_SAFE_MIN_FREE_PCT="$MIN_FREE_PCT" \
    "$RUN_SAFE" "$CLICKHOUSE_BIN" "$TIMEOUT_SEC" "$MAX_TREE_MIB" "$@"
}

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --multiquery \
  --query "CREATE TABLE qbit_cache (cache_id UInt64, layer Int32, kind UInt8, tile UInt32, value_count UInt16, mean Float32, sigma Float32, codes QBit(Int8, 1024)) ENGINE=MergeTree ORDER BY (cache_id, layer, kind, tile)"

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --input-format Native \
  --structure "$SCHEMA" \
  --query "INSERT INTO qbit_cache SELECT * FROM table" < "$INPUT_PATH"

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --query "SELECT sum(rows), sum(data_compressed_bytes), sum(data_uncompressed_bytes), sum(bytes_on_disk), count() FROM system.parts WHERE database = currentDatabase() AND table = 'qbit_cache' AND active FORMAT TSVRaw" \
  > "$PART_STATS_PATH"
PART_STATS="$(<"$PART_STATS_PATH")"

IFS=$'\t' read -r PART_ROWS PART_COMPRESSED_BYTES PART_UNCOMPRESSED_BYTES PART_BYTES_ON_DISK PART_COUNT <<< "$PART_STATS"
for VALUE in "$PART_ROWS" "$PART_COMPRESSED_BYTES" "$PART_UNCOMPRESSED_BYTES" "$PART_BYTES_ON_DISK" "$PART_COUNT"; do
  if ! [[ "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid active-part statistics: $PART_STATS" >&2
    exit 1
  fi
done

for ((READ_INDEX = 1; READ_INDEX <= READ_REPEATS; READ_INDEX++)); do
  run_clickhouse local \
    --path "$STORE_PATH" \
    --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
    --max_block_size 1000000 \
    --preferred_block_size_bytes "$MAX_INPUT_BYTES" \
    --time \
    --query "SELECT cache_id, layer, kind, tile, value_count, mean, sigma, codes FROM qbit_cache ORDER BY cache_id, layer, kind, tile FORMAT Native" \
    > "$ROUNDTRIP_PATH" 2> "$READ_LOG"

  READ_SECONDS="$(awk '/^[0-9]+([.][0-9]+)?$/ { value=$0 } END { print value }' "$READ_LOG")"
  if [[ -z "$READ_SECONDS" ]]; then
    echo "ClickHouse did not report query time" >&2
    sed -n '1,120p' "$READ_LOG" >&2
    exit 1
  fi
  echo "$READ_SECONDS" >> "$READ_TIMES_PATH"

  if ! cmp -s "$INPUT_PATH" "$ROUNDTRIP_PATH"; then
    echo "QBit Native round trip changed the block on read $READ_INDEX" >&2
    shasum -a 256 "$INPUT_PATH" "$ROUNDTRIP_PATH" >&2
    exit 1
  fi
done

if [[ "$(uname -s)" == "Darwin" ]]; then
  RESPONSE_BYTES="$(stat -f %z "$ROUNDTRIP_PATH")"
else
  RESPONSE_BYTES="$(stat -c %s "$ROUNDTRIP_PATH")"
fi
INPUT_SHA256="$(shasum -a 256 "$INPUT_PATH" | awk '{print $1}')"
COMPRESSED_RATIO="$(awk -v compressed="$PART_COMPRESSED_BYTES" -v logical="$INPUT_BYTES" 'BEGIN { printf "%.6f", compressed / logical }')"
ON_DISK_RATIO="$(awk -v physical="$PART_BYTES_ON_DISK" -v logical="$INPUT_BYTES" 'BEGIN { printf "%.6f", physical / logical }')"
READ_FIRST_MS="$(awk 'NR == 1 { printf "%.3f", $1 * 1000.0 }' "$READ_TIMES_PATH")"
READ_MEDIAN_MS="$(sort -n "$READ_TIMES_PATH" | awk '{ values[NR]=$1 } END { middle=int((NR + 1) / 2); if (NR % 2) value=values[middle]; else value=(values[middle] + values[middle + 1]) / 2.0; printf "%.3f", value * 1000.0 }')"

echo "qwen_qbit_clickhouse_probe"
echo "  clickhouse_version=$($CLICKHOUSE_BIN --version | head -1)"
echo "  input_sha256=$INPUT_SHA256"
echo "  logical_native_bytes=$INPUT_BYTES"
echo "  part_rows=$PART_ROWS part_count=$PART_COUNT"
echo "  part_compressed_bytes=$PART_COMPRESSED_BYTES ratio_vs_native=$COMPRESSED_RATIO"
echo "  part_uncompressed_bytes=$PART_UNCOMPRESSED_BYTES"
echo "  part_bytes_on_disk=$PART_BYTES_ON_DISK ratio_vs_native=$ON_DISK_RATIO"
echo "  response_native_bytes=$RESPONSE_BYTES read_repeats=$READ_REPEATS read_first_ms=$READ_FIRST_MS read_median_ms=$READ_MEDIAN_MS"
echo "  exact_native_roundtrip=true"
