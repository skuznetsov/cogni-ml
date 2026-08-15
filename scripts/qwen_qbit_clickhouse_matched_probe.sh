#!/usr/bin/env bash
# Compare complete QBit+exact-KV and recurrent-INT8 cache states in bounded local ClickHouse parts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SAFE="${COGNI_RUN_SAFE:-$ROOT/scripts/run_safe.sh}"
CLICKHOUSE_BIN="${CLICKHOUSE_BIN:-/Users/sergey/.local/bin/clickhouse}"
MAX_INPUT_MIB="${QWEN_QBIT_CH_MATCHED_MAX_INPUT_MIB:-256}"
MAX_TOTAL_MIB="${QWEN_QBIT_CH_MATCHED_MAX_TOTAL_MIB:-512}"
MAX_TREE_MIB="${QWEN_QBIT_CH_MATCHED_MAX_TREE_MIB:-2048}"
TIMEOUT_SEC="${QWEN_QBIT_CH_MATCHED_TIMEOUT_SEC:-120}"
MIN_FREE_PCT="${COGNI_RUN_SAFE_MIN_FREE_PCT:-12}"
READ_REPEATS="${QWEN_QBIT_CH_MATCHED_READ_REPEATS:-5}"
EXPORT_DIR="${QWEN_QBIT_CH_MATCHED_EXPORT_DIR:-}"

usage() {
  echo "Usage: $0 QBIT_RECURRENT_NATIVE EXACT_KV_ARTIFACT FULL_INT8_ARTIFACT" >&2
}

if [[ "$#" -ne 3 ]]; then
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
if ! [[ "$MAX_INPUT_MIB" =~ ^[1-9][0-9]*$ && "$MAX_TOTAL_MIB" =~ ^[1-9][0-9]*$ && "$MAX_TREE_MIB" =~ ^[1-9][0-9]*$ && "$TIMEOUT_SEC" =~ ^[1-9][0-9]*$ && "$MIN_FREE_PCT" =~ ^[0-9]+$ && "$READ_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "matched QBit ClickHouse limits must be integers (zero is allowed only for the memory-pressure floor)" >&2
  exit 2
fi
if ((MAX_INPUT_MIB > 1024 || MAX_TOTAL_MIB > 2048 || MAX_TREE_MIB > 8192 || TIMEOUT_SEC > 600 || MIN_FREE_PCT > 99 || READ_REPEATS > 20)); then
  echo "matched QBit ClickHouse limits exceed the diagnostic safety envelope" >&2
  exit 2
fi

absolute_file() {
  local input="$1"
  if [[ ! -f "$input" ]]; then
    echo "diagnostic input not found: $input" >&2
    return 2
  fi
  local input_dir
  input_dir="$(cd "$(dirname "$input")" && pwd)"
  printf '%s/%s\n' "$input_dir" "$(basename "$input")"
}

file_bytes() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    stat -f %z "$1"
  else
    stat -c %s "$1"
  fi
}

QBIT_PATH="$(absolute_file "$1")"
KV_PATH="$(absolute_file "$2")"
INT8_PATH="$(absolute_file "$3")"
if [[ "$QBIT_PATH" == "$KV_PATH" || "$QBIT_PATH" == "$INT8_PATH" || "$KV_PATH" == "$INT8_PATH" ]]; then
  echo "matched diagnostic inputs must be distinct files" >&2
  exit 2
fi
QBIT_BYTES="$(file_bytes "$QBIT_PATH")"
KV_BYTES="$(file_bytes "$KV_PATH")"
INT8_BYTES="$(file_bytes "$INT8_PATH")"
MAX_INPUT_BYTES=$((MAX_INPUT_MIB * 1024 * 1024))
MAX_TOTAL_BYTES=$((MAX_TOTAL_MIB * 1024 * 1024))
for INPUT_BYTES in "$QBIT_BYTES" "$KV_BYTES" "$INT8_BYTES"; do
  if [[ "$INPUT_BYTES" -le 0 || "$INPUT_BYTES" -gt "$MAX_INPUT_BYTES" ]]; then
    echo "diagnostic input size $INPUT_BYTES is outside 1..$MAX_INPUT_BYTES bytes" >&2
    exit 2
  fi
done
TOTAL_INPUT_BYTES=$((QBIT_BYTES + KV_BYTES + INT8_BYTES))
if ((TOTAL_INPUT_BYTES > MAX_TOTAL_BYTES)); then
  echo "combined diagnostic input size $TOTAL_INPUT_BYTES exceeds $MAX_TOTAL_BYTES bytes" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d /private/tmp/cogni-qbit-matched.XXXXXX)"
cleanup() {
  case "$WORK_DIR" in
    /private/tmp/cogni-qbit-matched.*|/tmp/cogni-qbit-matched.*) rm -rf -- "$WORK_DIR" ;;
    *) echo "refusing to remove unexpected work directory: $WORK_DIR" >&2 ;;
  esac
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

STORE_PATH="$WORK_DIR/store"
PART_STATS_PATH="$WORK_DIR/part-stats.tsv"
QBIT_ROUNDTRIP_PATH="$WORK_DIR/qbit-roundtrip.native"
KV_ROUNDTRIP_PATH="$WORK_DIR/kv-roundtrip.qkv"
INT8_ROUNDTRIP_PATH="$WORK_DIR/int8-roundtrip.qkv"
QBIT_NATURAL_PATH="$WORK_DIR/qbit-natural.native"
QBIT_READ_TIMES_PATH="$WORK_DIR/qbit-read-times.txt"
KV_READ_TIMES_PATH="$WORK_DIR/kv-read-times.txt"
INT8_READ_TIMES_PATH="$WORK_DIR/int8-read-times.txt"
mkdir -p "$STORE_PATH"

QBIT_SCHEMA="cache_id UInt64, layer Int32, kind UInt8, tile UInt32, value_count UInt16, mean Float32, sigma Float32, codes QBit(Int8, 1024)"
BLOB_SCHEMA="payload String"

run_clickhouse() {
  RUN_SAFE_PASSTHROUGH_STDIO=1 \
    COGNI_RUN_SAFE_MIN_FREE_PCT="$MIN_FREE_PCT" \
    "$RUN_SAFE" "$CLICKHOUSE_BIN" "$TIMEOUT_SEC" "$MAX_TREE_MIB" "$@"
}

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --multiquery \
  --query "
    CREATE TABLE qbit_recurrent (
      cache_id UInt64, layer Int32, kind UInt8, tile UInt32,
      value_count UInt16, mean Float32, sigma Float32,
      codes QBit(Int8, 1024)
    ) ENGINE=MergeTree ORDER BY (cache_id, layer, kind, tile);
    CREATE TABLE qbit_kv (cache_id UInt64, payload String CODEC(LZ4))
      ENGINE=MergeTree ORDER BY cache_id;
    CREATE TABLE int8_cache (cache_id UInt64, payload String CODEC(LZ4))
      ENGINE=MergeTree ORDER BY cache_id;
  "

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --input-format Native \
  --structure "$QBIT_SCHEMA" \
  --query "INSERT INTO qbit_recurrent SELECT * FROM table" < "$QBIT_PATH"

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --input-format RawBLOB \
  --structure "$BLOB_SCHEMA" \
  --query "INSERT INTO qbit_kv SELECT toUInt64(0), payload FROM table" < "$KV_PATH"

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --input-format RawBLOB \
  --structure "$BLOB_SCHEMA" \
  --query "INSERT INTO int8_cache SELECT toUInt64(0), payload FROM table" < "$INT8_PATH"

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --query "
    SELECT table, sum(rows), sum(data_compressed_bytes), sum(data_uncompressed_bytes), sum(bytes_on_disk), count()
    FROM system.parts
    WHERE database = currentDatabase() AND table IN ('qbit_recurrent', 'qbit_kv', 'int8_cache') AND active
    GROUP BY table
    ORDER BY table
    FORMAT TSVRaw
  " > "$PART_STATS_PATH"

part_stats() {
  local table="$1"
  awk -F '\t' -v table="$table" '$1 == table { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 }' "$PART_STATS_PATH"
}

IFS=$'\t' read -r INT8_ROWS INT8_COMPRESSED INT8_UNCOMPRESSED INT8_DISK INT8_PARTS <<< "$(part_stats int8_cache)"
IFS=$'\t' read -r KV_ROWS KV_COMPRESSED KV_UNCOMPRESSED KV_DISK KV_PARTS <<< "$(part_stats qbit_kv)"
IFS=$'\t' read -r QBIT_ROWS QBIT_COMPRESSED QBIT_UNCOMPRESSED QBIT_DISK QBIT_REC_PARTS <<< "$(part_stats qbit_recurrent)"
for VALUE in "$INT8_ROWS" "$INT8_COMPRESSED" "$INT8_UNCOMPRESSED" "$INT8_DISK" "$INT8_PARTS" \
             "$KV_ROWS" "$KV_COMPRESSED" "$KV_UNCOMPRESSED" "$KV_DISK" "$KV_PARTS" \
             "$QBIT_ROWS" "$QBIT_COMPRESSED" "$QBIT_UNCOMPRESSED" "$QBIT_DISK" "$QBIT_REC_PARTS"; do
  if ! [[ "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid active-part statistics:" >&2
    sed -n '1,20p' "$PART_STATS_PATH" >&2
    exit 1
  fi
done
if [[ "$INT8_ROWS" -ne 1 || "$KV_ROWS" -ne 1 ]]; then
  echo "blob inputs must each produce exactly one ClickHouse row" >&2
  exit 1
fi

run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --max_block_size 1000000 \
  --preferred_block_size_bytes "$MAX_INPUT_BYTES" \
  --query "SELECT cache_id, layer, kind, tile, value_count, mean, sigma, codes FROM qbit_recurrent ORDER BY cache_id, layer, kind, tile FORMAT Native" \
  > "$QBIT_ROUNDTRIP_PATH"
run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --query "SELECT payload FROM qbit_kv WHERE cache_id = 0 FORMAT RawBLOB" \
  > "$KV_ROUNDTRIP_PATH"
run_clickhouse local \
  --path "$STORE_PATH" \
  --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
  --query "SELECT payload FROM int8_cache WHERE cache_id = 0 FORMAT RawBLOB" \
  > "$INT8_ROUNDTRIP_PATH"

timed_read() {
  local output_path="$1"
  local times_path="$2"
  local query="$3"
  local read_log="$WORK_DIR/read.log"
  run_clickhouse local \
    --path "$STORE_PATH" \
    --max_memory_usage "$((MAX_TREE_MIB * 1024 * 1024))" \
    --time \
    --query "$query" \
    > "$output_path" 2> "$read_log"
  local read_seconds
  read_seconds="$(awk '/^[0-9]+([.][0-9]+)?$/ { value=$0 } END { print value }' "$read_log")"
  if [[ -z "$read_seconds" ]]; then
    echo "ClickHouse did not report query time" >&2
    sed -n '1,120p' "$read_log" >&2
    exit 1
  fi
  echo "$read_seconds" >> "$times_path"
}

for ((READ_INDEX = 1; READ_INDEX <= READ_REPEATS; READ_INDEX++)); do
  timed_read \
    "$QBIT_NATURAL_PATH" \
    "$QBIT_READ_TIMES_PATH" \
    "SELECT cache_id, layer, kind, tile, value_count, mean, sigma, codes FROM qbit_recurrent ORDER BY cache_id, layer, kind, tile FORMAT Native"
  timed_read \
    "$KV_ROUNDTRIP_PATH" \
    "$KV_READ_TIMES_PATH" \
    "SELECT payload FROM qbit_kv WHERE cache_id = 0 FORMAT RawBLOB"
  timed_read \
    "$INT8_ROUNDTRIP_PATH" \
    "$INT8_READ_TIMES_PATH" \
    "SELECT payload FROM int8_cache WHERE cache_id = 0 FORMAT RawBLOB"
done

assert_roundtrip() {
  local source_path="$1"
  local roundtrip_path="$2"
  if ! cmp -s "$source_path" "$roundtrip_path"; then
    echo "ClickHouse round trip changed $(basename "$source_path")" >&2
    shasum -a 256 "$source_path" "$roundtrip_path" >&2
    exit 1
  fi
}
assert_roundtrip "$QBIT_PATH" "$QBIT_ROUNDTRIP_PATH"
assert_roundtrip "$KV_PATH" "$KV_ROUNDTRIP_PATH"
assert_roundtrip "$INT8_PATH" "$INT8_ROUNDTRIP_PATH"

QBIT_LOGICAL=$((QBIT_BYTES + KV_BYTES))
QBIT_PHYSICAL_COMPRESSED=$((QBIT_COMPRESSED + KV_COMPRESSED))
QBIT_PHYSICAL_DISK=$((QBIT_DISK + KV_DISK))
QBIT_TOTAL_PARTS=$((QBIT_REC_PARTS + KV_PARTS))
COMPRESSED_RATIO="$(awk -v qbit="$QBIT_PHYSICAL_COMPRESSED" -v int8="$INT8_COMPRESSED" 'BEGIN { printf "%.6f", qbit / int8 }')"
DISK_RATIO="$(awk -v qbit="$QBIT_PHYSICAL_DISK" -v int8="$INT8_DISK" 'BEGIN { printf "%.6f", qbit / int8 }')"
DISK_SAVING_PCT="$(awk -v qbit="$QBIT_PHYSICAL_DISK" -v int8="$INT8_DISK" 'BEGIN { printf "%.3f", (1.0 - qbit / int8) * 100.0 }')"
median_ms() {
  sort -n "$1" | awk '{ values[NR]=$1 } END { middle=int((NR + 1) / 2); if (NR % 2) value=values[middle]; else value=(values[middle] + values[middle + 1]) / 2.0; printf "%.3f", value * 1000.0 }'
}
QBIT_READ_MEDIAN_MS="$(median_ms "$QBIT_READ_TIMES_PATH")"
KV_READ_MEDIAN_MS="$(median_ms "$KV_READ_TIMES_PATH")"
INT8_READ_MEDIAN_MS="$(median_ms "$INT8_READ_TIMES_PATH")"
QBIT_COMPLETE_READ_MEDIAN_MS="$(awk -v recurrent="$QBIT_READ_MEDIAN_MS" -v kv="$KV_READ_MEDIAN_MS" 'BEGIN { printf "%.3f", recurrent + kv }')"

if [[ -n "$EXPORT_DIR" ]]; then
  if [[ ! -d "$EXPORT_DIR" ]]; then
    echo "matched export directory does not exist: $EXPORT_DIR" >&2
    exit 2
  fi
  EXPORT_DIR="$(cd "$EXPORT_DIR" && pwd)"
  for TARGET in "$EXPORT_DIR/qbit-natural.native" "$EXPORT_DIR/exact-kv.qkv" "$EXPORT_DIR/recurrent-int8.qkv"; do
    if [[ -e "$TARGET" ]]; then
      echo "refusing to overwrite matched export: $TARGET" >&2
      exit 2
    fi
  done
  cp "$QBIT_NATURAL_PATH" "$EXPORT_DIR/qbit-natural.native"
  cp "$KV_ROUNDTRIP_PATH" "$EXPORT_DIR/exact-kv.qkv"
  cp "$INT8_ROUNDTRIP_PATH" "$EXPORT_DIR/recurrent-int8.qkv"
fi

echo "qwen_qbit_clickhouse_matched_probe"
echo "  clickhouse_version=$($CLICKHOUSE_BIN --version | head -1)"
echo "  qbit_recurrent_logical_bytes=$QBIT_BYTES rows=$QBIT_ROWS compressed_bytes=$QBIT_COMPRESSED uncompressed_bytes=$QBIT_UNCOMPRESSED disk_bytes=$QBIT_DISK parts=$QBIT_REC_PARTS"
echo "  qbit_kv_logical_bytes=$KV_BYTES rows=$KV_ROWS compressed_bytes=$KV_COMPRESSED uncompressed_bytes=$KV_UNCOMPRESSED disk_bytes=$KV_DISK parts=$KV_PARTS codec=LZ4"
echo "  qbit_complete_logical_bytes=$QBIT_LOGICAL compressed_bytes=$QBIT_PHYSICAL_COMPRESSED disk_bytes=$QBIT_PHYSICAL_DISK parts=$QBIT_TOTAL_PARTS"
echo "  int8_complete_logical_bytes=$INT8_BYTES rows=$INT8_ROWS compressed_bytes=$INT8_COMPRESSED uncompressed_bytes=$INT8_UNCOMPRESSED disk_bytes=$INT8_DISK parts=$INT8_PARTS codec=LZ4"
echo "  qbit_vs_int8_compressed_ratio=$COMPRESSED_RATIO qbit_vs_int8_disk_ratio=$DISK_RATIO disk_saving_pct=$DISK_SAVING_PCT"
echo "  read_repeats=$READ_REPEATS qbit_recurrent_read_median_ms=$QBIT_READ_MEDIAN_MS qbit_kv_read_median_ms=$KV_READ_MEDIAN_MS qbit_complete_sequential_read_median_ms=$QBIT_COMPLETE_READ_MEDIAN_MS int8_complete_read_median_ms=$INT8_READ_MEDIAN_MS"
echo "  exact_native_roundtrip=true exact_kv_roundtrip=true exact_int8_roundtrip=true"
if [[ -n "$EXPORT_DIR" ]]; then
  echo "  exported_qbit_native=$EXPORT_DIR/qbit-natural.native exported_exact_kv=$EXPORT_DIR/exact-kv.qkv exported_int8=$EXPORT_DIR/recurrent-int8.qkv"
fi
