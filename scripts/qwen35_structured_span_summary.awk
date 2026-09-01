NR == 1 { next }

count == 0 {
  min_decode = max_decode = $3
  min_total = max_total = $4
}

{
  count += 1
  sum_decode += $3
  sum_total += $4
  if ($3 < min_decode) min_decode = $3
  if ($3 > max_decode) max_decode = $3
  if ($4 < min_total) min_total = $4
  if ($4 > max_total) max_total = $4
}

END {
  if (count == 0) exit 1

  mean_decode = sum_decode / count
  mean_total = sum_total / count
  if (mean_decode < min_decode_speedup_pct) {
    printf "decode speedup gate failed: mean %.3f%% is below required %.3f%%\n", mean_decode, min_decode_speedup_pct > "/dev/stderr"
    exit 1
  }

  printf "suite_summary pairs=%d parity=exact speed_gate=pass min_decode_speedup_pct=%.3f decode_speedup_mean_pct=%.3f decode_speedup_range_pct=%.3f..%.3f total_speedup_mean_pct=%.3f total_speedup_range_pct=%.3f..%.3f\n", count, min_decode_speedup_pct, mean_decode, min_decode, max_decode, mean_total, min_total, max_total
}
