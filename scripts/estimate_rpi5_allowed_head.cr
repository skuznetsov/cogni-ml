#!/usr/bin/env crystal

def usage
  STDERR.puts "usage: crystal scripts/estimate_rpi5_allowed_head.cr FULL_HEAD_MS VOCAB_ROWS CPU_MS [allowed_counts_csv]"
  exit 2
end

full_head_ms = (ARGV[0]? || usage).to_f64
vocab_rows = (ARGV[1]? || usage).to_f64
cpu_ms = (ARGV[2]? || usage).to_f64
counts = (ARGV[3]? || "1,4,16,64,256,1024,4096,8192,16384,32768,65536").split(",").map(&.to_f64)

per_row_us = full_head_ms * 1000.0 / vocab_rows
cpu_equal_rows = cpu_ms / full_head_ms * vocab_rows

puts "full_head_ms=#{full_head_ms}"
puts "vocab_rows=#{vocab_rows.to_i64}"
puts "cpu_ms_per_token=#{cpu_ms}"
puts "head_per_row_us=#{per_row_us.round(4)}"
puts "allowed_rows_equal_cpu_token=#{cpu_equal_rows.round(1)}"
puts
puts "allowed_rows\test_head_ms\tpct_of_full_head\tpct_of_cpu_token"
counts.each do |n|
  ms = full_head_ms * n / vocab_rows
  puts "#{n.to_i64}\t#{ms.round(4)}\t#{(ms / full_head_ms * 100.0).round(4)}%\t#{(ms / cpu_ms * 100.0).round(4)}%"
end
