#!/usr/bin/env crystal

require "option_parser"

record CounterRow, name : String, calls : Int32, encode_ms : Float64, wait_ms : Float64, read_ms : Float64
record TrafficRow, name : String, calls : Int32, mib : Float64, pct : Float64, kind : String
record TraceRow, name : String, calls : Int32, ms : Float64

class ProfileAtlas
  property gemv : CounterRow?
  property gemm : CounterRow?
  property dn : CounterRow?
  property attn : CounterRow?
  property wave : CounterRow?
  property cpu_fallback : Int32 = 0
  property total_syncs : Int32 = 0
  getter groups = [] of CounterRow
  getter matmuls = [] of TrafficRow
  getter conversions = [] of TrafficRow
  getter traces = [] of TraceRow

  def add_counter(kind : String, row : CounterRow)
    case kind
    when "gemv" then @gemv = row
    when "gemm" then @gemm = row
    when "dn"   then @dn = row
    when "attn" then @attn = row
    when "wave" then @wave = row
    end
  end

  def total_matmul_mib : Float64
    @matmuls.sum(&.mib)
  end

  def total_conversion_mib : Float64
    @conversions.sum(&.mib)
  end

  def total_logical_mib : Float64
    total_matmul_mib + total_conversion_mib
  end

  def total_wait_ms : Float64
    [@gemv, @gemm, @dn, @attn, @wave].compact.sum(&.wait_ms)
  end

  def grouped_wait_ms : Float64
    @groups.sum(&.wait_ms)
  end
end

def median(xs : Array(Float64)) : Float64
  return 0.0 if xs.empty?

  sorted = xs.sort
  mid = sorted.size // 2
  if sorted.size.odd?
    sorted[mid]
  else
    (sorted[mid - 1] + sorted[mid]) / 2.0
  end
end

input_paths = [] of String
top_n = 12
show_tsv = false

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_profile_atlas.cr [profile.log ...] [--top=N] [--tsv]"
  p.on("--top=N", "Number of rows per ranked section (default: 12)") { |v| top_n = v.to_i }
  p.on("--tsv", "Emit machine-readable TSV rows") { show_tsv = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
  p.unknown_args { |args| input_paths = args }
end

text = if input_paths.empty?
         STDIN.gets_to_end
       else
         input_paths.map { |path| File.read(path) }.join("\n")
       end

atlas = ProfileAtlas.new
section = nil.as(Symbol?)

text.each_line do |line|
  case line
  when /gemv:\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms/
    atlas.add_counter("gemv", CounterRow.new("gemv", $1.to_i, $2.to_f, $3.to_f, $4.to_f))
  when /gemm:\s+(\d+) calls\s+wait\s+([0-9.]+) ms/
    atlas.add_counter("gemm", CounterRow.new("gemm", $1.to_i, 0.0, $2.to_f, 0.0))
  when /dn:\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms/
    atlas.add_counter("dn", CounterRow.new("dn", $1.to_i, $2.to_f, $3.to_f, $4.to_f))
  when /attn:\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms/
    atlas.add_counter("attn", CounterRow.new("attn", $1.to_i, $2.to_f, $3.to_f, $4.to_f))
  when /wave:\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms/
    atlas.add_counter("wave", CounterRow.new("wave", $1.to_i, $2.to_f, $3.to_f, $4.to_f))
  when /wave encode trace:/
    section = :trace
  when /grouped command buffers:/
    section = :groups
  when /matmul shapes:/
    section = :matmuls
  when /conversion kernels:/
    section = :conversions
  when /logical traffic mix:/
    section = nil
  when /cpu_fallback matvecs:\s+(\d+)/
    atlas.cpu_fallback = $1.to_i
  when /total metal syncs:\s+(\d+)/
    atlas.total_syncs = $1.to_i
  else
    case section
    when :groups
      if line =~ /^\s{4}(.+?)\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms\s*$/
        atlas.groups << CounterRow.new($1.strip, $2.to_i, $3.to_f, $4.to_f, $5.to_f)
      end
    when :matmuls
      if line =~ /^\s{4}(.+?)\s+(\d+) calls\s+([0-9.]+) MiB logical weights\s+([0-9.]+)%\s*$/
        atlas.matmuls << TrafficRow.new($1.strip, $2.to_i, $3.to_f, $4.to_f, "matmul")
      end
    when :conversions
      if line =~ /^\s{4}(.+?)\s+(\d+) calls\s+([0-9.]+) MiB logical traffic\s+([0-9.]+)%\s*$/
        atlas.conversions << TrafficRow.new($1.strip, $2.to_i, $3.to_f, $4.to_f, "conversion")
      end
    when :trace
      if line =~ /^\s{4}(.+?)\s+(\d+) calls\s+([0-9.]+) ms\s*$/
        atlas.traces << TraceRow.new($1.strip, $2.to_i, $3.to_f)
      end
    end
  end
end

if show_tsv
  puts "kind\tname\tcalls\tms_or_mib\tpct"
  [atlas.gemv, atlas.gemm, atlas.dn, atlas.attn, atlas.wave].compact.each do |row|
    puts ["wait", row.name, row.calls, row.wait_ms, ""].join('\t')
  end
  atlas.groups.each { |row| puts ["group", row.name, row.calls, row.wait_ms, ""].join('\t') }
  atlas.matmuls.each { |row| puts ["matmul", row.name, row.calls, row.mib, row.pct].join('\t') }
  atlas.conversions.each { |row| puts ["conversion", row.name, row.calls, row.mib, row.pct].join('\t') }
  exit
end

puts "Qwen35 profile atlas"
printf "  total_syncs=%d total_wait_ms=%.2f grouped_wait_ms=%.2f logical_mib=%.2f matmul_mib=%.2f conversion_mib=%.2f cpu_fallback=%d\n",
  atlas.total_syncs, atlas.total_wait_ms, atlas.grouped_wait_ms, atlas.total_logical_mib, atlas.total_matmul_mib, atlas.total_conversion_mib, atlas.cpu_fallback

puts "\nWait buckets"
[atlas.gemv, atlas.gemm, atlas.dn, atlas.attn, atlas.wave].compact.sort_by { |row| -row.wait_ms }.each do |row|
  pct = atlas.total_wait_ms > 0 ? row.wait_ms * 100.0 / atlas.total_wait_ms : 0.0
  printf "  %-6s %4d calls wait=%8.2f ms encode=%7.2f read=%7.2f pct=%5.1f%%\n",
    row.name, row.calls, row.wait_ms, row.encode_ms, row.read_ms, pct
end

unless atlas.groups.empty?
  waits = atlas.groups.map(&.wait_ms).sort
  group_median = median(waits)
  puts "\nGrouped command-buffer waits"
  atlas.groups.sort_by { |row| {-row.wait_ms, row.name} }.first(top_n).each do |row|
    ratio = group_median > 0 ? row.wait_ms / group_median : 0.0
    pct_group = atlas.grouped_wait_ms > 0 ? row.wait_ms * 100.0 / atlas.grouped_wait_ms : 0.0
    printf "  %-24s %3d calls wait=%8.2f ms pct_group=%5.1f%% ratio_to_median=%4.2f\n",
      row.name, row.calls, row.wait_ms, pct_group, ratio
  end
end

unless atlas.matmuls.empty?
  puts "\nTop logical matmul traffic"
  atlas.matmuls.sort_by { |row| {-row.mib, row.name} }.first(top_n).each do |row|
    printf "  %-54s %3d calls %9.2f MiB %5.1f%%\n", row.name, row.calls, row.mib, row.pct
  end
end

unless atlas.conversions.empty?
  puts "\nTop conversion traffic"
  atlas.conversions.sort_by { |row| {-row.mib, row.name} }.first(top_n).each do |row|
    pct_total = atlas.total_logical_mib > 0 ? row.mib * 100.0 / atlas.total_logical_mib : 0.0
    printf "  %-54s %3d calls %9.2f MiB %5.1f%% of conversions %5.1f%% total\n",
      row.name, row.calls, row.mib, row.pct, pct_total
  end
end

puts "\nLTP/WBA candidate windows"
if atlas.cpu_fallback > 0
  puts "  Spike: cpu_fallback>0. Window=CPU matvec escape; corridor=offloaded row span; potential=(fallback_calls,syncs,bytes). Legal move=cover missing Metal route or fail closed earlier."
end
if atlas.total_conversion_mib > 0 && atlas.total_logical_mib > 0
  conversion_pct = atlas.total_conversion_mib * 100.0 / atlas.total_logical_mib
  if conversion_pct >= 15.0
    puts "  Ladder: conversion traffic is material (#{conversion_pct.round(1)}%). Window=F32/F16 staging; corridor=activation batch span; potential=(conversion_mib,conversion_calls,syncs,wall). Legal move=fuse conversion into producer/consumer or keep native staged type without changing exact math."
  end
end
if top = atlas.matmuls.sort_by { |row| {-row.mib, row.name} }.first?
  puts "  Ladder: dominant matmul scope '#{top.name}' carries #{top.mib.round(2)} MiB. Window=top logical-weight reader; corridor=layer/batch band; potential=(logical_weight_mib,calls,barriers). Legal move=prepack/fuse/re-route only if profile wall and parity improve."
end
unless atlas.groups.empty?
  waits = atlas.groups.map(&.wait_ms).sort
  group_median = median(waits)
  max = waits.last
  if group_median > 0 && max / group_median < 1.25
    puts "  Collapse: grouped waits are flat (max/median=#{(max / group_median).round(2)}). This refutes a single bad group; reduce repeated per-layer work instead of boundary reshuffle."
  else
    hot = atlas.groups.max_by(&.wait_ms)
    puts "  Diamond: hot group '#{hot.name}' is above median. Window=group conflict; corridor=fused command buffer; potential=(group_wait_skew,syncs,bytes). Legal move=split/fuse only with paired wall proof."
  end
end
puts "  Dual frame: if a candidate does not lower the potential under paired timing, fall back to exact baseline/profile-only mode; do not stack speculative complexity."
