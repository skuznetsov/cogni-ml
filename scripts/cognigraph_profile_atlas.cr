#!/usr/bin/env crystal
# Parse Qwen35Metal.Profile text logs into a compact graph/phase atlas.
# This is observational only: it does not execute model code or change routing.

require "option_parser"

log_path = ""
limit = 12
json = false

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: scripts/cognigraph_profile_atlas.cr --log PATH [--limit 12] [--json]"
  p.on("--log PATH", "Profile log produced by a CogniQwen/CogniGemma --profile run") { |v| log_path = v }
  p.on("--limit N", "Rows to print per section (default: 12)") { |v| limit = v.to_i }
  p.on("--json", "Emit machine-readable JSON-ish output") { json = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

abort "--log is required" if log_path.empty?
abort "log not found: #{log_path}" unless File.exists?(log_path)

record GroupRow, name : String, calls : Int32, encode_ms : Float64, wait_ms : Float64, read_ms : Float64, upload_mib : Float64, readback_mib : Float64
record MatmulRow, name : String, calls : Int32, weight_mib : Float64, pct : Float64
record ConversionRow, name : String, calls : Int32, traffic_mib : Float64, pct : Float64

text = File.read(log_path)
groups = [] of GroupRow
matmuls = [] of MatmulRow
conversions = [] of ConversionRow
syncs = nil.as(Int32?)
mode = ""

text.each_line do |line|
  stripped = line.strip
  case stripped
  when "grouped command buffers:"
    mode = "groups"
    next
  when "matmul shapes:"
    mode = "matmuls"
    next
  when "conversion kernels:"
    mode = "conversions"
    next
  end

  if m = stripped.match(/^total metal syncs:\s+(\d+)/)
    syncs = m[1].to_i
    next
  end

  case mode
  when "groups"
    if m = stripped.match(/^(.+?)\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms\s+upload\s+([0-9.]+) MiB\s+readback\s+([0-9.]+) MiB/)
      groups << GroupRow.new(m[1].strip, m[2].to_i, m[3].to_f, m[4].to_f, m[5].to_f, m[6].to_f, m[7].to_f)
    elsif !stripped.starts_with?("grouped") && !stripped.empty? && !stripped.includes?("calls")
      mode = ""
    end
  when "matmuls"
    if m = stripped.match(/^(.+?)\s+(\d+) calls\s+([0-9.]+) MiB logical weights\s+([0-9.]+)%/)
      matmuls << MatmulRow.new(m[1].strip, m[2].to_i, m[3].to_f, m[4].to_f)
    elsif stripped.starts_with?("matmul") || stripped.starts_with?("logical traffic")
      mode = ""
    end
  when "conversions"
    if m = stripped.match(/^(.+?)\s+(\d+) calls\s+([0-9.]+) MiB logical traffic\s+([0-9.]+)%/)
      conversions << ConversionRow.new(m[1].strip, m[2].to_i, m[3].to_f, m[4].to_f)
    elsif stripped.starts_with?("conversion") || stripped.starts_with?("logical traffic")
      mode = ""
    end
  end
end

total_wait = groups.sum(&.wait_ms)
total_encode = groups.sum(&.encode_ms)
total_read = groups.sum(&.read_ms)
total_weight = matmuls.sum(&.weight_mib)
total_conversion = conversions.sum(&.traffic_mib)
dominant_group = groups.max_by?(&.wait_ms)
dominant_matmul = matmuls.max_by?(&.weight_mib)
dominant_conversion = conversions.max_by?(&.traffic_mib)
tied_groups = dominant_group ? groups.count { |g| dominant_group.not_nil!.wait_ms > 0 && g.wait_ms >= dominant_group.not_nil!.wait_ms * 0.80 } : 0
tied_matmuls = dominant_matmul ? matmuls.count { |m| dominant_matmul.not_nil!.weight_mib > 0 && m.weight_mib >= dominant_matmul.not_nil!.weight_mib * 0.80 } : 0
conflict_count = (syncs || groups.size) + groups.count { |g| g.readback_mib > 0 || g.upload_mib > 0 }

if json
  puts "{"
  puts "  \"log\": #{log_path.inspect},"
  puts "  \"potential\": {\"dominant_wait_bucket\": #{dominant_group.try(&.name).inspect}, \"tied_wait_routes\": #{tied_groups}, \"conflict_or_sync_count\": #{conflict_count}, \"remaining_work_mib\": #{(total_weight + total_conversion).round(3)}},"
  puts "  \"totals\": {\"encode_ms\": #{total_encode.round(3)}, \"wait_ms\": #{total_wait.round(3)}, \"read_ms\": #{total_read.round(3)}, \"matmul_mib\": #{total_weight.round(3)}, \"conversion_mib\": #{total_conversion.round(3)}, \"syncs\": #{(syncs || 0)}}"
  puts "}"
  exit
end

puts "CogniGraph profile atlas"
puts "  log: #{log_path}"
puts "  totals: encode=#{total_encode.round(3)}ms wait=#{total_wait.round(3)}ms read=#{total_read.round(3)}ms syncs=#{syncs || groups.size}"
puts "  traffic: matmul=#{total_weight.round(3)}MiB conversion=#{total_conversion.round(3)}MiB"
puts
puts "LTP/WBA potential Phi = (dominant_wait_bucket, tied_dominant_routes, conflict_or_sync_count, remaining_work)"
puts "  Phi=(#{dominant_group.try(&.name) || "none"}, #{tied_groups}, #{conflict_count}, #{(total_weight + total_conversion).round(3)}MiB)"
puts "  dominant_matmul=#{dominant_matmul.try(&.name) || "none"} tied_matmuls=#{tied_matmuls}"
puts "  dominant_conversion=#{dominant_conversion.try(&.name) || "none"}"
puts
puts "Top command-buffer groups by wait:"
groups.sort_by { |g| {-g.wait_ms, g.name} }.first(limit).each do |g|
  puts "  #{g.name}: calls=#{g.calls} encode=#{g.encode_ms.round(3)}ms wait=#{g.wait_ms.round(3)}ms read=#{g.read_ms.round(3)}ms upload=#{g.upload_mib.round(3)}MiB readback=#{g.readback_mib.round(3)}MiB"
end
puts
puts "Top matmul shapes by logical weight traffic:"
matmuls.sort_by { |m| {-m.weight_mib, m.name} }.first(limit).each do |m|
  puts "  #{m.name}: calls=#{m.calls} weight=#{m.weight_mib.round(3)}MiB pct=#{m.pct.round(2)}"
end
unless conversions.empty?
  puts
  puts "Top conversion kernels by logical traffic:"
  conversions.sort_by { |c| {-c.traffic_mib, c.name} }.first(limit).each do |c|
    puts "  #{c.name}: calls=#{c.calls} traffic=#{c.traffic_mib.round(3)}MiB pct=#{c.pct.round(2)}"
  end
end
