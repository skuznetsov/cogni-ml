require "option_parser"
require "./support/hy_nomas_block47_cache_probe"

source = ""
source_sha = ""
oracle = ""
oracle_sha = ""
cache = ""
report = ""
index = -1
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "Setup-only block4..7 native GPU BF16 cache conversion and exact reload."
  parser.on("--source PATH", "Pinned original F32 source subset") { |v| source = v }
  parser.on("--source-sha256 SHA", "Trusted original subset digest") { |v| source_sha = v }
  parser.on("--oracle PATH", "Independent upstream MPS BF16 oracle") { |v| oracle = v }
  parser.on("--oracle-sha256 SHA", "Trusted independent oracle digest") { |v| oracle_sha = v }
  parser.on("--block-index N", "Exact block4 through block7") { |v| index = v.to_i }
  parser.on("--cache PATH", "Fresh native GPU-produced BF16 cache") { |v| cache = v }
  parser.on("--report PATH", "Fresh qualification report") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate setup GPU buffer budget") { |v| budget = v.to_i64 }
end
unless ARGV.empty? && {source, source_sha, oracle, oracle_sha, cache, report}.all? { |v| !v.empty? } && {4, 5, 6, 7}.includes?(index) && budget > 0
  raise ArgumentError.new("ordered source/oracle digests, exact block4..7, fresh cache/report and positive budget required")
end
exit(HyNomasBlock47CacheProbe.run(source, source_sha, oracle, oracle_sha, index, cache, report, budget) ? 0 : 1)
