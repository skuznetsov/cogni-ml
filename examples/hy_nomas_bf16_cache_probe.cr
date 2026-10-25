require "option_parser"
require "./support/hy_nomas_bf16_cache_probe"

reference = ""
reference_sha = ""
conditioning = ""
oracle = ""
oracle_sha = ""
cache = ""
report = ""
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "One pretrained Hy-Nomas block: GPU BF16 cache setup/reload and upstream parity; no generation."
  parser.on("--reference PATH", "Authenticated saved upstream intermediate reference") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted upstream reference digest") { |v| reference_sha = v }
  parser.on("--conditioning PATH", "Pinned saved conditioning fixture") { |v| conditioning = v }
  parser.on("--oracle PATH", "Independent MPS BF16 weight safetensors oracle") { |v| oracle = v }
  parser.on("--oracle-sha256 SHA", "Trusted BF16 weight oracle digest") { |v| oracle_sha = v }
  parser.on("--cache PATH", "Fresh native GPU-produced BF16 safetensors derivative") { |v| cache = v }
  parser.on("--report PATH", "Fresh qualification report") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate named-buffer allowance, external guard required") { |v| budget = v.to_i64 }
end
unless ARGV.empty? && {reference, reference_sha, conditioning, oracle, oracle_sha, cache, report}.all? { |v| !v.empty? } && budget > 0
  raise ArgumentError.new("reference/digest, conditioning, oracle/digest, fresh cache/report and positive budget required")
end
exit(HyNomasBF16CacheProbe.run(reference, reference_sha, conditioning, oracle, oracle_sha, cache, report, budget) ? 0 : 1)
