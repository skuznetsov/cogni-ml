require "option_parser"
require "./support/hy_nomas_bf16_serial_probe"

reference = ""
reference_sha = ""
conditioning = ""
caches = ["", ""]
cache_shas = ["", ""]
oracles = ["", ""]
oracle_shas = ["", ""]
report = ""
budget = 0_i64
negative = false
OptionParser.parse do |parser|
  parser.banner = "Authenticated BF16 caches for pretrained Hy-Nomas blocks 0 to 1; no generation."
  parser.on("--reference PATH", "Saved upstream two-block intermediate reference") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted upstream reference digest") { |v| reference_sha = v }
  parser.on("--conditioning PATH", "Pinned saved conditioning fixture") { |v| conditioning = v }
  parser.on("--cache0 PATH", "Qualified native BF16 block-0 derivative") { |v| caches[0] = v }
  parser.on("--cache1 PATH", "Qualified native BF16 block-1 derivative") { |v| caches[1] = v }
  parser.on("--cache0-sha256 SHA", "Trusted block-0 cache digest") { |v| cache_shas[0] = v }
  parser.on("--cache1-sha256 SHA", "Trusted block-1 cache digest") { |v| cache_shas[1] = v }
  parser.on("--oracle0 PATH", "Independent MPS BF16 block-0 weight oracle") { |v| oracles[0] = v }
  parser.on("--oracle1 PATH", "Independent MPS BF16 block-1 weight oracle") { |v| oracles[1] = v }
  parser.on("--oracle0-sha256 SHA", "Trusted block-0 oracle digest") { |v| oracle_shas[0] = v }
  parser.on("--oracle1-sha256 SHA", "Trusted block-1 oracle digest") { |v| oracle_shas[1] = v }
  parser.on("--report PATH", "Fresh qualification report") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate named buffer budget, external guard required") { |v| budget = v.to_i64 }
  parser.on("--successor-block0-control", "Use block 0 weights twice and require successor mismatch") { negative = true }
end
unless ARGV.empty? && budget > 0 && {reference, reference_sha, conditioning, report}.all? { |v| !v.empty? } &&
       (caches + cache_shas + oracles + oracle_shas).all? { |v| !v.empty? }
  raise ArgumentError.new("reference/digest, conditioning, two cache/oracle pairs with digests, fresh report and positive budget required")
end
sources = 2.times.map { |i| {path: caches[i], sha256: cache_shas[i]} }.to_a
oracle_sources = 2.times.map { |i| {path: oracles[i], sha256: oracle_shas[i]} }.to_a
exit(HyNomasBF16SerialProbe.run(reference, reference_sha, conditioning,
  sources, oracle_sources, report, budget, negative) ? 0 : 1)
