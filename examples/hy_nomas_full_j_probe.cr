require "option_parser"
require "./support/hy_nomas_full_j_probe"

reference = ""
sha = ""
caches = ["", ""]
digests = ["", ""]
report = ""
budget = 0_i64
mode = "positive"
OptionParser.parse do |parser|
  parser.banner = "Bounded full-J pretrained blocks 0 to 1; selected RoPE/outputs only."
  parser.on("--reference PATH", "Fresh upstream selected-tensor reference") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted reference SHA256") { |v| sha = v }
  parser.on("--cache0 PATH", "Qualified BF16 block 0") { |v| caches[0] = v }
  parser.on("--cache1 PATH", "Qualified BF16 block 1") { |v| caches[1] = v }
  parser.on("--cache0-sha256 SHA", "Qualified block-0 digest") { |v| digests[0] = v }
  parser.on("--cache1-sha256 SHA", "Qualified block-1 digest") { |v| digests[1] = v }
  parser.on("--report PATH", "Fresh JSON report") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate named buffer budget; external guard required") { |v| budget = v.to_i64 }
  parser.on("--identity-rope-control", "Require detection of identity RoPE") { mode = "identity-rope" }
  parser.on("--successor-block0-control", "Require detection of block 0 used twice") { mode = "successor-block0" }
end
unless ARGV.empty? && budget > 0 && {reference, sha, report}.all? { |v| !v.empty? } &&
       (caches + digests).all? { |v| !v.empty? }
  raise ArgumentError.new("reference/digest, ordered cache pairs, fresh report and positive budget required")
end
sources = 2.times.map { |i| {path: caches[i], sha256: digests[i]} }.to_a
exit(HyNomasFullJProbe.run(reference, sha, sources, report, budget, mode) ? 0 : 1)
