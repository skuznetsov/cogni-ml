require "option_parser"
require "./support/hy_nomas_pretrained_serial_probe"

reference = ""
reference_sha256 = ""
conditioning = ""
conditioning_sha256 = ""
report = ""
named_budget = 0_i64
successor_block0_control = false

OptionParser.parse do |parser|
  parser.banner = "Authenticated two-block Hy-Nomas serial workspace replay; no full generation."
  parser.on("--reference PATH", "Fresh upstream two-block reference JSON") { |value| reference = value }
  parser.on("--reference-sha256 SHA256", "Trusted upstream reference JSON digest") { |value| reference_sha256 = value }
  parser.on("--conditioning PATH", "Authenticated saved text-conditioning fixture") { |value| conditioning = value }
  parser.on("--conditioning-sha256 SHA256", "Pinned conditioning fixture digest") { |value| conditioning_sha256 = value }
  parser.on("--report PATH", "Fresh-only post-terminal native report") { |value| report = value }
  parser.on("--max-named-bytes N", "Aggregate named Metal allocation budget (must leave 10% headroom)") { |value| named_budget = value.to_i64 }
  parser.on("--successor-block0-control", "Feed block 0 weights to successor; require block 1 mismatch") { successor_block0_control = true }
end

unless !reference.empty? && !reference_sha256.empty? && !conditioning.empty? &&
       !conditioning_sha256.empty? && !report.empty? && named_budget > 0
  raise ArgumentError.new("reference, trusted digests, conditioning, report and positive named budget are required")
end
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("reference digest must be lowercase SHA256") unless reference_sha256.matches?(/\A[0-9a-f]{64}\z/)
raise ArgumentError.new("conditioning digest must be lowercase SHA256") unless conditioning_sha256.matches?(/\A[0-9a-f]{64}\z/)
raise ArgumentError.new("conditioning digest is not the pinned saved fixture") unless conditioning_sha256 == HyNomasPretrainedSerialProbe::PINNED_CONDITIONING_SHA
raise ArgumentError.new("refusing to replace native report") if File.exists?(report) || File.symlink?(report)

passed = HyNomasPretrainedSerialProbe.run(reference, reference_sha256,
  conditioning, conditioning_sha256, report, named_budget,
  successor_block0_control: successor_block0_control)
exit(passed ? 0 : 1)
