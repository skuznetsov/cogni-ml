#!/usr/bin/env crystal

# Offline falsifier for exact early pruning of a Q6_K Qwen output head.
#
# The candidate reads a block-aligned prefix of every vocabulary row, fully
# evaluates a small seed set, and bounds each remaining suffix with Cauchy-
# Schwarz. Rows whose upper bound cannot enter the exact top-2 are discarded.
# This probe deliberately charges only weight and one Float32 sidecar read; a
# production two-pass Metal route would also pay dispatch, scratch, compaction,
# and conservative floating-point-bound overhead.

require "option_parser"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL         = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_PROMPT        = "Implement a bounded retry helper in Crystal and explain the invariant."
QK_K                  = 256
Q6_BLOCK_BYTES        = 210
SIDECAR_BYTES_PER_ROW =   4

record Top2, best_id : Int32, best : Float64, second_id : Int32, second : Float64

private def parse_positive_i32(value : String, option : String) : Int32
  parsed = value.to_i32
  raise ArgumentError.new("#{option} must be positive") unless parsed > 0
  parsed
end

private def parse_cuts(value : String) : Array(Float64)
  cuts = value.split(',').map(&.strip).reject(&.empty?).map(&.to_f64)
  raise ArgumentError.new("--cuts must contain at least one fraction") if cuts.empty?
  cuts.each do |cut|
    raise ArgumentError.new("--cuts values must be finite and in (0, 1)") unless cut.finite? && cut > 0.0 && cut < 1.0
  end
  cuts.uniq.sort
end

private def top2(scores : Array(Float64), ids : Array(Int32)? = nil) : Top2
  best = -Float64::INFINITY
  second = -Float64::INFINITY
  best_id = 0_i32
  second_id = 0_i32
  count = ids ? ids.not_nil!.size : scores.size
  raise ArgumentError.new("top2 requires at least two rows") if count < 2

  count.times do |index|
    id = ids ? ids.not_nil![index] : index.to_i32
    value = scores[id]
    if value > best || (value == best && id < best_id)
      second = best
      second_id = best_id
      best = value
      best_id = id
    elsif id != best_id && (value > second || (value == second && id < second_id))
      second = value
      second_id = id
    end
  end
  Top2.new(best_id, best, second_id, second)
end

private def better_score?(scores : Array(Float64), left : Int32, right : Int32) : Bool
  lv = scores[left]
  rv = scores[right]
  lv > rv || (lv == rv && left < right)
end

private def top_seed_ids(scores : Array(Float64), count : Int32) : Array(Int32)
  ids = Array(Int32).new(scores.size) { |index| index.to_i32 }
  ids.sort! do |left, right|
    if better_score?(scores, left, right)
      -1
    elsif better_score?(scores, right, left)
      1
    else
      0
    end
  end
  ids[0, Math.min(count, ids.size)]
end

private def fp16_at(raw : Bytes, offset : Int32) : Float64
  bits = raw[offset].to_u16 | (raw[offset + 1].to_u16 << 8)
  ML::GGUF::Dequant.fp16_to_f32(bits).to_f64
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
prompt = DEFAULT_PROMPT
sample_count = 3_i32
seed_count = 128_i32
cut_fractions = [0.25_f64, 0.50_f64, 0.75_f64, 0.90_f64]
head_share = 0.066_f64
required_whole_gain = 0.03_f64

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_q6_head_bound_probe [options]"
  parser.on("--model PATH", "Qwen GGUF model path") { |value| model = value }
  parser.on("--prompt TEXT", "Prompt used to obtain real final hidden rows") { |value| prompt = value }
  parser.on("--samples N", "Consecutive real hidden rows (default: 3)") { |value| sample_count = parse_positive_i32(value, "--samples") }
  parser.on("--seed N", "Rows fully evaluated before pruning (default: 128)") { |value| seed_count = parse_positive_i32(value, "--seed") }
  parser.on("--cuts LIST", "Comma-separated input fractions (default: 0.25,0.50,0.75,0.90)") { |value| cut_fractions = parse_cuts(value) }
  parser.on("--head-share F", "Measured whole-decode share of the output head (default: 0.066)") { |value| head_share = value.to_f64 }
  parser.on("--required-whole-gain F", "Promotion threshold as a fraction (default: 0.03)") { |value| required_whole_gain = value.to_f64 }
  parser.on("-h", "--help", "Show help") do
    puts parser
    exit
  end
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--head-share must be finite and in (0, 1)" unless head_share.finite? && head_share > 0.0 && head_share < 1.0
raise "--required-whole-gain must be finite and positive" unless required_whole_gain.finite? && required_whole_gain > 0.0

tokenizer_gguf = ML::GGUF::GGUFFile.new(model, mmap_tensors: false)
tokenizer = begin
  ML::GGUF::Qwen35Tokenizer.from_gguf(tokenizer_gguf, model)
ensure
  tokenizer_gguf.close
end

weights = ML::GGUF::Qwen35Weights.from_gguf(model)
begin
  hp = weights.hparams
  output = weights.output
  raise "output head must be Q6_K, got #{output.type}" unless output.type.q6_k?
  raise "output in_dim must be divisible by #{QK_K}" unless output.in_dim % QK_K == 0
  raise "output dimensions do not match hparams" unless output.in_dim == hp.n_embd

  blocks_per_row = output.in_dim // QK_K
  row_bytes = blocks_per_row * Q6_BLOCK_BYTES
  expected_bytes = output.out_dim.to_i64 * row_bytes.to_i64
  raise "output tensor byte size mismatch: #{output.raw.size} != #{expected_bytes}" unless output.raw.size.to_i64 == expected_bytes

  cut_blocks = cut_fractions.map do |fraction|
    Math.max(1, Math.min(blocks_per_row - 1, (blocks_per_row.to_f64 * fraction).round.to_i32))
  end.uniq.sort
  raise "no block-aligned cuts remain" if cut_blocks.empty?

  ids = tokenizer.encode(prompt)
  raise "prompt tokenized to zero tokens" if ids.empty?
  max_seq = ids.size + sample_count + 4
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)

  puts "Qwen Q6 output-head exact-bound falsifier"
  puts "model=#{File.basename(model)} prompt_tokens=#{ids.size} samples=#{sample_count} seed=#{seed_count}"
  puts "shape=#{output.out_dim}x#{output.in_dim} blocks_per_row=#{blocks_per_row} cuts=#{cut_blocks.join(',')} head_share=#{head_share}"

  hidden = ML::GGUF::Qwen35CPU.prefill_tokens_last_hidden(weights, ids, 0, state)
  normalized = Array(Array(Float32)).new(sample_count)
  routed_top2 = Array(Top2).new(sample_count)
  token_id = 0_i32
  pos = ids.size.to_i32

  sample_count.times do |sample|
    best_id, best, second_id, second = ML::GGUF::Qwen35CPU.hidden_top2(weights, hidden)
    routed_top2 << Top2.new(best_id, best.to_f64, second_id, second.to_f64)
    normalized << ML::GGUF::Qwen35CPU.rms_norm(hidden, weights.output_norm, hp.rms_eps)
    puts "sample=#{sample} routed_top1=#{best_id} routed_top2=#{second_id} margin=#{(best - second).round(6)}"
    token_id = best_id
    if sample + 1 < sample_count
      hidden = ML::GGUF::Qwen35CPU.forward_hidden(weights, token_id, pos, state)
      pos += 1
    end
  end

  exact_scores = Array(Array(Float64)).new(sample_count) do
    Array(Float64).new(output.out_dim, 0.0_f64)
  end
  partial_scores = Array(Array(Array(Float64))).new(sample_count) do
    Array(Array(Float64)).new(cut_blocks.size) do
      Array(Float64).new(output.out_dim, 0.0_f64)
    end
  end
  suffix_norms = Array(Array(Float64)).new(cut_blocks.size) do
    Array(Float64).new(output.out_dim, 0.0_f64)
  end

  block_norms = Array(Float64).new(blocks_per_row, 0.0_f64)
  sample_block_dots = Array(Array(Float64)).new(sample_count) do
    Array(Float64).new(blocks_per_row, 0.0_f64)
  end
  raw = output.raw
  scan_started = Time.instant

  output.out_dim.times do |row|
    block_norms.fill(0.0_f64)
    sample_block_dots.each(&.fill(0.0_f64))
    row_offset = row * row_bytes

    blocks_per_row.times do |block|
      block_offset = row_offset + block * Q6_BLOCK_BYTES
      d = fp16_at(raw, block_offset + 208)
      block_base = block * QK_K

      2.times do |half|
        ql_offset = block_offset + half * 64
        qh_offset = block_offset + 128 + half * 32
        scales_offset = block_offset + 192 + half * 8
        value_base = block_base + half * 128

        32.times do |lane|
          scale_index = lane // 16
          ql0 = raw[ql_offset + lane]
          ql1 = raw[ql_offset + lane + 32]
          qh = raw[qh_offset + lane]
          q1 = ((ql0.to_i32 & 0x0f) | (((qh.to_i32 >> 0) & 3) << 4)) - 32
          q2 = ((ql1.to_i32 & 0x0f) | (((qh.to_i32 >> 2) & 3) << 4)) - 32
          q3 = ((ql0.to_i32 >> 4) | (((qh.to_i32 >> 4) & 3) << 4)) - 32
          q4 = ((ql1.to_i32 >> 4) | (((qh.to_i32 >> 6) & 3) << 4)) - 32

          w1 = d * raw[scales_offset + scale_index].unsafe_as(Int8).to_f64 * q1.to_f64
          w2 = d * raw[scales_offset + scale_index + 2].unsafe_as(Int8).to_f64 * q2.to_f64
          w3 = d * raw[scales_offset + scale_index + 4].unsafe_as(Int8).to_f64 * q3.to_f64
          w4 = d * raw[scales_offset + scale_index + 6].unsafe_as(Int8).to_f64 * q4.to_f64
          j1 = value_base + lane
          j2 = j1 + 32
          j3 = j1 + 64
          j4 = j1 + 96

          block_norms[block] += w1 * w1 + w2 * w2 + w3 * w3 + w4 * w4
          sample_count.times do |sample|
            x = normalized[sample]
            sample_block_dots[sample][block] +=
              x[j1].to_f64 * w1 + x[j2].to_f64 * w2 +
                x[j3].to_f64 * w3 + x[j4].to_f64 * w4
          end
        end
      end
    end

    total_norm_sq = block_norms.sum
    sample_count.times do |sample|
      dots = sample_block_dots[sample]
      exact_scores[sample][row] = dots.sum
      cut_blocks.each_with_index do |cut, cut_index|
        partial_scores[sample][cut_index][row] = dots[0, cut].sum
      end
    end
    cut_blocks.each_with_index do |cut, cut_index|
      prefix_norm_sq = block_norms[0, cut].sum
      suffix_norms[cut_index][row] = Math.sqrt(Math.max(0.0_f64, total_norm_sq - prefix_norm_sq))
    end

    if (row + 1) % 32768 == 0 || row + 1 == output.out_dim
      elapsed = (Time.instant - scan_started).total_seconds
      puts "scan_rows=#{row + 1}/#{output.out_dim} elapsed_s=#{elapsed.round(3)}"
    end
  end

  raw_weight_bytes = expected_bytes.to_f64
  required_head_saving = required_whole_gain / head_share
  all_sound = true
  all_top2 = true
  best_min_head_saving = -Float64::INFINITY
  best_cut = 0_i32
  best_min_oracle_head_saving = -Float64::INFINITY
  best_oracle_cut = 0_i32

  cut_blocks.each_with_index do |cut, cut_index|
    per_sample_savings = [] of Float64
    per_sample_oracle_savings = [] of Float64
    sample_count.times do |sample|
      scores = exact_scores[sample]
      partial = partial_scores[sample][cut_index]
      seed_ids = top_seed_ids(partial, Math.min(seed_count, output.out_dim))
      seed_top2 = top2(scores, seed_ids)
      hidden_suffix_norm_sq = 0.0_f64
      (cut * QK_K...output.in_dim).each do |index|
        value = normalized[sample][index].to_f64
        hidden_suffix_norm_sq += value * value
      end
      hidden_suffix_norm = Math.sqrt(hidden_suffix_norm_sq)

      survivors = [] of Int32
      oracle_survivors = 0_i32
      violations = 0_i32
      exact_top2 = top2(scores)
      output.out_dim.times do |row|
        upper = partial[row] + hidden_suffix_norm * suffix_norms[cut_index][row]
        tolerance = 1.0e-9_f64 * (1.0_f64 + upper.abs + scores[row].abs)
        violations += 1 if scores[row] > upper + tolerance
        survivors << row.to_i32 if upper + tolerance >= seed_top2.second
        oracle_survivors += 1 if upper + tolerance >= exact_top2.second
      end

      survivor_top2 = top2(scores, survivors)
      routed = routed_top2[sample]
      top2_ok = survivor_top2.best_id == exact_top2.best_id &&
                survivor_top2.second_id == exact_top2.second_id
      routed_ids_ok = routed.best_id == exact_top2.best_id && routed.second_id == exact_top2.second_id
      all_sound &&= violations == 0
      all_top2 &&= top2_ok && routed_ids_ok

      prefix_bytes = output.out_dim.to_i64 * cut.to_i64 * Q6_BLOCK_BYTES
      suffix_bytes = survivors.size.to_i64 * (blocks_per_row - cut).to_i64 * Q6_BLOCK_BYTES
      sidecar_bytes = output.out_dim.to_i64 * SIDECAR_BYTES_PER_ROW
      optimistic_bytes = prefix_bytes + suffix_bytes + sidecar_bytes
      head_saving = 1.0_f64 - optimistic_bytes.to_f64 / raw_weight_bytes
      whole_ceiling = head_saving * head_share
      oracle_suffix_bytes = oracle_survivors.to_i64 * (blocks_per_row - cut).to_i64 * Q6_BLOCK_BYTES
      oracle_bytes = prefix_bytes + oracle_suffix_bytes + sidecar_bytes
      oracle_head_saving = 1.0_f64 - oracle_bytes.to_f64 / raw_weight_bytes
      oracle_whole_ceiling = oracle_head_saving * head_share
      per_sample_savings << head_saving
      per_sample_oracle_savings << oracle_head_saving

      puts "cut_blocks=#{cut} sample=#{sample} survivors=#{survivors.size}/#{output.out_dim} " +
           "pruned_pct=#{((1.0 - survivors.size.to_f64 / output.out_dim) * 100.0).round(3)} " +
           "optimistic_head_saving_pct=#{(head_saving * 100.0).round(3)} " +
           "whole_decode_ceiling_pct=#{(whole_ceiling * 100.0).round(3)} " +
           "oracle_survivors=#{oracle_survivors} " +
           "oracle_head_saving_pct=#{(oracle_head_saving * 100.0).round(3)} " +
           "oracle_whole_ceiling_pct=#{(oracle_whole_ceiling * 100.0).round(3)} " +
           "bound_violations=#{violations} exact_top2=#{exact_top2.best_id},#{exact_top2.second_id} " +
           "survivor_top2=#{survivor_top2.best_id},#{survivor_top2.second_id} " +
           "routed_top2=#{routed.best_id},#{routed.second_id} top2_ok=#{top2_ok && routed_ids_ok}"
    end

    min_saving = per_sample_savings.min
    if min_saving > best_min_head_saving
      best_min_head_saving = min_saving
      best_cut = cut
    end
    min_oracle_saving = per_sample_oracle_savings.min
    if min_oracle_saving > best_min_oracle_head_saving
      best_min_oracle_head_saving = min_oracle_saving
      best_oracle_cut = cut
    end
  end

  promotable = all_sound && all_top2 && best_min_oracle_head_saving >= required_head_saving
  puts "summary best_cut_blocks=#{best_cut} best_min_head_saving_pct=#{(best_min_head_saving * 100.0).round(3)} " +
       "best_oracle_cut_blocks=#{best_oracle_cut} best_min_oracle_head_saving_pct=#{(best_min_oracle_head_saving * 100.0).round(3)} " +
       "required_head_saving_pct=#{(required_head_saving * 100.0).round(3)} " +
       "bound_sound=#{all_sound} top2_exact=#{all_top2} promotable=#{promotable}"
  puts "caveat=optimistic_byte_ceiling_excludes_second_dispatch_compaction_scratch_and_float_bound_inflation"
ensure
  weights.close
end
