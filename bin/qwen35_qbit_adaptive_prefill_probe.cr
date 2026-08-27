require "option_parser"

require "../src/ml/core/buffer"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen_qbit_adaptive_kv"
require "../src/ml/gguf/qwen_qbit_adaptive_resident_kv"

# Bounded synthetic attribution probe for adaptive QBit prefill. It separates
# append-only packing from attention over an existing packed prefix and never
# loads model weights.
module Qwen35QBitAdaptivePrefillProbe
  extend self

  MAX_PREFIX  = 4096
  MAX_REPEATS =   10
  CHUNK       =   64

  def median(samples : Array(Float64)) : Float64
    ordered = samples.sort
    ordered[ordered.size // 2]
  end

  def tier_for(name : String) : ML::GGUF::QwenQBitAdaptiveKV::Tier
    case name
    when "p4"   then ML::GGUF::QwenQBitAdaptiveKV::Tier::P4
    when "bf16" then ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16
    when "f32"  then ML::GGUF::QwenQBitAdaptiveKV::Tier::F32
    else
      raise ArgumentError.new("tier must be p4, bf16, or f32")
    end
  end

  def append_until(cache : ML::GGUF::QwenQBitAdaptiveResidentKV::Cache,
                   k_source : ML::MetalBuffer,
                   v_source : ML::MetalBuffer,
                   target : Int32) : Nil
    while cache.cache_len < target
      count = Math.min(CHUNK, target - cache.cache_len)
      ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
        cache, k_source, v_source, count,
      )
    end
  end

  def run(prefixes : Array(Int32), repeats : Int32) : Nil
    raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?
    unless repeats > 0 && repeats <= MAX_REPEATS
      raise ArgumentError.new("repeats must be in 1..#{MAX_REPEATS}")
    end
    unless !prefixes.empty? && prefixes == prefixes.sort && prefixes.uniq == prefixes
      raise ArgumentError.new("prefixes must be unique and sorted")
    end
    unless prefixes.all? { |prefix| prefix > 0 && prefix <= MAX_PREFIX && prefix % CHUNK == 0 }
      raise ArgumentError.new("prefixes must be 64-aligned and in 64..#{MAX_PREFIX}")
    end

    n_head = 24
    n_head_kv = 4
    head_dim = 256
    heads_per_group = 6
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    q_values = CHUNK * n_head * head_dim
    kv_values = CHUNK * n_head_kv * head_dim
    rng = Random.new(0xA77B10C_u64)
    q = Array(Float32).new(q_values) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(q_values) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    k = Array(Float32).new(kv_values) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    v = Array(Float32).new(kv_values) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    q_buffer = ML::MetalBuffer.from_array(q)
    gate_buffer = ML::MetalBuffer.from_array(gate)
    k_buffer = ML::MetalBuffer.from_array(k)
    v_buffer = ML::MetalBuffer.from_array(v)
    output = ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32))

    puts "tier prefix_start prefix_end chunk pack_only_ms prefill_pack_ms attention_estimate_ms"
    begin
      ["p4", "bf16", "f32"].each do |tier_name|
        tier = tier_for(tier_name)
        pack_plan = ML::GGUF::QwenQBitAdaptiveKV.plan(
          Array.new((repeats + 1) * CHUNK * n_head_kv, tier),
        )
        pack_cache = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
          pack_plan, pack_plan, (repeats + 1) * CHUNK, n_head_kv, head_dim,
        )
        pack_samples = Array(Float64).new(repeats)
        begin
          ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
            pack_cache, k_buffer, v_buffer, CHUNK,
          )
          repeats.times do
            started = Time.instant
            ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
              pack_cache, k_buffer, v_buffer, CHUNK,
            )
            pack_samples << (Time.instant - started).total_milliseconds
          end
        ensure
          pack_cache.release
        end
        pack_ms = median(pack_samples)

        capacity = prefixes.last + repeats * CHUNK
        plan = ML::GGUF::QwenQBitAdaptiveKV.plan(
          Array.new(capacity * n_head_kv, tier),
        )
        cache = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
          plan, plan, capacity, n_head_kv, head_dim,
        )
        begin
          prefixes.each do |prefix|
            append_until(cache, k_buffer, v_buffer, prefix)
            samples = Array(Float64).new(repeats)
            repeats.times do
              started = Time.instant
              ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
                cache,
                q_buffer, gate_buffer, k_buffer, v_buffer, output,
                CHUNK, n_head, heads_per_group, scale,
              )
              samples << (Time.instant - started).total_milliseconds
            end
            prefill_ms = median(samples)
            prefix_end = prefix + (repeats - 1) * CHUNK
            printf "%4s %12d %10d %5d %12.3f %15.3f %21.3f\n",
              tier_name, prefix, prefix_end, CHUNK,
              pack_ms, prefill_ms, prefill_ms - pack_ms
          end
        ensure
          cache.release
        end
      end
    ensure
      q_buffer.release
      gate_buffer.release
      k_buffer.release
      v_buffer.release
      output.release
    end
  end
end

prefixes = [512, 1536, 3072]
repeats = 5
OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_qbit_adaptive_prefill_probe [options]"
  parser.on("--prefixes LIST", "Comma-separated 64-aligned prefixes") do |value|
    prefixes = value.split(',').map(&.to_i)
  end
  parser.on("--repeats N", "Median sample count, 1..10") { |value| repeats = value.to_i }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

Qwen35QBitAdaptivePrefillProbe.run(prefixes, repeats)
