require "option_parser"

require "../src/ml/core/buffer"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen_qbit_adaptive_kv"
require "../src/ml/gguf/qwen_qbit_adaptive_resident_kv"

# Bounded synthetic attribution probe for adaptive QBit prefill. It reports the
# completed prefill/pack/finalize command's GPU interval separately from wall
# time and never loads model weights.
module Qwen35QBitAdaptivePrefillProbe
  extend self

  MAX_PREFIX   =          8192
  MAX_REPEATS  =            10
  SOURCE_CHUNK =            64
  SEED         = 0xA77B10C_u64

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
      count = Math.min(SOURCE_CHUNK, target - cache.cache_len)
      ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
        cache, k_source, v_source, count,
      )
    end
  end

  def timed_prefill_chunk_and_append(
    cache : ML::GGUF::QwenQBitAdaptiveResidentKV::Cache,
    q_source : ML::MetalBuffer,
    gate_source : ML::MetalBuffer,
    k_source : ML::MetalBuffer,
    v_source : ML::MetalBuffer,
    output : ML::MetalBuffer,
    token_count : Int32,
    n_head : Int32,
    heads_per_group : Int32,
    scale : Float32,
  ) : {Float64, Float64}
    started = Time.instant
    command = ML::Metal::CommandBuffer.new
    ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
      command, cache,
      q_source, gate_source, k_source, v_source, output,
      token_count, n_head, heads_per_group, scale,
      expected_start_token: cache.cache_len,
    )
    begin
      ML::GGUF::QwenQBitAdaptiveResidentKV.finalize_pending_append(command, cache)
      gpu_ms = command.commit_and_wait_gpu_elapsed_seconds * 1000.0
      ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(cache, command)
      {(Time.instant - started).total_milliseconds, gpu_ms}
    rescue ex
      if !command.committed? || command.completed?
        ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(cache, command)
      end
      raise ex
    end
  end

  def run(prefixes : Array(Int32), repeats : Int32,
          token_count : Int32, tier_names : Array(String)) : Nil
    raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?
    unless repeats > 0 && repeats <= MAX_REPEATS
      raise ArgumentError.new("repeats must be in 1..#{MAX_REPEATS}")
    end
    unless token_count > 0 && token_count <= SOURCE_CHUNK
      raise ArgumentError.new("chunk must be in 1..#{SOURCE_CHUNK}")
    end
    unless !tier_names.empty? && tier_names.uniq == tier_names
      raise ArgumentError.new("tiers must be unique and non-empty")
    end
    tier_names.each { |name| tier_for(name) }
    unless !prefixes.empty? && prefixes == prefixes.sort && prefixes.uniq == prefixes
      raise ArgumentError.new("prefixes must be unique and sorted")
    end
    unless prefixes.all? { |prefix| prefix > 0 && prefix <= MAX_PREFIX && prefix % SOURCE_CHUNK == 0 }
      raise ArgumentError.new("prefixes must be 64-aligned and in 64..#{MAX_PREFIX}")
    end

    n_head = 24
    n_head_kv = 4
    head_dim = 256
    heads_per_group = 6
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    q_values = SOURCE_CHUNK * n_head * head_dim
    kv_values = SOURCE_CHUNK * n_head_kv * head_dim
    rng = Random.new(SEED)
    q = Array(Float32).new(q_values) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(q_values) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    k = Array(Float32).new(kv_values) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    v = Array(Float32).new(kv_values) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    q_buffer = ML::MetalBuffer.from_array(q)
    gate_buffer = ML::MetalBuffer.from_array(gate)
    k_buffer = ML::MetalBuffer.from_array(k)
    v_buffer = ML::MetalBuffer.from_array(v)
    output = ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32))

    device_name = ML::Metal::Device.instance.name
    tile = ML::GGUF::QwenQBitAdaptiveMetalPolicy.gqa6_tile(
      device_name, ENV["QWEN35_ADAPTIVE_GQA6_TILE"]?,
    )
    dequant_t4_override = ENV["QWEN35_ADAPTIVE_DEQUANT_T4"]?
    dequant_t4_mode = dequant_t4_override.nil? ? "auto" : dequant_t4_override.strip.inspect
    puts %(probe device=#{device_name.inspect} tile=#{tile} dequant_t4_mode=#{dequant_t4_mode} seed=0x#{SEED.to_s(16)} fixed_snapshot=true)
    puts "tier prefix chunk route    t4 pack_wall_ms fused_wall_ms prefill_pack_finalize_gpu_ms non_gpu_ms"
    begin
      tier_names.each do |tier_name|
        tier = tier_for(tier_name)
        pack_plan = ML::GGUF::QwenQBitAdaptiveKV.plan(
          Array.new((repeats + 1) * token_count * n_head_kv, tier),
        )
        pack_cache = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
          pack_plan, pack_plan, (repeats + 1) * token_count, n_head_kv, head_dim,
        )
        pack_samples = Array(Float64).new(repeats)
        begin
          ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
            pack_cache, k_buffer, v_buffer, token_count,
          )
          repeats.times do
            started = Time.instant
            ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
              pack_cache, k_buffer, v_buffer, token_count,
            )
            pack_samples << (Time.instant - started).total_milliseconds
          end
        ensure
          pack_cache.release
        end
        pack_ms = median(pack_samples)

        prefixes.each do |prefix|
          splitk = ML::GGUF::QwenQBitAdaptiveMetalPolicy.decode_splitk?(
            prefix, token_count, true,
            ENV["QWEN35_ADAPTIVE_SPLITK"]?,
            ENV["QWEN35_ADAPTIVE_SPLITK_MIN_CTX"]?,
          )
          route = splitk ? "splitk" : (token_count == 1 ? "serial1" : "prefill")
          automatic_t4 = ML::GGUF::QwenQBitAdaptiveMetalPolicy.automatic_dequant_t4?(
            token_count, splitk,
            tier == ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
            tier == ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16,
          )
          selected_t4 = ML::GGUF::QwenQBitAdaptiveMetalPolicy.dequant_t4?(
            device_name, automatic_t4, dequant_t4_override,
          )
          capacity = prefix + token_count
          plan = ML::GGUF::QwenQBitAdaptiveKV.plan(
            Array.new(capacity * n_head_kv, tier),
          )
          seed_cache = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
            plan, plan, capacity, n_head_kv, head_dim,
          )
          begin
            append_until(seed_cache, k_buffer, v_buffer, prefix)
            snapshot_k, snapshot_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(seed_cache)
          ensure
            seed_cache.release
          end

          warmup_cache = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
            plan, plan, capacity, n_head_kv, head_dim,
          )
          begin
            ML::GGUF::QwenQBitAdaptiveResidentKV.restore_snapshot!(
              warmup_cache, snapshot_k, snapshot_v, prefix,
            )
            timed_prefill_chunk_and_append(
              warmup_cache,
              q_buffer, gate_buffer, k_buffer, v_buffer, output,
              token_count, n_head, heads_per_group, scale,
            )
          ensure
            warmup_cache.release
          end

          wall_samples = Array(Float64).new(repeats)
          gpu_samples = Array(Float64).new(repeats)
          repeats.times do
            cache = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
              plan, plan, capacity, n_head_kv, head_dim,
            )
            begin
              ML::GGUF::QwenQBitAdaptiveResidentKV.restore_snapshot!(
                cache, snapshot_k, snapshot_v, prefix,
              )
              wall_ms, gpu_ms = timed_prefill_chunk_and_append(
                cache,
                q_buffer, gate_buffer, k_buffer, v_buffer, output,
                token_count, n_head, heads_per_group, scale,
              )
              wall_samples << wall_ms
              gpu_samples << gpu_ms
            ensure
              cache.release
            end
          end
          fused_wall_ms = median(wall_samples)
          fused_gpu_ms = median(gpu_samples)
          printf "%4s %6d %5d %-8s %3s %12.3f %13.3f %28.3f %10.3f\n",
            tier_name, prefix, token_count, route, selected_t4,
            pack_ms, fused_wall_ms, fused_gpu_ms,
            fused_wall_ms - fused_gpu_ms
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
token_count = Qwen35QBitAdaptivePrefillProbe::SOURCE_CHUNK
tier_names = ["p4", "bf16", "f32"]
OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_qbit_adaptive_prefill_probe [options]"
  parser.on("--prefixes LIST", "Comma-separated 64-aligned prefixes") do |value|
    prefixes = value.split(',').map(&.to_i)
  end
  parser.on("--repeats N", "Median sample count, 1..10") { |value| repeats = value.to_i }
  parser.on("--chunk N", "Current token count, 1..64") { |value| token_count = value.to_i }
  parser.on("--tiers LIST", "Comma-separated tiers: p4,bf16,f32") do |value|
    tier_names = value.split(',')
  end
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

Qwen35QBitAdaptivePrefillProbe.run(prefixes, repeats, token_count, tier_names)
