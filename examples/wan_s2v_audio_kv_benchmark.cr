# Bounded fixed-input K/V reuse qualification; never instantiates a Wan model.
# Run serially under wan_memory_guard.py with cache-aware admission.
require "json"
require "option_parser"
require "../src/ml/video/metal_mps_command_chain"
require "./support/wan_audio_kv_cache"

private ITERATIONS = 8
private PAIRS      = 3

private def audio_kv_initialize(prepared : WanAudioKVCacheProbe::Prepared) : Nil
  chain = ML::Metal::MPSCommandChain.new
  begin
    prepared.encode_nan_initialization!(chain.encoder)
    chain.commit_and_wait
  ensure
    # Exceptional encodes may already have submitted work. Drain it before
    # releasing any command, plan, stage, weight or output storage.
    chain.commit_and_wait unless chain.completed?
    chain.release
  end
end

private def audio_kv_arm(prepared : WanAudioKVCacheProbe::Prepared,
                         arm : String, pair : Int32, warmup : Bool)
  allocation_start = ML::MetalBuffer.video_spec_allocations
  started = Time.monotonic
  chain = ML::Metal::MPSCommandChain.new
  begin
    fills, hits, identity = prepared.encode_arm!(chain.encoder, arm, ITERATIONS)
    chain.pause!
    encoded = Time.monotonic
    chain.commit_and_wait
    waited = Time.monotonic
    prepared.complete_cached_arm_after_terminal_wait!(identity)
    finished = Time.monotonic
    allocations = ML::MetalBuffer.video_spec_allocations - allocation_start
    raise "Timed audio K/V arm allocated MetalBuffers" unless allocations == 0
    raise "Timed audio K/V arm added a host wait" unless chain.wait_count == 1
    raise "Audio K/V command root accounting is empty" unless chain.segment_count > 0
    {
      schema: "wan_s2v_audio_kv_arm_v1", pair: pair, warmup: warmup, arm: arm,
      iterations: ITERATIONS, fills: fills, hits: hits,
      encode_seconds: (encoded - started).total_seconds,
      terminal_wait_seconds: (waited - encoded).total_seconds,
      completion_bookkeeping_seconds: (finished - waited).total_seconds,
      total_seconds: (finished - started).total_seconds,
      summed_command_root_gpu_interval_seconds: chain.gpu_elapsed_seconds,
      command_roots: chain.segment_count, terminal_waits: chain.wait_count,
      inter_operator_host_waits: 0, activation_readbacks: 0,
      encode_metal_buffer_allocations: allocations,
      consumption: "identical K and V GPU copies after every recomputation or hit",
      scope: "eight fixed-input projection/norm uses; fill and key checks included; no full injection",
    }
  rescue ex
    prepared.discard_pending!
    raise ex
  ensure
    chain.commit_and_wait unless chain.completed?
    chain.release
  end
end

reference = nil.as(String?)
OptionParser.parse do |parser|
  parser.on("--reference=PATH", "Pinned upstream K/V oracle JSON") { |value| reference = value }
end
fixture = reference || raise ArgumentError.new("--reference=PATH is required")
prepared = WanAudioKVCacheProbe::Prepared.new(fixture.not_nil!)
begin
  device = ML::Metal::Device.instance
  puts({schema: "wan_s2v_audio_kv_device_v1", device_name: device.name,
        recommended_working_set_bytes: device.recommended_working_set_size}.to_json)
  puts prepared.setup_json
  STDOUT.flush
  audio_kv_initialize(prepared)
  puts({schema: "wan_s2v_audio_kv_upstream_validation_v1",
        metrics: prepared.validate_upstream!,
        absolute_tolerance: 2e-5, relative_tolerance: 2e-5,
        require_finite: true, diagnostic_readbacks_outside_timed_arms: true}.to_json)
  puts prepared.lifecycle_contract_json
  STDOUT.flush

  (0..PAIRS).each do |pair|
    warmup = pair == 0
    order = pair == 2 ? %w[cached recompute] : %w[recompute cached]
    order.each do |arm|
      puts audio_kv_arm(prepared, arm, pair, warmup).to_json
      STDOUT.flush
    end
    puts({schema: "wan_s2v_audio_kv_native_validation_v1", pair: pair,
          warmup: warmup, metrics: prepared.validate_native_pair!,
          require_bit_identity: true, require_finite: true,
          diagnostic_readbacks_outside_timed_arms: true}.to_json)
    STDOUT.flush
  end

  puts({schema:                                  "wan_s2v_audio_kv_mutation_validation_v1",
        metrics:                                 prepared.validate_lifecycle_mutations!,
        diagnostic_readbacks_outside_timed_arms: true}.to_json)
  puts({schema: "wan_s2v_audio_kv_completion_v1", measured_pairs: PAIRS,
        warmup_pairs: 1, iterations: ITERATIONS, all_gates_passed: true,
        production_cache_enabled: false, full_block_or_model_run: false}.to_json)
  STDOUT.flush
ensure
  prepared.release_after_terminal_wait!
end
