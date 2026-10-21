# Copyright 2024-2025 The Alibaba Wan Team Authors.
# Apache-2.0. Adapted fixture/spec harness for Wan2.2 S2V at
# 1ea34ff48f87168174e12956e200b1d908b1c5ff.
require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/wan_s2v_block"
require "./support/video_allocation_counter"

# Test-only native-dispatch observation: every block starts with a native
# normalization dispatch, so rejected fusion admission must leave this at zero.
struct ML::Metal::ComputeEncoder
  @@wan_preflight_dispatches : Int64 = 0

  def self.wan_preflight_dispatches : Int64
    @@wan_preflight_dispatches
  end

  def dispatch(grid_size : {Int32, Int32, Int32}, threadgroup_size : {Int32, Int32, Int32}) : self
    @@wan_preflight_dispatches += 1
    previous_def(grid_size, threadgroup_size)
  end
end

lib LibWanVideoSpecMemory
  fun posix_memalign(ptr : Void**, alignment : LibC::SizeT, size : LibC::SizeT) : LibC::Int
  fun getpagesize : LibC::Int
  fun free(ptr : Void*) : Void
end

private WAN_SOURCE_SHA256 = {
  "wan_modules_model.py"               => "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432",
  "wan_modules_s2v_model_s2v.py"       => "91fb14df44edab2d01b5634420b463c5f1563aecdf6dac905da6b3acac3145cc",
  "wan_modules_s2v_audio_utils.py"     => "76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd",
  "wan_modules_s2v_s2v_utils.py"       => "e45ae6c274b17e4fdddcb0439b598897a37679b8934ebd71f87255f92d5f0b2f",
  "wan_modules_s2v_auxi_blocks.py"     => "6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6",
  "diffusers-normalization-v0.31.0.py" => "7f2d2086f669e19bc008d71743f3963017274fbc6ad86d0ab9bf18db211e7c5a",
}

private def wan_json_f32(value : JSON::Any) : Array(Float32)
  value.as_a.map(&.as_f.to_f32)
end

private def wan_comparison(name : String, actual : Array(Float32), expected : JSON::Any) : Tuple(Bool, String)
  reference = wan_json_f32(expected)
  unless actual.size == reference.size
    return {false, "#{name}: checkpoint size mismatch native=#{actual.size} oracle=#{reference.size}"}
  end

  violations = 0
  finite_count = 0
  native_non_finite = 0
  oracle_non_finite = 0
  worst_ratio = -1.0_f64
  worst_index = 0
  worst_actual = 0.0_f32
  worst_reference = 0.0_f32
  worst_difference = 0.0_f32
  worst_limit = 0.0_f32

  actual.each_with_index do |value, index|
    target = reference[index]
    native_non_finite += 1 unless value.finite?
    oracle_non_finite += 1 unless target.finite?
    unless value.finite? && target.finite?
      violations += 1
      if worst_ratio < Float64::INFINITY
        worst_ratio = Float64::INFINITY
        worst_index = index
        worst_actual = value
        worst_reference = target
        worst_difference = (value - target).abs
        worst_limit = 2e-5_f32 + 2e-5_f32 * target.abs
      end
      next
    end

    finite_count += 1
    difference = (value - target).abs
    limit = 2e-5_f32 + 2e-5_f32 * target.abs
    ratio = difference.to_f64 / limit.to_f64
    if difference > limit
      violations += 1
    end
    if ratio > worst_ratio
      worst_ratio = ratio
      worst_index = index
      worst_actual = value
      worst_reference = target
      worst_difference = difference
      worst_limit = limit
    end
  end

  detail = String.build do |io|
    io << "#{name}: checked=#{actual.size} finite=#{finite_count}"
    io << " native_non_finite=#{native_non_finite} oracle_non_finite=#{oracle_non_finite}"
    io << " violations=#{violations}"
    io << " max_normalized_error=#{worst_ratio} at flat=#{worst_index}"
    io << " native=#{worst_actual} oracle=#{worst_reference}"
    io << " abs=#{worst_difference} limit=#{worst_limit}"
  end
  {violations == 0, detail}
end

private def wan_matches(actual : Array(Float32), expected : JSON::Any) : Bool
  wan_comparison("fixture value", actual, expected)[0]
end

private def wan_snapshot_json(snapshots : Hash(String, Array(Float32))) : JSON::Any
  encoded = {} of String => JSON::Any
  snapshots.each do |name, values|
    encoded_values = [] of JSON::Any
    values.each do |value|
      if value.finite?
        encoded_values << JSON::Any.new(value.to_f64)
      else
        encoded_values << JSON::Any.new(value.to_s)
      end
    end
    encoded[name] = JSON::Any.new(encoded_values)
  end
  JSON::Any.new(encoded)
end

private def wan_rope_diagnostic(actual : Array(Float32), expected : JSON::Any,
                                snapshots : Hash(String, Array(Float32)), fixture : JSON::Any) : String
  reference = wan_json_f32(expected)
  raw_q_reference = wan_json_f32(fixture["expected"]["self_q"])
  norm_q_reference = wan_json_f32(fixture["expected"]["self_q_norm"])
  dim = fixture["config"]["dim"].as_i
  seq = fixture["config"]["seq"].as_i
  heads = fixture["config"]["heads"].as_i
  head_dim = dim // heads
  worst_index = 0
  worst_ratio = 0.0_f64
  violations = 0
  actual.each_with_index do |value, index|
    target = reference[index]
    difference = (value - target).abs.to_f64
    limit = 2e-5_f64 + 2e-5_f64 * target.abs.to_f64
    ratio = difference / limit
    violations += 1 if ratio > 1.0
    if ratio > worst_ratio
      worst_ratio = ratio
      worst_index = index
    end
  end

  row = worst_index // dim
  batch = row // seq
  token = row % seq
  channel = worst_index % dim
  head = channel // head_dim
  lane = channel % head_dim
  position_values = fixture["inputs"]["positions"].as_a
  position_offset = (batch * seq + token) * 3
  position = (0...3).map { |axis| position_values[position_offset + axis].as_f }
  difference = (actual[worst_index] - reference[worst_index]).abs
  limit = 2e-5_f32 + 2e-5_f32 * reference[worst_index].abs
  String.build do |io|
    io << "worst normalized error=#{worst_ratio} violations=#{violations} flat=#{worst_index}"
    io << " b=#{batch} token=#{token} head=#{head} channel_in_head=#{lane} pair=#{lane // 2} lane=#{lane % 2}"
    io << " position=#{position} native=#{actual[worst_index]} oracle=#{reference[worst_index]}"
    io << " abs=#{difference} limit=#{limit}"
    io << " raw_q(native/oracle)=#{snapshots["self_q"][worst_index]}/#{raw_q_reference[worst_index]}"
    io << " norm_q(native/oracle)=#{snapshots["self_q_norm"][worst_index]}/#{norm_q_reference[worst_index]}"
  end
end

private def wan_checkpoint_size(name : String, cfg : JSON::Any) : Int32
  dim = cfg["dim"].as_i
  rows = cfg["batches"].as_i * cfg["seq"].as_i
  context = cfg["batches"].as_i * cfg["context_tokens"].as_i
  ffn_dim = cfg["ffn_dim"].as_i
  case name
  when "cross_k", "cross_v"
    context * dim
  when "ffn_in", "ffn_gelu"
    rows * ffn_dim
  else
    rows * dim
  end
end

private def wan_audio_checkpoint_size(name : String, cfg : JSON::Any) : Int32
  dim = cfg["dim"].as_i
  frames = cfg["batches"].as_i * cfg["video_frames"].as_i
  video_rows = frames * cfg["spatial_tokens"].as_i
  audio_rows = frames * cfg["audio_tokens"].as_i
  case name
  when "global_silu"
    frames * dim
  when "adain_projection"
    frames * 2 * dim
  when "audio_k", "audio_v"
    audio_rows * dim
  when "audio_output"
    cfg["batches"].as_i * cfg["seq"].as_i * dim
  else
    video_rows * dim
  end
end

private def wan_fixture : Tuple(JSON::Any, String)
  path = ENV["WAN_S2V_BLOCK_FIXTURE"]? || raise "Set WAN_S2V_BLOCK_FIXTURE to the pinned MPS oracle fixture"
  source = File.read(path)
  {JSON.parse(source), Digest::SHA256.hexdigest(source)}
end

private def wan_fixture_path : String
  ENV["WAN_S2V_BLOCK_FIXTURE"]? || raise "Set WAN_S2V_BLOCK_FIXTURE to the pinned MPS oracle fixture"
end

private def wan_resident_weight_dtype : ML::Video::WeightDType
  case ENV["WAN_S2V_WEIGHT_DTYPE"]? || "f32"
  when "f32"
    ML::Video::WeightDType::F32
  when "bf16"
    ML::Video::WeightDType::BF16
  else
    raise ArgumentError.new("WAN_S2V_WEIGHT_DTYPE must be f32 or bf16")
  end
end

private def wan_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1024 * 1024)
  File.open(path, "rb") do |file|
    while (count = file.read(buffer)) > 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

# The pinned oracle executes B=2 with independent batch members. Selecting its
# first contiguous batch changes neither a token/head reduction nor the weights.
# Keep the original two-batch prechecks and label this projection explicitly;
# it is not a new upstream execution at B=1.
private def wan_first_batch_fixture(fixture : JSON::Any, parent_sha256 : String) : JSON::Any
  cfg = fixture["config"]
  cfg["batches"].as_i.should eq(2)
  dim = cfg["dim"].as_i
  seq = cfg["seq"].as_i
  frames = cfg["video_frames"].as_i
  sizes = {
    "x"            => seq * dim,
    "modulation"   => 12 * dim,
    "positions"    => seq * 3,
    "context"      => cfg["context_tokens"].as_i * dim,
    "local_audio"  => frames * cfg["audio_tokens"].as_i * dim,
    "global_audio" => frames * dim,
  }
  fixture["inputs"].as_h.keys.sort.should eq(sizes.keys.sort)
  projected = JSON.parse(fixture.to_json)
  sizes.each do |name, count|
    values = fixture["inputs"][name].as_a
    values.size.should eq(2 * count), "first-batch input shape #{name}"
    projected["inputs"].as_h[name] = JSON::Any.new(values[0, count])
  end
  fixture["expected"].as_h.each do |name, value|
    count = if ML::Video::WanS2VBlock::CHECKPOINTS.has_key?(name)
              wan_checkpoint_size(name, cfg)
            else
              ML::Video::WanS2VAudioInjector::CHECKPOINTS.has_key?(name).should be_true
              wan_audio_checkpoint_size(name, cfg)
            end
    values = value.as_a
    values.size.should eq(count), "first-batch checkpoint shape #{name}"
    (count % 2).should eq(0)
    projected["expected"].as_h[name] = JSON::Any.new(values[0, count // 2])
  end
  projected["config"].as_h["batches"] = JSON::Any.new(1_i64)
  puts({schema: "wan_s2v_first_batch_projection_v1",
        parent_sha256: parent_sha256, parent_batches: 2,
        selected_batch: 0, projected_batches: 1, neural_weights_changed: false,
        fresh_upstream_execution: false, input_arrays: sizes.size,
        checkpoint_arrays: fixture["expected"].as_h.size}.to_json)
  projected
end

# Test-only fixed bindings. Chunk 1 exercises repeated graph calls on the short
# fixture; chunk 3 exercises its padded tail. Setup never enters block encode.
private def wan_prepare_mps_self_attention(
  cfg : ML::Video::WanS2VConfig, scratch : ML::Video::Arena,
  attention_scratch : ML::Video::Arena,
) : ML::Video::WanAttentionMPSChunkWorkspace?
  prepared_kv = ENV["WAN_S2V_MPS_PREPARED_KV"]? == "1"
  if prepared_kv && ENV["WAN_S2V_MPS_SELF_ATTENTION"]? != "1"
    raise ArgumentError.new("MPS prepared K/V fixture requires WAN_S2V_MPS_SELF_ATTENTION=1")
  end
  return nil unless ENV["WAN_S2V_MPS_SELF_ATTENTION"]? == "1"
  chunk_rows = ENV["WAN_S2V_MPS_SELF_ATTENTION_CHUNK_ROWS"]?.try(&.to_i32) || 1
  unless 1 <= chunk_rows <= ML::Video::WanAttentionMPSChunkWorkspace::MAX_CHUNK_ROWS
    raise ArgumentError.new("MPS self-attention fixture chunk rows are out of range")
  end
  count = (chunk_rows.to_i64 * cfg.dim).to_i32
  prepared_count = (cfg.seq.to_i64 * cfg.dim).to_i32
  prepared_k = prepared_kv ? attention_scratch.alloc("prepared_k_bhkd", prepared_count) : nil
  prepared_v = prepared_kv ? attention_scratch.alloc("prepared_v_bhkd", prepared_count) : nil
  ML::Video::WanAttentionMPSChunkWorkspace.new(
    cfg.batches, cfg.seq, cfg.seq, cfg.heads, cfg.head_dim, chunk_rows,
    scratch["wan_b_self_q"], scratch["wan_b_self_k"], scratch["wan_b_self_v"],
    scratch["wan_b_self_attn"], attention_scratch.alloc("chunk_q", count),
    attention_scratch.alloc("chunk_out_a", count), attention_scratch.alloc("chunk_out_b", count),
    prepared_k, prepared_v)
end

private WAN_KV_PACKED_CHECK_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;
  kernel void wan_spec_packed_zero(device atomic_uint *m [[buffer(0)]],
      uint id [[thread_position_in_grid]]) {
    if (id < 5u) atomic_store_explicit(m + id, 0u, memory_order_relaxed);
  }
  kernel void wan_spec_packed_check(
      device const float *reference_k [[buffer(0)]],
      device const float *reference_v [[buffer(1)]],
      device const float *packed_k [[buffer(2)]],
      device const float *packed_v [[buffer(3)]],
      device atomic_uint *m [[buffer(4)]],
      constant uint &seq [[buffer(5)]], constant uint &heads [[buffer(6)]],
      constant uint &head_dim [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= seq * heads * head_dim) return;
    const uint d = id % head_dim;
    const uint token = (id / head_dim) % seq;
    const uint head = id / (head_dim * seq);
    const uint source = (token * heads + head) * head_dim + d;
    const uint rk = as_type<uint>(reference_k[source]);
    const uint rv = as_type<uint>(reference_v[source]);
    const uint ak = as_type<uint>(packed_k[id]);
    const uint av = as_type<uint>(packed_v[id]);
    if (rk != ak) atomic_fetch_add_explicit(m, 1u, memory_order_relaxed);
    if (rv != av) atomic_fetch_add_explicit(m + 1, 1u, memory_order_relaxed);
    // Qualify both counters with a one-bit defect in the diagnostic register.
    if (rk != (ak ^ (id == 0u ? 1u : 0u)))
      atomic_fetch_add_explicit(m + 2, 1u, memory_order_relaxed);
    if (rv != (av ^ (id == 0u ? 1u : 0u)))
      atomic_fetch_add_explicit(m + 3, 1u, memory_order_relaxed);
    if (!isfinite(reference_k[source]) || !isfinite(reference_v[source]) ||
        !isfinite(packed_k[id]) || !isfinite(packed_v[id]))
      atomic_fetch_add_explicit(m + 4, 1u, memory_order_relaxed);
  }
  METAL

# A separate block-only Arena keeps the existing audio/reused-scratch contract
# unchanged. Replay all intermediates before any colored slot is overwritten.
private def wan_check_reused_block_scratch(
  fixture : JSON::Any, cfg : ML::Video::WanS2VConfig,
  audio_cfg : ML::Video::WanS2VAudioConfig, inputs : ML::Video::Arena,
  weights : ML::Video::Arena, audio_scratch : ML::Video::Arena,
  original_trace : Hash(String, ML::MetalBuffer), original_chained : ML::MetalBuffer,
  tiled_attention : Bool, precompute_rope : Bool, groups : Int32,
  original_attention : ML::Video::WanAttentionMPSChunkWorkspace?,
) : Nil
  scratch = ML::Video::Arena.new
  attention_scratch = ML::Video::Arena.new
  outputs = ML::Video::Arena.new
  traces = ML::Video::Arena.new
  workspace = nil.as(ML::Video::WanS2VMPSLinearWorkspace?)
  attention = nil.as(ML::Video::WanAttentionMPSChunkWorkspace?)
  chain = nil.as(ML::Metal::MPSCommandChain?)
  begin
    fuse_kv_preparation = ENV["WAN_S2V_FUSED_KV_PREPARATION"]? == "1"
    ML::Video::WanS2VBlock.prepare_scratch!(scratch, cfg,
      precompute_rope: precompute_rope, reuse_scratch: true)
    scratch.resident_bytes.should eq(ML::Video::WanS2VBlock.scratch_bytes(cfg,
      precompute_rope: precompute_rope, reuse_scratch: true))
    if ENV["WAN_S2V_MPS_PROJECTIONS"]? == "1"
      workspace = ML::Video::WanS2VMPSLinearWorkspace.new(cfg, audio_cfg,
        scratch, audio_scratch, inputs["context"], inputs["local_audio"])
    end
    attention = wan_prepare_mps_self_attention(cfg, scratch, attention_scratch)
    packed_metrics = fuse_kv_preparation ? outputs.alloc("packed_metrics", 5) : nil
    packed_zero = fuse_kv_preparation ? ML::Metal::ComputePipeline.new("wan_spec_packed_zero", WAN_KV_PACKED_CHECK_SOURCE) : nil
    packed_check = fuse_kv_preparation ? ML::Metal::ComputePipeline.new("wan_spec_packed_check", WAN_KV_PACKED_CHECK_SOURCE) : nil
    if original = original_attention
      expect_raises(ArgumentError, "Prepared Wan attention shape or caller buffers changed") do
        original.validate_for!(cfg.batches, cfg.seq, cfg.seq, cfg.heads, cfg.head_dim,
          scratch["wan_b_self_q"], scratch["wan_b_self_k"], scratch["wan_b_self_v"],
          scratch["wan_b_self_attn"])
      end
    end
    first = outputs.alloc("first", cfg.values)
    repeat = outputs.alloc("repeat", cfg.values)
    chained = outputs.alloc("chained", cfg.values)
    replayed = outputs.alloc("replayed", cfg.values)
    spare = outputs.alloc("spare", cfg.values)
    trace = {} of String => ML::MetalBuffer
    ML::Video::WanS2VBlock::CHECKPOINTS.each_key do |name|
      trace[name] = traces.alloc(name, wan_checkpoint_size(name, fixture["config"]))
    end

    # Every rejected call stops in validation, before any dispatch/continuation.
    guard_command = ML::Metal::CommandBuffer.new
    guard_encoder = ML::Metal::ComputeEncoder.new(guard_command)
    begin
      expect_raises(ArgumentError, "Wan S2V buffers overlap unsafely") do
        ML::Video::WanS2VBlock.encode(guard_encoder, inputs["x"], inputs["modulation"],
          inputs["positions"], inputs["context"], first, weights, scratch, cfg,
          precompute_rope: precompute_rope)
      end
      original = scratch.buffers["wan_b_ffn_out"]
      begin
        scratch.buffers["wan_b_ffn_out"] = spare
        expect_raises(ArgumentError, "Wan S2V scratch reuse requires complete same-object F32 groups") do
          ML::Video::WanS2VBlock.encode(guard_encoder, inputs["x"], inputs["modulation"],
            inputs["positions"], inputs["context"], first, weights, scratch, cfg,
            precompute_rope: precompute_rope, reuse_scratch: true)
        end
      ensure
        scratch.buffers["wan_b_ffn_out"] = original
      end
      # Regression for the rejected six-buffer-saving proposal: cross_attn
      # would overwrite self_res before the following residual add consumes it.
      original = scratch.buffers["wan_b_self_res"]
      begin
        scratch.buffers["wan_b_self_res"] = scratch["wan_b_norm1"]
        expect_raises(ArgumentError, "Wan S2V buffers overlap unsafely") do
          ML::Video::WanS2VBlock.encode(guard_encoder, inputs["x"], inputs["modulation"],
            inputs["positions"], inputs["context"], first, weights, scratch, cfg,
            precompute_rope: precompute_rope, reuse_scratch: true)
        end
      ensure
        scratch.buffers["wan_b_self_res"] = original
      end
      color_c = {"wan_b_self_v", "wan_b_cross_norm", "wan_b_ffn_out"}
      original_c = color_c.map { |name| scratch[name] }
      begin
        color_c.each { |name| scratch.buffers[name] = scratch["wan_b_self_q"] }
        expect_raises(ArgumentError, "Wan S2V buffers overlap unsafely") do
          ML::Video::WanS2VBlock.encode(guard_encoder, inputs["x"], inputs["modulation"],
            inputs["positions"], inputs["context"], first, weights, scratch, cfg,
            precompute_rope: precompute_rope, reuse_scratch: true)
        end
      ensure
        color_c.each_with_index { |name, index| scratch.buffers[name] = original_c[index] }
      end
      {"wan_b_self_q", "wan_b_self_k"}.each do |input_name|
        expect_raises(ArgumentError, "Wan S2V buffers overlap unsafely") do
          ML::Video::WanS2VBlock.encode(guard_encoder, inputs["x"], inputs["modulation"],
            inputs["positions"], inputs["context"], scratch[input_name], weights, scratch, cfg,
            precompute_rope: precompute_rope, reuse_scratch: true)
        end
      end
      expect_raises(ArgumentError, "Wan S2V buffers overlap unsafely") do
        ML::Video::WanS2VBlock.encode(guard_encoder, scratch["wan_b_self_res"], inputs["modulation"],
          inputs["positions"], inputs["context"], first, weights, scratch, cfg,
          precompute_rope: precompute_rope, reuse_scratch: true)
      end
      expect_raises(ArgumentError, "Wan S2V buffers overlap unsafely") do
        ML::Video::WanS2VBlock.encode(guard_encoder, inputs["x"], inputs["modulation"],
          inputs["positions"], inputs["context"], first, weights, scratch, cfg,
          {"norm1" => scratch["wan_b_norm1"]},
          precompute_rope: precompute_rope, reuse_scratch: true)
      end
      expect_raises(ArgumentError, "Wan S2V buffers overlap unsafely") do
        ML::Video::WanS2VBlock.encode(guard_encoder, inputs["x"], inputs["modulation"],
          inputs["positions"], inputs["context"], weights["self_q_weight"], weights, scratch, cfg,
          precompute_rope: precompute_rope, reuse_scratch: true)
      end
      page = LibWanVideoSpecMemory.getpagesize.to_u64
      bytes = ((cfg.values.to_u64 * 4 + page - 1) // page) * page
      ptr = Pointer(Void).null
      LibWanVideoSpecMemory.posix_memalign(pointerof(ptr), page, bytes).should eq(0)
      first_wrapper = second_wrapper = nil.as(ML::MetalBuffer?)
      original_a = scratch["wan_b_norm1"]
      begin
        first_wrapper = ML::MetalBuffer.wrap_no_copy(ptr, bytes.to_i64)
        second_wrapper = ML::MetalBuffer.wrap_no_copy(ptr, bytes.to_i64)
        first_wrapper.handle.should_not eq(second_wrapper.handle)
        scratch.buffers["wan_b_norm1"] = first_wrapper
        scratch.buffers["wan_b_self_attn"] = second_wrapper
        expect_raises(ArgumentError, "Wan S2V scratch reuse requires complete same-object F32 groups") do
          ML::Video::WanS2VBlock.encode(guard_encoder, inputs["x"], inputs["modulation"],
            inputs["positions"], inputs["context"], first, weights, scratch, cfg,
            precompute_rope: precompute_rope, reuse_scratch: true)
        end
      ensure
        scratch.buffers["wan_b_norm1"] = original_a
        scratch.buffers["wan_b_self_attn"] = original_a
        first_wrapper.try(&.release)
        second_wrapper.try(&.release)
        LibWanVideoSpecMemory.free(ptr)
      end
    ensure
      guard_encoder.end_encoding
      guard_command.discard
    end

    active = ML::Metal::MPSCommandChain.new
    chain = active
    guard_dispatches = ML::Metal::ComputeEncoder.wan_preflight_dispatches
    if fuse_kv_preparation
      # All fusion admission failures must occur before the first projection.
      expect_raises(ArgumentError) do
        ML::Video::WanS2VBlock.encode(active.encoder, inputs["x"], inputs["modulation"],
          inputs["positions"], inputs["context"], first, weights, scratch, cfg,
          tiled_attention: tiled_attention, mps_workspace: workspace,
          attention_workspace: attention, reuse_scratch: true, fuse_kv_preparation: true)
      end
      expect_raises(ArgumentError) do
        ML::Video::WanS2VBlock.encode(active.encoder, inputs["x"], inputs["modulation"],
          inputs["positions"], inputs["context"], first, weights, scratch, cfg,
          tiled_attention: tiled_attention, execution_chain: active,
          attention_workspace: attention, reuse_scratch: true, fuse_kv_preparation: true)
      end
      expect_raises(ArgumentError) do
        ML::Video::WanS2VBlock.encode(active.encoder, inputs["x"], inputs["modulation"],
          inputs["positions"], inputs["context"], first, weights, scratch, cfg,
          tiled_attention: tiled_attention, execution_chain: active, mps_workspace: workspace,
          reuse_scratch: true, fuse_kv_preparation: true)
      end
      chunk_rows = attention.not_nil!.chunk_rows
      {nil, scratch["wan_b_self_res"], inputs["x"], trace["self_k_rope"]}.each do |bad_prepared_k|
        rejected = ML::Video::WanAttentionMPSChunkWorkspace.new(
          cfg.batches, cfg.seq, cfg.seq, cfg.heads, cfg.head_dim, chunk_rows,
          scratch["wan_b_self_q"], scratch["wan_b_self_k"], scratch["wan_b_self_v"],
          scratch["wan_b_self_attn"], attention_scratch["chunk_q"],
          attention_scratch["chunk_out_a"], attention_scratch["chunk_out_b"],
          bad_prepared_k, bad_prepared_k ? attention_scratch["prepared_v_bhkd"] : nil)
        begin
          expect_raises(ArgumentError) do
            ML::Video::WanS2VBlock.encode(active.encoder, inputs["x"], inputs["modulation"],
              inputs["positions"], inputs["context"], first, weights, scratch, cfg, trace,
              tiled_attention: tiled_attention, execution_chain: active, mps_workspace: workspace,
              attention_workspace: rejected, reuse_scratch: true, fuse_kv_preparation: true)
          end
        ensure
          rejected.release
        end
      end
      active.wait_count.should eq(0)
      ML::Metal::ComputeEncoder.wan_preflight_dispatches.should eq(guard_dispatches)
    end
    allocations = ML::MetalBuffer.video_spec_allocations
    ML::Video::WanS2VBlock.encode(active.encoder, inputs["x"], inputs["modulation"],
      inputs["positions"], inputs["context"], first, weights, scratch, cfg, trace,
      tiled_attention: tiled_attention, execution_chain: active, mps_workspace: workspace,
      attention_workspace: attention,
      cross_attention_groups_per_segment: groups, precompute_rope: precompute_rope, reuse_scratch: true,
      fuse_kv_preparation: fuse_kv_preparation)
    ML::Metal::ComputeEncoder.wan_preflight_dispatches.should be > guard_dispatches
    if metric_buffer = packed_metrics
      encoder = active.encoder
      encoder.set_pipeline(packed_zero.not_nil!)
        .set_buffer(metric_buffer, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
      encoder.set_pipeline(packed_check.not_nil!)
        .set_buffer(original_trace["self_k_rope"], 0).set_buffer(original_trace["self_v"], 1)
        .set_buffer(attention_scratch["prepared_k_bhkd"], 2).set_buffer(attention_scratch["prepared_v_bhkd"], 3)
        .set_buffer(metric_buffer, 4, ML::Metal::BufferAccess::ReadWrite)
        .set_value(cfg.seq.to_u32, 5).set_value(cfg.heads.to_u32, 6).set_value(cfg.head_dim.to_u32, 7)
        .dispatch_1d(cfg.values)
    end
    ML::Video::WanS2VBlock.encode(active.encoder, inputs["x"], inputs["modulation"],
      inputs["positions"], inputs["context"], repeat, weights, scratch, cfg,
      tiled_attention: tiled_attention, execution_chain: active, mps_workspace: workspace,
      attention_workspace: attention,
      cross_attention_groups_per_segment: groups, precompute_rope: precompute_rope, reuse_scratch: true,
      fuse_kv_preparation: fuse_kv_preparation)
    ML::Video::WanS2VBlock.encode(active.encoder, first, inputs["modulation"],
      inputs["positions"], inputs["context"], chained, weights, scratch, cfg,
      tiled_attention: tiled_attention, execution_chain: active, mps_workspace: workspace,
      attention_workspace: attention,
      cross_attention_groups_per_segment: groups, precompute_rope: precompute_rope, reuse_scratch: true,
      fuse_kv_preparation: fuse_kv_preparation)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    active.wait_count.should eq(0)
    active.commit_and_wait
    active.wait_count.should eq(1)
    if metric_buffer = packed_metrics
      values = metric_buffer.contents.as(UInt32*)
      (0...5).map { |i| values[i] }.should eq([0_u32, 0_u32, 1_u32, 1_u32, 0_u32]),
        "fused prepared K/V must match original logical checkpoints on GPU and reject seeded bit defects"
      puts({schema: "wan_s2v_fused_kv_prepared_check_v1", values_per_buffer: cfg.values,
            k_bit_differences: values[0], v_bit_differences: values[1],
            k_corruption_differences: values[2], v_corruption_differences: values[3],
            nonfinite: values[4], diagnostic_waits_added: 0,
            reference: "unfused retained-upstream-qualified row-major checkpoints"}.to_json)
    end
    active.release
    chain = nil

    # Reuse the same Arena and fixed MPS bindings after a completed chain.
    active = ML::Metal::MPSCommandChain.new
    chain = active
    ML::Video::WanS2VBlock.encode(active.encoder, inputs["x"], inputs["modulation"],
      inputs["positions"], inputs["context"], replayed, weights, scratch, cfg,
      tiled_attention: tiled_attention, execution_chain: active, mps_workspace: workspace,
      attention_workspace: attention,
      cross_attention_groups_per_segment: groups, precompute_rope: precompute_rope, reuse_scratch: true,
      fuse_kv_preparation: fuse_kv_preparation)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    active.wait_count.should eq(0)
    active.commit_and_wait
    active.wait_count.should eq(1)
    active.release
    chain = nil

    trace.each do |name, buffer|
      count = wan_checkpoint_size(name, fixture["config"])
      actual = buffer.read(count).map(&.to_f32)
      actual.map(&.unsafe_as(UInt32)).should eq(original_trace[name].read(count).map(&.to_f32.unsafe_as(UInt32))),
        "reused scratch checkpoint #{name} must match default bits"
      passed, detail = wan_comparison("reused scratch #{name}", actual, fixture["expected"][name])
      passed.should be_true, detail
    end
    original_trace["output"].read(cfg.values).map(&.to_f32.unsafe_as(UInt32)).should eq(
      first.read(cfg.values).map(&.to_f32.unsafe_as(UInt32)))
    repeat.read(cfg.values).map(&.to_f32.unsafe_as(UInt32)).should eq(first.read(cfg.values).map(&.to_f32.unsafe_as(UInt32)))
    chained.read(cfg.values).map(&.to_f32.unsafe_as(UInt32)).should eq(
      original_chained.read(cfg.values).map(&.to_f32.unsafe_as(UInt32)))
    replayed.read(cfg.values).map(&.to_f32.unsafe_as(UInt32)).should eq(
      first.read(cfg.values).map(&.to_f32.unsafe_as(UInt32)))
    puts({schema: "wan_s2v_scratch_reuse_fixture_v2", dtype: wan_resident_weight_dtype.to_s,
          traced_checkpoints: trace.size, strict_pass: true, default_bit_differences: 0,
          repeat_bit_differences: 0, chained_bit_differences: 0, completed_chain_replay_bit_differences: 0,
          encode_metal_buffer_allocations: 0, completed_chains: 2, terminal_waits_per_chain: 1,
          resident_scratch_bytes: scratch.resident_bytes,
          default_scratch_bytes: ML::Video::WanS2VBlock.scratch_bytes(cfg, precompute_rope: precompute_rope),
          precompute_rope: precompute_rope, mps_projections: !workspace.nil?,
          fuse_kv_preparation: fuse_kv_preparation,
          self_attention_backend: attention ? "mpsgraph_sdpa_chunked" : "native",
          self_attention_kv_layout: attention.try { |workspace| workspace.head_major_kv? ? "B,H,K,D" : "B,K,H,D" },
          prepared_kv_refreshed_inside_each_block_encode: attention.try(&.head_major_kv?) || false,
          attention_chunk_rows: attention.try(&.chunk_rows), attention_scratch_bytes: attention_scratch.resident_bytes,
          scope: "real-width short block fixture; not full spatial geometry or video"}.to_json)
  ensure
    if active = chain
      active.commit_and_wait unless active.completed?
      active.release if active.completed?
    end
    attention.try(&.release)
    workspace.try(&.release_after_terminal_wait!)
    attention_scratch.release
    traces.release
    outputs.release
    scratch.release
  end
end

private def wan_run_fixture(fixture : JSON::Any) : Hash(String, Array(Float32))
  resident_dtype = wan_resident_weight_dtype
  inputs = ML::Video::Arena.new
  weights = ML::Video::Arena.new
  scratch = ML::Video::Arena.new
  attention_scratch = ML::Video::Arena.new
  traces = ML::Video::Arena.new
  output = ML::Video::Arena.new
  checkpoint_file = nil.as(ML::Video::CheckpointFile?)
  prepared_weights = nil.as(ML::Video::PreparedWeights?)
  execution_chain = nil.as(ML::Metal::MPSCommandChain?)
  phase_root_chain = nil.as(ML::Metal::MPSCommandChain?)
  mps_workspace = nil.as(ML::Video::WanS2VMPSLinearWorkspace?)
  attention_workspace = nil.as(ML::Video::WanAttentionMPSChunkWorkspace?)
  begin
    fixture["inputs"].as_h.each do |name, values|
      inputs.upload(name, wan_json_f32(values))
    end
    config_json = fixture["config"]
    block_cfg = ML::Video::WanS2VConfig.new(
      batches: config_json["batches"].as_i,
      seq: config_json["seq"].as_i,
      context_tokens: config_json["context_tokens"].as_i,
      dim: config_json["dim"].as_i,
      heads: config_json["heads"].as_i,
      ffn_dim: config_json["ffn_dim"].as_i,
      split: config_json["split"].as_i,
      theta: config_json["theta"].as_f.to_f32,
      video_frames: config_json["video_frames"].as_i,
      spatial_tokens: config_json["spatial_tokens"].as_i,
    )
    audio_cfg = ML::Video::WanS2VAudioConfig.new(
      batches: config_json["batches"].as_i,
      seq: config_json["seq"].as_i,
      dim: config_json["dim"].as_i,
      heads: config_json["heads"].as_i,
      video_frames: config_json["video_frames"].as_i,
      spatial_tokens: config_json["spatial_tokens"].as_i,
      audio_tokens: config_json["audio_tokens"].as_i,
    )
    if ENV["WAN_S2V_MPS_SELF_ATTENTION"]? == "1"
      unless block_cfg.batches == 1 && ENV["WAN_S2V_SEGMENTED_COMMANDS"]? == "1"
        raise ArgumentError.new("MPS self-attention fixture requires B=1 and segmented commands")
      end
    end

    if weight_manifest = fixture["weights_file"]?
      fixture["upstream"]["fixture_mode"].as_s.should eq("production_width")
      fixture_path = wan_fixture_path
      relative_path = weight_manifest["path"].as_s
      (File.basename(relative_path) == relative_path && relative_path != "." && relative_path != "..").should be_true,
        "weight sidecar must stay beside the fixture"
      sidecar_path = File.join(File.dirname(File.expand_path(fixture_path)), relative_path)
      File.size(sidecar_path).should eq(weight_manifest["bytes"].as_i64)
      wan_sha256_file(sidecar_path).should eq(weight_manifest["sha256"].as_s)

      fixture_bindings = fixture["weight_bindings"].as_a
      fixture_bindings.size.should eq(39)
      dtypes = fixture_bindings.map { |item| item["dtype"].as_s }.uniq
      dtypes.size.should eq(1), "all checkpoint fixture tensors must use one source dtype"
      checkpoint_dtype = ML::GGUF::SafeTensorDType.parse(dtypes.first)
      if resident_dtype.bf16?
        checkpoint_dtype.should eq(ML::GGUF::SafeTensorDType::BF16),
          "resident BF16 mode requires an exact BF16 source checkpoint"
      end
      bindings = ML::Video::WanS2VBlock.checkpoint_bindings(block_cfg, checkpoint_dtype)
      fixture_bindings.map { |item| item["source"].as_s }.sort.should eq(bindings.map(&.source).sort)
      fixture_bindings.map { |item| item["target"].as_s }.sort.should eq(bindings.map(&.target).sort)
      fixture_bindings.each do |item|
        native = bindings.find! { |binding| binding.source == item["source"].as_s }
        item["shape"].as_a.map(&.as_i64).should eq(native.shape)
        ML::GGUF::SafeTensorDType.parse(item["dtype"].as_s).should eq(native.dtype)
      end

      checkpoint_file = ML::Video::CheckpointFile.new(sidecar_path)
      plan = ML::Video::WeightPlan.new(
        [checkpoint_file.not_nil!], bindings,
        8_i64 * 1024 * 1024 * 1024, 1024_i64 * 1024 * 1024,
        resident_dtype: resident_dtype, require_all: true,
      )
      plan.entries.size.should eq(39)
      expected_destination_bytes = resident_dtype.bf16? ? 1_017_443_328_i64 : 2_034_886_656_i64
      plan.destination_bytes.should eq(expected_destination_bytes)
      expected_staging_bytes = if resident_dtype.bf16?
                                 0_i64
                               else
                                 expected_destination_bytes * checkpoint_dtype.bytes_per_element // 4
                               end
      if resident_dtype.bf16?
        expected_staging_bytes.should eq(0_i64),
          "native BF16 residency must not decode or stage a second copy"
      else
        expected_staging_bytes.should eq(
          checkpoint_dtype.f32? ? 2_034_886_656_i64 : 1_017_443_328_i64
        )
      end
      plan.staging_bytes.should eq(expected_staging_bytes)
      plan.peak_bytes.should be <= 8_i64 * 1024 * 1024 * 1024
      prepared_weights = ML::Video::PreparedWeights.new(plan, weights)
      bindings.each do |binding|
        weights.dtype(binding.target).should eq(resident_dtype)
      end
    else
      resident_dtype.should eq(ML::Video::WeightDType::F32),
        "the synthetic JSON fixture contains F32 parameters only"
      fixture["weights"].as_h.each do |name, values|
        weights.upload(name, wan_json_f32(values))
      end
    end

    precompute_rope = ENV["WAN_S2V_PRECOMPUTE_ROPE"]? == "1"
    ML::Video::WanS2VBlock.prepare!(precompute_rope: precompute_rope)
    ML::Video::WanS2VAudioInjector.prepare!
    tiled_attention = ENV["WAN_S2V_TILED_ATTENTION"]? == "1"
    segmented_commands = ENV["WAN_S2V_SEGMENTED_COMMANDS"]? == "1"
    phase_root_profile = ENV["WAN_S2V_PHASE_ROOT_PROFILE"]? == "1"
    if phase_root_profile && !segmented_commands
      raise ArgumentError.new("Wan phase-root profiling requires segmented commands")
    end
    cross_attention_groups_per_segment =
      ENV["WAN_S2V_CROSS_ATTENTION_GROUPS_PER_SEGMENT"]?.try(&.to_i32) || 320
    if cross_attention_groups_per_segment != 320 && (!tiled_attention || !segmented_commands)
      raise ArgumentError.new("Wan cross-attention segment overrides require tiled segmented commands")
    end
    ML::Video::MetalOps.prepare_tiled_attention! if tiled_attention
    ML::Video::WanS2VBlock.prepare_scratch!(scratch, block_cfg, precompute_rope: precompute_rope)
    ML::Video::WanS2VAudioInjector.prepare_scratch!(scratch, audio_cfg)
    attention_workspace = wan_prepare_mps_self_attention(block_cfg, scratch, attention_scratch)
    if ENV["WAN_S2V_MPS_PROJECTIONS"]? == "1"
      mps_workspace = ML::Video::WanS2VMPSLinearWorkspace.new(block_cfg, audio_cfg,
        scratch, scratch, inputs["context"], inputs["local_audio"])
      mps_workspace.not_nil!.plan_count.should eq(15)
    end
    block_output = output.alloc("block_output", block_cfg.values)
    audio_output = output.alloc("audio_output", audio_cfg.values)
    repeat_block_output = output.alloc("repeat_block_output", block_cfg.values)
    repeat_audio_output = output.alloc("repeat_audio_output", audio_cfg.values)
    chained_block_output = output.alloc("chained_block_output", block_cfg.values)
    phase_root_block_output = nil.as(ML::MetalBuffer?)
    phase_root_block_output = output.alloc("phase_root_block_output", block_cfg.values) if phase_root_profile
    {inputs, scratch, traces, output}.each do |arena|
      arena.buffers.each_key do |name|
        arena.dtype(name).should eq(ML::Video::WeightDType::F32),
          "activations, scratch, and checkpoints must remain F32"
      end
    end

    expected = fixture["expected"].as_h
    block_trace = {} of String => ML::MetalBuffer
    audio_trace = {} of String => ML::MetalBuffer
    expected.each_key do |name|
      if ML::Video::WanS2VBlock::CHECKPOINTS.has_key?(name)
        block_trace[name] = traces.alloc("block.#{name}", wan_checkpoint_size(name, config_json))
      elsif ML::Video::WanS2VAudioInjector::CHECKPOINTS.has_key?(name)
        audio_trace[name] = traces.alloc("audio.#{name}", wan_audio_checkpoint_size(name, config_json))
      end
    end

    mismatched_cfg = ML::Video::WanS2VConfig.new(
      batches: block_cfg.batches, seq: block_cfg.seq,
      context_tokens: block_cfg.context_tokens, dim: block_cfg.dim,
      heads: block_cfg.heads, ffn_dim: block_cfg.ffn_dim,
      split: block_cfg.split, theta: block_cfg.theta,
      video_frames: block_cfg.video_frames, spatial_tokens: block_cfg.spatial_tokens + 1,
    )
    mismatched_audio_cfg = ML::Video::WanS2VAudioConfig.new(
      batches: audio_cfg.batches, seq: audio_cfg.seq, dim: audio_cfg.dim,
      heads: audio_cfg.heads, video_frames: audio_cfg.video_frames,
      spatial_tokens: audio_cfg.spatial_tokens + 1, audio_tokens: audio_cfg.audio_tokens,
    )
    rejected_command = ML::Metal::CommandBuffer.new
    rejected_encoder = ML::Metal::ComputeEncoder.new(rejected_command)
    expect_raises(ArgumentError, "Wan S2V block split must equal the shared video prefix") do
      ML::Video::WanS2VBlock.encode_with_audio(
        rejected_encoder, inputs["x"], inputs["modulation"], inputs["positions"],
        inputs["context"], inputs["local_audio"], inputs["global_audio"],
        block_output, audio_output, weights, scratch, mismatched_cfg, mismatched_audio_cfg,
      )
    end
    rejected_encoder.end_encoding
    rejected_command.discard

    different_rope_cfg = ML::Video::WanS2VConfig.new(
      batches: block_cfg.batches, seq: block_cfg.seq,
      context_tokens: block_cfg.context_tokens, dim: block_cfg.dim,
      heads: block_cfg.heads, ffn_dim: block_cfg.ffn_dim,
      split: block_cfg.split, theta: block_cfg.theta + 0.5_f32,
      video_frames: block_cfg.video_frames, spatial_tokens: block_cfg.spatial_tokens,
    )
    rope_guard_command = ML::Metal::CommandBuffer.new
    rope_guard_encoder = ML::Metal::ComputeEncoder.new(rope_guard_command)
    expect_raises(KeyError) do
      ML::Video::WanS2VBlock.encode_with_audio(
        rope_guard_encoder, inputs["x"], inputs["modulation"], inputs["positions"],
        inputs["context"], inputs["local_audio"], inputs["global_audio"],
        block_output, audio_output, weights, scratch, different_rope_cfg, audio_cfg,
      )
    end
    rope_guard_encoder.end_encoding
    rope_guard_command.discard

    cross_guard_command = ML::Metal::CommandBuffer.new
    cross_guard_encoder = ML::Metal::ComputeEncoder.new(cross_guard_command)
    expect_raises(ArgumentError, "Wan cross-attention segment size must be positive") do
      ML::Video::WanS2VBlock.encode(
        cross_guard_encoder, inputs["x"], inputs["modulation"], inputs["positions"],
        inputs["context"], block_output, weights, scratch, block_cfg,
        tiled_attention: false, cross_attention_groups_per_segment: 0,
      )
    end
    cross_guard_encoder.end_encoding
    cross_guard_command.discard

    cross_guard_command = ML::Metal::CommandBuffer.new
    cross_guard_encoder = ML::Metal::ComputeEncoder.new(cross_guard_command)
    expect_raises(ArgumentError, "Wan cross-attention segment override requires an MPS command chain") do
      ML::Video::WanS2VBlock.encode(
        cross_guard_encoder, inputs["x"], inputs["modulation"], inputs["positions"],
        inputs["context"], block_output, weights, scratch, block_cfg,
        tiled_attention: true, cross_attention_groups_per_segment: 1024,
      )
    end
    cross_guard_encoder.end_encoding
    cross_guard_command.discard

    cross_guard_command = ML::Metal::CommandBuffer.new
    cross_guard_encoder = ML::Metal::ComputeEncoder.new(cross_guard_command)
    expect_raises(ArgumentError, "Wan cross-attention segment size must be positive") do
      ML::Video::WanS2VBlock.encode_with_audio(
        cross_guard_encoder, inputs["x"], inputs["modulation"], inputs["positions"],
        inputs["context"], inputs["local_audio"], inputs["global_audio"],
        block_output, audio_output, weights, scratch, block_cfg, audio_cfg,
        tiled_attention: false, cross_attention_groups_per_segment: -1,
      )
    end
    cross_guard_encoder.end_encoding
    cross_guard_command.discard

    allocation_start = ML::MetalBuffer.video_spec_allocations
    execution_chain = ML::Metal::MPSCommandChain.new if segmented_commands
    command = execution_chain ? nil : ML::Metal::CommandBuffer.new
    encoder = execution_chain.try(&.encoder) || ML::Metal::ComputeEncoder.new(command.not_nil!)
    prepared_weights.try(&.encode(encoder))
    ML::Video::WanS2VBlock.encode_with_audio(
      encoder, inputs["x"], inputs["modulation"], inputs["positions"],
      inputs["context"], inputs["local_audio"], inputs["global_audio"],
      block_output, audio_output, weights, scratch, block_cfg, audio_cfg,
      block_trace, audio_trace, tiled_attention: tiled_attention, execution_chain: execution_chain,
      mps_workspace: mps_workspace,
      attention_workspace: attention_workspace,
      cross_attention_groups_per_segment: cross_attention_groups_per_segment,
      precompute_rope: precompute_rope)
    if active = execution_chain
      encoder = active.encoder
    end
    ML::Video::WanS2VBlock.encode_with_audio(
      encoder, inputs["x"], inputs["modulation"], inputs["positions"],
      inputs["context"], inputs["local_audio"], inputs["global_audio"],
      repeat_block_output, repeat_audio_output, weights, scratch, block_cfg, audio_cfg,
      tiled_attention: tiled_attention, execution_chain: execution_chain,
      mps_workspace: mps_workspace,
      attention_workspace: attention_workspace,
      cross_attention_groups_per_segment: cross_attention_groups_per_segment,
      precompute_rope: precompute_rope,
    )
    if active = execution_chain
      encoder = active.encoder
    end
    ML::Video::WanS2VBlock.encode(
      encoder, block_output, inputs["modulation"], inputs["positions"],
      inputs["context"], chained_block_output, weights, scratch, block_cfg,
      tiled_attention: tiled_attention, execution_chain: execution_chain,
      mps_workspace: mps_workspace,
      attention_workspace: attention_workspace,
      cross_attention_groups_per_segment: cross_attention_groups_per_segment,
      precompute_rope: precompute_rope,
    )
    ML::MetalBuffer.video_spec_allocations.should eq(allocation_start),
      "Wan encode must not allocate MetalBuffers after setup"
    if active = execution_chain
      active.wait_count.should eq(0), "segmented encoding must not wait"
      active.commit_and_wait
      active.wait_count.should eq(1), "segmented blocks must share one terminal wait"
      active.segment_count.should be > 1
    else
      encoder.end_encoding
      command.not_nil!.commit
      command.not_nil!.wait
    end

    if phase_root_profile
      active = ML::Metal::MPSCommandChain.new
      phase_root_chain = active
      active.enable_phase_root_profile!
      profile_output = phase_root_block_output || raise "phase-root output was not allocated"
      profile_allocations = ML::MetalBuffer.video_spec_allocations
      ML::Video::WanS2VBlock.encode(
        active.encoder, inputs["x"], inputs["modulation"], inputs["positions"],
        inputs["context"], profile_output, weights, scratch, block_cfg,
        tiled_attention: tiled_attention, execution_chain: active,
        mps_workspace: mps_workspace,
        attention_workspace: attention_workspace,
        cross_attention_groups_per_segment: cross_attention_groups_per_segment,
        precompute_rope: precompute_rope,
      )
      ML::MetalBuffer.video_spec_allocations.should eq(profile_allocations),
        "phase-root block encode must not allocate MetalBuffers"
      active.wait_count.should eq(0), "phase-root block encoding must not wait"
      active.commit_and_wait
      active.wait_count.should eq(1), "phase-root profile must use one terminal wait"
      active.segment_count.should be > 1, "phase-root profile must span segmented commands"

      profile = active.phase_root_profile
      profile.root_count.should eq(active.segment_count)
      profile.root_count.should be > 1
      profile.phase_union_seconds.keys.sort.should eq([
        "cross_attention_projection_residual",
        "ffn",
        "self_attention_projection_residual",
        "self_prepare",
      ])
      [
        profile.root_duration_seconds,
        profile.envelope_seconds,
        profile.union_seconds,
        profile.overlap_seconds,
      ].each { |seconds| seconds.finite?.should be_true }
      profile.root_duration_seconds.should be > 0.0
      profile.envelope_seconds.should be > 0.0
      profile.union_seconds.should be > 0.0
      profile.overlap_seconds.should be >= 0.0
      profile.phase_union_seconds.values.each do |seconds|
        seconds.finite?.should be_true
        seconds.should be > 0.0
      end
      active.release
      phase_root_chain = nil
    end

    if ENV["WAN_S2V_SCRATCH_REUSE"]? == "1"
      wan_check_reused_block_scratch(fixture, block_cfg, audio_cfg, inputs, weights, scratch,
        block_trace, chained_block_output, tiled_attention, precompute_rope,
        cross_attention_groups_per_segment, attention_workspace)
    end

    prepared_weights.try(&.release)
    checkpoint_file.try(&.close)

    snapshots = {} of String => Array(Float32)
    block_trace.each { |name, buffer| snapshots[name] = buffer.read((buffer.size // 4).to_i).map(&.to_f32) }
    audio_trace.each { |name, buffer| snapshots[name] = buffer.read((buffer.size // 4).to_i).map(&.to_f32) }
    snapshots["repeat_block_output"] = repeat_block_output.read((repeat_block_output.size // 4).to_i).map(&.to_f32)
    snapshots["repeat_audio_output"] = repeat_audio_output.read((repeat_audio_output.size // 4).to_i).map(&.to_f32)
    snapshots["chained_block_output"] = chained_block_output.read((chained_block_output.size // 4).to_i).map(&.to_f32)
    if profile_output = phase_root_block_output
      snapshots["phase_root_block_output"] = profile_output.read((profile_output.size // 4).to_i).map(&.to_f32)
      original_output = block_output.read((block_output.size // 4).to_i).map(&.to_f32)
      snapshots["phase_root_block_output"].map(&.unsafe_as(UInt32)).should eq(
        original_output.map(&.unsafe_as(UInt32))),
        "isolated profiled block output must be bit-identical to the original block_output buffer"
      passed, detail = wan_comparison(
        "phase_root_block_output", snapshots["phase_root_block_output"], fixture["expected"]["output"])
      passed.should be_true, detail
    end
    if snapshot_path = ENV["WAN_S2V_SNAPSHOT_DUMP"]?
      File.write(snapshot_path, wan_snapshot_json(snapshots).to_json)
    end
    snapshots
  ensure
    if active = execution_chain
      active.commit_and_wait unless active.completed?
      active.release if active.completed?
    end
    if active = phase_root_chain
      active.commit_and_wait unless active.completed?
      active.release if active.completed?
    end
    prepared_weights.try(&.release)
    checkpoint_file.try(&.close)
    attention_workspace.try(&.release)
    mps_workspace.try(&.release_after_terminal_wait!)
    attention_scratch.release
    output.release
    traces.release
    scratch.release
    weights.release
    inputs.release
  end
end

describe ML::Video::WanS2VConfig do
  it "preserves the tiny upstream 3-axis RoPE partition" do
    config = ML::Video::WanS2VConfig.new(
      batches: 1, seq: 12, context_tokens: 3, dim: 32, heads: 2,
      ffn_dim: 64, split: 8, theta: 10_000.0_f32,
      video_frames: 2, spatial_tokens: 4,
    )
    config.head_dim.should eq(16)
    config.rope_axis_dims.should eq({4, 2, 2})
    config.video_tokens.should eq(8)
  end

  it "rejects invalid segment and attention dimensions before encoding" do
    expect_raises(ArgumentError) do
      ML::Video::WanS2VConfig.new(
        batches: 1, seq: 12, context_tokens: 3, dim: 32, heads: 2,
        ffn_dim: 64, split: 0, theta: 10_000.0_f32,
        video_frames: 2, spatial_tokens: 4,
      )
    end
    expect_raises(ArgumentError) do
      ML::Video::WanS2VConfig.new(
        batches: 1, seq: 12, context_tokens: 3, dim: 30, heads: 2,
        ffn_dim: 64, split: 8, theta: 10_000.0_f32,
        video_frames: 2, spatial_tokens: 4,
      )
    end
  end
end

describe ML::Video::WanS2VAudioConfig do
  it "requires the upstream five local audio tokens per frame" do
    config = ML::Video::WanS2VAudioConfig.new(
      batches: 1, seq: 12, dim: 32, heads: 2,
      video_frames: 2, spatial_tokens: 4, audio_tokens: 5,
    )
    config.audio_rows.should eq(10)
    expect_raises(ArgumentError) do
      ML::Video::WanS2VAudioConfig.new(
        batches: 1, seq: 12, dim: 32, heads: 2,
        video_frames: 2, spatial_tokens: 4, audio_tokens: 4,
      )
    end
  end
end

describe ML::Video::WanS2VBlock do
  it "rejects stage-pool scratch reuse before allocating constants or owners" do
    cfg = ML::Video::WanS2VConfig.new(
      batches: 1, seq: 2, context_tokens: 1, dim: 8, heads: 1,
      ffn_dim: 16, split: 1, theta: 10_000.0_f32,
      video_frames: 1, spatial_tokens: 1)
    page = LibC.sysconf(LibC::SC_PAGESIZE).to_i64
    pool = ML::Video::MetalStagePool.new(page * 2, page)
    scratch = ML::Video::Arena.new(stage_pool: pool, stage_phase: "scratch")
    begin
      allocations = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, "Wan S2V scratch reuse requires an unpooled Arena") do
        ML::Video::WanS2VBlock.prepare_scratch!(scratch, cfg, reuse_scratch: true)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      scratch.buffers.empty?.should be_true
      pool.live_view_count.should eq(0)
    ensure
      scratch.release
      pool.release unless pool.released?
    end
  end

  it "matches pinned block/audio checkpoints with optional first-batch MPSGraph and phase-root probes" do
    if ENV["WAN_S2V_PHASE_ROOT_PROFILE"]? == "1"
      ENV["WAN_S2V_SEGMENTED_COMMANDS"]?.should eq("1"),
        "WAN_S2V_PHASE_ROOT_PROFILE requires WAN_S2V_SEGMENTED_COMMANDS=1"
    end
    unless ENV["WAN_S2V_BLOCK_FIXTURE"]?
      pending "MPS oracle fixture is external; set WAN_S2V_BLOCK_FIXTURE"
    end
    fixture, fixture_sha256 = wan_fixture
    fixture["upstream"]["revision"].as_s.should eq("1ea34ff48f87168174e12956e200b1d908b1c5ff")
    fixture["upstream"]["diffusers_version"].as_s.should eq("0.31.0")
    fixture["upstream"]["oracle_device"].as_s.should eq("mps")
    fixture["upstream"]["fallback_disabled"].as_bool.should be_true
    fixture_mode = fixture["upstream"]["fixture_mode"].as_s
    {"tiny", "production_width"}.includes?(fixture_mode).should be_true
    source_hashes = fixture["upstream"]["source_sha256"].as_h
    source_hashes.keys.sort.should eq(WAN_SOURCE_SHA256.keys.sort)
    WAN_SOURCE_SHA256.each { |name, expected| source_hashes[name].as_s.should eq(expected) }
    fixture["config"]["audio_attention_query_rows_per_frame"].as_i.should eq(fixture["config"]["spatial_tokens"].as_i)
    fixture["config"]["batches"].as_i.should eq(2), "the MPS parity fixture must exercise two batches"
    fixture["config"]["heads"].as_i.should be > 1
    config = fixture["config"]
    if fixture_mode == "production_width"
      config["dim"].as_i.should eq(5120)
      config["ffn_dim"].as_i.should eq(13824)
      config["heads"].as_i.should eq(40)
      (config["dim"].as_i // config["heads"].as_i).should eq(128)
      config["qk_norm"].as_bool.should be_true
      config["cross_attn_norm"].as_bool.should be_true
      config["enable_adain"].as_bool.should be_true
      config["adain_mode"].as_s.should eq("attn_norm")
      config["audio_inject_layers"].as_a.map(&.as_i).should eq([0])
      config["split"].as_i.should eq(1)
      config["video_frames"].as_i.should eq(1)
      config["spatial_tokens"].as_i.should eq(1)
      config["audio_tokens"].as_i.should eq(5)
    end
    dim = config["dim"].as_i
    seq = config["seq"].as_i
    positions = fixture["inputs"]["positions"].as_a
    position_at = ->(batch : Int32, token : Int32) {
      offset = (batch * seq + token) * 3
      (0...3).map { |axis| positions[offset + axis].as_f }
    }
    position_at.call(0, 0).should eq([255.0, 511.0, 1022.0])
    position_at.call(1, 0).should eq([301.0, 700.0, 800.0])
    if fixture_mode == "production_width"
      position_at.call(0, 1).should eq([-30.0, 0.0, 0.0])
      position_at.call(1, 1).should eq([-45.0, 10.0, 20.0])
    else
      position_at.call(0, 7).should eq([256.0, 512.0, 1023.0])
      position_at.call(1, 7).should eq([302.0, 701.0, 801.0])
      position_at.call(0, 8).should eq([-30.0, 0.0, 0.0])
      position_at.call(1, 8).should eq([-45.0, 10.0, 20.0])
    end
    position_at.call(0, 0).should_not eq(position_at.call(0, 1)),
      "RoPE regression fixture must distinguish neighboring token positions"
    position_at.call(0, 0).should_not eq(position_at.call(1, 0)),
      "RoPE regression fixture must distinguish batches"
    batch_sizes = {
      "x"            => seq * dim,
      "modulation"   => 12 * dim,
      "context"      => config["context_tokens"].as_i * dim,
      "local_audio"  => config["video_frames"].as_i * config["audio_tokens"].as_i * dim,
      "global_audio" => config["video_frames"].as_i * dim,
    }
    batch_sizes.each do |name, batch_size|
      values = fixture["inputs"][name].as_a
      values[0, batch_size].should_not eq(values[batch_size, batch_size]),
        "fixture input #{name} must differ between batches"
    end
    required_checkpoints = ML::Video::WanS2VBlock::CHECKPOINTS.keys + ML::Video::WanS2VAudioInjector::CHECKPOINTS.keys
    fixture["expected"].as_h.keys.sort.should eq(required_checkpoints.sort)
    if ENV["WAN_S2V_MPS_SELF_ATTENTION"]? == "1"
      fixture = wan_first_batch_fixture(fixture, fixture_sha256)
      config = fixture["config"]
    end
    snapshots = wan_run_fixture(fixture)
    checkpoint_failures = [] of String
    fixture["expected"].as_h.each do |name, expected|
      unless snapshots.has_key?(name)
        checkpoint_failures << "missing Wan checkpoint #{name}"
        next
      end

      passed, detail = wan_comparison(name, snapshots[name], expected)
      unless passed
        if name == "self_q_rope"
          detail += " (#{wan_rope_diagnostic(snapshots[name], expected, snapshots, fixture)})"
        end
        checkpoint_failures << detail
      end
    end
    checkpoint_failures.empty?.should be_true, checkpoint_failures.join("\n")
    corrupted_audio_reference = wan_json_f32(fixture["expected"]["audio_output"])
    corrupted_audio_reference[0] += 0.01_f32
    wan_matches(snapshots["audio_output"], JSON.parse(corrupted_audio_reference.to_json)).should be_false,
      "the fixed gate must reject a corrupted value from the actual audio output fixture"

    split = config["split"].as_i
    config["batches"].as_i.times do |batch|
      start = (batch * seq + split) * dim
      count = (seq - split) * dim
      snapshots["audio_output"][start, count].should eq(snapshots["output"][start, count]),
        "audio injection must preserve the reference/motion suffix exactly"
    end
    snapshots["repeat_block_output"].should eq(snapshots["output"])
    snapshots["repeat_audio_output"].should eq(snapshots["audio_output"])
    snapshots["chained_block_output"].all?(&.finite?).should be_true,
      "a second block must consume the first block's resident output"
    snapshots["chained_block_output"].should_not eq(snapshots["output"]),
      "the chained block must execute on the prior block output"
    fixture["inputs"]["local_audio"].as_a.any? { |value| value.as_f != 0 }.should be_true
    prefix = fixture["config"]["split"].as_i * dim
    snapshots["audio_output"][0, prefix].should_not eq(snapshots["output"][0, prefix]),
      "nonzero audio must perturb the video prefix"
  end

  it "covers every channel in AdaIN when C exceeds 256" do
    dim = 320
    inputs = ML::Video::Arena.new
    weights = ML::Video::Arena.new
    scratch = ML::Video::Arena.new
    trace = ML::Video::Arena.new
    outputs = ML::Video::Arena.new
    begin
      cfg = ML::Video::WanS2VAudioConfig.new(
        batches: 1, seq: 2, dim: dim, heads: 20,
        video_frames: 1, spatial_tokens: 1, audio_tokens: 5,
      )
      video_row = Array(Float32).new(dim) { |i| Math.sin(i * 0.071).to_f32 }
      inputs.upload("x", video_row + Array(Float32).new(dim, 7.0_f32))
      inputs.upload("local_audio", Array(Float32).new(cfg.audio_rows * dim, 0.0_f32))
      inputs.upload("global_audio", Array(Float32).new(dim, 0.0_f32))

      weights.upload("adain_weight", Array(Float32).new(2 * dim * dim, 0.0_f32))
      adain_bias = Array(Float32).new(2 * dim, 0.0_f32)
      (256...dim).each { |i| adain_bias[i] = 0.2_f32 + (i - 256).to_f32 * 0.003_f32 }
      weights.upload("adain_bias", adain_bias)
      {"audio_q", "audio_k", "audio_v", "audio_o"}.each do |prefix|
        weights.upload("#{prefix}_weight", Array(Float32).new(dim * dim, 0.0_f32))
        weights.upload("#{prefix}_bias", Array(Float32).new(dim, 0.0_f32))
      end
      weights.upload("audio_q_norm_weight", Array(Float32).new(dim, 1.0_f32))
      weights.upload("audio_k_norm_weight", Array(Float32).new(dim, 1.0_f32))

      ML::Video::WanS2VAudioInjector.prepare!
      ML::Video::WanS2VAudioInjector.prepare_scratch!(scratch, cfg)
      y = outputs.alloc("y", cfg.values)
      adain_trace = trace.alloc("adain_video", dim)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      ML::Video::WanS2VAudioInjector.encode(
        encoder, inputs["x"], inputs["local_audio"], inputs["global_audio"],
        y, weights, scratch, cfg, {"adain_video" => adain_trace},
      )
      encoder.end_encoding
      command.commit
      command.wait

      mean = video_row.sum(&.to_f64) / dim
      variance = video_row.sum { |value| (value.to_f64 - mean) ** 2 } / dim
      denominator = Math.sqrt(variance + 1e-5)
      actual = adain_trace.read(dim)
      dim.times do |i|
        normalized = (video_row[i].to_f64 - mean) / denominator
        expected = normalized + adain_bias[i].to_f64
        actual[i].should be_close(expected, 2e-5 + 2e-5 * expected.abs),
          "AdaIN channel #{i} was not covered"
      end
      (256...dim).any? { |i| actual[i] != 0.0_f32 }.should be_true
    ensure
      outputs.release
      trace.release
      scratch.release
      weights.release
      inputs.release
    end
  end

  it "rejects overlapping no-copy Wan audio scratch wrappers" do
    page = LibWanVideoSpecMemory.getpagesize.to_u64
    ptr = Pointer(Void).null
    LibWanVideoSpecMemory.posix_memalign(pointerof(ptr), page, 3_u64 * page).should eq(0)
    inputs = ML::Video::Arena.new
    weights = ML::Video::Arena.new
    scratch = ML::Video::Arena.new
    outputs = ML::Video::Arena.new
    first = second = nil.as(ML::MetalBuffer?)
    first_in_scratch = false
    second_in_scratch = false
    begin
      dim = 8
      cfg = ML::Video::WanS2VAudioConfig.new(
        batches: 1, seq: 2, dim: dim, heads: 1,
        video_frames: 1, spatial_tokens: 1, audio_tokens: 5,
      )
      inputs.upload("x", Array(Float32).new(2 * dim, 0.1_f32))
      inputs.upload("local_audio", Array(Float32).new(5 * dim, 0.1_f32))
      inputs.upload("global_audio", Array(Float32).new(dim, 0.1_f32))
      weights.upload("adain_weight", Array(Float32).new(2 * dim * dim, 0.0_f32))
      weights.upload("adain_bias", Array(Float32).new(2 * dim, 0.0_f32))
      {"audio_q", "audio_k", "audio_v", "audio_o"}.each do |prefix|
        weights.upload("#{prefix}_weight", Array(Float32).new(dim * dim, 0.0_f32))
        weights.upload("#{prefix}_bias", Array(Float32).new(dim, 0.0_f32))
      end
      weights.upload("audio_q_norm_weight", Array(Float32).new(dim, 1.0_f32))
      weights.upload("audio_k_norm_weight", Array(Float32).new(dim, 1.0_f32))
      ML::Video::WanS2VAudioInjector.prepare_scratch!(scratch, cfg)
      outputs.alloc("y", cfg.values)
      first = ML::MetalBuffer.wrap_no_copy(ptr, (2 * page).to_i64)
      second = ML::MetalBuffer.wrap_no_copy((ptr.as(UInt8*) + page).as(Void*), (2 * page).to_i64)
      first.handle.should_not eq(second.handle)
      scratch.buffers["wan_a_gather"].release
      scratch.buffers["wan_a_q"].release
      scratch.buffers["wan_a_gather"] = first
      first_in_scratch = true
      scratch.buffers["wan_a_q"] = second
      second_in_scratch = true

      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError, "Wan audio scratch buffers overlap") do
        ML::Video::WanS2VAudioInjector.encode(
          encoder, inputs["x"], inputs["local_audio"], inputs["global_audio"],
          outputs["y"], weights, scratch, cfg,
        )
      end
      encoder.end_encoding
      command.discard
    ensure
      outputs.release
      scratch.release
      weights.release
      inputs.release
      first.try(&.release) unless first_in_scratch
      second.try(&.release) unless second_in_scratch
      LibWanVideoSpecMemory.free(ptr)
    end
  end
end
