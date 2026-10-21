# Fixed-fixture qualification for serial reuse of one Wan S2V audio injector's
# normalized K and raw V tensors. This helper is probe-local; production routing
# and cache state remain untouched.
require "digest/sha256"
require "json"
require "set"
require "../../src/ml/video/checkpoint"
require "../../src/ml/video/metal_linear_mps"
require "../../spec/support/video_allocation_counter"

module WanAudioKVCacheProbe
  REFERENCE_SHA256  = "c2376b714ea63c184adab7f5ce1817e458e5dc9deb9d909abaccbb65cea15f1a"
  WEIGHTS_SHA256    = "c172519063c1c34fb36f7f930429ca7db655a0e335524d7487525db1df5ad220"
  MANIFEST_SHA256   = "fb7943d637bc8a6853707cd2ed284595e6744c4368083c36993fd1dd26d78da9"
  GENERATOR_SHA256  = "2f0e9388da3e3aea80bbb08242e6b990111eaf5d04981aa755697fd298d0939b"
  UPSTREAM_REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
  EXPLICIT_CAP      = 1_i64 * 1024 * 1024 * 1024
  ROWS              =   105
  DIM               = 5_120
  HEADS             =    40
  FRAMES            =    21
  TOKENS_PER_FRAME  =     5
  VALUES            = ROWS * DIM
  EPS               = 1e-6_f32
  METRIC_WORDS      =        5
  alias Metric = NamedTuple(bit_differences: UInt32, tolerance_violations: UInt32,
    nonfinite: UInt32, max_abs: Float32, max_ratio: Float32)
  WEIGHT_REVISION_NAMES = [
    "audio_k_weight", "audio_k_bias", "audio_v_weight", "audio_v_bias", "audio_k_norm_weight",
  ]

  WEIGHT_SOURCES = {
    "audio_k_weight"      => "audio_injector.injector.0.k.weight",
    "audio_k_bias"        => "audio_injector.injector.0.k.bias",
    "audio_v_weight"      => "audio_injector.injector.0.v.weight",
    "audio_v_bias"        => "audio_injector.injector.0.v.bias",
    "audio_k_norm_weight" => "audio_injector.injector.0.norm_k.weight",
  }

  SOURCE_HASHES = {
    "wan_modules_model.py"               => "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432",
    "wan_modules_s2v_model_s2v.py"       => "91fb14df44edab2d01b5634420b463c5f1563aecdf6dac905da6b3acac3145cc",
    "wan_modules_s2v_audio_utils.py"     => "76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd",
    "wan_modules_s2v_s2v_utils.py"       => "e45ae6c274b17e4fdddcb0439b598897a37679b8934ebd71f87255f92d5f0b2f",
    "wan_modules_s2v_auxi_blocks.py"     => "6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6",
    "diffusers-normalization-v0.31.0.py" => "7f2d2086f669e19bc008d71743f3963017274fbc6ad86d0ab9bf18db211e7c5a",
  }

  def self.sha256_file(path : String) : String
    digest = Digest::SHA256.new
    buffer = Bytes.new(1 << 20)
    File.open(path, "rb") do |file|
      while (count = file.read(buffer)) > 0
        digest.update(buffer[0, count])
      end
    end
    digest.final.hexstring
  end

  def self.checked_f32_file(path : String, bytes : Int64, digest : String,
                            count : Int32, name : String) : String
    raise ArgumentError.new("Missing audio K/V fixture artifact #{name}") unless File.file?(path)
    unless File.size(path) == bytes && bytes == count.to_i64 * sizeof(Float32)
      raise ArgumentError.new("Audio K/V fixture byte count mismatch for #{name}")
    end
    actual_digest = sha256_file(path)
    raise ArgumentError.new("Audio K/V fixture SHA-256 mismatch for #{name}") unless actual_digest == digest
    raw = File.read(path)
    values = raw.to_unsafe.as(Float32*)
    count.times do |index|
      raise ArgumentError.new("Audio K/V fixture contains non-finite #{name}") unless values[index].finite?
    end
    raw
  end

  record CacheIdentity,
    request_id : String,
    request_revision : UInt64,
    audio_revision : UInt64,
    weight_revisions : Array(UInt64),
    injector_id : String,
    injector_revision : UInt64,
    config_revision : UInt64,
    chain_id : UInt64,
    bindings : Array(UInt64),
    rows : Int32,
    dim : Int32,
    heads : Int32,
    input_dtype : String,
    weight_dtype : String,
    cache_dtype : String,
    eps_bits : UInt32 do
    def with(request_revision : UInt64 = self.request_revision,
             audio_revision : UInt64 = self.audio_revision,
             weight_revisions : Array(UInt64) = self.weight_revisions,
             injector_revision : UInt64 = self.injector_revision,
             config_revision : UInt64 = self.config_revision,
             chain_id : UInt64 = self.chain_id,
             bindings : Array(UInt64) = self.bindings) : self
      self.class.new(request_id, request_revision, audio_revision, weight_revisions.dup,
        injector_id, injector_revision, config_revision, chain_id, bindings.dup,
        rows, dim, heads, input_dtype, weight_dtype, cache_dtype, eps_bits)
    end

    def snapshot : self
      self.class.new(request_id, request_revision, audio_revision, weight_revisions.dup,
        injector_id, injector_revision, config_revision, chain_id, bindings.dup,
        rows, dim, heads, input_dtype, weight_dtype, cache_dtype, eps_bits)
    end
  end

  # One serial command chain only. A pending producer may be reused later in
  # that same chain; it becomes completed only after the caller reports its
  # terminal wait. A different key or chain must fill again.
  class CacheLedger
    enum State
      Invalid
      Pending
      Ready
    end

    @identity : CacheIdentity?
    @state = State::Invalid
    getter hits = 0_i64
    getter misses = 0_i64

    def lookup(identity : CacheIdentity) : Bool
      if @identity == identity && @state != State::Invalid
        @hits += 1
        true
      else
        @misses += 1
        false
      end
    end

    def begin_fill(identity : CacheIdentity) : Bool
      if @state == State::Pending && @identity != identity
        raise ArgumentError.new("Cannot replace an incomplete audio K/V producer")
      end
      return false if lookup(identity)
      @identity = identity.snapshot
      @state = State::Pending
      true
    end

    def complete_after_terminal_wait!(identity : CacheIdentity) : Nil
      unless @identity == identity && @state == State::Pending
        raise ArgumentError.new("Cannot complete an audio K/V producer that is not pending")
      end
      @state = State::Ready
    end

    def discard! : Nil
      @identity = nil
      @state = State::Invalid
    end

    def clear! : Nil
      discard!
    end

    def counter_snapshot : Tuple(Int64, Int64)
      {@hits, @misses}
    end
  end

  private SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    kernel void wan_audio_kv_fill_nan(device float *x [[buffer(0)]],
                                      constant uint &count [[buffer(1)]],
                                      uint i [[thread_position_in_grid]]) {
      if (i < count) x[i] = as_type<float>(0x7fc00000u);
    }

    kernel void wan_audio_kv_zero(device atomic_uint *metrics [[buffer(0)]],
                                  uint i [[thread_position_in_grid]]) {
      if (i < 5u) atomic_store_explicit(metrics + i, 0u, memory_order_relaxed);
    }

    // Metrics: bit differences, strict-tolerance violations, non-finite
    // values, maximum absolute error bits, maximum tolerance-ratio bits.
    // Equal finite values take no atomic path; bitwise-equal NaNs still fail.
    kernel void wan_audio_kv_compare(device const float *reference [[buffer(0)]],
                                     device const float *actual [[buffer(1)]],
                                     device atomic_uint *metrics [[buffer(2)]],
                                     constant uint &count [[buffer(3)]],
                                     uint i [[thread_position_in_grid]]) {
      if (i >= count) return;
      const float ref = reference[i], got = actual[i];
      const uint ref_bits = as_type<uint>(ref), got_bits = as_type<uint>(got);
      const bool same_bits = ref_bits == got_bits;
      if (!isfinite(ref) || !isfinite(got)) {
        if (!same_bits) atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(metrics + 2, 1u, memory_order_relaxed);
        return;
      }
      if (same_bits) return;
      atomic_fetch_add_explicit(metrics, 1u, memory_order_relaxed);
      const float error = abs(ref - got);
      const float tolerance = 2.0e-5f + 2.0e-5f * abs(ref);
      const float ratio = error / tolerance;
      if (error > tolerance)
        atomic_fetch_add_explicit(metrics + 1, 1u, memory_order_relaxed);
      if (error > 0.0f) {
        atomic_fetch_max_explicit(metrics + 3, as_type<uint>(error), memory_order_relaxed);
        atomic_fetch_max_explicit(metrics + 4, as_type<uint>(ratio), memory_order_relaxed);
      }
    }

    kernel void wan_audio_kv_mutate_f32(device float *x [[buffer(0)]],
                                        uint i [[thread_position_in_grid]]) {
      if (i == 0u) x[0] = x[0] == 0.0f ? 1.0f : 0.0f;
    }

    kernel void wan_audio_kv_mutate_bf16(device ushort *x [[buffer(0)]],
                                         uint i [[thread_position_in_grid]]) {
      if (i == 0u) {
        x[0] = x[0] == ushort(0x3f80u) ? ushort(0u) : ushort(0x3f80u);
      }
    }

    kernel void wan_audio_kv_corrupt_plus_one(device float *x [[buffer(0)]],
                                              uint i [[thread_position_in_grid]]) {
      if (i == 0u) x[0] += 1.0f;
    }

    kernel void wan_audio_kv_corrupt_nan(device float *x [[buffer(0)]],
                                         uint i [[thread_position_in_grid]]) {
      if (i == 0u) x[0] = as_type<float>(0x7fc00000u);
    }
  METAL

  class Prepared
    @arena : ML::Video::Arena
    @checkpoint : ML::Video::CheckpointFile?
    @prepared_weights : ML::Video::PreparedWeights?
    @baseline_k_plan : ML::Video::WanT5MPSLinearPlan?
    @baseline_v_plan : ML::Video::WanT5MPSLinearPlan?
    @cache_k_plan : ML::Video::WanT5MPSLinearPlan?
    @cache_v_plan : ML::Video::WanT5MPSLinearPlan?
    @weight_stage : ML::MetalBuffer
    @input : ML::MetalBuffer
    @zero_bias : ML::MetalBuffer
    @expected_raw_k : ML::MetalBuffer
    @expected_norm_k : ML::MetalBuffer
    @expected_v : ML::MetalBuffer
    @baseline_k : ML::MetalBuffer
    @baseline_v : ML::MetalBuffer
    @cache_k : ML::MetalBuffer
    @cache_v : ML::MetalBuffer
    @baseline_sink_k : ML::MetalBuffer
    @baseline_sink_v : ML::MetalBuffer
    @cache_sink_k : ML::MetalBuffer
    @cache_sink_v : ML::MetalBuffer
    @stale_k : ML::MetalBuffer
    @stale_v : ML::MetalBuffer
    @raw_k_trace : ML::MetalBuffer
    @control_expected : ML::MetalBuffer
    @control_plus : ML::MetalBuffer
    @control_nan : ML::MetalBuffer
    @metrics = {} of String => ML::MetalBuffer
    @fill_nan_pipeline : ML::Metal::ComputePipeline
    @zero_pipeline : ML::Metal::ComputePipeline
    @compare_pipeline : ML::Metal::ComputePipeline
    @mutate_f32_pipeline : ML::Metal::ComputePipeline
    @mutate_bf16_pipeline : ML::Metal::ComputePipeline
    @corrupt_plus_pipeline : ML::Metal::ComputePipeline
    @corrupt_nan_pipeline : ML::Metal::ComputePipeline
    @fixture_path : String
    @fixture_sha256 : String
    @weights_path : String
    @weights_sha256 : String
    @source_manifest_path : String
    @generator_path : String
    @source_paths = {} of String => String
    @artifact_paths = {} of String => String
    @artifact_sha256 = {} of String => String
    @ledger = CacheLedger.new
    @chain_number = 0_u64
    @released = false

    def initialize(fixture_path : String)
      @checkpoint = nil
      @prepared_weights = nil
      @baseline_k_plan = nil
      @baseline_v_plan = nil
      @cache_k_plan = nil
      @cache_v_plan = nil
      @fixture_path = File.expand_path(fixture_path)
      unless File.file?(@fixture_path)
        raise ArgumentError.new("Missing pinned Wan audio K/V reference JSON")
      end
      @fixture_sha256 = WanAudioKVCacheProbe.sha256_file(@fixture_path)
      unless @fixture_sha256 == REFERENCE_SHA256
        raise ArgumentError.new("Pinned Wan audio K/V reference SHA-256 mismatch")
      end
      manifest = JSON.parse(File.read(@fixture_path))
      validate_manifest!(manifest)

      files = manifest["files"]
      artifacts = manifest["artifacts"]
      expected_bytes = VALUES.to_i64 * sizeof(Float32)
      input_raw = WanAudioKVCacheProbe.checked_f32_file(files["local_audio"].as_s,
        artifacts["local_audio"]["bytes"].as_i64, artifacts["local_audio"]["sha256"].as_s,
        VALUES, "local_audio")
      raw_k = WanAudioKVCacheProbe.checked_f32_file(files["raw_k"].as_s,
        artifacts["raw_k"]["bytes"].as_i64, artifacts["raw_k"]["sha256"].as_s,
        VALUES, "raw_k")
      norm_k = WanAudioKVCacheProbe.checked_f32_file(files["normalized_k"].as_s,
        artifacts["normalized_k"]["bytes"].as_i64, artifacts["normalized_k"]["sha256"].as_s,
        VALUES, "normalized_k")
      value_v = WanAudioKVCacheProbe.checked_f32_file(files["v"].as_s,
        artifacts["v"]["bytes"].as_i64, artifacts["v"]["sha256"].as_s,
        VALUES, "v")
      unless {input_raw.bytesize, raw_k.bytesize, norm_k.bytesize, value_v.bytesize}.all? do |size|
               size.to_i64 == expected_bytes
             end
        raise ArgumentError.new("Wan audio K/V fixture data length mismatch")
      end

      weights_path = manifest["weights_path"].as_s
      unless Path.new(weights_path).absolute?
        raise ArgumentError.new("Audio K/V source checkpoint must use an absolute path")
      end
      unless File.file?(weights_path) && File.size(weights_path) == manifest["weights_bytes"].as_i64 &&
             manifest["weights_bytes"].as_i64 == 1_017_447_344_i64
        raise ArgumentError.new("Audio K/V checkpoint sidecar size mismatch")
      end
      @weights_path = weights_path
      @weights_sha256 = WanAudioKVCacheProbe.sha256_file(@weights_path)
      unless @weights_sha256 == WEIGHTS_SHA256 && @weights_sha256 == manifest["weights_sha256"].as_s
        raise ArgumentError.new("Audio K/V checkpoint sidecar SHA-256 mismatch")
      end
      validate_source_provenance!(manifest)
      @source_manifest_path = manifest["input_manifest"].as_s
      @generator_path = manifest["generator"]["path"].as_s
      manifest["upstream_sources"].as_h.each do |name, source|
        @source_paths[name] = source["path"].as_s
      end
      @artifact_paths = files.as_h.transform_values(&.as_s)
      artifacts.as_h.each do |name, data|
        @artifact_sha256[name] = data["sha256"].as_s
      end

      @arena = ML::Video::Arena.new
      ML::Video::MetalLinearMPS.prepare!
      @fill_nan_pipeline = ML::Metal::ComputePipeline.new("wan_audio_kv_fill_nan", SOURCE)
      @zero_pipeline = ML::Metal::ComputePipeline.new("wan_audio_kv_zero", SOURCE)
      @compare_pipeline = ML::Metal::ComputePipeline.new("wan_audio_kv_compare", SOURCE)
      @mutate_f32_pipeline = ML::Metal::ComputePipeline.new("wan_audio_kv_mutate_f32", SOURCE)
      @mutate_bf16_pipeline = ML::Metal::ComputePipeline.new("wan_audio_kv_mutate_bf16", SOURCE)
      @corrupt_plus_pipeline = ML::Metal::ComputePipeline.new("wan_audio_kv_corrupt_plus_one", SOURCE)
      @corrupt_nan_pipeline = ML::Metal::ComputePipeline.new("wan_audio_kv_corrupt_nan", SOURCE)
      ML::Video::MetalOps.pipeline("video_copy_f32")
      ML::Video::MetalOps.pipeline("video_norm_mixed")
      ML::Video::WanT5.pipeline("wan_t5_expand_fc2")

      @input = @arena.alloc("audio_kv_local_audio", VALUES)
      @zero_bias = @arena.alloc("audio_kv_zero_bias", DIM)
      @expected_raw_k = @arena.alloc("audio_kv_expected_raw_k", VALUES)
      @expected_norm_k = @arena.alloc("audio_kv_expected_norm_k", VALUES)
      @expected_v = @arena.alloc("audio_kv_expected_v", VALUES)
      @weight_stage = @arena.alloc("audio_kv_weight_stage_f32", DIM * DIM)
      @baseline_k = @arena.alloc("audio_kv_baseline_k", VALUES)
      @baseline_v = @arena.alloc("audio_kv_baseline_v", VALUES)
      @cache_k = @arena.alloc("audio_kv_cache_k", VALUES)
      @cache_v = @arena.alloc("audio_kv_cache_v", VALUES)
      @baseline_sink_k = @arena.alloc("audio_kv_baseline_sink_k", VALUES)
      @baseline_sink_v = @arena.alloc("audio_kv_baseline_sink_v", VALUES)
      @cache_sink_k = @arena.alloc("audio_kv_cache_sink_k", VALUES)
      @cache_sink_v = @arena.alloc("audio_kv_cache_sink_v", VALUES)
      @stale_k = @arena.alloc("audio_kv_stale_k", VALUES)
      @stale_v = @arena.alloc("audio_kv_stale_v", VALUES)
      @raw_k_trace = @arena.alloc("audio_kv_raw_k_trace", VALUES)
      @control_expected = @arena.alloc("audio_kv_control_expected", 1)
      @control_plus = @arena.alloc("audio_kv_control_plus", 1)
      @control_nan = @arena.alloc("audio_kv_control_nan", 1)
      metric_names.each do |name|
        @metrics[name] = @arena.alloc("audio_kv_metrics_#{name}", METRIC_WORDS)
      end
      @input.write_bytes(input_raw.to_unsafe.as(Pointer(UInt8)), expected_bytes)
      @zero_bias.write(Array(Float32).new(DIM, 0.0_f32))
      @expected_raw_k.write_bytes(raw_k.to_unsafe.as(Pointer(UInt8)), expected_bytes)
      @expected_norm_k.write_bytes(norm_k.to_unsafe.as(Pointer(UInt8)), expected_bytes)
      @expected_v.write_bytes(value_v.to_unsafe.as(Pointer(UInt8)), expected_bytes)
      @control_expected.write([0.5_f32])
      @control_plus.write([0.5_f32])
      @control_nan.write([0.5_f32])

      checkpoint = ML::Video::CheckpointFile.new(@weights_path)
      @checkpoint = checkpoint
      bindings = [] of ML::Video::WeightBinding
      WEIGHT_SOURCES.each do |target, source|
        shape = if target == "audio_k_weight" || target == "audio_v_weight"
                  [DIM.to_i64, DIM.to_i64]
                else
                  [DIM.to_i64]
                end
        bindings << ML::Video::WeightBinding.new(target, source, shape,
          ML::GGUF::SafeTensorDType::BF16)
      end
      weight_plan = ML::Video::WeightPlan.new([checkpoint], bindings, EXPLICIT_CAP,
        @arena.resident_bytes, ML::Video::WeightDType::BF16, require_all: false)
      expected_weight_bytes = (2_i64 * DIM * DIM + 3_i64 * DIM) * sizeof(UInt16)
      unless weight_plan.entries.size == 5 && weight_plan.destination_bytes == expected_weight_bytes
        raise ArgumentError.new("Audio K/V selected-weight plan admitted an unexpected tensor set")
      end
      prepared = ML::Video::PreparedWeights.new(weight_plan, @arena)
      @prepared_weights = prepared
      cfg = ML::Video::WanT5Config.new(vocab: 8, dim: DIM, ffn_dim: DIM,
        heads: HEADS, layers: 1, batches: 1, tokens: ROWS)
      @baseline_k_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, DIM, DIM, @input,
        @weight_stage, @baseline_k)
      @baseline_v_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, DIM, DIM, @input,
        @weight_stage, @baseline_v)
      @cache_k_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, DIM, DIM, @input,
        @weight_stage, @cache_k)
      @cache_v_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, DIM, DIM, @input,
        @weight_stage, @cache_v)
      if @arena.resident_bytes > EXPLICIT_CAP
        raise ArgumentError.new("Audio K/V explicit allocation estimate exceeds 1 GiB")
      end
    end

    def setup_json : String
      {
        schema: "wan_s2v_audio_kv_setup_v1",
        domain: "wan.s2v.audio_kv",
        fixture_path: @fixture_path,
        fixture_sha256: @fixture_sha256,
        source_checkpoint_path: @weights_path,
        source_checkpoint_sha256: @weights_sha256,
        source_manifest_sha256: MANIFEST_SHA256,
        source_manifest_path: @source_manifest_path,
        generator_path: @generator_path,
        generator_sha256: GENERATOR_SHA256,
        upstream_sources: SOURCE_HASHES.keys.sort.map do |name|
          {name: name, path: @source_paths[name], sha256: SOURCE_HASHES[name]}
        end,
        upstream_revision: UPSTREAM_REVISION,
        rows: ROWS, dim: DIM, heads: HEADS, batches: 1, frames: FRAMES,
        tokens_per_frame: TOKENS_PER_FRAME, eps: EPS,
        activation_dtype: "F32", source_weight_dtype: "BF16",
        selected_checkpoint_tensors: WEIGHT_SOURCES.values,
        require_all_checkpoint_tensors: false,
        explicit_buffer_bytes: @arena.resident_bytes,
        explicit_buffer_cap_bytes: EXPLICIT_CAP,
        oracle_artifacts: @artifact_paths.keys.sort.map do |name|
          {name: name, path: @artifact_paths[name], sha256: @artifact_sha256[name]}
        end,
        activation_readbacks: 0,
        scope: "fixed injector 0 K projection, K RMS normalization, and V projection; not full injection or generation",
      }.to_json
    end

    def encode_nan_initialization!(encoder : ML::Metal::ComputeEncoder) : Nil
      # BF16 bytes are staged during PreparedWeights construction; this validates
      # the selected destination bindings on the untimed setup command.
      @prepared_weights.not_nil!.encode(encoder)
      {@baseline_k, @baseline_v, @cache_k, @cache_v,
       @baseline_sink_k, @baseline_sink_v, @cache_sink_k, @cache_sink_v,
       @stale_k, @stale_v, @raw_k_trace}.each do |buffer|
        encode_fill_nan(encoder, buffer, VALUES)
      end
    end

    # Saves raw K before in-place K RMS normalization, then checks raw K,
    # normalized K, and V against the fixture using only compact GPU metrics.
    def validate_upstream! : Hash(String, Metric)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      allocation_start = ML::MetalBuffer.video_spec_allocations
      encode_project(encoder, @baseline_k, @baseline_v, true)
      ML::Video::MetalOps.copy(encoder, @baseline_k, @raw_k_trace, VALUES)
      encode_norm(encoder, @baseline_k)
      encode_compare(encoder, "upstream_raw_k", @expected_raw_k, @raw_k_trace, VALUES)
      encode_compare(encoder, "upstream_norm_k", @expected_norm_k, @baseline_k, VALUES)
      encode_compare(encoder, "upstream_v", @expected_v, @baseline_v, VALUES)
      encoder.end_encoding
      raise "Audio K/V diagnostic encode allocated MetalBuffers" unless ML::MetalBuffer.video_spec_allocations == allocation_start
      command.commit_and_wait_gpu_elapsed_seconds
      result = read_metrics(%w[upstream_raw_k upstream_norm_k upstream_v])
      require_strict!(result, %w[upstream_raw_k upstream_norm_k upstream_v])
      result
    end

    # Caller owns the command and its sole terminal wait.
    def encode_arm!(encoder : ML::Metal::ComputeEncoder, arm : String,
                    iterations : Int32) : Tuple(Int32, Int32, CacheIdentity?)
      raise "Audio K/V probe has been released" if @released
      allocation_start = ML::MetalBuffer.video_spec_allocations
      if arm == "recompute"
        iterations.times do
          encode_project(encoder, @baseline_k, @baseline_v, true)
          encode_norm(encoder, @baseline_k)
          encode_consumption(encoder, @baseline_k, @baseline_v,
            @baseline_sink_k, @baseline_sink_v)
        end
        raise "Audio K/V recompute encode allocated MetalBuffers" unless ML::MetalBuffer.video_spec_allocations == allocation_start
        return {iterations, 0, nil}
      end
      raise ArgumentError.new("Unknown audio K/V arm #{arm}") unless arm == "cached"

      @chain_number += 1
      identity = initial_identity(@chain_number)
      raise "Fresh command chain unexpectedly reported an audio K/V cache hit" unless @ledger.begin_fill(identity)
      encode_project(encoder, @cache_k, @cache_v, false)
      encode_norm(encoder, @cache_k)
      hits = 0
      iterations.times do |index|
        if index > 0
          raise "Same-chain audio K/V cache hit was not available" unless @ledger.lookup(identity)
          hits += 1
        end
        encode_consumption(encoder, @cache_k, @cache_v,
          @cache_sink_k, @cache_sink_v)
      end
      unless ML::MetalBuffer.video_spec_allocations == allocation_start
        @ledger.discard!
        raise "Audio K/V cache encode allocated MetalBuffers"
      end
      {1, hits, identity}
    end

    def complete_cached_arm_after_terminal_wait!(identity : CacheIdentity?) : Nil
      return unless key = identity
      @ledger.complete_after_terminal_wait!(key)
    end

    def discard_pending! : Nil
      @ledger.discard!
    end

    def validate_native_pair! : Hash(String, Metric)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      encode_compare(encoder, "native_k", @baseline_sink_k, @cache_sink_k, VALUES)
      encode_compare(encoder, "native_v", @baseline_sink_v, @cache_sink_v, VALUES)
      encoder.end_encoding
      command.commit_and_wait_gpu_elapsed_seconds
      result = read_metrics(%w[native_k native_v])
      require_bitwise!(result, %w[native_k native_v])
      result
    end

    # Real GPU mutation controls are serialized into one command buffer so each
    # stale hit is attempted on the same identity/chain that produced its cache.
    def validate_lifecycle_mutations! : Hash(String, Metric)
      raise "Audio K/V probe has been released" if @released
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      allocation_start = ML::MetalBuffer.video_spec_allocations
      @chain_number += 1
      identity = initial_identity(@chain_number)
      raise "Fresh mutation-control chain did not miss" unless @ledger.begin_fill(identity)
      encode_project(encoder, @cache_k, @cache_v, false)
      encode_norm(encoder, @cache_k)

      identity = encode_mutation_cycle!(encoder, "audio", identity, @mutate_f32_pipeline,
        @input, true)
      WEIGHT_REVISION_NAMES.each_with_index do |name, index|
        identity = encode_mutation_cycle!(encoder, name, identity, @mutate_bf16_pipeline,
          @arena[name], false, index)
      end

      # The two positive controls use independent metric buffers so neither can
      # hide the other's failure. They execute after all expected stale checks.
      encoder.set_pipeline(@corrupt_plus_pipeline)
        .set_buffer(@control_plus, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
      encode_compare(encoder, "control_plus", @control_expected, @control_plus, 1)
      encoder.set_pipeline(@corrupt_nan_pipeline)
        .set_buffer(@control_nan, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
      encode_compare(encoder, "control_nan", @control_expected, @control_nan, 1)

      encoder.end_encoding
      allocations = ML::MetalBuffer.video_spec_allocations - allocation_start
      unless allocations == 0
        @ledger.discard!
        raise "Audio K/V lifecycle encode allocated MetalBuffers"
      end
      command.commit_and_wait_gpu_elapsed_seconds
      @ledger.complete_after_terminal_wait!(identity)

      names = mutation_metric_names + mutation_refill_metric_names + %w[control_plus control_nan]
      result = read_metrics(names)
      require_mutation_rejections!(result)
      require_revision_refills!(result)
      require_positive_controls!(result)
      result
    rescue ex
      @ledger.discard!
      raise ex
    end

    def lifecycle_contract_json : String
      base = initial_identity(11_u64)
      ledger = CacheLedger.new
      raise "Lifecycle contract initial fill failed" unless ledger.begin_fill(base)
      raise "Pending same-chain cache reuse failed" unless ledger.lookup(base.snapshot)
      ledger.complete_after_terminal_wait!(base)
      raise "Ready same-chain cache reuse failed" unless ledger.lookup(base.snapshot)

      request_change = base.with(request_revision: base.request_revision + 1_u64)
      audio_change = base.with(audio_revision: base.audio_revision + 1_u64)
      weights_change = base.with(weight_revisions: base.weight_revisions.map(&.+(1_u64)))
      injector_change = base.with(injector_revision: base.injector_revision + 1_u64)
      config_change = base.with(config_revision: base.config_revision + 1_u64)
      chain_change = base.with(chain_id: base.chain_id + 1_u64)
      binding_change = base.bindings.dup
      binding_change[0] += 16_u64
      bindings_identity = base.with(bindings: binding_change)
      mismatches = {
        "request_revision"  => request_change,
        "audio_revision"    => audio_change,
        "weight_revisions"  => weights_change,
        "injector_revision" => injector_change,
        "config_revision"   => config_change,
        "fresh_chain"       => chain_change,
        "buffer_binding"    => bindings_identity,
      }
      mismatches.each do |name, changed|
        raise "Lifecycle key failed to invalidate #{name}" if ledger.lookup(changed)
      end

      clear_ledger = CacheLedger.new
      clear_ledger.begin_fill(base)
      clear_ledger.clear!
      raise "Explicit clear failed to invalidate cache" if clear_ledger.lookup(base)

      alias_ledger = CacheLedger.new
      aliased_identity = base.snapshot
      alias_ledger.begin_fill(aliased_identity)
      aliased_identity.weight_revisions[0] += 1_u64
      aliased_identity.bindings[0] += 16_u64
      raise "Cache identity snapshot retained caller-owned arrays" unless alias_ledger.lookup(base)

      pending = CacheLedger.new
      pending.begin_fill(base)
      pending_key_rejected = false
      begin
        pending.begin_fill(request_change)
      rescue ArgumentError
        pending_key_rejected = true
      end
      raise "Pending producer allowed replacement by a different identity" unless pending_key_rejected

      aborted = CacheLedger.new
      aborted.begin_fill(base)
      aborted.discard!
      raise "Discarded producer was advertised as valid" if aborted.lookup(base)

      {
        schema:                         "wan_s2v_audio_kv_lifecycle_contract_v1",
        pending_same_chain_hit:         true,
        completed_same_chain_hit:       true,
        changed_keys_miss:              mismatches.keys,
        explicit_clear_miss:            true,
        identity_arrays_snapshotted:    true,
        discarded_producer_invalidated: true,
        caller_revision_obligation:     "same-address content mutation without revision change intentionally remains a hit; GPU controls below prove stale-result detection",
        identity_scope:                 "one serial command chain only",
      }.to_json
    end

    def release_after_terminal_wait! : Nil
      return if @released
      @ledger.discard!
      @baseline_k_plan.try(&.release)
      @baseline_v_plan.try(&.release)
      @cache_k_plan.try(&.release)
      @cache_v_plan.try(&.release)
      @prepared_weights.try(&.release)
      @checkpoint.try(&.close)
      @arena.release
      @released = true
    end

    private def encode_mutation_cycle!(encoder : ML::Metal::ComputeEncoder,
                                       label : String, identity : CacheIdentity,
                                       mutation_pipeline : ML::Metal::ComputePipeline,
                                       source : ML::MetalBuffer,
                                       input_mutation : Bool,
                                       weight_index : Int32 = -1) : CacheIdentity
      encode_consumption(encoder, @cache_k, @cache_v, @stale_k, @stale_v)
      encoder.set_pipeline(mutation_pipeline)
        .set_buffer(source, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
      unless @ledger.lookup(identity)
        raise "Same-address unannounced mutation did not expose the existing stale cache key for #{label}"
      end

      encode_project(encoder, @baseline_k, @baseline_v, true)
      encode_norm(encoder, @baseline_k)
      encode_compare(encoder, "unannounced_#{label}_k", @baseline_k, @stale_k, VALUES)
      encode_compare(encoder, "unannounced_#{label}_v", @baseline_v, @stale_v, VALUES)

      revised = if input_mutation
                  identity.with(audio_revision: identity.audio_revision + 1_u64)
                else
                  revisions = identity.weight_revisions.dup
                  revisions[weight_index] += 1_u64
                  identity.with(weight_revisions: revisions)
                end
      # Prior cache consumers and stale comparisons are already ordered before
      # this point. Explicit discard prevents replacing a pending producer.
      @ledger.discard!
      raise "Revised audio K/V identity did not force a miss for #{label}" unless @ledger.begin_fill(revised)
      encode_project(encoder, @cache_k, @cache_v, false)
      encode_norm(encoder, @cache_k)
      encode_compare(encoder, "revision_#{label}_k", @baseline_k, @cache_k, VALUES)
      encode_compare(encoder, "revision_#{label}_v", @baseline_v, @cache_v, VALUES)
      revised
    end

    private def validate_manifest!(manifest : JSON::Any) : Nil
      unless manifest["schema"].as_s == "wan_s2v_audio_kv_reference_v1" &&
             manifest["rows"].as_i == ROWS && manifest["dim"].as_i == DIM &&
             manifest["heads"].as_i == HEADS && manifest["batches"].as_i == 1 &&
             manifest["frames"].as_i == FRAMES && manifest["tokens_per_frame"].as_i == TOKENS_PER_FRAME &&
             manifest["dtype"].as_s == "F32" && manifest["weight_dtype"].as_s == "BF16" &&
             manifest["eps"].as_f == 1.0e-6 && manifest["cpu_fallback"].as_bool == false
        raise ArgumentError.new("Audio K/V fixture geometry, dtype, epsilon, or MPS provenance mismatch")
      end
      files = manifest["files"].as_h
      artifacts = manifest["artifacts"].as_h
      expected_names = Set{"local_audio", "raw_k", "normalized_k", "v"}
      unless files.keys.to_set == expected_names && artifacts.keys.to_set == expected_names
        raise ArgumentError.new("Audio K/V fixture must bind exactly four local F32 artifacts")
      end
      files.each_value do |path|
        raise ArgumentError.new("Audio K/V fixture paths must be absolute") unless Path.new(path.as_s).absolute?
      end
      bindings = manifest["weight_bindings"].as_a
      raise ArgumentError.new("Audio K/V fixture must bind exactly five checkpoint tensors") unless bindings.size == 5
      WEIGHT_SOURCES.each do |_, source|
        matches = bindings.select { |binding| binding["source"].as_s == source }
        raise ArgumentError.new("Audio K/V fixture binding mismatch for #{source}") unless matches.size == 1
        shape = source.ends_with?(".weight") && !source.ends_with?("norm_k.weight") ? [DIM.to_i64, DIM.to_i64] : [DIM.to_i64]
        binding = matches.first
        unless binding["dtype"].as_s == "BF16" && binding["shape"].as_a.map(&.as_i) == shape
          raise ArgumentError.new("Audio K/V fixture shape/dtype mismatch for #{source}")
        end
      end
      raise ArgumentError.new("Audio K/V fixture checkpoint hash mismatch") unless manifest["weights_sha256"].as_s == WEIGHTS_SHA256 &&
                                                                                   manifest["weights_bytes"].as_i64 == 1_017_447_344_i64
    end

    private def validate_source_provenance!(manifest : JSON::Any) : Nil
      unless manifest["input_manifest_sha256"].as_s == MANIFEST_SHA256 &&
             manifest["upstream_revision"].as_s == UPSTREAM_REVISION
        raise ArgumentError.new("Audio K/V upstream manifest/revision provenance mismatch")
      end
      input_manifest = manifest["input_manifest"].as_s
      raise ArgumentError.new("Audio K/V source manifest must be absolute") unless Path.new(input_manifest).absolute?
      unless File.file?(input_manifest) && WanAudioKVCacheProbe.sha256_file(input_manifest) == MANIFEST_SHA256
        raise ArgumentError.new("Audio K/V source manifest SHA-256 mismatch")
      end
      generator = manifest["generator"]
      generator_path = generator["path"].as_s
      unless Path.new(generator_path).absolute? && generator["sha256"].as_s == GENERATOR_SHA256 &&
             File.file?(generator_path) && WanAudioKVCacheProbe.sha256_file(generator_path) == GENERATOR_SHA256
        raise ArgumentError.new("Audio K/V reference generator provenance mismatch")
      end
      sources = manifest["upstream_sources"].as_h
      unless sources.keys.to_set == SOURCE_HASHES.keys.to_set
        raise ArgumentError.new("Audio K/V fixture upstream source inventory mismatch")
      end
      SOURCE_HASHES.each do |name, expected_hash|
        source = sources[name]
        source_path = source["path"].as_s
        unless Path.new(source_path).absolute? && source["sha256"].as_s == expected_hash &&
               File.file?(source_path) && WanAudioKVCacheProbe.sha256_file(source_path) == expected_hash
          raise ArgumentError.new("Audio K/V upstream source provenance mismatch for #{name}")
        end
      end
    end

    private def metric_names : Array(String)
      names = %w[upstream_raw_k upstream_norm_k upstream_v native_k native_v control_plus control_nan]
      names.concat(mutation_metric_names)
      names.concat(mutation_refill_metric_names)
      names
    end

    private def mutation_metric_names : Array(String)
      labels = ["audio"] + WEIGHT_REVISION_NAMES
      labels.flat_map { |label| ["unannounced_#{label}_k", "unannounced_#{label}_v"] }
    end

    private def mutation_refill_metric_names : Array(String)
      labels = ["audio"] + WEIGHT_REVISION_NAMES
      labels.flat_map { |label| ["revision_#{label}_k", "revision_#{label}_v"] }
    end

    private def initial_identity(chain_id : UInt64) : CacheIdentity
      weight_names = WEIGHT_REVISION_NAMES
      bindings = [@input, @arena[weight_names[0]], @arena[weight_names[1]],
                  @arena[weight_names[2]], @arena[weight_names[3]], @arena[weight_names[4]],
                  @zero_bias, @weight_stage, @cache_k, @cache_v].map do |buffer|
        buffer.handle.address.to_u64
      end
      bindings.concat([
        @baseline_k_plan.not_nil!.handle.address.to_u64,
        @baseline_v_plan.not_nil!.handle.address.to_u64,
        @cache_k_plan.not_nil!.handle.address.to_u64,
        @cache_v_plan.not_nil!.handle.address.to_u64,
      ])
      CacheIdentity.new("audio-kv-probe-request", 0_u64, 0_u64, Array(UInt64).new(5, 0_u64),
        "audio-injector-0", 0_u64, 0_u64, chain_id, bindings, ROWS, DIM, HEADS,
        "F32", "BF16", "F32", EPS.unsafe_as(UInt32))
    end

    private def encode_fill_nan(encoder : ML::Metal::ComputeEncoder,
                                buffer : ML::MetalBuffer, count : Int32) : Nil
      encoder.set_pipeline(@fill_nan_pipeline)
        .set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 1).dispatch_1d(count)
    end

    private def encode_project(encoder : ML::Metal::ComputeEncoder,
                               k_output : ML::MetalBuffer, v_output : ML::MetalBuffer,
                               baseline : Bool) : Nil
      k_plan = baseline ? @baseline_k_plan.not_nil! : @cache_k_plan.not_nil!
      v_plan = baseline ? @baseline_v_plan.not_nil! : @cache_v_plan.not_nil!
      ML::Video::MetalLinearMPS.project(encoder, @input, @arena["audio_k_weight"],
        @arena["audio_k_bias"], k_output, ROWS, DIM, DIM,
        ML::Video::WeightDType::BF16, ML::Video::WeightDType::BF16,
        k_plan, @weight_stage)
      ML::Video::MetalLinearMPS.project(encoder, @input, @arena["audio_v_weight"],
        @arena["audio_v_bias"], v_output, ROWS, DIM, DIM,
        ML::Video::WeightDType::BF16, ML::Video::WeightDType::BF16,
        v_plan, @weight_stage)
    end

    private def encode_norm(encoder : ML::Metal::ComputeEncoder, k_buffer : ML::MetalBuffer) : Nil
      ML::Video::MetalOps.norm(encoder, k_buffer, @arena["audio_k_norm_weight"],
        @zero_bias, k_buffer, ROWS, DIM, EPS, rms: true, affine: true,
        weight_dtype: ML::Video::WeightDType::BF16,
        bias_dtype: ML::Video::WeightDType::F32)
    end

    private def encode_consumption(encoder : ML::Metal::ComputeEncoder,
                                   k_buffer : ML::MetalBuffer, v_buffer : ML::MetalBuffer,
                                   k_sink : ML::MetalBuffer, v_sink : ML::MetalBuffer) : Nil
      ML::Video::MetalOps.copy(encoder, k_buffer, k_sink, VALUES)
      ML::Video::MetalOps.copy(encoder, v_buffer, v_sink, VALUES)
    end

    private def encode_compare(encoder : ML::Metal::ComputeEncoder, name : String,
                               reference : ML::MetalBuffer, actual : ML::MetalBuffer,
                               count : Int32) : Nil
      metrics = @metrics[name]
      encoder.set_pipeline(@zero_pipeline)
        .set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(METRIC_WORDS)
      encoder.set_pipeline(@compare_pipeline)
        .set_buffer(reference, 0).set_buffer(actual, 1)
        .set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_value(count.to_u32, 3).dispatch_1d(count)
    end

    private def read_metrics(names : Array(String)) : Hash(String, Metric)
      names.to_h do |name|
        words = @metrics[name].contents.as(UInt32*)
        {name, {
          bit_differences: words[0], tolerance_violations: words[1], nonfinite: words[2],
          max_abs: words[3].unsafe_as(Float32), max_ratio: words[4].unsafe_as(Float32),
        }}
      end
    end

    private def require_strict!(metrics : Hash(String, Metric), names : Array(String)) : Nil
      names.each do |name|
        metric = metrics[name]
        unless metric[:tolerance_violations] == 0 && metric[:nonfinite] == 0
          raise "Audio K/V strict finite comparison failed for #{name}: #{metric}"
        end
      end
    end

    private def require_bitwise!(metrics : Hash(String, Metric), names : Array(String)) : Nil
      names.each do |name|
        metric = metrics[name]
        unless metric[:bit_differences] == 0 && metric[:tolerance_violations] == 0 && metric[:nonfinite] == 0
          raise "Audio K/V native bitwise comparison failed for #{name}: #{metric}"
        end
      end
    end

    private def require_mutation_rejections!(metrics : Hash(String, Metric)) : Nil
      labels = ["audio"] + WEIGHT_REVISION_NAMES
      labels.each do |label|
        affected = case label
                   when "audio"                          then %w[k v]
                   when "audio_v_weight", "audio_v_bias" then %w[v]
                   else                                       %w[k]
                   end
        affected.each do |stream|
          metric = metrics["unannounced_#{label}_#{stream}"]
          unless metric[:bit_differences] > 0 && metric[:tolerance_violations] > 0 && metric[:nonfinite] == 0
            raise "Audio K/V stale-cache control failed to reject same-address #{label} mutation for #{stream}: #{metric}"
          end
        end
      end
    end

    private def require_revision_refills!(metrics : Hash(String, Metric)) : Nil
      require_bitwise!(metrics, mutation_refill_metric_names)
    end

    private def require_positive_controls!(metrics : Hash(String, Metric)) : Nil
      plus = metrics["control_plus"]
      unless plus[:bit_differences] == 1 && plus[:tolerance_violations] == 1 && plus[:nonfinite] == 0
        raise "Audio K/V +1 positive control failed: #{plus}"
      end
      nan = metrics["control_nan"]
      unless nan[:bit_differences] == 1 && nan[:tolerance_violations] == 1 && nan[:nonfinite] == 1
        raise "Audio K/V NaN positive control failed: #{nan}"
      end
    end
  end
end
