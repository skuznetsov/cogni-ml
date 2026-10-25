require "./hy_nomas_text_probe"
require "../../src/ml/video/hy_nomas_text_refiner"
require "../../src/ml/video/hy_nomas_block_mps"

# One diagnostic consumer of a resident Qwen conditioning buffer. Setup owns
# weights, all scratch and prepared MPS plans; encode appends to the producer's
# existing chain. The caller performs the terminal drain and explicitly releases
# the session afterward.
module HyNomasTextProbe
  # Tracks the chain before a session starts any setup-owned weight conversion.
  # Session.encode may fail partway through, so the lease must outlive each
  # individual component's preflight and remain active until the terminal wait.
  class SessionChainLease(T)
    @last_chain : T? = nil

    def begin_encode!(chain : T) : Nil
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas text session cannot span in-flight command chains")
        end
      end
      # Register before the first possible dispatch. If a later prepared owner
      # or plan raises, release still sees the chain that may contain work.
      @last_chain = chain
    end

    def ensure_terminal! : Nil
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas text session command chain before release") unless chain.completed?
      end
    end
  end

  class Session
    getter named_bytes : Int64

    # The command-chain API has no deferred-resource hook. Retain sessions
    # strongly from successful setup until explicit release after terminal
    # completion, matching the refiner and workspace ownership policy.
    @@lifetime_mutex = Mutex.new
    @@live_sessions = [] of Session

    @refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
    @block : ML::Video::HyNomasDualStreamBlock? = nil
    @workspace : ML::Video::HyNomasMPSWorkspace? = nil
    @chain_lease = SessionChainLease(ML::Metal::MPSCommandChain).new

    def refiner : ML::Video::HyNomasTextRefinerMPSPlan
      @refiner.not_nil!
    end

    def block : ML::Video::HyNomasDualStreamBlock
      @block.not_nil!
    end

    def workspace : ML::Video::HyNomasMPSWorkspace
      @workspace.not_nil!
    end

    def self.retained_session_count : Int32
      @@lifetime_mutex.synchronize { @@live_sessions.size.to_i32 }
    end

    @inputs = ML::Video::Arena.new
    @text_weights = ML::Video::Arena.new
    @block_weights = ML::Video::Arena.new
    @scratch = ML::Video::Arena.new
    @prepared = [] of ML::Video::PreparedWeights
    @files = [] of ML::Video::CheckpointFile
    @released = false

    def self.config : ML::Video::HyNomasConfig
      ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 6, text_tokens: 23,
        dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192,
        attention_precision: ML::Video::HyNomasAttentionPrecision::F32Control)
    end

    def self.workspace_bytes : Int64
      config = self.config
      weight_stage = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(config).to_i64 * 4
      attention = 4_i64 * config.joint_rows * config.dim * 4
      # Separate Q and K RMS plans for each stream: square + per-head mean.
      rms = 2_i64 * config.joint_rows * (config.dim + config.heads) * 4
      weight_stage + attention + rms
    end

    # Header-only admission before any device allocation. Opaque MPS resources
    # are deliberately outside named bytes and remain externally guarded.
    def self.admit(fixture : Fixture, max_named_bytes : Int64) : Int64
      files = [] of ML::Video::CheckpointFile
      begin
        text_file = ML::Video::CheckpointFile.new(fixture.text_weights)
        files << text_file
        block_file = ML::Video::CheckpointFile.new(fixture.block_weights)
        files << block_file
        text = ML::Video::WeightPlan.new([text_file],
          ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings,
          1024_i64 * 1024 * 1024, resident_dtype: ML::Video::WeightDType::BF16,
          require_all: true, allow_f32_to_bf16: true)
        block = ML::Video::HyNomasDualStreamBlock.weight_plan([block_file], config,
          1024_i64 * 1024 * 1024, resident_dtype: ML::Video::WeightDType::BF16,
          require_all: true, allow_f32_to_bf16: true)
        input_bytes = (6_i64 * 2048 + 2_i64 * 6 * 128) * 4
        scratch_bytes = ML::Video::HyNomasDualStreamBlock.scratch_plan(config).sum { |_, count| count.to_i64 * 4 }
        bytes = text.peak_bytes + block.peak_bytes + input_bytes + scratch_bytes +
                workspace_bytes + ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(fixture.mask).total
        raise ArgumentError.new("conditioning session exceeds aggregate named byte budget") unless bytes <= max_named_bytes
        {"input.video" => [1, 6, 2048], "input.rope_cos" => [6, 128], "input.rope_sin" => [6, 128]}.each do |name, shape|
          raise ArgumentError.new("conditioning input shape mismatch: #{name}") unless fixture.shape(name) == shape
        end
        bytes
      ensure
        files.each(&.close)
      end
    end

    def initialize(@fixture : Fixture, source : ML::MetalBuffer, mask : Array(Int32), max_named_bytes : Int64)
      raise ArgumentError.new("conditioning mask differs from authenticated oracle") unless mask == @fixture.mask
      @named_bytes = self.class.admit(@fixture, max_named_bytes)
      begin
        text_file = ML::Video::CheckpointFile.new(@fixture.text_weights)
        @files << text_file
        block_file = ML::Video::CheckpointFile.new(@fixture.block_weights)
        @files << block_file
        text_plan = ML::Video::WeightPlan.new([text_file],
          ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings,
          1024_i64 * 1024 * 1024, resident_dtype: ML::Video::WeightDType::BF16,
          require_all: true, allow_f32_to_bf16: true)
        block_plan = ML::Video::HyNomasDualStreamBlock.weight_plan([block_file], self.class.config,
          1024_i64 * 1024 * 1024, resident_dtype: ML::Video::WeightDType::BF16,
          require_all: true, allow_f32_to_bf16: true)
        @prepared << ML::Video::PreparedWeights.new(text_plan, @text_weights)
        @prepared << ML::Video::PreparedWeights.new(block_plan, @block_weights)
        @refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(source, mask,
          @fixture.timestep, @text_weights,
          ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask).total)
        {"video", "rope_cos", "rope_sin"}.each do |name|
          @inputs.upload(name, @fixture.values("input.#{name}"))
        end
        ML::Video::HyNomasDualStreamBlock.prepare_scratch!(@scratch, self.class.config)
        @block = ML::Video::HyNomasDualStreamBlock.new(self.class.config, @block_weights, @scratch)
        @workspace = ML::Video::HyNomasMPSWorkspace.new(block,
          @inputs["video"], refiner.text_output, refiner.vec, refiner.vec_txt,
          @inputs["rope_cos"], @inputs["rope_sin"], allow_shared_modulation: true)
        raise "conditioning workspace named-byte estimate differs from allocation" unless workspace.resident_bytes == self.class.workspace_bytes
        expected = @fixture.document["tensors"].as_h.keys.select { |name| name.starts_with?("refiner.") || name.starts_with?("dit.") }.sort
        actual = checkpoint_buffers.keys.sort
        raise ArgumentError.new("conditioning reference checkpoint inventory differs from native plan") unless expected == actual
        retain_lifetime_root!
      rescue error
        # No commands can have been submitted by these setup constructors.
        release_lifetime_root!
        @workspace.try(&.release)
        @refiner.try(&.release)
        @prepared.each(&.release)
        {@inputs, @text_weights, @block_weights, @scratch}.each(&.release)
        @files.each(&.close)
        raise error
      end
    end

    def checkpoint_buffers : Hash(String, ML::MetalBuffer)
      result = {} of String => ML::MetalBuffer
      refiner.checkpoints.each { |name, buffer| result["refiner.#{name}"] = buffer }
      block.checkpoints.each { |name, buffer| result["dit.#{name}"] = buffer }
      result
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("conditioning session was released") if @released
      @chain_lease.begin_encode!(chain)
      @prepared.each { |weights| weights.encode(chain.encoder) }
      refiner.encode(chain)
      block.encode_mps(chain, @inputs["video"], refiner.text_output,
        refiner.vec, refiner.vec_txt, @inputs["rope_cos"], @inputs["rope_sin"], workspace)
    end

    def compare : Array(Comparison)
      checkpoint_buffers.map { |name, buffer| @fixture.compare(buffer, name) }
    end

    def release : Nil
      return if @released
      # Check the session lease first: conversion can already be in flight even
      # when a later refiner or block preflight failed before recording a lease.
      @chain_lease.ensure_terminal!
      workspace.release
      refiner.release
      @prepared.each(&.release)
      {@inputs, @text_weights, @block_weights, @scratch}.each(&.release)
      @files.each(&.close)
      @released = true
      release_lifetime_root!
    end

    private def retain_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        unless @@live_sessions.any? { |current| current.same?(self) }
          @@live_sessions << self
        end
      end
    end

    private def release_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        @@live_sessions.reject! { |current| current.same?(self) }
      end
    end
  end
end
