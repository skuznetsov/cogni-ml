require "./hy_nomas_pretrained_serial_probe"
require "../../src/ml/video/bf16_weight_cache"

# Setup-only derivative qualification. Neural arithmetic stays on GPU; raw
# weight bytes and activation diagnostics are inspected only after the normal
# request terminal. The saved upstream reference is not a fresh block rerun.
module HyNomasBF16CacheProbe
  extend self
  alias Serial = HyNomasPretrainedSerialProbe
  alias Comparison = NamedTuple(name: String, elements: Int32, failures: Int32,
    max_abs: Float64, upstream_raw_differences: Int32, control_raw_differences: Int32)
  alias WeightComparison = NamedTuple(name: String, words: Int64,
    different_words: Int64, native_sha256: String, upstream_sha256: String)
  alias Phase = NamedTuple(name: String, destination_bytes: Int64, staging_bytes: Int64,
    named_bytes: Int64, encode_allocations: Int64, preterminal_activation_reads: Int64,
    terminal_waits: Int32, postterminal_activation_reads: Int64,
    conversion_dispatches: Int32, weights: Array(WeightComparison), captures: Array(Comparison))

  def named_peak(plan : ML::Video::WeightPlan) : Int64
    c = Serial::CONFIG
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(c).sum { |_, count| count.to_i64 * 4 }
    workspace = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(c).to_i64 * 4 +
                4_i64 * c.joint_rows * c.dim * 4 +
                2_i64 * c.joint_rows * (c.dim + c.heads) * 4
    input = Serial::INPUT_SHAPES.values.sum do |shape|
      Serial.checked_element_count(shape).to_i64 * 4
    end
    plan.peak_bytes + scratch + workspace + input
  end

  def admit!(plan : ML::Video::WeightPlan, budget : Int64) : Int64
    unless 0 < budget <= Serial::NAMED_BUDGET_LIMIT_BYTES
      raise ArgumentError.new("cache probe named budget exceeds the admitted process allowance")
    end
    peak = named_peak(plan)
    raise ArgumentError.new("cache probe named peak #{peak} exceeds budget #{budget}") if peak > budget
    peak
  end

  # Shape equality alone is insufficient: the exact original source names,
  # complete inventory and physical BF16 storage must all agree before Metal.
  def validate_oracle!(plan : ML::Video::WeightPlan, oracle : ML::Video::CheckpointFile,
                       block_index : Int32 = 0) : Nil
    entries = plan.entries
    tensors = oracle.tensors
    unless 0 <= block_index <= 1 && entries.size == 32 &&
           entries.all?(&.binding.source.starts_with?("double_blocks.#{block_index}.")) &&
           tensors.keys.to_set == entries.map(&.binding.source).to_set
      raise ArgumentError.new("BF16 oracle must contain exactly the 32 block-#{block_index} tensors")
    end
    entries.each do |entry|
      info = tensors[entry.binding.source]
      unless info.dtype.bf16? && info.shape == entry.binding.shape && info.n_elements == entry.info.n_elements
        raise ArgumentError.new("BF16 oracle source/shape/dtype mismatch: #{info.name}")
      end
    end
  end

  def block_pin(index : Int32) : Serial::BlockPin
    case index
    when 0 then Serial::BLOCK0_PIN
    when 1 then Serial::BLOCK1_PIN
    else        raise ArgumentError.new("cache block index must be 0 or 1")
    end
  end

  # Keep the derivative private until the producer, consumer and repeat gates
  # all succeed and their owner has released after terminal completion. A
  # failed callback cannot leave a final cache that implies qualification.
  def qualify_cache!(path : String, & : String -> Bool) : Nil
    if File.info?(path, false)
      raise ArgumentError.new("refusing to replace qualified cache")
    end
    directory = File.tempname(".hy-nomas-cache-qualification-", dir: File.dirname(path))
    Dir.mkdir(directory, 0o700)
    staged = File.join(directory, "weights.safetensors")
    begin
      raise "BF16 cache qualification failed" unless yield staged
      info = File.info?(staged, false)
      unless info && info.file? && !info.symlink?
        raise ArgumentError.new("qualified cache staging must be a regular file")
      end
      File.link(staged, path)
    ensure
      File.delete(staged) if File.info?(staged, false)
      Dir.delete(directory)
    end
  end

  # Only for callers that already established successful terminal completion.
  # Read shared resident bytes in bounded chunks; never dequantize on CPU.
  def compare_weight_buffers!(plan : ML::Video::WeightPlan, weights : ML::Video::Arena,
                              oracle : ML::Video::CheckpointFile) : Array(WeightComparison)
    oracle.unchanged!
    chunk = Bytes.new(64 * 1024)
    results = [] of WeightComparison
    File.open(oracle.path, "rb") do |io|
      plan.entries.each do |entry|
        info = oracle.tensors[entry.binding.source]
        buffer = weights[entry.binding.target]
        pointer = buffer.contents.as(UInt8*)
        expected_hash = Digest::SHA256.new
        actual_hash = Digest::SHA256.new
        different = 0_i64
        offset = 0_i64
        io.seek(oracle.data_offset + info.data_start)
        while offset < info.data_bytes
          count = Math.min(chunk.size.to_i64, info.data_bytes - offset).to_i32
          expected = chunk[0, count]
          io.read_fully(expected)
          actual = Bytes.new(pointer + offset, count)
          expected_hash.update(expected)
          actual_hash.update(actual)
          (count // 2).times do |index|
            byte = index * 2
            different += 1 if expected[byte] != actual[byte] || expected[byte + 1] != actual[byte + 1]
          end
          offset += count
        end
        results << {name: entry.binding.source, words: info.n_elements,
                    different_words: different, native_sha256: actual_hash.hexfinal,
                    upstream_sha256: expected_hash.hexfinal}
      end
    end
    oracle.unchanged!
    results.sort_by(&.[:name])
  end

  class RunOwner
    @@roots = [] of RunOwner
    @@root_mutex = Mutex.new
    @reference : Serial::ReferenceFixture
    @oracle : ML::Video::CheckpointFile
    @budget : Int64
    @input : ML::Video::Arena? = nil
    @weights : ML::Video::Arena? = nil
    @scratch : ML::Video::Arena? = nil
    @file : ML::Video::CheckpointFile? = nil
    @plan : ML::Video::WeightPlan? = nil
    @prepared : ML::Video::PreparedWeights? = nil
    @block : ML::Video::HyNomasDualStreamBlock? = nil
    @workspace : ML::Video::HyNomasMPSWorkspace? = nil
    @chain : ML::Metal::MPSCommandChain? = nil
    @lease = HyNomasTextProbe::SessionChainLease(ML::Metal::MPSCommandChain).new
    @control = {} of String => Array(Float32)
    getter phases = [] of Phase
    getter cache_sha256 = ""
    getter released = false
    @rooted = false
    @block_index : Int32

    def initialize(@reference : Serial::ReferenceFixture,
                   @oracle : ML::Video::CheckpointFile, @budget : Int64,
                   @block_index : Int32 = 0)
      begin
        # No Metal allocation precedes the complete original/oracle admission.
        HyNomasBF16CacheProbe.block_pin(@block_index)
        @file = ML::Video::CheckpointFile.new(reference.block_path("block#{@block_index}"))
        plan = plan_for(@file.not_nil!, conversion: true)
        validate_original!(plan)
        HyNomasBF16CacheProbe.admit!(plan, @budget)
        HyNomasBF16CacheProbe.validate_oracle!(plan, @oracle, @block_index)
        oracle_plan = plan_for(@oracle, conversion: false)
        HyNomasBF16CacheProbe.admit!(oracle_plan, @budget)
        root!
        input = ML::Video::Arena.new
        @input = input
        Serial::INPUT_SHAPES.each_key do |name|
          source = if @block_index == 1 && {"input.img", "input.txt"}.includes?(name)
                     "block0.#{name.lchop("input.")}.output"
                   else
                     name
                   end
          values = @reference.values(source)
          raise ArgumentError.new("non-finite saved input: #{name}") unless values.all?(&.finite?)
          input.upload(name, values)
        end
        prepare_phase!(plan)
      rescue ex
        cleanup!
        raise ex
      end
    end

    def run!(cache_path : String) : Nil
      run_phase!("f32_source", publish: cache_path)
      release_phase!
      @file = ML::Video::BF16WeightCache.authenticate(cache_path, @cache_sha256)
      cache_plan = plan_for(@file.not_nil!, conversion: false)
      HyNomasBF16CacheProbe.validate_oracle!(cache_plan, @oracle, @block_index)
      raise "BF16 cache reload allocated conversion staging" unless cache_plan.staging_bytes == 0
      prepare_phase!(cache_plan)
      run_phase!("bf16_cache")
      run_phase!("bf16_cache_repeat")
    rescue ex
      cleanup!
      raise ex
    end

    def cleanup! : Nil
      return if @released
      if chain = @chain
        # Failed or incomplete drains never free borrowed resources. A completed
        # GPU error permits cleanup but remains an exception, not publication.
        Qwen25PrefixProbe.cleanup_after_drain(chain) do
          @lease.ensure_terminal!
          chain.release
          @chain = nil
          release_resources!
        end
      else
        release_resources!
      end
    end

    private def plan_for(file : ML::Video::CheckpointFile, conversion : Bool) : ML::Video::WeightPlan
      ML::Video::HyNomasDualStreamBlock.weight_plan([file], Serial::CONFIG,
        Serial::NAMED_BUDGET_LIMIT_BYTES, resident_dtype: ML::Video::WeightDType::BF16,
        block_index: @block_index, require_all: true, allow_f32_to_bf16: conversion)
    end

    private def validate_original!(plan : ML::Video::WeightPlan) : Nil
      unless plan.entries.size == 32 && plan.entries.all?(&.info.dtype.f32?) &&
             plan.destination_bytes == 302_113_792_i64 && plan.staging_bytes == 604_227_584_i64
        raise ArgumentError.new("original block-#{@block_index} source plan differs from the pinned F32 inventory")
      end
    end

    private def prepare_phase!(plan : ML::Video::WeightPlan) : Nil
      raise "previous block phase is still owned" if @weights || @scratch || @workspace || @chain
      expected = HyNomasBF16CacheProbe.admit!(plan, @budget)
      @plan = plan
      weights = ML::Video::Arena.new
      @weights = weights
      @prepared = ML::Video::PreparedWeights.new(plan, weights)
      scratch = ML::Video::Arena.new
      @scratch = scratch
      ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, Serial::CONFIG)
      block = ML::Video::HyNomasDualStreamBlock.new(Serial::CONFIG, weights, scratch)
      @block = block
      @workspace = ML::Video::HyNomasMPSWorkspace.new(block, input("img"), input("txt"),
        input("vec"), input("vec_txt"), input("rope_cos"), input("rope_sin"))
      unless block.checkpoints.keys.to_set == Serial::CAPTURE_NAMES.to_set
        raise "block checkpoint map differs from authenticated 40-capture inventory"
      end
      block.checkpoints.each do |name, buffer|
        unless buffer.size == @reference.tensor_bytes("block#{@block_index}.#{name}") &&
               @reference.tensor_shape("block#{@block_index}.#{name}") == Serial.expected_capture_shape(name)
          raise "block checkpoint shape/extent differs from upstream: #{name}"
        end
      end
      actual = weights.resident_bytes + plan.staging_bytes + scratch.resident_bytes +
               @workspace.not_nil!.resident_bytes + @input.not_nil!.resident_bytes
      raise "cache phase named-byte estimate differs from allocated plans" unless actual == expected
    end

    private def run_phase!(name : String, publish : String? = nil) : Nil
      raise "previous request still in flight" if @chain
      raise "cache owner lost its lifetime root" unless @rooted
      chain = ML::Metal::MPSCommandChain.new
      @chain = chain
      @lease.begin_encode!(chain)
      plan = @plan.not_nil!
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      reads = ML::MetalBuffer.hy_text_probe_readbacks
      @prepared.not_nil!.encode(chain.encoder)
      @block.not_nil!.encode_mps(chain, input("img"), input("txt"), input("vec"), input("vec_txt"),
        input("rope_cos"), input("rope_sin"), @workspace.not_nil!)
      encoded_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
      early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
      unless encoded_allocations == 0 && early_reads == 0 && chain.wait_count == 0
        raise "cache inference introduced an allocation, early activation readback or wait"
      end
      chain.commit_and_wait
      raise "cache request did not retain one terminal wait" unless chain.completed? && chain.wait_count == 1
      @lease.ensure_terminal!

      weight_results = compare_weights!(chain, plan)
      results = [] of Comparison
      @block.not_nil!.checkpoints.each do |capture, buffer|
        values = buffer.read((buffer.size // 4).to_i32)
        expected = @reference.values("block#{@block_index}.#{capture}")
        comparison = HyNomasTextProbe.compare_values(values, expected)
        control = @control[capture]?
        control_differences = if control
                                HyNomasTextProbe.compare_values(values, control)[:raw_bit_differences]
                              else
                                @control[capture] = values
                                0
                              end
        results << {name: capture, elements: values.size.to_i32,
                    failures: comparison[:failures], max_abs: comparison[:max_abs],
                    upstream_raw_differences: comparison[:raw_bit_differences],
                    control_raw_differences: control_differences}
      end
      final_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
      raise "cache probe diagnostic read count differs from 40" unless final_reads == 40
      @phases << {name: name, destination_bytes: plan.destination_bytes,
                  staging_bytes: plan.staging_bytes, named_bytes: HyNomasBF16CacheProbe.named_peak(plan),
                  encode_allocations: encoded_allocations, preterminal_activation_reads: early_reads,
                  terminal_waits: chain.wait_count, postterminal_activation_reads: final_reads,
                  conversion_dispatches: plan.entries.count { |entry| entry.info.dtype.f32? },
                  weights: weight_results, captures: results.sort_by(&.[:name])}
      unless weight_results.all? { |result| result[:different_words] == 0 } &&
             results.all? { |result| result[:failures] == 0 && result[:control_raw_differences] == 0 }
        raise "BF16 cache weight/intermediate gate failed in #{name}"
      end
      if path = publish
        @cache_sha256 = ML::Video::BF16WeightCache.write_after_terminal!(path, plan,
          @weights.not_nil!, chain, HyNomasBF16CacheProbe.block_pin(@block_index).file_sha256)
      end
      raise "post-terminal cache serialization added a native wait" unless chain.wait_count == 1
      chain.release
      @chain = nil
    end

    # Bounded raw-byte inspection after the successful terminal; no neural
    # calculation or Float32 weight materialization happens on CPU. This proves
    # all BF16 words, rather than sampling a few projection entries.
    private def compare_weights!(chain : ML::Metal::MPSCommandChain,
                                 plan : ML::Video::WeightPlan) : Array(WeightComparison)
      raise "weight diagnostics require the completed request" unless chain.completed? && chain.wait_count == 1
      HyNomasBF16CacheProbe.compare_weight_buffers!(plan, @weights.not_nil!, @oracle)
    end

    private def input(name : String) : ML::MetalBuffer
      @input.not_nil!["input.#{name}"]
    end

    private def release_phase! : Nil
      raise "cannot release cache resources before terminal" if @chain
      @lease.ensure_terminal!
      @workspace.try(&.release)
      @workspace = nil
      @block = nil
      @scratch.try(&.release)
      @scratch = nil
      @prepared.try(&.release)
      @prepared = nil
      @weights.try(&.release)
      @weights = nil
      @plan = nil
      @file.try(&.close)
      @file = nil
    end

    private def release_resources! : Nil
      release_phase!
      @input.try(&.release)
      @input = nil
      @oracle.close
      @released = true
      @@root_mutex.synchronize { @@roots.reject! { |owner| owner.same?(self) } }
      @rooted = false
    end

    private def root! : Nil
      @@root_mutex.synchronize { @@roots << self }
      @rooted = true
    end
  end

  def run(reference_path : String, reference_sha : String, conditioning_path : String,
          oracle_path : String, oracle_sha : String, cache_path : String,
          report_path : String, budget : Int64, block_index : Int32 = 0) : Bool
    pin = block_pin(block_index)
    {cache_path, report_path}.each do |path|
      raise ArgumentError.new("refusing to replace cache/report") if File.exists?(path) || File.symlink?(path)
      raise ArgumentError.new("cache/report parent does not exist") unless Dir.exists?(File.dirname(File.expand_path(path)))
    end
    raise ArgumentError.new("cache and report paths alias") if File.expand_path(cache_path) == File.expand_path(report_path)
    reference = Serial::ReferenceFixture.new(reference_path, reference_sha)
    conditioning_sha = Serial::PINNED_CONDITIONING_SHA
    qwen_sha = Serial.conditioning_qwen_sha256(conditioning_path, conditioning_sha)
    conditioning = HyNomasTextProbe::Fixture.new(conditioning_path, conditioning_sha, qwen_sha)
    reference.validate_conditioning_inputs!(conditioning)
    reference.authenticate_weight_artifacts!(conditioning)
    oracle = ML::Video::BF16WeightCache.authenticate(oracle_path, oracle_sha)
    owner = nil.as(RunOwner?)
    begin
      owner = RunOwner.new(reference, oracle, budget, block_index)
      active = owner
      qualify_cache!(cache_path) do |staged|
        active.run!(staged)
        active.cleanup!
        true
      end
      report = {
        schema: "cogni_ml.hy_nomas_bf16_cache_probe.v1", status: "pass",
        reference: {path: reference.path, sha256: reference.sha256,
                    scope: "saved block-#{block_index} upstream MPS captures; no fresh upstream block rerun"},
        oracle: {path: File.realpath(oracle_path), sha256: oracle_sha},
        cache: {path: File.realpath(cache_path), sha256: owner.cache_sha256,
                file_bytes: File.size(cache_path), payload_bytes: 302_113_792_i64,
                publication: "after source, reload and repeat gates and terminal owner cleanup"},
        source: {path: reference.block_path("block#{block_index}"), sha256: pin.file_sha256,
                 block_index: block_index, manifest_sha256: pin.manifest_sha256,
                 upstream_revision: Serial::PINNED_UPSTREAM_REVISION},
        tolerance: {atol: Serial::ATOL, rtol: Serial::RTOL},
        geometry: {batches: 1, video_tokens: 6, text_tokens: 23, dim: 2048,
                   heads: 16, head_dim: 128, ffn_dim: 8192},
        precision: {native: {parameter_storage: "BF16", compute: "F32", attention: "MPS F32"},
                    upstream_reference: reference.document["precision"]},
        named_byte_budget: budget, phases: owner.phases,
        owner_released_after_terminal: owner.released,
        scope: "one pretrained block at B1/V6/T23; native BF16 parameter storage, F32 math/MPS attention; setup byte I/O and post-terminal diagnostics only on CPU; no full generation or speed claim",
      }
      temporary = File.tempfile("hy-nomas-cache-report-", ".partial", dir: File.dirname(report_path))
      begin
        temporary << report.to_pretty_json << '\n'
        temporary.fsync
        temporary.close
        File.link(temporary.path, report_path)
      ensure
        temporary.close unless temporary.closed?
        File.delete(temporary.path) if File.exists?(temporary.path)
      end
      puts "PASS: block #{block_index} BF16 cache exact weights and upstream intermediate gate"
      true
    ensure
      if active = owner
        active.cleanup!
      else
        oracle.close
      end
    end
  end
end
