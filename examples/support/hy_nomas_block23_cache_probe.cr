require "./hy_nomas_bf16_cache_probe"
require "./hy_nomas_full_j_probe"

# Setup-only conversion qualification. Neural parity is a separate four-block
# gate; no inference graph gains a wait from this cache creation tool.
module HyNomasBlock23CacheProbe
  extend self
  alias Cache = HyNomasBF16CacheProbe
  alias Serial = HyNomasPretrainedSerialProbe
  CONFIG            = Serial::CONFIG
  DESTINATION_BYTES = 302_113_792_i64
  SOURCE_BYTES      = 604_227_584_i64
  alias PhaseResult = NamedTuple(source_f32: Bool, destination_bytes: Int64,
    source_staging_bytes: Int64, source_conversion_dispatches: Int32,
    allocations_during_encode: Int64, readbacks_before_terminal_wait: Int64,
    terminal_wait_count: Int32, owner_rooted_through_terminal: Bool,
    weights: Array(Cache::WeightComparison), cache_sha256: String?)

  def validate_plan!(plan : ML::Video::WeightPlan, index : Int32, source_f32 : Bool) : Nil
    unless {2, 3}.includes?(index) && plan.entries.size == 32 &&
           plan.entries.all? { |entry| entry.binding.source.starts_with?("double_blocks.#{index}.") &&
           (source_f32 ? entry.info.dtype.f32? : entry.info.dtype.bf16?) } &&
           plan.resident_dtype.bf16? && plan.destination_bytes == DESTINATION_BYTES &&
           plan.staging_bytes == (source_f32 ? SOURCE_BYTES : 0_i64) &&
           plan.allow_f32_to_bf16 == source_f32
      raise ArgumentError.new("block-2/3 cache requires its complete exact-index source plan")
    end
  end

  def validate_oracle!(plan : ML::Video::WeightPlan, oracle : ML::Video::CheckpointFile) : Nil
    entries = plan.entries
    expected = entries.map(&.binding.source).to_set
    tensors = oracle.tensors
    unless tensors.keys.to_set == expected && entries.all? do |entry|
             info = tensors[entry.binding.source]
             info.dtype.bf16? && info.shape == entry.binding.shape && info.n_elements == entry.info.n_elements
           end
      raise ArgumentError.new("block-2/3 independent BF16 oracle inventory mismatch")
    end
  end

  def authenticate_source!(path : String, sha : String) : ML::Video::CheckpointFile
    unless sha.matches?(/\A[0-9a-f]{64}\z/) && File.file?(path) && !File.symlink?(path)
      raise ArgumentError.new("source must be a regular file with a SHA256 pin")
    end
    before = File.info(path)
    raise ArgumentError.new("source SHA256 mismatch") unless Serial.sha256_file(path) == sha
    after = File.info(path)
    unless before.same_file?(after) && before.size == after.size && before.modification_time == after.modification_time
      raise ArgumentError.new("source changed while authenticating")
    end
    ML::Video::CheckpointFile.new(path)
  end

  class Owner
    @@roots = [] of Owner
    @file : ML::Video::CheckpointFile? = nil
    @oracle : ML::Video::CheckpointFile? = nil
    @arena : ML::Video::Arena? = nil
    @prepared : ML::Video::PreparedWeights? = nil
    @chain : ML::Metal::MPSCommandChain? = nil
    @lease = HyNomasTextProbe::SessionChainLease(ML::Metal::MPSCommandChain).new
    @released = false

    def initialize(source : String, source_sha : String, oracle : String, oracle_sha : String,
                   @index : Int32, @budget : Int64)
      unless {2, 3}.includes?(@index) && @budget >= SOURCE_BYTES + DESTINATION_BYTES && @budget <= 2_i64 * 1024**3
        raise ArgumentError.new("cache conversion index/bounded aggregate budget mismatch")
      end
      begin
        @file = HyNomasBlock23CacheProbe.authenticate_source!(source, source_sha)
        @oracle = ML::Video::BF16WeightCache.authenticate(oracle, oracle_sha)
        if source_sha == oracle_sha || File.info(source).same_file?(File.info(oracle))
          raise ArgumentError.new("independent oracle must not alias the source")
        end
        plan = make_plan(@file.not_nil!, true)
        HyNomasBlock23CacheProbe.validate_oracle!(plan, @oracle.not_nil!)
        @@roots << self # Before first Metal allocation, including partial setup.
      rescue ex
        cleanup!
        raise ex
      end
    end

    private def make_plan(file : ML::Video::CheckpointFile, source_f32 : Bool)
      plan = ML::Video::HyNomasDualStreamBlock.weight_plan([file], CONFIG, @budget,
        resident_dtype: ML::Video::WeightDType::BF16, block_index: @index,
        require_all: true, allow_f32_to_bf16: source_f32)
      HyNomasBlock23CacheProbe.validate_plan!(plan, @index, source_f32)
      plan
    end

    def phase(source_f32 : Bool, publish : String? = nil, source_sha : String? = nil) : PhaseResult
      plan = make_plan(@file.not_nil!, source_f32)
      arena = ML::Video::Arena.new
      @arena = arena
      prepared = ML::Video::PreparedWeights.new(plan, arena)
      @prepared = prepared
      chain = ML::Metal::MPSCommandChain.new
      @chain = chain
      @lease.begin_encode!(chain)
      allocs = ML::MetalBuffer.hy_text_probe_allocations
      reads = ML::MetalBuffer.hy_text_probe_readbacks
      prepared.encode(chain.encoder)
      encoded_allocs = ML::MetalBuffer.hy_text_probe_allocations - allocs
      early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
      raise "cache conversion appended an allocation/readback/wait" unless encoded_allocs == 0 && early_reads == 0 && chain.wait_count == 0
      chain.commit_and_wait
      @lease.ensure_terminal!
      raise "cache conversion needs one completed terminal" unless chain.completed? && chain.wait_count == 1
      rows = Cache.compare_weight_buffers!(plan, arena, @oracle.not_nil!)
      raise "cache BF16 word comparison failed" unless rows.size == 32 && rows.all? { |row| row[:different_words] == 0 }
      sha = if path = publish
              ML::Video::BF16WeightCache.write_after_terminal!(path, plan, arena, chain, source_sha.not_nil!)
            else
              nil
            end
      raise "cache serialization added a terminal wait" unless chain.wait_count == 1
      result = {source_f32: source_f32, destination_bytes: plan.destination_bytes,
                source_staging_bytes: plan.staging_bytes,
                source_conversion_dispatches: source_f32 ? 32 : 0,
                allocations_during_encode: encoded_allocs, readbacks_before_terminal_wait: early_reads,
                terminal_wait_count: chain.wait_count, owner_rooted_through_terminal: @@roots.any?(&.same?(self)),
                weights: rows, cache_sha256: sha}
      chain.release
      @chain = nil
      release_phase!
      result
    rescue ex
      drain!
      raise ex
    end

    def reload!(path : String, sha : String) : Nil
      raise "cannot reload cache with live GPU resources" if @chain || @arena || @prepared
      replacement = ML::Video::BF16WeightCache.authenticate(path, sha)
      begin
        plan = make_plan(replacement, false)
        HyNomasBlock23CacheProbe.validate_oracle!(plan, @oracle.not_nil!)
      rescue ex
        replacement.close
        raise ex
      end
      @file.try(&.close)
      @file = replacement
    end

    def drain! : Nil
      if chain = @chain
        Qwen25PrefixProbe.cleanup_after_drain(chain) do
          @lease.ensure_terminal!
          chain.release
          @chain = nil
        end
      end
    end

    private def release_phase! : Nil
      raise "retain cache owner until terminal completion" if @chain
      @prepared.try(&.release)
      @prepared = nil
      @arena.try(&.release)
      @arena = nil
    end

    def cleanup! : Nil
      return if @released
      release_phase!
      @file.try(&.close)
      @oracle.try(&.close)
      @released = true
      @@roots.reject!(&.same?(self))
    end
  end

  def run(source : String, source_sha : String, oracle : String, oracle_sha : String,
          index : Int32, cache : String, report : String, budget : Int64) : Bool
    {cache, report}.each do |path|
      raise ArgumentError.new("cache/report must be fresh paths with existing parents") if File.info?(path, false) || !Dir.exists?(File.dirname(File.expand_path(path)))
    end
    raise ArgumentError.new("cache and report paths must differ") if File.expand_path(cache) == File.expand_path(report)
    owner = nil.as(Owner?)
    begin
      owner = Owner.new(source, source_sha, oracle, oracle_sha, index, budget)
      phases = [] of PhaseResult
      cache_sha = ""
      Cache.qualify_cache!(cache) do |staged|
        phases << owner.phase(true, staged, source_sha)
        cache_sha = phases.last[:cache_sha256].not_nil!
        owner.reload!(staged, cache_sha)
        phases << owner.phase(false)
        true
      end
      owner.cleanup!
      result = {schema: "cogni_ml.hy_nomas_block23_cache_probe", schema_version: 1,
                passed: true, block_index: index, source: {path: source, sha256: source_sha},
                oracle: {path: oracle, sha256: oracle_sha}, cache: {path: cache, sha256: cache_sha},
                phases: phases,
                scope: "setup-only GPU F32-to-BF16 conversion and direct BF16 reload; exact weight bytes; neural four-block gate is separate"}
      HyNomasFullJProbe.write_report_new!(report, result.to_pretty_json)
      puts({passed: true, block_index: index, cache_sha256: cache_sha}.to_json)
      true
    ensure
      if current = owner
        current.drain!
        current.cleanup!
      end
    end
  end
end
