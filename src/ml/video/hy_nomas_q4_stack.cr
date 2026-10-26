# Authenticated, setup-only owner for the full sequence of per-block Q4_K
# caches. All cache files are admitted before this object allocates Metal.
require "digest/sha256"
require "json"
require "set"
require "./hy_nomas_q4_cache"
require "./hy_nomas_serial_mps"

module ML::Video::HyNomasQ4StackIndex
  extend self

  SCHEMA         = "hy-nomas-q4-dit-cache-index-v1"
  SCHEMA_VERSION = 1

  record BlockPin, block_index : Int32, manifest_path : String,
    manifest_sha256 : String, payload_sha256 : String,
    bf16_path : String, bf16_sha256 : String

  record Index, source_sha256 : String, source_bytes : Int64,
    globals_bf16_path : String, globals_bf16_sha256 : String,
    blocks : Array(BlockPin), caches : Array(HyNomasQ4Cache::Block),
    sha256 : String

  def load!(path : String, expected_index_sha256 : String,
            expected_source_sha256 : String, config : HyNomasConfig) : Index
    HyNomasQ4Cache::Integrity.validate_sha256!(expected_index_sha256, "index")
    HyNomasQ4Cache::Integrity.validate_sha256!(expected_source_sha256, "source")
    info = HyNomasQ4Cache::Integrity.regular_file_info!(path, "cache index")
    unless info.size > 0 && info.size <= HyNomasQ4Cache::MAX_MANIFEST_BYTES
      raise ArgumentError.new("Hy-Nomas Q4 index size is invalid")
    end
    bytes = File.read(path)
    sha256 = Digest::SHA256.hexdigest(bytes)
    unless sha256 == expected_index_sha256
      raise ArgumentError.new("Hy-Nomas Q4 index SHA-256 mismatch")
    end
    unless HyNomasQ4Cache::Integrity.same_snapshot?(info,
             HyNomasQ4Cache::Integrity.regular_file_info!(path, "cache index"))
      raise ArgumentError.new("Hy-Nomas Q4 index changed while reading")
    end

    document = JSON.parse(bytes).as_h
    HyNomasQ4Cache::Integrity.exact_keys!(document,
      Set{"schema", "schema_version", "source", "blocks", "globals"},
      "cache index")
    unless document["schema"].as_s == SCHEMA &&
           document["schema_version"].as_i == SCHEMA_VERSION
      raise ArgumentError.new("Hy-Nomas Q4 cache index schema mismatch")
    end

    source = document["source"].as_h
    HyNomasQ4Cache::Integrity.exact_keys!(source, Set{"sha256", "bytes"},
      "index source")
    source_sha256 = source["sha256"].as_s
    HyNomasQ4Cache::Integrity.validate_sha256!(source_sha256, "source")
    unless source_sha256 == expected_source_sha256
      raise ArgumentError.new("Hy-Nomas Q4 index source checkpoint mismatch")
    end
    source_bytes = source["bytes"].as_i64
    unless source_bytes > 0
      raise ArgumentError.new("Hy-Nomas Q4 index source byte size is invalid")
    end

    directory = File.dirname(path)
    index_info = info
    referenced_infos = [index_info] of File::Info
    manifest_names = Set(String).new
    payload_hashes = Set(String).new
    manifest_hashes = Set(String).new
    block_pins = [] of BlockPin
    block_objects = [] of HyNomasQ4Cache::Block
    entries = document["blocks"].as_a
    unless entries.size == HyNomasQ4Cache::DIT_BLOCK_COUNT
      raise ArgumentError.new("Hy-Nomas Q4 index must name all 54 blocks")
    end

    entries.each_with_index do |entry, expected_index|
      record = entry.as_h
      HyNomasQ4Cache::Integrity.exact_keys!(record,
        Set{"block_index", "manifest", "manifest_sha256", "payload_sha256",
            "bf16", "bf16_sha256"}, "block index entry")
      block_index = record["block_index"].as_i
      unless block_index == expected_index.to_i64
        raise ArgumentError.new("Hy-Nomas Q4 index block order must be exactly 0...54")
      end
      manifest_name = record["manifest"].as_s
      unless manifest_name.ends_with?(".json") &&
             HyNomasQ4Cache::Integrity.safe_basename?(manifest_name)
        raise ArgumentError.new("Hy-Nomas Q4 block manifest must be a safe sibling basename")
      end
      manifest_sha256 = record["manifest_sha256"].as_s
      payload_sha256 = record["payload_sha256"].as_s
      HyNomasQ4Cache::Integrity.validate_sha256!(manifest_sha256, "manifest")
      HyNomasQ4Cache::Integrity.validate_sha256!(payload_sha256, "payload")
      unless manifest_names.add?(manifest_name) &&
             manifest_hashes.add?(manifest_sha256) &&
             payload_hashes.add?(payload_sha256)
        raise ArgumentError.new("Hy-Nomas Q4 index contains duplicate block cache identities")
      end

      manifest_path = File.join(directory, manifest_name)
      manifest_info = HyNomasQ4Cache::Integrity.regular_file_info!(manifest_path,
        "block manifest")
      ensure_distinct_file!(manifest_info, referenced_infos, "block cache files")
      referenced_infos << manifest_info
      block = HyNomasQ4Cache::Block.new(manifest_path, manifest_sha256,
        expected_source_sha256, config, block_index.to_i32)
      unless block.source_bytes == source_bytes
        raise ArgumentError.new("Hy-Nomas Q4 block source byte size disagrees with index")
      end
      unless block.payload_sha256 == payload_sha256
        raise ArgumentError.new("Hy-Nomas Q4 index payload pin disagrees with block manifest")
      end
      payload_info = HyNomasQ4Cache::Integrity.regular_file_info!(block.payload_path,
        "block payload")
      ensure_distinct_file!(payload_info, referenced_infos, "block cache files")
      referenced_infos << payload_info
      block_objects << block

      bf16_name = record["bf16"].as_s
      unless bf16_name.ends_with?(".safetensors") &&
             HyNomasQ4Cache::Integrity.safe_basename?(bf16_name)
        raise ArgumentError.new("Hy-Nomas BF16 control must be a safe sibling basename")
      end
      bf16_sha256 = record["bf16_sha256"].as_s
      HyNomasQ4Cache::Integrity.validate_sha256!(bf16_sha256, "BF16 control")
      bf16_path = File.join(directory, bf16_name)
      bf16_info = HyNomasQ4Cache::Integrity.regular_file_info!(bf16_path,
        "BF16 control")
      ensure_distinct_file!(bf16_info, referenced_infos, "cache index files")
      referenced_infos << bf16_info
      block_pins << BlockPin.new(block_index.to_i32, manifest_path,
        manifest_sha256, payload_sha256, bf16_path, bf16_sha256)
    end

    globals = document["globals"].as_h
    HyNomasQ4Cache::Integrity.exact_keys!(globals, Set{"bf16", "sha256"},
      "global BF16 control")
    globals_name = globals["bf16"].as_s
    unless globals_name.ends_with?(".safetensors") &&
           HyNomasQ4Cache::Integrity.safe_basename?(globals_name)
      raise ArgumentError.new("Hy-Nomas global BF16 control must be a safe sibling basename")
    end
    globals_sha256 = globals["sha256"].as_s
    HyNomasQ4Cache::Integrity.validate_sha256!(globals_sha256, "global BF16 control")
    globals_path = File.join(directory, globals_name)
    globals_info = HyNomasQ4Cache::Integrity.regular_file_info!(globals_path,
      "global BF16 control")
    ensure_distinct_file!(globals_info, referenced_infos, "cache index files")

    Index.new(source_sha256, source_bytes, globals_path, globals_sha256,
      block_pins, block_objects, sha256)
  end

  private def ensure_distinct_file!(candidate : File::Info,
                                    prior : Array(File::Info), label : String) : Nil
    if prior.any? { |info| candidate.same_file?(info) }
      raise ArgumentError.new("Hy-Nomas Q4 #{label} must use distinct regular files")
    end
  end
end

module ML::Video
  # One setup-admitted resident Q4_K full-DiT cache. The serialized BF16 files
  # named by the index are independent controls and are deliberately loaded by
  # BF16WeightCache, never treated as Q4 inference weights here.
  class HyNomasQ4Stack < HyNomasSerialWeightLoader
    getter source_sha256 : String
    getter index_sha256 : String
    getter block_count : Int32 = 54
    getter cache_payload_bytes : Int64
    getter admitted_bytes : Int64
    getter budget_bytes : Int64
    getter stage_bytes : Int64
    getter reserve_bytes : Int64

    @blocks : Array(HyNomasQ4Cache::Block)
    @block_pins : Array(HyNomasQ4StackIndex::BlockPin)
    @stage_arena : Arena? = nil
    @stage_buffers = {} of String => ML::MetalBuffer
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @next_block = 0
    @loaded = false
    @released = false
    @poisoned = false

    def initialize(index_path : String, expected_index_sha256 : String,
                   expected_source_sha256 : String, config : HyNomasConfig,
                   @budget_bytes : Int64, *, @stage_bytes : Int64,
                   @reserve_bytes : Int64 = 0_i64)
      unless @budget_bytes > 0 && @stage_bytes > 0 && @reserve_bytes >= 0
        raise ArgumentError.new("Hy-Nomas Q4 stack memory budget is invalid")
      end
      index = HyNomasQ4StackIndex.load!(index_path, expected_index_sha256,
        expected_source_sha256, config)
      @source_sha256 = index.source_sha256
      @index_sha256 = index.sha256
      @block_pins = index.blocks
      @blocks = index.caches
      unless @blocks.size == HyNomasQ4Cache::DIT_BLOCK_COUNT &&
             @blocks.each_with_index.all? { |block, index| block.block_index == index }
        raise ArgumentError.new("Hy-Nomas Q4 stack must contain ordered blocks 0...54")
      end
      @cache_payload_bytes = 0_i64
      @blocks.each do |block|
        if block.payload_bytes > Int64::MAX - @cache_payload_bytes
          raise ArgumentError.new("Hy-Nomas Q4 cache byte count overflow")
        end
        @cache_payload_bytes += block.payload_bytes
      end
      expected_stage_bytes = @blocks.first.tensors.sum do |tensor|
        tensor.elements * WeightDType::BF16.bytes_per_element
      end
      unless @stage_bytes == expected_stage_bytes
        raise ArgumentError.new("Hy-Nomas Q4 stage budget must equal the exact 32-tensor BF16 inventory")
      end
      unless @cache_payload_bytes <= @budget_bytes &&
             @stage_bytes <= @budget_bytes - @cache_payload_bytes &&
             @reserve_bytes <= @budget_bytes - @cache_payload_bytes - @stage_bytes
        raise ArgumentError.new("Hy-Nomas Q4 cache, shared stage, and reserve exceed the memory budget")
      end
      @admitted_bytes = @cache_payload_bytes + @stage_bytes + @reserve_bytes
    end

    # Return authenticated metadata only; callers cannot release or otherwise
    # mutate the cache objects owned by this stack.
    def blocks : Array(HyNomasQ4StackIndex::BlockPin)
      @block_pins.dup
    end

    # Loading is a setup-only operation after every manifest, payload, index,
    # source, and aggregate-byte check has succeeded. Failure releases only
    # buffers created by this stack and poisons it against unsafe retry.
    def load!(arena : Arena) : Nil
      raise ArgumentError.new("Hy-Nomas Q4 stack has been released") if @released
      raise ArgumentError.new("Hy-Nomas Q4 stack is already loaded") if @loaded
      validate_stage_arena!(arena)
      unless arena.resident_bytes == @stage_bytes
        raise ArgumentError.new("Hy-Nomas BF16 staging Arena does not match its admitted byte size")
      end
      loaded = [] of HyNomasQ4Cache::Block
      begin
        @blocks.each do |block|
          block.load!
          loaded << block
        end
        HyNomasQ4Cache.prepare_decode!
        payloads = [] of ML::MetalBuffer
        @blocks.each do |block|
          block.validate_destinations!(arena)
          # The payload accessor remains private; verify cross-block aliasing
          # through the resident MetalBuffer API added for this owner.
          payload = block.resident_payload
          payloads.each { |prior| MetalOps.distinct(payload, prior) }
          @stage_buffers.each_value { |stage| MetalOps.distinct(payload, stage) }
          payloads << payload
        end
        @stage_arena = arena
        @loaded = true
      rescue ex
        loaded.each(&.release)
        @poisoned = true
        raise ex
      end
    end

    # Called for every registered block during the owner's full CPU preflight.
    # It allocates nothing and appends no native work.
    def validate_for!(block_index : Int32, arena : Arena) : Nil
      raise ArgumentError.new("Hy-Nomas Q4 stack has been released") if @released
      raise ArgumentError.new("Hy-Nomas Q4 stack is not loaded") unless @loaded
      raise ArgumentError.new("Hy-Nomas Q4 stack is poisoned") if @poisoned
      unless (0...HyNomasQ4Cache::DIT_BLOCK_COUNT).includes?(block_index) &&
             @blocks[block_index].block_index == block_index
        raise ArgumentError.new("Hy-Nomas Q4 block index is outside its authenticated order")
      end
      owner = @stage_arena || raise ArgumentError.new("Hy-Nomas Q4 staging Arena is unavailable")
      unless owner.same?(arena)
        raise ArgumentError.new("Hy-Nomas Q4 serial weights changed staging Arena identity")
      end
      validate_stage_bindings!(arena)
      @blocks[block_index].validate_destinations!(arena)
    end

    # Append Q4 decode and BF16 vector copies to the existing serial encoder.
    # Order and chain checks make it impossible to skip/reorder cache blocks.
    def encode!(chain : ML::Metal::MPSCommandChain,
                block_index : Int32, arena : Arena) : Nil
      validate_for!(block_index, arena)
      chain.encoder
      if block_index == 0
        if @next_block == HyNomasQ4Cache::DIT_BLOCK_COUNT
          previous = @last_chain || raise ArgumentError.new("Hy-Nomas Q4 previous chain is missing")
          unless previous.completed? && previous.wait_count == 1 && !previous.same?(chain)
            raise ArgumentError.new("Hy-Nomas Q4 stack may restart only on a fresh chain after terminal completion")
          end
          @next_block = 0
        elsif @next_block != 0
          raise ArgumentError.new("Hy-Nomas Q4 serial block order was interrupted")
        end
        @last_chain = chain
      else
        same_chain = if previous = @last_chain
                       previous.same?(chain)
                     else
                       false
                     end
        unless block_index == @next_block && same_chain
          raise ArgumentError.new("Hy-Nomas Q4 blocks must decode in order on one command chain")
        end
      end
      begin
        @blocks[block_index].decode_into(chain.encoder, arena)
        @next_block += 1
      rescue ex
        @poisoned = true
        raise ex
      end
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas Q4 command chain before release") unless chain.completed?
      end
      @blocks.each(&.release)
      @stage_buffers.clear
      @stage_arena = nil
      @released = true
    end

    private def validate_stage_arena!(arena : Arena) : Nil
      first = @blocks.first
      expected = first.tensors.map(&.target).to_set
      unless arena.buffers.keys.to_set == expected
        raise ArgumentError.new("Hy-Nomas Q4 staging Arena tensor inventory mismatch")
      end
      prior = [] of ML::MetalBuffer
      first.tensors.each do |tensor|
        buffer = arena.buffers[tensor.target]?
        unless buffer && arena.dtype(tensor.target).bf16? && buffer.valid? &&
               buffer.storage_mode.shared? &&
               buffer.size == tensor.elements * WeightDType::BF16.bytes_per_element
          raise ArgumentError.new("Hy-Nomas Q4 staging Arena binding is invalid #{tensor.target}")
        end
        prior.each { |other| MetalOps.distinct(buffer, other) }
        prior << buffer
      end
      @stage_buffers = arena.buffers.dup
    end

    private def validate_stage_bindings!(arena : Arena) : Nil
      unless arena.buffers.keys.to_set == @stage_buffers.keys.to_set
        raise ArgumentError.new("Hy-Nomas Q4 staging Arena map changed")
      end
      @stage_buffers.each do |name, expected|
        current = arena.buffers[name]?
        unless current && expected.same?(current) && current.valid? && arena.dtype(name).bf16?
          raise ArgumentError.new("Hy-Nomas Q4 staging Arena buffer identity changed #{name}")
        end
      end
    end
  end
end
