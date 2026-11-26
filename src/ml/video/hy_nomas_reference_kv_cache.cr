# Physical BF16 K/V storage for the bounded Hy-Nomas reference roundtrip.
# This is a one-write cache with repeatable F32 staging reads; it does not
# admit appends or multiple logical cache slots. A separate BF16 attention
# consumer may borrow the stored pair through a guarded command-chain lease.
require "../core/buffer"
require "../metal/dispatch"
require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video::HyNomasReferenceKVCache
  STORE_KERNEL = "hy_nomas_reference_kv_cache_store_bf16"
  LOAD_KERNEL  = "hy_nomas_reference_kv_cache_load_f32"
  SOURCE       = {{ read_file("#{__DIR__}/../metal/kernels/hy_nomas/hy_nomas_block.metal") }} + "\n" +
                 {{ read_file("#{__DIR__}/../metal/kernels/hy_nomas/hy_nomas_reference_kv_cache.metal") }}

  # Element counts describe one K or V plane. named_bytes covers the two
  # physical BF16 planes and their two exact-size F32 read stages.
  record Geometry, storage_rows : Int32, active_rows : Int32,
    heads : Int32, head_dim : Int32, source_elements : Int32,
    cache_elements : Int32, named_bytes : Int64 do
    def shape : Tuple(Int32, Int32, Int32)
      {active_rows, heads, head_dim}
    end
  end

  def self.geometry(storage_rows : Int32, active_rows : Int32,
                    heads : Int32, head_dim : Int32) : Geometry
    unless storage_rows > 0 && active_rows > 0 && heads > 0 && head_dim > 0
      raise ArgumentError.new("Hy-Nomas reference K/V dimensions must be positive")
    end
    unless active_rows <= storage_rows
      raise ArgumentError.new("Hy-Nomas reference K/V active rows exceed source storage rows")
    end

    source_elements = checked_elements(storage_rows, heads, head_dim)
    cache_elements = checked_elements(active_rows, heads, head_dim)
    # K/V BF16 cache (4 bytes per pair element) plus separate K/V F32 read
    # stages (8 bytes per pair element).
    named_bytes = cache_elements.to_i64 * 12_i64
    Geometry.new(storage_rows, active_rows, heads, head_dim,
      source_elements, cache_elements, named_bytes)
  end

  def self.named_bytes(storage_rows : Int32, active_rows : Int32,
                       heads : Int32, head_dim : Int32) : Int64
    geometry(storage_rows, active_rows, heads, head_dim).named_bytes
  end

  private def self.checked_elements(rows : Int32, heads : Int32,
                                    head_dim : Int32) : Int32
    product = 1_i64
    {rows, heads, head_dim}.each do |dimension|
      # Check before multiplying so even three maximal Int32 inputs cannot
      # overflow Int64 while constructing the pre-allocation geometry.
      if product > Int32::MAX.to_i64 // dimension
        raise ArgumentError.new("Hy-Nomas reference K/V element count exceeds the Metal dispatch limit")
      end
      product *= dimension
    end
    product.to_i32
  end

  class Plan
    getter geometry : Geometry

    @key_source : ML::MetalBuffer
    @value_source : ML::MetalBuffer
    @key_storage : ML::MetalBuffer? = nil
    @value_storage : ML::MetalBuffer? = nil
    @key_read : ML::MetalBuffer? = nil
    @value_read : ML::MetalBuffer? = nil
    @store_pipeline : ML::Metal::ComputePipeline
    @load_pipeline : ML::Metal::ComputePipeline
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @stored = false
    @poisoned = false
    @released = false

    def initialize(key : ML::MetalBuffer, value : ML::MetalBuffer, *,
                   storage_rows : Int32, active_rows : Int32,
                   heads : Int32, head_dim : Int32)
      # Keep pure geometry first: invalid shapes fail before touching Metal or
      # allocating a cache/staging buffer.
      @geometry = HyNomasReferenceKVCache.geometry(storage_rows, active_rows,
        heads, head_dim)
      validate_source!(key, "key")
      validate_source!(value, "value")
      reject_alias!(key, value)
      @key_source = key
      @value_source = value

      # Pipeline construction is setup-only. Encode methods below only bind
      # fixed buffers and dispatch on the caller-owned serial chain.
      @store_pipeline = ML::Metal::PipelineCache.get(STORE_KERNEL) do
        ML::Metal::ComputePipeline.new(STORE_KERNEL, SOURCE)
      end
      @load_pipeline = ML::Metal::PipelineCache.get(LOAD_KERNEL) do
        ML::Metal::ComputePipeline.new(LOAD_KERNEL, SOURCE)
      end

      created = [] of ML::MetalBuffer
      begin
        key_storage = ML::MetalBuffer.new(@geometry.cache_elements.to_i64 * 2_i64,
          ML::StorageMode::Shared)
        created << key_storage
        value_storage = ML::MetalBuffer.new(@geometry.cache_elements.to_i64 * 2_i64,
          ML::StorageMode::Shared)
        created << value_storage
        key_read = ML::MetalBuffer.new(@geometry.cache_elements.to_i64 * sizeof(Float32),
          ML::StorageMode::Shared)
        created << key_read
        value_read = ML::MetalBuffer.new(@geometry.cache_elements.to_i64 * sizeof(Float32),
          ML::StorageMode::Shared)
        created << value_read

        @key_storage = key_storage
        @value_storage = value_storage
        @key_read = key_read
        @value_read = value_read
        validate_bindings!
      rescue ex
        created.each(&.release)
        @key_storage = nil
        @value_storage = nil
        @key_read = nil
        @value_read = nil
        raise ex
      end
    end

    def storage_rows : Int32
      @geometry.storage_rows
    end

    def active_rows : Int32
      @geometry.active_rows
    end

    def heads : Int32
      @geometry.heads
    end

    def head_dim : Int32
      @geometry.head_dim
    end

    # Cache/read tensor shape, in the same flat row-major order as
    # [1, active_rows, heads, head_dim].
    def shape : Tuple(Int32, Int32, Int32)
      @geometry.shape
    end

    def key_storage : ML::MetalBuffer
      @key_storage || raise ArgumentError.new("Hy-Nomas reference K/V plan has been released")
    end

    def value_storage : ML::MetalBuffer
      @value_storage || raise ArgumentError.new("Hy-Nomas reference K/V plan has been released")
    end

    def key_read : ML::MetalBuffer
      @key_read || raise ArgumentError.new("Hy-Nomas reference K/V plan has been released")
    end

    def value_read : ML::MetalBuffer
      @value_read || raise ArgumentError.new("Hy-Nomas reference K/V plan has been released")
    end

    def resident_bytes : Int64
      @released ? 0_i64 : @geometry.named_bytes
    end

    def validate_store!(chain : ML::Metal::MPSCommandChain) : Nil
      validate_live!
      raise ArgumentError.new("Hy-Nomas reference K/V cache store is single-write") if @stored
      raise ArgumentError.new("Hy-Nomas reference K/V plan is poisoned after an encode failure") if @poisoned
      validate_chain!(chain)
      validate_bindings!
    end

    # Write exactly the active prefix from each fixed F32 source into physical
    # BF16 row-major storage. No allocation, wait, readback, or source-tail
    # access occurs here.
    def store(chain : ML::Metal::MPSCommandChain) : Nil
      validate_store!(chain)
      @last_chain = chain
      @poisoned = true
      begin
        e = chain.encoder
        encode_store!(e, @key_source, key_storage)
        encode_store!(e, @value_source, value_storage)
        @stored = true
        @poisoned = false
      rescue ex
        raise ex
      end
    end

    # Widen the stored physical BF16 pair into separate F32 read stages. Reads
    # may repeat on the same live chain, or resume on another chain after the
    # previous chain has terminally drained.
    def read(chain : ML::Metal::MPSCommandChain) : Nil
      validate_read!(chain)
      @last_chain = chain
      @poisoned = true
      begin
        e = chain.encoder
        encode_load!(e, key_storage, key_read)
        encode_load!(e, value_storage, value_read)
        @poisoned = false
      rescue ex
        raise ex
      end
    end

    def validate_read!(chain : ML::Metal::MPSCommandChain) : Nil
      validate_live!
      raise ArgumentError.new("Hy-Nomas reference K/V cache must be stored before read") unless @stored
      raise ArgumentError.new("Hy-Nomas reference K/V plan is poisoned after an encode failure") if @poisoned
      validate_chain!(chain)
      validate_bindings!
      validate_chain_successor!(chain)
    end

    # The consumer reads physical BF16 storage directly. Advancing this lease
    # prevents release of the owned cache while a later attention chain runs.
    def bind_consumer!(chain : ML::Metal::MPSCommandChain) : Nil
      validate_read!(chain)
      @last_chain = chain
    end

    # This plan owns all four output buffers. The caller (normally its
    # workspace) must keep the plan rooted until the final chain drains.
    def release : Nil
      return if @released
      validate_release!
      @key_storage.try(&.release)
      @value_storage.try(&.release)
      @key_read.try(&.release)
      @value_read.try(&.release)
      @key_storage = nil
      @value_storage = nil
      @key_read = nil
      @value_read = nil
      @last_chain = nil
      @released = true
    end

    def validate_release! : Nil
      if chain = @last_chain
        raise ArgumentError.new("Drain Hy-Nomas reference K/V command chain before release") unless chain.completed?
      end
    end

    private def encode_store!(e : ML::Metal::ComputeEncoder,
                              source : ML::MetalBuffer,
                              destination : ML::MetalBuffer) : Nil
      e.set_pipeline(@store_pipeline)
        .set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(@geometry.cache_elements.to_u32, 2)
        .dispatch_1d(@geometry.cache_elements)
    end

    private def encode_load!(e : ML::Metal::ComputeEncoder,
                             source : ML::MetalBuffer,
                             destination : ML::MetalBuffer) : Nil
      e.set_pipeline(@load_pipeline)
        .set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(@geometry.cache_elements.to_u32, 2)
        .dispatch_1d(@geometry.cache_elements)
    end

    private def validate_live! : Nil
      raise ArgumentError.new("Hy-Nomas reference K/V plan has been released") if @released
    end

    private def validate_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      chain.encoder
    end

    private def validate_chain_successor!(chain : ML::Metal::MPSCommandChain) : Nil
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas reference K/V plan cannot span in-flight command chains")
        end
      end
    end

    private def validate_bindings! : Nil
      validate_source!(@key_source, "key")
      validate_source!(@value_source, "value")
      key_storage = self.key_storage
      value_storage = self.value_storage
      key_read = self.key_read
      value_read = self.value_read
      unless key_storage.valid? && value_storage.valid? && key_read.valid? && value_read.valid? &&
             key_storage.storage_mode == ML::StorageMode::Shared &&
             value_storage.storage_mode == ML::StorageMode::Shared &&
             key_read.storage_mode == ML::StorageMode::Shared &&
             value_read.storage_mode == ML::StorageMode::Shared &&
             key_storage.size == @geometry.cache_elements.to_i64 * 2_i64 &&
             value_storage.size == @geometry.cache_elements.to_i64 * 2_i64 &&
             key_read.size == @geometry.cache_elements.to_i64 * sizeof(Float32) &&
             value_read.size == @geometry.cache_elements.to_i64 * sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas reference K/V cache buffer extent or storage mode changed")
      end
      MetalOps.check_storage(key_storage, @geometry.cache_elements.to_i64, WeightDType::BF16)
      MetalOps.check_storage(value_storage, @geometry.cache_elements.to_i64, WeightDType::BF16)
      MetalOps.check(key_read, @geometry.cache_elements.to_i64)
      MetalOps.check(value_read, @geometry.cache_elements.to_i64)

      reject_alias!(@key_source, @value_source)
      reject_alias!(@key_source, key_storage)
      reject_alias!(@key_source, value_storage)
      reject_alias!(@key_source, key_read)
      reject_alias!(@key_source, value_read)
      reject_alias!(@value_source, key_storage)
      reject_alias!(@value_source, value_storage)
      reject_alias!(@value_source, key_read)
      reject_alias!(@value_source, value_read)
      reject_alias!(key_storage, value_storage)
      reject_alias!(key_storage, key_read)
      reject_alias!(key_storage, value_read)
      reject_alias!(value_storage, key_read)
      reject_alias!(value_storage, value_read)
      reject_alias!(key_read, value_read)
    end

    private def validate_source!(buffer : ML::MetalBuffer, label : String) : Nil
      expected_bytes = @geometry.source_elements.to_i64 * sizeof(Float32)
      unless buffer.valid? && buffer.storage_mode == ML::StorageMode::Shared &&
             buffer.size == expected_bytes
        raise ArgumentError.new("Hy-Nomas reference K/V #{label} source must be exact-size shared F32")
      end
      MetalOps.check(buffer, @geometry.source_elements.to_i64)
    end

    private def reject_alias!(left : ML::MetalBuffer, right : ML::MetalBuffer) : Nil
      if MetalOps.overlap?(left, right)
        raise ArgumentError.new("Hy-Nomas reference K/V buffers must not alias")
      end
    end
  end
end
