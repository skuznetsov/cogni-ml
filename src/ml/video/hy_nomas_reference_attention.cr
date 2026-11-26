# Fixed-binding first-T0 BF16 attention diagnostic, separate from the ordinary
# F32 compact graph. The physical stored cache is the only K/V binding.
require "./hy_nomas_reference_kv_cache"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib HyNomasReferenceAttentionFFI
    fun plan_create = gs_hy_nomas_reference_attention_plan_create(
      device : Pointer(Void), tokens : Int32, heads : Int32, head_dim : Int32,
      query : Pointer(Void), key : Pointer(Void), value : Pointer(Void),
      output : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_hy_nomas_reference_attention_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun plan_release = gs_hy_nomas_reference_attention_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video::HyNomasReferenceAttention
  # Q and output each own one physical BF16 active-prefix buffer. F32 source
  # Q and destination context borrow the block's existing storage-capacity
  # buffers. No padded row is read or written by this diagnostic.
  def self.named_bytes(storage_rows : Int32, active_rows : Int32,
                       heads : Int32, head_dim : Int32) : Int64
    geometry = HyNomasReferenceKVCache.geometry(storage_rows, active_rows, heads, head_dim)
    geometry.cache_elements.to_i64 * 4_i64
  end

  class Plan
    @@lifetime_mutex = Mutex.new
    @@live_plans = [] of Plan

    getter cache : HyNomasReferenceKVCache::Plan
    getter output : ML::MetalBuffer
    @query : ML::MetalBuffer
    @query_bf16 : ML::MetalBuffer? = nil
    @output_bf16 : ML::MetalBuffer? = nil
    @store_pipeline : ML::Metal::ComputePipeline
    @load_pipeline : ML::Metal::ComputePipeline
    @handle = Pointer(Void).null
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @poisoned = false
    @released = false

    def initialize(@query : ML::MetalBuffer, @cache : HyNomasReferenceKVCache::Plan,
                   @output : ML::MetalBuffer)
      @store_pipeline = ML::Metal::PipelineCache.get(HyNomasReferenceKVCache::STORE_KERNEL) do
        ML::Metal::ComputePipeline.new(HyNomasReferenceKVCache::STORE_KERNEL, HyNomasReferenceKVCache::SOURCE)
      end
      @load_pipeline = ML::Metal::PipelineCache.get(HyNomasReferenceKVCache::LOAD_KERNEL) do
        ML::Metal::ComputePipeline.new(HyNomasReferenceKVCache::LOAD_KERNEL, HyNomasReferenceKVCache::SOURCE)
      end
      begin
        @query_bf16 = ML::MetalBuffer.new(@cache.geometry.cache_elements.to_i64 * 2_i64, ML::StorageMode::Shared)
        @output_bf16 = ML::MetalBuffer.new(@cache.geometry.cache_elements.to_i64 * 2_i64, ML::StorageMode::Shared)
        validate_bindings!
        {% if flag?(:darwin) && !flag?(:cpu_only) %}
          @handle = HyNomasReferenceAttentionFFI.plan_create(
            ML::Metal::Device.instance.device_handle,
            @cache.active_rows, @cache.heads, @cache.head_dim,
            query_bf16.handle, @cache.key_storage.handle, @cache.value_storage.handle,
            output_bf16.handle)
          raise "Failed to prepare Hy-Nomas reference BF16 attention graph" if @handle.null?
          @@lifetime_mutex.synchronize { @@live_plans << self }
        {% else %}
          raise ArgumentError.new("Hy-Nomas reference BF16 attention requires macOS Metal Performance Shaders")
        {% end %}
      rescue ex
        release
        raise ex
      end
    end

    def query_bf16 : ML::MetalBuffer
      @query_bf16 || raise ArgumentError.new("Reference BF16 attention plan has been released")
    end

    def output_bf16 : ML::MetalBuffer
      @output_bf16 || raise ArgumentError.new("Reference BF16 attention plan has been released")
    end

    def resident_bytes : Int64
      @released ? 0_i64 : @cache.geometry.cache_elements.to_i64 * 4_i64
    end

    def validate_encode!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Reference BF16 attention plan has been released") if @released || @handle.null?
      raise ArgumentError.new("Reference BF16 attention plan is poisoned after an encode failure") if @poisoned
      validate_bindings!
      @cache.validate_read!(chain)
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Reference BF16 attention cannot span in-flight command chains")
        end
      end
    end

    # Prepared graph execution only: all tensor math stays on the caller's
    # serial GPU chain, with no allocation, wait or readback here.
    def encode(chain : ML::Metal::MPSCommandChain) : ML::MetalBuffer
      validate_encode!(chain)
      @cache.bind_consumer!(chain)
      @last_chain = chain
      @poisoned = true
      count = @cache.geometry.cache_elements
      chain.encoder.set_pipeline(@store_pipeline)
        .set_buffer(@query, 0)
        .set_buffer(query_bf16, 1, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 2)
        .dispatch_1d(count)
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = HyNomasReferenceAttentionFFI.encode(@handle, chain.handle)
        ensure
          chain.resume!
        end
        raise "Reference BF16 attention encode rejected (#{status})" unless status == 0
      {% end %}
      chain.encoder.set_pipeline(@load_pipeline)
        .set_buffer(output_bf16, 0)
        .set_buffer(@output, 1, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 2)
        .dispatch_1d(count)
      @poisoned = false
      @output
    end

    def validate_release! : Nil
      if chain = @last_chain
        raise ArgumentError.new("Drain reference BF16 attention chain before release") unless chain.completed?
      end
    end

    def release : Nil
      return if @released
      validate_release!
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        HyNomasReferenceAttentionFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @query_bf16.try(&.release)
      @output_bf16.try(&.release)
      @query_bf16 = nil
      @output_bf16 = nil
      @last_chain = nil
      @released = true
      @@lifetime_mutex.synchronize { @@live_plans.reject! { |current| current.same?(self) } }
    end

    def finalize
      return if @released || (@last_chain && !@last_chain.not_nil!.completed?)
      release
    end

    private def validate_bindings! : Nil
      buffers = {@query, @cache.key_storage, @cache.value_storage, query_bf16, output_bf16, @output}
      buffers.each do |buffer|
        unless buffer.valid? && buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("Reference BF16 attention requires valid shared buffers")
        end
      end
      full_bytes = @cache.geometry.source_elements.to_i64 * sizeof(Float32)
      raw_bytes = @cache.geometry.cache_elements.to_i64 * 2_i64
      unless @query.size == full_bytes && @output.size == full_bytes &&
             query_bf16.size == raw_bytes && output_bf16.size == raw_bytes &&
             @cache.key_storage.size == raw_bytes && @cache.value_storage.size == raw_bytes
        raise ArgumentError.new("Reference BF16 attention fixed buffer extent changed")
      end
      buffers.each_with_index do |left, index|
        buffers.each_with_index do |right, other_index|
          next if other_index <= index
          raise ArgumentError.new("Reference BF16 attention buffers must not alias") if MetalOps.overlap?(left, right)
        end
      end
    end
  end
end
