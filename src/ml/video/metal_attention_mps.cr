require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video::MetalAttentionMPS
  # Small in-place input perturbation used only by the changed-input reuse
  # discriminator. Compile this pipeline during setup, before encode accounting.
  Q_NEGATE_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    kernel void wan_attention_mps_negate_q(
        device float *q [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
      if (tid < count) q[tid] = -q[tid];
    }
    METAL

  Q_TAIL_ZERO_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    kernel void wan_attention_mps_zero_q_tail(
        device float *q [[buffer(0)]],
        constant uint &offset [[buffer(1)]],
        constant uint &count [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
      if (tid < count) q[offset + tid] = 0.0f;
    }
    METAL

  # Read BTHD and write BHTD in one caller-chain dispatch. The output index is
  # linearized in BHTD order so writes remain contiguous; each thread copies
  # the matching F32 K and V element.
  HEAD_MAJOR_KV_TRANSPOSE_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    kernel void wan_attention_mps_transpose_head_major_kv(
        const device float *k [[buffer(0)]],
        device float *head_major_k [[buffer(1)]],
        const device float *v [[buffer(2)]],
        device float *head_major_v [[buffer(3)]],
        constant uint &batches [[buffer(4)]],
        constant uint &key_rows [[buffer(5)]],
        constant uint &heads [[buffer(6)]],
        constant uint &head_dim [[buffer(7)]],
        constant uint &count [[buffer(8)]],
        uint tid [[thread_position_in_grid]]) {
      if (tid >= count || tid / (head_dim * key_rows * heads) >= batches) return;
      const uint d = tid % head_dim;
      const uint key = (tid / head_dim) % key_rows;
      const uint head = (tid / (head_dim * key_rows)) % heads;
      const uint batch = tid / (head_dim * key_rows * heads);
      const uint source = ((batch * key_rows + key) * heads + head) * head_dim + d;
      head_major_k[tid] = k[source];
      head_major_v[tid] = v[source];
    }
    METAL

  def self.prepare! : Nil
    q_negate_pipeline
  end

  def self.prepare_chunk! : Nil
    q_tail_zero_pipeline
  end

  # Compile the fused K/V transpose during setup, before any command-chain
  # accounting begins.
  def self.prepare_head_major_kv! : Nil
    head_major_kv_transpose_pipeline
  end

  def self.negate_q!(chain : ML::Metal::MPSCommandChain,
                     q : ML::MetalBuffer, count : Int32) : Nil
    raise ArgumentError.new("MPS attention Q mutation count must be positive") unless count > 0
    MetalOps.check(q, count.to_i64)
    unless q.storage_mode == ML::StorageMode::Shared
      raise ArgumentError.new("MPS attention Q mutation requires a shared F32 buffer")
    end
    chain.encoder.set_pipeline(q_negate_pipeline)
      .set_buffer(q, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_value(count.to_u32, 1)
      .dispatch_1d(count)
  end

  def self.zero_q_tail!(chain : ML::Metal::MPSCommandChain,
                        q : ML::MetalBuffer, element_offset : Int32,
                        count : Int32) : Nil
    unless element_offset >= 0 && count > 0 && element_offset.to_i64 + count <= Int32::MAX
      raise ArgumentError.new("MPS attention Q tail extent must be positive and in range")
    end
    MetalOps.check(q, element_offset.to_i64 + count)
    unless q.storage_mode == ML::StorageMode::Shared
      raise ArgumentError.new("MPS attention Q tail fill requires a shared F32 buffer")
    end
    chain.encoder.set_pipeline(q_tail_zero_pipeline)
      .set_buffer(q, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_value(element_offset.to_u32, 1)
      .set_value(count.to_u32, 2)
      .dispatch_1d(count)
  end

  # Convert caller-owned BTHD K/V into caller-owned BHTD buffers in this chain.
  # The fused dispatch writes each prepared buffer once and introduces no wait
  # or MetalBuffer allocation. Direct callers must serialize access and retain
  # all four buffers until terminal completion; the workspace below enforces
  # reuse of its own buffers across command chains.
  def self.transpose_kv!(chain : ML::Metal::MPSCommandChain,
                         k : ML::MetalBuffer, v : ML::MetalBuffer,
                         prepared_k : ML::MetalBuffer,
                         prepared_v : ML::MetalBuffer,
                         batches : Int32, key_rows : Int32,
                         heads : Int32, head_dim : Int32) : Nil
    count = checked_elements(batches, key_rows, heads, head_dim)
    {k, v, prepared_k, prepared_v}.each do |buffer|
      MetalOps.check(buffer, count)
      unless buffer.storage_mode == ML::StorageMode::Shared
        raise ArgumentError.new("MPS attention K/V transpose requires shared F32 buffers")
      end
    end
    buffers = {k, v, prepared_k, prepared_v}
    buffers.each_with_index do |left, i|
      buffers.each_with_index do |right, j|
        next if j <= i
        if MetalOps.overlap?(left, right)
          raise ArgumentError.new("MPS attention K/V transpose buffers must be pairwise disjoint")
        end
      end
    end
    chain.encoder
      .set_pipeline(head_major_kv_transpose_pipeline)
      .set_buffer(k, 0, ML::Metal::BufferAccess::Read)
      .set_buffer(prepared_k, 1, ML::Metal::BufferAccess::Write)
      .set_buffer(v, 2, ML::Metal::BufferAccess::Read)
      .set_buffer(prepared_v, 3, ML::Metal::BufferAccess::Write)
      .set_value(batches.to_u32, 4)
      .set_value(key_rows.to_u32, 5)
      .set_value(heads.to_u32, 6)
      .set_value(head_dim.to_u32, 7)
      .set_value(count.to_u32, 8)
      .dispatch_1d(count.to_i32)
  end

  private def self.q_negate_pipeline : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get("wan_attention_mps_negate_q") do
      ML::Metal::ComputePipeline.new("wan_attention_mps_negate_q", Q_NEGATE_SOURCE)
    end
  end

  private def self.q_tail_zero_pipeline : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get("wan_attention_mps_zero_q_tail") do
      ML::Metal::ComputePipeline.new("wan_attention_mps_zero_q_tail", Q_TAIL_ZERO_SOURCE)
    end
  end

  private def self.head_major_kv_transpose_pipeline : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get("wan_attention_mps_transpose_head_major_kv") do
      ML::Metal::ComputePipeline.new("wan_attention_mps_transpose_head_major_kv",
        HEAD_MAJOR_KV_TRANSPOSE_SOURCE)
    end
  end

  private def self.checked_elements(*dimensions : Int32) : Int64
    raise ArgumentError.new("MPS attention K/V dimensions must be positive") unless dimensions.all? { |n| n > 0 }
    product = 1_i64
    dimensions.each do |dimension|
      if product > Int32::MAX.to_i64 // dimension
        raise ArgumentError.new("MPS attention K/V transpose exceeds supported F32 extent")
      end
      product *= dimension
    end
    product
  end
end

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib WanAttentionMPSFFI
    fun plan_create = gs_wan_attention_mps_plan_create(device : Pointer(Void),
                                                       batches : Int32, query_rows : Int32, key_rows : Int32, heads : Int32,
                                                       head_dim : Int32, q : Pointer(Void), k : Pointer(Void), v : Pointer(Void),
                                                       output_a : Pointer(Void), output_b : Pointer(Void)) : Pointer(Void)
    fun plan_create_head_major_kv = gs_wan_attention_mps_plan_create_head_major_kv(
      device : Pointer(Void), batches : Int32, query_rows : Int32, key_rows : Int32,
      heads : Int32, head_dim : Int32, q : Pointer(Void), k : Pointer(Void),
      v : Pointer(Void), output_a : Pointer(Void), output_b : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_wan_attention_mps_encode(plan : Pointer(Void),
                                             command : Pointer(Void), output_slot : Int32) : Int32
    fun copy = gs_wan_attention_mps_copy(command : Pointer(Void),
                                         source : Pointer(Void), destination : Pointer(Void),
                                         source_offset : UInt64, destination_offset : UInt64,
                                         byte_count : UInt64) : Int32
    fun plan_release = gs_wan_attention_mps_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video
  # Opt-in prepared MPSGraph SDPA. Q and both result backings are fixed
  # caller-owned shared F32 BTHD buffers. K/V are BTHD by default; the optional
  # head-major entrypoint binds caller-owned shared F32 BHTD K/V. The caller
  # must keep every binding valid and immutable until the chain's terminal wait.
  # The executable targets only context; MPSGraph may keep private workspace
  # outside MetalBuffer accounting.
  class WanAttentionMPSPlan
    getter batches : Int32
    getter query_rows : Int32
    getter key_rows : Int32
    getter heads : Int32
    getter head_dim : Int32

    @q : ML::MetalBuffer
    @k : ML::MetalBuffer
    @v : ML::MetalBuffer
    @output_a : ML::MetalBuffer
    @output_b : ML::MetalBuffer
    @head_major_kv : Bool
    @handle : Pointer(Void)
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@batches : Int32, @query_rows : Int32, @key_rows : Int32,
                   @heads : Int32, @head_dim : Int32, @q : ML::MetalBuffer,
                   @k : ML::MetalBuffer, @v : ML::MetalBuffer,
                   @output_a : ML::MetalBuffer, @output_b : ML::MetalBuffer,
                   head_major_kv : Bool = false)
      @head_major_kv = head_major_kv
      @handle = Pointer(Void).null
      q_count = checked_elements(@batches, @query_rows, @heads, @head_dim)
      kv_count = checked_elements(@batches, @key_rows, @heads, @head_dim)
      {@q, @output_a, @output_b}.each { |buffer| MetalOps.check(buffer, q_count) }
      {@k, @v}.each { |buffer| MetalOps.check(buffer, kv_count) }
      all = {@q, @k, @v, @output_a, @output_b}
      all.each do |buffer|
        unless buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("MPS attention requires shared F32 buffers")
        end
      end
      all.each_with_index do |left, i|
        all.each_with_index do |right, j|
          next if j <= i
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("MPS attention buffers must be pairwise disjoint")
          end
        end
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        device = ML::Metal::Device.instance.device_handle
        @handle = if @head_major_kv
                    WanAttentionMPSFFI.plan_create_head_major_kv(device,
                      @batches, @query_rows, @key_rows, @heads, @head_dim,
                      @q.handle, @k.handle, @v.handle, @output_a.handle,
                      @output_b.handle)
                  else
                    WanAttentionMPSFFI.plan_create(device, @batches, @query_rows,
                      @key_rows, @heads, @head_dim, @q.handle, @k.handle, @v.handle,
                      @output_a.handle, @output_b.handle)
                  end
        raise "Failed to prepare Wan MPSGraph attention plan" if @handle.null?
      {% else %}
        raise ArgumentError.new("MPSGraph attention requires macOS 15 or later")
      {% end %}
    end

    # Reuse the prepared executable and its fixed inputs, choosing one of two
    # setup-time output bindings. Graph may continue the wrapped command buffer;
    # resume! refreshes the native encoder from the chain's current root.
    def encode(chain : ML::Metal::MPSCommandChain, output_slot : Int32 = 0) : Nil
      raise ArgumentError.new("MPS attention output slot must be 0 or 1") unless {0, 1}.includes?(output_slot)
      raise "Wan MPSGraph attention plan has been released" if @released || @handle.null?
      validate_bindings!
      chain.encoder
      if previous = @chain
        unless previous.same?(chain)
          unless previous.completed?
            raise ArgumentError.new("MPS attention plan cannot span in-flight command chains")
          end
        end
      end
      # A prepared plan may serve a later request after the previous chain has
      # completed. Keep only the most recent chain for release/finalize safety.
      @chain = chain

      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = WanAttentionMPSFFI.encode(@handle, chain.handle, output_slot)
        ensure
          chain.resume!
        end
        raise "Wan MPSGraph attention encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("MPSGraph attention requires macOS 15 or later")
      {% end %}
    end

    # Native blit helper for copying caller-owned regions in the same chain,
    # without allocating replacement MetalBuffers or waiting.
    def self.copy(chain : ML::Metal::MPSCommandChain,
                  source : ML::MetalBuffer, destination : ML::MetalBuffer,
                  byte_count : Int64, source_offset_bytes : Int64 = 0,
                  destination_offset_bytes : Int64 = 0) : Nil
      validate_copy_region!(source, source_offset_bytes, byte_count)
      validate_copy_region!(destination, destination_offset_bytes, byte_count)
      unless source.storage_mode == ML::StorageMode::Shared &&
             destination.storage_mode == ML::StorageMode::Shared
        raise ArgumentError.new("MPS attention copy requires shared buffers")
      end
      if active_ranges_overlap?(source, source_offset_bytes, destination,
           destination_offset_bytes, byte_count)
        raise ArgumentError.new("MPS attention copy active ranges must be disjoint")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = WanAttentionMPSFFI.copy(chain.handle, source.handle,
            destination.handle, source_offset_bytes.to_u64,
            destination_offset_bytes.to_u64, byte_count.to_u64)
        ensure
          chain.resume!
        end
        raise "MPS attention native copy rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("MPSGraph attention requires macOS 15 or later")
      {% end %}
    end

    private def self.validate_copy_region!(buffer : ML::MetalBuffer,
                                           offset_bytes : Int64,
                                           byte_count : Int64) : Nil
      unless buffer.valid? && offset_bytes >= 0 && byte_count > 0 &&
             offset_bytes % sizeof(Float32) == 0 &&
             byte_count % sizeof(Float32) == 0 &&
             offset_bytes <= buffer.size && byte_count <= buffer.size - offset_bytes
        raise ArgumentError.new("MPS attention copy region must be aligned and in range")
      end
      end_elements = (offset_bytes + byte_count) // sizeof(Float32)
      MetalOps.check(buffer, end_elements)
    end

    private def self.active_ranges_overlap?(source : ML::MetalBuffer,
                                            source_offset : Int64,
                                            destination : ML::MetalBuffer,
                                            destination_offset : Int64,
                                            byte_count : Int64) : Bool
      source_start = checked_address(source.contents.address, source_offset)
      destination_start = checked_address(destination.contents.address, destination_offset)
      source_end = checked_address(source_start, byte_count)
      destination_end = checked_address(destination_start, byte_count)
      source_start < destination_end && destination_start < source_end
    end

    private def self.checked_address(base : UInt64, offset : Int64) : UInt64
      raise ArgumentError.new("MPS attention copy address is null") if base == 0
      unless offset >= 0 && offset.to_u64 <= UInt64::MAX - base
        raise ArgumentError.new("MPS attention copy address overflows")
      end
      base + offset.to_u64
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("Complete MPS attention command chain before releasing plan") unless chain.completed?
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        WanAttentionMPSFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
      @released = true
    end

    # If the plan becomes unreachable while commands are in flight, preserve
    # the native graph/buffer ownership rather than creating a use-after-free.
    def finalize
      return if @released
      return if @chain && !@chain.not_nil!.completed?
      release
    end

    private def checked_elements(*dimensions : Int32) : Int64
      raise ArgumentError.new("MPS attention dimensions must be positive") unless dimensions.all? { |n| n > 0 }
      product = 1_i64
      dimensions.each do |dimension|
        if product > Int32::MAX.to_i64 // dimension
          raise ArgumentError.new("MPS attention buffer exceeds supported F32 extent")
        end
        product *= dimension
      end
      product
    end

    private def validate_bindings! : Nil
      q_count = checked_elements(@batches, @query_rows, @heads, @head_dim)
      kv_count = checked_elements(@batches, @key_rows, @heads, @head_dim)
      {@q, @output_a, @output_b}.each { |buffer| MetalOps.check(buffer, q_count) }
      {@k, @v}.each { |buffer| MetalOps.check(buffer, kv_count) }
      {@q, @k, @v, @output_a, @output_b}.each do |buffer|
        unless buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("MPS attention requires shared F32 buffers")
        end
      end
    end
  end

  # Bounded B1 attention workspace for query lengths larger than the prepared
  # graph shape. Original K/V stay global and resident. An opt-in head-major
  # path refreshes caller-owned prepared K/V on every encode before reusing the
  # same graph plan. Query and context scratch also stay caller-owned; each span
  # uses the same serial command chain, with no wait or MetalBuffer allocation.
  class WanAttentionMPSChunkWorkspace
    # The caller still selects the query chunk and owns its scratch/memory
    # admission. A larger maximum does not change any pipeline default.
    MAX_CHUNK_ROWS = 1024

    getter batches : Int32
    getter query_rows : Int32
    getter key_rows : Int32
    getter heads : Int32
    getter head_dim : Int32
    getter chunk_rows : Int32

    @q : ML::MetalBuffer
    @k : ML::MetalBuffer
    @v : ML::MetalBuffer
    @output : ML::MetalBuffer
    @chunk_q : ML::MetalBuffer
    @chunk_output_a : ML::MetalBuffer
    @chunk_output_b : ML::MetalBuffer
    @head_major_k : ML::MetalBuffer?
    @head_major_v : ML::MetalBuffer?
    @plan : WanAttentionMPSPlan
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    # Bytes held by reusable chunk buffers: Q scratch and two prebound MPSGraph
    # result backings. When enabled, the full-size head-major K/V buffers are
    # included too. Original full Q/K/V/output remain caller-accounted.
    def scratch_bytes : Int64
      bytes = checked_elements(1, @chunk_rows, @heads, @head_dim) * sizeof(Float32) * 3
      if head_major_kv?
        bytes += checked_elements(@batches, @key_rows, @heads, @head_dim) * sizeof(Float32) * 2
      end
      bytes
    end

    def head_major_kv? : Bool
      !@head_major_k.nil?
    end

    def initialize(@batches : Int32, @query_rows : Int32, @key_rows : Int32,
                   @heads : Int32, @head_dim : Int32, @chunk_rows : Int32,
                   @q : ML::MetalBuffer, @k : ML::MetalBuffer, @v : ML::MetalBuffer,
                   @output : ML::MetalBuffer, @chunk_q : ML::MetalBuffer,
                   @chunk_output_a : ML::MetalBuffer,
                   @chunk_output_b : ML::MetalBuffer,
                   head_major_k : ML::MetalBuffer? = nil,
                   head_major_v : ML::MetalBuffer? = nil)
      @head_major_k = head_major_k
      @head_major_v = head_major_v
      raise ArgumentError.new("MPS attention chunk workspace supports B=1 only") unless @batches == 1
      unless @chunk_rows > 0 && @chunk_rows <= MAX_CHUNK_ROWS
        raise ArgumentError.new("MPS attention chunk rows must be within 1..#{MAX_CHUNK_ROWS}")
      end
      if @head_major_k.nil? != @head_major_v.nil?
        raise ArgumentError.new("Head-major MPS attention requires both prepared K and V buffers")
      end

      q_count = checked_elements(@batches, @query_rows, @heads, @head_dim)
      kv_count = checked_elements(@batches, @key_rows, @heads, @head_dim)
      chunk_count = checked_elements(1, @chunk_rows, @heads, @head_dim)
      {@q, @output}.each { |buffer| MetalOps.check(buffer, q_count) }
      {@k, @v}.each { |buffer| MetalOps.check(buffer, kv_count) }
      {@chunk_q, @chunk_output_a, @chunk_output_b}.each do |buffer|
        MetalOps.check(buffer, chunk_count)
      end
      if prepared_k = @head_major_k
        prepared_v = @head_major_v || raise ArgumentError.new("Head-major MPS attention requires both prepared K and V buffers")
        {prepared_k, prepared_v}.each do |buffer|
          MetalOps.check(buffer, kv_count)
          unless buffer.storage_mode == ML::StorageMode::Shared
            raise ArgumentError.new("MPS attention chunk workspace requires shared F32 buffers")
          end
        end
      end

      all = {@q, @k, @v, @output, @chunk_q, @chunk_output_a, @chunk_output_b}
      all.each do |buffer|
        unless buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("MPS attention chunk workspace requires shared F32 buffers")
        end
      end
      all.each_with_index do |left, i|
        all.each_with_index do |right, j|
          next if j <= i
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("MPS attention chunk workspace buffers must be pairwise disjoint")
          end
        end
      end
      if prepared_k = @head_major_k
        prepared_v = @head_major_v || raise ArgumentError.new("Head-major MPS attention requires both prepared K and V buffers")
        all.each do |buffer|
          if MetalOps.overlap?(buffer, prepared_k) || MetalOps.overlap?(buffer, prepared_v)
            raise ArgumentError.new("MPS attention chunk workspace buffers must be pairwise disjoint")
          end
        end
        if MetalOps.overlap?(prepared_k, prepared_v)
          raise ArgumentError.new("MPS attention chunk workspace buffers must be pairwise disjoint")
        end
      end

      MetalAttentionMPS.prepare_chunk!
      if prepared_k = @head_major_k
        prepared_v = @head_major_v || raise ArgumentError.new("Head-major MPS attention requires both prepared K and V buffers")
        MetalAttentionMPS.prepare_head_major_kv!
        @plan = WanAttentionMPSPlan.new(1, @chunk_rows, @key_rows, @heads,
          @head_dim, @chunk_q, prepared_k, prepared_v, @chunk_output_a,
          @chunk_output_b, true)
      else
        @plan = WanAttentionMPSPlan.new(1, @chunk_rows, @key_rows, @heads,
          @head_dim, @chunk_q, @k, @v, @chunk_output_a, @chunk_output_b)
      end
    end

    # Validate the fixed block bindings before any enclosing phase is encoded.
    def validate_for!(batches : Int32, query_rows : Int32, key_rows : Int32,
                      heads : Int32, head_dim : Int32,
                      q : ML::MetalBuffer, k : ML::MetalBuffer,
                      v : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      unless batches == @batches && query_rows == @query_rows && key_rows == @key_rows &&
             heads == @heads && head_dim == @head_dim &&
             q.same?(@q) && k.same?(@k) && v.same?(@v) && output.same?(@output)
        raise ArgumentError.new("Prepared Wan attention shape or caller buffers changed")
      end
      raise "Wan MPS attention chunk workspace has been released" if @released
      validate_bindings!
    end

    # Splits Q into disjoint spans. The last short span is padded with zeros
    # before Graph execution; only its valid rows are copied to the result.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise "Wan MPS attention chunk workspace has been released" if @released
      validate_bindings!
      chain.encoder
      if previous = @chain
        unless previous.same?(chain)
          unless previous.completed?
            raise ArgumentError.new("MPS attention chunk workspace cannot span in-flight command chains")
          end
        end
      end
      # Requests are serialized through the same caller-owned buffers. A fresh
      # chain is admitted only after the prior request's terminal wait.
      @chain = chain

      if prepared_k = @head_major_k
        prepared_v = @head_major_v || raise ArgumentError.new("Head-major MPS attention requires both prepared K and V buffers")
        # Refresh the prepared layout for every encode because K/V contents may
        # have changed since the preceding request.
        MetalAttentionMPS.transpose_kv!(chain, @k, @v, prepared_k, prepared_v,
          @batches, @key_rows, @heads, @head_dim)
      end

      channels = @heads.to_i64 * @head_dim
      query_offset = 0
      while query_offset < @query_rows
        valid_rows = Math.min(@chunk_rows, @query_rows - query_offset)
        valid_elements = valid_rows.to_i64 * channels
        chunk_elements = @chunk_rows.to_i64 * channels
        source_offset_bytes = query_offset.to_i64 * channels * sizeof(Float32)
        destination_offset_bytes = query_offset.to_i64 * channels * sizeof(Float32)
        valid_bytes = valid_elements * sizeof(Float32)

        WanAttentionMPSPlan.copy(chain, @q, @chunk_q, valid_bytes,
          source_offset_bytes, 0_i64)
        if valid_rows < @chunk_rows
          MetalAttentionMPS.zero_q_tail!(chain, @chunk_q,
            valid_elements.to_i32, (chunk_elements - valid_elements).to_i32)
        end
        @plan.encode(chain, 0)
        WanAttentionMPSPlan.copy(chain, @chunk_output_a, @output, valid_bytes,
          0_i64, destination_offset_bytes)
        query_offset += valid_rows
      end
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("Complete MPS attention command chain before releasing workspace") unless chain.completed?
      end
      @plan.release
      @released = true
    end

    def finalize
      return if @released
      return if @chain && !@chain.not_nil!.completed?
      release
    end

    private def validate_bindings! : Nil
      q_count = checked_elements(@batches, @query_rows, @heads, @head_dim)
      kv_count = checked_elements(@batches, @key_rows, @heads, @head_dim)
      chunk_count = checked_elements(1, @chunk_rows, @heads, @head_dim)
      {@q, @output}.each { |buffer| MetalOps.check(buffer, q_count) }
      {@k, @v}.each { |buffer| MetalOps.check(buffer, kv_count) }
      {@chunk_q, @chunk_output_a, @chunk_output_b}.each do |buffer|
        MetalOps.check(buffer, chunk_count)
      end
      all = {@q, @k, @v, @output, @chunk_q, @chunk_output_a, @chunk_output_b}
      all.each do |buffer|
        unless buffer.storage_mode == ML::StorageMode::Shared
          raise ArgumentError.new("MPS attention chunk workspace requires shared F32 buffers")
        end
      end
      all.each_with_index do |left, i|
        all.each_with_index do |right, j|
          next if j <= i
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("MPS attention chunk workspace buffers must be pairwise disjoint")
          end
        end
      end
      if prepared_k = @head_major_k
        prepared_v = @head_major_v || raise ArgumentError.new("Head-major MPS attention requires both prepared K and V buffers")
        {prepared_k, prepared_v}.each do |buffer|
          MetalOps.check(buffer, kv_count)
          unless buffer.storage_mode == ML::StorageMode::Shared
            raise ArgumentError.new("MPS attention chunk workspace requires shared F32 buffers")
          end
          all.each do |other|
            if MetalOps.overlap?(other, buffer)
              raise ArgumentError.new("MPS attention chunk workspace buffers must be pairwise disjoint")
            end
          end
        end
        if MetalOps.overlap?(prepared_k, prepared_v)
          raise ArgumentError.new("MPS attention chunk workspace buffers must be pairwise disjoint")
        end
      end
    end

    private def checked_elements(*dimensions : Int32) : Int64
      raise ArgumentError.new("MPS attention dimensions must be positive") unless dimensions.all? { |n| n > 0 }
      product = 1_i64
      dimensions.each do |dimension|
        if product > Int32::MAX.to_i64 // dimension
          raise ArgumentError.new("MPS attention buffer exceeds supported F32 extent")
        end
        product *= dimension
      end
      product
    end
  end
end
