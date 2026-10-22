require "../../src/ml/video/metal_attention_mps"

# Probe adapter for the pinned, single-specialization STEEL shader. The shader
# is loaded and compiled at setup; no MLX runtime participates in execution.
class WanS2VSteelAttentionProbe
  QUERY_TILE         =  32
  KEY_TILE           =  32
  SIMD_GROUPS        =   8
  HEAD_DIM           = 128
  GROUPS_PER_SEGMENT =  80 # 320 eight-query/SIMD tile equivalents.

  @params : StaticArray(Int32, 38)
  @pipeline : ML::Metal::ComputePipeline
  @chain : ML::Metal::MPSCommandChain? = nil
  @released = false

  def initialize(@query_rows : Int32, @key_rows : Int32, @heads : Int32,
                 @q : ML::MetalBuffer, @k : ML::MetalBuffer,
                 @v : ML::MetalBuffer, @output : ML::MetalBuffer,
                 @prepared_k : ML::MetalBuffer, @prepared_v : ML::MetalBuffer,
                 source : String, @key_shard_rows : Int32 = 0,
                 @row_state : ML::MetalBuffer? = nil)
    unless @query_rows > 0 && @key_rows > 0 && @heads > 0 && @heads % 4 == 0
      raise ArgumentError.new("STEEL probe dimensions must be positive")
    end
    unless @key_shard_rows == 0 || @key_shard_rows == 2_048
      raise ArgumentError.new("STEEL probe key shard must be zero or 2048")
    end
    if (@key_shard_rows > 0) != !@row_state.nil?
      raise ArgumentError.new("STEEL key shards require a preallocated row-state buffer")
    end
    validate_bindings!
    # AttnParams ABI: fourteen 32-bit words followed by twelve int64 strides.
    # The sixth word after B,H,D,qL,kL,gqa_factor is the F32 scale.
    @params = StaticArray(Int32, 38).new(0)
    words = @params.to_unsafe
    values = StaticArray[1, @heads, HEAD_DIM, @query_rows, @key_rows, 1,
      (1_f32 / Math.sqrt(HEAD_DIM.to_f32)).unsafe_as(Int32),
      (@query_rows + QUERY_TILE - 1) // QUERY_TILE,
      (@key_rows + KEY_TILE - 1) // KEY_TILE,
      @query_rows // QUERY_TILE, @key_rows // KEY_TILE,
      @query_rows % QUERY_TILE, @key_rows % KEY_TILE, @key_rows - @query_rows]
    values.each_with_index { |value, i| words[i] = value }
    channels = @heads.to_i64 * HEAD_DIM
    strides = (words + 14).as(Int64*)
    # Q/O remain BQHD; K/V are refreshed into BHKD on every encode.
    {@query_rows.to_i64 * channels, HEAD_DIM.to_i64, channels,
     @key_rows.to_i64 * channels, @key_rows.to_i64 * HEAD_DIM, HEAD_DIM.to_i64,
     @key_rows.to_i64 * channels, @key_rows.to_i64 * HEAD_DIM, HEAD_DIM.to_i64,
     @query_rows.to_i64 * channels, HEAD_DIM.to_i64, channels}.each_with_index do |value, i|
      strides[i] = value
    end
    function = @key_shard_rows == 0 ? "wan_steel_attention_f32_128" : "wan_steel_sharded_attention_f32_128"
    @pipeline = ML::Metal::ComputePipeline.new(function, source)
    unless @pipeline.thread_execution_width == 32 &&
           @pipeline.max_total_threads_per_threadgroup >= 32 * SIMD_GROUPS &&
           @pipeline.static_threadgroup_memory_length <= 32_768
      raise ArgumentError.new("STEEL probe pipeline exceeds admitted capabilities")
    end
    ML::Video::MetalAttentionMPS.prepare_head_major_kv!
  end

  def distinct_executable_count : Int32
    0 # Native Metal pipeline; no MPSGraph executable.
  end

  def encode(chain : ML::Metal::MPSCommandChain) : Nil
    begin_encode!(chain)
    ML::Video::MetalAttentionMPS.transpose_kv!(chain, @k, @v,
      @prepared_k, @prepared_v, 1, @key_rows, @heads, HEAD_DIM)
    encode_attention(chain)
  end

  # The enclosing block supplies its validated fused K/V writer. Track the
  # chain before the callback so an exception cannot bypass the release guard.
  def encode_with_kv_preparation(chain : ML::Metal::MPSCommandChain,
                                 &block : ML::Metal::ComputeEncoder, ML::MetalBuffer, ML::MetalBuffer -> Nil) : Nil
    begin_encode!(chain)
    yield chain.encoder, @prepared_k, @prepared_v
    encode_attention(chain)
  end

  private def begin_encode!(chain : ML::Metal::MPSCommandChain) : Nil
    raise "STEEL probe has been released" if @released
    validate_bindings!
    if previous = @chain
      unless previous.same?(chain) || previous.completed?
        raise ArgumentError.new("STEEL probe cannot span in-flight chains")
      end
    end
    @chain = chain
  end

  private def encode_attention(chain : ML::Metal::MPSCommandChain) : Nil
    if @key_shard_rows > 0
      encode_key_shards(chain)
      return
    end
    total_groups = ((@query_rows + QUERY_TILE - 1) // QUERY_TILE) * @heads
    first = 0
    while first < total_groups
      count = Math.min(GROUPS_PER_SEGMENT, total_groups - first)
      e = chain.encoder
      e.set_pipeline(@pipeline).set_buffer(@q, 0).set_buffer(@prepared_k, 1)
        .set_buffer(@prepared_v, 2).set_buffer(@output, 3, ML::Metal::BufferAccess::Write)
      e.set_bytes(@params.to_unsafe.as(Void*), sizeof(typeof(@params)), 4)
        .set_value(first.to_u32, 9)
        .dispatch_threadgroups({count, 1, 1}, {32, SIMD_GROUPS, 1})
      first += count
      if first < total_groups
        chain.pause!
        chain.continue_async!
        chain.resume!
      end
    end
  end

  private def encode_key_shards(chain : ML::Metal::MPSCommandChain) : Nil
    query_tiles = (@query_rows + QUERY_TILE - 1) // QUERY_TILE
    group_extent = query_tiles * 4
    # Bound key-loop tile work per dispatch by the unsharded 80-group arm.
    max_tile_work = GROUPS_PER_SEGMENT * ((@key_rows + KEY_TILE - 1) // KEY_TILE)
    (@heads // 4).times do |head_group|
      first_key = 0
      while first_key < @key_rows
        key_count = Math.min(@key_shard_rows, @key_rows - first_key)
        key_tiles = (key_count + KEY_TILE - 1) // KEY_TILE
        groups_per_dispatch = Math.max(1, max_tile_work // key_tiles)
        @params[4] = key_count
        @params[8] = key_tiles
        @params[10] = key_count // KEY_TILE
        @params[12] = key_count % KEY_TILE
        @params[13] = key_count - @query_rows
        final_shard = first_key + key_count == @key_rows
        first_tile = head_group * group_extent
        group_end = first_tile + group_extent
        while first_tile < group_end
          count = Math.min(groups_per_dispatch, group_end - first_tile)
          e = chain.encoder
          e.set_pipeline(@pipeline).set_buffer(@q, 0).set_buffer(@prepared_k, 1)
            .set_buffer(@prepared_v, 2).set_buffer(@output, 3, ML::Metal::BufferAccess::ReadWrite)
          e.set_bytes(@params.to_unsafe.as(Void*), sizeof(typeof(@params)), 4)
            .set_value(first_tile.to_u32, 9).set_value(first_key.to_u32, 10)
            .set_buffer(@row_state.not_nil!, 11, ML::Metal::BufferAccess::ReadWrite)
            .set_value(final_shard ? 1_u32 : 0_u32, 12)
            .dispatch_threadgroups({count, 1, 1}, {32, SIMD_GROUPS, 1})
          first_tile += count
          unless first_tile == group_end && final_shard && head_group == @heads // 4 - 1
            # Ordered queue roots preserve the spilled state without a host wait.
            chain.pause!
            chain.continue_async!
            chain.resume!
          end
        end
        first_key += key_count
      end
    end
  end

  def release : Nil
    return if @released
    if chain = @chain
      raise ArgumentError.new("STEEL probe still in flight") unless chain.completed?
    end
    @released = true
  end

  private def validate_bindings! : Nil
    q_count = @query_rows.to_i64 * @heads * HEAD_DIM
    kv_count = @key_rows.to_i64 * @heads * HEAD_DIM
    raise ArgumentError.new("STEEL probe extent exceeds Int32") if Math.max(q_count, kv_count) > Int32::MAX
    {@q, @output}.each { |buffer| ML::Video::MetalOps.check(buffer, q_count) }
    {@k, @v, @prepared_k, @prepared_v}.each { |buffer| ML::Video::MetalOps.check(buffer, kv_count) }
    buffers = {@q, @k, @v, @output, @prepared_k, @prepared_v}
    if state = @row_state
      ML::Video::MetalOps.check(state, @query_rows.to_i64 * @heads * 2)
      buffers.each do |buffer|
        if ML::Video::MetalOps.overlap?(state, buffer)
          raise ArgumentError.new("STEEL row state must be disjoint from tensor buffers")
        end
      end
    end
    buffers.each_with_index do |left, i|
      buffers.each_with_index do |right, j|
        if j > i && ML::Video::MetalOps.overlap?(left, right)
          raise ArgumentError.new("STEEL probe buffers must be pairwise disjoint")
        end
      end
    end
  end
end
