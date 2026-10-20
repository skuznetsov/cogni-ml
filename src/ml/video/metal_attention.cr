require "./metal_ops"
require "./metal_mps_command_chain"

module ML::Video::MetalOps
  TILED_ATTENTION_SOURCE = {{ read_file("#{__DIR__}/kernels/attention_tiled.metal") }}

  def self.tiled_attention_pipeline(head_dim : Int32) : ML::Metal::ComputePipeline
    unless {64, 128}.includes?(head_dim)
      raise ArgumentError.new("Tiled video attention requires head dimension 64 or 128")
    end
    name = "video_tiled_attention_#{head_dim}"
    ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, TILED_ATTENTION_SOURCE) }
  end

  # Setup only; the unchanged attention() primitive remains the default.
  def self.prepare_tiled_attention! : Nil
    tiled_attention_pipeline(64)
    tiled_attention_pipeline(128)
  end

  # Unmasked F32 SDPA: Q[B,Q,H,D], K/V[B,K,H,D]. Eight query rows share each
  # ascending block of 32 keys. All operands, products and accumulators are F32;
  # no half/BF16 activation conversion or quadratic score buffer is used.
  # The caller owns the serial encoder and all buffer lifetimes through completion.
  def self.attention_tiled(e : ML::Metal::ComputeEncoder, q : ML::MetalBuffer,
                           k : ML::MetalBuffer, v : ML::MetalBuffer,
                           y : ML::MetalBuffer, q_rows : Int32, k_rows : Int32,
                           heads : Int32, head_dim : Int32,
                           batches : Int32 = 1) : Nil
    groups = tiled_attention_group_count(q_rows, heads, batches)
    attention_tiled_range(e, q, k, v, y, q_rows, k_rows, heads, head_dim,
      0, groups, batches)
  end

  # Dispatch a contiguous range of global threadgroups. A global group index
  # maps to the same (batch, query tile, head) as the unsplit kernel, so splitting
  # never changes a query's ascending key reduction or its accumulation order.
  def self.attention_tiled_range(e : ML::Metal::ComputeEncoder, q : ML::MetalBuffer,
                                 k : ML::MetalBuffer, v : ML::MetalBuffer,
                                 y : ML::MetalBuffer, q_rows : Int32, k_rows : Int32,
                                 heads : Int32, head_dim : Int32,
                                 first_tile : Int32, tile_count : Int32,
                                 batches : Int32 = 1) : Nil
    dims(q_rows, k_rows, heads, head_dim, batches)
    unless {64, 128}.includes?(head_dim)
      raise ArgumentError.new("Tiled video attention requires head dimension 64 or 128")
    end
    total_groups = tiled_attention_group_count(q_rows, heads, batches)
    range_end = first_tile.to_i64 + tile_count
    unless first_tile >= 0 && tile_count > 0 && range_end <= total_groups
      raise ArgumentError.new("Tiled video attention range is outside the dispatch")
    end

    query_rows = batches.to_i64 * q_rows
    key_rows = batches.to_i64 * k_rows
    channels = heads.to_i64 * head_dim
    if query_rows > Int32::MAX || key_rows > Int32::MAX || channels > Int32::MAX
      raise ArgumentError.new("Tiled video attention extent is too large")
    end
    q_count = query_rows * channels
    kv_count = key_rows * channels
    check(q, q_count)
    check(k, kv_count)
    check(v, kv_count)
    check(y, q_count)
    distinct(y, q, k, v)
    scale = 1_f32 / Math.sqrt(head_dim.to_f32)
    params = StaticArray[q_rows.to_u32, k_rows.to_u32, heads.to_u32,
      scale.unsafe_as(UInt32), first_tile.to_u32]
    e.set_pipeline(tiled_attention_pipeline(head_dim))
    e.set_buffer(q, 0).set_buffer(k, 1).set_buffer(v, 2)
    e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
    e.set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 4)
    e.dispatch_threadgroups({tile_count, 1, 1}, {32, 1, 1})
  end

  # Chain overload for long attention dispatches. Each segment dispatches a
  # contiguous range of whole query-tile/head groups; no intermediate output is
  # read and no buffer is allocated or rebound with an offset. The caller must
  # retain q/k/v/y (normally by retaining their Arena) until its single terminal
  # wait. The final segment leaves the current encoder open for subsequent work.
  def self.attention_tiled(chain : ML::Metal::MPSCommandChain,
                           q : ML::MetalBuffer, k : ML::MetalBuffer,
                           v : ML::MetalBuffer, y : ML::MetalBuffer,
                           q_rows : Int32, k_rows : Int32, heads : Int32,
                           head_dim : Int32, batches : Int32 = 1,
                           groups_per_segment : Int32 = 4096) : Nil
    unless groups_per_segment > 0
      raise ArgumentError.new("Tiled video attention segment size must be positive")
    end
    total_groups = tiled_attention_group_count(q_rows, heads, batches)
    first_tile = 0
    while first_tile < total_groups
      count = Math.min(groups_per_segment, total_groups - first_tile)
      attention_tiled_range(chain.encoder, q, k, v, y, q_rows, k_rows, heads,
        head_dim, first_tile, count, batches)
      first_tile += count
      unless first_tile == total_groups
        chain.pause!
        chain.continue_async!
        chain.resume!
      end
    end
  end

  private def self.tiled_attention_group_count(q_rows : Int32, heads : Int32,
                                               batches : Int32) : Int32
    dims(q_rows, heads, batches)
    groups_per_batch = ((q_rows.to_i64 + 7) // 8) * heads
    if groups_per_batch > Int32::MAX.to_i64 // batches
      raise ArgumentError.new("Tiled video attention dispatch is too large")
    end
    (groups_per_batch * batches).to_i32
  end
end
