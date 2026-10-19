require "./metal_ops"

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
  # The caller owns the serial encoder and all lifetimes through completion.
  def self.attention_tiled(e : ML::Metal::ComputeEncoder, q : ML::MetalBuffer,
                           k : ML::MetalBuffer, v : ML::MetalBuffer,
                           y : ML::MetalBuffer, q_rows : Int32, k_rows : Int32,
                           heads : Int32, head_dim : Int32,
                           batches : Int32 = 1) : Nil
    dims(q_rows, k_rows, heads, head_dim, batches)
    unless {64, 128}.includes?(head_dim)
      raise ArgumentError.new("Tiled video attention requires head dimension 64 or 128")
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
      scale.unsafe_as(UInt32)]
    groups = batches.to_i64 * ((q_rows.to_i64 + 7) // 8) * heads
    raise ArgumentError.new("Tiled video attention dispatch is too large") if groups > Int32::MAX
    e.set_pipeline(tiled_attention_pipeline(head_dim))
    e.set_buffer(q, 0).set_buffer(k, 1).set_buffer(v, 2)
    e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
    e.set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 4)
    e.dispatch_threadgroups({groups.to_i32, 1, 1}, {32, 1, 1})
  end
end
