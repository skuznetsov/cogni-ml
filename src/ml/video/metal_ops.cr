require "../metal/dispatch"

module ML::Video
  # Setup-only owner. Keep this object and all external inputs alive until the
  # caller's command completes. Do not release or upload while work is in flight.
  class Arena
    getter buffers = {} of String => ML::MetalBuffer

    def initialize
      raise "Native video requires Metal" unless ML::Metal::Device.init!
      MetalOps.prepare!
    end

    def alloc(name : String, count : Int32) : ML::MetalBuffer
      raise ArgumentError.new("Duplicate buffer #{name}") if @buffers.has_key?(name)
      raise ArgumentError.new("Buffer count must be positive") unless count > 0
      @buffers[name] = ML::MetalBuffer.new(count.to_i64 * sizeof(Float32))
    end

    def upload(name : String, values : Array(Float32)) : ML::MetalBuffer
      buffer = alloc(name, values.size)
      buffer.write(values)
      buffer
    end

    def [](name : String) : ML::MetalBuffer
      @buffers[name]
    end

    def release : Nil
      @buffers.each_value(&.release)
      @buffers.clear
    end
  end

  # Independent F32 correctness primitives for opt-in video block admission.
  # All encode methods only bind/dispatch into an existing SERIAL encoder.
  # This deliberately does not promise production-size GEMM/attention speed.
  module MetalOps
    SOURCE = {{ read_file("#{__DIR__}/kernels/common.metal") }}

    def self.check(buffer : ML::MetalBuffer, count : Int64) : Nil
      unless buffer.valid? && count > 0 && count <= Int32::MAX && buffer.size >= count * 4
        raise ArgumentError.new("Invalid or undersized resident video buffer")
      end
    end

    def self.dims(*values : Int32) : Nil
      raise ArgumentError.new("Video dimensions must be positive") unless values.all? { |v| v > 0 }
    end

    def self.distinct(output : ML::MetalBuffer, *inputs : ML::MetalBuffer) : Nil
      if inputs.any? { |input| overlap?(input, output) }
        raise ArgumentError.new("Unsafe video buffer alias")
      end
    end

    # Distinct no-copy MTLBuffer wrappers can expose the same unified-memory
    # region. Handle inequality alone is not an alias certificate.
    def self.overlap?(a : ML::MetalBuffer, b : ML::MetalBuffer) : Bool
      return true if a.handle == b.handle
      pa = a.contents.address
      pb = b.contents.address
      pa != 0 && pb != 0 && pa < pb + b.size.to_u64 && pb < pa + a.size.to_u64
    end

    def self.inplace_or_distinct(input : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      return unless overlap?(input, output)
      return if input.handle == output.handle || input.contents == output.contents
      raise ArgumentError.new("Partially overlapping video buffers")
    end

    def self.prepare! : Nil
      {"video_linear_f32", "video_norm_f32", "video_modulate_f32",
       "video_add_f32", "video_gelu_f32", "video_copy_f32",
       "video_attention_f32"}.each { |name| pipeline(name) }
    end

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.linear(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                    w : ML::MetalBuffer, bias : ML::MetalBuffer, y : ML::MetalBuffer,
                    rows : Int32, in_dim : Int32, out_dim : Int32) : Nil
      dims(rows, in_dim, out_dim)
      check(x, rows.to_i64 * in_dim)
      check(w, out_dim.to_i64 * in_dim)
      check(bias, out_dim.to_i64)
      check(y, rows.to_i64 * out_dim)
      distinct(y, x, w, bias)
      e.set_pipeline(pipeline("video_linear_f32"))
      e.set_buffer(x, 0).set_buffer(w, 1).set_buffer(bias, 2)
      e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      e.set_value(rows.to_u32, 4).set_value(in_dim.to_u32, 5).set_value(out_dim.to_u32, 6)
      e.dispatch_1d(rows * out_dim)
    end

    def self.norm(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                  weight : ML::MetalBuffer, bias : ML::MetalBuffer, y : ML::MetalBuffer,
                  rows : Int32, dim : Int32, eps : Float32, rms : Bool = false,
                  affine : Bool = false) : Nil
      dims(rows, dim)
      raise ArgumentError.new("Invalid norm epsilon") unless eps.finite? && eps > 0
      check(x, rows.to_i64 * dim)
      check(y, rows.to_i64 * dim)
      inplace_or_distinct(x, y)
      if affine
        check(weight, dim.to_i64)
        check(bias, dim.to_i64)
        distinct(y, weight, bias)
      end
      e.set_pipeline(pipeline("video_norm_f32"))
      # Non-affine kernels never read these slots. Bind a known-valid input,
      # rather than accepting stale handles in nominally unused arguments.
      e.set_buffer(x, 0).set_buffer(affine ? weight : x, 1).set_buffer(affine ? bias : x, 2)
      e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      e.set_value(dim.to_u32, 4).set_value(eps, 5)
      e.set_value(rms ? 1_u32 : 0_u32, 6).set_value(affine ? 1_u32 : 0_u32, 7)
      threads = Math.min(256, ((dim + 31) // 32) * 32)
      e.dispatch_threadgroups({rows, 1, 1}, {threads, 1, 1})
    end

    def self.modulate(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                      shift : ML::MetalBuffer, scale : ML::MetalBuffer,
                      y : ML::MetalBuffer, rows : Int32, dim : Int32,
                      mod_rows : Int32 = 1) : Nil
      dims(rows, dim, mod_rows)
      raise ArgumentError.new("Invalid modulation broadcast") unless mod_rows == 1 || mod_rows == rows
      count = rows.to_i64 * dim
      check(x, count)
      check(y, count)
      inplace_or_distinct(x, y)
      check(shift, mod_rows.to_i64 * dim)
      check(scale, mod_rows.to_i64 * dim)
      distinct(y, shift, scale)
      e.set_pipeline(pipeline("video_modulate_f32"))
      e.set_buffer(x, 0).set_buffer(shift, 1).set_buffer(scale, 2)
      e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 4).set_value(dim.to_u32, 5).set_value(mod_rows.to_u32, 6)
      e.dispatch_1d(count.to_i32)
    end

    def self.add(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                 delta : ML::MetalBuffer, y : ML::MetalBuffer, count : Int32,
                 gate : ML::MetalBuffer? = nil, dim : Int32 = 0,
                 gate_rows : Int32 = 1) : Nil
      dims(count)
      check(x, count.to_i64)
      check(delta, count.to_i64)
      check(y, count.to_i64)
      inplace_or_distinct(x, y)
      inplace_or_distinct(delta, y)
      if gate
        dims(dim, gate_rows)
        unless count % dim == 0 && (gate_rows == 1 || gate_rows == count // dim)
          raise ArgumentError.new("Invalid gate broadcast")
        end
        check(gate, gate_rows.to_i64 * dim)
        distinct(y, gate)
      end
      e.set_pipeline(pipeline("video_add_f32"))
      e.set_buffer(x, 0).set_buffer(delta, 1).set_buffer(gate || x, 2)
      e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 4).set_value(gate ? 1_u32 : 0_u32, 5)
      e.set_value(Math.max(dim, 1).to_u32, 6).set_value(gate_rows.to_u32, 7)
      e.dispatch_1d(count)
    end

    def self.gelu(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                  y : ML::MetalBuffer, count : Int32, approximate : Bool = true) : Nil
      dims(count)
      check(x, count.to_i64)
      check(y, count.to_i64)
      inplace_or_distinct(x, y)
      e.set_pipeline(pipeline("video_gelu_f32"))
      e.set_buffer(x, 0).set_buffer(y, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2).set_value(approximate ? 1_u32 : 0_u32, 3)
      e.dispatch_1d(count)
    end

    def self.copy(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                  y : ML::MetalBuffer, count : Int32) : Nil
      dims(count)
      check(x, count.to_i64)
      check(y, count.to_i64)
      inplace_or_distinct(x, y)
      return if x.handle == y.handle
      e.set_pipeline(pipeline("video_copy_f32"))
      e.set_buffer(x, 0).set_buffer(y, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2)
      e.dispatch_1d(count)
    end

    # Dense unmasked attention only. Different batches never attend each other.
    # Q:[B,Q,H,D], K/V:[B,K,H,D]. One threadgroup owns each output query/head.
    def self.attention(e : ML::Metal::ComputeEncoder, q : ML::MetalBuffer,
                       k : ML::MetalBuffer, v : ML::MetalBuffer, y : ML::MetalBuffer,
                       q_rows : Int32, k_rows : Int32, heads : Int32,
                       head_dim : Int32, batches : Int32 = 1) : Nil
      dims(q_rows, k_rows, heads, head_dim, batches)
      raise ArgumentError.new("Unsupported attention head dimension") if head_dim > 256
      q_count = batches.to_i64 * q_rows * heads * head_dim
      kv_count = batches.to_i64 * k_rows * heads * head_dim
      check(q, q_count)
      check(k, kv_count)
      check(v, kv_count)
      check(y, q_count)
      distinct(y, q, k, v)
      e.set_pipeline(pipeline("video_attention_f32"))
      e.set_buffer(q, 0).set_buffer(k, 1).set_buffer(v, 2)
      e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      e.set_value(q_rows.to_u32, 4).set_value(k_rows.to_u32, 5)
      e.set_value(heads.to_u32, 6).set_value(head_dim.to_u32, 7)
      e.set_value((1.0_f32 / Math.sqrt(head_dim.to_f32)), 8)
      threads = ((head_dim + 31) // 32) * 32
      e.dispatch_threadgroups({(batches.to_i64 * q_rows * heads).to_i32, 1, 1}, {threads, 1, 1})
    end
  end
end
