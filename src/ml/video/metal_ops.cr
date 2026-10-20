require "../metal/dispatch"
require "./metal_stage_pool"

module ML::Video
  enum WeightDType
    F32
    BF16

    def bytes_per_element : Int32
      case self
      in .f32?  then 4
      in .bf16? then 2
      end
    end
  end

  # Setup-only owner. Keep this object and all external inputs alive until the
  # caller's command completes. Do not release or upload while work is in flight.
  class Arena
    getter buffers = {} of String => ML::MetalBuffer
    @dtypes = {} of String => WeightDType
    @typed_buffers = {} of String => ML::MetalBuffer
    @stage_pool : MetalStagePool?
    @stage_phase : String?
    @stage_lane : String
    @stage_views = {} of UInt64 => MetalStageBufferView
    @stage_names = {} of String => UInt64

    # When stage_pool is provided, only alloc uses its uninitialized scratch
    # path. upload always owns a normal MetalBuffer so small CPU-written
    # constants cannot overwrite another phase's shared backing.
    def initialize(@stage_pool : MetalStagePool? = nil,
                   @stage_phase : String? = nil,
                   @stage_lane : String = "default")
      if @stage_pool && !@stage_phase
        raise ArgumentError.new("A stage phase is required with a stage pool")
      end
      if phase = @stage_phase
        raise ArgumentError.new("Stage phase name must not be empty") if phase.empty?
      end
      raise ArgumentError.new("Stage lane name must not be empty") if @stage_lane.empty?
      raise "Native video requires Metal" unless ML::Metal::Device.init!
      MetalOps.prepare!
    end

    # Setup-only accounting. Typed allocations remain Arena-owned even if the
    # public buffer map is edited, so include both sets and deduplicate aliases.
    def resident_bytes : Int64
      resident = [] of ML::MetalBuffer
      @buffers.each_value do |buffer|
        resident << buffer unless resident.any? { |other| other.same?(buffer) }
      end
      @typed_buffers.each_value do |buffer|
        resident << buffer unless resident.any? { |other| other.same?(buffer) }
      end
      @stage_views.each_value do |view|
        buffer = view.buffer
        resident << buffer unless resident.any? { |other| other.same?(buffer) }
      end

      total = 0_i64
      resident.each do |buffer|
        next unless buffer.valid?
        raise ArgumentError.new("Arena resident byte count exceeds Int64") if buffer.size > Int64::MAX - total
        total += buffer.size
      end
      total
    end

    def allocated?(name : String) : Bool
      @buffers.has_key?(name) || @typed_buffers.has_key?(name)
    end

    # Setup-only capability query. Lifetime aliases currently require ordinary
    # allocations; reject unsupported layouts before allocating their owners.
    def pooled_allocations? : Bool
      !@stage_pool.nil?
    end

    def alloc(name : String, count : Int32) : ML::MetalBuffer
      raise ArgumentError.new("Duplicate buffer #{name}") if allocated?(name)
      raise ArgumentError.new("Buffer count must be positive") unless count > 0
      if pool = @stage_pool
        view = pool.allocate!(@stage_phase.not_nil!, @stage_lane, count.to_i64 * sizeof(Float32))
        begin
          adopt_stage_view(name, view, WeightDType::F32, false)
        rescue ex
          if view.arena_owned?
            view.release_from_arena!(object_id)
          else
            view.release!
          end
          raise ex
        end
        return view.buffer
      end
      buffer = ML::MetalBuffer.new(count.to_i64 * sizeof(Float32))
      @buffers[name] = buffer
      @dtypes[name] = WeightDType::F32
      buffer
    end

    # Setup-only alias for a normal F32 allocation. The name shares the exact
    # MetalBuffer object and backing storage with its owner; removing either
    # name leaves the storage alive until the final normal or typed reference
    # is removed. Stage-backed and typed allocations have separate ownership
    # rules and cannot be aliased through this API.
    def alias_f32(name : String, owner : String) : ML::MetalBuffer
      raise ArgumentError.new("Duplicate buffer #{name}") if allocated?(name)
      buffer = @buffers[owner]?
      raise ArgumentError.new("Missing F32 alias owner #{owner}") unless buffer
      raise ArgumentError.new("F32 alias owner is not valid") unless buffer.valid?
      raise ArgumentError.new("Cannot alias stage-backed F32 owner #{owner}") if pooled_buffer?(buffer)

      owner_dtype = @dtypes[owner]?
      typed_owner = @typed_buffers.values.any? { |typed| typed.same?(buffer) }
      unless owner_dtype && owner_dtype.f32? && !typed_owner
        raise ArgumentError.new("F32 alias owner must be a normal F32 buffer")
      end

      @buffers[name] = buffer
      @dtypes[name] = WeightDType::F32
      buffer
    end

    def alloc_weight(name : String, count : Int32, dtype : WeightDType) : ML::MetalBuffer
      raise ArgumentError.new("Duplicate buffer #{name}") if allocated?(name)
      raise ArgumentError.new("Buffer count must be positive") unless count > 0
      buffer = ML::MetalBuffer.new(count.to_i64 * dtype.bytes_per_element)
      @buffers[name] = buffer
      @dtypes[name] = dtype
      @typed_buffers[name] = buffer
      buffer
    end

    # Transfer a no-copy stage view to this Arena as an F32 activation or
    # scratch buffer. The view remains alive until Arena.release/remove.
    def adopt(name : String, view : MetalStageBufferView) : ML::MetalBuffer
      adopt_stage_view(name, view, WeightDType::F32, false)
      view.buffer
    end

    # Weights in the shared stage pool require an explicit storage dtype so
    # kernels never infer BF16 from byte counts or a buffer's public map entry.
    def adopt_weight(name : String, view : MetalStageBufferView,
                     dtype : WeightDType) : ML::MetalBuffer
      unless view.requested_bytes % dtype.bytes_per_element == 0
        raise ArgumentError.new("Stage weight byte size is not aligned to its dtype")
      end
      adopt_stage_view(name, view, dtype, true)
      view.buffer
    end

    def upload(name : String, values : Array(Float32)) : ML::MetalBuffer
      raise ArgumentError.new("Duplicate buffer #{name}") if allocated?(name)
      raise ArgumentError.new("Buffer count must be positive") if values.empty?
      # Uploads are setup constants, not phase-resident stage data.
      buffer = ML::MetalBuffer.new(values.size.to_i64 * sizeof(Float32))
      @buffers[name] = buffer
      @dtypes[name] = WeightDType::F32
      buffer.write(values)
      buffer
    end

    def [](name : String) : ML::MetalBuffer
      @buffers[name]
    end

    def dtype(name : String) : WeightDType
      buffer = @buffers[name]
      dtype = @dtypes[name]? || WeightDType::F32
      if typed_buffer = @typed_buffers[name]?
        unless typed_buffer.same?(buffer)
          raise ArgumentError.new("Resident video weight buffer identity changed")
        end
      end
      dtype
    end

    def check_weight(name : String, count : Int64) : Nil
      MetalOps.check_storage(@buffers[name], count, dtype(name))
    end

    def remove(name : String) : Nil
      buffer = @buffers.delete(name)
      typed_buffer = @typed_buffers.delete(name)
      @dtypes.delete(name)
      @stage_names.delete(name)
      stage_owned = false
      if buffer
        stage_owned ||= release_stage_view_if_unreferenced(buffer)
      end
      if typed_buffer
        stage_owned ||= release_stage_view_if_unreferenced(typed_buffer)
      end
      unless stage_owned
        if buffer && !named_buffer_references?(buffer)
          buffer.release
        end
        if typed_buffer && !(buffer && typed_buffer.same?(buffer)) &&
           !named_buffer_references?(typed_buffer)
          typed_buffer.release
        end
      end
    end

    def release : Nil
      @buffers.each do |_, buffer|
        next if pooled_buffer?(buffer)
        buffer.release
      end
      @typed_buffers.each do |_, buffer|
        next if pooled_buffer?(buffer)
        buffer.release
      end
      @stage_views.each_value { |view| view.release_from_arena!(object_id) }
      @stage_views.clear
      @stage_names.clear
      @buffers.clear
      @dtypes.clear
      @typed_buffers.clear
    end

    private def adopt_stage_view(name : String, view : MetalStageBufferView,
                                 dtype : WeightDType, typed : Bool) : Nil
      raise ArgumentError.new("Duplicate buffer #{name}") if allocated?(name)
      raise ArgumentError.new("Invalid or released stage buffer view") unless view.buffer.valid?
      unless view.requested_bytes >= dtype.bytes_per_element &&
             view.requested_bytes % dtype.bytes_per_element == 0
        raise ArgumentError.new("Stage view size is incompatible with its dtype")
      end
      view.claim_for_arena!(object_id)
      @buffers[name] = view.buffer
      @dtypes[name] = dtype
      @typed_buffers[name] = view.buffer if typed
      @stage_views[view.lease_id] = view
      @stage_names[name] = view.lease_id
    end

    private def pooled_buffer?(buffer : ML::MetalBuffer) : Bool
      @stage_views.values.any? { |view| view.buffer.same?(buffer) }
    end

    private def named_buffer_references?(buffer : ML::MetalBuffer) : Bool
      @buffers.values.any? { |other| other.same?(buffer) } ||
        @typed_buffers.values.any? { |other| other.same?(buffer) }
    end

    private def release_stage_view_if_unreferenced(buffer : ML::MetalBuffer) : Bool
      view = @stage_views.values.find { |candidate| candidate.buffer.same?(buffer) }
      return false unless view
      still_referenced = @buffers.values.any? { |other| other.same?(buffer) } ||
                         @typed_buffers.values.any? { |other| other.same?(buffer) }
      return true if still_referenced
      @stage_views.delete(view.lease_id)
      @stage_names.reject! { |_, lease_id| lease_id == view.lease_id }
      view.release_from_arena!(object_id)
      true
    end
  end

  # Independent F32 correctness primitives for opt-in video block admission.
  # All encode methods only bind/dispatch into an existing SERIAL encoder.
  # This deliberately does not promise production-size GEMM/attention speed.
  module MetalOps
    SOURCE = {{ read_file("#{__DIR__}/kernels/common.metal") }}

    def self.check(buffer : ML::MetalBuffer, count : Int64) : Nil
      unless count > 0 && count <= Int32::MAX
        raise ArgumentError.new("Invalid or undersized resident video buffer")
      end
      check_storage(buffer, count, WeightDType::F32)
    end

    def self.check_storage(buffer : ML::MetalBuffer, count : Int64, dtype : WeightDType) : Nil
      unless buffer.valid? && count > 0 && count <= Int32::MAX &&
             count <= buffer.size // dtype.bytes_per_element
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
      {"video_linear_f32", "video_linear_mixed", "video_norm_f32", "video_norm_mixed",
       "video_modulate_f32",
       "video_add_f32", "video_gelu_f32", "video_copy_f32",
       "video_attention_f32"}.each { |name| pipeline(name) }
    end

    def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
    end

    def self.linear(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                    w : ML::MetalBuffer, bias : ML::MetalBuffer, y : ML::MetalBuffer,
                    rows : Int32, in_dim : Int32, out_dim : Int32,
                    weight_dtype : WeightDType = WeightDType::F32,
                    bias_dtype : WeightDType = WeightDType::F32) : Nil
      dims(rows, in_dim, out_dim)
      check(x, rows.to_i64 * in_dim)
      check_storage(w, out_dim.to_i64 * in_dim, weight_dtype)
      check_storage(bias, out_dim.to_i64, bias_dtype)
      check(y, rows.to_i64 * out_dim)
      distinct(y, x, w, bias)
      if weight_dtype.f32? && bias_dtype.f32?
        # Preserve the established F32 path for existing users.
        e.set_pipeline(pipeline("video_linear_f32"))
        e.set_buffer(x, 0).set_buffer(w, 1).set_buffer(bias, 2)
        e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
        e.set_value(rows.to_u32, 4).set_value(in_dim.to_u32, 5).set_value(out_dim.to_u32, 6)
      else
        e.set_pipeline(pipeline("video_linear_mixed"))
        e.set_buffer(x, 0).set_buffer(w, 1).set_buffer(bias, 2)
        e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
        e.set_value(rows.to_u32, 4).set_value(in_dim.to_u32, 5).set_value(out_dim.to_u32, 6)
        e.set_buffer(w, 7).set_buffer(bias, 8)
        e.set_value(weight_dtype.bf16? ? 1_u32 : 0_u32, 9)
        e.set_value(bias_dtype.bf16? ? 1_u32 : 0_u32, 10)
      end
      e.dispatch_1d(rows * out_dim)
    end

    def self.norm(e : ML::Metal::ComputeEncoder, x : ML::MetalBuffer,
                  weight : ML::MetalBuffer, bias : ML::MetalBuffer, y : ML::MetalBuffer,
                  rows : Int32, dim : Int32, eps : Float32, rms : Bool = false,
                  affine : Bool = false, weight_dtype : WeightDType = WeightDType::F32,
                  bias_dtype : WeightDType = WeightDType::F32) : Nil
      dims(rows, dim)
      raise ArgumentError.new("Invalid norm epsilon") unless eps.finite? && eps > 0
      check(x, rows.to_i64 * dim)
      check(y, rows.to_i64 * dim)
      inplace_or_distinct(x, y)
      if affine
        check_storage(weight, dim.to_i64, weight_dtype)
        check_storage(bias, dim.to_i64, bias_dtype)
        distinct(y, weight, bias)
      end
      mixed_affine = affine && (!weight_dtype.f32? || !bias_dtype.f32?)
      e.set_pipeline(pipeline(mixed_affine ? "video_norm_mixed" : "video_norm_f32"))
      # Non-affine kernels never read these slots. Bind a known-valid input,
      # rather than accepting stale handles in nominally unused arguments.
      e.set_buffer(x, 0).set_buffer(affine ? weight : x, 1).set_buffer(affine ? bias : x, 2)
      e.set_buffer(y, 3, ML::Metal::BufferAccess::Write)
      e.set_value(dim.to_u32, 4).set_value(eps, 5)
      e.set_value(rms ? 1_u32 : 0_u32, 6).set_value(affine ? 1_u32 : 0_u32, 7)
      if mixed_affine
        e.set_buffer(weight, 8).set_buffer(bias, 9)
        e.set_value(weight_dtype.bf16? ? 1_u32 : 0_u32, 10)
        e.set_value(bias_dtype.bf16? ? 1_u32 : 0_u32, 11)
      end
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
