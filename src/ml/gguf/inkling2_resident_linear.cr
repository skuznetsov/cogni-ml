require "./inkling2_weights"
require "./inkling2_metal_view"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  require "../metal/device"
  require "../metal/dispatch"

  lib Inkling2ResidentSystem
    fun getpagesize : Int32
  end
{% end %}

module ML::GGUF
  {% if flag?(:darwin) && !flag?(:cpu_only) %}
    # Encode-only projection and MoE helpers for an inference-wide Metal
    # command buffer. The GGUF mmap remains owned by its GGUFFile/Weights.
    class Inkling2ResidentLinear
      MAX_GROUPED_ROWS = 512
      SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_q56k.metal") }} + "\n" + {{ read_file("#{__DIR__}/kernels/inkling2_resident_moe.metal") }}

      private MAX_ROUTE_SCRATCH_BYTES = 16_384_i32
      private MAX_ROWS_PER_GRID = 4_096_i32
      private MAX_LINEAR_ELEMENTS_PER_GRID = 4_194_304_i32
      private BATCH_TILE = 4_i32
      {% if flag?(:aarch64) %}
        private Q8_MATRIX_TILE = 32_i32
        private Q8_MATRIX_MIN_ROWS = 16_i32
        private Q8_MATRIX_SOURCE = SOURCE + "\n" + {{ read_file("#{__DIR__}/kernels/inkling2_resident_q8_gemm.metal") }}
      {% end %}

      private GROUPED_EXPERT_SOURCE = SOURCE + "\n" + {{ read_file("#{__DIR__}/kernels/inkling2_resident_moe_grouped.metal") }}
      private EXPERT_TILE_ROUTES = 16_i32
      private EXPERT_TILE_OUTPUTS = 64_i32

      private record ExpertGrouping, routes : Inkling2View, tiles : Inkling2View, max_tiles : Int32

      @base : UInt64
      @file_size : UInt64
      @model : ML::MetalBuffer
      @closed : Bool = false

      def initialize(gguf : GGUFFile)
        raise "Metal is unavailable. Use --cpu for CPU inference." unless ML::Metal::Device.init!

        base, @file_size = gguf.mmap_region || raise ArgumentError.new("Inkling 2 resident Metal needs a memory-mapped GGUF")
        @base = base.address.to_u64
        page = Inkling2ResidentSystem.getpagesize.to_i64
        length = ((@file_size.to_i64 + page - 1) // page) * page
        @model = ML::MetalBuffer.wrap_no_copy(base.as(Void*), length)
      end

      # Return a zero-copy view into the GGUF-backed Metal buffer.
      def view(weight : QuantWeight) : Inkling2View
        ensure_open!
        Inkling2View.new(@model, model_offset(weight.raw))
      end

      # Current linear kernels admit the checkpoint's F32 and Q8_0 matrices.
      # Q8_0 input width must contain whole 32-value blocks.
      def self.projection_supported?(weight : QuantWeight) : Bool
        return false unless weight.in_dim > 0 && weight.out_dim > 0

        row_bytes = case weight.type
                    when .f32?
                      weight.in_dim.to_i64 * sizeof(Float32).to_i64
                    when .q8_0?
                      return false unless weight.in_dim % 32 == 0
                      (weight.in_dim // 32).to_i64 * 34_i64
                    else
                      return false
                    end

        return false if row_bytes > Int64::MAX // weight.out_dim.to_i64
        expected = row_bytes * weight.out_dim.to_i64
        expected == weight.raw.size.to_i64
      end

      # Static capability guard for the resident MoE path, including all
      # dimensions and raw extents that its GPU indexing assumes.
      def self.moe_supported?(layer : Inkling2LayerWeights, config : Inkling2Config) : Bool
        return false unless projection_supported?(layer.router)
        return false unless layer.router.out_dim == config.n_expert
        return false unless config.n_expert > 0 && config.n_expert_used > 0 && config.n_expert_used <= config.n_expert
        return false unless config.n_embd > 0 && config.expert_dim > 0
        return false if (layer.router_bias && layer.router_bias.not_nil!.size != config.n_expert)
        return false if config.n_expert.to_i64 * 2_i64 * sizeof(Float32).to_i64 > MAX_ROUTE_SCRATCH_BYTES.to_i64
        return false unless expert_supported?(layer.gate_up, config.n_expert)
        return false unless expert_supported?(layer.down, config.n_expert)
        return false unless layer.gate_up.in_dim == config.n_embd
        return false unless layer.gate_up.out_dim.to_i64 == config.expert_dim.to_i64 * 2_i64
        return false unless layer.down.in_dim == config.expert_dim
        return false unless layer.down.out_dim == config.n_embd

        true
      end

      # Encode a row-major F32 activation matrix `[rows, in_dim]` into an
      # output matrix `[rows, out_dim]`. This method never submits or waits.
      def project(encoder : ML::Metal::ComputeEncoder, weight : QuantWeight,
                  input : Inkling2View, output : Inkling2View, rows : Int32) : Nil
        ensure_open!
        raise ArgumentError.new("Inkling 2 resident projection has no rows") unless rows > 0
        raise ArgumentError.new("Unsupported Inkling 2 resident projection #{weight.type}") unless self.class.projection_supported?(weight)
        ensure_view!(input, checked_bytes(rows, weight.in_dim, "projection input"))
        ensure_view!(output, checked_bytes(rows, weight.out_dim, "projection output"))

        row_base = 0_i32
        while row_base < rows
          count = Math.min(MAX_ROWS_PER_GRID, rows - row_base)
          input_offset = input.offset + row_base.to_i64 * weight.in_dim.to_i64 * sizeof(Float32)
          output_offset = output.offset + row_base.to_i64 * weight.out_dim.to_i64 * sizeof(Float32)
          encode_projection(encoder, weight, model_offset(weight.raw), input.buffer, input_offset,
            output.buffer, output_offset, count)
          row_base += count
        end
      end

      # Encode router, expert up/gate, SiTU activation, expert down, and
      # ordered top-k combine for every row. All temporary and output views
      # live in the caller's resident scratch arena until its rewind.
      def moe(encoder : ML::Metal::ComputeEncoder, scratch : Inkling2ResidentScratch,
              layer : Inkling2LayerWeights, config : Inkling2Config,
              input : Inkling2View, bias : Inkling2View?, rows : Int32) : Inkling2View
        ensure_open!
        raise ArgumentError.new("Inkling 2 resident MoE has no rows") unless rows > 0
        raise ArgumentError.new("Unsupported Inkling 2 resident MoE weights or dimensions") unless self.class.moe_supported?(layer, config)
        ensure_view!(input, checked_bytes(rows, config.n_embd, "MoE input"))
        if router_bias = bias
          ensure_view!(router_bias, checked_bytes(1_i32, config.n_expert, "MoE router bias"))
        elsif layer.router_bias
          raise ArgumentError.new("Inkling 2 resident MoE is missing the router bias view")
        end

        routed_rows = checked_product(rows.to_i64, config.n_expert_used.to_i64, "routed row count")
        raise ArgumentError.new("Inkling 2 resident MoE routed row count exceeds Int32") if routed_rows > Int32::MAX.to_i64
        logits = scratch.take(checked_elements(rows, config.n_expert, "router logits"))
        ids = scratch.take(checked_elements(rows, config.n_expert_used, "selected experts"))
        routing_weights = scratch.take(checked_elements(rows, config.n_expert_used, "routing weights"))
        gated = scratch.take(checked_elements(routed_rows.to_i32, config.expert_dim, "expert activation"))
        down = scratch.take(checked_elements(routed_rows.to_i32, config.n_embd, "expert down"))
        output = scratch.take(checked_elements(rows, config.n_embd, "MoE output"))

        project(encoder, layer.router, input, logits, rows)
        encode_route(encoder, logits, bias || logits, bias != nil, ids, routing_weights,
          config.n_expert, config.n_expert_used, rows, config.expert_scale)
        grouping = if grouped_experts?(rows, config.n_expert)
                     group_expert_routes(encoder, scratch, ids, config.n_expert, rows, routed_rows.to_i32)
                   end
        if grouping
          encode_grouped_experts(encoder, layer.gate_up, input, gated, grouping, config.n_expert_used,
            situ: {config.situ_beta, config.situ_linear_beta})
        else
          gate_up = scratch.take(checked_elements(routed_rows.to_i32, layer.gate_up.out_dim, "expert gate/up"))
          encode_experts(encoder, layer.gate_up, input, ids, gate_up,
            config.n_expert_used, routed_rows.to_i32, grouping, input_is_routed: false)
          encode_situ(encoder, gate_up, gated, config.expert_dim, routed_rows.to_i32,
            config.situ_beta, config.situ_linear_beta)
        end
        encode_experts(encoder, layer.down, gated, ids, down,
          config.n_expert_used, routed_rows.to_i32, grouping, input_is_routed: true)
        encode_combine(encoder, down, routing_weights, output,
          config.n_embd, config.n_expert_used, rows)
        output
      end

      def close : Nil
        return if @closed
        @closed = true
        @model.release
      end

      private def encode_projection(encoder : ML::Metal::ComputeEncoder, weight : QuantWeight,
                                    weight_offset : Int64, input : ML::MetalBuffer, input_offset : Int64,
                                    output : ML::MetalBuffer, output_offset : Int64, rows : Int32) : Nil
        if rows == 1
          name = weight.type.f32? ? "simd_mv_f32_f32" : "simd_mv_q8_0_f32"
          encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(name, SOURCE))
          encoder.set_buffer(@model, 0, offset: weight_offset)
          encoder.set_buffer(input, 1, offset: input_offset)
          encoder.set_buffer(output, 2, ML::Metal::BufferAccess::Write, offset: output_offset)
          encoder.set_value(weight.in_dim.to_u32, 3)
          encoder.set_value(weight.out_dim.to_u32, 4)
          encoder.set_value(1_u32, 5)
          encoder.dispatch_threadgroups({(weight.out_dim + 3) // 4, 1, 1}, {128, 1, 1})
          return
        end

        name = weight.type.f32? ? "inkling2_resident_gemm_f32" : "inkling2_resident_gemm_q8_0"
        source = SOURCE
        tile = BATCH_TILE
        {% if flag?(:aarch64) %}
          if weight.type.q8_0? && rows >= Q8_MATRIX_MIN_ROWS
            name = "inkling2_resident_gemm_q8_0_simd"
            source = Q8_MATRIX_SOURCE
            tile = Q8_MATRIX_TILE
          end
        {% end %}
        encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(name, source))
        encoder.set_buffer(@model, 0, offset: weight_offset)
        encoder.set_buffer(input, 1, offset: input_offset)
        encoder.set_buffer(output, 2, ML::Metal::BufferAccess::Write, offset: output_offset)
        encoder.set_value(weight.in_dim.to_u32, 3)
        encoder.set_value(weight.out_dim.to_u32, 4)
        encoder.set_value(rows.to_u32, 5)
        encoder.dispatch_threadgroups(
          {(weight.out_dim + tile - 1) // tile, (rows + tile - 1) // tile, 1},
          {128, 1, 1},
        )
      end

      private def encode_route(encoder : ML::Metal::ComputeEncoder,
                               logits : Inkling2View, bias : Inkling2View,
                               has_bias : Bool,
                               ids : Inkling2View, weights : Inkling2View,
                               n_experts : Int32, k : Int32, rows : Int32,
                               scale : Float32) : Nil
        pipeline = ML::Metal::PipelineCache.get_or_compile("inkling2_resident_moe_route", SOURCE)
        scratch_bytes = n_experts.to_i64 * 2_i64 * sizeof(Float32).to_i64
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(logits.buffer, 0, offset: logits.offset)
        encoder.set_buffer(bias.buffer, 1, offset: bias.offset)
        encoder.set_buffer(ids.buffer, 2, ML::Metal::BufferAccess::Write, offset: ids.offset)
        encoder.set_buffer(weights.buffer, 3, ML::Metal::BufferAccess::Write, offset: weights.offset)
        encoder.set_value(n_experts.to_u32, 4)
        encoder.set_value(k.to_u32, 5)
        encoder.set_value(scale, 6)
        encoder.set_value(has_bias ? 1_u32 : 0_u32, 7)
        encoder.set_value(rows.to_u32, 8)
        encoder.set_threadgroup_memory(scratch_bytes.to_i32, 0)

        row_base = 0_i32
        while row_base < rows
          count = Math.min(MAX_ROWS_PER_GRID, rows - row_base)
          encoder.set_value(row_base.to_u32, 9)
          encoder.dispatch_threadgroups({1, count, 1}, {32, 1, 1})
          row_base += count
        end
      end

      private def encode_experts(encoder : ML::Metal::ComputeEncoder,
                                 weight : Inkling2ExpertWeights,
                                 input : Inkling2View, ids : Inkling2View,
                                 output : Inkling2View, top_k : Int32,
                                 routed_rows : Int32, grouping : ExpertGrouping?,
                                 *, input_is_routed : Bool) : Nil
        if grouping
          encode_grouped_experts(encoder, weight, input, output, grouping, input_is_routed ? 1_i32 : top_k)
          return
        end

        name, threads, output_tile = expert_kernel(weight.type)
        pipeline = ML::Metal::PipelineCache.get_or_compile(name, SOURCE)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(@model, 0, offset: model_offset(weight.raw))
        encoder.set_buffer(input.buffer, 1, offset: input.offset)
        encoder.set_buffer(ids.buffer, 2, offset: ids.offset)
        encoder.set_buffer(output.buffer, 3, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(weight.in_dim.to_u32, 4)
        encoder.set_value(weight.out_dim.to_u32, 5)
        encoder.set_value(weight.expert_bytes.to_u32, 6)
        encoder.set_value(weight.in_dim.to_u32, 7)
        encoder.set_value(top_k.to_u32, 8)
        encoder.set_value(input_is_routed ? 1_u32 : 0_u32, 9)

        routed_base = 0_i32
        while routed_base < routed_rows
          count = Math.min(MAX_ROWS_PER_GRID, routed_rows - routed_base)
          encoder.set_value(routed_base.to_u32, 10)
          encoder.dispatch_threadgroups({(weight.out_dim + output_tile - 1) // output_tile, count, 1}, {threads, 1, 1})
          routed_base += count
        end
      end

      private def grouped_experts?(rows : Int32, n_experts : Int32) : Bool
        {% if flag?(:aarch64) %}
          16 <= rows <= MAX_GROUPED_ROWS && n_experts <= 256
        {% else %}
          false
        {% end %}
      end

      private def group_expert_routes(encoder : ML::Metal::ComputeEncoder, scratch : Inkling2ResidentScratch,
                                      ids : Inkling2View, n_experts : Int32, rows : Int32,
                                      routed_rows : Int32) : ExpertGrouping
        routes = scratch.take(n_experts * rows)
        # At most one partial tile per expert; every other tile is full.
        max_tiles = Math.min(routed_rows, n_experts + (routed_rows - 1) // EXPERT_TILE_ROUTES)
        tiles = scratch.take(1 + 3 * max_tiles)

        name = "inkling2_resident_moe_group_routes"
        encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(name, GROUPED_EXPERT_SOURCE))
        encoder.set_buffer(ids.buffer, 0, offset: ids.offset)
        encoder.set_buffer(routes.buffer, 1, ML::Metal::BufferAccess::Write, offset: routes.offset)
        encoder.set_buffer(tiles.buffer, 2, ML::Metal::BufferAccess::Write, offset: tiles.offset)
        encoder.set_value(rows.to_u32, 3)
        encoder.set_value(routed_rows.to_u32, 4)
        encoder.set_value(n_experts.to_u32, 5)
        encoder.set_threadgroup_memory(n_experts * sizeof(UInt32), 0)
        encoder.dispatch_threadgroups({1, 1, 1}, {((n_experts + 31) // 32) * 32, 1, 1})
        ExpertGrouping.new(routes, tiles, max_tiles)
      end

      private def encode_grouped_experts(encoder : ML::Metal::ComputeEncoder, weight : Inkling2ExpertWeights,
                                         input : Inkling2View, output : Inkling2View, grouping : ExpertGrouping,
                                         input_route_divisor : Int32, *, situ : {Float32, Float32}? = nil) : Nil
        name = if situ
                 weight.type.iq4_nl? ? "inkling2_resident_moe_gemm_situ_iq4_nl" : "inkling2_resident_moe_gemm_situ_q8_0"
               else
                 weight.type.iq4_nl? ? "inkling2_resident_moe_gemm_iq4_nl" : "inkling2_resident_moe_gemm_q8_0"
               end
        encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(name, GROUPED_EXPERT_SOURCE))
        encoder.set_buffer(@model, 0, offset: model_offset(weight.raw))
        encoder.set_buffer(input.buffer, 1, offset: input.offset)
        encoder.set_buffer(grouping.routes.buffer, 2, offset: grouping.routes.offset)
        encoder.set_buffer(grouping.tiles.buffer, 3, offset: grouping.tiles.offset)
        encoder.set_buffer(output.buffer, 4, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(weight.in_dim.to_u32, 5)
        encoder.set_value(weight.out_dim.to_u32, 6)
        encoder.set_value(weight.expert_bytes.to_u32, 7)
        encoder.set_value(input_route_divisor.to_u32, 8)
        if situ
          encoder.set_value(situ[0], 9)
          encoder.set_value(situ[1], 10)
        end
        output_tiles = (weight.out_dim - 1) // EXPERT_TILE_OUTPUTS + 1
        encoder.dispatch_threadgroups({output_tiles, grouping.max_tiles, 1}, {128, 1, 1})
      end

      private def encode_situ(encoder : ML::Metal::ComputeEncoder,
                              gate_up : Inkling2View, gated : Inkling2View,
                              width : Int32, routed_rows : Int32,
                              beta : Float32, linear_beta : Float32) : Nil
        total = checked_product(routed_rows.to_i64, width.to_i64, "SiTU elements")
        raise ArgumentError.new("Inkling 2 resident SiTU element count exceeds UInt32") if total > UInt32::MAX.to_i64
        pipeline = ML::Metal::PipelineCache.get_or_compile("inkling2_resident_moe_situ", SOURCE)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(gate_up.buffer, 0, offset: gate_up.offset)
        encoder.set_buffer(gated.buffer, 1, ML::Metal::BufferAccess::Write, offset: gated.offset)
        encoder.set_value(width.to_u32, 2)
        encoder.set_value(total.to_u32, 3)
        encoder.set_value(beta, 5)
        encoder.set_value(linear_beta, 6)

        element_base = 0_i64
        while element_base < total
          count = Math.min(MAX_LINEAR_ELEMENTS_PER_GRID.to_i64, total - element_base)
          encoder.set_value(element_base.to_u32, 4)
          encoder.dispatch_1d(count.to_i32)
          element_base += count
        end
      end

      private def encode_combine(encoder : ML::Metal::ComputeEncoder,
                                 values : Inkling2View, weights : Inkling2View,
                                 output : Inkling2View, dim : Int32, k : Int32,
                                 rows : Int32) : Nil
        total = checked_product(rows.to_i64, dim.to_i64, "MoE combine elements")
        raise ArgumentError.new("Inkling 2 resident MoE output exceeds UInt32 elements") if total > UInt32::MAX.to_i64
        pipeline = ML::Metal::PipelineCache.get_or_compile("inkling2_resident_moe_combine", SOURCE)
        encoder.set_pipeline(pipeline)
        encoder.set_buffer(values.buffer, 0, offset: values.offset)
        encoder.set_buffer(weights.buffer, 1, offset: weights.offset)
        encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(dim.to_u32, 3)
        encoder.set_value(k.to_u32, 4)
        encoder.set_value(total.to_u32, 5)

        element_base = 0_i64
        while element_base < total
          count = Math.min(MAX_LINEAR_ELEMENTS_PER_GRID.to_i64, total - element_base)
          encoder.set_value(element_base.to_u32, 6)
          encoder.dispatch_1d(count.to_i32)
          element_base += count
        end
      end

      private def expert_kernel(type : TensorType) : {String, Int32, Int32}
        case type
        when .iq4_nl? then {"inkling2_resident_moe_matvec_iq4_nl", 64, 4}
        when .q8_0?   then {"inkling2_resident_moe_matvec_q8_0", 128, 4}
        else
          raise ArgumentError.new("Unsupported Inkling 2 resident expert type #{type}")
        end
      end

      private def self.expert_supported?(weight : Inkling2ExpertWeights, n_experts : Int32) : Bool
        return false unless weight.in_dim > 0 && weight.out_dim > 0 && weight.n_expert == n_experts
        return false unless weight.in_dim % 32 == 0

        bytes_per_row = (weight.in_dim // 32).to_i64 * case weight.type
                                                    when .iq4_nl? then 18_i64
                                                    when .q8_0?   then 34_i64
                                                    else               return false
                                                    end
        return false if bytes_per_row > Int64::MAX // weight.out_dim.to_i64
        bytes_per_expert = bytes_per_row * weight.out_dim.to_i64
        return false if bytes_per_expert > Int64::MAX // n_experts.to_i64
        weight.expert_bytes.to_i64 == bytes_per_expert &&
          weight.raw.size.to_i64 == bytes_per_expert * n_experts.to_i64
      end

      private def ensure_view!(view : Inkling2View, bytes : Int64) : Nil
        unless view.offset >= 0 && bytes >= 0 && view.offset <= view.buffer.size && bytes <= view.buffer.size - view.offset
          raise ArgumentError.new("Inkling 2 resident view exceeds its Metal buffer")
        end
      end

      private def checked_bytes(rows : Int32, width : Int32, label : String) : Int64
        raise ArgumentError.new("Inkling 2 resident #{label} dimensions must be positive") unless rows > 0 && width > 0
        elements = checked_product(rows.to_i64, width.to_i64, label)
        raise ArgumentError.new("Inkling 2 resident #{label} exceeds UInt32 elements") if elements > UInt32::MAX.to_i64
        checked_product(elements, sizeof(Float32).to_i64, label)
      end

      private def checked_elements(rows : Int32, width : Int32, label : String) : Int
        elements = checked_product(rows.to_i64, width.to_i64, label)
        raise ArgumentError.new("Inkling 2 resident #{label} exceeds UInt32 elements") if elements > UInt32::MAX.to_i64
        elements.to_i
      end

      private def checked_product(a : Int64, b : Int64, label : String) : Int64
        raise ArgumentError.new("Inkling 2 resident #{label} overflows") if a < 0 || b < 0 || (b > 0 && a > Int64::MAX // b)
        a * b
      end

      private def model_offset(raw : Bytes) : Int64
        address = raw.to_unsafe.address.to_u64
        size = raw.size.to_u64
        if address < @base || size > @file_size || address - @base > @file_size - size
          raise ArgumentError.new("Inkling 2 resident tensor is outside the mapped GGUF")
        end
        (address - @base).to_i64
      end

      private def ensure_open! : Nil
        raise "Inkling 2 resident linear backend is closed" if @closed
      end
    end
  {% end %}
end
