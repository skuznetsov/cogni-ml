require "./scratch"
require "../../../metal/device"
require "../../../metal/dispatch"
require "../../../core/buffer"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::GGUF
    # BF16 checkpoint-weight to F32 activation projection. The matrix is
    # row-major [out_dim, in_dim], and the activation/output matrices are
    # row-major [rows, in_dim] and [rows, out_dim].
    module Inkling2DSparkDense
      private ROW_TILE = 8_i32
      private THREADS  = 256_i32
      private SOURCE   = {{ read_file("#{__DIR__}/../../../metal/kernels/inkling2/inkling2_dspark_dense.metal") }}

      def self.project(
        encoder : ML::Metal::ComputeEncoder,
        weight : Inkling2View,
        input : Inkling2View,
        output : Inkling2View,
        rows : Int32,
        in_dim : Int32,
        out_dim : Int32,
        *, tiled : Bool = false,
      ) : Nil
        raise ArgumentError.new("Inkling 2 DSpark dense rows must be positive") unless rows > 0
        raise ArgumentError.new("Inkling 2 DSpark dense input width must be positive") unless in_dim > 0
        raise ArgumentError.new("Inkling 2 DSpark dense output width must be positive") unless out_dim > 0

        input_elements = checked_elements(rows, in_dim, "input")
        weight_elements = checked_elements(out_dim, in_dim, "weight")
        output_elements = checked_elements(rows, out_dim, "output")
        input_bytes = checked_bytes(input_elements, sizeof(Float32).to_i64, "input")
        weight_bytes = checked_bytes(weight_elements, 2_i64, "weight")
        output_bytes = checked_bytes(output_elements, sizeof(Float32).to_i64, "output")

        validate_extent!(weight, weight_bytes, "BF16 weight")
        validate_extent!(input, input_bytes, "F32 input")
        validate_extent!(output, output_bytes, "F32 output")
        validate_alignment!(weight, 16_i64, "BF16 weight")
        validate_alignment!(input, 4_i64, "F32 input")
        validate_alignment!(output, 4_i64, "F32 output")
        reject_overlap!(weight, weight_bytes, input, input_bytes, "weight", "input")
        reject_overlap!(weight, weight_bytes, output, output_bytes, "weight", "output")
        reject_overlap!(input, input_bytes, output, output_bytes, "input", "output")

        full_tile = rows == ROW_TILE
        name = if full_tile
                 "inkling2_dspark_dense_rows8"
               elsif tiled && rows > ROW_TILE
                 "inkling2_dspark_dense_tiled"
               else
                 "inkling2_dspark_dense"
               end
        encoder.set_pipeline(pipeline(name))
        encoder.set_buffer(weight.buffer, 0, offset: weight.offset)
        encoder.set_buffer(input.buffer, 1, offset: input.offset)
        encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(in_dim.to_u32, 3)
        encoder.set_value(out_dim.to_u32, 4)
        encoder.set_value(rows.to_u32, 5) unless full_tile
        encoder.dispatch_threadgroups(
          {
            ((out_dim.to_i64 + ROW_TILE - 1) // ROW_TILE).to_i32,
            ((rows.to_i64 + ROW_TILE - 1) // ROW_TILE).to_i32,
            1,
          },
          {THREADS, 1, 1},
        )
      end

      private def self.pipeline(name : String) : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get_or_compile(name, SOURCE)
      end

      private def self.checked_elements(a : Int32, b : Int32, label : String) : Int64
        product = a.to_i64 * b.to_i64
        raise ArgumentError.new("Inkling 2 DSpark dense #{label} element count overflows") if product <= 0
        product
      end

      private def self.checked_bytes(elements : Int64, bytes_per_element : Int64, label : String) : Int64
        raise ArgumentError.new("Inkling 2 DSpark dense #{label} byte extent overflows") if elements > Int64::MAX // bytes_per_element
        elements * bytes_per_element
      end

      private def self.validate_extent!(view : Inkling2View, bytes : Int64, label : String) : Nil
        unless view.offset >= 0 && view.offset <= view.buffer.size && bytes <= view.buffer.size - view.offset
          raise ArgumentError.new("Inkling 2 DSpark dense #{label} view is smaller than its matrix extent")
        end
      end

      private def self.validate_alignment!(view : Inkling2View, alignment : Int64, label : String) : Nil
        # MTLBuffer's base address meets the Metal resource alignment; inspect
        # only the slice offset so private scratch buffers remain supported.
        raise ArgumentError.new("Inkling 2 DSpark dense #{label} view is not #{alignment}-byte aligned") unless view.offset % alignment == 0
      end

      private def self.reject_overlap!(
        left : Inkling2View,
        left_bytes : Int64,
        right : Inkling2View,
        right_bytes : Int64,
        left_name : String,
        right_name : String,
      ) : Nil
        return unless left.buffer.handle == right.buffer.handle

        left_end = left.offset + left_bytes
        right_end = right.offset + right_bytes
        if left.offset < right_end && right.offset < left_end
          raise ArgumentError.new("Inkling 2 DSpark dense #{left_name} and #{right_name} views overlap")
        end
      end
    end
  end
{% end %}
