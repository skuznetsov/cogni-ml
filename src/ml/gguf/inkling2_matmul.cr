require "./compute"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  require "../metal/device"
  require "../metal/dispatch"

  lib Inkling2System
    fun getpagesize : Int32
  end
{% end %}

module ML::GGUF
  abstract class Inkling2Matmul
    abstract def multiply(weight : QuantWeight, input : Array(Float32)) : Array(Float32)

    def multiply_many(jobs : Array({QuantWeight, Array(Float32)})) : Array(Array(Float32))
      jobs.map { |weight, input| multiply(weight, input) }
    end

    def close : Nil
    end
  end

  class Inkling2CPUMatmul < Inkling2Matmul
    def multiply(weight : QuantWeight, input : Array(Float32)) : Array(Float32)
      raise ArgumentError.new("Inkling 2 projection input width mismatch") unless input.size == weight.in_dim

      QuantMatmul.matmul_add(input, 1, weight.in_dim, weight.raw, weight.type,
        weight.out_dim, Array(Float32).new(weight.out_dim, 0_f32))
    end
  end

  {% if flag?(:darwin) && !flag?(:cpu_only) %}
    class Inkling2MetalMatmul < Inkling2Matmul
      alias Job = {QuantWeight, Array(Float32)}

      SOURCE = {{ read_file("#{__DIR__}/kernels/gemm_q56k.metal") }}

      @base : UInt64
      @file_size : UInt64
      @model : ML::MetalBuffer
      @input : ML::MetalBuffer? = nil
      @output : ML::MetalBuffer? = nil

      def initialize(gguf : GGUFFile)
        raise "Metal is unavailable. Use --cpu for CPU inference." unless ML::Metal::Device.init!

        base, @file_size = gguf.mmap_region || raise "Inkling 2 Metal needs a memory-mapped GGUF"
        @base = base.address.to_u64
        page = Inkling2System.getpagesize.to_i64
        length = ((@file_size.to_i64 + page - 1) // page) * page
        @model = ML::MetalBuffer.wrap_no_copy(base.as(Void*), length)
      end

      def multiply(weight : QuantWeight, input : Array(Float32)) : Array(Float32)
        multiply_many([{weight, input}]).first
      end

      def multiply_many(jobs : Array(Job)) : Array(Array(Float32))
        return [] of Array(Float32) if jobs.empty?

        input_size = 0_i64
        output_size = 0_i64
        offsets = jobs.map do |weight, input|
          raise ArgumentError.new("Inkling 2 projection input width mismatch") unless input.size == weight.in_dim

          address = weight.raw.to_unsafe.address.to_u64
          size = weight.raw.size.to_u64
          if address < @base || size > @file_size || address - @base > @file_size - size
            raise ArgumentError.new("Inkling 2 projection is outside the mapped GGUF")
          end

          job_offsets = {(address - @base).to_i64, input_size, output_size}
          input_size += aligned_bytes(input.size)
          output_size += aligned_bytes(weight.out_dim)
          job_offsets
        end

        input_buffer = @input = staging_buffer(@input, input_size)
        output_buffer = @output = staging_buffer(@output, output_size)
        jobs.each_with_index do |(_, input), index|
          destination = input_buffer.contents.as(UInt8*) + offsets[index][1]
          destination.as(Float32*).copy_from(input.to_unsafe, input.size)
        end

        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        begin
          jobs.each_with_index do |(weight, _), index|
            name, threads = case weight.type
                            when .f32?    then {"simd_mv_f32_f32", 128}
                            when .q8_0?   then {"simd_mv_q8_0_f32", 128}
                            when .iq4_nl? then {"simd_mv_iq4_nl_f32", 64}
                            else
                              raise ArgumentError.new("Unsupported Inkling 2 Metal projection type #{weight.type}")
                            end
            weight_offset, input_offset, output_offset = offsets[index]
            encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(name, SOURCE))
            encoder.set_buffer(@model, 0, offset: weight_offset)
            encoder.set_buffer(input_buffer, 1, offset: input_offset)
            encoder.set_buffer(output_buffer, 2, offset: output_offset)
            encoder.set_value(weight.in_dim.to_u32, 3)
            encoder.set_value(weight.out_dim.to_u32, 4)
            encoder.set_value(1_u32, 5)
            encoder.dispatch_threadgroups({(weight.out_dim + 3) // 4, 1, 1}, {threads, 1, 1})
          end
        ensure
          encoder.end_encoding
        end
        command.commit_and_wait

        jobs.map_with_index do |(weight, _), index|
          source = (output_buffer.contents.as(UInt8*) + offsets[index][2]).as(Float32*)
          Array(Float32).new(weight.out_dim) { |row| source[row] }
        end
      end

      def close : Nil
        @input.try(&.release)
        @output.try(&.release)
        @model.release
      end

      private def aligned_bytes(elements : Int32) : Int64
        ((elements.to_i64 * sizeof(Float32) + 15) // 16) * 16
      end

      private def staging_buffer(current : ML::MetalBuffer?, size : Int64) : ML::MetalBuffer
        return current if current && current.size >= size

        replacement = ML::MetalBuffer.new(size)
        current.try(&.release)
        replacement
      end
    end
  {% end %}
end
