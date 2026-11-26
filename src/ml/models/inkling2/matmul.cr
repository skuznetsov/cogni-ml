require "./weights"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  require "../../metal/device"
  require "../../metal/dispatch"

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

    def try_moe(layer : Inkling2LayerWeights, input : Array(Float32), config : Inkling2Config) : Array(Float32)?
      nil
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

      SOURCE = {{ read_file("#{__DIR__}/../../metal/kernels/shared/gemm_q56k.metal") }}
      MOE_SOURCE = SOURCE + "\n" + {{ read_file("#{__DIR__}/../../metal/kernels/inkling2/inkling2_moe.metal") }}

      # Keep the router's dynamic threadgroup storage bounded. Larger shapes
      # can use the same host routing path as other expert quantizations.
      private MAX_ROUTE_SCRATCH = 16_384_i32

      @base : UInt64
      @file_size : UInt64
      @model : ML::MetalBuffer
      @input : ML::MetalBuffer? = nil
      @output : ML::MetalBuffer? = nil
      @moe_scratch : ML::MetalBuffer? = nil

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

          job_offsets = {model_offset(weight.raw), input_size, output_size}
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
            weight_offset, input_offset, output_offset = offsets[index]
            encode_projection(encoder, weight, weight_offset, input_buffer, output_buffer, input_offset, output_offset)
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

      def try_moe(layer : Inkling2LayerWeights, input : Array(Float32), config : Inkling2Config) : Array(Float32)?
        gate_up_kernel = expert_kernel(layer.gate_up.type)
        down_kernel = expert_kernel(layer.down.type)
        return unless gate_up_kernel && down_kernel

        route_scratch = aligned_bytes(2_i64 * config.n_expert)
        return if route_scratch > MAX_ROUTE_SCRATCH

        raise ArgumentError.new("Inkling 2 MoE input width mismatch") unless input.size == config.n_embd

        picked = config.n_expert_used
        width = config.expert_dim
        dim = config.n_embd
        bias = layer.router_bias
        input_bytes = aligned_bytes(dim)
        router_bytes = aligned_bytes(config.n_expert)
        input_buffer = @input = staging_buffer(@input, input_bytes + (bias ? router_bytes : 0_i64))
        output_buffer = @output = staging_buffer(@output, input_bytes)
        input_data = input_buffer.contents.as(UInt8*)
        input_data.as(Float32*).copy_from(input.to_unsafe, input.size)
        if bias
          (input_data + input_bytes).as(Float32*).copy_from(bias.to_unsafe, bias.size)
        end

        ids_offset = router_bytes
        weights_offset = ids_offset + aligned_bytes(picked)
        gate_up_offset = weights_offset + aligned_bytes(picked)
        gated_offset = gate_up_offset + aligned_bytes(picked.to_i64 * layer.gate_up.out_dim)
        down_offset = gated_offset + aligned_bytes(picked.to_i64 * width)
        scratch_bytes = down_offset + aligned_bytes(picked.to_i64 * dim)
        scratch = @moe_scratch = staging_buffer(@moe_scratch, scratch_bytes, ML::StorageMode::Private)

        # The engine processes one token at a time. Serial dispatches share an
        # engine-owned workspace, and the final wait ends every GPU use of it.
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        begin
          encode_projection(encoder, layer.router, model_offset(layer.router.raw), input_buffer, scratch)

          encoder.set_pipeline(moe_pipeline("inkling2_moe_route"))
          encoder.set_buffer(scratch, 0)
          encoder.set_buffer(input_buffer, 1, offset: bias ? input_bytes : 0_i64)
          encoder.set_buffer(scratch, 2, ML::Metal::BufferAccess::Write, offset: ids_offset)
          encoder.set_buffer(scratch, 3, ML::Metal::BufferAccess::Write, offset: weights_offset)
          encoder.set_value(config.n_expert.to_u32, 4)
          encoder.set_value(picked.to_u32, 5)
          encoder.set_value(config.expert_scale, 6)
          encoder.set_value(bias ? 1_u32 : 0_u32, 7)
          encoder.set_threadgroup_memory(route_scratch.to_i32, 0)
          encoder.dispatch_threadgroups({1, 1, 1}, {32, 1, 1})

          encode_experts(encoder, gate_up_kernel, layer.gate_up, input_buffer, scratch, ids_offset, gate_up_offset, picked)

          encoder.set_pipeline(moe_pipeline("inkling2_moe_situ"))
          encoder.set_buffer(scratch, 0, offset: gate_up_offset)
          encoder.set_buffer(scratch, 1, ML::Metal::BufferAccess::Write, offset: gated_offset)
          encoder.set_value(width.to_u32, 2)
          encoder.set_value(picked.to_u32, 3)
          encoder.set_value(config.situ_beta, 4)
          encoder.set_value(config.situ_linear_beta, 5)
          encoder.dispatch_1d(picked * width)

          encode_experts(encoder, down_kernel, layer.down, scratch, scratch, ids_offset, down_offset, picked,
            input_offset: gated_offset, input_stride: width)

          encoder.set_pipeline(moe_pipeline("inkling2_moe_combine"))
          encoder.set_buffer(scratch, 0, offset: down_offset)
          encoder.set_buffer(scratch, 1, offset: weights_offset)
          encoder.set_buffer(output_buffer, 2, ML::Metal::BufferAccess::Write)
          encoder.set_value(dim.to_u32, 3)
          encoder.set_value(picked.to_u32, 4)
          encoder.dispatch_1d(dim)
        ensure
          encoder.end_encoding
        end
        command.commit_and_wait
        output_buffer.read(dim)
      end

      def close : Nil
        @input.try(&.release)
        @output.try(&.release)
        @moe_scratch.try(&.release)
        @model.release
      end

      private def encode_projection(encoder : ML::Metal::ComputeEncoder, weight : QuantWeight, weight_offset : Int64,
                                    input : ML::MetalBuffer, output : ML::MetalBuffer,
                                    input_offset : Int64 = 0_i64, output_offset : Int64 = 0_i64) : Nil
        name, threads = case weight.type
                        when .f32?    then {"simd_mv_f32_f32", 128}
                        when .q8_0?   then {"simd_mv_q8_0_f32", 128}
                        when .iq4_nl? then {"simd_mv_iq4_nl_f32", 64}
                        else
                          raise ArgumentError.new("Unsupported Inkling 2 Metal projection type #{weight.type}")
                        end
        encoder.set_pipeline(ML::Metal::PipelineCache.get_or_compile(name, SOURCE))
        encoder.set_buffer(@model, 0, offset: weight_offset)
        encoder.set_buffer(input, 1, offset: input_offset)
        encoder.set_buffer(output, 2, ML::Metal::BufferAccess::Write, offset: output_offset)
        encoder.set_value(weight.in_dim.to_u32, 3)
        encoder.set_value(weight.out_dim.to_u32, 4)
        encoder.set_value(1_u32, 5)
        encoder.dispatch_threadgroups({(weight.out_dim + 3) // 4, 1, 1}, {threads, 1, 1})
      end

      private def expert_kernel(type : TensorType) : {String, Int32}?
        case type
        when .iq4_nl? then {"inkling2_moe_matvec_iq4_nl", 64}
        when .q8_0?   then {"inkling2_moe_matvec_q8_0", 128}
        end
      end

      private def encode_experts(encoder : ML::Metal::ComputeEncoder, kernel : {String, Int32}, weight : Inkling2ExpertWeights,
                                 input : ML::MetalBuffer, scratch : ML::MetalBuffer,
                                 ids_offset : Int64, output_offset : Int64, picked : Int32,
                                 *, input_offset : Int64 = 0_i64, input_stride : Int32 = 0) : Nil
        name, threads = kernel
        encoder.set_pipeline(moe_pipeline(name))
        encoder.set_buffer(@model, 0, offset: model_offset(weight.raw))
        encoder.set_buffer(input, 1, offset: input_offset)
        encoder.set_buffer(scratch, 2, offset: ids_offset)
        encoder.set_buffer(scratch, 3, ML::Metal::BufferAccess::Write, offset: output_offset)
        encoder.set_value(weight.in_dim.to_u32, 4)
        encoder.set_value(weight.out_dim.to_u32, 5)
        encoder.set_value(weight.expert_bytes.to_u32, 6)
        encoder.set_value(input_stride.to_u32, 7)
        encoder.dispatch_threadgroups({(weight.out_dim + 3) // 4, picked, 1}, {threads, 1, 1})
      end

      private def moe_pipeline(name : String) : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get_or_compile(name, MOE_SOURCE)
      end

      private def model_offset(raw : Bytes) : Int64
        address = raw.to_unsafe.address.to_u64
        size = raw.size.to_u64
        if address < @base || size > @file_size || address - @base > @file_size - size
          raise ArgumentError.new("Inkling 2 projection is outside the mapped GGUF")
        end
        (address - @base).to_i64
      end

      private def aligned_bytes(elements : Int) : Int64
        ((elements.to_i64 * sizeof(Float32) + 15) // 16) * 16
      end

      private def staging_buffer(current : ML::MetalBuffer?, size : Int64, mode = ML::StorageMode::Shared) : ML::MetalBuffer
        return current if current && current.size >= size

        replacement = ML::MetalBuffer.new(size, mode)
        current.try(&.release)
        replacement
      end
    end
  {% end %}
end
