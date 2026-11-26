require "./native"
require "./metal"

module ML::Audio::Drama3
  # Exact-order Metal kernels for the DAC convolution stages. Weight
  # normalization remains on the CPU; callers pass the effective FP32 vector.
  module CodecMetal
    extend self

    {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
      CODEC_SOURCE = {{ read_file("#{__DIR__}/kernels/codec.metal") }}
      MAX_CACHED_WEIGHT_BYTES = 128_i64 * 1024 * 1024
      MAX_CACHED_WEIGHT_ENTRIES = 128

      private class WeightBuffers
        getter weight_values : Vector
        getter bias_values : Vector
        getter weight : ML::MetalBuffer
        getter bias : ML::MetalBuffer
        getter byte_size : Int64

        def initialize(weights : Vector, biases : Vector, retain_values : Bool = true)
          @weight_values = retain_values ? weights.dup : weights
          @bias_values = retain_values ? biases.dup : biases
          weight_buffer = ML::MetalBuffer.from_array(@weight_values)
          bias_buffer = begin
            ML::MetalBuffer.from_array(@bias_values)
          rescue ex
            weight_buffer.release
            raise ex
          end
          @weight = weight_buffer
          @bias = bias_buffer
          @byte_size = (@weight_values.size.to_i64 + @bias_values.size.to_i64) * 4_i64
        end

        def matches?(weights : Vector, biases : Vector) : Bool
          @weight_values == weights && @bias_values == biases
        end

        def release : Nil
          @weight.release
          @bias.release
        end
      end

      @@mutex = Mutex.new
      @@weight_buffers = {} of String => WeightBuffers
      @@weight_lru = [] of String
      @@cached_weight_bytes = 0_i64

      # Release cached device-side copies after a decode or when changing model
      # weights. Scratch and dispatch state are serialized by MetalBackend.
      def clear! : Nil
        @@mutex.synchronize do
          @@weight_buffers.each_value(&.release)
          @@weight_buffers.clear
          @@weight_lru.clear
          @@cached_weight_bytes = 0_i64
        end
      end

      # input is [time, in_channels]; weights are [out_channels, in/group, k].
      # Output time intentionally matches the existing native Conv1d behavior.
      def conv(input : Matrix, effective : Vector, bias : Vector, kernel : Int32,
               stride : Int32, groups : Int32, dilation : Int32, prefix : String) : Matrix
        validate_conv!(input, effective, bias, kernel, stride, groups, dilation)
        out_channels = bias.size.to_i32
        output_count = checked_count(input.rows, out_channels)

        MetalBackend.synchronize do
          @@mutex.synchronize do
            Profile.measure("metal.codec_conv1d", category: "dac.conv1d") do
              input_buffer = MetalBackend.scratch(input.values.size.to_i32, true)
              output_buffer = MetalBackend.scratch(output_count, false)
              input_buffer.write(input.values)
              binding, cached = weight_buffers(prefix, effective, bias)
              begin
                params = StaticArray[
                  input.rows.to_u32,
                  input.cols.to_u32,
                  out_channels.to_u32,
                  kernel.to_u32,
                  stride.to_u32,
                  groups.to_u32,
                  dilation.to_u32,
                ]
                MetalBackend.run("drama3_codec_conv1d", CODEC_SOURCE) do |encoder|
                  encoder.set_buffer(input_buffer, 0)
                  encoder.set_buffer(binding.weight, 1)
                  encoder.set_buffer(binding.bias, 2)
                  encoder.set_buffer(output_buffer, 3, ML::Metal::BufferAccess::Write)
                  encoder.set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 4)
                  encoder.dispatch_1d(output_count, 64)
                end
                Matrix.new(input.rows, out_channels, output_buffer.read(output_count))
              ensure
                binding.release unless cached
              end
            end
          end
        end
      end

      # input is [time, in_channels]; weights are [in_channels, out, k]. The
      # decoder truncates to input.rows * stride instead of full convolution size.
      def transpose(input : Matrix, effective : Vector, bias : Vector, kernel : Int32,
                    stride : Int32, prefix : String) : Matrix
        validate_transpose!(input, effective, bias, kernel, stride)
        in_channels = input.cols
        out_channels = bias.size.to_i32
        output_time = (input.rows.to_i64 * stride).to_i32
        output_count = checked_count(output_time, out_channels)

        MetalBackend.synchronize do
          @@mutex.synchronize do
            Profile.measure("metal.codec_conv_transpose1d", category: "dac.conv_transpose1d") do
              input_buffer = MetalBackend.scratch(input.values.size.to_i32, true)
              output_buffer = MetalBackend.scratch(output_count, false)
              input_buffer.write(input.values)
              binding, cached = weight_buffers(prefix, effective, bias)
              begin
                params = StaticArray[
                  input.rows.to_u32,
                  in_channels.to_u32,
                  out_channels.to_u32,
                  kernel.to_u32,
                  stride.to_u32,
                ]
                MetalBackend.run("drama3_codec_conv_transpose1d", CODEC_SOURCE) do |encoder|
                  encoder.set_buffer(input_buffer, 0)
                  encoder.set_buffer(binding.weight, 1)
                  encoder.set_buffer(binding.bias, 2)
                  encoder.set_buffer(output_buffer, 3, ML::Metal::BufferAccess::Write)
                  encoder.set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 4)
                  encoder.dispatch_1d(output_count, 64)
                end
                Matrix.new(output_time, out_channels, output_buffer.read(output_count))
              ensure
                binding.release unless cached
              end
            end
          end
        end
      end

      private def validate_conv!(input : Matrix, weights : Vector, bias : Vector,
                                 kernel : Int32, stride : Int32, groups : Int32,
                                 dilation : Int32) : Nil
        raise ArgumentError.new("Conv1d kernel, stride, groups, and dilation must be positive") if kernel <= 0 || stride <= 0 || groups <= 0 || dilation <= 0
        raise ArgumentError.new("invalid Conv1d groups") if input.cols % groups != 0
        raise ArgumentError.new("Conv1d bias must be non-empty and divisible by groups") if bias.empty? || bias.size % groups != 0
        expected = bias.size.to_i64 * (input.cols // groups) * kernel
        raise ArgumentError.new("Conv1d effective weight size mismatch") unless expected == weights.size
        left_pad = (kernel.to_i64 - 1) * dilation
        last_stride = (input.rows.to_i64 - 1) * stride
        raise ArgumentError.new("Conv1d index exceeds Int32 range") if left_pad > Int32::MAX || last_stride > Int32::MAX
        checked_count(input.rows, bias.size.to_i32)
      end

      private def validate_transpose!(input : Matrix, weights : Vector, bias : Vector,
                                      kernel : Int32, stride : Int32) : Nil
        raise ArgumentError.new("ConvTranspose1d kernel and stride must be positive") if kernel <= 0 || stride <= 0
        raise ArgumentError.new("ConvTranspose1d bias must be non-empty") if bias.empty?
        expected = input.cols.to_i64 * bias.size * kernel
        raise ArgumentError.new("ConvTranspose1d effective weight size mismatch") unless expected == weights.size
        output_time = input.rows.to_i64 * stride
        raise ArgumentError.new("ConvTranspose1d output time exceeds Int32 range") if output_time > Int32::MAX
        checked_count(output_time.to_i32, bias.size.to_i32)
      end

      private def checked_count(rows : Int32, cols : Int32) : Int32
        checked_count(rows.to_i64, cols.to_i64)
      end

      private def checked_count(rows : Int64, cols : Int64) : Int32
        count = rows * cols
        raise ArgumentError.new("Metal convolution output is too large") if rows <= 0 || cols <= 0 || count > Int32::MAX
        count.to_i32
      end

      private def weight_buffers(prefix : String, weights : Vector, biases : Vector) : Tuple(WeightBuffers, Bool)
        if current = @@weight_buffers[prefix]?
          if current.matches?(weights, biases)
            @@weight_lru.delete(prefix)
            @@weight_lru << prefix
            return {current, true}
          end
          remove_weight_buffers(prefix)
        end

        byte_size = (weights.size.to_i64 + biases.size.to_i64) * 4_i64
        if byte_size > MAX_CACHED_WEIGHT_BYTES
          return {WeightBuffers.new(weights, biases, false), false}
        end
        created = WeightBuffers.new(weights, biases)

        while @@weight_buffers.size >= MAX_CACHED_WEIGHT_ENTRIES ||
              @@cached_weight_bytes + created.byte_size > MAX_CACHED_WEIGHT_BYTES
          evict_oldest_weight_buffers
        end
        @@weight_buffers[prefix] = created
        @@weight_lru << prefix
        @@cached_weight_bytes += created.byte_size
        {created, true}
      end

      private def remove_weight_buffers(prefix : String) : Nil
        @@weight_lru.delete(prefix)
        if old = @@weight_buffers.delete(prefix)
          @@cached_weight_bytes -= old.byte_size
          old.release
        end
      end

      private def evict_oldest_weight_buffers : Nil
        oldest = @@weight_lru.first?
        return unless oldest
        remove_weight_buffers(oldest)
      end
    {% else %}
      def clear! : Nil
      end

      def conv(input : Matrix, effective : Vector, bias : Vector, kernel : Int32,
               stride : Int32, groups : Int32, dilation : Int32, prefix : String) : Matrix
        raise "DRAMA-3 codec Metal requires a Darwin build with -Ddrama3_metal"
      end

      def transpose(input : Matrix, effective : Vector, bias : Vector, kernel : Int32,
                    stride : Int32, prefix : String) : Matrix
        raise "DRAMA-3 codec Metal requires a Darwin build with -Ddrama3_metal"
      end
    {% end %}
  end
end
