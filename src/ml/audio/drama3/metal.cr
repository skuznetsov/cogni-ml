# Optional native Metal acceleration; the portable CPU implementation remains
# available in the same binary. Build with -Ddrama3_metal and bridge.o.
{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  require "../../metal/dispatch"
{% end %}

module ML::Audio::Drama3
  module MetalBackend
    extend self

    def enabled? : Bool
      backend = ENV["DRAMA3_BACKEND"]? || "cpu"
      raise ArgumentError.new("DRAMA3_BACKEND must be cpu or metal") unless backend == "cpu" || backend == "metal"
      return false if backend == "cpu"
      {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
        raise "DRAMA-3 Metal requires a unified-memory device" unless ML::Metal::Device.available? && ML::Metal::Device.instance.has_unified_memory?
        true
      {% else %}
        raise "DRAMA-3 Metal requires a Darwin build with -Ddrama3_metal (without -Dcpu_only)"
      {% end %}
    end

    {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
      lib Platform
        fun getpagesize : Int32
        fun gs_max_buffer_length : Int64
      end

      # Maximum size of one MTLBuffer on the active device. Callers still need
      # checked element-count arithmetic before comparing their byte size.
      def max_buffer_length : Int64
        Platform.gs_max_buffer_length
      end

      AFFINE_SOURCE = {{ read_file("#{__DIR__}/kernels/affine.metal") }}

      # Retain the tensor (and its mmap owner) for the entire GPU view lifetime.
      # Commands are synchronous, so callers may safely reuse host activations.
      class MappedBuffer
        getter buffer : ML::MetalBuffer

        def initialize(@owner : MappedFile)
          page = Platform.getpagesize.to_i64
          # mmap rounds the allocation to whole pages. The rounded tail is
          # never accessed by kernels: tensor slices carry checked byte ranges.
          length = ((@owner.size + page - 1) // page) * page
          limit = Platform.gs_max_buffer_length
          raise ArgumentError.new("safetensors shard requires #{length} Metal bytes; device limit is #{limit}") if length > limit
          @buffer = ML::MetalBuffer.wrap_no_copy(@owner.pointer.as(Void*), length)
        end
      end

      @@mappings = {} of UInt64 => MappedBuffer

      def mapped_buffer(owner : MappedFile) : MappedBuffer
        @@mappings[owner.object_id] ||= MappedBuffer.new(owner)
      end

      class TensorBuffer
        getter buffer : ML::MetalBuffer
        getter offset : Int64
        @owns_buffer : Bool

        def initialize(@owner : Tensor, raw_view : Bool)
          if raw_view && (raw = @owner.raw_bytes) && (mapping = @owner.owner)
            @owns_buffer = false
            @offset = (raw.to_unsafe.address - mapping.pointer.address).to_i64
            raise ArgumentError.new("tensor outside mapping") unless @offset >= 0 && raw.size <= mapping.size - @offset
            @buffer = MetalBackend.mapped_buffer(mapping).buffer
          else
            @owns_buffer = true
            @offset = 0_i64
            @buffer = ML::MetalBuffer.from_array(@owner.values)
          end
        end

        def release : Nil
          @buffer.release if @owns_buffer
        end
      end

      @@tensors = {} of Tuple(UInt64, Bool) => TensorBuffer
      @@input : ML::MetalBuffer?
      @@output : ML::MetalBuffer?
      @@mutex = Mutex.new

      # A transaction includes scratch writes, dispatch, completion, and readback.
      # All DRAMA-3 kernels share this lock, including the codec kernels.
      def synchronize(&block : -> T) : T forall T
        @@mutex.synchronize { yield }
      end

      # Call between inferences when unloading a model. Shared mmap buffers are
      # released once, after all tensor views, before dropping their owners.
      def reset! : Nil
        synchronize do
          @@tensors.each_value(&.release)
          @@tensors.clear
          @@mappings.each_value { |mapping| mapping.buffer.release }
          @@mappings.clear
          @@input.try(&.release)
          @@output.try(&.release)
          @@input = @@output = nil
        end
      end

      def tensor_buffer(tensor : Tensor, raw_view = false) : TensorBuffer
        @@tensors[{tensor.object_id, raw_view}] ||= TensorBuffer.new(tensor, raw_view)
      end

      private def float_encoding(tensor : Tensor) : UInt64
        case tensor.dtype
        when "F32"  then 0_u64
        when "BF16" then 1_u64
        when "F16"  then 2_u64
        else raise ArgumentError.new("Metal affine scale/bias requires F32, BF16, or F16")
        end
      end

      # Bounded reusable activation scratch; each call waits before reuse.
      def scratch(size : Int32, input : Bool) : ML::MetalBuffer
        current = input ? @@input : @@output
        if current.nil? || current.size < size.to_i64 * 4
          replacement = ML::MetalBuffer.new(size.to_i64 * 4)
          current.try(&.release)
          current = replacement
          if input
            @@input = current
          else
            @@output = current
          end
        end
        current
      end

      def run(name : String, source : String, &block : ML::Metal::ComputeEncoder -> Nil) : Nil
        Profile.measure("metal.submit_wait") do
          pipeline = ML::Metal::PipelineCache.get_or_compile(name, source)
          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          begin
            encoder.set_pipeline(pipeline)
            yield encoder
          ensure
            encoder.end_encoding
          end
          command.commit_and_wait
        end
      end

      def affine(matrix : Matrix, x : Vector) : Vector
        raise ArgumentError.new("Metal affine requires a quantized matrix") unless matrix.scales && matrix.biases
        raise ArgumentError.new("matrix input width #{x.size} != #{matrix.cols}") unless x.size == matrix.cols
        raise ArgumentError.new("too many affine groups for Metal threadgroup memory") if matrix.cols // matrix.group_size > 8192
        synchronize do
        Profile.measure("metal.affine") do
          weight = tensor_buffer(matrix.tensor, true)
            scales_type = float_encoding(matrix.scales.not_nil!)
            biases_type = float_encoding(matrix.biases.not_nil!)
            scales = tensor_buffer(matrix.scales.not_nil!, true)
            biases = tensor_buffer(matrix.biases.not_nil!, true)
          input = scratch(x.size, true)
          output = scratch(matrix.rows, false)
          input.write(x)
            params = StaticArray[matrix.rows.to_u64, matrix.cols.to_u64, matrix.bits.to_u64, matrix.group_size.to_u64, matrix.expert.to_u64,
              weight.offset.to_u64, scales.offset.to_u64, biases.offset.to_u64, scales_type, biases_type]
          run("drama3_affine_matvec", AFFINE_SOURCE) do |encoder|
              encoder.set_buffer(weight.buffer, 0)
            encoder.set_buffer(scales.buffer, 1)
            encoder.set_buffer(biases.buffer, 2)
            encoder.set_buffer(input, 3)
            encoder.set_buffer(output, 4, ML::Metal::BufferAccess::Write)
            encoder.set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 5)
            encoder.set_threadgroup_memory(matrix.cols // matrix.group_size * 4, 0)
            encoder.dispatch_threadgroups({matrix.rows, 1, 1}, {64, 1, 1})
          end
          output.read(matrix.rows)
        end
        end
      end

      # Batch independent expert matvecs into one command buffer submission.
      # Each dispatch keeps the same affine kernel and reduction order; only
      # the input/output activation slices differ between dispatches.
      def affine_batch(matrices : Array(Matrix), inputs : Array(Vector)) : Array(Vector)
        raise ArgumentError.new("affine matrix/input count mismatch") unless matrices.size == inputs.size
        return [] of Vector if matrices.empty?

        first = matrices[0]
        matrices.each_with_index do |matrix, index|
          raise ArgumentError.new("affine batch matrix dimensions differ") unless matrix.rows == first.rows && matrix.cols == first.cols
          raise ArgumentError.new("Metal affine requires a quantized matrix") unless matrix.scales && matrix.biases
          raise ArgumentError.new("matrix input width #{inputs[index].size} != #{matrix.cols}") unless inputs[index].size == matrix.cols
          raise ArgumentError.new("too many affine groups for Metal threadgroup memory") if matrix.cols // matrix.group_size > 8192
          float_encoding(matrix.scales.not_nil!)
          float_encoding(matrix.biases.not_nil!)
        end

        input_count = first.cols.to_i64 * matrices.size
        output_count = first.rows.to_i64 * matrices.size
        raise ArgumentError.new("affine batch input is too large") if input_count > Int32::MAX.to_i64
        raise ArgumentError.new("affine batch output is too large") if output_count > Int32::MAX.to_i64

        synchronize do
          Profile.measure("metal.affine_batch") do
            flat_input = Array(Float32).new(input_count.to_i32, 0_f32)
            inputs.each_with_index do |values, index|
              values.each_with_index { |value, column| flat_input[index * first.cols + column] = value }
            end
            input_buffer = scratch(input_count.to_i32, true)
            output_buffer = scratch(output_count.to_i32, false)
            input_buffer.write(flat_input)

            run("drama3_affine_matvec", AFFINE_SOURCE) do |encoder|
              matrices.each_with_index do |matrix, index|
                weight = tensor_buffer(matrix.tensor, true)
                scales_tensor = matrix.scales.not_nil!
                biases_tensor = matrix.biases.not_nil!
                scales = tensor_buffer(scales_tensor, true)
                biases = tensor_buffer(biases_tensor, true)
                params = StaticArray[
                  matrix.rows.to_u64,
                  matrix.cols.to_u64,
                  matrix.bits.to_u64,
                  matrix.group_size.to_u64,
                  matrix.expert.to_u64,
                  weight.offset.to_u64,
                  scales.offset.to_u64,
                  biases.offset.to_u64,
                  float_encoding(scales_tensor),
                  float_encoding(biases_tensor),
                ]
                input_offset = index.to_i64 * first.cols.to_i64 * sizeof(Float32).to_i64
                output_offset = index.to_i64 * first.rows.to_i64 * sizeof(Float32).to_i64
                encoder.set_buffer(weight.buffer, 0)
                encoder.set_buffer(scales.buffer, 1)
                encoder.set_buffer(biases.buffer, 2)
                encoder.set_buffer(input_buffer, 3, offset: input_offset)
                encoder.set_buffer(output_buffer, 4, ML::Metal::BufferAccess::Write, offset: output_offset)
                encoder.set_bytes(params.to_unsafe.as(Void*), sizeof(typeof(params)), 5)
                encoder.set_threadgroup_memory(matrix.cols // matrix.group_size * 4, 0)
                encoder.dispatch_threadgroups({matrix.rows, 1, 1}, {64, 1, 1})
              end
            end

            flat_output = output_buffer.read(output_count.to_i32)
            Array(Vector).new(matrices.size) do |index|
              Array(Float32).new(first.rows) { |row| flat_output[index * first.rows + row] }
            end
          end
        end
      end
    {% end %}
  end
end
