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

      # Retain the mmap owner for the entire GPU view lifetime. Resident callers
      # must keep the wrapper alive until their existing terminal boundary.
      class MappedBuffer
        getter buffer : ML::MetalBuffer
        getter owner : MappedFile

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

        def initialize(@owner : Tensor, raw_view : Bool, mapped : MappedBuffer? = nil)
          if raw_view && (raw = @owner.raw_bytes) && (mapping = @owner.owner)
            @owns_buffer = false
            @offset = (raw.to_unsafe.address - mapping.pointer.address).to_i64
            raise ArgumentError.new("tensor outside mapping") unless @offset >= 0 && raw.size <= mapping.size - @offset
            raise ArgumentError.new("mapped buffer owner differs") if mapped && !mapped.owner.same?(mapping)
            @buffer = (mapped || MetalBackend.mapped_buffer(mapping)).buffer
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

      # Diagnostic bank for a single Resident context. Normal inference keeps
      # the existing global caches. Only Context's checked idle boundary may
      # call prepare: releasing raw packed bindings while in flight is unsafe.
      class WeightBufferBank
        @mappings = {} of UInt64 => MappedBuffer
        @tensors = {} of UInt64 => TensorBuffer
        @generation = 0_i64
        @view_generation = 0_i64
        @factory_calls = 0_i64
        @tensor_views_created = 0_i64
        @shard_ids = {} of UInt64 => Int32
        @buffer_owners = {} of UInt64 => UInt64
        @binding_counts = Hash(UInt64, Int64).new(0_i64)
        @capture_armed = false
        @capture_observed = false
        @capture_submitted = false
        @capture_index = -1_i64
        @buffer_bindings = 0_i64
        @raw_weight_bindings = 0_i64

        def tensor_buffer(tensor : Tensor) : TensorBuffer
          if tensor.raw_bytes && (owner = tensor.owner)
            @tensors[tensor.object_id] ||= begin
              mapping = @mappings[owner.object_id] ||= create(owner)
              @tensor_views_created += 1
              TensorBuffer.new(tensor, true, mapping)
            end
          else
            MetalBackend.tensor_buffer(tensor, true)
          end
        end

        private def create(owner : MappedFile) : MappedBuffer
          mapping = MappedBuffer.new(owner)
          @factory_calls += 1
          @shard_ids[owner.object_id] ||= @shard_ids.size + 1
          @buffer_owners[mapping.buffer.object_id] = owner.object_id
          @binding_counts[owner.object_id] ||= 0_i64
          mapping
        end

        # This is host API exposure, not shader access, residency or a driver
        # resource ID. Context arms at an idle job boundary and seals only after
        # its existing first submit succeeds. Snapshot publication additionally
        # requires Context's existing terminal boundary.
        def start_first_command(index : Int64) : Nil
          @capture_index = index
          @capture_armed = true
          @capture_observed = @capture_submitted = false
          @buffer_bindings = @raw_weight_bindings = 0_i64
          @binding_counts.clear
          @mappings.each_key { |owner| @binding_counts[owner] = 0_i64 }
        end

        def first_command_capture? : Bool
          @capture_armed
        end

        def first_command_started! : Nil
          raise ArgumentError.new("first weight command already observed") unless @capture_armed && !@capture_observed
          @capture_observed = true
        end

        # Called only after the underlying encoder accepted set_buffer. The
        # reverse table is prepared at mapping construction, never by a tensor
        # lookup or an Objective-C address that could be recycled after release.
        def record_binding(buffer : ML::MetalBuffer) : Nil
          return unless @capture_armed && @capture_observed
          @buffer_bindings += 1
          if owner = @buffer_owners[buffer.object_id]?
            @binding_counts[owner] += 1
            @raw_weight_bindings += 1
          end
        end

        def first_command_submitted! : Nil
          if @capture_armed && @capture_observed
            @capture_submitted = true
            @capture_armed = false
          end
        end

        def discard_first_command! : Nil
          @capture_armed = @capture_submitted = false
        end

        def first_command_bindings
          raise ArgumentError.new("first weight command has not been submitted") unless @capture_submitted
          shards = @mappings.map do |owner, mapping|
            {shard: @shard_ids[owner], mapped_bytes: mapping.buffer.size, bindings: @binding_counts[owner]}
          end.sort_by { |shard| shard[:shard] }
          {command_index: @capture_index, observed: @capture_observed, submitted: @capture_submitted,
           buffer_bindings: @buffer_bindings, raw_weight_bindings: @raw_weight_bindings, shards: shards}
        end

        # Recreate exactly one buffer per previously used shard. Keep old
        # Crystal objects/owners for the qualification, but release ALL Metal
        # handles before constructing any replacement: no overlapping aliases.
        # Objective-C addresses may be recycled and are not resource IDs.
        def prepare(refresh : Bool, clear_views : Bool = false) : Hash(String, Int64)
          retired = constructed = same_backing = address_reuses = 0_i64
          retire_ns = construct_ns = 0_i64
          views_cleared = 0_i64
          if refresh || clear_views
            views_cleared = @tensors.size.to_i64
            @tensors.clear
            @view_generation += 1
          end
          if refresh
            old = @mappings.values.map do |mapping|
              buffer = mapping.buffer
              {mapping, buffer.contents, buffer.size, buffer.storage_mode, buffer.handle.address}
            end
            started = Time.instant
            old.each { |item| item[0].buffer.release; retired += 1 }
            retire_ns = (Time.instant - started).total_nanoseconds.to_i64
            @mappings.clear
            @buffer_owners.clear
            old.each do |item|
              previous, pointer, size, mode, address = item
              raise "old weight wrapper still valid" if previous.buffer.valid?
              started = Time.instant
              fresh = create(previous.owner)
              construct_ns += (Time.instant - started).total_nanoseconds.to_i64
              @mappings[previous.owner.object_id] = fresh
              buffer = fresh.buffer
              raise "weight backing changed" unless buffer.valid? && buffer.contents == pointer && buffer.size == size && buffer.storage_mode == mode
              raise "weight wrapper object reused" if buffer.same?(previous.buffer)
              same_backing += 1
              address_reuses += 1 if buffer.handle.address == address
              constructed += 1
            end
            @generation += 1
          end
          {"generation" => @generation, "view_generation" => @view_generation,
           "tensor_views_created" => @tensor_views_created, "tensor_views" => @tensors.size.to_i64,
           "mappings" => @mappings.size.to_i64,
           "mapped_bytes" => @mappings.values.sum(0_i64) { |m| m.buffer.size },
           "factory_calls" => @factory_calls, "retired" => retired, "constructed" => constructed,
           "same_backing" => same_backing, "handle_address_reuses" => address_reuses,
           "tensor_views_cleared" => views_cleared, "retire_ns" => retire_ns, "construct_ns" => construct_ns}
        end
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
