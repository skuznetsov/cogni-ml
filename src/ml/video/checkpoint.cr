require "json"
require "set"
require "../gguf/safetensors"
require "./metal_ops"

module ML::Video
  record CheckpointTensor, name : String, dtype : ML::GGUF::SafeTensorDType,
    shape : Array(Int64), data_start : Int64, data_end : Int64, n_elements : Int64 do
    def data_bytes : Int64
      data_end - data_start
    end

    def snapshot : self
      self.class.new(name, dtype, shape.dup, data_start, data_end, n_elements)
    end
  end

  # Header-only, strict admission for immutable local safetensors. Uses the
  # engine's dtype representation, but never invokes its CPU F32 decoder or
  # maps an entire multi-GB shard. Keep files immutable while a plan is used.
  class CheckpointFile
    MAX_HEADER_BYTES = 16_i64 * 1024 * 1024
    getter path : String
    @tensors = {} of String => CheckpointTensor
    getter data_offset : Int64 = 0_i64
    @io : File
    @initial_info : File::Info
    @closed = false

    def initialize(@path : String)
      @io = File.open(path, "rb")
      @initial_info = @io.info
      begin
        parse!
      rescue ex
        close
        raise ex
      end
    end

    def close : Nil
      return if @closed
      @closed = true
      @io.close
    end

    def finalize
      close
    end

    def unchanged! : Nil
      raise ArgumentError.new("Checkpoint is closed") if @closed
      info = @io.info
      unless info.same_file?(@initial_info) && info.size == @initial_info.size && info.modification_time == @initial_info.modification_time
        raise ArgumentError.new("Checkpoint changed after admission")
      end
    end

    def tensors : Hash(String, CheckpointTensor)
      @tensors.transform_values(&.snapshot)
    end

    def validate_tensor!(info : CheckpointTensor) : Nil
      unchanged!
      unless @tensors[info.name]? == info
        raise ArgumentError.new("Invalid checkpoint tensor descriptor")
      end
    end

    def copy_raw(info : CheckpointTensor, destination : Bytes) : Nil
      validate_tensor!(info)
      unless destination.size == info.data_bytes
        raise ArgumentError.new("Invalid checkpoint tensor copy")
      end
      @io.seek(@data_offset + info.data_start)
      @io.read_fully(destination)
      unchanged!
    end

    private def parse! : Nil
      size = @initial_info.size
      raise ArgumentError.new("Truncated checkpoint") if size < 10
      prefix = uninitialized UInt64
      @io.read_fully(Bytes.new(pointerof(prefix).as(UInt8*), 8))
      length = IO::ByteFormat::LittleEndian.decode(UInt64, Bytes.new(pointerof(prefix).as(UInt8*), 8))
      unless length >= 2 && length <= MAX_HEADER_BYTES && length <= size - 8
        raise ArgumentError.new("Invalid checkpoint header length")
      end
      @data_offset = 8_i64 + length.to_i64
      raw = Bytes.new(length.to_i32)
      @io.read_fully(raw)
      raise ArgumentError.new("Checkpoint header must start with an object") unless raw[0] == '{'.ord
      pull = JSON::PullParser.new(String.new(raw))
      keys = Set(String).new
      pull.read_object do |name|
        raise ArgumentError.new("Duplicate checkpoint key #{name}") unless keys.add?(name)
        if name == "__metadata__"
          metadata_keys = Set(String).new
          pull.read_object do |key|
            raise ArgumentError.new("Duplicate checkpoint metadata") unless metadata_keys.add?(key)
            pull.read_string
          end
          next
        end
        raise ArgumentError.new("Empty checkpoint tensor name") if name.empty?
        dtype = nil.as(ML::GGUF::SafeTensorDType?)
        shape = [] of Int64
        offsets = [] of Int64
        fields = Set(String).new
        pull.read_object do |key|
          raise ArgumentError.new("Duplicate checkpoint tensor field") unless fields.add?(key)
          case key
          when "dtype"
            dtype = ML::GGUF::SafeTensorDType.parse(pull.read_string)
          when "shape"
            pull.read_array do
              raise ArgumentError.new("Excessive checkpoint tensor rank") if shape.size >= 16
              shape << pull.read_int
            end
          when "data_offsets"
            pull.read_array do
              raise ArgumentError.new("Invalid checkpoint offset count") if offsets.size >= 2
              offsets << pull.read_int
            end
          else
            raise ArgumentError.new("Unknown checkpoint tensor field #{key}")
          end
        end
        unless fields.size == 3 && dtype && fields.includes?("shape") && offsets.size == 2
          raise ArgumentError.new("Incomplete checkpoint tensor #{name}")
        end
        count = 1_i64
        shape.each do |dimension|
          unless dimension > 0 && dimension <= Int32::MAX && count <= Int32::MAX // dimension
            raise ArgumentError.new("Unsupported checkpoint tensor shape #{name}")
          end
          count *= dimension
        end
        bytes = count * dtype.not_nil!.bytes_per_element
        unless bytes <= Int32::MAX && offsets[0] >= 0 && offsets[1] >= offsets[0] &&
               offsets[1] <= size - @data_offset && offsets[1] - offsets[0] == bytes
          raise ArgumentError.new("Invalid checkpoint tensor range #{name}")
        end
        @tensors[name] = CheckpointTensor.new(name, dtype.not_nil!, shape, offsets[0], offsets[1], count)
      end
      raise ArgumentError.new("Trailing checkpoint JSON") unless pull.kind.eof?
      cursor = 0_i64
      @tensors.values.sort_by(&.data_start).each do |info|
        raise ArgumentError.new("Overlapping or incomplete checkpoint data") unless info.data_start == cursor
        cursor = info.data_end
      end
      unless !@tensors.empty? && cursor == size - @data_offset
        raise ArgumentError.new("Incomplete checkpoint payload")
      end
    end
  end

  record WeightBinding, target : String, source : String, shape : Array(Int64),
    dtype : ML::GGUF::SafeTensorDType? = nil do
    def snapshot : self
      self.class.new(target, source, shape.dup, dtype)
    end
  end
  record PlannedWeight, binding : WeightBinding, file : CheckpointFile,
    info : CheckpointTensor, raw_offset : Int64 do
    def snapshot : self
      self.class.new(binding.snapshot, file, info.snapshot, raw_offset)
    end
  end

  # No device initialization, mapping or tensor allocation. The budget includes
  # destinations, raw GPU staging and caller-reserved scratch/other residency.
  class WeightPlan
    @entries = [] of PlannedWeight
    getter destination_bytes = 0_i64
    getter staging_bytes = 0_i64
    getter peak_bytes : Int64
    getter reserve_bytes : Int64

    def entries : Array(PlannedWeight)
      @entries.map(&.snapshot)
    end

    def initialize(files : Array(CheckpointFile), bindings : Array(WeightBinding),
                   budget_bytes : Int64, @reserve_bytes : Int64 = 0_i64)
      unless budget_bytes > 0 && reserve_bytes >= 0 && reserve_bytes <= budget_bytes && !bindings.empty?
        raise ArgumentError.new("Invalid checkpoint memory budget or bindings")
      end
      tensors = {} of String => {CheckpointFile, CheckpointTensor}
      files.each do |file|
        file.unchanged!
        file.tensors.each do |name, info|
          raise ArgumentError.new("Duplicate checkpoint tensor across shards #{name}") if tensors.has_key?(name)
          tensors[name] = {file, info}
        end
      end
      targets = Set(String).new
      bindings.each do |binding|
        unless !binding.target.empty? && targets.add?(binding.target)
          raise ArgumentError.new("Duplicate or empty checkpoint destination")
        end
        file, info = tensors[binding.source]? || raise ArgumentError.new("Missing checkpoint tensor #{binding.source}")
        unless info.shape == binding.shape
          raise ArgumentError.new("Checkpoint shape mismatch #{binding.source}: #{info.shape} != #{binding.shape}")
        end
        if binding.dtype && binding.dtype != info.dtype
          raise ArgumentError.new("Checkpoint dtype mismatch #{binding.source}")
        end
        # The reader admitted <= Int32::MAX bytes, so these additions cannot
        # overflow Int64 before the explicit bounded-budget rejection.
        @staging_bytes = ((@staging_bytes + 3) // 4) * 4
        @entries << PlannedWeight.new(binding.snapshot, file, info.snapshot, @staging_bytes)
        @staging_bytes += info.data_bytes
        @destination_bytes += (info.data_bytes // info.dtype.bytes_per_element) * 4
        if destination_bytes > budget_bytes ||
           staging_bytes > budget_bytes - destination_bytes ||
           reserve_bytes > budget_bytes - destination_bytes - staging_bytes
          raise ArgumentError.new("Checkpoint exceeds memory budget")
        end
      end
      @peak_bytes = destination_bytes + staging_bytes + reserve_bytes
    end
  end

  # Setup owns raw staging; Arena owns F32 destinations. Keep both alive until
  # the caller's final command completes. encode only binds and dispatches.
  class PreparedWeights
    SOURCE = {{ read_file("#{__DIR__}/kernels/weight_decode.metal") }}
    @raw : ML::MetalBuffer?
    @destinations = [] of ML::MetalBuffer
    @entries : Array(PlannedWeight)
    @pipeline : ML::Metal::ComputePipeline

    def initialize(plan : WeightPlan, arena : Arena)
      @raw = nil
      @entries = plan.entries
      @entries.each do |entry|
        if arena.buffers.has_key?(entry.binding.target)
          raise ArgumentError.new("Checkpoint destination already exists #{entry.binding.target}")
        end
        entry.file.validate_tensor!(entry.info)
        unless entry.raw_offset >= 0 && entry.raw_offset % 4 == 0 &&
               entry.raw_offset <= plan.staging_bytes && entry.info.data_bytes <= plan.staging_bytes - entry.raw_offset
          raise ArgumentError.new("Invalid checkpoint staging range")
        end
      end
      unless arena.buffers.values.sum(0_i64, &.size) <= plan.reserve_bytes
        raise ArgumentError.new("Existing arena buffers exceed checkpoint reserve")
      end
      @pipeline = ML::Metal::PipelineCache.get("video_weight_decode") do
        ML::Metal::ComputePipeline.new("video_weight_decode", SOURCE)
      end
      names = [] of String
      begin
        raw = ML::MetalBuffer.new(plan.staging_bytes)
        @raw = raw
        @entries.each do |entry|
          info = entry.info
          destination = arena.alloc(entry.binding.target, (info.data_bytes // info.dtype.bytes_per_element).to_i32)
          names << entry.binding.target
          @destinations << destination
          entry.file.copy_raw(info, Bytes.new(raw.contents.as(UInt8*) + entry.raw_offset, info.data_bytes.to_i32))
        end
      rescue ex
        release
        names.each { |name| arena.buffers.delete(name).try(&.release) }
        raise ex
      end
    end

    def encode(encoder : ML::Metal::ComputeEncoder) : Nil
      raw = @raw || raise ArgumentError.new("Checkpoint staging has been released")
      raise ArgumentError.new("Invalid checkpoint staging") unless raw.valid?
      # Validate all lifetimes before the first dispatch.
      @destinations.each_with_index do |buffer, i|
        MetalOps.check(buffer, @entries[i].info.n_elements)
        MetalOps.distinct(buffer, raw)
      end
      @entries.each_with_index do |entry, i|
        info = entry.info
        count = (info.data_bytes // info.dtype.bytes_per_element).to_i32
        tag = case info.dtype
              in .f32?  then 0_u32
              in .bf16? then 1_u32
              in .f16?  then 2_u32
              end
        encoder.set_pipeline(@pipeline)
        encoder.set_buffer(raw, 0, offset: entry.raw_offset)
        encoder.set_buffer(@destinations[i], 1, ML::Metal::BufferAccess::Write)
        encoder.set_value(count.to_u32, 2).set_value(tag, 3)
        encoder.dispatch_1d(count)
      end
    end

    def release : Nil
      @raw.try(&.release)
      @raw = nil
    end

    def finalize
      release
    end
  end
end
