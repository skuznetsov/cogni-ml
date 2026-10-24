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
    getter resident_dtype : WeightDType
    getter allow_f32_to_bf16 : Bool

    def entries : Array(PlannedWeight)
      @entries.map(&.snapshot)
    end

    def initialize(files : Array(CheckpointFile), bindings : Array(WeightBinding),
                   budget_bytes : Int64, @reserve_bytes : Int64 = 0_i64,
                   @resident_dtype : WeightDType = WeightDType::F32,
                   require_all : Bool = false,
                   allow_f32_to_bf16 : Bool = false)
      unless budget_bytes > 0 && reserve_bytes >= 0 && reserve_bytes <= budget_bytes && !bindings.empty?
        raise ArgumentError.new("Invalid checkpoint memory budget or bindings")
      end
      if allow_f32_to_bf16 && !resident_dtype.bf16?
        raise ArgumentError.new("F32-to-BF16 conversion requires BF16 residency")
      end
      @allow_f32_to_bf16 = allow_f32_to_bf16
      tensors = {} of String => {CheckpointFile, CheckpointTensor}
      files.each do |file|
        file.unchanged!
        file.tensors.each do |name, info|
          raise ArgumentError.new("Duplicate checkpoint tensor across shards #{name}") if tensors.has_key?(name)
          tensors[name] = {file, info}
        end
      end
      targets = Set(String).new
      sources = Set(String).new
      bindings.each do |binding|
        unless !binding.target.empty? && targets.add?(binding.target)
          raise ArgumentError.new("Duplicate or empty checkpoint destination")
        end
        unless sources.add?(binding.source)
          raise ArgumentError.new("Duplicate checkpoint source #{binding.source}")
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
        if resident_dtype.bf16?
          if info.dtype.bf16?
            # Preserve the direct-copy path for original BF16 tensors.
            @entries << PlannedWeight.new(binding.snapshot, file, info.snapshot, 0_i64)
          elsif allow_f32_to_bf16 && info.dtype.f32?
            # Only F32 sources need a conversion staging slice. Keep every
            # offset four-byte aligned for the raw F32 reader kernel.
            @staging_bytes = ((@staging_bytes + 3) // 4) * 4
            @entries << PlannedWeight.new(binding.snapshot, file, info.snapshot, @staging_bytes)
            @staging_bytes += info.data_bytes
          elsif allow_f32_to_bf16
            raise ArgumentError.new("F32-to-BF16 conversion requires F32 source #{binding.source}")
          else
            raise ArgumentError.new("Native BF16 residency requires original BF16 source #{binding.source}")
          end
        else
          @staging_bytes = ((@staging_bytes + 3) // 4) * 4
          @entries << PlannedWeight.new(binding.snapshot, file, info.snapshot, @staging_bytes)
          @staging_bytes += info.data_bytes
        end
        @destination_bytes += info.n_elements * resident_dtype.bytes_per_element
        if destination_bytes > budget_bytes ||
           staging_bytes > budget_bytes - destination_bytes ||
           reserve_bytes > budget_bytes - destination_bytes - staging_bytes
          raise ArgumentError.new("Checkpoint exceeds memory budget")
        end
      end
      if require_all && sources.size != tensors.size
        raise ArgumentError.new("Checkpoint bindings are not complete")
      end
      @peak_bytes = destination_bytes + staging_bytes + reserve_bytes
    end
  end

  # Arena owns destinations. F32 residency and opt-in F32-to-BF16 conversion
  # additionally own raw staging until the caller's final command completes.
  # Original BF16 tensors copy directly into final buffers at setup.
  class PreparedWeights
    SOURCE = {{ read_file("#{__DIR__}/kernels/weight_decode.metal") }}
    @raw : ML::MetalBuffer?
    @destinations = [] of ML::MetalBuffer
    @entries : Array(PlannedWeight)
    @pipeline : ML::Metal::ComputePipeline?
    @bf16_pipeline : ML::Metal::ComputePipeline?
    @resident_dtype : WeightDType
    @staging_bytes : Int64
    @arena : Arena
    @released = false

    def initialize(plan : WeightPlan, arena : Arena)
      @arena = arena
      @raw = nil
      @pipeline = nil
      @bf16_pipeline = nil
      @resident_dtype = plan.resident_dtype
      @staging_bytes = plan.staging_bytes
      @entries = plan.entries
      @entries.each do |entry|
        if arena.allocated?(entry.binding.target)
          raise ArgumentError.new("Checkpoint destination already exists #{entry.binding.target}")
        end
        entry.file.validate_tensor!(entry.info)
        needs_staging = !@resident_dtype.bf16? || entry.info.dtype.f32?
        unless !needs_staging ||
               (entry.raw_offset >= 0 && entry.raw_offset % 4 == 0 &&
               entry.raw_offset <= plan.staging_bytes && entry.info.data_bytes <= plan.staging_bytes - entry.raw_offset)
          raise ArgumentError.new("Invalid checkpoint staging range")
        end
      end
      unless arena.resident_bytes <= plan.reserve_bytes
        raise ArgumentError.new("Existing arena buffers exceed checkpoint reserve")
      end
      if !@resident_dtype.bf16?
        @pipeline = ML::Metal::PipelineCache.get("video_weight_decode") do
          ML::Metal::ComputePipeline.new("video_weight_decode", SOURCE)
        end
      elsif plan.allow_f32_to_bf16 && @entries.any? { |entry| entry.info.dtype.f32? }
        @bf16_pipeline = ML::Metal::PipelineCache.get("video_weight_encode_bf16") do
          ML::Metal::ComputePipeline.new("video_weight_encode_bf16", SOURCE)
        end
      end
      names = [] of String
      begin
        @raw = ML::MetalBuffer.new(plan.staging_bytes) if plan.staging_bytes > 0
        @entries.each do |entry|
          info = entry.info
          destination = arena.alloc_weight(entry.binding.target, info.n_elements.to_i32, @resident_dtype)
          names << entry.binding.target
          @destinations << destination
          pointer = if @resident_dtype.bf16? && entry.info.dtype.bf16?
                      destination.contents.as(UInt8*)
                    else
                      @raw.not_nil!.contents.as(UInt8*) + entry.raw_offset
                    end
          entry.file.copy_raw(info, Bytes.new(pointer, info.data_bytes.to_i32))
        end
      rescue ex
        release
        names.each { |name| arena.remove(name) }
        raise ex
      end
    end

    def encode(encoder : ML::Metal::ComputeEncoder) : Nil
      raise ArgumentError.new("Prepared checkpoint has been released") if @released
      # Validate all lifetimes before the first dispatch.
      @destinations.each_with_index do |buffer, i|
        entry = @entries[i]
        current = @arena.buffers[entry.binding.target]?
        unless current && current.same?(buffer) && @arena.dtype(entry.binding.target) == @resident_dtype
          raise ArgumentError.new("Prepared checkpoint destination was replaced #{entry.binding.target}")
        end
        MetalOps.check_storage(buffer, entry.info.n_elements, @resident_dtype)
      end
      if @staging_bytes > 0
        raw = @raw || raise ArgumentError.new("Checkpoint staging has been released")
        raise ArgumentError.new("Invalid checkpoint staging") unless raw.valid?
        @destinations.each { |buffer| MetalOps.distinct(buffer, raw) }
      end
      if @resident_dtype.bf16?
        return unless @staging_bytes > 0
        raw = @raw.not_nil!
        @entries.each_with_index do |entry, i|
          next unless entry.info.dtype.f32?
          count = entry.info.n_elements.to_i32
          encoder.set_pipeline(@bf16_pipeline.not_nil!)
          encoder.set_buffer(raw, 0, offset: entry.raw_offset)
          encoder.set_buffer(@destinations[i], 1, ML::Metal::BufferAccess::Write)
          encoder.set_value(count.to_u32, 2)
          encoder.dispatch_1d(count)
        end
      else
        raw = @raw || raise ArgumentError.new("Checkpoint staging has been released")
        @entries.each_with_index do |entry, i|
          info = entry.info
          count = (info.data_bytes // info.dtype.bytes_per_element).to_i32
          tag = case info.dtype
                in .f32?  then 0_u32
                in .bf16? then 1_u32
                in .f16?  then 2_u32
                end
          encoder.set_pipeline(@pipeline.not_nil!)
          encoder.set_buffer(raw, 0, offset: entry.raw_offset)
          encoder.set_buffer(@destinations[i], 1, ML::Metal::BufferAccess::Write)
          encoder.set_value(count.to_u32, 2).set_value(tag, 3)
          encoder.dispatch_1d(count)
        end
      end
    end

    def release : Nil
      @raw.try(&.release)
      @raw = nil
      @released = true
    end

    def finalize
      release
    end
  end
end
