require "digest/sha256"
require "json"
require "../../src/ml/gguf/qwen_qbit_gaussian_codec"
require "../../src/ml/metal/dispatch"
require "../../src/ml/metal/device"
require "../../src/ml/video/metal_ops"
require "../../src/ml/video/metal_mps_command_chain"

# This setup-only accessor keeps the nearest raw 8-bit centers single-sourced
# in the Gaussian codec. It is called while Plan builds its fixed codebook,
# never while encoding cache tensors.
module ML::GGUF::QwenQBitGaussianCodec
  def self.positive_levels_for_setup : Array(Float32)
    POSITIVE_LEVELS.dup
  end
end

module HyNomasCacheQBit
  extend self

  SCHEMA                    = "hy-nomas-cache-qbit-report-v1"
  POLICY                    = "hy-nomas-cache-qbit-row-error-v1"
  ROW_VALUES                = 128
  HEADER_BYTES              =  16
  PLANE_BYTES               = ROW_VALUES // 8
  MAX_PAYLOAD_BYTES_PER_ROW = ROW_VALUES * sizeof(Float32)
  STATS_WORDS               = 8
  STATS_BYTES               = STATS_WORDS * sizeof(UInt32)
  RAW_LEVEL_COUNT           = 128
  RECONSTRUCTION_LUT_COUNT  = 3 * 256
  CODEBOOK_FLOATS           = RAW_LEVEL_COUNT + RECONSTRUCTION_LUT_COUNT
  CODEBOOK_BYTES            = CODEBOOK_FLOATS * sizeof(Float32)
  MAX_DIAGNOSTIC_PAYLOAD    = 16 * 1024 * 1024

  SOURCE = {{ read_file("#{__DIR__}/hy_nomas_cache_qbit.metal") }}

  record Binding,
    name : String,
    buffer : ML::MetalBuffer,
    count : Int32

  record RowHeader,
    mean : Float32,
    sigma : Float32,
    payload_offset : UInt32,
    tier : UInt32

  record PackedDiagnostics,
    headers : Array(RowHeader),
    payload : Bytes

  def self.pipeline(function_suffix : String) : ML::Metal::ComputePipeline
    name = "hy_nomas_cache_qbit_#{function_suffix}"
    ML::Metal::PipelineCache.get(name) do
      ML::Metal::ComputePipeline.new(name, SOURCE, name)
    end
  end

  # A single in-place GPU roundtrip plan for a sequence of independent cache
  # bindings. Header and payload scratch is sized for only the largest record;
  # each binding is packed, restored, and then relinquishes that workspace
  # before the next binding is dispatched.
  class Plan
    HEADER_BYTES      = HyNomasCacheQBit::HEADER_BYTES
    ROW_VALUES        = HyNomasCacheQBit::ROW_VALUES
    PLANE_BYTES       = HyNomasCacheQBit::PLANE_BYTES
    STATS_BYTES       = HyNomasCacheQBit::STATS_BYTES
    CODEBOOK_BYTES    = HyNomasCacheQBit::CODEBOOK_BYTES
    CODEBOOK_FLOATS   = HyNomasCacheQBit::CODEBOOK_FLOATS
    MAX_PAYLOAD_BYTES = HyNomasCacheQBit::MAX_PAYLOAD_BYTES_PER_ROW

    @bindings : Array(Binding)
    @headers : ML::MetalBuffer
    @payload : ML::MetalBuffer
    @statistics : ML::MetalBuffer
    @codebook : ML::MetalBuffer
    @codebook_sha256 : String
    @max_rows : Int32
    @init_pipeline : ML::Metal::ComputePipeline
    @pack_pipeline : ML::Metal::ComputePipeline
    @mutate_pipeline : ML::Metal::ComputePipeline
    @unpack_pipeline : ML::Metal::ComputePipeline
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @encode_started = false
    @encode_complete = false
    @released = false

    def initialize(bindings : Array(Binding))
      raise ArgumentError.new("QBit cache plan requires at least one binding") if bindings.empty?
      raise "Metal is unavailable" unless ML::Metal::Device.init!

      @bindings = bindings.dup
      validate_bindings!
      @max_rows = @bindings.max_of(&.count) // ROW_VALUES
      max_count = @bindings.max_of(&.count)
      # Keep the byte cursor and all payload offsets representable in the
      # uint32 row header before allocating any MetalBuffer storage.
      self.class.named_bytes(max_count, @bindings.size.to_i32)

      # Pipeline compilation and all named MetalBuffer allocations happen at
      # setup time, before callers enter their measured encode boundary.
      @init_pipeline = HyNomasCacheQBit.pipeline("init")
      @pack_pipeline = HyNomasCacheQBit.pipeline("pack")
      @mutate_pipeline = HyNomasCacheQBit.pipeline("mutate_control")
      @unpack_pipeline = HyNomasCacheQBit.pipeline("unpack")

      header_bytes = @max_rows.to_i64 * HEADER_BYTES
      payload_bytes = @max_rows.to_i64 * MAX_PAYLOAD_BYTES
      statistics_bytes = @bindings.size.to_i64 * STATS_BYTES
      allocated = [] of ML::MetalBuffer
      headers = nil.as(ML::MetalBuffer?)
      payload = nil.as(ML::MetalBuffer?)
      statistics = nil.as(ML::MetalBuffer?)
      codebook = nil.as(ML::MetalBuffer?)
      @codebook_sha256 = ""
      begin
        headers = ML::MetalBuffer.new(header_bytes)
        allocated << headers
        payload = ML::MetalBuffer.new(payload_bytes)
        allocated << payload
        statistics = ML::MetalBuffer.new(statistics_bytes)
        allocated << statistics
        codebook_values = build_codebook
        codebook = ML::MetalBuffer.new(CODEBOOK_BYTES.to_i64)
        allocated << codebook
        codebook.write(codebook_values)
        @codebook_sha256 = Digest::SHA256.hexdigest(
          Bytes.new(codebook_values.to_unsafe.as(Pointer(UInt8)), CODEBOOK_BYTES)
        )
      rescue ex
        allocated.each(&.release)
        raise ex
      end
      @headers = headers.not_nil!
      @payload = payload.not_nil!
      @statistics = statistics.not_nil!
      @codebook = codebook.not_nil!
    end

    # Return the exact named MetalBuffer footprint for admission checks.
    # max_count is the largest binding size in F32 elements, not row count.
    def self.named_bytes(max_count : Int32, records : Int32) : Int64
      unless max_count > 0 && max_count % ROW_VALUES == 0
        raise ArgumentError.new("QBit cache max_count must be positive and divisible by 128")
      end
      raise ArgumentError.new("QBit cache records must be positive") unless records > 0

      rows = max_count.to_i64 // ROW_VALUES
      payload_bytes = rows * MAX_PAYLOAD_BYTES
      if payload_bytes > UInt32::MAX.to_i64
        raise ArgumentError.new("QBit cache payload cursor exceeds uint32 capacity")
      end
      (rows * HEADER_BYTES) + payload_bytes +
        (records.to_i64 * STATS_BYTES) + HyNomasCacheQBit::CODEBOOK_BYTES
    end

    # Total live MetalBuffer capacity owned by this plan, including the
    # retained stats slots and the fixed Gaussian codebook.
    def resident_bytes : Int64
      @headers.size + @payload.size + @statistics.size + @codebook.size
    end

    # Header and packed-payload workspace capacity, excluding the small
    # retained stats and fixed codebook allocations.
    def scratch_capacity_bytes : Int64
      @headers.size + @payload.size
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      encode_internal(chain, nil, nil, 0.0_f32)
    end

    # Probe-only sensitivity control. It changes the last record's selected
    # header mean and flips one stored plane bit before the GPU decoder runs.
    # Production callers use encode(chain), which never mutates the packed
    # representation.
    def encode_mutation_control(chain : ML::Metal::MPSCommandChain,
                                header_row : Int32,
                                bitplane_row : Int32,
                                header_delta : Float32) : Nil
      unless @bindings.size == 1
        raise ArgumentError.new("QBit mutation control requires one diagnostic binding")
      end
      rows = @bindings[0].count // ROW_VALUES
      unless header_row >= 0 && header_row < rows && bitplane_row >= 0 && bitplane_row < rows
        raise ArgumentError.new("QBit mutation control row is out of range")
      end
      unless header_delta.finite? && header_delta != 0.0_f32
        raise ArgumentError.new("QBit header mutation must be finite and nonzero")
      end
      encode_internal(chain, header_row, bitplane_row, header_delta)
    end

    # Read terminal diagnostics only after the exact chain which encoded the
    # plan has completed its single terminal wait. Invalid rows are included
    # in the returned report so callers can reject export without losing the
    # reason.
    def report_after_drain(chain : ML::Metal::MPSCommandChain) : JSON::Any
      check_drained_chain!(chain)
      stats = read_stats

      total_logical = 0_i64
      total_exact = 0_i64
      total_p4 = 0_i64
      total_p5 = 0_i64
      total_p8 = 0_i64
      total_f32 = 0_i64
      total_invalid = 0_i64

      report = JSON.build do |json|
        json.object do
          json.field "schema", HyNomasCacheQBit::SCHEMA
          json.field "policy" do
            json.object do
              json.field "name", HyNomasCacheQBit::POLICY
              json.field "semantics", "experimental quality selector; thresholds are not upstream exactness tolerances"
              json.field "row_values", ROW_VALUES
              json.field "header_bytes_per_row", HEADER_BYTES
              json.field "p4_max_error_over_sigma", 0.50_f32
              json.field "p4_rms_error_over_sigma", 0.12_f32
              json.field "p5_max_error_over_sigma", 0.25_f32
              json.field "p5_rms_error_over_sigma", 0.06_f32
              json.field "p8_max_error_over_sigma", 0.10_f32
              json.field "p8_rms_error_over_sigma", 0.02_f32
              json.field "escape", "exact-f32"
            end
          end
          json.field "codebook" do
            json.object do
              json.field "raw_centers", "ML::GGUF::QwenQBitGaussianCodec::POSITIVE_LEVELS"
              json.field "reconstruction", "ML::GGUF::QwenQBitGaussianCodec::reconstruct_raw_code"
              json.field "sha256", @codebook_sha256
            end
          end
          json.field "resident_bytes", resident_bytes
          json.field "scratch_capacity_bytes", scratch_capacity_bytes
          json.field "records" do
            json.array do
              @bindings.each_with_index do |binding, record_index|
                cursor = (record_index * HyNomasCacheQBit::STATS_WORDS)
                payload_bytes = stats[cursor].to_i64
                p4 = stats[cursor + 1].to_i64
                p5 = stats[cursor + 2].to_i64
                p8 = stats[cursor + 3].to_i64
                f32 = stats[cursor + 4].to_i64
                invalid = stats[cursor + 5].to_i64
                rows = binding.count.to_i64 // ROW_VALUES
                logical = rows * HEADER_BYTES + payload_bytes
                exact = binding.count.to_i64 * sizeof(Float32)

                total_logical += logical
                total_exact += exact
                total_p4 += p4
                total_p5 += p5
                total_p8 += p8
                total_f32 += f32
                total_invalid += invalid

                json.object do
                  json.field "name", binding.name
                  json.field "count", binding.count
                  json.field "rows", rows
                  json.field "logical_packed_bytes", logical
                  json.field "exact_bytes", exact
                  json.field "tier_counts" do
                    json.object do
                      json.field "p4", p4
                      json.field "p5", p5
                      json.field "p8", p8
                      json.field "f32", f32
                    end
                  end
                  json.field "invalid_rows", invalid
                end
              end
            end
          end
          json.field "logical_packed_bytes", total_logical
          json.field "exact_bytes", total_exact
          json.field "tier_counts" do
            json.object do
              json.field "p4", total_p4
              json.field "p5", total_p5
              json.field "p8", total_p8
              json.field "f32", total_f32
            end
          end
          json.field "invalid_rows", total_invalid
        end
      end
      JSON.parse(report)
    end

    # Probe-only, tiny readback of the final record's compact bytes. Production
    # paths consume report_after_drain only; this is never called from encode.
    def diagnostics_after_drain(chain : ML::Metal::MPSCommandChain) : PackedDiagnostics
      check_drained_chain!(chain)
      record_index = @bindings.size - 1
      binding = @bindings[record_index]
      rows = binding.count // ROW_VALUES
      stats = read_stats
      payload_size = stats[record_index * HyNomasCacheQBit::STATS_WORDS].to_i64
      unless payload_size >= 0 && payload_size <= @payload.size &&
             payload_size <= HyNomasCacheQBit::MAX_DIAGNOSTIC_PAYLOAD
        raise ArgumentError.new("QBit diagnostic payload exceeds the small-readback limit")
      end

      header_bytes = Bytes.new(rows * HEADER_BYTES, 0_u8)
      @headers.read_bytes(header_bytes.to_unsafe, header_bytes.size)
      headers = [] of RowHeader
      rows.times do |row|
        offset = row * HEADER_BYTES
        headers << RowHeader.new(
          read_f32(header_bytes, offset),
          read_f32(header_bytes, offset + sizeof(Float32)),
          read_u32(header_bytes, offset + 2 * sizeof(UInt32)),
          read_u32(header_bytes, offset + 3 * sizeof(UInt32)),
        )
      end

      payload = Bytes.new(payload_size.to_i, 0_u8)
      @payload.read_bytes(payload.to_unsafe, payload.size) unless payload.empty?
      PackedDiagnostics.new(headers, payload)
    end

    def release_after_drain(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if @encode_started
        active_chain = @last_chain
        unless chain && active_chain && active_chain.same?(chain) && chain.completed? && chain.wait_count == 1
          raise ArgumentError.new("QBit cache plan release requires its exact completed one-wait chain")
        end
      end
      @headers.release
      @payload.release
      @statistics.release
      @codebook.release
      @released = true
    end

    def self.validate_report!(report : JSON::Any) : Nil
      invalid_rows = report["invalid_rows"].as_i64
      raise ArgumentError.new("QBit cache report contains #{invalid_rows} invalid rows") unless invalid_rows == 0
    end

    private def validate_bindings! : Nil
      buffers = [] of ML::MetalBuffer
      @bindings.each do |binding|
        raise ArgumentError.new("QBit cache binding name must not be empty") if binding.name.empty?
        unless binding.count > 0 && binding.count % ROW_VALUES == 0
          raise ArgumentError.new("QBit cache binding count must be positive and divisible by 128")
        end
        expected_bytes = binding.count.to_i64 * sizeof(Float32)
        unless binding.buffer.valid? && binding.buffer.storage_mode.shared? && binding.buffer.size >= expected_bytes
          raise ArgumentError.new("QBit cache binding must be a live shared F32 buffer with sufficient capacity")
        end
        buffers.each do |other|
          if ML::Video::MetalOps.overlap?(binding.buffer, other)
            raise ArgumentError.new("QBit cache bindings must not alias")
          end
        end
        buffers << binding.buffer
      end
    end

    private def build_codebook : Array(Float32)
      values = Array(Float32).new(CODEBOOK_FLOATS, 0.0_f32)
      positive_levels = ML::GGUF::QwenQBitGaussianCodec.positive_levels_for_setup
      raise "Gaussian raw codebook size changed" unless positive_levels.size == HyNomasCacheQBit::RAW_LEVEL_COUNT
      positive_levels.each_with_index { |value, i| values[i] = value }

      {4, 5, 8}.each_with_index do |precision, tier_index|
        offset = HyNomasCacheQBit::RAW_LEVEL_COUNT + tier_index * 256
        256.times do |raw|
          values[offset + raw] = ML::GGUF::QwenQBitGaussianCodec.reconstruct_raw_code(raw.to_u8, precision)
        end
      end
      values
    end

    private def encode_internal(chain : ML::Metal::MPSCommandChain,
                                header_row : Int32?,
                                bitplane_row : Int32?,
                                header_delta : Float32) : Nil
      raise ArgumentError.new("QBit cache plan has been released") if @released
      raise ArgumentError.new("QBit cache plan can encode only once") if @encode_started
      @last_chain = chain
      encoder = chain.encoder
      @encode_started = true

      @bindings.each_with_index do |binding, record_index|
        rows = binding.count // ROW_VALUES
        stats_offset = record_index.to_i64 * STATS_BYTES

        encoder.set_pipeline(@init_pipeline)
          .set_buffer(@statistics, 0, ML::Metal::BufferAccess::ReadWrite, stats_offset)
          .dispatch_1d(1, 1)

        encoder.set_pipeline(@pack_pipeline)
          .set_buffer(binding.buffer, 0, ML::Metal::BufferAccess::Read)
          .set_buffer(@headers, 1, ML::Metal::BufferAccess::Write)
          .set_buffer(@payload, 2, ML::Metal::BufferAccess::Write)
          .set_buffer(@statistics, 3, ML::Metal::BufferAccess::ReadWrite, stats_offset)
          .set_buffer(@codebook, 4, ML::Metal::BufferAccess::Read)
          .set_value(rows.to_u32, 5)
          .dispatch_1d(rows)

        if record_index == @bindings.size - 1 && header_row && bitplane_row
          encoder.set_pipeline(@mutate_pipeline)
            .set_buffer(@headers, 0, ML::Metal::BufferAccess::ReadWrite)
            .set_buffer(@payload, 1, ML::Metal::BufferAccess::ReadWrite)
            .set_value(header_row.to_u32, 2)
            .set_value(bitplane_row.to_u32, 3)
            .set_value(header_delta, 4)
            .dispatch_1d(1, 1)
        end

        encoder.set_pipeline(@unpack_pipeline)
          .set_buffer(binding.buffer, 0, ML::Metal::BufferAccess::ReadWrite)
          .set_buffer(@headers, 1, ML::Metal::BufferAccess::Read)
          .set_buffer(@payload, 2, ML::Metal::BufferAccess::Read)
          .set_buffer(@codebook, 3, ML::Metal::BufferAccess::Read)
          .set_value(rows.to_u32, 4)
          .dispatch_1d(rows)
      end
      @encode_complete = true
    end

    private def check_drained_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("QBit cache plan has been released") if @released
      unless @encode_complete && @last_chain.try(&.same?(chain)) && chain.completed? && chain.wait_count == 1
        raise ArgumentError.new("QBit cache report requires its exact completed one-wait chain")
      end
    end

    private def read_stats : Array(UInt32)
      raw = Bytes.new(@statistics.size.to_i, 0_u8)
      @statistics.read_bytes(raw.to_unsafe, raw.size)
      words = Array(UInt32).new(raw.size // sizeof(UInt32), 0_u32)
      words.size.times do |index|
        words[index] = read_u32(raw, index * sizeof(UInt32))
      end
      words
    end

    private def read_u32(bytes : Bytes, offset : Int) : UInt32
      bytes[offset].to_u32 |
        (bytes[offset + 1].to_u32 << 8) |
        (bytes[offset + 2].to_u32 << 16) |
        (bytes[offset + 3].to_u32 << 24)
    end

    private def read_f32(bytes : Bytes, offset : Int) : Float32
      read_u32(bytes, offset).unsafe_as(Float32)
    end
  end
end
