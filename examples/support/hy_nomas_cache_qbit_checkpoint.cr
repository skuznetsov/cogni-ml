require "digest/sha256"
require "json"
require "set"
require "../../src/ml/video/hy_nomas_resume_bundle"
require "./hy_nomas_cache_qbit"

# Checkpoint-only compact serialization over the existing Gaussian QBit codec.
# Disk payloads are compact and canonical; live GPU slots remain strided at
# 512 bytes per row so saving adds only headers, counters, and the codebook.
module HyNomasCacheQBitCheckpoint
  extend self

  SCHEMA                  = "hy-nomas-cache-qbit-checkpoint-v1"
  MANIFEST_SCHEMA         = "hy-nomas-cache-qbit-checkpoint-manifest-v1"
  MANIFEST_NAME           = "codec.manifest"
  LAYOUT                  = "strided-gpu-512-compact-disk-v1"
  ROW_SLOT_BYTES          = HyNomasCacheQBit::MAX_PAYLOAD_BYTES_PER_ROW
  ROW_VALUES              = HyNomasCacheQBit::ROW_VALUES
  HEADER_BYTES            = HyNomasCacheQBit::HEADER_BYTES
  STATS_WORDS             = HyNomasCacheQBit::STATS_WORDS
  STATS_BYTES             = HyNomasCacheQBit::STATS_BYTES
  CODEBOOK_BYTES          = HyNomasCacheQBit::CODEBOOK_BYTES
  GROUP_ROWS              = 4096
  GROUP_HEADER_BYTES      = GROUP_ROWS * HEADER_BYTES
  GROUP_PAYLOAD_MAX_BYTES = GROUP_ROWS * ROW_SLOT_BYTES
  MANIFEST_MAX_BYTES      = 64 * 1024

  record RowHeader, mean : Float32, sigma : Float32,
    payload_offset : UInt32, tier : UInt32

  record RecordStats, name : String, count : Int32, rows : Int32,
    payload_bytes : Int64, p4 : Int64, p5 : Int64, p8 : Int64,
    f32 : Int64, invalid_rows : Int64

  record RecordLayout, binding : HyNomasCacheQBit::Binding,
    rows : Int32, header_row_base : UInt32, stats_offset : Int64

  def identity_json_for_setup : String
    JSON.build do |json|
      json.object do
        json.field "schema", SCHEMA
        json.field "policy", HyNomasCacheQBit::POLICY
        json.field "row_values", ROW_VALUES
        json.field "header_bytes_per_row", HEADER_BYTES
        json.field "gpu_payload_slot_bytes_per_row", ROW_SLOT_BYTES
        json.field "disk_layout", LAYOUT
        json.field "p4_max_error_over_sigma", 0.50_f32
        json.field "p4_rms_error_over_sigma", 0.12_f32
        json.field "p5_max_error_over_sigma", 0.25_f32
        json.field "p5_rms_error_over_sigma", 0.06_f32
        json.field "p8_max_error_over_sigma", 0.10_f32
        json.field "p8_rms_error_over_sigma", 0.02_f32
        json.field "escape", "exact-f32"
        json.field "raw_centers", "ML::GGUF::QwenQBitGaussianCodec::POSITIVE_LEVELS"
        json.field "reconstruction", "ML::GGUF::QwenQBitGaussianCodec::reconstruct_raw_code"
        json.field "codebook_sha256", HyNomasCacheQBit.codebook_sha256_for_setup
      end
    end
  end

  def contract_for(base_contract : String) : String
    ML::Video::HyNomasResumeBundle.validate_contract!(base_contract)
    identity = identity_json_for_setup
    JSON.build do |json|
      json.object do
        json.field "schema", SCHEMA
        json.field "base_contract", base_contract
        json.field "identity_json", identity
      end
    end
  end

  def maximum_inventory(logical_inventory : Hash(String, Int64)) : Hash(String, Int64)
    entries, _total_bytes = ML::Video::HyNomasResumeBundle.validate_inventory!(logical_inventory)
    raise ArgumentError.new("QBit checkpoint logical inventory is empty") if entries.empty?
    raise ArgumentError.new("QBit checkpoint inventory reserves its manifest name") if logical_inventory.has_key?(MANIFEST_NAME)

    maximum = {} of String => Int64
    logical_inventory.each do |name, exact_bytes|
      if name == "timeline"
        maximum[name] = exact_bytes
      else
        unless exact_bytes > 0 && exact_bytes % ROW_SLOT_BYTES == 0
          raise ArgumentError.new("QBit checkpoint cache inventory must contain whole 128-value rows for #{name}")
        end
        rows = exact_bytes // ROW_SLOT_BYTES
        nbytes = checked_multiply!(rows, HEADER_BYTES + ROW_SLOT_BYTES, "maximum cache record")
        unless nbytes < Int32::MAX
          raise ArgumentError.new("QBit checkpoint record exceeds bounded bundle record size for #{name}")
        end
        maximum[name] = nbytes
      end
    end
    maximum[MANIFEST_NAME] = MANIFEST_MAX_BYTES
    maximum
  end

  def checked_multiply!(left : Int64, right : Int64, label : String) : Int64
    if left < 0 || right < 0 || (right > 0 && left > Int64::MAX // right)
      raise ArgumentError.new("QBit checkpoint #{label} size overflows Int64")
    end
    left * right
  end

  def payload_bytes_for_tier(tier : UInt32) : Int32
    case tier
    when  4_u32 then 64
    when  5_u32 then 80
    when  8_u32 then 128
    when 32_u32 then 512
    else             raise ArgumentError.new("QBit checkpoint row tier is invalid")
    end
  end

  def read_u32(bytes : Bytes, offset : Int) : UInt32
    bytes[offset].to_u32 |
      (bytes[offset + 1].to_u32 << 8) |
      (bytes[offset + 2].to_u32 << 16) |
      (bytes[offset + 3].to_u32 << 24)
  end

  def write_u32(bytes : Bytes, offset : Int, value : UInt32) : Nil
    bytes[offset] = (value & 0xff_u32).to_u8
    bytes[offset + 1] = ((value >> 8) & 0xff_u32).to_u8
    bytes[offset + 2] = ((value >> 16) & 0xff_u32).to_u8
    bytes[offset + 3] = ((value >> 24) & 0xff_u32).to_u8
  end

  def read_f32(bytes : Bytes, offset : Int) : Float32
    read_u32(bytes, offset).unsafe_as(Float32)
  end

  def write_f32(bytes : Bytes, offset : Int, value : Float32) : Nil
    write_u32(bytes, offset, value.unsafe_as(UInt32))
  end

  def row_header(bytes : Bytes, offset : Int) : RowHeader
    RowHeader.new(
      read_f32(bytes, offset),
      read_f32(bytes, offset + 4),
      read_u32(bytes, offset + 8),
      read_u32(bytes, offset + 12),
    )
  end

  def write_row_header(bytes : Bytes, offset : Int, header : RowHeader) : Nil
    write_f32(bytes, offset, header.mean)
    write_f32(bytes, offset + 4, header.sigma)
    write_u32(bytes, offset + 8, header.payload_offset)
    write_u32(bytes, offset + 12, header.tier)
  end

  def exact_keys!(hash : Hash(String, JSON::Any), keys : Set(String), label : String) : Nil
    unless hash.keys.to_set == keys
      raise ArgumentError.new("QBit checkpoint #{label} fields mismatch")
    end
  end

  def build_stats_json(records : Array(RecordStats)) : String
    logical_bytes = 0_i64
    exact_bytes = 0_i64
    invalid_rows = 0_i64
    tier_counts = {"p4" => 0_i64, "p5" => 0_i64, "p8" => 0_i64, "f32" => 0_i64}
    records.each do |record|
      logical_bytes += record.rows.to_i64 * HEADER_BYTES + record.payload_bytes
      exact_bytes += record.count.to_i64 * sizeof(Float32)
      invalid_rows += record.invalid_rows
      tier_counts["p4"] += record.p4
      tier_counts["p5"] += record.p5
      tier_counts["p8"] += record.p8
      tier_counts["f32"] += record.f32
    end

    JSON.build do |json|
      json.object do
        json.field "records" do
          json.array do
            records.each do |record|
              json.object do
                json.field "name", record.name
                json.field "count", record.count
                json.field "rows", record.rows
                json.field "logical_packed_bytes", record.rows.to_i64 * HEADER_BYTES + record.payload_bytes
                json.field "exact_bytes", record.count.to_i64 * sizeof(Float32)
                json.field "payload_bytes", record.payload_bytes
                json.field "tier_counts" do
                  json.object do
                    json.field "p4", record.p4
                    json.field "p5", record.p5
                    json.field "p8", record.p8
                    json.field "f32", record.f32
                  end
                end
                json.field "invalid_rows", record.invalid_rows
              end
            end
          end
        end
        json.field "logical_packed_bytes", logical_bytes
        json.field "exact_bytes", exact_bytes
        json.field "tier_counts" do
          json.object do
            json.field "p4", tier_counts["p4"]
            json.field "p5", tier_counts["p5"]
            json.field "p8", tier_counts["p8"]
            json.field "f32", tier_counts["f32"]
          end
        end
        json.field "invalid_rows", invalid_rows
      end
    end
  end

  def build_manifest(timeline_bytes : Int64, stats_json : String) : String
    JSON.build do |json|
      json.object do
        json.field "schema", MANIFEST_SCHEMA
        json.field "identity_json", identity_json_for_setup
        json.field "timeline_bytes", timeline_bytes
        json.field "stats_json", stats_json
      end
    end
  end

  def byte_count_for_tier(tier : UInt32) : Int64
    payload_bytes_for_tier(tier).to_i64
  end

  def self.validate_binding!(binding : HyNomasCacheQBit::Binding) : Nil
    raise ArgumentError.new("QBit checkpoint binding name must not be empty") if binding.name.empty?
    unless binding.count > 0 && binding.count % ROW_VALUES == 0
      raise ArgumentError.new("QBit checkpoint binding count must be positive and divisible by 128")
    end
    expected = binding.count.to_i64 * sizeof(Float32)
    unless binding.buffer.valid? && binding.buffer.storage_mode.shared? && binding.buffer.size >= expected
      raise ArgumentError.new("QBit checkpoint binding must be a live shared F32 buffer with sufficient capacity")
    end
  end

  def self.validate_timeline_binding!(binding : HyNomasCacheQBit::Binding) : Nil
    unless binding.name == "timeline" && binding.count > 0
      raise ArgumentError.new("QBit checkpoint timeline binding must be nonempty and named timeline")
    end
    expected = binding.count.to_i64 * sizeof(Float32)
    unless binding.buffer.valid? && binding.buffer.storage_mode.shared? && binding.buffer.size >= expected
      raise ArgumentError.new("QBit checkpoint timeline must be a live shared F32 buffer with sufficient capacity")
    end
  end

  class PackPlan
    getter codebook_sha256 : String
    ROW_SLOT_BYTES = HyNomasCacheQBitCheckpoint::ROW_SLOT_BYTES

    @bindings : Array(HyNomasCacheQBit::Binding)
    @records : Array(HyNomasCacheQBitCheckpoint::RecordLayout)
    @headers : ML::MetalBuffer
    @statistics : ML::MetalBuffer
    @codebook : ML::MetalBuffer
    @init_pipeline : ML::Metal::ComputePipeline
    @pack_pipeline : ML::Metal::ComputePipeline
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @encode_started = false
    @encode_complete = false
    @written = false
    @released = false
    @disk_bytes : Int64? = nil

    def initialize(bindings : Array(HyNomasCacheQBit::Binding))
      raise ArgumentError.new("QBit checkpoint pack plan requires cache bindings") if bindings.empty?
      raise "Metal is unavailable" unless ML::Metal::Device.init!
      @bindings = bindings.dup
      counts = @bindings.map(&.count)
      self.class.named_bytes(counts)
      names = Set(String).new
      buffers = [] of ML::MetalBuffer
      total_rows = 0_i64
      @records = [] of HyNomasCacheQBitCheckpoint::RecordLayout
      @bindings.each_with_index do |binding, index|
        HyNomasCacheQBitCheckpoint.validate_binding!(binding)
        raise ArgumentError.new("QBit checkpoint binding names must be unique") unless names.add?(binding.name)
        buffers.each do |other|
          if ML::Video::MetalOps.overlap?(binding.buffer, other)
            raise ArgumentError.new("QBit checkpoint cache bindings must not alias")
          end
        end
        rows = binding.count // HyNomasCacheQBit::ROW_VALUES
        @records << HyNomasCacheQBitCheckpoint::RecordLayout.new(
          binding, rows, total_rows.to_u32,
          index.to_i64 * HyNomasCacheQBit::STATS_BYTES
        )
        total_rows += rows
        buffers << binding.buffer
      end

      @init_pipeline = HyNomasCacheQBit.pipeline("init")
      @pack_pipeline = HyNomasCacheQBit.pipeline("pack_checkpoint")
      header_bytes = total_rows * HyNomasCacheQBit::HEADER_BYTES
      statistics_bytes = @bindings.size.to_i64 * HyNomasCacheQBit::STATS_BYTES
      allocated = [] of ML::MetalBuffer
      headers = nil.as(ML::MetalBuffer?)
      statistics = nil.as(ML::MetalBuffer?)
      codebook = nil.as(ML::MetalBuffer?)
      values = HyNomasCacheQBit.codebook_values_for_setup
      @codebook_sha256 = HyNomasCacheQBit.codebook_sha256_for_setup
      begin
        headers = ML::MetalBuffer.new(header_bytes)
        allocated << headers
        statistics = ML::MetalBuffer.new(statistics_bytes)
        allocated << statistics
        codebook = ML::MetalBuffer.new(HyNomasCacheQBit::CODEBOOK_BYTES.to_i64)
        allocated << codebook
        codebook.write(values)
      rescue ex
        allocated.each(&.release)
        raise ex
      end
      @headers = headers.not_nil!
      @statistics = statistics.not_nil!
      @codebook = codebook.not_nil!
    end

    def self.named_bytes(counts : Array(Int32)) : Int64
      raise ArgumentError.new("QBit checkpoint named bytes requires at least one record") if counts.empty?
      total_rows = 0_i64
      counts.each do |count|
        unless count > 0 && count % HyNomasCacheQBit::ROW_VALUES == 0
          raise ArgumentError.new("QBit checkpoint count must be positive and divisible by 128")
        end
        rows = count.to_i64 // HyNomasCacheQBit::ROW_VALUES
        maximum_record = rows * (HyNomasCacheQBit::HEADER_BYTES + ROW_SLOT_BYTES)
        if maximum_record >= Int32::MAX
          raise ArgumentError.new("QBit checkpoint record exceeds bounded bundle record size")
        end
        total_rows += rows
      end
      if total_rows > UInt32::MAX.to_i64 // 4
        raise ArgumentError.new("QBit checkpoint header row index exceeds uint32 capacity")
      end
      header_bytes = total_rows * HyNomasCacheQBit::HEADER_BYTES
      stats_bytes = counts.size.to_i64 * HyNomasCacheQBit::STATS_BYTES
      header_bytes + stats_bytes + HyNomasCacheQBit::CODEBOOK_BYTES
    end

    def resident_bytes : Int64
      @headers.size + @statistics.size + @codebook.size
    end

    def named_resident_bytes : Int64
      self.class.named_bytes(@bindings.map(&.count))
    end

    def encode_pack(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("QBit checkpoint pack plan has been released") if @released
      raise ArgumentError.new("QBit checkpoint pack can encode only once") if @encode_started
      @last_chain = chain
      @encode_started = true
      encoder = chain.encoder
      @records.each do |record|
        encoder.set_pipeline(@init_pipeline)
          .set_buffer(@statistics, 0, ML::Metal::BufferAccess::ReadWrite, record.stats_offset)
          .dispatch_1d(1, 1)
        encoder.set_pipeline(@pack_pipeline)
          .set_buffer(record.binding.buffer, 0, ML::Metal::BufferAccess::ReadWrite)
          .set_buffer(@headers, 1, ML::Metal::BufferAccess::Write)
          .set_buffer(@statistics, 2, ML::Metal::BufferAccess::ReadWrite, record.stats_offset)
          .set_buffer(@codebook, 3, ML::Metal::BufferAccess::Read)
          .set_value(record.rows.to_u32, 4)
          .set_value(record.header_row_base, 5)
          .dispatch_1d(record.rows)
      end
      @encode_complete = true
    end

    def report_after_drain(chain : ML::Metal::MPSCommandChain) : JSON::Any
      check_drained_chain!(chain)
      records = read_record_stats
      stats_json = HyNomasCacheQBitCheckpoint.build_stats_json(records)
      timeline_bytes = 0_i64
      manifest_bytes = HyNomasCacheQBitCheckpoint.build_manifest(timeline_bytes, stats_json).bytesize.to_i64
      predicted_disk_bytes = ML::Video::HyNomasResumeBundle::PAYLOAD_OFFSET.to_i64 +
                             records.sum { |record| record.rows.to_i64 * HyNomasCacheQBit::HEADER_BYTES + record.payload_bytes } +
                             manifest_bytes
      disk_bytes = @disk_bytes || predicted_disk_bytes
      JSON.parse(build_report_json(records, disk_bytes, !@disk_bytes.nil?))
    end

    def write_after_drain(chain : ML::Metal::MPSCommandChain, path : String,
                          base_contract : String,
                          timeline_binding : HyNomasCacheQBit::Binding) : String
      check_drained_chain!(chain)
      raise ArgumentError.new("QBit checkpoint was already written") if @written
      HyNomasCacheQBitCheckpoint.validate_timeline_binding!(timeline_binding)
      @bindings.each do |binding|
        if ML::Video::MetalOps.overlap?(binding.buffer, timeline_binding.buffer)
          raise ArgumentError.new("QBit checkpoint timeline must not alias cache bindings")
        end
      end

      records = read_record_stats
      invalid_rows = records.sum(&.invalid_rows)
      unless invalid_rows == 0
        raise ArgumentError.new("QBit checkpoint contains #{invalid_rows} invalid rows")
      end
      timeline_bytes = timeline_binding.count.to_i64 * sizeof(Float32)
      stats_json = HyNomasCacheQBitCheckpoint.build_stats_json(records)
      manifest = HyNomasCacheQBitCheckpoint.build_manifest(timeline_bytes, stats_json)
      descriptors = {"timeline"                                => timeline_bytes,
                     HyNomasCacheQBitCheckpoint::MANIFEST_NAME => manifest.bytesize.to_i64}
      records.each do |record|
        descriptors[record.name] = record.rows.to_i64 * HyNomasCacheQBit::HEADER_BYTES + record.payload_bytes
      end

      digest = ML::Video::HyNomasResumeBundle.write(path,
        HyNomasCacheQBitCheckpoint.contract_for(base_contract), descriptors) do |name, sink|
        if name == "timeline"
          write_buffer_record(sink, timeline_binding.buffer, timeline_bytes)
        elsif name == HyNomasCacheQBitCheckpoint::MANIFEST_NAME
          sink.write(manifest.to_slice)
        else
          record = (@records.find { |item| item.binding.name == name } ||
                    raise ArgumentError.new("QBit checkpoint has no cache binding for #{name}"))
          record_stats = (records.find { |item| item.name == name } ||
                          raise ArgumentError.new("QBit checkpoint stats are missing for #{name}"))
          write_compact_record(sink, record, record_stats)
        end
      end
      @disk_bytes = ML::Video::HyNomasResumeBundle::PAYLOAD_OFFSET.to_i64 + descriptors.values.sum
      @written = true
      digest
    end

    def release_after_drain(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if @encode_started
        active = @last_chain
        unless chain && active && active.same?(chain) && chain.completed? && chain.wait_count == 1
          raise ArgumentError.new("QBit checkpoint pack release requires its exact completed one-wait chain")
        end
      end
      @headers.release
      @statistics.release
      @codebook.release
      @released = true
    end

    private def read_record_stats : Array(HyNomasCacheQBitCheckpoint::RecordStats)
      raw = Bytes.new(@statistics.size.to_i)
      @statistics.read_bytes(raw.to_unsafe, raw.size)
      words = Array(UInt32).new(raw.size // sizeof(UInt32), 0_u32)
      words.size.times do |index|
        words[index] = HyNomasCacheQBitCheckpoint.read_u32(raw, index * sizeof(UInt32))
      end
      @records.map_with_index do |record, index|
        cursor = index * HyNomasCacheQBit::STATS_WORDS
        HyNomasCacheQBitCheckpoint::RecordStats.new(
          record.binding.name, record.binding.count, record.rows,
          words[cursor].to_i64, words[cursor + 1].to_i64,
          words[cursor + 2].to_i64, words[cursor + 3].to_i64,
          words[cursor + 4].to_i64, words[cursor + 5].to_i64
        )
      end
    end

    private def build_report_json(records : Array(HyNomasCacheQBitCheckpoint::RecordStats),
                                  disk_bytes : Int64, disk_bytes_known : Bool) : String
      stats = HyNomasCacheQBitCheckpoint.build_stats_json(records)
      JSON.build do |json|
        json.object do
          json.field "schema", HyNomasCacheQBitCheckpoint::SCHEMA
          json.field "identity_json", HyNomasCacheQBitCheckpoint.identity_json_for_setup
          json.field "layout", HyNomasCacheQBitCheckpoint::LAYOUT
          json.field "disk_bytes", disk_bytes
          json.field "disk_bytes_known", disk_bytes_known
          json.field "resident_bytes", resident_bytes
          json.field "payload_scratch_bytes", 0
          json.field "statistics_json", stats
          stats_hash = JSON.parse(stats).as_h
          JSON.parse(stats).as_h.each do |name, value|
            json.field name, value
          end
          json.field "logical_packed_kv_bytes", stats_hash["logical_packed_bytes"]
        end
      end
    end

    private def write_buffer_record(sink : IO, buffer : ML::MetalBuffer, nbytes : Int64) : Nil
      pointer = buffer.contents.as(Pointer(UInt8))
      chunk = Bytes.new(ML::Video::HyNomasResumeBundle::IO_CHUNK_BYTES)
      offset = 0_i64
      while offset < nbytes
        count = Math.min(chunk.size.to_i64, nbytes - offset).to_i
        chunk_slice = chunk[0, count]
        chunk_slice.copy_from(pointer + offset, count)
        sink.write(chunk_slice)
        offset += count
      end
    end

    private def write_compact_record(sink : IO,
                                     record : HyNomasCacheQBitCheckpoint::RecordLayout,
                                     record_stats : HyNomasCacheQBitCheckpoint::RecordStats) : Nil
      rows = record.rows.to_i64
      header_chunk = Bytes.new(HyNomasCacheQBitCheckpoint::GROUP_HEADER_BYTES)
      header_pointer = @headers.contents.as(Pointer(UInt8))
      compact_offset = 0_i64
      row_base = 0_i64
      while row_base < rows
        group_rows = Math.min(HyNomasCacheQBitCheckpoint::GROUP_ROWS.to_i64, rows - row_base).to_i
        nbytes = group_rows * HyNomasCacheQBit::HEADER_BYTES
        slice = header_chunk[0, nbytes]
        gpu_offset = (record.header_row_base.to_i64 + row_base) * HyNomasCacheQBit::HEADER_BYTES
        slice.copy_from(header_pointer + gpu_offset, nbytes)
        group_rows.times do |index|
          byte_offset = index * HyNomasCacheQBit::HEADER_BYTES
          header = HyNomasCacheQBitCheckpoint.row_header(slice, byte_offset)
          unless header.payload_offset.to_i64 == (row_base + index).to_i64 * HyNomasCacheQBitCheckpoint::ROW_SLOT_BYTES
            raise ArgumentError.new("QBit checkpoint GPU payload offset is invalid")
          end
          HyNomasCacheQBitCheckpoint.write_u32(slice, byte_offset + 8, compact_offset.to_u32)
          compact_offset += HyNomasCacheQBitCheckpoint.byte_count_for_tier(header.tier)
        end
        sink.write(slice)
        row_base += group_rows
      end
      unless compact_offset == record_stats.payload_bytes
        raise ArgumentError.new("QBit checkpoint packed payload size changed before write")
      end

      payload_pointer = record.binding.buffer.contents.as(Pointer(UInt8))
      output_chunk = Bytes.new(ML::Video::HyNomasResumeBundle::IO_CHUNK_BYTES)
      row_base = 0_i64
      written = 0
      written_total = 0_i64
      while row_base < rows
        group_rows = Math.min(HyNomasCacheQBitCheckpoint::GROUP_ROWS.to_i64, rows - row_base).to_i
        nbytes = group_rows * HyNomasCacheQBit::HEADER_BYTES
        slice = header_chunk[0, nbytes]
        gpu_offset = (record.header_row_base.to_i64 + row_base) * HyNomasCacheQBit::HEADER_BYTES
        slice.copy_from(header_pointer + gpu_offset, nbytes)
        group_rows.times do |index|
          header = HyNomasCacheQBitCheckpoint.row_header(slice, index * HyNomasCacheQBit::HEADER_BYTES)
          payload_bytes = HyNomasCacheQBitCheckpoint.payload_bytes_for_tier(header.tier)
          if written + payload_bytes > output_chunk.size
            sink.write(output_chunk[0, written])
            written_total += written
            written = 0
          end
          source_offset = (row_base + index).to_i64 * HyNomasCacheQBitCheckpoint::ROW_SLOT_BYTES
          output_chunk[written, payload_bytes].copy_from(payload_pointer + source_offset, payload_bytes)
          written += payload_bytes
        end
        row_base += group_rows
      end
      if written > 0
        sink.write(output_chunk[0, written])
        written_total += written
      end
      unless written_total == record_stats.payload_bytes
        raise ArgumentError.new("QBit checkpoint compact payload byte count mismatch")
      end
    end

    private def check_drained_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("QBit checkpoint pack plan has been released") if @released
      unless @encode_complete && @last_chain.try(&.same?(chain)) && chain.completed? && chain.wait_count == 1
        raise ArgumentError.new("QBit checkpoint pack requires its exact completed one-wait chain")
      end
    end
  end

  class RestorePlan
    @bindings : Array(HyNomasCacheQBit::Binding)
    @records : Array(HyNomasCacheQBitCheckpoint::RecordLayout)
    @headers : ML::MetalBuffer
    @codebook : ML::MetalBuffer
    @unpack_pipeline : ML::Metal::ComputePipeline
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @encode_started = false
    @encode_complete = false
    @released = false

    def initialize(bindings : Array(HyNomasCacheQBit::Binding),
                   preflight : HyNomasCacheQBitCheckpoint::Preflight,
                   timeline_binding : HyNomasCacheQBit::Binding)
      raise ArgumentError.new("QBit checkpoint restore plan requires cache bindings") if bindings.empty?
      raise "Metal is unavailable" unless ML::Metal::Device.init!
      @bindings = bindings.dup
      names = Set(String).new
      buffers = [] of ML::MetalBuffer
      total_rows = 0_i64
      @records = [] of HyNomasCacheQBitCheckpoint::RecordLayout
      counts = @bindings.map(&.count)
      self.class.named_bytes(counts)
      @bindings.each do |binding|
        HyNomasCacheQBitCheckpoint.validate_binding!(binding)
        raise ArgumentError.new("QBit checkpoint restore binding names must be unique") unless names.add?(binding.name)
        expected = (preflight.logical_inventory[binding.name]? ||
                    raise ArgumentError.new("QBit checkpoint has no cache record for #{binding.name}"))
        unless expected == binding.count.to_i64 * sizeof(Float32)
          raise ArgumentError.new("QBit checkpoint restore count mismatch for #{binding.name}")
        end
        buffers.each do |other|
          if ML::Video::MetalOps.overlap?(binding.buffer, other)
            raise ArgumentError.new("QBit checkpoint restore cache bindings must not alias")
          end
        end
        rows = binding.count // HyNomasCacheQBit::ROW_VALUES
        @records << HyNomasCacheQBitCheckpoint::RecordLayout.new(binding, rows,
          total_rows.to_u32, 0_i64)
        total_rows += rows
        buffers << binding.buffer
      end

      HyNomasCacheQBitCheckpoint.validate_timeline_binding!(timeline_binding)
      unless timeline_binding.count.to_i64 * sizeof(Float32) == preflight.timeline_bytes
        raise ArgumentError.new("QBit checkpoint restore timeline count mismatch")
      end
      buffers.each do |other|
        if ML::Video::MetalOps.overlap?(timeline_binding.buffer, other)
          raise ArgumentError.new("QBit checkpoint timeline must not alias cache bindings")
        end
      end
      expected_cache_names = preflight.logical_inventory.keys.reject { |name| name == "timeline" }.to_set
      unless expected_cache_names == names
        raise ArgumentError.new("QBit checkpoint restore bindings do not cover the logical cache inventory")
      end

      @unpack_pipeline = HyNomasCacheQBit.pipeline("unpack_checkpoint")
      allocated = [] of ML::MetalBuffer
      headers = nil.as(ML::MetalBuffer?)
      codebook = nil.as(ML::MetalBuffer?)
      values = HyNomasCacheQBit.codebook_values_for_setup
      begin
        headers = ML::MetalBuffer.new(total_rows * HyNomasCacheQBit::HEADER_BYTES)
        allocated << headers
        codebook = ML::MetalBuffer.new(HyNomasCacheQBit::CODEBOOK_BYTES.to_i64)
        allocated << codebook
        codebook.write(values)
        preflight.claim_restore!
        timeline_bytes = timeline_binding.count.to_i64 * sizeof(Float32)
        preflight.copy_timeline_to!(Bytes.new(timeline_binding.buffer.contents.as(Pointer(UInt8)), timeline_bytes.to_i))
        @records.each do |record|
          preflight.copy_cache_rows_to!(record.binding.name, record.binding, headers.not_nil!, record.header_row_base)
        end
        preflight.finish_restore!
      rescue ex
        allocated.each(&.release)
        raise ex
      end
      @headers = headers.not_nil!
      @codebook = codebook.not_nil!
    end

    def self.named_bytes(counts : Array(Int32)) : Int64
      pack_bytes = HyNomasCacheQBitCheckpoint::PackPlan.named_bytes(counts)
      stats_bytes = counts.size.to_i64 * HyNomasCacheQBit::STATS_BYTES
      pack_bytes - stats_bytes
    end

    def resident_bytes : Int64
      @headers.size + @codebook.size
    end

    def encode_restore(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("QBit checkpoint restore plan has been released") if @released
      raise ArgumentError.new("QBit checkpoint restore can encode only once") if @encode_started
      @last_chain = chain
      @encode_started = true
      encoder = chain.encoder
      @records.each do |record|
        encoder.set_pipeline(@unpack_pipeline)
          .set_buffer(record.binding.buffer, 0, ML::Metal::BufferAccess::ReadWrite)
          .set_buffer(@headers, 1, ML::Metal::BufferAccess::Read)
          .set_buffer(@codebook, 2, ML::Metal::BufferAccess::Read)
          .set_value(record.rows.to_u32, 3)
          .set_value(record.header_row_base, 4)
          .dispatch_1d(record.rows)
      end
      @encode_complete = true
    end

    def release_after_drain(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      return if @released
      if @encode_started
        active = @last_chain
        unless chain && active && active.same?(chain) && chain.completed? && chain.wait_count == 1
          raise ArgumentError.new("QBit checkpoint restore release requires its exact completed one-wait chain")
        end
      end
      @headers.release
      @codebook.release
      @released = true
    end
  end

  class Preflight
    getter path : String
    getter sha256 : String
    getter identity : JSON::Any
    getter stats : JSON::Any
    getter disk_bytes : Int64
    getter timeline_bytes : Int64
    getter logical_inventory : Hash(String, Int64)

    @reader : ML::Video::HyNomasResumeBundle::Reader
    @maximum_inventory : Hash(String, Int64)
    @restore_claimed = false
    @restore_failed = false
    @timeline_copied = false
    @restore_complete = false
    @copied_cache_records = Set(String).new
    @closed = false

    def self.open(path : String, trusted_sha256 : String, base_contract : String,
                  logical_inventory : Hash(String, Int64)) : Preflight
      maximum = HyNomasCacheQBitCheckpoint.maximum_inventory(logical_inventory)
      contract = HyNomasCacheQBitCheckpoint.contract_for(base_contract)
      reader = ML::Video::HyNomasResumeBundle.open_bounded(
        path, trusted_sha256, contract, maximum
      )
      begin
        new(reader, path, trusted_sha256, base_contract, logical_inventory, maximum)
      rescue ex
        reader.close
        raise ex
      end
    end

    def initialize(@reader : ML::Video::HyNomasResumeBundle::Reader,
                   @path : String, @sha256 : String, base_contract : String,
                   logical_inventory : Hash(String, Int64),
                   @maximum_inventory : Hash(String, Int64))
      @logical_inventory = logical_inventory.dup
      @identity = JSON.parse(HyNomasCacheQBitCheckpoint.identity_json_for_setup)
      @reader.validate_contract!(HyNomasCacheQBitCheckpoint.contract_for(base_contract),
        @maximum_inventory)
      actual = @reader.inventory
      unless actual.keys.to_set == @maximum_inventory.keys.to_set
        raise ArgumentError.new("QBit checkpoint actual inventory mismatch")
      end
      @timeline_bytes = (@logical_inventory["timeline"]? ||
                         raise ArgumentError.new("QBit checkpoint logical inventory has no timeline"))
      unless actual["timeline"]? == @timeline_bytes
        raise ArgumentError.new("QBit checkpoint timeline byte count mismatch")
      end
      manifest_size = (actual[HyNomasCacheQBitCheckpoint::MANIFEST_NAME]? ||
                       raise ArgumentError.new("QBit checkpoint manifest is missing"))
      unless manifest_size > 0 && manifest_size <= HyNomasCacheQBitCheckpoint::MANIFEST_MAX_BYTES
        raise ArgumentError.new("QBit checkpoint manifest size is invalid")
      end
      manifest_bytes = Bytes.new(manifest_size.to_i)
      @reader.copy_to!(HyNomasCacheQBitCheckpoint::MANIFEST_NAME, manifest_bytes)
      manifest = begin
        JSON.parse(String.new(manifest_bytes)).as_h
      rescue
        raise ArgumentError.new("QBit checkpoint manifest is malformed")
      end
      HyNomasCacheQBitCheckpoint.exact_keys!(manifest,
        Set{"schema", "identity_json", "timeline_bytes", "stats_json"}, "manifest")
      unless manifest["schema"].as_s == HyNomasCacheQBitCheckpoint::MANIFEST_SCHEMA
        raise ArgumentError.new("QBit checkpoint manifest schema mismatch")
      end
      unless manifest["identity_json"].as_s == HyNomasCacheQBitCheckpoint.identity_json_for_setup
        raise ArgumentError.new("QBit checkpoint codec identity mismatch")
      end
      unless manifest["timeline_bytes"].as_i64 == @timeline_bytes
        raise ArgumentError.new("QBit checkpoint manifest timeline size mismatch")
      end
      @stats = begin
        JSON.parse(manifest["stats_json"].as_s)
      rescue
        raise ArgumentError.new("QBit checkpoint manifest stats are malformed")
      end
      validate_stats!(actual)
      validate_cache_headers!(actual)
      @disk_bytes = File.info(@path, false).size
    end

    def validate_contract!(base_contract : String,
                           logical_inventory : Hash(String, Int64)) : Nil
      check_open!
      unless logical_inventory == @logical_inventory
        raise ArgumentError.new("QBit checkpoint logical inventory mismatch")
      end
      maximum = HyNomasCacheQBitCheckpoint.maximum_inventory(logical_inventory)
      unless maximum == @maximum_inventory
        raise ArgumentError.new("QBit checkpoint inventory bounds mismatch")
      end
      @reader.validate_contract!(HyNomasCacheQBitCheckpoint.contract_for(base_contract), maximum)
    end

    def close : Nil
      return if @closed
      @closed = true
      @reader.close
    end

    def finalize
      close
    end

    # A preflight can be consumed only once. RestorePlan calls this before it
    # copies any row into live Metal storage; a failed partial setup cannot be
    # retried against already-mutated buffers.
    def claim_restore! : Nil
      check_open!
      raise ArgumentError.new("QBit checkpoint restore has already been claimed") if @restore_claimed
      @restore_claimed = true
    end

    def finish_restore! : Nil
      check_restore_copy!
      expected_names = @logical_inventory.keys.reject { |name| name == "timeline" }.to_set
      unless @timeline_copied && @copied_cache_records == expected_names
        @restore_failed = true
        raise ArgumentError.new("QBit checkpoint restore setup is partial")
      end
      @restore_complete = true
    end

    def copy_timeline_to!(destination : Bytes) : Nil
      check_restore_copy!
      raise ArgumentError.new("QBit checkpoint timeline was already restored") if @timeline_copied
      unless destination.size.to_i64 == @timeline_bytes
        raise ArgumentError.new("QBit checkpoint timeline destination size mismatch")
      end
      @reader.copy_to!("timeline", destination)
      @timeline_copied = true
    rescue ex
      @restore_failed = true
      raise ex
    end

    def copy_cache_rows_to!(name : String, binding : HyNomasCacheQBit::Binding,
                            headers : ML::MetalBuffer, header_row_base : UInt32) : Nil
      check_restore_copy!
      raise ArgumentError.new("QBit checkpoint cache record was already restored: #{name}") unless @copied_cache_records.add?(name)
      rows = binding.count // HyNomasCacheQBit::ROW_VALUES
      expected = (@logical_inventory[name]? ||
                  raise ArgumentError.new("QBit checkpoint has no logical cache record #{name}"))
      unless expected == binding.count.to_i64 * sizeof(Float32)
        raise ArgumentError.new("QBit checkpoint cache binding size mismatch for #{name}")
      end

      total_rows = rows.to_i64
      header_region_bytes = total_rows * HyNomasCacheQBit::HEADER_BYTES
      cache_ptr = binding.buffer.contents.as(Pointer(UInt8))
      header_ptr = headers.contents.as(Pointer(UInt32))
      header_chunk = Bytes.new(HyNomasCacheQBitCheckpoint::GROUP_HEADER_BYTES)
      payload_chunk = Bytes.new(HyNomasCacheQBitCheckpoint::GROUP_PAYLOAD_MAX_BYTES)
      digest = Digest::SHA256.new
      row_base = 0_i64
      disk_payload_cursor = 0_i64

      # Consume and authenticate the canonical header region first. The GPU
      # headers are populated from these exact bytes; no second file read can
      # change the metadata that the later decoder will use.
      while row_base < total_rows
        group_rows = Math.min(HyNomasCacheQBitCheckpoint::GROUP_ROWS.to_i64,
          total_rows - row_base).to_i
        header_bytes = group_rows * HyNomasCacheQBit::HEADER_BYTES
        header_slice = header_chunk[0, header_bytes]
        @reader.read_range!(name, row_base * HyNomasCacheQBit::HEADER_BYTES, header_slice)
        digest.update(header_slice)

        group_rows.times do |index|
          header = HyNomasCacheQBitCheckpoint.row_header(header_slice, index * HyNomasCacheQBit::HEADER_BYTES)
          payload_bytes = HyNomasCacheQBitCheckpoint.payload_bytes_for_tier(header.tier)
          unless header.payload_offset.to_i64 == disk_payload_cursor
            raise ArgumentError.new("QBit checkpoint canonical payload offsets changed after preflight")
          end
          unless header.mean.finite? && header.sigma.finite? && header.sigma >= 0_f32
            raise ArgumentError.new("QBit checkpoint row moments changed after preflight")
          end
          disk_payload_cursor += payload_bytes

          gpu_header_index = header_row_base.to_i64 + row_base + index
          gpu_header = header_ptr + gpu_header_index * 4
          gpu_header[0] = header.mean.unsafe_as(UInt32)
          gpu_header[1] = header.sigma.unsafe_as(UInt32)
          gpu_header[2] = ((index.to_i64 + row_base) * HyNomasCacheQBitCheckpoint::ROW_SLOT_BYTES).to_u32
          gpu_header[3] = header.tier
        end
        row_base += group_rows
      end

      unless disk_payload_cursor == @reader.inventory[name] - header_region_bytes
        raise ArgumentError.new("QBit checkpoint cache payload coverage changed after preflight")
      end

      # Stream the compact payload region in canonical row order, hashing the
      # exact bytes copied into each live shared row. Tier information comes
      # from the already-authenticated GPU header bank rather than rereading
      # the on-disk header region.
      row_base = 0_i64
      disk_payload_cursor = 0_i64
      while row_base < total_rows
        group_rows = Math.min(HyNomasCacheQBitCheckpoint::GROUP_ROWS.to_i64,
          total_rows - row_base).to_i
        group_payload_bytes = 0
        group_rows.times do |index|
          gpu_header_index = header_row_base.to_i64 + row_base + index
          tier = (header_ptr + gpu_header_index * 4)[3]
          group_payload_bytes += HyNomasCacheQBitCheckpoint.payload_bytes_for_tier(tier)
        end

        payload_slice = payload_chunk[0, group_payload_bytes]
        @reader.read_range!(name, header_region_bytes + disk_payload_cursor, payload_slice)
        digest.update(payload_slice)
        local_payload_offset = 0
        group_rows.times do |index|
          gpu_header_index = header_row_base.to_i64 + row_base + index
          tier = (header_ptr + gpu_header_index * 4)[3]
          payload_bytes = HyNomasCacheQBitCheckpoint.payload_bytes_for_tier(tier)
          row_byte_offset = (row_base + index).to_i64 * HyNomasCacheQBitCheckpoint::ROW_SLOT_BYTES
          (cache_ptr + row_byte_offset).copy_from(payload_slice.to_unsafe + local_payload_offset, payload_bytes)
          local_payload_offset += payload_bytes
        end
        disk_payload_cursor += group_payload_bytes
        row_base += group_rows
      end

      @reader.verify_record_digest!(name, digest.hexfinal)
    rescue ex
      @restore_failed = true
      raise ex
    end

    private def validate_stats!(actual : Hash(String, Int64)) : Nil
      stats_hash = @stats.as_h
      HyNomasCacheQBitCheckpoint.exact_keys!(stats_hash,
        Set{"records", "logical_packed_bytes", "exact_bytes", "tier_counts", "invalid_rows"}, "stats")
      raise ArgumentError.new("QBit checkpoint contains invalid rows") unless stats_hash["invalid_rows"].as_i64 == 0

      record_stats = {} of String => JSON::Any
      stats_hash["records"].as_a.each do |entry_any|
        entry = entry_any.as_h
        HyNomasCacheQBitCheckpoint.exact_keys!(entry,
          Set{"name", "count", "rows", "logical_packed_bytes", "exact_bytes", "payload_bytes", "tier_counts", "invalid_rows"}, "record stats")
        name = entry["name"].as_s
        raise ArgumentError.new("QBit checkpoint duplicate stats record #{name}") if record_stats.has_key?(name)
        record_stats[name] = entry_any
      end

      cache_names = @logical_inventory.keys.reject { |name| name == "timeline" }.to_set
      unless record_stats.keys.to_set == cache_names
        raise ArgumentError.new("QBit checkpoint stats inventory mismatch")
      end

      logical_total = 0_i64
      exact_total = 0_i64
      total_tiers = {"p4" => 0_i64, "p5" => 0_i64, "p8" => 0_i64, "f32" => 0_i64}
      cache_names.each do |name|
        entry = record_stats[name].as_h
        exact_bytes = @logical_inventory[name]
        rows = exact_bytes // HyNomasCacheQBitCheckpoint::ROW_SLOT_BYTES
        actual_bytes = (actual[name]? ||
                        raise ArgumentError.new("QBit checkpoint cache record is missing: #{name}"))
        payload_bytes = entry["payload_bytes"].as_i64
        logical_bytes = entry["logical_packed_bytes"].as_i64
        entry_tiers = entry["tier_counts"].as_h
        HyNomasCacheQBitCheckpoint.exact_keys!(entry_tiers,
          Set{"p4", "p5", "p8", "f32"}, "record tier counts")
        unless entry["count"].as_i64 == rows * HyNomasCacheQBitCheckpoint::ROW_VALUES &&
               entry["rows"].as_i64 == rows && entry["exact_bytes"].as_i64 == exact_bytes &&
               entry["invalid_rows"].as_i64 == 0 && logical_bytes == actual_bytes &&
               payload_bytes == actual_bytes - rows * HyNomasCacheQBit::HEADER_BYTES
          raise ArgumentError.new("QBit checkpoint record stats mismatch for #{name}")
        end
        tier_total = 0_i64
        expected_payload = 0_i64
        {"p4" => 64_i64, "p5" => 80_i64, "p8" => 128_i64, "f32" => 512_i64}.each do |tier, bytes_per_row|
          count = entry_tiers[tier].as_i64
          raise ArgumentError.new("QBit checkpoint tier count is negative for #{name}") if count < 0
          tier_total += count
          expected_payload += count * bytes_per_row
          total_tiers[tier] += count
        end
        unless tier_total == rows && expected_payload == payload_bytes
          raise ArgumentError.new("QBit checkpoint tier payload totals mismatch for #{name}")
        end
        logical_total += logical_bytes
        exact_total += exact_bytes
      end

      unless stats_hash["logical_packed_bytes"].as_i64 == logical_total &&
             stats_hash["exact_bytes"].as_i64 == exact_total
        raise ArgumentError.new("QBit checkpoint aggregate byte totals mismatch")
      end
      aggregate_tiers = stats_hash["tier_counts"].as_h
      HyNomasCacheQBitCheckpoint.exact_keys!(aggregate_tiers,
        Set{"p4", "p5", "p8", "f32"}, "aggregate tier counts")
      total_tiers.each do |name, count|
        unless aggregate_tiers[name].as_i64 == count
          raise ArgumentError.new("QBit checkpoint aggregate tier count mismatch")
        end
      end
    end

    private def validate_cache_headers!(actual : Hash(String, Int64)) : Nil
      stats_by_name = @stats["records"].as_a.to_h { |entry| {entry["name"].as_s, entry} }
      @logical_inventory.each do |name, exact_bytes|
        next if name == "timeline"
        rows = exact_bytes // HyNomasCacheQBitCheckpoint::ROW_SLOT_BYTES
        header_region_bytes = rows * HyNomasCacheQBit::HEADER_BYTES
        payload_region_bytes = actual[name] - header_region_bytes
        raise ArgumentError.new("QBit checkpoint cache record is shorter than its headers") if payload_region_bytes < 0
        expected_stats = stats_by_name[name].as_h
        observed_tiers = {"p4" => 0_i64, "p5" => 0_i64, "p8" => 0_i64, "f32" => 0_i64}
        payload_cursor = 0_i64
        row_base = 0_i64
        header_chunk = Bytes.new(HyNomasCacheQBitCheckpoint::GROUP_HEADER_BYTES)
        payload_chunk = Bytes.new(HyNomasCacheQBitCheckpoint::GROUP_PAYLOAD_MAX_BYTES)

        while row_base < rows
          group_rows = Math.min(HyNomasCacheQBitCheckpoint::GROUP_ROWS.to_i64, rows - row_base).to_i
          header_bytes = group_rows * HyNomasCacheQBit::HEADER_BYTES
          header_slice = header_chunk[0, header_bytes]
          @reader.read_range!(name, row_base * HyNomasCacheQBit::HEADER_BYTES, header_slice)
          headers = Array(HyNomasCacheQBitCheckpoint::RowHeader).new(group_rows)
          group_payload_bytes = 0_i64

          group_rows.times do |index|
            header = HyNomasCacheQBitCheckpoint.row_header(header_slice, index * HyNomasCacheQBit::HEADER_BYTES)
            unless header.mean.finite? && header.sigma.finite? && header.sigma >= 0.0_f32
              raise ArgumentError.new("QBit checkpoint row moments are nonfinite or invalid in #{name}")
            end
            unless header.payload_offset.to_i64 == payload_cursor
              raise ArgumentError.new("QBit checkpoint payload offset is noncanonical in #{name}")
            end
            payload_size = HyNomasCacheQBitCheckpoint.payload_bytes_for_tier(header.tier)
            if header.tier != 4_u32 && header.tier != 32_u32 && header.sigma <= 0.0_f32
              raise ArgumentError.new("QBit checkpoint row sigma is invalid for its tier in #{name}")
            end
            tier_name = case header.tier
                        when 4_u32 then "p4"
                        when 5_u32 then "p5"
                        when 8_u32 then "p8"
                        else            "f32"
                        end
            observed_tiers[tier_name] += 1
            headers << header
            group_payload_bytes += payload_size
            payload_cursor += payload_size
          end

          payload_bytes = payload_chunk[0, group_payload_bytes.to_i]
          @reader.read_range!(name, header_region_bytes + payload_cursor - group_payload_bytes,
            payload_bytes)
          payload_offset = 0
          headers.each do |header|
            payload_size = HyNomasCacheQBitCheckpoint.payload_bytes_for_tier(header.tier)
            if header.tier == 32_u32
              128.times do |value_index|
                value = HyNomasCacheQBitCheckpoint.read_f32(payload_bytes, payload_offset + value_index * 4)
                raise ArgumentError.new("QBit checkpoint exact escape contains a nonfinite value in #{name}") unless value.finite?
              end
            end
            payload_offset += payload_size
          end
          row_base += group_rows
        end

        unless payload_cursor == payload_region_bytes && observed_tiers == expected_stats["tier_counts"].as_h.transform_values(&.as_i64)
          raise ArgumentError.new("QBit checkpoint row-header coverage mismatch for #{name}")
        end
      end
    end

    private def check_open! : Nil
      raise ArgumentError.new("QBit checkpoint preflight is closed") if @closed
    end

    private def check_restore_copy! : Nil
      check_open!
      unless @restore_claimed && !@restore_failed && !@restore_complete
        raise ArgumentError.new("QBit checkpoint restore setup is not writable")
      end
    end
  end
end
