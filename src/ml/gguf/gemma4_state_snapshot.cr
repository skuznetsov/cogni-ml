require "digest/sha256"
require "file_utils"
require "./gemma4_metal"

module ML::GGUF
  # Exact Gemma 4 resident K/V snapshot for session-prefix reuse.
  #
  # The authoritative state is raw F32 K/V. Optional H16 side-caches are
  # deliberately excluded because they are optimization artifacts, not truth.
  module Gemma4StateSnapshot
    extend self

    enum RecordKind : UInt8
      KCache
      VCache
    end

    record Record,
      layer : Int32,
      kind : RecordKind,
      kv_dim : Int32,
      bytes : Bytes

    record ArtifactInfo,
      path : String,
      sha256 : String,
      byte_size : Int64

    class Snapshot
      getter max_seq : Int32
      getter prefix_len : Int32
      getter layer_count : Int32
      getter records : Array(Record)

      def initialize(@max_seq : Int32, @prefix_len : Int32, @layer_count : Int32, @records : Array(Record))
        raise ArgumentError.new("Gemma4 snapshot max_seq must be positive") unless @max_seq > 0
        raise ArgumentError.new("Gemma4 snapshot prefix_len must be non-negative") unless @prefix_len >= 0
        raise ArgumentError.new("Gemma4 snapshot prefix_len exceeds max_seq") if @prefix_len > @max_seq
        raise ArgumentError.new("Gemma4 snapshot layer_count must be positive") unless @layer_count > 0
      end

      def byte_size : Int64
        @records.sum(0_i64) { |r| r.bytes.size.to_i64 }
      end
    end

    ARTIFACT_MAGIC   = Bytes[0x43, 0x47, 0x4b, 0x56] # "CGKV"
    ARTIFACT_VERSION = 1_u32

    {% if flag?(:cpu_only) %}
      def capture(state, prefix_len : Int32) : Snapshot
        raise "Gemma4 resident snapshots require Metal"
      end
    {% else %}
      def capture(state : Gemma4Metal::ResidentState, prefix_len : Int32) : Snapshot
        raise ArgumentError.new("Gemma4 snapshot prefix_len must be non-negative") unless prefix_len >= 0
        raise ArgumentError.new("Gemma4 snapshot prefix_len exceeds max_seq") if prefix_len > state.max_seq

        records = [] of Record
        state.layers.each_with_index do |layer, i|
          capture_pair(records, i, RecordKind::KCache, layer.kv_dim, state.max_seq, prefix_len, layer.k_cache_buf)
          capture_pair(records, i, RecordKind::VCache, layer.kv_dim, state.max_seq, prefix_len, layer.v_cache_buf)
        end
        Snapshot.new(state.max_seq, prefix_len, state.layers.size, records)
      end

      def restore_into(snapshot : Snapshot, state : Gemma4Metal::ResidentState) : Nil
        validate_snapshot_for_state(snapshot, state)

        snapshot.records.each do |record|
          raise ArgumentError.new("Gemma4 snapshot record layer out of range: #{record.layer}") if record.layer < 0 || record.layer >= state.layers.size

          layer = state.layers[record.layer]
          expected_payload_size = snapshot.prefix_len.to_i64 * layer.kv_dim * sizeof(Float32)
          raise ArgumentError.new("Gemma4 snapshot record kv_dim mismatch at layer #{record.layer}: #{record.kv_dim} != #{layer.kv_dim}") unless record.kv_dim == layer.kv_dim
          raise ArgumentError.new("Gemma4 snapshot record byte size mismatch") unless record.bytes.size.to_i64 == expected_payload_size

          next if record.bytes.empty?

          target = case record.kind
                   in RecordKind::KCache
                     layer.k_cache_buf
                   in RecordKind::VCache
                     layer.v_cache_buf
                   end
          target.write_bytes(record.bytes.to_unsafe, record.bytes.size)
        end
      end

      def restore(snapshot : Snapshot, kv_dims : Array(Int32)) : Gemma4Metal::ResidentState
        state = Gemma4Metal::ResidentState.new(kv_dims, snapshot.max_seq)
        restore_into(snapshot, state)
        state
      end
    {% end %}

    def write_artifact(snapshot : Snapshot, path : String) : ArtifactInfo
      bytes = encode_artifact_bytes(snapshot)
      sha = Digest::SHA256.hexdigest(bytes)
      if parent = Path[path].parent
        FileUtils.mkdir_p(parent.to_s)
      end
      File.open(path, "w") { |file| file.write(bytes) }
      ArtifactInfo.new(path, sha, bytes.size.to_i64)
    end

    def read_artifact(path : String, expected_sha256 : String? = nil) : Snapshot
      bytes = read_all_bytes(path)
      if expected = expected_sha256
        actual = Digest::SHA256.hexdigest(bytes)
        raise ArgumentError.new("Gemma4 state artifact checksum mismatch") unless actual == expected
      end
      decode_artifact_bytes(bytes)
    end

    def encode_artifact_bytes(snapshot : Snapshot) : Bytes
      io = IO::Memory.new
      io.write(ARTIFACT_MAGIC)
      io.write_bytes(ARTIFACT_VERSION, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.max_seq.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.prefix_len.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.layer_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.records.size.to_u32, IO::ByteFormat::LittleEndian)
      snapshot.records.each do |record|
        io.write_bytes(record.layer.to_u32, IO::ByteFormat::LittleEndian)
        io.write_byte(record.kind.value)
        io.write_byte(0_u8)
        io.write_bytes(0_u16, IO::ByteFormat::LittleEndian)
        io.write_bytes(record.kv_dim.to_u32, IO::ByteFormat::LittleEndian)
        io.write_bytes(record.bytes.size.to_u64, IO::ByteFormat::LittleEndian)
        io.write(record.bytes)
      end
      io.to_slice
    end

    def decode_artifact_bytes(bytes : Bytes) : Snapshot
      io = IO::Memory.new(bytes)
      magic = Bytes.new(ARTIFACT_MAGIC.size)
      io.read_fully(magic)
      raise ArgumentError.new("not a Gemma4 state artifact") unless magic == ARTIFACT_MAGIC

      version = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      raise ArgumentError.new("unsupported Gemma4 state artifact version: #{version}") unless version == ARTIFACT_VERSION

      max_seq = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      prefix_len = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      layer_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      record_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      records = Array(Record).new(record_count)

      record_count.times do
        layer = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
        raise ArgumentError.new("Gemma4 state artifact record layer out of range: #{layer}") if layer < 0 || layer >= layer_count

        kind = RecordKind.from_value(io.read_byte.not_nil!)
        reserved0 = io.read_byte.not_nil!
        reserved1 = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian)
        raise ArgumentError.new("corrupt Gemma4 state artifact record") unless reserved0 == 0_u8 && reserved1 == 0_u16

        kv_dim = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
        payload_size = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
        raise ArgumentError.new("Gemma4 state artifact payload too large") if payload_size > Int32::MAX

        payload = Bytes.new(payload_size.to_i)
        io.read_fully(payload)
        records << Record.new(layer, kind, kv_dim, payload)
      end

      raise ArgumentError.new("trailing bytes in Gemma4 state artifact") unless io.pos == bytes.size
      Snapshot.new(max_seq, prefix_len, layer_count, records)
    end

    private def read_all_bytes(path : String) : Bytes
      File.open(path, "r") do |file|
        bytes = Bytes.new(file.size.to_i)
        file.read_fully(bytes)
        bytes
      end
    end

    {% unless flag?(:cpu_only) %}
      private def capture_pair(records : Array(Record),
                               layer_index : Int32,
                               kind : RecordKind,
                               kv_dim : Int32,
                               max_seq : Int32,
                               prefix_len : Int32,
                               buffer : ML::MetalBuffer) : Nil
        raise ArgumentError.new("Gemma4 snapshot kv_dim must be positive") unless kv_dim > 0
        raise ArgumentError.new("Gemma4 snapshot buffer size mismatch") unless buffer.size == max_seq.to_i64 * kv_dim * sizeof(Float32)

        values = buffer.read(max_seq * kv_dim)
        live_values = prefix_len * kv_dim
        bytes = floats_to_bytes(values, live_values)
        records << Record.new(layer_index, kind, kv_dim, bytes)
      end

      private def validate_snapshot_for_state(snapshot : Snapshot, state : Gemma4Metal::ResidentState) : Nil
        raise ArgumentError.new("Gemma4 snapshot max_seq mismatch: #{snapshot.max_seq} != #{state.max_seq}") unless snapshot.max_seq == state.max_seq
        raise ArgumentError.new("Gemma4 snapshot layer count mismatch: #{snapshot.layer_count} != #{state.layers.size}") unless snapshot.layer_count == state.layers.size

        expected_records = snapshot.layer_count * 2
        raise ArgumentError.new("Gemma4 snapshot record count mismatch: #{snapshot.records.size} != #{expected_records}") unless snapshot.records.size == expected_records
      end
    {% end %}

    private def floats_to_bytes(values : Array(Float32), count : Int32) : Bytes
      raise ArgumentError.new("Gemma4 snapshot float count out of bounds") if count < 0 || count > values.size

      bytes = Bytes.new(count * sizeof(Float32))
      return bytes if bytes.empty?

      source = Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size)
      bytes.copy_from(source)
      bytes
    end
  end
end
