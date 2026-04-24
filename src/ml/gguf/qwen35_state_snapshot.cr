require "digest/sha256"
require "file_utils"
require "./qwen35_cpu"

module ML::GGUF
  # In-memory Qwen 3.5/3.6 decode-state snapshot.
  #
  # This is the correctness foundation for prompt-prefix KV cache. It captures
  # the active state owner (MetalBuffer when present, otherwise CPU arrays) and
  # restores into a fresh Qwen35CPU::State. Durable .qkv artifacts and the
  # pg_sorted_heap metadata index are intentionally separate layers.
  module Qwen35StateSnapshot
    extend self

    enum RecordKind : UInt8
      KCache
      VCache
      ConvState
      SsmState
    end

    record Record,
      layer : Int32,
      kind : RecordKind,
      bytes : Bytes,
      storage_mode : ML::StorageMode

    record ArtifactInfo,
      path : String,
      sha256 : String,
      byte_size : Int64

    class Snapshot
      getter max_seq : Int32
      getter layer_count : Int32
      getter positions : Array(Int32)
      getter records : Array(Record)

      def initialize(@max_seq : Int32, @layer_count : Int32, @positions : Array(Int32), @records : Array(Record))
        raise ArgumentError.new("position count mismatch: positions=#{@positions.size}, layers=#{@layer_count}") unless @positions.size == @layer_count
      end

      def byte_size : Int64
        @records.sum(0_i64) { |r| r.bytes.size.to_i64 }
      end
    end

    ARTIFACT_MAGIC   = Bytes[0x43, 0x51, 0x4b, 0x56] # "CQKV"
    ARTIFACT_VERSION = 1_u32

    def capture(state : Qwen35CPU::State) : Snapshot
      records = [] of Record
      positions = Array(Int32).new(state.layers.size)
      state.layers.each_with_index do |layer, i|
        positions << layer.position
        capture_pair(records, i, RecordKind::KCache, layer.k_cache_buf, layer.k_cache)
        capture_pair(records, i, RecordKind::VCache, layer.v_cache_buf, layer.v_cache)
        capture_pair(records, i, RecordKind::ConvState, layer.conv_state_buf, layer.conv_state)
        capture_pair(records, i, RecordKind::SsmState, layer.ssm_state_buf, layer.ssm_state)
      end
      Snapshot.new(state.max_seq, state.layers.size, positions, records)
    end

    def restore(snapshot : Snapshot, hp : Qwen35Hparams, prefer_metal : Bool = Qwen35Metal.available?) : Qwen35CPU::State
      raise ArgumentError.new("layer count mismatch: snapshot=#{snapshot.layer_count}, hp=#{hp.n_layer}") unless snapshot.layer_count == hp.n_layer

      state = Qwen35CPU::State.new(hp, max_seq: snapshot.max_seq)
      snapshot.positions.each_with_index do |position, i|
        state.layers[i].position = position
      end
      snapshot.records.each do |record|
        raise ArgumentError.new("state record layer out of range: #{record.layer}") if record.layer < 0 || record.layer >= state.layers.size

        layer = state.layers[record.layer]
        case record.kind
        in RecordKind::KCache
          if prefer_metal
            layer.k_cache_buf = buffer_from(record)
          else
            layer.k_cache = float_array_from(record.bytes)
          end
        in RecordKind::VCache
          if prefer_metal
            layer.v_cache_buf = buffer_from(record)
          else
            layer.v_cache = float_array_from(record.bytes)
          end
        in RecordKind::ConvState
          if prefer_metal
            layer.conv_state_buf = buffer_from(record)
          else
            layer.conv_state = float_array_from(record.bytes)
          end
        in RecordKind::SsmState
          if prefer_metal
            layer.ssm_state_buf = buffer_from(record)
          else
            layer.ssm_state = float_array_from(record.bytes)
          end
        end
      end
      state
    end

    def write_artifact(snapshot : Snapshot, path : String) : ArtifactInfo
      bytes = encode_artifact(snapshot)
      sha = Digest::SHA256.hexdigest(bytes)
      if parent = Path[path].parent
        FileUtils.mkdir_p(parent.to_s)
      end
      File.open(path, "w") { |file| file.write(bytes) }
      ArtifactInfo.new(path, sha, bytes.size.to_i64)
    end

    def read_artifact(path : String, expected_sha256 : String? = nil) : Snapshot
      bytes = read_all_bytes(path)
      sha = Digest::SHA256.hexdigest(bytes)
      if expected = expected_sha256
        raise ArgumentError.new("Qwen state artifact sha256 mismatch") unless sha == expected.downcase
      end
      decode_artifact(bytes)
    end

    private def capture_pair(records : Array(Record),
                             layer : Int32,
                             kind : RecordKind,
                             buf : ML::MetalBuffer?,
                             array : Array(Float32)?) : Nil
      if active_buf = buf
        bytes = Bytes.new(active_buf.size.to_i)
        active_buf.read_bytes(bytes.to_unsafe, bytes.size)
        records << Record.new(layer, kind, bytes, active_buf.storage_mode)
      elsif active_array = array
        records << Record.new(layer, kind, bytes_from(active_array), ML::StorageMode::Shared)
      end
    end

    private def bytes_from(values : Array(Float32)) : Bytes
      bytes = Bytes.new(values.size * sizeof(Float32))
      src = Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size)
      bytes.copy_from(src)
      bytes
    end

    private def float_array_from(bytes : Bytes) : Array(Float32)
      raise ArgumentError.new("state record byte size is not Float32-aligned: #{bytes.size}") unless bytes.size % sizeof(Float32) == 0

      count = bytes.size // sizeof(Float32)
      values = Array(Float32).new(count, 0.0_f32)
      dst = Slice.new(values.to_unsafe.as(Pointer(UInt8)), bytes.size)
      dst.copy_from(bytes)
      values
    end

    private def buffer_from(record : Record) : ML::MetalBuffer
      buf = ML::MetalBuffer.new(record.bytes.size.to_i64, record.storage_mode)
      buf.write_bytes(record.bytes.to_unsafe, record.bytes.size)
      buf
    end

    private def encode_artifact(snapshot : Snapshot) : Bytes
      io = IO::Memory.new
      io.write(ARTIFACT_MAGIC)
      io.write_bytes(ARTIFACT_VERSION, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.max_seq.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.layer_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.records.size.to_u32, IO::ByteFormat::LittleEndian)
      snapshot.positions.each do |position|
        io.write_bytes(position.to_u32, IO::ByteFormat::LittleEndian)
      end

      snapshot.records.each do |record|
        io.write_bytes(record.layer.to_u32, IO::ByteFormat::LittleEndian)
        io.write_byte(record.kind.value)
        io.write_byte(storage_mode_value(record.storage_mode))
        io.write_bytes(0_u16, IO::ByteFormat::LittleEndian)
        io.write_bytes(record.bytes.size.to_u64, IO::ByteFormat::LittleEndian)
        io.write(record.bytes)
      end

      io.to_slice
    end

    private def decode_artifact(bytes : Bytes) : Snapshot
      io = IO::Memory.new(bytes)
      magic = Bytes.new(ARTIFACT_MAGIC.size)
      io.read_fully(magic)
      raise ArgumentError.new("not a Qwen state artifact") unless magic == ARTIFACT_MAGIC

      version = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      raise ArgumentError.new("unsupported Qwen state artifact version: #{version}") unless version == ARTIFACT_VERSION

      max_seq = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      layer_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      record_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      positions = Array(Int32).new(layer_count)
      layer_count.times do
        positions << io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      end
      records = Array(Record).new(record_count)

      record_count.times do
        layer = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
        raise ArgumentError.new("Qwen state artifact record layer out of range: #{layer}") if layer < 0 || layer >= layer_count

        kind = RecordKind.from_value(io.read_byte.not_nil!)
        storage_mode = storage_mode_from(io.read_byte.not_nil!)
        reserved = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian)
        raise ArgumentError.new("corrupt Qwen state artifact record") unless reserved == 0_u16
        byte_size = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
        raise ArgumentError.new("Qwen state artifact record too large") if byte_size > Int32::MAX

        payload = Bytes.new(byte_size.to_i)
        io.read_fully(payload)
        records << Record.new(layer, kind, payload, storage_mode)
      end

      raise ArgumentError.new("trailing bytes in Qwen state artifact") unless io.pos == bytes.size
      Snapshot.new(max_seq, layer_count, positions, records)
    end

    private def read_all_bytes(path : String) : Bytes
      File.open(path, "r") do |file|
        bytes = Bytes.new(file.size.to_i)
        file.read_fully(bytes)
        bytes
      end
    end

    private def storage_mode_value(mode : ML::StorageMode) : UInt8
      mode.value.to_u8
    end

    private def storage_mode_from(value : UInt8) : ML::StorageMode
      case value.to_i32
      when ML::StorageMode::Shared.value
        ML::StorageMode::Shared
      when ML::StorageMode::Private.value
        ML::StorageMode::Private
      when ML::StorageMode::Managed.value
        ML::StorageMode::Managed
      else
        raise ArgumentError.new("unsupported Metal storage mode in Qwen state artifact: #{value}")
      end
    end
  end
end
