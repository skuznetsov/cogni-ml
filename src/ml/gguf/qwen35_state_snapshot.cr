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

    enum RecordCodec : UInt8
      RawF32
      Bf16
      BlockI8
    end

    record Record,
      layer : Int32,
      kind : RecordKind,
      bytes : Bytes,
      storage_mode : ML::StorageMode

    record EncodedRecord,
      layer : Int32,
      kind : RecordKind,
      storage_mode : ML::StorageMode,
      codec : RecordCodec,
      original_byte_size : Int32,
      payload : Bytes

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

    class EncodedSnapshot
      getter max_seq : Int32
      getter layer_count : Int32
      getter positions : Array(Int32)
      getter records : Array(EncodedRecord)
      getter codec : RecordCodec
      getter codec_block : Int32

      def initialize(@max_seq : Int32,
                     @layer_count : Int32,
                     @positions : Array(Int32),
                     @records : Array(EncodedRecord),
                     @codec : RecordCodec,
                     @codec_block : Int32)
        raise ArgumentError.new("position count mismatch: positions=#{@positions.size}, layers=#{@layer_count}") unless @positions.size == @layer_count
      end

      def byte_size : Int64
        @records.sum(0_i64) { |r| r.original_byte_size.to_i64 }
      end

      def payload_byte_size : Int64
        @records.sum(0_i64) { |r| r.payload.size.to_i64 }
      end
    end

    ARTIFACT_MAGIC      = Bytes[0x43, 0x51, 0x4b, 0x56] # "CQKV"
    ARTIFACT_VERSION_V1 = 1_u32
    ARTIFACT_VERSION_V2 = 2_u32
    ARTIFACT_VERSION    = ARTIFACT_VERSION_V1

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

    def write_artifact(snapshot : Snapshot,
                       path : String,
                       artifact_codec : String? = nil,
                       artifact_codec_block : Int32? = nil) : ArtifactInfo
      codec = record_codec_for(artifact_codec)
      bytes = if codec.raw_f32?
                encode_artifact(snapshot)
              else
                encode_artifact_v2(snapshot, codec, artifact_codec_block)
              end
      sha = Digest::SHA256.hexdigest(bytes)
      if parent = Path[path].parent
        FileUtils.mkdir_p(parent.to_s)
      end
      File.open(path, "w") { |file| file.write(bytes) }
      ArtifactInfo.new(path, sha, bytes.size.to_i64)
    end

    def read_artifact(path : String,
                      expected_sha256 : String? = nil,
                      expected_codec : String? = nil,
                      expected_codec_block : Int32? = nil) : Snapshot
      decode_encoded_snapshot(read_artifact_encoded(
        path,
        expected_sha256: expected_sha256,
        expected_codec: expected_codec,
        expected_codec_block: expected_codec_block,
      ))
    end

    def read_artifact_encoded(path : String,
                              expected_sha256 : String? = nil,
                              expected_codec : String? = nil,
                              expected_codec_block : Int32? = nil) : EncodedSnapshot
      bytes = read_all_bytes(path)
      sha = Digest::SHA256.hexdigest(bytes)
      if expected = expected_sha256
        raise ArgumentError.new("Qwen state artifact sha256 mismatch") unless sha == expected.downcase
      end
      decode_artifact_encoded(bytes, expected_codec: expected_codec, expected_codec_block: expected_codec_block)
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
      io.write_bytes(ARTIFACT_VERSION_V1, IO::ByteFormat::LittleEndian)
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

    private def encode_artifact_v2(snapshot : Snapshot, codec : RecordCodec, block_size : Int32?) : Bytes
      block = artifact_block_size(codec, block_size)
      io = IO::Memory.new
      io.write(ARTIFACT_MAGIC)
      io.write_bytes(ARTIFACT_VERSION_V2, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.max_seq.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.layer_count.to_u32, IO::ByteFormat::LittleEndian)
      io.write_bytes(snapshot.records.size.to_u32, IO::ByteFormat::LittleEndian)
      io.write_byte(codec.value)
      io.write_byte(0_u8)
      io.write_bytes(0_u16, IO::ByteFormat::LittleEndian)
      io.write_bytes(block.to_u32, IO::ByteFormat::LittleEndian)
      snapshot.positions.each do |position|
        io.write_bytes(position.to_u32, IO::ByteFormat::LittleEndian)
      end

      snapshot.records.each do |record|
        record_codec = recurrent_record_kind?(record.kind) ? codec : RecordCodec::RawF32
        payload = encode_record_payload(record.bytes, record_codec, block)
        io.write_bytes(record.layer.to_u32, IO::ByteFormat::LittleEndian)
        io.write_byte(record.kind.value)
        io.write_byte(storage_mode_value(record.storage_mode))
        io.write_byte(record_codec.value)
        io.write_byte(0_u8)
        io.write_bytes(record.bytes.size.to_u64, IO::ByteFormat::LittleEndian)
        io.write_bytes(payload.size.to_u64, IO::ByteFormat::LittleEndian)
        io.write(payload)
      end

      io.to_slice
    end

    private def decode_artifact_encoded(bytes : Bytes,
                                        expected_codec : String? = nil,
                                        expected_codec_block : Int32? = nil) : EncodedSnapshot
      io = IO::Memory.new(bytes)
      magic = Bytes.new(ARTIFACT_MAGIC.size)
      io.read_fully(magic)
      raise ArgumentError.new("not a Qwen state artifact") unless magic == ARTIFACT_MAGIC

      version = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      raise ArgumentError.new("unsupported Qwen state artifact version: #{version}") unless version == ARTIFACT_VERSION_V1 || version == ARTIFACT_VERSION_V2

      max_seq = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      layer_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      record_count = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian)
      artifact_codec = RecordCodec::RawF32
      artifact_block = 0_i32
      if version == ARTIFACT_VERSION_V2
        artifact_codec = RecordCodec.from_value(io.read_byte.not_nil!)
        reserved0 = io.read_byte.not_nil!
        reserved1 = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian)
        raise ArgumentError.new("corrupt Qwen state artifact header") unless reserved0 == 0_u8 && reserved1 == 0_u16
        artifact_block = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      end
      validate_expected_artifact_codec(artifact_codec, artifact_block, expected_codec, expected_codec_block)

      positions = Array(Int32).new(layer_count)
      layer_count.times do
        positions << io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
      end
      records = Array(EncodedRecord).new(record_count)

      record_count.times do
        layer = io.read_bytes(UInt32, IO::ByteFormat::LittleEndian).to_i32
        raise ArgumentError.new("Qwen state artifact record layer out of range: #{layer}") if layer < 0 || layer >= layer_count

        kind = RecordKind.from_value(io.read_byte.not_nil!)
        storage_mode = storage_mode_from(io.read_byte.not_nil!)
        record_codec = RecordCodec::RawF32
        original_byte_size = 0_u64
        payload_byte_size = 0_u64
        if version == ARTIFACT_VERSION_V1
          reserved = io.read_bytes(UInt16, IO::ByteFormat::LittleEndian)
          raise ArgumentError.new("corrupt Qwen state artifact record") unless reserved == 0_u16
          original_byte_size = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
          payload_byte_size = original_byte_size
        else
          record_codec = RecordCodec.from_value(io.read_byte.not_nil!)
          reserved = io.read_byte.not_nil!
          raise ArgumentError.new("corrupt Qwen state artifact record") unless reserved == 0_u8
          original_byte_size = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
          payload_byte_size = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
          validate_record_codec(record_codec, artifact_codec)
        end
        raise ArgumentError.new("Qwen state artifact record too large") if original_byte_size > Int32::MAX || payload_byte_size > Int32::MAX

        payload = Bytes.new(payload_byte_size.to_i)
        io.read_fully(payload)
        records << EncodedRecord.new(layer, kind, storage_mode, record_codec, original_byte_size.to_i, payload)
      end

      raise ArgumentError.new("trailing bytes in Qwen state artifact") unless io.pos == bytes.size
      EncodedSnapshot.new(max_seq, layer_count, positions, records, artifact_codec, artifact_block)
    end

    private def decode_encoded_snapshot(encoded : EncodedSnapshot) : Snapshot
      records = encoded.records.map do |record|
        bytes = decode_record_payload(record.payload, record.codec, encoded.codec_block, record.original_byte_size)
        Record.new(record.layer, record.kind, bytes, record.storage_mode)
      end
      Snapshot.new(encoded.max_seq, encoded.layer_count, encoded.positions, records)
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

    private def record_codec_for(codec : String?) : RecordCodec
      case codec.try(&.downcase)
      when nil, "", "raw", "raw-fp32", "qkv-raw"
        RecordCodec::RawF32
      when "recurrent-bf16"
        RecordCodec::Bf16
      when "recurrent-int8"
        RecordCodec::BlockI8
      else
        raise ArgumentError.new("unsupported Qwen state artifact codec: #{codec.inspect}")
      end
    end

    private def recurrent_record_kind?(kind : RecordKind) : Bool
      case kind
      in RecordKind::ConvState, RecordKind::SsmState
        true
      in RecordKind::KCache, RecordKind::VCache
        false
      end
    end

    private def validate_record_codec(record_codec : RecordCodec, artifact_codec : RecordCodec) : Nil
      return if record_codec == RecordCodec::RawF32
      return if record_codec == artifact_codec

      raise ArgumentError.new("Qwen state artifact record codec mismatch")
    end

    private def codec_name(codec : RecordCodec) : String
      case codec
      in RecordCodec::RawF32
        "raw"
      in RecordCodec::Bf16
        "recurrent-bf16"
      in RecordCodec::BlockI8
        "recurrent-int8"
      end
    end

    private def artifact_block_size(codec : RecordCodec, block_size : Int32?) : Int32
      case codec
      in RecordCodec::RawF32, RecordCodec::Bf16
        0
      in RecordCodec::BlockI8
        block = block_size || 8
        raise ArgumentError.new("recurrent-int8 artifact block size must be positive") unless block > 0

        block
      end
    end

    private def validate_expected_artifact_codec(actual : RecordCodec,
                                                 actual_block : Int32,
                                                 expected_codec : String?,
                                                 expected_codec_block : Int32?) : Nil
      if expected_codec.nil?
        return if actual.raw_f32?

        raise ArgumentError.new("compressed Qwen state artifact requires explicit codec metadata")
      end

      expected = record_codec_for(expected_codec)
      raise ArgumentError.new("Qwen state artifact codec mismatch: expected #{codec_name(expected)}, found #{codec_name(actual)}") unless expected == actual

      if expected.block_i8? && expected_codec_block
        raise ArgumentError.new("Qwen state artifact codec block mismatch: expected #{expected_codec_block}, found #{actual_block}") unless expected_codec_block == actual_block
      end
    end

    private def encode_record_payload(bytes : Bytes, codec : RecordCodec, block_size : Int32) : Bytes
      return bytes if codec.raw_f32?
      raise ArgumentError.new("state record byte size is not Float32-aligned: #{bytes.size}") unless bytes.size % sizeof(Float32) == 0

      case codec
      in RecordCodec::RawF32
        bytes
      in RecordCodec::Bf16
        encode_bf16_payload(bytes)
      in RecordCodec::BlockI8
        encode_block_i8_payload(bytes, block_size)
      end
    end

    private def decode_record_payload(payload : Bytes, codec : RecordCodec, block_size : Int32, original_byte_size : Int32) : Bytes
      return payload if codec.raw_f32?
      raise ArgumentError.new("state record byte size is not Float32-aligned: #{original_byte_size}") unless original_byte_size % sizeof(Float32) == 0

      case codec
      in RecordCodec::RawF32
        payload
      in RecordCodec::Bf16
        decode_bf16_payload(payload, original_byte_size)
      in RecordCodec::BlockI8
        decode_block_i8_payload(payload, block_size, original_byte_size)
      end
    end

    private def encode_bf16_payload(bytes : Bytes) : Bytes
      output = Bytes.new(bytes.size // 2)
      (bytes.size // sizeof(Float32)).times do |i|
        bits = read_u32_le(bytes, i * sizeof(Float32))
        lsb = (bits >> 16) & 1_u32
        half = (((bits + 0x7fff_u32 + lsb) >> 16) & 0xffff_u32).to_u16
        write_u16_le(output, i * sizeof(UInt16), half)
      end
      output
    end

    private def decode_bf16_payload(payload : Bytes, original_byte_size : Int32) : Bytes
      expected = original_byte_size // 2
      raise ArgumentError.new("corrupt BF16 Qwen state artifact payload") unless payload.size == expected

      output = Bytes.new(original_byte_size)
      (original_byte_size // sizeof(Float32)).times do |i|
        half = read_u16_le(payload, i * sizeof(UInt16))
        write_u32_le(output, i * sizeof(Float32), half.to_u32 << 16)
      end
      output
    end

    private def encode_block_i8_payload(bytes : Bytes, block_size : Int32) : Bytes
      raise ArgumentError.new("recurrent-int8 artifact block size must be positive") unless block_size > 0

      count = bytes.size // sizeof(Float32)
      io = IO::Memory.new
      offset = 0
      while offset < count
        block_count = Math.min(block_size, count - offset)
        max_abs = 0.0_f32
        block_count.times do |j|
          value = read_f32_le(bytes, (offset + j) * sizeof(Float32)).abs
          max_abs = value if value > max_abs
        end
        scale = max_abs == 0.0_f32 ? 0.0_f32 : max_abs / 127.0_f32
        io.write_bytes(scale, IO::ByteFormat::LittleEndian)
        block_count.times do |j|
          value = read_f32_le(bytes, (offset + j) * sizeof(Float32))
          q = scale == 0.0_f32 ? 0 : (value / scale).round.to_i
          q = -127 if q < -127
          q = 127 if q > 127
          io.write_byte(q.to_i8.unsafe_as(UInt8))
        end
        offset += block_count
      end
      io.to_slice
    end

    private def decode_block_i8_payload(payload : Bytes, block_size : Int32, original_byte_size : Int32) : Bytes
      raise ArgumentError.new("recurrent-int8 artifact block size must be positive") unless block_size > 0

      count = original_byte_size // sizeof(Float32)
      output = Bytes.new(original_byte_size)
      io = IO::Memory.new(payload)
      offset = 0
      while offset < count
        raise ArgumentError.new("corrupt INT8 Qwen state artifact payload") if io.pos + sizeof(Float32) > payload.size

        scale = io.read_bytes(Float32, IO::ByteFormat::LittleEndian)
        block_count = Math.min(block_size, count - offset)
        block_count.times do |j|
          raise ArgumentError.new("corrupt INT8 Qwen state artifact payload") if io.pos >= payload.size

          q = io.read_byte.not_nil!.unsafe_as(Int8).to_i
          write_f32_le(output, (offset + j) * sizeof(Float32), q.to_f32 * scale)
        end
        offset += block_count
      end
      raise ArgumentError.new("trailing bytes in INT8 Qwen state artifact payload") unless io.pos == payload.size
      output
    end

    private def read_u16_le(bytes : Bytes, offset : Int32) : UInt16
      bytes[offset].to_u16 | (bytes[offset + 1].to_u16 << 8)
    end

    private def write_u16_le(bytes : Bytes, offset : Int32, value : UInt16) : Nil
      bytes[offset] = (value & 0xff).to_u8
      bytes[offset + 1] = ((value >> 8) & 0xff).to_u8
    end

    private def read_u32_le(bytes : Bytes, offset : Int32) : UInt32
      bytes[offset].to_u32 |
        (bytes[offset + 1].to_u32 << 8) |
        (bytes[offset + 2].to_u32 << 16) |
        (bytes[offset + 3].to_u32 << 24)
    end

    private def write_u32_le(bytes : Bytes, offset : Int32, value : UInt32) : Nil
      bytes[offset] = (value & 0xff).to_u8
      bytes[offset + 1] = ((value >> 8) & 0xff).to_u8
      bytes[offset + 2] = ((value >> 16) & 0xff).to_u8
      bytes[offset + 3] = ((value >> 24) & 0xff).to_u8
    end

    private def read_f32_le(bytes : Bytes, offset : Int32) : Float32
      read_u32_le(bytes, offset).unsafe_as(Float32)
    end

    private def write_f32_le(bytes : Bytes, offset : Int32, value : Float32) : Nil
      write_u32_le(bytes, offset, value.unsafe_as(UInt32))
    end
  end
end
