# Minimal Protocol Buffers wire encoder.
# Only the subset ONNX serialization needs: varint, length-delimited, fixed32/64.

require "io/memory"

module ML
  module ONNX
    class PBWriter
      WIRE_VARINT  = 0_u32
      WIRE_FIXED64 = 1_u32
      WIRE_LEN     = 2_u32
      WIRE_FIXED32 = 5_u32

      def initialize
        @io = IO::Memory.new
      end

      def to_slice : Bytes
        @io.to_slice
      end

      def size : Int32
        @io.bytesize
      end

      private def tag(field : Int32, wire : UInt32) : Nil
        write_varint(((field.to_u64) << 3) | wire.to_u64)
      end

      def write_varint(value : Int) : Nil
        v = value.to_u64
        while v >= 0x80
          @io.write_byte(((v & 0x7F) | 0x80).to_u8)
          v >>= 7
        end
        @io.write_byte(v.to_u8)
      end

      def int32(field : Int32, value : Int32) : Nil
        tag(field, WIRE_VARINT)
        write_varint(value)
      end

      def int64(field : Int32, value : Int64) : Nil
        tag(field, WIRE_VARINT)
        write_varint(value)
      end

      def string(field : Int32, value : String) : Nil
        b = value.to_slice
        tag(field, WIRE_LEN)
        write_varint(b.size)
        @io.write(b)
      end

      def bytes_field(field : Int32, value : Bytes) : Nil
        tag(field, WIRE_LEN)
        write_varint(value.size)
        @io.write(value)
      end

      def message(field : Int32, & : PBWriter ->) : Nil
        sub = PBWriter.new
        yield sub
        b = sub.to_slice
        tag(field, WIRE_LEN)
        write_varint(b.size)
        @io.write(b)
      end

      def float(field : Int32, value : Float32) : Nil
        tag(field, WIRE_FIXED32)
        buf = Bytes.new(4)
        IO::ByteFormat::LittleEndian.encode(value, buf)
        @io.write(buf)
      end

      def packed_int64(field : Int32, values : Array(Int64)) : Nil
        return if values.empty?
        sub = IO::Memory.new
        values.each do |v|
          x = v.to_u64
          while x >= 0x80
            sub.write_byte(((x & 0x7F) | 0x80).to_u8)
            x >>= 7
          end
          sub.write_byte(x.to_u8)
        end
        b = sub.to_slice
        tag(field, WIRE_LEN)
        write_varint(b.size)
        @io.write(b)
      end

      def raw_floats(field : Int32, values : Array(Float32)) : Nil
        tag(field, WIRE_LEN)
        write_varint(values.size * 4)
        buf = Bytes.new(4)
        values.each do |v|
          IO::ByteFormat::LittleEndian.encode(v, buf)
          @io.write(buf)
        end
      end
    end
  end
end
