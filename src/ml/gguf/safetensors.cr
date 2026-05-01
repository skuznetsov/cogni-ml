require "json"

require "./dequant"

module ML::GGUF
  enum SafeTensorDType
    BF16
    F16
    F32

    def self.parse(value : String) : SafeTensorDType
      case value
      when "BF16" then BF16
      when "F16"  then F16
      when "F32"  then F32
      else
        raise "safetensors: unsupported dtype #{value.inspect}"
      end
    end

    def bytes_per_element : Int32
      case self
      in .bf16?, .f16? then 2
      in .f32?         then 4
      end
    end
  end

  struct SafeTensorInfo
    getter name : String
    getter dtype : SafeTensorDType
    getter shape : Array(Int64)
    getter data_start : Int64
    getter data_end : Int64

    def initialize(@name, @dtype, @shape, @data_start, @data_end)
    end

    def n_elements : Int64
      shape.empty? ? 0_i64 : shape.reduce(1_i64) { |a, b| a * b }
    end

    def data_bytes : Int64
      data_end - data_start
    end

    def validate_size! : Nil
      expected = n_elements * dtype.bytes_per_element
      raise "safetensors: #{name} byte size #{data_bytes} != expected #{expected}" unless data_bytes == expected
    end
  end

  # Minimal safetensors reader for dense sidecars. It supports the BF16/F16/F32
  # tensors needed by Qwen3.6 MTP without pulling in a heavyweight dependency.
  class SafetensorsFile
    getter path : String
    getter metadata : Hash(String, String)
    getter tensors : Array(SafeTensorInfo)
    getter data_offset : Int64

    @io : File
    @mmap : Pointer(UInt8)?
    @mmap_size : UInt64 = 0_u64

    def initialize(@path : String)
      @metadata = {} of String => String
      @tensors = [] of SafeTensorInfo
      @data_offset = 0_i64
      @io = File.new(@path, "rb")
      parse_header
      setup_mmap
    end

    def close
      if ptr = @mmap
        LibC.munmap(ptr.as(Void*), @mmap_size)
        @mmap = nil
      end
      @io.close
    end

    def tensor(name : String) : SafeTensorInfo?
      @tensors.find { |t| t.name == name }
    end

    def tensor_bytes(info : SafeTensorInfo) : Bytes
      off = @data_offset + info.data_start
      size = info.data_bytes.to_i32
      if ptr = @mmap
        Bytes.new(ptr + off, size, read_only: true)
      else
        @io.seek(off)
        buf = Bytes.new(size)
        @io.read_fully(buf)
        buf
      end
    end

    def read_tensor_f32(info : SafeTensorInfo) : Array(Float32)
      raw = tensor_bytes(info)
      case info.dtype
      in .bf16?
        Array(Float32).new(info.n_elements.to_i32) do |i|
          lo = raw[i * 2].to_u32
          hi = raw[i * 2 + 1].to_u32
          ((hi << 24) | (lo << 16)).unsafe_as(Float32)
        end
      in .f16?
        Array(Float32).new(info.n_elements.to_i32) do |i|
          h = raw[i * 2].to_u16 | (raw[i * 2 + 1].to_u16 << 8)
          Dequant.fp16_to_f32(h)
        end
      in .f32?
        Array(Float32).new(info.n_elements.to_i32) do |i|
          IO::ByteFormat::LittleEndian.decode(Float32, raw[i * 4, 4])
        end
      end
    end

    private def parse_header : Nil
      len_buf = Bytes.new(8)
      @io.read_fully(len_buf)
      header_len = IO::ByteFormat::LittleEndian.decode(UInt64, len_buf).to_i64
      raise "safetensors: invalid header length #{header_len}" if header_len <= 0
      header = Bytes.new(header_len.to_i32)
      @io.read_fully(header)
      @data_offset = 8_i64 + header_len

      root = JSON.parse(String.new(header)).as_h
      root.each do |name, value|
        if name == "__metadata__"
          value.as_h.each do |k, v|
            @metadata[k] = v.as_s
          end
          next
        end

        h = value.as_h
        dtype = SafeTensorDType.parse(h["dtype"].as_s)
        shape = h["shape"].as_a.map(&.as_i64)
        offsets = h["data_offsets"].as_a.map(&.as_i64)
        raise "safetensors: #{name} has invalid offsets" unless offsets.size == 2
        info = SafeTensorInfo.new(name, dtype, shape, offsets[0], offsets[1])
        info.validate_size!
        @tensors << info
      end
    end

    private def setup_mmap : Nil
      @mmap_size = File.size(@path).to_u64
      ptr = LibC.mmap(
        Pointer(Void).null,
        @mmap_size,
        LibC::PROT_READ,
        LibC::MAP_PRIVATE,
        @io.fd,
        0
      )
      if ptr.address == LibC::MAP_FAILED.address
        @mmap = nil
      else
        @mmap = ptr.as(Pointer(UInt8))
        LibC.madvise(ptr, @mmap_size, LibC::MADV_RANDOM)
      end
    rescue
      @mmap = nil
    end
  end
end
