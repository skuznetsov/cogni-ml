# GGUF file reader — loads model metadata and tensor data from GGUF v3 files.
#
# GGUF format (ggml):
#   [magic "GGUF"] [version u32] [n_tensors i64] [n_kv i64]
#   [kv_pairs...] [tensor_infos...] [aligned tensor data blob]
#
# Supports quantized types: F32, F16, Q4_K, Q5_K, Q6_K, Q8_0

require "./dequant"

module ML::GGUF
  MAGIC     = "GGUF"
  VERSION   =     3
  ALIGNMENT =    32

  # GGUF value types
  enum ValueType : UInt32
    UINT8   =  0
    INT8    =  1
    UINT16  =  2
    INT16   =  3
    UINT32  =  4
    INT32   =  5
    FLOAT32 =  6
    BOOL    =  7
    STRING  =  8
    ARRAY   =  9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12
  end

  # ggml tensor types (quantization formats)
  enum TensorType : UInt32
    F32  =  0
    F16  =  1
    Q4_0 =  2
    Q4_1 =  3
    Q5_0 =  6
    Q5_1 =  7
    Q8_0 =  8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

    # Bytes per block of QK_K=256 elements
    def block_bytes : Int32
      case self
      in .f32?  then 4      # per element, not block
      in .f16?  then 2      # per element
      in .q4_0? then 18     # 2 + 16
      in .q4_1? then 20     # 2 + 2 + 16
      in .q5_0? then 22     # 2 + 4 + 16
      in .q5_1? then 24     # 2 + 2 + 4 + 16
      in .q8_0? then 34     # 2 + 32
      in .q2_k? then 84
      in .q3_k? then 110
      in .q4_k? then 144    # 2 + 2 + 12 + 128
      in .q5_k? then 176    # 2 + 2 + 12 + 32 + 128
      in .q6_k? then 210    # 128 + 64 + 16 + 2
      in .q8_k? then 292
      end
    end

    # Elements per quantization block
    def block_elements : Int32
      case self
      in .f32?, .f16? then 1 # scalar, no blocking
      in .q4_0?, .q4_1?, .q5_0?, .q5_1?, .q8_0? then 32
      in .q2_k?, .q3_k?, .q4_k?, .q5_k?, .q6_k?, .q8_k? then 256 # QK_K
      end
    end

    def name : String
      case self
      in .f32?  then "F32"
      in .f16?  then "F16"
      in .q4_0? then "Q4_0"
      in .q4_1? then "Q4_1"
      in .q5_0? then "Q5_0"
      in .q5_1? then "Q5_1"
      in .q8_0? then "Q8_0"
      in .q2_k? then "Q2_K"
      in .q3_k? then "Q3_K"
      in .q4_k? then "Q4_K"
      in .q5_k? then "Q5_K"
      in .q6_k? then "Q6_K"
      in .q8_k? then "Q8_K"
      end
    end
  end

  # A value from the KV metadata section
  alias Value = String | UInt8 | Int8 | UInt16 | Int16 | UInt32 | Int32 |
                UInt64 | Int64 | Float32 | Float64 | Bool | Array(Value)

  # Tensor metadata (before data blob)
  struct TensorInfo
    getter name : String
    getter dims : Array(Int64)
    getter type : TensorType
    getter offset : UInt64  # Offset into data blob

    def initialize(@name, @dims, @type, @offset)
    end

    def n_elements : Int64
      dims.empty? ? 0_i64 : dims.reduce(1_i64) { |a, b| a * b }
    end

    # Total bytes of quantized data for this tensor
    def data_bytes : Int64
      n = n_elements
      if type.f32? || type.f16?
        n * type.block_bytes
      else
        blocks = (n + type.block_elements - 1) // type.block_elements
        blocks * type.block_bytes
      end
    end
  end

  # Parsed GGUF file — uses mmap for zero-copy tensor access
  class GGUFFile
    getter path : String
    getter version : UInt32
    getter metadata : Hash(String, Value)
    getter tensors : Array(TensorInfo)
    getter data_offset : Int64  # File offset where tensor data blob starts

    @io : File
    @mmap : Pointer(UInt8)?
    @mmap_size : UInt64 = 0

    def initialize(@path)
      @version = 0_u32
      @metadata = {} of String => Value
      @tensors = [] of TensorInfo
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

    # Read a tensor's data, dequantize to Float32.
    # Uses mmap (zero-copy) when available, falls back to read.
    def read_tensor_f32(info : TensorInfo) : Array(Float32)
      n = info.n_elements.to_i32
      raw = tensor_bytes(info)
      Dequant.dequantize(raw, info.type, n)
    end

    # Read raw tensor bytes — zero-copy slice from mmap
    def read_tensor_raw(info : TensorInfo) : Bytes
      tensor_bytes(info)
    end

    private def tensor_bytes(info : TensorInfo) : Bytes
      offset = data_offset + info.offset.to_i64
      size = info.data_bytes.to_i32
      if ptr = @mmap
        Bytes.new(ptr + offset, size, read_only: true)
      else
        @io.seek(offset)
        buf = Bytes.new(size)
        @io.read_fully(buf)
        buf
      end
    end

    private def setup_mmap
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
        @mmap = nil  # Fallback to read
      else
        @mmap = ptr.as(Pointer(UInt8))
        # Hint: sequential access for header, random for tensors
        LibC.madvise(ptr, @mmap_size, LibC::MADV_RANDOM)
      end
    rescue
      @mmap = nil
    end

    # Find tensor by name
    def tensor(name : String) : TensorInfo?
      @tensors.find { |t| t.name == name }
    end

    # Get string metadata
    def get_string(key : String) : String?
      @metadata[key]?.as?(String)
    end

    # Get integer metadata
    def get_int(key : String) : Int64?
      case v = @metadata[key]?
      when Int32  then v.to_i64
      when UInt32 then v.to_i64
      when Int64  then v
      when UInt64 then v.to_i64
      else             nil
      end
    end

    # Get float metadata
    def get_float(key : String) : Float64?
      case v = @metadata[key]?
      when Float32 then v.to_f64
      when Float64 then v
      else              nil
      end
    end

    private def parse_header
      # Magic
      magic = read_bytes(4)
      magic_str = String.new(magic)
      raise "Not a GGUF file: magic=#{magic_str.inspect}" unless magic_str == MAGIC

      # Version
      @version = read_u32
      raise "Unsupported GGUF version: #{@version}" unless @version >= 2 && @version <= 3

      # Counts
      n_tensors = read_i64
      n_kv = read_i64

      # KV pairs
      n_kv.times do
        key = read_string
        vtype = ValueType.new(read_u32)
        value = read_value(vtype)
        @metadata[key] = value
      end

      # Tensor infos
      n_tensors.times do
        name = read_string
        n_dims = read_u32.to_i32
        dims = Array(Int64).new(n_dims) { read_i64 }
        ttype = TensorType.new(read_u32)
        offset = read_u64
        @tensors << TensorInfo.new(name, dims, ttype, offset)
      end

      # Compute data blob offset (aligned)
      alignment = (@metadata["general.alignment"]?.as?(UInt32) || ALIGNMENT).to_i64
      pos = @io.pos
      @data_offset = (pos + alignment - 1) & ~(alignment - 1)
    end

    private def read_value(vtype : ValueType) : Value
      case vtype
      in .uint8?   then read_u8.as(Value)
      in .int8?    then read_i8.as(Value)
      in .uint16?  then read_u16.as(Value)
      in .int16?   then read_i16.as(Value)
      in .uint32?  then read_u32.as(Value)
      in .int32?   then read_i32.as(Value)
      in .uint64?  then read_u64.as(Value)
      in .int64?   then read_i64.as(Value)
      in .float32? then read_f32.as(Value)
      in .float64? then read_f64.as(Value)
      in .bool?    then (read_u8 != 0).as(Value)
      in .string?  then read_string.as(Value)
      in .array?
        atype = ValueType.new(read_u32)
        alen = read_u64.to_i32
        arr = Array(Value).new(alen) { read_value(atype) }
        arr.as(Value)
      end
    end

    private def read_string : String
      len = read_u64.to_i32
      return "" if len <= 0
      bytes = read_bytes(len)
      String.new(bytes)
    end

    private def read_bytes(n : Int32) : Bytes
      buf = Bytes.new(n)
      @io.read_fully(buf)
      buf
    end

    private def read_u8 : UInt8
      @io.read_byte.not_nil!
    end

    private def read_i8 : Int8
      read_u8.to_i8!
    end

    private def read_u16 : UInt16
      IO::ByteFormat::LittleEndian.decode(UInt16, read_bytes(2))
    end

    private def read_i16 : Int16
      IO::ByteFormat::LittleEndian.decode(Int16, read_bytes(2))
    end

    private def read_u32 : UInt32
      IO::ByteFormat::LittleEndian.decode(UInt32, read_bytes(4))
    end

    private def read_i32 : Int32
      IO::ByteFormat::LittleEndian.decode(Int32, read_bytes(4))
    end

    private def read_u64 : UInt64
      IO::ByteFormat::LittleEndian.decode(UInt64, read_bytes(8))
    end

    private def read_i64 : Int64
      IO::ByteFormat::LittleEndian.decode(Int64, read_bytes(8))
    end

    private def read_f32 : Float32
      IO::ByteFormat::LittleEndian.decode(Float32, read_bytes(4))
    end

    private def read_f64 : Float64
      IO::ByteFormat::LittleEndian.decode(Float64, read_bytes(8))
    end
  end
end
