require "json"
require "c/sys/mman"
require "../../../ml/gguf/dequant"

# Native CPU correctness primitives for the MLX-converted DRAMA-3 checkpoint.
# MLX affine quantization is stored as low-bit-first UInt32 words; it is not GGUF.
module ML::Audio::Drama3
  alias Vector = Array(Float32)

  {% if flag?(:darwin) %}
    @[Link(framework: "Accelerate")]
    lib BLAS
      fun cblas_sgemv(layout : Int32, trans : Int32, m : Int32, n : Int32,
                      alpha : Float32, a : Float32*, lda : Int32, x : Float32*, incx : Int32,
                      beta : Float32, y : Float32*, incy : Int32) : Void
    end
  {% end %}

  class MappedFile
    getter size : Int64
    getter pointer : Pointer(UInt8)

    def initialize(path : String)
      @size = File.size(path)
      raise ArgumentError.new("empty safetensors file") if @size < 10
      @pointer = File.open(path, "rb") do |file|
        p = LibC.mmap(Pointer(Void).null, @size.to_u64, LibC::PROT_READ, LibC::MAP_PRIVATE, file.fd, 0)
        raise "mmap failed: #{path}" if p == LibC::MAP_FAILED
        p.as(UInt8*)
      end
    end

    def slice(offset : Int64, count : Int64) : Bytes
      raise ArgumentError.new("tensor range outside file") if offset < 0 || count < 0 || count > Int32::MAX || offset > @size - count
      Bytes.new(@pointer + offset, count.to_i32, read_only: true)
    end

    def finalize
      LibC.munmap(@pointer.as(Void*), @size.to_u64)
    end
  end

  class Tensor
    getter shape : Array(Int32)
    getter dtype : String
    getter count : Int64
    @values : Vector?
    @raw : Bytes?
    @owner : MappedFile?

    def initialize(@shape : Array(Int32), values : Vector)
      @dtype = "F32"
      @values = values
      @raw = nil
      @owner = nil
      validate_shape!
      @count = @shape.reduce(1_i64) { |n, d| n * d }
      raise ArgumentError.new("tensor element count mismatch") unless count == values.size
    end

    def initialize(@shape : Array(Int32), @dtype : String, raw : Bytes, @owner : MappedFile)
      @values = nil
      @raw = raw
      validate_shape!
      @count = @shape.reduce(1_i64) { |n, d| n * d }
      bytes = case @dtype
              when "BOOL"              then 1
              when "F32", "U32", "I32" then 4
              when "BF16", "F16"       then 2
              else                          raise ArgumentError.new("unsupported converted tensor dtype #{@dtype}")
              end
      raise ArgumentError.new("tensor byte count mismatch") unless raw.size.to_i64 == count * bytes
    end

    private def validate_shape!
      raise ArgumentError.new("invalid tensor shape") if @shape.empty? || @shape.any? { |n| n <= 0 }
    end

    def [](index : Int32) : Float32
      raise IndexError.new if index < 0 || index.to_i64 >= count
      return @values.not_nil![index] if @values
      raw = @raw.not_nil!
      case @dtype
      when "F32" then IO::ByteFormat::LittleEndian.decode(Float32, raw[index * 4, 4])
      when "BF16"
        (IO::ByteFormat::LittleEndian.decode(UInt16, raw[index * 2, 2]).to_u32 << 16).unsafe_as(Float32)
      when "F16" then ML::GGUF::Dequant.fp16_to_f32(IO::ByteFormat::LittleEndian.decode(UInt16, raw[index * 2, 2]))
      else            raise ArgumentError.new("integer tensor cannot be read as a floating value")
      end
    end

    def word(index : Int32) : UInt32
      raise ArgumentError.new("expected packed U32 tensor") unless @dtype == "U32"
      raise IndexError.new if index < 0 || index.to_i64 >= count
      IO::ByteFormat::LittleEndian.decode(UInt32, @raw.not_nil![index * 4, 4])
    end

    def values : Vector
      @values ||= values_slice(0, count.to_i32)
    end

    def values_slice(start : Int32, length : Int32) : Vector
      raise IndexError.new if start < 0 || length < 0 || start.to_i64 + length > count
      Array(Float32).new(length) { |i| self[start + i] }
    end
  end

  class Matrix
    getter rows : Int32
    getter cols : Int32
    @tensor : Tensor
    @scales : Tensor?
    @biases : Tensor?
    @bits : Int32
    @group_size : Int32
    @expert : Int32
    @dense : Vector?

    def initialize(@rows : Int32, @cols : Int32, values : Vector)
      raise ArgumentError.new("invalid matrix dimensions") if @rows <= 0 || @cols <= 0
      @tensor = Tensor.new([@rows, @cols], values)
      @dense = values
      @scales = @biases = nil
      @bits = 0
      @group_size = 64
      @expert = 0
    end

    def initialize(@tensor : Tensor, @scales : Tensor? = nil, @biases : Tensor? = nil,
                   @bits : Int32 = 4, @group_size : Int32 = 64, @expert : Int32 = 0)
      shape = @tensor.shape
      raise ArgumentError.new("matrix requires rank 2 or expert rank 3") unless shape.size == 2 || shape.size == 3
      raise ArgumentError.new("expert index outside tensor") if @expert < 0 || (shape.size == 2 && @expert != 0) || (shape.size == 3 && @expert >= shape[0])
      @rows = shape[-2]
      @cols = shape[-1]
      @dense = nil
      if @scales
        raise ArgumentError.new("unsupported affine quantization") unless (@bits == 4 || @bits == 8) && @group_size > 0 && @tensor.dtype == "U32" && @biases
        raise ArgumentError.new("quantization group must contain whole packed words") unless @group_size % (32 // @bits) == 0
        @cols *= 32 // @bits
        expected = shape.dup
        raise ArgumentError.new("quantization group does not divide columns") unless @cols % @group_size == 0
        expected[-1] = @cols // @group_size
        raise ArgumentError.new("invalid affine scale/bias shape") unless @scales.not_nil!.shape == expected && @biases.not_nil!.shape == expected
      elsif @tensor.dtype == "U32"
        raise ArgumentError.new("packed matrix requires affine scales and biases")
      end
    end

    def row(index : Int32) : Vector
      raise IndexError.new if index < 0 || index >= @rows
      unless @scales
        return @tensor.values_slice((@expert * @rows + index) * @cols, @cols)
      end
      values = Array(Float32).new(@cols, 0_f32)
      groups = @cols // @group_size
      groups.times do |g|
        off = (@expert * @rows + index) * groups + g
        scale = @scales.not_nil![off]
        bias = @biases.not_nil![off]
        @group_size.times do |j|
          col = g * @group_size + j
          word = @tensor.word((@expert * @rows + index) * (@cols // (32 // @bits)) + col // (32 // @bits))
          q = (word >> ((col % (32 // @bits)) * @bits)) & ((1_u32 << @bits) - 1)
          values[col] = scale * q.to_f32 + bias
        end
      end
      values
    end

    # Codec activations are dense time-by-channel matrices. Never implicitly
    # expand a packed model/expert matrix through this convenience accessor.
    def values : Vector
      raise ArgumentError.new("packed matrix has no dense values") if @scales
      @dense ||= @tensor.values_slice(@expert * @rows * @cols, @rows * @cols)
    end

    def matvec(x : Vector) : Vector
      raise ArgumentError.new("matrix input width #{x.size} != #{@cols}") unless x.size == @cols
      y = Array(Float32).new(@rows, 0_f32)
      unless @scales
        dense = @dense ||= @tensor.values_slice(@expert * @rows * @cols, @rows * @cols)
        {% if flag?(:darwin) %}
          BLAS.cblas_sgemv(101, 111, @rows, @cols, 1_f32, dense.to_unsafe, @cols, x.to_unsafe, 1, 0_f32, y.to_unsafe, 1)
        {% else %}
          @rows.times { |r| @cols.times { |c| y[r] += dense[r * @cols + c] * x[c] } }
        {% end %}
      else
        # Decode one affine group directly into the dot product. No expanded
        # expert matrix survives the call, keeping 35B model memory bounded.
        groups = @cols // @group_size
        per_word = 32 // @bits
        mask = (1_u32 << @bits) - 1
        @rows.times do |r|
          sum = 0_f32
          groups.times do |g|
            # Bound reduction depth to the affine group, rather than adding
            # thousands of products to one progressively larger F32 sum.
            group_sum = 0_f32
            off = (@expert * @rows + r) * groups + g
            scale = @scales.not_nil![off]
            bias = @biases.not_nil![off]
            first_col = g * @group_size
            (@group_size // per_word).times do |j|
              col = first_col + j * per_word
              word = @tensor.word((@expert * @rows + r) * (@cols // per_word) + col // per_word)
              per_word.times do |lane|
                q = (word >> (lane * @bits)) & mask
                group_sum += (scale * q.to_f32 + bias) * x[col + lane]
              end
            end
            sum += group_sum
          end
          y[r] = sum
        end
      end
      y
    end
  end

  class WeightStore
    @tensors : Hash(String, Tensor)
    @matrices = {} of String => Matrix
    @quantization : JSON::Any?

    def initialize(@tensors : Hash(String, Tensor))
      @quantization = nil
    end

    def initialize(path : String)
      @tensors = {} of String => Tensor
      @quantization = nil
      if File.file?(File.join(path, "config.json"))
        @quantization = JSON.parse(File.read(File.join(path, "config.json")))["quantization"]?
      end
      files = if File.file?(File.join(path, "model.safetensors.index.json"))
                index = JSON.parse(File.read(File.join(path, "model.safetensors.index.json")))
                index["weight_map"].as_h.values.map(&.as_s).uniq
              else
                ["model.safetensors"]
              end
      files.each do |name|
        raise ArgumentError.new("invalid shard path") unless File.basename(name) == name
        load_file(File.join(path, name))
      end
      codec = File.join(path, "codec.safetensors")
      load_file(codec, "codec.") if File.file?(codec)
    end

    private def load_file(path : String, prefix = "")
      owner = MappedFile.new(path)
      n = IO::ByteFormat::LittleEndian.decode(UInt64, owner.slice(0_i64, 8_i64)).to_i64
      raise ArgumentError.new("invalid safetensors header size") unless n > 1 && n <= 64 * 1024 * 1024 && n <= owner.size - 8
      header = JSON.parse(String.new(owner.slice(8_i64, n))).as_h
      header.each do |name, value|
        next if name == "__metadata__"
        shape = value["shape"].as_a.map(&.as_i)
        offsets = value["data_offsets"].as_a.map(&.as_i64)
        raise ArgumentError.new("invalid tensor offsets") unless offsets.size == 2 && offsets[0] >= 0 && offsets[1] >= offsets[0]
        key = prefix + name
        raise ArgumentError.new("duplicate tensor #{key}") if @tensors.has_key?(key)
        @tensors[key] = Tensor.new(shape, value["dtype"].as_s, owner.slice(8_i64 + n + offsets[0], offsets[1] - offsets[0]), owner)
      end
    end

    def tensor(name : String) : Tensor
      @tensors[name]? || raise ArgumentError.new("missing tensor #{name}")
    end

    def vector(name : String) : Vector
      t = tensor(name)
      raise ArgumentError.new("expected vector #{name}") unless t.shape.size == 1
      t.values
    end

    def matrix(name : String, expert : Int32? = nil) : Matrix
      key = "#{name}:#{expert}"
      @matrices[key] ||= begin
        stem = name.ends_with?(".weight") ? name[0...-7] : name
        scales = @tensors[stem + ".scales"]?
        biases = @tensors[stem + ".biases"]?
        bits, group = 4, 64
        if config = @quantization
          selected = config[stem]? || config
          mode = selected["mode"]?.try(&.as_s) || config["mode"]?.try(&.as_s) || "affine"
          raise ArgumentError.new("unsupported quantization mode #{mode}") unless mode == "affine"
          bits = selected["bits"]?.try(&.as_i) || 4
          group = selected["group_size"]?.try(&.as_i) || 64
        end
        Matrix.new(tensor(name), scales, biases, bits, group, expert || 0)
      end
    end
  end

  module Ops
    extend self

    def finite!(x : Vector, label : String) : Vector
      raise ArgumentError.new("non-finite #{label}") unless x.all?(&.finite?)
      x
    end

    def linear(x : Vector, weight : Matrix) : Vector
      weight.matvec(x)
    end

    def add(a : Vector, b : Vector) : Vector
      raise ArgumentError.new("addition width mismatch") unless a.size == b.size
      Array(Float32).new(a.size) { |i| a[i] + b[i] }
    end

    def sigmoid(x : Float32) : Float32
      x >= 0 ? (1.0 / (1.0 + Math.exp(-x))).to_f32 : (Math.exp(x) / (1.0 + Math.exp(x))).to_f32
    end

    def silu(x : Vector) : Vector
      x.map { |v| v * sigmoid(v) }
    end

    def rms_norm(x : Vector, w : Vector, eps : Float64, zero_centered : Bool = true) : Vector
      raise ArgumentError.new("RMSNorm width mismatch") unless x.size == w.size && !x.empty? && eps > 0
      sum = 0_f32
      x.each { |v| sum += v * v }
      inv = (1.0 / Math.sqrt(sum / x.size + eps)).to_f32
      Array(Float32).new(x.size) { |i| x[i] * inv * (zero_centered ? 1_f32 + w[i] : w[i]) }
    end

    def softmax(x : Vector) : Vector
      raise ArgumentError.new("empty softmax") if x.empty?
      maximum = x.max
      y = x.map { |v| Math.exp(v - maximum).to_f32 }
      sum = y.sum
      y.map { |v| v / sum }
    end

    def rope!(x : Vector, n_heads : Int32, head_dim : Int32, rotary_dim : Int32,
              position : Int32, base : Float64, interleaved : Bool) : Vector
      raise ArgumentError.new("invalid rotary shape") unless x.size == n_heads * head_dim && rotary_dim >= 0 && rotary_dim <= head_dim && rotary_dim.even? && position >= 0 && base > 0
      n_heads.times do |h|
        (rotary_dim // 2).times do |i|
          angle = position / (base ** (2.0 * i / rotary_dim))
          c, s = Math.cos(angle).to_f32, Math.sin(angle).to_f32
          a = h * head_dim + (interleaved ? 2 * i : i)
          b = h * head_dim + (interleaved ? 2 * i + 1 : i + rotary_dim // 2)
          u, v = x[a], x[b]
          x[a], x[b] = u * c - v * s, u * s + v * c
        end
      end
      x
    end
  end
end
