require "json"

module ML::GGUF
  # Portable FFN up/down surrogate adapter used by the same-weight self-spec
  # experiments. It is intentionally pure data + math: proposal routes may
  # select an adapter, but exact verification remains the correctness boundary.
  struct Qwen35FFNUpDownAdapter
    FORMAT = "qwen35_ffn_updown_adapter_v1"

    getter x_mean : Array(Float64)
    getter c_mean : Array(Float64)
    getter coeff_weights : Array(Array(Float64))
    getter down_basis : Array(Array(Float32))

    def initialize(@x_mean : Array(Float64),
                   @c_mean : Array(Float64),
                   @coeff_weights : Array(Array(Float64)),
                   @down_basis : Array(Array(Float32)))
      validate!
    end

    def rank : Int32
      Math.min(@coeff_weights.size, @down_basis.size)
    end

    def hidden_dim : Int32
      @x_mean.size
    end

    def project(ffn_in : Array(Float32), requested_rank : Int32) : Array(Float32)
      raise "FFN up/down adapter input dimension mismatch" unless ffn_in.size == hidden_dim
      limit = Math.min(requested_rank, rank)
      raise "FFN up/down adapter projection rank must be positive" unless limit > 0
      out_dim = @down_basis[0].size
      out = Array(Float32).new(out_dim, 0.0_f32)
      limit.times do |j|
        w = @coeff_weights[j]
        coeff = @c_mean[j]
        ffn_in.size.times { |d| coeff += (ffn_in[d].to_f64 - @x_mean[d]) * w[d] }
        coeff_f = coeff.to_f32
        down = @down_basis[j]
        out_dim.times { |d| out[d] += coeff_f * down[d] }
      end
      out
    end

    def quantized(bits : Int32, hadamard_block : Int32? = nil) : Qwen35FFNUpDownAdapter
      quant = ->(row : Array(Float64)) {
        if block = hadamard_block
          Qwen35FFNUpDownAdapter.block_hadamard_quant_dequant(row, bits, block)
        else
          Qwen35FFNUpDownAdapter.symmetric_quant_dequant(row, bits)
        end
      }

      coeff = @coeff_weights.map { |row| quant.call(row) }
      down = @down_basis.map { |row| quant.call(row.map(&.to_f64)).map(&.to_f32) }
      Qwen35FFNUpDownAdapter.new(@x_mean, @c_mean, coeff, down)
    end

    def validate! : Nil
      raise "FFN up/down adapter x_mean must not be empty" if @x_mean.empty?
      raise "FFN up/down adapter c_mean must not be empty" if @c_mean.empty?
      raise "FFN up/down adapter coeff_weights must not be empty" if @coeff_weights.empty?
      raise "FFN up/down adapter down_basis must not be empty" if @down_basis.empty?
      @coeff_weights.each do |row|
        raise "FFN up/down adapter coeff weight dimension mismatch" unless row.size == @x_mean.size
      end
      out_dim = @down_basis[0].size
      raise "FFN up/down adapter down basis must not be empty" if out_dim == 0
      @down_basis.each do |row|
        raise "FFN up/down adapter down basis dimension mismatch" unless row.size == out_dim
      end
      raise "FFN up/down adapter c_mean rank mismatch" if @c_mean.size < @coeff_weights.size
    end

    def self.power_of_two?(n : Int32) : Bool
      n > 0 && (n & (n - 1)) == 0
    end

    def self.block_hadamard_inplace!(values : Array(Float64), block_size : Int32) : Nil
      raise "Hadamard block size must be a positive power of two" unless power_of_two?(block_size)
      raise "Hadamard vector dimension must be divisible by block size" unless values.size % block_size == 0

      offset = 0
      scale = 1.0 / Math.sqrt(block_size.to_f64)
      while offset < values.size
        width = 1
        while width < block_size
          step = width * 2
          i = 0
          while i < block_size
            width.times do |j|
              a_i = offset + i + j
              b_i = a_i + width
              a = values[a_i]
              b = values[b_i]
              values[a_i] = a + b
              values[b_i] = a - b
            end
            i += step
          end
          width = step
        end
        block_size.times { |i| values[offset + i] *= scale }
        offset += block_size
      end
    end

    def self.symmetric_quant_dequant(values : Array(Float64), bits : Int32) : Array(Float64)
      raise "quant bits must be 2..8" unless bits >= 2 && bits <= 8
      qmax = ((1 << (bits - 1)) - 1).to_f64
      max_abs = values.reduce(0.0) { |m, v| {m, v.abs}.max }
      return Array(Float64).new(values.size, 0.0) if max_abs <= 0.0
      scale = max_abs / qmax
      values.map do |v|
        q = (v / scale).round.clamp(-qmax, qmax)
        q * scale
      end
    end

    def self.block_hadamard_quant_dequant(values : Array(Float64), bits : Int32, block_size : Int32) : Array(Float64)
      tmp = values.dup
      block_hadamard_inplace!(tmp, block_size)
      tmp = symmetric_quant_dequant(tmp, bits)
      # Normalized Hadamard is self-inverse.
      block_hadamard_inplace!(tmp, block_size)
      tmp
    end
  end

  alias Qwen35FFNUpDownAdapterMap = Hash(Int32, Qwen35FFNUpDownAdapter)

  module Qwen35FFNUpDownAdapterArtifact
    extend self

    def dump(path : String,
             adapters : Qwen35FFNUpDownAdapterMap,
             rank : Int32,
             hidden_dim : Int32,
             source : String) : Nil
      raise "FFN up/down adapter dump rank must be positive" unless rank > 0
      json = JSON.build do |j|
        j.object do
          j.field "format", Qwen35FFNUpDownAdapter::FORMAT
          j.field "source", source
          j.field "hidden_dim", hidden_dim
          j.field "rank", rank
          j.field "layers" do
            j.array do
              adapters.keys.sort.each do |layer_id|
                adapter = adapters[layer_id]
                limit = Math.min(rank, adapter.rank)
                raise "adapter #{layer_id} has no coefficient weights" unless limit > 0
                raise "adapter #{layer_id} x_mean size mismatch" unless adapter.x_mean.size == hidden_dim
                raise "adapter #{layer_id} down_basis size mismatch" unless adapter.down_basis[0].size == hidden_dim

                j.object do
                  j.field "layer", layer_id
                  j.field "rank", limit
                  j.field "x_mean" do
                    j.array { adapter.x_mean.each { |v| j.number v } }
                  end
                  j.field "c_mean" do
                    j.array { adapter.c_mean[0, limit].each { |v| j.number v } }
                  end
                  j.field "coeff_w" do
                    j.array do
                      limit.times do |r|
                        hidden_dim.times { |d| j.number adapter.coeff_weights[r][d] }
                      end
                    end
                  end
                  j.field "down" do
                    j.array do
                      limit.times do |r|
                        hidden_dim.times { |d| j.number adapter.down_basis[r][d] }
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
      File.write(path, json)
    end

    def load(path : String) : NamedTuple(source: String, hidden_dim: Int32, rank: Int32, adapters: Qwen35FFNUpDownAdapterMap)
      data = JSON.parse(File.read(path))
      format = data["format"].as_s
      raise "unsupported FFN up/down adapter format #{format.inspect}" unless format == Qwen35FFNUpDownAdapter::FORMAT
      source = data["source"]?.try(&.as_s) || "unknown"
      hidden_dim = data["hidden_dim"].as_i
      rank = data["rank"].as_i
      raise "FFN up/down adapter artifact hidden_dim must be positive" unless hidden_dim > 0
      raise "FFN up/down adapter artifact rank must be positive" unless rank > 0

      adapters = Qwen35FFNUpDownAdapterMap.new
      data["layers"].as_a.each do |node|
        layer_id = node["layer"].as_i
        layer_rank = node["rank"].as_i
        raise "FFN up/down adapter layer rank must be positive" unless layer_rank > 0
        x_mean = node["x_mean"].as_a.map(&.as_f)
        c_mean = node["c_mean"].as_a.map(&.as_f)
        coeff_flat = node["coeff_w"].as_a.map(&.as_f)
        down_flat = node["down"].as_a.map(&.as_f.to_f32)
        raise "FFN up/down adapter x_mean size mismatch" unless x_mean.size == hidden_dim
        raise "FFN up/down adapter c_mean size mismatch" unless c_mean.size == layer_rank
        raise "FFN up/down adapter coeff_w size mismatch" unless coeff_flat.size == layer_rank * hidden_dim
        raise "FFN up/down adapter down size mismatch" unless down_flat.size == layer_rank * hidden_dim
        coeff = Array(Array(Float64)).new(layer_rank) do |r|
          coeff_flat[r * hidden_dim, hidden_dim]
        end
        down = Array(Array(Float32)).new(layer_rank) do |r|
          down_flat[r * hidden_dim, hidden_dim]
        end
        adapters[layer_id] = Qwen35FFNUpDownAdapter.new(x_mean, c_mean, coeff, down)
      end

      {source: source, hidden_dim: hidden_dim, rank: rank, adapters: adapters}
    end
  end
end
