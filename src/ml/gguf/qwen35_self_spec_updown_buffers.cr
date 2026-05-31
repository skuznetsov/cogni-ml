require "../core/buffer"
require "./qwen35_ffn_updown_adapter"

module ML::GGUF
  # Resident Metal buffer preparation for PCA-updown FFN proposal adapters.
  #
  # This is still setup-only; it does not launch self-spec proposal bodies.
  module Qwen35SelfSpecUpdownBuffers
    extend self

    alias F32Maps = NamedTuple(
      x_mean: Hash(Int32, ML::MetalBuffer),
      c_mean: Hash(Int32, ML::MetalBuffer),
      coeff_w: Hash(Int32, ML::MetalBuffer),
      down: Hash(Int32, ML::MetalBuffer),
      rank: Int32)

    def build_f32_maps(adapters : Qwen35FFNUpDownAdapterMap,
                       layer_ids : Enumerable(Int32),
                       rank : Int32,
                       hidden_dim : Int32) : F32Maps
      raise "GPU pipeline pca-updown rank must be positive" unless rank > 0
      raise "GPU pipeline pca-updown rank too large for current Metal kernel" if rank > 64

      x_mean = {} of Int32 => ML::MetalBuffer
      c_mean = {} of Int32 => ML::MetalBuffer
      coeff_w = {} of Int32 => ML::MetalBuffer
      down = {} of Int32 => ML::MetalBuffer
      actual_rank = nil.as(Int32?)
      layer_ids.each do |il|
        adapter = adapters[il]? || raise "GPU pipeline pca-updown missing adapter for layer #{il}"
        bufs = build_f32(adapter, rank, hidden_dim)
        if prev_rank = actual_rank
          raise "GPU pipeline pca-updown inconsistent adapter ranks: #{prev_rank} vs #{bufs[:rank]} at layer #{il}" unless prev_rank == bufs[:rank]
        else
          actual_rank = bufs[:rank]
        end
        x_mean[il] = bufs[:x_mean]
        c_mean[il] = bufs[:c_mean]
        coeff_w[il] = bufs[:coeff_w]
        down[il] = bufs[:down]
      end
      {
        x_mean:  x_mean,
        c_mean:  c_mean,
        coeff_w: coeff_w,
        down:    down,
        rank:    actual_rank || raise("GPU pipeline pca-updown has no layers"),
      }
    end

    def build_f32(adapter : Qwen35FFNUpDownAdapter,
                  rank : Int32,
                  hidden_dim : Int32) : NamedTuple(x_mean: ML::MetalBuffer, c_mean: ML::MetalBuffer, coeff_w: ML::MetalBuffer, down: ML::MetalBuffer, rank: Int32)
      limit = Math.min(rank, adapter.coeff_weights.size)
      raise "FFN up/down adapter has no coefficient weights" unless limit > 0
      raise "FFN up/down adapter output dim mismatch" unless adapter.down_basis[0].size == hidden_dim
      coeff_weights = Array(Float32).new(limit * hidden_dim)
      down_basis = Array(Float32).new(limit * hidden_dim)
      limit.times do |j|
        hidden_dim.times { |d| coeff_weights << adapter.coeff_weights[j][d].to_f32 }
        hidden_dim.times { |d| down_basis << adapter.down_basis[j][d] }
      end
      {
        x_mean:  ML::MetalBuffer.from_array(adapter.x_mean.map(&.to_f32)),
        c_mean:  ML::MetalBuffer.from_array(adapter.c_mean[0, limit].map(&.to_f32)),
        coeff_w: ML::MetalBuffer.from_array(coeff_weights),
        down:    ML::MetalBuffer.from_array(down_basis),
        rank:    limit,
      }
    end
  end
end
