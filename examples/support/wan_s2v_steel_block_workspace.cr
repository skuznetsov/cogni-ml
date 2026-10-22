require "./wan_s2v_steel_attention_probe"

# A probe-only bridge to the block's existing concrete workspace parameter.
# The inherited MPS plans are retained and memory-accounted but never encoded.
# This does not select a native backend in the production inference path.
class WanS2VSteelBlockWorkspace < ML::Video::WanAttentionMPSChunkWorkspace
  @native : WanS2VSteelAttentionProbe? = nil
  # Probe-only GPU comparison hook. It must encode into the borrowed chain,
  # without waiting, reading activations, or allocating MetalBuffers.
  property diagnostic_after_attention : Proc(ML::Metal::MPSCommandChain, Nil)? = nil

  def initialize(query_rows : Int32, key_rows : Int32, heads : Int32,
                 q : ML::MetalBuffer, k : ML::MetalBuffer, v : ML::MetalBuffer,
                 output : ML::MetalBuffer, chunk_q : ML::MetalBuffer,
                 chunk_a : ML::MetalBuffer, chunk_b : ML::MetalBuffer,
                 prepared_k : ML::MetalBuffer, prepared_v : ML::MetalBuffer,
                 row_state : ML::MetalBuffer, source : String)
    super(1, query_rows, key_rows, heads, 128, 128,
      q, k, v, output, chunk_q, chunk_a, chunk_b,
      prepared_k, prepared_v, heads_per_group: 4)
    @native = WanS2VSteelAttentionProbe.new(query_rows, key_rows, heads,
      q, k, v, output, prepared_k, prepared_v, source, 2_048, row_state)
  end

  def scratch_bytes : Int64
    super + query_rows.to_i64 * heads * 2 * sizeof(Float32)
  end

  def distinct_executable_count : Int32
    0 # No MPSGraph executable participates in this arm.
  end

  def encode(chain : ML::Metal::MPSCommandChain) : Nil
    begin_encode!(chain)
    @native.not_nil!.encode(chain)
    @diagnostic_after_attention.try(&.call(chain))
  end

  def encode_with_kv_preparation(chain : ML::Metal::MPSCommandChain,
                                 &block : ML::Metal::ComputeEncoder, ML::MetalBuffer, ML::MetalBuffer -> Nil) : Nil
    begin_encode!(chain)
    @native.not_nil!.encode_with_kv_preparation(chain) do |encoder, prepared_k, prepared_v|
      yield encoder, prepared_k, prepared_v
    end
    @diagnostic_after_attention.try(&.call(chain))
  end

  def release : Nil
    @native.try(&.release)
    super
  end
end
