require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_metal"
require "../src/ml/core/buffer"

# Gated attention decode kernel: Metal vs CPU reference.
# Shape matches Qwen 3.5 9B full-attention layer:
#   n_head = 40, n_head_kv = 8, head_dim = 256, heads_per_group = 5.
# We simulate a realistic decode step at pos=17 (cache_len=18 positions).
describe "Qwen35Metal.attn_decode" do
  it "matches the CPU reference on Qwen 3.5 9B shapes" do
    next unless ML::GGUF::Qwen35Metal.available?

    n_head          = 40
    n_head_kv       =  8
    head_dim        = 256
    heads_per_group = n_head // n_head_kv
    pos             = 17
    cache_len       = pos + 1
    max_seq         = 64
    kv_dim          = n_head_kv * head_dim
    q_dim           = n_head * head_dim
    scale           = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32

    rng = Random.new(0xC0FFEE_u64)

    q    = Array(Float32).new(q_dim)  { ((rng.next_float - 0.5) * 1.0_f64).to_f32 }
    gate = Array(Float32).new(q_dim)  { ((rng.next_float - 0.5) * 2.0_f64).to_f32 }

    # Full k_cache / v_cache is [max_seq * kv_dim], but only the first
    # `cache_len` rows contain valid data (past tokens). Remaining rows
    # must still be present so strides match; fill with zeros.
    k_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
    v_cache = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
    (cache_len * kv_dim).times do |i|
      k_cache[i] = ((rng.next_float - 0.5) * 1.0_f64).to_f32
      v_cache[i] = ((rng.next_float - 0.5) * 1.0_f64).to_f32
    end

    # ── CPU reference (copy the inner loop from forward_full_attn_layer)
    cpu_out = Array(Float32).new(q_dim, 0.0_f32)
    scores  = Array(Float32).new(cache_len, 0.0_f32)

    n_head.times do |h|
      kv_h  = h // heads_per_group
      q_off = h * head_dim

      cache_len.times do |p|
        k_off = p * kv_dim + kv_h * head_dim
        s = 0.0_f32
        head_dim.times { |d| s += q[q_off + d] * k_cache[k_off + d] }
        scores[p] = s * scale
      end
      ML::GGUF::Qwen35CPU.softmax_slice!(scores, 0, cache_len)

      out_off = h * head_dim
      cache_len.times do |p|
        v_off = p * kv_dim + kv_h * head_dim
        w = scores[p]
        head_dim.times { |d| cpu_out[out_off + d] += w * v_cache[v_off + d] }
      end
    end
    # Apply gate
    q_dim.times { |i| cpu_out[i] = cpu_out[i] * ML::GGUF::Qwen35CPU.sigmoid(gate[i]) }

    # ── Metal path
    kbuf = ML::MetalBuffer.new((max_seq * kv_dim).to_i64 * sizeof(Float32))
    vbuf = ML::MetalBuffer.new((max_seq * kv_dim).to_i64 * sizeof(Float32))
    kbuf.write(k_cache)
    vbuf.write(v_cache)

    gpu_out = ML::GGUF::Qwen35Metal.attn_decode(
      q, gate, kbuf, vbuf,
      pos, n_head, n_head_kv, head_dim, heads_per_group, scale,
    )

    dot, ncpu, ngpu = 0.0_f64, 0.0_f64, 0.0_f64
    max_diff = 0.0_f32
    cpu_out.size.times do |i|
      a = cpu_out[i].to_f64
      b = gpu_out[i].to_f64
      dot  += a * b
      ncpu += a * a
      ngpu += b * b
      d = (cpu_out[i] - gpu_out[i]).abs
      max_diff = d if d > max_diff
    end
    cos = dot / (Math.sqrt(ncpu) * Math.sqrt(ngpu))
    printf "  [attn_decode] cos=%.12f max|Δ|=%g\n", cos, max_diff

    cos.should be > 0.9999999
    max_diff.should be < 1.0e-4_f32

    kbuf.release
    vbuf.release
  end
end
