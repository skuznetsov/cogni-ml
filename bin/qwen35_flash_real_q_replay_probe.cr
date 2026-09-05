# Replay one completed real-model attention operator without changing engine routing.
require "json"
require "option_parser"
require "digest/sha256"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"

alias CPU = ML::GGUF::Qwen35CPU
alias GPU = ML::GGUF::Qwen35Metal

# Probe-only read access: never create a missing slot and mistake it for capture.
module ML::GGUF::Qwen35Metal::Scratch
  def self.replay_existing!(tag : Symbol, bytes : Int64) : ML::MetalBuffer
    @@mutex.synchronize { @@pool[{tag, bytes}]? || raise "missing completed scratch #{tag}" }
  end
end

ROW_SOURCE   = "#define QWEN35_KV_CACHE_F16 1\n" + {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/fullattn_qwen35.metal") }}
FLASH_SOURCE = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/qwen35_attn_flash_d256.metal") }}
ROUND_SOURCE = <<-METAL
#include <metal_stdlib>
using namespace metal;
kernel void replay_round_q(device const float* x [[buffer(0)]],
 device float* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
 if (i < n) y[i] = float(half(x[i]));
}
METAL

private def metrics(a : Array(Float32), b : Array(Float32))
  raise "invalid metric shape" unless a.size == b.size && !a.empty?
  sq = max = 0.0
  outside = 0
  a.each_with_index do |v, i|
    raise "nonfinite replay value" unless v.finite? && b[i].finite?
    d = (v.to_f64 - b[i].to_f64).abs
    sq += d * d
    max = Math.max(max, d)
    outside += 1 if d > 0.001 + 0.0001 * v.abs
  end
  {max_abs: max, rmse: Math.sqrt(sq / a.size), outside_oracle_budget: outside, count: a.size}
end

private def half(bits : UInt16) : Float64
  sign = bits & 0x8000 == 0 ? 1.0 : -1.0
  e, m = ((bits >> 10) & 31).to_i, (bits & 1023).to_i
  return m == 0 ? sign * Float64::INFINITY : Float64::NAN if e == 31
  return sign * m * 2.0 ** -24 if e == 0
  sign * (1.0 + m / 1024.0) * 2.0 ** (e - 15)
end

private def read_values(b : ML::MetalBuffer, count : Int32) : Array(Float32)
  raise "read exceeds buffer" unless count.to_i64 * 4 <= b.size
  Array(Float32).new(count) { |i| b.contents.as(Float32*)[i] }
end

private def replay(pipe : ML::Metal::ComputePipeline, q : ML::MetalBuffer, g : ML::MetalBuffer,
                   k : ML::MetalBuffer, v : ML::MetalBuffer, p : Int32, t : Int32, flash : Bool) : Array(Float32)
  count = t * 24 * 256
  output_buf = ML::MetalBuffer.new((count + 256).to_i64 * 4)
  output_buf.write(Array(Float32).new(count + 256, 12345.0_f32))
  ML::Metal::Dispatch.execute_sequence do |cmd|
    enc = ML::Metal::ComputeEncoder.new(cmd)
    enc.set_pipeline(pipe)
    [q, g, k, v, output_buf].each_with_index { |b, i| enc.set_buffer(b, i) }
    [p, t, 24, 4, 256, 6].each_with_index { |n, i| enc.set_value(n.to_u32, i + 5) }
    enc.set_value(0.0625_f32, 11)
    if flash
      enc.set_threadgroup_memory(16384, 0)
      enc.dispatch_threadgroups({(t + 7) // 8, 24, 1}, {32, 4, 1})
    else
      enc.dispatch_threadgroups({24, t, 1}, {32, 1, 1})
    end
    enc.end_encoding
  end
  result = read_values(output_buf, count + 256)
  raise "output guard overwritten" unless result[count, 256].all? { |x| x == 12345.0_f32 }
  raise "unwritten output" if result.first(count).any? { |x| x == 12345.0_f32 }
  result.first(count)
ensure
  output_buf.try(&.release)
end

# Differently constructed Float64 oracle for 12 complete query/head rows.
private def oracle(q : Array(Float32), gate : Array(Float32), k : ML::MetalBuffer, v : ML::MetalBuffer,
                   p : Int32, t : Int32, output : Array(Float32))
  expected, observed = [] of Float32, [] of Float32
  [0, Math.min(38, t - 1), t - 1].uniq.each do |row|
    [0, 6, 12, 18].each do |h|
      start = (row * 24 + h) * 256
      scores = Array(Float64).new(p + row + 1) do |j|
        off = (j * 4 + h // 6) * 256
        dot = 0.0
        256.times { |d| dot += q[start + d].to_f64 * half(k.contents.as(UInt16*)[off + d]) }
        dot * 0.0625
      end
      peak = scores.max
      probs = scores.map { |s| Math.exp(s - peak) }
      denom = probs.sum
      256.times do |d|
        sum = 0.0
        probs.each_with_index { |prob, j| sum += prob * half(v.contents.as(UInt16*)[(j * 4 + h // 6) * 256 + d]) }
        expected << (sum / denom / (1.0 + Math.exp(-gate[start + d].to_f64))).to_f32
        observed << output[start + d]
      end
    end
  end
  metrics(expected, observed)
end

model = ENV["QWEN35_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
p, t = 256, 65
self_only = false
OptionParser.parse do |o|
  o.on("--model PATH", "Model GGUF") { |x| model = x }
  o.on("--append N", "64 aligned or 65 tail rows") { |x| t = x.to_i }
  o.on("--self-test", "No-model comparator qualification") { self_only = true }
end
raise "comparator did not detect seeded perturbation" unless metrics([1.0_f32], [2.0_f32])[:outside_oracle_budget] == 1
raise "equal comparator failed" unless metrics([1.0_f32], [1.0_f32])[:max_abs] == 0
begin
  metrics([1.0_f32], [Float32::NAN])
  raise "nonfinite not detected"
rescue ex
  raise ex unless ex.message == "nonfinite replay value"
end
raise "half decoder failed" unless half(0x3c00_u16) == 1.0 && half(0xbc00_u16) == -1.0 && half(1_u16) == 2.0 ** -24
puts "self_test=PASS (equal, perturbation, nonfinite, half)"
exit if self_only
raise "bounded T64/T65 only" unless t == 64 || t == 65

ENV.keys.select { |key| key.starts_with?("QWEN35_") }.each { |key| ENV.delete(key) }
ENV["QWEN35_PREFILL_ATTN_FLASH_D256"] = "0"
ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_OFF"] = "1"
ENV["QWEN35_PREFILL_CHUNK_SIZE"] = "2048"
ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = "1"
ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"] = "50"
gguf = ML::GGUF::GGUFFile.new(model, mmap_tensors: false)
begin
  tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model)
ensure
  gguf.close
end
weights = ML::GGUF::Qwen35Weights.from_gguf(model)
state = nil.as(CPU::State?)
rounded = nil.as(ML::MetalBuffer?)
begin
  hp = weights.hparams
  raise "only verified Qwen3.8 GQA6/M2 Max" unless hp.n_layer == 64 && hp.n_head == 24 && hp.n_head_kv == 4 && hp.head_dim == 256 && hp.full_attention?(63) && ML::Metal::Device.instance.name.includes?("M2 Max")
  filler = tokenizer.encode("# Keep insertion order and remove repeated integers.\nvalues = [3, 1, 3, 2, 1]\n" * 300)
  suffix = tokenizer.encode("\n# Return unique integers in their original order.\ndef stable_unique(values):\n    ")
  ids = filler.first(p + t - suffix.size) + suffix
  raise "wrong fixture length" unless ids.size == p + t
  state = CPU::State.new(hp, p + t + 8, kv_cache_f16: true)
  CPU.prepare_state_metal!(state, hp)
  CPU.prefill_tokens(weights, ids.first(p), 0, state)
  CPU.prefill_tokens_last_hidden(weights, ids[p, t], p, state)
  ML::Metal::Device.synchronize
  count = t * 24 * 256
  bytes = count.to_i64 * 4
  q = GPU::Scratch.replay_existing!(:full_chunk_q, bytes)
  g = GPU::Scratch.replay_existing!(:full_chunk_gate, bytes)
  stored = read_values(GPU::Scratch.replay_existing!(:full_chunk_attn, bytes), count)
  layer = state.layers[63]
  raise "wrong KV owner" unless !layer.k_cache && !layer.v_cache && !state.adaptive_kv?
  k, v = layer.k_cache_buf.not_nil!, layer.v_cache_buf.not_nil!
  row_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows", ROW_SOURCE)
  flash_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_flash_d256", FLASH_SOURCE)
  rounding = ML::Metal::ComputePipeline.new("replay_round_q", ROUND_SOURCE)
  rounded = ML::MetalBuffer.new(bytes)
  ML::Metal::Dispatch.execute(rounding) do |enc|
    enc.set_buffer(q, 0)
    enc.set_buffer(rounded.not_nil!, 1)
    enc.set_value(count.to_u32, 2)
    enc.dispatch_1d(count, 256)
  end
  q_values, qr_values, gates = read_values(q, count), read_values(rounded, count), read_values(g, count)
  a = replay(row_pipe, q, g, k, v, p, t, false)
  capture = metrics(stored, a)
  puts({event: "capture", model: model, device: ML::Metal::Device.instance.name,
        layer: 63, prefix: p, rows: t, tokens_sha256: Digest::SHA256.hexdigest(ids.join(",")),
        comparator: "rows_sg4_off_for_both_shapes", source_flash_sha256: Digest::SHA256.hexdigest(FLASH_SOURCE),
        source_row_sha256: Digest::SHA256.hexdigest(ROW_SOURCE), stored_vs_replay: capture}.to_json)
  raise "capture is not the executed layer operator" unless capture[:max_abs] == 0
  b = replay(row_pipe, rounded, g, k, v, p, t, false)
  c = replay(flash_pipe, q, g, k, v, p, t, true)
  d = replay(flash_pipe, rounded, g, k, v, p, t, true)
  rounded_control = metrics(c, d)
  raise "Flash Q-round idempotence failed" unless rounded_control[:max_abs] == 0
  oa = oracle(q_values, gates, k, v, p, t, a)
  ob = oracle(qr_values, gates, k, v, p, t, b)
  oc = oracle(qr_values, gates, k, v, p, t, c)
  ab, ac, bc = metrics(a, b), metrics(a, c), metrics(b, c)
  puts({event: "replay", q_rounding: metrics(q_values, qr_values), row_vs_rounded_row: ab, row_vs_flash: ac,
        rounded_row_vs_flash: bc, flash_round_idempotence: rounded_control,
        residual_rmse_fraction: bc[:rmse] / Math.max(ac[:rmse], 1e-30),
        oracle_row: oa, oracle_rounded_row: ob, oracle_flash_rounded_q: oc,
        oracle_sampled_head_rows: 12, model_quality_claim: false, timing_claim: false}.to_json)
  raise "sampled CPU oracle mismatch" unless [oa, ob, oc].all? { |m| m[:outside_oracle_budget] == 0 }
  puts "replay_qualification=PASS"
ensure
  ML::Metal::Device.synchronize
  rounded.try(&.release)
  state.try do |s|
    s.layers.each do |l|
      l.k_cache_buf.try(&.release)
      l.v_cache_buf.try(&.release)
      l.conv_state_buf.try(&.release)
      l.ssm_state_buf.try(&.release)
    end
  end
  weights.close
end
