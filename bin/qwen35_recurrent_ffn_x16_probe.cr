require "option_parser"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_metal"

record Timing, wall_ms : Float64, gpu_ms : Float64

private def percentile(values : Array(Float64), pct : Int32) : Float64
  sorted = values.sort
  sorted[(sorted.size * pct // 100).clamp(0, sorted.size - 1)]
end

private def timed_matmul(qw : ML::GGUF::QuantWeight,
                         x_buf : ML::MetalBuffer,
                         out_buf : ML::MetalBuffer,
                         x16 : Bool) : Timing
  ENV["QWEN35_Q4K_GEMV_X16"] = x16 ? "1" : "0"
  cmd = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(cmd)
  unless ML::GGUF::Qwen35Metal.encode_matmul_to_buffer(enc, qw, x_buf, out_buf, 1)
    raise "operator is not Metal routable"
  end
  enc.end_encoding
  started = Time.instant
  gpu_ms = cmd.commit_and_wait_gpu_elapsed_seconds * 1_000.0
  Timing.new((Time.instant - started).total_milliseconds, gpu_ms)
end

private def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  raise "output size mismatch" unless a.size == b.size
  max = 0.0_f32
  a.each_with_index do |value, i|
    diff = (value - b[i]).abs
    max = diff if diff > max
  end
  max
end

private def print_summary(label : String,
                          qw : ML::GGUF::QuantWeight,
                          x_buf : ML::MetalBuffer,
                          warmup : Int32,
                          pairs : Int32) : Nil
  base_out = ML::MetalBuffer.new(qw.out_dim.to_i64 * sizeof(Float32))
  x16_out = ML::MetalBuffer.new(qw.out_dim.to_i64 * sizeof(Float32))

  warmup.times do
    timed_matmul(qw, x_buf, base_out, false)
    timed_matmul(qw, x_buf, x16_out, true)
  end

  base = Array(Timing).new(pairs)
  x16 = Array(Timing).new(pairs)
  pairs.times do |i|
    if i.even?
      base << timed_matmul(qw, x_buf, base_out, false)
      x16 << timed_matmul(qw, x_buf, x16_out, true)
    else
      x16 << timed_matmul(qw, x_buf, x16_out, true)
      base << timed_matmul(qw, x_buf, base_out, false)
    end
  end

  timed_matmul(qw, x_buf, base_out, false)
  timed_matmul(qw, x_buf, x16_out, true)
  diff = max_abs_diff(base_out.read(qw.out_dim), x16_out.read(qw.out_dim))
  raise "x16 output drift #{diff} exceeds 1e-3" if diff > 1.0e-3_f32

  base_wall = base.map(&.wall_ms)
  x16_wall = x16.map(&.wall_ms)
  base_gpu = base.map(&.gpu_ms)
  x16_gpu = x16.map(&.gpu_ms)
  gpu_wins = base.zip(x16).count { |pair| pair[1].gpu_ms < pair[0].gpu_ms }

  puts "op=#{label} type=#{qw.type.name} in=#{qw.in_dim} out=#{qw.out_dim}"
  puts "route_tag=#{qw.route_tag} capability=#{qw.q4_gemv_x16_capability}"
  printf "base_wall_p50_ms=%.6f x16_wall_p50_ms=%.6f\n",
    percentile(base_wall, 50), percentile(x16_wall, 50)
  printf "base_gpu_p50_ms=%.6f x16_gpu_p50_ms=%.6f gpu_delta_pct=%.3f\n",
    percentile(base_gpu, 50), percentile(x16_gpu, 50),
    100.0 * (percentile(x16_gpu, 50) / percentile(base_gpu, 50) - 1.0)
  printf "base_gpu_mean_ms=%.6f x16_gpu_mean_ms=%.6f gpu_wins=%d/%d\n",
    base_gpu.sum / pairs, x16_gpu.sum / pairs, gpu_wins, pairs
  puts "max_abs_diff=#{diff}"
end

model = ENV["QWEN35_MODEL"]? || ""
warmup = 5
pairs = 40

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_recurrent_ffn_x16_probe --model PATH [--warmup N] [--pairs N]"
  parser.on("--model=PATH", "Qwen3.8 GGUF path") { |value| model = value }
  parser.on("--warmup=N", "Warmup pairs (default: 5)") { |value| warmup = value.to_i }
  parser.on("--pairs=N", "Measured alternating pairs per operator (default: 40)") { |value| pairs = value.to_i }
  parser.on("-h", "--help", "Show help") { puts parser; exit }
end

raise "--model is required" if model.empty?
raise "--warmup must be non-negative" if warmup < 0
raise "--pairs must be positive" unless pairs > 0
raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?

weights = ML::GGUF::Qwen35Weights.from_gguf(model)
original_x16 = ENV["QWEN35_Q4K_GEMV_X16"]?
begin
  recurrent = weights.layers.compact_map do |layer|
    layer.as?(ML::GGUF::Qwen35RecurrentWeights)
  end.find do |layer|
    layer.ffn_gate_qw.type.q4_k? && layer.ffn_up_qw.type.q4_k?
  end || raise "no recurrent Q4 FFN gate/up pair found"

  expected_capability = ML::GGUF::Q4GemvX16Capability::Qwen38
  unless recurrent.ffn_gate_qw.q4_gemv_x16_capability == expected_capability &&
         recurrent.ffn_up_qw.q4_gemv_x16_capability == expected_capability
    raise "probe requires the measured Qwen3.8 x16 capability"
  end

  x = Array(Float32).new(recurrent.ffn_gate_qw.in_dim) do |i|
    ((((i.to_i64 * 1103515245_i64 + 12345_i64) & 0xffff_i64) / 32768.0) - 1.0).to_f32
  end
  x_buf = ML::MetalBuffer.from_array(x)

  puts "Qwen3.8 recurrent FFN Q4 x16 isolated A/B/BA"
  puts "model=#{model} warmup=#{warmup} pairs=#{pairs}"
  print_summary("gate", recurrent.ffn_gate_qw, x_buf, warmup, pairs)
  print_summary("up", recurrent.ffn_up_qw, x_buf, warmup, pairs)
ensure
  if value = original_x16
    ENV["QWEN35_Q4K_GEMV_X16"] = value
  else
    ENV.delete("QWEN35_Q4K_GEMV_X16")
  end
  weights.close
end
