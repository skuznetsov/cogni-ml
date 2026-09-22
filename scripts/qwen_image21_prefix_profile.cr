#!/usr/bin/env crystal

require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_weights"

private def median(values : Array(Float64)) : Float64
  sorted = values.sort
  (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2.0
end

private def report(label : String, wall : Array(Float64), encode : Array(Float64), gpu : Array(Float64)) : Nil
  gpu_label = gpu.empty? ? "unavailable" : "#{median(gpu).round(3)}"
  puts "#{label}: n=#{wall.size} wall_median_ms=#{median(wall).round(3)} " \
       "input_encode_median_ms=#{median(encode).round(3)} gpu_command_median_ms=#{gpu_label}"
end

path = ARGV[0]? || abort "usage: crystal run scripts/qwen_image21_prefix_profile.cr -- MODEL.gguf [samples]"
samples = ARGV[1]?.try(&.to_i) || 8
abort "samples must be positive" unless samples > 0
ENV["QWEN_IMAGE21_PROFILE"] = "1"

model = ML::GGUF::QwenImage21Weights.from_gguf(path)
stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
begin
  config = model.transformer_config
  hidden = Array(Float32).new(8 * config.input_dim) do |index|
    (((index * 19 + 3) % 101) - 50).to_f32 / 149.0_f32
  end
  encoder = Array(Float32).new(2 * config.context_dim) do |index|
    (((index * 31 + 9) % 127) - 63).to_f32 / 173.0_f32
  end
  shapes = [StaticArray[1, 2, 2], StaticArray[1, 2, 2]]
  mask = [false, true, true]
  backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
  build_wall = [] of Float64
  build_encode = [] of Float64
  build_gpu = [] of Float64
  hit_wall = [] of Float64
  hit_encode = [] of Float64
  hit_gpu = [] of Float64

  (samples + 2).times do |index|
    input = hidden.dup
    input[0] += (index + 1).to_f32 / 64.0_f32
    input[4 * config.input_dim] += index.to_f32 / 128.0_f32
    builds_before = stack.prefix_cache_builds
    started = Time.instant
    ML::GGUF::QwenImage21TransformerCPU.forward(
      input, encoder, 0.25_f32, shapes, mask, model.transformer_weights, config,
      backend: backend, layer_stack_backend: stack,
    )
    build_ms = (Time.instant - started).total_milliseconds
    build_stats = stack.last_stats.not_nil!
    abort "expected cache rebuild" unless stack.prefix_cache_builds == builds_before + 1
    abort "expected eight image projection rows on rebuild" unless build_stats.image_projection_rows == 8

    input[4 * config.input_dim] += 0.125_f32
    hits_before = stack.prefix_cache_hits
    started = Time.instant
    ML::GGUF::QwenImage21TransformerCPU.forward(
      input, encoder, 0.75_f32, shapes, mask, model.transformer_weights, config,
      backend: backend, layer_stack_backend: stack,
    )
    hit_ms = (Time.instant - started).total_milliseconds
    hit_stats = stack.last_stats.not_nil!
    abort "expected cache hit" unless stack.prefix_cache_hits == hits_before + 1
    abort "expected four image projection rows on hit" unless hit_stats.image_projection_rows == 4

    next if index < 2
    build_wall << build_ms
    build_encode << build_stats.input_encode_ms.not_nil!
    build_gpu << build_stats.gpu_command_ms.not_nil! if build_stats.gpu_command_ms
    hit_wall << hit_ms
    hit_encode << hit_stats.input_encode_ms.not_nil!
    hit_gpu << hit_stats.gpu_command_ms.not_nil! if hit_stats.gpu_command_ms
  end

  report("rebuild", build_wall, build_encode, build_gpu)
  report("hit", hit_wall, hit_encode, hit_gpu)
ensure
  stack.close
  model.close
end
