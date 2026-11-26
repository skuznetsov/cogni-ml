#!/usr/bin/env crystal

# Synthetic-latent, real-weight DiT profile. The diagnostic phase pass splits
# the resident stack into many command buffers and must never be read as the
# latency of the normal one-command inference path.
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_weights"
require "../src/ml/gguf/qwen_image21_flow_match"

private def median(values : Array(Float64)) : Float64
  sorted = values.sort
  (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2.0
end

private def format_ms(value : Float64?) : String
  value.try(&.round(3).to_s) || "unavailable"
end

private def format_ratio(candidate : Float64?, baseline : Float64?) : String
  return "unavailable" unless candidate && baseline
  (candidate / baseline).round(3).to_s
end

private def median_ratio(values : Array(Float64), expected : Int32) : String
  return "unavailable" unless expected > 0 && values.size == expected
  median(values).round(3).to_s
end

private def restore_environment(key : String, value : String?) : Nil
  if value
    ENV[key] = value
  else
    ENV.delete(key)
  end
end

private def report(label : String, wall : Array(Float64), stats : Array(ML::GGUF::QwenImage21MetalBlockStats)) : Nil
  gpu = stats.compact_map(&.gpu_command_ms)
  prepare = stats.compact_map(&.buffer_prepare_ms)
  readback = stats.compact_map(&.readback_ms)
  encode = stats.compact_map(&.input_encode_ms)
  puts "#{label}: n=#{wall.size} wall_median_ms=#{median(wall).round(3)} " \
       "gpu_command_median_ms=#{gpu.empty? ? "unavailable" : median(gpu).round(3)} " \
       "buffer_prepare_median_ms=#{prepare.empty? ? "unavailable" : median(prepare).round(3)} " \
       "input_encode_median_ms=#{encode.empty? ? "unavailable" : median(encode).round(3)} " \
       "readback_median_ms=#{readback.empty? ? "unavailable" : median(readback).round(3)} " \
       "command_buffers=#{stats.first.command_buffers}"
end

private def report_phases(label : String, stats : ML::GGUF::QwenImage21MetalBlockStats) : Nil
  phases = stats.phase_gpu_ms
  unless phases
    puts "#{label}: GPU phase timestamps unavailable"
    return
  end
  puts "#{label}: diagnostic_split_commands=#{stats.command_buffers} " \
       "gpu_phase_sum_ms=#{phases.values.sum.round(3)}"
  phases.keys.sort.each do |phase|
    wait = stats.phase_wait_ms.not_nil![phase]
    puts "  #{phase}: gpu_ms=#{phases[phase].round(3)} submit_wait_wall_ms=#{wait.round(3)}"
  end
end

private def run_pair(
  model : ML::GGUF::QwenImage21Weights,
  stack : ML::GGUF::QwenImage21MetalLayerStackBackend,
  backend : ML::GGUF::QwenImage21MetalProjectionBackend,
  hidden : Array(Float32), encoder : Array(Float32),
  shapes : Array(StaticArray(Int32, 3)), mask : Array(Bool),
  build_timestep : Float32, hit_timestep : Float32,
  target_tokens : Int32,
) : {Array(Float32), Array(Float32), Float64, Float64, ML::GGUF::QwenImage21MetalBlockStats, ML::GGUF::QwenImage21MetalBlockStats}
  config = model.transformer_config
  input = hidden.dup
  input[0] += 1.0_f32 / 64.0_f32
  started = Time.instant
  build = ML::GGUF::QwenImage21TransformerCPU.forward(
    input, encoder, build_timestep, shapes, mask, model.transformer_weights, config,
    backend: backend, layer_stack_backend: stack,
  )
  build_wall = (Time.instant - started).total_milliseconds
  build_stats = stack.last_stats.not_nil!
  abort "expected cache build" unless stack.prefix_cache_builds == 1
  abort "unexpected rebuild image rows" unless build_stats.image_projection_rows == shapes.sum { |shape| shape[1] * shape[2] }

  target_start = (shapes.sum { |shape| shape[1] * shape[2] } - target_tokens) * config.input_dim
  input[target_start] += 0.125_f32
  started = Time.instant
  hit = ML::GGUF::QwenImage21TransformerCPU.forward(
    input, encoder, hit_timestep, shapes, mask, model.transformer_weights, config,
    backend: backend, layer_stack_backend: stack,
  )
  hit_wall = (Time.instant - started).total_milliseconds
  hit_stats = stack.last_stats.not_nil!
  abort "expected cache hit" unless stack.prefix_cache_hits == 1
  abort "unexpected hit image rows" unless hit_stats.image_projection_rows == target_tokens
  {build.output, hit.output, build_wall, hit_wall, build_stats, hit_stats}
end

path = ARGV[0]? || abort "usage: crystal run scripts/qwen_image21_prefix_profile.cr -- MODEL.gguf [samples=3] [target_side=16] [condition_side=16] [phase_samples=1]"
samples = ARGV[1]?.try(&.to_i) || 3
target_side = ARGV[2]?.try(&.to_i) || 16
condition_side = ARGV[3]?.try(&.to_i) || 16
phase_samples = ARGV[4]?.try(&.to_i) || 1
abort "samples must be positive" unless samples > 0
abort "phase_samples must be nonnegative" unless phase_samples >= 0
abort "image sides must be positive and even" unless target_side > 0 && target_side.even? && condition_side > 0 && condition_side.even?
register_auto_ab = ENV["QWEN_IMAGE21_Q8_REGISTER_AB"]? == "1"
register_force_ab = ENV["QWEN_IMAGE21_Q8_REGISTER_FORCE_AB"]? == "1"
register_reuse_ab = register_auto_ab || register_force_ab
batch_ab = ENV["QWEN_IMAGE21_Q8_BATCH_AB"]? == "1"
abort "choose only one Q8 A/B mode" if register_reuse_ab && batch_ab
abort "choose only one Q8 register A/B mode" if register_auto_ab && register_force_ab
if register_reuse_ab
  abort "register reuse AB requires at least 3 measured pairs" unless samples >= 3
  abort "register reuse AB requires 513 total tokens" unless 1 + target_side * target_side + condition_side * condition_side == 513
end
if register_auto_ab
  ML::Metal::Device.init!
  device_name = ML::Metal::Device.instance.name
  abort "automatic register A/B requires Apple M2 Max; found #{device_name}" unless device_name == "Apple M2 Max"
  puts "register_reuse_device=#{device_name}"
end
abort "model file does not exist: #{path}" unless File.file?(path)

model = ML::GGUF::QwenImage21Weights.from_gguf(path)
begin
  config = model.transformer_config
  condition_tokens = condition_side * condition_side
  target_tokens = target_side * target_side
  total_image_tokens = condition_tokens + target_tokens
  encoder_tokens = 1 + condition_tokens // 4
  hidden = Array(Float32).new(total_image_tokens * config.input_dim) do |index|
    (((index * 19 + 3) % 101) - 50).to_f32 / 149.0_f32
  end
  encoder = Array(Float32).new(encoder_tokens * config.context_dim) do |index|
    (((index * 31 + 9) % 127) - 63).to_f32 / 173.0_f32
  end
  shapes = [StaticArray[1, condition_side, condition_side], StaticArray[1, target_side, target_side]]
  mask = [false] + Array(Bool).new((condition_tokens + target_tokens) // 4, true)
  schedule = ML::GGUF::QwenImage21FlowMatch.schedule(2, target_tokens)
  puts "model=#{File.basename(path)} target_tokens=#{target_tokens} condition_tokens=#{condition_tokens} " \
       "text_tokens=1 total_tokens=#{1 + total_image_tokens} layers=#{model.layers.size} " \
       "f32_prefix_kv_bytes=#{2_i64 * model.layers.size * (1 + condition_tokens) * config.hidden_dim * sizeof(Float32)}"
  puts "normal pass: one command per DiT evaluation; diagnostic phases: split commands, synthetic latents"
  q8_route = if register_auto_ab
               "alternating AUTO vs baseline"
             elsif register_force_ab
               "alternating forced reuse vs baseline"
             elsif ENV["QWEN_IMAGE21_Q8_BATCH_AB"]? == "1"
               "alternating A/B"
             elsif ENV["QWEN_IMAGE21_Q8_BATCH"]? == "0"
               "disabled"
             else
               "enabled for batch >= 16"
             end
  puts "Q8_0 batch route: #{q8_route}"

  if register_reuse_ab
    prior_batch = ENV["QWEN_IMAGE21_Q8_BATCH"]?
    prior_reuse = ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"]?
    prior_profile = ENV["QWEN_IMAGE21_PROFILE"]?
    candidate_label = register_auto_ab ? "auto" : "forced_reuse"
    begin
      candidate_override = register_auto_ab ? "unset_auto" : "1"
      puts "Q8_0 #{candidate_label} vs baseline: candidate_override=#{candidate_override} " \
           "baseline_override=0 warm_pairs=1 measured_pairs=#{samples} parity_max_abs<1e-4"
      rebuild_wall_ratios = [] of Float64
      hit_wall_ratios = [] of Float64
      rebuild_gpu_ratios = [] of Float64
      hit_gpu_ratios = [] of Float64
      (samples + 1).times do |index|
        modes = index.even? ? ["baseline", "candidate"] : ["candidate", "baseline"]
        outputs = Hash(String, {Array(Float32), Array(Float32)}).new
        walls = Hash(String, {Float64, Float64}).new
        gpu = Hash(String, {Float64?, Float64?}).new
        modes.each do |mode|
          ENV["QWEN_IMAGE21_Q8_BATCH"] = "1"
          if mode == "baseline"
            ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"] = "0"
          elsif register_auto_ab
            ENV.delete("QWEN_IMAGE21_Q8_REGISTER_REUSE")
          else
            ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"] = "1"
          end
          ENV["QWEN_IMAGE21_PROFILE"] = "1"
          stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
          begin
            pair = run_pair(model, stack,
              ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
              hidden, encoder, shapes, mask, schedule.model_timestep(0), schedule.model_timestep(1),
              target_tokens)
            abort "register reuse AB rebuild did not use one command buffer" unless pair[4].command_buffers == 1
            abort "register reuse AB prefix hit did not use one command buffer" unless pair[5].command_buffers == 1
            outputs[mode] = {pair[0], pair[1]}
            walls[mode] = {pair[2], pair[3]}
            gpu[mode] = {pair[4].gpu_command_ms, pair[5].gpu_command_ms}
          ensure
            stack.close
          end
        end
        build_abs = outputs["baseline"][0].zip(outputs["candidate"][0]).max_of { |a, b| (a - b).abs }
        hit_abs = outputs["baseline"][1].zip(outputs["candidate"][1]).max_of { |a, b| (a - b).abs }
        abort "Q8_0 #{candidate_label} changed rebuild output beyond strict parity bound" unless build_abs < 1.0e-4
        abort "Q8_0 #{candidate_label} changed prefix-hit output beyond strict parity bound" unless hit_abs < 1.0e-4
        next if index == 0 # Warm both routes and their pipelines once.

        base = walls["baseline"]
        candidate = walls["candidate"]
        base_gpu = gpu["baseline"]
        candidate_gpu = gpu["candidate"]
        build_ratio = candidate[0] / base[0]
        hit_ratio = candidate[1] / base[1]
        rebuild_wall_ratios << build_ratio
        hit_wall_ratios << hit_ratio
        if base_gpu[0] && candidate_gpu[0]
          rebuild_gpu_ratios << candidate_gpu[0].not_nil! / base_gpu[0].not_nil!
        end
        if base_gpu[1] && candidate_gpu[1]
          hit_gpu_ratios << candidate_gpu[1].not_nil! / base_gpu[1].not_nil!
        end
        puts "pair=#{index} order=#{modes.map { |mode| mode == "baseline" ? "baseline" : candidate_label }.join("/")} " \
             "rebuild_wall_baseline_ms=#{base[0].round(3)} rebuild_wall_#{candidate_label}_ms=#{candidate[0].round(3)} " \
             "rebuild_gpu_command_baseline_ms=#{format_ms(base_gpu[0])} rebuild_gpu_command_#{candidate_label}_ms=#{format_ms(candidate_gpu[0])} " \
             "hit_wall_baseline_ms=#{base[1].round(3)} hit_wall_#{candidate_label}_ms=#{candidate[1].round(3)} " \
             "hit_gpu_command_baseline_ms=#{format_ms(base_gpu[1])} hit_gpu_command_#{candidate_label}_ms=#{format_ms(candidate_gpu[1])} " \
             "rebuild_wall_ratio=#{build_ratio.round(3)} rebuild_gpu_command_ratio=#{format_ratio(candidate_gpu[0], base_gpu[0])} " \
             "hit_wall_ratio=#{hit_ratio.round(3)} hit_gpu_command_ratio=#{format_ratio(candidate_gpu[1], base_gpu[1])} " \
             "parity_max_abs_build=#{build_abs} parity_max_abs_hit=#{hit_abs}"
      end
      puts "n=#{samples} paired_median_rebuild_wall_ratio_#{candidate_label}_over_baseline=#{median_ratio(rebuild_wall_ratios, samples)} " \
           "paired_median_hit_wall_ratio_#{candidate_label}_over_baseline=#{median_ratio(hit_wall_ratios, samples)} " \
           "paired_median_rebuild_gpu_command_ratio_#{candidate_label}_over_baseline=#{median_ratio(rebuild_gpu_ratios, samples)} " \
           "paired_median_hit_gpu_command_ratio_#{candidate_label}_over_baseline=#{median_ratio(hit_gpu_ratios, samples)}"
    ensure
      restore_environment("QWEN_IMAGE21_Q8_BATCH", prior_batch)
      restore_environment("QWEN_IMAGE21_Q8_REGISTER_REUSE", prior_reuse)
      restore_environment("QWEN_IMAGE21_PROFILE", prior_profile)
    end
  elsif ENV["QWEN_IMAGE21_Q8_BATCH_AB"]? == "1"
    baseline_build = [] of Float64
    baseline_hit = [] of Float64
    candidate_build = [] of Float64
    candidate_hit = [] of Float64
    (samples + 1).times do |index|
      modes = index.even? ? ["0", "1"] : ["1", "0"]
      outputs = Hash(String, {Array(Float32), Array(Float32)}).new
      walls = Hash(String, {Float64, Float64}).new
      modes.each do |mode|
        ENV["QWEN_IMAGE21_Q8_BATCH"] = mode
        ENV["QWEN_IMAGE21_PROFILE"] = "1"
        stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
        begin
          pair = run_pair(model, stack,
            ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
            hidden, encoder, shapes, mask, schedule.model_timestep(0), schedule.model_timestep(1),
            target_tokens)
          outputs[mode] = {pair[0], pair[1]}
          walls[mode] = {pair[2], pair[3]}
        ensure
          stack.close
        end
      end
      build_abs = outputs["0"][0].zip(outputs["1"][0]).max_of { |a, b| (a - b).abs }
      hit_abs = outputs["0"][1].zip(outputs["1"][1]).max_of { |a, b| (a - b).abs }
      abort "Q8_0 batch candidate changed output beyond tolerance" unless build_abs < 1.0e-3 && hit_abs < 1.0e-3
      next if index == 0 # Both routes and their pipelines are warmed once.
      base = walls["0"]
      candidate = walls["1"]
      baseline_build << base[0]
      baseline_hit << base[1]
      candidate_build << candidate[0]
      candidate_hit << candidate[1]
      puts "pair=#{index} order=#{modes.join("/")} " \
           "rebuild_base_ms=#{base[0].round(3)} rebuild_q8_ms=#{candidate[0].round(3)} " \
           "hit_base_ms=#{base[1].round(3)} hit_q8_ms=#{candidate[1].round(3)} " \
           "parity_max_abs_build=#{build_abs} parity_max_abs_hit=#{hit_abs}"
    end
    puts "paired_median_rebuild_ratio=#{median(candidate_build.zip(baseline_build).map { |a, b| a / b }).round(3)} " \
         "paired_median_hit_ratio=#{median(candidate_hit.zip(baseline_hit).map { |a, b| a / b }).round(3)}"
  else
    build_wall = [] of Float64
    hit_wall = [] of Float64
    build_stats = [] of ML::GGUF::QwenImage21MetalBlockStats
    hit_stats = [] of ML::GGUF::QwenImage21MetalBlockStats
    ordinary_build = nil.as(Array(Float32)?)
    ordinary_hit = nil.as(Array(Float32)?)
    ENV["QWEN_IMAGE21_PROFILE"] = "1"
    (samples + 1).times do |index|
      stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
      begin
        pair = run_pair(model, stack,
          ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
          hidden, encoder, shapes, mask, schedule.model_timestep(0), schedule.model_timestep(1),
          target_tokens)
        ordinary_build = pair[0] if index == 1
        ordinary_hit = pair[1] if index == 1
        next if index == 0
        build_wall << pair[2]
        hit_wall << pair[3]
        build_stats << pair[4]
        hit_stats << pair[5]
      ensure
        stack.close
      end
    end
    report("rebuild", build_wall, build_stats)
    report("hit", hit_wall, hit_stats)

    ENV["QWEN_IMAGE21_PROFILE"] = "phases"
    phase_samples.times do |index|
      stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
      begin
        pair = run_pair(model, stack,
          ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
          hidden, encoder, shapes, mask, schedule.model_timestep(0), schedule.model_timestep(1),
          target_tokens)
        build_abs = pair[0].zip(ordinary_build.not_nil!).max_of { |value, reference| (value - reference).abs }
        hit_abs = pair[1].zip(ordinary_hit.not_nil!).max_of { |value, reference| (value - reference).abs }
        abort "diagnostic command split changed output" unless build_abs < 1.0e-4 && hit_abs < 1.0e-4
        puts "phase_sample=#{index + 1} parity_max_abs_build=#{build_abs} parity_max_abs_hit=#{hit_abs}"
        report_phases("rebuild", pair[4])
        report_phases("hit", pair[5])
      ensure
        stack.close
      end
    end
  end
ensure
  model.close
end
