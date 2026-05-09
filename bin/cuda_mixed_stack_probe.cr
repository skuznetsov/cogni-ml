# CUDA mixed recurrent/full-attention stack scaffold probe for Qwen GGUF weights.
#
# This is a correctness scaffold, not an end-to-end decoder: it composes
# recurrent-layer and full-attention-layer CUDA runners in model layer order
# with device-resident hidden handoff between layers.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/cuda/qwen_recurrent_layer_runner"
require "../src/ml/cuda/qwen_full_attn_layer_runner"
require "../src/ml/cuda/qwen_output_head_runner"
require "../src/ml/cuda/qwen_mixed_stack_runner"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

def parse_layers(value : String) : Array(Int32)
  value.split(",").map(&.strip).reject(&.empty?).map(&.to_i)
end

def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  raise ArgumentError.new("size mismatch") unless a.size == b.size
  max = 0.0_f32
  a.each_with_index do |v, i|
    d = (v - b[i]).abs
    max = d if d > max
  end
  max
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64
  na = 0.0_f64
  nb = 0.0_f64
  a.each_with_index do |v, i|
    av = v.to_f64
    bv = b[i].to_f64
    dot += av * bv
    na += av * av
    nb += bv * bv
  end
  dot / Math.sqrt(na * nb)
end

def report_pair(name : String, gpu : Array(Float32), cpu : Array(Float32), lines : Array(String), max_allowed : Float32) : Bool
  cos = cosine(gpu, cpu)
  max_diff = max_abs_diff(gpu, cpu)
  ok = cos >= 0.99999 && max_diff <= max_allowed
  lines << "#{name}_cos=#{cos.round(8)}"
  lines << "#{name}_max_diff=#{max_diff}"
  lines << "#{name}_ok=#{ok}"
  ok
end

def rope_tables(tokens : Int32, start_pos : Int32, rope_dim : Int32, freq_base : Float32) : {Array(Float32), Array(Float32)}
  half = rope_dim // 2
  cos_table = Array(Float32).new(tokens * half, 0.0_f32)
  sin_table = Array(Float32).new(tokens * half, 0.0_f32)
  tokens.times do |tok|
    pos = start_pos + tok
    half.times do |i|
      freq = 1.0_f32 / (freq_base ** (2.0_f32 * i / rope_dim))
      theta = pos.to_f32 * freq
      cos_table[tok * half + i] = Math.cos(theta).to_f32
      sin_table[tok * half + i] = Math.sin(theta).to_f32
    end
  end
  {cos_table, sin_table}
end

def load_quant_weight(gguf : ML::GGUF::GGUFFile, name : String) : ML::GGUF::QuantWeight
  info = gguf.tensor(name) || raise "missing #{name}"
  raw = gguf.read_tensor_raw(info)
  in_dim = info.dims[0].to_i32
  out_dim = info.dims.size >= 2 ? info.dims[1].to_i32 : 1
  ML::GGUF::QuantWeight.new(raw, info.type, out_dim, in_dim)
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layers = [0, 1, 2, 3, 4]
seed = 41_u64
tokens = 2
start_pos = 0
max_seq = 16
warmup = 0
steady_reps = 0
steady_graph_reps = 0
read_logits = false
profile_phases = false
debug_readback = true
perf_only = false
all_layers = false
greedy_loop_tokens = 0
greedy_loop_graph = false
seed_token = 0

OptionParser.parse do |p|
  p.banner = "Usage: cuda_mixed_stack_probe [--model PATH] [--layers LIST] [--tokens N] [--start-pos N] [--max-seq N] [--seed N] [--warmup N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layers LIST", "Comma-separated layer ids in model order") { |v| layers = parse_layers(v) }
  p.on("--tokens N", "Sequence length for state progression") { |v| tokens = v.to_i }
  p.on("--start-pos N", "Starting decode position for full-attention KV cache") { |v| start_pos = v.to_i }
  p.on("--max-seq N", "KV cache capacity") { |v| max_seq = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--warmup N", "Untimed warmup stack runs") { |v| warmup = v.to_i }
  p.on("--steady-reps N", "After one reset priming run, time N runs without sequence/state reset; requires --perf-only") { |v| steady_reps = v.to_i }
  p.on("--steady-graph-reps N", "Capture one reset-free steady wave as a CUDA graph and replay it N times; requires --perf-only") { |v| steady_graph_reps = v.to_i }
  p.on("--read-logits", "Read full logits back for attribution; default reads resident CUDA top1 only") { read_logits = true }
  p.on("--profile-phases", "Synchronize after each runner and print attribution timings; slower than default") { profile_phases = true }
  p.on("--skip-debug-readback", "Read only output-head results; skip final hidden/state/KV debug buffers for perf attribution") { debug_readback = false }
  p.on("--perf-only", "Skip CPU reference and hidden/state checks; reports CUDA timing/top1 only") { perf_only = true }
  p.on("--all-layers", "Run all model layers instead of the explicit/default layer slice") { all_layers = true }
  p.on("--greedy-loop-tokens N", "Run an embedding-driven greedy decode loop for N generated tokens; forces --tokens=1") { |v| greedy_loop_tokens = v.to_i }
  p.on("--greedy-loop-graph", "Capture the reset-free greedy-loop body as a CUDA graph and replay it after the first token") { greedy_loop_graph = true }
  p.on("--seed-token ID", "Seed token id for --greedy-loop-tokens") { |v| seed_token = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layers must not be empty" if layers.empty?
raise "layers must be non-negative" unless layers.all? { |layer| layer >= 0 }
raise "layers must be strictly increasing for this probe" unless layers.each_cons(2).all? { |pair| pair[0] < pair[1] }
raise "tokens must be positive" unless tokens > 0
raise "start-pos must be non-negative" unless start_pos >= 0
raise "max-seq must cover start-pos + tokens" unless max_seq >= start_pos + tokens
raise "warmup must be non-negative" unless warmup >= 0
raise "steady-reps must be non-negative" unless steady_reps >= 0
raise "steady-graph-reps must be non-negative" unless steady_graph_reps >= 0
raise "--steady-reps requires --perf-only" if steady_reps > 0 && !perf_only
raise "--steady-reps does not support --profile-phases" if steady_reps > 0 && profile_phases
raise "--steady-graph-reps requires --perf-only" if steady_graph_reps > 0 && !perf_only
raise "--steady-graph-reps does not support --profile-phases" if steady_graph_reps > 0 && profile_phases
raise "use either --steady-reps or --steady-graph-reps, not both" if steady_reps > 0 && steady_graph_reps > 0
raise "--greedy-loop-tokens must be non-negative" unless greedy_loop_tokens >= 0
raise "--greedy-loop-graph requires --greedy-loop-tokens" if greedy_loop_graph && greedy_loop_tokens == 0
if greedy_loop_tokens > 0
  raise "--greedy-loop-tokens currently requires --perf-only; it is a semantic timing harness, not a CPU oracle" unless perf_only
  raise "--greedy-loop-tokens is incompatible with --steady-reps/--steady-graph-reps" if steady_reps > 0 || steady_graph_reps > 0
  raise "--greedy-loop-tokens is incompatible with --profile-phases" if profile_phases
  raise "--greedy-loop-tokens requires --tokens=1" unless tokens == 1
  raise "max-seq must cover start-pos + greedy-loop-tokens" unless max_seq >= start_pos + greedy_loop_tokens
end

eps = 1.0e-6_f32
gguf = ML::GGUF::GGUFFile.new(model)
hparams = ML::GGUF::Qwen35Hparams.new(gguf)
layers = (0...hparams.n_layer).map(&.to_i32) if all_layers
layers.each { |layer| raise "layer #{layer} out of range" unless layer < hparams.n_layer }
hidden = hparams.n_embd
debug_readback = false if perf_only
read_logits = false if perf_only

rng = Random.new(seed)
token_embd = load_quant_weight(gguf, "token_embd.weight")
raise "seed-token #{seed_token} out of range" if seed_token < 0 || seed_token >= token_embd.out_dim
xs = if greedy_loop_tokens > 0
       ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, seed_token)
     else
       Array(Float32).new(tokens * hidden) { ((rng.next_float - 0.5) * 0.2).to_f32 }
     end

recurrent_weights = {} of Int32 => ML::CUDA::QwenRecurrentLayerRunner::Weights
full_weights = {} of Int32 => ML::CUDA::QwenFullAttnLayerRunner::Weights
head_weights = ML::CUDA::QwenOutputHeadRunner::Weights.load(gguf)
conv_state_inits = {} of Int32 => Array(Float32)
ssm_state_inits = {} of Int32 => Array(Float32)

layers.each do |layer|
  if hparams.full_attention?(layer)
    full_weights[layer] = ML::CUDA::QwenFullAttnLayerRunner::Weights.load(gguf, layer)
  else
    weights = ML::CUDA::QwenRecurrentLayerRunner::Weights.load(gguf, layer, eps)
    recurrent_weights[layer] = weights
    conv_state_inits[layer] = Array(Float32).new((weights.conv_k - 1) * weights.qkv_dim) do
      greedy_loop_tokens > 0 ? 0.0_f32 : ((rng.next_float - 0.5) * 0.05).to_f32
    end
    ssm_state_inits[layer] = Array(Float32).new(weights.h_v * weights.s * weights.s) do
      greedy_loop_tokens > 0 ? 0.0_f32 : ((rng.next_float - 0.5) * 0.05).to_f32
    end
  end
end

cpu_current = xs.dup
cpu_states = Array(ML::GGUF::Qwen35CPU::LayerState).new(hparams.n_layer) { ML::GGUF::Qwen35CPU::LayerState.new }
cpu_ms = 0.0
cpu_top1_ids = [] of Int32
cpu_logits_all = [] of Float32
cpu_weights = nil.as(ML::GGUF::Qwen35Weights?)

unless perf_only
  cpu_weights = ML::GGUF::Qwen35Weights.new(gguf, hparams)
  recurrent_weights.each_key do |layer|
    cpu_states[layer].conv_state = conv_state_inits[layer].dup
    cpu_states[layer].ssm_state = ssm_state_inits[layer].dup
  end

  cpu_t0 = Time.instant
  layers.each do |layer|
    lw = cpu_weights.not_nil!.layers[layer]
    out = Array(Float32).new(tokens * hidden, 0.0_f32)
    tokens.times do |tok|
      row = cpu_current[tok * hidden, hidden]
      y = case lw
          in ML::GGUF::Qwen35FullAttnWeights
            ML::GGUF::Qwen35CPU.forward_full_attn_layer(row, start_pos + tok, lw, cpu_states[layer], hparams, max_seq)
          in ML::GGUF::Qwen35RecurrentWeights
            ML::GGUF::Qwen35CPU.forward_recurrent_layer(row, 0, lw, cpu_states[layer], hparams, max_seq)
          end
      hidden.times { |i| out[tok * hidden + i] = y[i] }
    end
    cpu_current = out
  end
  cpu_ms = (Time.instant - cpu_t0).total_milliseconds
  cpu_logits_all = read_logits ? Array(Float32).new(tokens * head_weights.vocab, 0.0_f32) : [] of Float32
  cpu_top1_ids = Array(Int32).new(tokens)
  tokens.times do |tok|
    row = cpu_current[tok * hidden, hidden]
    normed = ML::GGUF::Qwen35CPU.rms_norm(row, head_weights.norm, hparams.rms_eps)
    logits = ML::GGUF::QuantMatmul.matmul_add(normed, 1, head_weights.hidden,
      head_weights.output_raw, head_weights.output_type, head_weights.vocab,
      Array(Float32).new(head_weights.vocab, 0.0_f32))
    best_id = 0
    best = logits[0]
    head_weights.vocab.times do |i|
      cpu_logits_all[tok * head_weights.vocab + i] = logits[i] if read_logits
      if logits[i] > best
        best = logits[i]
        best_id = i
      end
    end
    cpu_top1_ids << best_id
  end
end

cuda_ctx = nil.as(ML::CUDA::Context?)
runners = [] of ML::CUDA::QwenMixedStackRunner::LayerRunner
head = nil.as(ML::CUDA::QwenOutputHeadRunner?)
stack = nil.as(ML::CUDA::QwenMixedStackRunner?)
final_gpu_all = Array(Float32).new(tokens * hidden, 0.0_f32)

begin
  cuda_ctx = ML::CUDA::Context.create
  cos_table, sin_table = rope_tables(tokens, start_pos, hparams.rope_dim_count, hparams.rope_freq_base)

  layers.each_with_index do |layer, idx|
    layer_input = idx == 0 ? xs : Array(Float32).new(tokens * hidden, 0.0_f32)
    if hparams.full_attention?(layer)
      runners << ML::CUDA::QwenFullAttnLayerRunner.from_weights(full_weights[layer], tokens, max_seq, start_pos,
        hparams.n_head, hparams.n_head_kv, hparams.head_dim, hparams.rope_dim_count, hparams.rms_eps,
        layer_input, cos_table, sin_table)
    else
      runners << ML::CUDA::QwenRecurrentLayerRunner.from_weights(recurrent_weights[layer], tokens, layer_input,
        conv_state_inits[layer], ssm_state_inits[layer])
    end
  end
  head = ML::CUDA::QwenOutputHeadRunner.from_weights(head_weights, tokens,
    Array(Float32).new(tokens * hidden, 0.0_f32), hparams.rms_eps, read_logits: read_logits)
  output_head = head.not_nil!
  stack = ML::CUDA::QwenMixedStackRunner.new(layers, runners, output_head, tokens, hidden, xs)
  mixed_stack = stack.not_nil!

  weight_upload_ms = mixed_stack.upload_weights(profile: profile_phases)

  measured_tokens = tokens
  greedy_gpu_ids = [] of Int32
  greedy_position_ms = 0.0
  greedy_embedding_ms = 0.0
  greedy_body_ms = 0.0
  greedy_read_ms = 0.0
  if greedy_loop_tokens > 0
    warmup.times do
      warm_token = seed_token
      Math.min(greedy_loop_tokens, 2).times do |i|
        pos = start_pos + i
        cos_pos, sin_pos = rope_tables(1, pos, hparams.rope_dim_count, hparams.rope_freq_base)
        mixed_stack.update_decode_position(pos, cos_pos, sin_pos)
        mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, warm_token))
        mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: i == 0)
        mixed_stack.read_head_outputs
        warm_token = output_head.top1_ids[0]
      end
    end

    graph_stream = nil.as(ML::CUDA::CUDAStream?)
    graph = nil.as(ML::CUDA::CUDAGraph?)
    graph_exec = nil.as(ML::CUDA::CUDAGraphExec?)
    if greedy_loop_graph && greedy_loop_tokens > 1
      graph_stream = ML::CUDA::CUDAStream.new
      ML::CUDA.with_stream(graph_stream.not_nil!) do
        graph_stream.not_nil!.begin_capture
        mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: false,
          sync_end: false, read_head_outputs: false)
        graph = graph_stream.not_nil!.end_capture
      end
      graph_exec = graph.not_nil!.instantiate
      graph.not_nil!.close
      graph = nil
      graph_exec.not_nil!.upload(graph_stream.not_nil!)
      graph_stream.not_nil!.synchronize
    end

    gpu_token = seed_token
    gpu_t0 = Time.instant
    begin
      greedy_loop_tokens.times do |i|
        pos = start_pos + i
        t_position = Time.instant
        cos_pos, sin_pos = rope_tables(1, pos, hparams.rope_dim_count, hparams.rope_freq_base)
        mixed_stack.update_decode_position(pos, cos_pos, sin_pos)
        greedy_position_ms += (Time.instant - t_position).total_milliseconds

        t_embedding = Time.instant
        mixed_stack.upload_first_sequence_input(ML::GGUF::Qwen35CPU.embedding_lookup(token_embd, gpu_token))
        greedy_embedding_ms += (Time.instant - t_embedding).total_milliseconds

        t_body = Time.instant
        if greedy_loop_graph && i > 0
          graph_exec.not_nil!.launch(graph_stream.not_nil!)
          graph_stream.not_nil!.synchronize
        else
          mixed_stack.run_sequence(profile_phases: false, debug_readback: debug_readback, reset_sequence: i == 0,
            read_head_outputs: false)
        end
        greedy_body_ms += (Time.instant - t_body).total_milliseconds

        t_read = Time.instant
        mixed_stack.read_head_outputs
        greedy_read_ms += (Time.instant - t_read).total_milliseconds
        gpu_token = output_head.top1_ids[0]
        greedy_gpu_ids << gpu_token
      end
    ensure
      graph_exec.try(&.close)
      graph.try(&.close)
      graph_stream.try(&.close)
    end
    gpu_ms = (Time.instant - gpu_t0).total_milliseconds
    measured_tokens = greedy_loop_tokens
  else
    warmup.times { mixed_stack.run_sequence(profile_phases: false, debug_readback: false) }

    if steady_graph_reps > 0
      mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: true)
      stream = ML::CUDA::CUDAStream.new
      graph = nil.as(ML::CUDA::CUDAGraph?)
      graph_exec = nil.as(ML::CUDA::CUDAGraphExec?)
      begin
        ML::CUDA.with_stream(stream) do
          stream.begin_capture
          mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: false,
            sync_end: false, read_head_outputs: false)
          graph = stream.end_capture
        end
        graph_exec = graph.not_nil!.instantiate
        graph.not_nil!.close
        graph_exec.not_nil!.upload(stream)
        stream.synchronize
        t_graph = Time.instant
        steady_graph_reps.times do
          graph_exec.not_nil!.launch(stream)
        end
        stream.synchronize
        gpu_ms = (Time.instant - t_graph).total_milliseconds
        measured_tokens = tokens * steady_graph_reps
        mixed_stack.read_head_outputs
      ensure
        graph_exec.try(&.close)
        graph.try(&.close)
        stream.close
      end
    elsif steady_reps > 0
      # Prime device inputs and decode states once, then measure the steady path
      # where recurrent/KV state stays resident across decode steps.
      mixed_stack.run_sequence(profile_phases: false, debug_readback: false, reset_sequence: true)
      t_steady = Time.instant
      steady_reps.times do
        mixed_stack.run_sequence(profile_phases: false, debug_readback: debug_readback, reset_sequence: false)
      end
      gpu_ms = (Time.instant - t_steady).total_milliseconds
      measured_tokens = tokens * steady_reps
    else
      gpu_ms = mixed_stack.run_sequence(profile_phases: profile_phases, debug_readback: debug_readback)
    end
  end
  final_gpu_all = mixed_stack.final_gpu_all if debug_readback

  lines = [] of String
  ok = true
  if greedy_loop_tokens > 0
    if perf_only
      lines << "perf_only=true"
    elsif debug_readback
      lines << "final_all_check=skipped_for_greedy_loop"
    else
      lines << "debug_readback=false"
    end
  elsif perf_only
    lines << "perf_only=true"
  elsif debug_readback
    ok = report_pair("final_all", final_gpu_all, cpu_current, lines, 1.0e-2_f32)
  else
    lines << "debug_readback=false"
  end
  gpu_top1_ids = greedy_loop_tokens > 0 ? greedy_gpu_ids : output_head.top1_ids
  top1_ok = perf_only || gpu_top1_ids == cpu_top1_ids
  if read_logits && greedy_loop_tokens == 0
    logits_ok = report_pair("logits", output_head.logits_gpu_all, cpu_logits_all, lines, 5.0e-3_f32)
    ok = ok && logits_ok
  else
    lines << "logits_readback=false"
  end
  lines << "top1_gpu=#{gpu_top1_ids.join(",")}"
  lines << "top1_cpu=#{perf_only ? "skipped" : cpu_top1_ids.join(",")}"
  lines << "top1_values_gpu=#{output_head.top1_values_gpu.map { |v| v.round(6) }.join(",")}"
  lines << "top1_ok=#{perf_only ? "skipped" : top1_ok}"
  ok = ok && top1_ok
  if debug_readback && !perf_only && greedy_loop_tokens == 0
    runners.each_with_index do |runner, idx|
      layer = layers[idx]
      case runner
      in ML::CUDA::QwenRecurrentLayerRunner
        conv_ok = report_pair("layer#{layer}_conv_state", runner.conv_state_gpu, cpu_states[layer].conv_state.not_nil!, lines, 2.0e-5_f32)
        ssm_ok = report_pair("layer#{layer}_ssm_state", runner.ssm_state_gpu, cpu_states[layer].ssm_state.not_nil!, lines, 1.0e-3_f32)
        ok = ok && conv_ok && ssm_ok
      in ML::CUDA::QwenFullAttnLayerRunner
        kv = runner.kv
        k_cpu = cpu_states[layer].k_cache || Array(Float32).new(max_seq * hparams.n_head_kv * hparams.head_dim, 0.0_f32)
        v_cpu = cpu_states[layer].v_cache || Array(Float32).new(max_seq * hparams.n_head_kv * hparams.head_dim, 0.0_f32)
        k_ok = report_pair("layer#{layer}_k_cache", kv.k_cache_gpu, k_cpu, lines, 2.0e-4_f32)
        v_ok = report_pair("layer#{layer}_v_cache", kv.v_cache_gpu, v_cpu, lines, 1.0e-3_f32)
        ok = ok && k_ok && v_ok
      end
    end
  end

  puts "device=#{cuda_ctx.device_name}"
  puts "compute_capability=#{cuda_ctx.compute_capability_major}.#{cuda_ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layers=#{layers.join(",")}"
  puts "tokens=#{tokens}"
  puts "start_pos=#{start_pos}"
  puts "max_seq=#{max_seq}"
  puts "warmup=#{warmup}"
  puts "steady_reps=#{steady_reps}"
  puts "steady_graph_reps=#{steady_graph_reps}"
  puts "greedy_loop_tokens=#{greedy_loop_tokens}"
  puts "greedy_loop_graph=#{greedy_loop_graph}"
  puts "seed_token=#{seed_token}"
  puts "read_logits=#{read_logits}"
  puts "profile_phases=#{profile_phases}"
  puts "debug_readback=#{debug_readback}"
  puts "perf_only=#{perf_only}"
  puts "hidden=#{hidden}"
  puts "vocab=#{head_weights.vocab}"
  puts "weight_upload_ms=#{weight_upload_ms.round(3)}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cuda_ms_per_token=#{(gpu_ms / measured_tokens).round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cpu_ms_per_token=#{(cpu_ms / (greedy_loop_tokens > 0 ? greedy_loop_tokens : tokens)).round(3)}"
  if greedy_loop_tokens > 0
    denom = greedy_loop_tokens.to_f64
    puts "greedy_position_ms=#{greedy_position_ms.round(3)}"
    puts "greedy_position_ms_per_token=#{(greedy_position_ms / denom).round(3)}"
    puts "greedy_embedding_ms=#{greedy_embedding_ms.round(3)}"
    puts "greedy_embedding_ms_per_token=#{(greedy_embedding_ms / denom).round(3)}"
    puts "greedy_body_ms=#{greedy_body_ms.round(3)}"
    puts "greedy_body_ms_per_token=#{(greedy_body_ms / denom).round(3)}"
    puts "greedy_read_ms=#{greedy_read_ms.round(3)}"
    puts "greedy_read_ms_per_token=#{(greedy_read_ms / denom).round(3)}"
  end
  mixed_stack.phase_lines.each { |line| puts line }
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  stack.try(&.close)
  cuda_ctx.try(&.close)
  gguf.close
end
