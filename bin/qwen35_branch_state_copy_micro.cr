require "option_parser"

require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"

DEFAULT_9B_MODEL  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_27B_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"

enum CopyMode
  RecOnly
  Used
  Full
end

private def parse_i32_list(value : String) : Array(Int32)
  value.split(',').reject(&.empty?).map(&.to_i32)
end

private def percentile(values : Array(Float64), q : Float64) : Float64
  return 0.0 if values.empty?

  sorted = values.sort
  idx = ((sorted.size - 1) * q).round.to_i
  sorted[idx]
end

private def model_label(path : String) : String
  case path
  when /Qwen3\.5-9B/
    "9B"
  when /Qwen3\.6-27B/
    "27B"
  else
    File.basename(path)
  end
end

private def checked_i32_bytes(bytes : Int64) : Int32
  raise "copy byte size exceeds Int32 encoder limit: #{bytes}" if bytes > Int32::MAX
  bytes.to_i32
end

private def hparams_from_model(path : String) : ML::GGUF::Qwen35Hparams
  gguf = ML::GGUF::GGUFFile.new(path)
  ML::GGUF::Qwen35Hparams.new(gguf)
end

private def qkv_dim(hp : ML::GGUF::Qwen35Hparams) : Int32
  2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
end

private def rec_conv_bytes(hp : ML::GGUF::Qwen35Hparams) : Int64
  ((hp.ssm_conv_kernel - 1) * qkv_dim(hp)).to_i64 * sizeof(Float32)
end

private def rec_ssm_bytes(hp : ML::GGUF::Qwen35Hparams) : Int64
  (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)
end

private def kv_row_bytes(hp : ML::GGUF::Qwen35Hparams) : Int64
  (hp.head_dim * hp.n_head_kv).to_i64 * sizeof(Float32)
end

private def used_kv_bytes(hp : ML::GGUF::Qwen35Hparams, pos : Int32) : Int64
  pos.to_i64 * kv_row_bytes(hp)
end

private def bytes_per_branch(hp : ML::GGUF::Qwen35Hparams, max_seq : Int32, pos : Int32, mode : CopyMode) : Int64
  rec = hp.recurrent_layers.size.to_i64 * (rec_conv_bytes(hp) + rec_ssm_bytes(hp))
  return rec if mode.rec_only?

  kv_bytes = mode.full? ? max_seq.to_i64 * kv_row_bytes(hp) : used_kv_bytes(hp, pos)
  rec + hp.full_attention_layers.size.to_i64 * 2_i64 * kv_bytes
end

private def prepare_state(hp : ML::GGUF::Qwen35Hparams, max_seq : Int32, pos : Int32) : ML::GGUF::Qwen35CPU::State
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
  state.layers.each { |layer| layer.position = pos }
  state
end

private def fill_state!(state : ML::GGUF::Qwen35CPU::State, value : UInt8) : Nil
  ML::Metal::Dispatch.execute_blit do |enc|
    state.layers.each do |layer|
      if buf = layer.k_cache_buf
        enc.fill_buffer(buf, value, 0, checked_i32_bytes(buf.size))
      end
      if buf = layer.v_cache_buf
        enc.fill_buffer(buf, value, 0, checked_i32_bytes(buf.size))
      end
      if buf = layer.conv_state_buf
        enc.fill_buffer(buf, value, 0, checked_i32_bytes(buf.size))
      end
      if buf = layer.ssm_state_buf
        enc.fill_buffer(buf, value, 0, checked_i32_bytes(buf.size))
      end
    end
  end
end

private def copy_branch_blit!(src : ML::GGUF::Qwen35CPU::State,
                              dst : ML::GGUF::Qwen35CPU::State,
                              hp : ML::GGUF::Qwen35Hparams,
                              pos : Int32,
                              mode : CopyMode,
                              enc : ML::Metal::BlitEncoder) : Nil
  ML::GGUF::Qwen35CPU.encode_state_metal_used_copy!(
    enc, dst, src, hp,
    used_tokens: pos,
    rec_only: mode.rec_only?,
    full_kv_capacity: mode.full?
  )
end

private def copy_branches_blit!(src : ML::GGUF::Qwen35CPU::State,
                                dsts : Array(ML::GGUF::Qwen35CPU::State),
                                hp : ML::GGUF::Qwen35Hparams,
                                pos : Int32,
                                mode : CopyMode) : Nil
  ML::Metal::Dispatch.execute_blit do |enc|
    dsts.each do |dst|
      copy_branch_blit!(src, dst, hp, pos, mode, enc)
    end
  end
end

private def copy_branch_memcpy_used!(src : ML::GGUF::Qwen35CPU::State,
                                     dst : ML::GGUF::Qwen35CPU::State,
                                     hp : ML::GGUF::Qwen35Hparams,
                                     pos : Int32,
                                     mode : CopyMode) : Nil
  src.layers.each_with_index do |src_layer, il|
    dst_layer = dst.layers[il]
    dst_layer.position = src_layer.position

    if hp.full_attention?(il)
      next if mode.rec_only?

      bytes = mode.full? ? src_layer.k_cache_buf.not_nil!.size : used_kv_bytes(hp, pos)
      next if bytes <= 0

      dst_layer.k_cache_buf.not_nil!.copy_from(src_layer.k_cache_buf.not_nil!, bytes)
      dst_layer.v_cache_buf.not_nil!.copy_from(src_layer.v_cache_buf.not_nil!, bytes)
    else
      dst_layer.conv_state_buf.not_nil!.copy_from(src_layer.conv_state_buf.not_nil!, src_layer.conv_state_buf.not_nil!.size)
      dst_layer.ssm_state_buf.not_nil!.copy_from(src_layer.ssm_state_buf.not_nil!, src_layer.ssm_state_buf.not_nil!.size)
    end
  end
end

private def copy_branches_memcpy_used!(src : ML::GGUF::Qwen35CPU::State,
                                       dsts : Array(ML::GGUF::Qwen35CPU::State),
                                       hp : ML::GGUF::Qwen35Hparams,
                                       pos : Int32,
                                       mode : CopyMode) : Nil
  dsts.each do |dst|
    copy_branch_memcpy_used!(src, dst, hp, pos, mode)
  end
end

private def copy_branches_memcpy_full_state!(src : ML::GGUF::Qwen35CPU::State,
                                             dsts : Array(ML::GGUF::Qwen35CPU::State)) : Nil
  dsts.each(&.copy_from!(src))
end

private def verify_first_bytes!(src : ML::GGUF::Qwen35CPU::State,
                                dst : ML::GGUF::Qwen35CPU::State,
                                hp : ML::GGUF::Qwen35Hparams,
                                mode : CopyMode) : Nil
  src.layers.each_with_index do |src_layer, il|
    dst_layer = dst.layers[il]
    if hp.full_attention?(il)
      next if mode.rec_only?

      raise "k byte mismatch at layer #{il}" unless src_layer.k_cache_buf.not_nil!.contents.as(Pointer(UInt8)).value == dst_layer.k_cache_buf.not_nil!.contents.as(Pointer(UInt8)).value
      raise "v byte mismatch at layer #{il}" unless src_layer.v_cache_buf.not_nil!.contents.as(Pointer(UInt8)).value == dst_layer.v_cache_buf.not_nil!.contents.as(Pointer(UInt8)).value
    else
      raise "conv byte mismatch at layer #{il}" unless src_layer.conv_state_buf.not_nil!.contents.as(Pointer(UInt8)).value == dst_layer.conv_state_buf.not_nil!.contents.as(Pointer(UInt8)).value
      raise "ssm byte mismatch at layer #{il}" unless src_layer.ssm_state_buf.not_nil!.contents.as(Pointer(UInt8)).value == dst_layer.ssm_state_buf.not_nil!.contents.as(Pointer(UInt8)).value
    end
  end
end

private def measure(label : String,
                    branches : Int32,
                    reps : Int32,
                    warmup : Int32,
                    total_bytes : Int64,
                    &block : -> Nil) : Nil
  times = [] of Float64
  (warmup + reps).times do |i|
    t0 = Time.instant
    yield
    ms = (Time.instant - t0).total_milliseconds
    times << ms if i >= warmup
  end

  p50 = percentile(times, 0.50)
  p90 = percentile(times, 0.90)
  p99 = percentile(times, 0.99)
  bandwidth = p50 > 0 ? total_bytes.to_f64 / (p50 / 1000.0) / 1_000_000_000.0 : 0.0
  puts "branch_copy label=#{label} branches=#{branches} bytes=#{total_bytes} mib=#{(total_bytes / 1024.0 / 1024.0).round(3)} p50_ms=#{p50.round(4)} p90_ms=#{p90.round(4)} p99_ms=#{p99.round(4)} gbps=#{bandwidth.round(2)}"
end

models = [DEFAULT_9B_MODEL]
max_seq = 2048
positions = [64, 512, 2048]
branches_list = [1, 2, 4, 8]
reps = 9
warmup = 2

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_branch_state_copy_micro [options]"
  p.on("--model=PATH", "GGUF model path") { |v| models = [v] }
  p.on("--models=LIST", "Comma-separated GGUF model paths; use 'default' for local 9B,27B") do |v|
    models = if v == "default"
               [DEFAULT_9B_MODEL, DEFAULT_27B_MODEL]
             else
               v.split(',').reject(&.empty?)
             end
  end
  p.on("--max-seq=N", "Allocated KV max sequence") { |v| max_seq = v.to_i32 }
  p.on("--positions=LIST", "Comma-separated used positions") { |v| positions = parse_i32_list(v) }
  p.on("--branches=LIST", "Comma-separated branch counts") { |v| branches_list = parse_i32_list(v) }
  p.on("--reps=N", "Timed repetitions") { |v| reps = v.to_i32 }
  p.on("--warmup=N", "Warmup repetitions") { |v| warmup = v.to_i32 }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "Metal is not available" unless ML::GGUF::Qwen35Metal.available?

models.each do |model|
  unless File.exists?(model)
    STDERR.puts "skip missing model=#{model}"
    next
  end

  hp = hparams_from_model(model)
  raise "max_seq must be positive" unless max_seq > 0

  positions.each do |pos|
    raise "position #{pos} exceeds max_seq #{max_seq}" if pos > max_seq
  end

  max_branches = branches_list.max? || 1
  base = prepare_state(hp, max_seq, 0)
  branches = Array(ML::GGUF::Qwen35CPU::State).new(max_branches) { prepare_state(hp, max_seq, 0) }
  fill_state!(base, 0x5a_u8)
  branches.each { |branch| fill_state!(branch, 0x00_u8) }

  puts "branch_copy_model label=#{model_label(model)} layers=#{hp.n_layer} rec_layers=#{hp.recurrent_layers.size} full_layers=#{hp.full_attention_layers.size} max_seq=#{max_seq} rec_ssm_mib=#{(rec_ssm_bytes(hp) / 1024.0 / 1024.0).round(3)} rec_conv_kib=#{(rec_conv_bytes(hp) / 1024.0).round(1)} kv_row_kib=#{(kv_row_bytes(hp) / 1024.0).round(1)}"

  positions.each do |pos|
    base.layers.each { |layer| layer.position = pos }

    branches_list.each do |branch_count|
      dsts = branches[0, branch_count]

      rec_bytes = bytes_per_branch(hp, max_seq, pos, CopyMode::RecOnly) * branch_count
      used_bytes = bytes_per_branch(hp, max_seq, pos, CopyMode::Used) * branch_count
      full_bytes = bytes_per_branch(hp, max_seq, pos, CopyMode::Full) * branch_count

      copy_branches_blit!(base, dsts, hp, pos, CopyMode::RecOnly)
      verify_first_bytes!(base, dsts[0], hp, CopyMode::RecOnly)
      measure("blit_rec_only:#{model_label(model)}:pos#{pos}", branch_count, reps, warmup, rec_bytes) do
        copy_branches_blit!(base, dsts, hp, pos, CopyMode::RecOnly)
      end

      copy_branches_blit!(base, dsts, hp, pos, CopyMode::Used)
      verify_first_bytes!(base, dsts[0], hp, CopyMode::Used)
      measure("blit_used:#{model_label(model)}:pos#{pos}", branch_count, reps, warmup, used_bytes) do
        copy_branches_blit!(base, dsts, hp, pos, CopyMode::Used)
      end

      copy_branches_memcpy_used!(base, dsts, hp, pos, CopyMode::Used)
      verify_first_bytes!(base, dsts[0], hp, CopyMode::Used)
      measure("memcpy_used:#{model_label(model)}:pos#{pos}", branch_count, reps, warmup, used_bytes) do
        copy_branches_memcpy_used!(base, dsts, hp, pos, CopyMode::Used)
      end

      copy_branches_memcpy_full_state!(base, dsts)
      verify_first_bytes!(base, dsts[0], hp, CopyMode::Full)
      measure("memcpy_full_state:#{model_label(model)}:pos#{pos}", branch_count, reps, warmup, full_bytes) do
        copy_branches_memcpy_full_state!(base, dsts)
      end
    end
  end
end
