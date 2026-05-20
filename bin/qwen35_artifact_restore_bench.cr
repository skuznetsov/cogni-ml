# Metal artifact restore economics probe for Qwen 3.5/3.6 state snapshots.
#
# This is a diagnostic bench, not a serving path. It measures raw, BF16, and
# block-INT8 recurrent artifact restore with the same captured prompt state so
# product cache policy can be driven by local timing and parity evidence.

require "option_parser"
require "file_utils"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
max_seq = 128
tokens = 32
warmups = 2
iters = 8
int8_block = 8
skip_hash = false

parser = OptionParser.new do |p|
  p.banner = "Usage: qwen35_artifact_restore_bench [options]"
  p.on("--model PATH", "Target GGUF model path (default: QWEN35_MODEL or local 9B)") { |v| model_path = v }
  p.on("--max-seq N", "State max sequence length (default: 128)") { |v| max_seq = v.to_i }
  p.on("--tokens N", "Prompt tokens used to build the source state (default: 32)") { |v| tokens = v.to_i }
  p.on("--warmups N", "Unmeasured restore iterations per route (default: 2)") { |v| warmups = v.to_i }
  p.on("--iters N", "Measured restore iterations per route (default: 8)") { |v| iters = v.to_i }
  p.on("--int8-block N", "Block size for recurrent-int8 artifacts (default: 8)") { |v| int8_block = v.to_i }
  p.on("--skip-hash", "Skip SHA-256 during read+restore timing") { skip_hash = true }
  p.on("-h", "--help", "Show this help") do
    puts p
    exit
  end
end
parser.parse(ARGV)

raise "--max-seq must be positive" unless max_seq > 0
raise "--tokens must be positive" unless tokens > 0
raise "--tokens must fit in --max-seq" unless tokens < max_seq
raise "--warmups must be non-negative" unless warmups >= 0
raise "--iters must be positive" unless iters > 0
raise "--int8-block must be positive" unless int8_block > 0
raise "Metal is required for this bench" unless ML::GGUF::Qwen35Metal.available?
raise "model not found: #{model_path}" unless File.exists?(model_path)

def elapsed_ms(&)
  start = Time.instant
  yield
  (Time.instant - start).total_milliseconds
end

def mean(values : Array(Float64)) : Float64
  values.sum / values.size
end

def bench(label : String, warmups : Int32, iters : Int32, &block : -> Nil) : Float64
  warmups.times { block.call }
  times = Array(Float64).new(iters)
  iters.times do
    times << elapsed_ms { block.call }
  end
  ms = mean(times)
  puts "#{label}_ms=#{ms.round(3)}"
  ms
end

def prepared_state(hp : ML::GGUF::Qwen35Hparams, max_seq : Int32) : ML::GGUF::Qwen35CPU::State
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
  state
end

def restore_top1(weights : ML::GGUF::Qwen35Weights,
                 snapshot : ML::GGUF::Qwen35StateSnapshot::Snapshot,
                 hp : ML::GGUF::Qwen35Hparams,
                 max_seq : Int32,
                 input_token : Int32,
                 pos : Int32) : {Int32, Float32}
  state = prepared_state(hp, max_seq)
  ML::GGUF::Qwen35StateSnapshot.restore_into(snapshot, hp, state, prefer_metal: true)
  ML::GGUF::Qwen35CPU.forward_top1(weights, input_token, pos, state)
end

def restore_encoded_top1(weights : ML::GGUF::Qwen35Weights,
                         encoded : ML::GGUF::Qwen35StateSnapshot::EncodedSnapshot,
                         hp : ML::GGUF::Qwen35Hparams,
                         max_seq : Int32,
                         input_token : Int32,
                         pos : Int32) : {Int32, Float32}
  state = prepared_state(hp, max_seq)
  ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(encoded, hp, state, prefer_metal: true)
  ML::GGUF::Qwen35CPU.forward_top1(weights, input_token, pos, state)
end

weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
hp = weights.hparams

seed_tokens = [
  760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 1614_i32, 13_i32,
  198_i32, 785_i32, 6722_i32, 315_i32, 9625_i32, 374_i32, 11751_i32, 11_i32,
]
prompt = Array(Int32).new(tokens) { |i| seed_tokens[i % seed_tokens.size] }
continuation_token = 11751_i32
continuation_pos = prompt.size.to_i32

source = prepared_state(hp, max_seq)
prompt.each_with_index do |token_id, pos|
  ML::GGUF::Qwen35CPU.forward_top1(weights, token_id, pos.to_i32, source)
end
snapshot = ML::GGUF::Qwen35StateSnapshot.capture(source)

root = File.tempname("qwen35-artifact-restore-bench")
Dir.mkdir_p(root)
raw_path = File.join(root, "raw.qkv")
bf16_path = File.join(root, "bf16.qkv")
i8_path = File.join(root, "i8.qkv")

begin
  raw_info = ML::GGUF::Qwen35StateSnapshot.write_artifact(snapshot, raw_path)
  bf16_info = ML::GGUF::Qwen35StateSnapshot.write_artifact(snapshot, bf16_path, artifact_codec: "recurrent-bf16")
  i8_info = ML::GGUF::Qwen35StateSnapshot.write_artifact(snapshot, i8_path, artifact_codec: "recurrent-int8", artifact_codec_block: int8_block)

  raw_snapshot = ML::GGUF::Qwen35StateSnapshot.read_artifact(raw_path, expected_sha256: raw_info.sha256)
  bf16_encoded = ML::GGUF::Qwen35StateSnapshot.read_artifact_encoded(bf16_path, expected_sha256: bf16_info.sha256, expected_codec: "recurrent-bf16")
  i8_encoded = ML::GGUF::Qwen35StateSnapshot.read_artifact_encoded(i8_path, expected_sha256: i8_info.sha256, expected_codec: "recurrent-int8", expected_codec_block: int8_block)

  source_top, source_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, continuation_token, continuation_pos, source)
  raw_top, raw_logit = restore_top1(weights, raw_snapshot, hp, max_seq, continuation_token, continuation_pos)
  bf16_top, bf16_logit = restore_encoded_top1(weights, bf16_encoded, hp, max_seq, continuation_token, continuation_pos)
  i8_top, i8_logit = restore_encoded_top1(weights, i8_encoded, hp, max_seq, continuation_token, continuation_pos)

  puts "model=#{model_path}"
  puts "max_seq=#{max_seq}"
  puts "prompt_tokens=#{tokens}"
  puts "iters=#{iters}"
  puts "warmups=#{warmups}"
  puts "int8_block=#{int8_block}"
  puts "skip_hash=#{skip_hash}"
  puts "raw_bytes=#{raw_info.byte_size}"
  puts "bf16_bytes=#{bf16_info.byte_size}"
  puts "int8_bytes=#{i8_info.byte_size}"
  puts "bf16_ratio=#{(bf16_info.byte_size.to_f / raw_info.byte_size).round(4)}"
  puts "int8_ratio=#{(i8_info.byte_size.to_f / raw_info.byte_size).round(4)}"
  puts "source_top=#{source_top}"
  puts "raw_top=#{raw_top}"
  puts "bf16_top=#{bf16_top}"
  puts "int8_top=#{i8_top}"
  puts "raw_logit_delta=#{(raw_logit - source_logit).abs.round(6)}"
  puts "bf16_logit_delta=#{(bf16_logit - source_logit).abs.round(6)}"
  puts "int8_logit_delta=#{(i8_logit - source_logit).abs.round(6)}"
  puts "raw_parity=#{raw_top == source_top}"
  puts "bf16_parity=#{bf16_top == source_top}"
  puts "int8_parity=#{i8_top == source_top}"

  reusable_raw = prepared_state(hp, max_seq)
  reusable_bf16 = prepared_state(hp, max_seq)
  reusable_i8 = prepared_state(hp, max_seq)

  bench("raw_restore_only", warmups, iters) do
    ML::GGUF::Qwen35StateSnapshot.restore_into(raw_snapshot, hp, reusable_raw, prefer_metal: true)
  end
  bench("bf16_restore_only", warmups, iters) do
    ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(bf16_encoded, hp, reusable_bf16, prefer_metal: true)
  end
  bench("int8_restore_only", warmups, iters) do
    ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(i8_encoded, hp, reusable_i8, prefer_metal: true)
  end

  raw_expected_sha = skip_hash ? nil : raw_info.sha256
  bf16_expected_sha = skip_hash ? nil : bf16_info.sha256
  i8_expected_sha = skip_hash ? nil : i8_info.sha256
  bench("raw_read_restore", warmups, iters) do
    loaded = ML::GGUF::Qwen35StateSnapshot.read_artifact(raw_path, expected_sha256: raw_expected_sha)
    ML::GGUF::Qwen35StateSnapshot.restore_into(loaded, hp, reusable_raw, prefer_metal: true)
  end
  bench("bf16_read_restore", warmups, iters) do
    loaded = ML::GGUF::Qwen35StateSnapshot.read_artifact_encoded(bf16_path, expected_sha256: bf16_expected_sha, expected_codec: "recurrent-bf16")
    ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(loaded, hp, reusable_bf16, prefer_metal: true)
  end
  bench("int8_read_restore", warmups, iters) do
    loaded = ML::GGUF::Qwen35StateSnapshot.read_artifact_encoded(i8_path, expected_sha256: i8_expected_sha, expected_codec: "recurrent-int8", expected_codec_block: int8_block)
    ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(loaded, hp, reusable_i8, prefer_metal: true)
  end

  bench("bf16_mmap_restore", warmups, iters) do
    mapped = ML::GGUF::Qwen35StateSnapshot.read_artifact_encoded_mmap(bf16_path, expected_sha256: bf16_expected_sha, expected_codec: "recurrent-bf16")
    begin
      ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(mapped.encoded, hp, reusable_bf16, prefer_metal: true)
    ensure
      mapped.close
    end
  end
  bench("int8_mmap_restore", warmups, iters) do
    mapped = ML::GGUF::Qwen35StateSnapshot.read_artifact_encoded_mmap(i8_path, expected_sha256: i8_expected_sha, expected_codec: "recurrent-int8", expected_codec_block: int8_block)
    begin
      ML::GGUF::Qwen35StateSnapshot.restore_encoded_into(mapped.encoded, hp, reusable_i8, prefer_metal: true)
    ensure
      mapped.close
    end
  end
ensure
  FileUtils.rm_rf(root) if root && Dir.exists?(root)
end
