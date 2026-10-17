#!/usr/bin/env crystal
# Setup-time GGUF export and one-block CUDA checkpoint probe. enqueue performs
# no host readback or synchronization; checkpoints are copied only after the
# explicit terminal boundary below.

require "json"
require "digest/sha256"
require "file_utils"
require "../src/ml/cuda/qwen_image21_block_runner"
require "../src/ml/gguf/qwen_image21_gguf"

FIXTURE_SCHEMA   = "qwen-image21-cuda-block-fixture"
CANDIDATE_SCHEMA = "qwen-image21-cuda-block-candidate"
PINNED_REPO      = "realrebelai/Qwen-Image-2.1_GGUFs"
PINNED_REVISION  = "8d393b750593a72ee040fe7d40611f479ccee679"
PINNED_FILENAME  = "Qwen-Image-2.1-Q4.gguf"
PINNED_SIZE      = 5_959_127_264_i64
PINNED_SHA256    = "51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a"
WEIGHT_SOURCES   = {
  "to_q"    => "transformer_blocks.0.attn.to_q.weight",
  "to_k"    => "transformer_blocks.0.attn.to_k.weight",
  "to_v"    => "transformer_blocks.0.attn.to_v.weight",
  "to_out"  => "transformer_blocks.0.attn.to_out.0.weight",
  "norm_q"  => "transformer_blocks.0.attn.norm_q.weight",
  "norm_k"  => "transformer_blocks.0.attn.norm_k.weight",
  "gate_up" => "transformer_blocks.0.img_mlp.gate_up.weight",
  "mlp_out" => "transformer_blocks.0.img_mlp.out.weight",
}
CHECKPOINTS     = ML::CUDA::QwenImage21BlockRunner::CHECKPOINT_NAMES
SUPPORTED_TYPES = {
  ML::GGUF::TensorType::F32,
  ML::GGUF::TensorType::BF16,
  ML::GGUF::TensorType::Q8_0,
  ML::GGUF::TensorType::Q6_K,
  ML::GGUF::TensorType::Q5_K,
}

private def fail(message : String) : NoReturn
  raise ArgumentError.new(message)
end

private def json_field(value : JSON::Any, name : String) : JSON::Any
  value.as_h[name]? || fail("missing JSON field #{name}")
end

private def json_string(value : JSON::Any, name : String) : String
  json_field(value, name).as_s
end

private def json_int(value : JSON::Any, name : String) : Int32
  json_field(value, name).as_i.to_i32
end

private def json_int64(value : JSON::Any, name : String) : Int64
  json_field(value, name).as_i64
end

private def json_f32(value : JSON::Any, name : String) : Float32
  json_field(value, name).as_f.to_f32
end

private def safe_file(root : Path, relative : String) : Path
  rel = Path.new(relative)
  fail("absolute fixture path is forbidden: #{relative}") if rel.absolute?
  base = Path[File.expand_path(root.to_s)]
  resolved = Path[File.expand_path(rel.to_s, base.to_s)]
  unless resolved == base || resolved.to_s.starts_with?(base.to_s + "/")
    fail("fixture path escapes its directory: #{relative}")
  end
  resolved
end

private def read_manifest(root : Path) : JSON::Any
  path = root / "manifest.json"
  fail("missing fixture manifest #{path}") unless File.file?(path)
  manifest = JSON.parse(File.read(path))
  fail("fixture schema/version mismatch") unless json_string(manifest, "schema") == FIXTURE_SCHEMA && json_int(manifest, "schema_version") == 1
  fixture = json_field(manifest, "fixture")
  fail("fixture must use six-token mini-Quad layout") unless json_int(fixture, "tokens") == 6
  config = json_field(fixture, "config")
  fail("fixture prefix must contain two text tokens") unless json_int(config, "prefix_length") == 2
  manifest
end

private def sha256_file(path : Path) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1024 * 1024)
  File.open(path, "rb") do |file|
    loop do
      count = file.read(buffer)
      break if count == 0
      digest.update(buffer[0, count])
    end
  end
  digest.final.hexstring
end

private def read_verified(root : Path, descriptor : JSON::Any, label : String) : String
  path = safe_file(root, json_string(descriptor, "file"))
  fail("missing #{label}: #{path}") unless File.file?(path)
  bytes = File.read(path)
  fail("#{label} byte count mismatch") unless bytes.bytesize.to_i64 == json_int64(descriptor, "nbytes")
  fail("#{label} SHA-256 mismatch") unless Digest::SHA256.hexdigest(bytes) == json_string(descriptor, "sha256")
  bytes
end

private def f32_bytes(values : Array(Float32)) : Bytes
  Slice.new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32))
end

private def f32_values(bytes : String, label : String) : Array(Float32)
  fail("#{label} is not a little-endian F32 tensor") unless bytes.bytesize % sizeof(Float32) == 0
  values = Array(Float32).new(bytes.bytesize // sizeof(Float32)) do |index|
    bytes.to_unsafe.as(Float32*)[index]
  end
  fail("#{label} contains non-finite values") unless values.all?(&.finite?)
  values
end

private def i32_values(bytes : String, label : String) : Array(Int32)
  fail("#{label} is not a little-endian I32 tensor") unless bytes.bytesize % sizeof(Int32) == 0
  Array(Int32).new(bytes.bytesize // sizeof(Int32)) { |index| bytes.to_unsafe.as(Int32*)[index] }
end

private def json_shape(shape : Array(Int32)) : JSON::Any
  JSON::Any.new(shape.map { |size| JSON::Any.new(size) })
end

private def write_tensor(root : Path, relative : String, dtype : String, shape : Array(Int32), bytes : Bytes) : JSON::Any
  path = safe_file(root, relative)
  FileUtils.mkdir_p(path.parent)
  File.open(path, "wb") { |file| file.write(bytes) }
  JSON::Any.new({
    "file"   => JSON::Any.new(relative),
    "dtype"  => JSON::Any.new(dtype),
    "shape"  => json_shape(shape),
    "nbytes" => JSON::Any.new(bytes.size.to_i64),
    "sha256" => JSON::Any.new(Digest::SHA256.hexdigest(bytes)),
  })
end

private def fixture_config(fixture : JSON::Any) : ML::GGUF::QwenImage21BlockConfig
  data = json_field(fixture, "config")
  axes = json_field(data, "axes_dims").as_a
  fail("RoPE axes must contain exactly three dimensions") unless axes.size == 3
  axes_dims = StaticArray[axes[0].as_i.to_i32, axes[1].as_i.to_i32, axes[2].as_i.to_i32]
  ML::GGUF::QwenImage21BlockConfig.new(
    json_int(data, "hidden_dim"),
    json_int(data, "heads"),
    json_int(data, "head_dim"),
    json_int(data, "intermediate_dim"),
    axes_dims,
    json_f32(data, "eps"),
    json_f32(data, "rope_theta"),
  )
end

private def expected_weight_shape(name : String, config : ML::GGUF::QwenImage21BlockConfig) : Array(Int32)
  case name
  when "to_q", "to_k", "to_v", "to_out" then [config.hidden_dim, config.hidden_dim]
  when "norm_q", "norm_k"               then [config.head_dim]
  when "gate_up"                        then [2 * config.intermediate_dim, config.hidden_dim]
  when "mlp_out"                        then [config.hidden_dim, config.intermediate_dim]
  else                                       fail("unexpected weight #{name}")
  end
end

private def tensor_type(name : String) : ML::GGUF::TensorType
  case name
  when "F32"  then ML::GGUF::TensorType::F32
  when "BF16" then ML::GGUF::TensorType::BF16
  when "Q8_0" then ML::GGUF::TensorType::Q8_0
  when "Q6_K" then ML::GGUF::TensorType::Q6_K
  when "Q5_K" then ML::GGUF::TensorType::Q5_K
  else             fail("unsupported fixture weight type #{name}")
  end
end

private def write_manifest(path : Path, value : JSON::Any) : Nil
  File.write(path, value.to_json + "\n")
end

private def export_real_weights(gguf_path : Path, fixture_root : Path, manifest : JSON::Any) : Nil
  model = json_field(manifest, "model")
  fail("unexpected GGUF repository pin") unless json_string(model, "repo") == PINNED_REPO
  fail("unexpected GGUF revision pin") unless json_string(model, "revision") == PINNED_REVISION
  fail("unexpected GGUF filename pin") unless json_string(model, "filename") == PINNED_FILENAME
  fail("unexpected GGUF size pin") unless json_int64(model, "size") == PINNED_SIZE
  fail("unexpected GGUF SHA pin") unless json_string(model, "sha256") == PINNED_SHA256
  fail("fixture must target block zero") unless json_int(model, "block_index") == 0
  fail("GGUF path must name the pinned file") unless gguf_path.basename == PINNED_FILENAME
  fail("GGUF file size mismatch") unless File.size(gguf_path).to_i64 == PINNED_SIZE
  actual_sha = sha256_file(gguf_path)
  fail("GGUF SHA-256 mismatch: got #{actual_sha}") unless actual_sha == PINNED_SHA256

  gguf = ML::GGUF::GGUFFile.new(gguf_path.to_s)
  begin
    inventory = ML::GGUF::QwenImage21GGUFInventory.from_file(gguf)
    config = fixture_config(json_field(manifest, "fixture"))
    unless config.hidden_dim == 4096 && config.intermediate_dim == 12288 && config.heads == 32 && config.head_dim == 128
      fail("real fixture dimensions do not match Qwen-Image 2.1 block 0")
    end

    entries = Hash(String, JSON::Any).new
    WEIGHT_SOURCES.each do |name, tensor_name|
      info = gguf.tensor(tensor_name) || fail("missing GGUF tensor #{tensor_name}")
      fail("unsupported block weight type #{name}=#{info.type.name}") unless SUPPORTED_TYPES.includes?(info.type)
      expected = expected_weight_shape(name, config)
      shape = if name == "norm_q" || name == "norm_k"
                info.dims.map(&.to_i32)
              else
                out_dim, in_dim = inventory.projection_dims(tensor_name)
                [out_dim, in_dim]
              end
      fail("#{tensor_name} shape mismatch: #{shape} != #{expected}") unless shape == expected

      raw = gguf.read_tensor_raw(info)
      packed_file = "weights/#{name}.packed"
      packed_path = safe_file(fixture_root, packed_file)
      FileUtils.mkdir_p(packed_path.parent)
      File.open(packed_path, "wb") { |file| file.write(raw) }
      dequantized = gguf.read_tensor_f32(info)
      fail("#{tensor_name} dequantized count mismatch") unless dequantized.size == expected.product
      f32_desc = write_tensor(fixture_root, "weights/#{name}.f32", "float32-le", expected, f32_bytes(dequantized))
      entries[name] = JSON::Any.new({
        "shape"         => json_shape(expected),
        "quant_type"    => JSON::Any.new(info.type.name),
        "source_tensor" => JSON::Any.new(tensor_name),
        "packed_file"   => JSON::Any.new(packed_file),
        "packed_nbytes" => JSON::Any.new(raw.size.to_i64),
        "packed_sha256" => JSON::Any.new(Digest::SHA256.hexdigest(raw)),
        "dequantized"   => f32_desc,
      })
      puts({event: "weight_export", name: name, tensor: tensor_name, quant_type: info.type.name,
            shape: shape, packed_bytes: raw.size, f32_values: dequantized.size}.to_json)
    end
    manifest.as_h["weights"] = JSON::Any.new(entries)
    write_manifest(fixture_root / "manifest.json", manifest)
  ensure
    gguf.close
  end
end

private def load_weights(root : Path, manifest : JSON::Any, config : ML::GGUF::QwenImage21BlockConfig) : ML::GGUF::QwenImage21BlockWeights
  weight_manifest = json_field(manifest, "weights")
  projections = {} of String => ML::GGUF::QuantWeight
  vectors = {} of String => Array(Float32)
  keep_raw_alive = [] of String

  WEIGHT_SOURCES.each_key do |name|
    entry = json_field(weight_manifest, name)
    shape = json_field(entry, "shape").as_a.map(&.as_i.to_i32)
    expected = expected_weight_shape(name, config)
    fail("fixture weight #{name} shape mismatch") unless shape == expected
    quant_name = json_string(entry, "quant_type")
    type = tensor_type(quant_name)

    packed_file = json_string(entry, "packed_file")
    packed_path = safe_file(root, packed_file)
    fail("missing packed weight #{name}") unless File.file?(packed_path)
    packed = File.read(packed_path)
    fail("packed weight #{name} byte count mismatch") unless packed.bytesize.to_i64 == json_int64(entry, "packed_nbytes")
    fail("packed weight #{name} SHA-256 mismatch") unless Digest::SHA256.hexdigest(packed) == json_string(entry, "packed_sha256")
    keep_raw_alive << packed
    dequant_desc = json_field(entry, "dequantized")
    dequantized = f32_values(read_verified(root, dequant_desc, "weight #{name}"), "weight #{name}")
    fail("weight #{name} dequantized count mismatch") unless dequantized.size == expected.product

    if name == "norm_q" || name == "norm_k"
      fail("norm weight #{name} must be a vector") unless type.f32? || type.bf16? || type.q8_0? || type.q5_k? || type.q6_k?
      vectors[name] = dequantized
    else
      projections[name] = ML::GGUF::QuantWeight.new(packed.to_slice, type, shape[0], shape[1])
    end
  end

  ML::GGUF::QwenImage21BlockWeights.new(
    projections["to_q"], projections["to_k"], projections["to_v"], projections["to_out"],
    vectors["norm_q"], vectors["norm_k"], projections["gate_up"], projections["mlp_out"],
  )
end

private def run_fixture(fixture_root : Path, candidate_root : Path) : Nil
  manifest = read_manifest(fixture_root)
  fixture = json_field(manifest, "fixture")
  config = fixture_config(fixture)
  tokens = json_int(fixture, "tokens")
  inputs = json_field(manifest, "inputs")

  hidden = f32_values(read_verified(fixture_root, json_field(inputs, "hidden"), "input hidden"), "input hidden")
  modulation = f32_values(read_verified(fixture_root, json_field(inputs, "modulation"), "input modulation"), "input modulation")
  position_values = i32_values(read_verified(fixture_root, json_field(inputs, "positions"), "input positions"), "input positions")
  image_ids = i32_values(read_verified(fixture_root, json_field(inputs, "image_ids"), "input image ids"), "input image ids")
  key_valid_bytes = read_verified(fixture_root, json_field(inputs, "key_valid"), "input key validity")
  fail("position shape mismatch") unless position_values.size == tokens * 3
  fail("image_ids shape mismatch") unless image_ids.size == tokens
  fail("key_valid shape mismatch") unless key_valid_bytes.bytesize == tokens
  fail("hidden shape mismatch") unless hidden.size == tokens * config.hidden_dim
  fail("modulation shape mismatch") unless modulation.size == tokens * 4 * config.hidden_dim
  positions = Array(StaticArray(Int32, 3)).new(tokens) do |token|
    StaticArray[position_values[3 * token], position_values[3 * token + 1], position_values[3 * token + 2]]
  end
  key_valid = Array(Bool).new(tokens) do |token|
    value = key_valid_bytes.to_slice[token]
    fail("key_valid must contain only 0 or 1") unless value <= 1
    value == 1
  end
  weights = load_weights(fixture_root, manifest, config)
  fixture_manifest_bytes = File.read(fixture_root / "manifest.json")

  context = ML::CUDA::Context.create
  runner_a = nil.as(ML::CUDA::QwenImage21BlockRunner?)
  runner_b = nil.as(ML::CUDA::QwenImage21BlockRunner?)
  begin
    runner_a = ML::CUDA::QwenImage21BlockRunner.new(weights, config, tokens, modulation, positions, image_ids, key_valid)
    runner_b = ML::CUDA::QwenImage21BlockRunner.new(weights, config, tokens, modulation, positions, image_ids, key_valid)
    runner_a.upload_hidden(hidden)
    runner_a.enqueue
    runner_b.use_device_input(runner_a.output_device_ptr)
    runner_b.enqueue

    # Covers runner A, runner B's device-only input chain, and A's checkpoint
    # batch with one terminal synchronization. No activation is read to host
    # between the two runner enqueues.
    ML::CUDA.synchronize!("Qwen-Image CUDA block terminal checkpoint batch")
    first_checkpoints = {} of String => Array(Float32)
    shapes = runner_a.checkpoint_shapes
    runner_a.checkpoint_names.each do |name|
      values = runner_a.read_checkpoint(name)
      fail("CUDA checkpoint #{name} contains a non-finite value") unless values.all?(&.finite?)
      first_checkpoints[name] = values
    end
    chained = runner_b.read_checkpoint("output")
    fail("device-chain output contains a non-finite value") unless chained.all?(&.finite?)
    first_output = first_checkpoints["output"]

    # Reuse runner A with the original hidden bytes. Bitwise output equality is
    # the stale-state guard for this same-input/same-weights fixture.
    runner_a.upload_hidden(hidden)
    runner_a.enqueue
    ML::CUDA.synchronize!("Qwen-Image CUDA repeated-input state guard")
    repeated = runner_a.read_checkpoint("output")
    repeat_exact = repeated.size == first_output.size && repeated.each_with_index.all? do |value, index|
      value.unsafe_as(UInt32) == first_output[index].unsafe_as(UInt32)
    end
    fail("repeated hidden input changed the block output") unless repeat_exact

    if File.exists?(candidate_root)
      fail("candidate directory must be empty: #{candidate_root}") if Dir.children(candidate_root).any?
    else
      FileUtils.mkdir_p(candidate_root)
    end
    checkpoint_entries = Hash(String, JSON::Any).new
    first_checkpoints.each do |name, values|
      rows, width = shapes[name]
      checkpoint_entries[name] = write_tensor(candidate_root, "checkpoints/#{name}.f32", "float32-le",
        [rows, width], f32_bytes(values))
    end
    chain_desc = write_tensor(candidate_root, "guards/device_chain_output.f32", "float32-le",
      [tokens, config.hidden_dim], f32_bytes(chained))
    guards = JSON::Any.new({
      "device_chain"                      => JSON::Any.new("runner_b_input=runner_a.output_device_ptr"),
      "repeat_input_output_bitwise_equal" => JSON::Any.new(repeat_exact),
      "device_name"                       => JSON::Any.new(context.device_name),
      "chain_output"                      => chain_desc,
    })
    candidate = JSON::Any.new({
      "schema"                  => JSON::Any.new(CANDIDATE_SCHEMA),
      "schema_version"          => JSON::Any.new(1),
      "fixture_manifest_sha256" => JSON::Any.new(Digest::SHA256.hexdigest(fixture_manifest_bytes)),
      "checkpoints"             => JSON::Any.new(checkpoint_entries),
      "guards"                  => guards,
    })
    File.write(candidate_root / "candidate.json", candidate.to_json + "\n")
    puts({event: "cuda_block_probe", fixture_kind: json_string(fixture, "kind"), tokens: tokens,
          checkpoint_count: first_checkpoints.size, device_chain: true, repeated_input_bitwise_equal: repeat_exact,
          terminal_syncs: 2, device: context.device_name}.to_json)
  ensure
    runner_b.try(&.close)
    runner_a.try(&.close)
    context.close
  end
end

private def main : Int32
  command = ARGV.shift? || ""
  case command
  when "--export-gguf"
    fail("usage: --export-gguf GGUF_PATH FIXTURE_DIR") unless ARGV.size == 2
    gguf_path = Path.new(ARGV[0])
    fixture_root = Path.new(ARGV[1])
    manifest = read_manifest(fixture_root)
    fail("GGUF export only supports the real block-0 fixture") unless json_string(json_field(manifest, "fixture"), "kind") == "real_block0_mini_quad"
    export_real_weights(gguf_path, fixture_root, manifest)
    puts({event: "gguf_export_complete", fixture: fixture_root.to_s, sha256: PINNED_SHA256}.to_json)
  when "--run"
    fail("usage: --run FIXTURE_DIR CANDIDATE_DIR") unless ARGV.size == 2
    run_fixture(Path.new(ARGV[0]), Path.new(ARGV[1]))
  when "--help", "-h"
    puts "Usage: crystal run bin/qwen_image21_cuda_block_probe.cr -- --export-gguf GGUF_PATH FIXTURE_DIR"
    puts "       crystal run bin/qwen_image21_cuda_block_probe.cr -- --run FIXTURE_DIR CANDIDATE_DIR"
  else
    fail("choose --export-gguf or --run (see --help)")
  end
  0
end

exit(main)
