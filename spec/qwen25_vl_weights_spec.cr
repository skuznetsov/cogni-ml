require "spec"
require "../src/ml/gguf/qwen25_vl_weights"

private def tiny_qwen25_weights_config : ML::GGUF::Qwen25VLConfig
  ML::GGUF::Qwen25VLConfig.new(
    dim: 4,
    ffn_dim: 8,
    heads: 2,
    kv_heads: 1,
    head_dim: 2,
    layers: 3,
    max_positions: 32,
  )
end

private def qwen25_weights_config_json(
  config : ML::GGUF::Qwen25VLConfig = tiny_qwen25_weights_config,
  vocab_size : Int32 = 5,
  model_type : String = "qwen2_5_vl",
  architecture : String = "Qwen2_5_VLForConditionalGeneration",
  dtype : String = "bfloat16",
  hidden_size : Int32 = config.dim,
) : String
  JSON.build do |json|
    json.object do
      json.field "architectures" do
        json.array { json.string architecture }
      end
      json.field "hidden_act", "silu"
      json.field "hidden_size", hidden_size
      json.field "intermediate_size", config.ffn_dim
      json.field "max_position_embeddings", config.max_positions
      json.field "model_type", model_type
      json.field "num_attention_heads", config.heads
      json.field "num_hidden_layers", config.layers
      json.field "num_key_value_heads", config.kv_heads
      json.field "rms_norm_eps", config.rms_norm_eps
      json.field "rope_theta", config.rope_theta
      json.field "tie_word_embeddings", false
      json.field "torch_dtype", dtype
      json.field "vocab_size", vocab_size
    end
  end
end

private def with_qwen25_weights_dir(&block : String ->) : Nil
  dir = File.join(Dir.tempdir, "qwen25-vl-weights-#{Random.rand(1_000_000_000)}")
  Dir.mkdir(dir)
  begin
    yield dir
  ensure
    Dir.glob(File.join(dir, "*")).each { |path| File.delete?(path) }
    Dir.delete(dir)
  end
end

private def write_qwen25_weights_index(dir : String, weight_map : Hash(String, String)) : Nil
  index = JSON.build do |json|
    json.object do
      json.field "weight_map" do
        json.object do
          weight_map.each { |name, shard| json.field name, shard }
        end
      end
    end
  end
  File.write(File.join(dir, "model.safetensors.index.json"), index)
end

private def write_qwen25_weights_fixture(
  dir : String,
  *,
  config : ML::GGUF::Qwen25VLConfig = tiny_qwen25_weights_config,
  vocab_size : Int32 = 5,
  config_json : String? = nil,
  skip_tensor : String? = nil,
  malformed_shape : String? = nil,
  wrong_dtype : String? = nil,
  overlap : Bool = false,
  out_of_bounds : Bool = false,
  shard_name : String = "model.safetensors",
) : Nil
  File.write(File.join(dir, "config.json"), config_json || qwen25_weights_config_json(config, vocab_size))
  shapes = ML::GGUF::Qwen25VLWeights.selected_tensor_shapes(config, vocab_size)
  weight_map = {} of String => String
  offset = 0_i64
  records = [] of NamedTuple(name: String, shape: Array(Int64), start: Int64, end: Int64, dtype: String)

  shapes.each do |name, shape|
    next if name == skip_tensor

    stored_shape = if name == malformed_shape
                     shape.dup.tap { |dims| dims[0] -= 1 }
                   else
                     shape
                   end
    bytes = stored_shape.reduce(1_i64) { |count, dimension| count * dimension } * 2_i64
    start = if overlap && records.size == 1
              records[0][:start]
            else
              offset
            end
    finish = start + bytes
    dtype = name == wrong_dtype ? "F16" : "BF16"
    records << {name: name, shape: stored_shape, start: start, end: finish, dtype: dtype}
    weight_map[name] = shard_name
    offset = Math.max(offset, finish)
  end

  header = JSON.build do |json|
    json.object do
      records.each do |record|
        json.field record[:name] do
          json.object do
            json.field "dtype", record[:dtype]
            json.field "shape", record[:shape]
            json.field "data_offsets", [record[:start], record[:end]]
          end
        end
      end
    end
  end
  File.open(File.join(dir, shard_name), "wb") do |io|
    io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    io << header
    payload = Bytes.new(offset.to_i, 0_u8)
    if embedding = records.find { |record| record[:name] == ML::GGUF::Qwen25VLWeights::EMBEDDING_NAME }
      # Distinct raw BF16 rows let the metadata-only fixture verify selected-row reads.
      payload[embedding[:start].to_i, 8].copy_from(Bytes[0x80_u8, 0x3f_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8])
      last_row = embedding[:start] + (vocab_size - 1).to_i64 * config.dim * 2_i64
      payload[last_row.to_i, 8].copy_from(Bytes[0_u8, 0x40_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8])
    end
    io.write(payload) unless out_of_bounds
  end
  write_qwen25_weights_index(dir, weight_map)
end

private def qwen25_selected_block_hashes : Hash(String, String)
  hashes = {} of String => String
  ML::GGUF::Qwen25VLWeights.selected_tensor_names.each do |name|
    hashes[name] = "a" * 64 if name.starts_with?("model.layers.")
  end
  hashes
end

describe ML::GGUF::Qwen25VLWeights do
  it "defines only the pinned embedding and selected 0...26 block inventory" do
    shapes = ML::GGUF::Qwen25VLWeights.selected_tensor_shapes
    names = ML::GGUF::Qwen25VLWeights.selected_tensor_names

    names.size.should eq(313)
    shapes.size.should eq(313)
    shapes["model.embed_tokens.weight"].should eq([152064_i64, 3584_i64])
    shapes["model.layers.0.self_attn.q_proj.bias"].should eq([3584_i64])
    shapes["model.layers.25.mlp.down_proj.weight"].should eq([3584_i64, 18944_i64])
    shapes.has_key?("model.layers.26.self_attn.q_proj.weight").should be_false
    shapes.has_key?("model.layers.27.self_attn.q_proj.weight").should be_false
    shapes.has_key?("model.norm.weight").should be_false
    shapes.has_key?("lm_head.weight").should be_false
    shapes.keys.any?(&.includes?("visual")).should be_false

    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    ML::GGUF::Qwen25VLWeights.resident_prefix_bytes(config).should eq(12_119_005_184_i64)
    ML::GGUF::Qwen25VLWeights.selected_bytes(config).should eq(13_208_999_936_i64)
  end

  it "validates a tiny indexed BF16 checkpoint and reads only selected embedding rows" do
    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir)
      weights = ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5)

      weights.selected_tensor_names.size.should eq(13)
      weights.embedding_table_bytes.should eq(40_i64)
      weights.resident_prefix_bytes.should eq(320_i64)
      weights.selected_bytes.should eq(360_i64)
      weights.block_bytes(0).should eq(320_i64)
      weights.embedding_rows_raw([0_i32, 4_i32]).should eq(Bytes[
        0x80_u8, 0x3f_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8,
        0_u8, 0x40_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8, 0_u8,
      ])
      {% if flag?(:cpu_only) %}
        # Compile and enter the upload path without creating a Metal device or buffer.
        expect_raises(Exception, /Metal disabled/) { weights.upload_block_weights(0) }
      {% end %}
      expect_raises(ArgumentError, /outside vocabulary/) { weights.embedding_rows_raw([5_i32]) }
      weights.close
      expect_raises(ArgumentError, /closed/) { weights.embedding_rows_raw([0_i32]) }
    end
  end

  it "validates requested block hashes before allocating and accepts a full selected-block map" do
    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir)
      weights = ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5)
      live_bytes = ML::MetalBuffer.stats[:live_bytes]

      missing = qwen25_selected_block_hashes
      missing.delete("model.layers.0.mlp.down_proj.weight")
      expect_raises(ArgumentError, /SHA-256 missing/) do
        weights.upload_block_weights(0, missing)
      end
      ML::MetalBuffer.stats[:live_bytes].should eq(live_bytes)

      malformed = qwen25_selected_block_hashes
      malformed["model.layers.0.self_attn.q_proj.weight"] = "abc"
      expect_raises(ArgumentError, /malformed SHA-256/) do
        weights.upload_block_weights(0, malformed)
      end
      ML::MetalBuffer.stats[:live_bytes].should eq(live_bytes)

      {% if flag?(:cpu_only) %}
        # The caller may pass hashes for every selected layer. Validation
        # extracts layer 0's twelve hashes and proceeds to the Metal boundary.
        expect_raises(Exception, /Metal disabled/) do
          weights.upload_block_weights(0, qwen25_selected_block_hashes)
        end
      {% end %}
      weights.close
    end
  end

  it "hashes the exact bounded chunks copied and rejects same-size source mutation" do
    original = Bytes[0x80_u8, 0x3f_u8, 0x00_u8, 0x01_u8, 0xff_u8, 0x7f_u8, 0x44_u8]
    expected = Digest::SHA256.hexdigest(original)
    destination = Bytes.new(original.size, 0_u8)
    actual = ML::GGUF::Qwen25VLWeights.copy_and_verify_sha256(
      IO::Memory.new(original), destination.to_unsafe, original.size.to_i64,
      expected, "fixture.tensor", 3,
    )
    actual.should eq(expected)
    destination.should eq(original)

    modified = original.dup
    modified[3] = 0x02_u8
    modified_destination = Bytes.new(modified.size, 0_u8)
    expect_raises(ArgumentError, /SHA-256 mismatch/) do
      ML::GGUF::Qwen25VLWeights.copy_and_verify_sha256(
        IO::Memory.new(modified), modified_destination.to_unsafe, modified.size.to_i64,
        expected, "fixture.tensor", 3,
      )
    end
    # It hashes the bytes that were streamed into the destination, not a
    # separately-read or metadata-derived copy.
    modified_destination.should eq(modified)

    untouched = Bytes.new(original.size, 0xaa_u8)
    expect_raises(ArgumentError, /malformed SHA-256/) do
      ML::GGUF::Qwen25VLWeights.copy_and_verify_sha256(
        IO::Memory.new(original), untouched.to_unsafe, original.size.to_i64,
        "bad", "fixture.tensor", 3,
      )
    end
    untouched.should eq(Bytes.new(original.size, 0xaa_u8))
  end

  {% unless flag?(:cpu_only) %}
    it "releases current and earlier Metal buffers when uploaded bytes fail their digest" do
      unless ENV["COGNI_QWEN25_WEIGHTS_GPU"]? == "1"
        pending! "set COGNI_QWEN25_WEIGHTS_GPU=1 under the memory guard"
        next
      end
      with_qwen25_weights_dir do |dir|
        write_qwen25_weights_fixture(dir)
        weights = ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5)
        hashes = {} of String => String
        weights.selected_tensor_shapes.each do |name, shape|
          next if name == ML::GGUF::Qwen25VLWeights::EMBEDDING_NAME
          bytes = shape.reduce(1_i64) { |count, dim| count * dim } * 2
          hashes[name] = Digest::SHA256.hexdigest(Bytes.new(bytes.to_i, 0_u8))
        end
        before = ML::MetalBuffer.stats
        uploaded = {} of String => ML::MetalBuffer
        begin
          [hashes.keys.first, hashes.keys.last].each do |name|
            incorrect = hashes.dup
            incorrect[name] = "f" * 64
            expect_raises(ArgumentError, /SHA-256 mismatch/) do
              weights.upload_block_weights(0, incorrect)
            end
            ML::MetalBuffer.stats[:live_bytes].should eq(before[:live_bytes])
            ML::MetalBuffer.stats[:live_buffers].should eq(before[:live_buffers])
          end
          uploaded = weights.upload_block_weights(0, hashes)
          uploaded.size.should eq(12)
          (ML::MetalBuffer.stats[:live_bytes] - before[:live_bytes]).should eq(320_i64)
        ensure
          uploaded.each_value(&.release)
          weights.close
        end
        ML::MetalBuffer.stats[:live_bytes].should eq(before[:live_bytes])
        ML::MetalBuffer.stats[:live_buffers].should eq(before[:live_buffers])
      end
    end
  {% end %}

  it "rejects mismatched config identity and geometry" do
    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, config_json: qwen25_weights_config_json(model_type: "qwen2_5_vl_text"))
      expect_raises(ArgumentError, /model_type/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, config_json: qwen25_weights_config_json(hidden_size: 6))
      expect_raises(ArgumentError, /hidden_size/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, config_json: qwen25_weights_config_json(architecture: "Qwen2ForCausalLM"))
      expect_raises(ArgumentError, /architectures/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, config_json: qwen25_weights_config_json(dtype: "float16"))
      expect_raises(ArgumentError, /torch_dtype/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end
  end

  it "rejects missing, wrong-shaped, and non-BF16 selected tensors" do
    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, skip_tensor: "model.layers.0.mlp.down_proj.weight")
      expect_raises(ArgumentError, /missing required tensors/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, malformed_shape: "model.layers.0.self_attn.q_proj.weight")
      expect_raises(ArgumentError, /shape/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, wrong_dtype: "model.layers.0.self_attn.q_proj.weight")
      expect_raises(ArgumentError, /BF16/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end
  end

  it "rejects path traversal, overlapping tensor ranges, and shard truncation" do
    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir)
      index = JSON.parse(File.read(File.join(dir, "model.safetensors.index.json"))).as_h
      weight_map = index["weight_map"].as_h
      weight_map[ML::GGUF::Qwen25VLWeights::EMBEDDING_NAME] = JSON::Any.new("../outside.safetensors")
      File.write(File.join(dir, "model.safetensors.index.json"), JSON.build { |json| json.object { json.field "weight_map", weight_map } })
      expect_raises(ArgumentError, /invalid shard path/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir)
      outside_path = "#{dir}-outside.safetensors"
      link_path = File.join(dir, "linked.safetensors")
      begin
        File.write(outside_path, "outside")
        File.symlink(outside_path, link_path)
        index = JSON.parse(File.read(File.join(dir, "model.safetensors.index.json"))).as_h
        weight_map = index["weight_map"].as_h
        weight_map.each_key { |name| weight_map[name] = JSON::Any.new("linked.safetensors") }
        File.write(File.join(dir, "model.safetensors.index.json"), JSON.build { |json| json.object { json.field "weight_map", weight_map } })
        expect_raises(ArgumentError, /escapes model directory/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
      ensure
        File.delete?(outside_path)
      end
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, overlap: true)
      expect_raises(ArgumentError, /overlap/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir, out_of_bounds: true)
      expect_raises(ArgumentError, /outside shard bounds/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir)
      other_shard = File.join(dir, "other.safetensors")
      empty_header = "{}"
      File.open(other_shard, "wb") do |io|
        io.write_bytes(empty_header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
        io.write(empty_header.to_slice)
      end
      index = JSON.parse(File.read(File.join(dir, "model.safetensors.index.json"))).as_h
      weight_map = index["weight_map"].as_h
      weight_map["model.layers.0.self_attn.q_proj.weight"] = JSON::Any.new("other.safetensors")
      File.write(File.join(dir, "model.safetensors.index.json"), JSON.build { |json| json.object { json.field "weight_map", weight_map } })
      expect_raises(ArgumentError, /index shard mapping disagrees/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end

    with_qwen25_weights_dir do |dir|
      write_qwen25_weights_fixture(dir)
      File.open(File.join(dir, "model.safetensors"), "r+") do |io|
        io.write_bytes(UInt64::MAX, IO::ByteFormat::LittleEndian)
      end
      expect_raises(ArgumentError, /invalid header bounds/) { ML::GGUF::Qwen25VLWeights.from_directory(dir, tiny_qwen25_weights_config, 5) }
    end
  end
end
