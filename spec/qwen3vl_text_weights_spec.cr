require "./spec_helper"
require "digest/sha256"
require "../src/ml/gguf/qwen3vl_text_weights"

private def qwen3vl_weights_config_json(
  *, include_text_config : Bool = true, model_type : String = "qwen3_vl",
  rope_theta : Float64? = nil, hidden_act : String = "silu", attention_bias : Bool = false,
) : String
  JSON.build do |json|
    json.object do
      json.field "model_type", model_type
      json.field "architectures" do
        json.array { json.string "Qwen3VLForConditionalGeneration" }
      end
      if include_text_config
        json.field "text_config" do
          json.object do
            json.field "model_type", "qwen3_vl_text"
            json.field "dtype", "bfloat16"
            json.field "hidden_size", 4096
            json.field "intermediate_size", 12288
            json.field "num_hidden_layers", 36
            json.field "num_attention_heads", 32
            json.field "num_key_value_heads", 8
            json.field "head_dim", 128
            json.field "vocab_size", 151936
            json.field "hidden_act", hidden_act
            json.field "attention_bias", attention_bias
            json.field "rms_norm_eps", 0.000001
            if rope_theta
              json.field "rope_theta", rope_theta
            end
            json.field "rope_scaling" do
              json.object do
                json.field "mrope_interleaved", true
                json.field "mrope_section" do
                  json.array { [24, 20, 20].each { |n| json.number n } }
                end
              end
            end
          end
        end
      end
    end
  end
end

private def write_qwen3vl_weights_metadata(
  dir : String,
  *,
  config : String = qwen3vl_weights_config_json,
  weight_map : Hash(String, String) = {} of String => String,
) : Nil
  File.write(File.join(dir, "config.json"), config)
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

private def with_qwen3vl_weights_dir(&block : String ->) : Nil
  dir = File.join(Dir.tempdir, "qwen3vl-text-weights-#{Random.rand(1_000_000_000)}")
  Dir.mkdir(dir)
  begin
    yield dir
  ensure
    Dir.glob(File.join(dir, "*")).each { |path| File.delete?(path) }
    Dir.delete(dir)
  end
end

private def write_qwen3vl_truncated_embedding_shard(path : String) : Nil
  tensor_bytes = 151936_i64 * 4096_i64 * 2_i64
  header = JSON.build do |json|
    json.object do
      json.field ML::GGUF::Qwen3VLTextWeights::EMBEDDING_NAME do
        json.object do
          json.field "dtype", "BF16"
          json.field "shape", [151936, 4096]
          json.field "data_offsets", [0, tensor_bytes]
        end
      end
    end
  end
  File.open(path, "wb") do |io|
    io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    io << header
  end
end

private def write_qwen3vl_embedding_shard(path : String, dtype : String, shape : Array(Int32), payload : Bytes) : Nil
  header = JSON.build do |json|
    json.object do
      json.field ML::GGUF::Qwen3VLTextWeights::EMBEDDING_NAME do
        json.object do
          json.field "dtype", dtype
          json.field "shape", shape
          json.field "data_offsets", [0, payload.size]
        end
      end
    end
  end
  File.open(path, "wb") do |io|
    io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    io << header
    io.write(payload)
  end
end

# A valid metadata layout backed by a sparse file: the full model's logical
# offsets are exercised without allocating or checking in its multi-GB weights.
private def write_qwen3vl_sparse_complete_shard(path : String) : {Bytes, Bytes}
  offset = 0_i64
  embedding_offset = -1_i64
  header = JSON.build do |json|
    json.object do
      ML::GGUF::Qwen3VLTextWeights.required_tensor_shapes.each do |name, shape|
        bytes = shape.reduce(1_i64) { |count, dimension| count * dimension } * 2_i64
        embedding_offset = offset if name == ML::GGUF::Qwen3VLTextWeights::EMBEDDING_NAME
        json.field name do
          json.object do
            json.field "dtype", "BF16"
            json.field "shape", shape
            json.field "data_offsets", [offset, offset + bytes]
          end
        end
        offset += bytes
      end
    end
  end
  raise "synthetic embedding tensor missing" if embedding_offset < 0

  first_row = Bytes.new(ML::GGUF::Qwen3VLTextWeights::ROW_BYTES, 0_u8)
  first_row[0] = 0x80_u8
  first_row[1] = 0x3f_u8 # BF16 1.0
  last_row = Bytes.new(ML::GGUF::Qwen3VLTextWeights::ROW_BYTES, 0_u8)
  last_row[1] = 0x40_u8 # BF16 2.0
  File.open(path, "wb") do |io|
    io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    io << header
    payload_start = 8_i64 + header.bytesize
    io.seek(payload_start + offset - 1)
    io.write_byte(0_u8)
    io.seek(payload_start + embedding_offset)
    io.write(first_row)
    last_token = ML::GGUF::Qwen3VLTextWeights::VOCAB_SIZE - 1
    io.seek(payload_start + embedding_offset + last_token.to_i64 * ML::GGUF::Qwen3VLTextWeights::ROW_BYTES)
    io.write(last_row)
  end
  {first_row, last_row}
end

describe ML::GGUF::Qwen3VLTextWeights do
  it "defines the pinned 398-tensor text inventory and projection shapes" do
    names = ML::GGUF::Qwen3VLTextWeights.required_tensor_names
    shapes = ML::GGUF::Qwen3VLTextWeights.required_tensor_shapes

    names.size.should eq(398)
    shapes.size.should eq(398)
    shapes["model.language_model.embed_tokens.weight"].should eq([151936_i64, 4096_i64])
    shapes["model.language_model.layers.0.self_attn.q_proj.weight"].should eq([4096_i64, 4096_i64])
    shapes["model.language_model.layers.0.self_attn.k_proj.weight"].should eq([1024_i64, 4096_i64])
    shapes["model.language_model.layers.0.mlp.gate_proj.weight"].should eq([12288_i64, 4096_i64])
    shapes["model.language_model.layers.35.mlp.down_proj.weight"].should eq([4096_i64, 12288_i64])
    shapes["model.language_model.norm.weight"].should eq([4096_i64])
  end

  it "rejects missing or mismatched Qwen3-VL config metadata before opening shards" do
    with_qwen3vl_weights_dir do |dir|
      write_qwen3vl_weights_metadata(dir, config: qwen3vl_weights_config_json(include_text_config: false))
      expect_raises(ArgumentError, /text_config/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end

    with_qwen3vl_weights_dir do |dir|
      write_qwen3vl_weights_metadata(dir, config: qwen3vl_weights_config_json(model_type: "qwen3_5"))
      expect_raises(ArgumentError, /model_type/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end

    with_qwen3vl_weights_dir do |dir|
      write_qwen3vl_weights_metadata(dir, config: qwen3vl_weights_config_json(rope_theta: 1_000_000_f64))
      expect_raises(ArgumentError, /rope_theta/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end

    with_qwen3vl_weights_dir do |dir|
      write_qwen3vl_weights_metadata(dir, config: qwen3vl_weights_config_json(rope_theta: 5_000_000_f64))
      expect_raises(ArgumentError, /missing required text tensors/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end

    with_qwen3vl_weights_dir do |dir|
      write_qwen3vl_weights_metadata(dir, config: qwen3vl_weights_config_json(hidden_act: "gelu"))
      expect_raises(ArgumentError, /hidden_act/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end

    with_qwen3vl_weights_dir do |dir|
      write_qwen3vl_weights_metadata(dir, config: qwen3vl_weights_config_json(attention_bias: true))
      expect_raises(ArgumentError, /attention_bias/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end
  end

  it "rejects an index missing any of the required text tensors" do
    with_qwen3vl_weights_dir do |dir|
      write_qwen3vl_weights_metadata(dir)
      expect_raises(ArgumentError, /missing required text tensors/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end
  end

  it "rejects shard traversal even when the index includes all required text names" do
    with_qwen3vl_weights_dir do |dir|
      weight_map = ML::GGUF::Qwen3VLTextWeights.required_tensor_names.to_h do |name|
        {name, "../outside.safetensors"}
      end
      write_qwen3vl_weights_metadata(dir, weight_map: weight_map)

      expect_raises(ArgumentError, /shard path/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end
  end

  it "rejects safetensors offsets extending beyond the shard payload" do
    with_qwen3vl_weights_dir do |dir|
      shard_name = "truncated.safetensors"
      weight_map = ML::GGUF::Qwen3VLTextWeights.required_tensor_names.to_h do |name|
        {name, shard_name}
      end
      write_qwen3vl_weights_metadata(dir, weight_map: weight_map)
      write_qwen3vl_truncated_embedding_shard(File.join(dir, shard_name))

      expect_raises(ArgumentError, /offsets outside shard payload/) do
        ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      end
    end
  end

  it "rejects an embedding with a wrong shape or non-BF16 dtype" do
    [
      {"BF16", [1], Bytes.new(2)},
      {"F32", [1], Bytes.new(4)},
    ].each do |dtype, shape, payload|
      with_qwen3vl_weights_dir do |dir|
        shard_name = "bad-embedding.safetensors"
        weight_map = ML::GGUF::Qwen3VLTextWeights.required_tensor_names.to_h do |name|
          {name, shard_name}
        end
        write_qwen3vl_weights_metadata(dir, weight_map: weight_map)
        write_qwen3vl_embedding_shard(File.join(dir, shard_name), dtype, shape, payload)

        message = dtype == "BF16" ? /shape/ : /must be BF16/
        expect_raises(ArgumentError, message) do
          ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
        end
      end
    end
  end

  it "reads only selected BF16 rows from a complete sparse shard and closes safely" do
    with_qwen3vl_weights_dir do |dir|
      shard_name = "synthetic-full.safetensors"
      weight_map = ML::GGUF::Qwen3VLTextWeights.required_tensor_names.to_h do |name|
        {name, shard_name}
      end
      write_qwen3vl_weights_metadata(dir,
        config: qwen3vl_weights_config_json(rope_theta: 5_000_000_f64),
        weight_map: weight_map)
      first_row, last_row = write_qwen3vl_sparse_complete_shard(File.join(dir, shard_name))
      weights = ML::GGUF::Qwen3VLTextWeights.from_directory(dir)
      begin
        last_token = ML::GGUF::Qwen3VLTextWeights::VOCAB_SIZE - 1
        rows = weights.embedding_rows_raw([0_i64, last_token.to_i64])
        rows[0, first_row.size].should eq(first_row)
        rows[first_row.size, last_row.size].should eq(last_row)
        decoded = weights.embedding_rows_f32([0_i64, last_token.to_i64])
        decoded[0].should eq(1.0_f32)
        decoded[ML::GGUF::Qwen3VLTextWeights::HIDDEN_SIZE].should eq(2.0_f32)
        expect_raises(ArgumentError, /outside vocabulary/) do
          weights.embedding_rows_raw([ML::GGUF::Qwen3VLTextWeights::VOCAB_SIZE.to_i64])
        end
      ensure
        weights.close
      end
      expect_raises(ArgumentError, /loader is closed/) do
        weights.embedding_rows_raw([] of Int64)
      end
    end
  end
end

if ENV["QWEN3VL_TEXT_ENCODER_DIR"]? && ENV["QWEN3VL_TEXT_REFERENCE_DIR"]?
  describe "optional real Qwen3-VL embedding parity" do
    it "matches all 24 red-cube hidden_state_000 rows bit-for-bit" do
      encoder_dir = ENV["QWEN3VL_TEXT_ENCODER_DIR"].not_nil!
      reference_dir = ENV["QWEN3VL_TEXT_REFERENCE_DIR"].not_nil!
      reference_path = File.join(reference_dir, "qwen_image21_text_reference.json")
      reference = JSON.parse(File.read(reference_path)).as_h
      reference["model"].as_h["revision_sha"].as_s.should eq(ML::GGUF::Qwen3VLTextWeights::EXPECTED_MODEL_REVISION)
      payload_path = File.join(reference_dir, reference["payload_file"].as_s)
      payload_file = File.read(payload_path)
      payload = payload_file.to_slice
      Digest::SHA256.hexdigest(payload).should eq(reference["payload_sha256"].as_s)
      descriptors = reference["tensors"].as_h
      ids_descriptor = descriptors["input_ids"].as_h
      hidden_descriptor = descriptors["hidden_state_000"].as_h
      ids_descriptor["dtype"].as_s.should eq("int64-le")
      hidden_descriptor["dtype"].as_s.should eq("bfloat16-le")
      ids_offset = ids_descriptor["offset_bytes"].as_i
      hidden_offset = hidden_descriptor["offset_bytes"].as_i
      token_count = ids_descriptor["shape"].as_a.last.as_i
      hidden_dim = hidden_descriptor["shape"].as_a.last.as_i
      token_count.should eq(24)
      hidden_dim.should eq(ML::GGUF::Qwen3VLTextWeights::HIDDEN_SIZE)
      row_bytes = hidden_dim * 2
      mask_descriptor = descriptors["attention_mask"].as_h
      mask_offset = mask_descriptor["offset_bytes"].as_i
      token_count.times do |index|
        IO::ByteFormat::LittleEndian.decode(Int64, payload[mask_offset + index * 8, 8]).should eq(1_i64)
      end
      weights = ML::GGUF::Qwen3VLTextWeights.from_directory(encoder_dir)

      begin
        ids = Array(Int64).new(token_count) do |index|
          IO::ByteFormat::LittleEndian.decode(Int64, payload[ids_offset + index * 8, 8])
        end
        actual = weights.embedding_rows_raw(ids)
        mismatch_count = 0
        token_count.times do |index|
          expected_row = payload[hidden_offset + index * row_bytes, row_bytes]
          actual_row = actual[index * row_bytes, row_bytes]
          row_bytes.times do |byte_index|
            mismatch_count += 1 if actual_row[byte_index] != expected_row[byte_index]
          end
        end

        mismatch_count.should eq(0), "expected all 24 BF16 embedding rows to match, got #{mismatch_count} mismatched bytes"
        actual.size.should eq(token_count * row_bytes)
        weights.embedding_rows_f32(ids).size.should eq(token_count * hidden_dim)
        weights.close
        expect_raises(ArgumentError, /loader is closed/) do
          weights.embedding_rows_raw([] of Int64)
        end
        fixture_revision = reference["model"].as_h["revision_sha"].as_s
        puts "Qwen3VL embedding parity recorded_fixture_revision=#{fixture_revision} token_rows=#{token_count} scalars=#{token_count * hidden_dim} mismatched_bytes=#{mismatch_count}"
      ensure
        weights.close
      end
    end
  end
end
