require "./spec_helper"
require "../src/ml/gguf/qwen3vl_text_reference"

QWEN3VL_TEST_REVISION = "0123456789012345678901234567890123456789"

private def qwen3vl_encode_test_float(data : Bytes, element : Int32, value : Float32, dtype : String) : Nil
  case dtype
  when "float32-le"
    IO::ByteFormat::LittleEndian.encode(value, data[element * 4, 4])
  when "bfloat16-le"
    bits = (value.unsafe_as(UInt32) >> 16).to_u16
    IO::ByteFormat::LittleEndian.encode(bits, data[element * 2, 2])
  else
    raise "unsupported test float dtype: #{dtype}"
  end
end

private def qwen3vl_reference_test_bundle(
  embedding_first : Float32 = 0.25_f32,
  final_hidden_first : Float32 = 0.25_f32,
  mask_values : Array(Int64) = [1_i64, 1_i64],
  drop_idx : Int32 = 0,
  final_hidden_second_first : Float32 = 0.0_f32,
  hidden_state_count : Int32 = 2,
  model_revision : String = QWEN3VL_TEST_REVISION,
  float_dtype : String = "float32-le",
) : {String, Bytes}
  raise "test mask helper expects two raw tokens" unless mask_values.size == 2
  actual_sequence_length = mask_values.count(1_i64).to_i32 - drop_idx
  bytes_per_float = float_dtype == "bfloat16-le" ? 2 : 4
  captured_float_dtype = float_dtype == "bfloat16-le" ? "bfloat16" : "float32"
  ids = Bytes.new(16, 0_u8)
  IO::ByteFormat::LittleEndian.encode(101_i64, ids[0, 8])
  IO::ByteFormat::LittleEndian.encode(202_i64, ids[8, 8])
  mask = Bytes.new(16, 0_u8)
  mask_values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, mask[index * 8, 8])
  end

  embeddings = Bytes.new(actual_sequence_length * 4096 * bytes_per_float, 0_u8)
  qwen3vl_encode_test_float(embeddings, 0, embedding_first, float_dtype)
  hidden_states = Array(Bytes).new(hidden_state_count) do |index|
    state = Bytes.new(2 * 4096 * bytes_per_float, 0_u8)
    first = if index == hidden_state_count - 1
              final_hidden_first
            elsif index == 0
              1.5_f32
            else
              index.to_f32
            end
    qwen3vl_encode_test_float(state, 0, first, float_dtype)
    if index == hidden_state_count - 1
      qwen3vl_encode_test_float(state, 4096, final_hidden_second_first, float_dtype)
    end
    state
  end

  pieces = [ids, mask, embeddings] of Bytes
  pieces.concat(hidden_states)
  payload_io = IO::Memory.new
  pieces.each { |piece| payload_io.write(piece) }
  payload = payload_io.to_slice

  offset = 0
  descriptors = [] of {String, String, Array(Int32), Bytes}
  descriptors << {"input_ids", "int64-le", [1, 2], ids}
  descriptors << {"attention_mask", "int64-le", [1, 2], mask}
  descriptors << {"pre_final_rmsnorm_embeddings", float_dtype, [1, actual_sequence_length, 4096], embeddings}
  hidden_states.each_with_index do |data, index|
    descriptors << {"hidden_state_#{index.to_s.rjust(3, '0')}", float_dtype, [1, 2, 4096], data}
  end

  tensors = JSON.build do |json|
    json.object do
      descriptors.each do |name, dtype, shape, data|
        json.field name do
          json.object do
            json.field "dtype", dtype
            json.field "captured_dtype", dtype == "int64-le" ? "int64" : captured_float_dtype
            json.field "shape", shape
            json.field "offset_bytes", offset
            json.field "nbytes", data.size
            json.field "sha256", Digest::SHA256.hexdigest(data)
          end
        end
        offset += data.size
      end
    end
  end

  manifest = JSON.build do |json|
    json.object do
      json.field "schema", "qwen-image21-text-reference"
      json.field "schema_version", 1
      json.field "model" do
        json.object do
          json.field "repo", "Qwen/Qwen-Image-2.1"
          json.field "revision_sha", model_revision
        end
      end
      json.field "prompt", "a tiny lighthouse"
      json.field "tokenization" do
        json.object do
          json.field "raw_template_text", "<user>a tiny lighthouse</user>"
        end
      end
      json.field "sequence" do
        json.object do
          json.field "max_sequence_length", 8
          json.field "actual_sequence_length", actual_sequence_length
          json.field "raw_input_shape", [1, 2]
          json.field "drop_idx", drop_idx
        end
      end
      json.field "embedding" do
        json.object do
          json.field "pre_final_rmsnorm", true
          json.field "shape", [1, actual_sequence_length, 4096]
          json.field "source_dtype", captured_float_dtype
          json.field "expected_hidden_state_count", hidden_state_count
          json.field "hidden_state_count", hidden_state_count
        end
      end
      json.field "payload_file", "qwen_image21_text_reference.bin"
      json.field "payload_nbytes", payload.size
      json.field "payload_sha256", Digest::SHA256.hexdigest(payload)
      json.field "tensors" do
        json.raw(tensors)
      end
    end
  end
  {manifest, payload}
end

describe ML::GGUF::Qwen3VLTextReference do
  it "loads token inputs and exposes final pre-norm embeddings and hidden states as Float32" do
    manifest, payload = qwen3vl_reference_test_bundle
    reference = ML::GGUF::Qwen3VLTextReference.parse(manifest, payload)

    reference.model_revision.should eq(QWEN3VL_TEST_REVISION)
    reference.prompt.should eq("a tiny lighthouse")
    reference.input_ids.should eq([101_i64, 202_i64])
    reference.attention_mask.should eq([true, true])
    reference.raw_sequence_length.should eq(2)
    reference.hidden_state_count.should eq(2)
    reference.pre_final_rmsnorm_embeddings.size.should eq(2 * 4096)
    reference.pre_final_rmsnorm_embeddings[0].should eq(0.25_f32)
    reference.hidden_state(0).size.should eq(2 * 4096)
    reference.hidden_state(0)[0].should eq(1.5_f32)
    reference.hidden_state(1)[0].should eq(0.25_f32)
  end

  it "loads the sibling payload from a manifest path" do
    manifest, payload = qwen3vl_reference_test_bundle
    dir = File.join(Dir.tempdir, "qwen3vl-reference-#{Random.rand(1_000_000_000)}")
    Dir.mkdir(dir)
    begin
      manifest_path = File.join(dir, "qwen_image21_text_reference.json")
      File.write(manifest_path, manifest)
      File.write(File.join(dir, "qwen_image21_text_reference.bin"), payload)

      reference = ML::GGUF::Qwen3VLTextReference.load(manifest_path)
      reference.input_ids.should eq([101_i64, 202_i64])
    ensure
      Dir.glob(File.join(dir, "*")).each { |path| File.delete(path) }
      Dir.delete(dir)
    end
  end

  it "rejects schema, version, and model revision drift" do
    manifest, payload = qwen3vl_reference_test_bundle
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest.gsub("qwen-image21-text-reference", "other"), payload)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest.gsub("\"schema_version\":1", "\"schema_version\":2"), payload)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest.gsub(QWEN3VL_TEST_REVISION, "bad-revision"), payload)
    end
  end

  it "rejects bad payload length, tensor bounds, shape, dtype, and checksums" do
    manifest, payload = qwen3vl_reference_test_bundle
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(
        manifest.gsub("\"payload_nbytes\":#{payload.size}", "\"payload_nbytes\":#{payload.size - 1}"),
        payload,
      )
    end

    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest.gsub("\"offset_bytes\":0", "\"offset_bytes\":999999"), payload)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest.gsub("\"shape\":[1,2]", "\"shape\":[1,3]"), payload)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest.gsub("\"dtype\":\"int64-le\"", "\"dtype\":\"float32-le\""), payload)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest.gsub("\"sha256\":\"#{Digest::SHA256.hexdigest(payload[0, 16])}\"", "\"sha256\":\"#{"0" * 64}\""), payload)
    end
    corrupt = payload.dup
    corrupt[0] = corrupt[0] ^ 1_u8
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest, corrupt)
    end
  end

  it "rejects embeddings that are not the attended final hidden state after drop_idx" do
    manifest, payload = qwen3vl_reference_test_bundle(0.75_f32, 0.25_f32)
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(manifest, payload)
    end
  end

  it "selects the attended token after padding and drop_idx" do
    manifest, payload = qwen3vl_reference_test_bundle(
      2.5_f32,
      0.25_f32,
      [0_i64, 1_i64],
      0,
      2.5_f32,
    )
    reference = ML::GGUF::Qwen3VLTextReference.parse(manifest, payload)
    reference.attention_mask.should eq([false, true])
    reference.actual_sequence_length.should eq(1)
    reference.pre_final_rmsnorm_embeddings[0].should eq(2.5_f32)
    reference.hidden_state(1)[4096].should eq(2.5_f32)
  end

  it "accepts the earlier schema-v1 count shape but still requires actual tensor count consistency" do
    manifest, payload = qwen3vl_reference_test_bundle
    earlier_manifest = manifest.gsub("\"expected_hidden_state_count\":2,", "")
    ML::GGUF::Qwen3VLTextReference.parse(earlier_manifest, payload).hidden_state_count.should eq(2)

    inconsistent_manifest = earlier_manifest.gsub("\"hidden_state_count\":2", "\"hidden_state_count\":1")
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(inconsistent_manifest, payload)
    end
  end

  it "requires 37 layers for the pinned reference revision even when the old manifest omits expected count" do
    matching_manifest, matching_payload = qwen3vl_reference_test_bundle(
      0.25_f32,
      0.25_f32,
      [1_i64, 1_i64],
      0,
      0.0_f32,
      37,
      ML::GGUF::Qwen3VLTextReference::RED_CUBE_REVISION,
      "bfloat16-le",
    )
    matching_earlier_manifest = matching_manifest.gsub("\"expected_hidden_state_count\":37,", "")
    ML::GGUF::Qwen3VLTextReference.parse(matching_earlier_manifest, matching_payload)
      .hidden_state_count.should eq(37)

    manifest, payload = qwen3vl_reference_test_bundle(
      0.25_f32,
      0.25_f32,
      [1_i64, 1_i64],
      0,
      0.0_f32,
      36,
      ML::GGUF::Qwen3VLTextReference::RED_CUBE_REVISION,
      "bfloat16-le",
    )
    earlier_manifest = manifest.gsub("\"expected_hidden_state_count\":36,", "")
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(earlier_manifest, payload)
    end

    wrong_dtype_manifest, wrong_dtype_payload = qwen3vl_reference_test_bundle(
      0.25_f32,
      0.25_f32,
      [1_i64, 1_i64],
      0,
      0.0_f32,
      37,
      ML::GGUF::Qwen3VLTextReference::RED_CUBE_REVISION,
    )
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextReference.parse(wrong_dtype_manifest, wrong_dtype_payload)
    end
  end
end

if fixture_dir = ENV["QWEN3VL_TEXT_REFERENCE_DIR"]?
  unless fixture_dir.empty?
    describe "optional real Qwen3-VL text reference fixture" do
      it "validates the pinned red-cube bundle and returns layer data" do
        reference = ML::GGUF::Qwen3VLTextReference.load(
          File.join(fixture_dir, "qwen_image21_text_reference.json")
        )
        reference.input_ids.size.should eq(reference.raw_sequence_length)
        reference.attention_mask.size.should eq(reference.raw_sequence_length)
        reference.hidden_state_count.should eq(37)
        reference.hidden_state(0).size.should eq(reference.raw_sequence_length * 4096)
      end
    end
  end
end
