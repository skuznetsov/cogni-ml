require "spec"
require "digest/sha256"
require "../examples/support/qwen25_prefix_probe"

private def qwen25_prompt_contract
  crop = 120
  tokens = Array(Int32).new(1280 + crop, 151643)
  tokens[0] = 151644
  tokens[crop - 3] = 151644
  tokens[crop - 2] = 872
  tokens[crop - 1] = 198
  tokens[crop] = 32
  mask = Array(Int32).new(tokens.size) { |index| index <= crop ? 1 : 0 }
  prompt = "A natural scene."
  rendered = "source fixture; native rendering checked separately"
  metadata = JSON.parse({
    "prompt" => prompt, "prompt_sha256" => Digest::SHA256.hexdigest(prompt),
    "rendered_prompt" => rendered, "rendered_prompt_sha256" => Digest::SHA256.hexdigest(rendered),
    "template_name" => "li-dit-encode-video-json", "upstream_revision" => "911cea6458a89c2663513ff7099561ec190dd491",
    "marker" => "<|im_start|>user\n", "marker_token_ids" => [151644, 872, 198],
    "crop_start" => crop, "raw_length" => tokens.size, "text_length" => 1280,
    "token_ids" => tokens, "attention_mask" => mask,
    "cropped_token_ids" => tokens[crop..], "cropped_attention_mask" => mask[crop..],
  }.to_json)
  {metadata, tokens, mask}
end

describe "Hy-Nomas prompt input binding" do
  it "admits a full raw request and its 1280-token output suffix" do
    metadata, tokens, mask = qwen25_prompt_contract
    Qwen25PrefixProbe.validate_prompt_contract!(metadata, tokens, mask, 0).should eq(120)
  end

  it "rejects cropping inputs before forward, wrong masks, and shifted crop" do
    metadata, tokens, mask = qwen25_prompt_contract
    expect_raises(ArgumentError) { Qwen25PrefixProbe.validate_prompt_contract!(metadata, tokens[120..], mask[120..], 0) }
    bad_mask = mask.dup
    bad_mask[1] = 0
    expect_raises(ArgumentError) { Qwen25PrefixProbe.validate_prompt_contract!(metadata, tokens, bad_mask, 0) }
    metadata.as_h["crop_start"] = JSON::Any.new(119_i64)
    expect_raises(ArgumentError) { Qwen25PrefixProbe.validate_prompt_contract!(metadata, tokens, mask, 0) }
  end

  it "rejects a changed prompt digest and nonzero model positions" do
    metadata, tokens, mask = qwen25_prompt_contract
    expect_raises(ArgumentError) { Qwen25PrefixProbe.validate_prompt_contract!(metadata, tokens, mask, 1) }
    metadata.as_h["prompt"] = JSON::Any.new("changed prompt")
    expect_raises(ArgumentError) { Qwen25PrefixProbe.validate_prompt_contract!(metadata, tokens, mask, 0) }
  end
end

private def with_qwen25_prefix_fixture(&block : String, String ->)
  root = ""
  checkpoint = ""
  references = ""
  begin
    root = File.join(Dir.tempdir, "qwen25-prefix-probe-#{Process.pid}-#{Random.rand(1_000_000_000)}")
    checkpoint = File.join(root, "checkpoint")
    references = File.join(root, "reference")
    Dir.mkdir(root)
    Dir.mkdir(checkpoint)
    Dir.mkdir(references)

    config_bytes = "small authenticated config file\n"
    index_bytes = "small authenticated index file\n"
    shard_name = "model-00001-of-00004.safetensors"
    shard_bytes = "metadata-only placeholder; deliberately not a safetensors header\n"
    File.write(File.join(checkpoint, "config.json"), config_bytes)
    File.write(File.join(checkpoint, "model.safetensors.index.json"), index_bytes)
    File.write(File.join(checkpoint, shard_name), shard_bytes)

    token_ids = [151644_i32, 872_i32, 151645_i32, 0_i32]
    attention_mask = [1_i32, 1_i32, 0_i32, 0_i32]
    sidecar_name = "states.bf16"
    sidecar_bytes = Bytes.new(27 * token_ids.size * 3584 * 2, 0_u8)
    File.write(File.join(references, sidecar_name), sidecar_bytes)

    config_sha = Digest::SHA256.hexdigest(config_bytes)
    index_sha = Digest::SHA256.hexdigest(index_bytes)
    shard_sha = Digest::SHA256.hexdigest(shard_bytes)
    verified = {
      "config.json"                  => {path: "config.json", nbytes: config_bytes.bytesize, sha256: config_sha},
      "model.safetensors.index.json" => {path: "model.safetensors.index.json", nbytes: index_bytes.bytesize, sha256: index_sha},
      shard_name                     => {path: shard_name, nbytes: shard_bytes.bytesize, sha256: shard_sha},
    }
    shapes = ML::GGUF::Qwen25VLWeights.selected_tensor_shapes
    inventory = shapes.reject { |name, _| name == ML::GGUF::Qwen25VLWeights::EMBEDDING_NAME }
    config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b
    expected_sidecar_size = sidecar_bytes.size

    manifest = JSON.build(indent: "  ") do |json|
      json.object do
        json.field "schema", "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference"
        json.field "schema_version", 1
        json.field "checkpoint" do
          json.object do
            json.field "repo_id", "Qwen/Qwen2.5-VL-7B-Instruct"
            json.field "revision", ML::GGUF::Qwen25VLWeights::EXPECTED_MODEL_REVISION
            json.field "root", checkpoint
            json.field "receipt_path", File.join(checkpoint, "download-receipt.json")
            json.field "receipt_sha256", "a" * 64
            json.field "config_sha256", config_sha
            json.field "index_sha256", index_sha
            json.field "selected_shards", [shard_name]
            json.field "verified_files" do
              json.object do
                verified.each do |name, file|
                  json.field name do
                    json.object do
                      json.field "path", file[:path]
                      json.field "nbytes", file[:nbytes]
                      json.field "sha256", file[:sha256]
                      json.field "hf_lfs_sha256", nil
                    end
                  end
                end
              end
            end
          end
        end
        json.field "config" do
          json.object do
            json.field "hidden_size", config.dim
            json.field "intermediate_size", config.ffn_dim
            json.field "num_attention_heads", config.heads
            json.field "num_key_value_heads", config.kv_heads
            json.field "num_hidden_layers", 28
            json.field "head_dim", config.head_dim
            json.field "max_position_embeddings", config.max_positions
            json.field "vocab_size", ML::GGUF::Qwen25VLWeights::VOCAB_SIZE
            json.field "rms_norm_eps", config.rms_norm_eps.to_f64
            json.field "rope_theta", config.rope_theta.to_f64
            json.field "hidden_act", "silu"
            json.field "rope_scaling" do
              json.object { json.field "mrope_section", [16, 24, 24] }
            end
            json.field "sequence_length", token_ids.size
            json.field "attention_implementation", "sdpa"
          end
        end
        json.field "inputs" do
          json.object do
            json.field "token_ids", token_ids
            json.field "attention_mask", attention_mask
            json.field "position_offset", 0
            json.field "position_ids_f32" do
              json.array do
                3.times do
                  json.array { token_ids.size.times { |index| json.number(index.to_f64) } }
                end
              end
            end
            json.field "position_ids_for_official_model" do
              json.array do
                3.times do
                  json.array { token_ids.size.times { |index| json.number(index) } }
                end
              end
            end
            json.field "multimodal_position_shape", [3, 1, token_ids.size]
            json.field "inv_freq_f32" do
              json.array do
                (config.head_dim // 2).times { |index| json.number(1.0_f64 / (index + 1)) }
              end
            end
          end
        end
        json.field "capture" do
          json.object do
            json.field "source_layer_count", 28
            json.field "prefix_layer_count", 26
            json.field "captured_state_count", 27
            json.field "state_indexing", "0=token_embedding; i+1=decoder_layer_i_output"
            json.field "hidden_states" do
              json.object do
                json.field "dtype", "bfloat16"
                json.field "shape", [27, 1, token_ids.size, config.dim]
                json.field "storage" do
                  json.object do
                    json.field "sidecar", "bf16"
                    json.field "offset_bytes", 0
                    json.field "nbytes", expected_sidecar_size
                  end
                end
              end
            end
            json.field "embedding_unique_rows" do
              json.array do
                token_ids.uniq.sort.each do |id|
                  json.object do
                    json.field "row", id
                    json.field "key", ML::GGUF::Qwen25VLWeights::EMBEDDING_NAME
                    json.field "dtype", "BF16"
                    json.field "tensor_sha256", "b" * 64
                  end
                end
              end
            end
          end
        end
        json.field "sidecars" do
          json.object do
            json.field "bf16" do
              json.object do
                json.field "path", sidecar_name
                json.field "encoding", "bfloat16-le"
                json.field "nbytes", expected_sidecar_size
                json.field "sha256", Digest::SHA256.hexdigest(sidecar_bytes)
              end
            end
          end
        end
        json.field "weight_sources" do
          json.array do
            inventory.each do |name, shape|
              json.object do
                json.field "key", name
                json.field "dtype", "BF16"
                json.field "shape", shape
                json.field "nbytes", shape.reduce(1_i64) { |count, dimension| count * dimension } * 2
                json.field "shard", shard_name
                json.field "tensor_sha256", "c" * 64
              end
            end
          end
        end
        json.field "provenance" do
          json.object do
            json.field "device", "mps"
            json.field "activation_dtype", "bfloat16"
            json.field "comparison_tolerance" do
              json.object do
                json.field "atol", Qwen25PrefixProbe::ATOL
                json.field "rtol", Qwen25PrefixProbe::RTOL
              end
            end
            json.field "explicit_mps_synchronizations", 1
            json.field "producer_source_sha256", "d" * 64
            json.field "upstream_helper_source_sha256", "e" * 64
            json.field "full_tuple_produced", false
          end
        end
      end
    end
    manifest_path = File.join(references, "prefix.json")
    File.write(manifest_path, manifest)
    yield manifest_path, checkpoint
  ensure
    if !checkpoint.empty? && Dir.exists?(checkpoint)
      Dir.glob(File.join(checkpoint, "*")).each { |path| File.delete?(path) }
      Dir.delete(checkpoint)
    end
    if !references.empty? && Dir.exists?(references)
      Dir.glob(File.join(references, "*")).each { |path| File.delete?(path) }
      Dir.delete(references)
    end
    Dir.delete(root) if !root.empty? && Dir.exists?(root)
  end
end

private def rewrite_qwen25_prefix_manifest(path : String, &block : JSON::Any -> Nil) : Nil
  document = JSON.parse(File.read(path))
  yield document
  File.write(path, document.to_json)
end

private def load_qwen25_prefix_fixture(manifest : String, checkpoint : String) : Qwen25PrefixProbe::Fixture
  trusted_digest = Digest::SHA256.hexdigest(File.read(manifest))
  Qwen25PrefixProbe::Fixture.new(manifest, checkpoint, trusted_digest)
end

private def qwen25_prefix_replace(document : JSON::Any, object_path : Array(String), key : String, value_json : String) : Nil
  target = object_path.reduce(document) { |node, member| node[member] }
  target.as_h[key] = JSON.parse(value_json)
end

describe Qwen25PrefixProbe::Fixture do
  it "accepts a minimal pinned-geometry fixture while authenticating regular source files only" do
    with_qwen25_prefix_fixture do |manifest, checkpoint|
      fixture = load_qwen25_prefix_fixture(manifest, checkpoint)
      fixture.tokens.should eq([151644_i32, 872_i32, 151645_i32, 0_i32])
      fixture.attention_mask.should eq([1_i32, 1_i32, 0_i32, 0_i32])
      fixture.positions.should eq([0_f32, 1_f32, 2_f32, 3_f32])
      fixture.inv_freq.size.should eq(64)
      fixture.manifest_sha256.should eq(Digest::SHA256.hexdigest(File.read(manifest)))
      File.size(fixture.sidecar).should eq(27 * 4 * 3584 * 2)
      tensor_hashes = fixture.tensor_sha256s
      tensor_hashes.size.should eq(312)
      tensor_hashes.clear
      fixture.tensor_sha256s.size.should eq(312)
    end
  end

  it "compares against the authenticated sidecar snapshot after the source path changes" do
    with_qwen25_prefix_fixture do |manifest, checkpoint|
      fixture = load_qwen25_prefix_fixture(manifest, checkpoint)
      expected = fixture.expected_state(0)
      expected.size.should eq(4 * 3584 * 2)
      File.write(fixture.sidecar, Bytes.new(File.size(fixture.sidecar), 0xff_u8))

      fixture.expected_state(0).should eq(expected)
      expected[0] = 1_u8
      fixture.expected_state(0)[0].should eq(0_u8)
    end
  end

  it "rejects malformed masks and packed hidden-state geometry" do
    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["inputs"], "attention_mask", "[1, 0, 1, 0]")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end

    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["capture", "hidden_states"], "shape", "[26, 1, 4, 3584]")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end
  end

  it "rejects sidecar and checkpoint hash mismatches" do
    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["sidecars", "bf16"], "sha256", "\"#{"0" * 64}\"")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end

    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["checkpoint", "verified_files", "config.json"], "sha256", "\"#{"0" * 64}\"")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end
  end

  it "rejects traversal and symlink paths for sidecars and checkpoint sources" do
    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["sidecars", "bf16"], "path", "\"../checkpoint/config.json\"")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end

    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["checkpoint", "verified_files"], "../outside", "{}")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end

    with_qwen25_prefix_fixture do |manifest, checkpoint|
      reference_dir = File.dirname(manifest)
      File.symlink("states.bf16", File.join(reference_dir, "linked.bf16"))
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["sidecars", "bf16"], "path", "\"linked.bf16\"")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end
  end

  it "rejects numerical provenance drift" do
    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["provenance"], "device", "\"cpu\"")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end

    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["provenance"], "explicit_mps_synchronizations", "2")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end
  end

  it "rejects a changed manifest even when its embedded sidecar hash is refreshed" do
    with_qwen25_prefix_fixture do |manifest, checkpoint|
      trusted_digest = Digest::SHA256.hexdigest(File.read(manifest))
      sidecar = File.join(File.dirname(manifest), "states.bf16")
      File.open(sidecar, "r+") do |io|
        first_byte = Bytes.new(1)
        io.read_fully(first_byte)
        first_byte[0] = 1_u8
        io.seek(0)
        io.write(first_byte)
      end
      refreshed_sidecar_hash = Digest::SHA256.hexdigest(File.read(sidecar))
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        qwen25_prefix_replace(document, ["sidecars", "bf16"], "sha256", "\"#{refreshed_sidecar_hash}\"")
        qwen25_prefix_replace(document, ["provenance"], "device", "\"cpu\"")
      end

      expect_raises(ArgumentError, /trusted manifest digest mismatch/) do
        Qwen25PrefixProbe::Fixture.new(manifest, checkpoint, trusted_digest)
      end
    end
  end

  it "rejects incomplete or malformed 26-layer weight provenance" do
    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        document.as_h["weight_sources"].as_a.pop
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end

    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        first = document["weight_sources"].as_a.first
        first.as_h["shard"] = JSON.parse("\"not-selected.safetensors\"")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end

    with_qwen25_prefix_fixture do |manifest, checkpoint|
      rewrite_qwen25_prefix_manifest(manifest) do |document|
        first = document["weight_sources"].as_a.first
        first.as_h["dtype"] = JSON.parse("\"F16\"")
      end
      expect_raises(ArgumentError) { load_qwen25_prefix_fixture(manifest, checkpoint) }
    end
  end
end

describe Qwen25PrefixProbe do
  it "applies tolerance to one-ULP BF16 deltas and reports their magnitude" do
    expected = Bytes[0x80_u8, 0x3f_u8] # 1.0
    actual = Bytes[0x81_u8, 0x3f_u8]   # next BF16 value
    comparison = Qwen25PrefixProbe.compare_bf16_bytes(expected, actual)
    comparison[:failures].should eq(1)
    comparison[:max_abs].should eq(0.0078125_f64)
    comparison[:different_bf16_words].should eq(1)

    subnormal = Qwen25PrefixProbe.compare_bf16_bytes(Bytes[0_u8, 0_u8], Bytes[1_u8, 0_u8])
    subnormal[:failures].should eq(0)
    subnormal[:different_bf16_words].should eq(1)
    (subnormal[:max_abs] > 0.0).should be_true
  end

  it "separates exact BF16 words from numerical equality without changing the gate" do
    positive_zero = Bytes[0_u8, 0_u8]
    negative_zero = Bytes[0_u8, 0x80_u8]
    signed_zero = Qwen25PrefixProbe.compare_bf16_bytes(positive_zero, negative_zero)
    signed_zero[:failures].should eq(0)
    signed_zero[:max_abs].should eq(0.0)
    signed_zero[:different_bf16_words].should eq(1)
    exact = Qwen25PrefixProbe.compare_bf16_bytes(negative_zero, negative_zero)
    exact[:failures].should eq(0)
    exact[:different_bf16_words].should eq(0)
  end

  it "rejects nonfinite BF16 values and unsafe byte extents" do
    comparison = Qwen25PrefixProbe.compare_bf16_bytes(Bytes[0_u8, 0_u8], Bytes[0x80_u8, 0x7f_u8]) # +infinity
    comparison[:failures].should eq(1)
    comparison[:max_abs].should eq(Float64::MAX)
    expect_raises(ArgumentError) { Qwen25PrefixProbe.compare_bf16_bytes(Bytes.empty, Bytes.empty) }
    expect_raises(ArgumentError) { Qwen25PrefixProbe.compare_bf16_bytes(Bytes[0_u8], Bytes[0_u8]) }
    expect_raises(ArgumentError) { Qwen25PrefixProbe.compare_bf16_bytes(Bytes[0_u8, 0_u8], Bytes[0_u8, 0_u8, 0_u8, 0_u8]) }
  end
end

private class FakeQwen25PrefixDrainChain
  getter drain_calls : Int32 = 0
  @completed = false

  def initialize(@complete_after_drain : Bool, @raise_on_drain : Bool)
  end

  def commit_and_wait : Nil
    @drain_calls += 1
    @completed = @complete_after_drain
    raise Exception.new("synthetic terminal drain failure") if @raise_on_drain
  end

  def completed? : Bool
    @completed
  end
end

describe Qwen25PrefixProbe do
  it "cleans up after a successful idempotent terminal drain" do
    chain = FakeQwen25PrefixDrainChain.new(true, false)
    cleaned = false

    Qwen25PrefixProbe.cleanup_after_drain(chain) { cleaned = true }

    chain.drain_calls.should eq(1)
    cleaned.should be_true
  end

  it "cleans up after a completed terminal error and preserves that error" do
    chain = FakeQwen25PrefixDrainChain.new(true, true)
    cleaned = false

    expect_raises(Exception, /synthetic terminal drain failure/) do
      Qwen25PrefixProbe.cleanup_after_drain(chain) { cleaned = true }
    end

    chain.drain_calls.should eq(1)
    cleaned.should be_true
  end

  it "retains resources when a failing drain has not completed" do
    chain = FakeQwen25PrefixDrainChain.new(false, true)
    cleaned = false

    expect_raises(Exception, /synthetic terminal drain failure/) do
      Qwen25PrefixProbe.cleanup_after_drain(chain) { cleaned = true }
    end

    chain.drain_calls.should eq(1)
    cleaned.should be_false
  end

  it "cleans up when no command chain was created" do
    cleaned = false

    Qwen25PrefixProbe.cleanup_after_drain(nil) { cleaned = true }

    cleaned.should be_true
  end

  it "narrows a nullable fake chain before attempting its terminal drain" do
    chain : FakeQwen25PrefixDrainChain? = FakeQwen25PrefixDrainChain.new(true, false)
    cleaned = false

    Qwen25PrefixProbe.cleanup_after_drain(chain) { cleaned = true }

    chain.not_nil!.drain_calls.should eq(1)
    cleaned.should be_true
  end
end
