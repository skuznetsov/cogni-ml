require "json"
require "digest/sha256"
require "../../src/ml/gguf/qwen25_vl_weights"

# Source authentication and post-drain comparison for the bounded pretrained
# probe. This module contains no CPU implementation of neural operations.
module Qwen25PrefixProbe
  extend self
  MODEL_ID                 = "Qwen/Qwen2.5-VL-7B-Instruct"
  PREFIX_LAYERS            =   26
  MAX_PREFIX_TOKENS        = 1280
  MAX_PREFIX_SIDECAR_BYTES = (PREFIX_LAYERS + 1).to_i64 * MAX_PREFIX_TOKENS * 3584 * 2
  ATOL                     = 2e-5
  RTOL                     = 2e-5
  alias Comparison = NamedTuple(state: Int32, elements: Int32, failures: Int32,
    max_abs: Float64, different_bf16_words: Int32)

  # Run cleanup only after the terminal drain proves that borrowed GPU state is
  # no longer in use. A completed command can still report a GPU error; keep
  # that drain exception as the primary failure while allowing cleanup to run.
  def cleanup_after_drain(chain : T?, &cleanup : ->) : Nil forall T
    unless active_chain = chain
      yield
      return
    end

    drain_error : Exception? = nil
    cleanup_error : Exception? = nil

    begin
      active_chain.commit_and_wait
    rescue error : Exception
      drain_error = error
    ensure
      if active_chain.completed?
        begin
          yield
        rescue error : Exception
          cleanup_error = error
        end
      end
    end

    raise error if error = drain_error
    raise error if error = cleanup_error
    raise ArgumentError.new("terminal drain returned before chain completion") unless active_chain.completed?
  end

  def sha256(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      chunk = Bytes.new(1024 * 1024)
      while (count = io.read(chunk)) > 0
        digest.update(chunk[0, count])
      end
    end
    digest.hexfinal
  end

  def sibling(root : String, name : String) : String
    raise ArgumentError.new("source name must be a sibling basename") unless name.matches?(/\A[A-Za-z0-9_.-]+\z/) && name != "." && name != ".."
    path = File.join(root, name)
    raise ArgumentError.new("source must be a regular sibling file") unless File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
    path
  end

  # Diagnostic comparator, used only after the caller's terminal drain.
  def compare_bf16_bytes(expected : Bytes, actual : Bytes) : NamedTuple(failures: Int32, max_abs: Float64, different_bf16_words: Int32)
    raise ArgumentError.new("BF16 comparison size mismatch") unless expected.size == actual.size && expected.size > 0 && expected.size.even?
    failures = 0
    different_words = 0
    maximum = 0.0
    (expected.size // 2).times do |i|
      reference_word = expected[2 * i].to_u32 | (expected[2 * i + 1].to_u32 << 8)
      actual_word = actual[2 * i].to_u32 | (actual[2 * i + 1].to_u32 << 8)
      different_words += 1 unless reference_word == actual_word
      reference = (reference_word << 16).unsafe_as(Float32).to_f64
      value = (actual_word << 16).unsafe_as(Float32).to_f64
      difference = (value - reference).abs
      if !reference.finite? || !value.finite?
        failures += 1
        maximum = Float64::MAX
      else
        maximum = Math.max(maximum, difference)
        failures += 1 if difference > ATOL + RTOL * reference.abs
      end
    end
    {failures: failures, max_abs: maximum, different_bf16_words: different_words}
  end

  class Fixture
    getter document : JSON::Any
    getter tokens : Array(Int32)
    getter attention_mask : Array(Int32)
    getter positions : Array(Float32)
    getter inv_freq : Array(Float32)
    getter sidecar : String
    getter checkpoint_root : String
    getter manifest_path : String
    getter manifest_sha256 : String
    getter config = ML::GGUF::Qwen25VLConfig.hy_nomas_7b

    @hidden_state_snapshot : Bytes
    @tensor_sha256s : Hash(String, String)

    def initialize(path : String, checkpoint_root : String, expected_manifest_sha256 : String)
      @manifest_path = File.realpath(path)
      @checkpoint_root = File.realpath(checkpoint_root)
      manifest_bytes = File.read(@manifest_path)
      @manifest_sha256 = Digest::SHA256.hexdigest(manifest_bytes)
      # This digest is caller authority from a separately observed successful
      # oracle run, not a self-attestation contained in the manifest itself.
      raise ArgumentError.new("trusted manifest digest malformed") unless expected_manifest_sha256.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("trusted manifest digest mismatch") unless expected_manifest_sha256 == @manifest_sha256
      @document = JSON.parse(manifest_bytes)
      raise ArgumentError.new("unexpected pretrained fixture schema") unless @document["schema"].as_s == "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference" && @document["schema_version"].as_i == 1
      checkpoint = @document["checkpoint"]
      raise ArgumentError.new("checkpoint identity mismatch") unless checkpoint["repo_id"].as_s == MODEL_ID && checkpoint["revision"].as_s == ML::GGUF::Qwen25VLWeights::EXPECTED_MODEL_REVISION
      conf = @document["config"]
      expected = {"hidden_size" => @config.dim, "intermediate_size" => @config.ffn_dim,
                  "num_attention_heads" => @config.heads, "num_key_value_heads" => @config.kv_heads,
                  "num_hidden_layers" => @config.layers, "head_dim" => @config.head_dim,
                  "max_position_embeddings" => @config.max_positions, "vocab_size" => ML::GGUF::Qwen25VLWeights::VOCAB_SIZE}
      expected.each { |key, value| raise ArgumentError.new("fixture geometry mismatch: #{key}") unless conf[key].as_i == value }
      raise ArgumentError.new("fixture numerical configuration mismatch") unless conf["rms_norm_eps"].as_f.to_f32 == @config.rms_norm_eps && conf["rope_theta"].as_f.to_f32 == @config.rope_theta && conf["hidden_act"].as_s == "silu" && conf["attention_implementation"].as_s == "sdpa"
      raise ArgumentError.new("fixture rotary sections mismatch") unless conf["rope_scaling"]["mrope_section"].as_a.map(&.as_i) == [16, 24, 24]
      input = @document["inputs"]
      @tokens = input["token_ids"].as_a.map(&.as_i)
      @attention_mask = input["attention_mask"].as_a.map(&.as_i)
      count = @tokens.size
      raise ArgumentError.new("fixture tokens outside bounded prefix context") unless 1 < count <= 1280 && conf["sequence_length"].as_i == count
      raise ArgumentError.new("fixture token outside vocabulary") unless @tokens.all? { |id| 0 <= id < ML::GGUF::Qwen25VLWeights::VOCAB_SIZE }
      raise ArgumentError.new("fixture mask must be visible then right padded") unless @attention_mask.size == count && @attention_mask.first == 1 && @attention_mask.last == 0 && @attention_mask.all? { |v| v == 0 || v == 1 } && @attention_mask == @attention_mask.sort.reverse
      offset = input["position_offset"].as_i
      raise ArgumentError.new("fixture positions outside context") unless offset >= 0 && offset.to_i64 + count <= @config.max_positions
      @positions = Array(Float32).new(count) { |i| (offset + i).to_f32 }
      axes = input["position_ids_f32"].as_a
      official = input["position_ids_for_official_model"].as_a
      raise ArgumentError.new("fixture rotary position shape mismatch") unless input["multimodal_position_shape"].as_a.map(&.as_i) == [3, 1, count] && axes.size == 3 && official.size == 3
      axes.each { |axis| raise ArgumentError.new("fixture text rotary axes mismatch") unless axis.as_a.map { |v| v.as_f.to_f32 } == @positions }
      official.each { |axis| raise ArgumentError.new("fixture official rotary axes mismatch") unless axis.as_a.map(&.as_i) == Array(Int32).new(count) { |i| offset + i } }
      @inv_freq = input["inv_freq_f32"].as_a.map { |v| v.as_f.to_f32 }
      raise ArgumentError.new("fixture inverse frequencies invalid") unless @inv_freq.size == @config.head_dim // 2 && @inv_freq.all? { |v| v.finite? && v > 0 }
      capture = @document["capture"]
      raise ArgumentError.new("fixture selected prefix mismatch") unless capture["source_layer_count"].as_i == 28 && capture["prefix_layer_count"].as_i == PREFIX_LAYERS && capture["captured_state_count"].as_i == PREFIX_LAYERS + 1 && capture["state_indexing"].as_s == "0=token_embedding; i+1=decoder_layer_i_output"
      state = capture["hidden_states"]
      expected_size = (PREFIX_LAYERS + 1).to_i64 * count * @config.dim * 2
      storage = state["storage"]
      raise ArgumentError.new("fixture packed hidden states invalid") unless state["dtype"].as_s == "bfloat16" && state["shape"].as_a.map(&.as_i) == [27, 1, count, @config.dim] && storage["sidecar"].as_s == "bf16" && storage["offset_bytes"].as_i64 == 0 && storage["nbytes"].as_i64 == expected_size
      raise ArgumentError.new("fixture hidden states exceed bounded sidecar size") unless expected_size > 0 && expected_size <= MAX_PREFIX_SIDECAR_BYTES
      raise ArgumentError.new("unexpected fixture sidecars") unless @document["sidecars"].as_h.keys == ["bf16"]
      descriptor = @document["sidecars"]["bf16"]
      @sidecar = Qwen25PrefixProbe.sibling(File.dirname(@manifest_path), descriptor["path"].as_s)
      raise ArgumentError.new("fixture sidecar size/encoding mismatch") unless descriptor["encoding"].as_s == "bfloat16-le" && descriptor["nbytes"].as_i64 == expected_size && File.size(@sidecar) == expected_size
      @hidden_state_snapshot = read_sidecar_snapshot(expected_size)
      raise ArgumentError.new("fixture sidecar hash mismatch") unless descriptor["sha256"].as_s == Digest::SHA256.hexdigest(@hidden_state_snapshot)
      verified = checkpoint["verified_files"].as_h
      ["config.json", "model.safetensors.index.json"].each { |name| raise ArgumentError.new("fixture missing checkpoint source #{name}") unless verified.has_key?(name) }
      selected = checkpoint["selected_shards"].as_a.map(&.as_s)
      raise ArgumentError.new("fixture selected shard inventory invalid") unless selected.size > 0 && selected.uniq == selected && selected.all? { |name| name.ends_with?(".safetensors") && verified.has_key?(name) }
      verified.each do |name, info|
        source = Qwen25PrefixProbe.sibling(@checkpoint_root, name)
        raise ArgumentError.new("fixture source path/size mismatch: #{name}") unless info["path"].as_s == name && info["nbytes"].as_i64 == File.size(source)
        expected_hash = info["sha256"].as_s
        raise ArgumentError.new("fixture source hash malformed: #{name}") unless expected_hash.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new("fixture source hash mismatch: #{name}") unless Qwen25PrefixProbe.sha256(source) == expected_hash
        if lfs = info["hf_lfs_sha256"]?.try(&.as_s?)
          raise ArgumentError.new("fixture LFS source hash mismatch: #{name}") unless lfs == expected_hash
        end
      end
      raise ArgumentError.new("fixture config/index hash mismatch") unless checkpoint["config_sha256"].as_s == verified["config.json"]["sha256"].as_s && checkpoint["index_sha256"].as_s == verified["model.safetensors.index.json"]["sha256"].as_s
      inventory = ML::GGUF::Qwen25VLWeights.selected_tensor_shapes.reject { |name, _| name == ML::GGUF::Qwen25VLWeights::EMBEDDING_NAME }
      sources = @document["weight_sources"].as_a
      raise ArgumentError.new("fixture weight source inventory mismatch") unless sources.size == inventory.size && sources.map { |s| s["key"].as_s }.sort == inventory.keys.sort
      @tensor_sha256s = {} of String => String
      sources.each do |source|
        key = source["key"].as_s
        shape = inventory[key]
        digest = source["tensor_sha256"].as_s
        raise ArgumentError.new("fixture weight source metadata mismatch") unless source["dtype"].as_s == "BF16" && source["shape"].as_a.map(&.as_i64) == shape && source["nbytes"].as_i64 == shape.reduce(1_i64) { |n, v| n * v } * 2 && selected.includes?(source["shard"].as_s) && digest.matches?(/\A[0-9a-f]{64}\z/)
        @tensor_sha256s[key] = digest
      end
      provenance = @document["provenance"]
      raise ArgumentError.new("fixture numerical provenance mismatch") unless provenance["device"].as_s == "mps" && provenance["activation_dtype"].as_s == "bfloat16" && provenance["comparison_tolerance"]["atol"].as_f == ATOL && provenance["comparison_tolerance"]["rtol"].as_f == RTOL && provenance["explicit_mps_synchronizations"].as_i == 1
    end

    # Return a caller-owned copy of the validated selected-layer digest map.
    def tensor_sha256s : Hash(String, String)
      @tensor_sha256s.dup
    end

    # Return a caller-owned copy of one packed BF16 state from the authenticated
    # in-memory sidecar snapshot. The sidecar path is never reopened here.
    def expected_state(state : Int32) : Bytes
      raise ArgumentError.new("state outside captured prefix") unless 0 <= state <= PREFIX_LAYERS
      state_bytes = @tokens.size * @config.dim * 2
      offset = state * state_bytes
      Bytes.new(state_bytes) { |index| @hidden_state_snapshot[offset + index] }
    end

    # Read all unique rows once in sorted token-ID order, authenticate every row
    # against the pinned manifest, and return those exact packed BF16 bytes.
    def verify_embedding_rows(weights : ML::GGUF::Qwen25VLWeights) : Bytes
      rows = @document["capture"]["embedding_unique_rows"].as_a
      ids = @tokens.uniq.sort
      raise ArgumentError.new("fixture embedding row inventory mismatch") unless rows.map { |row| row["row"].as_i } == ids
      row_bytes = @config.dim * 2
      raw_rows = weights.embedding_rows_raw(ids)
      raise ArgumentError.new("fixture embedding row byte count mismatch") unless raw_rows.size == ids.size * row_bytes
      rows.each_with_index do |row, index|
        digest = Digest::SHA256.hexdigest(raw_rows[index * row_bytes, row_bytes])
        raise ArgumentError.new("fixture embedding row source mismatch") unless row["key"].as_s == ML::GGUF::Qwen25VLWeights::EMBEDDING_NAME && row["dtype"].as_s == "BF16" && row["tensor_sha256"].as_s == digest
      end
      raw_rows
    end

    def compare(buffer : ML::MetalBuffer, state : Int32) : Comparison
      raise ArgumentError.new("state outside captured prefix") unless 0 <= state <= PREFIX_LAYERS
      size = @tokens.size * @config.dim
      raise ArgumentError.new("hidden buffer size mismatch") unless buffer.size == size.to_i64 * 2
      expected = expected_state(state)
      actual = Bytes.new(size * 2)
      buffer.read_bytes(actual.to_unsafe, actual.size)
      comparison = Qwen25PrefixProbe.compare_bf16_bytes(expected, actual)
      {state: state, elements: size, failures: comparison[:failures], max_abs: comparison[:max_abs],
       different_bf16_words: comparison[:different_bf16_words]}
    end

    private def read_sidecar_snapshot(expected_size : Int64) : Bytes
      bytes = Bytes.new(expected_size.to_i)
      begin
        File.open(@sidecar) do |io|
          io.read_fully(bytes)
          raise ArgumentError.new("fixture sidecar grew while being read") unless io.read_byte.nil?
        end
      rescue error : IO::Error
        raise ArgumentError.new("fixture sidecar changed or was truncated while being read: #{error.message}")
      end
      bytes
    end
  end
end
