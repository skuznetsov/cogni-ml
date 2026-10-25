require "option_parser"
require "json"
require "digest/sha256"
require "set"
require "./support/qwen25_prefix_probe"
require "../src/ml/gguf/qwen25_vl_weights"

module ML::GGUF
  {% unless flag?(:cpu_only) %}
    lib Qwen25MPSFFI
      fun create = gs_qwen25_mps_block_plan_create(
        device : Void*, batches : Int32, tokens : Int32, dim : Int32, ffn : Int32,
        heads : Int32, kv_heads : Int32, head_dim : Int32, rms_norm_eps : Float32,
        input : Void*, mask : Void*, positions : Void*, inv_freq : Void*,
        weights : Void**, stages : Void**, scratch : Void**,
        error : UInt8*, error_capacity : LibC::SizeT,
      ) : Void*
      fun encode = gs_qwen25_mps_block_encode(
        plan : Void*, chain : Void*, error : UInt8*, error_capacity : LibC::SizeT,
      ) : Int32
      fun destroy = gs_qwen25_mps_block_plan_destroy(plan : Void*) : Void
      fun rms_trace_create = gs_qwen25_mps_rms_trace_create(
        device : Void*, batches : Int32, tokens : Int32, dim : Int32,
        epsilon : Float32, input : Void*, outputs : Void**,
        error : UInt8*, error_capacity : LibC::SizeT,
      ) : Void*
      fun rms_precise_trace_create = gs_qwen25_mps_rms_precise_trace_create(
        device : Void*, batches : Int32, tokens : Int32, dim : Int32,
        epsilon : Float32, input : Void*, outputs : Void**,
        error : UInt8*, error_capacity : LibC::SizeT,
      ) : Void*
    end
  {% end %}

  # Exposed only by the one-block diagnostic; the production API does not
  # otherwise expose this intermediate scratch binding.
  class Qwen25VLBlockPlan
    def diagnostic_input_normalized : MetalBuffer
      check_live!
      @scratch["input_normalized"]
    end

    def diagnostic_rms_normalized(scratch_name : String) : MetalBuffer
      check_live!
      unless scratch_name == "input_normalized" || scratch_name == "post_normalized"
        raise ArgumentError.new("unknown diagnostic RMS normalized scratch: #{scratch_name}")
      end
      @scratch[scratch_name]
    end

    def diagnostic_rms_statistics : MetalBuffer
      check_live!
      @scratch["rms_statistics"]
    end
  end
end

# Count named Metal buffers allocated and released during encoding.
class ML::MetalBuffer
  @@qwen25_block_probe_allocations = 0_i64

  def self.qwen25_block_probe_allocations : Int64
    @@qwen25_block_probe_allocations
  end

  private def track_alloc(bytes : Int64) : Nil
    @@qwen25_block_probe_allocations += 1
    previous_def(bytes)
  end
end

module Qwen25BlockProbe
  extend self

  MAX_STAGE_SIDECAR_BYTES = 512_i64 * 1024 * 1024
  RMS_SOURCE_SHA256       = "08742a7c82a8713bc22f632f01c97b7eca3ef0e947180e7d6c9beec4897c4ff8"
  RMS_CAPTURE_METHOD      = "TorchDispatchMode pass-through around the official Qwen2RMSNorm.forward call"
  RMS_SITES               = {"input" => "input_layernorm", "post-attention" => "post_attention_layernorm"}
  WEIGHT_SUFFIXES         = {
    "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight",
    "self_attn.v_proj.bias", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
  }

  alias Extent = NamedTuple(offset: Int64, nbytes: Int64)
  alias Sample = NamedTuple(index: Int64, actual: String, reference: String)
  alias Comparison = NamedTuple(name: String, elements: Int64, failures: Int64,
    max_abs: Float64, first_failures: Array(Sample), different_bf16_words: Int64,
    first_differences: Array(Sample))
  alias RMSComparison = NamedTuple(name: String, dtype: String, elements: Int64,
    failures: Int64, max_abs: Float64, first_failures: Array(Sample),
    different_elements: Int64, first_differences: Array(Sample),
    actual_sha256: String, reference_sha256: String)

  def rms_trace_buffer_bytes(config : ML::GGUF::Qwen25VLConfig, tokens : Int32) : Array(Int64)
    scalar_bytes = tokens.to_i64 * 4
    hidden_f32_bytes = tokens.to_i64 * config.dim * 4
    hidden_bf16_bytes = tokens.to_i64 * config.dim * 2
    [scalar_bytes, scalar_bytes, scalar_bytes, hidden_f32_bytes, hidden_bf16_bytes]
  end

  def canonical_rms_site(site_argument : String) : String
    RMS_SITES[site_argument]? || raise ArgumentError.new("RMS site must be input or post-attention")
  end

  def rms_stage_names(layer_index : Int32, canonical_site : String) : Array(String)
    unless canonical_site == "input_layernorm" || canonical_site == "post_attention_layernorm"
      raise ArgumentError.new("unknown canonical RMS site")
    end
    prefix = "layers.#{layer_index}.#{canonical_site}."
    ["#{prefix}variance_f32", "#{prefix}variance_plus_eps_f32",
     "#{prefix}rsqrt_f32", "#{prefix}normalized_f32", "#{prefix}normalized_bf16"]
  end

  def rms_input_source(canonical_site : String) : String
    canonical_site == "input_layernorm" ? "prefix_input" : "residual_after_attention"
  end

  def rms_scratch_name(canonical_site : String) : String
    canonical_site == "input_layernorm" ? "input_normalized" : "post_normalized"
  end

  def stage_bindings(layer_index : Int32) : Array(Tuple(String, String))
    [
      {"rotary_cos", "rotary_cos"},
      {"rotary_sin", "rotary_sin"},
      {"layers.#{layer_index}.input_layernorm", "input_norm"},
      {"layers.#{layer_index}.self_attn.q_proj", "q_proj"},
      {"layers.#{layer_index}.self_attn.k_proj", "k_proj"},
      {"layers.#{layer_index}.self_attn.v_proj", "v_proj"},
      {"post_rope_q", "q_rope"},
      {"post_rope_k", "k_rope"},
      {"attended", "attended"},
      {"layers.#{layer_index}.self_attn.o_proj", "o_proj"},
      {"residual_after_attention", "residual"},
      {"layers.#{layer_index}.post_attention_layernorm", "post_norm"},
      {"layers.#{layer_index}.mlp.gate_proj", "gate_proj"},
      {"layers.#{layer_index}.mlp.up_proj", "up_proj"},
      {"silu_product", "silu_product"},
      {"layers.#{layer_index}.mlp.down_proj", "down_proj"},
      {"hidden_states_out", "hidden_out"},
    ] of Tuple(String, String)
  end

  class StageFixture
    getter document : JSON::Any
    getter manifest_path : String
    getter manifest_sha256 : String
    getter sidecar_path : String
    getter sidecar : Bytes
    getter f32_sidecar_path : String
    getter f32_sidecar : Bytes
    getter layer_index : Int32
    getter rms_site : String
    getter rms_stage_names : Array(String)

    @extents = {} of String => Extent
    @rms_f32_extents = {} of String => Extent

    def initialize(path : String, trusted_sha256 : String,
                   prefix : Qwen25PrefixProbe::Fixture, requested_layer_index : Int32,
                   requested_rms_site : String = "input")
      unless 0 <= requested_layer_index < prefix.config.layers &&
             requested_layer_index + 1 <= Qwen25PrefixProbe::PREFIX_LAYERS
        raise ArgumentError.new("diagnostic layer must be captured by the authenticated prefix")
      end
      @rms_site = Qwen25BlockProbe.canonical_rms_site(requested_rms_site)
      @rms_stage_names = Qwen25BlockProbe.rms_stage_names(requested_layer_index, @rms_site)
      @manifest_path = File.realpath(path)
      manifest_bytes = File.read(@manifest_path)
      @manifest_sha256 = Digest::SHA256.hexdigest(manifest_bytes)
      unless trusted_sha256.matches?(/\A[0-9a-f]{64}\z/) && trusted_sha256 == @manifest_sha256
        raise ArgumentError.new("trusted stage-reference digest mismatch")
      end
      @document = JSON.parse(manifest_bytes)
      unless @document["schema"].as_s == "cogni_ml.qwen25_vl_pretrained_decoder_block_stage_reference" &&
             @document["schema_version"].as_i == 1
        raise ArgumentError.new("unexpected block stage-reference schema")
      end

      @layer_index = @document["layer"]["index"].as_i
      raise ArgumentError.new("stage reference layer differs from requested decoder layer") unless @layer_index == requested_layer_index
      validate_prefix_link!(prefix)
      validate_checkpoint_link!(prefix)
      validate_config_and_inputs!(prefix)
      validate_layer_weights!(prefix)
      validate_stage_records!(prefix.config, prefix.tokens.size)
      validate_rms_trace!(prefix.config, prefix.tokens.size)

      descriptors = @document["sidecars"]
      unless descriptors.as_h.keys.sort == ["bf16", "f32"]
        raise ArgumentError.new("unexpected stage-reference sidecar inventory")
      end
      bf16_descriptor = descriptors["bf16"]
      f32_descriptor = descriptors["f32"]
      @sidecar_path = Qwen25PrefixProbe.sibling(File.dirname(@manifest_path), bf16_descriptor["path"].as_s)
      @f32_sidecar_path = Qwen25PrefixProbe.sibling(File.dirname(@manifest_path), f32_descriptor["path"].as_s)
      bf16_size = bf16_descriptor["nbytes"].as_i64
      f32_size = f32_descriptor["nbytes"].as_i64
      unless bf16_descriptor["encoding"].as_s == "bfloat16-le" &&
             bf16_size > 0 && bf16_size <= MAX_STAGE_SIDECAR_BYTES &&
             bf16_size <= Int32::MAX && File.size(@sidecar_path) == bf16_size
        raise ArgumentError.new("stage-reference BF16 sidecar size or encoding mismatch")
      end
      unless f32_descriptor["encoding"].as_s == "float32-le" &&
             f32_descriptor["byte_order"].as_s == "little" &&
             f32_descriptor["storage_dtype"].as_s == "float32" &&
             f32_size > 0 && f32_size <= MAX_STAGE_SIDECAR_BYTES &&
             f32_size <= Int32::MAX && File.size(@f32_sidecar_path) == f32_size
        raise ArgumentError.new("RMS trace F32 sidecar size or encoding mismatch")
      end
      validate_extents!(@extents, bf16_size)
      validate_extents!(@rms_f32_extents, f32_size)
      @sidecar = read_sidecar_snapshot(@sidecar_path, bf16_size, "BF16")
      @f32_sidecar = read_sidecar_snapshot(@f32_sidecar_path, f32_size, "F32")
      unless bf16_descriptor["sha256"].as_s == Digest::SHA256.hexdigest(@sidecar)
        raise ArgumentError.new("stage-reference BF16 sidecar digest mismatch")
      end
      unless f32_descriptor["sha256"].as_s == Digest::SHA256.hexdigest(@f32_sidecar)
        raise ArgumentError.new("RMS trace F32 sidecar digest mismatch")
      end
      validate_input_state!(prefix)
      validate_rms_input_state!(prefix)
    end

    def expected_stage(name : String) : Bytes
      extent = @extents[name]?
      raise ArgumentError.new("stage reference is missing #{name}") unless extent
      Bytes.new(extent[:nbytes].to_i) do |index|
        @sidecar[extent[:offset].to_i + index]
      end
    end

    def expected_rms_stage(name : String) : Bytes
      if name == @rms_stage_names.last
        return expected_stage("rms:#{name}")
      end
      extent = @rms_f32_extents[name]?
      raise ArgumentError.new("RMS trace reference is missing #{name}") unless extent
      Bytes.new(extent[:nbytes].to_i) do |index|
        @f32_sidecar[extent[:offset].to_i + index]
      end
    end

    private def validate_prefix_link!(prefix : Qwen25PrefixProbe::Fixture) : Nil
      record = @document["prefix_reference"]
      raise ArgumentError.new("stage reference does not authenticate the supplied prefix") unless record["manifest_sha256"].as_s == prefix.manifest_sha256
      prefix_sidecar_sha = prefix.document["sidecars"]["bf16"]["sha256"].as_s
      raise ArgumentError.new("stage reference prefix-sidecar digest mismatch") unless record["sidecar_sha256"].as_s == prefix_sidecar_sha
      unless record["state_indexing"].as_s == "0=token_embedding; i+1=decoder_layer_i_output" &&
             record["input_state_index"].as_i == @layer_index &&
             record["expected_output_state_index"].as_i == @layer_index + 1
        raise ArgumentError.new("stage reference state indexing mismatch")
      end
      input = prefix.expected_state(@layer_index)
      output = prefix.expected_state(@layer_index + 1)
      raise ArgumentError.new("stage reference input-state digest mismatch") unless record["input_state_sha256"].as_s == Digest::SHA256.hexdigest(input)
      raise ArgumentError.new("stage reference output-state digest mismatch") unless record["expected_output_state_sha256"].as_s == Digest::SHA256.hexdigest(output)
    end

    private def validate_checkpoint_link!(prefix : Qwen25PrefixProbe::Fixture) : Nil
      stage_checkpoint = @document["checkpoint"]
      prefix_checkpoint = prefix.document["checkpoint"]
      %w(repo_id revision receipt_sha256 config_sha256 index_sha256).each do |key|
        unless stage_checkpoint[key]?.try(&.to_json) == prefix_checkpoint[key]?.try(&.to_json)
          raise ArgumentError.new("stage checkpoint identity mismatch: #{key}")
        end
      end
      verified = stage_checkpoint["verified_files"].as_h
      unless verified.has_key?("config.json") && verified.has_key?("model.safetensors.index.json")
        raise ArgumentError.new("stage checkpoint file inventory omits config or index")
      end
      prefix_verified = prefix_checkpoint["verified_files"].as_h
      verified.each do |name, info|
        raise ArgumentError.new("stage checkpoint file path mismatch: #{name}") unless info["path"].as_s == name
        source = prefix_verified[name]?
        raise ArgumentError.new("stage checkpoint file is not prefix-authenticated: #{name}") unless source
        %w(path nbytes sha256 hf_lfs_sha256).each do |key|
          unless info[key]?.try(&.to_json) == source[key]?.try(&.to_json)
            raise ArgumentError.new("stage checkpoint file digest mismatch: #{name}")
          end
        end
      end
      shards = @document["layer"]["weights"].as_a.map { |weight| weight["shard"].as_s }.uniq
      expected_names = (["config.json", "model.safetensors.index.json"] + shards).uniq.sort
      unless verified.keys.sort == expected_names
        raise ArgumentError.new("stage checkpoint files are not exact for layer #{@layer_index}")
      end
      prefix_shards = prefix_checkpoint["selected_shards"].as_a.map(&.as_s)
      raise ArgumentError.new("stage references an unselected checkpoint shard") unless shards.all? { |name| prefix_shards.includes?(name) }
    end

    private def validate_config_and_inputs!(prefix : Qwen25PrefixProbe::Fixture) : Nil
      actual = @document["config"]
      expected = prefix.document["config"]
      %w(hidden_size intermediate_size num_attention_heads num_key_value_heads
        num_hidden_layers head_dim max_position_embeddings vocab_size sequence_length
        attention_implementation hidden_act).each do |key|
        unless actual[key]?.try(&.to_json) == expected[key]?.try(&.to_json)
          raise ArgumentError.new("stage config mismatch: #{key}")
        end
      end
      %w(rms_norm_eps rope_theta).each do |key|
        unless actual[key].as_f.to_f32 == expected[key].as_f.to_f32
          raise ArgumentError.new("stage config mismatch: #{key}")
        end
      end
      unless actual["rope_scaling"]["mrope_section"].as_a.map(&.as_i) ==
               expected["rope_scaling"]["mrope_section"].as_a.map(&.as_i)
        raise ArgumentError.new("stage rotary configuration mismatch")
      end
      left = @document["inputs"]
      right = prefix.document["inputs"]
      %w(token_ids attention_mask position_offset position_ids_f32
        position_ids_for_official_model multimodal_position_shape inv_freq_f32).each do |key|
        unless left[key]?.try(&.to_json) == right[key]?.try(&.to_json)
          raise ArgumentError.new("stage inputs mismatch: #{key}")
        end
      end
    end

    private def validate_layer_weights!(prefix : Qwen25PrefixProbe::Fixture) : Nil
      records = @document["layer"]["weights"].as_a
      raise ArgumentError.new("stage layer weight inventory mismatch") unless records.size == WEIGHT_SUFFIXES.size
      prefix_sources = {} of String => JSON::Any
      prefix.document["weight_sources"].as_a.each do |record|
        prefix_sources[record["key"].as_s] = record
      end
      trusted_hashes = prefix.tensor_sha256s
      selected_shapes = ML::GGUF::Qwen25VLWeights.selected_tensor_shapes
      seen = Set(String).new
      records.each do |record|
        key = record["key"].as_s
        prefix_text = "model.layers.#{@layer_index}."
        raise ArgumentError.new("stage weight outside layer #{@layer_index}: #{key}") unless key.starts_with?(prefix_text)
        suffix = key.sub(prefix_text, "")
        raise ArgumentError.new("stage weight name is not selected: #{key}") unless WEIGHT_SUFFIXES.includes?(suffix)
        raise ArgumentError.new("duplicate stage weight: #{key}") if seen.includes?(key)
        seen << key
        source = prefix_sources[key]?
        raise ArgumentError.new("stage weight is not prefix-authenticated: #{key}") unless source
        expected_shape = selected_shapes[key]?
        raise ArgumentError.new("stage weight shape metadata missing: #{key}") unless expected_shape
        expected_bytes = expected_shape.reduce(1_i64) { |count, dimension| count * dimension } * 2
        unless record["dtype"].as_s == "BF16" &&
               record["shape"].as_a.map(&.as_i64) == expected_shape &&
               record["nbytes"].as_i64 == expected_bytes &&
               record["tensor_sha256"].as_s == trusted_hashes[key] &&
               record["tensor_sha256"].as_s == source["tensor_sha256"].as_s &&
               record["shard"].as_s == source["shard"].as_s
          raise ArgumentError.new("stage weight metadata mismatch: #{key}")
        end
      end
      expected_names = WEIGHT_SUFFIXES.map { |suffix| "model.layers.#{@layer_index}.#{suffix}" }.to_set
      raise ArgumentError.new("stage layer weight inventory is incomplete") unless seen == expected_names
    end

    private def validate_stage_records!(config : ML::GGUF::Qwen25VLConfig, tokens : Int32) : Nil
      shapes = ML::GGUF::Qwen25VLBlockPlan.stage_shapes(config, tokens)
      records = @document["stages"].as_a
      bindings = Qwen25BlockProbe.stage_bindings(@layer_index)
      raise ArgumentError.new("stage capture count mismatch") unless records.size == bindings.size
      @extents = {} of String => Extent
      records.each_with_index do |record, index|
        name, binding = bindings[index]
        raise ArgumentError.new("stage capture order/name mismatch at #{index}") unless record["name"].as_s == name
        raise ArgumentError.new("stage capture dtype mismatch: #{name}") unless record["dtype"].as_s == "bfloat16"
        expected_shape = name.starts_with?("rotary_") ? [tokens, config.head_dim] : shapes[binding]
        unless record["shape"].as_a.map(&.as_i) == expected_shape
          raise ArgumentError.new("stage capture shape mismatch: #{name}")
        end
        expected_bytes = expected_shape.reduce(1_i64) { |count, dimension| count * dimension } * 2
        add_extent!(@extents, name, record["storage"], "bf16", expected_bytes)
      end

      hidden = @document["hidden_states_in"]
      unless hidden["dtype"].as_s == "bfloat16" &&
             hidden["shape"].as_a.map(&.as_i) == [1, tokens, config.dim]
        raise ArgumentError.new("stage input-state shape or dtype mismatch")
      end
      add_extent!(@extents, "hidden_states_in", hidden["storage"], "bf16", tokens.to_i64 * config.dim * 2)

      # The oracle may include this explicit hook output; it is authenticated
      # and bounds-checked, but the native block API exposes only silu_product.
      if extras = @document["extra_stages"]?
        records = extras.as_a
        raise ArgumentError.new("unexpected optional extra-stage count") unless records.size == 1
        record = records.first
        extra_name = "layers.#{@layer_index}.mlp.act_fn"
        unless record["name"].as_s == extra_name &&
               record["dtype"].as_s == "bfloat16" &&
               record["shape"].as_a.map(&.as_i) == [1, tokens, config.ffn_dim]
          raise ArgumentError.new("unsupported optional extra stage")
        end
        add_extent!(@extents, "extra:#{extra_name}", record["storage"], "bf16", tokens.to_i64 * config.ffn_dim * 2)
      end
    end

    private def validate_rms_trace!(config : ML::GGUF::Qwen25VLConfig, tokens : Int32) : Nil
      trace = @document["rms_trace"]
      source = trace["source"]
      unless source["schema_version"].as_i == 1 &&
             source["source_file"].as_s == "models/qwen2/modeling_qwen2.py" &&
             source["source_sha256"].as_s == RMS_SOURCE_SHA256 &&
             source["capture_method"].as_s == RMS_CAPTURE_METHOD
        raise ArgumentError.new("RMS trace source identity mismatch")
      end
      if site = trace["site"]?
        raise ArgumentError.new("RMS trace site mismatch") unless site.as_s == @rms_site
      elsif @rms_site != "input_layernorm"
        raise ArgumentError.new("RMS trace site is required for post-attention diagnostics")
      end
      records = trace["stages"].as_a
      raise ArgumentError.new("RMS trace stage count mismatch") unless records.size == @rms_stage_names.size
      @rms_f32_extents = {} of String => Extent
      records.each_with_index do |record, index|
        name = @rms_stage_names[index]
        raise ArgumentError.new("RMS trace stage order/name mismatch at #{index}") unless record["name"].as_s == name
        expected_bf16 = index == @rms_stage_names.size - 1
        expected_dtype = expected_bf16 ? "bfloat16" : "float32"
        expected_sidecar = expected_bf16 ? "bf16" : "f32"
        expected_shape = if index < 3
                           [1, tokens, 1]
                         else
                           [1, tokens, config.dim]
                         end
        unless record["dtype"].as_s == expected_dtype &&
               record["shape"].as_a.map(&.as_i) == expected_shape
          raise ArgumentError.new("RMS trace dtype or shape mismatch: #{name}")
        end
        element_bytes = expected_bf16 ? 2_i64 : 4_i64
        expected_bytes = expected_shape.reduce(1_i64) { |count, dimension| count * dimension } * element_bytes
        extents = expected_bf16 ? @extents : @rms_f32_extents
        extent_name = expected_bf16 ? "rms:#{name}" : name
        add_extent!(extents, extent_name, record["storage"], expected_sidecar, expected_bytes)
      end
    end

    private def validate_rms_input_state!(prefix : Qwen25PrefixProbe::Fixture) : Nil
      expected = if @rms_site == "input_layernorm"
                   prefix.expected_state(@layer_index)
                 else
                   expected_stage("residual_after_attention")
                 end
      actual_sha256 = Digest::SHA256.hexdigest(expected)
      unless @document["rms_trace"]["input_state_sha256"].as_s == actual_sha256
        raise ArgumentError.new("RMS trace selected-site input-state digest mismatch")
      end
    end

    private def add_extent!(extents : Hash(String, Extent), name : String,
                            storage : JSON::Any, expected_sidecar : String,
                            expected_bytes : Int64) : Nil
      raise ArgumentError.new("stage extent references unexpected sidecar: #{name}") unless storage["sidecar"].as_s == expected_sidecar
      offset = storage["offset_bytes"].as_i64
      nbytes = storage["nbytes"].as_i64
      unless offset >= 0 && nbytes == expected_bytes && nbytes > 0 && offset <= Int64::MAX - nbytes
        raise ArgumentError.new("stage extent byte count invalid: #{name}")
      end
      raise ArgumentError.new("duplicate stage extent: #{name}") if extents.has_key?(name)
      extents[name] = {offset: offset, nbytes: nbytes}
    end

    private def validate_extents!(extents : Hash(String, Extent), sidecar_bytes : Int64) : Nil
      sorted = extents.to_a.sort_by { |_, extent| extent[:offset] }
      cursor = 0_i64
      sorted.each do |name, extent|
        raise ArgumentError.new("stage sidecar extent gap or overlap at #{name}") unless extent[:offset] == cursor
        cursor += extent[:nbytes]
        raise ArgumentError.new("stage sidecar extent exceeds file at #{name}") if cursor > sidecar_bytes
      end
      raise ArgumentError.new("stage sidecar extents do not cover the complete file") unless cursor == sidecar_bytes
    end

    private def validate_input_state!(prefix : Qwen25PrefixProbe::Fixture) : Nil
      actual = expected_stage("hidden_states_in")
      expected = prefix.expected_state(@layer_index)
      raise ArgumentError.new("stage sidecar input state differs from authenticated prefix input") unless actual == expected
    end

    private def read_sidecar_snapshot(path : String, size : Int64, label : String) : Bytes
      bytes = Bytes.new(size.to_i)
      begin
        File.open(path) do |io|
          io.read_fully(bytes)
          raise ArgumentError.new("#{label} stage sidecar grew while being read") unless io.read_byte.nil?
        end
      rescue error : IO::Error
        raise ArgumentError.new("#{label} stage sidecar changed or was truncated while being read: #{error.message}")
      end
      bytes
    end
  end

  def compare(name : String, expected : Bytes, actual : Bytes) : Comparison
    unless expected.size == actual.size && expected.size > 0 && expected.size.even?
      raise ArgumentError.new("BF16 comparison size mismatch for #{name}")
    end
    failures = 0_i64
    different_words = 0_i64
    maximum = 0.0
    failure_samples = [] of Sample
    difference_samples = [] of Sample
    (expected.size // 2).times do |index|
      reference_word = expected[2 * index].to_u32 | (expected[2 * index + 1].to_u32 << 8)
      actual_word = actual[2 * index].to_u32 | (actual[2 * index + 1].to_u32 << 8)
      reference = (reference_word << 16).unsafe_as(Float32).to_f64
      value = (actual_word << 16).unsafe_as(Float32).to_f64
      if reference_word != actual_word
        different_words += 1
        if difference_samples.size < 8
          difference_samples << {index: index.to_i64, actual: value.to_s, reference: reference.to_s}
        end
      end
      difference = (value - reference).abs
      failed = !reference.finite? || !value.finite? ||
               difference > Qwen25PrefixProbe::ATOL + Qwen25PrefixProbe::RTOL * reference.abs
      if !reference.finite? || !value.finite?
        maximum = Float64::MAX
      else
        maximum = Math.max(maximum, difference)
      end
      if failed
        failures += 1
        if failure_samples.size < 8
          actual_text = value.finite? ? value.to_s : (value.nan? ? "NaN" : (value > 0 ? "Infinity" : "-Infinity"))
          reference_text = reference.finite? ? reference.to_s : (reference.nan? ? "NaN" : (reference > 0 ? "Infinity" : "-Infinity"))
          failure_samples << {index: index.to_i64, actual: actual_text, reference: reference_text}
        end
      end
    end
    {name: name, elements: (expected.size // 2).to_i64, failures: failures,
     max_abs: maximum, first_failures: failure_samples,
     different_bf16_words: different_words, first_differences: difference_samples}
  end

  def compare_rms_bf16(name : String, expected : Bytes, actual : Bytes) : RMSComparison
    base = compare(name, expected, actual)
    different = 0_i64
    samples = [] of Sample
    (expected.size // 2).times do |index|
      reference_word = expected[2 * index].to_u32 | (expected[2 * index + 1].to_u32 << 8)
      actual_word = actual[2 * index].to_u32 | (actual[2 * index + 1].to_u32 << 8)
      next if reference_word == actual_word
      different += 1
      if samples.size < 8
        reference = (reference_word << 16).unsafe_as(Float32)
        value = (actual_word << 16).unsafe_as(Float32)
        samples << {index: index.to_i64, actual: scalar_string(value), reference: scalar_string(reference)}
      end
    end
    {name: name, dtype: "bfloat16", elements: base[:elements],
     failures: base[:failures], max_abs: base[:max_abs],
     first_failures: base[:first_failures], different_elements: different,
     first_differences: samples, actual_sha256: Digest::SHA256.hexdigest(actual),
     reference_sha256: Digest::SHA256.hexdigest(expected)}
  end

  def compare_rms_f32(name : String, expected : Bytes, actual : Bytes) : RMSComparison
    unless expected.size == actual.size && expected.size > 0 && expected.size % 4 == 0
      raise ArgumentError.new("F32 comparison size mismatch for #{name}")
    end
    failures = 0_i64
    different = 0_i64
    maximum = 0.0
    failure_samples = [] of Sample
    difference_samples = [] of Sample
    (expected.size // 4).times do |index|
      reference_word = expected[4 * index].to_u32 |
                       (expected[4 * index + 1].to_u32 << 8) |
                       (expected[4 * index + 2].to_u32 << 16) |
                       (expected[4 * index + 3].to_u32 << 24)
      actual_word = actual[4 * index].to_u32 |
                    (actual[4 * index + 1].to_u32 << 8) |
                    (actual[4 * index + 2].to_u32 << 16) |
                    (actual[4 * index + 3].to_u32 << 24)
      reference = reference_word.unsafe_as(Float32)
      value = actual_word.unsafe_as(Float32)
      if reference_word != actual_word
        different += 1
        if difference_samples.size < 8
          difference_samples << {index: index.to_i64, actual: scalar_string(value), reference: scalar_string(reference)}
        end
      end
      delta = (value.to_f64 - reference.to_f64).abs
      failed = !reference.finite? || !value.finite? ||
               delta > Qwen25PrefixProbe::ATOL + Qwen25PrefixProbe::RTOL * reference.to_f64.abs
      if !reference.finite? || !value.finite?
        maximum = Float64::MAX
      else
        maximum = Math.max(maximum, delta)
      end
      if failed
        failures += 1
        if failure_samples.size < 8
          failure_samples << {index: index.to_i64, actual: scalar_string(value), reference: scalar_string(reference)}
        end
      end
    end
    {name: name, dtype: "float32", elements: (expected.size // 4).to_i64,
     failures: failures, max_abs: maximum, first_failures: failure_samples,
     different_elements: different, first_differences: difference_samples,
     actual_sha256: Digest::SHA256.hexdigest(actual),
     reference_sha256: Digest::SHA256.hexdigest(expected)}
  end

  private def scalar_string(value : Float32) : String
    return "NaN" if value.nan?
    return value > 0 ? "Infinity" : "-Infinity" unless value.finite?
    value.to_s
  end

  def read_buffer(buffer : ML::MetalBuffer) : Bytes
    raise ArgumentError.new("invalid stage Metal buffer") unless buffer.valid?
    bytes = Bytes.new(buffer.size.to_i)
    buffer.read_bytes(bytes.to_unsafe, bytes.size)
    bytes
  end
end

checkpoint_root = ""
prefix_reference = ""
prefix_reference_sha256 = ""
stage_reference = ""
stage_reference_sha256 = ""
report_path = ""
layer_index = -1_i32
rms_site_argument = "input"
max_named_bytes = 0_i64
OptionParser.parse do |parser|
  parser.banner = "One-block pretrained Qwen2.5-VL stage discriminator; use the external memory guard."
  parser.on("--checkpoint-root PATH", "Pinned local safetensors checkpoint") { |value| checkpoint_root = value }
  parser.on("--prefix-reference PATH", "Authenticated upstream prefix manifest") { |value| prefix_reference = value }
  parser.on("--prefix-reference-sha256 SHA256", "Trusted upstream prefix-manifest digest") { |value| prefix_reference_sha256 = value }
  parser.on("--stage-reference PATH", "Authenticated layer stage manifest") { |value| stage_reference = value }
  parser.on("--stage-reference-sha256 SHA256", "Trusted layer stage-manifest digest") { |value| stage_reference_sha256 = value }
  parser.on("--layer-index N", "Captured decoder layer index (0...26)") { |value| layer_index = value.to_i32 }
  parser.on("--rms-site SITE", "RMS site: input (default) or post-attention") { |value| rms_site_argument = value }
  parser.on("--report PATH", "New native comparison report") { |value| report_path = value }
  parser.on("--max-named-bytes N", "Aggregate named-buffer admission cap") { |value| max_named_bytes = value.to_i64 }
end
if checkpoint_root.empty? || prefix_reference.empty? || prefix_reference_sha256.empty? ||
   stage_reference.empty? || stage_reference_sha256.empty? || layer_index < 0 ||
   report_path.empty? || max_named_bytes <= 0
  raise ArgumentError.new("checkpoint, both references, layer index, report and positive named-buffer budget required")
end
raise ArgumentError.new("diagnostic layer must be captured by prefix states 0...26") unless 0 <= layer_index < Qwen25PrefixProbe::PREFIX_LAYERS
canonical_rms_site = Qwen25BlockProbe.canonical_rms_site(rms_site_argument)
raise ArgumentError.new("refusing to replace native report") if File.exists?(report_path) || File.symlink?(report_path)
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?

fixture = Qwen25PrefixProbe::Fixture.new(prefix_reference, checkpoint_root, prefix_reference_sha256)
stage_fixture = Qwen25BlockProbe::StageFixture.new(
  stage_reference, stage_reference_sha256, fixture, layer_index, rms_site_argument)
config = fixture.config
tokens = fixture.tokens.size
stage_bindings = Qwen25BlockProbe.stage_bindings(layer_index)
rms_stage_names = stage_fixture.rms_stage_names
block_named_bytes = ML::GGUF::Qwen25VLPrefixPlan.named_bytes(config, tokens, 1)
rms_output_bytes = Qwen25BlockProbe.rms_trace_buffer_bytes(config, tokens)
rms_trace_named_bytes = rms_output_bytes.sum
rms_precise_trace_named_bytes = rms_output_bytes.sum
rms_named_bytes = rms_trace_named_bytes + rms_precise_trace_named_bytes
named_bytes = block_named_bytes + rms_named_bytes
raise ArgumentError.new("one-block diagnostic and both RMS traces exceed aggregate named-buffer budget") unless named_bytes <= max_named_bytes
STDERR.puts "admitted #{named_bytes} named bytes for layer #{layer_index}, #{canonical_rms_site}, and both RMS traces, #{tokens} tokens"

weights : ML::GGUF::Qwen25VLWeights? = nil
owned = [] of ML::MetalBuffer
block : ML::GGUF::Qwen25VLBlockPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
rms_trace_native = Pointer(Void).null
rms_trace_error = Bytes.new(1024, 0_u8)
rms_precise_trace_native = Pointer(Void).null
rms_precise_trace_error = Bytes.new(1024, 0_u8)
begin
  weights = ML::GGUF::Qwen25VLWeights.from_directory(checkpoint_root)
  raise ArgumentError.new("checkpoint loader geometry differs from authenticated reference") unless weights.config == config
  expected_layer_bytes = ML::GGUF::Qwen25VLBlockPlan.weight_shapes(config).values.sum do |shape|
    ML::GGUF::Qwen25VLBlockPlan.elements(shape) * 2
  end
  raise ArgumentError.new("selected layer source inventory has unexpected byte count") unless weights.block_bytes(layer_index) == expected_layer_bytes

  input_raw = fixture.expected_state(layer_index)
  input = ML::MetalBuffer.new(input_raw.size.to_i64)
  owned << input
  input.write_bytes(input_raw.to_unsafe, input_raw.size)

  token_count = fixture.tokens.size
  mask_words = Array(UInt16).new(token_count * token_count) do |index|
    query, key = index // token_count, index % token_count
    key <= query && fixture.attention_mask[key] == 1 ? 0_u16 : 0xff7f_u16
  end
  mask = ML::MetalBuffer.new(mask_words.size.to_i64 * 2)
  owned << mask
  mask.write_bytes(mask_words.to_unsafe.as(UInt8*), mask_words.size * 2)
  positions = ML::MetalBuffer.new(token_count.to_i64 * 4)
  owned << positions
  positions.write(fixture.positions)
  inv_freq = ML::MetalBuffer.new(fixture.inv_freq.size.to_i64 * 4)
  owned << inv_freq
  inv_freq.write(fixture.inv_freq)

  layer_weights = weights.upload_block_weights(layer_index, fixture.tensor_sha256s)
  layer_weights.each_value { |buffer| owned << buffer }
  block = ML::GGUF::Qwen25VLBlockPlan.new(config, token_count, input, mask,
    positions, inv_freq, layer_weights, layer_index)
  rms_input = if canonical_rms_site == "input_layernorm"
                input
              else
                block.not_nil!.stage("residual")
              end

  rms_trace_outputs = [] of ML::MetalBuffer
  rms_output_bytes.each do |byte_size|
    buffer = ML::MetalBuffer.new(byte_size)
    rms_trace_outputs << buffer
    owned << buffer
  end
  rms_trace_handles = rms_trace_outputs.map(&.handle)
  rms_precise_trace_outputs = [] of ML::MetalBuffer
  rms_output_bytes.each do |byte_size|
    buffer = ML::MetalBuffer.new(byte_size)
    rms_precise_trace_outputs << buffer
    owned << buffer
  end
  rms_precise_trace_handles = rms_precise_trace_outputs.map(&.handle)
  {% unless flag?(:cpu_only) %}
    rms_trace_native = ML::GGUF::Qwen25MPSFFI.rms_trace_create(
      ML::Metal::Device.instance.device_handle, 1, token_count, config.dim,
      config.rms_norm_eps, rms_input.handle, rms_trace_handles.to_unsafe,
      rms_trace_error.to_unsafe, rms_trace_error.size)
    raise "Qwen2.5-VL RMS trace preparation failed: #{String.new(rms_trace_error.to_unsafe)}" if rms_trace_native.null?
    rms_precise_trace_native = ML::GGUF::Qwen25MPSFFI.rms_precise_trace_create(
      ML::Metal::Device.instance.device_handle, 1, token_count, config.dim,
      config.rms_norm_eps, rms_input.handle, rms_precise_trace_handles.to_unsafe,
      rms_precise_trace_error.to_unsafe, rms_precise_trace_error.size)
    raise "Qwen2.5-VL precise RMS trace preparation failed: #{String.new(rms_precise_trace_error.to_unsafe)}" if rms_precise_trace_native.null?
  {% else %}
    raise "Metal disabled (cpu_only)"
  {% end %}

  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.qwen25_block_probe_allocations
  encode_started = Time.instant
  block.not_nil!.encode(chain) if canonical_rms_site == "post_attention_layernorm"
  {% unless flag?(:cpu_only) %}
    chain.pause!
    rms_trace_status = ML::GGUF::Qwen25MPSFFI.encode(
      rms_trace_native, chain.handle, rms_trace_error.to_unsafe, rms_trace_error.size)
    chain.resume!
    raise "Qwen2.5-VL RMS trace encode failed (#{rms_trace_status}): #{String.new(rms_trace_error.to_unsafe)}" unless rms_trace_status == 0
    chain.pause!
    rms_precise_trace_status = ML::GGUF::Qwen25MPSFFI.encode(
      rms_precise_trace_native, chain.handle,
      rms_precise_trace_error.to_unsafe, rms_precise_trace_error.size)
    chain.resume!
    raise "Qwen2.5-VL precise RMS trace encode failed (#{rms_precise_trace_status}): #{String.new(rms_precise_trace_error.to_unsafe)}" unless rms_precise_trace_status == 0
  {% else %}
    raise "Metal disabled (cpu_only)"
  {% end %}
  block.not_nil!.encode(chain) if canonical_rms_site == "input_layernorm"
  encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.qwen25_block_probe_allocations - allocations_before
  raise "encoding allocated named Metal buffers" unless encode_allocations == 0
  raise "encoding introduced a device wait" unless chain.wait_count == 0
  wait_started = Time.instant
  chain.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  raise "expected exactly one terminal device wait" unless chain.wait_count == 1 && chain.completed?

  # Every activation readback below occurs after the sole terminal wait.
  comparisons = [] of Qwen25BlockProbe::Comparison
  stage_outputs = {} of String => Bytes
  stage_bindings.each do |reference_name, binding|
    expected = stage_fixture.expected_stage(reference_name)
    actual = Qwen25BlockProbe.read_buffer(block.stage(binding))
    stage_outputs[binding] = actual
    comparisons << Qwen25BlockProbe.compare(reference_name, expected, actual)
  end
  output_expected = fixture.expected_state(layer_index + 1)
  # hidden_out is already the final stage binding, so reuse its one readback.
  output_actual = stage_outputs["hidden_out"]
  output_comparison = Qwen25BlockProbe.compare("prefix_expected_state_#{layer_index + 1}", output_expected, output_actual)
  failures = comparisons.select { |comparison| comparison[:failures] > 0 }
  output_passed = output_comparison[:failures] == 0

  rms_comparisons = [] of Qwen25BlockProbe::RMSComparison
  rms_trace_actuals = {} of String => Bytes
  rms_stage_names.each_with_index do |name, index|
    expected = stage_fixture.expected_rms_stage(name)
    actual = Qwen25BlockProbe.read_buffer(rms_trace_outputs[index])
    rms_trace_actuals[name] = actual
    comparison = if index == rms_stage_names.size - 1
                   Qwen25BlockProbe.compare_rms_bf16(name, expected, actual)
                 else
                   Qwen25BlockProbe.compare_rms_f32(name, expected, actual)
                 end
    rms_comparisons << comparison
  end
  scratch_name = Qwen25BlockProbe.rms_scratch_name(canonical_rms_site)
  scratch_actual = Qwen25BlockProbe.read_buffer(block.diagnostic_rms_normalized(scratch_name))
  legacy_scratch_comparison = Qwen25BlockProbe.compare_rms_bf16(
    "block_#{canonical_rms_site}_normalized_vs_legacy_graph_trace",
    rms_trace_actuals[rms_stage_names.last], scratch_actual)
  legacy_rms_passed = rms_comparisons.all? { |comparison| comparison[:failures] == 0 }

  rms_precise_comparisons = [] of Qwen25BlockProbe::RMSComparison
  rms_precise_trace_actuals = [] of Bytes
  precise_normalized_actual = Bytes.empty
  rms_stage_names.each_with_index do |name, index|
    expected = stage_fixture.expected_rms_stage(name)
    actual = Qwen25BlockProbe.read_buffer(rms_precise_trace_outputs[index])
    rms_precise_trace_actuals << actual
    precise_normalized_actual = actual if index == rms_stage_names.size - 1
    comparison = if index == rms_stage_names.size - 1
                   Qwen25BlockProbe.compare_rms_bf16(name, expected, actual)
                 else
                   Qwen25BlockProbe.compare_rms_f32(name, expected, actual)
                 end
    rms_precise_comparisons << comparison
  end
  precise_normalized_bf16_exact = rms_precise_comparisons.last[:different_elements] == 0
  rms_precise_passed = rms_precise_comparisons.all? { |comparison| comparison[:failures] == 0 } &&
                       precise_normalized_bf16_exact
  scratch_comparison = Qwen25BlockProbe.compare_rms_bf16(
    "block_#{canonical_rms_site}_normalized_vs_precise_trace", precise_normalized_actual, scratch_actual)
  scratch_bitwise_equal = scratch_comparison[:different_elements] == 0
  production_root_comparison : Qwen25BlockProbe::RMSComparison? = nil
  production_root_bitwise_equal = true
  if canonical_rms_site == "post_attention_layernorm"
    production_root_actual = Qwen25BlockProbe.read_buffer(block.diagnostic_rms_statistics)
    production_root_comparison = Qwen25BlockProbe.compare_rms_f32(
      "production_post_attention_inverse_root_vs_precise_trace_rsqrt_f32",
      rms_precise_trace_actuals[2], production_root_actual)
    production_root_bitwise_equal = production_root_comparison.not_nil![:different_elements] == 0
  end
  rms_passed = rms_precise_passed && scratch_bitwise_equal && production_root_bitwise_equal

  File.write(report_path, JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "cogni_ml.qwen25_vl_pretrained_decoder_block_native_comparison"
      json.field "schema_version", 1
      json.field "checkpoint_revision", ML::GGUF::Qwen25VLWeights::EXPECTED_MODEL_REVISION
      json.field "prefix_reference", fixture.manifest_path
      json.field "prefix_reference_sha256", fixture.manifest_sha256
      json.field "stage_reference", stage_fixture.manifest_path
      json.field "stage_reference_sha256", stage_fixture.manifest_sha256
      json.field "layer_index", layer_index
      json.field "expected_output_state_index", layer_index + 1
      json.field "rms_site", canonical_rms_site
      json.field "rms_input_source", Qwen25BlockProbe.rms_input_source(canonical_rms_site)
      json.field "tokens", tokens
      json.field "named_bytes", named_bytes
      json.field "block_named_bytes", block_named_bytes
      json.field "rms_trace_named_bytes", rms_trace_named_bytes
      json.field "rms_precise_trace_named_bytes", rms_precise_trace_named_bytes
      json.field "named_allocations_during_encode", encode_allocations
      json.field "terminal_wait_count", chain.wait_count
      json.field "diagnostic_readbacks_after_terminal_wait",
        comparisons.size + rms_comparisons.size + rms_precise_comparisons.size + 1 + (production_root_comparison ? 1 : 0)
      json.field "activation_readbacks_before_terminal_wait", 0
      json.field "encode_seconds", encode_seconds
      json.field "terminal_wait_seconds", wait_seconds
      json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
      json.field "command_segments", chain.segment_count
      json.field "tolerance", {atol: Qwen25PrefixProbe::ATOL, rtol: Qwen25PrefixProbe::RTOL}
      json.field "stage_comparisons", comparisons
      json.field "prefix_output_comparison", output_comparison
      json.field "rms_trace_source_sha256", Qwen25BlockProbe::RMS_SOURCE_SHA256
      json.field "rms_trace_input_state_sha256", stage_fixture.document["rms_trace"]["input_state_sha256"].as_s
      json.field "rms_trace_comparisons", rms_comparisons
      json.field "rms_trace_role", "legacy_graph_negative_control_not_a_production_gate"
      json.field "legacy_rms_trace_passed", legacy_rms_passed
      json.field "legacy_rms_block_scratch_comparison", legacy_scratch_comparison
      json.field "rms_block_scratch_comparison", scratch_comparison
      json.field "rms_block_scratch_bitwise_equal", scratch_bitwise_equal
      json.field "production_root_vs_trace_checked", !production_root_comparison.nil?
      json.field "production_root_vs_trace_bitwise_equal", production_root_bitwise_equal if production_root_comparison
      json.field "production_root_vs_trace_comparison", production_root_comparison.not_nil! if production_root_comparison
      json.field "rms_trace_passed", rms_passed
      json.field "rms_precise_trace_comparisons", rms_precise_comparisons
      json.field "rms_precise_normalized_bf16_exact_official", precise_normalized_bf16_exact
      json.field "rms_precise_trace_passed", rms_precise_passed
      json.field "first_mismatch_stage", failures.first?.try(&.[:name])
      json.field "all_stages_passed", failures.empty?
      json.field "standalone_output_matches_prefix_state", output_passed
      json.field "authenticated_extra_stages_not_native_bindings",
        stage_fixture.document["extra_stages"]?.try(&.as_a.map { |record| record["name"].as_s }) || [] of String
      json.field "passed", failures.empty? && output_passed && rms_passed
    end
  end)
  comparisons.each { |comparison| puts comparison.to_json }
  puts output_comparison.to_json
  rms_comparisons.each { |comparison| puts comparison.to_json }
  puts scratch_comparison.to_json
  puts production_root_comparison.not_nil!.to_json if production_root_comparison
  puts legacy_scratch_comparison.to_json
  rms_precise_comparisons.each { |comparison| puts comparison.to_json }
  unless failures.empty?
    raise "layer #{layer_index} first mismatch at captured stage #{failures.first[:name]} (#{failures.first[:failures]} elements)"
  end
  raise "standalone layer #{layer_index} output differs from authenticated prefix state #{layer_index + 1}" unless output_passed
  raise "layer #{layer_index} RMS trace or normalized scratch comparison failed" unless rms_passed
  raise "precise layer #{layer_index} RMS trace differs from the authenticated official stages" unless rms_precise_passed
  puts "PASS: all 17 layer-#{layer_index} stages, precise #{canonical_rms_site} trace, production RMS scratch, and standalone output match; legacy trace reported separately; one terminal wait; zero named allocations during encode"
ensure
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    block.try(&.release)
    {% unless flag?(:cpu_only) %}
      ML::GGUF::Qwen25MPSFFI.destroy(rms_precise_trace_native) unless rms_precise_trace_native.null?
      ML::GGUF::Qwen25MPSFFI.destroy(rms_trace_native) unless rms_trace_native.null?
    {% end %}
    chain.try(&.release)
    owned.reverse_each { |buffer| buffer.release if buffer.valid? }
    weights.try(&.close)
  end
end
