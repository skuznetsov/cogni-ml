require "json"
require "set"
require "./safetensors"
require "./qwen3vl_text_block"

module ML::GGUF
  # Validated metadata and bounded embedding access for the expected Qwen-Image
  # 2.1 Qwen3-VL text encoder. Shard metadata is validated before reads;
  # embedding rows and bounded block tensors are copied from their source
  # files, so returned values do not depend on an open shard mapping.
  class Qwen3VLTextWeights
    # Expected checkpoint identity for this package revision; the local
    # safetensors files do not carry independently verifiable HF cache metadata.
    EXPECTED_MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"

    HIDDEN_SIZE         =   4096
    INTERMEDIATE_SIZE   =  12288
    NUM_LAYERS          =     36
    NUM_ATTENTION_HEADS =     32
    NUM_KEY_VALUE_HEADS =      8
    HEAD_DIM            =    128
    VOCAB_SIZE          = 151936
    ROW_BYTES           = HIDDEN_SIZE * 2

    EMBEDDING_NAME = "model.language_model.embed_tokens.weight"

    getter text_encoder_dir : String
    getter vocab_size : Int32
    getter hidden_size : Int32

    @embedding_io : File?
    @embedding_data_offset : Int64
    @tensor_sources : Hash(String, NamedTuple(path: String, data_offset: Int64, byte_size: Int64))
    @closed : Bool
    @mutex : Mutex

    def self.from_directory(text_encoder_dir : String) : self
      new(text_encoder_dir)
    end

    # The text model has 36 blocks × 11 tensors plus its token embedding and
    # final RMSNorm. The untied LM head is intentionally outside this encoder
    # inventory.
    def self.required_tensor_names : Array(String)
      required_tensor_shapes.keys
    end

    # Safetensors matrices use PyTorch [out, in] order. The token table is
    # [vocab, hidden], while all projection tensors are [out, in].
    def self.required_tensor_shapes : Hash(String, Array(Int64))
      hidden = HIDDEN_SIZE.to_i64
      intermediate = INTERMEDIATE_SIZE.to_i64
      query_dim = (NUM_ATTENTION_HEADS * HEAD_DIM).to_i64
      key_value_dim = (NUM_KEY_VALUE_HEADS * HEAD_DIM).to_i64

      shapes = {
        EMBEDDING_NAME                     => [VOCAB_SIZE.to_i64, hidden],
        "model.language_model.norm.weight" => [hidden],
      } of String => Array(Int64)

      NUM_LAYERS.times do |layer|
        prefix = "model.language_model.layers.#{layer}."
        shapes[prefix + "input_layernorm.weight"] = [hidden]
        shapes[prefix + "self_attn.q_proj.weight"] = [query_dim, hidden]
        shapes[prefix + "self_attn.k_proj.weight"] = [key_value_dim, hidden]
        shapes[prefix + "self_attn.v_proj.weight"] = [key_value_dim, hidden]
        shapes[prefix + "self_attn.o_proj.weight"] = [hidden, query_dim]
        shapes[prefix + "self_attn.q_norm.weight"] = [HEAD_DIM.to_i64]
        shapes[prefix + "self_attn.k_norm.weight"] = [HEAD_DIM.to_i64]
        shapes[prefix + "post_attention_layernorm.weight"] = [hidden]
        shapes[prefix + "mlp.gate_proj.weight"] = [intermediate, hidden]
        shapes[prefix + "mlp.up_proj.weight"] = [intermediate, hidden]
        shapes[prefix + "mlp.down_proj.weight"] = [hidden, intermediate]
      end

      shapes
    end

    def initialize(text_encoder_dir : String)
      @text_encoder_dir = File.realpath(text_encoder_dir)
      raise ArgumentError.new("qwen3vl text weights: text encoder path is not a directory") unless Dir.exists?(@text_encoder_dir)

      @vocab_size = VOCAB_SIZE
      @hidden_size = HIDDEN_SIZE
      @embedding_io = nil
      @embedding_data_offset = -1_i64
      @tensor_sources = {} of String => NamedTuple(path: String, data_offset: Int64, byte_size: Int64)
      @closed = false
      @mutex = Mutex.new

      begin
        validate_config!
        weight_map = read_weight_map!
        validate_required_index_entries!(weight_map)
        shard_paths = resolve_shard_paths!(weight_map)
        open_and_validate_shards!(weight_map, shard_paths)
        set_embedding_source!(weight_map, shard_paths)
      rescue ex
        @embedding_io.try(&.close)
        @embedding_io = nil
        raise ex
      end
    end

    # Return selected token embeddings in their original little-endian BF16
    # representation. Only requested rows are copied out of the shard file.
    def embedding_rows_raw(token_ids : Array(Int32)) : Bytes
      copy_embedding_rows(token_ids.map(&.to_i64))
    end

    # Int64 overload matches the reference fixture's input_ids without forcing
    # callers to narrow before the loader checks token bounds.
    def embedding_rows_raw(token_ids : Array(Int64)) : Bytes
      copy_embedding_rows(token_ids)
    end

    # Return selected token embeddings converted from BF16 to Float32.
    def embedding_rows_f32(token_ids : Array(Int32)) : Array(Float32)
      decode_bf16_rows(embedding_rows_raw(token_ids))
    end

    def embedding_rows_f32(token_ids : Array(Int64)) : Array(Float32)
      decode_bf16_rows(embedding_rows_raw(token_ids))
    end

    # Read one named tensor from decoder layer 0, converted from BF16 to F32.
    # The name must be one of the validated tensors in the layer inventory.
    def layer0_tensor_f32(name : String) : Array(Float32)
      @mutex.synchronize do
        raise ArgumentError.new("qwen3vl text weights: loader is closed") if @closed
        read_layer_tensor_f32_unlocked(name, 0)
      end
    end

    # Load one decoder block's weights in the order consumed by
    # Qwen3VLTextBlock. This expands the pinned block to roughly 0.72 GiB of
    # Float32 arrays, so it is intended for bounded CPU parity probes.
    def block_weights(layer_index : Int32) : Qwen3VLTextBlockWeights
      unless 0 <= layer_index < NUM_LAYERS
        raise ArgumentError.new("qwen3vl text weights: layer index #{layer_index} outside 0...#{NUM_LAYERS}")
      end

      @mutex.synchronize do
        raise ArgumentError.new("qwen3vl text weights: loader is closed") if @closed
        prefix = "model.language_model.layers.#{layer_index}."
        Qwen3VLTextBlockWeights.new(
          input_layernorm: read_layer_tensor_f32_unlocked(prefix + "input_layernorm.weight", layer_index),
          q_proj: read_layer_tensor_f32_unlocked(prefix + "self_attn.q_proj.weight", layer_index),
          k_proj: read_layer_tensor_f32_unlocked(prefix + "self_attn.k_proj.weight", layer_index),
          v_proj: read_layer_tensor_f32_unlocked(prefix + "self_attn.v_proj.weight", layer_index),
          q_norm: read_layer_tensor_f32_unlocked(prefix + "self_attn.q_norm.weight", layer_index),
          k_norm: read_layer_tensor_f32_unlocked(prefix + "self_attn.k_norm.weight", layer_index),
          o_proj: read_layer_tensor_f32_unlocked(prefix + "self_attn.o_proj.weight", layer_index),
          post_attention_layernorm: read_layer_tensor_f32_unlocked(prefix + "post_attention_layernorm.weight", layer_index),
          gate_proj: read_layer_tensor_f32_unlocked(prefix + "mlp.gate_proj.weight", layer_index),
          up_proj: read_layer_tensor_f32_unlocked(prefix + "mlp.up_proj.weight", layer_index),
          down_proj: read_layer_tensor_f32_unlocked(prefix + "mlp.down_proj.weight", layer_index),
        )
      end
    end

    # Returned embedding rows are copies, so closing only stops future reads.
    def close : Nil
      @mutex.synchronize do
        return if @closed
        @embedding_io.try(&.close)
        @embedding_io = nil
        @closed = true
      end
    end

    def closed? : Bool
      @mutex.synchronize { @closed }
    end

    private def copy_embedding_rows(token_ids : Array(Int64)) : Bytes
      token_ids.each do |token_id|
        unless 0_i64 <= token_id < VOCAB_SIZE
          raise ArgumentError.new("qwen3vl text weights: token id #{token_id} outside vocabulary 0...#{VOCAB_SIZE}")
        end
      end

      total_bytes = token_ids.size.to_i64 * ROW_BYTES
      if total_bytes > Int32::MAX
        raise ArgumentError.new("qwen3vl text weights: selected embedding rows exceed addressable slice size")
      end

      output = Bytes.new(total_bytes.to_i)
      @mutex.synchronize do
        raise ArgumentError.new("qwen3vl text weights: loader is closed") if @closed
        return output if token_ids.empty?
        io = @embedding_io || raise ArgumentError.new("qwen3vl text weights: embedding shard is unavailable")

        token_ids.each_with_index do |token_id, index|
          source_offset = @embedding_data_offset + token_id * ROW_BYTES
          destination_offset = index * ROW_BYTES
          io.seek(source_offset)
          io.read_fully(output[destination_offset, ROW_BYTES])
        end
      end

      output
    end

    private def decode_bf16_rows(raw : Bytes) : Array(Float32)
      Array(Float32).new(raw.size // 2) do |index|
        lo = raw[index * 2].to_u32
        hi = raw[index * 2 + 1].to_u32
        ((hi << 24) | (lo << 16)).unsafe_as(Float32)
      end
    end

    private def read_layer_tensor_f32_unlocked(name : String, layer_index : Int32) : Array(Float32)
      prefix = "model.language_model.layers.#{layer_index}."
      expected_shape = self.class.required_tensor_shapes[name]?
      unless name.starts_with?(prefix) && expected_shape
        raise ArgumentError.new("qwen3vl text weights: requested tensor #{name.inspect} is not a required layer-#{layer_index} tensor")
      end

      expected_bytes = expected_shape.reduce(1_i64) { |count, dimension| count * dimension } * 2_i64
      if expected_bytes > Int32::MAX
        raise ArgumentError.new("qwen3vl text weights: tensor #{name} exceeds addressable slice size")
      end

      source = @tensor_sources[name]?
      raise ArgumentError.new("qwen3vl text weights: validated source for #{name} is unavailable") unless source
      unless source[:byte_size] == expected_bytes
        raise ArgumentError.new("qwen3vl text weights: tensor #{name} source size differs from its validated shape")
      end

      source_path = File.realpath(source[:path])
      root_prefix = @text_encoder_dir.ends_with?(File::SEPARATOR) ? @text_encoder_dir : "#{@text_encoder_dir}#{File::SEPARATOR}"
      unless source_path == source[:path] && source_path.starts_with?(root_prefix)
        raise ArgumentError.new("qwen3vl text weights: tensor #{name} shard path escapes text_encoder directory")
      end

      raw = Bytes.new(expected_bytes.to_i)
      File.open(source_path, "rb") do |io|
        file_size = io.size
        data_offset = source[:data_offset]
        unless data_offset >= 0 && data_offset <= file_size && expected_bytes <= file_size - data_offset
          raise ArgumentError.new("qwen3vl text weights: tensor #{name} source outside shard bounds")
        end
        io.seek(data_offset)
        io.read_fully(raw)
      end
      decode_bf16_rows(raw)
    end

    private def validate_config! : Nil
      config_path = File.join(@text_encoder_dir, "config.json")
      config = read_json_object(config_path, "config.json")
      require_string!(config, "model_type", "config.json", "qwen3_vl")
      architectures = config["architectures"]?.try(&.as_a?)
      unless architectures && architectures.size == 1 && architectures[0].as_s? == "Qwen3VLForConditionalGeneration"
        raise ArgumentError.new("qwen3vl text weights: config.json architectures must identify Qwen3VLForConditionalGeneration")
      end

      text_config = config["text_config"]?.try(&.as_h?)
      raise ArgumentError.new("qwen3vl text weights: config.json is missing text_config object") unless text_config

      require_string!(text_config, "model_type", "config.json text_config", "qwen3_vl_text")
      require_string!(text_config, "dtype", "config.json text_config", "bfloat16")
      require_integer!(text_config, "hidden_size", "config.json text_config", HIDDEN_SIZE)
      require_integer!(text_config, "intermediate_size", "config.json text_config", INTERMEDIATE_SIZE)
      require_integer!(text_config, "num_hidden_layers", "config.json text_config", NUM_LAYERS)
      require_integer!(text_config, "num_attention_heads", "config.json text_config", NUM_ATTENTION_HEADS)
      require_integer!(text_config, "num_key_value_heads", "config.json text_config", NUM_KEY_VALUE_HEADS)
      require_integer!(text_config, "head_dim", "config.json text_config", HEAD_DIM)
      require_integer!(text_config, "vocab_size", "config.json text_config", VOCAB_SIZE)
      require_string!(text_config, "hidden_act", "config.json text_config", "silu")
      unless text_config["attention_bias"]?.try(&.as_bool?) == false
        raise ArgumentError.new("qwen3vl text weights: config.json text_config attention_bias must be false")
      end
      require_float!(text_config, "rms_norm_eps", "config.json text_config", 0.000001_f64)
      if rope_theta_value = text_config["rope_theta"]?
        rope_theta = rope_theta_value.as_f? || rope_theta_value.as_i?.try(&.to_f64)
        unless rope_theta == 5_000_000_f64
          raise ArgumentError.new("qwen3vl text weights: config.json text_config rope_theta must be 5000000 when present")
        end
      end

      rope_scaling = text_config["rope_scaling"]?.try(&.as_h?)
      raise ArgumentError.new("qwen3vl text weights: config.json text_config is missing rope_scaling") unless rope_scaling
      unless rope_scaling["mrope_interleaved"]?.try(&.as_bool?) == true
        raise ArgumentError.new("qwen3vl text weights: config.json requires interleaved M-RoPE")
      end
      sections = rope_scaling["mrope_section"]?.try(&.as_a?)
      actual_sections = sections.try(&.map(&.as_i?))
      unless actual_sections == [24, 20, 20]
        raise ArgumentError.new("qwen3vl text weights: config.json M-RoPE sections must be [24, 20, 20]")
      end
    end

    private def read_weight_map! : Hash(String, String)
      index_path = File.join(@text_encoder_dir, "model.safetensors.index.json")
      index = read_json_object(index_path, "model.safetensors.index.json")
      raw_map = index["weight_map"]?.try(&.as_h?)
      raise ArgumentError.new("qwen3vl text weights: index is missing weight_map object") unless raw_map

      weight_map = {} of String => String
      raw_map.each do |name, raw_shard|
        shard = raw_shard.as_s?
        raise ArgumentError.new("qwen3vl text weights: index shard path for #{name} must be a string") unless shard
        weight_map[name] = shard
      end
      weight_map
    end

    private def validate_required_index_entries!(weight_map : Hash(String, String)) : Nil
      missing = self.class.required_tensor_names.reject { |name| weight_map.has_key?(name) }
      return if missing.empty?

      preview = missing.first(8).join(", ")
      raise ArgumentError.new(
        "qwen3vl text weights: missing required text tensors (#{missing.size}); first missing: #{preview}"
      )
    end

    private def resolve_shard_paths!(weight_map : Hash(String, String)) : Hash(String, String)
      resolved = {} of String => String
      root_prefix = @text_encoder_dir.ends_with?(File::SEPARATOR) ? @text_encoder_dir : "#{@text_encoder_dir}#{File::SEPARATOR}"

      weight_map.each_value do |relative_path|
        next if resolved.has_key?(relative_path)
        if relative_path.empty? || relative_path.starts_with?("/") || relative_path.includes?('\\') ||
           relative_path.split('/').any? { |part| part == ".." || part.empty? }
          raise ArgumentError.new("qwen3vl text weights: invalid shard path #{relative_path.inspect}")
        end

        candidate = File.expand_path(relative_path, @text_encoder_dir)
        real_path = begin
          File.realpath(candidate)
        rescue ex
          raise ArgumentError.new("qwen3vl text weights: shard path #{relative_path.inspect} is missing or unreadable: #{ex.message}")
        end
        unless real_path.starts_with?(root_prefix) && File.file?(real_path)
          raise ArgumentError.new("qwen3vl text weights: shard path escapes text_encoder directory: #{relative_path.inspect}")
        end
        resolved[relative_path] = real_path
      end

      resolved
    end

    private def open_and_validate_shards!(
      weight_map : Hash(String, String), shard_paths : Hash(String, String),
    ) : Nil
      shapes = self.class.required_tensor_shapes
      required = Set(String).new(shapes.keys)
      seen = Set(String).new

      shard_paths.each do |shard_name, path|
        preflight_safetensors_header!(path, shard_name)
        reader = begin
          SafetensorsFile.new(path)
        rescue ex
          raise ArgumentError.new("qwen3vl text weights: invalid safetensors shard #{shard_name}: #{ex.message}")
        end

        begin
          validate_shard_bounds!(reader, path, shard_name)
          reader.tensors.each do |info|
            next unless required.includes?(info.name)
            mapped_shard = weight_map[info.name]?
            unless mapped_shard == shard_name
              raise ArgumentError.new("qwen3vl text weights: index shard mapping disagrees for #{info.name}")
            end

            unless info.dtype.bf16?
              raise ArgumentError.new("qwen3vl text weights: #{info.name} must be BF16, got #{info.dtype}")
            end
            expected_shape = shapes[info.name]
            unless info.shape == expected_shape
              raise ArgumentError.new("qwen3vl text weights: #{info.name} shape #{info.shape} != expected #{expected_shape}")
            end
            if info.name == EMBEDDING_NAME
              @embedding_data_offset = reader.data_offset + info.data_start
            end
            @tensor_sources[info.name] = {
              path:        path,
              data_offset: reader.data_offset + info.data_start,
              byte_size:   info.data_bytes,
            }
            seen << info.name
          end
        rescue ex
          raise ex
        ensure
          reader.close
        end
      end

      missing = required.reject { |name| seen.includes?(name) }
      unless missing.empty?
        preview = missing.first(8).join(", ")
        raise ArgumentError.new("qwen3vl text weights: shards are missing required tensors (#{missing.size}); first missing: #{preview}")
      end
    end

    private def preflight_safetensors_header!(path : String, shard_name : String) : Nil
      file_size = File.size(path)
      raise ArgumentError.new("qwen3vl text weights: shard #{shard_name} is shorter than a safetensors header") if file_size < 8

      File.open(path, "rb") do |io|
        prefix = Bytes.new(8)
        io.read_fully(prefix)
        header_size = IO::ByteFormat::LittleEndian.decode(UInt64, prefix)
        unless header_size > 0 && header_size <= (file_size - 8).to_u64 && header_size <= Int32::MAX.to_u64
          raise ArgumentError.new("qwen3vl text weights: shard #{shard_name} has invalid header bounds")
        end
      end
    end

    private def validate_shard_bounds!(reader : SafetensorsFile, path : String, shard_name : String) : Nil
      file_size = File.size(path)
      payload_size = file_size - reader.data_offset
      raise ArgumentError.new("qwen3vl text weights: shard #{shard_name} has invalid payload start") if payload_size < 0

      reader.tensors.each do |info|
        unless info.data_start >= 0 && info.data_end >= info.data_start && info.data_end <= payload_size
          raise ArgumentError.new("qwen3vl text weights: tensor #{info.name} offsets outside shard payload")
        end
      end
    end

    private def set_embedding_source!(
      weight_map : Hash(String, String), shard_paths : Hash(String, String),
    ) : Nil
      shard_name = weight_map[EMBEDDING_NAME]
      path = shard_paths[shard_name]
      raise ArgumentError.new("qwen3vl text weights: token embedding tensor is missing") if @embedding_data_offset < 0
      @embedding_io = File.new(path, "rb")
    end

    private def read_json_object(path : String, label : String) : Hash(String, JSON::Any)
      JSON.parse(File.read(path)).as_h
    rescue ex
      raise ArgumentError.new("qwen3vl text weights: #{label} is missing or malformed: #{ex.message}")
    end

    private def require_string!(
      object : Hash(String, JSON::Any), key : String, label : String, expected : String,
    ) : Nil
      value = object[key]?.try(&.as_s?)
      unless value == expected
        raise ArgumentError.new("qwen3vl text weights: #{label} #{key} must be #{expected.inspect}")
      end
    end

    private def require_integer!(
      object : Hash(String, JSON::Any), key : String, label : String, expected : Int32,
    ) : Nil
      value = object[key]?.try(&.as_i?)
      unless value == expected
        raise ArgumentError.new("qwen3vl text weights: #{label} #{key} must be #{expected}")
      end
    end

    private def require_float!(
      object : Hash(String, JSON::Any), key : String, label : String, expected : Float64,
    ) : Nil
      value = object[key]?.try(&.as_f?)
      unless value == expected
        raise ArgumentError.new("qwen3vl text weights: #{label} #{key} must be #{expected}")
      end
    end
  end
end
