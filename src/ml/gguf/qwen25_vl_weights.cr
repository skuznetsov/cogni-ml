require "json"
require "set"
require "digest/sha256"
require "./safetensors"
require "./qwen25_hy_nomas"

module ML::GGUF
  # Validated native BF16 access to the selected text prefix of the pinned
  # Qwen2.5-VL checkpoint. Metadata is retained for the selected tensors only;
  # weights are streamed into one block's Metal buffers on demand.
  class Qwen25VLWeights
    EXPECTED_MODEL_REVISION = "eb6ba55b40ada64977147b52e50a4145ee89ac4f"

    MODEL_TYPE   = "qwen2_5_vl"
    ARCHITECTURE = "Qwen2_5_VLForConditionalGeneration"
    VOCAB_SIZE   = 152_064

    EMBEDDING_NAME       = "model.embed_tokens.weight"
    SKIPPED_FINAL_LAYERS = 2
    UPLOAD_CHUNK_BYTES   = 1024 * 1024

    BLOCK_SOURCE_SUFFIXES = {
      "input_norm" => "input_layernorm.weight",
      "q_w"        => "self_attn.q_proj.weight",
      "q_bias"     => "self_attn.q_proj.bias",
      "k_w"        => "self_attn.k_proj.weight",
      "k_bias"     => "self_attn.k_proj.bias",
      "v_w"        => "self_attn.v_proj.weight",
      "v_bias"     => "self_attn.v_proj.bias",
      "o_w"        => "self_attn.o_proj.weight",
      "post_norm"  => "post_attention_layernorm.weight",
      "gate_w"     => "mlp.gate_proj.weight",
      "up_w"       => "mlp.up_proj.weight",
      "down_w"     => "mlp.down_proj.weight",
    }

    alias TensorSource = NamedTuple(path: String, data_offset: Int64, byte_size: Int64)

    getter model_dir : String
    getter config : Qwen25VLConfig
    getter vocab_size : Int32
    getter selected_layer_count : Int32
    getter selected_tensor_names : Array(String)
    getter selected_tensor_shapes : Hash(String, Array(Int64))
    getter embedding_table_bytes : Int64
    # Includes all selected block tensors and the full on-disk embedding table.
    getter selected_bytes : Int64
    # Bytes in decoder blocks 0...selected_layer_count; excludes embeddings.
    getter resident_prefix_bytes : Int64

    @tensor_sources : Hash(String, TensorSource)
    @closed : Bool
    @mutex : Mutex

    def self.from_directory(
      model_dir : String,
      config : Qwen25VLConfig = Qwen25VLConfig.hy_nomas_7b,
      vocab_size : Int32 = VOCAB_SIZE,
    ) : self
      new(model_dir, config, vocab_size)
    end

    # The text-only inference prefix is the embedding table plus layers 0..25.
    # The final RMSNorm, LM head, visual tower, and final two decoder blocks are
    # deliberately outside this inventory.
    def self.selected_tensor_shapes(
      config : Qwen25VLConfig = Qwen25VLConfig.hy_nomas_7b,
      vocab_size : Int32 = VOCAB_SIZE,
    ) : Hash(String, Array(Int64))
      raise ArgumentError.new("qwen25-vl weights: vocabulary size must be positive") unless vocab_size > 0
      raise ArgumentError.new("qwen25-vl weights: at least one selected block is required") unless config.layers > SKIPPED_FINAL_LAYERS

      shapes = {
        EMBEDDING_NAME => [vocab_size.to_i64, config.dim.to_i64],
      } of String => Array(Int64)
      block_shapes = Qwen25VLBlockPlan.weight_shapes(config)

      (config.layers - SKIPPED_FINAL_LAYERS).times do |layer_index|
        prefix = "model.layers.#{layer_index}."
        BLOCK_SOURCE_SUFFIXES.each do |binding_name, suffix|
          binding_shape = block_shapes[binding_name]?
          raise ArgumentError.new("qwen25-vl weights: missing block binding shape #{binding_name}") unless binding_shape
          shapes[prefix + suffix] = binding_shape.map(&.to_i64)
        end
      end

      shapes
    end

    def self.selected_tensor_names(
      config : Qwen25VLConfig = Qwen25VLConfig.hy_nomas_7b,
      vocab_size : Int32 = VOCAB_SIZE,
    ) : Array(String)
      selected_tensor_shapes(config, vocab_size).keys
    end

    def self.embedding_table_bytes(
      config : Qwen25VLConfig = Qwen25VLConfig.hy_nomas_7b,
      vocab_size : Int32 = VOCAB_SIZE,
    ) : Int64
      checked_shape_bytes([vocab_size.to_i64, config.dim.to_i64])
    end

    def self.selected_bytes(
      config : Qwen25VLConfig = Qwen25VLConfig.hy_nomas_7b,
      vocab_size : Int32 = VOCAB_SIZE,
    ) : Int64
      sum_shape_bytes(selected_tensor_shapes(config, vocab_size).values)
    end

    def self.resident_prefix_bytes(
      config : Qwen25VLConfig = Qwen25VLConfig.hy_nomas_7b,
      vocab_size : Int32 = VOCAB_SIZE,
    ) : Int64
      shapes = selected_tensor_shapes(config, vocab_size)
        .reject { |name, _| name == EMBEDDING_NAME }
        .values
      sum_shape_bytes(shapes)
    end

    def initialize(model_dir : String, @config : Qwen25VLConfig = Qwen25VLConfig.hy_nomas_7b,
                   @vocab_size : Int32 = VOCAB_SIZE)
      @model_dir = begin
        File.realpath(model_dir)
      rescue ex
        raise ArgumentError.new("qwen25-vl weights: model directory is missing or unreadable: #{ex.message}")
      end
      raise ArgumentError.new("qwen25-vl weights: model path is not a directory") unless Dir.exists?(@model_dir)
      raise ArgumentError.new("qwen25-vl weights: vocabulary size must be positive") unless @vocab_size > 0
      unless @config.layers > SKIPPED_FINAL_LAYERS
        raise ArgumentError.new("qwen25-vl weights: at least one selected block is required")
      end

      @selected_layer_count = @config.layers - SKIPPED_FINAL_LAYERS
      @selected_tensor_shapes = self.class.selected_tensor_shapes(@config, @vocab_size)
      @selected_tensor_names = @selected_tensor_shapes.keys
      @embedding_table_bytes = self.class.embedding_table_bytes(@config, @vocab_size)
      @selected_bytes = self.class.selected_bytes(@config, @vocab_size)
      @resident_prefix_bytes = self.class.resident_prefix_bytes(@config, @vocab_size)
      @tensor_sources = {} of String => TensorSource
      @closed = false
      @mutex = Mutex.new

      validate_config!
      weight_map = read_weight_map!
      shard_paths = resolve_shard_paths!(weight_map)
      validate_selected_shards!(weight_map, shard_paths)
    end

    # Exact byte size of one selected block's 12 BF16 source tensors.
    def block_bytes(layer_index : Int32) : Int64
      layer_prefix!(layer_index)
      prefix = "model.layers.#{layer_index}."
      sum_shape_bytes(@selected_tensor_shapes.select { |name, _| name.starts_with?(prefix) }.values)
    end

    # Return selected embedding rows in their source little-endian BF16 bytes.
    # The result is bounded by the caller's token list, never the full table.
    def embedding_rows_raw(token_ids : Array(Int32)) : Bytes
      copy_embedding_rows(token_ids.map(&.to_i64))
    end

    def embedding_rows_raw(token_ids : Array(Int64)) : Bytes
      copy_embedding_rows(token_ids)
    end

    # Upload only the requested rows to a shared MetalBuffer. The returned
    # buffer contains token rows concatenated in the caller's order.
    def upload_embedding_rows(token_ids : Array(Int32)) : MetalBuffer
      upload_embedding_rows(token_ids.map(&.to_i64))
    end

    def upload_embedding_rows(token_ids : Array(Int64)) : MetalBuffer
      raw = embedding_rows_raw(token_ids)
      raise ArgumentError.new("qwen25-vl weights: at least one embedding row is required for upload") if raw.empty?
      buffer = MetalBuffer.new(raw.size.to_i64)
      begin
        buffer.write_bytes(raw.to_unsafe, raw.size)
        buffer
      rescue ex
        buffer.release if buffer.valid?
        raise ex
      end
    end

    # Stream one decoder block's tensors into fresh shared Metal buffers. The
    # largest temporary CPU allocation is UPLOAD_CHUNK_BYTES; BF16 bytes are
    # copied without conversion or full-tensor staging arrays.
    def upload_block_weights(
      layer_index : Int32,
      expected_tensor_sha256s : Hash(String, String)? = nil,
    ) : Hash(String, MetalBuffer)
      layer_prefix!(layer_index)
      @mutex.synchronize do
        raise ArgumentError.new("qwen25-vl weights: loader is closed") if @closed
        expected_hashes = expected_block_hashes(layer_index, expected_tensor_sha256s)

        result = {} of String => MetalBuffer
        prefix = "model.layers.#{layer_index}."
        begin
          BLOCK_SOURCE_SUFFIXES.each do |binding_name, suffix|
            tensor_name = prefix + suffix
            expected_sha256 = expected_hashes.try(&.[binding_name]?)
            result[binding_name] = stream_tensor_to_buffer_unlocked(tensor_name, expected_sha256)
          end
          result
        rescue ex
          result.each_value { |buffer| buffer.release if buffer.valid? }
          raise ex
        end
      end
    end

    # Testable streaming copy path used by block uploads. Each chunk is hashed
    # from the same CPU bytes copied to the destination, with bounded staging.
    # This performs no Metal work and can falsify same-size source mutations.
    def self.copy_and_verify_sha256(
      io : IO,
      destination : Pointer(UInt8),
      byte_size : Int64,
      expected_sha256 : String,
      tensor_name : String = "tensor",
      chunk_bytes : Int32 = UPLOAD_CHUNK_BYTES,
    ) : String
      unless valid_sha256?(expected_sha256)
        raise ArgumentError.new("qwen25-vl weights: malformed SHA-256 for #{tensor_name}")
      end
      raise ArgumentError.new("qwen25-vl weights: negative byte count for #{tensor_name}") if byte_size < 0
      raise ArgumentError.new("qwen25-vl weights: chunk size must be positive") unless chunk_bytes > 0
      if byte_size > 0 && destination.null?
        raise ArgumentError.new("qwen25-vl weights: null destination for #{tensor_name}")
      end

      digest = Digest::SHA256.new
      staging = Bytes.new(Math.min(chunk_bytes.to_i64, byte_size).to_i)
      offset = 0_i64
      while offset < byte_size
        count = Math.min(staging.size.to_i64, byte_size - offset).to_i
        chunk = staging[0, count]
        io.read_fully(chunk)
        digest.update(chunk)
        (destination + offset.to_i).copy_from(chunk.to_unsafe, count)
        offset += count
      end

      actual = digest.hexfinal
      unless actual == expected_sha256.downcase
        raise ArgumentError.new("qwen25-vl weights: SHA-256 mismatch for #{tensor_name}")
      end
      actual
    end

    def self.valid_sha256?(value : String) : Bool
      return false unless value.bytesize == 64
      value.each_byte.all? do |byte|
        (byte >= '0'.ord && byte <= '9'.ord) ||
          (byte >= 'a'.ord && byte <= 'f'.ord) ||
          (byte >= 'A'.ord && byte <= 'F'.ord)
      end
    end

    private def expected_block_hashes(
      layer_index : Int32,
      supplied : Hash(String, String)?,
    ) : Hash(String, String)?
      return nil unless supplied

      prefix = "model.layers.#{layer_index}."
      hashes = {} of String => String
      BLOCK_SOURCE_SUFFIXES.each do |binding_name, suffix|
        tensor_name = prefix + suffix
        value = supplied[tensor_name]?
        unless value
          raise ArgumentError.new("qwen25-vl weights: expected SHA-256 missing for #{tensor_name}")
        end
        unless self.class.valid_sha256?(value)
          raise ArgumentError.new("qwen25-vl weights: malformed SHA-256 for #{tensor_name}")
        end
        # Own the values used after this call begins; callers may reuse or
        # mutate their full-checkpoint hash map after upload returns.
        hashes[binding_name] = value.downcase.dup
      end
      hashes
    end

    # Closing is a lifecycle guard; files are opened only during bounded reads
    # and uploads, so this does not retain mmap or file-descriptor resources.
    def close : Nil
      @mutex.synchronize { @closed = true }
    end

    def closed? : Bool
      @mutex.synchronize { @closed }
    end

    private def copy_embedding_rows(token_ids : Array(Int64)) : Bytes
      token_ids.each do |token_id|
        unless 0_i64 <= token_id < @vocab_size
          raise ArgumentError.new("qwen25-vl weights: token id #{token_id} outside vocabulary 0...#{@vocab_size}")
        end
      end

      row_bytes = @config.dim.to_i64 * 2_i64
      total_bytes = checked_multiply(token_ids.size.to_i64, row_bytes, "selected embedding rows")
      if total_bytes > Int32::MAX
        raise ArgumentError.new("qwen25-vl weights: selected embedding rows exceed addressable byte size")
      end
      output = Bytes.new(total_bytes.to_i)

      @mutex.synchronize do
        raise ArgumentError.new("qwen25-vl weights: loader is closed") if @closed
        return output if token_ids.empty?

        source = @tensor_sources[EMBEDDING_NAME]?
        raise ArgumentError.new("qwen25-vl weights: validated embedding source is unavailable") unless source
        verify_source_path!(source)
        File.open(source[:path], "rb") do |io|
          validate_source_bounds!(source, io.size, source[:byte_size], EMBEDDING_NAME)
          token_ids.each_with_index do |token_id, index|
            row_offset = checked_multiply(token_id, row_bytes, "embedding row offset")
            destination_offset = index.to_i64 * row_bytes
            io.seek(source[:data_offset] + row_offset)
            io.read_fully(output[destination_offset.to_i, row_bytes.to_i])
          end
        end
      end

      output
    end

    private def stream_tensor_to_buffer_unlocked(name : String, expected_sha256 : String? = nil) : MetalBuffer
      expected_shape = @selected_tensor_shapes[name]?
      raise ArgumentError.new("qwen25-vl weights: #{name} is not a selected tensor") unless expected_shape
      expected_bytes = self.class.checked_shape_bytes(expected_shape)
      source = @tensor_sources[name]?
      raise ArgumentError.new("qwen25-vl weights: validated source for #{name} is unavailable") unless source
      unless source[:byte_size] == expected_bytes
        raise ArgumentError.new("qwen25-vl weights: #{name} source bytes differ from its validated shape")
      end

      verify_source_path!(source)
      buffer = MetalBuffer.new(expected_bytes)
      begin
        File.open(source[:path], "rb") do |io|
          file_size = io.size
          validate_source_bounds!(source, file_size, expected_bytes, name)
          io.seek(source[:data_offset])
          destination = buffer.contents.as(UInt8*)
          if expected_sha256
            self.class.copy_and_verify_sha256(io, destination, expected_bytes, expected_sha256, name)
          else
            chunk = Bytes.new(Math.min(UPLOAD_CHUNK_BYTES.to_i64, expected_bytes).to_i)
            offset = 0_i64
            while offset < expected_bytes
              count = Math.min(chunk.size.to_i64, expected_bytes - offset).to_i
              io.read_fully(chunk[0, count])
              (destination + offset.to_i).copy_from(chunk.to_unsafe, count)
              offset += count
            end
          end
        end
        buffer
      rescue ex
        buffer.release if buffer.valid?
        raise ex
      end
    end

    private def validate_config! : Nil
      config = read_json_object("config.json", "config.json")
      require_string!(config, "model_type", "config.json", MODEL_TYPE)
      architectures = config["architectures"]?.try(&.as_a?)
      unless architectures && architectures.size == 1 && architectures[0].as_s? == ARCHITECTURE
        raise ArgumentError.new("qwen25-vl weights: config.json architectures must identify #{ARCHITECTURE}")
      end

      require_string!(config, "torch_dtype", "config.json", "bfloat16")
      require_string!(config, "hidden_act", "config.json", "silu")
      require_integer!(config, "hidden_size", "config.json", @config.dim)
      require_integer!(config, "intermediate_size", "config.json", @config.ffn_dim)
      require_integer!(config, "num_hidden_layers", "config.json", @config.layers)
      require_integer!(config, "num_attention_heads", "config.json", @config.heads)
      require_integer!(config, "num_key_value_heads", "config.json", @config.kv_heads)
      require_integer!(config, "vocab_size", "config.json", @vocab_size)
      require_integer!(config, "max_position_embeddings", "config.json", @config.max_positions)
      require_float!(config, "rms_norm_eps", "config.json", @config.rms_norm_eps)
      require_float!(config, "rope_theta", "config.json", @config.rope_theta)
      unless config["tie_word_embeddings"]?.try(&.as_bool?) == false
        raise ArgumentError.new("qwen25-vl weights: config.json tie_word_embeddings must be false")
      end
    end

    private def read_weight_map! : Hash(String, String)
      index = read_json_object("model.safetensors.index.json", "model.safetensors.index.json")
      raw_map = index["weight_map"]?.try(&.as_h?)
      raise ArgumentError.new("qwen25-vl weights: index is missing weight_map object") unless raw_map

      selected = Set(String).new(@selected_tensor_names)
      weight_map = {} of String => String
      raw_map.each do |name, raw_shard|
        shard = raw_shard.as_s?
        raise ArgumentError.new("qwen25-vl weights: index shard path for #{name} must be a string") unless shard
        next unless selected.includes?(name)
        weight_map[name] = shard
      end

      missing = @selected_tensor_names.reject { |name| weight_map.has_key?(name) }
      unless missing.empty?
        preview = missing.first(8).join(", ")
        raise ArgumentError.new("qwen25-vl weights: index is missing required tensors (#{missing.size}); first missing: #{preview}")
      end
      weight_map
    end

    private def resolve_shard_paths!(weight_map : Hash(String, String)) : Hash(String, String)
      resolved = {} of String => String
      weight_map.each_value do |relative_path|
        next if resolved.has_key?(relative_path)
        if relative_path.empty? || relative_path.starts_with?("/") || relative_path.includes?('\\') ||
           relative_path.split('/').any? { |part| part.empty? || part == ".." }
          raise ArgumentError.new("qwen25-vl weights: invalid shard path #{relative_path.inspect}")
        end

        candidate = File.expand_path(relative_path, @model_dir)
        real_path = begin
          File.realpath(candidate)
        rescue ex
          raise ArgumentError.new("qwen25-vl weights: shard #{relative_path.inspect} is missing or unreadable: #{ex.message}")
        end
        unless contained_model_file?(real_path) && File.file?(real_path)
          raise ArgumentError.new("qwen25-vl weights: shard path escapes model directory: #{relative_path.inspect}")
        end
        resolved[relative_path] = real_path
      end
      resolved
    end

    private def validate_selected_shards!(
      weight_map : Hash(String, String), shard_paths : Hash(String, String),
    ) : Nil
      required = Set(String).new(@selected_tensor_names)
      seen = Set(String).new

      shard_paths.each do |shard_name, path|
        preflight_header!(path, shard_name)
        reader = begin
          SafetensorsFile.new(path)
        rescue ex
          raise ArgumentError.new("qwen25-vl weights: invalid safetensors shard #{shard_name}: #{ex.message}")
        end

        begin
          validate_shard_bounds!(reader, path, shard_name)
          reader.tensors.each do |info|
            next unless required.includes?(info.name)

            mapped_shard = weight_map[info.name]?
            unless mapped_shard == shard_name
              raise ArgumentError.new("qwen25-vl weights: index shard mapping disagrees for #{info.name}")
            end
            if seen.includes?(info.name)
              raise ArgumentError.new("qwen25-vl weights: duplicate selected tensor #{info.name}")
            end
            unless info.dtype.bf16?
              raise ArgumentError.new("qwen25-vl weights: #{info.name} must be BF16, got #{info.dtype}")
            end
            expected_shape = @selected_tensor_shapes[info.name]
            unless info.shape == expected_shape
              raise ArgumentError.new("qwen25-vl weights: #{info.name} shape #{info.shape} != expected #{expected_shape}")
            end
            expected_bytes = self.class.checked_shape_bytes(expected_shape)
            unless info.data_bytes == expected_bytes
              raise ArgumentError.new("qwen25-vl weights: #{info.name} byte size #{info.data_bytes} != expected #{expected_bytes}")
            end

            data_offset = reader.data_offset + info.data_start
            @tensor_sources[info.name] = {path: path, data_offset: data_offset, byte_size: info.data_bytes}
            seen << info.name
          end
        ensure
          reader.close
        end
      end

      missing = required.reject { |name| seen.includes?(name) }
      unless missing.empty?
        preview = missing.first(8).join(", ")
        raise ArgumentError.new("qwen25-vl weights: shards are missing required tensors (#{missing.size}); first missing: #{preview}")
      end
    end

    private def preflight_header!(path : String, shard_name : String) : Nil
      file_size = File.size(path)
      raise ArgumentError.new("qwen25-vl weights: shard #{shard_name} is shorter than a safetensors header") if file_size < 8
      File.open(path, "rb") do |io|
        prefix = Bytes.new(8)
        io.read_fully(prefix)
        header_size = IO::ByteFormat::LittleEndian.decode(UInt64, prefix)
        unless header_size > 0_u64 && header_size <= (file_size - 8).to_u64 && header_size <= Int32::MAX.to_u64
          raise ArgumentError.new("qwen25-vl weights: shard #{shard_name} has invalid header bounds")
        end
      end
    end

    private def validate_shard_bounds!(reader : SafetensorsFile, path : String, shard_name : String) : Nil
      file_size = File.size(path)
      payload_size = file_size - reader.data_offset
      raise ArgumentError.new("qwen25-vl weights: shard #{shard_name} has invalid payload start") if payload_size < 0

      previous_end = 0_i64
      reader.tensors.sort_by(&.data_start).each do |info|
        unless info.data_start >= 0 && info.data_end >= info.data_start && info.data_end <= payload_size
          raise ArgumentError.new("qwen25-vl weights: tensor #{info.name} offsets outside shard bounds")
        end
        if info.data_start < previous_end
          raise ArgumentError.new("qwen25-vl weights: tensor #{info.name} offsets overlap another tensor")
        end
        previous_end = info.data_end if info.data_end > previous_end
      end
    end

    private def read_json_object(relative_path : String, label : String) : Hash(String, JSON::Any)
      path = resolve_model_file!(relative_path, label)
      JSON.parse(File.read(path)).as_h
    rescue ex
      raise ArgumentError.new("qwen25-vl weights: #{label} is missing or malformed: #{ex.message}")
    end

    private def resolve_model_file!(relative_path : String, label : String) : String
      if relative_path.empty? || relative_path.starts_with?("/") || relative_path.includes?('\\') ||
         relative_path.split('/').any? { |part| part.empty? || part == ".." }
        raise ArgumentError.new("qwen25-vl weights: invalid #{label} path")
      end
      candidate = File.expand_path(relative_path, @model_dir)
      real_path = begin
        File.realpath(candidate)
      rescue ex
        raise ArgumentError.new("qwen25-vl weights: #{label} is missing or unreadable: #{ex.message}")
      end
      unless contained_model_file?(real_path) && File.file?(real_path)
        raise ArgumentError.new("qwen25-vl weights: #{label} path escapes model directory")
      end
      real_path
    end

    private def contained_model_file?(path : String) : Bool
      root_prefix = @model_dir.ends_with?(File::SEPARATOR) ? @model_dir : "#{@model_dir}#{File::SEPARATOR}"
      path.starts_with?(root_prefix)
    end

    private def verify_source_path!(source : TensorSource) : Nil
      real_path = begin
        File.realpath(source[:path])
      rescue ex
        raise ArgumentError.new("qwen25-vl weights: selected shard is no longer readable: #{ex.message}")
      end
      unless real_path == source[:path] && contained_model_file?(real_path)
        raise ArgumentError.new("qwen25-vl weights: selected shard escaped model directory after validation")
      end
    end

    private def validate_source_bounds!(
      source : TensorSource, file_size : Int64, expected_bytes : Int64, name : String,
    ) : Nil
      unless source[:data_offset] >= 0 && source[:data_offset] <= file_size &&
             expected_bytes >= 0 && expected_bytes <= file_size - source[:data_offset]
        raise ArgumentError.new("qwen25-vl weights: #{name} source outside shard bounds")
      end
    end

    private def layer_prefix!(layer_index : Int32) : Nil
      unless 0 <= layer_index < @selected_layer_count
        raise ArgumentError.new("qwen25-vl weights: layer index #{layer_index} outside selected prefix 0...#{@selected_layer_count}")
      end
    end

    private def require_string!(
      object : Hash(String, JSON::Any), key : String, label : String, expected : String,
    ) : Nil
      value = object[key]?.try(&.as_s?)
      unless value == expected
        raise ArgumentError.new("qwen25-vl weights: #{label} #{key} must be #{expected.inspect}")
      end
    end

    private def require_integer!(
      object : Hash(String, JSON::Any), key : String, label : String, expected : Int32,
    ) : Nil
      value = object[key]?.try(&.as_i?)
      unless value == expected
        raise ArgumentError.new("qwen25-vl weights: #{label} #{key} must be #{expected}")
      end
    end

    private def require_float!(
      object : Hash(String, JSON::Any), key : String, label : String, expected : Float32,
    ) : Nil
      raw = object[key]?
      actual = raw.try(&.as_f?) || raw.try(&.as_i?.try(&.to_f64))
      unless actual && (actual - expected.to_f64).abs <= Math.max(1e-12_f64, expected.to_f64.abs * 1e-7_f64)
        raise ArgumentError.new("qwen25-vl weights: #{label} #{key} must equal #{expected}")
      end
    end

    def self.checked_shape_bytes(shape : Array(Int64)) : Int64
      elements = shape.reduce(1_i64) do |count, dimension|
        unless dimension > 0
          raise ArgumentError.new("qwen25-vl weights: tensor shape dimensions must be positive")
        end
        checked_multiply(count, dimension, "tensor element count")
      end
      checked_multiply(elements, 2_i64, "BF16 tensor bytes")
    end

    def self.sum_shape_bytes(shapes : Enumerable(Array(Int64))) : Int64
      shapes.reduce(0_i64) do |total, shape|
        bytes = checked_shape_bytes(shape)
        raise ArgumentError.new("qwen25-vl weights: selected byte count overflow") if total > Int64::MAX - bytes
        total + bytes
      end
    end

    def self.checked_multiply(left : Int64, right : Int64, label : String) : Int64
      if left < 0 || right < 0 || (right > 0 && left > Int64::MAX // right)
        raise ArgumentError.new("qwen25-vl weights: #{label} overflow")
      end
      left * right
    end

    private def checked_multiply(left : Int64, right : Int64, label : String) : Int64
      self.class.checked_multiply(left, right, label)
    end

    private def sum_shape_bytes(shapes : Enumerable(Array(Int64))) : Int64
      self.class.sum_shape_bytes(shapes)
    end
  end
end
