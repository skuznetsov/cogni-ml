require "json"

require "../weights"
require "../../../formats/safetensors/reader"
require "./scratch"

module ML::GGUF
  # Lazily uploaded, release-validated DSpark checkpoint tensors. Each view
  # owns an aligned shared Metal buffer so Safetensors' header-relative
  # alignment never leaks into the GPU binding alignment.
  class Inkling2DSparkWeights
    private MAX_FILE_BYTES   = 1_073_741_824_i64
    private MAX_HEADER_BYTES = 64_i64 * 1024 * 1024
    private MAX_CONFIG_BYTES = 1_i64 * 1024 * 1024
    private VIEW_ALIGNMENT   = 16_i64

    @safetensors : SafetensorsFile?
    @infos : Hash(String, SafeTensorInfo)
    @buffers : Hash(String, ML::MetalBuffer)
    @allocated_bytes : Int64
    @closed : Bool
    @mutex : Mutex

    def initialize(path : String, config_path : String, target : Inkling2Config)
      @safetensors = nil
      @infos = {} of String => SafeTensorInfo
      @buffers = {} of String => ML::MetalBuffer
      @allocated_bytes = 0_i64
      @closed = false
      @mutex = Mutex.new

      begin
        validate_config!(config_path, target)
        preflight_safetensors!(path)

        reader = SafetensorsFile.new(path)
        @safetensors = reader
        validate_inventory!(reader, path)
      rescue ex
        if reader = @safetensors
          reader.close rescue nil
        end
        @safetensors = nil
        @closed = true
        raise ex
      end
    end

    # Return a 16-byte-aligned BF16 tensor view. Uploads are cached on first
    # use; this keeps cold construction bounded while avoiding repeated copies.
    def view(name : String) : Inkling2View
      @mutex.synchronize do
        ensure_open!
        if buffer = @buffers[name]?
          return Inkling2View.new(buffer, 0_i64)
        end

        info = @infos[name]? || raise ArgumentError.new("inkling2_dspark_weights: unknown tensor #{name.inspect}")
        byte_count = info.data_bytes
        raise ArgumentError.new("inkling2_dspark_weights: invalid byte extent for #{name.inspect}") unless byte_count > 0
        raise ArgumentError.new("inkling2_dspark_weights: tensor #{name.inspect} exceeds the allocation bound") if byte_count > MAX_FILE_BYTES
        raise ArgumentError.new("inkling2_dspark_weights: aggregate tensor uploads exceed the allocation bound") if @allocated_bytes > MAX_FILE_BYTES - byte_count

        reader = @safetensors || raise "inkling2_dspark_weights: checkpoint is closed"
        raw = reader.tensor_bytes(info)
        raise ArgumentError.new("inkling2_dspark_weights: #{name.inspect} raw byte extent changed") unless raw.size.to_i64 == byte_count

        buffer = ML::MetalBuffer.new(byte_count, ML::StorageMode::Shared)
        begin
          raise ArgumentError.new("inkling2_dspark_weights: Metal buffer base is not 16-byte aligned") unless buffer.contents.address % VIEW_ALIGNMENT.to_u64 == 0
          buffer.write_bytes(raw.to_unsafe, byte_count.to_i)

          view = Inkling2View.new(buffer, 0_i64)
          @buffers[name] = buffer
          @allocated_bytes += byte_count
          view
        rescue ex
          buffer.release
          raise ex
        end
      end
    end

    def close : Nil
      @mutex.synchronize do
        return if @closed

        @closed = true
        @buffers.each_value(&.release)
        @buffers.clear
        @allocated_bytes = 0_i64
        if reader = @safetensors
          begin
            reader.close
          ensure
            @safetensors = nil
          end
        end
      end
    end

    def finalize
      close rescue nil
    end

    private def ensure_open! : Nil
      raise "inkling2_dspark_weights: checkpoint is closed" if @closed
    end

    private def validate_config!(path : String, target : Inkling2Config) : Nil
      size = File.size(path)
      raise ArgumentError.new("inkling2_dspark_weights: config exceeds #{MAX_CONFIG_BYTES} bytes") if size > MAX_CONFIG_BYTES

      root = JSON.parse(File.read(path)).as_h
      raise ArgumentError.new("inkling2_dspark_weights: unsupported model_type") unless string(root, "model_type") == "inkling2_loop"
      architectures = root["architectures"]?.try(&.as_a.map(&.as_s)) || [] of String
      raise ArgumentError.new("inkling2_dspark_weights: unsupported architecture") unless architectures == ["Inkling2LoopForConditionalGeneration"]

      text = object(root, "text_config")
      mtp = object(root, "mtp_config")
      loop_config = object(text, "loop_config")

      expect_int(text, "hidden_size", target.n_embd)
      expect_int(text, "num_hidden_layers", target.n_layer)
      expect_int(text, "vocab_size", target.padded_vocab_size)
      expect_int(text, "unpadded_vocab_size", target.vocab_size)
      expect_int(loop_config, "core_start", target.core_start)
      expect_int(loop_config, "core_end", target.core_end)

      raise ArgumentError.new("inkling2_dspark_weights: target width must be 2304") unless target.n_embd == 2304
      raise ArgumentError.new("inkling2_dspark_weights: target vocabulary must be 200058/201024") unless target.vocab_size == 200_058 && target.padded_vocab_size == 201_024
      unless target.n_layer == 24 && target.core_start == 6 && target.core_end == 18
        raise ArgumentError.new("inkling2_dspark_weights: target loop geometry must be 24 layers with core [6,18)")
      end

      expect_int(mtp, "hidden_size", target.n_embd)
      expect_int(mtp, "num_hidden_layers", 3)
      expect_int(mtp, "num_attention_heads", 24)
      expect_int(mtp, "num_key_value_heads", 8)
      expect_int(mtp, "head_dim", 128)
      expect_int(mtp, "intermediate_size", 9_216)
      expect_int(mtp, "max_position_embeddings", 1_048_576)
      expect_int(mtp, "sliding_window_size", 1_024)
      expect_int(mtp, "block_size", 8)
      expect_int(mtp, "markov_rank", 256)
      raise ArgumentError.new("inkling2_dspark_weights: mtp dtype must be bfloat16") unless string(mtp, "torch_dtype") == "bfloat16"
      raise ArgumentError.new("inkling2_dspark_weights: unsupported Markov head type") unless string(mtp, "markov_head_type") == "rnn_hidden"
      raise ArgumentError.new("inkling2_dspark_weights: unsupported local layer schedule") unless integer_array(mtp, "local_layer_ids") == [0_i64, 2_i64]
      raise ArgumentError.new("inkling2_dspark_weights: unsupported target tap schedule") unless integer_array(mtp, "target_layer_ids") == [5_i64, 23_i64, 41_i64, 48_i64]

      rms_eps = number(mtp, "rms_norm_eps")
      raise ArgumentError.new("inkling2_dspark_weights: mtp RMS epsilon must be 1e-6") unless rms_eps == 1e-6
      rope_theta = number(mtp, "rope_theta")
      raise ArgumentError.new("inkling2_dspark_weights: mtp rope theta must be 1e6") unless rope_theta == 1_000_000.0
      raise ArgumentError.new("inkling2_dspark_weights: MTP projections must be bias-free") unless boolean(mtp, "q_bias") == false && boolean(mtp, "o_bias") == false

      target_loop_start = integer(loop_config, "core_start")
      target_loop_end = integer(loop_config, "core_end")
      unless target_loop_start == 6 && target_loop_end == 18
        raise ArgumentError.new("inkling2_dspark_weights: config loop geometry must be core [6,18)")
      end
    rescue ex : KeyError | TypeCastError | JSON::ParseException
      raise ArgumentError.new("inkling2_dspark_weights: invalid config #{path.inspect}: #{ex.message}")
    end

    private def preflight_safetensors!(path : String) : Nil
      file_size = File.size(path)
      raise ArgumentError.new("inkling2_dspark_weights: checkpoint must be nonempty") unless file_size > 8
      raise ArgumentError.new("inkling2_dspark_weights: checkpoint exceeds #{MAX_FILE_BYTES} bytes") if file_size > MAX_FILE_BYTES

      File.open(path, "rb") do |io|
        length_bytes = Bytes.new(8)
        io.read_fully(length_bytes)
        header_length = IO::ByteFormat::LittleEndian.decode(UInt64, length_bytes).to_i64
        if header_length <= 0 || header_length > MAX_HEADER_BYTES || header_length > file_size - 8
          raise ArgumentError.new("inkling2_dspark_weights: invalid safetensors header extent #{header_length}")
        end

        header_bytes = Bytes.new(header_length.to_i32)
        io.read_fully(header_bytes)
        header = JSON.parse(String.new(header_bytes)).as_h
        expected = expected_shapes
        actual_names = header.keys.reject { |name| name == "__metadata__" }
        unless actual_names.size == expected.size && actual_names.all? { |name| expected.has_key?(name) }
          unknown = actual_names.reject { |name| expected.has_key?(name) }
          missing = expected.keys.reject { |name| actual_names.includes?(name) }
          raise ArgumentError.new("inkling2_dspark_weights: tensor inventory mismatch (missing=#{missing.join(",")}, unknown=#{unknown.join(",")})")
        end

        data_bytes = file_size - 8_i64 - header_length
        ranges = [] of Tuple(Int64, Int64, String)
        expected_total = 0_i64
        expected.each do |name, shape|
          entry = header[name].as_h
          dtype = entry["dtype"].as_s
          raise ArgumentError.new("inkling2_dspark_weights: #{name} must be BF16") unless dtype == "BF16"
          shape_values = entry["shape"].as_a.map(&.as_i64)
          unless shape_values == shape.map(&.to_i64)
            raise ArgumentError.new("inkling2_dspark_weights: #{name} shape #{shape_values} != expected #{shape}")
          end

          offsets = entry["data_offsets"].as_a.map(&.as_i64)
          raise ArgumentError.new("inkling2_dspark_weights: #{name} must have a two-element extent") unless offsets.size == 2
          start_byte, end_byte = offsets
          expected_bytes = expected_tensor_bytes(shape)
          unless start_byte >= 0 && end_byte >= start_byte && end_byte - start_byte == expected_bytes && end_byte <= data_bytes
            raise ArgumentError.new("inkling2_dspark_weights: #{name} has invalid data extent [#{start_byte},#{end_byte})")
          end
          # Safetensors aligns offsets relative to the data section. The
          # section itself may start at an address that is only 8-byte aligned;
          # `view` copies each tensor into its own aligned Metal allocation.
          unless start_byte % 2 == 0
            raise ArgumentError.new("inkling2_dspark_weights: #{name} is not BF16-aligned in the checkpoint")
          end
          ranges << {start_byte, end_byte, name}
          expected_total += expected_bytes
        end

        ranges.sort_by!(&.[0])
        cursor = 0_i64
        ranges.each do |range|
          unless range[0] == cursor
            raise ArgumentError.new("inkling2_dspark_weights: tensor extent gap or overlap before #{range[2]}")
          end
          cursor = range[1]
        end
        unless cursor == data_bytes && cursor == expected_total
          raise ArgumentError.new("inkling2_dspark_weights: tensor data extent #{cursor} != file data extent #{data_bytes}")
        end
      end
    rescue ex : KeyError | TypeCastError | JSON::ParseException
      raise ArgumentError.new("inkling2_dspark_weights: invalid safetensors header in #{path.inspect}: #{ex.message}")
    end

    private def validate_inventory!(reader : SafetensorsFile, path : String) : Nil
      expected = expected_shapes
      raise ArgumentError.new("inkling2_dspark_weights: expected #{expected.size} tensors, found #{reader.tensors.size}") unless reader.tensors.size == expected.size

      expected.each do |name, shape|
        info = reader.tensor(name) || raise ArgumentError.new("inkling2_dspark_weights: missing tensor #{name.inspect}")
        raise ArgumentError.new("inkling2_dspark_weights: #{name} must be BF16") unless info.dtype.bf16?
        unless info.shape == shape.map(&.to_i64)
          raise ArgumentError.new("inkling2_dspark_weights: #{name} shape #{info.shape} != expected #{shape}")
        end
        info.validate_size!
        unless info.data_start >= 0 && info.data_end <= File.size(path) - reader.data_offset
          raise ArgumentError.new("inkling2_dspark_weights: #{name} extent lies outside the safetensors data section")
        end
        @infos[name] = info
      end
    end

    private def expected_shapes : Hash(String, Array(Int32))
      shapes = {} of String => Array(Int32)
      shapes["fuse_weight"] = [4, 4]
      3.times do |layer|
        prefix = "layers.#{layer}."
        shapes[prefix + "dflare_norm.weight"] = [2_304]
        shapes[prefix + "input_layernorm.weight"] = [2_304]
        shapes[prefix + "mlp.down_proj.weight"] = [2_304, 9_216]
        shapes[prefix + "mlp.gate_proj.weight"] = [9_216, 2_304]
        shapes[prefix + "mlp.up_proj.weight"] = [9_216, 2_304]
        shapes[prefix + "post_attention_layernorm.weight"] = [2_304]
        shapes[prefix + "self_attn.k_norm.weight"] = [128]
        shapes[prefix + "self_attn.k_norm_ctx.weight"] = [128]
        shapes[prefix + "self_attn.k_proj.weight"] = [1_024, 2_304]
        shapes[prefix + "self_attn.k_proj_ctx.weight"] = [1_024, 2_304]
        shapes[prefix + "self_attn.o_proj.weight"] = [2_304, 3_072]
        shapes[prefix + "self_attn.q_norm.weight"] = [128]
        shapes[prefix + "self_attn.q_proj.weight"] = [3_072, 2_304]
        shapes[prefix + "self_attn.v_proj.weight"] = [1_024, 2_304]
        shapes[prefix + "self_attn.v_proj_ctx.weight"] = [1_024, 2_304]
      end

      shapes["markov_head.dflare_norm.weight"] = [2_304]
      shapes["markov_head.hidden_out.bias"] = [2_304]
      shapes["markov_head.hidden_out.weight"] = [2_304, 256]
      shapes["markov_head.markov_w1.weight"] = [201_024, 256]
      shapes["markov_head.rnn_proj.bias"] = [768]
      shapes["markov_head.rnn_proj.weight"] = [768, 5_120]
      shapes["mask_embed"] = [2_304]
      shapes["norm.weight"] = [2_304]
      shapes
    end

    private def expected_tensor_bytes(shape : Array(Int32)) : Int64
      elements = 1_i64
      shape.each do |dimension|
        raise ArgumentError.new("inkling2_dspark_weights: invalid expected tensor shape") unless dimension > 0
        raise ArgumentError.new("inkling2_dspark_weights: tensor shape overflows") if elements > Int64::MAX // dimension.to_i64
        elements *= dimension.to_i64
      end
      raise ArgumentError.new("inkling2_dspark_weights: tensor byte size overflows") if elements > Int64::MAX // 2_i64
      elements * 2_i64
    end

    private def object(parent : Hash(String, JSON::Any), key : String) : Hash(String, JSON::Any)
      parent[key].as_h
    end

    private def string(parent : Hash(String, JSON::Any), key : String) : String
      parent[key].as_s
    end

    private def integer(parent : Hash(String, JSON::Any), key : String) : Int64
      parent[key].as_i64
    end

    private def number(parent : Hash(String, JSON::Any), key : String) : Float64
      value = parent[key]
      value.raw.is_a?(Float64) ? value.as_f : value.as_i64.to_f64
    end

    private def boolean(parent : Hash(String, JSON::Any), key : String) : Bool
      parent[key].as_bool
    end

    private def integer_array(parent : Hash(String, JSON::Any), key : String) : Array(Int64)
      parent[key].as_a.map(&.as_i64)
    end

    private def expect_int(parent : Hash(String, JSON::Any), key : String, expected : Int32) : Nil
      actual = integer(parent, key)
      raise ArgumentError.new("inkling2_dspark_weights: #{key} #{actual} != target #{expected}") unless actual == expected
    end
  end
end
