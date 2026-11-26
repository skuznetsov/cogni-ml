require "json"
require "digest/sha256"
require "set"
require "./reader"

module ML::GGUF
  # Reader for the versioned text-only reference emitted by
  # scripts/qwen_image21_text_reference.py. The payload remains owned by this
  # object; large hidden-state tensors are decoded only when requested.
  class Qwen3VLTextReference
    SCHEMA                      = "qwen-image21-text-reference"
    SCHEMA_VERSION              = 1
    MODEL_REPO                  = "Qwen/Qwen-Image-2.1"
    PAYLOAD_FILE                = "qwen_image21_text_reference.bin"
    CONTEXT_DIM                 = 4096
    RED_CUBE_REVISION           = "790c92633540aa0cb11d9abf19eb46d861714758"
    RED_CUBE_HIDDEN_STATE_COUNT = 37

    REVISION_RE = /\A[0-9a-fA-F]{40}\z/
    SHA256_RE   = /\A[0-9a-fA-F]{64}\z/

    private struct TensorDescriptor
      getter name : String
      getter dtype : String
      getter shape : Array(Int64)
      getter offset_bytes : Int64
      getter nbytes : Int64
      getter sha256 : String

      def initialize(@name, @dtype, @shape, @offset_bytes, @nbytes, @sha256)
      end
    end

    getter model_revision : String
    getter prompt : String
    getter raw_template_text : String
    getter input_ids : Array(Int64)
    getter attention_mask : Array(Bool)
    getter mm_token_type_ids : Array(Int64)?
    getter raw_sequence_length : Int32
    getter actual_sequence_length : Int32
    getter max_sequence_length : Int32
    getter drop_idx : Int32
    getter hidden_state_count : Int32
    getter pre_final_rmsnorm_embeddings : Array(Float32)

    @payload : Bytes
    # `Bytes` is a view into the string returned by File.read in `load`; retain
    # that string so its backing storage stays alive. `parse` owns a copy.
    @payload_owner : String?
    @hidden_states : Array(TensorDescriptor)

    private def initialize(
      @model_revision : String,
      @prompt : String,
      @raw_template_text : String,
      @input_ids : Array(Int64),
      @attention_mask : Array(Bool),
      @mm_token_type_ids : Array(Int64)?,
      @raw_sequence_length : Int32,
      @actual_sequence_length : Int32,
      @max_sequence_length : Int32,
      @drop_idx : Int32,
      @hidden_state_count : Int32,
      @pre_final_rmsnorm_embeddings : Array(Float32),
      @hidden_states : Array(TensorDescriptor),
      @payload : Bytes,
      @payload_owner : String?,
    )
    end

    def self.load(manifest_path : String) : self
      manifest_json = File.read(manifest_path)
      payload_path = File.join(File.dirname(manifest_path), PAYLOAD_FILE)
      raise ArgumentError.new("text reference payload not found: #{payload_path}") unless File.file?(payload_path)
      payload_owner = File.read(payload_path)
      parse_owned(manifest_json, payload_owner.to_slice, payload_owner)
    end

    # Parses a manifest and takes an owned copy of the supplied payload bytes.
    def self.parse(manifest_json : String, payload : Bytes) : self
      payload_copy = payload.dup
      parse_owned(manifest_json, payload_copy, nil)
    end

    # Zero-based index, matching hidden_state_000, hidden_state_001, ... in
    # the reference payload. Each result is flattened row-major Float32.
    def hidden_state(layer_index : Int32) : Array(Float32)
      unless layer_index >= 0 && layer_index < @hidden_state_count
        raise ArgumentError.new("hidden-state layer index out of range: #{layer_index}")
      end
      descriptor = @hidden_states[layer_index]
      decode_float_tensor(descriptor)
    end

    private def self.parse_owned(manifest_json : String, payload : Bytes, payload_owner : String?) : self
      manifest = begin
        JSON.parse(manifest_json)
      rescue error : JSON::ParseException
        invalid("invalid text reference JSON: #{error.message}")
      end

      expect(string(required(manifest, "schema"), "schema") == SCHEMA, "unsupported text reference schema")
      expect(integer(required(manifest, "schema_version"), "schema_version") == SCHEMA_VERSION,
        "unsupported text reference schema version")

      model = required(manifest, "model")
      expect(string(required(model, "repo"), "model.repo") == MODEL_REPO, "wrong text reference model repository")
      revision = string(required(model, "revision_sha"), "model.revision_sha")
      expect(REVISION_RE.matches?(revision), "model revision must be a full 40-character commit SHA")
      model_fields = object(model, "model")
      {
        "pipeline_class"     => "QwenImage21Pipeline",
        "text_encoder_class" => "Qwen3VLForConditionalGeneration",
        "processor_class"    => "Qwen3VLProcessor",
      }.each do |field, expected|
        if actual = model_fields[field]?
          expect(string(actual, "model.#{field}") == expected, "wrong text reference model #{field}")
        end
      end

      prompt = string(required(manifest, "prompt"), "prompt")
      tokenization = required(manifest, "tokenization")
      raw_template_text = string(required(tokenization, "raw_template_text"), "tokenization.raw_template_text")

      expect(string(required(manifest, "payload_file"), "payload_file") == PAYLOAD_FILE,
        "text reference payload file mismatch")
      payload_nbytes = integer(required(manifest, "payload_nbytes"), "payload_nbytes")
      expect(payload_nbytes == payload.size, "text reference payload length mismatch")
      verify_sha256(payload, string(required(manifest, "payload_sha256"), "payload_sha256"), "payload")

      sequence = required(manifest, "sequence")
      max_sequence_length = positive_int32(required(sequence, "max_sequence_length"), "sequence.max_sequence_length")
      actual_sequence_length = positive_int32(required(sequence, "actual_sequence_length"), "sequence.actual_sequence_length")
      raw_input_shape = dimensions(required(sequence, "raw_input_shape"), "sequence.raw_input_shape")
      drop_idx = nonnegative_int32(required(sequence, "drop_idx"), "sequence.drop_idx")
      expect(raw_input_shape.size == 2 && raw_input_shape[0] == 1,
        "raw input shape must be batch-one [1, raw_sequence_length]")

      embedding = required(manifest, "embedding")
      expect(boolean(required(embedding, "pre_final_rmsnorm"), "embedding.pre_final_rmsnorm"),
        "text reference must contain pre-final-RMSNorm embeddings")
      hidden_count = positive_int32(required(embedding, "hidden_state_count"), "embedding.hidden_state_count")
      if revision.downcase == RED_CUBE_REVISION
        expect(hidden_count == RED_CUBE_HIDDEN_STATE_COUNT,
          "pinned Qwen-Image 2.1 revision must contain 37 text hidden states")
      end
      embedding_fields = object(embedding, "embedding")
      # Early schema-v1 captures predate the model-config count fields. The
      # actual count remains mandatory and is checked against the exact tensor
      # set; when expectations are present they must agree with that count.
      if expected_value = embedding_fields["expected_hidden_state_count"]?
        expected_count = positive_int32(expected_value, "embedding.expected_hidden_state_count")
        expect(expected_count == hidden_count, "hidden-state count differs from the model-config expectation")
      end
      if decoder_count_value = embedding_fields["expected_decoder_layer_count"]?
        decoder_count = nonnegative_int32(decoder_count_value, "embedding.expected_decoder_layer_count")
        expect(decoder_count + 1 == hidden_count, "decoder-layer count differs from hidden-state count")
      end
      if hook_value = embedding_fields["rmsnorm_hook_observed_and_verified"]?
        expect(boolean(hook_value, "embedding.rmsnorm_hook_observed_and_verified"),
          "pre-final RMSNorm capture was not verified")
      end
      embedding_shape = dimensions(required(embedding, "shape"), "embedding.shape")

      tensor_json = required(manifest, "tensors")
      tensor_map = object(tensor_json, "tensors")
      has_mm_token_type_ids = tensor_map.has_key?("mm_token_type_ids")
      expected_tensor_count = 3_i64 + hidden_count.to_i64 + (has_mm_token_type_ids ? 1_i64 : 0_i64)
      expect(tensor_map.size.to_i64 == expected_tensor_count,
        "text reference tensor set does not match its hidden-state count")

      descriptors = {} of String => TensorDescriptor
      tensor_map.each do |name, descriptor_json|
        descriptors[name] = parse_descriptor(name, descriptor_json, payload)
      end

      input_descriptor = descriptor(descriptors, "input_ids")
      mask_descriptor = descriptor(descriptors, "attention_mask")
      embedding_descriptor = descriptor(descriptors, "pre_final_rmsnorm_embeddings")
      raw_sequence_length = raw_input_shape[1]
      expect(raw_sequence_length > 0 && raw_sequence_length <= Int32::MAX, "raw sequence length out of range")
      expected_raw_shape = [1_i64, raw_sequence_length]
      expect(raw_input_shape == expected_raw_shape, "raw input shape is invalid")
      validate_descriptor(input_descriptor, "int64-le", expected_raw_shape)
      validate_descriptor(mask_descriptor, "int64-le", expected_raw_shape)
      expect(embedding_shape == [1_i64, actual_sequence_length.to_i64, CONTEXT_DIM.to_i64],
        "pre-final embedding manifest shape mismatch")
      validate_descriptor(embedding_descriptor, float_dtype(embedding_descriptor.dtype), embedding_shape)

      input_ids = read_int64_tensor(input_descriptor, payload)
      mask_values = read_int64_tensor(mask_descriptor, payload)
      attention_mask = mask_values.map do |value|
        expect(value == 0 || value == 1, "attention_mask must contain only 0/1 values")
        value == 1
      end
      attended_count = attention_mask.count(true)
      expect(drop_idx < attended_count, "drop_idx must be smaller than the attended input token count")
      expect(attended_count - drop_idx == actual_sequence_length,
        "actual sequence length does not match attention_mask and drop_idx")
      expect(actual_sequence_length <= max_sequence_length,
        "post-drop sequence length exceeds max_sequence_length guard")

      mm_token_type_ids = if has_mm_token_type_ids
                            mm_descriptor = descriptor(descriptors, "mm_token_type_ids")
                            validate_descriptor(mm_descriptor, "int64-le", expected_raw_shape)
                            read_int64_tensor(mm_descriptor, payload)
                          end

      expected_hidden_shape = [1_i64, raw_sequence_length, CONTEXT_DIM.to_i64]
      hidden_states = Array(TensorDescriptor).new(hidden_count) do |index|
        name = "hidden_state_#{index.to_s.rjust(3, '0')}"
        hidden_descriptor = descriptor(descriptors, name)
        validate_descriptor(hidden_descriptor, float_dtype(hidden_descriptor.dtype), expected_hidden_shape)
        hidden_descriptor
      end
      if revision.downcase == RED_CUBE_REVISION
        expect(embedding_descriptor.dtype == "bfloat16-le",
          "pinned red-cube reference embeddings must use bfloat16")
        hidden_states.each do |hidden_descriptor|
          expect(hidden_descriptor.dtype == "bfloat16-le",
            "pinned red-cube hidden states must use bfloat16")
        end
        if source_dtype_value = embedding_fields["source_dtype"]?
          expect(string(source_dtype_value, "embedding.source_dtype") == "bfloat16",
            "pinned red-cube source dtype must be bfloat16")
        end
      end
      expected_names = Set(String).new(["input_ids", "attention_mask", "pre_final_rmsnorm_embeddings"])
      expected_names << "mm_token_type_ids" if has_mm_token_type_ids
      hidden_count.times { |index| expected_names << "hidden_state_#{index.to_s.rjust(3, '0')}" }
      expect(descriptors.keys.to_set == expected_names, "text reference contains an unexpected tensor name")
      validate_payload_layout(descriptors.values, payload.size)

      embeddings = decode_float_tensor(embedding_descriptor, payload)
      final_hidden = decode_float_tensor(hidden_states.last, payload)
      validate_returned_embeddings(embeddings, final_hidden, attention_mask, drop_idx, actual_sequence_length)

      new(
        revision.downcase,
        prompt,
        raw_template_text,
        input_ids,
        attention_mask,
        mm_token_type_ids,
        raw_sequence_length.to_i32,
        actual_sequence_length,
        max_sequence_length,
        drop_idx,
        hidden_count,
        embeddings,
        hidden_states,
        payload,
        payload_owner,
      )
    end

    private def self.parse_descriptor(name : String, raw : JSON::Any, payload : Bytes) : TensorDescriptor
      dtype = string(required(raw, "dtype"), "tensors.#{name}.dtype")
      shape = dimensions(required(raw, "shape"), "tensors.#{name}.shape")
      offset = integer(required(raw, "offset_bytes"), "tensors.#{name}.offset_bytes")
      nbytes = integer(required(raw, "nbytes"), "tensors.#{name}.nbytes")
      sha256 = string(required(raw, "sha256"), "tensors.#{name}.sha256")
      expect(offset >= 0 && nbytes > 0, "tensor #{name} has an invalid byte range")
      expect(offset <= payload.size && nbytes <= payload.size - offset,
        "tensor #{name} byte range is outside the payload")
      expect(SHA256_RE.matches?(sha256), "tensor #{name} has an invalid SHA-256")
      tensor = TensorDescriptor.new(name, dtype, shape, offset, nbytes, sha256.downcase)
      verify_sha256(payload[offset.to_i, nbytes.to_i], tensor.sha256, "tensor #{name}")
      tensor
    end

    private def self.validate_descriptor(descriptor : TensorDescriptor, expected_dtype : String, expected_shape : Array(Int64)) : Nil
      expect(descriptor.dtype == expected_dtype, "tensor #{descriptor.name} dtype mismatch")
      expect(descriptor.shape == expected_shape, "tensor #{descriptor.name} shape mismatch")
      elements = element_count(expected_shape, "tensor #{descriptor.name} shape")
      bytes_per_element = case expected_dtype
                          when "int64-le"                  then 8_i64
                          when "float32-le"                then 4_i64
                          when "float16-le", "bfloat16-le" then 2_i64
                          else                                  invalid("unsupported tensor dtype: #{expected_dtype}")
                          end
      expect(elements <= Int64::MAX // bytes_per_element, "tensor #{descriptor.name} byte size overflows")
      expect(descriptor.nbytes == elements * bytes_per_element, "tensor #{descriptor.name} byte size mismatch")
    end

    private def self.validate_payload_layout(tensors : Array(TensorDescriptor), payload_size : Int32) : Nil
      ordered = tensors.sort_by(&.offset_bytes)
      cursor = 0_i64
      ordered.each do |tensor|
        expect(tensor.offset_bytes == cursor, "tensor payload ranges must be contiguous and non-overlapping")
        cursor += tensor.nbytes
      end
      expect(cursor == payload_size, "tensor descriptors do not cover the complete payload")
    end

    private def self.read_int64_tensor(descriptor : TensorDescriptor, payload : Bytes) : Array(Int64)
      data = payload[descriptor.offset_bytes.to_i, descriptor.nbytes.to_i]
      Array(Int64).new(data.size // 8) do |index|
        IO::ByteFormat::LittleEndian.decode(Int64, data[index * 8, 8])
      end
    end

    private def decode_float_tensor(descriptor : TensorDescriptor) : Array(Float32)
      elements = 1_i64
      descriptor.shape.each do |dimension|
        raise ArgumentError.new("tensor #{descriptor.name} shape dimensions must be positive") unless dimension > 0
        raise ArgumentError.new("tensor #{descriptor.name} shape element count overflows") if elements > Int64::MAX // dimension
        elements *= dimension
      end
      raise ArgumentError.new("tensor #{descriptor.name} is too large to decode") if elements > Int32::MAX
      data = @payload[descriptor.offset_bytes.to_i, descriptor.nbytes.to_i]
      tensor_type = case descriptor.dtype
                    when "float32-le"  then TensorType::F32
                    when "float16-le"  then TensorType::F16
                    when "bfloat16-le" then TensorType::BF16
                    else                    raise ArgumentError.new("unsupported float tensor dtype: #{descriptor.dtype}")
                    end
      values = Dequant.dequantize(data, tensor_type, elements.to_i32)
      values.each do |value|
        raise ArgumentError.new("tensor #{descriptor.name} contains a non-finite value") unless value.finite?
      end
      values
    end

    private def self.decode_float_tensor(descriptor : TensorDescriptor, payload : Bytes) : Array(Float32)
      elements = element_count(descriptor.shape, "tensor #{descriptor.name} shape")
      expect(elements <= Int32::MAX, "tensor #{descriptor.name} is too large to decode")
      data = payload[descriptor.offset_bytes.to_i, descriptor.nbytes.to_i]
      tensor_type = case descriptor.dtype
                    when "float32-le"  then TensorType::F32
                    when "float16-le"  then TensorType::F16
                    when "bfloat16-le" then TensorType::BF16
                    else                    invalid("unsupported float tensor dtype: #{descriptor.dtype}")
                    end
      values = Dequant.dequantize(data, tensor_type, elements.to_i32)
      values.each do |value|
        expect(value.finite?, "tensor #{descriptor.name} contains a non-finite value")
      end
      values
    end

    private def self.validate_returned_embeddings(
      embeddings : Array(Float32),
      final_hidden : Array(Float32),
      attention_mask : Array(Bool),
      drop_idx : Int32,
      actual_sequence_length : Int32,
    ) : Nil
      output_token = 0
      attended_token = 0
      attention_mask.each_with_index do |attended, raw_token|
        next unless attended
        if attended_token >= drop_idx
          CONTEXT_DIM.times do |column|
            expected = final_hidden[raw_token * CONTEXT_DIM + column]
            observed = embeddings[output_token * CONTEXT_DIM + column]
            expect(observed == expected,
              "returned embeddings differ from attended final hidden state after drop_idx")
          end
          output_token += 1
        end
        attended_token += 1
      end
      expect(output_token == actual_sequence_length, "embedding selection length mismatch")
    end

    private def self.float_dtype(dtype : String) : String
      case dtype
      when "float32-le", "float16-le", "bfloat16-le"
        dtype
      else
        invalid("unsupported floating-point tensor dtype: #{dtype}")
      end
    end

    private def self.element_count(shape : Array(Int64), field : String) : Int64
      expect(!shape.empty?, "#{field} cannot be empty")
      shape.reduce(1_i64) do |total, dimension|
        expect(dimension > 0, "#{field} dimensions must be positive")
        expect(total <= Int64::MAX // dimension, "#{field} element count overflows")
        total * dimension
      end
    end

    private def self.dimensions(value : JSON::Any, field : String) : Array(Int64)
      raw = value.as_a? || invalid("#{field} must be an array")
      raw.map_with_index do |dimension, index|
        positive = integer(dimension, "#{field}[#{index}]")
        expect(positive > 0, "#{field} dimensions must be positive")
        positive
      end
    end

    private def self.descriptor(descriptors : Hash(String, TensorDescriptor), name : String) : TensorDescriptor
      descriptors[name]? || invalid("missing tensor descriptor: #{name}")
    end

    private def self.required(value : JSON::Any, key : String) : JSON::Any
      fields = object(value, "manifest object")
      fields[key]? || invalid("missing manifest field: #{key}")
    end

    private def self.object(value : JSON::Any, field : String) : Hash(String, JSON::Any)
      value.as_h? || invalid("#{field} must be an object")
    end

    private def self.string(value : JSON::Any, field : String) : String
      value.as_s? || invalid("#{field} must be a string")
    end

    private def self.integer(value : JSON::Any, field : String) : Int64
      value.as_i64? || invalid("#{field} must be an integer")
    end

    private def self.boolean(value : JSON::Any, field : String) : Bool
      value.as_bool? || invalid("#{field} must be a boolean")
    end

    private def self.positive_int32(value : JSON::Any, field : String) : Int32
      parsed = integer(value, field)
      expect(parsed > 0 && parsed <= Int32::MAX, "#{field} out of range")
      parsed.to_i32
    end

    private def self.nonnegative_int32(value : JSON::Any, field : String) : Int32
      parsed = integer(value, field)
      expect(parsed >= 0 && parsed <= Int32::MAX, "#{field} out of range")
      parsed.to_i32
    end

    private def self.verify_sha256(data : Bytes, expected : String, field : String) : Nil
      expect(SHA256_RE.matches?(expected), "#{field} has an invalid SHA-256")
      actual = Digest::SHA256.hexdigest(data)
      expect(actual == expected.downcase, "#{field} checksum mismatch")
    end

    private def self.expect(condition : Bool, message : String) : Nil
      raise ArgumentError.new(message) unless condition
    end

    private def self.invalid(message : String) : NoReturn
      raise ArgumentError.new(message)
    end
  end
end
