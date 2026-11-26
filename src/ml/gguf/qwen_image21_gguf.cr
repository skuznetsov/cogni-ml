# Metadata-only validation and inventory for Qwen-Image 2.1 DiT GGUF files.
# This intentionally makes no image-quality or execution-readiness claim.

require "set"
require "./reader"

module ML::GGUF
  class QwenImage21GGUFInventory
    ARCHITECTURE      = "qwen_image"
    BLOCK_COUNT       = 32
    ORIG_SHAPE_PREFIX = "comfy.gguf.orig_shape."

    REQUIRED_TOP_LEVEL = [
      "img_in.weight",
      "modulation.1.weight",
      "norm_out.linear.weight",
      "proj_out.weight",
      "time_text_embed.timestep_embedder.linear_1.weight",
      "time_text_embed.timestep_embedder.linear_2.weight",
      "txt_in.in_layer.weight",
      "txt_in.out_layer.weight",
      "txt_in.text_norm.weight",
    ]

    REQUIRED_BLOCK_SUFFIXES = [
      "attn.norm_k.weight",
      "attn.norm_q.weight",
      "attn.to_k.weight",
      "attn.to_out.0.weight",
      "attn.to_q.weight",
      "attn.to_v.weight",
      "img_mlp.gate_up.weight",
      "img_mlp.out.weight",
    ]

    getter architecture : String
    getter block_count : Int32
    getter orig_shape_count : Int32
    getter type_counts : Hash(String, Int32)
    getter unsupported_type_labels : Array(String)
    getter total_tensor_bytes : Int64
    getter required_payload_bytes : Int64

    @tensors_by_name : Hash(String, TensorInfo)
    @logical_shapes : Hash(String, Array(Int64))

    def initialize(metadata : Hash(String, Value), tensors : Array(TensorInfo))
      @architecture = metadata["general.architecture"]?.as?(String) ||
                      raise ArgumentError.new("missing string metadata general.architecture")
      unless @architecture == ARCHITECTURE
        raise ArgumentError.new("expected architecture #{ARCHITECTURE}, got #{@architecture}")
      end

      @tensors_by_name = Hash(String, TensorInfo).new
      tensors.each do |tensor|
        if @tensors_by_name.has_key?(tensor.name)
          raise ArgumentError.new("duplicate tensor #{tensor.name}")
        end
        @tensors_by_name[tensor.name] = tensor
      end

      validate_required_tensors
      @block_count = validate_block_indices
      @logical_shapes = parse_original_shapes(metadata)
      @orig_shape_count = @logical_shapes.size

      @type_counts = Hash(String, Int32).new(0)
      unsupported = Set(String).new
      @total_tensor_bytes = 0_i64
      tensors.each do |tensor|
        label = tensor.type.name
        @type_counts[label] += 1
        unsupported << label unless reader_supported?(tensor.type)
        @total_tensor_bytes += tensor.data_bytes
      end
      @unsupported_type_labels = unsupported.to_a.sort!
      @required_payload_bytes = tensors.max_of? { |tensor| tensor.offset.to_i64 + tensor.data_bytes } || 0_i64
    end

    def self.from_file(file : GGUFFile) : self
      new(file.metadata, file.tensors)
    end

    def reader_compatible? : Bool
      @unsupported_type_labels.empty?
    end

    # Minimum complete-file size implied by the tensor directory. This is not
    # the sum of tensor sizes: GGUF offsets may contain alignment gaps.
    def required_file_bytes(data_offset : Int64) : Int64
      raise ArgumentError.new("data offset must be non-negative") if data_offset < 0
      data_offset + @required_payload_bytes
    end

    def tensor_data_complete?(data_offset : Int64, actual_file_bytes : Int64) : Bool
      actual_file_bytes >= required_file_bytes(data_offset)
    end

    def ensure_tensor_data_complete!(data_offset : Int64, actual_file_bytes : Int64) : Nil
      required = required_file_bytes(data_offset)
      return if actual_file_bytes >= required

      raise ArgumentError.new(
        "tensor payload is incomplete: file has #{actual_file_bytes} bytes, requires at least #{required}"
      )
    end

    # Source-framework shape when ComfyUI recorded a physical GGUF reshape;
    # otherwise the storage shape from the tensor directory.
    def logical_shape(name : String) : Array(Int64)
      if shape = @logical_shapes[name]?
        shape
      elsif tensor = @tensors_by_name[name]?
        tensor.dims
      else
        raise KeyError.new("unknown tensor #{name}")
      end
    end

    # QuantWeight uses mathematical [out, in] dimensions. Comfy's orig_shape
    # metadata records the source torch matrix in that order; ordinary GGUF
    # tensor-directory dimensions use [in, out].
    def projection_dims(name : String) : {Int32, Int32}
      tensor = @tensors_by_name[name]? || raise KeyError.new("unknown tensor #{name}")
      if source_shape = @logical_shapes[name]?
        unless source_shape.size == 2
          raise ArgumentError.new("original shape for #{name} must be a matrix, got #{source_shape}")
        end
        {source_shape[0].to_i32, source_shape[1].to_i32}
      else
        unless tensor.dims.size == 2
          raise ArgumentError.new("tensor #{name} must be a matrix, got #{tensor.dims}")
        end
        {tensor.dims[1].to_i32, tensor.dims[0].to_i32}
      end
    end

    private def validate_required_tensors
      REQUIRED_TOP_LEVEL.each do |name|
        require_tensor(name)
      end

      BLOCK_COUNT.times do |layer|
        REQUIRED_BLOCK_SUFFIXES.each do |suffix|
          require_tensor("transformer_blocks.#{layer}.#{suffix}")
        end
      end
    end

    private def require_tensor(name : String)
      unless @tensors_by_name.has_key?(name)
        raise ArgumentError.new("missing required tensor #{name}")
      end
    end

    private def validate_block_indices : Int32
      indices = Set(Int32).new
      @tensors_by_name.each_key do |name|
        if match = /\Atransformer_blocks\.(\d+)\./.match(name)
          indices << match[1].to_i
        end
      end

      expected = (0...BLOCK_COUNT).to_a
      actual = indices.to_a.sort!
      unless actual == expected
        raise ArgumentError.new("expected transformer block indices 0..#{BLOCK_COUNT - 1}, got #{actual}")
      end
      actual.size
    end

    private def parse_original_shapes(metadata : Hash(String, Value)) : Hash(String, Array(Int64))
      shapes = Hash(String, Array(Int64)).new
      metadata.each do |key, value|
        next unless key.starts_with?(ORIG_SHAPE_PREFIX)

        tensor_name = key[ORIG_SHAPE_PREFIX.size..]
        tensor = @tensors_by_name[tensor_name]? ||
                 raise ArgumentError.new("original shape references unknown tensor #{tensor_name}")
        raw_dims = value.as?(Array(Value)) ||
                   raise ArgumentError.new("#{key} must be an integer array")
        dims = raw_dims.map do |raw|
          dim = integer_value(raw) ||
                raise ArgumentError.new("#{key} must contain only integers")
          raise ArgumentError.new("#{key} dimensions must be positive") unless dim > 0
          dim
        end
        raise ArgumentError.new("#{key} must not be empty") if dims.empty?

        logical_elements = dims.reduce(1_i64) { |product, dim| product * dim }
        unless logical_elements == tensor.n_elements
          raise ArgumentError.new(
            "#{key} element count #{logical_elements} does not match storage element count #{tensor.n_elements}"
          )
        end
        shapes[tensor_name] = dims
      end
      shapes
    end

    private def integer_value(value : Value) : Int64?
      case value
      when Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64 then value.to_i64
      when UInt64                                           then value.to_i64
      else                                                       nil
      end
    end

    # This is deliberately narrower than the set of types whose byte layout is
    # known by TensorType: it means the current CPU dequant layer can read it.
    private def reader_supported?(type : TensorType) : Bool
      case type
      when .f32?, .f16?, .bf16?, .q4_k?, .q5_k?, .q6_k?, .q8_0?, .iq4_nl?
        true
      else
        false
      end
    end
  end
end
