require "./checkpoint_inventory"
require "./runtime"

module ML::Vision::DinoV3
  # A semantic inventory binds the source-backed Transformers module layout to
  # the already parsed safetensors header. It never reads tensor payload bytes
  # and it is not a loader or an encoder.
  class SemanticInventoryError < CheckpointInventoryError
  end

  struct SemanticTensorSpec
    getter role : String
    getter name : String
    getter dtype : String

    @shape : Array(Int64)

    def initialize(
      role : String,
      name : String,
      dtype : String,
      shape : Array(Int64),
    )
      @role = role.dup
      @name = name.dup
      @dtype = dtype.dup
      @shape = shape.dup
    end

    def shape : Array(Int64)
      @shape.dup
    end
  end

  struct SemanticTensorBinding
    getter spec : SemanticTensorSpec
    getter tensor : CheckpointTensor

    def initialize(@spec : SemanticTensorSpec, @tensor : CheckpointTensor)
    end
  end

  class SemanticInventory
    SCHEMA = "cogni-ml/vision/dino-v3/semantic-inventory/v1"

    getter schema : String
    getter tensor_count : Int32

    @bindings : Array(SemanticTensorBinding)
    @roles : Hash(String, CheckpointTensor)

    private def initialize(bindings : Array(SemanticTensorBinding))
      @schema = SCHEMA.dup
      @bindings = bindings.dup
      @tensor_count = bindings.size.to_i32
      @roles = {} of String => CheckpointTensor
      bindings.each do |binding|
        @roles[binding.spec.role] = binding.tensor
      end
    end

    # Generate the exact state-dict role schema from the pinned HF DINOv3 ViT
    # source. The schema is source evidence only until a real gated header is
    # materialized and bound by #bind.
    def self.expected_specs(
      certificate : ConfigCertificate,
      runtime : RuntimeAdapter,
    ) : Array(SemanticTensorSpec)
      validate_provenance!(certificate, runtime)

      hidden = certificate.hidden_size
      intermediate = certificate.intermediate_size
      specs = [] of SemanticTensorSpec
      checkpoint_paths = CheckpointPathAdapter.new(certificate)

      append_spec!(specs, "embedding.cls_token", "embeddings.cls_token", [1_i64, 1_i64, hidden])
      append_spec!(specs, "embedding.mask_token", "embeddings.mask_token", [1_i64, 1_i64, hidden])
      append_spec!(
        specs,
        "embedding.register_tokens",
        "embeddings.register_tokens",
        [1_i64, certificate.num_register_tokens, hidden]
      )
      append_spec!(
        specs,
        "embedding.patch_weight",
        "embeddings.patch_embeddings.weight",
        [hidden, certificate.num_channels, certificate.patch_size, certificate.patch_size]
      )
      append_spec!(
        specs,
        "embedding.patch_bias",
        "embeddings.patch_embeddings.bias",
        [hidden]
      )

      depth = certificate.num_hidden_layers.to_i32
      (0...depth).each do |layer_index|
        prefix = checkpoint_paths.serialized_state_dict_prefix(layer_index)
        role_prefix = "block.#{layer_index}"
        append_spec!(specs, "#{role_prefix}.norm1_weight", "#{prefix}.norm1.weight", [hidden])
        append_spec!(specs, "#{role_prefix}.norm1_bias", "#{prefix}.norm1.bias", [hidden])
        append_spec!(
          specs,
          "#{role_prefix}.attention.key_weight",
          "#{prefix}.attention.k_proj.weight",
          [hidden, hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.attention.value_weight",
          "#{prefix}.attention.v_proj.weight",
          [hidden, hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.attention.value_bias",
          "#{prefix}.attention.v_proj.bias",
          [hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.attention.query_weight",
          "#{prefix}.attention.q_proj.weight",
          [hidden, hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.attention.query_bias",
          "#{prefix}.attention.q_proj.bias",
          [hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.attention.output_weight",
          "#{prefix}.attention.o_proj.weight",
          [hidden, hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.attention.output_bias",
          "#{prefix}.attention.o_proj.bias",
          [hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.layer_scale1",
          "#{prefix}.layer_scale1.lambda1",
          [hidden]
        )
        append_spec!(specs, "#{role_prefix}.norm2_weight", "#{prefix}.norm2.weight", [hidden])
        append_spec!(specs, "#{role_prefix}.norm2_bias", "#{prefix}.norm2.bias", [hidden])
        append_spec!(
          specs,
          "#{role_prefix}.mlp.up_weight",
          "#{prefix}.mlp.up_proj.weight",
          [intermediate, hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.mlp.up_bias",
          "#{prefix}.mlp.up_proj.bias",
          [intermediate]
        )
        append_spec!(
          specs,
          "#{role_prefix}.mlp.down_weight",
          "#{prefix}.mlp.down_proj.weight",
          [hidden, intermediate]
        )
        append_spec!(
          specs,
          "#{role_prefix}.mlp.down_bias",
          "#{prefix}.mlp.down_proj.bias",
          [hidden]
        )
        append_spec!(
          specs,
          "#{role_prefix}.layer_scale2",
          "#{prefix}.layer_scale2.lambda1",
          [hidden]
        )
      end

      append_spec!(specs, "final_norm.weight", "norm.weight", [hidden])
      append_spec!(specs, "final_norm.bias", "norm.bias", [hidden])
      specs
    end

    # Bind the source-backed schema to a parsed header. No payload byte is
    # touched; all checks are names, dtypes, shapes, and bounded provenance.
    def self.bind(
      inventory : CheckpointInventory,
      *,
      certificate : ConfigCertificate,
      runtime : RuntimeAdapter,
    ) : SemanticInventory
      specs = expected_specs(certificate, runtime)
      unless inventory.file_byte_length == CheckpointManifest::PINNED_WEIGHTS_BYTE_LENGTH
        raise SemanticInventoryError.new(
          "semantic inventory requires the pinned checkpoint byte length"
        )
      end

      expected = {} of String => SemanticTensorSpec
      specs.each { |spec| expected[spec.name] = spec }
      actual = {} of String => CheckpointTensor
      inventory.tensors.each do |tensor|
        if actual.has_key?(tensor.name)
          raise SemanticInventoryError.new(
            "duplicate DINOv3 tensor name #{tensor.name.inspect}"
          )
        end
        actual[tensor.name] = tensor
      end

      actual.each_key do |name|
        unless expected.has_key?(name)
          raise SemanticInventoryError.new(
            "unexpected DINOv3 tensor #{name.inspect}"
          )
        end
      end
      expected.each_key do |name|
        unless actual.has_key?(name)
          raise SemanticInventoryError.new(
            "missing DINOv3 tensor #{name.inspect}"
          )
        end
      end

      bindings = specs.map do |spec|
        tensor = actual[spec.name]
        unless tensor.dtype == spec.dtype
          raise SemanticInventoryError.new(
            "DINOv3 tensor #{spec.name.inspect} dtype #{tensor.dtype.inspect} " \
            "does not match expected #{spec.dtype.inspect}"
          )
        end
        unless tensor.shape == spec.shape
          raise SemanticInventoryError.new(
            "DINOv3 tensor #{spec.name.inspect} shape #{tensor.shape.inspect} " \
            "does not match expected #{spec.shape.inspect}"
          )
        end
        SemanticTensorBinding.new(spec, tensor)
      end

      new(bindings)
    end

    def bindings : Array(SemanticTensorBinding)
      @bindings.dup
    end

    def tensor_for(role : String) : CheckpointTensor
      tensor = @roles[role]?
      unless tensor
        raise SemanticInventoryError.new("unknown DINOv3 semantic role #{role.inspect}")
      end
      tensor
    end

    private def self.append_spec!(
      specs : Array(SemanticTensorSpec),
      role : String,
      name : String,
      shape : Array(Int64),
    ) : Nil
      specs << SemanticTensorSpec.new(role, name, "F32", shape)
    end

    private def self.validate_provenance!(
      certificate : ConfigCertificate,
      runtime : RuntimeAdapter,
    ) : Nil
      unless certificate.source_model == ConfigCertificate::PINNED_SOURCE_MODEL &&
             certificate.source_revision == ConfigCertificate::PINNED_SOURCE_REVISION &&
             certificate.config_sha256 == ConfigCertificate::PINNED_CONFIG_SHA256 &&
             certificate.torch_dtype == "float32" &&
             certificate.image_size == 224_i64 &&
             certificate.patch_size == 16_i64 &&
             certificate.num_channels == 3_i64 &&
             certificate.hidden_size == 1024_i64 &&
             certificate.intermediate_size == 4096_i64 &&
             certificate.num_hidden_layers == 24_i64 &&
             certificate.num_attention_heads == 16_i64 &&
             certificate.num_register_tokens == 4_i64
        raise SemanticInventoryError.new(
          "semantic inventory requires the pinned DINOv3 L/16 configuration"
        )
      end
      unless runtime.certificate.source_model == certificate.source_model &&
             runtime.certificate.source_revision == certificate.source_revision &&
             runtime.certificate.config_sha256 == certificate.config_sha256 &&
             runtime.depth == certificate.num_hidden_layers.to_i32
        raise SemanticInventoryError.new(
          "semantic inventory runtime adapter does not match the config certificate"
        )
      end
    end
  end
end
