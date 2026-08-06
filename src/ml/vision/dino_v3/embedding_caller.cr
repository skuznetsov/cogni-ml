require "./checkpoint_decoder"
require "./embeddings"

module ML::Vision::DinoV3
  class EmbeddingCallerError < EmbeddingError
  end

  # A bounded caller for the real DINOv3 embedding boundary. The checkpoint
  # constructor decodes only the four roles needed before transformer blocks;
  # it never loads block or final-layer weights and never executes attention.
  class DinoV3EmbeddingCaller
    MAX_PARAMETER_BYTES        = 16_i64 * 1024_i64 * 1024_i64
    TEMPORARY_ARRAY_MULTIPLIER = 3_i64

    REQUIRED_ROLES = {
      "embedding.patch_weight",
      "embedding.patch_bias",
      "embedding.cls_token",
      "embedding.register_tokens",
    }

    getter parameters : EmbeddingParameters
    getter max_parameter_bytes : Int64

    def initialize(
      @parameters : EmbeddingParameters,
      *,
      max_parameter_bytes : Int64 = MAX_PARAMETER_BYTES,
    )
      validate_budget!(max_parameter_bytes)
      @parameters.validate!
      parameter_bytes = @parameters.parameter_byte_length
      if parameter_bytes * TEMPORARY_ARRAY_MULTIPLIER > max_parameter_bytes
        raise EmbeddingCallerError.new(
          "DINOv3 embedding caller parameter byte budget #{max_parameter_bytes} " \
          "is smaller than required temporary residency #{parameter_bytes * TEMPORARY_ARRAY_MULTIPLIER}"
        )
      end
      @max_parameter_bytes = max_parameter_bytes
      @embedding = EmbeddingCPU.new(@parameters)
    end

    def self.from_checkpoint(
      inventory : CheckpointInventory,
      *,
      certificate : ConfigCertificate,
      runtime : RuntimeAdapter,
      semantic : SemanticInventory,
      digest : CheckpointDigestReceipt,
      max_parameter_bytes : Int64 = MAX_PARAMETER_BYTES,
    ) : DinoV3EmbeddingCaller
      config = certified_embedding_config(certificate, runtime)
      expected_specs = SemanticInventory.expected_specs(certificate, runtime)
      expected_by_role = {} of String => SemanticTensorSpec
      expected_specs.each { |spec| expected_by_role[spec.role] = spec }

      required_bytes = 0_i64
      REQUIRED_ROLES.each do |role|
        expected = expected_by_role[role]
        tensor = semantic.tensor_for(role)
        unless tensor.name == expected.name &&
               tensor.dtype == expected.dtype &&
               tensor.shape == expected.shape
          raise EmbeddingCallerError.new(
            "DINOv3 embedding role #{role.inspect} does not match the certified tensor"
          )
        end
        required_bytes += tensor.data_bytes
      end
      validate_budget!(max_parameter_bytes)
      temporary_bytes = required_bytes * TEMPORARY_ARRAY_MULTIPLIER
      if temporary_bytes > max_parameter_bytes
        raise EmbeddingCallerError.new(
          "DINOv3 embedding caller parameter byte budget #{max_parameter_bytes} " \
          "is smaller than required temporary residency #{temporary_bytes}"
        )
      end

      decoder = CheckpointF32Decoder.new(
        inventory,
        semantic: semantic,
        digest: digest,
        max_resident_bytes: max_parameter_bytes
      )
      patch_weight = decoder.decode("embedding.patch_weight").values
      patch_bias = decoder.decode("embedding.patch_bias").values
      cls_token = decoder.decode("embedding.cls_token").values
      register_tokens = decoder.decode("embedding.register_tokens").values
      parameters = EmbeddingParameters.new(
        config,
        patch_weight,
        patch_bias,
        cls_token,
        register_tokens
      )
      new(parameters, max_parameter_bytes: max_parameter_bytes)
    end

    def config : EmbeddingConfig
      @parameters.config
    end

    def forward(input : Tensor) : EmbeddingResult
      @embedding.forward(input)
    end

    private def self.certified_embedding_config(
      certificate : ConfigCertificate,
      runtime : RuntimeAdapter,
    ) : EmbeddingConfig
      # expected_specs performs the complete pinned certificate/runtime check;
      # the result is intentionally discarded after it establishes provenance.
      SemanticInventory.expected_specs(certificate, runtime)
      EmbeddingConfig.new(
        certificate.patch_size.to_i32,
        certificate.num_channels.to_i32,
        certificate.hidden_size.to_i32,
        certificate.num_attention_heads.to_i32,
        certificate.num_register_tokens.to_i32,
        certificate.rope_theta.to_f32
      )
    rescue ex : OverflowError
      raise EmbeddingCallerError.new(
        "certified DINOv3 embedding geometry exceeds Crystal Int32"
      )
    rescue ex : EmbeddingError
      raise ex
    rescue ex : ConfigError
      raise EmbeddingCallerError.new(ex.message)
    end

    def self.validate_budget!(max_parameter_bytes : Int64) : Nil
      unless 0_i64 < max_parameter_bytes <= MAX_PARAMETER_BYTES
        raise EmbeddingCallerError.new(
          "DINOv3 embedding caller parameter byte budget must be in 1..#{MAX_PARAMETER_BYTES}"
        )
      end
    end

    private def validate_budget!(max_parameter_bytes : Int64) : Nil
      DinoV3EmbeddingCaller.validate_budget!(max_parameter_bytes)
    end
  end
end
