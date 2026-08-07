require "./checkpoint_decoder"
require "./block"

module ML::Vision::DinoV3
  class BlockCallerError < BlockError
  end

  # A source-era layer caller for one DINOv3 block. It deliberately loads one
  # layer only; it is not a stack/encoder loader and does not create a model
  # graph or choose a device.
  class DinoV3BlockCaller
    MAX_PARAMETER_BYTES        = 256_i64 * 1024_i64 * 1024_i64
    TEMPORARY_ARRAY_MULTIPLIER = 3_i64
    DECODER_MAX_RESIDENT_BYTES = CheckpointF32Decoder::MAX_RESIDENT_BYTES

    REQUIRED_ROLES = [
      "norm1_weight",
      "norm1_bias",
      "query_weight",
      "query_bias",
      "key_weight",
      "value_weight",
      "value_bias",
      "output_weight",
      "output_bias",
      "layer_scale1",
      "norm2_weight",
      "norm2_bias",
      "up_weight",
      "up_bias",
      "down_weight",
      "down_bias",
      "layer_scale2",
    ] of String

    getter config : BlockConfig
    getter parameters : BlockParameters
    getter layer_index : Int32
    getter max_parameter_bytes : Int64

    def initialize(
      @parameters : BlockParameters,
      @layer_index : Int32,
      *,
      max_parameter_bytes : Int64 = MAX_PARAMETER_BYTES,
    )
      self.class.validate_budget!(max_parameter_bytes)
      @parameters.validate!
      required_bytes = @parameters.parameter_byte_length
      temporary_bytes = begin
        required_bytes * TEMPORARY_ARRAY_MULTIPLIER
      rescue OverflowError
        raise BlockCallerError.new("DINOv3 block temporary residency overflows Int64")
      end
      if temporary_bytes > max_parameter_bytes
        raise BlockCallerError.new(
          "DINOv3 block parameter budget #{max_parameter_bytes} is smaller than " \
          "estimated temporary residency #{temporary_bytes}"
        )
      end
      @max_parameter_bytes = max_parameter_bytes
      @config = @parameters.config
    end

    def self.from_checkpoint(
      inventory : CheckpointInventory,
      *,
      certificate : ConfigCertificate,
      runtime : RuntimeAdapter,
      semantic : SemanticInventory,
      digest : CheckpointDigestReceipt,
      layer_index : Int32,
      max_parameter_bytes : Int64 = MAX_PARAMETER_BYTES,
    ) : DinoV3BlockCaller
      self.validate_budget!(max_parameter_bytes)
      config = certified_block_config(certificate, runtime)
      runtime.serialized_state_dict_prefix(layer_index)
      checkpoint_prefix = CheckpointPathAdapter.new(certificate).serialized_state_dict_prefix(layer_index)
      expected_specs = SemanticInventory.expected_specs(certificate, runtime)
      expected_by_role = {} of String => SemanticTensorSpec
      expected_specs.each do |spec|
        if spec.role.starts_with?("block.")
          expected_by_role[spec.role] = spec
        end
      end

      role_specs = {
        "norm1_weight"  => "block.#{layer_index}.norm1_weight",
        "norm1_bias"    => "block.#{layer_index}.norm1_bias",
        "query_weight"  => "block.#{layer_index}.attention.query_weight",
        "query_bias"    => "block.#{layer_index}.attention.query_bias",
        "key_weight"    => "block.#{layer_index}.attention.key_weight",
        "value_weight"  => "block.#{layer_index}.attention.value_weight",
        "value_bias"    => "block.#{layer_index}.attention.value_bias",
        "output_weight" => "block.#{layer_index}.attention.output_weight",
        "output_bias"   => "block.#{layer_index}.attention.output_bias",
        "layer_scale1"  => "block.#{layer_index}.layer_scale1",
        "norm2_weight"  => "block.#{layer_index}.norm2_weight",
        "norm2_bias"    => "block.#{layer_index}.norm2_bias",
        "up_weight"     => "block.#{layer_index}.mlp.up_weight",
        "up_bias"       => "block.#{layer_index}.mlp.up_bias",
        "down_weight"   => "block.#{layer_index}.mlp.down_weight",
        "down_bias"     => "block.#{layer_index}.mlp.down_bias",
        "layer_scale2"  => "block.#{layer_index}.layer_scale2",
      } of String => String

      required_bytes = 0_i64
      role_specs.each do |local_name, role|
        expected = expected_by_role[role]?
        unless expected
          raise BlockCallerError.new("missing certified DINOv3 block role #{role.inspect}")
        end
        tensor = semantic.tensor_for(role)
        unless tensor.name == expected.name &&
               tensor.dtype == expected.dtype &&
               tensor.shape == expected.shape &&
               tensor.name.starts_with?("#{checkpoint_prefix}.")
          raise BlockCallerError.new(
            "DINOv3 block role #{role.inspect} does not match the certified #{checkpoint_prefix} tensor"
          )
        end
        required_bytes += tensor.data_bytes
        # Keep the local name live in the preflight loop so a future refactor
        # cannot silently drop a role from the explicit ordered load below.
        raise BlockCallerError.new("empty DINOv3 block role") if local_name.empty?
      end
      temporary_bytes = required_bytes * TEMPORARY_ARRAY_MULTIPLIER
      if temporary_bytes > max_parameter_bytes
        raise BlockCallerError.new(
          "DINOv3 block parameter budget #{max_parameter_bytes} is smaller than " \
          "estimated checkpoint residency #{temporary_bytes}"
        )
      end

      decoder = CheckpointF32Decoder.new(
        inventory,
        semantic: semantic,
        digest: digest,
        max_resident_bytes: DECODER_MAX_RESIDENT_BYTES
      )
      values = {} of String => Array(Float32)
      role_specs.each_key do |local_name|
        role = role_specs[local_name]
        values[local_name] = decoder.decode(role).values
      end
      parameters = BlockParameters.new(
        config,
        norm1_weight: values["norm1_weight"],
        norm1_bias: values["norm1_bias"],
        q_weight: values["query_weight"],
        q_bias: values["query_bias"],
        k_weight: values["key_weight"],
        v_weight: values["value_weight"],
        v_bias: values["value_bias"],
        o_weight: values["output_weight"],
        o_bias: values["output_bias"],
        layer_scale1: values["layer_scale1"],
        norm2_weight: values["norm2_weight"],
        norm2_bias: values["norm2_bias"],
        up_weight: values["up_weight"],
        up_bias: values["up_bias"],
        down_weight: values["down_weight"],
        down_bias: values["down_bias"],
        layer_scale2: values["layer_scale2"]
      )
      new(parameters, layer_index, max_parameter_bytes: max_parameter_bytes)
    rescue ex : OverflowError
      raise BlockCallerError.new("DINOv3 block checkpoint size overflows Int64")
    rescue ex : ConfigError
      raise BlockCallerError.new(ex.message)
    end

    {% if flag?(:dinov3_real_block_parity_probe) %}
      def forward_for_model_scale_probe(
        input : Tensor,
        rope_cos : Tensor,
        rope_sin : Tensor,
        *,
        max_multiply_adds : Int64,
      ) : BlockTrace
        BlockCPU.new(@parameters).forward_for_model_scale_probe(
          input,
          rope_cos,
          rope_sin,
          max_multiply_adds: max_multiply_adds
        )
      end
    {% end %}

    private def self.certified_block_config(
      certificate : ConfigCertificate,
      runtime : RuntimeAdapter,
    ) : BlockConfig
      SemanticInventory.expected_specs(certificate, runtime)
      BlockConfig.new(
        certificate.hidden_size.to_i32,
        certificate.intermediate_size.to_i32,
        certificate.num_attention_heads.to_i32,
        certificate.num_register_tokens.to_i32,
        certificate.layer_norm_eps.to_f32,
        hidden_act: certificate.hidden_act,
        query_bias: certificate.query_bias,
        key_bias: certificate.key_bias,
        value_bias: certificate.value_bias,
        proj_bias: certificate.proj_bias,
        mlp_bias: certificate.mlp_bias,
        attention_dropout: certificate.attention_dropout.to_f32,
        drop_path_rate: certificate.drop_path_rate.to_f32,
        use_gated_mlp: certificate.use_gated_mlp,
        attention_backend: "eager",
        training: false,
        extractor_final_layer_norm_eps: certificate.layer_norm_eps.to_f32
      )
    rescue ex : OverflowError
      raise BlockCallerError.new("certified DINOv3 block geometry exceeds Crystal Int32")
    end

    def self.validate_budget!(max_parameter_bytes : Int64) : Nil
      unless 0_i64 < max_parameter_bytes <= MAX_PARAMETER_BYTES
        raise BlockCallerError.new(
          "DINOv3 block parameter budget must be in 1..#{MAX_PARAMETER_BYTES}"
        )
      end
    end
  end
end
