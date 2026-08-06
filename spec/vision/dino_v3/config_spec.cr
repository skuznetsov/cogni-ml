require "json"

require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def dino_v3_config_fixture : String
  File.read(File.join(__DIR__, "../../fixtures/trellis2/dino_v3_config_certificate_v1.json"))
end

private def dino_v3_config_certificate : ML::Vision::DinoV3::ConfigCertificate
  ML::Vision::DinoV3::ConfigCertificate.parse(
    dino_v3_config_fixture,
    source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
    source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  )
end

describe ML::Vision::DinoV3::ConfigCertificate do
  it "certifies the exact pinned config without constructing a model block" do
    certificate = dino_v3_config_certificate

    certificate.source_model.should eq(
      "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    certificate.source_url.should eq(
      "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/ea8dc2863c51be0a264bab82070e3e8836b02d51/config.json"
    )
    certificate.license_name.should eq("dinov3-license")
    certificate.license_link.should eq(
      "https://ai.meta.com/resources/models-and-libraries/dinov3-license"
    )
    certificate.access_mode.should eq("manual")
    certificate.source_revision.should eq(
      "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    )
    certificate.config_sha256.should eq(
      "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
    )
    certificate.model_type.should eq("dinov3_vit")
    certificate.architectures.should contain("DINOv3ViTModel")
    certificate.patch_size.should eq(16_i64)
    certificate.num_channels.should eq(3_i64)
    certificate.image_size.should eq(224_i64)
    certificate.hidden_size.should eq(1024_i64)
    certificate.intermediate_size.should eq(4096_i64)
    certificate.num_hidden_layers.should eq(24_i64)
    certificate.num_attention_heads.should eq(16_i64)
    certificate.num_register_tokens.should eq(4_i64)
    certificate.head_dim.should eq(64_i64)
    certificate.rope_theta.should eq(100.0)
    certificate.layer_norm_eps.should eq(1.0e-5)
    certificate.initializer_range.should eq(0.02)
    certificate.layerscale_value.should eq(1.0)
    certificate.pos_embed_rescale.should eq(2.0)
    certificate.hidden_act.should eq("gelu")
    certificate.query_bias.should be_true
    certificate.key_bias.should be_false
    certificate.value_bias.should be_true
    certificate.proj_bias.should be_true
    certificate.mlp_bias.should be_true
    certificate.attention_dropout.should eq(0.0)
    certificate.drop_path_rate.should eq(0.0)
    certificate.use_gated_mlp.should be_false
    certificate.torch_dtype.should eq("float32")
    certificate.transformers_version.should eq("4.56.0.dev0")
    certificate.pos_embed_jitter.should be_nil
    certificate.pos_embed_shift.should be_nil

    certificate.fields.keys.should eq(
      [
        "architectures", "attention_dropout", "drop_path_rate", "hidden_act",
        "hidden_size", "image_size", "initializer_range", "intermediate_size",
        "key_bias", "layer_norm_eps", "layerscale_value", "mlp_bias", "model_type",
        "num_attention_heads", "num_channels", "num_hidden_layers",
        "num_register_tokens", "patch_size", "pos_embed_jitter", "pos_embed_rescale",
        "pos_embed_shift", "proj_bias", "query_bias", "rope_theta", "torch_dtype",
        "transformers_version", "use_gated_mlp", "value_bias",
      ]
    )
    certificate.fields["pos_embed_jitter"].raw.nil?.should be_true

    mutable_copy = certificate.fields
    mutable_copy["architectures"].as_a << JSON::Any.new("spoof")
    certificate.fields["architectures"].as_a.size.should eq(1)
    mutable_architectures = certificate.architectures
    mutable_architectures << "spoof"
    certificate.architectures.should eq(["DINOv3ViTModel"])
  end

  it "accepts bytes and rejects unknown, missing, or disallowed null fields" do
    source = dino_v3_config_fixture
    kwargs = {
      source_model:    ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
      source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION,
    }
    from_bytes = ML::Vision::DinoV3::ConfigCertificate.parse(source.to_slice, **kwargs)
    from_bytes.config_sha256.should eq(
      "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
    )

    changed_bytes = source.sub(/\n\z/, "\n\n")
    expect_raises(ML::Vision::DinoV3::ConfigError, /SHA-256/) do
      ML::Vision::DinoV3::ConfigCertificate.parse(changed_bytes, **kwargs)
    end

    expect_raises(ML::Vision::DinoV3::ConfigError, /source_revision/) do
      ML::Vision::DinoV3::ConfigCertificate.parse(
        source,
        source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
        source_revision: "0000000000000000000000000000000000000000"
      )
    end

    unknown = JSON.parse(source).as_h
    unknown["surprise"] = JSON::Any.new(true)
    expect_raises(ML::Vision::DinoV3::ConfigError, /unknown config key/) do
      ML::Vision::DinoV3::ConfigCertificate.parse(unknown.to_json, **kwargs)
    end

    missing = JSON.parse(source).as_h
    missing.delete("num_hidden_layers")
    expect_raises(ML::Vision::DinoV3::ConfigError, /missing config key/) do
      ML::Vision::DinoV3::ConfigCertificate.parse(missing.to_json, **kwargs)
    end

    disallowed_null = JSON.parse(source).as_h
    disallowed_null["hidden_size"] = JSON::Any.new(nil)
    expect_raises(ML::Vision::DinoV3::ConfigError, /hidden_size must be an integer/) do
      ML::Vision::DinoV3::ConfigCertificate.parse(disallowed_null.to_json, **kwargs)
    end

    execution = JSON.parse(source).as_h
    execution["training"] = JSON::Any.new(false)
    execution["attn_implementation"] = JSON::Any.new("eager")
    expect_raises(ML::Vision::DinoV3::ConfigError, /unknown config key/) do
      ML::Vision::DinoV3::ConfigCertificate.parse(execution.to_json, **kwargs)
    end

    duplicate = source.sub(/\}\n\z/, ",\n  \"hidden_size\": 1024\n}\n")
    expect_raises(ML::Vision::DinoV3::ConfigError, /duplicate JSON key/) do
      ML::Vision::DinoV3::ConfigCertificate.parse(duplicate, **kwargs)
    end

    trailing = source + "{}"
    expect_raises(ML::Vision::DinoV3::ConfigError, /invalid JSON/) do
      ML::Vision::DinoV3::ConfigCertificate.parse(trailing, **kwargs)
    end
  end

  it "keeps runtime and serialized layer paths as separate contracts" do
    adapter = ML::Vision::DinoV3::LayerPathAdapter.new("model.layer")
    adapter.runtime_path.should eq("model.layer")
    adapter.serialized_state_dict_prefix(23).should eq("model.layer.23")
    adapter.layer_index_from_serialized_prefix("model.layer.23").should eq(23_i32)

    expect_raises(ML::Vision::DinoV3::ConfigError, /root layer path/) do
      ML::Vision::DinoV3::LayerPathAdapter.new("layer")
    end
    expect_raises(ML::Vision::DinoV3::ConfigError, /runtime layer path/) do
      ML::Vision::DinoV3::LayerPathAdapter.new("model.model.layer")
    end
    expect_raises(ML::Vision::DinoV3::ConfigError, /serialized layer prefix/) do
      adapter.layer_index_from_serialized_prefix("model.layer")
    end
    expect_raises(ML::Vision::DinoV3::ConfigError, /serialized layer prefix/) do
      adapter.layer_index_from_serialized_prefix("model.layer.01")
    end
    expect_raises(ML::Vision::DinoV3::ConfigError, /layer index/) do
      adapter.serialized_state_dict_prefix(-1)
    end
  end
end
