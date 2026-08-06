require "json"

require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def dino_v3_runtime_certificate : ML::Vision::DinoV3::ConfigCertificate
  source = File.read(
    File.join(__DIR__, "../../fixtures/trellis2/dino_v3_config_certificate_v1.json")
  )
  ML::Vision::DinoV3::ConfigCertificate.parse(
    source,
    source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
    source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  )
end

private def dino_v3_runtime_fixture : JSON::Any
  JSON.parse(
    File.read(File.join(__DIR__, "../../fixtures/trellis2/dino_v3_embeddings_cpu_v1.json"))
  )
end

describe ML::Vision::DinoV3::RuntimeAdapter do
  it "binds source, instance, and serialized paths to the pinned runtime evidence" do
    fixture = dino_v3_runtime_fixture
    provenance = fixture["provenance"]
    compatibility = fixture["compatibility"]
    adapter = ML::Vision::DinoV3::RuntimeAdapter.new(
      dino_v3_runtime_certificate,
      trellis_source_revision: provenance["commit"].as_s,
      transformers_version: provenance["transformers_version"].as_s,
      transformers_modeling_sha256: provenance["transformers_modeling_sha256"].as_s,
      transformers_config_sha256: provenance["transformers_config_sha256"].as_s,
      source_runtime_path: compatibility["pinned_upstream_layer_path"].as_s,
      instance_runtime_path: compatibility["transformers_5_8_1_layer_path"].as_s,
      source_path_available: compatibility["pinned_upstream_path_available_on_instance"].as_bool,
      instance_path_available: compatibility["transformers_5_8_1_path_available_on_instance"].as_bool
    )

    adapter.trellis_source_revision.should eq(provenance["commit"].as_s)
    adapter.transformers_version.should eq("5.8.1")
    adapter.transformers_modeling_sha256.should eq(
      provenance["transformers_modeling_sha256"].as_s
    )
    adapter.transformers_config_sha256.should eq(
      provenance["transformers_config_sha256"].as_s
    )
    adapter.source_runtime_path.should eq("model.layer")
    adapter.instance_runtime_path.should eq("model.model.layer")
    adapter.source_path_available?.should be_false
    adapter.instance_path_available?.should be_true
    adapter.depth.should eq(24)
    adapter.serialized_state_dict_prefix(0).should eq("model.layer.0")
    adapter.serialized_state_dict_prefix(23).should eq("model.layer.23")
    adapter.layer_index_from_serialized_prefix("model.layer.23").should eq(23)
  end

  it "rejects path drift instead of normalizing a different runtime context" do
    certificate = dino_v3_runtime_certificate

    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /source runtime path/) do
      ML::Vision::DinoV3::RuntimeAdapter.new(
        certificate,
        source_runtime_path: "model.model.layer"
      )
    end
    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /instance runtime path/) do
      ML::Vision::DinoV3::RuntimeAdapter.new(
        certificate,
        instance_runtime_path: "model.layer"
      )
    end
    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /source path availability/) do
      ML::Vision::DinoV3::RuntimeAdapter.new(
        certificate,
        source_path_available: true
      )
    end
    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /instance path availability/) do
      ML::Vision::DinoV3::RuntimeAdapter.new(
        certificate,
        instance_path_available: false
      )
    end
  end

  it "rejects provenance drift instead of accepting a stale source certificate" do
    certificate = dino_v3_runtime_certificate

    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /source revision/) do
      ML::Vision::DinoV3::RuntimeAdapter.new(
        certificate,
        trellis_source_revision: "0000000000000000000000000000000000000000"
      )
    end
    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /runtime version/) do
      ML::Vision::DinoV3::RuntimeAdapter.new(
        certificate,
        transformers_version: "5.8.0"
      )
    end
    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /modeling source digest/) do
      ML::Vision::DinoV3::RuntimeAdapter.new(
        certificate,
        transformers_modeling_sha256: "0" * 64
      )
    end
    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /config source digest/) do
      ML::Vision::DinoV3::RuntimeAdapter.new(
        certificate,
        transformers_config_sha256: "0" * 64
      )
    end
  end

  it "keeps serialized layer indices bounded by the certified model depth" do
    adapter = ML::Vision::DinoV3::RuntimeAdapter.new(dino_v3_runtime_certificate)

    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /depth/) do
      adapter.serialized_state_dict_prefix(24)
    end
    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /depth/) do
      adapter.layer_index_from_serialized_prefix("model.layer.24")
    end
    expect_raises(ML::Vision::DinoV3::RuntimeAdapterError, /serialized layer prefix/) do
      adapter.layer_index_from_serialized_prefix("model.layer.01")
    end
  end
end
