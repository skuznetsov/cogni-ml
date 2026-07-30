require "../../spec_helper"
require "file_utils"
require "../../../src/ml/three_d/trellis2"
require "./support"

private def expect_manifest_error(json : String, pattern : Regex)
  expect_raises(ML::ThreeD::Trellis2::ManifestError, pattern) do
    ML::ThreeD::Trellis2::Manifest.parse(json)
  end
end

describe ML::ThreeD::Trellis2::Manifest do
  it "bounds strict JSON nesting before schema validation" do
    nested = "[" * 65 + "0" + "]" * 65
    expect_raises(
      ML::ThreeD::Trellis2::StrictJSONError,
      /nesting depth/
    ) do
      ML::ThreeD::Trellis2::StrictJSON.parse(nested)
    end
  end

  it "parses a sealed manifest and canonicalizes root key order" do
    json = Trellis2SpecSupport.sealed_manifest_json
    manifest = ML::ThreeD::Trellis2::Manifest.parse(json)

    manifest.schema_version.should eq(1)
    manifest.execution_order.should eq(["dino_v3"])
    manifest.stages.first.id.should eq("dino_v3")
    manifest.stages.first.tensors.first.dtype.should eq("F16")

    root = JSON.parse(json).as_h
    reordered = {} of String => JSON::Any
    root.to_a.reverse_each { |key, value| reordered[key] = value }
    reparsed = ML::ThreeD::Trellis2::Manifest.parse(reordered.to_json)
    reparsed.canonical_identity_json.should eq(manifest.canonical_identity_json)
    reparsed.computed_pack_id.should eq(manifest.computed_pack_id)
  end

  it "uses domain-separated length framing for the pack identity" do
    draft = Trellis2SpecSupport.manifest_json
    manifest = ML::ThreeD::Trellis2::Manifest.parse(
      Trellis2SpecSupport.seal_manifest_json(draft)
    )

    manifest.computed_pack_id.should eq(
      Trellis2SpecSupport.independently_framed_pack_id(manifest)
    )
  end

  it "rejects duplicate and unknown keys at every parsed boundary" do
    duplicate = Trellis2SpecSupport.manifest_json.sub(
      %("schema_version":1),
      %("schema_version":1,"schema_version":1)
    )
    expect_manifest_error(duplicate, /duplicate JSON key.*schema_version/)

    unknown_root = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["surprise"] = JSON::Any.new(true)
    end
    expect_manifest_error(unknown_root, /unknown manifest key.*surprise/)

    unknown_stage = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["stages"].as_a.first.as_h["surprise"] = JSON::Any.new(true)
    end
    expect_manifest_error(unknown_stage, /unknown stage key.*surprise/)

    unknown_tensor = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["stages"].as_a.first.as_h["tensors"].as_a.first.as_h["surprise"] = JSON::Any.new(true)
    end
    expect_manifest_error(unknown_tensor, /unknown tensor key.*surprise/)
  end

  it "rejects unknown versions, stages, dtypes, layouts, and byte order" do
    bad_version = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["schema_version"] = JSON::Any.new(2_i64)
    end
    expect_manifest_error(bad_version, /unsupported schema_version/)

    {
      "id"         => {"mystery_stage", /unsupported stage/},
      "dtype"      => {"F8", /unsupported dtype/},
      "layout"     => {"magic_pack", /unsupported layout/},
      "byte_order" => {"native", /unsupported byte_order/},
    }.each do |field, (value, pattern)|
      bad = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
        if field == "id"
          root["stages"].as_a.first.as_h[field] = JSON::Any.new(value)
          root["execution_order"].as_a[0] = JSON::Any.new(value)
        else
          root["stages"].as_a.first.as_h["tensors"].as_a.first.as_h[field] = JSON::Any.new(value)
        end
      end
      expect_manifest_error(bad, pattern)
    end
  end

  it "rejects duplicate identities and inconsistent execution order" do
    duplicate_stage = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["stages"].as_a << root["stages"].as_a.first
      root["execution_order"].as_a << JSON::Any.new("dino_v3")
    end
    expect_manifest_error(duplicate_stage, /duplicate stage/)

    tensors = [
      Trellis2SpecSupport.tensor_json,
      Trellis2SpecSupport.tensor_json(source_name: "other.weight"),
    ]
    duplicate_destination = Trellis2SpecSupport.manifest_json(tensors: tensors)
    expect_manifest_error(duplicate_destination, /duplicate destination tensor/)

    wrong_order = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["execution_order"].as_a.clear
    end
    expect_manifest_error(wrong_order, /execution_order/)
  end

  it "rejects mutable revisions, unsafe paths, invalid hashes, and shape overflow" do
    mutable_revision = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["model"].as_h["revision"] = JSON::Any.new("main")
    end
    expect_manifest_error(mutable_revision, /immutable revision/)

    ["/tmp/model.safetensors", "../model.safetensors", "stage/../model.safetensors", "stage\\model.safetensors"].each do |path|
      unsafe_path = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
        root["files"].as_a.first.as_h["path"] = JSON::Any.new(path)
      end
      expect_manifest_error(unsafe_path, /unsafe relative path/)
    end

    bad_hash = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["converter_config_sha256"] = JSON::Any.new("abc")
    end
    expect_manifest_error(bad_hash, /SHA-256/)

    overflow_shape = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["stages"].as_a.first.as_h["tensors"].as_a.first.as_h["shape"] =
        JSON::Any.new([
          JSON::Any.new(Int64::MAX),
          JSON::Any.new(2_i64),
        ])
    end
    expect_manifest_error(overflow_shape, /shape.*overflow/)
  end

  it "fails closed on a mismatched pack id" do
    expect_manifest_error(Trellis2SpecSupport.manifest_json, /pack_id mismatch/)
  end

  it "admits only explicit immutable external dependency records" do
    dependency = {
      "id"          => JSON::Any.new("dinov3"),
      "repository"  => JSON::Any.new("facebook/dinov3-vitl16-pretrain-lvd1689m"),
      "revision"    => JSON::Any.new("3" * 40),
      "path"        => JSON::Any.new("model.safetensors"),
      "byte_length" => JSON::Any.new(128_i64),
      "sha256"      => JSON::Any.new("d" * 64),
      "license"     => JSON::Any.new("DINOv3"),
    }
    explicit = Trellis2SpecSupport.mutate(Trellis2SpecSupport.manifest_json) do |root|
      root["external_dependencies"] = JSON::Any.new([JSON::Any.new(dependency)])
    end

    manifest = ML::ThreeD::Trellis2::Manifest.parse(
      Trellis2SpecSupport.seal_manifest_json(explicit)
    )
    manifest.external_dependencies.map(&.id).should eq(["dinov3"])

    mutable = Trellis2SpecSupport.mutate(explicit) do |root|
      root["external_dependencies"].as_a.first.as_h["revision"] =
        JSON::Any.new("main")
    end
    expect_manifest_error(mutable, /external_dependency.*immutable revision/)
  end

  it "represents safetensors scalar and empty tensor shapes" do
    tensors = [
      Trellis2SpecSupport.tensor_json(
        source_name: "scalar",
        destination_name: "dino_v3.scalar",
        shape: [] of Int64,
        data_offsets: [0_i64, 2_i64]
      ),
      Trellis2SpecSupport.tensor_json(
        source_name: "empty",
        destination_name: "dino_v3.empty",
        shape: [0_i64, 4_i64],
        data_offsets: [2_i64, 2_i64]
      ),
    ]

    manifest = ML::ThreeD::Trellis2::Manifest.parse(
      Trellis2SpecSupport.seal_manifest_json(
        Trellis2SpecSupport.manifest_json(tensors: tensors)
      )
    )
    manifest.stages.first.tensors.map(&.shape).should eq([
      [] of Int64,
      [0_i64, 4_i64],
    ])
  end

  it "rejects symlink pack roots and symlinked ancestors" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      real_parent = File.join(dir, "real")
      pack = File.join(real_parent, "pack")
      Dir.mkdir(real_parent)
      Dir.mkdir(pack)
      File.write(
        File.join(pack, "manifest.json"),
        Trellis2SpecSupport.sealed_manifest_json
      )

      root_alias = File.join(dir, "pack-link")
      File.symlink(pack, root_alias)
      expect_raises(ML::ThreeD::Trellis2::ManifestError, /symlink/) do
        ML::ThreeD::Trellis2::Manifest.load(root_alias + "/")
      end

      ancestor_alias = File.join(dir, "ancestor-link")
      File.symlink(real_parent, ancestor_alias)
      expect_raises(ML::ThreeD::Trellis2::ManifestError, /symlink/) do
        ML::ThreeD::Trellis2::Manifest.load(
          File.join(ancestor_alias, "pack")
        )
      end
    end
  end

  it "binds standalone pack validation to the manifest stored at the root" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      Trellis2SpecSupport.write_valid_pack(dir)
      manifest_path = File.join(dir, "manifest.json")
      original_json = File.read(manifest_path)
      manifest = ML::ThreeD::Trellis2::Manifest.parse(original_json)

      File.delete(manifest_path)
      expect_raises(
        ML::ThreeD::Trellis2::ManifestError,
        /cannot load manifest/
      ) do
        manifest.validate_pack!(dir)
      end

      different_json = Trellis2SpecSupport.mutate(original_json) do |root|
        root["converter_version"] = JSON::Any.new("0.1.1")
      end
      File.write(
        manifest_path,
        Trellis2SpecSupport.seal_manifest_json(different_json)
      )
      expect_raises(
        ML::ThreeD::Trellis2::ManifestError,
        /root manifest does not match/
      ) do
        manifest.validate_pack!(dir)
      end
    end
  end
end
