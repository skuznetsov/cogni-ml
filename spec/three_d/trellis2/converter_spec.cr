require "../../spec_helper"
require "file_utils"
require "../../../src/ml/three_d/trellis2/converter"
require "./support"

private alias SourceTensor = NamedTuple(
  name: String,
  shape: Array(Int64),
  values: Array(UInt16))

private def write_source_safetensors(
  path : String,
  tensors : Array(SourceTensor),
) : Nil
  offset = 0_i64
  payload = IO::Memory.new
  entries = [] of {SourceTensor, Int64, Int64}
  tensors.each do |tensor|
    start_offset = offset
    tensor[:values].each do |value|
      payload.write_bytes(value, IO::ByteFormat::LittleEndian)
      offset += 2
    end
    entries << {tensor, start_offset, offset}
  end

  header = JSON.build do |json|
    json.object do
      entries.each do |tensor, start_offset, end_offset|
        json.field tensor[:name] do
          json.object do
            json.field "dtype", "F16"
            json.field "shape" do
              json.array do
                tensor[:shape].each { |dimension| json.number(dimension) }
              end
            end
            json.field "data_offsets" do
              json.array do
                json.number start_offset
                json.number end_offset
              end
            end
          end
        end
      end
    end
  end
  padding = (8 - header.bytesize % 8) % 8
  Trellis2SpecSupport.write_safetensors(
    path,
    header + " " * padding,
    payload.to_slice
  )
end

private def source_tensors : Array(SourceTensor)
  [
    {name: "identity", shape: [2_i64], values: [10_u16, 11_u16]},
    {name: "matrix", shape: [2_i64, 3_i64], values: (0_u16..5_u16).to_a},
    {name: "cube", shape: [2_i64, 2_i64, 3_i64], values: (20_u16..31_u16).to_a},
    {name: "wide", shape: [2_i64, 4_i64], values: (40_u16..47_u16).to_a},
    {name: "concat_a", shape: [2_i64, 1_i64], values: [50_u16, 51_u16]},
    {name: "concat_b", shape: [2_i64, 2_i64], values: [52_u16, 53_u16, 54_u16, 55_u16]},
  ]
end

private def operation_specs : Array(Hash(String, JSON::Any))
  [
    {
      "id"          => JSON::Any.new("identity_op"),
      "kind"        => JSON::Any.new("identity"),
      "source"      => JSON::Any.new("identity"),
      "destination" => JSON::Any.new("identity_out"),
    },
    {
      "id"          => JSON::Any.new("transpose_op"),
      "kind"        => JSON::Any.new("transpose"),
      "source"      => JSON::Any.new("matrix"),
      "destination" => JSON::Any.new("matrix_t"),
      "axes"        => JSON::Any.new([JSON::Any.new(0_i64), JSON::Any.new(1_i64)]),
    },
    {
      "id"          => JSON::Any.new("permute_op"),
      "kind"        => JSON::Any.new("permute"),
      "source"      => JSON::Any.new("cube"),
      "destination" => JSON::Any.new("cube_p"),
      "axes"        => JSON::Any.new([
        JSON::Any.new(2_i64),
        JSON::Any.new(0_i64),
        JSON::Any.new(1_i64),
      ]),
    },
    {
      "id"           => JSON::Any.new("split_op"),
      "kind"         => JSON::Any.new("split"),
      "source"       => JSON::Any.new("wide"),
      "destinations" => JSON::Any.new([
        JSON::Any.new("wide_left"),
        JSON::Any.new("wide_right"),
      ]),
      "axis"  => JSON::Any.new(1_i64),
      "sizes" => JSON::Any.new([
        JSON::Any.new(1_i64),
        JSON::Any.new(3_i64),
      ]),
    },
    {
      "id"      => JSON::Any.new("concat_op"),
      "kind"    => JSON::Any.new("concat"),
      "sources" => JSON::Any.new([
        JSON::Any.new("concat_a"),
        JSON::Any.new("concat_b"),
      ]),
      "destination" => JSON::Any.new("concat_out"),
      "axis"        => JSON::Any.new(1_i64),
    },
  ]
end

private def plan_json(
  source_path : String,
  source_size : Int64,
  source_sha : String,
  operations : Array(Hash(String, JSON::Any)) = operation_specs,
) : String
  JSON.build do |json|
    json.object do
      json.field "schema_version", 1
      json.field "converter_version", "0.2.0"
      json.field "converter_license", "MIT"
      json.field "source" do
        json.object do
          json.field "repository", "https://github.com/microsoft/TRELLIS.2"
          json.field "revision", "1" * 40
          json.field "license", "MIT"
        end
      end
      json.field "model" do
        json.object do
          json.field "repository", "microsoft/TRELLIS.2-4B"
          json.field "revision", "2" * 40
          json.field "license", "MIT"
        end
      end
      json.field "stage_id", "dino_v3"
      json.field "source_file" do
        json.object do
          json.field "path", source_path
          json.field "byte_length", source_size
          json.field "sha256", source_sha
          json.field "license", "MIT"
        end
      end
      json.field "output_file", "stages/dino_v3.safetensors"
      json.field "config_path", "configs/dino_v3.json"
      json.field "operations" do
        json.array do
          operations.each { |operation| operation.to_json(json) }
        end
      end
    end
  end
end

private def directory_digest(path : String) : String
  digest = Digest::SHA256.new
  Dir.glob(
    File.join(path, "**", "*"),
    match: File::MatchOptions.glob_default | File::MatchOptions::DotFiles
  )
    .select { |entry| File.file?(entry) }
    .sort
    .each do |entry|
      relative = Path[entry].relative_to(Path[path]).to_s
      digest.update(relative.to_slice)
      digest.update(Bytes[0_u8])
      digest.file(entry)
    end
  digest.hexfinal
end

describe ML::ThreeD::Trellis2::Converter do
  it "publishes byte-identical packs independent of operation order" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      source_root = File.join(dir, "source with spaces")
      Dir.mkdir(source_root)
      tensor_path = File.join(source_root, "source.safetensors")
      write_source_safetensors(tensor_path, source_tensors)
      source_sha = Digest::SHA256.new.file(tensor_path).hexfinal
      source_size = File.size(tensor_path)

      File.write(
        File.join(source_root, "plan-a.json"),
        plan_json("source.safetensors", source_size, source_sha)
      )
      File.write(
        File.join(source_root, "plan-b.json"),
        plan_json(
          "source.safetensors",
          source_size,
          source_sha,
          operation_specs.reverse
        )
      )
      changed_operations = operation_specs
      concat_sources = changed_operations
        .find! { |operation| operation["kind"].as_s == "concat" }["sources"]
        .as_a
      concat_sources.reverse!
      File.write(
        File.join(source_root, "plan-c.json"),
        plan_json(
          "source.safetensors",
          source_size,
          source_sha,
          changed_operations
        )
      )

      output_a = File.join(dir, "pack a")
      output_b = File.join(dir, "pack b")
      output_c = File.join(dir, "pack c")
      before_sha = Digest::SHA256.new.file(tensor_path).hexfinal
      manifest_a = ML::ThreeD::Trellis2::Converter.convert!(
        source_root,
        output_a,
        "plan-a.json",
        before_publish: ->(staged : String) do
          File.dirname(staged).should eq(File.dirname(output_a))
          ML::ThreeD::Trellis2::Manifest.load(staged)
          nil
        end
      )
      manifest_b = ML::ThreeD::Trellis2::Converter.convert!(
        source_root,
        output_b,
        "plan-b.json"
      )
      manifest_c = ML::ThreeD::Trellis2::Converter.convert!(
        source_root,
        output_c,
        "plan-c.json"
      )

      directory_digest(output_a).should eq(directory_digest(output_b))
      manifest_a.pack_id.should eq(manifest_b.pack_id)
      directory_digest(output_a).should_not eq(directory_digest(output_c))
      manifest_a.pack_id.should_not eq(manifest_c.pack_id)
      Digest::SHA256.new.file(tensor_path).hexfinal.should eq(before_sha)
      manifest_a.stages.first.tensors.map(&.layout).sort.should eq([
        "concat",
        "identity",
        "permute",
        "split",
        "split",
        "transpose",
      ])
      ML::ThreeD::Trellis2::Manifest.load(output_a).pack_id
        .should eq(manifest_a.pack_id)

      File.write(
        File.join(output_a, "conversion", "plan.json"),
        %({"stripped":true})
      )
      expect_raises(
        ML::ThreeD::Trellis2::ManifestError,
        /byte length|SHA-256/
      ) do
        ML::ThreeD::Trellis2::Manifest.load(output_a)
      end
    end
  end

  it "fails before publication on unknown or dropped source tensors" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      source_root = File.join(dir, "source")
      Dir.mkdir(source_root)
      tensor_path = File.join(source_root, "source.safetensors")
      write_source_safetensors(
        tensor_path,
        [{name: "only", shape: [1_i64], values: [1_u16]}]
      )
      source_sha = Digest::SHA256.new.file(tensor_path).hexfinal
      source_size = File.size(tensor_path)
      bad_operations = [
        {
          "id"          => JSON::Any.new("missing_op"),
          "kind"        => JSON::Any.new("identity"),
          "source"      => JSON::Any.new("missing"),
          "destination" => JSON::Any.new("result"),
        },
      ]
      File.write(
        File.join(source_root, "conversion.json"),
        plan_json(
          "source.safetensors",
          source_size,
          source_sha,
          bad_operations
        )
      )

      output = File.join(dir, "pack")
      expect_raises(
        ML::ThreeD::Trellis2::ConverterError,
        /missing|unexpected|unconsumed/
      ) do
        ML::ThreeD::Trellis2::Converter.convert!(source_root, output)
      end
      File.exists?(output).should be_false
      Dir.children(dir).any?(&.includes?(".tmp-")).should be_false
    end
  end

  it "never overwrites an existing output path" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      source_root = File.join(dir, "source")
      Dir.mkdir(source_root)
      tensor_path = File.join(source_root, "source.safetensors")
      write_source_safetensors(
        tensor_path,
        [{name: "only", shape: [1_i64], values: [1_u16]}]
      )
      source_sha = Digest::SHA256.new.file(tensor_path).hexfinal
      File.write(
        File.join(source_root, "conversion.json"),
        plan_json(
          "source.safetensors",
          File.size(tensor_path),
          source_sha,
          [{
            "id"          => JSON::Any.new("only_op"),
            "kind"        => JSON::Any.new("identity"),
            "source"      => JSON::Any.new("only"),
            "destination" => JSON::Any.new("result"),
          }]
        )
      )

      output = File.join(dir, "pack")
      Dir.mkdir(output)
      marker = File.join(output, "owned-by-user")
      File.write(marker, "keep")

      expect_raises(
        ML::ThreeD::Trellis2::ConverterError,
        /already exists/
      ) do
        ML::ThreeD::Trellis2::Converter.convert!(source_root, output)
      end
      File.read(marker).should eq("keep")
    end
  end

  it "keeps output outside the immutable source after filesystem canonicalization" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      source_root = File.join(dir, "Source")
      Dir.mkdir(source_root)
      tensor_path = File.join(source_root, "source.safetensors")
      write_source_safetensors(
        tensor_path,
        [{name: "only", shape: [1_i64], values: [1_u16]}]
      )
      source_sha = Digest::SHA256.new.file(tensor_path).hexfinal
      File.write(
        File.join(source_root, "conversion.json"),
        plan_json(
          "source.safetensors",
          File.size(tensor_path),
          source_sha,
          [{
            "id"          => JSON::Any.new("only_op"),
            "kind"        => JSON::Any.new("identity"),
            "source"      => JSON::Any.new("only"),
            "destination" => JSON::Any.new("result"),
          }]
        )
      )

      case_alias = File.join(dir, "source")
      parent = File.directory?(case_alias) ? case_alias : source_root
      output = File.join(parent, "pack")
      expect_raises(
        ML::ThreeD::Trellis2::ConverterError,
        /outside the immutable source root/
      ) do
        ML::ThreeD::Trellis2::Converter.convert!(source_root, output)
      end
      File.exists?(output).should be_false
    end
  end

  it "revalidates staged output and source immediately before publication" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      source_root = File.join(dir, "source")
      Dir.mkdir(source_root)
      tensor_path = File.join(source_root, "source.safetensors")
      write_source_safetensors(
        tensor_path,
        [{name: "only", shape: [1_i64], values: [1_u16]}]
      )
      source_sha = Digest::SHA256.new.file(tensor_path).hexfinal
      File.write(
        File.join(source_root, "conversion.json"),
        plan_json(
          "source.safetensors",
          File.size(tensor_path),
          source_sha,
          [{
            "id"          => JSON::Any.new("only_op"),
            "kind"        => JSON::Any.new("identity"),
            "source"      => JSON::Any.new("only"),
            "destination" => JSON::Any.new("result"),
          }]
        )
      )

      corrupted_output = File.join(dir, "corrupted-pack")
      expect_raises(
        ML::ThreeD::Trellis2::ConverterError,
        /byte length|SHA-256/
      ) do
        ML::ThreeD::Trellis2::Converter.convert!(
          source_root,
          corrupted_output,
          before_publish: ->(staged : String) do
            File.open(
              File.join(staged, "stages", "dino_v3.safetensors"),
              "ab"
            ) { |io| io.write_byte(0_u8) }
          end
        )
      end
      File.exists?(corrupted_output).should be_false

      mutated_source_output = File.join(dir, "mutated-source-pack")
      expect_raises(
        ML::ThreeD::Trellis2::ConverterError,
        /byte length|SHA-256/
      ) do
        ML::ThreeD::Trellis2::Converter.convert!(
          source_root,
          mutated_source_output,
          before_publish: ->(_staged : String) do
            File.open(tensor_path, "ab") { |io| io.write_byte(0_u8) }
          end
        )
      end
      File.exists?(mutated_source_output).should be_false
      Dir.children(dir).any?(&.includes?(".tmp-")).should be_false
    end
  end

  it "rejects symlinked source files and conversion plans" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      source_root = File.join(dir, "source")
      Dir.mkdir(source_root)
      real_tensor = File.join(dir, "real.safetensors")
      write_source_safetensors(
        real_tensor,
        [{name: "only", shape: [1_i64], values: [1_u16]}]
      )
      tensor_alias = File.join(source_root, "source.safetensors")
      File.symlink(real_tensor, tensor_alias)
      source_sha = Digest::SHA256.new.file(real_tensor).hexfinal
      plan = plan_json(
        "source.safetensors",
        File.size(real_tensor),
        source_sha,
        [{
          "id"          => JSON::Any.new("only_op"),
          "kind"        => JSON::Any.new("identity"),
          "source"      => JSON::Any.new("only"),
          "destination" => JSON::Any.new("result"),
        }]
      )
      real_plan = File.join(dir, "real-plan.json")
      File.write(real_plan, plan)
      File.symlink(real_plan, File.join(source_root, "conversion.json"))

      expect_raises(
        ML::ThreeD::Trellis2::ConverterError,
        /symlink/
      ) do
        ML::ThreeD::Trellis2::Converter.convert!(
          source_root,
          File.join(dir, "pack")
        )
      end

      File.delete(File.join(source_root, "conversion.json"))
      File.write(File.join(source_root, "conversion.json"), plan)
      expect_raises(
        ML::ThreeD::Trellis2::ConverterError,
        /symlink/
      ) do
        ML::ThreeD::Trellis2::Converter.convert!(
          source_root,
          File.join(dir, "pack")
        )
      end
    end
  end

  it "strictly rejects malformed or ambiguous operation graphs" do
    source_size = 16_i64
    source_sha = "a" * 64

    unknown = JSON.parse(
      plan_json("source.safetensors", source_size, source_sha)
    ).as_h
    unknown["surprise"] = JSON::Any.new(true)
    expect_raises(
      ML::ThreeD::Trellis2::ConversionPlanError,
      /unknown conversion plan key.*surprise/
    ) do
      ML::ThreeD::Trellis2::ConversionPlan.parse(unknown.to_json)
    end

    duplicate_destination = [
      {
        "id"          => JSON::Any.new("first"),
        "kind"        => JSON::Any.new("identity"),
        "source"      => JSON::Any.new("a"),
        "destination" => JSON::Any.new("same"),
      },
      {
        "id"          => JSON::Any.new("second"),
        "kind"        => JSON::Any.new("identity"),
        "source"      => JSON::Any.new("b"),
        "destination" => JSON::Any.new("same"),
      },
    ]
    expect_raises(
      ML::ThreeD::Trellis2::ConversionPlanError,
      /duplicate conversion destination/
    ) do
      ML::ThreeD::Trellis2::ConversionPlan.parse(
        plan_json(
          "source.safetensors",
          source_size,
          source_sha,
          duplicate_destination
        )
      )
    end

    duplicate_source = duplicate_destination.map(&.dup)
    duplicate_source[1]["destination"] = JSON::Any.new("other")
    duplicate_source[1]["source"] = JSON::Any.new("a")
    expect_raises(
      ML::ThreeD::Trellis2::ConversionPlanError,
      /consumed more than once/
    ) do
      ML::ThreeD::Trellis2::ConversionPlan.parse(
        plan_json(
          "source.safetensors",
          source_size,
          source_sha,
          duplicate_source
        )
      )
    end

    invalid_transpose = [{
      "id"          => JSON::Any.new("bad_transpose"),
      "kind"        => JSON::Any.new("transpose"),
      "source"      => JSON::Any.new("a"),
      "destination" => JSON::Any.new("b"),
      "axes"        => JSON::Any.new([
        JSON::Any.new(0_i64),
        JSON::Any.new(0_i64),
      ]),
    }]
    expect_raises(
      ML::ThreeD::Trellis2::ConversionPlanError,
      /transpose axes/
    ) do
      ML::ThreeD::Trellis2::ConversionPlan.parse(
        plan_json(
          "source.safetensors",
          source_size,
          source_sha,
          invalid_transpose
        )
      )
    end

    invalid_permutation = [{
      "id"          => JSON::Any.new("bad_permutation"),
      "kind"        => JSON::Any.new("permute"),
      "source"      => JSON::Any.new("a"),
      "destination" => JSON::Any.new("b"),
      "axes"        => JSON::Any.new([
        JSON::Any.new(1_i64),
        JSON::Any.new(1_i64),
      ]),
    }]
    expect_raises(
      ML::ThreeD::Trellis2::ConversionPlanError,
      /permutation axes/
    ) do
      ML::ThreeD::Trellis2::ConversionPlan.parse(
        plan_json(
          "source.safetensors",
          source_size,
          source_sha,
          invalid_permutation
        )
      )
    end
  end
end
