require "spec"
require "../examples/support/qwen25_rms_statistics_probe"

private alias RMSProbeFixture = NamedTuple(
  root: String,
  source_manifest: String,
  source_manifest_sha: String,
  source_bf16: String,
  rms_manifest: String,
  rms_manifest_sha: String,
  rms_f32: String,
  rms_bf16: String,
  expected_square: Bytes,
  expected_variance: Bytes,
)

private def rms_probe_sha(bytes : Bytes) : String
  Digest::SHA256.hexdigest(bytes)
end

private def rms_probe_write_bytes(path : String, bytes : Bytes) : Nil
  File.open(path, "wb") { |io| io.write(bytes) }
end

private def rms_probe_rewrite_json(path : String, &block : JSON::Any ->) : String
  document = JSON.parse(File.read(path))
  yield document
  bytes = document.to_json
  File.write(path, bytes)
  Digest::SHA256.hexdigest(bytes)
end

private def rms_probe_fixture(&block : RMSProbeFixture ->) : Nil
  root = File.join(Dir.tempdir, "qwen25-rms-probe-#{Random.rand(1_000_000_000)}")
  Dir.mkdir(root)
  begin
    source_bf16_path = File.join(root, "source.bf16le")
    source_f32_path = File.join(root, "source.f32le")
    rms_f32_path = File.join(root, "rms.f32le")
    rms_bf16_path = File.join(root, "rms.bf16le")

    residual = Bytes[0x80_u8, 0x3f_u8, 0_u8, 0_u8, 0x80_u8, 0xbf_u8, 0_u8, 0_u8,
      0_u8, 0x40_u8, 0_u8, 0_u8, 0_u8, 0xc0_u8, 0_u8, 0_u8]
    old_output = Bytes[0x01_u8, 0_u8, 0_u8, 0_u8, 0x02_u8, 0_u8, 0_u8, 0_u8,
      0x03_u8, 0_u8, 0_u8, 0_u8, 0x04_u8, 0_u8, 0_u8, 0_u8]
    source_bf16 = residual + old_output
    source_f32 = Bytes[0_u8, 0_u8, 0x80_u8, 0x3f_u8]
    rms_f32 = Bytes.new(120) { |index| (index % 251).to_u8 }
    rms_bf16 = Bytes.new(32) { |index| (index + 1).to_u8 }
    rms_expected_square = rms_f32[32, 32].dup
    rms_expected_variance = rms_f32[64, 8].dup

    rms_probe_write_bytes(source_bf16_path, source_bf16)
    rms_probe_write_bytes(source_f32_path, source_f32)
    rms_probe_write_bytes(rms_f32_path, rms_f32)
    rms_probe_write_bytes(rms_bf16_path, rms_bf16)

    source_manifest = JSON.build do |json|
      json.object do
        json.field "schema", Qwen25RMSStatisticsProbe::STAGE_SCHEMA
        json.field "schema_version", 1
        json.field "checkpoint" do
          json.object do
            json.field "repo_id", Qwen25RMSStatisticsProbe::MODEL_ID
            json.field "revision", Qwen25RMSStatisticsProbe::REVISION
          end
        end
        json.field "config" do
          json.object do
            json.field "sequence_length", 2
            json.field "hidden_size", 4
            json.field "num_hidden_layers", 28
            json.field "intermediate_size", 18944
            json.field "num_attention_heads", 28
            json.field "num_key_value_heads", 4
            json.field "head_dim", 128
            json.field "rms_norm_eps", 1e-6
          end
        end
        json.field "layer" do
          json.object do
            json.field "index", 0
            json.field "weights" do
              json.array do
                json.object do
                  json.field "key", Qwen25RMSStatisticsProbe::NORM_WEIGHT_KEY
                  json.field "shard", "model-00001.safetensors"
                  json.field "dtype", "BF16"
                  json.field "shape", [4]
                  json.field "nbytes", 8
                  json.field "tensor_sha256", "a" * 64
                end
              end
            end
          end
        end
        json.field "prefix_reference" do
          json.object do
            json.field "state_indexing", "0=token_embedding; i+1=decoder_layer_i_output"
            json.field "input_state_index", 0
            json.field "expected_output_state_index", 1
            json.field "manifest_sha256", "b" * 64
          end
        end
        json.field "stages" do
          json.array do
            {name: "residual_after_attention", offset: 0_i64, bytes: 16_i64,
             hash: rms_probe_sha(residual)}.tap do |stage|
              json.object do
                json.field "name", stage[:name]
                json.field "dtype", "bfloat16"
                json.field "shape", [1, 2, 4]
                json.field "storage" do
                  json.object do
                    json.field "sidecar", "bf16"
                    json.field "offset_bytes", stage[:offset]
                    json.field "nbytes", stage[:bytes]
                  end
                end
              end
            end
            json.object do
              json.field "name", "layers.0.post_attention_layernorm"
              json.field "dtype", "bfloat16"
              json.field "shape", [1, 2, 4]
              json.field "storage" do
                json.object do
                  json.field "sidecar", "bf16"
                  json.field "offset_bytes", 16
                  json.field "nbytes", 16
                end
              end
            end
          end
        end
        json.field "sidecars" do
          json.object do
            json.field "bf16" do
              json.object do
                json.field "path", "source.bf16le"
                json.field "encoding", "bfloat16-le"
                json.field "nbytes", source_bf16.size
                json.field "sha256", rms_probe_sha(source_bf16)
              end
            end
            json.field "f32" do
              json.object do
                json.field "path", "source.f32le"
                json.field "encoding", "float32-le"
                json.field "byte_order", "little"
                json.field "storage_dtype", "float32"
                json.field "nbytes", source_f32.size
                json.field "sha256", rms_probe_sha(source_f32)
              end
            end
          end
        end
        json.field "rms_trace" do
          json.object do
            json.field "site", "post_attention_layernorm"
            json.field "input_state_sha256", rms_probe_sha(residual)
          end
        end
      end
    end
    source_manifest_path = File.join(root, "source.json")
    File.write(source_manifest_path, source_manifest)
    source_manifest_sha = Digest::SHA256.hexdigest(source_manifest)
    source = Qwen25RMSStatisticsProbe::SourceFixture.new(
      source_manifest_path, source_manifest_sha,
      expected_tokens: 2, expected_dim: 4)

    rms_records = [] of NamedTuple(name: String, dtype: String,
      shape: Array(Int32), sidecar: String, offset: Int32, bytes: Int32)
    f32_offset = 0
    bf16_offset = 0
    Qwen25RMSStatisticsProbe::RMS_SUFFIXES.each do |suffix|
      f32 = Qwen25RMSStatisticsProbe::F32_SUFFIXES.includes?(suffix)
      shape = if %w(variance_f32 variance_plus_eps_f32 rsqrt_f32).includes?(suffix)
                [1, 2, 1]
              else
                [1, 2, 4]
              end
      bytes = shape.product * (f32 ? 4 : 2)
      offset = f32 ? f32_offset : bf16_offset
      rms_records << {
        name: "#{Qwen25RMSStatisticsProbe::RMS_PREFIX}#{suffix}",
        dtype: f32 ? "float32" : "bfloat16", shape: shape,
        sidecar: f32 ? "f32" : "bf16", offset: offset, bytes: bytes,
      }
      if f32
        f32_offset += bytes
      else
        bf16_offset += bytes
      end
    end

    rms_manifest = JSON.build do |json|
      json.object do
        json.field "schema", Qwen25RMSStatisticsProbe::RMS_SCHEMA
        json.field "schema_version", 1
        json.field "source_stage_manifest" do
          json.object do
            json.field "path", "source.json"
            json.field "manifest_sha256", source.manifest_sha256
            json.field "sidecar_sha256s" do
              json.object do
                json.field "bf16", source.bf16_sidecar_sha256
                json.field "f32", source.f32_sidecar_sha256
              end
            end
            json.field "layer_index", 0
            json.field "prefix_reference_manifest_sha256", source.prefix_manifest_sha256
            json.field "input_stage_name", Qwen25RMSStatisticsProbe::RESIDUAL_STAGE_NAME
            json.field "input_state_sha256", source.residual_sha256
            json.field "expected_output_stage_name", source.expected_output_name
            json.field "expected_output_state_sha256", source.output_state_sha256
          end
        end
        json.field "layer" do
          json.object do
            json.field "norm_weight", source.norm_weight
          end
        end
        json.field "rms_trace" do
          json.object do
            json.field "site", "post_attention_layernorm"
            json.field "input_state_sha256", source.residual_sha256
            json.field "variance_epsilon", 1e-6
            json.field "source" do
              json.object do
                json.field "schema_version", 1
                json.field "source_file", "models/qwen2/modeling_qwen2.py"
                json.field "source_sha256", Qwen25RMSStatisticsProbe::RMS_SOURCE_SHA256
              end
            end
            json.field "operation_sequence" do
              json.array do
                [
                  {"input_f32", "aten::to", "dtype"},
                  {"square_f32", "aten::pow", "Tensor_Scalar"},
                  {"variance_f32", "aten::mean", "dim"},
                  {"variance_plus_eps_f32", "aten::add", "Tensor"},
                  {"rsqrt_f32", "aten::rsqrt", ""},
                  {"normalized_f32", "aten::mul", "Tensor"},
                  {"normalized_bf16", "aten::to", "dtype"},
                  {"gained_output_bf16", "aten::mul", "Tensor"},
                ].each do |step|
                  json.object do
                    json.field "stage", step[0]
                    json.field "operator", step[1]
                    json.field "overload", step[2]
                  end
                end
              end
            end
            json.field "stages" do
              json.array do
                rms_records.each do |record|
                  json.object do
                    json.field "name", record[:name]
                    json.field "dtype", record[:dtype]
                    json.field "shape", record[:shape]
                    json.field "storage" do
                      json.object do
                        json.field "sidecar", record[:sidecar]
                        json.field "offset_bytes", record[:offset]
                        json.field "nbytes", record[:bytes]
                      end
                    end
                  end
                end
              end
            end
          end
        end
        json.field "sidecars" do
          json.object do
            json.field "f32" do
              json.object do
                json.field "path", "rms.f32le"
                json.field "encoding", "float32-le"
                json.field "byte_order", "little"
                json.field "storage_dtype", "float32"
                json.field "nbytes", rms_f32.size
                json.field "sha256", rms_probe_sha(rms_f32)
              end
            end
            json.field "bf16" do
              json.object do
                json.field "path", "rms.bf16le"
                json.field "encoding", "bfloat16-le"
                json.field "nbytes", rms_bf16.size
                json.field "sha256", rms_probe_sha(rms_bf16)
              end
            end
          end
        end
      end
    end
    rms_manifest_path = File.join(root, "rms.json")
    File.write(rms_manifest_path, rms_manifest)
    block.call({
      root:                root,
      source_manifest:     source_manifest_path,
      source_manifest_sha: source_manifest_sha,
      source_bf16:         source_bf16_path,
      rms_manifest:        rms_manifest_path,
      rms_manifest_sha:    Digest::SHA256.hexdigest(rms_manifest),
      rms_f32:             rms_f32_path,
      rms_bf16:            rms_bf16_path,
      expected_square:     rms_expected_square,
      expected_variance:   rms_expected_variance,
    })
  ensure
    Dir.glob(File.join(root, "*")).each { |path| File.delete?(path) }
    Dir.delete(root)
  end
end

describe Qwen25RMSStatisticsProbe do
  it "admits exact named-buffer accounting and rejects a one-byte-short cap" do
    Qwen25RMSStatisticsProbe.required_named_bytes(2, 4).should eq(316_i64)
    Qwen25RMSStatisticsProbe.admit_named_bytes!(2, 4, 316).should eq(316_i64)
    expect_raises(ArgumentError, /budget rejected/) do
      Qwen25RMSStatisticsProbe.admit_named_bytes!(2, 4, 315)
    end

    # S=1280, D=3584: BF16 input; six F32 panels (input, native square,
    # authenticated upstream square, and the three exponent-power outputs);
    # twelve scalar-output buffers and the two exponent feeds.
    required = Qwen25RMSStatisticsProbe.required_named_bytes(1280, 3584)
    required.should eq(119_336_972_i64)
    Qwen25RMSStatisticsProbe.admit_named_bytes!(1280, 3584, required).should eq(required)
    expect_raises(ArgumentError, /budget rejected/) do
      Qwen25RMSStatisticsProbe.admit_named_bytes!(1280, 3584, required - 1)
    end
  end

  it "pins matching F32 power operands and the separate Torch Int64 feed route" do
    f32 = Qwen25RMSStatisticsProbe::EXPONENT_POWER_CONTROL
    f32[:dtype].should eq("float32")
    f32[:shape].should eq([1])
    f32[:value].should eq(2.0)
    f32[:bits].should eq("40000000")
    Qwen25RMSStatisticsProbe.exponent_two_f32le.should eq(Bytes[0, 0, 0, 0x40])
    f32[:matches_torch_post_cast_operand].should be_true
    f32[:reproduces_torch_scalar_feed_and_cast_topology].should be_false

    torch = Qwen25RMSStatisticsProbe::TORCH_INT64_EXPONENT_CONTROL
    torch[:dtype].should eq("int64")
    torch[:shape].should eq([1])
    torch[:value].should eq(2_i64)
    torch[:bytes].should eq(8)
    Qwen25RMSStatisticsProbe.exponent_two_i64le.should eq(Bytes[2, 0, 0, 0, 0, 0, 0, 0])
    torch[:reproduces_torch_scalar_feed_and_cast_topology].should be_true
  end

  it "retains only selected source ranges from the exact authenticated stream" do
    rms_probe_fixture do |fixture|
      source = Qwen25RMSStatisticsProbe::SourceFixture.new(
        fixture[:source_manifest], fixture[:source_manifest_sha],
        expected_tokens: 2, expected_dim: 4)
      original_residual = source.residual.dup
      File.write(fixture[:source_bf16], "changed-length-content")
      source.residual.should eq(original_residual)
      source.residual_sha256.should eq(rms_probe_sha(original_residual))
      original_output = Bytes[0x01_u8, 0_u8, 0_u8, 0_u8,
        0x02_u8, 0_u8, 0_u8, 0_u8, 0x03_u8, 0_u8, 0_u8, 0_u8,
        0x04_u8, 0_u8, 0_u8, 0_u8]
      source.output_state_sha256.should eq(rms_probe_sha(original_output))
    end
  end

  it "uses immutable RMS snapshots after manifest and sidecar authentication" do
    rms_probe_fixture do |fixture|
      source = Qwen25RMSStatisticsProbe::SourceFixture.new(
        fixture[:source_manifest], fixture[:source_manifest_sha],
        expected_tokens: 2, expected_dim: 4)
      reference = Qwen25RMSStatisticsProbe::ReferenceFixture.new(
        fixture[:rms_manifest], fixture[:rms_manifest_sha], source)
      reference.expected("square_f32").should eq(fixture[:expected_square])
      reference.expected("variance_f32").should eq(fixture[:expected_variance])

      rms_probe_write_bytes(fixture[:rms_f32], Bytes.new(120, 0xff_u8))
      rms_probe_write_bytes(fixture[:rms_bf16], Bytes.new(32, 0xee_u8))
      reference.expected("square_f32").should eq(fixture[:expected_square])
      reference.expected("variance_f32").should eq(fixture[:expected_variance])
      reference.expected("gained_output_bf16").should eq(Bytes.new(16) { |index| (index + 17).to_u8 })
    end
  end

  it "rejects same-size modified bytes before accepting a reference snapshot" do
    rms_probe_fixture do |fixture|
      source = Qwen25RMSStatisticsProbe::SourceFixture.new(
        fixture[:source_manifest], fixture[:source_manifest_sha],
        expected_tokens: 2, expected_dim: 4)
      rms_probe_write_bytes(fixture[:rms_f32], Bytes.new(120, 0xff_u8))
      expect_raises(ArgumentError, /digest mismatch/) do
        Qwen25RMSStatisticsProbe::ReferenceFixture.new(
          fixture[:rms_manifest], fixture[:rms_manifest_sha], source)
      end
    end
  end

  it "rejects an RMS input digest that no longer binds to the authenticated residual" do
    rms_probe_fixture do |fixture|
      source = Qwen25RMSStatisticsProbe::SourceFixture.new(
        fixture[:source_manifest], fixture[:source_manifest_sha],
        expected_tokens: 2, expected_dim: 4)
      changed_sha = rms_probe_rewrite_json(fixture[:rms_manifest]) do |document|
        document["rms_trace"].as_h["input_state_sha256"] = JSON::Any.new("c" * 64)
      end
      expect_raises(ArgumentError, /RMS input digest mismatch/) do
        Qwen25RMSStatisticsProbe::ReferenceFixture.new(
          fixture[:rms_manifest], changed_sha, source)
      end
    end
  end

  it "rejects gaps, overlaps, and extents beyond authenticated sidecar bounds" do
    binding = {name: "x", dtype: "float32", shape: [1_i64], sidecar: "f32",
               offset_bytes: 0_i64, nbytes: 4_i64}
    contiguous = {name: "y", dtype: "float32", shape: [1_i64], sidecar: "f32",
                  offset_bytes: 4_i64, nbytes: 4_i64}
    Qwen25RMSStatisticsProbe.validate_contiguous_extents!(
      [binding, contiguous], 8_i64, "tiny")

    gap = {name: "y", dtype: "float32", shape: [1_i64], sidecar: "f32",
           offset_bytes: 8_i64, nbytes: 4_i64}
    expect_raises(ArgumentError, /overlap or leave gaps/) do
      Qwen25RMSStatisticsProbe.validate_contiguous_extents!(
        [binding, gap], 12_i64, "tiny")
    end

    overlap = {name: "y", dtype: "float32", shape: [1_i64], sidecar: "f32",
               offset_bytes: 2_i64, nbytes: 4_i64}
    expect_raises(ArgumentError, /overlap or leave gaps/) do
      Qwen25RMSStatisticsProbe.validate_contiguous_extents!(
        [binding, overlap], 8_i64, "tiny")
    end

    oversized = {name: "x", dtype: "float32", shape: [3_i64], sidecar: "f32",
                 offset_bytes: 0_i64, nbytes: 12_i64}
    expect_raises(ArgumentError, /exceeds sidecar bounds/) do
      Qwen25RMSStatisticsProbe.validate_contiguous_extents!(
        [oversized], 8_i64, "tiny")
    end
  end

  it "describes the optimization-level-zero control as compile-only and one-factor" do
    control = Qwen25RMSStatisticsProbe::OPTIMIZATION0_CONTROL
    control[:compilation_descriptor_type].should eq("MPSGraphCompilationDescriptor")
    control[:compilation_api].should contain("compilationDescriptor:")
    control[:optimization_level].should eq(0)
    control[:reduction_axis].should eq([2])
    control[:target].should eq("variance_plus_epsilon only")
    control[:explicit_reshape].should be_true
    control[:encode_route].should eq("precompiled MPSGraphExecutable")
    control[:executable_descriptor_type].should eq("MPSGraphExecutableExecutionDescriptor")
    control[:executable_wait_until_completed].should be_false
    control[:differs_from_torch_graph_execution_descriptor_route].should be_true
  end
end
