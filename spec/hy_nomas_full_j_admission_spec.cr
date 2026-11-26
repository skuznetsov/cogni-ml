require "spec"
require "json"
require "../examples/support/hy_nomas_full_j_probe"

private def full_j_admission_document : JSON::Any
  shapes = HyNomasFullJProbe.shapes
  cursor = 0_i64
  raw = JSON.build do |json|
    json.object do
      json.field "schema", HyNomasFullJProbe::SCHEMA
      json.field "schema_version", 1
      json.field "upstream" do
        json.object do
          json.field "revision", HyNomasFullJProbe::Serial::PINNED_UPSTREAM_REVISION
        end
      end
      json.field "geometry" do
        json.object do
          json.field "batches", 1
          json.field "video_tokens", 5280
          json.field "text_tokens", 1280
          json.field "dim", 2048
          json.field "heads", 16
          json.field "head_dim", 128
          json.field "ffn_dim", 8192
          json.field "query_chunk", 256
          json.field "attention_segments", 1
          json.field "chunk_size_tokens", 0
          json.field "rope_grid", [6, 22, 40]
          json.field "rope_axis_dims", [16, 56, 56]
          json.field "rope_theta", 256.0
        end
      end
      json.field "tolerance" do
        json.object do
          json.field "atol", HyNomasFullJProbe::Serial::ATOL
          json.field "rtol", HyNomasFullJProbe::Serial::RTOL
        end
      end
      json.field "precision" do
        json.object do
          json.field "compute", "F32"
          json.field "parameter_storage", "F32"
          json.field "attention", "PyTorch MPS SDPA F32"
          json.field "parameter_values", "BF16-rounded values"
        end
      end
      json.field "producer" do
        json.object do
          json.field "device", "mps"
          json.field "activation_readback", "one packed final F32 tensor sidecar readback"
          json.field "attention_adapter_calls", 2
          json.field "attention_query_chunks_per_block", 26
          json.field "captures_per_block", HyNomasFullJProbe::CAPTURE_NAMES
          json.field "terminal_tensor_count", 14
        end
      end
      json.field "source" do
        json.object do
          json.field "blocks" do
            json.object do
              {"block0" => HyNomasFullJProbe::Serial::BLOCK0_PIN,
               "block1" => HyNomasFullJProbe::Serial::BLOCK1_PIN}.each do |name, pin|
                json.field name do
                  json.object do
                    json.field "block_index", pin.index
                    json.field "file_sha256", pin.file_sha256
                    json.field "manifest_sha256", pin.manifest_sha256
                    json.field "tensor_count", 32
                  end
                end
              end
            end
          end
        end
      end
      json.field "tensor_file" do
        json.object do
          json.field "name", "full-j-admission.tensors.f32le"
          json.field "nbytes", HyNomasFullJProbe::SIDECAR_BYTES
          json.field "encoding", "f32-le"
          json.field "sha256", "a" * 64
        end
      end
      json.field "tensors" do
        json.object do
          shapes.keys.sort.each do |name|
            shape = shapes[name]
            nbytes = shape.reduce(1_i64) { |product, dimension| product * dimension.to_i64 } * 4
            json.field name do
              json.object do
                json.field "dtype", "F32"
                json.field "shape", shape
                json.field "offset_bytes", cursor
                json.field "nbytes", nbytes
              end
            end
            cursor += nbytes
          end
        end
      end
    end
  end
  JSON.parse(raw)
end

private def full_j_admission_clone(document : JSON::Any) : JSON::Any
  JSON.parse(document.to_json)
end

private def full_j_admission_rejects(document : JSON::Any) : Nil
  expect_raises(ArgumentError) do
    HyNomasFullJProbe.validate_document!(document)
  end
end

private def full_j_metal_counters
  {allocations: ML::MetalBuffer.hy_text_probe_allocations,
   readbacks:   ML::MetalBuffer.hy_text_probe_readbacks}
end

private def full_j_assert_no_metal_activity(before : NamedTuple(allocations: Int64, readbacks: Int64))
  ML::MetalBuffer.hy_text_probe_allocations.should eq(before[:allocations])
  ML::MetalBuffer.hy_text_probe_readbacks.should eq(before[:readbacks])
end

describe HyNomasFullJProbe do
  it "publishes a complete report and refuses to replace an existing file" do
    counters = full_j_metal_counters
    directory = File.tempname("hy-nomas-full-j-report")
    Dir.mkdir(directory)
    report = File.join(directory, "report.json")
    begin
      HyNomasFullJProbe.write_report_new!(report, %({"passed":true}))
      File.read(report).should eq(%({"passed":true}\n))
      expect_raises(File::Error) do
        HyNomasFullJProbe.write_report_new!(report, %({"passed":false}))
      end
      File.read(report).should eq(%({"passed":true}\n))
      Dir.children(directory).should eq(["report.json"])
      full_j_assert_no_metal_activity(counters)
    ensure
      File.delete(report) if File.exists?(report)
      Dir.delete(directory)
    end
  end

  it "admits the exact fourteen-tensor full-J layout, geometry, and producer receipt" do
    counters = full_j_metal_counters
    document = full_j_admission_document

    HyNomasFullJProbe.validate_document!(document)
    document["tensors"].as_h.size.should eq(14)
    document["tensor_file"]["nbytes"].as_i64.should eq(339_656_704_i64)
    document["producer"]["attention_query_chunks_per_block"].as_i.should eq(26)
    full_j_assert_no_metal_activity(counters)
  end

  it "rejects missing or extra captures in the selected block inventory" do
    counters = full_j_metal_counters

    missing = full_j_admission_clone(full_j_admission_document)
    missing["tensors"].as_h.delete("block1.txt.output")
    full_j_admission_rejects(missing)

    extra = full_j_admission_clone(full_j_admission_document)
    extra["tensors"].as_h["block1.unselected"] = JSON.parse(%({"dtype":"F32","shape":[1],"offset_bytes":0,"nbytes":4}))
    full_j_admission_rejects(extra)
    full_j_assert_no_metal_activity(counters)
  end

  it "rejects swapped RoPE tensor axes, grid order, or axis allocation" do
    counters = full_j_metal_counters

    swapped_tensor_axes = full_j_admission_clone(full_j_admission_document)
    swapped_tensor_axes["tensors"].as_h["input.rope_cos"].as_h["shape"] = JSON.parse("[128,5280]")
    full_j_admission_rejects(swapped_tensor_axes)

    swapped_grid = full_j_admission_clone(full_j_admission_document)
    swapped_grid["geometry"].as_h["rope_grid"] = JSON.parse("[6,40,22]")
    full_j_admission_rejects(swapped_grid)

    swapped_axes = full_j_admission_clone(full_j_admission_document)
    swapped_axes["geometry"].as_h["rope_axis_dims"] = JSON.parse("[56,16,56]")
    full_j_admission_rejects(swapped_axes)
    full_j_assert_no_metal_activity(counters)
  end

  it "rejects expanded numerical tolerances and wrong block geometry" do
    counters = full_j_metal_counters

    loose = full_j_admission_clone(full_j_admission_document)
    loose["tolerance"].as_h["atol"] = JSON.parse("0.0001")
    full_j_admission_rejects(loose)

    wrong_geometry = full_j_admission_clone(full_j_admission_document)
    wrong_geometry["geometry"].as_h["head_dim"] = JSON.parse("64")
    full_j_admission_rejects(wrong_geometry)
    full_j_assert_no_metal_activity(counters)
  end

  it "rejects a block-0 identity substituted for the pinned block-1 source" do
    counters = full_j_metal_counters
    substituted = full_j_admission_clone(full_j_admission_document)
    block1 = substituted["source"]["blocks"]["block1"].as_h
    block1["block_index"] = JSON.parse("0")
    block1["file_sha256"] = JSON.parse(%("#{HyNomasFullJProbe::Serial::BLOCK0_PIN.file_sha256}"))
    block1["manifest_sha256"] = JSON.parse(%("#{HyNomasFullJProbe::Serial::BLOCK0_PIN.manifest_sha256}"))

    full_j_admission_rejects(substituted)
    full_j_assert_no_metal_activity(counters)
  end

  it "rejects shifted and overlapping tensor ranges and a trailing sidecar extent" do
    counters = full_j_metal_counters
    names = HyNomasFullJProbe.shapes.keys.sort

    shifted = full_j_admission_clone(full_j_admission_document)
    metadata = shifted["tensors"].as_h[names[1]].as_h
    metadata["offset_bytes"] = JSON.parse((metadata["offset_bytes"].as_i64 + 4).to_s)
    full_j_admission_rejects(shifted)

    overlap = full_j_admission_clone(full_j_admission_document)
    overlap["tensors"].as_h[names[1]].as_h["offset_bytes"] = JSON.parse("0")
    full_j_admission_rejects(overlap)

    trailing = full_j_admission_clone(full_j_admission_document)
    trailing["tensor_file"].as_h["nbytes"] = JSON.parse((HyNomasFullJProbe::SIDECAR_BYTES + 4).to_s)
    full_j_admission_rejects(trailing)
    full_j_assert_no_metal_activity(counters)
  end

  it "requires the MPS single-terminal-readback producer receipt" do
    counters = full_j_metal_counters

    wrong_device = full_j_admission_clone(full_j_admission_document)
    wrong_device["producer"].as_h["device"] = JSON.parse(%("cpu"))
    full_j_admission_rejects(wrong_device)

    wrong_terminal = full_j_admission_clone(full_j_admission_document)
    wrong_terminal["producer"].as_h["activation_readback"] = JSON.parse(%("per-tensor readbacks"))
    full_j_admission_rejects(wrong_terminal)

    wrong_inventory = full_j_admission_clone(full_j_admission_document)
    wrong_inventory["producer"].as_h["captures_per_block"] = JSON.parse(%(["img.output"]))
    full_j_admission_rejects(wrong_inventory)
    full_j_assert_no_metal_activity(counters)
  end

  it "rejects a named-byte budget one byte below the full-J estimate" do
    counters = full_j_metal_counters
    required = HyNomasFullJProbe.named_bytes(604_227_584_i64)[:total]

    HyNomasFullJProbe.admit_budget!(required, required)
    expect_raises(ArgumentError) do
      HyNomasFullJProbe.admit_budget!(required, required - 1)
    end
    full_j_assert_no_metal_activity(counters)
  end
end
