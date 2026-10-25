require "spec"
require "json"
require "../examples/support/hy_nomas_four_block_probe"

private def four_block_admission_document : JSON::Any
  shapes = HyNomasFourBlockProbe.shapes
  cursor = 0_i64
  raw = JSON.build do |json|
    json.object do
      json.field "schema", HyNomasFourBlockProbe::SCHEMA
      json.field "schema_version", 1
      json.field "upstream" do
        json.object do
          json.field "revision", HyNomasFourBlockProbe::Serial::PINNED_UPSTREAM_REVISION
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
          json.field "atol", HyNomasFourBlockProbe::Serial::ATOL
          json.field "rtol", HyNomasFourBlockProbe::Serial::RTOL
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
          json.field "attention_adapter_calls", 4
          json.field "attention_query_chunks_per_block", 26
          json.field "captures_per_block", HyNomasFourBlockProbe::CAPTURE_NAMES
          json.field "terminal_tensor_count", 14
        end
      end
      json.field "source" do
        json.object do
          json.field "block_order", ["block0", "block1", "block2", "block3"]
          json.field "blocks" do
            json.object do
              HyNomasFourBlockProbe::SOURCE_PINS.each_with_index do |pin, index|
                json.field "block#{index}" do
                  json.object do
                    json.field "block_index", index
                    json.field "file_sha256", pin[:file_sha256]
                    json.field "manifest_sha256", pin[:manifest_sha256]
                    json.field "tensor_prefix", "double_blocks.#{index}."
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
          json.field "nbytes", HyNomasFourBlockProbe::SIDECAR_BYTES
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

private def four_block_admission_clone(document : JSON::Any) : JSON::Any
  JSON.parse(document.to_json)
end

private def four_block_admission_rejects(document : JSON::Any) : Nil
  expect_raises(ArgumentError) do
    HyNomasFourBlockProbe.validate_document!(document)
  end
end

private def four_block_metal_counters
  {allocations: ML::MetalBuffer.hy_text_probe_allocations,
   readbacks:   ML::MetalBuffer.hy_text_probe_readbacks}
end

private def four_block_assert_no_metal_activity(before : NamedTuple(allocations: Int64, readbacks: Int64))
  ML::MetalBuffer.hy_text_probe_allocations.should eq(before[:allocations])
  ML::MetalBuffer.hy_text_probe_readbacks.should eq(before[:readbacks])
end

describe HyNomasFourBlockProbe do
  it "publishes a complete report and refuses to replace an existing file" do
    counters = four_block_metal_counters
    directory = File.tempname("hy-nomas-full-j-report")
    Dir.mkdir(directory)
    report = File.join(directory, "report.json")
    begin
      HyNomasFourBlockProbe.write_report_new!(report, %({"passed":true}))
      File.read(report).should eq(%({"passed":true}\n))
      expect_raises(File::Error) do
        HyNomasFourBlockProbe.write_report_new!(report, %({"passed":false}))
      end
      File.read(report).should eq(%({"passed":true}\n))
      Dir.children(directory).should eq(["report.json"])
      four_block_assert_no_metal_activity(counters)
    ensure
      File.delete(report) if File.exists?(report)
      Dir.delete(directory)
    end
  end

  it "admits the exact fourteen-tensor full-J layout, geometry, and producer receipt" do
    counters = four_block_metal_counters
    document = four_block_admission_document

    HyNomasFourBlockProbe.validate_document!(document)
    document["tensors"].as_h.size.should eq(14)
    document["tensor_file"]["nbytes"].as_i64.should eq(274_120_704_i64)
    document["producer"]["attention_query_chunks_per_block"].as_i.should eq(26)
    four_block_assert_no_metal_activity(counters)
  end

  it "rejects missing or extra captures in the selected block inventory" do
    counters = four_block_metal_counters

    missing = four_block_admission_clone(four_block_admission_document)
    missing["tensors"].as_h.delete("block3.txt.output")
    four_block_admission_rejects(missing)

    extra = four_block_admission_clone(four_block_admission_document)
    extra["tensors"].as_h["block3.unselected"] = JSON.parse(%({"dtype":"F32","shape":[1],"offset_bytes":0,"nbytes":4}))
    four_block_admission_rejects(extra)
    four_block_assert_no_metal_activity(counters)
  end

  it "rejects swapped RoPE tensor axes, grid order, or axis allocation" do
    counters = four_block_metal_counters

    swapped_tensor_axes = four_block_admission_clone(four_block_admission_document)
    swapped_tensor_axes["tensors"].as_h["input.rope_cos"].as_h["shape"] = JSON.parse("[128,5280]")
    four_block_admission_rejects(swapped_tensor_axes)

    swapped_grid = four_block_admission_clone(four_block_admission_document)
    swapped_grid["geometry"].as_h["rope_grid"] = JSON.parse("[6,40,22]")
    four_block_admission_rejects(swapped_grid)

    swapped_axes = four_block_admission_clone(four_block_admission_document)
    swapped_axes["geometry"].as_h["rope_axis_dims"] = JSON.parse("[56,16,56]")
    four_block_admission_rejects(swapped_axes)
    four_block_assert_no_metal_activity(counters)
  end

  it "rejects expanded numerical tolerances and wrong block geometry" do
    counters = four_block_metal_counters

    loose = four_block_admission_clone(four_block_admission_document)
    loose["tolerance"].as_h["atol"] = JSON.parse("0.0001")
    four_block_admission_rejects(loose)

    wrong_geometry = four_block_admission_clone(four_block_admission_document)
    wrong_geometry["geometry"].as_h["head_dim"] = JSON.parse("64")
    four_block_admission_rejects(wrong_geometry)
    four_block_assert_no_metal_activity(counters)
  end

  it "rejects a block-0 identity substituted for the pinned block-3 source" do
    counters = four_block_metal_counters
    substituted = four_block_admission_clone(four_block_admission_document)
    block1 = substituted["source"]["blocks"]["block3"].as_h
    block1["block_index"] = JSON.parse("0")
    block1["file_sha256"] = JSON.parse(%("#{HyNomasFourBlockProbe::Serial::BLOCK0_PIN.file_sha256}"))
    block1["manifest_sha256"] = JSON.parse(%("#{HyNomasFourBlockProbe::Serial::BLOCK0_PIN.manifest_sha256}"))

    four_block_admission_rejects(substituted)
    four_block_assert_no_metal_activity(counters)
  end

  it "rejects shifted and overlapping tensor ranges and a trailing sidecar extent" do
    counters = four_block_metal_counters
    names = HyNomasFourBlockProbe.shapes.keys.sort

    shifted = four_block_admission_clone(four_block_admission_document)
    metadata = shifted["tensors"].as_h[names[1]].as_h
    metadata["offset_bytes"] = JSON.parse((metadata["offset_bytes"].as_i64 + 4).to_s)
    four_block_admission_rejects(shifted)

    overlap = four_block_admission_clone(four_block_admission_document)
    overlap["tensors"].as_h[names[1]].as_h["offset_bytes"] = JSON.parse("0")
    four_block_admission_rejects(overlap)

    trailing = four_block_admission_clone(four_block_admission_document)
    trailing["tensor_file"].as_h["nbytes"] = JSON.parse((HyNomasFourBlockProbe::SIDECAR_BYTES + 4).to_s)
    four_block_admission_rejects(trailing)
    four_block_assert_no_metal_activity(counters)
  end

  it "requires the MPS single-terminal-readback producer receipt" do
    counters = four_block_metal_counters

    wrong_device = four_block_admission_clone(four_block_admission_document)
    wrong_device["producer"].as_h["device"] = JSON.parse(%("cpu"))
    four_block_admission_rejects(wrong_device)

    wrong_terminal = four_block_admission_clone(four_block_admission_document)
    wrong_terminal["producer"].as_h["activation_readback"] = JSON.parse(%("per-tensor readbacks"))
    four_block_admission_rejects(wrong_terminal)

    wrong_inventory = four_block_admission_clone(four_block_admission_document)
    wrong_inventory["producer"].as_h["captures_per_block"] = JSON.parse(%(["img.output"]))
    four_block_admission_rejects(wrong_inventory)
    four_block_assert_no_metal_activity(counters)
  end

  it "rejects a named-byte budget one byte below the full-J estimate" do
    counters = four_block_metal_counters
    required = HyNomasFourBlockProbe.named_bytes(1_208_455_168_i64)[:total]

    required.should eq(3_555_779_072_i64)
    HyNomasFourBlockProbe.admit_budget!(required, required)
    expect_raises(ArgumentError) do
      HyNomasFourBlockProbe.admit_budget!(required, required - 1)
    end
    four_block_assert_no_metal_activity(counters)
  end
  it "requires finite complete outputs and intact predecessors for a wrong-block control" do
    counters = four_block_metal_counters
    {"block2-as-block0" => 2, "block3-as-block2" => 3}.each do |mode, fault|
      rows = (0...4).flat_map do |index|
        HyNomasFourBlockProbe::CAPTURE_NAMES.map do |name|
          {name: "block#{index}.#{name}", finite: true, failures: index >= fault ? 1 : 0}
        end
      end
      HyNomasFourBlockProbe.wrong_block_detected?(mode, rows).should be_true
      HyNomasFourBlockProbe.wrong_block_detected?("positive", rows).should be_false
      HyNomasFourBlockProbe.wrong_block_detected?(mode, rows[1..]).should be_false
      HyNomasFourBlockProbe.wrong_block_detected?(mode, rows + [rows.last]).should be_false
      damaged = rows.dup
      damaged[0] = damaged[0].merge({failures: 1})
      HyNomasFourBlockProbe.wrong_block_detected?(mode, damaged).should be_false
      nonfinite = rows.dup
      nonfinite[-1] = nonfinite[-1].merge({finite: false})
      HyNomasFourBlockProbe.wrong_block_detected?(mode, nonfinite).should be_false
      missed = rows.dup
      slot = missed.index { |row| row[:name] == "block#{fault}.img.output" }.not_nil!
      missed[slot] = missed[slot].merge({failures: 0})
      HyNomasFourBlockProbe.wrong_block_detected?(mode, missed).should be_false
    end
    four_block_assert_no_metal_activity(counters)
  end

  it "rejects reordered block provenance and truncated producer invocation counts" do
    counters = four_block_metal_counters
    wrong_order = four_block_admission_clone(four_block_admission_document)
    wrong_order["source"].as_h["block_order"] = JSON.parse(%(["block0","block1","block3","block2"]))
    four_block_admission_rejects(wrong_order)
    too_short = four_block_admission_clone(four_block_admission_document)
    too_short["producer"].as_h["attention_adapter_calls"] = JSON.parse("2")
    four_block_admission_rejects(too_short)
    four_block_assert_no_metal_activity(counters)
  end
end
