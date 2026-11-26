require "spec"
require "digest/sha256"
require "../examples/support/pixelumm_ffn_graph_probe"

private def pixelumm_ffn_graph_slot(name : String) : Int32
  PixelUMMFFNGraphProbe::STAGE_NAMES.index(name).not_nil!.to_i32
end

private def pixelumm_ffn_graph_valid_metrics(topology = PixelUMMFFNGraphProbe::Topology::StageOutputs)
  counts = PixelUMMFFNGraphProbe.expected_stage_counts(topology)
  counts.map_with_index do |elements, slot|
    if slot == PixelUMMFFNGraphProbe::STAGE_NAMES.size - 1
      {failures: 1_u32, different_words: 0_u32, nonfinite: 1_u32,
       elements: elements.to_u32}
    else
      {failures: 0_u32, different_words: 0_u32, nonfinite: 0_u32,
       elements: elements.to_u32}
    end
  end
end

describe PixelUMMFFNGraphProbe do
  it "selects full, split-down and mandatory stage-output topologies explicitly" do
    PixelUMMFFNGraphProbe::Topology::FullGraph.label.should eq("full_graph")
    PixelUMMFFNGraphProbe::Topology::SplitDown.label.should eq("split_down")
    PixelUMMFFNGraphProbe::Topology::StageOutputs.label.should eq("stage_outputs")
    PixelUMMFFNGraphProbe::Topology.select.should eq(
      PixelUMMFFNGraphProbe::Topology::FullGraph)
    PixelUMMFFNGraphProbe::Topology.select(split_down: true).should eq(
      PixelUMMFFNGraphProbe::Topology::SplitDown)
    PixelUMMFFNGraphProbe::Topology.select(stage_outputs: true).should eq(
      PixelUMMFFNGraphProbe::Topology::StageOutputs)
    PixelUMMFFNGraphProbe::Topology.select(true).should eq(
      PixelUMMFFNGraphProbe::Topology::SplitDown)
    PixelUMMFFNGraphProbe::Topology.parse("full_graph").should eq(
      PixelUMMFFNGraphProbe::Topology::FullGraph)
    PixelUMMFFNGraphProbe::Topology.parse("split_down").should eq(
      PixelUMMFFNGraphProbe::Topology::SplitDown)
    PixelUMMFFNGraphProbe::Topology.parse("stage_outputs").should eq(
      PixelUMMFFNGraphProbe::Topology::StageOutputs)
    PixelUMMFFNGraphProbe::Topology::SplitDown.split_down?.should be_true
    PixelUMMFFNGraphProbe::Topology::StageOutputs.stage_outputs?.should be_true
    PixelUMMFFNGraphProbe::Topology::StageOutputs.uses_split_down?.should be_true
    PixelUMMFFNGraphProbe::Topology::FullGraph.uses_split_down?.should be_false
    expect_raises(ArgumentError) do
      PixelUMMFFNGraphProbe::Topology.parse("unknown")
    end
    expect_raises(ArgumentError) do
      PixelUMMFFNGraphProbe::Topology.select(split_down: true, stage_outputs: true)
    end
  end

  it "requires the exact per-path graph operation counts for all topologies" do
    full = PixelUMMFFNGraphProbe::Topology::FullGraph
    split = PixelUMMFFNGraphProbe::Topology::SplitDown
    stages = PixelUMMFFNGraphProbe::Topology::StageOutputs
    PixelUMMFFNGraphProbe.validate_candidate_counts!(
      full, PixelUMMFFNGraphProbe::FULL_GRAPH_OPERATION_COUNTS).should be_empty
    PixelUMMFFNGraphProbe.validate_candidate_counts!(
      split, PixelUMMFFNGraphProbe::SPLIT_DOWN_OPERATION_COUNTS).should be_empty
    PixelUMMFFNGraphProbe.validate_candidate_counts!(
      stages, PixelUMMFFNGraphProbe::SPLIT_DOWN_OPERATION_COUNTS).should be_empty
    PixelUMMFFNGraphProbe.expected_candidate_counts(stages).should eq(
      PixelUMMFFNGraphProbe::SPLIT_DOWN_OPERATION_COUNTS)

    undercount = PixelUMMFFNGraphProbe::SPLIT_DOWN_OPERATION_COUNTS.dup
    undercount[3] = 0_u64
    errors = PixelUMMFFNGraphProbe.validate_candidate_counts!(split, undercount)
    errors.should eq(["captured_split_down_graph_encodes: observed 0 expected 1 for split_down"])

    extra = PixelUMMFFNGraphProbe::FULL_GRAPH_OPERATION_COUNTS.dup
    extra[1] = 1_u64
    PixelUMMFFNGraphProbe.validate_candidate_counts!(full, extra).should eq(
      ["output_only_split_down_graph_encodes: observed 1 expected 0 for full_graph"])
    expect_raises(ArgumentError) do
      PixelUMMFFNGraphProbe.validate_candidate_counts!(full, [1_u64, 1_u64])
    end
  end

  it "reports candidate operation counts by named topology path" do
    PixelUMMFFNGraphProbe.operation_count_report(
      PixelUMMFFNGraphProbe::SPLIT_DOWN_OPERATION_COUNTS).should eq({
        "output_only_topology_graph_encodes" => 1_u64,
        "output_only_split_down_graph_encodes" => 1_u64,
        "captured_topology_graph_encodes" => 1_u64,
        "captured_split_down_graph_encodes" => 1_u64,
        "reference_silu_kernel_encodes" => 1_u64,
      })
  end

  it "hashes empty and small files with their known SHA-256 values" do
    {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
    }.each do |contents, expected|
      file = File.tempfile("pixelumm-ffn-hash-", ".bin")
      begin
        file.print(contents)
        file.close
        PixelUMMFFNGraphProbe.sha_file(file.path).should eq(expected)
        PixelUMMFFNGraphProbe.sha_file(file.path).should eq(Digest::SHA256.new.file(file.path).hexfinal)
      ensure
        file.delete
      end
    end
  end

  it "hashes content beyond its one-megabyte read chunk like Digest.file" do
    payload = Bytes.new(1024 * 1024 + 37) { |index| ((index % 251) + 4).to_u8 }
    file = File.tempfile("pixelumm-ffn-hash-large-", ".bin")
    begin
      file.write(payload)
      file.close
      PixelUMMFFNGraphProbe.sha_file(file.path).should eq(Digest::SHA256.new.file(file.path).hexfinal)
    ensure
      file.delete
    end
  end

  it "accepts exact numerical coverage for every stage and the injected-NaN control" do
    PixelUMMFFNGraphProbe.validate_metrics!(
      pixelumm_ffn_graph_valid_metrics, PixelUMMFFNGraphProbe::Topology::StageOutputs).should be_empty
    PixelUMMFFNGraphProbe.validate_metrics!(
      pixelumm_ffn_graph_valid_metrics(PixelUMMFFNGraphProbe::Topology::FullGraph),
      PixelUMMFFNGraphProbe::Topology::FullGraph).should be_empty
    PixelUMMFFNGraphProbe.validate_metrics!(
      pixelumm_ffn_graph_valid_metrics(PixelUMMFFNGraphProbe::Topology::SplitDown),
      PixelUMMFFNGraphProbe::Topology::SplitDown).should be_empty
  end

  it "includes output-only product and stage parity rows before captured rows" do
    product_slot = pixelumm_ffn_graph_slot("output_only.product_vs_saved")
    gate_slot = pixelumm_ffn_graph_slot("output_only.gate_vs_saved")
    PixelUMMFFNGraphProbe::STAGE_NAMES[product_slot, 2].should eq([
      "output_only.product_vs_saved", "output_only.product_vs_native_baseline"])
    PixelUMMFFNGraphProbe::STAGE_NAMES[gate_slot, 5].should eq([
      "output_only.gate_vs_saved", "output_only.gate_vs_native_baseline",
      "output_only.up_vs_saved", "output_only.up_vs_native_baseline",
      "output_only.silu_vs_native_baseline"])
    PixelUMMFFNGraphProbe::STAGE_COUNTS[product_slot, 2].should eq([
      PixelUMMFFNGraphProbe::C::PRODUCT_ELEMENTS,
      PixelUMMFFNGraphProbe::C::PRODUCT_ELEMENTS])
    front_slots = ["output_only.product_vs_saved", "output_only.product_vs_native_baseline",
                   "output_only.gate_vs_saved", "output_only.gate_vs_native_baseline",
                   "output_only.up_vs_saved", "output_only.up_vs_native_baseline",
                   "output_only.silu_vs_native_baseline"].map { |name| pixelumm_ffn_graph_slot(name) }
    full_counts = PixelUMMFFNGraphProbe.expected_stage_counts(PixelUMMFFNGraphProbe::Topology::FullGraph)
    front_slots.map { |slot| full_counts[slot] }.should eq([0, 0, 0, 0, 0, 0, 0])
    split_counts = PixelUMMFFNGraphProbe.expected_stage_counts(PixelUMMFFNGraphProbe::Topology::SplitDown)
    front_slots.map { |slot| split_counts[slot] }.should eq([
      PixelUMMFFNGraphProbe::C::PRODUCT_ELEMENTS,
      PixelUMMFFNGraphProbe::C::PRODUCT_ELEMENTS, 0, 0, 0, 0, 0])
    stage_counts = PixelUMMFFNGraphProbe.expected_stage_counts(PixelUMMFFNGraphProbe::Topology::StageOutputs)
    front_slots.map { |slot| stage_counts[slot] }.should eq(front_slots.map { |slot| PixelUMMFFNGraphProbe::STAGE_COUNTS[slot] })
  end

  it "rejects nonzero metrics for graph stages inactive in a topology" do
    full_metrics = pixelumm_ffn_graph_valid_metrics(PixelUMMFFNGraphProbe::Topology::FullGraph)
    product_slot = pixelumm_ffn_graph_slot("output_only.product_vs_saved")
    full_metrics[product_slot] = {failures: 0_u32, different_words: 0_u32,
                                  nonfinite: 0_u32, elements: 1_u32}
    PixelUMMFFNGraphProbe.validate_metrics!(
      full_metrics, PixelUMMFFNGraphProbe::Topology::FullGraph).should contain(
        "output_only.product_vs_saved: inactive metric must be exactly zero, observed #{full_metrics[product_slot]}")

    split_metrics = pixelumm_ffn_graph_valid_metrics(PixelUMMFFNGraphProbe::Topology::SplitDown)
    gate_slot = pixelumm_ffn_graph_slot("output_only.gate_vs_saved")
    split_metrics[gate_slot] = {failures: 0_u32, different_words: 0_u32,
                                nonfinite: 0_u32, elements: 1_u32}
    PixelUMMFFNGraphProbe.validate_metrics!(
      split_metrics, PixelUMMFFNGraphProbe::Topology::SplitDown).should contain(
        "output_only.gate_vs_saved: inactive metric must be exactly zero, observed #{split_metrics[gate_slot]}")
  end

  it "rejects missing and extra stage rows" do
    valid = pixelumm_ffn_graph_valid_metrics
    expect_raises(ArgumentError) do
      PixelUMMFFNGraphProbe.validate_metrics!(valid[0...-1], PixelUMMFFNGraphProbe::Topology::StageOutputs)
    end
    expect_raises(ArgumentError) do
      PixelUMMFFNGraphProbe.validate_metrics!(valid + [valid.last], PixelUMMFFNGraphProbe::Topology::StageOutputs)
    end
  end

  it "rejects an undercounted stage even when no element failed" do
    metrics = pixelumm_ffn_graph_valid_metrics
    slot = pixelumm_ffn_graph_slot("captured.gate_vs_saved")
    metrics[slot] = {failures: 0_u32, different_words: 0_u32, nonfinite: 0_u32,
                     elements: (PixelUMMFFNGraphProbe::STAGE_COUNTS[slot] - 1).to_u32}

    errors = PixelUMMFFNGraphProbe.validate_metrics!(metrics, PixelUMMFFNGraphProbe::Topology::StageOutputs)
    errors.any? { |error| error.starts_with?("captured.gate_vs_saved: failed/inactive metric") }.should be_true
  end

  it "rejects numeric failures and nonfinite values in captured stages" do
    metrics = pixelumm_ffn_graph_valid_metrics
    saved_silu_slot = pixelumm_ffn_graph_slot("captured.silu_vs_derived_saved")
    baseline_silu_slot = pixelumm_ffn_graph_slot("captured.silu_vs_baseline")
    count = PixelUMMFFNGraphProbe::STAGE_COUNTS[saved_silu_slot].to_u32
    metrics[saved_silu_slot] = {failures: 1_u32, different_words: 1_u32, nonfinite: 0_u32,
                  elements: count}
    metrics[baseline_silu_slot] = {failures: 1_u32, different_words: 0_u32, nonfinite: 1_u32,
                   elements: count}

    errors = PixelUMMFFNGraphProbe.validate_metrics!(metrics, PixelUMMFFNGraphProbe::Topology::StageOutputs)
    errors.any? { |error| error.starts_with?("captured.silu_vs_derived_saved: failed/inactive metric") }.should be_true
    errors.any? { |error| error.starts_with?("captured.silu_vs_baseline: failed/inactive metric") }.should be_true
  end

  it "requires the exact positive-control failure, nonfinite and element counts" do
    slot = pixelumm_ffn_graph_slot("positive_control.injected_nan")
    expected_elements = PixelUMMFFNGraphProbe::STAGE_COUNTS[slot].to_u32
    invalid_controls = [
      {failures: 0_u32, different_words: 0_u32, nonfinite: 1_u32,
       elements: expected_elements},
      {failures: 1_u32, different_words: 1_u32, nonfinite: 1_u32,
       elements: expected_elements},
      {failures: 1_u32, different_words: 0_u32, nonfinite: 0_u32,
       elements: expected_elements},
      {failures: 1_u32, different_words: 0_u32, nonfinite: 1_u32,
       elements: expected_elements - 1_u32},
    ]

    invalid_controls.each do |control|
      metrics = pixelumm_ffn_graph_valid_metrics
      metrics[slot] = control
      PixelUMMFFNGraphProbe.validate_metrics!(
        metrics, PixelUMMFFNGraphProbe::Topology::StageOutputs).should contain(
        "positive_control.injected_nan: expected the exact injected-NaN control")
    end
  end
end
