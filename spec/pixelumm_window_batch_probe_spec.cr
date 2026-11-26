require "spec"
require "../examples/support/pixelumm_window_batch_probe"

class PixelUMMWindowBatchWorkspaceToken
end

module PixelUMMWindowBatchProbeSpecHelpers
  extend self

  def contracts(workspace : PixelUMMWindowBatchWorkspaceToken = PixelUMMWindowBatchWorkspaceToken.new,
                slots : Array(Int32) = (0...36).map(&.to_i32),
                geometry : PixelUMMWindowBatchProbe::PlanGeometry = {6, 44, 80, 8, 4096, 12_288, 32, 8, 128,
                                                                     1e-6_f32, 1_000_000_f32, 10_000_f32},
                workspace_slot_count : Int32 = 36,
                reuse_outputs : Bool = true,
                output_bank_count : Int32 = 2) : Array(PixelUMMWindowBatchProbe::PlanContract(PixelUMMWindowBatchWorkspaceToken))
    slots.map do |slot|
      PixelUMMWindowBatchProbe::PlanContract.new(workspace, slot, geometry,
        workspace_slot_count, reuse_outputs, output_bank_count)
    end
  end
end

describe PixelUMMWindowBatchProbe do
  it "requires all 36 serial slots to share one workspace and geometry" do
    PixelUMMWindowBatchProbe.validate_plan_contracts!(PixelUMMWindowBatchProbeSpecHelpers.contracts)
    PixelUMMWindowBatchProbe.validate_plan_contracts!(PixelUMMWindowBatchProbeSpecHelpers.contracts(
      slots: [0], workspace_slot_count: 1, reuse_outputs: false, output_bank_count: 1))

    expect_raises(ArgumentError, "same PixelUMM workspace") do
      items = PixelUMMWindowBatchProbeSpecHelpers.contracts
      items[-1] = PixelUMMWindowBatchProbe::PlanContract.new(
        PixelUMMWindowBatchWorkspaceToken.new, 35, items.last.geometry, 36, true, 2)
      PixelUMMWindowBatchProbe.validate_plan_contracts!(items)
    end

    expect_raises(ArgumentError, "serial slot order 0..35") do
      items = PixelUMMWindowBatchProbeSpecHelpers.contracts
      items[0], items[1] = items[1], items[0]
      PixelUMMWindowBatchProbe.validate_plan_contracts!(items)
    end

    expect_raises(ArgumentError, "serial slot order 0..35") do
      items = PixelUMMWindowBatchProbeSpecHelpers.contracts
      items[-1] = PixelUMMWindowBatchProbe::PlanContract.new(
        items.last.workspace, 34, items.last.geometry, 36, true, 2)
      PixelUMMWindowBatchProbe.validate_plan_contracts!(items)
    end

    expect_raises(ArgumentError, "homogeneous plan geometry") do
      items = PixelUMMWindowBatchProbeSpecHelpers.contracts
      items[-1] = PixelUMMWindowBatchProbe::PlanContract.new(
        items.last.workspace, 35, {6, 44, 88, 8, 4096, 12_288, 32, 8, 128,
                                   1e-6_f32, 1_000_000_f32, 10_000_f32}, 36, true, 2)
      PixelUMMWindowBatchProbe.validate_plan_contracts!(items)
    end

    expect_raises(ArgumentError, "36-slot reused-output workspace with two banks") do
      items = PixelUMMWindowBatchProbeSpecHelpers.contracts
      items[-1] = PixelUMMWindowBatchProbe::PlanContract.new(
        items.last.workspace, 35, items.last.geometry, 35, true, 2)
      PixelUMMWindowBatchProbe.validate_plan_contracts!(items)
    end

    expect_raises(ArgumentError, "36-slot reused-output workspace with two banks") do
      items = PixelUMMWindowBatchProbeSpecHelpers.contracts
      items[-1] = PixelUMMWindowBatchProbe::PlanContract.new(
        items.last.workspace, 35, items.last.geometry, 36, false, 2)
      PixelUMMWindowBatchProbe.validate_plan_contracts!(items)
    end

    expect_raises(ArgumentError, "36-slot reused-output workspace with two banks") do
      items = PixelUMMWindowBatchProbeSpecHelpers.contracts
      items[-1] = PixelUMMWindowBatchProbe::PlanContract.new(
        items.last.workspace, 35, items.last.geometry, 36, true, 36)
      PixelUMMWindowBatchProbe.validate_plan_contracts!(items)
    end
  end

  it "requires both saved upstream attention and block output before allocation" do
    PixelUMMWindowBatchProbe.validate_capture_inventory!(["attention", "output"])
    expect_raises(ArgumentError, "exactly attention and output") do
      PixelUMMWindowBatchProbe.validate_capture_inventory!(["output"])
    end
    expect_raises(ArgumentError) do
      PixelUMMWindowBatchProbe.validate_capture_inventory!(["output", "attention"])
    end
  end

  it "pairs all full-J windows without changing either reduction geometry" do
    layout = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80).layout
    pairs = PixelUMMWindowBatchProbe.pairs(layout)
    pairs.size.should eq(30)
    pairs.count { |p| {p.queries, p.keys} == {384, 3456} }.should eq(20)
    pairs.count { |p| {p.queries, p.keys} == {384, 2880} }.should eq(5)
    pairs.count { |p| {p.queries, p.keys} == {192, 2880} }.should eq(5)
    pairs.flat_map { |p| layout.query_indices(p.first) + layout.query_indices(p.second) }.sort.should eq((0...21120).to_a)
    PixelUMMWindowBatchProbe::EXTRA_PANEL_BYTES.should eq(931135488_i64)
    PixelUMMWindowBatchProbe.expected_elements(1, layout).should eq(2229534720_u32)
    PixelUMMWindowBatchProbe.expected_elements(2, layout).should eq(86507520_u32)
    PixelUMMWindowBatchProbe.expected_elements(3, layout).should eq(84934656_u32)
    expect_raises(ArgumentError) { PixelUMMWindowBatchProbe.expected_elements(0, layout) }
  end

  it "retains distinct contexts even when both windows have the same shape" do
    layout = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80).layout
    pair = PixelUMMWindowBatchProbe.pairs(layout).find do |p|
      layout.key_indices(p.first) != layout.key_indices(p.second)
    end.not_nil!
    PixelUMMWindowBatchProbe.validate_lane_context!(layout, pair.first,
      layout.query_indices(pair.first), layout.key_indices(pair.first))
    expect_raises(ArgumentError, "ordered query/key context") do
      PixelUMMWindowBatchProbe.validate_lane_context!(layout, pair.first,
        layout.query_indices(pair.first), layout.key_indices(pair.second))
    end
    expect_raises(ArgumentError, "ordered query/key context") do
      PixelUMMWindowBatchProbe.validate_lane_context!(layout, pair.first,
        layout.query_indices(pair.first).reverse, layout.key_indices(pair.first))
    end
  end

  it "rejects duplicate coverage and mixing partial border windows with full windows" do
    layout = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80).layout
    pairs = PixelUMMWindowBatchProbe.pairs(layout)
    duplicate = pairs.dup
    duplicate[-1] = duplicate.first
    expect_raises(ArgumentError, "exactly once") { PixelUMMWindowBatchProbe.validate_pairs!(layout, duplicate) }
    wrong = pairs.dup
    p = wrong.last
    wrong[-1] = PixelUMMWindowBatchProbe::Pair.new(p.first, p.second, p.queries + 1, p.keys)
    expect_raises(ArgumentError, "reduction geometry") { PixelUMMWindowBatchProbe.validate_pairs!(layout, wrong) }
  end
end
