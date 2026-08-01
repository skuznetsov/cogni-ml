# Opt-in fresh-process Darwin CPU measurement for output-only versus named-trace
# retention. Run trellis2_retention_sensor_probe separately first to qualify
# the live-heap/RSS instrument. This probe does not execute Metal and does not
# establish allocator, device-memory, latency, or performance fitness.

{% if flag?(:trellis2_dense_flow_retention_probe) %}
{% unless flag?(:darwin) %}
  {% raise "dense-flow retention probe is admitted only for local Darwin measurement" %}
{% end %}

require "json"
require "../../spec_helper"
require "../../../src/ml/three_d/trellis2/dense_flow"

module Trellis2DenseFlowRetentionProbe
  extend self

  ITERATIONS = 16

  record Snapshot,
    total_bytes : Int64,
    live_managed_bytes : Int64,
    rss_kb : Int64

  def snapshot : Snapshot
    GC.collect
    stats = GC.stats
    Snapshot.new(
      stats.total_bytes.to_i64,
      stats.heap_size.to_i64 - stats.free_bytes.to_i64 - stats.unmapped_bytes.to_i64,
      current_rss_kb
    )
  end

  def current_rss_kb : Int64
    output = IO::Memory.new
    status = Process.run(
      "ps",
      ["-o", "rss=", "-p", Process.pid.to_s],
      output: output,
      error: Process::Redirect::Close
    )
    raise "RSS sensor command failed" unless status.success?
    output.to_s.strip.to_i64? || raise "RSS sensor returned no integer value"
  rescue ex : IO::Error
    raise "RSS sensor unavailable: #{ex.message}"
  end

  def stage : ML::ThreeD::Trellis2::DenseFlowStageCPU
    ML::ThreeD::Trellis2::DenseFlowStageCPU.new(
      resolution: 4,
      in_channels: 4,
      model_channels: 32,
      context_channels: 12,
      out_channels: 6,
      num_heads: 4,
      frequency_dim: 8,
      device: ML::Tensor::Device::CPU
    )
  end

  def reference_output(
    stage : ML::ThreeD::Trellis2::DenseFlowStageCPU,
    voxels : ML::Tensor,
    timesteps : ML::Tensor,
    context : ML::Tensor,
  ) : Array(Float32)
    trace_values = stage.forward_with_trace(voxels, timesteps, context)["output"].to_a
    forward_values = stage.forward(voxels, timesteps, context).to_a
    unless forward_values == trace_values
      raise "forward output differs from forward_with_trace output"
    end
    trace_values
  end
end

describe "TRELLIS.2 dense-flow trace retention" do
  it "measures one warmed output or trace retention window" do
    mode = ENV["T2N2D1D_MODE"]? || raise "T2N2D1D_MODE must be forward or trace"
    unless {"forward", "trace"}.includes?(mode)
      raise "T2N2D1D_MODE must be forward or trace"
    end

    stage = Trellis2DenseFlowRetentionProbe.stage
    voxels = ML::Tensor.zeros(2, 4, 4, 4, 4, device: ML::Tensor::Device::CPU)
    timesteps = ML::Tensor.zeros(2, device: ML::Tensor::Device::CPU)
    context = ML::Tensor.zeros(2, 8, 12, device: ML::Tensor::Device::CPU)
    reference = Trellis2DenseFlowRetentionProbe.reference_output(
      stage, voxels, timesteps, context
    )

    iterations = Trellis2DenseFlowRetentionProbe::ITERATIONS
    before = Trellis2DenseFlowRetentionProbe.snapshot
    held = before
    retained_tensors = 0_i64
    output_elements = 0_i64

    if mode == "forward"
      outputs = Array(ML::Tensor).new(iterations)
      iterations.times { outputs << stage.forward(voxels, timesteps, context) }
      held = Trellis2DenseFlowRetentionProbe.snapshot
      outputs.last.to_a.should eq(reference)
      retained_tensors = outputs.size.to_i64
      output_elements = outputs.sum(0_i64) { |tensor| tensor.numel.to_i64 }
      outputs.clear
    else
      traces = Array(Hash(String, ML::Tensor)).new(iterations)
      iterations.times { traces << stage.forward_with_trace(voxels, timesteps, context) }
      held = Trellis2DenseFlowRetentionProbe.snapshot
      traces.last["output"].to_a.should eq(reference)
      retained_tensors = traces.sum(0_i64) { |trace| trace.size.to_i64 }
      output_elements = traces.sum(0_i64) { |trace| trace["output"].numel.to_i64 }
      traces.clear
    end

    dropped = Trellis2DenseFlowRetentionProbe.snapshot
    held_live_delta = held.live_managed_bytes - before.live_managed_bytes
    released_live = held.live_managed_bytes - dropped.live_managed_bytes
    held_live_delta.should be > 0
    released_live.should be > 0
    output_elements.should eq(iterations.to_i64 * reference.size)
    if mode == "trace"
      retained_tensors.should be > iterations
    else
      retained_tensors.should eq(iterations)
    end

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-dense-flow-retention-v1"
        json.field "scope", "darwin-local-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "live_managed_kind", "post-gc-managed-heap-orientation"
        json.field "rss_kind", "process-resident-orientation"
        json.field "mode", mode
        json.field "retained_results", iterations
        json.field "retained_tensors", retained_tensors
        json.field "output_elements", output_elements
        json.field "allocation_bytes", held.total_bytes - before.total_bytes
        json.field "held_live_managed_delta", held_live_delta
        json.field "held_rss_delta_kb", held.rss_kb - before.rss_kb
        json.field "dropped_live_managed_delta",
          dropped.live_managed_bytes - before.live_managed_bytes
        json.field "dropped_rss_delta_kb", dropped.rss_kb - before.rss_kb
        json.field "released_live_managed_bytes", released_live
      end
    end
    puts measurement
  end
end
{% end %}
