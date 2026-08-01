# Opt-in fresh-process Darwin CPU allocation falsifier for the bounded rotary
# output path. This compares cumulative managed allocation traffic only. It
# does not establish latency, RSS, native-memory, production-scale, or device
# fitness and never executes Metal.

{% if flag?(:trellis2_conditioning_output_probe) %}
{% unless flag?(:darwin) %}
  {% raise "conditioning output probe is admitted only for local Darwin measurement" %}
{% end %}

require "json"
require "../../spec_helper"
require "../../../src/ml/three_d/trellis2/conditioning"

module Trellis2ConditioningOutputProbe
  extend self

  ITERATIONS = 16
  OUTPUT_TO_TRACE_PERCENT_LIMIT = 80_i64

  def coordinates : ML::Tensor
    values = Array(Float32).new(3 * 512) do |index|
      ((index % 37) - 18).to_f32 * 0.03125_f32
    end
    ML::Tensor.from_array(values, ML::Shape.new(3_i32, 512_i32)).transpose
  end

  def allocation_bytes(& : ->) : Int64
    before = GC.stats.total_bytes.to_i64
    yield
    GC.stats.total_bytes.to_i64 - before
  end

  def forward_window(
    rotary : ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU,
    coordinates : ML::Tensor,
  ) : Int64
    elements = 0_i64
    allocation_bytes do
      ITERATIONS.times do
        elements += rotary.forward(coordinates).numel.to_i64
      end
    end.tap do
      elements.should eq(ITERATIONS.to_i64 * 512_i64 * 64_i64 * 2_i64)
    end
  end

  def trace_window(
    rotary : ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU,
    coordinates : ML::Tensor,
  ) : Int64
    elements = 0_i64
    allocation_bytes do
      ITERATIONS.times do
        trace = rotary.forward_with_trace(coordinates)
        raise "rotary trace must contain three tensors" unless trace.size == 3
        elements += trace["phases"].numel.to_i64
      end
    end.tap do
      elements.should eq(ITERATIONS.to_i64 * 512_i64 * 64_i64 * 2_i64)
    end
  end
end

describe "TRELLIS.2 conditioning output allocation" do
  it "preserves rotary output while removing trace-only allocation traffic" do
    rotary = ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU.new(
      head_dim: 128,
      dim: 3,
      device: ML::Tensor::Device::CPU
    )
    coordinates = Trellis2ConditioningOutputProbe.coordinates
    expected = rotary.forward_with_trace(coordinates)["phases"].to_a
    rotary.forward(coordinates).to_a.should eq(expected)

    2.times do
      rotary.forward(coordinates)
      rotary.forward_with_trace(coordinates)
    end

    forward_first = Trellis2ConditioningOutputProbe.forward_window(rotary, coordinates)
    trace_second = Trellis2ConditioningOutputProbe.trace_window(rotary, coordinates)
    trace_first = Trellis2ConditioningOutputProbe.trace_window(rotary, coordinates)
    forward_second = Trellis2ConditioningOutputProbe.forward_window(rotary, coordinates)
    forward_bytes = Math.min(forward_first, forward_second)
    trace_bytes = Math.min(trace_first, trace_second)

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-conditioning-output-v1"
        json.field "scope", "darwin-local-manual"
        json.field "iterations", Trellis2ConditioningOutputProbe::ITERATIONS
        json.field "coordinates", 512
        json.field "head_dim", 128
        json.field "forward_first_bytes", forward_first
        json.field "forward_second_bytes", forward_second
        json.field "trace_first_bytes", trace_first
        json.field "trace_second_bytes", trace_second
        json.field "forward_to_trace_percent", forward_bytes.to_f64 * 100.0 / trace_bytes.to_f64
      end
    end
    puts measurement

    (forward_bytes * 100_i64).should be <=(
      trace_bytes * Trellis2ConditioningOutputProbe::OUTPUT_TO_TRACE_PERCENT_LIMIT
    )
  end
end
{% end %}
