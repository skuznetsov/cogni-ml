require "json"
require "set"

# CPU-only metadata guard for the fixed DRAMA-3 decode replay capture.
module Drama3DecodeReplay
  COVERAGE_FRAMES = [0, 29, 59] of Int32

  private def self.expected_projection_names(frame : Int32) : Set(String)
    names = Set(String).new

    40.times do |layer|
      prefix = "backbone.layers.#{layer}.mlp"
      names << "#{prefix}.gate.weight"
      names << "#{prefix}.shared_expert_gate.weight"
      {"shared_expert", "switch_mlp"}.each do |expert|
        {"gate_proj", "up_proj", "down_proj"}.each do |projection|
          names << "#{prefix}.#{expert}.#{projection}.weight"
        end
      end

      if layer % 4 == 3
        {"q_proj", "k_proj", "v_proj", "o_proj"}.each do |projection|
          names << "backbone.layers.#{layer}.self_attn.#{projection}.weight"
        end
      else
        {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}.each do |projection|
          names << "backbone.layers.#{layer}.linear_attn.#{projection}.weight"
        end
      end
    end

    {"shared_layers" => 4, "residual_layers" => 9}.each do |kind, count|
      count.times do |layer|
        prefix = "tower.#{kind}.#{layer}"
        {"eh_proj", "transformer_layer.attention.wqkv", "transformer_layer.attention.wo",
         "transformer_layer.feed_forward.w1", "transformer_layer.feed_forward.w3",
         "transformer_layer.feed_forward.w2"}.each do |projection|
          names << "#{prefix}.#{projection}.weight"
        end
      end
    end

    9.times do |head|
      names << "tower.codebook_heads.#{head}.weight"
    end
    names << "lm_head.weight" if frame == 29 || frame == 59
    names
  end

  # Assert the exact selected-frame projection-name inventory. This deliberately
  # validates names per row rather than accepting an aggregate record count.
  def self.assert_coverage!(records : Array(JSON::Any)) : Nil
    expected = COVERAGE_FRAMES.to_h { |frame| {frame, expected_projection_names(frame)} }
    observed = COVERAGE_FRAMES.to_h { |frame| {frame, Set(String).new} }

    records.each_with_index do |record, index|
      frame_value = record["frame"]?.try(&.as_i) ||
                    raise ArgumentError.new("capture record #{index} is missing an integer frame")
      unless COVERAGE_FRAMES.any? { |frame| frame.to_i64 == frame_value }
        raise ArgumentError.new("unexpected capture frame #{frame_value}")
      end
      frame = frame_value.to_i32
      name = record["name"]?.try(&.as_s) ||
             raise ArgumentError.new("capture record #{index} is missing a projection name")

      unless expected[frame].includes?(name)
        raise ArgumentError.new("unexpected projection at frame #{frame}: #{name}")
      end
      if observed[frame].includes?(name)
        raise ArgumentError.new("duplicate projection at frame #{frame}: #{name}")
      end
      observed[frame] << name
    end

    COVERAGE_FRAMES.each do |frame|
      raise ArgumentError.new("missing capture frame #{frame}") if observed[frame].empty?
      missing = expected[frame].reject { |name| observed[frame].includes?(name) }.to_a.sort
      unless missing.empty?
        raise ArgumentError.new("missing projection at frame #{frame}: #{missing.first}")
      end
    end
  end

  # Different shapes may share a family, but kernel and arithmetic must agree.
  def self.assert_homogeneous!(records : Array(JSON::Any)) : Nil
    records.group_by { |record| {record["frame"].as_i, record["family"].as_s} }.each do |key, projections|
      signatures = projections.map do |record|
        {record["kernel"], record["reduction"], record["dtype"], record["bits"],
         record["group_size"], record["threadgroup_width"]}
      end.uniq
      raise ArgumentError.new("heterogeneous replay group #{key}") unless signatures.size == 1
    end
  end
end
