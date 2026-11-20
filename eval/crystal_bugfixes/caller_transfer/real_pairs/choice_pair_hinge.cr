require "../../../../src/ml/gguf/qwen35_qlora"

module ChoicePairHinge
  def self.calculate(left : Float32, right : Float32, left_correct_is_x : Bool,
                     weight : Float32, minimum_margin : Float32, *, on_difference : Bool = false)
    sign = left_correct_is_x ? 1_f32 : -1_f32
    left_margin = sign * left
    right_margin = -sign * right
    anchor, mate = left_margin, right_margin
    if on_difference
      # The mean of the two correct-label margins is half the directed
      # difference of the raw margins. The tie splits its slope equally.
      anchor = mate = ((left_margin.to_f64 + right_margin.to_f64) * 0.5).to_f32
    end
    hinge = ML::GGUF::Qwen35QLoRA::PairedObjective.calculate(0.0, anchor, mate, weight, minimum_margin)

    # The output head differentiates X-minus-Y on both sides.
    {margin_a: left_margin, margin_b: right_margin, penalty: hinge[:loss],
     scale_a: sign * hinge[:anchor_scale], scale_b: -sign * hinge[:mate_scale]}
  end
end
