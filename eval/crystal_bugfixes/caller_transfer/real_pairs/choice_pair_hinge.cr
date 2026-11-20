require "../../../../src/ml/gguf/qwen35_qlora"

module ChoicePairHinge
  def self.calculate(left : Float32, right : Float32, left_correct_is_x : Bool,
                     weight : Float32, minimum_margin : Float32)
    sign = left_correct_is_x ? 1_f32 : -1_f32
    left_margin = sign * left
    right_margin = -sign * right
    hinge = ML::GGUF::Qwen35QLoRA::PairedObjective.calculate(0.0, left_margin, right_margin, weight, minimum_margin)

    # The output head differentiates X-minus-Y on both sides.
    {margin_a: left_margin, margin_b: right_margin, penalty: hinge[:loss],
     scale_a: sign * hinge[:anchor_scale], scale_b: -sign * hinge[:mate_scale]}
  end
end
