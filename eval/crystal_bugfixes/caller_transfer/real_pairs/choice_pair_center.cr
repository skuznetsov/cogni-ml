module ChoicePairCenter
  def self.calculate(left : Float32, right : Float32, weight : Float32)
    raise "nonfinite QLoRA pair log-odds" unless left.finite? && right.finite?

    mean = (left.to_f64 + right.to_f64) * 0.5
    # Both margins use logit(X) - logit(Y), independent of their targets.
    {log_odds_a: left, log_odds_b: right, mean_log_odds: mean,
     penalty: weight.to_f64 * mean * mean, scale: (weight.to_f64 * mean).to_f32}
  end
end
