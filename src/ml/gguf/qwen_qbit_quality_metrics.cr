module ML::GGUF
  module QwenQBitQualityMetrics
    extend self

    record Top2,
      first_id : Int32,
      first_logit : Float32,
      second_id : Int32,
      second_logit : Float32 do
      def margin : Float32
        first_logit - second_logit
      end
    end

    record Top2Comparison,
      ranked_matches : Int32,
      set_overlap : Int32,
      exact_top1_covered : Bool,
      exact_top2_covered : Bool,
      first_logit_delta : Float32,
      second_logit_delta : Float32,
      margin_delta : Float32

    def top2(logits : Array(Float32)) : Top2
      raise ArgumentError.new("top-2 requires at least two logits") if logits.size < 2

      best = -Float32::INFINITY
      second = -Float32::INFINITY
      best_id = 0_i32
      second_id = 0_i32
      logits.each_with_index do |value, index|
        raise ArgumentError.new("top-2 logits must be finite") unless value.finite?

        id = index.to_i32
        if value > best || (value == best && id < best_id)
          second = best
          second_id = best_id
          best = value
          best_id = id
        elsif id != best_id && (value > second || (value == second && id < second_id))
          second = value
          second_id = id
        end
      end
      Top2.new(best_id, best, second_id, second)
    end

    def compare_top2(exact : Top2, candidate : Top2) : Top2Comparison
      ranked_matches = 0_i32
      ranked_matches += 1 if exact.first_id == candidate.first_id
      ranked_matches += 1 if exact.second_id == candidate.second_id

      candidate_ids = {candidate.first_id, candidate.second_id}
      exact_top1_covered = candidate_ids.includes?(exact.first_id)
      exact_top2_covered = candidate_ids.includes?(exact.second_id)
      set_overlap = (exact_top1_covered ? 1 : 0) + (exact_top2_covered ? 1 : 0)

      Top2Comparison.new(
        ranked_matches,
        set_overlap,
        exact_top1_covered,
        exact_top2_covered,
        (exact.first_logit - candidate.first_logit).abs,
        (exact.second_logit - candidate.second_logit).abs,
        (exact.margin - candidate.margin).abs,
      )
    end

    def embedding_cosine(a : Array(Float32), b : Array(Float32)) : Float64
      unless !a.empty? && a.size == b.size
        raise ArgumentError.new("embedding cosine requires the same non-zero dimension")
      end

      dot = 0.0_f64
      aa = 0.0_f64
      bb = 0.0_f64
      a.each_with_index do |value, index|
        other = b[index]
        unless value.finite? && other.finite?
          raise ArgumentError.new("embedding cosine requires finite non-zero vectors")
        end

        x = value.to_f64
        y = other.to_f64
        dot += x * y
        aa += x * x
        bb += y * y
      end
      unless aa > 0.0 && bb > 0.0
        raise ArgumentError.new("embedding cosine requires finite non-zero vectors")
      end

      result = dot / (Math.sqrt(aa) * Math.sqrt(bb))
      unless result.finite?
        raise ArgumentError.new("embedding cosine requires finite non-zero vectors")
      end
      result
    end
  end
end
