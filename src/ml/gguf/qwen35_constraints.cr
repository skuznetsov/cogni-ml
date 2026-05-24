require "./qwen35_tokenizer"

module ML::GGUF
  # Tokenizer-aware frontiers for exact constrained decode.
  #
  # This helper intentionally covers only finite literal corridors. A caller can
  # use it for certified grammar states such as JSON/XML punctuation or fixed
  # tool/function names, then fall back to unconstrained decode for free-form
  # string/value spans.
  module Qwen35Constraints
    def self.literal_frontier_ids(tokenizer : Qwen35Tokenizer,
                                  remaining_literals : Array(String)) : Array(Int32)
      return [] of Int32 if remaining_literals.empty?

      allowed = [] of Int32
      tokenizer.vocab.each_with_index do |_piece, id|
        decoded = begin
          tokenizer.decode_single(id.to_i32)
        rescue
          next
        end
        next if decoded.empty?
        if remaining_literals.any? { |literal| literal.starts_with?(decoded) }
          allowed << id.to_i32
        end
      end
      allowed
    end

    def self.advance_literal_options(remaining_literals : Array(String),
                                     emitted : String) : Array(String)
      return remaining_literals if emitted.empty?

      next_literals = [] of String
      remaining_literals.each do |literal|
        next unless literal.starts_with?(emitted)

        next_literals << literal[emitted.size..]
      end
      next_literals
    end
  end
end
