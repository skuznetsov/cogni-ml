require "json"
require "./qwen35_tokenizer"

module ML::GGUF
  # Tokenizer-aware frontiers for exact constrained decode.
  #
  # This helper intentionally covers only finite literal corridors. A caller can
  # use it for certified grammar states such as JSON/XML punctuation or fixed
  # tool/function names, then fall back to unconstrained decode for free-form
  # string/value spans.
  module Qwen35Constraints
    class TokenTextIndex
      @texts : Array(String)
      @by_first : Hash(Char, Array({Int32, String}))

      def initialize(tokenizer : Qwen35Tokenizer)
        @texts = Array(String).new(tokenizer.vocab.size, "")
        @by_first = Hash(Char, Array({Int32, String})).new { |h, k| h[k] = [] of {Int32, String} }
        tokenizer.vocab.each_index do |id|
          text = begin
            tokenizer.decode_single(id.to_i32)
          rescue
            ""
          end
          @texts[id] = text
          next if text.empty?

          @by_first[text[0]] << {id.to_i32, text}
        end
      end

      def text_for_id(id : Int32) : String
        return "" if id < 0 || id >= @texts.size

        @texts[id]
      end

      def literal_frontier_ids(remaining_literals : Array(String)) : Array(Int32)
        return [] of Int32 if remaining_literals.empty?

        allowed = [] of Int32
        seen = Set(Int32).new
        remaining_literals.each do |literal|
          next if literal.empty?

          bucket = @by_first[literal[0]]?
          next unless bucket

          bucket.each do |id, decoded|
            next if seen.includes?(id)
            next unless literal.starts_with?(decoded)

            allowed << id
            seen << id
          end
        end
        allowed
      end
    end

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

    def self.literal_frontier_ids(index : TokenTextIndex,
                                  remaining_literals : Array(String)) : Array(Int32)
      index.literal_frontier_ids(remaining_literals)
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

    def self.tool_function_names(tools : Array(JSON::Any)) : Array(String)
      names = [] of String
      tools.each do |tool|
        obj = tool.as_h?
        next unless obj
        function = obj["function"]?.try(&.as_h?)
        name = function.try { |f| f["name"]?.try(&.as_s?) }
        names << name.not_nil! if name && !name.empty?
      end
      names.uniq
    end

    def self.tool_required_parameters(tools : Array(JSON::Any)) : Hash(String, Array(String))
      required_by_name = {} of String => Array(String)
      tools.each do |tool|
        obj = tool.as_h?
        next unless obj
        function = obj["function"]?.try(&.as_h?)
        next unless function
        name = function["name"]?.try(&.as_s?)
        next unless name && !name.empty?

        parameters = function["parameters"]?.try(&.as_h?)
        required = parameters.try { |p| p["required"]?.try(&.as_a?) }
        required_by_name[name] = if required
                                   required.compact_map(&.as_s?)
                                 else
                                   [] of String
                                 end
      end
      required_by_name
    end

    def self.qwen_tool_call_prefix_options(function_names : Array(String)) : Array(String)
      function_names.reject(&.empty?).uniq.map do |name|
        "<tool_call>\n<function=#{name}>\n"
      end
    end

    def self.qwen_parameter_open_options(parameter_names : Array(String)) : Array(String)
      parameter_names.reject(&.empty?).uniq.map do |name|
        "<parameter=#{name}>\n"
      end
    end

    def self.qwen_single_parameter_close_options : Array(String)
      ["</parameter>\n</function>\n</tool_call>"]
    end
  end
end
