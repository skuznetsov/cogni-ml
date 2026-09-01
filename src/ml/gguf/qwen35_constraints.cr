require "json"
require "./qwen35_tokenizer"
require "./token_option_corridor"

module ML::GGUF
  # Tokenizer-aware frontiers for exact constrained decode.
  #
  # This helper intentionally covers only finite literal corridors. A caller can
  # use it for certified grammar states such as JSON/XML punctuation or fixed
  # tool/function names, then fall back to unconstrained decode for free-form
  # string/value spans.
  module Qwen35Constraints
    alias TokenOptionCorridor = ML::GGUF::TokenOptionCorridor

    MAX_ENUMERATED_INTEGER_VALUES = 256

    class LiteralFrontierError < Exception
    end

    record LabeledFrontierToken,
      token_id : Int32,
      text : String,
      labels : Array(String)

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

    # An incomplete finite corridor must never fall back to unrestricted
    # decoding. An empty frontier means the tokenizer and grammar cannot
    # represent the next literal byte sequence under the current contract.
    def self.required_literal_frontier_ids(index : TokenTextIndex,
                                           remaining_literals : Array(String)) : Array(Int32)
      return [] of Int32 if remaining_literals.empty? || remaining_literals.any?(&.empty?)

      allowed = literal_frontier_ids(index, remaining_literals)
      if allowed.empty?
        raise LiteralFrontierError.new(
          "incomplete literal corridor has no tokenizer frontier (options=#{remaining_literals.size})")
      end
      allowed
    end

    # Canonically tokenize complete finite options once. The byte-exact decode
    # check prevents a tokenizer implementation or metadata mismatch from
    # turning an acceleration hint into a different grammar.
    def self.token_option_corridor(tokenizer : Qwen35Tokenizer,
                                   literals : Array(String)) : TokenOptionCorridor
      return TokenOptionCorridor.from_options([] of Array(Int32)) if literals.empty?

      options = literals.map do |literal|
        ids = tokenizer.encode(literal, add_bos_override: false)
        if ids.empty? || tokenizer.decode(ids) != literal
          raise LiteralFrontierError.new("literal tokenization is not byte-exact")
        end
        ids
      end
      required_token_option_corridor(options)
    rescue ex : LiteralFrontierError
      raise ex
    rescue ex
      raise LiteralFrontierError.new("literal token-option corridor unavailable: #{ex.message}")
    end

    # Qwen's accelerated grammar cannot represent "stop or continue" at one
    # token frontier, so admitted finite stages must contain no empty option and
    # no option that is a token-prefix of another.
    def self.required_token_option_corridor(options : Array(Array(Int32))) : TokenOptionCorridor
      normalized = options.map(&.dup).uniq
      if normalized.any?(&.empty?)
        raise ArgumentError.new("token-option corridor options must be non-empty")
      end
      normalized.each_with_index do |prefix, prefix_index|
        normalized.each_with_index do |option, option_index|
          next if prefix_index == option_index || prefix.size > option.size
          if option[0, prefix.size] == prefix
            raise ArgumentError.new("token-option corridor options must be prefix-free")
          end
        end
      end
      TokenOptionCorridor.from_options(normalized)
    end

    # Preserve the source label for each finite literal while a constrained
    # decoder consumes a shared prefix. This lets diagnostics distinguish
    # tokenizer alternatives for one choice from the first real choice between
    # labels without changing the grammar itself.
    def self.labeled_literal_frontier(index : TokenTextIndex,
                                      remaining_by_label : Hash(String, String)) : Array(LabeledFrontierToken)
      literal_frontier_ids(index, remaining_by_label.values).map do |token_id|
        text = index.text_for_id(token_id)
        labels = remaining_by_label.compact_map do |label, literal|
          label if literal.starts_with?(text)
        end
        LabeledFrontierToken.new(token_id, text, labels)
      end
    end

    def self.labeled_frontier_diverged?(frontier : Array(LabeledFrontierToken)) : Bool
      return false if frontier.empty?

      labels = frontier.flat_map(&.labels).uniq
      label_sets = frontier.map { |candidate| candidate.labels.sort }.uniq
      return false unless labels.size >= 2 && label_sets.size >= 2

      all_labels = labels.sort
      frontier.none? { |candidate| candidate.labels.sort == all_labels }
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

    def self.advance_labeled_literal_options(remaining_by_label : Hash(String, String),
                                             emitted : String) : Hash(String, String)
      return remaining_by_label if emitted.empty?

      next_by_label = {} of String => String
      remaining_by_label.each do |label, literal|
        next unless literal.starts_with?(emitted)

        next_by_label[label] = literal[emitted.size..]
      end
      next_by_label
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

    def self.tool_optional_parameters(tools : Array(JSON::Any)) : Hash(String, Array(String))
      optional_by_name = {} of String => Array(String)
      tools.each do |tool|
        obj = tool.as_h?
        next unless obj
        function = obj["function"]?.try(&.as_h?)
        next unless function
        name = function["name"]?.try(&.as_s?)
        next unless name && !name.empty?

        parameters = function["parameters"]?.try(&.as_h?)
        properties = parameters.try { |p| p["properties"]?.try(&.as_h?) }
        required = parameters.try { |p| p["required"]?.try(&.as_a?) }
        required_names = Set(String).new((required || [] of JSON::Any).compact_map(&.as_s?))
        optional_by_name[name] = if properties
                                   properties.keys.reject { |key| required_names.includes?(key) }
                                 else
                                   [] of String
                                 end
      end
      optional_by_name
    end

    def self.tool_finite_parameter_value_options(tools : Array(JSON::Any)) : Hash(String, Hash(String, Array(String)))
      by_function = {} of String => Hash(String, Array(String))
      tools.each do |tool|
        obj = tool.as_h?
        next unless obj
        function = obj["function"]?.try(&.as_h?)
        next unless function
        name = function["name"]?.try(&.as_s?)
        next unless name && !name.empty?

        parameters = function["parameters"]?.try(&.as_h?)
        properties = parameters.try { |p| p["properties"]?.try(&.as_h?) }
        next unless properties

        by_parameter = {} of String => Array(String)
        properties.each do |parameter_name, raw_schema|
          schema = raw_schema.as_h?
          next unless schema

          values = finite_schema_values(schema)
          next if values.empty?

          by_parameter[parameter_name] = qwen_parameter_value_options(values)
        end
        by_function[name] = by_parameter unless by_parameter.empty?
      end
      by_function
    end

    def self.qwen_tool_call_prefix_options(function_names : Array(String)) : Array(String)
      function_names.reject(&.empty?).uniq.map do |name|
        "<tool_call>\n<function=#{name}>\n"
      end
    end

    def self.labeled_tool_call_prefixes(function_names : Array(String)) : Hash(String, String)
      prefixes = {} of String => String
      function_names.reject(&.empty?).uniq.each do |name|
        prefixes[name] = "<tool_call>\n<function=#{name}>\n"
      end
      prefixes
    end

    def self.qwen_tool_required_parameter_prefix_options(tools : Array(JSON::Any)) : Array(String)
      required_by_name = tool_required_parameters(tools)
      options = [] of String
      tool_function_names(tools).each do |name|
        required = required_by_name[name]? || [] of String
        if required.empty?
          options << "<tool_call>\n<function=#{name}>\n"
        else
          required.each do |parameter_name|
            options << "<tool_call>\n<function=#{name}>\n<parameter=#{parameter_name}>\n"
          end
        end
      end
      options
    end

    def self.qwen_tool_finite_call_options(tools : Array(JSON::Any)) : Array(String)
      finite_by_name = tool_finite_parameter_value_options(tools)
      options = [] of String
      tool_function_names(tools).each do |name|
        parameter_values = finite_by_name[name]?
        next unless parameter_values

        parameter_values.each do |parameter_name, values|
          values.each do |value|
            options << "<tool_call>\n<function=#{name}>\n<parameter=#{parameter_name}>\n#{value}</parameter>\n</function>\n</tool_call>"
          end
        end
      end
      options
    end

    def self.qwen_parameter_open_options(parameter_names : Array(String)) : Array(String)
      parameter_names.reject(&.empty?).uniq.map do |name|
        "<parameter=#{name}>\n"
      end
    end

    def self.qwen_single_parameter_close_options : Array(String)
      ["</parameter>\n</function>\n</tool_call>"]
    end

    def self.qwen_parameter_continue_options(parameter_names : Array(String)) : Array(String)
      parameter_names.reject(&.empty?).uniq.map do |name|
        "</parameter>\n<parameter=#{name}>\n"
      end
    end

    def self.qwen_parameter_value_options(values : Array(String)) : Array(String)
      values.reject(&.empty?).uniq.map { |value| "#{value}\n" }
    end

    private def self.finite_schema_values(schema : Hash(String, JSON::Any)) : Array(String)
      enum_values = schema["enum"]?.try(&.as_a?)
      if enum_values
        return enum_values.compact_map { |value| json_scalar_to_text(value) }
      end

      type_name = schema["type"]?.try(&.as_s?)
      return ["true", "false"] if type_name == "boolean"

      if type_name == "integer"
        minimum = json_integer(schema["minimum"]?)
        maximum = json_integer(schema["maximum"]?)
        if minimum && maximum && maximum >= minimum && (maximum - minimum) < MAX_ENUMERATED_INTEGER_VALUES
          return (minimum..maximum).map(&.to_s)
        end
      end

      [] of String
    end

    private def self.json_integer(value : JSON::Any?) : Int64?
      return nil unless value

      case raw = value.raw
      when Int64
        raw
      when Float64
        raw.to_i64 if raw.finite? && raw == raw.trunc
      else
        nil
      end
    end

    private def self.json_scalar_to_text(value : JSON::Any) : String?
      case raw = value.raw
      when String
        raw
      when Bool
        raw ? "true" : "false"
      when Int64, Float64
        raw.to_s
      else
        nil
      end
    end
  end
end
