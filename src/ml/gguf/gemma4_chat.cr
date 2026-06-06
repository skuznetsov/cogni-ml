require "json"
require "./qwen35_chat"

module ML::GGUF
  # Minimal Gemma4 chat-template support for text and native function calls.
  #
  # This intentionally mirrors the Gemma4 template subset used by llama.cpp:
  # turn markers are Gemma-native, and tool calls use
  # `<|tool_call>call:name{arg:value}<tool_call|>`.  It reuses Qwen35Chat's
  # OpenAI message parsing and JSON response normalization because those parts
  # are model-agnostic boundary adapters.
  module Gemma4Chat
    alias Message = Qwen35Chat::Message
    alias ToolCall = Qwen35Chat::ToolCall

    def self.parse_tools_json(json : String) : Array(JSON::Any)
      Qwen35Chat.parse_tools_json(json)
    end

    def self.messages_from_openai_json(json : String) : Array(Message)
      Qwen35Chat.messages_from_openai_json(json)
    end

    def self.render_user_prompt(prompt : String,
                                system : String? = nil,
                                tools : Array(JSON::Any) = [] of JSON::Any,
                                add_generation_prompt : Bool = true,
                                enable_thinking : Bool = false) : String
      messages = [] of Message
      messages << Message.new("system", system.not_nil!) if system && !system.empty?
      messages << Message.new("user", prompt)
      render(messages, tools, add_generation_prompt, enable_thinking)
    end

    def self.render(messages : Array(Message),
                    tools : Array(JSON::Any) = [] of JSON::Any,
                    add_generation_prompt : Bool = true,
                    enable_thinking : Bool = false) : String
      raise ArgumentError.new("Gemma4Chat.render requires at least one message") if messages.empty?

      String.build do |io|
        loop_messages = messages
        if enable_thinking || !tools.empty? || {"system", "developer"}.includes?(messages[0].role)
          io << "<|turn>system\n"
          io << "<|think|>\n" if enable_thinking
          if {"system", "developer"}.includes?(messages[0].role)
            io << messages[0].content.strip
            loop_messages = messages[1..] || [] of Message
          end
          tools.each do |tool|
            io << "<|tool>"
            io << format_function_declaration(tool)
            io << "<tool|>"
          end
          io << "<turn|>\n"
        end

        loop_messages.each do |message|
          emit_message(io, message)
        end
        if add_generation_prompt
          io << "<|turn>model\n"
          io << "<|channel>thought\n<channel|>" unless enable_thinking
        end
      end
    end

    def self.parse_tool_calls(text : String) : Array(ToolCall)
      calls = [] of ToolCall
      cursor = 0
      while start_idx = text.index("<|tool_call>call:", cursor)
        body_start = start_idx + "<|tool_call>call:".size
        close_idx = text.index("<tool_call|>", body_start)
        break unless close_idx

        body = text[body_start...close_idx]
        if call = parse_tool_call_body(body)
          calls << call
        end
        cursor = close_idx + "<tool_call|>".size
      end
      calls
    end

    def self.content_without_tool_calls(text : String) : String
      String.build do |io|
        cursor = 0
        while start_idx = text.index("<|tool_call>call:", cursor)
          io << text[cursor...start_idx]
          body_start = start_idx + "<|tool_call>call:".size
          close_idx = text.index("<tool_call|>", body_start)
          unless close_idx
            cursor = text.size
            break
          end
          cursor = close_idx + "<tool_call|>".size
        end
        io << text[cursor..]? if cursor < text.size
      end.strip
    end

    def self.tool_response_to_json(calls : Array(ToolCall), content : String? = nil, tools : Array(JSON::Any) = [] of JSON::Any) : String
      Qwen35Chat.tool_response_to_json(calls, content, tools)
    end

    def self.tool_response_to_openai_json(calls : Array(ToolCall), content : String? = nil, tools : Array(JSON::Any) = [] of JSON::Any) : String
      Qwen35Chat.tool_response_to_openai_json(calls, content, tools)
    end

    def self.format_function_declaration(tool : JSON::Any) : String
      obj = tool.as_h? || raise ArgumentError.new("Gemma tool must be an object")
      function = obj["function"]?.try(&.as_h?) || raise ArgumentError.new("Gemma tool missing function")
      name = function["name"]?.try(&.as_s?) || raise ArgumentError.new("Gemma tool function missing name")
      description = function["description"]?.try(&.as_s?) || ""
      params = function["parameters"]?.try(&.as_h?)

      String.build do |io|
        io << "declaration:" << name << "{description:"
        emit_gemma_string(io, description)
        if params
          io << ",parameters:{"
          if properties = params["properties"]?.try(&.as_h?)
            io << "properties:{"
            emit_properties(io, properties)
            io << "}"
          end
          if required = params["required"]?.try(&.as_a?)
            io << ",required:["
            required.compact_map(&.as_s?).each_with_index do |item, i|
              io << ',' if i > 0
              emit_gemma_string(io, item)
            end
            io << "]"
          end
          if type = params["type"]?.try(&.as_s?)
            io << ",type:"
            emit_gemma_string(io, type.upcase)
          end
          io << "}"
        end
        io << "}"
      end
    end

    def self.native_tool_finite_call_options(tools : Array(JSON::Any), max_options_per_tool : Int32 = 256) : Array(String)
      options = [] of String
      tools.each do |tool|
        function = tool.as_h?.try { |obj| obj["function"]?.try(&.as_h?) }
        next unless function
        name = function["name"]?.try(&.as_s?)
        next unless name && !name.empty?
        parameters = function["parameters"]?.try(&.as_h?)
        properties = parameters.try { |p| p["properties"]?.try(&.as_h?) }
        next unless properties
        required_names = parameters.try { |p| p["required"]?.try(&.as_a?).try { |items| items.compact_map(&.as_s?) } } || [] of String

        if required_names.empty?
          tool_options = [] of String
          properties.each do |parameter_name, raw_schema|
            schema = raw_schema.as_h?
            next unless schema
            finite_gemma_values(schema).each do |value_literal|
              break if tool_options.size >= max_options_per_tool
              tool_options << native_tool_call_literal(name, [{parameter_name, value_literal}])
            end
            break if tool_options.size >= max_options_per_tool
          end
          options.concat(tool_options)
          next
        end

        required_sets = [] of NamedTuple(name: String, values: Array(String))
        missing_or_open = false
        product = 1_i64
        required_names.each do |parameter_name|
          schema = properties[parameter_name]?.try(&.as_h?)
          unless schema
            missing_or_open = true
            break
          end

          values = finite_gemma_values(schema)
          if values.empty?
            missing_or_open = true
            break
          end

          product *= values.size
          if product > max_options_per_tool
            missing_or_open = true
            break
          end
          required_sets << {name: parameter_name, values: values}
        end
        next if missing_or_open

        tool_options = [] of String
        build_required_finite_call_options(tool_options, name, required_sets, 0, [] of Tuple(String, String), max_options_per_tool)
        options.concat(tool_options)
      end
      options
    end

    # Tokenized finite-literal corridor used by constrained Gemma tool calls.
    # It lets callers force singleton-token spans and reserve allowed-head work
    # for true branch points without depending on decoded token text.
    struct TokenOptionCorridor
      getter options : Array(Array(Int32))

      def initialize(options : Array(Array(Int32)))
        @options = options.map(&.dup)
      end

      def self.from_options(options : Array(Array(Int32))) : TokenOptionCorridor
        new(options)
      end

      def empty? : Bool
        @options.empty?
      end

      def complete? : Bool
        !@options.empty? && @options.all?(&.empty?)
      end

      def next_ids : Array(Int32)
        seen = Set(Int32).new
        ids = [] of Int32
        @options.each do |option|
          next if option.empty?

          id = option[0]
          next if seen.includes?(id)

          seen << id
          ids << id
        end
        ids
      end

      def advance(emitted_id : Int32) : TokenOptionCorridor
        advanced = [] of Array(Int32)
        @options.each do |option|
          next if option.empty? || option[0] != emitted_id

          advanced << option[1..]
        end
        TokenOptionCorridor.new(advanced)
      end

      def self.selected_literal_index?(full_options : Array(Array(Int32)), emitted_ids : Array(Int32)) : Int32?
        selected_idx = nil.as(Int32?)
        selected_size = -1
        full_options.each_with_index do |ids, idx|
          next unless ids.size <= emitted_ids.size
          next unless emitted_ids[0, ids.size] == ids
          next unless ids.size > selected_size

          selected_idx = idx
          selected_size = ids.size
        end
        selected_idx
      end
    end

    private def self.emit_message(io : IO, message : Message) : Nil
      role = case message.role
             when "assistant"
               "model"
             when "system", "developer"
               "system"
             else
               message.role
             end
      raise ArgumentError.new("unsupported Gemma4 chat role: #{message.role.inspect}") unless {"system", "user", "model", "tool"}.includes?(role)

      io << "<|turn>" << role << '\n'
      unless message.tool_calls.empty?
        message.tool_calls.each { |call| emit_tool_call(io, call) }
      end
      io << message.content.strip unless message.content.empty?
      io << "<turn|>\n"
    end

    private def self.emit_tool_call(io : IO, call : ToolCall) : Nil
      io << "<|tool_call>call:" << call.name << '{'
      call.arguments.keys.sort.each_with_index do |name, i|
        io << ',' if i > 0
        io << name << ':'
        emit_gemma_string(io, call.arguments[name])
      end
      io << "}<tool_call|>"
    end

    private def self.emit_properties(io : IO, properties : Hash(String, JSON::Any)) : Nil
      properties.keys.sort.each_with_index do |key, i|
        io << ',' if i > 0
        schema = properties[key].as_h? || next
        io << key << ":{"
        comma = false
        if description = schema["description"]?.try(&.as_s?)
          io << "description:"
          emit_gemma_string(io, description)
          comma = true
        end
        if enum_values = schema["enum"]?.try(&.as_a?)
          io << ',' if comma
          io << "enum:"
          emit_argument(io, JSON::Any.new(enum_values))
          comma = true
        end
        if type = schema["type"]?.try(&.as_s?)
          io << ',' if comma
          io << "type:"
          emit_gemma_string(io, type.upcase)
        end
        io << '}'
      end
    end

    private def self.emit_argument(io : IO, value : JSON::Any) : Nil
      if str = value.as_s?
        emit_gemma_string(io, str)
      elsif bool = value.as_bool?
        io << (bool ? "true" : "false")
      elsif int = value.as_i64?
        io << int
      elsif float = value.as_f?
        io << float
      elsif arr = value.as_a?
        io << '['
        arr.each_with_index do |item, i|
          io << ',' if i > 0
          emit_argument(io, item)
        end
        io << ']'
      elsif h = value.as_h?
        io << '{'
        h.keys.sort.each_with_index do |key, i|
          io << ',' if i > 0
          emit_gemma_string(io, key)
          io << ':'
          emit_argument(io, h[key])
        end
        io << '}'
      else
        io << "null"
      end
    end

    private def self.emit_gemma_string(io : IO, value : String) : Nil
      io << "<|\"|>" << value << "<|\"|>"
    end

    private def self.gemma_string_literal(value : String) : String
      String.build { |io| emit_gemma_string(io, value) }
    end

    private def self.finite_gemma_values(schema : Hash(String, JSON::Any)) : Array(String)
      enum_values = schema["enum"]?.try(&.as_a?)
      if enum_values
        return enum_values.compact_map do |value|
          if str = value.as_s?
            gemma_string_literal(str)
          elsif bool = value.as_bool?
            bool ? "true" : "false"
          elsif int = value.as_i64?
            int.to_s
          elsif float = value.as_f?
            float.to_s
          end
        end
      end

      type_name = schema["type"]?.try(&.as_s?).try(&.downcase)
      return ["true", "false"] if type_name == "boolean"

      if type_name == "integer"
        minimum = schema["minimum"]?.try(&.as_i64?)
        maximum = schema["maximum"]?.try(&.as_i64?)
        if minimum && maximum && maximum >= minimum && (maximum - minimum) < 256
          return (minimum..maximum).map(&.to_s)
        end
      end

      [] of String
    end

    private def self.native_tool_call_literal(name : String, args : Array(Tuple(String, String))) : String
      String.build do |io|
        io << "<|tool_call>call:" << name << '{'
        args.each_with_index do |arg, i|
          io << ',' if i > 0
          io << arg[0] << ':' << arg[1]
        end
        io << "}<tool_call|>"
      end
    end

    private def self.build_required_finite_call_options(output : Array(String),
                                                        function_name : String,
                                                        sets : Array(NamedTuple(name: String, values: Array(String))),
                                                        idx : Int32,
                                                        args : Array(Tuple(String, String)),
                                                        max_options : Int32) : Nil
      return if output.size >= max_options
      if idx >= sets.size
        output << native_tool_call_literal(function_name, args)
        return
      end

      set = sets[idx]
      set[:values].each do |value_literal|
        break if output.size >= max_options
        next_args = args.dup
        next_args << {set[:name], value_literal}
        build_required_finite_call_options(output, function_name, sets, idx + 1, next_args, max_options)
      end
    end

    private def self.parse_tool_call_body(body : String) : ToolCall?
      name_end = body.index('{')
      return nil unless name_end
      name = body[0...name_end].strip
      return nil if name.empty?
      args_end = body.rindex('}') || body.size
      args_body = body[(name_end + 1)...args_end]
      ToolCall.new(name, parse_arguments(args_body))
    end

    private def self.parse_arguments(text : String) : Hash(String, String)
      args = Hash(String, String).new
      split_top_level(text, ',').each do |part|
        if sep = find_top_level(part, ':')
          key = part[0...sep].strip
          value = decode_gemma_value(part[(sep + 1)..].strip)
          args[key] = value unless key.empty?
        end
      end
      args
    end

    private def self.decode_gemma_value(value : String) : String
      if value.starts_with?("<|\"|>") && value.ends_with?("<|\"|>") && value.size >= 10
        value[5...(value.size - 5)]
      else
        value
      end
    end

    private def self.split_top_level(text : String, separator : Char) : Array(String)
      parts = [] of String
      start = 0
      depth = 0
      in_gemma_string = false
      i = 0
      while i < text.size
        if gemma_string_marker_at?(text, i)
          in_gemma_string = !in_gemma_string
          i += 5
          next
        end

        unless in_gemma_string
          case text[i]
          when '{', '['
            depth += 1
          when '}', ']'
            depth -= 1 if depth > 0
          when separator
            if depth == 0
              parts << text[start...i]
              start = i + 1
            end
          end
        end
        i += 1
      end
      parts << text[start..]
      parts
    end

    private def self.find_top_level(text : String, target : Char) : Int32?
      depth = 0
      in_gemma_string = false
      i = 0
      while i < text.size
        if gemma_string_marker_at?(text, i)
          in_gemma_string = !in_gemma_string
          i += 5
          next
        end

        unless in_gemma_string
          case text[i]
          when '{', '['
            depth += 1
          when '}', ']'
            depth -= 1 if depth > 0
          when target
            return i if depth == 0
          end
        end
        i += 1
      end
      nil
    end

    private def self.gemma_string_marker_at?(text : String, offset : Int32) : Bool
      marker = "<|\"|>"
      offset + marker.size <= text.size && text[offset, marker.size] == marker
    end
  end
end
