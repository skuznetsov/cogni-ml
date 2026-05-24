require "json"

module ML::GGUF
  # Minimal Qwen3.5/Qwen3.6 chat-template support for text and function calls.
  # This intentionally implements the tool-call subset embedded in the GGUF
  # tokenizer.chat_template instead of a generic Jinja interpreter.
  module Qwen35Chat
    TOOL_SYSTEM_PREFIX = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    TOOL_SYSTEM_SUFFIX = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

    record Message,
      role : String,
      content : String,
      tool_calls : Array(ToolCall) = [] of ToolCall

    record ToolCall,
      name : String,
      arguments : Hash(String, String)

    def self.render(messages : Array(Message),
                    tools : Array(JSON::Any) = [] of JSON::Any,
                    add_generation_prompt : Bool = true) : String
      raise ArgumentError.new("Qwen35Chat.render requires at least one message") if messages.empty?

      String.build do |io|
        start_index = 0
        if tools.empty?
          if messages[0].role == "system"
            emit_message(io, messages[0])
            start_index = 1
          end
        else
          io << "<|im_start|>system\n"
          io << TOOL_SYSTEM_PREFIX
          tools.each do |tool|
            io << '\n'
            tool.to_json(io)
          end
          io << TOOL_SYSTEM_SUFFIX
          if messages[0].role == "system"
            content = messages[0].content.strip
            io << "\n\n" << content unless content.empty?
            start_index = 1
          end
          io << "<|im_end|>\n"
        end

        messages[start_index..].each do |message|
          emit_message(io, message)
        end
        io << "<|im_start|>assistant\n" if add_generation_prompt
      end
    end

    def self.render_user_prompt(prompt : String,
                                system : String? = nil,
                                tools : Array(JSON::Any) = [] of JSON::Any,
                                add_generation_prompt : Bool = true) : String
      messages = [] of Message
      messages << Message.new("system", system.not_nil!) if system && !system.empty?
      messages << Message.new("user", prompt)
      render(messages, tools, add_generation_prompt)
    end

    def self.messages_from_openai_json(json : String) : Array(Message)
      any = JSON.parse(json)
      arr = any.as_a?
      raise ArgumentError.new("Qwen messages JSON must be an array") unless arr

      arr.map do |message|
        obj = message.as_h?
        raise ArgumentError.new("Qwen message must be an object") unless obj
        role = obj["role"]?.try(&.as_s?) || raise ArgumentError.new("Qwen message missing role")
        content = openai_message_content(obj["content"]?)
        tool_calls = parse_openai_tool_calls(obj["tool_calls"]?)
        Message.new(role, content, tool_calls)
      end
    end

    def self.parse_tools_json(json : String) : Array(JSON::Any)
      any = JSON.parse(json)
      arr = any.as_a?
      raise ArgumentError.new("Qwen tools JSON must be an array") unless arr
      arr
    end

    def self.parse_tool_calls(text : String) : Array(ToolCall)
      calls = [] of ToolCall
      cursor = 0
      while start_idx = text.index("<tool_call>", cursor)
        body_start = start_idx + "<tool_call>".size
        close_idx = text.index("</tool_call>", body_start)
        break unless close_idx

        body = text[body_start...close_idx]
        if call = parse_tool_call_body(body)
          calls << call
        end
        cursor = close_idx + "</tool_call>".size
      end
      calls
    end

    def self.tool_calls_to_json(calls : Array(ToolCall)) : String
      tool_calls_to_json(calls, [] of JSON::Any)
    end

    def self.tool_calls_to_json(calls : Array(ToolCall), tools : Array(JSON::Any)) : String
      schemas = tool_argument_schemas(tools)
      String.build do |io|
        io << '['
        calls.each_with_index do |call, i|
          io << ',' if i > 0
          io << JSON::Any.new({
            "name"      => JSON::Any.new(call.name),
            "arguments" => JSON::Any.new(argument_json_hash(call, schemas[call.name]?)),
          }).to_json
        end
        io << ']'
      end
    end

    def self.tool_response_to_json(calls : Array(ToolCall), content : String? = nil, tools : Array(JSON::Any) = [] of JSON::Any) : String
      schemas = tool_argument_schemas(tools)
      payload = Hash(String, JSON::Any).new
      payload["content"] = content && !content.empty? ? JSON::Any.new(content) : JSON::Any.new(nil)
      payload["tool_calls"] = JSON::Any.new(calls.map do |call|
        JSON::Any.new({
          "name"      => JSON::Any.new(call.name),
          "arguments" => JSON::Any.new(argument_json_hash(call, schemas[call.name]?)),
        })
      end)
      payload.to_json
    end

    def self.tool_response_to_openai_json(calls : Array(ToolCall), content : String? = nil, tools : Array(JSON::Any) = [] of JSON::Any) : String
      schemas = tool_argument_schemas(tools)
      payload = Hash(String, JSON::Any).new
      payload["content"] = content && !content.empty? ? JSON::Any.new(content) : JSON::Any.new(nil)
      payload["tool_calls"] = JSON::Any.new(calls.map_with_index do |call, i|
        JSON::Any.new({
          "id"       => JSON::Any.new("call_#{i}"),
          "type"     => JSON::Any.new("function"),
          "function" => JSON::Any.new({
            "name"      => JSON::Any.new(call.name),
            "arguments" => JSON::Any.new(arguments_to_json(call, schemas[call.name]?)),
          }),
        })
      end)
      payload.to_json
    end

    def self.content_without_tool_calls(text : String) : String
      String.build do |io|
        cursor = 0
        while start_idx = text.index("<tool_call>", cursor)
          io << text[cursor...start_idx]
          body_start = start_idx + "<tool_call>".size
          close_idx = text.index("</tool_call>", body_start)
          unless close_idx
            cursor = text.size
            break
          end
          cursor = close_idx + "</tool_call>".size
        end
        io << text[cursor..]? if cursor < text.size
      end.strip
    end

    private def self.emit_message(io : IO, message : Message) : Nil
      role = message.role
      raise ArgumentError.new("unsupported Qwen chat role: #{role.inspect}") unless {"system", "user", "assistant", "tool"}.includes?(role)

      io << "<|im_start|>" << role << '\n'
      io << message.content
      unless message.tool_calls.empty?
        io << "\n\n" unless message.content.strip.empty?
        message.tool_calls.each_with_index do |tool_call, i|
          io << '\n' if i > 0
          emit_tool_call(io, tool_call)
        end
      end
      io << "<|im_end|>\n"
    end

    private def self.emit_tool_call(io : IO, call : ToolCall) : Nil
      io << "<tool_call>\n<function=" << call.name << ">\n"
      call.arguments.each do |name, value|
        io << "<parameter=" << name << ">\n"
        io << value
        io << "\n</parameter>\n"
      end
      io << "</function>\n</tool_call>"
    end

    private def self.parse_tool_call_body(body : String) : ToolCall?
      fn_open = body.index("<function=")
      return nil unless fn_open
      name_start = fn_open + "<function=".size
      name_end = body.index('>', name_start)
      return nil unless name_end
      name = body[name_start...name_end].strip
      return nil if name.empty?

      fn_close = body.rindex("</function>") || body.size
      fn_body = body[(name_end + 1)...fn_close]
      args = Hash(String, String).new
      cursor = 0
      while param_open = fn_body.index("<parameter=", cursor)
        pname_start = param_open + "<parameter=".size
        pname_end = fn_body.index('>', pname_start)
        break unless pname_end
        pname = fn_body[pname_start...pname_end].strip
        value_start = pname_end + 1
        param_close = fn_body.index("</parameter>", value_start)
        break unless param_close
        value = fn_body[value_start...param_close]
        value = value[1..] if value.starts_with?('\n')
        value = value[...-1] if value.ends_with?('\n')
        args[pname] = value unless pname.empty?
        cursor = param_close + "</parameter>".size
      end

      ToolCall.new(name, args)
    end

    private def self.openai_message_content(raw : JSON::Any?) : String
      return "" unless raw
      return "" if raw.raw.nil?
      if str = raw.as_s?
        str
      elsif arr = raw.as_a?
        arr.compact_map do |part|
          obj = part.as_h?
          next unless obj
          text = obj["text"]?.try(&.as_s?)
          text || obj["content"]?.try(&.as_s?)
        end.join("\n")
      else
        raw.to_json
      end
    end

    private def self.parse_openai_tool_calls(raw : JSON::Any?) : Array(ToolCall)
      return [] of ToolCall unless raw
      arr = raw.as_a?
      return [] of ToolCall unless arr

      calls = [] of ToolCall
      arr.each do |item|
        obj = item.as_h?
        next unless obj
        fn = obj["function"]?.try(&.as_h?)
        name = fn.try(&.["name"]?.try(&.as_s?)) || obj["name"]?.try(&.as_s?)
        next unless name
        args_raw = fn.try(&.["arguments"]?) || obj["arguments"]?
        calls << ToolCall.new(name, openai_arguments_to_qwen(args_raw))
      end
      calls
    end

    private def self.openai_arguments_to_qwen(raw : JSON::Any?) : Hash(String, String)
      args = Hash(String, String).new
      return args unless raw

      obj = if str = raw.as_s?
              JSON.parse(str).as_h? rescue nil
            else
              raw.as_h?
            end
      return args unless obj

      obj.each do |key, value|
        args[key] = value.as_s? || value.to_json
      end
      args
    end

    private def self.tool_argument_schemas(tools : Array(JSON::Any)) : Hash(String, Hash(String, Hash(String, JSON::Any)))
      schemas = {} of String => Hash(String, Hash(String, JSON::Any))
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

        by_parameter = {} of String => Hash(String, JSON::Any)
        properties.each do |parameter_name, raw_schema|
          schema = raw_schema.as_h?
          by_parameter[parameter_name] = schema if schema
        end
        schemas[name] = by_parameter unless by_parameter.empty?
      end
      schemas
    end

    private def self.argument_json_hash(call : ToolCall, schemas : Hash(String, Hash(String, JSON::Any))? = nil) : Hash(String, JSON::Any)
      args = Hash(String, JSON::Any).new
      call.arguments.each do |key, value|
        args[key] = argument_value_to_json_any(value, schemas.try(&.[key]?))
      end
      args
    end

    private def self.arguments_to_json(call : ToolCall, schemas : Hash(String, Hash(String, JSON::Any))? = nil) : String
      JSON::Any.new(argument_json_hash(call, schemas)).to_json
    end

    private def self.argument_value_to_json_any(value : String, schema : Hash(String, JSON::Any)? = nil) : JSON::Any
      if schema
        typed = schema_argument_value_to_json_any(value, schema)
        return typed if typed
      end

      stripped = value.strip
      unless stripped.empty?
        begin
          parsed = JSON.parse(stripped)
          return parsed unless parsed.raw.is_a?(String)
        rescue JSON::ParseException
        end
      end
      JSON::Any.new(value)
    end

    private def self.schema_argument_value_to_json_any(value : String, schema : Hash(String, JSON::Any)) : JSON::Any?
      stripped = value.strip
      type_name = schema["type"]?.try(&.as_s?)

      case type_name
      when "string"
        JSON::Any.new(value)
      when "boolean"
        return JSON::Any.new(true) if stripped == "true"
        return JSON::Any.new(false) if stripped == "false"
        nil
      when "integer"
        return nil unless stripped.matches?(/\A-?\d+\z/)
        JSON::Any.new(stripped.to_i64)
      when "number"
        parsed = JSON.parse(stripped) rescue nil
        return parsed if parsed && (parsed.raw.is_a?(Int64) || parsed.raw.is_a?(Float64))
        nil
      else
        nil
      end
    end
  end
end
