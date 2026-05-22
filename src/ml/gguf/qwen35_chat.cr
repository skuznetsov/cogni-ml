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
      String.build do |io|
        io << '['
        calls.each_with_index do |call, i|
          io << ',' if i > 0
          io << {"name" => call.name, "arguments" => call.arguments}.to_json
        end
        io << ']'
      end
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
  end
end
