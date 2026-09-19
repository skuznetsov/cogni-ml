module ML::GGUF
  # Diagnostic-only unwind after a successful, cleaned-up shared command.
  # The production caller is compiled only with -Dqwen_first_command_probe.
  # Partially updated inference state must be destroyed, never resumed.
  module QwenFirstCommandProbe
    class Completed < Exception
    end

    def self.after_flush!(boundary : Tuple(Bool, Int32, Int32, Int32, Int32, Int32, Int32)) : NoReturn
      unless boundary == {true, 0, 2048, 0, 7, 1, 0}
        raise ArgumentError.new("first-command diagnostic boundary mismatch: #{boundary}")
      end
      raise Completed.new("first shared command completed; prefix intentionally incomplete")
    end
  end
end
