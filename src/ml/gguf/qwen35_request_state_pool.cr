require "./qwen35_cpu"

module ML::GGUF
  # Small resident-owner primitive for request-scoped Qwen state reuse.
  #
  # The pool never hands out the same state twice concurrently. Returned states
  # are reset for a fresh start_pos=0 request before they become available again.
  class Qwen35RequestStatePool
    getter max_seq : Int32
    getter capacity : Int32

    @available : Array(Qwen35CPU::State)
    @checked_out : Array(Qwen35CPU::State)
    @mutex : Mutex

    def initialize(@hparams : Qwen35Hparams,
                   @max_seq : Int32,
                   capacity : Int32 = 1,
                   @prepare_metal : Bool = Qwen35Metal.available?)
      raise ArgumentError.new("max_seq must be positive") unless @max_seq > 0
      raise ArgumentError.new("capacity must be positive") unless capacity > 0

      @capacity = capacity
      @available = [] of Qwen35CPU::State
      @checked_out = [] of Qwen35CPU::State
      @mutex = Mutex.new
    end

    def available_count : Int32
      @mutex.synchronize { @available.size }
    end

    def checked_out_count : Int32
      @mutex.synchronize { @checked_out.size }
    end

    def checkout : Qwen35CPU::State
      if state = take_available
        return state
      end

      state = Qwen35CPU::State.new(@hparams, max_seq: @max_seq)
      Qwen35CPU.prepare_state_metal!(state, @hparams) if @prepare_metal
      @mutex.synchronize { @checked_out << state }
      state
    end

    def release(state : Qwen35CPU::State) : Nil
      keep = false
      @mutex.synchronize do
        index = @checked_out.index { |candidate| candidate.same?(state) }
        raise ArgumentError.new("attempted to release a state not owned by this pool") unless index

        @checked_out.delete_at(index)
        keep = @available.size < @capacity
      end

      Qwen35CPU.reset_prepared_request_state!(state)

      return unless keep

      @mutex.synchronize do
        @available << state if @available.size < @capacity
      end
    end

    def with_state(& : Qwen35CPU::State -> T) : T forall T
      state = checkout
      begin
        yield state
      ensure
        release(state)
      end
    end

    private def take_available : Qwen35CPU::State?
      @mutex.synchronize do
        if state = @available.pop?
          @checked_out << state
          state
        end
      end
    end
  end
end
