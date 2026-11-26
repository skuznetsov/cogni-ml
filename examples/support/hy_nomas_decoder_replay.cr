require "./hy_nomas_generation"

# Diagnostic-only same-owner decoder repetition. The helper never waits or
# reallocates; callers must first finish the pass on its sole terminal wait.
module HyNomasDecoderReplay
  struct RestartState
    getter chain_present : Bool
    getter terminal_completed : Bool
    getter wait_count : Int32
    getter released : Bool
    getter failed : Bool
    getter schedule_complete : Bool
    getter finalized : Bool

    def initialize(@chain_present : Bool, @terminal_completed : Bool,
                   @wait_count : Int32, @released : Bool, @failed : Bool,
                   @schedule_complete : Bool, @finalized : Bool)
    end
  end

  module RestartGuard
    extend self

    def validate!(state : RestartState, owner : String,
                  *, require_finalized : Bool = false) : Nil
      raise ArgumentError.new("#{owner} has been released") if state.released
      raise ArgumentError.new("#{owner} cannot restart after an encode failure") if state.failed
      raise ArgumentError.new("#{owner} restart requires its prior command chain") unless state.chain_present
      raise ArgumentError.new("#{owner} restart requires terminal chain completion") unless state.terminal_completed
      unless state.wait_count == 1
        raise ArgumentError.new("#{owner} restart requires exactly one terminal wait")
      end
      unless state.schedule_complete
        raise ArgumentError.new("#{owner} restart requires the complete configured window schedule")
      end
      if require_finalized && !state.finalized
        raise ArgumentError.new("#{owner} restart requires finalized output")
      end
    end
  end
end

class ML::Metal::MPSCommandChain
  # A drained chain can still have a failed GPU completion status. Such a pass
  # permits cleanup, but must never authorize a diagnostic replay.
  def diagnostic_successfully_completed? : Bool
    @drained && @status == 0
  end
end

module ML::Video::PixelUMMGeneration
  class StemPlan
    # Preflight is public only so Decoder can validate every owner before any
    # cursor is changed. It requires the exact completed chain used by Decoder.
    def diagnostic_validate_restart!(expected_chain : ML::Metal::MPSCommandChain) : Nil
      chain = @chain
      unless chain && chain.same?(expected_chain)
        raise ArgumentError.new("PixelUMM stem restart chain does not match decoder chain")
      end
      HyNomasDecoderReplay::RestartGuard.validate!(
        HyNomasDecoderReplay::RestartState.new(
          !chain.nil?, chain.try(&.diagnostic_successfully_completed?) || false, chain.try(&.wait_count) || 0,
          @released, @failed, @next_window_index == @config.window_count, true),
        "PixelUMM stem")
    end

    def diagnostic_restart! : Nil
      chain = @chain
      diagnostic_validate_restart!(chain || raise ArgumentError.new("PixelUMM stem restart requires its prior command chain"))
      @chain = nil
      @next_window_index = 0
      # Keep rounded GPU weights and all allocated buffers resident.
    end
  end

  class AssemblyPlan
    def diagnostic_validate_restart!(expected_chain : ML::Metal::MPSCommandChain) : Nil
      chain = @chain
      unless chain && chain.same?(expected_chain)
        raise ArgumentError.new("PixelUMM assembly restart chain does not match decoder chain")
      end
      HyNomasDecoderReplay::RestartGuard.validate!(
        HyNomasDecoderReplay::RestartState.new(
          !chain.nil?, chain.try(&.diagnostic_successfully_completed?) || false, chain.try(&.wait_count) || 0,
          @released, @failed, @merged_windows == @config.window_count, @finalized),
        "PixelUMM assembly", require_finalized: true)
    end

    def diagnostic_restart! : Nil
      chain = @chain
      diagnostic_validate_restart!(chain || raise ArgumentError.new("PixelUMM assembly restart requires its prior command chain"))
      @chain = nil
      @merged_windows = 0
      @finalized = false
      # Keep the once-encoded blend weights and output buffer resident.
    end
  end
end

module HyNomasGeneration
  class Decoder
    def diagnostic_owner_ids : Array(UInt64)
      [object_id, @stem.object_id, @assembly.object_id, @workspace.object_id,
       @head.object_id, output.object_id] + @blocks.map(&.object_id)
    end

    # Reuse the same decoder owners for a warmup followed by measured decodes.
    # Blocks, head, and workspace already admit a new chain after one terminal
    # wait; stem and assembly cursors need this guarded reset. The caller owns
    # chain construction and the sole wait, so this method performs no GPU work.
    def diagnostic_restart! : Nil
      chain = @chain || raise ArgumentError.new("decoder restart requires its prior command chain")
      schedule_complete = @window_starts == @stem.config.window_starts &&
                          @stem.next_window_index == @stem.config.window_count &&
                          @assembly.merged_windows == @stem.config.window_count
      HyNomasDecoderReplay::RestartGuard.validate!(
        HyNomasDecoderReplay::RestartState.new(
          true, chain.diagnostic_successfully_completed?, chain.wait_count, @released, false,
          schedule_complete, @assembly.finalized?),
        "decoder", require_finalized: true)

      # Preflight both single-use owners before mutating either cursor.
      @stem.diagnostic_validate_restart!(chain)
      @assembly.diagnostic_validate_restart!(chain)
      @stem.diagnostic_restart!
      @assembly.diagnostic_restart!
      @chain = nil
    end
  end
end
