require "./qwen_recurrent_layer_runner"
require "./qwen_full_attn_layer_runner"
require "./qwen_output_head_runner"

module ML::CUDA
  # Model-body scaffold for a CUDA Qwen layer slice.
  #
  # It owns already-constructed recurrent/full-attention layer runners plus
  # the output head, and centralizes the decode-state handoff loop. This is not
  # a full decoder yet: tokenizer/sampling and complete model construction stay
  # outside this object for now.
  class QwenMixedStackRunner
    alias LayerRunner = QwenRecurrentLayerRunner | QwenFullAttnLayerRunner

    getter layer_ids : Array(Int32)
    getter runners : Array(LayerRunner)
    getter head : QwenOutputHeadRunner
    getter final_gpu_all : Array(Float32)
    getter phase_lines : Array(String)

    def initialize(@layer_ids : Array(Int32),
                   @runners : Array(LayerRunner),
                   @head : QwenOutputHeadRunner,
                   @tokens : Int32,
                   @hidden : Int32,
                   @xs : Array(Float32))
      raise ArgumentError.new("layer/runner count mismatch") unless @layer_ids.size == @runners.size
      raise ArgumentError.new("tokens must be positive") unless @tokens > 0
      raise ArgumentError.new("xs size mismatch") unless @xs.size == @tokens * @hidden

      @final_gpu_all = Array(Float32).new(@tokens * @hidden, 0.0_f32)
      @phase_lines = [] of String
      @closed = false
    end

    def upload_weights(profile : Bool = false) : Float64
      t0 = Time.instant
      @runners.each(&.upload_weights)
      @head.upload_weights
      ML::CUDA.synchronize!("cuCtxSynchronize(mixed upload)")
      elapsed = (Time.instant - t0).total_milliseconds
      @phase_lines << "phase_upload_ms=#{elapsed.round(3)}" if profile
      elapsed
    end

    def upload_first_sequence_input(xs : Array(Float32)) : Nil
      raise ArgumentError.new("xs size mismatch") unless xs.size == @tokens * @hidden

      @xs = xs
      case first = @runners.first
      in QwenRecurrentLayerRunner
        first.upload_sequence_input(xs)
      in QwenFullAttnLayerRunner
        first.upload_sequence_input(xs)
      end
    end

    def update_decode_position(start_pos : Int32, cos_table : Array(Float32), sin_table : Array(Float32)) : Nil
      @runners.each do |runner|
        case runner
        in QwenFullAttnLayerRunner
          runner.update_decode_position(start_pos, cos_table, sin_table)
        in QwenRecurrentLayerRunner
          # Recurrent DeltaNet layers keep position only through their resident state.
        end
      end
    end

    def run_sequence(profile_phases : Bool = false,
                     debug_readback : Bool = true,
                     reset_sequence : Bool = true,
                     sync_end : Bool = true,
                     read_head_outputs : Bool = true) : Float64
      @phase_lines.clear
      previous_output = 0_u64
      t_all = Time.instant

      @runners.each_with_index do |runner, idx|
        if idx == 0
          case runner
          in QwenRecurrentLayerRunner
            runner.replace_sequence_input(@xs)
          in QwenFullAttnLayerRunner
            # A first full-attention layer owns its initial host-backed input.
          end
        else
          runner.use_device_sequence_input(previous_output)
        end

        if reset_sequence
          t_reset = Time.instant
          runner.reset_sequence
          if profile_phases
            ML::CUDA.synchronize!("cuCtxSynchronize(mixed layer #{idx} reset)")
            @phase_lines << "phase_layer#{@layer_ids[idx]}_reset_ms=#{((Time.instant - t_reset).total_milliseconds).round(3)}"
          end
        elsif profile_phases
          @phase_lines << "phase_layer#{@layer_ids[idx]}_reset_ms=skipped"
        end
        t0 = Time.instant
        if profile_phases
          case runner
          in QwenRecurrentLayerRunner
            runner.run_sequence_profiled(@phase_lines, "phase_layer#{@layer_ids[idx]}")
          in QwenFullAttnLayerRunner
            runner.run_sequence_profiled(@phase_lines, "phase_layer#{@layer_ids[idx]}")
          end
          @phase_lines << "phase_layer#{@layer_ids[idx]}_ms=#{((Time.instant - t0).total_milliseconds).round(3)}"
        else
          runner.run_sequence
        end
        previous_output = runner.output_device_ptr
      end

      @head.use_device_sequence_input(previous_output)
      if reset_sequence
        t_head_reset = Time.instant
        @head.reset_sequence
        if profile_phases
          ML::CUDA.synchronize!("cuCtxSynchronize(mixed head reset)")
          @phase_lines << "phase_head_reset_ms=#{((Time.instant - t_head_reset).total_milliseconds).round(3)}"
        end
      elsif profile_phases
        @phase_lines << "phase_head_reset_ms=skipped"
      end
      t_head = Time.instant
      if profile_phases
        @head.run_sequence_profiled(@phase_lines)
        @phase_lines << "phase_head_ms=#{((Time.instant - t_head).total_milliseconds).round(3)}"
      else
        @head.run_sequence
        ML::CUDA.synchronize!("cuCtxSynchronize(mixed stack)") if sync_end
      end

      t_read = Time.instant
      @runners.each(&.read_outputs) if debug_readback
      @head.read_outputs if read_head_outputs
      @phase_lines << "phase_readback_ms=#{((Time.instant - t_read).total_milliseconds).round(3)}" if profile_phases

      if debug_readback
        case last = @runners.last
        in QwenRecurrentLayerRunner
          @final_gpu_all = last.final_gpu_all.dup
        in QwenFullAttnLayerRunner
          @final_gpu_all = last.final_gpu_all.dup
        end
      end

      elapsed = (Time.instant - t_all).total_milliseconds
      @phase_lines << "phase_total_ms=#{elapsed.round(3)}" if profile_phases
      elapsed
    end

    def read_head_outputs : Nil
      @head.read_outputs
    end

    def close : Nil
      return if @closed

      @head.close
      @runners.reverse_each(&.close)
      @closed = true
    end
  end
end
