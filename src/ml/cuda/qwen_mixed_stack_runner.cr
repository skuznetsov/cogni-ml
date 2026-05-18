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

    class RecurrentStateSnapshot
      getter buffers : Array(DeviceBuffer)

      def initialize(@buffers : Array(DeviceBuffer))
        @closed = false
      end

      def close : Nil
        return if @closed

        @buffers.each(&.close)
        @closed = true
      end
    end

    class DecodeStateSnapshot
      getter buffers : Array(DeviceBuffer)
      getter include_kv : Bool

      def initialize(@buffers : Array(DeviceBuffer), @include_kv : Bool)
        @closed = false
      end

      def close : Nil
        return if @closed

        @buffers.each(&.close)
        @closed = true
      end
    end

    class HostDecodeStateSnapshot
      getter buffers : Array(Bytes)
      getter include_kv : Bool
      getter kv_tokens : Int32?
      getter bytesize_total : UInt64

      def initialize(@buffers : Array(Bytes), @include_kv : Bool, @kv_tokens : Int32? = nil)
        @bytesize_total = @buffers.sum(0_u64) { |buffer| buffer.size.to_u64 }
      end
    end

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

    def update_decode_position(start_pos : Int32) : Nil
      @runners.each do |runner|
        case runner
        in QwenFullAttnLayerRunner
          runner.update_decode_position(start_pos)
        in QwenRecurrentLayerRunner
          # Recurrent DeltaNet layers keep position only through their resident state.
        end
      end
    end

    def increment_decode_position : Nil
      @runners.each do |runner|
        case runner
        in QwenFullAttnLayerRunner
          runner.increment_decode_position
        in QwenRecurrentLayerRunner
          # Recurrent DeltaNet layers keep position only through their resident state.
        end
      end
    end

    def first_sequence_input_device_ptr : DevicePtr
      case first = @runners.first
      in QwenRecurrentLayerRunner
        first.sequence_input_device_ptr
      in QwenFullAttnLayerRunner
        first.sequence_input_device_ptr
      end
    end

    def top1_ids_device_ptr : DevicePtr
      @head.top1_ids_device_ptr
    end

    def active_tokens=(count : Int32) : Int32
      raise ArgumentError.new("active tokens must be positive") unless count > 0
      raise ArgumentError.new("active tokens must be <= stack tokens") unless count <= @tokens

      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          runner.active_tokens = count
        in QwenFullAttnLayerRunner
          runner.active_tokens = count
        end
      end
      @head.active_tokens = count
      count
    end

    def reset_active_tokens : Nil
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          runner.reset_active_tokens
        in QwenFullAttnLayerRunner
          runner.reset_active_tokens
        end
      end
      @head.reset_active_tokens
    end

    def set_recurrent_ffn_raw_q8(enabled : Bool) : Nil
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          runner.ffn_raw_q8_enabled = enabled
        in QwenFullAttnLayerRunner
          # Full-attention layers do not use the recurrent FFN raw-Q8 path.
        end
      end
    end

    def set_recurrent_ffn_skip(enabled : Bool) : Nil
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          runner.ffn_skip_enabled = enabled
        in QwenFullAttnLayerRunner
          # Full-attention layers keep their exact FFN path in this probe.
        end
      end
    end

    def set_recurrent_ffn_skip_layers(skip_layers : Array(Int32)) : Nil
      skip_set = skip_layers.to_set
      @layer_ids.zip(@runners).each do |layer_id, runner|
        case runner
        in QwenRecurrentLayerRunner
          runner.ffn_skip_enabled = skip_set.includes?(layer_id)
        in QwenFullAttnLayerRunner
          # Full-attention layers keep their exact FFN path in this probe.
        end
      end
    end

    def set_recurrent_ffn_pca_updown_zero(rank : Int32, layer_ids : Array(Int32)? = nil) : Nil
      selected = layer_ids.try(&.to_set)
      @layer_ids.zip(@runners).each do |layer_id, runner|
        case runner
        in QwenRecurrentLayerRunner
          if selected.nil? || selected.not_nil!.includes?(layer_id)
            runner.set_zero_ffn_pca_updown_adapter(rank)
          else
            runner.clear_ffn_pca_updown_adapter
          end
        in QwenFullAttnLayerRunner
          # Full-attention layers keep their exact FFN path in this probe.
        end
      end
    end

    def set_recurrent_ffn_pca_updown_enabled(enabled : Bool, layer_ids : Array(Int32)? = nil) : Nil
      selected = layer_ids.try(&.to_set)
      @layer_ids.zip(@runners).each do |layer_id, runner|
        case runner
        in QwenRecurrentLayerRunner
          if selected.nil? || selected.not_nil!.includes?(layer_id)
            runner.ffn_pca_updown_enabled = enabled
          end
        in QwenFullAttnLayerRunner
          # Full-attention layers keep their exact FFN path in this probe.
        end
      end
    end

    def snapshot_recurrent_states : RecurrentStateSnapshot
      buffers = [] of DeviceBuffer
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          conv = DeviceBuffer.new(runner.conv_state_bytesize)
          ssm = DeviceBuffer.new(runner.ssm_state_bytesize)
          ML::CUDA.copy_dtod!(conv.ptr, runner.conv_state_device_ptr, conv.bytesize, "snapshot conv_state")
          ML::CUDA.copy_dtod!(ssm.ptr, runner.ssm_state_device_ptr, ssm.bytesize, "snapshot ssm_state")
          buffers << conv
          buffers << ssm
        in QwenFullAttnLayerRunner
          # Full-attention KV for the current position is overwritten by the
          # verifier pass; prior positions are not mutated by a proposal pass.
        end
      end
      RecurrentStateSnapshot.new(buffers)
    end

    def restore_recurrent_states(snapshot : RecurrentStateSnapshot) : Nil
      idx = 0
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          conv = snapshot.buffers[idx]
          ssm = snapshot.buffers[idx + 1]
          raise "snapshot conv_state size mismatch" unless conv.bytesize == runner.conv_state_bytesize
          raise "snapshot ssm_state size mismatch" unless ssm.bytesize == runner.ssm_state_bytesize
          ML::CUDA.copy_dtod!(runner.conv_state_device_ptr, conv.ptr, conv.bytesize, "restore conv_state")
          ML::CUDA.copy_dtod!(runner.ssm_state_device_ptr, ssm.ptr, ssm.bytesize, "restore ssm_state")
          idx += 2
        in QwenFullAttnLayerRunner
        end
      end
      raise "unused recurrent snapshot buffers" unless idx == snapshot.buffers.size
    end

    def snapshot_decode_state(include_kv : Bool = true) : DecodeStateSnapshot
      buffers = [] of DeviceBuffer
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          snapshot_device_buffer!(buffers, runner.conv_state_device_ptr, runner.conv_state_bytesize, "snapshot conv_state")
          snapshot_device_buffer!(buffers, runner.ssm_state_device_ptr, runner.ssm_state_bytesize, "snapshot ssm_state")
        in QwenFullAttnLayerRunner
          if include_kv
            snapshot_device_buffer!(buffers, runner.k_cache_device_ptr, runner.kv_cache_bytesize, "snapshot k_cache")
            snapshot_device_buffer!(buffers, runner.v_cache_device_ptr, runner.kv_cache_bytesize, "snapshot v_cache")
          end
        end
      end
      DecodeStateSnapshot.new(buffers, include_kv)
    end

    def snapshot_decode_state_host(include_kv : Bool = true, kv_tokens : Int32? = nil) : HostDecodeStateSnapshot
      buffers = [] of Bytes
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          snapshot_host_buffer!(buffers, runner.conv_state_device_ptr, runner.conv_state_bytesize, "snapshot host conv_state")
          snapshot_host_buffer!(buffers, runner.ssm_state_device_ptr, runner.ssm_state_bytesize, "snapshot host ssm_state")
        in QwenFullAttnLayerRunner
          if include_kv
            kv_bytesize = kv_tokens ? runner.kv_cache_bytesize_for_tokens(kv_tokens.not_nil!) : runner.kv_cache_bytesize
            snapshot_host_buffer!(buffers, runner.k_cache_device_ptr, kv_bytesize, "snapshot host k_cache")
            snapshot_host_buffer!(buffers, runner.v_cache_device_ptr, kv_bytesize, "snapshot host v_cache")
          end
        end
      end
      HostDecodeStateSnapshot.new(buffers, include_kv, kv_tokens)
    end

    def restore_decode_state(snapshot : DecodeStateSnapshot) : Nil
      idx = 0
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          conv = snapshot.buffers[idx]
          ssm = snapshot.buffers[idx + 1]
          raise "snapshot conv_state size mismatch" unless conv.bytesize == runner.conv_state_bytesize
          raise "snapshot ssm_state size mismatch" unless ssm.bytesize == runner.ssm_state_bytesize
          ML::CUDA.copy_dtod!(runner.conv_state_device_ptr, conv.ptr, conv.bytesize, "restore conv_state")
          ML::CUDA.copy_dtod!(runner.ssm_state_device_ptr, ssm.ptr, ssm.bytesize, "restore ssm_state")
          idx += 2
        in QwenFullAttnLayerRunner
          if snapshot.include_kv
            k_cache = snapshot.buffers[idx]
            v_cache = snapshot.buffers[idx + 1]
            raise "snapshot k_cache size mismatch" unless k_cache.bytesize == runner.kv_cache_bytesize
            raise "snapshot v_cache size mismatch" unless v_cache.bytesize == runner.kv_cache_bytesize
            ML::CUDA.copy_dtod!(runner.k_cache_device_ptr, k_cache.ptr, k_cache.bytesize, "restore k_cache")
            ML::CUDA.copy_dtod!(runner.v_cache_device_ptr, v_cache.ptr, v_cache.bytesize, "restore v_cache")
            idx += 2
          end
        end
      end
      raise "unused decode snapshot buffers" unless idx == snapshot.buffers.size
    end

    def restore_decode_state(snapshot : HostDecodeStateSnapshot) : Nil
      idx = 0
      @runners.each do |runner|
        case runner
        in QwenRecurrentLayerRunner
          conv = snapshot.buffers[idx]
          ssm = snapshot.buffers[idx + 1]
          raise "snapshot host conv_state size mismatch" unless conv.size.to_u64 == runner.conv_state_bytesize
          raise "snapshot host ssm_state size mismatch" unless ssm.size.to_u64 == runner.ssm_state_bytesize
          ML::CUDA.copy_htod!(runner.conv_state_device_ptr, conv.to_unsafe.as(Void*), runner.conv_state_bytesize, "restore host conv_state")
          ML::CUDA.copy_htod!(runner.ssm_state_device_ptr, ssm.to_unsafe.as(Void*), runner.ssm_state_bytesize, "restore host ssm_state")
          idx += 2
        in QwenFullAttnLayerRunner
          if snapshot.include_kv
            k_cache = snapshot.buffers[idx]
            v_cache = snapshot.buffers[idx + 1]
            kv_bytesize = snapshot.kv_tokens ? runner.kv_cache_bytesize_for_tokens(snapshot.kv_tokens.not_nil!) : runner.kv_cache_bytesize
            raise "snapshot host k_cache size mismatch" unless k_cache.size.to_u64 == kv_bytesize
            raise "snapshot host v_cache size mismatch" unless v_cache.size.to_u64 == kv_bytesize
            ML::CUDA.copy_htod!(runner.k_cache_device_ptr, k_cache.to_unsafe.as(Void*), kv_bytesize, "restore host k_cache")
            ML::CUDA.copy_htod!(runner.v_cache_device_ptr, v_cache.to_unsafe.as(Void*), kv_bytesize, "restore host v_cache")
            idx += 2
          end
        end
      end
      raise "unused host decode snapshot buffers" unless idx == snapshot.buffers.size
    end

    def copy_decode_state_to!(target : QwenMixedStackRunner, include_kv : Bool = true) : Nil
      raise ArgumentError.new("layer ids mismatch") unless @layer_ids == target.layer_ids
      raise ArgumentError.new("runner count mismatch") unless @runners.size == target.runners.size

      @runners.zip(target.runners).each_with_index do |(source, dest), idx|
        case source
        in QwenRecurrentLayerRunner
          raise "runner #{idx} type mismatch" unless dest.is_a?(QwenRecurrentLayerRunner)
          copy_recurrent_state_to!(source, dest)
        in QwenFullAttnLayerRunner
          raise "runner #{idx} type mismatch" unless dest.is_a?(QwenFullAttnLayerRunner)
          copy_kv_state_to!(source, dest) if include_kv
        end
      end
    end

    private def snapshot_device_buffer!(buffers : Array(DeviceBuffer), source : DevicePtr, bytesize : LibC::SizeT, label : String) : Nil
      buffer = DeviceBuffer.new(bytesize)
      ML::CUDA.copy_dtod!(buffer.ptr, source, bytesize, label)
      buffers << buffer
    end

    private def snapshot_host_buffer!(buffers : Array(Bytes), source : DevicePtr, bytesize : LibC::SizeT, label : String) : Nil
      buffer = Bytes.new(bytesize.to_i)
      ML::CUDA.copy_dtoh!(buffer.to_unsafe.as(Void*), source, bytesize, label)
      buffers << buffer
    end

    private def copy_recurrent_state_to!(source : QwenRecurrentLayerRunner, dest : QwenRecurrentLayerRunner) : Nil
      raise "conv_state size mismatch" unless source.conv_state_bytesize == dest.conv_state_bytesize
      raise "ssm_state size mismatch" unless source.ssm_state_bytesize == dest.ssm_state_bytesize

      ML::CUDA.copy_dtod!(dest.conv_state_device_ptr, source.conv_state_device_ptr,
        source.conv_state_bytesize, "copy conv_state")
      ML::CUDA.copy_dtod!(dest.ssm_state_device_ptr, source.ssm_state_device_ptr,
        source.ssm_state_bytesize, "copy ssm_state")
    end

    private def copy_kv_state_to!(source : QwenFullAttnLayerRunner, dest : QwenFullAttnLayerRunner) : Nil
      raise "kv cache size mismatch" unless source.kv_cache_bytesize == dest.kv_cache_bytesize

      ML::CUDA.copy_dtod!(dest.k_cache_device_ptr, source.k_cache_device_ptr,
        source.kv_cache_bytesize, "copy k_cache")
      ML::CUDA.copy_dtod!(dest.v_cache_device_ptr, source.v_cache_device_ptr,
        source.kv_cache_bytesize, "copy v_cache")
    end

    def run_sequence(profile_phases : Bool = false,
                     debug_readback : Bool = true,
                     reset_sequence : Bool = true,
                     sync_end : Bool = true,
                     read_head_outputs : Bool = true,
                     run_head : Bool = true) : Float64
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

      if run_head
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
      else
        @phase_lines << "phase_head_reset_ms=skipped"
        @phase_lines << "phase_head_ms=skipped"
        ML::CUDA.synchronize!("cuCtxSynchronize(mixed stack)") if sync_end
      end

      t_read = Time.instant
      @runners.each(&.read_outputs) if debug_readback
      @head.read_outputs if run_head && read_head_outputs
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
