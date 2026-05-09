require "./qwen_full_attn_projection_runner"
require "./qwen_full_attn_kv_runner"

module ML::CUDA
  # Composes the reusable CUDA full-attention projection and KV/tail runners
  # into one residual-hidden -> final-hidden layer boundary.
  class QwenFullAttnLayerRunner
    class Weights
      getter projection : QwenFullAttnProjectionRunner::Weights
      getter kv : QwenFullAttnKVRunner::Weights

      def self.load(gguf : ML::GGUF::GGUFFile, layer : Int32) : self
        new(QwenFullAttnProjectionRunner::Weights.load(gguf, layer),
          QwenFullAttnKVRunner::Weights.load(gguf, layer))
      end

      def initialize(@projection : QwenFullAttnProjectionRunner::Weights,
                     @kv : QwenFullAttnKVRunner::Weights)
      end
    end

    getter projection : QwenFullAttnProjectionRunner
    getter kv : QwenFullAttnKVRunner

    def self.from_weights(weights : Weights,
                          tokens : Int32,
                          max_seq : Int32,
                          start_pos : Int32,
                          n_head : Int32,
                          n_head_kv : Int32,
                          head_dim : Int32,
                          rope_dim : Int32,
                          eps : Float32,
                          residual_input : Array(Float32),
                          cos_table : Array(Float32),
                          sin_table : Array(Float32)) : self
      new(weights.projection, weights.kv, tokens, max_seq, start_pos, n_head, n_head_kv,
        head_dim, rope_dim, eps, residual_input, cos_table, sin_table)
    end

    private def initialize(proj_weights : QwenFullAttnProjectionRunner::Weights,
                           kv_weights : QwenFullAttnKVRunner::Weights,
                           @tokens : Int32,
                           @max_seq : Int32,
                           @start_pos : Int32,
                           @n_head : Int32,
                           @n_head_kv : Int32,
                           @head_dim : Int32,
                           @rope_dim : Int32,
                           @eps : Float32,
                           @residual_input : Array(Float32),
                           @cos_table : Array(Float32),
                           @sin_table : Array(Float32))
      @projection = QwenFullAttnProjectionRunner.from_weights_with_input_norm(proj_weights, @tokens,
        @residual_input, kv_weights.attn_norm, @eps)
      @kv = QwenFullAttnKVRunner.new(@tokens, @max_seq, @start_pos, @n_head, @n_head_kv,
        @head_dim, @rope_dim, @eps, @projection.q_device_ptr, @projection.k_device_ptr,
        @projection.v_device_ptr, kv_weights, @residual_input, @cos_table, @sin_table)
      @kv.use_device_residual_input(@projection.sequence_input_device_ptr)
      @closed = false
    end

    def upload_weights : Nil
      @projection.upload_weights
      @kv.upload_constants
    end

    def reset_sequence : Nil
      @projection.reset_sequence
      @kv.reset_sequence
    end

    def run_sequence : Nil
      @projection.run_sequence
      @kv.run_sequence
    end

    def run_sequence_profiled(phase_lines : Array(String), prefix : String) : Nil
      t_total = Time.instant

      t_projection = Time.instant
      @projection.run_sequence
      ML::CUDA.synchronize!("cuCtxSynchronize(full attention projection)")
      phase_lines << "#{prefix}_projection_ms=#{((Time.instant - t_projection).total_milliseconds).round(3)}"

      t_kv = Time.instant
      @kv.run_sequence_profiled(phase_lines, "#{prefix}_kv")
      phase_lines << "#{prefix}_kv_tail_ms=#{((Time.instant - t_kv).total_milliseconds).round(3)}"
      phase_lines << "#{prefix}_profiled_ms=#{((Time.instant - t_total).total_milliseconds).round(3)}"
    end

    def read_outputs : Nil
      @kv.read_outputs
    end

    def use_device_sequence_input(ptr : DevicePtr) : Nil
      @projection.use_device_sequence_input(ptr)
      @kv.use_device_residual_input(ptr)
    end

    def upload_sequence_input(xs : Array(Float32)) : Nil
      @projection.upload_sequence_input(xs)
      @kv.use_device_residual_input(@projection.sequence_input_device_ptr)
    end

    def update_decode_position(start_pos : Int32, cos_table : Array(Float32), sin_table : Array(Float32)) : Nil
      @kv.update_decode_position(start_pos, cos_table, sin_table)
    end

    def output_device_ptr : DevicePtr
      @kv.output_device_ptr
    end

    def final_gpu_all : Array(Float32)
      @kv.final_gpu_all
    end

    def close : Nil
      return if @closed

      @kv.close
      @projection.close
      @closed = true
    end
  end
end
