require "./diffusion_gemma_runtime"
require "./gemma4_cpu"
require "./gemma4_metal"
require "./qwen35_metal"

module ML::GGUF
  module DiffusionGemmaCPU
    extend self

    struct AttentionProjection
      getter q : Array(Float32)
      getter k : Array(Float32)
      getter v : Array(Float32)
      getter reused_k_as_v : Bool

      def initialize(@q, @k, @v, @reused_k_as_v)
      end
    end

    class PromptLayerMetalCache
      getter k_cache_buf : ML::MetalBuffer
      getter v_cache_buf : ML::MetalBuffer
      getter q_buf : ML::MetalBuffer
      getter out_buf : ML::MetalBuffer
      getter prompt_len : Int32
      getter canvas_len : Int32
      getter q_dim : Int32
      getter kv_dim : Int32

      def initialize(prompt_projections : Array(AttentionProjection),
                     @prompt_len : Int32,
                     @canvas_len : Int32,
                     @q_dim : Int32,
                     @kv_dim : Int32)
        raise ArgumentError.new("prompt projection count mismatch") unless prompt_projections.size == @prompt_len
        total_tokens = @prompt_len + @canvas_len
        k_cache = Array(Float32).new(total_tokens * @kv_dim, 0.0_f32)
        v_cache = Array(Float32).new(total_tokens * @kv_dim, 0.0_f32)
        prompt_projections.each_with_index do |proj, pos|
          raise ArgumentError.new("prompt k size mismatch") unless proj.k.size == @kv_dim
          raise ArgumentError.new("prompt v size mismatch") unless proj.v.size == @kv_dim
          copy_projection_kv_to_rows!(proj, k_cache, v_cache, pos, @kv_dim)
        end
        @k_cache_buf = ML::MetalBuffer.from_array(k_cache)
        @v_cache_buf = ML::MetalBuffer.from_array(v_cache)
        @q_buf = ML::MetalBuffer.new(@canvas_len.to_i64 * @q_dim * sizeof(Float32))
        @out_buf = ML::MetalBuffer.new(@canvas_len.to_i64 * @q_dim * sizeof(Float32))
      end

      def write_canvas!(canvas_projections : Array(AttentionProjection)) : Nil
        raise ArgumentError.new("canvas projection count mismatch") unless canvas_projections.size == @canvas_len
        canvas_projections.each_with_index do |proj, canvas_pos|
          raise ArgumentError.new("canvas k size mismatch") unless proj.k.size == @kv_dim
          raise ArgumentError.new("canvas v size mismatch") unless proj.v.size == @kv_dim
          offset = (@prompt_len + canvas_pos) * @kv_dim
          (@k_cache_buf.contents.as(Pointer(Float32)) + offset).copy_from(proj.k.to_unsafe, @kv_dim)
          (@v_cache_buf.contents.as(Pointer(Float32)) + offset).copy_from(proj.v.to_unsafe, @kv_dim)
        end
      end

      def write_query!(query : AttentionProjection) : Nil
        raise ArgumentError.new("query q size mismatch") unless query.q.size == @q_dim
        @q_buf.write(query.q)
      end

      def write_queries!(queries : Array(AttentionProjection)) : Nil
        raise ArgumentError.new("query count mismatch") unless queries.size == @canvas_len
        q_ptr = @q_buf.contents.as(Pointer(Float32))
        queries.each_with_index do |query, pos|
          raise ArgumentError.new("query q size mismatch") unless query.q.size == @q_dim
          (q_ptr + pos * @q_dim).copy_from(query.q.to_unsafe, @q_dim)
        end
      end

      private def copy_projection_kv_to_rows!(proj : AttentionProjection,
                                              k_rows : Array(Float32),
                                              v_rows : Array(Float32),
                                              pos : Int32,
                                              kv_dim : Int32) : Nil
        k_dst = k_rows.to_unsafe + pos * kv_dim
        v_dst = v_rows.to_unsafe + pos * kv_dim
        k_src = proj.k.to_unsafe
        v_src = proj.v.to_unsafe
        d = 0
        while d < kv_dim
          k_dst[d] = k_src[d]
          v_dst[d] = v_src[d]
          d += 1
        end
      end
    end

    struct ExpertRoute
      getter expert : Int32
      getter weight : Float32

      def initialize(@expert, @weight)
      end
    end

    struct BoundedDenoisePrediction
      getter candidate_token_ids : Array(Int32)
      getter logits : Array(Float32)
      getter probabilities : Array(Float32)
      getter argmax_token_id : Int32
      getter sampled_token_id : Int32
      getter entropy : Float32

      def initialize(@candidate_token_ids,
                     @logits,
                     @probabilities,
                     @argmax_token_id,
                     @sampled_token_id,
                     @entropy)
      end
    end

    struct BoundedCanvasUpdate
      getter updated_canvas_tokens : Array(Int32)
      getter accepted : Array(Bool)
      getter predictions : Array(BoundedDenoisePrediction)
      getter updated_canvas_rows : Array(Float32)?

      def initialize(@updated_canvas_tokens,
                     @accepted,
                     @predictions,
                     @updated_canvas_rows = nil)
      end
    end

    struct BoundedDenoiseStepTrace
      getter step : Int32
      getter prediction_count : Int32
      getter accepted_count : Int32
      getter total_candidate_tokens : Int32
      getter max_candidate_tokens : Int32
      getter mean_candidate_tokens : Float32
      getter mean_entropy : Float32
      getter prediction_ms : Float64
      getter decode_stack_ms : Float64
      getter decode_qkv_ms : Float64
      getter decode_context_ms : Float64
      getter decode_context_score_ms : Float64
      getter decode_context_softmax_ms : Float64
      getter decode_context_value_ms : Float64
      getter decode_attention_out_ms : Float64
      getter decode_shared_ffn_ms : Float64
      getter decode_moe_ffn_ms : Float64
      getter decode_combine_scale_ms : Float64
      getter output_head_ms : Float64
      getter update_ms : Float64
      getter regenerate_ms : Float64
      getter proposal_ms : Float64
      getter decode_attention_residual_context_buffer : Bool

      def initialize(@step,
                     @prediction_count,
                     @accepted_count,
                     @total_candidate_tokens,
                     @max_candidate_tokens,
                     @mean_candidate_tokens,
                     @mean_entropy,
                     @prediction_ms = 0.0,
                     @decode_stack_ms = 0.0,
                     @decode_qkv_ms = 0.0,
                     @decode_context_ms = 0.0,
                     @decode_context_score_ms = 0.0,
                     @decode_context_softmax_ms = 0.0,
                     @decode_context_value_ms = 0.0,
                     @decode_attention_out_ms = 0.0,
                     @decode_shared_ffn_ms = 0.0,
                     @decode_moe_ffn_ms = 0.0,
                     @decode_combine_scale_ms = 0.0,
                     @output_head_ms = 0.0,
                     @update_ms = 0.0,
                     @regenerate_ms = 0.0,
                     @proposal_ms = 0.0,
                     @decode_attention_residual_context_buffer = false)
      end
    end

    struct BoundedDenoiseLoopSummary
      getter steps_run : Int32
      getter converged : Bool
      getter stop_reason : String
      getter prediction_count : Int32
      getter accepted_count : Int32
      getter total_candidate_tokens : Int32
      getter max_candidate_tokens : Int32
      getter mean_candidate_tokens : Float32
      getter mean_entropy : Float32
      getter acceptance_rate : Float32

      def initialize(@steps_run,
                     @converged,
                     @stop_reason,
                     @prediction_count,
                     @accepted_count,
                     @total_candidate_tokens,
                     @max_candidate_tokens,
                     @mean_candidate_tokens,
                     @mean_entropy,
                     @acceptance_rate)
      end
    end

    struct BoundedDenoisePredictionTiming
      getter predictions : Array(BoundedDenoisePrediction)
      getter decode_stack_ms : Float64
      getter decode_qkv_ms : Float64
      getter decode_context_ms : Float64
      getter decode_context_score_ms : Float64
      getter decode_context_softmax_ms : Float64
      getter decode_context_value_ms : Float64
      getter decode_attention_out_ms : Float64
      getter decode_shared_ffn_ms : Float64
      getter decode_moe_ffn_ms : Float64
      getter decode_combine_scale_ms : Float64
      getter output_head_ms : Float64
      getter decode_attention_residual_context_buffer : Bool

      def initialize(@predictions,
                     @decode_stack_ms,
                     @decode_qkv_ms,
                     @decode_context_ms,
                     @decode_attention_out_ms,
                     @decode_shared_ffn_ms,
                     @decode_moe_ffn_ms,
                     @decode_combine_scale_ms,
                     @output_head_ms,
                     @decode_context_score_ms = 0.0,
                     @decode_context_softmax_ms = 0.0,
                     @decode_context_value_ms = 0.0,
                     @decode_attention_residual_context_buffer = false)
      end
    end

    struct BoundedDenoiseStepTiming
      getter update : BoundedCanvasUpdate
      getter prediction_ms : Float64
      getter decode_stack_ms : Float64
      getter decode_qkv_ms : Float64
      getter decode_context_ms : Float64
      getter decode_context_score_ms : Float64
      getter decode_context_softmax_ms : Float64
      getter decode_context_value_ms : Float64
      getter decode_attention_out_ms : Float64
      getter decode_shared_ffn_ms : Float64
      getter decode_moe_ffn_ms : Float64
      getter decode_combine_scale_ms : Float64
      getter output_head_ms : Float64
      getter update_ms : Float64
      getter regenerate_ms : Float64
      getter decode_attention_residual_context_buffer : Bool

      def initialize(@update,
                     @prediction_ms,
                     @decode_stack_ms,
                     @decode_qkv_ms,
                     @decode_context_ms,
                     @decode_attention_out_ms,
                     @decode_shared_ffn_ms,
                     @decode_moe_ffn_ms,
                     @decode_combine_scale_ms,
                     @output_head_ms,
                     @update_ms,
                     @regenerate_ms,
                     @decode_context_score_ms = 0.0,
                     @decode_context_softmax_ms = 0.0,
                     @decode_context_value_ms = 0.0,
                     @decode_attention_residual_context_buffer = false)
      end
    end

    struct DecodeCanvasRowsTiming
      getter rows : Array(Float32)
      getter qkv_ms : Float64
      getter context_ms : Float64
      getter context_score_ms : Float64
      getter context_softmax_ms : Float64
      getter context_value_ms : Float64
      getter attention_out_ms : Float64
      getter shared_ffn_ms : Float64
      getter moe_ffn_ms : Float64
      getter moe_grouped_prep_ms : Float64
      getter moe_grouped_gate_up_ms : Float64
      getter moe_grouped_activation_ms : Float64
      getter moe_grouped_down_ms : Float64
      getter moe_grouped_scatter_combine_norm_ms : Float64
      getter combine_scale_ms : Float64
      getter attention_residual_context_buffer : Bool

      def initialize(@rows,
                     @qkv_ms,
                     @context_ms,
                     @attention_out_ms,
                     @shared_ffn_ms,
                     @moe_ffn_ms,
                     @combine_scale_ms,
                     @context_score_ms = 0.0,
                     @context_softmax_ms = 0.0,
                     @context_value_ms = 0.0,
                     @attention_residual_context_buffer = false,
                     @moe_grouped_prep_ms = 0.0,
                     @moe_grouped_gate_up_ms = 0.0,
                     @moe_grouped_activation_ms = 0.0,
                     @moe_grouped_down_ms = 0.0,
                     @moe_grouped_scatter_combine_norm_ms = 0.0)
      end
    end

    struct GroupedMoeRowsTiming
      getter rows : Array(Float32)
      getter prep_ms : Float64
      getter gate_up_ms : Float64
      getter activation_ms : Float64
      getter down_ms : Float64
      getter scatter_combine_norm_ms : Float64

      def initialize(@rows,
                     @prep_ms,
                     @gate_up_ms,
                     @activation_ms,
                     @down_ms,
                     @scatter_combine_norm_ms)
      end
    end

    enum CogniGraphPlanAccess
      Read
      Write
      ReadWrite
    end

    record CogniGraphPlanBinding,
      buffer : String,
      access : CogniGraphPlanAccess,
      offset : Int64 = 0_i64,
      length : Int64 = -1_i64,
      partition : Int32 = -1 do
      def conflicts?(other : CogniGraphPlanBinding) : Bool
        return false if @buffer != other.buffer
        return false if @partition >= 0 && other.partition >= 0 && @partition != other.partition
        return true if @length < 0 || other.length < 0

        a_end = @offset + @length
        b_end = other.offset + other.length
        @offset < b_end && other.offset < a_end
      end
    end

    struct CogniGraphPlanOp
      getter name : String
      getter bindings : Array(CogniGraphPlanBinding)

      def initialize(@name, @bindings)
      end
    end

    struct CogniGraphPlan
      getter n_ops : Int32
      getter n_waves : Int32
      getter n_barriers : Int32
      getter max_wave_width : Int32
      getter active_experts : Int32
      getter route_slots : Int32
      getter row_count : Int32
      getter wave_widths : Array(Int32)
      getter wave_names : Array(Array(String))

      def initialize(@n_ops,
                     @n_waves,
                     @n_barriers,
                     @max_wave_width,
                     @active_experts,
                     @route_slots,
                     @row_count,
                     @wave_widths,
                     @wave_names)
      end

      def phi : String
        "Phi=(moe_ffn,#{@active_experts},#{@n_barriers},#{@route_slots})"
      end
    end

    struct GroupedMoeResidentGraphStats
      getter n_ops : Int32
      getter n_waves : Int32
      getter n_barriers : Int32
      getter max_wave_width : Int32
      getter active_experts : Int32
      getter route_slots : Int32
      getter row_count : Int32

      def initialize(@n_ops,
                     @n_waves,
                     @n_barriers,
                     @max_wave_width,
                     @active_experts,
                     @route_slots,
                     @row_count)
      end

      def phi : String
        "Phi=(resident_moe_matmul,#{@active_experts},#{@n_barriers},#{@route_slots})"
      end
    end

    struct AttentionContextTiming
      getter context : Array(Float32)
      getter score_ms : Float64
      getter softmax_ms : Float64
      getter value_ms : Float64

      def initialize(@context,
                     @score_ms,
                     @softmax_ms,
                     @value_ms)
      end
    end

    struct BoundedDenoiseLoopResult
      getter final_canvas_tokens : Array(Int32)
      getter final_canvas_rows : Array(Float32)?
      getter updates : Array(BoundedCanvasUpdate)
      getter stable_counts : Array(Int32)
      getter steps_run : Int32
      getter converged : Bool
      getter step_traces : Array(BoundedDenoiseStepTrace)
      getter stop_reason : String

      def initialize(@final_canvas_tokens,
                     @final_canvas_rows,
                     @updates,
                     @stable_counts,
                     @steps_run,
                     @converged,
                     step_traces = nil,
                     stop_reason = nil)
        @step_traces = step_traces || [] of BoundedDenoiseStepTrace
        @stop_reason = stop_reason || (@converged ? "converged" : "exhausted")
      end

      def accepted_token_count : Int32
        total = 0
        @step_traces.each { |trace| total += trace.accepted_count }
        total
      end

      def summary : BoundedDenoiseLoopSummary
        prediction_count = 0
        accepted_count = 0
        total_candidate_tokens = 0
        max_candidate_tokens = 0
        entropy_weighted_sum = 0.0_f32
        @step_traces.each do |trace|
          prediction_count += trace.prediction_count
          accepted_count += trace.accepted_count
          total_candidate_tokens += trace.total_candidate_tokens
          max_candidate_tokens = trace.max_candidate_tokens if trace.max_candidate_tokens > max_candidate_tokens
          entropy_weighted_sum += trace.mean_entropy * trace.prediction_count.to_f32
        end

        mean_candidate_tokens = prediction_count == 0 ? 0.0_f32 : total_candidate_tokens.to_f32 / prediction_count.to_f32
        mean_entropy = prediction_count == 0 ? 0.0_f32 : entropy_weighted_sum / prediction_count.to_f32
        acceptance_rate = prediction_count == 0 ? 0.0_f32 : accepted_count.to_f32 / prediction_count.to_f32

        BoundedDenoiseLoopSummary.new(
          steps_run: @steps_run,
          converged: @converged,
          stop_reason: @stop_reason,
          prediction_count: prediction_count,
          accepted_count: accepted_count,
          total_candidate_tokens: total_candidate_tokens,
          max_candidate_tokens: max_candidate_tokens,
          mean_candidate_tokens: mean_candidate_tokens,
          mean_entropy: mean_entropy,
          acceptance_rate: acceptance_rate,
        )
      end
    end

    struct PromptLayerCache
      getter final_rows : Array(Float32)
      getter projections_by_layer : Array(Array(AttentionProjection))
      getter projection_ms_by_layer : Array(Float64)
      getter projection_norm_ms_by_layer : Array(Float64)
      getter projection_matmul_ms_by_layer : Array(Float64)
      getter projection_assemble_ms_by_layer : Array(Float64)
      getter projection_copy_ms_by_layer : Array(Float64)
      getter projection_head_norm_ms_by_layer : Array(Float64)
      getter projection_q_norm_ms_by_layer : Array(Float64)
      getter projection_k_norm_ms_by_layer : Array(Float64)
      getter projection_v_norm_ms_by_layer : Array(Float64)
      getter projection_rope_ms_by_layer : Array(Float64)
      getter projection_rope_table_ms_by_layer : Array(Float64)
      getter projection_rope_apply_ms_by_layer : Array(Float64)
      getter projection_rope_q_apply_ms_by_layer : Array(Float64)
      getter projection_rope_k_apply_ms_by_layer : Array(Float64)
      getter materialize_ms_by_layer : Array(Float64)
      getter metal_cache_by_layer : Array(PromptLayerMetalCache?)

      def initialize(@final_rows,
                     @projections_by_layer,
                     @projection_ms_by_layer = [] of Float64,
                     @projection_norm_ms_by_layer = [] of Float64,
                     @projection_matmul_ms_by_layer = [] of Float64,
                     @projection_assemble_ms_by_layer = [] of Float64,
                     @projection_copy_ms_by_layer = [] of Float64,
                     @projection_head_norm_ms_by_layer = [] of Float64,
                     @projection_q_norm_ms_by_layer = [] of Float64,
                     @projection_k_norm_ms_by_layer = [] of Float64,
                     @projection_v_norm_ms_by_layer = [] of Float64,
                     @projection_rope_ms_by_layer = [] of Float64,
                     @projection_rope_table_ms_by_layer = [] of Float64,
                     @projection_rope_apply_ms_by_layer = [] of Float64,
                     @projection_rope_q_apply_ms_by_layer = [] of Float64,
                     @projection_rope_k_apply_ms_by_layer = [] of Float64,
                     @materialize_ms_by_layer = [] of Float64,
                     @metal_cache_by_layer = [] of PromptLayerMetalCache?)
      end

      def layers : Int32
        @projections_by_layer.size
      end
    end

    struct PromptProjectionTiming
      getter projections : Array(AttentionProjection)
      getter norm_ms : Float64
      getter matmul_ms : Float64
      getter assemble_ms : Float64
      getter copy_ms : Float64
      getter head_norm_ms : Float64
      getter q_norm_ms : Float64
      getter k_norm_ms : Float64
      getter v_norm_ms : Float64
      getter rope_ms : Float64
      getter rope_table_ms : Float64
      getter rope_apply_ms : Float64
      getter rope_q_apply_ms : Float64
      getter rope_k_apply_ms : Float64

      def initialize(@projections,
                     @norm_ms,
                     @matmul_ms,
                     @assemble_ms,
                     @copy_ms,
                     @head_norm_ms,
                     @q_norm_ms,
                     @k_norm_ms,
                     @v_norm_ms,
                     @rope_ms,
                     @rope_table_ms,
                     @rope_apply_ms,
                     @rope_q_apply_ms,
                     @rope_k_apply_ms)
      end
    end

    def embedding_lookup(weights : DiffusionGemmaWeights, token_id : Int32) : Array(Float32)
      Gemma4CPU.embedding_lookup(weights.token_embd, token_id)
    end

    def scaled_embedding_lookup(weights : DiffusionGemmaWeights, token_id : Int32) : Array(Float32)
      x = embedding_lookup(weights, token_id)
      scale = Math.sqrt(weights.hparams.n_embd.to_f64).to_f32
      x.size.times { |i| x[i] *= scale }
      x
    end

    # Canvas rows use no-scale RMSNorm after the shared scaled token embedding.
    # Self-conditioning is intentionally excluded here; this is the zero-SC
    # exactness path and the first native boundary for oracle comparison.
    def zero_sc_canvas_embedding(weights : DiffusionGemmaWeights, token_id : Int32) : Array(Float32)
      x = scaled_embedding_lookup(weights, token_id)
      Gemma4CPU.rms_norm_plain!(x, weights.hparams.rms_eps)
      x
    end

    def self_conditioning_soft_embedding(weights : DiffusionGemmaWeights,
                                         token_ids : Array(Int32),
                                         logits : Array(Float32),
                                         temp_inv : Float32 = 1.0_f32) : Array(Float32)
      hp = weights.hparams
      raise ArgumentError.new("self-conditioning token_ids must not be empty") if token_ids.empty?
      raise ArgumentError.new("self-conditioning logits size mismatch") unless logits.size == token_ids.size
      raise ArgumentError.new("self-conditioning temp_inv must be finite and positive") unless temp_inv.finite? && temp_inv > 0.0_f32

      scaled_logits = logits.map { |v| v * temp_inv }
      probs = softmax(scaled_logits)
      result = Array(Float32).new(hp.n_embd, 0.0_f32)
      token_ids.each_with_index do |token_id, i|
        emb = scaled_embedding_lookup(weights, token_id)
        weight = probs[i]
        hp.n_embd.times { |j| result[j] += weight * emb[j] }
      end
      result
    end

    def self_conditioning_signal(weights : DiffusionGemmaWeights,
                                 soft_embedding : Array(Float32)) : Array(Float32)
      hp = weights.hparams
      sc = weights.self_conditioning
      raise ArgumentError.new("self-conditioning embedding size mismatch") unless soft_embedding.size == hp.n_embd

      normed = Gemma4CPU.rms_norm(soft_embedding, sc.pre_norm, hp.rms_eps)
      gate = Gemma4CPU.matmul(sc.gate_qw, normed)
      up = Gemma4CPU.matmul(sc.up_qw, normed)
      raise ArgumentError.new("self-conditioning gate/up size mismatch") unless gate.size == hp.n_ff && up.size == hp.n_ff
      gate.size.times { |i| gate[i] = Gemma4CPU.gelu(gate[i]) * up[i] }
      signal = Gemma4CPU.matmul(sc.down_qw, gate)
      raise ArgumentError.new("self-conditioning signal size mismatch") unless signal.size == hp.n_embd
      signal
    end

    def canvas_embedding_with_self_conditioning(weights : DiffusionGemmaWeights,
                                                token_id : Int32,
                                                sc_token_ids : Array(Int32),
                                                sc_logits : Array(Float32),
                                                temp_inv : Float32 = 1.0_f32,
                                                sc_use : Float32 = 1.0_f32) : Array(Float32)
      hp = weights.hparams
      raise ArgumentError.new("self-conditioning use gate must be finite") unless sc_use.finite?

      canvas = scaled_embedding_lookup(weights, token_id)
      if sc_use != 0.0_f32
        soft = self_conditioning_soft_embedding(weights, sc_token_ids, sc_logits, temp_inv)
        signal = self_conditioning_signal(weights, soft)
        hp.n_embd.times { |i| canvas[i] += sc_use * signal[i] }
      end
      Gemma4CPU.rms_norm_plain!(canvas, hp.rms_eps)
      canvas
    end

    def region_embeddings(weights : DiffusionGemmaWeights,
                          request : DiffusionGemmaRequest) : Array(Float32)
      hp = weights.hparams
      result = Array(Float32).new(request.total_tokens * hp.n_embd, 0.0_f32)
      row = 0
      request.prompt_tokens.each do |token_id|
        copy_row!(result, row, hp.n_embd, scaled_embedding_lookup(weights, token_id))
        row += 1
      end
      request.canvas_tokens.each do |token_id|
        copy_row!(result, row, hp.n_embd, zero_sc_canvas_embedding(weights, token_id))
        row += 1
      end
      result
    end

    def canvas_rows_from_tokens(weights : DiffusionGemmaWeights,
                                canvas_tokens : Array(Int32),
                                sc_token_ids_by_canvas_row : Array(Array(Int32))? = nil,
                                sc_logits_by_canvas_row : Array(Array(Float32))? = nil,
                                sc_temp_inv : Float32 = 1.0_f32,
                                sc_use : Float32 = 1.0_f32) : Array(Float32)
      hp = weights.hparams
      if sc_token_ids_by_canvas_row || sc_logits_by_canvas_row
        raise ArgumentError.new("self-conditioning token/logit rows must be supplied together") unless sc_token_ids_by_canvas_row && sc_logits_by_canvas_row
        raise ArgumentError.new("self-conditioning token rows size mismatch") unless sc_token_ids_by_canvas_row.not_nil!.size == canvas_tokens.size
        raise ArgumentError.new("self-conditioning logit rows size mismatch") unless sc_logits_by_canvas_row.not_nil!.size == canvas_tokens.size
      end

      result = Array(Float32).new(canvas_tokens.size * hp.n_embd, 0.0_f32)
      canvas_tokens.each_with_index do |token_id, row|
        values = if sc_token_ids_by_canvas_row && sc_logits_by_canvas_row
                   canvas_embedding_with_self_conditioning(
                     weights,
                     token_id,
                     sc_token_ids_by_canvas_row.not_nil![row],
                     sc_logits_by_canvas_row.not_nil![row],
                     temp_inv: sc_temp_inv,
                     sc_use: sc_use,
                   )
                 else
                   zero_sc_canvas_embedding(weights, token_id)
                 end
        copy_row!(result, row, hp.n_embd, values)
      end
      result
    end

    def canvas_rows_from_prediction_self_conditioning(weights : DiffusionGemmaWeights,
                                                      canvas_tokens : Array(Int32),
                                                      predictions : Array(BoundedDenoisePrediction),
                                                      sc_temp_inv : Float32 = 1.0_f32,
                                                      sc_use : Float32 = 1.0_f32) : Array(Float32)
      raise ArgumentError.new("prediction self-conditioning count mismatch") unless predictions.size == canvas_tokens.size
      canvas_rows_from_tokens(
        weights,
        canvas_tokens,
        sc_token_ids_by_canvas_row: predictions.map(&.candidate_token_ids),
        sc_logits_by_canvas_row: predictions.map(&.logits),
        sc_temp_inv: sc_temp_inv,
        sc_use: sc_use,
      )
    end

    def attention_project_pre_norm(lw : DiffusionGemmaLayerWeights, x_norm : Array(Float32)) : AttentionProjection
      q = prompt_projection_matmul(lw.attn_q_qw, x_norm, 1)
      k = prompt_projection_matmul(lw.attn_k_qw, x_norm, 1)
      if v_qw = lw.attn_v_qw
        v = prompt_projection_matmul(v_qw, x_norm, 1)
        AttentionProjection.new(q, k, v, false)
      else
        AttentionProjection.new(q, k, k.dup, true)
      end
    end

    def attention_project_normed(weights : DiffusionGemmaWeights,
                                 il : Int32,
                                 x : Array(Float32),
                                 pos : Int32) : AttentionProjection
      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("attention_project input size mismatch") unless x.size == hp.n_embd
      x_norm = Gemma4CPU.rms_norm(x, lw.attn_norm, hp.rms_eps)
      proj = attention_project_pre_norm(lw, x_norm)
      normalize_attention_projection!(proj, lw, hp, il)
      apply_rope_to_qk!(proj, hp, il, pos, weights.rope_freqs)
      proj
    end

    def normalize_attention_projection!(proj : AttentionProjection,
                                        lw : DiffusionGemmaLayerWeights,
                                        hp : DiffusionGemmaHparams,
                                        il : Int32) : Nil
      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)

      raise ArgumentError.new("q projection size mismatch at layer #{il}") unless proj.q.size == n_head * head_dim
      raise ArgumentError.new("k projection size mismatch at layer #{il}") unless proj.k.size == n_head_kv * head_dim
      raise ArgumentError.new("v projection size mismatch at layer #{il}") unless proj.v.size == n_head_kv * head_dim

      n_head.times do |h|
        fast_rms_norm_slice!(proj.q, h * head_dim, head_dim, lw.attn_q_norm, hp.rms_eps)
      end
      n_head_kv.times do |h|
        off = h * head_dim
        fast_rms_norm_slice!(proj.k, off, head_dim, lw.attn_k_norm, hp.rms_eps)
        fast_rms_norm_plain_slice!(proj.v, off, head_dim, hp.rms_eps)
      end
    end

    private def normalize_attention_projection_timed!(proj : AttentionProjection,
                                                      lw : DiffusionGemmaLayerWeights,
                                                      hp : DiffusionGemmaHparams,
                                                      il : Int32) : {Float64, Float64, Float64}
      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)

      raise ArgumentError.new("q projection size mismatch at layer #{il}") unless proj.q.size == n_head * head_dim
      raise ArgumentError.new("k projection size mismatch at layer #{il}") unless proj.k.size == n_head_kv * head_dim
      raise ArgumentError.new("v projection size mismatch at layer #{il}") unless proj.v.size == n_head_kv * head_dim

      q_t0 = Time.instant
      n_head.times do |h|
        fast_rms_norm_slice!(proj.q, h * head_dim, head_dim, lw.attn_q_norm, hp.rms_eps)
      end
      q_ms = (Time.instant - q_t0).total_milliseconds

      k_t0 = Time.instant
      n_head_kv.times do |h|
        fast_rms_norm_slice!(proj.k, h * head_dim, head_dim, lw.attn_k_norm, hp.rms_eps)
      end
      k_ms = (Time.instant - k_t0).total_milliseconds

      v_t0 = Time.instant
      n_head_kv.times do |h|
        fast_rms_norm_plain_slice!(proj.v, h * head_dim, head_dim, hp.rms_eps)
      end
      v_ms = (Time.instant - v_t0).total_milliseconds

      {q_ms, k_ms, v_ms}
    end

    def apply_rope_to_qk!(proj : AttentionProjection,
                          hp : DiffusionGemmaHparams,
                          il : Int32,
                          pos : Int32,
                          rope_freqs : Array(Float32)? = nil) : Nil
      head_dim = hp.head_dim_for_layer(il)
      n_rot = hp.rope_dim_for_layer(il)
      base = hp.rope_freq_base_for_layer(il)
      freqs = hp.full_attention?(il) ? rope_freqs : nil
      cos_table, sin_table = rope_tables(pos, n_rot, base, freqs)

      hp.n_head.times do |h|
        fast_rope_neox_slice!(proj.q, h * head_dim, n_rot, head_dim, cos_table, sin_table)
      end
      hp.n_head_kv(il).times do |h|
        fast_rope_neox_slice!(proj.k, h * head_dim, n_rot, head_dim, cos_table, sin_table)
      end
    end

    private def apply_rope_to_qk_timed!(proj : AttentionProjection,
                                        hp : DiffusionGemmaHparams,
                                        il : Int32,
                                        pos : Int32,
                                        rope_freqs : Array(Float32)? = nil) : {Float64, Float64, Float64, Float64}
      head_dim = hp.head_dim_for_layer(il)
      n_rot = hp.rope_dim_for_layer(il)
      base = hp.rope_freq_base_for_layer(il)
      freqs = hp.full_attention?(il) ? rope_freqs : nil
      table_t0 = Time.instant
      cos_table, sin_table = rope_tables(pos, n_rot, base, freqs)
      table_ms = (Time.instant - table_t0).total_milliseconds

      apply_t0 = Time.instant
      q_apply_t0 = Time.instant
      hp.n_head.times do |h|
        fast_rope_neox_slice!(proj.q, h * head_dim, n_rot, head_dim, cos_table, sin_table)
      end
      q_apply_ms = (Time.instant - q_apply_t0).total_milliseconds
      k_apply_t0 = Time.instant
      hp.n_head_kv(il).times do |h|
        fast_rope_neox_slice!(proj.k, h * head_dim, n_rot, head_dim, cos_table, sin_table)
      end
      k_apply_ms = (Time.instant - k_apply_t0).total_milliseconds
      apply_ms = (Time.instant - apply_t0).total_milliseconds
      {table_ms, apply_ms, q_apply_ms, k_apply_ms}
    end

    private def normalize_rope_attention_projection_timed!(proj : AttentionProjection,
                                                           lw : DiffusionGemmaLayerWeights,
                                                           hp : DiffusionGemmaHparams,
                                                           il : Int32,
                                                           pos : Int32,
                                                           rope_freqs : Array(Float32)? = nil) : {Float64, Float64, Float64, Float64}
      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)
      n_rot = hp.rope_dim_for_layer(il)
      base = hp.rope_freq_base_for_layer(il)
      freqs = hp.full_attention?(il) ? rope_freqs : nil

      raise ArgumentError.new("q projection size mismatch at layer #{il}") unless proj.q.size == n_head * head_dim
      raise ArgumentError.new("k projection size mismatch at layer #{il}") unless proj.k.size == n_head_kv * head_dim
      raise ArgumentError.new("v projection size mismatch at layer #{il}") unless proj.v.size == n_head_kv * head_dim

      table_t0 = Time.instant
      cos_table, sin_table = rope_tables(pos, n_rot, base, freqs)
      table_ms = (Time.instant - table_t0).total_milliseconds

      q_t0 = Time.instant
      n_head.times do |h|
        fast_rms_norm_rope_neox_slice!(proj.q, h * head_dim, head_dim, lw.attn_q_norm, hp.rms_eps, n_rot, cos_table, sin_table)
      end
      q_ms = (Time.instant - q_t0).total_milliseconds

      k_t0 = Time.instant
      n_head_kv.times do |h|
        fast_rms_norm_rope_neox_slice!(proj.k, h * head_dim, head_dim, lw.attn_k_norm, hp.rms_eps, n_rot, cos_table, sin_table)
      end
      k_ms = (Time.instant - k_t0).total_milliseconds

      v_t0 = Time.instant
      n_head_kv.times do |h|
        fast_rms_norm_plain_slice!(proj.v, h * head_dim, head_dim, hp.rms_eps)
      end
      v_ms = (Time.instant - v_t0).total_milliseconds
      {q_ms, k_ms, v_ms, table_ms}
    end

    def attention_context_unified(projections : Array(AttentionProjection),
                                  hp : DiffusionGemmaHparams,
                                  il : Int32,
                                  query_pos : Int32,
                                  mask : DiffusionGemmaAttentionMask) : Array(Float32)
      raise ArgumentError.new("projection count #{projections.size} != total tokens #{mask.total_tokens}") unless projections.size == mask.total_tokens
      raise ArgumentError.new("query_pos out of bounds") if query_pos < 0 || query_pos >= projections.size

      attention_context_from_keyspace(
        query: projections[query_pos],
        keyspace: projections,
        hp: hp,
        il: il,
        allowed: ->(key_pos : Int32) { mask.allow_unified?(query_pos, key_pos, hp.sliding_window?(il)) },
      )
    end

    def attention_context_decode(prompt_projections : Array(AttentionProjection),
                                 canvas_projections : Array(AttentionProjection),
                                 hp : DiffusionGemmaHparams,
                                 il : Int32,
                                 canvas_query_index : Int32,
                                 mask : DiffusionGemmaAttentionMask) : Array(Float32)
      raise ArgumentError.new("prompt projection count mismatch") unless prompt_projections.size == mask.prompt_len
      raise ArgumentError.new("canvas projection count mismatch") unless canvas_projections.size == mask.canvas_len
      raise ArgumentError.new("canvas_query_index out of bounds") if canvas_query_index < 0 || canvas_query_index >= canvas_projections.size

      keyspace = prompt_projections + canvas_projections
      low = hp.sliding_window?(il) ? mask.canvas_prompt_low : 0
      attention_context_from_range(
        query: canvas_projections[canvas_query_index],
        keyspace: keyspace,
        hp: hp,
        il: il,
        low: low,
        high: keyspace.size - 1,
      )
    end

    def attention_context_decode_timed(prompt_projections : Array(AttentionProjection),
                                       canvas_projections : Array(AttentionProjection),
                                       hp : DiffusionGemmaHparams,
                                       il : Int32,
                                       canvas_query_index : Int32,
                                       mask : DiffusionGemmaAttentionMask,
                                       prompt_metal_cache : PromptLayerMetalCache? = nil) : AttentionContextTiming
      raise ArgumentError.new("prompt projection count mismatch") unless prompt_projections.size == mask.prompt_len
      raise ArgumentError.new("canvas projection count mismatch") unless canvas_projections.size == mask.canvas_len
      raise ArgumentError.new("canvas_query_index out of bounds") if canvas_query_index < 0 || canvas_query_index >= canvas_projections.size

      if context_metal_enabled? && prompt_metal_cache
        if context = attention_context_decode_metal_resident(
             query: canvas_projections[canvas_query_index],
             cache: prompt_metal_cache.not_nil!,
             hp: hp,
             il: il,
             high: mask.total_tokens - 1,
             low: hp.sliding_window?(il) ? mask.canvas_prompt_low : 0,
           )
          return AttentionContextTiming.new(context, 0.0, 0.0, 0.0)
        end
      end

      keyspace = prompt_projections + canvas_projections
      low = hp.sliding_window?(il) ? mask.canvas_prompt_low : 0
      attention_context_from_range_timed(
        query: canvas_projections[canvas_query_index],
        keyspace: keyspace,
        hp: hp,
        il: il,
        low: low,
        high: keyspace.size - 1,
      )
    end

    def attention_context_prompt(projections : Array(AttentionProjection),
                                 hp : DiffusionGemmaHparams,
                                 il : Int32,
                                 query_pos : Int32,
                                 sliding_window : Int32) : Array(Float32)
      raise ArgumentError.new("prompt projections must not be empty") if projections.empty?
      raise ArgumentError.new("query_pos out of bounds") if query_pos < 0 || query_pos >= projections.size
      raise ArgumentError.new("sliding_window must be positive") unless sliding_window > 0

      low = hp.sliding_window?(il) ? Math.max(0, query_pos - sliding_window + 1) : 0
      attention_context_from_range(
        query: projections[query_pos],
        keyspace: projections,
        hp: hp,
        il: il,
        low: low,
        high: query_pos,
      )
    end

    def attention_output_project(weights : DiffusionGemmaWeights,
                                 il : Int32,
                                 context : Array(Float32)) : Array(Float32)
      hp = weights.hparams
      expected = hp.n_head * hp.head_dim_for_layer(il)
      raise ArgumentError.new("attention context size mismatch at layer #{il}: #{context.size} != #{expected}") unless context.size == expected

      prompt_projection_matmul(weights.layers[il].attn_output_qw, context, 1)
    end

    def attention_output_project_rows(weights : DiffusionGemmaWeights,
                                      il : Int32,
                                      context_rows : Array(Float32),
                                      row_count : Int32) : Array(Float32)
      hp = weights.hparams
      context_dim = hp.n_head * hp.head_dim_for_layer(il)
      expected = row_count * context_dim
      raise ArgumentError.new("attention_output_project_rows row_count must be positive") unless row_count > 0
      raise ArgumentError.new("attention context rows size mismatch at layer #{il}: #{context_rows.size} != #{expected}") unless context_rows.size == expected

      prompt_projection_matmul(weights.layers[il].attn_output_qw, context_rows, row_count)
    end

    def attention_residual_from_context(weights : DiffusionGemmaWeights,
                                        il : Int32,
                                        x : Array(Float32),
                                        context : Array(Float32)) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("attention residual input size mismatch") unless x.size == hp.n_embd

      projected = attention_output_project(weights, il, context)
      normed = Gemma4CPU.rms_norm(projected, lw.post_attention_norm, hp.rms_eps)
      Array(Float32).new(hp.n_embd) { |i| x[i] + normed[i] }
    end

    def attention_residual_from_context_rows(weights : DiffusionGemmaWeights,
                                             il : Int32,
                                             x_rows : Array(Float32),
                                             context_rows : Array(Float32),
                                             row_count : Int32) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      expected = row_count * hp.n_embd
      context_dim = hp.n_head * hp.head_dim_for_layer(il)
      context_expected = row_count * context_dim
      raise ArgumentError.new("attention_residual_from_context_rows row_count must be positive") unless row_count > 0
      raise ArgumentError.new("attention residual rows input size mismatch") unless x_rows.size == expected
      raise ArgumentError.new("attention residual context rows size mismatch") unless context_rows.size == context_expected

      if attention_residual_metal_rows_enabled?(row_count)
        if metal_rows = Gemma4Metal.attention_residual_rows_from_context(
             context_rows,
             x_rows,
             lw.attn_output_qw,
             lw.post_attention_norm,
             row_count,
             hp.n_embd,
             context_dim,
             hp.rms_eps,
           )
          return metal_rows
        end
      end

      result = attention_output_project_rows(weights, il, context_rows, row_count)
      row_count.times do |row|
        offset = row * hp.n_embd
        fast_rms_norm_slice!(result, offset, hp.n_embd, lw.post_attention_norm, hp.rms_eps)
        hp.n_embd.times { |i| result[offset + i] += x_rows[offset + i] }
      end
      result
    end

    def attention_residual_from_context_buffer(weights : DiffusionGemmaWeights,
                                               il : Int32,
                                               x_rows : Array(Float32),
                                               context_buf : ML::MetalBuffer,
                                               row_count : Int32) : Array(Float32)?
      hp = weights.hparams
      lw = weights.layers[il]
      expected = row_count * hp.n_embd
      context_dim = hp.n_head * hp.head_dim_for_layer(il)
      raise ArgumentError.new("attention_residual_from_context_buffer row_count must be positive") unless row_count > 0
      raise ArgumentError.new("attention residual rows input size mismatch") unless x_rows.size == expected
      raise ArgumentError.new("attention residual context buffer too small") if context_buf.size < row_count.to_i64 * context_dim * sizeof(Float32)

      Gemma4Metal.attention_residual_rows_from_context_buffer(
        context_buf,
        x_rows,
        lw.attn_output_qw,
        lw.post_attention_norm,
        row_count,
        hp.n_embd,
        context_dim,
        hp.rms_eps,
      )
    end

    # Dense shared FFN branch inside DiffusionGemma's Gemma4-MoE block. This is
    # only one branch of the oracle `dense + MoE -> post_ffw_norm -> residual`
    # path; expert routing is deliberately separate.
    def shared_dense_ffn(weights : DiffusionGemmaWeights,
                         il : Int32,
                         attn_out : Array(Float32)) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("shared_dense_ffn input size mismatch") unless attn_out.size == hp.n_embd

      ffn_in = Gemma4CPU.rms_norm(attn_out, lw.ffn_norm, hp.rms_eps)
      up = prompt_projection_matmul(lw.ffn_up_qw, ffn_in, 1)
      gate = prompt_projection_matmul(lw.ffn_gate_qw, ffn_in, 1)
      gate.size.times { |i| gate[i] = Gemma4CPU.gelu(gate[i]) * up[i] }
      down = prompt_projection_matmul(lw.ffn_down_qw, gate, 1)
      Gemma4CPU.rms_norm(down, lw.post_ffw_norm_1, hp.rms_eps)
    end

    def shared_dense_ffn_rows(weights : DiffusionGemmaWeights,
                              il : Int32,
                              attn_out_rows : Array(Float32),
                              row_count : Int32) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      expected = row_count * hp.n_embd
      raise ArgumentError.new("shared_dense_ffn_rows row_count must be positive") unless row_count > 0
      raise ArgumentError.new("shared_dense_ffn_rows input size mismatch") unless attn_out_rows.size == expected

      ffn_in_rows = attn_out_rows.dup
      row_count.times do |row|
        fast_rms_norm_slice!(ffn_in_rows, row * hp.n_embd, hp.n_embd, lw.ffn_norm, hp.rms_eps)
      end

      up_rows, gate_rows = if projected = prompt_projection_many_matmul([lw.ffn_up_qw, lw.ffn_gate_qw], ffn_in_rows, row_count)
                             {projected[0], projected[1]}
                           else
                             {
                               prompt_projection_matmul(lw.ffn_up_qw, ffn_in_rows, row_count),
                               prompt_projection_matmul(lw.ffn_gate_qw, ffn_in_rows, row_count),
                             }
                           end
      gate_rows.size.times { |i| gate_rows[i] = Gemma4CPU.gelu(gate_rows[i]) * up_rows[i] }
      down_rows = prompt_projection_matmul(lw.ffn_down_qw, gate_rows, row_count)
      row_count.times do |row|
        fast_rms_norm_slice!(down_rows, row * hp.n_embd, hp.n_embd, lw.post_ffw_norm_1, hp.rms_eps)
      end
      down_rows
    end

    def router_input(weights : DiffusionGemmaWeights,
                     il : Int32,
                     attn_out : Array(Float32)) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("router input size mismatch") unless attn_out.size == hp.n_embd
      raise ArgumentError.new("router scale size mismatch") unless lw.ffn_gate_inp_scale.size == hp.n_embd

      result = Gemma4CPU.rms_norm_plain(attn_out, hp.rms_eps)
      inv_sqrt_dim = (1.0_f64 / Math.sqrt(hp.n_embd.to_f64)).to_f32
      hp.n_embd.times { |i| result[i] *= inv_sqrt_dim * lw.ffn_gate_inp_scale[i] }
      result
    end

    def router_logits(weights : DiffusionGemmaWeights,
                      il : Int32,
                      attn_out : Array(Float32)) : Array(Float32)
      hp = weights.hparams
      logits = Gemma4CPU.matmul(weights.layers[il].ffn_gate_inp_qw, router_input(weights, il, attn_out))
      raise ArgumentError.new("router logits size mismatch") unless logits.size == hp.expert_count
      logits
    end

    def softmax(values : Array(Float32)) : Array(Float32)
      probs = values.dup
      Gemma4CPU.softmax_slice!(probs, 0, probs.size)
      probs
    end

    def top_k_experts(weights : Array(Float32), k : Int32) : Array(ExpertRoute)
      raise ArgumentError.new("top_k_experts k must be positive") unless k > 0
      raise ArgumentError.new("top_k_experts k exceeds weights") if k > weights.size

      best = [] of ExpertRoute
      weights.each_with_index do |weight, expert|
        route = ExpertRoute.new(expert.to_i32, weight)
        if best.size < k
          best << route
          sort_routes!(best)
        elsif better_route?(route, best[-1])
          best[-1] = route
          sort_routes!(best)
        end
      end
      best
    end

    def route_experts(weights : DiffusionGemmaWeights,
                      il : Int32,
                      attn_out : Array(Float32)) : Array(ExpertRoute)
      hp = weights.hparams
      probs = softmax(router_logits(weights, il, attn_out))
      top_k_experts(probs, hp.expert_used_count)
    end

    def moe_expert_output(weights : DiffusionGemmaWeights,
                          il : Int32,
                          expert : Int32,
                          ffn_in : Array(Float32)) : Array(Float32)
      hp = weights.hparams
      raise ArgumentError.new("expert id out of range") if expert < 0 || expert >= hp.expert_count
      raise ArgumentError.new("expert input size mismatch") unless ffn_in.size == hp.n_embd

      gate_up_qw = expert_gate_up_qw(weights.layers[il], hp, expert)
      gate_up = prompt_projection_matmul(gate_up_qw, ffn_in, 1)
      raise ArgumentError.new("expert gate_up size mismatch") unless gate_up.size == hp.expert_ff * 2

      hidden = Array(Float32).new(hp.expert_ff) do |i|
        Gemma4CPU.gelu(gate_up[i]) * gate_up[hp.expert_ff + i]
      end

      down = prompt_projection_matmul(expert_down_qw(weights.layers[il], hp, expert), hidden, 1)
      raise ArgumentError.new("expert down size mismatch") unless down.size == hp.n_embd

      scale = weights.layers[il].ffn_down_exps_scale[expert]
      down.size.times { |i| down[i] *= scale }
      down
    end

    def moe_ffn(weights : DiffusionGemmaWeights,
                il : Int32,
                attn_out : Array(Float32),
                routes : Array(ExpertRoute)? = nil) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("moe_ffn input size mismatch") unless attn_out.size == hp.n_embd
      raise ArgumentError.new("down expert scale size mismatch") unless lw.ffn_down_exps_scale.size == hp.expert_count

      selected = routes || route_experts(weights, il, attn_out)
      raise ArgumentError.new("moe_ffn routes must not be empty") if selected.empty?

      ffn_in = Gemma4CPU.rms_norm(attn_out, lw.pre_ffw_norm_2, hp.rms_eps)
      combined = Array(Float32).new(hp.n_embd, 0.0_f32)
      selected.each do |route|
        expert_out = moe_expert_output(weights, il, route.expert, ffn_in)
        hp.n_embd.times { |i| combined[i] += route.weight * expert_out[i] }
      end
      Gemma4CPU.rms_norm(combined, lw.post_ffw_norm_2, hp.rms_eps)
    end

    def moe_ffn_rows(weights : DiffusionGemmaWeights,
                     il : Int32,
                     attn_out_rows : Array(Float32),
                     row_count : Int32,
                     routes_by_row : Array(Array(ExpertRoute))? = nil) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      expected = row_count * hp.n_embd
      raise ArgumentError.new("moe_ffn_rows row_count must be positive") unless row_count > 0
      raise ArgumentError.new("moe_ffn_rows input size mismatch") unless attn_out_rows.size == expected
      if supplied_routes = routes_by_row
        raise ArgumentError.new("moe_ffn_rows route row count mismatch") unless supplied_routes.size == row_count
      end
      raise ArgumentError.new("down expert scale size mismatch") unless lw.ffn_down_exps_scale.size == hp.expert_count

      ffn_in_rows = attn_out_rows.dup
      row_count.times do |row|
        fast_rms_norm_slice!(ffn_in_rows, row * hp.n_embd, hp.n_embd, lw.pre_ffw_norm_2, hp.rms_eps)
      end

      result = Array(Float32).new(expected, 0.0_f32)
      row_count.times do |row|
        row_offset = row * hp.n_embd
        attn_out = attn_out_rows[row_offset, hp.n_embd]
        ffn_in = ffn_in_rows[row_offset, hp.n_embd]
        selected = if supplied_routes = routes_by_row
                     supplied_routes[row]
                   else
                     route_experts(weights, il, attn_out)
                   end
        raise ArgumentError.new("moe_ffn_rows routes must not be empty") if selected.empty?

        combined = Array(Float32).new(hp.n_embd, 0.0_f32)
        selected.each do |route|
          expert_out = moe_expert_output(weights, il, route.expert, ffn_in)
          hp.n_embd.times { |i| combined[i] += route.weight * expert_out[i] }
        end
        normed = Gemma4CPU.rms_norm(combined, lw.post_ffw_norm_2, hp.rms_eps)
        copy_row!(result, row, hp.n_embd, normed)
      end
      result
    end

    def moe_ffn_grouped_expert_rows(weights : DiffusionGemmaWeights,
                                    il : Int32,
                                    attn_out_rows : Array(Float32),
                                    row_count : Int32,
                                    routes_by_row : Array(Array(ExpertRoute))? = nil) : Array(Float32)
      moe_ffn_grouped_expert_rows_timed(weights, il, attn_out_rows, row_count, routes_by_row).rows
    end

    def grouped_moe_cognigraph_plan(routes_by_row : Array(Array(ExpertRoute)),
                                    hidden_dim : Int32,
                                    expert_ff : Int32,
                                    expert_count : Int32? = nil) : CogniGraphPlan
      raise ArgumentError.new("grouped_moe_cognigraph_plan rows must not be empty") if routes_by_row.empty?
      raise ArgumentError.new("grouped_moe_cognigraph_plan hidden_dim must be positive") unless hidden_dim > 0
      raise ArgumentError.new("grouped_moe_cognigraph_plan expert_ff must be positive") unless expert_ff > 0

      assignments_by_expert = Hash(Int32, Array(Int32)).new do |hash, expert|
        hash[expert] = [] of Int32
      end
      route_slots = 0
      routes_by_row.each_with_index do |routes, row|
        raise ArgumentError.new("grouped_moe_cognigraph_plan routes must not be empty") if routes.empty?
        routes.each do |route|
          raise ArgumentError.new("grouped_moe_cognigraph_plan expert id out of range") if route.expert < 0
          if max_experts = expert_count
            raise ArgumentError.new("grouped_moe_cognigraph_plan expert id out of range") unless route.expert < max_experts
          end
          assignments_by_expert[route.expert] << row
          route_slots += 1
        end
      end

      ops = [] of CogniGraphPlanOp
      assignments_by_expert.keys.sort.each do |expert|
        rows = assignments_by_expert[expert]
        batch = rows.size
        gather_bindings = rows.map do |row|
          CogniGraphPlanBinding.new(
            buffer: "ffn_in_rows",
            access: CogniGraphPlanAccess::Read,
            offset: row.to_i64 * hidden_dim,
            length: hidden_dim.to_i64,
            partition: row,
          )
        end
        gather_bindings << CogniGraphPlanBinding.new(
          buffer: "expert_inputs",
          access: CogniGraphPlanAccess::Write,
          offset: expert.to_i64 * hidden_dim,
          length: batch.to_i64 * hidden_dim,
          partition: expert,
        )
        ops << CogniGraphPlanOp.new("expert#{expert}.gather", gather_bindings)

        ops << CogniGraphPlanOp.new("expert#{expert}.gate_up", [
          CogniGraphPlanBinding.new("expert_inputs", CogniGraphPlanAccess::Read, expert.to_i64 * hidden_dim, batch.to_i64 * hidden_dim, expert),
          CogniGraphPlanBinding.new("gate_up_rows", CogniGraphPlanAccess::Write, expert.to_i64 * expert_ff * 2_i64, batch.to_i64 * expert_ff * 2_i64, expert),
        ])
        ops << CogniGraphPlanOp.new("expert#{expert}.activation", [
          CogniGraphPlanBinding.new("gate_up_rows", CogniGraphPlanAccess::Read, expert.to_i64 * expert_ff * 2_i64, batch.to_i64 * expert_ff * 2_i64, expert),
          CogniGraphPlanBinding.new("hidden_rows", CogniGraphPlanAccess::Write, expert.to_i64 * expert_ff, batch.to_i64 * expert_ff, expert),
        ])
        ops << CogniGraphPlanOp.new("expert#{expert}.down", [
          CogniGraphPlanBinding.new("hidden_rows", CogniGraphPlanAccess::Read, expert.to_i64 * expert_ff, batch.to_i64 * expert_ff, expert),
          CogniGraphPlanBinding.new("expert_outputs", CogniGraphPlanAccess::Write, 0_i64, -1_i64, expert),
        ])
      end

      routes_by_row.each_with_index do |routes, row|
        bindings = routes.map do |route|
          CogniGraphPlanBinding.new(
            buffer: "expert_outputs",
            access: CogniGraphPlanAccess::Read,
            offset: 0_i64,
            length: -1_i64,
            partition: route.expert,
          )
        end
        bindings << CogniGraphPlanBinding.new(
          buffer: "moe_result_rows",
          access: CogniGraphPlanAccess::Write,
          offset: row.to_i64 * hidden_dim,
          length: hidden_dim.to_i64,
          partition: row,
        )
        ops << CogniGraphPlanOp.new("row#{row}.combine_norm", bindings)
      end

      compile_cognigraph_plan(ops, assignments_by_expert.size, route_slots, routes_by_row.size)
    end

    def grouped_moe_resident_matmul_graph_stats(weights : DiffusionGemmaWeights,
                                                il : Int32,
                                                routes_by_row : Array(Array(ExpertRoute))) : GroupedMoeResidentGraphStats?
      return nil unless Qwen35Metal.available?

      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("grouped_moe_resident_matmul_graph_stats rows must not be empty") if routes_by_row.empty?
      raise ArgumentError.new("down expert scale size mismatch") unless lw.ffn_down_exps_scale.size == hp.expert_count

      assignments_by_expert = Hash(Int32, Int32).new(0)
      route_slots = 0
      routes_by_row.each do |routes|
        raise ArgumentError.new("grouped_moe_resident_matmul_graph_stats routes must not be empty") if routes.empty?
        routes.each do |route|
          raise ArgumentError.new("grouped_moe_resident_matmul_graph_stats expert id out of range") if route.expert < 0 || route.expert >= hp.expert_count
          assignments_by_expert[route.expert] += 1
          route_slots += 1
        end
      end

      graph = ML::Metal::ComputeGraph.new
      enc = ML::Metal::GraphEncoder.new(graph)
      assignments_by_expert.keys.sort.each do |expert|
        batch = assignments_by_expert[expert]
        ffn_in_buf = ML::MetalBuffer.new(batch.to_i64 * hp.n_embd * sizeof(Float32))
        gate_buf = ML::MetalBuffer.new(batch.to_i64 * hp.expert_ff * sizeof(Float32))
        up_buf = ML::MetalBuffer.new(batch.to_i64 * hp.expert_ff * sizeof(Float32))
        hidden_buf = ML::MetalBuffer.new(batch.to_i64 * hp.expert_ff * sizeof(Float32))
        down_buf = ML::MetalBuffer.new(batch.to_i64 * hp.n_embd * sizeof(Float32))

        gate_qw = expert_gate_qw(lw, hp, expert)
        up_qw = expert_up_qw(lw, hp, expert)
        down_qw = expert_down_qw(lw, hp, expert)
        return nil unless Qwen35Metal.encode_matmul_many_to_buffers(enc, [gate_qw, up_qw], ffn_in_buf, [gate_buf, up_buf], batch)
        return nil unless Gemma4Metal.encode_gelu_mul_to_buffer(enc, gate_buf, up_buf, hidden_buf, batch * hp.expert_ff)
        return nil unless Qwen35Metal.encode_matmul_to_buffer(enc, down_qw, hidden_buf, down_buf, batch)
      end

      graph.compile!
      stats = graph.stats
      GroupedMoeResidentGraphStats.new(
        n_ops: stats.n_ops,
        n_waves: stats.n_waves,
        n_barriers: stats.n_barriers,
        max_wave_width: stats.max_wave_width,
        active_experts: assignments_by_expert.size,
        route_slots: route_slots,
        row_count: routes_by_row.size,
      )
    end

    def moe_ffn_grouped_expert_rows_timed(weights : DiffusionGemmaWeights,
                                          il : Int32,
                                          attn_out_rows : Array(Float32),
                                          row_count : Int32,
                                          routes_by_row : Array(Array(ExpertRoute))? = nil) : GroupedMoeRowsTiming
      hp = weights.hparams
      lw = weights.layers[il]
      expected = row_count * hp.n_embd
      raise ArgumentError.new("moe_ffn_grouped_expert_rows row_count must be positive") unless row_count > 0
      raise ArgumentError.new("moe_ffn_grouped_expert_rows input size mismatch") unless attn_out_rows.size == expected
      if supplied_routes = routes_by_row
        raise ArgumentError.new("moe_ffn_grouped_expert_rows route row count mismatch") unless supplied_routes.size == row_count
      end
      raise ArgumentError.new("down expert scale size mismatch") unless lw.ffn_down_exps_scale.size == hp.expert_count

      prep_ms = 0.0
      gate_up_ms = 0.0
      activation_ms = 0.0
      down_ms = 0.0
      scatter_combine_norm_ms = 0.0

      prep_t0 = Time.instant
      ffn_in_rows = attn_out_rows.dup
      row_count.times do |row|
        fast_rms_norm_slice!(ffn_in_rows, row * hp.n_embd, hp.n_embd, lw.pre_ffw_norm_2, hp.rms_eps)
      end

      selected_by_row = Array(Array(ExpertRoute)).new(row_count) do |row|
        if supplied_routes = routes_by_row
          supplied_routes[row]
        else
          row_offset = row * hp.n_embd
          route_experts(weights, il, attn_out_rows[row_offset, hp.n_embd])
        end
      end
      selected_by_row.each do |selected|
        raise ArgumentError.new("moe_ffn_grouped_expert_rows routes must not be empty") if selected.empty?
      end

      if grouped_moe_cognigraph_plan_enabled?
        plan = grouped_moe_cognigraph_plan(selected_by_row, hp.n_embd, hp.expert_ff, hp.expert_count)
        emit_grouped_moe_cognigraph_plan(il, row_count, plan)
      end

      assignments_by_expert = Hash(Int32, Array(Tuple(Int32, Int32))).new do |hash, expert|
        hash[expert] = [] of Tuple(Int32, Int32)
      end
      route_offsets_by_row = Array(Int32).new(row_count, 0)
      route_slot_count = 0
      selected_by_row.each_with_index do |selected, row|
        route_offsets_by_row[row] = route_slot_count
        route_slot_count += selected.size
        selected.each_with_index do |route, route_index|
          assignments_by_expert[route.expert] << {row, route_index}
        end
      end
      expert_outputs_by_route_slot = Array(Float32).new(route_slot_count * hp.n_embd, 0.0_f32)
      prep_ms += (Time.instant - prep_t0).total_milliseconds

      assignments_by_expert.each do |expert, assignments|
        batch = assignments.size
        gather_t0 = Time.instant
        ffn_inputs = Array(Float32).new(batch * hp.n_embd, 0.0_f32)
        assignments.each_with_index do |assignment, batch_row|
          row = assignment[0]
          src_offset = row * hp.n_embd
          dst_offset = batch_row * hp.n_embd
          (ffn_inputs.to_unsafe + dst_offset).copy_from(ffn_in_rows.to_unsafe + src_offset, hp.n_embd)
        end
        prep_ms += (Time.instant - gather_t0).total_milliseconds

        gate_up_qw = expert_gate_up_qw(lw, hp, expert)
        gate_up_t0 = Time.instant
        gate_up_rows = prompt_projection_matmul(gate_up_qw, ffn_inputs, batch)
        gate_up_ms += (Time.instant - gate_up_t0).total_milliseconds
        raise ArgumentError.new("expert gate_up rows size mismatch") unless gate_up_rows.size == batch * hp.expert_ff * 2

        activation_t0 = Time.instant
        hidden_rows = Array(Float32).new(batch * hp.expert_ff, 0.0_f32)
        batch.times do |batch_row|
          gate_up_offset = batch_row * hp.expert_ff * 2
          hidden_offset = batch_row * hp.expert_ff
          hp.expert_ff.times do |i|
            hidden_rows[hidden_offset + i] = Gemma4CPU.gelu(gate_up_rows[gate_up_offset + i]) *
                                             gate_up_rows[gate_up_offset + hp.expert_ff + i]
          end
        end
        activation_ms += (Time.instant - activation_t0).total_milliseconds

        down_t0 = Time.instant
        down_rows = prompt_projection_matmul(expert_down_qw(lw, hp, expert), hidden_rows, batch)
        down_ms += (Time.instant - down_t0).total_milliseconds
        raise ArgumentError.new("expert down rows size mismatch") unless down_rows.size == batch * hp.n_embd

        scatter_t0 = Time.instant
        scale = lw.ffn_down_exps_scale[expert]
        assignments.each_with_index do |assignment, batch_row|
          row = assignment[0]
          route_index = assignment[1]
          src_offset = batch_row * hp.n_embd
          dst_offset = (route_offsets_by_row[row] + route_index) * hp.n_embd
          hp.n_embd.times { |i| expert_outputs_by_route_slot[dst_offset + i] = down_rows[src_offset + i] * scale }
        end
        scatter_combine_norm_ms += (Time.instant - scatter_t0).total_milliseconds
      end

      combine_t0 = Time.instant
      result = Array(Float32).new(expected, 0.0_f32)
      row_count.times do |row|
        combined = Array(Float32).new(hp.n_embd, 0.0_f32)
        selected_by_row[row].each_with_index do |route, route_index|
          src_offset = (route_offsets_by_row[row] + route_index) * hp.n_embd
          hp.n_embd.times { |i| combined[i] += route.weight * expert_outputs_by_route_slot[src_offset + i] }
        end
        normed = Gemma4CPU.rms_norm(combined, lw.post_ffw_norm_2, hp.rms_eps)
        copy_row!(result, row, hp.n_embd, normed)
      end
      scatter_combine_norm_ms += (Time.instant - combine_t0).total_milliseconds
      GroupedMoeRowsTiming.new(result, prep_ms, gate_up_ms, activation_ms, down_ms, scatter_combine_norm_ms)
    end

    def ffn_residual(weights : DiffusionGemmaWeights,
                     il : Int32,
                     attn_out : Array(Float32),
                     routes : Array(ExpertRoute)? = nil) : Array(Float32)
      shared = shared_dense_ffn(weights, il, attn_out)
      moe = moe_ffn(weights, il, attn_out, routes)
      ffn_residual_from_parts(weights, il, attn_out, shared, moe)
    end

    def layer_output_from_context(weights : DiffusionGemmaWeights,
                                  il : Int32,
                                  x : Array(Float32),
                                  context : Array(Float32),
                                  canvas : Bool,
                                  routes : Array(ExpertRoute)? = nil) : Array(Float32)
      attn_out = attention_residual_from_context(weights, il, x, context)
      ffn_out = ffn_residual(weights, il, attn_out, routes)
      scale_layer_output(weights, il, ffn_out, canvas)
    end

    def scale_layer_output(weights : DiffusionGemmaWeights,
                           il : Int32,
                           x : Array(Float32),
                           canvas : Bool) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("layer scale input size mismatch") unless x.size == hp.n_embd
      scale = canvas ? lw.layer_output_scale[0] : lw.encoder_layer_output_scale[0]
      Array(Float32).new(hp.n_embd) { |i| x[i] * scale }
    end

    def layer_forward_prompt_rows(weights : DiffusionGemmaWeights,
                                  il : Int32,
                                  prompt_rows : Array(Float32),
                                  mask : DiffusionGemmaAttentionMask,
                                  routes_by_prompt_row : Array(Array(ExpertRoute))? = nil) : Array(Float32)
      projections = prompt_attention_projections(weights, il, prompt_rows, mask)
      layer_forward_prompt_rows_with_projections(weights, il, prompt_rows, projections, mask, routes_by_prompt_row)
    end

    def layer_forward_prompt_rows_with_projections(weights : DiffusionGemmaWeights,
                                                   il : Int32,
                                                   prompt_rows : Array(Float32),
                                                   prompt_projections : Array(AttentionProjection),
                                                   mask : DiffusionGemmaAttentionMask,
                                                   routes_by_prompt_row : Array(Array(ExpertRoute))? = nil) : Array(Float32)
      hp = weights.hparams
      prompt_size = mask.prompt_len * hp.n_embd
      raise ArgumentError.new("prompt rows size mismatch: #{prompt_rows.size} != #{prompt_size}") unless prompt_rows.size == prompt_size
      raise ArgumentError.new("prompt projection count mismatch") unless prompt_projections.size == mask.prompt_len
      if supplied_routes = routes_by_prompt_row
        raise ArgumentError.new("routes_by_prompt_row size mismatch: #{supplied_routes.size} != #{mask.prompt_len}") unless supplied_routes.size == mask.prompt_len
      end

      if prompt_materialize_batch_rows_enabled?(mask.prompt_len) && mask.prompt_len > 1
        return layer_forward_prompt_rows_with_projections_batched(
          weights,
          il,
          prompt_rows,
          prompt_projections,
          mask,
          routes_by_prompt_row,
        )
      end

      result = Array(Float32).new(prompt_size, 0.0_f32)
      mask.prompt_len.times do |pos|
        x = prompt_rows[pos * hp.n_embd, hp.n_embd]
        context = attention_context_prompt(prompt_projections, hp, il, query_pos: pos, sliding_window: mask.sliding_window)
        layer_row = if supplied_routes = routes_by_prompt_row
                      layer_output_from_context(weights, il, x, context, canvas: false, routes: supplied_routes[pos])
                    else
                      layer_output_from_context(weights, il, x, context, canvas: false)
                    end
        copy_row!(result, pos, hp.n_embd, layer_row)
      end
      result
    end

    def layer_forward_prompt_rows_with_projections_batched(weights : DiffusionGemmaWeights,
                                                           il : Int32,
                                                           prompt_rows : Array(Float32),
                                                           prompt_projections : Array(AttentionProjection),
                                                           mask : DiffusionGemmaAttentionMask,
                                                           routes_by_prompt_row : Array(Array(ExpertRoute))? = nil) : Array(Float32)
      hp = weights.hparams
      prompt_size = mask.prompt_len * hp.n_embd
      q_context_dim = hp.n_head * hp.head_dim_for_layer(il)
      raise ArgumentError.new("prompt rows size mismatch: #{prompt_rows.size} != #{prompt_size}") unless prompt_rows.size == prompt_size
      raise ArgumentError.new("prompt projection count mismatch") unless prompt_projections.size == mask.prompt_len
      if supplied_routes = routes_by_prompt_row
        raise ArgumentError.new("routes_by_prompt_row size mismatch: #{supplied_routes.size} != #{mask.prompt_len}") unless supplied_routes.size == mask.prompt_len
      end

      context_rows = Array(Float32).new(mask.prompt_len * q_context_dim, 0.0_f32)
      mask.prompt_len.times do |pos|
        context = attention_context_prompt(prompt_projections, hp, il, query_pos: pos, sliding_window: mask.sliding_window)
        copy_row!(context_rows, pos, q_context_dim, context)
      end

      attn_out_rows = attention_residual_from_context_rows(weights, il, prompt_rows, context_rows, mask.prompt_len)
      shared_rows = shared_dense_ffn_rows(weights, il, attn_out_rows, mask.prompt_len)
      moe_rows = if prompt_materialize_grouped_moe_enabled?
                   moe_ffn_grouped_expert_rows(weights, il, attn_out_rows, mask.prompt_len, routes_by_prompt_row)
                 else
                   moe_ffn_rows(weights, il, attn_out_rows, mask.prompt_len, routes_by_prompt_row)
                 end
      ffn_residual_from_parts_rows(weights, il, attn_out_rows, shared_rows, moe_rows, mask.prompt_len, canvas: false)
    end

    def build_prompt_layer_cache(weights : DiffusionGemmaWeights,
                                 prompt_rows : Array(Float32),
                                 mask : DiffusionGemmaAttentionMask,
                                 max_layers : Int32 = weights.hparams.n_layer,
                                 routes_by_layer_by_prompt_row : Array(Array(Array(ExpertRoute)))? = nil,
                                 materialize_final_rows : Bool = true) : PromptLayerCache
      hp = weights.hparams
      prompt_size = mask.prompt_len * hp.n_embd
      raise ArgumentError.new("prompt rows size mismatch: #{prompt_rows.size} != #{prompt_size}") unless prompt_rows.size == prompt_size
      raise ArgumentError.new("max_layers out of range") if max_layers <= 0 || max_layers > hp.n_layer
      if supplied_routes = routes_by_layer_by_prompt_row
        raise ArgumentError.new("routes_by_layer_by_prompt_row size mismatch") unless supplied_routes.size == max_layers
      end

      rows = prompt_rows.dup
      projections_by_layer = [] of Array(AttentionProjection)
      projection_ms_by_layer = [] of Float64
      projection_norm_ms_by_layer = [] of Float64
      projection_matmul_ms_by_layer = [] of Float64
      projection_assemble_ms_by_layer = [] of Float64
      projection_copy_ms_by_layer = [] of Float64
      projection_head_norm_ms_by_layer = [] of Float64
      projection_q_norm_ms_by_layer = [] of Float64
      projection_k_norm_ms_by_layer = [] of Float64
      projection_v_norm_ms_by_layer = [] of Float64
      projection_rope_ms_by_layer = [] of Float64
      projection_rope_table_ms_by_layer = [] of Float64
      projection_rope_apply_ms_by_layer = [] of Float64
      projection_rope_q_apply_ms_by_layer = [] of Float64
      projection_rope_k_apply_ms_by_layer = [] of Float64
      materialize_ms_by_layer = [] of Float64
      metal_cache_by_layer = [] of PromptLayerMetalCache?
      max_layers.times do |il|
        projection_t0 = Time.instant
        timed_projections = prompt_attention_projections_timed(weights, il, rows, mask)
        projections = timed_projections.projections
        projection_ms_by_layer << (Time.instant - projection_t0).total_milliseconds
        projection_norm_ms_by_layer << timed_projections.norm_ms
        projection_matmul_ms_by_layer << timed_projections.matmul_ms
        projection_assemble_ms_by_layer << timed_projections.assemble_ms
        projection_copy_ms_by_layer << timed_projections.copy_ms
        projection_head_norm_ms_by_layer << timed_projections.head_norm_ms
        projection_q_norm_ms_by_layer << timed_projections.q_norm_ms
        projection_k_norm_ms_by_layer << timed_projections.k_norm_ms
        projection_v_norm_ms_by_layer << timed_projections.v_norm_ms
        projection_rope_ms_by_layer << timed_projections.rope_ms
        projection_rope_table_ms_by_layer << timed_projections.rope_table_ms
        projection_rope_apply_ms_by_layer << timed_projections.rope_apply_ms
        projection_rope_q_apply_ms_by_layer << timed_projections.rope_q_apply_ms
        projection_rope_k_apply_ms_by_layer << timed_projections.rope_k_apply_ms
        projections_by_layer << projections
        if context_metal_enabled? && Gemma4Metal.available?
          q_dim = hp.n_head * hp.head_dim_for_layer(il)
          kv_dim = hp.n_head_kv(il) * hp.head_dim_for_layer(il)
          metal_cache_by_layer << PromptLayerMetalCache.new(projections, mask.prompt_len, mask.canvas_len, q_dim, kv_dim)
        else
          metal_cache_by_layer << nil
        end
        break if !materialize_final_rows && il == max_layers - 1

        routes = routes_by_layer_by_prompt_row ? routes_by_layer_by_prompt_row.not_nil![il] : nil
        materialize_t0 = Time.instant
        rows = layer_forward_prompt_rows_with_projections(weights, il, rows, projections, mask, routes)
        materialize_ms_by_layer << (Time.instant - materialize_t0).total_milliseconds
      end

      PromptLayerCache.new(
        rows,
        projections_by_layer,
        projection_ms_by_layer,
        projection_norm_ms_by_layer,
        projection_matmul_ms_by_layer,
        projection_assemble_ms_by_layer,
        projection_copy_ms_by_layer,
        projection_head_norm_ms_by_layer,
        projection_q_norm_ms_by_layer,
        projection_k_norm_ms_by_layer,
        projection_v_norm_ms_by_layer,
        projection_rope_ms_by_layer,
        projection_rope_table_ms_by_layer,
        projection_rope_apply_ms_by_layer,
        projection_rope_q_apply_ms_by_layer,
        projection_rope_k_apply_ms_by_layer,
        materialize_ms_by_layer,
        metal_cache_by_layer,
      )
    end

    def decode_canvas_rows_with_prompt_cache(weights : DiffusionGemmaWeights,
                                             canvas_rows : Array(Float32),
                                             mask : DiffusionGemmaAttentionMask,
                                             prompt_cache : PromptLayerCache,
                                             max_layers : Int32 = prompt_cache.layers,
                                             routes_by_layer_by_canvas_row : Array(Array(Array(ExpertRoute)))? = nil) : Array(Float32)
      decode_canvas_rows_with_prompt_cache_timed(
        weights: weights,
        canvas_rows: canvas_rows,
        mask: mask,
        prompt_cache: prompt_cache,
        max_layers: max_layers,
        routes_by_layer_by_canvas_row: routes_by_layer_by_canvas_row,
      ).rows
    end

    def decode_canvas_rows_with_prompt_cache_timed(weights : DiffusionGemmaWeights,
                                                   canvas_rows : Array(Float32),
                                                   mask : DiffusionGemmaAttentionMask,
                                                   prompt_cache : PromptLayerCache,
                                                   max_layers : Int32 = prompt_cache.layers,
                                                   routes_by_layer_by_canvas_row : Array(Array(Array(ExpertRoute)))? = nil) : DecodeCanvasRowsTiming
      hp = weights.hparams
      canvas_size = mask.canvas_len * hp.n_embd
      prompt_size = mask.prompt_len * hp.n_embd
      raise ArgumentError.new("canvas rows size mismatch: #{canvas_rows.size} != #{canvas_size}") unless canvas_rows.size == canvas_size
      raise ArgumentError.new("prompt cache final rows size mismatch") unless prompt_cache.final_rows.size == prompt_size
      raise ArgumentError.new("max_layers out of range") if max_layers <= 0 || max_layers > hp.n_layer
      raise ArgumentError.new("prompt cache has fewer layers than requested") if prompt_cache.layers < max_layers
      if supplied_routes = routes_by_layer_by_canvas_row
        raise ArgumentError.new("routes_by_layer_by_canvas_row size mismatch") unless supplied_routes.size == max_layers
      end

      rows = canvas_rows.dup
      qkv_ms = 0.0
      context_ms = 0.0
      context_score_ms = 0.0
      context_softmax_ms = 0.0
      context_value_ms = 0.0
      attention_residual_context_buffer_hit = false
      attention_out_ms = 0.0
      shared_ffn_ms = 0.0
      moe_ffn_ms = 0.0
      moe_grouped_prep_ms = 0.0
      moe_grouped_gate_up_ms = 0.0
      moe_grouped_activation_ms = 0.0
      moe_grouped_down_ms = 0.0
      moe_grouped_scatter_combine_norm_ms = 0.0
      combine_scale_ms = 0.0
      max_layers.times do |il|
        routes = routes_by_layer_by_canvas_row ? routes_by_layer_by_canvas_row.not_nil![il] : nil
        timed = layer_forward_decode_canvas_rows_with_prompt_projections_timed(
          weights: weights,
          il: il,
          prompt_projections: prompt_cache.projections_by_layer[il],
          canvas_rows: rows,
          mask: mask,
          prompt_metal_cache: prompt_cache.metal_cache_by_layer[il]?,
          routes_by_canvas_row: routes,
        )
        rows = timed.rows
        qkv_ms += timed.qkv_ms
        context_ms += timed.context_ms
        context_score_ms += timed.context_score_ms
        context_softmax_ms += timed.context_softmax_ms
        context_value_ms += timed.context_value_ms
        attention_residual_context_buffer_hit ||= timed.attention_residual_context_buffer
        attention_out_ms += timed.attention_out_ms
        shared_ffn_ms += timed.shared_ffn_ms
        moe_ffn_ms += timed.moe_ffn_ms
        moe_grouped_prep_ms += timed.moe_grouped_prep_ms
        moe_grouped_gate_up_ms += timed.moe_grouped_gate_up_ms
        moe_grouped_activation_ms += timed.moe_grouped_activation_ms
        moe_grouped_down_ms += timed.moe_grouped_down_ms
        moe_grouped_scatter_combine_norm_ms += timed.moe_grouped_scatter_combine_norm_ms
        combine_scale_ms += timed.combine_scale_ms
      end
      DecodeCanvasRowsTiming.new(rows, qkv_ms, context_ms, attention_out_ms, shared_ffn_ms, moe_ffn_ms, combine_scale_ms, context_score_ms, context_softmax_ms, context_value_ms, attention_residual_context_buffer_hit, moe_grouped_prep_ms, moe_grouped_gate_up_ms, moe_grouped_activation_ms, moe_grouped_down_ms, moe_grouped_scatter_combine_norm_ms)
    end

    def layer_forward_unified_rows(weights : DiffusionGemmaWeights,
                                   il : Int32,
                                   rows : Array(Float32),
                                   mask : DiffusionGemmaAttentionMask,
                                   routes_by_row : Array(Array(ExpertRoute))? = nil) : Array(Float32)
      hp = weights.hparams
      total_tokens = mask.total_tokens
      expected_size = total_tokens * hp.n_embd
      raise ArgumentError.new("layer rows size mismatch: #{rows.size} != #{expected_size}") unless rows.size == expected_size
      if supplied_routes = routes_by_row
        raise ArgumentError.new("routes_by_row size mismatch: #{supplied_routes.size} != #{total_tokens}") unless supplied_routes.size == total_tokens
      end

      projections = Array(AttentionProjection).new(total_tokens) do |pos|
        x = rows[pos * hp.n_embd, hp.n_embd]
        attention_project_normed(weights, il, x, pos)
      end

      result = Array(Float32).new(expected_size, 0.0_f32)
      total_tokens.times do |pos|
        x = rows[pos * hp.n_embd, hp.n_embd]
        context = attention_context_unified(projections, hp, il, query_pos: pos, mask: mask)
        canvas = pos >= mask.prompt_len
        layer_row = if supplied_routes = routes_by_row
                      layer_output_from_context(weights, il, x, context, canvas: canvas, routes: supplied_routes[pos])
                    else
                      layer_output_from_context(weights, il, x, context, canvas: canvas)
                    end
        copy_row!(result, pos, hp.n_embd, layer_row)
      end
      result
    end

    def layer_forward_decode_canvas_rows(weights : DiffusionGemmaWeights,
                                         il : Int32,
                                         prompt_rows : Array(Float32),
                                         canvas_rows : Array(Float32),
                                         mask : DiffusionGemmaAttentionMask,
                                         routes_by_canvas_row : Array(Array(ExpertRoute))? = nil) : Array(Float32)
      prompt_projections = prompt_attention_projections(weights, il, prompt_rows, mask)
      layer_forward_decode_canvas_rows_with_prompt_projections(
        weights: weights,
        il: il,
        prompt_projections: prompt_projections,
        canvas_rows: canvas_rows,
        mask: mask,
        routes_by_canvas_row: routes_by_canvas_row,
      )
    end

    def layer_forward_decode_canvas_rows_with_prompt_projections(weights : DiffusionGemmaWeights,
                                                                 il : Int32,
                                                                 prompt_projections : Array(AttentionProjection),
                                                                 canvas_rows : Array(Float32),
                                                                 mask : DiffusionGemmaAttentionMask,
                                                                 routes_by_canvas_row : Array(Array(ExpertRoute))? = nil) : Array(Float32)
      layer_forward_decode_canvas_rows_with_prompt_projections_timed(
        weights: weights,
        il: il,
        prompt_projections: prompt_projections,
        canvas_rows: canvas_rows,
        mask: mask,
        routes_by_canvas_row: routes_by_canvas_row,
      ).rows
    end

    def layer_forward_decode_canvas_rows_with_prompt_projections_timed(weights : DiffusionGemmaWeights,
                                                                       il : Int32,
                                                                       prompt_projections : Array(AttentionProjection),
                                                                       canvas_rows : Array(Float32),
                                                                       mask : DiffusionGemmaAttentionMask,
                                                                       prompt_metal_cache : PromptLayerMetalCache? = nil,
                                                                       routes_by_canvas_row : Array(Array(ExpertRoute))? = nil) : DecodeCanvasRowsTiming
      hp = weights.hparams
      lw = weights.layers[il]
      canvas_size = mask.canvas_len * hp.n_embd
      q_context_dim = hp.n_head * hp.head_dim_for_layer(il)
      raise ArgumentError.new("prompt projection count mismatch") unless prompt_projections.size == mask.prompt_len
      raise ArgumentError.new("canvas rows size mismatch: #{canvas_rows.size} != #{canvas_size}") unless canvas_rows.size == canvas_size
      if supplied_routes = routes_by_canvas_row
        raise ArgumentError.new("routes_by_canvas_row size mismatch: #{supplied_routes.size} != #{mask.canvas_len}") unless supplied_routes.size == mask.canvas_len
      end

      qkv_t0 = Time.instant
      canvas_projections = attention_projections_timed(
        weights,
        il,
        canvas_rows,
        mask.canvas_len,
        mask.prompt_len,
        "canvas",
      ).projections
      qkv_ms = (Time.instant - qkv_t0).total_milliseconds
      if prompt_metal_cache
        prompt_metal_cache.not_nil!.write_canvas!(canvas_projections)
      end
      context_ms = 0.0
      context_score_ms = 0.0
      context_softmax_ms = 0.0
      context_value_ms = 0.0
      batch_context_t0 = Time.instant
      batched_context_buf = if prompt_metal_cache &&
                               context_metal_batch_rows_enabled? &&
                               attention_out_batch_rows_enabled?(mask.canvas_len) &&
                               attention_residual_context_buffer_enabled?(mask.canvas_len)
                              attention_context_decode_batch_metal_resident_buffer(
                                canvas_projections,
                                prompt_metal_cache.not_nil!,
                                hp,
                                il,
                                low: hp.sliding_window?(il) ? mask.canvas_prompt_low : 0,
                                high: mask.total_tokens - 1,
                              )
                            else
                              nil
                            end
      batched_context = if batched_context_buf.nil? && prompt_metal_cache && context_metal_batch_rows_enabled?
                          attention_context_decode_batch_metal_resident(
                            canvas_projections,
                            prompt_metal_cache.not_nil!,
                            hp,
                            il,
                            low: hp.sliding_window?(il) ? mask.canvas_prompt_low : 0,
                            high: mask.total_tokens - 1,
                          )
                        else
                          nil
                        end
      context_ms += (Time.instant - batch_context_t0).total_milliseconds if batched_context
      context_ms += (Time.instant - batch_context_t0).total_milliseconds if batched_context_buf
      attention_residual_context_buffer_hit = false
      attention_out_ms = 0.0
      shared_ffn_ms = 0.0
      moe_ffn_ms = 0.0
      moe_grouped_prep_ms = 0.0
      moe_grouped_gate_up_ms = 0.0
      moe_grouped_activation_ms = 0.0
      moe_grouped_down_ms = 0.0
      moe_grouped_scatter_combine_norm_ms = 0.0
      combine_scale_ms = 0.0
      result = Array(Float32).new(canvas_size, 0.0_f32)
      attn_out_rows = Array(Float32).new(canvas_size, 0.0_f32)
      context_rows = if attention_out_batch_rows_enabled?(mask.canvas_len) && mask.canvas_len > 1
                       batched_context || (batched_context_buf ? nil : Array(Float32).new(mask.canvas_len * q_context_dim, 0.0_f32))
                     else
                       nil
                     end
      skip_context_row_loop = !batched_context_buf.nil? || (!batched_context.nil? && !context_rows.nil?)
      unless skip_context_row_loop
        mask.canvas_len.times do |canvas_pos|
          x = canvas_rows[canvas_pos * hp.n_embd, hp.n_embd]
          context_t0 = Time.instant
          context_timing = nil.as(AttentionContextTiming?)
          context = if batched = batched_context
                      batched[canvas_pos * q_context_dim, q_context_dim]
                    else
                      context_timing = attention_context_decode_timed(prompt_projections, canvas_projections, hp, il, canvas_query_index: canvas_pos, mask: mask, prompt_metal_cache: prompt_metal_cache)
                      context_timing.not_nil!.context
                    end
          context_ms += (Time.instant - context_t0).total_milliseconds
          if timing = context_timing
            context_score_ms += timing.score_ms
            context_softmax_ms += timing.softmax_ms
            context_value_ms += timing.value_ms
          end

          if batched_context_rows = context_rows
            copy_row!(batched_context_rows, canvas_pos, q_context_dim, context)
          else
            attention_t0 = Time.instant
            projected = attention_output_project(weights, il, context)
            normed = Gemma4CPU.rms_norm(projected, lw.post_attention_norm, hp.rms_eps)
            attn_out = Array(Float32).new(hp.n_embd) { |i| x[i] + normed[i] }
            attention_out_ms += (Time.instant - attention_t0).total_milliseconds
            copy_row!(attn_out_rows, canvas_pos, hp.n_embd, attn_out)
          end
        end
      end

      if batched_context_rows = context_rows
        attention_t0 = Time.instant
        attn_out_rows = attention_residual_from_context_rows(weights, il, canvas_rows, batched_context_rows, mask.canvas_len)
        attention_out_ms += (Time.instant - attention_t0).total_milliseconds
      elsif context_buf = batched_context_buf
        attention_t0 = Time.instant
        if metal_rows = attention_residual_from_context_buffer(weights, il, canvas_rows, context_buf, mask.canvas_len)
          attn_out_rows = metal_rows
          attention_residual_context_buffer_hit = true
        else
          context_rows_fallback = context_buf.read(mask.canvas_len * q_context_dim)
          attn_out_rows = attention_residual_from_context_rows(weights, il, canvas_rows, context_rows_fallback, mask.canvas_len)
        end
        attention_out_ms += (Time.instant - attention_t0).total_milliseconds
      end

      shared_rows = nil.as(Array(Float32)?)
      if shared_ffn_batch_rows_enabled?(mask.canvas_len) && mask.canvas_len > 1
        shared_t0 = Time.instant
        shared_rows = shared_dense_ffn_rows(weights, il, attn_out_rows, mask.canvas_len)
        shared_ffn_ms += (Time.instant - shared_t0).total_milliseconds
      end

      moe_rows = nil.as(Array(Float32)?)
      if moe_ffn_batch_rows_enabled?(mask.canvas_len) && mask.canvas_len > 1
        moe_t0 = Time.instant
        moe_rows = if moe_ffn_grouped_expert_rows_enabled?(mask.canvas_len)
                     grouped_timing = moe_ffn_grouped_expert_rows_timed(weights, il, attn_out_rows, mask.canvas_len, routes_by_canvas_row)
                     moe_grouped_prep_ms += grouped_timing.prep_ms
                     moe_grouped_gate_up_ms += grouped_timing.gate_up_ms
                     moe_grouped_activation_ms += grouped_timing.activation_ms
                     moe_grouped_down_ms += grouped_timing.down_ms
                     moe_grouped_scatter_combine_norm_ms += grouped_timing.scatter_combine_norm_ms
                     grouped_timing.rows
                   else
                     moe_ffn_rows(weights, il, attn_out_rows, mask.canvas_len, routes_by_canvas_row)
                   end
        moe_ffn_ms += (Time.instant - moe_t0).total_milliseconds
      end

      mask.canvas_len.times do |canvas_pos|
        attn_out = attn_out_rows[canvas_pos * hp.n_embd, hp.n_embd]

        shared = if batched_shared = shared_rows
                   batched_shared[canvas_pos * hp.n_embd, hp.n_embd]
                 else
                   shared_t0 = Time.instant
                   row_shared = shared_dense_ffn(weights, il, attn_out)
                   shared_ffn_ms += (Time.instant - shared_t0).total_milliseconds
                   row_shared
                 end

        moe = if batched_moe = moe_rows
                batched_moe[canvas_pos * hp.n_embd, hp.n_embd]
              else
                moe_t0 = Time.instant
                row_moe = if supplied_routes = routes_by_canvas_row
                            moe_ffn(weights, il, attn_out, supplied_routes[canvas_pos])
                          else
                            moe_ffn(weights, il, attn_out)
                          end
                moe_ffn_ms += (Time.instant - moe_t0).total_milliseconds
                row_moe
              end

        combine_t0 = Time.instant
        ffn_out = ffn_residual_from_parts(weights, il, attn_out, shared, moe)
        layer_row = scale_layer_output(weights, il, ffn_out, canvas: true)
        combine_scale_ms += (Time.instant - combine_t0).total_milliseconds
        copy_row!(result, canvas_pos, hp.n_embd, layer_row)
      end
      DecodeCanvasRowsTiming.new(result, qkv_ms, context_ms, attention_out_ms, shared_ffn_ms, moe_ffn_ms, combine_scale_ms, context_score_ms, context_softmax_ms, context_value_ms, attention_residual_context_buffer_hit, moe_grouped_prep_ms, moe_grouped_gate_up_ms, moe_grouped_activation_ms, moe_grouped_down_ms, moe_grouped_scatter_combine_norm_ms)
    end

    def prompt_attention_projections(weights : DiffusionGemmaWeights,
                                     il : Int32,
                                     prompt_rows : Array(Float32),
                                     mask : DiffusionGemmaAttentionMask) : Array(AttentionProjection)
      prompt_attention_projections_timed(weights, il, prompt_rows, mask).projections
    end

    def prompt_attention_projections_timed(weights : DiffusionGemmaWeights,
                                           il : Int32,
                                           prompt_rows : Array(Float32),
                                           mask : DiffusionGemmaAttentionMask) : PromptProjectionTiming
      attention_projections_timed(weights, il, prompt_rows, mask.prompt_len, 0, "prompt")
    end

    private def attention_projections_timed(weights : DiffusionGemmaWeights,
                                            il : Int32,
                                            rows : Array(Float32),
                                            row_count : Int32,
                                            start_pos : Int32,
                                            label : String) : PromptProjectionTiming
      hp = weights.hparams
      lw = weights.layers[il]
      rows_size = row_count * hp.n_embd
      raise ArgumentError.new("#{label} rows size mismatch: #{rows.size} != #{rows_size}") unless rows.size == rows_size

      norm_t0 = Time.instant
      normed_rows = rows.dup
      row_count.times do |pos|
        fast_rms_norm_slice!(normed_rows, pos * hp.n_embd, hp.n_embd, lw.attn_norm, hp.rms_eps)
      end
      norm_ms = (Time.instant - norm_t0).total_milliseconds

      head_dim = hp.head_dim_for_layer(il)
      q_dim = hp.n_head * head_dim
      kv_dim = hp.n_head_kv(il) * head_dim
      matmul_t0 = Time.instant
      q_rows, k_rows, v_rows = if v_qw = lw.attn_v_qw
                                 if projected = prompt_projection_many_matmul([lw.attn_q_qw, lw.attn_k_qw, v_qw], normed_rows, row_count)
                                   {projected[0], projected[1], projected[2]}
                                 else
                                   {
                                     prompt_projection_matmul(lw.attn_q_qw, normed_rows, row_count),
                                     prompt_projection_matmul(lw.attn_k_qw, normed_rows, row_count),
                                     prompt_projection_matmul(v_qw, normed_rows, row_count),
                                   }
                                 end
                               elsif projected = prompt_projection_many_matmul([lw.attn_q_qw, lw.attn_k_qw], normed_rows, row_count)
                                 {projected[0], projected[1], projected[1].dup}
                               else
                                 k_rows = prompt_projection_matmul(lw.attn_k_qw, normed_rows, row_count)
                                 {
                                   prompt_projection_matmul(lw.attn_q_qw, normed_rows, row_count),
                                   k_rows,
                                   k_rows.dup,
                                 }
                               end
      matmul_ms = (Time.instant - matmul_t0).total_milliseconds
      reused_k_as_v = lw.attn_v_qw.nil?

      assemble_ms = 0.0
      copy_ms = 0.0
      head_norm_ms = 0.0
      q_norm_ms = 0.0
      k_norm_ms = 0.0
      v_norm_ms = 0.0
      rope_ms = 0.0
      rope_table_ms = 0.0
      rope_apply_ms = 0.0
      rope_q_apply_ms = 0.0
      rope_k_apply_ms = 0.0
      projections = Array(AttentionProjection).new(row_count) do |pos|
        copy_t0 = Time.instant
        q = q_rows[pos * q_dim, q_dim]
        k = k_rows[pos * kv_dim, kv_dim]
        v = v_rows[pos * kv_dim, kv_dim]
        proj = AttentionProjection.new(q, k, v, reused_k_as_v)
        copy_elapsed = (Time.instant - copy_t0).total_milliseconds
        copy_ms += copy_elapsed
        projection_pos = start_pos + pos

        if prompt_projection_fused_norm_rope_enabled?
          head_norm_t0 = Time.instant
          q_elapsed, k_elapsed, v_elapsed, table_elapsed = normalize_rope_attention_projection_timed!(proj, lw, hp, il, projection_pos, weights.rope_freqs)
          head_norm_elapsed = (Time.instant - head_norm_t0).total_milliseconds
          q_norm_ms += q_elapsed
          k_norm_ms += k_elapsed
          v_norm_ms += v_elapsed
          head_norm_ms += head_norm_elapsed
          assemble_ms += copy_elapsed + head_norm_elapsed
          rope_table_ms += table_elapsed
          rope_ms += table_elapsed
        else
          head_norm_t0 = Time.instant
          q_elapsed, k_elapsed, v_elapsed = normalize_attention_projection_timed!(proj, lw, hp, il)
          head_norm_elapsed = (Time.instant - head_norm_t0).total_milliseconds
          q_norm_ms += q_elapsed
          k_norm_ms += k_elapsed
          v_norm_ms += v_elapsed
          head_norm_ms += head_norm_elapsed
          assemble_ms += copy_elapsed + head_norm_elapsed
          rope_t0 = Time.instant
          table_elapsed, apply_elapsed, q_apply_elapsed, k_apply_elapsed = apply_rope_to_qk_timed!(proj, hp, il, projection_pos, weights.rope_freqs)
          rope_elapsed = (Time.instant - rope_t0).total_milliseconds
          rope_table_ms += table_elapsed
          rope_apply_ms += apply_elapsed
          rope_q_apply_ms += q_apply_elapsed
          rope_k_apply_ms += k_apply_elapsed
          rope_ms += rope_elapsed
        end
        proj
      end
      PromptProjectionTiming.new(projections, norm_ms, matmul_ms, assemble_ms, copy_ms, head_norm_ms, q_norm_ms, k_norm_ms, v_norm_ms, rope_ms, rope_table_ms, rope_apply_ms, rope_q_apply_ms, rope_k_apply_ms)
    end

    def prompt_projection_fused_norm_rope_enabled? : Bool
      ENV["DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE"]? == "1"
    end

    def prompt_materialize_batch_rows_enabled?(prompt_len : Int32) : Bool
      return false unless ENV["DIFFUSION_GEMMA_PROMPT_MATERIALIZE_BATCH_ROWS"]? == "1"
      return false if ENV["DIFFUSION_GEMMA_PROMPT_MATERIALIZE_BATCH_ROWS_OFF"]? == "1"
      return false if prompt_len < prompt_materialize_batch_min_prompt
      max_prompt = prompt_materialize_batch_max_prompt
      return false if max_prompt > 0 && prompt_len > max_prompt
      true
    end

    def prompt_materialize_batch_min_prompt : Int32
      env_i32("DIFFUSION_GEMMA_PROMPT_MATERIALIZE_BATCH_MIN_PROMPT", 2)
    end

    def prompt_materialize_batch_max_prompt : Int32
      env_i32("DIFFUSION_GEMMA_PROMPT_MATERIALIZE_BATCH_MAX_PROMPT", 0)
    end

    def prompt_materialize_grouped_moe_enabled? : Bool
      ENV["DIFFUSION_GEMMA_PROMPT_MATERIALIZE_GROUPED_MOE"]? == "1" &&
        ENV["DIFFUSION_GEMMA_PROMPT_MATERIALIZE_GROUPED_MOE_OFF"]? != "1"
    end

    def attention_residual_metal_rows_enabled?(row_count : Int32) : Bool
      return false unless ENV["DIFFUSION_GEMMA_ATTENTION_RESIDUAL_METAL_ROWS"]? == "1"
      return false if ENV["DIFFUSION_GEMMA_ATTENTION_RESIDUAL_METAL_ROWS_OFF"]? == "1"
      return false if row_count < attention_residual_metal_min_rows
      max_rows = attention_residual_metal_max_rows
      return false if max_rows > 0 && row_count > max_rows
      Gemma4Metal.available?
    end

    def attention_residual_metal_min_rows : Int32
      env_i32("DIFFUSION_GEMMA_ATTENTION_RESIDUAL_METAL_MIN_ROWS", 1)
    end

    def attention_residual_metal_max_rows : Int32
      env_i32("DIFFUSION_GEMMA_ATTENTION_RESIDUAL_METAL_MAX_ROWS", 0)
    end

    def attention_residual_context_buffer_enabled?(canvas_len : Int32) : Bool
      return false unless ENV["DIFFUSION_GEMMA_ATTENTION_RESIDUAL_CONTEXT_BUFFER"]? == "1"
      return false if ENV["DIFFUSION_GEMMA_ATTENTION_RESIDUAL_CONTEXT_BUFFER_OFF"]? == "1"
      return false unless attention_residual_metal_rows_enabled?(canvas_len)
      return false if canvas_len < attention_residual_context_buffer_min_canvas
      max_canvas = attention_residual_context_buffer_max_canvas
      return false if max_canvas > 0 && canvas_len > max_canvas
      Gemma4Metal.available?
    end

    def attention_residual_context_buffer_min_canvas : Int32
      env_i32("DIFFUSION_GEMMA_ATTENTION_RESIDUAL_CONTEXT_BUFFER_MIN_CANVAS", 1)
    end

    def attention_residual_context_buffer_max_canvas : Int32
      env_i32("DIFFUSION_GEMMA_ATTENTION_RESIDUAL_CONTEXT_BUFFER_MAX_CANVAS", 0)
    end

    def grouped_moe_policy_enabled?(canvas_len : Int32) : Bool
      return false unless ENV["DIFFUSION_GEMMA_GROUPED_MOE_POLICY"]? == "1"
      return false if ENV["DIFFUSION_GEMMA_GROUPED_MOE_POLICY_OFF"]? == "1"
      return false if canvas_len < grouped_moe_policy_min_canvas
      max_canvas = grouped_moe_policy_max_canvas
      return false if max_canvas > 0 && canvas_len > max_canvas
      true
    end

    def grouped_moe_policy_min_canvas : Int32
      env_i32("DIFFUSION_GEMMA_GROUPED_MOE_POLICY_MIN_CANVAS", 4)
    end

    def grouped_moe_policy_max_canvas : Int32
      env_i32("DIFFUSION_GEMMA_GROUPED_MOE_POLICY_MAX_CANVAS", 16)
    end

    def shared_ffn_batch_rows_enabled?(canvas_len : Int32) : Bool
      return false if ENV["DIFFUSION_GEMMA_SHARED_FFN_BATCH_ROWS_OFF"]? == "1"
      if ENV["DIFFUSION_GEMMA_SHARED_FFN_BATCH_ROWS"]? == "1"
        return false if canvas_len < shared_ffn_batch_min_canvas
        max_canvas = shared_ffn_batch_max_canvas
        return false if max_canvas > 0 && canvas_len > max_canvas
        return true
      end
      return false unless grouped_moe_policy_enabled?(canvas_len)
      true
    end

    def shared_ffn_batch_min_canvas : Int32
      env_i32("DIFFUSION_GEMMA_SHARED_FFN_BATCH_MIN_CANVAS", 1)
    end

    def shared_ffn_batch_max_canvas : Int32
      env_i32("DIFFUSION_GEMMA_SHARED_FFN_BATCH_MAX_CANVAS", 0)
    end

    def moe_ffn_batch_rows_enabled? : Bool
      ENV["DIFFUSION_GEMMA_MOE_FFN_BATCH_ROWS"]? == "1" &&
        ENV["DIFFUSION_GEMMA_MOE_FFN_BATCH_ROWS_OFF"]? != "1"
    end

    def moe_ffn_batch_rows_enabled?(canvas_len : Int32) : Bool
      return false if ENV["DIFFUSION_GEMMA_MOE_FFN_BATCH_ROWS_OFF"]? == "1"
      return true if ENV["DIFFUSION_GEMMA_MOE_FFN_BATCH_ROWS"]? == "1"
      grouped_moe_policy_enabled?(canvas_len)
    end

    def moe_ffn_grouped_expert_rows_enabled?(canvas_len : Int32) : Bool
      return false if ENV["DIFFUSION_GEMMA_MOE_GROUPED_EXPERT_ROWS_OFF"]? == "1"
      if ENV["DIFFUSION_GEMMA_MOE_GROUPED_EXPERT_ROWS"]? == "1"
        return false if canvas_len < moe_ffn_grouped_expert_min_canvas
        max_canvas = moe_ffn_grouped_expert_max_canvas
        return false if max_canvas > 0 && canvas_len > max_canvas
        return true
      end
      return false unless grouped_moe_policy_enabled?(canvas_len)
      true
    end

    def moe_ffn_grouped_expert_min_canvas : Int32
      env_i32("DIFFUSION_GEMMA_MOE_GROUPED_EXPERT_MIN_CANVAS", 1)
    end

    def moe_ffn_grouped_expert_max_canvas : Int32
      env_i32("DIFFUSION_GEMMA_MOE_GROUPED_EXPERT_MAX_CANVAS", 0)
    end

    def attention_out_batch_rows_enabled?(canvas_len : Int32) : Bool
      return false unless ENV["DIFFUSION_GEMMA_ATTENTION_OUT_BATCH_ROWS"]? == "1"
      return false if ENV["DIFFUSION_GEMMA_ATTENTION_OUT_BATCH_ROWS_OFF"]? == "1"
      return false if canvas_len < attention_out_batch_min_canvas
      max_canvas = attention_out_batch_max_canvas
      return false if max_canvas > 0 && canvas_len > max_canvas
      true
    end

    def attention_out_batch_min_canvas : Int32
      env_i32("DIFFUSION_GEMMA_ATTENTION_OUT_BATCH_MIN_CANVAS", 1)
    end

    def attention_out_batch_max_canvas : Int32
      env_i32("DIFFUSION_GEMMA_ATTENTION_OUT_BATCH_MAX_CANVAS", 0)
    end

    def prompt_projection_metal_enabled? : Bool
      ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL"]? == "1" &&
        ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL_OFF"]? != "1"
    end

    def prompt_projection_metal_min_batch : Int32
      (ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH"]? || "16").to_i
    end

    def grouped_moe_cognigraph_plan_enabled? : Bool
      ENV["DIFFUSION_GEMMA_MOE_COGNIGRAPH_PLAN"]? == "1" &&
        ENV["DIFFUSION_GEMMA_MOE_COGNIGRAPH_PLAN_OFF"]? != "1"
    end

    private record CogniGraphAccessRecord,
      op_index : Int32,
      binding : CogniGraphPlanBinding

    private def compile_cognigraph_plan(ops : Array(CogniGraphPlanOp),
                                        active_experts : Int32,
                                        route_slots : Int32,
                                        row_count : Int32) : CogniGraphPlan
      writers = {} of String => Array(CogniGraphAccessRecord)
      readers = {} of String => Array(CogniGraphAccessRecord)
      wave_of = Array(Int32).new(ops.size, 0)

      ops.each_with_index do |op, i|
        max_dep_wave = -1
        op.bindings.each do |binding|
          case binding.access
          when .read?
            if ws = writers[binding.buffer]?
              ws.each do |wr|
                next unless binding.conflicts?(wr.binding)
                w = wave_of[wr.op_index]
                max_dep_wave = w if w > max_dep_wave
              end
            end
            (readers[binding.buffer] ||= [] of CogniGraphAccessRecord) << CogniGraphAccessRecord.new(i, binding)
          when .write?
            if rs = readers[binding.buffer]?
              rs.each do |rd|
                next unless binding.conflicts?(rd.binding)
                w = wave_of[rd.op_index]
                max_dep_wave = w if w > max_dep_wave
              end
            end
            if ws = writers[binding.buffer]?
              ws.each do |wr|
                next unless binding.conflicts?(wr.binding)
                w = wave_of[wr.op_index]
                max_dep_wave = w if w > max_dep_wave
              end
            end
            (writers[binding.buffer] ||= [] of CogniGraphAccessRecord) << CogniGraphAccessRecord.new(i, binding)
          when .read_write?
            if ws = writers[binding.buffer]?
              ws.each do |wr|
                next unless binding.conflicts?(wr.binding)
                w = wave_of[wr.op_index]
                max_dep_wave = w if w > max_dep_wave
              end
            end
            if rs = readers[binding.buffer]?
              rs.each do |rd|
                next unless binding.conflicts?(rd.binding)
                w = wave_of[rd.op_index]
                max_dep_wave = w if w > max_dep_wave
              end
            end
            rec = CogniGraphAccessRecord.new(i, binding)
            (writers[binding.buffer] ||= [] of CogniGraphAccessRecord) << rec
            (readers[binding.buffer] ||= [] of CogniGraphAccessRecord) << rec
          end
        end
        wave_of[i] = max_dep_wave + 1
      end

      max_wave = wave_of.max? || 0
      wave_names = (0..max_wave).map do |w|
        (0...ops.size).select { |i| wave_of[i] == w }.map { |i| ops[i].name }
      end
      wave_widths = wave_names.map(&.size.to_i32)
      CogniGraphPlan.new(
        n_ops: ops.size.to_i32,
        n_waves: wave_names.size.to_i32,
        n_barriers: {wave_names.size - 1, 0}.max.to_i32,
        max_wave_width: (wave_widths.max? || 0).to_i32,
        active_experts: active_experts,
        route_slots: route_slots,
        row_count: row_count,
        wave_widths: wave_widths,
        wave_names: wave_names,
      )
    end

    private def emit_grouped_moe_cognigraph_plan(il : Int32,
                                                 row_count : Int32,
                                                 plan : CogniGraphPlan) : Nil
      STDERR.puts "diffusion_gemma_moe_cognigraph_plan layer=#{il} rows=#{row_count} active_experts=#{plan.active_experts} route_slots=#{plan.route_slots} ops=#{plan.n_ops} waves=#{plan.n_waves} barriers=#{plan.n_barriers} max_wave_width=#{plan.max_wave_width} wave_widths=#{plan.wave_widths.join(",")} #{plan.phi}"
    end

    def env_i32(name : String, default : Int32) : Int32
      raw = ENV[name]? || return default
      raw.to_i? || default
    end

    def decode_context_metal_backend_enabled? : Bool
      context_metal_enabled? && Gemma4Metal.available?
    end

    def decode_context_metal_batch_rows_enabled? : Bool
      context_metal_batch_rows_enabled? && Gemma4Metal.available?
    end

    def decode_context_fixed_gqa2_enabled? : Bool
      decode_context_metal_batch_rows_enabled? &&
        ENV["GEMMA4_ROW_PREFILL_ATTN_FIXED_SWA256_VEC_GQA2"]? == "1" &&
        ENV["GEMMA4_ROW_PREFILL_ATTN_FIXED_SWA256_VEC_GQA2_OFF"]? != "1"
    end

    def prompt_projection_matmul(qw : QuantWeight,
                                 x : Array(Float32),
                                 batch : Int32) : Array(Float32)
      if prompt_projection_metal_enabled? && batch >= prompt_projection_metal_min_batch
        metal_rows = case qw.type
                     when .q4_k?
                       Qwen35Metal.matmul_q4k(x, qw.raw, qw.in_dim, qw.out_dim, batch)
                     when .q5_k?
                       Qwen35Metal.matmul_q5k(x, qw.raw, qw.in_dim, qw.out_dim, batch)
                     when .q6_k?
                       Qwen35Metal.matmul_q6k(x, qw.raw, qw.in_dim, qw.out_dim, batch)
                     else
                       Qwen35Metal.matmul(qw, x, batch)
                     end
        unless metal_rows.nil?
          return metal_rows
        end
      end
      Gemma4CPU.matmul(qw, x, batch)
    end

    def prompt_projection_many_matmul(qws : Array(QuantWeight),
                                      x : Array(Float32),
                                      batch : Int32) : Array(Array(Float32))?
      {% if flag?(:cpu_only) %}
        nil
      {% else %}
        return [] of Array(Float32) if qws.empty?
        return nil unless prompt_projection_metal_enabled? && batch >= prompt_projection_metal_min_batch
        in_dim = qws[0].in_dim
        return nil unless x.size == batch * in_dim
        return nil unless qws.all? { |qw| qw.in_dim == in_dim }

        x_buf = ML::MetalBuffer.new((batch * in_dim).to_i64 * sizeof(Float32))
        out_bufs = qws.map do |qw|
          ML::MetalBuffer.new((batch * qw.out_dim).to_i64 * sizeof(Float32))
        end
        begin
          x_buf.write(x)
          return nil unless Qwen35Metal.matmul_many_to_buffers(qws, x_buf, out_bufs, batch)

          qws.map_with_index do |qw, i|
            out_bufs[i].read(batch * qw.out_dim)
          end
        ensure
          x_buf.release
          out_bufs.each(&.release)
        end
      {% end %}
    end

    def output_hidden_norm(weights : DiffusionGemmaWeights,
                           hidden : Array(Float32)) : Array(Float32)
      hp = weights.hparams
      raise ArgumentError.new("output hidden size mismatch") unless hidden.size == hp.n_embd
      Gemma4CPU.rms_norm(hidden, weights.output_norm, hp.rms_eps)
    end

    def output_logits_for_tokens(weights : DiffusionGemmaWeights,
                                 hidden : Array(Float32),
                                 token_ids : Array(Int32)) : Array(Float32)
      hp = weights.hparams
      raise ArgumentError.new("output token_ids must not be empty") if token_ids.empty?
      normed = output_hidden_norm(weights, hidden)
      logits = Array(Float32).new(token_ids.size, 0.0_f32)
      token_ids.each_with_index do |token_id, i|
        raise ArgumentError.new("output token id out of range") if token_id < 0 || token_id >= hp.vocab_size
        row = quant_row_slice(weights.token_embd, token_id, 1, hp.n_embd)
        logits[i] = Gemma4CPU.matmul(row, normed)[0]
      end
      Gemma4CPU.logit_softcap!(logits, hp.final_logit_softcapping)
      logits
    end

    def bounded_candidate_prediction(candidate_token_ids : Array(Int32),
                                     raw_logits : Array(Float32),
                                     temp_inv : Float32 = 1.0_f32,
                                     sample_u : Float32 = 0.0_f32) : BoundedDenoisePrediction
      validate_candidate_prediction_inputs!(candidate_token_ids, raw_logits, temp_inv, sample_u)

      logits = raw_logits.map { |v| v * temp_inv }
      probs = softmax(logits)
      entropy = categorical_entropy(probs)

      best = 0
      probs.each_with_index do |p, i|
        if p > probs[best] || (p == probs[best] && candidate_token_ids[i] < candidate_token_ids[best])
          best = i
        end
      end

      sampled = probs.size - 1
      cum = 0.0_f32
      probs.each_with_index do |p, i|
        cum += p
        if cum >= sample_u
          sampled = i
          break
        end
      end

      BoundedDenoisePrediction.new(
        candidate_token_ids: candidate_token_ids.dup,
        logits: logits,
        probabilities: probs,
        argmax_token_id: candidate_token_ids[best],
        sampled_token_id: candidate_token_ids[sampled],
        entropy: entropy,
      )
    end

    def bounded_denoise_prediction(weights : DiffusionGemmaWeights,
                                   hidden : Array(Float32),
                                   candidate_token_ids : Array(Int32),
                                   temp_inv : Float32 = 1.0_f32,
                                   sample_u : Float32 = 0.0_f32) : BoundedDenoisePrediction
      raw_logits = output_logits_for_tokens(weights, hidden, candidate_token_ids)
      bounded_candidate_prediction(candidate_token_ids, raw_logits, temp_inv, sample_u)
    end

    def current_token_candidate_rows(canvas_tokens : Array(Int32),
                                     vocab_size : Int32) : Array(Array(Int32))
      validate_candidate_tokens!(canvas_tokens, vocab_size)
      canvas_tokens.map { |token_id| [token_id] }
    end

    def generated_candidate_rows(canvas_tokens : Array(Int32),
                                 count : Int32,
                                 vocab_size : Int32) : Array(Array(Int32))
      raise ArgumentError.new("candidate count must be positive") unless count > 0
      raise ArgumentError.new("candidate count exceeds vocab size") if count > vocab_size
      validate_candidate_tokens!(canvas_tokens, vocab_size)
      canvas_tokens.map do |token_id|
        Array(Int32).new(count) { |i| (token_id + i) % vocab_size }.sort
      end
    end

    def merge_candidate_rows(canvas_tokens : Array(Int32),
                             proposal_token_ids_by_canvas_row : Array(Array(Int32)),
                             vocab_size : Int32) : Array(Array(Int32))
      raise ArgumentError.new("proposal rows size mismatch") unless proposal_token_ids_by_canvas_row.size == canvas_tokens.size
      validate_candidate_tokens!(canvas_tokens, vocab_size)

      canvas_tokens.map_with_index do |token_id, row|
        merged = proposal_token_ids_by_canvas_row[row].dup
        validate_candidate_tokens!(merged, vocab_size)
        merged << token_id
        merged.uniq!
        merged.sort!
        merged
      end
    end

    def current_token_candidate_steps(canvas_tokens : Array(Int32),
                                      vocab_size : Int32,
                                      steps : Int32) : Array(Array(Array(Int32)))
      raise ArgumentError.new("candidate steps must be positive") unless steps > 0
      rows = current_token_candidate_rows(canvas_tokens, vocab_size)
      Array(Array(Array(Int32))).new(steps) { rows.map(&.dup) }
    end

    def merge_candidate_steps(canvas_tokens : Array(Int32),
                              proposal_token_ids_by_step_by_canvas_row : Array(Array(Array(Int32))),
                              vocab_size : Int32) : Array(Array(Array(Int32)))
      raise ArgumentError.new("proposal steps must not be empty") if proposal_token_ids_by_step_by_canvas_row.empty?
      proposal_token_ids_by_step_by_canvas_row.map do |proposal_rows|
        merge_candidate_rows(canvas_tokens, proposal_rows, vocab_size)
      end
    end

    def top_k_prediction_tokens(prediction : BoundedDenoisePrediction,
                                k : Int32) : Array(Int32)
      raise ArgumentError.new("prediction top-k must be positive") unless k > 0
      order = (0...prediction.candidate_token_ids.size).to_a
      order.sort! do |a, b|
        cmp = prediction.probabilities[b] <=> prediction.probabilities[a]
        cmp == 0 ? prediction.candidate_token_ids[a] <=> prediction.candidate_token_ids[b] : cmp
      end
      order[0, Math.min(k, order.size)].map { |i| prediction.candidate_token_ids[i] }
    end

    def prediction_proposal_rows(predictions : Array(BoundedDenoisePrediction),
                                 top_k : Int32) : Array(Array(Int32))
      raise ArgumentError.new("prediction proposal rows must not be empty") if predictions.empty?
      predictions.map { |prediction| top_k_prediction_tokens(prediction, top_k) }
    end

    def next_candidate_rows_from_predictions(canvas_tokens : Array(Int32),
                                             predictions : Array(BoundedDenoisePrediction),
                                             vocab_size : Int32,
                                             top_k : Int32) : Array(Array(Int32))
      raise ArgumentError.new("prediction rows size mismatch") unless predictions.size == canvas_tokens.size
      merge_candidate_rows(canvas_tokens, prediction_proposal_rows(predictions, top_k), vocab_size)
    end

    def repeated_candidate_steps_from_predictions(canvas_tokens : Array(Int32),
                                                  predictions : Array(BoundedDenoisePrediction),
                                                  vocab_size : Int32,
                                                  top_k : Int32,
                                                  steps : Int32) : Array(Array(Array(Int32)))
      raise ArgumentError.new("candidate steps must be positive") unless steps > 0
      rows = next_candidate_rows_from_predictions(canvas_tokens, predictions, vocab_size, top_k)
      Array(Array(Array(Int32))).new(steps) { rows.map(&.dup) }
    end

    def sample_u_rows(seed : Int32,
                      canvas_len : Int32,
                      step : Int32 = 0) : Array(Float32)
      raise ArgumentError.new("sample_u canvas_len must be positive") unless canvas_len > 0
      raise ArgumentError.new("sample_u step must be non-negative") if step < 0

      state = sample_seed_state(seed) &+ step.to_u64 &* 0x9E3779B97F4A7C15_u64
      Array(Float32).new(canvas_len) do
        state = splitmix64_next(state)
        ((state >> 40).to_u32.to_f32 / 16_777_216.0_f32)
      end
    end

    def sample_u_steps(seed : Int32,
                       steps : Int32,
                       canvas_len : Int32) : Array(Array(Float32))
      raise ArgumentError.new("sample_u steps must be positive") unless steps > 0
      Array(Array(Float32)).new(steps) { |step| sample_u_rows(seed, canvas_len, step) }
    end

    def decode_canvas_bounded_predictions(weights : DiffusionGemmaWeights,
                                          canvas_rows : Array(Float32),
                                          mask : DiffusionGemmaAttentionMask,
                                          prompt_cache : PromptLayerCache,
                                          candidate_token_ids_by_canvas_row : Array(Array(Int32)),
                                          max_layers : Int32 = prompt_cache.layers,
                                          temp_inv : Float32 = 1.0_f32,
                                          sample_us : Array(Float32)? = nil,
                                          routes_by_layer_by_canvas_row : Array(Array(Array(ExpertRoute)))? = nil) : Array(BoundedDenoisePrediction)
      decode_canvas_bounded_predictions_timed(
        weights: weights,
        canvas_rows: canvas_rows,
        mask: mask,
        prompt_cache: prompt_cache,
        candidate_token_ids_by_canvas_row: candidate_token_ids_by_canvas_row,
        max_layers: max_layers,
        temp_inv: temp_inv,
        sample_us: sample_us,
        routes_by_layer_by_canvas_row: routes_by_layer_by_canvas_row,
      ).predictions
    end

    def decode_canvas_bounded_predictions_timed(weights : DiffusionGemmaWeights,
                                                canvas_rows : Array(Float32),
                                                mask : DiffusionGemmaAttentionMask,
                                                prompt_cache : PromptLayerCache,
                                                candidate_token_ids_by_canvas_row : Array(Array(Int32)),
                                                max_layers : Int32 = prompt_cache.layers,
                                                temp_inv : Float32 = 1.0_f32,
                                                sample_us : Array(Float32)? = nil,
                                                routes_by_layer_by_canvas_row : Array(Array(Array(ExpertRoute)))? = nil) : BoundedDenoisePredictionTiming
      hp = weights.hparams
      raise ArgumentError.new("candidate rows size mismatch") unless candidate_token_ids_by_canvas_row.size == mask.canvas_len
      if supplied_sample_us = sample_us
        raise ArgumentError.new("sample_us size mismatch") unless supplied_sample_us.size == mask.canvas_len
      end

      decode_t0 = Time.instant
      decode_timing = decode_canvas_rows_with_prompt_cache_timed(
        weights: weights,
        canvas_rows: canvas_rows,
        mask: mask,
        prompt_cache: prompt_cache,
        max_layers: max_layers,
        routes_by_layer_by_canvas_row: routes_by_layer_by_canvas_row,
      )
      decode_stack_ms = (Time.instant - decode_t0).total_milliseconds

      output_t0 = Time.instant
      predictions = Array(BoundedDenoisePrediction).new(mask.canvas_len) do |canvas_pos|
        hidden = decode_timing.rows[canvas_pos * hp.n_embd, hp.n_embd]
        sample_u = sample_us ? sample_us.not_nil![canvas_pos] : 0.0_f32
        bounded_denoise_prediction(weights, hidden, candidate_token_ids_by_canvas_row[canvas_pos], temp_inv, sample_u)
      end
      BoundedDenoisePredictionTiming.new(
        predictions,
        decode_stack_ms,
        decode_timing.qkv_ms,
        decode_timing.context_ms,
        decode_timing.attention_out_ms,
        decode_timing.shared_ffn_ms,
        decode_timing.moe_ffn_ms,
        decode_timing.combine_scale_ms,
        (Time.instant - output_t0).total_milliseconds,
        decode_timing.context_score_ms,
        decode_timing.context_softmax_ms,
        decode_timing.context_value_ms,
        decode_timing.attention_residual_context_buffer,
      )
    end

    def decode_canvas_bounded_step(weights : DiffusionGemmaWeights,
                                   canvas_tokens : Array(Int32),
                                   canvas_rows : Array(Float32),
                                   mask : DiffusionGemmaAttentionMask,
                                   prompt_cache : PromptLayerCache,
                                   candidate_token_ids_by_canvas_row : Array(Array(Int32)),
                                   entropy_bound : Float32,
                                   max_layers : Int32 = prompt_cache.layers,
                                   temp_inv : Float32 = 1.0_f32,
                                   sample_us : Array(Float32)? = nil,
                                   routes_by_layer_by_canvas_row : Array(Array(Array(ExpertRoute)))? = nil,
                                   use_sampled_token : Bool = true,
                                   sc_token_ids_by_canvas_row : Array(Array(Int32))? = nil,
                                   sc_logits_by_canvas_row : Array(Array(Float32))? = nil,
                                   sc_temp_inv : Float32 = 1.0_f32,
                                   sc_use : Float32 = 1.0_f32) : BoundedCanvasUpdate
      decode_canvas_bounded_step_timed(
        weights: weights,
        canvas_tokens: canvas_tokens,
        canvas_rows: canvas_rows,
        mask: mask,
        prompt_cache: prompt_cache,
        candidate_token_ids_by_canvas_row: candidate_token_ids_by_canvas_row,
        entropy_bound: entropy_bound,
        max_layers: max_layers,
        temp_inv: temp_inv,
        sample_us: sample_us,
        routes_by_layer_by_canvas_row: routes_by_layer_by_canvas_row,
        use_sampled_token: use_sampled_token,
        sc_token_ids_by_canvas_row: sc_token_ids_by_canvas_row,
        sc_logits_by_canvas_row: sc_logits_by_canvas_row,
        sc_temp_inv: sc_temp_inv,
        sc_use: sc_use,
      ).update
    end

    def decode_canvas_bounded_step_timed(weights : DiffusionGemmaWeights,
                                         canvas_tokens : Array(Int32),
                                         canvas_rows : Array(Float32),
                                         mask : DiffusionGemmaAttentionMask,
                                         prompt_cache : PromptLayerCache,
                                         candidate_token_ids_by_canvas_row : Array(Array(Int32)),
                                         entropy_bound : Float32,
                                         max_layers : Int32 = prompt_cache.layers,
                                         temp_inv : Float32 = 1.0_f32,
                                         sample_us : Array(Float32)? = nil,
                                         routes_by_layer_by_canvas_row : Array(Array(Array(ExpertRoute)))? = nil,
                                         use_sampled_token : Bool = true,
                                         sc_token_ids_by_canvas_row : Array(Array(Int32))? = nil,
                                         sc_logits_by_canvas_row : Array(Array(Float32))? = nil,
                                         sc_temp_inv : Float32 = 1.0_f32,
                                         sc_use : Float32 = 1.0_f32) : BoundedDenoiseStepTiming
      raise ArgumentError.new("canvas token count mismatch") unless canvas_tokens.size == mask.canvas_len
      prediction_t0 = Time.instant
      prediction_timing = decode_canvas_bounded_predictions_timed(
        weights: weights,
        canvas_rows: canvas_rows,
        mask: mask,
        prompt_cache: prompt_cache,
        candidate_token_ids_by_canvas_row: candidate_token_ids_by_canvas_row,
        max_layers: max_layers,
        temp_inv: temp_inv,
        sample_us: sample_us,
        routes_by_layer_by_canvas_row: routes_by_layer_by_canvas_row,
      )
      prediction_ms = (Time.instant - prediction_t0).total_milliseconds
      update_t0 = Time.instant
      update = apply_entropy_bound_predictions(canvas_tokens, prediction_timing.predictions, entropy_bound, use_sampled_token: use_sampled_token)
      update_ms = (Time.instant - update_t0).total_milliseconds
      regenerate_t0 = Time.instant
      updated_rows = canvas_rows_from_tokens(
        weights,
        update.updated_canvas_tokens,
        sc_token_ids_by_canvas_row: sc_token_ids_by_canvas_row,
        sc_logits_by_canvas_row: sc_logits_by_canvas_row,
        sc_temp_inv: sc_temp_inv,
        sc_use: sc_use,
      )
      regenerate_ms = (Time.instant - regenerate_t0).total_milliseconds
      timed_update = BoundedCanvasUpdate.new(update.updated_canvas_tokens, update.accepted, update.predictions, updated_rows)
      BoundedDenoiseStepTiming.new(
        timed_update,
        prediction_ms,
        prediction_timing.decode_stack_ms,
        prediction_timing.decode_qkv_ms,
        prediction_timing.decode_context_ms,
        prediction_timing.decode_attention_out_ms,
        prediction_timing.decode_shared_ffn_ms,
        prediction_timing.decode_moe_ffn_ms,
        prediction_timing.decode_combine_scale_ms,
        prediction_timing.output_head_ms,
        update_ms,
        regenerate_ms,
        prediction_timing.decode_context_score_ms,
        prediction_timing.decode_context_softmax_ms,
        prediction_timing.decode_context_value_ms,
        prediction_timing.decode_attention_residual_context_buffer,
      )
    end

    def decode_canvas_bounded_loop(weights : DiffusionGemmaWeights,
                                   canvas_tokens : Array(Int32),
                                   canvas_rows : Array(Float32),
                                   mask : DiffusionGemmaAttentionMask,
                                   prompt_cache : PromptLayerCache,
                                   candidate_token_ids_by_step_by_canvas_row : Array(Array(Array(Int32))),
                                   entropy_bound : Float32,
                                   stability_threshold : Int32,
                                   max_layers : Int32 = prompt_cache.layers,
                                   temp_inv : Float32 = 1.0_f32,
                                   sample_us_by_step_by_canvas_row : Array(Array(Float32))? = nil,
                                   routes_by_layer_by_canvas_row : Array(Array(Array(ExpertRoute)))? = nil,
                                   use_sampled_token : Bool = true,
                                   use_sparse_self_conditioning : Bool = false,
                                   sc_temp_inv : Float32 = 1.0_f32,
                                   sc_use : Float32 = 1.0_f32) : BoundedDenoiseLoopResult
      raise ArgumentError.new("denoise steps must not be empty") if candidate_token_ids_by_step_by_canvas_row.empty?
      raise ArgumentError.new("stability_threshold must be positive") unless stability_threshold > 0
      if supplied_sample_us = sample_us_by_step_by_canvas_row
        raise ArgumentError.new("sample_us step count mismatch") unless supplied_sample_us.size == candidate_token_ids_by_step_by_canvas_row.size
      end

      tokens = canvas_tokens.dup
      rows = canvas_rows.dup
      stable_counts = Array(Int32).new(canvas_tokens.size, 0)
      updates = [] of BoundedCanvasUpdate
      step_traces = [] of BoundedDenoiseStepTrace
      converged = false

      candidate_token_ids_by_step_by_canvas_row.each_with_index do |candidate_rows, step|
        sample_us = sample_us_by_step_by_canvas_row ? sample_us_by_step_by_canvas_row.not_nil![step] : nil
        timed = decode_canvas_bounded_step_timed(
          weights: weights,
          canvas_tokens: tokens,
          canvas_rows: rows,
          mask: mask,
          prompt_cache: prompt_cache,
          candidate_token_ids_by_canvas_row: candidate_rows,
          entropy_bound: entropy_bound,
          max_layers: max_layers,
          temp_inv: temp_inv,
          sample_us: sample_us,
          routes_by_layer_by_canvas_row: routes_by_layer_by_canvas_row,
          use_sampled_token: use_sampled_token,
        )
        update = timed.update
        stable_counts = advance_stability_counts(tokens, update.updated_canvas_tokens, update.accepted, stable_counts)
        tokens = update.updated_canvas_tokens
        if use_sparse_self_conditioning
          regenerate_t0 = Time.instant
          rows = canvas_rows_from_prediction_self_conditioning(weights, tokens, update.predictions, sc_temp_inv, sc_use)
          timed = BoundedDenoiseStepTiming.new(
            update,
            timed.prediction_ms,
            timed.decode_stack_ms,
            timed.decode_qkv_ms,
            timed.decode_context_ms,
            timed.decode_attention_out_ms,
            timed.decode_shared_ffn_ms,
            timed.decode_moe_ffn_ms,
            timed.decode_combine_scale_ms,
            timed.output_head_ms,
            timed.update_ms,
            (Time.instant - regenerate_t0).total_milliseconds,
            timed.decode_context_score_ms,
            timed.decode_context_softmax_ms,
            timed.decode_context_value_ms,
            timed.decode_attention_residual_context_buffer,
          )
          update = BoundedCanvasUpdate.new(update.updated_canvas_tokens, update.accepted, update.predictions, rows)
        else
          rows = update.updated_canvas_rows || rows
        end
        updates << update
        step_traces << bounded_denoise_step_trace(step, update, timed)
        if stable_counts.all? { |count| count >= stability_threshold }
          converged = true
          break
        end
      end

      stop_reason = converged ? "converged" : "step_budget"
      BoundedDenoiseLoopResult.new(tokens, rows, updates, stable_counts, updates.size, converged, step_traces, stop_reason)
    end

    def decode_canvas_adaptive_bounded_loop(weights : DiffusionGemmaWeights,
                                            canvas_tokens : Array(Int32),
                                            canvas_rows : Array(Float32),
                                            mask : DiffusionGemmaAttentionMask,
                                            prompt_cache : PromptLayerCache,
                                            initial_candidate_token_ids_by_canvas_row : Array(Array(Int32)),
                                            entropy_bound : Float32,
                                            stability_threshold : Int32,
                                            max_steps : Int32,
                                            proposal_top_k : Int32,
                                            max_layers : Int32 = prompt_cache.layers,
                                            temp_inv : Float32 = 1.0_f32,
                                            sample_us_by_step_by_canvas_row : Array(Array(Float32))? = nil,
                                            routes_by_layer_by_canvas_row : Array(Array(Array(ExpertRoute)))? = nil,
                                            use_sampled_token : Bool = true,
                                            use_sparse_self_conditioning : Bool = false,
                                            sc_temp_inv : Float32 = 1.0_f32,
                                            sc_use : Float32 = 1.0_f32) : BoundedDenoiseLoopResult
      raise ArgumentError.new("max_steps must be positive") unless max_steps > 0
      raise ArgumentError.new("proposal_top_k must be positive") unless proposal_top_k > 0
      if supplied_sample_us = sample_us_by_step_by_canvas_row
        raise ArgumentError.new("sample_us step count mismatch") unless supplied_sample_us.size == max_steps
      end

      tokens = canvas_tokens.dup
      rows = canvas_rows.dup
      candidate_rows = initial_candidate_token_ids_by_canvas_row
      stable_counts = Array(Int32).new(canvas_tokens.size, 0)
      updates = [] of BoundedCanvasUpdate
      step_traces = [] of BoundedDenoiseStepTrace
      converged = false

      max_steps.times do |step|
        sample_us = sample_us_by_step_by_canvas_row ? sample_us_by_step_by_canvas_row.not_nil![step] : nil
        timed = decode_canvas_bounded_step_timed(
          weights: weights,
          canvas_tokens: tokens,
          canvas_rows: rows,
          mask: mask,
          prompt_cache: prompt_cache,
          candidate_token_ids_by_canvas_row: candidate_rows,
          entropy_bound: entropy_bound,
          max_layers: max_layers,
          temp_inv: temp_inv,
          sample_us: sample_us,
          routes_by_layer_by_canvas_row: routes_by_layer_by_canvas_row,
          use_sampled_token: use_sampled_token,
        )
        update = timed.update
        stable_counts = advance_stability_counts(tokens, update.updated_canvas_tokens, update.accepted, stable_counts)
        tokens = update.updated_canvas_tokens
        if use_sparse_self_conditioning
          regenerate_t0 = Time.instant
          rows = canvas_rows_from_prediction_self_conditioning(weights, tokens, update.predictions, sc_temp_inv, sc_use)
          timed = BoundedDenoiseStepTiming.new(
            update,
            timed.prediction_ms,
            timed.decode_stack_ms,
            timed.decode_qkv_ms,
            timed.decode_context_ms,
            timed.decode_attention_out_ms,
            timed.decode_shared_ffn_ms,
            timed.decode_moe_ffn_ms,
            timed.decode_combine_scale_ms,
            timed.output_head_ms,
            timed.update_ms,
            (Time.instant - regenerate_t0).total_milliseconds,
            timed.decode_context_score_ms,
            timed.decode_context_softmax_ms,
            timed.decode_context_value_ms,
            timed.decode_attention_residual_context_buffer,
          )
          update = BoundedCanvasUpdate.new(update.updated_canvas_tokens, update.accepted, update.predictions, rows)
        else
          rows = update.updated_canvas_rows || rows
        end
        updates << update
        proposal_ms = 0.0
        if stable_counts.all? { |count| count >= stability_threshold }
          step_traces << bounded_denoise_step_trace(step, update, timed, proposal_ms)
          converged = true
          break
        end
        proposal_t0 = Time.instant
        candidate_rows = next_candidate_rows_from_predictions(tokens, update.predictions, weights.hparams.vocab_size, proposal_top_k)
        proposal_ms = (Time.instant - proposal_t0).total_milliseconds
        step_traces << bounded_denoise_step_trace(step, update, timed, proposal_ms)
      end

      stop_reason = converged ? "converged" : "step_budget"
      BoundedDenoiseLoopResult.new(tokens, rows, updates, stable_counts, updates.size, converged, step_traces, stop_reason)
    end

    def apply_entropy_bound_prediction_steps(canvas_tokens : Array(Int32),
                                             predictions_by_step : Array(Array(BoundedDenoisePrediction)),
                                             entropy_bound : Float32,
                                             stability_threshold : Int32,
                                             use_sampled_token : Bool = true) : BoundedDenoiseLoopResult
      raise ArgumentError.new("prediction steps must not be empty") if predictions_by_step.empty?
      raise ArgumentError.new("stability_threshold must be positive") unless stability_threshold > 0

      tokens = canvas_tokens.dup
      stable_counts = Array(Int32).new(canvas_tokens.size, 0)
      updates = [] of BoundedCanvasUpdate
      step_traces = [] of BoundedDenoiseStepTrace
      converged = false
      predictions_by_step.each_with_index do |predictions, step|
        update = apply_entropy_bound_predictions(tokens, predictions, entropy_bound, use_sampled_token: use_sampled_token)
        stable_counts = advance_stability_counts(tokens, update.updated_canvas_tokens, update.accepted, stable_counts)
        tokens = update.updated_canvas_tokens
        updates << update
        step_traces << bounded_denoise_step_trace(step, update)
        if stable_counts.all? { |count| count >= stability_threshold }
          converged = true
          break
        end
      end

      stop_reason = converged ? "converged" : "prediction_budget"
      BoundedDenoiseLoopResult.new(tokens, nil, updates, stable_counts, updates.size, converged, step_traces, stop_reason)
    end

    def bounded_denoise_step_trace(step : Int32,
                                   update : BoundedCanvasUpdate,
                                   timing : BoundedDenoiseStepTiming? = nil,
                                   proposal_ms : Float64 = 0.0) : BoundedDenoiseStepTrace
      raise ArgumentError.new("trace step must be non-negative") unless step >= 0
      raise ArgumentError.new("trace accepted size mismatch") unless update.accepted.size == update.predictions.size

      accepted_count = update.accepted.count(true)
      total_candidate_tokens = 0
      max_candidate_tokens = 0
      entropy_sum = 0.0_f32
      update.predictions.each do |prediction|
        width = prediction.candidate_token_ids.size
        total_candidate_tokens += width
        max_candidate_tokens = width if width > max_candidate_tokens
        entropy_sum += prediction.entropy
      end

      prediction_count = update.predictions.size
      mean_candidate_tokens = prediction_count == 0 ? 0.0_f32 : total_candidate_tokens.to_f32 / prediction_count.to_f32
      mean_entropy = prediction_count == 0 ? 0.0_f32 : entropy_sum / prediction_count.to_f32

      BoundedDenoiseStepTrace.new(
        step: step,
        prediction_count: prediction_count,
        accepted_count: accepted_count,
        total_candidate_tokens: total_candidate_tokens,
        max_candidate_tokens: max_candidate_tokens,
        mean_candidate_tokens: mean_candidate_tokens,
        mean_entropy: mean_entropy,
        prediction_ms: timing ? timing.not_nil!.prediction_ms : 0.0,
        decode_stack_ms: timing ? timing.not_nil!.decode_stack_ms : 0.0,
        decode_qkv_ms: timing ? timing.not_nil!.decode_qkv_ms : 0.0,
        decode_context_ms: timing ? timing.not_nil!.decode_context_ms : 0.0,
        decode_context_score_ms: timing ? timing.not_nil!.decode_context_score_ms : 0.0,
        decode_context_softmax_ms: timing ? timing.not_nil!.decode_context_softmax_ms : 0.0,
        decode_context_value_ms: timing ? timing.not_nil!.decode_context_value_ms : 0.0,
        decode_attention_out_ms: timing ? timing.not_nil!.decode_attention_out_ms : 0.0,
        decode_shared_ffn_ms: timing ? timing.not_nil!.decode_shared_ffn_ms : 0.0,
        decode_moe_ffn_ms: timing ? timing.not_nil!.decode_moe_ffn_ms : 0.0,
        decode_combine_scale_ms: timing ? timing.not_nil!.decode_combine_scale_ms : 0.0,
        output_head_ms: timing ? timing.not_nil!.output_head_ms : 0.0,
        update_ms: timing ? timing.not_nil!.update_ms : 0.0,
        regenerate_ms: timing ? timing.not_nil!.regenerate_ms : 0.0,
        proposal_ms: proposal_ms,
        decode_attention_residual_context_buffer: timing ? timing.not_nil!.decode_attention_residual_context_buffer : false,
      )
    end

    def apply_entropy_bound_predictions(canvas_tokens : Array(Int32),
                                        predictions : Array(BoundedDenoisePrediction),
                                        entropy_bound : Float32,
                                        use_sampled_token : Bool = true) : BoundedCanvasUpdate
      raise ArgumentError.new("canvas prediction count mismatch") unless predictions.size == canvas_tokens.size
      accepted = entropy_bound_accept(predictions.map(&.entropy), entropy_bound)
      updated = canvas_tokens.dup
      accepted.each_with_index do |ok, i|
        next unless ok

        pred = predictions[i]
        updated[i] = use_sampled_token ? pred.sampled_token_id : pred.argmax_token_id
      end
      BoundedCanvasUpdate.new(updated, accepted, predictions)
    end

    def advance_stability_counts(previous_tokens : Array(Int32),
                                 updated_tokens : Array(Int32),
                                 accepted : Array(Bool),
                                 previous_counts : Array(Int32)) : Array(Int32)
      raise ArgumentError.new("stability token size mismatch") unless previous_tokens.size == updated_tokens.size
      raise ArgumentError.new("stability accepted size mismatch") unless accepted.size == previous_tokens.size
      raise ArgumentError.new("stability count size mismatch") unless previous_counts.size == previous_tokens.size

      Array(Int32).new(previous_tokens.size) do |i|
        if accepted[i] && updated_tokens[i] == previous_tokens[i]
          previous_counts[i] + 1
        else
          0
        end
      end
    end

    def update_canvas_token(canvas_tokens : Array(Int32),
                            canvas_index : Int32,
                            token_id : Int32) : Array(Int32)
      raise ArgumentError.new("canvas_index out of bounds") if canvas_index < 0 || canvas_index >= canvas_tokens.size
      result = canvas_tokens.dup
      result[canvas_index] = token_id
      result
    end

    def entropy_bound_accept(entropies : Array(Float32),
                             entropy_bound : Float32) : Array(Bool)
      raise ArgumentError.new("entropy_bound must be finite and non-negative") unless entropy_bound.finite? && entropy_bound >= 0.0_f32
      raise ArgumentError.new("entropies must not be empty") if entropies.empty?
      entropies.each { |entropy| raise ArgumentError.new("entropy must be finite and non-negative") unless entropy.finite? && entropy >= 0.0_f32 }

      order = (0...entropies.size).to_a
      order.sort! { |a, b| (entropies[a] <=> entropies[b]) == 0 ? a <=> b : entropies[a] <=> entropies[b] }
      accepted = Array(Bool).new(entropies.size, false)
      cum = 0.0_f64
      order.each do |pos|
        entropy = entropies[pos]
        cum += entropy
        accepted[pos] = true if cum - entropy <= entropy_bound
      end
      accepted
    end

    def ffn_residual_from_parts(weights : DiffusionGemmaWeights,
                                il : Int32,
                                attn_out : Array(Float32),
                                shared_dense : Array(Float32),
                                moe : Array(Float32)? = nil) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("ffn residual input size mismatch") unless attn_out.size == hp.n_embd
      raise ArgumentError.new("shared_dense size mismatch") unless shared_dense.size == hp.n_embd
      if moe_branch = moe
        raise ArgumentError.new("moe size mismatch") unless moe_branch.size == hp.n_embd
      end

      combined = if moe_branch = moe
                   Array(Float32).new(hp.n_embd) { |i| shared_dense[i] + moe_branch[i] }
                 else
                   shared_dense
                 end
      normed = Gemma4CPU.rms_norm(combined, lw.post_ffw_norm, hp.rms_eps)
      Array(Float32).new(hp.n_embd) { |i| attn_out[i] + normed[i] }
    end

    def ffn_residual_from_parts_rows(weights : DiffusionGemmaWeights,
                                     il : Int32,
                                     attn_out_rows : Array(Float32),
                                     shared_dense_rows : Array(Float32),
                                     moe_rows : Array(Float32)?,
                                     row_count : Int32,
                                     canvas : Bool) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      expected = row_count * hp.n_embd
      raise ArgumentError.new("ffn residual rows row_count must be positive") unless row_count > 0
      raise ArgumentError.new("ffn residual rows input size mismatch") unless attn_out_rows.size == expected
      raise ArgumentError.new("shared_dense rows size mismatch") unless shared_dense_rows.size == expected
      if moe_branch = moe_rows
        raise ArgumentError.new("moe rows size mismatch") unless moe_branch.size == expected
      end

      result = Array(Float32).new(expected, 0.0_f32)
      scale = canvas ? lw.layer_output_scale[0] : lw.encoder_layer_output_scale[0]
      moe_branch = moe_rows
      row_count.times do |row|
        offset = row * hp.n_embd
        hp.n_embd.times do |i|
          result[offset + i] = shared_dense_rows[offset + i] + (moe_branch ? moe_branch[offset + i] : 0.0_f32)
        end
        fast_rms_norm_slice!(result, offset, hp.n_embd, lw.post_ffw_norm, hp.rms_eps)
        hp.n_embd.times do |i|
          result[offset + i] = (attn_out_rows[offset + i] + result[offset + i]) * scale
        end
      end
      result
    end

    def row_rms(x : Array(Float32), offset : Int32, len : Int32) : Float32
      raise ArgumentError.new("row_rms out of bounds") if offset < 0 || len <= 0 || offset + len > x.size
      ss = 0.0_f64
      len.times do |i|
        v = x[offset + i]
        ss += v.to_f64 * v.to_f64
      end
      Math.sqrt(ss / len.to_f64).to_f32
    end

    private def attention_context_from_keyspace(query : AttentionProjection,
                                                keyspace : Array(AttentionProjection),
                                                hp : DiffusionGemmaHparams,
                                                il : Int32,
                                                allowed : Proc(Int32, Bool)) : Array(Float32)
      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      heads_per_group = n_head // n_head_kv
      raise ArgumentError.new("invalid GQA layout at layer #{il}") unless heads_per_group > 0 && n_head % n_head_kv == 0
      validate_projection_shape!(query, q_dim, kv_dim, il)
      keyspace.each { |proj| validate_projection_shape!(proj, q_dim, kv_dim, il) }

      allowed_keys = [] of Int32
      keyspace.size.times { |key_pos| allowed_keys << key_pos if allowed.call(key_pos) }
      raise ArgumentError.new("attention has no allowed keys") if allowed_keys.empty?

      result = Array(Float32).new(q_dim, 0.0_f32)
      scores = Array(Float32).new(allowed_keys.size, 0.0_f32)
      n_head.times do |h|
        kvh = h // heads_per_group
        q_off = h * head_dim
        allowed_keys.each_with_index do |key_pos, i|
          k_off = kvh * head_dim
          scores[i] = dot(query.q, q_off, keyspace[key_pos].k, k_off, head_dim)
        end
        Gemma4CPU.softmax_slice!(scores, 0, scores.size)

        out_off = h * head_dim
        allowed_keys.each_with_index do |key_pos, i|
          v_off = kvh * head_dim
          weight = scores[i]
          result_ptr = result.to_unsafe + out_off
          value_ptr = keyspace[key_pos].v.to_unsafe + v_off
          d = 0
          while d < head_dim
            result_ptr[d] += weight * value_ptr[d]
            d += 1
          end
        end
      end
      result
    end

    private def attention_context_from_range(query : AttentionProjection,
                                             keyspace : Array(AttentionProjection),
                                             hp : DiffusionGemmaHparams,
                                             il : Int32,
                                             low : Int32,
                                             high : Int32) : Array(Float32)
      attention_context_from_range_timed(query, keyspace, hp, il, low, high).context
    end

    private def attention_context_from_range_timed(query : AttentionProjection,
                                                   keyspace : Array(AttentionProjection),
                                                   hp : DiffusionGemmaHparams,
                                                   il : Int32,
                                                   low : Int32,
                                                   high : Int32) : AttentionContextTiming
      raise ArgumentError.new("attention has no allowed keys") if low > high
      raise ArgumentError.new("attention range out of bounds") if low < 0 || high >= keyspace.size

      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      heads_per_group = n_head // n_head_kv
      raise ArgumentError.new("invalid GQA layout at layer #{il}") unless heads_per_group > 0 && n_head % n_head_kv == 0
      validate_projection_shape!(query, q_dim, kv_dim, il)
      if context_metal_enabled?
        if context = attention_context_from_range_metal(query, keyspace, q_dim, kv_dim, hp, il, low, high)
          return AttentionContextTiming.new(context, 0.0, 0.0, 0.0)
        end
      end

      key_count = high - low + 1
      result = Array(Float32).new(q_dim, 0.0_f32)
      scores = Array(Float32).new(key_count, 0.0_f32)
      score_ms = 0.0
      softmax_ms = 0.0
      value_ms = 0.0
      n_head.times do |h|
        kvh = h // heads_per_group
        q_off = h * head_dim
        score_t0 = Time.instant
        key_count.times do |i|
          key_pos = low + i
          k_off = kvh * head_dim
          scores[i] = dot(query.q, q_off, keyspace[key_pos].k, k_off, head_dim)
        end
        score_ms += (Time.instant - score_t0).total_milliseconds
        softmax_t0 = Time.instant
        Gemma4CPU.softmax_slice!(scores, 0, scores.size)
        softmax_ms += (Time.instant - softmax_t0).total_milliseconds

        out_off = h * head_dim
        value_t0 = Time.instant
        key_count.times do |i|
          key_pos = low + i
          v_off = kvh * head_dim
          weight = scores[i]
          result_ptr = result.to_unsafe + out_off
          value_ptr = keyspace[key_pos].v.to_unsafe + v_off
          d = 0
          while d < head_dim
            result_ptr[d] += weight * value_ptr[d]
            d += 1
          end
        end
        value_ms += (Time.instant - value_t0).total_milliseconds
      end
      AttentionContextTiming.new(result, score_ms, softmax_ms, value_ms)
    end

    private def context_metal_enabled? : Bool
      ENV["DIFFUSION_GEMMA_CONTEXT_METAL"]? == "1" &&
        ENV["DIFFUSION_GEMMA_CONTEXT_METAL_OFF"]? != "1"
    end

    private def context_metal_batch_rows_enabled? : Bool
      context_metal_enabled? &&
        ENV["DIFFUSION_GEMMA_CONTEXT_METAL_BATCH_ROWS_OFF"]? != "1"
    end

    private def attention_context_decode_metal_resident(query : AttentionProjection,
                                                        cache : PromptLayerMetalCache,
                                                        hp : DiffusionGemmaHparams,
                                                        il : Int32,
                                                        low : Int32,
                                                        high : Int32) : Array(Float32)?
      return nil unless Gemma4Metal.available?
      return nil unless high >= low
      return nil unless high < cache.prompt_len + cache.canvas_len

      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      return nil unless cache.kv_dim == kv_dim
      validate_projection_shape!(query, q_dim, kv_dim, il)

      sliding_window = low == 0 ? 0 : high - low + 1
      cache.write_query!(query)
      Gemma4Metal.attention_context_rows_resident_buffers(
        cache.q_buf, cache.k_cache_buf, cache.v_cache_buf, cache.out_buf, high, 1, n_head, n_head_kv, head_dim, sliding_window)
    end

    private def attention_context_decode_batch_metal_resident(queries : Array(AttentionProjection),
                                                              cache : PromptLayerMetalCache,
                                                              hp : DiffusionGemmaHparams,
                                                              il : Int32,
                                                              low : Int32,
                                                              high : Int32) : Array(Float32)?
      return nil unless Gemma4Metal.available?
      return nil unless queries.size == cache.canvas_len
      return nil unless high >= low
      return nil unless high < cache.prompt_len + cache.canvas_len

      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      return nil unless cache.q_dim == q_dim && cache.kv_dim == kv_dim
      queries.each { |query| validate_projection_shape!(query, q_dim, kv_dim, il) }

      sliding_window = low == 0 ? 0 : high - low + 1
      cache.write_queries!(queries)
      Gemma4Metal.attention_context_rows_fixed_resident_buffers(
        cache.q_buf, cache.k_cache_buf, cache.v_cache_buf, cache.out_buf, high, queries.size, n_head, n_head_kv, head_dim, sliding_window)
    end

    private def attention_context_decode_batch_metal_resident_buffer(queries : Array(AttentionProjection),
                                                                     cache : PromptLayerMetalCache,
                                                                     hp : DiffusionGemmaHparams,
                                                                     il : Int32,
                                                                     low : Int32,
                                                                     high : Int32) : ML::MetalBuffer?
      return nil unless Gemma4Metal.available?
      return nil unless queries.size == cache.canvas_len
      return nil unless high >= low
      return nil unless high < cache.prompt_len + cache.canvas_len

      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)
      q_dim = n_head * head_dim
      kv_dim = n_head_kv * head_dim
      return nil unless cache.q_dim == q_dim && cache.kv_dim == kv_dim
      queries.each { |query| validate_projection_shape!(query, q_dim, kv_dim, il) }

      sliding_window = low == 0 ? 0 : high - low + 1
      cache.write_queries!(queries)
      return cache.out_buf if Gemma4Metal.attention_context_rows_fixed_resident_buffers_no_read(
                                cache.q_buf, cache.k_cache_buf, cache.v_cache_buf, cache.out_buf, high, queries.size, n_head, n_head_kv, head_dim, sliding_window)
      nil
    end

    private def attention_context_from_range_metal(query : AttentionProjection,
                                                   keyspace : Array(AttentionProjection),
                                                   q_dim : Int32,
                                                   kv_dim : Int32,
                                                   hp : DiffusionGemmaHparams,
                                                   il : Int32,
                                                   low : Int32,
                                                   high : Int32) : Array(Float32)?
      return nil unless Gemma4Metal.available?
      return nil unless high >= low
      return nil unless high < keyspace.size

      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)
      sliding_window = low == 0 ? 0 : high - low + 1
      k_cache = Array(Float32).new((high + 1) * kv_dim, 0.0_f32)
      v_cache = Array(Float32).new((high + 1) * kv_dim, 0.0_f32)
      (0..high).each do |pos|
        proj = keyspace[pos]
        validate_projection_shape!(proj, q_dim, kv_dim, il)
        k_dst = k_cache.to_unsafe + pos * kv_dim
        v_dst = v_cache.to_unsafe + pos * kv_dim
        k_src = proj.k.to_unsafe
        v_src = proj.v.to_unsafe
        d = 0
        while d < kv_dim
          k_dst[d] = k_src[d]
          v_dst[d] = v_src[d]
          d += 1
        end
      end

      Gemma4Metal.attention_context_rows(
        query.q, k_cache, v_cache, high, 1, n_head, n_head_kv, head_dim, sliding_window)
    end

    private def validate_projection_shape!(proj : AttentionProjection,
                                           q_dim : Int32,
                                           kv_dim : Int32,
                                           il : Int32) : Nil
      raise ArgumentError.new("q projection size mismatch at layer #{il}") unless proj.q.size == q_dim
      raise ArgumentError.new("k projection size mismatch at layer #{il}") unless proj.k.size == kv_dim
      raise ArgumentError.new("v projection size mismatch at layer #{il}") unless proj.v.size == kv_dim
    end

    private def dot(a : Array(Float32), a_off : Int32, b : Array(Float32), b_off : Int32, len : Int32) : Float32
      sum = 0.0_f32
      a_ptr = a.to_unsafe + a_off
      b_ptr = b.to_unsafe + b_off
      i = 0
      while i < len
        sum += a_ptr[i] * b_ptr[i]
        i += 1
      end
      sum
    end

    private def fast_rms_norm_slice!(x : Array(Float32), offset : Int32, len : Int32,
                                     w : Array(Float32), eps : Float32) : Nil
      raise ArgumentError.new("rms_norm_slice weight size mismatch: #{w.size} != #{len}") unless w.size == len
      raise ArgumentError.new("rms_norm_slice out of bounds") if offset < 0 || len < 0 || offset + len > x.size

      x_ptr = x.to_unsafe + offset
      ss = 0.0_f64
      i = 0
      while i < len
        v = x_ptr[i]
        ss += v.to_f64 * v.to_f64
        i += 1
      end
      inv_rms = (1.0 / Math.sqrt(ss / len.to_f64 + eps.to_f64)).to_f32
      w_ptr = w.to_unsafe
      i = 0
      while i < len
        x_ptr[i] = x_ptr[i] * inv_rms * w_ptr[i]
        i += 1
      end
    end

    private def fast_rms_norm_plain_slice!(x : Array(Float32), offset : Int32, len : Int32,
                                           eps : Float32) : Nil
      raise ArgumentError.new("rms_norm_plain_slice out of bounds") if offset < 0 || len < 0 || offset + len > x.size

      x_ptr = x.to_unsafe + offset
      ss = 0.0_f64
      i = 0
      while i < len
        v = x_ptr[i]
        ss += v.to_f64 * v.to_f64
        i += 1
      end
      inv_rms = (1.0 / Math.sqrt(ss / len.to_f64 + eps.to_f64)).to_f32
      i = 0
      while i < len
        x_ptr[i] *= inv_rms
        i += 1
      end
    end

    private def rope_tables(pos : Int32,
                            n_rot : Int32,
                            freq_base : Float32,
                            freq_factors : Array(Float32)? = nil) : {Array(Float32), Array(Float32)}
      raise ArgumentError.new("rope n_rot must be even") unless n_rot.even?
      if factors = freq_factors
        raise ArgumentError.new("rope freq_factors too small: #{factors.size} < #{n_rot // 2}") if factors.size < n_rot // 2
      end

      half = n_rot // 2
      cos_table = Array(Float32).new(half, 0.0_f32)
      sin_table = Array(Float32).new(half, 0.0_f32)
      factors_ptr = freq_factors.try(&.to_unsafe)
      i = 0
      while i < half
        i0 = 2 * i
        freq_factor = factors_ptr ? factors_ptr[i] : 1.0_f32
        theta = pos.to_f32 * (freq_base ** (-i0.to_f32 / n_rot.to_f32)) / freq_factor
        cos_table[i] = Math.cos(theta).to_f32
        sin_table[i] = Math.sin(theta).to_f32
        i += 1
      end
      {cos_table, sin_table}
    end

    private def fast_rope_neox_slice!(x : Array(Float32), offset : Int32, n_rot : Int32,
                                      head_dim : Int32,
                                      cos_table : Array(Float32),
                                      sin_table : Array(Float32)) : Nil
      raise ArgumentError.new("rope n_rot must be even") unless n_rot.even?
      raise ArgumentError.new("rope n_rot #{n_rot} exceeds head_dim #{head_dim}") if n_rot > head_dim
      raise ArgumentError.new("rope slice out of bounds") if offset < 0 || offset + head_dim > x.size
      raise ArgumentError.new("rope table size mismatch") unless cos_table.size == n_rot // 2 && sin_table.size == n_rot // 2

      x_ptr = x.to_unsafe + offset
      cos_ptr = cos_table.to_unsafe
      sin_ptr = sin_table.to_unsafe
      half = n_rot // 2
      i = 0
      while i < half
        x0 = x_ptr[i]
        x1 = x_ptr[i + half]
        x_ptr[i] = x0 * cos_ptr[i] - x1 * sin_ptr[i]
        x_ptr[i + half] = x0 * sin_ptr[i] + x1 * cos_ptr[i]
        i += 1
      end
    end

    private def fast_rms_norm_rope_neox_slice!(x : Array(Float32), offset : Int32, len : Int32,
                                               w : Array(Float32), eps : Float32,
                                               n_rot : Int32,
                                               cos_table : Array(Float32),
                                               sin_table : Array(Float32)) : Nil
      raise ArgumentError.new("rms_norm_rope weight size mismatch: #{w.size} != #{len}") unless w.size == len
      raise ArgumentError.new("rope n_rot must be even") unless n_rot.even?
      raise ArgumentError.new("rope n_rot #{n_rot} exceeds len #{len}") if n_rot > len
      raise ArgumentError.new("rms_norm_rope slice out of bounds") if offset < 0 || len < 0 || offset + len > x.size
      raise ArgumentError.new("rope table size mismatch") unless cos_table.size == n_rot // 2 && sin_table.size == n_rot // 2

      x_ptr = x.to_unsafe + offset
      ss = 0.0_f64
      i = 0
      while i < len
        v = x_ptr[i]
        ss += v.to_f64 * v.to_f64
        i += 1
      end
      inv_rms = (1.0 / Math.sqrt(ss / len.to_f64 + eps.to_f64)).to_f32
      w_ptr = w.to_unsafe
      cos_ptr = cos_table.to_unsafe
      sin_ptr = sin_table.to_unsafe
      half = n_rot // 2
      i = 0
      while i < half
        x0 = x_ptr[i] * inv_rms * w_ptr[i]
        x1_idx = i + half
        x1 = x_ptr[x1_idx] * inv_rms * w_ptr[x1_idx]
        x_ptr[i] = x0 * cos_ptr[i] - x1 * sin_ptr[i]
        x_ptr[x1_idx] = x0 * sin_ptr[i] + x1 * cos_ptr[i]
        i += 1
      end
      i = n_rot
      while i < len
        x_ptr[i] = x_ptr[i] * inv_rms * w_ptr[i]
        i += 1
      end
    end

    private def sort_routes!(routes : Array(ExpertRoute)) : Nil
      routes.sort! do |a, b|
        cmp = b.weight <=> a.weight
        cmp == 0 ? a.expert <=> b.expert : cmp
      end
    end

    private def better_route?(a : ExpertRoute, b : ExpertRoute) : Bool
      a.weight > b.weight || (a.weight == b.weight && a.expert < b.expert)
    end

    private def categorical_entropy(probs : Array(Float32)) : Float32
      entropy = 0.0_f64
      probs.each do |p|
        entropy -= p.to_f64 * Math.log(p.to_f64) if p > 0.0_f32
      end
      entropy.to_f32
    end

    private def validate_candidate_prediction_inputs!(candidate_token_ids : Array(Int32),
                                                      raw_logits : Array(Float32),
                                                      temp_inv : Float32,
                                                      sample_u : Float32) : Nil
      raise ArgumentError.new("candidate_token_ids must not be empty") if candidate_token_ids.empty?
      raise ArgumentError.new("candidate logits size mismatch") unless raw_logits.size == candidate_token_ids.size
      raise ArgumentError.new("candidate temp_inv must be finite and positive") unless temp_inv.finite? && temp_inv > 0.0_f32
      raise ArgumentError.new("candidate sample_u must be in [0, 1)") unless sample_u.finite? && sample_u >= 0.0_f32 && sample_u < 1.0_f32
      candidate_token_ids.each_cons_pair do |a, b|
        raise ArgumentError.new("candidate_token_ids must be strictly increasing") unless a < b
      end
      raw_logits.each do |logit|
        raise ArgumentError.new("candidate logits must be finite") unless logit.finite?
      end
    end

    private def validate_candidate_tokens!(token_ids : Array(Int32), vocab_size : Int32) : Nil
      raise ArgumentError.new("candidate vocab_size must be positive") unless vocab_size > 0
      token_ids.each do |token_id|
        raise ArgumentError.new("candidate token id out of range") if token_id < 0 || token_id >= vocab_size
      end
    end

    private def sample_seed_state(seed : Int32) : UInt64
      (seed.to_i64 & 0xffffffff_i64).to_u64
    end

    private def splitmix64_next(state : UInt64) : UInt64
      z = state &+ 0x9E3779B97F4A7C15_u64
      z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9_u64
      z = (z ^ (z >> 27)) &* 0x94D049BB133111EB_u64
      z ^ (z >> 31)
    end

    private def expert_gate_up_qw(lw : DiffusionGemmaLayerWeights,
                                  hp : DiffusionGemmaHparams,
                                  expert : Int32) : QuantWeight
      qw = lw.ffn_gate_up_exps_qw || raise ArgumentError.new("combined gate_up experts are required")
      raise ArgumentError.new("gate_up expert tensor shape mismatch") unless qw.in_dim == hp.n_embd && qw.out_dim == hp.expert_count * hp.expert_ff * 2
      quant_row_slice(qw, expert * hp.expert_ff * 2, hp.expert_ff * 2, hp.n_embd)
    end

    private def expert_gate_qw(lw : DiffusionGemmaLayerWeights,
                               hp : DiffusionGemmaHparams,
                               expert : Int32) : QuantWeight
      qw = lw.ffn_gate_up_exps_qw || raise ArgumentError.new("combined gate_up experts are required")
      raise ArgumentError.new("gate expert tensor shape mismatch") unless qw.in_dim == hp.n_embd && qw.out_dim == hp.expert_count * hp.expert_ff * 2
      quant_row_slice(qw, expert * hp.expert_ff * 2, hp.expert_ff, hp.n_embd)
    end

    private def expert_up_qw(lw : DiffusionGemmaLayerWeights,
                             hp : DiffusionGemmaHparams,
                             expert : Int32) : QuantWeight
      qw = lw.ffn_gate_up_exps_qw || raise ArgumentError.new("combined gate_up experts are required")
      raise ArgumentError.new("up expert tensor shape mismatch") unless qw.in_dim == hp.n_embd && qw.out_dim == hp.expert_count * hp.expert_ff * 2
      quant_row_slice(qw, expert * hp.expert_ff * 2 + hp.expert_ff, hp.expert_ff, hp.n_embd)
    end

    private def expert_down_qw(lw : DiffusionGemmaLayerWeights,
                               hp : DiffusionGemmaHparams,
                               expert : Int32) : QuantWeight
      qw = lw.ffn_down_exps_qw
      raise ArgumentError.new("down expert tensor shape mismatch") unless qw.in_dim == hp.expert_ff && qw.out_dim == hp.expert_count * hp.n_embd
      quant_row_slice(qw, expert * hp.n_embd, hp.n_embd, hp.expert_ff)
    end

    private def quant_row_slice(qw : QuantWeight,
                                first_row : Int32,
                                row_count : Int32,
                                in_dim : Int32) : QuantWeight
      row_bytes = QuantMatmul.row_bytes(qw.type, in_dim)
      offset = first_row.to_i64 * row_bytes.to_i64
      bytes = row_count.to_i64 * row_bytes.to_i64
      raise ArgumentError.new("quant row slice out of bounds") if first_row < 0 || row_count <= 0 || offset + bytes > qw.raw.size

      raw = Bytes.new(qw.raw.to_unsafe + offset, bytes.to_i32, read_only: true)
      QuantWeight.new(raw, qw.type, row_count, in_dim, qw.route_tag)
    end

    private def copy_row!(result : Array(Float32), row : Int32, dim : Int32, values : Array(Float32)) : Nil
      raise ArgumentError.new("row size mismatch") unless values.size == dim
      offset = row * dim
      dim.times { |i| result[offset + i] = values[i] }
    end
  end
end
