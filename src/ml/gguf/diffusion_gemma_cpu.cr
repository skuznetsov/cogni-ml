require "./diffusion_gemma_runtime"
require "./gemma4_cpu"
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
      getter decode_attention_out_ms : Float64
      getter decode_shared_ffn_ms : Float64
      getter decode_moe_ffn_ms : Float64
      getter decode_combine_scale_ms : Float64
      getter output_head_ms : Float64
      getter update_ms : Float64
      getter regenerate_ms : Float64
      getter proposal_ms : Float64

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
                     @decode_attention_out_ms = 0.0,
                     @decode_shared_ffn_ms = 0.0,
                     @decode_moe_ffn_ms = 0.0,
                     @decode_combine_scale_ms = 0.0,
                     @output_head_ms = 0.0,
                     @update_ms = 0.0,
                     @regenerate_ms = 0.0,
                     @proposal_ms = 0.0)
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
      getter decode_attention_out_ms : Float64
      getter decode_shared_ffn_ms : Float64
      getter decode_moe_ffn_ms : Float64
      getter decode_combine_scale_ms : Float64
      getter output_head_ms : Float64

      def initialize(@predictions,
                     @decode_stack_ms,
                     @decode_qkv_ms,
                     @decode_context_ms,
                     @decode_attention_out_ms,
                     @decode_shared_ffn_ms,
                     @decode_moe_ffn_ms,
                     @decode_combine_scale_ms,
                     @output_head_ms)
      end
    end

    struct BoundedDenoiseStepTiming
      getter update : BoundedCanvasUpdate
      getter prediction_ms : Float64
      getter decode_stack_ms : Float64
      getter decode_qkv_ms : Float64
      getter decode_context_ms : Float64
      getter decode_attention_out_ms : Float64
      getter decode_shared_ffn_ms : Float64
      getter decode_moe_ffn_ms : Float64
      getter decode_combine_scale_ms : Float64
      getter output_head_ms : Float64
      getter update_ms : Float64
      getter regenerate_ms : Float64

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
                     @regenerate_ms)
      end
    end

    struct DecodeCanvasRowsTiming
      getter rows : Array(Float32)
      getter qkv_ms : Float64
      getter context_ms : Float64
      getter attention_out_ms : Float64
      getter shared_ffn_ms : Float64
      getter moe_ffn_ms : Float64
      getter combine_scale_ms : Float64

      def initialize(@rows,
                     @qkv_ms,
                     @context_ms,
                     @attention_out_ms,
                     @shared_ffn_ms,
                     @moe_ffn_ms,
                     @combine_scale_ms)
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
      getter materialize_ms_by_layer : Array(Float64)

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
                     @materialize_ms_by_layer = [] of Float64)
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
                     @rope_apply_ms)
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
                                        rope_freqs : Array(Float32)? = nil) : {Float64, Float64}
      head_dim = hp.head_dim_for_layer(il)
      n_rot = hp.rope_dim_for_layer(il)
      base = hp.rope_freq_base_for_layer(il)
      freqs = hp.full_attention?(il) ? rope_freqs : nil
      table_t0 = Time.instant
      cos_table, sin_table = rope_tables(pos, n_rot, base, freqs)
      table_ms = (Time.instant - table_t0).total_milliseconds

      apply_t0 = Time.instant
      hp.n_head.times do |h|
        fast_rope_neox_slice!(proj.q, h * head_dim, n_rot, head_dim, cos_table, sin_table)
      end
      hp.n_head_kv(il).times do |h|
        fast_rope_neox_slice!(proj.k, h * head_dim, n_rot, head_dim, cos_table, sin_table)
      end
      apply_ms = (Time.instant - apply_t0).total_milliseconds
      {table_ms, apply_ms}
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

      gate = gate_up[0, hp.expert_ff]
      up = gate_up[hp.expert_ff, hp.expert_ff]
      hidden = Array(Float32).new(hp.expert_ff) { |i| Gemma4CPU.gelu(gate[i]) * up[i] }

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
      materialize_ms_by_layer = [] of Float64
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
        projections_by_layer << projections
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
        materialize_ms_by_layer,
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
      attention_out_ms = 0.0
      shared_ffn_ms = 0.0
      moe_ffn_ms = 0.0
      combine_scale_ms = 0.0
      max_layers.times do |il|
        routes = routes_by_layer_by_canvas_row ? routes_by_layer_by_canvas_row.not_nil![il] : nil
        timed = layer_forward_decode_canvas_rows_with_prompt_projections_timed(
          weights: weights,
          il: il,
          prompt_projections: prompt_cache.projections_by_layer[il],
          canvas_rows: rows,
          mask: mask,
          routes_by_canvas_row: routes,
        )
        rows = timed.rows
        qkv_ms += timed.qkv_ms
        context_ms += timed.context_ms
        attention_out_ms += timed.attention_out_ms
        shared_ffn_ms += timed.shared_ffn_ms
        moe_ffn_ms += timed.moe_ffn_ms
        combine_scale_ms += timed.combine_scale_ms
      end
      DecodeCanvasRowsTiming.new(rows, qkv_ms, context_ms, attention_out_ms, shared_ffn_ms, moe_ffn_ms, combine_scale_ms)
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
                                                                       routes_by_canvas_row : Array(Array(ExpertRoute))? = nil) : DecodeCanvasRowsTiming
      hp = weights.hparams
      lw = weights.layers[il]
      canvas_size = mask.canvas_len * hp.n_embd
      raise ArgumentError.new("prompt projection count mismatch") unless prompt_projections.size == mask.prompt_len
      raise ArgumentError.new("canvas rows size mismatch: #{canvas_rows.size} != #{canvas_size}") unless canvas_rows.size == canvas_size
      if supplied_routes = routes_by_canvas_row
        raise ArgumentError.new("routes_by_canvas_row size mismatch: #{supplied_routes.size} != #{mask.canvas_len}") unless supplied_routes.size == mask.canvas_len
      end

      qkv_t0 = Time.instant
      canvas_projections = Array(AttentionProjection).new(mask.canvas_len) do |pos|
        x = canvas_rows[pos * hp.n_embd, hp.n_embd]
        attention_project_normed(weights, il, x, mask.prompt_len + pos)
      end
      qkv_ms = (Time.instant - qkv_t0).total_milliseconds
      context_ms = 0.0
      attention_out_ms = 0.0
      shared_ffn_ms = 0.0
      moe_ffn_ms = 0.0
      combine_scale_ms = 0.0
      result = Array(Float32).new(canvas_size, 0.0_f32)
      mask.canvas_len.times do |canvas_pos|
        x = canvas_rows[canvas_pos * hp.n_embd, hp.n_embd]
        context_t0 = Time.instant
        context = attention_context_decode(prompt_projections, canvas_projections, hp, il, canvas_query_index: canvas_pos, mask: mask)
        context_ms += (Time.instant - context_t0).total_milliseconds

        attention_t0 = Time.instant
        projected = attention_output_project(weights, il, context)
        normed = Gemma4CPU.rms_norm(projected, lw.post_attention_norm, hp.rms_eps)
        attn_out = Array(Float32).new(hp.n_embd) { |i| x[i] + normed[i] }
        attention_out_ms += (Time.instant - attention_t0).total_milliseconds

        shared_t0 = Time.instant
        shared = shared_dense_ffn(weights, il, attn_out)
        shared_ffn_ms += (Time.instant - shared_t0).total_milliseconds

        moe_t0 = Time.instant
        moe = if supplied_routes = routes_by_canvas_row
                moe_ffn(weights, il, attn_out, supplied_routes[canvas_pos])
              else
                moe_ffn(weights, il, attn_out)
              end
        moe_ffn_ms += (Time.instant - moe_t0).total_milliseconds

        combine_t0 = Time.instant
        ffn_out = ffn_residual_from_parts(weights, il, attn_out, shared, moe)
        layer_row = scale_layer_output(weights, il, ffn_out, canvas: true)
        combine_scale_ms += (Time.instant - combine_t0).total_milliseconds
        copy_row!(result, canvas_pos, hp.n_embd, layer_row)
      end
      DecodeCanvasRowsTiming.new(result, qkv_ms, context_ms, attention_out_ms, shared_ffn_ms, moe_ffn_ms, combine_scale_ms)
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
      hp = weights.hparams
      lw = weights.layers[il]
      prompt_size = mask.prompt_len * hp.n_embd
      raise ArgumentError.new("prompt rows size mismatch: #{prompt_rows.size} != #{prompt_size}") unless prompt_rows.size == prompt_size

      norm_t0 = Time.instant
      normed_rows = prompt_rows.dup
      mask.prompt_len.times do |pos|
        fast_rms_norm_slice!(normed_rows, pos * hp.n_embd, hp.n_embd, lw.attn_norm, hp.rms_eps)
      end
      norm_ms = (Time.instant - norm_t0).total_milliseconds

      head_dim = hp.head_dim_for_layer(il)
      q_dim = hp.n_head * head_dim
      kv_dim = hp.n_head_kv(il) * head_dim
      matmul_t0 = Time.instant
      q_rows, k_rows, v_rows = if v_qw = lw.attn_v_qw
                                 if rows = prompt_projection_many_matmul([lw.attn_q_qw, lw.attn_k_qw, v_qw], normed_rows, mask.prompt_len)
                                   {rows[0], rows[1], rows[2]}
                                 else
                                   {
                                     prompt_projection_matmul(lw.attn_q_qw, normed_rows, mask.prompt_len),
                                     prompt_projection_matmul(lw.attn_k_qw, normed_rows, mask.prompt_len),
                                     prompt_projection_matmul(v_qw, normed_rows, mask.prompt_len),
                                   }
                                 end
                               elsif rows = prompt_projection_many_matmul([lw.attn_q_qw, lw.attn_k_qw], normed_rows, mask.prompt_len)
                                 {rows[0], rows[1], rows[1].dup}
                               else
                                 k_rows = prompt_projection_matmul(lw.attn_k_qw, normed_rows, mask.prompt_len)
                                 {
                                   prompt_projection_matmul(lw.attn_q_qw, normed_rows, mask.prompt_len),
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
      projections = Array(AttentionProjection).new(mask.prompt_len) do |pos|
        copy_t0 = Time.instant
        q = q_rows[pos * q_dim, q_dim]
        k = k_rows[pos * kv_dim, kv_dim]
        v = v_rows[pos * kv_dim, kv_dim]
        proj = AttentionProjection.new(q, k, v, reused_k_as_v)
        copy_elapsed = (Time.instant - copy_t0).total_milliseconds
        copy_ms += copy_elapsed

        head_norm_t0 = Time.instant
        q_elapsed, k_elapsed, v_elapsed = normalize_attention_projection_timed!(proj, lw, hp, il)
        head_norm_elapsed = (Time.instant - head_norm_t0).total_milliseconds
        q_norm_ms += q_elapsed
        k_norm_ms += k_elapsed
        v_norm_ms += v_elapsed
        head_norm_ms += head_norm_elapsed
        assemble_ms += copy_elapsed + head_norm_elapsed
        rope_t0 = Time.instant
        table_elapsed, apply_elapsed = apply_rope_to_qk_timed!(proj, hp, il, pos, weights.rope_freqs)
        rope_elapsed = (Time.instant - rope_t0).total_milliseconds
        rope_table_ms += table_elapsed
        rope_apply_ms += apply_elapsed
        rope_ms += rope_elapsed
        proj
      end
      PromptProjectionTiming.new(projections, norm_ms, matmul_ms, assemble_ms, copy_ms, head_norm_ms, q_norm_ms, k_norm_ms, v_norm_ms, rope_ms, rope_table_ms, rope_apply_ms)
    end

    def prompt_projection_metal_enabled? : Bool
      ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL"]? == "1" &&
        ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL_OFF"]? != "1"
    end

    def prompt_projection_metal_min_batch : Int32
      (ENV["DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH"]? || "16").to_i
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
          timed = BoundedDenoiseStepTiming.new(update, timed.prediction_ms, timed.decode_stack_ms, timed.decode_qkv_ms, timed.decode_context_ms, timed.decode_attention_out_ms, timed.decode_shared_ffn_ms, timed.decode_moe_ffn_ms, timed.decode_combine_scale_ms, timed.output_head_ms, timed.update_ms, (Time.instant - regenerate_t0).total_milliseconds)
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
          timed = BoundedDenoiseStepTiming.new(update, timed.prediction_ms, timed.decode_stack_ms, timed.decode_qkv_ms, timed.decode_context_ms, timed.decode_attention_out_ms, timed.decode_shared_ffn_ms, timed.decode_moe_ffn_ms, timed.decode_combine_scale_ms, timed.output_head_ms, timed.update_ms, (Time.instant - regenerate_t0).total_milliseconds)
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
        decode_attention_out_ms: timing ? timing.not_nil!.decode_attention_out_ms : 0.0,
        decode_shared_ffn_ms: timing ? timing.not_nil!.decode_shared_ffn_ms : 0.0,
        decode_moe_ffn_ms: timing ? timing.not_nil!.decode_moe_ffn_ms : 0.0,
        decode_combine_scale_ms: timing ? timing.not_nil!.decode_combine_scale_ms : 0.0,
        output_head_ms: timing ? timing.not_nil!.output_head_ms : 0.0,
        update_ms: timing ? timing.not_nil!.update_ms : 0.0,
        regenerate_ms: timing ? timing.not_nil!.regenerate_ms : 0.0,
        proposal_ms: proposal_ms,
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

      key_count = high - low + 1
      result = Array(Float32).new(q_dim, 0.0_f32)
      scores = Array(Float32).new(key_count, 0.0_f32)
      n_head.times do |h|
        kvh = h // heads_per_group
        q_off = h * head_dim
        key_count.times do |i|
          key_pos = low + i
          k_off = kvh * head_dim
          scores[i] = dot(query.q, q_off, keyspace[key_pos].k, k_off, head_dim)
        end
        Gemma4CPU.softmax_slice!(scores, 0, scores.size)

        out_off = h * head_dim
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
      end
      result
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
