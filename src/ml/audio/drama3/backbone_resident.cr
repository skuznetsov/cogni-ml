require "./backbone"
require "./resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3
  # The CPU only schedules commands. Activations, routing decisions, and both
  # temporal attention states remain in device allocations between rows.
  class BackboneResident
    SOURCE = {{ read_file("#{__DIR__}/kernels/backbone_resident.metal") }}
    getter config : BackboneConfig, position = 0
    property trace : Proc(String, Vector, Nil)?
    # Device-only diagnostic observers. They are nil on the ordinary inference
    # path; the detailed MoE callback is additionally restricted to one layer.
    property moe_trace : Proc(String, Resident::Buffer, Nil)?
    property layer_buffer_trace : Proc(String, Resident::Buffer, Nil)?
    property moe_trace_auxiliary = false
    getter diagnostic_layer : Int32?
    @keys : Array(Resident::Buffer?)
    @values : Array(Resident::Buffer?)
    @conv : Array(Resident::Buffer?)
    @delta : Array(Resident::Buffer?)
    @delta_kernel : String
    @batch_experts : Bool
    getter last_moe_batched = false
    getter dense_reduction : String
    getter batch_experts : Bool

    def initialize(@config : BackboneConfig, @store : WeightStore, @ctx : Resident::Context, @capacity : Int32)
      raise ArgumentError.new("resident backbone capacity must be 1..1024") unless @capacity.in?(1..1024)
      raise ArgumentError.new("resident router supports at most 512 experts") unless @config.experts <= 512
      layout = ENV["DRAMA3_DELTA_TRANSPOSE"]? || "1"
      raise ArgumentError.new("DRAMA3_DELTA_TRANSPOSE must be 0 or 1") unless layout.in?("0", "1")
      # Freeze the layout for the entire recurrent state lifetime. The legacy
      # route is retained for matched performance/parity probes and rollback.
      @delta_kernel = layout == "1" ? "dr3_delta_step_transposed" : "dr3_delta_step"
      batch = ENV["DRAMA3_MOE_BATCH"]? || "1"
      raise ArgumentError.new("DRAMA3_MOE_BATCH must be 0 or 1") unless batch.in?("0", "1")
      @batch_experts = batch == "1"
      @dense_reduction = ENV["DRAMA3_BACKBONE_DENSE_REDUCTION"]? || "tree"
      unless @dense_reduction.in?("tree", "simd")
        raise ArgumentError.new("DRAMA3_BACKBONE_DENSE_REDUCTION must be tree or simd")
      end
      @keys = Array(Resident::Buffer?).new(@config.layers, nil)
      @values = Array(Resident::Buffer?).new(@config.layers, nil)
      @conv = Array(Resident::Buffer?).new(@config.layers, nil)
      @delta = Array(Resident::Buffer?).new(@config.layers, nil)
      @trace = nil
      @moe_trace = nil
      @layer_buffer_trace = nil
      @moe_trace_auxiliary = false
      @diagnostic_layer = nil
      @ones = @ctx.upload(Array(Float32).new(@config.key_dim, 1_f32))
      @config.layers.times do |i|
        if (i + 1) % @config.full_interval == 0
          @keys[i] = @ctx.alloc(@capacity * @config.kv_heads * @config.head_dim)
          @values[i] = @ctx.alloc(@capacity * @config.kv_heads * @config.head_dim)
        else
          width = 2 * @config.key_heads * @config.key_dim + @config.value_heads * @config.value_dim
          @conv[i] = @ctx.zeros(width * @config.conv_kernel)
          @delta[i] = @ctx.zeros(@config.value_heads * @config.value_dim * @config.key_dim)
        end
      end
    end

    def diagnostic_layer=(layer : Int32?) : Int32?
      if value = layer
        raise ArgumentError.new("resident diagnostic layer") unless value >= 0 && value < @config.layers
      end
      @diagnostic_layer = layer
    end

    def embedding(token : Resident::Buffer) : Resident::Buffer
      @ctx.embedding(@store.matrix("backbone.embed_tokens.weight"), token)
    end

    def step(input : Resident::Buffer) : Resident::Buffer
      Profile.measure("backbone.step", category: "backbone") do
        raise ArgumentError.new("resident backbone capacity exceeded") unless @position < @capacity
        raise ArgumentError.new("resident backbone width") unless input.count == @config.hidden
        hidden = input
        @config.layers.times do |i|
          prefix = "backbone.layers.#{i}"
          x = @ctx.rms(hidden, weight(prefix + ".input_layernorm.weight"), @config.eps)
          attention = if (i + 1) % @config.full_interval == 0
                        full_attention(x, prefix + ".self_attn", i)
                      else
                        linear_attention(x, prefix + ".linear_attn", i)
                      end
          hidden = @ctx.add(hidden, attention)
          trace_value("backbone.layers.#{i}.attention_residual", hidden)
          trace_layer_buffer(i, "attention_residual", hidden)
          x = @ctx.rms(hidden, weight(prefix + ".post_attention_layernorm.weight"), @config.eps)
          hidden = @ctx.add(hidden, moe(x, prefix + ".mlp", i))
          @ctx.guard(hidden)
          trace_value("backbone.layers.#{i}.hidden", hidden)
          trace_layer_buffer(i, "hidden_output", hidden)
        end
        @position += 1
        result = @ctx.rms(hidden, weight("backbone.norm.weight"), @config.eps)
        @ctx.guard(result)
        trace_value("backbone.hidden", result)
        if layer = @diagnostic_layer
          @layer_buffer_trace.try(&.call("backbone_output", result))
        end
        # Component profiling includes GPU completion. Normal inference can
        # compose the next tower operation in the same serial command buffer.
        @ctx.finish if Profile.enabled?
        result
      end
    end

    private def weight(name : String) : Resident::Buffer
      @ctx.tensor(@store.tensor(name))
    end

    private def project(name : String, x : Resident::Buffer) : Resident::Buffer
      @ctx.matvec(@store.matrix(name + ".weight"), x)
    end

    private def trace_value(name : String, x : Resident::Buffer) : Nil
      if callback = @trace
        callback.call(name, @ctx.read(x, "diagnostic"))
      end
    end

    private def trace_layer_buffer(layer : Int32, name : String, value : Resident::Buffer) : Nil
      if @diagnostic_layer == layer
        @layer_buffer_trace.try(&.call(name, value))
      end
    end

    private def trace_moe_buffer(callback : Proc(String, Resident::Buffer, Nil)?, name : String, value : Resident::Buffer, snapshot : Bool = false) : Nil
      if trace = callback
        if snapshot
          stable = @ctx.alloc(value.count)
          @ctx.copy(value, stable)
          trace.call(name, stable)
        else
          trace.call(name, value)
        end
      end
    end

    # Replay one already-normalized activation through the selected MoE layer.
    # This does not advance the recurrent attention position or mutate caches.
    def diagnose_moe(input : Resident::Buffer, layer : Int32) : Resident::Buffer
      raise ArgumentError.new("resident diagnostic layer") unless layer >= 0 && layer < @config.layers
      raise ArgumentError.new("resident diagnostic width") unless input.count == @config.hidden
      raise ArgumentError.new("resident diagnostic layer must be selected") unless @diagnostic_layer == layer
      moe(input, "backbone.layers.#{layer}.mlp", layer)
    end

    private def mlp(x : Resident::Buffer, prefix : String) : Resident::Buffer
      gate = project(prefix + ".gate_proj", x)
      up = project(prefix + ".up_proj", x)
      project(prefix + ".down_proj", @ctx.mul(@ctx.silu(gate), up))
    end

    private def moe(x : Resident::Buffer, prefix : String, layer : Int32) : Resident::Buffer
      Profile.measure("backbone.moe", category: "moe") do
        callback = @diagnostic_layer == layer ? @moe_trace : nil
        trace_moe_buffer(callback, "normalized_input", x)
        scores = project(prefix + ".gate", x)
        trace_moe_buffer(callback, "router_logits", scores)
        routing = @ctx.alloc(@config.top_k * 2)
        @ctx.dispatch("dr3_route", SOURCE, [scores, routing], [scores.count.to_u64, @config.top_k.to_u64, @config.norm_topk ? 1_u64 : 0_u64], 1)
        ids = routing.slice(0, @config.top_k)
        trace_moe_buffer(callback, "expert_ids_argpartition_order", ids)
        trace_moe_buffer(callback, "selected_weights", routing.slice(@config.top_k, @config.top_k))
        gate_matrix = @store.matrix(prefix + ".switch_mlp.gate_proj.weight", 0)
        up_matrix = @store.matrix(prefix + ".switch_mlp.up_proj.weight", 0)
        down_matrix = @store.matrix(prefix + ".switch_mlp.down_proj.weight", 0)
        batched = @batch_experts && !!gate_matrix.scales && !!up_matrix.scales && !!down_matrix.scales
        @last_moe_batched = batched if callback
        result = if batched
                   gate = @ctx.experts(gate_matrix, x, ids)
                   up = @ctx.experts(up_matrix, x, ids)
                   activation = @ctx.mul(@ctx.silu(gate), up)
                   down = @ctx.experts(down_matrix, activation, ids, per_rank_input: true)
                   if callback
                     @config.top_k.times do |rank|
                       trace_moe_buffer(callback, "expert_gate_outputs.rank_#{rank}", gate.slice(rank * gate_matrix.rows, gate_matrix.rows))
                       trace_moe_buffer(callback, "expert_up_outputs.rank_#{rank}", up.slice(rank * up_matrix.rows, up_matrix.rows))
                       trace_moe_buffer(callback, "expert_activation_outputs.rank_#{rank}", activation.slice(rank * down_matrix.cols, down_matrix.cols))
                       trace_moe_buffer(callback, "expert_down_outputs.rank_#{rank}", down.slice(rank * down_matrix.rows, down_matrix.rows))
                     end
                   end
                   combined = @ctx.alloc(@config.hidden)
                   @ctx.dispatch("dr3_expert_sum", SOURCE, [down, routing, combined], [combined.count.to_u64, @config.top_k.to_u64], combined.count)
                   if callback
                     trace_moe_buffer(callback, "routed_sum_official", combined, snapshot: true)
                     if @moe_trace_auxiliary
                       auxiliary_prefix = @ctx.zeros(@config.hidden)
                       @config.top_k.times do |rank|
                         rank_down = down.slice(rank * down_matrix.rows, down_matrix.rows)
                         rank_weight = routing.slice(@config.top_k + rank, 1)
                         weighted = @ctx.zeros(@config.hidden)
                         @ctx.dispatch("dr3_weighted_add", SOURCE, [weighted, rank_down, rank_weight], [@config.hidden.to_u64, 0_u64], @config.hidden)
                         trace_moe_buffer(callback, "expert_weighted_outputs.rank_#{rank}", weighted)
                         @ctx.dispatch("dr3_weighted_add", SOURCE, [auxiliary_prefix, rank_down, rank_weight], [@config.hidden.to_u64, 0_u64], @config.hidden)
                         snapshot = @ctx.alloc(@config.hidden)
                         @ctx.copy(auxiliary_prefix, snapshot)
                         trace_moe_buffer(callback, "diagnostic_prefix_sums_argpartition_order.rank_#{rank}", snapshot)
                       end
                     end
                   end
                   combined
                 else
                   combined = @ctx.zeros(@config.hidden)
                   @config.top_k.times do |rank|
                     gate = @ctx.expert(gate_matrix, x, ids, rank)
                     up = @ctx.expert(up_matrix, x, ids, rank)
                     activation = @ctx.mul(@ctx.silu(gate), up)
                     down = @ctx.expert(down_matrix, activation, ids, rank)
                     if callback
                       trace_moe_buffer(callback, "expert_gate_outputs.rank_#{rank}", gate)
                       trace_moe_buffer(callback, "expert_up_outputs.rank_#{rank}", up)
                       trace_moe_buffer(callback, "expert_activation_outputs.rank_#{rank}", activation)
                       trace_moe_buffer(callback, "expert_down_outputs.rank_#{rank}", down)
                       if @moe_trace_auxiliary
                         weighted = @ctx.zeros(@config.hidden)
                         rank_weight = routing.slice(@config.top_k + rank, 1)
                         @ctx.dispatch("dr3_weighted_add", SOURCE, [weighted, down, rank_weight], [@config.hidden.to_u64, 0_u64], @config.hidden)
                         trace_moe_buffer(callback, "expert_weighted_outputs.rank_#{rank}", weighted)
                       end
                     end
                     @ctx.dispatch("dr3_weighted_add", SOURCE, [combined, down, routing.slice(@config.top_k + rank, 1)], [combined.count.to_u64, 0_u64], combined.count)
                     if callback
                       trace_moe_buffer(callback, "routed_prefix_sums_argpartition_order.rank_#{rank}", combined, snapshot: true)
                     end
                   end
                   trace_moe_buffer(callback, "routed_sum_official", combined, snapshot: true)
                   combined
                 end
        shared = mlp(x, prefix + ".shared_expert")
        trace_moe_buffer(callback, "shared_expert_output", shared)
        shared_gate = project(prefix + ".shared_expert_gate", x)
        trace_moe_buffer(callback, "shared_gate_logits", shared_gate)
        if callback && @moe_trace_auxiliary
          shared_weighted = @ctx.zeros(@config.hidden)
          @ctx.dispatch("dr3_weighted_add", SOURCE, [shared_weighted, shared, shared_gate], [@config.hidden.to_u64, 1_u64], @config.hidden)
          trace_moe_buffer(callback, "shared_weighted_output", shared_weighted)
        end
        @ctx.dispatch("dr3_weighted_add", SOURCE, [result, shared, shared_gate], [result.count.to_u64, 1_u64], result.count)
        trace_moe_buffer(callback, "moe_output_official", result)
        result
      end
    end

    private def full_attention(x : Resident::Buffer, prefix : String, layer : Int32) : Resident::Buffer
      c = @config
      projected = project(prefix + ".q_proj", x)
      q = @ctx.alloc(c.heads * c.head_dim)
      gate = @ctx.alloc(q.count)
      @ctx.dispatch("dr3_split_q_gate", SOURCE, [projected, q, gate], [q.count.to_u64, c.head_dim.to_u64], q.count)
      q = @ctx.rms(q, weight(prefix + ".q_norm.weight"), c.eps, heads: c.heads)
      k = @ctx.rms(project(prefix + ".k_proj", x), weight(prefix + ".k_norm.weight"), c.eps, heads: c.kv_heads)
      v = project(prefix + ".v_proj", x)
      @ctx.rope!(q, c.heads, c.head_dim, c.rotary_dim, @position, c.rope_base, false)
      @ctx.rope!(k, c.kv_heads, c.head_dim, c.rotary_dim, @position, c.rope_base, false)
      keys, values = @keys[layer].not_nil!, @values[layer].not_nil!
      @ctx.copy(k, keys.slice(@position * k.count, k.count))
      @ctx.copy(v, values.slice(@position * v.count, v.count))
      output = @ctx.alloc(q.count)
      enc = @ctx.encoder
      enc.set_pipeline(ML::Metal::PipelineCache.get_or_compile("dr3_full_attention", SOURCE))
      [q, keys, values, gate, output].each_with_index { |b, i| enc.set_buffer(b.buffer, i, offset: b.offset) }
      p = [c.heads.to_u64, c.kv_heads.to_u64, c.head_dim.to_u64, (@position + 1).to_u64]
      enc.set_bytes(p.to_unsafe.as(Void*), (p.size * 8).to_i32, 5)
      enc.dispatch_threadgroups({c.heads, 1, 1}, {64, 1, 1})
      project(prefix + ".o_proj", output)
    end

    private def linear_attention(x : Resident::Buffer, prefix : String, layer : Int32) : Resident::Buffer
      c = @config
      # Keep the reduction choice local to this measured shared-input group.
      # The returned device allocations feed the existing temporal/gating path.
      qkv = @ctx.matvec(@store.matrix(prefix + ".in_proj_qkv.weight"), x, dense_reduction: @dense_reduction)
      z = @ctx.matvec(@store.matrix(prefix + ".in_proj_z.weight"), x, dense_reduction: @dense_reduction)
      b = @ctx.matvec(@store.matrix(prefix + ".in_proj_b.weight"), x, dense_reduction: @dense_reduction)
      a = @ctx.matvec(@store.matrix(prefix + ".in_proj_a.weight"), x, dense_reduction: @dense_reduction)
      history = @conv[layer].not_nil!
      @ctx.copy(qkv, history.slice((@position % c.conv_kernel) * qkv.count, qkv.count))
      convolved = @ctx.alloc(qkv.count)
      @ctx.dispatch("dr3_delta_conv", SOURCE, [history, weight(prefix + ".conv1d.weight"), convolved], [qkv.count.to_u64, c.conv_kernel.to_u64, @position.to_u64], qkv.count)
      kw = c.key_heads * c.key_dim
      q = @ctx.rms(convolved.slice(0, kw), @ones, 1e-6, heads: c.key_heads)
      k = @ctx.rms(convolved.slice(kw, kw), @ones, 1e-6, heads: c.key_heads)
      @ctx.dispatch("dr3_delta_scale", SOURCE, [q, k], [kw.to_u64, c.key_dim.to_u64], kw)
      v = convolved.slice(2 * kw, c.value_heads * c.value_dim)
      output = @ctx.alloc(v.count)
      @ctx.dispatch(@delta_kernel, SOURCE, [q, k, v, a, b, weight(prefix + ".A_log"), weight(prefix + ".dt_bias"), @delta[layer].not_nil!, output], [c.value_heads.to_u64, c.value_dim.to_u64, c.key_heads.to_u64, c.key_dim.to_u64], v.count)
      normalized = @ctx.rms(output, weight(prefix + ".norm.weight"), c.eps, heads: c.value_heads)
      project(prefix + ".out_proj", @ctx.mul(normalized, @ctx.silu(z)))
    end
  end
end
{% end %}
