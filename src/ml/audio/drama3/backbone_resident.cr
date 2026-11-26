require "./backbone"
require "./resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3
  # The CPU only schedules commands. Activations, routing decisions, and both
  # temporal attention states remain in device allocations between rows.
  class BackboneResident
    SOURCE = {{ read_file("#{__DIR__}/kernels/backbone_resident.metal") }}
    MAX_CAPACITY = 4096
    getter config : BackboneConfig, position = 0
    getter capacity : Int32, maximum_capacity : Int32
    property trace : Proc(String, Vector, Nil)?
    property buffer_trace : Proc(String, Resident::Buffer, Nil)?
    # Device-only diagnostic observers. They are nil on the ordinary inference
    # path; the detailed MoE callback is additionally restricted to one layer.
    property moe_trace : Proc(String, Resident::Buffer, Nil)?
    property layer_buffer_trace : Proc(String, Resident::Buffer, Nil)?
    property moe_trace_auxiliary = false
    getter diagnostic_layer : Int32?
    @keys : Array(Resident::Buffer?)
    @values : Array(Resident::Buffer?)
    @full_attention_scores : Resident::Buffer?
    @wide_attention : Bool
    @conv : Array(Resident::Buffer?)
    @delta : Array(Resident::Buffer?)
    @delta_kernel : String
    @batch_experts : Bool
    getter last_moe_batched = false
    getter dense_reduction : String
    getter dense_fusion_enabled : Bool
    getter expert_down_reduction : String
    getter batch_experts : Bool

    def initialize(@config : BackboneConfig, @store : WeightStore, @ctx : Resident::Context,
                   capacity : Int32, maximum_capacity : Int32 = capacity)
      raise ArgumentError.new("resident backbone capacity must be 1..#{MAX_CAPACITY}") unless capacity.in?(1..MAX_CAPACITY)
      raise ArgumentError.new("resident backbone maximum capacity must be #{capacity}..#{MAX_CAPACITY}") unless maximum_capacity.in?(capacity..MAX_CAPACITY)
      @capacity = capacity
      @maximum_capacity = maximum_capacity
      # Keep reduction order stable for the request even when its backing grows
      # through the 1024-token threshold.
      @wide_attention = @maximum_capacity > 1024
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
      dense_fusion = ENV["DRAMA3_BACKBONE_DENSE_FUSION"]? || "0"
      raise ArgumentError.new("DRAMA3_BACKBONE_DENSE_FUSION must be 0 or 1") unless dense_fusion.in?("0", "1")
      @dense_fusion_enabled = dense_fusion == "1"
      @expert_down_reduction = ENV["DRAMA3_EXPERT_DOWN_REDUCTION"]? || "native"
      unless @expert_down_reduction.in?("native", "mlx_q4")
        raise ArgumentError.new("DRAMA3_EXPERT_DOWN_REDUCTION must be native or mlx_q4")
      end
      @keys = Array(Resident::Buffer?).new(@config.layers, nil)
      @values = Array(Resident::Buffer?).new(@config.layers, nil)
      @full_attention_scores = nil
      @conv = Array(Resident::Buffer?).new(@config.layers, nil)
      @delta = Array(Resident::Buffer?).new(@config.layers, nil)
      @trace = nil
      @buffer_trace = nil
      @moe_trace = nil
      @layer_buffer_trace = nil
      @moe_trace_auxiliary = false
      @diagnostic_layer = nil
      @ones = @ctx.upload(Array(Float32).new(@config.key_dim, 1_f32))
      if @wide_attention && @config.layers >= @config.full_interval
        score_count = checked_buffer_count("full-attention score scratch", [@capacity, @config.heads])
        @full_attention_scores = @ctx.alloc_persistent(score_count)
      end
      @config.layers.times do |i|
        if (i + 1) % @config.full_interval == 0
          cache_count = checked_buffer_count("full-attention KV cache", [@capacity, @config.kv_heads, @config.head_dim])
          @keys[i] = @ctx.alloc_persistent(cache_count)
          @values[i] = @ctx.alloc_persistent(cache_count)
        else
          width = 2 * @config.key_heads * @config.key_dim + @config.value_heads * @config.value_dim
          @conv[i] = @ctx.zeros(width * @config.conv_kernel)
          @delta[i] = @ctx.zeros(@config.value_heads * @config.value_dim * @config.key_dim)
        end
      end
    end

    # Grow only the full-attention backing. Recurrent convolution/delta state
    # has fixed shape and survives unchanged. All new storage is allocated
    # before any prefix copy is queued, then the new backing becomes visible
    # only after every copy has been encoded.
    def grow_capacity!(new_capacity : Int32) : Nil
      raise ArgumentError.new("resident backbone growth must increase capacity") unless new_capacity > @capacity
      raise ArgumentError.new("resident backbone growth exceeds maximum capacity") unless new_capacity <= @maximum_capacity

      replacement_keys = Array(Resident::Buffer?).new(@config.layers, nil)
      replacement_values = Array(Resident::Buffer?).new(@config.layers, nil)
      key_copies = [] of Tuple(Resident::Buffer, Resident::Buffer)
      value_copies = [] of Tuple(Resident::Buffer, Resident::Buffer)
      cache_count = checked_buffer_count("full-attention KV cache", [new_capacity, @config.kv_heads, @config.head_dim])
      prefix_count = @position > 0 ? checked_buffer_count("full-attention KV prefix", [@position, @config.kv_heads, @config.head_dim]) : 0

      @config.layers.times do |i|
        next unless (i + 1) % @config.full_interval == 0
        old_keys = @keys[i].not_nil!
        old_values = @values[i].not_nil!
        new_keys = @ctx.alloc_persistent(cache_count)
        new_values = @ctx.alloc_persistent(cache_count)
        replacement_keys[i] = new_keys
        replacement_values[i] = new_values
        if prefix_count > 0
          key_copies << {old_keys.slice(0, prefix_count), new_keys.slice(0, prefix_count)}
          value_copies << {old_values.slice(0, prefix_count), new_values.slice(0, prefix_count)}
        end
      end

      replacement_scores = if @wide_attention && @config.layers >= @config.full_interval
                             @ctx.alloc_persistent(checked_buffer_count("full-attention score scratch", [new_capacity, @config.heads]))
                           else
                             nil
                           end

      key_copies.each { |source, target| @ctx.copy(source, target) }
      value_copies.each { |source, target| @ctx.copy(source, target) }
      @keys = replacement_keys
      @values = replacement_values
      @full_attention_scores = replacement_scores
      @capacity = new_capacity
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

    private def checked_buffer_count(name : String, factors : Array(Int32)) : Int32
      count = 1_i64
      factors.each do |factor|
        unless factor > 0 && count <= Int32::MAX.to_i64 // factor.to_i64
          raise ArgumentError.new("resident #{name} exceeds Int32 element capacity")
        end
        count *= factor.to_i64
      end
      bytes = count * sizeof(Float32).to_i64
      limit = MetalBackend.max_buffer_length
      unless limit > 0 && bytes <= limit
        raise ArgumentError.new("resident #{name} requires #{bytes} Metal bytes; device limit is #{limit}")
      end
      count.to_i32
    end

    private def project(name : String, x : Resident::Buffer) : Resident::Buffer
      @ctx.matvec(@store.matrix(name + ".weight"), x)
    end

    private def trace_value(name : String, x : Resident::Buffer) : Nil
      @buffer_trace.try(&.call(name, x))
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
      gate_matrix = @store.matrix(prefix + ".gate_proj.weight")
      up_matrix = @store.matrix(prefix + ".up_proj.weight")
      swiglu = @ctx.matvec_swiglu(gate_matrix, up_matrix, x)
      project(prefix + ".down_proj", swiglu[:output])
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
                   swiglu = @ctx.experts_swiglu(gate_matrix, up_matrix, x, ids, capture_projections: !!callback)
                   activation = swiglu[:output]
                   projected = @ctx.experts_down_sum(down_matrix, activation, ids, routing.slice(@config.top_k, @config.top_k),
                     reduction: @expert_down_reduction, capture_down: !!callback)
                   combined = projected[:output]
                   if callback
                     gate = swiglu[:gate].not_nil!
                     up = swiglu[:up].not_nil!
                     down = projected[:down].not_nil!
                     @config.top_k.times do |rank|
                       trace_moe_buffer(callback, "expert_gate_outputs.rank_#{rank}", gate.slice(rank * gate_matrix.rows, gate_matrix.rows))
                       trace_moe_buffer(callback, "expert_up_outputs.rank_#{rank}", up.slice(rank * up_matrix.rows, up_matrix.rows))
                       trace_moe_buffer(callback, "expert_activation_outputs.rank_#{rank}", activation.slice(rank * down_matrix.cols, down_matrix.cols))
                       trace_moe_buffer(callback, "expert_down_outputs.rank_#{rank}", down.slice(rank * down_matrix.rows, down_matrix.rows))
                     end
                   end
                   if callback
                     down = projected[:down].not_nil!
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
                     down = @ctx.expert(down_matrix, activation, ids, rank, reduction: @expert_down_reduction)
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
      project(prefix + ".o_proj", full_attention_ready(q, k, v, gate, layer, @position))
    end

    # Shared temporal kernel path for scalar decode and layer-major prefill.
    private def full_attention_ready(q : Resident::Buffer, k : Resident::Buffer, v : Resident::Buffer,
                                     gate : Resident::Buffer, layer : Int32, position : Int32) : Resident::Buffer
      c = @config
      @ctx.rope!(q, c.heads, c.head_dim, c.rotary_dim, position, c.rope_base, false)
      @ctx.rope!(k, c.kv_heads, c.head_dim, c.rotary_dim, position, c.rope_base, false)
      keys, values = @keys[layer].not_nil!, @values[layer].not_nil!
      @ctx.copy(k, keys.slice(position * k.count, k.count))
      @ctx.copy(v, values.slice(position * v.count, v.count))
      output = @ctx.alloc(q.count)
      enc = @ctx.encoder
      wide = @wide_attention
      kernel = wide ? "dr3_full_attention_wide" : "dr3_full_attention"
      enc.set_pipeline(@ctx.pipeline(kernel, SOURCE))
      [q, keys, values, gate, output].each_with_index { |b, i| enc.set_buffer(b.buffer, i, offset: b.offset) }
      if wide
        scores = @full_attention_scores.not_nil!
        enc.set_buffer(scores.buffer, 5, offset: scores.offset)
        p = [c.heads.to_u64, c.kv_heads.to_u64, c.head_dim.to_u64, (position + 1).to_u64, @capacity.to_u64]
        enc.set_bytes(p.to_unsafe.as(Void*), (p.size * 8).to_i32, 6)
      else
        p = [c.heads.to_u64, c.kv_heads.to_u64, c.head_dim.to_u64, (position + 1).to_u64]
        enc.set_bytes(p.to_unsafe.as(Void*), (p.size * 8).to_i32, 5)
      end
      enc.dispatch_threadgroups({c.heads, 1, 1}, {64, 1, 1})
      output
    end

    private def linear_attention(x : Resident::Buffer, prefix : String, layer : Int32) : Resident::Buffer
      c = @config
      if @dense_fusion_enabled
        qkv, z, b, a = @ctx.matvec_four({
          @store.matrix(prefix + ".in_proj_qkv.weight"), @store.matrix(prefix + ".in_proj_z.weight"),
          @store.matrix(prefix + ".in_proj_b.weight"), @store.matrix(prefix + ".in_proj_a.weight"),
        }, x, dense_reduction: @dense_reduction, fused: true)
      else
        qkv = @ctx.matvec(@store.matrix(prefix + ".in_proj_qkv.weight"), x, dense_reduction: @dense_reduction)
        z = @ctx.matvec(@store.matrix(prefix + ".in_proj_z.weight"), x, dense_reduction: @dense_reduction)
        b = @ctx.matvec(@store.matrix(prefix + ".in_proj_b.weight"), x, dense_reduction: @dense_reduction)
        a = @ctx.matvec(@store.matrix(prefix + ".in_proj_a.weight"), x, dense_reduction: @dense_reduction)
      end
      project(prefix + ".out_proj", linear_attention_ready(qkv, z, b, a, prefix, layer, @position))
    end

    private def linear_attention_ready(qkv : Resident::Buffer, z : Resident::Buffer, b : Resident::Buffer,
                                       a : Resident::Buffer, prefix : String, layer : Int32, position : Int32) : Resident::Buffer
      c = @config
      history = @conv[layer].not_nil!
      @ctx.copy(qkv, history.slice((position % c.conv_kernel) * qkv.count, qkv.count))
      convolved = @ctx.alloc(qkv.count)
      @ctx.dispatch("dr3_delta_conv", SOURCE, [history, weight(prefix + ".conv1d.weight"), convolved], [qkv.count.to_u64, c.conv_kernel.to_u64, position.to_u64], qkv.count)
      kw = c.key_heads * c.key_dim
      q = @ctx.rms(convolved.slice(0, kw), @ones, 1e-6, heads: c.key_heads)
      k = @ctx.rms(convolved.slice(kw, kw), @ones, 1e-6, heads: c.key_heads)
      @ctx.dispatch("dr3_delta_scale", SOURCE, [q, k], [kw.to_u64, c.key_dim.to_u64], kw)
      v = convolved.slice(2 * kw, c.value_heads * c.value_dim)
      output = @ctx.alloc(v.count)
      @ctx.dispatch(@delta_kernel, SOURCE, [q, k, v, a, b, weight(prefix + ".A_log"), weight(prefix + ".dt_bias"), @delta[layer].not_nil!, output], [c.value_heads.to_u64, c.value_dim.to_u64, c.key_heads.to_u64, c.key_dim.to_u64], v.count)
      @ctx.gdn_post(output, z, weight(prefix + ".norm.weight"), c.eps, heads: c.value_heads)
    end
  end
end
{% end %}
