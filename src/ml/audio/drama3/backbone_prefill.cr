require "./backbone_resident"
require "./resident_batch"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::Audio::Drama3
    class BackboneResident
      # Device-only tables in [token, channel] order. Observers may enqueue GPU
      # comparisons without introducing intermediate completion boundaries.
      property prefill_trace : Proc(String, Resident::Buffer, Nil)? = nil

      # A complete layer-major sequence producer. Only the temporal kernels stay
      # sequential within a layer; projections and MoE consume all token rows.
      # Callers bound the chunk size and retain/consume its output before reuse.
      def prefill(input : Resident::Buffer, rows : Int32) : Resident::Buffer
        raise ArgumentError.new("resident prefill row count/capacity") unless rows > 0 && @position.to_i64 + rows <= @capacity
        expected = rows.to_i64 * @config.hidden
        raise ArgumentError.new("resident prefill width") unless input.count == expected
        # This check happens before model encoding, including on foreign buffers.
        # ones is a context-owned allocation for the full state lifetime.
        raise ArgumentError.new("buffer belongs to a different resident context") unless input.owner.same?(@ones.owner)
        base = @position
        hidden = input
        @config.layers.times do |layer|
          prefix = "backbone.layers.#{layer}"
          x = @ctx.rms(hidden, weight(prefix + ".input_layernorm.weight"), @config.eps, heads: rows)
          attention = if (layer + 1) % @config.full_interval == 0
                        full_attention_rows(x, prefix + ".self_attn", layer, base, rows)
                      else
                        linear_attention_rows(x, prefix + ".linear_attn", layer, base, rows)
                      end
          hidden = @ctx.add(hidden, attention)
          trace_prefill(prefix + ".attention_residual", hidden, rows)
          x = @ctx.rms(hidden, weight(prefix + ".post_attention_layernorm.weight"), @config.eps, heads: rows)
          hidden = @ctx.add(hidden, moe_rows(x, prefix + ".mlp", layer, rows))
          @ctx.guard(hidden)
          trace_prefill(prefix + ".hidden", hidden, rows)
        end
        @position = base + rows
        result = @ctx.rms(hidden, weight("backbone.norm.weight"), @config.eps, heads: rows)
        @ctx.guard(result)
        trace_prefill("backbone.hidden", result, rows)
        result
      end

      # Isolated MoE measurement uses the same batch implementation as prefill.
      # Neither this method nor prefill submits, waits or reads an activation.
      def diagnose_moe_rows(input : Resident::Buffer, layer : Int32, rows : Int32) : Resident::Buffer
        raise ArgumentError.new("resident diagnostic layer") unless layer >= 0 && layer < @config.layers
        raise ArgumentError.new("resident diagnostic row width") unless rows > 0 && input.count.to_i64 == rows.to_i64 * @config.hidden
        raise ArgumentError.new("buffer belongs to a different resident context") unless input.owner.same?(@ones.owner)
        moe_rows(input, "backbone.layers.#{layer}.mlp", layer, rows)
      end

      private def trace_prefill(name : String, values : Resident::Buffer, rows : Int32) : Nil
        @prefill_trace.try(&.call(name, values))
        # Preserve single-row callback shapes for existing diagnostic callers.
        if @trace || @buffer_trace
          width = values.count // rows
          rows.times { |row| trace_value(name, values.slice(row * width, width)) }
        end
      end

      private def project_rows(name : String, x : Resident::Buffer, rows : Int32, reduction : String? = nil) : Resident::Buffer
        @ctx.matvec_rows(@store.matrix(name + ".weight"), x, rows, dense_reduction: reduction)
      end

      private def full_attention_rows(x : Resident::Buffer, prefix : String, layer : Int32, base : Int32, rows : Int32) : Resident::Buffer
        c = @config
        projected = project_rows(prefix + ".q_proj", x, rows)
        keys = project_rows(prefix + ".k_proj", x, rows)
        values = project_rows(prefix + ".v_proj", x, rows)
        width = c.heads * c.head_dim
        kv_width = c.kv_heads * c.head_dim
        outputs = @ctx.alloc(rows * width)
        rows.times do |row|
          q, gate = @ctx.alloc(width), @ctx.alloc(width)
          @ctx.dispatch("dr3_split_q_gate", SOURCE,
            [projected.slice(row * width * 2, width * 2), q, gate], [width.to_u64, c.head_dim.to_u64], width)
          q = @ctx.rms(q, weight(prefix + ".q_norm.weight"), c.eps, heads: c.heads)
          k = @ctx.rms(keys.slice(row * kv_width, kv_width), weight(prefix + ".k_norm.weight"), c.eps, heads: c.kv_heads)
          output = full_attention_ready(q, k, values.slice(row * kv_width, kv_width), gate, layer, base + row)
          @ctx.copy(output, outputs.slice(row * width, width))
        end
        project_rows(prefix + ".o_proj", outputs, rows)
      end

      private def linear_attention_rows(x : Resident::Buffer, prefix : String, layer : Int32, base : Int32, rows : Int32) : Resident::Buffer
        c = @config
        qkv = project_rows(prefix + ".in_proj_qkv", x, rows, @dense_reduction)
        z = project_rows(prefix + ".in_proj_z", x, rows, @dense_reduction)
        b = project_rows(prefix + ".in_proj_b", x, rows, @dense_reduction)
        a = project_rows(prefix + ".in_proj_a", x, rows, @dense_reduction)
        qkv_width = qkv.count // rows
        value_width = c.value_heads * c.value_dim
        outputs = @ctx.alloc(rows * value_width)
        rows.times do |row|
          output = linear_attention_ready(qkv.slice(row * qkv_width, qkv_width), z.slice(row * value_width, value_width),
            b.slice(row * c.value_heads, c.value_heads), a.slice(row * c.value_heads, c.value_heads), prefix, layer, base + row)
          @ctx.copy(output, outputs.slice(row * value_width, value_width))
        end
        project_rows(prefix + ".out_proj", outputs, rows)
      end

      private def moe_rows(x : Resident::Buffer, prefix : String, layer : Int32, rows : Int32) : Resident::Buffer
        gate_matrix = @store.matrix(prefix + ".switch_mlp.gate_proj.weight", 0)
        up_matrix = @store.matrix(prefix + ".switch_mlp.up_proj.weight", 0)
        down_matrix = @store.matrix(prefix + ".switch_mlp.down_proj.weight", 0)
        # Dense tiny fixtures and the explicit legacy expert switch retain the
        # original per-row expert route. Real affine checkpoints batch all pairs.
        unless @batch_experts && gate_matrix.scales && up_matrix.scales && down_matrix.scales
          result = @ctx.alloc(rows * @config.hidden)
          rows.times do |row|
            value = moe(x.slice(row * @config.hidden, @config.hidden), prefix, layer)
            @ctx.copy(value, result.slice(row * @config.hidden, @config.hidden))
          end
          return result
        end
        scores = project_rows(prefix + ".gate", x, rows)
        ids, weights = @ctx.route_rows(scores, rows, @config.experts, @config.top_k, @config.norm_topk)
        gate = @ctx.experts_rows(gate_matrix, x, ids, rows, @config.top_k)
        up = @ctx.experts_rows(up_matrix, x, ids, rows, @config.top_k)
        activation = @ctx.mul(@ctx.silu(gate), up)
        down = @ctx.experts_rows(down_matrix, activation, ids, rows, @config.top_k,
          per_rank_input: true, reduction: @expert_down_reduction)
        result = @ctx.expert_sum_rows(down, weights, rows, @config.hidden, @config.top_k)
        shared_gate = project_rows(prefix + ".shared_expert.gate_proj", x, rows)
        shared_up = project_rows(prefix + ".shared_expert.up_proj", x, rows)
        shared = project_rows(prefix + ".shared_expert.down_proj", @ctx.mul(@ctx.silu(shared_gate), shared_up), rows)
        shared_weight = project_rows(prefix + ".shared_expert_gate", x, rows)
        if @diagnostic_layer == layer
          # Observe immutable rank-major tables before result's in-place add.
          {"normalized_input" => x, "router_logits" => scores, "expert_ids" => ids,
           "selected_weights" => weights, "expert_gate_outputs" => gate, "expert_up_outputs" => up,
           "expert_activation_outputs" => activation, "expert_down_outputs" => down,
           "shared_expert_output" => shared, "shared_gate_logits" => shared_weight}.each do |name, value|
            @prefill_trace.try(&.call(prefix + "." + name, value))
          end
          if callback = @prefill_trace
            stable = @ctx.alloc(result.count)
            @ctx.copy(result, stable)
            callback.call(prefix + ".routed_sum", stable)
          end
        end
        @ctx.weighted_add_rows!(result, shared, shared_weight, rows, @config.hidden)
        @prefill_trace.try(&.call(prefix + ".moe_output", result)) if @diagnostic_layer == layer
        result
      end
    end
  end
{% end %}
