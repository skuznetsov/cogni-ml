require "./native"

# Text-only Qwen3.5-MoE backbone. CPU single-row decode is deliberately used
# for both prompt prefill and generation, giving one auditable cache path.
# Architecture reference: Apple mlx-lm / cguldogan mlx-audio qwen3_5.py (MIT).
module ML::Audio::Drama3
  class BackboneConfig
    getter hidden : Int32, layers : Int32, heads : Int32, kv_heads : Int32, head_dim : Int32
    getter experts : Int32, top_k : Int32, full_interval : Int32
    getter key_heads : Int32, value_heads : Int32, key_dim : Int32, value_dim : Int32, conv_kernel : Int32
    getter eps : Float64, rope_base : Float64, rotary_dim : Int32, norm_topk : Bool

    def initialize(config : JSON::Any)
      c = config["text_config"]? || config
      @hidden = c["hidden_size"].as_i
      @layers = c["num_hidden_layers"].as_i
      @heads = c["num_attention_heads"].as_i
      @kv_heads = c["num_key_value_heads"].as_i
      @head_dim = c["head_dim"]?.try(&.as_i) || (@heads > 0 ? @hidden // @heads : 0)
      @experts = c["num_experts"].as_i
      @top_k = c["num_experts_per_tok"].as_i
      @full_interval = c["full_attention_interval"]?.try(&.as_i) || 4
      @key_heads = c["linear_num_key_heads"].as_i
      @value_heads = c["linear_num_value_heads"].as_i
      @key_dim = c["linear_key_head_dim"].as_i
      @value_dim = c["linear_value_head_dim"].as_i
      @conv_kernel = c["linear_conv_kernel_dim"]?.try(&.as_i) || 4
      @eps = c["rms_norm_eps"]?.try(&.as_f) || 1e-6
      rope = c["rope_parameters"]?
      @rope_base = rope.try { |r| r["rope_theta"]?.try(&.as_f) } || 100000.0
      fraction = rope.try { |r| r["partial_rotary_factor"]?.try(&.as_f) } || 0.25
      @rotary_dim = (@head_dim * fraction).to_i
      @norm_topk = c["norm_topk_prob"]?.try(&.as_bool) != false
      raise ArgumentError.new("invalid backbone configuration") unless @hidden > 0 && @layers > 0 && @heads > 0 && @kv_heads > 0 && @head_dim > 0 && @heads % @kv_heads == 0 && @key_heads > 0 && @value_heads > 0 && @value_heads % @key_heads == 0 && @key_dim > 0 && @value_dim > 0 && @conv_kernel > 0 && @top_k > 0 && @top_k <= @experts && @full_interval > 0
      raise ArgumentError.new("invalid backbone normalization or RoPE") unless @eps.finite? && @eps > 0 && @rope_base.finite? && @rope_base > 0 && @rotary_dim > 0 && @rotary_dim <= @head_dim && @rotary_dim.even?
      raise ArgumentError.new("attention bias unsupported") if c["attention_bias"]?.try(&.as_bool) == true
      if types = c["layer_types"]?
        expected = Array(String).new(@layers) { |i| (i + 1) % @full_interval == 0 ? "full_attention" : "linear_attention" }
        raise ArgumentError.new("unsupported backbone layer schedule") unless types.as_a.map(&.as_s) == expected
      end
    end
  end

  class BackboneState
    getter keys = [] of Vector
    getter values = [] of Vector
    getter conv = [] of Vector
    getter delta : Vector

    def initialize(config : BackboneConfig)
      @delta = Array(Float32).new(config.value_heads * config.value_dim * config.key_dim, 0_f32)
    end
  end

  class Backbone
    getter config : BackboneConfig
    getter position = 0
    getter states : Array(BackboneState)
    property trace : Proc(String, Vector, Nil)?

    def initialize(@config : BackboneConfig, @store : WeightStore)
      @states = Array(BackboneState).new(@config.layers) { BackboneState.new(@config) }
      @trace = nil
    end

    def embedding(token : Int32) : Vector
      @store.matrix("backbone.embed_tokens.weight").row(token)
    end

    def step(input : Vector) : Vector
      Profile.measure("backbone.step", category: "backbone") { step_impl(input) }
    end

    private def step_impl(input : Vector) : Vector
      raise ArgumentError.new("backbone input width mismatch") unless input.size == @config.hidden
      hidden = input
      @config.layers.times do |i|
        prefix = "backbone.layers.#{i}"
        x = Ops.rms_norm(hidden, @store.vector(prefix + ".input_layernorm.weight"), @config.eps, false)
        residual = if (i + 1) % @config.full_interval == 0
                     full_attention(x, prefix + ".self_attn", @states[i])
                   else
                     linear_attention(x, prefix + ".linear_attn", @states[i])
                   end
        hidden = Ops.add(hidden, residual)
        @trace.try(&.call("backbone.layers.#{i}.attention_residual", hidden))
        x = Ops.rms_norm(hidden, @store.vector(prefix + ".post_attention_layernorm.weight"), @config.eps, false)
        hidden = Ops.finite!(Ops.add(hidden, moe(x, prefix + ".mlp")), "backbone layer #{i}")
        @trace.try(&.call("backbone.layers.#{i}.hidden", hidden))
      end
      @position += 1
      result_vector = Ops.rms_norm(hidden, @store.vector("backbone.norm.weight"), @config.eps, false)
      @trace.try(&.call("backbone.hidden", result_vector))
      result_vector
    end

    def mlp(x : Vector, prefix : String) : Vector
      Profile.measure("backbone.mlp", category: "backbone") { mlp_impl(x, prefix) }
    end

    private def mlp_impl(x : Vector, prefix : String) : Vector
      gate = @store.matrix(prefix + ".gate_proj.weight").matvec(x)
      up = @store.matrix(prefix + ".up_proj.weight").matvec(x)
      gated = Array(Float32).new(gate.size) { |i| gate[i] * Ops.sigmoid(gate[i]) * up[i] }
      @store.matrix(prefix + ".down_proj.weight").matvec(gated)
    end

    def moe(x : Vector, prefix : String) : Vector
      Profile.measure("backbone.moe", category: "moe") { moe_impl(x, prefix) }
    end

    private def moe_impl(x : Vector, prefix : String) : Vector
      scores = Ops.softmax(@store.matrix(prefix + ".gate.weight").matvec(x))
      ids = (0...scores.size).to_a.sort_by { |i| {-scores[i], i} }.first(@config.top_k)
      denominator = @config.norm_topk ? ids.sum { |i| scores[i] } : 1_f32
      result = Array(Float32).new(@config.hidden, 0_f32)
      projections = expert_projections(x, prefix, ids)
      ids.each_with_index do |e, index|
        result_vector = projections[index]
        weight = scores[e] / denominator
        result.size.times { |j| result[j] += result_vector[j] * weight }
      end
      shared = mlp(x, prefix + ".shared_expert")
      shared_weight = Ops.sigmoid(@store.matrix(prefix + ".shared_expert_gate.weight").matvec(x)[0])
      result.size.times { |j| result[j] += shared[j] * shared_weight }
      result
    end

    private def expert_projections(x : Vector, prefix : String, ids : Array(Int32)) : Array(Vector)
      {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
        if MetalBackend.enabled?
          matrices = ids.flat_map do |e|
            [@store.matrix(prefix + ".switch_mlp.gate_proj.weight", e),
             @store.matrix(prefix + ".switch_mlp.up_proj.weight", e)]
          end
          down = ids.map { |e| @store.matrix(prefix + ".switch_mlp.down_proj.weight", e) }
          if matrices.all?(&.scales) && down.all?(&.scales)
            # Gate/up projections are independent. Down projections depend only
            # on their own gated activations. Keep routing and final expert
            # accumulation on the CPU in their original order.
            projected = MetalBackend.affine_batch(matrices, Array(Vector).new(matrices.size) { x })
            activations = Array(Vector).new(ids.size) do |index|
              gate, up = projected[index * 2], projected[index * 2 + 1]
              Array(Float32).new(gate.size) { |j| gate[j] * Ops.sigmoid(gate[j]) * up[j] }
            end
            return MetalBackend.affine_batch(down, activations)
          end
        end
      {% end %}
      ids.map do |e|
        gate = @store.matrix(prefix + ".switch_mlp.gate_proj.weight", e).matvec(x)
        up = @store.matrix(prefix + ".switch_mlp.up_proj.weight", e).matvec(x)
        act = Array(Float32).new(gate.size) { |j| gate[j] * Ops.sigmoid(gate[j]) * up[j] }
        @store.matrix(prefix + ".switch_mlp.down_proj.weight", e).matvec(act)
      end
    end

    private def full_attention(x : Vector, prefix : String, cache : BackboneState) : Vector
      Profile.measure("backbone.attention.full", category: "attention") { full_attention_impl(x, prefix, cache) }
    end

    private def full_attention_impl(x : Vector, prefix : String, cache : BackboneState) : Vector
      c = @config
      projected = @store.matrix(prefix + ".q_proj.weight").matvec(x)
      q = [] of Float32
      gate = [] of Float32
      c.heads.times do |h|
        q.concat(Ops.rms_norm(projected[h * c.head_dim * 2, c.head_dim], @store.vector(prefix + ".q_norm.weight"), c.eps, false))
        gate.concat(projected[h * c.head_dim * 2 + c.head_dim, c.head_dim])
      end
      k = @store.matrix(prefix + ".k_proj.weight").matvec(x)
      c.kv_heads.times do |h|
        norm = Ops.rms_norm(k[h * c.head_dim, c.head_dim], @store.vector(prefix + ".k_norm.weight"), c.eps, false)
        c.head_dim.times { |d| k[h * c.head_dim + d] = norm[d] }
      end
      v = @store.matrix(prefix + ".v_proj.weight").matvec(x)
      Ops.rope!(q, c.heads, c.head_dim, c.rotary_dim, @position, c.rope_base, false)
      Ops.rope!(k, c.kv_heads, c.head_dim, c.rotary_dim, @position, c.rope_base, false)
      cache.keys << k
      cache.values << v
      result_vector = Array(Float32).new(c.heads * c.head_dim, 0_f32)
      c.heads.times do |h|
        kv = h // (c.heads // c.kv_heads)
        logits = Array(Float32).new(cache.keys.size) do |t|
          dot = 0_f32
          c.head_dim.times { |d| dot += q[h * c.head_dim + d] * cache.keys[t][kv * c.head_dim + d] }
          (dot / Math.sqrt(c.head_dim)).to_f32
        end
        weights = Ops.softmax(logits)
        c.head_dim.times do |d|
          sum = 0_f32
          weights.size.times { |t| sum += weights[t] * cache.values[t][kv * c.head_dim + d] }
          result_vector[h * c.head_dim + d] = sum * Ops.sigmoid(gate[h * c.head_dim + d])
        end
      end
      @store.matrix(prefix + ".o_proj.weight").matvec(result_vector)
    end

    private def linear_attention(x : Vector, prefix : String, cache : BackboneState) : Vector
      Profile.measure("backbone.attention.linear", category: "attention") { linear_attention_impl(x, prefix, cache) }
    end

    private def linear_attention_impl(x : Vector, prefix : String, cache : BackboneState) : Vector
      c = @config
      qkv = @store.matrix(prefix + ".in_proj_qkv.weight").matvec(x)
      z = @store.matrix(prefix + ".in_proj_z.weight").matvec(x)
      b = @store.matrix(prefix + ".in_proj_b.weight").matvec(x)
      a = @store.matrix(prefix + ".in_proj_a.weight").matvec(x)
      conv_w = @store.tensor(prefix + ".conv1d.weight")
      raise ArgumentError.new("invalid DeltaNet conv shape") unless conv_w.shape == [qkv.size, c.conv_kernel, 1]
      history = cache.conv
      history << qkv
      if history.size > c.conv_kernel
        history.shift
      end
      conv = Array(Float32).new(qkv.size) do |d|
        sum = 0_f32
        history.each_with_index do |row, t|
          sum += row[d] * conv_w[d * c.conv_kernel + c.conv_kernel - history.size + t]
        end
        sum * Ops.sigmoid(sum)
      end
      k_width = c.key_heads * c.key_dim
      q, k = conv[0, k_width], conv[k_width, k_width]
      v = conv[2 * k_width, c.value_heads * c.value_dim]
      c.key_heads.times do |h|
        ones = Array(Float32).new(c.key_dim, 1_f32)
        q_norm = Ops.rms_norm(q[h * c.key_dim, c.key_dim], ones, 1e-6, false)
        k_norm = Ops.rms_norm(k[h * c.key_dim, c.key_dim], ones, 1e-6, false)
        c.key_dim.times do |d|
          q[h * c.key_dim + d] = q_norm[d] / c.key_dim
          k[h * c.key_dim + d] = (k_norm[d] / Math.sqrt(c.key_dim)).to_f32
        end
      end
      alog, dt = @store.vector(prefix + ".A_log"), @store.vector(prefix + ".dt_bias")
      norm_w = @store.vector(prefix + ".norm.weight")
      output = Array(Float32).new(c.value_heads * c.value_dim, 0_f32)
      c.value_heads.times do |h|
        kh = h // (c.value_heads // c.key_heads)
        argument = a[h] + dt[h]
        softplus = argument > 20 ? argument.to_f64 : Math.log(1.0 + Math.exp(argument))
        decay = Math.exp(-Math.exp(alog[h]) * softplus).to_f32
        beta = Ops.sigmoid(b[h])
        c.value_dim.times do |d|
          offset = (h * c.value_dim + d) * c.key_dim
          memory = 0_f32
          c.key_dim.times do |j|
            cache.delta[offset + j] *= decay
            memory += cache.delta[offset + j] * k[kh * c.key_dim + j]
          end
          delta = (v[h * c.value_dim + d] - memory) * beta
          sum = 0_f32
          c.key_dim.times do |j|
            cache.delta[offset + j] += k[kh * c.key_dim + j] * delta
            sum += cache.delta[offset + j] * q[kh * c.key_dim + j]
          end
          output[h * c.value_dim + d] = sum
        end
        norm = Ops.rms_norm(output[h * c.value_dim, c.value_dim], norm_w, c.eps, false)
        c.value_dim.times { |d| output[h * c.value_dim + d] = norm[d] * z[h * c.value_dim + d] * Ops.sigmoid(z[h * c.value_dim + d]) }
      end
      @store.matrix(prefix + ".out_proj.weight").matvec(output)
    end
  end
end
