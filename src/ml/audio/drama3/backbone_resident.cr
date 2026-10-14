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
    @keys : Array(Resident::Buffer?)
    @values : Array(Resident::Buffer?)
    @conv : Array(Resident::Buffer?)
    @delta : Array(Resident::Buffer?)

    def initialize(@config : BackboneConfig, @store : WeightStore, @ctx : Resident::Context, @capacity : Int32)
      raise ArgumentError.new("resident backbone capacity must be 1..1024") unless @capacity.in?(1..1024)
      raise ArgumentError.new("resident router supports at most 512 experts") unless @config.experts <= 512
      @keys = Array(Resident::Buffer?).new(@config.layers, nil)
      @values = Array(Resident::Buffer?).new(@config.layers, nil)
      @conv = Array(Resident::Buffer?).new(@config.layers, nil)
      @delta = Array(Resident::Buffer?).new(@config.layers, nil)
      @trace = nil
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
          x = @ctx.rms(hidden, weight(prefix + ".post_attention_layernorm.weight"), @config.eps)
          hidden = @ctx.add(hidden, moe(x, prefix + ".mlp"))
          @ctx.guard(hidden)
          trace_value("backbone.layers.#{i}.hidden", hidden)
        end
        @position += 1
        result = @ctx.rms(hidden, weight("backbone.norm.weight"), @config.eps)
        @ctx.guard(result)
        trace_value("backbone.hidden", result)
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

    private def mlp(x : Resident::Buffer, prefix : String) : Resident::Buffer
      gate = project(prefix + ".gate_proj", x)
      up = project(prefix + ".up_proj", x)
      project(prefix + ".down_proj", @ctx.mul(@ctx.silu(gate), up))
    end

    private def moe(x : Resident::Buffer, prefix : String) : Resident::Buffer
      Profile.measure("backbone.moe", category: "moe") do
        scores = project(prefix + ".gate", x)
        routing = @ctx.alloc(@config.top_k * 2)
        @ctx.dispatch("dr3_route", SOURCE, [scores, routing], [scores.count.to_u64, @config.top_k.to_u64, @config.norm_topk ? 1_u64 : 0_u64], 1)
        ids = routing.slice(0, @config.top_k)
        result = @ctx.zeros(@config.hidden)
        @config.top_k.times do |rank|
          gate = @ctx.expert(@store.matrix(prefix + ".switch_mlp.gate_proj.weight", 0), x, ids, rank)
          up = @ctx.expert(@store.matrix(prefix + ".switch_mlp.up_proj.weight", 0), x, ids, rank)
          down = @ctx.expert(@store.matrix(prefix + ".switch_mlp.down_proj.weight", 0), @ctx.mul(@ctx.silu(gate), up), ids, rank)
          @ctx.dispatch("dr3_weighted_add", SOURCE, [result, down, routing.slice(@config.top_k + rank, 1)], [result.count.to_u64, 0_u64], result.count)
        end
        shared = mlp(x, prefix + ".shared_expert")
        shared_gate = project(prefix + ".shared_expert_gate", x)
        @ctx.dispatch("dr3_weighted_add", SOURCE, [result, shared, shared_gate], [result.count.to_u64, 1_u64], result.count)
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
      qkv = project(prefix + ".in_proj_qkv", x)
      z = project(prefix + ".in_proj_z", x)
      b = project(prefix + ".in_proj_b", x)
      a = project(prefix + ".in_proj_a", x)
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
      @ctx.dispatch("dr3_delta_step", SOURCE, [q, k, v, a, b, weight(prefix + ".A_log"), weight(prefix + ".dt_bias"), @delta[layer].not_nil!, output], [c.value_heads.to_u64, c.value_dim.to_u64, c.key_heads.to_u64, c.key_dim.to_u64], v.count)
      normalized = @ctx.rms(output, weight(prefix + ".norm.weight"), c.eps, heads: c.value_heads)
      project(prefix + ".out_proj", @ctx.mul(normalized, @ctx.silu(z)))
    end
  end
end
{% end %}
