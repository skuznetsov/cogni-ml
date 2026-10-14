require "spec"
require "../src/ml/audio/drama3/tower_resident"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_resident_tower_config : TowerConfig
    TowerConfig.new(
      vocab_size: 4,
      num_codebooks: 3,
      dim: 4,
      n_head: 2,
      n_local_heads: 1,
      head_dim: 2,
      intermediate_size: 6,
      norm_eps: 1e-6,
      rope_base: 100.0,
      rotary_percent: 1.0,
      shared_enrichment_layers: 1,
      vq_backbone_codebooks: 1,
      semantic_begin: 10,
      semantic_end: 13,
    )
  end

  private def drama3_resident_tower_store(config : TowerConfig) : WeightStore
    tensors = {} of String => Tensor
    seed = 1
    add_matrix = ->(name : String, rows : Int32, cols : Int32) do
      values = Array(Float32).new(rows * cols) do |i|
        (((i + seed * 3) % 11) - 5).to_f32 * 0.017_f32
      end
      tensors[name] = Tensor.new([rows, cols], values)
      seed += 1
    end
    add_vector = ->(name : String, size : Int32) do
      values = Array(Float32).new(size) do |i|
        (((i + seed * 2) % 7) - 3).to_f32 * 0.006_f32
      end
      tensors[name] = Tensor.new([size], values)
      seed += 1
    end

    add_matrix.call("tower.codebook_embeddings.weight", config.vocab_size * config.num_codebooks, config.dim)
    layer_count = config.shared_enrichment_layers + config.num_codebooks - 1
    layer_count.times do |index|
      layer_prefix = if index < config.shared_enrichment_layers
                       "tower.shared_layers.#{index}"
                     else
                       "tower.residual_layers.#{index - config.shared_enrichment_layers}"
                     end
      embedding_name = index < config.shared_enrichment_layers ? "semantic_embedding" : "codebook_embedding"
      add_matrix.call("#{layer_prefix}.#{embedding_name}.weight", config.vocab_size, config.dim)
      {"enorm", "hnorm", "final_layernorm"}.each do |name|
        add_vector.call("#{layer_prefix}.#{name}.weight", config.dim)
      end
      add_matrix.call("#{layer_prefix}.eh_proj.weight", config.dim, config.dim * 2)

      transformer = "#{layer_prefix}.transformer_layer"
      add_vector.call("#{transformer}.attention_norm.weight", config.dim)
      add_vector.call("#{transformer}.ffn_norm.weight", config.dim)
      q_width = config.n_head * config.head_dim
      kv_width = config.n_local_heads * config.head_dim
      add_matrix.call("#{transformer}.attention.wqkv.weight", q_width * 2 + kv_width * 2, config.dim)
      add_matrix.call("#{transformer}.attention.wo.weight", config.dim, q_width)
      add_vector.call("#{transformer}.attention.q_norm.weight", config.head_dim)
      add_vector.call("#{transformer}.attention.k_norm.weight", config.head_dim)
      add_matrix.call("#{transformer}.feed_forward.w1.weight", config.intermediate_size, config.dim)
      add_matrix.call("#{transformer}.feed_forward.w3.weight", config.intermediate_size, config.dim)
      add_matrix.call("#{transformer}.feed_forward.w2.weight", config.dim, config.intermediate_size)

      if index >= config.shared_enrichment_layers
        head = index - config.shared_enrichment_layers
        add_matrix.call("tower.codebook_heads.#{head}.weight", config.vocab_size, config.dim)
      end
    end
    WeightStore.new(tensors)
  end

  private def drama3_resident_tower_capture : Tuple(Hash(String, Tuple(Array(Int32), Vector)), TowerTraceCallback)
    traces = {} of String => Tuple(Array(Int32), Vector)
    capture = ->(name : String, shape : Array(Int32), values : Vector) do
      traces[name] = {shape, values.dup}
    end
    {traces, capture}
  end

  private def drama3_resident_tower_argmax(logits : Vector) : Int32
    best_index = 0
    1.upto(logits.size - 1) do |index|
      best_index = index if logits[index] > logits[best_index]
    end
    best_index.to_i32
  end

  private def drama3_resident_tower_assert_traces_match(
    expected : Hash(String, Tuple(Array(Int32), Vector)),
    actual : Hash(String, Tuple(Array(Int32), Vector)),
  ) : Nil
    actual.keys.sort.should eq(expected.keys.sort)
    expected.each do |name, (shape, values)|
      actual_shape, actual_values = actual[name]
      actual_shape.should eq(shape)
      actual_values.size.should eq(values.size)
      actual_values.zip(values).each do |a, b|
        (a - b).abs.should be < 2.0e-4_f32
      end
    end
  end

  describe "DRAMA-3 resident tower" do
    it "matches tiny CPU reference traces through backbone embedding, teacher force, and sampled frame" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        config = drama3_resident_tower_config
        store = drama3_resident_tower_store(config)
        cpu = Tower.new(config, store)
        context = Resident::Context.new
        resident = TowerResident.new(config, store, context, 2)

        frame = [12_i32, 2_i32, 1_i32, 0_i32]
        text = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32]
        expected_embed, expected_embed_trace = drama3_resident_tower_capture
        expected_text = cpu.embed_backbone([frame], [text], expected_embed_trace)
        actual_embed, actual_embed_trace = drama3_resident_tower_capture
        actual_text = context.read(
          resident.embed_backbone(resident.upload_frame(frame), context.upload(text), actual_embed_trace),
          "terminal",
        )
        drama3_resident_tower_assert_traces_match(expected_embed, actual_embed)
        actual_text.zip(expected_text[0]).each { |a, b| (a - b).abs.should be < 2.0e-4_f32 }

        hidden = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32]
        previous = [11_i32, 1_i32, 2_i32, 3_i32]
        current = [12_i32, 2_i32, 1_i32, 0_i32]
        expected_prefill, expected_prefill_trace = drama3_resident_tower_capture
        expected_output = cpu.teacher_force(hidden, previous, current, cpu.make_cache, expected_prefill_trace, "tower.prefill.row0")
        actual_prefill, actual_prefill_trace = drama3_resident_tower_capture
        actual_output = resident.teacher_force(
          resident.upload_hidden(hidden), resident.upload_frame(previous), resident.upload_frame(current),
          actual_prefill_trace, "tower.prefill.row0",
        )
        drama3_resident_tower_assert_traces_match(expected_prefill, actual_prefill)
        context.read(actual_output.hidden, "terminal").zip(expected_output.hidden).each do |a, b|
          (a - b).abs.should be < 2.0e-4_f32
        end
        actual_output.head_logits.each_with_index do |logits, index|
          context.read(logits, "terminal").zip(expected_output.head_logits[index]).each do |a, b|
            (a - b).abs.should be < 2.0e-4_f32
          end
        end

        next_hidden = [0.4_f32, 0.1_f32, -0.3_f32, 0.2_f32]
        decode_previous = current
        decode_main = 13_i32
        expected_decode, expected_decode_trace = drama3_resident_tower_capture
        # Reuse the cache populated by teacher forcing in both implementations.
        cpu_cached = Tower.new(config, store)
        cpu_cached_cache = cpu_cached.make_cache
        cpu_cached.teacher_force(hidden, previous, current, cpu_cached_cache)
        expected_sample = cpu_cached.sample_frame(decode_main, next_hidden, decode_previous, cpu_cached_cache, expected_decode_trace, "tower.decode.row0") do |logits|
          drama3_resident_tower_argmax(logits)
        end

        actual_decode, actual_decode_trace = drama3_resident_tower_capture
        draw = context.upload([0.0_f32])
        actual_frame = resident.sample_frame(
          context.upload([decode_main.to_f32]), resident.upload_hidden(next_hidden), resident.upload_frame(decode_previous),
          actual_decode_trace, "tower.decode.row0",
        ) do |logits|
          context.sample(logits, draw, 1.0, 1, 1.0)
        end
        drama3_resident_tower_assert_traces_match(expected_decode, actual_decode)
        actual_codes = context.read(actual_frame, "terminal").map(&.to_i32)
        actual_codes.should eq([decode_main] + expected_sample.codes)
        resident.offset.should eq(2)
        context.readbacks.keys.all? { |purpose| purpose.in?("terminal", "control", "diagnostic") }.should be_true
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
      end
    end
  end
{% end %}
