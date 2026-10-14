require "spec"
require "../src/ml/audio/drama3/tower"

include ML::Audio::Drama3

private def tiny_tower_config : TowerConfig
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

private def tiny_tower_store(config : TowerConfig) : WeightStore
  tensors = Hash(String, Tensor).new
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
    embedding = index < config.shared_enrichment_layers ? "semantic_embedding" : "codebook_embedding"
    add_matrix.call("#{layer_prefix}.#{embedding}.weight", config.vocab_size, config.dim)
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

describe ML::Audio::Drama3::Tower do
  it "teacher-forces one frame pair, produces each residual head's logits, and advances temporal caches" do
    config = tiny_tower_config
    tower = Tower.new(config, tiny_tower_store(config))
    caches = tower.make_cache
    hidden = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32]
    previous = [11_i32, 1_i32, 2_i32, 3_i32]
    current = [12_i32, 2_i32, 1_i32, 0_i32]

    output = tower.teacher_force(hidden, previous, current, caches)

    output.head_logits.size.should eq(config.num_codebooks - 1)
    output.head_logits.each do |logits|
      logits.size.should eq(config.vocab_size)
      logits.all?(&.finite?).should be_true
    end
    output.hidden.size.should eq(config.dim)
    caches.each { |cache| cache.offset.should eq(1) }

    next_hidden = [0.4_f32, 0.1_f32, -0.3_f32, 0.2_f32]
    next_previous = [12_i32, 2_i32, 1_i32, 0_i32]
    next_current = [13_i32, 3_i32, 0_i32, 1_i32]
    cached = tower.teacher_force(next_hidden, next_previous, next_current, caches)
    fresh = tower.teacher_force(next_hidden, next_previous, next_current, tower.make_cache)
    cached.head_logits.should_not eq(fresh.head_logits)
    caches.each { |cache| cache.offset.should eq(2) }
  end

  it "feeds each sampled residual code into the next codebook layer" do
    config = tiny_tower_config
    tower = Tower.new(config, tiny_tower_store(config))
    hidden = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32]
    previous = [11_i32, 1_i32, 2_i32, 3_i32]
    sampled = [1_i32, 3_i32]
    calls = 0

    result = tower.sample_frame(12_i32, hidden, previous, tower.make_cache) do |logits|
      logits.size.should eq(config.vocab_size)
      code = sampled[calls]
      calls += 1
      code
    end

    result.codes.should eq([2_i32, 1_i32, 3_i32])
    result.head_logits.size.should eq(config.num_codebooks - 1)
    calls.should eq(config.num_codebooks - 1)

    expected = tower.teacher_force(hidden, previous, [12_i32, 2_i32, 1_i32, 0_i32], tower.make_cache)
    wrong_input = tower.teacher_force(hidden, previous, [12_i32, 2_i32, 2_i32, 0_i32], tower.make_cache)
    result.head_logits.should eq(expected.head_logits)
    result.head_logits[1].should_not eq(wrong_input.head_logits[1])
  end

  it "gates unseen previous codes and current code embeddings by their own semantic frames" do
    config = tiny_tower_config
    store = tiny_tower_store(config)
    tower = Tower.new(config, store)
    traces = {} of String => Tuple(Array(Int32), Vector)
    capture = ->(name : String, shape : Array(Int32), values : Vector) do
      traces[name] = {shape, values}
    end
    hidden = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32]
    zeros = Array(Float32).new(config.dim, 0_f32)

    nonsemantic_previous = [9_i32, 1_i32, 2_i32, 3_i32]
    semantic_next = [12_i32, 2_i32, 1_i32, 0_i32]
    tower.teacher_force(
      hidden,
      nonsemantic_previous,
      semantic_next,
      tower.make_cache,
      capture,
      "tower.nonsemantic_to_semantic",
    )

    traces["tower.nonsemantic_to_semantic.unseen_embedding"][1].should eq(zeros)
    semantic_embedding = store.matrix("tower.shared_layers.0.semantic_embedding.weight").row(2_i32)
    traces["tower.nonsemantic_to_semantic.shared_layers.0.embedding"][1].should eq(semantic_embedding)
    residual_embedding = store.matrix("tower.residual_layers.0.codebook_embedding.weight").row(2_i32)
    traces["tower.nonsemantic_to_semantic.residual_layers.0.embedding"][1].should eq(residual_embedding)

    semantic_previous = [11_i32, 1_i32, 2_i32, 3_i32]
    nonsemantic_next = [9_i32, 2_i32, 1_i32, 0_i32]
    tower.teacher_force(
      hidden,
      semantic_previous,
      nonsemantic_next,
      tower.make_cache,
      capture,
      "tower.semantic_to_nonsemantic",
    )

    unseen = tower.codebook_embeddings.row(2 * config.vocab_size + semantic_previous[3])
    traces["tower.semantic_to_nonsemantic.unseen_embedding"][1].should eq(unseen)
    traces["tower.semantic_to_nonsemantic.shared_layers.0.embedding"][1].should eq(unseen)
    traces["tower.semantic_to_nonsemantic.residual_layers.0.embedding"][1].should eq(zeros)
  end

  it "emits named intermediate traces with their tensor shapes" do
    config = tiny_tower_config
    tower = Tower.new(config, tiny_tower_store(config))
    traces = {} of String => Tuple(Array(Int32), Vector)
    capture = ->(name : String, shape : Array(Int32), values : Vector) do
      traces[name] = {shape, values}
    end
    hidden = [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32]
    previous = [11_i32, 1_i32, 2_i32, 3_i32]
    current = [12_i32, 2_i32, 1_i32, 0_i32]

    output = tower.teacher_force(hidden, previous, current, tower.make_cache, capture, "tower.prefill.row0")

    traces["tower.prefill.row0.shared_layers.0.eh_concat"][0].should eq([1_i32, 1_i32, config.dim * 2])
    traces["tower.prefill.row0.shared_layers.0.attention.q_rope"][0].should eq([1_i32, config.n_head, 1_i32, config.head_dim])
    traces["tower.prefill.row0.shared_layers.0.attention.key_cache"][0].should eq([1_i32, config.n_local_heads, 1_i32, config.head_dim])
    traces["tower.prefill.row0.head_logits.0"][1].should eq(output.head_logits[0])
    traces["tower.prefill.row0.hidden_output"][1].should eq(output.hidden)
  end

  it "parses a compact audio decoder config while preserving tiny-test dimensions" do
    config = TowerConfig.from_json(%({"vocab_size": 8, "num_codebooks": 4, "dim": 24, "n_head": 3, "n_local_heads": 1, "head_dim": 8, "intermediate_size": 32, "shared_enrichment_layers": 2, "vq_backbone_codebooks": 1}))

    config.vocab_size.should eq(8)
    config.num_codebooks.should eq(4)
    config.dim.should eq(24)
    config.rotary_dim.should eq(2)
    config.semantic_end.should eq(7)
  end

  it "rejects explicitly requested tower architecture variants outside the pinned implementation" do
    unsupported = [
      {"attention_o_bias", "true"},
      {"attention_qkv_bias", "true"},
      {"attention_output_gate", "false"},
      {"layernorm_zero_centered_gamma", "false"},
      {"rotary_interleaved", "false"},
      {"mtp_use_swa", "true"},
      {"use_moe", "true"},
      {"vq_feature_placement", "\"backbone\""},
      {"decoder_type", "\"codebook_mtp\""},
    ]

    unsupported.each do |(name, value)|
      json = %({"audio_decoder_config":{"#{name}":#{value}}})
      expect_raises(ArgumentError, /#{name}/) { TowerConfig.from_json(json) }
    end

    # Older compact configs omit these fields and continue to use pinned defaults.
    TowerConfig.from_json(%({"audio_decoder_config":{"vocab_size":8}})).vocab_size.should eq(8)
  end

  it "rejects nonfinite norm, rotary-base, and rotary-percent values" do
    [Float64::INFINITY, Float64::NAN].each do |value|
      expect_raises(ArgumentError, /norm_eps/) { TowerConfig.new(norm_eps: value) }
      expect_raises(ArgumentError, /rope_base/) { TowerConfig.new(rope_base: value) }
      expect_raises(ArgumentError, /rotary_percent/) { TowerConfig.new(rotary_percent: value) }
    end
  end
end
