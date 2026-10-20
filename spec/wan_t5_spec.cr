require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/wan_t5"
require "./support/video_allocation_counter"

private def t5_assert_gate(name : String, actual : Array(Float32), expected : JSON::Any) : Nil
  ref = expected.as_a
  actual.size.should eq(ref.size), name
  worst = 0_f64
  violations = 0
  actual.each_with_index do |x, i|
    target = ref[i].as_f.to_f32
    ratio = (x - target).abs.to_f64 / (2e-5 + 2e-5 * target.abs)
    violations += 1 unless x.finite? && target.finite? && ratio <= 1
    worst = Math.max(worst, ratio)
  end
  violations.should eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}"
end

describe ML::Video::WanT5 do
  it "requires an explicit map for partial tracing before device setup" do
    expect_raises(ArgumentError, "Wan UMT5 partial trace requires a trace buffer map") do
      ML::Video::WanT5.validate_trace_request!(nil, true)
    end
    ML::Video::WanT5.validate_trace_request!(nil, false)
  end

  it "rejects unsupported shapes and reports exact setup scratch bytes" do
    expect_raises(ArgumentError) { ML::Video::WanT5Config.new(dim: 33, heads: 4).validate! }
    expect_raises(ArgumentError) { ML::Video::WanT5Config.new(buckets: 6).validate! }
    cfg = ML::Video::WanT5Config.new(vocab: 96, dim: 32, ffn_dim: 64, heads: 4, layers: 2, batches: 2, tokens: 9)
    ML::Video::WanT5.checkpoint_bindings(cfg).size.should eq(22)
    a = ML::Video::Arena.new
    begin
      graph_attention = ENV["WAN_T5_MPS_ATTENTION"]? == "1"
      ML::Video::WanT5.prepare_scratch!(a, cfg, graph_attention: graph_attention)
      a.resident_bytes.should eq(ML::Video::WanT5.scratch_bytes(cfg, graph_attention: graph_attention))
      expect_raises(ArgumentError) do
        ML::Video::WanT5.upload_inputs!(a, Array(Int32).new(cfg.rows, 96), Array(Int32).new(cfg.rows, 1), cfg)
      end
      expect_raises(ArgumentError) do
        ML::Video::WanT5.upload_inputs!(a, Array(Int32).new(cfg.rows, 3), Array(Int32).new(cfg.rows, 0), cfg)
      end
    ensure
      a.release
    end
  end

  it "matches every pinned upstream intermediate and repeats with resident scratch" do
    fixture_path = ENV["WAN_T5_FIXTURE"]?
    unless fixture_path
      pending! "Set WAN_T5_FIXTURE to a pinned MPS fixture"
    end
    fixture = JSON.parse(File.read(fixture_path))
    fixture["revision"].as_s.should eq("1ea34ff48f87168174e12956e200b1d908b1c5ff")
    fixture["source_sha256"].as_s.should eq("8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f")
    fixture["device"].as_s.should eq("mps")
    fixture["arithmetic"].as_s.should eq("F32")
    digest = Digest::SHA256.new
    File.open(fixture["weights"].as_s, "rb") do |io|
      bytes = Bytes.new(1024 * 1024)
      while (size = io.read(bytes)) > 0
        digest.update(bytes[0, size])
      end
    end
    digest.hexfinal.should eq(fixture["weight_sha256"].as_s)
    c = fixture["config"]
    cfg = ML::Video::WanT5Config.new(vocab: c["vocab"].as_i, dim: c["dim"].as_i,
      ffn_dim: c["ffn_dim"].as_i, heads: c["heads"].as_i, layers: c["layers"].as_i,
      buckets: c["buckets"].as_i, max_distance: c["max_distance"].as_i,
      batches: c["batches"].as_i, tokens: c["tokens"].as_i, eps: c["eps"].as_f.to_f32)
    file = ML::Video::CheckpointFile.new(fixture["weights"].as_s)
    a = ML::Video::Arena.new
    traces = ML::Video::Arena.new
    prepared = nil.as(ML::Video::PreparedWeights?)
    mps_plans = nil.as(ML::Video::WanT5MPSPlans?)
    attention_graph_plan = nil.as(ML::Video::WanT5MPSAttentionPlan?)
    begin
      graph_attention = ENV["WAN_T5_MPS_ATTENTION"]? == "1"
      dtype = fixture["storage"].as_s == "BF16" ? ML::Video::WeightDType::BF16 : ML::Video::WeightDType::F32
      plan = ML::Video::WeightPlan.new([file], ML::Video::WanT5.checkpoint_bindings(cfg),
        8_i64 * 1024 * 1024 * 1024, ML::Video::WanT5.scratch_bytes(cfg, graph_attention: graph_attention), resident_dtype: dtype, require_all: true)
      prepared = ML::Video::PreparedWeights.new(plan, a)
      ML::Video::WanT5.prepare_scratch!(a, cfg, graph_attention: graph_attention)
      ML::Video::WanT5.upload_inputs!(a, fixture["ids"].as_a.map(&.as_i), fixture["mask"].as_a.map(&.as_i), cfg)
      mps_plans = ML::Video::WanT5.prepare_mps_plans(cfg, a, graph_attention: graph_attention)
      n = cfg.rows * cfg.dim
      output = a.alloc("output", n)
      repeated = a.alloc("repeated", n)
      masked = a.alloc("masked", n)
      isolated_product = a.upload("isolated_product", fixture["expected"]["blocks.0.product"].as_a.map(&.as_f.to_f32))
      isolated_fc2 = a.alloc("isolated_fc2", n)
      isolated_fc2_repeat = a.alloc("isolated_fc2_repeat", n)
      probe_norm1_input = a.upload("probe_norm1_input", fixture["expected"]["blocks.0.norm1.input"].as_a.map(&.as_f.to_f32))
      probe_norm1_output = a.alloc("probe_norm1_output", n)
      probe_norm1_mean_square = a.alloc("probe_norm1_mean_square", cfg.rows)
      probe_norm1_inverse_rms = a.alloc("probe_norm1_inverse_rms", cfg.rows)
      probe_norm2_input = a.upload("probe_norm2_input", fixture["expected"]["blocks.0.norm2.input"].as_a.map(&.as_f.to_f32))
      probe_norm2_output = a.alloc("probe_norm2_output", n)
      probe_norm2_mean_square = a.alloc("probe_norm2_mean_square", cfg.rows)
      probe_norm2_inverse_rms = a.alloc("probe_norm2_inverse_rms", cfg.rows)
      contract_off_norm1_output = a.alloc("probe_contract_off_norm1_output", n)
      contract_off_norm1_mean_square = a.alloc("probe_contract_off_norm1_mean_square", cfg.rows)
      contract_off_norm1_inverse_rms = a.alloc("probe_contract_off_norm1_inverse_rms", cfg.rows)
      contract_off_norm2_output = a.alloc("probe_contract_off_norm2_output", n)
      contract_off_norm2_mean_square = a.alloc("probe_contract_off_norm2_mean_square", cfg.rows)
      contract_off_norm2_inverse_rms = a.alloc("probe_contract_off_norm2_inverse_rms", cfg.rows)
      rms_probe_modes = [] of Tuple(Int32, Bool)
      rms_probe_outputs = {} of String => ML::MetalBuffer
      {32, 64, 128, 256, 512, 1024}.each do |threads|
        {false, true}.each do |vector4|
          rms_probe_modes << {threads, vector4}
          suffix = "t#{threads}_#{vector4 ? "vec4" : "scalar"}"
          rms_probe_outputs["norm1_#{suffix}"] = a.alloc("probe_norm1_#{suffix}", n)
          rms_probe_outputs["norm2_#{suffix}"] = a.alloc("probe_norm2_#{suffix}", n)
        end
      end
      probe_q = a.upload("probe_attention_q", fixture["expected"]["blocks.0.q"].as_a.map(&.as_f.to_f32))
      probe_k = a.upload("probe_attention_k", fixture["expected"]["blocks.0.k"].as_a.map(&.as_f.to_f32))
      probe_v = a.upload("probe_attention_v", fixture["expected"]["blocks.0.v"].as_a.map(&.as_f.to_f32))
      probe_attention = a.alloc("probe_attention", n)
      attention_graph_bias = nil.as(ML::MetalBuffer?)
      attention_graph_logits = nil.as(ML::MetalBuffer?)
      attention_graph_scores = nil.as(ML::MetalBuffer?)
      attention_graph_probabilities = nil.as(ML::MetalBuffer?)
      attention_graph_context = nil.as(ML::MetalBuffer?)
      attention_score_count = 0_i32
      if ENV["WAN_T5_MPS_ATTENTION_PROBE"]? == "1"
        score_count = cfg.batches.to_i64 * cfg.heads * cfg.tokens * cfg.tokens
        unless score_count > 0 && score_count <= Int32::MAX
          raise ArgumentError.new("Attention MPS probe scores exceed addressable storage")
        end
        attention_score_count = score_count.to_i32
        expected = fixture["expected"].as_h
        {"blocks.0.attention_bias", "blocks.0.attention_logits",
         "blocks.0.attention_scores", "blocks.0.attention_probs"}.each do |name|
          raise ArgumentError.new("Attention MPS probe requires fixture key #{name}; regenerate with --attention-diagnostics") unless expected.has_key?(name)
        end
        attention_graph_bias = a.upload("probe_attention_bias",
          expected["blocks.0.attention_bias"].as_a.map(&.as_f.to_f32))
        attention_graph_logits = a.alloc("probe_attention_mps_logits", attention_score_count)
        attention_graph_scores = a.alloc("probe_attention_mps_scores", attention_score_count)
        attention_graph_probabilities = a.alloc("probe_attention_mps_probs", attention_score_count)
        attention_graph_context = a.alloc("probe_attention_mps_context", n)
        attention_graph_plan = ML::Video::WanT5MPSAttentionPlan.new(cfg,
          probe_q, probe_k, probe_v, attention_graph_bias.not_nil!,
          attention_graph_logits.not_nil!, attention_graph_scores.not_nil!,
          attention_graph_probabilities.not_nil!, attention_graph_context.not_nil!)
      end
      gate_count = cfg.rows * cfg.ffn_dim
      probe_gate_input = a.upload("probe_gate_input",
        fixture["expected"]["blocks.0.gate_pre"].as_a.map(&.as_f.to_f32))
      probe_gate_generic = a.alloc("probe_gate_generic", gate_count)
      probe_gate_source = a.alloc("probe_gate_source", gate_count)
      trace = {} of String => ML::MetalBuffer
      fixture["expected"].as_h.each do |name, values|
        # Raw squares are checked by the standalone MPS RMS probe, not encode's trace map.
        next if name == "masked_context" || name.ends_with?(".input") ||
                name.ends_with?(".mean_square") || name.ends_with?(".inverse_rms") ||
                name.ends_with?(".square") || name.ends_with?(".attention_bias") ||
                name.ends_with?(".attention_logits") || name.ends_with?(".attention_scores") ||
                name.ends_with?(".attention_probs")
        trace[name] = traces.alloc(name, values.as_a.size)
      end
      rejected = ML::Metal::MPSCommandChain.new
      begin
        expect_raises(ArgumentError, "Unsafe video buffer alias") do
          ML::Video::WanT5.encode(rejected, a, cfg, a["t5.s.x"], a["t5.input.mask"],
            a["t5.s.x"], mps_plans.not_nil!)
        end
      ensure
        rejected.commit_and_wait unless rejected.completed?
        rejected.release
      end
      before = ML::MetalBuffer.video_spec_allocations
      command = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(command)
      prepared.encode(e)
      ML::Video::MetalOps.gelu(e, probe_gate_input, probe_gate_generic, gate_count)
      ML::Video::WanT5.gelu_source(e, probe_gate_input, probe_gate_source, gate_count)
      norm1_weight = a["t5.blocks.0.norm1.weight"]
      ML::Video::WanT5.rms_norm_trace(e, probe_norm1_input, norm1_weight,
        probe_norm1_output, probe_norm1_mean_square, probe_norm1_inverse_rms,
        cfg.rows, cfg.dim, cfg.eps,
        a.dtype("t5.blocks.0.norm1.weight"))
      norm2_weight = a["t5.blocks.0.norm2.weight"]
      ML::Video::WanT5.rms_norm_trace(e, probe_norm2_input, norm2_weight,
        probe_norm2_output, probe_norm2_mean_square, probe_norm2_inverse_rms,
        cfg.rows, cfg.dim, cfg.eps,
        a.dtype("t5.blocks.0.norm2.weight"))
      ML::Video::WanT5.rms_norm_contract_off_trace(e, probe_norm1_input, norm1_weight,
        contract_off_norm1_output, contract_off_norm1_mean_square, contract_off_norm1_inverse_rms,
        cfg.rows, cfg.dim, cfg.eps,
        a.dtype("t5.blocks.0.norm1.weight"))
      ML::Video::WanT5.rms_norm_contract_off_trace(e, probe_norm2_input, norm2_weight,
        contract_off_norm2_output, contract_off_norm2_mean_square, contract_off_norm2_inverse_rms,
        cfg.rows, cfg.dim, cfg.eps,
        a.dtype("t5.blocks.0.norm2.weight"))
      norm1_dtype = a.dtype("t5.blocks.0.norm1.weight")
      norm2_dtype = a.dtype("t5.blocks.0.norm2.weight")
      rms_probe_modes.each do |threads, vector4|
        suffix = "t#{threads}_#{vector4 ? "vec4" : "scalar"}"
        ML::Video::WanT5.rms_norm_probe(e, probe_norm1_input, norm1_weight,
          rms_probe_outputs["norm1_#{suffix}"], cfg.rows, cfg.dim, cfg.eps,
          norm1_dtype, threads, vector4)
        ML::Video::WanT5.rms_norm_probe(e, probe_norm2_input, norm2_weight,
          rms_probe_outputs["norm2_#{suffix}"], cfg.rows, cfg.dim, cfg.eps,
          norm2_dtype, threads, vector4)
      end
      bias_name = "t5.blocks.0.pos_embedding.embedding.weight"
      ML::Video::WanT5.attention_kernel(e, probe_q, probe_k, probe_v, a[bias_name],
        a["t5.input.mask"], probe_attention, cfg, a.dtype(bias_name))
      fc1 = a["t5.s.fc1"]
      fc2_output = a["t5.s.ffn_out"]
      fc2_weight = a["t5.blocks.0.ffn.fc2.weight"]
      ML::Video::MetalOps.copy(e, isolated_product, fc1, cfg.rows * cfg.ffn_dim)
      ML::Video::WanT5.down_projection(e, fc1, fc2_weight, fc2_output,
        cfg.rows, cfg.ffn_dim, cfg.dim, a.dtype("t5.blocks.0.ffn.fc2.weight"),
        mps_plans.not_nil!.fc2, a["t5.s.fc2_weight_f32"])
      ML::Video::MetalOps.copy(e, fc2_output, isolated_fc2, n)
      ML::Video::WanT5.down_projection(e, fc1, fc2_weight, fc2_output,
        cfg.rows, cfg.ffn_dim, cfg.dim, a.dtype("t5.blocks.0.ffn.fc2.weight"),
        mps_plans.not_nil!.fc2, a["t5.s.fc2_weight_f32"])
      ML::Video::MetalOps.copy(e, fc2_output, isolated_fc2_repeat, n)
      e.end_encoding
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        expect_raises(ArgumentError) { e.encode_wan_t5_mps_fc2(mps_plans.not_nil!.fc2.handle) }
      {% end %}
      ML::MetalBuffer.video_spec_allocations.should eq(before), "encode must allocate no GPU buffers"
      command.commit
      command.wait

      if attention_plan = attention_graph_plan
        graph_chain = ML::Metal::MPSCommandChain.new
        begin
          graph_chain.pause!
          attention_plan.validate_buffers!(probe_q, probe_k, probe_v,
            attention_graph_bias.not_nil!, attention_graph_logits.not_nil!,
            attention_graph_scores.not_nil!, attention_graph_probabilities.not_nil!,
            attention_graph_context.not_nil!)
          attention_plan.encode(graph_chain.handle)
          ML::MetalBuffer.video_spec_allocations.should eq(before), "attention graph encode must not allocate GPU buffers"
          graph_chain.commit_and_wait
          graph_chain.wait_count.should eq(1), "attention diagnostic must use one terminal wait"
          graph_chain.segment_count.should be > 0, "attention graph must encode command work"
        ensure
          graph_chain.commit_and_wait unless graph_chain.completed?
          graph_chain.release
        end
      end

      chain = nil.as(ML::Metal::MPSCommandChain?)
      source_gelu = ENV["WAN_T5_SOURCE_GELU"]? == "1"
      begin
        active = ML::Metal::MPSCommandChain.new
        chain = active
        prepared.encode(active.encoder)
        ML::Video::WanT5.encode(active, a, cfg, a["t5.input.ids"], a["t5.input.mask"], output,
          mps_plans.not_nil!, trace: trace, source_gelu: source_gelu)
        ML::Video::WanT5.mask_context(active.encoder, output, a["t5.input.mask"], masked, cfg)
        ML::Video::WanT5.encode(active, a, cfg, a["t5.input.ids"], a["t5.input.mask"], repeated,
          mps_plans.not_nil!, source_gelu: source_gelu)
        ML::MetalBuffer.video_spec_allocations.should eq(before), "encode must allocate no GPU buffers"
        active.commit_and_wait
        active.wait_count.should eq(1), "T5 must use one terminal wait"
        active.segment_count.should be > 0, "T5 graph mean must submit through the caller chain"
      ensure
        if active = chain
          active.commit_and_wait unless active.completed?
          active.release
        end
      end
      if actual_path = ENV["WAN_T5_ACTUAL"]?
        captured = trace.to_h { |name, buffer| {name, buffer.read((buffer.size // 4).to_i)} }
        captured["isolated_fc2"] = isolated_fc2.read(n)
        captured["isolated_fc2_repeat"] = isolated_fc2_repeat.read(n)
        captured["probe_norm1"] = probe_norm1_output.read(n)
        captured["probe_norm2"] = probe_norm2_output.read(n)
        captured["probe_norm1_mean_square"] = probe_norm1_mean_square.read(cfg.rows)
        captured["probe_norm1_inverse_rms"] = probe_norm1_inverse_rms.read(cfg.rows)
        captured["probe_norm2_mean_square"] = probe_norm2_mean_square.read(cfg.rows)
        captured["probe_norm2_inverse_rms"] = probe_norm2_inverse_rms.read(cfg.rows)
        captured["probe_contract_off_norm1"] = contract_off_norm1_output.read(n)
        captured["probe_contract_off_norm1_mean_square"] = contract_off_norm1_mean_square.read(cfg.rows)
        captured["probe_contract_off_norm1_inverse_rms"] = contract_off_norm1_inverse_rms.read(cfg.rows)
        captured["probe_contract_off_norm2"] = contract_off_norm2_output.read(n)
        captured["probe_contract_off_norm2_mean_square"] = contract_off_norm2_mean_square.read(cfg.rows)
        captured["probe_contract_off_norm2_inverse_rms"] = contract_off_norm2_inverse_rms.read(cfg.rows)
        rms_probe_outputs.each do |name, buffer|
          captured["probe_#{name}"] = buffer.read(n)
        end
        captured["probe_attention"] = probe_attention.read(n)
        if attention_graph_plan
          captured["probe_attention_mps_logits"] = attention_graph_logits.not_nil!.read(attention_score_count)
          captured["probe_attention_mps_scores"] = attention_graph_scores.not_nil!.read(attention_score_count)
          captured["probe_attention_mps_probs"] = attention_graph_probabilities.not_nil!.read(attention_score_count)
          captured["probe_attention_mps_context"] = attention_graph_context.not_nil!.read(n)
        end
        captured["probe_gate_generic"] = probe_gate_generic.read(gate_count)
        captured["probe_gate_source"] = probe_gate_source.read(gate_count)
        File.write(actual_path, captured.to_json)
      end
      t5_assert_gate("probe_norm1", probe_norm1_output.read(n), fixture["expected"]["blocks.0.norm1"])
      t5_assert_gate("probe_norm2", probe_norm2_output.read(n), fixture["expected"]["blocks.0.norm2"])
      t5_assert_gate("probe_norm1_mean_square", probe_norm1_mean_square.read(cfg.rows),
        fixture["expected"]["blocks.0.norm1.mean_square"])
      t5_assert_gate("probe_norm1_inverse_rms", probe_norm1_inverse_rms.read(cfg.rows),
        fixture["expected"]["blocks.0.norm1.inverse_rms"])
      t5_assert_gate("probe_norm2_mean_square", probe_norm2_mean_square.read(cfg.rows),
        fixture["expected"]["blocks.0.norm2.mean_square"])
      t5_assert_gate("probe_norm2_inverse_rms", probe_norm2_inverse_rms.read(cfg.rows),
        fixture["expected"]["blocks.0.norm2.inverse_rms"])
      # Keep the diagnostic candidate under the same strict gate, but do not
      # route production through it unless its raw stats identify the mismatch.
      t5_assert_gate("probe_contract_off_norm1", contract_off_norm1_output.read(n),
        fixture["expected"]["blocks.0.norm1"])
      t5_assert_gate("probe_contract_off_norm1_mean_square", contract_off_norm1_mean_square.read(cfg.rows),
        fixture["expected"]["blocks.0.norm1.mean_square"])
      t5_assert_gate("probe_contract_off_norm1_inverse_rms", contract_off_norm1_inverse_rms.read(cfg.rows),
        fixture["expected"]["blocks.0.norm1.inverse_rms"])
      t5_assert_gate("probe_contract_off_norm2", contract_off_norm2_output.read(n),
        fixture["expected"]["blocks.0.norm2"])
      t5_assert_gate("probe_contract_off_norm2_mean_square", contract_off_norm2_mean_square.read(cfg.rows),
        fixture["expected"]["blocks.0.norm2.mean_square"])
      t5_assert_gate("probe_contract_off_norm2_inverse_rms", contract_off_norm2_inverse_rms.read(cfg.rows),
        fixture["expected"]["blocks.0.norm2.inverse_rms"])
      t5_assert_gate("probe_attention", probe_attention.read(n), fixture["expected"]["blocks.0.attention"])
      if attention_graph_plan
        t5_assert_gate("probe_attention_mps_logits", attention_graph_logits.not_nil!.read(attention_score_count),
          fixture["expected"]["blocks.0.attention_logits"])
        t5_assert_gate("probe_attention_mps_scores", attention_graph_scores.not_nil!.read(attention_score_count),
          fixture["expected"]["blocks.0.attention_scores"])
        t5_assert_gate("probe_attention_mps_probs", attention_graph_probabilities.not_nil!.read(attention_score_count),
          fixture["expected"]["blocks.0.attention_probs"])
        t5_assert_gate("probe_attention_mps_context", attention_graph_context.not_nil!.read(n),
          fixture["expected"]["blocks.0.attention"])
      end
      t5_assert_gate("probe_gate_generic", probe_gate_generic.read(gate_count),
        fixture["expected"]["blocks.0.gate"])
      t5_assert_gate("probe_gate_source", probe_gate_source.read(gate_count),
        fixture["expected"]["blocks.0.gate"])
      t5_assert_gate("isolated_fc2", isolated_fc2.read(n), fixture["expected"]["blocks.0.ffn"])
      isolated_fc2_repeat.read(n).should eq(isolated_fc2.read(n))
      trace.each { |name, buf| t5_assert_gate(name, buf.read((buf.size // 4).to_i), fixture["expected"][name]) }
      t5_assert_gate("repeated", repeated.read(n), fixture["expected"]["output"])
      t5_assert_gate("masked_context", masked.read(n), fixture["expected"]["masked_context"])
      repeated.read(n).should eq(output.read(n))
    ensure
      attention_graph_plan.try(&.release)
      mps_plans.try(&.release)
      prepared.try(&.release)
      traces.release
      a.release
      file.close
    end
  end
end
