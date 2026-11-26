require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/wan_t5"
require "./support/video_allocation_counter"

private def wan_t5_full_context_sha256(path : String) : String
  digest = Digest::SHA256.new
  File.open(path, "rb") do |io|
    bytes = Bytes.new(1024 * 1024)
    while (size = io.read(bytes)) > 0
      digest.update(bytes[0, size])
    end
  end
  digest.hexfinal
end

private def wan_t5_full_context_assert_gate(name : String, actual : Array(Float32), expected : Array(Float32)) : Nil
  actual.size.should eq(expected.size), name
  worst = 0_f64
  violations = 0
  actual.each_with_index do |value, i|
    reference = expected[i]
    ratio = (value - reference).abs.to_f64 / (2e-5 + 2e-5 * reference.abs)
    violations += 1 unless value.finite? && reference.finite? && ratio <= 1
    worst = Math.max(worst, ratio)
  end
  violations.should eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}"
end

describe "Wan UMT5 full-context qualification" do
  it "matches selected traces from the pinned 24-layer, 512-token BF16 checkpoint" do
    fixture_path = ENV["WAN_T5_FULL_CONTEXT_FIXTURE"]?
    unless fixture_path
      pending! "Set WAN_T5_FULL_CONTEXT_FIXTURE to a --full-context pinned MPS fixture"
    end

    fixture = JSON.parse(File.read(fixture_path))
    fixture["revision"].as_s.should eq("1ea34ff48f87168174e12956e200b1d908b1c5ff")
    fixture["source_sha256"].as_s.should eq("8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f")
    fixture["device"].as_s.should eq("mps")
    fixture["arithmetic"].as_s.should eq("F32")
    fixture["storage"].as_s.should eq("BF16")
    fixture["expected"].as_nil

    config = fixture["config"]
    cfg = ML::Video::WanT5Config.new(
      vocab: config["vocab"].as_i,
      dim: config["dim"].as_i,
      ffn_dim: config["ffn_dim"].as_i,
      heads: config["heads"].as_i,
      layers: config["layers"].as_i,
      buckets: config["buckets"].as_i,
      max_distance: config["max_distance"].as_i,
      batches: config["batches"].as_i,
      tokens: config["tokens"].as_i,
      eps: config["eps"].as_f.to_f32,
    )
    {
      cfg.vocab,
      cfg.dim,
      cfg.ffn_dim,
      cfg.heads,
      cfg.layers,
      cfg.buckets,
      cfg.max_distance,
      cfg.batches,
      cfg.tokens,
    }.should eq({256384, 4096, 10240, 64, 24, 32, 128, 1, 512})
    cfg.eps.should eq(1e-6_f32)
    cfg.validate!

    checkpoint_path = fixture["weights"].as_s
    File.exists?(checkpoint_path).should be_true
    wan_t5_full_context_sha256(checkpoint_path).should eq(fixture["weight_sha256"].as_s)
    expected_path = fixture["expected_safetensors"].as_s
    File.exists?(expected_path).should be_true
    wan_t5_full_context_sha256(expected_path).should eq(fixture["expected_safetensors_sha256"].as_s)

    first_block_points = {
      "norm1", "q", "k", "v", "attention", "attn_projection", "attn_residual",
      "norm2", "gate_pre", "gate", "fc1", "product", "ffn", "residual",
    }
    selected = ["embedding"] of String
    cfg.layers.times do |layer|
      if layer == 0
        first_block_points.each { |point| selected << "blocks.0.#{point}" }
      else
        selected << "blocks.#{layer}.residual"
      end
    end
    selected << "output"
    selected.size.should eq(39)

    trace_shapes = fixture["trace_shapes"].as_h
    sidecar = ML::GGUF::SafetensorsFile.new(expected_path)
    sidecar.metadata["upstream_revision"]?.should eq("1ea34ff48f87168174e12956e200b1d908b1c5ff")
    sidecar.metadata["source_sha256"]?.should eq("8b0cebf3192c542f92a344255a06c203df3ba24160715899a055cc8de0cd930f")
    sidecar.metadata["arithmetic"]?.should eq("F32")
    sidecar.metadata["device"]?.should eq("mps")

    checkpoint = ML::Video::CheckpointFile.new(checkpoint_path)
    arena = ML::Video::Arena.new
    traces = ML::Video::Arena.new
    prepared = nil.as(ML::Video::PreparedWeights?)
    plans = nil.as(ML::Video::WanT5MPSPlans?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      bindings = ML::Video::WanT5.checkpoint_bindings(cfg)
      expected_checkpoint_bytes = bindings.sum do |binding|
        binding.shape.reduce(1_i64) { |product, dimension| product * dimension } * 2_i64
      end
      n = cfg.rows.to_i64 * cfg.dim
      trace_elements = 0_i64
      selected.each do |name|
        shape = trace_shapes[name].as_a.map(&.as_i64)
        expected_shape = if {"gate_pre", "gate", "fc1", "product"}.any? { |suffix| name.ends_with?(suffix) }
                           [cfg.batches.to_i64, cfg.tokens.to_i64, cfg.ffn_dim.to_i64]
                         else
                           [cfg.batches.to_i64, cfg.tokens.to_i64, cfg.dim.to_i64]
                         end
        shape.should eq(expected_shape), "fixture shape for #{name}"
        info = sidecar.tensor(name) || raise "Expected T5 sidecar tensor missing: #{name}"
        info.dtype.should eq(ML::GGUF::SafeTensorDType::F32)
        info.shape.should eq(shape)
        trace_elements += info.n_elements
      end
      masked_shape = trace_shapes["masked_context"].as_a.map(&.as_i64)
      masked_shape.should eq([cfg.batches.to_i64, cfg.tokens.to_i64, cfg.dim.to_i64])
      masked_info = sidecar.tensor("masked_context") || raise "Expected T5 sidecar tensor missing: masked_context"
      masked_info.dtype.should eq(ML::GGUF::SafeTensorDType::F32)
      masked_info.shape.should eq(masked_shape)

      graph_attention = true
      scratch_bytes = ML::Video::WanT5.scratch_bytes(cfg, graph_attention: graph_attention)
      input_bytes = 2_i64 * cfg.rows * sizeof(Float32)
      output_bytes = 2_i64 * n * sizeof(Float32)
      trace_bytes = trace_elements * sizeof(Float32)
      working_bytes = scratch_bytes + input_bytes + output_bytes + trace_bytes
      safety_reserve = Math.max(1_i64 * 1024 * 1024 * 1024,
        (expected_checkpoint_bytes * 15 + 99) // 100)
      reserve_bytes = working_bytes + safety_reserve
      budget_bytes = 24_i64 * 1024 * 1024 * 1024

      plan = ML::Video::WeightPlan.new([checkpoint], bindings, budget_bytes, reserve_bytes,
        resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
      plan.destination_bytes.should eq(expected_checkpoint_bytes)
      plan.staging_bytes.should eq(0_i64)
      plan.reserve_bytes.should eq(reserve_bytes)
      plan.peak_bytes.should be <= budget_bytes
      prepared = ML::Video::PreparedWeights.new(plan, arena)

      ML::Video::WanT5.prepare_scratch!(arena, cfg, graph_attention: graph_attention)
      ML::Video::WanT5.upload_inputs!(arena, fixture["ids"].as_a.map(&.as_i),
        fixture["mask"].as_a.map(&.as_i), cfg)
      output = arena.alloc("t5.full.output", n.to_i32)
      masked = arena.alloc("t5.full.masked", n.to_i32)
      selected.each do |name|
        info = sidecar.tensor(name).not_nil!
        traces.alloc("trace.#{name}", info.n_elements.to_i32)
      end
      plans = ML::Video::WanT5.prepare_mps_plans(cfg, arena, graph_attention: graph_attention)

      arena.resident_bytes.should eq(plan.destination_bytes + scratch_bytes + input_bytes + output_bytes)
      traces.resident_bytes.should eq(trace_bytes)
      arena.resident_bytes + traces.resident_bytes + safety_reserve.should be <= budget_bytes

      trace = {} of String => ML::MetalBuffer
      selected.each { |name| trace[name] = traces["trace.#{name}"] }
      chain = ML::Metal::MPSCommandChain.new
      before_encode_allocations = ML::MetalBuffer.video_spec_allocations
      prepared.not_nil!.encode(chain.not_nil!.encoder)
      ML::Video::WanT5.encode(chain.not_nil!, arena, cfg,
        arena["t5.input.ids"], arena["t5.input.mask"], output,
        plans.not_nil!, trace: trace, source_gelu: true, partial_trace: true)
      ML::Video::WanT5.mask_context(chain.not_nil!.encoder, output,
        arena["t5.input.mask"], masked, cfg)
      ML::MetalBuffer.video_spec_allocations.should eq(before_encode_allocations),
        "full-context encode must allocate no Metal buffers"
      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1), "full-context qualifier must use one terminal wait"
      chain.not_nil!.segment_count.should be > 0, "full-context qualifier must encode command work"

      selected.each do |name|
        expected = sidecar.read_tensor_f32(sidecar.tensor(name).not_nil!)
        actual = traces["trace.#{name}"].read(expected.size.to_i32)
        wan_t5_full_context_assert_gate(name, actual, expected)
      end
      expected_masked = sidecar.read_tensor_f32(masked_info)
      wan_t5_full_context_assert_gate("masked_context", masked.read(expected_masked.size.to_i32), expected_masked)
    ensure
      if active_chain = chain
        active_chain.commit_and_wait unless active_chain.completed?
        active_chain.release
      end
      plans.try(&.release)
      prepared.try(&.release)
      traces.release
      arena.release
      checkpoint.close
      sidecar.close
    end
  end
end
