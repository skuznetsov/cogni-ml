require "spec"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  private def with_drama3_moe_trace_environment(batch : String, &block : -> T) : T forall T
    previous = ENV.to_h
    ENV["DRAMA3_BACKEND"] = "metal"
    ENV["DRAMA3_MOE_BATCH"] = batch
    ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
    ENV.delete("DRAMA3_PROFILE")
    Profile.configure_from_env!
    begin
      yield
    ensure
      MetalBackend.reset!
      {"DRAMA3_BACKEND", "DRAMA3_MOE_BATCH", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_PROFILE"}.each do |key|
        if previous.has_key?(key)
          ENV[key] = previous[key]
        else
          ENV.delete(key)
        end
      end
      Profile.configure_from_env!
    end
  end

  private def drama3_moe_trace_fixture : Tuple(String, BackboneConfig, Array(Array(Float32)))
    path = File.join(__DIR__, "support/drama3/reference_fixture")
    config = BackboneConfig.new(JSON.parse(File.read(File.join(path, "config.json"))))
    rows = JSON.parse(File.read(File.join(path, "inputs.json")))["input_embeddings"].as_a.map do |row|
      row.as_a.map { |value| value.as_f.to_f32 }
    end
    {path, config, rows}
  end

  private def drama3_moe_trace_run_rows(capture : Bool, path : String, config : BackboneConfig, rows : Array(Array(Float32))) : Tuple(Array(Array(UInt32)), Array(UInt32))
    context = Resident::Context.new
    model = BackboneResident.new(config, WeightStore.new(path), context, rows.size.to_i32)
    moe_buffers = {} of String => Resident::Buffer
    layer_buffers = {} of String => Resident::Buffer
    outputs = [] of Resident::Buffer
    rows.each_with_index do |row, index|
      if capture && index == 1
        model.diagnostic_layer = 0
        model.moe_trace_auxiliary = true
        model.moe_trace = ->(name : String, buffer : Resident::Buffer) { moe_buffers[name] = buffer; nil }
        model.layer_buffer_trace = ->(name : String, buffer : Resident::Buffer) { layer_buffers[name] = buffer; nil }
      else
        model.diagnostic_layer = nil
        model.moe_trace = nil
        model.layer_buffer_trace = nil
      end
      outputs << model.step(context.upload(row))
    end

    context.readbacks.values.all?(&.zero?).should be_true
    context.submissions.should eq(0_i64)
    context.graph_stats["completion_waits"].should eq(0_i64)
    context.finish

    output_bits = outputs.map do |output|
      context.read(output, "terminal").map(&.unsafe_as(UInt32))
    end
    hidden_bits = if capture
                    context.read(layer_buffers["hidden_output"], "terminal").map(&.unsafe_as(UInt32))
                  else
                    [] of UInt32
                  end
    {output_bits, hidden_bits}
  end

  describe "DRAMA-3 resident MoE diagnostics" do
    it "captures actual expert stages and ordered prefixes without an intermediate host boundary" do
      with_drama3_moe_trace_environment("1") do
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        path, config, rows = drama3_moe_trace_fixture
        context = Resident::Context.new
        store = WeightStore.new(path)
        model = BackboneResident.new(config, store, context, rows.size.to_i32)
        model.diagnostic_layer = 0
        model.moe_trace_auxiliary = true
        buffers = {} of String => Resident::Buffer
        model.moe_trace = ->(name : String, buffer : Resident::Buffer) { buffers[name] = buffer; nil }

        model.step(context.upload(rows[0]))
        # The fixture stores dense F32 expert matrices, so the configured batch
        # preference correctly falls back to the sequential official path.
        model.last_moe_batched.should be_false
        context.readbacks.values.all?(&.zero?).should be_true
        context.submissions.should eq(0_i64)
        context.graph_stats["completion_waits"].should eq(0_i64)
        {"normalized_input", "router_logits", "expert_ids_argpartition_order", "selected_weights",
         "routed_sum_official", "shared_expert_output", "shared_gate_logits", "moe_output_official",
         "expert_gate_outputs.rank_0", "expert_up_outputs.rank_0", "expert_activation_outputs.rank_0",
         "expert_down_outputs.rank_0", "expert_weighted_outputs.rank_0",
         "routed_prefix_sums_argpartition_order.rank_1"}.each do |name|
          buffers.has_key?(name).should be_true
        end

        context.finish
        actual = context.read(buffers["routed_sum_official"], "terminal")
        final_prefix = context.read(buffers["routed_prefix_sums_argpartition_order.rank_1"], "terminal")
        actual.map(&.unsafe_as(UInt32)).should eq(final_prefix.map(&.unsafe_as(UInt32)))
        activation_width = store.matrix("backbone.layers.0.mlp.switch_mlp.down_proj.weight", 0).cols
        buffers["expert_activation_outputs.rank_0"].count.should eq(activation_width)
      end
    end

    it "keeps full backbone outputs bitwise unchanged when capture is enabled" do
      with_drama3_moe_trace_environment("0") do
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        path, config, rows = drama3_moe_trace_fixture
        baseline_outputs, _ = drama3_moe_trace_run_rows(false, path, config, rows)
        captured_outputs, captured_hidden = drama3_moe_trace_run_rows(true, path, config, rows)

        captured_outputs.should eq(baseline_outputs)
        captured_hidden.size.should eq(config.hidden)
      end
    end

    it "rejects invalid isolated replays and foreign context buffers before readback" do
      with_drama3_moe_trace_environment("0") do
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        path, config, _ = drama3_moe_trace_fixture
        context = Resident::Context.new
        other = Resident::Context.new
        model = BackboneResident.new(config, WeightStore.new(path), context, 1)
        model.diagnostic_layer = 0
        input = context.upload(Array(Float32).new(config.hidden, 0.25_f32))

        expect_raises(ArgumentError, /resident diagnostic layer/) { model.diagnostic_layer = config.layers }
        expect_raises(ArgumentError, /resident diagnostic layer/) { model.diagnostic_layer = -1 }
        expect_raises(ArgumentError, /resident diagnostic layer/) { model.diagnose_moe(input, config.layers) }
        expect_raises(ArgumentError, /resident diagnostic width/) { model.diagnose_moe(context.upload([1_f32]), 0) }
        expect_raises(ArgumentError, /must be selected/) { model.diagnose_moe(input, 1) }
        expect_raises(ArgumentError, /different resident context/) { model.diagnose_moe(other.upload(Array(Float32).new(config.hidden, 1_f32)), 0) }

        context.readbacks.should be_empty
        context.submissions.should eq(0_i64)
        context.graph_stats["completion_waits"].should eq(0_i64)
      end
    end
  end
{% end %}
