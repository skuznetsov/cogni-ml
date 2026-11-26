require "spec"
require "../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private class Drama3GeneratorWorkspaceHeadStore < ML::Audio::Drama3::WeightStore
    def tensor(name : String) : ML::Audio::Drama3::Tensor
      return ML::Audio::Drama3::Tensor.new([5, 8], Array(Float32).new(40, 0_f32)) if name == "lm_head.weight"
      super
    end
  end

  private struct Drama3GeneratorWorkspaceObservation
    getter frames : Array(Array(Int32))
    getter stopped : Bool
    getter backbone_position : Int32
    getter tower_offset : Int32
    getter sampler_inputs : NamedTuple(dtype: String, uniform_count: Int32, sha256: String, provenance: String)?
    getter readbacks : Hash(String, Int64)
    getter graph_ledger : Hash(String, Int64)
    getter workspace_stats : NamedTuple(enabled: Bool, workspaces: Hash(String, Hash(String, Int64)))

    def initialize(@frames : Array(Array(Int32)), @stopped : Bool, @backbone_position : Int32,
                   @tower_offset : Int32,
                   @sampler_inputs : NamedTuple(dtype: String, uniform_count: Int32, sha256: String, provenance: String)?,
                   @readbacks : Hash(String, Int64), @graph_ledger : Hash(String, Int64),
                   @workspace_stats : NamedTuple(enabled: Bool, workspaces: Hash(String, Hash(String, Int64))))
    end
  end

  private struct Drama3GeneratorWorkspaceDiagnosticObservation
    getter names : Array(String)
    getter values : Hash(String, Array(UInt32))
    getter readbacks : Hash(String, Int64)
    getter workspace_stats : NamedTuple(enabled: Bool, workspaces: Hash(String, Hash(String, Int64)))

    def initialize(@names : Array(String), @values : Hash(String, Array(UInt32)),
                   @readbacks : Hash(String, Int64),
                   @workspace_stats : NamedTuple(enabled: Bool, workspaces: Hash(String, Hash(String, Int64))))
    end
  end

  private def drama3_generator_workspace_seed_for_distinct_semantics : UInt64
    seed = 0_u64
    loop do
      random = Random.new(seed)
      draws = Array(Float32).new(16) { random.rand.to_f32 }
      semantics = [0, 5, 10].map { |index| (draws[index] * 5_f32).to_i32 }
      return seed if semantics.all? { |token| token.in?(2..4) } && semantics.uniq.size == semantics.size
      seed += 1
    end
  end

  private def drama3_generator_workspace_run(depth : Int32, scratch : Bool, eos_first : Bool,
                                             max_frames : Int32) : Drama3GeneratorWorkspaceObservation
    previous_backend = ENV["DRAMA3_BACKEND"]?
    previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
    previous_scratch = ENV["DRAMA3_GENERATION_SCRATCH"]?
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
      ENV["DRAMA3_GENERATION_SCRATCH"] = scratch ? "1" : "0"
      pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = JSON.parse(File.read(File.join(path, "config.json")))
      config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
      context = ML::Audio::Drama3::Resident::Context.new
      generator = ML::Audio::Drama3::ResidentGenerator.new(config, Drama3GeneratorWorkspaceHeadStore.new(path), context, 4)
      seed = eos_first ? 0_u64 : drama3_generator_workspace_seed_for_distinct_semantics
      result = generator.generate([[20, 0, 0, 0, 0]], max_frames, seed,
        top_k: eos_first ? 1 : 5, top_p: eos_first ? 0.9 : 1.0)
      graph = context.graph_stats
      ledger = {
        "submitted"        => graph["submitted"],
        "completed"        => graph["completed"],
        "pending"          => graph["pending"],
        "completion_waits" => graph["completion_waits"],
        "max_pending"      => graph["max_pending"],
      }
      Drama3GeneratorWorkspaceObservation.new(
        result.generation.frames,
        result.generation.stopped,
        generator.backbone.position,
        generator.tower.offset,
        generator.sampler_inputs,
        context.readbacks.dup,
        ledger,
        generator.workspace_stats,
      )
    ensure
      if previous_backend
        ENV["DRAMA3_BACKEND"] = previous_backend
      else
        ENV.delete("DRAMA3_BACKEND")
      end
      if previous_depth
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
      else
        ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
      end
      if previous_scratch
        ENV["DRAMA3_GENERATION_SCRATCH"] = previous_scratch
      else
        ENV.delete("DRAMA3_GENERATION_SCRATCH")
      end
    end
  end

  private def drama3_generator_workspace_compare(fresh : Drama3GeneratorWorkspaceObservation,
                                                 reused : Drama3GeneratorWorkspaceObservation) : Nil
    reused.frames.should eq(fresh.frames)
    reused.stopped.should eq(fresh.stopped)
    reused.backbone_position.should eq(fresh.backbone_position)
    reused.tower_offset.should eq(fresh.tower_offset)
    reused.sampler_inputs.should eq(fresh.sampler_inputs)
    reused.readbacks.should eq(fresh.readbacks)
    reused.graph_ledger.should eq(fresh.graph_ledger)
    reused.graph_ledger["pending"].should eq(0_i64)
    reused.graph_ledger["completed"].should eq(reused.graph_ledger["submitted"])

    fresh.workspace_stats[:enabled].should be_false
    fresh.workspace_stats[:workspaces].should be_empty
    reused.workspace_stats[:enabled].should be_true
    reused.workspace_stats[:workspaces].should_not be_empty
    reused.workspace_stats[:workspaces].each_value do |stats|
      stats["released"].should eq(1_i64)
      stats["slots"].should eq(0_i64)
      stats["bytes"].should eq(0_i64)
    end
  end

  private def drama3_generator_workspace_diagnostic_run(depth : Int32, scratch : Bool) : Drama3GeneratorWorkspaceDiagnosticObservation
    previous_backend = ENV["DRAMA3_BACKEND"]?
    previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
    previous_scratch = ENV["DRAMA3_GENERATION_SCRATCH"]?
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
      ENV["DRAMA3_GENERATION_SCRATCH"] = scratch ? "1" : "0"
      pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = JSON.parse(File.read(File.join(path, "config.json")))
      config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
      context = ML::Audio::Drama3::Resident::Context.new
      generator = ML::Audio::Drama3::ResidentGenerator.new(config, Drama3GeneratorWorkspaceHeadStore.new(path), context, 4)
      traces = [] of Tuple(String, ML::Audio::Drama3::Resident::Buffer)
      generator.device_trace = ->(name : String, buffer : ML::Audio::Drama3::Resident::Buffer) do
        traces << {name, buffer}
        nil
      end
      generator.generate([[20, 0, 0, 0, 0]], 2, 0_u64, top_k: 1, top_p: 0.9)

      values = {} of String => Array(UInt32)
      traces.each do |name, buffer|
        values[name] = context.read(buffer, "diagnostic").map(&.unsafe_as(UInt32))
      end
      Drama3GeneratorWorkspaceDiagnosticObservation.new(
        traces.map(&.[0]), values, context.readbacks.dup, generator.workspace_stats,
      )
    ensure
      if previous_backend
        ENV["DRAMA3_BACKEND"] = previous_backend
      else
        ENV.delete("DRAMA3_BACKEND")
      end
      if previous_depth
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
      else
        ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
      end
      if previous_scratch
        ENV["DRAMA3_GENERATION_SCRATCH"] = previous_scratch
      else
        ENV.delete("DRAMA3_GENERATION_SCRATCH")
      end
    end
  end

  describe "DRAMA-3 generator allocation workspace integration" do
    it "matches fresh buffers at depths zero and two for EOS and three generated frames" do
      [0, 2].each do |depth|
        eos_fresh = drama3_generator_workspace_run(depth, false, true, 3)
        eos_reused = drama3_generator_workspace_run(depth, true, true, 3)
        eos_fresh.frames.should be_empty
        eos_fresh.stopped.should be_true
        eos_fresh.readbacks.should eq({"control" => 8_i64})
        drama3_generator_workspace_compare(eos_fresh, eos_reused)

        fresh = drama3_generator_workspace_run(depth, false, false, 3)
        reused = drama3_generator_workspace_run(depth, true, false, 3)
        fresh.frames.size.should eq(3)
        fresh.stopped.should be_false
        fresh.frames.each do |frame|
          frame.size.should eq(5)
          (21..23).includes?(frame[0]).should be_true
        end
        fresh.readbacks["terminal"].should eq(60_i64)
        drama3_generator_workspace_compare(fresh, reused)
      end
    end

    it "keeps captured hidden and logits valid when diagnostic callbacks disable reuse" do
      fresh = drama3_generator_workspace_diagnostic_run(2, false)
      requested = drama3_generator_workspace_diagnostic_run(2, true)
      fresh.names.should eq(["prompt.backbone.hidden", "prompt.main_logits"])
      requested.names.should eq(fresh.names)
      requested.values.should eq(fresh.values)
      requested.readbacks.should eq(fresh.readbacks)
      requested.readbacks["diagnostic"].should eq(requested.values.values.sum(&.size).to_i64 * 4_i64)
      requested.workspace_stats[:enabled].should be_false
      requested.workspace_stats[:workspaces].should be_empty
      requested.values.each_value { |values| values.all? { |value| value.unsafe_as(Float32).finite? }.should be_true }
    end

    it "rejects diagnostic trace enablement after workspace-backed prefill before exposing a borrowed buffer" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      previous_scratch = ENV["DRAMA3_GENERATION_SCRATCH"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ENV["DRAMA3_GENERATION_SCRATCH"] = "1"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        path = File.join(__DIR__, "support/drama3/reference_fixture")
        config = JSON.parse(File.read(File.join(path, "config.json")))
        config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
        context = ML::Audio::Drama3::Resident::Context.new
        generator = ML::Audio::Drama3::ResidentGenerator.new(config, Drama3GeneratorWorkspaceHeadStore.new(path), context, 4)
        captured = [] of String
        generator.device_trace = nil
        generator.progress = ->(message : String) do
          if message.starts_with?("prefill ")
            generator.device_trace = ->(name : String, _buffer : ML::Audio::Drama3::Resident::Buffer) do
              captured << name
              nil
            end
          end
          nil
        end

        expect_raises(ArgumentError, /cannot enable resident traces during workspace generation/) do
          generator.generate([[20, 0, 0, 0, 0]], 3, drama3_generator_workspace_seed_for_distinct_semantics, top_k: 5, top_p: 1.0)
        end
        captured.should be_empty
        context.readbacks.should be_empty
        context.graph_stats["pending"].should eq(0_i64)
        context.graph_stats["completed"].should eq(context.graph_stats["submitted"])
        context.graph_stats["submitted"].should eq(1_i64)
        generator.workspace_stats[:workspaces].should_not be_empty
        generator.workspace_stats[:workspaces].each_value do |stats|
          stats["quarantined"].should eq(1_i64)
        end
      ensure
        if previous_backend
          ENV["DRAMA3_BACKEND"] = previous_backend
        else
          ENV.delete("DRAMA3_BACKEND")
        end
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
        if previous_scratch
          ENV["DRAMA3_GENERATION_SCRATCH"] = previous_scratch
        else
          ENV.delete("DRAMA3_GENERATION_SCRATCH")
        end
      end
    end
  end
{% end %}
