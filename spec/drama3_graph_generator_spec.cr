require "spec"
require "../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private class Drama3GraphGeneratorHeadStore < ML::Audio::Drama3::WeightStore
    def tensor(name : String) : ML::Audio::Drama3::Tensor
      return ML::Audio::Drama3::Tensor.new([5, 8], Array(Float32).new(40, 0_f32)) if name == "lm_head.weight"
      super
    end
  end

  private struct Drama3GraphGeneratorObservation
    getter frames : Array(Array(Int32))
    getter stopped : Bool
    getter backbone_position : Int32
    getter tower_offset : Int32
    getter readbacks : Hash(String, Int64)
    getter graph_stats : Hash(String, Int64)

    def initialize(@frames : Array(Array(Int32)), @stopped : Bool, @backbone_position : Int32,
                   @tower_offset : Int32, @readbacks : Hash(String, Int64), @graph_stats : Hash(String, Int64))
    end
  end

  private def drama3_graph_generator_seed_for_two_distinct_semantics : UInt64
    # With the synthetic zero head, candidate ties preserve index order. Pick
    # seed draws that select distinct semantic candidates in both bounded rows.
    seed = 0_u64
    loop do
      random = Random.new(seed)
      draws = Array(Float32).new(11) { random.rand.to_f32 }
      first = (draws[0] * 5_f32).to_i32
      second = (draws[5] * 5_f32).to_i32
      return seed if first.in?(2..4) && second.in?(2..4) && first != second
      seed += 1
    end
  end

  private def drama3_graph_generator_run(depth : Int32, eos_first : Bool, max_frames : Int32) : Drama3GraphGeneratorObservation
    previous_backend = ENV["DRAMA3_BACKEND"]?
    previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
      pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = JSON.parse(File.read(File.join(path, "config.json")))
      config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
      context = ML::Audio::Drama3::Resident::Context.new
      generator = ML::Audio::Drama3::ResidentGenerator.new(config, Drama3GraphGeneratorHeadStore.new(path), context, 4)
      seed = eos_first ? 0_u64 : drama3_graph_generator_seed_for_two_distinct_semantics
      top_k = eos_first ? 1 : 5
      top_p = eos_first ? 0.9 : 1.0
      result = generator.generate([[20, 0, 0, 0, 0]], max_frames, seed, top_k: top_k, top_p: top_p)
      Drama3GraphGeneratorObservation.new(
        result.generation.frames,
        result.generation.stopped,
        generator.backbone.position,
        generator.tower.offset,
        context.readbacks.dup,
        context.graph_stats.dup,
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
    end
  end

  describe "DRAMA-3 generator CogniGraph handoff" do
    it "preserves EOS-first state and exact control readbacks at depths zero, one, and two" do
      observations = [0, 1, 2].map { |depth| drama3_graph_generator_run(depth, true, 3) }

      observations.each do |observation|
        observation.frames.should be_empty
        observation.stopped.should be_true
        observation.backbone_position.should eq(1)
        observation.tower_offset.should eq(0)
        observation.readbacks.should eq({"control" => 8_i64})
      end
      observations.map(&.frames).uniq.size.should eq(1)
      observations.map(&.stopped).uniq.size.should eq(1)
      observations.map(&.backbone_position).uniq.size.should eq(1)
      observations.map(&.tower_offset).uniq.size.should eq(1)
      observations.map(&.readbacks).uniq.size.should eq(1)
    end

    it "preserves bounded non-EOS frames while combining each frame with the next main sample" do
      legacy, depth_one, depth_two = [0, 1, 2].map { |depth| drama3_graph_generator_run(depth, false, 2) }
      [legacy, depth_one, depth_two].each do |observation|
        observation.frames.size.should eq(2)
        observation.frames.each do |frame|
          frame.size.should eq(5)
          (21..23).includes?(frame[0]).should be_true
        end
        observation.stopped.should be_false
        observation.backbone_position.should eq(2)
        observation.tower_offset.should eq(2)
        observation.readbacks.should eq({"control" => 20_i64, "terminal" => 40_i64})
      end
      [depth_one, depth_two].each do |observation|
        observation.frames.should eq(legacy.frames)
        observation.stopped.should eq(legacy.stopped)
        observation.backbone_position.should eq(legacy.backbone_position)
        observation.tower_offset.should eq(legacy.tower_offset)
        observation.readbacks.should eq(legacy.readbacks)
        (1_i64..2_i64).includes?(observation.graph_stats["depth"]).should be_true
      end
      unless ML::Audio::Drama3::Profile.enabled?
        depth_one.graph_stats["submitted"].should be < legacy.graph_stats["submitted"]
        depth_two.graph_stats["submitted"].should be < legacy.graph_stats["submitted"]
        depth_one.graph_stats["completion_waits"].should be < legacy.graph_stats["completion_waits"]
        depth_two.graph_stats["completion_waits"].should be < legacy.graph_stats["completion_waits"]
      end
    end

    it "samples into a nonzero-offset output view without changing sentinel neighbors" do
      previous_backend = ENV["DRAMA3_BACKEND"]?
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

        path = File.join(__DIR__, "support/drama3/reference_fixture")
        config = ML::Audio::Drama3::TowerConfig.from_json(JSON.parse(File.read(File.join(path, "config.json"))).to_json)
        reference_context = ML::Audio::Drama3::Resident::Context.new
        reference_tower = ML::Audio::Drama3::TowerResident.new(config, ML::Audio::Drama3::WeightStore.new(path), reference_context, 2)
        reference_frame = reference_tower.sample_frame(
          reference_tower.upload_token(20),
          reference_tower.upload_hidden(Array(Float32).new(config.dim, 0_f32)),
          reference_tower.upload_frame([20, 0, 0, 0, 0]),
        ) do |logits|
          reference_context.sample(logits, reference_context.upload([0_f32]), 1.0, 1, 1.0)
        end
        expected = reference_context.read(reference_frame, "terminal")
        context = ML::Audio::Drama3::Resident::Context.new
        tower = ML::Audio::Drama3::TowerResident.new(config, ML::Audio::Drama3::WeightStore.new(path), context, 2)
        main = tower.upload_token(20)
        hidden = tower.upload_hidden(Array(Float32).new(config.dim, 0_f32))
        previous_frame = tower.upload_frame([20, 0, 0, 0, 0])
        draw = context.upload([0_f32])
        output = context.upload([111_f32, 222_f32, -333_f32, 444_f32, 555_f32, 666_f32, 777_f32, 888_f32])
        output_frame = output.slice(2, 5)

        wrong_width = context.upload([1_f32, 2_f32, 3_f32, 4_f32])
        expect_raises(ArgumentError, /output_frame has width/) do
          tower.sample_frame(main, hidden, previous_frame, output_frame: wrong_width) do |logits|
            context.sample(logits, draw, 1.0, 1, 1.0)
          end
        end
        tower.offset.should eq(0)
        context.submissions.should eq(0)

        main_alias = output_frame.slice(0, 1)
        expect_raises(ArgumentError, /output_frame overlaps main token/) do
          tower.sample_frame(main_alias, hidden, previous_frame, output_frame: output_frame) do |logits|
            context.sample(logits, draw, 1.0, 1, 1.0)
          end
        end
        exact_previous_alias = output_frame
        expect_raises(ArgumentError, /output_frame overlaps previous_frame/) do
          tower.sample_frame(main, hidden, exact_previous_alias, output_frame: output_frame) do |logits|
            context.sample(logits, draw, 1.0, 1, 1.0)
          end
        end
        partial_previous_alias = output.slice(1, 5)
        expect_raises(ArgumentError, /output_frame overlaps previous_frame/) do
          tower.sample_frame(main, hidden, partial_previous_alias, output_frame: output_frame) do |logits|
            context.sample(logits, draw, 1.0, 1, 1.0)
          end
        end
        tower.offset.should eq(0)
        context.submissions.should eq(0)

        frame = tower.sample_frame(main, hidden, previous_frame, output_frame: output_frame) do |logits|
          context.sample(logits, draw, 1.0, 1, 1.0)
        end
        frame.offset.should eq(output_frame.offset)
        frame.count.should eq(5)
        context.read(output, "terminal").should eq([111_f32, 222_f32] + expected + [888_f32])
        tower.offset.should eq(1)

        expect_raises(ArgumentError, /output_frame overlaps sampled code/) do
          tower.sample_frame(main, hidden, previous_frame, output_frame: output_frame) do |_logits|
            output_frame.slice(2, 1)
          end
        end
        tower.offset.should eq(1)
        context.abort
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
      end
    end

    it "aborts queued prefill and an open generated row while preserving callback failures" do
      ["prefill ", "frame "].each do |failure_prefix|
        previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
        begin
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
          path = File.join(__DIR__, "support/drama3/reference_fixture")
          config = JSON.parse(File.read(File.join(path, "config.json")))
          config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
          context = ML::Audio::Drama3::Resident::Context.new
          generator = ML::Audio::Drama3::ResidentGenerator.new(config, Drama3GraphGeneratorHeadStore.new(path), context, 4)
          generator.progress = ->(message : String) { raise "resident callback failed" if message.starts_with?(failure_prefix); nil }
          expect_raises(Exception, "resident callback failed") do
            generator.generate([[20, 0, 0, 0, 0]], 2, drama3_graph_generator_seed_for_two_distinct_semantics, top_k: 5, top_p: 1.0)
          end
          context.graph_stats["pending"].should eq(0)
          context.graph_stats["completed"].should eq(context.graph_stats["submitted"])
          context.graph_stats["completed"].should be > 0
          context.graph_stats["retained_bytes"].should eq(0)
          expect_raises(ArgumentError, /aborted/) { context.zeros(1) }
        ensure
          if previous
            ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous
          else
            ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
          end
        end
      end
    end
  end
{% end %}
