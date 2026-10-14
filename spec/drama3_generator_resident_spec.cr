require "spec"
require "../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private class Drama3ResidentEosStore < ML::Audio::Drama3::WeightStore
    def tensor(name : String) : ML::Audio::Drama3::Tensor
      return ML::Audio::Drama3::Tensor.new([5, 8], Array(Float32).new(40, 0_f32)) if name == "lm_head.weight"
      super
    end
  end

  describe ML::Audio::Drama3::ResidentGenerator do
    it "stops on immediate EOS without reading activations or empty frame storage" do
      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = JSON.parse(File.read(File.join(path, "config.json")))
      config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
      context = ML::Audio::Drama3::Resident::Context.new
      generator = ML::Audio::Drama3::ResidentGenerator.new(config, Drama3ResidentEosStore.new(path), context, 3)
      result = generator.generate([[1, 0, 0, 0, 0]], 3, 0_u64, top_k: 1)
      result.generation.stopped.should be_true
      result.generation.frames.should be_empty
      result.deviceframes.count.should eq(15)
      context.readbacks.should eq({"control" => 8_i64})
      expect_raises(ArgumentError, /single-use/) { generator.generate([[1, 0, 0, 0, 0]]) }
    end

    it "maps candidate ids and applies the semantic repeat fallback on device" do
      context = ML::Audio::Drama3::Resident::Context.new
      first = context.upload([1_f32])
      retry = context.upload([2_f32])
      window = context.upload([10_f32, 12_f32])
      selected = context.alloc(1)
      context.dispatch(
        "dr3_choose_main",
        ML::Audio::Drama3::RESIDENT_GENERATOR_SOURCE,
        [first, retry, window, selected],
        [2_u64, 10_u64, 12_u64, 2_u64, 0_u64, 3_u64],
        1,
      )

      context.read(selected, "control")[0].should eq(11_f32)
      context.read(window, "terminal").should eq([11_f32, 12_f32])
      context.readbacks["control"].should eq(4_i64)
    end

    it "matches fixed-uniform CPU sampling and drops the top-p crossing candidate" do
      logits = [2_f32, 1_f32, 0_f32]
      candidates = ML::Audio::Drama3::Sampler.new(0_u64).candidates(logits, 1.0, 3, 0.91)
      candidates.map(&.[0]).should eq([0, 1])

      context = ML::Audio::Drama3::Resident::Context.new
      device_logits = context.upload(logits)
      [0_f32, 0.99_f32].each do |uniform|
        draw = context.upload([uniform])
        actual = context.read(context.sample(device_logits, draw, 1.0, 3, 0.91), "terminal")[0].to_i32
        weights = candidates.sum(&.[1])
        target = uniform.to_f64 * weights
        expected = candidates.last[0]
        candidates.each do |token, weight|
          target -= weight
          if target < 0
            expected = token
            break
          end
        end
        actual.should eq(expected)
      end
      context.readbacks["terminal"].should eq(8_i64)
      context.readbacks["diagnostic"].should eq(0_i64)
    end

    it "accepts the last semantic candidate including repeat fallback and rejects beyond it" do
      [{3_f32, 1_f32}, {1_f32, 3_f32}, {4_f32, 1_f32}].each_with_index do |(first, retry), index|
        context = ML::Audio::Drama3::Resident::Context.new
        selected = context.alloc(1)
        context.dispatch(
          "dr3_choose_main",
          ML::Audio::Drama3::RESIDENT_GENERATOR_SOURCE,
          [context.upload([first]), context.upload([retry]), context.upload([10_f32]), selected],
          [2_u64, 10_u64, 12_u64, 1_u64, 0_u64, 3_u64],
          1,
        )
        if index < 2
          context.read(selected, "control").should eq([12_f32])
        else
          expect_raises(ArgumentError, /non-finite resident control/) { context.read(selected, "control") }
        end
      end
    end

    it "exposes the FP32 cutoff boundary instead of claiming exact CPU sampling" do
      logits = [0_f32, 0_f32]
      top_p = 0.99999999
      cpu = ML::Audio::Drama3::Sampler.new(0_u64).candidates(logits, 1.0, 2, top_p)
      cpu.map(&.[0]).should eq([0])
      context = ML::Audio::Drama3::Resident::Context.new
      sampled = context.sample(context.upload(logits), context.upload([0.75_f32]), 1.0, 2, top_p)
      context.read(sampled, "terminal").should eq([1_f32])
    end

    it "rejects a non-finite sampled candidate before integer conversion" do
      context = ML::Audio::Drama3::Resident::Context.new
      first = context.upload([Float32::NAN])
      retry = context.upload([2_f32])
      window = context.upload([10_f32])
      selected = context.alloc(1)
      context.dispatch(
        "dr3_choose_main",
        ML::Audio::Drama3::RESIDENT_GENERATOR_SOURCE,
        [first, retry, window, selected],
        [2_u64, 10_u64, 12_u64, 1_u64, 0_u64, 3_u64],
        1,
      )

      expect_raises(ArgumentError, "non-finite resident control") do
        context.read(selected, "control")
      end
    end

    it "keeps first-index tie order and rejects non-finite logits" do
      context = ML::Audio::Drama3::Resident::Context.new
      tied = context.upload([2_f32, 2_f32, -1000_f32])
      [0.0, 1.0].each do |temperature|
        output = context.sample(tied, context.upload([0.75_f32]), temperature, 1, 1.0)
        context.read(output, "terminal").should eq([0_f32])
      end
      invalid = context.sample(context.upload([0_f32, Float32::NAN]), context.upload([0_f32]), 1.0, 2, 1.0)
      expect_raises(ArgumentError, /non-finite resident intermediate/) { context.read(invalid, "terminal") }
    end
  end
{% end %}
