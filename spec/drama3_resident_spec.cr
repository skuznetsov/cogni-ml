require "spec"
require "../src/ml/audio/drama3/resident"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  describe ML::Audio::Drama3::Resident do
    it "preserves recurrent history across ring wrap and growing full-attention caches" do
      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = ML::Audio::Drama3::BackboneConfig.new(JSON.parse(File.read(File.join(path, "config.json"))))
      store = ML::Audio::Drama3::WeightStore.new(path)
      cpu = ML::Audio::Drama3::Backbone.new(config, store)
      ctx = ML::Audio::Drama3::Resident::Context.new
      gpu = ML::Audio::Drama3::BackboneResident.new(config, store, ctx, 8)
      rows = JSON.parse(File.read(File.join(path, "inputs.json")))["input_embeddings"].as_a
      8.times do |index|
        input = rows[index % rows.size].as_a.map { |v| v.as_f.to_f32 }
        expected = cpu.step(input)
        actual = ctx.read(gpu.step(ctx.upload(input)), "terminal")
        actual.zip(expected).each { |a, b| a.should be_close(b, 3e-5) }
      end
      gpu.position.should eq(8)
      expect_raises(ArgumentError, /capacity exceeded/) { gpu.step(ctx.upload(Array(Float32).new(config.hidden, 0_f32))) }
      ctx.readbacks["diagnostic"].should eq(0_i64)
    end

    it "matches upstream backbone boundaries across recurrent and full attention cache updates" do
      path = File.join(__DIR__, "support/drama3/reference_fixture")
      expected = {} of String => ML::Audio::Drama3::Vector
      JSON.parse(File.read(File.join(path, "reference_traces.json")))["traces"].as_a.each do |record|
        name = record["name"].as_s
        expected[name] = record["values"].as_a.map { |v| v.as_f.to_f32 } if name.starts_with?("backbone.")
      end
      ctx = ML::Audio::Drama3::Resident::Context.new
      model = ML::Audio::Drama3::BackboneResident.new(ML::Audio::Drama3::BackboneConfig.new(JSON.parse(File.read(File.join(path, "config.json")))), ML::Audio::Drama3::WeightStore.new(path), ctx, 4)
      row = 0
      observed = [] of String
      model.trace = ->(name : String, values : ML::Audio::Drama3::Vector) {
        key = name.sub("backbone.", "backbone.row#{row}.")
        values.zip(expected[key]).each { |a, b| a.should be_close(b, 1e-5) }
        observed << key
        nil
      }
      JSON.parse(File.read(File.join(path, "inputs.json")))["input_embeddings"].as_a.each_with_index do |input, index|
        row = index
        model.step(ctx.upload(input.as_a.map { |v| v.as_f.to_f32 }))
      end
      observed.sort.should eq(expected.keys.sort)
      model.position.should eq(3)
      ctx.readbacks.keys.sort.should eq(["control", "diagnostic"])
    end
    it "keeps a dependent norm/projection chain resident and rejects unlabelled readback" do
      ctx = ML::Audio::Drama3::Resident::Context.new
      x = ctx.upload([1_f32, -2_f32, 3_f32, 0.5_f32])
      w = ctx.upload([0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32])
      matrix = ML::Audio::Drama3::Matrix.new(2, 4, [1_f32, 2_f32, 3_f32, 4_f32, -1_f32, 0_f32, 2_f32, 0.5_f32])
      y = ctx.matvec(matrix, ctx.rms(x, w, 1e-6, true))
      expect_raises(ArgumentError, /readback purpose/) { ctx.read(y, "activation") }
      ctx.readbacks.empty?.should be_true
      actual = ctx.read(y, "terminal")
      expected = matrix.matvec(ML::Audio::Drama3::Ops.rms_norm([1_f32, -2_f32, 3_f32, 0.5_f32], [0.1_f32, -0.2_f32, 0.3_f32, 0.4_f32], 1e-6, true))
      actual.zip(expected).each { |a, b| (a - b).abs.should be < 1e-5_f32 }
      ctx.readbacks["terminal"].should eq(8_i64)
    end

    it "rejects a non-finite intermediate even when the terminal tensor is finite" do
      ctx = ML::Audio::Drama3::Resident::Context.new
      ctx.guard(ctx.upload([Float32::NAN]))
      terminal = ctx.upload([1_f32])
      expect_raises(ArgumentError, /non-finite resident intermediate/) { ctx.read(terminal, "terminal") }
      ctx.readbacks["control"].should eq(4_i64)
      ctx.readbacks["terminal"].should eq(0_i64)
    end

    it "rejects an out-of-range device expert id before reading weight storage" do
      store = ML::Audio::Drama3::WeightStore.new(File.join(__DIR__, "support/drama3/reference_fixture"))
      matrix = store.matrix("affine_fixture.q4.weight")
      ctx = ML::Audio::Drama3::Resident::Context.new
      input = ctx.upload(Array(Float32).new(matrix.cols, 1_f32))
      output = ctx.expert(matrix, input, ctx.upload([999_f32]))
      expect_raises(ArgumentError, /non-finite resident terminal/) { ctx.read(output, "terminal") }
      expect_raises(ArgumentError, /expert rank/) { ctx.expert(matrix, input, ctx.upload([0_f32]), 1) }
    end
  end
{% end %}
