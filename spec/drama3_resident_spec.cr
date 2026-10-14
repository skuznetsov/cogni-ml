require "spec"
require "../src/ml/audio/drama3/resident"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  describe ML::Audio::Drama3::Resident do
    it "batches affine expert ranks exactly with shared or per-rank inputs and unaligned weights" do
      path = File.tempname("drama3-resident-experts", ".bin")
      begin
        [4, 8].each do |bits|
          [1, 3, 65].each do |groups|
            rows, experts, group = 5, 3, 64
            cols, per = groups * group, 32 // bits
            io = IO::Memory.new
            io.write(Bytes.new(22, 0_u8))
            words = experts * rows * cols // per
            words.times do |i|
              word = 0_u32
              per.times { |lane| word |= (((i * 13 + lane * 7) % (1 << bits)).to_u32 << (lane * bits)) }
              io.write_bytes(word, IO::ByteFormat::LittleEndian)
            end
            File.write(path, io.to_slice)
            owner = ML::Audio::Drama3::MappedFile.new(path)
            packed = ML::Audio::Drama3::Tensor.new([experts, rows, cols // per], "U32", owner.slice(22_i64, words.to_i64 * 4), owner)
            scales = ML::Audio::Drama3::Tensor.new([experts, rows, groups], Array(Float32).new(experts * rows * groups) { |i| (i % 9 + 1).to_f32 * 0.00073_f32 })
            biases = ML::Audio::Drama3::Tensor.new([experts, rows, groups], Array(Float32).new(experts * rows * groups) { |i| (i % 11 - 5).to_f32 * 0.017_f32 })
            matrix = ML::Audio::Drama3::Matrix.new(packed, scales, biases, bits, group)
            [false, true].each do |per_rank|
              ctx = ML::Audio::Drama3::Resident::Context.new
              ids = ctx.upload([2_f32, 0_f32, 2_f32])
              input = ctx.upload(Array(Float32).new(cols * (per_rank ? 3 : 1)) { |i| (i % 17 - 8).to_f32 / 7_f32 })
              batched = ctx.experts(matrix, input, ids, per_rank)
              3.times do |rank|
                x = per_rank ? input.slice(rank * cols, cols) : input
                expected = ctx.read(ctx.expert(matrix, x, ids, rank), "terminal")
                actual = ctx.read(batched.slice(rank * rows, rows), "terminal")
                actual.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))
              end
              bad = ctx.experts(matrix, input, ctx.upload([0_f32, 3_f32, 0.5_f32]), per_rank)
              expect_raises(ArgumentError, /non-finite resident terminal/) { ctx.read(bad, "terminal") }
              expect_raises(ArgumentError, /input width/) { ctx.experts(matrix, ctx.upload([1_f32]), ids, per_rank) }
            end
            # All commands have completed; release mapped views before the
            # next case rewrites this temporary backing file.
            ML::Audio::Drama3::MetalBackend.reset!
          end
        end
      ensure
        ML::Audio::Drama3::MetalBackend.reset!
        File.delete(path) if File.exists?(path)
      end
    end

    it "preserves rank-ordered expert accumulation under cancellation" do
      ctx = ML::Audio::Drama3::Resident::Context.new
      ranks, width = 8, 3
      terms = [1e20_f32, -1e20_f32, 3_f32, 2_f32, -7_f32, 9_f32, -2_f32, 1_f32]
      values = ctx.upload(Array(Float32).new(ranks * width) { |i| terms[i // width] * (i % width + 1).to_f32 })
      routing = ctx.upload(Array(Float32).new(ranks, 0_f32) + Array(Float32).new(ranks, 1_f32))
      serial = ctx.zeros(width)
      ranks.times do |rank|
        ctx.dispatch("dr3_weighted_add", ML::Audio::Drama3::BackboneResident::SOURCE,
          [serial, values.slice(rank * width, width), routing.slice(ranks + rank, 1)], [width.to_u64, 0_u64], width)
      end
      batched = ctx.alloc(width)
      ctx.dispatch("dr3_expert_sum", ML::Audio::Drama3::BackboneResident::SOURCE,
        [values, routing, batched], [width.to_u64, ranks.to_u64], width)
      expected = ctx.read(serial, "terminal")
      expected[0].should eq(6_f32)
      ctx.read(batched, "terminal").map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))
    end

    it "preserves exact Delta outputs and every recurrent state value with transposed storage" do
      # Unequal, non-SIMD-aligned dimensions catch head/key/value index swaps.
      [{6, 7, 2, 5}, {32, 128, 16, 128}].each do |shape|
        vh, vd, kh, kd = shape
        ctx = ML::Audio::Drama3::Resident::Context.new
        sequence = ->(n : Int32, phase : Int32) { Array(Float32).new(n) { |i| (((i * 13 + phase) % 31) - 15).to_f32 / 128_f32 } }
        initial = sequence.call(vh * vd * kd, 4)
        transposed = Array(Float32).new(initial.size) do |i|
          h, rem = i // (kd * vd), i % (kd * vd)
          initial[(h * vd + rem % vd) * kd + rem // vd]
        end
        states = [ctx.upload(initial), ctx.upload(transposed)]
        names = ["dr3_delta_step", "dr3_delta_step_transposed"]
        4.times do |row|
          q, k = ctx.upload(sequence.call(kh * kd, row)), ctx.upload(sequence.call(kh * kd, row + 7))
          v = ctx.upload(sequence.call(vh * vd, row + 11))
          parameters = [0, 3, 9, 15].map { |phase| ctx.upload(sequence.call(vh, phase)) }
          outputs = names.map_with_index do |name, index|
            output = ctx.alloc(vh * vd)
            ctx.dispatch(name, ML::Audio::Drama3::BackboneResident::SOURCE, [q, k, v] + parameters + [states[index], output], shape.to_a.map(&.to_u64), output.count)
            output
          end
          ctx.read(outputs[1], "terminal").map(&.unsafe_as(UInt32)).should eq(ctx.read(outputs[0], "terminal").map(&.unsafe_as(UInt32)))
          legacy = ctx.read(states[0], "terminal")
          current = ctx.read(states[1], "terminal")
          current.each_with_index do |value, i|
            h, rem = i // (kd * vd), i % (kd * vd)
            value.unsafe_as(UInt32).should eq(legacy[(h * vd + rem % vd) * kd + rem // vd].unsafe_as(UInt32))
          end
        end
      end
    end

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
