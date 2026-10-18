require "spec"
require "../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  # Test-only state oracle; inference never materializes these activations.
  class ML::Audio::Drama3::BackboneResident
    def prefill_test_state : Array(Resident::Buffer)
      result = [] of Resident::Buffer
      @config.layers.times do |layer|
        if keys = @keys[layer]
          count = @position * @config.kv_heads * @config.head_dim
          result << keys.slice(0, count)
          result << @values[layer].not_nil!.slice(0, count)
        else
          result << @conv[layer].not_nil!
          result << @delta[layer].not_nil!
        end
      end
      result
    end
  end

  describe "DRAMA-3 genuine layer-major prefill" do
    it "preserves every residual, temporal state and decode continuation over chunk boundaries" do
      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = ML::Audio::Drama3::BackboneConfig.new(JSON.parse(File.read(File.join(path, "config.json"))))
      store = ML::Audio::Drama3::WeightStore.new(path)
      inputs = Array(Array(Float32)).new(9) { |row| Array(Float32).new(config.hidden) { |i| Math.sin(row * 0.37 + i * 0.23).to_f32 } }
      previous_depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      previous_gdn_post = ENV["DRAMA3_GDN_POST_FUSION"]?
      begin
        [0, 2].each do |depth|
          ["0", "1"].each do |gdn_post|
            ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
            ENV["DRAMA3_GDN_POST_FUSION"] = "0"
            scalar_ctx = ML::Audio::Drama3::Resident::Context.new
            ENV["DRAMA3_GDN_POST_FUSION"] = gdn_post
            batch_ctx = ML::Audio::Drama3::Resident::Context.new
            scalar = ML::Audio::Drama3::BackboneResident.new(config, store, scalar_ctx, 9)
            batch = ML::Audio::Drama3::BackboneResident.new(config, store, batch_ctx, 9)
            expected = Hash(String, Array(ML::Audio::Drama3::Resident::Buffer)).new { |h, k| h[k] = [] of ML::Audio::Drama3::Resident::Buffer }
            observed = {} of String => ML::Audio::Drama3::Resident::Buffer
            scalar.buffer_trace = ->(name : String, value : ML::Audio::Drama3::Resident::Buffer) { expected[name] << value; nil }
            batch.prefill_trace = ->(name : String, value : ML::Audio::Drama3::Resident::Buffer) { observed[name] = value; nil }

            # Nonzero base, N=2 followed by N=5 crosses warmup and ring wrap.
            scalar.step(scalar_ctx.upload(inputs[0]))
            batch.step(batch_ctx.upload(inputs[0]))
            cursor = 1
            [2, 5].each do |rows|
              expected.clear
              scalar_reads = scalar_ctx.readbacks.dup
              batch_reads = batch_ctx.readbacks.dup
              rows.times { |i| scalar.step(scalar_ctx.upload(inputs[cursor + i])); scalar_ctx.submit }
              output = batch.prefill(batch_ctx.upload(inputs[cursor, rows].flatten), rows)
              batch_ctx.submit
              output.count.should eq(rows * config.hidden)
              scalar_ctx.readbacks.should eq(scalar_reads)
              batch_ctx.readbacks.should eq(batch_reads)
              observed.keys.sort.should eq(expected.keys.sort)
              expected.each do |name, buffers|
                actual = batch_ctx.read(observed[name], "diagnostic")
                reference = buffers.flat_map { |buffer| scalar_ctx.read(buffer, "diagnostic") }
                actual.map(&.unsafe_as(UInt32)).should eq(reference.map(&.unsafe_as(UInt32)))
              end
              scalar.prefill_test_state.zip(batch.prefill_test_state).each do |a, b|
                batch_ctx.read(b, "diagnostic").map(&.unsafe_as(UInt32)).should eq(scalar_ctx.read(a, "diagnostic").map(&.unsafe_as(UInt32)))
              end
              cursor += rows
              batch.position.should eq(cursor)
            end
            scalar_next = scalar.step(scalar_ctx.upload(inputs[8]))
            batch_next = batch.step(batch_ctx.upload(inputs[8]))
            batch_ctx.read(batch_next, "diagnostic").map(&.unsafe_as(UInt32)).should eq(scalar_ctx.read(scalar_next, "diagnostic").map(&.unsafe_as(UInt32)))
            expected_gdn = gdn_post == "1" ? 9_i64 * (config.layers - config.layers // config.full_interval) : 0_i64
            batch_ctx.gdn_post_fused_dispatches.should eq(expected_gdn)
            scalar_ctx.gdn_post_fused_dispatches.should eq(0_i64)
          end
        end
      ensure
        if previous_depth
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous_depth
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
        if previous_gdn_post
          ENV["DRAMA3_GDN_POST_FUSION"] = previous_gdn_post
        else
          ENV.delete("DRAMA3_GDN_POST_FUSION")
        end
      end
    end

    it "rejects invalid row counts, widths, capacity and foreign ownership before advancing" do
      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = ML::Audio::Drama3::BackboneConfig.new(JSON.parse(File.read(File.join(path, "config.json"))))
      ctx = ML::Audio::Drama3::Resident::Context.new
      other = ML::Audio::Drama3::Resident::Context.new
      model = ML::Audio::Drama3::BackboneResident.new(config, ML::Audio::Drama3::WeightStore.new(path), ctx, 5)
      input = ctx.zeros(16)
      expect_raises(ArgumentError) { model.prefill(input, 0) }
      expect_raises(ArgumentError) { model.prefill(input, 3) }
      expect_raises(ArgumentError) { model.prefill(input, Int32::MAX) }
      expect_raises(ArgumentError) { model.prefill(other.zeros(16), 2) }
      model.position.should eq(0)
      ctx.readbacks.should be_empty
    end
  end
{% end %}
