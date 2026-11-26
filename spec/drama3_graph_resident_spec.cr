require "spec"
require "../src/ml/audio/drama3/resident"
require "../src/ml/audio/drama3/codec_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  describe "DRAMA-3 CogniGraph ownership" do
    it "rejects foreign-context views before host reads or GPU bindings" do
      previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        producer = ML::Audio::Drama3::Resident::Context.new
        consumer = ML::Audio::Drama3::Resident::Context.new
        foreign = producer.zeros(3).slice(1, 1)
        producer.submit
        own = consumer.upload([1_f32])
        matrix = ML::Audio::Drama3::Matrix.new(ML::Audio::Drama3::Tensor.new([1, 1], [1_f32]))
        expect_raises(ArgumentError, /different resident context/) { consumer.read(foreign, "terminal") }
        expect_raises(ArgumentError, /different resident context/) { consumer.copy(foreign, own) }
        expect_raises(ArgumentError, /different resident context/) { consumer.matvec(matrix, foreign) }
        expect_raises(ArgumentError, /different resident context/) { consumer.expert(matrix, own, foreign) }
        expect_raises(ArgumentError, /different resident context/) { consumer.embedding(matrix, foreign) }
        expect_raises(ArgumentError, /different resident context/) { consumer.experts(matrix, own, foreign) }
        consumer.submissions.should eq(0)
        consumer.readbacks.should be_empty
        producer.graph_stats["pending"].should eq(1)
        producer.read(foreign, "terminal").should eq([0_f32])
        consumer.read(own, "terminal").should eq([1_f32])
      ensure
        producer.try(&.abort)
        consumer.try(&.abort)
        if previous
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "keeps shared output views alive across queued commands and GC without a host handoff" do
      previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ctx = ML::Audio::Drama3::Resident::Context.new
        output = ctx.zeros(7)
        source = ctx.upload([1_f32, 2_f32, 3_f32])
        ctx.copy(source, output.slice(2, 3))
        ctx.submit
        ctx.graph_stats["pending"].should eq(1)
        GC.collect
        # Consume the preceding allocation directly, including its offset.
        doubled = ctx.add(output.slice(2, 3), output.slice(2, 3))
        ctx.submit
        ctx.graph_stats["max_pending"].should eq(2)
        ctx.graph_stats["completion_waits"].should eq(0)
        GC.collect
        ctx.copy(doubled, output.slice(2, 3))
        ctx.submit
        ctx.graph_stats["completion_waits"].should eq(1)
        ctx.read(output, "terminal").should eq([0_f32, 0_f32, 2_f32, 4_f32, 6_f32, 0_f32, 0_f32])
        ctx.graph_stats["submitted"].should eq(3)
        ctx.graph_stats["completed"].should eq(3)
        ctx.graph_stats["pending"].should eq(0)
        ctx.graph_stats["retained_bytes"].should eq(0)
        ctx.graph_stats["max_retained_bytes"].should be > 0
        ctx.readbacks.should eq({"terminal" => 28_i64})
      ensure
        if previous
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "drains submitted work, cancels open work, and rejects reuse after abort" do
      previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ctx = ML::Audio::Drama3::Resident::Context.new
        output = ctx.zeros(3)
        ctx.submit
        ctx.copy(ctx.upload([1_f32, 2_f32, 3_f32]), output)
        ctx.abort
        ctx.graph_stats["pending"].should eq(0)
        ctx.graph_stats["completed"].should eq(1)
        ctx.graph_stats["retained_bytes"].should eq(0)
        expect_raises(ArgumentError, /failed|aborted/) { ctx.zeros(3) }
      ensure
        if previous
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "preserves sticky finite rejection across queued boundaries" do
      previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ctx = ML::Audio::Drama3::Resident::Context.new
        ctx.guard(ctx.upload([Float32::NAN]))
        ctx.submit
        finite = ctx.zeros(1)
        ctx.submit
        expect_raises(ArgumentError, /non-finite resident intermediate/) { ctx.read(finite, "terminal") }
        ctx.graph_stats["pending"].should eq(0)
      ensure
        if previous
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end

    it "drains older work when the public codec rejects an invalid input" do
      previous = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]?
      begin
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "2"
        ctx = ML::Audio::Drama3::Resident::Context.new
        ctx.zeros(1)
        ctx.submit
        store = ML::Audio::Drama3::WeightStore.new({} of String => ML::Audio::Drama3::Tensor)
        codec = ML::Audio::Drama3::CodecResident.new(store, ctx)
        expect_raises(ArgumentError, /token buffer shape mismatch/) do
          codec.decode(ctx.upload([0_f32]), 1)
        end
        ctx.graph_stats["pending"].should eq(0)
        ctx.graph_stats["completed"].should eq(1)
        ctx.graph_stats["retained_bytes"].should eq(0)
        expect_raises(ArgumentError, /aborted/) { ctx.zeros(1) }
      ensure
        if previous
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = previous
        else
          ENV.delete("DRAMA3_COGNIGRAPH_MAX_INFLIGHT")
        end
      end
    end
  end
{% end %}
