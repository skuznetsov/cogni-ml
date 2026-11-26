require "spec"
require "../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private class Drama3PrefillGeneratorZeroHeadStore < ML::Audio::Drama3::WeightStore
    def tensor(name : String) : ML::Audio::Drama3::Tensor
      return ML::Audio::Drama3::Tensor.new([5, 8], Array(Float32).new(40, 0_f32)) if name == "lm_head.weight"
      super
    end
  end

  # Test-only snapshots compact the persistent KV state after generation's
  # existing terminal read. They do not add reads to the inference path.
  class ML::Audio::Drama3::BackboneResident
    def prefill_generator_regression_state : Array(Resident::Buffer)
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

  class ML::Audio::Drama3::TowerResident
    def prefill_generator_regression_state : Array(Resident::Buffer)
      result = [] of Resident::Buffer
      return result if @offset == 0
      @caches.each do |cache|
        count = @config.n_local_heads * @offset * @config.head_dim
        [cache.keys, cache.values].each do |source|
          compact = @context.alloc(count)
          @context.dispatch(
            "drama3_tower_compact_cache", RESIDENT_TOWER_SOURCE, [source, compact],
            [@config.n_local_heads.to_u64, @config.head_dim.to_u64, @capacity.to_u64, @offset.to_u64], count,
          )
          result << compact
        end
      end
      result
    end
  end

  private struct Drama3PrefillGeneratorObservation
    getter frames : Array(Array(Int32))
    getter stopped : Bool
    getter backbone_position : Int32
    getter tower_offset : Int32
    getter readbacks : Hash(String, Int64)
    getter graph_stats : Hash(String, Int64)
    getter hidden_bits : Array(UInt32)?
    getter logits_bits : Array(UInt32)?
    getter backbone_state : Array(Array(UInt32))
    getter tower_state : Array(Array(UInt32))
    getter trace_names : Array(String)
    getter capacity : Int32
    getter growth_count : Int32
    getter sampler_sha256 : String

    def initialize(@frames : Array(Array(Int32)), @stopped : Bool, @backbone_position : Int32,
                   @tower_offset : Int32, @readbacks : Hash(String, Int64), @graph_stats : Hash(String, Int64),
                   @hidden_bits : Array(UInt32)?, @logits_bits : Array(UInt32)?,
                   @backbone_state : Array(Array(UInt32)), @tower_state : Array(Array(UInt32)),
                   @trace_names : Array(String), @capacity : Int32, @growth_count : Int32,
                   @sampler_sha256 : String)
    end
  end

  private def drama3_prefill_generator_prompt(rows : Int32) : Array(Array(Int32))
    all_rows = [
      [1, 0, 0, 0, 0],
      [20, 1, 2, 3, 0],
      [5, 3, 1, 0, 2],
      [23, 2, 3, 1, 0],
      [1, 1, 2, 0, 3],
      [5, 0, 3, 2, 1],
      [20, 3, 1, 0, 2],
      [23, 0, 2, 1, 3],
    ]
    all_rows[0, rows]
  end

  private def drama3_prefill_generator_non_eos_seed(frames : Int32 = 2) : UInt64
    seed = 0_u64
    loop do
      random = Random.new(seed)
      draws = Array(Float32).new(frames * 5) { random.rand.to_f32 }
      candidates = Array(Int32).new(frames) { |index| (draws[index * 5] * 5_f32).to_i32 }
      # Candidate indices 2 and 3 map to semantic tokens 21 and 22. Those
      # tokens are absent from the mixed prompt. Alternation also keeps a
      # one-row repeat window from invoking the retry sampler in growth tests.
      return seed if candidates.all?(&.in?(2..3)) && candidates.each_cons_pair.all? { |left, right| left != right }
      seed += 1
    end
  end

  private def drama3_prefill_generator_run(chunk : Int32, depth : Int32, scratch : Bool,
                                           eos_first : Bool, prompt_rows : Int32 = 7,
                                           max_frames : Int32 = 2,
                                           resident_prompt : Bool = false, initial_capacity : Int32 = 8,
                                           maximum_capacity : Int32 = 8) : Drama3PrefillGeneratorObservation
    env_keys = [
      "DRAMA3_BACKEND", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_GENERATION_SCRATCH",
      "DRAMA3_PREFILL_CHUNK", "DRAMA3_DELTA_TRANSPOSE", "DRAMA3_MOE_BATCH",
      "DRAMA3_BACKBONE_DENSE_REDUCTION", "DRAMA3_EXPERT_DOWN_REDUCTION",
      "DRAMA3_TOWER_EMBEDDING_FUSION",
    ]
    previous = {} of String => String?
    env_keys.each { |key| previous[key] = ENV[key]? }
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
      ENV["DRAMA3_GENERATION_SCRATCH"] = scratch ? "1" : "0"
      ENV["DRAMA3_PREFILL_CHUNK"] = chunk.to_s
      ENV["DRAMA3_DELTA_TRANSPOSE"] = "0"
      ENV["DRAMA3_MOE_BATCH"] = "0"
      ENV["DRAMA3_BACKBONE_DENSE_REDUCTION"] = "tree"
      ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "native"
      ENV["DRAMA3_TOWER_EMBEDDING_FUSION"] = "0"

      pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?

      path = File.join(__DIR__, "support/drama3/reference_fixture")
      config = JSON.parse(File.read(File.join(path, "config.json")))
      config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
      context = ML::Audio::Drama3::Resident::Context.new
      generator = ML::Audio::Drama3::ResidentGenerator.new(config, Drama3PrefillGeneratorZeroHeadStore.new(path), context,
        initial_capacity, maximum_capacity: maximum_capacity)
      captured = {} of String => ML::Audio::Drama3::Resident::Buffer
      capture = !scratch && !eos_first
      if capture
        generator.device_trace = ->(name : String, buffer : ML::Audio::Drama3::Resident::Buffer) do
          captured[name] = buffer
          nil
        end
      end

      prompt = drama3_prefill_generator_prompt(prompt_rows)
      seed = eos_first ? 0_u64 : drama3_prefill_generator_non_eos_seed(max_frames)
      top_k = eos_first ? 1 : 5
      top_p = eos_first ? 0.9 : 1.0
      result = if resident_prompt
                 flat_frames = prompt.flat_map(&.map(&.to_f32))
                 rows = context.upload(flat_frames)
                 resident = ML::Audio::Drama3::ResidentPrompt.new(
                   rows, prompt_rows, 5, 1, 1, 4, 20, 4, context,
                 )
                 generator.generate(resident, max_frames, seed, top_k: top_k, top_p: top_p)
               else
                 generator.generate(prompt, max_frames, seed, top_k: top_k, top_p: top_p)
               end

      # Record inference-only counters before materializing callback buffers or
      # test-only state snapshots. The latter deliberately use diagnostic reads.
      inference_readbacks = context.readbacks.dup
      inference_graph = context.graph_stats.dup
      hidden_bits = if capture
                      captured.keys.should contain("prompt.backbone.hidden")
                      context.read(captured["prompt.backbone.hidden"], "diagnostic").map(&.unsafe_as(UInt32))
                    end
      logits_bits = if capture
                      captured.keys.should contain("prompt.main_logits")
                      context.read(captured["prompt.main_logits"], "diagnostic").map(&.unsafe_as(UInt32))
                    end
      backbone_state = generator.backbone.prefill_generator_regression_state.map do |buffer|
        context.read(buffer, "diagnostic").map(&.unsafe_as(UInt32))
      end
      tower_state = generator.tower.prefill_generator_regression_state.map do |buffer|
        context.read(buffer, "diagnostic").map(&.unsafe_as(UInt32))
      end

      Drama3PrefillGeneratorObservation.new(
        result.generation.frames, result.generation.stopped, generator.backbone.position, generator.tower.offset,
        inference_readbacks, inference_graph, hidden_bits, logits_bits, backbone_state, tower_state, captured.keys.sort,
        generator.capacity, generator.window_growths.size, generator.sampler_inputs.not_nil![:sha256],
      )
    ensure
      env_keys.each do |key|
        if value = previous[key]?
          ENV[key] = value
        else
          ENV.delete(key)
        end
      end
    end
  end

  private def drama3_prefill_generator_assert_equal(expected : Drama3PrefillGeneratorObservation,
                                                    actual : Drama3PrefillGeneratorObservation,
                                                    compare_prompt_traces : Bool) : Nil
    actual.frames.should eq(expected.frames)
    actual.stopped.should eq(expected.stopped)
    actual.backbone_position.should eq(expected.backbone_position)
    actual.tower_offset.should eq(expected.tower_offset)
    actual.readbacks.should eq(expected.readbacks)
    actual.backbone_state.should eq(expected.backbone_state)
    actual.tower_state.should eq(expected.tower_state)
    actual.graph_stats["submitted"].should be <= expected.graph_stats["submitted"]
    actual.graph_stats["completion_waits"].should be <= expected.graph_stats["completion_waits"]
    actual.graph_stats["pending"].should eq(0_i64)
    actual.graph_stats["completed"].should eq(actual.graph_stats["submitted"])
    actual.readbacks.keys.sort.should eq(expected.readbacks.keys.sort)
    (actual.readbacks.keys - ["control", "terminal"]).should be_empty

    if compare_prompt_traces
      actual.hidden_bits.should eq(expected.hidden_bits)
      actual.logits_bits.should eq(expected.logits_bits)
      actual.trace_names.should eq(["prompt.backbone.hidden", "prompt.main_logits"])
    else
      actual.hidden_bits.should be_nil
      actual.logits_bits.should be_nil
      actual.trace_names.should be_empty
    end
  end

  describe "DRAMA-3 generator sequence prefill" do
    it "grows only after non-EOS without changing state, draws or synchronization" do
      [0, 2].each do |depth|
        [false, true].each do |scratch|
          [false, true].each do |resident|
            rows = resident ? 7 : 1
            frames = resident ? 2 : 4
            capacity = rows + frames - 1
            fixed = drama3_prefill_generator_run(1, depth, scratch, false, rows, frames, resident, capacity, capacity)
            grown = drama3_prefill_generator_run(1, depth, scratch, false, rows, frames, resident, rows, capacity)
            grown.frames.size.should eq(frames)
            grown.capacity.should eq(capacity)
            grown.growth_count.should eq(resident ? 1 : 2)
            grown.sampler_sha256.should eq(fixed.sampler_sha256)
            drama3_prefill_generator_assert_equal(fixed, grown, !scratch)
          end
        end
      end
    end

    it "does not grow when EOS precedes the first capacity boundary" do
      [0, 2].each do |depth|
        fixed = drama3_prefill_generator_run(1, depth, false, true, 1, 4, false, 4, 4)
        grown = drama3_prefill_generator_run(1, depth, false, true, 1, 4, false, 1, 4)
        grown.capacity.should eq(1)
        grown.growth_count.should eq(0)
        drama3_prefill_generator_assert_equal(fixed, grown, false)
      end
    end

    it "rejects growth before allocation and aborts instead of resuming partial state" do
      keys = ["DRAMA3_BACKEND", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_GENERATION_SCRATCH", "DRAMA3_PREFILL_CHUNK"]
      previous = keys.to_h { |key| {key, ENV[key]?} }
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        ENV["DRAMA3_GENERATION_SCRATCH"] = "1"
        ENV["DRAMA3_PREFILL_CHUNK"] = "1"
        [0, 2].each do |depth|
          ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
          [false, true].each do |fail_after_copy|
            path = File.join(__DIR__, "support/drama3/reference_fixture")
            config = JSON.parse(File.read(File.join(path, "config.json")))
            config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
            context = ML::Audio::Drama3::Resident::Context.new
            generator = ML::Audio::Drama3::ResidentGenerator.new(config, Drama3PrefillGeneratorZeroHeadStore.new(path),
              context, 1, maximum_capacity: 4)
            before = 0_i64
            reject = ->(_from : Int32, _to : Int32) do
              before = context.allocation_stats[:allocations]
              raise "injected window failure"
              nil
            end
            if fail_after_copy
              generator.after_window_growth = reject
            else
              generator.before_window_growth = reject
            end
            expect_raises(Exception, /injected window failure/) do
              generator.generate(drama3_prefill_generator_prompt(1), 4, drama3_prefill_generator_non_eos_seed(4),
                top_k: 5, top_p: 1.0)
            end
            context.allocation_stats[:allocations].should eq(before)
            generator.capacity.should eq(fail_after_copy ? 2 : 1)
            generator.window_growths.size.should eq(fail_after_copy ? 1 : 0)
            context.graph_stats["pending"].should eq(0)
            expect_raises(ArgumentError, /aborted/) { context.alloc(1) }
          end
        end
      ensure
        previous.each do |key, value|
          if value
            ENV[key] = value
          else
            ENV.delete(key)
          end
        end
      end
    end

    it "matches scalar generation, prompt activations, persistent state and read boundaries at graph depths zero and two" do
      [false, true].each do |eos_first|
        [0, 2].each do |depth|
          scalar = drama3_prefill_generator_run(1, depth, false, eos_first)
          if eos_first
            scalar.frames.should be_empty
            scalar.stopped.should be_true
            scalar.backbone_position.should eq(7)
            scalar.tower_offset.should eq(6)
          else
            scalar.frames.size.should eq(2)
            scalar.frames.each do |frame|
              frame.size.should eq(5)
              (20..23).includes?(frame[0]).should be_true
            end
            scalar.stopped.should be_false
            scalar.backbone_position.should eq(8)
            scalar.tower_offset.should eq(8)
          end

          [false, true].each do |scratch|
            [2, 5].each do |chunk|
              batched = drama3_prefill_generator_run(chunk, depth, scratch, eos_first)
              drama3_prefill_generator_assert_equal(scalar, batched, !scratch && !eos_first)
            end
          end
        end
      end
    end

    it "keeps a full-sized final chunk and borrowed hidden alive through scratch-backed decoding" do
      scalar = drama3_prefill_generator_run(1, 2, true, false, 8, 1)
      scalar.frames.size.should eq(1)
      scalar.backbone_position.should eq(8)
      scalar.tower_offset.should eq(8)
      [2].each do |chunk|
        batched = drama3_prefill_generator_run(chunk, 2, true, false, 8, 1)
        drama3_prefill_generator_assert_equal(scalar, batched, false)
      end
    end

    it "matches scalar generation for chunked resident prompts without diagnostic readbacks" do
      scalar = drama3_prefill_generator_run(1, 2, true, false, 7, 2, true)
      scalar.frames.size.should eq(2)
      scalar.readbacks.keys.sort.should eq(["control", "terminal"])

      [2, 5].each do |chunk|
        batched = drama3_prefill_generator_run(chunk, 2, true, false, 7, 2, true)
        drama3_prefill_generator_assert_equal(scalar, batched, false)
        batched.readbacks.keys.sort.should eq(["control", "terminal"])
        # The last prompt row has no teacher forcing. When its chunk has
        # multiple rows, its backbone was already submitted with the first
        # row; the final submit therefore has no encoder and is elided.
        final_chunk_rows = (7 - 1) % chunk + 1
        fewer = final_chunk_rows > 1 ? 1_i64 : 0_i64
        batched.graph_stats["submitted"].should eq(scalar.graph_stats["submitted"] - fewer)
        batched.graph_stats["completion_waits"].should eq(scalar.graph_stats["completion_waits"] - fewer)
      end
    end

    it "rejects malformed chunk bounds before submitting model work" do
      env_keys = [
        "DRAMA3_BACKEND", "DRAMA3_COGNIGRAPH_MAX_INFLIGHT", "DRAMA3_GENERATION_SCRATCH",
        "DRAMA3_PREFILL_CHUNK", "DRAMA3_DELTA_TRANSPOSE", "DRAMA3_MOE_BATCH",
        "DRAMA3_BACKBONE_DENSE_REDUCTION", "DRAMA3_EXPERT_DOWN_REDUCTION",
        "DRAMA3_TOWER_EMBEDDING_FUSION",
      ]
      previous = {} of String => String?
      env_keys.each { |key| previous[key] = ENV[key]? }
      begin
        ENV["DRAMA3_BACKEND"] = "metal"
        pending!("Metal is unavailable or lacks unified memory") unless ML::Metal::Device.instance.available? && ML::Metal::Device.instance.has_unified_memory?
        ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = "0"
        ENV["DRAMA3_GENERATION_SCRATCH"] = "0"
        ENV["DRAMA3_DELTA_TRANSPOSE"] = "0"
        ENV["DRAMA3_MOE_BATCH"] = "0"
        ENV["DRAMA3_BACKBONE_DENSE_REDUCTION"] = "tree"
        ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "native"
        ENV["DRAMA3_TOWER_EMBEDDING_FUSION"] = "0"
        path = File.join(__DIR__, "support/drama3/reference_fixture")
        config = JSON.parse(File.read(File.join(path, "config.json")))
        # Keep the fixture's semantic band valid so each case reaches the
        # chunk-range guard under test.
        config.as_h["eos_token_id"] = JSON::Any.new(2_i64)
        ["0", "33", "not-an-integer"].each do |value|
          ENV["DRAMA3_PREFILL_CHUNK"] = value
          context = ML::Audio::Drama3::Resident::Context.new
          expect_raises(ArgumentError, /DRAMA3_PREFILL_CHUNK must be 1\.\.32/) do
            ML::Audio::Drama3::ResidentGenerator.new(config, Drama3PrefillGeneratorZeroHeadStore.new(path), context, 8)
          end
          context.submissions.should eq(0_i64)
          context.readbacks.should be_empty
          context.abort
        end
      ensure
        env_keys.each do |key|
          if value = previous[key]?
            ENV[key] = value
          else
            ENV.delete(key)
          end
        end
      end
    end
  end
{% end %}
