require "./backbone"
require "./tower"
require "./prompt"

module ML::Audio::Drama3
  # Upstream sampling keeps the first rank, but drops the candidate that
  # crosses top_p. Its cumulative probabilities use the full vocabulary.
  class Sampler
    def initialize(seed : UInt64)
      @random = Random.new(seed)
    end

    def candidates(logits : Vector, temperature : Float64, top_k : Int32, top_p : Float64) : Array(Tuple(Int32, Float64))
      raise ArgumentError.new("invalid sampling parameters") unless temperature.finite? && temperature >= 0 && top_k >= 1 && top_p.finite? && top_p > 0 && top_p <= 1
      raise ArgumentError.new("empty logits") if logits.empty?
      Ops.finite!(logits, "sampling logits")
      scaled = logits.map { |v| v.to_f64 / Math.max(temperature, 1e-5) }
      maximum = scaled.max
      probabilities = scaled.map { |v| Math.exp(v - maximum) }
      denominator = probabilities.sum
      ids = (0...logits.size).to_a.sort_by { |i| {-scaled[i], i} }.first(Math.min(top_k, logits.size))
      cumulative = 0.0
      kept = [] of Tuple(Int32, Float64)
      ids.each_with_index do |id, rank|
        cumulative += probabilities[id] / denominator
        kept << {id, probabilities[id]} if rank == 0 || cumulative <= top_p
      end
      kept
    end

    def sample(logits : Vector, temperature : Float64, top_k : Int32, top_p : Float64) : Int32
      Profile.measure("sampler.sample", category: "sampling") { sample_impl(logits, temperature, top_k, top_p) }
    end

    private def sample_impl(logits : Vector, temperature : Float64, top_k : Int32, top_p : Float64) : Int32
      kept = candidates(logits, temperature, top_k, top_p)
      draw = @random.rand * kept.sum(&.[1])
      kept.each do |id, weight|
        draw -= weight
        return id if draw < 0
      end
      kept.last[0]
    end
  end

  struct Generation
    getter frames : Array(Array(Int32))
    getter stopped : Bool
    getter prompt_rows : Int32

    def initialize(@frames : Array(Array(Int32)), @stopped : Bool, @prompt_rows : Int32)
    end

    def codebooks : Array(Array(Int32))
      return [] of Array(Int32) if @frames.empty?
      Array(Array(Int32)).new(@frames[0].size - 1) { |i| @frames.map { |f| f[i + 1] } }
    end
  end

  class Generator
    getter backbone : Backbone
    getter tower : Tower
    property progress : Proc(String, Nil)?
    @eos : Int32
    @semantic_begin : Int32
    @semantic_end : Int32
    @head : Matrix

    def initialize(config : JSON::Any, @store : WeightStore)
      @backbone, @tower, @eos, @semantic_begin, @semantic_end, @head = Profile.measure("generator.initialize", category: "loading") do
        backbone = Backbone.new(BackboneConfig.new(config), @store)
        tower = Tower.new(TowerConfig.from_json(config.to_json), @store)
        eos = config["eos_token_id"].as_i
        semantic_begin = config["semantic_start_token_id"].as_i
        semantic_end = config["semantic_end_token_id"].as_i
        raise ArgumentError.new("invalid candidate token band") unless eos < semantic_begin && semantic_end - semantic_begin + 1 == tower.config.vocab_size
        head = @store.matrix("lm_head.weight")
        raise ArgumentError.new("invalid main head dimensions") unless head.cols == backbone.config.hidden && head.rows == 1 + semantic_end - semantic_begin + 1
        {backbone, tower, eos, semantic_begin, semantic_end, head}
      end
      @progress = nil
    end

    # This instance has stateful caches and is deliberately single-use.
    def generate(prompt : Array(Array(Int32)), max_frames : Int32 = 160, seed : UInt64 = 0_u64,
                 temperature : Float64 = 0.8, top_k : Int32 = 30, top_p : Float64 = 0.9) : Generation
      Profile.measure("generator.generate", category: "generation") do
        generate_impl(prompt, max_frames, seed, temperature, top_k, top_p)
      end
    end

    private def generate_impl(prompt : Array(Array(Int32)), max_frames : Int32, seed : UInt64,
                              temperature : Float64, top_k : Int32, top_p : Float64) : Generation
      raise ArgumentError.new("generator is single-use") unless @backbone.position == 0
      raise ArgumentError.new("empty prompt or non-positive frame bound") if prompt.empty? || max_frames < 1
      raise ArgumentError.new("invalid prompt width") unless prompt.all? { |row| row.size == @tower.config.num_codebooks + 1 }
      sampler = Sampler.new(seed)
      # Validate before running any model work.
      sampler.candidates([0_f32], temperature, top_k, top_p)
      caches = @tower.make_cache
      hidden = [] of Float32
      Profile.measure("generator.prefill", category: "generation.prefill") do
        prompt.each_with_index do |frame, index|
          embedding = @tower.embed_backbone([frame], [@backbone.embedding(frame[0])])[0]
          hidden = @backbone.step(embedding)
          @tower.teacher_force(hidden, frame, prompt[index + 1], caches) if index + 1 < prompt.size
          @progress.try(&.call("prefill #{index + 1}/#{prompt.size}"))
        end
      end
      previous = prompt.last
      window = prompt.last(16).map(&.[0])
      frames = [] of Array(Int32)
      stopped = false
      Profile.measure("generator.autoregressive", category: "generation.decode") do
        max_frames.times do |step|
          logits = @head.matvec(hidden)
          first = candidate_to_token(sampler.sample(logits, temperature, top_k, top_p))
          retry = candidate_to_token(sampler.sample(logits, 1.0, 30, 0.95))
          main = first.in?(@semantic_begin..@semantic_end) && window.includes?(first) ? retry : first
          if main == @eos
            stopped = true
            break
          end
          sampled = @tower.sample_frame(main, hidden, previous, caches) { |head| sampler.sample(head, temperature, top_k, top_p) }
          frame = [main] + sampled.codes
          frames << frame
          @progress.try(&.call("frame #{frames.size}/#{max_frames}"))
          # Upstream preserves the initial window width even for raw prompts
          # shorter than 16 rows (the formatted text prompt is longer).
          window.shift
          window << main
          previous = frame
          if step + 1 < max_frames
            embedding = @tower.embed_backbone([frame], [@backbone.embedding(main)])[0]
            hidden = @backbone.step(embedding)
          end
        end
      end
      Generation.new(frames, stopped, prompt.size)
    end

    private def candidate_to_token(index : Int32) : Int32
      index == 0 ? @eos : @semantic_begin + index - 1
    end
  end
end
