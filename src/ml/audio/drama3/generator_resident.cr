require "json"
require "./generator"
require "./resident"
require "./backbone_resident"
require "./tower_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3
  RESIDENT_GENERATOR_SOURCE = {{ read_file("#{__DIR__}/kernels/generator_resident.metal") }}

  # CPU-visible token results plus the original resident frame allocation. The
  # valid device prefix has generation.frames.size rows of [main,c0,...,cN-1].
  struct ResidentGeneration
    getter generation : Generation
    getter deviceframes : Resident::Buffer

    def initialize(@generation : Generation, @deviceframes : Resident::Buffer)
    end
  end

  # End-to-end resident DRAMA-3 generation. Activations, logits, selected code
  # ids, and generated frames stay in Metal buffers. The only decode-time
  # token read is the chosen main token (4 bytes) per attempted frame, which
  # lets EOS stop expensive tower work. Context also reads its 4-byte finite
  # status at each read boundary (8 bytes of control per attempt). Complete
  # frames are read once at the terminal
  # boundary to populate the existing Generation return value.
  class ResidentGenerator
    MAX_REPEAT_WINDOW = 16

    getter backbone : BackboneResident
    getter tower : TowerResident
    getter ctx : Resident::Context
    getter capacity : Int32
    property progress : Proc(String, Nil)?

    @head : Matrix
    @eos : Int32
    @semantic_begin : Int32
    @semantic_end : Int32

    def initialize(config : JSON::Any, @store : WeightStore, @ctx : Resident::Context, @capacity : Int32)
      backbone_config = BackboneConfig.new(config)
      tower_config = TowerConfig.from_json(config.to_json)
      @eos = config["eos_token_id"].as_i
      @semantic_begin = config["semantic_start_token_id"].as_i
      @semantic_end = config["semantic_end_token_id"].as_i
      raise ArgumentError.new("invalid candidate token band") unless @eos < @semantic_begin && @semantic_end - @semantic_begin + 1 == tower_config.vocab_size

      @backbone = BackboneResident.new(backbone_config, @store, @ctx, @capacity)
      @tower = TowerResident.new(tower_config, @store, @ctx, @capacity)
      @head = @store.matrix("lm_head.weight")
      raise ArgumentError.new("invalid main head dimensions") unless @head.cols == backbone_config.hidden && @head.rows == 1 + @semantic_end - @semantic_begin + 1
      @progress = nil
    end

    # A resident model owns temporal caches and is deliberately single-use.
    # The output buffer has max_frames rows; only generation.frames.size rows
    # are valid, including the empty-prefix case when EOS is selected first.
    def generate(prompt : Array(Array(Int32)), max_frames : Int32 = 160, seed : UInt64 = 0_u64,
                 temperature : Float64 = 0.8, top_k : Int32 = 30, top_p : Float64 = 0.9) : ResidentGeneration
      Profile.measure("generator_resident.generate", category: "generation") do
        generate_impl(prompt, max_frames, seed, temperature, top_k, top_p)
      end
    end

    private def generate_impl(prompt : Array(Array(Int32)), max_frames : Int32, seed : UInt64,
                              temperature : Float64, top_k : Int32, top_p : Float64) : ResidentGeneration
      raise ArgumentError.new("generator is single-use") unless @backbone.position == 0 && @tower.offset == 0
      raise ArgumentError.new("empty prompt or non-positive frame bound") if prompt.empty? || max_frames < 1
      raise ArgumentError.new("invalid prompt width") unless prompt.all? { |row| row.size == @tower.config.num_codebooks + 1 }
      raise ArgumentError.new("sampling top_k exceeds resident sampler limit 1024") unless top_k.in?(1..1024)
      raise ArgumentError.new("resident cache capacity exceeded") unless prompt.size.to_i64 + max_frames - 1 <= @capacity

      # Preserve the CPU path's parameter validation before any model work.
      Sampler.new(seed).candidates([0_f32], temperature, top_k, top_p)
      frame_width = @tower.config.num_codebooks + 1
      device_frames = @ctx.alloc(max_frames * frame_width)
      draw_stride = 2 + @tower.config.num_codebooks - 1
      draws = sampling_draws(seed, max_frames, draw_stride)
      draw_buffer = @ctx.upload(draws)

      frames_on_device = prompt.map { |frame| @tower.upload_frame(frame) }
      hidden = @ctx.alloc(@backbone.config.hidden)
      Profile.measure("generator_resident.prefill", category: "generation.prefill") do
        frames_on_device.each_with_index do |frame, index|
          Profile.measure("generator_resident.prefill_row", category: "generation.prefill.row") do
            token_embedding = @backbone.embedding(frame.slice(0, 1))
            hidden = @backbone.step(@tower.embed_backbone(frame, token_embedding))
            if index + 1 < frames_on_device.size
              @tower.teacher_force(hidden, frame, frames_on_device[index + 1])
            end
            # Bound live temporary allocations while persistent KV/recurrent
            # state remains owned by the resident model objects.
            @ctx.finish
          end
          @progress.try(&.call("prefill #{index + 1}/#{frames_on_device.size}"))
        end
      end

      window_values = prompt.last(MAX_REPEAT_WINDOW).map { |row| row[0].to_f32 }
      window = @ctx.upload(window_values)
      window_size = window_values.size
      window_cursor = 0
      previous_frame = frames_on_device.last
      generated_count = 0
      stopped = false

      Profile.measure("generator_resident.autoregressive", category: "generation.decode") do
        max_frames.times do |step|
          logits = @ctx.matvec(@head, hidden)
          draw_offset = step * draw_stride
          first = @ctx.sample(logits, draw_buffer.slice(draw_offset, 1), temperature, top_k, top_p)
          retry = @ctx.sample(logits, draw_buffer.slice(draw_offset + 1, 1), 1.0, 30, 0.95)
          main = choose_main(first, retry, window, window_size, window_cursor)

          # This scalar is the explicit EOS control boundary. `main` remains a
          # device buffer and is passed directly into tower frame generation.
          main_token = @ctx.read(main, "control")[0].to_i32
          unless main_token == @eos || main_token.in?(@semantic_begin..@semantic_end)
            raise "resident main-token selector produced out-of-band token #{main_token}"
          end
          if main_token == @eos
            stopped = true
            break
          end

          frame = Profile.measure("generator_resident.device_frame", category: "generation.frame") do
            sampled_codes = 0
            resident_frame = @tower.sample_frame(main, hidden, previous_frame) do |code_logits|
              draw_index = draw_offset + 2 + sampled_codes
              sampled_codes += 1
              @ctx.sample(code_logits, draw_buffer.slice(draw_index, 1), temperature, top_k, top_p)
            end
            raise "resident tower sampled #{sampled_codes} residual codes; expected #{@tower.config.num_codebooks - 1}" unless sampled_codes == @tower.config.num_codebooks - 1

            @ctx.copy(resident_frame, device_frames.slice(step * frame_width, frame_width))
            if step + 1 < max_frames
              token_embedding = @backbone.embedding(main)
              hidden = @backbone.step(@tower.embed_backbone(resident_frame, token_embedding))
            end
            # Submit the complete tower/backbone chain for this generated row.
            # This keeps scratch bounded and makes the enclosing profile span
            # include actual GPU execution time.
            @ctx.finish
            resident_frame
          end

          generated_count += 1
          @progress.try(&.call("frame #{generated_count}/#{max_frames}"))
          window_cursor = (window_cursor + 1) % window_size
          previous_frame = frame
        end
      end

      terminal_values = if generated_count == 0
                          [] of Float32
                        else
                          @ctx.read(device_frames.slice(0, generated_count * frame_width), "terminal")
                        end
      terminal_frames = decode_terminal_frames(terminal_values, generated_count, frame_width)
      generation = Generation.new(terminal_frames, stopped, prompt.size)
      ResidentGeneration.new(generation, device_frames)
    end

    private def choose_main(first : Resident::Buffer, retry : Resident::Buffer, window : Resident::Buffer,
                            window_size : Int32, window_cursor : Int32) : Resident::Buffer
      selected = @ctx.alloc(1)
      @ctx.dispatch(
        "dr3_choose_main",
        RESIDENT_GENERATOR_SOURCE,
        [first, retry, window, selected],
        [@eos.to_i64.to_u64, @semantic_begin.to_i64.to_u64, @semantic_end.to_i64.to_u64,
         window_size.to_u64, window_cursor.to_u64, @tower.config.vocab_size.to_u64],
        1,
      )
      selected
    end

    private def sampling_draws(seed : UInt64, max_frames : Int32, draw_stride : Int32) : Array(Float32)
      random = Random.new(seed)
      Array(Float32).new(max_frames * draw_stride) { random.rand.to_f32 }
    end

    private def decode_terminal_frames(values : Vector, frame_count : Int32, frame_width : Int32) : Array(Array(Int32))
      raise "resident terminal frame prefix has wrong size" unless values.size == frame_count * frame_width
      Array(Array(Int32)).new(frame_count) do |frame_index|
        frame = Array(Int32).new(frame_width) do |codebook|
          value = values[frame_index * frame_width + codebook]
          raise "non-integral resident token output" unless value % 1_f32 == 0_f32
          token = value.to_i32
          if codebook == 0
            raise "resident generated main token is out of band" unless token.in?(@semantic_begin..@semantic_end)
          else
            raise "resident generated code is out of range" unless token.in?(0...@tower.config.vocab_size)
          end
          token
        end
        frame
      end
    end
  end
end
{% end %}
