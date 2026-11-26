require "json"
require "digest/sha256"
require "./generator"
require "./resident"
require "./backbone_prefill"
require "./tower_resident"
require "./prompt_resident"

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

    record WindowExpansion, from : Int32, to : Int32, backbone_position : Int32, tower_offset : Int32 do
      include JSON::Serializable
    end

    getter backbone : BackboneResident
    getter tower : TowerResident
    getter ctx : Resident::Context
    getter capacity : Int32
    getter maximum_capacity : Int32
    getter window_growths = [] of WindowExpansion
    getter prefill_chunk : Int32
    getter sampler_inputs : NamedTuple(dtype: String, uniform_count: Int32, sha256: String, provenance: String)?
    property progress : Proc(String, Nil)?
    # Host-only admission/telemetry at an existing drained EOS boundary.
    # A rejection happens before any replacement allocation or cache copy.
    property before_window_growth : Proc(Int32, Int32, Nil)?
    property after_window_growth : Proc(Int32, Int32, Nil)?
    # Optional device-only diagnostic handoff. Callbacks enqueue comparisons;
    # normal generation has no additional work or readbacks when nil.
    property device_trace : Proc(String, Resident::Buffer, Nil)?

    @head : Matrix
    @eos : Int32
    @semantic_begin : Int32
    @semantic_end : Int32
    @generation_scratch : Bool
    @workspaces = {} of String => Resident::Workspace

    def initialize(config : JSON::Any, @store : WeightStore, @ctx : Resident::Context, capacity : Int32,
                   maximum_capacity : Int32? = nil)
      @capacity = capacity
      @maximum_capacity = maximum_capacity || capacity
      unless @capacity.in?(1..4096) && @maximum_capacity.in?(@capacity..4096)
        raise ArgumentError.new("invalid resident initial/maximum cache capacity")
      end
      backbone_config = BackboneConfig.new(config)
      tower_config = TowerConfig.from_json(config.to_json)
      @eos = config["eos_token_id"].as_i
      @semantic_begin = config["semantic_start_token_id"].as_i
      @semantic_end = config["semantic_end_token_id"].as_i
      raise ArgumentError.new("invalid candidate token band") unless @eos < @semantic_begin && @semantic_end - @semantic_begin + 1 == tower_config.vocab_size

      @backbone = BackboneResident.new(backbone_config, @store, @ctx, @capacity, maximum_capacity: @maximum_capacity)
      @tower = TowerResident.new(tower_config, @store, @ctx, @capacity, maximum_capacity: @maximum_capacity)
      @head = @store.matrix("lm_head.weight")
      raise ArgumentError.new("invalid main head dimensions") unless @head.cols == backbone_config.hidden && @head.rows == 1 + @semantic_end - @semantic_begin + 1
      @progress = nil
      @device_trace = nil
      @sampler_inputs = nil
      chunk = ENV["DRAMA3_PREFILL_CHUNK"]? || "1"
      @prefill_chunk = chunk.to_i?
        .try { |value| value.in?(1..32) ? value : nil } || raise ArgumentError.new("DRAMA3_PREFILL_CHUNK must be 1..32")
      @generation_scratch = case ENV["DRAMA3_GENERATION_SCRATCH"]? || "0"
                            when "0" then false
                            when "1" then true
                            else raise ArgumentError.new("DRAMA3_GENERATION_SCRATCH must be 0 or 1")
                            end
    end

    # A resident model owns temporal caches and is deliberately single-use.
    # The output buffer has max_frames rows; only generation.frames.size rows
    # are valid, including the empty-prefix case when EOS is selected first.
    def generate(prompt : Array(Array(Int32)), max_frames : Int32 = 160, seed : UInt64 = 0_u64,
                 temperature : Float64 = 0.8, top_k : Int32 = 30, top_p : Float64 = 0.9) : ResidentGeneration
      Profile.measure("generator_resident.generate", category: "generation") do
        result = generate_impl(prompt, max_frames, seed, temperature, top_k, top_p)
        release_workspaces
        result
      end
    rescue ex
      begin
        @ctx.abort
      rescue
        # Keep the inference failure as the reported error if draining also fails.
      end
      raise ex
    end

    # GPU-resident prompt path used when reference encoder codes are already in
    # this context. The flat prompt rows flow directly into backbone prefill.
    def generate(prompt : ResidentPrompt, max_frames : Int32 = 160, seed : UInt64 = 0_u64,
                 temperature : Float64 = 0.8, top_k : Int32 = 30, top_p : Float64 = 0.9) : ResidentGeneration
      Profile.measure("generator_resident.generate", category: "generation") do
        result = generate_resident_impl(prompt, max_frames, seed, temperature, top_k, top_p)
        release_workspaces
        result
      end
    rescue ex
      begin
        @ctx.abort
      rescue
        # Keep the inference failure as the reported error if draining also fails.
      end
      raise ex
    end

    private def generate_impl(prompt : Array(Array(Int32)), max_frames : Int32, seed : UInt64,
                              temperature : Float64, top_k : Int32, top_p : Float64) : ResidentGeneration
      raise ArgumentError.new("generator is single-use") unless @backbone.position == 0 && @tower.offset == 0
      raise ArgumentError.new("empty prompt or non-positive frame bound") if prompt.empty? || max_frames < 1
      raise ArgumentError.new("invalid prompt width") unless prompt.all? { |row| row.size == @tower.config.num_codebooks + 1 }
      raise ArgumentError.new("sampling top_k exceeds resident sampler limit 1024") unless top_k.in?(1..1024)
      validate_window!(prompt.size, max_frames)

      # Preserve the CPU path's parameter validation before any model work.
      Sampler.new(seed).candidates([0_f32], temperature, top_k, top_p)
      prepare_workspaces
      frame_width = @tower.config.num_codebooks + 1
      device_frames = @ctx.alloc(max_frames * frame_width)
      draw_stride = 2 + @tower.config.num_codebooks - 1
      draws = sampling_draws(seed, max_frames, draw_stride)
      draw_buffer = @ctx.upload(draws)

      frames_on_device = prompt.map { |frame| @tower.upload_frame(frame) }
      hidden = @ctx.alloc(@backbone.config.hidden)
      Profile.measure("generator_resident.prefill", category: "generation.prefill") do
        if @prefill_chunk > 1
          hidden = prefill_frames(frames_on_device)
        else
          frames_on_device.each_with_index do |frame, index|
            Profile.measure("generator_resident.prefill_row", category: "generation.prefill.row") do
              hidden = backbone_step(frame, frame.slice(0, 1))
              if index + 1 < frames_on_device.size
                with_workspace("tower.prefill") { @tower.teacher_force(hidden, frame, frames_on_device[index + 1]) }
              end
              # Bound live temporary allocations while persistent KV/recurrent
              # state remains owned by the resident model objects.
              @ctx.submit
            end
            @progress.try(&.call("prefill #{index + 1}/#{frames_on_device.size}"))
          end
        end
      end

      if trace = @device_trace
        check_workspace_traces!
        trace.call("prompt.backbone.hidden", hidden)
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
          draw_offset = step * draw_stride
          main = with_workspace("main") do
            logits = @ctx.matvec(@head, hidden)
            if step == 0
              @device_trace.try(&.call("prompt.main_logits", logits))
            end
            first = @ctx.sample(logits, draw_buffer.slice(draw_offset, 1), temperature, top_k, top_p)
            retry = @ctx.sample(logits, draw_buffer.slice(draw_offset + 1, 1), 1.0, 30, 0.95)
            choose_main(first, retry, window, window_size, window_cursor)
          end

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

          grow_for_frame!(step + 1 == max_frames)

          frame = Profile.measure("generator_resident.device_frame", category: "generation.frame") do
            # A generated frame escapes into previous_frame and the terminal
            # code array, so its backing is outside the reusable tower plan.
            output_frame = (@ctx.graph_enabled? || !@workspaces.empty?) ? device_frames.slice(step * frame_width, frame_width) : nil
            resident_frame = with_workspace("tower.sample") do
              sampled_codes = 0
              sampled = @tower.sample_frame(main, hidden, previous_frame, output_frame: output_frame) do |code_logits|
                draw_index = draw_offset + 2 + sampled_codes
                sampled_codes += 1
                @ctx.sample(code_logits, draw_buffer.slice(draw_index, 1), temperature, top_k, top_p)
              end
              raise "resident tower sampled #{sampled_codes} residual codes; expected #{@tower.config.num_codebooks - 1}" unless sampled_codes == @tower.config.num_codebooks - 1
              sampled
            end

            @ctx.copy(resident_frame, device_frames.slice(step * frame_width, frame_width)) unless output_frame
            if step + 1 < max_frames
              hidden = backbone_step(resident_frame, main)
            end
            # Submit the complete tower/backbone chain for this generated row.
            # In graph mode, the next main-token control read submits and
            # drains this chain together with that sample. The terminal read
            # drains the last generated row. Legacy mode remains synchronous.
            @ctx.finish unless @ctx.graph_enabled?
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

    private def generate_resident_impl(prompt : ResidentPrompt, max_frames : Int32, seed : UInt64,
                                       temperature : Float64, top_k : Int32, top_p : Float64) : ResidentGeneration
      raise ArgumentError.new("generator is single-use") unless @backbone.position == 0 && @tower.offset == 0
      raise ArgumentError.new("empty prompt or non-positive frame bound") if prompt.row_count < 1 || max_frames < 1
      raise ArgumentError.new("invalid prompt width") unless prompt.row_width == @tower.config.num_codebooks + 1
      raise ArgumentError.new("reference codebook count does not match model") unless prompt.codebooks == @tower.config.num_codebooks
      raise ArgumentError.new("reference codebook size does not match model") unless prompt.codebook_size == @tower.config.vocab_size
      raise ArgumentError.new("reference semantic token band does not match model") unless prompt.semantic_begin == @semantic_begin
      raise ArgumentError.new("prompt belongs to a different resident context") unless prompt.context.same?(@ctx)
      raise ArgumentError.new("sampling top_k exceeds resident sampler limit 1024") unless top_k.in?(1..1024)
      validate_window!(prompt.row_count, max_frames)

      # Validate all host-controlled inputs before model work, matching the
      # original resident API's boundary.
      Sampler.new(seed).candidates([0_f32], temperature, top_k, top_p)
      prepare_workspaces
      frame_width = @tower.config.num_codebooks + 1
      frame_values = max_frames.to_i64 * frame_width
      raise ArgumentError.new("resident output frame buffer is too large") if frame_values > Int32::MAX
      device_frames = @ctx.alloc(frame_values.to_i32)
      draw_stride = 2 + @tower.config.num_codebooks - 1
      draws = sampling_draws(seed, max_frames, draw_stride)
      draw_buffer = @ctx.upload(draws)
      frames_on_device = Array(Resident::Buffer).new(prompt.row_count) { |index| prompt.row(index) }

      hidden = @ctx.alloc(@backbone.config.hidden)
      Profile.measure("generator_resident.prefill", category: "generation.prefill") do
        if @prefill_chunk > 1
          hidden = prefill_frames(frames_on_device)
        else
          frames_on_device.each_with_index do |frame, index|
            Profile.measure("generator_resident.prefill_row", category: "generation.prefill.row") do
              hidden = backbone_step(frame, frame.slice(0, 1))
              if index + 1 < frames_on_device.size
                with_workspace("tower.prefill") { @tower.teacher_force(hidden, frame, frames_on_device[index + 1]) }
              end
              @ctx.submit
            end
            @progress.try(&.call("prefill #{index + 1}/#{frames_on_device.size}"))
          end
        end
      end

      if trace = @device_trace
        check_workspace_traces!
        trace.call("prompt.backbone.hidden", hidden)
      end

      window_size = Math.min(MAX_REPEAT_WINDOW, prompt.row_count)
      window = @ctx.alloc(window_size)
      @ctx.dispatch(
        "dr3_prompt_main_window",
        RESIDENT_PROMPT_SOURCE,
        [prompt.rows, window],
        [prompt.row_count.to_u64, prompt.row_width.to_u64, window_size.to_u64],
        window_size,
      )
      window_cursor = 0
      previous_frame = frames_on_device.last
      generated_count = 0
      stopped = false

      Profile.measure("generator_resident.autoregressive", category: "generation.decode") do
        max_frames.times do |step|
          draw_offset = step * draw_stride
          main = with_workspace("main") do
            logits = @ctx.matvec(@head, hidden)
            if step == 0
              @device_trace.try(&.call("prompt.main_logits", logits))
            end
            first = @ctx.sample(logits, draw_buffer.slice(draw_offset, 1), temperature, top_k, top_p)
            retry = @ctx.sample(logits, draw_buffer.slice(draw_offset + 1, 1), 1.0, 30, 0.95)
            choose_main(first, retry, window, window_size, window_cursor)
          end

          # This scalar is the existing explicit EOS control boundary. Invalid
          # reference IDs poison Context's finite status and fail here without
          # ever reading any codebook values to the CPU.
          main_token = @ctx.read(main, "control")[0].to_i32
          unless main_token == @eos || main_token.in?(@semantic_begin..@semantic_end)
            raise "resident main-token selector produced out-of-band token #{main_token}"
          end
          if main_token == @eos
            stopped = true
            break
          end

          grow_for_frame!(step + 1 == max_frames)

          frame = Profile.measure("generator_resident.device_frame", category: "generation.frame") do
            # A generated frame escapes into previous_frame and the terminal
            # code array, so its backing is outside the reusable tower plan.
            output_frame = (@ctx.graph_enabled? || !@workspaces.empty?) ? device_frames.slice(step * frame_width, frame_width) : nil
            resident_frame = with_workspace("tower.sample") do
              sampled_codes = 0
              sampled = @tower.sample_frame(main, hidden, previous_frame, output_frame: output_frame) do |code_logits|
                draw_index = draw_offset + 2 + sampled_codes
                sampled_codes += 1
                @ctx.sample(code_logits, draw_buffer.slice(draw_index, 1), temperature, top_k, top_p)
              end
              raise "resident tower sampled #{sampled_codes} residual codes; expected #{@tower.config.num_codebooks - 1}" unless sampled_codes == @tower.config.num_codebooks - 1
              sampled
            end

            @ctx.copy(resident_frame, device_frames.slice(step * frame_width, frame_width)) unless output_frame
            if step + 1 < max_frames
              hidden = backbone_step(resident_frame, main)
            end
            @ctx.finish unless @ctx.graph_enabled?
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
      generation = Generation.new(terminal_frames, stopped, prompt.row_count)
      ResidentGeneration.new(generation, device_frames)
    end

    private def validate_window!(prompt_rows : Int32, max_frames : Int32) : Nil
      raise ArgumentError.new("prompt exceeds initial resident cache capacity") if prompt_rows > @capacity
      raise ArgumentError.new("resident cache capacity exceeded") if prompt_rows.to_i64 + max_frames - 1 > @maximum_capacity
    end

    private def grow_for_frame!(last_frame : Bool) : Nil
      required = Math.max(@tower.offset + 1, @backbone.position + (last_frame ? 0 : 1))
      return if required <= @capacity
      raise ArgumentError.new("resident maximum cache capacity exceeded") if required > @maximum_capacity
      target = Math.min(@maximum_capacity, Math.max(required, @capacity * 2))
      previous = @capacity
      @before_window_growth.try(&.call(previous, target))
      Profile.measure("generator_resident.window_growth", category: "generation.window") do
        # The scalar control read has already drained preceding commands. New
        # bindings retain old storage through the queued GPU prefix copies.
        # Any partial allocation/copy failure aborts this single-use generator.
        @backbone.grow_capacity!(target)
        @tower.grow_capacity!(target)
        @capacity = target
        @window_growths << WindowExpansion.new(previous, target, @backbone.position, @tower.offset)
      end
      @after_window_growth.try(&.call(previous, target))
    end

    def workspace_stats
      {enabled: !@workspaces.empty?, workspaces: @workspaces.transform_values(&.stats)}
    end

    private def diagnostic_traces? : Bool
      !!(@device_trace || @backbone.trace || @backbone.buffer_trace || @backbone.moe_trace || @backbone.layer_buffer_trace || @backbone.prefill_trace)
    end

    private def prepare_workspaces : Nil
      # Diagnostic APIs may retain any intermediate. Preserve their original
      # fresh-output lifetime instead of returning borrowed workspace storage.
      return unless @generation_scratch && !diagnostic_traces?
      ["backbone", "backbone.prefill", "tower.prefill", "tower.sample", "main"].each do |name|
        @workspaces[name] = @ctx.new_workspace
      end
    end

    private def check_workspace_traces! : Nil
      raise ArgumentError.new("cannot enable resident traces during workspace generation") if !@workspaces.empty? && diagnostic_traces?
    end

    private def with_workspace(name : String, &)
      check_workspace_traces!
      if workspace = @workspaces[name]?
        @ctx.with_workspace(workspace) { yield }
      else
        yield
      end
    end

    private def backbone_step(frame : Resident::Buffer, token : Resident::Buffer) : Resident::Buffer
      with_workspace("backbone") do
        token_embedding = @backbone.embedding(token)
        @backbone.step(@tower.embed_backbone(frame, token_embedding))
      end
    end

    private def prefill_frames(frames : Array(Resident::Buffer)) : Resident::Buffer
      hidden = @ctx.alloc(@backbone.config.hidden)
      base = 0
      while base < frames.size
        rows = Math.min(@prefill_chunk, frames.size - base)
        table = with_workspace("backbone.prefill") do
          inputs = @ctx.alloc(rows * @backbone.config.hidden)
          rows.times do |row|
            frame = frames[base + row]
            embedding = @backbone.embedding(frame.slice(0, 1))
            mixed = @tower.embed_backbone(frame, embedding)
            @ctx.copy(mixed, inputs.slice(row * @backbone.config.hidden, @backbone.config.hidden))
          end
          @backbone.prefill(inputs, rows)
        end
        rows.times do |row|
          index = base + row
          hidden = table.slice(row * @backbone.config.hidden, @backbone.config.hidden)
          if index + 1 < frames.size
            with_workspace("tower.prefill") { @tower.teacher_force(hidden, frames[index], frames[index + 1]) }
          end
          # Existing prompt submission boundaries only. The first includes
          # this chunk's backbone; subsequent ones contain teacher forcing.
          @ctx.submit
          @progress.try(&.call("prefill #{index + 1}/#{frames.size}"))
        end
        base += rows
      end
      hidden
    end

    private def release_workspaces : Nil
      # The existing final control/terminal read has already drained the queue.
      # No extra finish or wait is introduced to retire these GPU backings.
      @workspaces.each_value { |workspace| @ctx.release_workspace(workspace) }
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
      draws = Array(Float32).new(max_frames * draw_stride) { random.rand.to_f32 }
      # Hash the actual host input array returned to the single GPU upload.
      # This is sampler-input provenance, never a model-activation readback.
      bytes = IO::Memory.new
      draws.each { |draw| bytes.write_bytes(draw, IO::ByteFormat::LittleEndian) }
      @sampler_inputs = {dtype: "float32-le", uniform_count: draws.size,
                         sha256: Digest::SHA256.hexdigest(bytes.to_slice),
                         provenance: "host input tape uploaded once"}
      draws
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
