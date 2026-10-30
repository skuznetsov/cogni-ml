require "./hy_nomas_timeline_inputs"
require "../../src/ml/video/hy_nomas_continuation_history"
require "../../src/ml/video/hy_nomas_timeline_ops"

# Opt-in finite continuation. The first-segment entry point remains separate;
# cache precision is F32 throughout this control, including persistent history.
module HyNomasSequence
  extend self

  STEPS               = 4
  HISTORY_TEXT_POLICY = ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene

  # Keep command-chain owners strongly reachable when a failed terminal wait
  # cannot prove that their GPU work has drained.
  class InFlightRetention
    @@retainers = [] of Proc(Nil)

    def self.retain(&block : -> Nil) : Nil
      @@retainers << block
    end
  end

  def release_actions(actions : Array(Proc(Nil))) : Array(String)
    failures = [] of String
    actions.each_with_index do |action, index|
      begin
        action.call
      rescue first
        begin
          # Owners often release several nested plans in sequence. One retry
          # lets them finish after a transient or partial first attempt.
          action.call
        rescue second
          failures << "release action #{index + 1}: #{first.class}: #{first.message}; retry: #{second.class}: #{second.message}"
        end
      end
    end
    failures
  end

  def retain_in_flight(&block : -> Nil) : Nil
    InFlightRetention.retain { block.call }
  end

  def validate_segments!(segments : Int32) : Int32
    raise ArgumentError.new("sequence requires two through six segments") unless (2..6).includes?(segments)
    segments
  end

  def forwards(segments : Int32) : Int32
    validate_segments!(segments) * STEPS + segments - 1
  end

  # The initial continuation entry point reuses one scene-level Qwen state.
  # Tagged actions require upstream's per-segment action-tail masks, which this
  # entry point does not yet implement. Reject them before Metal allocation.
  def validate_shared_scene_prompt!(prompt : String) : Nil
    document = JSON.parse(prompt)
    object = document.as_h?
    segments = object.try { |fields| fields["segments"]?.try(&.as_a?) }
    unless segments && segments.size == 1
      raise ArgumentError.new("shared continuation requires exactly one JSON scene segment")
    end
    segment = segments.first.as_h?
    raise ArgumentError.new("shared continuation requires a JSON scene object") unless segment
    if actions = segment["contained_actions"]?
      unless actions.raw.nil? || actions.as_a?.try(&.empty?)
        raise ArgumentError.new("shared continuation does not support action tails; move motion into the scene description")
      end
    end
  rescue ex : JSON::ParseException
    raise ArgumentError.new("shared continuation requires a JSON scene prompt")
  end

  def history_bytes(inputs : HyNomasGeneration::Inputs, segments : Int32,
                    text_cache_policy = HISTORY_TEXT_POLICY) : Int64
    ML::Video::HyNomasContinuationHistory.named_bytes(segment_count: validate_segments!(segments),
      video_tokens: 4320, valid_text_tokens: inputs.valid_text_tokens,
      storage_text_tokens: inputs.compact_text_tokens, layers: 54, heads: 16, head_dim: 128,
      text_cache_policy: text_cache_policy)
  end

  def additional_inputs_bytes(inputs : HyNomasGeneration::TimelineInputs, segments : Int32) : Int64
    # The fixed current latent, Qwen and one pair of current RoPE tables are
    # already included by the first-segment calculation. Noise for PixelUMM is
    # uploaded after this entire weight phase is drained and released.
    inputs.bytes(%w(input.latent_bank input.rope_cos input.rope_sin cache.eps_bf16)) +
      (validate_segments!(segments) - 1).to_i64 * HyNomasGeneration.euler_schedule_bytes(STEPS)
  end

  def old_attention_bytes(inputs : HyNomasGeneration::Inputs) : Int64
    ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1,
      4320 + inputs.valid_text_tokens, 16, 128, 256, 16).resident_bytes
  end

  # Include the temporarily coexisting original attention plan while the
  # prepared provider is attached. Steady residency subtracts that old plan.
  def peak_named_bytes(inputs : HyNomasGeneration::TimelineInputs, segments : Int32,
                       text_cache_policy = HISTORY_TEXT_POLICY) : Int64
    HyNomasGeneration.dit_resident_weight_bytes(HyNomasGeneration::DiTWeights::BF16,
      HyNomasGeneration.stage_bytes, HyNomasGeneration.stage_bytes * 54) +
      HyNomasGeneration.dit_reserve_bytes(inputs.base, STEPS) +
      history_bytes(inputs.base, segments, text_cache_policy) + additional_inputs_bytes(inputs, segments)
  end

  class DiT
    getter named_bytes : Int64
    getter peak_named_bytes : Int64
    getter segment_count : Int32
    @inputs_arena : ML::Video::Arena
    @bank_arena : ML::Video::Arena
    @globals : ML::Video::Arena
    @stage : ML::Video::Arena
    @file : ML::Video::CheckpointFile
    @prepared : ML::Video::PreparedWeights
    @loader : ML::Video::HyNomasBF16Stack
    @patch : ML::Video::HyNomasPatchMPSPlan
    @refiner : ML::Video::HyNomasTextRefinerMPSPlan
    @serial : ML::Video::HyNomasMPSSerialWorkspace
    @final : ML::Video::HyNomasFinalMPSPlan
    @eulers = [] of ML::Video::HyNomasEulerPlan
    @history : ML::Video::HyNomasContinuationHistory
    @timeline_ops : ML::Video::HyNomasTimelineOps::Plan
    @latent : ML::MetalBuffer
    @timeline : ML::MetalBuffer
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false
    @timeline_released = false

    def initialize(inputs : HyNomasGeneration::TimelineInputs, @timeline : ML::MetalBuffer,
                   @latent : ML::MetalBuffer, budget : Int64, @segment_count : Int32,
                   admission : ML::Video::HyNomasQ4StackIndex::Admission,
                   text_cache_policy = HISTORY_TEXT_POLICY)
      HyNomasSequence.validate_segments!(@segment_count)
      base = inputs.base
      @peak_named_bytes = HyNomasSequence.peak_named_bytes(inputs, @segment_count, text_cache_policy)
      @named_bytes = @peak_named_bytes - HyNomasSequence.old_attention_bytes(base)
      raise ArgumentError.new("sequence peak named buffers exceed budget") unless @peak_named_bytes <= budget
      admission.validate!(HyNomasGeneration::INDEX_SHA, HyNomasQ4DiTIO::SOURCE_SHA,
        HyNomasGeneration.weight_cache_config)
      plan_cleanup = [] of Proc(Nil)
      arena_cleanup = [] of Proc(Nil)
      loader_cleanup : Proc(Nil)? = nil
      prepared_cleanup : Proc(Nil)? = nil
      file_cleanup : Proc(Nil)? = nil
      # The constructor re-raises every setup failure after rollback. Mark the
      # remaining fields initialized for Crystal's flow analysis; callbacks
      # are registered only after their corresponding owner exists.
      @inputs_arena = uninitialized ML::Video::Arena
      @bank_arena = uninitialized ML::Video::Arena
      @globals = uninitialized ML::Video::Arena
      @stage = uninitialized ML::Video::Arena
      @file = uninitialized ML::Video::CheckpointFile
      @prepared = uninitialized ML::Video::PreparedWeights
      @loader = uninitialized ML::Video::HyNomasBF16Stack
      @patch = uninitialized ML::Video::HyNomasPatchMPSPlan
      @refiner = uninitialized ML::Video::HyNomasTextRefinerMPSPlan
      @serial = uninitialized ML::Video::HyNomasMPSSerialWorkspace
      @final = uninitialized ML::Video::HyNomasFinalMPSPlan
      @history = uninitialized ML::Video::HyNomasContinuationHistory
      @timeline_ops = uninitialized ML::Video::HyNomasTimelineOps::Plan
      begin
        @file = ML::Video::BF16WeightCache.authenticate(admission.globals_bf16_path, admission.globals_bf16_sha256)
        file_cleanup = -> { @file.close }
        bindings = HyNomasQ4DiTIO::GLOBAL_BINDINGS.map do |b|
          ML::Video::WeightBinding.new(b.target, b.source, b.shape, ML::GGUF::SafeTensorDType::BF16)
        end
        plan = ML::Video::WeightPlan.new([@file], bindings, budget,
          resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        unless plan.staging_bytes == 0 && plan.entries.size == 48
          raise ArgumentError.new("sequence globals must use exact BF16")
        end
        weights_bytes = 55_i64 * HyNomasGeneration.stage_bytes
        @loader = ML::Video::HyNomasBF16Stack.new(admission, HyNomasGeneration::INDEX_SHA,
          HyNomasQ4DiTIO::SOURCE_SHA, HyNomasGeneration.weight_cache_config, budget,
          stage_bytes: HyNomasGeneration.stage_bytes,
          reserve_bytes: @peak_named_bytes - weights_bytes,
          max_forwards_per_chain: HyNomasSequence.forwards(@segment_count))
        loader_cleanup = -> { @loader.release }
        @inputs_arena = ML::Video::Arena.new
        arena_cleanup << -> { @inputs_arena.release }
        @bank_arena = ML::Video::Arena.new
        arena_cleanup << -> { @bank_arena.release }
        @globals = ML::Video::Arena.new
        arena_cleanup << -> { @globals.release }
        @stage = ML::Video::Arena.new
        arena_cleanup << -> { @stage.release }
        @prepared = ML::Video::PreparedWeights.new(plan, @globals)
        prepared_cleanup = -> { @prepared.release }
        ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(HyNomasGeneration::CONFIG).each do |b|
          @stage.alloc_weight(b.target, b.shape.reduce(1_i64) { |n, d| n * d }.to_i32, ML::Video::WeightDType::BF16)
        end
        HyNomasGeneration.emit("sequence_load_blocks", peak_named_bytes: @peak_named_bytes)
        @loader.load!(@stage)
        qwen = base.upload!("input.qwen_bf16", @inputs_arena)
        cos = base.upload!("input.rope_cos", @inputs_arena)
        sin = base.upload!("input.rope_sin", @inputs_arena)
        cos_bank = inputs.upload!("input.rope_cos", @bank_arena)
        sin_bank = inputs.upload!("input.rope_sin", @bank_arena)
        eps = inputs.upload!("cache.eps_bf16", @bank_arena)
        @patch = ML::Video::HyNomasPatchMPSPlan.new(@latent, 4320, @globals)
        plan_cleanup << -> { @patch.release }
        @refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, base.mask, 1000_f32, @globals,
          ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(base.mask, retain_padding: true).total,
          retain_padding: true)
        plan_cleanup << -> { @refiner.release }
        @serial = ML::Video::HyNomasMPSSerialWorkspace.new(HyNomasGeneration::CONFIG,
          Array(ML::Video::Arena).new(54) { @stage }, @patch.output, @refiner.text_output,
          @refiner.vec, @refiner.vec_txt, cos, sin, attention_query_chunk: 256,
          attention_heads_per_group: 16, valid_text_tokens: base.valid_text_tokens,
          allow_shared_modulation: true, full_dit: true, weight_loader: @loader,
          compact_valid_text: true, compact_text_tokens: base.compact_text_tokens)
        plan_cleanup << -> { @serial.release }
        blocks = @serial.blocks
        joint = blocks.first.checkpoints
        @history = ML::Video::HyNomasContinuationHistory.new(segment_count: @segment_count,
          video_tokens: 4320, valid_text_tokens: base.valid_text_tokens,
          storage_text_tokens: base.compact_text_tokens, layers: 54, heads: 16, head_dim: 128,
          text_cache_policy: text_cache_policy,
          query: joint["joint.query"], key: joint["joint.key"],
          value: joint["joint.value"], output: joint["joint.attention"])
        plan_cleanup << -> { @history.release }
        @serial.attach_history_provider!(@history)
        @final = ML::Video::HyNomasFinalMPSPlan.new(blocks.last.image_output, @refiner.vec, 4320, @globals)
        plan_cleanup << -> { @final.release }
        @segment_count.times do
          @eulers << ML::Video::HyNomasEulerPlan.new(@latent, @final.output,
            @refiner.checkpoints["timestep"], steps: STEPS, shift: 5_f32)
        end
        @timeline_ops = ML::Video::HyNomasTimelineOps::Plan.new(ML::Video::HyNomasTimelineOps::Geometry.new,
          timeline: @timeline, segment_latent: @latent, cosine_bank: cos_bank, sine_bank: sin_bank,
          cosine: cos, sine: sin, cache_eps_bf16: eps,
          cache_timestep: @refiner.checkpoints["timestep"], run_segments: @segment_count)
        actual = @loader.cache_payload_bytes + @stage.resident_bytes + @globals.resident_bytes +
                 @inputs_arena.resident_bytes + @bank_arena.resident_bytes + @latent.size + @timeline.size + @patch.resident_bytes +
                 @refiner.resident_bytes + @serial.resident_bytes + @final.resident_bytes +
                 @eulers.sum(&.resident_bytes)
        unless actual == @named_bytes
          raise ArgumentError.new("sequence allocation/admission mismatch: #{actual} != #{@named_bytes}")
        end
        HyNomasGeneration.emit("sequence_prepared", named_bytes: actual, segments: @segment_count,
          history_precision: "F32 exact native cache boundary")
      rescue ex
        cleanup = [] of Proc(Nil)
        cleanup << -> { @eulers.reverse_each(&.release) } unless @eulers.empty?
        cleanup.concat(plan_cleanup.reverse)
        cleanup << loader_cleanup.not_nil! if loader_cleanup
        cleanup << prepared_cleanup.not_nil! if prepared_cleanup
        cleanup.concat(arena_cleanup.reverse)
        cleanup << file_cleanup.not_nil! if file_cleanup
        HyNomasSequence.release_actions(cleanup).each do |message|
          STDERR.puts("sequence constructor rollback: #{message}")
        end
        raise ex
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("sequence request is single-use") if @released || @chain
      @chain = chain
      @prepared.encode(chain.encoder)
      @segment_count.times do |segment|
        @timeline_ops.prepare_segment!(chain, segment)
        @eulers[segment].step_count.times do |step|
          @eulers[segment].encode_timestep(chain, step)
          @patch.encode(chain)
          @refiner.encode(chain)
          @history.select_segment!(segment, writer: false)
          @serial.encode(chain)
          @final.encode(chain)
          @eulers[segment].encode_update(chain, step)
          HyNomasGeneration.emit("sequence_step_encoded", segment: segment + 1, step: step + 1)
        end
        @timeline_ops.mark_denoise_complete!(chain, segment)
        @timeline_ops.scatter_generated!(chain, segment)
        if segment + 1 < @segment_count
          @timeline_ops.write_cache_noise!(chain, segment)
          @history.select_segment!(segment, writer: true)
          @patch.encode(chain)
          @refiner.encode(chain)
          @serial.encode(chain)
          @timeline_ops.record_cache_forward_enqueued!(chain, segment)
        end
        @timeline_ops.advance_segment!(chain, segment)
        HyNomasGeneration.emit("sequence_segment_encoded", segment: segment + 1,
          clean_global_latent_stop: (segment + 1) * 30, cache_writer: segment + 1 < @segment_count)
      end
    end

    def release : Nil
      return if @released
      active_chain = @chain
      if chain = active_chain
        raise ArgumentError.new("drain sequence before releasing resources") unless chain.completed?
      end
      actions = [] of Proc(Nil)
      if chain = active_chain
        actions << -> do
          unless @timeline_released
            @timeline_ops.release_after_drain!(chain)
            @timeline_released = true
          end
        end
      end
      @eulers.reverse_each do |euler|
        actions << -> { euler.release }
      end
      actions << -> { @final.release }
      actions << -> { @serial.release }
      # The serial workspace normally releases this provider itself. Preserve
      # an independent fallback if that nested cleanup fails partway through.
      actions << -> { @history.release }
      actions << -> { @refiner.release }
      actions << -> { @patch.release }
      actions << -> { @loader.release }
      actions << -> { @prepared.release }
      actions << -> { @stage.release }
      actions << -> { @globals.release }
      actions << -> { @inputs_arena.release }
      actions << -> { @bank_arena.release }
      actions << -> { @file.close }
      failures = HyNomasSequence.release_actions(actions)
      unless failures.empty?
        raise ArgumentError.new("sequence cleanup failed: #{failures.join("; ")}")
      end
      @released = true
    end
  end
end
