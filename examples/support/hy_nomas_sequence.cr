require "./hy_nomas_timeline_inputs"
require "../../src/ml/video/hy_nomas_continuation_history"
require "../../src/ml/video/hy_nomas_timeline_ops"
require "../../src/ml/video/hy_nomas_resume_bundle"
require "./hy_nomas_cache_qbit_checkpoint"

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

  def checkpoint_inventory(inputs : HyNomasGeneration::TimelineInputs, segments : Int32,
                           next_segment : Int32) : Hash(String, Int64)
    shape = ML::Video::HyNomasContinuationHistory.geometry(validate_segments!(segments),
      4320, inputs.base.valid_text_tokens, inputs.base.compact_text_tokens, 54, 16, 128,
      text_cache_policy: HISTORY_TEXT_POLICY)
    bytes = shape.boundary_cache_elements(next_segment).to_i64 * sizeof(Float32)
    inventory = {"timeline" => inputs.records["input.latent_bank"].nbytes}
    54.times do |layer|
      inventory["history.layer#{layer}.key"] = bytes
      inventory["history.layer#{layer}.value"] = bytes
    end
    inventory
  end

  # Compile-time source bytes bind restart to the native implementation as well
  # as the authenticated model/conditioning/noise inventory. No reseeding occurs.
  def checkpoint_contract(inputs : HyNomasGeneration::TimelineInputs, segments : Int32,
                          next_segment : Int32) : String
    checkpoint_inventory(inputs, segments, next_segment)
    source_hash = Digest::SHA256.new
    {% for path in ["hy_nomas_sequence.cr", "../../src/ml/video/hy_nomas_continuation_history.cr", "../../src/ml/video/hy_nomas_timeline_ops.cr", "../../src/ml/video/kernels/hy_nomas_timeline_ops.metal", "../../src/ml/video/hy_nomas_euler.cr", "../../src/ml/video/kernels/hy_nomas_euler.metal", "../../src/ml/video/hy_nomas_block_mps.cr", "../../src/ml/video/hy_nomas_serial_mps.cr", "../../src/ml/video/hy_nomas_history_attention.cr", "../../src/ml/video/hy_nomas_history_attention_bridge.mm", "../../src/ml/video/hy_nomas_mps_bf16_linear_bridge.mm", "../../src/ml/video/hy_nomas_bf16_stack.cr"] %}
      source_hash.update({{ read_file(__DIR__ + "/" + path) }})
    {% end %}
    {schema: "cogni_ml.hy_nomas_sequence_boundary/v1", boundary: "after_noisy_writer",
     upstream: HyNomasGeneration::TimelineInputs::PINNED_REVISION,
     implementation_sha256: source_hash.hexfinal,
     executable_sha256: PixelUMMBlockProbe.sha_file(Process.executable_path || raise ArgumentError.new("checkpoint requires an identifiable executable")),
     crystal: Crystal::VERSION,
     model_index_sha256: HyNomasGeneration::INDEX_SHA,
     input_manifest_sha256: inputs.file.sha256, input_payload_sha256: inputs.file.document["payload_sha256"].as_s,
     conditioning_manifest_sha256: inputs.base.file.sha256,
     segments: segments, next_segment: next_segment, steps: STEPS, shift: 5,
     text_cache_policy: "once_per_scene", history_dtype: "F32",
     valid_text_tokens: inputs.base.valid_text_tokens, storage_text_tokens: inputs.base.compact_text_tokens,
     video_tokens: 4320, layers: 54, heads: 16, head_dim: 128,
     timeline_shape: [1, 32, 180, 9, 16], cache_sigma: 0.206897_f32}.to_json
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
    @next_segment = 0
    @resume_failed = false
    @capture_copy : ML::Video::HyNomasContinuationHistoryKernels::AppendPlan? = nil
    @checkpoint_contracts = {} of Int32 => String
    @checkpoint_inventories = {} of Int32 => Hash(String, Int64)
    @compact_restore : HyNomasCacheQBitCheckpoint::RestorePlan? = nil

    def initialize(inputs : HyNomasGeneration::TimelineInputs, @timeline : ML::MetalBuffer,
                   @latent : ML::MetalBuffer, budget : Int64, @segment_count : Int32,
                   admission : ML::Video::HyNomasQ4StackIndex::Admission,
                   text_cache_policy = HISTORY_TEXT_POLICY, *, prepare_capture : Bool = false)
      HyNomasSequence.validate_segments!(@segment_count)
      if text_cache_policy.once_per_scene?
        (1...@segment_count).each do |next_segment|
          @checkpoint_contracts[next_segment] = HyNomasSequence.checkpoint_contract(inputs, @segment_count, next_segment)
          @checkpoint_inventories[next_segment] = HyNomasSequence.checkpoint_inventory(inputs, @segment_count, next_segment)
        end
      end
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
        @capture_copy = ML::Video::HyNomasContinuationHistoryKernels::AppendPlan.new if prepare_capture
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

    def encode(chain : ML::Metal::MPSCommandChain, *, stop_after_segments : Int32 = @segment_count,
               capture_segment : Int32? = nil, captures : Array(ML::MetalBuffer)? = nil,
               capture_steps : Hash(Int32, Array(ML::MetalBuffer))? = nil,
               after_boundary : Proc(Int32, ML::Metal::MPSCommandChain, Nil)? = nil) : Nil
      raise ArgumentError.new("sequence request is single-use or has failed restore") if @released || @chain || @resume_failed
      unless @next_segment < stop_after_segments <= @segment_count
        raise ArgumentError.new("sequence stop must follow the next segment within the admitted run")
      end
      if capture_segment || captures
        raise ArgumentError.new("Do not combine capture interfaces") if capture_steps
        unless capture_segment && captures && @capture_copy &&
               @next_segment <= capture_segment < stop_after_segments && captures.size == STEPS
          raise ArgumentError.new("sequence stage capture must be prepared with one buffer per denoise step")
        end
        captures.each_with_index do |buffer, index|
          unless buffer.valid? && buffer.size == @latent.size && buffer.storage_mode.shared? &&
                 !ML::Video::MetalOps.overlap?(buffer, @latent) && !ML::Video::MetalOps.overlap?(buffer, @timeline) &&
                 captures.first(index).none? { |other| ML::Video::MetalOps.overlap?(buffer, other) }
            raise ArgumentError.new("sequence stage capture requires distinct exact latent buffers")
          end
        end
      end
      if selected_steps = capture_steps
        all_captures = [] of ML::MetalBuffer
        selected_steps.each do |segment, buffers|
          unless @capture_copy && @next_segment <= segment < stop_after_segments && buffers.size == STEPS
            raise ArgumentError.new("Sequence captures must cover admitted steps")
          end
          buffers.each do |buffer|
            unless buffer.valid? && buffer.size == @latent.size && buffer.storage_mode.shared? &&
                   !ML::Video::MetalOps.overlap?(buffer, @latent) && !ML::Video::MetalOps.overlap?(buffer, @timeline) &&
                   all_captures.none? { |other| ML::Video::MetalOps.overlap?(buffer, other) }
              raise ArgumentError.new("Sequence captures require distinct exact latent buffers")
            end
            all_captures << buffer
          end
        end
      end
      @chain = chain
      @compact_restore.try(&.encode_restore(chain))
      @prepared.encode(chain.encoder)
      (@next_segment...stop_after_segments).each do |segment|
        selected_captures = capture_steps.try(&.[segment]?)
        @timeline_ops.prepare_segment!(chain, segment)
        @eulers[segment].step_count.times do |step|
          @eulers[segment].encode_timestep(chain, step)
          @patch.encode(chain)
          @refiner.encode(chain)
          @history.select_segment!(segment, writer: false)
          @serial.encode(chain)
          @final.encode(chain)
          @eulers[segment].encode_update(chain, step)
          if capture_segment == segment && captures
            @capture_copy.not_nil!.encode(chain.encoder, @latent, captures[step],
              (@latent.size // sizeof(Float32)).to_i32, 0)
          end
          if selected_captures
            @capture_copy.not_nil!.encode(chain.encoder, @latent, selected_captures[step],
              (@latent.size // sizeof(Float32)).to_i32, 0)
          end
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
        @next_segment = segment + 1
        if @next_segment < @segment_count
          after_boundary.try(&.call(@next_segment, chain))
        end
        HyNomasGeneration.emit("sequence_segment_encoded", segment: segment + 1,
          clean_global_latent_stop: (segment + 1) * 30, cache_writer: segment + 1 < @segment_count)
      end
    end

    def prepare_checkpoint_bindings!(next_segment : Int32) : Hash(String, Tuple(ML::MetalBuffer, Int32))
      unless !@released && !@resume_failed && @chain.nil? && @next_segment == 0 &&
             @checkpoint_inventories.has_key?(next_segment)
        raise ArgumentError.new("Prepare checkpoint bindings on a fresh live request")
      end
      bindings = {"timeline" => {@timeline, (@timeline.size // sizeof(Float32)).to_i32}}
      @history.prepare_checkpoint_bindings(next_segment) do |name, buffer, count|
        bindings[name] = {buffer, count}
      end
      unless bindings.transform_values { |entry| entry[1].to_i64 * sizeof(Float32) } == @checkpoint_inventories[next_segment]
        raise ArgumentError.new("Prepared boundary inventory mismatch")
      end
      bindings
    end

    # A compact reader is authenticated before model allocation. Mark the owner
    # restored only after every setup copy succeeds, and always prepend GPU
    # reconstruction inside encode so no caller can omit it before attention.
    def restore_compact_boundary!(reader : HyNomasCacheQBitCheckpoint::Preflight, next_segment : Int32) : Nil
      bindings = prepare_checkpoint_bindings!(next_segment)
      contract = @checkpoint_contracts[next_segment]
      reader.validate_contract!(contract, @checkpoint_inventories[next_segment])
      begin
        cache_bindings = bindings.reject { |name, _| name == "timeline" }.map do |name, entry|
          HyNomasCacheQBit::Binding.new(name, entry[0], entry[1])
        end
        timeline_binding = HyNomasCacheQBit::Binding.new("timeline", @timeline, (@timeline.size // sizeof(Float32)).to_i32)
        @compact_restore = HyNomasCacheQBitCheckpoint::RestorePlan.new(cache_bindings, reader, timeline_binding)
        @history.restore_boundary!(next_segment) { |_, _, _| }
        @timeline_ops.restore_boundary!(next_segment)
        @next_segment = next_segment
      rescue ex
        @resume_failed = true
        raise ex
      end
    end

    def compact_restore_resident_bytes : Int64
      @compact_restore.try(&.resident_bytes) || 0_i64
    end

    def save_boundary!(path : String) : String
      chain = @chain
      unless !@released && !@resume_failed && chain && chain.completed? &&
             @timeline_ops.schedule.phase.ready? && @timeline_ops.schedule.next_segment == @next_segment
        raise ArgumentError.new("sequence checkpoint requires a drained extendable boundary")
      end
      contract = @checkpoint_contracts[@next_segment]? || raise ArgumentError.new("checkpoint requires the once-per-scene cache policy")
      buffers = {"timeline" => {@timeline, (@timeline.size // sizeof(Float32)).to_i32}}
      @history.each_checkpoint_cache(@next_segment) do |name, buffer, count|
        buffers[name] = {buffer, count}
      end
      inventory = buffers.transform_values { |entry| entry[1].to_i64 * sizeof(Float32) }
      unless inventory == @checkpoint_inventories[@next_segment]
        raise ArgumentError.new("sequence checkpoint inventory changed after setup")
      end
      ML::Video::HyNomasResumeBundle.write(path, contract, inventory) do |name, io|
        buffer, count = buffers[name]
        io.write(Bytes.new(buffer.contents.as(UInt8*), count * sizeof(Float32)))
      end
    end

    # The optional setup visitor lets a diagnostic prepare GPU bindings without
    # exposing history storage during an active request. No visitor can bypass
    # authentication, partial-restore poisoning, or the fresh-owner boundary.
    def restore_boundary!(reader : ML::Video::HyNomasResumeBundle::Reader, next_segment : Int32, *,
                          visit_cache : Proc(String, ML::MetalBuffer, Int32, Nil)? = nil) : Nil
      unless !@released && !@resume_failed && @chain.nil? && @next_segment == 0
        raise ArgumentError.new("sequence restore requires a fresh live request")
      end
      contract = @checkpoint_contracts[next_segment]? || raise ArgumentError.new("restore requires an extendable once-per-scene boundary")
      inventory = @checkpoint_inventories[next_segment]
      reader.validate_contract!(contract, inventory)
      begin
        reader.copy_to!("timeline", Bytes.new(@timeline.contents.as(UInt8*), @timeline.size.to_i32))
        @history.restore_boundary!(next_segment) do |name, buffer, count|
          reader.copy_to!(name, Bytes.new(buffer.contents.as(UInt8*), count * sizeof(Float32)))
          visit_cache.try(&.call(name, buffer, count))
        end
        @timeline_ops.restore_boundary!(next_segment)
        @next_segment = next_segment
      rescue ex
        @resume_failed = true
        raise ex
      end
    end

    def release : Nil
      return if @released
      active_chain = @chain
      if chain = active_chain
        raise ArgumentError.new("drain sequence before releasing resources") unless chain.completed?
      end
      actions = [] of Proc(Nil)
      @compact_restore.try { |plan| actions << -> { plan.release_after_drain(active_chain) } }
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
