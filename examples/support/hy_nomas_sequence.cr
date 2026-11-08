require "./hy_nomas_timeline_inputs"
require "./hy_nomas_cinematic"
require "../../src/ml/video/hy_nomas_continuation_history"
require "../../src/ml/video/hy_nomas_source_bf16_mps"
require "../../src/ml/video/hy_nomas_timeline_ops"
require "../../src/ml/video/hy_nomas_resume_bundle"
require "../../src/ml/video/hy_nomas_segment_edit"
require "../../src/ml/video/hy_nomas_reference_action"
require "./hy_nomas_cache_qbit_checkpoint"

# Opt-in finite continuation. The first-segment entry point remains separate;
# cache precision is F32 throughout this control, including persistent history.
module HyNomasSequence
  extend self

  STEPS                  = 4
  HISTORY_TEXT_POLICY    = ML::Video::HyNomasContinuationHistory::TextCachePolicy::OncePerScene
  REFERENCE_LATENT_BYTES = 1_i64 * 32 * 30 * 9 * 16 * sizeof(UInt16)
  T0_HISTORY_LATENTS     =  30
  T0_HISTORY_TOKENS      = 144

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

  def weight_index_sha(cinematic : HyNomasCinematic::IndexIdentity?) : String
    cinematic.try(&.sha256) || HyNomasGeneration::INDEX_SHA
  end

  def weight_identity(cinematic : HyNomasCinematic::IndexIdentity?)
    {
      dit_variant:              cinematic ? "cinematic-active-blocks" : "original",
      dit_weight_index_sha256:  weight_index_sha(cinematic),
      cinematic_lineage:        cinematic.try(&.lineage_path),
      cinematic_lineage_sha256: cinematic.try(&.lineage_sha256),
    }
  end

  def validate_checkpoint_variant!(cinematic : HyNomasCinematic::IndexIdentity?, *,
                                   prepare_capture : Bool = false,
                                   checkpoint : Bool = false,
                                   restore : Bool = false) : Nil
    return unless cinematic && (prepare_capture || checkpoint || restore)
    raise ArgumentError.new("Cinematic sequence checkpoints and captures are base-only")
  end

  def forwards(segments : Int32) : Int32
    validate_segments!(segments) * STEPS + segments - 1
  end

  # The initial continuation entry point reuses one scene-level Qwen state.
  # Tagged actions are admitted only by the separately authenticated reference
  # action profile; the shared scene path continues to reject them.
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

  # The bounded reference test maps Subject1 to Alex in the global T0 roster.
  # Subject tags are literal prompt text, so a valid T1 profile alone cannot
  # establish which character its action is meant to address.
  def validate_reference_subject_roster!(caption_prompt : String, *,
                                         sequential_actions : Bool = false) : Nil
    validate_shared_scene_prompt!(caption_prompt)
    document = JSON.parse(caption_prompt)
    subjects = document.as_h["subject_list"].as_a
    if sequential_actions
      unless subjects.size == 2
        raise ArgumentError.new("Sequential reference actions require the ordered T0 Subject1 Alex and Subject2 Resume roster")
      end
      [{"<|Subject1|>", "Alex"}, {"<|Subject2|>", "Resume"}].each_with_index do |(identifier, name), index|
        subject = subjects[index].as_h
        unless subject["subject_id"].as_s == identifier &&
               subject["display_name"].as_s == name &&
               !subject["subject_attribute"].as_s.strip.empty?
          raise ArgumentError.new("Sequential reference actions require the ordered T0 Subject1 Alex and Subject2 Resume roster")
        end
      end
      return
    end
    subject = subjects.size == 1 ? subjects.first.as_h : nil
    unless subject && subject["subject_id"].as_s == "<|Subject1|>" &&
           subject["display_name"].as_s == "Alex" &&
           !subject["subject_attribute"].as_s.strip.empty?
      raise ArgumentError.new("Structured reference action requires the single T0 Subject1 roster entry for Alex")
    end
  rescue ex : JSON::ParseException | KeyError | TypeCastError
    if sequential_actions
      raise ArgumentError.new("Sequential reference actions require the ordered T0 Subject1 Alex and Subject2 Resume roster")
    end
    raise ArgumentError.new("Structured reference action requires the single T0 Subject1 roster entry for Alex")
  end

  def text_tokens(inputs : HyNomasGeneration::Inputs, segments : Int32,
                  prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil, *,
                  reference_action : ML::Video::HyNomasReferenceAction? = nil) : Array(Int32)
    validate_text_condition_pair!(prompt_edit, reference_action)
    if action = reference_action
      validate_reference_action_segments!(action, segments)
      action.text_tokens(inputs.valid_text_tokens)
    else
      prompt_edit.try(&.text_tokens(inputs.valid_text_tokens)) || Array.new(validate_segments!(segments), inputs.valid_text_tokens)
    end
  end

  def cache_text(segments : Int32, prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil, *,
                 reference_action : ML::Video::HyNomasReferenceAction? = nil) : Array(Bool)
    validate_text_condition_pair!(prompt_edit, reference_action)
    if action = reference_action
      validate_reference_action_segments!(action, segments)
      action.cache_text
    else
      prompt_edit.try(&.cache_text) || Array.new(validate_segments!(segments)) { |segment| segment == 0 }
    end
  end

  def query_text_tokens(inputs : HyNomasGeneration::Inputs,
                        prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil, *,
                        reference_action : ML::Video::HyNomasReferenceAction? = nil) : Int32
    validate_text_condition_pair!(prompt_edit, reference_action)
    if action = reference_action
      action.query_text_tokens(inputs.valid_text_tokens)
    else
      Math.max(inputs.valid_text_tokens, prompt_edit.try(&.valid_text_tokens) || inputs.valid_text_tokens)
    end
  end

  private def validate_text_condition_pair!(prompt_edit : ML::Video::HyNomasSegmentPromptEdit?,
                                            reference_action : ML::Video::HyNomasReferenceAction?) : Nil
    if prompt_edit && reference_action
      raise ArgumentError.new("Reference actions cannot be combined with segment prompt edits")
    end
  end

  private def validate_reference_action_segments!(action : ML::Video::HyNomasReferenceAction,
                                                  segments : Int32) : Nil
    unless segments == 2 && action.run_segments == segments
      raise ArgumentError.new("Reference action requires exactly two segments")
    end
  end

  def reference_action_bytes(action : ML::Video::HyNomasReferenceAction?) : Int64
    return 0_i64 unless action
    mask = Array.new(1280) { |i| i < action.valid_text_tokens ? 1 : 0 }
    qwen_bytes = 1280_i64 * 3584 * sizeof(UInt16)
    qwen_bytes + ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask, retain_padding: true).total
  end

  def prompt_edit_bytes(edit : ML::Video::HyNomasSegmentPromptEdit?) : Int64
    return 0_i64 unless edit
    mask = Array.new(1280) { |i| i < edit.valid_text_tokens ? 1 : 0 }
    1280_i64 * 3584 * sizeof(UInt16) +
      ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask, retain_padding: true).total
  end

  def history_bytes(inputs : HyNomasGeneration::Inputs, segments : Int32,
                    text_cache_policy = HISTORY_TEXT_POLICY, *,
                    prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil,
                    reference_action : ML::Video::HyNomasReferenceAction? = nil,
                    source_bf16 : Bool = false,
                    t0_key_visible_latents : Int32? = nil,
                    t0_hide_caption_keys : Bool = false,
                    t1_current_text_kv : Bool = false) : Int64
    valid = query_text_tokens(inputs, prompt_edit, reference_action: reference_action)
    validate_t0_history_visibility!(t0_key_visible_latents, t0_hide_caption_keys, t1_current_text_kv)
    t0_profile_enabled = !t0_key_visible_latents.nil? || t0_hide_caption_keys || t1_current_text_kv
    if t0_profile_enabled && (prompt_edit || reference_action)
      raise ArgumentError.new("T0 history visibility cannot be combined with prompt or action edits")
    end
    history_text = history_text_profile(inputs, segments, prompt_edit, reference_action: reference_action,
      t1_current_text_kv: t1_current_text_kv)
    attention_profiles = if t0_profile_enabled
                           t0_history_attention_profiles(inputs.valid_text_tokens, t0_key_visible_latents.not_nil!,
                             t0_hide_caption_keys: t0_hide_caption_keys,
                             t1_current_text_kv: t1_current_text_kv)
                         else
                           nil
                         end
    ML::Video::HyNomasContinuationHistory.named_bytes(segment_count: validate_segments!(segments),
      video_tokens: 4320, valid_text_tokens: valid,
      storage_text_tokens: ((valid + 63) // 64) * 64, layers: 54, heads: 16, head_dim: 128,
      text_cache_policy: text_cache_policy,
      segment_text_tokens: history_text[:segment_text_tokens],
      segment_cache_text: history_text[:segment_cache_text],
      segment_attention_labels: reference_attention_labels(inputs.valid_text_tokens, reference_action),
      segment_attention_profiles: attention_profiles,
      source_bf16_attention: source_bf16)
  end

  # T1 uses a fixed-shape masked workspace in every visibility arm.
  # `visible_latents` controls which T0 video keys receive label -1; the
  # optional caption mask affects only T1. T0 itself remains unmasked.
  def validate_t0_history_visibility!(visible_latents : Int32?, t0_hide_caption_keys : Bool,
                                      t1_current_text_kv : Bool = false) : Nil
    if t1_current_text_kv && (!t0_hide_caption_keys || !(visible_latents == 30 || visible_latents == 0))
      raise ArgumentError.new("T1 current text K/V requires hidden T0 caption keys and 30 or 0 T0 video latents visible")
    end
    zero_video_control = visible_latents == 0 && t0_hide_caption_keys && t1_current_text_kv
    if visible_latents == 0 && !zero_video_control
      raise ArgumentError.new("Hiding all T0 video keys requires hidden T0 caption keys and current T1 text K/V")
    end
    return if visible_latents.nil? && !t0_hide_caption_keys && !t1_current_text_kv
    if t0_hide_caption_keys && visible_latents != 30 && !zero_video_control
      raise ArgumentError.new("Hiding T0 caption keys requires all 30 T0 video latents visible")
    end
    visible = visible_latents || raise ArgumentError.new("T0 history probe keeps exactly 10 or 30 latents and requires valid caption text")
    unless {10, 30}.includes?(visible) || zero_video_control
      raise ArgumentError.new("T0 history probe keeps exactly 10 or 30 latents and requires valid caption text")
    end
  end

  def validate_t0_history_keep_argument!(visible_latents : Int32) : Nil
    raise ArgumentError.new("--keep-t0-latents is required") if visible_latents == -1
    unless {0, 10, 30}.includes?(visible_latents)
      raise ArgumentError.new("--keep-t0-latents must be 10, 30, or the guarded 0-latent control")
    end
  end

  # Pure pre-allocation admission shared by the sizing path and GPU owner.
  def validate_t0_history_admission!(visible_latents : Int32?, t0_hide_caption_keys : Bool,
                                     t1_current_text_kv : Bool = false, *,
                                     source_bf16 : Bool,
                                     reference_latent_bytes : Int64,
                                     segments : Int32,
                                     text_cache_policy = HISTORY_TEXT_POLICY,
                                     cinematic : HyNomasCinematic::IndexIdentity? = nil,
                                     prepare_capture : Bool = false,
                                     noise_edit : ML::Video::HyNomasSegmentNoiseEdit? = nil,
                                     prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil,
                                     reference_action : ML::Video::HyNomasReferenceAction? = nil) : Nil
    validate_t0_history_visibility!(visible_latents, t0_hide_caption_keys, t1_current_text_kv)
    return if visible_latents.nil? && !t0_hide_caption_keys && !t1_current_text_kv
    unless source_bf16 && reference_latent_bytes == REFERENCE_LATENT_BYTES && segments == 2 &&
           text_cache_policy.once_per_scene? && !cinematic && !prepare_capture &&
           !noise_edit && !prompt_edit && !reference_action
      raise ArgumentError.new("T0 history visibility requires a two-segment source-BF16 reference run without edits")
    end
  end

  # The final T1 current rows are included in its reader workspace. The
  # continuation schedule has no final-segment writer, so these rows are not
  # appended to persistent history.
  def t1_current_text_profile(valid_text_tokens : Int32, segments : Int32) : NamedTuple(
    segment_text_tokens: Array(Int32), segment_cache_text: Array(Bool))
    unless valid_text_tokens > 0
      raise ArgumentError.new("T1 current text K/V requires valid caption text")
    end
    count = validate_segments!(segments)
    {segment_text_tokens: Array.new(count, valid_text_tokens),
     segment_cache_text:  Array.new(count, true)}
  end

  def history_text_profile(inputs : HyNomasGeneration::Inputs, segments : Int32,
                           prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil, *,
                           reference_action : ML::Video::HyNomasReferenceAction? = nil,
                           t1_current_text_kv : Bool = false) : NamedTuple(
    segment_text_tokens: Array(Int32)?, segment_cache_text: Array(Bool)?)
    if t1_current_text_kv
      profile = t1_current_text_profile(inputs.valid_text_tokens, segments)
      {segment_text_tokens: profile[:segment_text_tokens],
       segment_cache_text:  profile[:segment_cache_text]}
    elsif prompt_edit || reference_action
      {segment_text_tokens: text_tokens(inputs, segments, prompt_edit, reference_action: reference_action),
       segment_cache_text:  cache_text(segments, prompt_edit, reference_action: reference_action)}
    else
      {segment_text_tokens: nil, segment_cache_text: nil}
    end
  end

  def t0_history_attention_profiles(caption_valid_text_tokens : Int32,
                                    visible_latents : Int32, *,
                                    t0_hide_caption_keys : Bool = false,
                                    t1_current_text_kv : Bool = false) : Array(ML::Video::HyNomasContinuationHistory::AttentionLabels?)
    validate_t0_history_visibility!(visible_latents, t0_hide_caption_keys, t1_current_text_kv)
    unless caption_valid_text_tokens > 0
      raise ArgumentError.new("T0 history probe keeps exactly 10 or 30 latents and requires valid caption text")
    end
    query_rows = 4320 + caption_valid_text_tokens
    key_rows = 2 * 4320 + caption_valid_text_tokens * (t1_current_text_kv ? 2 : 1)
    query = Array(Int32).new(query_rows, 0_i32)
    key = Array(Int32).new(key_rows, 0_i32)
    hidden_rows = (T0_HISTORY_LATENTS - visible_latents) * T0_HISTORY_TOKENS
    hidden_rows.times { |row| key[row] = -1_i32 }
    if t0_hide_caption_keys
      caption_valid_text_tokens.times { |row| key[4320 + row] = -1_i32 }
    end
    label = ML::Video::HyNomasContinuationHistory::AttentionLabels.new(query, key)
    [nil, label]
  end

  # The current action path is deliberately bounded to caption-only T0 and one
  # final T1. Earlier action tails would need upstream's head-only history mask.
  def reference_attention_labels(caption_valid_text_tokens : Int32,
                                 action : ML::Video::HyNomasReferenceAction?) : Array(ML::Video::HyNomasContinuationHistory::AttentionLabels)?
    return nil unless action && (profile = action.action_conditioning)
    query_text = action.query_text_tokens(caption_valid_text_tokens)
    caption_query = Array(Int32).new(4320 + query_text) { |i| i < 4320 + caption_valid_text_tokens ? 0_i32 : -1_i32 }
    caption_key = Array(Int32).new(4320 + caption_valid_text_tokens, 0_i32)
    labels = profile.labels(video_tokens: 4320, query_text_tokens: query_text,
      history_rows: 4320 + caption_valid_text_tokens, current_text_tokens: action.valid_text_tokens)
    [ML::Video::HyNomasContinuationHistory::AttentionLabels.new(caption_query, caption_key),
     ML::Video::HyNomasContinuationHistory::AttentionLabels.new(labels.query_labels, labels.key_labels)]
  end

  def additional_inputs_bytes(inputs : HyNomasGeneration::TimelineInputs, segments : Int32) : Int64
    # The fixed current latent, Qwen and one pair of current RoPE tables are
    # already included by the first-segment calculation. Noise for PixelUMM is
    # uploaded after this entire weight phase is drained and released.
    inputs.bytes(%w(input.latent_bank input.rope_cos input.rope_sin cache.eps_bf16)) +
      (validate_segments!(segments) - 1).to_i64 * HyNomasGeneration.euler_schedule_bytes(STEPS)
  end

  def old_attention_bytes(inputs : HyNomasGeneration::Inputs,
                          prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil, *,
                          reference_action : ML::Video::HyNomasReferenceAction? = nil) : Int64
    ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(1,
      4320 + query_text_tokens(inputs, prompt_edit, reference_action: reference_action), 16, 128, 256, 16).resident_bytes
  end

  def validate_portrait_t0_context!(mode : String, visible_latents : Int32,
                                    t0_hide_caption_keys : Bool = false,
                                    t1_current_text_kv : Bool = false) : Nil
    unless {"repeated-photo", "generated"}.includes?(mode)
      raise ArgumentError.new("T0 context must be repeated-photo or generated")
    end
    return unless mode == "generated"
    unless visible_latents == 30 && !t0_hide_caption_keys && !t1_current_text_kv
      raise ArgumentError.new("generated T0 context requires keep=30 with no T0 history masks")
    end
  end

  def validate_source_bf16_initialization!(*,
                                           source_bf16 : Bool,
                                           reference_latent_bytes : Int64,
                                           segments : Int32,
                                           generated_t0_control : Bool = false,
                                           text_cache_policy = HISTORY_TEXT_POLICY,
                                           cinematic : HyNomasCinematic::IndexIdentity? = nil,
                                           prepare_capture : Bool = false,
                                           noise_edit : ML::Video::HyNomasSegmentNoiseEdit? = nil,
                                           prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil,
                                           reference_action : ML::Video::HyNomasReferenceAction? = nil,
                                           t0_key_visible_latents : Int32? = nil,
                                           t0_hide_caption_keys : Bool = false,
                                           t1_current_text_kv : Bool = false) : Nil
    if generated_t0_control
      unless source_bf16 && reference_latent_bytes == 0 && segments == 2 &&
             text_cache_policy.once_per_scene? && !cinematic && !prepare_capture &&
             !noise_edit && !prompt_edit && !reference_action &&
             t0_key_visible_latents.nil? && !t0_hide_caption_keys && !t1_current_text_kv
        raise ArgumentError.new("generated T0 source-BF16 control requires two unmasked segments, shared once-per-scene text, no reference latent, and no edits, LoRA, captures, actions, or checkpoint profiles")
      end
      return
    end
    if source_bf16 && (reference_latent_bytes == 0 || segments != 2 || prompt_edit)
      raise ArgumentError.new("Source BF16 requires two-segment reference initialization")
    end
  end

  # Include the temporarily coexisting original attention plan while the
  # prepared provider is attached. Steady residency subtracts that old plan.
  def peak_named_bytes(inputs : HyNomasGeneration::TimelineInputs, segments : Int32,
                       text_cache_policy = HISTORY_TEXT_POLICY, *,
                       prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil,
                       noise_edit : ML::Video::HyNomasSegmentNoiseEdit? = nil,
                       reference_action : ML::Video::HyNomasReferenceAction? = nil,
                       reference_latent_bytes : Int64 = 0_i64,
                       source_bf16 : Bool = false,
                       generated_t0_control : Bool = false,
                       cinematic : HyNomasCinematic::IndexIdentity? = nil,
                       prepare_capture : Bool = false,
                       t0_key_visible_latents : Int32? = nil,
                       t0_hide_caption_keys : Bool = false,
                       t1_current_text_kv : Bool = false) : Int64
    unless reference_latent_bytes == 0 || reference_latent_bytes == REFERENCE_LATENT_BYTES
      raise ArgumentError.new("reference latent residency must match the fixed BF16 DCAE T0 extent")
    end
    validate_source_bf16_initialization!(source_bf16: source_bf16,
      reference_latent_bytes: reference_latent_bytes, segments: segments,
      generated_t0_control: generated_t0_control, text_cache_policy: text_cache_policy,
      cinematic: cinematic, prepare_capture: prepare_capture,
      noise_edit: noise_edit, prompt_edit: prompt_edit, reference_action: reference_action,
      t0_key_visible_latents: t0_key_visible_latents,
      t0_hide_caption_keys: t0_hide_caption_keys, t1_current_text_kv: t1_current_text_kv)
    validate_t0_history_admission!(t0_key_visible_latents, t0_hide_caption_keys, t1_current_text_kv,
      source_bf16: source_bf16, reference_latent_bytes: reference_latent_bytes, segments: segments,
      text_cache_policy: text_cache_policy, cinematic: cinematic,
      prepare_capture: prepare_capture, noise_edit: noise_edit,
      prompt_edit: prompt_edit, reference_action: reference_action)
    if !t0_key_visible_latents.nil? || t0_hide_caption_keys || t1_current_text_kv
      t0_history_attention_profiles(inputs.base.valid_text_tokens, t0_key_visible_latents.not_nil!,
        t0_hide_caption_keys: t0_hide_caption_keys,
        t1_current_text_kv: t1_current_text_kv)
    end
    if source_bf16
      ML::Video::HyNomasTextRefinerMPSPlan.validate_source_bf16_request!(true, inputs.base.valid_text_tokens)
      if action = reference_action
        ML::Video::HyNomasTextRefinerMPSPlan.validate_source_bf16_request!(true, action.valid_text_tokens)
      end
    end
    valid = query_text_tokens(inputs.base, prompt_edit, reference_action: reference_action)
    scratch_growth = HyNomasGeneration.dit_scratch_bytes(valid, ((valid + 63) // 64) * 64) -
                     HyNomasGeneration.dit_scratch_bytes(inputs.base.valid_text_tokens, inputs.base.compact_text_tokens)
    source_scratch = if source_bf16
                       config = dit_config(source_bf16).compact_valid_text_prefix(valid, ((valid + 63) // 64) * 64)
                       # Source BF16 RMS computes only the compact stream;
                       # the ordinary reserve includes two full text reductions.
                       rms_savings = 2_i64 * (HyNomasGeneration::CONFIG.text_tokens - config.text_tokens) *
                                     (config.dim + config.heads) * sizeof(Float32)
                       ML::Video::HyNomasSourceBF16MPSWorkspace.extra_bytes(config) - rms_savings
                     else
                       0_i64
                     end
    HyNomasGeneration.dit_resident_weight_bytes(HyNomasGeneration::DiTWeights::BF16,
      HyNomasGeneration.stage_bytes, HyNomasGeneration.stage_bytes * 54) +
      HyNomasGeneration.dit_reserve_bytes(inputs.base, STEPS) +
      scratch_growth + prompt_edit_bytes(prompt_edit) + reference_action_bytes(reference_action) +
      history_bytes(inputs.base, segments, text_cache_policy, prompt_edit: prompt_edit,
        reference_action: reference_action, source_bf16: source_bf16,
        t0_key_visible_latents: t0_key_visible_latents,
        t0_hide_caption_keys: t0_hide_caption_keys,
        t1_current_text_kv: t1_current_text_kv) +
      additional_inputs_bytes(inputs, segments) + reference_latent_bytes + source_scratch
  end

  def dit_config(source_bf16 : Bool) : ML::Video::HyNomasConfig
    config = HyNomasGeneration::CONFIG
    return config unless source_bf16
    config.copy_with(attention_precision: ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl)
  end

  def checkpoint_inventory(inputs : HyNomasGeneration::TimelineInputs, segments : Int32,
                           next_segment : Int32, prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil) : Hash(String, Int64)
    valid = query_text_tokens(inputs.base, prompt_edit)
    shape = ML::Video::HyNomasContinuationHistory.geometry(validate_segments!(segments),
      4320, valid, ((valid + 63) // 64) * 64, 54, 16, 128,
      text_cache_policy: HISTORY_TEXT_POLICY,
      segment_text_tokens: prompt_edit ? text_tokens(inputs.base, segments, prompt_edit) : nil,
      segment_cache_text: prompt_edit.try(&.cache_text))
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
                          next_segment : Int32, noise_edit : ML::Video::HyNomasSegmentNoiseEdit? = nil,
                          prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil) : String
    checkpoint_inventory(inputs, segments, next_segment, prompt_edit)
    validate_noise_edit!(noise_edit, segments)
    raise ArgumentError.new("Prompt edits cannot be combined with noise edits") if prompt_edit && noise_edit
    raise ArgumentError.new("Prompt edit geometry does not match the sequence") if prompt_edit && prompt_edit.run_segments != segments
    source_hash = Digest::SHA256.new
    {% for path in ["hy_nomas_sequence.cr", "../../src/ml/video/hy_nomas_segment_edit.cr", "../../src/ml/video/hy_nomas_continuation_history.cr", "../../src/ml/video/hy_nomas_timeline_ops.cr", "../../src/ml/video/kernels/hy_nomas_timeline_ops.metal", "../../src/ml/video/hy_nomas_euler.cr", "../../src/ml/video/kernels/hy_nomas_euler.metal", "../../src/ml/video/hy_nomas_block_mps.cr", "../../src/ml/video/hy_nomas_serial_mps.cr", "../../src/ml/video/hy_nomas_history_attention.cr", "../../src/ml/video/hy_nomas_history_attention_bridge.mm", "../../src/ml/video/hy_nomas_mps_bf16_linear_bridge.mm", "../../src/ml/video/hy_nomas_bf16_stack.cr"] %}
      source_hash.update({{ read_file(__DIR__ + "/" + path) }})
    {% end %}
    base = {schema: "cogni_ml.hy_nomas_sequence_boundary/v1", boundary: "after_noisy_writer",
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
    prompt_edit.try(&.bind_contract(base, next_segment)) || noise_edit.try(&.bind_contract(base, next_segment)) || base
  end

  def validate_noise_edit!(edit : ML::Video::HyNomasSegmentNoiseEdit?, segments : Int32) : Nil
    if edit && (edit.run_segments != segments || edit.resident_segments != 6)
      raise ArgumentError.new("Noise edit geometry does not match the admitted sequence")
    end
  end

  def validate_prompt_edit!(edit : ML::Video::HyNomasSegmentPromptEdit?,
                            conditioning : HyNomasGeneration::Inputs?, segments : Int32) : Nil
    unless edit.nil? == conditioning.nil?
      raise ArgumentError.new("Prompt edits require separate authenticated conditioning")
    end
    return unless edit && conditioning
    validate_shared_scene_prompt!(conditioning.prompt)
    unless edit.run_segments == segments && conditioning.geometry == HyNomasGeneration::SpatialGeometry::DEFAULT &&
           conditioning.file.document["schema"].as_s == "cogni_ml.hy_nomas_generation_inputs/v2" &&
           edit.conditioning_manifest_sha256 == conditioning.file.sha256 &&
           edit.conditioning_payload_sha256 == conditioning.file.document["payload_sha256"].as_s &&
           edit.prompt_sha256 == Digest::SHA256.hexdigest(conditioning.prompt) &&
           edit.valid_text_tokens == conditioning.valid_text_tokens
      raise ArgumentError.new("Prompt edit identity does not match its authenticated conditioning")
    end
  end

  # Pure mode gate shared by CLI preflight and the GPU owner. It deliberately
  # does not inspect or change SegmentPromptEdit's checkpoint contract.
  def validate_reference_action_admission!(action : ML::Video::HyNomasReferenceAction, *,
                                           segments : Int32,
                                           reference_latent_bytes : Int64,
                                           text_cache_policy = HISTORY_TEXT_POLICY,
                                           cinematic : HyNomasCinematic::IndexIdentity? = nil,
                                           prepare_capture : Bool = false,
                                           noise_edit : ML::Video::HyNomasSegmentNoiseEdit? = nil,
                                           prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil) : Nil
    unless segments == 2 && action.run_segments == segments
      raise ArgumentError.new("Reference action requires exactly two segments")
    end
    unless reference_latent_bytes == REFERENCE_LATENT_BYTES
      raise ArgumentError.new("Reference action requires the exact real T0 reference latent")
    end
    unless text_cache_policy.once_per_scene?
      raise ArgumentError.new("Reference action requires the OncePerScene text-cache policy")
    end
    if cinematic || prepare_capture || noise_edit || prompt_edit
      raise ArgumentError.new("Reference action does not support Cinematic, capture, noise edits, or prompt edits")
    end
  end

  # Scalar metadata validator so command-line admission can reject incompatible
  # conditioning before creating Metal resources. The Inputs adapter below
  # supplies these fields from its already authenticated manifest.
  def validate_reference_action_identity!(action : ML::Video::HyNomasReferenceAction, *,
                                          conditioning_manifest_sha256 : String,
                                          conditioning_payload_sha256 : String,
                                          conditioning_prompt : String,
                                          conditioning_valid_text_tokens : Int32,
                                          conditioning_geometry : HyNomasGeneration::SpatialGeometry,
                                          conditioning_schema : String,
                                          conditioning_schema_version : Int32) : Nil
    if profile = action.action_conditioning
      profile.validate_binding!(conditioning_manifest_sha256: conditioning_manifest_sha256,
        conditioning_payload_sha256: conditioning_payload_sha256,
        prompt: conditioning_prompt, valid_text_tokens: conditioning_valid_text_tokens)
    else
      validate_shared_scene_prompt!(conditioning_prompt)
    end
    unless conditioning_geometry == HyNomasGeneration::SpatialGeometry::DEFAULT &&
           conditioning_schema == "cogni_ml.hy_nomas_generation_inputs/v2" &&
           conditioning_schema_version == 2 &&
           action.conditioning_manifest_sha256 == conditioning_manifest_sha256 &&
           action.conditioning_payload_sha256 == conditioning_payload_sha256 &&
           action.prompt_sha256 == Digest::SHA256.hexdigest(conditioning_prompt) &&
           action.valid_text_tokens == conditioning_valid_text_tokens
      raise ArgumentError.new("Reference action identity does not match its authenticated v2 default-geometry conditioning")
    end
  end

  def validate_reference_action!(action : ML::Video::HyNomasReferenceAction?,
                                 action_inputs : HyNomasGeneration::Inputs?, *,
                                 segments : Int32,
                                 reference_latent_bytes : Int64,
                                 caption_prompt : String? = nil,
                                 text_cache_policy = HISTORY_TEXT_POLICY,
                                 cinematic : HyNomasCinematic::IndexIdentity? = nil,
                                 prepare_capture : Bool = false,
                                 noise_edit : ML::Video::HyNomasSegmentNoiseEdit? = nil,
                                 prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil) : Nil
    unless action.nil? == action_inputs.nil?
      raise ArgumentError.new("Reference actions require separate authenticated conditioning")
    end
    return unless action && action_inputs
    if profile = action.action_conditioning
      raise ArgumentError.new("Structured reference action requires an authenticated T0 caption") unless caption_prompt
      validate_reference_subject_roster!(caption_prompt, sequential_actions: profile.sequential_actions?)
    end
    validate_reference_action_admission!(action, segments: segments,
      reference_latent_bytes: reference_latent_bytes, text_cache_policy: text_cache_policy,
      cinematic: cinematic, prepare_capture: prepare_capture,
      noise_edit: noise_edit, prompt_edit: prompt_edit)
    doc = action_inputs.file.document
    validate_reference_action_identity!(action,
      conditioning_manifest_sha256: action_inputs.file.sha256,
      conditioning_payload_sha256: doc["payload_sha256"].as_s,
      conditioning_prompt: action_inputs.prompt,
      conditioning_valid_text_tokens: action_inputs.valid_text_tokens,
      conditioning_geometry: action_inputs.geometry,
      conditioning_schema: doc["schema"].as_s,
      conditioning_schema_version: doc["schema_version"].as_i.to_i32)
  end

  class DiT
    getter named_bytes : Int64
    getter peak_named_bytes : Int64
    getter segment_count : Int32
    getter generated_t0_control : Bool
    getter t0_denoise_steps : Int32
    getter t0_cache_writer_forwards : Int32
    getter t0_key_visible_latents : Int32?
    getter t0_hide_caption_keys : Bool
    getter t1_current_text_kv : Bool
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
    @reference_latent : ML::MetalBuffer?
    @generated_t0_control = false
    @t0_denoise_steps = 0
    @t0_cache_writer_forwards = 0
    @latent : ML::MetalBuffer
    @timeline : ML::MetalBuffer
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false
    @timeline_released = false
    @next_segment = 0
    @resume_failed = false
    @capture_copy : ML::Video::HyNomasContinuationHistoryKernels::AppendPlan? = nil
    @prompt_copy : ML::Video::HyNomasContinuationHistoryKernels::AppendPlan? = nil
    @prompt_arena : ML::Video::Arena? = nil
    @prompt_refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
    @checkpoint_contracts = {} of Int32 => String
    @checkpoint_inventories = {} of Int32 => Hash(String, Int64)
    @compact_restore : HyNomasCacheQBitCheckpoint::RestorePlan? = nil
    @cinematic : HyNomasCinematic::IndexIdentity?

    def initialize(inputs : HyNomasGeneration::TimelineInputs, @timeline : ML::MetalBuffer,
                   @latent : ML::MetalBuffer, budget : Int64, @segment_count : Int32,
                   admission : ML::Video::HyNomasQ4StackIndex::Admission,
                   text_cache_policy = HISTORY_TEXT_POLICY, *,
                   cinematic : HyNomasCinematic::IndexIdentity? = nil,
                   prepare_capture : Bool = false,
                   @noise_edit : ML::Video::HyNomasSegmentNoiseEdit? = nil,
                   @prompt_edit : ML::Video::HyNomasSegmentPromptEdit? = nil,
                   prompt_inputs : HyNomasGeneration::Inputs? = nil,
                   @reference_action : ML::Video::HyNomasReferenceAction? = nil,
                   @reference_action_inputs : HyNomasGeneration::Inputs? = nil,
                   reference_latent : ML::MetalBuffer? = nil,
                   source_bf16 : Bool = false,
                   generated_t0_control : Bool = false,
                   @t0_key_visible_latents : Int32? = nil,
                   @t0_hide_caption_keys : Bool = false,
                   @t1_current_text_kv : Bool = false)
      HyNomasSequence.validate_segments!(@segment_count)
      @cinematic = cinematic
      @reference_latent = reference_latent
      @generated_t0_control = generated_t0_control
      if reference = reference_latent
        unless reference.valid? && reference.size == HyNomasSequence::REFERENCE_LATENT_BYTES
          raise ArgumentError.new("reference initialization requires an exact-size caller-owned BF16 DCAE latent")
        end
        if cinematic || prepare_capture || @noise_edit || @prompt_edit || !text_cache_policy.once_per_scene?
          raise ArgumentError.new("reference initialization does not support Cinematic, captures, edits, or legacy text caching")
        end
      end
      HyNomasSequence.validate_source_bf16_initialization!(source_bf16: source_bf16,
        reference_latent_bytes: reference_latent.try(&.size) || 0_i64,
        segments: @segment_count, generated_t0_control: @generated_t0_control,
        text_cache_policy: text_cache_policy, cinematic: cinematic,
        prepare_capture: prepare_capture, noise_edit: @noise_edit, prompt_edit: @prompt_edit,
        reference_action: @reference_action,
        t0_key_visible_latents: @t0_key_visible_latents,
        t0_hide_caption_keys: @t0_hide_caption_keys,
        t1_current_text_kv: @t1_current_text_kv)
      HyNomasSequence.validate_t0_history_admission!(@t0_key_visible_latents, @t0_hide_caption_keys,
        @t1_current_text_kv,
        source_bf16: source_bf16,
        reference_latent_bytes: reference_latent.try(&.size) || 0_i64,
        segments: @segment_count, text_cache_policy: text_cache_policy, cinematic: cinematic,
        prepare_capture: prepare_capture, noise_edit: @noise_edit, prompt_edit: @prompt_edit,
        reference_action: @reference_action)
      if visible_latents = @t0_key_visible_latents
        HyNomasSequence.t0_history_attention_profiles(inputs.base.valid_text_tokens, visible_latents,
          t0_hide_caption_keys: @t0_hide_caption_keys,
          t1_current_text_kv: @t1_current_text_kv)
      end
      HyNomasSequence.validate_checkpoint_variant!(cinematic, prepare_capture: prepare_capture)
      HyNomasSequence.validate_noise_edit!(@noise_edit, @segment_count)
      HyNomasSequence.validate_prompt_edit!(@prompt_edit, prompt_inputs, @segment_count)
      HyNomasSequence.validate_reference_action!(@reference_action, @reference_action_inputs,
        caption_prompt: inputs.base.prompt,
        segments: @segment_count, reference_latent_bytes: reference_latent.try(&.size) || 0_i64,
        text_cache_policy: text_cache_policy, cinematic: cinematic,
        prepare_capture: prepare_capture, noise_edit: @noise_edit, prompt_edit: @prompt_edit)
      if @reference_action && prompt_inputs
        raise ArgumentError.new("Reference actions cannot use segment prompt-edit conditioning")
      end
      raise ArgumentError.new("Prompt edits cannot be combined with noise edits") if @prompt_edit && @noise_edit
      if (@noise_edit || @prompt_edit) && !text_cache_policy.once_per_scene?
        raise ArgumentError.new("Segment edits require the admitted once-per-scene policy")
      end
      if text_cache_policy.once_per_scene? && !cinematic && !@reference_latent && !@generated_t0_control
        (1...@segment_count).each do |next_segment|
          @checkpoint_contracts[next_segment] = HyNomasSequence.checkpoint_contract(inputs, @segment_count, next_segment, @noise_edit, @prompt_edit)
          @checkpoint_inventories[next_segment] = HyNomasSequence.checkpoint_inventory(inputs, @segment_count, next_segment, @prompt_edit)
        end
      end
      base = inputs.base
      valid = HyNomasSequence.query_text_tokens(base, @prompt_edit, reference_action: @reference_action)
      storage = ((valid + 63) // 64) * 64
      @peak_named_bytes = HyNomasSequence.peak_named_bytes(inputs, @segment_count, text_cache_policy,
        prompt_edit: @prompt_edit, reference_action: @reference_action,
        reference_latent_bytes: reference_latent.try(&.size) || 0_i64, source_bf16: source_bf16,
        generated_t0_control: @generated_t0_control,
        cinematic: cinematic, prepare_capture: prepare_capture, noise_edit: @noise_edit,
        t0_key_visible_latents: @t0_key_visible_latents,
        t0_hide_caption_keys: @t0_hide_caption_keys,
        t1_current_text_kv: @t1_current_text_kv)
      @named_bytes = @peak_named_bytes - HyNomasSequence.old_attention_bytes(base, @prompt_edit,
        reference_action: @reference_action)
      raise ArgumentError.new("sequence peak named buffers exceed budget") unless @peak_named_bytes <= budget
      index_sha = HyNomasSequence.weight_index_sha(cinematic)
      admission.validate!(index_sha, HyNomasQ4DiTIO::SOURCE_SHA,
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
        @loader = ML::Video::HyNomasBF16Stack.new(admission, index_sha,
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
        @patch = ML::Video::HyNomasPatchMPSPlan.new(@latent, 4320, @globals, source_bf16: source_bf16)
        plan_cleanup << -> { @patch.release }
        @refiner = prepare_refiner(qwen, base.mask, source_bf16)
        plan_cleanup << -> { @refiner.release }
        if edited = prompt_inputs
          arena = ML::Video::Arena.new
          @prompt_arena = arena
          arena_cleanup << -> { arena.release }
          edited_qwen = edited.upload!("input.qwen_bf16", arena)
          refiner = prepare_refiner(edited_qwen, edited.mask, source_bf16: false)
          @prompt_refiner = refiner
          plan_cleanup << -> { refiner.release }
          @prompt_copy = ML::Video::HyNomasContinuationHistoryKernels::AppendPlan.new
        elsif action_inputs = @reference_action_inputs
          arena = ML::Video::Arena.new
          @prompt_arena = arena
          arena_cleanup << -> { arena.release }
          action_qwen = action_inputs.upload!("input.qwen_bf16", arena)
          refiner = prepare_refiner(action_qwen, action_inputs.mask, source_bf16)
          @prompt_refiner = refiner
          plan_cleanup << -> { refiner.release }
          @prompt_copy = ML::Video::HyNomasContinuationHistoryKernels::AppendPlan.new
        end
        @serial = ML::Video::HyNomasMPSSerialWorkspace.new(HyNomasSequence.dit_config(source_bf16),
          Array(ML::Video::Arena).new(54) { @stage }, @patch.output, @refiner.text_output,
          @refiner.vec, @refiner.vec_txt, cos, sin, attention_query_chunk: 256,
          attention_heads_per_group: 16, valid_text_tokens: valid,
          allow_shared_modulation: true, full_dit: true, weight_loader: @loader,
          compact_valid_text: true, compact_text_tokens: storage,
          workspace_factory: source_bf16 ? ML::Video::HyNomasSourceBF16MPSWorkspace.factory : nil)
        plan_cleanup << -> { @serial.release }
        blocks = @serial.blocks
        joint = blocks.first.checkpoints
        history_text = HyNomasSequence.history_text_profile(base, @segment_count, @prompt_edit,
          reference_action: @reference_action, t1_current_text_kv: @t1_current_text_kv)
        @history = ML::Video::HyNomasContinuationHistory.new(segment_count: @segment_count,
          video_tokens: 4320, valid_text_tokens: valid,
          storage_text_tokens: storage, layers: 54, heads: 16, head_dim: 128,
          text_cache_policy: text_cache_policy,
          reference_initialization: !@reference_latent.nil?,
          segment_text_tokens: history_text[:segment_text_tokens],
          segment_cache_text: history_text[:segment_cache_text],
          segment_attention_labels: HyNomasSequence.reference_attention_labels(base.valid_text_tokens, @reference_action),
          segment_attention_profiles: if visible_latents = @t0_key_visible_latents
            HyNomasSequence.t0_history_attention_profiles(base.valid_text_tokens, visible_latents,
              t0_hide_caption_keys: @t0_hide_caption_keys,
              t1_current_text_kv: @t1_current_text_kv)
          else
            nil
          end,
          source_bf16_attention: source_bf16,
          query: joint["joint.query"], key: joint["joint.key"],
          value: joint["joint.value"], output: joint["joint.attention"])
        plan_cleanup << -> { @history.release }
        @serial.attach_history_provider!(@history)
        @final = ML::Video::HyNomasFinalMPSPlan.new(blocks.last.image_output, @refiner.vec, 4320, @globals,
          source_bf16: source_bf16, source_autocast: source_bf16)
        plan_cleanup << -> { @final.release }
        @segment_count.times do
          @eulers << ML::Video::HyNomasEulerPlan.new(@latent, @final.output,
            @refiner.checkpoints["timestep"], steps: STEPS, shift: 5_f32)
        end
        @timeline_ops = ML::Video::HyNomasTimelineOps::Plan.new(ML::Video::HyNomasTimelineOps::Geometry.new,
          timeline: @timeline, segment_latent: @latent, cosine_bank: cos_bank, sine_bank: sin_bank,
          cosine: cos, sine: sin, cache_eps_bf16: eps,
          cache_timestep: @refiner.checkpoints["timestep"], run_segments: @segment_count,
          reference_initialization: !@reference_latent.nil?)
        @capture_copy = ML::Video::HyNomasContinuationHistoryKernels::AppendPlan.new if prepare_capture
        actual = @loader.cache_payload_bytes + @stage.resident_bytes + @globals.resident_bytes +
                 @inputs_arena.resident_bytes + @bank_arena.resident_bytes + @latent.size + @timeline.size + @patch.resident_bytes +
                 (@reference_latent.try(&.size) || 0_i64) +
                 @refiner.resident_bytes + @serial.resident_bytes + @final.resident_bytes +
                 @eulers.sum(&.resident_bytes) + (@prompt_arena.try(&.resident_bytes) || 0_i64) +
                 (@prompt_refiner.try(&.resident_bytes) || 0_i64)
        unless actual == @named_bytes
          raise ArgumentError.new("sequence allocation/admission mismatch: #{actual} != #{@named_bytes}")
        end
        HyNomasGeneration.emit("sequence_prepared", named_bytes: actual, segments: @segment_count,
          generated_t0_control: @generated_t0_control,
          history_precision: source_bf16 ? "F32 storage of source-BF16 post-RoPE K/V" : "F32 exact native cache boundary",
          t0_key_visible_latents: @t0_key_visible_latents,
          t0_hide_caption_keys: @t0_hide_caption_keys,
          t1_current_text_kv: @t1_current_text_kv,
          t1_current_text_key_rows: @t1_current_text_kv ? base.valid_text_tokens : 0,
          t0_writer_attention: @t0_key_visible_latents ? "unmasked" : nil,
          t1_attention_profile: if @t1_current_text_kv
            if @t0_key_visible_latents == 0
              "all T0 video and caption keys masked; current T1 text K/V included"
            else
              "T0 caption mask; current T1 text K/V included"
            end
          elsif @t0_key_visible_latents
            "same-size masked workspace"
          else
            nil
          end)
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
      if @reference_latent
        if stop_after_segments <= 1 || !capture_segment.nil? || !captures.nil? ||
           !capture_steps.nil? || !after_boundary.nil?
          raise ArgumentError.new("reference initialization requires generated T1 and does not support captures or boundary callbacks")
        end
      end
      if @generated_t0_control && (stop_after_segments != @segment_count || !capture_segment.nil? ||
         !captures.nil? || !capture_steps.nil? || !after_boundary.nil?)
        raise ArgumentError.new("Generated T0 source-BF16 control requires the complete uncheckpointed two-segment run")
      end
      if edit = @prompt_edit
        if @next_segment < edit.target_segment
          raise ArgumentError.new("Prompt edits must restore their preserved prefix before encode")
        end
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
      if reference = @reference_latent
        @timeline_ops.admit_reference_latent!(chain, reference)
        @timeline_ops.prepare_reference_segment!(chain, 0)
        @timeline_ops.write_cache_noise!(chain, 0)
        @history.select_reference_init!(0)
        @patch.encode(chain)
        encode_refiner(chain, 0)
        @serial.encode(chain)
        @timeline_ops.record_cache_forward_enqueued!(chain, 0)
        @t0_cache_writer_forwards += 1
        @timeline_ops.advance_segment!(chain, 0)
        @next_segment = 1
        HyNomasGeneration.emit("sequence_reference_initialized",
          technical_reference_segment_index: 0, technical_reference_label: "T0",
          next_generated_segment_index: 1, next_generated_label: "T1",
          clean_global_latent_stop: 30, technical_reference_cache_writer: true)
      end
      (@next_segment...stop_after_segments).each do |segment|
        selected_captures = capture_steps.try(&.[segment]?)
        noise_source = @noise_edit.try(&.source_for(segment)) || segment
        @timeline_ops.prepare_segment!(chain, segment, noise_source_segment: noise_source)
        @eulers[segment].step_count.times do |step|
          @eulers[segment].encode_timestep(chain, step)
          @patch.encode(chain)
          encode_refiner(chain, segment)
          @history.select_segment!(segment, writer: false)
          @serial.encode(chain)
          @final.encode(chain)
          @eulers[segment].encode_update(chain, step)
          @t0_denoise_steps += 1 if segment == 0
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
          encode_refiner(chain, segment)
          @serial.encode(chain)
          @timeline_ops.record_cache_forward_enqueued!(chain, segment)
          @t0_cache_writer_forwards += 1 if segment == 0
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

    private def prepare_refiner(qwen : ML::MetalBuffer, mask : Array(Int32), source_bf16 : Bool) : ML::Video::HyNomasTextRefinerMPSPlan
      # Source-BF16 is an explicit precision request. Reject an unqualified
      # prefix instead of silently running its refiner with F32 boundaries.
      valid = ML::Video::HyNomasTextRefinerMPSPlan.valid_tokens(mask)
      ML::Video::HyNomasTextRefinerMPSPlan.validate_source_bf16_request!(source_bf16, valid)
      bytes = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask, retain_padding: true).total
      enabled = source_bf16
      ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, mask, 1000_f32, @globals, bytes,
        retain_padding: true, prelude_input_bf16_control: enabled,
        prelude_linear_bf16_control: enabled, prelude_silu_bf16_control: enabled,
        prelude_second_linear_bf16_control: enabled, prelude_add_bf16_control: enabled,
        block_boundaries_bf16_control: enabled, qkv_bf16_control: enabled,
        source_attention_control: enabled, source_projection_residual_control: enabled,
        source_norm2_control: enabled, source_fc1_control: enabled,
        source_mlp_silu_control: enabled, source_fc2_residual_control: enabled,
        source_mean_autocast: enabled)
    end

    # Keep serial/final bindings stable. The edited refiner has its own mask,
    # but transfers its results directly into the common GPU-owned input slots.
    # No host observation or command-chain boundary is required to change text.
    private def encode_refiner(chain : ML::Metal::MPSCommandChain, segment : Int32) : Nil
      if @reference_action && segment == 1
        refiner = @prompt_refiner.not_nil!
        copy = @prompt_copy.not_nil!
        copy.encode(chain.encoder, @refiner.checkpoints["timestep"], refiner.checkpoints["timestep"], 1, 0)
        refiner.encode(chain)
        copy.encode(chain.encoder, refiner.text_output, @refiner.text_output,
          (refiner.text_output.size // sizeof(Float32)).to_i32, 0)
        # vec_txt is the same stable buffer as vec; copying it once updates both
        # serial inputs without changing their setup-time bindings.
        copy.encode(chain.encoder, refiner.vec, @refiner.vec,
          (refiner.vec.size // sizeof(Float32)).to_i32, 0)
      elsif @prompt_edit.try(&.edited_segment?(segment))
        refiner = @prompt_refiner.not_nil!
        copy = @prompt_copy.not_nil!
        copy.encode(chain.encoder, @refiner.checkpoints["timestep"], refiner.checkpoints["timestep"], 1, 0)
        refiner.encode(chain)
        copy.encode(chain.encoder, refiner.text_output, @refiner.text_output,
          (refiner.text_output.size // sizeof(Float32)).to_i32, 0)
        copy.encode(chain.encoder, refiner.vec, @refiner.vec,
          (refiner.vec.size // sizeof(Float32)).to_i32, 0)
      else
        @refiner.encode(chain)
      end
    end

    def prepare_checkpoint_bindings!(next_segment : Int32) : Hash(String, Tuple(ML::MetalBuffer, Int32))
      reject_reference_checkpoint!
      HyNomasSequence.validate_checkpoint_variant!(@cinematic, checkpoint: true)
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
      reject_reference_checkpoint!
      HyNomasSequence.validate_checkpoint_variant!(@cinematic, restore: true)
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
      reject_reference_checkpoint!
      HyNomasSequence.validate_checkpoint_variant!(@cinematic, checkpoint: true)
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
      reject_reference_checkpoint!
      HyNomasSequence.validate_checkpoint_variant!(@cinematic, restore: true)
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

    private def reject_reference_checkpoint! : Nil
      if @generated_t0_control
        raise ArgumentError.new("Generated T0 source-BF16 control does not support checkpoint or restore")
      end
      if @reference_latent
        raise ArgumentError.new("Reference-initialized sequences do not support checkpoint or restore")
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
      @prompt_refiner.try { |refiner| actions << -> { refiner.release } }
      actions << -> { @refiner.release }
      actions << -> { @patch.release }
      actions << -> { @loader.release }
      actions << -> { @prepared.release }
      actions << -> { @stage.release }
      actions << -> { @globals.release }
      actions << -> { @inputs_arena.release }
      actions << -> { @bank_arena.release }
      @prompt_arena.try { |arena| actions << -> { arena.release } }
      actions << -> { @file.close }
      failures = HyNomasSequence.release_actions(actions)
      unless failures.empty?
        raise ArgumentError.new("sequence cleanup failed: #{failures.join("; ")}")
      end
      @reference_latent = nil
      @released = true
    end
  end
end
