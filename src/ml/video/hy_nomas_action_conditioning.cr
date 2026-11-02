require "digest/sha256"
require "json"

module ML::Video::HyNomasActionConditioning
  PROFILE_SCHEMA        = "cogni_ml.hy_nomas_action_conditioning/v1"
  PROMPT_SCHEMA         = "cogni_ml.hy_nomas_qwen25_video_prompt_reference"
  UPSTREAM_REVISION     = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
  TEMPLATE_NAME         = "li-dit-encode-video-json"
  TEMPLATE_SHA256       = "3e7b69bdf2a057bf2d6a8322dfb8ad619e686e4e8240b65f29e94492bee0c4a6"
  TOKENIZER_REPO        = "Qwen/Qwen2.5-VL-7B-Instruct"
  TOKENIZER_REVISION    = "eb6ba55b40ada64977147b52e50a4145ee89ac4f"
  TEXT_LENGTH           = 1280
  MAX_VALID_TEXT_TOKENS = 1000
  SEGMENT_LATENT_FRAMES =   30
  TOKENS_PER_FRAME      =  144
  VIDEO_TOKENS          = SEGMENT_LATENT_FRAMES * TOKENS_PER_FRAME

  EXPECTED_SOURCE_SHA256 = {
    "hymm/pipeline/base.py"               => "77a039514ff494cb496a194c60ca3270f09c15b194bda25448a036a6fcc79582",
    "hymm/inference.py"                   => "7b0a8152c9ab5c1682dc7e4aee41d7fb3b2f3416873b3b4a6795a45e65b82e6d",
    "hymm/models/text_encoder/encoder.py" => "aa84b366d6e0a5cf9c73e591e6827746e8cab97852a112d1b4783dc7c2d507f1",
    "hymm/models/dit/nomas_cpt.py"        => "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee",
    "configs/constants.py"                => "b4ab0055d0a2fade687f012f0e305915f5305df58dc82e2ad8bde25ef4236af9",
  }

  record LabelPair, query_labels : Array(Int32), key_labels : Array(Int32)

  struct Profile
    getter profile_sha256 : String
    getter conditioning_manifest_sha256 : String
    getter conditioning_payload_sha256 : String
    getter prompt_sha256 : String
    getter valid_text_tokens : Int32
    getter head_end : Int32
    getter action_start : Int32
    getter action_end : Int32
    getter source_offsets_raw : Array(Int32)
    getter effective_offsets : Array(Int32)
    getter action_start_seconds : Float64
    getter action_end_seconds : Float64
    getter latent_start : Int32
    getter latent_end : Int32
    getter intra_segment_active : Bool
    getter video_windows : Array(Tuple(Int32, Int32, Int32))?

    private def initialize(@profile_sha256 : String,
                           @conditioning_manifest_sha256 : String,
                           @conditioning_payload_sha256 : String,
                           @prompt_sha256 : String,
                           @valid_text_tokens : Int32,
                           @head_end : Int32,
                           @action_start : Int32,
                           @action_end : Int32,
                           @source_offsets_raw : Array(Int32),
                           @effective_offsets : Array(Int32),
                           @action_start_seconds : Float64,
                           @action_end_seconds : Float64,
                           @latent_start : Int32,
                           @latent_end : Int32,
                           @intra_segment_active : Bool,
                           @video_windows : Array(Tuple(Int32, Int32, Int32))?)
    end

    def self.from_file(path : String, sha256 : String,
                       conditioning_manifest_sha256 : String,
                       conditioning_payload_sha256 : String,
                       prompt : String,
                       valid_text_tokens : Int32) : Profile
      unless valid_sha?(sha256) && valid_sha?(conditioning_manifest_sha256) && valid_sha?(conditioning_payload_sha256)
        raise ArgumentError.new("Hy-Nomas action profile binding requires lowercase SHA-256 values")
      end
      raw = File.read(path)
      actual_sha = Digest::SHA256.hexdigest(raw)
      unless actual_sha == sha256
        raise ArgumentError.new("Hy-Nomas action profile file SHA-256 does not match")
      end
      begin
        document = JSON.parse(raw)
        validate_document!(document, sha256, conditioning_manifest_sha256,
          conditioning_payload_sha256, prompt, valid_text_tokens)
      rescue ex : JSON::ParseException | KeyError | TypeCastError | IndexError
        raise ArgumentError.new("Malformed Hy-Nomas action profile: #{ex.message}")
      end
    end

    def validate_binding!(conditioning_manifest_sha256 : String,
                          conditioning_payload_sha256 : String,
                          prompt : String,
                          valid_text_tokens : Int32) : Nil
      unless @conditioning_manifest_sha256 == conditioning_manifest_sha256 &&
             @conditioning_payload_sha256 == conditioning_payload_sha256 &&
             @prompt_sha256 == Digest::SHA256.hexdigest(prompt) &&
             @valid_text_tokens == valid_text_tokens
        raise ArgumentError.new("Hy-Nomas action profile does not match its conditioning binding")
      end
    end

    def identity
      {
        schema:                       PROFILE_SCHEMA,
        profile_sha256:               @profile_sha256,
        conditioning_manifest_sha256: @conditioning_manifest_sha256,
        conditioning_payload_sha256:  @conditioning_payload_sha256,
        prompt_sha256:                @prompt_sha256,
        valid_text_tokens:            @valid_text_tokens,
        action_id:                    1,
        action_start_seconds:         @action_start_seconds,
        action_end_seconds:           @action_end_seconds,
        head_end:                     @head_end,
        action_start:                 @action_start,
        action_end:                   @action_end,
        source_offsets_raw:           @source_offsets_raw,
        effective_offsets:            @effective_offsets,
        latent_start:                 @latent_start,
        latent_end:                   @latent_end,
        intra_segment_active:         @intra_segment_active,
      }
    end

    # Query rows are current video rows followed by the padded text query. Keys
    # are global history rows, current video rows, and optionally valid T1 text.
    def labels(video_tokens : Int32, query_text_tokens : Int32,
               history_rows : Int32, current_text_tokens : Int32) : LabelPair
      unless video_tokens == VIDEO_TOKENS && query_text_tokens >= @valid_text_tokens &&
             query_text_tokens <= MAX_VALID_TEXT_TOKENS && history_rows >= 0 &&
             (current_text_tokens == 0 || current_text_tokens == @valid_text_tokens)
        raise ArgumentError.new("Hy-Nomas action labels do not match bounded T1 attention geometry")
      end
      if history_rows.to_i64 + video_tokens + current_text_tokens > Int32::MAX
        raise ArgumentError.new("Hy-Nomas action key row count exceeds native limits")
      end

      video_labels = Array(Int32).new(video_tokens, 0)
      if @intra_segment_active
        first = @latent_start * TOKENS_PER_FRAME
        last = @latent_end * TOKENS_PER_FRAME
        video_labels[first...last] = Array(Int32).new(last - first, 1)
      end

      text_query_labels = Array(Int32).new(query_text_tokens, -1)
      @valid_text_tokens.times do |index|
        text_query_labels[index] = if @intra_segment_active && index >= @action_start && index < @action_end
                                     1
                                   else
                                     0
                                   end
      end
      query_labels = video_labels + text_query_labels

      # Upstream's action ranges restrict action-text keys. Video keys remain
      # globally visible even when their video queries belong to an action span.
      key_labels = Array(Int32).new(history_rows + video_tokens, 0)
      if current_text_tokens > 0
        key_labels.concat(text_query_labels[0, current_text_tokens])
      end
      LabelPair.new(query_labels, key_labels)
    end

    private def self.validate_document!(document : JSON::Any,
                                        profile_sha256 : String,
                                        expected_manifest_sha256 : String,
                                        expected_payload_sha256 : String,
                                        expected_prompt : String,
                                        expected_valid_text_tokens : Int32) : Profile
      root = document.as_h
      unless string(root, "schema") == PROFILE_SCHEMA
        raise ArgumentError.new("Unsupported Hy-Nomas action profile schema")
      end
      manifest_sha = string(root, "conditioning_manifest_sha256")
      payload_sha = string(root, "conditioning_payload_sha256")
      prompt_sha = string(root, "prompt_sha256")
      prompt = string(root, "prompt")
      valid = integer(root, "valid_text_tokens")
      unless valid_sha?(manifest_sha) && valid_sha?(payload_sha) && valid_sha?(prompt_sha) &&
             manifest_sha == expected_manifest_sha256 && payload_sha == expected_payload_sha256 &&
             prompt == expected_prompt && prompt_sha == Digest::SHA256.hexdigest(prompt) &&
             valid == expected_valid_text_tokens && 1 <= valid <= MAX_VALID_TEXT_TOKENS
        raise ArgumentError.new("Hy-Nomas action profile conditioning identity does not match")
      end

      geometry = hash(root, "geometry")
      unless integer(geometry, "run_segments") == 2 && integer(geometry, "segment_index") == 1 &&
             number(geometry, "segment_start_seconds") == 5.0 && number(geometry, "segment_end_seconds") == 10.0 &&
             integer(geometry, "latent_frames") == SEGMENT_LATENT_FRAMES &&
             integer(geometry, "tokens_per_frame") == TOKENS_PER_FRAME &&
             integer(geometry, "video_tokens") == VIDEO_TOKENS && integer(geometry, "text_length") == TEXT_LENGTH
        raise ArgumentError.new("Hy-Nomas action profile exceeds the fixed two-segment geometry")
      end

      token_contract = hash(root, "token_contract")
      validate_token_contract!(token_contract, prompt_sha, valid)
      source = hash(root, "source")
      validate_source!(source)
      action = hash(root, "action")
      action_start_seconds = number(action, "start_time_local_seconds")
      action_end_seconds = number(action, "end_time_local_seconds")
      unless integer(action, "action_id") == 1 && string(action, "subject_id") == "<|Subject1|>" &&
             string(action, "time_origin") == "segment_local_seconds" &&
             number(action, "source_segment_start_seconds") == 5.0 &&
             number(action, "source_segment_end_seconds") == 10.0 &&
             0.0 <= action_start_seconds < action_end_seconds <= 5.0
        raise ArgumentError.new("Malformed Hy-Nomas Action1 metadata or wrong time origin")
      end
      validate_prompt_action!(prompt, action_start_seconds, action_end_seconds,
        string(action, "action_description"))

      offsets = hash(root, "text_offsets")
      head_end = integer(offsets, "head_end")
      action_start = integer(offsets, "action_start")
      action_end = integer(offsets, "action_end")
      source_offsets_raw = integer_array(offsets, "source_offsets_raw")
      effective_offsets = integer_array(offsets, "effective_offsets")
      unless integer(offsets, "valid_text_tokens") == valid && head_end == action_start &&
             0 < head_end < action_end && action_end == valid &&
             source_offsets_raw.size == 2 && effective_offsets == [head_end, action_end] &&
             source_offsets_raw.all? { |offset| 0 <= offset <= TEXT_LENGTH } &&
             effective_offsets == source_offsets_raw.map { |offset| offset < valid ? offset : valid } &&
             string(offsets, "offset_method_endpoint_note") ==
               "BasePipeline returns the raw attention-mask sum; pinned nomas_cpt clamps it with min(offset, valid current text query tokens), yielding the authenticated cropped-mask count"
        raise ArgumentError.new("Hy-Nomas action text offsets have invalid extents or coverage")
      end

      windows_json = field(root, "video_windows")
      active = boolean(root, "intra_segment_active")
      windows, latent_start, latent_end = validate_windows!(
        windows_json, active, action_start_seconds, action_end_seconds, action
      )
      labels = hash(root, "labels")
      unless integer(labels, "global") == 0 && integer(labels, "action") == 1 &&
             integer(labels, "padding") == -1 && integer(labels, "video_gap_window_index") == -1 &&
             string(labels, "query_scope") == "global_head_and_matching_action_window" &&
             string(labels, "key_scope") == "global_history_or_matching_action"
        raise ArgumentError.new("Hy-Nomas action label semantics do not match the native mask contract")
      end

      new(profile_sha256, manifest_sha, payload_sha, prompt_sha, valid,
        head_end, action_start, action_end, source_offsets_raw, effective_offsets,
        action_start_seconds, action_end_seconds,
        latent_start, latent_end, active, windows)
    end

    private def self.validate_token_contract!(contract : Hash(String, JSON::Any),
                                              prompt_sha : String,
                                              valid : Int32) : Nil
      crop = integer(contract, "crop_start")
      unless string(contract, "schema") == PROMPT_SCHEMA && integer(contract, "schema_version") == 1 &&
             string(contract, "prompt_sha256") == prompt_sha && crop > 0 && crop <= 256 &&
             integer(contract, "text_length") == TEXT_LENGTH &&
             string(contract, "template_name") == TEMPLATE_NAME &&
             string(contract, "template_sha256") == TEMPLATE_SHA256 &&
             integer(contract, "valid_text_tokens") == valid
        raise ArgumentError.new("Hy-Nomas action profile Qwen token contract is incompatible")
      end
      {"raw_token_ids_sha256", "raw_attention_mask_sha256", "cropped_token_ids_sha256", "cropped_attention_mask_sha256"}.each do |name|
        unless valid_sha?(string(contract, name))
          raise ArgumentError.new("Hy-Nomas action profile has malformed token contract digest")
        end
      end
      assets = hash(contract, "tokenizer_assets")
      {"config.json", "tokenizer.json", "tokenizer_config.json"}.each do |name|
        record = hash(assets, name)
        unless integer(record, "nbytes") > 0 && valid_sha?(string(record, "sha256"))
          raise ArgumentError.new("Hy-Nomas action profile tokenizer asset digest is malformed")
        end
      end
      tokenizer = hash(contract, "tokenizer_identity")
      unless string(tokenizer, "repo_id") == TOKENIZER_REPO &&
             string(tokenizer, "revision") == TOKENIZER_REVISION &&
             !string(tokenizer, "transformers_version").empty? &&
             !string(tokenizer, "tokenizers_version").empty?
        raise ArgumentError.new("Hy-Nomas action profile tokenizer identity is incompatible")
      end
    end

    private def self.validate_source!(source : Hash(String, JSON::Any)) : Nil
      tokenizer_methods = field(source, "text_tokenizer_methods").as_a.map(&.as_s)
      unless string(source, "upstream_revision") == UPSTREAM_REVISION &&
             string(source, "offset_method") == "BasePipeline._compute_action_text_offsets (AST-extracted from pinned base.py)" &&
             string(source, "video_window_method") == "BasePipeline._compute_action_video_windows (AST-extracted from pinned base.py)" &&
             string(source, "offset_clamp_method") == "nomas_cpt.py clamps every source offset with min(offset, valid_current_text_query_tokens)" &&
             tokenizer_methods == ["TextEncoder.apply_text_to_template", "TextEncoder.text2tokens"] &&
             string(source, "time_normalizer") == "hymm.inference._normalize_contained_action_timestamps (AST-extracted)" &&
             string(source, "time_origin") == "segment_local_seconds" &&
             boolean(source, "resolve_subject_ids") == false &&
             string(source, "action_wrapper_representation") == "literal_text_markup_not_token_ids"
        raise ArgumentError.new("Hy-Nomas action profile source and prompt shaping contract is invalid")
      end
      actual_sources = hash(source, "source_sha256")
      EXPECTED_SOURCE_SHA256.each do |path, expected|
        unless string(actual_sources, path) == expected
          raise ArgumentError.new("Hy-Nomas action profile upstream source digest differs: #{path}")
        end
      end
    end

    private def self.validate_prompt_action!(prompt : String,
                                             expected_start : Float64,
                                             expected_end : Float64,
                                             expected_description : String) : Nil
      parsed = JSON.parse(prompt).as_h
      segments = parsed["segments"].as_a
      unless segments.size == 1
        raise ArgumentError.new("Hy-Nomas action prompt must contain one T1 segment")
      end
      segment = segments[0].as_h
      unless timestamp_seconds(segment["start_time"].as_s) == 5.0 &&
             timestamp_seconds(segment["end_time"].as_s) == 10.0
        raise ArgumentError.new("Hy-Nomas action prompt has the wrong global segment time origin")
      end
      actions = segment["contained_actions"].as_a
      unless actions.size == 1
        raise ArgumentError.new("Hy-Nomas action prompt must contain exactly one action")
      end
      action = actions[0].as_h
      unless action.keys.sort == ["action_description", "end_time", "start_time"]
        raise ArgumentError.new("Hy-Nomas Action1 prompt fields are malformed")
      end
      start_time = action["start_time"].as_s
      end_time = action["end_time"].as_s
      description = action["action_description"].as_s
      unless timestamp_seconds(start_time) == expected_start && timestamp_seconds(end_time) == expected_end &&
             description == expected_description && description.matches?(/\A<Action1>.*<\/Action1>\.\z/) &&
             description.scan(/<Action/).size == 1 && description.scan(/<\/Action/).size == 1 &&
             description.includes?("<|Subject1|>") && !description.includes?("<|Subject2|>") &&
             description.downcase.includes?("head") && description.downcase.matches?(/\bturn(ing|s|ed)?\b/) &&
             prompt.includes?("Resume")
        raise ArgumentError.new("Hy-Nomas Action1 wrapper, subject ID, time, or Resume context is malformed")
      end
    end

    private def self.validate_windows!(value : JSON::Any,
                                       active : Bool,
                                       action_start_seconds : Float64,
                                       action_end_seconds : Float64,
                                       action : Hash(String, JSON::Any)) : Tuple(Array(Tuple(Int32, Int32, Int32))?, Int32, Int32)
      if !active
        unless value.raw.nil? && action_start_seconds == 0.0 && action_end_seconds == 5.0 &&
               integer(action, "latent_start") == 0 && integer(action, "latent_end") == SEGMENT_LATENT_FRAMES
          raise ArgumentError.new("Legacy full-segment action must be explicitly unmasked")
        end
        return {nil, 0, SEGMENT_LATENT_FRAMES}
      end
      windows_json = value.as_a
      unless windows_json.size.in?(2..3)
        raise ArgumentError.new("Active bounded action requires gap and action windows")
      end
      windows = [] of Tuple(Int32, Int32, Int32)
      cursor = 0_i32
      action_windows = [] of Tuple(Int32, Int32, Int32)
      windows_json.each do |raw_window|
        entries = raw_window.as_a
        unless entries.size == 3
          raise ArgumentError.new("Hy-Nomas action window must contain three integers")
        end
        start_frame = entries[0].as_i.to_i32
        end_frame = entries[1].as_i.to_i32
        action_index = entries[2].as_i.to_i32
        unless start_frame == cursor && end_frame > start_frame && end_frame <= SEGMENT_LATENT_FRAMES &&
               action_index.in?(-1..0)
          raise ArgumentError.new("Hy-Nomas action windows overlap, leave a gap, or exceed the segment")
        end
        window = {start_frame, end_frame, action_index}
        windows << window
        action_windows << window if action_index == 0
        cursor = end_frame
      end
      unless cursor == SEGMENT_LATENT_FRAMES && action_windows.size == 1
        raise ArgumentError.new("Hy-Nomas action windows must tile the segment with exactly one action")
      end
      active_window = action_windows.first
      expected_start = (action_start_seconds * 24).to_i32 // 4
      expected_end = (action_end_seconds * 24).to_i32 // 4
      unless active_window[0] == expected_start && active_window[1] == expected_end &&
             (active_window[0] > 0 || active_window[1] < SEGMENT_LATENT_FRAMES) &&
             integer(action, "latent_start") == active_window[0] &&
             integer(action, "latent_end") == active_window[1]
        raise ArgumentError.new("Hy-Nomas action video window does not match normalized timestamps")
      end
      {windows, active_window[0], active_window[1]}
    end

    private def self.timestamp_seconds(value : String) : Float64
      match = value.match(/\A(\d+):(\d{2}\.\d{3})\z/)
      unless match
        raise ArgumentError.new("Hy-Nomas action timestamp must be canonical MM:SS.mmm text")
      end
      match[1].to_i * 60 + match[2].to_f
    end

    private def self.valid_sha?(value : String) : Bool
      value.matches?(/\A[0-9a-f]{64}\z/)
    end

    private def self.field(object : Hash(String, JSON::Any), name : String) : JSON::Any
      object.fetch(name) { raise KeyError.new("missing JSON field: #{name}") }
    end

    private def self.string(object : Hash(String, JSON::Any), name : String) : String
      field(object, name).as_s
    end

    private def self.integer(object : Hash(String, JSON::Any), name : String) : Int32
      value = field(object, name).as_i64
      unless Int32::MIN <= value <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas action metadata integer is outside native limits")
      end
      value.to_i32
    end

    private def self.integer_array(object : Hash(String, JSON::Any), name : String) : Array(Int32)
      values = field(object, name).as_a
      values.map do |value|
        integer_value = value.as_i64
        unless Int32::MIN <= integer_value <= Int32::MAX
          raise ArgumentError.new("Hy-Nomas action offset is outside native limits")
        end
        integer_value.to_i32
      end
    end

    private def self.number(object : Hash(String, JSON::Any), name : String) : Float64
      value = field(object, name)
      value.raw.is_a?(Int64) ? value.as_i64.to_f64 : value.as_f
    end

    private def self.boolean(object : Hash(String, JSON::Any), name : String) : Bool
      field(object, name).as_bool
    end

    private def self.hash(object : Hash(String, JSON::Any), name : String) : Hash(String, JSON::Any)
      field(object, name).as_h
    end
  end
end
