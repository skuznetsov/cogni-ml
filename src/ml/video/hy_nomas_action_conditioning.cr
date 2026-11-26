require "digest/sha256"
require "json"

module ML::Video::HyNomasActionConditioning
  PROFILE_SCHEMA        = "cogni_ml.hy_nomas_action_conditioning/v1"
  PROFILE_SCHEMA_V2     = "cogni_ml.hy_nomas_action_conditioning/v2"
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

  record Action,
    action_id : Int32,
    subject_id : String,
    start_time_local_seconds : Float64,
    end_time_local_seconds : Float64,
    action_description : String,
    time_origin : String,
    source_segment_start_seconds : Float64,
    source_segment_end_seconds : Float64,
    latent_start : Int32,
    latent_end : Int32 do
    def identity
      {
        action_id:                    @action_id,
        subject_id:                   @subject_id,
        start_time_local_seconds:     @start_time_local_seconds,
        end_time_local_seconds:       @end_time_local_seconds,
        action_description:           @action_description,
        time_origin:                  @time_origin,
        source_segment_start_seconds: @source_segment_start_seconds,
        source_segment_end_seconds:   @source_segment_end_seconds,
        latent_start:                 @latent_start,
        latent_end:                   @latent_end,
      }
    end
  end

  struct Profile
    getter schema : String
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
    getter actions : Array(Action)

    private def initialize(@schema : String,
                           @profile_sha256 : String,
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
                           @video_windows : Array(Tuple(Int32, Int32, Int32))?,
                           @actions : Array(Action))
    end

    def sequential_actions? : Bool
      @schema == PROFILE_SCHEMA_V2
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
      if sequential_actions?
        {
          schema:                       @schema,
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
          actions:                      @actions.map(&.identity),
          video_windows:                @video_windows,
        }
      else
        # Preserve the exact v1 tuple and serialization; its action fields
        # remain the scalar Action1 compatibility view used above for v2.
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
        if sequential_actions?
          @video_windows.not_nil!.each do |start_frame, end_frame, action_index|
            next if action_index < 0
            first = start_frame * TOKENS_PER_FRAME
            last = end_frame * TOKENS_PER_FRAME
            video_labels[first...last] = Array(Int32).new(last - first, action_index + 1)
          end
        else
          first = @latent_start * TOKENS_PER_FRAME
          last = @latent_end * TOKENS_PER_FRAME
          video_labels[first...last] = Array(Int32).new(last - first, 1)
        end
      end

      text_query_labels = Array(Int32).new(query_text_tokens, -1)
      @valid_text_tokens.times do |index|
        if sequential_actions?
          text_query_labels[index] = if index < @effective_offsets[0]
                                       0
                                     elsif index < @effective_offsets[1]
                                       1
                                     else
                                       2
                                     end
        else
          text_query_labels[index] = if @intra_segment_active && index >= @action_start && index < @action_end
                                       1
                                     else
                                       0
                                     end
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
      schema = string(root, "schema")
      unless schema == PROFILE_SCHEMA || schema == PROFILE_SCHEMA_V2
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

      if schema == PROFILE_SCHEMA_V2
        return validate_v2_document!(root, profile_sha256, manifest_sha, payload_sha,
          prompt_sha, prompt, valid)
      end

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

      action_value = Action.new(1, "<|Subject1|>", action_start_seconds, action_end_seconds,
        string(action, "action_description"), "segment_local_seconds", 5.0, 10.0,
        latent_start, latent_end)
      new(PROFILE_SCHEMA, profile_sha256, manifest_sha, payload_sha, prompt_sha, valid,
        head_end, action_start, action_end, source_offsets_raw, effective_offsets,
        action_start_seconds, action_end_seconds,
        latent_start, latent_end, active, windows, [action_value])
    end

    private def self.validate_v2_document!(root : Hash(String, JSON::Any),
                                           profile_sha256 : String,
                                           manifest_sha : String,
                                           payload_sha : String,
                                           prompt_sha : String,
                                           prompt : String,
                                           valid : Int32) : Profile
      if root.has_key?("action")
        raise ArgumentError.new("V2 Hy-Nomas action profile must replace action with actions")
      end
      raw_actions = field(root, "actions").as_a
      unless raw_actions.size == 2
        raise ArgumentError.new("V2 Hy-Nomas action profile requires exactly two ordered actions")
      end

      actions = [] of Action
      expected_subjects = ["<|Subject1|>", "<|Subject2|>"]
      raw_actions.each_with_index do |raw_action, index|
        action = raw_action.as_h
        unless action.keys.sort == [
                 "action_description", "action_id", "end_time_local_seconds",
                 "latent_end", "latent_start", "source_segment_end_seconds",
                 "source_segment_start_seconds", "start_time_local_seconds",
                 "subject_id", "time_origin",
               ].sort
          raise ArgumentError.new("V2 Hy-Nomas action metadata fields are malformed")
        end

        action_id = integer(action, "action_id")
        subject_id = string(action, "subject_id")
        start_seconds = number(action, "start_time_local_seconds")
        end_seconds = number(action, "end_time_local_seconds")
        description = string(action, "action_description")
        time_origin = string(action, "time_origin")
        source_start = number(action, "source_segment_start_seconds")
        source_end = number(action, "source_segment_end_seconds")
        latent_start = integer(action, "latent_start")
        latent_end = integer(action, "latent_end")
        expected_id = index + 1
        expected_subject = expected_subjects[index]

        unless action_id == expected_id && subject_id == expected_subject &&
               time_origin == "segment_local_seconds" && source_start == 5.0 && source_end == 10.0 &&
               0.0 <= start_seconds < end_seconds <= 5.0
          raise ArgumentError.new("Malformed V2 Action#{expected_id} metadata or subject binding")
        end
        unless latent_start == latent_frame_for(start_seconds) &&
               latent_end == latent_frame_for(end_seconds) &&
               0 <= latent_start < latent_end <= SEGMENT_LATENT_FRAMES
          raise ArgumentError.new("V2 Action#{expected_id} must occupy a nonempty timestamp-derived latent span")
        end
        validate_action_wrapper!(description, expected_id, expected_subject)
        actions << Action.new(action_id, subject_id, start_seconds, end_seconds,
          description, time_origin, source_start, source_end, latent_start, latent_end)
      end
      unless actions[0].start_time_local_seconds < actions[1].start_time_local_seconds &&
             actions[0].end_time_local_seconds <= actions[1].start_time_local_seconds &&
             actions[0].latent_end <= actions[1].latent_start
        raise ArgumentError.new("V2 actions must be chronological and nonoverlapping")
      end
      validate_prompt_actions!(prompt, actions)

      offsets = hash(root, "text_offsets")
      head_end = integer(offsets, "head_end")
      source_offsets_raw = integer_array(offsets, "source_offsets_raw")
      effective_offsets = integer_array(offsets, "effective_offsets")
      expected_effective = source_offsets_raw.map { |offset| offset < valid ? offset : valid }
      unless integer(offsets, "valid_text_tokens") == valid &&
             source_offsets_raw.size == 3 && effective_offsets.size == 3 &&
             source_offsets_raw.all? { |offset| 0 <= offset <= TEXT_LENGTH } &&
             effective_offsets == expected_effective && effective_offsets[0] == head_end &&
             0 < effective_offsets[0] < effective_offsets[1] < effective_offsets[2] &&
             effective_offsets[2] == valid &&
             string(offsets, "offset_method_endpoint_note") ==
               "BasePipeline returns the raw attention-mask sum; pinned nomas_cpt clamps it with min(offset, valid current text query tokens), yielding the authenticated cropped-mask count"
        raise ArgumentError.new("V2 Hy-Nomas action text offsets do not partition head and two actions")
      end

      windows = validate_v2_windows!(field(root, "video_windows"), actions)
      unless boolean(root, "intra_segment_active")
        raise ArgumentError.new("V2 Hy-Nomas actions require active intra-segment windows")
      end

      labels = hash(root, "labels")
      action_labels_raw = field(labels, "actions").as_a.map(&.as_i64)
      unless action_labels_raw == [1_i64, 2_i64]
        raise ArgumentError.new("V2 Hy-Nomas action label IDs must be exactly [1, 2]")
      end
      action_labels = action_labels_raw.map(&.to_i32)
      unless integer(labels, "global") == 0 && action_labels == [1, 2] &&
             !labels.has_key?("action") && integer(labels, "padding") == -1 &&
             integer(labels, "video_gap_window_index") == -1 &&
             string(labels, "query_scope") == "global_head_and_matching_action_window" &&
             string(labels, "key_scope") == "global_history_or_matching_action"
        raise ArgumentError.new("V2 Hy-Nomas action labels do not match the native mask contract")
      end

      # Existing scalar getters keep their v1 meaning and expose Action1 for
      # callers that have not migrated to the typed actions array.
      first_action = actions[0]
      new(PROFILE_SCHEMA_V2, profile_sha256, manifest_sha, payload_sha, prompt_sha, valid,
        head_end, effective_offsets[0], effective_offsets[1], source_offsets_raw,
        effective_offsets, first_action.start_time_local_seconds,
        first_action.end_time_local_seconds, first_action.latent_start,
        first_action.latent_end, true, windows, actions)
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

    private def self.validate_prompt_actions!(prompt : String, actions : Array(Action)) : Nil
      parsed = JSON.parse(prompt).as_h
      segments = parsed["segments"].as_a
      unless segments.size == 1
        raise ArgumentError.new("V2 Hy-Nomas action prompt must contain one T1 segment")
      end
      segment = segments[0].as_h
      unless timestamp_seconds(segment["start_time"].as_s) == 5.0 &&
             timestamp_seconds(segment["end_time"].as_s) == 10.0
        raise ArgumentError.new("V2 Hy-Nomas prompt has the wrong global T1 segment time origin")
      end
      prompt_actions = segment["contained_actions"].as_a
      unless prompt_actions.size == 2
        raise ArgumentError.new("V2 Hy-Nomas prompt must contain exactly Action1 and Action2")
      end

      prompt_actions.each_with_index do |raw_action, index|
        prompt_action = raw_action.as_h
        unless prompt_action.keys.sort == ["action_description", "end_time", "start_time"]
          raise ArgumentError.new("V2 Hy-Nomas Action#{index + 1} prompt fields are malformed")
        end
        action = actions[index]
        start_time = prompt_action["start_time"].as_s
        end_time = prompt_action["end_time"].as_s
        unless start_time == format_timestamp(action.start_time_local_seconds) &&
               end_time == format_timestamp(action.end_time_local_seconds) &&
               timestamp_seconds(start_time) == action.start_time_local_seconds &&
               timestamp_seconds(end_time) == action.end_time_local_seconds &&
               prompt_action["action_description"].as_s == action.action_description
          raise ArgumentError.new("V2 Hy-Nomas Action#{index + 1} prompt and profile metadata differ")
        end
      end
    end

    private def self.validate_action_wrapper!(description : String,
                                              action_id : Int32,
                                              subject_id : String) : Nil
      opening = "<Action#{action_id}>"
      closing = "</Action#{action_id}>."
      unless description.starts_with?(opening) && description.ends_with?(closing) &&
             description.scan(/<Action/).size == 1 && description.scan(/<\/Action/).size == 1
        raise ArgumentError.new("V2 action description has a missing, truncated, or mismatched Action#{action_id} wrapper")
      end
      body = description[opening.size...(description.size - closing.size)]
      unless body && !body.empty? && body.scan(/<\|Subject\d+\|>/).size == 1 &&
             body.includes?(subject_id) && !body.gsub(subject_id, "").includes?("<|Subject")
        raise ArgumentError.new("V2 Action#{action_id} must refer only to #{subject_id} inside its wrapper")
      end
      if body.gsub(subject_id, "").strip.empty?
        raise ArgumentError.new("V2 Action#{action_id} must include nonempty action text after #{subject_id}")
      end
    end

    private def self.validate_v2_windows!(value : JSON::Any,
                                          actions : Array(Action)) : Array(Tuple(Int32, Int32, Int32))
      windows_json = value.as_a
      windows = [] of Tuple(Int32, Int32, Int32)
      windows_json.each do |raw_window|
        entries = raw_window.as_a
        unless entries.size == 3
          raise ArgumentError.new("V2 Hy-Nomas action window must contain three integers")
        end
        start_value = entries[0].as_i64
        end_value = entries[1].as_i64
        action_value = entries[2].as_i64
        unless 0 <= start_value < end_value <= SEGMENT_LATENT_FRAMES && action_value.in?(-1_i64..1_i64)
          raise ArgumentError.new("V2 Hy-Nomas action window is out of range")
        end
        windows << {start_value.to_i32, end_value.to_i32, action_value.to_i32}
      end

      expected = [] of Tuple(Int32, Int32, Int32)
      cursor = 0_i32
      actions.each_with_index do |action, index|
        if action.latent_start > cursor
          expected << {cursor, action.latent_start, -1}
        end
        expected << {action.latent_start, action.latent_end, index.to_i32}
        cursor = action.latent_end
      end
      if cursor < SEGMENT_LATENT_FRAMES
        expected << {cursor, SEGMENT_LATENT_FRAMES, -1}
      end
      unless windows == expected
        raise ArgumentError.new("V2 Hy-Nomas video windows must tile the segment with exact timestamp-derived action spans")
      end
      windows
    end

    private def self.latent_frame_for(seconds : Float64) : Int32
      (seconds * 24).to_i32 // 4
    end

    private def self.format_timestamp(seconds : Float64) : String
      minutes = (seconds / 60.0).to_i
      remainder_millis = (seconds * 1000.0).round.to_i - minutes * 60_000
      seconds_part = remainder_millis // 1000
      millis_part = remainder_millis % 1000
      "#{minutes.to_s.rjust(2, '0')}:#{seconds_part.to_s.rjust(2, '0')}.#{millis_part.to_s.rjust(3, '0')}"
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
