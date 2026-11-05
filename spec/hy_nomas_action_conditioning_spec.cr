require "spec"
require "digest/sha256"
require "json"
require "../src/ml/video/hy_nomas_action_conditioning"

private ACTION_PROMPT            = %({"segments":[{"start_time":"00:05.000","end_time":"00:10.000","description_change":"Resume remains beside the laptop.","contained_actions":[{"start_time":"00:00.500","end_time":"00:04.500","action_description":"<Action1><|Subject1|> turns his head toward Resume</Action1>."}]}]})
private SEQUENTIAL_ACTION_PROMPT = %({"segments":[{"start_time":"00:05.000","end_time":"00:10.000","description_change":"Alex and Resume remain in the same shot.","contained_actions":[{"start_time":"00:00.500","end_time":"00:02.500","action_description":"<Action1><|Subject1|> turns his head toward the paper figure</Action1>."},{"start_time":"00:02.500","end_time":"00:04.500","action_description":"<Action2><|Subject2|> raises one gloved arm toward its chest</Action2>."}]}]})
private MANIFEST_SHA             = "a" * 64
private PAYLOAD_SHA              = "b" * 64

private def action_profile_document(prompt : String = ACTION_PROMPT) : String
  prompt_sha = Digest::SHA256.hexdigest(prompt)
  {
    "schema"                       => ML::Video::HyNomasActionConditioning::PROFILE_SCHEMA,
    "conditioning_manifest_sha256" => MANIFEST_SHA,
    "conditioning_payload_sha256"  => PAYLOAD_SHA,
    "prompt_sha256"                => prompt_sha,
    "prompt"                       => prompt,
    "valid_text_tokens"            => 80,
    "geometry"                     => {
      "run_segments"          => 2,
      "segment_index"         => 1,
      "segment_start_seconds" => 5.0,
      "segment_end_seconds"   => 10.0,
      "latent_frames"         => 30,
      "tokens_per_frame"      => 144,
      "video_tokens"          => 4320,
      "text_length"           => 1280,
    },
    "token_contract" => {
      "schema"                        => ML::Video::HyNomasActionConditioning::PROMPT_SCHEMA,
      "schema_version"                => 1,
      "prompt_sha256"                 => prompt_sha,
      "crop_start"                    => 108,
      "text_length"                   => 1280,
      "template_name"                 => ML::Video::HyNomasActionConditioning::TEMPLATE_NAME,
      "template_sha256"               => ML::Video::HyNomasActionConditioning::TEMPLATE_SHA256,
      "valid_text_tokens"             => 80,
      "raw_token_ids_sha256"          => "c" * 64,
      "raw_attention_mask_sha256"     => "d" * 64,
      "cropped_token_ids_sha256"      => "e" * 64,
      "cropped_attention_mask_sha256" => "f" * 64,
      "tokenizer_assets"              => {
        "config.json"           => {"nbytes" => 1, "sha256" => "1" * 64},
        "tokenizer.json"        => {"nbytes" => 2, "sha256" => "2" * 64},
        "tokenizer_config.json" => {"nbytes" => 3, "sha256" => "3" * 64},
      },
      "tokenizer_identity" => {
        "repo_id"              => ML::Video::HyNomasActionConditioning::TOKENIZER_REPO,
        "revision"             => ML::Video::HyNomasActionConditioning::TOKENIZER_REVISION,
        "transformers_version" => "4.57.3",
        "tokenizers_version"   => "0.22.1",
      },
    },
    "source" => {
      "upstream_revision"             => ML::Video::HyNomasActionConditioning::UPSTREAM_REVISION,
      "source_sha256"                 => ML::Video::HyNomasActionConditioning::EXPECTED_SOURCE_SHA256,
      "offset_method"                 => "BasePipeline._compute_action_text_offsets (AST-extracted from pinned base.py)",
      "video_window_method"           => "BasePipeline._compute_action_video_windows (AST-extracted from pinned base.py)",
      "text_tokenizer_methods"        => ["TextEncoder.apply_text_to_template", "TextEncoder.text2tokens"],
      "offset_clamp_method"           => "nomas_cpt.py clamps every source offset with min(offset, valid_current_text_query_tokens)",
      "time_normalizer"               => "hymm.inference._normalize_contained_action_timestamps (AST-extracted)",
      "time_origin"                   => "segment_local_seconds",
      "resolve_subject_ids"           => false,
      "action_wrapper_representation" => "literal_text_markup_not_token_ids",
    },
    "action" => {
      "action_id"                    => 1,
      "subject_id"                   => "<|Subject1|>",
      "start_time_local_seconds"     => 0.5,
      "end_time_local_seconds"       => 4.5,
      "action_description"           => "<Action1><|Subject1|> turns his head toward Resume</Action1>.",
      "time_origin"                  => "segment_local_seconds",
      "source_segment_start_seconds" => 5.0,
      "source_segment_end_seconds"   => 10.0,
      "latent_start"                 => 3,
      "latent_end"                   => 27,
    },
    "text_offsets" => {
      "head_end"                    => 12,
      "action_start"                => 12,
      "action_end"                  => 80,
      "valid_text_tokens"           => 80,
      "source_offsets_raw"          => [12, 188],
      "effective_offsets"           => [12, 80],
      "offset_method_endpoint_note" => "BasePipeline returns the raw attention-mask sum; pinned nomas_cpt clamps it with min(offset, valid current text query tokens), yielding the authenticated cropped-mask count",
    },
    "video_windows"        => [[0, 3, -1], [3, 27, 0], [27, 30, -1]],
    "intra_segment_active" => true,
    "labels"               => {
      "global"                 => 0,
      "action"                 => 1,
      "padding"                => -1,
      "video_gap_window_index" => -1,
      "query_scope"            => "global_head_and_matching_action_window",
      "key_scope"              => "global_history_or_matching_action",
    },
  }.to_json
end

private def sequential_action_profile_document(prompt : String = SEQUENTIAL_ACTION_PROMPT) : String
  document = JSON.parse(action_profile_document).as_h
  document["schema"] = JSON::Any.new(ML::Video::HyNomasActionConditioning::PROFILE_SCHEMA_V2)
  document["prompt"] = JSON::Any.new(prompt)
  document["prompt_sha256"] = JSON::Any.new(Digest::SHA256.hexdigest(prompt))
  document["token_contract"].as_h["prompt_sha256"] = JSON::Any.new(Digest::SHA256.hexdigest(prompt))
  document.delete("action")
  document["actions"] = JSON.parse(%([
    {"action_id":1,"subject_id":"<|Subject1|>","start_time_local_seconds":0.5,"end_time_local_seconds":2.5,"action_description":"<Action1><|Subject1|> turns his head toward the paper figure</Action1>.","time_origin":"segment_local_seconds","source_segment_start_seconds":5.0,"source_segment_end_seconds":10.0,"latent_start":3,"latent_end":15},
    {"action_id":2,"subject_id":"<|Subject2|>","start_time_local_seconds":2.5,"end_time_local_seconds":4.5,"action_description":"<Action2><|Subject2|> raises one gloved arm toward its chest</Action2>.","time_origin":"segment_local_seconds","source_segment_start_seconds":5.0,"source_segment_end_seconds":10.0,"latent_start":15,"latent_end":27}
  ]))
  document["text_offsets"] = JSON.parse(%({"head_end":12,"valid_text_tokens":80,"source_offsets_raw":[12,43,188],"effective_offsets":[12,43,80],"offset_method_endpoint_note":"BasePipeline returns the raw attention-mask sum; pinned nomas_cpt clamps it with min(offset, valid current text query tokens), yielding the authenticated cropped-mask count"}))
  document["video_windows"] = JSON.parse(%([[0,3,-1],[3,15,0],[15,27,1],[27,30,-1]]))
  document["labels"] = JSON.parse(%({"global":0,"actions":[1,2],"padding":-1,"video_gap_window_index":-1,"query_scope":"global_head_and_matching_action_window","key_scope":"global_history_or_matching_action"}))
  document.to_json
end

private def with_action_profile(document : String, prompt : String = ACTION_PROMPT,
                                &block : ML::Video::HyNomasActionConditioning::Profile ->)
  File.tempfile("hy-nomas-action-conditioning", ".json") do |file|
    file.print(document)
    file.flush
    sha = Digest::SHA256.hexdigest(document)
    profile = ML::Video::HyNomasActionConditioning::Profile.from_file(
      file.path, sha,
      conditioning_manifest_sha256: MANIFEST_SHA,
      conditioning_payload_sha256: PAYLOAD_SHA,
      prompt: prompt,
      valid_text_tokens: 80)
    yield profile
  end
end

describe ML::Video::HyNomasActionConditioning::Profile do
  it "loads an authenticated bounded profile and exposes its binding identity" do
    with_action_profile(action_profile_document) do |profile|
      profile.schema.should eq(ML::Video::HyNomasActionConditioning::PROFILE_SCHEMA)
      profile.sequential_actions?.should be_false
      profile.actions.map(&.action_id).should eq([1])
      profile.valid_text_tokens.should eq(80)
      profile.profile_sha256.should eq(Digest::SHA256.hexdigest(action_profile_document))
      profile.intra_segment_active.should be_true
      profile.identity[:conditioning_manifest_sha256].should eq(MANIFEST_SHA)
      profile.identity[:action_start_seconds].should eq(0.5)
      profile.identity[:latent_start].should eq(3)
      v1_identity = JSON.parse(profile.identity.to_json).as_h
      v1_identity.keys.sort.should eq([
        "schema", "profile_sha256", "conditioning_manifest_sha256", "conditioning_payload_sha256",
        "prompt_sha256", "valid_text_tokens", "action_id", "action_start_seconds",
        "action_end_seconds", "head_end", "action_start", "action_end", "source_offsets_raw",
        "effective_offsets", "latent_start", "latent_end", "intra_segment_active",
      ].sort)
      v1_identity.has_key?("actions").should be_false
      profile.validate_binding!(
        conditioning_manifest_sha256: MANIFEST_SHA,
        conditioning_payload_sha256: PAYLOAD_SHA,
        prompt: ACTION_PROMPT,
        valid_text_tokens: 80)
    end
  end

  it "loads two ordered subject actions and exposes their full authenticated identity" do
    with_action_profile(sequential_action_profile_document, SEQUENTIAL_ACTION_PROMPT) do |profile|
      profile.schema.should eq(ML::Video::HyNomasActionConditioning::PROFILE_SCHEMA_V2)
      profile.sequential_actions?.should be_true
      profile.actions.map(&.action_id).should eq([1, 2])
      profile.actions.map(&.subject_id).should eq(["<|Subject1|>", "<|Subject2|>"])
      profile.actions.map(&.latent_start).should eq([3, 15])
      profile.actions.map(&.latent_end).should eq([15, 27])
      profile.action_start_seconds.should eq(0.5)
      profile.action_end_seconds.should eq(2.5)
      profile.identity[:schema].should eq(ML::Video::HyNomasActionConditioning::PROFILE_SCHEMA_V2)
      profile.identity[:action_start_seconds].should eq(0.5)
      identity_document = JSON.parse(profile.identity.to_json).as_h
      identity_document["actions"].as_a.size.should eq(2)
      profile.validate_binding!(
        conditioning_manifest_sha256: MANIFEST_SHA,
        conditioning_payload_sha256: PAYLOAD_SHA,
        prompt: SEQUENTIAL_ACTION_PROMPT,
        valid_text_tokens: 80)
    end
  end

  it "labels sequential action query ranges while keeping video and history keys global" do
    with_action_profile(sequential_action_profile_document, SEQUENTIAL_ACTION_PROMPT) do |profile|
      labels = profile.labels(video_tokens: 4320, query_text_tokens: 100,
        history_rows: 2, current_text_tokens: 80)

      labels.query_labels[0, 3 * 144].should eq(Array.new(3 * 144, 0))
      labels.query_labels[3 * 144, 12 * 144].should eq(Array.new(12 * 144, 1))
      labels.query_labels[15 * 144, 12 * 144].should eq(Array.new(12 * 144, 2))
      labels.query_labels[27 * 144, 3 * 144].should eq(Array.new(3 * 144, 0))
      labels.query_labels[4320, 12].should eq(Array.new(12, 0))
      labels.query_labels[4332, 31].should eq(Array.new(31, 1))
      labels.query_labels[4363, 37].should eq(Array.new(37, 2))
      labels.query_labels[4400, 20].should eq(Array.new(20, -1))
      labels.key_labels[0, 2].should eq([0, 0])
      labels.key_labels[2, 4320].should eq(Array.new(4320, 0))
      labels.key_labels[2 + 3 * 144, 12 * 144].should eq(Array.new(12 * 144, 0))
      labels.key_labels[2 + 15 * 144, 12 * 144].should eq(Array.new(12 * 144, 0))
      labels.key_labels[2 + 4320, 80].should eq(labels.query_labels[4320, 80])
    end
  end

  it "rejects overlapping, reversed, missing, unknown-subject, empty-span, and wrapper-invalid v2 actions" do
    overlap = JSON.parse(sequential_action_profile_document).as_h
    overlap["actions"].as_a[0].as_h["end_time_local_seconds"] = JSON::Any.new(3.0)
    overlap["actions"].as_a[0].as_h["latent_end"] = JSON::Any.new(18_i64)
    expect_raises(ArgumentError, /chronological and nonoverlapping/) do
      with_action_profile(overlap.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    reversed = JSON.parse(sequential_action_profile_document).as_h
    reversed["actions"].as_a.reverse!
    expect_raises(ArgumentError, /metadata or subject binding/) do
      with_action_profile(reversed.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    duplicate = JSON.parse(sequential_action_profile_document).as_h
    duplicate["actions"].as_a[1].as_h["action_id"] = JSON::Any.new(1_i64)
    expect_raises(ArgumentError, /metadata or subject binding/) do
      with_action_profile(duplicate.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    missing = JSON.parse(sequential_action_profile_document).as_h
    missing["actions"].as_a.pop
    expect_raises(ArgumentError, /exactly two ordered actions/) do
      with_action_profile(missing.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    unknown_subject = JSON.parse(sequential_action_profile_document).as_h
    unknown_subject["actions"].as_a[1].as_h["subject_id"] = JSON::Any.new("<|Subject3|>")
    expect_raises(ArgumentError, /metadata or subject binding/) do
      with_action_profile(unknown_subject.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    empty_span = JSON.parse(sequential_action_profile_document).as_h
    empty_span["actions"].as_a[1].as_h["start_time_local_seconds"] = JSON::Any.new(0.1)
    empty_span["actions"].as_a[1].as_h["end_time_local_seconds"] = JSON::Any.new(0.2)
    empty_span["actions"].as_a[1].as_h["latent_start"] = JSON::Any.new(0_i64)
    empty_span["actions"].as_a[1].as_h["latent_end"] = JSON::Any.new(0_i64)
    expect_raises(ArgumentError, /nonempty timestamp-derived latent span/) do
      with_action_profile(empty_span.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    wrong_wrapper_prompt = SEQUENTIAL_ACTION_PROMPT.gsub("<Action2>", "<Action1>").gsub("</Action2>", "</Action1>")
    wrong_wrapper = JSON.parse(sequential_action_profile_document(wrong_wrapper_prompt)).as_h
    wrong_wrapper["actions"].as_a[1].as_h["action_description"] = JSON::Any.new(
      "<Action1><|Subject2|> raises one gloved arm toward its chest</Action1>.")
    expect_raises(ArgumentError, /missing, truncated, or mismatched/) do
      with_action_profile(wrong_wrapper.to_json, wrong_wrapper_prompt) { |_| }
    end

    wrong_subject_prompt = SEQUENTIAL_ACTION_PROMPT.gsub("<|Subject2|>", "<|Subject3|>")
    wrong_subject_body = JSON.parse(sequential_action_profile_document(wrong_subject_prompt)).as_h
    wrong_subject_body["actions"].as_a[1].as_h["action_description"] = JSON::Any.new(
      "<Action2><|Subject3|> raises one gloved arm toward its chest</Action2>.")
    expect_raises(ArgumentError, /refer only to <\|Subject2\|>/) do
      with_action_profile(wrong_subject_body.to_json, wrong_subject_prompt) { |_| }
    end

    truncated_prompt = SEQUENTIAL_ACTION_PROMPT.gsub("</Action2>.", "")
    truncated = JSON.parse(sequential_action_profile_document(truncated_prompt)).as_h
    truncated["actions"].as_a[1].as_h["action_description"] = JSON::Any.new(
      "<Action2><|Subject2|> raises one gloved arm toward its chest")
    expect_raises(ArgumentError, /missing, truncated, or mismatched/) do
      with_action_profile(truncated.to_json, truncated_prompt) { |_| }
    end
  end

  it "rejects extra valid and malformed subject tags inside a v2 action" do
    original = "<Action2><|Subject2|> raises one gloved arm toward its chest</Action2>."
    ["<|SubjectX|>", "<|Subject", "<|Subject3|>"].each do |extra_subject|
      description = "<Action2><|Subject2|> raises one gloved arm toward its chest #{extra_subject}</Action2>."
      prompt = SEQUENTIAL_ACTION_PROMPT.gsub(original, description)
      document = JSON.parse(sequential_action_profile_document(prompt)).as_h
      document["actions"].as_a[1].as_h["action_description"] = JSON::Any.new(description)

      expect_raises(ArgumentError, /must refer only to <\\|Subject2\\|>/) do
        with_action_profile(document.to_json, prompt) { |_| }
      end
    end
  end

  it "rejects v2 actions without nonempty text after the subject identifier" do
    [
      "<Action2><|Subject2|></Action2>.",
      "<Action2><|Subject2|>   </Action2>.",
    ].each do |description|
      prompt = SEQUENTIAL_ACTION_PROMPT.gsub(
        "<Action2><|Subject2|> raises one gloved arm toward its chest</Action2>.", description)
      document = JSON.parse(sequential_action_profile_document(prompt)).as_h
      document["actions"].as_a[1].as_h["action_description"] = JSON::Any.new(description)

      expect_raises(ArgumentError, /nonempty action text/) do
        with_action_profile(document.to_json, prompt) { |_| }
      end
    end
  end

  it "rejects v2 text offset and video-window mismatches" do
    bad_offsets = JSON.parse(sequential_action_profile_document).as_h
    bad_offsets["text_offsets"].as_h["effective_offsets"].as_a[1] = JSON::Any.new(12_i64)
    expect_raises(ArgumentError, /text offsets/) do
      with_action_profile(bad_offsets.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    bad_window = JSON.parse(sequential_action_profile_document).as_h
    bad_window["video_windows"].as_a[2].as_a[0] = JSON::Any.new(14_i64)
    expect_raises(ArgumentError, /exact timestamp-derived action spans/) do
      with_action_profile(bad_window.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    legacy_windows = JSON.parse(sequential_action_profile_document).as_h
    legacy_windows["intra_segment_active"] = JSON::Any.new(false)
    expect_raises(ArgumentError, /require active intra-segment windows/) do
      with_action_profile(legacy_windows.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end

    changed_binding = JSON.parse(sequential_action_profile_document).as_h
    changed_binding["conditioning_payload_sha256"] = JSON::Any.new("9" * 64)
    expect_raises(ArgumentError, /conditioning identity/) do
      with_action_profile(changed_binding.to_json, SEQUENTIAL_ACTION_PROMPT) { |_| }
    end
  end

  it "maps action windows and text spans into query and key labels" do
    with_action_profile(action_profile_document) do |profile|
      labels = profile.labels(video_tokens: 4320, query_text_tokens: 100,
        history_rows: 2, current_text_tokens: 80)

      labels.query_labels.size.should eq(4420)
      labels.query_labels[0, 3 * 144].should eq(Array.new(3 * 144, 0))
      labels.query_labels[3 * 144, 24 * 144].should eq(Array.new(24 * 144, 1))
      labels.query_labels[27 * 144, 3 * 144].should eq(Array.new(3 * 144, 0))
      labels.query_labels[4320, 12].should eq(Array.new(12, 0))
      labels.query_labels[4332, 68].should eq(Array.new(68, 1))
      labels.query_labels[4400, 20].should eq(Array.new(20, -1))
      labels.key_labels.size.should eq(2 + 4320 + 80)
      labels.key_labels[0, 2].should eq([0, 0])
      labels.key_labels[2, 4320].should eq(Array.new(4320, 0))
      labels.key_labels[2 + 3 * 144, 24 * 144].should eq(Array.new(24 * 144, 0))
      labels.key_labels[2 + 4320, 80].should eq(labels.query_labels[4320, 80])

      no_current_text = profile.labels(video_tokens: 4320, query_text_tokens: 80,
        history_rows: 2, current_text_tokens: 0)
      no_current_text.key_labels.size.should eq(2 + 4320)
    end
  end

  it "admits only an explicit full-segment legacy action without active windows" do
    full_prompt = ACTION_PROMPT.gsub("00:00.500", "00:00.000").gsub("00:04.500", "00:05.000")
    document = JSON.parse(action_profile_document(full_prompt))
    document.as_h["action"].as_h["start_time_local_seconds"] = JSON::Any.new(0.0)
    document.as_h["action"].as_h["end_time_local_seconds"] = JSON::Any.new(5.0)
    document.as_h["action"].as_h["latent_start"] = JSON::Any.new(0_i64)
    document.as_h["action"].as_h["latent_end"] = JSON::Any.new(30_i64)
    document.as_h["video_windows"] = JSON::Any.new(nil)
    document.as_h["intra_segment_active"] = JSON::Any.new(false)

    with_action_profile(document.to_json, full_prompt) do |profile|
      profile.intra_segment_active.should be_false
      profile.video_windows.should be_nil
      labels = profile.labels(video_tokens: 4320, query_text_tokens: 80,
        history_rows: 0, current_text_tokens: 80)
      labels.query_labels.uniq.should eq([0])
      labels.key_labels.uniq.should eq([0])
    end
  end

  it "rejects wrong hashes, crop, offsets, window coverage, and time origin" do
    document = action_profile_document
    sha = Digest::SHA256.hexdigest(document)
    File.tempfile("hy-nomas-action-binding", ".json") do |file|
      file.print(document)
      file.flush
      expect_raises(ArgumentError, /file SHA-256/) do
        ML::Video::HyNomasActionConditioning::Profile.from_file(
          file.path, "0" * 64,
          conditioning_manifest_sha256: MANIFEST_SHA,
          conditioning_payload_sha256: PAYLOAD_SHA,
          prompt: ACTION_PROMPT,
          valid_text_tokens: 80)
      end
      expect_raises(ArgumentError, /conditioning identity/) do
        ML::Video::HyNomasActionConditioning::Profile.from_file(
          file.path, sha,
          conditioning_manifest_sha256: "9" * 64,
          conditioning_payload_sha256: PAYLOAD_SHA,
          prompt: ACTION_PROMPT,
          valid_text_tokens: 80)
      end
    end

    bad_crop = JSON.parse(document)
    bad_crop.as_h["token_contract"].as_h["crop_start"] = JSON::Any.new(0_i64)
    expect_raises(ArgumentError, /Qwen token contract/) do
      with_action_profile(bad_crop.to_json) { |_| }
    end

    bad_offsets = JSON.parse(document)
    bad_offsets.as_h["text_offsets"].as_h["action_end"] = JSON::Any.new(79_i64)
    expect_raises(ArgumentError, /text offsets/) do
      with_action_profile(bad_offsets.to_json) { |_| }
    end

    bad_windows = JSON.parse(document)
    bad_windows["video_windows"].as_a[1].as_a[0] = JSON::Any.new(4_i64)
    expect_raises(ArgumentError, /action windows/) do
      with_action_profile(bad_windows.to_json) { |_| }
    end

    out_of_range_windows = JSON.parse(document)
    out_of_range_windows["video_windows"].as_a[2].as_a[1] = JSON::Any.new(31_i64)
    expect_raises(ArgumentError, /action windows/) do
      with_action_profile(out_of_range_windows.to_json) { |_| }
    end

    inactive_subwindow = JSON.parse(document)
    inactive_subwindow.as_h["intra_segment_active"] = JSON::Any.new(false)
    inactive_subwindow.as_h["video_windows"] = JSON::Any.new(nil)
    expect_raises(ArgumentError, /Legacy full-segment/) do
      with_action_profile(inactive_subwindow.to_json) { |_| }
    end

    bad_origin = JSON.parse(document)
    bad_origin.as_h["action"].as_h["time_origin"] = JSON::Any.new("global_seconds")
    expect_raises(ArgumentError, /time origin/) do
      with_action_profile(bad_origin.to_json) { |_| }
    end
  end

  it "rejects an action prompt whose wrapper or subject does not match its profile" do
    malformed_prompt = ACTION_PROMPT.gsub("<Action1>", "<Action2>").gsub("</Action1>", "</Action2>")
    document = action_profile_document(malformed_prompt)
    File.tempfile("hy-nomas-action-prompt", ".json") do |file|
      file.print(document)
      file.flush
      sha = Digest::SHA256.hexdigest(document)
      expect_raises(ArgumentError, /Action1 wrapper/) do
        ML::Video::HyNomasActionConditioning::Profile.from_file(
          file.path, sha,
          conditioning_manifest_sha256: MANIFEST_SHA,
          conditioning_payload_sha256: PAYLOAD_SHA,
          prompt: malformed_prompt,
          valid_text_tokens: 80)
      end
    end
  end
end
