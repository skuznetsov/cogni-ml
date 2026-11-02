require "spec"
require "digest/sha256"
require "json"
require "../src/ml/video/hy_nomas_action_conditioning"

private ACTION_PROMPT = %({"segments":[{"start_time":"00:05.000","end_time":"00:10.000","description_change":"Resume remains beside the laptop.","contained_actions":[{"start_time":"00:00.500","end_time":"00:04.500","action_description":"<Action1><|Subject1|> turns his head toward Resume</Action1>."}]}]})
private MANIFEST_SHA  = "a" * 64
private PAYLOAD_SHA   = "b" * 64

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
      profile.valid_text_tokens.should eq(80)
      profile.profile_sha256.should eq(Digest::SHA256.hexdigest(action_profile_document))
      profile.intra_segment_active.should be_true
      profile.identity[:conditioning_manifest_sha256].should eq(MANIFEST_SHA)
      profile.identity[:action_start_seconds].should eq(0.5)
      profile.identity[:latent_start].should eq(3)
      profile.validate_binding!(
        conditioning_manifest_sha256: MANIFEST_SHA,
        conditioning_payload_sha256: PAYLOAD_SHA,
        prompt: ACTION_PROMPT,
        valid_text_tokens: 80)
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
