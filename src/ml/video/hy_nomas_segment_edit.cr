require "digest/sha256"
require "json"

module ML::Video
  # One deterministic new take within an authenticated six-segment input bank.
  # The donor is outside the whole active run, so it remains initial noise
  # through every save/restart. Shared scene conditioning is unchanged.
  struct HyNomasSegmentNoiseEdit
    getter run_segments : Int32
    getter target_segment : Int32
    getter source_segment : Int32
    getter resident_segments : Int32

    def initialize(@run_segments : Int32, @target_segment : Int32,
                   @source_segment : Int32, @resident_segments : Int32 = 6)
      unless 2 <= @run_segments < @resident_segments <= 6 &&
             1 <= @target_segment < @run_segments &&
             @run_segments <= @source_segment < @resident_segments
        raise ArgumentError.new("Noise edit requires a noninitial active target and an unused future donor")
      end
    end

    def identity
      {schema: "cogni_ml.hy_nomas_segment_noise_edit/v1", run_segments: @run_segments,
       target_segment: @target_segment, source_segment: @source_segment,
       resident_segments: @resident_segments}
    end

    def source_for(segment : Int32) : Int32
      unless 0 <= segment < @run_segments
        raise ArgumentError.new("Noise edit segment is outside the active run")
      end
      segment == @target_segment ? @source_segment : segment
    end

    # Future edits cannot change an already-produced prefix. Once applied,
    # their identity is mandatory for every descendant checkpoint. The base
    # hash retains the executable, input, conditioning, weights and schedule
    # binding; this is not a cross-build migration or an equality relaxation.
    def bind_contract(base : String, next_segment : Int32) : String
      unless 1 <= next_segment < @run_segments
        raise ArgumentError.new("Noise edit contract requires an extendable boundary")
      end
      return base if next_segment <= @target_segment
      {schema: "cogni_ml.hy_nomas_edited_sequence_boundary/v1",
       base_contract_sha256: Digest::SHA256.hexdigest(base), noise_edit: identity}.to_json
    end
  end

  # A single replacement text turn. Earlier cache/latent bytes remain valid;
  # every descendant must be regenerated after the replacement is consumed.
  # The next chunk returns to the original prompt as a new upstream text turn.
  struct HyNomasSegmentPromptEdit
    getter run_segments : Int32
    getter target_segment : Int32
    getter conditioning_manifest_sha256 : String
    getter conditioning_payload_sha256 : String
    getter prompt_sha256 : String
    getter valid_text_tokens : Int32

    def initialize(@run_segments : Int32, @target_segment : Int32,
                   @conditioning_manifest_sha256 : String,
                   @conditioning_payload_sha256 : String, @prompt_sha256 : String,
                   @valid_text_tokens : Int32)
      unless 2 <= @run_segments <= 6 && 1 <= @target_segment < @run_segments &&
             1 <= @valid_text_tokens <= 1000 &&
             {@conditioning_manifest_sha256, @conditioning_payload_sha256, @prompt_sha256}.all? { |sha| /\A[0-9a-f]{64}\z/.matches?(sha) }
        raise ArgumentError.new("Prompt edit requires a noninitial active target and authenticated bounded conditioning")
      end
    end

    def identity
      {schema: "cogni_ml.hy_nomas_segment_prompt_edit/v1", run_segments: @run_segments,
       target_segment: @target_segment,
       conditioning_manifest_sha256: @conditioning_manifest_sha256,
       conditioning_payload_sha256: @conditioning_payload_sha256,
       prompt_sha256: @prompt_sha256, valid_text_tokens: @valid_text_tokens,
       following_prompt: "original_as_new_text_turn"}
    end

    def edited_segment?(segment : Int32) : Bool
      raise ArgumentError.new("Prompt edit segment is outside the active run") unless 0 <= segment < @run_segments
      segment == @target_segment
    end

    def text_tokens(original : Int32) : Array(Int32)
      raise ArgumentError.new("Original prompt token count is outside the admitted range") unless 1 <= original <= 1000
      Array(Int32).new(@run_segments) { |segment| edited_segment?(segment) ? @valid_text_tokens : original }
    end

    def cache_text : Array(Bool)
      Array(Bool).new(@run_segments) { |segment| segment == 0 || segment == @target_segment || segment == @target_segment + 1 }
    end

    def bind_contract(base : String, next_segment : Int32) : String
      unless 1 <= next_segment < @run_segments
        raise ArgumentError.new("Prompt edit contract requires an extendable boundary")
      end
      return base if next_segment <= @target_segment
      {schema: "cogni_ml.hy_nomas_prompt_edited_sequence_boundary/v1",
       base_contract_sha256: Digest::SHA256.hexdigest(base), prompt_edit: identity}.to_json
    end
  end
end
