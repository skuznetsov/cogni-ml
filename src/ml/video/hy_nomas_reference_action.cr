require "digest/sha256"
require "./hy_nomas_action_conditioning"

module ML::Video
  # A fixed T0 reference followed by one separately authenticated T1 action
  # condition. This is deliberately independent of segment prompt edits and
  # their checkpoint lineage.
  struct HyNomasReferenceAction
    getter run_segments : Int32
    getter conditioning_manifest_sha256 : String
    getter conditioning_payload_sha256 : String
    getter prompt_sha256 : String
    getter valid_text_tokens : Int32
    getter action_conditioning : HyNomasActionConditioning::Profile?

    def initialize(@run_segments : Int32,
                   @conditioning_manifest_sha256 : String,
                   @conditioning_payload_sha256 : String,
                   @prompt_sha256 : String,
                   @valid_text_tokens : Int32,
                   @action_conditioning : HyNomasActionConditioning::Profile? = nil)
      unless @run_segments == 2 && 1 <= @valid_text_tokens <= 1000 &&
             {@conditioning_manifest_sha256, @conditioning_payload_sha256, @prompt_sha256}.all? { |sha| /\A[0-9a-f]{64}\z/.matches?(sha) }
        raise ArgumentError.new("Reference action requires exactly two segments and authenticated bounded conditioning")
      end
    end

    def identity
      base = {schema:                       "cogni_ml.hy_nomas_reference_action/v1",
              run_segments:                 @run_segments,
              conditioning_manifest_sha256: @conditioning_manifest_sha256,
              conditioning_payload_sha256:  @conditioning_payload_sha256,
              prompt_sha256:                @prompt_sha256,
              valid_text_tokens:            @valid_text_tokens,
              cache_text:                   cache_text}
      if profile = @action_conditioning
        base.merge({schema:              "cogni_ml.hy_nomas_reference_action/v2",
                    action_conditioning: profile.identity})
      else
        base
      end
    end

    # T0 retains its caption length. The T1 action is a new text turn whose
    # own valid prefix is cached with the same OncePerScene history policy.
    def text_tokens(caption_valid_text_tokens : Int32) : Array(Int32)
      unless 1 <= caption_valid_text_tokens <= 1000
        raise ArgumentError.new("Reference caption token count is outside the admitted range")
      end
      [caption_valid_text_tokens, @valid_text_tokens]
    end

    def cache_text : Array(Bool)
      [true, true]
    end

    def query_text_tokens(caption_valid_text_tokens : Int32) : Int32
      Math.max(caption_valid_text_tokens, @valid_text_tokens)
    end
  end
end
