require "json"

module HyNomasLatentDecodeAdmission
  extend self

  ROPE_SHAPE              = [1, 5760, 1, 128]
  ROPE_NBYTES             = 2_949_120_i64
  DEFAULT_GEOMETRY_FIELDS = JSON.parse(%({
    "dit_latent_shape": [1, 32, 30, 9, 16],
    "dit_rope_grid": [30, 9, 16],
    "dit_rope_axis_dims": [16, 56, 56],
    "dit_rope_theta": 256.0,
    "qwen_shape": [1, 1280, 3584],
    "qwen_state_index": 26,
    "decoder_latent_grid": [30, 9, 16],
    "decoder_noise_shape": [1, 3, 120, 288, 512],
    "decoder_noise_piece_latents": 1,
    "decoder_noise_piece_frames": 4,
    "decoder_rope_grid": [10, 18, 32],
    "decoder_rope_shape": [1, 5760, 1, 128],
    "decoder_rope_sections": [64, 32, 32],
    "decoder_rope_theta": [1000000.0, 10000.0, 10000.0],
    "decoder_local_attention_tile_tokens": 8,
    "decoder_local_attention_tile_pixels": 128
  })).as_h

  alias RopeRecordIdentity = Tuple(String, Array(Int32), Int64, String)
  alias ContinuationIdentity = Tuple(String, String)

  # Recovered caches have a new provenance manifest even when their tensor
  # payload is identical. Never silently replace the historical default pin.
  def resolve_decoder_cache!(path : String, sha : String) : ContinuationIdentity?
    unless path.empty? == sha.empty?
      raise ArgumentError.new("--decoder-cache and --decoder-cache-sha must be supplied together")
    end
    return nil if path.empty?
    unless path.starts_with?("/") && sha.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("decoder cache override requires an absolute path and lowercase SHA-256")
    end
    {path, sha}
  end

  def resolve_continuation!(latent_frames : Int32, default_base : Bool,
                            path_override : String, sha_override : String,
                            default_path : String, default_sha : String) : ContinuationIdentity?
    has_path = !path_override.empty?
    has_sha = !sha_override.empty?
    unless has_path == has_sha
      raise ArgumentError.new("--continuation and --continuation-sha must be supplied together")
    end

    if latent_frames == 30
      raise ArgumentError.new("continuation overrides apply only to 60/90-frame decodes") if has_path
      return nil
    end

    if has_path
      unless path_override.starts_with?("/") && sha_override.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new("continuation override requires an absolute path and lowercase SHA-256")
      end
      {path_override, sha_override}
    elsif default_base
      {default_path, default_sha}
    else
      raise ArgumentError.new("non-default inputs require explicit --continuation and --continuation-sha")
    end
  end

  # The imported previous-window route is intentionally limited to decoding
  # one current 30-latent segment with exactly one previous 30-latent segment
  # available for its context. It has its own authenticated noise input and
  # cannot be mixed with the full-timeline continuation bundle.
  def resolve_previous_window!(latent_frames : Int32, latent_bank_frames : Int32,
                               continuation_path : String, continuation_sha : String,
                               previous_latent_path : String, previous_latent_sha : String,
                               previous_noise_path : String, previous_noise_sha : String) : Tuple(String, String, String, String)?
    values = {previous_latent_path, previous_latent_sha, previous_noise_path, previous_noise_sha}
    return nil if values.all?(&.empty?)
    unless values.all? { |value| !value.empty? }
      raise ArgumentError.new("all four previous-window options must be supplied together")
    end
    unless latent_frames == 30 && latent_bank_frames == 30
      raise ArgumentError.new("previous-window context supports only a 30-frame current latent with a 30-frame latent bank")
    end
    unless continuation_path.empty? && continuation_sha.empty?
      raise ArgumentError.new("previous-window context cannot be combined with --continuation")
    end
    unless previous_latent_path.starts_with?("/") && previous_noise_path.starts_with?("/")
      raise ArgumentError.new("previous-window latent and noise paths must be absolute")
    end
    unless previous_latent_sha.matches?(/\A[0-9a-f]{64}\z/) &&
           previous_noise_sha.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("previous-window latent and noise identities require lowercase SHA-256 digests")
    end
    {previous_latent_path, previous_latent_sha, previous_noise_path, previous_noise_sha}
  end

  def validate_base_binding!(continuation_base_path : String, continuation_base_sha : String,
                             input_path : String, input_sha : String) : Nil
    unless continuation_base_path == input_path && continuation_base_sha == input_sha
      raise ArgumentError.new("long decoder noise bundle does not bind the supplied base input bundle")
    end
  end

  def validate_default_geometry!(geometry : JSON::Any, qwen_source : JSON::Any,
                                 valid_text_tokens : Int32) : Nil
    expected_keys = DEFAULT_GEOMETRY_FIELDS.keys + %w(qwen_crop qwen_valid_tokens)
    unless geometry.as_h.keys.sort == expected_keys.sort
      raise ArgumentError.new("conditioned input geometry inventory differs from fixed 512x288 v2")
    end
    DEFAULT_GEOMETRY_FIELDS.each do |name, expected|
      unless geometry[name] == expected
        raise ArgumentError.new("conditioned input geometry mismatch: #{name}")
      end
    end
    unless geometry["qwen_crop"] == qwen_source["crop"] &&
           geometry["qwen_valid_tokens"].as_i == valid_text_tokens
      raise ArgumentError.new("conditioned input Qwen geometry differs from the authenticated crop and mask")
    end
  end

  def validate_matching_rope_records!(name : String, base : RopeRecordIdentity,
                                      continuation : RopeRecordIdentity) : Nil
    unless base == continuation
      raise ArgumentError.new("base/continuation decoder RoPE record mismatch: #{name}")
    end
    unless base[0] == "F32" && base[1] == ROPE_SHAPE && base[2] == ROPE_NBYTES &&
           base[3].matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("base decoder RoPE geometry or digest mismatch: #{name}")
    end
  end
end
