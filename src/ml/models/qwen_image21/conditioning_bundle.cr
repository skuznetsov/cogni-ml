require "json"
require "digest/sha256"

# Versioned text-to-image boundary between the reference Qwen3-VL encoder and
# the native Qwen-Image 2.1 Metal denoiser. No model weights live in this file.
module ML::GGUF
  class QwenImage21ConditioningBundle
    PAYLOAD_FILE     = "qwen_image21_conditioning.bin"
    CONTEXT_DIM      = 4096
    LATENT_CHANNELS  =   64
    VAE_SCALE_FACTOR =   16

    getter prompt : String
    getter seed : Int64
    getter model_revision : String
    getter conditioning_payload_sha256 : String
    getter image_width : Int32
    getter image_height : Int32
    getter latent_width : Int32
    getter latent_height : Int32
    getter img_shapes : Array(StaticArray(Int32, 3))
    getter encoder_hidden_states : Array(Float32)
    getter encoder_hidden_states_mask : Array(Bool)
    getter encoder_img_mask : Array(Bool)
    getter initial_target_latents : Array(Float32)

    def initialize(
      @prompt, @seed, @model_revision, @conditioning_payload_sha256, @image_width, @image_height,
      @latent_width, @latent_height, @img_shapes, @encoder_hidden_states,
      @encoder_hidden_states_mask, @encoder_img_mask, @initial_target_latents,
    )
    end

    def self.load(manifest_path : String) : self
      manifest = File.read(manifest_path)
      payload_path = File.join(File.dirname(manifest_path), PAYLOAD_FILE)
      raise ArgumentError.new("conditioning payload not found: #{payload_path}") unless File.file?(payload_path)
      parse(manifest, File.read(payload_path).to_slice)
    end

    def self.parse(manifest_json : String, payload : Bytes) : self
      manifest = JSON.parse(manifest_json)
      expect(manifest["schema"].as_s == "qwen-image21-conditioning", "unsupported conditioning schema")
      expect(manifest["schema_version"].as_i == 1, "unsupported conditioning schema version")
      model = manifest["model"]
      expect(model["repo"].as_s == "Qwen/Qwen-Image-2.1", "wrong conditioning model")
      revision = model["revision"].as_s
      expect(!revision.empty?, "missing conditioning model revision")
      prompt = manifest["prompt"].as_s
      expect(!prompt.empty?, "empty conditioning prompt")
      seed = manifest["noise"]["seed"].as_i64

      image = manifest["image"]
      width = image["width"].as_i
      height = image["height"].as_i
      expect(width >= 32 && height >= 32 && width <= 4096 && height <= 4096, "image dimensions out of range")
      expect(width.divisible_by?(32) && height.divisible_by?(32), "image dimensions must be multiples of 32")
      expect(image["vae_scale_factor"].as_i == VAE_SCALE_FACTOR, "wrong VAE spatial factor")
      latent_width = image["latent_width"].as_i
      latent_height = image["latent_height"].as_i
      expect(latent_width == width // VAE_SCALE_FACTOR && latent_height == height // VAE_SCALE_FACTOR,
        "Qwen-Image 2.1 latent grid must be unpatched image/16")
      shapes = image["img_shapes"].as_a
      expect(shapes.size == 1, "text-to-image requires exactly one target shape")
      target_shape = shapes.first.as_a.map(&.as_i)
      expect(target_shape == [1, latent_height, latent_width], "target img_shapes mismatch")
      target_tokens = latent_width * latent_height
      expect(target_tokens.divisible_by?(4), "target image must use complete four-token VLM slots")

      tensors = manifest["tensors"]
      expect(manifest["payload_file"].as_s == PAYLOAD_FILE, "conditioning payload file mismatch")
      expect(manifest["payload_nbytes"].as_i == payload.size, "conditioning payload length mismatch")
      payload_sha256 = Digest::SHA256.hexdigest(payload)
      expect(manifest["payload_sha256"].as_s == payload_sha256, "conditioning payload checksum mismatch")
      state_descriptor = tensors["encoder_hidden_states"]
      state_shape = state_descriptor["shape"].as_a.map(&.as_i)
      expect(state_shape.size == 2 && state_shape[1] == CONTEXT_DIM, "Qwen3-VL context width mismatch")
      text_tokens = state_shape[0]
      expect(text_tokens > 0 && text_tokens <= 1024, "text sequence length out of range")
      offset = 0
      state_bytes = tensor_bytes(state_descriptor, "float32-le", [text_tokens, CONTEXT_DIM], offset)
      offset += state_bytes
      valid_descriptor = tensors["encoder_hidden_states_mask"]
      valid_bytes = tensor_bytes(valid_descriptor, "uint8", [text_tokens], offset)
      offset += valid_bytes
      image_descriptor = tensors["encoder_img_mask"]
      image_bytes = tensor_bytes(image_descriptor, "uint8", [text_tokens], offset)
      offset += image_bytes
      latent_descriptor = tensors["initial_target_latents"]
      latent_bytes = tensor_bytes(latent_descriptor, "float32-le", [target_tokens, LATENT_CHANNELS], offset)
      offset += latent_bytes
      expect(payload.size == offset, "conditioning payload length mismatch")

      hidden = read_floats(payload[0, state_bytes])
      valid = read_mask(payload[state_bytes, valid_bytes])
      image_mask = read_mask(payload[state_bytes + valid_bytes, image_bytes])
      expect(valid.any?, "conditioning has no valid text tokens")
      expect(image_mask.none?, "text-to-image conditioning contains image placeholders")
      latents = read_floats(payload[state_bytes + valid_bytes + image_bytes, latent_bytes])
      new(prompt, seed, revision, payload_sha256, width, height, latent_width, latent_height,
        [StaticArray[1, latent_height, latent_width]], hidden, valid, image_mask, latents)
    end

    private def self.tensor_bytes(descriptor : JSON::Any, dtype : String, shape : Array(Int32), offset : Int32) : Int32
      expect(descriptor["dtype"].as_s == dtype, "conditioning tensor dtype mismatch")
      expect(descriptor["shape"].as_a.map(&.as_i) == shape, "conditioning tensor shape mismatch")
      expect(descriptor["offset_bytes"].as_i == offset, "conditioning tensor offset mismatch")
      expected_bytes = shape.product * (dtype == "float32-le" ? 4 : 1)
      expect(descriptor["nbytes"].as_i == expected_bytes, "conditioning tensor byte size mismatch")
      expected_bytes
    end

    private def self.read_floats(data : Bytes) : Array(Float32)
      Array(Float32).new(data.size // 4) do |index|
        value = IO::ByteFormat::LittleEndian.decode(Float32, data[index * 4, 4])
        expect(value.finite?, "conditioning contains a non-finite float")
        value
      end
    end

    private def self.read_mask(data : Bytes) : Array(Bool)
      Array(Bool).new(data.size) do |index|
        value = data[index]
        expect(value <= 1, "conditioning mask is not binary")
        value == 1
      end
    end

    private def self.expect(condition : Bool, message : String) : Nil
      raise ArgumentError.new(message) unless condition
    end
  end
end
