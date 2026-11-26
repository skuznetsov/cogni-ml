require "digest/sha256"
require "json"
require "../../src/ml/core/buffer"

module ML::Video::HyNomasReferenceImage
  SCHEMA                 = "cogni_ml.hy_nomas_reference_image/v1"
  SCHEMA_VERSION         = 1
  UPSTREAM_REVISION      = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
  CHECKPOINT_SHA256      = "ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"
  WIDTH                  = 512
  HEIGHT                 = 288
  FRAMES                 = 120
  PIXEL_CHANNELS         =   3
  PIXEL_BYTES            = WIDTH.to_i64 * HEIGHT * PIXEL_CHANNELS
  FIT_POLICY             = "exif-rgb-lanczos-pad-black-center"
  MAX_MANIFEST_BYTES     = 1_i64 * 1024 * 1024
  EXPECTED_MANIFEST_KEYS = %w(checkpoint_sha256 encoder_safetensors_sha256 fit_policy frames height image_path image_sha256 payload payload_bytes payload_sha256 schema schema_version upstream_revision width)

  # Resolve a CLI path only after rejecting a final-component symlink. Calling
  # File.realpath first would erase that evidence before Inputs can enforce its
  # non-symlink file contract.
  def self.canonical_manifest_argument!(path : String) : String
    candidate = Path[path].absolute? ? path : File.join(Dir.current, path)
    if File.symlink?(candidate)
      raise ArgumentError.new("reference manifest path must not be a symlink")
    end
    unless File.file?(candidate)
      raise ArgumentError.new("reference manifest must be a regular file")
    end
    resolved = File.realpath(candidate)
    unless File.file?(candidate) && !File.symlink?(candidate) && File.realpath(candidate) == resolved
      raise ArgumentError.new("reference manifest path changed while resolving")
    end
    resolved
  end

  # Authenticated setup input for the one fixed T0 reference image. The source
  # image path and digest are provenance from preparation; runtime consumes the
  # already-bound RGB8 payload and does not require the original file to remain.
  class Inputs
    getter manifest_path : String
    getter manifest_sha256 : String
    getter image_path : String
    getter image_sha256 : String
    getter payload_path : String
    getter payload_sha256 : String
    getter encoder_sha : String
    getter pixel_bytes : Int64

    def initialize(path : String, trusted_manifest_sha256 : String,
                   expected_encoder_safetensors_sha256 : String)
      unless valid_sha256?(trusted_manifest_sha256) && valid_sha256?(expected_encoder_safetensors_sha256)
        raise ArgumentError.new("reference manifest and encoder SHA-256 pins are required")
      end
      @manifest_path = canonical_regular_file!(path, "reference manifest")
      raw = read_stable_manifest!(@manifest_path)
      @manifest_sha256 = Digest::SHA256.hexdigest(raw)
      unless @manifest_sha256 == trusted_manifest_sha256
        raise ArgumentError.new("reference manifest trusted SHA-256 mismatch")
      end

      document = begin
        JSON.parse(raw)
      rescue ex : JSON::ParseException
        raise ArgumentError.new("reference image manifest is malformed")
      end
      validate_document!(document, expected_encoder_safetensors_sha256)
      @image_path = document["image_path"].as_s
      @image_sha256 = document["image_sha256"].as_s
      @payload_sha256 = document["payload_sha256"].as_s
      @encoder_sha = document["encoder_safetensors_sha256"].as_s
      @pixel_bytes = document["payload_bytes"].as_i64
      payload_name = document["payload"].as_s
      unless payload_name.matches?(/\A[A-Za-z0-9_.-]+\z/) && payload_name != "." && payload_name != ".."
        raise ArgumentError.new("reference RGB payload must be a sibling basename")
      end
      @payload_path = File.join(File.dirname(@manifest_path), payload_name)
      canonical_regular_file!(@payload_path, "reference RGB payload")
      unless File.size(@payload_path) == @pixel_bytes
        raise ArgumentError.new("reference RGB payload extent mismatch")
      end
      verify_payload!
    end

    # Revalidate the bound pixel payload before image admission or upload. This
    # check covers bytes the DCAE encoder will actually consume, not metadata
    # fields merely because they were present in the manifest. The manifest is
    # also rechecked so receipts cannot silently outlive their authenticated
    # provenance file.
    def verify_payload! : Nil
      verify_manifest!
      info = File.info(@payload_path)
      unless File.file?(@payload_path) && !File.symlink?(@payload_path) &&
             File.realpath(@payload_path) == @payload_path && info.size == @pixel_bytes
        raise ArgumentError.new("reference RGB payload path or extent changed")
      end
      digest = Digest::SHA256.new
      scratch = Bytes.new(64 * 1024)
      File.open(@payload_path) do |io|
        remaining = @pixel_bytes
        while remaining > 0
          chunk = scratch[0, Math.min(remaining, scratch.size.to_i64).to_i32]
          io.read_fully(chunk)
          digest.update(chunk)
          remaining -= chunk.size
        end
        unless io.read_byte.nil?
          raise ArgumentError.new("reference RGB payload grew while verifying")
        end
      end
      after = File.info(@payload_path)
      unless info.same_file?(after) && info.size == after.size &&
             info.modification_time == after.modification_time &&
             digest.hexfinal == @payload_sha256
        raise ArgumentError.new("reference RGB payload SHA-256 or file identity mismatch")
      end
    end

    # Stream RGB8 directly into a caller-allocated shared Metal buffer. The
    # caller may consume the destination only after this method returns.
    def upload!(destination : ML::MetalBuffer) : Nil
      unless destination.valid? && destination.storage_mode.shared? && destination.size == @pixel_bytes
        raise ArgumentError.new("reference RGB destination must be exact-size shared storage")
      end
      verify_payload!
      info = File.info(@payload_path)
      unless File.file?(@payload_path) && !File.symlink?(@payload_path) &&
             File.realpath(@payload_path) == @payload_path && info.size == @pixel_bytes
        raise ArgumentError.new("reference RGB payload path or extent changed before upload")
      end
      bytes = Bytes.new(destination.contents.as(UInt8*), @pixel_bytes.to_i32)
      File.open(@payload_path) { |io| io.read_fully(bytes) }
      after = File.info(@payload_path)
      unless info.same_file?(after) && info.size == after.size &&
             info.modification_time == after.modification_time &&
             Digest::SHA256.hexdigest(bytes) == @payload_sha256
        raise ArgumentError.new("reference RGB payload changed during upload")
      end
    end

    private def validate_document!(document : JSON::Any,
                                   expected_encoder_safetensors_sha256 : String) : Nil
      unless document.as_h.keys.sort == EXPECTED_MANIFEST_KEYS.sort &&
             document["schema"].as_s == SCHEMA &&
             document["schema_version"].as_i == SCHEMA_VERSION &&
             document["upstream_revision"].as_s == UPSTREAM_REVISION &&
             document["width"].as_i == WIDTH && document["height"].as_i == HEIGHT &&
             document["frames"].as_i == FRAMES && document["fit_policy"].as_s == FIT_POLICY &&
             document["payload_bytes"].as_i64 == PIXEL_BYTES &&
             document["checkpoint_sha256"].as_s == CHECKPOINT_SHA256 &&
             valid_sha256?(document["image_sha256"].as_s) &&
             valid_sha256?(document["payload_sha256"].as_s) &&
             document["encoder_safetensors_sha256"].as_s == expected_encoder_safetensors_sha256
        raise ArgumentError.new("reference image schema, geometry, or encoder identity mismatch")
      end
      source_image_path = document["image_path"].as_s
      payload_name = document["payload"].as_s
      unless !source_image_path.empty? && Path[source_image_path].absolute?
        raise ArgumentError.new("reference source image path must be absolute provenance")
      end
      unless payload_name.matches?(/\A[A-Za-z0-9_.-]+\z/) && payload_name != "." && payload_name != ".."
        raise ArgumentError.new("reference RGB payload must be a sibling basename")
      end
    rescue ex : TypeCastError | KeyError
      raise ArgumentError.new("reference image manifest has a missing or malformed field")
    end

    private def canonical_regular_file!(path : String, label : String) : String
      unless Path[path].absolute? && File.file?(path) && !File.symlink?(path) &&
             File.realpath(path) == path
        raise ArgumentError.new("#{label} must be a canonical absolute regular file")
      end
      path
    end

    private def verify_manifest! : Nil
      canonical_regular_file!(@manifest_path, "reference manifest")
      raw = read_stable_manifest!(@manifest_path)
      unless Digest::SHA256.hexdigest(raw) == @manifest_sha256
        raise ArgumentError.new("reference manifest changed after authentication")
      end
    end

    private def read_stable_manifest!(path : String) : String
      canonical_regular_file!(path, "reference manifest")
      info = File.info(path)
      unless info.size > 0 && info.size <= MAX_MANIFEST_BYTES
        raise ArgumentError.new("reference manifest exceeds its size bound")
      end
      raw = File.read(path)
      after = File.info(path)
      canonical_regular_file!(path, "reference manifest")
      unless info.same_file?(after) && info.size == after.size &&
             info.modification_time == after.modification_time && raw.bytesize == info.size
        raise ArgumentError.new("reference manifest changed while reading")
      end
      raw
    end

    private def valid_sha256?(value : String) : Bool
      /\A[0-9a-f]{64}\z/.matches?(value)
    end
  end
end
