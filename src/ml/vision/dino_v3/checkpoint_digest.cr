require "digest/sha256"

require "./checkpoint_manifest"
require "./checkpoint_payload"

module ML::Vision::DinoV3
  class CheckpointDigestError < CheckpointPayloadError
  end

  # A receipt is issued only after the complete local file has been hashed
  # against a trusted expected digest. It is not a signature or a concurrent
  # file-lock; same-size replacement after verification remains outside this
  # boundary.
  class CheckpointDigestReceipt
    getter byte_length : Int64

    @path : String
    @sha256 : String

    private def initialize(
      path : String,
      @byte_length : Int64,
      sha256 : String,
    )
      @path = path.dup
      @sha256 = sha256.dup
    end

    def self.verify(
      inventory : CheckpointInventory,
      *,
      expected_sha256 : String,
    ) : CheckpointDigestReceipt
      validate_digest!(expected_sha256)
      actual_size = File.size(inventory.path)
      unless actual_size == inventory.file_byte_length
        raise CheckpointDigestError.new(
          "checkpoint file changed before SHA-256 verification"
        )
      end

      actual_sha256 = Digest::SHA256.new.file(inventory.path).hexfinal
      unless actual_sha256 == expected_sha256
        raise CheckpointDigestError.new(
          "checkpoint SHA-256 mismatch: expected #{expected_sha256}, got #{actual_sha256}"
        )
      end
      new(inventory.path, actual_size, actual_sha256)
    rescue ex : CheckpointDigestError
      raise ex
    rescue ex : File::Error
      raise CheckpointDigestError.new(
        "cannot hash checkpoint file: #{ex.message}"
      )
    end

    def self.verify(
      inventory : CheckpointInventory,
      *,
      manifest : CheckpointManifest,
    ) : CheckpointDigestReceipt
      unless manifest.weights_path == CheckpointManifest::PINNED_WEIGHTS_PATH &&
             manifest.weights_format == CheckpointManifest::PINNED_WEIGHTS_FORMAT &&
             manifest.weights_byte_length == inventory.file_byte_length
        raise CheckpointDigestError.new(
          "checkpoint manifest does not match the inspected inventory"
        )
      end
      verify(inventory, expected_sha256: manifest.weights_sha256)
    end

    def path : String
      @path.dup
    end

    def sha256 : String
      @sha256.dup
    end

    def matches?(inventory : CheckpointInventory) : Bool
      @path == inventory.path && @byte_length == inventory.file_byte_length
    end

    private def self.validate_digest!(digest : String) : Nil
      unless digest =~ /\A[0-9a-f]{64}\z/
        raise CheckpointDigestError.new(
          "checkpoint SHA-256 must be 64 lowercase hexadecimal characters"
        )
      end
    end
  end
end
