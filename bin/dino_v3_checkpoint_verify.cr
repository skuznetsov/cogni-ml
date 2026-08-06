require "json"
require "option_parser"

require "../src/ml/vision/dino_v3"

module ML::Vision::DinoV3
  class CheckpointVerificationCLI
    DEFAULT_ROLE = "embedding.cls_token"

    def self.run : Nil
      checkpoint_path, role = parse_args
      certificate = load_certificate
      runtime = RuntimeAdapter.new(certificate)
      manifest = CheckpointManifest.parse(
        File.read(fixture_path("trellis2/dino_v3_checkpoint_manifest_v1.json")),
        certificate: certificate
      )

      inventory = CheckpointInventory.load(checkpoint_path, manifest: manifest)
      semantic = SemanticInventory.bind(
        inventory,
        certificate: certificate,
        runtime: runtime
      )
      receipt = CheckpointDigestReceipt.verify(inventory, manifest: manifest)
      decoded = CheckpointF32Decoder.new(
        inventory,
        semantic: semantic,
        digest: receipt
      ).decode(role)

      report = JSON.build do |json|
        json.object do
          json.field("checkpoint", checkpoint_path)
          json.field("bytes", inventory.file_byte_length)
          json.field("header_bytes", inventory.header_byte_length)
          json.field("tensor_count", inventory.tensors.size)
          json.field("semantic_tensor_count", semantic.tensor_count)
          json.field("sha256", receipt.sha256)
          json.field("role", decoded.role)
          json.field("name", decoded.name)
          json.field("shape") do
            json.array { decoded.shape.each { |dimension| json.number(dimension) } }
          end
          json.field("decoded_values", decoded.values.size)
        end
      end
      puts report
    end

    private def self.parse_args : Tuple(String, String)
      checkpoint_path : String? = nil
      role = DEFAULT_ROLE

      OptionParser.parse do |parser|
        parser.banner = "Usage: dino_v3_checkpoint_verify --checkpoint PATH [--role ROLE]"
        parser.on("-c PATH", "--checkpoint=PATH", "materialized model.safetensors path") do |path|
          checkpoint_path = path
        end
        parser.on("-r ROLE", "--role=ROLE", "one semantic F32 role (default: #{DEFAULT_ROLE})") do |value|
          role = value
        end
        parser.on("-h", "--help", "show this help") do
          puts parser
          exit
        end
      end

      path = checkpoint_path
      unless path
        STDERR.puts "missing --checkpoint PATH"
        exit 2
      end
      {path, role}
    rescue ex : OptionParser::InvalidOption
      STDERR.puts ex.message
      exit 2
    end

    private def self.load_certificate : ConfigCertificate
      ConfigCertificate.parse(
        File.read(fixture_path("trellis2/dino_v3_config_certificate_v1.json")),
        source_model: ConfigCertificate::PINNED_SOURCE_MODEL,
        source_revision: ConfigCertificate::PINNED_SOURCE_REVISION
      )
    end

    private def self.fixture_path(relative : String) : String
      File.expand_path(File.join(__DIR__, "..", "spec", "fixtures", relative))
    end
  end
end

begin
  ML::Vision::DinoV3::CheckpointVerificationCLI.run
rescue ex : Exception
  STDERR.puts "dino_v3_checkpoint_verify: #{ex.message}"
  exit 1
end
