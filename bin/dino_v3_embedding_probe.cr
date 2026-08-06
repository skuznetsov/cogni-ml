require "json"
require "option_parser"

require "../src/ml/vision/dino_v3"

module ML::Vision::DinoV3
  class EmbeddingProbeCLI
    def self.run : Nil
      checkpoint_path, max_parameter_bytes, preflight_edge, max_multiply_adds = parse_args
      certificate = load_certificate
      runtime = RuntimeAdapter.new(certificate)
      if edge = preflight_edge
        plan = DinoV3EmbeddingCaller.preflight(
          certificate,
          runtime,
          edge,
          max_multiply_adds: max_multiply_adds
        )
        report = JSON.build do |json|
          json.object do
            json.field("mode", "preflight")
            json.field("input_edge", plan.input_edge)
            json.field("patch_edge", plan.patch_edge)
            json.field("patch_count", plan.patch_count)
            json.field("input_bytes", plan.input_bytes)
            json.field("output_elements", plan.output_elements)
            json.field("output_bytes", plan.output_bytes)
            json.field("multiply_adds", plan.multiply_adds)
            json.field("max_input_bytes", plan.max_input_bytes)
            json.field("max_output_bytes", plan.max_output_bytes)
            json.field("max_multiply_adds", plan.max_multiply_adds)
            json.field("config_hidden_size", certificate.hidden_size)
            json.field("config_register_tokens", certificate.num_register_tokens)
            json.field("payload_loaded", false)
            json.field("forward", "not-run")
            json.field("blocks_loaded", 0)
          end
        end
        puts report
        return
      end

      path = checkpoint_path
      unless path
        STDERR.puts "missing --checkpoint PATH or --preflight-edge EDGE"
        exit 2
      end
      manifest = CheckpointManifest.parse(
        File.read(fixture_path("trellis2/dino_v3_checkpoint_manifest_v1.json")),
        certificate: certificate
      )
      inventory = CheckpointInventory.load(path, manifest: manifest)
      semantic = SemanticInventory.bind(
        inventory,
        certificate: certificate,
        runtime: runtime
      )
      receipt = CheckpointDigestReceipt.verify(inventory, manifest: manifest)
      caller = DinoV3EmbeddingCaller.from_checkpoint(
        inventory,
        certificate: certificate,
        runtime: runtime,
        semantic: semantic,
        digest: receipt,
        max_parameter_bytes: max_parameter_bytes
      )

      report = JSON.build do |json|
        json.object do
          json.field("checkpoint", path)
          json.field("bytes", inventory.file_byte_length)
          json.field("header_bytes", inventory.header_byte_length)
          json.field("tensor_count", inventory.tensors.size)
          json.field("semantic_tensor_count", semantic.tensor_count)
          json.field("sha256", receipt.sha256)
          json.field("required_roles") do
            json.array { DinoV3EmbeddingCaller::REQUIRED_ROLES.each { |role| json.string(role) } }
          end
          json.field("parameter_bytes", caller.parameters.parameter_byte_length)
          json.field("parameter_f32le_sha256", caller.parameters.f32le_sha256)
          json.field("config_hidden_size", caller.config.hidden_size)
          json.field("config_register_tokens", caller.config.num_register_tokens)
          json.field("forward", "not-run")
          json.field("blocks_loaded", 0)
        end
      end
      puts report
    end

    private def self.parse_args : Tuple(String?, Int64, Int32?, Int64)
      checkpoint_path : String? = nil
      max_parameter_bytes = DinoV3EmbeddingCaller::MAX_PARAMETER_BYTES
      preflight_edge : Int32? = nil
      max_multiply_adds = EmbeddingCPU::MAX_MULTIPLY_ADDS

      OptionParser.parse do |parser|
        parser.banner = "Usage: dino_v3_embedding_probe --checkpoint PATH [--max-parameter-bytes BYTES] | --preflight-edge EDGE [--max-multiply-adds COUNT]"
        parser.on("-c PATH", "--checkpoint=PATH", "materialized model.safetensors path") do |path|
          checkpoint_path = path
        end
        parser.on("-b BYTES", "--max-parameter-bytes=BYTES", "aggregate parameter budget") do |value|
          begin
            max_parameter_bytes = value.to_i64
          rescue
            raise OptionParser::InvalidOption.new("invalid --max-parameter-bytes value")
          end
        end
        parser.on("-p EDGE", "--preflight-edge=EDGE", "allocation-free geometry/work preflight") do |value|
          begin
            preflight_edge = value.to_i32
          rescue
            raise OptionParser::InvalidOption.new("invalid --preflight-edge value")
          end
        end
        parser.on("-m COUNT", "--max-multiply-adds=COUNT", "operation budget for preflight") do |value|
          begin
            max_multiply_adds = value.to_i64
          rescue
            raise OptionParser::InvalidOption.new("invalid --max-multiply-adds value")
          end
        end
        parser.on("-h", "--help", "show this help") do
          puts parser
          exit
        end
      end

      if checkpoint_path && preflight_edge
        STDERR.puts "choose either --checkpoint PATH or --preflight-edge EDGE"
        exit 2
      end
      unless checkpoint_path || preflight_edge
        STDERR.puts "missing --checkpoint PATH or --preflight-edge EDGE"
        exit 2
      end
      {checkpoint_path, max_parameter_bytes, preflight_edge, max_multiply_adds}
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
  ML::Vision::DinoV3::EmbeddingProbeCLI.run
rescue ex : Exception
  STDERR.puts "dino_v3_embedding_probe: #{ex.message}"
  exit 1
end
