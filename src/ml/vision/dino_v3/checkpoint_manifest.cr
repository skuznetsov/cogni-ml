require "json"

require "./config"
require "../../three_d/trellis2/strict_json"

module ML::Vision::DinoV3
  # A checkpoint manifest is provenance only. It identifies the exact gated
  # artifact metadata that a future loader may materialize; it never downloads,
  # opens, or executes a checkpoint.
  class CheckpointManifestError < ConfigError
  end

  class CheckpointManifest
    SCHEMA = "cogni-ml/vision/dino-v3/checkpoint-manifest/v1"

    PINNED_CONFIG_PATH         = "config.json"
    PINNED_CONFIG_BYTE_LENGTH  = 745_i64
    PINNED_WEIGHTS_PATH        = "model.safetensors"
    PINNED_WEIGHTS_FORMAT      = "safetensors"
    PINNED_WEIGHTS_BYTE_LENGTH = 1_212_559_808_i64
    PINNED_WEIGHTS_SHA256      = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"

    getter config_byte_length : Int64
    getter weights_byte_length : Int64

    @schema : String
    @model : String
    @revision : String
    @access_mode : String
    @license_name : String
    @license_link : String
    @config_path : String
    @config_sha256 : String
    @weights_path : String
    @weights_format : String
    @weights_sha256 : String

    private def initialize(
      @schema : String,
      model : String,
      revision : String,
      access_mode : String,
      license_name : String,
      license_link : String,
      config_path : String,
      @config_byte_length : Int64,
      config_sha256 : String,
      weights_path : String,
      weights_format : String,
      @weights_byte_length : Int64,
      weights_sha256 : String,
    )
      @schema = schema.dup
      @model = model.dup
      @revision = revision.dup
      @access_mode = access_mode.dup
      @license_name = license_name.dup
      @license_link = license_link.dup
      @config_path = config_path.dup
      @config_sha256 = config_sha256.dup
      @weights_path = weights_path.dup
      @weights_format = weights_format.dup
      @weights_sha256 = weights_sha256.dup
    end

    def schema : String
      @schema.dup
    end

    def self.parse(
      source : String,
      *,
      certificate : ConfigCertificate,
    ) : CheckpointManifest
      validate_certificate!(certificate)
      root = strict_object(source, "checkpoint manifest")
      expect_exact_keys!(
        root,
        ["schema", "model", "revision", "access_mode", "license", "config", "weights"],
        "checkpoint manifest"
      )

      schema = string(root["schema"], "schema")
      unless schema == SCHEMA
        raise CheckpointManifestError.new("schema is not the pinned checkpoint manifest")
      end

      model = string(root["model"], "model")
      unless model == certificate.source_model
        raise CheckpointManifestError.new("model does not match the config certificate")
      end

      revision = immutable_revision(root["revision"], "revision")
      unless revision == certificate.source_revision
        raise CheckpointManifestError.new("revision does not match the config certificate")
      end

      access_mode = string(root["access_mode"], "access_mode")
      unless access_mode == certificate.access_mode
        raise CheckpointManifestError.new("access mode does not match the config certificate")
      end

      license = strict_object(root["license"], "license")
      expect_exact_keys!(license, ["name", "link"], "license")
      license_name = string(license["name"], "license.name")
      license_link = string(license["link"], "license.link")
      unless license_name == certificate.license_name && license_link == certificate.license_link
        raise CheckpointManifestError.new("license does not match the config certificate")
      end

      config = strict_object(root["config"], "config")
      expect_exact_keys!(config, ["path", "byte_length", "sha256"], "config")
      config_path = string(config["path"], "config.path")
      unless config_path == PINNED_CONFIG_PATH
        raise CheckpointManifestError.new("config path is not the pinned config.json artifact")
      end
      config_byte_length = positive_integer(config["byte_length"], "config.byte_length")
      unless config_byte_length == PINNED_CONFIG_BYTE_LENGTH
        raise CheckpointManifestError.new("config byte length does not match the pinned artifact")
      end
      config_sha256 = sha256(config["sha256"], "config digest")
      unless config_sha256 == certificate.config_sha256
        raise CheckpointManifestError.new("config digest does not match the config certificate")
      end

      weights = strict_object(root["weights"], "weights")
      expect_exact_keys!(weights, ["path", "format", "byte_length", "sha256"], "weights")
      weights_path = string(weights["path"], "weights.path")
      unless weights_path == PINNED_WEIGHTS_PATH
        raise CheckpointManifestError.new("weights path is not the pinned model.safetensors artifact")
      end
      weights_format = string(weights["format"], "weights.format")
      unless weights_format == PINNED_WEIGHTS_FORMAT
        raise CheckpointManifestError.new("weights format is not safetensors")
      end
      weights_byte_length = positive_integer(weights["byte_length"], "weights.byte_length")
      unless weights_byte_length == PINNED_WEIGHTS_BYTE_LENGTH
        raise CheckpointManifestError.new("weights byte length does not match the pinned artifact")
      end
      weights_sha256 = sha256(weights["sha256"], "weights digest")
      unless weights_sha256 == PINNED_WEIGHTS_SHA256
        raise CheckpointManifestError.new("weights digest does not match the pinned artifact")
      end

      new(
        schema,
        model,
        revision,
        access_mode,
        license_name,
        license_link,
        config_path,
        config_byte_length,
        config_sha256,
        weights_path,
        weights_format,
        weights_byte_length,
        weights_sha256
      )
    rescue ex : CheckpointManifestError
      raise ex
    rescue ex : ML::ThreeD::Trellis2::StrictJSONError
      raise CheckpointManifestError.new(ex.message)
    rescue ex : JSON::ParseException
      raise CheckpointManifestError.new("invalid checkpoint manifest JSON: #{ex.message}")
    rescue
      raise CheckpointManifestError.new("invalid checkpoint manifest")
    end

    def model : String
      @model.dup
    end

    def revision : String
      @revision.dup
    end

    def access_mode : String
      @access_mode.dup
    end

    def license_name : String
      @license_name.dup
    end

    def license_link : String
      @license_link.dup
    end

    def config_path : String
      @config_path.dup
    end

    def config_sha256 : String
      @config_sha256.dup
    end

    def weights_path : String
      @weights_path.dup
    end

    def weights_format : String
      @weights_format.dup
    end

    def weights_sha256 : String
      @weights_sha256.dup
    end

    def config_url : String
      artifact_url(@config_path)
    end

    def weights_url : String
      artifact_url(@weights_path)
    end

    private def artifact_url(path : String) : String
      "https://huggingface.co/#{@model}/resolve/#{@revision}/#{path}"
    end

    private def self.validate_certificate!(certificate : ConfigCertificate) : Nil
      unless certificate.source_model == ConfigCertificate::PINNED_SOURCE_MODEL &&
             certificate.source_revision == ConfigCertificate::PINNED_SOURCE_REVISION &&
             certificate.config_sha256 == ConfigCertificate::PINNED_CONFIG_SHA256 &&
             certificate.access_mode == ConfigCertificate::PINNED_ACCESS_MODE &&
             certificate.license_name == ConfigCertificate::PINNED_LICENSE_NAME &&
             certificate.license_link == ConfigCertificate::PINNED_LICENSE_LINK
        raise CheckpointManifestError.new(
          "checkpoint manifest requires the pinned DINOv3 config certificate"
        )
      end
    end

    private def self.strict_object(
      source : String,
      context : String,
    ) : Hash(String, JSON::Any)
      ML::ThreeD::Trellis2::StrictJSON.parse(source).as_h
    rescue ex : ML::ThreeD::Trellis2::StrictJSONError
      raise ex
    rescue
      raise CheckpointManifestError.new("#{context} must be a JSON object")
    end

    private def self.strict_object(
      value : JSON::Any,
      context : String,
    ) : Hash(String, JSON::Any)
      value.as_h
    rescue
      raise CheckpointManifestError.new("#{context} must be a JSON object")
    end

    private def self.expect_exact_keys!(
      object : Hash(String, JSON::Any),
      allowed : Array(String),
      context : String,
    ) : Nil
      object.each_key do |key|
        unless allowed.includes?(key)
          raise CheckpointManifestError.new("unknown #{context} key #{key.inspect}")
        end
      end
      allowed.each do |key|
        unless object.has_key?(key)
          raise CheckpointManifestError.new("missing #{context} key #{key.inspect}")
        end
      end
    end

    private def self.string(value : JSON::Any, context : String) : String
      value.as_s
    rescue
      raise CheckpointManifestError.new("#{context} must be a string")
    end

    private def self.immutable_revision(value : JSON::Any, context : String) : String
      revision = string(value, context)
      unless revision.matches?(/\A(?:[0-9a-f]{40}|[0-9a-f]{64})\z/)
        raise CheckpointManifestError.new("#{context} must be an immutable revision")
      end
      revision
    end

    private def self.positive_integer(value : JSON::Any, context : String) : Int64
      result = value.as_i64
      unless result > 0
        raise CheckpointManifestError.new("#{context} must be positive")
      end
      result
    rescue ex : CheckpointManifestError
      raise ex
    rescue
      raise CheckpointManifestError.new("#{context} must be an integer")
    end

    private def self.sha256(value : JSON::Any, context : String) : String
      digest = string(value, context)
      unless digest.matches?(/\A[0-9a-f]{64}\z/)
        raise CheckpointManifestError.new("#{context} must be a SHA-256 digest")
      end
      digest
    end
  end
end
