require "digest/sha256"
require "json"

require "../../three_d/trellis2/strict_json"

module ML::Vision::DinoV3
  # A configuration certificate is metadata only. It deliberately does not
  # construct BlockConfig, allocate model-scale parameters, or admit model
  # execution.
  class ConfigError < Exception
  end

  class ConfigCertificate
    PINNED_SOURCE_MODEL    = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    PINNED_SOURCE_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    PINNED_CONFIG_SHA256   = "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
    PINNED_SOURCE_URL      = "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/ea8dc2863c51be0a264bab82070e3e8836b02d51/config.json"
    PINNED_LICENSE_NAME    = "dinov3-license"
    PINNED_LICENSE_LINK    = "https://ai.meta.com/resources/models-and-libraries/dinov3-license"
    PINNED_ACCESS_MODE     = "manual"

    KNOWN_FIELDS = [
      "architectures",
      "attention_dropout",
      "drop_path_rate",
      "hidden_act",
      "hidden_size",
      "image_size",
      "initializer_range",
      "intermediate_size",
      "key_bias",
      "layer_norm_eps",
      "layerscale_value",
      "mlp_bias",
      "model_type",
      "num_attention_heads",
      "num_channels",
      "num_hidden_layers",
      "num_register_tokens",
      "patch_size",
      "pos_embed_jitter",
      "pos_embed_rescale",
      "pos_embed_shift",
      "proj_bias",
      "query_bias",
      "rope_theta",
      "torch_dtype",
      "transformers_version",
      "use_gated_mlp",
      "value_bias",
    ]

    getter source_model : String
    getter source_revision : String
    getter config_sha256 : String

    getter attention_dropout : Float64
    getter drop_path_rate : Float64
    getter hidden_act : String
    getter hidden_size : Int64
    getter image_size : Int64
    getter initializer_range : Float64
    getter intermediate_size : Int64
    getter key_bias : Bool
    getter layer_norm_eps : Float64
    getter layerscale_value : Float64
    getter mlp_bias : Bool
    getter model_type : String
    getter num_attention_heads : Int64
    getter num_channels : Int64
    getter num_hidden_layers : Int64
    getter num_register_tokens : Int64
    getter patch_size : Int64
    getter pos_embed_jitter : Nil
    getter pos_embed_rescale : Float64
    getter pos_embed_shift : Nil
    getter proj_bias : Bool
    getter query_bias : Bool
    getter rope_theta : Float64
    getter torch_dtype : String
    getter transformers_version : String
    getter use_gated_mlp : Bool
    getter value_bias : Bool

    private def initialize(
      @source_model : String,
      @source_revision : String,
      @config_sha256 : String,
      @fields : Hash(String, JSON::Any),
      @architectures : Array(String),
      @attention_dropout : Float64,
      @drop_path_rate : Float64,
      @hidden_act : String,
      @hidden_size : Int64,
      @image_size : Int64,
      @initializer_range : Float64,
      @intermediate_size : Int64,
      @key_bias : Bool,
      @layer_norm_eps : Float64,
      @layerscale_value : Float64,
      @mlp_bias : Bool,
      @model_type : String,
      @num_attention_heads : Int64,
      @num_channels : Int64,
      @num_hidden_layers : Int64,
      @num_register_tokens : Int64,
      @patch_size : Int64,
      @pos_embed_jitter : Nil,
      @pos_embed_rescale : Float64,
      @pos_embed_shift : Nil,
      @proj_bias : Bool,
      @query_bias : Bool,
      @rope_theta : Float64,
      @torch_dtype : String,
      @transformers_version : String,
      @use_gated_mlp : Bool,
      @value_bias : Bool,
    )
    end

    # Parse the exact bytes supplied by the caller. Source metadata is required
    # rather than defaulted so provenance cannot silently drift.
    def self.parse(
      source : String,
      *,
      source_model : String,
      source_revision : String,
    ) : ConfigCertificate
      parse_bytes(source.to_slice, source_model, source_revision)
    end

    def self.parse(
      source : Bytes,
      *,
      source_model : String,
      source_revision : String,
    ) : ConfigCertificate
      parse_bytes(source, source_model, source_revision)
    end

    def self.load(
      path : String,
      *,
      source_model : String,
      source_revision : String,
    ) : ConfigCertificate
      parse(File.read(path), source_model: source_model, source_revision: source_revision)
    rescue ex : File::Error
      raise ConfigError.new("cannot load DINOv3 config: #{ex.message}")
    end

    # Exact decoded field inventory. A copy prevents callers from mutating the
    # certificate's evidence map after parsing.
    def fields : Hash(String, JSON::Any)
      decoded_fields
    end

    def architectures : Array(String)
      @architectures.dup
    end

    def decoded_fields : Hash(String, JSON::Any)
      # JSON::Any may contain nested mutable arrays, so a shallow Hash#dup is
      # not enough to keep the certificate's provenance snapshot isolated.
      JSON.parse(@fields.to_json).as_h
    end

    def source_url : String
      PINNED_SOURCE_URL
    end

    def license_name : String
      PINNED_LICENSE_NAME
    end

    def license_link : String
      PINNED_LICENSE_LINK
    end

    def access_mode : String
      PINNED_ACCESS_MODE
    end

    def head_dim : Int64
      @hidden_size // @num_attention_heads
    end

    private def self.parse_bytes(
      source : Bytes,
      source_model : String,
      source_revision : String,
    ) : ConfigCertificate
      validate_source!(source_model, source_revision)
      digest = Digest::SHA256.hexdigest(source)

      root = begin
        ML::ThreeD::Trellis2::StrictJSON.parse(String.new(source)).as_h
      rescue ex : ML::ThreeD::Trellis2::StrictJSONError
        raise ConfigError.new("invalid DINOv3 config JSON: #{ex.message}")
      rescue ex : TypeCastError
        raise ConfigError.new("DINOv3 config must be a JSON object")
      end
      expect_exact_keys!(root)

      architectures = string_array(root["architectures"], "architectures")
      unless architectures.includes?("DINOv3ViTModel")
        raise ConfigError.new(
          "architectures must include DINOv3ViTModel"
        )
      end
      model_type = string(root["model_type"], "model_type")
      unless model_type == "dinov3_vit"
        raise ConfigError.new("model_type must be dinov3_vit")
      end

      image_size = positive_integer(root["image_size"], "image_size")
      unless image_size == 224_i64
        raise ConfigError.new("image_size must equal 224")
      end
      patch_size = integer(root["patch_size"], "patch_size")
      unless patch_size == 16_i64
        raise ConfigError.new("patch_size must equal 16")
      end
      num_channels = positive_integer(root["num_channels"], "num_channels")
      unless num_channels == 3_i64
        raise ConfigError.new("num_channels must equal 3")
      end

      hidden_size = positive_integer(root["hidden_size"], "hidden_size")
      intermediate_size = positive_integer(root["intermediate_size"], "intermediate_size")
      num_hidden_layers = positive_integer(root["num_hidden_layers"], "num_hidden_layers")
      num_attention_heads = positive_integer(root["num_attention_heads"], "num_attention_heads")
      num_register_tokens = positive_integer(root["num_register_tokens"], "num_register_tokens")
      unless hidden_size % num_attention_heads == 0
        raise ConfigError.new(
          "hidden_size must be divisible by num_attention_heads"
        )
      end
      unless intermediate_size % hidden_size == 0
        raise ConfigError.new(
          "intermediate_size must be divisible by hidden_size"
        )
      end

      rope_theta = positive_float(root["rope_theta"], "rope_theta")
      layer_norm_eps = positive_float(root["layer_norm_eps"], "layer_norm_eps")
      initializer_range = positive_float(root["initializer_range"], "initializer_range")
      layerscale_value = finite_float(root["layerscale_value"], "layerscale_value")
      unless layerscale_value == 1.0
        raise ConfigError.new("layerscale_value must equal 1.0")
      end
      pos_embed_rescale = positive_float(root["pos_embed_rescale"], "pos_embed_rescale")
      unless pos_embed_rescale == 2.0
        raise ConfigError.new("pos_embed_rescale must equal 2.0")
      end

      hidden_act = string(root["hidden_act"], "hidden_act")
      unless hidden_act == "gelu"
        raise ConfigError.new("hidden_act must be exact GELU (gelu)")
      end
      query_bias = boolean(root["query_bias"], "query_bias")
      key_bias = boolean(root["key_bias"], "key_bias")
      value_bias = boolean(root["value_bias"], "value_bias")
      proj_bias = boolean(root["proj_bias"], "proj_bias")
      mlp_bias = boolean(root["mlp_bias"], "mlp_bias")
      unless query_bias && !key_bias && value_bias && proj_bias && mlp_bias
        raise ConfigError.new(
          "DINOv3 requires q/v/proj/mlp bias true and key bias false"
        )
      end

      attention_dropout = finite_float(root["attention_dropout"], "attention_dropout")
      drop_path_rate = finite_float(root["drop_path_rate"], "drop_path_rate")
      unless attention_dropout == 0.0
        raise ConfigError.new("attention_dropout must be zero")
      end
      unless drop_path_rate == 0.0
        raise ConfigError.new("drop_path_rate must be zero")
      end
      use_gated_mlp = boolean(root["use_gated_mlp"], "use_gated_mlp")
      if use_gated_mlp
        raise ConfigError.new("gated MLP is not admitted")
      end
      torch_dtype = string(root["torch_dtype"], "torch_dtype")
      unless torch_dtype == "float32"
        raise ConfigError.new("torch_dtype must be float32")
      end

      pos_embed_jitter = null_value(root["pos_embed_jitter"], "pos_embed_jitter")
      pos_embed_shift = null_value(root["pos_embed_shift"], "pos_embed_shift")
      transformers_version = string(root["transformers_version"], "transformers_version")
      unless transformers_version == "4.56.0.dev0"
        raise ConfigError.new("transformers_version must equal 4.56.0.dev0")
      end

      unless digest == PINNED_CONFIG_SHA256
        raise ConfigError.new(
          "DINOv3 config SHA-256 #{digest} does not match pinned #{PINNED_CONFIG_SHA256}"
        )
      end

      new(
        source_model,
        source_revision,
        digest,
        root.dup,
        architectures,
        attention_dropout,
        drop_path_rate,
        hidden_act,
        hidden_size,
        image_size,
        initializer_range,
        intermediate_size,
        key_bias,
        layer_norm_eps,
        layerscale_value,
        mlp_bias,
        model_type,
        num_attention_heads,
        num_channels,
        num_hidden_layers,
        num_register_tokens,
        patch_size,
        pos_embed_jitter,
        pos_embed_rescale,
        pos_embed_shift,
        proj_bias,
        query_bias,
        rope_theta,
        torch_dtype,
        transformers_version,
        use_gated_mlp,
        value_bias
      )
    rescue ex : ConfigError
      raise ex
    rescue ex : JSON::ParseException
      raise ConfigError.new("invalid DINOv3 config JSON: #{ex.message}")
    end

    private def self.validate_source!(source_model : String, source_revision : String) : Nil
      if source_model.empty?
        raise ConfigError.new("source_model must not be empty")
      end
      unless source_revision.matches?(/\A(?:[0-9a-f]{40}|[0-9a-f]{64})\z/)
        raise ConfigError.new("source_revision must be an immutable hex revision")
      end
      unless source_model == PINNED_SOURCE_MODEL
        raise ConfigError.new("source_model must equal pinned DINOv3 model")
      end
      unless source_revision == PINNED_SOURCE_REVISION
        raise ConfigError.new("source_revision must equal pinned DINOv3 revision")
      end
    end

    private def self.expect_exact_keys!(root : Hash(String, JSON::Any)) : Nil
      root.each_key do |key|
        unless KNOWN_FIELDS.includes?(key)
          raise ConfigError.new("unknown config key #{key.inspect}")
        end
      end
      KNOWN_FIELDS.each do |key|
        unless root.has_key?(key)
          raise ConfigError.new("missing config key #{key.inspect}")
        end
      end
    end

    private def self.string(value : JSON::Any, context : String) : String
      value.as_s
    rescue
      raise ConfigError.new("#{context} must be a string")
    end

    private def self.string_array(value : JSON::Any, context : String) : Array(String)
      values = value.as_a
      values.map_with_index do |entry, index|
        string(entry, "#{context}[#{index}]")
      end
    rescue ex : ConfigError
      raise ex
    rescue
      raise ConfigError.new("#{context} must be an array")
    end

    private def self.integer(value : JSON::Any, context : String) : Int64
      value.as_i64
    rescue
      raise ConfigError.new("#{context} must be an integer")
    end

    private def self.positive_integer(value : JSON::Any, context : String) : Int64
      result = integer(value, context)
      unless result > 0
        raise ConfigError.new("#{context} must be positive")
      end
      result
    end

    private def self.finite_float(value : JSON::Any, context : String) : Float64
      result = begin
        value.as_f
      rescue
        value.as_i64.to_f64
      end
      unless result.finite?
        raise ConfigError.new("#{context} must be finite")
      end
      result
    rescue ex : ConfigError
      raise ex
    rescue
      raise ConfigError.new("#{context} must be a number")
    end

    private def self.positive_float(value : JSON::Any, context : String) : Float64
      result = finite_float(value, context)
      unless result > 0.0
        raise ConfigError.new("#{context} must be finite and positive")
      end
      result
    end

    private def self.boolean(value : JSON::Any, context : String) : Bool
      value.as_bool
    rescue
      raise ConfigError.new("#{context} must be a boolean")
    end

    private def self.null_value(value : JSON::Any, context : String) : Nil
      unless value.raw.nil?
        raise ConfigError.new("#{context} must be null")
      end
      nil
    end
  end

  # Maps the runtime attribute path used by the pinned Transformers 5.8.1
  # synthetic contract to its state-dict prefixes. The gated checkpoint has a
  # separate source-era serialization contract; see CheckpointPathAdapter.
  class LayerPathAdapter
    PINNED_RUNTIME_PATH          = "model.layer"
    ROOT_RUNTIME_PATH            = "layer"
    SERIALIZED_STATE_DICT_PREFIX = "model.layer"

    getter runtime_path : String

    def initialize(runtime_path : String)
      if runtime_path == ROOT_RUNTIME_PATH
        raise ConfigError.new(
          "root layer path #{runtime_path.inspect} is not admitted; use model.layer"
        )
      end
      unless runtime_path == PINNED_RUNTIME_PATH
        raise ConfigError.new(
          "runtime layer path #{runtime_path.inspect} is not the pinned model.layer path"
        )
      end
      @runtime_path = runtime_path
    end

    def serialized_state_dict_prefix(layer_index : Int32) : String
      unless layer_index >= 0
        raise ConfigError.new("layer index must be non-negative")
      end
      "#{SERIALIZED_STATE_DICT_PREFIX}.#{layer_index}"
    end

    def layer_index_from_serialized_prefix(prefix : String) : Int32
      marker = "#{SERIALIZED_STATE_DICT_PREFIX}."
      unless prefix.starts_with?(marker)
        raise ConfigError.new(
          "invalid serialized layer prefix #{prefix.inspect}; expected model.layer.N"
        )
      end
      suffix = prefix[marker.size..]
      unless suffix.matches?(/\A(?:0|[1-9][0-9]*)\z/)
        raise ConfigError.new(
          "invalid serialized layer prefix #{prefix.inspect}; expected model.layer.N"
        )
      end
      value = suffix.to_i64
      unless value <= Int32::MAX
        raise ConfigError.new("serialized layer index exceeds Int32")
      end
      value.to_i32
    rescue ex : ConfigError
      raise ex
    rescue
      raise ConfigError.new(
        "invalid serialized layer prefix #{prefix.inspect}; expected model.layer.N"
      )
    end
  end

  # Maps the source-era gated checkpoint keys independently from the runtime
  # object path. The authorized config records Transformers 4.56.0.dev0, whose
  # DINOv3 model serialized encoder layers at root `layer.N`; the compatibility
  # runtime used by the native implementation exposes `model.model.layer`.
  # Keeping these contracts separate prevents a shared path string from being
  # treated as proof of a checkpoint key name.
  class CheckpointPathAdapter
    PINNED_TRANSFORMERS_VERSION  = "4.56.0.dev0"
    SERIALIZED_STATE_DICT_PREFIX = "layer"
    ROOT_SERIALIZED_PREFIX       = "model.layer"

    getter transformers_version : String

    def initialize(certificate : ConfigCertificate)
      unless certificate.source_model == ConfigCertificate::PINNED_SOURCE_MODEL &&
             certificate.source_revision == ConfigCertificate::PINNED_SOURCE_REVISION &&
             certificate.config_sha256 == ConfigCertificate::PINNED_CONFIG_SHA256
        raise ConfigError.new(
          "checkpoint path adapter requires the pinned DINOv3 config certificate"
        )
      end
      unless certificate.transformers_version == PINNED_TRANSFORMERS_VERSION
        raise ConfigError.new(
          "checkpoint path adapter requires Transformers #{PINNED_TRANSFORMERS_VERSION}"
        )
      end
      @transformers_version = certificate.transformers_version.dup
    end

    def serialized_state_dict_prefix(layer_index : Int32) : String
      unless layer_index >= 0
        raise ConfigError.new("layer index must be non-negative")
      end
      "#{SERIALIZED_STATE_DICT_PREFIX}.#{layer_index}"
    end

    def layer_index_from_serialized_prefix(prefix : String) : Int32
      marker = "#{SERIALIZED_STATE_DICT_PREFIX}."
      unless prefix.starts_with?(marker)
        raise ConfigError.new(
          "invalid checkpoint layer prefix #{prefix.inspect}; expected layer.N"
        )
      end
      suffix = prefix[marker.size..]
      unless suffix.matches?(/\A(?:0|[1-9][0-9]*)\z/)
        raise ConfigError.new(
          "invalid checkpoint layer prefix #{prefix.inspect}; expected layer.N"
        )
      end
      value = suffix.to_i64
      unless value <= Int32::MAX
        raise ConfigError.new("checkpoint layer index exceeds Int32")
      end
      value.to_i32
    rescue ex : ConfigError
      raise ex
    rescue
      raise ConfigError.new(
        "invalid checkpoint layer prefix #{prefix.inspect}; expected layer.N"
      )
    end
  end
end
