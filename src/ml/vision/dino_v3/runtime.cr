# Source-pinned metadata adapter for the DINOv3 runtime boundary used by
# TRELLIS.2. This maps the upstream logical path to the concrete Transformers
# instance path without constructing a model, loading weights, or executing a
# 24-layer encoder. Its `serialized_state_dict_prefix` describes the pinned
# Transformers 5.8.1 synthetic/runtime contract; source-era checkpoint keys
# are handled separately by CheckpointPathAdapter.

require "./config"
require "./embeddings"

module ML::Vision::DinoV3
  class RuntimeAdapterError < ConfigError
  end

  class RuntimeAdapter
    PINNED_TRANSFORMERS_VERSION    = "5.8.1"
    PINNED_SOURCE_RUNTIME_PATH     = "model.layer"
    PINNED_INSTANCE_RUNTIME_PATH   = "model.model.layer"
    PINNED_SOURCE_PATH_AVAILABLE   = false
    PINNED_INSTANCE_PATH_AVAILABLE = true

    getter certificate : ConfigCertificate
    getter depth : Int32

    @source_path_available : Bool
    @instance_path_available : Bool
    @layer_path_adapter : LayerPathAdapter

    def initialize(
      @certificate : ConfigCertificate,
      *,
      trellis_source_revision : String = TRELLIS2_SOURCE_REVISION,
      transformers_version : String = PINNED_TRANSFORMERS_VERSION,
      transformers_modeling_sha256 : String = TRANSFORMERS_MODELING_SHA256,
      transformers_config_sha256 : String = TRANSFORMERS_CONFIG_SHA256,
      source_runtime_path : String = PINNED_SOURCE_RUNTIME_PATH,
      instance_runtime_path : String = PINNED_INSTANCE_RUNTIME_PATH,
      source_path_available : Bool = PINNED_SOURCE_PATH_AVAILABLE,
      instance_path_available : Bool = PINNED_INSTANCE_PATH_AVAILABLE,
    )
      validate_certificate!
      validate_provenance!(
        trellis_source_revision,
        transformers_version,
        transformers_modeling_sha256,
        transformers_config_sha256
      )
      validate_paths!(
        source_runtime_path,
        instance_runtime_path,
        source_path_available,
        instance_path_available
      )

      @trellis_source_revision = trellis_source_revision.dup
      @transformers_version = transformers_version.dup
      @transformers_modeling_sha256 = transformers_modeling_sha256.dup
      @transformers_config_sha256 = transformers_config_sha256.dup
      @source_runtime_path = source_runtime_path.dup
      @instance_runtime_path = instance_runtime_path.dup
      @source_path_available = source_path_available
      @instance_path_available = instance_path_available
      @depth = @certificate.num_hidden_layers.to_i32
      @layer_path_adapter = LayerPathAdapter.new(@source_runtime_path)
    end

    def trellis_source_revision : String
      @trellis_source_revision.dup
    end

    def transformers_version : String
      @transformers_version.dup
    end

    def transformers_modeling_sha256 : String
      @transformers_modeling_sha256.dup
    end

    def transformers_config_sha256 : String
      @transformers_config_sha256.dup
    end

    def source_runtime_path : String
      @source_runtime_path.dup
    end

    def instance_runtime_path : String
      @instance_runtime_path.dup
    end

    def source_path_available? : Bool
      @source_path_available
    end

    def instance_path_available? : Bool
      @instance_path_available
    end

    def serialized_state_dict_prefix(layer_index : Int32) : String
      validate_layer_index!(layer_index)
      @layer_path_adapter.serialized_state_dict_prefix(layer_index)
    end

    def layer_index_from_serialized_prefix(prefix : String) : Int32
      layer_index = begin
        @layer_path_adapter.layer_index_from_serialized_prefix(prefix)
      rescue ex : ConfigError
        raise RuntimeAdapterError.new(ex.message)
      end
      validate_layer_index!(layer_index)
      layer_index
    end

    private def validate_certificate! : Nil
      unless @certificate.source_model == ConfigCertificate::PINNED_SOURCE_MODEL &&
             @certificate.source_revision == ConfigCertificate::PINNED_SOURCE_REVISION &&
             @certificate.config_sha256 == ConfigCertificate::PINNED_CONFIG_SHA256
        raise RuntimeAdapterError.new(
          "DINOv3 runtime adapter requires the pinned model config certificate"
        )
      end
      unless @certificate.num_hidden_layers <= Int32::MAX
        raise RuntimeAdapterError.new("certified DINOv3 layer count exceeds Int32")
      end
    end

    private def validate_provenance!(
      trellis_source_revision : String,
      transformers_version : String,
      transformers_modeling_sha256 : String,
      transformers_config_sha256 : String,
    ) : Nil
      unless trellis_source_revision == TRELLIS2_SOURCE_REVISION
        raise RuntimeAdapterError.new(
          "TRELLIS.2 source revision is not pinned"
        )
      end
      unless transformers_version == PINNED_TRANSFORMERS_VERSION
        raise RuntimeAdapterError.new(
          "Transformers runtime version is not pinned"
        )
      end
      unless transformers_modeling_sha256 == TRANSFORMERS_MODELING_SHA256
        raise RuntimeAdapterError.new(
          "Transformers modeling source digest is not pinned"
        )
      end
      unless transformers_config_sha256 == TRANSFORMERS_CONFIG_SHA256
        raise RuntimeAdapterError.new(
          "Transformers config source digest is not pinned"
        )
      end
    end

    private def validate_paths!(
      source_runtime_path : String,
      instance_runtime_path : String,
      source_path_available : Bool,
      instance_path_available : Bool,
    ) : Nil
      unless source_runtime_path == PINNED_SOURCE_RUNTIME_PATH
        raise RuntimeAdapterError.new(
          "source runtime path #{source_runtime_path.inspect} is not pinned"
        )
      end
      unless instance_runtime_path == PINNED_INSTANCE_RUNTIME_PATH
        raise RuntimeAdapterError.new(
          "instance runtime path #{instance_runtime_path.inspect} is not pinned"
        )
      end
      unless source_path_available == PINNED_SOURCE_PATH_AVAILABLE
        raise RuntimeAdapterError.new(
          "source path availability does not match pinned runtime evidence"
        )
      end
      unless instance_path_available == PINNED_INSTANCE_PATH_AVAILABLE
        raise RuntimeAdapterError.new(
          "instance path availability does not match pinned runtime evidence"
        )
      end
    end

    private def validate_layer_index!(layer_index : Int32) : Nil
      unless 0 <= layer_index < @depth
        raise RuntimeAdapterError.new(
          "DINOv3 layer index #{layer_index} is outside certified depth #{@depth}"
        )
      end
    end
  end
end
