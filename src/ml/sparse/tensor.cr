require "../core/tensor"
require "./coordinate_map"

module ML::Sparse
  class SparseTensorError < Exception
  end

  class SparseTensorBudgetError < SparseTensorError
  end

  # Immutable graphless CPU/F32 sparse feature value for T2N4a.
  class TensorCPU
    MAX_CHANNELS                    =   256_i32
    PRODUCTION_MAX_CHANNELS         = 1_536_i32
    MAX_SELF_ATTENTION_QKV_CHANNELS = PRODUCTION_MAX_CHANNELS * 3_i32
    MAX_FEATURE_BYTES               = 64_i64 * 1024_i64 * 1024_i64

    # Carrier roles are base-API construction authority, not descriptive tags.
    # Only production-width packed storage needs a distinct role; legacy QKV
    # remains the same bounded TensorCPU value admitted before this slice.
    # Same-process subclasses are trusted extension code, as for the protected
    # owned-buffer transfer below; public base APIs never trust virtual getters.
    private enum CarrierRole : UInt8
      Bounded
      Production
      ProductionQKV
    end

    getter coordinate_map : CoordinateMap3D
    getter point_count : Int32
    getter channels : Int32
    getter max_feature_bytes : Int64

    @features : Array(Float32)
    @carrier_role : CarrierRole
    @initialized : Bool = false

    def initialize(
      features : Tensor,
      @coordinate_map : CoordinateMap3D,
      @max_feature_bytes : Int64 = MAX_FEATURE_BYTES,
    )
      @carrier_role = CarrierRole::Bounded
      @features, @point_count, @channels = TensorCPU.validated_feature_payload(
        features,
        @coordinate_map,
        @max_feature_bytes,
        MAX_CHANNELS,
        "sparse feature"
      )
      @initialized = true
    end

    # Explicitly constructs the production-width standard carrier needed by
    # TRELLIS.2. The ordinary constructor remains bounded to MAX_CHANNELS.
    def self.production(
      features : Tensor,
      coordinate_map : CoordinateMap3D,
      max_feature_bytes : Int64 = MAX_FEATURE_BYTES,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "production sparse carrier requires the base TensorCPU receiver"
        )
      end
      values, point_count, channels = validated_feature_payload(
        features,
        coordinate_map,
        max_feature_bytes,
        PRODUCTION_MAX_CHANNELS,
        "production sparse feature"
      )
      TensorCPU.from_owned_features(
        values,
        coordinate_map,
        point_count,
        channels,
        max_feature_bytes,
        CarrierRole::Production
      )
    end

    # Read-only diagnostics for tests and callers. Operations never trust these
    # virtual methods; they inspect the base-owned role ivar directly.
    def production_width? : Bool
      @carrier_role == CarrierRole::Production ||
        @carrier_role == CarrierRole::ProductionQKV
    end

    def packed_qkv? : Bool
      @carrier_role == CarrierRole::ProductionQKV
    end

    # Returns the legal standard-channel ceiling while checking the base role.
    # Cross-attention uses this class authority rather than a virtual role getter.
    def self.standard_carrier_channel_limit(
      input : TensorCPU,
      operation : String,
    ) : Int32
      unless TensorCPU == self
        raise SparseTensorError.new(
          "#{operation} requires the base TensorCPU receiver"
        )
      end
      unless input.@initialized
        raise SparseTensorError.new(
          "#{operation} requires an initialized sparse value"
        )
      end
      case input.@carrier_role
      when CarrierRole::Bounded
        MAX_CHANNELS
      when CarrierRole::Production
        PRODUCTION_MAX_CHANNELS
      else
        raise SparseTensorError.new(
          "#{operation} requires a standard sparse carrier, not packed QKV storage"
        )
      end
    end

    # Validates the flat storage admitted by the public QKV logical view.
    # Legacy bounded QKV fixtures remain valid, while a standard production
    # carrier cannot be relabelled as packed QKV merely because 3 divides C.
    def self.validate_self_attention_qkv_carrier!(
      input : TensorCPU,
      operation : String,
    ) : Nil
      unless TensorCPU == self
        raise SparseTensorError.new(
          "#{operation} requires the base TensorCPU receiver"
        )
      end
      unless input.@initialized
        raise SparseTensorError.new(
          "#{operation} requires an initialized sparse value"
        )
      end

      channel_limit = case input.@carrier_role
                      when CarrierRole::Bounded
                        MAX_CHANNELS
                      when CarrierRole::ProductionQKV
                        MAX_SELF_ATTENTION_QKV_CHANNELS
                      else
                        raise SparseTensorError.new(
                          "#{operation} requires packed production QKV storage"
                        )
                      end
      unless 1 <= input.@channels <= channel_limit
        raise SparseTensorError.new(
          "#{operation} carrier channel count must be in 1..#{channel_limit}"
        )
      end
    end

    protected def self.validated_feature_payload(
      features : Tensor,
      coordinate_map : CoordinateMap3D,
      max_feature_bytes : Int64,
      max_channels : Int32,
      label : String,
    ) : Tuple(Array(Float32), Int32, Int32)
      unless 0_i64 < max_feature_bytes <= MAX_FEATURE_BYTES
        raise SparseTensorError.new(
          "sparse feature byte budget must be in 1..#{MAX_FEATURE_BYTES}"
        )
      end
      unless features.on_cpu?
        raise SparseTensorError.new("sparse features must be on CPU")
      end
      unless features.dtype.f32?
        raise SparseTensorError.new("sparse features must use F32")
      end
      unless features.contiguous?
        raise SparseTensorError.new("sparse features must be contiguous")
      end
      unless features.shape.ndim == 2
        raise SparseTensorError.new("sparse features must have rank 2 [N, C]")
      end

      point_count = features.shape[0]
      channels = features.shape[1]
      _, coordinate_point_count = CoordinateMap3D.kernel_layout(coordinate_map)
      unless point_count == coordinate_point_count
        raise SparseTensorError.new(
          "sparse feature point count #{point_count} does not match coordinate point count #{coordinate_point_count}"
        )
      end
      unless 1 <= channels <= max_channels
        raise SparseTensorError.new(
          "#{label} channel count must be in 1..#{max_channels}"
        )
      end
      feature_bytes = point_count.to_i64 * channels.to_i64 * 4_i64
      if feature_bytes > max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse features require #{feature_bytes} bytes, limit is #{max_feature_bytes}"
        )
      end

      source = features.cpu_read
      values = Array(Float32).new(source.size) do |index|
        value = source[index]
        unless value.finite?
          raise SparseTensorError.new("sparse feature[#{index}] must be finite")
        end
        value
      end
      {values, point_count, channels}
    end

    # Private ownership transfer for class operations that already validated
    # every invariant and produced a fresh, non-escaping feature buffer.
    private def initialize(
      @features : Array(Float32),
      @coordinate_map : CoordinateMap3D,
      @point_count : Int32,
      @channels : Int32,
      @max_feature_bytes : Int64,
    ) : Nil
      @carrier_role = CarrierRole::Bounded
      @initialized = true
    end

    private def initialize(
      @features : Array(Float32),
      @coordinate_map : CoordinateMap3D,
      @point_count : Int32,
      @channels : Int32,
      @max_feature_bytes : Int64,
      @carrier_role : CarrierRole,
    ) : Nil
      @initialized = true
    end

    protected def self.from_owned_features(
      features : Array(Float32),
      coordinate_map : CoordinateMap3D,
      point_count : Int32,
      channels : Int32,
      max_feature_bytes : Int64,
      carrier_role : CarrierRole,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse owned carrier construction requires the base TensorCPU receiver"
        )
      end
      new(
        features,
        coordinate_map,
        point_count,
        channels,
        max_feature_bytes,
        carrier_role
      )
    end

    def shape : Tuple(Int32, Int32)
      batch_size, _ = CoordinateMap3D.kernel_layout(@coordinate_map)
      {batch_size, @channels}
    end

    def feature(row : Int32, channel : Int32) : Float32
      unless 0 <= row < @point_count
        raise IndexError.new("sparse feature row #{row} is out of bounds")
      end
      unless 0 <= channel < @channels
        raise IndexError.new("sparse feature channel #{channel} is out of bounds")
      end
      @features[row * @channels + channel]
    end

    def features_copy : Array(Float32)
      @features.dup
    end

    def replace_features(features : Tensor) : TensorCPU
      channel_limit = TensorCPU.standard_carrier_channel_limit(
        self,
        "sparse feature replacement"
      )
      values, point_count, channels = TensorCPU.validated_feature_payload(
        features,
        @coordinate_map,
        @max_feature_bytes,
        channel_limit,
        "sparse feature"
      )
      TensorCPU.from_owned_features(
        values,
        @coordinate_map,
        point_count,
        channels,
        @max_feature_bytes,
        @carrier_role
      )
    end

    def replace_coordinates(
      coordinates : Indexable(Int32),
      batch_size : Int32,
      spatial_shape : Tuple(Int32, Int32, Int32),
    ) : TensorCPU
      TensorCPU.standard_carrier_channel_limit(
        self,
        "sparse coordinate replacement"
      )
      replacement_map = CoordinateMap3D.new(
        coordinates,
        batch_size,
        spatial_shape
      )
      _, replacement_point_count = CoordinateMap3D.kernel_layout(replacement_map)
      unless replacement_point_count == @point_count
        raise SparseTensorError.new(
          "replacement coordinate point count #{replacement_point_count} does not match feature point count #{@point_count}"
        )
      end
      TensorCPU.from_owned_features(
        @features.dup,
        replacement_map,
        @point_count,
        @channels,
        @max_feature_bytes,
        @carrier_role
      )
    end

    protected def self.carrier_channel_limit(role : CarrierRole) : Int32
      case role
      when CarrierRole::Bounded
        MAX_CHANNELS
      when CarrierRole::Production
        PRODUCTION_MAX_CHANNELS
      when CarrierRole::ProductionQKV
        MAX_SELF_ATTENTION_QKV_CHANNELS
      else
        raise SparseTensorError.new("unknown sparse carrier role")
      end
    end

    private def self.packed_qkv_role(role : CarrierRole) : CarrierRole
      case role
      when CarrierRole::Bounded
        CarrierRole::Bounded
      when CarrierRole::Production
        CarrierRole::ProductionQKV
      else
        raise SparseTensorError.new(
          "sparse self-attention QKV requires a standard sparse carrier"
        )
      end
    end

    private def self.attention_output_role(role : CarrierRole) : CarrierRole
      case role
      when CarrierRole::Bounded
        CarrierRole::Bounded
      when CarrierRole::Production, CarrierRole::ProductionQKV
        CarrierRole::Production
      else
        raise SparseTensorError.new("unknown sparse carrier role")
      end
    end
  end
end
