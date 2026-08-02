require "../core/tensor"
require "./coordinate_map"

module ML::Sparse
  class SparseTensorError < Exception
  end

  class SparseTensorBudgetError < SparseTensorError
  end

  # Immutable graphless CPU/F32 sparse feature value for T2N4a.
  class TensorCPU
    MAX_CHANNELS      = 256_i32
    MAX_FEATURE_BYTES = 64_i64 * 1024_i64 * 1024_i64

    getter coordinate_map : CoordinateMap3D
    getter point_count : Int32
    getter channels : Int32
    getter max_feature_bytes : Int64

    @features : Array(Float32)
    @initialized : Bool = false

    def initialize(
      features : Tensor,
      @coordinate_map : CoordinateMap3D,
      @max_feature_bytes : Int64 = MAX_FEATURE_BYTES,
    )
      unless 0_i64 < @max_feature_bytes <= MAX_FEATURE_BYTES
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

      @point_count = features.shape[0]
      @channels = features.shape[1]
      _, coordinate_point_count = CoordinateMap3D.kernel_layout(@coordinate_map)
      unless @point_count == coordinate_point_count
        raise SparseTensorError.new(
          "sparse feature point count #{@point_count} does not match coordinate point count #{coordinate_point_count}"
        )
      end
      unless 1 <= @channels <= MAX_CHANNELS
        raise SparseTensorError.new(
          "sparse feature channel count must be in 1..#{MAX_CHANNELS}"
        )
      end
      feature_bytes = @point_count.to_i64 * @channels.to_i64 * 4_i64
      if feature_bytes > @max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse features require #{feature_bytes} bytes, limit is #{@max_feature_bytes}"
        )
      end

      source = features.cpu_read
      @features = Array(Float32).new(source.size) do |index|
        value = source[index]
        unless value.finite?
          raise SparseTensorError.new("sparse feature[#{index}] must be finite")
        end
        value
      end
      @initialized = true
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
      @initialized = true
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
      TensorCPU.new(features, @coordinate_map, @max_feature_bytes)
    end

    def replace_coordinates(
      coordinates : Indexable(Int32),
      batch_size : Int32,
      spatial_shape : Tuple(Int32, Int32, Int32),
    ) : TensorCPU
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
      TensorCPU.new(
        Tensor.from_array(
          @features,
          Shape.new(@point_count, @channels)
        ),
        replacement_map,
        @max_feature_bytes
      )
    end
  end
end
