require "./tensor"

module ML::Sparse
  class TensorCPU
    ADAPTIVE_LAYER_NORM_EPSILON = 1e-6_f32
    PYTORCH_CPU_VECTOR_WIDTH    =    4_i32
    PYTORCH_MOMENTS_CHUNK_SIZE  =   16_i32
    PYTORCH_MOMENTS_STACK_DEPTH =        4
    PYTORCH_MOMENTS_STACK_CELLS =       16

    # Executes the exact non-affine LayerNorm32 + batch-adaptive scale/shift
    # seam used immediately before TRELLIS.2 sparse attention. This bounded
    # CPU/F32 reference borrows canonical immutable inputs, allocates one output
    # feature array, and deliberately does not construct an autograd graph.
    def self.apply_adaptive_layer_norm(
      input : TensorCPU,
      scale : ML::Tensor,
      shift : ML::Tensor,
      epsilon : Float32 = ADAPTIVE_LAYER_NORM_EPSILON,
    ) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse adaptive layer norm requires the base TensorCPU receiver"
        )
      end
      unless input.@initialized
        raise SparseTensorError.new(
          "sparse adaptive layer norm requires an initialized sparse value"
        )
      end
      unless epsilon == ADAPTIVE_LAYER_NORM_EPSILON
        raise SparseTensorError.new(
          "sparse adaptive layer norm epsilon must be exactly 1.0e-6"
        )
      end

      batch_size, map_point_count = CoordinateMap3D.kernel_layout(
        input.@coordinate_map
      )
      unless 1 <= batch_size <= CoordinateMap3D::MAX_BATCH_SIZE
        raise SparseTensorError.new(
          "sparse adaptive layer norm batch size must be in 1..#{CoordinateMap3D::MAX_BATCH_SIZE}"
        )
      end
      unless 0 <= map_point_count <= CoordinateMap3D::MAX_POINTS
        raise SparseTensorError.new(
          "sparse adaptive layer norm point count must be in 0..#{CoordinateMap3D::MAX_POINTS}"
        )
      end
      unless map_point_count == input.@point_count
        raise SparseTensorError.new(
          "sparse adaptive layer norm requires coordinate and feature point counts to match"
        )
      end
      channels = input.@channels
      unless 1 <= channels <= MAX_CHANNELS
        raise SparseTensorError.new(
          "sparse adaptive layer norm channel count must be in 1..#{MAX_CHANNELS}"
        )
      end
      unless 0_i64 < input.@max_feature_bytes <= MAX_FEATURE_BYTES
        raise SparseTensorError.new(
          "sparse adaptive layer norm feature byte budget must be in 1..#{MAX_FEATURE_BYTES}"
        )
      end
      expected_elements = input.@point_count.to_i64 * channels.to_i64
      unless input.@features.size.to_i64 == expected_elements
        raise SparseTensorError.new(
          "sparse adaptive layer norm feature storage size must match [N, C]"
        )
      end
      scale_values = adaptive_parameter_values(
        "scale",
        scale,
        batch_size,
        channels
      )
      shift_values = adaptive_parameter_values(
        "shift",
        shift,
        batch_size,
        channels
      )

      point_count = input.@point_count
      output_elements = point_count.to_i64 * channels.to_i64
      output_bytes = output_elements * 4_i64
      if output_bytes > input.@max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse adaptive layer norm output features require #{output_bytes} bytes, limit is #{input.@max_feature_bytes}"
        )
      end
      output_features = Array(Float32).new(output_elements.to_i)
      point_count.times do |row|
        feature_offset = row * channels
        mean, variance = pytorch_rowwise_moments_f32(
          input.@features,
          feature_offset,
          channels
        )
        inverse_std = 1.0_f32 / Math.sqrt(variance + epsilon)

        batch = CoordinateMap3D.kernel_batch_index(input.@coordinate_map, row)
        unless 0 <= batch < batch_size
          raise SparseTensorError.new(
            "sparse adaptive layer norm batch map entry #{batch} is outside 0...#{batch_size}"
          )
        end
        adaptive_offset = batch * channels
        channels.times do |channel|
          normalized = (
            input.@features[feature_offset + channel] + (-mean)
          ) * inverse_std
          scaled = normalized * (
            1.0_f32 + scale_values[adaptive_offset + channel]
          )
          value = scaled + shift_values[adaptive_offset + channel]
          unless value.finite?
            raise SparseTensorError.new(
              "sparse adaptive layer norm output[#{output_features.size}] must be finite"
            )
          end
          output_features << value
        end
      end

      new(
        output_features,
        input.@coordinate_map,
        point_count,
        channels,
        input.@max_feature_bytes
      )
    end

    # Mirrors PyTorch 2.9 RowwiseMoments<float> on the pinned Darwin/arm64
    # DEFAULT CPU capability: four F32 lanes, 16 vectors per Welford chunk,
    # then the same binary-cascade and lane merge order. MAX_CHANNELS bounds
    # the required stack depth to four without heap scratch storage.
    private def self.pytorch_rowwise_moments_f32(
      features : Array(Float32),
      offset : Int32,
      channels : Int32,
    ) : Tuple(Float32, Float32)
      full_vectors = channels // PYTORCH_CPU_VECTOR_WIDTH
      chunks = (
        full_vectors + PYTORCH_MOMENTS_CHUNK_SIZE - 1
      ) // PYTORCH_MOMENTS_CHUNK_SIZE
      depth = if chunks <= 2
                1
              else
                value = chunks - 1
                result = 0
                while value > 0
                  result += 1
                  value >>= 1
                end
                result
              end
      if depth > PYTORCH_MOMENTS_STACK_DEPTH
        raise SparseTensorError.new(
          "sparse adaptive layer norm moment stack exceeds the admitted channel bound"
        )
      end

      stack_counts = StaticArray(Int32, PYTORCH_MOMENTS_STACK_DEPTH).new(0_i32)
      stack_means = StaticArray(
        Float32,
        PYTORCH_MOMENTS_STACK_CELLS,
      ).new(0.0_f32)
      stack_m2 = StaticArray(
        Float32,
        PYTORCH_MOMENTS_STACK_CELLS,
      ).new(0.0_f32)

      chunks.times do |chunk|
        remaining = full_vectors - chunk * PYTORCH_MOMENTS_CHUNK_SIZE
        chunk_vectors = remaining < PYTORCH_MOMENTS_CHUNK_SIZE ? remaining : PYTORCH_MOMENTS_CHUNK_SIZE
        chunk_means = StaticArray(Float32, PYTORCH_CPU_VECTOR_WIDTH).new(0.0_f32)
        chunk_m2 = StaticArray(Float32, PYTORCH_CPU_VECTOR_WIDTH).new(0.0_f32)
        chunk_vectors.times do |vector|
          coefficient = 1.0_f32 / (vector + 1).to_f32
          PYTORCH_CPU_VECTOR_WIDTH.times do |lane|
            feature_index = offset +
                            (chunk * PYTORCH_MOMENTS_CHUNK_SIZE + vector) *
                            PYTORCH_CPU_VECTOR_WIDTH +
                            lane
            value = features[feature_index]
            delta = value - chunk_means[lane]
            chunk_means[lane] += delta * coefficient
            chunk_m2[lane] += delta * (value - chunk_means[lane])
          end
        end

        target_count = stack_counts[0]
        combined_count = target_count + chunk_vectors
        coefficient = chunk_vectors.to_f32 / combined_count.to_f32
        PYTORCH_CPU_VECTOR_WIDTH.times do |lane|
          delta = chunk_means[lane] - stack_means[lane]
          stack_means[lane] += coefficient * delta
          stack_m2[lane] += chunk_m2[lane] +
                            delta * delta * coefficient * target_count.to_f32
        end
        stack_counts[0] = combined_count

        mask = chunk + 1
        stack = 1
        while stack < depth && (mask & 1) == 0
          source_offset = (stack - 1) * PYTORCH_CPU_VECTOR_WIDTH
          target_offset = stack * PYTORCH_CPU_VECTOR_WIDTH
          source_count = stack_counts[stack - 1]
          target_count = stack_counts[stack]
          combined_count = target_count + source_count
          coefficient = source_count.to_f32 / combined_count.to_f32
          PYTORCH_CPU_VECTOR_WIDTH.times do |lane|
            source = source_offset + lane
            target = target_offset + lane
            delta = stack_means[source] - stack_means[target]
            stack_means[target] += coefficient * delta
            stack_m2[target] += stack_m2[source] +
                                delta * delta * coefficient * target_count.to_f32
            stack_means[source] = 0.0_f32
            stack_m2[source] = 0.0_f32
          end
          stack_counts[stack] = combined_count
          stack_counts[stack - 1] = 0_i32
          mask >>= 1
          stack += 1
        end
      end

      stack = 1
      while stack < depth
        source_offset = stack * PYTORCH_CPU_VECTOR_WIDTH
        source_count = stack_counts[stack]
        target_count = stack_counts[0]
        combined_count = target_count + source_count
        coefficient = source_count.to_f32 / combined_count.to_f32
        PYTORCH_CPU_VECTOR_WIDTH.times do |lane|
          delta = stack_means[source_offset + lane] - stack_means[lane]
          stack_means[lane] += coefficient * delta
          stack_m2[lane] += stack_m2[source_offset + lane] +
                            delta * delta * coefficient * target_count.to_f32
        end
        stack_counts[0] = combined_count
        stack += 1
      end

      count = 0_i32
      mean = 0.0_f32
      m2 = 0.0_f32
      tail_start = full_vectors * PYTORCH_CPU_VECTOR_WIDTH
      tail_start.upto(channels - 1) do |channel|
        value = features[offset + channel]
        delta = value - mean
        count += 1
        mean += delta / count.to_f32
        m2 += delta * (value - mean)
      end

      lane_count = full_vectors
      PYTORCH_CPU_VECTOR_WIDTH.times do |lane|
        combined_count = count + lane_count
        coefficient = combined_count == 0 ? 0.0_f32 : lane_count.to_f32 / combined_count.to_f32
        delta = stack_means[lane] - mean
        mean += coefficient * delta
        m2 += stack_m2[lane] +
              delta * delta * coefficient * count.to_f32
        count = combined_count
      end

      {mean, m2 / channels.to_f32}
    end

    private def self.adaptive_parameter_values(
      name : String,
      parameter : ML::Tensor,
      batch_size : Int32,
      channels : Int32,
    ) : ML::Tensor::CPUReadView
      unless parameter.on_cpu?
        raise SparseTensorError.new(
          "sparse adaptive layer norm #{name} must be on CPU"
        )
      end
      unless parameter.dtype.f32?
        raise SparseTensorError.new(
          "sparse adaptive layer norm #{name} must use F32"
        )
      end
      unless parameter.contiguous?
        raise SparseTensorError.new(
          "sparse adaptive layer norm #{name} must be contiguous"
        )
      end
      unless parameter.ndim == 2 &&
             parameter.shape[0] == batch_size &&
             parameter.shape[1] == channels
        raise SparseTensorError.new(
          "sparse adaptive layer norm #{name} shape must be [#{batch_size}, #{channels}]"
        )
      end

      values = parameter.cpu_read
      unless values.borrowed? && values.materialized_bytes == 0_i64
        raise SparseTensorError.new(
          "sparse adaptive layer norm #{name} read must not materialize storage"
        )
      end
      values.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse adaptive layer norm #{name}[#{index}] must be finite"
          )
        end
      end
      values
    end
  end
end
