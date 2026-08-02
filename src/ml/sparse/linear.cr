require "./tensor"
require "../nn/linear"

module ML::Sparse
  class TensorCPU
    # Applies one frozen dense Linear projection to the owned feature rows.
    # This is an inference-only CPU reference: it borrows the existing sparse
    # input and contiguous parameters, allocates one output feature array, and
    # deliberately does not construct an autograd graph.
    def self.apply_linear(input : TensorCPU, linear : ML::NN::Linear) : TensorCPU
      unless TensorCPU == self
        raise SparseTensorError.new(
          "sparse linear requires the base TensorCPU receiver"
        )
      end
      unless input.@initialized
        raise SparseTensorError.new(
          "sparse linear requires an initialized sparse value"
        )
      end

      input_channels = input.@channels
      linear_input_channels = linear.in_features
      output_channels = linear.out_features
      unless input_channels == linear_input_channels
        raise SparseTensorError.new(
          "sparse linear input channels #{input_channels} do not match Linear in_features #{linear_input_channels}"
        )
      end
      unless 1 <= output_channels <= MAX_CHANNELS
        raise SparseTensorBudgetError.new(
          "sparse linear output channel count #{output_channels} exceeds #{MAX_CHANNELS}"
        )
      end

      point_count = input.@point_count
      output_elements = point_count.to_i64 * output_channels.to_i64
      output_bytes = output_elements * 4_i64
      output_budget = input.@max_feature_bytes
      if output_bytes > output_budget
        raise SparseTensorBudgetError.new(
          "sparse linear output features require #{output_bytes} bytes, limit is #{output_budget}"
        )
      end

      weight = linear.weight
      bias = linear.bias
      if weight.requires_grad? || (bias.try(&.requires_grad?) || false)
        raise SparseTensorError.new(
          "sparse linear requires frozen graphless parameters"
        )
      end

      weight_tensor = weight.data
      unless weight_tensor.on_cpu?
        raise SparseTensorError.new("sparse linear weight must be on CPU")
      end
      unless weight_tensor.dtype.f32?
        raise SparseTensorError.new("sparse linear weight must use F32")
      end
      unless weight_tensor.contiguous?
        raise SparseTensorError.new("sparse linear weight must be contiguous")
      end
      unless weight_tensor.ndim == 2 &&
             weight_tensor.shape[0] == output_channels &&
             weight_tensor.shape[1] == input_channels
        raise SparseTensorError.new(
          "sparse linear weight shape must be [#{output_channels}, #{input_channels}]"
        )
      end

      weight_values = weight_tensor.cpu_read
      unless weight_values.borrowed? && weight_values.materialized_bytes == 0_i64
        raise SparseTensorError.new(
          "sparse linear weight read must not materialize parameter storage"
        )
      end
      weight_values.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse linear weight[#{index}] must be finite"
          )
        end
      end

      bias_values = bias.try do |bias_variable|
        bias_tensor = bias_variable.data
        unless bias_tensor.on_cpu?
          raise SparseTensorError.new("sparse linear bias must be on CPU")
        end
        unless bias_tensor.dtype.f32?
          raise SparseTensorError.new("sparse linear bias must use F32")
        end
        unless bias_tensor.contiguous?
          raise SparseTensorError.new("sparse linear bias must be contiguous")
        end
        unless bias_tensor.ndim == 1 && bias_tensor.shape[0] == output_channels
          raise SparseTensorError.new(
            "sparse linear bias shape must be [#{output_channels}]"
          )
        end

        values = bias_tensor.cpu_read
        unless values.borrowed? && values.materialized_bytes == 0_i64
          raise SparseTensorError.new(
            "sparse linear bias read must not materialize parameter storage"
          )
        end
        values.each_with_index do |value, index|
          unless value.finite?
            raise SparseTensorError.new(
              "sparse linear bias[#{index}] must be finite"
            )
          end
        end
        values
      end

      # TensorCPU owns this input storage. Parameter reads above are borrowed;
      # callers must not mutate the input or layer concurrently with this call.
      output_features = Array(Float32).new(output_elements.to_i)
      point_count.times do |row|
        input_offset = row * input_channels
        output_channels.times do |output_channel|
          weight_offset = output_channel * input_channels
          sum = 0.0_f32
          input_channels.times do |input_channel|
            sum += input.@features[input_offset + input_channel] *
                   weight_values[weight_offset + input_channel]
          end
          value = if values = bias_values
                    sum + values[output_channel]
                  else
                    sum
                  end
          unless value.finite?
            raise SparseTensorError.new(
              "sparse linear output[#{output_features.size}] must be finite"
            )
          end
          output_features << value
        end
      end

      new(
        output_features,
        input.@coordinate_map,
        point_count,
        output_channels,
        output_budget
      )
    end
  end
end
