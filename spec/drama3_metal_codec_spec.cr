require "spec"
require "../src/ml/audio/drama3/native"
require "../src/ml/audio/drama3/metal"
require "../src/ml/audio/drama3/codec_metal"

include ML::Audio::Drama3

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  private def drama3_metal_with_device(&block : -> T) : T forall T
    previous = ENV["DRAMA3_BACKEND"]?
    begin
      ENV["DRAMA3_BACKEND"] = "metal"
      device = ML::Metal::Device.instance
      pending!("Metal is unavailable or lacks unified memory") unless device.available? && device.has_unified_memory?
      MetalBackend.enabled?.should be_true
      yield
    ensure
      if previous
        ENV["DRAMA3_BACKEND"] = previous
      else
        ENV.delete("DRAMA3_BACKEND")
      end
    end
  end

  private def drama3_codec_cpu_conv(
    input : Matrix, weights : Vector, bias : Vector, kernel : Int32,
    stride : Int32, groups : Int32, dilation : Int32,
  ) : Matrix
    input_channels = input.cols
    output_channels = bias.size.to_i32
    input_per_group = input_channels // groups
    output_per_group = output_channels // groups
    left_pad = (kernel - 1) * dilation
    output = Array(Float32).new(input.rows * output_channels, 0.0_f32)
    input.rows.times do |t|
      output_channels.times do |out_channel|
        group = out_channel // output_per_group
        sum = bias[out_channel]
        input_per_group.times do |in_local|
          input_channel = group * input_per_group + in_local
          kernel.times do |tap|
            input_t = t * stride - left_pad + tap * dilation
            next if input_t < 0 || input_t >= input.rows
            wi = (out_channel * input_per_group + in_local) * kernel + tap
            sum += input.values[input_t * input_channels + input_channel] * weights[wi]
          end
        end
        output[t * output_channels + out_channel] = sum
      end
    end
    Matrix.new(input.rows, output_channels, output)
  end

  private def drama3_codec_cpu_transpose(
    input : Matrix, weights : Vector, bias : Vector, kernel : Int32, stride : Int32,
  ) : Matrix
    input_channels = input.cols
    output_channels = bias.size.to_i32
    output_time = input.rows * stride
    output = Array(Float32).new(output_time * output_channels, 0.0_f32)
    input.rows.times do |t|
      input_channels.times do |input_channel|
        kernel.times do |tap|
          output_t = t * stride + tap
          next if output_t >= output_time
          output_channels.times do |output_channel|
            wi = (input_channel * output_channels + output_channel) * kernel + tap
            oi = output_t * output_channels + output_channel
            output[oi] += input.values[t * input_channels + input_channel] * weights[wi]
          end
        end
      end
    end
    output_time.times do |t|
      output_channels.times { |channel| output[t * output_channels + channel] += bias[channel] }
    end
    Matrix.new(output_time, output_channels, output)
  end

  describe "DRAMA-3 DAC convolution Metal" do
    it "preserves grouped Conv1d order, dilation, stride indexing, and bias" do
      drama3_metal_with_device do
        input = Matrix.new(4, 4, [
          0.25_f32, -0.5_f32, 0.75_f32, 1.0_f32,
          -1.25_f32, 0.5_f32, -0.75_f32, 0.25_f32,
          1.5_f32, -0.25_f32, 0.125_f32, -1.0_f32,
          0.5_f32, 1.25_f32, -1.5_f32, 0.75_f32,
        ])
        weights = Array(Float32).new(4 * 2 * 3) do |index|
          (((index * 7) % 17) - 8).to_f32 / 8.0_f32
        end
        bias = [0.375_f32, -0.625_f32, 0.875_f32, -0.125_f32]
        expected = drama3_codec_cpu_conv(input, weights, bias, 3, 2, 2, 2)

        actual = CodecMetal.conv(input, weights, bias, 3, 2, 2, 2, "spec.grouped")

        actual.rows.should eq(input.rows)
        actual.cols.should eq(bias.size)
        actual.values.should eq(expected.values)

        # Reusing a prefix with changed values must refresh the cached buffers.
        changed_weights = weights.map { |value| -value }
        changed_bias = bias.map { |value| value + 0.25_f32 }
        changed_expected = drama3_codec_cpu_conv(input, changed_weights, changed_bias, 3, 2, 2, 2)
        changed_actual = CodecMetal.conv(input, changed_weights, changed_bias, 3, 2, 2, 2, "spec.grouped")
        changed_actual.values.should eq(changed_expected.values)
      end
    end

    it "matches transpose scatter order, adds bias last, and right-crops for unequal stride" do
      drama3_metal_with_device do
        input = Matrix.new(3, 2, [
          0.5_f32, -1.0_f32,
          1.25_f32, 0.25_f32,
          -0.75_f32, 1.5_f32,
        ])
        weights = Array(Float32).new(2 * 3 * 4) do |index|
          (((index * 5) % 19) - 9).to_f32 / 8.0_f32
        end
        bias = [0.375_f32, -0.25_f32, 0.625_f32]
        expected = drama3_codec_cpu_transpose(input, weights, bias, 4, 2)

        actual = CodecMetal.transpose(input, weights, bias, 4, 2, "spec.transpose")

        # A full transpose convolution would be 8 rows; the native decoder
        # keeps only input_time * stride (6 rows), cutting the right tail.
        actual.rows.should eq(6)
        actual.cols.should eq(3)
        actual.values.should eq(expected.values)
      end
    end
  end
{% end %}
