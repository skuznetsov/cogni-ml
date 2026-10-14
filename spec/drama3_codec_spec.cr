require "spec"
require "../src/ml/audio/drama3/codec"

include ML::Audio::Drama3

describe "DRAMA-3 native codec" do
  it "validates codec dimensions and rejects malformed codebook layouts" do
    expect_raises(ArgumentError, /heads \* head_dim/) do
      CodecConfig.new(latent_channels: 8, transformer_heads: 2, transformer_head_dim: 3)
    end

    config = CodecConfig.new(
      latent_channels: 8,
      codebook_dim: 2,
      semantic_codebook_size: 5,
      residual_codebooks: 1,
      residual_codebook_size: 4,
      post_layers: 0,
      transformer_heads: 2,
      transformer_head_dim: 4,
      transformer_ff_dim: 12,
      transformer_window: 2,
      quantizer_upsample_rates: [2],
      decoder_channels: 8,
      decoder_rates: [2],
    )
    codec = Codec.new(WeightStore.new({} of String => Tensor), config)
    expect_raises(ArgumentError, /at least the semantic/) { codec.decode([] of Array(Int32)) }
    expect_raises(ArgumentError, /same frame count/) do
      codec.decode([[0, 1], [0]] of Array(Int32))
    end
  end

  if mode = ENV["DRAMA3_CODEC_REFERENCE"]?
    it "matches upstream MLX stages for #{mode} codec weights" do
      python = ENV["DRAMA3_CODEC_PYTHON"]? || "/private/tmp/cogni-drama3-reference/venv/bin/python"
      mlx_root = ENV["DRAMA3_MLX_AUDIO_ROOT"]? || "/private/tmp/cogni-drama3-reference/mlx-audio"
      helper = File.join(__DIR__, "support/drama3/codec_reference.py")
      codec_path = ENV["DRAMA3_CODEC_WEIGHTS"]?
      fixture_file = File.tempfile("drama3-codec-", ".json") do |file|
        arguments = [helper, mode, "--output", file.path]
        if mode == "real"
          path = codec_path || raise "DRAMA3_CODEC_WEIGHTS is required for real parity"
          arguments.concat(["--codec", path])
        end
        output = IO::Memory.new
        status = Process.run(python, arguments, env: {"PYTHONPATH" => mlx_root}, output: output, error: output)
        raise "upstream MLX codec reference failed: #{output}" unless status.success?
      end
      fixture = File.read(fixture_file.path)
      fixture_file.delete

      reference = JSON.parse(fixture)
      tensors = {} of String => Tensor
      reference["weights"].as_h.each do |name, value|
        shape = value["shape"].as_a.map(&.as_i)
        values = value["values"].as_a.map { |item| item.as_f.to_f32 }
        tensors["codec.#{name}"] = Tensor.new(shape, values)
      end

      config = if mode == "real"
                 CodecConfig.default
               else
                 values = reference["config"].as_h
                 CodecConfig.new(
                   latent_channels: values["latent_channels"].as_i,
                   codebook_dim: values["codebook_dim"].as_i,
                   semantic_codebook_size: values["semantic_codebook_size"].as_i,
                   residual_codebooks: values["residual_codebooks"].as_i,
                   residual_codebook_size: values["residual_codebook_size"].as_i,
                   post_layers: values["post_layers"].as_i,
                   transformer_heads: values["transformer_heads"].as_i,
                   transformer_head_dim: values["transformer_head_dim"].as_i,
                   transformer_ff_dim: values["transformer_ff_dim"].as_i,
                   transformer_window: values["transformer_window"].as_i,
                   quantizer_upsample_rates: values["quantizer_upsample_rates"].as_a.map(&.as_i),
                   decoder_channels: values["decoder_channels"].as_i,
                   decoder_rates: values["decoder_rates"].as_a.map(&.as_i),
                   norm_eps: values["norm_eps"].as_f.to_f32,
                   rope_base: values["rope_base"].as_f.to_f32,
                 )
               end
      codes = reference["codes"].as_a.map do |book|
        book.as_a.map(&.as_i.to_i32)
      end
      actual = {} of String => Vector
      store = if mode == "real"
                path = codec_path || raise "DRAMA3_CODEC_WEIGHTS is required for real parity"
                WeightStore.new(File.dirname(path))
              else
                WeightStore.new(tensors)
              end
      codec = Codec.new(store, config)
      codec.trace = ->(name : String, values : Vector) { actual[name] = values; nil }
      waveform = codec.decode(codes)
      actual["waveform"].should eq(waveform)

      expected = reference["traces"].as_h
      actual.keys.sort.should eq(expected.keys.sort)
      parity_failures = [] of String
      expected.each do |name, expected_values|
        got = actual[name]
        values = expected_values.as_a
        got.size.should eq(values.size)
        max_error = 0.0_f32
        squared_error_sum = 0.0_f64
        reference_squared_sum = 0.0_f64
        worst_index = 0
        worst_target = 0.0_f32
        worst_actual = 0.0_f32
        values.each_with_index do |value, index|
          target = value.as_f.to_f32
          actual_value = got[index]
          error = (actual_value - target).abs
          squared_error_sum += error.to_f64 * error.to_f64
          reference_squared_sum += target.to_f64 * target.to_f64
          if error > max_error
            max_error = error
            worst_index = index
            worst_target = target
            worst_actual = actual_value
          end
          if mode == "synthetic" || name == "waveform"
            tolerance = {5.0e-5_f32, target.abs * 1.0e-3_f32}.max
            if error > tolerance
              parity_failures << "#{name}[#{index}] error #{error} exceeds tolerance #{tolerance}"
            end
          end
        end
        rms_error = Math.sqrt(squared_error_sum / got.size).to_f32
        reference_rms = Math.sqrt(reference_squared_sum / got.size).to_f32
        normalized_rms = if reference_rms > 0.0_f32
                           rms_error / reference_rms
                         elsif rms_error == 0.0_f32
                           0.0_f32
                         else
                           Float32::INFINITY
                         end
        if mode == "real"
          puts "codec parity #{name}: max_abs=#{max_error}, rms=#{rms_error}, normalized_rms=#{normalized_rms}, worst_index=#{worst_index}, native=#{worst_actual}, upstream=#{worst_target}"
          # MLX fast-attention and native serial F32 reductions use different
          # accumulation trees. Bound internal stage absolute/RMS error and
          # normalize RMS against upstream stage energy; PCM stays per-sample.
          if name != "waveform" && (max_error > 1.0e-3_f32 || rms_error > 2.0e-4_f32 || normalized_rms > 2.0e-4_f32)
            parity_failures << "#{name} aggregate error max_abs=#{max_error} (limit 0.001), rms=#{rms_error} (limit 0.0002), normalized_rms=#{normalized_rms} (limit 0.0002)"
          end
        end
      end
      raise "codec parity failures:\n#{parity_failures.join("\n")}" unless parity_failures.empty?
      waveform.all?(&.finite?).should be_true
      waveform.any? { |sample| sample.abs > 1.0e-7_f32 }.should be_true
    end
  end
end
