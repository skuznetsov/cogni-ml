require "./spec_helper"
require "../src/ml/onnx/builder"

describe ML::ONNX::Builder do
  it "builds a Conv network matching neurogolf_utils pattern" do
    # Replicate single_layer_conv2d_network with kernel_size=3, CHANNELS=10,
    # GRID=[1,10,30,30]. Identity-ish: output channel 0 mirrors input channel 0.
    channels = 10
    ksize = 3
    shape = [1, channels, 30, 30]
    w_shape = [channels, channels, ksize, ksize]

    weights = [] of Float32
    channels.times do |o|
      channels.times do |i|
        ksize.times do |r|
          ksize.times do |c|
            # Identity kernel: 1.0 at center only when o==i, else 0.
            w = (r == 1 && c == 1 && o == i) ? 1.0_f32 : 0.0_f32
            weights << w
          end
        end
      end
    end

    b = ML::ONNX::Builder.new
    b.input("input", shape)
    b.output("output", shape)
    b.initializer("W", w_shape, weights)
    b.conv("input", "W", "output",
      kernel_shape: [ksize, ksize], pads: [1, 1, 1, 1])

    path = File.tempname("cogni_conv", ".onnx")
    b.write(path, "conv_identity")

    begin
      check = `python3 -c "import onnx; m = onnx.load('#{path}'); onnx.checker.check_model(m); print('OK')" 2>&1`.strip
      check.should eq("OK")

      # Inference parity: random-ish input, verify channel-0 output == channel-0 input
      script = "import onnxruntime as ort, numpy as np; " \
               "s = ort.InferenceSession('#{path}'); " \
               "x = np.zeros((1,10,30,30), dtype=np.float32); " \
               "x[0,3,5,7] = 1.0; x[0,0,2,4] = 1.0; " \
               "y = s.run(['output'], {'input': x})[0]; " \
               "print('Y:'+str(int(y[0,3,5,7])) + ',' + str(int(y[0,0,2,4])))"
      result = `python3 -c "#{script}" 2>/dev/null`.strip
      line = result.lines.find { |l| l.starts_with?("Y:") }
      raise "no output: #{result.inspect}" unless line
      line.should eq("Y:1,1")
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
