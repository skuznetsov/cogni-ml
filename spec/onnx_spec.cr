require "./spec_helper"
require "../src/ml/core/tensor"
require "../src/ml/autograd/variable"
require "../src/ml/onnx/exporter"

describe ML::ONNX::Exporter do
  it "emits ONNX bytes that pass onnx.checker" do
    input = ML::Autograd::Variable.new(
      ML::Tensor.zeros(1, 3, device: ML::Tensor::Device::CPU),
      requires_grad: true
    )
    w = ML::Autograd::Variable.new(
      ML::Tensor.ones(3, 2, device: ML::Tensor::Device::CPU),
      requires_grad: true
    )
    output = input.matmul(w).relu

    exp = ML::ONNX::Exporter.new
    exp.input(input, "X")
    exp.output(output, "Y")

    path = File.tempname("cogni_onnx", ".onnx")
    exp.write(path, "matmul_relu")
    begin
      check = `python3 -c "import onnx; m = onnx.load('#{path}'); onnx.checker.check_model(m); print('OK')" 2>&1`.strip
      check.should eq("OK")
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "onnxruntime output matches cogni-ml forward pass" do
    x = ML::Tensor.zeros(2, 3, device: ML::Tensor::Device::CPU)
    xd = x.cpu_data.not_nil!
    x_vals = [1.0_f32, -2.0_f32, 3.0_f32, 0.5_f32, -1.5_f32, 2.5_f32]
    x_vals.each_with_index { |v, i| xd[i] = v }

    w = ML::Tensor.zeros(3, 2, device: ML::Tensor::Device::CPU)
    wd = w.cpu_data.not_nil!
    [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32, 0.5_f32, 0.6_f32].each_with_index { |v, i| wd[i] = v }

    input = ML::Autograd::Variable.new(x, requires_grad: true)
    weight = ML::Autograd::Variable.new(w, requires_grad: true)
    output = input.matmul(weight).relu
    crystal_out = output.data.cpu_data.not_nil!.dup

    exp = ML::ONNX::Exporter.new
    exp.input(input, "X")
    exp.output(output, "Y")
    path = File.tempname("cogni_onnx", ".onnx")
    exp.write(path)

    begin
      x_py = x_vals.map(&.to_s).join(",")
      script = "import onnxruntime as ort, numpy as np; " \
               "s=ort.InferenceSession('#{path}'); " \
               "x=np.array([#{x_py}],dtype=np.float32).reshape(2,3); " \
               "out=s.run(['Y'],{'X':x})[0]; " \
               "print('R:'+','.join(f'{v:.6f}' for v in out.flatten().tolist()))"
      result = `python3 -c "#{script}" 2>/dev/null`.strip
      line = result.lines.find { |l| l.starts_with?("R:") }
      raise "no output: #{result.inspect}" unless line
      ort_out = line[2..].split(",").map(&.to_f32)

      ort_out.size.should eq(crystal_out.size)
      ort_out.each_with_index do |v, i|
        (v - crystal_out[i]).abs.should be < 1e-4_f32
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
