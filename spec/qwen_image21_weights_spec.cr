require "./spec_helper"
require "../src/ml/gguf/qwen_image21_weights"

describe ML::GGUF::QwenImage21Weights do
  it "maps the complete mixed Q4 model into executable projection weights" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the 5.96 GB model-backed loader check") unless path && File.file?(path)

    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      weights.layers.size.should eq(32)
      weights.img_in.in_dim.should eq(64)
      weights.img_in.out_dim.should eq(4096)
      weights.modulation.in_dim.should eq(4096)
      weights.modulation.out_dim.should eq(16384)
      weights.layers[0].to_q.type.should eq(ML::GGUF::TensorType::Q8_0)
      weights.layers[0].gate_up.type.should eq(ML::GGUF::TensorType::Q5_K)
      weights.layers[0].gate_up.out_dim.should eq(24576)
      weights.layers[0].mlp_out.in_dim.should eq(12288)
    ensure
      weights.close
    end
  end
end
