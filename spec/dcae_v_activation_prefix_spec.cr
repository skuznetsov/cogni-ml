require "spec"
require "../examples/support/dcae_v_activation_prefix"

describe DCAEVActivationPrefix do
  it "cuts checkpoint bindings immediately before the stage-2 target convolution" do
    config = ML::Video::DCAEVDecoderConfig.new(9, 16, 20)
    bindings = DCAEVActivationPrefix.prefix_bindings(config)

    bindings.size.should eq(90)
    bindings.last.target.should eq("decoder.stages.2.op_list.0.main.conv.conv.bias")
    bindings.any? { |binding| binding.target == DCAEVActivationPrefix::TARGET_WEIGHT }.should be_false
    bindings.any? { |binding| binding.target == DCAEVActivationPrefix::TARGET_BIAS }.should be_false
    bindings.any? { |binding| binding.target.starts_with?("decoder.stages.1.") }.should be_false
    bindings.any? { |binding| binding.target.starts_with?("decoder.project_out.") }.should be_false
    bindings.each do |binding|
      binding.dtype.should eq(ML::GGUF::SafeTensorDType::BF16)
    end
  end

  it "accounts only the bounded prefix residency and two causal target caches" do
    config = ML::Video::DCAEVDecoderConfig.new(9, 16, 20)
    bindings = DCAEVActivationPrefix.prefix_bindings(config)

    DCAEVActivationPrefix.binding_bytes(bindings).should eq(2_096_936_960_i64)
    DCAEVActivationPrefix.workspace_bytes(config).should eq(1_944_032_256_i64)
    (DCAEVActivationPrefix.binding_bytes(bindings) + DCAEVActivationPrefix.workspace_bytes(config))
      .should eq(4_040_969_216_i64)
  end
end
