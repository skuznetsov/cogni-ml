require "spec"
require "../src/ml/gguf/inkling2_resident_linear"

describe ML::GGUF::Inkling2ControllerBF16Storage do
  it "accepts only words exactly representable in BF16" do
    ML::GGUF::Inkling2ControllerBF16Storage.exact_word?(0x3f80_0000_u32).should be_true
    ML::GGUF::Inkling2ControllerBF16Storage.exact_word?(0x8000_0000_u32).should be_true
    ML::GGUF::Inkling2ControllerBF16Storage.exact_word?(0x3f80_0001_u32).should be_false
  end
end
