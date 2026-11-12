require "spec"
require "../src/ml/gguf/qwen35_native_runtime"

describe ML::GGUF::Qwen35NativeRuntime do
  it "requires a complete adaptive KV map for an explicitly enabled session" do
    expect_raises(ArgumentError, /require QWEN35_ADAPTIVE_RESIDENT_KV_MAP/) do
      ML::GGUF::Qwen35NativeRuntime.adaptive_qbit_restore_enabled?("session-a", "3", "p4", nil, "1")
    end
    ML::GGUF::Qwen35NativeRuntime.adaptive_qbit_restore_enabled?("session-a", nil, nil, "p4", "1").should be_true
    ML::GGUF::Qwen35NativeRuntime.adaptive_qbit_restore_enabled?("session-a", "3", "p4", nil, "0").should be_false
    ML::GGUF::Qwen35NativeRuntime.adaptive_qbit_restore_enabled?(nil, "3", "p4", nil, nil).should be_true
  end
end
